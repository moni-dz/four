mod arithmetic;
mod huffman;
mod idct;
mod progressive;
mod reader;
#[cfg(test)]
mod test_data;

use std::array;
use std::fmt;
use std::num::NonZeroU32;

use arithmetic::ConditioningTables;
use exn::{ErrorExt, OptionExt};
use huffman::HuffmanTable;
use reader::{BitReader, Reader};

use super::DecodedImage;
#[cfg(test)]
use super::Image;

const BLOCK_SIDE: u32 = 8;
const COMPONENTS_MAX: usize = 3;
const DIMENSION_MAX: u32 = 16_384;
const HUFFMAN_TABLES_MAX: usize = 4;
const PIXELS_MAX: u64 = 64 * 1024 * 1024;
const PROGRESSIVE_COEFFICIENT_BYTES_MAX: u64 = 512 * 1024 * 1024;
const QUANTIZATION_TABLES_MAX: usize = 4;
const SCANS_MAX: u32 = 4_096;

const MARKER_SOI: u8 = 0xd8;
const MARKER_EOI: u8 = 0xd9;
const MARKER_SOF0: u8 = 0xc0;
const MARKER_SOF2: u8 = 0xc2;
const MARKER_SOF9: u8 = 0xc9;
const MARKER_SOF10: u8 = 0xca;
const MARKER_DHT: u8 = 0xc4;
const MARKER_DAC: u8 = 0xcc;
const MARKER_DQT: u8 = 0xdb;
const MARKER_DRI: u8 = 0xdd;
const MARKER_SOS: u8 = 0xda;

// JPEG stores coefficients diagonally, while the IDCT consumes ordinary row-major blocks.
const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// These relationships are part of the JPEG DCT grammar, so breaking one is a code defect.
const _: () = {
    assert!(BLOCK_SIDE == 8);
    assert!(COMPONENTS_MAX == 3);
    assert!(DIMENSION_MAX > 0);
    assert!(HUFFMAN_TABLES_MAX == 4);
    assert!(QUANTIZATION_TABLES_MAX == HUFFMAN_TABLES_MAX);
    assert!(PIXELS_MAX >= DIMENSION_MAX as u64);
    assert!(size_of::<NonZeroU32>() == size_of::<u32>());
    assert!(ZIGZAG_TO_NATURAL.len() == 64);
};

pub type Error = exn::Exn<JPEGError>;
pub type Result<T> = exn::Result<T, JPEGError>;

/// A decoder failure classified by the JPEG grammar section that rejected the input.
///
/// Keeping the classification in the type lets callers choose a recovery policy without parsing
/// display text. Static detail strings retain precise diagnostics without allocating on an error
/// path, while values that callers may need are stored in dedicated variants below.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JPEGError {
    ArithmeticOverflow(&'static str),
    Entropy(&'static str),
    ExpectedMarkerPrefix { context: &'static str, found: u8 },
    Frame(&'static str),
    LimitExceeded(JPEGLimit),
    Marker(&'static str),
    RestartMarkerMismatch { expected: u8, found: u8 },
    Scan(&'static str),
    Segment(&'static str),
    Table(JPEGTableKind, &'static str),
    UnexpectedEntropyMarker(u8),
    UnexpectedEnd(&'static str),
    UnexpectedMarker { context: &'static str, found: u8 },
    Unsupported(UnsupportedJPEG),
}

/// A bounded resource whose configured maximum was exceeded by the input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGLimit {
    Dimensions(u32),
    FrameDataUnits(u8),
    HuffmanSymbols(u16),
    Pixels(u64),
    ProgressiveCoefficientBytes(u64),
    Scans(u32),
}

/// The table class is retained so applications can identify the broken decoder dependency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGTableKind {
    ArithmeticConditioning,
    ACHuffman,
    DCHuffman,
    Huffman,
    Quantization,
}

/// A valid JPEG feature that this deliberately small decoder does not implement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedJPEG {
    AdobeColorTransform(u8),
    ComponentCount(u8),
    FrameType(u8),
    Marker(u8),
    SamplePrecision(u8),
}

impl fmt::Display for JPEGError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow(detail)
            | Self::Entropy(detail)
            | Self::Frame(detail)
            | Self::Marker(detail)
            | Self::Scan(detail)
            | Self::Segment(detail)
            | Self::Table(_, detail)
            | Self::UnexpectedEnd(detail) => formatter.write_str(detail),
            Self::ExpectedMarkerPrefix { context, found } => {
                write!(
                    formatter,
                    "expected a JPEG marker FF prefix {context}, found {found:02X}"
                )
            }
            Self::LimitExceeded(limit) => write_limit_error(formatter, *limit),
            Self::RestartMarkerMismatch { expected, found } => write!(
                formatter,
                "expected restart marker FF{expected:02X}, found FF{found:02X}"
            ),
            Self::UnexpectedEntropyMarker(marker) => {
                write!(
                    formatter,
                    "unexpected marker FF{marker:02X} inside entropy data"
                )
            }
            Self::UnexpectedMarker { context, found } => {
                write!(formatter, "expected {context}, found FF{found:02X}")
            }
            Self::Unsupported(feature) => write_unsupported_error(formatter, *feature),
        }
    }
}

impl std::error::Error for JPEGError {}

/// Raise a leaf JPEG error at its actual validation site so `exn` records a useful location.
#[track_caller]
fn error(error: JPEGError) -> Error {
    assert!(!error.to_string().is_empty());
    assert!(size_of::<JPEGError>() > 0);
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: JPEGLimit) -> fmt::Result {
    match limit {
        JPEGLimit::Dimensions(max) => {
            write!(formatter, "JPEG dimensions exceed the {max}-pixel limit")
        }
        JPEGLimit::FrameDataUnits(max) => {
            write!(formatter, "frame has more than {max} data units per MCU")
        }
        JPEGLimit::HuffmanSymbols(max) => {
            write!(formatter, "Huffman table has more than {max} symbols")
        }
        JPEGLimit::Pixels(max) => write!(
            formatter,
            "JPEG pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
        JPEGLimit::ProgressiveCoefficientBytes(max) => write!(
            formatter,
            "progressive coefficient storage exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        JPEGLimit::Scans(max) => write!(formatter, "JPEG contains more than {max} scans"),
    }
}

fn write_unsupported_error(
    formatter: &mut fmt::Formatter<'_>,
    feature: UnsupportedJPEG,
) -> fmt::Result {
    match feature {
        UnsupportedJPEG::AdobeColorTransform(value) => {
            write!(
                formatter,
                "Adobe JPEG color transform {value} is not supported"
            )
        }
        UnsupportedJPEG::ComponentCount(count) => write!(
            formatter,
            "JPEG component count {count} is unsupported; expected one or three"
        ),
        UnsupportedJPEG::FrameType(marker) => {
            write!(formatter, "JPEG frame type FF{marker:02X} is not supported")
        }
        UnsupportedJPEG::Marker(marker) => {
            write!(formatter, "unsupported JPEG marker FF{marker:02X}")
        }
        UnsupportedJPEG::SamplePrecision(precision) => write!(
            formatter,
            "JPEG sample precision {precision} is unsupported; expected 8"
        ),
    }
}

/// Decode an 8-bit Huffman- or arithmetic-coded sequential or progressive JPEG.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage> {
    assert!(bytes.len() <= isize::MAX as usize);
    Parser::<Headers>::new(bytes).decode()
}

// The parser phases make it impossible to decode entropy before a frame exists or to produce an
// image before at least one scan has completed. JPEG table presence stays dynamic because the file
// format permits tables to be redefined between scans; that is input state, not parser lifecycle.
struct Parser<'a, State> {
    reader: Reader<'a>,
    quantization_tables: [Option<[u16; 64]>; QUANTIZATION_TABLES_MAX],
    dc_tables: [Option<HuffmanTable>; HUFFMAN_TABLES_MAX],
    ac_tables: [Option<HuffmanTable>; HUFFMAN_TABLES_MAX],
    arithmetic_conditioning: ConditioningTables,
    restart_interval: u32,
    color_transform: ColorTransform,
    state: State,
}

struct Headers;

struct FrameData {
    frame: Frame,
    coefficient_bits: [[Option<u8>; 64]; COMPONENTS_MAX],
    component_scanned: [bool; COMPONENTS_MAX],
}

struct FrameReady {
    data: FrameData,
}

struct Scanned {
    data: FrameData,
    scan_count: NonZeroU32,
}

trait FramePhase {
    fn data(&self) -> &FrameData;
    fn data_mut(&mut self) -> &mut FrameData;
}

impl FramePhase for FrameReady {
    fn data(&self) -> &FrameData {
        assert!(!self.data.frame.components.is_empty());
        assert!(self.data.frame.components.len() <= COMPONENTS_MAX);
        &self.data
    }

    fn data_mut(&mut self) -> &mut FrameData {
        assert!(!self.data.frame.components.is_empty());
        assert!(self.data.frame.components.len() <= COMPONENTS_MAX);
        &mut self.data
    }
}

impl FramePhase for Scanned {
    fn data(&self) -> &FrameData {
        assert!(self.scan_count.get() <= SCANS_MAX);
        assert!(!self.data.frame.components.is_empty());
        &self.data
    }

    fn data_mut(&mut self) -> &mut FrameData {
        assert!(self.scan_count.get() <= SCANS_MAX);
        assert!(!self.data.frame.components.is_empty());
        &mut self.data
    }
}

impl<'a> Parser<'a, Headers> {
    fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        Self {
            reader: Reader::new(bytes),
            quantization_tables: array::from_fn(|_| None),
            dc_tables: array::from_fn(|_| None),
            ac_tables: array::from_fn(|_| None),
            arithmetic_conditioning: ConditioningTables::defaults(),
            restart_interval: 0,
            color_transform: ColorTransform::YCbCr,
            state: Headers,
        }
    }

    fn decode(mut self) -> Result<DecodedImage> {
        assert_eq!(self.restart_interval, 0);

        let first_marker = self.reader.marker()?;
        if first_marker != MARKER_SOI {
            return Err(error(JPEGError::UnexpectedMarker {
                context: "an SOI marker",
                found: first_marker,
            }));
        }
        let mut marker = self.reader.marker()?;
        loop {
            match marker {
                MARKER_DQT => self.parse_quantization_tables()?,
                MARKER_SOF0 => {
                    return self
                        .parse_frame(FrameMode::BaselineHuffman)?
                        .decode_until_first_scan();
                }
                MARKER_SOF2 => {
                    return self
                        .parse_frame(FrameMode::ProgressiveHuffman)?
                        .decode_until_first_scan();
                }
                MARKER_SOF9 => {
                    return self
                        .parse_frame(FrameMode::SequentialArithmetic)?
                        .decode_until_first_scan();
                }
                MARKER_SOF10 => {
                    return self
                        .parse_frame(FrameMode::ProgressiveArithmetic)?
                        .decode_until_first_scan();
                }
                MARKER_DHT => self.parse_huffman_tables()?,
                MARKER_DAC => self.parse_arithmetic_conditioning()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_SOS => {
                    return Err(error(JPEGError::Frame(
                        "SOS marker appeared before a frame",
                    )));
                }
                0xe0..=0xef => self.parse_application_segment(marker)?,
                0xfe => self.skip_segment()?,
                MARKER_EOI => {
                    return Err(error(JPEGError::Frame(
                        "JPEG ended before defining a frame",
                    )));
                }
                MARKER_SOI => return Err(error(JPEGError::Marker("duplicate SOI marker"))),
                0xc1 | 0xc3 | 0xc5..=0xc7 | 0xcb | 0xcd..=0xcf => {
                    return Err(unsupported_frame_error(marker));
                }
                _ => return Err(unsupported_marker_error(marker)),
            }
            marker = self.reader.marker()?;
        }
    }

    fn parse_frame(mut self, mode: FrameMode) -> Result<Parser<'a, FrameReady>> {
        assert_eq!(size_of::<Headers>(), 0);
        assert!(self.reader.remaining() <= isize::MAX as usize);

        let mut segment = self.reader.segment()?;
        let precision = segment.read_u8()?;
        if precision != 8 {
            return Err(error(JPEGError::Unsupported(
                UnsupportedJPEG::SamplePrecision(precision),
            )));
        }
        let height = u32::from(segment.read_u16()?);
        let width = u32::from(segment.read_u16()?);
        validate_dimensions(width, height)?;
        let component_count = usize::from(segment.read_u8()?);
        if component_count != 1 && component_count != 3 {
            return Err(error(JPEGError::Unsupported(
                UnsupportedJPEG::ComponentCount(component_count as u8),
            )));
        }

        let components = parse_frame_components(&mut segment, component_count)?;
        if segment.remaining() != 0 {
            return Err(error(JPEGError::Segment("SOF segment has trailing bytes")));
        }
        let frame = Frame::new(width, height, components, mode)?;
        Ok(self.map_state(|_headers| FrameReady {
            data: FrameData {
                frame,
                coefficient_bits: [[None; 64]; COMPONENTS_MAX],
                component_scanned: [false; COMPONENTS_MAX],
            },
        }))
    }
}

impl<'a, State> Parser<'a, State> {
    fn map_state<NextState>(
        self,
        transition: impl FnOnce(State) -> NextState,
    ) -> Parser<'a, NextState> {
        assert!(self.reader.remaining() <= isize::MAX as usize);
        assert!(self.restart_interval <= u32::from(u16::MAX));

        let state = transition(self.state);
        Parser {
            reader: self.reader,
            quantization_tables: self.quantization_tables,
            dc_tables: self.dc_tables,
            ac_tables: self.ac_tables,
            arithmetic_conditioning: self.arithmetic_conditioning,
            restart_interval: self.restart_interval,
            color_transform: self.color_transform,
            state,
        }
    }

    fn parse_quantization_tables(&mut self) -> Result<()> {
        assert_eq!(self.quantization_tables.len(), QUANTIZATION_TABLES_MAX);
        assert!(self.reader.remaining() <= isize::MAX as usize);

        let mut segment = self.reader.segment()?;
        while segment.remaining() > 0 {
            let descriptor = segment.read_u8()?;
            let precision = descriptor >> 4;
            let table_index = usize::from(descriptor & 0x0f);
            if table_index >= QUANTIZATION_TABLES_MAX {
                return Err(error(JPEGError::Table(
                    JPEGTableKind::Quantization,
                    "quantization table index is out of range",
                )));
            }
            if precision > 1 {
                return Err(error(JPEGError::Table(
                    JPEGTableKind::Quantization,
                    "quantization table precision is invalid",
                )));
            }

            let mut table = [0_u16; 64];
            for natural_index in ZIGZAG_TO_NATURAL {
                let value = if precision == 0 {
                    u16::from(segment.read_u8()?)
                } else {
                    segment.read_u16()?
                };
                if value == 0 {
                    return Err(error(JPEGError::Table(
                        JPEGTableKind::Quantization,
                        "quantization table contains a zero value",
                    )));
                }
                table[natural_index] = value;
            }
            self.quantization_tables[table_index] = Some(table);
        }
        Ok(())
    }

    fn parse_huffman_tables(&mut self) -> Result<()> {
        assert_eq!(self.dc_tables.len(), HUFFMAN_TABLES_MAX);
        assert_eq!(self.ac_tables.len(), HUFFMAN_TABLES_MAX);

        let mut segment = self.reader.segment()?;
        while segment.remaining() > 0 {
            let descriptor = segment.read_u8()?;
            let class = descriptor >> 4;
            let table_index = usize::from(descriptor & 0x0f);
            if class > 1 || table_index >= HUFFMAN_TABLES_MAX {
                return Err(error(JPEGError::Table(
                    JPEGTableKind::Huffman,
                    "Huffman table descriptor is invalid",
                )));
            }

            let mut counts = [0_u8; 16];
            for count in &mut counts {
                *count = segment.read_u8()?;
            }
            let symbol_count: usize = counts.iter().map(|count| usize::from(*count)).sum();
            if symbol_count > 256 {
                return Err(error(JPEGError::LimitExceeded(JPEGLimit::HuffmanSymbols(
                    256,
                ))));
            }
            let symbols = segment.read_slice(symbol_count)?.to_vec();
            validate_huffman_symbols(class, &symbols)?;
            let table = HuffmanTable::new(counts, symbols)?;
            if class == 0 {
                self.dc_tables[table_index] = Some(table);
            } else {
                self.ac_tables[table_index] = Some(table);
            }
        }
        Ok(())
    }

    fn parse_arithmetic_conditioning(&mut self) -> Result<()> {
        assert_eq!(self.arithmetic_conditioning.dc.len(), HUFFMAN_TABLES_MAX);
        assert_eq!(self.arithmetic_conditioning.ac.len(), HUFFMAN_TABLES_MAX);

        let mut segment = self.reader.segment()?;
        if segment.remaining() == 0 || segment.remaining() % 2 != 0 {
            return Err(error(JPEGError::Table(
                JPEGTableKind::ArithmeticConditioning,
                "DAC segment must contain one or more descriptor-value pairs",
            )));
        }
        while segment.remaining() > 0 {
            let descriptor = segment.read_u8()?;
            let class = descriptor >> 4;
            let table = usize::from(descriptor & 0x0f);
            let value = segment.read_u8()?;
            if class > 1 {
                return Err(error(JPEGError::Table(
                    JPEGTableKind::ArithmeticConditioning,
                    "arithmetic conditioning table class is invalid",
                )));
            }
            if table >= HUFFMAN_TABLES_MAX {
                return Err(error(JPEGError::Table(
                    JPEGTableKind::ArithmeticConditioning,
                    "arithmetic conditioning table identifier is out of range",
                )));
            }
            if class == 0 {
                let lower = value & 0x0f;
                let upper = value >> 4;
                if lower > upper {
                    return Err(error(JPEGError::Table(
                        JPEGTableKind::ArithmeticConditioning,
                        "DC arithmetic conditioning requires L <= U",
                    )));
                }
                self.arithmetic_conditioning.dc[table] =
                    arithmetic::DCConditioning { lower, upper };
            } else {
                if !(1..=63).contains(&value) {
                    return Err(error(JPEGError::Table(
                        JPEGTableKind::ArithmeticConditioning,
                        "AC arithmetic conditioning must be in 1..=63",
                    )));
                }
                self.arithmetic_conditioning.ac[table] = value;
            }
        }
        Ok(())
    }

    fn parse_restart_interval(&mut self) -> Result<()> {
        assert!(self.restart_interval <= u32::from(u16::MAX));
        assert!(self.reader.remaining() <= isize::MAX as usize);

        let mut segment = self.reader.segment()?;
        if segment.remaining() != 2 {
            return Err(error(JPEGError::Segment(
                "DRI segment must contain exactly two bytes",
            )));
        }
        self.restart_interval = u32::from(segment.read_u16()?);
        Ok(())
    }

    fn parse_application_segment(&mut self, marker: u8) -> Result<()> {
        assert!((0xe0..=0xef).contains(&marker));
        assert!(self.reader.remaining() <= isize::MAX as usize);

        let mut segment = self.reader.segment()?;
        let bytes = segment.read_slice(segment.remaining())?;
        if marker == 0xe0 && bytes.starts_with(b"JFIF\0") {
            self.color_transform = ColorTransform::YCbCr;
        }
        if marker == 0xee && bytes.starts_with(b"Adobe") && bytes.len() >= 12 {
            self.color_transform = match bytes[11] {
                0 => ColorTransform::Rgb,
                1 => ColorTransform::YCbCr,
                value => {
                    return Err(error(JPEGError::Unsupported(
                        UnsupportedJPEG::AdobeColorTransform(value),
                    )));
                }
            };
        }
        Ok(())
    }

    fn skip_segment(&mut self) -> Result<()> {
        let mut segment = self.reader.segment()?;
        segment.read_slice(segment.remaining())?;
        Ok(())
    }
}

impl<'a> Parser<'a, FrameReady> {
    fn decode_until_first_scan(mut self) -> Result<DecodedImage> {
        assert_eq!(
            self.state.data.coefficient_bits,
            [[None; 64]; COMPONENTS_MAX]
        );
        assert_eq!(self.state.data.component_scanned, [false; COMPONENTS_MAX]);
        assert!(!self.state.data.frame.components.is_empty());

        let mut marker = self.reader.marker()?;
        loop {
            match marker {
                MARKER_DQT => self.parse_quantization_tables()?,
                MARKER_DHT => self.parse_huffman_tables()?,
                MARKER_DAC => self.parse_arithmetic_conditioning()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_SOS => {
                    let next_marker = self.parse_scan_and_decode(0)?;
                    let parser = self.map_state(|ready| Scanned {
                        data: ready.data,
                        scan_count: NonZeroU32::MIN,
                    });
                    return parser.decode_after_scan(next_marker);
                }
                0xe0..=0xef => self.parse_application_segment(marker)?,
                0xfe => self.skip_segment()?,
                MARKER_EOI => {
                    return Err(error(JPEGError::Scan("JPEG ended before its first scan")));
                }
                MARKER_SOI => return Err(error(JPEGError::Marker("duplicate SOI marker"))),
                MARKER_SOF0 | MARKER_SOF2 | MARKER_SOF9 | MARKER_SOF10 => {
                    return Err(error(JPEGError::Frame(
                        "multiple JPEG frames are not supported",
                    )));
                }
                0xc1 | 0xc3 | 0xc5..=0xc7 | 0xcb | 0xcd..=0xcf => {
                    return Err(unsupported_frame_error(marker));
                }
                _ => return Err(unsupported_marker_error(marker)),
            }
            marker = self.reader.marker()?;
        }
    }
}

impl<'a> Parser<'a, Scanned> {
    fn decode_after_scan(mut self, mut marker: u8) -> Result<DecodedImage> {
        assert!(self.state.scan_count.get() <= SCANS_MAX);
        assert!(!self.state.data.frame.components.is_empty());

        loop {
            match marker {
                MARKER_DQT => self.parse_quantization_tables()?,
                MARKER_DHT => self.parse_huffman_tables()?,
                MARKER_DAC => self.parse_arithmetic_conditioning()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_SOS => {
                    marker = self.parse_scan_and_decode(self.state.scan_count.get())?;
                    self.state.scan_count = self
                        .state
                        .scan_count
                        .checked_add(1)
                        .expect("scan count was bounded before incrementing");
                    continue;
                }
                0xe0..=0xef => self.parse_application_segment(marker)?,
                0xfe => self.skip_segment()?,
                MARKER_EOI => return self.finish(),
                MARKER_SOI => return Err(error(JPEGError::Marker("duplicate SOI marker"))),
                MARKER_SOF0 | MARKER_SOF2 | MARKER_SOF9 | MARKER_SOF10 => {
                    return Err(error(JPEGError::Frame(
                        "multiple JPEG frames are not supported",
                    )));
                }
                0xc1 | 0xc3 | 0xc5..=0xc7 | 0xcb | 0xcd..=0xcf => {
                    return Err(unsupported_frame_error(marker));
                }
                _ => return Err(unsupported_marker_error(marker)),
            }
            marker = self.reader.marker()?;
        }
    }

    fn finish(self) -> Result<DecodedImage> {
        assert!(self.state.scan_count.get() <= SCANS_MAX);
        assert!(!self.state.data.frame.components.is_empty());

        let mut frame = self.state.data.frame;
        if frame.mode.process() == CodingProcess::Progressive {
            for component_index in 0..frame.components.len() {
                if self.state.data.coefficient_bits[component_index][0].is_none() {
                    return Err(error(JPEGError::Scan(
                        "progressive JPEG is missing a DC scan",
                    )));
                }
            }
            frame.materialize_progressive(&self.quantization_tables)?;
        } else {
            for component_index in 0..frame.components.len() {
                if !self.state.data.component_scanned[component_index] {
                    return Err(error(JPEGError::Scan(
                        "sequential JPEG is missing a component scan",
                    )));
                }
            }
        }
        frame.into_image(self.color_transform)
    }
}

impl<'a, State: FramePhase> Parser<'a, State> {
    fn parse_scan_and_decode(&mut self, scan_count: u32) -> Result<u8> {
        if scan_count >= SCANS_MAX {
            return Err(error(JPEGError::LimitExceeded(JPEGLimit::Scans(SCANS_MAX))));
        }
        let scan = {
            let mut segment = self.reader.segment()?;
            parse_scan_header(&mut segment, &self.state.data().frame)?
        };
        let mode = self.state.data().frame.mode;
        let result = match mode {
            FrameMode::BaselineHuffman => self.decode_huffman_sequential_scan(&scan, scan_count)?,
            FrameMode::SequentialArithmetic => self.decode_arithmetic_sequential_scan(&scan)?,
            FrameMode::ProgressiveHuffman => self.decode_huffman_progressive_scan(&scan)?,
            FrameMode::ProgressiveArithmetic => self.decode_arithmetic_progressive_scan(&scan)?,
        };
        let (bytes_consumed, marker) = result;
        let entropy_length = self.reader.remaining();
        self.reader.advance(bytes_consumed)?;
        assert!(bytes_consumed <= entropy_length);
        if mode.process() == CodingProcess::Progressive {
            self.commit_progression(&scan);
        } else {
            for component in &scan.components {
                self.state.data_mut().component_scanned[component.frame_index] = true;
            }
        }
        Ok(marker)
    }

    fn decode_huffman_sequential_scan(
        &mut self,
        scan: &ScanHeader,
        scan_count: u32,
    ) -> Result<(usize, u8)> {
        assert_eq!(
            scan.components.len(),
            self.state.data().frame.components.len()
        );
        assert_eq!(
            self.state.data().frame.mode.entropy(),
            EntropyCoding::Huffman
        );
        assert!(scan.spectral_start == 0 && scan.spectral_end == 63);

        if scan_count != 0 {
            return Err(error(JPEGError::Scan(
                "baseline JPEG contains more than one scan",
            )));
        }
        let plans = self.build_scan_plans(&scan.components)?;
        let entropy = self.reader.remaining_slice();
        let frame = &mut self.state.data_mut().frame;
        decode_entropy(entropy, frame, &plans, self.restart_interval)
    }

    fn decode_arithmetic_sequential_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        assert!(!scan.components.is_empty());
        assert!(scan.components.len() <= COMPONENTS_MAX);
        assert_eq!(
            self.state.data().frame.mode.entropy(),
            EntropyCoding::Arithmetic
        );

        for component in &scan.components {
            if self.state.data().component_scanned[component.frame_index] {
                return Err(error(JPEGError::Scan(
                    "sequential arithmetic component was decoded twice",
                )));
            }
        }
        let plans = self.build_arithmetic_sequential_plans(&scan.components)?;
        let entropy = self.reader.remaining_slice();
        let conditioning = self.arithmetic_conditioning;
        let frame = &mut self.state.data_mut().frame;
        arithmetic::decode_sequential(entropy, frame, &plans, &conditioning, self.restart_interval)
    }

    fn decode_huffman_progressive_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        assert!(scan.spectral_start <= scan.spectral_end);
        assert!(scan.spectral_end < 64);
        assert_eq!(
            self.state.data().frame.mode.entropy(),
            EntropyCoding::Huffman
        );

        self.validate_progression(scan)?;
        let plans = self.build_progressive_plans(scan)?;
        let entropy = self.reader.remaining_slice();
        let frame = &mut self.state.data_mut().frame;
        progressive::decode_scan(entropy, frame, &plans, scan, self.restart_interval)
    }

    fn decode_arithmetic_progressive_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        assert!(scan.spectral_start <= scan.spectral_end);
        assert!(scan.spectral_end < 64);
        assert_eq!(
            self.state.data().frame.mode.entropy(),
            EntropyCoding::Arithmetic
        );

        self.validate_progression(scan)?;
        let plans = self.build_arithmetic_progressive_plans(scan);
        let entropy = self.reader.remaining_slice();
        let conditioning = self.arithmetic_conditioning;
        let frame = &mut self.state.data_mut().frame;
        arithmetic::decode_progressive(
            entropy,
            frame,
            &plans,
            scan,
            &conditioning,
            self.restart_interval,
        )
    }

    fn build_scan_plans(&self, scan: &[ScanComponent]) -> Result<Vec<ScanPlan>> {
        let frame = &self.state.data().frame;
        assert!(scan.len() <= COMPONENTS_MAX);
        assert_eq!(scan.len(), frame.components.len());

        let mut plans = Vec::with_capacity(scan.len());
        for component in scan {
            let frame_component = &frame.components[component.frame_index];
            let quantization = self.quantization_tables[frame_component.quantization_table]
                .ok_or_raise(|| {
                    JPEGError::Table(
                        JPEGTableKind::Quantization,
                        "scan references a missing quantization table",
                    )
                })?;
            let dc = self.dc_tables[component.dc_table].clone().ok_or_raise(|| {
                JPEGError::Table(
                    JPEGTableKind::DCHuffman,
                    "scan references a missing DC Huffman table",
                )
            })?;
            let ac = self.ac_tables[component.ac_table].clone().ok_or_raise(|| {
                JPEGError::Table(
                    JPEGTableKind::ACHuffman,
                    "scan references a missing AC Huffman table",
                )
            })?;
            plans.push(ScanPlan {
                frame_index: component.frame_index,
                horizontal_sampling: frame_component.horizontal_sampling,
                vertical_sampling: frame_component.vertical_sampling,
                quantization,
                dc,
                ac,
            });
        }
        Ok(plans)
    }

    fn build_arithmetic_sequential_plans(
        &self,
        scan: &[ScanComponent],
    ) -> Result<Vec<arithmetic::SequentialPlan>> {
        let frame = &self.state.data().frame;
        assert!(!scan.is_empty());
        assert!(scan.len() <= COMPONENTS_MAX);

        let mut plans = Vec::with_capacity(scan.len());
        for component in scan {
            let frame_component = &frame.components[component.frame_index];
            let quantization = self.quantization_tables[frame_component.quantization_table]
                .ok_or_raise(|| {
                    JPEGError::Table(
                        JPEGTableKind::Quantization,
                        "arithmetic scan references a missing quantization table",
                    )
                })?;
            plans.push(arithmetic::SequentialPlan {
                frame_index: component.frame_index,
                horizontal_sampling: frame_component.horizontal_sampling,
                vertical_sampling: frame_component.vertical_sampling,
                quantization,
                dc_table: component.dc_table,
                ac_table: component.ac_table,
            });
        }
        Ok(plans)
    }

    fn build_arithmetic_progressive_plans(
        &self,
        scan: &ScanHeader,
    ) -> Vec<arithmetic::ProgressivePlan> {
        let frame = &self.state.data().frame;
        assert!(!scan.components.is_empty());
        assert!(scan.components.len() <= COMPONENTS_MAX);

        let mut plans = Vec::with_capacity(scan.components.len());
        for component in &scan.components {
            let frame_component = &frame.components[component.frame_index];
            plans.push(arithmetic::ProgressivePlan {
                frame_index: component.frame_index,
                horizontal_sampling: frame_component.horizontal_sampling,
                vertical_sampling: frame_component.vertical_sampling,
                dc_table: component.dc_table,
                ac_table: component.ac_table,
            });
        }
        plans
    }

    fn build_progressive_plans(&self, scan: &ScanHeader) -> Result<Vec<ProgressivePlan>> {
        let frame = &self.state.data().frame;
        assert!(scan.components.len() <= COMPONENTS_MAX);
        assert_eq!(frame.mode.process(), CodingProcess::Progressive);

        let mut plans = Vec::with_capacity(scan.components.len());
        for component in &scan.components {
            let frame_component = &frame.components[component.frame_index];
            let entropy = if scan.spectral_start == 0 {
                if scan.successive_high == 0 {
                    let table = self.dc_tables[component.dc_table].clone().ok_or_raise(|| {
                        JPEGError::Table(
                            JPEGTableKind::DCHuffman,
                            "progressive DC scan references a missing Huffman table",
                        )
                    })?;
                    ProgressiveEntropy::DCFirst(table)
                } else {
                    ProgressiveEntropy::DCRefinement
                }
            } else {
                let table = self.ac_tables[component.ac_table].clone().ok_or_raise(|| {
                    JPEGError::Table(
                        JPEGTableKind::ACHuffman,
                        "progressive AC scan references a missing Huffman table",
                    )
                })?;
                if scan.successive_high == 0 {
                    ProgressiveEntropy::ACFirst(table)
                } else {
                    ProgressiveEntropy::ACRefinement(table)
                }
            };
            plans.push(ProgressivePlan {
                frame_index: component.frame_index,
                horizontal_sampling: frame_component.horizontal_sampling,
                vertical_sampling: frame_component.vertical_sampling,
                entropy,
            });
        }
        Ok(plans)
    }

    fn validate_progression(&self, scan: &ScanHeader) -> Result<()> {
        assert!(scan.spectral_start <= scan.spectral_end);
        assert!(scan.components.len() <= COMPONENTS_MAX);

        for component in &scan.components {
            if scan.spectral_start > 0
                && self.state.data().coefficient_bits[component.frame_index][0].is_none()
            {
                return Err(error(JPEGError::Scan(
                    "progressive AC scan appeared before its DC scan",
                )));
            }
            for coefficient in scan.spectral_start..=scan.spectral_end {
                let previous = self.state.data().coefficient_bits[component.frame_index]
                    [usize::from(coefficient)];
                if scan.successive_high == 0 && previous.is_some() {
                    return Err(error(JPEGError::Scan(
                        "progressive coefficient band was initialized twice",
                    )));
                }
                if scan.successive_high > 0 && previous != Some(scan.successive_high) {
                    return Err(error(JPEGError::Scan(
                        "progressive refinement has an inconsistent bit order",
                    )));
                }
            }
        }
        Ok(())
    }

    fn commit_progression(&mut self, scan: &ScanHeader) {
        assert!(scan.spectral_start <= scan.spectral_end);
        assert!(scan.components.len() <= COMPONENTS_MAX);

        for component in &scan.components {
            for coefficient in scan.spectral_start..=scan.spectral_end {
                self.state.data_mut().coefficient_bits[component.frame_index]
                    [usize::from(coefficient)] = Some(scan.successive_low);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ColorTransform {
    YCbCr,
    Rgb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodingProcess {
    Sequential,
    Progressive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EntropyCoding {
    Huffman,
    Arithmetic,
}

// The marker grammar admits exactly these four DCT modes. A single sum type prevents invalid
// combinations such as baseline arithmetic coding from leaking into scan dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameMode {
    BaselineHuffman,
    SequentialArithmetic,
    ProgressiveHuffman,
    ProgressiveArithmetic,
}

impl FrameMode {
    const fn process(self) -> CodingProcess {
        match self {
            Self::BaselineHuffman | Self::SequentialArithmetic => CodingProcess::Sequential,
            Self::ProgressiveHuffman | Self::ProgressiveArithmetic => CodingProcess::Progressive,
        }
    }

    const fn entropy(self) -> EntropyCoding {
        match self {
            Self::BaselineHuffman | Self::ProgressiveHuffman => EntropyCoding::Huffman,
            Self::SequentialArithmetic | Self::ProgressiveArithmetic => EntropyCoding::Arithmetic,
        }
    }
}

struct FrameComponent {
    identifier: u8,
    horizontal_sampling: u8,
    vertical_sampling: u8,
    quantization_table: usize,
    plane_width: u32,
    plane: Vec<u8>,
    block_columns: u32,
    block_rows: u32,
    data_block_columns: u32,
    data_block_rows: u32,
    coefficients: Vec<[i32; 64]>,
}

struct Frame {
    width: u32,
    height: u32,
    mcu_columns: u32,
    mcu_rows: u32,
    max_horizontal_sampling: u8,
    max_vertical_sampling: u8,
    components: Vec<FrameComponent>,
    mode: FrameMode,
}

impl Frame {
    fn new(
        width: u32,
        height: u32,
        mut components: Vec<FrameComponent>,
        mode: FrameMode,
    ) -> Result<Self> {
        assert!(!components.is_empty());
        assert!(components.len() <= COMPONENTS_MAX);

        let max_horizontal_sampling = components
            .iter()
            .map(|component| component.horizontal_sampling)
            .max()
            .expect("frame has components");
        let max_vertical_sampling = components
            .iter()
            .map(|component| component.vertical_sampling)
            .max()
            .expect("frame has components");
        let mcu_width = u32::from(max_horizontal_sampling) * BLOCK_SIDE;
        let mcu_height = u32::from(max_vertical_sampling) * BLOCK_SIDE;
        let mcu_columns = divide_ceil(width, mcu_width);
        let mcu_rows = divide_ceil(height, mcu_height);

        let storage = ComponentStorageLayout {
            width,
            height,
            mcu_columns,
            mcu_rows,
            max_horizontal_sampling,
            max_vertical_sampling,
            process: mode.process(),
        };
        validate_progressive_storage(&components, &storage)?;
        for component in &mut components {
            allocate_component_storage(component, &storage)?;
        }
        Ok(Self {
            width,
            height,
            mcu_columns,
            mcu_rows,
            max_horizontal_sampling,
            max_vertical_sampling,
            components,
            mode,
        })
    }

    fn materialize_progressive(
        &mut self,
        quantization_tables: &[Option<[u16; 64]>; QUANTIZATION_TABLES_MAX],
    ) -> Result<()> {
        assert_eq!(self.mode.process(), CodingProcess::Progressive);
        assert!(!self.components.is_empty());

        for component in &mut self.components {
            let quantization =
                quantization_tables[component.quantization_table].ok_or_raise(|| {
                    JPEGError::Table(
                        JPEGTableKind::Quantization,
                        "frame references a missing quantization table",
                    )
                })?;
            let block_count = component
                .block_columns
                .checked_mul(component.block_rows)
                .ok_or_raise(|| {
                    JPEGError::ArithmeticOverflow("component block count overflowed")
                })?;
            if component.coefficients.len() != block_count as usize {
                return Err(error(JPEGError::Frame(
                    "progressive coefficient plane has an invalid size",
                )));
            }
            for block_index in 0..block_count {
                let quantized = component.coefficients[block_index as usize];
                let coefficients = dequantize_block(&quantized, &quantization)?;
                let samples = idct::inverse(&coefficients);
                let block_x = block_index % component.block_columns;
                let block_y = block_index / component.block_columns;
                write_block(component, block_x, block_y, &samples);
            }
            component.coefficients = Vec::new();
        }
        Ok(())
    }

    fn into_image(self, transform: ColorTransform) -> Result<DecodedImage> {
        assert!(self.components.len() == 1 || self.components.len() == 3);
        assert!(u64::from(self.width) * u64::from(self.height) <= PIXELS_MAX);

        let byte_count = u64::from(self.width) * u64::from(self.height) * 4;
        let mut rgba = Vec::with_capacity(byte_count as usize);
        for y in 0..self.height {
            for x in 0..self.width {
                let first = self.sample(0, x, y);
                let pixel = if self.components.len() == 1 {
                    [first, first, first, 255]
                } else {
                    let second = self.sample(1, x, y);
                    let third = self.sample(2, x, y);
                    convert_color(first, second, third, transform)
                };
                rgba.extend_from_slice(&pixel);
            }
        }
        assert_eq!(rgba.len() as u64, byte_count);
        Ok(DecodedImage::new(self.width, self.height, rgba))
    }

    fn sample(&self, component_index: usize, x: u32, y: u32) -> u8 {
        assert!(component_index < self.components.len());
        assert!(x < self.width);
        assert!(y < self.height);

        let component = &self.components[component_index];
        let sample_x =
            x * u32::from(component.horizontal_sampling) / u32::from(self.max_horizontal_sampling);
        let sample_y =
            y * u32::from(component.vertical_sampling) / u32::from(self.max_vertical_sampling);
        let index = u64::from(sample_y) * u64::from(component.plane_width) + u64::from(sample_x);
        component.plane[index as usize]
    }
}

fn dequantize_block(values: &[i32; 64], quantization: &[u16; 64]) -> Result<[i32; 64]> {
    assert!(quantization.iter().all(|value| *value > 0));
    assert!(values.iter().all(|value| value.checked_abs().is_some()));

    let mut coefficients = [0_i32; 64];
    for index in 0..64 {
        coefficients[index] = values[index]
            .checked_mul(i32::from(quantization[index]))
            .ok_or_raise(|| {
                JPEGError::ArithmeticOverflow("dequantized progressive coefficient overflowed")
            })?;
    }
    Ok(coefficients)
}

struct ScanComponent {
    frame_index: usize,
    dc_table: usize,
    ac_table: usize,
}

struct ScanHeader {
    components: Vec<ScanComponent>,
    spectral_start: u8,
    spectral_end: u8,
    successive_high: u8,
    successive_low: u8,
}

struct ScanPlan {
    frame_index: usize,
    horizontal_sampling: u8,
    vertical_sampling: u8,
    quantization: [u16; 64],
    dc: HuffmanTable,
    ac: HuffmanTable,
}

struct ProgressivePlan {
    frame_index: usize,
    horizontal_sampling: u8,
    vertical_sampling: u8,
    entropy: ProgressiveEntropy,
}

// Scan mode is selected by untrusted input, so an exhaustive sum type is the boundary equivalent
// of type-state. Every mode owns exactly the Huffman table it can consume.
enum ProgressiveEntropy {
    DCFirst(HuffmanTable),
    DCRefinement,
    ACFirst(HuffmanTable),
    ACRefinement(HuffmanTable),
}

fn parse_frame_components(
    segment: &mut Reader<'_>,
    component_count: usize,
) -> Result<Vec<FrameComponent>> {
    assert!(component_count == 1 || component_count == 3);
    assert!(component_count <= COMPONENTS_MAX);

    let mut components = Vec::with_capacity(component_count);
    let mut blocks_per_mcu = 0_u32;
    for _ in 0..component_count {
        let identifier = segment.read_u8()?;
        if components
            .iter()
            .any(|component: &FrameComponent| component.identifier == identifier)
        {
            return Err(error(JPEGError::Frame(
                "frame contains duplicate component identifiers",
            )));
        }
        let sampling = segment.read_u8()?;
        let horizontal_sampling = sampling >> 4;
        let vertical_sampling = sampling & 0x0f;
        if !(1..=4).contains(&horizontal_sampling) || !(1..=4).contains(&vertical_sampling) {
            return Err(error(JPEGError::Frame(
                "component sampling factor is outside 1 through 4",
            )));
        }
        blocks_per_mcu += u32::from(horizontal_sampling) * u32::from(vertical_sampling);
        let quantization_table = usize::from(segment.read_u8()?);
        if quantization_table >= QUANTIZATION_TABLES_MAX {
            return Err(error(JPEGError::Table(
                JPEGTableKind::Quantization,
                "component quantization table index is out of range",
            )));
        }
        components.push(FrameComponent {
            identifier,
            horizontal_sampling,
            vertical_sampling,
            quantization_table,
            plane_width: 0,
            plane: Vec::new(),
            block_columns: 0,
            block_rows: 0,
            data_block_columns: 0,
            data_block_rows: 0,
            coefficients: Vec::new(),
        });
    }
    if blocks_per_mcu > 10 {
        return Err(error(JPEGError::LimitExceeded(JPEGLimit::FrameDataUnits(
            10,
        ))));
    }
    Ok(components)
}

fn parse_scan_header(segment: &mut Reader<'_>, frame: &Frame) -> Result<ScanHeader> {
    assert!(!frame.components.is_empty());
    assert!(frame.components.len() <= COMPONENTS_MAX);

    let component_count = usize::from(segment.read_u8()?);
    if component_count == 0 || component_count > frame.components.len() {
        return Err(error(JPEGError::Scan(
            "scan component count is out of range",
        )));
    }
    let mut components = Vec::with_capacity(component_count);
    for _ in 0..component_count {
        let identifier = segment.read_u8()?;
        let frame_index = frame
            .components
            .iter()
            .position(|component| component.identifier == identifier)
            .ok_or_raise(|| JPEGError::Scan("scan references an unknown frame component"))?;
        if components
            .iter()
            .any(|component: &ScanComponent| component.frame_index == frame_index)
        {
            return Err(error(JPEGError::Scan(
                "scan contains a duplicate component",
            )));
        }
        let selectors = segment.read_u8()?;
        let dc_table = usize::from(selectors >> 4);
        let ac_table = usize::from(selectors & 0x0f);
        if dc_table >= HUFFMAN_TABLES_MAX || ac_table >= HUFFMAN_TABLES_MAX {
            return Err(error(JPEGError::Scan(
                "scan entropy table selector is out of range",
            )));
        }
        components.push(ScanComponent {
            frame_index,
            dc_table,
            ac_table,
        });
    }
    let spectral_start = segment.read_u8()?;
    let spectral_end = segment.read_u8()?;
    let approximation = segment.read_u8()?;
    if segment.remaining() != 0 {
        return Err(error(JPEGError::Segment("SOS segment has trailing bytes")));
    }
    let header = ScanHeader {
        components,
        spectral_start,
        spectral_end,
        successive_high: approximation >> 4,
        successive_low: approximation & 0x0f,
    };
    validate_scan_header(&header, frame.mode, frame.components.len())?;
    Ok(header)
}

fn validate_scan_header(
    scan: &ScanHeader,
    mode: FrameMode,
    frame_component_count: usize,
) -> Result<()> {
    assert!(!scan.components.is_empty());
    assert!(scan.components.len() <= COMPONENTS_MAX);

    if mode.process() == CodingProcess::Sequential {
        let is_full_scan = scan.components.len() == frame_component_count;
        let is_sequential = scan.spectral_start == 0
            && scan.spectral_end == 63
            && scan.successive_high == 0
            && scan.successive_low == 0;
        if !is_sequential {
            return Err(error(JPEGError::Scan(
                "sequential JPEG scan parameters are invalid",
            )));
        }
        if mode == FrameMode::BaselineHuffman && !is_full_scan {
            return Err(error(JPEGError::Scan(
                "baseline JPEG requires one full interleaved scan",
            )));
        }
        return Ok(());
    }
    if scan.spectral_start > scan.spectral_end || scan.spectral_end >= 64 {
        return Err(error(JPEGError::Scan(
            "progressive spectral selection is out of range",
        )));
    }
    if scan.spectral_start == 0 && scan.spectral_end != 0 {
        return Err(error(JPEGError::Scan(
            "a progressive DC scan must end at coefficient zero",
        )));
    }
    if scan.spectral_start > 0 && scan.components.len() != 1 {
        return Err(error(JPEGError::Scan(
            "a progressive AC scan must contain one component",
        )));
    }
    if scan.successive_high > 0 && scan.successive_low + 1 != scan.successive_high {
        return Err(error(JPEGError::Scan(
            "progressive refinement must advance exactly one bit",
        )));
    }
    if scan.successive_low > 13 {
        return Err(error(JPEGError::Scan(
            "progressive approximation bit exceeds 13",
        )));
    }
    Ok(())
}

fn decode_entropy(
    entropy: &[u8],
    frame: &mut Frame,
    plans: &[ScanPlan],
    restart_interval: u32,
) -> Result<(usize, u8)> {
    assert_eq!(plans.len(), frame.components.len());
    assert!(plans.len() <= COMPONENTS_MAX);

    let mcu_count = frame
        .mcu_columns
        .checked_mul(frame.mcu_rows)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("MCU count overflowed"))?;
    let mut reader = BitReader::new(entropy);
    let mut dc_predictors = [0_i32; COMPONENTS_MAX];
    let mut restart_index = 0_u8;
    for mcu_index in 0..mcu_count {
        let mcu_x = mcu_index % frame.mcu_columns;
        let mcu_y = mcu_index / frame.mcu_columns;
        decode_mcu(&mut reader, frame, plans, &mut dc_predictors, mcu_x, mcu_y)?;

        let completed = mcu_index + 1;
        if restart_interval > 0 && completed < mcu_count && completed % restart_interval == 0 {
            reader.restart(0xd0 + restart_index)?;
            dc_predictors.fill(0);
            restart_index = (restart_index + 1) & 7;
        }
    }
    reader.finish()
}

fn decode_mcu(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    plans: &[ScanPlan],
    dc_predictors: &mut [i32; COMPONENTS_MAX],
    mcu_x: u32,
    mcu_y: u32,
) -> Result<()> {
    assert!(mcu_x < frame.mcu_columns);
    assert!(mcu_y < frame.mcu_rows);

    for (scan_index, plan) in plans.iter().enumerate() {
        for block_y in 0..plan.vertical_sampling {
            for block_x in 0..plan.horizontal_sampling {
                let coefficients = decode_block(reader, plan, &mut dc_predictors[scan_index])?;
                let samples = idct::inverse(&coefficients);
                let component = &mut frame.components[plan.frame_index];
                let plane_block_x =
                    mcu_x * u32::from(plan.horizontal_sampling) + u32::from(block_x);
                let plane_block_y = mcu_y * u32::from(plan.vertical_sampling) + u32::from(block_y);
                write_block(component, plane_block_x, plane_block_y, &samples);
            }
        }
    }
    Ok(())
}

fn decode_block(
    reader: &mut BitReader<'_>,
    plan: &ScanPlan,
    dc_predictor: &mut i32,
) -> Result<[i32; 64]> {
    assert!(plan.frame_index < COMPONENTS_MAX);
    assert!(plan.quantization.iter().all(|value| *value > 0));

    let mut coefficients = [0_i32; 64];
    let dc_category = plan.dc.decode(reader)?;
    if dc_category > 11 {
        return Err(error(JPEGError::Entropy(
            "baseline DC coefficient category exceeds 11 bits",
        )));
    }
    let dc_difference = receive_extend(reader, dc_category)?;
    *dc_predictor = dc_predictor
        .checked_add(dc_difference)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("DC predictor overflowed"))?;
    coefficients[0] = dc_predictor
        .checked_mul(i32::from(plan.quantization[0]))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("dequantized DC coefficient overflowed"))?;

    let mut zigzag_index = 1_usize;
    while zigzag_index < 64 {
        let symbol = plan.ac.decode(reader)?;
        let zero_run = usize::from(symbol >> 4);
        let category = symbol & 0x0f;
        if category == 0 {
            if zero_run == 0 {
                break;
            }
            if zero_run != 15 {
                return Err(error(JPEGError::Entropy(
                    "invalid zero-size AC Huffman symbol",
                )));
            }
            zigzag_index += 16;
            continue;
        }
        if category > 10 {
            return Err(error(JPEGError::Entropy(
                "baseline AC coefficient category exceeds 10 bits",
            )));
        }
        zigzag_index += zero_run;
        if zigzag_index >= 64 {
            return Err(error(JPEGError::Entropy(
                "AC coefficient run extends past its block",
            )));
        }
        let natural_index = ZIGZAG_TO_NATURAL[zigzag_index];
        let value = receive_extend(reader, category)?;
        coefficients[natural_index] = value
            .checked_mul(i32::from(plan.quantization[natural_index]))
            .ok_or_raise(|| {
                JPEGError::ArithmeticOverflow("dequantized AC coefficient overflowed")
            })?;
        zigzag_index += 1;
    }
    Ok(coefficients)
}

fn receive_extend(reader: &mut BitReader<'_>, category: u8) -> Result<i32> {
    assert!(category <= 16);
    assert!(u32::from(category) < i32::BITS);

    if category == 0 {
        return Ok(0);
    }
    let bits = i32::from(reader.read_bits(category)?);
    let threshold = 1_i32 << (category - 1);
    if bits < threshold {
        Ok(bits + 1 - (1_i32 << category))
    } else {
        Ok(bits)
    }
}

fn write_block(component: &mut FrameComponent, block_x: u32, block_y: u32, samples: &[u8; 64]) {
    assert!(component.plane_width > 0);
    assert!(!component.plane.is_empty());

    let pixel_x = block_x * BLOCK_SIDE;
    let pixel_y = block_y * BLOCK_SIDE;
    for y in 0..BLOCK_SIDE {
        let target_start =
            u64::from(pixel_y + y) * u64::from(component.plane_width) + u64::from(pixel_x);
        let target_end = target_start + u64::from(BLOCK_SIDE);
        let source_start = (y * BLOCK_SIDE) as usize;
        let source_end = source_start + BLOCK_SIDE as usize;
        component.plane[target_start as usize..target_end as usize]
            .copy_from_slice(&samples[source_start..source_end]);
    }
}

struct ComponentStorageLayout {
    width: u32,
    height: u32,
    mcu_columns: u32,
    mcu_rows: u32,
    max_horizontal_sampling: u8,
    max_vertical_sampling: u8,
    process: CodingProcess,
}

fn validate_progressive_storage(
    components: &[FrameComponent],
    layout: &ComponentStorageLayout,
) -> Result<()> {
    assert!(!components.is_empty());
    assert!(components.len() <= COMPONENTS_MAX);

    if layout.process == CodingProcess::Sequential {
        return Ok(());
    }
    let mut byte_count = 0_u64;
    for component in components {
        let blocks = u64::from(layout.mcu_columns)
            * u64::from(component.horizontal_sampling)
            * u64::from(layout.mcu_rows)
            * u64::from(component.vertical_sampling);
        let component_bytes = blocks * 64 * size_of::<i32>() as u64;
        byte_count = byte_count.checked_add(component_bytes).ok_or_raise(|| {
            JPEGError::ArithmeticOverflow("progressive coefficient storage overflowed")
        })?;
    }
    if byte_count > PROGRESSIVE_COEFFICIENT_BYTES_MAX {
        return Err(error(JPEGError::LimitExceeded(
            JPEGLimit::ProgressiveCoefficientBytes(PROGRESSIVE_COEFFICIENT_BYTES_MAX),
        )));
    }
    Ok(())
}

fn allocate_component_storage(
    component: &mut FrameComponent,
    layout: &ComponentStorageLayout,
) -> Result<()> {
    assert!(component.plane.is_empty());
    assert!(component.horizontal_sampling > 0);

    let block_columns = layout
        .mcu_columns
        .checked_mul(u32::from(component.horizontal_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component block width overflowed"))?;
    let block_rows = layout
        .mcu_rows
        .checked_mul(u32::from(component.vertical_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component block height overflowed"))?;
    let plane_width = block_columns
        .checked_mul(BLOCK_SIDE)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component plane width overflowed"))?;
    let plane_height = block_rows
        .checked_mul(BLOCK_SIDE)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component plane height overflowed"))?;
    let sample_count = plane_width
        .checked_mul(plane_height)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component plane size overflowed"))?;
    let data_width = layout
        .width
        .checked_mul(u32::from(component.horizontal_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component data width overflowed"))?;
    let data_height = layout
        .height
        .checked_mul(u32::from(component.vertical_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component data height overflowed"))?;
    let data_block_columns = divide_ceil(
        data_width,
        u32::from(layout.max_horizontal_sampling) * BLOCK_SIDE,
    );
    let data_block_rows = divide_ceil(
        data_height,
        u32::from(layout.max_vertical_sampling) * BLOCK_SIDE,
    );

    component.plane_width = plane_width;
    component.plane = vec![0; sample_count as usize];
    component.block_columns = block_columns;
    component.block_rows = block_rows;
    component.data_block_columns = data_block_columns;
    component.data_block_rows = data_block_rows;
    if layout.process == CodingProcess::Progressive {
        let block_count = block_columns.checked_mul(block_rows).ok_or_raise(|| {
            JPEGError::ArithmeticOverflow("component coefficient count overflowed")
        })?;
        component.coefficients = vec![[0; 64]; block_count as usize];
    }
    assert_eq!(component.plane.len(), sample_count as usize);
    Ok(())
}

fn convert_color(first: u8, second: u8, third: u8, transform: ColorTransform) -> [u8; 4] {
    match transform {
        ColorTransform::Rgb => [first, second, third, 255],
        ColorTransform::YCbCr => {
            let luminance = f32::from(first);
            let blue_difference = f32::from(second) - 128.0;
            let red_difference = f32::from(third) - 128.0;
            let red = luminance + 1.402 * red_difference;
            let green = luminance - 0.344_136 * blue_difference - 0.714_136 * red_difference;
            let blue = luminance + 1.772 * blue_difference;
            [clamp_color(red), clamp_color(green), clamp_color(blue), 255]
        }
    }
}

fn clamp_color(value: f32) -> u8 {
    assert!(value.is_finite());
    value.round().clamp(f32::from(u8::MIN), f32::from(u8::MAX)) as u8
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(error(JPEGError::Frame("JPEG dimensions must be nonzero")));
    }
    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(JPEGError::LimitExceeded(JPEGLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }
    if u64::from(width) * u64::from(height) > PIXELS_MAX {
        return Err(error(JPEGError::LimitExceeded(JPEGLimit::Pixels(
            PIXELS_MAX,
        ))));
    }
    Ok(())
}

fn validate_huffman_symbols(class: u8, symbols: &[u8]) -> Result<()> {
    assert!(class <= 1);
    assert!(symbols.len() <= 256);

    for symbol in symbols {
        if class == 0 && *symbol > 11 {
            return Err(error(JPEGError::Table(
                JPEGTableKind::DCHuffman,
                "DC Huffman table contains a category above 11",
            )));
        }
        if class == 1 {
            let category = symbol & 0x0f;
            if category > 10 {
                return Err(error(JPEGError::Table(
                    JPEGTableKind::ACHuffman,
                    "AC Huffman table contains an invalid symbol",
                )));
            }
        }
    }
    Ok(())
}

fn unsupported_frame_error(marker: u8) -> Error {
    assert!((0xc1..=0xcf).contains(&marker));
    assert_ne!(marker, MARKER_DHT);

    error(JPEGError::Unsupported(UnsupportedJPEG::FrameType(marker)))
}

fn unsupported_marker_error(marker: u8) -> Error {
    assert_ne!(marker, 0x00);
    assert_ne!(marker, 0xff);

    error(JPEGError::Unsupported(UnsupportedJPEG::Marker(marker)))
}

fn divide_ceil(value: u32, divisor: u32) -> u32 {
    assert!(value > 0);
    assert!(divisor > 0);
    value.div_ceil(divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_progressive_spectral_and_refinement_scans() {
        let jpeg = test_data::progressive_color_jpeg();
        let image = decode(&jpeg).unwrap();

        assert_eq!(image.dimensions(), (227, 149));
        assert_eq!(image.rgba8().len(), 227 * 149 * 4);
        assert_matches_reference_decoder(&jpeg, &image);
    }

    #[test]
    fn decodes_sequential_arithmetic_scans() {
        // This is libjpeg-turbo's arithmetic-coded interoperability fixture. Using an
        // independently produced stream guards the QM state transitions and the JPEG
        // coefficient contexts together; testing either layer alone would miss their seam.
        let jpeg = test_data::sequential_arithmetic_jpeg();
        let image = decode(&jpeg).unwrap();
        let huffman = decode(&test_data::progressive_color_jpeg()).unwrap();

        assert_eq!(image.dimensions(), (227, 149));
        assert_eq!(image.rgba8().len(), 227 * 149 * 4);
        // Both official fixtures were encoded from the same source with the same
        // quantization. Their pixels must agree even though every entropy path differs.
        assert_eq!(image.rgba8(), huffman.rgba8());
    }

    #[test]
    fn decodes_progressive_arithmetic_refinement_scans() {
        // The arithmetic and Huffman fixtures use the same source, DCT, scan script,
        // sampling, and quantization. Equality therefore verifies SOF10 first-pass and
        // refinement contexts without allowing the shared color pipeline any tolerance.
        let jpeg = test_data::progressive_arithmetic_jpeg();
        let image = decode(&jpeg).unwrap();
        let huffman = decode(&test_data::progressive_color_jpeg()).unwrap();

        assert_eq!(image.dimensions(), (227, 149));
        assert_eq!(image.rgba8().len(), 227 * 149 * 4);
        assert_eq!(image.rgba8(), huffman.rgba8());
    }

    #[test]
    fn rejects_an_inverted_dc_arithmetic_conditioning_range() {
        // L=1 and U=0 cannot describe a DC magnitude interval. Reject it at DAC so
        // corrupt context thresholds never enter the entropy decoder.
        let jpeg = [0xff, 0xd8, 0xff, 0xcc, 0x00, 0x04, 0x00, 0x01];
        let error = decode(&jpeg).unwrap_err();

        assert_eq!(
            &*error,
            &JPEGError::Table(
                JPEGTableKind::ArithmeticConditioning,
                "DC arithmetic conditioning requires L <= U",
            )
        );
        assert!(error.to_string().contains("L <= U"));
    }

    #[test]
    fn rejects_a_zero_ac_arithmetic_conditioning_value() {
        // K=0 is outside the standard's 1..=63 interval and would collapse the
        // AC magnitude-context split, so it is an input error rather than a default.
        let jpeg = [0xff, 0xd8, 0xff, 0xcc, 0x00, 0x04, 0x10, 0x00];
        let error = decode(&jpeg).unwrap_err();

        assert_eq!(
            &*error,
            &JPEGError::Table(
                JPEGTableKind::ArithmeticConditioning,
                "AC arithmetic conditioning must be in 1..=63",
            )
        );
        assert!(error.to_string().contains("1..=63"));
    }

    #[test]
    fn rejects_a_scan_before_the_frame_phase() {
        let jpeg = [0xff, 0xd8, 0xff, 0xda];
        let error = decode(&jpeg).unwrap_err();

        assert_eq!(
            &*error,
            &JPEGError::Frame("SOS marker appeared before a frame")
        );
        assert!(error.to_string().contains("before a frame"));
        assert!(!error.to_string().contains("entropy"));
    }

    #[test]
    fn preserves_the_value_of_an_unsupported_marker() {
        let jpeg = [0xff, 0xd8, 0xff, 0x01];
        let error = decode(&jpeg).unwrap_err();

        assert_eq!(
            &*error,
            &JPEGError::Unsupported(UnsupportedJPEG::Marker(0x01))
        );
        assert_eq!(error.to_string(), "unsupported JPEG marker FF01");
    }

    #[test]
    fn rejects_end_of_image_before_the_scanned_phase() {
        let jpeg = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11,
            0x00, 0xff, 0xd9,
        ];
        let error = decode(&jpeg).unwrap_err();

        assert!(error.to_string().contains("before its first scan"));
        assert!(!error.to_string().contains("Huffman"));
    }

    #[test]
    fn rejects_zero_dimensions_before_allocating() {
        let jpeg = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x11,
            0x00,
        ];
        let error = decode(&jpeg).unwrap_err();

        assert!(error.to_string().contains("nonzero"));
        assert!(!error.to_string().contains("entropy"));
    }

    #[test]
    fn truncated_segment_returns_an_error_instead_of_panicking() {
        let jpeg = [0xff, 0xd8, 0xff, 0xfe, 0x00];
        let error = decode(&jpeg).unwrap_err();

        assert!(error.to_string().contains("end"));
        assert!(!error.to_string().is_empty());
        assert!(error.frame().location().file().ends_with("reader.rs"));
        assert!(error.frame().children().is_empty());
    }

    #[test]
    fn decodes_a_subsampled_color_jpeg_end_to_end() {
        let image = decode(&test_data::baseline_color_jpeg()).unwrap();

        assert_eq!(image.dimensions(), (16, 16));
        assert_eq!(image.rgba8().len(), 16 * 16 * 4);
        assert_dominant(&image, 3, 3, 0);
        assert_dominant(&image, 12, 3, 1);
        assert_dominant(&image, 3, 12, 2);
        let white = pixel(&image, 12, 12);
        assert!(white[0] > 220 && white[1] > 220 && white[2] > 220);
    }

    fn assert_dominant(image: &dyn Image, x: usize, y: usize, channel: usize) {
        assert!(x < image.width() as usize);
        assert!(y < image.height() as usize);

        let pixel = pixel(image, x, y);
        assert!(pixel[channel] > 180);
        for other_channel in 0..3 {
            if other_channel != channel {
                assert!(pixel[channel] > pixel[other_channel] + 80);
            }
        }
    }

    fn pixel(image: &dyn Image, x: usize, y: usize) -> &[u8] {
        assert!(x < image.width() as usize);
        assert!(y < image.height() as usize);

        let start = (y * image.width() as usize + x) * 4;
        &image.rgba8()[start..start + 4]
    }

    fn assert_matches_reference_decoder(jpeg: &[u8], image: &dyn Image) {
        use std::sync::Arc;

        let reference = gpui::Image::from_bytes(gpui::ImageFormat::Jpeg, jpeg.to_vec());
        let renderer = gpui::SvgRenderer::new(Arc::new(()));
        let rendered = reference.to_image_data(renderer).unwrap();
        let reference_bgra = rendered.as_bytes(0).unwrap();
        assert_eq!(reference_bgra.len(), image.rgba8().len());

        let (actual_pixels, actual_remainder) = image.rgba8().as_chunks::<4>();
        let (expected_pixels, expected_remainder) = reference_bgra.as_chunks::<4>();
        assert!(actual_remainder.is_empty());
        assert!(expected_remainder.is_empty());
        let mut error_sum = 0_u64;
        for (actual, expected) in actual_pixels.iter().zip(expected_pixels) {
            error_sum += u64::from(actual[0].abs_diff(expected[2]));
            error_sum += u64::from(actual[1].abs_diff(expected[1]));
            error_sum += u64::from(actual[2].abs_diff(expected[0]));
        }
        let sample_count = u64::from(image.width()) * u64::from(image.height()) * 3;
        assert!(error_sum / sample_count < 20);
    }
}
