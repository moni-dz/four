//! Drives the marker stream through explicit header, frame, and scanned phases.

use std::array;
use std::num::NonZeroU32;

use exn::OptionExt;

use super::arithmetic::ConditioningTables;
use super::reader::Reader;
use super::{
    COMPONENTS_MAX, CodingProcess, ColorTransform, DecodedImage, Error, Frame, JPEGError,
    JPEGLimit, JPEGTableKind, QUANTIZATION_TABLES_MAX, Result, SCANS_MAX, ScanComponent,
    ScanHeader, UnsupportedJPEG, ZIGZAG_TO_NATURAL, arithmetic, error, parse_frame_components,
    parse_scan_header, validate_dimensions,
};

const MARKER_SOI: u8 = 0xd8;
const MARKER_EOI: u8 = 0xd9;
const MARKER_SOF9: u8 = 0xc9;
const MARKER_SOF10: u8 = 0xca;
const MARKER_DHT: u8 = 0xc4;
const MARKER_DAC: u8 = 0xcc;
const MARKER_DQT: u8 = 0xdb;
const MARKER_DRI: u8 = 0xdd;
const MARKER_SOS: u8 = 0xda;

// NonZeroU32 makes the scanned state unrepresentable until the first scan succeeds.
const _: () = invariant!(size_of::<NonZeroU32>() == size_of::<u32>());

pub(super) fn decode(bytes: &[u8]) -> Result<DecodedImage> {
    invariant!(isize::try_from(bytes.len()).is_ok());
    Parser::<Headers>::new(bytes).decode()
}

// The parser phases make it impossible to decode entropy before a frame exists or to produce an
// image before at least one scan has completed. JPEG table presence stays dynamic because the file
// format permits tables to be redefined between scans; that is input state, not parser lifecycle.
struct Parser<'a, State> {
    reader: Reader<'a>,
    quantization_tables: [Option<[u16; 64]>; QUANTIZATION_TABLES_MAX],
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
        invariant!(!self.data.frame.components.is_empty());
        invariant!(self.data.frame.components.len() <= COMPONENTS_MAX);
        &self.data
    }

    fn data_mut(&mut self) -> &mut FrameData {
        invariant!(!self.data.frame.components.is_empty());
        invariant!(self.data.frame.components.len() <= COMPONENTS_MAX);
        &mut self.data
    }
}

impl FramePhase for Scanned {
    fn data(&self) -> &FrameData {
        invariant!(self.scan_count.get() <= SCANS_MAX);
        invariant!(!self.data.frame.components.is_empty());
        &self.data
    }

    fn data_mut(&mut self) -> &mut FrameData {
        invariant!(self.scan_count.get() <= SCANS_MAX);
        invariant!(!self.data.frame.components.is_empty());
        &mut self.data
    }
}

impl<'a> Parser<'a, Headers> {
    fn new(bytes: &'a [u8]) -> Self {
        invariant!(isize::try_from(bytes.len()).is_ok());

        Self {
            reader: Reader::new(bytes),
            quantization_tables: array::from_fn(|_| None),
            arithmetic_conditioning: ConditioningTables::defaults(),
            restart_interval: 0,
            color_transform: ColorTransform::YCbCr,
            state: Headers,
        }
    }

    fn decode(mut self) -> Result<DecodedImage> {
        invariant_eq!(self.restart_interval, 0);

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
                MARKER_SOF9 => {
                    return self
                        .parse_frame(CodingProcess::Sequential)?
                        .decode_until_first_scan();
                }
                MARKER_SOF10 => {
                    return self
                        .parse_frame(CodingProcess::Progressive)?
                        .decode_until_first_scan();
                }
                MARKER_DHT => self.skip_segment()?,
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
                0xc0..=0xc3 | 0xc5..=0xc8 | 0xcb | 0xcd..=0xcf => {
                    return Err(unsupported_frame_error(marker));
                }
                _ => return Err(unsupported_marker_error(marker)),
            }

            marker = self.reader.marker()?;
        }
    }

    fn parse_frame(mut self, process: CodingProcess) -> Result<Parser<'a, FrameReady>> {
        invariant_eq!(size_of::<Headers>(), 0);
        invariant!(isize::try_from(self.reader.remaining()).is_ok());

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
                UnsupportedJPEG::ComponentCount(
                    u8::try_from(component_count)
                        .expect("the component count originated from a single input byte"),
                ),
            )));
        }

        let components = parse_frame_components(&mut segment, component_count)?;
        if segment.remaining() != 0 {
            return Err(error(JPEGError::Segment("SOF segment has trailing bytes")));
        }

        let frame = Frame::new(width, height, components, process)?;
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
        invariant!(isize::try_from(self.reader.remaining()).is_ok());
        invariant!(u16::try_from(self.restart_interval).is_ok());

        let state = transition(self.state);

        Parser {
            reader: self.reader,
            quantization_tables: self.quantization_tables,
            arithmetic_conditioning: self.arithmetic_conditioning,
            restart_interval: self.restart_interval,
            color_transform: self.color_transform,
            state,
        }
    }

    fn parse_quantization_tables(&mut self) -> Result<()> {
        invariant_eq!(self.quantization_tables.len(), QUANTIZATION_TABLES_MAX);
        invariant!(isize::try_from(self.reader.remaining()).is_ok());

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

    fn parse_arithmetic_conditioning(&mut self) -> Result<()> {
        invariant_eq!(
            self.arithmetic_conditioning.dc.len(),
            arithmetic::TABLES_MAX
        );
        invariant_eq!(
            self.arithmetic_conditioning.ac.len(),
            arithmetic::TABLES_MAX
        );

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

            if table >= arithmetic::TABLES_MAX {
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
        invariant!(u16::try_from(self.restart_interval).is_ok());
        invariant!(isize::try_from(self.reader.remaining()).is_ok());

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
        invariant!((0xe0..=0xef).contains(&marker));
        invariant!(isize::try_from(self.reader.remaining()).is_ok());

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

impl Parser<'_, FrameReady> {
    fn decode_until_first_scan(mut self) -> Result<DecodedImage> {
        invariant_eq!(
            self.state.data.coefficient_bits,
            [[None; 64]; COMPONENTS_MAX]
        );
        invariant_eq!(self.state.data.component_scanned, [false; COMPONENTS_MAX]);
        invariant!(!self.state.data.frame.components.is_empty());

        let mut marker = self.reader.marker()?;
        loop {
            match marker {
                MARKER_DQT => self.parse_quantization_tables()?,
                MARKER_DHT => self.skip_segment()?,
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
                0xc0..=0xc3 | 0xc5..=0xca | 0xcb | 0xcd..=0xcf => {
                    return Err(error(JPEGError::Frame(
                        "multiple JPEG frames are not supported",
                    )));
                }
                _ => return Err(unsupported_marker_error(marker)),
            }

            marker = self.reader.marker()?;
        }
    }
}

impl Parser<'_, Scanned> {
    fn decode_after_scan(mut self, mut marker: u8) -> Result<DecodedImage> {
        invariant!(self.state.scan_count.get() <= SCANS_MAX);
        invariant!(!self.state.data.frame.components.is_empty());

        loop {
            match marker {
                MARKER_DQT => self.parse_quantization_tables()?,
                MARKER_DHT => self.skip_segment()?,
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
                0xc0..=0xc3 | 0xc5..=0xca | 0xcb | 0xcd..=0xcf => {
                    return Err(error(JPEGError::Frame(
                        "multiple JPEG frames are not supported",
                    )));
                }
                _ => return Err(unsupported_marker_error(marker)),
            }

            marker = self.reader.marker()?;
        }
    }

    fn finish(self) -> Result<DecodedImage> {
        invariant!(self.state.scan_count.get() <= SCANS_MAX);
        invariant!(!self.state.data.frame.components.is_empty());

        let mut frame = self.state.data.frame;
        if frame.process == CodingProcess::Progressive {
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

        Ok(frame.into_image(self.color_transform))
    }
}

impl<State: FramePhase> Parser<'_, State> {
    fn parse_scan_and_decode(&mut self, scan_count: u32) -> Result<u8> {
        if scan_count >= SCANS_MAX {
            return Err(error(JPEGError::LimitExceeded(JPEGLimit::Scans(SCANS_MAX))));
        }

        let scan = {
            let mut segment = self.reader.segment()?;
            parse_scan_header(&mut segment, &self.state.data().frame)?
        };

        let process = self.state.data().frame.process;
        let result = match process {
            CodingProcess::Sequential => self.decode_arithmetic_sequential_scan(&scan)?,
            CodingProcess::Progressive => self.decode_arithmetic_progressive_scan(&scan)?,
        };

        let (bytes_consumed, marker) = result;
        let entropy_length = self.reader.remaining();
        self.reader.advance(bytes_consumed)?;
        invariant!(bytes_consumed <= entropy_length);

        if process == CodingProcess::Progressive {
            self.commit_progression(&scan);
        } else {
            for component in &scan.components {
                self.state.data_mut().component_scanned[component.frame_index] = true;
            }
        }

        Ok(marker)
    }

    fn decode_arithmetic_sequential_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        invariant!(!scan.components.is_empty());
        invariant!(scan.components.len() <= COMPONENTS_MAX);
        invariant_eq!(self.state.data().frame.process, CodingProcess::Sequential);

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

    fn decode_arithmetic_progressive_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        invariant!(scan.spectral_start <= scan.spectral_end);
        invariant!(scan.spectral_end < 64);
        invariant_eq!(self.state.data().frame.process, CodingProcess::Progressive);

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

    fn build_arithmetic_sequential_plans(
        &self,
        scan: &[ScanComponent],
    ) -> Result<Vec<arithmetic::SequentialPlan>> {
        let frame = &self.state.data().frame;
        invariant!(!scan.is_empty());
        invariant!(scan.len() <= COMPONENTS_MAX);

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
        invariant!(!scan.components.is_empty());
        invariant!(scan.components.len() <= COMPONENTS_MAX);

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

    fn validate_progression(&self, scan: &ScanHeader) -> Result<()> {
        invariant!(scan.spectral_start <= scan.spectral_end);
        invariant!(scan.components.len() <= COMPONENTS_MAX);

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
        invariant!(scan.spectral_start <= scan.spectral_end);
        invariant!(scan.components.len() <= COMPONENTS_MAX);

        for component in &scan.components {
            for coefficient in scan.spectral_start..=scan.spectral_end {
                self.state.data_mut().coefficient_bits[component.frame_index]
                    [usize::from(coefficient)] = Some(scan.successive_low);
            }
        }
    }
}

fn unsupported_frame_error(marker: u8) -> Error {
    invariant!((0xc1..=0xcf).contains(&marker));
    invariant_ne!(marker, MARKER_DHT);

    error(JPEGError::Unsupported(UnsupportedJPEG::FrameType(marker)))
}

fn unsupported_marker_error(marker: u8) -> Error {
    invariant_ne!(marker, 0x00);
    invariant_ne!(marker, 0xff);

    error(JPEGError::Unsupported(UnsupportedJPEG::Marker(marker)))
}
