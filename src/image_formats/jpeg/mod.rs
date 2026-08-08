mod huffman;
mod idct;
mod progressive;
mod reader;
#[cfg(test)]
mod test_data;

use std::array;
use std::fmt;

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
const MARKER_DHT: u8 = 0xc4;
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
    assert!(ZIGZAG_TO_NATURAL.len() == 64);
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    message: String,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!message.is_empty());
        assert!(message.len() <= 1024);
        Self { message }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(!self.message.is_empty());
        assert!(self.message.len() <= 1024);
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// Decode an 8-bit Huffman-coded baseline or progressive JPEG without an image-decoding crate.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage> {
    assert!(bytes.len() <= isize::MAX as usize);
    Parser::new(bytes).decode()
}

struct Parser<'a> {
    reader: Reader<'a>,
    quantization_tables: [Option<[u16; 64]>; QUANTIZATION_TABLES_MAX],
    dc_tables: [Option<HuffmanTable>; HUFFMAN_TABLES_MAX],
    ac_tables: [Option<HuffmanTable>; HUFFMAN_TABLES_MAX],
    frame: Option<Frame>,
    restart_interval: u32,
    color_transform: ColorTransform,
    coefficient_bits: [[Option<u8>; 64]; COMPONENTS_MAX],
    scan_count: u32,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        Self {
            reader: Reader::new(bytes),
            quantization_tables: array::from_fn(|_| None),
            dc_tables: array::from_fn(|_| None),
            ac_tables: array::from_fn(|_| None),
            frame: None,
            restart_interval: 0,
            color_transform: ColorTransform::Ycbcr,
            coefficient_bits: [[None; 64]; COMPONENTS_MAX],
            scan_count: 0,
        }
    }

    fn decode(mut self) -> Result<DecodedImage> {
        assert!(self.frame.is_none());
        assert_eq!(self.restart_interval, 0);

        if self.reader.marker()? != MARKER_SOI {
            return Err(Error::new("JPEG does not begin with an SOI marker"));
        }
        let mut marker = self.reader.marker()?;
        loop {
            match marker {
                MARKER_DQT => self.parse_quantization_tables()?,
                MARKER_SOF0 => self.parse_frame(CodingProcess::Baseline)?,
                MARKER_SOF2 => self.parse_frame(CodingProcess::Progressive)?,
                MARKER_DHT => self.parse_huffman_tables()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_SOS => {
                    marker = self.parse_scan_and_decode()?;
                    continue;
                }
                0xe0..=0xef => self.parse_application_segment(marker)?,
                0xfe => self.skip_segment()?,
                MARKER_EOI => return self.finish(),
                MARKER_SOI => return Err(Error::new("duplicate SOI marker")),
                0xc1 | 0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                    return Err(unsupported_frame_error(marker));
                }
                0xcc => return Err(Error::new("arithmetic-coded JPEG is not supported")),
                _ => {
                    return Err(Error::new(format!(
                        "unsupported JPEG marker FF{marker:02X}"
                    )));
                }
            }
            marker = self.reader.marker()?;
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
                return Err(Error::new("quantization table index is out of range"));
            }
            if precision > 1 {
                return Err(Error::new("quantization table precision is invalid"));
            }

            let mut table = [0_u16; 64];
            for natural_index in ZIGZAG_TO_NATURAL {
                let value = if precision == 0 {
                    u16::from(segment.read_u8()?)
                } else {
                    segment.read_u16()?
                };
                if value == 0 {
                    return Err(Error::new("quantization table contains a zero value"));
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
                return Err(Error::new("Huffman table descriptor is invalid"));
            }

            let mut counts = [0_u8; 16];
            for count in &mut counts {
                *count = segment.read_u8()?;
            }
            let symbol_count: usize = counts.iter().map(|count| usize::from(*count)).sum();
            if symbol_count > 256 {
                return Err(Error::new("Huffman table has more than 256 symbols"));
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

    fn parse_frame(&mut self, process: CodingProcess) -> Result<()> {
        if self.frame.is_some() {
            return Err(Error::new("multiple JPEG frames are not supported"));
        }
        assert!(self.frame.is_none());
        let mut segment = self.reader.segment()?;
        if segment.read_u8()? != 8 {
            return Err(Error::new("only 8-bit DCT JPEG samples are supported"));
        }
        let height = u32::from(segment.read_u16()?);
        let width = u32::from(segment.read_u16()?);
        validate_dimensions(width, height)?;
        let component_count = usize::from(segment.read_u8()?);
        if component_count != 1 && component_count != 3 {
            return Err(Error::new(
                "only grayscale and three-component JPEGs are supported",
            ));
        }

        let components = parse_frame_components(&mut segment, component_count)?;
        if segment.remaining() != 0 {
            return Err(Error::new("SOF0 segment has trailing bytes"));
        }
        self.frame = Some(Frame::new(width, height, components, process)?);
        Ok(())
    }

    fn parse_restart_interval(&mut self) -> Result<()> {
        assert!(self.restart_interval <= u32::from(u16::MAX));
        assert!(self.reader.remaining() <= isize::MAX as usize);

        let mut segment = self.reader.segment()?;
        if segment.remaining() != 2 {
            return Err(Error::new("DRI segment must contain exactly two bytes"));
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
            self.color_transform = ColorTransform::Ycbcr;
        }
        if marker == 0xee && bytes.starts_with(b"Adobe") && bytes.len() >= 12 {
            self.color_transform = match bytes[11] {
                0 => ColorTransform::Rgb,
                1 => ColorTransform::Ycbcr,
                value => {
                    return Err(Error::new(format!(
                        "Adobe JPEG color transform {value} is not supported"
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

    fn parse_scan_and_decode(&mut self) -> Result<u8> {
        if self.scan_count >= SCANS_MAX {
            return Err(Error::new("JPEG contains more than 4096 scans"));
        }
        let scan = {
            let mut segment = self.reader.segment()?;
            parse_scan_header(&mut segment, self.frame.as_ref())?
        };
        let process = self
            .frame
            .as_ref()
            .map(|frame| frame.process)
            .ok_or_else(|| Error::new("SOS marker appeared before SOF0"))?;
        let result = match process {
            CodingProcess::Baseline => self.decode_baseline_scan(&scan)?,
            CodingProcess::Progressive => self.decode_progressive_scan(&scan)?,
        };
        let (bytes_consumed, marker) = result;
        let entropy_length = self.reader.remaining();
        self.reader.advance(bytes_consumed)?;
        assert!(bytes_consumed <= entropy_length);
        self.commit_progression(&scan);
        self.scan_count += 1;
        Ok(marker)
    }

    fn decode_baseline_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        assert_eq!(
            scan.components.len(),
            self.frame.as_ref().unwrap().components.len()
        );
        assert!(scan.spectral_start == 0 && scan.spectral_end == 63);

        if self.scan_count != 0 {
            return Err(Error::new("baseline JPEG contains more than one scan"));
        }
        let plans = self.build_scan_plans(&scan.components)?;
        let entropy = self.reader.remaining_slice();
        let frame = self.frame.as_mut().expect("frame was checked");
        decode_entropy(entropy, frame, &plans, self.restart_interval)
    }

    fn decode_progressive_scan(&mut self, scan: &ScanHeader) -> Result<(usize, u8)> {
        assert!(scan.spectral_start <= scan.spectral_end);
        assert!(scan.spectral_end < 64);

        self.validate_progression(scan)?;
        let plans = self.build_progressive_plans(scan)?;
        let entropy = self.reader.remaining_slice();
        let frame = self.frame.as_mut().expect("frame was checked");
        progressive::decode_scan(entropy, frame, &plans, scan, self.restart_interval)
    }

    fn build_scan_plans(&self, scan: &[ScanComponent]) -> Result<Vec<ScanPlan>> {
        let frame = self.frame.as_ref().expect("frame was checked");
        assert!(scan.len() <= COMPONENTS_MAX);
        assert_eq!(scan.len(), frame.components.len());

        let mut plans = Vec::with_capacity(scan.len());
        for component in scan {
            let frame_component = &frame.components[component.frame_index];
            let quantization = self.quantization_tables[frame_component.quantization_table]
                .ok_or_else(|| Error::new("scan references a missing quantization table"))?;
            let dc = self.dc_tables[component.dc_table]
                .clone()
                .ok_or_else(|| Error::new("scan references a missing DC Huffman table"))?;
            let ac = self.ac_tables[component.ac_table]
                .clone()
                .ok_or_else(|| Error::new("scan references a missing AC Huffman table"))?;
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

    fn build_progressive_plans(&self, scan: &ScanHeader) -> Result<Vec<ProgressivePlan>> {
        let frame = self.frame.as_ref().expect("frame was checked");
        assert!(scan.components.len() <= COMPONENTS_MAX);
        assert_eq!(frame.process, CodingProcess::Progressive);

        let mut plans = Vec::with_capacity(scan.components.len());
        for component in &scan.components {
            let frame_component = &frame.components[component.frame_index];
            let dc = if scan.spectral_start == 0 && scan.successive_high == 0 {
                Some(self.dc_tables[component.dc_table].clone().ok_or_else(|| {
                    Error::new("progressive DC scan references a missing Huffman table")
                })?)
            } else {
                None
            };
            let ac = if scan.spectral_start > 0 {
                Some(self.ac_tables[component.ac_table].clone().ok_or_else(|| {
                    Error::new("progressive AC scan references a missing Huffman table")
                })?)
            } else {
                None
            };
            plans.push(ProgressivePlan {
                frame_index: component.frame_index,
                horizontal_sampling: frame_component.horizontal_sampling,
                vertical_sampling: frame_component.vertical_sampling,
                dc,
                ac,
            });
        }
        Ok(plans)
    }

    fn validate_progression(&self, scan: &ScanHeader) -> Result<()> {
        assert!(scan.spectral_start <= scan.spectral_end);
        assert!(scan.components.len() <= COMPONENTS_MAX);

        for component in &scan.components {
            if scan.spectral_start > 0 && self.coefficient_bits[component.frame_index][0].is_none()
            {
                return Err(Error::new(
                    "progressive AC scan appeared before its DC scan",
                ));
            }
            for coefficient in scan.spectral_start..=scan.spectral_end {
                let previous =
                    self.coefficient_bits[component.frame_index][usize::from(coefficient)];
                if scan.successive_high == 0 && previous.is_some() {
                    return Err(Error::new(
                        "progressive coefficient band was initialized twice",
                    ));
                }
                if scan.successive_high > 0 && previous != Some(scan.successive_high) {
                    return Err(Error::new(
                        "progressive refinement has an inconsistent bit order",
                    ));
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
                self.coefficient_bits[component.frame_index][usize::from(coefficient)] =
                    Some(scan.successive_low);
            }
        }
    }

    fn finish(mut self) -> Result<DecodedImage> {
        if self.scan_count == 0 {
            return Err(Error::new("JPEG ended before its first scan"));
        }
        let mut frame = self
            .frame
            .take()
            .ok_or_else(|| Error::new("JPEG ended before defining a frame"))?;
        if frame.process == CodingProcess::Progressive {
            for component_index in 0..frame.components.len() {
                if self.coefficient_bits[component_index][0].is_none() {
                    return Err(Error::new("progressive JPEG is missing a DC scan"));
                }
            }
            frame.materialize_progressive(&self.quantization_tables)?;
        }
        frame.into_image(self.color_transform)
    }
}

#[derive(Clone, Copy)]
enum ColorTransform {
    Ycbcr,
    Rgb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodingProcess {
    Baseline,
    Progressive,
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
    process: CodingProcess,
}

impl Frame {
    fn new(
        width: u32,
        height: u32,
        mut components: Vec<FrameComponent>,
        process: CodingProcess,
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
            process,
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
            process,
        })
    }

    fn materialize_progressive(
        &mut self,
        quantization_tables: &[Option<[u16; 64]>; QUANTIZATION_TABLES_MAX],
    ) -> Result<()> {
        assert_eq!(self.process, CodingProcess::Progressive);
        assert!(!self.components.is_empty());

        for component in &mut self.components {
            let quantization = quantization_tables[component.quantization_table]
                .ok_or_else(|| Error::new("frame references a missing quantization table"))?;
            let block_count = component
                .block_columns
                .checked_mul(component.block_rows)
                .ok_or_else(|| Error::new("component block count overflowed"))?;
            if component.coefficients.len() != block_count as usize {
                return Err(Error::new(
                    "progressive coefficient plane has an invalid size",
                ));
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
            .ok_or_else(|| Error::new("dequantized progressive coefficient overflowed"))?;
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
    dc: Option<HuffmanTable>,
    ac: Option<HuffmanTable>,
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
            return Err(Error::new("frame contains duplicate component identifiers"));
        }
        let sampling = segment.read_u8()?;
        let horizontal_sampling = sampling >> 4;
        let vertical_sampling = sampling & 0x0f;
        if !(1..=4).contains(&horizontal_sampling) || !(1..=4).contains(&vertical_sampling) {
            return Err(Error::new(
                "component sampling factor is outside 1 through 4",
            ));
        }
        blocks_per_mcu += u32::from(horizontal_sampling) * u32::from(vertical_sampling);
        let quantization_table = usize::from(segment.read_u8()?);
        if quantization_table >= QUANTIZATION_TABLES_MAX {
            return Err(Error::new(
                "component quantization table index is out of range",
            ));
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
        return Err(Error::new("frame has more than ten data units per MCU"));
    }
    Ok(components)
}

fn parse_scan_header(segment: &mut Reader<'_>, frame: Option<&Frame>) -> Result<ScanHeader> {
    assert!(frame.is_none_or(|frame| frame.components.len() <= COMPONENTS_MAX));

    let frame = frame.ok_or_else(|| Error::new("SOS marker appeared before SOF0"))?;
    let component_count = usize::from(segment.read_u8()?);
    if component_count == 0 || component_count > frame.components.len() {
        return Err(Error::new("scan component count is out of range"));
    }
    let mut components = Vec::with_capacity(component_count);
    for _ in 0..component_count {
        let identifier = segment.read_u8()?;
        let frame_index = frame
            .components
            .iter()
            .position(|component| component.identifier == identifier)
            .ok_or_else(|| Error::new("scan references an unknown frame component"))?;
        if components
            .iter()
            .any(|component: &ScanComponent| component.frame_index == frame_index)
        {
            return Err(Error::new("scan contains a duplicate component"));
        }
        let selectors = segment.read_u8()?;
        let dc_table = usize::from(selectors >> 4);
        let ac_table = usize::from(selectors & 0x0f);
        if dc_table >= HUFFMAN_TABLES_MAX || ac_table >= HUFFMAN_TABLES_MAX {
            return Err(Error::new("scan Huffman table selector is out of range"));
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
        return Err(Error::new("SOS segment has trailing bytes"));
    }
    let header = ScanHeader {
        components,
        spectral_start,
        spectral_end,
        successive_high: approximation >> 4,
        successive_low: approximation & 0x0f,
    };
    validate_scan_header(&header, frame.process, frame.components.len())?;
    Ok(header)
}

fn validate_scan_header(
    scan: &ScanHeader,
    process: CodingProcess,
    frame_component_count: usize,
) -> Result<()> {
    assert!(!scan.components.is_empty());
    assert!(scan.components.len() <= COMPONENTS_MAX);

    if process == CodingProcess::Baseline {
        let is_full_scan = scan.components.len() == frame_component_count;
        let is_sequential = scan.spectral_start == 0
            && scan.spectral_end == 63
            && scan.successive_high == 0
            && scan.successive_low == 0;
        if !is_full_scan || !is_sequential {
            return Err(Error::new(
                "baseline JPEG requires one full interleaved scan",
            ));
        }
        return Ok(());
    }
    if scan.spectral_start > scan.spectral_end || scan.spectral_end >= 64 {
        return Err(Error::new("progressive spectral selection is out of range"));
    }
    if scan.spectral_start == 0 && scan.spectral_end != 0 {
        return Err(Error::new(
            "a progressive DC scan must end at coefficient zero",
        ));
    }
    if scan.spectral_start > 0 && scan.components.len() != 1 {
        return Err(Error::new(
            "a progressive AC scan must contain one component",
        ));
    }
    if scan.successive_high > 0 && scan.successive_low + 1 != scan.successive_high {
        return Err(Error::new(
            "progressive refinement must advance exactly one bit",
        ));
    }
    if scan.successive_low > 13 {
        return Err(Error::new("progressive approximation bit exceeds 13"));
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
        .ok_or_else(|| Error::new("MCU count overflowed"))?;
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
        return Err(Error::new(
            "baseline DC coefficient category exceeds 11 bits",
        ));
    }
    let dc_difference = receive_extend(reader, dc_category)?;
    *dc_predictor = dc_predictor
        .checked_add(dc_difference)
        .ok_or_else(|| Error::new("DC predictor overflowed"))?;
    coefficients[0] = dc_predictor
        .checked_mul(i32::from(plan.quantization[0]))
        .ok_or_else(|| Error::new("dequantized DC coefficient overflowed"))?;

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
                return Err(Error::new("invalid zero-size AC Huffman symbol"));
            }
            zigzag_index += 16;
            continue;
        }
        if category > 10 {
            return Err(Error::new(
                "baseline AC coefficient category exceeds 10 bits",
            ));
        }
        zigzag_index += zero_run;
        if zigzag_index >= 64 {
            return Err(Error::new("AC coefficient run extends past its block"));
        }
        let natural_index = ZIGZAG_TO_NATURAL[zigzag_index];
        let value = receive_extend(reader, category)?;
        coefficients[natural_index] = value
            .checked_mul(i32::from(plan.quantization[natural_index]))
            .ok_or_else(|| Error::new("dequantized AC coefficient overflowed"))?;
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

    if layout.process == CodingProcess::Baseline {
        return Ok(());
    }
    let mut byte_count = 0_u64;
    for component in components {
        let blocks = u64::from(layout.mcu_columns)
            * u64::from(component.horizontal_sampling)
            * u64::from(layout.mcu_rows)
            * u64::from(component.vertical_sampling);
        let component_bytes = blocks * 64 * size_of::<i32>() as u64;
        byte_count = byte_count
            .checked_add(component_bytes)
            .ok_or_else(|| Error::new("progressive coefficient storage overflowed"))?;
    }
    if byte_count > PROGRESSIVE_COEFFICIENT_BYTES_MAX {
        return Err(Error::new(
            "progressive coefficient storage exceeds the 512 MiB limit",
        ));
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
        .ok_or_else(|| Error::new("component block width overflowed"))?;
    let block_rows = layout
        .mcu_rows
        .checked_mul(u32::from(component.vertical_sampling))
        .ok_or_else(|| Error::new("component block height overflowed"))?;
    let plane_width = block_columns
        .checked_mul(BLOCK_SIDE)
        .ok_or_else(|| Error::new("component plane width overflowed"))?;
    let plane_height = block_rows
        .checked_mul(BLOCK_SIDE)
        .ok_or_else(|| Error::new("component plane height overflowed"))?;
    let sample_count = plane_width
        .checked_mul(plane_height)
        .ok_or_else(|| Error::new("component plane size overflowed"))?;
    let data_width = layout
        .width
        .checked_mul(u32::from(component.horizontal_sampling))
        .ok_or_else(|| Error::new("component data width overflowed"))?;
    let data_height = layout
        .height
        .checked_mul(u32::from(component.vertical_sampling))
        .ok_or_else(|| Error::new("component data height overflowed"))?;
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
        let block_count = block_columns
            .checked_mul(block_rows)
            .ok_or_else(|| Error::new("component coefficient count overflowed"))?;
        component.coefficients = vec![[0; 64]; block_count as usize];
    }
    assert_eq!(component.plane.len(), sample_count as usize);
    Ok(())
}

fn convert_color(first: u8, second: u8, third: u8, transform: ColorTransform) -> [u8; 4] {
    match transform {
        ColorTransform::Rgb => [first, second, third, 255],
        ColorTransform::Ycbcr => {
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
        return Err(Error::new("JPEG dimensions must be nonzero"));
    }
    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(Error::new(format!(
            "JPEG dimensions exceed the {DIMENSION_MAX}-pixel limit"
        )));
    }
    if u64::from(width) * u64::from(height) > PIXELS_MAX {
        return Err(Error::new(
            "JPEG pixel count exceeds the 64-megapixel limit",
        ));
    }
    Ok(())
}

fn validate_huffman_symbols(class: u8, symbols: &[u8]) -> Result<()> {
    assert!(class <= 1);
    assert!(symbols.len() <= 256);

    for symbol in symbols {
        if class == 0 && *symbol > 11 {
            return Err(Error::new("DC Huffman table contains a category above 11"));
        }
        if class == 1 {
            let category = symbol & 0x0f;
            if category > 10 {
                return Err(Error::new("AC Huffman table contains an invalid symbol"));
            }
        }
    }
    Ok(())
}

fn unsupported_frame_error(marker: u8) -> Error {
    assert!((0xc1..=0xcf).contains(&marker));
    assert_ne!(marker, MARKER_DHT);

    Error::new(format!("JPEG frame type FF{marker:02X} is not supported"))
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
