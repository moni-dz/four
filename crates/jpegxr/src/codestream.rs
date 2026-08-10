//! Parses JPEG XR image and image-plane headers.

use crate::bitstream::BitReader;
use crate::container::Codestream;
use crate::error::{ErrorKind, Result};

const CODESTREAM_SIGNATURE: u64 = 0x574D_5048_4F54_4F00;
const MAX_TILES: usize = 4_096;

#[derive(Clone, Debug)]
pub(crate) struct ParsedCodestream<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) offset: usize,
    pub(crate) info: CodestreamInfo,
    pub(crate) header: ImageHeader,
    pub(crate) primary_plane: PlaneHeader,
    pub(crate) alpha_plane: Option<PlaneHeader>,
    pub(crate) index_offsets: Vec<u64>,
    pub(crate) tiles_offset: usize,
}

/// Parsed properties of one JPEG XR codestream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodestreamInfo {
    width: u32,
    height: u32,
    output_color_format: OutputColorFormat,
    output_bit_depth: OutputBitDepth,
    internal_color_format: InternalColorFormat,
    bands: Bands,
    tile_columns: u16,
    tile_rows: u16,
    frequency_mode: bool,
    overlap_mode: OverlapMode,
    alpha_plane: bool,
}

impl CodestreamInfo {
    /// Returns the output width before orientation is applied.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the output height before orientation is applied.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the output color format.
    #[must_use]
    pub const fn output_color_format(&self) -> OutputColorFormat {
        self.output_color_format
    }

    /// Returns the output sample representation.
    #[must_use]
    pub const fn output_bit_depth(&self) -> OutputBitDepth {
        self.output_bit_depth
    }

    /// Returns the transform's internal color format.
    #[must_use]
    pub const fn internal_color_format(&self) -> InternalColorFormat {
        self.internal_color_format
    }

    /// Returns the frequency bands present in the codestream.
    #[must_use]
    pub const fn bands(&self) -> Bands {
        self.bands
    }

    /// Returns the number of tile columns.
    #[must_use]
    pub const fn tile_columns(&self) -> u16 {
        self.tile_columns
    }

    /// Returns the number of tile rows.
    #[must_use]
    pub const fn tile_rows(&self) -> u16 {
        self.tile_rows
    }

    /// Returns whether frequency-mode packet layout is used.
    #[must_use]
    pub const fn is_frequency_mode(&self) -> bool {
        self.frequency_mode
    }

    /// Returns the overlap-filtering mode.
    #[must_use]
    pub const fn overlap_mode(&self) -> OverlapMode {
        self.overlap_mode
    }

    /// Returns whether an interleaved alpha plane follows the primary plane.
    #[must_use]
    pub const fn has_interleaved_alpha(&self) -> bool {
        self.alpha_plane
    }
}

/// Output color format from T.832 Table 22.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputColorFormat {
    /// Luminance only.
    YOnly,
    /// YUV with 4:2:0 chroma subsampling.
    YUV420,
    /// YUV with 4:2:2 chroma subsampling.
    YUV422,
    /// Full-resolution YUV.
    YUV444,
    /// Subtractive CMYK.
    CMYK,
    /// Direct CMYK.
    CMYKDirect,
    /// Arbitrary component count.
    NComponent,
    /// Red, green, and blue.
    RGB,
    /// RGB with a shared exponent.
    RGBE,
}

impl OutputColorFormat {
    fn parse(value: u8, reader: &BitReader<'_>) -> Result<Self> {
        match value {
            0 => Ok(Self::YOnly),
            1 => Ok(Self::YUV420),
            2 => Ok(Self::YUV422),
            3 => Ok(Self::YUV444),
            4 => Ok(Self::CMYK),
            5 => Ok(Self::CMYKDirect),
            6 => Ok(Self::NComponent),
            7 => Ok(Self::RGB),
            8 => Ok(Self::RGBE),
            _ => Err(reader.error(ErrorKind::InvalidCodestream("reserved output color format"))),
        }
    }
}

/// Output sample representation from T.832 Table 23.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputBitDepth {
    /// One bit where one represents white.
    OneWhite,
    /// Unsigned eight-bit samples.
    Eight,
    /// Unsigned 16-bit samples.
    Sixteen,
    /// Signed 16-bit samples.
    SixteenSigned,
    /// IEEE 754 binary16 samples.
    SixteenFloat,
    /// Signed 32-bit samples.
    ThirtyTwoSigned,
    /// IEEE 754 binary32 samples.
    ThirtyTwoFloat,
    /// Packed five-bit components.
    Five,
    /// Packed 10-bit components.
    Ten,
    /// Packed 5:6:5 components.
    FiveSixFive,
    /// One bit where one represents black.
    OneBlack,
}

impl OutputBitDepth {
    fn parse(value: u8, reader: &BitReader<'_>) -> Result<Self> {
        match value {
            0 => Ok(Self::OneWhite),
            1 => Ok(Self::Eight),
            2 => Ok(Self::Sixteen),
            3 => Ok(Self::SixteenSigned),
            4 => Ok(Self::SixteenFloat),
            6 => Ok(Self::ThirtyTwoSigned),
            7 => Ok(Self::ThirtyTwoFloat),
            8 => Ok(Self::Five),
            9 => Ok(Self::Ten),
            10 => Ok(Self::FiveSixFive),
            15 => Ok(Self::OneBlack),
            _ => Err(reader.error(ErrorKind::InvalidCodestream("reserved output bit depth"))),
        }
    }
}

/// Internal transform color format from T.832 Table 28.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InternalColorFormat {
    /// Luminance only.
    YOnly,
    /// YUV with 4:2:0 chroma subsampling.
    YUV420,
    /// YUV with 4:2:2 chroma subsampling.
    YUV422,
    /// Full-resolution YUV.
    YUV444,
    /// YUV plus a black component.
    YUVK,
    /// Arbitrary component count.
    NComponent,
}

impl InternalColorFormat {
    fn parse(value: u8, reader: &BitReader<'_>) -> Result<Self> {
        match value {
            0 => Ok(Self::YOnly),
            1 => Ok(Self::YUV420),
            2 => Ok(Self::YUV422),
            3 => Ok(Self::YUV444),
            4 => Ok(Self::YUVK),
            6 => Ok(Self::NComponent),
            _ => Err(reader.error(ErrorKind::InvalidCodestream(
                "reserved internal color format",
            ))),
        }
    }

    const fn component_count(self) -> Option<u16> {
        match self {
            Self::YOnly => Some(1),
            Self::YUV420 | Self::YUV422 | Self::YUV444 => Some(3),
            Self::YUVK => Some(4),
            Self::NComponent => None,
        }
    }
}

/// Frequency-band presence from T.832 Table 29.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bands {
    /// DC, lowpass, highpass, and flexbits are present.
    All,
    /// Flexbits are absent.
    NoFlexbits,
    /// Highpass and flexbits are absent.
    NoHighpass,
    /// Only DC is present.
    DCOnly,
}

impl Bands {
    fn parse(value: u8, reader: &BitReader<'_>) -> Result<Self> {
        match value {
            0 => Ok(Self::All),
            1 => Ok(Self::NoFlexbits),
            2 => Ok(Self::NoHighpass),
            3 => Ok(Self::DCOnly),
            _ => Err(reader.error(ErrorKind::InvalidCodestream("reserved band-presence value"))),
        }
    }

    pub(crate) const fn count(self) -> usize {
        match self {
            Self::All => 4,
            Self::NoFlexbits => 3,
            Self::NoHighpass => 2,
            Self::DCOnly => 1,
        }
    }
}

/// Overlap-filtering mode from T.832 clause 8.3.10.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlapMode {
    /// No overlap filtering.
    None,
    /// Second-level overlap filtering only.
    SecondLevel,
    /// First- and second-level overlap filtering.
    FirstAndSecondLevel,
}

impl OverlapMode {
    fn parse(value: u8, reader: &BitReader<'_>) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::SecondLevel),
            2 => Ok(Self::FirstAndSecondLevel),
            _ => Err(reader.error(ErrorKind::InvalidCodestream("reserved overlap mode"))),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ImageHeader {
    pub(crate) frequency_mode: bool,
    pub(crate) spatial_transform: u8,
    pub(crate) index_table_present: bool,
    pub(crate) overlap_mode: OverlapMode,
    pub(crate) trim_flexbits: bool,
    pub(crate) red_blue_swapped: bool,
    pub(crate) premultiplied_alpha: bool,
    pub(crate) alpha_plane: bool,
    pub(crate) output_color_format: OutputColorFormat,
    pub(crate) output_bit_depth: OutputBitDepth,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_widths: Vec<u16>,
    pub(crate) tile_heights: Vec<u16>,
    pub(crate) margins: Margins,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Margins {
    pub(crate) top: u8,
    pub(crate) left: u8,
    pub(crate) bottom: u8,
    pub(crate) right: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct PlaneHeader {
    pub(crate) internal_color_format: InternalColorFormat,
    pub(crate) scaled: bool,
    pub(crate) bands: Bands,
    pub(crate) component_count: u16,
    pub(crate) mantissa_bits: Option<u8>,
    pub(crate) exponent_bias: Option<i8>,
    pub(crate) dc_uniform: bool,
    pub(crate) dc_quantization: Option<QuantizationSet>,
    pub(crate) lowpass_uniform: bool,
    pub(crate) lowpass_quantization: Option<QuantizationSet>,
    pub(crate) highpass_uniform: bool,
    pub(crate) highpass_quantization: Option<QuantizationSet>,
}

#[derive(Clone, Debug)]
pub(crate) struct QuantizationSet {
    pub(crate) components: Vec<u8>,
}

pub(crate) fn parse(codestream: Codestream<'_>) -> Result<ParsedCodestream<'_>> {
    let mut reader = BitReader::new(codestream.bytes, codestream.offset);

    let header = parse_image_header(&mut reader)?;
    let primary_plane = parse_plane_header(&mut reader, &header)?;

    let alpha_plane = if header.alpha_plane {
        let plane = parse_plane_header(&mut reader, &header)?;

        if plane.internal_color_format != InternalColorFormat::YOnly {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "alpha plane must use YONLY internally",
            )));
        }

        if plane.bands.count() > primary_plane.bands.count() {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "alpha plane has more bands than primary plane",
            )));
        }
        Some(plane)
    } else {
        None
    };

    let tile_count = header
        .tile_widths
        .len()
        .checked_mul(header.tile_heights.len())
        .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("tile count")))?;

    let index_count = if header.frequency_mode {
        tile_count
            .checked_mul(primary_plane.bands.count())
            .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("index table")))?
    } else {
        tile_count
    };
    let index_offsets = if header.index_table_present {
        let start_code = reader.read_u16(16)?;

        if start_code != 1 {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "index table start code must be 1",
            )));
        }

        (0..index_count)
            .map(|_| read_vlw(&mut reader))
            .collect::<Result<Vec<_>>>()?
    } else {
        if header.frequency_mode || tile_count > 1 {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "frequency or tiled codestream requires an index table",
            )));
        }
        Vec::new()
    };

    let subsequent_bytes = read_vlw(&mut reader)?;

    if subsequent_bytes != 0 && subsequent_bytes < 4 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "profile-level block is shorter than four bytes",
        )));
    }

    let subsequent_bytes = usize::try_from(subsequent_bytes).map_err(|_conversion_error| {
        reader.error(ErrorKind::LimitExceeded("profile-level block"))
    })?;

    if subsequent_bytes > 0 {
        let start = reader.byte_position();

        loop {
            let _profile = reader.read_u8(8)?;
            let _level = reader.read_u8(8)?;
            let _reserved = reader.read(15)?;
            let last = reader.read_bool()?;

            if last {
                break;
            }

            if reader.byte_position().saturating_sub(start) >= subsequent_bytes {
                return Err(reader.error(ErrorKind::InvalidCodestream(
                    "profile-level block has no last entry",
                )));
            }
        }

        let consumed = reader.byte_position().saturating_sub(start);

        if consumed > subsequent_bytes {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "profile-level entries exceed declared size",
            )));
        }

        for _ in consumed..subsequent_bytes {
            let _reserved = reader.read_u8(8)?;
        }
    }

    let tiles_offset = reader.byte_position();
    let info = CodestreamInfo {
        width: header.width,
        height: header.height,
        output_color_format: header.output_color_format,
        output_bit_depth: header.output_bit_depth,
        internal_color_format: primary_plane.internal_color_format,
        bands: primary_plane.bands,
        tile_columns: u16::try_from(header.tile_widths.len())
            .expect("tile-column count is limited to 4096"),
        tile_rows: u16::try_from(header.tile_heights.len())
            .expect("tile-row count is limited to 4096"),
        frequency_mode: header.frequency_mode,
        overlap_mode: header.overlap_mode,
        alpha_plane: header.alpha_plane,
    };

    Ok(ParsedCodestream {
        bytes: codestream.bytes,
        offset: codestream.offset,
        info,
        header,
        primary_plane,
        alpha_plane,
        index_offsets,
        tiles_offset,
    })
}

fn parse_image_header(reader: &mut BitReader<'_>) -> Result<ImageHeader> {
    if reader.read(64)? != CODESTREAM_SIGNATURE {
        return Err(reader.error(ErrorKind::InvalidSignature));
    }

    if reader.read_u8(4)? != 1 {
        return Err(reader.error(ErrorKind::InvalidCodestream("RESERVED_B must equal 1")));
    }

    let _hard_tiling = reader.read_bool()?;
    let _reserved_c = reader.read_u8(3)?;

    let tiling = reader.read_bool()?;
    let frequency_mode = reader.read_bool()?;
    let spatial_transform = reader.read_u8(3)?;
    let index_table_present = reader.read_bool()?;
    let overlap_mode = OverlapMode::parse(reader.read_u8(2)?, reader)?;

    let short_header = reader.read_bool()?;
    let _long_word = reader.read_bool()?;
    let windowing = reader.read_bool()?;
    let trim_flexbits = reader.read_bool()?;

    let _reserved_d = reader.read_bool()?;
    let red_blue_swapped = reader.read_bool()?;
    let premultiplied_alpha = reader.read_bool()?;
    let alpha_plane = reader.read_bool()?;

    let output_color_format = OutputColorFormat::parse(reader.read_u8(4)?, reader)?;
    let output_bit_depth = OutputBitDepth::parse(reader.read_u8(4)?, reader)?;

    let dimension_bits = if short_header { 16 } else { 32 };
    let width = reader
        .read_u32(dimension_bits)?
        .checked_add(1)
        .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("image width")))?;
    let height = reader
        .read_u32(dimension_bits)?
        .checked_add(1)
        .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("image height")))?;

    let (tile_columns, tile_rows) = if tiling {
        (
            usize::from(reader.read_u16(12)?) + 1,
            usize::from(reader.read_u16(12)?) + 1,
        )
    } else {
        (1, 1)
    };
    let tile_count = tile_columns
        .checked_mul(tile_rows)
        .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("tile count")))?;

    if tile_count > MAX_TILES {
        return Err(reader.error(ErrorKind::LimitExceeded("tile count")));
    }

    let tile_size_bits = if short_header { 8 } else { 16 };
    let transmitted_widths = (0..tile_columns.saturating_sub(1))
        .map(|_| reader.read_u16(tile_size_bits))
        .collect::<Result<Vec<_>>>()?;
    let transmitted_heights = (0..tile_rows.saturating_sub(1))
        .map(|_| reader.read_u16(tile_size_bits))
        .collect::<Result<Vec<_>>>()?;

    let margins = if windowing {
        Margins {
            top: reader.read_u8(6)?,
            left: reader.read_u8(6)?,
            bottom: reader.read_u8(6)?,
            right: reader.read_u8(6)?,
        }
    } else {
        Margins {
            top: 0,
            left: 0,
            bottom: inferred_margin(height),
            right: inferred_margin(width),
        }
    };
    let extended_width = width
        .checked_add(u32::from(margins.left) + u32::from(margins.right))
        .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("extended image width")))?;
    let extended_height = height
        .checked_add(u32::from(margins.top) + u32::from(margins.bottom))
        .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("extended image height")))?;

    if !extended_width.is_multiple_of(16) || !extended_height.is_multiple_of(16) {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "extended dimensions must be multiples of 16",
        )));
    }

    let macroblock_width = u16::try_from(extended_width / 16)
        .map_err(|_conversion_error| reader.error(ErrorKind::LimitExceeded("macroblock width")))?;
    let macroblock_height = u16::try_from(extended_height / 16)
        .map_err(|_conversion_error| reader.error(ErrorKind::LimitExceeded("macroblock height")))?;
    let tile_widths = complete_tile_sizes(transmitted_widths, macroblock_width, reader)?;
    let tile_heights = complete_tile_sizes(transmitted_heights, macroblock_height, reader)?;

    Ok(ImageHeader {
        frequency_mode,
        spatial_transform,
        index_table_present,
        overlap_mode,
        trim_flexbits,
        red_blue_swapped,
        premultiplied_alpha,
        alpha_plane,
        output_color_format,
        output_bit_depth,
        width,
        height,
        tile_widths,
        tile_heights,
        margins,
    })
}

fn parse_plane_header(reader: &mut BitReader<'_>, header: &ImageHeader) -> Result<PlaneHeader> {
    let internal_color_format = InternalColorFormat::parse(reader.read_u8(3)?, reader)?;
    let scaled = reader.read_bool()?;
    let bands = Bands::parse(reader.read_u8(4)?, reader)?;

    if matches!(
        internal_color_format,
        InternalColorFormat::YUV420 | InternalColorFormat::YUV422 | InternalColorFormat::YUV444
    ) {
        if matches!(
            internal_color_format,
            InternalColorFormat::YUV420 | InternalColorFormat::YUV422
        ) {
            let _reserved = reader.read_bool()?;
            let _chroma_centering_x = reader.read_u8(3)?;
        } else {
            let _reserved = reader.read_u8(4)?;
        }

        if internal_color_format == InternalColorFormat::YUV420 {
            let _reserved = reader.read_bool()?;
            let _chroma_centering_y = reader.read_u8(3)?;
        } else {
            let _reserved = reader.read_u8(4)?;
        }
    }

    let component_count = if internal_color_format == InternalColorFormat::NComponent {
        let minus_one = reader.read_u8(4)?;
        if minus_one == 15 {
            reader
                .read_u16(12)?
                .checked_add(16)
                .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("component count")))?
        } else {
            let _reserved = reader.read_u8(4)?;
            u16::from(minus_one) + 1
        }
    } else {
        internal_color_format
            .component_count()
            .expect("only NCOMPONENT has a dynamic component count")
    };

    let _shift_bits = if matches!(
        header.output_bit_depth,
        OutputBitDepth::Sixteen | OutputBitDepth::SixteenSigned | OutputBitDepth::ThirtyTwoSigned
    ) {
        Some(reader.read_u8(8)?)
    } else {
        None
    };

    let (mantissa_bits, exponent_bias) =
        if header.output_bit_depth == OutputBitDepth::ThirtyTwoFloat {
            (
                Some(reader.read_u8(8)?),
                Some(i8::from_ne_bytes([reader.read_u8(8)?])),
            )
        } else {
            (None, None)
        };

    let dc_uniform = reader.read_bool()?;
    let dc_quantization = dc_uniform
        .then(|| parse_quantization(reader, component_count))
        .transpose()?;

    let mut lowpass_uniform = false;
    let mut lowpass_quantization = None;
    let mut highpass_uniform = false;
    let mut highpass_quantization = None;
    if bands != Bands::DCOnly {
        let _reserved_i = reader.read_bool()?;
        lowpass_uniform = reader.read_bool()?;

        if lowpass_uniform {
            lowpass_quantization = Some(parse_quantization(reader, component_count)?);
        }

        if bands != Bands::NoHighpass {
            let _reserved_j = reader.read_bool()?;
            highpass_uniform = reader.read_bool()?;

            if highpass_uniform {
                highpass_quantization = Some(parse_quantization(reader, component_count)?);
            }
        }
    }

    reader.align_zero()?;

    Ok(PlaneHeader {
        internal_color_format,
        scaled,
        bands,
        component_count,
        mantissa_bits,
        exponent_bias,
        dc_uniform,
        dc_quantization,
        lowpass_uniform,
        lowpass_quantization,
        highpass_uniform,
        highpass_quantization,
    })
}

fn parse_quantization(reader: &mut BitReader<'_>, component_count: u16) -> Result<QuantizationSet> {
    let mode = if component_count == 1 {
        0
    } else {
        reader.read_u8(2)?
    };

    let components = match mode {
        0 => vec![reader.read_u8(8)?; usize::from(component_count)],
        1 => {
            let luma = reader.read_u8(8)?;
            let chroma = reader.read_u8(8)?;
            let mut components = vec![chroma; usize::from(component_count)];
            components[0] = luma;
            components
        }
        2 => (0..component_count)
            .map(|_| reader.read_u8(8))
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "reserved quantization component mode",
            )));
        }
    };

    Ok(QuantizationSet { components })
}

fn complete_tile_sizes(
    mut transmitted: Vec<u16>,
    total: u16,
    reader: &BitReader<'_>,
) -> Result<Vec<u16>> {
    if transmitted.contains(&0) {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "tile dimensions must be positive",
        )));
    }

    let consumed = transmitted.iter().try_fold(0_u16, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| reader.error(ErrorKind::LimitExceeded("tile dimensions")))
    })?;

    let final_size = total.checked_sub(consumed).ok_or_else(|| {
        reader.error(ErrorKind::InvalidCodestream(
            "tile dimensions exceed extended image",
        ))
    })?;

    if final_size == 0 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "final tile dimension must be positive",
        )));
    }

    transmitted.push(final_size);

    Ok(transmitted)
}

const fn inferred_margin(dimension: u32) -> u8 {
    ((16 - dimension % 16) % 16) as u8
}

fn read_vlw(reader: &mut BitReader<'_>) -> Result<u64> {
    let first = reader.read_u8(8)?;

    match first {
        0x00..=0xFA => Ok(u64::from(first) * 256 + reader.read(8)?),
        0xFB => reader.read(32),
        0xFC => reader.read(64),
        0xFD..=0xFF => Ok(0),
    }
}
