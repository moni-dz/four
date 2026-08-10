//! Parses the T.832 Annex A tag-based file format.

use std::fmt;

use zerocopy::{
    FromBytes,
    byteorder::little_endian::{U16, U32},
};

use crate::codestream::{OutputBitDepth, OutputColorFormat};
use crate::error::{Error, ErrorKind, Result};

pub(crate) const FILE_SIGNATURE: [u8; 4] = [0x49, 0x49, 0xBC, 0x01];
const MAX_IFD_ENTRIES: usize = 4_096;
const PIXEL_FORMAT_PREFIX: [u8; 15] = [
    0x24, 0xC3, 0xDD, 0x6F, 0x03, 0x4E, 0xFE, 0x4B, 0xB1, 0x85, 0x3D, 0x77, 0x76, 0x8D, 0xC9,
];

const TAG_PIXEL_FORMAT: u16 = 0xBC01;
const TAG_SPATIAL_TRANSFORM: u16 = 0xBC02;
const TAG_IMAGE_WIDTH: u16 = 0xBC80;
const TAG_IMAGE_HEIGHT: u16 = 0xBC81;
const TAG_WIDTH_RESOLUTION: u16 = 0xBC82;
const TAG_HEIGHT_RESOLUTION: u16 = 0xBC83;
const TAG_IMAGE_OFFSET: u16 = 0xBCC0;
const TAG_IMAGE_BYTE_COUNT: u16 = 0xBCC1;
const TAG_ALPHA_OFFSET: u16 = 0xBCC2;
const TAG_ALPHA_BYTE_COUNT: u16 = 0xBCC3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Container<'a> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixel_format: PixelFormat,
    pub(crate) spatial_transform: u8,
    pub(crate) resolution: Option<Resolution>,
    pub(crate) primary: Codestream<'a>,
    pub(crate) alpha: Option<Codestream<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Codestream<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) offset: usize,
}

/// A T.832 Annex A pixel-format identifier.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct PixelFormat {
    bytes: [u8; 16],
    code: u8,
}

impl PixelFormat {
    /// Packed 10-bit BGR with two padding bits.
    pub const BGR101010: Self = Self::from_code(0x14);
    /// Four 32-bit floating-point RGBA channels.
    pub const RGBA128_FLOAT: Self = Self::from_code(0x19);
    /// Four premultiplied 32-bit floating-point RGBA channels.
    pub const PRGBA128_FLOAT: Self = Self::from_code(0x1A);

    const fn from_code(code: u8) -> Self {
        let mut bytes = [0_u8; 16];
        let mut index = 0;
        while index < PIXEL_FORMAT_PREFIX.len() {
            bytes[index] = PIXEL_FORMAT_PREFIX[index];
            index += 1;
        }
        bytes[15] = code;
        Self { bytes, code }
    }

    #[cfg(test)]
    pub(crate) const fn from_test_code(code: u8) -> [u8; 16] {
        Self::from_code(code).bytes
    }

    fn parse(bytes: [u8; 16], offset: usize) -> Result<Self> {
        let code = bytes[15];
        let defined = matches!(code, 0x05 | 0x08..=0x3B | 0x3D..=0x56);
        if bytes[..15] != PIXEL_FORMAT_PREFIX || !defined {
            return Err(Error::new(ErrorKind::UnsupportedPixelFormat(bytes), offset));
        }
        Ok(Self { bytes, code })
    }

    /// Returns the 16 bytes stored in the file.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 16] {
        self.bytes
    }

    /// Returns the mnemonic from T.832 Table A.6.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.code {
            0x05 => "BlackWhite",
            0x08 => "8bppGray",
            0x09 => "16bppBGR555",
            0x0A => "16bppBGR565",
            0x0B => "16bppGray",
            0x0C => "24bppBGR",
            0x0D => "24bppRGB",
            0x0E => "32bppBGR",
            0x0F => "32bppBGRA",
            0x10 => "32bppPBGRA",
            0x11 => "32bppGrayFloat",
            0x12 => "48bppRGBFixedPoint",
            0x13 => "16bppGrayFixedPoint",
            0x14 => "32bppBGR101010",
            0x15 => "48bppRGB",
            0x16 => "64bppRGBA",
            0x17 => "64bppPRGBA",
            0x18 => "96bppRGBFixedPoint",
            0x19 => "128bppRGBAFloat",
            0x1A => "128bppPRGBAFloat",
            0x1B => "128bppRGBFloat",
            0x1C => "32bppCMYK",
            0x1D => "64bppRGBAFixedPoint",
            0x1E => "128bppRGBAFixedPoint",
            0x1F => "64bppCMYK",
            0x20..=0x25 => "8-bit N-component",
            0x26..=0x2B => "16-bit N-component",
            0x2C => "40bppCMYKAlpha",
            0x2D => "80bppCMYKAlpha",
            0x2E..=0x33 => "8-bit N-component alpha",
            0x34..=0x39 => "16-bit N-component alpha",
            0x3A => "64bppRGBAHalf",
            0x3B => "48bppRGBHalf",
            0x3D => "32bppRGBE",
            0x3E => "16bppGrayHalf",
            0x3F => "32bppGrayFixedPoint",
            0x40 => "64bppRGBFixedPoint",
            0x41 => "128bppRGBFixedPoint",
            0x42 => "64bppRGBHalf",
            0x43 => "80bppCMYKDIRECTAlpha",
            0x44 => "12bppYCC420",
            0x45 => "16bppYCC422",
            0x46 => "20bppYCC422",
            0x47 => "32bppYCC422",
            0x48 => "24bppYCC444",
            0x49 => "30bppYCC444",
            0x4A => "48bppYCC444",
            0x4B => "48bppYCC444FixedPoint",
            0x4C => "20bppYCC420Alpha",
            0x4D => "24bppYCC422Alpha",
            0x4E => "30bppYCC422Alpha",
            0x4F => "48bppYCC422Alpha",
            0x50 => "32bppYCC444Alpha",
            0x51 => "40bppYCC444Alpha",
            0x52 => "64bppYCC444Alpha",
            0x53 => "64bppYCC444AlphaFixedPoint",
            0x54 => "32bppCMYKDIRECT",
            0x55 => "64bppCMYKDIRECT",
            0x56 => "40bppCMYKDIRECTAlpha",
            _ => "unknown",
        }
    }

    /// Returns whether the format includes alpha.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(
            self.code,
            0x0F | 0x10
                | 0x16
                | 0x17
                | 0x19
                | 0x1A
                | 0x1D
                | 0x1E
                | 0x2C..=0x3A
                | 0x43
                | 0x4C..=0x53
                | 0x56
        )
    }

    /// Returns whether color channels are premultiplied by alpha.
    #[must_use]
    pub const fn is_premultiplied(self) -> bool {
        matches!(self.code, 0x10 | 0x17 | 0x1A)
    }

    pub(crate) const fn expected_color_format(self) -> OutputColorFormat {
        match self.code {
            0x05 | 0x08 | 0x0B | 0x11 | 0x13 | 0x3E | 0x3F => OutputColorFormat::YOnly,
            0x1C | 0x1F | 0x2C | 0x2D => OutputColorFormat::CMYK,
            0x20..=0x2B | 0x2E..=0x39 => OutputColorFormat::NComponent,
            0x3D => OutputColorFormat::RGBE,
            0x43 | 0x54..=0x56 => OutputColorFormat::CMYKDirect,
            0x44 | 0x4C => OutputColorFormat::YUV420,
            0x45..=0x47 | 0x4D..=0x4F => OutputColorFormat::YUV422,
            0x48..=0x4B | 0x50..=0x53 => OutputColorFormat::YUV444,
            _ => OutputColorFormat::RGB,
        }
    }

    pub(crate) const fn accepts_bit_depth(self, depth: OutputBitDepth) -> bool {
        match self.code {
            0x05 => matches!(depth, OutputBitDepth::OneWhite | OutputBitDepth::OneBlack),
            0x08
            | 0x0C..=0x10
            | 0x1C
            | 0x20..=0x25
            | 0x2C
            | 0x2E..=0x33
            | 0x3D
            | 0x44
            | 0x45
            | 0x48
            | 0x4C
            | 0x4D
            | 0x50
            | 0x54
            | 0x56 => {
                matches!(depth, OutputBitDepth::Eight)
            }
            0x09 => matches!(depth, OutputBitDepth::Five),
            0x0A => matches!(depth, OutputBitDepth::FiveSixFive),
            0x0B
            | 0x15..=0x17
            | 0x1F
            | 0x26..=0x2B
            | 0x2D
            | 0x34..=0x39
            | 0x47
            | 0x4A
            | 0x4F
            | 0x52
            | 0x55 => matches!(depth, OutputBitDepth::Sixteen),
            0x12 | 0x13 | 0x1D | 0x40 | 0x4B | 0x53 => {
                matches!(depth, OutputBitDepth::SixteenSigned)
            }
            0x14 | 0x46 | 0x49 | 0x4E | 0x51 => matches!(depth, OutputBitDepth::Ten),
            0x18 | 0x1E | 0x3F | 0x41 => matches!(depth, OutputBitDepth::ThirtyTwoSigned),
            0x11 | 0x19..=0x1B => matches!(depth, OutputBitDepth::ThirtyTwoFloat),
            0x3A | 0x3B | 0x3E | 0x42 => matches!(depth, OutputBitDepth::SixteenFloat),
            0x43 => matches!(depth, OutputBitDepth::Sixteen),
            _ => false,
        }
    }
}

impl fmt::Debug for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixelFormat")
            .field("name", &self.name())
            .field("code", &self.code)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Horizontal and vertical resolution in dots per inch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolution {
    /// Horizontal resolution.
    pub horizontal: Option<f32>,
    /// Vertical resolution.
    pub vertical: Option<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ElementType {
    Byte,
    UTF8,
    UShort,
    ULong,
    URational,
    SByte,
    Undefined,
    SShort,
    SLong,
    SRational,
    Float,
    Double,
}

impl ElementType {
    fn parse(value: u16, offset: usize) -> Result<Self> {
        let element_type = match value {
            1 => Self::Byte,
            2 => Self::UTF8,
            3 => Self::UShort,
            4 => Self::ULong,
            5 => Self::URational,
            6 => Self::SByte,
            7 => Self::Undefined,
            8 => Self::SShort,
            9 => Self::SLong,
            10 => Self::SRational,
            11 => Self::Float,
            12 => Self::Double,
            _ => return Err(Error::new(ErrorKind::InvalidElementType(value), offset)),
        };
        Ok(element_type)
    }

    const fn size(self) -> usize {
        match self {
            Self::Byte | Self::UTF8 | Self::SByte | Self::Undefined => 1,
            Self::UShort | Self::SShort => 2,
            Self::ULong | Self::SLong | Self::Float => 4,
            Self::URational | Self::SRational | Self::Double => 8,
        }
    }

    const fn is_unsigned_scalar(self) -> bool {
        matches!(self, Self::Byte | Self::UShort | Self::ULong)
    }
}

#[derive(Clone, Copy, Debug)]
struct Entry<'a> {
    tag: u16,
    element_type: ElementType,
    count: u32,
    data: &'a [u8],
    offset: usize,
}

impl Entry<'_> {
    fn require(&self, element_type: ElementType, count: u32) -> Result<()> {
        if self.element_type != element_type || self.count != count {
            return Err(Error::new(
                ErrorKind::InvalidTag(self.tag, "unexpected element type or count"),
                self.offset,
            ));
        }
        Ok(())
    }

    fn unsigned_scalar(&self) -> Result<u32> {
        if !self.element_type.is_unsigned_scalar() || self.count != 1 {
            return Err(Error::new(
                ErrorKind::InvalidTag(self.tag, "expected one BYTE, USHORT, or ULONG"),
                self.offset,
            ));
        }
        Ok(match self.element_type {
            ElementType::Byte => u32::from(self.data[0]),
            ElementType::UShort => u32::from(read_u16(self.data, 0)?),
            ElementType::ULong => read_u32(self.data, 0)?,
            _ => unreachable!("unsigned scalar type checked above"),
        })
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Container<'_>> {
    if bytes.get(..4) != Some(FILE_SIGNATURE.as_slice()) {
        return Err(Error::new(ErrorKind::InvalidSignature, 0));
    }

    let first_ifd = usize::try_from(read_u32(bytes, 4)?).map_err(|_conversion_error| {
        Error::new(ErrorKind::InvalidOffset("first image directory"), 4)
    })?;

    if first_ifd < 8 || !first_ifd.is_multiple_of(2) {
        return Err(Error::new(
            ErrorKind::InvalidOffset("first image directory"),
            4,
        ));
    }

    let entry_count = usize::from(read_u16(bytes, first_ifd)?);
    if entry_count == 0 {
        return Err(Error::new(
            ErrorKind::InvalidTag(0, "image directory must not be empty"),
            first_ifd,
        ));
    }

    if entry_count > MAX_IFD_ENTRIES {
        return Err(Error::new(ErrorKind::TooManyEntries, first_ifd));
    }

    let entries_start = first_ifd
        .checked_add(2)
        .ok_or_else(|| Error::new(ErrorKind::InvalidOffset("image directory"), first_ifd))?;
    let entries_bytes = entry_count
        .checked_mul(12)
        .ok_or_else(|| Error::new(ErrorKind::InvalidOffset("image directory"), first_ifd))?;
    let next_ifd_offset = entries_start
        .checked_add(entries_bytes)
        .ok_or_else(|| Error::new(ErrorKind::InvalidOffset("image directory"), first_ifd))?;
    let next_ifd = read_u32(bytes, next_ifd_offset)?;
    if next_ifd != 0
        && (!next_ifd.is_multiple_of(2) || usize::try_from(next_ifd).ok() >= Some(bytes.len()))
    {
        return Err(Error::new(
            ErrorKind::InvalidOffset("next image directory"),
            next_ifd_offset,
        ));
    }

    let mut previous_tag = None;
    let mut pixel_format = None;
    let mut width = None;
    let mut height = None;
    let mut spatial_transform = 0;
    let mut horizontal_resolution = None;
    let mut vertical_resolution = None;
    let mut image_offset = None;
    let mut image_byte_count = None;
    let mut alpha_offset = None;
    let mut alpha_byte_count = None;

    for index in 0..entry_count {
        let offset = entries_start + index * 12;
        let entry = parse_entry(bytes, offset)?;

        if previous_tag.is_some_and(|tag| entry.tag <= tag) {
            return Err(Error::new(ErrorKind::UnsortedTags, offset));
        }

        previous_tag = Some(entry.tag);

        match entry.tag {
            TAG_PIXEL_FORMAT => {
                entry.require(ElementType::Byte, 16)?;
                let value = <[u8; 16]>::try_from(entry.data).map_err(|_conversion_error| {
                    Error::new(ErrorKind::UnexpectedEof, entry.offset)
                })?;
                pixel_format = Some(PixelFormat::parse(value, entry.offset)?);
            }
            TAG_SPATIAL_TRANSFORM => {
                let value = entry.unsigned_scalar()?;
                spatial_transform = u8::try_from(value)
                    .ok()
                    .filter(|value| *value <= 7)
                    .unwrap_or(0);
            }
            TAG_IMAGE_WIDTH => width = Some(entry.unsigned_scalar()?),
            TAG_IMAGE_HEIGHT => height = Some(entry.unsigned_scalar()?),
            TAG_WIDTH_RESOLUTION => {
                entry.require(ElementType::Float, 1)?;
                horizontal_resolution = Some(f32::from_bits(read_u32(entry.data, 0)?));
            }
            TAG_HEIGHT_RESOLUTION => {
                entry.require(ElementType::Float, 1)?;
                vertical_resolution = Some(f32::from_bits(read_u32(entry.data, 0)?));
            }
            TAG_IMAGE_OFFSET => image_offset = Some(entry.unsigned_scalar()?),
            TAG_IMAGE_BYTE_COUNT => image_byte_count = Some(entry.unsigned_scalar()?),
            TAG_ALPHA_OFFSET => alpha_offset = Some(entry.unsigned_scalar()?),
            TAG_ALPHA_BYTE_COUNT => alpha_byte_count = Some(entry.unsigned_scalar()?),
            _ => {}
        }
    }

    let pixel_format = required(pixel_format, TAG_PIXEL_FORMAT, first_ifd)?;
    let width = required(width, TAG_IMAGE_WIDTH, first_ifd)?;
    let height = required(height, TAG_IMAGE_HEIGHT, first_ifd)?;

    if width == 0 || height == 0 {
        return Err(Error::new(
            ErrorKind::InvalidTag(TAG_IMAGE_WIDTH, "dimensions must be nonzero"),
            first_ifd,
        ));
    }

    let primary = codestream(
        bytes,
        required(image_offset, TAG_IMAGE_OFFSET, first_ifd)?,
        required(image_byte_count, TAG_IMAGE_BYTE_COUNT, first_ifd)?,
        "primary image",
        first_ifd,
    )?;

    let alpha = match (alpha_offset, alpha_byte_count) {
        (Some(offset), Some(count)) => {
            Some(codestream(bytes, offset, count, "alpha image", first_ifd)?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidTag(TAG_ALPHA_OFFSET, "alpha offset and byte count must coexist"),
                first_ifd,
            ));
        }
    };

    if pixel_format.has_alpha() != alpha.is_some() {
        // Interleaved alpha is also valid; the codestream header resolves that case later.
        if !pixel_format.has_alpha() {
            return Err(Error::new(
                ErrorKind::ContainerMismatch("unexpected separate alpha image"),
                first_ifd,
            ));
        }
    }

    if horizontal_resolution.is_some_and(|value| !value.is_finite())
        || vertical_resolution.is_some_and(|value| !value.is_finite())
    {
        return Err(Error::new(
            ErrorKind::InvalidTag(TAG_WIDTH_RESOLUTION, "resolution must be finite"),
            first_ifd,
        ));
    }

    let resolution =
        (horizontal_resolution.is_some() || vertical_resolution.is_some()).then_some(Resolution {
            horizontal: horizontal_resolution,
            vertical: vertical_resolution,
        });

    Ok(Container {
        width,
        height,
        pixel_format,
        spatial_transform,
        resolution,
        primary,
        alpha,
    })
}

fn parse_entry(bytes: &[u8], offset: usize) -> Result<Entry<'_>> {
    let tag = read_u16(bytes, offset)?;
    let element_type = ElementType::parse(read_u16(bytes, offset + 2)?, offset + 2)?;
    let count = read_u32(bytes, offset + 4)?;

    let size = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(element_type.size()))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("tag payload size"), offset))?;

    let data_offset = if size <= 4 {
        offset + 8
    } else {
        let value = usize::try_from(read_u32(bytes, offset + 8)?).map_err(|_conversion_error| {
            Error::new(ErrorKind::InvalidOffset("tag payload"), offset + 8)
        })?;
        if !value.is_multiple_of(2) {
            return Err(Error::new(
                ErrorKind::InvalidOffset("tag payload"),
                offset + 8,
            ));
        }
        value
    };

    let end = data_offset
        .checked_add(size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidOffset("tag payload"), offset + 8))?;
    let data = bytes
        .get(data_offset..end)
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, data_offset))?;

    Ok(Entry {
        tag,
        element_type,
        count,
        data,
        offset,
    })
}

fn codestream<'a>(
    bytes: &'a [u8],
    offset: u32,
    byte_count: u32,
    name: &'static str,
    error_offset: usize,
) -> Result<Codestream<'a>> {
    let offset = usize::try_from(offset)
        .map_err(|_conversion_error| Error::new(ErrorKind::InvalidOffset(name), error_offset))?;

    let byte_count = usize::try_from(byte_count)
        .map_err(|_conversion_error| Error::new(ErrorKind::LimitExceeded(name), error_offset))?;

    let end = offset
        .checked_add(byte_count)
        .ok_or_else(|| Error::new(ErrorKind::InvalidOffset(name), error_offset))?;
    let bytes = bytes
        .get(offset..end)
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, offset))?;

    Ok(Codestream { bytes, offset })
}

fn required<T>(value: Option<T>, tag: u16, offset: usize) -> Result<T> {
    value.ok_or_else(|| Error::new(ErrorKind::MissingTag(tag), offset))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, offset))?;

    Ok(U16::read_from_bytes(value)
        .expect("validated two-byte slice")
        .get())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, offset))?;

    Ok(U32::read_from_bytes(value)
        .expect("validated four-byte slice")
        .get())
}
