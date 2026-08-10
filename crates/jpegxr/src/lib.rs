//! Decodes JPEG XR images in safe Rust.
//!
//! [`Decoder`] validates the Annex A tag container and embedded T.832 codestream headers before
//! coefficient decoding begins. Input is borrowed, so inspection does not copy compressed data.
//! Pixel reconstruction supports the native Windows HDR screenshot profiles for packed
//! `BGR101010` and `RGBA128Float` pixels. Other valid profiles return a precise
//! [`ErrorKind::Unsupported`] error.

#![forbid(unsafe_code)]
#![feature(portable_simd)]
#![allow(
    clippy::comparison_chain,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::match_same_arms,
    clippy::same_functions_in_if_condition,
    clippy::struct_excessive_bools,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    reason = "private decoder code follows the T.832 tables and bit-reading order closely"
)]

mod bitstream;
mod codestream;
mod container;
mod decode;
mod entropy;
mod error;

use std::fmt;

pub use codestream::{
    Bands, CodestreamInfo, InternalColorFormat, OutputBitDepth, OutputColorFormat, OverlapMode,
};
pub use container::{PixelFormat, Resolution};
pub use error::{Error, ErrorKind, Result};

use codestream::ParsedCodestream;
use container::Container;

/// Four-byte signature of the T.832 Annex A file format.
pub const SIGNATURE: [u8; 4] = container::FILE_SIGNATURE;

/// A validated JPEG XR file ready for decoding.
pub struct Decoder<'a> {
    container: Container<'a>,
    primary: ParsedCodestream<'a>,
    alpha: Option<ParsedCodestream<'a>>,
    info: ImageInfo,
}

impl<'a> Decoder<'a> {
    /// Parses a JPEG XR file from `bytes`.
    ///
    /// Parsing validates container bounds, required tags, codestream headers, image-plane headers,
    /// and cross-layer metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when input is truncated, malformed, unsupported, or internally
    /// inconsistent.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let container = container::parse(bytes)?;
        let primary = codestream::parse(container.primary)?;

        validate_primary(&container, &primary)?;

        let alpha = container.alpha.map(codestream::parse).transpose()?;

        if let Some(alpha) = &alpha {
            validate_separate_alpha(&container, &primary, alpha)?;
        }

        let info = ImageInfo {
            width: container.width,
            height: container.height,
            pixel_format: container.pixel_format,
            orientation: Orientation::from_code(container.spatial_transform),
            resolution: container.resolution,
            primary: primary.info.clone(),
            alpha: alpha.as_ref().map(|stream| stream.info.clone()),
        };

        Ok(Self {
            container,
            primary,
            alpha,
            info,
        })
    }

    /// Returns validated image and codestream properties.
    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Decodes an `RGBA128Float` image into interleaved `f32` samples.
    ///
    /// The returned samples are ordered R, G, B, A for each pixel in row-major order.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when coefficient data is malformed or the image is outside the currently
    /// supported unscaled frequency-mode, YUV444, no-overlap `RGBA128Float` profile.
    pub fn decode_rgba_f32(&self) -> Result<RGBAF32Image> {
        if self.info.pixel_format != PixelFormat::RGBA128_FLOAT {
            return Err(Error::new(
                ErrorKind::Unsupported("pixel format other than 128bppRGBAFloat"),
                self.primary.offset,
            ));
        }

        if self.info.orientation != Orientation::Identity {
            return Err(Error::new(
                ErrorKind::Unsupported("non-identity container orientation"),
                self.primary.offset,
            ));
        }

        let alpha = self.alpha.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported("RGBA128Float without separate alpha"),
                self.primary.offset,
            )
        })?;

        let pixels = decode::decode_rgba_f32(&self.primary, alpha)?;

        Ok(RGBAF32Image {
            width: self.info.width,
            height: self.info.height,
            pixels,
        })
    }

    /// Decodes a `32bppBGR101010` image into packed native-endian words.
    ///
    /// Each word stores blue in bits 0–9, green in bits 10–19, and red in bits 20–29.
    /// The top two bits are zero.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when coefficient data is malformed or the image is outside the currently
    /// supported frequency-mode, YUV444, no-overlap `32bppBGR101010` profile.
    pub fn decode_bgr101010(&self) -> Result<BGR101010Image> {
        if self.info.pixel_format != PixelFormat::BGR101010 {
            return Err(Error::new(
                ErrorKind::Unsupported("pixel format other than 32bppBGR101010"),
                self.primary.offset,
            ));
        }

        if self.info.orientation != Orientation::Identity {
            return Err(Error::new(
                ErrorKind::Unsupported("non-identity container orientation"),
                self.primary.offset,
            ));
        }

        let pixels = decode::decode_bgr101010(&self.primary)?;

        Ok(BGR101010Image {
            width: self.info.width,
            height: self.info.height,
            pixels,
        })
    }
}

impl fmt::Debug for Decoder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decoder")
            .field("info", &self.info)
            .field("primary_bytes", &self.container.primary.bytes.len())
            .field(
                "alpha_bytes",
                &self.container.alpha.map(|stream| stream.bytes.len()),
            )
            .finish_non_exhaustive()
    }
}

/// A decoded image containing interleaved RGBA `f32` samples.
#[derive(Clone, Debug, PartialEq)]
pub struct RGBAF32Image {
    width: u32,
    height: u32,
    pixels: Vec<f32>,
}

impl RGBAF32Image {
    /// Returns the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns row-major samples in interleaved R, G, B, A order.
    #[must_use]
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    /// Consumes the image and returns its interleaved RGBA samples.
    #[must_use]
    pub fn into_pixels(self) -> Vec<f32> {
        self.pixels
    }
}

/// A decoded image containing packed 10-bit BGR pixels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BGR101010Image {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
}

impl BGR101010Image {
    /// Returns the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns row-major packed BGR pixels.
    #[must_use]
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    /// Consumes the image and returns its packed BGR pixels.
    #[must_use]
    pub fn into_pixels(self) -> Vec<u32> {
        self.pixels
    }
}

/// Validated properties of a JPEG XR image.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfo {
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    orientation: Orientation,
    resolution: Option<Resolution>,
    primary: CodestreamInfo,
    alpha: Option<CodestreamInfo>,
}

impl ImageInfo {
    /// Returns image width before orientation is applied.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns image height before orientation is applied.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the output pixel format.
    #[must_use]
    pub const fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Returns the preferred spatial transform.
    #[must_use]
    pub const fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Returns optional horizontal and vertical resolution.
    #[must_use]
    pub const fn resolution(&self) -> Option<Resolution> {
        self.resolution
    }

    /// Returns primary codestream properties.
    #[must_use]
    pub const fn primary(&self) -> &CodestreamInfo {
        &self.primary
    }

    /// Returns separate alpha codestream properties when present.
    #[must_use]
    pub const fn alpha(&self) -> Option<&CodestreamInfo> {
        self.alpha.as_ref()
    }
}

/// Preferred spatial transformation from T.832 Table 21.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Orientation {
    /// No transformation.
    Identity,
    /// Flip vertically.
    FlipVertical,
    /// Flip horizontally.
    FlipHorizontal,
    /// Rotate 180 degrees.
    Rotate180,
    /// Rotate 90 degrees clockwise.
    Rotate90,
    /// Rotate 90 degrees clockwise, then flip vertically.
    Rotate90FlipVertical,
    /// Rotate 90 degrees clockwise, then flip horizontally.
    Rotate90FlipHorizontal,
    /// Rotate 90 degrees clockwise, then flip both axes.
    Rotate90FlipBoth,
}

impl Orientation {
    const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::FlipVertical,
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::Rotate90,
            5 => Self::Rotate90FlipVertical,
            6 => Self::Rotate90FlipHorizontal,
            7 => Self::Rotate90FlipBoth,
            _ => Self::Identity,
        }
    }
}

fn validate_primary(container: &Container<'_>, primary: &ParsedCodestream<'_>) -> Result<()> {
    if primary.info.width() != container.width || primary.info.height() != container.height {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("primary dimensions"),
            container.primary.offset,
        ));
    }

    if primary.info.output_color_format() != container.pixel_format.expected_color_format() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("output color format"),
            container.primary.offset,
        ));
    }

    if !container
        .pixel_format
        .accepts_bit_depth(primary.info.output_bit_depth())
    {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("output bit depth"),
            container.primary.offset,
        ));
    }

    if container.alpha.is_none()
        && primary.info.has_interleaved_alpha() != container.pixel_format.has_alpha()
    {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("interleaved alpha presence"),
            container.primary.offset,
        ));
    }

    if container.alpha.is_some() && primary.info.has_interleaved_alpha() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("alpha is both separate and interleaved"),
            container.primary.offset,
        ));
    }

    if primary.header.premultiplied_alpha != container.pixel_format.is_premultiplied() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("premultiplied-alpha flag"),
            container.primary.offset,
        ));
    }

    Ok(())
}

fn validate_separate_alpha(
    container: &Container<'_>,
    primary: &ParsedCodestream<'_>,
    alpha: &ParsedCodestream<'_>,
) -> Result<()> {
    let alpha_offset = container
        .alpha
        .expect("separate-alpha validation requires an alpha codestream")
        .offset;

    if !container.pixel_format.has_alpha() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("pixel format has no alpha channel"),
            alpha_offset,
        ));
    }

    if alpha.info.width() != primary.info.width() || alpha.info.height() != primary.info.height() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("separate alpha dimensions"),
            alpha_offset,
        ));
    }

    if alpha.info.output_color_format() != OutputColorFormat::YOnly {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("separate alpha must be YONLY"),
            alpha_offset,
        ));
    }

    if alpha.info.has_interleaved_alpha() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("separate alpha contains another alpha plane"),
            alpha_offset,
        ));
    }

    if alpha.info.bands().count() > primary.info.bands().count() {
        return Err(Error::new(
            ErrorKind::ContainerMismatch("separate alpha has too many bands"),
            alpha_offset,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Decoder, Orientation, OutputBitDepth, OutputColorFormat, PixelFormat};

    #[test]
    fn parses_minimal_annex_a_headers() {
        let file = minimal_rgb_file();
        let decoder = Decoder::new(&file).expect("synthetic T.832 headers must parse");
        let info = decoder.info();

        assert_eq!((info.width(), info.height()), (16, 16));
        assert_eq!(info.orientation(), Orientation::Identity);
        assert_eq!(info.pixel_format().name(), "24bppRGB");
        assert_eq!(info.primary().output_color_format(), OutputColorFormat::RGB);
        assert_eq!(info.primary().output_bit_depth(), OutputBitDepth::Eight);
        assert!(info.alpha().is_none());
    }

    fn minimal_rgb_file() -> Vec<u8> {
        let codestream = minimal_rgb_codestream();
        let pixel_offset = 74_u32;
        let image_offset = pixel_offset + 16;
        let mut file = vec![0_u8; usize::try_from(image_offset).expect("offset fits usize")];

        file[..4].copy_from_slice(&super::SIGNATURE);
        file[4..8].copy_from_slice(&8_u32.to_le_bytes());
        file[8..10].copy_from_slice(&5_u16.to_le_bytes());
        write_entry(&mut file, 10, 0xBC01, 1, 16, pixel_offset);
        write_entry(&mut file, 22, 0xBC80, 4, 1, 16);
        write_entry(&mut file, 34, 0xBC81, 4, 1, 16);
        write_entry(&mut file, 46, 0xBCC0, 4, 1, image_offset);
        write_entry(
            &mut file,
            58,
            0xBCC1,
            4,
            1,
            u32::try_from(codestream.len()).expect("fixture length fits u32"),
        );

        file[74..90].copy_from_slice(&PixelFormat::from_test_code(0x0D));
        file.extend_from_slice(&codestream);

        file
    }

    fn write_entry(
        bytes: &mut [u8],
        offset: usize,
        tag: u16,
        element_type: u16,
        count: u32,
        value: u32,
    ) {
        bytes[offset..offset + 2].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 2..offset + 4].copy_from_slice(&element_type.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&count.to_le_bytes());
        bytes[offset + 8..offset + 12].copy_from_slice(&value.to_le_bytes());
    }

    fn minimal_rgb_codestream() -> Vec<u8> {
        let mut bits = TestBits::default();
        bits.push(0x574D_5048_4F54_4F00, 64);
        bits.push(1, 4); // RESERVED_B
        bits.push(0, 1); // hard tiling
        bits.push(1, 3); // RESERVED_C
        bits.push(0, 1); // tiling
        bits.push(0, 1); // spatial mode
        bits.push(0, 3); // transform
        bits.push(0, 1); // index
        bits.push(0, 2); // overlap
        bits.push(1, 1); // short header
        bits.push(0, 1); // long word
        bits.push(0, 1); // windowing
        bits.push(0, 1); // trim flexbits
        bits.push(0, 1); // RESERVED_D
        bits.push(0, 1); // red/blue flag
        bits.push(0, 1); // premultiplied
        bits.push(0, 1); // alpha plane
        bits.push(7, 4); // RGB
        bits.push(1, 4); // BD8
        bits.push(15, 16); // width minus one
        bits.push(15, 16); // height minus one
        bits.push(3, 3); // internal YUV444
        bits.push(0, 1); // scaled
        bits.push(3, 4); // DC only
        bits.push(0, 4); // RESERVED_F
        bits.push(0, 4); // RESERVED_H
        bits.push(1, 1); // DC uniform
        bits.push(0, 2); // uniform component QP
        bits.push(1, 8); // DC quantization
        bits.align_zero();
        bits.push(0xFD, 8); // no profile-level block
        bits.finish()
    }

    #[derive(Default)]
    struct TestBits {
        bytes: Vec<u8>,
        bit_position: usize,
    }

    impl TestBits {
        fn push(&mut self, value: u64, width: u8) {
            for shift in (0..width).rev() {
                if self.bit_position.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let bit = ((value >> shift) & 1) as u8;
                let index = self.bytes.len() - 1;
                self.bytes[index] |= bit << (7 - self.bit_position % 8);
                self.bit_position += 1;
            }
        }

        fn align_zero(&mut self) {
            while !self.bit_position.is_multiple_of(8) {
                self.push(0, 1);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            self.align_zero();
            self.bytes
        }
    }
}
