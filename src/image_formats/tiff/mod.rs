//! Decodes bounded TIFF images into row-major RGBA8 pixels.
//!
//! The `tiff` crate parses and decompresses the first image directory. This adapter accepts
//! chunky and planar integer grayscale, grayscale-alpha, RGB, RGBA, CMYK, and CMYK-alpha samples,
//! scales their declared bit depth to eight bits, and normalizes them to the shared RGBA8 output.

mod error;

use std::io::Cursor;

use ::tiff::ColorType;
use ::tiff::decoder::{BufferLayoutPreference, Decoder, DecodingResult, Limits};
use rayon::prelude::*;

use super::{DIMENSION_MAX, DecodedImage, PIXELS_MAX};
use error::error;

pub use error::{Error, Result, TIFFError, TIFFLimit};

/// The little-endian classic TIFF signature.
pub const SIGNATURE_CLASSIC_LE: [u8; 4] = *b"II*\0";
/// The big-endian classic TIFF signature.
pub const SIGNATURE_CLASSIC_BE: [u8; 4] = *b"MM\0*";
/// The little-endian `BigTIFF` signature.
pub const SIGNATURE_BIG_LE: [u8; 4] = *b"II+\0";
/// The big-endian `BigTIFF` signature.
pub const SIGNATURE_BIG_BE: [u8; 4] = *b"MM\0+";

const CODEC_BUFFER_MAX: usize = 512 * 1024 * 1024;
const PARALLEL_PIXELS_MIN: usize = 256 * 1024;
const PARALLEL_PIXELS_PER_JOB: usize = 64 * 1024;

/// Returns whether `bytes` begin with a classic TIFF or `BigTIFF` signature.
#[must_use]
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE_CLASSIC_LE)
        || bytes.starts_with(&SIGNATURE_CLASSIC_BE)
        || bytes.starts_with(&SIGNATURE_BIG_LE)
        || bytes.starts_with(&SIGNATURE_BIG_BE)
}

/// Decodes the first TIFF image directory without performing I/O.
///
/// Integer samples between one and 64 bits are scaled to RGBA8. Palette, YCbCr, Lab, multiband,
/// signed-integer, and floating-point representations return a structured unsupported error.
///
/// # Errors
///
/// Returns [`TIFFError`] when the TIFF is malformed, exceeds a resource bound, uses an unsupported
/// sample representation, or the codec output is inconsistent with its declared layout.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    let bytes = bytes.as_ref();
    if !has_signature(bytes) {
        return Err(error(TIFFError::Signature));
    }

    let mut limits = Limits::default();
    limits.decoding_buffer_size = CODEC_BUFFER_MAX;
    let mut decoder = Decoder::new(Cursor::new(bytes))
        .map_err(|source| codec_error(&source))?
        .with_limits(limits);
    let (width, height) = decoder
        .dimensions()
        .map_err(|source| codec_error(&source))?;
    validate_dimensions(width, height)?;
    let color = decoder.colortype().map_err(|source| codec_error(&source))?;
    let format = PixelFormat::from_color(color)?;
    validate_raw_size(width, height, format)?;

    let mut sample_buffer = DecodingResult::U8(Vec::new());
    let layout = decoder
        .read_image_to_buffer(&mut sample_buffer)
        .map_err(|source| codec_error(&source))?;
    let rgba = match sample_buffer {
        DecodingResult::U8(samples) => {
            normalize_unsigned(&samples, width, height, format, &layout, u64::from)?
        }
        DecodingResult::U16(samples) => {
            normalize_unsigned(&samples, width, height, format, &layout, u64::from)?
        }
        DecodingResult::U32(samples) => {
            normalize_unsigned(&samples, width, height, format, &layout, u64::from)?
        }
        DecodingResult::U64(samples) => {
            normalize_unsigned(&samples, width, height, format, &layout, |sample| sample)?
        }
        DecodingResult::F16(_)
        | DecodingResult::F32(_)
        | DecodingResult::F64(_)
        | DecodingResult::I8(_)
        | DecodingResult::I16(_)
        | DecodingResult::I32(_)
        | DecodingResult::I64(_) => {
            return Err(error(TIFFError::Unsupported(
                "signed and floating-point samples are not supported".to_owned(),
            )));
        }
    };
    Ok(DecodedImage::new(width, height, rgba))
}

#[derive(Clone, Copy, Debug)]
enum PixelKind {
    Grayscale,
    GrayscaleAlpha,
    RGB,
    RGBA,
    CMYK,
    CMYKA,
}

#[derive(Clone, Copy, Debug)]
struct PixelFormat {
    kind: PixelKind,
    channels: usize,
    bit_depth: u8,
}

impl PixelFormat {
    fn from_color(color: ColorType) -> Result<Self> {
        let (kind, channels, bit_depth) = match color {
            ColorType::Gray(depth) => (PixelKind::Grayscale, 1, depth),
            ColorType::GrayA(depth) => (PixelKind::GrayscaleAlpha, 2, depth),
            ColorType::RGB(depth) => (PixelKind::RGB, 3, depth),
            ColorType::RGBA(depth) => (PixelKind::RGBA, 4, depth),
            ColorType::CMYK(depth) => (PixelKind::CMYK, 4, depth),
            ColorType::CMYKA(depth) => (PixelKind::CMYKA, 5, depth),
            other => {
                return Err(error(TIFFError::Unsupported(format!(
                    "color type {other:?} is not mapped to RGBA8"
                ))));
            }
        };

        if !(1..=64).contains(&bit_depth) {
            return Err(error(TIFFError::Unsupported(format!(
                "{bit_depth}-bit integer samples are not supported"
            ))));
        }

        Ok(Self {
            kind,
            channels,
            bit_depth,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct SampleLayout {
    row_stride: usize,
    plane_stride: usize,
    planes: usize,
    channels: usize,
}

impl SampleLayout {
    fn new<T>(codec: &BufferLayoutPreference, samples: &[T], channels: usize) -> Result<Self> {
        let sample_bytes = size_of::<T>();
        let row_stride_bytes = codec
            .row_stride
            .ok_or_else(|| error(TIFFError::Output("TIFF codec omitted its row stride")))?
            .get();
        let plane_stride_bytes = codec
            .plane_stride
            .ok_or_else(|| error(TIFFError::Output("TIFF codec omitted its plane stride")))?
            .get();

        if !row_stride_bytes.is_multiple_of(sample_bytes)
            || !plane_stride_bytes.is_multiple_of(sample_bytes)
            || !codec.complete_len.is_multiple_of(sample_bytes)
        {
            return Err(error(TIFFError::Output(
                "TIFF codec returned a misaligned sample layout",
            )));
        }

        if codec.complete_len > samples.len().saturating_mul(sample_bytes) {
            return Err(error(TIFFError::Output(
                "TIFF codec returned an incomplete planar buffer",
            )));
        }

        if codec.planes != 1 && codec.planes != channels {
            return Err(error(TIFFError::Output(
                "TIFF plane count does not match its color channels",
            )));
        }

        Ok(Self {
            row_stride: row_stride_bytes / sample_bytes,
            plane_stride: plane_stride_bytes / sample_bytes,
            planes: codec.planes,
            channels,
        })
    }

    fn index(self, x: usize, y: usize, channel: usize) -> Option<usize> {
        let row = y.checked_mul(self.row_stride)?;
        if self.planes == 1 {
            row.checked_add(x.checked_mul(self.channels)?)?
                .checked_add(channel)
        } else {
            channel
                .checked_mul(self.plane_stride)?
                .checked_add(row)?
                .checked_add(x)
        }
    }
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(error(TIFFError::Output(
            "TIFF dimensions must both be nonzero",
        )));
    }

    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(TIFFError::LimitExceeded(TIFFLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }

    let pixels = u64::from(width) * u64::from(height);
    if pixels > PIXELS_MAX {
        return Err(error(TIFFError::LimitExceeded(TIFFLimit::Pixels(
            PIXELS_MAX,
        ))));
    }

    Ok(())
}

fn validate_raw_size(width: u32, height: u32, format: PixelFormat) -> Result<()> {
    let sample_bytes = u64::from(format.bit_depth).div_ceil(8);
    let bytes = u64::from(width)
        * u64::from(height)
        * u64::try_from(format.channels).expect("TIFF channel count fits u64")
        * sample_bytes;

    if bytes > CODEC_BUFFER_MAX as u64 {
        return Err(error(TIFFError::LimitExceeded(
            TIFFLimit::CodecBufferBytes(CODEC_BUFFER_MAX),
        )));
    }

    Ok(())
}

fn normalize_unsigned<T: Copy + Sync>(
    samples: &[T],
    width: u32,
    height: u32,
    format: PixelFormat,
    codec_layout: &BufferLayoutPreference,
    into_u64: impl Fn(T) -> u64 + Sync,
) -> Result<Vec<u8>> {
    let layout = SampleLayout::new(codec_layout, samples, format.channels)?;
    let width = usize::try_from(width).expect("validated TIFF width fits usize");
    let height = usize::try_from(height).expect("validated TIFF height fits usize");
    normalize_unsigned_with_layout(samples, width, height, format, layout, &into_u64)
}

fn normalize_unsigned_with_layout<T: Copy + Sync>(
    samples: &[T],
    width: usize,
    height: usize,
    format: PixelFormat,
    layout: SampleLayout,
    into_u64: &(impl Fn(T) -> u64 + Sync),
) -> Result<Vec<u8>> {
    if width * height >= PARALLEL_PIXELS_MIN {
        let mut rgba = vec![0; width * height * 4];
        rgba.par_chunks_exact_mut(4)
            .with_min_len(PARALLEL_PIXELS_PER_JOB)
            .enumerate()
            .try_for_each(|(index, target)| {
                let pixel = normalize_pixel(
                    samples,
                    index % width,
                    index / width,
                    format,
                    layout,
                    into_u64,
                )?;
                target.copy_from_slice(&pixel);
                Ok::<(), Error>(())
            })?;
        Ok(rgba)
    } else {
        let mut rgba = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                rgba.extend_from_slice(&normalize_pixel(samples, x, y, format, layout, into_u64)?);
            }
        }
        Ok(rgba)
    }
}

fn normalize_pixel<T: Copy>(
    samples: &[T],
    x: usize,
    y: usize,
    format: PixelFormat,
    layout: SampleLayout,
    into_u64: &impl Fn(T) -> u64,
) -> Result<[u8; 4]> {
    let sample = |channel| -> Result<u8> {
        let index = layout
            .index(x, y, channel)
            .ok_or_else(|| error(TIFFError::Output("TIFF sample offset overflowed")))?;
        let value = samples.get(index).copied().ok_or_else(|| {
            error(TIFFError::Output(
                "TIFF sample layout exceeds its decoded buffer",
            ))
        })?;
        Ok(scale_sample(into_u64(value), format.bit_depth))
    };

    match format.kind {
        PixelKind::Grayscale => {
            let gray = sample(0)?;
            Ok([gray, gray, gray, u8::MAX])
        }
        PixelKind::GrayscaleAlpha => {
            let gray = sample(0)?;
            Ok([gray, gray, gray, sample(1)?])
        }
        PixelKind::RGB => Ok([sample(0)?, sample(1)?, sample(2)?, u8::MAX]),
        PixelKind::RGBA => Ok([sample(0)?, sample(1)?, sample(2)?, sample(3)?]),
        PixelKind::CMYK => {
            let rgb = cmyk_to_rgb(sample(0)?, sample(1)?, sample(2)?, sample(3)?);
            Ok([rgb[0], rgb[1], rgb[2], u8::MAX])
        }
        PixelKind::CMYKA => {
            let rgb = cmyk_to_rgb(sample(0)?, sample(1)?, sample(2)?, sample(3)?);
            Ok([rgb[0], rgb[1], rgb[2], sample(4)?])
        }
    }
}

fn scale_sample(value: u64, bit_depth: u8) -> u8 {
    let max = if bit_depth == 64 {
        u64::MAX
    } else {
        (1_u64 << bit_depth) - 1
    };
    let value = value.min(max);
    let scaled = (u128::from(value) * 255 + u128::from(max) / 2) / u128::from(max);
    u8::try_from(scaled).expect("scaled TIFF sample is in the eight-bit range")
}

fn cmyk_to_rgb(cyan: u8, magenta: u8, yellow: u8, black: u8) -> [u8; 3] {
    let black_scale = u16::from(u8::MAX - black);
    let convert = |component: u8| {
        let value = (u16::from(u8::MAX - component) * black_scale + 127) / 255;
        u8::try_from(value).expect("CMYK conversion stays in the eight-bit range")
    };
    [convert(cyan), convert(magenta), convert(yellow)]
}

fn codec_error(source: &::tiff::TiffError) -> Error {
    if matches!(source, ::tiff::TiffError::LimitsExceeded) {
        error(TIFFError::LimitExceeded(TIFFLimit::CodecBufferBytes(
            CODEC_BUFFER_MAX,
        )))
    } else {
        error(TIFFError::Codec(source.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_normalization_preserves_rgba_pixel_order() {
        let width = 512;
        let height = 512;
        let samples: Vec<_> = (0..width * height * 4)
            .map(|index| u8::try_from(index % 256).expect("sample fits u8"))
            .collect();
        let format = PixelFormat {
            kind: PixelKind::RGBA,
            channels: 4,
            bit_depth: 8,
        };
        let layout = SampleLayout {
            row_stride: width * 4,
            plane_stride: samples.len(),
            planes: 1,
            channels: 4,
        };

        let rgba =
            normalize_unsigned_with_layout(&samples, width, height, format, layout, &u64::from)
                .unwrap();

        assert_eq!(rgba, samples);
    }
}
