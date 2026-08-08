//! Decodes bounded PNG images into row-major RGBA8 pixels.
//!
//! The `png` crate validates the chunk stream, checksums, DEFLATE data, filters, and Adam7
//! interlacing. This adapter applies the viewer's resource limits and normalizes the decoded
//! grayscale, grayscale-alpha, RGB, or RGBA samples to the shared RGBA8 representation.

mod error;

use std::io::Cursor;

use ::png::{BitDepth, ColorType, Decoder, Limits, Transformations};

use super::DecodedImage;
use error::error;

pub use error::{Error, PNGError, PNGLimit, Result};

/// The eight-byte signature at the beginning of every PNG datastream.
pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

const CODEC_MEMORY_MAX: usize = 320 * 1024 * 1024;
const DIMENSION_MAX: u32 = 16_384;
const PIXELS_MAX: u64 = 64 * 1024 * 1024;

/// Returns whether `bytes` begin with the PNG signature.
#[must_use]
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE)
}

/// Decodes a PNG image without performing I/O.
///
/// Text and embedded ICC metadata are ignored because the shared output contract contains only
/// pixels. Palette expansion, transparency expansion, 16-to-8-bit reduction, and Adam7
/// deinterlacing are performed by the codec.
///
/// # Errors
///
/// Returns [`PNGError`] when the PNG is malformed, corrupt, exceeds a resource bound, or cannot be
/// represented by the shared RGBA8 contract.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    let bytes = bytes.as_ref();
    if !has_signature(bytes) {
        return Err(error(PNGError::Signature));
    }

    let mut decoder = Decoder::new_with_limits(
        Cursor::new(bytes),
        Limits {
            bytes: CODEC_MEMORY_MAX,
        },
    );
    decoder.set_transformations(Transformations::EXPAND | Transformations::STRIP_16);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);

    let mut reader = decoder.read_info().map_err(|source| codec_error(&source))?;
    let (width, height) = (reader.info().width, reader.info().height);
    validate_dimensions(width, height)?;

    let output_size = reader.output_buffer_size().ok_or_else(|| {
        error(PNGError::Output(
            "PNG codec could not determine the decoded buffer size",
        ))
    })?;
    let rgba_size = rgba_size(width, height)?;
    if output_size > rgba_size {
        return Err(error(PNGError::LimitExceeded(PNGLimit::DecodedBytes(
            rgba_size,
        ))));
    }

    let mut pixel_buffer = vec![0; output_size];
    let output = reader
        .next_frame(&mut pixel_buffer)
        .map_err(|source| codec_error(&source))?;
    if output.width != width || output.height != height {
        return Err(error(PNGError::Output(
            "animated PNG subframes are not supported",
        )));
    }
    if output.bit_depth != BitDepth::Eight {
        return Err(error(PNGError::Output(
            "PNG codec did not produce eight-bit samples",
        )));
    }
    let used = output.buffer_size();
    let samples = pixel_buffer.get(..used).ok_or_else(|| {
        error(PNGError::Output(
            "PNG codec reported an invalid output length",
        ))
    })?;
    let rgba = normalize_rgba(samples, output.color_type, width, height)?;
    Ok(DecodedImage::new(width, height, rgba))
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(error(PNGError::Output(
            "PNG dimensions must both be nonzero",
        )));
    }
    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(PNGError::LimitExceeded(PNGLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > PIXELS_MAX {
        return Err(error(PNGError::LimitExceeded(PNGLimit::Pixels(PIXELS_MAX))));
    }
    Ok(())
}

fn rgba_size(width: u32, height: u32) -> Result<usize> {
    let bytes = u64::from(width) * u64::from(height) * 4;
    usize::try_from(bytes).map_err(|_source| {
        error(PNGError::Output(
            "PNG output size does not fit this platform",
        ))
    })
}

fn normalize_rgba(
    samples: &[u8],
    color_type: ColorType,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
        .expect("validated PNG pixel count fits usize");
    let channels = match color_type {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => {
            return Err(error(PNGError::Output(
                "PNG palette was not expanded by the codec",
            )));
        }
    };
    let expected = pixel_count
        .checked_mul(channels)
        .ok_or_else(|| error(PNGError::Output("PNG sample count exceeds this platform")))?;
    if samples.len() != expected {
        return Err(error(PNGError::Output(
            "PNG codec returned an unexpected sample count",
        )));
    }
    if color_type == ColorType::Rgba {
        return Ok(samples.to_vec());
    }

    let mut rgba = Vec::with_capacity(pixel_count * 4);
    match color_type {
        ColorType::Grayscale => {
            for &gray in samples {
                rgba.extend_from_slice(&[gray, gray, gray, u8::MAX]);
            }
        }
        ColorType::GrayscaleAlpha => {
            for sample in samples.as_chunks::<2>().0 {
                rgba.extend_from_slice(&[sample[0], sample[0], sample[0], sample[1]]);
            }
        }
        ColorType::Rgb => {
            for sample in samples.as_chunks::<3>().0 {
                rgba.extend_from_slice(&[sample[0], sample[1], sample[2], u8::MAX]);
            }
        }
        ColorType::Rgba | ColorType::Indexed => {
            unreachable!("RGBA and indexed PNG outputs were handled before sample normalization")
        }
    }
    Ok(rgba)
}

fn codec_error(source: &::png::DecodingError) -> Error {
    let detail = source.to_string();
    if detail.to_ascii_lowercase().contains("limit") {
        error(PNGError::LimitExceeded(PNGLimit::CodecMemory(
            CODEC_MEMORY_MAX,
        )))
    } else {
        error(PNGError::Codec(detail))
    }
}
