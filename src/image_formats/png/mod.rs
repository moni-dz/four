//! Decodes bounded PNG images into row-major RGBA8 pixels.
//!
//! The `png` crate validates chunks, checksums, DEFLATE data, filters, and Adam7 interlacing.
//! The adapter applies resource limits and normalizes grayscale, grayscale-alpha, RGB, and RGBA
//! samples to RGBA8.
//!
//! PNGs that carry a `cICP` chunk declaring PQ or HLG transfer (the PNG HDR convention) are decoded
//! by [`decode_hdr`] to linear scRGB instead; [`is_hdr`] detects them from the header alone.

mod error;
mod hdr;

use std::io::Cursor;

use ::png::{BitDepth, ColorType, Decoder, Limits, Transformations};

use super::{
    DIMENSION_MAX, DecodedImage, Dimensions, PIXELS_MAX, map_dimensions_error,
    widen_to_rgba_in_place,
};
use error::error;

pub use error::{Error, PNGError, PNGLimit, Result};
pub use hdr::{HDRImage, decode_hdr, is_hdr};

/// The eight-byte signature at the beginning of every PNG datastream.
pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

const CODEC_MEMORY_MAX: usize = 320 * 1024 * 1024;

/// Returns whether `bytes` begin with the PNG signature.
#[must_use]
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE)
}

/// Decodes a PNG image without performing I/O.
///
/// Text and embedded ICC metadata are ignored. The codec handles palette and transparency
/// expansion, 16-to-8-bit reduction, and Adam7 deinterlacing.
///
/// # Errors
///
/// Returns [`PNGError`] for malformed or corrupt input, resource-limit failures, and unsupported
/// output formats.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage> {
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

    let mut reader = decoder.read_info().map_err(codec_error)?;
    let (width, height) = (reader.info().width, reader.info().height);
    validate_dimensions(width, height)?;

    let output_size = reader.output_buffer_size().ok_or_else(|| {
        error(PNGError::Output(
            "PNG codec could not determine the decoded buffer size",
        ))
    })?;

    let rgba_size = rgba_size(width, height)?;
    if output_size > rgba_size {
        return Err(error(PNGError::LimitExceeded(PNGLimit::DecodedBytes {
            actual: output_size,
            max: rgba_size,
        })));
    }

    // Size the buffer for the RGBA result up front, so narrower samples widen in place rather
    // than into a second full-image allocation.
    let mut pixel_buffer = vec![0; rgba_size];
    let output = reader
        .next_frame(&mut pixel_buffer[..output_size])
        .map_err(codec_error)?;

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
    if used > output_size {
        return Err(error(PNGError::Output(
            "PNG codec reported an invalid output length",
        )));
    }

    let rgba = normalize_rgba(pixel_buffer, used, output.color_type, width, height)?;
    Ok(DecodedImage::new(width, height, rgba))
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    Dimensions::try_new((width, height))
        .map(|_| ())
        .map_err(|dimensions_error| {
            map_dimensions_error(
                dimensions_error,
                || error(PNGError::Output("PNG dimensions must both be nonzero")),
                |width, height| {
                    error(PNGError::LimitExceeded(PNGLimit::Dimensions {
                        actual_width: width,
                        actual_height: height,
                        max: DIMENSION_MAX,
                    }))
                },
                |pixels| {
                    error(PNGError::LimitExceeded(PNGLimit::Pixels {
                        actual: pixels,
                        max: PIXELS_MAX,
                    }))
                },
            )
        })
}

fn rgba_size(width: u32, height: u32) -> Result<usize> {
    let bytes = u64::from(width) * u64::from(height) * 4;
    usize::try_from(bytes).map_err(|_source| {
        error(PNGError::Output(
            "PNG output size does not fit this platform",
        ))
    })
}

/// Widens the first `used` bytes of `buffer` to RGBA8 in place.
///
/// `buffer` must be sized for the RGBA result; its prefix holds the codec's samples.
fn normalize_rgba(
    mut buffer: Vec<u8>,
    used: usize,
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

    if used != expected || buffer.len() != pixel_count * 4 {
        return Err(error(PNGError::Output(
            "PNG codec returned an unexpected sample count",
        )));
    }

    match color_type {
        ColorType::Grayscale => {
            widen_to_rgba_in_place(&mut buffer, pixel_count, |[gray]| {
                [gray, gray, gray, u8::MAX]
            });
        }
        ColorType::GrayscaleAlpha => {
            widen_to_rgba_in_place(&mut buffer, pixel_count, |[gray, alpha]| {
                [gray, gray, gray, alpha]
            });
        }
        ColorType::Rgb => {
            widen_to_rgba_in_place(&mut buffer, pixel_count, |[red, green, blue]| {
                [red, green, blue, u8::MAX]
            });
        }
        ColorType::Rgba => {}
        ColorType::Indexed => {
            unreachable!("indexed PNG outputs were rejected before sample normalization")
        }
    }

    Ok(buffer)
}

fn codec_error(source: ::png::DecodingError) -> Error {
    // `png::DecodingError::LimitsExceeded` is a structured signal for the configured memory
    // cap; it does not need substring matching against the error text.
    if matches!(source, ::png::DecodingError::LimitsExceeded) {
        error(PNGError::LimitExceeded(PNGLimit::CodecMemory(
            CODEC_MEMORY_MAX,
        )))
    } else {
        error(PNGError::Codec(Box::new(source)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_normalization_reuses_the_decoded_buffer() {
        let samples = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let allocation = samples.as_ptr();
        let rgba = normalize_rgba(samples, 8, ColorType::Rgba, 2, 1).unwrap();

        assert_eq!(rgba.as_ptr(), allocation);
        assert_eq!(rgba, [10, 20, 30, 40, 50, 60, 70, 80]);
        normalize_rgba(vec![0; 8], 7, ColorType::Rgba, 2, 1).unwrap_err();
        normalize_rgba(vec![0; 9], 9, ColorType::Rgba, 2, 1).unwrap_err();
    }

    #[test]
    fn narrower_samples_widen_in_the_decoded_buffer() {
        let mut samples = vec![0; 8];
        samples[..6].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        let allocation = samples.as_ptr();
        let rgba = normalize_rgba(samples, 6, ColorType::Rgb, 2, 1).unwrap();
        assert_eq!(rgba.as_ptr(), allocation);
        assert_eq!(rgba, [1, 2, 3, 255, 4, 5, 6, 255]);

        let mut samples = vec![0; 8];
        samples[..2].copy_from_slice(&[7, 9]);
        let rgba = normalize_rgba(samples, 2, ColorType::Grayscale, 2, 1).unwrap();
        assert_eq!(rgba, [7, 7, 7, 255, 9, 9, 9, 255]);

        let mut samples = vec![0; 8];
        samples[..4].copy_from_slice(&[7, 8, 9, 10]);
        let rgba = normalize_rgba(samples, 4, ColorType::GrayscaleAlpha, 2, 1).unwrap();
        assert_eq!(rgba, [7, 7, 7, 8, 9, 9, 9, 10]);
    }
}
