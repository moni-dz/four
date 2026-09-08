//! Decodes bounded PNG images into row-major RGBA8 pixels.
//!
//! The `png` crate validates chunks, checksums, DEFLATE data, filters, and Adam7 interlacing.
//! The adapter applies resource limits and normalizes grayscale, grayscale-alpha, RGB, and RGBA
//! samples to RGBA8.

mod error;

use std::io::Cursor;

use ::png::{BitDepth, ColorType, Decoder, Limits, Transformations};

use super::{DIMENSION_MAX, DecodedImage, Dimensions, PIXELS_MAX, map_dimensions_error};
use error::error;

pub use error::{Error, PNGError, PNGLimit, Result};

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

    let mut pixel_buffer = vec![0; output_size];
    let output = reader.next_frame(&mut pixel_buffer).map_err(codec_error)?;

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
            rgba.extend(samples.iter().flat_map(|&gray| [gray, gray, gray, u8::MAX]));
        }
        ColorType::GrayscaleAlpha => rgba.extend(
            samples
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|sample| [sample[0], sample[0], sample[0], sample[1]]),
        ),
        ColorType::Rgb => rgba.extend(
            samples
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|sample| [sample[0], sample[1], sample[2], u8::MAX]),
        ),
        ColorType::Rgba | ColorType::Indexed => {
            unreachable!("RGBA and indexed PNG outputs were handled before sample normalization")
        }
    }

    Ok(rgba)
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
