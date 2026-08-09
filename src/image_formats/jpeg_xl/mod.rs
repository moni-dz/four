//! Decodes JPEG XL images into row-major RGBA8 pixels.
//!
//! Both raw JPEG XL codestreams and ISO BMFF-style `.jxl` containers are accepted. The first
//! displayable keyframe is rendered with orientation applied and color-managed into sRGB before
//! normalization to RGBA8. Decoder dimensions, pixel count, and tracked working memory are bounded.

mod error;

use std::io::Cursor;

use jxl_oxide::{AllocTracker, EnumColourEncoding, JxlImage, PixelFormat, RenderingIntent};

use super::{DIMENSION_MAX, DecodedImage, PIXELS_MAX};

use error::error;
pub use error::{Error, JPEGXLError, JPEGXLLimit, Result};

/// The two-byte signature at the beginning of a raw JPEG XL codestream.
pub const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];

/// The signature box at the beginning of a JPEG XL container.
pub const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
];

const DECODER_MEMORY_MAX: usize = 512 * 1024 * 1024;

/// Returns whether `bytes` begins with a standard JPEG XL signature.
#[must_use]
pub fn has_signature(bytes: impl AsRef<[u8]>) -> bool {
    let bytes = bytes.as_ref();
    bytes.starts_with(&CODESTREAM_SIGNATURE) || bytes.starts_with(&CONTAINER_SIGNATURE)
}

/// Decodes a JPEG XL image without performing I/O.
///
/// The decoder accepts a raw codestream or container and renders its first displayable keyframe.
/// Embedded and enumerated color encodings are converted to sRGB, and image orientation is applied.
///
/// # Errors
///
/// Returns [`JPEGXLError`] when the JPEG XL data is malformed, has no displayable frame, exceeds a
/// resource bound, or cannot be rendered into the shared RGBA8 representation.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    let bytes = bytes.as_ref();
    if !has_signature(bytes) {
        return Err(error(JPEGXLError::Signature));
    }

    let tracker = AllocTracker::with_limit(DECODER_MEMORY_MAX);
    let mut image = JxlImage::builder()
        .alloc_tracker(tracker)
        .read(Cursor::new(bytes))
        .map_err(|source| codec_error(source.as_ref()))?;

    let width = image.width();
    let height = image.height();

    validate_dimensions(width, height)?;

    if image.num_loaded_keyframes() == 0 {
        return Err(error(JPEGXLError::NoFrame));
    }

    image.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Perceptual));

    let pixel_format = image.pixel_format();
    let render = image
        .render_frame(0)
        .map_err(|source| codec_error(source.as_ref()))?;

    let mut stream = render.stream();
    if stream.width() != width || stream.height() != height {
        return Err(error(JPEGXLError::Output(
            "JPEG XL rendered dimensions do not match the image header",
        )));
    }

    let pixel_count = pixel_count(width, height)?;
    let channels = pixel_format.channels();

    let sample_count = pixel_count.checked_mul(channels).ok_or_else(|| {
        error(JPEGXLError::Output(
            "JPEG XL rendered sample count exceeds usize",
        ))
    })?;

    if usize::try_from(stream.channels()).ok() != Some(channels) {
        return Err(error(JPEGXLError::Output(
            "JPEG XL rendered channel count does not match its pixel format",
        )));
    }

    let mut samples = vec![0_u8; sample_count];
    if stream.write_to_buffer(&mut samples) != sample_count {
        return Err(error(JPEGXLError::Output(
            "JPEG XL renderer did not produce a complete image",
        )));
    }

    let rgba = normalize_rgba(samples, pixel_format, pixel_count)?;
    Ok(DecodedImage::new(width, height, rgba))
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(error(JPEGXLError::Output(
            "JPEG XL dimensions must both be nonzero",
        )));
    }

    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(JPEGXLError::LimitExceeded(JPEGXLLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }

    if u64::from(width) * u64::from(height) > PIXELS_MAX {
        return Err(error(JPEGXLError::LimitExceeded(JPEGXLLimit::Pixels(
            PIXELS_MAX,
        ))));
    }

    Ok(())
}

fn pixel_count(width: u32, height: u32) -> Result<usize> {
    usize::try_from(u64::from(width) * u64::from(height)).map_err(|_conversion_error| {
        error(JPEGXLError::Output(
            "JPEG XL pixel count does not fit usize",
        ))
    })
}

fn normalize_rgba(
    samples: Vec<u8>,
    pixel_format: PixelFormat,
    pixel_count: usize,
) -> Result<Vec<u8>> {
    let output_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| error(JPEGXLError::Output("JPEG XL RGBA byte count exceeds usize")))?;
    match pixel_format {
        PixelFormat::Rgba => {
            if samples.len() != output_len {
                return Err(error(JPEGXLError::Output(
                    "JPEG XL RGBA output has an invalid length",
                )));
            }
            Ok(samples)
        }
        PixelFormat::Rgb => {
            let mut rgba = Vec::with_capacity(output_len);
            let (pixels, remainder) = samples.as_chunks::<3>();
            if !remainder.is_empty() || pixels.len() != pixel_count {
                return Err(error(JPEGXLError::Output(
                    "JPEG XL RGB output has an invalid length",
                )));
            }
            rgba.extend(
                pixels
                    .iter()
                    .flat_map(|&[red, green, blue]| [red, green, blue, u8::MAX]),
            );

            Ok(rgba)
        }
        PixelFormat::Gray | PixelFormat::Graya | PixelFormat::Cmyk | PixelFormat::Cmyka => {
            Err(error(JPEGXLError::Output(
                "JPEG XL color management did not produce sRGB samples",
            )))
        }
    }
}

fn codec_error(source: &(dyn std::error::Error + Send + Sync + 'static)) -> Error {
    let detail = source.to_string();
    let lowercase_detail = detail.to_ascii_lowercase();

    if lowercase_detail.contains("memory limit") || lowercase_detail.contains("out of memory") {
        error(JPEGXLError::LimitExceeded(JPEGXLLimit::DecoderMemory(
            DECODER_MEMORY_MAX,
        )))
    } else {
        error(JPEGXLError::Codec(detail))
    }
}
