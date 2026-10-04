//! Decodes JPEG XL images into row-major RGBA8 pixels.
//!
//! Both raw JPEG XL codestreams and ISO BMFF-style `.jxl` containers are accepted. The first
//! displayable keyframe is rendered with orientation applied and color-managed into sRGB before
//! normalization to RGBA8. Decoder dimensions, pixel count, and tracked working memory are bounded.

mod error;

use std::io::Cursor;

use jxl_oxide::{AllocTracker, EnumColourEncoding, JxlImage, PixelFormat, RenderingIntent};

use super::{
    DIMENSION_MAX, DecodedImage, Dimensions, PIXELS_MAX, map_dimensions_error,
    widen_to_rgba_in_place,
};

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
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&CODESTREAM_SIGNATURE) || bytes.starts_with(&CONTAINER_SIGNATURE)
}

/// Decodes a JPEG XL image without performing I/O.
///
/// The decoder accepts a raw codestream or container and renders its first displayable keyframe.
/// Embedded and enumerated color encodings are converted to sRGB, and image orientation is applied.
///
/// # Errors
///
/// Returns [`JPEGXLError`] for malformed input, missing displayable frames, resource-limit
/// failures, and unsupported output.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage> {
    if !has_signature(bytes) {
        return Err(error(JPEGXLError::Signature));
    }

    let tracker = AllocTracker::with_limit(DECODER_MEMORY_MAX);
    let mut image = JxlImage::builder()
        .alloc_tracker(tracker)
        .read(Cursor::new(bytes))
        .map_err(codec_error)?;

    let width = image.width();
    let height = image.height();

    validate_dimensions(width, height)?;

    if image.num_loaded_keyframes() == 0 {
        return Err(error(JPEGXLError::NoFrame));
    }

    image.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Perceptual));

    let pixel_format = image.pixel_format();
    let render = image.render_frame(0).map_err(codec_error)?;

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

    // Render into a buffer sized for the RGBA result, so RGB samples widen in place rather than
    // into a second full-image allocation.
    let output_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| error(JPEGXLError::Output("JPEG XL RGBA byte count exceeds usize")))?;
    if sample_count > output_len {
        return Err(error(JPEGXLError::Output(
            "JPEG XL color management did not produce sRGB samples",
        )));
    }

    let mut buffer = vec![0_u8; output_len];
    if stream.write_to_buffer(&mut buffer[..sample_count]) != sample_count {
        return Err(error(JPEGXLError::Output(
            "JPEG XL renderer did not produce a complete image",
        )));
    }

    let rgba = normalize_rgba(buffer, sample_count, pixel_format, pixel_count)?;
    Ok(DecodedImage::new(width, height, rgba))
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    Dimensions::try_new((width, height))
        .map(|_| ())
        .map_err(|dimensions_error| {
            map_dimensions_error(
                dimensions_error,
                || {
                    error(JPEGXLError::Output(
                        "JPEG XL dimensions must both be nonzero",
                    ))
                },
                |width, height| {
                    error(JPEGXLError::LimitExceeded(JPEGXLLimit::Dimensions {
                        actual_width: width,
                        actual_height: height,
                        max: DIMENSION_MAX,
                    }))
                },
                |pixels| {
                    error(JPEGXLError::LimitExceeded(JPEGXLLimit::Pixels {
                        actual: pixels,
                        max: PIXELS_MAX,
                    }))
                },
            )
        })
}

fn pixel_count(width: u32, height: u32) -> Result<usize> {
    usize::try_from(u64::from(width) * u64::from(height)).map_err(|_conversion_error| {
        error(JPEGXLError::Output(
            "JPEG XL pixel count does not fit usize",
        ))
    })
}

/// Widens the first `sample_count` bytes of `buffer`, which is sized for the RGBA result, in place.
fn normalize_rgba(
    mut buffer: Vec<u8>,
    sample_count: usize,
    pixel_format: PixelFormat,
    pixel_count: usize,
) -> Result<Vec<u8>> {
    let output_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| error(JPEGXLError::Output("JPEG XL RGBA byte count exceeds usize")))?;
    if buffer.len() != output_len {
        return Err(error(JPEGXLError::Output(
            "JPEG XL output buffer has an invalid length",
        )));
    }

    match pixel_format {
        PixelFormat::Rgba => {
            if sample_count != output_len {
                return Err(error(JPEGXLError::Output(
                    "JPEG XL RGBA output has an invalid length",
                )));
            }

            Ok(buffer)
        }
        PixelFormat::Rgb => {
            if pixel_count.checked_mul(3) != Some(sample_count) {
                return Err(error(JPEGXLError::Output(
                    "JPEG XL RGB output has an invalid length",
                )));
            }

            widen_to_rgba_in_place(&mut buffer, pixel_count, |[red, green, blue]| {
                [red, green, blue, u8::MAX]
            });

            Ok(buffer)
        }
        PixelFormat::Gray | PixelFormat::Graya | PixelFormat::Cmyk | PixelFormat::Cmyka => {
            Err(error(JPEGXLError::Output(
                "JPEG XL color management did not produce sRGB samples",
            )))
        }
    }
}

fn codec_error(source: Box<dyn std::error::Error + Send + Sync + 'static>) -> Error {
    let detail = source.to_string();
    let lowercase_detail = detail.to_ascii_lowercase();

    // jxl-oxide exposes allocation failures only through these display strings.
    if lowercase_detail.contains("failed to allocate") || lowercase_detail.contains("out of memory")
    {
        error(JPEGXLError::LimitExceeded(JPEGXLLimit::DecoderMemory(
            DECODER_MEMORY_MAX,
        )))
    } else {
        error(JPEGXLError::Codec(source))
    }
}
