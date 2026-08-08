//! Decodes bounded GIF images into row-major RGBA8 pixels.
//!
//! The first image frame is decoded by the `gif` crate and composited at its descriptor offset on
//! a transparent logical screen. This produces a deterministic still image while retaining GIF
//! transparency and the dimensions declared by the logical screen descriptor.

mod error;

use std::io::Cursor;
use std::num::NonZeroU64;

use ::gif::{ColorOutput, DecodeOptions, MemoryLimit};

use super::DecodedImage;
use error::error;

pub use error::{Error, GIFError, GIFLimit, Result};

/// The `GIF87a` signature.
pub const SIGNATURE_87A: [u8; 6] = *b"GIF87a";
/// The `GIF89a` signature.
pub const SIGNATURE_89A: [u8; 6] = *b"GIF89a";

const DIMENSION_MAX: u32 = 16_384;
const PIXELS_MAX: u64 = 64 * 1024 * 1024;
const FRAME_BYTES_MAX: u64 = PIXELS_MAX * 4;

/// Returns whether `bytes` begin with a supported GIF signature.
#[must_use]
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE_87A) || bytes.starts_with(&SIGNATURE_89A)
}

/// Decodes the first GIF frame without performing I/O.
///
/// The frame is expanded to RGBA by the codec and composited on a transparent logical screen. GIF
/// animation timing and later frames are intentionally outside the shared still-image contract.
///
/// # Errors
///
/// Returns [`GIFError`] when the GIF is malformed, has no frame, exceeds a resource bound, or the
/// codec produces samples inconsistent with its frame descriptor.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    let bytes = bytes.as_ref();
    if !has_signature(bytes) {
        return Err(error(GIFError::Signature));
    }

    let mut options = DecodeOptions::new();
    options.set_color_output(ColorOutput::RGBA);
    let frame_limit = NonZeroU64::new(FRAME_BYTES_MAX)
        .ok_or_else(|| error(GIFError::Output("GIF frame byte limit must be nonzero")))?;
    options.set_memory_limit(MemoryLimit::Bytes(frame_limit));
    options.check_frame_consistency(true);
    let mut decoder = options
        .read_info(Cursor::new(bytes))
        .map_err(|source| codec_error(&source))?;
    let (width, height) = (u32::from(decoder.width()), u32::from(decoder.height()));
    validate_dimensions(width, height)?;

    let canvas_size =
        usize::try_from(u64::from(width) * u64::from(height) * 4).map_err(|_source| {
            error(GIFError::Output(
                "GIF output size does not fit this platform",
            ))
        })?;
    let frame = decoder
        .read_next_frame()
        .map_err(|source| codec_error(&source))?
        .ok_or_else(|| error(GIFError::NoFrame))?;
    let frame_width = usize::from(frame.width);
    let frame_height = usize::from(frame.height);
    let frame_size = frame_width
        .checked_mul(frame_height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| error(GIFError::Output("GIF frame sample count overflowed")))?;
    if frame.buffer.len() != frame_size {
        return Err(error(GIFError::Output(
            "GIF codec returned an unexpected frame sample count",
        )));
    }

    let canvas_width = usize::try_from(width)
        .map_err(|_source| error(GIFError::Output("GIF width does not fit this platform")))?;
    let left = usize::from(frame.left);
    let top = usize::from(frame.top);
    let mut rgba = vec![0; canvas_size];
    for row in 0..frame_height {
        let source_start = row * frame_width * 4;
        let target_start = ((top + row) * canvas_width + left) * 4;
        let source = &frame.buffer[source_start..source_start + frame_width * 4];
        rgba[target_start..target_start + frame_width * 4].copy_from_slice(source);
    }
    Ok(DecodedImage::new(width, height, rgba))
}

fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(error(GIFError::Output(
            "GIF logical-screen dimensions must both be nonzero",
        )));
    }
    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(GIFError::LimitExceeded(GIFLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > PIXELS_MAX {
        return Err(error(GIFError::LimitExceeded(GIFLimit::Pixels(PIXELS_MAX))));
    }
    Ok(())
}

fn codec_error(source: &::gif::DecodingError) -> Error {
    let detail = source.to_string();
    let lowercase = detail.to_ascii_lowercase();
    if lowercase.contains("memory limit") || lowercase.contains("out of memory") {
        error(GIFError::LimitExceeded(GIFLimit::CodecFrameBytes(
            FRAME_BYTES_MAX,
        )))
    } else {
        error(GIFError::Codec(detail))
    }
}
