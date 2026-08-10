//! Validates bounded GIF animations and decodes their first frame into RGBA8 pixels.
//!
//! Every frame is decoded to validate the animation and bound its eventual full-canvas memory.
//! The first frame is composited at its descriptor offset on a transparent logical screen for the
//! shared still-image representation. Playback is delegated to the display layer.

mod error;

use std::io::Cursor;
use std::num::NonZeroU64;

use ::gif::{ColorOutput, DecodeOptions, MemoryLimit};

use super::{DIMENSION_MAX, DecodedImage, PIXELS_MAX};
use error::error;

pub use error::{Error, GIFError, GIFLimit, Result};

/// The `GIF87a` signature.
pub const SIGNATURE_87A: [u8; 6] = *b"GIF87a";
/// The `GIF89a` signature.
pub const SIGNATURE_89A: [u8; 6] = *b"GIF89a";

const FRAME_BYTES_MAX: u64 = PIXELS_MAX * 4;
const ANIMATION_BYTES_MAX: u64 = FRAME_BYTES_MAX * 2;
const FRAMES_MAX: u64 = 10_000;

/// Returns whether `bytes` begin with a supported GIF signature.
#[must_use]
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE_87A) || bytes.starts_with(&SIGNATURE_89A)
}

/// Validates a GIF animation and decodes its first frame without performing I/O.
///
/// Frames are expanded to RGBA by the codec. The first is composited on a transparent logical
/// screen, while every later frame is decoded and checked against the animation resource bounds.
///
/// # Errors
///
/// Returns [`GIFError`] when the GIF is malformed, has no frame, exceeds a resource bound, or a
/// decoded frame is inconsistent with its descriptor.
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

    let canvas_bytes = u64::from(width) * u64::from(height) * 4;
    let canvas_size = usize::try_from(canvas_bytes).map_err(|_source| {
        error(GIFError::Output(
            "GIF output size does not fit this platform",
        ))
    })?;
    let canvas_width = usize::try_from(width)
        .map_err(|_source| error(GIFError::Output("GIF width does not fit this platform")))?;
    let canvas_height = usize::try_from(height)
        .map_err(|_source| error(GIFError::Output("GIF height does not fit this platform")))?;

    let mut decoded_animation_bytes = 0_u64;
    let mut frame_count = 0_u64;
    let mut first_rgba = None;

    while let Some(frame) = decoder
        .read_next_frame()
        .map_err(|source| codec_error(&source))?
    {
        account_frame(&mut frame_count, &mut decoded_animation_bytes, canvas_bytes)?;
        let (frame_width, frame_height, left, top) =
            validate_frame(frame, canvas_width, canvas_height)?;

        if first_rgba.is_none() {
            first_rgba = Some(composite_first_frame(
                frame,
                canvas_size,
                canvas_width,
                frame_width,
                frame_height,
                left,
                top,
            ));
        }
    }

    let rgba = first_rgba.ok_or_else(|| error(GIFError::NoFrame))?;
    Ok(DecodedImage::new(width, height, rgba))
}

fn account_frame(frame_count: &mut u64, decoded_bytes: &mut u64, canvas_bytes: u64) -> Result<()> {
    *frame_count = frame_count
        .checked_add(1)
        .ok_or_else(|| error(GIFError::Output("GIF frame count overflowed")))?;
    if *frame_count > FRAMES_MAX {
        return Err(error(GIFError::LimitExceeded(GIFLimit::Frames(FRAMES_MAX))));
    }

    *decoded_bytes = decoded_bytes
        .checked_add(canvas_bytes)
        .ok_or_else(|| error(GIFError::Output("GIF animation byte count overflowed")))?;
    if *decoded_bytes > ANIMATION_BYTES_MAX {
        return Err(error(GIFError::LimitExceeded(GIFLimit::AnimationBytes(
            ANIMATION_BYTES_MAX,
        ))));
    }

    Ok(())
}

fn validate_frame(
    frame: &::gif::Frame<'_>,
    canvas_width: usize,
    canvas_height: usize,
) -> Result<(usize, usize, usize, usize)> {
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

    let left = usize::from(frame.left);
    let top = usize::from(frame.top);
    let right = left
        .checked_add(frame_width)
        .ok_or_else(|| error(GIFError::Output("GIF frame bounds overflowed")))?;
    let bottom = top
        .checked_add(frame_height)
        .ok_or_else(|| error(GIFError::Output("GIF frame bounds overflowed")))?;
    if right > canvas_width || bottom > canvas_height {
        return Err(error(GIFError::Output(
            "GIF frame extends beyond the logical screen",
        )));
    }

    Ok((frame_width, frame_height, left, top))
}

fn composite_first_frame(
    frame: &::gif::Frame<'_>,
    canvas_size: usize,
    canvas_width: usize,
    frame_width: usize,
    frame_height: usize,
    left: usize,
    top: usize,
) -> Vec<u8> {
    let mut rgba = vec![0; canvas_size];
    for row in 0..frame_height {
        let source_start = row * frame_width * 4;
        let target_start = ((top + row) * canvas_width + left) * 4;
        let source = &frame.buffer[source_start..source_start + frame_width * 4];
        rgba[target_start..target_start + frame_width * 4].copy_from_slice(source);
    }
    rgba
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use ::gif::{Encoder, Frame};

    use super::{FRAMES_MAX, decode};

    #[test]
    fn rejects_animation_frame_counts_above_the_limit() {
        let mut bytes = Vec::new();
        {
            let mut encoder = Encoder::new(&mut bytes, 1, 1, &[0, 0, 0]).unwrap();
            let frame = Frame {
                width: 1,
                height: 1,
                buffer: Cow::Borrowed(&[0]),
                ..Frame::default()
            };
            for _ in 0..=FRAMES_MAX {
                encoder.write_frame(&frame).unwrap();
            }
        }

        let error = decode(bytes).unwrap_err();

        assert!(error.to_string().contains("10000-frame limit"));
    }
}
