//! Defines structured JPEG XR decoder failures.

use std::fmt;

use exn::ErrorExt;

/// A JPEG XR decoding or normalization failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JPEGXRError {
    /// The underlying JPEG XR codec rejected the input.
    Codec(jpegxr::Error),
    /// Input exceeded an explicit decoder resource bound.
    LimitExceeded(JPEGXRLimit),
    /// Decoded pixels or dimensions violate the output contract.
    Output(&'static str),
    /// The input does not begin with a JPEG XR file signature.
    Signature,
    /// Source pixel representation is unsupported for RGBA8 normalization.
    Unsupported(String),
}

/// A bounded JPEG XR resource whose configured maximum was exceeded.
///
/// `actual` is `None` when the codec reports a bound without the measured value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGXRLimit {
    /// Maximum accepted width or height in pixels.
    Dimensions {
        /// The rejected width or height, when known.
        actual: Option<u32>,
        /// The configured maximum.
        max: u32,
    },
    /// Maximum accepted decoded source-buffer size in bytes.
    SourceBufferBytes {
        /// The rejected buffer size, when known.
        actual: Option<usize>,
        /// The configured maximum.
        max: usize,
    },
    /// Maximum accepted decoded pixel count.
    Pixels {
        /// The rejected pixel count, when known.
        actual: Option<u64>,
        /// The configured maximum.
        max: u64,
    },
}

impl fmt::Display for JPEGXRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(source) => write!(f, "JPEG XR codec error: {source}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("expected a JPEG XR file signature"),
            Self::Unsupported(detail) => {
                write!(f, "unsupported JPEG XR pixel format: {detail}")
            }
        }
    }
}

format_error_boilerplate!("JPEG XR", JPEGXRError, source = |source| Some(source));

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: JPEGXRLimit) -> fmt::Result {
    match limit {
        JPEGXRLimit::Dimensions {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "JPEG XR dimension {actual} exceeds the {max}-pixel limit"
        ),
        JPEGXRLimit::Dimensions { actual: None, max } => {
            write!(formatter, "JPEG XR dimensions exceed the {max}-pixel limit")
        }
        JPEGXRLimit::SourceBufferBytes {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "JPEG XR source buffer of {} MiB exceeds the {} MiB limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        JPEGXRLimit::SourceBufferBytes { actual: None, max } => write!(
            formatter,
            "JPEG XR source pixels exceed the {} MiB buffer limit",
            max / 1024 / 1024
        ),
        JPEGXRLimit::Pixels {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "JPEG XR pixel count of {} megapixels exceeds the {}-megapixel limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        JPEGXRLimit::Pixels { actual: None, max } => write!(
            formatter,
            "JPEG XR pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
    }
}
