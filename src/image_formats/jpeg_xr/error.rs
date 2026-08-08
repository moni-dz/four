//! Defines structured JPEG XR decoder failures.

use std::fmt;

use exn::ErrorExt;

/// An exception carrying a [`JPEGXRError`] and its propagation frames.
pub type Error = exn::Exn<JPEGXRError>;

/// The result returned by JPEG XR decoder operations.
pub type Result<T> = exn::Result<T, JPEGXRError>;

/// A JPEG XR decoding or normalization failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JPEGXRError {
    /// The underlying JPEG XR codec rejected the input.
    Codec(String),
    /// Input exceeded an explicit decoder resource bound.
    LimitExceeded(JPEGXRLimit),
    /// Decoded pixels or dimensions violate the output contract.
    Output(&'static str),
    /// The input does not begin with a JPEG XR file signature.
    Signature,
    /// The source pixel representation cannot be normalized to RGBA8.
    Unsupported(String),
}

/// A bounded JPEG XR resource whose configured maximum was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGXRLimit {
    /// Maximum accepted width or height in pixels.
    Dimensions(u32),
    /// Maximum accepted decoded source-buffer size in bytes.
    SourceBufferBytes(usize),
    /// Maximum accepted decoded pixel count.
    Pixels(u64),
}

impl fmt::Display for JPEGXRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "JPEG XR codec error: {detail}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("expected a JPEG XR file signature"),
            Self::Unsupported(detail) => {
                write!(f, "unsupported JPEG XR pixel format: {detail}")
            }
        }
    }
}

impl std::error::Error for JPEGXRError {}

#[track_caller]
pub(super) fn error(error: JPEGXRError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "a JPEG XR error must have a useful display message"
    );
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: JPEGXRLimit) -> fmt::Result {
    match limit {
        JPEGXRLimit::Dimensions(max) => {
            write!(formatter, "JPEG XR dimensions exceed the {max}-pixel limit")
        }
        JPEGXRLimit::SourceBufferBytes(max) => write!(
            formatter,
            "JPEG XR source pixels exceed the {} MiB buffer limit",
            max / 1024 / 1024
        ),
        JPEGXRLimit::Pixels(max) => write!(
            formatter,
            "JPEG XR pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
    }
}
