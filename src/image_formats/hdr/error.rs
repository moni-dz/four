//! Defines structured HDR analysis and tone-mapping failures.

use std::fmt;

use exn::ErrorExt;

/// An HDR source validation or normalization failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HDRError {
    /// Input exceeded an explicit resource bound.
    LimitExceeded(HDRLimit),
    /// Pixels or dimensions violate the output contract.
    Output(&'static str),
}

/// A bounded HDR resource whose configured maximum was exceeded.
///
/// `actual` is `None` when the bound is reported without the measured value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HDRLimit {
    /// Maximum accepted width or height in pixels.
    Dimensions {
        /// The rejected width or height, when known.
        actual: Option<u32>,
        /// The configured maximum.
        max: u32,
    },
    /// Maximum accepted source-buffer size in bytes.
    SourceBufferBytes {
        /// The rejected buffer size, when known.
        actual: Option<usize>,
        /// The configured maximum.
        max: usize,
    },
    /// Maximum accepted pixel count.
    Pixels {
        /// The rejected pixel count, when known.
        actual: Option<u64>,
        /// The configured maximum.
        max: u64,
    },
}

impl fmt::Display for HDRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
        }
    }
}

impl std::error::Error for HDRError {}

/// An error with propagation frames.
pub type Error = exn::Exn<HDRError>;
/// Result returned by HDR normalization operations.
pub type Result<T> = exn::Result<T, HDRError>;

/// Creates a leaf error at its validation site.
#[track_caller]
pub(super) fn error(error: HDRError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "an HDR error must have a useful display message"
    );
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: HDRLimit) -> fmt::Result {
    match limit {
        HDRLimit::Dimensions {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "HDR dimension {actual} exceeds the {max}-pixel limit"
        ),
        HDRLimit::Dimensions { actual: None, max } => {
            write!(formatter, "HDR dimensions exceed the {max}-pixel limit")
        }
        HDRLimit::SourceBufferBytes {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "HDR source buffer of {} MiB exceeds the {} MiB limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        HDRLimit::SourceBufferBytes { actual: None, max } => write!(
            formatter,
            "HDR source pixels exceed the {} MiB buffer limit",
            max / 1024 / 1024
        ),
        HDRLimit::Pixels {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "HDR pixel count of {} megapixels exceeds the {}-megapixel limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        HDRLimit::Pixels { actual: None, max } => write!(
            formatter,
            "HDR pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
    }
}
