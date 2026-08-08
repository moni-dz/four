//! Classifies PNG decoding failures.

use std::fmt;

use exn::ErrorExt;

/// An exception carrying a [`PNGError`] and its propagation frames.
pub type Error = exn::Exn<PNGError>;

/// The result returned by PNG decoder operations.
pub type Result<T> = exn::Result<T, PNGError>;

/// A PNG decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PNGError {
    /// The underlying PNG codec rejected the datastream.
    Codec(String),
    /// Input exceeded an explicit decoder resource limit.
    LimitExceeded(PNGLimit),
    /// Decoded samples violated the codec adapter's output contract.
    Output(&'static str),
    /// The input does not begin with the PNG signature.
    Signature,
}

/// A bounded PNG resource whose configured maximum was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PNGLimit {
    /// Maximum memory available to the underlying codec.
    CodecMemory(usize),
    /// Maximum decoded byte count accepted from the codec.
    DecodedBytes(usize),
    /// Maximum accepted width or height in pixels.
    Dimensions(u32),
    /// Maximum accepted decoded pixel count.
    Pixels(u64),
}

impl fmt::Display for PNGError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "PNG codec error: {detail}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("input does not begin with the PNG signature"),
        }
    }
}

impl std::error::Error for PNGError {}

/// Raises a leaf error at its validation site.
#[track_caller]
pub(super) fn error(error: PNGError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "a PNG error must have a useful display message"
    );
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: PNGLimit) -> fmt::Result {
    match limit {
        PNGLimit::CodecMemory(max) => write!(
            formatter,
            "PNG codec memory exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        PNGLimit::DecodedBytes(max) => write!(
            formatter,
            "PNG decoded output exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        PNGLimit::Dimensions(max) => {
            write!(formatter, "PNG dimensions exceed the {max}-pixel limit")
        }
        PNGLimit::Pixels(max) => write!(
            formatter,
            "PNG pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
    }
}
