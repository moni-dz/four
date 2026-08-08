//! Classifies TIFF decoding failures.

use std::fmt;

use exn::ErrorExt;

/// An exception carrying a [`TIFFError`] and its propagation frames.
pub type Error = exn::Exn<TIFFError>;

/// The result returned by TIFF decoder operations.
pub type Result<T> = exn::Result<T, TIFFError>;

/// A TIFF decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TIFFError {
    /// The underlying TIFF codec rejected the datastream.
    Codec(String),
    /// Input exceeded an explicit decoder resource limit.
    LimitExceeded(TIFFLimit),
    /// Decoded samples violated the codec adapter's output contract.
    Output(&'static str),
    /// The input does not begin with a classic TIFF or `BigTIFF` signature.
    Signature,
    /// The TIFF uses a color or sample representation not mapped to RGBA8.
    Unsupported(String),
}

/// A bounded TIFF resource whose configured maximum was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TIFFLimit {
    /// Maximum decoded buffer size available to the codec.
    CodecBufferBytes(usize),
    /// Maximum accepted width or height in pixels.
    Dimensions(u32),
    /// Maximum accepted decoded pixel count.
    Pixels(u64),
}

impl fmt::Display for TIFFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "TIFF codec error: {detail}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("input does not begin with a TIFF signature"),
            Self::Unsupported(detail) => write!(f, "unsupported TIFF representation: {detail}"),
        }
    }
}

impl std::error::Error for TIFFError {}

/// Raises a leaf error at its validation site.
#[track_caller]
pub(super) fn error(error: TIFFError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "a TIFF error must have a useful display message"
    );
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: TIFFLimit) -> fmt::Result {
    match limit {
        TIFFLimit::CodecBufferBytes(max) => write!(
            formatter,
            "TIFF decoded buffer exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        TIFFLimit::Dimensions(max) => {
            write!(formatter, "TIFF dimensions exceed the {max}-pixel limit")
        }
        TIFFLimit::Pixels(max) => write!(
            formatter,
            "TIFF pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
    }
}
