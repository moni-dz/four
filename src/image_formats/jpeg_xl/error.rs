//! Classifies JPEG XL decoding failures.

use std::fmt;

use exn::ErrorExt;

/// An exception carrying a [`JPEGXLError`] and its propagation frames.
pub type Error = exn::Exn<JPEGXLError>;

/// The result returned by JPEG XL decoder operations.
pub type Result<T> = exn::Result<T, JPEGXLError>;

/// A JPEG XL decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JPEGXLError {
    /// The underlying JPEG XL codec rejected the input.
    Codec(String),
    /// Input exceeded an explicit decoder resource limit.
    LimitExceeded(JPEGXLLimit),
    /// The decoded image did not contain a displayable keyframe.
    NoFrame,
    /// The codec produced an inconsistent rendered buffer.
    Output(&'static str),
    /// The input does not begin with either JPEG XL signature.
    Signature,
}

/// A bounded resource whose configured maximum was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGXLLimit {
    /// Maximum accepted width or height in pixels.
    Dimensions(u32),
    /// Maximum memory tracked while decoding the codestream.
    DecoderMemory(usize),
    /// Maximum accepted decoded pixel count.
    Pixels(u64),
}

impl fmt::Display for JPEGXLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "JPEG XL codec error: {detail}"),
            Self::LimitExceeded(JPEGXLLimit::Dimensions(max)) => {
                write!(f, "JPEG XL dimensions exceed the {max}-pixel limit")
            }
            Self::LimitExceeded(JPEGXLLimit::DecoderMemory(max)) => write!(
                f,
                "JPEG XL decoding exceeds the {} MiB memory limit",
                max / 1024 / 1024
            ),
            Self::LimitExceeded(JPEGXLLimit::Pixels(max)) => write!(
                f,
                "JPEG XL pixel count exceeds the {}-megapixel limit",
                max / 1024 / 1024
            ),
            Self::NoFrame => f.write_str("JPEG XL image contains no displayable keyframe"),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("expected a JPEG XL codestream or container signature"),
        }
    }
}

impl std::error::Error for JPEGXLError {}

/// Raises a leaf error at its validation site.
#[track_caller]
pub(super) fn error(error: JPEGXLError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "a JPEG XL error must have a useful display message"
    );
    error.raise()
}
