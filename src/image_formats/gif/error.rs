//! Classifies GIF decoding failures.

use std::fmt;

use exn::ErrorExt;

/// An exception carrying a [`GIFError`] and its propagation frames.
pub type Error = exn::Exn<GIFError>;

/// The result returned by GIF decoder operations.
pub type Result<T> = exn::Result<T, GIFError>;

/// A GIF decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GIFError {
    /// The underlying GIF codec rejected the datastream.
    Codec(String),
    /// Input exceeded an explicit decoder resource limit.
    LimitExceeded(GIFLimit),
    /// The datastream contains no image frame.
    NoFrame,
    /// Decoded samples violated the codec adapter's output contract.
    Output(&'static str),
    /// The input does not begin with a supported GIF signature.
    Signature,
}

/// A bounded GIF resource whose configured maximum was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GIFLimit {
    /// Maximum total decoded bytes across full-canvas animation frames.
    AnimationBytes(u64),
    /// Maximum decoded bytes available to one codec frame.
    CodecFrameBytes(u64),
    /// Maximum accepted width or height in pixels.
    Dimensions(u32),
    /// Maximum accepted animation frame count.
    Frames(u64),
    /// Maximum accepted logical-screen pixel count.
    Pixels(u64),
}

impl fmt::Display for GIFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "GIF codec error: {detail}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::NoFrame => f.write_str("GIF datastream contains no image frame"),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("input does not begin with GIF87a or GIF89a"),
        }
    }
}

impl std::error::Error for GIFError {}

/// Raises a leaf error at its validation site.
#[track_caller]
pub(super) fn error(error: GIFError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "a GIF error must have a useful display message"
    );
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: GIFLimit) -> fmt::Result {
    match limit {
        GIFLimit::AnimationBytes(max) => write!(
            formatter,
            "GIF animation output exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        GIFLimit::CodecFrameBytes(max) => write!(
            formatter,
            "GIF frame output exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        GIFLimit::Dimensions(max) => {
            write!(formatter, "GIF dimensions exceed the {max}-pixel limit")
        }
        GIFLimit::Frames(max) => write!(formatter, "GIF animation exceeds the {max}-frame limit"),
        GIFLimit::Pixels(max) => write!(
            formatter,
            "GIF pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
    }
}
