//! Classifies GIF decoding failures.

use std::fmt;

use exn::ErrorExt;

/// A GIF decoding failure.
#[derive(Debug)]
pub enum GIFError {
    /// The underlying GIF codec rejected the datastream.
    Codec(Box<dyn std::error::Error + Send + Sync + 'static>),
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
    AnimationBytes {
        /// The decoded byte count that exceeded `max`.
        actual: u64,
        /// The configured maximum decoded byte count.
        max: u64,
    },
    /// Maximum decoded bytes available to one codec frame.
    ///
    /// The configured maximum; the codec's memory errors provide no observed byte count.
    CodecFrameBytes(u64),
    /// Maximum accepted width or height in pixels.
    Dimensions {
        /// The observed width that exceeded `max`, in pixels.
        actual_width: u32,
        /// The observed height that exceeded `max`, in pixels.
        actual_height: u32,
        /// The configured maximum width or height, in pixels.
        max: u32,
    },
    /// Maximum accepted animation frame count.
    Frames {
        /// The frame count that exceeded `max`.
        actual: u64,
        /// The configured maximum frame count.
        max: u64,
    },
    /// Maximum accepted logical-screen pixel count.
    Pixels {
        /// The pixel count that exceeded `max`.
        actual: u64,
        /// The configured maximum pixel count.
        max: u64,
    },
}

impl fmt::Display for GIFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(source) => write!(f, "GIF codec error: {source}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::NoFrame => f.write_str("GIF datastream contains no image frame"),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("input does not begin with GIF87a or GIF89a"),
        }
    }
}

format_error_boilerplate!("GIF", GIFError, source = |source| Some(source.as_ref()));

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: GIFLimit) -> fmt::Result {
    match limit {
        GIFLimit::AnimationBytes { actual, max } => write!(
            formatter,
            "GIF animation output of {} MiB exceeds the {} MiB limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        GIFLimit::CodecFrameBytes(max) => write!(
            formatter,
            "GIF frame output exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        GIFLimit::Dimensions {
            actual_width,
            actual_height,
            max,
        } => write!(
            formatter,
            "GIF dimensions {actual_width}x{actual_height} exceed the {max}-pixel limit"
        ),
        GIFLimit::Frames { actual, max } => write!(
            formatter,
            "GIF animation of {actual} frames exceeds the {max}-frame limit"
        ),
        GIFLimit::Pixels { actual, max } => write!(
            formatter,
            "GIF pixel count of {} megapixels exceeds the {}-megapixel limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
    }
}
