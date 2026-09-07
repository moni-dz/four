//! Classifies JPEG XL decoding failures.

use std::fmt;

use exn::ErrorExt;

/// A JPEG XL decoding failure.
#[derive(Debug)]
pub enum JPEGXLError {
    /// The underlying JPEG XL codec rejected the input.
    Codec(Box<dyn std::error::Error + Send + Sync + 'static>),
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
    Dimensions {
        /// The observed width that exceeded `max`, in pixels.
        actual_width: u32,
        /// The observed height that exceeded `max`, in pixels.
        actual_height: u32,
        /// The configured maximum width or height, in pixels.
        max: u32,
    },
    /// Maximum memory tracked while decoding the codestream.
    ///
    /// `jxl-oxide`'s allocation-tracker error (`jxl_grid::OutOfMemory`) does carry the failed
    /// allocation's own byte count via `OutOfMemory::bytes()`, but that type is reachable only
    /// by adding `jxl_grid` as a direct dependency: `jxl-oxide` re-exports `AllocTracker` but
    /// not the error type it raises. Only the configured maximum is reported here.
    DecoderMemory(usize),
    /// Maximum accepted decoded pixel count.
    Pixels {
        /// The pixel count that exceeded `max`.
        actual: u64,
        /// The configured maximum pixel count.
        max: u64,
    },
}

impl fmt::Display for JPEGXLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(source) => write!(f, "JPEG XL codec error: {source}"),
            Self::LimitExceeded(JPEGXLLimit::Dimensions {
                actual_width,
                actual_height,
                max,
            }) => write!(
                f,
                "JPEG XL dimensions {actual_width}x{actual_height} exceed the {max}-pixel limit"
            ),
            Self::LimitExceeded(JPEGXLLimit::DecoderMemory(max)) => write!(
                f,
                "JPEG XL decoding exceeds the {} MiB memory limit",
                max / 1024 / 1024
            ),
            Self::LimitExceeded(JPEGXLLimit::Pixels { actual, max }) => write!(
                f,
                "JPEG XL pixel count of {} megapixels exceeds the {}-megapixel limit",
                actual / 1024 / 1024,
                max / 1024 / 1024
            ),
            Self::NoFrame => f.write_str("JPEG XL image contains no displayable keyframe"),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("expected a JPEG XL codestream or container signature"),
        }
    }
}

format_error_boilerplate!(
    "JPEG XL",
    JPEGXLError,
    source = |source| Some(source.as_ref())
);
