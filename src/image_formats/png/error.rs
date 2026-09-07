//! Classifies PNG decoding failures.

use std::fmt;

use exn::ErrorExt;

/// A PNG decoding failure.
#[derive(Debug)]
pub enum PNGError {
    /// The underlying PNG codec rejected the datastream.
    Codec(Box<dyn std::error::Error + Send + Sync + 'static>),
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
    ///
    /// The `png` crate's `DecodingError::LimitsExceeded` carries no observed byte count, so
    /// only the configured maximum can be reported here.
    CodecMemory(usize),
    /// Maximum decoded byte count accepted from the codec.
    DecodedBytes {
        /// The decoded byte count that exceeded `max`.
        actual: usize,
        /// The configured maximum decoded byte count.
        max: usize,
    },
    /// Maximum accepted width or height in pixels.
    Dimensions {
        /// The observed width that exceeded `max`, in pixels.
        actual_width: u32,
        /// The observed height that exceeded `max`, in pixels.
        actual_height: u32,
        /// The configured maximum width or height, in pixels.
        max: u32,
    },
    /// Maximum accepted decoded pixel count.
    Pixels {
        /// The pixel count that exceeded `max`.
        actual: u64,
        /// The configured maximum pixel count.
        max: u64,
    },
}

impl fmt::Display for PNGError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(source) => write!(f, "PNG codec error: {source}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("input does not begin with the PNG signature"),
        }
    }
}

format_error_boilerplate!("PNG", PNGError, source = |source| Some(source.as_ref()));

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: PNGLimit) -> fmt::Result {
    match limit {
        PNGLimit::CodecMemory(max) => write!(
            formatter,
            "PNG codec memory exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        PNGLimit::DecodedBytes { actual, max } => write!(
            formatter,
            "PNG decoded output of {} MiB exceeds the {} MiB limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        PNGLimit::Dimensions {
            actual_width,
            actual_height,
            max,
        } => write!(
            formatter,
            "PNG dimensions {actual_width}x{actual_height} exceed the {max}-pixel limit"
        ),
        PNGLimit::Pixels { actual, max } => write!(
            formatter,
            "PNG pixel count of {} megapixels exceeds the {}-megapixel limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
    }
}
