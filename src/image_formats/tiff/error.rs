//! Classifies TIFF decoding failures.

use std::fmt;

use exn::ErrorExt;

/// A TIFF decoding failure.
#[derive(Debug)]
pub enum TIFFError {
    /// The underlying TIFF codec rejected the datastream.
    Codec(Box<dyn std::error::Error + Send + Sync + 'static>),
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
    ///
    /// `actual` is the raw sample buffer size computed before decoding. It is `None` when the
    /// codec's own `TIFFError::LimitsExceeded` fires on an internal allocation (for example
    /// while reading IFD tags) before that size is known, since that variant carries no
    /// observed byte count.
    CodecBufferBytes {
        /// The raw buffer size that exceeded `max`, when known.
        actual: Option<u64>,
        /// The configured maximum decoded buffer size.
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

impl fmt::Display for TIFFError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(source) => write!(f, "TIFF codec error: {source}"),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::Output(detail) => f.write_str(detail),
            Self::Signature => f.write_str("input does not begin with a TIFF signature"),
            Self::Unsupported(detail) => write!(f, "unsupported TIFF representation: {detail}"),
        }
    }
}

format_error_boilerplate!("TIFF", TIFFError, source = |source| Some(source.as_ref()));

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: TIFFLimit) -> fmt::Result {
    match limit {
        TIFFLimit::CodecBufferBytes {
            actual: Some(actual),
            max,
        } => write!(
            formatter,
            "TIFF decoded buffer of {} MiB exceeds the {} MiB limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
        TIFFLimit::CodecBufferBytes { actual: None, max } => write!(
            formatter,
            "TIFF decoded buffer exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        TIFFLimit::Dimensions {
            actual_width,
            actual_height,
            max,
        } => write!(
            formatter,
            "TIFF dimensions {actual_width}x{actual_height} exceed the {max}-pixel limit"
        ),
        TIFFLimit::Pixels { actual, max } => write!(
            formatter,
            "TIFF pixel count of {} megapixels exceeds the {}-megapixel limit",
            actual / 1024 / 1024,
            max / 1024 / 1024
        ),
    }
}
