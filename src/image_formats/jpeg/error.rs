//! Classifies decoder failures without allocating on ordinary error paths.

use std::fmt;

use exn::ErrorExt;

/// An exception carrying a [`JPEGError`] and its propagation frames.
pub type Error = exn::Exn<JPEGError>;

/// The result returned by JPEG decoder operations.
pub type Result<T> = exn::Result<T, JPEGError>;

/// A decoder failure classified by the JPEG grammar section that rejected the input.
///
/// Keeping the classification in the type lets callers choose a recovery policy without parsing
/// display text. Static detail strings retain precise diagnostics without allocating on an error
/// path, while values that callers may need are stored in dedicated variants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JPEGError {
    /// An integer calculation exceeded its representable range.
    ArithmeticOverflow(&'static str),
    /// The Huffman JPEG codec rejected the datastream.
    Codec(String),
    /// Entropy-coded data violated the selected coding process.
    Entropy(&'static str),
    /// A byte that should introduce a marker was not `0xFF`.
    ExpectedMarkerPrefix {
        /// The parser operation that expected the marker.
        context: &'static str,
        /// The byte found instead of `0xFF`.
        found: u8,
    },
    /// A frame header or frame invariant is invalid.
    Frame(&'static str),
    /// Input exceeded an explicit decoder resource limit.
    LimitExceeded(JPEGLimit),
    /// The marker stream is malformed.
    Marker(&'static str),
    /// An entropy restart marker did not match the expected sequence.
    RestartMarkerMismatch {
        /// The expected marker code without its `0xFF` prefix.
        expected: u8,
        /// The marker code found without its `0xFF` prefix.
        found: u8,
    },
    /// A scan header or scan progression invariant is invalid.
    Scan(&'static str),
    /// A length-delimited marker segment is malformed.
    Segment(&'static str),
    /// A decoder table is malformed or missing.
    Table(JPEGTableKind, &'static str),
    /// The input ended before a required value could be read.
    UnexpectedEnd(&'static str),
    /// A valid marker appeared in an invalid parser phase.
    UnexpectedMarker {
        /// The marker or phase expected by the parser.
        context: &'static str,
        /// The marker code found without its `0xFF` prefix.
        found: u8,
    },
    /// The input uses a valid JPEG feature outside this decoder's scope.
    Unsupported(UnsupportedJPEG),
}

/// A bounded resource whose configured maximum was exceeded by the input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGLimit {
    /// Maximum accepted width or height in pixels.
    Dimensions(u32),
    /// Maximum accepted number of data units in one MCU.
    FrameDataUnits(u8),
    /// Maximum accepted decoded pixel count.
    Pixels(u64),
    /// Maximum accepted progressive coefficient storage in bytes.
    ProgressiveCoefficientBytes(u64),
    /// Maximum accepted number of scans.
    Scans(u32),
}

/// Identifies the decoder table associated with a [`JPEGError::Table`] failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGTableKind {
    /// Arithmetic conditioning table.
    ArithmeticConditioning,
    /// Quantization table.
    Quantization,
}

/// A valid JPEG feature that this deliberately small decoder does not implement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedJPEG {
    /// Adobe APP14 requested an unsupported color transform.
    AdobeColorTransform(u8),
    /// The frame contains an unsupported number of image components.
    ComponentCount(u8),
    /// The start-of-frame marker selects an unsupported coding process.
    FrameType(u8),
    /// The marker is valid JPEG syntax but is not implemented.
    Marker(u8),
    /// The frame uses a sample precision other than eight bits.
    SamplePrecision(u8),
}

impl fmt::Display for JPEGError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "JPEG codec error: {detail}"),
            Self::ArithmeticOverflow(detail)
            | Self::Entropy(detail)
            | Self::Frame(detail)
            | Self::Marker(detail)
            | Self::Scan(detail)
            | Self::Segment(detail)
            | Self::Table(_, detail)
            | Self::UnexpectedEnd(detail) => f.write_str(detail),
            Self::ExpectedMarkerPrefix { context, found } => write!(
                f,
                "expected a JPEG marker FF prefix {context}, found {found:02X}"
            ),
            Self::LimitExceeded(limit) => write_limit_error(f, *limit),
            Self::RestartMarkerMismatch { expected, found } => write!(
                f,
                "expected restart marker FF{expected:02X}, found FF{found:02X}"
            ),
            Self::UnexpectedMarker { context, found } => {
                write!(f, "expected {context}, found FF{found:02X}")
            }
            Self::Unsupported(feature) => write_unsupported_error(f, *feature),
        }
    }
}

impl std::error::Error for JPEGError {}

/// Raises a leaf error at its validation site so `exn` records the useful source location.
#[track_caller]
pub(super) fn error(error: JPEGError) -> Error {
    invariant!(
        !error.to_string().is_empty(),
        "a JPEG error must have a useful display message"
    );
    invariant!(
        size_of::<JPEGError>() > 0,
        "JPEGError must remain inhabited"
    );
    error.raise()
}

fn write_limit_error(formatter: &mut fmt::Formatter<'_>, limit: JPEGLimit) -> fmt::Result {
    match limit {
        JPEGLimit::Dimensions(max) => {
            write!(formatter, "JPEG dimensions exceed the {max}-pixel limit")
        }
        JPEGLimit::FrameDataUnits(max) => {
            write!(formatter, "frame has more than {max} data units per MCU")
        }
        JPEGLimit::Pixels(max) => write!(
            formatter,
            "JPEG pixel count exceeds the {}-megapixel limit",
            max / 1024 / 1024
        ),
        JPEGLimit::ProgressiveCoefficientBytes(max) => write!(
            formatter,
            "progressive coefficient storage exceeds the {} MiB limit",
            max / 1024 / 1024
        ),
        JPEGLimit::Scans(max) => write!(formatter, "JPEG contains more than {max} scans"),
    }
}

fn write_unsupported_error(
    formatter: &mut fmt::Formatter<'_>,
    feature: UnsupportedJPEG,
) -> fmt::Result {
    match feature {
        UnsupportedJPEG::AdobeColorTransform(value) => write!(
            formatter,
            "Adobe JPEG color transform {value} is not supported"
        ),
        UnsupportedJPEG::ComponentCount(count) => write!(
            formatter,
            "JPEG component count {count} is unsupported; expected one or three"
        ),
        UnsupportedJPEG::FrameType(marker) => {
            write!(formatter, "JPEG frame type FF{marker:02X} is not supported")
        }
        UnsupportedJPEG::Marker(marker) => {
            write!(formatter, "unsupported JPEG marker FF{marker:02X}")
        }
        UnsupportedJPEG::SamplePrecision(precision) => write!(
            formatter,
            "JPEG sample precision {precision} is unsupported; expected 8"
        ),
    }
}
