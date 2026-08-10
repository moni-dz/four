//! Defines JPEG XR parser and decoder errors.

/// Result returned by JPEG XR operations.
pub type Result<T> = std::result::Result<T, Error>;

/// A JPEG XR failure with its byte position.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("JPEG XR error at byte {offset}: {kind}")]
pub struct Error {
    kind: ErrorKind,
    offset: usize,
}

impl Error {
    pub(crate) const fn new(kind: ErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    /// Returns the failure category.
    #[must_use]
    pub const fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Returns the byte position where decoding failed.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

/// Category of a JPEG XR failure.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ErrorKind {
    /// Input ended before a complete syntax element was available.
    #[error("unexpected end of input")]
    UnexpectedEof,

    /// File header does not contain the JPEG XR signature.
    #[error("invalid JPEG XR signature")]
    InvalidSignature,
    /// File uses an unsupported tag-container version.
    #[error("unsupported container version {0}")]
    UnsupportedVersion(u8),

    /// An offset is odd, out of range, or overlaps required header data.
    #[error("invalid {0} offset")]
    InvalidOffset(&'static str),
    /// Image file directory contains too many entries.
    #[error("too many image directory entries")]
    TooManyEntries,
    /// Image file directory tags are not strictly increasing.
    #[error("image directory tags are not sorted")]
    UnsortedTags,

    /// An element type is reserved or unknown.
    #[error("invalid element type {0}")]
    InvalidElementType(u16),
    /// A required tag is absent.
    #[error("missing required tag 0x{0:04X}")]
    MissingTag(u16),
    /// A tag has a forbidden type, count, or value.
    #[error("invalid tag 0x{0:04X}: {1}")]
    InvalidTag(u16, &'static str),

    /// Pixel-format identifier is not defined by T.832 Table A.6.
    #[error("unsupported pixel format {0:02X?}")]
    UnsupportedPixelFormat([u8; 16]),

    /// Codestream syntax violates a T.832 requirement.
    #[error("invalid codestream: {0}")]
    InvalidCodestream(&'static str),
    /// Codestream feature is valid but not implemented by this decoder.
    #[error("unsupported JPEG XR feature: {0}")]
    Unsupported(&'static str),

    /// Declared resource size exceeds a decoder bound.
    #[error("{0} limit exceeded")]
    LimitExceeded(&'static str),

    /// Container tags disagree with the embedded codestream.
    #[error("container and codestream disagree: {0}")]
    ContainerMismatch(&'static str),
}
