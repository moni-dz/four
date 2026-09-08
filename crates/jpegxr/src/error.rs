//! Defines JPEG XR parser and decoder errors.

use std::backtrace::Backtrace;
use std::sync::Arc;

/// Result returned by JPEG XR operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A JPEG XR failure with its byte position.
#[derive(Debug, thiserror::Error)]
#[error("JPEG XR error at byte {offset}: {kind}")]
pub struct Error {
    kind: ErrorKind,
    offset: usize,
    // `Backtrace` implements neither `Clone` nor `PartialEq`/`Eq`, so it is kept behind an `Arc`
    // (cloning shares the captured frames instead of re-unwinding) and excluded from equality,
    // which compares only the classification and byte position below.
    backtrace: Arc<Backtrace>,
}

impl Clone for Error {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            offset: self.offset,
            backtrace: Arc::clone(&self.backtrace),
        }
    }
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.offset == other.offset
    }
}

impl Eq for Error {}

impl Error {
    // Keep backtrace capture and allocation out of the decoder's successful paths.
    #[cold]
    #[inline(never)]
    pub(crate) fn new(kind: ErrorKind, offset: usize) -> Self {
        Self {
            kind,
            offset,
            backtrace: Arc::new(Backtrace::capture()),
        }
    }

    /// Returns the byte position where decoding failed.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the captured backtrace.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }

    /// Returns whether input ended before a complete syntax element was available.
    #[must_use]
    pub const fn is_unexpected_eof(&self) -> bool {
        matches!(self.kind, ErrorKind::UnexpectedEOF)
    }

    /// Returns whether the file header or codestream is missing its required signature.
    #[must_use]
    pub const fn is_invalid_signature(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidSignature)
    }

    /// Returns whether the codestream or pixel format is unsupported.
    #[must_use]
    pub const fn is_unsupported(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::Unsupported(_) | ErrorKind::UnsupportedPixelFormat(_)
        )
    }

    /// Returns whether a declared image dimension (width or height) exceeds a decoder bound.
    #[must_use]
    pub fn is_dimension_limit_exceeded(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::LimitExceeded("image dimension" | "image width" | "image height")
        )
    }

    /// Returns whether the declared pixel count exceeds a decoder bound.
    #[must_use]
    pub fn is_pixel_count_limit_exceeded(&self) -> bool {
        matches!(self.kind, ErrorKind::LimitExceeded("pixel count"))
    }

    /// Returns whether any declared resource size exceeds a decoder bound.
    ///
    /// Includes dimension, pixel-count, tag-payload, tile-count, and output-buffer limits.
    #[must_use]
    pub const fn is_limit_exceeded(&self) -> bool {
        matches!(self.kind, ErrorKind::LimitExceeded(_))
    }

    /// Returns whether the tag container and the embedded codestream disagree about the image
    /// they describe.
    #[must_use]
    pub const fn is_container_mismatch(&self) -> bool {
        matches!(self.kind, ErrorKind::ContainerMismatch(_))
    }

    /// Returns whether a tag container entry is malformed: an invalid offset, element type, or
    /// tag value, an unsorted or missing tag, or too many directory entries.
    #[must_use]
    pub const fn is_invalid_tag_container(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::InvalidOffset(_)
                | ErrorKind::TooManyEntries
                | ErrorKind::UnsortedTags
                | ErrorKind::InvalidElementType(_)
                | ErrorKind::MissingTag(_)
                | ErrorKind::InvalidTag(_, _)
        )
    }

    /// Returns whether the codestream violates a T.832 syntax requirement not covered by a more
    /// specific classification above.
    #[must_use]
    pub const fn is_invalid_codestream(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidCodestream(_))
    }
}

/// Category of a JPEG XR failure.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum ErrorKind {
    /// Input ended before a complete syntax element was available.
    #[error("unexpected end of input")]
    UnexpectedEOF,

    /// File header does not contain the JPEG XR signature.
    #[error("invalid JPEG XR signature")]
    InvalidSignature,

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
