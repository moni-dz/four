//! Decodes bounded 8-bit DCT JPEG images without performing I/O.
//!
//! Baseline and progressive Huffman streams are decoded by `zune-jpeg`. Arithmetic-coded
//! sequential and progressive streams are routed to the handwritten JPEG parser because the codec
//! does not implement arithmetic entropy coding. Both paths share dimension and pixel limits.

mod arithmetic;
mod error;
mod idct;
mod model;
mod parser;
mod reader;
mod scan;

use error::error;
use model::{
    CodingProcess, ColorTransform, Frame, FrameComponent, dequantize_block, validate_dimensions,
    write_block,
};
use scan::{ScanComponent, ScanHeader, parse_frame_components, parse_scan_header};

use super::{DIMENSION_MAX, DecodedImage, PIXELS_MAX};

use zune_jpeg::JpegDecoder;
use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

pub use error::{Error, JPEGError, JPEGLimit, JPEGTableKind, Result, UnsupportedJPEG};

/// The start-of-image marker at the beginning of every JPEG datastream.
pub const SIGNATURE: [u8; 2] = [0xff, 0xd8];

const BLOCK_SIDE: u32 = 8;
const COMPONENTS_MAX: usize = 3;
const PROGRESSIVE_COEFFICIENT_BYTES_MAX: u64 = 512 * 1024 * 1024;
const QUANTIZATION_TABLES_MAX: usize = 4;
const SCANS_MAX: u32 = 4_096;

// JPEG stores coefficients diagonally, while the IDCT consumes ordinary row-major blocks.
const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Decodes an 8-bit Huffman- or arithmetic-coded sequential or progressive JPEG.
///
/// The decoder performs no I/O. It accepts any byte container that can be viewed as a slice so the
/// caller retains control over input limits and storage.
///
/// # Errors
///
/// Returns [`JPEGError`] when the input is malformed, exceeds a resource bound, or uses a JPEG
/// feature that this decoder does not support.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    let bytes = bytes.as_ref();
    if uses_arithmetic_coding(bytes) {
        parser::decode(bytes)
    } else {
        decode_huffman(bytes)
    }
}

fn decode_huffman(bytes: &[u8]) -> Result<DecodedImage> {
    let options = DecoderOptions::new_fast()
        .set_max_width(usize::try_from(DIMENSION_MAX).expect("JPEG dimension limit fits usize"))
        .set_max_height(usize::try_from(DIMENSION_MAX).expect("JPEG dimension limit fits usize"))
        .set_strict_mode(true)
        .jpeg_set_max_scans(usize::try_from(SCANS_MAX).expect("JPEG scan limit fits usize"))
        .jpeg_set_out_colorspace(ColorSpace::RGBA);

    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder
        .decode_headers()
        .map_err(|source| codec_error(&source))?;

    let (width, height) = decoder.dimensions().ok_or_else(|| {
        error(JPEGError::Frame(
            "JPEG codec did not report decoded dimensions",
        ))
    })?;

    let width = u32::try_from(width).map_err(|_source| {
        error(JPEGError::LimitExceeded(JPEGLimit::Dimensions(
            DIMENSION_MAX,
        )))
    })?;

    let height = u32::try_from(height).map_err(|_source| {
        error(JPEGError::LimitExceeded(JPEGLimit::Dimensions(
            DIMENSION_MAX,
        )))
    })?;
    validate_dimensions(width, height)?;

    let expected = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .expect("validated JPEG output length fits usize");

    let output_size = decoder.output_buffer_size().ok_or_else(|| {
        error(JPEGError::ArithmeticOverflow(
            "JPEG codec output size overflowed",
        ))
    })?;

    if output_size != expected {
        return Err(error(JPEGError::Frame(
            "JPEG codec reported an unexpected RGBA output size",
        )));
    }

    let mut rgba = vec![0; output_size];
    decoder
        .decode_into(&mut rgba)
        .map_err(|source| codec_error(&source))?;

    Ok(DecodedImage::new(width, height, rgba))
}

fn uses_arithmetic_coding(bytes: &[u8]) -> bool {
    if !bytes.starts_with(&SIGNATURE) {
        return false;
    }

    let mut offset = SIGNATURE.len();
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            return false;
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let Some(&marker) = bytes.get(offset) else {
            return false;
        };
        offset += 1;

        if matches!(marker, 0xc9..=0xcf) {
            return true;
        }

        if matches!(marker, 0xda | 0xd9) {
            return false;
        }

        if marker == 0x01 || marker == 0xd8 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }

        let Some(length_bytes) = bytes.get(offset..offset.saturating_add(2)) else {
            return false;
        };

        let length = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
        if length < 2 {
            return false;
        }

        let Some(next_offset) = offset.checked_add(length) else {
            return false;
        };

        if next_offset > bytes.len() {
            return false;
        }

        offset = next_offset;
    }

    false
}

fn codec_error(source: &DecodeErrors) -> Error {
    error(JPEGError::Codec(source.to_string()))
}

fn divide_ceil(value: u32, divisor: u32) -> u32 {
    invariant!(value > 0);
    invariant!(divisor > 0);
    value.div_ceil(divisor)
}
