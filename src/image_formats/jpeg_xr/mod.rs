//! Decodes bounded JPEG XR images and tone-maps HDR pixels to SDR RGBA8.
//!
//! Decoding stops at the native pixel representation; HDR analysis and tone mapping are shared
//! with other formats through the [`hdr`](super::hdr) module.

mod error;

use super::hdr::{self, DecodeOptions, DecodedHDR, NativeFormat, NativeHDR, PixelBuffer};
use super::{DIMENSION_MAX, DecodedImage, Dimensions, PIXELS_MAX, map_dimensions_error};
use error::error;

pub use error::{Error, JPEGXRError, JPEGXRLimit, Result};

/// The four-byte signature at the beginning of a JPEG XR file.
pub const SIGNATURE: [u8; 4] = [0x49, 0x49, 0xbc, 0x01];

impl PixelBuffer for ::jpegxr::BGR101010Image {
    fn bytes(&self) -> &[u8] {
        zerocopy::IntoBytes::as_bytes(self.pixels())
    }
}

impl PixelBuffer for ::jpegxr::RGBAF32Image {
    fn bytes(&self) -> &[u8] {
        zerocopy::IntoBytes::as_bytes(self.pixels())
    }
}

impl PixelBuffer for ::jpegxr::RGBAF16Image {
    fn bytes(&self) -> &[u8] {
        zerocopy::IntoBytes::as_bytes(self.pixels())
    }
}

/// Returns whether `bytes` begins with the JPEG XR file signature.
#[must_use]
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE)
}

/// Decodes a JPEG XR image and normalizes it to SDR RGBA8.
///
/// Unsigned integer RGB and grayscale inputs retain their sRGB encoding. `PixelFormat32bppRGB101010`
/// is treated as BT.2100 PQ and Rec. 2020 HDR screenshot data, then converted to linear scRGB. Other
/// HDR inputs use linear scRGB. HDR values use ITU-R BT.2446 Method A; premultiplied inputs return
/// straight alpha.
///
/// # Errors
///
/// Returns [`JPEGXRError`] for malformed input, resource-limit failures, and unsupported pixel
/// representations.
pub fn decode(bytes: &[u8]) -> Result<DecodedImage> {
    decode_with_options(bytes, DecodeOptions::default())
}

/// Decodes JPEG XR pixels using the selected HDR normalization `options`.
///
/// SDR sources bypass tone mapping. Image-derived white points are floored at display white;
/// methods without a white point apply their curves to HDR sources.
///
/// # Errors
///
/// Returns [`JPEGXRError`] for malformed input, resource-limit failures, and unsupported pixel
/// representations.
pub fn decode_with_options(bytes: &[u8], options: DecodeOptions) -> Result<DecodedImage> {
    Ok(decode_with_metadata_and_options(bytes, options.with_hdr_metrics(false))?.into_image())
}

/// Decodes JPEG XR pixels together with their source representation metadata.
///
/// Performs the bounded decode and HDR-to-SDR normalization of [`decode`]. HDR metadata includes
/// percentile `MaxCLL`; the default BT.2446 mapper does not use it.
///
/// # Errors
///
/// Returns [`JPEGXRError`] for malformed input, resource-limit failures, and unsupported pixel
/// representations.
pub fn decode_with_metadata(bytes: &[u8]) -> Result<DecodedHDR> {
    decode_with_metadata_and_options(bytes, DecodeOptions::default())
}

/// Decodes JPEG XR pixels and metadata using the selected HDR normalization `options`.
///
/// Image-dependent parameters come from the decoded source. Component-wise white-point methods use
/// the options' `MaxCLLMode`; extended luminance Reinhard uses p99.99 Rec. 709 luminance. SDR
/// sources bypass tone mapping. Image-derived white points are floored at display white.
///
/// # Errors
///
/// Returns [`JPEGXRError`] for malformed input, resource-limit failures, and unsupported pixel
/// representations.
pub fn decode_with_metadata_and_options(
    bytes: &[u8],
    options: DecodeOptions,
) -> Result<DecodedHDR> {
    hdr::tonemap_native(&decode_native(bytes)?, options)
        .map_err(|source| source.raise(JPEGXRError::Normalize))
}

/// Decodes JPEG XR pixels to their native representation, without HDR-to-SDR tone mapping.
///
/// Entropy decoding dominates JPEG XR's cost; retain the result and call [`hdr::tonemap_native`]
/// with different options to re-tone-map without repeating it.
///
/// # Errors
///
/// Returns [`JPEGXRError`] for malformed input, resource-limit failures, and unsupported pixel
/// representations.
pub fn decode_native(bytes: &[u8]) -> Result<NativeHDR> {
    if !has_signature(bytes) {
        return Err(error(JPEGXRError::Signature));
    }

    let decoder = ::jpegxr::Decoder::new(bytes).map_err(|source| codec_error(&source))?;
    let width = decoder.info().width();
    let height = decoder.info().height();
    let (width, height) = validate_dimensions(
        i32::try_from(width).map_err(|_error| dimension_overflow_error(width))?,
        i32::try_from(height).map_err(|_error| dimension_overflow_error(height))?,
    )?;

    let pixel_format = decoder.info().pixel_format();
    let format = match pixel_format {
        ::jpegxr::PixelFormat::BGR101010 => NativeFormat::BGR101010,
        ::jpegxr::PixelFormat::RGBA128_FLOAT => NativeFormat::RGBAF32,
        ::jpegxr::PixelFormat::RGBA64_HALF => NativeFormat::RGBAF16,
        _ => {
            return Err(error(JPEGXRError::Unsupported(
                pixel_format.name().to_owned(),
            )));
        }
    };

    // Reject oversized sources before the expensive entropy decode.
    let source_len = usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(format.bytes_per_pixel()))
        .and_then(|stride| stride.checked_mul(usize::try_from(height).ok()?))
        .ok_or_else(|| {
            error(JPEGXRError::Output(
                "JPEG XR source-buffer size exceeds usize",
            ))
        })?;

    if source_len > hdr::SOURCE_BUFFER_MAX {
        return Err(error(JPEGXRError::LimitExceeded(
            JPEGXRLimit::SourceBufferBytes {
                actual: Some(source_len),
                max: hdr::SOURCE_BUFFER_MAX,
            },
        )));
    }

    let pixels: Box<dyn PixelBuffer> = match format {
        NativeFormat::BGR101010 => Box::new(
            decoder
                .decode_bgr101010()
                .map_err(|source| codec_error(&source))?,
        ),
        NativeFormat::RGBAF32 => Box::new(
            decoder
                .decode_rgba_f32()
                .map_err(|source| codec_error(&source))?,
        ),
        NativeFormat::RGBAF16 => Box::new(
            decoder
                .decode_rgba_half()
                .map_err(|source| codec_error(&source))?,
        ),
    };

    NativeHDR::from_buffer(width, height, format, pixels)
        .map_err(|source| source.raise(JPEGXRError::Normalize))
}

/// Builds the error for a decoder-reported dimension too large to convert to `i32`.
fn dimension_overflow_error(value: u32) -> Error {
    error(JPEGXRError::LimitExceeded(JPEGXRLimit::Dimensions {
        actual: Some(value),
        max: DIMENSION_MAX,
    }))
}

fn validate_dimensions(width: i32, height: i32) -> Result<(u32, u32)> {
    let width = u32::try_from(width).map_err(|_conversion_error| {
        error(JPEGXRError::Output("JPEG XR width must be positive"))
    })?;

    let height = u32::try_from(height).map_err(|_conversion_error| {
        error(JPEGXRError::Output("JPEG XR height must be positive"))
    })?;

    Dimensions::try_new((width, height))
        .map(|_| (width, height))
        .map_err(|dimensions_error| {
            map_dimensions_error(
                dimensions_error,
                || {
                    error(JPEGXRError::Output(
                        "JPEG XR dimensions must both be nonzero",
                    ))
                },
                |width, height| {
                    error(JPEGXRError::LimitExceeded(JPEGXRLimit::Dimensions {
                        actual: Some(width.max(height)),
                        max: DIMENSION_MAX,
                    }))
                },
                |pixels| {
                    error(JPEGXRError::LimitExceeded(JPEGXRLimit::Pixels {
                        actual: Some(pixels),
                        max: PIXELS_MAX,
                    }))
                },
            )
        })
}

fn codec_error(source: &jpegxr::Error) -> Error {
    if source.is_unsupported() {
        error(JPEGXRError::Unsupported(source.to_string()))
    } else if source.is_dimension_limit_exceeded() {
        error(JPEGXRError::LimitExceeded(JPEGXRLimit::Dimensions {
            actual: None,
            max: DIMENSION_MAX,
        }))
    } else if source.is_pixel_count_limit_exceeded() {
        error(JPEGXRError::LimitExceeded(JPEGXRLimit::Pixels {
            actual: None,
            max: PIXELS_MAX,
        }))
    } else if source.is_limit_exceeded() {
        error(JPEGXRError::LimitExceeded(JPEGXRLimit::SourceBufferBytes {
            actual: None,
            max: hdr::SOURCE_BUFFER_MAX,
        }))
    } else {
        error(JPEGXRError::Codec(source.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires JPEGXR_SAMPLE to name a local HDR image"]
    fn decodes_real_hdr_sample_with_pure_rust_codec() {
        let path = std::env::var("JPEGXR_SAMPLE").expect("set JPEGXR_SAMPLE");
        let bytes = std::fs::read(path).expect("read sample");
        let decoded = decode_with_metadata(&bytes).expect("decode HDR sample");

        assert_eq!(decoded.image().dimensions(), (3440, 1440));
        assert_eq!(decoded.image().rgba8().len(), 3440 * 1440 * 4);
        assert!(decoded.metadata().has_alpha());
        assert!(decoded.metadata().is_hdr());
    }

    #[test]
    #[ignore = "requires JPEGXR_BGR101010_SAMPLE to name a local HDR image"]
    fn decodes_real_bgr101010_sample_with_pure_rust_codec() {
        let path = std::env::var("JPEGXR_BGR101010_SAMPLE").expect("set JPEGXR_BGR101010_SAMPLE");
        let bytes = std::fs::read(path).expect("read sample");
        let decoded = decode_with_metadata(&bytes).expect("decode BGR101010 HDR sample");

        assert_eq!(decoded.image().dimensions(), (3840, 2160));
        assert_eq!(decoded.image().rgba8().len(), 3840 * 2160 * 4);
        assert!(!decoded.metadata().has_alpha());
        assert!(decoded.metadata().is_hdr());
    }

    #[test]
    #[ignore = "requires JPEGXR_RGBA64_HALF_SAMPLE to name a local HDR image"]
    fn decodes_real_rgba64_half_sample_with_pure_rust_codec() {
        let path =
            std::env::var("JPEGXR_RGBA64_HALF_SAMPLE").expect("set JPEGXR_RGBA64_HALF_SAMPLE");
        let bytes = std::fs::read(path).expect("read sample");
        let decoded = decode_with_metadata(&bytes).expect("decode RGBA64Half HDR sample");

        assert_eq!(decoded.image().dimensions(), (2560, 1440));
        assert_eq!(decoded.image().rgba8().len(), 2560 * 1440 * 4);
        assert!(decoded.metadata().has_alpha());
        assert!(decoded.metadata().is_hdr());
        assert_eq!(decoded.metadata().bits_per_channel(), 16);
    }
}
