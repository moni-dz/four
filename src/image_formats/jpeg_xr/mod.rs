//! Decodes bounded JPEG XR images and tone-maps HDR pixels to SDR RGBA8.
//!
//! The JPEG XR reference codec produces pixels in their native WIC representation. Unsigned RGB
//! and grayscale samples are normalized as sRGB. Fixed-point, half-float, floating-point, and RGBE
//! samples are interpreted as linear scRGB, tone-mapped with a MaxCLL-aware shoulder curve,
//! gamut-compressed without changing chromaticity, and encoded with the sRGB transfer function.
//! `MaxCLL` is estimated from the 99.99th-percentile `max(R, G, B)` light level, rejecting isolated
//! outliers at the cost of clipping the brightest 0.01% of a sufficiently large image.
//! A wholly empty, non-premultiplied HDR alpha plane is treated as unspecified and made opaque,
//! matching JPEG XR screenshots that store zero in an otherwise unused alpha channel.

mod error;

use std::io::Cursor;

use ::jpegxr::{
    BitDepthBits, ColorFormat, ImageDecode, JXRError, PixelFormat as CodecPixelFormat, PixelInfo,
};

use super::DecodedImage;
use error::error;

pub use error::{Error, JPEGXRError, JPEGXRLimit, Result};

/// The four-byte signature at the beginning of a JPEG XR file.
pub const SIGNATURE: [u8; 4] = [0x49, 0x49, 0xbc, 0x01];

const DIMENSION_MAX: u32 = 16_384;
const MAX_CLL_HIGH_BINS: usize = 1 << 16;
const MAX_CLL_LOW_BINS: usize = 1 << 15;
const MAX_CLL_LOW_BITS: u32 = 15;
const MAX_CLL_PERCENTILE_DENOMINATOR: u64 = 10_000;
const PIXELS_MAX: u64 = 64 * 1024 * 1024;
const SC_RGB_REFERENCE_WHITE_NITS: f32 = 80.0;
const SOURCE_BUFFER_MAX: usize = 512 * 1024 * 1024;
const TONE_MAP_KNEE: f32 = 0.75;

/// Returns whether `bytes` begins with the JPEG XR file signature.
#[must_use]
pub fn has_signature(bytes: impl AsRef<[u8]>) -> bool {
    bytes.as_ref().starts_with(&SIGNATURE)
}

/// Decodes a JPEG XR image and normalizes it to SDR RGBA8.
///
/// Unsigned integer RGB and grayscale inputs retain their sRGB encoding. Linear-light HDR inputs
/// use a `MaxCLL`-aware shoulder curve before conversion to sRGB. `MaxCLL` is estimated from the
/// 99.99th-percentile brightest pixel to prevent isolated outliers from dimming the image.
/// Premultiplied inputs are returned with straight alpha.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    let bytes = bytes.as_ref();
    if !has_signature(bytes) {
        return Err(error(JPEGXRError::Signature));
    }

    let mut decoder =
        ImageDecode::with_reader(Cursor::new(bytes)).map_err(|source| codec_error(&source))?;
    let (width, height) = decoder.get_size().map_err(|source| codec_error(&source))?;
    let (width, height) = validate_dimensions(width, height)?;
    let format = decoder
        .get_pixel_format()
        .map_err(|source| codec_error(&source))?;
    let layout = PixelLayout::new(format)?;
    let row_stride = layout.row_stride(width)?;
    let height_usize = usize::try_from(height)
        .map_err(|_conversion_error| error(JPEGXRError::Output("JPEG XR height exceeds usize")))?;
    let source_len = row_stride.checked_mul(height_usize).ok_or_else(|| {
        error(JPEGXRError::Output(
            "JPEG XR source-buffer size exceeds usize",
        ))
    })?;
    if source_len > SOURCE_BUFFER_MAX {
        return Err(error(JPEGXRError::LimitExceeded(
            JPEGXRLimit::SourceBufferBytes(SOURCE_BUFFER_MAX),
        )));
    }

    let mut source = vec![0_u8; source_len];
    decoder
        .copy_all(&mut source, row_stride)
        .map_err(|source| codec_error(&source))?;
    let rgba = normalize(&source, width, height, row_stride, layout)?;
    Ok(DecodedImage::new(width, height, rgba))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleEncoding {
    Fixed16,
    Fixed32,
    Float16,
    Float32,
    Rgbe,
    Unsigned8,
    Unsigned16,
}

impl SampleEncoding {
    const fn bytes(self) -> usize {
        match self {
            Self::Unsigned8 | Self::Rgbe => 1,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 2,
            Self::Fixed32 | Self::Float32 => 4,
        }
    }

    const fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Fixed16 | Self::Fixed32 | Self::Float16 | Self::Float32 | Self::Rgbe
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PixelLayout {
    encoding: SampleEncoding,
    color_channels: usize,
    source_channels: usize,
    bytes_per_pixel: usize,
    has_alpha: bool,
    premultiplied_alpha: bool,
    blue_first: bool,
}

impl PixelLayout {
    fn new(format: CodecPixelFormat) -> Result<Self> {
        let info = PixelInfo::from_format(format);
        let encoding = sample_encoding(format, &info)?;
        let color_channels = match info.color_format() {
            ColorFormat::YOnly => 1,
            ColorFormat::RGB => 3,
            ColorFormat::RGBE if encoding == SampleEncoding::Rgbe => 3,
            other => {
                return Err(unsupported(format, &format!("{other:?} color data")));
            }
        };
        let has_alpha = info.has_alpha();
        let source_channels = info.channels();
        let expected_channels = color_channels + usize::from(has_alpha);
        if encoding != SampleEncoding::Rgbe && source_channels != expected_channels {
            return Err(unsupported(
                format,
                "channel metadata is inconsistent with the color format",
            ));
        }
        if encoding == SampleEncoding::Rgbe && has_alpha {
            return Err(unsupported(format, "RGBE with alpha is not supported"));
        }
        let bits_per_pixel = info.bits_per_pixel();
        if !bits_per_pixel.is_multiple_of(8) {
            return Err(unsupported(format, "packed pixels are not supported"));
        }
        let bytes_per_pixel = bits_per_pixel / 8;
        let sample_bytes = encoding.bytes();
        if !bytes_per_pixel.is_multiple_of(sample_bytes)
            || bytes_per_pixel / sample_bytes < source_channels
        {
            return Err(unsupported(
                format,
                "pixel stride is inconsistent with its samples",
            ));
        }

        Ok(Self {
            encoding,
            color_channels,
            source_channels,
            bytes_per_pixel,
            has_alpha,
            premultiplied_alpha: info.premultiplied_alpha(),
            blue_first: info.bgr(),
        })
    }

    fn row_stride(self, width: u32) -> Result<usize> {
        usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(self.bytes_per_pixel))
            .ok_or_else(|| error(JPEGXRError::Output("JPEG XR row stride exceeds usize")))
    }

    fn read_pixel(self, pixel: &[u8]) -> Result<([f32; 3], f32)> {
        invariant_eq!(pixel.len(), self.bytes_per_pixel);
        if self.encoding == SampleEncoding::Rgbe {
            return Ok((decode_rgbe(pixel)?, 1.0));
        }

        let sample = |channel: usize| -> Result<f32> {
            invariant!(channel < self.source_channels);
            let start = channel * self.encoding.bytes();
            let end = start + self.encoding.bytes();
            let bytes = pixel.get(start..end).ok_or_else(|| {
                error(JPEGXRError::Output(
                    "JPEG XR sample exceeds its pixel stride",
                ))
            })?;
            Ok(decode_sample(bytes, self.encoding))
        };

        let mut color = if self.color_channels == 1 {
            let gray = sample(0)?;
            [gray, gray, gray]
        } else {
            [sample(0)?, sample(1)?, sample(2)?]
        };
        if self.blue_first && self.color_channels == 3 {
            color.swap(0, 2);
        }
        let alpha = if self.has_alpha {
            normalize_alpha(sample(self.color_channels)?)
        } else {
            1.0
        };
        if self.premultiplied_alpha {
            if alpha > 0.0 {
                for channel in &mut color {
                    *channel /= alpha;
                }
            } else {
                color.fill(0.0);
            }
        }
        Ok((color, alpha))
    }
}

fn sample_encoding(format: CodecPixelFormat, info: &PixelInfo) -> Result<SampleEncoding> {
    if format == CodecPixelFormat::PixelFormat32bppRGBE {
        return Ok(SampleEncoding::Rgbe);
    }
    if matches!(
        format,
        CodecPixelFormat::PixelFormat48bppRGBHalf
            | CodecPixelFormat::PixelFormat64bppRGBHalf
            | CodecPixelFormat::PixelFormat64bppRGBAHalf
            | CodecPixelFormat::PixelFormat16bppGrayHalf
    ) {
        return Ok(SampleEncoding::Float16);
    }

    match info.bit_depth() {
        BitDepthBits::Eight => Ok(SampleEncoding::Unsigned8),
        BitDepthBits::Sixteen => Ok(SampleEncoding::Unsigned16),
        BitDepthBits::SixteenS => Ok(SampleEncoding::Fixed16),
        BitDepthBits::SixteenF => Ok(SampleEncoding::Float16),
        BitDepthBits::ThirtyTwoS => Ok(SampleEncoding::Fixed32),
        BitDepthBits::ThirtyTwoF => Ok(SampleEncoding::Float32),
        other => Err(unsupported(
            format,
            &format!("{other:?} samples are not supported"),
        )),
    }
}

fn normalize(
    source: &[u8],
    width: u32,
    height: u32,
    row_stride: usize,
    layout: PixelLayout,
) -> Result<Vec<u8>> {
    let width = usize::try_from(width).expect("validated JPEG XR width fits usize");
    let height = usize::try_from(height).expect("validated JPEG XR height fits usize");
    let expected_source_len = row_stride.checked_mul(height).ok_or_else(|| {
        error(JPEGXRError::Output(
            "JPEG XR source-buffer size exceeds usize",
        ))
    })?;
    if source.len() != expected_source_len {
        return Err(error(JPEGXRError::Output(
            "JPEG XR codec returned an incomplete source buffer",
        )));
    }
    let output_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| error(JPEGXRError::Output("JPEG XR RGBA size exceeds usize")))?;
    let max_cll = if layout.encoding.is_hdr() {
        MaxCll::estimate(source, row_stride, layout)?
    } else {
        MaxCll::SDR
    };
    let mut rgba = Vec::with_capacity(output_len);
    let mut has_nonzero_alpha = false;

    for row in source.chunks_exact(row_stride) {
        for x in 0..width {
            let start = x
                .checked_mul(layout.bytes_per_pixel)
                .ok_or_else(|| error(JPEGXRError::Output("JPEG XR pixel offset exceeds usize")))?;
            let end = start + layout.bytes_per_pixel;
            let pixel = row.get(start..end).ok_or_else(|| {
                error(JPEGXRError::Output("JPEG XR pixel exceeds its decoded row"))
            })?;
            let (color, alpha) = layout.read_pixel(pixel)?;
            has_nonzero_alpha |= alpha > 0.0;
            let color = if layout.encoding.is_hdr() {
                tone_map(color, max_cll)
            } else {
                color.map(normalized_to_u8)
            };
            rgba.extend_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }
    if layout.encoding.is_hdr()
        && layout.has_alpha
        && !layout.premultiplied_alpha
        && !has_nonzero_alpha
    {
        for alpha in rgba.iter_mut().skip(3).step_by(4) {
            *alpha = u8::MAX;
        }
    }
    invariant_eq!(rgba.len(), output_len);
    Ok(rgba)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaxCll {
    nits: f32,
}

impl MaxCll {
    const SDR: Self = Self {
        nits: SC_RGB_REFERENCE_WHITE_NITS,
    };

    fn estimate(source: &[u8], row_stride: usize, layout: PixelLayout) -> Result<Self> {
        invariant!(layout.encoding.is_hdr());
        invariant!(row_stride >= layout.bytes_per_pixel);

        let pixels_per_row = row_stride / layout.bytes_per_pixel;
        let row_count = source.len() / row_stride;
        let pixel_count = u64::try_from(
            pixels_per_row
                .checked_mul(row_count)
                .expect("validated JPEG XR pixel count fits usize"),
        )
        .expect("validated JPEG XR pixel count fits u64");
        invariant!(pixel_count > 0);
        let target_rank = pixel_count - pixel_count / MAX_CLL_PERCENTILE_DENOMINATOR;

        let mut high_histogram = vec![0_u32; MAX_CLL_HIGH_BINS];
        visit_pixels(source, row_stride, layout, |color| {
            let (high, _low) = histogram_parts(max_rgb(color));
            high_histogram[high] += 1;
        })?;
        let (high, count_before_high) = percentile_bin(&high_histogram, target_rank);

        let mut low_histogram = vec![0_u32; MAX_CLL_LOW_BINS];
        visit_pixels(source, row_stride, layout, |color| {
            let (pixel_high, pixel_low) = histogram_parts(max_rgb(color));
            if pixel_high == high {
                low_histogram[pixel_low] += 1;
            }
        })?;
        let rank_within_high = target_rank - count_before_high;
        let (low, _count_before_low) = percentile_bin(&low_histogram, rank_within_high);

        let high = u32::try_from(high).expect("a MaxCLL high histogram index fits u32");
        let low = u32::try_from(low).expect("a MaxCLL low histogram index fits u32");
        let relative_light_level = f32::from_bits((high << MAX_CLL_LOW_BITS) | low);
        invariant!(relative_light_level.is_finite());
        invariant!(relative_light_level >= 0.0);
        Ok(Self {
            nits: relative_light_level * SC_RGB_REFERENCE_WHITE_NITS,
        })
    }

    fn relative_light_level(self) -> f32 {
        invariant!(self.nits.is_finite());
        invariant!(self.nits >= 0.0);
        self.nits / SC_RGB_REFERENCE_WHITE_NITS
    }
}

fn visit_pixels(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
    mut visitor: impl FnMut([f32; 3]),
) -> Result<()> {
    invariant!(row_stride >= layout.bytes_per_pixel);

    let mut rows = source.chunks_exact(row_stride);
    for row in &mut rows {
        let mut pixels = row.chunks_exact(layout.bytes_per_pixel);
        for pixel in &mut pixels {
            let (color, _alpha) = layout.read_pixel(pixel)?;
            visitor(color);
        }
        if !pixels.remainder().is_empty() {
            return Err(error(JPEGXRError::Output(
                "JPEG XR row contains a partial pixel",
            )));
        }
    }
    if !rows.remainder().is_empty() {
        return Err(error(JPEGXRError::Output(
            "JPEG XR source buffer contains a partial row",
        )));
    }
    Ok(())
}

fn max_rgb(color: [f32; 3]) -> f32 {
    let color = color.map(sanitize_hdr_sample);
    color[0].max(color[1]).max(color[2])
}

fn histogram_parts(value: f32) -> (usize, usize) {
    invariant!(value.is_finite());
    invariant!(value >= 0.0);

    let bits = value.to_bits();
    let high = usize::try_from(bits >> MAX_CLL_LOW_BITS)
        .expect("a positive f32 high-bit index fits usize");
    let low_mask =
        u32::try_from(MAX_CLL_LOW_BINS - 1).expect("the MaxCLL low histogram mask fits u32");
    let low = usize::try_from(bits & low_mask).expect("an f32 low-bit index fits usize");
    invariant!(high < MAX_CLL_HIGH_BINS);
    invariant!(low < MAX_CLL_LOW_BINS);
    (high, low)
}

fn percentile_bin(histogram: &[u32], target_rank: u64) -> (usize, u64) {
    invariant!(target_rank > 0);

    let mut count_before = 0_u64;
    for (index, count) in histogram.iter().copied().enumerate() {
        let count_after = count_before + u64::from(count);
        if count_after >= target_rank {
            return (index, count_before);
        }
        count_before = count_after;
    }
    panic!(
        "MaxCLL histogram contains {count_before} samples, fewer than target rank {target_rank}"
    );
}

fn validate_dimensions(width: i32, height: i32) -> Result<(u32, u32)> {
    let width = u32::try_from(width).map_err(|_conversion_error| {
        error(JPEGXRError::Output("JPEG XR width must be positive"))
    })?;
    let height = u32::try_from(height).map_err(|_conversion_error| {
        error(JPEGXRError::Output("JPEG XR height must be positive"))
    })?;
    if width == 0 || height == 0 {
        return Err(error(JPEGXRError::Output(
            "JPEG XR dimensions must both be nonzero",
        )));
    }
    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(JPEGXRError::LimitExceeded(JPEGXRLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }
    if u64::from(width) * u64::from(height) > PIXELS_MAX {
        return Err(error(JPEGXRError::LimitExceeded(JPEGXRLimit::Pixels(
            PIXELS_MAX,
        ))));
    }
    Ok((width, height))
}

fn decode_sample(bytes: &[u8], encoding: SampleEncoding) -> f32 {
    invariant_eq!(bytes.len(), encoding.bytes());
    match encoding {
        SampleEncoding::Unsigned8 => f32::from(bytes[0]) / f32::from(u8::MAX),
        SampleEncoding::Unsigned16 => {
            f32::from(u16::from_ne_bytes([bytes[0], bytes[1]])) / f32::from(u16::MAX)
        }
        SampleEncoding::Fixed16 => f32::from(i16::from_ne_bytes([bytes[0], bytes[1]])) / 8192.0,
        SampleEncoding::Fixed32 => {
            fixed32_to_f32(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        SampleEncoding::Float16 => half_to_f32(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleEncoding::Float32 => f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        SampleEncoding::Rgbe => {
            unreachable!("RGBE pixels are decoded as a unit")
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "s7.24 fixed-point values are intentionally converted to f32 for tone mapping"
)]
fn fixed32_to_f32(value: i32) -> f32 {
    (f64::from(value) / 16_777_216.0) as f32
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    match exponent {
        0 if mantissa == 0 => sign * 0.0,
        0 => sign * f32::from(mantissa) * 2.0_f32.powi(-24),
        0x1f if mantissa == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => {
            let significand = 1.0 + f32::from(mantissa) / 1024.0;
            sign * significand * 2.0_f32.powi(i32::from(exponent) - 15)
        }
    }
}

fn decode_rgbe(pixel: &[u8]) -> Result<[f32; 3]> {
    if pixel.len() < 4 {
        return Err(error(JPEGXRError::Output(
            "JPEG XR RGBE pixel is shorter than four bytes",
        )));
    }
    let exponent = pixel[3];
    if exponent == 0 {
        return Ok([0.0; 3]);
    }
    let scale = 2.0_f32.powi(i32::from(exponent) - 136);
    Ok([
        f32::from(pixel[0]) * scale,
        f32::from(pixel[1]) * scale,
        f32::from(pixel[2]) * scale,
    ])
}

fn tone_map(color: [f32; 3], max_cll: MaxCll) -> [u8; 3] {
    let linear = color.map(sanitize_hdr_sample);
    let luminance = 0.212_6 * linear[0] + 0.715_2 * linear[1] + 0.072_2 * linear[2];
    if luminance == 0.0 {
        return [0; 3];
    }

    let mapped_luminance = shoulder_curve(luminance, max_cll.relative_light_level());
    let luminance_scale = mapped_luminance / luminance;
    let mut mapped = linear.map(|channel| channel * luminance_scale);
    let peak = mapped[0].max(mapped[1]).max(mapped[2]);
    if peak > 1.0 {
        for channel in &mut mapped {
            *channel /= peak;
        }
    }
    mapped.map(|channel| normalized_to_u8(linear_to_srgb(channel)))
}

fn sanitize_hdr_sample(value: f32) -> f32 {
    if value.is_nan() || value <= 0.0 {
        0.0
    } else if value.is_infinite() {
        65_504.0
    } else {
        value.min(65_504.0)
    }
}

fn normalize_alpha(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn shoulder_curve(value: f32, max_cll: f32) -> f32 {
    invariant!(value >= 0.0);
    invariant!(max_cll >= 0.0);

    if max_cll <= 1.0 || value <= TONE_MAP_KNEE {
        return value;
    }

    let input_range = max_cll - TONE_MAP_KNEE;
    let output_range = 1.0 - TONE_MAP_KNEE;
    let softness = input_range * output_range / (max_cll - 1.0);
    let distance = value - TONE_MAP_KNEE;
    TONE_MAP_KNEE + distance / (1.0 + distance / softness)
}

fn linear_to_srgb(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the normalized sample is rounded and clamped to u8 before conversion"
)]
fn normalized_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * f32::from(u8::MAX)).round() as u8
}

fn unsupported(format: CodecPixelFormat, detail: &str) -> Error {
    invariant!(!detail.is_empty());
    error(JPEGXRError::Unsupported(format!("{format:?}: {detail}")))
}

fn codec_error(source: &JXRError) -> Error {
    if matches!(source, JXRError::OutOfMemory | JXRError::BufferOverflow) {
        error(JPEGXRError::LimitExceeded(JPEGXRLimit::SourceBufferBytes(
            SOURCE_BUFFER_MAX,
        )))
    } else {
        error(JPEGXRError::Codec(source.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_half_precision_boundaries() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn tone_mapping_compresses_hdr_white_into_sdr() {
        let max_cll = MaxCll { nits: 320.0 };
        assert_eq!(tone_map([0.0; 3], max_cll), [0; 3]);
        let reference_white = tone_map([1.0; 3], max_cll);
        let hdr_white = tone_map([4.0; 3], max_cll);

        assert!(reference_white[0] >= 235 && reference_white[0] <= 245);
        assert!(hdr_white[0] > reference_white[0]);
        assert_eq!(hdr_white[0], u8::MAX);
        assert_eq!(reference_white[0], reference_white[1]);
        assert_eq!(hdr_white[1], hdr_white[2]);
    }

    #[test]
    fn tone_mapping_preserves_chromaticity_during_gamut_compression() {
        let mapped = tone_map([4.0, 2.0, 1.0], MaxCll { nits: 320.0 });

        assert_eq!(mapped[0], u8::MAX);
        assert!(mapped[0] > mapped[1]);
        assert!(mapped[1] > mapped[2]);
    }

    #[test]
    fn max_cll_rejects_the_brightest_point_zero_one_percent() {
        let layout = PixelLayout {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 3,
            bytes_per_pixel: 12,
            has_alpha: false,
            premultiplied_alpha: false,
            blue_first: false,
        };
        let mut source = Vec::with_capacity(10_000 * layout.bytes_per_pixel);
        for value in std::iter::repeat_n(1.0_f32, 9_998).chain([4.0, 126.0]) {
            for channel in [value; 3] {
                source.extend_from_slice(&channel.to_ne_bytes());
            }
        }

        let max_cll = MaxCll::estimate(&source, source.len(), layout).unwrap();

        assert_eq!(max_cll.nits, 320.0);
    }

    #[test]
    fn empty_hdr_alpha_plane_is_treated_as_unspecified() {
        let layout = PixelLayout {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 4,
            bytes_per_pixel: 16,
            has_alpha: true,
            premultiplied_alpha: false,
            blue_first: false,
        };
        let source = [
            0.18_f32.to_ne_bytes(),
            0.18_f32.to_ne_bytes(),
            0.18_f32.to_ne_bytes(),
            0.0_f32.to_ne_bytes(),
        ]
        .concat();

        let rgba = normalize(&source, 1, 1, 16, layout).unwrap();

        assert_eq!(rgba[3], u8::MAX);
    }
}
