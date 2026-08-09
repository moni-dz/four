//! Decodes bounded JPEG XR images and tone-maps HDR pixels to SDR RGBA8.
//!
//! The JPEG XR reference codec produces pixels in their native WIC representation. Unsigned RGB
//! and grayscale samples are normalized as sRGB. Fixed-point, half-float, floating-point, and RGBE
//! samples are interpreted as linear scRGB, tone-mapped with a MaxCLL-aware shoulder curve,
//! gamut-compressed without changing chromaticity, and encoded with the sRGB transfer function.
//!
//! `MaxCLL` is estimated from the 99.99th-percentile `max(R, G, B)` light level, rejecting isolated
//! outliers at the cost of clipping the brightest 0.01% of a sufficiently large image.
//! A wholly empty, non-premultiplied HDR alpha plane is treated as unspecified and made opaque,
//! matching JPEG XR screenshots that store zero in an otherwise unused alpha channel.

mod error;

use std::io::Cursor;
use std::num::NonZeroUsize;

use ::jpegxr::{
    BitDepthBits, ColorFormat, ImageDecode, JXRError, PixelFormat as CodecPixelFormat, PixelInfo,
};
use tonemapping::{
    ColorChannel as ToneColorChannel, LinearRgb, LinearShoulder, MaxCllEstimator, ToneMapper,
};

use super::{DIMENSION_MAX, DecodedImage, PIXELS_MAX};
use error::error;

pub use error::{Error, JPEGXRError, JPEGXRLimit, Result};

/// The four-byte signature at the beginning of a JPEG XR file.
pub const SIGNATURE: [u8; 4] = [0x49, 0x49, 0xbc, 0x01];

const SC_RGB_REFERENCE_WHITE_NITS: f32 = 80.0;
const SOURCE_BUFFER_MAX: usize = 512 * 1024 * 1024;

/// Identifies a color channel in decoded JPEG XR RGB data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JPEGXRColorChannel {
    /// The red color channel.
    Red,
    /// The green color channel.
    Green,
    /// The blue color channel.
    Blue,
}

impl JPEGXRColorChannel {
    /// Returns the conventional one-letter channel symbol.
    #[must_use]
    pub const fn symbol(self) -> char {
        match self {
            Self::Red => 'R',
            Self::Green => 'G',
            Self::Blue => 'B',
        }
    }
}

/// A decoded JPEG XR image and its source metadata.
#[derive(Debug)]
pub struct DecodedJPEGXR {
    image: DecodedImage,
    metadata: JPEGXRMetadata,
}

impl DecodedJPEGXR {
    /// Returns the normalized SDR image.
    #[must_use]
    pub const fn image(&self) -> &DecodedImage {
        &self.image
    }

    /// Returns metadata derived from the JPEG XR pixel representation.
    #[must_use]
    pub const fn metadata(&self) -> JPEGXRMetadata {
        self.metadata
    }

    /// Consumes the result and returns the normalized SDR image.
    #[must_use]
    pub fn into_image(self) -> DecodedImage {
        self.image
    }
}

/// Describes the native JPEG XR samples used to produce the SDR image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JPEGXRMetadata {
    bits_per_channel: u8,
    color_channels: u8,
    has_alpha: bool,
    is_hdr: bool,
    hdr_metrics: Option<HDRMetrics>,
}

impl JPEGXRMetadata {
    /// Returns the number of bits in each native color sample.
    #[must_use]
    pub const fn bits_per_channel(self) -> u8 {
        self.bits_per_channel
    }

    /// Returns the number of native color channels, excluding alpha.
    #[must_use]
    pub const fn color_channels(self) -> u8 {
        self.color_channels
    }

    /// Returns whether the source pixel representation contains alpha.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        self.has_alpha
    }

    /// Returns whether the source samples were interpreted as linear HDR light levels.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        self.is_hdr
    }

    /// Returns the estimated maximum content light level in nits for HDR sources.
    ///
    /// The estimate is the 99.99th-percentile `max(R, G, B)` light level. SDR sources return
    /// `None` because they do not pass through the HDR tone-mapping pipeline.
    #[must_use]
    pub const fn max_cll_nits(self) -> Option<f32> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.max_cll.nits),
            None => None,
        }
    }

    /// Returns the estimated maximum content light level in scRGB units for HDR sources.
    #[must_use]
    pub fn max_cll_scrgb(self) -> Option<f32> {
        self.max_cll_nits()
            .map(|nits| nits / SC_RGB_REFERENCE_WHITE_NITS)
    }

    /// Returns the color channel that determines the percentile `MaxCLL` value.
    #[must_use]
    pub const fn max_cll_channel(self) -> Option<JPEGXRColorChannel> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.max_cll.channel),
            None => None,
        }
    }

    /// Returns the maximum decoded luminance in nits for HDR sources.
    #[must_use]
    pub const fn max_luminance_nits(self) -> Option<f32> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.max_luminance_nits),
            None => None,
        }
    }

    /// Returns the mean decoded luminance in nits for HDR sources.
    #[must_use]
    pub const fn average_luminance_nits(self) -> Option<f32> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.average_luminance_nits),
            None => None,
        }
    }

    /// Returns the minimum decoded luminance in nits for HDR sources.
    #[must_use]
    pub const fn min_luminance_nits(self) -> Option<f32> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.min_luminance_nits),
            None => None,
        }
    }

    /// Returns the percentage of HDR pixels inside the linear Rec. 709 gamut cone.
    #[must_use]
    pub const fn rec709_percentage(self) -> Option<f32> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.rec709_percentage),
            None => None,
        }
    }

    /// Returns the percentage of HDR pixels inside Display-P3 but outside Rec. 709.
    ///
    /// Pixels outside Display-P3 count toward neither gamut percentage, so the two reported
    /// percentages may sum to less than 100 percent.
    #[must_use]
    pub const fn dci_p3_percentage(self) -> Option<f32> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.dci_p3_percentage),
            None => None,
        }
    }
}

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
    Ok(decode_with_metadata(bytes)?.into_image())
}

/// Decodes JPEG XR pixels together with their source representation metadata.
///
/// This performs the same bounded decode and HDR-to-SDR normalization as [`decode`]. For HDR
/// sources, the returned metadata includes the percentile `MaxCLL` used by tone mapping.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode_with_metadata(bytes: impl AsRef<[u8]>) -> Result<DecodedJPEGXR> {
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
    let normalized = normalize(&source, width, height, row_stride, layout)?;
    let metadata = JPEGXRMetadata::new(layout, normalized.hdr_metrics);
    Ok(DecodedJPEGXR {
        image: DecodedImage::new(width, height, normalized.rgba),
        metadata,
    })
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

    const fn bits_per_channel(self) -> u8 {
        match self {
            Self::Unsigned8 | Self::Rgbe => 8,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 16,
            Self::Fixed32 | Self::Float32 => 32,
        }
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

impl JPEGXRMetadata {
    fn new(layout: PixelLayout, hdr_metrics: Option<HDRMetrics>) -> Self {
        invariant_eq!(layout.encoding.is_hdr(), hdr_metrics.is_some());

        Self {
            bits_per_channel: layout.encoding.bits_per_channel(),
            color_channels: u8::try_from(layout.color_channels)
                .expect("a JPEG XR color-channel count fits u8"),
            has_alpha: layout.has_alpha,
            is_hdr: layout.encoding.is_hdr(),
            hdr_metrics,
        }
    }
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
                color = color.map(|channel| channel / alpha);
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
) -> Result<NormalizedImage> {
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
    let hdr_metrics = if layout.encoding.is_hdr() {
        Some(HDRMetrics::estimate(source, row_stride, layout)?)
    } else {
        None
    };
    let tone_mapper = hdr_metrics.map(|metrics| {
        LinearShoulder::new(metrics.max_cll.relative_light_level())
            .expect("estimated MaxCLL is finite and nonnegative")
    });
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
            let color = if let Some(mapper) = tone_mapper {
                hdr_to_srgb8(color, &mapper)
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
    Ok(NormalizedImage { rgba, hdr_metrics })
}

#[derive(Debug)]
struct NormalizedImage {
    rgba: Vec<u8>,
    hdr_metrics: Option<HDRMetrics>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaxCll {
    nits: f32,
    channel: JPEGXRColorChannel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HDRMetrics {
    max_cll: MaxCll,
    max_luminance_nits: f32,
    average_luminance_nits: f32,
    min_luminance_nits: f32,
    rec709_percentage: f32,
    dci_p3_percentage: f32,
}

impl HDRMetrics {
    fn estimate(source: &[u8], row_stride: usize, layout: PixelLayout) -> Result<Self> {
        invariant!(layout.encoding.is_hdr());
        invariant!(row_stride >= layout.bytes_per_pixel);

        let pixels_per_row = row_stride / layout.bytes_per_pixel;
        let row_count = source.len() / row_stride;

        let source_pixel_count = pixels_per_row
            .checked_mul(row_count)
            .expect("validated JPEG XR pixel count fits usize");

        invariant!(source_pixel_count > 0);

        let exclude_fully_transparent =
            layout.has_alpha && has_visible_alpha(source, row_stride, layout)?;

        let mut accumulator = HDRMetricAccumulator::new();

        visit_pixels(source, row_stride, layout, |color, alpha| {
            if exclude_fully_transparent && alpha == 0.0 {
                return;
            }
            accumulator.observe(color);
        })?;

        let pixel_count = usize::try_from(accumulator.pixel_count)
            .expect("the bounded JPEG XR pixel count fits usize");

        invariant!(pixel_count > 0);
        invariant!(pixel_count <= source_pixel_count);

        let mut max_cll_estimator = MaxCllEstimator::new(
            NonZeroUsize::new(pixel_count).expect("HDR metrics include at least one pixel"),
        );

        visit_pixels(source, row_stride, layout, |color, alpha| {
            if exclude_fully_transparent && alpha == 0.0 {
                return;
            }
            max_cll_estimator.observe(LinearRgb::new(color));
        })?;

        let estimate = max_cll_estimator
            .finish()
            .expect("the MaxCLL pass visits the measured number of HDR pixels");
        let relative_light_level = estimate.level();

        invariant!(relative_light_level.is_finite());
        invariant!(relative_light_level >= 0.0);

        let max_cll = MaxCll {
            nits: relative_light_level * SC_RGB_REFERENCE_WHITE_NITS,
            channel: jpeg_xr_color_channel(estimate.channel()),
        };

        Ok(accumulator.finish(max_cll))
    }
}

impl MaxCll {
    fn relative_light_level(self) -> f32 {
        invariant!(self.nits.is_finite());
        invariant!(self.nits >= 0.0);
        self.nits / SC_RGB_REFERENCE_WHITE_NITS
    }
}

struct HDRMetricAccumulator {
    pixel_count: u64,
    luminance_sum_nits: f64,
    max_luminance_nits: f64,
    min_luminance_nits: f64,
    rec709_pixels: u64,
    dci_p3_pixels: u64,
}

impl HDRMetricAccumulator {
    const fn new() -> Self {
        Self {
            pixel_count: 0,
            luminance_sum_nits: 0.0,
            max_luminance_nits: f64::NEG_INFINITY,
            min_luminance_nits: f64::INFINITY,
            rec709_pixels: 0,
            dci_p3_pixels: 0,
        }
    }

    fn observe(&mut self, color: [f32; 3]) {
        let color = color.map(sanitize_metric_sample);
        let luminance = (0.212_6 * color[0] + 0.715_2 * color[1] + 0.072_2 * color[2]).max(0.0)
            * f64::from(SC_RGB_REFERENCE_WHITE_NITS);

        self.pixel_count += 1;
        self.luminance_sum_nits += luminance;
        self.max_luminance_nits = self.max_luminance_nits.max(luminance);
        self.min_luminance_nits = self.min_luminance_nits.min(luminance);

        match gamut_membership(color) {
            GamutMembership::Rec709 => self.rec709_pixels += 1,
            GamutMembership::DisplayP3Only => self.dci_p3_pixels += 1,
            GamutMembership::OutsideDisplayP3 => {}
        }
    }

    fn finish(self, max_cll: MaxCll) -> HDRMetrics {
        invariant!(self.pixel_count > 0);
        invariant!(self.max_luminance_nits.is_finite());
        invariant!(self.min_luminance_nits.is_finite());

        let pixel_count = f64::from(
            u32::try_from(self.pixel_count)
                .expect("the bounded JPEG XR pixel count fits u32 metadata arithmetic"),
        );

        HDRMetrics {
            max_cll,
            max_luminance_nits: nonnegative_f64_to_f32(self.max_luminance_nits),
            average_luminance_nits: nonnegative_f64_to_f32(self.luminance_sum_nits / pixel_count),
            min_luminance_nits: nonnegative_f64_to_f32(self.min_luminance_nits),
            rec709_percentage: percentage(self.rec709_pixels, self.pixel_count),
            dci_p3_percentage: percentage(self.dci_p3_pixels, self.pixel_count),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GamutMembership {
    Rec709,
    DisplayP3Only,
    OutsideDisplayP3,
}

fn gamut_membership(color: [f64; 3]) -> GamutMembership {
    let scale = color.iter().copied().map(f64::abs).fold(1.0_f64, f64::max);
    let epsilon = 1.0e-6 * scale;

    if color.iter().all(|channel| *channel >= -epsilon) {
        return GamutMembership::Rec709;
    }

    let display_p3 = [
        0.822_592_87 * color[0] + 0.177_533_95 * color[1],
        0.033_199_51 * color[0] + 0.966_783_50 * color[1],
        0.017_085_35 * color[0] + 0.072_395_72 * color[1] + 0.910_301_48 * color[2],
    ];
    if display_p3.iter().all(|channel| *channel >= -epsilon) {
        GamutMembership::DisplayP3Only
    } else {
        GamutMembership::OutsideDisplayP3
    }
}

fn sanitize_metric_sample(value: f32) -> f64 {
    if value.is_nan() {
        0.0
    } else if value == f32::INFINITY {
        65_504.0
    } else if value == f32::NEG_INFINITY {
        -65_504.0
    } else {
        f64::from(value)
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "bounded metadata values are intentionally returned as the decoder's f32 scalar type"
)]
fn nonnegative_f64_to_f32(value: f64) -> f32 {
    invariant!(value.is_finite());
    invariant!(value >= 0.0);
    value.min(f64::from(f32::MAX)) as f32
}

fn percentage(part: u64, total: u64) -> f32 {
    invariant!(part <= total);
    invariant!(total > 0);
    let part = u32::try_from(part).expect("the bounded JPEG XR pixel count fits u32");
    let total = u32::try_from(total).expect("the bounded JPEG XR pixel count fits u32");
    percentage_from_u32(part, total)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "a bounded 0..=100 metadata percentage is intentionally stored as f32"
)]
fn percentage_from_u32(part: u32, total: u32) -> f32 {
    (f64::from(part) * 100.0 / f64::from(total)) as f32
}

const fn jpeg_xr_color_channel(channel: ToneColorChannel) -> JPEGXRColorChannel {
    match channel {
        ToneColorChannel::Red => JPEGXRColorChannel::Red,
        ToneColorChannel::Green => JPEGXRColorChannel::Green,
        ToneColorChannel::Blue => JPEGXRColorChannel::Blue,
    }
}

fn visit_pixels(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
    mut visitor: impl FnMut([f32; 3], f32),
) -> Result<()> {
    invariant!(row_stride >= layout.bytes_per_pixel);

    let mut rows = source.chunks_exact(row_stride);
    for row in &mut rows {
        let mut pixels = row.chunks_exact(layout.bytes_per_pixel);

        for pixel in &mut pixels {
            let (color, alpha) = layout.read_pixel(pixel)?;
            visitor(color, alpha);
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

fn has_visible_alpha(source: &[u8], row_stride: usize, layout: PixelLayout) -> Result<bool> {
    invariant!(layout.has_alpha);

    let mut has_visible_alpha = false;
    visit_pixels(source, row_stride, layout, |_color, alpha| {
        has_visible_alpha |= alpha > 0.0;
    })?;
    Ok(has_visible_alpha)
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

fn hdr_to_srgb8(color: [f32; 3], mapper: &impl ToneMapper) -> [u8; 3] {
    mapper
        .map(LinearRgb::new(color))
        .components()
        .map(linear_to_srgb)
        .map(normalized_to_u8)
}

fn normalize_alpha(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
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

    fn float_rgb_layout() -> PixelLayout {
        PixelLayout {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 3,
            bytes_per_pixel: 12,
            has_alpha: false,
            premultiplied_alpha: false,
            blue_first: false,
        }
    }

    fn float_rgb_source(colors: &[[f32; 3]]) -> Vec<u8> {
        colors
            .iter()
            .flatten()
            .flat_map(|channel| channel.to_ne_bytes())
            .collect()
    }

    fn float_rgba_layout() -> PixelLayout {
        PixelLayout {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 4,
            bytes_per_pixel: 16,
            has_alpha: true,
            premultiplied_alpha: false,
            blue_first: false,
        }
    }

    fn float_rgba_source(colors: &[[f32; 4]]) -> Vec<u8> {
        colors
            .iter()
            .flatten()
            .flat_map(|channel| channel.to_ne_bytes())
            .collect()
    }

    fn assert_approximately_equal(actual: f32, expected: f32, tolerance: f32) {
        assert!((actual - expected).abs() <= tolerance);
    }

    #[test]
    fn decodes_half_precision_boundaries() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn tone_mapper_output_is_encoded_as_srgb() {
        let mapper = LinearShoulder::new(4.0).unwrap();
        assert_eq!(hdr_to_srgb8([0.0; 3], &mapper), [0; 3]);
        let reference_white = hdr_to_srgb8([1.0; 3], &mapper);
        let hdr_white = hdr_to_srgb8([4.0; 3], &mapper);

        assert!(reference_white[0] >= 235 && reference_white[0] <= 245);
        assert!(hdr_white[0] > reference_white[0]);
        assert_eq!(hdr_white[0], u8::MAX);
        assert_eq!(reference_white[0], reference_white[1]);
        assert_eq!(hdr_white[1], hdr_white[2]);
    }

    #[test]
    fn max_cll_rejects_the_brightest_point_zero_one_percent() {
        let layout = float_rgb_layout();
        let mut source = Vec::with_capacity(10_000 * layout.bytes_per_pixel);
        source.extend(
            std::iter::repeat_n(1.0_f32, 9_998)
                .chain([4.0, 126.0])
                .flat_map(|value| [value; 3])
                .flat_map(f32::to_ne_bytes),
        );

        let metrics = HDRMetrics::estimate(&source, source.len(), layout).unwrap();
        let metadata = JPEGXRMetadata::new(layout, Some(metrics));

        assert_eq!(metrics.max_cll.nits, 320.0);
        assert!(metadata.is_hdr());
        assert_eq!(metadata.bits_per_channel(), 32);
        assert_eq!(metadata.color_channels(), 3);
        assert!(!metadata.has_alpha());
        assert_eq!(metadata.max_cll_scrgb(), Some(4.0));
        assert_eq!(metadata.max_cll_nits(), Some(320.0));
        assert_eq!(metadata.max_cll_channel(), Some(JPEGXRColorChannel::Red));
    }

    #[test]
    fn hdr_metrics_use_signed_linear_scrgb_and_disjoint_gamuts() {
        let colors = [
            [1.0, 1.0, 1.0],
            [-0.1, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.000_000_5, 0.0, 0.0],
        ];
        let source = float_rgb_source(&colors);
        let metrics = HDRMetrics::estimate(&source, source.len(), float_rgb_layout()).unwrap();

        assert_approximately_equal(metrics.max_luminance_nits, 80.0, 0.000_1);
        assert_approximately_equal(metrics.average_luminance_nits, 33.878_8, 0.000_1);
        assert_eq!(metrics.min_luminance_nits, 0.0);
        assert_eq!(metrics.rec709_percentage, 50.0);
        assert_eq!(metrics.dci_p3_percentage, 25.0);
    }

    #[test]
    fn max_cll_reports_the_winning_color_channel() {
        let source = float_rgb_source(&[[1.0, 2.0, 5.0]]);
        let metrics = HDRMetrics::estimate(&source, source.len(), float_rgb_layout()).unwrap();
        let metadata = JPEGXRMetadata::new(float_rgb_layout(), Some(metrics));

        assert_eq!(metadata.max_cll_scrgb(), Some(5.0));
        assert_eq!(metadata.max_cll_channel(), Some(JPEGXRColorChannel::Blue));
        assert_eq!(
            metadata.max_cll_channel().map(JPEGXRColorChannel::symbol),
            Some('B')
        );
    }

    #[test]
    fn normalization_quantizes_alpha_to_eight_bits() {
        let layout = float_rgba_layout();
        let source: Vec<u8> = std::iter::repeat_n([0.01_f32, 0.01, 0.01, 0.5], 16)
            .flatten()
            .flat_map(f32::to_ne_bytes)
            .collect();

        let rgba = normalize(&source, 4, 4, 64, layout).unwrap().rgba;

        assert!(rgba.iter().skip(3).step_by(4).all(|alpha| *alpha == 128));
    }

    #[test]
    fn hidden_transparent_rgb_does_not_affect_hdr_metrics() {
        let layout = float_rgba_layout();
        let source = float_rgba_source(&[[1.0, 1.0, 1.0, 1.0], [100.0, -100.0, 0.0, 0.0]]);

        let metrics = HDRMetrics::estimate(&source, source.len(), layout).unwrap();

        assert_eq!(metrics.max_cll.nits, SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.max_luminance_nits, SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.average_luminance_nits, SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.min_luminance_nits, SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.rec709_percentage, 100.0);
        assert_eq!(metrics.dci_p3_percentage, 0.0);
    }

    #[test]
    fn all_zero_hdr_alpha_plane_is_visible_to_metrics_and_made_opaque() {
        let layout = float_rgba_layout();
        let source = float_rgba_source(&[[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 4.0, 0.0]]);

        let metrics = HDRMetrics::estimate(&source, source.len(), layout).unwrap();
        let rgba = normalize(&source, 2, 1, 32, layout).unwrap().rgba;

        assert_eq!(metrics.max_cll.nits, 4.0 * SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.max_cll.channel, JPEGXRColorChannel::Blue);
        assert!(
            rgba.iter()
                .skip(3)
                .step_by(4)
                .all(|alpha| *alpha == u8::MAX)
        );
    }
}
