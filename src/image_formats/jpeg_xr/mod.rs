//! Decodes bounded JPEG XR images and tone-maps HDR pixels to SDR RGBA8.
//!
//! The JPEG XR reference codec produces pixels in their native WIC representation. Unsigned RGB
//! and grayscale samples are normalized as sRGB. Fixed-point, half-float, floating-point, and RGBE
//! samples are interpreted as linear scRGB and encoded with the sRGB transfer function. By default,
//! HDR values are mapped component-wise with extended Reinhard using a `MaxCLL` white point. Its
//! white point is floored at display white so content already inside the SDR range is unchanged.
//!
//! By default, `MaxCLL` is estimated from the 99.99th-percentile `max(R, G, B)` light level,
//! rejecting isolated outliers at the cost of clipping the brightest 0.01% of a sufficiently large
//! image. [`DecodeOptions`] can instead select the true maximum. Callers can also select any
//! built-in [`ToneMappingMethod`]; white-point methods use matching statistics estimated from the
//! decoded image.
//! A wholly empty, non-premultiplied HDR alpha plane is treated as unspecified and made opaque,
//! matching JPEG XR screenshots that store zero in an otherwise unused alpha channel.

mod error;

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::io::Cursor;
use std::num::NonZeroUsize;

use ::jpegxr::{
    BitDepthBits, ColorFormat, ImageDecode, JXRError, PixelFormat as CodecPixelFormat, PixelInfo,
};
use tonemapping::{
    AcesApproximate, AcesFitted, Clamp, ColorChannel as ToneColorChannel,
    ExtendedLuminanceReinhard, ExtendedReinhard, Hable, LinearRGB, LuminanceReinhard,
    LuminanceWhitePoint, MaxCllEstimator, MaxCllMode, Reinhard, ReinhardJodie, ScaledClamp,
    ToneMapper, ToneMappingMethod, WhitePoint,
};
use zerocopy::FromBytes;

use super::{DIMENSION_MAX, DecodedImage, PIXELS_MAX};
use error::error;

pub use error::{Error, JPEGXRError, JPEGXRLimit, Result};

/// The four-byte signature at the beginning of a JPEG XR file.
pub const SIGNATURE: [u8; 4] = [0x49, 0x49, 0xbc, 0x01];

const SC_RGB_REFERENCE_WHITE_NITS: f32 = 80.0;
const SOURCE_BUFFER_MAX: usize = 512 * 1024 * 1024;
// Three f32 channels plus staged alpha occupy roughly 13 KiB, leaving room in common L1 caches.
const HDR_BATCH_PIXELS: usize = 1_024;

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

/// Configures HDR normalization during JPEG XR decoding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeOptions {
    tone_mapping: ToneMappingMethod,
    max_cll_mode: MaxCllMode,
}

impl DecodeOptions {
    /// Creates options using `tone_mapping` and `max_cll_mode`.
    #[must_use]
    pub const fn new(tone_mapping: ToneMappingMethod, max_cll_mode: MaxCllMode) -> Self {
        Self {
            tone_mapping,
            max_cll_mode,
        }
    }

    /// Returns the selected HDR tone-mapping operator.
    #[must_use]
    pub const fn tone_mapping(self) -> ToneMappingMethod {
        self.tone_mapping
    }

    /// Returns the selected `MaxCLL` estimator mode.
    #[must_use]
    pub const fn max_cll_mode(self) -> MaxCllMode {
        self.max_cll_mode
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
    /// The estimate follows the [`MaxCllMode`] used for decoding. SDR sources return `None` because
    /// they do not pass through the HDR tone-mapping pipeline. A finite result above the `f32`
    /// range saturates at `f32::MAX`.
    #[must_use]
    pub fn max_cll_nits(self) -> Option<f32> {
        self.hdr_metrics.map(|metrics| metrics.max_cll.nits())
    }

    /// Returns the estimated maximum content light level in scRGB units for HDR sources.
    #[must_use]
    pub fn max_cll_scrgb(self) -> Option<f32> {
        self.hdr_metrics
            .map(|metrics| metrics.max_cll.relative_light_level())
    }

    /// Returns the color channel that determines the selected `MaxCLL` value.
    #[must_use]
    pub const fn max_cll_channel(self) -> Option<JPEGXRColorChannel> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.max_cll.channel),
            None => None,
        }
    }

    /// Returns the `MaxCLL` mode used for HDR normalization.
    #[must_use]
    pub const fn max_cll_mode(self) -> Option<MaxCllMode> {
        match self.hdr_metrics {
            Some(metrics) => Some(metrics.max_cll_mode),
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
/// use extended Reinhard with a `MaxCLL` white point before conversion to sRGB. `MaxCLL` is
/// estimated from the 99.99th-percentile brightest pixel to prevent isolated outliers from dimming
/// the image. The white point is floored at display white so in-range content is unchanged.
/// Premultiplied inputs are returned with straight alpha.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode(bytes: impl AsRef<[u8]>) -> Result<DecodedImage> {
    decode_with_options(bytes, DecodeOptions::default())
}

/// Decodes JPEG XR pixels using the selected HDR normalization `options`.
///
/// SDR sources do not pass through tone mapping. Image-derived white points are floored at display
/// white; methods without a white point apply their curves to every HDR source.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode_with_options(
    bytes: impl AsRef<[u8]>,
    options: DecodeOptions,
) -> Result<DecodedImage> {
    Ok(decode_with_metadata_and_options(bytes, options)?.into_image())
}

/// Decodes JPEG XR pixels using the selected HDR tone-mapping `method`.
///
/// SDR sources do not pass through tone mapping. Image-derived white points are floored at display
/// white; methods without a white point apply their curves to every HDR source.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode_with_tone_mapping(
    bytes: impl AsRef<[u8]>,
    method: ToneMappingMethod,
) -> Result<DecodedImage> {
    decode_with_options(
        bytes,
        DecodeOptions::new(method, MaxCllMode::Percentile99_99),
    )
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
    decode_with_metadata_and_options(bytes, DecodeOptions::default())
}

/// Decodes JPEG XR pixels and metadata using the selected HDR tone-mapping `method`.
///
/// Image-dependent parameters are derived from the decoded source. Component-wise white-point
/// methods use p99.99 `MaxCLL`, while extended luminance Reinhard uses p99.99 Rec. 709 luminance.
/// SDR sources do not pass through tone mapping. Image-derived white points are floored at display
/// white; methods without a white point apply their curves to every HDR source.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode_with_metadata_and_tone_mapping(
    bytes: impl AsRef<[u8]>,
    method: ToneMappingMethod,
) -> Result<DecodedJPEGXR> {
    decode_with_metadata_and_options(
        bytes,
        DecodeOptions::new(method, MaxCllMode::Percentile99_99),
    )
}

/// Decodes JPEG XR pixels and metadata using the selected HDR normalization `options`.
///
/// Image-dependent parameters are derived from the decoded source. Component-wise white-point
/// methods use the selected [`MaxCllMode`], while extended luminance Reinhard always uses p99.99
/// Rec. 709 luminance. SDR sources do not pass through tone mapping. Image-derived white points
/// are floored at display white; methods without a white point apply their curves to every HDR
/// source.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode_with_metadata_and_options(
    bytes: impl AsRef<[u8]>,
    options: DecodeOptions,
) -> Result<DecodedJPEGXR> {
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
    let normalized = normalize(&source, width, height, row_stride, layout, options)?;
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
    RGBE,
    Unsigned8,
    Unsigned16,
}

impl SampleEncoding {
    const fn bytes(self) -> usize {
        match self {
            Self::Unsigned8 | Self::RGBE => 1,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 2,
            Self::Fixed32 | Self::Float32 => 4,
        }
    }

    const fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Fixed16 | Self::Fixed32 | Self::Float16 | Self::Float32 | Self::RGBE
        )
    }

    const fn bits_per_channel(self) -> u8 {
        match self {
            Self::Unsigned8 | Self::RGBE => 8,
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
            ColorFormat::RGBE if encoding == SampleEncoding::RGBE => 3,
            other => {
                return Err(unsupported(format, &format!("{other:?} color data")));
            }
        };

        let has_alpha = info.has_alpha();
        let source_channels = info.channels();
        let expected_channels = color_channels + usize::from(has_alpha);

        if encoding != SampleEncoding::RGBE && source_channels != expected_channels {
            return Err(unsupported(
                format,
                "channel metadata is inconsistent with the color format",
            ));
        }

        if encoding == SampleEncoding::RGBE && has_alpha {
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
        if self.encoding == SampleEncoding::RGBE {
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
        return Ok(SampleEncoding::RGBE);
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
    options: DecodeOptions,
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
        Some(HDRMetrics::estimate(
            source,
            row_stride,
            layout,
            options.max_cll_mode(),
            options.tone_mapping() == ToneMappingMethod::ExtendedLuminanceReinhard,
        )?)
    } else {
        None
    };
    let mut rgba = Vec::with_capacity(output_len);
    let has_nonzero_alpha = if let Some(metrics) = hdr_metrics {
        let mapper = ResolvedToneMapper::new(options.tone_mapping(), metrics);
        if matches!(mapper, ResolvedToneMapper::Clamp) {
            append_hdr_pixels_scalar(source, width, row_stride, layout, &Clamp, &mut rgba)?
        } else {
            append_hdr_pixels(source, width, row_stride, layout, &mapper, &mut rgba)?
        }
    } else {
        append_sdr_pixels(source, width, row_stride, layout, &mut rgba)?
    };
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

fn append_sdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    rgba: &mut Vec<u8>,
) -> Result<bool> {
    let mut has_nonzero_alpha = false;
    for row in source.chunks_exact(row_stride) {
        for x in 0..width {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;
            has_nonzero_alpha |= alpha > 0.0;
            let color = color.map(normalized_to_u8);
            rgba.extend_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }
    Ok(has_nonzero_alpha)
}

fn append_hdr_pixels_scalar(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &impl ToneMapper,
    rgba: &mut Vec<u8>,
) -> Result<bool> {
    let mut has_nonzero_alpha = false;
    for row in source.chunks_exact(row_stride) {
        for x in 0..width {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;
            has_nonzero_alpha |= alpha > 0.0;
            let color = display_linear_to_srgb8(mapper.map(LinearRGB::new(color)));
            rgba.extend_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }
    Ok(has_nonzero_alpha)
}

fn append_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &impl ToneMapper,
    rgba: &mut Vec<u8>,
) -> Result<bool> {
    let row_count = source.len() / row_stride;
    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");
    let batch_capacity = HDR_BATCH_PIXELS.min(pixel_count);
    let mut colors = Vec::with_capacity(batch_capacity);
    let mut alphas = Vec::with_capacity(batch_capacity);
    let mut has_nonzero_alpha = false;

    for row in source.chunks_exact(row_stride) {
        for x in 0..width {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;
            has_nonzero_alpha |= alpha > 0.0;
            colors.push(LinearRGB::new(color));
            alphas.push(normalized_to_u8(alpha));

            if colors.len() == HDR_BATCH_PIXELS {
                append_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba);
            }
        }
    }

    append_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba);
    Ok(has_nonzero_alpha)
}

fn append_tone_mapped_batch(
    mapper: &impl ToneMapper,
    colors: &mut Vec<LinearRGB>,
    alphas: &mut Vec<u8>,
    rgba: &mut Vec<u8>,
) {
    invariant_eq!(colors.len(), alphas.len());
    if colors.is_empty() {
        return;
    }
    mapper.map_in_place(colors);

    for (color, alpha) in colors.iter().copied().zip(alphas.iter().copied()) {
        let color = display_linear_to_srgb8(color);
        rgba.extend_from_slice(&[color[0], color[1], color[2], alpha]);
    }
    colors.clear();
    alphas.clear();
}

fn pixel_at(row: &[u8], x: usize, layout: PixelLayout) -> Result<&[u8]> {
    let start = x
        .checked_mul(layout.bytes_per_pixel)
        .ok_or_else(|| error(JPEGXRError::Output("JPEG XR pixel offset exceeds usize")))?;
    let end = start + layout.bytes_per_pixel;
    row.get(start..end)
        .ok_or_else(|| error(JPEGXRError::Output("JPEG XR pixel exceeds its decoded row")))
}

#[derive(Debug)]
struct NormalizedImage {
    rgba: Vec<u8>,
    hdr_metrics: Option<HDRMetrics>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaxCll {
    relative_light_level: f32,
    channel: JPEGXRColorChannel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HDRMetrics {
    max_cll: MaxCll,
    max_cll_mode: MaxCllMode,
    luminance_white_point: Option<LuminanceWhitePoint>,
    max_luminance_nits: f32,
    average_luminance_nits: f32,
    min_luminance_nits: f32,
    rec709_percentage: f32,
    dci_p3_percentage: f32,
}

impl HDRMetrics {
    fn estimate(
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        max_cll_mode: MaxCllMode,
        estimate_luminance_white_point: bool,
    ) -> Result<Self> {
        invariant!(layout.encoding.is_hdr());
        invariant!(row_stride >= layout.bytes_per_pixel);

        let pixels_per_row = row_stride / layout.bytes_per_pixel;
        let row_count = source.len() / row_stride;

        let source_pixel_count = pixels_per_row
            .checked_mul(row_count)
            .expect("validated JPEG XR pixel count fits usize");

        invariant!(source_pixel_count > 0);

        let visible_alpha_pixels = if layout.has_alpha {
            visible_alpha_pixel_count(source, row_stride, layout)?
        } else {
            source_pixel_count
        };
        let exclude_fully_transparent = layout.has_alpha && visible_alpha_pixels > 0;
        let pixel_count = if exclude_fully_transparent {
            visible_alpha_pixels
        } else {
            source_pixel_count
        };

        invariant!(pixel_count > 0);
        invariant!(pixel_count <= source_pixel_count);

        let mut accumulator = HDRMetricAccumulator::new();
        let mut max_cll_estimator = MaxCllEstimator::with_mode(
            NonZeroUsize::new(pixel_count).expect("HDR metrics include at least one pixel"),
            max_cll_mode,
        );
        let mut luminance_white_point_estimator =
            estimate_luminance_white_point.then(|| LuminanceWhitePointEstimator::new(pixel_count));
        let mut max_cll_batch = Vec::with_capacity(HDR_BATCH_PIXELS.min(pixel_count));

        visit_pixels(source, row_stride, layout, |color, alpha| {
            if exclude_fully_transparent && alpha == 0.0 {
                return;
            }
            accumulator.observe(color);
            let color = LinearRGB::new(color);
            if let Some(estimator) = &mut luminance_white_point_estimator {
                estimator.observe(color);
            }
            max_cll_batch.push(color);
            if max_cll_batch.len() == HDR_BATCH_PIXELS {
                max_cll_estimator.observe_many(&max_cll_batch);
                max_cll_batch.clear();
            }
        })?;
        max_cll_estimator.observe_many(&max_cll_batch);

        invariant_eq!(
            usize::try_from(accumulator.pixel_count)
                .expect("the bounded JPEG XR pixel count fits usize"),
            pixel_count
        );

        let estimate = max_cll_estimator
            .finish()
            .expect("the HDR metrics pass visits the measured number of pixels");
        let relative_light_level = estimate.level();

        invariant!(relative_light_level.is_finite());
        invariant!(relative_light_level >= 0.0);

        let max_cll = MaxCll {
            relative_light_level,
            channel: jpeg_xr_color_channel(estimate.channel()),
        };
        let luminance_white_point =
            luminance_white_point_estimator.and_then(LuminanceWhitePointEstimator::finish);

        Ok(accumulator.finish(max_cll, max_cll_mode, luminance_white_point))
    }
}

impl MaxCll {
    fn relative_light_level(self) -> f32 {
        invariant!(self.relative_light_level.is_finite());
        invariant!(self.relative_light_level >= 0.0);
        self.relative_light_level
    }

    fn nits(self) -> f32 {
        nonnegative_f64_to_f32(
            f64::from(self.relative_light_level) * f64::from(SC_RGB_REFERENCE_WHITE_NITS),
        )
    }
}

fn hdr_white_point(max_cll: MaxCll) -> WhitePoint {
    WhitePoint::new(max_cll.relative_light_level().max(1.0))
        .expect("MaxCLL floored at display white is a positive finite white point")
}

enum ResolvedToneMapper {
    Clamp,
    ScaledClamp(ScaledClamp),
    Reinhard,
    ExtendedReinhard(ExtendedReinhard),
    LuminanceReinhard,
    ExtendedLuminanceReinhard(ExtendedLuminanceReinhard),
    ReinhardJodie,
    Uncharted2,
    AcesFitted,
    AcesApproximate,
}

impl ResolvedToneMapper {
    fn new(method: ToneMappingMethod, metrics: HDRMetrics) -> Self {
        let white_point = hdr_white_point(metrics.max_cll);

        match method {
            ToneMappingMethod::Clamp => Self::Clamp,
            ToneMappingMethod::ScaledClamp => Self::ScaledClamp(ScaledClamp::new(white_point)),
            ToneMappingMethod::Reinhard => Self::Reinhard,
            ToneMappingMethod::ExtendedReinhard => {
                Self::ExtendedReinhard(ExtendedReinhard::new(white_point))
            }
            ToneMappingMethod::LuminanceReinhard => Self::LuminanceReinhard,
            ToneMappingMethod::ExtendedLuminanceReinhard => {
                let luminance = metrics
                    .luminance_white_point
                    .map_or(1.0, LuminanceWhitePoint::luminance)
                    .max(1.0);
                let luminance_white_point = LuminanceWhitePoint::new(luminance).expect(
                    "p99.99 luminance floored at display white is a positive finite white point",
                );
                Self::ExtendedLuminanceReinhard(ExtendedLuminanceReinhard::new(
                    luminance_white_point,
                ))
            }
            ToneMappingMethod::ReinhardJodie => Self::ReinhardJodie,
            ToneMappingMethod::Hable => Self::Uncharted2,
            ToneMappingMethod::ACESFitted => Self::AcesFitted,
            ToneMappingMethod::ACESApproximate => Self::AcesApproximate,
        }
    }
}

impl ToneMapper for ResolvedToneMapper {
    fn map(&self, color: LinearRGB) -> LinearRGB {
        match self {
            Self::Clamp => Clamp.map(color),
            Self::ScaledClamp(mapper) => mapper.map(color),
            Self::Reinhard => Reinhard.map(color),
            Self::ExtendedReinhard(mapper) => mapper.map(color),
            Self::LuminanceReinhard => LuminanceReinhard.map(color),
            Self::ExtendedLuminanceReinhard(mapper) => mapper.map(color),
            Self::ReinhardJodie => ReinhardJodie.map(color),
            Self::Uncharted2 => Hable.map(color),
            Self::AcesFitted => AcesFitted.map(color),
            Self::AcesApproximate => AcesApproximate.map(color),
        }
    }

    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        match self {
            Self::Clamp => Clamp.map_in_place(colors),
            Self::ScaledClamp(mapper) => mapper.map_in_place(colors),
            Self::Reinhard => Reinhard.map_in_place(colors),
            Self::ExtendedReinhard(mapper) => mapper.map_in_place(colors),
            Self::LuminanceReinhard => LuminanceReinhard.map_in_place(colors),
            Self::ExtendedLuminanceReinhard(mapper) => mapper.map_in_place(colors),
            Self::ReinhardJodie => ReinhardJodie.map_in_place(colors),
            Self::Uncharted2 => Hable.map_in_place(colors),
            Self::AcesFitted => AcesFitted.map_in_place(colors),
            Self::AcesApproximate => AcesApproximate.map_in_place(colors),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OrderedLuminance(f32);

impl PartialEq for OrderedLuminance {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedLuminance {}

impl PartialOrd for OrderedLuminance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedLuminance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

struct LuminanceWhitePointEstimator {
    retained: usize,
    luminances: BinaryHeap<Reverse<OrderedLuminance>>,
}

impl LuminanceWhitePointEstimator {
    fn new(pixel_count: usize) -> Self {
        invariant!(pixel_count > 0);
        let retained = pixel_count / 10_000 + 1;
        Self {
            retained,
            luminances: BinaryHeap::with_capacity(retained),
        }
    }

    fn observe(&mut self, color: LinearRGB) {
        let luminance = OrderedLuminance(color.luminance());
        if self.luminances.len() < self.retained {
            self.luminances.push(Reverse(luminance));
        } else if let Some(mut threshold) = self.luminances.peek_mut()
            && luminance > threshold.0
        {
            *threshold = Reverse(luminance);
        }
    }

    fn finish(self) -> Option<LuminanceWhitePoint> {
        self.luminances
            .peek()
            .and_then(|luminance| LuminanceWhitePoint::new(luminance.0.0))
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

    fn finish(
        self,
        max_cll: MaxCll,
        max_cll_mode: MaxCllMode,
        luminance_white_point: Option<LuminanceWhitePoint>,
    ) -> HDRMetrics {
        invariant!(self.pixel_count > 0);
        invariant!(self.max_luminance_nits.is_finite());
        invariant!(self.min_luminance_nits.is_finite());

        let pixel_count = f64::from(
            u32::try_from(self.pixel_count)
                .expect("the bounded JPEG XR pixel count fits u32 metadata arithmetic"),
        );

        HDRMetrics {
            max_cll,
            max_cll_mode,
            luminance_white_point,
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
        f64::from(f32::MAX)
    } else if value == f32::NEG_INFINITY {
        -f64::from(f32::MAX)
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

fn visible_alpha_pixel_count(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
) -> Result<usize> {
    invariant!(layout.has_alpha);

    let mut visible_pixels = 0_usize;
    visit_pixels(source, row_stride, layout, |_color, alpha| {
        if alpha > 0.0 {
            visible_pixels = visible_pixels
                .checked_add(1)
                .expect("validated JPEG XR pixel count fits usize");
        }
    })?;
    Ok(visible_pixels)
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
        SampleEncoding::Unsigned8 => {
            f32::from(read_sample::<u8>(bytes)) / f32::from(u8::MAX)
        }
        SampleEncoding::Unsigned16 => {
            f32::from(read_sample::<u16>(bytes)) / f32::from(u16::MAX)
        }
        SampleEncoding::Fixed16 => f32::from(read_sample::<i16>(bytes)) / 8192.0,
        SampleEncoding::Fixed32 => fixed32_to_f32(read_sample::<i32>(bytes)),
        SampleEncoding::Float16 => half_to_f32(read_sample::<u16>(bytes)),
        SampleEncoding::Float32 => read_sample::<f32>(bytes),
        SampleEncoding::RGBE => {
            unreachable!("RGBE pixels are decoded as a unit")
        }
    }
}

fn read_sample<T: FromBytes + Sized>(bytes: &[u8]) -> T {
    T::read_from_bytes(bytes).expect("JPEG XR sample length must match its encoding")
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

fn display_linear_to_srgb8(color: LinearRGB) -> [u8; 3] {
    color.components().map(linear_to_srgb).map(normalized_to_u8)
}

#[cfg(test)]
fn hdr_to_srgb8(color: [f32; 3], mapper: &impl ToneMapper) -> [u8; 3] {
    display_linear_to_srgb8(mapper.map(LinearRGB::new(color)))
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

    fn normalize_float_rgb(
        colors: &[[f32; 3]],
        width: usize,
        method: ToneMappingMethod,
    ) -> Vec<u8> {
        normalize_float_rgb_with_options(
            colors,
            width,
            DecodeOptions::new(method, MaxCllMode::Percentile99_99),
        )
    }

    fn normalize_float_rgb_with_options(
        colors: &[[f32; 3]],
        width: usize,
        options: DecodeOptions,
    ) -> Vec<u8> {
        assert_eq!(
            colors.len() % width,
            0,
            "test pixels must contain complete rows"
        );
        let layout = float_rgb_layout();
        let source = float_rgb_source(colors);
        let height = colors.len() / width;
        let row_stride = width * layout.bytes_per_pixel;

        normalize(
            &source,
            u32::try_from(width).expect("test width fits u32"),
            u32::try_from(height).expect("test height fits u32"),
            row_stride,
            layout,
            options,
        )
        .expect("synthetic HDR pixels normalize")
        .rgba
    }

    fn assert_hdr_normalization_matches_scalar(
        colors: &[[f32; 3]],
        width: usize,
        mapper: &impl ToneMapper,
    ) {
        let actual = normalize_float_rgb(colors, width, ToneMappingMethod::default());
        let expected: Vec<_> = colors
            .iter()
            .copied()
            .flat_map(|color| {
                let [red, green, blue] = hdr_to_srgb8(color, mapper);
                [red, green, blue, u8::MAX]
            })
            .collect();

        assert_eq!(actual, expected);
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
        let mapper = ExtendedReinhard::new(WhitePoint::new(4.0).unwrap());
        assert_eq!(hdr_to_srgb8([0.0; 3], &mapper), [0; 3]);
        let reference_white = hdr_to_srgb8([1.0; 3], &mapper);
        let hdr_white = hdr_to_srgb8([4.0; 3], &mapper);

        assert!(reference_white[0] >= 190 && reference_white[0] <= 195);
        assert!(hdr_white[0] > reference_white[0]);
        assert_eq!(hdr_white[0], u8::MAX);
        assert_eq!(reference_white[0], reference_white[1]);
        assert_eq!(hdr_white[1], hdr_white[2]);
    }

    #[test]
    fn every_selectable_method_normalizes_hdr_pixels() {
        let colors = [[4.0, 2.0, 1.0], [0.18, 0.5, 1.5]];

        for method in ToneMappingMethod::ALL {
            let rgba = normalize_float_rgb(&colors, 2, method);

            assert_eq!(rgba.len(), colors.len() * 4);
            assert!(
                rgba.iter()
                    .skip(3)
                    .step_by(4)
                    .all(|alpha| *alpha == u8::MAX)
            );
        }
    }

    #[test]
    fn selected_method_changes_hdr_normalization() {
        let colors = [[4.0, 2.0, 1.0]];
        let clamped = normalize_float_rgb(&colors, 1, ToneMappingMethod::Clamp);
        let reinhard = normalize_float_rgb(&colors, 1, ToneMappingMethod::Reinhard);

        assert_eq!(clamped, [u8::MAX, u8::MAX, u8::MAX, u8::MAX]);
        assert_eq!(reinhard, [231, 213, 188, u8::MAX]);
    }

    #[test]
    fn decode_options_default_to_percentile_extended_reinhard() {
        let options = DecodeOptions::default();

        assert_eq!(options.tone_mapping(), ToneMappingMethod::ExtendedReinhard);
        assert_eq!(options.max_cll_mode(), MaxCllMode::Percentile99_99);
    }

    #[test]
    fn extended_luminance_method_uses_the_luminance_white_point() {
        let color = [4.0, 2.0, 1.0];
        let source = float_rgb_source(&[color]);
        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            float_rgb_layout(),
            MaxCllMode::Percentile99_99,
            true,
        )
        .unwrap();
        let white_point = metrics
            .luminance_white_point
            .expect("a nonblack HDR pixel has a luminance white point");
        let mapper = ExtendedLuminanceReinhard::new(white_point);
        let expected = hdr_to_srgb8(color, &mapper);
        let actual = normalize_float_rgb(&[color], 1, ToneMappingMethod::ExtendedLuminanceReinhard);

        assert_approximately_equal(white_point.luminance(), 2.353, 0.000_1);
        assert_eq!(actual, [expected[0], expected[1], expected[2], u8::MAX]);
    }

    #[test]
    fn white_point_methods_preserve_content_that_fits_the_target() {
        let fitting = MaxCll {
            relative_light_level: 1.0,
            channel: JPEGXRColorChannel::Red,
        };
        let hdr = MaxCll {
            relative_light_level: 4.0,
            channel: JPEGXRColorChannel::Red,
        };

        assert_eq!(hdr_white_point(fitting).level(), 1.0);
        assert_eq!(hdr_white_point(hdr).level(), 4.0);

        let colors = [[0.25, 0.5, 1.0]];
        let expected = normalize_float_rgb(&colors, 1, ToneMappingMethod::Clamp);
        for method in [
            ToneMappingMethod::ScaledClamp,
            ToneMappingMethod::ExtendedReinhard,
            ToneMappingMethod::ExtendedLuminanceReinhard,
        ] {
            assert_eq!(normalize_float_rgb(&colors, 1, method), expected);
        }
        assert_ne!(
            normalize_float_rgb(&colors, 1, ToneMappingMethod::Reinhard),
            expected
        );
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

        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            layout,
            MaxCllMode::Percentile99_99,
            false,
        )
        .unwrap();
        let metadata = JPEGXRMetadata::new(layout, Some(metrics));

        assert_eq!(metrics.max_cll.nits(), 320.0);
        assert!(metadata.is_hdr());
        assert_eq!(metadata.bits_per_channel(), 32);
        assert_eq!(metadata.color_channels(), 3);
        assert!(!metadata.has_alpha());
        assert_eq!(metadata.max_cll_scrgb(), Some(4.0));
        assert_eq!(metadata.max_cll_nits(), Some(320.0));
        assert_eq!(metadata.max_cll_channel(), Some(JPEGXRColorChannel::Red));
        assert_eq!(metadata.max_cll_mode(), Some(MaxCllMode::Percentile99_99));
    }

    #[test]
    fn true_max_cll_mode_drives_metadata_and_normalization() {
        let mut colors = vec![[1.0_f32; 3]; 9_998];
        colors.extend([[4.0, 0.0, 0.0], [0.0, 0.0, 126.0]]);
        let layout = float_rgb_layout();
        let source = float_rgb_source(&colors);
        let width = u32::try_from(colors.len()).expect("test width fits u32");

        let percentile = normalize(
            &source,
            width,
            1,
            source.len(),
            layout,
            DecodeOptions::default(),
        )
        .unwrap();
        let true_maximum = normalize(
            &source,
            width,
            1,
            source.len(),
            layout,
            DecodeOptions::new(ToneMappingMethod::ExtendedReinhard, MaxCllMode::TrueMaximum),
        )
        .unwrap();
        let percentile_metadata = JPEGXRMetadata::new(layout, percentile.hdr_metrics);
        let true_maximum_metadata = JPEGXRMetadata::new(layout, true_maximum.hdr_metrics);

        assert_eq!(percentile_metadata.max_cll_scrgb(), Some(4.0));
        assert_eq!(
            percentile_metadata.max_cll_channel(),
            Some(JPEGXRColorChannel::Red)
        );
        assert_eq!(true_maximum_metadata.max_cll_scrgb(), Some(126.0));
        assert_eq!(true_maximum_metadata.max_cll_nits(), Some(10_080.0));
        assert_eq!(
            true_maximum_metadata.max_cll_channel(),
            Some(JPEGXRColorChannel::Blue)
        );
        assert_eq!(
            true_maximum_metadata.max_cll_mode(),
            Some(MaxCllMode::TrueMaximum)
        );
        assert_ne!(percentile.rgba, true_maximum.rgba);
    }

    #[test]
    fn luminance_white_point_rejects_the_brightest_point_zero_one_percent() {
        let colors: Vec<_> = std::iter::repeat_n([1.0_f32; 3], 9_998)
            .chain([[4.0; 3], [126.0; 3]])
            .collect();
        let source = float_rgb_source(&colors);
        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            float_rgb_layout(),
            MaxCllMode::Percentile99_99,
            true,
        )
        .unwrap();

        assert_eq!(
            metrics
                .luminance_white_point
                .map(LuminanceWhitePoint::luminance),
            Some(4.0)
        );
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
        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            float_rgb_layout(),
            MaxCllMode::Percentile99_99,
            false,
        )
        .unwrap();

        assert_approximately_equal(metrics.max_luminance_nits, 80.0, 0.000_1);
        assert_approximately_equal(metrics.average_luminance_nits, 33.878_8, 0.000_1);
        assert_eq!(metrics.min_luminance_nits, 0.0);
        assert_eq!(metrics.rec709_percentage, 50.0);
        assert_eq!(metrics.dci_p3_percentage, 25.0);
    }

    #[test]
    fn max_cll_reports_the_winning_color_channel() {
        let source = float_rgb_source(&[[1.0, 2.0, 5.0]]);
        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            float_rgb_layout(),
            MaxCllMode::Percentile99_99,
            false,
        )
        .unwrap();
        let metadata = JPEGXRMetadata::new(float_rgb_layout(), Some(metrics));

        assert_eq!(metadata.max_cll_scrgb(), Some(5.0));
        assert_eq!(metadata.max_cll_channel(), Some(JPEGXRColorChannel::Blue));
        assert_eq!(
            metadata.max_cll_channel().map(JPEGXRColorChannel::symbol),
            Some('B')
        );
    }

    #[test]
    fn max_cll_preserves_float32_levels_above_binary16_range() {
        let source = float_rgb_source(&[[70_000.0, 0.0, 100_000.0]]);
        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            float_rgb_layout(),
            MaxCllMode::Percentile99_99,
            false,
        )
        .unwrap();
        let metadata = JPEGXRMetadata::new(float_rgb_layout(), Some(metrics));

        assert_eq!(metadata.max_cll_scrgb(), Some(100_000.0));
        assert_eq!(metadata.max_cll_nits(), Some(8_000_000.0));
        assert_eq!(metadata.max_cll_channel(), Some(JPEGXRColorChannel::Blue));
    }

    #[test]
    fn normalization_quantizes_alpha_to_eight_bits() {
        let layout = float_rgba_layout();
        let source: Vec<u8> = std::iter::repeat_n([0.01_f32, 0.01, 0.01, 0.5], 16)
            .flatten()
            .flat_map(f32::to_ne_bytes)
            .collect();

        let rgba = normalize(&source, 4, 4, 64, layout, DecodeOptions::default())
            .unwrap()
            .rgba;

        assert!(rgba.iter().skip(3).step_by(4).all(|alpha| *alpha == 128));
    }

    #[test]
    fn hdr_normalization_batches_across_rows_and_preserves_its_tail() {
        const WIDTH: usize = 17;
        const HEIGHT: usize = HDR_BATCH_PIXELS / WIDTH + 1;

        let extended_colors: Vec<_> = (0..WIDTH * HEIGHT)
            .map(|index| match index % 4 {
                0 => [4.0, 2.0, 1.0],
                1 => [0.18, 0.5, 1.0],
                2 => [0.0, 0.25, 2.0],
                _ => [1.0, 3.0, 0.75],
            })
            .collect();
        let extended = ExtendedReinhard::new(WhitePoint::new(4.0).unwrap());
        assert_hdr_normalization_matches_scalar(&extended_colors, WIDTH, &extended);

        let clamp_colors: Vec<_> = (0..WIDTH * HEIGHT)
            .map(|index| match index % 3 {
                0 => [1.0, 0.5, 0.25],
                1 => [0.18, 0.0, 0.75],
                _ => [0.01, 0.02, 0.03],
            })
            .collect();
        assert_hdr_normalization_matches_scalar(&clamp_colors, WIDTH, &Clamp);
    }

    #[test]
    fn hdr_batch_preserves_alpha_alignment_across_a_row_and_batch_boundary() {
        const WIDTH: usize = 17;
        const HEIGHT: usize = HDR_BATCH_PIXELS / WIDTH + 1;

        let pixels: Vec<_> = (0..WIDTH * HEIGHT)
            .map(|index| {
                let color = match index % 3 {
                    0 => [1.0, 0.5, 0.25],
                    1 => [0.18, 0.0, 0.75],
                    _ => [0.01, 0.02, 0.03],
                };
                let alpha = match index % 5 {
                    0 => 0.0,
                    1 => f32::from_bits(1),
                    2 => 0.25,
                    3 => 0.5,
                    _ => 1.0,
                };
                [color[0], color[1], color[2], alpha]
            })
            .collect();
        let layout = float_rgba_layout();
        let source = float_rgba_source(&pixels);
        let row_stride = WIDTH * layout.bytes_per_pixel;
        let mut actual = Vec::with_capacity(WIDTH * HEIGHT * 4);

        let has_nonzero_alpha =
            append_hdr_pixels(&source, WIDTH, row_stride, layout, &Clamp, &mut actual).unwrap();
        let expected: Vec<_> = pixels
            .iter()
            .copied()
            .flat_map(|[red, green, blue, alpha]| {
                let [red, green, blue] = hdr_to_srgb8([red, green, blue], &Clamp);
                [red, green, blue, normalized_to_u8(alpha)]
            })
            .collect();

        assert!(has_nonzero_alpha);
        assert_eq!(actual, expected);
    }

    #[test]
    fn positive_hdr_alpha_below_one_byte_step_does_not_become_opaque() {
        let layout = float_rgba_layout();
        let source =
            float_rgba_source(&[[0.5, 0.5, 0.5, f32::from_bits(1)], [0.25, 0.25, 0.25, 0.0]]);

        let rgba = normalize(&source, 2, 1, 32, layout, DecodeOptions::default())
            .unwrap()
            .rgba;

        assert!(rgba.iter().skip(3).step_by(4).all(|alpha| *alpha == 0));
    }

    #[test]
    fn hidden_transparent_rgb_does_not_affect_hdr_metrics() {
        let layout = float_rgba_layout();
        let source = float_rgba_source(&[[1.0, 1.0, 1.0, 1.0], [100.0, -100.0, 0.0, 0.0]]);

        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            layout,
            MaxCllMode::TrueMaximum,
            false,
        )
        .unwrap();

        assert_eq!(metrics.max_cll.nits(), SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.max_cll_mode, MaxCllMode::TrueMaximum);
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

        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            layout,
            MaxCllMode::Percentile99_99,
            false,
        )
        .unwrap();
        let rgba = normalize(&source, 2, 1, 32, layout, DecodeOptions::default())
            .unwrap()
            .rgba;

        assert_eq!(metrics.max_cll.nits(), 4.0 * SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.max_cll.channel, JPEGXRColorChannel::Blue);
        assert!(
            rgba.iter()
                .skip(3)
                .step_by(4)
                .all(|alpha| *alpha == u8::MAX)
        );
    }
}
