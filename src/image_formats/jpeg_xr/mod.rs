//! Decodes bounded JPEG XR images and tone-maps HDR pixels to SDR RGBA8.
//!
//! When metadata or a selected white-point method requires it, `MaxCLL` is estimated from the
//! 99.99th-percentile `max(R, G, B)` light level. This excludes isolated outliers from the
//! white-point statistic; only white-point methods may clip values above that threshold.
//! [`DecodeOptions`] can instead select the true maximum. Callers can also select any built-in
//! [`ToneMappingMethod`]; white-point methods use matching statistics estimated from the decoded
//! image.
//! A wholly empty, non-premultiplied HDR alpha plane is treated as unspecified and made opaque,
//! matching JPEG XR screenshots that store zero in an otherwise unused alpha channel.
//!
//! # References
//!
//! - [WIC native pixel formats](https://learn.microsoft.com/en-us/windows/win32/wic/-wic-codec-native-pixel-formats)
//!   defines packed-channel layout, numerical encoding, and color-space inference.
//! - [Windows GDK Screen Capture](https://learn.microsoft.com/en-us/gaming/gdk/docs/tools/tools-pc/commandlinetools/gr-wdcapture)
//!   documents `.jxr` as the Windows HDR screenshot format.
//! - [ITU-R BT.2100-3](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en) defines the PQ transfer
//!   function and HDR wide-color-gamut system used by the screenshot compatibility policy.
//! - [ITU-R BT.2446-1](https://www.itu.int/pub/R-REP-BT.2446-1-2021) defines the default HDR-to-SDR
//!   conversion.
//! - [Smith and Zink](https://doi.org/10.5594/JMI.2021.3090176) propose the per-frame p99.99
//!   `max(R, G, B)` outlier-rejection step used for still-image `MaxCLL` estimation.

mod error;

use std::num::NonZeroUsize;
use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};

use ::jpegxr::{Decoder as JXRDecoder, PixelFormat as CodecPixelFormat};
use multiversion::multiversion;
use rayon::prelude::*;
use tonemapping::{
    Clamp, ColorChannel as ToneColorChannel, LinearRGB, LinearRGBPlanes, LuminanceWhitePoint,
    LuminanceWhitePointEstimator, MaxCLLEstimator, MaxCLLMode, ToneMapper, ToneMappingMethod,
    WhitePoint, exp2, log2,
};
use zerocopy::{FromBytes, IntoBytes};

use super::{
    DIMENSION_MAX, DecodedImage, Dimensions, DimensionsError, PARALLEL_PIXELS_MIN,
    PARALLEL_PIXELS_PER_JOB, PIXELS_MAX,
};
use error::error;

pub use error::{Error, JPEGXRError, JPEGXRLimit, Result};

/// The four-byte signature at the beginning of a JPEG XR file.
pub const SIGNATURE: [u8; 4] = [0x49, 0x49, 0xbc, 0x01];

const SC_RGB_REFERENCE_WHITE_NITS: f32 = 80.0;

/// `BT2446A` is calibrated against Report ITU-R BT.2446-1's own fixed convention, where an input
/// component of `1.0` is 100 cd/m^2 (the SDR target peak). Every other tone mapper here is
/// white-point relative and works in whatever unit the caller's linear light happens to use, but
/// BT2446 hardcodes real nits, so scRGB (`1.0` == 80 cd/m^2) must be rescaled before it reaches it.
const BT2446_INPUT_SCALE: f32 = SC_RGB_REFERENCE_WHITE_NITS / 100.0;

const SOURCE_BUFFER_MAX: usize = 512 * 1024 * 1024;
// Three f32 channels plus staged alpha occupy roughly 13 KiB, leaving room in common L1 caches.
const HDR_BATCH_PIXELS: usize = 1_024;
const COLOR_LANES: usize = 4;
const SRGB_LANES: usize = 8;

type F32x4 = Simd<f32, COLOR_LANES>;
type F32x8 = Simd<f32, SRGB_LANES>;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeOptions {
    tone_mapping: ToneMappingMethod,
    max_cll_mode: MaxCLLMode,
    include_hdr_metrics: bool,
}

impl DecodeOptions {
    /// Creates options using `tone_mapping` and `max_cll_mode`.
    #[must_use]
    pub const fn new(tone_mapping: ToneMappingMethod, max_cll_mode: MaxCLLMode) -> Self {
        Self {
            tone_mapping,
            max_cll_mode,
            include_hdr_metrics: true,
        }
    }

    /// Returns the selected HDR tone-mapping operator.
    #[must_use]
    pub const fn tone_mapping(self) -> ToneMappingMethod {
        self.tone_mapping
    }

    /// Returns the selected `MaxCLL` estimator mode.
    #[must_use]
    pub const fn max_cll_mode(self) -> MaxCLLMode {
        self.max_cll_mode
    }

    /// Selects whether returned metadata includes image-wide HDR metrics.
    ///
    /// Disabling metrics avoids their analysis pass unless the selected tone mapper needs an
    /// image-derived white point. This setting only affects functions that return metadata.
    #[must_use]
    pub const fn with_hdr_metrics(self, include_hdr_metrics: bool) -> Self {
        Self {
            include_hdr_metrics,
            ..self
        }
    }

    const fn includes_hdr_metrics(self) -> bool {
        self.include_hdr_metrics
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self::new(ToneMappingMethod::default(), MaxCLLMode::default())
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
    is_bgr: bool,
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

    /// Returns whether source pixels use a BGR-family WIC format.
    #[must_use]
    pub const fn is_bgr(self) -> bool {
        self.is_bgr
    }

    /// Returns whether the source uses the HDR processing path.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        self.is_hdr
    }

    /// Returns the estimated maximum content light level in nits for HDR sources.
    ///
    /// The estimate follows the [`MaxCLLMode`] used for decoding. SDR sources and decodes that
    /// disable HDR metrics return `None`. A finite result above the `f32` range saturates at
    /// `f32::MAX`.
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

    /// Returns the mode used to estimate the reported `MaxCLL` metric.
    #[must_use]
    pub const fn max_cll_mode(self) -> Option<MaxCLLMode> {
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
pub fn has_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&SIGNATURE)
}

/// Decodes a JPEG XR image and normalizes it to SDR RGBA8.
///
/// Unsigned integer RGB and grayscale inputs retain their sRGB encoding.
/// `PixelFormat32bppRGB101010` is unconditionally treated as BT.2100 PQ and Rec. 2020 HDR screenshot
/// data and converted to linear scRGB. Other HDR inputs are interpreted as linear scRGB. HDR values
/// are mapped with ITU-R BT.2446 Method A before conversion to sRGB; this default path does not
/// estimate an image white point. Premultiplied inputs are returned with straight alpha.
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
    Ok(decode_with_metadata_and_options(bytes, options.with_hdr_metrics(false))?.into_image())
}

/// Decodes JPEG XR pixels together with their source representation metadata.
///
/// This performs the same bounded decode and HDR-to-SDR normalization as [`decode`]. For HDR
/// sources, the returned metadata includes percentile `MaxCLL`; the default BT.2446 mapper does not
/// use that image-derived metric.
///
/// # Errors
///
/// Returns [`JPEGXRError`] when the input is malformed, exceeds a resource bound, or uses a pixel
/// representation that cannot be normalized to RGB.
pub fn decode_with_metadata(bytes: impl AsRef<[u8]>) -> Result<DecodedJPEGXR> {
    decode_with_metadata_and_options(bytes, DecodeOptions::default())
}

/// Decodes JPEG XR pixels and metadata using the selected HDR normalization `options`.
///
/// Image-dependent parameters are derived from the decoded source. Component-wise white-point
/// methods use the selected [`MaxCLLMode`], while extended luminance Reinhard always uses p99.99
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

    let decoder = JXRDecoder::new(bytes).map_err(|source| codec_error(&source))?;
    let width = decoder.info().width();
    let height = decoder.info().height();
    let (width, height) = validate_dimensions(
        i32::try_from(width).map_err(|_error| dimension_overflow_error(width))?,
        i32::try_from(height).map_err(|_error| dimension_overflow_error(height))?,
    )?;

    let pixel_format = decoder.info().pixel_format();
    let layout = match pixel_format {
        CodecPixelFormat::BGR101010 => PixelLayout::bgr101010(),
        CodecPixelFormat::RGBA128_FLOAT => PixelLayout::rgba128_float(),
        _ => {
            return Err(error(JPEGXRError::Unsupported(
                pixel_format.name().to_owned(),
            )));
        }
    };

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
            JPEGXRLimit::SourceBufferBytes {
                actual: Some(source_len),
                max: SOURCE_BUFFER_MAX,
            },
        )));
    }

    let normalized = match pixel_format {
        CodecPixelFormat::BGR101010 => {
            let native_image = decoder
                .decode_bgr101010()
                .map_err(|source| codec_error(&source))?;
            let source = native_image.pixels().as_bytes();

            invariant_eq!(source.len(), source_len);
            normalize(source, width, height, row_stride, layout, options)?
        }
        CodecPixelFormat::RGBA128_FLOAT => {
            let native_image = decoder
                .decode_rgba_f32()
                .map_err(|source| codec_error(&source))?;
            let source = native_image.pixels().as_bytes();

            invariant_eq!(source.len(), source_len);
            normalize(source, width, height, row_stride, layout, options)?
        }
        _ => unreachable!("pixel format validated when selecting its layout"),
    };
    let metadata = JPEGXRMetadata::new(layout, normalized.hdr_metrics);

    Ok(DecodedJPEGXR {
        image: DecodedImage::new(width, height, normalized.rgba),
        metadata,
    })
}

#[expect(
    dead_code,
    reason = "normalization primitives remain available for future decoder profiles"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleEncoding {
    Fixed16,
    Fixed32,
    Float16,
    Float32,
    PackedBGR101010,
    RGBE,
    Unsigned8,
    Unsigned16,
}

impl SampleEncoding {
    const fn bytes(self) -> usize {
        match self {
            Self::Unsigned8 | Self::RGBE => 1,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 2,
            Self::Fixed32 | Self::Float32 | Self::PackedBGR101010 => 4,
        }
    }

    const fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Fixed16
                | Self::Fixed32
                | Self::Float16
                | Self::Float32
                | Self::PackedBGR101010
                | Self::RGBE
        )
    }

    const fn bits_per_channel(self) -> u8 {
        match self {
            Self::Unsigned8 | Self::RGBE => 8,
            Self::PackedBGR101010 => 10,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 16,
            Self::Fixed32 | Self::Float32 => 32,
        }
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields describe independent WIC pixel-layout properties"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PixelLayout {
    encoding: SampleEncoding,
    color_channels: usize,
    source_channels: usize,
    bytes_per_pixel: usize,
    has_alpha: bool,
    premultiplied_alpha: bool,
    blue_first: bool,
    source_is_bgr: bool,
}

impl JPEGXRMetadata {
    fn new(layout: PixelLayout, hdr_metrics: Option<HDRMetrics>) -> Self {
        invariant!(layout.encoding.is_hdr() || hdr_metrics.is_none());

        Self {
            bits_per_channel: layout.encoding.bits_per_channel(),
            color_channels: u8::try_from(layout.color_channels)
                .expect("a JPEG XR color-channel count fits u8"),
            has_alpha: layout.has_alpha,
            is_bgr: layout.source_is_bgr,
            is_hdr: layout.encoding.is_hdr(),
            hdr_metrics,
        }
    }
}

impl PixelLayout {
    const fn bgr101010() -> Self {
        Self {
            encoding: SampleEncoding::PackedBGR101010,
            color_channels: 3,
            source_channels: 3,
            bytes_per_pixel: 4,
            has_alpha: false,
            premultiplied_alpha: false,
            blue_first: false,
            source_is_bgr: true,
        }
    }

    const fn rgba128_float() -> Self {
        Self {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 4,
            bytes_per_pixel: 16,
            has_alpha: true,
            premultiplied_alpha: false,
            blue_first: false,
            source_is_bgr: false,
        }
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

        if self.encoding == SampleEncoding::PackedBGR101010 {
            return Ok((decode_bgr101010(pixel), 1.0));
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

    let method = options.tone_mapping();

    let needs_hdr_analysis = options.includes_hdr_metrics()
        || method.uses_white_point()
        || method.uses_luminance_white_point();

    let analysis = if layout.encoding.is_hdr() && needs_hdr_analysis {
        Some(HDRAnalysis::estimate(
            source,
            row_stride,
            layout,
            options.max_cll_mode(),
            AnalysisScope {
                estimate_max_cll: options.includes_hdr_metrics() || method.uses_white_point(),
                estimate_luminance_white_point: method.uses_luminance_white_point(),
                collect_hdr_metrics: options.includes_hdr_metrics(),
            },
        )?)
    } else {
        None
    };

    let hdr_metrics = analysis.and_then(|analysis| analysis.hdr_metrics);
    let mut rgba = vec![0; output_len];

    let has_nonzero_alpha = write_normalized_pixels(
        source, width, row_stride, layout, method, analysis, &mut rgba,
    )?;

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

/// Scales scRGB into the fixed unit each tone mapper expects; only `BT2446` needs rescaling.
fn hdr_color_scale(method: ToneMappingMethod) -> f32 {
    if method == ToneMappingMethod::BT2446 {
        BT2446_INPUT_SCALE
    } else {
        1.0
    }
}

fn write_normalized_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    method: ToneMappingMethod,
    analysis: Option<HDRAnalysis>,
    rgba: &mut [u8],
) -> Result<bool> {
    if !layout.encoding.is_hdr() {
        return write_pixel_slabs(source, width, row_stride, rgba, |source, rgba| {
            write_sdr_pixels(source, width, row_stride, layout, rgba)
        });
    }

    if method == ToneMappingMethod::Clamp {
        return write_pixel_slabs(source, width, row_stride, rgba, |source, rgba| {
            write_hdr_pixels_scalar(source, width, row_stride, layout, &Clamp, 1.0, rgba)
        });
    }

    let white_point = analysis
        .and_then(|analysis| analysis.max_cll)
        .map_or_else(display_white_point, hdr_white_point);
    let luminance_white_point = analysis
        .and_then(|analysis| analysis.luminance_white_point)
        .map_or_else(display_luminance_white_point, hdr_luminance_white_point);
    let mapper = method.resolve(white_point, luminance_white_point);
    let color_scale = hdr_color_scale(method);

    write_pixel_slabs(source, width, row_stride, rgba, |source, rgba| {
        write_hdr_pixels(
            source,
            width,
            row_stride,
            layout,
            &mapper,
            color_scale,
            rgba,
        )
    })
}

fn write_pixel_slabs(
    source: &[u8],
    width: usize,
    row_stride: usize,
    rgba: &mut [u8],
    writer: impl Fn(&[u8], &mut [u8]) -> Result<bool> + Sync,
) -> Result<bool> {
    let row_count = source.len() / row_stride;

    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");

    if pixel_count < PARALLEL_PIXELS_MIN {
        return writer(source, rgba);
    }

    let rows_per_job = PARALLEL_PIXELS_PER_JOB.div_ceil(width);
    let source_bytes_per_job = rows_per_job * row_stride;
    let rgba_bytes_per_job = rows_per_job * width * 4;

    source
        .par_chunks(source_bytes_per_job)
        .zip(rgba.par_chunks_mut(rgba_bytes_per_job))
        .map(|(source, rgba)| writer(source, rgba))
        .try_reduce(|| false, |left, right| Ok::<bool, Error>(left || right))
}

fn write_sdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    rgba: &mut [u8],
) -> Result<bool> {
    let mut has_nonzero_alpha = false;
    for (row, target_row) in source
        .chunks_exact(row_stride)
        .zip(rgba.chunks_exact_mut(width * 4))
    {
        let (targets, remainder) = target_row.as_chunks_mut::<4>();

        invariant!(remainder.is_empty());

        for (x, target) in targets.iter_mut().enumerate() {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;

            has_nonzero_alpha |= alpha > 0.0;

            let color = color.map(normalized_to_u8);
            target.copy_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }

    Ok(has_nonzero_alpha)
}

fn write_hdr_pixels_scalar(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &(impl ToneMapper + Sync),
    color_scale: f32,
    rgba: &mut [u8],
) -> Result<bool> {
    let mut has_nonzero_alpha = false;
    for (row, target_row) in source
        .chunks_exact(row_stride)
        .zip(rgba.chunks_exact_mut(width * 4))
    {
        let (targets, remainder) = target_row.as_chunks_mut::<4>();

        invariant!(remainder.is_empty());

        for (x, target) in targets.iter_mut().enumerate() {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;

            has_nonzero_alpha |= alpha > 0.0;

            let color = color.map(|component| component * color_scale);
            let color = display_linear_to_srgb8(mapper.map(LinearRGB::new(color)));
            target.copy_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }

    Ok(has_nonzero_alpha)
}

fn write_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &(impl ToneMapper + Sync),
    color_scale: f32,
    rgba: &mut [u8],
) -> Result<bool> {
    let row_count = source.len() / row_stride;

    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");

    let batch_capacity = HDR_BATCH_PIXELS.min(pixel_count);
    let mut colors = LinearRGBPlanes::with_capacity(batch_capacity);
    let mut alphas = Vec::with_capacity(batch_capacity);
    let mut has_nonzero_alpha = false;
    let mut rgba_offset = 0;

    for row in source.chunks_exact(row_stride) {
        for x in 0..width {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;

            has_nonzero_alpha |= alpha > 0.0;

            let color = color.map(|component| component * color_scale);
            colors.push(LinearRGB::new(color));
            alphas.push(normalized_to_u8(alpha));

            if colors.len() == HDR_BATCH_PIXELS {
                write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
            }
        }
    }

    write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
    invariant_eq!(rgba_offset, rgba.len());
    Ok(has_nonzero_alpha)
}

fn write_tone_mapped_batch(
    mapper: &(impl ToneMapper + Sync),
    colors: &mut LinearRGBPlanes,
    alphas: &mut Vec<u8>,
    rgba: &mut [u8],
    rgba_offset: &mut usize,
) {
    invariant_eq!(colors.len(), alphas.len());

    if colors.is_empty() {
        return;
    }

    mapper.map_planes_in_place(colors);

    let byte_count = colors.len() * 4;
    let target = &mut rgba[*rgba_offset..*rgba_offset + byte_count];
    let (targets, remainder) = target.as_chunks_mut::<4>();

    invariant!(remainder.is_empty());
    write_display_pixels(colors, alphas, targets);

    *rgba_offset += byte_count;
    colors.clear();
    alphas.clear();
}

#[multiversion(targets = "simd")]
fn write_display_pixels(colors: &LinearRGBPlanes, alphas: &[u8], targets: &mut [[u8; 4]]) {
    invariant_eq!(colors.len(), alphas.len());
    invariant_eq!(colors.len(), targets.len());

    let [red, green, blue] = colors.channels();
    let (red_chunks, red_tail) = red.as_chunks::<SRGB_LANES>();
    let (green_chunks, green_tail) = green.as_chunks::<SRGB_LANES>();
    let (blue_chunks, blue_tail) = blue.as_chunks::<SRGB_LANES>();
    let (alpha_chunks, alpha_tail) = alphas.as_chunks::<SRGB_LANES>();
    let (target_chunks, target_tail) = targets.as_chunks_mut::<SRGB_LANES>();

    for ((((red, green), blue), alphas), targets) in red_chunks
        .iter()
        .zip(green_chunks)
        .zip(blue_chunks)
        .zip(alpha_chunks)
        .zip(target_chunks)
    {
        // Quantize in the vector too. Only the transfer function used to be vectorized, leaving
        // twenty-four scalar clamp-scale-round-convert sequences per eight-pixel group. The
        // operations mirror `normalized_to_u8` exactly: `Simd::round` is also half-away-from-zero,
        // and a float-to-integer `cast` saturates the same way `as` does.
        let encoded = [*red, *green, *blue].map(|channel| {
            let srgb = linear_to_srgb_simd(F32x8::from_array(channel));
            (srgb.simd_clamp(F32x8::splat(0.0), F32x8::splat(1.0))
                * F32x8::splat(f32::from(u8::MAX)))
            .round()
            .cast::<u8>()
            .to_array()
        });

        for lane in 0..SRGB_LANES {
            targets[lane] = [
                encoded[0][lane],
                encoded[1][lane],
                encoded[2][lane],
                alphas[lane],
            ];
        }
    }

    for ((((red, green), blue), alpha), target) in red_tail
        .iter()
        .zip(green_tail)
        .zip(blue_tail)
        .zip(alpha_tail)
        .zip(target_tail)
    {
        let color = LinearRGB::new([*red, *green, *blue]);
        let [red, green, blue] = display_linear_to_srgb8(color);
        *target = [red, green, blue, *alpha];
    }
}

#[cfg(test)]
fn append_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &(impl ToneMapper + Sync),
    rgba: &mut Vec<u8>,
) -> Result<bool> {
    let byte_count = source.len() / row_stride * width * 4;
    let start = rgba.len();
    rgba.resize(start + byte_count, 0);
    write_hdr_pixels(
        source,
        width,
        row_stride,
        layout,
        mapper,
        1.0,
        &mut rgba[start..],
    )
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
    max_cll_mode: MaxCLLMode,
    luminance_white_point: Option<LuminanceWhitePoint>,
    max_luminance_nits: f32,
    average_luminance_nits: f32,
    min_luminance_nits: f32,
    rec709_percentage: f32,
    dci_p3_percentage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HDRAnalysis {
    max_cll: Option<MaxCll>,
    luminance_white_point: Option<LuminanceWhitePoint>,
    hdr_metrics: Option<HDRMetrics>,
}

#[derive(Clone, Copy, Debug)]
struct HDRPixelSelection {
    count: usize,
    exclude_fully_transparent: bool,
}

impl HDRPixelSelection {
    fn new(source: &[u8], row_stride: usize, layout: PixelLayout) -> Result<Self> {
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

        let count = if exclude_fully_transparent {
            visible_alpha_pixels
        } else {
            source_pixel_count
        };

        invariant!(count > 0);
        invariant!(count <= source_pixel_count);

        Ok(Self {
            count,
            exclude_fully_transparent,
        })
    }
}

impl HDRMetrics {
    #[cfg(test)]
    fn estimate(
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        max_cll_mode: MaxCLLMode,
        estimate_luminance_white_point: bool,
    ) -> Result<Self> {
        HDRAnalysis::estimate(
            source,
            row_stride,
            layout,
            max_cll_mode,
            AnalysisScope {
                estimate_max_cll: true,
                estimate_luminance_white_point,
                collect_hdr_metrics: true,
            },
        )
        .map(|analysis| {
            analysis
                .hdr_metrics
                .expect("full HDR analysis produces display metrics")
        })
    }
}

/// Which measurements an HDR analysis pass collects.
#[derive(Clone, Copy, Debug)]
struct AnalysisScope {
    estimate_max_cll: bool,
    estimate_luminance_white_point: bool,
    collect_hdr_metrics: bool,
}

/// The parameters that decide which measurements an analysis pass collects.
#[derive(Clone, Copy, Debug)]
struct AnalysisRequest {
    selection: HDRPixelSelection,
    pixel_count: usize,
    max_cll_mode: MaxCLLMode,
    estimate_max_cll: bool,
    estimate_luminance_white_point: bool,
    collect_hdr_metrics: bool,
}

/// The measurements one worker gathers from its share of the image.
///
/// Every field merges associatively, which is what lets the analysis pass run in parallel: each
/// worker builds its own totals over a row group, and the results fold together into the answer
/// the sequential pass would have produced.
#[derive(Debug)]
struct AnalysisTotals {
    accumulator: Option<HDRMetricAccumulator>,
    max_cll_estimator: Option<MaxCLLEstimator>,
    luminance_white_point_estimator: Option<LuminanceWhitePointEstimator>,
    max_cll_batch: Option<Vec<LinearRGB>>,
}

impl AnalysisTotals {
    fn new(request: &AnalysisRequest) -> Self {
        Self {
            accumulator: request.collect_hdr_metrics.then(HDRMetricAccumulator::new),
            // Every partial estimator declares the whole image, so the merged observation count
            // matches what `finish` requires and the retained sample counts agree.
            max_cll_estimator: request.estimate_max_cll.then(|| {
                MaxCLLEstimator::with_mode(
                    NonZeroUsize::new(request.pixel_count)
                        .expect("HDR analysis includes at least one pixel"),
                    request.max_cll_mode,
                )
            }),
            luminance_white_point_estimator: request.estimate_luminance_white_point.then(|| {
                LuminanceWhitePointEstimator::new(
                    NonZeroUsize::new(request.pixel_count)
                        .expect("HDR analysis includes at least one pixel"),
                )
            }),
            max_cll_batch: request
                .estimate_max_cll
                .then(|| Vec::with_capacity(HDR_BATCH_PIXELS.min(request.pixel_count))),
        }
    }

    fn observe_slab(
        &mut self,
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        request: &AnalysisRequest,
    ) -> Result<()> {
        visit_pixels(source, row_stride, layout, |color, alpha| {
            if request.selection.exclude_fully_transparent && alpha == 0.0 {
                return;
            }

            if let Some(accumulator) = &mut self.accumulator {
                accumulator.observe(color);
            }

            let color = LinearRGB::new(color);
            if let Some(estimator) = &mut self.luminance_white_point_estimator {
                estimator.observe(color);
            }

            if let Some(batch) = &mut self.max_cll_batch {
                batch.push(color);
                if batch.len() == HDR_BATCH_PIXELS {
                    self.max_cll_estimator
                        .as_mut()
                        .expect("a MaxCLL batch has an estimator")
                        .observe_many(batch);
                    batch.clear();
                }
            }
        })
    }

    fn merge(&mut self, other: Self) {
        if let (Some(accumulator), Some(other)) = (&mut self.accumulator, other.accumulator) {
            accumulator.merge(other);
        }

        // Drain the other worker's pending batch into its own estimator before merging, so no
        // observation is lost and the counts still add up.
        let mut other_max_cll = other.max_cll_estimator;
        if let (Some(estimator), Some(batch)) =
            (other_max_cll.as_mut(), other.max_cll_batch.as_ref())
        {
            estimator.observe_many(batch);
        }
        if let (Some(estimator), Some(other)) = (&mut self.max_cll_estimator, other_max_cll) {
            estimator.merge(other);
        }

        if let (Some(estimator), Some(other)) = (
            &mut self.luminance_white_point_estimator,
            other.luminance_white_point_estimator,
        ) {
            estimator.merge(other);
        }
    }
}

impl HDRAnalysis {
    fn estimate(
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        max_cll_mode: MaxCLLMode,
        scope: AnalysisScope,
    ) -> Result<Self> {
        invariant!(layout.encoding.is_hdr());
        invariant!(row_stride >= layout.bytes_per_pixel);
        invariant!(!scope.collect_hdr_metrics || scope.estimate_max_cll);

        let selection = HDRPixelSelection::new(source, row_stride, layout)?;
        let pixel_count = selection.count;

        let request = AnalysisRequest {
            selection,
            pixel_count,
            max_cll_mode,
            estimate_max_cll: scope.estimate_max_cll,
            estimate_luminance_white_point: scope.estimate_luminance_white_point,
            collect_hdr_metrics: scope.collect_hdr_metrics,
        };

        let totals = compute_totals(source, row_stride, layout, &request)?;

        let AnalysisTotals {
            accumulator,
            mut max_cll_estimator,
            luminance_white_point_estimator,
            max_cll_batch,
        } = totals;

        if let Some(batch) = &max_cll_batch {
            max_cll_estimator
                .as_mut()
                .expect("a MaxCLL batch has an estimator")
                .observe_many(batch);
        }

        if let Some(accumulator) = &accumulator {
            invariant_eq!(
                usize::try_from(accumulator.pixel_count)
                    .expect("the bounded JPEG XR pixel count fits usize"),
                pixel_count
            );
        }

        let max_cll = max_cll_estimator.map(finish_max_cll);
        let luminance_white_point = luminance_white_point_estimator.and_then(|estimator| {
            estimator
                .finish()
                .expect("HDR analysis observes exactly its declared pixel count")
        });
        let hdr_metrics = accumulator.map(|accumulator| {
            accumulator.finish(
                max_cll.expect("display HDR metrics include MaxCLL"),
                max_cll_mode,
                luminance_white_point,
            )
        });

        Ok(Self {
            max_cll,
            luminance_white_point,
            hdr_metrics,
        })
    }
}

/// Gathers analysis measurements over `source`, splitting the work by row group when the image is
/// large enough for the split to pay for itself.
///
/// This pass used to run on one thread while the write pass that follows it was already parallel,
/// which made it the largest serial block in an HDR decode. All three accumulators merge
/// associatively, so the image can be split by row groups.
///
/// Job size is computed in pixels, not row-stride bytes: `write_pixel_slabs` below uses the same
/// `width`-based formula, and BGR101010/RGBA32F have very different bytes-per-pixel, so sizing off
/// `row_stride` alone produced wildly different job counts per format.
fn compute_totals(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
    request: &AnalysisRequest,
) -> Result<AnalysisTotals> {
    let width = row_stride / layout.bytes_per_pixel;
    let rows_per_job = PARALLEL_PIXELS_PER_JOB.div_ceil(width);

    // `visit_pixels` below scans every source pixel regardless of transparency, so the
    // serial/parallel split must be sized off the pixels actually scanned, not
    // `request.pixel_count` (which, for images with transparency, counts only the visible pixels
    // the estimators retain). Otherwise a large mostly-transparent image runs its full scan
    // serially.
    let row_count = source.len() / row_stride;
    let scanned_pixel_count = width * row_count;

    if scanned_pixel_count < PARALLEL_PIXELS_MIN {
        let mut totals = AnalysisTotals::new(request);
        totals.observe_slab(source, row_stride, layout, request)?;
        return Ok(totals);
    }

    // `try_fold` reuses one `AnalysisTotals` (and its full-image-sized percentile heaps) across
    // every chunk rayon assigns to the same split, instead of allocating a fresh one per chunk:
    // a `map` here would size those heaps for the whole image on every one of the ~1000 chunks a
    // large image produces.
    source
        .par_chunks(rows_per_job * row_stride)
        .try_fold(
            || AnalysisTotals::new(request),
            |mut totals, slab| {
                totals.observe_slab(slab, row_stride, layout, request)?;
                Ok::<AnalysisTotals, Error>(totals)
            },
        )
        .try_reduce(
            || AnalysisTotals::new(request),
            |mut left, right| {
                left.merge(right);
                Ok::<AnalysisTotals, Error>(left)
            },
        )
}

fn finish_max_cll(estimator: MaxCLLEstimator) -> MaxCll {
    let estimate = estimator
        .finish()
        .expect("the HDR analysis pass visits the measured number of pixels");
    let relative_light_level = estimate.level();

    invariant!(relative_light_level.is_finite());
    invariant!(relative_light_level >= 0.0);

    MaxCll {
        relative_light_level,
        channel: jpeg_xr_color_channel(estimate.channel()),
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

fn display_white_point() -> WhitePoint {
    WhitePoint::new(1.0).expect("display white is a positive finite white point")
}

fn hdr_luminance_white_point(white_point: LuminanceWhitePoint) -> LuminanceWhitePoint {
    let luminance = white_point.luminance().max(1.0);
    LuminanceWhitePoint::new(luminance)
        .expect("p99.99 luminance floored at display white is a positive finite white point")
}

fn display_luminance_white_point() -> LuminanceWhitePoint {
    LuminanceWhitePoint::new(1.0).expect("display white is a positive finite luminance white point")
}

#[derive(Clone, Copy, Debug)]
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

        // Rec. 709 luma weights (ITU-R BT.709-6 section 3.2), applied to linear scRGB components
        // to get relative luminance, then scaled to absolute nits by the scRGB reference white.
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

    /// Folds `other` into these metrics.
    ///
    /// Counts add, extremes take the wider bound, and the luminance sum adds. Summation order
    /// changes with the number of workers, so the average luminance can move by a rounding step
    /// between runs on differently sized machines.
    fn merge(&mut self, other: Self) {
        self.pixel_count += other.pixel_count;
        self.luminance_sum_nits += other.luminance_sum_nits;
        self.max_luminance_nits = self.max_luminance_nits.max(other.max_luminance_nits);
        self.min_luminance_nits = self.min_luminance_nits.min(other.min_luminance_nits);
        self.rec709_pixels += other.rec709_pixels;
        self.dci_p3_pixels += other.dci_p3_pixels;
    }

    fn finish(
        self,
        max_cll: MaxCll,
        max_cll_mode: MaxCLLMode,
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

    // Converts Rec. 709 linear RGB (already established as out-of-gamut above) to Display-P3
    // linear RGB via the direct Rec. 709 -> Display-P3 primaries matrix (both D65 white points,
    // so no chromatic adaptation step is needed). Nonnegative components here mean the color is
    // representable in Display-P3 even though it fell outside Rec. 709.
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

#[expect(
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

#[expect(
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

/// Counts source pixels with nonzero alpha, splitting the work by row group when the image is
/// large enough for the split to pay for itself.
///
/// This scan runs before `compute_totals` on every RGBA image, so leaving it serial would reinstate
/// the same single-thread bottleneck `compute_totals` was parallelized to remove.
fn visible_alpha_pixel_count(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
) -> Result<usize> {
    invariant!(layout.has_alpha);

    let width = row_stride / layout.bytes_per_pixel;
    let row_count = source.len() / row_stride;
    let scanned_pixel_count = width * row_count;

    if scanned_pixel_count < PARALLEL_PIXELS_MIN {
        return visible_alpha_pixel_count_slab(source, row_stride, layout);
    }

    let rows_per_job = PARALLEL_PIXELS_PER_JOB.div_ceil(width);
    source
        .par_chunks(rows_per_job * row_stride)
        .try_fold(
            || 0_usize,
            |count, slab| {
                Ok::<usize, Error>(
                    count + visible_alpha_pixel_count_slab(slab, row_stride, layout)?,
                )
            },
        )
        .try_reduce(|| 0_usize, |left, right| Ok(left + right))
}

fn visible_alpha_pixel_count_slab(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
) -> Result<usize> {
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
        .map_err(|dimensions_error| match dimensions_error {
            DimensionsError::Zero => error(JPEGXRError::Output(
                "JPEG XR dimensions must both be nonzero",
            )),
            DimensionsError::TooLarge { width, height } => {
                error(JPEGXRError::LimitExceeded(JPEGXRLimit::Dimensions {
                    actual: Some(width.max(height)),
                    max: DIMENSION_MAX,
                }))
            }
            DimensionsError::TooManyPixels { pixels } => {
                error(JPEGXRError::LimitExceeded(JPEGXRLimit::Pixels {
                    actual: Some(pixels),
                    max: PIXELS_MAX,
                }))
            }
        })
}

fn decode_sample(bytes: &[u8], encoding: SampleEncoding) -> f32 {
    invariant_eq!(bytes.len(), encoding.bytes());

    match encoding {
        SampleEncoding::Unsigned8 => f32::from(read_sample::<u8>(bytes)) / f32::from(u8::MAX),
        SampleEncoding::Unsigned16 => f32::from(read_sample::<u16>(bytes)) / f32::from(u16::MAX),
        SampleEncoding::Fixed16 => f32::from(read_sample::<i16>(bytes)) / 8192.0,
        SampleEncoding::Fixed32 => fixed32_to_f32(read_sample::<i32>(bytes)),
        SampleEncoding::Float16 => half_to_f32(read_sample::<u16>(bytes)),
        SampleEncoding::Float32 => read_sample::<f32>(bytes),
        SampleEncoding::PackedBGR101010 => {
            unreachable!("packed BGR101010 pixels are decoded as a unit")
        }
        SampleEncoding::RGBE => {
            unreachable!("RGBE pixels are decoded as a unit")
        }
    }
}

fn decode_bgr101010(pixel: &[u8]) -> [f32; 3] {
    rec2100_pq_to_scrgb(unpack_bgr101010(pixel))
}

fn unpack_bgr101010(pixel: &[u8]) -> [f32; 3] {
    const MASK: u32 = 0x03ff;
    const SCALE: f32 = 1.0 / 1023.0;

    invariant_eq!(pixel.len(), SampleEncoding::PackedBGR101010.bytes());

    let packed = read_sample::<u32>(pixel);
    let channel = |shift| {
        let sample = u16::try_from((packed >> shift) & MASK)
            .expect("a masked 10-bit JPEG XR sample fits u16");
        f32::from(sample) * SCALE
    };

    [channel(20), channel(10), channel(0)]
}

fn rec2100_pq_to_scrgb(encoded: [f32; 3]) -> [f32; 3] {
    const REC2100_MAX_NITS: f32 = 10_000.0;
    const SCALE: f32 = REC2100_MAX_NITS / SC_RGB_REFERENCE_WHITE_NITS;

    let [red, green, blue, _padding] =
        pq_to_linear_simd(F32x4::from_array([encoded[0], encoded[1], encoded[2], 0.0])).to_array();

    [
        (1.660_491 * red - 0.587_641 * green - 0.072_850 * blue) * SCALE,
        (-0.124_550 * red + 1.132_9 * green - 0.008_349 * blue) * SCALE,
        (-0.018_151 * red - 0.100_579 * green + 1.118_73 * blue) * SCALE,
    ]
}

#[cfg(test)]
fn pq_to_linear(encoded: f32) -> f32 {
    const INVERSE_M1: f32 = 16_384.0 / 2_610.0;
    const INVERSE_M2: f32 = 32.0 / 2_523.0;
    const C1: f32 = 3_424.0 / 4_096.0;
    const C2: f32 = 2_413.0 / 128.0;
    const C3: f32 = 2_392.0 / 128.0;

    let powered = encoded.powf(INVERSE_M2);
    ((powered - C1).max(0.0) / (C2 - C3 * powered)).powf(INVERSE_M1)
}

fn pq_to_linear_simd(encoded: F32x4) -> F32x4 {
    const INVERSE_M1: f32 = 16_384.0 / 2_610.0;
    const INVERSE_M2: f32 = 32.0 / 2_523.0;
    const C1: f32 = 3_424.0 / 4_096.0;
    const C2: f32 = 2_413.0 / 128.0;
    const C3: f32 = 2_392.0 / 128.0;

    let powered = exp2(log2(encoded) * F32x4::splat(INVERSE_M2));
    let ratio = (powered - F32x4::splat(C1)).simd_max(F32x4::splat(0.0))
        / (F32x4::splat(C2) - F32x4::splat(C3) * powered);
    exp2(log2(ratio) * F32x4::splat(INVERSE_M1))
}

fn read_sample<T: FromBytes + Sized>(bytes: &[u8]) -> T {
    T::read_from_bytes(bytes).expect("JPEG XR sample length must match its encoding")
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "s7.24 fixed-point values are intentionally converted to f32 for tone mapping"
)]
fn fixed32_to_f32(value: i32) -> f32 {
    (f64::from(value) / 16_777_216.0) as f32
}

/// Widens an IEEE 754 binary16 sample to `f32`.
///
/// The hand-rolled decomposition this replaced called `powi` — a libm call — once per sample, and
/// twice on the subnormal path. Widening is a single instruction wherever `f16c` or NEON is
/// available, and a short branchless sequence where it is not.
fn half_to_f32(bits: u16) -> f32 {
    f32::from(f16::from_bits(bits))
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

fn linear_to_srgb_simd(value: F32x8) -> F32x8 {
    let value = value.simd_clamp(F32x8::splat(0.0), F32x8::splat(1.0));
    let linear = value * F32x8::splat(12.92);
    let nonlinear =
        exp2(log2(value) * F32x8::splat(1.0 / 2.4)) * F32x8::splat(1.055) - F32x8::splat(0.055);
    value
        .simd_le(F32x8::splat(0.003_130_8))
        .select(linear, nonlinear)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the normalized sample is rounded and clamped to u8 before conversion"
)]
fn normalized_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * f32::from(u8::MAX)).round() as u8
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
            max: SOURCE_BUFFER_MAX,
        }))
    } else {
        error(JPEGXRError::Codec(source.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonemapping::{ACESFitted, BT2446A, ExtendedLuminanceReinhard, ExtendedReinhard};

    #[test]
    fn display_pixel_batches_match_the_scalar_tail() {
        // `write_display_pixels` vectorizes complete lane groups and falls back to
        // `display_linear_to_srgb8` for the remainder. The two must agree exactly, or a pixel
        // would change appearance depending on its position within the batch. Lengths straddle
        // the lane boundary so both paths run on the same values.
        let sample = |index: usize| {
            let step = f32::from(u16::try_from(index).expect("test index fits u16"));
            [
                step * 0.013,
                (1.0 - step * 0.017).max(0.0),
                (step * 0.0007).min(1.5),
            ]
        };

        for length in [1, 7, 8, 9, 15, 16, 17, 33] {
            let colors: LinearRGBPlanes = (0..length).map(|i| LinearRGB::new(sample(i))).collect();
            let alphas: Vec<u8> = (0..length)
                .map(|i| u8::try_from(i % 256).expect("test alpha fits u8"))
                .collect();

            let mut batched = vec![[0_u8; 4]; length];
            write_display_pixels(&colors, &alphas, &mut batched);

            for (index, actual) in batched.into_iter().enumerate() {
                let [red, green, blue] = display_linear_to_srgb8(LinearRGB::new(sample(index)));
                assert_eq!(
                    actual,
                    [red, green, blue, alphas[index]],
                    "pixel {index} of a {length}-pixel batch"
                );
            }
        }
    }

    #[test]
    fn parallel_analysis_matches_the_sequential_pass() {
        // Above `PARALLEL_PIXELS_MIN` the analysis splits across worker threads and folds the
        // partial estimators back together. Below it, one thread does the whole image. The two
        // must produce the same measurements, and the same image is run at both sizes rather than
        // comparing against restated expectations.
        let colors: Vec<[f32; 3]> = (0..PARALLEL_PIXELS_MIN + 4_096)
            .map(|index| {
                let step = f32::from(u16::try_from(index % 4_099).expect("test step fits u16"));
                [
                    step * 0.011,
                    (400.0 - step * 0.017).max(0.0),
                    step * 0.000_7 + 0.25,
                ]
            })
            .collect();

        let row_stride = 12 * 512;
        let full = float_rgb_source(&colors);
        assert_eq!(full.len() % row_stride, 0);

        let parallel = HDRAnalysis::estimate(
            &full,
            row_stride,
            float_rgb_layout(),
            MaxCLLMode::Percentile99_99,
            AnalysisScope {
                estimate_max_cll: true,
                estimate_luminance_white_point: true,
                collect_hdr_metrics: true,
            },
        )
        .unwrap();

        // The same pixels, analysed one row group at a time and folded by hand, which is the
        // sequential path the parallel one has to agree with.
        let request = AnalysisRequest {
            selection: HDRPixelSelection::new(&full, row_stride, float_rgb_layout()).unwrap(),
            pixel_count: colors.len(),
            max_cll_mode: MaxCLLMode::Percentile99_99,
            estimate_max_cll: true,
            estimate_luminance_white_point: true,
            collect_hdr_metrics: true,
        };
        let mut sequential = AnalysisTotals::new(&request);
        sequential
            .observe_slab(&full, row_stride, float_rgb_layout(), &request)
            .unwrap();

        let mut max_cll_estimator = sequential.max_cll_estimator.unwrap();
        max_cll_estimator.observe_many(&sequential.max_cll_batch.unwrap());
        let expected_max_cll = finish_max_cll(max_cll_estimator);
        let expected_white_point = sequential
            .luminance_white_point_estimator
            .unwrap()
            .finish()
            .unwrap()
            .unwrap();

        assert_eq!(
            parallel.max_cll.unwrap().relative_light_level(),
            expected_max_cll.relative_light_level()
        );
        assert_eq!(parallel.max_cll.unwrap().channel, expected_max_cll.channel);
        assert_eq!(
            parallel.luminance_white_point.unwrap().luminance(),
            expected_white_point.luminance()
        );

        let metrics = parallel.hdr_metrics.unwrap();
        let expected = sequential.accumulator.unwrap().finish(
            expected_max_cll,
            MaxCLLMode::Percentile99_99,
            Some(expected_white_point),
        );

        assert_eq!(metrics.max_luminance_nits, expected.max_luminance_nits);
        assert_eq!(metrics.min_luminance_nits, expected.min_luminance_nits);
        assert_eq!(metrics.rec709_percentage, expected.rec709_percentage);
        assert_eq!(metrics.dci_p3_percentage, expected.dci_p3_percentage);

        // Summation order differs between the two, so the mean agrees to a rounding step rather
        // than to the bit.
        assert_approximately_equal(
            metrics.average_luminance_nits,
            expected.average_luminance_nits,
            expected.average_luminance_nits * 1.0e-5,
        );
    }

    #[test]
    fn binary16_widening_round_trips_every_bit_pattern() {
        // Exhaustive over the whole input domain, which is only 65,536 values. Round-tripping back
        // to binary16 is an independent property: it holds for the correct widening and fails for
        // any that misplaces the exponent bias, drops the implicit bit, or mishandles subnormals.
        for bits in 0..=u16::MAX {
            let widened = half_to_f32(bits);
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = bits & 0x03ff;

            if exponent == 0x1f && mantissa != 0 {
                assert!(widened.is_nan(), "{bits:#06x} should widen to NaN");
                continue;
            }

            assert_eq!(
                (widened as f16).to_bits(),
                bits,
                "{bits:#06x} did not survive the round trip"
            );

            // Independently reconstruct the value from the binary16 field definitions.
            let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
            let expected = if exponent == 0x1f {
                sign * f32::INFINITY
            } else if exponent == 0 {
                sign * f32::from(mantissa) * 2.0_f32.powi(-24)
            } else {
                sign * (1.0 + f32::from(mantissa) / 1024.0) * 2.0_f32.powi(i32::from(exponent) - 15)
            };
            assert_eq!(
                widened.to_bits(),
                expected.to_bits(),
                "{bits:#06x} widened to the wrong value"
            );
        }
    }

    #[test]
    fn binary16_widening_places_the_documented_values() {
        // Spot values taken from the IEEE 754 binary16 definition, not from any implementation.
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!(half_to_f32(0x8000).is_sign_negative());
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x3555), 0.333_251_95);
        // Largest finite binary16, and the smallest positive subnormal.
        assert_eq!(half_to_f32(0x7bff), 65_504.0);
        assert_eq!(half_to_f32(0x0001), 2.0_f32.powi(-24));
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    #[ignore = "requires JPEGXR_SAMPLE to name a local HDR image"]
    fn decodes_real_hdr_sample_with_pure_rust_codec() {
        let path = std::env::var("JPEGXR_SAMPLE").expect("set JPEGXR_SAMPLE");
        let bytes = std::fs::read(path).expect("read sample");
        let decoded = decode_with_metadata(bytes).expect("decode HDR sample");

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
        let decoded = decode_with_metadata(bytes).expect("decode BGR101010 HDR sample");

        assert_eq!(decoded.image().dimensions(), (3840, 2160));
        assert_eq!(decoded.image().rgba8().len(), 3840 * 2160 * 4);
        assert!(!decoded.metadata().has_alpha());
        assert!(decoded.metadata().is_hdr());
    }

    fn float_rgb_layout() -> PixelLayout {
        PixelLayout {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 3,
            bytes_per_pixel: 12,
            has_alpha: false,
            premultiplied_alpha: false,
            blue_first: false,
            source_is_bgr: false,
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
            source_is_bgr: false,
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
            DecodeOptions::new(method, MaxCLLMode::Percentile99_99),
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
        method: ToneMappingMethod,
        mapper: &impl ToneMapper,
    ) {
        let actual = normalize_float_rgb(colors, width, method);
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
    fn parallel_hdr_normalization_matches_scalar() {
        let colors = vec![[4.0, 2.0, 0.5]; PARALLEL_PIXELS_MIN];
        assert_hdr_normalization_matches_scalar(
            &colors,
            512,
            ToneMappingMethod::ACESFitted,
            &ACESFitted,
        );
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
    fn decodes_packed_bgr101010_as_rec2100_hdr() {
        let layout = PixelLayout {
            encoding: SampleEncoding::PackedBGR101010,
            color_channels: 3,
            source_channels: 3,
            bytes_per_pixel: 4,
            has_alpha: false,
            premultiplied_alpha: false,
            blue_first: false,
            source_is_bgr: true,
        };
        let packed = (0b11_u32 << 30) | 1023_u32;
        let encoded = unpack_bgr101010(&packed.to_ne_bytes());
        let (color, alpha) = layout.read_pixel(&packed.to_ne_bytes()).unwrap();
        let metadata = JPEGXRMetadata::new(layout, None);

        assert_eq!(layout.bytes_per_pixel, 4);
        assert_eq!(encoded, [0.0, 0.0, 1.0]);
        assert_approximately_equal(color[0], -9.106_25, 0.000_1);
        assert_approximately_equal(color[1], -1.043_625, 0.000_1);
        assert_approximately_equal(color[2], 139.841_25, 0.000_1);
        assert_eq!(alpha, 1.0);
        assert_eq!(metadata.bits_per_channel(), 10);
        assert!(metadata.is_bgr());
        assert!(metadata.is_hdr());
    }

    #[test]
    fn pq_eotf_maps_a_hundred_nits_to_one_percent_of_peak() {
        assert_approximately_equal(pq_to_linear(0.508_078_4), 0.01, 0.000_001);

        let encoded = [0.0, 0.1, 0.508_078_4, 1.0];
        let actual = pq_to_linear_simd(F32x4::from_array(encoded)).to_array();
        for (actual, encoded) in actual.into_iter().zip(encoded) {
            assert_approximately_equal(actual, pq_to_linear(encoded), 0.000_001);
        }
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
    fn hdr_metrics_can_be_deferred_without_changing_tone_mapping() {
        let colors = [[0.25, 1.0, 4.0], [2.0, 0.5, 0.125]];
        let layout = float_rgb_layout();
        let source = float_rgb_source(&colors);
        let row_stride = colors.len() * layout.bytes_per_pixel;

        for method in [
            ToneMappingMethod::BT2446,
            ToneMappingMethod::ExtendedReinhard,
            ToneMappingMethod::ExtendedLuminanceReinhard,
        ] {
            let options = DecodeOptions::new(method, MaxCLLMode::Percentile99_99);

            let with_metrics = normalize(&source, 2, 1, row_stride, layout, options)
                .expect("synthetic HDR pixels normalize with metrics");

            let without_metrics = normalize(
                &source,
                2,
                1,
                row_stride,
                layout,
                options.with_hdr_metrics(false),
            )
            .expect("synthetic HDR pixels normalize without metrics");

            assert!(with_metrics.hdr_metrics.is_some());
            assert!(without_metrics.hdr_metrics.is_none());
            assert_eq!(without_metrics.rgba, with_metrics.rgba);
        }
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
    fn bt2446_method_dispatches_to_bt2446a() {
        let color = [4.0, 2.0, 1.0];
        let actual = normalize_float_rgb(&[color], 1, ToneMappingMethod::BT2446);
        // The pipeline rescales scRGB into BT2446A's own fixed nits convention before mapping.
        let scaled = color.map(|component| component * BT2446_INPUT_SCALE);
        let [red, green, blue] = hdr_to_srgb8(scaled, &BT2446A);

        assert_eq!(actual, [red, green, blue, u8::MAX]);
    }

    #[test]
    fn decode_options_default_to_percentile_bt2446() {
        let options = DecodeOptions::default();

        assert_eq!(options.tone_mapping(), ToneMappingMethod::BT2446);
        assert_eq!(options.max_cll_mode(), MaxCLLMode::Percentile99_99);
    }

    #[test]
    fn extended_luminance_method_uses_the_luminance_white_point() {
        let color = [4.0, 2.0, 1.0];
        let source = float_rgb_source(&[color]);

        let metrics = HDRMetrics::estimate(
            &source,
            source.len(),
            float_rgb_layout(),
            MaxCLLMode::Percentile99_99,
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
            MaxCLLMode::Percentile99_99,
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
        assert_eq!(metadata.max_cll_mode(), Some(MaxCLLMode::Percentile99_99));
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
            DecodeOptions::new(ToneMappingMethod::ExtendedReinhard, MaxCLLMode::TrueMaximum),
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
            Some(MaxCLLMode::TrueMaximum)
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
            MaxCLLMode::Percentile99_99,
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
            MaxCLLMode::Percentile99_99,
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
            MaxCLLMode::Percentile99_99,
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
            MaxCLLMode::Percentile99_99,
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

        assert_hdr_normalization_matches_scalar(
            &extended_colors,
            WIDTH,
            ToneMappingMethod::ExtendedReinhard,
            &extended,
        );

        let clamp_colors: Vec<_> = (0..WIDTH * HEIGHT)
            .map(|index| match index % 3 {
                0 => [1.0, 0.5, 0.25],
                1 => [0.18, 0.0, 0.75],
                _ => [0.01, 0.02, 0.03],
            })
            .collect();

        assert_hdr_normalization_matches_scalar(
            &clamp_colors,
            WIDTH,
            ToneMappingMethod::Clamp,
            &Clamp,
        );
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
            MaxCLLMode::TrueMaximum,
            false,
        )
        .unwrap();

        assert_eq!(metrics.max_cll.nits(), SC_RGB_REFERENCE_WHITE_NITS);
        assert_eq!(metrics.max_cll_mode, MaxCLLMode::TrueMaximum);
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
            MaxCLLMode::Percentile99_99,
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
