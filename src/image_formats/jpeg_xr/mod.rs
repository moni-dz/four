//! Decodes bounded JPEG XR images and tone-maps HDR pixels to SDR RGBA8.
//!
//! Metadata and white-point methods use a 99.99th-percentile `max(R, G, B)` light level by default;
//! [`DecodeOptions`] can select the true maximum. White-point methods may clip values above the
//! selected threshold. Other [`ToneMappingMethod`] variants use their own statistics.
//! An empty, non-premultiplied HDR alpha plane is treated as unspecified and made opaque.
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
mod hdr;
mod normalize;
mod pixel;

#[cfg(test)]
use tonemapping::{Clamp, LinearRGB, LinearRGBPlanes, LuminanceWhitePoint, ToneMapper, WhitePoint};
use tonemapping::{MaxCLLMode, ToneMappingMethod};

#[cfg(test)]
use hdr::{
    AnalysisRequest, AnalysisTotals, HDRPixelSelection, MaxCLL, finish_max_cll, hdr_white_point,
};
use hdr::{AnalysisScope, HDRAnalysis, HDRMetrics};
use normalize::write_normalized_pixels;
#[cfg(test)]
use normalize::{
    append_hdr_pixels, display_linear_to_srgb8, hdr_to_srgb8, normalized_to_u8,
    write_bgr101010_hdr_pixels, write_display_pixels, write_hdr_pixels_scalar,
};
use pixel::PixelLayout;
#[cfg(test)]
use pixel::{SampleEncoding, half_to_f32, pq_to_linear, pq_to_linear_simd, unpack_bgr101010};

use super::{
    DIMENSION_MAX, DecodedImage, Dimensions, PARALLEL_PIXELS_MIN, PARALLEL_PIXELS_PER_JOB,
    PIXELS_MAX, map_dimensions_error, round_clamp_u8,
};
use error::error;

pub use error::{Error, JPEGXRError, JPEGXRLimit, Result};

/// The four-byte signature at the beginning of a JPEG XR file.
pub const SIGNATURE: [u8; 4] = [0x49, 0x49, 0xbc, 0x01];

const SC_RGB_REFERENCE_WHITE_NITS: f32 = 80.0;

/// `BT2446A` uses the report's fixed convention: `1.0` is 100 cd/m^2. Other operators are
/// white-point relative. scRGB (`1.0` = 80 cd/m^2) is rescaled before BT.2446 conversion.
const BT2446_INPUT_SCALE: f32 = SC_RGB_REFERENCE_WHITE_NITS / 100.0;

const SOURCE_BUFFER_MAX: usize = 512 * 1024 * 1024;
// Three f32 channels plus staged alpha occupy roughly 13 KiB, leaving room in common L1 caches.
const HDR_BATCH_PIXELS: usize = 1_024;
#[cfg(test)]
const COLOR_LANES: usize = 4;
const SRGB_LANES: usize = 8;

#[cfg(test)]
type F32x4 = std::simd::Simd<f32, COLOR_LANES>;
type F32x8 = std::simd::Simd<f32, SRGB_LANES>;

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
    /// Disabling metrics skips analysis unless the selected tone mapper needs an image-derived
    /// white point. It affects only functions that return metadata.
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
    /// The estimate follows the [`MaxCLLMode`] used for decoding. SDR sources and disabled metrics
    /// return `None`. Finite results above the `f32` range saturate at `f32::MAX`.
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
    /// Pixels outside Display-P3 count toward neither percentage; the total may be below 100%.
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
pub fn decode_with_metadata(bytes: &[u8]) -> Result<DecodedJPEGXR> {
    decode_with_metadata_and_options(bytes, DecodeOptions::default())
}

/// Decodes JPEG XR pixels and metadata using the selected HDR normalization `options`.
///
/// Image-dependent parameters come from the decoded source. Component-wise white-point methods use
/// [`MaxCLLMode`]; extended luminance Reinhard uses p99.99 Rec. 709 luminance. SDR sources bypass
/// tone mapping. Image-derived white points are floored at display white.
///
/// # Errors
///
/// Returns [`JPEGXRError`] for malformed input, resource-limit failures, and unsupported pixel
/// representations.
pub fn decode_with_metadata_and_options(
    bytes: &[u8],
    options: DecodeOptions,
) -> Result<DecodedJPEGXR> {
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
    let layout = match pixel_format {
        ::jpegxr::PixelFormat::BGR101010 => PixelLayout::bgr101010(),
        ::jpegxr::PixelFormat::RGBA128_FLOAT => PixelLayout::rgba128_float(),
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
        ::jpegxr::PixelFormat::BGR101010 => {
            let native_image = decoder
                .decode_bgr101010()
                .map_err(|source| codec_error(&source))?;
            let source = zerocopy::IntoBytes::as_bytes(native_image.pixels());

            invariant_eq!(source.len(), source_len);
            normalize(source, width, height, row_stride, layout, options)?
        }
        ::jpegxr::PixelFormat::RGBA128_FLOAT => {
            let native_image = decoder
                .decode_rgba_f32()
                .map_err(|source| codec_error(&source))?;
            let source = zerocopy::IntoBytes::as_bytes(native_image.pixels());

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

#[derive(Debug)]
struct NormalizedImage {
    rgba: Vec<u8>,
    hdr_metrics: Option<HDRMetrics>,
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

    // `write_normalized_pixels` writes every one of `output_len` bytes before returning `Ok`
    // (each of its pixel-format paths advances a running offset from `0` to `rgba.len()` in
    // contiguous, non-overlapping spans; `invariant_eq!` checks that at the end of the HDR
    // paths). On an early `Err`, `rgba` is dropped unread by the `?` below. So skip zeroing a
    // buffer this immediately overwrites in full: it can be multiple megabytes per decode.
    #[expect(
        unsafe_code,
        reason = "avoids zeroing an output buffer this function immediately overwrites in full"
    )]
    // SAFETY: see the coverage argument above.
    let mut rgba: Vec<u8> = unsafe { super::uninit_vec(output_len) };

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
    fn packed_hdr_batches_match_scalar_across_row_boundaries() {
        const WIDTH: usize = 13;
        const HEIGHT: usize = HDR_BATCH_PIXELS / WIDTH + 2;

        let source: Vec<u8> = (0..WIDTH * HEIGHT)
            .flat_map(|index| {
                let red = u32::try_from(index * 17 % 1_024).expect("test sample fits u32");
                let green = u32::try_from(index * 31 % 1_024).expect("test sample fits u32");
                let blue = u32::try_from(index * 47 % 1_024).expect("test sample fits u32");
                ((red << 20) | (green << 10) | blue).to_ne_bytes()
            })
            .collect();
        let mut scalar = vec![0_u8; WIDTH * HEIGHT * 4];
        let mut batched = vec![0_u8; scalar.len()];

        write_hdr_pixels_scalar(
            &source,
            WIDTH,
            WIDTH * 4,
            PixelLayout::bgr101010(),
            &BT2446A,
            BT2446_INPUT_SCALE,
            &mut scalar,
        )
        .unwrap();
        write_bgr101010_hdr_pixels(
            &source,
            WIDTH,
            WIDTH * 4,
            &BT2446A,
            BT2446_INPUT_SCALE,
            &mut batched,
        );

        assert_eq!(batched, scalar);
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
        let fitting = MaxCLL {
            relative_light_level: 1.0,
            channel: JPEGXRColorChannel::Red,
        };

        let hdr = MaxCLL {
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
