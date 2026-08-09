//! Maps high-dynamic-range linear RGB colors into a displayable range.
//!
//! Tone-mapping operators accept [`LinearRgb`] values whose components are relative linear-light
//! levels. A component of `1.0` conventionally represents the target display's reference white.
//! Operators return finite components in the inclusive range `0.0..=1.0`; transfer encoding and
//! integer quantization remain the caller's responsibility.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt;
use std::num::NonZeroUsize;

const HDR_COMPONENT_MAX: f32 = 65_504.0;
const REC709_LUMINANCE: [f32; 3] = [0.212_6, 0.715_2, 0.072_2];

/// Stores a finite, nonnegative linear RGB color.
///
/// Construction replaces negative and `NaN` components with zero. Positive infinity and finite
/// values above the largest finite binary16 value are limited to `65_504`, which keeps subsequent
/// tone-mapping arithmetic bounded for common HDR interchange formats.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinearRgb([f32; 3]);

impl LinearRgb {
    /// Creates a sanitized linear RGB color from `components`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::LinearRgb;
    ///
    /// let color = LinearRgb::new([2.0, -1.0, f32::NAN]);
    /// assert_eq!(color.components(), [2.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    pub fn new(components: [f32; 3]) -> Self {
        Self(components.map(sanitize_component))
    }

    /// Returns the red, green, and blue components in that order.
    #[must_use]
    pub const fn components(self) -> [f32; 3] {
        self.0
    }

    /// Returns linear Rec. 709 luminance.
    #[must_use]
    pub fn luminance(self) -> f32 {
        REC709_LUMINANCE[0] * self.0[0]
            + REC709_LUMINANCE[1] * self.0[1]
            + REC709_LUMINANCE[2] * self.0[2]
    }

    fn displayable(components: [f32; 3]) -> Self {
        Self(components.map(|component| component.clamp(0.0, 1.0)))
    }

    fn max_component(self) -> PeakSample {
        let mut channel = ColorChannel::Red;
        if self.0[1] > self.0[channel.index()] {
            channel = ColorChannel::Green;
        }
        if self.0[2] > self.0[channel.index()] {
            channel = ColorChannel::Blue;
        }

        PeakSample {
            level: self.0[channel.index()],
            channel,
        }
    }
}

impl From<[f32; 3]> for LinearRgb {
    fn from(components: [f32; 3]) -> Self {
        Self::new(components)
    }
}

impl From<LinearRgb> for [f32; 3] {
    fn from(color: LinearRgb) -> Self {
        color.components()
    }
}

/// Maps a linear HDR color into display-linear RGB.
pub trait ToneMapper {
    /// Maps `color` into finite `0.0..=1.0` display-linear components.
    #[must_use]
    fn map(&self, color: LinearRgb) -> LinearRgb;
}

/// Identifies a positive finite scene level that maps to display white.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhitePoint(f32);

impl WhitePoint {
    /// Creates a white point from a positive finite linear-light level.
    ///
    /// Returns `None` when `level` is zero, negative, or non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::WhitePoint;
    ///
    /// assert_eq!(WhitePoint::new(4.0).map(WhitePoint::level), Some(4.0));
    /// assert!(WhitePoint::new(0.0).is_none());
    /// ```
    #[must_use]
    pub fn new(level: f32) -> Option<Self> {
        (level.is_finite() && level > 0.0).then_some(Self(level))
    }

    /// Returns the relative linear-light level represented by this white point.
    #[must_use]
    pub const fn level(self) -> f32 {
        self.0
    }
}

/// Identifies the RGB component that determines a content-light level.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ColorChannel {
    /// Red determines the light level.
    Red,
    /// Green determines the light level.
    Green,
    /// Blue determines the light level.
    Blue,
}

impl ColorChannel {
    const fn index(self) -> usize {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
        }
    }
}

/// Stores a still image's 99.99th-percentile maximum content light level.
///
/// The level uses the same relative or absolute unit as the linear RGB input. It is selected from
/// per-pixel `max(R, G, B)` values using the nearest-rank convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxCll {
    level: f32,
    channel: ColorChannel,
}

impl MaxCll {
    /// Returns the 99.99th-percentile content level.
    #[must_use]
    pub const fn level(self) -> f32 {
        self.level
    }

    /// Returns the component that determines the selected content level.
    #[must_use]
    pub const fn channel(self) -> ColorChannel {
        self.channel
    }

    /// Returns this content level as a white point when it is nonzero.
    #[must_use]
    pub fn white_point(self) -> Option<WhitePoint> {
        WhitePoint::new(self.level)
    }
}

/// Reports a mismatch between declared and observed `MaxCLL` pixel counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxCllPixelCountError {
    expected: usize,
    observed: usize,
}

impl MaxCllPixelCountError {
    /// Returns the pixel count declared when the estimator was created.
    #[must_use]
    pub const fn expected(self) -> usize {
        self.expected
    }

    /// Returns the number of colors passed to the estimator.
    #[must_use]
    pub const fn observed(self) -> usize {
        self.observed
    }
}

impl fmt::Display for MaxCllPixelCountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MaxCLL estimator expected {} pixels but observed {}",
            self.expected, self.observed
        )
    }
}

impl std::error::Error for MaxCllPixelCountError {}

/// Computes nearest-rank p99.99 `MaxCLL` from a stream of linear RGB colors.
///
/// The estimator retains only the brightest `floor(pixel_count / 10_000) + 1` samples. Declaring
/// the count up front therefore bounds memory while producing the same result as sorting all
/// per-pixel `max(R, G, B)` values.
#[derive(Debug)]
pub struct MaxCllEstimator {
    expected: NonZeroUsize,
    retained: usize,
    observed: usize,
    peaks: BinaryHeap<Reverse<PeakSample>>,
}

impl MaxCllEstimator {
    /// Creates an estimator for exactly `pixel_count` active-image pixels.
    #[must_use]
    pub fn new(pixel_count: NonZeroUsize) -> Self {
        let retained = pixel_count.get() / 10_000 + 1;
        Self {
            expected: pixel_count,
            retained,
            observed: 0,
            peaks: BinaryHeap::with_capacity(retained),
        }
    }

    /// Includes one active-image color in the `MaxCLL` estimate.
    pub fn observe(&mut self, color: LinearRgb) {
        self.observed = self.observed.saturating_add(1);
        let peak = color.max_component();

        if self.peaks.len() < self.retained {
            self.peaks.push(Reverse(peak));
            return;
        }

        if let Some(mut threshold) = self.peaks.peek_mut() {
            if peak > threshold.0 {
                *threshold = Reverse(peak);
            }
        } else {
            self.peaks.push(Reverse(peak));
        }
    }

    /// Finishes the estimate after exactly the declared number of observations.
    ///
    /// # Errors
    ///
    /// Returns [`MaxCllPixelCountError`] when the observed pixel count differs from the count passed
    /// to [`MaxCllEstimator::new`].
    pub fn finish(self) -> Result<MaxCll, MaxCllPixelCountError> {
        if self.observed != self.expected.get() {
            return Err(MaxCllPixelCountError {
                expected: self.expected.get(),
                observed: self.observed,
            });
        }

        let peak = self.peaks.peek().map_or(
            PeakSample {
                level: 0.0,
                channel: ColorChannel::Red,
            },
            |peak| peak.0,
        );
        Ok(MaxCll {
            level: peak.level,
            channel: peak.channel,
        })
    }
}

/// Estimates p99.99 `MaxCLL` for a complete still-image color slice.
///
/// Returns `None` when `colors` is empty.
///
/// # Examples
///
/// ```
/// use tonemapping::{LinearRgb, estimate_max_cll};
///
/// let colors = [LinearRgb::new([1.0, 2.0, 4.0])];
/// assert_eq!(estimate_max_cll(&colors).map(|value| value.level()), Some(4.0));
/// ```
#[must_use]
pub fn estimate_max_cll(colors: &[LinearRgb]) -> Option<MaxCll> {
    let pixel_count = NonZeroUsize::new(colors.len())?;
    let mut estimator = MaxCllEstimator::new(pixel_count);
    for color in colors {
        estimator.observe(*color);
    }
    estimator.finish().ok()
}

#[derive(Clone, Copy, Debug)]
struct PeakSample {
    level: f32,
    channel: ColorChannel,
}

impl PartialEq for PeakSample {
    fn eq(&self, other: &Self) -> bool {
        self.level.to_bits() == other.level.to_bits() && self.channel == other.channel
    }
}

impl Eq for PeakSample {}

impl PartialOrd for PeakSample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PeakSample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level
            .total_cmp(&other.level)
            .then_with(|| self.channel.cmp(&other.channel))
    }
}

/// Clamps every component to the displayable range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Clamp;

impl ToneMapper for Clamp {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        LinearRgb::displayable(color.components())
    }
}

/// Scales a scene white point to one before clamping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaledClamp {
    white_point: WhitePoint,
}

impl ScaledClamp {
    /// Creates a scaled clamp that maps `white_point` to one.
    #[must_use]
    pub const fn new(white_point: WhitePoint) -> Self {
        Self { white_point }
    }

    /// Returns the scene white point mapped to the display maximum.
    #[must_use]
    pub const fn white_point(self) -> WhitePoint {
        self.white_point
    }
}

impl ToneMapper for ScaledClamp {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let divisor = self.white_point.level();
        LinearRgb::displayable(color.components().map(|component| component / divisor))
    }
}

/// Applies the simple Reinhard curve independently to each component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reinhard;

impl ToneMapper for Reinhard {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        LinearRgb::displayable(
            color
                .components()
                .map(|component| component / (1.0 + component)),
        )
    }
}

/// Applies the white-point Reinhard curve independently to each component.
///
/// Components at the white point map to one, while brighter components clip at the display
/// boundary. For a still image, [`MaxCll`] supplies the p99.99 `max(R, G, B)` white point described
/// by Smith and Zink's outlier-rejection method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedReinhard {
    white_point: WhitePoint,
}

impl ExtendedReinhard {
    /// Creates an extended Reinhard operator with `white_point`.
    #[must_use]
    pub const fn new(white_point: WhitePoint) -> Self {
        Self { white_point }
    }

    /// Creates an operator from a nonzero p99.99 `MaxCLL` estimate.
    ///
    /// Returns `None` for an entirely black image, whose `MaxCLL` is zero.
    #[must_use]
    pub fn from_max_cll(max_cll: MaxCll) -> Option<Self> {
        max_cll.white_point().map(Self::new)
    }

    /// Returns the scene white point mapped to the display maximum.
    #[must_use]
    pub const fn white_point(self) -> WhitePoint {
        self.white_point
    }
}

impl ToneMapper for ExtendedReinhard {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let white_squared = self.white_point.level().powi(2);
        LinearRgb::displayable(
            color
                .components()
                .map(|component| component * (1.0 + component / white_squared) / (1.0 + component)),
        )
    }
}

/// Applies the simple Reinhard curve to luminance while retaining color ratios.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LuminanceReinhard;

impl ToneMapper for LuminanceReinhard {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let scale = 1.0 / (1.0 + color.luminance());
        LinearRgb::displayable(color.components().map(|component| component * scale))
    }
}

/// Identifies a positive finite luminance that maps to display white.
///
/// This type is distinct from [`MaxCll`], which is computed from `max(R, G, B)` rather than
/// luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LuminanceWhitePoint(WhitePoint);

impl LuminanceWhitePoint {
    /// Creates a white point from a positive finite luminance.
    ///
    /// Returns `None` when `luminance` is zero, negative, or non-finite.
    #[must_use]
    pub fn new(luminance: f32) -> Option<Self> {
        WhitePoint::new(luminance).map(Self)
    }

    /// Returns the linear luminance represented by this white point.
    #[must_use]
    pub const fn luminance(self) -> f32 {
        self.0.level()
    }
}

/// Estimates a p99.99 luminance white point for a complete still image.
///
/// This applies the paper's spatial outlier-rejection percentile to Rec. 709 luminance. It is an
/// analogous statistic for luminance-based curves, not `MaxCLL`.
#[must_use]
pub fn estimate_luminance_white_point(colors: &[LinearRgb]) -> Option<LuminanceWhitePoint> {
    let pixel_count = colors.len();
    if pixel_count == 0 {
        return None;
    }

    let mut luminances: Vec<_> = colors.iter().map(|color| color.luminance()).collect();
    let rank_index = pixel_count - pixel_count / 10_000 - 1;
    let (_lower, luminance, _upper) = luminances.select_nth_unstable_by(rank_index, f32::total_cmp);
    LuminanceWhitePoint::new(*luminance)
}

/// Applies extended Reinhard to luminance while retaining color ratios.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedLuminanceReinhard {
    white_point: LuminanceWhitePoint,
}

impl ExtendedLuminanceReinhard {
    /// Creates an operator with the supplied luminance `white_point`.
    #[must_use]
    pub const fn new(white_point: LuminanceWhitePoint) -> Self {
        Self { white_point }
    }

    /// Returns the luminance mapped to display white.
    #[must_use]
    pub const fn white_point(self) -> LuminanceWhitePoint {
        self.white_point
    }
}

impl ToneMapper for ExtendedLuminanceReinhard {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let luminance = color.luminance();
        let white_squared = self.white_point.luminance().powi(2);
        let scale = (1.0 + luminance / white_squared) / (1.0 + luminance);
        LinearRgb::displayable(color.components().map(|component| component * scale))
    }
}

/// Blends component-wise and luminance-based Reinhard results per component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReinhardJodie;

impl ToneMapper for ReinhardJodie {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let components = color.components();
        let luminance_scale = 1.0 / (1.0 + color.luminance());
        let component_mapped = components.map(|component| component / (1.0 + component));
        let luminance_mapped = components.map(|component| component * luminance_scale);
        let blended = std::array::from_fn(|index| {
            let weight = component_mapped[index];
            luminance_mapped[index] * (1.0 - weight) + component_mapped[index] * weight
        });
        LinearRgb::displayable(blended)
    }
}

/// Applies John Hable's Uncharted 2 filmic curve component-wise.
///
/// The operator includes the article's exposure bias of two and normalizes the curve at its `11.2`
/// reference input. Consequently, a scene component of `5.6` maps to display white.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Uncharted2;

impl ToneMapper for Uncharted2 {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let white_scale = 1.0 / uncharted2_partial(11.2);
        LinearRgb::displayable(
            color
                .components()
                .map(|component| uncharted2_partial(component * 2.0) * white_scale),
        )
    }
}

fn uncharted2_partial(value: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;

    ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - E / F
}

/// Applies Stephen Hill's fitted ACES reference and display transform.
///
/// This compact fit uses the article's linear sRGB input and output matrices. It is a practical
/// filmic curve rather than a complete Academy Color Encoding System pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AcesFitted;

impl ToneMapper for AcesFitted {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        const INPUT_MATRIX: [[f32; 3]; 3] = [
            [0.597_19, 0.354_58, 0.048_23],
            [0.076_00, 0.908_34, 0.015_66],
            [0.028_40, 0.133_83, 0.837_77],
        ];
        const OUTPUT_MATRIX: [[f32; 3]; 3] = [
            [1.604_75, -0.531_08, -0.073_67],
            [-0.102_08, 1.108_13, -0.006_05],
            [-0.003_27, -0.072_76, 1.076_02],
        ];

        let transformed = multiply_rgb(INPUT_MATRIX, color.components());
        let fitted = transformed.map(|component| {
            let numerator = component * (component + 0.024_578_6) - 0.000_090_537;
            let denominator = component * (0.983_729 * component + 0.432_951) + 0.238_081;
            numerator / denominator
        });
        LinearRgb::displayable(multiply_rgb(OUTPUT_MATRIX, fitted))
    }
}

fn multiply_rgb(matrix: [[f32; 3]; 3], color: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * color[0] + row[1] * color[1] + row[2] * color[2])
}

/// Applies Krzysztof Narkowicz's scalar ACES approximation component-wise.
///
/// The input is pre-exposed by `0.6`, matching the article's comparison with the fitted transform.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AcesApproximate;

impl ToneMapper for AcesApproximate {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        const A: f32 = 2.51;
        const B: f32 = 0.03;
        const C: f32 = 2.43;
        const D: f32 = 0.59;
        const E: f32 = 0.14;

        LinearRgb::displayable(color.components().map(|component| {
            let exposed = component * 0.6;
            exposed * (A * exposed + B) / (exposed * (C * exposed + D) + E)
        }))
    }
}

/// Stores a normalized camera response function sampled as a lookup table.
///
/// Irradiance samples must be strictly increasing within `0.0..=1.0`. Intensity samples must be
/// nondecreasing within the same range. Mapping uses linear interpolation between adjacent samples.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraResponse {
    irradiance: Box<[f32]>,
    intensity: Box<[f32]>,
}

impl CameraResponse {
    /// Creates a validated camera response from corresponding sample arrays.
    ///
    /// # Errors
    ///
    /// Returns [`CameraResponseError`] when the arrays have different lengths, contain fewer than
    /// two samples, contain non-finite or out-of-range values, or are not ordered as required.
    pub fn new(
        irradiance: impl Into<Box<[f32]>>,
        intensity: impl Into<Box<[f32]>>,
    ) -> Result<Self, CameraResponseError> {
        let irradiance = irradiance.into();
        let intensity = intensity.into();

        if irradiance.len() != intensity.len() {
            return Err(CameraResponseError::LengthMismatch {
                irradiance: irradiance.len(),
                intensity: intensity.len(),
            });
        }
        if irradiance.len() < 2 {
            return Err(CameraResponseError::TooFewSamples(irradiance.len()));
        }

        for (index, sample) in irradiance.iter().copied().enumerate() {
            if !sample.is_finite() || !(0.0..=1.0).contains(&sample) {
                return Err(CameraResponseError::IrradianceOutOfRange(index));
            }
            if index > 0 && sample <= irradiance[index - 1] {
                return Err(CameraResponseError::IrradianceNotIncreasing(index));
            }
        }

        for (index, sample) in intensity.iter().copied().enumerate() {
            if !sample.is_finite() || !(0.0..=1.0).contains(&sample) {
                return Err(CameraResponseError::IntensityOutOfRange(index));
            }
            if index > 0 && sample < intensity[index - 1] {
                return Err(CameraResponseError::IntensityDecreases(index));
            }
        }

        Ok(Self {
            irradiance,
            intensity,
        })
    }

    /// Returns the number of samples in the response curve.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.irradiance.len()
    }

    /// Creates a tone mapper whose horizontal curve extent is `white_point`.
    ///
    /// The article calls this parameter "ISO", but it acts as a linear-light white point: larger
    /// values darken the same input color.
    #[must_use]
    pub const fn tone_mapper(&self, white_point: WhitePoint) -> CameraToneMapper<'_> {
        CameraToneMapper {
            response: self,
            white_point,
        }
    }

    fn intensity_at(&self, irradiance: f32) -> f32 {
        match self
            .irradiance
            .binary_search_by(|sample| sample.total_cmp(&irradiance))
        {
            Ok(index) => self.intensity[index],
            Err(0) => self.intensity[0],
            Err(upper) if upper == self.irradiance.len() => self.intensity[upper - 1],
            Err(upper) => {
                let lower = upper - 1;
                let low_irradiance = self.irradiance[lower];
                let high_irradiance = self.irradiance[upper];
                let position = (irradiance - low_irradiance) / (high_irradiance - low_irradiance);
                let low_intensity = self.intensity[lower];
                let high_intensity = self.intensity[upper];
                low_intensity * (1.0 - position) + high_intensity * position
            }
        }
    }
}

/// Maps colors through a sampled camera response curve.
#[derive(Clone, Copy, Debug)]
pub struct CameraToneMapper<'a> {
    response: &'a CameraResponse,
    white_point: WhitePoint,
}

impl CameraToneMapper<'_> {
    /// Returns the scene level normalized to the end of the response curve.
    #[must_use]
    pub const fn white_point(self) -> WhitePoint {
        self.white_point
    }
}

impl ToneMapper for CameraToneMapper<'_> {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        let white = self.white_point.level();
        let mapped = color.components().map(|component| {
            let irradiance = (component / white).clamp(0.0, 1.0);
            self.response.intensity_at(irradiance)
        });
        LinearRgb::displayable(mapped)
    }
}

/// Describes invalid camera-response lookup-table data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraResponseError {
    /// The irradiance and intensity arrays have different lengths.
    LengthMismatch {
        /// Number of irradiance samples.
        irradiance: usize,
        /// Number of intensity samples.
        intensity: usize,
    },
    /// The curve contains fewer than two samples.
    TooFewSamples(usize),
    /// An irradiance sample is non-finite or outside `0.0..=1.0`.
    IrradianceOutOfRange(usize),
    /// An irradiance sample does not exceed its predecessor.
    IrradianceNotIncreasing(usize),
    /// An intensity sample is non-finite or outside `0.0..=1.0`.
    IntensityOutOfRange(usize),
    /// An intensity sample is lower than its predecessor.
    IntensityDecreases(usize),
}

impl fmt::Display for CameraResponseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LengthMismatch {
                irradiance,
                intensity,
            } => write!(
                formatter,
                "camera response has {irradiance} irradiance samples and {intensity} intensity samples"
            ),
            Self::TooFewSamples(count) => {
                write!(
                    formatter,
                    "camera response requires two samples, got {count}"
                )
            }
            Self::IrradianceOutOfRange(index) => write!(
                formatter,
                "camera irradiance sample {index} is not finite normalized data"
            ),
            Self::IrradianceNotIncreasing(index) => write!(
                formatter,
                "camera irradiance sample {index} does not increase"
            ),
            Self::IntensityOutOfRange(index) => write!(
                formatter,
                "camera intensity sample {index} is not finite normalized data"
            ),
            Self::IntensityDecreases(index) => {
                write!(formatter, "camera intensity sample {index} decreases")
            }
        }
    }
}

impl std::error::Error for CameraResponseError {}

/// Compresses highlights above a fixed knee using the content peak.
///
/// This operator preserves input luminance below `0.75`, maps `max_content_level` to `1.0`, and
/// scales all channels together before fitting the result into the target gamut. It exists for the
/// decoder integration that predates the named operators provided by this crate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearShoulder {
    max_content_level: f32,
}

impl LinearShoulder {
    /// Creates an operator for a finite, nonnegative `max_content_level`.
    ///
    /// Returns `None` for negative or non-finite levels.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::LinearShoulder;
    ///
    /// assert!(LinearShoulder::new(4.0).is_some());
    /// assert!(LinearShoulder::new(f32::NAN).is_none());
    /// ```
    #[must_use]
    pub fn new(max_content_level: f32) -> Option<Self> {
        (max_content_level.is_finite() && max_content_level >= 0.0)
            .then_some(Self { max_content_level })
    }
}

impl ToneMapper for LinearShoulder {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        const KNEE: f32 = 0.75;

        let luminance = color.luminance();
        if luminance == 0.0 {
            return LinearRgb::default();
        }

        let mapped_luminance = if self.max_content_level <= 1.0 || luminance <= KNEE {
            luminance
        } else {
            let input_range = self.max_content_level - KNEE;
            let output_range = 1.0 - KNEE;
            let softness = input_range * output_range / (self.max_content_level - 1.0);
            let distance = luminance - KNEE;
            KNEE + distance / (1.0 + distance / softness)
        };

        let scale = mapped_luminance / luminance;
        let mut mapped = color.0.map(|component| component * scale);
        let peak = mapped.into_iter().fold(0.0_f32, f32::max);
        if peak > 1.0 {
            mapped = mapped.map(|component| component / peak);
        }

        LinearRgb::displayable(mapped)
    }
}

fn sanitize_component(component: f32) -> f32 {
    if component.is_nan() || component <= 0.0 {
        0.0
    } else {
        component.min(HDR_COMPONENT_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approximately_equal(actual: f32, expected: f32) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }

    fn assert_components_close(actual: LinearRgb, expected: [f32; 3]) {
        for (actual, expected) in actual.components().into_iter().zip(expected) {
            assert_approximately_equal(actual, expected);
        }
    }

    #[test]
    fn shoulder_compresses_hdr_white_into_sdr() {
        let mapper = LinearShoulder::new(4.0).unwrap();

        assert_eq!(mapper.map(LinearRgb::default()), LinearRgb::default());
        let reference_white = mapper.map(LinearRgb::new([1.0; 3]));
        let hdr_white = mapper.map(LinearRgb::new([4.0; 3]));

        assert!(reference_white.components()[0] > 0.85);
        assert_eq!(hdr_white.components(), [1.0; 3]);
    }

    #[test]
    fn shoulder_preserves_channel_order_during_gamut_compression() {
        let mapper = LinearShoulder::new(4.0).unwrap();
        let [red, green, blue] = mapper.map(LinearRgb::new([4.0, 2.0, 1.0])).components();

        assert_eq!(red, 1.0);
        assert!(red > green);
        assert!(green > blue);
    }

    #[test]
    fn clamp_discards_values_outside_the_display_range() {
        let color = Clamp.map(LinearRgb::new([0.25, 1.0, 4.0]));

        assert_eq!(color.components(), [0.25, 1.0, 1.0]);
    }

    #[test]
    fn scaled_clamp_maps_its_white_point_to_one() {
        let mapper = ScaledClamp::new(WhitePoint::new(4.0).unwrap());
        let color = mapper.map(LinearRgb::new([1.0, 4.0, 8.0]));

        assert_eq!(color.components(), [0.25, 1.0, 1.0]);
        assert_eq!(mapper.white_point().level(), 4.0);
    }

    #[test]
    fn reinhard_maps_reference_white_to_middle_gray() {
        let color = Reinhard.map(LinearRgb::new([0.0, 0.18, 1.0]));
        let [black, middle_gray, reference_white] = color.components();

        assert_eq!(black, 0.0);
        assert_approximately_equal(middle_gray, 0.152_542_37);
        assert_eq!(reference_white, 0.5);
    }

    #[test]
    fn extended_reinhard_maps_the_percentile_white_point_to_one() {
        let colors = [LinearRgb::new([4.0, 2.0, 1.0])];
        let max_cll = estimate_max_cll(&colors).unwrap();
        let mapper = ExtendedReinhard::from_max_cll(max_cll).unwrap();
        let [white, middle, low] = mapper.map(colors[0]).components();

        assert_eq!(white, 1.0);
        assert_approximately_equal(middle, 0.75);
        assert_approximately_equal(low, 0.531_25);
        assert_eq!(mapper.white_point().level(), 4.0);
    }

    #[test]
    fn extended_reinhard_rejects_a_zero_white_point() {
        let black = estimate_max_cll(&[LinearRgb::default()]).unwrap();

        assert!(ExtendedReinhard::from_max_cll(black).is_none());
    }

    #[test]
    fn luminance_reinhard_preserves_unclipped_color_ratios() {
        let input = LinearRgb::new([1.0, 0.5, 0.25]);
        let output = LuminanceReinhard.map(input);
        let [red, green, blue] = output.components();

        assert_approximately_equal(red / green, 2.0);
        assert_approximately_equal(green / blue, 2.0);
        assert_approximately_equal(
            output.luminance(),
            input.luminance() / (1.0 + input.luminance()),
        );
    }

    #[test]
    fn extended_luminance_reinhard_uses_a_luminance_white_point() {
        let white_point = LuminanceWhitePoint::new(4.0).unwrap();
        let mapper = ExtendedLuminanceReinhard::new(white_point);
        let [red, green, blue] = mapper.map(LinearRgb::new([1.0, 0.5, 0.25])).components();

        assert_approximately_equal(red, 0.652_772_3);
        assert_approximately_equal(green, 0.326_386_15);
        assert_approximately_equal(blue, 0.163_193_08);
        assert_eq!(mapper.white_point().luminance(), 4.0);
    }

    #[test]
    fn luminance_white_point_rejects_the_brightest_outlier() {
        let mut colors = vec![LinearRgb::new([1.0; 3]); 9_998];
        colors.extend([LinearRgb::new([4.0; 3]), LinearRgb::new([126.0; 3])]);

        let white_point = estimate_luminance_white_point(&colors).unwrap();

        assert_approximately_equal(white_point.luminance(), 4.0);
    }

    #[test]
    fn reinhard_jodie_blends_luminance_and_component_curves() {
        let [red, green, blue] = ReinhardJodie
            .map(LinearRgb::new([1.0, 0.5, 0.25]))
            .components();

        assert_approximately_equal(red, 0.564_811_9);
        assert_approximately_equal(green, 0.320_985_7);
        assert_approximately_equal(blue, 0.165_924_76);
        assert_eq!(
            ReinhardJodie.map(LinearRgb::new([1.0; 3])),
            Reinhard.map(LinearRgb::new([1.0; 3]))
        );
    }

    #[test]
    fn uncharted_two_uses_the_article_exposure_and_white_scale() {
        let [middle_gray, reference_white, filmic_white] = Uncharted2
            .map(LinearRgb::new([0.18, 1.0, 5.6]))
            .components();

        assert_approximately_equal(middle_gray, 0.128_338_45);
        assert_approximately_equal(reference_white, 0.492_918_55);
        assert_eq!(filmic_white, 1.0);
    }

    #[test]
    fn aces_fitted_uses_the_reference_matrix_orientation() {
        assert_components_close(
            AcesFitted.map(LinearRgb::new([1.0, 0.0, 0.0])),
            [0.688_027_86, 0.0, 0.002_639_006_8],
        );
        assert_components_close(
            AcesFitted.map(LinearRgb::new([0.0, 1.0, 0.0])),
            [0.101_613_28, 0.623_659, 0.028_843_72],
        );
        assert_components_close(
            AcesFitted.map(LinearRgb::new([0.0, 0.0, 1.0])),
            [0.0, 0.0, 0.601_758_84],
        );
    }

    #[test]
    fn aces_approximate_includes_the_article_pre_exposure() {
        let [middle_gray, reference_white, clipped] = AcesApproximate
            .map(LinearRgb::new([0.18, 1.0, 100.0]))
            .components();

        assert_approximately_equal(middle_gray, 0.140_119_57);
        assert_approximately_equal(reference_white, 0.673_290_5);
        assert_eq!(clipped, 1.0);
    }

    #[test]
    fn camera_response_interpolates_knots_and_clamps_endpoints() {
        let response = CameraResponse::new(vec![0.0, 0.5, 1.0], vec![0.0, 0.25, 1.0]).unwrap();
        let mapper = response.tone_mapper(WhitePoint::new(6.0).unwrap());

        assert_components_close(
            mapper.map(LinearRgb::new([0.0, 1.5, 3.0])),
            [0.0, 0.125, 0.25],
        );
        assert_components_close(
            mapper.map(LinearRgb::new([3.0, 6.0, 9.0])),
            [0.25, 1.0, 1.0],
        );
        assert_eq!(response.sample_count(), 3);
        assert_eq!(mapper.white_point().level(), 6.0);
    }

    #[test]
    fn camera_response_rejects_malformed_curves() {
        assert_eq!(
            CameraResponse::new(vec![0.0, 1.0], vec![0.0]).unwrap_err(),
            CameraResponseError::LengthMismatch {
                irradiance: 2,
                intensity: 1
            }
        );
        assert_eq!(
            CameraResponse::new(vec![0.0, 0.5, 0.5], vec![0.0, 0.5, 1.0]).unwrap_err(),
            CameraResponseError::IrradianceNotIncreasing(2)
        );
        assert_eq!(
            CameraResponse::new(vec![0.0, 0.5, 1.0], vec![0.0, 0.75, 0.5]).unwrap_err(),
            CameraResponseError::IntensityDecreases(2)
        );
    }

    #[test]
    fn max_cll_rejects_the_brightest_point_zero_one_percent() {
        let mut colors = vec![LinearRgb::new([1.0; 3]); 9_998];
        colors.extend([LinearRgb::new([4.0; 3]), LinearRgb::new([126.0; 3])]);

        let max_cll = estimate_max_cll(&colors).unwrap();

        assert_eq!(max_cll.level(), 4.0);
        assert_eq!(max_cll.channel(), ColorChannel::Red);
        assert_eq!(max_cll.white_point().map(WhitePoint::level), Some(4.0));
    }

    #[test]
    fn max_cll_uses_max_rgb_instead_of_luminance() {
        let max_cll = estimate_max_cll(&[LinearRgb::new([0.0, 0.0, 5.0])]).unwrap();

        assert_eq!(max_cll.level(), 5.0);
        assert_eq!(max_cll.channel(), ColorChannel::Blue);
    }

    #[test]
    fn max_cll_reports_incomplete_pixel_streams() {
        let mut estimator = MaxCllEstimator::new(NonZeroUsize::new(2).unwrap());
        estimator.observe(LinearRgb::new([1.0; 3]));

        let error = estimator.finish().unwrap_err();

        assert_eq!(error.expected(), 2);
        assert_eq!(error.observed(), 1);
        assert_eq!(
            error.to_string(),
            "MaxCLL estimator expected 2 pixels but observed 1"
        );
    }
}
