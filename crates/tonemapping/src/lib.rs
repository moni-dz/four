#![feature(portable_simd)]
#![warn(missing_docs)]
//! Maps high-dynamic-range linear RGB colors into a displayable range.
//!
//! Tone-mapping operators accept [`LinearRGB`] values whose components are relative linear-light
//! levels. A component of `1.0` conventionally represents the target display's reference white.
//! Operators return finite components in the inclusive range `0.0..=1.0`; transfer encoding and
//! integer quantization remain the caller's responsibility.
//!
//! [`Clamp`] and [`ScaledClamp`] provide clipping baselines. The Reinhard family includes
//! component-wise, luminance-preserving, white-point, and Reinhard-Jodie variants. [`Hable`],
//! [`AcesFitted`], and [`AcesApproximate`] provide filmic curves, while [`CameraResponse`] applies
//! a caller-supplied normalized camera-response lookup table.
//! [`ToneMappingMethod`] enumerates the built-in operator families that need no custom response.
//!
//! [`estimate_max_cll`] selects the nearest-rank 99.99th percentile of per-pixel `max(R, G, B)`.
//! [`MaxCllEstimator`] can select either that percentile or the true maximum through
//! [`MaxCllMode`]. Inputs determine the unit: absolute-nit inputs produce `MaxCLL` in nits, while
//! relative inputs produce a relative light level. Luminance-based Reinhard uses the distinct
//! [`estimate_luminance_white_point`] statistic.
//!
//! # Example
//!
//! ```
//! use tonemapping::{ExtendedReinhard, LinearRGB, ToneMapper, estimate_max_cll};
//!
//! let pixels = [
//!     LinearRGB::new([0.5, 1.0, 2.0]),
//!     LinearRGB::new([1.0, 2.0, 4.0]),
//! ];
//! let max_cll = estimate_max_cll(&pixels).expect("the image is not empty");
//! let mapper = ExtendedReinhard::from_max_cll(max_cll).expect("the image is not black");
//! let display_linear = mapper.map(pixels[0]);
//!
//! assert!(display_linear.components().into_iter().all(|value| (0.0..=1.0).contains(&value)));
//! ```

use multiversion::multiversion;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt;
use std::num::NonZeroUsize;
use std::simd::{
    Select, Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

const REC709_LUMINANCE: [f64; 3] = [0.212_6, 0.715_2, 0.072_2];
const TONE_MAPPING_LANES: usize = 4;
const MAX_CLL_LANES: usize = 8;

type F64x4 = Simd<f64, TONE_MAPPING_LANES>;
type F32x8 = Simd<f32, MAX_CLL_LANES>;
type I32x8 = Simd<i32, MAX_CLL_LANES>;

const ACES_INPUT_MATRIX: [[f64; 3]; 3] = [
    [0.597_19, 0.354_58, 0.048_23],
    [0.076_00, 0.908_34, 0.015_66],
    [0.028_40, 0.133_83, 0.837_77],
];
const ACES_OUTPUT_MATRIX: [[f64; 3]; 3] = [
    [1.604_75, -0.531_08, -0.073_67],
    [-0.102_08, 1.108_13, -0.006_05],
    [-0.003_27, -0.072_76, 1.076_02],
];

/// Stores a finite, nonnegative linear RGB color.
///
/// Construction replaces negative and `NaN` components with zero and positive infinity with
/// `f32::MAX`. Every finite nonnegative `f32` is preserved.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinearRGB([f32; 3]);

impl LinearRGB {
    /// Creates a sanitized linear RGB color from `components`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::LinearRGB;
    ///
    /// let color = LinearRGB::new([2.0, -1.0, f32::NAN]);
    /// assert_eq!(color.components(), [2.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn new(components: [f32; 3]) -> Self {
        Self(components.map(sanitize_component))
    }

    /// Returns the red, green, and blue components in that order.
    #[must_use]
    #[inline]
    pub const fn components(self) -> [f32; 3] {
        self.0
    }

    #[inline]
    fn components_f64(self) -> [f64; 3] {
        self.0.map(f64::from)
    }

    /// Returns linear Rec. 709 luminance.
    #[must_use]
    #[inline]
    pub fn luminance(self) -> f32 {
        nonnegative_f64_to_f32(self.luminance_f64())
    }

    #[inline]
    fn luminance_f64(self) -> f64 {
        REC709_LUMINANCE[0] * f64::from(self.0[0])
            + REC709_LUMINANCE[1] * f64::from(self.0[1])
            + REC709_LUMINANCE[2] * f64::from(self.0[2])
    }

    #[inline]
    fn displayable(components: [f64; 3]) -> Self {
        Self(components.map(display_component))
    }

    #[inline]
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

impl From<[f32; 3]> for LinearRGB {
    fn from(components: [f32; 3]) -> Self {
        Self::new(components)
    }
}

impl From<LinearRGB> for [f32; 3] {
    fn from(color: LinearRGB) -> Self {
        color.components()
    }
}

/// Maps a linear HDR color into display-linear RGB.
pub trait ToneMapper {
    /// Maps `color` into finite `0.0..=1.0` display-linear components.
    #[must_use]
    fn map(&self, color: LinearRGB) -> LinearRGB;

    /// Maps every color in `colors` in place.
    ///
    /// The default implementation calls [`ToneMapper::map`] for each color. Built-in operators may
    /// override this method with a batch implementation while preserving the scalar result.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::{Clamp, LinearRGB, ToneMapper};
    ///
    /// let mut colors = [LinearRGB::new([0.5, 1.0, 4.0]); 4];
    /// Clamp.map_in_place(&mut colors);
    ///
    /// assert!(colors
    ///     .into_iter()
    ///     .all(|color| color.components() == [0.5, 1.0, 1.0]));
    /// ```
    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        for color in colors {
            *color = self.map(*color);
        }
    }
}

/// Identifies a built-in tone-mapping operator family.
///
/// White-point methods still require an image-specific white point when they are constructed.
/// [`CameraToneMapper`] is not listed because it additionally requires a caller-supplied
/// [`CameraResponse`], for which there is no universal built-in preset.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToneMappingMethod {
    /// Clips each component to the display range.
    Clamp,
    /// Scales components by an image white point before clipping.
    ScaledClamp,
    /// Applies simple Reinhard independently to each component.
    Reinhard,
    /// Applies white-point Reinhard independently to each component.
    #[default]
    ExtendedReinhard,
    /// Applies simple Reinhard to luminance while preserving color ratios.
    LuminanceReinhard,
    /// Applies white-point Reinhard to luminance while preserving color ratios.
    ExtendedLuminanceReinhard,
    /// Blends component-wise and luminance-based Reinhard results.
    ReinhardJodie,
    /// Applies the Uncharted 2 filmic curve.
    Hable,
    /// Applies the fitted ACES reference and display transform.
    ACESFitted,
    /// Applies the scalar ACES curve approximation.
    ACESApproximate,
}

impl ToneMappingMethod {
    /// Contains every selectable built-in method in display order.
    pub const ALL: [Self; 10] = [
        Self::Clamp,
        Self::ScaledClamp,
        Self::Reinhard,
        Self::ExtendedReinhard,
        Self::LuminanceReinhard,
        Self::ExtendedLuminanceReinhard,
        Self::ReinhardJodie,
        Self::Hable,
        Self::ACESFitted,
        Self::ACESApproximate,
    ];

    /// Returns the concise user-facing method name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Clamp => "Clamp",
            Self::ScaledClamp => "Scaled clamp",
            Self::Reinhard => "Reinhard",
            Self::ExtendedReinhard => "Extended Reinhard",
            Self::LuminanceReinhard => "Luminance Reinhard",
            Self::ExtendedLuminanceReinhard => "Extended luminance Reinhard",
            Self::ReinhardJodie => "Reinhard-Jodie",
            Self::Hable => "Hable",
            Self::ACESFitted => "ACES fitted",
            Self::ACESApproximate => "ACES approximate",
        }
    }
}

impl fmt::Display for ToneMappingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
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

    const fn simd_code(self) -> i32 {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
        }
    }

    fn from_simd_code(code: i32) -> Self {
        match code {
            0 => Self::Red,
            1 => Self::Green,
            2 => Self::Blue,
            _ => unreachable!("SIMD peak channel must be in 0..=2, got {code}"),
        }
    }
}

/// Selects how a still image's `MaxCLL` is estimated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaxCllMode {
    /// Uses the nearest-rank 99.99th percentile of per-pixel peak components.
    #[default]
    Percentile99_99,
    /// Uses the greatest per-pixel peak component without outlier rejection.
    TrueMaximum,
}

/// Stores a selected maximum content light level.
///
/// The level uses the same relative or absolute unit as the linear RGB input. It is selected from
/// per-pixel `max(R, G, B)` values according to a [`MaxCllMode`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxCll {
    level: f32,
    channel: ColorChannel,
}

impl MaxCll {
    /// Returns the selected content level.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MaxCLL estimator expected {} pixels but observed {}",
            self.expected, self.observed
        )
    }
}

impl std::error::Error for MaxCllPixelCountError {}

/// Computes a selected `MaxCLL` from a stream of linear RGB colors.
///
/// Percentile mode retains the brightest `floor(pixel_count / 10_000) + 1` samples. True-maximum
/// mode retains one sample. Declaring the count up front therefore bounds memory while producing
/// the same result as sorting all per-pixel `max(R, G, B)` values.
#[derive(Debug)]
pub struct MaxCllEstimator {
    expected: NonZeroUsize,
    retained: usize,
    observed: usize,
    peaks: BinaryHeap<Reverse<PeakSample>>,
}

impl MaxCllEstimator {
    /// Creates a p99.99 estimator for exactly `pixel_count` active-image pixels.
    #[must_use]
    pub fn new(pixel_count: NonZeroUsize) -> Self {
        Self::with_mode(pixel_count, MaxCllMode::Percentile99_99)
    }

    /// Creates a `mode` estimator for exactly `pixel_count` active-image pixels.
    #[must_use]
    pub fn with_mode(pixel_count: NonZeroUsize, mode: MaxCllMode) -> Self {
        let retained = match mode {
            MaxCllMode::Percentile99_99 => pixel_count.get() / 10_000 + 1,
            MaxCllMode::TrueMaximum => 1,
        };
        Self {
            expected: pixel_count,
            retained,
            observed: 0,
            peaks: BinaryHeap::with_capacity(retained),
        }
    }

    /// Includes one active-image color in the `MaxCLL` estimate.
    #[inline]
    pub fn observe(&mut self, color: LinearRGB) {
        self.observed = self.observed.saturating_add(1);
        self.retain_peak(color.max_component());
    }

    /// Includes active-image `colors` in the `MaxCLL` estimate.
    ///
    /// This produces the same estimate as calling [`MaxCllEstimator::observe`] in slice order.
    pub fn observe_many(&mut self, colors: &[LinearRGB]) {
        self.observed = self.observed.saturating_add(colors.len());

        let fill_count = (self.retained - self.peaks.len()).min(colors.len());
        for color in &colors[..fill_count] {
            self.retain_peak(color.max_component());
        }

        let remaining = &colors[fill_count..];
        if remaining.len() >= MAX_CLL_LANES {
            observe_max_cll_candidates(self, remaining);
        } else {
            for color in remaining {
                self.retain_peak(color.max_component());
            }
        }
    }

    #[inline]
    fn retain_peak(&mut self, peak: PeakSample) {
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
    /// when constructing the estimator.
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
/// use tonemapping::{LinearRGB, estimate_max_cll};
///
/// let colors = [LinearRGB::new([1.0, 2.0, 4.0])];
/// assert_eq!(estimate_max_cll(&colors).map(|value| value.level()), Some(4.0));
/// ```
#[must_use]
pub fn estimate_max_cll(colors: &[LinearRGB]) -> Option<MaxCll> {
    let pixel_count = NonZeroUsize::new(colors.len())?;
    let mut estimator = MaxCllEstimator::new(pixel_count);
    estimator.observe_many(colors);
    estimator.finish().ok()
}

#[derive(Clone, Copy, Debug)]
struct PeakSample {
    level: f32,
    channel: ColorChannel,
}

#[derive(Clone, Copy, Debug)]
struct OrderedLevel(f32);

impl PartialEq for OrderedLevel {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedLevel {}

impl PartialOrd for OrderedLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
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

// The heap threshold only rises once it is full. A lane that does not beat the threshold captured
// at the start of its chunk therefore cannot become a candidate while earlier lanes are retained.
#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn observe_max_cll_candidates(estimator: &mut MaxCllEstimator, colors: &[LinearRGB]) {
    debug_assert_eq!(
        estimator.peaks.len(),
        estimator.retained,
        "MaxCLL SIMD filtering requires a full retained-sample heap"
    );

    let (chunks, tail) = colors.as_chunks::<MAX_CLL_LANES>();
    for chunk in chunks {
        let red = F32x8::from_array(std::array::from_fn(|lane| chunk[lane].0[0]));
        let green = F32x8::from_array(std::array::from_fn(|lane| chunk[lane].0[1]));
        let blue = F32x8::from_array(std::array::from_fn(|lane| chunk[lane].0[2]));

        let green_is_higher = green.simd_gt(red);
        let mut levels = green_is_higher.select(green, red);
        let mut channels = green_is_higher.select(I32x8::splat(1), I32x8::splat(0));
        let blue_is_higher = blue.simd_gt(levels);
        levels = blue_is_higher.select(blue, levels);
        channels = blue_is_higher.select(I32x8::splat(2), channels);

        let threshold = estimator
            .peaks
            .peek()
            .expect("a full MaxCLL retained-sample heap has a threshold")
            .0;
        let threshold_levels = F32x8::splat(threshold.level);
        let candidates = levels.simd_gt(threshold_levels)
            | (levels.simd_eq(threshold_levels)
                & channels.simd_gt(I32x8::splat(threshold.channel.simd_code())));
        let candidate_lanes = candidates.to_array();
        let levels = levels.to_array();
        let channels = channels.to_array();

        for lane in 0..MAX_CLL_LANES {
            if candidate_lanes[lane] {
                estimator.retain_peak(PeakSample {
                    level: levels[lane],
                    channel: ColorChannel::from_simd_code(channels[lane]),
                });
            }
        }
    }

    for color in tail {
        estimator.retain_peak(color.max_component());
    }
}

/// Clamps every component to the displayable range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Clamp;

impl ToneMapper for Clamp {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        clamp_color(color)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        for color in colors {
            *color = clamp_color(*color);
        }
    }
}

#[inline]
fn clamp_color(color: LinearRGB) -> LinearRGB {
    LinearRGB(color.0.map(display_component_f32))
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
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let divisor = f64::from(self.white_point.level());
        LinearRGB::displayable(color.components_f64().map(|component| component / divisor))
    }
}

/// Applies the simple Reinhard curve independently to each component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reinhard;

impl ToneMapper for Reinhard {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        LinearRGB::displayable(
            color
                .components_f64()
                .map(|component| component / (1.0 + component)),
        )
    }
}

/// Applies the white-point Reinhard curve independently to each component.
///
/// Components at the white point map to one, while brighter components clip at the display
/// boundary. For a still image, [`MaxCll`] supplies a `max(R, G, B)` white point selected through
/// [`MaxCllMode`].
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

    /// Creates an operator from a selected nonzero `MaxCLL` statistic.
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
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let white_squared = f64::from(self.white_point.level()).powi(2);
        extended_reinhard(color, white_squared)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        let white_squared = f64::from(self.white_point.level()).powi(2);
        for color in colors {
            *color = extended_reinhard(*color, white_squared);
        }
    }
}

#[inline]
fn extended_reinhard(color: LinearRGB, white_squared: f64) -> LinearRGB {
    LinearRGB::displayable(
        color
            .components_f64()
            .map(|component| component * (1.0 + component / white_squared) / (1.0 + component)),
    )
}

/// Applies the simple Reinhard curve to luminance while retaining color ratios.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LuminanceReinhard;

impl ToneMapper for LuminanceReinhard {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let scale = 1.0 / (1.0 + color.luminance_f64());
        LinearRGB::displayable(color.components_f64().map(|component| component * scale))
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
pub fn estimate_luminance_white_point(colors: &[LinearRGB]) -> Option<LuminanceWhitePoint> {
    let pixel_count = colors.len();
    if pixel_count == 0 {
        return None;
    }

    let retained = pixel_count / 10_000 + 1;
    let mut luminances = BinaryHeap::with_capacity(retained);
    for color in colors {
        let luminance = OrderedLevel(color.luminance());
        if luminances.len() < retained {
            luminances.push(Reverse(luminance));
        } else if let Some(mut threshold) = luminances.peek_mut()
            && luminance > threshold.0
        {
            *threshold = Reverse(luminance);
        }
    }

    luminances
        .peek()
        .and_then(|luminance| LuminanceWhitePoint::new(luminance.0.0))
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
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let luminance = color.luminance_f64();
        let white_squared = f64::from(self.white_point.luminance()).powi(2);
        let scale = (1.0 + luminance / white_squared) / (1.0 + luminance);
        LinearRGB::displayable(color.components_f64().map(|component| component * scale))
    }
}

/// Blends component-wise and luminance-based Reinhard results per component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReinhardJodie;

impl ToneMapper for ReinhardJodie {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let components = color.components_f64();
        let luminance_scale = 1.0 / (1.0 + color.luminance_f64());
        let component_mapped = components.map(|component| component / (1.0 + component));
        let luminance_mapped = components.map(|component| component * luminance_scale);
        let blended = std::array::from_fn(|index| {
            let weight = component_mapped[index];
            luminance_mapped[index] * (1.0 - weight) + component_mapped[index] * weight
        });
        LinearRGB::displayable(blended)
    }
}

/// Applies John Hable's Uncharted 2 filmic curve component-wise.
///
/// The operator includes the article's exposure bias of two and normalizes the curve at its `11.2`
/// reference input. Consequently, a scene component of `5.6` maps to display white.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Hable;

impl ToneMapper for Hable {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let white_scale = 1.0 / hable_partial(11.2);
        LinearRGB::displayable(
            color
                .components_f64()
                .map(|component| hable_partial(component * 2.0) * white_scale),
        )
    }
}

fn hable_partial(value: f64) -> f64 {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;

    ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - E / F
}

/// Applies Stephen Hill's fitted ACES reference and display transform.
///
/// This compact fit uses the article's linear sRGB input and output matrices. It is a practical
/// filmic curve rather than a complete Academy Color Encoding System pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AcesFitted;

impl ToneMapper for AcesFitted {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        aces_fitted(color)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        let simd_len = colors.len() / TONE_MAPPING_LANES * TONE_MAPPING_LANES;
        let (simd_colors, tail) = colors.split_at_mut(simd_len);
        if !simd_colors.is_empty() {
            aces_fitted_batch(simd_colors);
        }
        for color in tail {
            *color = aces_fitted(*color);
        }
    }
}

#[inline]
fn aces_fitted(color: LinearRGB) -> LinearRGB {
    let transformed = multiply_rgb(ACES_INPUT_MATRIX, color.components_f64());
    let fitted = transformed.map(|component| {
        let numerator = component * (component + 0.024_578_6) - 0.000_090_537;
        let denominator = component * (0.983_729 * component + 0.432_951) + 0.238_081;
        numerator / denominator
    });
    LinearRGB::displayable(multiply_rgb(ACES_OUTPUT_MATRIX, fitted))
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn aces_fitted_batch(colors: &mut [LinearRGB]) {
    let (chunks, tail) = colors.as_chunks_mut::<TONE_MAPPING_LANES>();
    debug_assert!(
        tail.is_empty(),
        "fitted ACES SIMD input must contain complete four-pixel chunks"
    );

    for chunk in chunks {
        let color = [0, 1, 2].map(|channel| {
            F64x4::from_array(std::array::from_fn(|lane| {
                f64::from(chunk[lane].0[channel])
            }))
        });
        
        let transformed = ACES_INPUT_MATRIX.map(|row| {
            (F64x4::splat(row[0]) * color[0] + F64x4::splat(row[1]) * color[1])
                + F64x4::splat(row[2]) * color[2]
        });
        
        let fitted = transformed.map(|component| {
            let numerator =
                component * (component + F64x4::splat(0.024_578_6)) - F64x4::splat(0.000_090_537);
            let denominator = component
                * (F64x4::splat(0.983_729) * component + F64x4::splat(0.432_951))
                + F64x4::splat(0.238_081);
            numerator / denominator
        });
        
        let mapped = ACES_OUTPUT_MATRIX.map(|row| {
            (F64x4::splat(row[0]) * fitted[0] + F64x4::splat(row[1]) * fitted[1])
                + F64x4::splat(row[2]) * fitted[2]
        });
        
        let mapped = mapped.map(|component| {
            let zero = F64x4::splat(0.0);
            let one = F64x4::splat(1.0);
            let below = component.is_nan() | component.simd_le(zero);
            let bounded = below.select(zero, component.simd_ge(one).select(one, component));
            bounded.cast::<f32>().to_array()
        });

        for lane in 0..TONE_MAPPING_LANES {
            chunk[lane] = LinearRGB([mapped[0][lane], mapped[1][lane], mapped[2][lane]]);
        }
    }
}

fn multiply_rgb(matrix: [[f64; 3]; 3], color: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0] * color[0] + row[1] * color[1] + row[2] * color[2])
}

/// Applies Krzysztof Narkowicz's scalar ACES approximation component-wise.
///
/// The input is pre-exposed by `0.6`, matching the article's comparison with the fitted transform.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AcesApproximate;

impl ToneMapper for AcesApproximate {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        const A: f64 = 2.51;
        const B: f64 = 0.03;
        const C: f64 = 2.43;
        const D: f64 = 0.59;
        const E: f64 = 0.14;

        LinearRGB::displayable(color.components_f64().map(|component| {
            let exposed = component * 0.6;
            exposed * (A * exposed + B) / (exposed * (C * exposed + D) + E)
        }))
    }
}

/// Stores a normalized camera response function sampled as a lookup table.
///
/// Irradiance samples must run from zero to one and be strictly increasing. Intensity samples must
/// run from zero to one and be nondecreasing. Mapping uses linear interpolation between adjacent
/// samples.
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
    /// two samples, do not span normalized black through white, contain non-finite or out-of-range
    /// values, or are not ordered as required.
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

        #[allow(
            clippy::float_cmp,
            reason = "normalized camera curves must contain the exact conventional endpoints"
        )]
        if irradiance[0] != 0.0 || irradiance[irradiance.len() - 1] != 1.0 {
            return Err(CameraResponseError::IrradianceEndpoints);
        }
        #[allow(
            clippy::float_cmp,
            reason = "normalized camera curves must contain the exact conventional endpoints"
        )]
        if intensity[0] != 0.0 || intensity[intensity.len() - 1] != 1.0 {
            return Err(CameraResponseError::IntensityEndpoints);
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

    fn intensity_at(&self, irradiance: f64) -> f64 {
        match self
            .irradiance
            .binary_search_by(|sample| f64::from(*sample).total_cmp(&irradiance))
        {
            Ok(index) => f64::from(self.intensity[index]),
            Err(0) => f64::from(self.intensity[0]),
            Err(upper) if upper == self.irradiance.len() => f64::from(self.intensity[upper - 1]),
            Err(upper) => {
                let lower = upper - 1;
                let low_irradiance = f64::from(self.irradiance[lower]);
                let high_irradiance = f64::from(self.irradiance[upper]);
                let position = (irradiance - low_irradiance) / (high_irradiance - low_irradiance);
                let low_intensity = f64::from(self.intensity[lower]);
                let high_intensity = f64::from(self.intensity[upper]);
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
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let white = f64::from(self.white_point.level());
        let mapped = color.components_f64().map(|component| {
            let irradiance = (component / white).clamp(0.0, 1.0);
            self.response.intensity_at(irradiance)
        });
        LinearRGB::displayable(mapped)
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
    /// The irradiance domain does not start at zero and end at one.
    IrradianceEndpoints,
    /// The intensity range does not start at zero and end at one.
    IntensityEndpoints,
}

impl fmt::Display for CameraResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LengthMismatch {
                irradiance,
                intensity,
            } => write!(
                f,
                "camera response has {irradiance} irradiance samples and {intensity} intensity samples"
            ),
            Self::TooFewSamples(count) => {
                write!(f, "camera response requires two samples, got {count}")
            }
            Self::IrradianceOutOfRange(index) => write!(
                f,
                "camera irradiance sample {index} is not finite normalized data"
            ),
            Self::IrradianceNotIncreasing(index) => {
                write!(f, "camera irradiance sample {index} does not increase")
            }
            Self::IntensityOutOfRange(index) => write!(
                f,
                "camera intensity sample {index} is not finite normalized data"
            ),
            Self::IntensityDecreases(index) => {
                write!(f, "camera intensity sample {index} decreases")
            }
            Self::IrradianceEndpoints => {
                f.write_str("camera irradiance must start at zero and end at one")
            }
            Self::IntensityEndpoints => {
                f.write_str("camera intensity must start at zero and end at one")
            }
        }
    }
}

impl std::error::Error for CameraResponseError {}

fn sanitize_component(component: f32) -> f32 {
    if component.is_nan() || component <= 0.0 {
        0.0
    } else {
        component.min(f32::MAX)
    }
}

#[inline]
fn display_component_f32(component: f32) -> f32 {
    if component.is_nan() || component <= 0.0 {
        0.0
    } else if component >= 1.0 {
        1.0
    } else {
        component
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "a finite display-linear component is clamped to the f32 unit interval before casting"
)]
fn display_component(component: f64) -> f32 {
    if component.is_nan() || component <= 0.0 {
        0.0
    } else if component >= 1.0 {
        1.0
    } else {
        component as f32
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "finite nonnegative luminance is saturated to the public f32 scalar range"
)]
fn nonnegative_f64_to_f32(value: f64) -> f32 {
    debug_assert!(value.is_finite());
    debug_assert!(value >= 0.0);
    value.min(f64::from(f32::MAX)) as f32
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    struct DefaultBatchMapper;

    impl ToneMapper for DefaultBatchMapper {
        fn map(&self, color: LinearRGB) -> LinearRGB {
            Reinhard.map(color)
        }
    }

    fn assert_approximately_equal(actual: f32, expected: f32) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }

    fn assert_components_close(actual: LinearRGB, expected: [f32; 3]) {
        for (actual, expected) in actual.components().into_iter().zip(expected) {
            assert_approximately_equal(actual, expected);
        }
    }

    fn assert_displayable(color: LinearRGB) {
        assert!(
            color
                .components()
                .into_iter()
                .all(|component| component.is_finite() && (0.0..=1.0).contains(&component))
        );
    }

    fn batch_inputs() -> Vec<LinearRGB> {
        let smallest_positive = f32::from_bits(1);
        let below_one = f32::from_bits(1.0_f32.to_bits() - 1);
        let above_one = f32::from_bits(1.0_f32.to_bits() + 1);
        let palette = [
            [0.0, smallest_positive, below_one],
            [above_one, 0.18, 4.0],
            [f32::MAX, 100_000.0, 0.25],
            [4.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [32.0, 0.5, 8.0],
            [0.01, 0.02, 0.03],
        ];

        (0..17)
            .map(|index| LinearRGB::new(palette[index % palette.len()]))
            .collect()
    }

    fn assert_batch_matches_scalar(mapper: &dyn ToneMapper, inputs: &[LinearRGB]) {
        let expected: Vec<_> = inputs
            .iter()
            .copied()
            .map(|color| mapper.map(color))
            .collect();
        let mut actual = inputs.to_vec();
        mapper.map_in_place(&mut actual);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_eq!(
                actual.components().map(f32::to_bits),
                expected.components().map(f32::to_bits)
            );
        }
    }

    fn estimate_max_cll_scalarly(colors: &[LinearRGB]) -> MaxCll {
        let mut estimator = MaxCllEstimator::new(
            NonZeroUsize::new(colors.len()).expect("test MaxCLL input is nonempty"),
        );
        for color in colors {
            estimator.observe(*color);
        }
        estimator
            .finish()
            .expect("the scalar test observes every declared pixel")
    }

    #[test]
    fn linear_rgb_preserves_every_finite_nonnegative_component() {
        let color = LinearRGB::new([70_000.0, 100_000.0, f32::INFINITY]);

        assert_eq!(color.components(), [70_000.0, 100_000.0, f32::MAX]);
        assert!(color.luminance().is_finite());
    }

    #[test]
    fn selectable_method_labels_are_unique_and_default_to_extended_reinhard() {
        let labels: HashSet<_> = ToneMappingMethod::ALL
            .into_iter()
            .map(ToneMappingMethod::label)
            .collect();

        assert_eq!(labels.len(), ToneMappingMethod::ALL.len());
        assert!(labels.iter().all(|label| !label.is_empty()));
        assert_eq!(
            ToneMappingMethod::default(),
            ToneMappingMethod::ExtendedReinhard
        );
        assert_eq!(
            ToneMappingMethod::default().to_string(),
            "Extended Reinhard"
        );
    }

    #[test]
    fn clamp_discards_values_outside_the_display_range() {
        let color = Clamp.map(LinearRGB::new([0.25, 1.0, 4.0]));

        assert_eq!(color.components(), [0.25, 1.0, 1.0]);
    }

    #[test]
    fn bulk_mapping_matches_scalar_mapping_at_simd_boundaries() {
        let inputs = batch_inputs();
        let extended = ExtendedReinhard::new(WhitePoint::new(4.0).unwrap());
        let mappers: [&dyn ToneMapper; 3] = [&Clamp, &extended, &AcesFitted];

        for mapper in mappers {
            for length in [0, 1, 3, 4, 5, 7, 8, 9, 17] {
                assert_batch_matches_scalar(mapper, &inputs[..length]);
            }
        }
    }

    #[test]
    fn bulk_extended_reinhard_preserves_extreme_f64_behavior() {
        let inputs = batch_inputs();
        for white_point in [f32::from_bits(1), 1.0, 4.0, f32::MAX] {
            let mapper = ExtendedReinhard::new(WhitePoint::new(white_point).unwrap());
            assert_batch_matches_scalar(&mapper, &inputs);
        }
    }

    #[test]
    fn tone_mapper_bulk_default_is_object_safe() {
        let mapper: &dyn ToneMapper = &DefaultBatchMapper;
        assert_batch_matches_scalar(mapper, &batch_inputs());
    }

    #[test]
    fn scaled_clamp_maps_its_white_point_to_one() {
        let mapper = ScaledClamp::new(WhitePoint::new(4.0).unwrap());
        let color = mapper.map(LinearRGB::new([1.0, 4.0, 8.0]));

        assert_eq!(color.components(), [0.25, 1.0, 1.0]);
        assert_eq!(mapper.white_point().level(), 4.0);
    }

    #[test]
    fn reinhard_maps_reference_white_to_middle_gray() {
        let color = Reinhard.map(LinearRGB::new([0.0, 0.18, 1.0]));
        let [black, middle_gray, reference_white] = color.components();

        assert_eq!(black, 0.0);
        assert_approximately_equal(middle_gray, 0.152_542_37);
        assert_eq!(reference_white, 0.5);
    }

    #[test]
    fn extended_reinhard_maps_the_percentile_white_point_to_one() {
        let colors = [LinearRGB::new([4.0, 2.0, 1.0])];
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
        let black = estimate_max_cll(&[LinearRGB::default()]).unwrap();

        assert!(ExtendedReinhard::from_max_cll(black).is_none());
    }

    #[test]
    fn tone_mappers_remain_finite_at_extreme_hdr_levels() {
        let smallest_positive = f32::from_bits(1);
        let tiny = WhitePoint::new(smallest_positive).unwrap();
        let tiny_luminance = LuminanceWhitePoint::new(smallest_positive).unwrap();
        let extended = ExtendedReinhard::new(tiny);
        let extended_luminance = ExtendedLuminanceReinhard::new(tiny_luminance);
        let camera_response = CameraResponse::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let camera = camera_response.tone_mapper(tiny);
        let mappers: [&dyn ToneMapper; 11] = [
            &Clamp,
            &ScaledClamp::new(tiny),
            &Reinhard,
            &extended,
            &LuminanceReinhard,
            &extended_luminance,
            &ReinhardJodie,
            &Hable,
            &AcesFitted,
            &AcesApproximate,
            &camera,
        ];

        for mapper in mappers {
            assert_displayable(mapper.map(LinearRGB::new([0.0, f32::MAX, 100_000.0])));
            assert_eq!(mapper.map(LinearRGB::default()), LinearRGB::default());
        }
    }

    #[test]
    fn luminance_reinhard_preserves_unclipped_color_ratios() {
        let input = LinearRGB::new([1.0, 0.5, 0.25]);
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
        let [red, green, blue] = mapper.map(LinearRGB::new([1.0, 0.5, 0.25])).components();

        assert_approximately_equal(red, 0.652_772_3);
        assert_approximately_equal(green, 0.326_386_15);
        assert_approximately_equal(blue, 0.163_193_08);
        assert_eq!(mapper.white_point().luminance(), 4.0);
    }

    #[test]
    fn luminance_white_point_rejects_the_brightest_outlier() {
        let mut colors = vec![LinearRGB::new([1.0; 3]); 9_998];
        colors.extend([LinearRGB::new([4.0; 3]), LinearRGB::new([126.0; 3])]);

        let white_point = estimate_luminance_white_point(&colors).unwrap();

        assert_approximately_equal(white_point.luminance(), 4.0);
    }

    #[test]
    fn reinhard_jodie_blends_luminance_and_component_curves() {
        let [red, green, blue] = ReinhardJodie
            .map(LinearRGB::new([1.0, 0.5, 0.25]))
            .components();

        assert_approximately_equal(red, 0.564_811_9);
        assert_approximately_equal(green, 0.320_985_7);
        assert_approximately_equal(blue, 0.165_924_76);
        assert_eq!(
            ReinhardJodie.map(LinearRGB::new([1.0; 3])),
            Reinhard.map(LinearRGB::new([1.0; 3]))
        );
    }

    #[test]
    fn uncharted_two_uses_the_article_exposure_and_white_scale() {
        let [middle_gray, reference_white, filmic_white] = Hable
            .map(LinearRGB::new([0.18, 1.0, 5.6]))
            .components();

        assert_approximately_equal(middle_gray, 0.128_338_45);
        assert_approximately_equal(reference_white, 0.492_918_55);
        assert_eq!(filmic_white, 1.0);
    }

    #[test]
    fn aces_fitted_uses_the_reference_matrix_orientation() {
        assert_components_close(
            AcesFitted.map(LinearRGB::new([1.0, 0.0, 0.0])),
            [0.688_027_86, 0.0, 0.002_639_006_8],
        );
        assert_components_close(
            AcesFitted.map(LinearRGB::new([0.0, 1.0, 0.0])),
            [0.101_613_28, 0.623_659, 0.028_843_72],
        );
        assert_components_close(
            AcesFitted.map(LinearRGB::new([0.0, 0.0, 1.0])),
            [0.0, 0.0, 0.601_758_84],
        );
    }

    #[test]
    fn aces_approximate_includes_the_article_pre_exposure() {
        let [middle_gray, reference_white, clipped] = AcesApproximate
            .map(LinearRGB::new([0.18, 1.0, 100.0]))
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
            mapper.map(LinearRGB::new([0.0, 1.5, 3.0])),
            [0.0, 0.125, 0.25],
        );
        assert_components_close(
            mapper.map(LinearRGB::new([3.0, 6.0, 9.0])),
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
        
        assert_eq!(
            CameraResponse::new(vec![0.1, 1.0], vec![0.0, 1.0]).unwrap_err(),
            CameraResponseError::IrradianceEndpoints
        );
        
        assert_eq!(
            CameraResponse::new(vec![0.0, 1.0], vec![0.1, 1.0]).unwrap_err(),
            CameraResponseError::IntensityEndpoints
        );
    }

    #[test]
    fn max_cll_rejects_the_brightest_point_zero_one_percent() {
        let mut colors = vec![LinearRGB::new([1.0; 3]); 9_998];
        colors.extend([LinearRGB::new([4.0; 3]), LinearRGB::new([126.0; 3])]);

        let max_cll = estimate_max_cll(&colors).unwrap();

        assert_eq!(max_cll.level(), 4.0);
        assert_eq!(max_cll.channel(), ColorChannel::Red);
        assert_eq!(max_cll.white_point().map(WhitePoint::level), Some(4.0));
    }

    #[test]
    fn max_cll_mode_selects_percentile_or_true_maximum() {
        let mut colors = vec![LinearRGB::new([1.0; 3]); 9_998];
        colors.extend([
            LinearRGB::new([4.0, 0.0, 0.0]),
            LinearRGB::new([0.0, 0.0, 126.0]),
        ]);
        
        let pixel_count = NonZeroUsize::new(colors.len()).expect("test MaxCLL input is nonempty");

        let mut percentile = MaxCllEstimator::with_mode(pixel_count, MaxCllMode::Percentile99_99);
        percentile.observe_many(&colors);
        
        let percentile = percentile
            .finish()
            .expect("the percentile estimator observes every declared pixel");

        let mut true_maximum = MaxCllEstimator::with_mode(pixel_count, MaxCllMode::TrueMaximum);
        true_maximum.observe_many(&colors);
        
        let true_maximum = true_maximum
            .finish()
            .expect("the maximum estimator observes every declared pixel");

        assert_eq!(MaxCllMode::default(), MaxCllMode::Percentile99_99);
        assert_eq!(percentile.level(), 4.0);
        assert_eq!(percentile.channel(), ColorChannel::Red);
        assert_eq!(true_maximum.level(), 126.0);
        assert_eq!(true_maximum.channel(), ColorChannel::Blue);
    }

    #[test]
    fn max_cll_uses_max_rgb_instead_of_luminance() {
        let max_cll = estimate_max_cll(&[LinearRGB::new([0.0, 0.0, 5.0])]).unwrap();

        assert_eq!(max_cll.level(), 5.0);
        assert_eq!(max_cll.channel(), ColorChannel::Blue);
    }

    #[test]
    fn max_cll_preserves_float32_levels_above_binary16_range() {
        let max_cll = estimate_max_cll(&[LinearRGB::new([70_000.0, 0.0, 100_000.0])]).unwrap();

        assert_eq!(max_cll.level(), 100_000.0);
        assert_eq!(max_cll.channel(), ColorChannel::Blue);
    }

    #[test]
    fn max_cll_reports_incomplete_pixel_streams() {
        let mut estimator = MaxCllEstimator::new(NonZeroUsize::new(2).unwrap());
        estimator.observe(LinearRGB::new([1.0; 3]));

        let error = estimator.finish().unwrap_err();

        assert_eq!(error.expected(), 2);
        assert_eq!(error.observed(), 1);
        assert_eq!(
            error.to_string(),
            "MaxCLL estimator expected 2 pixels but observed 1"
        );
    }

    #[test]
    fn max_cll_batches_match_scalar_streams_at_percentile_boundaries() {
        let palette = [
            LinearRGB::new([1.0, 0.0, 0.0]),
            LinearRGB::new([0.0, 2.0, 0.0]),
            LinearRGB::new([0.0, 0.0, 3.0]),
            LinearRGB::new([4.0, 1.0, 2.0]),
            LinearRGB::new([0.25, 0.5, 0.75]),
        ];

        for pixel_count in [9_999, 10_000, 10_001] {
            let colors: Vec<_> = (0..pixel_count)
                .map(|index| palette[index % palette.len()])
                .collect();
            
            let expected = estimate_max_cll_scalarly(&colors);
            
            let mut estimator = MaxCllEstimator::new(
                NonZeroUsize::new(colors.len()).expect("test MaxCLL input is nonempty"),
            );
            
            for batch in colors.chunks(257) {
                estimator.observe_many(batch);
            }

            assert_eq!(
                estimator
                    .finish()
                    .expect("the batch test observes every declared pixel"),
                expected
            );
        }
    }

    #[test]
    fn max_cll_batch_retains_channel_order_for_equal_levels() {
        let colors = [
            LinearRGB::new([5.0, 0.0, 0.0]),
            LinearRGB::new([0.0, 5.0, 0.0]),
            LinearRGB::new([0.0, 0.0, 5.0]),
            LinearRGB::new([1.0; 3]),
            LinearRGB::new([2.0; 3]),
            LinearRGB::new([3.0; 3]),
            LinearRGB::new([4.0; 3]),
            LinearRGB::default(),
            LinearRGB::new([0.5; 3]),
        ];
        
        let mut estimator =
            MaxCllEstimator::new(NonZeroUsize::new(colors.len()).expect("test input is nonempty"));
        estimator.observe_many(&colors);

        let max_cll = estimator
            .finish()
            .expect("the batch test observes every declared pixel");
        
        assert_eq!(max_cll.level(), 5.0);
        assert_eq!(max_cll.channel(), ColorChannel::Blue);
    }

    #[test]
    fn max_cll_batch_counts_lanes_filtered_below_the_threshold() {
        let mut colors = vec![LinearRGB::new([10.0, 0.0, 0.0])];
        colors.extend(std::iter::repeat_n(LinearRGB::new([1.0; 3]), 8));
        
        let mut estimator =
            MaxCllEstimator::new(NonZeroUsize::new(colors.len()).expect("test input is nonempty"));
        estimator.observe_many(&colors);

        let max_cll = estimator
            .finish()
            .expect("filtered lanes still count as observations");
        
        assert_eq!(max_cll.level(), 10.0);
        assert_eq!(max_cll.channel(), ColorChannel::Red);
    }
}
