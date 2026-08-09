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
