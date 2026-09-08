use multiversion::multiversion;
use std::backtrace::Backtrace;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::simd::{Select, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};

use super::{
    LinearRGB, LinearRGBPlanes, MaxCLL, OrderedLevel, ToneMapper, WhitePoint, WhitePointError,
};
use crate::math::recip;
use crate::simd::{COLOR_LANES, F32x8, map_colors, map_planes};
use thiserror::Error;

/// Applies the simple Reinhard curve independently to each component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reinhard;

impl ToneMapper for Reinhard {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        LinearRGB::displayable(
            color
                .components()
                .map(|component| component / (1.0 + component)),
        )
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, reinhard_batch, self);
    }
}

#[multiversion(targets = "simd")]
fn reinhard_batch(colors: &mut LinearRGBPlanes) {
    let one = F32x8::splat(1.0);
    map_colors(colors, |components| {
        components.map(|component| component / (one + component))
    });
}

/// Applies the white-point Reinhard curve independently to each component.
///
/// Components at the white point map to one; brighter components clip at the display boundary.
/// [`MaxCLL`] supplies a `max(R, G, B)` white point for still images.
///
/// [Reinhard et al.]: https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf
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
    pub fn from_max_cll(max_cll: MaxCLL) -> Option<Self> {
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
        let white_squared = self.white_point.level().powi(2);
        extended_reinhard(color, white_squared)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        let white_squared = self.white_point.level().powi(2);
        for color in colors {
            *color = extended_reinhard(*color, white_squared);
        }
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let white_squared = self.white_point.level().powi(2);
        map_planes(
            colors,
            COLOR_LANES,
            |colors| extended_reinhard_batch(colors, white_squared),
            self,
        );
    }
}

#[multiversion(targets = "simd")]
fn extended_reinhard_batch(colors: &mut LinearRGBPlanes, white_squared: f32) {
    let one = F32x8::splat(1.0);
    let white_squared = F32x8::splat(white_squared);
    map_colors(colors, |components| {
        components
            .map(|component| component * (one + component / white_squared) / (one + component))
    });
}

#[inline]
fn extended_reinhard(color: LinearRGB, white_squared: f32) -> LinearRGB {
    LinearRGB::displayable(
        color
            .components()
            .map(|component| component * (1.0 + component / white_squared) / (1.0 + component)),
    )
}

/// Applies the simple Reinhard curve to luminance while retaining color ratios.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LuminanceReinhard;

impl ToneMapper for LuminanceReinhard {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let scale = recip(F32x8::splat(1.0 + color.luminance()))[0];
        LinearRGB::displayable(color.components().map(|component| component * scale))
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, luminance_reinhard_batch, self);
    }
}

#[multiversion(targets = "simd")]
fn luminance_reinhard_batch(colors: &mut LinearRGBPlanes) {
    let one = F32x8::splat(1.0);

    map_colors(colors, |components| {
        let luminance = components[2].mul_add(
            F32x8::splat(super::REC709_LUMINANCE[2]),
            components[1].mul_add(
                F32x8::splat(super::REC709_LUMINANCE[1]),
                components[0] * F32x8::splat(super::REC709_LUMINANCE[0]),
            ),
        );

        let scale = recip(one + luminance);

        components.map(|component| component * scale)
    });
}

/// Identifies a positive finite luminance that maps to display white.
///
/// Unlike [`MaxCLL`], this value is based on luminance rather than `max(R, G, B)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LuminanceWhitePoint(WhitePoint);

impl LuminanceWhitePoint {
    /// Creates a white point from a positive finite luminance.
    ///
    /// # Errors
    ///
    /// Returns [`WhitePointError`] when `luminance` is zero, negative, or non-finite.
    pub fn new(luminance: f32) -> Result<Self, WhitePointError> {
        WhitePoint::new(luminance).map(Self)
    }

    /// Returns the linear luminance represented by this white point.
    #[must_use]
    pub fn luminance(self) -> f32 {
        self.0.level()
    }
}

impl TryFrom<f32> for LuminanceWhitePoint {
    type Error = WhitePointError;

    fn try_from(luminance: f32) -> Result<Self, Self::Error> {
        Self::new(luminance)
    }
}

/// Estimates a p99.99 luminance white point for a complete still image.
///
/// Adapts [Smith and Zink]'s per-frame `MaxCLL` percentile to Rec. 709 luminance.
///
/// [Smith and Zink]: https://doi.org/10.5594/JMI.2021.3090176
#[must_use]
#[expect(
    clippy::missing_panics_doc,
    reason = "the estimator observes the exact slice length declared above"
)]
pub fn estimate_luminance_white_point(colors: &[LinearRGB]) -> Option<LuminanceWhitePoint> {
    let pixel_count = NonZeroUsize::new(colors.len())?;
    let mut estimator = LuminanceWhitePointEstimator::new(pixel_count);
    for color in colors {
        estimator.observe(*color);
    }

    estimator
        .finish()
        .expect("every declared color was observed above")
}

/// Estimates a p99.99 luminance white point from a stream of colors.
#[derive(Debug)]
pub struct LuminanceWhitePointEstimator {
    expected: NonZeroUsize,
    retained: usize,
    observed: usize,
    luminances: BinaryHeap<Reverse<OrderedLevel>>,
}

impl LuminanceWhitePointEstimator {
    /// Creates an estimator sized for `pixel_count` observations.
    ///
    /// Retains the brightest `floor(pixel_count / 10_000) + 1` luminances.
    #[must_use]
    pub fn new(pixel_count: NonZeroUsize) -> Self {
        let retained = pixel_count.get() / 10_000 + 1;

        Self {
            expected: pixel_count,
            retained,
            observed: 0,
            luminances: BinaryHeap::with_capacity(retained),
        }
    }

    /// Includes one color in the estimate.
    #[inline]
    pub fn observe(&mut self, color: LinearRGB) {
        self.observed = self.observed.saturating_add(1);
        self.retain(OrderedLevel(color.luminance()));
    }

    #[inline]
    fn retain(&mut self, luminance: OrderedLevel) {
        if self.luminances.len() < self.retained {
            self.luminances.push(Reverse(luminance));
        } else if let Some(mut threshold) = self.luminances.peek_mut()
            && luminance > threshold.0
        {
            *threshold = Reverse(luminance);
        }
    }

    /// Folds `other` into this estimate.
    ///
    /// The estimators must use the same pixel count.
    ///
    /// # Panics
    ///
    /// Panics when the estimators declare different pixel counts or retained sample counts.
    pub fn merge(&mut self, other: Self) {
        assert_eq!(
            self.expected, other.expected,
            "merged luminance white point estimators must share a declared pixel count: {} vs {}",
            self.expected, other.expected
        );

        assert_eq!(
            self.retained, other.retained,
            "merged luminance white point estimators must retain the same sample count: {} vs {}",
            self.retained, other.retained
        );

        self.observed = self.observed.saturating_add(other.observed);
        for luminance in other.luminances {
            self.retain(luminance.0);
        }
    }

    /// Finishes the estimate after the declared number of observations.
    ///
    /// Returns `Ok(None)` when every observed color mapped to zero luminance, since no positive
    /// white point exists to report.
    ///
    /// # Errors
    ///
    /// Returns [`LuminanceWhitePointCountError`] when the observed pixel count differs from the
    /// count passed when constructing the estimator.
    pub fn finish(self) -> Result<Option<LuminanceWhitePoint>, LuminanceWhitePointCountError> {
        if self.observed != self.expected.get() {
            return Err(LuminanceWhitePointCountError {
                expected: self.expected.get(),
                observed: self.observed,
                backtrace: Backtrace::capture(),
            });
        }

        Ok(self
            .luminances
            .peek()
            .and_then(|luminance| LuminanceWhitePoint::new(luminance.0.0).ok()))
    }
}

/// Reports a mismatch between declared and observed luminance white point pixel counts.
#[derive(Debug, Error)]
#[error("luminance white point estimator expected {expected} pixels but observed {observed}")]
pub struct LuminanceWhitePointCountError {
    expected: usize,
    observed: usize,
    backtrace: Backtrace,
}

impl LuminanceWhitePointCountError {
    /// Returns the pixel count declared when the estimator was created.
    #[must_use]
    pub const fn expected(&self) -> usize {
        self.expected
    }

    /// Returns the number of colors passed to the estimator.
    #[must_use]
    pub const fn observed(&self) -> usize {
        self.observed
    }

    /// Returns the backtrace captured when the mismatch was detected.
    #[must_use]
    pub const fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

/// Applies extended Reinhard to luminance while retaining color ratios.
///
/// Uses the global white-point operator described by [Reinhard et al.].
///
/// [Reinhard et al.]: https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf
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
        let luminance = color.luminance();
        let white_squared = self.white_point.luminance().powi(2);
        let scale = (1.0 + luminance / white_squared) / (1.0 + luminance);
        LinearRGB::displayable(color.components().map(|component| component * scale))
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let white_squared = self.white_point.luminance().powi(2);
        map_planes(
            colors,
            COLOR_LANES,
            |colors| extended_luminance_reinhard_batch(colors, white_squared),
            self,
        );
    }
}

#[multiversion(targets = "simd")]
fn extended_luminance_reinhard_batch(colors: &mut LinearRGBPlanes, white_squared: f32) {
    let one = F32x8::splat(1.0);
    let white_squared = F32x8::splat(white_squared);

    map_colors(colors, |components| {
        let luminance = components[2].mul_add(
            F32x8::splat(super::REC709_LUMINANCE[2]),
            components[1].mul_add(
                F32x8::splat(super::REC709_LUMINANCE[1]),
                components[0] * F32x8::splat(super::REC709_LUMINANCE[0]),
            ),
        );

        let scale = (one + luminance / white_squared) / (one + luminance);

        components.map(|component| component * scale)
    });
}

/// Blends component-wise and luminance-based Reinhard results per component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReinhardJodie;

impl ToneMapper for ReinhardJodie {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let components = color.components();
        let luminance_scale = recip(F32x8::splat(1.0 + color.luminance()))[0];
        let component_mapped = components.map(|component| component / (1.0 + component));
        let luminance_mapped = components.map(|component| component * luminance_scale);

        let blended = std::array::from_fn(|index| {
            let weight = component_mapped[index];
            luminance_mapped[index] * (1.0 - weight) + component_mapped[index] * weight
        });

        LinearRGB::displayable(blended)
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, reinhard_jodie_batch, self);
    }
}

#[multiversion(targets = "simd")]
fn reinhard_jodie_batch(colors: &mut LinearRGBPlanes) {
    let one = F32x8::splat(1.0);

    map_colors(colors, |components| {
        let luminance = components[2].mul_add(
            F32x8::splat(super::REC709_LUMINANCE[2]),
            components[1].mul_add(
                F32x8::splat(super::REC709_LUMINANCE[1]),
                components[0] * F32x8::splat(super::REC709_LUMINANCE[0]),
            ),
        );

        let luminance_scale = recip(one + luminance);
        let component_mapped = components.map(|component| component / (one + component));
        let luminance_mapped = components.map(|component| component * luminance_scale);

        std::array::from_fn(|index| {
            let weight = component_mapped[index];
            luminance_mapped[index] * (one - weight) + component_mapped[index] * weight
        })
    });
}

/// Applies a generalized Reinhard based on the Mobius transform.
///
/// Scalar evaluation may vary slightly across builds and targets because of floating-point
/// reassociation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mobius {
    white_point: LuminanceWhitePoint,
    transition: f32,
}

impl Mobius {
    /// Creates an operator with the supplied white point and `transition`.
    ///
    /// `transition` is the unchanged scene level below `white_point`. It must be positive, finite,
    /// and below the white point's luminance. The white point must exceed display white (`1.0`).
    ///
    /// # Errors
    ///
    /// Returns [`MobiusError::WhitePointNotAboveDisplayWhite`] when `white_point`'s luminance does
    /// not exceed `1.0`. Returns [`MobiusError::InvalidTransition`] when `transition` is zero,
    /// negative, non-finite, or not less than `white_point`'s luminance.
    pub fn new(white_point: LuminanceWhitePoint, transition: f32) -> Result<Self, MobiusError> {
        if white_point.luminance() <= 1.0 {
            return Err(MobiusError::WhitePointNotAboveDisplayWhite(
                white_point.luminance(),
            ));
        }

        if transition.is_finite() && transition > 0.0 && transition < white_point.luminance() {
            Ok(Self {
                white_point,
                transition,
            })
        } else {
            Err(MobiusError::InvalidTransition {
                transition,
                white_point: white_point.luminance(),
            })
        }
    }

    /// Returns the luminance mapped to display white.
    #[must_use]
    pub const fn white_point(self) -> LuminanceWhitePoint {
        self.white_point
    }

    /// Returns the scene level below which input remains unchanged.
    #[must_use]
    pub const fn transition(self) -> f32 {
        self.transition
    }
}

/// Reports why a [`Mobius`] operator could not be constructed.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum MobiusError {
    /// The white point's luminance does not exceed display white (`1.0`).
    #[error("Mobius white point must exceed display white (1.0), got {0}")]
    WhitePointNotAboveDisplayWhite(f32),
    /// `transition` is zero, negative, non-finite, or not less than the white point's luminance.
    #[error(
        "Mobius transition must be positive, finite, and less than the white point ({white_point}), got {transition}"
    )]
    InvalidTransition {
        /// The rejected transition.
        transition: f32,
        /// The white point's luminance the transition was checked against.
        white_point: f32,
    },
}

impl ToneMapper for Mobius {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let components = color.components();
        let signal = components
            .iter()
            .copied()
            .fold(MOBIUS_SIGNAL_FLOOR, f32::max);

        let mapped = mobius_signal(signal, self.transition, self.white_point.luminance());
        let scale = mapped.algebraic_div(signal);

        LinearRGB::displayable(components.map(|component| component.algebraic_mul(scale)))
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(
            colors,
            COLOR_LANES,
            |colors| {
                mobius_batch(colors, self.transition, self.white_point.luminance());
            },
            self,
        );
    }
}

/// The minimum per-pixel signal fed into the Mobius curve.
///
/// Floor for the per-pixel signal used by the Mobius curve.
const MOBIUS_SIGNAL_FLOOR: f32 = 1e-6;

/// The minimum denominator when computing the Mobius curve's `b` coefficient.
///
/// Floor for the Mobius `b` coefficient denominator.
const MOBIUS_COEFFICIENT_FLOOR: f32 = 1e-6;

#[inline]
fn mobius_signal(signal: f32, transition: f32, peak: f32) -> f32 {
    if signal <= transition {
        return signal;
    }

    let (a, b, scale) = mobius_coefficients(transition, peak);

    scale
        .algebraic_mul(signal.algebraic_add(a))
        .algebraic_div(signal.algebraic_add(b))
}

#[inline]
fn mobius_coefficients(transition: f32, peak: f32) -> (f32, f32, f32) {
    let transition_squared = transition.algebraic_mul(transition);
    let doubled_transition = 2.0_f32.algebraic_mul(transition);
    let a = 0.0_f32
        .algebraic_sub(transition_squared)
        .algebraic_mul(peak.algebraic_sub(1.0))
        .algebraic_div(
            transition_squared
                .algebraic_sub(doubled_transition)
                .algebraic_add(peak),
        );
    let b = transition_squared
        .algebraic_sub(doubled_transition.algebraic_mul(peak))
        .algebraic_add(peak)
        .algebraic_div(peak.algebraic_sub(1.0).max(MOBIUS_COEFFICIENT_FLOOR));
    let b_plus_transition = b.algebraic_add(transition);
    let scale = b_plus_transition
        .algebraic_mul(b_plus_transition)
        .algebraic_div(b.algebraic_sub(a));

    (a, b, scale)
}

#[multiversion(targets = "simd")]
fn mobius_batch(colors: &mut LinearRGBPlanes, transition: f32, peak: f32) {
    let (a, b, scale) = mobius_coefficients(transition, peak);
    let transition = F32x8::splat(transition);
    let a = F32x8::splat(a);
    let b = F32x8::splat(b);
    let curve_scale = F32x8::splat(scale);

    map_colors(colors, |components| {
        let signal = components[0]
            .simd_max(components[1])
            .simd_max(components[2])
            .simd_max(F32x8::splat(MOBIUS_SIGNAL_FLOOR));

        let curved = curve_scale * (signal + a) / (signal + b);
        let mapped = signal.simd_le(transition).select(signal, curved);
        let scale = mapped / signal;

        components.map(|component| component * scale)
    });
}
