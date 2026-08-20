use multiversion::multiversion;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::simd::{Select, cmp::SimdPartialOrd, num::SimdFloat};

use super::{LinearRGB, LinearRGBPlanes, MaxCll, OrderedLevel, ToneMapper, WhitePoint};
use crate::simd::{COLOR_LANES, F32x8, map_colors, map_planes};

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
/// Components at the white point map to one, while brighter components clip at the display
/// boundary. For a still image, [`MaxCll`] supplies a `max(R, G, B)` white point. This is a
/// component-wise adaptation of the global operator in [Reinhard et al.].
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
        let scale = 1.0 / (1.0 + color.luminance());
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
        let luminance = F32x8::splat(super::REC709_LUMINANCE[0]) * components[0]
            + F32x8::splat(super::REC709_LUMINANCE[1]) * components[1]
            + F32x8::splat(super::REC709_LUMINANCE[2]) * components[2];

        let scale = one / (one + luminance);

        components.map(|component| component * scale)
    });
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
/// This adapts [Smith and Zink]'s per-frame `MaxCLL` outlier percentile to Rec. 709 luminance. It is
/// an analogous statistic for luminance-based curves, not `MaxCLL`.
///
/// [Smith and Zink]: https://doi.org/10.5594/JMI.2021.3090176
#[must_use]
pub fn estimate_luminance_white_point(colors: &[LinearRGB]) -> Option<LuminanceWhitePoint> {
    if colors.is_empty() {
        return None;
    }

    let mut estimator = LuminanceWhitePointEstimator::new(colors.len());
    for color in colors {
        estimator.observe(*color);
    }
    estimator.finish()
}

/// Estimates a p99.99 luminance white point from a stream of colors.
///
/// The slice form above materializes every color first. A decoder that visits pixels once, as the
/// JPEG XR HDR analysis pass does, needs to feed them in as it goes; it previously carried its own
/// copy of this heap, without the parallel-merge support below.
#[derive(Debug)]
pub struct LuminanceWhitePointEstimator {
    retained: usize,
    luminances: BinaryHeap<Reverse<OrderedLevel>>,
}

impl LuminanceWhitePointEstimator {
    /// Creates an estimator sized for `pixel_count` observations.
    ///
    /// Retains the brightest `floor(pixel_count / 10_000) + 1` luminances, which bounds memory
    /// while producing the same answer as sorting them all.
    #[must_use]
    pub fn new(pixel_count: usize) -> Self {
        let retained = pixel_count / 10_000 + 1;

        Self {
            retained,
            luminances: BinaryHeap::with_capacity(retained),
        }
    }

    /// Includes one color in the estimate.
    #[inline]
    pub fn observe(&mut self, color: LinearRGB) {
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
    /// Both estimators must have been created with the same pixel count, so that they retain the
    /// same number of samples. This is what lets a caller split an image across worker threads.
    pub fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.retained, other.retained);

        for luminance in other.luminances {
            self.retain(luminance.0);
        }
    }

    /// Returns the estimated white point, or `None` if nothing was observed.
    #[must_use]
    pub fn finish(self) -> Option<LuminanceWhitePoint> {
        self.luminances
            .peek()
            .and_then(|luminance| LuminanceWhitePoint::new(luminance.0.0))
    }
}

/// Applies extended Reinhard to luminance while retaining color ratios.
///
/// This follows the global white-point operator described by [Reinhard et al.].
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
        let luminance = F32x8::splat(super::REC709_LUMINANCE[0]) * components[0]
            + F32x8::splat(super::REC709_LUMINANCE[1]) * components[1]
            + F32x8::splat(super::REC709_LUMINANCE[2]) * components[2];

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
        let luminance_scale = 1.0 / (1.0 + color.luminance());
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
        let luminance = F32x8::splat(super::REC709_LUMINANCE[0]) * components[0]
            + F32x8::splat(super::REC709_LUMINANCE[1]) * components[1]
            + F32x8::splat(super::REC709_LUMINANCE[2]) * components[2];

        let luminance_scale = one / (one + luminance);
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
/// Scalar evaluation permits algebraic floating-point optimizations, so results are approximate
/// and may vary slightly across builds and targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mobius {
    white_point: LuminanceWhitePoint,
    transition: f32,
}

impl Mobius {
    /// Creates an operator with the supplied white point and `transition`.
    #[must_use]
    pub const fn new(white_point: LuminanceWhitePoint, transition: f32) -> Self {
        Self {
            white_point,
            transition,
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

impl ToneMapper for Mobius {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let components = color.components();
        let signal = components.iter().copied().fold(1e-6, f32::max);
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
        .algebraic_div(peak.algebraic_sub(1.0).max(1e-6));
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
            .simd_max(F32x8::splat(1e-6));

        let curved = curve_scale * (signal + a) / (signal + b);
        let mapped = signal.simd_le(transition).select(signal, curved);
        let scale = mapped / signal;

        components.map(|component| component * scale)
    });
}
