use std::cmp::Reverse;
use std::collections::BinaryHeap;

use multiversion::multiversion;

use super::{LinearRGB, LinearRGBPlanes, MaxCll, OrderedLevel, ToneMapper, WhitePoint};
use crate::simd::{COLOR_LANES, F64x4, map_colors};

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

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        reinhard_batch(colors);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn reinhard_batch(colors: &mut LinearRGBPlanes) {
    let one = F64x4::splat(1.0);
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

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let white_squared = f64::from(self.white_point.level()).powi(2);
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        extended_reinhard_batch(colors, white_squared);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn extended_reinhard_batch(colors: &mut LinearRGBPlanes, white_squared: f64) {
    let one = F64x4::splat(1.0);
    let white_squared = F64x4::splat(white_squared);
    map_colors(colors, |components| {
        components
            .map(|component| component * (one + component / white_squared) / (one + component))
    });
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

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        luminance_reinhard_batch(colors);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn luminance_reinhard_batch(colors: &mut LinearRGBPlanes) {
    let one = F64x4::splat(1.0);

    map_colors(colors, |components| {
        let luminance = F64x4::splat(super::REC709_LUMINANCE[0]) * components[0]
            + F64x4::splat(super::REC709_LUMINANCE[1]) * components[1]
            + F64x4::splat(super::REC709_LUMINANCE[2]) * components[2];

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
/// This adapts [Smith and Zink's] per-frame MaxCLL outlier percentile to Rec. 709 luminance. It is
/// an analogous statistic for luminance-based curves, not `MaxCLL`.
///
/// [Smith and Zink's]: https://doi.org/10.5594/JMI.2021.3090176
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
        let luminance = color.luminance_f64();
        let white_squared = f64::from(self.white_point.luminance()).powi(2);
        let scale = (1.0 + luminance / white_squared) / (1.0 + luminance);
        LinearRGB::displayable(color.components_f64().map(|component| component * scale))
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let white_squared = f64::from(self.white_point.luminance()).powi(2);
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        extended_luminance_reinhard_batch(colors, white_squared);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn extended_luminance_reinhard_batch(colors: &mut LinearRGBPlanes, white_squared: f64) {
    let one = F64x4::splat(1.0);
    let white_squared = F64x4::splat(white_squared);

    map_colors(colors, |components| {
        let luminance = F64x4::splat(super::REC709_LUMINANCE[0]) * components[0]
            + F64x4::splat(super::REC709_LUMINANCE[1]) * components[1]
            + F64x4::splat(super::REC709_LUMINANCE[2]) * components[2];

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

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        reinhard_jodie_batch(colors);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn reinhard_jodie_batch(colors: &mut LinearRGBPlanes) {
    let one = F64x4::splat(1.0);

    map_colors(colors, |components| {
        let luminance = F64x4::splat(super::REC709_LUMINANCE[0]) * components[0]
            + F64x4::splat(super::REC709_LUMINANCE[1]) * components[1]
            + F64x4::splat(super::REC709_LUMINANCE[2]) * components[2];

        let luminance_scale = one / (one + luminance);
        let component_mapped = components.map(|component| component / (one + component));
        let luminance_mapped = components.map(|component| component * luminance_scale);

        std::array::from_fn(|index| {
            let weight = component_mapped[index];
            luminance_mapped[index] * (one - weight) + component_mapped[index] * weight
        })
    });
}
