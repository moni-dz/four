use multiversion::multiversion;

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};
use crate::simd::{COLOR_LANES, F64x4, map_colors};

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

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        hable_batch(colors);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn hable_batch(colors: &mut LinearRGBPlanes) {
    let exposure = F64x4::splat(2.0);
    let white_scale = F64x4::splat(1.0 / hable_partial(11.2));

    map_colors(colors, |components| {
        components.map(|component| hable_partial_simd(component * exposure) * white_scale)
    });
}

#[inline]
fn hable_partial_simd(value: F64x4) -> F64x4 {
    let a = F64x4::splat(0.15);
    let b = F64x4::splat(0.50);
    let c = F64x4::splat(0.10);
    let d = F64x4::splat(0.20);
    let e = F64x4::splat(0.02);
    let f = F64x4::splat(0.30);

    ((value * (a * value + c * b) + d * e) / (value * (a * value + b) + d * f)) - e / f
}

#[inline]
fn hable_partial(value: f64) -> f64 {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;

    ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - E / F
}
