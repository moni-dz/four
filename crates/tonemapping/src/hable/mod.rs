use multiversion::multiversion;

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};
use crate::simd::{COLOR_LANES, F32x8, map_colors, map_planes};

/// Applies [John Hable's Uncharted 2 filmic curve] component-wise.
///
/// The operator includes the article's exposure bias of two and normalizes the curve at its `11.2`
/// reference input. Consequently, a scene component of `5.6` maps to display white.
///
/// [John Hable's Uncharted 2 filmic curve]: https://filmicworlds.com/blog/filmic-tonemapping-operators/
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Hable;

impl ToneMapper for Hable {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let white_scale = 1.0 / hable_partial(11.2);
        LinearRGB::displayable(
            color
                .components()
                .map(|component| hable_partial(component * 2.0) * white_scale),
        )
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, hable_batch, self);
    }
}

#[multiversion(targets = "simd")]
fn hable_batch(colors: &mut LinearRGBPlanes) {
    let exposure = F32x8::splat(2.0);
    let white_scale = F32x8::splat(1.0 / hable_partial(11.2));

    map_colors(colors, |components| {
        components.map(|component| hable_partial_simd(component * exposure) * white_scale)
    });
}

// The article names the curve parameters A through F. Sharing them between the scalar and batch
// paths keeps the two expressions provably identical, which the parity tests assert bit-for-bit.
const A: f32 = 0.15;
const B: f32 = 0.50;
const C: f32 = 0.10;
const D: f32 = 0.20;
const E: f32 = 0.02;
const F: f32 = 0.30;

// The subtracted toe offset exists so the curve passes through the origin. Written as `E / F` it
// does not: at zero the curve evaluates `(D * E) / (D * F)`, which rounds to a different f32, and
// black would map to 1e-8 instead of to black. Spelling the offset the way the curve computes it
// makes the cancellation exact.
const TOE: f32 = (D * E) / (D * F);

#[inline]
fn hable_partial_simd(value: F32x8) -> F32x8 {
    // Mirrors `hable_partial` operation for operation. `C * B`, `D * E` and `D * F` are constant
    // folded there, so splatting the folded values keeps the two paths bit-identical.
    let shoulder_strength = F32x8::splat(A);
    let linear_strength = F32x8::splat(B);
    let linear_angle_strength = F32x8::splat(C * B);
    let toe_numerator = F32x8::splat(D * E);
    let toe_denominator = F32x8::splat(D * F);

    ((value * (shoulder_strength * value + linear_angle_strength) + toe_numerator)
        / (value * (shoulder_strength * value + linear_strength) + toe_denominator))
        - F32x8::splat(TOE)
}

#[inline]
fn hable_partial(value: f32) -> f32 {
    ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - TOE
}
