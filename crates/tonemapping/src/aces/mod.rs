use std::simd::StdFloat;

use multiversion::multiversion;

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};
use crate::simd::{COLOR_LANES, F32x8, displayable, map_colors, map_planes};

/// Converts linear sRGB into the fitted curve's working space.
///
/// From [Stephen Hill's fitted ACES reference and display transform].
///
/// [Stephen Hill's fitted ACES reference and display transform]: https://64.github.io/tonemapping/
const ACES_INPUT_MATRIX: [[f32; 3]; 3] = [
    [0.597_19, 0.354_58, 0.048_23],
    [0.076_00, 0.908_34, 0.015_66],
    [0.028_40, 0.133_83, 0.837_77],
];

/// Converts the fitted curve's working space back to linear sRGB.
///
/// From [Stephen Hill's fitted ACES reference and display transform].
///
/// [Stephen Hill's fitted ACES reference and display transform]: https://64.github.io/tonemapping/
const ACES_OUTPUT_MATRIX: [[f32; 3]; 3] = [
    [1.604_75, -0.531_08, -0.073_67],
    [-0.102_08, 1.108_13, -0.006_05],
    [-0.003_27, -0.072_76, 1.076_02],
];

/// Rational-fit coefficients for one channel of [Stephen Hill's fitted ACES curve].
///
/// [Stephen Hill's fitted ACES curve]: https://64.github.io/tonemapping/
const ACES_FIT_A: f32 = 0.024_578_6;
const ACES_FIT_B: f32 = 0.000_090_537;
const ACES_FIT_C: f32 = 0.983_729;
const ACES_FIT_D: f32 = 0.432_951;
const ACES_FIT_E: f32 = 0.238_081;

/// Applies [Stephen Hill's fitted ACES reference and display transform].
///
/// Uses the article's linear sRGB input and output matrices; not a complete ACES pipeline.
///
/// [Stephen Hill's fitted ACES reference and display transform]: https://64.github.io/tonemapping/
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ACESFitted;

impl ToneMapper for ACESFitted {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        aces_fitted(color)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        let simd_len = colors.len() / COLOR_LANES * COLOR_LANES;
        let (simd_colors, tail) = colors.split_at_mut(simd_len);

        if !simd_colors.is_empty() {
            aces_fitted_batch(simd_colors);
        }

        for color in tail {
            *color = aces_fitted(*color);
        }
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, aces_fitted_planes, self);
    }
}

#[inline]
fn aces_fitted(color: LinearRGB) -> LinearRGB {
    let transformed = multiply_rgb(ACES_INPUT_MATRIX, color.components());

    let fitted = transformed.map(|component| {
        let numerator = component.mul_add(component + ACES_FIT_A, -ACES_FIT_B);
        let denominator = component.mul_add(ACES_FIT_C.mul_add(component, ACES_FIT_D), ACES_FIT_E);

        numerator / denominator
    });

    LinearRGB::displayable(multiply_rgb(ACES_OUTPUT_MATRIX, fitted))
}

#[multiversion(targets = "simd")]
fn aces_fitted_batch(colors: &mut [LinearRGB]) {
    let (chunks, tail) = colors.as_chunks_mut::<COLOR_LANES>();
    debug_assert!(
        tail.is_empty(),
        "fitted ACES SIMD input must contain complete eight-pixel chunks"
    );

    for chunk in chunks {
        let color = [0, 1, 2]
            .map(|channel| F32x8::from_array(std::array::from_fn(|lane| chunk[lane].0[channel])));

        let transformed = ACES_INPUT_MATRIX.map(|row| {
            color[2].mul_add(
                F32x8::splat(row[2]),
                color[1].mul_add(F32x8::splat(row[1]), color[0] * F32x8::splat(row[0])),
            )
        });

        let fitted = transformed.map(|component| {
            let numerator = component.mul_add(
                component + F32x8::splat(ACES_FIT_A),
                -F32x8::splat(ACES_FIT_B),
            );

            let denominator = component.mul_add(
                F32x8::splat(ACES_FIT_C).mul_add(component, F32x8::splat(ACES_FIT_D)),
                F32x8::splat(ACES_FIT_E),
            );

            numerator / denominator
        });

        let mapped = ACES_OUTPUT_MATRIX.map(|row| {
            fitted[2].mul_add(
                F32x8::splat(row[2]),
                fitted[1].mul_add(F32x8::splat(row[1]), fitted[0] * F32x8::splat(row[0])),
            )
        });

        let mapped = mapped.map(displayable);

        for lane in 0..COLOR_LANES {
            chunk[lane] = LinearRGB([mapped[0][lane], mapped[1][lane], mapped[2][lane]]);
        }
    }
}

#[multiversion(targets = "simd")]
fn aces_fitted_planes(colors: &mut LinearRGBPlanes) {
    map_colors(colors, |color| {
        let transformed = ACES_INPUT_MATRIX.map(|row| {
            color[2].mul_add(
                F32x8::splat(row[2]),
                color[1].mul_add(F32x8::splat(row[1]), color[0] * F32x8::splat(row[0])),
            )
        });

        let fitted = transformed.map(|component| {
            let numerator = component.mul_add(
                component + F32x8::splat(ACES_FIT_A),
                -F32x8::splat(ACES_FIT_B),
            );

            let denominator = component.mul_add(
                F32x8::splat(ACES_FIT_C).mul_add(component, F32x8::splat(ACES_FIT_D)),
                F32x8::splat(ACES_FIT_E),
            );

            numerator / denominator
        });

        ACES_OUTPUT_MATRIX.map(|row| {
            fitted[2].mul_add(
                F32x8::splat(row[2]),
                fitted[1].mul_add(F32x8::splat(row[1]), fitted[0] * F32x8::splat(row[0])),
            )
        })
    });
}

fn multiply_rgb(matrix: [[f32; 3]; 3], color: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| color[2].mul_add(row[2], color[1].mul_add(row[1], color[0] * row[0])))
}

/// Applies [Krzysztof Narkowicz's scalar ACES approximation] component-wise.
///
/// The input is pre-exposed by `0.6`, matching the article's comparison with the fitted transform.
///
/// [Krzysztof Narkowicz's scalar ACES approximation]: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ACESApproximate;

impl ToneMapper for ACESApproximate {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        const A: f32 = 2.51;
        const B: f32 = 0.03;
        const C: f32 = 2.43;
        const D: f32 = 0.59;
        const E: f32 = 0.14;

        LinearRGB::displayable(color.components().map(|component| {
            let exposed = component * 0.6;
            exposed * A.mul_add(exposed, B) / exposed.mul_add(C.mul_add(exposed, D), E)
        }))
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, aces_approximate_planes, self);
    }
}

#[multiversion(targets = "simd")]
fn aces_approximate_planes(colors: &mut LinearRGBPlanes) {
    let a = F32x8::splat(2.51);
    let b = F32x8::splat(0.03);
    let c = F32x8::splat(2.43);
    let d = F32x8::splat(0.59);
    let e = F32x8::splat(0.14);
    let exposure = F32x8::splat(0.6);

    map_colors(colors, |components| {
        components.map(|component| {
            let exposed = component * exposure;
            exposed * a.mul_add(exposed, b) / exposed.mul_add(c.mul_add(exposed, d), e)
        })
    });
}
