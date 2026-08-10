use multiversion::multiversion;
use std::simd::{Select, Simd, cmp::SimdPartialOrd, num::SimdFloat};

use super::{LinearRGB, ToneMapper};

const TONE_MAPPING_LANES: usize = 4;
type F64x4 = Simd<f64, TONE_MAPPING_LANES>;

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
