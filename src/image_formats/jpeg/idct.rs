use multiversion::multiversion;
use std::simd::{
    Select, Simd, StdFloat,
    cmp::SimdPartialOrd,
    num::{SimdFloat, SimdInt, SimdUint},
};

use super::round_clamp_u8;

const BLOCK_SIDE: usize = 8;
// These literals carry the same 36 decimal places as Rust's f32 mathematical constants. The
// compiler still rounds each one to f32 once, but retaining the source precision makes that
// rounding reproducible and prevents hand-rounded matrix entries from becoming the definition.
const COSINE_PI_OVER_SIXTEEN: f32 = 0.980_785_280_403_230_449_126_182_236_134_239_036_f32;
const COSINE_PI_OVER_EIGHT: f32 = 0.923_879_532_511_286_756_128_183_189_396_788_286_f32;
const COSINE_THREE_PI_OVER_SIXTEEN: f32 = 0.831_469_612_302_545_237_078_788_377_617_905_756_f32;
const COSINE_FIVE_PI_OVER_SIXTEEN: f32 = 0.555_570_233_019_602_224_742_830_813_948_532_874_f32;
const COSINE_THREE_PI_OVER_EIGHT: f32 = 0.382_683_432_365_089_771_728_459_984_030_398_866_f32;
const COSINE_SEVEN_PI_OVER_SIXTEEN: f32 = 0.195_090_322_016_128_267_848_284_868_477_022_240_f32;
const LEVEL_SHIFT: f32 = 128.0;

type F32x8 = Simd<f32, BLOCK_SIDE>;

// C(u) * cos((2x + 1)u*pi/16), stored once so decoding does no trigonometry per block.
const BASIS: [[f32; BLOCK_SIDE]; BLOCK_SIDE] = [
    [
        std::f32::consts::FRAC_1_SQRT_2,
        COSINE_PI_OVER_SIXTEEN,
        COSINE_PI_OVER_EIGHT,
        COSINE_THREE_PI_OVER_SIXTEEN,
        std::f32::consts::FRAC_1_SQRT_2,
        COSINE_FIVE_PI_OVER_SIXTEEN,
        COSINE_THREE_PI_OVER_EIGHT,
        COSINE_SEVEN_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        COSINE_THREE_PI_OVER_SIXTEEN,
        COSINE_THREE_PI_OVER_EIGHT,
        -COSINE_SEVEN_PI_OVER_SIXTEEN,
        -std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_PI_OVER_SIXTEEN,
        -COSINE_PI_OVER_EIGHT,
        -COSINE_FIVE_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        COSINE_FIVE_PI_OVER_SIXTEEN,
        -COSINE_THREE_PI_OVER_EIGHT,
        -COSINE_PI_OVER_SIXTEEN,
        -std::f32::consts::FRAC_1_SQRT_2,
        COSINE_SEVEN_PI_OVER_SIXTEEN,
        COSINE_PI_OVER_EIGHT,
        COSINE_THREE_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        COSINE_SEVEN_PI_OVER_SIXTEEN,
        -COSINE_PI_OVER_EIGHT,
        -COSINE_FIVE_PI_OVER_SIXTEEN,
        std::f32::consts::FRAC_1_SQRT_2,
        COSINE_THREE_PI_OVER_SIXTEEN,
        -COSINE_THREE_PI_OVER_EIGHT,
        -COSINE_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_SEVEN_PI_OVER_SIXTEEN,
        -COSINE_PI_OVER_EIGHT,
        COSINE_FIVE_PI_OVER_SIXTEEN,
        std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_THREE_PI_OVER_SIXTEEN,
        -COSINE_THREE_PI_OVER_EIGHT,
        COSINE_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_FIVE_PI_OVER_SIXTEEN,
        -COSINE_THREE_PI_OVER_EIGHT,
        COSINE_PI_OVER_SIXTEEN,
        -std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_SEVEN_PI_OVER_SIXTEEN,
        COSINE_PI_OVER_EIGHT,
        -COSINE_THREE_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_THREE_PI_OVER_SIXTEEN,
        COSINE_THREE_PI_OVER_EIGHT,
        COSINE_SEVEN_PI_OVER_SIXTEEN,
        -std::f32::consts::FRAC_1_SQRT_2,
        COSINE_PI_OVER_SIXTEEN,
        -COSINE_PI_OVER_EIGHT,
        COSINE_FIVE_PI_OVER_SIXTEEN,
    ],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_PI_OVER_SIXTEEN,
        COSINE_PI_OVER_EIGHT,
        -COSINE_THREE_PI_OVER_SIXTEEN,
        std::f32::consts::FRAC_1_SQRT_2,
        -COSINE_FIVE_PI_OVER_SIXTEEN,
        COSINE_THREE_PI_OVER_EIGHT,
        -COSINE_SEVEN_PI_OVER_SIXTEEN,
    ],
];

/// Returns `matrix` with rows and columns exchanged.
const fn transpose(matrix: [[f32; BLOCK_SIDE]; BLOCK_SIDE]) -> [[f32; BLOCK_SIDE]; BLOCK_SIDE] {
    let mut transposed = [[0.0; BLOCK_SIDE]; BLOCK_SIDE];
    let mut row = 0;
    while row < BLOCK_SIDE {
        let mut column = 0;
        while column < BLOCK_SIDE {
            transposed[column][row] = matrix[row][column];
            column += 1;
        }
        row += 1;
    }
    transposed
}

// `BASIS` is indexed by sample position then frequency, which is the wrong order for accumulating
// across frequencies. `BASIS` is not symmetric, so its transpose is a distinct table.
const BASIS_TRANSPOSED: [[f32; BLOCK_SIDE]; BLOCK_SIDE] = transpose(BASIS);

#[expect(
    clippy::cast_precision_loss,
    reason = "the floating-point IDCT intentionally maps integer coefficients into f32 lanes"
)]
pub(super) fn inverse(coefficients: &[i32; 64]) -> [u8; 64] {
    invariant_eq!(coefficients.len(), BLOCK_SIDE * BLOCK_SIDE);
    invariant!(
        coefficients
            .iter()
            .all(|value| value.checked_abs().is_some())
    );

    if coefficients[1..].iter().all(|value| *value == 0) {
        let sample = round_clamp_u8(coefficients[0] as f32 / 8.0 + LEVEL_SHIFT);
        return [sample; 64];
    }

    inverse_simd(coefficients)
}

// Dispatch only non-flat blocks: a dispatch before the cheap DC-only path would slow down the most
// common case. Each target gets the same portable-SIMD algorithm, plus a safe baseline copy.
#[multiversion(targets = "simd")]
fn inverse_simd(coefficients: &[i32; 64]) -> [u8; 64] {
    invariant_eq!(coefficients.len(), BLOCK_SIDE * BLOCK_SIDE);
    invariant!(coefficients[1..].iter().any(|value| *value != 0));

    // Convert whole rows at once, so pass one broadcasts each coefficient straight from memory
    // instead of converting and broadcasting it one lane at a time.
    let mut converted = [[0.0_f32; BLOCK_SIDE]; BLOCK_SIDE];
    let (coefficient_rows, coefficient_remainder) = coefficients.as_chunks::<BLOCK_SIDE>();

    invariant_eq!(coefficient_remainder.len(), 0);
    for (converted, coefficients) in converted.iter_mut().zip(coefficient_rows) {
        *converted = Simd::<i32, BLOCK_SIDE>::from_array(*coefficients)
            .cast::<f32>()
            .to_array();
    }

    let rows = converted.map(|coefficients| {
        let mut values = F32x8::splat(0.0);

        for (coefficient, basis) in coefficients.into_iter().zip(BASIS_TRANSPOSED.iter()) {
            values = F32x8::from_array(*basis).mul_add(F32x8::splat(coefficient), values);
        }

        values
    });

    let scaled = |sample: usize, frequency: usize| F32x8::splat(BASIS[sample][frequency] * 0.25);
    let level_shift = F32x8::splat(LEVEL_SHIFT);

    let dc_sum = (rows[0] + rows[4]).mul_add(scaled(0, 0), level_shift);
    let dc_difference = (rows[0] - rows[4]).mul_add(scaled(1, 0), level_shift);
    let even_first = rows[6].mul_add(scaled(0, 6), rows[2] * scaled(0, 2));
    let even_second = rows[6].mul_add(scaled(1, 6), rows[2] * scaled(1, 2));
    let even = [
        dc_sum + even_first,
        dc_difference + even_second,
        dc_difference - even_second,
        dc_sum - even_first,
    ];

    let odd: [F32x8; 4] = std::array::from_fn(|sample| {
        rows[7].mul_add(
            scaled(sample, 7),
            rows[5].mul_add(
                scaled(sample, 5),
                rows[3].mul_add(scaled(sample, 3), rows[1] * scaled(sample, 1)),
            ),
        )
    });

    let mut samples = [0_u8; 64];
    let (sample_rows, sample_remainder) = samples.as_chunks_mut::<BLOCK_SIDE>();
    invariant_eq!(sample_remainder.len(), 0);

    for (sample, (even, odd)) in even.into_iter().zip(odd).enumerate() {
        sample_rows[sample] = encode_samples(even + odd);
        sample_rows[BLOCK_SIDE - 1 - sample] = encode_samples(even - odd);
    }

    samples
}

/// Rounds level-shifted samples half away from zero and saturates them to `u8`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
fn encode_samples(values: F32x8) -> [u8; BLOCK_SIDE] {
    let values = values
        .simd_gt(F32x8::splat(0.0))
        .select(values, F32x8::splat(0.0));

    let values = values
        .simd_lt(F32x8::splat(255.0))
        .select(values, F32x8::splat(255.0));

    #[expect(unsafe_code, reason = "an unchecked conversion is one instruction")]
    let rounded = unsafe { (values + F32x8::splat(0.499_999_97)).to_int_unchecked::<u32>() };
    rounded.cast::<u8>().to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluates the ITU T.81 inverse DCT directly, in f64.
    ///
    /// Uses the specification's double sum as an independent reference.
    fn reference_inverse(coefficients: &[i32; 64]) -> [f64; 64] {
        fn normalization(frequency: usize) -> f64 {
            if frequency == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            }
        }

        // Every index here is below eight, so the conversion is exact.
        fn position(index: usize) -> f64 {
            f64::from(u8::try_from(index).expect("a block index fits u8"))
        }

        std::array::from_fn(|index| {
            let (x, y) = (index % BLOCK_SIDE, index / BLOCK_SIDE);
            let mut sample = 0.0;

            for vertical in 0..BLOCK_SIDE {
                for horizontal in 0..BLOCK_SIDE {
                    let angle_x =
                        (2.0 * position(x) + 1.0) * position(horizontal) * std::f64::consts::PI
                            / 16.0;
                    let angle_y =
                        (2.0 * position(y) + 1.0) * position(vertical) * std::f64::consts::PI
                            / 16.0;

                    sample += normalization(horizontal)
                        * normalization(vertical)
                        * f64::from(coefficients[vertical * BLOCK_SIDE + horizontal])
                        * angle_x.cos()
                        * angle_y.cos();
                }
            }

            sample / 4.0 + f64::from(LEVEL_SHIFT)
        })
    }

    #[test]
    fn the_transform_matches_a_direct_evaluation_of_the_specification() {
        // The flat-block test above exercises only the DC-only shortcut, which returns before the
        // transform runs. These blocks all carry alternating-current energy, so they reach it.
        let mut worst = 0.0_f64;

        for seed in 0..24_u32 {
            let mut state = u64::from(seed).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            let coefficients: [i32; 64] = std::array::from_fn(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Realistic magnitudes: a large DC term and quantized alternating-current terms
                // that decay with frequency.
                let magnitude = if index == 0 { 1024 } else { 512 >> (index / 8) };
                let value = i64::from((state >> 40) as u32 % 1024) - 512;
                i32::try_from(value * i64::from(magnitude) / 512).expect("test coefficient fits")
            });

            let actual = inverse(&coefficients);
            let expected = reference_inverse(&coefficients);

            for (actual, expected) in actual.into_iter().zip(expected) {
                let clamped = expected.round().clamp(0.0, 255.0);
                worst = worst.max((f64::from(actual) - clamped).abs());
            }
        }

        assert!(
            worst <= 1.0,
            "the transform drifted {worst} levels from a direct evaluation of the specification"
        );
    }

    #[test]
    fn dc_coefficient_produces_a_flat_block() {
        let mut coefficients = [0; 64];
        coefficients[0] = 80;
        let samples = inverse(&coefficients);

        invariant_eq!(samples, [138; 64]);
        invariant!(samples.windows(2).all(|window| window[0] == window[1]));
    }
}
