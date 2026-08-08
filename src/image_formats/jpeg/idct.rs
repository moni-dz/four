use multiversion::multiversion;
use std::simd::{Simd, StdFloat, num::SimdFloat};

const BLOCK_SIDE: usize = 8;
// These literals carry the same 36 decimal places as Rust's f32 mathematical constants. The
// compiler still rounds each one to f32 once, but retaining the source precision makes that
// rounding reproducible and prevents hand-rounded matrix entries from becoming the definition.
const COSINE_PI_OVER_SIXTEEN: f32 = 0.980785280403230449126182236134239036_f32;
const COSINE_PI_OVER_EIGHT: f32 = 0.923879532511286756128183189396788286_f32;
const COSINE_THREE_PI_OVER_SIXTEEN: f32 = 0.831469612302545237078788377617905756_f32;
const COSINE_FIVE_PI_OVER_SIXTEEN: f32 = 0.555570233019602224742830813948532874_f32;
const COSINE_THREE_PI_OVER_EIGHT: f32 = 0.382683432365089771728459984030398866_f32;
const COSINE_SEVEN_PI_OVER_SIXTEEN: f32 = 0.195090322016128267848284868477022240_f32;
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

pub(super) fn inverse(coefficients: &[i32; 64]) -> [u8; 64] {
    assert_eq!(coefficients.len(), BLOCK_SIDE * BLOCK_SIDE);
    assert!(
        coefficients
            .iter()
            .all(|value| value.checked_abs().is_some())
    );

    if coefficients[1..].iter().all(|value| *value == 0) {
        let sample = clamp_sample(coefficients[0] as f32 / 8.0 + LEVEL_SHIFT);
        return [sample; 64];
    }
    inverse_simd(coefficients)
}

// Dispatch only non-flat blocks: a dispatch before the cheap DC-only path would slow down the most
// common case. Each target gets the same portable-SIMD algorithm, plus a safe baseline copy.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1", "aarch64+neon"))]
fn inverse_simd(coefficients: &[i32; 64]) -> [u8; 64] {
    assert_eq!(coefficients.len(), BLOCK_SIDE * BLOCK_SIDE);
    assert!(coefficients[1..].iter().any(|value| *value != 0));

    let mut intermediate = [[0.0_f32; BLOCK_SIDE]; BLOCK_SIDE];
    for (vertical_frequency, intermediate_row) in intermediate.iter_mut().enumerate() {
        let coefficient_start = vertical_frequency * BLOCK_SIDE;
        let coefficient_row = F32x8::from_array(std::array::from_fn(|index| {
            coefficients[coefficient_start + index] as f32
        }));
        for (x, target) in intermediate_row.iter_mut().enumerate() {
            let basis = F32x8::from_array(BASIS[x]);
            *target = (coefficient_row * basis).reduce_sum();
        }
    }

    let mut samples = [0_u8; 64];
    let (sample_rows, sample_remainder) = samples.as_chunks_mut::<BLOCK_SIDE>();
    assert!(sample_remainder.is_empty());
    for (sample_row, basis_row) in sample_rows.iter_mut().zip(BASIS.iter()) {
        let mut values = F32x8::splat(0.0);
        for (intermediate_row, basis) in intermediate.iter().zip(basis_row.iter()) {
            values = F32x8::from_array(*intermediate_row).mul_add(F32x8::splat(*basis), values);
        }
        values = (values / F32x8::splat(4.0) + F32x8::splat(LEVEL_SHIFT))
            .round()
            .simd_clamp(F32x8::splat(0.0), F32x8::splat(255.0));
        for (sample, value) in sample_row.iter_mut().zip(values.to_array()) {
            *sample = value as u8;
        }
    }
    samples
}

fn clamp_sample(value: f32) -> u8 {
    assert!(value.is_finite());
    value.round().clamp(f32::from(u8::MIN), f32::from(u8::MAX)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_coefficient_produces_a_flat_block() {
        let mut coefficients = [0; 64];
        coefficients[0] = 80;
        let samples = inverse(&coefficients);

        assert_eq!(samples, [138; 64]);
        assert!(samples.windows(2).all(|window| window[0] == window[1]));
    }
}
