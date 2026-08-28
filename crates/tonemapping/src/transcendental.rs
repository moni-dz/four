//! Vectorized `log2` and `exp2` for `f32` lanes.
//!
//! `std::simd`'s [`StdFloat::log2`] and [`StdFloat::exp2`] lower to one scalar libm call per lane —
//! rustc links no vector libm — so a sixteen-lane BT.2446 kernel that evaluates fourteen of them
//! per pixel spends nearly all of its time in scalar code. These polynomial approximations run
//! entirely in registers.
//!
//! Both are written once, over `Simd<f32, N>`. The scalar wrappers evaluate the same functions at
//! `N == 1`, so a scalar result is bit-identical to the corresponding lane of a wide one by
//! construction rather than by inspection — which is what the batch parity tests require.
//!
//! Accuracy is within a few units in the last place across the normal range; `transcendental`'s own
//! tests pin that against the standard library.
//!
//! [`StdFloat::log2`]: StdFloat::log2
//! [`StdFloat::exp2`]: StdFloat::exp2

use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::{SimdFloat, SimdInt, SimdUint},
};

/// Minimax coefficients for `log(1 + t) - t + t^2/2`, from Cephes' `logf`.
const LOG_POLYNOMIAL: [f32; 9] = [
    7.037_683_6e-2,
    -1.151_461e-1,
    1.167_699_9e-1,
    -1.242_014_1e-1,
    1.424_932_3e-1,
    -1.666_805_8e-1,
    2.000_071_5e-1,
    -2.499_999_4e-1,
    3.333_333e-1,
];

/// Minimax coefficients for `2^x - 1` on `[-0.5, 0.5]`, from Cephes' `exp2f`.
const EXP2_POLYNOMIAL: [f32; 6] = [
    1.535_336_2e-4,
    1.339_887_4e-3,
    9.618_437e-3,
    5.550_332_5e-2,
    2.402_264_8e-1,
    6.931_472e-1,
];

/// `1 / ln(2)`, for converting a natural logarithm to base two.
const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// `sqrt(2)`, the upper end of the mantissa range the logarithm polynomial covers.
const SQRT_TWO: f32 = std::f32::consts::SQRT_2;

/// Evaluates a polynomial in `value` by Horner's method, highest coefficient first.
#[inline]
fn polynomial<const N: usize, const DEGREE: usize>(
    value: Simd<f32, N>,
    coefficients: [f32; DEGREE],
) -> Simd<f32, N> {
    let mut accumulator = Simd::splat(coefficients[0]);
    for coefficient in &coefficients[1..] {
        accumulator = accumulator.mul_add(value, Simd::splat(*coefficient));
    }
    accumulator
}

/// Returns the base-two logarithm of each lane.
///
/// Zero and negative inputs return negative infinity and `NaN` respectively, matching
/// [`f32::log2`]. Subnormal inputs are scaled into the normal range first, so they are as accurate
/// as any other value.
#[inline]
pub(crate) fn log2<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    // A subnormal has a zero exponent field, so decomposing it directly would read a mantissa
    // without its implicit leading bit. Multiplying by 2^24 makes it normal; the exponent
    // correction comes back out below.
    const SUBNORMAL_SCALE: f32 = 16_777_216.0;
    let subnormal = value.simd_lt(Simd::splat(f32::MIN_POSITIVE)) & value.simd_gt(Simd::splat(0.0));
    let scaled = subnormal.select(value * Simd::splat(SUBNORMAL_SCALE), value);
    let scale_correction = subnormal.select(Simd::splat(-24.0), Simd::splat(0.0));

    // Split into a mantissa in [1, 2) and an unbiased exponent.
    let bits = scaled.to_bits();
    let exponent = ((bits >> Simd::splat(23)) & Simd::splat(0xff)).cast::<i32>() - Simd::splat(127);
    let mantissa =
        Simd::from_bits((bits & Simd::splat(0x007f_ffff)) | Simd::splat(0x3f80_0000_u32));

    // Recentre onto [sqrt(2)/2, sqrt(2)), where the polynomial is accurate, by lending a power of
    // two to the exponent. Without this the argument reaches 1.0 at the top of the binade, where
    // the series converges far too slowly.
    let lend = mantissa.simd_gt(Simd::splat(SQRT_TWO));
    let mantissa = lend.select(mantissa * Simd::splat(0.5), mantissa);
    let exponent = lend
        .select(exponent + Simd::splat(1), exponent)
        .cast::<f32>();

    // log(mantissa) = t + t^3 * P(t) - t^2 / 2, with t = mantissa - 1.
    let t = mantissa - Simd::splat(1.0);
    let squared = t * t;
    let corrected = (t * squared).mul_add(
        polynomial(t, LOG_POLYNOMIAL),
        squared * Simd::splat(-0.5) + t,
    );

    let result = corrected.mul_add(Simd::splat(LOG2_E), exponent + scale_correction);

    // Zero, negatives and non-finite inputs never reach the polynomial's assumptions.
    let result = value
        .simd_eq(Simd::splat(0.0))
        .select(Simd::splat(f32::NEG_INFINITY), result);
    let result = value
        .simd_lt(Simd::splat(0.0))
        .select(Simd::splat(f32::NAN), result);
    value
        .simd_eq(Simd::splat(f32::INFINITY))
        .select(Simd::splat(f32::INFINITY), result)
}

/// Returns two raised to the power of each lane.
///
/// Underflows to zero below `-149` and overflows to infinity above `128`, matching [`f32::exp2`].
#[inline]
pub(crate) fn exp2<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    // Clamping first keeps the exponent assembly below in range; the true extremes are restored by
    // the selects at the end.
    let bounded = value.simd_clamp(Simd::splat(-150.0), Simd::splat(129.0));

    // Split into an integer part and a remainder in [-0.5, 0.5].
    let whole = bounded.round();
    let fraction = bounded - whole;

    // 2^fraction, via the minimax polynomial for 2^x - 1.
    let fractional = fraction.mul_add(polynomial(fraction, EXP2_POLYNOMIAL), Simd::splat(1.0));

    // 2^whole, assembled directly into the exponent field.
    let exponent = whole.cast::<i32>() + Simd::splat(127);
    let scale = Simd::<f32, N>::from_bits((exponent << Simd::splat(23)).cast::<u32>());

    let result = fractional * scale;

    let result = value
        .simd_le(Simd::splat(-150.0))
        .select(Simd::splat(0.0), result);
    let result = value
        .simd_ge(Simd::splat(128.0))
        .select(Simd::splat(f32::INFINITY), result);
    value.is_nan().select(Simd::splat(f32::NAN), result)
}

/// Returns the base-two logarithm of `value`, by the same code the vector path uses.
#[inline]
pub(crate) fn log2_scalar(value: f32) -> f32 {
    log2(Simd::<f32, 1>::splat(value))[0]
}

/// Returns two raised to the power of `value`, by the same code the vector path uses.
#[inline]
pub(crate) fn exp2_scalar(value: f32) -> f32 {
    exp2(Simd::<f32, 1>::splat(value))[0]
}

#[cfg(test)]
mod tests {
    use super::{exp2, exp2_scalar, log2, log2_scalar};
    use std::simd::Simd;

    /// Returns the distance between two floats in units in the last place.
    fn ulp_distance(actual: f32, expected: f32) -> i64 {
        // Map the sign-magnitude representation onto a monotonic integer ordering so that the
        // difference counts representable values rather than bit patterns.
        fn ordered(value: f32) -> i64 {
            let bits = i64::from(value.to_bits());
            if value.is_sign_negative() {
                // Negative floats count downward in bit order; mirror them so the full range is
                // monotonic and a subtraction counts representable values.
                (1_i64 << 31) - bits
            } else {
                bits
            }
        }
        (ordered(actual) - ordered(expected)).abs()
    }

    #[test]
    fn log2_tracks_the_standard_library_across_the_normal_range() {
        let mut worst = 0;
        // Sweep exponents and mantissas rather than a linear range, so every binade is covered.
        for exponent in -60..=60_i32 {
            for step in 0..64_u32 {
                let mantissa = 1.0 + f32::from(u16::try_from(step).unwrap()) / 64.0;
                let value = mantissa * 2.0_f32.powi(exponent);
                worst = worst.max(ulp_distance(log2_scalar(value), value.log2()));
            }
        }
        assert!(
            worst <= 4,
            "log2 drifted {worst} ULP from the standard library"
        );
    }

    #[test]
    fn exp2_tracks_the_standard_library_across_the_normal_range() {
        let mut worst = 0;
        for step in -12_400..=12_400_i32 {
            let value = f32::from(i16::try_from(step).unwrap()) / 100.0;
            worst = worst.max(ulp_distance(exp2_scalar(value), value.exp2()));
        }
        assert!(
            worst <= 4,
            "exp2 drifted {worst} ULP from the standard library"
        );
    }

    #[test]
    fn the_edges_match_the_standard_library_exactly() {
        for value in [0.0_f32, 1.0, 2.0, 0.5, f32::INFINITY, f32::MIN_POSITIVE] {
            assert_eq!(
                log2_scalar(value).to_bits(),
                value.log2().to_bits(),
                "log2({value})"
            );
        }
        assert!(log2_scalar(-1.0).is_nan());

        for value in [0.0_f32, 1.0, -1.0, 10.0, -160.0, 200.0, f32::NEG_INFINITY] {
            assert_eq!(
                exp2_scalar(value).to_bits(),
                value.exp2().to_bits(),
                "exp2({value})"
            );
        }
        assert!(exp2(Simd::<f32, 1>::splat(f32::NAN))[0].is_nan());
    }

    #[test]
    fn every_lane_agrees_with_the_scalar_result() {
        // The scalar wrappers evaluate the same generic function at one lane, so this holds by
        // construction; it is asserted because the batch parity tests depend on it.
        let inputs = [0.0_f32, 0.001, 0.25, 1.0, 3.5, 17.0, 1.0e12, 4.7e-30];
        let vector = Simd::<f32, 8>::from_array(inputs);

        assert_eq!(
            log2(vector).to_array().map(f32::to_bits),
            inputs.map(|value| log2_scalar(value).to_bits())
        );
        assert_eq!(
            exp2(vector).to_array().map(f32::to_bits),
            inputs.map(|value| exp2_scalar(value).to_bits())
        );
    }
}
