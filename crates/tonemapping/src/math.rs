//! Approximates `log2`, `exp2`, and `recip` over SIMD `f32` lanes.
//!
//! Scalar wrappers share the SIMD implementation for bit parity. Tests enforce four-ULP accuracy.

use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::{SimdFloat, SimdInt, SimdUint},
};

use crate::simd::F32x8;

/// Approximates `1.0 / x` via AVX `vrcpps` plus one Newton-Raphson step.
///
/// Falls back to a plain divide off x86_64 or without AVX at runtime.
#[inline]
#[expect(unsafe_code, reason = "AVX raw intrinsics")]
pub(crate) fn recip(x: F32x8) -> F32x8 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") {
        // SAFETY: AVX support is present.
        return unsafe { recip_avx(x) };
    }

    F32x8::splat(1.0) / x
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[expect(unsafe_code, reason = "AVX raw intrinsics")]
unsafe fn recip_avx(x: F32x8) -> F32x8 {
    use std::arch::x86_64::{
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_rcp_ps, _mm256_set1_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    let input = x.to_array();
    // SAFETY: `input` is fully-initialized.
    unsafe {
        let v = _mm256_loadu_ps(input.as_ptr());
        let approx = _mm256_rcp_ps(v);
        // Newton-Raphson
        let two = _mm256_set1_ps(2.0);
        let refined = _mm256_mul_ps(approx, _mm256_sub_ps(two, _mm256_mul_ps(v, approx)));

        let mut output = [0.0f32; 8];
        _mm256_storeu_ps(output.as_mut_ptr(), refined);
        F32x8::from_array(output)
    }
}

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
    std::f32::consts::LN_2,
];

/// `1 / ln(2)`, for converting a natural logarithm to base two.
const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// `sqrt(2)`, the upper end of the mantissa range the logarithm polynomial covers.
const SQRT_TWO: f32 = std::f32::consts::SQRT_2;

/// Evaluates a polynomial in `value` by Horner's method, highest coefficient first.
#[inline]
pub(crate) fn polynomial<const N: usize, const DEGREE: usize>(
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
/// Zero and negative inputs return negative infinity and `NaN`, matching [`f32::log2`]. Subnormal
/// inputs are scaled into the normal range.
#[must_use]
#[inline]
pub fn log2<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
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

    let result = value
        .simd_eq(Simd::splat(f32::INFINITY))
        .select(Simd::splat(f32::INFINITY), result);

    value.is_nan().select(Simd::splat(f32::NAN), result)
}

/// Returns two raised to the power of each lane.
///
/// Underflows to zero below `-149` and overflows to infinity above `128`, matching [`f32::exp2`].
#[must_use]
#[inline]
pub fn exp2<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
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

/// `log2`, for finite values at least `1.0`.
///
/// Omits handling for subnormal, zero, negative, infinite, and `NaN` inputs.
///
/// Callers must not rely on a defined result outside `1.0..=f32::MAX`.
#[must_use]
#[inline]
pub(crate) fn log2_positive_normal<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    let bits = value.to_bits();
    let exponent = ((bits >> Simd::splat(23)) & Simd::splat(0xff)).cast::<i32>() - Simd::splat(127);
    let mantissa =
        Simd::from_bits((bits & Simd::splat(0x007f_ffff)) | Simd::splat(0x3f80_0000_u32));

    let lend = mantissa.simd_gt(Simd::splat(SQRT_TWO));
    let mantissa = lend.select(mantissa * Simd::splat(0.5), mantissa);
    let exponent = lend
        .select(exponent + Simd::splat(1), exponent)
        .cast::<f32>();

    let t = mantissa - Simd::splat(1.0);
    let squared = t * t;
    let corrected = (t * squared).mul_add(
        polynomial(t, LOG_POLYNOMIAL),
        squared * Simd::splat(-0.5) + t,
    );

    corrected.mul_add(Simd::splat(LOG2_E), exponent)
}

/// `exp2`, for finite values away from the `-150.0..=128.0` extremes.
///
/// Omits input clamping and underflow, overflow, and `NaN` handling. Results are defined only
/// roughly in `-16.0..=16.0`.
#[must_use]
#[inline]
pub(crate) fn exp2_bounded<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    let whole = value.round();
    let fraction = value - whole;

    let fractional = fraction.mul_add(polynomial(fraction, EXP2_POLYNOMIAL), Simd::splat(1.0));

    let exponent = whole.cast::<i32>() + Simd::splat(127);
    let scale = Simd::<f32, N>::from_bits((exponent << Simd::splat(23)).cast::<u32>());

    fractional * scale
}

/// Returns the base-two logarithm of `value` using the vector path.
#[inline]
pub(crate) fn log2_scalar(value: f32) -> f32 {
    log2(Simd::<f32, 1>::splat(value))[0]
}

/// Returns two raised to the power of `value` using the vector path.
#[inline]
pub(crate) fn exp2_scalar(value: f32) -> f32 {
    exp2(Simd::<f32, 1>::splat(value))[0]
}

#[cfg(test)]
mod tests {
    use super::{exp2, exp2_bounded, exp2_scalar, log2, log2_positive_normal, log2_scalar};
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
    fn narrow_domain_variants_match_the_general_ones_in_their_documented_domain() {
        for exponent in -4..=4_i32 {
            for step in 0..64_u32 {
                let mantissa = 1.0 + f32::from(u16::try_from(step).unwrap()) / 64.0;
                let value = mantissa * 2.0_f32.powi(exponent);
                assert_eq!(
                    log2_positive_normal(Simd::<f32, 1>::splat(value))[0].to_bits(),
                    log2(Simd::<f32, 1>::splat(value))[0].to_bits(),
                    "log2_positive_normal({value})"
                );
            }
        }

        for step in -1_600..=1_600_i32 {
            let value = f32::from(i16::try_from(step).unwrap()) / 100.0;
            assert_eq!(
                exp2_bounded(Simd::<f32, 1>::splat(value))[0].to_bits(),
                exp2(Simd::<f32, 1>::splat(value))[0].to_bits(),
                "exp2_bounded({value})"
            );
        }
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
        assert!(log2(Simd::<f32, 1>::splat(f32::NAN))[0].is_nan());

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
