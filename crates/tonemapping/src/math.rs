//! Approximates `log2`, `exp2`, and `recip` over SIMD `f32` lanes.
//!
//! Scalar wrappers share the SIMD implementation for bit parity. Tests enforce four-ULP accuracy.

use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::{SimdFloat, SimdInt, SimdUint},
};

/// Select instructions in the enclosing `#[multiversion]` function, without another dispatcher
/// or SIMD ABI boundary inside its loop.
macro_rules! recip_for_target {
    ($x:expr) => {{
        let x = $x;
        #[cfg(target_arch = "x86_64")]
        {
            use multiversion::target::target_cfg_f;
            use std::arch::x86_64::*;
            use std::simd::Simd;

            // Pad only the last register. Constant lane counts let LLVM remove the copies;
            // even vectors shorter than a register use RCP instead of a divide.
            macro_rules! estimate {
                ($width:literal, $rcp:ident) => {{
                    let input = x.to_array();
                    let mut output = input;
                    for (src, dst) in input.chunks($width).zip(output.chunks_mut($width)) {
                        let mut lanes = [1.0; $width];
                        lanes[..src.len()].copy_from_slice(src);
                        let lanes = Simd::<f32, $width>::from_array(lanes);
                        // SAFETY: the enclosing multiversioned function enables the selected
                        // instruction's features. SIMD/native conversions preserve all lanes.
                        #[expect(unsafe_code, reason = "hardware reciprocal intrinsic")]
                        let approx = unsafe { Simd::<f32, $width>::from($rcp(lanes.into())) };
                        dst.copy_from_slice(&approx.to_array()[..src.len()]);
                    }
                    Simd::from_array(output)
                }};
            }

            let approx = if x.len() > 8 && target_cfg_f!(target_feature = "avx512f") {
                estimate!(16, _mm512_rcp14_ps)
            } else if x.len() > 4 && target_cfg_f!(target_feature = "avx") {
                if target_cfg_f!(all(target_feature = "avx512f", target_feature = "avx512vl")) {
                    estimate!(8, _mm256_rcp14_ps)
                } else {
                    estimate!(8, _mm256_rcp_ps)
                }
            } else if target_cfg_f!(all(target_feature = "avx512f", target_feature = "avx512vl")) {
                estimate!(4, _mm_rcp14_ps)
            } else {
                estimate!(4, _mm_rcp_ps)
            };

            let two = Simd::splat(2.0);
            if target_cfg_f!(any(target_feature = "fma", target_feature = "avx512f")) {
                approx * std::simd::StdFloat::mul_add(-x, approx, two)
            } else {
                approx * (two - x * approx)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::simd::Simd::splat(1.0) / x
        }
    }};
}
pub(crate) use recip_for_target;

/// Approximates `1.0 / x` with a hardware estimate and one Newton-Raphson step.
///
/// Intended for tone-mapping denominators `1 + luminance`, in `1.0..=1e19`. Zero, subnormal,
/// non-finite inputs and reciprocal underflow are outside this approximation's contract.
/// Non-x86-64 targets use division. Inside multiversioned loops use `recip_for_target!` to avoid
/// dispatching again for each vector.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512vl",
    "x86_64+avx512f",
    "x86_64+avx+fma",
    "x86_64+avx",
    "x86_64+sse2",
))]
#[inline]
pub(crate) fn recip<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    recip_for_target!(x)
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

/// The bits of `sqrt(2) / 2`. Subtracting them from an input's bits splits it into an exponent and
/// a mantissa in `[sqrt(2)/2, sqrt(2))`, where the logarithm polynomial is accurate, with integer
/// arithmetic alone: no compare, select, or halving multiply recentres the mantissa.
const HALF_SQRT_TWO_BITS: i32 = 0x3f35_04f3;

/// `1.5 * 2^23`. Adding it to an `f32` below `2^22` in magnitude rounds to the nearest integer and
/// leaves that integer in the low mantissa bits, which replaces a `round` and an `f32`-to-`i32`
/// conversion with one add.
const ROUNDING_SHIFTER: f32 = 12_582_912.0;

/// `2^24`, which scales a subnormal into the normal range.
const SUBNORMAL_SCALE: f32 = 16_777_216.0;

/// Evaluates a polynomial in `value` by Horner's method, highest coefficient first.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
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

/// Returns `a` where it exceeds `b`, otherwise `b`, including where either is `NaN`.
///
/// Matches x86 `MAXPS` operand for operand, so it lowers to one instruction where
/// [`SimdFloat::simd_max`] needs extra `NaN` fixups.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
pub(crate) fn max_or_second<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>) -> Simd<f32, N> {
    a.simd_gt(b).select(a, b)
}

/// Returns `a` where it is below `b`, otherwise `b`, including where either is `NaN`.
///
/// The `MINPS` counterpart of [`max_or_second`].
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
pub(crate) fn min_or_second<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>) -> Simd<f32, N> {
    a.simd_lt(b).select(a, b)
}

/// `log2` of positive, finite, normal lanes; other lanes return unspecified finite values.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
fn log2_normal<const N: usize>(value: Simd<f32, N>, exponent_adjust: Simd<i32, N>) -> Simd<f32, N> {
    let offset = value.to_bits().cast::<i32>() - Simd::splat(HALF_SQRT_TWO_BITS);
    let exponent = ((offset >> Simd::splat(23)) + exponent_adjust).cast::<f32>();
    let mantissa = Simd::<f32, N>::from_bits(
        ((offset & Simd::splat(0x007f_ffff)) + Simd::splat(HALF_SQRT_TWO_BITS)).cast::<u32>(),
    );

    // log(mantissa) = t + t^3 * P(t) - t^2 / 2, with t = mantissa - 1.
    let t = mantissa - Simd::splat(1.0);
    let squared = t * t;
    let corrected = (t * squared).mul_add(
        polynomial(t, LOG_POLYNOMIAL),
        squared.mul_add(Simd::splat(-0.5), t),
    );

    corrected.mul_add(Simd::splat(LOG2_E), exponent)
}

/// Splits `value` into `2^fraction`, with `fraction` in `[-0.5, 0.5]`, and the bits of `value`
/// plus [`ROUNDING_SHIFTER`], whose low bits hold the rounded integer part.
///
/// Requires `value` below `2^22` in magnitude.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
fn exp2_split<const N: usize>(value: Simd<f32, N>) -> (Simd<f32, N>, Simd<u32, N>) {
    let shifted = value + Simd::splat(ROUNDING_SHIFTER);
    let fraction = value - (shifted - Simd::splat(ROUNDING_SHIFTER));
    let fractional = fraction.mul_add(polynomial(fraction, EXP2_POLYNOMIAL), Simd::splat(1.0));

    (fractional, shifted.to_bits())
}

/// Returns `2^whole` for an integer `whole` in `-126..=127`, from [`exp2_split`]'s shifted bits.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
fn exp2_scale<const N: usize>(shifted: Simd<u32, N>) -> Simd<f32, N> {
    // The shifter's own bits are a multiple of 2^22, so shifting by 23 discards them and leaves
    // `whole` in the exponent field; adding the bias completes `2^whole`.
    Simd::from_bits((shifted << Simd::splat(23)) + Simd::splat(127 << 23))
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
    // correction comes back out through the mask below.
    let subnormal = value.simd_lt(Simd::splat(f32::MIN_POSITIVE));
    let scaled = subnormal.select(value * Simd::splat(SUBNORMAL_SCALE), value);
    let result = log2_normal(scaled, subnormal.to_simd() & Simd::splat(-24));

    // Zero, negatives and non-finite inputs never reach the polynomial's assumptions.
    let result = value
        .simd_eq(Simd::splat(0.0))
        .select(Simd::splat(f32::NEG_INFINITY), result);

    let result = value
        .simd_eq(Simd::splat(f32::INFINITY))
        .select(Simd::splat(f32::INFINITY), result);

    // Also true for `NaN`.
    value
        .simd_ge(Simd::splat(0.0))
        .select(result, Simd::splat(f32::NAN))
}

/// Returns two raised to the power of each lane.
///
/// Rounds to zero below `-149` and overflows to infinity from `128`, matching [`f32::exp2`].
#[must_use]
#[inline]
pub fn exp2<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    // `NaN` passes both bounds and propagates through the arithmetic below. The bounds keep the
    // two half scales below normal, and `2^-150` and `2^129` still round to zero and infinity.
    let bounded = min_or_second(
        Simd::splat(129.0),
        max_or_second(Simd::splat(-150.0), value),
    );

    let (fractional, shifted) = exp2_split(bounded);

    // Apply `2^whole` as two normal halves so that results below `2^-126` round once, to the
    // correct subnormal, rather than wrapping the exponent field. Both multiplies are exact for
    // normal results.
    let whole = shifted.cast::<i32>() - Simd::splat(ROUNDING_SHIFTER.to_bits().cast_signed());
    let low = whole >> Simd::splat(1);
    let scale = |exponent: Simd<i32, N>| {
        Simd::<f32, N>::from_bits(((exponent + Simd::splat(127)) << Simd::splat(23)).cast::<u32>())
    };

    fractional * scale(low) * scale(whole - low)
}

/// `log2`, for finite values at least `1.0`.
///
/// Omits handling for subnormal, zero, negative, infinite, and `NaN` inputs.
///
/// Callers must not rely on a defined result outside `1.0..=f32::MAX`.
#[must_use]
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
pub(crate) fn log2_positive_normal<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    log2_normal(value, Simd::splat(0))
}

/// `exp2`, for finite values away from the `-150.0..=128.0` extremes.
///
/// Omits input clamping and underflow, overflow, and `NaN` handling. Results are defined only
/// in `-126.0..=127.0`.
#[must_use]
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
pub(crate) fn exp2_bounded<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    let (fractional, shifted) = exp2_split(value);
    fractional * exp2_scale(shifted)
}

/// Returns `value` raised to `exponent`, for `value` in `0.0..=1.0` and positive `exponent`.
///
/// Zero maps to zero. Results below `2^-126` flush to zero, and subnormal inputs are treated as
/// approximately `2^-127`, so this suits display-referred encodings where both are invisible. It
/// skips the special-value handling of [`log2`] and [`exp2`], which matters on hot paths that
/// apply a transfer function to every component.
#[must_use]
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
pub fn pow_unit_interval<const N: usize>(value: Simd<f32, N>, exponent: f32) -> Simd<f32, N> {
    // At least -127 times `exponent`; the floor at -127 rounds to an exponent field of zero, which
    // `exp2_scale` turns into a zero scale.
    let power = log2_positive_normal(value) * Simd::splat(exponent);
    let (fractional, shifted) = exp2_split(max_or_second(power, Simd::splat(-127.0)));

    value
        .simd_gt(Simd::splat(0.0))
        .select(fractional * exp2_scale(shifted), Simd::splat(0.0))
}

/// Returns the base-two logarithm of `value` using the vector path.
#[cfg(test)]
pub(crate) fn log2_scalar(value: f32) -> f32 {
    log2(Simd::<f32, 1>::splat(value))[0]
}

/// Returns two raised to the power of `value` using the vector path.
#[cfg(test)]
pub(crate) fn exp2_scalar(value: f32) -> f32 {
    exp2(Simd::<f32, 1>::splat(value))[0]
}

#[cfg(test)]
mod tests {
    use super::{
        exp2, exp2_bounded, exp2_scalar, log2, log2_positive_normal, log2_scalar,
        pow_unit_interval, recip,
    };
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
    fn exp2_rounds_subnormal_results_like_the_standard_library() {
        let mut worst = 0;
        for step in -15_100..=-12_500_i32 {
            let value = f32::from(i16::try_from(step).unwrap()) / 100.0;
            worst = worst.max(ulp_distance(exp2_scalar(value), value.exp2()));
        }
        assert!(
            worst <= 1,
            "exp2 drifted {worst} ULP from the standard library below 2^-125"
        );
    }

    #[test]
    fn pow_unit_interval_tracks_powf_and_pins_the_ends() {
        for exponent in [1.0 / 2.4, 2.4, 0.5, 4.0] {
            assert_eq!(
                pow_unit_interval(Simd::<f32, 1>::splat(0.0), exponent)[0],
                0.0
            );
            assert_eq!(
                pow_unit_interval(Simd::<f32, 1>::splat(1.0), exponent)[0],
                1.0
            );

            for step in 1..=100_000_u32 {
                let value = f32::from(u16::try_from(step % 50_000).unwrap() + 1) / 50_001.0
                    * if step > 50_000 { 1.0e-6 } else { 1.0 };

                let actual = pow_unit_interval(Simd::<f32, 1>::splat(value), exponent)[0];
                let expected = f64::from(value).powf(f64::from(exponent));

                if expected < f64::from(f32::MIN_POSITIVE) {
                    assert!(
                        actual < f32::MIN_POSITIVE * 2.0,
                        "{value}^{exponent} = {actual}"
                    );
                } else {
                    let error = (f64::from(actual) - expected).abs() / expected;
                    assert!(
                        error < 2.0e-5,
                        "{value}^{exponent} = {actual}, expected {expected}"
                    );
                }
            }
        }
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
    fn recip_matches_a_plain_divide_at_every_lane_width() {
        fn check<const N: usize>(reciprocal: fn(Simd<f32, N>) -> Simd<f32, N>) {
            for exponent in -120..=120 {
                for step in 0..64_u16 {
                    let values = std::array::from_fn(|lane| {
                        let mantissa =
                            1.0 + f32::from((step + u16::try_from(lane).unwrap()) % 64) / 64.0;
                        let sign = if lane % 2 == 0 { 1.0 } else { -1.0 };
                        sign * mantissa * 2.0_f32.powi(exponent)
                    });
                    let actual = reciprocal(std::hint::black_box(Simd::from_array(values)));
                    for (actual, value) in actual.to_array().into_iter().zip(values) {
                        assert!(
                            ulp_distance(actual, 1.0 / value) <= 4,
                            "recip({value}) = {actual}, N = {N}"
                        );
                    }
                }
            }
        }

        macro_rules! widths {
            ($reciprocal:ident) => {
                check::<1>($reciprocal);
                check::<2>($reciprocal);
                check::<3>($reciprocal);
                check::<4>($reciprocal);
                check::<5>($reciprocal);
                check::<8>($reciprocal);
                check::<12>($reciprocal);
                check::<13>($reciprocal);
                check::<16>($reciprocal);
                check::<32>($reciprocal);
                check::<64>($reciprocal);
            };
        }

        widths!(recip);

        // Compile with RUSTFLAGS=-Ctarget-cpu=x86-64 to exercise each ISA independently.
        #[cfg(target_arch = "x86_64")]
        {
            macro_rules! backend {
                ($name:ident, $target:literal, $supported:expr) => {
                    #[multiversion::multiversion(targets($target), attrs(inline(never)))]
                    fn $name<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
                        recip_for_target!(x)
                    }
                    if $supported {
                        widths!($name);
                    }
                };
            }
            backend!(sse, "x86_64+sse2", true);
            backend!(avx, "x86_64+avx", is_x86_feature_detected!("avx"));
            backend!(
                avx_fma,
                "x86_64+avx+fma",
                is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")
            );
            backend!(
                avx512,
                "x86_64+avx512f",
                is_x86_feature_detected!("avx512f")
            );
            backend!(
                avx512vl,
                "x86_64+avx512f+avx512vl",
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
            );
        }
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
