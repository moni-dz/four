use multiversion::multiversion;
use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};
use crate::math::{
    exp2_bounded, log2_positive_normal, max_or_second, min_or_second, pow_unit_interval, recip,
    recip_for_target,
};
use crate::simd::map_planes;

const BT2446_LANES: usize = 16;
type F32x16 = Simd<f32, BT2446_LANES>;

const HDR_TO_SDR_PEAK_RATIO: f32 = 10.0;
const BT2020_LUMA: [f32; 3] = [0.262_7, 0.678_0, 0.059_3];
const CB_DIVISOR: f32 = 1.881_4;
const CR_DIVISOR: f32 = 1.474_6;
const RHO_HDR: f32 = 13.259_798;
const RHO_SDR: f32 = 5.696_957_6;

// `ln(x) / ln(RHO)` and `RHO.powf(x)` both reduce to a base-two logarithm scaled by a constant.
// Folding the logarithm of each rho into a constant removes two of the fourteen transcendental
// evaluations this operator performs per pixel.
const LOG2_RHO_HDR: f32 = 3.728_987;
const LOG2_RHO_SDR: f32 = 2.510_191_7;

/// Applies BT.2446 HDR-to-SDR conversion Method A.
///
/// Method A converts BT.2020 display-linear HDR mastered at 1,000 cd/m^2 to SDR at 100 cd/m^2. In
/// this crate's target-relative representation, input `10.0` is the HDR peak and output `1.0` is
/// the SDR peak. Inputs outside the specified range are clipped.
///
/// The conversion applies the `2.4` transfer function, maps BT.2020 luma through the perceptual
/// knee, corrects chroma for the Hunt effect, reconstructs BT.2020 RGB, and returns display-linear
/// components.
///
/// See [Report ITU-R BT.2446-1], Tables 2 and 3.
///
/// # Examples
///
/// ```
/// use tonemapping::{BT2446A, LinearRGB, ToneMapper};
///
/// // The mastering peak maps to display white within floating-point rounding.
/// let display_linear = BT2446A.map(LinearRGB::new([10.0; 3]));
/// for component in display_linear.components() {
///     assert!((component - 1.0).abs() < 1.0e-6);
/// }
/// ```
///
/// [Report ITU-R BT.2446-1]: https://www.itu.int/pub/R-REP-BT.2446-1-2021
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BT2446A;

impl ToneMapper for BT2446A {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        bt2446a(color)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        let simd_len = colors.len() / BT2446_LANES * BT2446_LANES;
        let (simd_colors, tail) = colors.split_at_mut(simd_len);

        if !simd_colors.is_empty() {
            bt2446a_batch(simd_colors);
        }

        for color in tail {
            *color = bt2446a(*color);
        }
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, BT2446_LANES, bt2446a_planes, self);
    }
}

#[multiversion(targets(
    "x86_64+avx512f+avx512vl",
    "x86_64+avx512f",
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
fn bt2446a_batch(colors: &mut [LinearRGB]) {
    let (chunks, tail) = colors.as_chunks_mut::<BT2446_LANES>();
    debug_assert!(
        tail.is_empty(),
        "BT.2446 SIMD input must contain complete chunks"
    );

    for chunk in chunks {
        let components = [0, 1, 2]
            .map(|channel| F32x16::from_array(std::array::from_fn(|lane| chunk[lane].0[channel])));

        let mapped = bt2446a_simd(&components, |x| recip_for_target!(x));

        for lane in 0..BT2446_LANES {
            chunk[lane] = LinearRGB([mapped[0][lane], mapped[1][lane], mapped[2][lane]]);
        }
    }
}

#[multiversion(targets(
    "x86_64+avx512f+avx512vl",
    "x86_64+avx512f",
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
fn bt2446a_planes(colors: &mut LinearRGBPlanes) {
    let [red, green, blue] = colors.channels_mut();
    let [red_chunks, green_chunks, blue_chunks] = [red, green, blue].map(|channel| {
        let (chunks, _) = channel.as_chunks_mut::<BT2446_LANES>();
        chunks
    });

    for ((red, green), blue) in red_chunks.iter_mut().zip(green_chunks).zip(blue_chunks) {
        let components = [*red, *green, *blue].map(F32x16::from_array);
        let mapped = bt2446a_simd(&components, |x| recip_for_target!(x));

        *red = mapped[0];
        *green = mapped[1];
        *blue = mapped[2];
    }
}

/// Maps sanitized colors, one per lane. Inlined into each multiversioned caller, so that it compiles
/// for the caller's instruction set rather than the baseline target.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
#[allow(
    clippy::similar_names,
    reason = "cb/cr mirror the CB_DIVISOR/CR_DIVISOR constants"
)]
fn bt2446a_simd<const N: usize>(
    components: &[Simd<f32, N>; 3],
    recip: impl Fn(Simd<f32, N>) -> Simd<f32, N>,
) -> [[f32; N]; 3] {
    let zero = Simd::splat(0.0);
    let one = Simd::splat(1.0);

    let nonlinear = components.map(|component| {
        let normalized = min_or_second(component * Simd::splat(1.0 / HDR_TO_SDR_PEAK_RATIO), one);
        pow_unit_interval(normalized, 1.0 / 2.4)
    });

    let input_luma = nonlinear[2].mul_add(
        Simd::splat(BT2020_LUMA[2]),
        nonlinear[1].mul_add(
            Simd::splat(BT2020_LUMA[1]),
            nonlinear[0] * Simd::splat(BT2020_LUMA[0]),
        ),
    );

    let perceptual_luma = log2_positive_normal(Simd::splat(RHO_HDR - 1.0).mul_add(input_luma, one))
        * Simd::splat(1.0 / LOG2_RHO_HDR);

    let compressed_luma = perceptual_luma.simd_le(Simd::splat(0.739_9)).select(
        Simd::splat(1.077_0) * perceptual_luma,
        perceptual_luma.simd_lt(Simd::splat(0.990_9)).select(
            perceptual_luma.mul_add(
                perceptual_luma.mul_add(Simd::splat(-1.151_0), Simd::splat(2.781_1)),
                Simd::splat(-0.630_2),
            ),
            perceptual_luma.mul_add(Simd::splat(0.5), Simd::splat(0.5)),
        ),
    );

    let output_luma = (exp2_bounded(compressed_luma * Simd::splat(LOG2_RHO_SDR)) - one)
        * Simd::splat(1.0 / (RHO_SDR - 1.0));

    let color_scale = input_luma
        .simd_eq(zero)
        .select(zero, output_luma * recip(Simd::splat(1.1) * input_luma));

    let blue_difference = color_scale * (nonlinear[2] - input_luma) * Simd::splat(1.0 / CB_DIVISOR);
    let red_difference = color_scale * (nonlinear[0] - input_luma) * Simd::splat(1.0 / CR_DIVISOR);

    let adjusted_luma = max_or_second(red_difference, zero).mul_add(Simd::splat(-0.1), output_luma);

    let output_nonlinear = [
        red_difference.mul_add(Simd::splat(CR_DIVISOR), adjusted_luma),
        red_difference.mul_add(
            Simd::splat(-(BT2020_LUMA[0] * CR_DIVISOR / BT2020_LUMA[1])),
            blue_difference.mul_add(
                Simd::splat(-(BT2020_LUMA[2] * CB_DIVISOR / BT2020_LUMA[1])),
                adjusted_luma,
            ),
        ),
        blue_difference.mul_add(Simd::splat(CB_DIVISOR), adjusted_luma),
    ];

    output_nonlinear
        .map(|component| pow_unit_interval(min_or_second(component, one), 2.4).to_array())
}

fn bt2446a(color: LinearRGB) -> LinearRGB {
    let components = color.components().map(Simd::<f32, 1>::splat);
    let mapped = bt2446a_simd(&components, recip);

    LinearRGB::displayable(mapped.map(|[component]| component))
}
