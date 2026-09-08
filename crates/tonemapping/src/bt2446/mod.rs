use multiversion::multiversion;
use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};
use crate::simd::map_planes;
use crate::transcendental::{
    exp2, exp2_bounded, exp2_scalar, log2, log2_positive_normal, log2_scalar,
};

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

#[multiversion(targets = "simd")]
fn bt2446a_batch(colors: &mut [LinearRGB]) {
    let (chunks, tail) = colors.as_chunks_mut::<BT2446_LANES>();
    debug_assert!(
        tail.is_empty(),
        "BT.2446 SIMD input must contain complete chunks"
    );

    for chunk in chunks {
        let components = [0, 1, 2]
            .map(|channel| F32x16::from_array(std::array::from_fn(|lane| chunk[lane].0[channel])));

        let mapped = bt2446a_simd(&components);

        for lane in 0..BT2446_LANES {
            chunk[lane] = LinearRGB([mapped[0][lane], mapped[1][lane], mapped[2][lane]]);
        }
    }
}

#[multiversion(targets = "simd")]
fn bt2446a_planes(colors: &mut LinearRGBPlanes) {
    let [red, green, blue] = colors.channels_mut();
    let [red_chunks, green_chunks, blue_chunks] = [red, green, blue].map(|channel| {
        let (chunks, _) = channel.as_chunks_mut::<BT2446_LANES>();
        chunks
    });

    for ((red, green), blue) in red_chunks.iter_mut().zip(green_chunks).zip(blue_chunks) {
        let components = [*red, *green, *blue].map(F32x16::from_array);
        let mapped = bt2446a_simd(&components);

        *red = mapped[0];
        *green = mapped[1];
        *blue = mapped[2];
    }
}

#[inline]
fn bt2446a_simd(components: &[F32x16; 3]) -> [[f32; BT2446_LANES]; 3] {
    let zero = F32x16::splat(0.0);
    let one = F32x16::splat(1.0);

    let nonlinear = components.map(|component| {
        let normalized = (component / F32x16::splat(HDR_TO_SDR_PEAK_RATIO)).simd_clamp(zero, one);
        exp2(log2(normalized) * F32x16::splat(1.0 / 2.4))
    });

    let input_luma = nonlinear[2].mul_add(
        F32x16::splat(BT2020_LUMA[2]),
        nonlinear[1].mul_add(
            F32x16::splat(BT2020_LUMA[1]),
            nonlinear[0] * F32x16::splat(BT2020_LUMA[0]),
        ),
    );

    // `input_luma` is a sum of nonnegative terms (each `nonlinear` channel and `BT2020_LUMA`
    // weight is nonnegative), so this argument is always finite and at least `1.0`.
    let perceptual_luma = log2_positive_normal(one + F32x16::splat(RHO_HDR - 1.0) * input_luma)
        * F32x16::splat(1.0 / LOG2_RHO_HDR);

    let compressed_luma = perceptual_luma.simd_le(F32x16::splat(0.739_9)).select(
        F32x16::splat(1.077_0) * perceptual_luma,
        perceptual_luma.simd_lt(F32x16::splat(0.990_9)).select(
            perceptual_luma.mul_add(
                perceptual_luma.mul_add(F32x16::splat(-1.151_0), F32x16::splat(2.781_1)),
                F32x16::splat(-0.630_2),
            ),
            F32x16::splat(0.5) * perceptual_luma + F32x16::splat(0.5),
        ),
    );

    // `perceptual_luma` is in `0.0..=~1.0` (a knee function of a `0.0..=1.0`-ish input), so this
    // argument stays near `0.0..=LOG2_RHO_SDR`, far inside `exp2_bounded`'s safe range.
    let output_luma = (exp2_bounded(compressed_luma * F32x16::splat(LOG2_RHO_SDR)) - one)
        / F32x16::splat(RHO_SDR - 1.0);

    let color_scale = input_luma
        .simd_eq(zero)
        .select(zero, output_luma / (F32x16::splat(1.1) * input_luma));

    let blue_difference = color_scale * (nonlinear[2] - input_luma) / F32x16::splat(CB_DIVISOR);
    let red_difference = color_scale * (nonlinear[0] - input_luma) / F32x16::splat(CR_DIVISOR);
    let adjusted_luma = red_difference
        .simd_max(zero)
        .mul_add(F32x16::splat(-0.1), output_luma);

    let output_nonlinear = [
        red_difference.mul_add(F32x16::splat(CR_DIVISOR), adjusted_luma),
        red_difference.mul_add(
            F32x16::splat(-(BT2020_LUMA[0] * CR_DIVISOR / BT2020_LUMA[1])),
            blue_difference.mul_add(
                F32x16::splat(-(BT2020_LUMA[2] * CB_DIVISOR / BT2020_LUMA[1])),
                adjusted_luma,
            ),
        ),
        blue_difference.mul_add(F32x16::splat(CB_DIVISOR), adjusted_luma),
    ];

    output_nonlinear.map(|component| {
        let bounded = component.simd_clamp(zero, one);

        exp2(log2(bounded) * F32x16::splat(2.4)).to_array()
    })
}

// Match the SIMD primitive order so batch tails remain bit-identical.
fn bt2446a(color: LinearRGB) -> LinearRGB {
    let nonlinear = color.components().map(|component| {
        let normalized = (component / HDR_TO_SDR_PEAK_RATIO).clamp(0.0, 1.0);
        exp2_scalar(log2_scalar(normalized) * (1.0 / 2.4))
    });

    let input_luma = nonlinear[2].mul_add(
        BT2020_LUMA[2],
        nonlinear[1].mul_add(BT2020_LUMA[1], nonlinear[0] * BT2020_LUMA[0]),
    );

    let output_luma = bt2446a_luma(input_luma);

    let color_scale = if input_luma == 0.0 {
        0.0
    } else {
        output_luma / (1.1 * input_luma)
    };

    let blue_difference = color_scale * (nonlinear[2] - input_luma) / CB_DIVISOR;
    let red_difference = color_scale * (nonlinear[0] - input_luma) / CR_DIVISOR;
    let adjusted_luma = red_difference.max(0.0).mul_add(-0.1, output_luma);

    let output_nonlinear = [
        red_difference.mul_add(CR_DIVISOR, adjusted_luma),
        red_difference.mul_add(
            -(BT2020_LUMA[0] * CR_DIVISOR / BT2020_LUMA[1]),
            blue_difference.mul_add(
                -(BT2020_LUMA[2] * CB_DIVISOR / BT2020_LUMA[1]),
                adjusted_luma,
            ),
        ),
        blue_difference.mul_add(CB_DIVISOR, adjusted_luma),
    ];

    LinearRGB::displayable(output_nonlinear.map(|component| {
        let bounded = component.clamp(0.0, 1.0);
        exp2_scalar(log2_scalar(bounded) * 2.4)
    }))
}

fn bt2446a_luma(input_luma: f32) -> f32 {
    let perceptual_luma = log2_scalar(1.0 + (RHO_HDR - 1.0) * input_luma) * (1.0 / LOG2_RHO_HDR);

    let compressed_luma = if perceptual_luma <= 0.739_9 {
        1.077_0 * perceptual_luma
    } else if perceptual_luma < 0.990_9 {
        // Horner form via `mul_add`, matching the batch path's fused rounding exactly.
        perceptual_luma.mul_add(perceptual_luma.mul_add(-1.151_0, 2.781_1), -0.630_2)
    } else {
        0.5 * perceptual_luma + 0.5
    };

    (exp2_scalar(compressed_luma * LOG2_RHO_SDR) - 1.0) / (RHO_SDR - 1.0)
}
