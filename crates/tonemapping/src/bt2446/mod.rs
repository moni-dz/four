use multiversion::multiversion;
use std::simd::{
    Select, Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};
use crate::simd::map_planes;
use crate::transcendental::{exp2, exp2_scalar, log2, log2_scalar};

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
/// Method A converts BT.2020 display-linear HDR mastered at 1,000 cd/m^2 to display-linear SDR
/// targeting 100 cd/m^2. In this crate's target-relative representation, an input component of
/// `10.0` represents the HDR mastering peak and an output component of `1.0` represents the SDR
/// target peak. Inputs outside the specified full range are clipped before conversion.
///
/// The conversion follows Tables 2 and 3 of Report ITU-R BT.2446-1: it applies the `2.4` transfer
/// function, maps BT.2020 luma through the three-stage perceptual knee, corrects chroma for the Hunt
/// effect, reconstructs BT.2020 RGB, and returns display-linear components.
///
/// See [Report ITU-R BT.2446-1], Tables 2 and 3.
///
/// # Examples
///
/// ```
/// use tonemapping::{BT2446A, LinearRGB, ToneMapper};
///
/// // The mastering peak reaches display white. The BT.2020 luma weights do not sum to exactly
/// // one in f32, so the result lands within a rounding step of it rather than on it.
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

    let input_luma = F32x16::splat(BT2020_LUMA[0]) * nonlinear[0]
        + F32x16::splat(BT2020_LUMA[1]) * nonlinear[1]
        + F32x16::splat(BT2020_LUMA[2]) * nonlinear[2];

    let perceptual_luma =
        log2(one + F32x16::splat(RHO_HDR - 1.0) * input_luma) * F32x16::splat(1.0 / LOG2_RHO_HDR);

    let compressed_luma = perceptual_luma.simd_le(F32x16::splat(0.739_9)).select(
        F32x16::splat(1.077_0) * perceptual_luma,
        perceptual_luma.simd_lt(F32x16::splat(0.990_9)).select(
            F32x16::splat(-1.151_0) * perceptual_luma * perceptual_luma
                + F32x16::splat(2.781_1) * perceptual_luma
                - F32x16::splat(0.630_2),
            F32x16::splat(0.5) * perceptual_luma + F32x16::splat(0.5),
        ),
    );

    let output_luma =
        (exp2(compressed_luma * F32x16::splat(LOG2_RHO_SDR)) - one) / F32x16::splat(RHO_SDR - 1.0);

    let color_scale = input_luma
        .simd_eq(zero)
        .select(zero, output_luma / (F32x16::splat(1.1) * input_luma));

    let blue_difference = color_scale * (nonlinear[2] - input_luma) / F32x16::splat(CB_DIVISOR);
    let red_difference = color_scale * (nonlinear[0] - input_luma) / F32x16::splat(CR_DIVISOR);
    let adjusted_luma = output_luma - F32x16::splat(0.1) * red_difference.simd_max(zero);

    let output_nonlinear = [
        adjusted_luma + F32x16::splat(CR_DIVISOR) * red_difference,
        adjusted_luma
            - F32x16::splat(BT2020_LUMA[2] * CB_DIVISOR / BT2020_LUMA[1]) * blue_difference
            - F32x16::splat(BT2020_LUMA[0] * CR_DIVISOR / BT2020_LUMA[1]) * red_difference,
        adjusted_luma + F32x16::splat(CB_DIVISOR) * blue_difference,
    ];

    output_nonlinear.map(|component| {
        let bounded = component.simd_clamp(zero, one);

        exp2(log2(bounded) * F32x16::splat(2.4)).to_array()
    })
}

// Measured slower than the libm `log2f`/`exp2f` this replaced: Horner's method is a serial
// dependency chain, and one scalar color cannot hide its latency the way sixteen lanes do. The
// batch path is 5x faster for the same reason, and it handles all but the final partial lane group
// of a 1024-pixel batch, so the trade is strongly positive in aggregate. Both paths must run the
// same code regardless, or the bit-exact parity test has nothing to assert.
fn bt2446a(color: LinearRGB) -> LinearRGB {
    // Spelled as `log2`/`exp2` rather than `powf`, because `bt2446a_simd` has no vector `powf` and
    // must use that pair. In f64 the ~1 ULP disagreement between the two vanished when the result
    // narrowed to f32; in f32 it does not, and the bit-exact batch parity test would fail. Both
    // paths therefore compute the same primitives in the same order.
    let nonlinear = color.components().map(|component| {
        let normalized = (component / HDR_TO_SDR_PEAK_RATIO).clamp(0.0, 1.0);
        exp2_scalar(log2_scalar(normalized) * (1.0 / 2.4))
    });

    let input_luma = BT2020_LUMA[0] * nonlinear[0]
        + BT2020_LUMA[1] * nonlinear[1]
        + BT2020_LUMA[2] * nonlinear[2];

    let output_luma = bt2446a_luma(input_luma);

    let color_scale = if input_luma == 0.0 {
        0.0
    } else {
        output_luma / (1.1 * input_luma)
    };

    let blue_difference = color_scale * (nonlinear[2] - input_luma) / CB_DIVISOR;
    let red_difference = color_scale * (nonlinear[0] - input_luma) / CR_DIVISOR;
    let adjusted_luma = output_luma - (0.1 * red_difference).max(0.0);

    let output_nonlinear = [
        adjusted_luma + CR_DIVISOR * red_difference,
        adjusted_luma
            - (BT2020_LUMA[2] * CB_DIVISOR / BT2020_LUMA[1]) * blue_difference
            - (BT2020_LUMA[0] * CR_DIVISOR / BT2020_LUMA[1]) * red_difference,
        adjusted_luma + CB_DIVISOR * blue_difference,
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
        // `-1.1510 * pl * pl`, not `-1.1510 * pl.powi(2)`: the batch path multiplies left to
        // right, and the two groupings do not round alike in f32.
        -1.151_0 * perceptual_luma * perceptual_luma + 2.781_1 * perceptual_luma - 0.630_2
    } else {
        0.5 * perceptual_luma + 0.5
    };

    (exp2_scalar(compressed_luma * LOG2_RHO_SDR) - 1.0) / (RHO_SDR - 1.0)
}
