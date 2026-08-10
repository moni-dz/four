use multiversion::multiversion;
use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use super::{LinearRGB, LinearRGBPlanes, ToneMapper};

const BT2446_LANES: usize = 16;
type F64x16 = Simd<f64, BT2446_LANES>;

const HDR_TO_SDR_PEAK_RATIO: f64 = 10.0;
const BT2020_LUMA: [f64; 3] = [0.262_7, 0.678_0, 0.059_3];
const CB_DIVISOR: f64 = 1.881_4;
const CR_DIVISOR: f64 = 1.474_6;
const RHO_HDR: f64 = 13.259_797_918_583_32;
const RHO_SDR: f64 = 5.696_957_656_390_622;

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
/// # Examples
///
/// ```
/// use tonemapping::{BT2446A, LinearRGB, ToneMapper};
///
/// let display_linear = BT2446A.map(LinearRGB::new([10.0; 3]));
/// assert_eq!(display_linear.components(), [1.0; 3]);
/// ```
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
        let simd_len = colors.len() / BT2446_LANES * BT2446_LANES;
        bt2446a_planes(colors);
        colors.map_from(simd_len, self);
    }
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn bt2446a_batch(colors: &mut [LinearRGB]) {
    let (chunks, tail) = colors.as_chunks_mut::<BT2446_LANES>();
    debug_assert!(
        tail.is_empty(),
        "BT.2446 SIMD input must contain complete chunks"
    );

    for chunk in chunks {
        let components = [0, 1, 2].map(|channel| {
            F64x16::from_array(std::array::from_fn(|lane| {
                f64::from(chunk[lane].0[channel])
            }))
        });

        let mapped = bt2446a_simd(&components);

        for lane in 0..BT2446_LANES {
            chunk[lane] = LinearRGB([mapped[0][lane], mapped[1][lane], mapped[2][lane]]);
        }
    }
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn bt2446a_planes(colors: &mut LinearRGBPlanes) {
    let [red, green, blue] = colors.channels_mut();
    let [red_chunks, green_chunks, blue_chunks] = [red, green, blue].map(|channel| {
        let (chunks, _) = channel.as_chunks_mut::<BT2446_LANES>();
        chunks
    });

    for ((red, green), blue) in red_chunks.iter_mut().zip(green_chunks).zip(blue_chunks) {
        let components =
            [*red, *green, *blue].map(|channel| F64x16::from_array(channel.map(f64::from)));
        let mapped = bt2446a_simd(&components);

        *red = mapped[0];
        *green = mapped[1];
        *blue = mapped[2];
    }
}

#[inline]
fn bt2446a_simd(components: &[F64x16; 3]) -> [[f32; BT2446_LANES]; 3] {
    let zero = F64x16::splat(0.0);
    let one = F64x16::splat(1.0);

    let nonlinear = components.map(|component| {
        let normalized = (component / F64x16::splat(HDR_TO_SDR_PEAK_RATIO)).simd_clamp(zero, one);
        (normalized.log2() * F64x16::splat(1.0 / 2.4)).exp2()
    });

    let input_luma = F64x16::splat(BT2020_LUMA[0]) * nonlinear[0]
        + F64x16::splat(BT2020_LUMA[1]) * nonlinear[1]
        + F64x16::splat(BT2020_LUMA[2]) * nonlinear[2];

    let perceptual_luma =
        (one + F64x16::splat(RHO_HDR - 1.0) * input_luma).ln() / F64x16::splat(RHO_HDR.ln());

    let compressed_luma = perceptual_luma.simd_le(F64x16::splat(0.739_9)).select(
        F64x16::splat(1.077_0) * perceptual_luma,
        perceptual_luma.simd_lt(F64x16::splat(0.990_9)).select(
            F64x16::splat(-1.151_0) * perceptual_luma * perceptual_luma
                + F64x16::splat(2.781_1) * perceptual_luma
                - F64x16::splat(0.630_2),
            F64x16::splat(0.5) * perceptual_luma + F64x16::splat(0.5),
        ),
    );

    let output_luma = ((compressed_luma * F64x16::splat(RHO_SDR.ln())).exp() - one)
        / F64x16::splat(RHO_SDR - 1.0);

    let color_scale = input_luma
        .simd_eq(zero)
        .select(zero, output_luma / (F64x16::splat(1.1) * input_luma));

    let blue_difference = color_scale * (nonlinear[2] - input_luma) / F64x16::splat(CB_DIVISOR);
    let red_difference = color_scale * (nonlinear[0] - input_luma) / F64x16::splat(CR_DIVISOR);
    let adjusted_luma = output_luma - F64x16::splat(0.1) * red_difference.simd_max(zero);

    let output_nonlinear = [
        adjusted_luma + F64x16::splat(CR_DIVISOR) * red_difference,
        adjusted_luma
            - F64x16::splat(BT2020_LUMA[2] * CB_DIVISOR / BT2020_LUMA[1]) * blue_difference
            - F64x16::splat(BT2020_LUMA[0] * CR_DIVISOR / BT2020_LUMA[1]) * red_difference,
        adjusted_luma + F64x16::splat(CB_DIVISOR) * blue_difference,
    ];

    output_nonlinear.map(|component| {
        let bounded = component.simd_clamp(zero, one);

        (bounded.log2() * F64x16::splat(2.4))
            .exp2()
            .cast::<f32>()
            .to_array()
    })
}

fn bt2446a(color: LinearRGB) -> LinearRGB {
    let nonlinear = color.components_f64().map(|component| {
        (component / HDR_TO_SDR_PEAK_RATIO)
            .clamp(0.0, 1.0)
            .powf(1.0 / 2.4)
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

    LinearRGB::displayable(output_nonlinear.map(|component| component.clamp(0.0, 1.0).powf(2.4)))
}

fn bt2446a_luma(input_luma: f64) -> f64 {
    let perceptual_luma = (1.0 + (RHO_HDR - 1.0) * input_luma).ln() / RHO_HDR.ln();

    let compressed_luma = if perceptual_luma <= 0.739_9 {
        1.077_0 * perceptual_luma
    } else if perceptual_luma < 0.990_9 {
        -1.151_0 * perceptual_luma.powi(2) + 2.781_1 * perceptual_luma - 0.630_2
    } else {
        0.5 * perceptual_luma + 0.5
    };

    (RHO_SDR.powf(compressed_luma) - 1.0) / (RHO_SDR - 1.0)
}
