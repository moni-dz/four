use super::{LinearRGB, ToneMapper};

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
}

fn bt2446a(color: LinearRGB) -> LinearRGB {
    const HDR_TO_SDR_PEAK_RATIO: f64 = 10.0;
    const BT2020_LUMA: [f64; 3] = [0.262_7, 0.678_0, 0.059_3];
    const CB_DIVISOR: f64 = 1.881_4;
    const CR_DIVISOR: f64 = 1.474_6;

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
    const RHO_HDR: f64 = 13.259_797_918_583_32;
    const RHO_SDR: f64 = 5.696_957_656_390_622;

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
