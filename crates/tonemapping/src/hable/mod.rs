use super::{LinearRGB, ToneMapper};

/// Applies John Hable's Uncharted 2 filmic curve component-wise.
///
/// The operator includes the article's exposure bias of two and normalizes the curve at its `11.2`
/// reference input. Consequently, a scene component of `5.6` maps to display white.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Hable;

impl ToneMapper for Hable {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let white_scale = 1.0 / hable_partial(11.2);
        LinearRGB::displayable(
            color
                .components_f64()
                .map(|component| hable_partial(component * 2.0) * white_scale),
        )
    }
}

fn hable_partial(value: f64) -> f64 {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;

    ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - E / F
}
