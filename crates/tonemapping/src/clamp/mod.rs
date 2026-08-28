use multiversion::multiversion;

use super::{LinearRGB, LinearRGBPlanes, ToneMapper, WhitePoint, display_component_f32};
use crate::simd::{COLOR_LANES, F32x8, map_colors, map_planes};

/// Clamps every component to the displayable range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Clamp;

impl ToneMapper for Clamp {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        clamp_color(color)
    }

    #[inline]
    fn map_in_place(&self, colors: &mut [LinearRGB]) {
        for color in colors {
            *color = clamp_color(*color);
        }
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        map_planes(colors, COLOR_LANES, clamp_batch, self);
    }
}

#[multiversion(targets = "simd")]
fn clamp_batch(colors: &mut LinearRGBPlanes) {
    map_colors(colors, |components| components);
}

#[inline]
fn clamp_color(color: LinearRGB) -> LinearRGB {
    LinearRGB(color.0.map(display_component_f32))
}

/// Scales a scene white point to one before clamping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaledClamp {
    white_point: WhitePoint,
}

impl ScaledClamp {
    /// Creates a scaled clamp that maps `white_point` to one.
    #[must_use]
    pub const fn new(white_point: WhitePoint) -> Self {
        Self { white_point }
    }

    /// Returns the scene white point mapped to the display maximum.
    #[must_use]
    pub const fn white_point(self) -> WhitePoint {
        self.white_point
    }
}

impl ToneMapper for ScaledClamp {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let divisor = self.white_point.level();
        LinearRGB::displayable(color.components().map(|component| component / divisor))
    }

    #[inline]
    fn map_planes_in_place(&self, colors: &mut LinearRGBPlanes) {
        let divisor = self.white_point.level();
        map_planes(
            colors,
            COLOR_LANES,
            |colors| scaled_clamp_batch(colors, divisor),
            self,
        );
    }
}

#[multiversion(targets = "simd")]
fn scaled_clamp_batch(colors: &mut LinearRGBPlanes, divisor: f32) {
    let divisor = F32x8::splat(divisor);

    map_colors(colors, |components| {
        components.map(|component| component / divisor)
    });
}
