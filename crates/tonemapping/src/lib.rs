//! Maps high-dynamic-range linear RGB colors into a displayable range.
//!
//! Tone-mapping operators accept [`LinearRgb`] values whose components are relative linear-light
//! levels. A component of `1.0` conventionally represents the target display's reference white.
//! Operators return finite components in the inclusive range `0.0..=1.0`; transfer encoding and
//! integer quantization remain the caller's responsibility.

const HDR_COMPONENT_MAX: f32 = 65_504.0;
const REC709_LUMINANCE: [f32; 3] = [0.212_6, 0.715_2, 0.072_2];

/// Stores a finite, nonnegative linear RGB color.
///
/// Construction replaces negative and `NaN` components with zero. Positive infinity and finite
/// values above the largest finite binary16 value are limited to `65_504`, which keeps subsequent
/// tone-mapping arithmetic bounded for common HDR interchange formats.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinearRgb([f32; 3]);

impl LinearRgb {
    /// Creates a sanitized linear RGB color from `components`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::LinearRgb;
    ///
    /// let color = LinearRgb::new([2.0, -1.0, f32::NAN]);
    /// assert_eq!(color.components(), [2.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    pub fn new(components: [f32; 3]) -> Self {
        Self(components.map(sanitize_component))
    }

    /// Returns the red, green, and blue components in that order.
    #[must_use]
    pub const fn components(self) -> [f32; 3] {
        self.0
    }

    /// Returns linear Rec. 709 luminance.
    #[must_use]
    pub fn luminance(self) -> f32 {
        REC709_LUMINANCE[0] * self.0[0]
            + REC709_LUMINANCE[1] * self.0[1]
            + REC709_LUMINANCE[2] * self.0[2]
    }

    fn displayable(components: [f32; 3]) -> Self {
        Self(components.map(|component| component.clamp(0.0, 1.0)))
    }
}

impl From<[f32; 3]> for LinearRgb {
    fn from(components: [f32; 3]) -> Self {
        Self::new(components)
    }
}

impl From<LinearRgb> for [f32; 3] {
    fn from(color: LinearRgb) -> Self {
        color.components()
    }
}

/// Maps a linear HDR color into display-linear RGB.
pub trait ToneMapper {
    /// Maps `color` into finite `0.0..=1.0` display-linear components.
    #[must_use]
    fn map(&self, color: LinearRgb) -> LinearRgb;
}

/// Identifies a positive finite scene level that maps to display white.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhitePoint(f32);

impl WhitePoint {
    /// Creates a white point from a positive finite linear-light level.
    ///
    /// Returns `None` when `level` is zero, negative, or non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::WhitePoint;
    ///
    /// assert_eq!(WhitePoint::new(4.0).map(WhitePoint::level), Some(4.0));
    /// assert!(WhitePoint::new(0.0).is_none());
    /// ```
    #[must_use]
    pub fn new(level: f32) -> Option<Self> {
        (level.is_finite() && level > 0.0).then_some(Self(level))
    }

    /// Returns the relative linear-light level represented by this white point.
    #[must_use]
    pub const fn level(self) -> f32 {
        self.0
    }
}

/// Clamps every component to the displayable range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Clamp;

impl ToneMapper for Clamp {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        LinearRgb::displayable(color.components())
    }
}

/// Compresses highlights above a fixed knee using the content peak.
///
/// This operator preserves input luminance below `0.75`, maps `max_content_level` to `1.0`, and
/// scales all channels together before fitting the result into the target gamut. It exists for the
/// decoder integration that predates the named operators provided by this crate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearShoulder {
    max_content_level: f32,
}

impl LinearShoulder {
    /// Creates an operator for a finite, nonnegative `max_content_level`.
    ///
    /// Returns `None` for negative or non-finite levels.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonemapping::LinearShoulder;
    ///
    /// assert!(LinearShoulder::new(4.0).is_some());
    /// assert!(LinearShoulder::new(f32::NAN).is_none());
    /// ```
    #[must_use]
    pub fn new(max_content_level: f32) -> Option<Self> {
        (max_content_level.is_finite() && max_content_level >= 0.0)
            .then_some(Self { max_content_level })
    }
}

impl ToneMapper for LinearShoulder {
    fn map(&self, color: LinearRgb) -> LinearRgb {
        const KNEE: f32 = 0.75;

        let luminance = color.luminance();
        if luminance == 0.0 {
            return LinearRgb::default();
        }

        let mapped_luminance = if self.max_content_level <= 1.0 || luminance <= KNEE {
            luminance
        } else {
            let input_range = self.max_content_level - KNEE;
            let output_range = 1.0 - KNEE;
            let softness = input_range * output_range / (self.max_content_level - 1.0);
            let distance = luminance - KNEE;
            KNEE + distance / (1.0 + distance / softness)
        };

        let scale = mapped_luminance / luminance;
        let mut mapped = color.0.map(|component| component * scale);
        let peak = mapped.into_iter().fold(0.0_f32, f32::max);
        if peak > 1.0 {
            mapped = mapped.map(|component| component / peak);
        }

        LinearRgb::displayable(mapped)
    }
}

fn sanitize_component(component: f32) -> f32 {
    if component.is_nan() || component <= 0.0 {
        0.0
    } else {
        component.min(HDR_COMPONENT_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shoulder_compresses_hdr_white_into_sdr() {
        let mapper = LinearShoulder::new(4.0).unwrap();

        assert_eq!(mapper.map(LinearRgb::default()), LinearRgb::default());
        let reference_white = mapper.map(LinearRgb::new([1.0; 3]));
        let hdr_white = mapper.map(LinearRgb::new([4.0; 3]));

        assert!(reference_white.components()[0] > 0.85);
        assert_eq!(hdr_white.components(), [1.0; 3]);
    }

    #[test]
    fn shoulder_preserves_channel_order_during_gamut_compression() {
        let mapper = LinearShoulder::new(4.0).unwrap();
        let [red, green, blue] = mapper.map(LinearRgb::new([4.0, 2.0, 1.0])).components();

        assert_eq!(red, 1.0);
        assert!(red > green);
        assert!(green > blue);
    }

    #[test]
    fn clamp_discards_values_outside_the_display_range() {
        let color = Clamp.map(LinearRgb::new([0.25, 1.0, 4.0]));

        assert_eq!(color.components(), [0.25, 1.0, 1.0]);
    }

}
