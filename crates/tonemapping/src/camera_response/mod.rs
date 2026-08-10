use std::fmt;

use super::{LinearRGB, ToneMapper, WhitePoint};

/// Stores a normalized camera response function sampled as a lookup table.
///
/// Irradiance samples must run from zero to one and be strictly increasing. Intensity samples must
/// run from zero to one and be nondecreasing. Mapping uses linear interpolation between adjacent
/// samples.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraResponse {
    irradiance: Box<[f32]>,
    intensity: Box<[f32]>,
}

impl CameraResponse {
    /// Creates a validated camera response from corresponding sample arrays.
    ///
    /// # Errors
    ///
    /// Returns [`CameraResponseError`] when the arrays have different lengths, contain fewer than
    /// two samples, do not span normalized black through white, contain non-finite or out-of-range
    /// values, or are not ordered as required.
    pub fn new(
        irradiance: impl Into<Box<[f32]>>,
        intensity: impl Into<Box<[f32]>>,
    ) -> Result<Self, CameraResponseError> {
        let irradiance = irradiance.into();
        let intensity = intensity.into();

        if irradiance.len() != intensity.len() {
            return Err(CameraResponseError::LengthMismatch {
                irradiance: irradiance.len(),
                intensity: intensity.len(),
            });
        }
        if irradiance.len() < 2 {
            return Err(CameraResponseError::TooFewSamples(irradiance.len()));
        }

        for (index, sample) in irradiance.iter().copied().enumerate() {
            if !sample.is_finite() || !(0.0..=1.0).contains(&sample) {
                return Err(CameraResponseError::IrradianceOutOfRange(index));
            }
            if index > 0 && sample <= irradiance[index - 1] {
                return Err(CameraResponseError::IrradianceNotIncreasing(index));
            }
        }

        for (index, sample) in intensity.iter().copied().enumerate() {
            if !sample.is_finite() || !(0.0..=1.0).contains(&sample) {
                return Err(CameraResponseError::IntensityOutOfRange(index));
            }
            if index > 0 && sample < intensity[index - 1] {
                return Err(CameraResponseError::IntensityDecreases(index));
            }
        }

        #[allow(
            clippy::float_cmp,
            reason = "normalized camera curves must contain the exact conventional endpoints"
        )]
        if irradiance[0] != 0.0 || irradiance[irradiance.len() - 1] != 1.0 {
            return Err(CameraResponseError::IrradianceEndpoints);
        }
        #[allow(
            clippy::float_cmp,
            reason = "normalized camera curves must contain the exact conventional endpoints"
        )]
        if intensity[0] != 0.0 || intensity[intensity.len() - 1] != 1.0 {
            return Err(CameraResponseError::IntensityEndpoints);
        }

        Ok(Self {
            irradiance,
            intensity,
        })
    }

    /// Returns the number of samples in the response curve.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.irradiance.len()
    }

    /// Creates a tone mapper whose horizontal curve extent is `white_point`.
    ///
    /// The article calls this parameter "ISO", but it acts as a linear-light white point: larger
    /// values darken the same input color.
    #[must_use]
    pub const fn tone_mapper(&self, white_point: WhitePoint) -> CameraToneMapper<'_> {
        CameraToneMapper {
            response: self,
            white_point,
        }
    }

    fn intensity_at(&self, irradiance: f64) -> f64 {
        match self
            .irradiance
            .binary_search_by(|sample| f64::from(*sample).total_cmp(&irradiance))
        {
            Ok(index) => f64::from(self.intensity[index]),
            Err(0) => f64::from(self.intensity[0]),
            Err(upper) if upper == self.irradiance.len() => f64::from(self.intensity[upper - 1]),
            Err(upper) => {
                let lower = upper - 1;
                let low_irradiance = f64::from(self.irradiance[lower]);
                let high_irradiance = f64::from(self.irradiance[upper]);
                let position = (irradiance - low_irradiance) / (high_irradiance - low_irradiance);
                let low_intensity = f64::from(self.intensity[lower]);
                let high_intensity = f64::from(self.intensity[upper]);
                low_intensity * (1.0 - position) + high_intensity * position
            }
        }
    }
}

/// Maps colors through a sampled camera response curve.
#[derive(Clone, Copy, Debug)]
pub struct CameraToneMapper<'a> {
    response: &'a CameraResponse,
    white_point: WhitePoint,
}

impl CameraToneMapper<'_> {
    /// Returns the scene level normalized to the end of the response curve.
    #[must_use]
    pub const fn white_point(self) -> WhitePoint {
        self.white_point
    }
}

impl ToneMapper for CameraToneMapper<'_> {
    #[inline]
    fn map(&self, color: LinearRGB) -> LinearRGB {
        let white = f64::from(self.white_point.level());
        let mapped = color.components_f64().map(|component| {
            let irradiance = (component / white).clamp(0.0, 1.0);
            self.response.intensity_at(irradiance)
        });
        LinearRGB::displayable(mapped)
    }
}

/// Describes invalid camera-response lookup-table data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraResponseError {
    /// The irradiance and intensity arrays have different lengths.
    LengthMismatch {
        /// Number of irradiance samples.
        irradiance: usize,
        /// Number of intensity samples.
        intensity: usize,
    },
    /// The curve contains fewer than two samples.
    TooFewSamples(usize),
    /// An irradiance sample is non-finite or outside `0.0..=1.0`.
    IrradianceOutOfRange(usize),
    /// An irradiance sample does not exceed its predecessor.
    IrradianceNotIncreasing(usize),
    /// An intensity sample is non-finite or outside `0.0..=1.0`.
    IntensityOutOfRange(usize),
    /// An intensity sample is lower than its predecessor.
    IntensityDecreases(usize),
    /// The irradiance domain does not start at zero and end at one.
    IrradianceEndpoints,
    /// The intensity range does not start at zero and end at one.
    IntensityEndpoints,
}

impl fmt::Display for CameraResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LengthMismatch {
                irradiance,
                intensity,
            } => write!(
                f,
                "camera response has {irradiance} irradiance samples and {intensity} intensity samples"
            ),
            Self::TooFewSamples(count) => {
                write!(f, "camera response requires two samples, got {count}")
            }
            Self::IrradianceOutOfRange(index) => write!(
                f,
                "camera irradiance sample {index} is not finite normalized data"
            ),
            Self::IrradianceNotIncreasing(index) => {
                write!(f, "camera irradiance sample {index} does not increase")
            }
            Self::IntensityOutOfRange(index) => write!(
                f,
                "camera intensity sample {index} is not finite normalized data"
            ),
            Self::IntensityDecreases(index) => {
                write!(f, "camera intensity sample {index} decreases")
            }
            Self::IrradianceEndpoints => {
                f.write_str("camera irradiance must start at zero and end at one")
            }
            Self::IntensityEndpoints => {
                f.write_str("camera intensity must start at zero and end at one")
            }
        }
    }
}

impl std::error::Error for CameraResponseError {}
