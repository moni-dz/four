//! Native JPEG XR pixel-layout description and per-pixel sample decode, including the PQ EOTF,
//! packed BGR101010, half-float, fixed-point, and RGBE sample paths (each with a scalar and SIMD
//! twin where the hot pixel-writing loops in `normalize` need it).

use std::simd::{
    Simd,
    num::{SimdFloat, SimdUint},
};

use tonemapping::{exp2, log2};
use zerocopy::FromBytes;

use super::{F32x4, JPEGXRError, Result, SC_RGB_REFERENCE_WHITE_NITS, error};

#[expect(
    dead_code,
    reason = "normalization primitives remain available for future decoder profiles"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SampleEncoding {
    Fixed16,
    Fixed32,
    Float16,
    Float32,
    PackedBGR101010,
    RGBE,
    Unsigned8,
    Unsigned16,
}

impl SampleEncoding {
    pub(super) const fn bytes(self) -> usize {
        match self {
            Self::Unsigned8 | Self::RGBE => 1,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 2,
            Self::Fixed32 | Self::Float32 | Self::PackedBGR101010 => 4,
        }
    }

    pub(super) const fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Fixed16
                | Self::Fixed32
                | Self::Float16
                | Self::Float32
                | Self::PackedBGR101010
                | Self::RGBE
        )
    }

    pub(super) const fn bits_per_channel(self) -> u8 {
        match self {
            Self::Unsigned8 | Self::RGBE => 8,
            Self::PackedBGR101010 => 10,
            Self::Unsigned16 | Self::Fixed16 | Self::Float16 => 16,
            Self::Fixed32 | Self::Float32 => 32,
        }
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields describe independent WIC pixel-layout properties"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PixelLayout {
    pub(super) encoding: SampleEncoding,
    pub(super) color_channels: usize,
    pub(super) source_channels: usize,
    pub(super) bytes_per_pixel: usize,
    pub(super) has_alpha: bool,
    pub(super) premultiplied_alpha: bool,
    pub(super) blue_first: bool,
    pub(super) source_is_bgr: bool,
}

impl PixelLayout {
    pub(super) const fn bgr101010() -> Self {
        Self {
            encoding: SampleEncoding::PackedBGR101010,
            color_channels: 3,
            source_channels: 3,
            bytes_per_pixel: 4,
            has_alpha: false,
            premultiplied_alpha: false,
            blue_first: false,
            source_is_bgr: true,
        }
    }

    pub(super) const fn rgba128_float() -> Self {
        Self {
            encoding: SampleEncoding::Float32,
            color_channels: 3,
            source_channels: 4,
            bytes_per_pixel: 16,
            has_alpha: true,
            premultiplied_alpha: false,
            blue_first: false,
            source_is_bgr: false,
        }
    }

    pub(super) fn row_stride(self, width: u32) -> Result<usize> {
        usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(self.bytes_per_pixel))
            .ok_or_else(|| error(JPEGXRError::Output("JPEG XR row stride exceeds usize")))
    }

    pub(super) fn read_pixel(self, pixel: &[u8]) -> Result<([f32; 3], f32)> {
        invariant_eq!(pixel.len(), self.bytes_per_pixel);

        if self.encoding == SampleEncoding::RGBE {
            return Ok((decode_rgbe(pixel)?, 1.0));
        }

        if self.encoding == SampleEncoding::PackedBGR101010 {
            return Ok((decode_bgr101010(pixel), 1.0));
        }

        let sample = |channel: usize| -> Result<f32> {
            invariant!(channel < self.source_channels);

            let start = channel * self.encoding.bytes();
            let end = start + self.encoding.bytes();
            let bytes = pixel.get(start..end).ok_or_else(|| {
                error(JPEGXRError::Output(
                    "JPEG XR sample exceeds its pixel stride",
                ))
            })?;

            Ok(decode_sample(bytes, self.encoding))
        };

        let mut color = if self.color_channels == 1 {
            let gray = sample(0)?;
            [gray, gray, gray]
        } else {
            [sample(0)?, sample(1)?, sample(2)?]
        };

        if self.blue_first && self.color_channels == 3 {
            color.swap(0, 2);
        }

        let alpha = if self.has_alpha {
            normalize_alpha(sample(self.color_channels)?)
        } else {
            1.0
        };

        if self.premultiplied_alpha {
            if alpha > 0.0 {
                color = color.map(|channel| channel / alpha);
            } else {
                color.fill(0.0);
            }
        }

        Ok((color, alpha))
    }
}

pub(super) fn decode_sample(bytes: &[u8], encoding: SampleEncoding) -> f32 {
    invariant_eq!(bytes.len(), encoding.bytes());

    match encoding {
        SampleEncoding::Unsigned8 => f32::from(read_sample::<u8>(bytes)) / f32::from(u8::MAX),
        SampleEncoding::Unsigned16 => f32::from(read_sample::<u16>(bytes)) / f32::from(u16::MAX),
        SampleEncoding::Fixed16 => f32::from(read_sample::<i16>(bytes)) / 8192.0,
        SampleEncoding::Fixed32 => fixed32_to_f32(read_sample::<i32>(bytes)),
        SampleEncoding::Float16 => half_to_f32(read_sample::<u16>(bytes)),
        SampleEncoding::Float32 => read_sample::<f32>(bytes),
        SampleEncoding::PackedBGR101010 => {
            unreachable!("packed BGR101010 pixels are decoded as a unit")
        }
        SampleEncoding::RGBE => {
            unreachable!("RGBE pixels are decoded as a unit")
        }
    }
}

pub(super) fn decode_rgba128_float(pixel: &[u8]) -> ([f32; 3], f32) {
    invariant_eq!(pixel.len(), PixelLayout::rgba128_float().bytes_per_pixel);

    let sample = |channel: usize| read_sample::<f32>(&pixel[channel * 4..channel * 4 + 4]);

    ([sample(0), sample(1), sample(2)], normalize_alpha(sample(3)))
}

#[inline]
pub(super) fn decode_rgba128_float_simd<const N: usize>(
    chunk: &[[u8; 16]; N],
) -> ([Simd<f32, N>; 3], Simd<f32, N>) {
    let mut red = [0.0f32; N];
    let mut green = [0.0f32; N];
    let mut blue = [0.0f32; N];
    let mut alpha = [0.0f32; N];

    for (lane, pixel) in chunk.iter().enumerate() {
        let (color, a) = decode_rgba128_float(pixel);
        [red[lane], green[lane], blue[lane]] = color;
        alpha[lane] = a;
    }

    (
        [Simd::from_array(red), Simd::from_array(green), Simd::from_array(blue)],
        Simd::from_array(alpha),
    )
}

pub(super) fn decode_bgr101010(pixel: &[u8]) -> [f32; 3] {
    rec2100_pq_to_scrgb(unpack_bgr101010(pixel))
}

#[inline]
pub(super) fn decode_bgr101010_simd<const N: usize>(packed: Simd<u32, N>) -> [Simd<f32, N>; 3] {
    const MASK: u32 = 0x03ff;
    const SCALE: f32 = 1.0 / 1023.0;

    let encoded = [20, 10, 0].map(|shift| {
        ((packed >> Simd::splat(shift)) & Simd::splat(MASK)).cast::<f32>() * Simd::splat(SCALE)
    });
    rec2100_pq_to_scrgb_simd(encoded)
}

pub(super) fn unpack_bgr101010(pixel: &[u8]) -> [f32; 3] {
    const MASK: u32 = 0x03ff;
    const SCALE: f32 = 1.0 / 1023.0;

    invariant_eq!(pixel.len(), SampleEncoding::PackedBGR101010.bytes());

    let packed = read_sample::<u32>(pixel);
    let channel = |shift| {
        let sample = u16::try_from((packed >> shift) & MASK)
            .expect("a masked 10-bit JPEG XR sample fits u16");
        f32::from(sample) * SCALE
    };

    [channel(20), channel(10), channel(0)]
}

fn rec2100_pq_to_scrgb(encoded: [f32; 3]) -> [f32; 3] {
    const REC2100_MAX_NITS: f32 = 10_000.0;
    const SCALE: f32 = REC2100_MAX_NITS / SC_RGB_REFERENCE_WHITE_NITS;

    let [red, green, blue, _padding] =
        pq_to_linear_simd(F32x4::from_array([encoded[0], encoded[1], encoded[2], 0.0])).to_array();

    [
        (1.660_491 * red - 0.587_641 * green - 0.072_850 * blue) * SCALE,
        (-0.124_550 * red + 1.132_9 * green - 0.008_349 * blue) * SCALE,
        (-0.018_151 * red - 0.100_579 * green + 1.118_73 * blue) * SCALE,
    ]
}

#[inline]
pub(super) fn rec2100_pq_to_scrgb_simd<const N: usize>(
    encoded: [Simd<f32, N>; 3],
) -> [Simd<f32, N>; 3] {
    const REC2100_MAX_NITS: f32 = 10_000.0;
    const SCALE: f32 = REC2100_MAX_NITS / SC_RGB_REFERENCE_WHITE_NITS;

    let [red, green, blue] = encoded.map(pq_to_linear_simd);
    [
        (Simd::splat(1.660_491) * red
            - Simd::splat(0.587_641) * green
            - Simd::splat(0.072_850) * blue)
            * Simd::splat(SCALE),
        (Simd::splat(-0.124_550) * red + Simd::splat(1.132_9) * green
            - Simd::splat(0.008_349) * blue)
            * Simd::splat(SCALE),
        (Simd::splat(-0.018_151) * red - Simd::splat(0.100_579) * green
            + Simd::splat(1.118_73) * blue)
            * Simd::splat(SCALE),
    ]
}

#[cfg(test)]
pub(super) fn pq_to_linear(encoded: f32) -> f32 {
    const INVERSE_M1: f32 = 16_384.0 / 2_610.0;
    const INVERSE_M2: f32 = 32.0 / 2_523.0;
    const C1: f32 = 3_424.0 / 4_096.0;
    const C2: f32 = 2_413.0 / 128.0;
    const C3: f32 = 2_392.0 / 128.0;

    let powered = encoded.powf(INVERSE_M2);
    ((powered - C1).max(0.0) / (C2 - C3 * powered)).powf(INVERSE_M1)
}

#[inline]
pub(super) fn pq_to_linear_simd<const N: usize>(encoded: Simd<f32, N>) -> Simd<f32, N> {
    const INVERSE_M1: f32 = 16_384.0 / 2_610.0;
    const INVERSE_M2: f32 = 32.0 / 2_523.0;
    const C1: f32 = 3_424.0 / 4_096.0;
    const C2: f32 = 2_413.0 / 128.0;
    const C3: f32 = 2_392.0 / 128.0;

    let powered = exp2(log2(encoded) * Simd::splat(INVERSE_M2));
    let ratio = (powered - Simd::splat(C1)).simd_max(Simd::splat(0.0))
        / (Simd::splat(C2) - Simd::splat(C3) * powered);
    exp2(log2(ratio) * Simd::splat(INVERSE_M1))
}

fn read_sample<T: FromBytes + Sized>(bytes: &[u8]) -> T {
    T::read_from_bytes(bytes).expect("JPEG XR sample length must match its encoding")
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "s7.24 fixed-point values are intentionally converted to f32 for tone mapping"
)]
fn fixed32_to_f32(value: i32) -> f32 {
    (f64::from(value) / 16_777_216.0) as f32
}

/// Widens an IEEE 754 binary16 sample to `f32`.
pub(super) fn half_to_f32(bits: u16) -> f32 {
    f32::from(f16::from_bits(bits))
}

fn decode_rgbe(pixel: &[u8]) -> Result<[f32; 3]> {
    if pixel.len() < 4 {
        return Err(error(JPEGXRError::Output(
            "JPEG XR RGBE pixel is shorter than four bytes",
        )));
    }

    let exponent = pixel[3];
    if exponent == 0 {
        return Ok([0.0; 3]);
    }

    let scale = 2.0_f32.powi(i32::from(exponent) - 136);
    Ok([
        f32::from(pixel[0]) * scale,
        f32::from(pixel[1]) * scale,
        f32::from(pixel[2]) * scale,
    ])
}

pub(super) fn normalize_alpha(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
