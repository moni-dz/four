//! Native JPEG XR pixel-layout description and per-pixel sample decode, including the PQ EOTF,
//! packed BGR101010, half-float, fixed-point, and RGBE sample paths (each with a scalar and SIMD
//! twin where the hot pixel-writing loops in `normalize` need it).

use std::simd::{
    Simd, StdFloat,
    num::{SimdFloat, SimdUint},
};
use std::sync::LazyLock;

use tonemapping::{exp2, log2};
use zerocopy::FromBytes;

#[cfg(test)]
use super::F32x4;
use super::{JPEGXRError, Result, SC_RGB_REFERENCE_WHITE_NITS, error};

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

    (
        [sample(0), sample(1), sample(2)],
        normalize_alpha(sample(3)),
    )
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
        [
            Simd::from_array(red),
            Simd::from_array(green),
            Simd::from_array(blue),
        ],
        Simd::from_array(alpha),
    )
}

pub(super) fn decode_bgr101010(pixel: &[u8]) -> [f32; 3] {
    decode_bgr101010_simd(Simd::<u32, 1>::splat(read_sample(pixel))).map(|channel| channel[0])
}

// All 10-bit PQ inputs fit in 4 KiB. Evaluate the existing transfer function once,
// preserving its exact f32 results instead of repeating log2/exp2 for every pixel.
static PQ_10BIT: LazyLock<[f32; 1024]> = LazyLock::new(|| {
    std::array::from_fn(|code| {
        let encoded = f32::from(u16::try_from(code).expect("10-bit PQ code")) * (1.0 / 1023.0);
        pq_to_linear_simd(Simd::<f32, 1>::splat(encoded))[0]
    })
});

#[inline]
pub(super) fn decode_bgr101010_simd<const N: usize>(packed: Simd<u32, N>) -> [Simd<f32, N>; 3] {
    const MASK: u32 = 0x03ff;

    let linear =
        [20, 10, 0].map(|shift| gather_pq((packed >> Simd::splat(shift)) & Simd::splat(MASK)));
    rec2100_linear_to_scrgb_simd(linear)
}

/// Looks up `indexes` (each `0..1024`) in [`PQ_10BIT`].
///
/// `Simd::gather_or`'s index type is pointer-width, so the portable path pays for a
/// software-widened per-lane address computation and scalarizes instead of using a real
/// gather instruction. On x86-64/AVX2, where a hardware gather takes 32-bit indices
/// directly, call it explicitly instead.
#[inline]
#[expect(
    unsafe_code,
    reason = "AVX2 gather needs 32-bit indices, unreachable through the safe portable_simd API"
)]
fn gather_pq<const N: usize>(indexes: Simd<u32, N>) -> Simd<f32, N> {
    #[cfg(target_arch = "x86_64")]
    if N == 8 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: `N == 8` was just checked, so `Simd<u32, N>` and `Simd<u32, 8>` are the
        // same type at this monomorphization.
        let indexes = unsafe { core::mem::transmute_copy::<Simd<u32, N>, Simd<u32, 8>>(&indexes) };
        // SAFETY: AVX2 support was just confirmed at runtime.
        let gathered = unsafe { gather_pq_avx2(indexes) };
        // SAFETY: `N == 8` was just checked, so `Simd<f32, 8>` and `Simd<f32, N>` are the
        // same type at this monomorphization.
        return unsafe { core::mem::transmute_copy::<Simd<f32, 8>, Simd<f32, N>>(&gathered) };
    }

    Simd::gather_or(&*PQ_10BIT, indexes.cast(), Simd::splat(0.0))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(
    unsafe_code,
    reason = "AVX2 gather needs 32-bit indices, unreachable through the safe portable_simd API"
)]
unsafe fn gather_pq_avx2(indexes: Simd<u32, 8>) -> Simd<f32, 8> {
    use std::arch::x86_64::{__m256, __m256i, _mm256_i32gather_ps};

    // SAFETY: the caller (`gather_pq`) confirmed AVX2 support. `indexes` is masked to
    // `0..1024` by every caller of `gather_pq`, so every gathered address stays inside
    // `PQ_10BIT` (a 1024-entry, 4 KiB table).
    unsafe {
        let indexes: __m256i = core::mem::transmute(indexes);
        let table = PQ_10BIT.as_ptr();
        let gathered: __m256 = _mm256_i32gather_ps::<4>(table, indexes);
        core::mem::transmute(gathered)
    }
}

#[cfg(test)]
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

// REC2100_MAX_NITS / SC_RGB_REFERENCE_WHITE_NITS folded into the matrix at compile time so
// the runtime path skips a fourth multiply per output channel.
const REC2100_MAX_NITS: f32 = 10_000.0;
const REC2100_TO_SCRGB: [[f32; 3]; 3] = {
    const SCALE: f32 = REC2100_MAX_NITS / SC_RGB_REFERENCE_WHITE_NITS;
    [
        [1.660_491 * SCALE, -0.587_641 * SCALE, -0.072_850 * SCALE],
        [-0.124_550 * SCALE, 1.132_9 * SCALE, -0.008_349 * SCALE],
        [-0.018_151 * SCALE, -0.100_579 * SCALE, 1.118_73 * SCALE],
    ]
};

#[cfg(test)]
fn rec2100_pq_to_scrgb(encoded: [f32; 3]) -> [f32; 3] {
    let [red, green, blue, _padding] =
        pq_to_linear_simd(F32x4::from_array([encoded[0], encoded[1], encoded[2], 0.0])).to_array();

    REC2100_TO_SCRGB.map(|[r, g, b]| blue.mul_add(b, green.mul_add(g, red * r)))
}

#[inline]
fn rec2100_linear_to_scrgb_simd<const N: usize>(
    [red, green, blue]: [Simd<f32, N>; 3],
) -> [Simd<f32, N>; 3] {
    REC2100_TO_SCRGB.map(|[r, g, b]| {
        blue.mul_add(
            Simd::splat(b),
            green.mul_add(Simd::splat(g), red * Simd::splat(r)),
        )
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_lookup_preserves_every_10_bit_code_and_channel_order() {
        for start in (0..1024_u32).step_by(8) {
            let packed = std::array::from_fn(|lane| {
                let code = start + u32::try_from(lane).unwrap();
                (3 << 30) | (code << 20) | (((code * 37) & 1023) << 10) | (1023 - code)
            });
            let actual = decode_bgr101010_simd(Simd::<u32, 8>::from_array(packed));
            for (lane, pixel) in packed.into_iter().enumerate() {
                let bytes = pixel.to_ne_bytes();
                let expected = rec2100_pq_to_scrgb(unpack_bgr101010(&bytes));
                assert_eq!(
                    actual.map(|channel| channel[lane].to_bits()),
                    expected.map(f32::to_bits),
                );
                assert_eq!(
                    decode_bgr101010(&bytes).map(f32::to_bits),
                    expected.map(f32::to_bits),
                );
            }
        }
    }
}
