//! Decodes HDR PNGs (`cICP` PQ or HLG) to linear scRGB.
//!
//! [PNG 3.0](https://www.w3.org/TR/png-3/#cICP-chunk) signals HDR through the `cICP` chunk. This
//! module accepts full-range RGB signals with BT.709, BT.2020, or Display-P3 primaries and either
//! the PQ (BT.2100 / SMPTE ST 2084) or HLG (BT.2100) transfer function. Samples become linear
//! Rec. 709 RGB where `1.0` is 80 cd/m^2, the same convention as JPEG XR's floating-point formats.
//! HLG is rendered for a 1000 cd/m^2 display with the BT.2100 system gamma of 1.2.

use std::io::Cursor;
use std::sync::LazyLock;

use ::png::{BitDepth, ColorType, Decoder, Info, Limits, Transformations};
use rayon::prelude::*;

use super::{CODEC_MEMORY_MAX, PNGError, PNGLimit, Result};
use super::{codec_error, error, has_signature, validate_dimensions};

const SC_RGB_REFERENCE_WHITE_NITS: f32 = 80.0;
const PQ_PEAK_NITS: f32 = 10_000.0;
const HLG_NOMINAL_PEAK_NITS: f32 = 1_000.0;
const HLG_SYSTEM_GAMMA: f32 = 1.2;

const PRIMARIES_BT709: u8 = 1;
const PRIMARIES_BT2020: u8 = 9;
const PRIMARIES_DISPLAY_P3: u8 = 12;
const TRANSFER_PQ: u8 = 16;
const TRANSFER_HLG: u8 = 18;

const BT2020_TO_BT709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641, -0.072_850],
    [-0.124_550, 1.132_9, -0.008_349],
    [-0.018_151, -0.100_579, 1.118_73],
];
const DISPLAY_P3_TO_BT709: [[f32; 3]; 3] = [
    [1.224_94, -0.224_94, 0.0],
    [-0.042_056, 1.042_056, 0.0],
    [-0.019_637, -0.078_636, 1.098_273],
];
const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A decoded HDR PNG: linear scRGB RGBA, `1.0` = 80 cd/m^2, straight alpha.
#[derive(Debug)]
pub struct HDRImage {
    width: u32,
    height: u32,
    rgba: Vec<f32>,
}

impl HDRImage {
    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns row-major `[r, g, b, a]` samples; color is linear scRGB and alpha is `0.0..=1.0`.
    #[must_use]
    pub fn rgba(&self) -> &[f32] {
        &self.rgba
    }

    /// Consumes the image and returns its row-major `[r, g, b, a]` samples.
    #[must_use]
    pub fn into_rgba(self) -> Vec<f32> {
        self.rgba
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Transfer {
    PQ,
    HLG,
}

#[derive(Clone, Copy, Debug)]
struct Profile {
    transfer: Transfer,
    to_rec709: [[f32; 3]; 3],
    /// Luma weights of the source primaries, used by the HLG OOTF.
    luma: [f32; 3],
}

fn profile(info: &Info<'_>) -> Option<Profile> {
    let cicp = info.coding_independent_code_points?;
    if cicp.matrix_coefficients != 0 || !cicp.is_video_full_range_image {
        return None;
    }

    let transfer = match cicp.transfer_function {
        TRANSFER_PQ => Transfer::PQ,
        TRANSFER_HLG => Transfer::HLG,
        _ => return None,
    };
    let (to_rec709, luma) = match cicp.color_primaries {
        PRIMARIES_BT709 => (IDENTITY, [0.2126, 0.7152, 0.0722]),
        PRIMARIES_BT2020 => (BT2020_TO_BT709, [0.2627, 0.6780, 0.0593]),
        PRIMARIES_DISPLAY_P3 => (DISPLAY_P3_TO_BT709, [0.2290, 0.6917, 0.0793]),
        _ => return None,
    };

    Some(Profile {
        transfer,
        to_rec709,
        luma,
    })
}

/// Returns whether `bytes` is a PNG whose `cICP` chunk declares a supported HDR signal.
///
/// Reads only the chunks before the image data, so it is cheap to call before choosing a decoder.
#[must_use]
pub fn is_hdr(bytes: &[u8]) -> bool {
    if !has_signature(bytes) {
        return false;
    }

    let mut decoder = Decoder::new(Cursor::new(bytes));
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    decoder
        .read_info()
        .is_ok_and(|reader| profile(reader.info()).is_some())
}

/// Decodes an HDR PNG to linear scRGB. See the [module documentation](self) for the conventions.
///
/// # Errors
///
/// Returns [`PNGError`] for malformed input, resource-limit failures, and PNGs without a supported
/// HDR `cICP` signal (check with [`is_hdr`]).
pub fn decode_hdr(bytes: &[u8]) -> Result<HDRImage> {
    if !has_signature(bytes) {
        return Err(error(PNGError::Signature));
    }

    let mut decoder = Decoder::new_with_limits(
        Cursor::new(bytes),
        Limits {
            bytes: CODEC_MEMORY_MAX,
        },
    );
    decoder.set_transformations(Transformations::EXPAND);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);

    let mut reader = decoder.read_info().map_err(codec_error)?;
    let profile = profile(reader.info()).ok_or_else(|| {
        error(PNGError::Output(
            "PNG does not declare a supported PQ or HLG cICP signal",
        ))
    })?;
    let (width, height) = (reader.info().width, reader.info().height);
    validate_dimensions(width, height)?;

    let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_source| error(PNGError::Output("PNG pixel count exceeds this platform")))?;
    // Sixteen-bit RGBA is the widest output the codec can produce.
    let max_output = pixel_count.saturating_mul(8);

    let output_size = reader.output_buffer_size().ok_or_else(|| {
        error(PNGError::Output(
            "PNG codec could not determine the decoded buffer size",
        ))
    })?;
    if output_size > max_output {
        return Err(error(PNGError::LimitExceeded(PNGLimit::DecodedBytes {
            actual: output_size,
            max: max_output,
        })));
    }

    let mut samples = vec![0; output_size];
    let output = reader.next_frame(&mut samples).map_err(codec_error)?;
    if output.width != width || output.height != height {
        return Err(error(PNGError::Output(
            "animated PNG subframes are not supported",
        )));
    }
    samples.truncate(output.buffer_size());

    let channels = match output.color_type {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => {
            return Err(error(PNGError::Output(
                "PNG palette was not expanded by the codec",
            )));
        }
    };
    let wide = match output.bit_depth {
        BitDepth::Eight => false,
        BitDepth::Sixteen => true,
        _ => {
            return Err(error(PNGError::Output(
                "PNG codec did not produce eight- or sixteen-bit samples",
            )));
        }
    };
    let bytes_per_pixel = channels * if wide { 2 } else { 1 };
    if samples.len() != pixel_count * bytes_per_pixel {
        return Err(error(PNGError::Output(
            "PNG codec returned an unexpected sample count",
        )));
    }

    let table = linearization_table(profile.transfer);
    let mut rgba = vec![0.0_f32; pixel_count * 4];

    samples
        .par_chunks(bytes_per_pixel * 4096)
        .zip(rgba.par_chunks_mut(4 * 4096))
        .for_each(|(source, target)| {
            for (pixel, out) in source
                .chunks_exact(bytes_per_pixel)
                .zip(target.chunks_exact_mut(4))
            {
                let sample = |index: usize| -> usize {
                    if wide {
                        usize::from(u16::from_be_bytes([pixel[index * 2], pixel[index * 2 + 1]]))
                    } else {
                        usize::from(pixel[index]) * 257
                    }
                };
                let alpha_index = (channels == 2 || channels == 4).then(|| channels - 1);
                let color_channels = channels - usize::from(alpha_index.is_some());
                let encoded = if color_channels == 1 {
                    [sample(0); 3]
                } else {
                    [sample(0), sample(1), sample(2)]
                };
                let alpha = alpha_index.map_or(1.0, |index| sample(index) as f32 / 65_535.0);

                let color = encoded.map(|code| table[code]);
                let color = scene_to_scrgb(profile, color);
                out.copy_from_slice(&[color[0], color[1], color[2], alpha]);
            }
        });

    Ok(HDRImage {
        width,
        height,
        rgba,
    })
}

/// Maps every 16-bit code to linear light: PQ to scRGB units (`1.0` = 80 nits), HLG to relative
/// scene light in `0.0..=1.0`.
fn linearization_table(transfer: Transfer) -> &'static [f32] {
    static PQ: LazyLock<Box<[f32]>> = LazyLock::new(|| build_table(pq_to_scrgb));
    static HLG: LazyLock<Box<[f32]>> = LazyLock::new(|| build_table(hlg_inverse_oetf));

    match transfer {
        Transfer::PQ => &PQ,
        Transfer::HLG => &HLG,
    }
}

fn build_table(curve: fn(f32) -> f32) -> Box<[f32]> {
    (0..=u16::MAX)
        .map(|code| curve(f32::from(code) / 65_535.0))
        .collect()
}

fn pq_to_scrgb(encoded: f32) -> f32 {
    const INVERSE_M1: f32 = 16_384.0 / 2_610.0;
    const INVERSE_M2: f32 = 32.0 / 2_523.0;
    const C1: f32 = 3_424.0 / 4_096.0;
    const C2: f32 = 2_413.0 / 128.0;
    const C3: f32 = 2_392.0 / 128.0;

    let powered = encoded.powf(INVERSE_M2);
    let normalized = ((powered - C1).max(0.0) / (C2 - C3 * powered)).powf(INVERSE_M1);
    normalized * (PQ_PEAK_NITS / SC_RGB_REFERENCE_WHITE_NITS)
}

fn hlg_inverse_oetf(encoded: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A;
    // 0.5 - A * ln(4A)
    const C: f32 = 0.559_910_7;

    if encoded <= 0.5 {
        encoded * encoded / 3.0
    } else {
        (((encoded - C) / A).exp() + B) / 12.0
    }
}

/// Applies the HLG OOTF when needed and converts to Rec. 709 primaries.
fn scene_to_scrgb(profile: Profile, color: [f32; 3]) -> [f32; 3] {
    let linear = match profile.transfer {
        Transfer::PQ => color,
        Transfer::HLG => {
            let luminance = profile
                .luma
                .iter()
                .zip(color)
                .map(|(weight, channel)| weight * channel)
                .sum::<f32>();
            let scale = HLG_NOMINAL_PEAK_NITS / SC_RGB_REFERENCE_WHITE_NITS
                * luminance.powf(HLG_SYSTEM_GAMMA - 1.0);
            color.map(|channel| channel * scale)
        }
    };

    profile
        .to_rec709
        .map(|row| row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_reference_points() {
        assert!(pq_to_scrgb(0.0).abs() < 1e-6);
        assert!((pq_to_scrgb(1.0) - 125.0).abs() < 0.05);
        // 203 cd/m^2 reference white encodes near 0.5807.
        assert!((pq_to_scrgb(0.5807) * 80.0 - 203.0).abs() < 1.5);
    }

    #[test]
    fn hlg_reference_points() {
        assert!(hlg_inverse_oetf(0.0).abs() < 1e-6);
        assert!((hlg_inverse_oetf(1.0) - 1.0).abs() < 1e-4);
        assert!((hlg_inverse_oetf(0.5) - 1.0 / 12.0).abs() < 1e-6);
    }

    fn encode(
        transfer: u8,
        primaries: u8,
        color: ColorType,
        samples: &[u8],
        width: u32,
        cicp: bool,
    ) -> Vec<u8> {
        let mut info = Info::with_size(width, 1);
        info.color_type = color;
        info.bit_depth = BitDepth::Sixteen;
        let mut out = Vec::new();
        let mut writer = ::png::Encoder::with_info(&mut out, info)
            .unwrap()
            .write_header()
            .unwrap();
        if cicp {
            // The encoder has no cICP support; the chunk must precede IDAT.
            writer
                .write_chunk(
                    ::png::chunk::ChunkType(*b"cICP"),
                    &[primaries, transfer, 0, 1],
                )
                .unwrap();
        }
        writer.write_image_data(samples).unwrap();
        writer.finish().unwrap();
        out
    }

    fn words(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect()
    }

    #[test]
    fn pq_bt2020_png_decodes_to_scrgb() {
        // Pixel 0: black. Pixel 1: PQ 1.0 on all channels (10,000 nits, white stays white).
        let samples = words(&[0, 0, 0, 65_535, 65_535, 65_535]);
        let png = encode(
            TRANSFER_PQ,
            PRIMARIES_BT2020,
            ColorType::Rgb,
            &samples,
            2,
            true,
        );

        assert!(is_hdr(&png));
        let image = decode_hdr(&png).unwrap();
        assert_eq!((image.width(), image.height()), (2, 1));

        let rgba = image.rgba();
        assert_eq!(&rgba[..4], &[0.0, 0.0, 0.0, 1.0]);
        for channel in &rgba[4..7] {
            assert!((channel - 125.0).abs() < 0.5, "{channel}");
        }
        assert_eq!(rgba[7], 1.0);
    }

    #[test]
    fn bt2020_primaries_are_converted_to_rec709() {
        // Pure BT.2020 red is outside Rec. 709: negative green and blue after conversion.
        let samples = words(&[40_000, 0, 0]);
        let png = encode(
            TRANSFER_PQ,
            PRIMARIES_BT2020,
            ColorType::Rgb,
            &samples,
            1,
            true,
        );
        let rgba = decode_hdr(&png).unwrap().into_rgba();
        assert!(rgba[0] > 0.0 && rgba[1] < 0.0 && rgba[2] < 0.0, "{rgba:?}");
    }

    #[test]
    fn hlg_png_with_alpha_keeps_straight_alpha() {
        let samples = words(&[65_535, 65_535, 65_535, 32_768]);
        let png = encode(
            TRANSFER_HLG,
            PRIMARIES_BT2020,
            ColorType::Rgba,
            &samples,
            1,
            true,
        );
        let rgba = decode_hdr(&png).unwrap().into_rgba();

        // HLG signal 1.0 is the 1000 nit nominal peak.
        for channel in &rgba[..3] {
            assert!((channel * 80.0 - 1000.0).abs() < 5.0, "{channel}");
        }
        assert!((rgba[3] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn pngs_without_a_supported_cicp_are_not_hdr() {
        let samples = words(&[1, 2, 3]);
        assert!(!is_hdr(&encode(0, 0, ColorType::Rgb, &samples, 1, false)));
        // sRGB-like transfer (13) is SDR.
        assert!(!is_hdr(&encode(
            13,
            PRIMARIES_BT709,
            ColorType::Rgb,
            &samples,
            1,
            true
        )));
        // Unsupported primaries.
        assert!(!is_hdr(&encode(
            TRANSFER_PQ,
            5,
            ColorType::Rgb,
            &samples,
            1,
            true
        )));
        assert!(decode_hdr(&encode(0, 0, ColorType::Rgb, &samples, 1, false)).is_err());
        assert!(!is_hdr(b"not a png"));
    }

    #[test]
    fn hdr_png_flows_through_jpeg_xr_tone_mapping() {
        use crate::jpeg_xr::{DecodeOptions, NativeJPEGXR, tonemap_native};

        let samples = words(&[0, 0, 0, 40_000, 40_000, 40_000, 65_535, 65_535, 65_535]);
        let png = encode(
            TRANSFER_PQ,
            PRIMARIES_BT2020,
            ColorType::Rgb,
            &samples,
            3,
            true,
        );
        let hdr = decode_hdr(&png).unwrap();
        let native = NativeJPEGXR::from_scrgb(hdr.width(), hdr.height(), hdr.into_rgba()).unwrap();
        let decoded = tonemap_native(&native, DecodeOptions::default()).unwrap();

        assert!(decoded.metadata().is_hdr());
        let pixels = &decoded.image().rgba;
        assert_eq!(&pixels[..4], &[0, 0, 0, 255]);
        assert!(pixels[4] > 0 && pixels[8] >= pixels[4]);
        assert!(NativeJPEGXR::from_scrgb(2, 2, vec![0.0; 4]).is_err());
    }
}
