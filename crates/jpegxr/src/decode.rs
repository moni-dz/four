//! Decodes transform coefficients and reconstructs image samples.

use crate::bitstream::BitReader;
use crate::codestream::{
    Bands, InternalColorFormat, OutputBitDepth, OutputColorFormat, OverlapMode, ParsedCodestream,
    PlaneHeader,
};
use crate::entropy::{self, AdaptiveVLC};
use crate::error::{Error, ErrorKind, Result};
use rayon::prelude::*;
use std::simd::{Simd, cmp::SimdOrd, num::SimdInt};
use multiversion::multiversion;

const MAX_DIMENSION: usize = 16_384;
const MAX_PIXELS: usize = 64 * 1024 * 1024;
const LOWPASS_BLOCKS_PER_JOB: usize = 512;
const MIN_PARALLEL_LOWPASS_BLOCKS: usize = LOWPASS_BLOCKS_PER_JOB * 4;
const MIN_PARALLEL_MACROBLOCKS: usize = 512;
const MIN_PARALLEL_PIXELS: usize = 256 * 1024;
const PIXEL_LANES: usize = 8;

type I32x8 = Simd<i32, PIXEL_LANES>;
type I64x2 = Simd<i64, 2>;
type I64x4 = Simd<i64, 4>;
type I64x8 = Simd<i64, PIXEL_LANES>;

#[derive(Clone, Debug)]
pub(crate) struct DcImage {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
}

#[derive(Clone, Debug)]
pub(crate) struct LowpassImage {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
}

#[derive(Clone, Debug)]
pub(crate) struct PredictedLowpass {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
    pub(crate) highpass_modes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct HighpassImage {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
    pub(crate) model_bits: Vec<[u8; 2]>,
}

#[derive(Clone, Debug)]
struct IntegerImage {
    width: usize,
    height: usize,
    components: usize,
    values: Vec<i32>,
}

pub(crate) fn decode_rgba_f32(
    primary: &ParsedCodestream<'_>,
    alpha: &ParsedCodestream<'_>,
) -> Result<Vec<f32>> {
    validate_float_rgb_profile(primary, alpha)?;

    let width = usize::try_from(primary.header.width).map_err(|_conversion_error| {
        Error::new(ErrorKind::LimitExceeded("image width"), primary.offset)
    })?;
    let height = usize::try_from(primary.header.height).map_err(|_conversion_error| {
        Error::new(ErrorKind::LimitExceeded("image height"), primary.offset)
    })?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("pixel count"), primary.offset))?;

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::new(
            ErrorKind::LimitExceeded("image dimension"),
            primary.offset,
        ));
    }

    if pixel_count > MAX_PIXELS {
        return Err(Error::new(
            ErrorKind::LimitExceeded("pixel count"),
            primary.offset,
        ));
    }

    let (color, alpha_image) = if pixel_count >= MIN_PARALLEL_PIXELS {
        let (color, alpha_image) = rayon::join(|| reconstruct(primary), || reconstruct(alpha));
        (color?, alpha_image?)
    } else {
        (reconstruct(primary)?, reconstruct(alpha)?)
    };

    let output_len = pixel_count.checked_mul(4).ok_or_else(|| {
        Error::new(
            ErrorKind::LimitExceeded("RGBA output buffer"),
            primary.offset,
        )
    })?;
    let row_len = width
        .checked_mul(4)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("RGBA output row"), primary.offset))?;
    let mut pixels = vec![0.0; output_len];

    let color_left = usize::from(primary.header.margins.left);
    let color_top = usize::from(primary.header.margins.top);
    let alpha_left = usize::from(alpha.header.margins.left);
    let alpha_top = usize::from(alpha.header.margins.top);
    let color_right = color_left.checked_add(width).ok_or_else(|| {
        Error::new(
            ErrorKind::LimitExceeded("cropped image width"),
            primary.offset,
        )
    })?;
    let color_bottom = color_top.checked_add(height).ok_or_else(|| {
        Error::new(
            ErrorKind::LimitExceeded("cropped image height"),
            primary.offset,
        )
    })?;
    let alpha_right = alpha_left
        .checked_add(width)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("alpha image width"), alpha.offset))?;
    let alpha_bottom = alpha_top
        .checked_add(height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("alpha image height"), alpha.offset))?;

    if color.components != 3
        || alpha_image.components != 1
        || color_right > color.width
        || color_bottom > color.height
        || alpha_right > alpha_image.width
        || alpha_bottom > alpha_image.height
    {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("decoded component dimensions are inconsistent"),
            primary.offset,
        ));
    }

    let color_format = FloatFormat::new(primary)?;
    let alpha_format = FloatFormat::new(alpha)?;

    let fill_row = |y, row: &mut [f32]| {
        fill_rgba_row(
            row,
            y,
            color_left,
            color_top,
            alpha_left,
            alpha_top,
            &color,
            &alpha_image,
            color_format,
            alpha_format,
        )
    };

    if pixel_count >= MIN_PARALLEL_PIXELS {
        pixels
            .par_chunks_mut(row_len)
            .with_min_len(8)
            .enumerate()
            .try_for_each(|(y, row)| fill_row(y, row))?;
    } else {
        pixels
            .chunks_mut(row_len)
            .enumerate()
            .try_for_each(|(y, row)| fill_row(y, row))?;
    }

    Ok(pixels)
}

pub(crate) fn decode_bgr101010(stream: &ParsedCodestream<'_>) -> Result<Vec<u32>> {
    validate_bgr101010_profile(stream)?;

    let width = usize::try_from(stream.header.width).map_err(|_conversion_error| {
        Error::new(ErrorKind::LimitExceeded("image width"), stream.offset)
    })?;
    let height = usize::try_from(stream.header.height).map_err(|_conversion_error| {
        Error::new(ErrorKind::LimitExceeded("image height"), stream.offset)
    })?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("pixel count"), stream.offset))?;

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::new(
            ErrorKind::LimitExceeded("image dimension"),
            stream.offset,
        ));
    }

    if pixel_count > MAX_PIXELS {
        return Err(Error::new(
            ErrorKind::LimitExceeded("pixel count"),
            stream.offset,
        ));
    }

    let color = reconstruct(stream)?;
    let left = usize::from(stream.header.margins.left);
    let top = usize::from(stream.header.margins.top);
    let right = left.checked_add(width).ok_or_else(|| {
        Error::new(
            ErrorKind::LimitExceeded("cropped image width"),
            stream.offset,
        )
    })?;
    let bottom = top.checked_add(height).ok_or_else(|| {
        Error::new(
            ErrorKind::LimitExceeded("cropped image height"),
            stream.offset,
        )
    })?;

    if color.components != 3 || right > color.width || bottom > color.height {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("decoded component dimensions are inconsistent"),
            stream.offset,
        ));
    }

    let shift = if stream.primary_plane.scaled { 3 } else { 0 };
    let rounding = if shift == 0 {
        0
    } else {
        (1 << (shift - 1)) - 1
    };
    let bias = (512_i64 << shift) + rounding;
    let swapped = stream.header.red_blue_swapped;
    let mut pixels = vec![0; pixel_count];
    let fill_row =
        |y, row: &mut [u32]| fill_bgr101010_row(row, y, left, top, &color, shift, bias, swapped);

    if pixel_count >= MIN_PARALLEL_PIXELS {
        pixels
            .par_chunks_mut(width)
            .with_min_len(8)
            .enumerate()
            .for_each(|(y, row)| fill_row(y, row));
    } else {
        pixels
            .chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| fill_row(y, row));
    }

    Ok(pixels)
}

#[allow(
    clippy::similar_names,
    reason = "the paired U and V plane names make their channel mapping explicit"
)]
fn fill_bgr101010_row(
    row: &mut [u32],
    y: usize,
    left: usize,
    top: usize,
    color: &IntegerImage,
    shift: u32,
    bias: i64,
    swapped: bool,
) {
    let component_len = color.width * color.height;
    let start = (y + top) * color.width + left;
    let end = start + row.len();
    let luma = &color.values[start..end];
    let chroma_u = &color.values[component_len + start..component_len + end];
    let chroma_v = &color.values[2 * component_len + start..2 * component_len + end];

    let (luma_chunks, luma_tail) = luma.as_chunks::<PIXEL_LANES>();
    let (chroma_u_chunks, chroma_u_tail) = chroma_u.as_chunks::<PIXEL_LANES>();
    let (chroma_v_chunks, chroma_v_tail) = chroma_v.as_chunks::<PIXEL_LANES>();
    let (pixel_chunks, pixel_tail) = row.as_chunks_mut::<PIXEL_LANES>();

    for (((luma, chroma_u), chroma_v), pixels) in luma_chunks
        .iter()
        .zip(chroma_u_chunks)
        .zip(chroma_v_chunks)
        .zip(pixel_chunks)
    {
        let [red, green, blue] = inverse_color_transform_simd(
            I32x8::from_array(*luma),
            I32x8::from_array(*chroma_u),
            I32x8::from_array(*chroma_v),
            bias,
        )
        .map(|channel| clip_10_bit_simd(channel >> i64::from(shift)));

        let packed = if swapped {
            blue | (green << 10) | (red << 20)
        } else {
            red | (green << 10) | (blue << 20)
        };
        *pixels = packed.to_array();
    }

    for (((luma, chroma_u), chroma_v), pixel) in luma_tail
        .iter()
        .zip(chroma_u_tail)
        .zip(chroma_v_tail)
        .zip(pixel_tail)
    {
        let [red, green, blue] = inverse_color_transform(*luma, *chroma_u, *chroma_v, bias);
        let red = clip_10_bit(red >> shift);
        let green = clip_10_bit(green >> shift);
        let blue = clip_10_bit(blue >> shift);

        *pixel = if swapped {
            blue | (green << 10) | (red << 20)
        } else {
            red | (green << 10) | (blue << 20)
        };
    }
}


#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn clip_10_bit(value: i64) -> u32 {
    u32::try_from(value.clamp(0, 1023)).expect("clamped 10-bit sample fits u32")
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn clip_10_bit_simd(value: I64x8) -> Simd<u32, PIXEL_LANES> {
    value.simd_clamp(I64x8::splat(0), I64x8::splat(1023)).cast()
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn inverse_color_transform(y: i32, u: i32, v: i32, bias: i64) -> [i64; 3] {
    let mut green = i64::from(y) + bias;
    let mut red = -i64::from(u);
    let mut blue = i64::from(v);

    green -= red >> 1;
    red -= ((blue + 1) >> 1) - green;
    blue += red;

    [red, green, blue]
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn inverse_color_transform_simd(y: I32x8, u: I32x8, v: I32x8, bias: i64) -> [I64x8; 3] {
    let mut green = y.cast::<i64>() + I64x8::splat(bias);
    let mut red = -u.cast::<i64>();
    let mut blue = v.cast::<i64>();

    green -= red >> 1;
    red -= ((blue + I64x8::splat(1)) >> 1) - green;
    blue += red;

    [red, green, blue]
}

#[allow(
    clippy::many_single_char_names,
    reason = "YUV-to-RGB conversion follows the T.832 sample names"
)]
fn fill_rgba_row(
    row: &mut [f32],
    y: usize,
    color_left: usize,
    color_top: usize,
    alpha_left: usize,
    alpha_top: usize,
    color: &IntegerImage,
    alpha: &IntegerImage,
    color_format: FloatFormat,
    alpha_format: FloatFormat,
) -> Result<()> {
    let color_component_len = color.width * color.height;
    let (pixels, remainder) = row.as_chunks_mut::<4>();
    debug_assert_eq!(remainder, []);

    for (x, pixel) in pixels.iter_mut().enumerate() {
        let color_index = (y + color_top) * color.width + x + color_left;
        let [red, green, blue] = inverse_color_transform(
            color.values[color_index],
            color.values[color_component_len + color_index],
            color.values[2 * color_component_len + color_index],
            0,
        );

        let alpha_index = (y + alpha_top) * alpha.width + x + alpha_left;

        pixel[0] = color_format.convert(red)?;
        pixel[1] = color_format.convert(green)?;
        pixel[2] = color_format.convert(blue)?;
        pixel[3] = alpha_format.convert(i64::from(alpha.values[alpha_index]))?;
    }

    Ok(())
}

pub(crate) fn decode_dc(stream: &ParsedCodestream<'_>) -> Result<DcImage> {
    if !stream.header.frequency_mode {
        return Err(Error::new(
            ErrorKind::Unsupported("spatial-mode coefficient decoding"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if stream.alpha_plane.is_some() {
        return Err(Error::new(
            ErrorKind::Unsupported("interleaved alpha coefficient decoding"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if !matches!(
        stream.primary_plane.internal_color_format,
        InternalColorFormat::YOnly | InternalColorFormat::YUV444
    ) {
        return Err(Error::new(
            ErrorKind::Unsupported("subsampled or multi-component coefficient decoding"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if !stream.primary_plane.dc_uniform {
        return Err(Error::new(
            ErrorKind::Unsupported("per-tile DC quantization"),
            stream.offset + stream.tiles_offset,
        ));
    }

    let macroblock_width = stream
        .header
        .tile_widths
        .iter()
        .try_fold(0_usize, |sum, width| sum.checked_add(usize::from(*width)))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock width"), stream.offset))?;
    let macroblock_height = stream
        .header
        .tile_heights
        .iter()
        .try_fold(0_usize, |sum, height| sum.checked_add(usize::from(*height)))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock height"), stream.offset))?;
    let components = usize::from(stream.primary_plane.component_count);

    let value_count = macroblock_width
        .checked_mul(macroblock_height)
        .and_then(|count| count.checked_mul(components))
        .ok_or_else(|| {
            Error::new(
                ErrorKind::LimitExceeded("DC coefficient buffer"),
                stream.offset,
            )
        })?;
    let mut image = DcImage {
        macroblock_width,
        macroblock_height,
        components,
        values: vec![0; value_count],
    };

    let mut top = 0_usize;
    for (tile_y, tile_height) in stream.header.tile_heights.iter().copied().enumerate() {
        let mut left = 0_usize;
        for (tile_x, tile_width) in stream.header.tile_widths.iter().copied().enumerate() {
            let tile = tile_y * stream.header.tile_widths.len() + tile_x;
            let packet = packet(stream, tile, 0)?;

            decode_dc_packet(
                packet,
                &stream.primary_plane,
                usize::from(tile_width),
                usize::from(tile_height),
                left,
                top,
                &mut image,
            )?;

            left += usize::from(tile_width);
        }

        top += usize::from(tile_height);
    }

    Ok(image)
}

pub(crate) fn decode_lowpass(stream: &ParsedCodestream<'_>) -> Result<LowpassImage> {
    validate_frequency_profile(stream)?;

    if stream.primary_plane.bands.count() < 2 {
        return Err(Error::new(
            ErrorKind::Unsupported("lowpass band is absent"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if !stream.primary_plane.lowpass_uniform {
        return Err(Error::new(
            ErrorKind::Unsupported("per-tile lowpass quantization"),
            stream.offset + stream.tiles_offset,
        ));
    }

    let (macroblock_width, macroblock_height, components) = image_shape(stream)?;

    let value_count = macroblock_width
        .checked_mul(macroblock_height)
        .and_then(|count| count.checked_mul(components))
        .and_then(|count| count.checked_mul(16))
        .ok_or_else(|| {
            Error::new(
                ErrorKind::LimitExceeded("lowpass coefficient buffer"),
                stream.offset,
            )
        })?;
    let mut image = LowpassImage {
        macroblock_width,
        macroblock_height,
        components,
        values: vec![0; value_count],
    };

    let mut top = 0_usize;
    for (tile_y, tile_height) in stream.header.tile_heights.iter().copied().enumerate() {
        let mut left = 0_usize;
        for (tile_x, tile_width) in stream.header.tile_widths.iter().copied().enumerate() {
            let tile = tile_y * stream.header.tile_widths.len() + tile_x;
            let packet = packet(stream, tile, 1)?;

            decode_lowpass_packet(
                packet,
                &stream.primary_plane,
                usize::from(tile_width),
                usize::from(tile_height),
                left,
                top,
                &mut image,
            )?;

            left += usize::from(tile_width);
        }

        top += usize::from(tile_height);
    }

    Ok(image)
}

pub(crate) fn predict_lowpass(
    stream: &ParsedCodestream<'_>,
    dc: &DcImage,
    lowpass: &LowpassImage,
) -> Result<PredictedLowpass> {
    if dc.macroblock_width != lowpass.macroblock_width
        || dc.macroblock_height != lowpass.macroblock_height
        || dc.components != lowpass.components
    {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("coefficient-band dimensions disagree"),
            stream.offset,
        ));
    }

    let dc_quantization = stream
        .primary_plane
        .dc_quantization
        .as_ref()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported("per-tile DC quantization"),
                stream.offset,
            )
        })?;

    let lowpass_quantization = stream
        .primary_plane
        .lowpass_quantization
        .as_ref()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported("per-tile lowpass quantization"),
                stream.offset,
            )
        })?;

    let value_count = lowpass.values.len();
    let macroblock_count = dc
        .macroblock_width
        .checked_mul(dc.macroblock_height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock count"), stream.offset))?;
    let mut raw = vec![0_i32; value_count];
    let mut output = vec![0_i32; value_count];
    let mut highpass_modes = vec![2_u8; macroblock_count];

    let mut top = 0_usize;
    for tile_height in stream.header.tile_heights.iter().copied() {
        let mut left = 0_usize;
        for tile_width in stream.header.tile_widths.iter().copied() {
            let width = usize::from(tile_width);
            let height = usize::from(tile_height);
            for local_y in 0..height {
                for local_x in 0..width {
                    let x = left + local_x;
                    let y = top + local_y;
                    let macroblock = y * dc.macroblock_width + x;
                    let dc_mode = dc_prediction_mode(
                        &raw,
                        dc.macroblock_width,
                        dc.components,
                        macroblock,
                        local_x == 0,
                        local_y == 0,
                    );

                    for component in 0..dc.components {
                        let start = (macroblock * dc.components + component) * 16;

                        raw[start..start + 16].copy_from_slice(&lowpass.values[start..start + 16]);
                        raw[start] = dc.values[macroblock * dc.components + component];

                        predict_dc(
                            &mut raw,
                            dc.macroblock_width,
                            dc.components,
                            macroblock,
                            component,
                            dc_mode,
                        )?;

                        predict_lp(
                            &mut raw,
                            dc.macroblock_width,
                            dc.components,
                            macroblock,
                            component,
                            dc_mode,
                        )?;

                        let dc_factor = quant_map(
                            dc_quantization.components[component],
                            stream.primary_plane.scaled,
                            u8::from(component == 0),
                        );

                        let lp_factor = quant_map(
                            lowpass_quantization.components[component],
                            stream.primary_plane.scaled,
                            u8::from(component == 0),
                        );

                        output[start] = raw[start].checked_mul(dc_factor).ok_or_else(|| {
                            Error::new(
                                ErrorKind::InvalidCodestream("dequantized DC coefficient overflow"),
                                stream.offset,
                            )
                        })?;

                        for coefficient in 1..16 {
                            output[start + coefficient] = raw[start + coefficient]
                                .checked_mul(lp_factor)
                                .ok_or_else(|| {
                                    Error::new(
                                        ErrorKind::InvalidCodestream(
                                            "dequantized lowpass coefficient overflow",
                                        ),
                                        stream.offset,
                                    )
                                })?;
                        }
                    }

                    highpass_modes[macroblock] = highpass_mode(&raw, dc.components, macroblock);
                }
            }

            left += width;
        }

        top += usize::from(tile_height);
    }

    Ok(PredictedLowpass {
        macroblock_width: dc.macroblock_width,
        macroblock_height: dc.macroblock_height,
        components: dc.components,
        values: output,
        highpass_modes,
    })
}

pub(crate) fn decode_highpass(
    stream: &ParsedCodestream<'_>,
    lowpass: &PredictedLowpass,
) -> Result<HighpassImage> {
    validate_frequency_profile(stream)?;

    if stream.primary_plane.bands.count() < 3 {
        return Err(Error::new(
            ErrorKind::Unsupported("highpass band is absent"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if !stream.primary_plane.highpass_uniform {
        return Err(Error::new(
            ErrorKind::Unsupported("per-tile highpass quantization"),
            stream.offset + stream.tiles_offset,
        ));
    }

    let (macroblock_width, macroblock_height, components) = image_shape(stream)?;

    if macroblock_width != lowpass.macroblock_width
        || macroblock_height != lowpass.macroblock_height
        || components != lowpass.components
    {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("lowpass and highpass dimensions disagree"),
            stream.offset,
        ));
    }

    let macroblock_count = macroblock_width
        .checked_mul(macroblock_height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock count"), stream.offset))?;

    let value_count = macroblock_count
        .checked_mul(components)
        .and_then(|count| count.checked_mul(256))
        .ok_or_else(|| {
            Error::new(
                ErrorKind::LimitExceeded("highpass coefficient buffer"),
                stream.offset,
            )
        })?;
    let mut image = HighpassImage {
        macroblock_width,
        macroblock_height,
        components,
        values: vec![0; value_count],
        model_bits: vec![[0; 2]; macroblock_count],
    };
    let mut cbphp = vec![0_u16; macroblock_count * components];

    let mut top = 0_usize;
    for (tile_y, tile_height) in stream.header.tile_heights.iter().copied().enumerate() {
        let mut left = 0_usize;
        for (tile_x, tile_width) in stream.header.tile_widths.iter().copied().enumerate() {
            let tile = tile_y * stream.header.tile_widths.len() + tile_x;
            let packet = packet(stream, tile, 2)?;

            decode_highpass_packet(
                packet,
                &stream.primary_plane,
                usize::from(tile_width),
                usize::from(tile_height),
                left,
                top,
                lowpass,
                &mut cbphp,
                &mut image,
            )?;

            left += usize::from(tile_width);
        }

        top += usize::from(tile_height);
    }

    Ok(image)
}

pub(crate) fn decode_flexbits(
    stream: &ParsedCodestream<'_>,
    highpass: &mut HighpassImage,
) -> Result<()> {
    let (macroblock_width, macroblock_height, components) = image_shape(stream)?;

    if macroblock_width != highpass.macroblock_width
        || macroblock_height != highpass.macroblock_height
        || components != highpass.components
    {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("highpass and flexbits dimensions disagree"),
            stream.offset,
        ));
    }

    match stream.primary_plane.bands {
        Bands::All => {
            let mut top = 0_usize;
            for (tile_y, tile_height) in stream.header.tile_heights.iter().copied().enumerate() {
                let mut left = 0_usize;
                for (tile_x, tile_width) in stream.header.tile_widths.iter().copied().enumerate() {
                    let tile = tile_y * stream.header.tile_widths.len() + tile_x;
                    let width = usize::from(tile_width);
                    let height = usize::from(tile_height);
                    if !tile_has_model_bits(highpass, left, top, width, height) {
                        left += width;
                        continue;
                    }

                    let packet = packet(stream, tile, 3)?;

                    decode_flexbits_packet(
                        packet,
                        stream.header.trim_flexbits,
                        width,
                        height,
                        left,
                        top,
                        highpass,
                    )?;

                    left += width;
                }

                top += usize::from(tile_height);
            }

            Ok(())
        }
        Bands::NoFlexbits => shift_highpass_without_flexbits(highpass),
        Bands::NoHighpass | Bands::DCOnly => Err(Error::new(
            ErrorKind::Unsupported("highpass band is absent"),
            stream.offset + stream.tiles_offset,
        )),
    }
}

fn validate_float_rgb_profile(
    primary: &ParsedCodestream<'_>,
    alpha: &ParsedCodestream<'_>,
) -> Result<()> {
    if primary.header.output_color_format != OutputColorFormat::RGB
        || primary.header.output_bit_depth != OutputBitDepth::ThirtyTwoFloat
        || primary.primary_plane.internal_color_format != InternalColorFormat::YUV444
        || primary.primary_plane.bands != Bands::All
        || primary.primary_plane.scaled
    {
        return Err(Error::new(
            ErrorKind::Unsupported("RGBA128Float primary image profile"),
            primary.offset,
        ));
    }

    if alpha.header.output_color_format != OutputColorFormat::YOnly
        || alpha.header.output_bit_depth != OutputBitDepth::ThirtyTwoFloat
        || alpha.primary_plane.internal_color_format != InternalColorFormat::YOnly
        || alpha.primary_plane.bands != Bands::All
        || alpha.primary_plane.scaled
    {
        return Err(Error::new(
            ErrorKind::Unsupported("RGBA128Float separate-alpha profile"),
            alpha.offset,
        ));
    }

    if primary.header.overlap_mode != OverlapMode::None
        || alpha.header.overlap_mode != OverlapMode::None
    {
        return Err(Error::new(
            ErrorKind::Unsupported("overlap-filtered sample reconstruction"),
            primary.offset,
        ));
    }

    if !primary.header.index_table_present || !alpha.header.index_table_present {
        return Err(Error::new(
            ErrorKind::Unsupported("frequency mode without an index table"),
            primary.offset,
        ));
    }

    if primary.header.spatial_transform != 0 || alpha.header.spatial_transform != 0 {
        return Err(Error::new(
            ErrorKind::Unsupported("codestream spatial transform"),
            primary.offset,
        ));
    }

    Ok(())
}

fn validate_bgr101010_profile(stream: &ParsedCodestream<'_>) -> Result<()> {
    if stream.header.output_color_format != OutputColorFormat::RGB
        || stream.header.output_bit_depth != OutputBitDepth::Ten
        || stream.primary_plane.internal_color_format != InternalColorFormat::YUV444
        || stream.primary_plane.bands != Bands::All
    {
        return Err(Error::new(
            ErrorKind::Unsupported("32bppBGR101010 image profile"),
            stream.offset,
        ));
    }

    if stream.header.overlap_mode != OverlapMode::None {
        return Err(Error::new(
            ErrorKind::Unsupported("overlap-filtered sample reconstruction"),
            stream.offset,
        ));
    }

    if !stream.header.index_table_present {
        return Err(Error::new(
            ErrorKind::Unsupported("frequency mode without an index table"),
            stream.offset,
        ));
    }

    if stream.header.spatial_transform != 0 {
        return Err(Error::new(
            ErrorKind::Unsupported("codestream spatial transform"),
            stream.offset,
        ));
    }

    Ok(())
}

fn reconstruct(stream: &ParsedCodestream<'_>) -> Result<IntegerImage> {
    let dc = decode_dc(stream)?;
    let lowpass = decode_lowpass(stream)?;
    let mut lowpass = predict_lowpass(stream, &dc, &lowpass)?;

    let mut highpass = decode_highpass(stream, &lowpass)?;
    decode_flexbits(stream, &mut highpass)?;
    dequantize_and_predict_highpass(stream, &lowpass, &mut highpass)?;

    let (blocks, remainder) = lowpass.values.as_chunks_mut::<16>();
    debug_assert_eq!(remainder, []);

    if blocks.len() >= MIN_PARALLEL_LOWPASS_BLOCKS {
        blocks
            .par_chunks_mut(LOWPASS_BLOCKS_PER_JOB)
            .enumerate()
            .try_for_each(|(job, blocks)| {
                inverse_lowpass_blocks(
                    blocks,
                    job * LOWPASS_BLOCKS_PER_JOB,
                    lowpass.components,
                    stream.primary_plane.scaled,
                    stream.offset,
                )
            })?;
    } else {
        inverse_lowpass_blocks(
            blocks,
            0,
            lowpass.components,
            stream.primary_plane.scaled,
            stream.offset,
        )?;
    }

    combine_and_transform(stream, &lowpass, &highpass)
}

fn inverse_lowpass_blocks(
    blocks: &mut [[i32; 16]],
    first_block: usize,
    components: usize,
    scaled: bool,
    offset: usize,
) -> Result<()> {
    for (local_block, coefficients) in blocks.iter_mut().enumerate() {
        inverse_transform_4x4(coefficients, offset)?;

        if scaled && !(first_block + local_block).is_multiple_of(components) {
            for value in coefficients {
                *value = value.checked_mul(2).ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidCodestream("scaled lowpass coefficient overflow"),
                        offset,
                    )
                })?;
            }
        }
    }

    Ok(())
}

fn dequantize_and_predict_highpass(
    stream: &ParsedCodestream<'_>,
    lowpass: &PredictedLowpass,
    highpass: &mut HighpassImage,
) -> Result<()> {
    let quantization = stream
        .primary_plane
        .highpass_quantization
        .as_ref()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported("per-tile highpass quantization"),
                stream.offset,
            )
        })?;

    let macroblock_count = highpass
        .macroblock_width
        .checked_mul(highpass.macroblock_height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock count"), stream.offset))?;

    let factors = (0..highpass.components)
        .map(|component| {
            quant_map(
                quantization.components[component],
                stream.primary_plane.scaled,
                1,
            )
        })
        .collect::<Vec<_>>();

    let macroblock_len = highpass.components * 256;
    let process = |macroblock: usize, values: &mut [i32]| {
        dequantize_and_predict_highpass_macroblock(
            values,
            &factors,
            lowpass.highpass_modes[macroblock],
            stream.offset,
        )
    };

    if macroblock_count >= MIN_PARALLEL_MACROBLOCKS {
        highpass
            .values
            .par_chunks_mut(macroblock_len)
            .with_min_len(32)
            .enumerate()
            .try_for_each(|(macroblock, values)| process(macroblock, values))?;
    } else {
        highpass
            .values
            .chunks_mut(macroblock_len)
            .enumerate()
            .try_for_each(|(macroblock, values)| process(macroblock, values))?;
    }

    Ok(())
}

fn dequantize_and_predict_highpass_macroblock(
    macroblock: &mut [i32],
    factors: &[i32],
    mode: u8,
    offset: usize,
) -> Result<()> {
    let (components, remainder) = macroblock.as_chunks_mut::<256>();
    debug_assert_eq!(remainder, []);

    for (values, &factor) in components.iter_mut().zip(factors) {
        for value in values.iter_mut() {
            *value = value.checked_mul(factor).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidCodestream("dequantized highpass coefficient overflow"),
                    offset,
                )
            })?;
        }

        match mode {
            0 => {
                for block in [1_usize, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] {
                    for coefficient in [4_usize, 8, 12] {
                        let reference = values[(block - 1) * 16 + coefficient];
                        let index = block * 16 + coefficient;
                        values[index] = values[index].checked_add(reference).ok_or_else(|| {
                            Error::new(
                                ErrorKind::InvalidCodestream(
                                    "predicted highpass coefficient overflow",
                                ),
                                offset,
                            )
                        })?;
                    }
                }
            }
            1 => {
                for block in 4_usize..16 {
                    for coefficient in [1_usize, 2, 3] {
                        let reference = values[(block - 4) * 16 + coefficient];
                        let index = block * 16 + coefficient;
                        values[index] = values[index].checked_add(reference).ok_or_else(|| {
                            Error::new(
                                ErrorKind::InvalidCodestream(
                                    "predicted highpass coefficient overflow",
                                ),
                                offset,
                            )
                        })?;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn combine_and_transform(
    stream: &ParsedCodestream<'_>,
    lowpass: &PredictedLowpass,
    highpass: &HighpassImage,
) -> Result<IntegerImage> {
    let width = lowpass
        .macroblock_width
        .checked_mul(16)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("extended width"), stream.offset))?;
    let height = lowpass
        .macroblock_height
        .checked_mul(16)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("extended height"), stream.offset))?;

    let component_len = width
        .checked_mul(height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("component buffer"), stream.offset))?;

    let value_count = component_len
        .checked_mul(lowpass.components)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("sample buffer"), stream.offset))?;
    let mut output = IntegerImage {
        width,
        height,
        components: lowpass.components,
        values: vec![0; value_count],
    };

    let band_len = width * 16;
    let fill_band = |band_index, band: &mut [i32]| {
        combine_and_transform_band(band, band_index, width, lowpass, highpass, stream.offset)
    };

    if value_count >= MIN_PARALLEL_PIXELS {
        output
            .values
            .par_chunks_mut(band_len)
            .enumerate()
            .try_for_each(|(band_index, band)| fill_band(band_index, band))?;
    } else {
        output
            .values
            .chunks_mut(band_len)
            .enumerate()
            .try_for_each(|(band_index, band)| fill_band(band_index, band))?;
    }

    Ok(output)
}

fn combine_and_transform_band(
    output: &mut [i32],
    band_index: usize,
    width: usize,
    lowpass: &PredictedLowpass,
    highpass: &HighpassImage,
    offset: usize,
) -> Result<()> {
    let macroblock_y = band_index % lowpass.macroblock_height;
    let component = band_index / lowpass.macroblock_height;

    for macroblock_x in 0..lowpass.macroblock_width {
        let macroblock = macroblock_y * lowpass.macroblock_width + macroblock_x;
        let lowpass_start = (macroblock * lowpass.components + component) * 16;
        let highpass_start = (macroblock * highpass.components + component) * 256;

        for block in 0_usize..16 {
            let mut coefficients = [0_i32; 16];
            coefficients[0] = lowpass.values[lowpass_start + block];

            let source =
                &highpass.values[highpass_start + block * 16..highpass_start + (block + 1) * 16];
            coefficients[1..].copy_from_slice(&source[1..]);

            inverse_transform_4x4(&mut coefficients, offset)?;

            let block_x = block % 4;
            let block_y = block / 4;
            for local_y in 0..4 {
                let row = block_y * 4 + local_y;
                for local_x in 0..4 {
                    let x = macroblock_x * 16 + block_x * 4 + local_x;
                    output[row * width + x] = coefficients[local_y * 4 + local_x];
                }
            }
        }
    }

    Ok(())
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn inverse_transform_4x4(coefficients: &mut [i32; 16], offset: usize) -> Result<()> {
    const PERMUTATION: [usize; 16] = [0, 8, 4, 13, 2, 15, 3, 14, 1, 12, 5, 9, 7, 11, 6, 10];

    let mut values = [0_i64; 16];
    for (input, destination) in PERMUTATION.into_iter().enumerate() {
        values[destination] = i64::from(coefficients[input]);
    }

    transform_group(&mut values, [0, 1, 4, 5], |group| t2x2(group, 1));
    inverse_odd_pair(&mut values);
    transform_group(&mut values, [10, 11, 14, 15], inverse_odd_odd);
    t2x2_quad(&mut values);

    for (destination, value) in coefficients.iter_mut().zip(values) {
        *destination = i32::try_from(value).map_err(|_conversion_error| {
            Error::new(
                ErrorKind::InvalidCodestream("inverse-transform coefficient overflow"),
                offset,
            )
        })?;
    }

    Ok(())
}

fn transform_group(
    values: &mut [i64; 16],
    indexes: [usize; 4],
    transform: impl FnOnce(&mut [i64; 4]),
) {
    let mut group = indexes.map(|index| values[index]);
    transform(&mut group);

    for (index, value) in indexes.into_iter().zip(group) {
        values[index] = value;
    }
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn inverse_odd_pair(values: &mut [i64; 16]) {
    let mut first = I64x2::from_array([values[2], values[8]]);
    let mut second = I64x2::from_array([values[3], values[12]]);
    let mut third = I64x2::from_array([values[6], values[9]]);
    let mut fourth = I64x2::from_array([values[7], values[13]]);

    second += fourth;
    first -= third;
    fourth -= second >> 1;
    third += (first + I64x2::splat(1)) >> 1;
    first -= (I64x2::splat(3) * second + I64x2::splat(4)) >> 3;
    second += (I64x2::splat(3) * first + I64x2::splat(4)) >> 3;
    third -= (I64x2::splat(3) * fourth + I64x2::splat(4)) >> 3;
    fourth += (I64x2::splat(3) * third + I64x2::splat(4)) >> 3;
    third -= (second + I64x2::splat(1)) >> 1;
    fourth = ((first + I64x2::splat(1)) >> 1) - fourth;
    second += third;
    first -= fourth;

    let [first_a, first_b] = first.to_array();
    let [second_a, second_b] = second.to_array();
    let [third_a, third_b] = third.to_array();
    let [fourth_a, fourth_b] = fourth.to_array();
    [values[2], values[8]] = [first_a, first_b];
    [values[3], values[12]] = [second_a, second_b];
    [values[6], values[9]] = [third_a, third_b];
    [values[7], values[13]] = [fourth_a, fourth_b];
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn t2x2_quad(values: &mut [i64; 16]) {
    let mut first = I64x4::from_array([values[0], values[5], values[1], values[4]]);
    let mut second = I64x4::from_array([values[3], values[6], values[2], values[7]]);
    let mut third = I64x4::from_array([values[12], values[9], values[13], values[8]]);
    let mut fourth = I64x4::from_array([values[15], values[10], values[14], values[11]]);

    first += fourth;
    second -= third;
    let midpoint = (first - second) >> 1;
    let previous_third = third;
    third = midpoint - fourth;
    fourth = midpoint - previous_third;
    first -= fourth;
    second += third;

    let [first_a, first_b, first_c, first_d] = first.to_array();
    let [second_a, second_b, second_c, second_d] = second.to_array();
    let [third_a, third_b, third_c, third_d] = third.to_array();
    let [fourth_a, fourth_b, fourth_c, fourth_d] = fourth.to_array();
    [values[0], values[5], values[1], values[4]] = [first_a, first_b, first_c, first_d];
    [values[3], values[6], values[2], values[7]] = [second_a, second_b, second_c, second_d];
    [values[12], values[9], values[13], values[8]] = [third_a, third_b, third_c, third_d];
    [values[15], values[10], values[14], values[11]] = [fourth_a, fourth_b, fourth_c, fourth_d];
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]

fn t2x2(values: &mut [i64; 4], rounding: i64) {
    values[0] += values[3];
    values[1] -= values[2];
    let first = (values[0] - values[1] + rounding) >> 1;
    let second = values[2];
    values[2] = first - values[3];
    values[3] = first - second;
    values[0] -= values[3];
    values[1] += values[2];
}

#[multiversion(targets("x86_64+avx2", "x86_64+sse4.1", "aarch64+neon"))]
fn inverse_odd_odd(values: &mut [i64; 4]) {
    values[3] += values[0];
    values[2] -= values[1];
    let first = values[3] >> 1;
    let second = values[2] >> 1;
    values[0] -= first;
    values[1] += second;
    values[0] -= (3 * values[1] + 3) >> 3;
    values[1] += (3 * values[0] + 3) >> 2;
    values[0] -= (3 * values[1] + 4) >> 3;
    values[1] -= second;
    values[0] += first;
    values[2] += values[1];
    values[3] -= values[0];
    values[1] = -values[1];
    values[2] = -values[2];
}

#[derive(Clone, Copy, Debug)]
struct FloatFormat {
    mantissa_bits: u8,
    exponent_bias: i8,
    offset: usize,
}

impl FloatFormat {
    fn new(stream: &ParsedCodestream<'_>) -> Result<Self> {
        let mantissa_bits = stream.primary_plane.mantissa_bits.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidCodestream("BD32F plane has no mantissa length"),
                stream.offset,
            )
        })?;

        let exponent_bias = stream.primary_plane.exponent_bias.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidCodestream("BD32F plane has no exponent bias"),
                stream.offset,
            )
        })?;

        if mantissa_bits > 23 {
            return Err(Error::new(
                ErrorKind::InvalidCodestream("BD32F sample is outside IEEE 754 range"),
                stream.offset,
            ));
        }

        Ok(Self {
            mantissa_bits,
            exponent_bias,
            offset: stream.offset,
        })
    }

    fn convert(self, value: i64) -> Result<f32> {
        let Self {
            mantissa_bits,
            exponent_bias,
            offset,
        } = self;

        let sign = u32::from(value < 0);
        let magnitude = value.unsigned_abs();
        let mantissa_mask = (1_u64 << mantissa_bits) - 1;

        let mut exponent =
            i64::try_from(magnitude >> mantissa_bits).map_err(|_conversion_error| {
                Error::new(
                    ErrorKind::InvalidCodestream("float exponent overflow"),
                    offset,
                )
            })?;

        let mut mantissa = (magnitude & mantissa_mask) | (1_u64 << mantissa_bits);

        if exponent == 0 {
            mantissa ^= 1_u64 << mantissa_bits;
            exponent = 1;
        }

        exponent = exponent - i64::from(exponent_bias) + 127;

        while mantissa < (1_u64 << mantissa_bits) && exponent > 1 && mantissa > 0 {
            exponent -= 1;
            mantissa <<= 1;
        }

        if mantissa < (1_u64 << mantissa_bits) {
            exponent = 0;
        } else {
            mantissa ^= 1_u64 << mantissa_bits;
        }

        if !(0..=255).contains(&exponent) {
            return Err(Error::new(
                ErrorKind::InvalidCodestream("BD32F sample is outside IEEE 754 range"),
                offset,
            ));
        }

        let exponent = u32::try_from(exponent).expect("validated float exponent fits u32");
        let mantissa =
            u32::try_from(mantissa << (23 - mantissa_bits)).map_err(|_conversion_error| {
                Error::new(
                    ErrorKind::InvalidCodestream("float mantissa overflow"),
                    offset,
                )
            })?;

        Ok(f32::from_bits((sign << 31) | (exponent << 23) | mantissa))
    }
}

fn tile_has_model_bits(
    highpass: &HighpassImage,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
) -> bool {
    (top..top + height).any(|y| {
        (left..left + width).any(|x| {
            highpass.model_bits[y * highpass.macroblock_width + x]
                .into_iter()
                .any(|bits| bits != 0)
        })
    })
}

fn shift_highpass_without_flexbits(highpass: &mut HighpassImage) -> Result<()> {
    for macroblock in 0..highpass.model_bits.len() {
        for component in 0..highpass.components {
            let bits = highpass.model_bits[macroblock][usize::from(component != 0)];
            let start = (macroblock * highpass.components + component) * 256;

            for coefficient in 0..256 {
                highpass.values[start + coefficient] = highpass.values[start + coefficient]
                    .checked_shl(u32::from(bits))
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidCodestream("highpass coefficient overflow"),
                            0,
                        )
                    })?;
            }
        }
    }

    Ok(())
}

fn dc_prediction_mode(
    raw: &[i32],
    width: usize,
    components: usize,
    macroblock: usize,
    left_edge: bool,
    top_edge: bool,
) -> u8 {
    if left_edge && top_edge {
        3
    } else if left_edge {
        1
    } else if top_edge {
        0
    } else {
        let left = raw[((macroblock - 1) * components) * 16];
        let top = raw[((macroblock - width) * components) * 16];
        let top_left = raw[((macroblock - width - 1) * components) * 16];
        let mut horizontal = i64::from(top_left).abs_diff(i64::from(left));
        let mut vertical = i64::from(top_left).abs_diff(i64::from(top));

        if components >= 3 {
            horizontal *= 2;
            vertical *= 2;
            for component in 1..=2 {
                let left = raw[((macroblock - 1) * components + component) * 16];
                let top = raw[((macroblock - width) * components + component) * 16];
                let top_left = raw[((macroblock - width - 1) * components + component) * 16];
                horizontal += i64::from(top_left).abs_diff(i64::from(left));
                vertical += i64::from(top_left).abs_diff(i64::from(top));
            }
        }

        if horizontal.saturating_mul(4) < vertical {
            1
        } else if vertical.saturating_mul(4) < horizontal {
            0
        } else {
            2
        }
    }
}

fn predict_dc(
    raw: &mut [i32],
    width: usize,
    components: usize,
    macroblock: usize,
    component: usize,
    mode: u8,
) -> Result<()> {
    let current = (macroblock * components + component) * 16;
    let prediction = match mode {
        0 => raw[((macroblock - 1) * components + component) * 16],
        1 => raw[((macroblock - width) * components + component) * 16],
        2 => {
            let left = raw[((macroblock - 1) * components + component) * 16];
            let top = raw[((macroblock - width) * components + component) * 16];
            (left + top) >> 1
        }
        _ => 0,
    };

    raw[current] = raw[current]
        .checked_add(prediction)
        .ok_or_else(|| Error::new(ErrorKind::InvalidCodestream("predicted DC overflow"), 0))?;

    Ok(())
}

fn predict_lp(
    raw: &mut [i32],
    width: usize,
    components: usize,
    macroblock: usize,
    component: usize,
    dc_mode: u8,
) -> Result<()> {
    let current = (macroblock * components + component) * 16;
    let (reference, coefficients): (Option<usize>, &[usize]) = match dc_mode {
        0 => (
            Some(((macroblock - 1) * components + component) * 16),
            &[4, 8, 12],
        ),
        1 => (
            Some(((macroblock - width) * components + component) * 16),
            &[1, 2, 3],
        ),
        _ => (None, &[]),
    };

    if let Some(reference) = reference {
        for coefficient in coefficients {
            raw[current + coefficient] = raw[current + coefficient]
                .checked_add(raw[reference + coefficient])
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidCodestream("predicted lowpass overflow"),
                        0,
                    )
                })?;
        }
    }

    Ok(())
}

fn highpass_mode(lowpass: &[i32], components: usize, macroblock: usize) -> u8 {
    let start = macroblock * components * 16;
    let mut horizontal = [1, 2, 3]
        .into_iter()
        .map(|coefficient| i64::from(lowpass[start + coefficient]).unsigned_abs())
        .sum::<u64>();
    let mut vertical = [4, 8, 12]
        .into_iter()
        .map(|coefficient| i64::from(lowpass[start + coefficient]).unsigned_abs())
        .sum::<u64>();

    for component in 1..components.min(3) {
        let start = (macroblock * components + component) * 16;
        horizontal += i64::from(lowpass[start + 1]).unsigned_abs();
        vertical += i64::from(lowpass[start + 4]).unsigned_abs();
    }

    if horizontal.saturating_mul(4) < vertical {
        0
    } else if vertical.saturating_mul(4) < horizontal {
        1
    } else {
        2
    }
}

fn quant_map(qp: u8, scaled: bool, scaled_shift: u8) -> i32 {
    if qp == 0 {
        return 1;
    }

    let qp = u32::from(qp);
    let (mantissa, exponent) = if !scaled {
        if qp < 32 {
            ((qp + 3) >> 2, 0)
        } else if qp < 48 {
            ((17 + qp % 16) >> 1, (qp >> 4) - 2)
        } else {
            (16 + qp % 16, (qp >> 4) - 3)
        }
    } else if qp < 16 {
        (qp, u32::from(scaled_shift))
    } else {
        (16 + qp % 16, (qp >> 4) - 1 + u32::from(scaled_shift))
    };

    i32::try_from(mantissa << exponent).expect("8-bit QP maps to an i32 scaling factor")
}

fn validate_frequency_profile(stream: &ParsedCodestream<'_>) -> Result<()> {
    if !stream.header.frequency_mode {
        return Err(Error::new(
            ErrorKind::Unsupported("spatial-mode coefficient decoding"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if stream.alpha_plane.is_some() {
        return Err(Error::new(
            ErrorKind::Unsupported("interleaved alpha coefficient decoding"),
            stream.offset + stream.tiles_offset,
        ));
    }

    if !matches!(
        stream.primary_plane.internal_color_format,
        InternalColorFormat::YOnly | InternalColorFormat::YUV444
    ) {
        return Err(Error::new(
            ErrorKind::Unsupported("subsampled or multi-component coefficient decoding"),
            stream.offset + stream.tiles_offset,
        ));
    }

    Ok(())
}

fn image_shape(stream: &ParsedCodestream<'_>) -> Result<(usize, usize, usize)> {
    let macroblock_width = stream
        .header
        .tile_widths
        .iter()
        .try_fold(0_usize, |sum, width| sum.checked_add(usize::from(*width)))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock width"), stream.offset))?;
    let macroblock_height = stream
        .header
        .tile_heights
        .iter()
        .try_fold(0_usize, |sum, height| sum.checked_add(usize::from(*height)))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("macroblock height"), stream.offset))?;

    Ok((
        macroblock_width,
        macroblock_height,
        usize::from(stream.primary_plane.component_count),
    ))
}

#[derive(Clone, Copy)]
struct Packet<'a> {
    bytes: &'a [u8],
    offset: usize,
}

fn packet<'a>(stream: &ParsedCodestream<'a>, tile: usize, band: usize) -> Result<Packet<'a>> {
    let bands = stream.primary_plane.bands.count();
    let index = tile
        .checked_mul(bands)
        .and_then(|index| index.checked_add(band))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("tile index"), stream.offset))?;

    let relative = *stream.index_offsets.get(index).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidCodestream("missing tile-packet index"),
            stream.offset + stream.tiles_offset,
        )
    })?;

    let coded_length = stream.bytes.len().saturating_sub(stream.tiles_offset);

    let relative = usize::try_from(relative).map_err(|_conversion_error| {
        Error::new(
            ErrorKind::InvalidCodestream("tile-packet offset does not fit memory"),
            stream.offset + stream.tiles_offset,
        )
    })?;
    let end_relative = stream
        .index_offsets
        .iter()
        .filter_map(|offset| usize::try_from(*offset).ok())
        .filter(|offset| *offset > relative)
        .min()
        .unwrap_or(coded_length);

    if relative >= end_relative || end_relative > coded_length {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("tile-packet index is out of range or duplicated"),
            stream.offset + stream.tiles_offset,
        ));
    }

    let start = stream.tiles_offset + relative;
    let end = stream.tiles_offset + end_relative;

    Ok(Packet {
        bytes: &stream.bytes[start..end],
        offset: stream.offset + start,
    })
}

fn decode_dc_packet(
    packet: Packet<'_>,
    plane: &PlaneHeader,
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    image: &mut DcImage,
) -> Result<()> {
    let mut reader = BitReader::new(packet.bytes, packet.offset);

    if reader.read_u32(24)? != 1 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "tile start code must equal 0x000001",
        )));
    }

    let _arbitrary_byte = reader.read_u8(8)?;
    let mut context = DcContext::new();

    for local_y in 0..height {
        for local_x in 0..width {
            let values = decode_dc_macroblock(&mut reader, plane, &mut context)?;
            let macroblock = (top + local_y) * image.macroblock_width + left + local_x;
            let start = macroblock * image.components;
            image.values[start..start + image.components].copy_from_slice(&values);

            if local_x.is_multiple_of(16) || local_x + 1 == width {
                context.adapt();
            }
        }
    }

    reader.align_zero()?;

    if reader.byte_position() != packet.bytes.len() {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "DC tile packet does not end at its indexed boundary",
        )));
    }

    Ok(())
}

fn decode_lowpass_packet(
    packet: Packet<'_>,
    plane: &PlaneHeader,
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    image: &mut LowpassImage,
) -> Result<()> {
    let mut reader = BitReader::new(packet.bytes, packet.offset);

    if reader.read_u32(24)? != 1 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "tile start code must equal 0x000001",
        )));
    }

    let _arbitrary_byte = reader.read_u8(8)?;
    let mut context = LowpassContext::new();

    for local_y in 0..height {
        for local_x in 0..width {
            if local_x.is_multiple_of(16) {
                context.scan.reset_totals();
            }

            let values = decode_lowpass_macroblock(&mut reader, plane, &mut context)?;
            let macroblock = (top + local_y) * image.macroblock_width + left + local_x;
            let start = macroblock * image.components * 16;
            image.values[start..start + image.components * 16].copy_from_slice(&values);

            if local_x.is_multiple_of(16) || local_x + 1 == width {
                context.adapt();
            }
        }
    }

    reader.align_zero()?;

    if reader.byte_position() != packet.bytes.len() {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "lowpass tile packet does not end at its indexed boundary",
        )));
    }

    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "packet geometry is explicit at the frequency-band boundary"
)]
fn decode_highpass_packet(
    packet: Packet<'_>,
    plane: &PlaneHeader,
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    lowpass: &PredictedLowpass,
    cbphp: &mut [u16],
    image: &mut HighpassImage,
) -> Result<()> {
    let mut reader = BitReader::new(packet.bytes, packet.offset);

    if reader.read_u32(24)? != 1 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "tile start code must equal 0x000001",
        )));
    }

    let _arbitrary_byte = reader.read_u8(8)?;
    let mut context = HighpassContext::new();

    for local_y in 0..height {
        for local_x in 0..width {
            if local_x.is_multiple_of(16) {
                context.horizontal_scan.reset_totals();
                context.vertical_scan.reset_totals();
            }

            let macroblock = (top + local_y) * image.macroblock_width + left + local_x;

            let patterns = decode_cbphp(
                &mut reader,
                plane,
                &mut context,
                cbphp,
                image.macroblock_width,
                macroblock,
                local_x == 0,
                local_y == 0,
            )?;
            cbphp[macroblock * image.components..(macroblock + 1) * image.components]
                .copy_from_slice(&patterns);

            decode_highpass_macroblock(
                &mut reader,
                plane,
                &mut context,
                macroblock,
                lowpass.highpass_modes[macroblock],
                &patterns,
                image,
            )?;

            if local_x.is_multiple_of(16) || local_x + 1 == width {
                context.adapt();
            }
        }
    }

    reader.align_zero()?;

    if reader.byte_position() != packet.bytes.len() {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "highpass tile packet does not end at its indexed boundary",
        )));
    }

    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "packet geometry is explicit at the frequency-band boundary"
)]
fn decode_flexbits_packet(
    packet: Packet<'_>,
    trim_present: bool,
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    highpass: &mut HighpassImage,
) -> Result<()> {
    const HIERARCHICAL_ORDER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
    const TRANSPOSE: [usize; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

    let mut reader = BitReader::new(packet.bytes, packet.offset);

    if reader.read_u32(24)? != 1 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "tile start code must equal 0x000001",
        )));
    }

    let _arbitrary_byte = reader.read_u8(8)?;
    let trim = if trim_present { reader.read_u8(4)? } else { 0 };

    for local_y in 0..height {
        for local_x in 0..width {
            let macroblock = (top + local_y) * highpass.macroblock_width + left + local_x;
            for component in 0..highpass.components {
                let model = usize::from(component != 0);
                let model_bits = highpass.model_bits[macroblock][model];
                let flex_bits = model_bits.saturating_sub(trim);

                for block in HIERARCHICAL_ORDER {
                    let start = (macroblock * highpass.components + component) * 256 + block * 16;
                    for coefficient in TRANSPOSE.into_iter().skip(1) {
                        let vlc = highpass.values[start + coefficient];
                        let refinement = if flex_bits == 0 {
                            0
                        } else {
                            i32::try_from(reader.read_u32(flex_bits)?)
                                .expect("at most 15 flexbits fit i32")
                        };
                        let flex = match vlc.cmp(&0) {
                            std::cmp::Ordering::Greater => refinement,
                            std::cmp::Ordering::Less => -refinement,
                            std::cmp::Ordering::Equal
                                if refinement != 0 && reader.read_bool()? =>
                            {
                                -refinement
                            }
                            std::cmp::Ordering::Equal => refinement,
                        };

                        let flex = flex.checked_shl(u32::from(trim)).ok_or_else(|| {
                            reader.error(ErrorKind::InvalidCodestream(
                                "flexbits coefficient overflow",
                            ))
                        })?;
                        highpass.values[start + coefficient] = vlc
                            .checked_shl(u32::from(model_bits))
                            .and_then(|value| value.checked_add(flex))
                            .ok_or_else(|| {
                                reader.error(ErrorKind::InvalidCodestream(
                                    "highpass coefficient overflow",
                                ))
                            })?;
                    }
                }
            }
        }
    }

    reader.align_zero()?;

    if reader.byte_position() != packet.bytes.len() {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "flexbits tile packet does not end at its indexed boundary",
        )));
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct HighpassContext {
    first_luma: AdaptiveVLC,
    index_luma_zero: AdaptiveVLC,
    index_luma_one: AdaptiveVLC,
    first_chroma: AdaptiveVLC,
    index_chroma_zero: AdaptiveVLC,
    index_chroma_one: AdaptiveVLC,
    level_zero: AdaptiveVLC,
    level_one: AdaptiveVLC,

    num_cbphp: AdaptiveVLC,
    num_block_cbphp: AdaptiveVLC,
    cbphp_state: [u8; 2],
    count_ones: [i8; 2],
    count_zeroes: [i8; 2],

    model_state: [i32; 2],
    model_bits: [u8; 2],

    horizontal_scan: AdaptiveScan,
    vertical_scan: AdaptiveScan,
}

impl HighpassContext {
    const fn new() -> Self {
        Self {
            first_luma: AdaptiveVLC::many_tables(),
            index_luma_zero: AdaptiveVLC::many_tables(),
            index_luma_one: AdaptiveVLC::many_tables(),
            first_chroma: AdaptiveVLC::many_tables(),
            index_chroma_zero: AdaptiveVLC::many_tables(),
            index_chroma_one: AdaptiveVLC::many_tables(),
            level_zero: AdaptiveVLC::two_tables(),
            level_one: AdaptiveVLC::two_tables(),

            num_cbphp: AdaptiveVLC::two_tables(),
            num_block_cbphp: AdaptiveVLC::two_tables(),
            cbphp_state: [0, 0],
            count_ones: [-4, -4],
            count_zeroes: [4, 4],

            model_state: [0, 0],
            model_bits: [0, 0],

            horizontal_scan: AdaptiveScan::new_highpass_horizontal(),
            vertical_scan: AdaptiveScan::new_highpass_vertical(),
        }
    }

    fn decode_first(&mut self, reader: &mut BitReader<'_>, chroma: bool) -> Result<u8> {
        entropy::first_index(
            reader,
            if chroma {
                &mut self.first_chroma
            } else {
                &mut self.first_luma
            },
        )
    }

    fn decode_index(
        &mut self,
        reader: &mut BitReader<'_>,
        chroma: bool,
        context: bool,
        location: u8,
    ) -> Result<u8> {
        if location < 15 {
            let adaptive = match (chroma, context) {
                (false, false) => &mut self.index_luma_zero,
                (false, true) => &mut self.index_luma_one,
                (true, false) => &mut self.index_chroma_zero,
                (true, true) => &mut self.index_chroma_one,
            };
            entropy::index_a(reader, adaptive)
        } else if location == 15 {
            if !reader.read_bool()? {
                Ok(0)
            } else if !reader.read_bool()? {
                Ok(2)
            } else {
                Ok(1 + 2 * reader.read_u8(1)?)
            }
        } else {
            reader.read_u8(1)
        }
    }

    fn decode_level(&mut self, reader: &mut BitReader<'_>, context: bool) -> Result<u32> {
        decode_absolute_level(
            reader,
            if context {
                &mut self.level_one
            } else {
                &mut self.level_zero
            },
        )
    }

    fn adapt(&mut self) {
        self.first_luma.adapt_many(4);
        self.index_luma_zero.adapt_many(3);
        self.index_luma_one.adapt_many(3);
        self.first_chroma.adapt_many(4);
        self.index_chroma_zero.adapt_many(3);
        self.index_chroma_one.adapt_many(3);
        self.level_zero.adapt_two();
        self.level_one.adapt_two();
        self.num_cbphp.adapt_two();
        self.num_block_cbphp.adapt_two();
    }

    fn update_model(&mut self, format: InternalColorFormat, mut laplacian: [i32; 2]) {
        if format == InternalColorFormat::YUV444 {
            laplacian[1] = (laplacian[1] * 8) >> 4;
        }

        let models = if format == InternalColorFormat::YOnly {
            1
        } else {
            2
        };
        update_model(
            &mut self.model_state,
            &mut self.model_bits,
            laplacian,
            models,
        );
    }

    fn update_cbphp_model(&mut self, model: usize, ones: i8) {
        self.count_ones[model] = (self.count_ones[model] + ones - 3).clamp(-16, 15);
        self.count_zeroes[model] = (self.count_zeroes[model] + 16 - ones - 3).clamp(-16, 15);

        self.cbphp_state[model] = if self.count_ones[model] < 0 {
            u8::from(self.count_ones[model] >= self.count_zeroes[model]) + 1
        } else if self.count_zeroes[model] < 0 {
            2
        } else {
            0
        };
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "CBPHP prediction needs explicit macroblock edge state"
)]
fn decode_cbphp(
    reader: &mut BitReader<'_>,
    plane: &PlaneHeader,
    context: &mut HighpassContext,
    history: &[u16],
    macroblock_width: usize,
    macroblock: usize,
    left_edge: bool,
    top_edge: bool,
) -> Result<Vec<u16>> {
    const FIXED_LENGTH: [u8; 6] = [0, 2, 1, 2, 2, 0];
    const OFFSET: [u8; 6] = [0, 4, 2, 8, 12, 1];
    const OUTPUT: [u8; 16] = [0, 15, 3, 12, 1, 2, 4, 8, 5, 6, 9, 10, 7, 11, 13, 14];

    let components = usize::from(plane.component_count);
    let group_count = entropy::num_cbphp(reader, &mut context.num_cbphp)?;
    let groups = refine_cbphp(reader, group_count)?;
    let mut residual = vec![0_u16; components];

    for group in 0..4 {
        if groups & (1 << group) == 0 {
            continue;
        }
        let block_count = if plane.internal_color_format == InternalColorFormat::YUV444 {
            entropy::num_block_cbphp_yuv(reader, &mut context.num_block_cbphp)?
        } else {
            entropy::num_block_cbphp_yonly(reader, &mut context.num_block_cbphp)?
        };

        let mut value = block_count + 1;
        let mut block_pattern = 0_u8;

        if value >= 6 {
            block_pattern = 0x10 * (entropy::ternary(reader)? + 1);
            if value >= 9 {
                value += entropy::ternary(reader)?;
            }
            value -= 6;
        }

        let value_index = usize::from(value);
        let mut code = OFFSET[value_index];

        if FIXED_LENGTH[value_index] != 0 {
            code += reader.read_u8(FIXED_LENGTH[value_index])?;
        }

        block_pattern += OUTPUT[usize::from(code)];
        residual[0] |= u16::from(block_pattern & 0x0f) << (group * 4);

        if plane.internal_color_format == InternalColorFormat::YUV444 {
            for chroma in 0..2 {
                if block_pattern & (0x10 << chroma) != 0 {
                    let count = entropy::num_chroma_block(reader)? + 1;
                    residual[chroma + 1] |= u16::from(refine_cbphp(reader, count)?) << (group * 4);
                }
            }
        }
    }

    for (component, value) in residual.iter_mut().enumerate() {
        let model = usize::from(component != 0);
        let mut pattern = u32::from(*value);
        if context.cbphp_state[model] == 0 {
            let seed = if left_edge {
                if top_edge {
                    1
                } else {
                    u32::from(
                        history[(macroblock - macroblock_width) * components + component] >> 10,
                    ) & 1
                }
            } else {
                u32::from(history[(macroblock - 1) * components + component] >> 5) & 1
            };
            pattern ^= seed;
            pattern ^= 0x02 & (pattern << 1);
            pattern ^= 0x10 & (pattern << 3);
            pattern ^= 0x20 & (pattern << 1);
            pattern ^= (pattern & 0x33) << 2;
            pattern ^= (pattern & 0x00cc) << 6;
            pattern ^= (pattern & 0x3300) << 2;
        } else if context.cbphp_state[model] == 2 {
            pattern ^= 0xffff;
        }

        *value = u16::try_from(pattern).map_err(|_conversion_error| {
            reader.error(ErrorKind::InvalidCodestream(
                "CBPHP pattern exceeds 16 bits",
            ))
        })?;

        context.update_cbphp_model(
            model,
            i8::try_from(pattern.count_ones()).expect("16-bit pattern has at most 16 ones"),
        );
    }

    Ok(residual)
}

fn refine_cbphp(reader: &mut BitReader<'_>, count: u8) -> Result<u8> {
    match count {
        1 => Ok(1 << reader.read_u8(2)?),
        2 => entropy::refine_cbphp_one(reader),
        3 => Ok(0x0f ^ (1 << reader.read_u8(2)?)),
        4 => Ok(0x0f),
        _ => Ok(0),
    }
}

fn decode_highpass_macroblock(
    reader: &mut BitReader<'_>,
    plane: &PlaneHeader,
    context: &mut HighpassContext,
    macroblock: usize,
    mode: u8,
    patterns: &[u16],
    image: &mut HighpassImage,
) -> Result<()> {
    const HIERARCHICAL_ORDER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

    image.model_bits[macroblock] = context.model_bits;
    let mut laplacian = [0_i32; 2];

    for (component, pattern) in patterns.iter().copied().enumerate() {
        let model = usize::from(component != 0);
        let mut pattern = pattern;
        for block in HIERARCHICAL_ORDER {
            if pattern & 1 != 0 {
                let start = (macroblock * image.components + component) * 256 + block * 16;
                let coefficients: &mut [i32; 16] = (&mut image.values[start..start + 16])
                    .try_into()
                    .expect("highpass block has 16 coefficients");
                laplacian[model] +=
                    decode_highpass_block(reader, component != 0, mode, context, coefficients)?;
            }
            pattern >>= 1;
        }
    }

    context.update_model(plane.internal_color_format, laplacian);

    Ok(())
}

fn decode_highpass_block(
    reader: &mut BitReader<'_>,
    chroma: bool,
    mode: u8,
    context: &mut HighpassContext,
    coefficients: &mut [i32; 16],
) -> Result<i32> {
    let first = context.decode_first(reader, chroma)?;
    let mut continuing = first >> 2;
    let mut level_context = (first & 1) & continuing;
    let negative = reader.read_bool()?;

    let magnitude = if first & 2 != 0 {
        context.decode_level(reader, level_context != 0)?
    } else {
        1
    };

    let mut value = signed_level(reader, magnitude, negative)?;
    let mut position = 1_u8;

    if first & 1 == 0 {
        position += entropy::run(reader, 14)?;
    }

    context.place_highpass(coefficients, position, value, mode);
    let mut location = position + 1;
    let mut nonzero = 1_i32;

    while continuing != 0 {
        if continuing & 1 == 0 {
            let maximum = 15_u8.checked_sub(location).ok_or_else(|| {
                reader.error(ErrorKind::InvalidCodestream(
                    "highpass coefficient run exceeds block",
                ))
            })?;
            position = location + entropy::run(reader, maximum)?;
        } else {
            position = location;
        }

        location = position + 1;

        if location > 16 {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "highpass coefficient position exceeds block",
            )));
        }

        let index = context.decode_index(reader, chroma, level_context != 0, location)?;
        continuing = index >> 1;
        level_context &= continuing;
        let negative = reader.read_bool()?;

        let magnitude = if index & 1 != 0 {
            context.decode_level(reader, level_context != 0)?
        } else {
            1
        };

        value = signed_level(reader, magnitude, negative)?;
        context.place_highpass(coefficients, position, value, mode);
        nonzero += 1;
    }

    Ok(nonzero)
}

impl HighpassContext {
    fn place_highpass(&mut self, coefficients: &mut [i32; 16], position: u8, value: i32, mode: u8) {
        if mode == 1 {
            self.vertical_scan.place(coefficients, position, value);
        } else {
            self.horizontal_scan.place(coefficients, position, value);
        }
    }
}

#[derive(Clone, Debug)]
struct LowpassContext {
    first_luma: AdaptiveVLC,
    index_luma_zero: AdaptiveVLC,
    index_luma_one: AdaptiveVLC,
    first_chroma: AdaptiveVLC,
    index_chroma_zero: AdaptiveVLC,
    index_chroma_one: AdaptiveVLC,
    level_zero: AdaptiveVLC,
    level_one: AdaptiveVLC,

    count_zero: i8,
    count_maximum: i8,

    model_state: [i32; 2],
    model_bits: [u8; 2],

    scan: AdaptiveScan,
}

impl LowpassContext {
    const fn new() -> Self {
        Self {
            first_luma: AdaptiveVLC::many_tables(),
            index_luma_zero: AdaptiveVLC::many_tables(),
            index_luma_one: AdaptiveVLC::many_tables(),
            first_chroma: AdaptiveVLC::many_tables(),
            index_chroma_zero: AdaptiveVLC::many_tables(),
            index_chroma_one: AdaptiveVLC::many_tables(),
            level_zero: AdaptiveVLC::two_tables(),
            level_one: AdaptiveVLC::two_tables(),

            count_zero: 1,
            count_maximum: 1,

            model_state: [0, 0],
            model_bits: [4, 4],

            scan: AdaptiveScan::new_lowpass(),
        }
    }

    fn decode_first(&mut self, reader: &mut BitReader<'_>, chroma: bool) -> Result<u8> {
        entropy::first_index(
            reader,
            if chroma {
                &mut self.first_chroma
            } else {
                &mut self.first_luma
            },
        )
    }

    fn decode_index(
        &mut self,
        reader: &mut BitReader<'_>,
        chroma: bool,
        context: bool,
        location: u8,
    ) -> Result<u8> {
        if location < 15 {
            let adaptive = match (chroma, context) {
                (false, false) => &mut self.index_luma_zero,
                (false, true) => &mut self.index_luma_one,
                (true, false) => &mut self.index_chroma_zero,
                (true, true) => &mut self.index_chroma_one,
            };
            entropy::index_a(reader, adaptive)
        } else if location == 15 {
            if !reader.read_bool()? {
                Ok(0)
            } else if !reader.read_bool()? {
                Ok(2)
            } else {
                Ok(1 + 2 * reader.read_u8(1)?)
            }
        } else {
            reader.read_u8(1)
        }
    }

    fn decode_level(&mut self, reader: &mut BitReader<'_>, context: bool) -> Result<u32> {
        decode_absolute_level(
            reader,
            if context {
                &mut self.level_one
            } else {
                &mut self.level_zero
            },
        )
    }

    fn adapt(&mut self) {
        self.first_luma.adapt_many(4);
        self.index_luma_zero.adapt_many(3);
        self.index_luma_one.adapt_many(3);
        self.first_chroma.adapt_many(4);
        self.index_chroma_zero.adapt_many(3);
        self.index_chroma_one.adapt_many(3);
        self.level_zero.adapt_two();
        self.level_one.adapt_two();
    }

    fn update_model(&mut self, format: InternalColorFormat, mut laplacian: [i32; 2]) {
        laplacian[0] *= 12;

        if format == InternalColorFormat::YUV444 {
            laplacian[1] *= 6;
        }

        let models = if format == InternalColorFormat::YOnly {
            1
        } else {
            2
        };
        update_model(
            &mut self.model_state,
            &mut self.model_bits,
            laplacian,
            models,
        );
    }
}

#[derive(Clone, Debug)]
struct AdaptiveScan {
    order: [u8; 16],
    totals: [u16; 16],
}

impl AdaptiveScan {
    const INITIAL_ORDER: [u8; 16] = [0, 4, 1, 5, 8, 2, 9, 6, 12, 3, 10, 13, 7, 14, 11, 15];
    const VERTICAL_ORDER: [u8; 16] = [0, 1, 2, 5, 4, 3, 6, 9, 8, 7, 12, 15, 13, 10, 11, 14];
    const INITIAL_TOTALS: [u16; 16] = [0, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4];

    const fn new_lowpass() -> Self {
        Self {
            order: Self::INITIAL_ORDER,
            totals: Self::INITIAL_TOTALS,
        }
    }

    const fn new_highpass_horizontal() -> Self {
        Self::new_lowpass()
    }

    const fn new_highpass_vertical() -> Self {
        Self {
            order: Self::VERTICAL_ORDER,
            totals: Self::INITIAL_TOTALS,
        }
    }

    fn reset_totals(&mut self) {
        self.totals = Self::INITIAL_TOTALS;
    }

    fn place(&mut self, coefficients: &mut [i32; 16], position: u8, value: i32) {
        let position = usize::from(position);

        coefficients[usize::from(self.order[position])] = value;
        self.totals[position] += 1;

        if position > 1 && self.totals[position] > self.totals[position - 1] {
            self.totals.swap(position, position - 1);
            self.order.swap(position, position - 1);
        }
    }
}

fn decode_lowpass_macroblock(
    reader: &mut BitReader<'_>,
    plane: &PlaneHeader,
    context: &mut LowpassContext,
) -> Result<Vec<i32>> {
    let components = usize::from(plane.component_count);
    let maximum = if plane.internal_color_format == InternalColorFormat::YUV444 {
        7
    } else {
        1
    };

    let coded_pattern = if plane.internal_color_format == InternalColorFormat::YUV444 {
        let pattern = if context.count_zero <= 0 || context.count_maximum < 0 {
            let transmitted = entropy::cbp_lowpass_yuv444(reader)?;
            if context.count_maximum < context.count_zero {
                maximum - transmitted
            } else {
                transmitted
            }
        } else {
            reader.read_u8(3)?
        };

        context.count_zero = (context.count_zero + 1 - 4 * i8::from(pattern == 0)).clamp(-8, 7);
        context.count_maximum =
            (context.count_maximum + 1 - 4 * i8::from(pattern == maximum)).clamp(-8, 7);
        pattern
    } else {
        reader.read_u8(1)?
    };

    let mut output = vec![0_i32; components * 16];
    let mut laplacian = [0_i32; 2];

    for component in 0..components {
        let model = usize::from(component != 0);
        let coefficients: &mut [i32; 16] = (&mut output[component * 16..(component + 1) * 16])
            .try_into()
            .expect("component coefficient slice has length 16");

        let nonzero = if coded_pattern & (1 << component) != 0 {
            decode_block(reader, component != 0, context, coefficients)?
        } else {
            0
        };

        laplacian[model] += nonzero;
        refine_lowpass(reader, coefficients, context.model_bits[model])?;
    }

    context.update_model(plane.internal_color_format, laplacian);

    Ok(output)
}

fn decode_block(
    reader: &mut BitReader<'_>,
    chroma: bool,
    context: &mut LowpassContext,
    coefficients: &mut [i32; 16],
) -> Result<i32> {
    let first = context.decode_first(reader, chroma)?;
    let mut continuing = first >> 2;
    let mut level_context = (first & 1) & continuing;
    let negative = reader.read_bool()?;

    let magnitude = if first & 2 != 0 {
        context.decode_level(reader, level_context != 0)?
    } else {
        1
    };

    let mut value = signed_level(reader, magnitude, negative)?;
    let mut position = 1_u8;

    if first & 1 == 0 {
        position += entropy::run(reader, 14)?;
    }

    context.scan.place(coefficients, position, value);
    let mut location = position + 1;
    let mut nonzero = 1_i32;

    while continuing != 0 {
        if continuing & 1 == 0 {
            let maximum = 15_u8.checked_sub(location).ok_or_else(|| {
                reader.error(ErrorKind::InvalidCodestream(
                    "lowpass coefficient run exceeds block",
                ))
            })?;
            position = location + entropy::run(reader, maximum)?;
        } else {
            position = location;
        }

        location = position + 1;

        if location > 16 {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                "lowpass coefficient position exceeds block",
            )));
        }

        let index = context.decode_index(reader, chroma, level_context != 0, location)?;
        continuing = index >> 1;
        level_context &= continuing;
        let negative = reader.read_bool()?;

        let magnitude = if index & 1 != 0 {
            context.decode_level(reader, level_context != 0)?
        } else {
            1
        };

        value = signed_level(reader, magnitude, negative)?;
        context.scan.place(coefficients, position, value);
        nonzero += 1;
    }

    Ok(nonzero)
}

fn signed_level(reader: &BitReader<'_>, magnitude: u32, negative: bool) -> Result<i32> {
    let magnitude = i32::try_from(magnitude).map_err(|_conversion_error| {
        reader.error(ErrorKind::InvalidCodestream("coefficient level overflow"))
    })?;

    Ok(if negative { -magnitude } else { magnitude })
}

fn refine_lowpass(
    reader: &mut BitReader<'_>,
    coefficients: &mut [i32; 16],
    model_bits: u8,
) -> Result<()> {
    const TRANSPOSE: [usize; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

    if model_bits == 0 {
        return Ok(());
    }

    for coefficient in TRANSPOSE.into_iter().skip(1) {
        let refinement = i32::try_from(reader.read_u32(model_bits)?)
            .expect("at most 15 refinement bits fit i32");
        coefficients[coefficient] = match coefficients[coefficient].cmp(&0) {
            std::cmp::Ordering::Greater => coefficients[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_add(refinement)),
            std::cmp::Ordering::Less => coefficients[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_sub(refinement)),
            std::cmp::Ordering::Equal if refinement != 0 && reader.read_bool()? => {
                Some(-refinement)
            }
            std::cmp::Ordering::Equal => Some(refinement),
        }
        .ok_or_else(|| {
            reader.error(ErrorKind::InvalidCodestream("lowpass coefficient overflow"))
        })?;
    }

    Ok(())
}

fn update_model(states: &mut [i32; 2], bits: &mut [u8; 2], laplacian: [i32; 2], models: usize) {
    for (model, laplacian) in laplacian.into_iter().enumerate().take(models) {
        let mut state = states[model];
        let mut delta = (laplacian - 70) >> 2;

        if delta <= -8 {
            delta = (delta + 4).max(-16);
            state += delta;

            if state < -8 {
                if bits[model] == 0 {
                    state = -8;
                } else {
                    state = 0;
                    bits[model] -= 1;
                }
            }
        } else if delta >= 8 {
            delta = (delta - 4).min(15);
            state += delta;

            if state > 8 {
                if bits[model] >= 15 {
                    bits[model] = 15;
                    state = 8;
                } else {
                    state = 0;
                    bits[model] += 1;
                }
            }
        }

        states[model] = state;
    }
}

#[derive(Clone, Debug)]
struct DcContext {
    luma_level: AdaptiveVLC,
    chroma_level: AdaptiveVLC,

    model_state: [i32; 2],
    model_bits: [u8; 2],
}

impl DcContext {
    const fn new() -> Self {
        Self {
            luma_level: AdaptiveVLC::two_tables(),
            chroma_level: AdaptiveVLC::two_tables(),

            model_state: [0, 0],
            model_bits: [8, 8],
        }
    }

    fn adapt(&mut self) {
        self.luma_level.adapt_two();
        self.chroma_level.adapt_two();
    }

    fn update_model(&mut self, format: InternalColorFormat, mut laplacian: [i32; 2]) {
        laplacian[0] *= 240;

        if format == InternalColorFormat::YUV444 {
            laplacian[1] *= 120;
        }

        let models = if format == InternalColorFormat::YOnly {
            1
        } else {
            2
        };
        for (model, laplacian) in laplacian.into_iter().enumerate().take(models) {
            let mut state = self.model_state[model];
            let mut delta = (laplacian - 70) >> 2;

            if delta <= -8 {
                delta = (delta + 4).max(-16);
                state += delta;

                if state < -8 {
                    if self.model_bits[model] == 0 {
                        state = -8;
                    } else {
                        state = 0;
                        self.model_bits[model] -= 1;
                    }
                }
            } else if delta >= 8 {
                delta = (delta - 4).min(15);
                state += delta;

                if state > 8 {
                    if self.model_bits[model] >= 15 {
                        self.model_bits[model] = 15;
                        state = 8;
                    } else {
                        state = 0;
                        self.model_bits[model] += 1;
                    }
                }
            }

            self.model_state[model] = state;
        }
    }
}

fn decode_dc_macroblock(
    reader: &mut BitReader<'_>,
    plane: &PlaneHeader,
    context: &mut DcContext,
) -> Result<Vec<i32>> {
    let components = usize::from(plane.component_count);
    let mut values = vec![0_i32; components];
    let mut laplacian = [0_i32; 2];

    match plane.internal_color_format {
        InternalColorFormat::YOnly => {
            let present = reader.read_bool()?;
            laplacian[0] += i32::from(present);
            values[0] = decode_dc_value(
                reader,
                context.model_bits[0],
                present,
                &mut context.luma_level,
            )?;
        }
        InternalColorFormat::YUV444 => {
            let present = entropy::val_dc_yuv(reader)?;

            for (component, value) in values.iter_mut().enumerate() {
                let model = usize::from(component != 0);
                let component_present = present & (4 >> component) != 0;
                laplacian[model] += i32::from(component_present);
                let level = if component == 0 {
                    &mut context.luma_level
                } else {
                    &mut context.chroma_level
                };

                *value =
                    decode_dc_value(reader, context.model_bits[model], component_present, level)?;
            }
        }
        _ => {
            return Err(reader.error(ErrorKind::Unsupported(
                "subsampled or multi-component DC coefficients",
            )));
        }
    }

    context.update_model(plane.internal_color_format, laplacian);

    Ok(values)
}

fn decode_dc_value(
    reader: &mut BitReader<'_>,
    model_bits: u8,
    absolute_level_present: bool,
    adaptive: &mut AdaptiveVLC,
) -> Result<i32> {
    let mut magnitude = if absolute_level_present {
        decode_absolute_level(reader, adaptive)? - 1
    } else {
        0
    };

    if model_bits != 0 {
        let refinement = reader.read_u32(model_bits)?;
        magnitude = magnitude
            .checked_shl(u32::from(model_bits))
            .and_then(|value| value.checked_add(refinement))
            .ok_or_else(|| reader.error(ErrorKind::InvalidCodestream("DC coefficient overflow")))?;
    }

    let magnitude = i32::try_from(magnitude).map_err(|_conversion_error| {
        reader.error(ErrorKind::InvalidCodestream("DC coefficient overflow"))
    })?;

    if magnitude != 0 && reader.read_bool()? {
        Ok(-magnitude)
    } else {
        Ok(magnitude)
    }
}

fn decode_absolute_level(reader: &mut BitReader<'_>, adaptive: &mut AdaptiveVLC) -> Result<u32> {
    const REMAP: [u32; 6] = [2, 3, 4, 6, 10, 14];
    const FIXED: [u8; 6] = [0, 0, 1, 2, 2, 2];

    let index = usize::from(entropy::abs_level_index(reader, adaptive)?);
    if index < 6 {
        return Ok(REMAP[index] + reader.read_u32(FIXED[index])?);
    }

    let mut fixed = u32::from(reader.read_u8(4)?) + 4;
    if fixed == 19 {
        fixed += u32::from(reader.read_u8(2)?);

        if fixed == 22 {
            fixed += u32::from(reader.read_u8(3)?);
        }
    }

    if fixed > 29 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "absolute coefficient level exceeds 32 bits",
        )));
    }

    let fixed = u8::try_from(fixed).expect("absolute-level width is at most 29");
    Ok(2 + (1_u32 << fixed) + reader.read_u32(fixed)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decoder, PixelFormat};

    #[test]
    fn simd_reconstruction_preserves_transform_and_row_layout() {
        let mut coefficients = [0; 16];
        coefficients[0] = 1024;
        inverse_transform_4x4(&mut coefficients, 0).unwrap();
        assert_eq!(coefficients, [256; 16]);

        let luma = [-512, -511, -1, 0, 1, 510, 511, 512, 1024];
        let mut values = Vec::from(luma);
        values.extend([0; 9]);
        values.extend([0; 9]);
        let color = IntegerImage {
            width: luma.len(),
            height: 1,
            components: 3,
            values,
        };
        let mut pixels = [0; 9];

        fill_bgr101010_row(&mut pixels, 0, 0, 0, &color, 0, 512, false);

        let expected = luma.map(|sample| {
            let channel = u32::try_from((sample + 512).clamp(0, 1023)).unwrap();
            channel | (channel << 10) | (channel << 20)
        });
        assert_eq!(pixels, expected);
    }

    #[test]
    #[ignore = "requires JPEGXR_SAMPLE to name a local conformance image"]
    fn decodes_real_sample_pixels() {
        let path = std::env::var("JPEGXR_SAMPLE").expect("set JPEGXR_SAMPLE");
        let bytes = std::fs::read(path).expect("read sample");
        let decoder = Decoder::new(&bytes).expect("parse sample headers");
        assert_eq!(decoder.info().pixel_format(), PixelFormat::RGBA128_FLOAT);
        let image = decoder.decode_rgba_f32().expect("decode sample pixels");
        assert_eq!((image.width(), image.height()), (3440, 1440));
        assert_eq!(image.pixels().len(), 3440 * 1440 * 4);
        assert_eq!(
            &image.pixels()[..8],
            &[
                0.001_037_597_7,
                0.001_907_348_6,
                0.002_914_428_7,
                1.0,
                0.001_037_597_7,
                0.001_907_348_6,
                0.002_914_428_7,
                1.0,
            ]
        );
    }

    #[test]
    #[ignore = "requires JPEGXR_BGR101010_SAMPLE to name a local conformance image"]
    fn decodes_real_bgr101010_sample_pixels() {
        let path = std::env::var("JPEGXR_BGR101010_SAMPLE").expect("set JPEGXR_BGR101010_SAMPLE");
        let bytes = std::fs::read(path).expect("read sample");
        let decoder = Decoder::new(&bytes).expect("parse sample headers");

        assert_eq!(decoder.info().pixel_format(), PixelFormat::BGR101010);

        let image = decoder
            .decode_bgr101010()
            .expect("decode BGR101010 sample pixels");

        assert_eq!((image.width(), image.height()), (3840, 2160));
        assert_eq!(image.pixels().len(), 3840 * 2160);
        assert!(image.pixels().iter().all(|pixel| pixel >> 30 == 0));
    }
}
