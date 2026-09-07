//! Decodes transform coefficients and reconstructs image samples.

use crate::bitstream::BitReader;
use crate::codestream::{
    Bands, ImageHeader, InternalColorFormat, Margins, OutputBitDepth, OutputColorFormat,
    OverlapMode, ParsedCodestream, PlaneHeader,
};
use crate::entropy::{self, AdaptiveVLC};
use crate::error::{Error, ErrorKind, Result};
use multiversion::multiversion;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::simd::{Simd, cmp::SimdOrd, num::SimdInt};

/// Largest accepted image width or height, in pixels.
///
/// This is a decoder-imposed safety limit, not a T.832 requirement: it keeps pixel-count and
/// byte-size arithmetic (and the allocations they size) representable and bounded well away from
/// pathological memory use, while remaining far larger than any real screenshot or HDR photo
/// dimension.
const MAX_DIMENSION: usize = 16_384;

/// Largest accepted total pixel count (width × height).
///
/// Bounds the size of the largest allocation this crate makes (the interleaved RGBA `f32` output
/// buffer) to a few hundred mebibytes. This is independent of [`MAX_DIMENSION`] above, which
/// alone cannot bound a very wide-and-short or tall-and-narrow image.
const MAX_PIXELS: usize = 64 * 1024 * 1024;

/// Number of 4×4 lowpass blocks handed to one Rayon job by [`inverse_lowpass_blocks`].
///
/// Sized to amortize per-job dispatch overhead while still splitting a typical image into several
/// jobs.
const LOWPASS_BLOCKS_PER_JOB: usize = 512;

/// Rayon crossover for the lowpass inverse transform.
///
/// Below this many blocks, [`inverse_lowpass_blocks`] runs serially rather than opening a
/// parallel scope that would cover fewer than four [`LOWPASS_BLOCKS_PER_JOB`] jobs.
const MIN_PARALLEL_LOWPASS_BLOCKS: usize = LOWPASS_BLOCKS_PER_JOB * 4;

/// Rayon crossover for highpass macroblock dequantization and prediction.
///
/// Below this many macroblocks, the work runs on the calling thread instead of splitting across
/// Rayon's pool.
const MIN_PARALLEL_MACROBLOCKS: usize = 512;

/// Rayon crossover, in pixels, shared by the final row-fill and plane-reconstruction stages
/// (color/alpha plane reconstruction, `BGR101010`/RGBA row filling, and band combination).
///
/// Below this many pixels, per-thread dispatch overhead is not worth paying.
const MIN_PARALLEL_PIXELS: usize = 256 * 1024;
const PIXEL_LANES: usize = 8;

type I32x8 = Simd<i32, PIXEL_LANES>;
type I64x2 = Simd<i64, 2>;
type I64x4 = Simd<i64, 4>;
type I64x8 = Simd<i64, PIXEL_LANES>;

#[derive(Debug)]
pub(crate) struct DCImage {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
}

#[derive(Debug)]
pub(crate) struct LowpassImage {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
}

#[derive(Debug)]
pub(crate) struct PredictedLowpass {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
    pub(crate) highpass_modes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct HighpassImage {
    pub(crate) macroblock_width: usize,
    pub(crate) macroblock_height: usize,
    pub(crate) components: usize,
    pub(crate) values: Vec<i32>,
    pub(crate) model_bits: Vec<[u8; 2]>,
}

#[derive(Debug)]
struct IntegerImage {
    width: usize,
    height: usize,
    components: usize,
    values: Vec<i32>,
}

/// Validates a plane's declared dimensions and returns `(width, height, pixel_count)`.
///
/// Shared by every entry point below: each rejects a width/height that doesn't fit `usize`,
/// exceeds [`MAX_DIMENSION`], or whose product exceeds [`MAX_PIXELS`] or overflows.
fn validated_dimensions(header: &ImageHeader, offset: usize) -> Result<(usize, usize, usize)> {
    let width = usize::try_from(header.width)
        .map_err(|_conversion_error| Error::new(ErrorKind::LimitExceeded("image width"), offset))?;
    let height = usize::try_from(header.height).map_err(|_conversion_error| {
        Error::new(ErrorKind::LimitExceeded("image height"), offset)
    })?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded("pixel count"), offset))?;

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::new(
            ErrorKind::LimitExceeded("image dimension"),
            offset,
        ));
    }

    if pixel_count > MAX_PIXELS {
        return Err(Error::new(ErrorKind::LimitExceeded("pixel count"), offset));
    }

    Ok((width, height, pixel_count))
}

struct CropRect {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

/// Computes the crop rectangle a plane's margins carve out of its decoded `width`x`height`,
/// rejecting an overflowing right/bottom edge. `width_message`/`height_message` let each call
/// site keep its own `ErrorKind::LimitExceeded` wording (e.g. "alpha image width").
fn crop_rect(
    margins: Margins,
    width: usize,
    height: usize,
    offset: usize,
    width_message: &'static str,
    height_message: &'static str,
) -> Result<CropRect> {
    let left = usize::from(margins.left);
    let top = usize::from(margins.top);
    let right = left
        .checked_add(width)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded(width_message), offset))?;
    let bottom = top
        .checked_add(height)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded(height_message), offset))?;

    Ok(CropRect {
        left,
        top,
        right,
        bottom,
    })
}

pub(crate) fn decode_rgba_f32(
    primary: &ParsedCodestream<'_>,
    alpha: &ParsedCodestream<'_>,
) -> Result<Vec<f32>> {
    validate_float_rgb_profile(primary, alpha)?;

    let (width, height, pixel_count) = validated_dimensions(&primary.header, primary.offset)?;

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

    let color_crop = crop_rect(
        primary.header.margins,
        width,
        height,
        primary.offset,
        "cropped image width",
        "cropped image height",
    )?;
    let alpha_crop = crop_rect(
        alpha.header.margins,
        width,
        height,
        alpha.offset,
        "alpha image width",
        "alpha image height",
    )?;

    if color.components != 3
        || alpha_image.components != 1
        || color_crop.right > color.width
        || color_crop.bottom > color.height
        || alpha_crop.right > alpha_image.width
        || alpha_crop.bottom > alpha_image.height
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
            color_crop.left,
            color_crop.top,
            alpha_crop.left,
            alpha_crop.top,
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

    let (width, height, pixel_count) = validated_dimensions(&stream.header, stream.offset)?;

    let color = reconstruct(stream)?;

    let crop = crop_rect(
        stream.header.margins,
        width,
        height,
        stream.offset,
        "cropped image width",
        "cropped image height",
    )?;
    let (left, top) = (crop.left, crop.top);

    if color.components != 3 || crop.right > color.width || crop.bottom > color.height {
        return Err(Error::new(
            ErrorKind::InvalidCodestream("decoded component dimensions are inconsistent"),
            stream.offset,
        ));
    }

    // Scaling: `scaled` planes carry three extra fractional bits that the row fill below removes.
    let shift = if stream.primary_plane.scaled { 3 } else { 0 };
    let rounding = if shift == 0 {
        0
    } else {
        (1 << (shift - 1)) - 1
    };
    let bias = (512_i64 << shift) + rounding;
    let swapped = stream.header.red_blue_swapped;

    // Allocation and per-row fill closure shared by the parallel and serial paths below.
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

#[multiversion(targets = "simd")]
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

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "clamped to 0..=1023 immediately above"
)]
fn clip_10_bit(value: i64) -> u32 {
    value.clamp(0, 1023) as u32
}

#[inline]
fn clip_10_bit_simd(value: I64x8) -> Simd<u32, PIXEL_LANES> {
    value.simd_clamp(I64x8::splat(0), I64x8::splat(1023)).cast()
}

#[inline]
fn inverse_color_transform(y: i32, u: i32, v: i32, bias: i64) -> [i64; 3] {
    let mut green = i64::from(y) + bias;
    let mut red = -i64::from(u);
    let mut blue = i64::from(v);

    green -= red >> 1;
    red -= ((blue + 1) >> 1) - green;
    blue += red;

    [red, green, blue]
}

#[inline]
fn inverse_color_transform_simd(y: I32x8, u: I32x8, v: I32x8, bias: i64) -> [I64x8; 3] {
    let mut green = y.cast::<i64>() + I64x8::splat(bias);
    let mut red = -u.cast::<i64>();
    let mut blue = v.cast::<i64>();

    green -= red >> 1;
    red -= ((blue + I64x8::splat(1)) >> 1) - green;
    blue += red;

    [red, green, blue]
}

#[multiversion(targets = "simd")]
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
    let (pixels, remainder) = row.as_chunks_mut::<4>();
    debug_assert_eq!(remainder, []);

    // Slice each plane once so the pixel loop needs no flat-index bounds checks.
    let color_component_len = color.width * color.height;
    let color_start = (y + color_top) * color.width + color_left;
    let color_end = color_start + pixels.len();
    let luma = &color.values[color_start..color_end];
    let chroma_u =
        &color.values[color_component_len + color_start..color_component_len + color_end];
    let chroma_v =
        &color.values[2 * color_component_len + color_start..2 * color_component_len + color_end];

    let alpha_start = (y + alpha_top) * alpha.width + alpha_left;
    let alphas = &alpha.values[alpha_start..alpha_start + pixels.len()];

    let color_samples = luma.iter().zip(chroma_u).zip(chroma_v);
    for ((pixel, ((luma, chroma_u), chroma_v)), alpha) in
        pixels.iter_mut().zip(color_samples).zip(alphas)
    {
        let [red, green, blue] = inverse_color_transform(*luma, *chroma_u, *chroma_v, 0);
        pixel[0] = color_format.convert(red)?;
        pixel[1] = color_format.convert(green)?;
        pixel[2] = color_format.convert(blue)?;
        pixel[3] = alpha_format.convert(i64::from(*alpha))?;
    }

    Ok(())
}

pub(crate) fn decode_dc(stream: &ParsedCodestream<'_>) -> Result<DCImage> {
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

    let mut image = DCImage {
        macroblock_width,
        macroblock_height,
        components,
        values: vec![0; value_count],
    };

    decode_tiles_into(
        stream,
        &mut image.values,
        macroblock_width,
        components,
        |tile, values| {
            decode_dc_packet(
                packet(stream, tile.index, 0)?,
                &stream.primary_plane,
                tile.width,
                tile.height,
                components,
                values,
            )
        },
    )?;

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

    decode_tiles_into(
        stream,
        &mut image.values,
        macroblock_width,
        components * 16,
        |tile, values| {
            decode_lowpass_packet(
                packet(stream, tile.index, 1)?,
                &stream.primary_plane,
                tile.width,
                tile.height,
                components,
                values,
            )
        },
    )?;

    Ok(image)
}

/// One tile of the tile grid, in macroblock units.
struct Tile {
    index: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

fn tile_grid(stream: &ParsedCodestream<'_>) -> Vec<Tile> {
    let mut tiles =
        Vec::with_capacity(stream.header.tile_widths.len() * stream.header.tile_heights.len());

    let mut top = 0_usize;
    for tile_height in stream.header.tile_heights.iter().copied() {
        let mut left = 0_usize;
        for tile_width in stream.header.tile_widths.iter().copied() {
            tiles.push(Tile {
                index: tiles.len(),
                left,
                top,
                width: usize::from(tile_width),
                height: usize::from(tile_height),
            });
            left += usize::from(tile_width);
        }

        top += usize::from(tile_height);
    }

    tiles
}

/// Decodes every tile packet into a tile-local buffer, in parallel for large images, then
/// scatters the rows into `values` (`stride` values per macroblock).
///
/// Tile packets are independent bitstreams whose prediction resets at tile edges, so tiles can
/// decode in any order; only the scatter needs the global macroblock layout.
fn decode_tiles_into(
    stream: &ParsedCodestream<'_>,
    values: &mut [i32],
    macroblock_width: usize,
    stride: usize,
    decode_packet: impl Fn(&Tile, &mut [i32]) -> Result<()> + Sync,
) -> Result<()> {
    let tiles = tile_grid(stream);
    let decode_tile = |tile: &Tile| {
        let mut local = vec![0_i32; tile.width * tile.height * stride];
        decode_packet(tile, &mut local)?;
        Ok(local)
    };

    let macroblock_count = values.len() / stride.max(1);
    let tile_values = if macroblock_count >= MIN_PARALLEL_MACROBLOCKS && tiles.len() > 1 {
        tiles
            .par_iter()
            .map(decode_tile)
            .collect::<Result<Vec<_>>>()?
    } else {
        tiles.iter().map(decode_tile).collect::<Result<Vec<_>>>()?
    };

    for (tile, local) in tiles.iter().zip(&tile_values) {
        let row_len = tile.width * stride;
        for local_y in 0..tile.height {
            let global_row = (tile.top + local_y) * macroblock_width + tile.left;
            values[global_row * stride..][..row_len]
                .copy_from_slice(&local[local_y * row_len..][..row_len]);
        }
    }

    Ok(())
}

pub(crate) fn predict_lowpass(
    stream: &ParsedCodestream<'_>,
    dc: &DCImage,
    lowpass: LowpassImage,
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

    let mut raw = lowpass.values;
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

                        raw[start] = dc.values[macroblock * dc.components + component];

                        predict_dc(
                            &mut raw,
                            dc.macroblock_width,
                            dc.components,
                            macroblock,
                            component,
                            dc_mode,
                            stream.offset,
                        )?;

                        predict_lp(
                            &mut raw,
                            dc.macroblock_width,
                            dc.components,
                            macroblock,
                            component,
                            dc_mode,
                            stream.offset,
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

    // Highpass tiles also decode their flexbits refinement while the tile is local, so the
    // scatter below runs once over finished coefficients.
    let tiles = tile_grid(stream);
    let decode_tile = |tile: &Tile| decode_highpass_tile(stream, tile, components, lowpass);
    let mut tile_images = if macroblock_count >= MIN_PARALLEL_MACROBLOCKS && tiles.len() > 1 {
        tiles
            .par_iter()
            .map(decode_tile)
            .collect::<Result<Vec<_>>>()?
    } else {
        tiles.iter().map(decode_tile).collect::<Result<Vec<_>>>()?
    };

    if tile_images.len() == 1 {
        // Parsing guarantees the tile grid exactly covers the macroblock grid.
        let only = tile_images.pop().expect("one tile image per tile");
        debug_assert_eq!(only.macroblock_width, macroblock_width);
        debug_assert_eq!(only.macroblock_height, macroblock_height);
        return Ok(only);
    }

    let mut image = HighpassImage {
        macroblock_width,
        macroblock_height,
        components,
        values: vec![0; value_count],
        model_bits: vec![[0; 2]; macroblock_count],
    };

    for (tile, tile_image) in tiles.iter().zip(&tile_images) {
        let row_len = tile.width * components * 256;
        for local_y in 0..tile.height {
            let global_row = (tile.top + local_y) * macroblock_width + tile.left;
            let local_row = local_y * tile.width;

            image.values[global_row * components * 256..][..row_len]
                .copy_from_slice(&tile_image.values[local_row * components * 256..][..row_len]);
            image.model_bits[global_row..][..tile.width]
                .copy_from_slice(&tile_image.model_bits[local_row..][..tile.width]);
        }
    }

    Ok(image)
}

/// Decodes one tile's highpass packet and its flexbits refinement into a tile-local image.
fn decode_highpass_tile(
    stream: &ParsedCodestream<'_>,
    tile: &Tile,
    components: usize,
    lowpass: &PredictedLowpass,
) -> Result<HighpassImage> {
    let macroblock_count = tile.width * tile.height;
    let mut modes = Vec::with_capacity(macroblock_count);
    for y in tile.top..tile.top + tile.height {
        let start = y * lowpass.macroblock_width + tile.left;
        modes.extend_from_slice(&lowpass.highpass_modes[start..start + tile.width]);
    }

    let mut image = HighpassImage {
        macroblock_width: tile.width,
        macroblock_height: tile.height,
        components,
        values: vec![0; macroblock_count * components * 256],
        model_bits: vec![[0; 2]; macroblock_count],
    };

    decode_highpass_packet(
        packet(stream, tile.index, 2)?,
        &stream.primary_plane,
        &modes,
        &mut image,
    )?;

    match stream.primary_plane.bands {
        Bands::All => {
            if image.model_bits.iter().any(|bits| *bits != [0, 0]) {
                decode_flexbits_packet(
                    packet(stream, tile.index, 3)?,
                    stream.header.trim_flexbits,
                    &mut image,
                )?;
            }
        }
        Bands::NoFlexbits => shift_highpass_without_flexbits(&mut image)?,
        Bands::NoHighpass | Bands::DCOnly => {
            return Err(Error::new(
                ErrorKind::Unsupported("highpass band is absent"),
                stream.offset + stream.tiles_offset,
            ));
        }
    }

    Ok(image)
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
    let mut lowpass = predict_lowpass(stream, &dc, lowpass)?;

    let mut highpass = decode_highpass(stream, &lowpass)?;
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

#[multiversion(targets = "simd")]
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

#[multiversion(targets = "simd")]
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

#[inline]
fn inverse_transform_4x4(coefficients: &mut [i32; 16], offset: usize) -> Result<()> {
    // Regroups the 16 macroblock-relative coefficient positions (T.832's raster scan order) into
    // the four disjoint 2x2 groups the butterfly stages below each expect contiguously: `[0, 1,
    // 4, 5]` for the DC/lowpass rotate (`t2x2`), `[2, 3, 6, 7, 8, 9, 12, 13]` for the two odd
    // pairs (`inverse_odd_pair`), and `[10, 11, 14, 15]` for the remaining odd-odd pair
    // (`inverse_odd_odd`). `t2x2_quad` then recombines across all four groups. This undoes the
    // forward encoder's equivalent regrouping permutation.
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

#[inline]
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

#[inline]
fn inverse_odd_pair(values: &mut [i64; 16]) {
    let mut first = I64x2::from_array([values[2], values[8]]);
    let mut second = I64x2::from_array([values[3], values[12]]);
    let mut third = I64x2::from_array([values[6], values[9]]);
    let mut fourth = I64x2::from_array([values[7], values[13]]);

    // Cross-couple the two odd pairs before rotating them.
    second += fourth;
    first -= third;

    // Half-step lifting prediction.
    fourth -= second >> 1;
    third += (first + I64x2::splat(1)) >> 1;

    // Four-step lifting approximation of a Givens rotation.
    first -= (I64x2::splat(3) * second + I64x2::splat(4)) >> 3;
    second += (I64x2::splat(3) * first + I64x2::splat(4)) >> 3;
    third -= (I64x2::splat(3) * fourth + I64x2::splat(4)) >> 3;
    fourth += (I64x2::splat(3) * third + I64x2::splat(4)) >> 3;

    // Final half-step combine back into the two odd pairs.
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

#[inline]
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

#[inline]
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

#[inline]
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
    offset: usize,
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

    raw[current] = raw[current].checked_add(prediction).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidCodestream("predicted DC overflow"),
            offset,
        )
    })?;

    Ok(())
}

fn predict_lp(
    raw: &mut [i32],
    width: usize,
    components: usize,
    macroblock: usize,
    component: usize,
    dc_mode: u8,
    offset: usize,
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
                        offset,
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
    debug_assert!(
        scaled_shift <= 1,
        "quant_map is only proven safe for scaled_shift 0 or 1"
    );

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

/// Splits `values` into per-tile-row bands, in tile order.
///
/// Band `i` covers `row_heights[i]` macroblock rows of `row_stride` elements each — the layout
/// `packet` boundaries assume, and the same layout every `decode_*_packet` writes with a
/// band-relative row index in place of a `top` offset.
fn decode_dc_packet(
    packet: Packet<'_>,
    plane: &PlaneHeader,
    width: usize,
    height: usize,
    components: usize,
    values: &mut [i32],
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
            let macroblock = decode_dc_macroblock(&mut reader, plane, &mut context)?;
            let start = (local_y * width + local_x) * components;
            values[start..start + components].copy_from_slice(&macroblock[..components]);

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
    components: usize,
    values: &mut [i32],
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

            let macroblock = decode_lowpass_macroblock(&mut reader, plane, &mut context)?;
            let start = (local_y * width + local_x) * components * 16;
            values[start..start + components * 16].copy_from_slice(&macroblock[..components * 16]);

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

fn decode_highpass_packet(
    packet: Packet<'_>,
    plane: &PlaneHeader,
    modes: &[u8],
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

    let (width, height, components) = (
        image.macroblock_width,
        image.macroblock_height,
        image.components,
    );
    let mut cbphp = vec![0_u16; image.model_bits.len() * components];

    for local_y in 0..height {
        for local_x in 0..width {
            if local_x.is_multiple_of(16) {
                context.horizontal_scan.reset_totals();
                context.vertical_scan.reset_totals();
            }

            let macroblock = local_y * width + local_x;

            let patterns = decode_cbphp(
                &mut reader,
                plane,
                &mut context,
                &cbphp,
                width,
                macroblock,
                local_x == 0,
                local_y == 0,
            )?;

            cbphp[macroblock * components..(macroblock + 1) * components]
                .copy_from_slice(&patterns[..components]);

            decode_highpass_macroblock(
                &mut reader,
                plane,
                &mut context,
                macroblock,
                modes[macroblock],
                &patterns[..components],
                components,
                &mut image.values,
                &mut image.model_bits,
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

fn decode_flexbits_packet(
    packet: Packet<'_>,
    trim_present: bool,
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
    let components = highpass.components;

    for macroblock in 0..highpass.model_bits.len() {
        for component in 0..components {
            let model = usize::from(component != 0);
            let model_bits = highpass.model_bits[macroblock][model];
            let flex_bits = model_bits.saturating_sub(trim);

            for block in HIERARCHICAL_ORDER {
                let start = (macroblock * components + component) * 256 + block * 16;
                let values: &mut [i32; 16] = (&mut highpass.values[start..start + 16])
                    .try_into()
                    .expect("block slice has length 16");

                for coefficient in TRANSPOSE.into_iter().skip(1) {
                    let vlc = values[coefficient];

                    let refinement = if flex_bits == 0 {
                        0
                    } else {
                        i32::try_from(reader.read_u32(flex_bits)?)
                            .expect("at most 15 flexbits fit i32")
                    };

                    let flex = match vlc.cmp(&0) {
                        Ordering::Greater => refinement,
                        Ordering::Less => -refinement,
                        Ordering::Equal if refinement != 0 && reader.read_bool()? => -refinement,
                        Ordering::Equal => refinement,
                    };

                    let flex = flex.checked_shl(u32::from(trim)).ok_or_else(|| {
                        reader.error(ErrorKind::InvalidCodestream(
                            "flexbits coefficient overflow",
                        ))
                    })?;

                    values[coefficient] = vlc
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

    reader.align_zero()?;

    if reader.byte_position() != packet.bytes.len() {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "flexbits tile packet does not end at its indexed boundary",
        )));
    }

    Ok(())
}

/// The eight adaptive VLC tables the lowpass and highpass coefficient decoders share.
///
/// Both bands read a first index, then a run of subsequent indices, then absolute levels, using
/// the same tables and the same adaptation schedule. Holding them in one place keeps that schedule
/// from drifting between the two.
#[derive(Clone, Debug)]
struct BandVLC {
    first_luma: AdaptiveVLC,
    index_luma_zero: AdaptiveVLC,
    index_luma_one: AdaptiveVLC,
    first_chroma: AdaptiveVLC,
    index_chroma_zero: AdaptiveVLC,
    index_chroma_one: AdaptiveVLC,
    level_zero: AdaptiveVLC,
    level_one: AdaptiveVLC,
}

impl BandVLC {
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
}

/// Names the two error messages that differ between the bands.
#[derive(Clone, Copy, Debug)]
struct BandMessages {
    run_exceeds_block: &'static str,
    position_exceeds_block: &'static str,
}

const LOWPASS_MESSAGES: BandMessages = BandMessages {
    run_exceeds_block: "lowpass coefficient run exceeds block",
    position_exceeds_block: "lowpass coefficient position exceeds block",
};

const HIGHPASS_MESSAGES: BandMessages = BandMessages {
    run_exceeds_block: "highpass coefficient run exceeds block",
    position_exceeds_block: "highpass coefficient position exceeds block",
};

/// Decodes one sixteen-coefficient block from either frequency band.
///
/// The lowpass and highpass decoders were sixty-six line-for-line identical lines apart from two
/// error strings and the choice of scan table. `scan` is taken separately from `vlc` so a caller
/// can borrow the two disjoint fields of its own context.
fn decode_band_block(
    reader: &mut BitReader<'_>,
    vlc: &mut BandVLC,
    scan: &mut AdaptiveScan,
    chroma: bool,
    coefficients: &mut [i32; 16],
    messages: BandMessages,
) -> Result<i32> {
    let first = vlc.decode_first(reader, chroma)?;
    let mut continuing = first >> 2;
    let mut level_context = (first & 1) & continuing;
    let negative = reader.read_bool()?;

    let magnitude = if first & 2 != 0 {
        vlc.decode_level(reader, level_context != 0)?
    } else {
        1
    };

    let mut value = signed_level(reader, magnitude, negative)?;
    let mut position = 1_u8;

    if first & 1 == 0 {
        position += entropy::run(reader, 14)?;
    }

    scan.place(coefficients, position, value);
    let mut location = position + 1;
    let mut nonzero = 1_i32;

    while continuing != 0 {
        if continuing & 1 == 0 {
            let maximum = 15_u8.checked_sub(location).ok_or_else(|| {
                reader.error(ErrorKind::InvalidCodestream(messages.run_exceeds_block))
            })?;

            position = location + entropy::run(reader, maximum)?;
        } else {
            position = location;
        }

        location = position + 1;

        if location > 16 {
            return Err(reader.error(ErrorKind::InvalidCodestream(
                messages.position_exceeds_block,
            )));
        }

        let index = vlc.decode_index(reader, chroma, level_context != 0, location)?;

        continuing = index >> 1;
        level_context &= continuing;

        let negative = reader.read_bool()?;

        let magnitude = if index & 1 != 0 {
            vlc.decode_level(reader, level_context != 0)?
        } else {
            1
        };

        value = signed_level(reader, magnitude, negative)?;
        scan.place(coefficients, position, value);
        nonzero += 1;
    }

    Ok(nonzero)
}

#[derive(Clone, Debug)]
struct HighpassContext {
    vlc: BandVLC,

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
            vlc: BandVLC::new(),

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

    fn adapt(&mut self) {
        self.vlc.adapt();
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

#[expect(
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
) -> Result<[u16; 3]> {
    const FIXED_LENGTH: [u8; 6] = [0, 2, 1, 2, 2, 0];
    const OFFSET: [u8; 6] = [0, 4, 2, 8, 12, 1];
    const OUTPUT: [u8; 16] = [0, 15, 3, 12, 1, 2, 4, 8, 5, 6, 9, 10, 7, 11, 13, 14];

    let components = usize::from(plane.component_count);
    let group_count = entropy::num_cbphp(reader, &mut context.num_cbphp)?;
    let groups = refine_cbphp(reader, group_count)?;
    let mut residual = [0_u16; 3];

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

    for (component, value) in residual[..components].iter_mut().enumerate() {
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
    components: usize,
    value_band: &mut [i32],
    model_bits_band: &mut [[u8; 2]],
) -> Result<()> {
    const HIERARCHICAL_ORDER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

    model_bits_band[macroblock] = context.model_bits;
    let mut laplacian = [0_i32; 2];

    for (component, pattern) in patterns.iter().copied().enumerate() {
        let model = usize::from(component != 0);
        let mut pattern = pattern;
        for block in HIERARCHICAL_ORDER {
            if pattern & 1 != 0 {
                let start = (macroblock * components + component) * 256 + block * 16;

                let coefficients: &mut [i32; 16] = (&mut value_band[start..start + 16])
                    .try_into()
                    .expect("highpass block has 16 coefficients");

                // Split-borrow the context: the shared tables and the chosen scan are disjoint
                // fields, so both can be handed to the decoder at once.
                let scan = if mode == 1 {
                    &mut context.vertical_scan
                } else {
                    &mut context.horizontal_scan
                };

                laplacian[model] += decode_band_block(
                    reader,
                    &mut context.vlc,
                    scan,
                    component != 0,
                    coefficients,
                    HIGHPASS_MESSAGES,
                )?;
            }
            pattern >>= 1;
        }
    }

    context.update_model(plane.internal_color_format, laplacian);

    Ok(())
}

#[derive(Clone, Debug)]
struct LowpassContext {
    vlc: BandVLC,

    count_zero: i8,
    count_maximum: i8,

    model_state: [i32; 2],
    model_bits: [u8; 2],

    scan: AdaptiveScan,
}

impl LowpassContext {
    const fn new() -> Self {
        Self {
            vlc: BandVLC::new(),

            count_zero: 1,
            count_maximum: 1,

            model_state: [0, 0],
            model_bits: [4, 4],

            scan: AdaptiveScan::new_lowpass(),
        }
    }

    fn adapt(&mut self) {
        self.vlc.adapt();
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
) -> Result<[i32; 48]> {
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

    let mut output = [0_i32; 48];
    let mut laplacian = [0_i32; 2];

    for component in 0..components {
        let model = usize::from(component != 0);
        let coefficients: &mut [i32; 16] = (&mut output[component * 16..(component + 1) * 16])
            .try_into()
            .expect("component coefficient slice has length 16");

        let nonzero = if coded_pattern & (1 << component) != 0 {
            decode_band_block(
                reader,
                &mut context.vlc,
                &mut context.scan,
                component != 0,
                coefficients,
                LOWPASS_MESSAGES,
            )?
        } else {
            0
        };

        laplacian[model] += nonzero;
        refine_lowpass(reader, coefficients, context.model_bits[model])?;
    }

    context.update_model(plane.internal_color_format, laplacian);

    Ok(output)
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
            Ordering::Greater => coefficients[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_add(refinement)),
            Ordering::Less => coefficients[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_sub(refinement)),
            Ordering::Equal if refinement != 0 && reader.read_bool()? => Some(-refinement),
            Ordering::Equal => Some(refinement),
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

        // The adaptation state machine is shared with the lowpass and highpass contexts. It used
        // to be spelled out a second time here, so a correction to the T.832 rules had to be
        // applied in two places sixty lines apart.
        update_model(
            &mut self.model_state,
            &mut self.model_bits,
            laplacian,
            models,
        );
    }
}

fn decode_dc_macroblock(
    reader: &mut BitReader<'_>,
    plane: &PlaneHeader,
    context: &mut DcContext,
) -> Result<[i32; 3]> {
    let mut values = [0_i32; 3];
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
    fn rgba_rows_read_color_and_alpha_at_their_own_strides() {
        // The color and alpha planes have different widths and different origins, which is the
        // case a shared index expression gets wrong. Expected samples are IEEE-754 words written
        // out directly rather than recomputed with `convert`, so this does not assert the
        // implementation against itself.
        const COLOR_WIDTH: usize = 6;
        const COLOR_HEIGHT: usize = 3;
        const ALPHA_WIDTH: usize = 5;
        const ALPHA_HEIGHT: usize = 3;
        const POISON: i32 = 0x7f7f_7f7f;

        let luma = [0x3f80_0000, 0x4000_0000, 0x3e80_0000];
        let chroma_u = [-0x0080_0000, 0x0040_0000, 0x0000_0000];
        let chroma_v = [0x3f00_0000, 0x3f80_0000, 0x4040_0000];
        let alpha_samples = [0x3f80_0000, 0x3f00_0000, 0x0000_0000];

        // Row 1, columns 2..5 of each color plane; row 2, columns 1..4 of the alpha plane.
        let (color_left, color_top) = (2, 1);
        let (alpha_left, alpha_top) = (1, 2);
        let color_start = (color_top * COLOR_WIDTH) + color_left;
        let alpha_start = (alpha_top * ALPHA_WIDTH) + alpha_left;

        let plane_len = COLOR_WIDTH * COLOR_HEIGHT;
        let mut values = vec![POISON; plane_len * 3];
        for (component, samples) in [luma, chroma_u, chroma_v].into_iter().enumerate() {
            for (index, sample) in samples.into_iter().enumerate() {
                values[component * plane_len + color_start + index] = sample;
            }
        }
        let color = IntegerImage {
            width: COLOR_WIDTH,
            height: COLOR_HEIGHT,
            components: 3,
            values,
        };

        let mut values = vec![POISON; ALPHA_WIDTH * ALPHA_HEIGHT];
        for (index, sample) in alpha_samples.into_iter().enumerate() {
            values[alpha_start + index] = sample;
        }

        let alpha = IntegerImage {
            width: ALPHA_WIDTH,
            height: ALPHA_HEIGHT,
            components: 1,
            values,
        };

        // Identity BD32F parameters: `convert` then reassembles the sample word unchanged.
        let format = FloatFormat {
            mantissa_bits: 23,
            exponent_bias: 127,
            offset: 0,
        };

        let mut row = [0.0_f32; 12];

        fill_rgba_row(
            &mut row, 0, color_left, color_top, alpha_left, alpha_top, &color, &alpha, format,
            format,
        )
        .unwrap();

        let expected = [
            [0x2040_0000_u32, 0x3f40_0000, 0x5f40_0000, 0x3f80_0000],
            [0x2020_0000, 0x4020_0000, 0x5fa0_0000, 0x3f00_0000],
            [0x1e60_0000, 0x3e80_0000, 0x5ea0_0000, 0x0000_0000],
        ];

        let actual: Vec<u32> = row.iter().map(|sample| sample.to_bits()).collect();
        assert_eq!(actual, expected.concat());
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
    fn decodes_real_bgr101010_sample_pixels() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/screenshot.jxr");

        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));

        let decoder = Decoder::new(&bytes).expect("parse screenshot.jxr headers");

        assert_eq!(decoder.info().pixel_format(), PixelFormat::BGR101010);

        let image = decoder
            .decode_bgr101010()
            .expect("decode BGR101010 sample pixels");

        assert_eq!((image.width(), image.height()), (3840, 2160));
        assert_eq!(image.pixels().len(), 3840 * 2160);
        assert!(image.pixels().iter().all(|pixel| pixel >> 30 == 0));

        // Dimensions, length, and the reserved-bits check above all pass for an all-zero buffer or
        // a row/tile permutation of the correct pixels. Hash the full decoded buffer so the
        // multi-tile-row parallel decode path has a byte-identical regression oracle.
        assert_eq!(fnv1a(image.pixels()), 13_953_505_876_719_867_180);
    }

    /// Hashes decoded pixels with stable FNV-1a.
    fn fnv1a(pixels: &[u32]) -> u64 {
        const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;

        let mut hash = OFFSET_BASIS;
        for pixel in pixels {
            for byte in pixel.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(PRIME);
            }
        }
        hash
    }
}
