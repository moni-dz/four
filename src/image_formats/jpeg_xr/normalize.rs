//! Writes SDR and tone-mapped HDR pixels into the output RGBA8 buffer.

use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};

use multiversion::multiversion;
use rayon::prelude::*;
use tonemapping::{Clamp, LinearRGB, LinearRGBPlanes, ToneMapper, ToneMappingMethod, exp2, log2};

use super::hdr::{
    HDRAnalysis, display_luminance_white_point, display_white_point, hdr_luminance_white_point,
    hdr_white_point,
};
use super::pixel::{
    PixelLayout, SampleEncoding, decode_bgr101010, decode_bgr101010_simd, decode_rgba128_float,
    decode_rgba128_float_simd,
};
use super::{
    BT2446_INPUT_SCALE, Error, F32x8, HDR_BATCH_PIXELS, JPEGXRError, PARALLEL_PIXELS_MIN,
    PARALLEL_PIXELS_PER_JOB, Result, SRGB_LANES, error, round_clamp_u8,
};

/// Scales scRGB into the fixed unit each tone mapper expects; only `BT2446` needs rescaling.
fn hdr_color_scale(method: ToneMappingMethod) -> f32 {
    if method == ToneMappingMethod::BT2446 {
        BT2446_INPUT_SCALE
    } else {
        1.0
    }
}

pub(super) fn write_normalized_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    method: ToneMappingMethod,
    analysis: Option<HDRAnalysis>,
    rgba: &mut [u8],
) -> Result<bool> {
    if !layout.encoding.is_hdr() {
        return write_pixel_slabs(source, width, row_stride, rgba, |source, rgba| {
            write_sdr_pixels(source, width, row_stride, layout, rgba)
        });
    }

    if method == ToneMappingMethod::Clamp {
        return write_pixel_slabs(source, width, row_stride, rgba, |source, rgba| {
            write_hdr_pixels_scalar(source, width, row_stride, layout, &Clamp, 1.0, rgba)
        });
    }

    let white_point = analysis
        .and_then(|analysis| analysis.max_cll)
        .map_or_else(display_white_point, hdr_white_point);
    let luminance_white_point = analysis
        .and_then(|analysis| analysis.luminance_white_point)
        .map_or_else(display_luminance_white_point, hdr_luminance_white_point);
    let mapper = method.resolve(white_point, luminance_white_point);
    let color_scale = hdr_color_scale(method);

    write_pixel_slabs(source, width, row_stride, rgba, |source, rgba| {
        write_hdr_pixels(
            source,
            width,
            row_stride,
            layout,
            &mapper,
            color_scale,
            rgba,
        )
    })
}

pub(super) fn write_pixel_slabs(
    source: &[u8],
    width: usize,
    row_stride: usize,
    rgba: &mut [u8],
    writer: impl Fn(&[u8], &mut [u8]) -> Result<bool> + Sync,
) -> Result<bool> {
    let row_count = source.len() / row_stride;

    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");

    if pixel_count < PARALLEL_PIXELS_MIN {
        return writer(source, rgba);
    }

    let rows_per_job = PARALLEL_PIXELS_PER_JOB.div_ceil(width);
    let source_bytes_per_job = rows_per_job * row_stride;
    let rgba_bytes_per_job = rows_per_job * width * 4;

    source
        .par_chunks(source_bytes_per_job)
        .zip(rgba.par_chunks_mut(rgba_bytes_per_job))
        .map(|(source, rgba)| writer(source, rgba))
        .try_reduce(|| false, |left, right| Ok::<bool, Error>(left || right))
}

pub(super) fn write_sdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    rgba: &mut [u8],
) -> Result<bool> {
    let mut has_nonzero_alpha = false;
    for (row, target_row) in source
        .chunks_exact(row_stride)
        .zip(rgba.chunks_exact_mut(width * 4))
    {
        let (targets, remainder) = target_row.as_chunks_mut::<4>();

        invariant!(remainder.is_empty());

        for (x, target) in targets.iter_mut().enumerate() {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;

            has_nonzero_alpha |= alpha > 0.0;

            let color = color.map(normalized_to_u8);
            target.copy_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }

    Ok(has_nonzero_alpha)
}

pub(super) fn write_hdr_pixels_scalar(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &(impl ToneMapper + Sync),
    color_scale: f32,
    rgba: &mut [u8],
) -> Result<bool> {
    let mut has_nonzero_alpha = false;
    for (row, target_row) in source
        .chunks_exact(row_stride)
        .zip(rgba.chunks_exact_mut(width * 4))
    {
        let (targets, remainder) = target_row.as_chunks_mut::<4>();

        invariant!(remainder.is_empty());

        for (x, target) in targets.iter_mut().enumerate() {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;

            has_nonzero_alpha |= alpha > 0.0;

            let color = color.map(|component| component * color_scale);
            let color = display_linear_to_srgb8(mapper.map(LinearRGB::new(color)));
            target.copy_from_slice(&[color[0], color[1], color[2], normalized_to_u8(alpha)]);
        }
    }

    Ok(has_nonzero_alpha)
}

pub(super) fn write_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &(impl ToneMapper + Sync),
    color_scale: f32,
    rgba: &mut [u8],
) -> Result<bool> {
    if layout.encoding == SampleEncoding::PackedBGR101010 {
        return Ok(write_bgr101010_hdr_pixels(
            source,
            width,
            row_stride,
            mapper,
            color_scale,
            rgba,
        ));
    }

    if layout == PixelLayout::rgba128_float() {
        return Ok(write_rgba128_float_hdr_pixels(
            source,
            width,
            row_stride,
            mapper,
            color_scale,
            rgba,
        ));
    }

    let row_count = source.len() / row_stride;

    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");

    let batch_capacity = HDR_BATCH_PIXELS.min(pixel_count);
    let mut colors = LinearRGBPlanes::with_capacity(batch_capacity);
    let mut alphas = Vec::with_capacity(batch_capacity);
    let mut has_nonzero_alpha = false;
    let mut rgba_offset = 0;

    for row in source.chunks_exact(row_stride) {
        for x in 0..width {
            let pixel = pixel_at(row, x, layout)?;
            let (color, alpha) = layout.read_pixel(pixel)?;

            has_nonzero_alpha |= alpha > 0.0;

            let color = color.map(|component| component * color_scale);
            colors.push(LinearRGB::new(color));
            alphas.push(normalized_to_u8(alpha));

            if colors.len() == HDR_BATCH_PIXELS {
                write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
            }
        }
    }

    write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
    invariant_eq!(rgba_offset, rgba.len());
    Ok(has_nonzero_alpha)
}

#[multiversion(targets = "simd")]
pub(super) fn write_bgr101010_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    mapper: &(impl ToneMapper + Sync),
    color_scale: f32,
    rgba: &mut [u8],
) -> bool {
    let row_count = source.len() / row_stride;
    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");
    let batch_capacity = HDR_BATCH_PIXELS.min(pixel_count);
    let mut colors = LinearRGBPlanes::with_capacity(batch_capacity);
    let mut alphas = Vec::with_capacity(batch_capacity);
    let mut rgba_offset = 0;

    for row in source.chunks_exact(row_stride) {
        let (pixels, remainder) = row.as_chunks::<4>();
        invariant!(remainder.is_empty());
        invariant_eq!(pixels.len(), width);

        let (chunks, tail) = pixels.as_chunks::<SRGB_LANES>();
        for chunk in chunks {
            let packed = Simd::<u32, SRGB_LANES>::from_array((*chunk).map(u32::from_ne_bytes));
            let [red, green, blue] = decode_bgr101010_simd(packed).map(Simd::to_array);

            for ((red, green), blue) in red.into_iter().zip(green).zip(blue) {
                colors.push(LinearRGB::new([
                    red * color_scale,
                    green * color_scale,
                    blue * color_scale,
                ]));
                alphas.push(u8::MAX);
            }

            if colors.len() >= HDR_BATCH_PIXELS {
                write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
            }
        }

        for pixel in tail {
            let color = decode_bgr101010(pixel).map(|component| component * color_scale);
            colors.push(LinearRGB::new(color));
            alphas.push(u8::MAX);

            if colors.len() == HDR_BATCH_PIXELS {
                write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
            }
        }
    }

    write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
    invariant_eq!(rgba_offset, rgba.len());
    true
}

#[multiversion(targets = "simd")]
pub(super) fn write_rgba128_float_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    mapper: &(impl ToneMapper + Sync),
    color_scale: f32,
    rgba: &mut [u8],
) -> bool {
    let row_count = source.len() / row_stride;
    let pixel_count = width
        .checked_mul(row_count)
        .expect("validated JPEG XR pixel count fits usize");
    let batch_capacity = HDR_BATCH_PIXELS.min(pixel_count);
    let mut colors = LinearRGBPlanes::with_capacity(batch_capacity);
    let mut alphas = Vec::with_capacity(batch_capacity);
    let mut has_nonzero_alpha = false;
    let mut rgba_offset = 0;

    for row in source.chunks_exact(row_stride) {
        let (pixels, remainder) = row.as_chunks::<16>();
        invariant!(remainder.is_empty());
        invariant_eq!(pixels.len(), width);

        let (chunks, tail) = pixels.as_chunks::<SRGB_LANES>();
        for chunk in chunks {
            let ([red, green, blue], alpha) = decode_rgba128_float_simd(chunk);
            let scale = Simd::splat(color_scale);
            let red = (red * scale).to_array();
            let green = (green * scale).to_array();
            let blue = (blue * scale).to_array();
            let alpha = alpha.to_array();

            for (((red, green), blue), alpha) in red.into_iter().zip(green).zip(blue).zip(alpha) {
                has_nonzero_alpha |= alpha > 0.0;
                colors.push(LinearRGB::new([red, green, blue]));
                alphas.push(normalized_to_u8(alpha));
            }

            if colors.len() >= HDR_BATCH_PIXELS {
                write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
            }
        }

        for pixel in tail {
            let (color, alpha) = decode_rgba128_float(pixel);
            has_nonzero_alpha |= alpha > 0.0;
            let color = color.map(|component| component * color_scale);
            colors.push(LinearRGB::new(color));
            alphas.push(normalized_to_u8(alpha));

            if colors.len() == HDR_BATCH_PIXELS {
                write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
            }
        }
    }

    write_tone_mapped_batch(mapper, &mut colors, &mut alphas, rgba, &mut rgba_offset);
    invariant_eq!(rgba_offset, rgba.len());
    has_nonzero_alpha
}

fn write_tone_mapped_batch(
    mapper: &(impl ToneMapper + Sync),
    colors: &mut LinearRGBPlanes,
    alphas: &mut Vec<u8>,
    rgba: &mut [u8],
    rgba_offset: &mut usize,
) {
    invariant_eq!(colors.len(), alphas.len());

    if colors.is_empty() {
        return;
    }

    mapper.map_planes_in_place(colors);

    let byte_count = colors.len() * 4;
    let target = &mut rgba[*rgba_offset..*rgba_offset + byte_count];
    let (targets, remainder) = target.as_chunks_mut::<4>();

    invariant!(remainder.is_empty());
    write_display_pixels(colors, alphas, targets);

    *rgba_offset += byte_count;
    colors.clear();
    alphas.clear();
}

#[multiversion(targets = "simd")]
pub(super) fn write_display_pixels(
    colors: &LinearRGBPlanes,
    alphas: &[u8],
    targets: &mut [[u8; 4]],
) {
    invariant_eq!(colors.len(), alphas.len());
    invariant_eq!(colors.len(), targets.len());

    let [red, green, blue] = colors.channels();
    let (red_chunks, red_tail) = red.as_chunks::<SRGB_LANES>();
    let (green_chunks, green_tail) = green.as_chunks::<SRGB_LANES>();
    let (blue_chunks, blue_tail) = blue.as_chunks::<SRGB_LANES>();
    let (alpha_chunks, alpha_tail) = alphas.as_chunks::<SRGB_LANES>();
    let (target_chunks, target_tail) = targets.as_chunks_mut::<SRGB_LANES>();

    for ((((red, green), blue), alphas), targets) in red_chunks
        .iter()
        .zip(green_chunks)
        .zip(blue_chunks)
        .zip(alpha_chunks)
        .zip(target_chunks)
    {
        // Match `normalized_to_u8`: round half away from zero, then saturate the cast.
        let encoded = [*red, *green, *blue].map(|channel| {
            let srgb = linear_to_srgb_simd(F32x8::from_array(channel));
            (srgb.simd_clamp(F32x8::splat(0.0), F32x8::splat(1.0))
                * F32x8::splat(f32::from(u8::MAX)))
            .round()
            .cast::<u8>()
            .to_array()
        });

        for lane in 0..SRGB_LANES {
            targets[lane] = [
                encoded[0][lane],
                encoded[1][lane],
                encoded[2][lane],
                alphas[lane],
            ];
        }
    }

    for ((((red, green), blue), alpha), target) in red_tail
        .iter()
        .zip(green_tail)
        .zip(blue_tail)
        .zip(alpha_tail)
        .zip(target_tail)
    {
        let color = LinearRGB::new([*red, *green, *blue]);
        let [red, green, blue] = display_linear_to_srgb8(color);
        *target = [red, green, blue, *alpha];
    }
}

#[cfg(test)]
pub(super) fn append_hdr_pixels(
    source: &[u8],
    width: usize,
    row_stride: usize,
    layout: PixelLayout,
    mapper: &(impl ToneMapper + Sync),
    rgba: &mut Vec<u8>,
) -> Result<bool> {
    let byte_count = source.len() / row_stride * width * 4;
    let start = rgba.len();
    rgba.resize(start + byte_count, 0);
    write_hdr_pixels(
        source,
        width,
        row_stride,
        layout,
        mapper,
        1.0,
        &mut rgba[start..],
    )
}

pub(super) fn pixel_at(row: &[u8], x: usize, layout: PixelLayout) -> Result<&[u8]> {
    let start = x
        .checked_mul(layout.bytes_per_pixel)
        .ok_or_else(|| error(JPEGXRError::Output("JPEG XR pixel offset exceeds usize")))?;

    let end = start + layout.bytes_per_pixel;

    row.get(start..end)
        .ok_or_else(|| error(JPEGXRError::Output("JPEG XR pixel exceeds its decoded row")))
}

pub(super) fn display_linear_to_srgb8(color: LinearRGB) -> [u8; 3] {
    color.components().map(linear_to_srgb).map(normalized_to_u8)
}

#[cfg(test)]
pub(super) fn hdr_to_srgb8(color: [f32; 3], mapper: &impl ToneMapper) -> [u8; 3] {
    display_linear_to_srgb8(mapper.map(LinearRGB::new(color)))
}

fn linear_to_srgb(value: f32) -> f32 {
    linear_to_srgb_simd(Simd::<f32, 1>::splat(value))[0]
}

#[inline]
fn linear_to_srgb_simd<const N: usize>(value: Simd<f32, N>) -> Simd<f32, N> {
    let value = value.simd_clamp(Simd::splat(0.0), Simd::splat(1.0));
    let linear = value * Simd::splat(12.92);
    let nonlinear =
        exp2(log2(value) * Simd::splat(1.0 / 2.4)) * Simd::splat(1.055) - Simd::splat(0.055);
    value
        .simd_le(Simd::splat(0.003_130_8))
        .select(linear, nonlinear)
}

pub(super) fn normalized_to_u8(value: f32) -> u8 {
    round_clamp_u8(value.clamp(0.0, 1.0) * f32::from(u8::MAX))
}
