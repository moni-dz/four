use std::simd::Simd;

use super::{LinearRGBPlanes, ToneMapper};
use crate::math::{max_or_second, min_or_second};

pub(crate) const COLOR_LANES: usize = 8;
pub(crate) type F32x8 = Simd<f32, COLOR_LANES>;

/// Applies `map` to every complete lane group of `colors`, in place.
#[inline]
pub(crate) fn map_colors(
    colors: &mut LinearRGBPlanes,
    mut map: impl FnMut([F32x8; 3]) -> [F32x8; 3],
) {
    let [red, green, blue] = colors.channels_mut();

    let [red_chunks, green_chunks, blue_chunks] = [red, green, blue].map(|channel| {
        let (chunks, _) = channel.as_chunks_mut::<COLOR_LANES>();
        chunks
    });

    for ((red, green), blue) in red_chunks.iter_mut().zip(green_chunks).zip(blue_chunks) {
        let components = [*red, *green, *blue].map(F32x8::from_array);

        let mapped = map(components).map(displayable);

        *red = mapped[0];
        *green = mapped[1];
        *blue = mapped[2];
    }
}

/// Runs `batch` over the complete lane groups of `colors`, then `mapper` over the remainder.
///
/// Computes the complete lane groups and scalar tail for a batch operation.
#[inline]
pub(crate) fn map_planes(
    colors: &mut LinearRGBPlanes,
    lanes: usize,
    batch: impl FnOnce(&mut LinearRGBPlanes),
    mapper: &(impl ToneMapper + ?Sized),
) {
    let simd_len = colors.len() / lanes * lanes;
    batch(colors);
    colors.map_from(simd_len, mapper);
}

/// Clips `component` to `0.0..=1.0`, mapping `NaN` and negative zero to positive zero.
///
/// Matches [`super::display_component_f32`] with one `MAXPS` and one `MINPS` per vector.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "must compile under the calling multiversioned function's target features"
)]
pub(crate) fn displayable<const N: usize>(component: Simd<f32, N>) -> [f32; N] {
    min_or_second(max_or_second(component, Simd::splat(0.0)), Simd::splat(1.0)).to_array()
}
