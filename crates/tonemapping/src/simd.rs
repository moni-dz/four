use std::simd::{Select, Simd, cmp::SimdPartialOrd, num::SimdFloat};

use super::{LinearRGBPlanes, ToneMapper};

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
/// Every operator needs this same prologue, and computing the boundary after `batch` has already
/// run would silently re-map the tail. Keeping it in one place makes that mistake unavailable.
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

#[inline]
pub(crate) fn displayable<const N: usize>(component: Simd<f32, N>) -> [f32; N] {
    let zero = Simd::splat(0.0);
    let one = Simd::splat(1.0);
    let below = component.is_nan() | component.simd_le(zero);

    below
        .select(zero, component.simd_ge(one).select(one, component))
        .to_array()
}
