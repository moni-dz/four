use std::simd::{Select, Simd, cmp::SimdPartialOrd, num::SimdFloat};

use super::LinearRGBPlanes;

pub(crate) const COLOR_LANES: usize = 4;
pub(crate) type F64x4 = Simd<f64, COLOR_LANES>;

#[inline]
pub(crate) fn map_colors(
    colors: &mut LinearRGBPlanes,
    mut map: impl FnMut([F64x4; 3]) -> [F64x4; 3],
) {
    let [red, green, blue] = colors.channels_mut();

    let [red_chunks, green_chunks, blue_chunks] = [red, green, blue].map(|channel| {
        let (chunks, _) = channel.as_chunks_mut::<COLOR_LANES>();
        chunks
    });

    for ((red, green), blue) in red_chunks.iter_mut().zip(green_chunks).zip(blue_chunks) {
        let components =
            [*red, *green, *blue].map(|channel| F64x4::from_array(channel.map(f64::from)));

        let mapped = map(components).map(displayable);

        *red = mapped[0];
        *green = mapped[1];
        *blue = mapped[2];
    }
}

#[inline]
fn displayable<const N: usize>(component: Simd<f64, N>) -> [f32; N] {
    let zero = Simd::splat(0.0);
    let one = Simd::splat(1.0);
    let below = component.is_nan() | component.simd_le(zero);

    below
        .select(zero, component.simd_ge(one).select(one, component))
        .cast::<f32>()
        .to_array()
}
