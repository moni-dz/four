//! Compares the scalar and bulk tone-mapping paths at the batch size production uses.

use divan::{Bencher, counter::ItemsCount};
use mimalloc::MiMalloc;
use tonemapping::{
    ACESApproximate, ACESFitted, BT2446A, Clamp, ExtendedLuminanceReinhard, ExtendedReinhard,
    Hable, LinearRGB, LinearRGBPlanes, LuminanceReinhard, LuminanceWhitePoint, MaxCLLEstimator,
    Reinhard, ReinhardJodie, ScaledClamp, ToneMapper, WhitePoint,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Mirrors the JPEG XR tile size so the measurement includes its production dispatch frequency.
const BATCH_PIXELS: usize = 1_024;

// One megapixel is large enough to leave cache and small enough to keep a full sweep interactive.
const PIXEL_COUNT: usize = 1_048_576;

fn main() {
    divan::main();
}

/// Generates deterministic HDR values without a SIMD-lane period.
fn benchmark_colors(pixel_count: usize) -> Vec<LinearRGB> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next_unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // The top bits are the well-mixed ones. Sixteen of them is ample resolution for a
        // benchmark input and converts to f32 exactly, with no lossy cast.
        f32::from((state >> 48) as u16) / 65_536.0
    };

    (0..pixel_count)
        .map(|index| {
            // Log-uniform over 1e-3..1e4 covers deep shadow through specular highlight. Every
            // 997th pixel is pinned to black so the zero path stays exercised.
            let components = [(); 3].map(|()| {
                if index % 997 == 0 {
                    0.0
                } else {
                    10.0_f32.powf(next_unit() * 7.0 - 3.0)
                }
            });
            LinearRGB::new(components)
        })
        .collect()
}

fn bulk_batches(colors: &[LinearRGB]) -> Vec<LinearRGBPlanes> {
    colors
        .chunks(BATCH_PIXELS)
        .map(|batch| batch.iter().copied().collect())
        .collect()
}

fn bench_scalar(bencher: Bencher<'_, '_>, mapper: &impl ToneMapper) {
    let colors = benchmark_colors(PIXEL_COUNT);
    bencher
        .counter(ItemsCount::new(PIXEL_COUNT))
        .with_inputs(|| colors.clone())
        .bench_local_refs(|scratch| {
            for color in scratch.iter_mut() {
                *color = mapper.map(*color);
            }
        });
}

fn bench_bulk(bencher: Bencher<'_, '_>, mapper: &impl ToneMapper) {
    let batches = bulk_batches(&benchmark_colors(PIXEL_COUNT));
    bencher
        .counter(ItemsCount::new(PIXEL_COUNT))
        .with_inputs(|| batches.clone())
        .bench_local_refs(|scratch| {
            for batch in scratch.iter_mut() {
                mapper.map_planes_in_place(batch);
            }
        });
}

fn white() -> WhitePoint {
    WhitePoint::new(16.0).expect("benchmark white is positive")
}

fn luminance_white() -> LuminanceWhitePoint {
    LuminanceWhitePoint::new(16.0).expect("benchmark luminance white is positive")
}

#[divan::bench_group(name = "scalar")]
mod scalar {
    use super::{
        ACESApproximate, ACESFitted, BT2446A, Bencher, Clamp, ExtendedLuminanceReinhard,
        ExtendedReinhard, Hable, LuminanceReinhard, Reinhard, ReinhardJodie, ScaledClamp,
        bench_scalar, luminance_white, white,
    };

    #[divan::bench]
    fn clamp(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &Clamp);
    }

    #[divan::bench]
    fn scaled_clamp(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &ScaledClamp::new(white()));
    }

    #[divan::bench]
    fn reinhard(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &Reinhard);
    }

    #[divan::bench]
    fn extended_reinhard(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &ExtendedReinhard::new(white()));
    }

    #[divan::bench]
    fn luminance_reinhard(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &LuminanceReinhard);
    }

    #[divan::bench]
    fn extended_luminance_reinhard(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &ExtendedLuminanceReinhard::new(luminance_white()));
    }

    #[divan::bench]
    fn reinhard_jodie(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &ReinhardJodie);
    }

    #[divan::bench]
    fn hable(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &Hable);
    }

    #[divan::bench]
    fn aces_fitted(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &ACESFitted);
    }

    #[divan::bench]
    fn aces_approximate(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &ACESApproximate);
    }

    #[divan::bench]
    fn bt2446a(bencher: Bencher<'_, '_>) {
        bench_scalar(bencher, &BT2446A);
    }
}

#[divan::bench_group(name = "bulk")]
mod bulk {
    use super::{
        ACESApproximate, ACESFitted, BT2446A, Bencher, Clamp, ExtendedLuminanceReinhard,
        ExtendedReinhard, Hable, LuminanceReinhard, Reinhard, ReinhardJodie, ScaledClamp,
        bench_bulk, luminance_white, white,
    };

    #[divan::bench]
    fn clamp(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &Clamp);
    }

    #[divan::bench]
    fn scaled_clamp(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &ScaledClamp::new(white()));
    }

    #[divan::bench]
    fn reinhard(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &Reinhard);
    }

    #[divan::bench]
    fn extended_reinhard(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &ExtendedReinhard::new(white()));
    }

    #[divan::bench]
    fn luminance_reinhard(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &LuminanceReinhard);
    }

    #[divan::bench]
    fn extended_luminance_reinhard(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &ExtendedLuminanceReinhard::new(luminance_white()));
    }

    #[divan::bench]
    fn reinhard_jodie(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &ReinhardJodie);
    }

    #[divan::bench]
    fn hable(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &Hable);
    }

    #[divan::bench]
    fn aces_fitted(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &ACESFitted);
    }

    #[divan::bench]
    fn aces_approximate(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &ACESApproximate);
    }

    #[divan::bench]
    fn bt2446a(bencher: Bencher<'_, '_>) {
        bench_bulk(bencher, &BT2446A);
    }
}

#[divan::bench_group(name = "max_cll")]
mod max_cll {
    use std::num::NonZeroUsize;

    use super::{
        BATCH_PIXELS, Bencher, ItemsCount, MaxCLLEstimator, PIXEL_COUNT, benchmark_colors,
    };

    fn pixel_count() -> NonZeroUsize {
        NonZeroUsize::new(PIXEL_COUNT).expect("the benchmark pixel count is positive")
    }

    #[divan::bench]
    fn scalar(bencher: Bencher<'_, '_>) {
        let colors = benchmark_colors(PIXEL_COUNT);
        bencher
            .counter(ItemsCount::new(PIXEL_COUNT))
            .bench_local(|| {
                let mut estimator = MaxCLLEstimator::new(pixel_count());
                for color in &colors {
                    estimator.observe(*color);
                }
                estimator.finish()
            });
    }

    #[divan::bench]
    fn bulk(bencher: Bencher<'_, '_>) {
        let colors = benchmark_colors(PIXEL_COUNT);
        bencher
            .counter(ItemsCount::new(PIXEL_COUNT))
            .bench_local(|| {
                let mut estimator = MaxCLLEstimator::new(pixel_count());
                for batch in colors.chunks(BATCH_PIXELS) {
                    estimator.observe_many(batch);
                }
                estimator.finish()
            });
    }
}
