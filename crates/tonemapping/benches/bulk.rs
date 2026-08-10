use std::num::{NonZeroU32, NonZeroUsize};
use std::time::{Duration, Instant};

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

fn main() {
    // Cargo passes this flag for `cargo bench`, but not when `cargo test --all-targets` executes the
    // harness-free target.
    if !std::env::args_os().any(|argument| argument == "--bench") {
        return;
    }

    let pixel_count = env_nonzero_usize("FOUR_TONEMAPPING_BENCH_PIXELS", 1_048_576);
    let iterations = env_nonzero_u32("FOUR_TONEMAPPING_BENCH_ITERATIONS", 20);
    let colors = benchmark_colors(pixel_count.get());

    let white = WhitePoint::new(16.0).expect("benchmark white is positive");
    let luminance_white =
        LuminanceWhitePoint::new(16.0).expect("benchmark luminance white is positive");

    benchmark_mapper("clamp", &Clamp, &colors, iterations);
    benchmark_mapper(
        "scaled clamp",
        &ScaledClamp::new(white),
        &colors,
        iterations,
    );
    benchmark_mapper("Reinhard", &Reinhard, &colors, iterations);
    benchmark_mapper(
        "extended Reinhard",
        &ExtendedReinhard::new(white),
        &colors,
        iterations,
    );
    benchmark_mapper(
        "luminance Reinhard",
        &LuminanceReinhard,
        &colors,
        iterations,
    );
    benchmark_mapper(
        "extended luminance Reinhard",
        &ExtendedLuminanceReinhard::new(luminance_white),
        &colors,
        iterations,
    );
    benchmark_mapper("Reinhard-Jodie", &ReinhardJodie, &colors, iterations);
    benchmark_mapper("Hable", &Hable, &colors, iterations);
    benchmark_mapper("fitted ACES", &ACESFitted, &colors, iterations);
    benchmark_mapper("approximate ACES", &ACESApproximate, &colors, iterations);
    benchmark_mapper("BT.2446A", &BT2446A, &colors, iterations);
    benchmark_max_cll(&colors, iterations);
}

fn benchmark_colors(pixel_count: usize) -> Vec<LinearRGB> {
    let palette = [
        [0.0, 0.18, 1.0],
        [4.0, 2.0, 1.0],
        [16.0, 8.0, 0.5],
        [0.25, 0.5, 2.0],
        [1.0, 3.0, 0.75],
        [32.0, 0.01, 4.0],
        [0.02, 0.03, 0.04],
        [100_000.0, 1.0, 0.0],
    ];
    (0..pixel_count)
        .map(|index| LinearRGB::new(palette[index % palette.len()]))
        .collect()
}

fn benchmark_mapper(
    name: &str,
    mapper: &impl ToneMapper,
    colors: &[LinearRGB],
    iterations: NonZeroU32,
) {
    let mut scalar_scratch = colors.to_vec();
    let bulk_source: Vec<LinearRGBPlanes> = colors
        .chunks(BATCH_PIXELS)
        .map(|batch| batch.iter().copied().collect())
        .collect();
    let mut bulk_scratch = bulk_source.clone();
    let (scalar, bulk) = benchmark_pair(
        iterations,
        || {
            scalar_scratch.copy_from_slice(colors);
            time_once(|| {
                for color in &mut scalar_scratch {
                    *color = mapper.map(*color);
                }
                std::hint::black_box(&scalar_scratch);
            })
        },
        || {
            bulk_scratch.clone_from(&bulk_source);
            time_once(|| {
                for batch in &mut bulk_scratch {
                    mapper.map_planes_in_place(batch);
                }
                std::hint::black_box(&bulk_scratch);
            })
        },
    );

    print_comparison(name, scalar, bulk, iterations);
}

fn benchmark_max_cll(colors: &[LinearRGB], iterations: NonZeroU32) {
    let pixel_count = NonZeroUsize::new(colors.len()).expect("benchmark colors are nonempty");
    let (scalar, bulk) = benchmark_pair(
        iterations,
        || {
            time_once(|| {
                let mut estimator = MaxCLLEstimator::new(pixel_count);
                for color in colors {
                    estimator.observe(*color);
                }
                std::hint::black_box(
                    estimator
                        .finish()
                        .expect("the scalar benchmark observes every color"),
                );
            })
        },
        || {
            time_once(|| {
                let mut estimator = MaxCLLEstimator::new(pixel_count);
                for batch in colors.chunks(BATCH_PIXELS) {
                    estimator.observe_many(batch);
                }
                std::hint::black_box(
                    estimator
                        .finish()
                        .expect("the bulk benchmark observes every color"),
                );
            })
        },
    );

    print_comparison("MaxCLL", scalar, bulk, iterations);
}

fn benchmark_pair(
    iterations: NonZeroU32,
    mut scalar_sample: impl FnMut() -> Duration,
    mut bulk_sample: impl FnMut() -> Duration,
) -> (Duration, Duration) {
    let _scalar_warmup = scalar_sample();
    let _bulk_warmup = bulk_sample();

    let mut scalar_elapsed = Duration::ZERO;
    let mut bulk_elapsed = Duration::ZERO;
    for iteration in 0..iterations.get() {
        if iteration % 2 == 0 {
            scalar_elapsed += scalar_sample();
            bulk_elapsed += bulk_sample();
        } else {
            bulk_elapsed += bulk_sample();
            scalar_elapsed += scalar_sample();
        }
    }
    (scalar_elapsed, bulk_elapsed)
}

fn time_once(mut operation: impl FnMut()) -> Duration {
    let started = Instant::now();
    operation();
    started.elapsed()
}

fn print_comparison(name: &str, scalar: Duration, bulk: Duration, iterations: NonZeroU32) {
    let scalar_average = scalar / iterations.get();
    let bulk_average = bulk / iterations.get();
    let speedup = scalar.as_secs_f64() / bulk.as_secs_f64();
    println!("{name}: scalar {scalar_average:.3?}, bulk {bulk_average:.3?}, {speedup:.2}x");
}

fn env_nonzero_usize(name: &str, default: usize) -> NonZeroUsize {
    std::env::var(name).map_or_else(
        |_missing| NonZeroUsize::new(default).expect("the benchmark default is positive"),
        |value| value.parse().expect("the benchmark size must be positive"),
    )
}

fn env_nonzero_u32(name: &str, default: u32) -> NonZeroU32 {
    std::env::var(name).map_or_else(
        |_missing| NonZeroU32::new(default).expect("the benchmark default is positive"),
        |value| {
            value
                .parse()
                .expect("the benchmark iteration count must be positive")
        },
    )
}
