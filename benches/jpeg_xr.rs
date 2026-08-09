use std::num::NonZeroU32;
use std::time::Instant;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // Cargo passes this flag for `cargo bench`, but not when `cargo test --all-targets` executes the
    // harness-free target.
    if !std::env::args_os().any(|argument| argument == "--bench") {
        return;
    }

    let path = std::env::var_os("FOUR_JPEG_XR_BENCH_INPUT")
        .expect("FOUR_JPEG_XR_BENCH_INPUT must name a representative JPEG XR file");
    let input = std::fs::read(path).expect("the JPEG XR benchmark input must be readable");
    let iterations = benchmark_iterations();

    let warmup = four::jpeg_xr::decode_with_metadata(std::hint::black_box(&input))
        .expect("the JPEG XR benchmark input must decode successfully");
    let max_cll = warmup.metadata().max_cll_scrgb();
    drop(warmup);

    let started = Instant::now();
    for _ in 0..iterations.get() {
        let image = four::jpeg_xr::decode(std::hint::black_box(&input))
            .expect("the JPEG XR benchmark input must decode successfully");
        std::hint::black_box(image);
    }
    let elapsed = started.elapsed();
    let average = elapsed / iterations.get();

    println!(
        "decoded {}-byte JPEG XR input with MaxCLL {max_cll:?} {} times in {:.3?}: {:.3?}/iteration",
        input.len(),
        iterations,
        elapsed,
        average
    );
}

fn benchmark_iterations() -> NonZeroU32 {
    match std::env::var("FOUR_JPEG_XR_BENCH_ITERATIONS") {
        Ok(iterations) => iterations
            .parse()
            .expect("FOUR_JPEG_XR_BENCH_ITERATIONS must be a positive u32"),
        Err(std::env::VarError::NotPresent) => {
            NonZeroU32::new(3).expect("the default benchmark iteration count is positive")
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("FOUR_JPEG_XR_BENCH_ITERATIONS must contain Unicode digits")
        }
    }
}
