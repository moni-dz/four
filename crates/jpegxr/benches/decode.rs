use std::time::{Duration, Instant};

use jpegxr::{Decoder, PixelFormat};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // Cargo passes this flag for `cargo bench`, but not when `cargo test --all-targets` executes the
    // harness-free target.
    if !std::env::args_os().any(|argument| argument == "--bench") {
        return;
    }

    let Ok(path) = std::env::var("JPEGXR_BENCH_SAMPLE") else {
        println!("decode: skipped, set JPEGXR_BENCH_SAMPLE to a local .jxr file");
        return;
    };
    let iterations: u32 = std::env::var("JPEGXR_BENCH_ITERATIONS")
        .map_or(10, |value| value.parse().expect("iteration count"));
    let bytes = std::fs::read(&path).expect("read sample");

    let decoder = Decoder::new(&bytes).expect("parse sample headers");
    let info = decoder.info();
    let (width, height) = (info.width(), info.height());
    let bgr101010 = info.pixel_format() == PixelFormat::BGR101010;

    let mut checksum = 0_u64;
    let mut decode = || {
        let decoder = Decoder::new(&bytes).expect("parse sample headers");
        let started = Instant::now();
        checksum = if bgr101010 {
            let image = decoder.decode_bgr101010().expect("decode sample");
            image
                .pixels()
                .iter()
                .fold(0_u64, |sum, pixel| sum.wrapping_add(u64::from(*pixel)))
        } else {
            let image = decoder.decode_rgba_f32().expect("decode sample");
            image.pixels().iter().fold(0_u64, |sum, pixel| {
                sum.wrapping_add(u64::from(pixel.to_bits()))
            })
        };
        started.elapsed()
    };

    let _warmup = decode();
    let mut elapsed = Duration::ZERO;
    for _iteration in 0..iterations {
        elapsed += decode();
    }

    let average = elapsed / iterations;
    let megapixels = f64::from(width) * f64::from(height) / 1e6;
    let throughput = megapixels / average.as_secs_f64();
    println!(
        "decode {width}x{height}: {average:.3?}, {throughput:.1} MP/s, checksum {checksum:016x}"
    );
}
