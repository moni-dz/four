//! Measures the end-to-end decode path for every fixture format.
//!
//! These go through the public `decode` entry points rather than reaching into private kernels, so
//! a win in the IDCT, in a row writer, or in a color conversion shows up here without the benchmark
//! needing to know those functions exist. Fixtures are tiny by design, which keeps the whole file
//! in cache: this measures per-pixel work, not memory bandwidth.

use divan::{Bencher, counter::BytesCount};
use four::{DecodedImage, encode_bmp, gif, jpeg, jpeg_xl, png, tiff};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("read fixture {}: {error}", path.display()))
}

/// Times `decode` over `name`, counting the compressed bytes consumed.
fn bench_decode<E>(
    bencher: Bencher<'_, '_>,
    name: &str,
    decode: fn(&[u8]) -> Result<DecodedImage, E>,
) {
    let bytes = fixture(name);
    bencher
        .counter(BytesCount::new(bytes.len()))
        .bench_local(|| decode(&bytes).map(|image| image.rgba8().len()));
}

#[divan::bench_group(name = "decode")]
mod decode {
    use super::{Bencher, bench_decode, gif, jpeg, jpeg_xl, png, tiff};

    #[divan::bench]
    fn jpeg_baseline(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "baseline.jpg", |bytes| jpeg::decode(bytes));
    }

    /// A 384x384 image with high-frequency content in every block.
    ///
    /// `baseline.jpg` is too small and too flat to show anything: at that size parsing dominates,
    /// and most of its blocks take the DC-only path that skips the inverse transform entirely.
    #[divan::bench]
    fn jpeg_busy(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "busy.jpg", |bytes| jpeg::decode(bytes));
    }

    #[divan::bench]
    fn png_rgb8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb8.png", |bytes| png::decode(bytes));
    }

    #[divan::bench]
    fn png_rgb16(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb16.png", |bytes| png::decode(bytes));
    }

    #[divan::bench]
    fn png_rgba8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgba8.png", |bytes| png::decode(bytes));
    }

    #[divan::bench]
    fn png_palette8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "palette8.png", |bytes| png::decode(bytes));
    }

    #[divan::bench]
    fn tiff_rgb8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb8.tiff", |bytes| tiff::decode(bytes));
    }

    /// A 640x480 image, above the threshold where TIFF normalization goes parallel.
    ///
    /// `rgb8.tiff` has 3,072 pixels and so only ever exercises the sequential branch.
    #[divan::bench]
    fn tiff_large(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "large.tiff", |bytes| tiff::decode(bytes));
    }

    #[divan::bench]
    fn jpeg_xl_rgb8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb8.jxl", |bytes| jpeg_xl::decode(bytes));
    }

    #[divan::bench]
    fn gif_animated(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "animated.gif", |bytes| gif::decode(bytes));
    }
}

/// Every decoded image is re-encoded as a BMP before GPUI will accept it, so the swizzle and the
/// extra full-buffer copy are part of the cost of displaying any non-GIF image.
#[divan::bench_group(name = "display")]
mod display {
    use super::{Bencher, BytesCount, encode_bmp, fixture, png};

    #[divan::bench]
    fn bmp_carrier(bencher: Bencher<'_, '_>) {
        let image = png::decode(fixture("rgba8.png")).expect("fixture decodes as PNG");
        bencher
            .counter(BytesCount::new(image.rgba8().len()))
            .bench_local(|| encode_bmp(&image).len());
    }
}
