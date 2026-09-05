//! Measures cached, end-to-end decoding through each format's public API.

use divan::{Bencher, counter::BytesCount};
use four::{DecodedImage, encode_bmp, gif, jpeg, jpeg_xl, jpeg_xr, png, tiff};
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

/// Times `decode`, counting decoded RGBA bytes produced.
fn bench_decode<E>(
    bencher: Bencher<'_, '_>,
    name: &str,
    decode: fn(&[u8]) -> Result<DecodedImage, E>,
) {
    let bytes = fixture(name);
    let output_bytes = decode(&bytes)
        .ok()
        .map(|image| image.rgba8().len())
        .expect("benchmark fixture decodes successfully");
    bencher
        .counter(BytesCount::new(output_bytes))
        .bench_local(|| decode(&bytes).map(|image| image.rgba8().len()));
}

#[divan::bench_group(name = "decode")]
mod decode {
    use super::{
        Bencher, BytesCount, bench_decode, fixture, gif, jpeg, jpeg_xl, jpeg_xr, png, tiff,
    };

    #[divan::bench]
    fn jpeg_baseline(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "baseline.jpg", jpeg::decode);
    }

    /// Exercises high-frequency JPEG blocks.
    #[divan::bench]
    fn jpeg_busy(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "busy.jpg", jpeg::decode);
    }

    #[divan::bench]
    fn png_rgb8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb8.png", png::decode);
    }

    #[divan::bench]
    fn png_rgb16(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb16.png", png::decode);
    }

    #[divan::bench]
    fn png_rgba8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgba8.png", png::decode);
    }

    #[divan::bench]
    fn png_palette8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "palette8.png", png::decode);
    }

    #[divan::bench]
    fn tiff_rgb8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb8.tiff", tiff::decode);
    }

    /// Exercises parallel TIFF normalization.
    #[divan::bench]
    fn tiff_large(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "large.tiff", tiff::decode);
    }

    #[divan::bench]
    fn jpeg_xl_rgb8(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "rgb8.jxl", jpeg_xl::decode);
    }

    /// Decodes a 3840x2160 Windows HDR screenshot without metadata analysis.
    #[divan::bench]
    fn jpeg_xr_bgr101010(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "screenshot.jxr", jpeg_xr::decode);
    }

    /// Decodes and analyzes the production JPEG XR path.
    #[divan::bench]
    fn jpeg_xr_bgr101010_metadata(bencher: Bencher<'_, '_>) {
        let bytes = fixture("screenshot.jxr");
        let output_bytes = jpeg_xr::decode_with_metadata(&bytes)
            .expect("benchmark fixture decodes successfully")
            .image()
            .rgba8()
            .len();

        bencher
            .counter(BytesCount::new(output_bytes))
            .bench_local(|| {
                jpeg_xr::decode_with_metadata(&bytes).map(|decoded| decoded.image().rgba8().len())
            });
    }

    #[divan::bench]
    fn gif_animated(bencher: Bencher<'_, '_>) {
        bench_decode(bencher, "animated.gif", gif::decode);
    }
}

/// Measures the BMP carrier used by GPUI.
#[divan::bench_group(name = "display")]
mod display {
    use super::{Bencher, BytesCount, encode_bmp, fixture, png};

    #[divan::bench]
    fn bmp_carrier(bencher: Bencher<'_, '_>) {
        let image = png::decode(&fixture("rgba8.png")).expect("fixture decodes as PNG");
        bencher
            .counter(BytesCount::new(image.rgba8().len()))
            .bench_local(|| encode_bmp(&image).len());
    }
}
