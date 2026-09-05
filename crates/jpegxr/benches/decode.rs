use divan::{Bencher, counter::ItemsCount};
use jpegxr::{Decoder, PixelFormat};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    if !std::env::args_os().any(|argument| argument == "--bench") {
        return;
    }

    if std::env::var_os("JPEGXR_BENCH_SAMPLE").is_none() {
        println!("decode: skipped, set JPEGXR_BENCH_SAMPLE to a local .jxr file");
        return;
    }

    divan::main();
}

#[divan::bench]
fn decode(bencher: Bencher<'_, '_>) {
    let path = std::env::var("JPEGXR_BENCH_SAMPLE").expect("sample path checked by main");
    let bytes = std::fs::read(path).expect("read sample");

    let decoder = Decoder::new(&bytes).expect("parse sample headers");
    let info = decoder.info();
    let pixel_count = usize::try_from(info.width()).expect("width fits usize")
        * usize::try_from(info.height()).expect("height fits usize");
    let bencher = bencher.counter(ItemsCount::new(pixel_count));

    if info.pixel_format() == PixelFormat::BGR101010 {
        bencher.bench_local(|| {
            Decoder::new(&bytes)
                .and_then(|decoder| decoder.decode_bgr101010())
                .expect("decode sample")
        });
    } else {
        bencher.bench_local(|| {
            Decoder::new(&bytes)
                .and_then(|decoder| decoder.decode_rgba_f32())
                .expect("decode sample")
        });
    }
}
