use divan::{Bencher, counter::ItemsCount};
use tonemapping::{
    LinearRGB, LinearRGBPlanes, LuminanceWhitePoint, ToneMapper, ToneMappingMethod, WhitePoint,
};

const BATCH_PIXELS: usize = 1_024;
const BATCHES: usize = 256;

fn main() {
    divan::main();
}

fn colors() -> Vec<LinearRGB> {
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        let unit = f32::from(u16::try_from(state >> 18).expect("14 bits fit u16")) / 16_384.0;
        (unit * 14.0 - 8.0).exp2()
    };

    (0..BATCH_PIXELS * BATCHES)
        .map(|_| LinearRGB::new([next(), next(), next()]))
        .collect()
}

fn mapper(method: ToneMappingMethod) -> impl ToneMapper {
    method.resolve(
        WhitePoint::new(48.0).expect("positive"),
        LuminanceWhitePoint::new(32.0).expect("positive"),
    )
}

#[divan::bench(args = ToneMappingMethod::ALL)]
fn planes(bencher: Bencher<'_, '_>, method: ToneMappingMethod) {
    let mapper = mapper(method);

    let source: Vec<LinearRGBPlanes> = colors()
        .chunks(BATCH_PIXELS)
        .map(|batch| batch.iter().copied().collect())
        .collect();

    bencher
        .counter(ItemsCount::new(BATCH_PIXELS * BATCHES))
        .with_inputs(|| source.clone())
        .bench_local_refs(|batches| {
            for batch in batches {
                mapper.map_planes_in_place(batch);
            }
        });
}

#[divan::bench(args = [ToneMappingMethod::BT2446, ToneMappingMethod::ACESFitted])]
fn interleaved(bencher: Bencher<'_, '_>, method: ToneMappingMethod) {
    let mapper = mapper(method);
    let source = colors();
    bencher
        .counter(ItemsCount::new(BATCH_PIXELS * BATCHES))
        .with_inputs(|| source.clone())
        .bench_local_refs(|colors| {
            for batch in colors.chunks_mut(BATCH_PIXELS) {
                mapper.map_in_place(batch);
            }
        });
}
