use std::io::Cursor;

use four::tiff as tiff_decoder;
use tiff::encoder::TiffEncoder;
use tiff::encoder::colortype::{CMYK8, Gray16, RGB8};

#[test]
fn normalizes_rgb8_to_rgba8() {
    let encoded = encode::<RGB8>(2, 1, &[255, 0, 0, 0, 128, 255]);
    let image = tiff_decoder::decode(encoded).unwrap();

    assert_eq!(image.dimensions(), (2, 1));
    assert_eq!(image.rgba8(), &[255, 0, 0, 255, 0, 128, 255, 255]);
}

#[test]
fn scales_sixteen_bit_grayscale_samples() {
    let encoded = encode::<Gray16>(3, 1, &[0, 32_768, u16::MAX]);
    let image = tiff_decoder::decode(encoded).unwrap();

    assert_eq!(
        image.rgba8(),
        &[0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]
    );
}

#[test]
fn converts_cmyk8_to_rgb() {
    let encoded = encode::<CMYK8>(2, 1, &[255, 0, 0, 0, 0, 255, 255, 0]);
    let image = tiff_decoder::decode(encoded).unwrap();

    assert_eq!(image.rgba8(), &[0, 255, 255, 255, 255, 0, 0, 255]);
}

fn encode<C: tiff::encoder::colortype::ColorType>(
    width: u32,
    height: u32,
    samples: &[C::Inner],
) -> Vec<u8>
where
    [C::Inner]: tiff::encoder::TiffValue,
{
    let mut output = Cursor::new(Vec::new());
    TiffEncoder::new(&mut output)
        .unwrap()
        .write_image::<C>(width, height, samples)
        .unwrap();
    output.into_inner()
}
