use std::borrow::Cow;

use four::gif as gif_decoder;
use gif::{Encoder, Frame};

#[test]
fn composites_the_first_frame_at_its_logical_screen_offset() {
    let mut bytes = Vec::new();
    let mut writer = Encoder::new(&mut bytes, 4, 3, &[0, 0, 0, 255, 0, 0]).unwrap();
    let frame = Frame {
        left: 1,
        top: 1,
        width: 2,
        height: 2,
        transparent: Some(0),
        buffer: Cow::Borrowed(&[1, 0, 0, 1]),
        ..Frame::default()
    };
    writer.write_frame(&frame).unwrap();
    drop(writer);

    let image = gif_decoder::decode(bytes).unwrap();

    assert_eq!(image.dimensions(), (4, 3));
    assert_eq!(pixel(image.rgba8(), 4, 1, 1), [255, 0, 0, 255]);
    assert_eq!(pixel(image.rgba8(), 4, 2, 1)[3], 0);
    assert_eq!(pixel(image.rgba8(), 4, 1, 2)[3], 0);
    assert_eq!(pixel(image.rgba8(), 4, 2, 2), [255, 0, 0, 255]);
    assert_eq!(pixel(image.rgba8(), 4, 0, 0), [0, 0, 0, 0]);
}

#[test]
fn recognizes_both_standard_signatures() {
    assert!(gif_decoder::has_signature(b"GIF87a"));
    assert!(gif_decoder::has_signature(b"GIF89a"));
    assert!(!gif_decoder::has_signature(b"GIF90a"));
}

fn pixel(samples: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let start = (y * width + x) * 4;
    samples[start..start + 4].try_into().unwrap()
}
