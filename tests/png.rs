use crc32fast::Hasher;
use four::png::{self, PNGError};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

#[test]
fn decodes_the_three_pixel_rgb_example() {
    let scanlines = [0, 255, 0, 0, 0, 255, 0, 0, 0, 255];
    let encoded = make_png(3, 1, 8, 2, 0, &[], None, &scanlines);
    let image = png::decode(encoded).unwrap();

    assert_eq!(image.dimensions(), (3, 1));
    assert_eq!(
        image.rgba8(),
        &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]
    );
}

#[test]
fn reverses_all_five_scanline_filters() {
    let rows = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [19, 18, 17, 16, 15, 14, 13, 12],
        [21, 34, 55, 89, 144, 233, 3, 5],
        [8, 13, 21, 34, 55, 89, 144, 233],
        [250, 200, 150, 100, 50, 25, 12, 6],
    ];
    let mut scanlines = Vec::new();
    let mut previous = [0_u8; 8];
    for (filter, row) in rows.iter().enumerate() {
        let filter = u8::try_from(filter).unwrap();
        scanlines.push(filter);
        scanlines.extend(filter_row(filter, row, &previous, 4));
        previous = *row;
    }
    let encoded = make_png(2, 5, 8, 6, 0, &[], None, &scanlines);
    let image = png::decode(encoded).unwrap();

    assert_eq!(image.rgba8(), rows.concat());
}

#[test]
fn expands_packed_palette_pixels_and_transparency() {
    let palette = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
    let scanlines = [0, 0b00_01_10_11];
    let encoded = make_png(4, 1, 2, 3, 0, &palette, Some(&[0, 64, 128]), &scanlines);
    let image = png::decode(encoded).unwrap();

    assert_eq!(
        image.rgba8(),
        &[
            255, 0, 0, 0, 0, 255, 0, 64, 0, 0, 255, 128, 255, 255, 255, 255,
        ]
    );
}

#[test]
fn decodes_one_bit_grayscale_with_a_transparent_sample() {
    let scanlines = [0, 0b1010_0000];
    let encoded = make_png(4, 1, 1, 0, 0, &[], Some(&[0, 1]), &scanlines);
    let image = png::decode(encoded).unwrap();

    assert_eq!(
        image.rgba8(),
        &[
            255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255,
        ]
    );
}

#[test]
fn reduces_sixteen_bit_rgba_samples_to_rgba8() {
    let scanlines = [0, 0xff, 0xff, 0x80, 0x00, 0, 0, 0x40, 0x00];
    let encoded = make_png(1, 1, 16, 6, 0, &[], None, &scanlines);
    let image = png::decode(encoded).unwrap();

    assert_eq!(image.rgba8(), &[255, 128, 0, 64]);
}

#[test]
fn combines_adam7_passes_into_display_order() {
    let width = 5_usize;
    let height = 5_usize;
    let pixels: Vec<[u8; 4]> = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                [
                    u8::try_from(x).unwrap() * 30,
                    u8::try_from(y).unwrap() * 40,
                    u8::try_from(x + y).unwrap() * 20,
                    255,
                ]
            })
        })
        .collect();
    let mut scanlines = Vec::new();
    for (x_start, y_start, x_step, y_step) in ADAM7 {
        for y in (y_start..height).step_by(y_step) {
            if x_start >= width {
                continue;
            }
            scanlines.push(0);
            for x in (x_start..width).step_by(x_step) {
                scanlines.extend_from_slice(&pixels[y * width + x]);
            }
        }
    }
    let encoded = make_png(
        u32::try_from(width).unwrap(),
        u32::try_from(height).unwrap(),
        8,
        6,
        1,
        &[],
        None,
        &scanlines,
    );
    let image = png::decode(encoded).unwrap();

    assert_eq!(image.rgba8(), pixels.concat());
}

#[test]
fn rejects_a_bad_chunk_crc_before_decoding_pixels() {
    let mut encoded = make_png(1, 1, 8, 0, 0, &[], None, &[0, 0]);
    encoded[29] ^= 1;
    let error = png::decode(encoded).unwrap_err();

    assert!(matches!(&*error, PNGError::Codec(_)));
    assert!(error.to_string().to_ascii_lowercase().contains("crc"));
}

#[allow(
    clippy::too_many_arguments,
    reason = "the test helper mirrors the compact IHDR and optional color chunks"
)]
fn make_png(
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    interlace: u8,
    palette: &[u8],
    transparency: Option<&[u8]>,
    scanlines: &[u8],
) -> Vec<u8> {
    let mut png = SIGNATURE.to_vec();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[bit_depth, color_type, 0, 0, interlace]);
    add_chunk(&mut png, *b"IHDR", &header);
    if !palette.is_empty() {
        add_chunk(&mut png, *b"PLTE", palette);
    }
    if let Some(transparency) = transparency {
        add_chunk(&mut png, *b"tRNS", transparency);
    }
    add_chunk(&mut png, *b"IDAT", &zlib_stored(scanlines));
    add_chunk(&mut png, *b"IEND", &[]);
    png
}

fn add_chunk(png: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    png.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
    png.extend_from_slice(&kind);
    png.extend_from_slice(data);
    let mut hasher = Hasher::new();
    hasher.update(&kind);
    hasher.update(data);
    png.extend_from_slice(&hasher.finalize().to_be_bytes());
}

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let length = u16::try_from(data.len()).unwrap();
    let mut encoded = vec![0x78, 0x01, 0x01];
    encoded.extend_from_slice(&length.to_le_bytes());
    encoded.extend_from_slice(&(!length).to_le_bytes());
    encoded.extend_from_slice(data);
    encoded.extend_from_slice(&adler32(data).to_be_bytes());
    encoded
}

fn adler32(bytes: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;

    let (a, b) = bytes.iter().fold((1_u32, 0_u32), |(a, b), &byte| {
        let a = (a + u32::from(byte)) % MODULUS;
        (a, (b + a) % MODULUS)
    });

    b << 16 | a
}

fn filter_row(filter: u8, row: &[u8], previous: &[u8], bpp: usize) -> Vec<u8> {
    row.iter()
        .enumerate()
        .map(|(index, &byte)| {
            let left = index.checked_sub(bpp).map_or(0, |left| row[left]);
            let above = previous[index];
            let upper_left = index.checked_sub(bpp).map_or(0, |left| previous[left]);
            let predictor = match filter {
                0 => 0,
                1 => left,
                2 => above,
                3 => u8::try_from(u16::midpoint(u16::from(left), u16::from(above))).unwrap(),
                4 => paeth(left, above, upper_left),
                _ => unreachable!(),
            };
            byte.wrapping_sub(predictor)
        })
        .collect()
}

fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let estimate = i32::from(left) + i32::from(above) - i32::from(upper_left);
    let distances = [
        (estimate - i32::from(left)).abs(),
        (estimate - i32::from(above)).abs(),
        (estimate - i32::from(upper_left)).abs(),
    ];
    if distances[0] <= distances[1] && distances[0] <= distances[2] {
        left
    } else if distances[1] <= distances[2] {
        above
    } else {
        upper_left
    }
}
