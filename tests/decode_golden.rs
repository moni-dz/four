//! Pins the decoded output of every fixture so a change cannot alter pixels unnoticed.
//!
//! Each fixture is checked three ways. Dimensions catch a structural break. An FNV-1a-64 hash over
//! the whole RGBA buffer catches any change at all, down to a single bit. A four-by-four
//! average-pooled thumbnail catches the same changes but reports *how far* the output moved, which
//! is what tells a deliberate sub-ULP shift apart from a real regression.
//!
//! Re-record a golden only alongside a change that is expected to alter output, in its own commit,
//! stating the reason and the magnitude of the difference. Set `FOUR_RECORD_GOLDEN=1` to print the
//! constants for the current build.

use four::{DecodedImage, gif, jpeg, jpeg_xl, png, tiff};

/// One pinned fixture.
struct Golden {
    /// Path under `tests/fixtures/`.
    file: &'static str,
    /// Expected `(width, height)`.
    dimensions: (u32, u32),
    /// FNV-1a-64 over the decoded RGBA bytes.
    hash: u64,
    /// Four-by-four average-pooled RGBA thumbnail.
    thumbnail: [[u8; 4]; 16],
}

/// Decodes `bytes` with the decoder that owns `file`'s extension.
fn decode(file: &str, bytes: &[u8]) -> DecodedImage {
    let extension = file
        .rsplit('.')
        .next()
        .expect("fixture names have extensions");
    match extension {
        "gif" => gif::decode(bytes).expect("fixture decodes as GIF"),
        "jpg" | "jpeg" => jpeg::decode(bytes).expect("fixture decodes as JPEG"),
        "jxl" => jpeg_xl::decode(bytes).expect("fixture decodes as JPEG XL"),
        "png" => png::decode(bytes).expect("fixture decodes as PNG"),
        "tiff" | "tif" => tiff::decode(bytes).expect("fixture decodes as TIFF"),
        other => panic!("no decoder is registered for the {other} fixture extension"),
    }
}

/// Returns the FNV-1a-64 hash of `bytes`.
///
/// Inlined rather than pulled from a crate: the golden test needs a stable digest, not a good one,
/// and a dependency here would have to be trusted to never change its output across versions.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Average-pools `image` into a four-by-four RGBA thumbnail.
///
/// Averaging in `u32` keeps every source pixel in the result, so a one-level shift in a single
/// pixel still moves the pooled value on a small fixture while staying stable under rounding.
fn thumbnail(image: &DecodedImage) -> [[u8; 4]; 16] {
    const SIDE: usize = 4;

    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;
    let rgba = image.rgba8();

    let mut sums = [[0_u64; 4]; SIDE * SIDE];
    let mut counts = [0_u64; SIDE * SIDE];

    for y in 0..height {
        // Integer bucketing, so the last row and column land in the final cell rather than
        // overflowing it.
        let cell_y = (y * SIDE / height).min(SIDE - 1);
        for x in 0..width {
            let cell_x = (x * SIDE / width).min(SIDE - 1);
            let cell = cell_y * SIDE + cell_x;
            let pixel = (y * width + x) * 4;
            for channel in 0..4 {
                sums[cell][channel] += u64::from(rgba[pixel + channel]);
            }
            counts[cell] += 1;
        }
    }

    std::array::from_fn(|cell| {
        let count = counts[cell].max(1);
        std::array::from_fn(|channel| {
            u8::try_from((sums[cell][channel] + count / 2) / count).expect("a channel mean is a u8")
        })
    })
}

/// Prints the constants for a fixture, for use when re-recording.
fn record(file: &str, image: &DecodedImage) {
    let (width, height) = image.dimensions();
    println!("    Golden {{");
    println!("        file: {file:?},");
    println!("        dimensions: ({width}, {height}),");
    let hash = format!("{:016x}", fnv1a64(image.rgba8()));
    let grouped = hash
        .as_bytes()
        .chunks(4)
        .map(|chunk| std::str::from_utf8(chunk).expect("hex digits are ASCII"))
        .collect::<Vec<_>>()
        .join("_");
    println!("        hash: 0x{grouped},");
    println!("        thumbnail: [");
    for pixel in thumbnail(image) {
        println!(
            "            [{}, {}, {}, {}],",
            pixel[0], pixel[1], pixel[2], pixel[3]
        );
    }
    println!("        ],");
    println!("    }},");
}

#[test]
fn fixtures_decode_to_their_recorded_pixels() {
    let recording = std::env::var_os("FOUR_RECORD_GOLDEN").is_some();
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    for golden in GOLDENS {
        let path = directory.join(golden.file);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read fixture {}: {error}", path.display()));
        let image = decode(golden.file, &bytes);

        if recording {
            record(golden.file, &image);
            continue;
        }

        assert_eq!(
            image.dimensions(),
            golden.dimensions,
            "{} decoded to the wrong dimensions",
            golden.file
        );
        assert_eq!(
            thumbnail(&image),
            golden.thumbnail,
            "{} pixels moved; the pooled thumbnail shows by how much",
            golden.file
        );
        assert_eq!(
            fnv1a64(image.rgba8()),
            golden.hash,
            "{} pixels changed without moving the pooled thumbnail",
            golden.file
        );
    }
}

/// Number of leading bytes of every fixture that the truncation test covers exhaustively.
const EXHAUSTIVE_PREFIX: usize = 512;

/// Every fixture decodes without panicking at any truncation point.
///
/// Truncation is the cheapest way to reach the error paths of a bit-oriented decoder, and those
/// paths are the ones an untrusted file exercises. Nothing here asserts a particular error, only
/// that a short read never panics, hangs, or reads out of bounds.
#[test]
fn truncated_fixtures_fail_without_panicking() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    for golden in GOLDENS {
        let path = directory.join(golden.file);
        let bytes = std::fs::read(&path).expect("read fixture");
        let extension = golden
            .file
            .rsplit('.')
            .next()
            .expect("fixture names have extensions");

        // Every offset for a small fixture. For a large one, every offset through the header
        // region — where the interesting state machines live — then a bounded sample of the
        // entropy-coded body, since the cost of this test grows with the square of the size.
        let step = (bytes.len() / EXHAUSTIVE_PREFIX).max(1);

        for length in (0..bytes.len())
            .filter(|length| *length < EXHAUSTIVE_PREFIX || length.is_multiple_of(step))
        {
            let prefix = &bytes[..length];
            // Each decoder has its own error type, so the arms are collapsed to the one fact
            // this test cares about: the call returned rather than panicking.
            let _decoded: bool = match extension {
                "gif" => gif::decode(prefix).is_ok(),
                "jpg" | "jpeg" => jpeg::decode(prefix).is_ok(),
                "jxl" => jpeg_xl::decode(prefix).is_ok(),
                "png" => png::decode(prefix).is_ok(),
                "tiff" | "tif" => tiff::decode(prefix).is_ok(),
                other => panic!("no decoder is registered for the {other} fixture extension"),
            };
        }
    }
}

const GOLDENS: &[Golden] = &[
    Golden {
        file: "large.tiff",
        dimensions: (640, 480),
        hash: 0x58a1_84c4_8a01_6025,
        thumbnail: [
            [31, 28, 124, 255],
            [95, 28, 143, 255],
            [159, 28, 134, 255],
            [223, 28, 122, 255],
            [31, 94, 145, 255],
            [95, 94, 125, 255],
            [159, 94, 131, 255],
            [223, 94, 141, 255],
            [31, 160, 125, 255],
            [95, 160, 142, 255],
            [159, 160, 133, 255],
            [223, 160, 121, 255],
            [31, 226, 144, 255],
            [95, 226, 124, 255],
            [159, 226, 130, 255],
            [223, 226, 140, 255],
        ],
    },
    Golden {
        file: "busy.jpg",
        dimensions: (384, 384),
        hash: 0x7c62_0eb0_8b07_3c7d,
        thumbnail: [
            [62, 127, 58, 255],
            [126, 127, 147, 255],
            [190, 127, 150, 255],
            [131, 127, 70, 255],
            [62, 127, 146, 255],
            [126, 127, 115, 255],
            [190, 127, 125, 255],
            [130, 127, 149, 255],
            [62, 127, 150, 255],
            [126, 127, 125, 255],
            [190, 127, 115, 255],
            [131, 127, 146, 255],
            [62, 127, 68, 255],
            [126, 127, 150, 255],
            [190, 127, 146, 255],
            [132, 127, 59, 255],
        ],
    },
    Golden {
        file: "baseline.jpg",
        dimensions: (64, 48),
        hash: 0x3d6b_f4f9_5e33_32ab,
        thumbnail: [
            [30, 30, 144, 255],
            [94, 30, 143, 255],
            [159, 30, 143, 255],
            [224, 30, 141, 255],
            [30, 94, 143, 255],
            [186, 38, 140, 255],
            [220, 37, 118, 255],
            [224, 95, 141, 255],
            [30, 160, 142, 255],
            [185, 68, 118, 255],
            [220, 69, 140, 255],
            [224, 160, 143, 255],
            [29, 225, 141, 255],
            [94, 225, 143, 255],
            [159, 225, 144, 255],
            [223, 225, 145, 255],
        ],
    },
    Golden {
        file: "rgb8.png",
        dimensions: (64, 48),
        hash: 0x2c36_ea9e_08c3_67c9,
        thumbnail: [
            [30, 29, 144, 255],
            [95, 29, 144, 255],
            [159, 29, 144, 255],
            [224, 29, 144, 255],
            [30, 94, 144, 255],
            [187, 37, 143, 255],
            [221, 37, 115, 255],
            [224, 94, 144, 255],
            [30, 160, 144, 255],
            [187, 68, 115, 255],
            [221, 68, 143, 255],
            [224, 160, 144, 255],
            [30, 225, 144, 255],
            [95, 225, 144, 255],
            [159, 225, 144, 255],
            [224, 225, 144, 255],
        ],
    },
    Golden {
        file: "rgb16.png",
        dimensions: (64, 48),
        hash: 0xbe2d_0758_68d5_131c,
        thumbnail: [
            [30, 29, 143, 255],
            [94, 29, 143, 255],
            [159, 29, 143, 255],
            [224, 29, 143, 255],
            [30, 94, 143, 255],
            [186, 36, 142, 255],
            [221, 36, 114, 255],
            [224, 94, 143, 255],
            [30, 159, 144, 255],
            [186, 67, 114, 255],
            [221, 67, 142, 255],
            [224, 159, 143, 255],
            [29, 224, 144, 255],
            [94, 224, 144, 255],
            [159, 225, 143, 255],
            [223, 225, 143, 255],
        ],
    },
    Golden {
        file: "rgba8.png",
        dimensions: (64, 48),
        hash: 0x236d_92bc_27a7_8da1,
        thumbnail: [
            [30, 29, 144, 46],
            [95, 29, 144, 111],
            [159, 29, 144, 175],
            [224, 29, 144, 238],
            [30, 94, 144, 46],
            [187, 37, 143, 111],
            [221, 37, 115, 175],
            [224, 94, 144, 238],
            [30, 160, 144, 46],
            [187, 68, 115, 111],
            [221, 68, 143, 175],
            [224, 160, 144, 238],
            [30, 225, 144, 46],
            [95, 225, 144, 111],
            [159, 225, 144, 175],
            [224, 225, 144, 238],
        ],
    },
    Golden {
        file: "palette8.png",
        dimensions: (64, 48),
        hash: 0x45b4_2042_8aa4_a55e,
        thumbnail: [
            [35, 32, 144, 255],
            [95, 34, 144, 255],
            [160, 39, 144, 255],
            [222, 40, 144, 255],
            [36, 94, 144, 255],
            [186, 37, 143, 255],
            [221, 37, 115, 255],
            [223, 94, 144, 255],
            [46, 160, 144, 255],
            [185, 69, 115, 255],
            [222, 69, 143, 255],
            [218, 160, 148, 255],
            [84, 225, 144, 255],
            [99, 225, 144, 255],
            [159, 224, 144, 255],
            [219, 224, 145, 255],
        ],
    },
    Golden {
        file: "rgb8.tiff",
        dimensions: (64, 48),
        hash: 0x2c36_ea9e_08c3_67c9,
        thumbnail: [
            [30, 29, 144, 255],
            [95, 29, 144, 255],
            [159, 29, 144, 255],
            [224, 29, 144, 255],
            [30, 94, 144, 255],
            [187, 37, 143, 255],
            [221, 37, 115, 255],
            [224, 94, 144, 255],
            [30, 160, 144, 255],
            [187, 68, 115, 255],
            [221, 68, 143, 255],
            [224, 160, 144, 255],
            [30, 225, 144, 255],
            [95, 225, 144, 255],
            [159, 225, 144, 255],
            [224, 225, 144, 255],
        ],
    },
    Golden {
        file: "rgb8.jxl",
        dimensions: (64, 48),
        hash: 0xf3ed_9e29_ff88_99de,
        thumbnail: [
            [30, 29, 143, 255],
            [95, 29, 143, 255],
            [159, 29, 142, 255],
            [224, 29, 142, 255],
            [31, 94, 143, 255],
            [187, 37, 142, 255],
            [221, 38, 115, 255],
            [224, 94, 140, 255],
            [30, 160, 142, 255],
            [186, 69, 114, 255],
            [221, 69, 143, 255],
            [224, 160, 142, 255],
            [29, 225, 144, 255],
            [95, 225, 143, 255],
            [159, 225, 143, 255],
            [224, 225, 141, 255],
        ],
    },
    Golden {
        file: "animated.gif",
        dimensions: (64, 48),
        hash: 0xc5b9_ec3e_8677_11e0,
        thumbnail: [
            [29, 28, 141, 255],
            [95, 28, 142, 255],
            [159, 29, 142, 255],
            [224, 29, 142, 255],
            [29, 94, 143, 255],
            [187, 36, 144, 255],
            [221, 37, 115, 255],
            [224, 94, 143, 255],
            [29, 159, 142, 255],
            [186, 68, 114, 255],
            [220, 68, 142, 255],
            [224, 159, 143, 255],
            [29, 226, 142, 255],
            [94, 225, 143, 255],
            [159, 225, 143, 255],
            [224, 226, 143, 255],
        ],
    },
];
