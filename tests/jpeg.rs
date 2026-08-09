mod support;

use std::sync::Arc;

use four::Image;
use four::jpeg::{self, JPEGError, JPEGTableKind};

#[test]
fn decodes_progressive_spectral_and_refinement_scans() {
    let jpeg = support::progressive_color_jpeg();
    let image = jpeg::decode(&jpeg).unwrap();

    assert_eq!(image.dimensions(), (227, 149));
    assert_eq!(image.rgba8().len(), 227 * 149 * 4);
    assert_matches_reference_decoder(&jpeg, &image);
}

#[test]
fn decodes_sequential_arithmetic_scans() {
    // This libjpeg-turbo fixture checks the QM transitions and JPEG contexts together.
    let jpeg = support::sequential_arithmetic_jpeg();
    let image = jpeg::decode(&jpeg).unwrap();
    let huffman = jpeg::decode(support::progressive_color_jpeg()).unwrap();

    assert_eq!(image.dimensions(), (227, 149));
    assert_eq!(image.rgba8().len(), 227 * 149 * 4);
    assert_images_close(&image, &huffman);
}

#[test]
fn decodes_progressive_arithmetic_refinement_scans() {
    // Matching entropy-independent pixels verify SOF10 first and refinement scan contexts.
    let jpeg = support::progressive_arithmetic_jpeg();
    let image = jpeg::decode(&jpeg).unwrap();
    let huffman = jpeg::decode(support::progressive_color_jpeg()).unwrap();

    assert_eq!(image.dimensions(), (227, 149));
    assert_eq!(image.rgba8().len(), 227 * 149 * 4);
    assert_images_close(&image, &huffman);
}

#[test]
fn rejects_an_inverted_dc_arithmetic_conditioning_range() {
    let jpeg = [0xff, 0xd8, 0xff, 0xcc, 0x00, 0x04, 0x00, 0x01];
    let error = jpeg::decode(jpeg).unwrap_err();

    assert_eq!(
        &*error,
        &JPEGError::Table(
            JPEGTableKind::ArithmeticConditioning,
            "DC arithmetic conditioning requires L <= U",
        )
    );
    assert!(error.to_string().contains("L <= U"));
}

#[test]
fn rejects_a_zero_ac_arithmetic_conditioning_value() {
    let jpeg = [0xff, 0xd8, 0xff, 0xcc, 0x00, 0x04, 0x10, 0x00];
    let error = jpeg::decode(jpeg).unwrap_err();

    assert_eq!(
        &*error,
        &JPEGError::Table(
            JPEGTableKind::ArithmeticConditioning,
            "AC arithmetic conditioning must be in 1..=63",
        )
    );
    assert!(error.to_string().contains("1..=63"));
}

#[test]
fn rejects_a_scan_before_the_frame_phase() {
    let error = jpeg::decode([0xff, 0xd8, 0xff, 0xda]).unwrap_err();

    assert!(matches!(&*error, JPEGError::Codec(_)));
    assert_ne!(error.to_string(), "");
}

#[test]
fn routes_an_unsupported_huffman_marker_to_the_codec() {
    let error = jpeg::decode([0xff, 0xd8, 0xff, 0x01]).unwrap_err();

    assert!(matches!(&*error, JPEGError::Codec(_)));
    assert!(error.to_string().contains("JPEG codec error"));
}

#[test]
fn rejects_end_of_image_before_the_scanned_phase() {
    let jpeg = [
        0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xff, 0xd9,
    ];
    let error = jpeg::decode(jpeg).unwrap_err();

    assert!(matches!(&*error, JPEGError::Codec(_)));
    assert_ne!(error.to_string(), "");
}

#[test]
fn rejects_zero_dimensions_before_allocating() {
    let jpeg = [
        0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
    ];
    let error = jpeg::decode(jpeg).unwrap_err();

    assert!(matches!(&*error, JPEGError::Codec(_)));
    assert_ne!(error.to_string(), "");
}

#[test]
fn truncated_segment_returns_an_error_instead_of_panicking() {
    let error = jpeg::decode([0xff, 0xd8, 0xff, 0xfe, 0x00]).unwrap_err();

    assert!(matches!(&*error, JPEGError::Codec(_)));
    assert_ne!(error.to_string(), "");
    assert!(error.frame().location().file().ends_with("mod.rs"));
    assert!(error.frame().children().is_empty());
}

#[test]
fn decodes_a_subsampled_color_jpeg_end_to_end() {
    let image = jpeg::decode(support::baseline_color_jpeg()).unwrap();

    assert_eq!(image.dimensions(), (16, 16));
    assert_eq!(image.rgba8().len(), 16 * 16 * 4);
    assert_dominant(&image, 3, 3, 0);
    assert_dominant(&image, 12, 3, 1);
    assert_dominant(&image, 3, 12, 2);
    let white = pixel(&image, 12, 12);
    assert!(white[0] > 220 && white[1] > 220 && white[2] > 220);
}

#[test]
fn ignores_arithmetic_marker_bytes_inside_metadata() {
    let baseline = support::baseline_color_jpeg();
    let mut jpeg = Vec::with_capacity(baseline.len() + 6);
    jpeg.extend_from_slice(&baseline[..2]);
    jpeg.extend_from_slice(&[0xff, 0xe1, 0x00, 0x04, 0xff, 0xcc]);
    jpeg.extend_from_slice(&baseline[2..]);

    let image = jpeg::decode(jpeg).unwrap();

    assert_eq!(image.dimensions(), (16, 16));
}

fn assert_dominant(image: &dyn Image, x: usize, y: usize, channel: usize) {
    let pixel = pixel(image, x, y);
    assert!(pixel[channel] > 180);
    for other_channel in 0..3 {
        if other_channel != channel {
            assert!(pixel[channel] > pixel[other_channel] + 80);
        }
    }
}

fn pixel(image: &dyn Image, x: usize, y: usize) -> &[u8] {
    assert!(x < image.width() as usize);
    assert!(y < image.height() as usize);

    let start = (y * image.width() as usize + x) * 4;
    &image.rgba8()[start..start + 4]
}

fn assert_matches_reference_decoder(jpeg: &[u8], image: &dyn Image) {
    let reference = gpui::Image::from_bytes(gpui::ImageFormat::Jpeg, jpeg.to_vec());
    let svg_renderer = gpui::SvgRenderer::new(Arc::new(()));
    let reference_image = reference.to_image_data(svg_renderer).unwrap();
    let reference_bgra = reference_image.as_bytes(0).unwrap();
    assert_eq!(reference_bgra.len(), image.rgba8().len());

    let (actual_pixels, actual_remainder) = image.rgba8().as_chunks::<4>();
    let (expected_pixels, expected_remainder) = reference_bgra.as_chunks::<4>();
    assert_eq!(actual_remainder.len(), 0);
    assert_eq!(expected_remainder.len(), 0);
    let error_sum = actual_pixels
        .iter()
        .zip(expected_pixels)
        .map(|(actual, expected)| {
            u64::from(actual[0].abs_diff(expected[2]))
                + u64::from(actual[1].abs_diff(expected[1]))
                + u64::from(actual[2].abs_diff(expected[0]))
        })
        .sum::<u64>();
    let sample_count = u64::from(image.width()) * u64::from(image.height()) * 3;
    assert!(error_sum / sample_count < 20);
}

fn assert_images_close(actual: &dyn Image, expected: &dyn Image) {
    assert_eq!(actual.dimensions(), expected.dimensions());
    let (error_sum, maximum_error) = actual
        .rgba8()
        .iter()
        .zip(expected.rgba8())
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .fold((0_u64, 0_u8), |(sum, maximum), error| {
            (sum + u64::from(error), maximum.max(error))
        });

    let sample_count = u64::try_from(actual.rgba8().len()).unwrap();
    assert!(
        error_sum / sample_count <= 3,
        "mean decoder difference was {}",
        error_sum / sample_count
    );
    assert!(
        maximum_error <= 32,
        "maximum decoder difference was {maximum_error}"
    );
}
