use four::jpeg_xl::{CODESTREAM_SIGNATURE, CONTAINER_SIGNATURE, decode, has_signature};

const CODESTREAM: [u8; 42] = [
    0xff, 0x0a, 0x30, 0x54, 0x10, 0x09, 0x08, 0x06, 0x01, 0x00, 0x78, 0x00, 0x4b, 0x38, 0x41, 0x3c,
    0xb6, 0x3a, 0x51, 0xfe, 0x00, 0x47, 0x1e, 0xa0, 0x85, 0xb8, 0x27, 0x1a, 0x48, 0x45, 0x84, 0x1b,
    0x71, 0x4f, 0xa8, 0x3e, 0x8e, 0x30, 0x03, 0x92, 0x84, 0x01,
];

#[test]
fn decodes_raw_codestream_to_rgba8() {
    let image = decode(CODESTREAM).expect("the reference codestream should decode");

    assert_eq!(image.dimensions(), (240, 135));
    assert_eq!(image.rgba8().len(), 240 * 135 * 4);
    assert_eq!(
        &image.rgba8()[..12],
        &[6, 6, 6, 255, 12, 12, 12, 255, 18, 18, 18, 255]
    );
    let (pixels, remainder) = image.rgba8().as_chunks::<4>();
    assert_eq!(remainder, []);
    assert!(pixels.iter().all(|pixel| pixel[3] == 255));
}

#[test]
fn decodes_container_identically_to_raw_codestream() {
    let box_size = u32::try_from(8 + CODESTREAM.len()).unwrap();
    let mut container = Vec::from(CONTAINER_SIGNATURE);
    container.extend_from_slice(&box_size.to_be_bytes());
    container.extend_from_slice(b"jxlc");
    container.extend_from_slice(&CODESTREAM);

    let raw = decode(CODESTREAM).expect("the reference codestream should decode");
    let boxed = decode(&container).expect("the reference container should decode");

    assert_eq!(boxed, raw);
}

#[test]
fn recognizes_only_complete_standard_signatures() {
    assert!(has_signature(CODESTREAM_SIGNATURE));
    assert!(has_signature(CONTAINER_SIGNATURE));
    assert!(!has_signature([CODESTREAM_SIGNATURE[0]]));
    assert!(!has_signature(&CONTAINER_SIGNATURE[..11]));
    assert!(!has_signature(b"not a JPEG XL image"));
}

#[test]
fn rejects_an_unrecognized_signature_before_codec_parsing() {
    let error = decode(b"not a JPEG XL image").unwrap_err();

    assert_eq!(
        error.to_string(),
        "expected a JPEG XL codestream or container signature"
    );
}

#[test]
fn reports_a_truncated_codestream_as_a_codec_error() {
    let error = decode(CODESTREAM_SIGNATURE).unwrap_err();

    assert!(error.to_string().starts_with("JPEG XL codec error:"));
}
