use four::jpeg_xr::{SIGNATURE, decode, has_signature};

#[test]
fn recognizes_only_the_complete_signature() {
    assert!(has_signature(SIGNATURE));
    assert!(!has_signature(&SIGNATURE[..3]));
    assert!(!has_signature(b"not a JPEG XR image"));
}

#[test]
fn rejects_an_unrecognized_signature_before_codec_parsing() {
    let error = decode(b"not a JPEG XR image").unwrap_err();

    assert_eq!(error.to_string(), "expected a JPEG XR file signature");
}

#[test]
fn reports_a_truncated_file_as_a_codec_error() {
    let error = decode(SIGNATURE).unwrap_err();

    assert!(error.to_string().starts_with("JPEG XR codec error:"));
}
