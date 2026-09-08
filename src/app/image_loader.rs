//! Loads bounded images.

use std::borrow::Cow;
use std::fmt;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use exn::{ErrorExt, ResultExt};
use gpui::{Image as GPUIImage, ImageFormat, SharedString};
use tonemapping::{MaxCLLMode, ToneMappingMethod};

use four::{DecodedImage, encode_bmp, gif, jpeg, jpeg_xl, jpeg_xr, png, tiff};

const ERROR_FRAMES_MAX: u32 = 8;
const MEBIBYTE_BYTES: u64 = 1024 * 1024;
const IMAGE_FILE_MEBIBYTES_MAX: u64 = 128;
const IMAGE_FILE_BYTES_MAX: u64 = IMAGE_FILE_MEBIBYTES_MAX * MEBIBYTE_BYTES;

pub(super) type LoadException = exn::Exn<LoadError>;
pub(super) type LoadResult<T> = exn::Result<T, LoadError>;

#[derive(Debug)]
pub(super) struct LoadError {
    message: String,
}

impl LoadError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        assert_ne!(message.len(), 0, "load error message must not be blank");
        Self { message }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LoadError {}

#[track_caller]
fn load_error(message: impl Into<String>) -> LoadException {
    LoadError::new(message).raise()
}

pub(super) fn format_load_error(error: &LoadException) -> String {
    let mut frame = error.frame();
    let mut message = frame.error().to_string();

    for _ in 0..ERROR_FRAMES_MAX {
        let Some(child) = frame.children().first() else {
            assert_ne!(message.len(), 0, "formatted load error must not be blank");
            return message;
        };

        write!(&mut message, ": {}", child.error()).expect("writing to a string cannot fail");
        frame = child;
    }

    if !frame.children().is_empty() {
        message.push_str(": additional error context omitted");
    }

    assert_ne!(message.len(), 0, "formatted load error must not be blank");
    message
}

#[derive(Clone)]
pub(super) struct DisplayedImage {
    pub(super) image: Arc<GPUIImage>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) source_path: Arc<Path>,
    pub(super) hdr_options: Option<HDROptions>,
    /// The native (pre-tone-mapping) decode, retained so [`retint_jpeg_xr`] can apply a different
    /// tone-mapping method without re-running JPEG XR's entropy decode.
    pub(super) native_jpeg_xr: Option<Arc<jpeg_xr::NativeJPEGXR>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HDROptions {
    tone_mapping: ToneMappingMethod,
}

impl HDROptions {
    pub(super) const fn tone_mapping(self) -> ToneMappingMethod {
        self.tone_mapping
    }

    pub(super) const fn with_tone_mapping(self, tone_mapping: ToneMappingMethod) -> Self {
        Self { tone_mapping }
    }
}

impl Default for HDROptions {
    fn default() -> Self {
        Self {
            tone_mapping: ToneMappingMethod::default(),
        }
    }
}

pub(super) struct LoadedImage {
    pub(super) displayed: DisplayedImage,
    pub(super) status: SharedString,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceFormat {
    GIF,
    JPEG,
    JPEGXL,
    JPEGXR,
    PNG,
    TIFF,
}

impl SourceFormat {
    fn detect(bytes: &[u8], extension: Option<&str>) -> Self {
        if jpeg_xl::has_signature(bytes) {
            return Self::JPEGXL;
        }

        if jpeg_xr::has_signature(bytes) {
            return Self::JPEGXR;
        }

        if png::has_signature(bytes) {
            return Self::PNG;
        }

        if gif::has_signature(bytes) {
            return Self::GIF;
        }

        if tiff::has_signature(bytes) {
            return Self::TIFF;
        }

        if jpeg::has_signature(bytes) {
            return Self::JPEG;
        }

        let extension = extension.map(str::to_ascii_lowercase);

        match extension.as_deref() {
            Some("jxl") => Self::JPEGXL,
            Some("jxr" | "wdp" | "hdp") => Self::JPEGXR,
            Some("png") => Self::PNG,
            Some("gif") => Self::GIF,
            Some("tif" | "tiff") => Self::TIFF,
            _ => Self::JPEG,
        }
    }

    fn decode(
        self,
        bytes: &[u8],
        path: &Path,
        hdr_options: HDROptions,
    ) -> LoadResult<DecodedSource> {
        match self {
            Self::GIF => gif::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),

            Self::JPEG => jpeg::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),

            Self::JPEGXL => jpeg_xl::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),

            Self::JPEGXR => {
                let native = jpeg_xr::decode_native(bytes).or_raise(|| image_decode_error(path))?;
                let decoded = jpeg_xr::tonemap_native(&native, jpeg_xr_options(hdr_options))
                    .or_raise(|| image_decode_error(path))?;
                let metadata = decoded.metadata();
                Ok(DecodedSource {
                    image: decoded.into_image(),
                    jpeg_xr_metadata: Some(metadata),
                    native_jpeg_xr: Some(Arc::new(native)),
                })
            }

            Self::PNG => png::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),

            Self::TIFF => tiff::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),
        }
    }
}

struct DecodedSource {
    image: DecodedImage,
    jpeg_xr_metadata: Option<jpeg_xr::JPEGXRMetadata>,
    native_jpeg_xr: Option<Arc<jpeg_xr::NativeJPEGXR>>,
}

impl DecodedSource {
    fn standard(image: DecodedImage) -> Self {
        Self {
            image,
            jpeg_xr_metadata: None,
            native_jpeg_xr: None,
        }
    }
}

fn jpeg_xr_options(hdr_options: HDROptions) -> jpeg_xr::DecodeOptions {
    jpeg_xr::DecodeOptions::new(hdr_options.tone_mapping(), MaxCLLMode::Percentile99_99)
        .with_hdr_metrics(false)
}

fn image_decode_error(path: &Path) -> LoadError {
    LoadError::new(format!("Could not decode {}", path.display()))
}

fn display_file_name(path: &Path) -> Cow<'_, str> {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
}

fn validate_image_file_size(path: &Path, byte_count: u64) -> LoadResult<()> {
    if byte_count > IMAGE_FILE_BYTES_MAX {
        return Err(load_error(format!(
            "{} is larger than the {IMAGE_FILE_MEBIBYTES_MAX} MiB input limit",
            path.display()
        )));
    }

    Ok(())
}

fn display_image(source_format: SourceFormat, bytes: Vec<u8>, decoded: &DecodedImage) -> GPUIImage {
    match source_format {
        SourceFormat::GIF => GPUIImage::from_bytes(ImageFormat::Gif, bytes),
        _ => GPUIImage::from_bytes(ImageFormat::Bmp, encode_bmp(decoded)),
    }
}

pub(super) fn load_image(path: &Path) -> LoadResult<LoadedImage> {
    load_image_with(path, HDROptions::default())
}

pub(super) fn load_image_with(path: &Path, hdr_options: HDROptions) -> LoadResult<LoadedImage> {
    let file = File::open(path)
        .or_raise(|| LoadError::new(format!("Could not open {}", path.display())))?;

    let file_metadata = file
        .metadata()
        .or_raise(|| LoadError::new(format!("Could not inspect {}", path.display())))?;
    validate_image_file_size(path, file_metadata.len())?;

    let capacity = usize::try_from(file_metadata.len())
        .expect("the validated image input limit fits every supported pointer width");

    let mut bytes = Vec::with_capacity(capacity);
    file.take(IMAGE_FILE_BYTES_MAX + 1)
        .read_to_end(&mut bytes)
        .or_raise(|| LoadError::new(format!("Could not read {}", path.display())))?;

    let byte_count = u64::try_from(bytes.len()).expect("the validated image input length fits u64");

    validate_image_file_size(path, byte_count)?;

    let extension = path.extension().map(|value| value.to_string_lossy());
    let source_format = SourceFormat::detect(&bytes, extension.as_deref());

    let decoded = source_format.decode(&bytes, path, hdr_options)?;

    let (width, height) = decoded.image.dimensions();
    assert!(width > 0, "decoded image width must be nonzero");
    assert!(height > 0, "decoded image height must be nonzero");

    let active_hdr_options = decoded
        .jpeg_xr_metadata
        .filter(|metadata| metadata.is_hdr())
        .map(|_| hdr_options);

    let image = Arc::new(display_image(source_format, bytes, &decoded.image));

    let loaded = LoadedImage {
        displayed: DisplayedImage {
            image,
            width,
            height,
            source_path: Arc::from(path),
            hdr_options: active_hdr_options,
            native_jpeg_xr: decoded.native_jpeg_xr,
        },
        status: format!("{} — {width} × {height}", display_file_name(path)).into(),
    };

    assert_ne!(
        loaded.status.len(),
        0,
        "loaded image status must not be blank"
    );
    Ok(loaded)
}

/// Re-tone-maps an already-decoded native JPEG XR image with different `hdr_options`.
///
/// Skips the file read, format detection, and entropy decode that [`load_image_with`] performs;
/// only [`jpeg_xr::tonemap_native`] and BMP re-encoding run.
pub(super) fn retint_jpeg_xr(
    native: &Arc<jpeg_xr::NativeJPEGXR>,
    path: &Path,
    hdr_options: HDROptions,
) -> LoadResult<LoadedImage> {
    let decoded = jpeg_xr::tonemap_native(native, jpeg_xr_options(hdr_options))
        .or_raise(|| image_decode_error(path))?;

    let (width, height) = decoded.image().dimensions();
    assert!(width > 0, "decoded image width must be nonzero");
    assert!(height > 0, "decoded image height must be nonzero");

    let active_hdr_options = decoded.metadata().is_hdr().then_some(hdr_options);
    let image = Arc::new(display_image(
        SourceFormat::JPEGXR,
        Vec::new(),
        &decoded.into_image(),
    ));

    let loaded = LoadedImage {
        displayed: DisplayedImage {
            image,
            width,
            height,
            source_path: Arc::from(path),
            hdr_options: active_hdr_options,
            native_jpeg_xr: Some(Arc::clone(native)),
        },
        status: format!("{} — {width} × {height}", display_file_name(path)).into(),
    };

    assert_ne!(
        loaded.status.len(),
        0,
        "loaded image status must not be blank"
    );
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use ::gif::{DisposalMethod, Encoder, Frame, Repeat};

    use super::*;

    fn animated_gif() -> Vec<u8> {
        const PALETTE: &[u8] = &[
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
        ];

        let mut bytes = Vec::new();
        let mut encoder = Encoder::new(&mut bytes, 3, 1, PALETTE).unwrap();
        encoder.set_repeat(Repeat::Infinite).unwrap();
        encoder
            .write_frame(&Frame {
                width: 3,
                height: 1,
                delay: 5,
                dispose: DisposalMethod::Keep,
                buffer: Cow::Borrowed(&[0, 0, 0]),
                ..Frame::default()
            })
            .unwrap();

        encoder
            .write_frame(&Frame {
                width: 1,
                height: 1,
                delay: 7,
                dispose: DisposalMethod::Background,
                buffer: Cow::Borrowed(&[1]),
                ..Frame::default()
            })
            .unwrap();

        encoder
            .write_frame(&Frame {
                left: 2,
                width: 1,
                height: 1,
                delay: 11,
                dispose: DisposalMethod::Keep,
                buffer: Cow::Borrowed(&[2]),
                ..Frame::default()
            })
            .unwrap();

        drop(encoder);
        bytes
    }

    #[test]
    fn animated_gif_preserves_frames_timing_and_disposal_through_gpui() {
        let bytes = animated_gif();
        let decoded = gif::decode(&bytes).unwrap();
        let image = display_image(SourceFormat::GIF, bytes, &decoded);
        let rendered = image
            .to_image_data(gpui::SvgRenderer::new(Arc::new(())))
            .unwrap();

        assert_eq!(image.format(), ImageFormat::Gif);
        assert_eq!(rendered.frame_count(), 3);
        assert_eq!(rendered.delay(0).numer_denom_ms(), (50, 1));
        assert_eq!(rendered.delay(1).numer_denom_ms(), (70, 1));
        assert_eq!(rendered.delay(2).numer_denom_ms(), (110, 1));
        assert_eq!(
            rendered.as_bytes(0).unwrap(),
            &[0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255]
        );
        assert_eq!(
            rendered.as_bytes(1).unwrap(),
            &[0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255]
        );
        assert_eq!(
            rendered.as_bytes(2).unwrap(),
            &[0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255]
        );
    }

    #[test]
    fn load_error_preserves_the_decoder_error_frame() {
        let decoder_error = jpeg::decode(&[0x00]).unwrap_err();
        let load_error = decoder_error.raise(LoadError::new("Could not decode test.jpg"));
        let message = format_load_error(&load_error);

        assert!(message.contains("Could not decode test.jpg"));
        assert!(message.contains("JPEG codec error"));
        assert_eq!(load_error.frame().children().len(), 1);
        assert!(load_error.frame().children()[0].children().is_empty());
    }

    #[test]
    fn oversized_file_error_uses_the_configured_limit() {
        let path = Path::new("oversized.png");
        let error = validate_image_file_size(path, IMAGE_FILE_BYTES_MAX + 1).unwrap_err();

        assert!(
            error
                .to_string()
                .contains(&format!("{IMAGE_FILE_MEBIBYTES_MAX} MiB input limit"))
        );
    }

    #[test]
    fn source_format_extensions_are_case_insensitive() {
        assert_eq!(SourceFormat::detect(&[], Some("GIF")), SourceFormat::GIF);
        assert_eq!(SourceFormat::detect(&[], Some("HDP")), SourceFormat::JPEGXR);
        assert_eq!(SourceFormat::detect(&[], Some("TIFF")), SourceFormat::TIFF);
        assert_eq!(
            SourceFormat::detect(&[], Some("unknown")),
            SourceFormat::JPEG
        );
    }

    #[test]
    fn source_signature_takes_precedence_over_extension() {
        assert_eq!(
            SourceFormat::detect(&jpeg_xl::CODESTREAM_SIGNATURE, Some("png")),
            SourceFormat::JPEGXL
        );
    }
}
