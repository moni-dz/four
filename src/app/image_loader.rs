//! Loads bounded images and builds metadata for display.

use std::borrow::Cow;
use std::fmt;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use exn::{ErrorExt, ResultExt};
use gpui::{Image as GPUIImage, ImageFormat, SharedString};

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
        invariant!(!message.is_empty());
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
            invariant!(!message.is_empty());
            return message;
        };

        write!(&mut message, ": {}", child.error()).expect("writing to a string cannot fail");
        frame = child;
    }

    if !frame.children().is_empty() {
        message.push_str(": additional error context omitted");
    }

    invariant!(!message.is_empty());
    message
}

#[derive(Clone)]
pub(super) struct DisplayedImage {
    pub(super) image: Arc<GPUIImage>,
    pub(super) metadata: Arc<ImageMetadata>,
}

pub(super) struct LoadedImage {
    pub(super) displayed: DisplayedImage,
    pub(super) status: SharedString,
}

#[derive(Debug)]
pub(super) struct ImageMetadata {
    pub(super) fields: Vec<MetadataField>,
}

impl ImageMetadata {
    fn new(path: &Path, decoded: &DecodedImageState) -> Self {
        let (width, height) = decoded.image.dimensions();
        invariant!(width > 0);
        invariant!(height > 0);
        invariant!(decoded.byte_count <= IMAGE_FILE_BYTES_MAX);

        let divisor = greatest_common_divisor(width, height);
        let pixel_count = u64::from(width) * u64::from(height);
        let (pixels, remainder) = decoded.image.rgba8().as_chunks::<4>();
        invariant!(remainder.is_empty());
        let transparency = pixels.iter().any(|pixel| pixel[3] != u8::MAX);

        let mut fields = vec![
            MetadataField::new("Image", display_file_name(path).into_owned()),
            MetadataField::new("Folder", display_parent(path)),
            MetadataField::new("File size", format_file_size(decoded.byte_count)),
            MetadataField::new("Format", decoded.source_format.label()),
            MetadataField::new("Dimensions", format!("{width} × {height} px")),
            MetadataField::new(
                "Aspect ratio",
                format!("{}:{}", width / divisor, height / divisor),
            ),
            MetadataField::new("Pixels", format_pixel_count(pixel_count)),
        ];

        if let Some(metadata) = decoded.jpeg_xr_metadata {
            fields.push(MetadataField::new(
                "Source samples",
                format_jpeg_xr_samples(metadata),
            ));
            fields.push(MetadataField::new(
                "Dynamic range",
                if metadata.is_hdr() {
                    "HDR → SDR"
                } else {
                    "SDR"
                },
            ));

            if let (Some(scrgb), Some(nits)) = (metadata.max_cll_scrgb(), metadata.max_cll_nits()) {
                fields.push(MetadataField::new(
                    "MaxCLL",
                    format!("{scrgb:.3} scRGB · {nits:.1} cd/m²"),
                ));
            }
        }

        fields.push(MetadataField::new("Output", "RGBA · 8 bpc"));
        fields.push(MetadataField::new(
            "Transparency",
            if transparency { "Present" } else { "None" },
        ));

        invariant!(fields.iter().all(|field| !field.value.is_empty()));
        Self { fields }
    }
}

#[derive(Debug)]
pub(super) struct MetadataField {
    pub(super) label: &'static str,
    pub(super) value: SharedString,
}

impl MetadataField {
    fn new(label: &'static str, value: impl Into<SharedString>) -> Self {
        let value = value.into();
        invariant!(!label.is_empty());
        invariant!(!value.is_empty());
        Self { label, value }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceFormat {
    Gif,
    Jpeg,
    JpegXl,
    JpegXr,
    Png,
    Tiff,
}

impl SourceFormat {
    const fn label(self) -> &'static str {
        match self {
            Self::Gif => "GIF",
            Self::Jpeg => "JPEG",
            Self::JpegXl => "JPEG XL",
            Self::JpegXr => "JPEG XR",
            Self::Png => "PNG",
            Self::Tiff => "TIFF",
        }
    }

    fn detect(bytes: &[u8], extension: Option<&str>) -> Self {
        if jpeg_xl::has_signature(bytes) {
            return Self::JpegXl;
        }
        if jpeg_xr::has_signature(bytes) {
            return Self::JpegXr;
        }
        if png::has_signature(bytes) {
            return Self::Png;
        }
        if gif::has_signature(bytes) {
            return Self::Gif;
        }
        if tiff::has_signature(bytes) {
            return Self::Tiff;
        }
        if bytes.starts_with(&jpeg::SIGNATURE) {
            return Self::Jpeg;
        }

        let extension = extension.map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("jxl") => Self::JpegXl,
            Some("jxr" | "wdp" | "hdp") => Self::JpegXr,
            Some("png") => Self::Png,
            Some("gif") => Self::Gif,
            Some("tif" | "tiff") => Self::Tiff,
            _ => Self::Jpeg,
        }
    }

    fn decode(self, bytes: &[u8], path: &Path) -> LoadResult<DecodedSource> {
        match self {
            Self::Gif => gif::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),
            
            Self::Jpeg => jpeg::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),
            
            Self::JpegXl => jpeg_xl::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),
            
            Self::JpegXr => jpeg_xr::decode_with_metadata(bytes)
                .or_raise(|| image_decode_error(path))
                .map(|decoded| {
                    let metadata = decoded.metadata();
                    DecodedSource {
                        image: decoded.into_image(),
                        jpeg_xr_metadata: Some(metadata),
                    }
                }),
            
            Self::Png => png::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),
            
            Self::Tiff => tiff::decode(bytes)
                .or_raise(|| image_decode_error(path))
                .map(DecodedSource::standard),
        }
    }
}

struct DecodedSource {
    image: DecodedImage,
    jpeg_xr_metadata: Option<jpeg_xr::JPEGXRMetadata>,
}

impl DecodedSource {
    fn standard(image: DecodedImage) -> Self {
        Self {
            image,
            jpeg_xr_metadata: None,
        }
    }
}

// Loading is a linear pipeline: only read bytes can be decoded, and only decoded pixels can be
// presented. Consuming transitions prevent callers from skipping or repeating a phase.
struct ImageLoad<State> {
    path: PathBuf,
    state: State,
}

struct SelectedImage;

struct EncodedImage {
    bytes: Vec<u8>,
}

struct DecodedImageState {
    image: DecodedImage,
    byte_count: u64,
    source_format: SourceFormat,
    jpeg_xr_metadata: Option<jpeg_xr::JPEGXRMetadata>,
}

impl ImageLoad<SelectedImage> {
    fn select(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            state: SelectedImage,
        }
    }

    fn read(self) -> LoadResult<ImageLoad<EncodedImage>> {
        let file = File::open(&self.path)
            .or_raise(|| LoadError::new(format!("Could not open {}", self.path.display())))?;
        let metadata = file
            .metadata()
            .or_raise(|| LoadError::new(format!("Could not inspect {}", self.path.display())))?;
        validate_image_file_size(&self.path, metadata.len())?;
        invariant!(metadata.len() <= IMAGE_FILE_BYTES_MAX);

        let capacity = usize::try_from(metadata.len())
            .expect("the validated image input limit fits every supported pointer width");
        let mut bytes = Vec::with_capacity(capacity);
        file.take(IMAGE_FILE_BYTES_MAX + 1)
            .read_to_end(&mut bytes)
            .or_raise(|| LoadError::new(format!("Could not read {}", self.path.display())))?;

        // The bounded read catches files that grow after metadata without allocating unboundedly.
        validate_image_file_size(&self.path, bytes.len() as u64)?;

        Ok(ImageLoad {
            path: self.path,
            state: EncodedImage { bytes },
        })
    }
}

impl ImageLoad<EncodedImage> {
    fn decode(self) -> LoadResult<ImageLoad<DecodedImageState>> {
        invariant!(self.state.bytes.len() as u64 <= IMAGE_FILE_BYTES_MAX);
        invariant!(isize::try_from(self.state.bytes.len()).is_ok());

        let byte_count = u64::try_from(self.state.bytes.len())
            .expect("the validated image input length fits u64");
        let extension = self.path.extension().map(|value| value.to_string_lossy());
        let source_format = SourceFormat::detect(&self.state.bytes, extension.as_deref());
        let decoded = source_format.decode(&self.state.bytes, &self.path)?;
        let image = decoded.image;
        invariant!(image.width() > 0);
        invariant!(image.height() > 0);

        Ok(ImageLoad {
            path: self.path,
            state: DecodedImageState {
                image,
                byte_count,
                source_format,
                jpeg_xr_metadata: decoded.jpeg_xr_metadata,
            },
        })
    }
}

impl ImageLoad<DecodedImageState> {
    fn present(self) -> LoadedImage {
        let (width, height) = self.state.image.dimensions();
        invariant!(width > 0);
        invariant!(height > 0);

        let metadata = Arc::new(ImageMetadata::new(&self.path, &self.state));
        let image = Arc::new(GPUIImage::from_bytes(
            ImageFormat::Bmp,
            encode_bmp(&self.state.image),
        ));
        let file_name = display_file_name(&self.path);

        LoadedImage {
            displayed: DisplayedImage { image, metadata },
            status: format!("{file_name} — {width} × {height}").into(),
        }
    }
}

fn image_decode_error(path: &Path) -> LoadError {
    LoadError::new(format!("Could not decode {}", path.display()))
}

fn display_file_name(path: &Path) -> Cow<'_, str> {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
}

fn display_parent(path: &Path) -> SharedString {
    let Some(parent) = path.parent() else {
        return SharedString::from(".");
    };

    let parent = parent.as_os_str().to_string_lossy();
    if parent.is_empty() {
        SharedString::from(".")
    } else {
        SharedString::from(parent.into_owned())
    }
}

fn format_jpeg_xr_samples(metadata: jpeg_xr::JPEGXRMetadata) -> String {
    let channels = match (metadata.color_channels(), metadata.has_alpha()) {
        (1, false) => "Gray",
        (1, true) => "GrayA",
        (3, false) => "RGB",
        (3, true) => "RGBA",
        _ => "Color",
    };
    format!("{channels} · {} bpc", metadata.bits_per_channel())
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    invariant!(left > 0);
    invariant!(right > 0);

    while right != 0 {
        (left, right) = (right, left % right);
    }

    invariant!(left > 0);
    left
}

fn format_file_size(bytes: u64) -> String {
    const KIBIBYTE: u64 = 1024;

    if bytes >= MEBIBYTE_BYTES {
        format_hundredths(bytes, MEBIBYTE_BYTES, "MiB")
    } else if bytes >= KIBIBYTE {
        format_hundredths(bytes, KIBIBYTE, "KiB")
    } else {
        format!("{bytes} B")
    }
}

fn format_pixel_count(pixels: u64) -> String {
    const MEGAPIXEL: u64 = 1_000_000;

    if pixels >= MEGAPIXEL {
        format_hundredths(pixels, MEGAPIXEL, "MP")
    } else {
        format!("{pixels} px")
    }
}

fn format_hundredths(value: u64, unit: u64, suffix: &str) -> String {
    invariant!(unit > 0);
    invariant!(!suffix.is_empty());

    let scaled = value
        .checked_mul(100)
        .and_then(|value| value.checked_add(unit / 2))
        .expect("bounded image metadata fits fixed-point display arithmetic")
        / unit;
    format!("{}.{:02} {suffix}", scaled / 100, scaled % 100)
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

pub(super) fn load_image(path: &Path) -> LoadResult<LoadedImage> {
    let loaded = ImageLoad::select(path).read()?.decode()?.present();
    invariant!(!loaded.status.is_empty());
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_error_preserves_the_decoder_error_frame() {
        let decoder_error = jpeg::decode([0x00]).unwrap_err();
        let load_error = decoder_error.raise(LoadError::new("Could not decode test.jpg"));
        let message = format_load_error(&load_error);

        assert!(message.contains("Could not decode test.jpg"));
        assert!(message.contains("JPEG codec error"));
        assert_eq!(load_error.frame().children().len(), 1);
        assert!(load_error.frame().children()[0].children().is_empty());
    }

    #[test]
    fn overlay_formats_bounded_file_and_pixel_sizes() {
        assert_eq!(format_file_size(999), "999 B");
        assert_eq!(format_file_size(14_669_660), "13.99 MiB");
        assert_eq!(format_pixel_count(3_686_400), "3.69 MP");
    }

    #[test]
    fn overlay_reduces_image_aspect_ratios() {
        let divisor = greatest_common_divisor(2560, 1440);

        assert_eq!((2560 / divisor, 1440 / divisor), (16, 9));
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
        assert_eq!(SourceFormat::detect(&[], Some("GIF")), SourceFormat::Gif);
        assert_eq!(SourceFormat::detect(&[], Some("HDP")), SourceFormat::JpegXr);
        assert_eq!(SourceFormat::detect(&[], Some("TIFF")), SourceFormat::Tiff);
        assert_eq!(
            SourceFormat::detect(&[], Some("unknown")),
            SourceFormat::Jpeg
        );
    }

    #[test]
    fn source_signature_takes_precedence_over_extension() {
        assert_eq!(
            SourceFormat::detect(&jpeg_xl::CODESTREAM_SIGNATURE, Some("png")),
            SourceFormat::JpegXl
        );
    }
}
