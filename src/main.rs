use std::fmt;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use exn::{ErrorExt, ResultExt};
use gpui::{
    App, Bounds, Image as GPUIImage, ImageFormat, MouseButton, MouseDownEvent, PathPromptOptions,
    Pixels, Point, SharedString, Window, WindowBounds, WindowControlArea, WindowOptions, deferred,
    div, img, prelude::*, px, rgb, rgba, size,
};
use gpui_platform::application;
use mimalloc::MiMalloc;

use four::{DecodedImage, encode_bmp, gif, jpeg, jpeg_xl, jpeg_xr, png, tiff};

// Keep invariant diagnostics consistent with the decoder crate while leaving test assertions free
// to use the standard macros, where custom panic messages add little value.
macro_rules! invariant {
    ($condition:expr, $($message:tt)+) => {
        assert!($condition, $($message)+)
    };
    ($condition:expr $(,)?) => {
        assert!(
            $condition,
            concat!("invariant failed: ", stringify!($condition))
        )
    };
}

macro_rules! invariant_eq {
    ($left:expr, $right:expr, $($message:tt)+) => {
        assert_eq!($left, $right, $($message)+)
    };
    ($left:expr, $right:expr $(,)?) => {
        assert_eq!(
            $left,
            $right,
            concat!(
                "invariant failed: ",
                stringify!($left),
                " == ",
                stringify!($right)
            )
        )
    };
}

#[global_allocator]
static GLOBAL_ALLOCATOR: MiMalloc = MiMalloc;

const CONTEXT_MENU_ITEM_HEIGHT: f32 = 36.0;
const CONTEXT_MENU_PADDING: f32 = 8.0;
const CONTEXT_MENU_WIDTH: f32 = 180.0;
const DRAG_REGION_HEIGHT: f32 = 40.0;
const ERROR_FRAMES_MAX: u32 = 8;
const IMAGE_FILE_BYTES_MAX: u64 = 128 * 1024 * 1024;
const METADATA_FIELD_GAP: f32 = 12.0;
const METADATA_LABEL_WIDTH: f32 = 112.0;
const METADATA_OVERLAY_WIDTH: f32 = 430.0;

type LoadException = exn::Exn<LoadError>;
type LoadResult<T> = exn::Result<T, LoadError>;

#[derive(Debug)]
struct LoadError {
    message: String,
}

impl LoadError {
    fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        invariant!(!message.is_empty());
        Self { message }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        invariant!(!self.message.is_empty());
        f.write_str(&self.message)
    }
}

impl std::error::Error for LoadError {}

#[track_caller]
fn load_error(message: impl Into<String>) -> LoadException {
    let error = LoadError::new(message);
    invariant!(!error.message.is_empty());
    error.raise()
}

fn format_load_error(error: &LoadException) -> String {
    invariant!(!error.to_string().is_empty());

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
struct DisplayedImage {
    image: Arc<GPUIImage>,
    metadata: Arc<ImageMetadata>,
}

struct LoadedImage {
    displayed: DisplayedImage,
    status: SharedString,
}

struct EmptyView {
    status: SharedString,
}

struct FailedView {
    status: SharedString,
}

struct LoadedFailedView {
    displayed: DisplayedImage,
    status: SharedString,
}

// GPUI entities keep one stable Rust type, so their dynamic states are an exhaustive sum type.
// Each variant owns exactly the payload valid in that state; independent flags cannot disagree.
enum ViewerState {
    Empty(EmptyView),
    Loaded(LoadedImage),
    Failed(FailedView),
    LoadedFailed(LoadedFailedView),
}

const _: () = invariant!(size_of::<ViewerState>() >= size_of::<SharedString>());

impl ViewerState {
    fn empty() -> Self {
        let state = Self::Empty(EmptyView {
            status: "Right-click to open an image".into(),
        });
        invariant!(!state.status().is_empty());
        invariant!(!state.has_image());
        state
    }

    fn from_result(result: LoadResult<LoadedImage>) -> Self {
        let state = match result {
            Ok(loaded) => Self::Loaded(loaded),
            Err(error) => Self::Failed(FailedView {
                status: format_load_error(&error).into(),
            }),
        };
        invariant!(!state.status().is_empty());
        invariant_eq!(state.has_image(), matches!(state, Self::Loaded(_)));
        state
    }

    fn apply_result(&mut self, result: LoadResult<LoadedImage>) {
        invariant!(!self.status().is_empty());
        invariant_eq!(self.image().is_some(), self.has_image());

        let previous_image = self.displayed().cloned();
        *self = match result {
            Ok(loaded) => Self::Loaded(loaded),
            Err(error) => match previous_image {
                Some(displayed) => Self::LoadedFailed(LoadedFailedView {
                    displayed,
                    status: format_load_error(&error).into(),
                }),
                None => Self::Failed(FailedView {
                    status: format_load_error(&error).into(),
                }),
            },
        };
        invariant!(!self.status().is_empty());
    }

    fn status(&self) -> &SharedString {
        let status = match self {
            Self::Empty(state) => &state.status,
            Self::Loaded(state) => &state.status,
            Self::Failed(state) => &state.status,
            Self::LoadedFailed(state) => &state.status,
        };
        invariant!(!status.is_empty());
        status
    }

    fn image(&self) -> Option<Arc<GPUIImage>> {
        invariant!(!self.status().is_empty());

        self.displayed()
            .map(|displayed| Arc::clone(&displayed.image))
    }

    fn metadata(&self) -> Option<Arc<ImageMetadata>> {
        invariant!(!self.status().is_empty());

        self.displayed()
            .map(|displayed| Arc::clone(&displayed.metadata))
    }

    fn displayed(&self) -> Option<&DisplayedImage> {
        match self {
            Self::Loaded(state) => Some(&state.displayed),
            Self::LoadedFailed(state) => Some(&state.displayed),
            Self::Empty(_) | Self::Failed(_) => None,
        }
    }

    fn has_image(&self) -> bool {
        invariant!(!self.status().is_empty());
        matches!(self, Self::Loaded(_) | Self::LoadedFailed(_))
    }
}

struct Root {
    context_menu_position: Option<Point<Pixels>>,
    metadata_visible: bool,
    viewer: ViewerState,
}

impl Root {
    fn new(viewer: ViewerState) -> Self {
        let root = Self {
            context_menu_position: None,
            metadata_visible: false,
            viewer,
        };
        invariant!(root.context_menu_position.is_none());
        invariant!(!root.metadata_visible);
        invariant!(!root.viewer.status().is_empty());
        root
    }

    fn show_context_menu(&mut self, event: &MouseDownEvent, window: &Window) {
        let mut position = event.position;
        let viewport_size = window.viewport_size();
        let menu_height = context_menu_height(self.viewer.has_image());
        position.x = position.x.min(viewport_size.width - px(CONTEXT_MENU_WIDTH));
        position.y = position.y.min(viewport_size.height - px(menu_height));
        position.x = position.x.max(px(0.0));
        position.y = position.y.max(px(DRAG_REGION_HEIGHT));
        invariant!(position.x >= px(0.0));
        invariant!(position.y >= px(DRAG_REGION_HEIGHT));
        self.context_menu_position = Some(position);
    }

    fn open_image(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        invariant!(!self.viewer.status().is_empty());

        self.context_menu_position = None;
        invariant!(self.context_menu_position.is_none());
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open an image".into()),
        });
        cx.spawn_in(window, async move |root, cx| {
            let path = match paths.await {
                Ok(Ok(Some(mut paths))) => paths.pop(),
                _ => None,
            };
            let Some(path) = path else {
                return;
            };
            let result = cx.background_spawn(async move { load_image(&path) }).await;
            let _ = root.update_in(cx, |root, _window, cx| {
                root.apply_load_result(result);
                cx.notify();
            });
        })
        .detach();
    }

    fn apply_load_result(&mut self, result: LoadResult<LoadedImage>) {
        invariant!(!self.viewer.status().is_empty());
        invariant!(self.context_menu_position.is_none());

        let load_succeeded = result.is_ok();
        self.viewer.apply_result(result);
        if load_succeeded {
            self.metadata_visible = false;
        }
        invariant!(!self.viewer.status().is_empty());
    }

    fn render_context_menu(
        position: Point<Pixels>,
        has_image: bool,
        metadata_visible: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        invariant!(position.x >= px(0.0));
        invariant!(position.y >= px(DRAG_REGION_HEIGHT));

        deferred(
            div()
                .absolute()
                .left(position.x)
                .top(position.y)
                .w(px(CONTEXT_MENU_WIDTH))
                .h(px(context_menu_height(has_image)))
                .p_1()
                .rounded_md()
                .shadow_lg()
                .border_1()
                .border_color(rgb(0x0045_4545))
                .bg(rgb(0x0029_2929))
                .flex()
                .flex_col()
                .on_mouse_down_out(cx.listener(|root, _, _, cx| {
                    root.context_menu_position = None;
                    cx.notify();
                }))
                .child(
                    menu_item("open-image", "Open image…")
                        .on_click(cx.listener(|root, _, window, cx| root.open_image(window, cx))),
                )
                .when(has_image, |menu| {
                    let label = if metadata_visible {
                        "Hide image info"
                    } else {
                        "Show image info"
                    };
                    menu.child(menu_item("toggle-image-info", label).on_click(cx.listener(
                        |root, _, _, cx| {
                            root.context_menu_position = None;
                            root.metadata_visible = !root.metadata_visible;
                            cx.notify();
                        },
                    )))
                })
                .child(menu_item("quit", "Quit").on_click(|_, _, cx| cx.quit())),
        )
        .priority(1)
    }

    fn render_metadata_overlay(metadata: &ImageMetadata) -> gpui::Div {
        invariant!(!metadata.fields.is_empty());

        let mut overlay = div()
            .absolute()
            .left(px(12.0))
            .top(px(DRAG_REGION_HEIGHT + 12.0))
            .w(px(METADATA_OVERLAY_WIDTH))
            .p_3()
            .rounded_md()
            .border_1()
            .border_color(rgba(0xff_ff_ff_22))
            .shadow_lg()
            .bg(rgba(0x0d_0d_0d_e8))
            .font_family("Consolas")
            .text_sm()
            .flex()
            .flex_col();

        for field in &metadata.fields {
            overlay = overlay.child(metadata_field(field));
        }

        overlay
    }

    fn render_image_content(image: Option<Arc<GPUIImage>>, has_image: bool) -> gpui::Div {
        invariant_eq!(image.is_some(), has_image);

        div()
            .flex_1()
            .min_h_0()
            .min_w_0()
            .flex()
            .items_center()
            .justify_center()
            .overflow_hidden()
            .when_some(image, |content, image| {
                content.child(img(image).size_full())
            })
            .when(!has_image, |content| {
                content.child(
                    div()
                        .text_sm()
                        .text_color(rgb(0x0088_8888))
                        .child("Right-click anywhere, then choose Open image…"),
                )
            })
    }
}

impl Render for Root {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        invariant!(!self.viewer.status().is_empty());
        invariant!(
            self.context_menu_position
                .is_none_or(|position| position.x >= px(0.0))
        );

        let context_menu_position = self.context_menu_position;
        let image = self.viewer.image();
        let has_image = self.viewer.has_image();
        let metadata = self.viewer.metadata();
        let metadata_visible = self.metadata_visible;
        let status = self.viewer.status().clone();
        invariant_eq!(image.is_some(), has_image);
        invariant_eq!(metadata.is_some(), has_image);
        div()
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0x0015_1515))
            .text_color(rgb(0x00d8_d8d8))
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|root, event: &MouseDownEvent, window, cx| {
                    root.show_context_menu(event, window);
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w_full()
                    .h(px(DRAG_REGION_HEIGHT))
                    .flex_none()
                    .flex()
                    .items_center()
                    .px_3()
                    .text_sm()
                    .bg(rgb(0x0020_2020))
                    .window_control_area(WindowControlArea::Drag)
                    .child(status),
            )
            .child(Self::render_image_content(image, has_image))
            .when_some(metadata.filter(|_| metadata_visible), |root, metadata| {
                root.child(Self::render_metadata_overlay(&metadata))
            })
            .when_some(context_menu_position, |root, position| {
                root.child(Self::render_context_menu(
                    position,
                    has_image,
                    metadata_visible,
                    cx,
                ))
            })
    }
}

const fn context_menu_height(has_image: bool) -> f32 {
    let item_count = if has_image { 3.0 } else { 2.0 };
    CONTEXT_MENU_PADDING + CONTEXT_MENU_ITEM_HEIGHT * item_count
}

fn menu_item(identifier: &'static str, label: &'static str) -> gpui::Stateful<gpui::Div> {
    invariant!(!identifier.is_empty());
    invariant!(!label.is_empty());

    div()
        .id(identifier)
        .h(px(36.0))
        .w_full()
        .flex()
        .items_center()
        .px_2()
        .rounded_sm()
        .cursor_pointer()
        .text_sm()
        .text_color(rgb(0x00ff_ffff))
        .hover(|style| style.bg(rgb(0x003d_3d3d)))
        .child(label)
}

fn metadata_field(field: &MetadataField) -> gpui::Div {
    invariant!(!field.label.is_empty());
    invariant!(!field.value.is_empty());

    div()
        .w_full()
        .flex()
        .items_start()
        .gap(px(METADATA_FIELD_GAP))
        .py_0p5()
        .child(
            div()
                .w(px(METADATA_LABEL_WIDTH))
                .flex_none()
                .text_color(rgb(0x009d_9d9d))
                .child(field.label),
        )
        .child(
            div()
                .min_w_0()
                .flex_1()
                .text_color(rgb(0x00e8_e8e8))
                .child(field.value.clone()),
        )
}

#[derive(Debug)]
struct ImageMetadata {
    fields: Vec<MetadataField>,
}

impl ImageMetadata {
    fn new(path: &Path, decoded: &DecodedImageState) -> Self {
        let (width, height) = decoded.image.dimensions();
        invariant!(width > 0);
        invariant!(height > 0);
        invariant!(decoded.byte_count <= IMAGE_FILE_BYTES_MAX);

        let file_name = path.file_name().map_or_else(
            || path.as_os_str().to_string_lossy(),
            |name| name.to_string_lossy(),
        );
        let folder = path.parent().map_or_else(
            || SharedString::from("."),
            |parent| {
                let parent = parent.as_os_str().to_string_lossy();
                if parent.is_empty() {
                    SharedString::from(".")
                } else {
                    SharedString::from(parent.into_owned())
                }
            },
        );
        let divisor = greatest_common_divisor(width, height);
        let pixel_count = u64::from(width) * u64::from(height);
        let (pixels, remainder) = decoded.image.rgba8().as_chunks::<4>();
        invariant!(remainder.is_empty());
        let transparency = pixels.iter().any(|pixel| pixel[3] != u8::MAX);
        let mut fields = vec![
            MetadataField::new("Image", file_name.into_owned()),
            MetadataField::new("Folder", folder),
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
struct MetadataField {
    label: &'static str,
    value: SharedString,
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
            Self::JpegXl
        } else if jpeg_xr::has_signature(bytes) {
            Self::JpegXr
        } else if png::has_signature(bytes) {
            Self::Png
        } else if gif::has_signature(bytes) {
            Self::Gif
        } else if tiff::has_signature(bytes) {
            Self::Tiff
        } else if bytes.starts_with(&jpeg::SIGNATURE) {
            Self::Jpeg
        } else if extension.is_some_and(|value| value.eq_ignore_ascii_case("jxl")) {
            Self::JpegXl
        } else if extension.is_some_and(|value| {
            value.eq_ignore_ascii_case("jxr")
                || value.eq_ignore_ascii_case("wdp")
                || value.eq_ignore_ascii_case("hdp")
        }) {
            Self::JpegXr
        } else if extension.is_some_and(|value| value.eq_ignore_ascii_case("png")) {
            Self::Png
        } else if extension.is_some_and(|value| value.eq_ignore_ascii_case("gif")) {
            Self::Gif
        } else if extension.is_some_and(|value| {
            value.eq_ignore_ascii_case("tif") || value.eq_ignore_ascii_case("tiff")
        }) {
            Self::Tiff
        } else {
            Self::Jpeg
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

fn image_decode_error(path: &Path) -> LoadError {
    LoadError::new(format!("Could not decode {}", path.display()))
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
    const MEBIBYTE: u64 = 1024 * KIBIBYTE;

    if bytes >= MEBIBYTE {
        format_hundredths(bytes, MEBIBYTE, "MiB")
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

// Loading is a linear pipeline: only read bytes can be decoded, and only decoded pixels can be
// presented. Consuming transitions prevent callers from skipping or repeating a phase.
struct ImageLoad<State> {
    path: PathBuf,
    state: State,
}

struct SelectedImage;

const _: () = invariant!(size_of::<SelectedImage>() == 0);

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
        let load = Self {
            path: path.to_path_buf(),
            state: SelectedImage,
        };
        invariant_eq!(load.path.as_os_str(), path.as_os_str());
        load
    }

    fn read(self) -> LoadResult<ImageLoad<EncodedImage>> {
        invariant_eq!(size_of_val(&self.state), 0);

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
        let file_name = self.path.file_name().map_or_else(
            || self.path.as_os_str().to_string_lossy(),
            |name| name.to_string_lossy(),
        );
        LoadedImage {
            displayed: DisplayedImage { image, metadata },
            status: format!("{file_name} — {width} × {height}").into(),
        }
    }
}

fn validate_image_file_size(path: &Path, byte_count: u64) -> LoadResult<()> {
    if byte_count > IMAGE_FILE_BYTES_MAX {
        return Err(load_error(format!(
            "{} is larger than the 128 MiB input limit",
            path.display()
        )));
    }
    Ok(())
}

fn load_image(path: &Path) -> LoadResult<LoadedImage> {
    let loaded = ImageLoad::select(path).read()?.decode()?.present();
    invariant!(!loaded.status.is_empty());
    invariant!(Arc::strong_count(&loaded.displayed.image) > 0);
    invariant!(Arc::strong_count(&loaded.displayed.metadata) > 0);
    Ok(loaded)
}

fn initial_viewer(path: Option<&Path>) -> ViewerState {
    let viewer = match path {
        Some(path) => ViewerState::from_result(load_image(path)),
        None => ViewerState::empty(),
    };
    invariant!(!viewer.status().is_empty());
    invariant_eq!(viewer.image().is_some(), viewer.has_image());
    viewer
}

fn main() {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);
    let initial_viewer = initial_viewer(initial_path.as_deref());
    application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.0), px(600.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: None,
                is_movable: true,
                window_min_size: Some(size(
                    px(METADATA_OVERLAY_WIDTH + 24.0),
                    px(DRAG_REGION_HEIGHT + context_menu_height(true)),
                )),
                ..Default::default()
            },
            |_window, cx| cx.new(|_| Root::new(initial_viewer)),
        )
        .expect("failed to open window");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_error_preserves_the_decoder_error_frame() {
        let decoder_error = jpeg::decode([0x00]).unwrap_err();
        let load_error = decoder_error.raise(LoadError::new("Could not decode test.jpg"));
        let message = format_load_error(&load_error);

        invariant!(message.contains("Could not decode test.jpg"));
        invariant!(message.contains("JPEG codec error"));
        invariant_eq!(load_error.frame().children().len(), 1);
        invariant!(load_error.frame().children()[0].children().is_empty());
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
        assert!(context_menu_height(true) > context_menu_height(false));
    }

    #[test]
    fn image_information_is_hidden_by_default() {
        let root = Root::new(ViewerState::empty());

        assert!(!root.metadata_visible);
    }
}
