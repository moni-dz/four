#![feature(portable_simd)]

mod image_formats;

use std::fmt;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use exn::{ErrorExt, ResultExt};
use gpui::{
    App, Bounds, Context, Image as GPUIImage, ImageFormat, MouseButton, MouseDownEvent,
    PathPromptOptions, Pixels, Point, SharedString, Window, WindowBounds, WindowControlArea,
    WindowOptions, deferred, div, img, prelude::*, px, rgb, size,
};
use gpui_platform::application;

use image_formats::{DecodedImage, Image as ImageData, encode_bmp, jpeg};

const CONTEXT_MENU_HEIGHT: f32 = 80.0;
const CONTEXT_MENU_WIDTH: f32 = 180.0;
const DRAG_REGION_HEIGHT: f32 = 40.0;
const ERROR_FRAMES_MAX: u32 = 8;
const JPEG_FILE_BYTES_MAX: u64 = 128 * 1024 * 1024;

// The menu has two rows and the drag strip remains reachable at the minimum window size.
const _: () = {
    assert!(CONTEXT_MENU_HEIGHT >= 2.0 * DRAG_REGION_HEIGHT);
    assert!(CONTEXT_MENU_WIDTH > 0.0);
    assert!(DRAG_REGION_HEIGHT > 0.0);
    assert!(ERROR_FRAMES_MAX > 0);
    assert!(JPEG_FILE_BYTES_MAX > 0);
    assert!(JPEG_FILE_BYTES_MAX <= usize::MAX as u64);
};

type LoadException = exn::Exn<LoadError>;
type LoadResult<T> = exn::Result<T, LoadError>;

#[derive(Debug)]
struct LoadError {
    message: String,
}

impl LoadError {
    fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!message.is_empty());
        Self { message }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(!self.message.is_empty());
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for LoadError {}

#[track_caller]
fn load_error(message: impl Into<String>) -> LoadException {
    let error = LoadError::new(message);
    assert!(!error.message.is_empty());
    error.raise()
}

fn format_load_error(error: &LoadException) -> String {
    assert!(!error.to_string().is_empty());

    let mut frame = error.frame();
    let mut message = frame.error().to_string();
    for _ in 0..ERROR_FRAMES_MAX {
        let Some(child) = frame.children().first() else {
            assert!(!message.is_empty());
            return message;
        };
        write!(&mut message, ": {}", child.error()).expect("writing to a string cannot fail");
        frame = child;
    }
    if !frame.children().is_empty() {
        message.push_str(": additional error context omitted");
    }
    assert!(!message.is_empty());
    message
}

struct LoadedImage {
    image: Arc<GPUIImage>,
    status: SharedString,
}

struct EmptyView {
    status: SharedString,
}

struct FailedView {
    status: SharedString,
}

struct LoadedFailedView {
    image: Arc<GPUIImage>,
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

const _: () = assert!(size_of::<ViewerState>() >= size_of::<SharedString>());

impl ViewerState {
    fn empty() -> Self {
        let state = Self::Empty(EmptyView {
            status: "Right-click to open a JPEG".into(),
        });
        assert!(!state.status().is_empty());
        assert!(!state.has_image());
        state
    }

    fn from_result(result: LoadResult<LoadedImage>) -> Self {
        let state = match result {
            Ok(loaded) => Self::Loaded(loaded),
            Err(error) => Self::Failed(FailedView {
                status: format_load_error(&error).into(),
            }),
        };
        assert!(!state.status().is_empty());
        assert_eq!(state.has_image(), matches!(state, Self::Loaded(_)));
        state
    }

    fn apply_result(&mut self, result: LoadResult<LoadedImage>) {
        assert!(!self.status().is_empty());
        assert_eq!(self.image().is_some(), self.has_image());

        let previous_image = self.image();
        *self = match result {
            Ok(loaded) => Self::Loaded(loaded),
            Err(error) => match previous_image {
                Some(image) => Self::LoadedFailed(LoadedFailedView {
                    image,
                    status: format_load_error(&error).into(),
                }),
                None => Self::Failed(FailedView {
                    status: format_load_error(&error).into(),
                }),
            },
        };
        assert!(!self.status().is_empty());
    }

    fn status(&self) -> &SharedString {
        let status = match self {
            Self::Empty(state) => &state.status,
            Self::Loaded(state) => &state.status,
            Self::Failed(state) => &state.status,
            Self::LoadedFailed(state) => &state.status,
        };
        assert!(!status.is_empty());
        status
    }

    fn image(&self) -> Option<Arc<GPUIImage>> {
        assert!(!self.status().is_empty());

        match self {
            Self::Loaded(state) => Some(state.image.clone()),
            Self::LoadedFailed(state) => Some(state.image.clone()),
            Self::Empty(_) | Self::Failed(_) => None,
        }
    }

    fn has_image(&self) -> bool {
        assert!(!self.status().is_empty());
        matches!(self, Self::Loaded(_) | Self::LoadedFailed(_))
    }
}

struct Root {
    context_menu_position: Option<Point<Pixels>>,
    viewer: ViewerState,
}

impl Root {
    fn new(viewer: ViewerState) -> Self {
        let root = Self {
            context_menu_position: None,
            viewer,
        };
        assert!(root.context_menu_position.is_none());
        assert!(!root.viewer.status().is_empty());
        root
    }

    fn show_context_menu(&mut self, event: &MouseDownEvent, window: &Window) {
        let mut position = event.position;
        let viewport_size = window.viewport_size();
        position.x = position.x.min(viewport_size.width - px(CONTEXT_MENU_WIDTH));
        position.y = position
            .y
            .min(viewport_size.height - px(CONTEXT_MENU_HEIGHT));
        position.x = position.x.max(px(0.0));
        position.y = position.y.max(px(DRAG_REGION_HEIGHT));
        assert!(position.x >= px(0.0));
        assert!(position.y >= px(DRAG_REGION_HEIGHT));
        self.context_menu_position = Some(position);
    }

    fn open_jpeg(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        assert!(!self.viewer.status().is_empty());

        self.context_menu_position = None;
        assert!(self.context_menu_position.is_none());
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open a JPEG".into()),
        });
        cx.spawn_in(window, async move |root, cx| {
            let path = match paths.await {
                Ok(Ok(Some(mut paths))) => paths.pop(),
                _ => None,
            };
            let Some(path) = path else {
                return;
            };
            let result = cx.background_spawn(async move { load_jpeg(&path) }).await;
            let _ = root.update_in(cx, |root, _window, cx| {
                root.apply_load_result(result);
                cx.notify();
            });
        })
        .detach();
    }

    fn apply_load_result(&mut self, result: LoadResult<LoadedImage>) {
        assert!(!self.viewer.status().is_empty());
        assert!(self.context_menu_position.is_none());

        self.viewer.apply_result(result);
        assert!(!self.viewer.status().is_empty());
    }

    fn render_context_menu(
        &self,
        position: Point<Pixels>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        assert!(position.x >= px(0.0));
        assert!(position.y >= px(DRAG_REGION_HEIGHT));

        deferred(
            div()
                .absolute()
                .left(position.x)
                .top(position.y)
                .w(px(CONTEXT_MENU_WIDTH))
                .h(px(CONTEXT_MENU_HEIGHT))
                .p_1()
                .rounded_md()
                .shadow_lg()
                .border_1()
                .border_color(rgb(0x454545))
                .bg(rgb(0x292929))
                .flex()
                .flex_col()
                .on_mouse_down_out(cx.listener(|root, _, _, cx| {
                    root.context_menu_position = None;
                    cx.notify();
                }))
                .child(
                    menu_item("open-jpeg", "Open JPEG…")
                        .on_click(cx.listener(|root, _, window, cx| root.open_jpeg(window, cx))),
                )
                .child(menu_item("quit", "Quit").on_click(|_, _, cx| cx.quit())),
        )
        .priority(1)
    }
}

impl Render for Root {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        assert!(!self.viewer.status().is_empty());
        assert!(
            self.context_menu_position
                .is_none_or(|position| position.x >= px(0.0))
        );

        let context_menu_position = self.context_menu_position;
        let image = self.viewer.image();
        let has_image = self.viewer.has_image();
        let status = self.viewer.status().clone();
        assert_eq!(image.is_some(), has_image);
        div()
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0x151515))
            .text_color(rgb(0xd8d8d8))
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
                    .bg(rgb(0x202020))
                    .window_control_area(WindowControlArea::Drag)
                    .child(status),
            )
            .child(
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
                                .text_color(rgb(0x888888))
                                .child("Right-click anywhere, then choose Open JPEG…"),
                        )
                    }),
            )
            .when_some(context_menu_position, |root, position| {
                root.child(self.render_context_menu(position, cx))
            })
    }
}

fn menu_item(identifier: &'static str, label: &'static str) -> gpui::Stateful<gpui::Div> {
    assert!(!identifier.is_empty());
    assert!(!label.is_empty());

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
        .text_color(rgb(0xffffff))
        .hover(|style| style.bg(rgb(0x3d3d3d)))
        .child(label)
}

// Loading is a linear pipeline: only read bytes can be decoded, and only decoded pixels can be
// presented. Consuming transitions prevent callers from skipping or repeating a phase.
struct ImageLoad<State> {
    path: PathBuf,
    state: State,
}

struct SelectedJPEG;

const _: () = assert!(size_of::<SelectedJPEG>() == 0);

struct EncodedJPEG {
    bytes: Vec<u8>,
}

struct DecodedJPEG {
    image: DecodedImage,
}

impl ImageLoad<SelectedJPEG> {
    fn select(path: &Path) -> Self {
        let load = Self {
            path: path.to_path_buf(),
            state: SelectedJPEG,
        };
        assert_eq!(load.path.as_os_str(), path.as_os_str());
        load
    }

    fn read(self) -> LoadResult<ImageLoad<EncodedJPEG>> {
        assert_eq!(size_of_val(&self.state), 0);

        let file = File::open(&self.path)
            .or_raise(|| LoadError::new(format!("Could not open {}", self.path.display())))?;
        let metadata = file
            .metadata()
            .or_raise(|| LoadError::new(format!("Could not inspect {}", self.path.display())))?;
        validate_jpeg_file_size(&self.path, metadata.len())?;
        assert!(metadata.len() <= JPEG_FILE_BYTES_MAX);

        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(JPEG_FILE_BYTES_MAX + 1)
            .read_to_end(&mut bytes)
            .or_raise(|| LoadError::new(format!("Could not read {}", self.path.display())))?;
        // The bounded read catches files that grow after metadata without allocating unboundedly.
        validate_jpeg_file_size(&self.path, bytes.len() as u64)?;
        Ok(ImageLoad {
            path: self.path,
            state: EncodedJPEG { bytes },
        })
    }
}

impl ImageLoad<EncodedJPEG> {
    fn decode(self) -> LoadResult<ImageLoad<DecodedJPEG>> {
        assert!(self.state.bytes.len() as u64 <= JPEG_FILE_BYTES_MAX);
        assert!(self.state.bytes.len() <= isize::MAX as usize);

        let image = jpeg::decode(&self.state.bytes)
            .or_raise(|| LoadError::new(format!("Could not decode {}", self.path.display())))?;
        assert!(image.width() > 0);
        assert!(image.height() > 0);
        Ok(ImageLoad {
            path: self.path,
            state: DecodedJPEG { image },
        })
    }
}

impl ImageLoad<DecodedJPEG> {
    fn present(self) -> LoadedImage {
        let (width, height) = self.state.image.dimensions();
        assert!(width > 0);
        assert!(height > 0);

        let image = Arc::new(GPUIImage::from_bytes(
            ImageFormat::Bmp,
            encode_bmp(&self.state.image),
        ));
        let file_name = self
            .path
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| self.path.as_os_str().to_string_lossy());
        LoadedImage {
            image,
            status: format!("{file_name} — {width} × {height}").into(),
        }
    }
}

fn validate_jpeg_file_size(path: &Path, byte_count: u64) -> LoadResult<()> {
    if byte_count > JPEG_FILE_BYTES_MAX {
        return Err(load_error(format!(
            "{} is larger than the 128 MiB input limit",
            path.display()
        )));
    }
    Ok(())
}

fn load_jpeg(path: &Path) -> LoadResult<LoadedImage> {
    let loaded = ImageLoad::select(path).read()?.decode()?.present();
    assert!(!loaded.status.is_empty());
    assert!(Arc::strong_count(&loaded.image) > 0);
    Ok(loaded)
}

fn initial_viewer(path: Option<&Path>) -> ViewerState {
    let viewer = match path {
        Some(path) => ViewerState::from_result(load_jpeg(path)),
        None => ViewerState::empty(),
    };
    assert!(!viewer.status().is_empty());
    assert_eq!(viewer.image().is_some(), viewer.has_image());
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
                    px(CONTEXT_MENU_WIDTH),
                    px(DRAG_REGION_HEIGHT + CONTEXT_MENU_HEIGHT),
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
        let decoder_error = jpeg::decode(&[0x00]).unwrap_err();
        let load_error = decoder_error.raise(LoadError::new("Could not decode test.jpg"));
        let message = format_load_error(&load_error);

        assert!(message.contains("Could not decode test.jpg"));
        assert!(message.contains("expected a JPEG marker"));
        assert_eq!(load_error.frame().children().len(), 1);
        assert!(load_error.frame().children()[0].children().is_empty());
    }
}
