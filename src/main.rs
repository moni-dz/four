#![feature(portable_simd)]

mod image_formats;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpui::{
    App, Bounds, Context, Image as GpuiImage, ImageFormat, MouseButton, MouseDownEvent,
    PathPromptOptions, Pixels, Point, SharedString, Window, WindowBounds, WindowControlArea,
    WindowOptions, deferred, div, img, prelude::*, px, rgb, size,
};
use gpui_platform::application;

use image_formats::{Image as ImageData, encode_bmp, jpeg};

const CONTEXT_MENU_HEIGHT: f32 = 80.0;
const CONTEXT_MENU_WIDTH: f32 = 180.0;
const DRAG_REGION_HEIGHT: f32 = 40.0;
const JPEG_FILE_BYTES_MAX: u64 = 128 * 1024 * 1024;

// The menu has two rows and the drag strip remains reachable at the minimum window size.
const _: () = {
    assert!(CONTEXT_MENU_HEIGHT >= 2.0 * DRAG_REGION_HEIGHT);
    assert!(CONTEXT_MENU_WIDTH > 0.0);
    assert!(DRAG_REGION_HEIGHT > 0.0);
    assert!(JPEG_FILE_BYTES_MAX > 0);
    assert!(JPEG_FILE_BYTES_MAX <= usize::MAX as u64);
};

struct LoadedImage {
    image: Arc<GpuiImage>,
    status: SharedString,
}

struct Root {
    context_menu_position: Option<Point<Pixels>>,
    image: Option<Arc<GpuiImage>>,
    status: SharedString,
}

impl Root {
    fn new(initial_image: Option<Result<LoadedImage, String>>) -> Self {
        let root = match initial_image {
            Some(Ok(loaded)) => Self {
                context_menu_position: None,
                image: Some(loaded.image),
                status: loaded.status,
            },
            Some(Err(error)) => Self {
                context_menu_position: None,
                image: None,
                status: error.into(),
            },
            None => Self {
                context_menu_position: None,
                image: None,
                status: "Right-click to open a baseline JPEG".into(),
            },
        };
        assert!(root.context_menu_position.is_none());
        assert!(!root.status.is_empty());
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
        assert!(!self.status.is_empty());

        self.context_menu_position = None;
        assert!(self.context_menu_position.is_none());
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open a baseline JPEG".into()),
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

    fn apply_load_result(&mut self, result: Result<LoadedImage, String>) {
        assert!(!self.status.is_empty());
        assert!(self.context_menu_position.is_none());

        match result {
            Ok(loaded) => {
                self.image = Some(loaded.image);
                self.status = loaded.status;
            }
            Err(error) => {
                self.status = error.into();
            }
        }
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
        assert!(!self.status.is_empty());
        assert!(
            self.context_menu_position
                .is_none_or(|position| position.x >= px(0.0))
        );

        let context_menu_position = self.context_menu_position;
        let image = self.image.clone();
        let status = self.status.clone();
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
                    .when(self.image.is_none(), |content| {
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

fn load_jpeg(path: &Path) -> Result<LoadedImage, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("Could not inspect {}: {error}", path.display()))?;
    if metadata.len() > JPEG_FILE_BYTES_MAX {
        return Err(format!(
            "{} is larger than the 128 MiB input limit",
            path.display()
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("Could not read {}: {error}", path.display()))?;
    let decoded = jpeg::decode(&bytes)
        .map_err(|error| format!("Could not decode {}: {error}", path.display()))?;
    let (width, height) = decoded.dimensions();
    let image = Arc::new(GpuiImage::from_bytes(
        ImageFormat::Bmp,
        encode_bmp(&decoded),
    ));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| path.as_os_str().to_string_lossy());
    Ok(LoadedImage {
        image,
        status: format!("{file_name} — {width} × {height}").into(),
    })
}

fn main() {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);
    let initial_image = initial_path.as_deref().map(load_jpeg);
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
            |_window, cx| cx.new(|_| Root::new(initial_image)),
        )
        .expect("failed to open window");
    });
}
