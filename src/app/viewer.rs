//! Owns viewer state and renders the GPUI interface.

use std::path::Path;
use std::sync::Arc;

use gpui::{
    Image as GPUIImage, MouseButton, MouseDownEvent, PathPromptOptions, Pixels, Point,
    SharedString, Window, WindowControlArea, deferred, div, img, prelude::*, px, rgb, rgba,
};

use super::image_loader::{
    DisplayedImage, ImageMetadata, LoadResult, LoadedImage, MetadataField, format_load_error,
    load_image,
};

const CONTEXT_MENU_ITEM_HEIGHT: f32 = 36.0;
const CONTEXT_MENU_PADDING: f32 = 8.0;
const CONTEXT_MENU_WIDTH: f32 = 180.0;
const DRAG_REGION_HEIGHT: f32 = 40.0;
const METADATA_FIELD_GAP: f32 = 12.0;
const METADATA_LABEL_WIDTH: f32 = 140.0;
const METADATA_OVERLAY_MARGIN: f32 = 12.0;
const METADATA_OVERLAY_WIDTH: f32 = 480.0;
const METADATA_WINDOW_MIN_HEIGHT: f32 = 500.0;

pub(super) const WINDOW_MIN_WIDTH: f32 = METADATA_OVERLAY_WIDTH + 2.0 * METADATA_OVERLAY_MARGIN;
pub(super) const WINDOW_MIN_HEIGHT: f32 = METADATA_WINDOW_MIN_HEIGHT;

pub(super) enum ViewerState {
    Empty {
        status: SharedString,
    },
    Loaded(LoadedImage),
    Failed {
        status: SharedString,
    },
    LoadedFailed {
        displayed: DisplayedImage,
        status: SharedString,
    },
}

impl ViewerState {
    fn empty() -> Self {
        Self::Empty {
            status: "Right-click to open an image".into(),
        }
    }

    fn from_result(result: LoadResult<LoadedImage>) -> Self {
        match result {
            Ok(loaded) => Self::Loaded(loaded),
            Err(error) => Self::Failed {
                status: format_load_error(&error).into(),
            },
        }
    }

    fn apply_result(&mut self, result: LoadResult<LoadedImage>) {
        invariant!(!self.status().is_empty());

        let previous_image = self.displayed().cloned();
        *self = match result {
            Ok(loaded) => Self::Loaded(loaded),
            Err(error) => match previous_image {
                Some(displayed) => Self::LoadedFailed {
                    displayed,
                    status: format_load_error(&error).into(),
                },
                None => Self::Failed {
                    status: format_load_error(&error).into(),
                },
            },
        };

        invariant!(!self.status().is_empty());
    }

    fn status(&self) -> &SharedString {
        let status = match self {
            Self::Empty { status }
            | Self::Failed { status }
            | Self::LoadedFailed { status, .. } => status,
            Self::Loaded(state) => &state.status,
        };

        invariant!(!status.is_empty());
        status
    }

    fn displayed(&self) -> Option<&DisplayedImage> {
        match self {
            Self::Loaded(state) => Some(&state.displayed),
            Self::LoadedFailed { displayed, .. } => Some(displayed),
            Self::Empty { .. } | Self::Failed { .. } => None,
        }
    }

    fn has_image(&self) -> bool {
        self.displayed().is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LoadRequest(u64);

pub(super) struct Root {
    context_menu_position: Option<Point<Pixels>>,
    load_generation: u64,
    metadata_visible: bool,
    viewer: ViewerState,
}

impl Root {
    pub(super) fn new(viewer: ViewerState) -> Self {
        Self {
            context_menu_position: None,
            load_generation: 0,
            metadata_visible: false,
            viewer,
        }
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

    fn begin_load_request(&mut self) -> LoadRequest {
        self.load_generation = self
            .load_generation
            .checked_add(1)
            .expect("image load request generation overflowed");

        LoadRequest(self.load_generation)
    }

    fn accepts_load_request(&self, request: LoadRequest) -> bool {
        request.0 == self.load_generation
    }

    fn open_image(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        invariant!(!self.viewer.status().is_empty());

        self.context_menu_position = None;
        let request = self.begin_load_request();
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
                if root.apply_load_result(request, result) {
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn apply_load_result(&mut self, request: LoadRequest, result: LoadResult<LoadedImage>) -> bool {
        if !self.accepts_load_request(request) {
            return false;
        }

        invariant!(!self.viewer.status().is_empty());
        invariant!(self.context_menu_position.is_none());

        let load_succeeded = result.is_ok();
        self.viewer.apply_result(result);
        if load_succeeded {
            self.metadata_visible = false;
        }

        invariant!(!self.viewer.status().is_empty());
        true
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

        div()
            .absolute()
            .left(px(METADATA_OVERLAY_MARGIN))
            .top(px(DRAG_REGION_HEIGHT + METADATA_OVERLAY_MARGIN))
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
            .flex_col()
            .children(metadata.fields.iter().map(metadata_field))
    }

    fn render_image_content(image: Option<Arc<GPUIImage>>) -> gpui::Div {
        let has_image = image.is_some();

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

    fn render_status_bar(status: SharedString) -> gpui::Div {
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
            .child(status)
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
        let displayed = self.viewer.displayed().cloned();
        let has_image = displayed.is_some();
        let image = displayed
            .as_ref()
            .map(|displayed| Arc::clone(&displayed.image));
        let metadata = displayed
            .as_ref()
            .map(|displayed| Arc::clone(&displayed.metadata));
        let metadata_visible = self.metadata_visible;
        let status = self.viewer.status().clone();

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
            .child(Self::render_status_bar(status))
            .child(Self::render_image_content(image))
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
        .h(px(CONTEXT_MENU_ITEM_HEIGHT))
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
        .when(field.starts_section, gpui::Styled::mt_2)
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

pub(super) fn initial_viewer(path: Option<&Path>) -> ViewerState {
    let viewer = match path {
        Some(path) => ViewerState::from_result(load_image(path)),
        None => ViewerState::empty(),
    };

    invariant!(!viewer.status().is_empty());
    viewer
}

#[cfg(test)]
mod tests {
    use exn::ErrorExt as _;

    use super::*;
    use crate::app::image_loader::LoadError;

    #[test]
    fn image_information_is_hidden_by_default() {
        let root = Root::new(ViewerState::empty());

        assert!(!root.metadata_visible);
    }

    #[test]
    fn context_menu_adds_an_item_for_a_loaded_image() {
        assert!(context_menu_height(true) > context_menu_height(false));
    }

    #[test]
    fn stale_load_result_cannot_replace_newer_request() {
        let mut root = Root::new(ViewerState::empty());
        let first = root.begin_load_request();
        let second = root.begin_load_request();

        let stale_error = LoadError::new("stale load failed").raise();
        assert!(!root.apply_load_result(first, Err(stale_error)));
        assert!(matches!(root.viewer, ViewerState::Empty { .. }));

        let current_error = LoadError::new("current load failed").raise();
        assert!(root.apply_load_result(second, Err(current_error)));
        assert!(matches!(root.viewer, ViewerState::Failed { .. }));
    }
}
