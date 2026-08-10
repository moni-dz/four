//! Owns viewer state and renders the GPUI interface.

use std::path::Path;
use std::sync::Arc;

use gpui::{
    Anchor, AnchoredPositionMode, Image as GPUIImage, MouseButton, MouseDownEvent,
    PathPromptOptions, Pixels, Point, Role, SharedString, Toggled, Window, WindowControlArea,
    anchored, deferred, div, img, point, prelude::*, px, rgb, rgba,
};
use tonemapping::{MaxCllMode, ToneMappingMethod};

use super::image_loader::{
    DisplayedImage, HdrOptions, ImageMetadata, LoadResult, LoadedImage, MetadataField,
    format_load_error, load_image, load_image_with_options,
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
const MAX_CLL_CHECKBOX_SIZE: f32 = 16.0;
const MAX_CLL_SELECTOR_HEIGHT: f32 = 30.0;
const TONE_MAPPING_MENU_ITEM_HEIGHT: f32 = 30.0;
const TONE_MAPPING_MENU_MARGIN: f32 = 4.0;
const TONE_MAPPING_MENU_WIDTH: f32 = 292.0;
const TONE_MAPPING_SELECTOR_HEIGHT: f32 = 30.0;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoadPurpose {
    Image,
    HdrOptions,
}

#[derive(Debug)]
struct DecodeJob<T> {
    request: LoadRequest,
    payload: T,
}

#[derive(Debug)]
struct DecodePayload {
    hdr_options: HdrOptions,
    path: Arc<Path>,
    purpose: LoadPurpose,
}

// Decoders are synchronous, so an active job must finish. Keeping only the latest waiting job
// bounds decode work to one allocation-heavy operation at a time without misrepresenting it as
// cancellable.
#[derive(Debug)]
struct LatestLoadCoordinator<T> {
    active: Option<LoadRequest>,
    queued: Option<DecodeJob<T>>,
}

impl<T> LatestLoadCoordinator<T> {
    fn new() -> Self {
        Self {
            active: None,
            queued: None,
        }
    }

    fn submit(&mut self, job: DecodeJob<T>) -> Option<DecodeJob<T>> {
        if self.active.is_none() {
            self.active = Some(job.request);
            return Some(job);
        }

        self.queued = Some(job);
        None
    }

    fn complete(&mut self, request: LoadRequest) -> Option<DecodeJob<T>> {
        let active = self
            .active
            .take()
            .expect("a decode completed while the load coordinator was idle");
        assert_eq!(
            active, request,
            "a decode other than the active request completed"
        );

        let next = self.queued.take();
        self.active = next.as_ref().map(|job| job.request);
        next
    }

    fn discard_queued(&mut self) {
        self.queued = None;
    }
}

pub(super) struct Root {
    context_menu_position: Option<Point<Pixels>>,
    decode_coordinator: LatestLoadCoordinator<DecodePayload>,
    load_generation: u64,
    metadata_visible: bool,
    pending_hdr_options: Option<(LoadRequest, HdrOptions)>,
    preferred_hdr_options: HdrOptions,
    tone_mapping_menu_open: bool,
    viewer: ViewerState,
}

impl Root {
    pub(super) fn new(viewer: ViewerState) -> Self {
        let preferred_hdr_options = viewer
            .displayed()
            .and_then(|displayed| displayed.hdr_options)
            .unwrap_or_default();

        Self {
            context_menu_position: None,
            decode_coordinator: LatestLoadCoordinator::new(),
            load_generation: 0,
            metadata_visible: false,
            pending_hdr_options: None,
            preferred_hdr_options,
            tone_mapping_menu_open: false,
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
        
        self.tone_mapping_menu_open = false;
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

    fn begin_hdr_options_selection(
        &mut self,
        options: HdrOptions,
        active_options: HdrOptions,
    ) -> Option<LoadRequest> {
        self.preferred_hdr_options = options;
        
        if options == active_options {
            if self.pending_hdr_options.take().is_some() {
                let _cancelled_request = self.begin_load_request();
                self.decode_coordinator.discard_queued();
            }
            return None;
        }
        
        if self
            .pending_hdr_options
            .is_some_and(|(_, pending_options)| pending_options == options)
        {
            return None;
        }

        let request = self.begin_load_request();
        self.pending_hdr_options = Some((request, options));
        Some(request)
    }

    fn open_image(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        invariant!(!self.viewer.status().is_empty());

        self.context_menu_position = None;
        self.tone_mapping_menu_open = false;
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

            let _ = root.update_in(cx, |root, window, cx| {
                root.context_menu_position = None;
                root.tone_mapping_menu_open = false;
                let request = root.begin_load_request();
                root.pending_hdr_options = None;
                root.schedule_decode(
                    DecodeJob {
                        request,
                        payload: DecodePayload {
                            hdr_options: root.preferred_hdr_options,
                            path: Arc::from(path),
                            purpose: LoadPurpose::Image,
                        },
                    },
                    window,
                    cx,
                );
                cx.notify();
            });
        })
        .detach();
    }

    fn schedule_decode(
        &mut self,
        job: DecodeJob<DecodePayload>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(job) = self.decode_coordinator.submit(job) {
            Self::spawn_decode(job, window, cx);
        }
    }

    fn spawn_decode(job: DecodeJob<DecodePayload>, window: &mut Window, cx: &mut Context<Self>) {
        let DecodeJob { request, payload } = job;
        let DecodePayload {
            hdr_options,
            path,
            purpose,
        } = payload;

        cx.spawn_in(window, async move |root, cx| {
            let result = cx
                .background_spawn(
                    async move { load_image_with_options(path.as_ref(), hdr_options) },
                )
                .await;
            
            let _ = root.update_in(cx, move |root, window, cx| {
                let next = root.decode_coordinator.complete(request);
                let applied = match purpose {
                    LoadPurpose::Image => root.apply_load_result(request, result),
                    LoadPurpose::HdrOptions => root.apply_hdr_options_result(request, result),
                };

                if let Some(next) = next {
                    Self::spawn_decode(next, window, cx);
                }
                if applied {
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
        self.context_menu_position = None;

        let load_succeeded = result.is_ok();
        self.viewer.apply_result(result);
        if load_succeeded {
            self.metadata_visible = false;
            self.tone_mapping_menu_open = false;
        }

        invariant!(!self.viewer.status().is_empty());
        true
    }

    fn apply_hdr_options_result(
        &mut self,
        request: LoadRequest,
        result: LoadResult<LoadedImage>,
    ) -> bool {
        if !self.accepts_load_request(request) {
            return false;
        }

        invariant!(!self.viewer.status().is_empty());
        self.context_menu_position = None;

        let displayed_options = self
            .viewer
            .displayed()
            .and_then(|displayed| displayed.hdr_options);
        
        let resolved_options = result
            .as_ref()
            .ok()
            .and_then(|loaded| loaded.displayed.hdr_options)
            .or(displayed_options);
        
        self.viewer.apply_result(result);
        
        self.pending_hdr_options = None;
        if let Some(options) = resolved_options {
            self.preferred_hdr_options = options;
        }
        
        self.tone_mapping_menu_open = false;

        invariant!(!self.viewer.status().is_empty());
        true
    }

    fn select_tone_mapping(
        &mut self,
        method: ToneMappingMethod,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let options = self.preferred_hdr_options.with_tone_mapping(method);
        self.select_hdr_options(options, window, cx);
    }

    fn select_max_cll_mode(
        &mut self,
        max_cll_mode: MaxCllMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let options = self.preferred_hdr_options.with_max_cll_mode(max_cll_mode);
        self.select_hdr_options(options, window, cx);
    }

    fn select_hdr_options(
        &mut self,
        options: HdrOptions,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.context_menu_position = None;
        self.tone_mapping_menu_open = false;

        let Some((active_options, source_path)) = self.viewer.displayed().and_then(|displayed| {
            displayed
                .hdr_options
                .map(|active| (active, Arc::clone(&displayed.source_path)))
        }) else {
            cx.notify();
            return;
        };

        let Some(request) = self.begin_hdr_options_selection(options, active_options) else {
            cx.notify();
            return;
        };

        self.schedule_decode(
            DecodeJob {
                request,
                payload: DecodePayload {
                    hdr_options: options,
                    path: source_path,
                    purpose: LoadPurpose::HdrOptions,
                },
            },
            window,
            cx,
        );
        cx.notify();
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
                            root.tone_mapping_menu_open = false;
                            cx.notify();
                        },
                    )))
                })
                .child(menu_item("quit", "Quit").on_click(|_, _, cx| cx.quit())),
        )
        .priority(1)
    }

    fn render_metadata_overlay(
        metadata: &ImageMetadata,
        hdr_options: Option<HdrOptions>,
        tone_mapping_menu_open: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
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
            .when_some(hdr_options, |overlay, options| {
                overlay
                    .child(Self::render_tone_mapping_selector(
                        options.tone_mapping(),
                        tone_mapping_menu_open,
                        cx,
                    ))
                    .child(Self::render_max_cll_selector(options.max_cll_mode(), cx))
            })
            .children(metadata.fields.iter().map(metadata_field))
    }

    fn render_tone_mapping_selector(
        active_method: ToneMappingMethod,
        menu_open: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let selector = div()
            .id("tone-mapping-selector")
            .relative()
            .h(px(TONE_MAPPING_SELECTOR_HEIGHT))
            .min_w_0()
            .flex_1()
            .flex()
            .items_center()
            .justify_between()
            .px_2()
            .rounded_sm()
            .border_1()
            .border_color(rgba(0xff_ff_ff_2e))
            .bg(rgb(0x0024_2424))
            .cursor_pointer()
            .hover(|style| style.bg(rgb(0x0032_3232)))
            .child(active_method.label())
            .child(div().ml_2().text_color(rgb(0x009d_9d9d)).child("▼"))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |root, _, _, cx| {
                    root.context_menu_position = None;
                    root.tone_mapping_menu_open = !menu_open;
                    cx.notify();
                }),
            )
            .when(menu_open, |selector| {
                selector.child(Self::render_tone_mapping_menu(active_method, cx))
            });

        div()
            .w_full()
            .flex()
            .items_center()
            .gap(px(METADATA_FIELD_GAP))
            .pb_1()
            .child(
                div()
                    .w(px(METADATA_LABEL_WIDTH))
                    .flex_none()
                    .text_color(rgb(0x009d_9d9d))
                    .child("Tone mapper"),
            )
            .child(selector)
    }

    fn render_max_cll_selector(selected_mode: MaxCllMode, cx: &mut Context<Self>) -> gpui::Div {
        let checked = selected_mode == MaxCllMode::TrueMaximum;
        let next_mode = toggled_max_cll_mode(selected_mode);
        let description = if checked {
            "True maximum"
        } else {
            "99.99th percentile"
        };

        let selector = div()
            .id("max-cll-selector")
            .role(Role::CheckBox)
            .aria_label("Use true maximum MaxCLL")
            .aria_toggled(if checked {
                Toggled::True
            } else {
                Toggled::False
            })
            .h(px(MAX_CLL_SELECTOR_HEIGHT))
            .min_w_0()
            .flex_1()
            .flex()
            .items_center()
            .gap_2()
            .px_2()
            .rounded_sm()
            .border_1()
            .border_color(rgba(0xff_ff_ff_2e))
            .bg(rgb(0x0024_2424))
            .cursor_pointer()
            .hover(|style| style.bg(rgb(0x0032_3232)))
            .child(
                div()
                    .size(px(MAX_CLL_CHECKBOX_SIZE))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_xs()
                    .border_1()
                    .border_color(rgba(0xff_ff_ff_55))
                    .when(checked, |checkbox| checkbox.bg(rgb(0x0038_3838)))
                    .text_color(rgb(0x00a9_d18e))
                    .child(if checked { "✓" } else { "" }),
            )
            .child(description)
            .on_click(cx.listener(move |root, _, window, cx| {
                root.select_max_cll_mode(next_mode, window, cx);
            }));

        div()
            .w_full()
            .flex()
            .items_center()
            .gap(px(METADATA_FIELD_GAP))
            .pb_1()
            .child(
                div()
                    .w(px(METADATA_LABEL_WIDTH))
                    .flex_none()
                    .text_color(rgb(0x009d_9d9d))
                    .child("MaxCLL"),
            )
            .child(selector)
    }

    fn render_tone_mapping_menu(
        active_method: ToneMappingMethod,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        deferred(
            anchored()
                .anchor(Anchor::TopLeft)
                .position(point(
                    px(0.0),
                    px(TONE_MAPPING_SELECTOR_HEIGHT + TONE_MAPPING_MENU_MARGIN),
                ))
                .position_mode(AnchoredPositionMode::Local)
                .snap_to_window_with_margin(px(METADATA_OVERLAY_MARGIN))
                .child(
                    div()
                        .occlude()
                        .w(px(TONE_MAPPING_MENU_WIDTH))
                        .p_1()
                        .rounded_md()
                        .shadow_lg()
                        .border_1()
                        .border_color(rgb(0x0045_4545))
                        .bg(rgb(0x0029_2929))
                        .flex()
                        .flex_col()
                        .children(ToneMappingMethod::ALL.map(|method| {
                            tone_mapping_menu_item(method, active_method).on_click(cx.listener(
                                move |root, _, window, cx| {
                                    root.select_tone_mapping(method, window, cx);
                                },
                            ))
                        }))
                        .on_mouse_down_out(cx.listener(|root, _, _, cx| {
                            root.tone_mapping_menu_open = false;
                            cx.notify();
                        })),
                ),
        )
        .priority(2)
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
        
        let active_hdr_options = displayed
            .as_ref()
            .and_then(|displayed| displayed.hdr_options);
        
        let hdr_options = active_hdr_options.map(|active_options| {
            self.pending_hdr_options
                .map_or(active_options, |(_, pending_options)| pending_options)
        });
        
        let metadata_visible = self.metadata_visible;
        let tone_mapping_menu_open = self.tone_mapping_menu_open;
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
                root.child(Self::render_metadata_overlay(
                    &metadata,
                    hdr_options,
                    tone_mapping_menu_open,
                    cx,
                ))
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

const fn toggled_max_cll_mode(mode: MaxCllMode) -> MaxCllMode {
    match mode {
        MaxCllMode::Percentile99_99 => MaxCllMode::TrueMaximum,
        MaxCllMode::TrueMaximum => MaxCllMode::Percentile99_99,
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

fn tone_mapping_menu_item(
    method: ToneMappingMethod,
    active_method: ToneMappingMethod,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(method.label())
        .h(px(TONE_MAPPING_MENU_ITEM_HEIGHT))
        .w_full()
        .flex()
        .items_center()
        .px_2()
        .rounded_sm()
        .cursor_pointer()
        .text_color(rgb(0x00ff_ffff))
        .hover(|style| style.bg(rgb(0x003d_3d3d)))
        .when(method == active_method, |item| item.bg(rgb(0x0038_3838)))
        .child(
            div()
                .w_5()
                .flex_none()
                .text_color(rgb(0x00a9_d18e))
                .child(if method == active_method { "✓" } else { "" }),
        )
        .child(method.label())
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
        .when(field.starts_section, Styled::mt_2)
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

    fn loaded_hdr_viewer(options: HdrOptions) -> ViewerState {
        ViewerState::Loaded(LoadedImage {
            displayed: DisplayedImage {
                image: Arc::new(GPUIImage::empty()),
                metadata: Arc::new(ImageMetadata { fields: Vec::new() }),
                source_path: Arc::from(Path::new("test.jxr")),
                hdr_options: Some(options),
            },
            status: "test.jxr".into(),
        })
    }

    #[test]
    fn image_information_is_hidden_by_default() {
        let root = Root::new(ViewerState::empty());

        assert!(!root.metadata_visible);
        assert!(!root.tone_mapping_menu_open);
        assert_eq!(
            root.preferred_hdr_options.tone_mapping(),
            ToneMappingMethod::ExtendedReinhard
        );
        assert_eq!(
            root.preferred_hdr_options.max_cll_mode(),
            MaxCllMode::Percentile99_99
        );
    }

    #[test]
    fn context_menu_adds_an_item_for_a_loaded_image() {
        assert!(context_menu_height(true) > context_menu_height(false));
    }

    #[test]
    fn load_coordinator_runs_one_decode_and_keeps_only_the_latest_waiter() {
        let mut coordinator = LatestLoadCoordinator::new();
        let first = DecodeJob {
            request: LoadRequest(1),
            payload: "first",
        };
        
        let second = DecodeJob {
            request: LoadRequest(2),
            payload: "second",
        };
        
        let latest = DecodeJob {
            request: LoadRequest(3),
            payload: "latest",
        };

        let started = coordinator
            .submit(first)
            .expect("the first request starts immediately");
        
        assert_eq!(started.payload, "first");
        assert!(coordinator.submit(second).is_none());
        assert!(coordinator.submit(latest).is_none());

        let started = coordinator
            .complete(LoadRequest(1))
            .expect("the latest waiting request starts next");
        
        assert_eq!(started.request, LoadRequest(3));
        assert_eq!(started.payload, "latest");
        assert!(coordinator.complete(LoadRequest(3)).is_none());
    }

    #[test]
    fn load_coordinator_can_discard_waiting_work_without_cancelling_active_work() {
        let mut coordinator = LatestLoadCoordinator::new();
        
        let active = DecodeJob {
            request: LoadRequest(1),
            payload: "active",
        };
        
        let waiting = DecodeJob {
            request: LoadRequest(2),
            payload: "waiting",
        };

        assert!(coordinator.submit(active).is_some());
        assert!(coordinator.submit(waiting).is_none());
        coordinator.discard_queued();

        assert!(coordinator.complete(LoadRequest(1)).is_none());
        let replacement = DecodeJob {
            request: LoadRequest(3),
            payload: "replacement",
        };
        assert_eq!(
            coordinator
                .submit(replacement)
                .expect("the coordinator is idle after active work completes")
                .payload,
            "replacement"
        );
    }

    #[test]
    fn stale_load_result_cannot_replace_newer_request() {
        let mut root = Root::new(ViewerState::empty());
        let first = root.begin_load_request();
        let second = root.begin_load_request();

        let stale_error = LoadError::new("stale load failed").raise();
        assert!(!root.apply_load_result(first, Err(stale_error)));
        assert!(matches!(root.viewer, ViewerState::Empty { .. }));

        root.context_menu_position = Some(point(px(5.0), px(DRAG_REGION_HEIGHT)));
        let current_error = LoadError::new("current load failed").raise();
        assert!(root.apply_load_result(second, Err(current_error)));
        assert!(matches!(root.viewer, ViewerState::Failed { .. }));
        assert!(root.context_menu_position.is_none());
    }

    #[test]
    fn reselecting_the_displayed_options_cancels_a_pending_change() {
        let mut root = Root::new(ViewerState::empty());
        let active = HdrOptions::default();
        let selected = active.with_tone_mapping(ToneMappingMethod::ACESFitted);
        let pending = root
            .begin_hdr_options_selection(selected, active)
            .expect("different HDR options start a request");

        assert_eq!(root.preferred_hdr_options, selected);
        assert!(root.begin_hdr_options_selection(selected, active).is_none());
        assert!(root.accepts_load_request(pending));
        assert!(root.begin_hdr_options_selection(active, active).is_none());
        assert_eq!(root.preferred_hdr_options, active);
        assert!(!root.accepts_load_request(pending));
    }

    #[test]
    fn reselecting_the_displayed_options_preserves_an_image_load() {
        let mut root = Root::new(ViewerState::empty());
        let image_load = root.begin_load_request();
        let active = HdrOptions::default();

        assert!(root.begin_hdr_options_selection(active, active).is_none());
        assert!(root.accepts_load_request(image_load));
    }

    #[test]
    fn hdr_options_result_closes_an_open_context_menu() {
        let mut root = Root::new(ViewerState::empty());
        let request = root.begin_load_request();
        root.pending_hdr_options = Some((
            request,
            HdrOptions::default().with_tone_mapping(ToneMappingMethod::ACESFitted),
        ));
        root.context_menu_position = Some(point(px(5.0), px(DRAG_REGION_HEIGHT)));
        let error = LoadError::new("HDR options reload failed").raise();

        assert!(root.apply_hdr_options_result(request, Err(error)));
        assert!(root.context_menu_position.is_none());
        assert!(root.pending_hdr_options.is_none());
    }

    #[test]
    fn failed_hdr_reload_restores_the_displayed_options() {
        let active = HdrOptions::default();
        let selected = active.with_tone_mapping(ToneMappingMethod::ACESFitted);
        let mut root = Root::new(loaded_hdr_viewer(active));
        
        let request = root
            .begin_hdr_options_selection(selected, active)
            .expect("different HDR options start a request");
        
        let error = LoadError::new("HDR options reload failed").raise();

        assert_eq!(root.preferred_hdr_options, selected);
        assert!(root.apply_hdr_options_result(request, Err(error)));
        assert_eq!(root.preferred_hdr_options, active);
        assert_eq!(
            root.viewer
                .displayed()
                .and_then(|displayed| displayed.hdr_options),
            Some(active)
        );
    }

    #[test]
    fn tone_mapping_and_max_cll_changes_compose_in_one_reload() {
        let mut root = Root::new(ViewerState::empty());
        let active = HdrOptions::default();
        let true_maximum = active.with_max_cll_mode(MaxCllMode::TrueMaximum);
        
        let max_cll_request = root
            .begin_hdr_options_selection(true_maximum, active)
            .expect("a MaxCLL change starts a request");

        let combined = root
            .preferred_hdr_options
            .with_tone_mapping(ToneMappingMethod::ACESFitted);
        
        let combined_request = root
            .begin_hdr_options_selection(combined, active)
            .expect("a composed HDR option change starts a new request");

        assert!(!root.accepts_load_request(max_cll_request));
        assert!(root.accepts_load_request(combined_request));
        assert_eq!(root.pending_hdr_options, Some((combined_request, combined)));
        assert_eq!(combined.tone_mapping(), ToneMappingMethod::ACESFitted);
        assert_eq!(combined.max_cll_mode(), MaxCllMode::TrueMaximum);
    }

    #[test]
    fn max_cll_toggle_returns_to_the_percentile_mode() {
        assert_eq!(
            toggled_max_cll_mode(toggled_max_cll_mode(MaxCllMode::Percentile99_99)),
            MaxCllMode::Percentile99_99
        );
    }
}
