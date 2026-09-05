//! Owns viewer state and renders the GPUI interface.

use std::cell::OnceCell;
use std::path::Path;
use std::sync::Arc;

use exn::ErrorExt;

use gpui::{
    Anchor, AnchoredPositionMode, App, CursorStyle, FocusHandle, Focusable, Image as GPUIImage,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathPromptOptions, Pixels, Point,
    Role, ScrollWheelEvent, SharedString, Toggled, Window, WindowControlArea, actions, anchored,
    deferred, div, img, point, prelude::*, px, rgb, rgba,
};
use tonemapping::{MaxCLLMode, ToneMappingMethod};

use super::image_loader::{
    DisplayedImage, HDROptions, ImageMetadata, LoadError, LoadResult, LoadedImage, MetadataField,
    format_load_error, load_image, load_image_with,
};

const CONTEXT_MENU_ITEM_HEIGHT: f32 = 36.0;
const CONTEXT_MENU_PADDING: f32 = 8.0;
const CONTEXT_MENU_WIDTH: f32 = 180.0;
const DRAG_REGION_HEIGHT: f32 = 40.0;
const METADATA_FIELD_GAP: f32 = 6.0;
const METADATA_LABEL_WIDTH: f32 = 140.0;
const METADATA_OVERLAY_MARGIN: f32 = 12.0;
const METADATA_OVERLAY_WIDTH: f32 = 480.0;
const MAX_CLL_CHECKBOX_SIZE: f32 = 16.0;
const MAX_CLL_SELECTOR_HEIGHT: f32 = 30.0;
const TONE_MAPPING_MENU_ITEM_HEIGHT: f32 = 30.0;
const TONE_MAPPING_MENU_MARGIN: f32 = 4.0;
const TONE_MAPPING_MENU_WIDTH: f32 = 292.0;
const TONE_MAPPING_LABEL_WIDTH: f32 = 100.0;
const TONE_MAPPING_TITLEBAR_WIDTH: f32 = 480.0;
const TONE_MAPPING_SELECTOR_HEIGHT: f32 = 30.0;

/// Background of the root viewer surface.
const COLOR_APP_BACKGROUND: u32 = 0x0015_1515;
/// Primary text on the root viewer surface.
const COLOR_TEXT_PRIMARY: u32 = 0x00d8_d8d8;
/// Secondary text: field labels, muted captions, the tone-mapping menu caret.
const COLOR_TEXT_SECONDARY: u32 = 0x009d_9d9d;
/// Text inside the metadata overlay's value column.
const COLOR_TEXT_VALUE: u32 = 0x00e8_e8e8;
/// Hint text shown when no image is loaded.
const COLOR_TEXT_HINT: u32 = 0x0088_8888;
/// Text in the right-click context menu and tone-mapping method list.
const COLOR_TEXT_MENU: u32 = 0x00ff_ffff;
/// Checkmark and selected-method accent color.
const COLOR_ACCENT_GREEN: u32 = 0x00a9_d18e;
/// Background shared by the context menu and the tone-mapping method list panel.
const COLOR_PANEL_BACKGROUND: u32 = 0x0029_2929;
/// Border shared by the context menu and the tone-mapping method list panel.
const COLOR_PANEL_BORDER: u32 = 0x0045_4545;
/// Hover background for context-menu and tone-mapping method-list items.
const COLOR_MENU_ITEM_HOVER: u32 = 0x003d_3d3d;
/// Background of the status bar and its embedded controls' resting state.
const COLOR_CONTROL_BACKGROUND: u32 = 0x0024_2424;
/// Hover background for the tone-mapping and `MaxCLL` selector controls.
const COLOR_CONTROL_HOVER: u32 = 0x0032_3232;
/// Border for the tone-mapping and `MaxCLL` selector controls (translucent white).
const COLOR_CONTROL_BORDER: u32 = 0xff_ff_ff_2e;
/// Border of the status bar strip (translucent white).
const COLOR_STATUS_BAR_BORDER: u32 = 0xff_ff_ff_22;
/// Border for the `MaxCLL` checkbox (translucent white).
const COLOR_CHECKBOX_BORDER: u32 = 0xff_ff_ff_55;
/// Background of a selected tone-mapping method or a checked `MaxCLL` checkbox.
const COLOR_SELECTED_BACKGROUND: u32 = 0x0038_3838;
/// Background of the status bar strip along the window's bottom edge.
const COLOR_STATUS_BAR_BACKGROUND: u32 = 0x0d_0d_0d_e8;
/// Background of the metadata overlay panel.
const COLOR_METADATA_OVERLAY_BACKGROUND: u32 = 0x0020_2020;

pub(super) const WINDOW_MIN_WIDTH: f32 = 1280.0;
pub(super) const WINDOW_MIN_HEIGHT: f32 = 720.0;

/// Zoom multiplier on top of the fit-to-window baseline; 1.0 means "fit".
const ZOOM_MIN: f32 = 0.1;
const ZOOM_MAX: f32 = 16.0;
/// Multiplier applied per normalized scroll step.
const ZOOM_STEP_BASE: f32 = 1.001;
/// Multiplier applied per `ZoomIn`/`ZoomOut` action.
const ZOOM_KEY_STEP: f32 = 1.25;
/// Assumed line height for normalizing line-based scroll deltas into pixels.
const SCROLL_LINE_HEIGHT: f32 = 24.0;

actions!(
    four,
    [Quit, OpenFile, ZoomIn, ZoomOut, ZoomReset, DismissMenu]
);

/// Scale that fits an `image_w`×`image_h` image inside `content_w`×`content_h`, preserving
/// aspect ratio (matches gpui's `ObjectFit::Contain`, which this replaces).
#[expect(
    clippy::cast_precision_loss,
    reason = "image dimensions stay far below f32's 2^24 exact-integer range"
)]
fn fit_scale(content_w: Pixels, content_h: Pixels, image_w: u32, image_h: u32) -> f32 {
    assert!(image_w > 0, "image width must be nonzero");
    assert!(image_h > 0, "image height must be nonzero");

    // Floored so a content area shrunk to zero or negative (viewport smaller than the drag region)
    // can't return a zero scale: a later zoom would then divide by zero in `zoom_to_cursor_pan` and
    // get NaN stuck in `self.pan` forever, since `f32::clamp` passes NaN through unchanged.
    let content_w = f32::from(content_w).max(1.0);
    let content_h = f32::from(content_h).max(1.0);

    (content_w / image_w as f32).min(content_h / image_h as f32)
}

/// Clamps a pan offset so a `display`-sized image centered in a `content`-sized container can't
/// be dragged fully out of view.
fn clamp_pan(
    pan: Point<Pixels>,
    display_w: Pixels,
    display_h: Pixels,
    content_w: Pixels,
    content_h: Pixels,
) -> Point<Pixels> {
    let max_x = ((display_w - content_w) * 0.5).max(px(0.0));
    let max_y = ((display_h - content_h) * 0.5).max(px(0.0));

    point(pan.x.clamp(-max_x, max_x), pan.y.clamp(-max_y, max_y))
}

/// Pan offset that keeps the image point under `cursor_offset` (relative to the content-area
/// center) fixed while the effective scale changes from `old_scale` to `new_scale`.
fn zoom_to_cursor_pan(
    cursor_offset: Point<Pixels>,
    old_pan: Point<Pixels>,
    old_scale: f32,
    new_scale: f32,
) -> Point<Pixels> {
    let image_point = (cursor_offset - old_pan) / old_scale;
    cursor_offset - image_point * new_scale
}

#[cfg(test)]
mod zoom_math_tests {
    use super::{clamp_pan, fit_scale, zoom_to_cursor_pan};
    use gpui::{point, px};

    #[test]
    fn fit_scale_stays_positive_when_content_area_collapses() {
        assert!(fit_scale(px(0.0), px(0.0), 100, 100) > 0.0);
        assert!(fit_scale(px(-5.0), px(200.0), 100, 100) > 0.0);
    }

    #[test]
    fn zoom_to_cursor_pan_is_identity_when_scale_unchanged() {
        let cursor = point(px(50.0), px(-30.0));
        let pan = point(px(10.0), px(5.0));

        assert_eq!(zoom_to_cursor_pan(cursor, pan, 2.0, 2.0), pan);
    }

    #[test]
    fn zoom_to_cursor_pan_leaves_pan_unchanged_when_cursor_is_centered() {
        let cursor = point(px(0.0), px(0.0));
        let pan = point(px(0.0), px(0.0));

        assert_eq!(zoom_to_cursor_pan(cursor, pan, 1.0, 4.0), pan);
    }

    #[test]
    fn clamp_pan_constrains_oversized_image_pan() {
        let pan = point(px(1000.0), px(1000.0));
        let clamped = clamp_pan(pan, px(400.0), px(300.0), px(200.0), px(200.0));

        assert_eq!(clamped, point(px(100.0), px(50.0)));
    }

    #[test]
    fn clamp_pan_forces_zero_when_image_fits_within_content() {
        let pan = point(px(40.0), px(40.0));
        let clamped = clamp_pan(pan, px(100.0), px(100.0), px(200.0), px(200.0));

        assert_eq!(clamped, point(px(0.0), px(0.0)));
    }
}

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
        assert!(
            !self.status().is_empty(),
            "viewer status must never be blank before a result is applied"
        );

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

        assert!(
            !self.status().is_empty(),
            "viewer status must never be blank after a result is applied"
        );
    }

    fn status(&self) -> &SharedString {
        let status = match self {
            Self::Empty { status }
            | Self::Failed { status }
            | Self::LoadedFailed { status, .. } => status,
            Self::Loaded(state) => &state.status,
        };

        assert_ne!(status.len(), 0, "viewer status must never be blank");
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
    HdrMetrics,
    HDROptions,
}

#[derive(Debug)]
struct DecodeJob<T> {
    request: LoadRequest,
    payload: T,
}

#[derive(Debug)]
struct DecodePayload {
    hdr_options: HDROptions,
    include_hdr_metrics: bool,
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
    /// Mouse position at the last drag event during an active left-drag pan; `None` when not
    /// panning. Updated every move so a pan change from another source (e.g. a scroll-wheel zoom)
    /// in between two drag events is preserved instead of overwritten from a stale anchor.
    drag_anchor: Option<Point<Pixels>>,
    /// Lazily created on first access, since `Root::new` runs in plain unit tests with no `App`
    /// available to call `cx.focus_handle()`.
    focus_handle: OnceCell<FocusHandle>,
    hdr_metrics_request: Option<LoadRequest>,
    last_window_title: Option<SharedString>,
    load_generation: u64,
    metadata_visible: bool,
    /// Pan offset in pixels, relative to the image being centered in the content area.
    pan: Point<Pixels>,
    pending_hdr_options: Option<(LoadRequest, HDROptions)>,
    preferred_hdr_options: HDROptions,
    tone_mapping_menu_open: bool,
    viewer: ViewerState,
    /// Zoom multiplier on top of fit-to-window; clamped to `[ZOOM_MIN, ZOOM_MAX]`.
    zoom: f32,
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
            drag_anchor: None,
            focus_handle: OnceCell::new(),
            hdr_metrics_request: None,
            last_window_title: None,
            load_generation: 0,
            metadata_visible: false,
            pan: Point::default(),
            pending_hdr_options: None,
            preferred_hdr_options,
            tone_mapping_menu_open: false,
            viewer,
            zoom: 1.0,
        }
    }

    fn show_context_menu(&mut self, event: &MouseDownEvent, window: &Window) {
        let mut position = event.position;
        let viewport_size = window.viewport_size();
        let menu_height = context_menu_height(self.viewer.has_image());

        // `.max(min_*)` on each ceiling guarantees min <= max even if the viewport is smaller than
        // the menu, so clamping to the floor afterward can never push the position back past the
        // ceiling.
        let max_x = (viewport_size.width - px(CONTEXT_MENU_WIDTH)).max(px(0.0));
        let max_y = (viewport_size.height - px(menu_height)).max(px(DRAG_REGION_HEIGHT));
        position.x = position.x.clamp(px(0.0), max_x);
        position.y = position.y.clamp(px(DRAG_REGION_HEIGHT), max_y);

        assert!(
            position.x >= px(0.0),
            "context menu x position must be clamped nonnegative, got {:?}",
            position.x
        );
        assert!(
            position.y >= px(DRAG_REGION_HEIGHT),
            "context menu y position must clear the drag region, got {:?}",
            position.y
        );

        self.tone_mapping_menu_open = false;
        self.context_menu_position = Some(position);
    }

    fn dismiss_menus(&mut self) {
        self.context_menu_position = None;
        self.tone_mapping_menu_open = false;
    }

    fn reset_zoom(&mut self) {
        self.zoom = 1.0;
        self.pan = Point::default();
        self.drag_anchor = None;
    }

    fn apply_zoom(&mut self, new_zoom: f32, cx: &mut Context<Self>) {
        let new_zoom = new_zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        if (new_zoom - self.zoom).abs() > f32::EPSILON {
            self.pan = zoom_to_cursor_pan(Point::default(), self.pan, self.zoom, new_zoom);
            self.zoom = new_zoom;
            cx.notify();
        }
    }

    fn zoom_in(&mut self, cx: &mut Context<Self>) {
        self.apply_zoom(self.zoom * ZOOM_KEY_STEP, cx);
    }

    fn zoom_out(&mut self, cx: &mut Context<Self>) {
        self.apply_zoom(self.zoom / ZOOM_KEY_STEP, cx);
    }

    fn zoom_reset(&mut self, cx: &mut Context<Self>) {
        self.reset_zoom();
        cx.notify();
    }

    /// Sets the window title only when it actually changed, so title bookkeeping isn't tied to
    /// render frequency (mirrors Zed's `Workspace::apply_window_title`).
    pub(super) fn sync_window_title(&mut self, window: &mut Window) {
        let title = self.viewer.status().clone();
        if self.last_window_title.as_ref() == Some(&title) {
            return;
        }
        window.set_window_title(&title);
        self.last_window_title = Some(title);
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
        options: HDROptions,
        active_options: HDROptions,
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
        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank"
        );

        self.dismiss_menus();
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open an image".into()),
        });

        cx.spawn_in(window, async move |root, cx| {
            let path = match paths.await {
                Ok(Ok(Some(mut paths))) => paths.pop(),
                Ok(Ok(None)) => None,
                // `Ok(Err(_))` is the platform reporting the dialog itself failed; `Err(_)` is the
                // response channel being dropped before it replied. Both are real failures, unlike
                // `Ok(Ok(None))` (the user just canceled), so both get surfaced the same way.
                Ok(Err(prompt_error)) => {
                    let _ = root.update_in(cx, |root, window, cx| {
                        root.dismiss_menus();
                        root.viewer
                            .apply_result(Err(LoadError::new(prompt_error.to_string()).raise()));
                        root.sync_window_title(window);
                        cx.notify();
                    });
                    return;
                }
                Err(prompt_error) => {
                    let _ = root.update_in(cx, |root, window, cx| {
                        root.dismiss_menus();
                        root.viewer
                            .apply_result(Err(LoadError::new(prompt_error.to_string()).raise()));
                        root.sync_window_title(window);
                        cx.notify();
                    });
                    return;
                }
            };
            let Some(path) = path else {
                return;
            };

            let _ = root.update_in(cx, |root, window, cx| {
                root.dismiss_menus();

                let request = root.begin_load_request();
                root.hdr_metrics_request = None;
                root.pending_hdr_options = None;

                root.schedule_decode(
                    DecodeJob {
                        request,
                        payload: DecodePayload {
                            hdr_options: root.preferred_hdr_options,
                            include_hdr_metrics: false,
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
            include_hdr_metrics,
            path,
            purpose,
        } = payload;

        cx.spawn_in(window, async move |root, cx| {
            let result = cx
                .background_spawn(async move {
                    load_image_with(path.as_ref(), hdr_options, include_hdr_metrics)
                })
                .await;

            let _ = root.update_in(cx, move |root, window, cx| {
                let next = root.decode_coordinator.complete(request);
                if root.hdr_metrics_request == Some(request) {
                    root.hdr_metrics_request = None;
                }

                let applied = match purpose {
                    LoadPurpose::Image => root.apply_load_result(request, result),
                    LoadPurpose::HdrMetrics | LoadPurpose::HDROptions => {
                        root.apply_hdr_options_result(request, result)
                    }
                };

                if let Some(next) = next {
                    Self::spawn_decode(next, window, cx);
                }

                if applied {
                    root.sync_window_title(window);
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

        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank before a load result is applied"
        );
        self.context_menu_position = None;

        let load_succeeded = result.is_ok();
        self.viewer.apply_result(result);
        if load_succeeded {
            self.metadata_visible = false;
            self.tone_mapping_menu_open = false;
            self.reset_zoom();
        }

        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank after a load result is applied"
        );
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

        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank before an HDR options result is applied"
        );
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

        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank after an HDR options result is applied"
        );
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
        max_cll_mode: MaxCLLMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let options = self.preferred_hdr_options.with_max_cll_mode(max_cll_mode);
        self.select_hdr_options(options, window, cx);
    }

    fn select_hdr_options(
        &mut self,
        options: HDROptions,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dismiss_menus();

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

        self.hdr_metrics_request = self.metadata_visible.then_some(request);

        self.schedule_decode(
            DecodeJob {
                request,
                payload: DecodePayload {
                    hdr_options: options,
                    include_hdr_metrics: self.metadata_visible,
                    path: source_path,
                    purpose: LoadPurpose::HDROptions,
                },
            },
            window,
            cx,
        );
        cx.notify();
    }

    fn toggle_metadata(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.dismiss_menus();
        self.metadata_visible = !self.metadata_visible;

        if !self.metadata_visible {
            cx.notify();
            return;
        }

        let Some(displayed) = self.viewer.displayed() else {
            cx.notify();
            return;
        };
        let Some(active_options) = displayed.hdr_options else {
            cx.notify();
            return;
        };
        if displayed.metadata.has_hdr_metrics || self.hdr_metrics_request.is_some() {
            cx.notify();
            return;
        }

        let path = Arc::clone(&displayed.source_path);
        let options = self
            .pending_hdr_options
            .map_or(active_options, |(_, options)| options);
        let request = self.begin_load_request();
        self.hdr_metrics_request = Some(request);
        self.schedule_decode(
            DecodeJob {
                request,
                payload: DecodePayload {
                    hdr_options: options,
                    include_hdr_metrics: true,
                    path,
                    purpose: LoadPurpose::HdrMetrics,
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
        assert!(
            position.x >= px(0.0),
            "context menu x position must be clamped nonnegative, got {:?}",
            position.x
        );
        assert!(
            position.y >= px(DRAG_REGION_HEIGHT),
            "context menu y position must clear the drag region, got {:?}",
            position.y
        );

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
                .border_color(rgb(COLOR_PANEL_BORDER))
                .bg(rgb(COLOR_PANEL_BACKGROUND))
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
                    menu.child(menu_item("toggle-image-info", label).on_click(
                        cx.listener(|root, _, window, cx| root.toggle_metadata(window, cx)),
                    ))
                })
                .child(menu_item("quit", "Quit").on_click(|_, _, cx| cx.quit())),
        )
        .priority(1)
    }

    fn render_metadata_overlay(
        metadata: &ImageMetadata,
        hdr_options: Option<HDROptions>,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        assert_ne!(
            metadata.fields.len(),
            0,
            "metadata overlay requires at least one field to display"
        );

        div()
            .absolute()
            .left(px(METADATA_OVERLAY_MARGIN))
            .top(px(DRAG_REGION_HEIGHT + METADATA_OVERLAY_MARGIN))
            .w(px(METADATA_OVERLAY_WIDTH))
            .p_3()
            .rounded_md()
            .border_1()
            .border_color(rgba(COLOR_STATUS_BAR_BORDER))
            .shadow_lg()
            .bg(rgba(COLOR_STATUS_BAR_BACKGROUND))
            .font_family("Consolas")
            .text_sm()
            .flex()
            .flex_col()
            .when_some(hdr_options, |overlay, options| {
                overlay.child(Self::render_max_cll_selector(options.max_cll_mode(), cx))
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
            .border_color(rgba(COLOR_CONTROL_BORDER))
            .bg(rgb(COLOR_CONTROL_BACKGROUND))
            .cursor_pointer()
            .hover(|style| style.bg(rgb(COLOR_CONTROL_HOVER)))
            .child(active_method.label())
            .child(
                div()
                    .ml_2()
                    .text_color(rgb(COLOR_TEXT_SECONDARY))
                    .child("▼"),
            )
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
            .child(
                div()
                    .w(px(TONE_MAPPING_LABEL_WIDTH))
                    .flex_none()
                    .text_color(rgb(COLOR_TEXT_SECONDARY))
                    .child("Tone mapper"),
            )
            .child(selector)
    }

    fn render_max_cll_selector(selected_mode: MaxCLLMode, cx: &mut Context<Self>) -> gpui::Div {
        let checked = selected_mode == MaxCLLMode::TrueMaximum;
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
            .border_color(rgba(COLOR_CONTROL_BORDER))
            .bg(rgb(COLOR_CONTROL_BACKGROUND))
            .cursor_pointer()
            .hover(|style| style.bg(rgb(COLOR_CONTROL_HOVER)))
            .child(
                div()
                    .size(px(MAX_CLL_CHECKBOX_SIZE))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_xs()
                    .border_1()
                    .border_color(rgba(COLOR_CHECKBOX_BORDER))
                    .when(checked, |checkbox| {
                        checkbox.bg(rgb(COLOR_SELECTED_BACKGROUND))
                    })
                    .text_color(rgb(COLOR_ACCENT_GREEN))
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
                    .text_color(rgb(COLOR_TEXT_SECONDARY))
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
                        .border_color(rgb(COLOR_PANEL_BORDER))
                        .bg(rgb(COLOR_PANEL_BACKGROUND))
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

    fn on_image_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        content_w: Pixels,
        content_h: Pixels,
        width: u32,
        height: u32,
        cx: &mut Context<Self>,
    ) {
        let delta = event.delta.pixel_delta(px(SCROLL_LINE_HEIGHT));
        let step = f32::from(delta.y) / SCROLL_LINE_HEIGHT;
        let new_zoom = (self.zoom * ZOOM_STEP_BASE.powf(step)).clamp(ZOOM_MIN, ZOOM_MAX);
        if (new_zoom - self.zoom).abs() > f32::EPSILON {
            let base_scale = fit_scale(content_w, content_h, width, height);
            let cursor_offset = point(
                event.position.x - content_w * 0.5,
                event.position.y - px(DRAG_REGION_HEIGHT) - content_h * 0.5,
            );
            self.pan = zoom_to_cursor_pan(
                cursor_offset,
                self.pan,
                base_scale * self.zoom,
                base_scale * new_zoom,
            );
            self.zoom = new_zoom;
            cx.notify();
        }
    }

    fn on_image_drag_start(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        if (self.zoom - 1.0).abs() > f32::EPSILON {
            self.drag_anchor = Some(event.position);
            cx.notify();
        }
    }

    fn on_image_drag_move(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        let Some(anchor_mouse) = self.drag_anchor else {
            return;
        };
        if event.pressed_button != Some(MouseButton::Left) {
            return;
        }
        self.pan += event.position - anchor_mouse;
        self.drag_anchor = Some(event.position);
        cx.notify();
    }

    fn on_image_drag_end(&mut self, cx: &mut Context<Self>) {
        if self.drag_anchor.take().is_some() {
            cx.notify();
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions stay far below f32's 2^24 exact-integer range"
    )]
    fn render_image_content(
        &mut self,
        image: Option<Arc<GPUIImage>>,
        image_dims: Option<(u32, u32)>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let has_image = image.is_some();

        div()
            .relative()
            .flex_1()
            .min_h_0()
            .min_w_0()
            .flex()
            .items_center()
            .justify_center()
            .overflow_hidden()
            .when_some(
                image.zip(image_dims),
                |content, (image, (width, height))| {
                    let viewport_size = window.viewport_size();
                    let content_w = viewport_size.width;
                    let content_h = viewport_size.height - px(DRAG_REGION_HEIGHT);

                    let scale = fit_scale(content_w, content_h, width, height) * self.zoom;
                    let display_w = px(width as f32) * scale;
                    let display_h = px(height as f32) * scale;

                    self.pan = clamp_pan(self.pan, display_w, display_h, content_w, content_h);
                    let pan = self.pan;
                    let zoom = self.zoom;
                    let left = (content_w - display_w) * 0.5 + pan.x;
                    let top = (content_h - display_h) * 0.5 + pan.y;

                    content
                        .on_scroll_wheel(cx.listener(
                            move |root, event: &ScrollWheelEvent, _, cx| {
                                root.on_image_scroll(
                                    event, content_w, content_h, width, height, cx,
                                );
                            },
                        ))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|root, event: &MouseDownEvent, _, cx| {
                                root.on_image_drag_start(event, cx);
                            }),
                        )
                        .on_mouse_move(cx.listener(move |root, event: &MouseMoveEvent, _, cx| {
                            root.on_image_drag_move(event, cx);
                        }))
                        .on_mouse_up(
                            MouseButton::Left,
                            cx.listener(|root, _: &MouseUpEvent, _, cx| {
                                root.on_image_drag_end(cx);
                            }),
                        )
                        .child(
                            img(image)
                                .id("displayed-image")
                                .absolute()
                                .left(left)
                                .top(top)
                                .w(display_w)
                                .h(display_h),
                        )
                        .when((zoom - 1.0).abs() > f32::EPSILON, |content| {
                            content.cursor(CursorStyle::OpenHand)
                        })
                },
            )
            .when(!has_image, |content| {
                content.child(
                    div()
                        .text_sm()
                        .text_color(rgb(COLOR_TEXT_HINT))
                        .child("Right-click anywhere, then choose Open image…"),
                )
            })
    }

    fn render_status_bar(
        status: SharedString,
        hdr_options: Option<HDROptions>,
        tone_mapping_menu_open: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        div()
            .w_full()
            .h(px(DRAG_REGION_HEIGHT))
            .flex_none()
            .flex()
            .items_center()
            .text_sm()
            .bg(rgb(COLOR_METADATA_OVERLAY_BACKGROUND))
            .child(
                div()
                    .h_full()
                    .min_w_0()
                    .flex_1()
                    .flex()
                    .items_center()
                    .overflow_hidden()
                    .px_3()
                    .window_control_area(WindowControlArea::Drag)
                    .child(status),
            )
            .when_some(hdr_options, |titlebar, options| {
                titlebar.child(
                    div()
                        .w(px(TONE_MAPPING_TITLEBAR_WIDTH))
                        .h_full()
                        .flex_none()
                        .flex()
                        .items_center()
                        .px_3()
                        .child(Self::render_tone_mapping_selector(
                            options.tone_mapping(),
                            tone_mapping_menu_open,
                            cx,
                        )),
                )
            })
    }
}

impl Focusable for Root {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.focus_handle.get_or_init(|| cx.focus_handle()).clone()
    }
}

impl Render for Root {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank"
        );
        assert!(
            self.context_menu_position
                .is_none_or(|position| position.x >= px(0.0)),
            "context menu x position must be clamped nonnegative, got {:?}",
            self.context_menu_position
        );

        let context_menu_position = self.context_menu_position;
        let displayed = self.viewer.displayed();
        let has_image = displayed.is_some();

        let image = displayed.map(|displayed| Arc::clone(&displayed.image));
        let image_dims = displayed.map(|displayed| (displayed.width, displayed.height));
        let metadata = displayed.map(|displayed| Arc::clone(&displayed.metadata));
        let active_hdr_options = displayed.and_then(|displayed| displayed.hdr_options);

        let hdr_options = active_hdr_options.map(|active_options| {
            self.pending_hdr_options
                .map_or(active_options, |(_, pending_options)| pending_options)
        });

        let metadata_visible = self.metadata_visible;
        let tone_mapping_menu_open = self.tone_mapping_menu_open;
        let status = self.viewer.status().clone();
        let focus_handle = self.focus_handle(cx);

        div()
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .key_context("Viewer")
            .track_focus(&focus_handle)
            .bg(rgb(COLOR_APP_BACKGROUND))
            .text_color(rgb(COLOR_TEXT_PRIMARY))
            .on_action(cx.listener(|root, _: &OpenFile, window, cx| root.open_image(window, cx)))
            .on_action(cx.listener(|root, _: &ZoomIn, _, cx| root.zoom_in(cx)))
            .on_action(cx.listener(|root, _: &ZoomOut, _, cx| root.zoom_out(cx)))
            .on_action(cx.listener(|root, _: &ZoomReset, _, cx| root.zoom_reset(cx)))
            .on_action(cx.listener(|root, _: &DismissMenu, _, cx| {
                root.dismiss_menus();
                cx.notify();
            }))
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|root, event: &MouseDownEvent, window, cx| {
                    root.show_context_menu(event, window);
                    cx.notify();
                }),
            )
            .child(Self::render_status_bar(
                status,
                hdr_options,
                tone_mapping_menu_open,
                cx,
            ))
            .child(self.render_image_content(image, image_dims, window, cx))
            .when_some(metadata.filter(|_| metadata_visible), |root, metadata| {
                root.child(Self::render_metadata_overlay(&metadata, hdr_options, cx))
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

const fn toggled_max_cll_mode(mode: MaxCLLMode) -> MaxCLLMode {
    match mode {
        MaxCLLMode::Percentile99_99 => MaxCLLMode::TrueMaximum,
        MaxCLLMode::TrueMaximum => MaxCLLMode::Percentile99_99,
    }
}

const fn context_menu_height(has_image: bool) -> f32 {
    let item_count = if has_image { 3.0 } else { 2.0 };
    CONTEXT_MENU_PADDING + CONTEXT_MENU_ITEM_HEIGHT * item_count
}

fn menu_item(identifier: &'static str, label: &'static str) -> gpui::Stateful<gpui::Div> {
    assert_ne!(
        identifier.len(),
        0,
        "menu item identifier must not be blank"
    );
    assert_ne!(label.len(), 0, "menu item label must not be blank");

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
        .text_color(rgb(COLOR_TEXT_MENU))
        .hover(|style| style.bg(rgb(COLOR_MENU_ITEM_HOVER)))
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
        .text_color(rgb(COLOR_TEXT_MENU))
        .hover(|style| style.bg(rgb(COLOR_MENU_ITEM_HOVER)))
        .when(method == active_method, |item| {
            item.bg(rgb(COLOR_SELECTED_BACKGROUND))
        })
        .child(
            div()
                .w_5()
                .flex_none()
                .text_color(rgb(COLOR_ACCENT_GREEN))
                .child(if method == active_method { "✓" } else { "" }),
        )
        .child(method.label())
}

fn metadata_field(field: &MetadataField) -> gpui::Div {
    assert_ne!(
        field.label.len(),
        0,
        "metadata field label must not be blank"
    );
    assert_ne!(
        field.value.len(),
        0,
        "metadata field value for {:?} must not be blank",
        field.label
    );

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
                .text_color(rgb(COLOR_TEXT_SECONDARY))
                .child(field.label),
        )
        .child(
            div()
                .min_w_0()
                .flex_1()
                .text_color(rgb(COLOR_TEXT_VALUE))
                .child(field.value.clone()),
        )
}

pub(super) fn initial_viewer(path: Option<&Path>) -> ViewerState {
    let viewer = match path {
        Some(path) => ViewerState::from_result(load_image(path)),
        None => ViewerState::empty(),
    };

    assert!(!viewer.status().is_empty());
    viewer
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded_hdr_viewer(options: HDROptions) -> ViewerState {
        ViewerState::Loaded(LoadedImage {
            displayed: DisplayedImage {
                image: Arc::new(GPUIImage::empty()),
                width: 1,
                height: 1,
                metadata: Arc::new(ImageMetadata {
                    fields: Vec::new(),
                    has_hdr_metrics: false,
                }),
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
        assert!(root.hdr_metrics_request.is_none());
        assert!(!root.tone_mapping_menu_open);
        assert_eq!(
            root.preferred_hdr_options.tone_mapping(),
            ToneMappingMethod::BT2446
        );
        assert_eq!(
            root.preferred_hdr_options.max_cll_mode(),
            MaxCLLMode::Percentile99_99
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
        let active = HDROptions::default();
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
        let active = HDROptions::default();

        assert!(root.begin_hdr_options_selection(active, active).is_none());
        assert!(root.accepts_load_request(image_load));
    }

    #[test]
    fn hdr_options_result_closes_an_open_context_menu() {
        let mut root = Root::new(ViewerState::empty());
        let request = root.begin_load_request();
        root.pending_hdr_options = Some((
            request,
            HDROptions::default().with_tone_mapping(ToneMappingMethod::ACESFitted),
        ));
        root.context_menu_position = Some(point(px(5.0), px(DRAG_REGION_HEIGHT)));
        let error = LoadError::new("HDR options reload failed").raise();

        assert!(root.apply_hdr_options_result(request, Err(error)));
        assert!(root.context_menu_position.is_none());
        assert!(root.pending_hdr_options.is_none());
    }

    #[test]
    fn failed_hdr_reload_restores_the_displayed_options() {
        let active = HDROptions::default();
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
        let active = HDROptions::default();
        let true_maximum = active.with_max_cll_mode(MaxCLLMode::TrueMaximum);

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
        assert_eq!(combined.max_cll_mode(), MaxCLLMode::TrueMaximum);
    }

    #[test]
    fn max_cll_toggle_returns_to_the_percentile_mode() {
        assert_eq!(
            toggled_max_cll_mode(toggled_max_cll_mode(MaxCLLMode::Percentile99_99)),
            MaxCLLMode::Percentile99_99
        );
    }
}
