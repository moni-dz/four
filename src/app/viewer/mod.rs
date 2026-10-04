//! Owns viewer state and renders the GPUI interface.

mod decode_scheduler;
mod geometry;
mod render;

use std::cell::OnceCell;
use std::path::Path;
use std::sync::Arc;

use exn::ErrorExt;

use gpui_kit::component::IndexPath;
use gpui_kit::component::menu::ContextMenuExt as _;
use gpui_kit::component::searchable_list::SearchableListItem;
use gpui_kit::component::select::{SelectEvent, SelectState};
use gpui_kit::{
    App, Entity, FocusHandle, Focusable, PathPromptOptions, Pixels, Point, SharedString, Window,
    actions, div, prelude::*, rgb,
};
use tonemapping::ToneMappingMethod;

use decode_scheduler::{
    DecodeJob, DecodePayload, DecodeSource, LatestLoadCoordinator, LoadPurpose, LoadRequest,
};
use geometry::zoom_to_cursor_pan;

use super::image_loader::{
    DisplayedImage, HDROptions, LoadError, LoadResult, LoadedImage, format_load_error, load_image,
    load_image_with, retint_jpeg_xr,
};

const DRAG_REGION_HEIGHT: f32 = 40.0;
const LABEL_ROW_GAP: f32 = 6.0;
const TONE_MAPPING_LABEL_WIDTH: f32 = 100.0;
const TONE_MAPPING_TITLEBAR_WIDTH: f32 = 480.0;

/// Background of the root viewer surface.
const COLOR_APP_BACKGROUND: u32 = 0x0015_1515;
/// Primary text on the root viewer surface.
const COLOR_TEXT_PRIMARY: u32 = 0x00d8_d8d8;
/// Secondary text: field labels and muted captions.
const COLOR_TEXT_SECONDARY: u32 = 0x009d_9d9d;
/// Hint text shown when no image is loaded.
const COLOR_TEXT_HINT: u32 = 0x0088_8888;
/// Background of the status bar strip along the window's bottom edge.
const COLOR_STATUS_BAR_BACKGROUND: u32 = 0x0020_2020;

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

actions!(four, [Quit, OpenFile, ZoomIn, ZoomOut, ZoomReset]);

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
}

pub(super) struct Root {
    decode_coordinator: LatestLoadCoordinator<DecodePayload>,
    /// Last mouse position during an active left-drag pan.
    drag_anchor: Option<Point<Pixels>>,
    /// Lazily created because unit tests construct `Root` without an `App`.
    focus_handle: OnceCell<FocusHandle>,
    last_window_title: Option<SharedString>,
    load_generation: u64,
    /// Pan offset in pixels, relative to the image being centered in the content area.
    pan: Point<Pixels>,
    pending_hdr_options: Option<(LoadRequest, HDROptions)>,
    preferred_hdr_options: HDROptions,
    /// Created by [`Root::init_tone_mapping_select`] because it needs a `Window`.
    tone_mapping_select: Option<Entity<SelectState<Vec<ToneMappingItem>>>>,
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
            decode_coordinator: LatestLoadCoordinator::new(),
            drag_anchor: None,
            focus_handle: OnceCell::new(),
            last_window_title: None,
            load_generation: 0,
            pan: Point::default(),
            pending_hdr_options: None,
            preferred_hdr_options,
            tone_mapping_select: None,
            viewer,
            zoom: 1.0,
        }
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

    /// Updates the window title when it changes.
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

        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open an image".into()),
        });

        cx.spawn_in(window, async move |root, cx| {
            // `Ok(Err(_))` is the platform reporting the dialog itself failed; `Err(_)` is the
            // response channel being dropped before it replied. Both are real failures, unlike
            // `Ok(Ok(None))` (the user just canceled), so both get surfaced the same way.
            let paths = match paths.await {
                Ok(prompt_result) => prompt_result.map_err(|prompt_error| prompt_error.to_string()),
                Err(prompt_error) => Err(prompt_error.to_string()),
            };

            let path = match paths {
                Ok(Some(mut paths)) => paths.pop(),
                Ok(None) => None,
                Err(prompt_error) => {
                    let _ = root.update_in(cx, |root, window, cx| {
                        root.viewer
                            .apply_result(Err(LoadError::new(prompt_error).raise()));
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
                let request = root.begin_load_request();
                root.pending_hdr_options = None;

                root.schedule_decode(
                    DecodeJob {
                        request,
                        payload: DecodePayload {
                            hdr_options: root.preferred_hdr_options,
                            path: Arc::from(path),
                            purpose: LoadPurpose::Image,
                            source: DecodeSource::File,
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
            source,
        } = payload;

        cx.spawn_in(window, async move |root, cx| {
            let result = cx
                .background_spawn(async move {
                    match source {
                        DecodeSource::File => load_image_with(path.as_ref(), hdr_options),
                        DecodeSource::RetainedJpegXr(native) => {
                            retint_jpeg_xr(&native, path.as_ref(), hdr_options)
                        }
                    }
                })
                .await;

            let _ = root.update_in(cx, move |root, window, cx| {
                let next = root.decode_coordinator.complete(request);

                let applied = match purpose {
                    LoadPurpose::Image => root.apply_load_result(request, result),
                    LoadPurpose::HDROptions => root.apply_hdr_options_result(request, result),
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

    /// Applies a current request and resets the context menu.
    fn apply_result_with(&mut self, request: LoadRequest, apply: impl FnOnce(&mut Self)) -> bool {
        if !self.accepts_load_request(request) {
            return false;
        }

        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank before a result is applied"
        );
        apply(self);

        assert!(
            !self.viewer.status().is_empty(),
            "viewer status must never be blank after a result is applied"
        );
        true
    }

    fn apply_load_result(&mut self, request: LoadRequest, result: LoadResult<LoadedImage>) -> bool {
        self.apply_result_with(request, |root| {
            let load_succeeded = result.is_ok();
            root.viewer.apply_result(result);
            if load_succeeded {
                root.reset_zoom();
            }
        })
    }

    fn apply_hdr_options_result(
        &mut self,
        request: LoadRequest,
        result: LoadResult<LoadedImage>,
    ) -> bool {
        self.apply_result_with(request, |root| {
            let displayed_options = root
                .viewer
                .displayed()
                .and_then(|displayed| displayed.hdr_options);

            let resolved_options = result
                .as_ref()
                .ok()
                .and_then(|loaded| loaded.displayed.hdr_options)
                .or(displayed_options);

            root.viewer.apply_result(result);

            root.pending_hdr_options = None;
            if let Some(options) = resolved_options {
                root.preferred_hdr_options = options;
            }
        })
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

    fn select_hdr_options(
        &mut self,
        options: HDROptions,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some((active_options, source_path, native_jpeg_xr)) =
            self.viewer.displayed().and_then(|displayed| {
                displayed.hdr_options.map(|active| {
                    (
                        active,
                        Arc::clone(&displayed.source_path),
                        displayed.native_jpeg_xr.clone(),
                    )
                })
            })
        else {
            cx.notify();
            return;
        };

        let Some(request) = self.begin_hdr_options_selection(options, active_options) else {
            cx.notify();
            return;
        };

        let source = match native_jpeg_xr {
            Some(native) => DecodeSource::RetainedJpegXr(native),
            None => DecodeSource::File,
        };

        self.schedule_decode(
            DecodeJob {
                request,
                payload: DecodePayload {
                    hdr_options: options,
                    path: source_path,
                    purpose: LoadPurpose::HDROptions,
                    source,
                },
            },
            window,
            cx,
        );
        cx.notify();
    }
}

/// A tone-mapping method as listed in the selector.
#[derive(Clone, Copy, Debug)]
pub(super) struct ToneMappingItem(ToneMappingMethod);

impl SearchableListItem for ToneMappingItem {
    type Value = ToneMappingMethod;

    fn title(&self) -> SharedString {
        self.0.label().into()
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }
}

impl Root {
    /// Builds the tone-mapping selector and routes its confirmations to [`Root::select_tone_mapping`].
    pub(super) fn init_tone_mapping_select(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let items = ToneMappingMethod::ALL.map(ToneMappingItem).to_vec();
        let selected = ToneMappingMethod::ALL
            .iter()
            .position(|method| *method == self.preferred_hdr_options.tone_mapping())
            .map(IndexPath::new);
        let select = cx.new(|cx| SelectState::new(items, selected, window, cx));
        cx.subscribe_in(
            &select,
            window,
            |root, _, event: &SelectEvent<Vec<ToneMappingItem>>, window, cx| {
                let SelectEvent::Confirm(Some(method)) = event else {
                    return;
                };
                root.select_tone_mapping(*method, window, cx);
            },
        )
        .detach();
        self.tone_mapping_select = Some(select);
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
        let displayed = self.viewer.displayed();

        let image = displayed.map(|displayed| Arc::clone(&displayed.image));
        let image_dims = displayed.map(|displayed| (displayed.width, displayed.height));
        let active_hdr_options = displayed.and_then(|displayed| displayed.hdr_options);

        let hdr_options = active_hdr_options.map(|active_options| {
            self.pending_hdr_options
                .map_or(active_options, |(_, pending_options)| pending_options)
        });

        let status = self.viewer.status().clone();
        let focus_handle = self.focus_handle(cx);

        if let (Some(select), Some(options)) = (&self.tone_mapping_select, hdr_options) {
            let method = options.tone_mapping();
            select.update(cx, |select, cx| {
                if select.selected_value() != Some(&method) {
                    select.set_selected_value(&method, window, cx);
                }
            });
        }
        let tone_mapping_select = hdr_options.and(self.tone_mapping_select.clone());

        let open_focus = focus_handle.clone();
        div()
            .id("viewer")
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
            .child(Self::render_status_bar(status, tone_mapping_select))
            .child(self.render_image_content(image, image_dims, window, cx))
            .context_menu(move |menu, _, _| {
                menu.action_context(open_focus.clone())
                    .menu("Open image…", Box::new(OpenFile))
                    .menu("Quit", Box::new(Quit))
            })
    }
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
    use gpui_kit::Image as GPUIImage;

    use super::*;

    fn loaded_hdr_viewer(options: HDROptions) -> ViewerState {
        ViewerState::Loaded(LoadedImage {
            displayed: DisplayedImage {
                image: Arc::new(GPUIImage::empty()),
                width: 1,
                height: 1,
                source_path: Arc::from(Path::new("test.jxr")),
                hdr_options: Some(options),
                native_jpeg_xr: None,
            },
            status: "test.jxr".into(),
        })
    }

    #[test]
    fn tone_mapping_defaults_are_bt2446_on_a_fresh_root() {
        let root = Root::new(ViewerState::empty());

        assert_eq!(
            root.preferred_hdr_options.tone_mapping(),
            ToneMappingMethod::BT2446
        );
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

        let current_error = LoadError::new("current load failed").raise();
        assert!(root.apply_load_result(second, Err(current_error)));
        assert!(matches!(root.viewer, ViewerState::Failed { .. }));
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
    fn failed_hdr_reload_clears_the_pending_options() {
        let mut root = Root::new(ViewerState::empty());
        let request = root.begin_load_request();
        root.pending_hdr_options = Some((
            request,
            HDROptions::default().with_tone_mapping(ToneMappingMethod::ACESFitted),
        ));
        let error = LoadError::new("HDR options reload failed").raise();

        assert!(root.apply_hdr_options_result(request, Err(error)));
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
    fn a_second_tone_mapping_change_supersedes_the_first_pending_request() {
        let mut root = Root::new(ViewerState::empty());
        let active = HDROptions::default();
        let first = active.with_tone_mapping(ToneMappingMethod::ACESFitted);

        let first_request = root
            .begin_hdr_options_selection(first, active)
            .expect("a tone-mapping change starts a request");

        let second = active.with_tone_mapping(ToneMappingMethod::Reinhard);

        let second_request = root
            .begin_hdr_options_selection(second, active)
            .expect("a second tone-mapping change starts a new request");

        assert!(!root.accepts_load_request(first_request));
        assert!(root.accepts_load_request(second_request));
        assert_eq!(root.pending_hdr_options, Some((second_request, second)));
        assert_eq!(second.tone_mapping(), ToneMappingMethod::Reinhard);
    }
}
