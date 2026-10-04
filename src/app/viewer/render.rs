//! Builds the GPUI element tree: tone-mapping controls, image surface, status bar,
//! and the mouse handlers that back the image surface.

use std::sync::Arc;

use gpui_kit::component::select::{Select, SelectState};
use gpui_kit::{
    CursorStyle, Entity, Image as GPUIImage, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, Pixels, ScrollWheelEvent, SharedString, Window, WindowControlArea, div, img,
    point, prelude::*, px, rgb,
};

use super::geometry::{clamp_pan, fit_scale, zoom_to_cursor_pan};
use super::{
    COLOR_STATUS_BAR_BACKGROUND, COLOR_TEXT_HINT, COLOR_TEXT_SECONDARY, DRAG_REGION_HEIGHT,
    LABEL_ROW_GAP, Root, SCROLL_LINE_HEIGHT, TONE_MAPPING_LABEL_WIDTH, TONE_MAPPING_TITLEBAR_WIDTH,
    ToneMappingItem, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP_BASE,
};

impl Root {
    pub(super) fn on_image_scroll(
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

    pub(super) fn on_image_drag_start(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        if (self.zoom - 1.0).abs() > f32::EPSILON {
            self.drag_anchor = Some(event.position);
            cx.notify();
        }
    }

    pub(super) fn on_image_drag_move(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
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

    pub(super) fn on_image_drag_end(&mut self, cx: &mut Context<Self>) {
        if self.drag_anchor.take().is_some() {
            cx.notify();
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions stay far below f32's 2^24 exact-integer range"
    )]
    pub(super) fn render_image_content(
        &mut self,
        image: Option<Arc<GPUIImage>>,
        image_dims: Option<(u32, u32)>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> gpui_kit::Div {
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

    pub(super) fn render_status_bar(
        status: SharedString,
        tone_mapping_select: Option<Entity<SelectState<Vec<ToneMappingItem>>>>,
    ) -> gpui_kit::Div {
        div()
            .w_full()
            .h(px(DRAG_REGION_HEIGHT))
            .flex_none()
            .flex()
            .items_center()
            .text_sm()
            .bg(rgb(COLOR_STATUS_BAR_BACKGROUND))
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
            .when_some(tone_mapping_select, |titlebar, select| {
                titlebar.child(
                    div()
                        .w(px(TONE_MAPPING_TITLEBAR_WIDTH))
                        .h_full()
                        .flex_none()
                        .flex()
                        .items_center()
                        .gap(px(LABEL_ROW_GAP))
                        .px_3()
                        .child(
                            div()
                                .w(px(TONE_MAPPING_LABEL_WIDTH))
                                .flex_none()
                                .text_color(rgb(COLOR_TEXT_SECONDARY))
                                .child("Tone mapper"),
                        )
                        .child(
                            div().min_w_0().flex_1().child(
                                Select::new(&select)
                                    .id("tone-mapping-selector")
                                    .accessibility_label("Tone mapper"),
                            ),
                        ),
                )
            })
    }
}
