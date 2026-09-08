//! Builds the GPUI element tree: context menu, tone-mapping controls, image surface, status bar,
//! and the mouse handlers that back the image surface.

use std::sync::Arc;

use gpui::{
    Anchor, AnchoredPositionMode, CursorStyle, Image as GPUIImage, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, Point, Role, ScrollWheelEvent, SharedString, Toggled,
    Window, WindowControlArea, anchored, deferred, div, img, point, prelude::*, px, rgb, rgba,
};
use tonemapping::{MaxCLLMode, ToneMappingMethod};

use super::geometry::{clamp_pan, fit_scale, zoom_to_cursor_pan};
use super::{
    COLOR_ACCENT_GREEN, COLOR_CHECKBOX_BORDER, COLOR_CONTROL_BACKGROUND, COLOR_CONTROL_BORDER,
    COLOR_CONTROL_HOVER, COLOR_MENU_ITEM_HOVER, COLOR_METADATA_OVERLAY_BACKGROUND,
    COLOR_PANEL_BACKGROUND, COLOR_PANEL_BORDER, COLOR_SELECTED_BACKGROUND, COLOR_TEXT_HINT,
    COLOR_TEXT_MENU, COLOR_TEXT_SECONDARY, CONTEXT_MENU_HEIGHT, CONTEXT_MENU_ITEM_HEIGHT,
    CONTEXT_MENU_WIDTH, DRAG_REGION_HEIGHT, HDROptions, MAX_CLL_CHECKBOX_SIZE,
    MAX_CLL_SELECTOR_HEIGHT, METADATA_FIELD_GAP, METADATA_LABEL_WIDTH, METADATA_OVERLAY_MARGIN,
    Root, SCROLL_LINE_HEIGHT, TONE_MAPPING_LABEL_WIDTH, TONE_MAPPING_MENU_ITEM_HEIGHT,
    TONE_MAPPING_MENU_MARGIN, TONE_MAPPING_MENU_WIDTH, TONE_MAPPING_SELECTOR_HEIGHT,
    TONE_MAPPING_TITLEBAR_WIDTH, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP_BASE,
};

impl Root {
    pub(super) fn render_context_menu(
        position: Point<Pixels>,
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
                .h(px(CONTEXT_MENU_HEIGHT))
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
                .child(menu_item("quit", "Quit").on_click(|_, _, cx| cx.quit())),
        )
        .priority(1)
    }

    pub(super) fn render_tone_mapping_selector(
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

    pub(super) fn render_max_cll_selector(
        selected_mode: MaxCLLMode,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
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

    pub(super) fn render_tone_mapping_menu(
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

    pub(super) fn render_status_bar(
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

pub(super) const fn toggled_max_cll_mode(mode: MaxCLLMode) -> MaxCLLMode {
    match mode {
        MaxCLLMode::Percentile99_99 => MaxCLLMode::TrueMaximum,
        MaxCLLMode::TrueMaximum => MaxCLLMode::Percentile99_99,
    }
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
