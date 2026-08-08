use gpui::{
    App, Bounds, Context, MouseButton, MouseDownEvent, Pixels, Point, Window, WindowBounds,
    WindowControlArea, WindowOptions, deferred, div, prelude::*, px, rgb, size,
};
use gpui_platform::application;

const CONTEXT_MENU_HEIGHT: f32 = 40.0;
const CONTEXT_MENU_WIDTH: f32 = 180.0;
const DRAG_REGION_HEIGHT: f32 = 40.0;

struct Root {
    context_menu_position: Option<Point<Pixels>>,
}

impl Render for Root {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let context_menu_position = self.context_menu_position;

        div()
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    let mut position = event.position;
                    let viewport_size = window.viewport_size();

                    position.x = position.x.min(viewport_size.width - px(CONTEXT_MENU_WIDTH));
                    position.y = position
                        .y
                        .min(viewport_size.height - px(CONTEXT_MENU_HEIGHT));

                    position.x = position.x.max(px(0.0));
                    position.y = position.y.max(px(DRAG_REGION_HEIGHT));

                    this.context_menu_position = Some(position);
                    cx.notify();
                }),
            )
            .child(
                div()
                    .w_full()
                    .h(px(DRAG_REGION_HEIGHT))
                    .window_control_area(WindowControlArea::Drag),
            )
            .when_some(context_menu_position, |root, position| {
                root.child(
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
                            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                                this.context_menu_position = None;
                                cx.notify();
                            }))
                            .child(
                                div()
                                    .id("close-application")
                                    .size_full()
                                    .flex()
                                    .items_center()
                                    .px_2()
                                    .rounded_sm()
                                    .cursor_pointer()
                                    .text_sm()
                                    .text_color(rgb(0xffffff))
                                    .hover(|style| style.bg(rgb(0x3d3d3d)))
                                    .on_click(|_, _, cx| cx.quit())
                                    .child("Quit"),
                            ),
                    )
                    .priority(1),
                )
            })
    }
}

fn main() {
    application().run(|cx: &mut App| {
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
            |_window, cx| {
                cx.new(|_| Root {
                    context_menu_position: None,
                })
            },
        )
        .expect("failed to open window");
    });
}
