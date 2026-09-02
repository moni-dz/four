//! Runs the image viewer application.

use std::path::Path;

use gpui::{App, AppContext, Bounds, Focusable, KeyBinding, WindowBounds, WindowOptions, px, size};
use gpui_platform::application;

use self::viewer::{
    DismissMenu, OpenFile, Quit, Root, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH, ZoomIn, ZoomOut,
    ZoomReset, initial_viewer,
};

mod image_loader;
mod viewer;

/// Runs the viewer with an optional image selected at startup.
pub(crate) fn run(initial_path: Option<&Path>) {
    let initial_viewer = initial_viewer(initial_path);

    application().run(move |cx: &mut App| {
        cx.on_action(|_: &Quit, cx| cx.quit());
        cx.bind_keys([
            KeyBinding::new("secondary-q", Quit, None),
            KeyBinding::new("secondary-o", OpenFile, Some("Viewer")),
            KeyBinding::new("secondary-=", ZoomIn, Some("Viewer")),
            KeyBinding::new("secondary-shift-=", ZoomIn, Some("Viewer")),
            KeyBinding::new("secondary--", ZoomOut, Some("Viewer")),
            KeyBinding::new("secondary-0", ZoomReset, Some("Viewer")),
            KeyBinding::new("escape", DismissMenu, Some("Viewer")),
        ]);

        // `window_min_size` below is only enforced by the OS on interactive border-drag resizing,
        // not at creation, so the initial size must already respect it.
        let bounds = Bounds::centered(None, size(px(WINDOW_MIN_WIDTH), px(WINDOW_MIN_HEIGHT)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: None,
                is_movable: true,
                window_min_size: Some(size(px(WINDOW_MIN_WIDTH), px(WINDOW_MIN_HEIGHT))),
                ..Default::default()
            },
            |window, cx| {
                let root = cx.new(|_| Root::new(initial_viewer));
                root.update(cx, |root, _cx| root.sync_window_title(window));
                root.focus_handle(cx).focus(window, cx);
                root
            },
        )
        .expect("failed to open window");
    });
}
