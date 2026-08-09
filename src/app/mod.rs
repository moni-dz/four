//! Runs the image viewer application.

use std::path::Path;

use gpui::{App, AppContext, Bounds, WindowBounds, WindowOptions, px, size};
use gpui_platform::application;

use self::viewer::{Root, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH, initial_viewer};

// Keep invariant diagnostics consistent with the decoder crate while leaving test assertions free
// to use the standard macros, where custom panic messages add little value.
macro_rules! invariant {
    ($condition:expr, $($message:tt)+) => {
        assert!($condition, $($message)+)
    };
    ($condition:expr $(,)?) => {
        assert!(
            $condition,
            concat!("invariant failed: ", stringify!($condition))
        )
    };
}

mod image_loader;
mod viewer;

/// Runs the viewer with an optional image selected at startup.
pub(crate) fn run(initial_path: Option<&Path>) {
    let initial_viewer = initial_viewer(initial_path);

    application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.0), px(600.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: None,
                is_movable: true,
                window_min_size: Some(size(px(WINDOW_MIN_WIDTH), px(WINDOW_MIN_HEIGHT))),
                ..Default::default()
            },
            |_window, cx| cx.new(|_| Root::new(initial_viewer)),
        )
        .expect("failed to open window");
    });
}
