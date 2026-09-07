//! Pure zoom/pan math shared by the render tree and the top-level viewer state.

use gpui::{Pixels, Point, point, px};

/// Scale that fits an `image_w`×`image_h` image inside `content_w`×`content_h`, preserving
/// aspect ratio (matches gpui's `ObjectFit::Contain`, which this replaces).
#[expect(
    clippy::cast_precision_loss,
    reason = "image dimensions stay far below f32's 2^24 exact-integer range"
)]
pub(super) fn fit_scale(content_w: Pixels, content_h: Pixels, image_w: u32, image_h: u32) -> f32 {
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
pub(super) fn clamp_pan(
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
pub(super) fn zoom_to_cursor_pan(
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
