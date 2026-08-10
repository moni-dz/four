#![feature(portable_simd)]
#![warn(missing_docs)]
//! Decodes images into a small, format-independent RGBA representation.
//!
//! The crate provides GIF, JPEG, JPEG XL, JPEG XR, PNG, and TIFF decoding into RGBA8 pixels.
//! Maintained codecs handle ordinary formats while the handwritten JPEG implementation preserves
//! arithmetic entropy decoding. Decoding is sans-I/O: callers supply encoded bytes and retain
//! control over file, network, and resource policy.

// Every invariant names the exact expression in its panic message. Centralizing that mechanical
// part prevents a future assertion from silently losing the diagnostic required by AGENTS.md.
macro_rules! invariant {
    ($condition:expr, $($message:tt)+) => {
        debug_assert!($condition, $($message)+)
    };
    ($condition:expr $(,)?) => {
        debug_assert!(
            $condition,
            concat!("invariant failed: ", stringify!($condition))
        )
    };
}

macro_rules! invariant_eq {
    ($left:expr, $right:expr, $($message:tt)+) => {
        debug_assert_eq!($left, $right, $($message)+)
    };
    ($left:expr, $right:expr $(,)?) => {
        debug_assert_eq!(
            $left,
            $right,
            concat!(
                "invariant failed: ",
                stringify!($left),
                " == ",
                stringify!($right)
            )
        )
    };
}

macro_rules! invariant_ne {
    ($left:expr, $right:expr, $($message:tt)+) => {
        debug_assert_ne!($left, $right, $($message)+)
    };
    ($left:expr, $right:expr $(,)?) => {
        debug_assert_ne!(
            $left,
            $right,
            concat!(
                "invariant failed: ",
                stringify!($left),
                " != ",
                stringify!($right)
            )
        )
    };
}

mod image_formats;

#[doc(inline)]
pub use image_formats::{DecodedImage, encode_bmp, gif, jpeg, jpeg_xl, jpeg_xr, png, tiff};
