#![feature(f16, f32_from_f16, portable_simd)]
#![warn(missing_docs)]
//! Decodes GIF, JPEG, JPEG XL, JPEG XR, PNG, and TIFF images into RGBA8 pixels.
//! Callers supply encoded bytes and control I/O and resource policy.

// Every invariant names the exact expression in its panic message. Centralizing that mechanical
// part prevents a future assertion from silently losing the diagnostic.
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

// Every format's error module needs the same `Error`/`Result` aliases, the `std::error::Error`
// `source()` impl, and a raise helper that asserts its message is non-empty. Only how `Codec`
// exposes its source varies (boxed vs. bare error), so that's the one thing callers supply.
macro_rules! format_error_boilerplate {
    ($name:literal, $error:ty, source = |$source:ident| $source_expr:expr) => {
        /// An error with propagation frames.
        pub type Error = exn::Exn<$error>;
        /// Result returned by this format's decoder operations.
        pub type Result<T> = exn::Result<T, $error>;

        impl std::error::Error for $error {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                match self {
                    Self::Codec($source) => $source_expr,
                    _ => None,
                }
            }
        }

        /// Creates a leaf error at its validation site.
        #[track_caller]
        pub(super) fn error(error: $error) -> Error {
            invariant!(
                !error.to_string().is_empty(),
                concat!("a ", $name, " error must have a useful display message")
            );
            error.raise()
        }
    };
}

mod image_formats;

#[doc(inline)]
pub use image_formats::{DecodedImage, encode_bmp, gif, jpeg, jpeg_xl, jpeg_xr, png, tiff};
