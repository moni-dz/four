//! Defines the shared decoded-image representation and format decoders.

use nutype::nutype;
use rayon::prelude::*;

pub mod gif;
pub mod jpeg;
pub mod jpeg_xl;
pub mod jpeg_xr;
pub mod png;
pub mod tiff;

const BMP_FILE_HEADER_BYTES: u32 = 14;
const BMP_DIB_HEADER_BYTES: u32 = 108;
const BMP_HEADER_BYTES: u32 = BMP_FILE_HEADER_BYTES + BMP_DIB_HEADER_BYTES;
const BMP_BITFIELDS_COMPRESSION: u32 = 3;
const BMP_RED_MASK: u32 = 0x00ff_0000;
const BMP_GREEN_MASK: u32 = 0x0000_ff00;
const BMP_BLUE_MASK: u32 = 0x0000_00ff;
const BMP_ALPHA_MASK: u32 = 0xff00_0000;
const BMP_SRGB_COLOR_SPACE: u32 = 0x7352_4742;
const DIMENSION_MAX: u32 = 16_384;
const PIXELS_MAX: u64 = 64 * 1024 * 1024;
const RGBA_BYTES_PER_PIXEL: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DimensionsError {
    Zero,
    TooLarge { width: u32, height: u32 },
    TooManyPixels { pixels: u64 },
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "nutype's validate(with = ...) always invokes the function with a reference to the \
              wrapped value, regardless of whether that value is Copy"
)]
fn validate_dimensions_pair(&(width, height): &(u32, u32)) -> Result<(), DimensionsError> {
    if width == 0 || height == 0 {
        return Err(DimensionsError::Zero);
    }
    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(DimensionsError::TooLarge { width, height });
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > PIXELS_MAX {
        return Err(DimensionsError::TooManyPixels { pixels });
    }
    Ok(())
}

/// A `(width, height)` pair already known to be nonzero and within [`DIMENSION_MAX`]/[`PIXELS_MAX`].
///
/// Collapses the six format decoders' near-identical `validate_dimensions` checks into one
/// validated construction; each decoder still maps [`DimensionsError`] into its own error type.
#[nutype(
    validate(with = validate_dimensions_pair, error = DimensionsError),
    derive(Clone, Copy, Debug, PartialEq, Eq)
)]
pub(crate) struct Dimensions((u32, u32));

/// Pixel count below which a decoder normalizes on the calling thread.
///
/// Spawning rayon jobs costs more than it saves for a small image, and the viewer opens far more
/// small images than large ones. A quarter of a megapixel is roughly where the two balance on a
/// typical desktop; it is a threshold, not a measured optimum, so moving it changes throughput
/// rather than correctness.
const PARALLEL_PIXELS_MIN: usize = 256 * 1024;

/// Pixels per rayon job once a decoder does go parallel.
///
/// Large enough that per-job overhead is negligible, small enough that a four-megapixel image
/// still splits into enough jobs to fill a many-core machine.
const PARALLEL_PIXELS_PER_JOB: usize = 64 * 1024;

/// Owned RGBA8 pixels produced by one of our format parsers.
///
/// Keeping this type independent of GPUI makes the parsers usable in tests and keeps the boundary
/// between decoding and display explicit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl DecodedImage {
    pub(crate) fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        invariant!(width > 0);
        invariant!(height > 0);

        let pixel_count = u64::from(width) * u64::from(height);
        let byte_count = u64::try_from(rgba.len())
            .expect("a Rust allocation length always fits the decoder's u64 accounting");
        invariant_eq!(byte_count, pixel_count * u64::from(RGBA_BYTES_PER_PIXEL));
        Self {
            width,
            height,
            rgba,
        }
    }

    /// Returns the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns row-major pixels with four bytes per pixel in RGBA order.
    #[must_use]
    pub fn rgba8(&self) -> &[u8] {
        &self.rgba
    }

    /// Returns the image dimensions as `(width, height)`.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Builds a row-major RGBA8 buffer by calling `pixel(x, y)` for every coordinate, going parallel
/// (one rayon job per row) once the image is large enough that the dispatch overhead pays for
/// itself, and running on the calling thread otherwise.
///
/// Shared by every decoder that reconstructs pixels from an existing sample buffer by coordinate
/// (JPEG, TIFF) rather than by walking a source byte buffer directly.
pub(crate) fn rgba_pixel_rows<E: Send>(
    width: usize,
    height: usize,
    pixel: impl Fn(usize, usize) -> Result<[u8; 4], E> + Sync,
) -> Result<Vec<u8>, E> {
    let pixel_count = width * height;
    let byte_count = pixel_count * usize::try_from(RGBA_BYTES_PER_PIXEL)
        .expect("four bytes per pixel always fits usize");

    if pixel_count >= PARALLEL_PIXELS_MIN {
        let mut rgba = vec![0; byte_count];
        rgba.par_chunks_mut(width * 4)
            .with_min_len(PARALLEL_PIXELS_PER_JOB / width.max(1))
            .enumerate()
            .try_for_each(|(y, row)| {
                let (targets, remainder) = row.as_chunks_mut::<4>();
                invariant_eq!(remainder.len(), 0);

                for (x, target) in targets.iter_mut().enumerate() {
                    *target = pixel(x, y)?;
                }
                Ok(())
            })?;

        Ok(rgba)
    } else {
        let mut rgba = Vec::with_capacity(byte_count);

        for y in 0..height {
            for x in 0..width {
                rgba.extend_from_slice(&pixel(x, y)?);
            }
        }

        Ok(rgba)
    }
}

/// GPUI accepts encoded images, so an uncompressed BMP is used only as a pixel carrier.
///
/// A V4 header declares explicit BGRA channel masks. Without those masks, BMP readers commonly
/// interpret the fourth byte of a 32-bit `BI_RGB` pixel as padding and discard image transparency.
/// The source format has already been fully decoded before this adapter runs. See Microsoft's
/// [`BITMAPV4HEADER`](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapv4header)
/// reference for the header and mask fields.
///
/// # Panics
///
/// Panics only if `image` violates internal [`DecodedImage`] invariants: its dimensions or encoded
/// size do not fit the BMP fields and address space, or its RGBA length does not match its
/// dimensions.
#[must_use]
pub fn encode_bmp(image: &DecodedImage) -> Vec<u8> {
    let (width, height) = image.dimensions();
    let width_i32 = i32::try_from(width).expect("BMP width must fit its signed 32-bit field");
    let height_i32 = i32::try_from(height).expect("BMP height must fit its signed 32-bit field");

    let pixel_bytes = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(RGBA_BYTES_PER_PIXEL))
        .expect("decoded image size was validated");

    let file_bytes = BMP_HEADER_BYTES
        .checked_add(pixel_bytes)
        .expect("decoded image size was validated");

    let pixel_bytes_usize =
        usize::try_from(pixel_bytes).expect("the validated decoded image allocation fits usize");

    let file_bytes_usize =
        usize::try_from(file_bytes).expect("the validated BMP allocation fits usize");

    assert_eq!(
        image.rgba8().len(),
        pixel_bytes_usize,
        "image RGBA byte count does not match its dimensions"
    );

    let mut bmp = Vec::with_capacity(file_bytes_usize);

    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&file_bytes.to_le_bytes());
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&BMP_HEADER_BYTES.to_le_bytes());

    bmp.extend_from_slice(&BMP_DIB_HEADER_BYTES.to_le_bytes());
    bmp.extend_from_slice(&width_i32.to_le_bytes());
    bmp.extend_from_slice(&(-height_i32).to_le_bytes());
    bmp.extend_from_slice(&1_u16.to_le_bytes());
    bmp.extend_from_slice(&32_u16.to_le_bytes());
    bmp.extend_from_slice(&BMP_BITFIELDS_COMPRESSION.to_le_bytes());
    bmp.extend_from_slice(&pixel_bytes.to_le_bytes());
    bmp.extend_from_slice(&[0; 16]);

    bmp.extend_from_slice(&BMP_RED_MASK.to_le_bytes());
    bmp.extend_from_slice(&BMP_GREEN_MASK.to_le_bytes());
    bmp.extend_from_slice(&BMP_BLUE_MASK.to_le_bytes());
    bmp.extend_from_slice(&BMP_ALPHA_MASK.to_le_bytes());
    bmp.extend_from_slice(&BMP_SRGB_COLOR_SPACE.to_le_bytes());
    bmp.extend_from_slice(&[0; 48]);

    let (pixels, remainder) = image
        .rgba8()
        .as_chunks::<{ RGBA_BYTES_PER_PIXEL as usize }>();
    invariant_eq!(remainder.len(), 0);
    bmp.extend(
        pixels
            .iter()
            .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]]),
    );

    invariant_eq!(bmp.len(), file_bytes_usize);
    bmp
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        BMP_DIB_HEADER_BYTES, BMP_HEADER_BYTES, DIMENSION_MAX, DecodedImage, Dimensions,
        DimensionsError, PIXELS_MAX, encode_bmp,
    };

    #[test]
    fn bmp_encoding_writes_top_down_bgra_pixels() {
        let image = DecodedImage::new(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let bmp = encode_bmp(&image);

        assert_eq!(&bmp[..2], b"BM");
        assert_eq!(&bmp[10..14], &BMP_HEADER_BYTES.to_le_bytes());
        assert_eq!(&bmp[14..18], &BMP_DIB_HEADER_BYTES.to_le_bytes());
        assert_eq!(&bmp[18..22], &2_i32.to_le_bytes());
        assert_eq!(&bmp[22..26], &(-1_i32).to_le_bytes());
        assert_eq!(&bmp[BMP_HEADER_BYTES as usize..], &[3, 2, 1, 4, 7, 6, 5, 8]);
    }

    #[test]
    fn bmp_encoding_preserves_alpha_through_gpui() {
        let image = DecodedImage::new(1, 1, vec![1, 2, 3, 4]);
        let carrier = gpui::Image::from_bytes(gpui::ImageFormat::Bmp, encode_bmp(&image));
        let decoded = carrier
            .to_image_data(gpui::SvgRenderer::new(Arc::new(())))
            .expect("the BMP carrier must decode through GPUI");

        assert_eq!(
            decoded
                .as_bytes(0)
                .expect("the static BMP carrier has one frame"),
            &[3, 2, 1, 4]
        );
    }

    #[test]
    fn dimensions_accepts_every_in_bounds_pair() {
        assert!(Dimensions::try_new((1, 1)).is_ok());
        // DIMENSION_MAX on one side alone still fits PIXELS_MAX; DIMENSION_MAX on both sides does
        // not (see `dimensions_rejects_a_pixel_count_above_the_max`).
        assert!(Dimensions::try_new((DIMENSION_MAX, 1)).is_ok());
    }

    #[test]
    fn dimensions_rejects_a_zero_width_or_height() {
        assert_eq!(
            Dimensions::try_new((0, 1)).unwrap_err(),
            DimensionsError::Zero
        );
        assert_eq!(
            Dimensions::try_new((1, 0)).unwrap_err(),
            DimensionsError::Zero
        );
    }

    #[test]
    fn dimensions_rejects_a_dimension_above_the_max() {
        assert_eq!(
            Dimensions::try_new((DIMENSION_MAX + 1, 1)).unwrap_err(),
            DimensionsError::TooLarge {
                width: DIMENSION_MAX + 1,
                height: 1
            }
        );
    }

    #[test]
    fn dimensions_rejects_a_pixel_count_above_the_max() {
        // Both dimensions individually fit DIMENSION_MAX, but their product does not fit PIXELS_MAX.
        let side = DIMENSION_MAX;
        assert!(u64::from(side) * u64::from(side) > PIXELS_MAX);

        assert_eq!(
            Dimensions::try_new((side, side)).unwrap_err(),
            DimensionsError::TooManyPixels {
                pixels: u64::from(side) * u64::from(side)
            }
        );
    }
}
