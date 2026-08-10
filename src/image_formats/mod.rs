//! Defines the shared decoded-image representation and format decoders.

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

/// GPUI accepts encoded images, so an uncompressed BMP is used only as a pixel carrier.
///
/// A V4 header declares explicit BGRA channel masks. Without those masks, BMP readers commonly
/// interpret the fourth byte of a 32-bit `BI_RGB` pixel as padding and discard image transparency.
/// The source format has already been fully decoded before this adapter runs.
///
/// # Panics
///
/// Panics when the dimensions exceed BMP's signed 32-bit fields.
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

    use super::{BMP_DIB_HEADER_BYTES, BMP_HEADER_BYTES, DecodedImage, encode_bmp};

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
}
