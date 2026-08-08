pub mod jpeg;

const BMP_HEADER_BYTES: u32 = 54;
const RGBA_BYTES_PER_PIXEL: u32 = 4;

/// Pixels decoded by one of our format parsers.
///
/// Keeping this type independent of GPUI makes the parsers usable in tests and keeps the boundary
/// between decoding and display explicit.
#[derive(Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl DecodedImage {
    pub(crate) fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        assert!(width > 0);
        assert!(height > 0);

        let pixel_count = u64::from(width) * u64::from(height);
        assert_eq!(
            rgba.len() as u64,
            pixel_count * u64::from(RGBA_BYTES_PER_PIXEL)
        );
        Self {
            width,
            height,
            rgba,
        }
    }

    /// GPUI accepts encoded images, so a trivial uncompressed BMP is used only as a pixel carrier.
    /// The JPEG bytes have already been fully decoded by our parser before this conversion runs.
    pub fn into_bmp(self) -> Vec<u8> {
        assert!(self.width <= i32::MAX as u32);
        assert!(self.height <= i32::MAX as u32);

        let pixel_bytes = self
            .width
            .checked_mul(self.height)
            .and_then(|count| count.checked_mul(RGBA_BYTES_PER_PIXEL))
            .expect("decoded image size was validated");
        let file_bytes = BMP_HEADER_BYTES
            .checked_add(pixel_bytes)
            .expect("decoded image size was validated");
        let mut bmp = Vec::with_capacity(file_bytes as usize);

        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&file_bytes.to_le_bytes());
        bmp.extend_from_slice(&[0; 4]);
        bmp.extend_from_slice(&BMP_HEADER_BYTES.to_le_bytes());
        bmp.extend_from_slice(&40_u32.to_le_bytes());
        bmp.extend_from_slice(&(self.width as i32).to_le_bytes());
        bmp.extend_from_slice(&(-(self.height as i32)).to_le_bytes());
        bmp.extend_from_slice(&1_u16.to_le_bytes());
        bmp.extend_from_slice(&32_u16.to_le_bytes());
        bmp.extend_from_slice(&0_u32.to_le_bytes());
        bmp.extend_from_slice(&pixel_bytes.to_le_bytes());
        bmp.extend_from_slice(&[0; 16]);

        let (pixels, remainder) = self.rgba.as_chunks::<{ RGBA_BYTES_PER_PIXEL as usize }>();
        assert!(remainder.is_empty());
        for pixel in pixels {
            bmp.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
        assert_eq!(bmp.len(), file_bytes as usize);
        bmp
    }
}
