//! Reads most-significant-bit-first JPEG XR syntax elements.

use crate::error::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_position: usize,
    base_offset: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) const fn new(bytes: &'a [u8], base_offset: usize) -> Self {
        Self {
            bytes,
            bit_position: 0,
            base_offset,
        }
    }

    pub(crate) fn read(&mut self, width: u8) -> Result<u64> {
        debug_assert!(width <= 64, "syntax elements cannot exceed 64 bits");

        let end = self
            .bit_position
            .checked_add(usize::from(width))
            .ok_or_else(|| self.error(ErrorKind::UnexpectedEof))?;
        if end > self.bytes.len().saturating_mul(8) {
            return Err(self.error(ErrorKind::UnexpectedEof));
        }

        let mut value = 0_u64;
        while self.bit_position < end {
            let byte = self.bytes[self.bit_position / 8];
            let bit = (byte >> (7 - self.bit_position % 8)) & 1;
            value = (value << 1) | u64::from(bit);
            self.bit_position += 1;
        }

        Ok(value)
    }

    pub(crate) fn read_u8(&mut self, width: u8) -> Result<u8> {
        u8::try_from(self.read(width)?).map_err(|_conversion_error| {
            self.error(ErrorKind::InvalidCodestream(
                "syntax element does not fit u8",
            ))
        })
    }

    pub(crate) fn read_u16(&mut self, width: u8) -> Result<u16> {
        u16::try_from(self.read(width)?).map_err(|_conversion_error| {
            self.error(ErrorKind::InvalidCodestream(
                "syntax element does not fit u16",
            ))
        })
    }

    pub(crate) fn read_u32(&mut self, width: u8) -> Result<u32> {
        u32::try_from(self.read(width)?).map_err(|_conversion_error| {
            self.error(ErrorKind::InvalidCodestream(
                "syntax element does not fit u32",
            ))
        })
    }

    pub(crate) fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read(1)? != 0)
    }

    pub(crate) fn align_zero(&mut self) -> Result<()> {
        while !self.bit_position.is_multiple_of(8) {
            if self.read(1)? != 0 {
                return Err(self.error(ErrorKind::InvalidCodestream(
                    "byte-alignment bit must be zero",
                )));
            }
        }

        Ok(())
    }

    pub(crate) const fn byte_position(&self) -> usize {
        self.bit_position.div_ceil(8)
    }

    pub(crate) const fn absolute_offset(&self) -> usize {
        self.base_offset + self.bit_position / 8
    }

    pub(crate) const fn error(&self, kind: ErrorKind) -> Error {
        Error::new(kind, self.absolute_offset())
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;

    #[test]
    fn reads_across_byte_boundaries() {
        let mut reader = BitReader::new(&[0b1011_0010, 0b0110_1001], 7);
        assert_eq!(reader.read(3), Ok(0b101));
        assert_eq!(reader.read(7), Ok(0b100_1001));
        assert_eq!(reader.read(6), Ok(0b10_1001));
        assert_eq!(reader.absolute_offset(), 9);
    }
}
