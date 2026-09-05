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

        if width == 0 {
            return Ok(0);
        }

        let end = self
            .bit_position
            .checked_add(usize::from(width))
            .ok_or_else(|| self.error(ErrorKind::UnexpectedEof))?;
        if end > self.bytes.len().saturating_mul(8) {
            return Err(self.error(ErrorKind::UnexpectedEof));
        }

        let byte_index = self.bit_position / 8;
        let bit_offset = self.bit_position % 8;

        let value = if usize::from(width) + bit_offset <= 64
            && let Some(window) = self.bytes.get(byte_index..byte_index + 8)
        {
            let word = u64::from_be_bytes(window.try_into().expect("eight-byte window"));
            (word << bit_offset) >> (64 - u32::from(width))
        } else {
            self.read_bytewise(width)
        };

        self.bit_position = end;
        Ok(value)
    }

    /// Reads bounds-checked bits near the end of the stream or wider than one word load.
    fn read_bytewise(&self, width: u8) -> u64 {
        let mut value = 0_u64;
        let mut position = self.bit_position;
        let mut remaining = usize::from(width);

        while remaining > 0 {
            let byte = self.bytes[position / 8];
            let available = 8 - position % 8;
            let take = available.min(remaining);
            let chunk = (byte >> (available - take)) & (0xFF_u8 >> (8 - take));
            value = (value << take) | u64::from(chunk);
            position += take;
            remaining -= take;
        }

        value
    }

    /// Returns the next eight bits without consuming them, zero-padded past the end.
    pub(crate) fn peek8(&self) -> u8 {
        let byte_index = self.bit_position / 8;
        let bit_offset = self.bit_position % 8;
        let high = self.bytes.get(byte_index).copied().unwrap_or(0);
        let low = self.bytes.get(byte_index + 1).copied().unwrap_or(0);
        (u16::from_be_bytes([high, low]) << bit_offset).to_be_bytes()[0]
    }

    pub(crate) fn consume(&mut self, width: u8) -> Result<()> {
        let end = self.bit_position + usize::from(width);
        if end > self.bytes.len().saturating_mul(8) {
            return Err(self.error(ErrorKind::UnexpectedEof));
        }

        self.bit_position = end;
        Ok(())
    }

    pub(crate) const fn remaining_bits(&self) -> usize {
        self.bytes.len().saturating_mul(8) - self.bit_position
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
        let byte = self
            .bytes
            .get(self.bit_position / 8)
            .copied()
            .ok_or_else(|| self.error(ErrorKind::UnexpectedEof))?;
        let bit = (byte >> (7 - self.bit_position % 8)) & 1;
        self.bit_position += 1;
        Ok(bit != 0)
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

    pub(crate) fn error(&self, kind: ErrorKind) -> Error {
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

    #[test]
    fn reads_the_stream_tail_and_peeks_past_it() {
        let mut reader = BitReader::new(&[0b1011_0010, 0b0110_1001], 0);
        assert_eq!(reader.read(9), Ok(0b1_0110_0100));
        assert_eq!(reader.peek8(), 0b1101_0010);
        assert_eq!(reader.remaining_bits(), 7);
        assert_eq!(reader.consume(7), Ok(()));
        reader.consume(1).unwrap_err();
        reader.read_bool().unwrap_err();
    }
}
