use super::{Error, Result};

pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        let reader = Self { bytes, offset: 0 };
        assert_eq!(reader.offset, 0);
        assert_eq!(reader.remaining(), bytes.len());
        reader
    }

    pub(super) fn remaining(&self) -> usize {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);
        self.bytes.len() - self.offset
    }

    pub(super) fn remaining_slice(&self) -> &'a [u8] {
        assert!(self.offset <= self.bytes.len());
        assert_eq!(self.remaining(), self.bytes.len() - self.offset);
        &self.bytes[self.offset..]
    }

    pub(super) fn advance(&mut self, length: usize) -> Result<()> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Error::new("JPEG input offset overflowed"))?;
        if end > self.bytes.len() {
            return Err(Error::new("JPEG input offset extends past the input"));
        }
        self.offset = end;
        Ok(())
    }

    pub(super) fn read_u8(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let value = self
            .bytes
            .get(self.offset)
            .copied()
            .ok_or_else(|| Error::new("unexpected end of JPEG data"))?;
        self.offset += 1;
        Ok(value)
    }

    pub(super) fn read_u16(&mut self) -> Result<u16> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let high = u16::from(self.read_u8()?);
        let low = u16::from(self.read_u8()?);
        Ok((high << 8) | low)
    }

    pub(super) fn read_slice(&mut self, length: usize) -> Result<&'a [u8]> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Error::new("JPEG segment length overflowed"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| Error::new("JPEG segment extends past the input"))?;
        self.offset = end;
        Ok(slice)
    }

    pub(super) fn segment(&mut self) -> Result<Reader<'a>> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let length = usize::from(self.read_u16()?);
        if length < 2 {
            return Err(Error::new("JPEG segment length is smaller than its header"));
        }
        Ok(Reader::new(self.read_slice(length - 2)?))
    }

    pub(super) fn marker(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        if self.read_u8()? != 0xff {
            return Err(Error::new("expected a JPEG marker"));
        }
        let mut marker = self.read_u8()?;
        while marker == 0xff {
            marker = self.read_u8()?;
        }
        if marker == 0x00 {
            return Err(Error::new("stuffed byte appeared outside entropy data"));
        }
        Ok(marker)
    }
}

pub(super) struct BitReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    reservoir: u32,
    bits_remaining: u8,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        let reader = Self {
            bytes,
            offset: 0,
            reservoir: 0,
            bits_remaining: 0,
        };
        assert_eq!(reader.offset, 0);
        assert_eq!(reader.bits_remaining, 0);
        reader
    }

    pub(super) fn read_bits(&mut self, count: u8) -> Result<u16> {
        assert!(count <= 16);
        assert!(self.bits_remaining <= 23);

        if self.peek_bits(count)?.is_none() {
            return Err(self.entropy_boundary_error());
        }
        let value = self.peek_filled(count);
        self.consume_bits(count);
        Ok(value)
    }

    /// A non-consuming prefix is what lets the Huffman decoder use one table lookup for common
    /// short codes. `None` preserves a real marker for the scan/restart parser.
    pub(super) fn peek_bits(&mut self, count: u8) -> Result<Option<u16>> {
        assert!(count <= 16);
        assert!(self.bits_remaining <= 23);

        if count == 0 {
            return Ok(Some(0));
        }
        if !self.fill(count)? {
            return Ok(None);
        }
        Ok(Some(self.peek_filled(count)))
    }

    pub(super) fn consume_bits(&mut self, count: u8) {
        assert!(count <= self.bits_remaining);
        assert!(self.bits_remaining <= 23);

        self.bits_remaining -= count;
        if self.bits_remaining == 0 {
            self.reservoir = 0;
        } else {
            self.reservoir &= (1_u32 << self.bits_remaining) - 1;
        }
    }

    fn fill(&mut self, count: u8) -> Result<bool> {
        assert!(count <= 16);
        assert!(self.bits_remaining <= 23);

        while self.bits_remaining < count {
            let Some(byte) = self.entropy_byte()? else {
                return Ok(false);
            };
            self.reservoir = (self.reservoir << 8) | u32::from(byte);
            self.bits_remaining += 8;
        }
        Ok(true)
    }

    fn peek_filled(&self, count: u8) -> u16 {
        assert!(count <= self.bits_remaining);
        assert!(count <= 16);

        if count == 0 {
            return 0;
        }
        let shift = self.bits_remaining - count;
        let mask = (1_u32 << count) - 1;
        ((self.reservoir >> shift) & mask) as u16
    }

    fn entropy_byte(&mut self) -> Result<Option<u8>> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bits_remaining <= 23);

        let Some(byte) = self.bytes.get(self.offset).copied() else {
            return Ok(None);
        };
        if byte != 0xff {
            self.offset += 1;
            return Ok(Some(byte));
        }
        let Some(next) = self.bytes.get(self.offset + 1).copied() else {
            return Ok(None);
        };
        if next == 0x00 {
            self.offset += 2;
            return Ok(Some(0xff));
        }
        Ok(None)
    }

    fn entropy_boundary_error(&self) -> Error {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bits_remaining <= 23);

        if self.bytes.get(self.offset) == Some(&0xff)
            && let Some(marker) = self.bytes.get(self.offset + 1)
        {
            return Error::new(format!(
                "unexpected marker FF{marker:02X} inside entropy data"
            ));
        }
        Error::new("unexpected end of JPEG entropy data")
    }

    pub(super) fn restart(&mut self, expected: u8) -> Result<()> {
        assert!((0xd0..=0xd7).contains(&expected));
        assert!(self.offset <= self.bytes.len());

        self.discard_bits();
        let marker = self.raw_marker()?;
        if marker != expected {
            return Err(Error::new(format!(
                "expected restart marker FF{expected:02X}, found FF{marker:02X}"
            )));
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<(usize, u8)> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bits_remaining <= 23);

        self.discard_bits();
        let marker = self.raw_marker()?;
        assert!(self.offset <= self.bytes.len());
        Ok((self.offset, marker))
    }

    fn raw_marker(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bits_remaining <= 23);

        if self.raw_byte()? != 0xff {
            return Err(Error::new("expected a marker after entropy data"));
        }
        let mut marker = self.raw_byte()?;
        while marker == 0xff {
            marker = self.raw_byte()?;
        }
        if marker == 0x00 {
            return Err(Error::new("unexpected stuffed byte after entropy data"));
        }
        Ok(marker)
    }

    fn discard_bits(&mut self) {
        assert!(self.bits_remaining <= 23);
        assert!(self.offset <= self.bytes.len());
        self.reservoir = 0;
        self.bits_remaining = 0;
    }

    fn raw_byte(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let byte = self
            .bytes
            .get(self.offset)
            .copied()
            .ok_or_else(|| Error::new("unexpected end of JPEG entropy data"))?;
        self.offset += 1;
        Ok(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_lookahead_does_not_consume_a_marker() {
        let mut reader = BitReader::new(&[0b0101_1111, 0xff, 0xd9]);

        assert_eq!(reader.read_bits(4).unwrap(), 0b0101);
        assert_eq!(reader.peek_bits(8).unwrap(), None);
        assert_eq!(reader.finish().unwrap(), (3, 0xd9));
    }

    #[test]
    fn stuffed_ff_remains_an_entropy_byte() {
        let mut reader = BitReader::new(&[0xff, 0x00, 0xff, 0xd9]);

        assert_eq!(reader.peek_bits(8).unwrap(), Some(0xff));
        assert_eq!(reader.read_bits(8).unwrap(), 0xff);
        assert_eq!(reader.finish().unwrap(), (4, 0xd9));
    }
}
