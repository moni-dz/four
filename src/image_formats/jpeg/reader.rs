//! Reads bounded marker segments from a JPEG byte slice.

use exn::OptionExt;

use super::{JPEGError, Result, error};

pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        invariant!(isize::try_from(bytes.len()).is_ok());
        let reader = Self { bytes, offset: 0 };
        invariant_eq!(reader.offset, 0);
        invariant_eq!(reader.remaining(), bytes.len());
        reader
    }

    pub(super) fn remaining(&self) -> usize {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());
        self.bytes.len() - self.offset
    }

    pub(super) fn remaining_slice(&self) -> &'a [u8] {
        invariant!(self.offset <= self.bytes.len());
        invariant_eq!(self.remaining(), self.bytes.len() - self.offset);
        &self.bytes[self.offset..]
    }

    pub(super) fn advance(&mut self, length: usize) -> Result<()> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let end = self
            .offset
            .checked_add(length)
            .ok_or_raise(|| JPEGError::ArithmeticOverflow("JPEG input offset overflowed"))?;

        if end > self.bytes.len() {
            return Err(error(JPEGError::UnexpectedEnd(
                "JPEG input offset extends past the input",
            )));
        }

        self.offset = end;
        Ok(())
    }

    pub(super) fn read_u8(&mut self) -> Result<u8> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let value = self
            .bytes
            .get(self.offset)
            .copied()
            .ok_or_raise(|| JPEGError::UnexpectedEnd("unexpected end of JPEG data"))?;

        self.offset += 1;
        Ok(value)
    }

    pub(super) fn read_u16(&mut self) -> Result<u16> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let high = u16::from(self.read_u8()?);
        let low = u16::from(self.read_u8()?);
        Ok((high << 8) | low)
    }

    pub(super) fn read_slice(&mut self, length: usize) -> Result<&'a [u8]> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let end = self
            .offset
            .checked_add(length)
            .ok_or_raise(|| JPEGError::ArithmeticOverflow("JPEG segment length overflowed"))?;

        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_raise(|| JPEGError::UnexpectedEnd("JPEG segment extends past the input"))?;

        self.offset = end;
        Ok(slice)
    }

    pub(super) fn segment(&mut self) -> Result<Reader<'a>> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let length = usize::from(self.read_u16()?);
        if length < 2 {
            return Err(error(JPEGError::Segment(
                "JPEG segment length is smaller than its header",
            )));
        }
        Ok(Reader::new(self.read_slice(length - 2)?))
    }

    pub(super) fn marker(&mut self) -> Result<u8> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let prefix = self.read_u8()?;
        if prefix != 0xff {
            return Err(error(JPEGError::ExpectedMarkerPrefix {
                context: "in the JPEG marker stream",
                found: prefix,
            }));
        }

        let mut marker = self.read_u8()?;
        while marker == 0xff {
            marker = self.read_u8()?;
        }

        if marker == 0x00 {
            return Err(error(JPEGError::Marker(
                "stuffed byte appeared outside entropy data",
            )));
        }

        Ok(marker)
    }
}
