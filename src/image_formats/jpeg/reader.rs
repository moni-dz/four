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
    current: u8,
    bits_remaining: u8,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        let reader = Self {
            bytes,
            offset: 0,
            current: 0,
            bits_remaining: 0,
        };
        assert_eq!(reader.offset, 0);
        assert_eq!(reader.bits_remaining, 0);
        reader
    }

    pub(super) fn read_bits(&mut self, count: u8) -> Result<u16> {
        assert!(count <= 16);
        assert!(self.bits_remaining <= 8);

        let mut value = 0_u16;
        for _ in 0..count {
            if self.bits_remaining == 0 {
                self.current = self.entropy_byte()?;
                self.bits_remaining = 8;
            }
            self.bits_remaining -= 1;
            value = (value << 1) | u16::from((self.current >> self.bits_remaining) & 1);
        }
        Ok(value)
    }

    fn entropy_byte(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert_eq!(self.bits_remaining, 0);

        let byte = self.raw_byte()?;
        if byte != 0xff {
            return Ok(byte);
        }
        let next = self.raw_byte()?;
        if next == 0x00 {
            return Ok(0xff);
        }
        Err(Error::new(format!(
            "unexpected marker FF{next:02X} inside entropy data"
        )))
    }

    pub(super) fn restart(&mut self, expected: u8) -> Result<()> {
        assert!((0xd0..=0xd7).contains(&expected));
        assert!(self.offset <= self.bytes.len());

        self.bits_remaining = 0;
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
        assert!(self.bits_remaining <= 8);

        self.bits_remaining = 0;
        let marker = self.raw_marker()?;
        assert!(self.offset <= self.bytes.len());
        Ok((self.offset, marker))
    }

    fn raw_marker(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bits_remaining <= 8);

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
