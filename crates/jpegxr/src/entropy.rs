//! Decodes the adaptive variable-length codes from T.832 clause 8.

use crate::bitstream::BitReader;
use crate::error::{ErrorKind, Result};

#[derive(Clone, Copy)]
struct Code {
    bits: u8,
    len: u8,
    value: u8,
}

macro_rules! code {
    ($bits:expr, $len:expr, $value:expr) => {
        Code {
            bits: $bits,
            len: $len,
            value: $value,
        }
    };
}

#[derive(Clone, Copy)]
struct Entry {
    value: u8,
    len: u8,
}

/// Direct-indexed decoding table: eight peeked bits map to a value and its code length.
struct Lut {
    max_len: u8,
    entries: [Entry; 256],
}

impl Lut {
    const fn new(codes: &[Code]) -> Self {
        let mut entries = [Entry { value: 0, len: 0 }; 256];
        let mut max_len = 0;
        let mut index = 0;

        while index < codes.len() {
            let code = codes[index];
            assert!(
                1 <= code.len && code.len <= 8,
                "code lengths must fit the eight-bit lookup"
            );

            if code.len > max_len {
                max_len = code.len;
            }

            // A code of `len` bits owns the whole 8-bit range sharing its prefix.
            let base = (code.bits as usize) << (8 - code.len);
            let count = 1_usize << (8 - code.len);
            let mut offset = 0;
            while offset < count {
                assert!(entries[base + offset].len == 0, "codes must be prefix-free");
                entries[base + offset] = Entry {
                    value: code.value,
                    len: code.len,
                };
                offset += 1;
            }

            index += 1;
        }

        Self { max_len, entries }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AdaptiveVLC {
    table: usize,
    delta_low: usize,
    delta_high: usize,
    discriminant_low: i16,
    discriminant_high: i16,
}

impl AdaptiveVLC {
    pub(crate) const fn two_tables() -> Self {
        Self {
            table: 0,
            delta_low: 0,
            delta_high: 0,
            discriminant_low: 0,
            discriminant_high: 0,
        }
    }

    pub(crate) const fn many_tables() -> Self {
        Self {
            table: 1,
            delta_low: 0,
            delta_high: 1,
            discriminant_low: 0,
            discriminant_high: 0,
        }
    }

    fn record(&mut self, value: u8, deltas: &[&[i8]]) {
        let value = usize::from(value);

        self.discriminant_low += i16::from(deltas[self.delta_low][value]);
        self.discriminant_high += i16::from(deltas[self.delta_high][value]);
    }

    pub(crate) fn adapt_two(&mut self) {
        if self.discriminant_low < -8 && self.table != 0 {
            self.table -= 1;
            self.discriminant_low = 0;
        } else if self.discriminant_low > 8 && self.table != 1 {
            self.table += 1;
            self.discriminant_low = 0;
        } else {
            self.discriminant_low = self.discriminant_low.clamp(-64, 64);
        }
    }

    pub(crate) fn adapt_many(&mut self, maximum_table: usize) {
        let changed = if self.discriminant_low < -8 && self.table != 0 {
            self.table -= 1;
            true
        } else if self.discriminant_high > 8 && self.table != maximum_table {
            self.table += 1;
            true
        } else {
            false
        };

        if changed {
            self.discriminant_low = 0;
            self.discriminant_high = 0;

            if self.table == maximum_table {
                self.delta_low = self.table - 1;
                self.delta_high = self.table - 1;
            } else if self.table == 0 {
                self.delta_low = 0;
                self.delta_high = 0;
            } else {
                self.delta_low = self.table - 1;
                self.delta_high = self.table;
            }
        } else {
            self.discriminant_low = self.discriminant_low.clamp(-64, 64);
            self.discriminant_high = self.discriminant_high.clamp(-64, 64);
        }
    }
}

pub(crate) fn val_dc_yuv(reader: &mut BitReader<'_>) -> Result<u8> {
    const CODES: [Code; 8] = [
        code!(0b10, 2, 0),
        code!(0b001, 3, 1),
        code!(0b00001, 5, 2),
        code!(0b0001, 4, 3),
        code!(0b11, 2, 4),
        code!(0b010, 3, 5),
        code!(0b00000, 5, 6),
        code!(0b011, 3, 7),
    ];
    static LUT: Lut = Lut::new(&CODES);
    decode(reader, &LUT)
}

pub(crate) fn abs_level_index(
    reader: &mut BitReader<'_>,
    adaptive: &mut AdaptiveVLC,
) -> Result<u8> {
    const TABLE_0: [Code; 7] = [
        code!(0b01, 2, 0),
        code!(0b10, 2, 1),
        code!(0b11, 2, 2),
        code!(0b001, 3, 3),
        code!(0b0001, 4, 4),
        code!(0b00000, 5, 5),
        code!(0b00001, 5, 6),
    ];
    const TABLE_1: [Code; 7] = [
        code!(0b1, 1, 0),
        code!(0b01, 2, 1),
        code!(0b001, 3, 2),
        code!(0b0001, 4, 3),
        code!(0b00001, 5, 4),
        code!(0b000000, 6, 5),
        code!(0b000001, 6, 6),
    ];
    const DELTA: [i8; 7] = [1, 0, -1, -1, -1, -1, -1];
    static LUTS: [Lut; 2] = [Lut::new(&TABLE_0), Lut::new(&TABLE_1)];

    let value = decode(reader, &LUTS[adaptive.table])?;

    adaptive.record(value, &[&DELTA]);

    Ok(value)
}

pub(crate) fn first_index(reader: &mut BitReader<'_>, adaptive: &mut AdaptiveVLC) -> Result<u8> {
    const TABLE_0: [Code; 12] = [
        code!(0b00001, 5, 0),
        code!(0b000001, 6, 1),
        code!(0b0000000, 7, 2),
        code!(0b0000001, 7, 3),
        code!(0b00100, 5, 4),
        code!(0b010, 3, 5),
        code!(0b00101, 5, 6),
        code!(0b1, 1, 7),
        code!(0b00110, 5, 8),
        code!(0b0001, 4, 9),
        code!(0b00111, 5, 10),
        code!(0b011, 3, 11),
    ];
    const TABLE_1: [Code; 12] = [
        code!(0b0010, 4, 0),
        code!(0b00010, 5, 1),
        code!(0b000000, 6, 2),
        code!(0b000001, 6, 3),
        code!(0b0011, 4, 4),
        code!(0b010, 3, 5),
        code!(0b00011, 5, 6),
        code!(0b11, 2, 7),
        code!(0b011, 3, 8),
        code!(0b100, 3, 9),
        code!(0b00001, 5, 10),
        code!(0b101, 3, 11),
    ];
    const TABLE_2: [Code; 12] = [
        code!(0b11, 2, 0),
        code!(0b001, 3, 1),
        code!(0b0000000, 7, 2),
        code!(0b0000001, 7, 3),
        code!(0b00001, 5, 4),
        code!(0b010, 3, 5),
        code!(0b0000010, 7, 6),
        code!(0b011, 3, 7),
        code!(0b100, 3, 8),
        code!(0b101, 3, 9),
        code!(0b0000011, 7, 10),
        code!(0b0001, 4, 11),
    ];
    const TABLE_3: [Code; 12] = [
        code!(0b001, 3, 0),
        code!(0b11, 2, 1),
        code!(0b0000000, 7, 2),
        code!(0b00001, 5, 3),
        code!(0b00010, 5, 4),
        code!(0b010, 3, 5),
        code!(0b0000001, 7, 6),
        code!(0b011, 3, 7),
        code!(0b00011, 5, 8),
        code!(0b100, 3, 9),
        code!(0b000001, 6, 10),
        code!(0b101, 3, 11),
    ];
    const TABLE_4: [Code; 12] = [
        code!(0b010, 3, 0),
        code!(0b1, 1, 1),
        code!(0b0000001, 7, 2),
        code!(0b0001, 4, 3),
        code!(0b0000010, 7, 4),
        code!(0b011, 3, 5),
        code!(0b00000000, 8, 6),
        code!(0b0010, 4, 7),
        code!(0b0000011, 7, 8),
        code!(0b0011, 4, 9),
        code!(0b00000001, 8, 10),
        code!(0b00001, 5, 11),
    ];
    const DELTA_0: [i8; 12] = [1, 1, 1, 1, 1, 0, 0, -1, 2, 1, 0, 0];
    const DELTA_1: [i8; 12] = [2, 2, -1, -1, -1, 0, -2, -1, 0, 0, -2, -1];
    const DELTA_2: [i8; 12] = [-1, 1, 0, 2, 0, 0, 0, 0, -2, 0, 1, 1];
    const DELTA_3: [i8; 12] = [0, 1, 0, 1, -2, 0, -1, -1, -2, -1, -2, -2];
    static LUTS: [Lut; 5] = [
        Lut::new(&TABLE_0),
        Lut::new(&TABLE_1),
        Lut::new(&TABLE_2),
        Lut::new(&TABLE_3),
        Lut::new(&TABLE_4),
    ];

    let value = decode(reader, &LUTS[adaptive.table])?;

    adaptive.record(value, &[&DELTA_0, &DELTA_1, &DELTA_2, &DELTA_3]);

    Ok(value)
}

pub(crate) fn index_a(reader: &mut BitReader<'_>, adaptive: &mut AdaptiveVLC) -> Result<u8> {
    const TABLE_0: [Code; 6] = [
        code!(0b1, 1, 0),
        code!(0b00000, 5, 1),
        code!(0b001, 3, 2),
        code!(0b00001, 5, 3),
        code!(0b01, 2, 4),
        code!(0b0001, 4, 5),
    ];
    const TABLE_1: [Code; 6] = [
        code!(0b01, 2, 0),
        code!(0b0000, 4, 1),
        code!(0b10, 2, 2),
        code!(0b0001, 4, 3),
        code!(0b11, 2, 4),
        code!(0b001, 3, 5),
    ];
    const TABLE_2: [Code; 6] = [
        code!(0b0000, 4, 0),
        code!(0b0001, 4, 1),
        code!(0b01, 2, 2),
        code!(0b10, 2, 3),
        code!(0b11, 2, 4),
        code!(0b001, 3, 5),
    ];
    const TABLE_3: [Code; 6] = [
        code!(0b00000, 5, 0),
        code!(0b00001, 5, 1),
        code!(0b01, 2, 2),
        code!(0b1, 1, 3),
        code!(0b0001, 4, 4),
        code!(0b001, 3, 5),
    ];
    const DELTA_0: [i8; 6] = [-1, 1, 1, 1, 0, 1];
    const DELTA_1: [i8; 6] = [-2, 0, 0, 2, 0, 0];
    const DELTA_2: [i8; 6] = [-1, -1, 0, 1, -2, 0];
    static LUTS: [Lut; 4] = [
        Lut::new(&TABLE_0),
        Lut::new(&TABLE_1),
        Lut::new(&TABLE_2),
        Lut::new(&TABLE_3),
    ];

    let value = decode(reader, &LUTS[adaptive.table])?;

    adaptive.record(value, &[&DELTA_0, &DELTA_1, &DELTA_2]);

    Ok(value)
}

pub(crate) fn run(reader: &mut BitReader<'_>, maximum: u8) -> Result<u8> {
    if maximum == 0 {
        return Err(reader.error(ErrorKind::InvalidCodestream(
            "coefficient run exceeds block",
        )));
    }

    if maximum < 5 {
        if maximum == 1 || reader.read_bool()? {
            return Ok(1);
        }
        if maximum == 2 || reader.read_bool()? {
            return Ok(2);
        }
        if maximum == 3 || reader.read_bool()? {
            return Ok(3);
        }

        return Ok(4);
    }

    const RUN_INDEX: [Code; 5] = [
        code!(0b1, 1, 0),
        code!(0b01, 2, 1),
        code!(0b001, 3, 2),
        code!(0b0000, 4, 3),
        code!(0b0001, 4, 4),
    ];
    const REMAP: [u8; 15] = [1, 2, 3, 5, 7, 1, 2, 3, 5, 7, 1, 2, 3, 4, 5];
    const BIN: [i8; 15] = [-1, -1, -1, -1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0];
    const FIXED: [u8; 15] = [0, 0, 1, 1, 3, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1];
    static LUT: Lut = Lut::new(&RUN_INDEX);

    let symbol = i16::from(decode(reader, &LUT)?);
    let index = usize::try_from(symbol + 5 * i16::from(BIN[usize::from(maximum)])).map_err(
        |_conversion_error| reader.error(ErrorKind::InvalidCodestream("invalid coefficient run")),
    )?;
    let mut value = REMAP[index];

    if FIXED[index] != 0 {
        value = value
            .checked_add(reader.read_u8(FIXED[index])?)
            .ok_or_else(|| {
                reader.error(ErrorKind::InvalidCodestream("coefficient run overflow"))
            })?;
    }

    Ok(value)
}

pub(crate) fn cbp_lowpass_yuv444(reader: &mut BitReader<'_>) -> Result<u8> {
    const CODES: [Code; 8] = [
        code!(0b0, 1, 0),
        code!(0b100, 3, 1),
        code!(0b1010, 4, 2),
        code!(0b1011, 4, 3),
        code!(0b1100, 4, 4),
        code!(0b1101, 4, 5),
        code!(0b1110, 4, 6),
        code!(0b1111, 4, 7),
    ];
    static LUT: Lut = Lut::new(&CODES);
    decode(reader, &LUT)
}

pub(crate) fn num_cbphp(reader: &mut BitReader<'_>, adaptive: &mut AdaptiveVLC) -> Result<u8> {
    const TABLE_0: [Code; 5] = [
        code!(0b1, 1, 0),
        code!(0b01, 2, 1),
        code!(0b001, 3, 2),
        code!(0b0000, 4, 3),
        code!(0b0001, 4, 4),
    ];
    const TABLE_1: [Code; 5] = [
        code!(0b1, 1, 0),
        code!(0b000, 3, 1),
        code!(0b001, 3, 2),
        code!(0b010, 3, 3),
        code!(0b011, 3, 4),
    ];
    const DELTA: [i8; 5] = [0, -1, 0, 1, 1];
    static LUTS: [Lut; 2] = [Lut::new(&TABLE_0), Lut::new(&TABLE_1)];

    let value = decode(reader, &LUTS[adaptive.table])?;

    adaptive.record(value, &[&DELTA]);

    Ok(value)
}

pub(crate) fn num_block_cbphp_yuv(
    reader: &mut BitReader<'_>,
    adaptive: &mut AdaptiveVLC,
) -> Result<u8> {
    const TABLE_0: [Code; 9] = [
        code!(0b010, 3, 0),
        code!(0b00000, 5, 1),
        code!(0b0010, 4, 2),
        code!(0b00001, 5, 3),
        code!(0b00010, 5, 4),
        code!(0b1, 1, 5),
        code!(0b011, 3, 6),
        code!(0b00011, 5, 7),
        code!(0b0011, 4, 8),
    ];
    const TABLE_1: [Code; 9] = [
        code!(0b1, 1, 0),
        code!(0b001, 3, 1),
        code!(0b010, 3, 2),
        code!(0b0001, 4, 3),
        code!(0b000001, 6, 4),
        code!(0b011, 3, 5),
        code!(0b00001, 5, 6),
        code!(0b0000000, 7, 7),
        code!(0b0000001, 7, 8),
    ];
    const DELTA: [i8; 9] = [2, 2, 1, 1, -1, -2, -2, -2, -3];
    static LUTS: [Lut; 2] = [Lut::new(&TABLE_0), Lut::new(&TABLE_1)];

    let value = decode(reader, &LUTS[adaptive.table])?;

    adaptive.record(value, &[&DELTA]);

    Ok(value)
}

pub(crate) fn num_block_cbphp_yonly(
    reader: &mut BitReader<'_>,
    adaptive: &mut AdaptiveVLC,
) -> Result<u8> {
    num_cbphp(reader, adaptive)
}

pub(crate) fn ternary(reader: &mut BitReader<'_>) -> Result<u8> {
    if reader.read_bool()? {
        Ok(0)
    } else if reader.read_bool()? {
        Ok(1)
    } else {
        Ok(2)
    }
}

pub(crate) fn num_chroma_block(reader: &mut BitReader<'_>) -> Result<u8> {
    if reader.read_bool()? {
        Ok(0)
    } else if reader.read_bool()? {
        Ok(1)
    } else {
        Ok(2 + reader.read_u8(1)?)
    }
}

pub(crate) fn refine_cbphp_one(reader: &mut BitReader<'_>) -> Result<u8> {
    const CODES: [Code; 6] = [
        code!(0b00, 2, 3),
        code!(0b01, 2, 5),
        code!(0b100, 3, 6),
        code!(0b101, 3, 9),
        code!(0b110, 3, 10),
        code!(0b111, 3, 12),
    ];
    static LUT: Lut = Lut::new(&CODES);
    decode(reader, &LUT)
}

fn decode(reader: &mut BitReader<'_>, lut: &Lut) -> Result<u8> {
    let entry = lut.entries[usize::from(reader.peek8())];

    if entry.len == 0 {
        if reader.remaining_bits() < usize::from(lut.max_len) {
            return Err(reader.error(ErrorKind::UnexpectedEof));
        }

        return Err(reader.error(ErrorKind::InvalidCodestream("invalid variable-length code")));
    }

    reader.consume(entry.len)?;
    Ok(entry.value)
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveVLC, abs_level_index, first_index, index_a, run, val_dc_yuv};
    use crate::bitstream::BitReader;

    #[test]
    fn decodes_entropy_codewords() {
        let mut reader = BitReader::new(&[0b0000_1111, 0b0000_0000], 0);
        assert_eq!(val_dc_yuv(&mut reader), Ok(2));
        assert_eq!(val_dc_yuv(&mut reader), Ok(4));
        assert_eq!(val_dc_yuv(&mut reader), Ok(0));

        let mut reader = BitReader::new(&[0b0000_1000], 0);
        let mut vlc = AdaptiveVLC::two_tables();
        assert_eq!(abs_level_index(&mut reader, &mut vlc), Ok(6));

        let mut reader = BitReader::new(&[0b0000_0000], 0);
        let mut vlc = AdaptiveVLC::many_tables();
        assert_eq!(first_index(&mut reader, &mut vlc), Ok(2));

        let mut reader = BitReader::new(&[0b0000_0000], 0);
        let mut vlc = AdaptiveVLC::many_tables();
        assert_eq!(index_a(&mut reader, &mut vlc), Ok(1));

        let mut reader = BitReader::new(&[0b0101_0000], 0);
        assert_eq!(run(&mut reader, 10), Ok(2));
    }
}
