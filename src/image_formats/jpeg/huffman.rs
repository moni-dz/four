use super::reader::BitReader;
use super::{Error, Result};

const HUFFMAN_BITS_MAX: usize = 16;
const HUFFMAN_SYMBOLS_MAX: usize = 256;
const LOOKUP_BITS: u8 = 8;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;
const LOOKUP_SYMBOL_MASK: u16 = 0x00ff;

#[derive(Clone)]
pub(super) struct HuffmanTable {
    counts: [u8; HUFFMAN_BITS_MAX],
    symbols: Vec<u8>,
    lookup: [u16; LOOKUP_SIZE],
}

impl HuffmanTable {
    pub(super) fn new(counts: [u8; HUFFMAN_BITS_MAX], symbols: Vec<u8>) -> Result<Self> {
        assert!(symbols.len() <= HUFFMAN_SYMBOLS_MAX);
        assert_eq!(counts.len(), HUFFMAN_BITS_MAX);

        let symbol_count: usize = counts.iter().map(|count| usize::from(*count)).sum();
        if symbol_count != symbols.len() {
            return Err(Error::new(
                "Huffman symbol count does not match its code lengths",
            ));
        }
        validate_code_space(&counts)?;
        let lookup = build_lookup(&counts, &symbols);
        Ok(Self {
            counts,
            symbols,
            lookup,
        })
    }

    pub(super) fn decode(&self, reader: &mut BitReader<'_>) -> Result<u8> {
        assert!(self.symbols.len() <= HUFFMAN_SYMBOLS_MAX);
        assert_eq!(self.counts.len(), HUFFMAN_BITS_MAX);

        if let Some(prefix) = reader.peek_bits(LOOKUP_BITS)? {
            let entry = self.lookup[usize::from(prefix)];
            let bit_length = (entry >> 8) as u8;
            if bit_length > 0 {
                reader.consume_bits(bit_length);
                return Ok((entry & LOOKUP_SYMBOL_MASK) as u8);
            }
        }
        self.decode_slow(reader)
    }

    fn decode_slow(&self, reader: &mut BitReader<'_>) -> Result<u8> {
        assert!(self.symbols.len() <= HUFFMAN_SYMBOLS_MAX);
        assert_eq!(self.counts.len(), HUFFMAN_BITS_MAX);

        let mut code = 0_u32;
        let mut first_code = 0_u32;
        let mut symbol_offset = 0_usize;
        for bit_length in 0..HUFFMAN_BITS_MAX {
            code = (code << 1) | u32::from(reader.read_bits(1)?);
            let count = u32::from(self.counts[bit_length]);
            if code >= first_code && code < first_code + count {
                let index = symbol_offset + (code - first_code) as usize;
                return self
                    .symbols
                    .get(index)
                    .copied()
                    .ok_or_else(|| Error::new("Huffman symbol index is out of range"));
            }
            symbol_offset += count as usize;
            first_code = (first_code + count) << 1;
        }
        Err(Error::new("entropy data contains an invalid Huffman code"))
    }
}

/// Expand codes of eight bits or fewer over every matching prefix. Most JPEG Huffman symbols fit
/// here, replacing up to eight dependent bit reads and branches with one lookup.
fn build_lookup(counts: &[u8; HUFFMAN_BITS_MAX], symbols: &[u8]) -> [u16; LOOKUP_SIZE] {
    assert!(symbols.len() <= HUFFMAN_SYMBOLS_MAX);
    assert_eq!(counts.len(), HUFFMAN_BITS_MAX);

    let mut lookup = [0_u16; LOOKUP_SIZE];
    let mut code = 0_u32;
    let mut symbol_index = 0_usize;
    for (length_index, count) in counts.iter().copied().enumerate() {
        let bit_length = length_index as u8 + 1;
        for _ in 0..count {
            if bit_length <= LOOKUP_BITS {
                let suffix_bits = LOOKUP_BITS - bit_length;
                let start = (code << suffix_bits) as usize;
                let end = start + (1_usize << suffix_bits);
                let entry = u16::from(bit_length) << 8 | u16::from(symbols[symbol_index]);
                assert!(lookup[start..end].iter().all(|value| *value == 0));
                lookup[start..end].fill(entry);
            }
            code += 1;
            symbol_index += 1;
        }
        code <<= 1;
    }
    assert_eq!(symbol_index, symbols.len());
    lookup
}

fn validate_code_space(counts: &[u8; HUFFMAN_BITS_MAX]) -> Result<()> {
    assert_eq!(counts.len(), HUFFMAN_BITS_MAX);
    assert!(
        counts
            .iter()
            .all(|count| usize::from(*count) <= HUFFMAN_SYMBOLS_MAX)
    );

    let mut available_codes = 1_i32;
    for count in counts {
        available_codes = available_codes * 2 - i32::from(*count);
        if available_codes < 0 {
            return Err(Error::new("Huffman table is oversubscribed"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_canonical_codes() {
        let mut counts = [0; HUFFMAN_BITS_MAX];
        counts[0] = 1;
        counts[1] = 2;
        let table = HuffmanTable::new(counts, vec![10, 20, 30]).unwrap();
        let mut reader = BitReader::new(&[0b0101_1000]);

        assert_eq!(table.decode(&mut reader).unwrap(), 10);
        assert_eq!(table.decode(&mut reader).unwrap(), 20);
        assert_eq!(table.decode(&mut reader).unwrap(), 30);
    }

    #[test]
    fn lookup_and_slow_path_decode_the_same_table() {
        let mut counts = [0; HUFFMAN_BITS_MAX];
        counts[0] = 1;
        counts[8] = 2;
        let table = HuffmanTable::new(counts, vec![10, 20, 30]).unwrap();
        let mut reader = BitReader::new(&[0b1000_0000, 0b0100_0000, 0b0100_0000]);

        assert_eq!(table.decode(&mut reader).unwrap(), 20);
        assert_eq!(table.decode(&mut reader).unwrap(), 30);
    }
}
