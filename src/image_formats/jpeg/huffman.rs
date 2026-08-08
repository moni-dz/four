use super::reader::BitReader;
use super::{Error, Result};

const HUFFMAN_BITS_MAX: usize = 16;
const HUFFMAN_SYMBOLS_MAX: usize = 256;

#[derive(Clone)]
pub(super) struct HuffmanTable {
    counts: [u8; HUFFMAN_BITS_MAX],
    symbols: Vec<u8>,
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
        Ok(Self { counts, symbols })
    }

    pub(super) fn decode(&self, reader: &mut BitReader<'_>) -> Result<u8> {
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
}
