//! Implements the bounded JPEG QM binary arithmetic decoder.

use exn::OptionExt;

use super::super::{JPEGError, Result, error};
use super::FIXED_STATE_INDEX;

const RENORMALIZATION_ITERATIONS_MAX: u8 = 17;

const _: () = {
    invariant!(PROBABILITY_ESTIMATES.len() == FIXED_STATE_INDEX as usize + 1);
};

pub(super) struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
    code: u32,
    interval: u32,
    bit_count: i8,
    pending_marker: Option<u8>,
}

impl<'a> Decoder<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        invariant!(isize::try_from(bytes.len()).is_ok());
        let decoder = Self {
            bytes,
            offset: 0,
            code: 0,
            interval: 0,
            bit_count: -16,
            pending_marker: None,
        };
        invariant_eq!(decoder.offset, 0);
        invariant_eq!(decoder.bit_count, -16);
        decoder
    }

    pub(super) fn decode(&mut self, state: &mut u8) -> Result<u8> {
        invariant!(usize::from(*state & 0x7f) < PROBABILITY_ESTIMATES.len());
        invariant!(*state >> 7 <= 1);

        self.renormalize()?;
        let mps = *state >> 7;
        let estimate = PROBABILITY_ESTIMATES[usize::from(*state & 0x7f)];
        self.interval -= u32::from(estimate.value);
        let mps_interval = self.interval;
        let boundary = self.interval << self.bit_count;
        let decision = if self.code >= boundary {
            self.code -= boundary;
            self.decode_lps_path(state, estimate, mps, mps_interval)
        } else if self.interval < 0x8000 {
            self.decode_mps_exchange(state, estimate, mps)
        } else {
            mps
        };
        invariant!(decision <= 1);
        Ok(decision)
    }

    fn decode_lps_path(
        &mut self,
        state: &mut u8,
        estimate: ProbabilityEstimate,
        mps: u8,
        mps_interval: u32,
    ) -> u8 {
        invariant!(mps <= 1);
        invariant!(mps_interval < 0x1_0000);

        self.interval = u32::from(estimate.value);
        if mps_interval < u32::from(estimate.value) {
            Self::update_mps(state, estimate);
            mps
        } else {
            Self::update_lps(state, estimate);
            mps ^ 1
        }
    }

    fn decode_mps_exchange(
        &mut self,
        state: &mut u8,
        estimate: ProbabilityEstimate,
        mps: u8,
    ) -> u8 {
        invariant!(self.interval < 0x8000);
        invariant!(mps <= 1);

        if self.interval < u32::from(estimate.value) {
            Self::update_lps(state, estimate);
            mps ^ 1
        } else {
            Self::update_mps(state, estimate);
            mps
        }
    }

    fn update_lps(state: &mut u8, estimate: ProbabilityEstimate) {
        let mut mps = *state >> 7;
        invariant!(mps <= 1);
        invariant!(usize::from(estimate.next_lps) < PROBABILITY_ESTIMATES.len());

        if estimate.switch_mps {
            mps ^= 1;
        }
        *state = (mps << 7) | estimate.next_lps;
    }

    fn update_mps(state: &mut u8, estimate: ProbabilityEstimate) {
        let mps = *state >> 7;
        invariant!(mps <= 1);
        invariant!(usize::from(estimate.next_mps) < PROBABILITY_ESTIMATES.len());
        *state = (mps << 7) | estimate.next_mps;
    }

    fn renormalize(&mut self) -> Result<()> {
        invariant!(self.interval <= 0x1_0000);
        invariant!((-16..=7).contains(&self.bit_count));

        let mut iterations = 0_u8;
        while self.interval < 0x8000 {
            iterations += 1;
            if iterations > RENORMALIZATION_ITERATIONS_MAX {
                return Err(error(JPEGError::Entropy(
                    "arithmetic decoder renormalization exceeded its bound",
                )));
            }
            self.bit_count -= 1;
            if self.bit_count < 0 {
                self.byte_in()?;
            }
            self.interval <<= 1;
        }
        invariant!(self.interval >= 0x8000);
        Ok(())
    }

    fn byte_in(&mut self) -> Result<()> {
        invariant!(self.bit_count < 0);
        invariant!(self.offset <= self.bytes.len());

        let byte = self.next_entropy_byte()?;
        self.code = (self.code << 8) | u32::from(byte);
        self.bit_count += 8;
        if self.bit_count < 0 {
            self.bit_count += 1;
            if self.bit_count == 0 {
                self.interval = 0x8000;
            }
        }
        Ok(())
    }

    fn next_entropy_byte(&mut self) -> Result<u8> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(self.pending_marker.is_none() || self.offset >= 2);

        if self.pending_marker.is_some() {
            return Ok(0);
        }
        let byte = self.raw_byte()?;
        if byte != 0xff {
            return Ok(byte);
        }
        let marker = self.marker_suffix()?;
        if marker == 0 {
            return Ok(0xff);
        }
        self.pending_marker = Some(marker);
        Ok(0)
    }

    fn marker_suffix(&mut self) -> Result<u8> {
        invariant!(self.offset > 0);
        invariant_eq!(self.bytes[self.offset - 1], 0xff);

        let mut marker = self.raw_byte()?;
        while marker == 0xff {
            marker = self.raw_byte()?;
        }
        Ok(marker)
    }

    pub(super) fn restart(&mut self, expected: u8) -> Result<()> {
        invariant!((0xd0..=0xd7).contains(&expected));
        invariant!(self.offset <= self.bytes.len());

        let marker = self.take_marker()?;
        if marker != expected {
            return Err(error(JPEGError::RestartMarkerMismatch {
                expected,
                found: marker,
            }));
        }
        self.code = 0;
        self.interval = 0;
        self.bit_count = -16;
        self.pending_marker = None;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<(usize, u8)> {
        invariant!(self.offset <= self.bytes.len());
        invariant!((-16..=7).contains(&self.bit_count));

        let marker = self.take_marker()?;
        invariant!(self.offset <= self.bytes.len());
        Ok((self.offset, marker))
    }

    fn take_marker(&mut self) -> Result<u8> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(self.pending_marker.is_none() || self.offset >= 2);

        if let Some(marker) = self.pending_marker.take() {
            return Ok(marker);
        }
        while self.offset < self.bytes.len() {
            let byte = self.raw_byte()?;
            if byte != 0xff {
                continue;
            }
            let marker = self.marker_suffix()?;
            if marker != 0 {
                return Ok(marker);
            }
        }
        Err(error(JPEGError::UnexpectedEnd(
            "arithmetic entropy data is missing its terminating marker",
        )))
    }

    fn raw_byte(&mut self) -> Result<u8> {
        invariant!(self.offset <= self.bytes.len());
        invariant!(isize::try_from(self.bytes.len()).is_ok());

        let byte = self.bytes.get(self.offset).copied().ok_or_raise(|| {
            JPEGError::UnexpectedEnd("unexpected end of arithmetic entropy data")
        })?;
        self.offset += 1;
        Ok(byte)
    }
}

#[derive(Clone, Copy)]
struct ProbabilityEstimate {
    value: u16,
    next_lps: u8,
    next_mps: u8,
    switch_mps: bool,
}

#[allow(
    clippy::similar_names,
    reason = "MPS and LPS are the canonical ITU-T arithmetic-coder transition names"
)]
const fn probability(
    value: u16,
    next_lps: u8,
    next_mps: u8,
    switch_mps: bool,
) -> ProbabilityEstimate {
    ProbabilityEstimate {
        value,
        next_lps,
        next_mps,
        switch_mps,
    }
}

// ITU-T T.81 Table D.3. Keeping the transition table literal avoids arithmetic or indexing that
// could turn a corrupt probability state into an out-of-bounds transition in the decoder hot path.
const PROBABILITY_ESTIMATES: [ProbabilityEstimate; 114] = [
    probability(0x5a1d, 1, 1, true),
    probability(0x2586, 14, 2, false),
    probability(0x1114, 16, 3, false),
    probability(0x080b, 18, 4, false),
    probability(0x03d8, 20, 5, false),
    probability(0x01da, 23, 6, false),
    probability(0x00e5, 25, 7, false),
    probability(0x006f, 28, 8, false),
    probability(0x0036, 30, 9, false),
    probability(0x001a, 33, 10, false),
    probability(0x000d, 35, 11, false),
    probability(0x0006, 9, 12, false),
    probability(0x0003, 10, 13, false),
    probability(0x0001, 12, 13, false),
    probability(0x5a7f, 15, 15, true),
    probability(0x3f25, 36, 16, false),
    probability(0x2cf2, 38, 17, false),
    probability(0x207c, 39, 18, false),
    probability(0x17b9, 40, 19, false),
    probability(0x1182, 42, 20, false),
    probability(0x0cef, 43, 21, false),
    probability(0x09a1, 45, 22, false),
    probability(0x072f, 46, 23, false),
    probability(0x055c, 48, 24, false),
    probability(0x0406, 49, 25, false),
    probability(0x0303, 51, 26, false),
    probability(0x0240, 52, 27, false),
    probability(0x01b1, 54, 28, false),
    probability(0x0144, 56, 29, false),
    probability(0x00f5, 57, 30, false),
    probability(0x00b7, 59, 31, false),
    probability(0x008a, 60, 32, false),
    probability(0x0068, 62, 33, false),
    probability(0x004e, 63, 34, false),
    probability(0x003b, 32, 35, false),
    probability(0x002c, 33, 9, false),
    probability(0x5ae1, 37, 37, true),
    probability(0x484c, 64, 38, false),
    probability(0x3a0d, 65, 39, false),
    probability(0x2ef1, 67, 40, false),
    probability(0x261f, 68, 41, false),
    probability(0x1f33, 69, 42, false),
    probability(0x19a8, 70, 43, false),
    probability(0x1518, 72, 44, false),
    probability(0x1177, 73, 45, false),
    probability(0x0e74, 74, 46, false),
    probability(0x0bfb, 75, 47, false),
    probability(0x09f8, 77, 48, false),
    probability(0x0861, 78, 49, false),
    probability(0x0706, 79, 50, false),
    probability(0x05cd, 48, 51, false),
    probability(0x04de, 50, 52, false),
    probability(0x040f, 50, 53, false),
    probability(0x0363, 51, 54, false),
    probability(0x02d4, 52, 55, false),
    probability(0x025c, 53, 56, false),
    probability(0x01f8, 54, 57, false),
    probability(0x01a4, 55, 58, false),
    probability(0x0160, 56, 59, false),
    probability(0x0125, 57, 60, false),
    probability(0x00f6, 58, 61, false),
    probability(0x00cb, 59, 62, false),
    probability(0x00ab, 61, 63, false),
    probability(0x008f, 61, 32, false),
    probability(0x5b12, 65, 65, true),
    probability(0x4d04, 80, 66, false),
    probability(0x412c, 81, 67, false),
    probability(0x37d8, 82, 68, false),
    probability(0x2fe8, 83, 69, false),
    probability(0x293c, 84, 70, false),
    probability(0x2379, 86, 71, false),
    probability(0x1edf, 87, 72, false),
    probability(0x1aa9, 87, 73, false),
    probability(0x174e, 72, 74, false),
    probability(0x1424, 72, 75, false),
    probability(0x119c, 74, 76, false),
    probability(0x0f6b, 74, 77, false),
    probability(0x0d51, 75, 78, false),
    probability(0x0bb6, 77, 79, false),
    probability(0x0a40, 77, 48, false),
    probability(0x5832, 80, 81, true),
    probability(0x4d1c, 88, 82, false),
    probability(0x438e, 89, 83, false),
    probability(0x3bdd, 90, 84, false),
    probability(0x34ee, 91, 85, false),
    probability(0x2eae, 92, 86, false),
    probability(0x299a, 93, 87, false),
    probability(0x2516, 86, 71, false),
    probability(0x5570, 88, 89, true),
    probability(0x4ca9, 95, 90, false),
    probability(0x44d9, 96, 91, false),
    probability(0x3e22, 97, 92, false),
    probability(0x3824, 99, 93, false),
    probability(0x32b4, 99, 94, false),
    probability(0x2e17, 93, 86, false),
    probability(0x56a8, 95, 96, true),
    probability(0x4f46, 101, 97, false),
    probability(0x47e5, 102, 98, false),
    probability(0x41cf, 103, 99, false),
    probability(0x3c3d, 104, 100, false),
    probability(0x375e, 99, 93, false),
    probability(0x5231, 105, 102, false),
    probability(0x4c0f, 106, 103, false),
    probability(0x4639, 107, 104, false),
    probability(0x415e, 103, 99, false),
    probability(0x5627, 105, 106, true),
    probability(0x50e7, 108, 107, false),
    probability(0x4b85, 109, 103, false),
    probability(0x5597, 110, 109, false),
    probability(0x504f, 111, 107, false),
    probability(0x5a10, 110, 111, true),
    probability(0x5522, 112, 109, false),
    probability(0x59eb, 112, 111, true),
    probability(0x5a1d, 113, 113, false),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_standard_arithmetic_test_sequence() {
        // ITU-T T.81 K.4.1 supplies both this entropy segment and the 256 binary decisions. The
        // appended EOI marker also exercises the arithmetic decoder's normative zero-fill path.
        let entropy = [
            0x65, 0x5b, 0x51, 0x44, 0xf7, 0x96, 0x9d, 0x51, 0x78, 0x55, 0xbf, 0xff, 0x00, 0xfc,
            0x51, 0x84, 0xc7, 0xce, 0xf9, 0x39, 0x00, 0x28, 0x7d, 0x46, 0x70, 0x8e, 0xcb, 0xc0,
            0xf6, 0xff, 0xd9, 0x00,
        ];
        let expected = [
            0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00, 0xc0, 0x03, 0x52, 0x87, 0x2a, 0xaa, 0xaa,
            0xaa, 0xaa, 0x82, 0xc0, 0x20, 0x00, 0xfc, 0xd7, 0x9e, 0xf6, 0x74, 0xea, 0xab, 0xf7,
            0x69, 0x7e, 0xe7, 0x4c,
        ];
        let mut decoder = Decoder::new(&entropy);
        let mut state = 0_u8;
        for bit_index in 0..256 {
            let expected_bit = (expected[bit_index / 8] >> (7 - bit_index % 8)) & 1;
            invariant_eq!(decoder.decode(&mut state).unwrap(), expected_bit);
        }
        invariant_eq!(decoder.finish().unwrap(), (31, 0xd9));
    }
}
