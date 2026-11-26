//! The writing side of the bool coder and of the uncompressed header's bit buffer, for frames
//! rebuilt from what nvdec is given.

/// `vpx_writer`: the inverse of [`super::bool_decoder::BoolDecoder`].
pub struct BoolEncoder {
    buffer: Vec<u8>,
    low: u32,
    range: u32,
    count: i32,
}

impl BoolEncoder {
    /// Starts a buffer with the zero marker bit the decoder checks.
    pub fn new() -> BoolEncoder {
        let mut encoder = BoolEncoder {
            buffer: Vec::new(),
            low: 0,
            range: 255,
            count: -24,
        };
        encoder.write_bit(false);
        encoder
    }

    /// One bool with probability `prob`/256 of being zero.
    pub fn write(&mut self, bit: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        let (mut low, mut range) = (self.low, split);
        if bit {
            low += split;
            range = self.range - split;
        }
        let mut shift = (range as u8).leading_zeros() as i32;
        range <<= shift;
        let mut count = self.count + shift;
        if count >= 0 {
            let offset = shift - count;
            if (low << (offset - 1)) & 0x8000_0000 != 0 {
                let mut at = self.buffer.len();
                while at > 0 && self.buffer[at - 1] == 0xff {
                    self.buffer[at - 1] = 0;
                    at -= 1;
                }
                // The marker bit keeps the carry from running off the first byte.
                self.buffer[at - 1] += 1;
            }
            self.buffer.push((low >> (24 - offset)) as u8);
            low <<= offset;
            shift = count;
            low &= 0xff_ffff;
            count -= 8;
        }
        low <<= shift;
        self.count = count;
        self.low = low;
        self.range = range;
    }

    pub fn write_bit(&mut self, bit: bool) {
        self.write(bit, 128);
    }

    /// `bits` bits of `value`, most significant first.
    pub fn write_literal(&mut self, value: u32, bits: u32) {
        for bit in (0..bits).rev() {
            self.write_bit(value >> bit & 1 != 0);
        }
    }

    /// Flushes the coder and returns its bytes.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_bit(false);
        }
        // A last byte shaped like a superframe marker could be taken for an index.
        if self.buffer.last().is_some_and(|&b| b & 0xe0 == 0xc0) {
            self.buffer.push(0);
        }
        self.buffer
    }
}

/// The MSB-first bit buffer [`super::bool_decoder::BitReader`] reads.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bits: usize,
}

impl BitWriter {
    pub fn write_bit(&mut self, bit: bool) {
        if self.bits.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if bit {
            *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bits % 8);
        }
        self.bits += 1;
    }

    pub fn write(&mut self, value: u32, bits: u32) {
        for bit in (0..bits).rev() {
            self.write_bit(value >> bit & 1 != 0);
        }
    }

    /// `bits` of magnitude, then a sign bit.
    pub fn write_signed(&mut self, value: i32, bits: u32) {
        self.write(value.unsigned_abs(), bits);
        self.write_bit(value < 0);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::super::bool_decoder::{BitReader, BoolDecoder};
    use super::*;

    /// A fixed pseudo-random sequence, so a failure reproduces.
    fn sequence(seed: u32) -> impl FnMut() -> u32 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        }
    }

    #[test]
    fn bools_read_back_as_written_at_every_probability() {
        let mut next = sequence(7);
        let symbols: Vec<(bool, u8)> = (0..20_000)
            .map(|_| {
                let prob = (next() % 255 + 1) as u8;
                // Drawn at the stated odds, so long runs of 0xff bytes and carries happen.
                (next() % 256 >= u32::from(prob), prob)
            })
            .collect();
        let mut encoder = BoolEncoder::new();
        for &(bit, prob) in &symbols {
            encoder.write(bit, prob);
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes).unwrap();
        for (i, &(bit, prob)) in symbols.iter().enumerate() {
            assert_eq!(decoder.read(prob), bit, "symbol {i}");
        }
        assert!(!decoder.overran());
    }

    #[test]
    fn literals_read_back_as_written() {
        let mut encoder = BoolEncoder::new();
        encoder.write_literal(0x5a, 7);
        encoder.write_literal(3, 2);
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes).unwrap();
        assert_eq!(decoder.read_literal(7), 0x5a);
        assert_eq!(decoder.read_literal(2), 3);
    }

    #[test]
    fn written_bits_and_signed_values_read_back() {
        let mut writer = BitWriter::default();
        writer.write(0b101, 3);
        writer.write_signed(-9, 6);
        writer.write_signed(4, 4);
        writer.write_bit(true);
        let bytes = writer.into_bytes();
        assert_eq!(bytes.len(), 2);
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(3).unwrap(), 0b101);
        assert_eq!(reader.read_signed(6).unwrap(), -9);
        assert_eq!(reader.read_signed(4).unwrap(), 4);
        assert!(reader.read_bit().unwrap());
    }
}
