//! The boolean entropy decoder VP9 codes its headers and tiles with, and the plain bit reader
//! for the uncompressed header.

use super::Error;

const BITS: i32 = 64;
const LOTS_OF_BITS: i32 = 0x4000_0000;

pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    value: u64,
    count: i32,
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    /// Starts decoding `data`, which must open with a zero marker bit.
    pub fn new(data: &'a [u8]) -> Result<BoolDecoder<'a>, Error> {
        if data.is_empty() {
            return Err(Error::Truncated);
        }
        let mut decoder = BoolDecoder {
            data,
            pos: 0,
            value: 0,
            count: -8,
            range: 255,
        };
        decoder.fill();
        if decoder.read_bit() {
            return Err(Error::Corrupt("the bool decoder's marker bit is set"));
        }
        Ok(decoder)
    }

    fn fill(&mut self) {
        let mut shift = BITS - 8 - (self.count + 8);
        while shift >= 0 {
            let Some(&byte) = self.data.get(self.pos) else {
                self.count += LOTS_OF_BITS;
                break;
            };
            self.pos += 1;
            self.count += 8;
            self.value |= u64::from(byte) << shift;
            shift -= 8;
        }
    }

    /// One bool coded with probability `prob`/256 of being zero.
    pub fn read(&mut self, prob: u8) -> bool {
        let prob = u32::from(prob);
        let split = (self.range * prob + (256 - prob)) >> 8;
        if self.count < 0 {
            self.fill();
        }
        let big_split = u64::from(split) << (BITS - 8);
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        let shift = (self.range as u8).leading_zeros();
        self.range <<= shift;
        self.value <<= shift;
        self.count -= shift as i32;
        bit
    }

    pub fn read_bit(&mut self) -> bool {
        self.read(128)
    }

    pub fn read_literal(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| (value << 1) | u32::from(self.read_bit()))
    }

    /// Walks a `vpx_tree`: a leaf is stored as the negated symbol.
    pub fn read_tree(&mut self, tree: &[i8], probs: &[u8]) -> u8 {
        let mut node = 0usize;
        loop {
            let next = tree[node + usize::from(self.read(probs[node >> 1]))];
            if next <= 0 {
                return (-next) as u8;
            }
            node = next as usize;
        }
    }

    /// Whether the decoder consumed bits the stream does not have.
    pub fn overran(&self) -> bool {
        self.count > BITS && self.count < LOTS_OF_BITS
    }
}

/// The MSB-first bit buffer of the uncompressed header.
pub struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> BitReader<'a> {
        BitReader { data, bit: 0 }
    }

    pub fn read_bit(&mut self) -> Result<bool, Error> {
        let byte = *self.data.get(self.bit >> 3).ok_or(Error::Truncated)?;
        let bit = (byte >> (7 - (self.bit & 7))) & 1;
        self.bit += 1;
        Ok(bit != 0)
    }

    pub fn read(&mut self, bits: u32) -> Result<u32, Error> {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// `bits` of magnitude, then a sign bit.
    pub fn read_signed(&mut self, bits: u32) -> Result<i32, Error> {
        let value = self.read(bits)? as i32;
        Ok(if self.read_bit()? { -value } else { value })
    }

    pub fn bytes_read(&self) -> usize {
        self.bit.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_reads_msb_first_and_signed_values() {
        let mut reader = BitReader::new(&[0b1010_0001, 0b1000_0000]);
        assert_eq!(reader.read(3).unwrap(), 0b101);
        assert_eq!(reader.read_signed(4).unwrap(), 0);
        assert!(reader.read_bit().unwrap());
        assert_eq!(reader.read(1).unwrap(), 0);
        assert_eq!(reader.bytes_read(), 2);
        assert!(reader.read(16).is_err());
    }

    #[test]
    fn a_stream_of_zero_bytes_decodes_zero_bits_after_a_clear_marker() {
        let data = [0u8; 8];
        let mut decoder = BoolDecoder::new(&data).unwrap();
        assert_eq!(decoder.read_literal(16), 0);
    }

    #[test]
    fn a_set_marker_bit_is_refused() {
        assert!(BoolDecoder::new(&[0xff, 0xff]).is_err());
    }
}
