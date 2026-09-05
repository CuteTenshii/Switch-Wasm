//! BC6H (`BPTC_FLOAT`): 16 bytes of HDR RGB in one of fourteen modes.
//!
//! Two things make this the most intricate of the family. Its endpoints are
//! half-floats reached through a quantise/delta/unquantise chain rather than
//! stored directly, and its header fields are *scattered*, a single endpoint
//! channel is assembled from up to six runs of bits that are nowhere near each
//! other in the block, in an order that differs per mode.
//!
//! [`MODES`] is that scatter, one entry per run: which endpoint field the run
//! belongs to, which bit of it the run starts at, how long it is, and whether
//! it arrives most-significant-bit first. Writing it as data rather than as
//! fourteen hand-written bit-twiddling routines is what makes it checkable,
//! every mode has to account for exactly 128 bits, and the test below adds
//! them up.
//!
//! There is no alpha: BC6H is an RGB format, and the decoder reports 1.0.

use super::{anchor, BitReader, Block, PARTITIONS_2, WEIGHTS_3, WEIGHTS_4};
use crate::gpu::surface::f16_to_f32;

/// The twelve endpoint fields, as `channel * 4 + endpoint` where the channels
/// run R, G, B and the endpoints are the specification's w, x, y and z.
struct E;

impl E {
    const RW: u8 = 0;
    const RX: u8 = 1;
    const RY: u8 = 2;
    const RZ: u8 = 3;
    const GW: u8 = 4;
    const GX: u8 = 5;
    const GY: u8 = 6;
    const GZ: u8 = 7;
    const BW: u8 = 8;
    const BX: u8 = 9;
    const BY: u8 = 10;
    const BZ: u8 = 11;
}

/// A field this run of bits belongs to, or the partition number.
const PARTITION: u8 = 0xFF;

/// One run of header bits.
#[derive(Clone, Copy)]
struct F {
    target: u8,
    /// The bit of the target field this run starts at.
    shift: u8,
    count: u8,
    /// The run arrives most-significant bit first, which only the modes with
    /// 16-bit endpoints use.
    reversed: bool,
}

impl F {
    const fn bits(target: u8, shift: u8, count: u8) -> F {
        F {
            target,
            shift,
            count,
            reversed: false,
        }
    }

    const fn reversed(target: u8, shift: u8, count: u8) -> F {
        F {
            target,
            shift,
            count,
            reversed: true,
        }
    }

    const fn partition(count: u8) -> F {
        F {
            target: PARTITION,
            shift: 0,
            count,
            reversed: false,
        }
    }
}

struct Mode {
    /// The mode's prefix, two bits for the first two modes and five for the
    /// rest.
    raw: u8,
    raw_bits: u32,
    fields: &'static [F],
}

const MODES: [Mode; 14] = [
    Mode {
        raw: 0b00,
        raw_bits: 2,
        fields: &[
            F::bits(E::GY, 4, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 5),
            F::bits(E::GZ, 4, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 5),
            F::bits(E::BZ, 0, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 5),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 5),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 5),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b01,
        raw_bits: 2,
        fields: &[
            F::bits(E::GY, 5, 1),
            F::bits(E::GZ, 4, 1),
            F::bits(E::GZ, 5, 1),
            F::bits(E::RW, 0, 7),
            F::bits(E::BZ, 0, 1),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::GW, 0, 7),
            F::bits(E::BY, 5, 1),
            F::bits(E::BZ, 2, 1),
            F::bits(E::GY, 4, 1),
            F::bits(E::BW, 0, 7),
            F::bits(E::BZ, 3, 1),
            F::bits(E::BZ, 5, 1),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RX, 0, 6),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 6),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 6),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 6),
            F::bits(E::RZ, 0, 6),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b00010,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 5),
            F::bits(E::RW, 10, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 4),
            F::bits(E::GW, 10, 1),
            F::bits(E::BZ, 0, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 4),
            F::bits(E::BW, 10, 1),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 5),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 5),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b00110,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 4),
            F::bits(E::RW, 10, 1),
            F::bits(E::GZ, 4, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 5),
            F::bits(E::GW, 10, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 4),
            F::bits(E::BW, 10, 1),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 4),
            F::bits(E::BZ, 0, 1),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 4),
            F::bits(E::GY, 4, 1),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b01010,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 4),
            F::bits(E::RW, 10, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 4),
            F::bits(E::GW, 10, 1),
            F::bits(E::BZ, 0, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 5),
            F::bits(E::BW, 10, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 4),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 4),
            F::bits(E::BZ, 4, 1),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b01110,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 9),
            F::bits(E::BY, 4, 1),
            F::bits(E::GW, 0, 9),
            F::bits(E::GY, 4, 1),
            F::bits(E::BW, 0, 9),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RX, 0, 5),
            F::bits(E::GZ, 4, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 5),
            F::bits(E::BZ, 0, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 5),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 5),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 5),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b10010,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 8),
            F::bits(E::GZ, 4, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::GW, 0, 8),
            F::bits(E::BZ, 2, 1),
            F::bits(E::GY, 4, 1),
            F::bits(E::BW, 0, 8),
            F::bits(E::BZ, 3, 1),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RX, 0, 6),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 5),
            F::bits(E::BZ, 0, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 5),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 6),
            F::bits(E::RZ, 0, 6),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b10110,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 8),
            F::bits(E::BZ, 0, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::GW, 0, 8),
            F::bits(E::GY, 5, 1),
            F::bits(E::GY, 4, 1),
            F::bits(E::BW, 0, 8),
            F::bits(E::GZ, 5, 1),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RX, 0, 5),
            F::bits(E::GZ, 4, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 6),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 5),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 5),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 5),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b11010,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 8),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::GW, 0, 8),
            F::bits(E::BY, 5, 1),
            F::bits(E::GY, 4, 1),
            F::bits(E::BW, 0, 8),
            F::bits(E::BZ, 5, 1),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RX, 0, 5),
            F::bits(E::GZ, 4, 1),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 5),
            F::bits(E::BZ, 0, 1),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 6),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 5),
            F::bits(E::BZ, 2, 1),
            F::bits(E::RZ, 0, 5),
            F::bits(E::BZ, 3, 1),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b11110,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 6),
            F::bits(E::GZ, 4, 1),
            F::bits(E::BZ, 0, 1),
            F::bits(E::BZ, 1, 1),
            F::bits(E::BY, 4, 1),
            F::bits(E::GW, 0, 6),
            F::bits(E::GY, 5, 1),
            F::bits(E::BY, 5, 1),
            F::bits(E::BZ, 2, 1),
            F::bits(E::GY, 4, 1),
            F::bits(E::BW, 0, 6),
            F::bits(E::GZ, 5, 1),
            F::bits(E::BZ, 3, 1),
            F::bits(E::BZ, 5, 1),
            F::bits(E::BZ, 4, 1),
            F::bits(E::RX, 0, 6),
            F::bits(E::GY, 0, 4),
            F::bits(E::GX, 0, 6),
            F::bits(E::GZ, 0, 4),
            F::bits(E::BX, 0, 6),
            F::bits(E::BY, 0, 4),
            F::bits(E::RY, 0, 6),
            F::bits(E::RZ, 0, 6),
            F::partition(5),
        ],
    },
    Mode {
        raw: 0b00011,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 10),
            F::bits(E::GX, 0, 10),
            F::bits(E::BX, 0, 10),
        ],
    },
    Mode {
        raw: 0b00111,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 9),
            F::bits(E::RW, 10, 1),
            F::bits(E::GX, 0, 9),
            F::bits(E::GW, 10, 1),
            F::bits(E::BX, 0, 9),
            F::bits(E::BW, 10, 1),
        ],
    },
    Mode {
        raw: 0b01011,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 8),
            F::reversed(E::RW, 10, 2),
            F::bits(E::GX, 0, 8),
            F::reversed(E::GW, 10, 2),
            F::bits(E::BX, 0, 8),
            F::reversed(E::BW, 10, 2),
        ],
    },
    Mode {
        raw: 0b01111,
        raw_bits: 5,
        fields: &[
            F::bits(E::RW, 0, 10),
            F::bits(E::GW, 0, 10),
            F::bits(E::BW, 0, 10),
            F::bits(E::RX, 0, 4),
            F::reversed(E::RW, 10, 6),
            F::bits(E::GX, 0, 4),
            F::reversed(E::GW, 10, 6),
            F::bits(E::BX, 0, 4),
            F::reversed(E::BW, 10, 6),
        ],
    },
];

/// Endpoint precision per mode: the base endpoint's bit count, then the bit
/// counts of the red, green and blue deltas.
const PRECISION: [[u32; 14]; 4] = [
    [10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16],
    [5, 6, 5, 4, 4, 5, 6, 5, 5, 6, 10, 9, 8, 4],
    [5, 6, 4, 5, 4, 5, 5, 6, 5, 6, 10, 9, 8, 4],
    [5, 6, 4, 4, 5, 5, 5, 5, 6, 6, 10, 9, 8, 4],
];

/// The modes whose deltas are as wide as their base endpoint, which is the
/// same as saying they store both endpoints outright and skip the transform.
fn stores_endpoints_directly(mode: usize) -> bool {
    mode == 9 || mode == 10
}

fn extend_sign(value: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (value << shift) >> shift
}

/// Undo the delta encoding: a non-base endpoint is stored as its difference
/// from the base, wrapped to the base's precision.
fn transform_inverse(value: i32, base: i32, bits: u32, signed: bool) -> i32 {
    let wrapped = (value.wrapping_add(base)) & ((1 << bits) - 1);
    if signed {
        extend_sign(wrapped, bits)
    } else {
        wrapped
    }
}

/// Spread a quantised endpoint back across the 16-bit range.
fn unquantize(value: i32, bits: u32, signed: bool) -> i32 {
    if !signed {
        if bits >= 15 {
            value
        } else if value == 0 {
            0
        } else if value == (1 << bits) - 1 {
            0xFFFF
        } else {
            ((value << 16) + 0x8000) >> bits
        }
    } else if bits >= 16 {
        value
    } else {
        let negative = value < 0;
        let magnitude = value.abs();
        let unquantized = if magnitude == 0 {
            0
        } else if magnitude >= (1 << (bits - 1)) - 1 {
            0x7FFF
        } else {
            ((magnitude << 15) + 0x4000) >> (bits - 1)
        };
        if negative {
            -unquantized
        } else {
            unquantized
        }
    }
}

/// The last step, applied after interpolation: scale the magnitude into the
/// range a half-float's bit pattern expects, and reattach the sign.
fn finish_unquantize(value: i32, signed: bool) -> u16 {
    if !signed {
        ((value * 31) >> 6) as u16
    } else {
        let scaled = if value < 0 {
            -(((-value) * 31) >> 5)
        } else {
            (value * 31) >> 5
        };
        if scaled < 0 {
            0x8000 | (-scaled) as u16
        } else {
            scaled as u16
        }
    }
}

pub fn decode_bc6h(block: &[u8], signed: bool) -> Block {
    let mut reader = BitReader::new(block);
    let prefix = reader.read(2);
    let raw = if prefix > 1 {
        prefix | (reader.read(3) << 2)
    } else {
        prefix
    };
    let raw_bits = if prefix > 1 { 5 } else { 2 };
    let Some(mode_index) = MODES
        .iter()
        .position(|m| m.raw_bits == raw_bits && m.raw as u32 == raw)
    else {
        // Four of the five-bit prefixes are reserved. The specification says a
        // block using one decodes to zero rather than to anything diagnostic.
        return [[0.0, 0.0, 0.0, 1.0]; 16];
    };
    let mode = &MODES[mode_index];

    // Assemble the endpoints out of the mode's scattered runs.
    let mut endpoint = [[0i32; 4]; 3];
    let mut partition = 0usize;
    for field in mode.fields {
        let mut value = reader.read(field.count as u32);
        if field.reversed {
            let mut flipped = 0u32;
            for i in 0..field.count as u32 {
                flipped = (flipped << 1) | ((value >> i) & 1);
            }
            value = flipped;
        }
        if field.target == PARTITION {
            partition = value as usize;
        } else {
            endpoint[(field.target / 4) as usize][(field.target % 4) as usize] |=
                (value as i32) << field.shift;
        }
    }

    let two_subsets = mode_index < 10;
    let endpoints = if two_subsets { 4 } else { 2 };
    let base_bits = PRECISION[0][mode_index];

    if signed {
        for channel in endpoint.iter_mut() {
            channel[0] = extend_sign(channel[0], base_bits);
        }
    }
    // A delta is signed whatever the format is; an outright endpoint is only
    // signed when the format is.
    if !stores_endpoints_directly(mode_index) || signed {
        for (c, channel) in endpoint.iter_mut().enumerate() {
            for slot in channel.iter_mut().take(endpoints).skip(1) {
                *slot = extend_sign(*slot, PRECISION[c + 1][mode_index]);
            }
        }
    }
    if !stores_endpoints_directly(mode_index) {
        for channel in endpoint.iter_mut() {
            let base = channel[0];
            for slot in channel.iter_mut().take(endpoints).skip(1) {
                *slot = transform_inverse(*slot, base, base_bits, signed);
            }
        }
    }
    for channel in endpoint.iter_mut() {
        for slot in channel.iter_mut().take(endpoints) {
            *slot = unquantize(*slot, base_bits, signed);
        }
    }

    let index_bits = if two_subsets { 3 } else { 4 };
    let weights: &[u32] = if two_subsets { &WEIGHTS_3 } else { &WEIGHTS_4 };
    let mut out = [[0.0f32, 0.0, 0.0, 1.0]; 16];
    for (texel, slot) in out.iter_mut().enumerate() {
        let subset = if two_subsets {
            PARTITIONS_2[partition][texel] as usize
        } else {
            0
        };
        let is_anchor = if two_subsets {
            anchor(2, partition, subset) == texel
        } else {
            texel == 0
        };
        let index = reader.read(index_bits - u32::from(is_anchor)) as usize;
        let weight = weights[index];
        for (channel, value) in endpoint.iter().zip(slot.iter_mut()) {
            let a = channel[subset * 2];
            let b = channel[subset * 2 + 1];
            let mixed = (a * (64 - weight as i32) + b * weight as i32 + 32) >> 6;
            *value = f16_to_f32(finish_unquantize(mixed, signed));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mode spends the block's 128 bits exactly: its prefix, its
    /// scattered header runs, and one index per texel less the anchor's high
    /// bit. A mistyped run length in the table shows up here and nowhere else.
    #[test]
    fn every_mode_accounts_for_all_128_bits() {
        for (index, mode) in MODES.iter().enumerate() {
            let header: u32 = mode.fields.iter().map(|f| f.count as u32).sum();
            let two_subsets = index < 10;
            let indices = if two_subsets { 3 * 16 - 2 } else { 4 * 16 - 1 };
            assert_eq!(mode.raw_bits + header + indices, 128, "mode {index}");
        }
    }

    /// A two-subset mode reads a partition and a one-subset mode does not, and
    /// that split has to line up with the precision table's own split.
    #[test]
    fn only_two_subset_modes_read_a_partition() {
        for (index, mode) in MODES.iter().enumerate() {
            let reads_partition = mode.fields.iter().any(|f| f.target == PARTITION);
            assert_eq!(reads_partition, index < 10, "mode {index}");
        }
        // The direct-endpoint modes are exactly those whose deltas are as wide
        // as their base.
        for (mode, &base) in PRECISION[0].iter().enumerate() {
            let direct = PRECISION[1..].iter().all(|deltas| deltas[mode] == base);
            assert_eq!(direct, stores_endpoints_directly(mode), "mode {mode}");
        }
    }

    /// BC6H indexes the first half of the table BC7 uses for two subsets.
    #[test]
    fn the_partition_number_stays_inside_the_shared_table() {
        for mode in MODES.iter().take(10) {
            let bits: u32 = mode
                .fields
                .iter()
                .filter(|f| f.target == PARTITION)
                .map(|f| f.count as u32)
                .sum();
            assert_eq!(bits, 5, "a five-bit partition indexes 32 of the 64 rows");
        }
    }

    #[test]
    fn a_reserved_mode_decodes_to_black() {
        // 0b10011 is one of the four reserved five-bit prefixes.
        let mut block = [0u8; 16];
        block[0] = 0b10011;
        assert_eq!(decode_bc6h(&block, false), [[0.0, 0.0, 0.0, 1.0]; 16]);
    }
}
