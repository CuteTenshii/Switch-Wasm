//! BC7 (`BPTC_UNORM`): 16 bytes of LDR RGBA in one of eight modes.
//!
//! Every mode packs the same fields in the same order — partition, rotation,
//! index selection, endpoints, P-bits, then indices — and differs only in how
//! many bits each gets and which are present at all. [`MODES`] is that table,
//! so the decoder below is one path rather than eight.

use super::{anchor, interpolate, BitReader, Block, PARTITIONS_2, PARTITIONS_3};

/// One row of the BPTC specification's mode table.
struct Mode {
    /// Number of subsets the block is partitioned into.
    subsets: u32,
    partition_bits: u32,
    rotation_bits: u32,
    /// Whether the mode carries the bit that swaps the colour and alpha index
    /// sets.
    index_selection: bool,
    colour_bits: u32,
    alpha_bits: u32,
    /// One P-bit per endpoint, appended below its endpoint's low bit.
    endpoint_p_bits: bool,
    /// One P-bit per *subset*, shared by both of its endpoints.
    shared_p_bits: bool,
    index_bits: u32,
    /// Bit width of the second index set, or zero when the mode has one set.
    index_bits_2: u32,
}

const MODES: [Mode; 8] = [
    Mode { subsets: 3, partition_bits: 4, rotation_bits: 0, index_selection: false,
           colour_bits: 4, alpha_bits: 0, endpoint_p_bits: true,  shared_p_bits: false, index_bits: 3, index_bits_2: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, index_selection: false,
           colour_bits: 6, alpha_bits: 0, endpoint_p_bits: false, shared_p_bits: true,  index_bits: 3, index_bits_2: 0 },
    Mode { subsets: 3, partition_bits: 6, rotation_bits: 0, index_selection: false,
           colour_bits: 5, alpha_bits: 0, endpoint_p_bits: false, shared_p_bits: false, index_bits: 2, index_bits_2: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, index_selection: false,
           colour_bits: 7, alpha_bits: 0, endpoint_p_bits: true,  shared_p_bits: false, index_bits: 2, index_bits_2: 0 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, index_selection: true,
           colour_bits: 5, alpha_bits: 6, endpoint_p_bits: false, shared_p_bits: false, index_bits: 2, index_bits_2: 3 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, index_selection: false,
           colour_bits: 7, alpha_bits: 8, endpoint_p_bits: false, shared_p_bits: false, index_bits: 2, index_bits_2: 2 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 0, index_selection: false,
           colour_bits: 7, alpha_bits: 7, endpoint_p_bits: true,  shared_p_bits: false, index_bits: 4, index_bits_2: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, index_selection: false,
           colour_bits: 5, alpha_bits: 5, endpoint_p_bits: true,  shared_p_bits: false, index_bits: 2, index_bits_2: 0 },
];

/// Left-justify an endpoint channel to eight bits, replicating its high bits
/// into the vacated low ones.
fn unquantize(value: u32, bits: u32) -> u32 {
    if bits >= 8 {
        return value;
    }
    let shifted = value << (8 - bits);
    shifted | (shifted >> bits)
}

pub fn decode_bc7(block: &[u8]) -> Block {
    let mut reader = BitReader::new(block);
    // The mode is a unary prefix: `m` zeroes then a one. All-zero is not a
    // mode, and the specification makes such a block transparent black rather
    // than an error.
    let mut mode_index = 0;
    while mode_index < 8 && reader.read_bit() == 0 {
        mode_index += 1;
    }
    if mode_index == 8 {
        return [[0.0; 4]; 16];
    }
    let mode = &MODES[mode_index];

    let partition = reader.read(mode.partition_bits) as usize;
    let rotation = reader.read(mode.rotation_bits);
    let index_selection = if mode.index_selection { reader.read_bit() } else { 0 };

    // Endpoints are stored channel-major: every endpoint's red, then every
    // green, then blue, then alpha.
    let endpoint_count = (2 * mode.subsets) as usize;
    let mut endpoints = [[0u32; 4]; 6];
    for channel in 0..3 {
        for endpoint in endpoints.iter_mut().take(endpoint_count) {
            endpoint[channel] = reader.read(mode.colour_bits);
        }
    }
    for endpoint in endpoints.iter_mut().take(endpoint_count) {
        endpoint[3] = if mode.alpha_bits > 0 { reader.read(mode.alpha_bits) } else { 255 };
    }

    // A P-bit is one more low bit of precision, either per endpoint or shared
    // by the two endpoints of a subset.
    if mode.endpoint_p_bits || mode.shared_p_bits {
        let mut p = [0u32; 6];
        if mode.endpoint_p_bits {
            for slot in p.iter_mut().take(endpoint_count) {
                *slot = reader.read_bit();
            }
        } else {
            for subset in 0..mode.subsets as usize {
                let bit = reader.read_bit();
                p[2 * subset] = bit;
                p[2 * subset + 1] = bit;
            }
        }
        for (endpoint, &bit) in endpoints.iter_mut().zip(p.iter()).take(endpoint_count) {
            for channel in endpoint.iter_mut().take(3) {
                *channel = (*channel << 1) | bit;
            }
            if mode.alpha_bits > 0 {
                endpoint[3] = (endpoint[3] << 1) | bit;
            }
        }
    }

    let p_width = u32::from(mode.endpoint_p_bits || mode.shared_p_bits);
    let colour_width = mode.colour_bits + p_width;
    let alpha_width = mode.alpha_bits + p_width;
    for endpoint in endpoints.iter_mut().take(endpoint_count) {
        for channel in endpoint.iter_mut().take(3) {
            *channel = unquantize(*channel, colour_width);
        }
        if mode.alpha_bits > 0 {
            endpoint[3] = unquantize(endpoint[3], alpha_width);
        }
    }

    let subset_of = |texel: usize| -> usize {
        match mode.subsets {
            1 => 0,
            2 => PARTITIONS_2[partition][texel] as usize,
            _ => PARTITIONS_3[partition][texel] as usize,
        }
    };

    // Both index sets are stored in texel order, and the anchor texel of each
    // subset spends one fewer bit because its high bit is known to be zero.
    let mut primary = [0u32; 16];
    for (texel, slot) in primary.iter_mut().enumerate() {
        let subset = subset_of(texel);
        let bits = if anchor(mode.subsets, partition, subset) == texel {
            mode.index_bits - 1
        } else {
            mode.index_bits
        };
        *slot = reader.read(bits);
    }
    let mut secondary = [0u32; 16];
    if mode.index_bits_2 > 0 {
        for (texel, slot) in secondary.iter_mut().enumerate() {
            // The second index set has a single subset, so only texel 0 is an
            // anchor however the block is partitioned.
            let bits = if texel == 0 { mode.index_bits_2 - 1 } else { mode.index_bits_2 };
            *slot = reader.read(bits);
        }
    }

    let (colour_index, colour_index_bits, alpha_index, alpha_index_bits) =
        if mode.index_bits_2 == 0 {
            (primary, mode.index_bits, primary, mode.index_bits)
        } else if index_selection == 0 {
            (primary, mode.index_bits, secondary, mode.index_bits_2)
        } else {
            (secondary, mode.index_bits_2, primary, mode.index_bits)
        };

    let mut out = [[0.0f32; 4]; 16];
    for (texel, slot) in out.iter_mut().enumerate() {
        let subset = subset_of(texel);
        let e0 = endpoints[2 * subset];
        let e1 = endpoints[2 * subset + 1];
        let mut rgba = [0u8; 4];
        for channel in 0..3 {
            rgba[channel] = interpolate(
                e0[channel],
                e1[channel],
                colour_index[texel],
                colour_index_bits,
            );
        }
        rgba[3] = if mode.alpha_bits > 0 {
            interpolate(e0[3], e1[3], alpha_index[texel], alpha_index_bits)
        } else {
            255
        };
        // Rotation moves alpha back into the channel it was swapped with at
        // encode time, which is how a mode with one index set can still track
        // a channel that correlates poorly with the other three.
        if rotation != 0 {
            rgba.swap(3, (rotation - 1) as usize);
        }
        *slot = rgba.map(|c| c as f32 / 255.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every BC7 mode spends the block's 128 bits exactly — that is what makes
    /// the format self-delimiting. Adding up the table is therefore a check on
    /// all eight rows at once, and it catches a mistyped field width that no
    /// individual decode would localise.
    #[test]
    fn every_mode_accounts_for_all_128_bits() {
        for (index, mode) in MODES.iter().enumerate() {
            let mut bits = index as u32 + 1; // the unary mode prefix
            bits += mode.partition_bits + mode.rotation_bits + u32::from(mode.index_selection);
            bits += 3 * 2 * mode.subsets * mode.colour_bits;
            if mode.alpha_bits > 0 {
                bits += 2 * mode.subsets * mode.alpha_bits;
            }
            if mode.endpoint_p_bits {
                bits += 2 * mode.subsets;
            }
            if mode.shared_p_bits {
                bits += mode.subsets;
            }
            // One index per texel, less the high bit of each subset's anchor.
            bits += mode.index_bits * 16 - mode.subsets;
            if mode.index_bits_2 > 0 {
                bits += mode.index_bits_2 * 16 - 1;
            }
            assert_eq!(bits, 128, "mode {index}");
        }
    }

    /// A mode with two index sets is the only kind that can select between
    /// them, and only a single-subset mode has the spare bits to carry two.
    #[test]
    fn only_single_subset_modes_carry_a_second_index_set() {
        for (index, mode) in MODES.iter().enumerate() {
            if mode.index_bits_2 > 0 {
                assert_eq!(mode.subsets, 1, "mode {index}");
            }
            if mode.index_selection {
                assert!(mode.index_bits_2 > 0, "mode {index} selects between one index set");
            }
            assert!(!(mode.endpoint_p_bits && mode.shared_p_bits), "mode {index}");
        }
    }
}
