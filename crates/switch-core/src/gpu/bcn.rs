//! Block-compressed texture codecs — the BC (a.k.a. DXT/RGTC/BPTC) family.
//!
//! Every one of these stores a 4x4 block of texels in a fixed 8 or 16 bytes,
//! so a texel is not addressable on its own: reading one means decoding the
//! block that holds it. Each decoder here is a pure function from those bytes
//! to sixteen RGBA texels, which is what lets them be tested against the
//! reference vectors in the specifications rather than against a rendered
//! frame.
//!
//! Channel conventions follow what the hardware hands the sampler, and the
//! TIC's own swizzle then decides where those channels land — BC4 produces
//! its one channel in red, BC5 its two in red and green, and the TIC says
//! whether the rest read as zero or one (see [`crate::gpu::texture`]).

use crate::{Error, Result};

/// A decoded 4x4 block, row-major: texel `(x, y)` is at `y * 4 + x`.
pub type Block = [[f32; 4]; 16];

/// Expand a 5- or 6-bit channel to 8 bits the way the decoders do it, by
/// replicating the high bits into the low ones — so an all-ones input reaches
/// exactly 255 rather than falling short of white.
fn expand(value: u32, bits: u32) -> u8 {
    let shifted = value << (8 - bits);
    (shifted | (shifted >> bits)) as u8
}

/// Unpack an RGB565 endpoint to 8-bit RGB.
fn rgb565(packed: u16) -> [u8; 3] {
    [
        expand((packed as u32 >> 11) & 0x1F, 5),
        expand((packed as u32 >> 5) & 0x3F, 6),
        expand(packed as u32 & 0x1F, 5),
    ]
}

fn u16_le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// The four-entry colour palette a BC1-style colour block describes, as 8-bit
/// RGBA.
///
/// `punchthrough` is BC1's one-bit alpha: when the endpoints are in the
/// second ordering the block trades its fourth colour for transparent black.
/// BC2 and BC3 carry their own alpha and so always take the four-colour
/// ordering, whatever the endpoints happen to compare as.
fn colour_palette(block: &[u8], punchthrough: bool) -> [[u8; 4]; 4] {
    let c0 = u16_le(block, 0);
    let c1 = u16_le(block, 2);
    let a = rgb565(c0);
    let b = rgb565(c1);
    let mut palette = [[0u8; 4]; 4];
    palette[0] = [a[0], a[1], a[2], 255];
    palette[1] = [b[0], b[1], b[2], 255];
    if punchthrough && c0 <= c1 {
        for i in 0..3 {
            palette[2][i] = ((a[i] as u32 + b[i] as u32) / 2) as u8;
        }
        palette[2][3] = 255;
        palette[3] = [0, 0, 0, 0];
    } else {
        for i in 0..3 {
            palette[2][i] = ((2 * a[i] as u32 + b[i] as u32) / 3) as u8;
            palette[3][i] = ((a[i] as u32 + 2 * b[i] as u32) / 3) as u8;
        }
        palette[2][3] = 255;
        palette[3][3] = 255;
    }
    palette
}

/// Decode the 8-byte colour half shared by BC1, BC2 and BC3.
fn colour_block(block: &[u8], punchthrough: bool, out: &mut Block) {
    let palette = colour_palette(block, punchthrough);
    let indices = u32_le(block, 4);
    for (texel, slot) in out.iter_mut().enumerate() {
        let entry = palette[((indices >> (2 * texel)) & 0x3) as usize];
        *slot = [
            entry[0] as f32 / 255.0,
            entry[1] as f32 / 255.0,
            entry[2] as f32 / 255.0,
            entry[3] as f32 / 255.0,
        ];
    }
}

/// The eight-entry palette of an interpolated-scalar block — BC3's alpha half
/// and the whole of BC4 and BC5.
///
/// The endpoint ordering picks between eight evenly spaced values and six plus
/// the two extremes of the range, which is how a block encodes "fully off" and
/// "fully on" exactly rather than approximately.
///
/// Unlike S3TC's colour interpolation, RGTC specifies this arithmetic exactly:
/// integer weights over the stored endpoints, truncating. Doing it in floats
/// instead lands within a part in 255 of every value and on the wrong side of
/// the boundary for some of them, so it is done as specified.
fn scalar_palette(e0: i32, e1: i32, lo: i32, hi: i32, six_value: bool) -> [i32; 8] {
    let mut palette = [0i32; 8];
    palette[0] = e0;
    palette[1] = e1;
    if six_value {
        for (i, slot) in palette.iter_mut().skip(2).take(4).enumerate() {
            let i = i as i32;
            *slot = ((4 - i) * e0 + (1 + i) * e1) / 5;
        }
        palette[6] = lo;
        palette[7] = hi;
    } else {
        for (i, slot) in palette.iter_mut().skip(2).take(6).enumerate() {
            let i = i as i32;
            *slot = ((6 - i) * e0 + (1 + i) * e1) / 7;
        }
    }
    palette
}

/// Decode an 8-byte interpolated-scalar block into `channel` of `out`.
fn scalar_block(block: &[u8], signed: bool, channel: usize, out: &mut Block) {
    let (e0, e1, six_value, lo, hi, scale) = if signed {
        // -128 and -127 both mean -1.0, so the range stays symmetric.
        let a = (block[0] as i8).max(-127) as i32;
        let b = (block[1] as i8).max(-127) as i32;
        (a, b, a <= b, -127, 127, 127.0)
    } else {
        let a = block[0] as i32;
        let b = block[1] as i32;
        (a, b, a <= b, 0, 255, 255.0)
    };
    let palette = scalar_palette(e0, e1, lo, hi, six_value);
    // Sixteen 3-bit indices packed little-endian across the last six bytes.
    let mut bits = 0u64;
    for (i, &byte) in block[2..8].iter().enumerate() {
        bits |= (byte as u64) << (8 * i);
    }
    for (texel, slot) in out.iter_mut().enumerate() {
        let value = palette[((bits >> (3 * texel)) & 0x7) as usize];
        slot[channel] = (value as f32 / scale).clamp(if signed { -1.0 } else { 0.0 }, 1.0);
    }
}

/// BC1 (`DXT1`): 8 bytes, RGB with an optional one-bit alpha.
pub fn decode_bc1(block: &[u8]) -> Block {
    let mut out = [[0.0f32; 4]; 16];
    colour_block(block, true, &mut out);
    out
}

/// BC2 (`DXT23`): 16 bytes, a BC1 colour block over four-bit explicit alpha.
pub fn decode_bc2(block: &[u8]) -> Block {
    let mut out = [[0.0f32; 4]; 16];
    colour_block(&block[8..], false, &mut out);
    for (texel, slot) in out.iter_mut().enumerate() {
        let nibble = (block[texel / 2] >> (4 * (texel % 2))) & 0xF;
        slot[3] = nibble as f32 / 15.0;
    }
    out
}

/// BC3 (`DXT45`): 16 bytes, a BC1 colour block over an interpolated alpha
/// block.
pub fn decode_bc3(block: &[u8]) -> Block {
    let mut out = [[0.0f32; 4]; 16];
    colour_block(&block[8..], false, &mut out);
    scalar_block(&block[..8], false, 3, &mut out);
    out
}

/// BC4 (`DXN1`): 8 bytes, one interpolated channel, delivered in red.
pub fn decode_bc4(block: &[u8], signed: bool) -> Block {
    let mut out = [[0.0f32, 0.0, 0.0, 1.0]; 16];
    scalar_block(block, signed, 0, &mut out);
    out
}

/// BC5 (`DXN2`): 16 bytes, two interpolated channels, in red and green.
pub fn decode_bc5(block: &[u8], signed: bool) -> Block {
    let mut out = [[0.0f32, 0.0, 0.0, 1.0]; 16];
    scalar_block(&block[..8], signed, 0, &mut out);
    scalar_block(&block[8..], signed, 1, &mut out);
    out
}

/// Reads fields out of a 128-bit block least-significant bit first, which is
/// the order both BPTC formats define their fields in.
struct BitReader<'a> {
    bytes: &'a [u8],
    position: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader { bytes, position: 0 }
    }

    fn read(&mut self, count: u32) -> u32 {
        let mut value = 0u32;
        for i in 0..count {
            let bit = self.position + i;
            let byte = self.bytes[(bit / 8) as usize];
            value |= (((byte >> (bit % 8)) & 1) as u32) << i;
        }
        self.position += count;
        value
    }

    fn read_bit(&mut self) -> u32 {
        self.read(1)
    }
}

/// Which subset each texel of a 4x4 block belongs to, for the two- and
/// three-subset partitionings BC6H and BC7 share (BPTC specification,
/// tables "Partition Table for 2 Subsets" and "for 3 Subsets").
const PARTITIONS_2: [[u8; 16]; 64] = [
    [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1], [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1], [0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1], [0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1],
    [0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1], [0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1],
    [0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    [0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1], [0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0], [0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0],
    [0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0], [0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1],
    [0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0],
    [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0], [0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0],
    [0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0], [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0], [0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0],
    [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0], [0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0],
    [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0], [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0],
    [0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1], [0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1],
    [0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0], [0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0],
    [0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0], [0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0],
    [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0], [0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],
    [0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1], [0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0],
    [0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0], [0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0], [0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0],
    [0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1], [0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1],
    [0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0], [0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0],
    [0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1], [0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1],
    [0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1], [0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1],
    [0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1], [0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0], [0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1],
];

const PARTITIONS_3: [[u8; 16]; 64] = [
    [0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2], [0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1],
    [0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1], [0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2], [0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2],
    [0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1], [0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1],
    [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2], [0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2],
    [0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2], [0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2],
    [0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2], [0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2],
    [0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2], [0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0],
    [0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2], [0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0],
    [0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2], [0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1],
    [0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2], [0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1],
    [0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2], [0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0],
    [0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0], [0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2],
    [0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0], [0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1],
    [0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2], [0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2],
    [0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1], [0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1],
    [0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2], [0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1],
    [0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2], [0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0],
    [0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0], [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0],
    [0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0], [0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1],
    [0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1], [0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2],
    [0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1], [0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2],
    [0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1], [0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1],
    [0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1], [0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1],
    [0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2], [0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1],
    [0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2], [0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2],
    [0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2], [0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2],
    [0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2], [0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2],
    [0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2], [0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2],
    [0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2], [0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2],
    [0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1], [0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2],
    [0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2], [0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0],
];

/// Which texel holds the implicit high-bit-zero index for the second subset
/// of each two-subset partitioning.
const ANCHORS_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,  2,  8,  2,  2,  8,  8,  2,  2,
    15, 15,  6,  8,  2,  8, 15, 15,  2,  8,  2,  2,  2, 15, 15,  6,
     6,  2,  6,  8, 15, 15,  2,  2, 15, 15, 15, 15, 15,  2,  2, 15,
];

/// The same, for the second and third subsets of each three-subset
/// partitioning.
const ANCHORS_3_SECOND: [u8; 64] = [
     3,  3, 15, 15,  8,  3, 15, 15,  8,  8,  6,  6,  6,  5,  3,  3,
     3,  3,  8, 15,  3,  3,  6, 10,  5,  8,  8,  6,  8,  5, 15, 15,
     8, 15,  3,  5,  6, 10,  8, 15, 15,  3, 15,  5, 15, 15, 15, 15,
     3, 15,  5,  5,  5,  8,  5, 10,  5, 10,  8, 13, 15, 12,  3,  3,
];

const ANCHORS_3_THIRD: [u8; 64] = [
    15,  8,  8,  3, 15, 15,  3,  8, 15, 15, 15, 15, 15, 15, 15,  8,
    15,  8, 15,  3, 15,  8, 15,  8,  3, 15,  6, 10, 15, 15, 10,  8,
    15,  3, 15, 10, 10,  8,  9, 10,  6, 15,  8, 15,  3,  6,  6,  8,
    15,  3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  3, 15, 15,  8,
];

/// The anchor texel of `subset` under a partitioning, which stores one fewer
/// index bit because its high bit is known to be zero.
fn anchor(subsets: u32, partition: usize, subset: usize) -> usize {
    match (subsets, subset) {
        (_, 0) => 0,
        (2, _) => ANCHORS_2[partition] as usize,
        (3, 1) => ANCHORS_3_SECOND[partition] as usize,
        _ => ANCHORS_3_THIRD[partition] as usize,
    }
}

/// The BPTC interpolation weights for 2-, 3- and 4-bit indices.
const WEIGHTS_2: [u32; 4] = [0, 21, 43, 64];
const WEIGHTS_3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS_4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn weight(index: u32, bits: u32) -> u32 {
    match bits {
        2 => WEIGHTS_2[index as usize],
        3 => WEIGHTS_3[index as usize],
        _ => WEIGHTS_4[index as usize],
    }
}

/// Interpolate two endpoint channels, as the BPTC specification's
/// `interpolate` does: a 6-bit weight and a rounding add.
fn interpolate(e0: u32, e1: u32, index: u32, bits: u32) -> u8 {
    let w = weight(index, bits);
    (((64 - w) * e0 + w * e1 + 32) >> 6) as u8
}

pub use astc::{decode_astc, MAX_TEXELS};
mod astc;

pub use bc6h::decode_bc6h;
mod bc6h;

pub use bc7::decode_bc7;
mod bc7;

/// Decode one block of `codec` into `out`, which must hold at least as many
/// texels as [`Codec::block_size`] describes.
///
/// This is the general entry point: the BC codecs all cover a 4x4 block, but
/// an ASTC one covers anything up to 12x12, so a caller cannot assume a
/// sixteen-texel result.
pub fn decode_into(codec: Codec, bytes: &[u8], out: &mut [[f32; 4]]) -> Result<()> {
    if let Codec::Astc { width, height } = codec {
        return decode_astc(bytes, width as u32, height as u32, out);
    }
    let block = decode(codec, bytes)?;
    if out.len() < 16 {
        return Err(Error::Gpu("bcn: destination is smaller than one block".into()));
    }
    out[..16].copy_from_slice(&block);
    Ok(())
}

/// Decode one 4x4 block of `codec` from `bytes`.
pub fn decode(codec: Codec, bytes: &[u8]) -> Result<Block> {
    if bytes.len() < codec.bytes_per_block() as usize {
        return Err(Error::Gpu(format!(
            "bcn: {:?} needs {} bytes per block, got {}",
            codec,
            codec.bytes_per_block(),
            bytes.len()
        )));
    }
    Ok(match codec {
        Codec::Astc { .. } => {
            return Err(Error::Gpu(
                "bcn: an ASTC block is not 4x4; decode_into takes its footprint".into(),
            ))
        }
        Codec::Bc1 => decode_bc1(bytes),
        Codec::Bc2 => decode_bc2(bytes),
        Codec::Bc3 => decode_bc3(bytes),
        Codec::Bc4Unorm => decode_bc4(bytes, false),
        Codec::Bc4Snorm => decode_bc4(bytes, true),
        Codec::Bc5Unorm => decode_bc5(bytes, false),
        Codec::Bc5Snorm => decode_bc5(bytes, true),
        Codec::Bc6hUf16 => decode_bc6h(bytes, false),
        Codec::Bc6hSf16 => decode_bc6h(bytes, true),
        Codec::Bc7 => decode_bc7(bytes),
    })
}

/// Which block codec a texture's texels are stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Bc1,
    Bc2,
    Bc3,
    Bc4Unorm,
    Bc4Snorm,
    Bc5Unorm,
    Bc5Snorm,
    Bc6hUf16,
    Bc6hSf16,
    Bc7,
    /// ASTC LDR, whose footprint is chosen per texture rather than fixed.
    Astc { width: u8, height: u8 },
}

impl Codec {
    /// How many bytes one block occupies. Every BC codec is 4x4 and differs
    /// only in this; ASTC is always sixteen bytes whatever its footprint.
    pub fn bytes_per_block(&self) -> u32 {
        match self {
            Codec::Bc1 | Codec::Bc4Unorm | Codec::Bc4Snorm => 8,
            _ => 16,
        }
    }

    /// The footprint one block covers, in texels.
    pub fn block_size(&self) -> (u32, u32) {
        match self {
            Codec::Astc { width, height } => (*width as u32, *height as u32),
            _ => (4, 4),
        }
    }

    /// Whether the codec carries high dynamic range, so a caller knows not to
    /// clamp what comes out of it into `[0, 1]`.
    pub fn is_hdr(&self) -> bool {
        matches!(self, Codec::Bc6hUf16 | Codec::Bc6hSf16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0 / 255.0
    }

    /// The two partition tables and the three anchor tables are separate
    /// transcriptions of the BPTC specification, and they encode overlapping
    /// facts: an anchor must name a texel that actually belongs to the subset
    /// it anchors. Checking them against each other catches a typo in either,
    /// which is the failure mode a table this size invites and which no
    /// rendered frame would localise.
    #[test]
    fn the_partition_and_anchor_tables_agree() {
        for (p, pattern) in PARTITIONS_2.iter().enumerate() {
            assert!(pattern.iter().all(|&s| s < 2), "partition {p} has a third subset");
            assert!(pattern.contains(&0) && pattern.contains(&1), "partition {p} is not 2-subset");
            assert_eq!(pattern[0], 0, "texel 0 must anchor subset 0 in partition {p}");
            assert_eq!(pattern[ANCHORS_2[p] as usize], 1, "anchor of partition {p}");
        }
        for (p, pattern) in PARTITIONS_3.iter().enumerate() {
            assert!(pattern.iter().all(|&s| s < 3), "partition {p} has a fourth subset");
            for subset in 0..3u8 {
                assert!(pattern.contains(&subset), "partition {p} is missing subset {subset}");
            }
            assert_eq!(pattern[0], 0, "texel 0 must anchor subset 0 in partition {p}");
            assert_eq!(pattern[ANCHORS_3_SECOND[p] as usize], 1, "second anchor of partition {p}");
            assert_eq!(pattern[ANCHORS_3_THIRD[p] as usize], 2, "third anchor of partition {p}");
        }
    }

    /// The interpolation weights are the specification's, and both ends of
    /// each table have to be exact or every block drifts.
    #[test]
    fn the_interpolation_weights_span_the_whole_range() {
        for (bits, table) in [(2u32, &WEIGHTS_2[..]), (3, &WEIGHTS_3[..]), (4, &WEIGHTS_4[..])] {
            assert_eq!(table[0], 0);
            assert_eq!(table[table.len() - 1], 64);
            assert_eq!(table.len(), 1 << bits);
            assert!(table.windows(2).all(|w| w[0] < w[1]), "{bits}-bit weights are not sorted");
        }
        // A weight of 0 is the first endpoint exactly, and 64 the second.
        assert_eq!(interpolate(10, 200, 0, 2), 10);
        assert_eq!(interpolate(10, 200, 3, 2), 200);
    }

    #[test]
    fn bc1_with_equal_endpoints_is_one_flat_colour() {
        // Both endpoints pure red, every index 0.
        let mut block = [0u8; 8];
        block[0..2].copy_from_slice(&0xF800u16.to_le_bytes());
        block[2..4].copy_from_slice(&0xF800u16.to_le_bytes());
        let out = decode_bc1(&block);
        for texel in out {
            assert_eq!(texel, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn bc1_interpolates_between_its_endpoints() {
        // c0 = white (> c1), c1 = black, so the four-colour ordering applies
        // and indices 2 and 3 are the two-thirds points.
        let mut block = [0u8; 8];
        block[0..2].copy_from_slice(&0xFFFFu16.to_le_bytes());
        block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
        // Texels 0..4 take indices 0, 1, 2, 3.
        block[4] = 0b11_10_01_00;
        let out = decode_bc1(&block);
        assert_eq!(out[0], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out[1], [0.0, 0.0, 0.0, 1.0]);
        assert!(approx(out[2][0], 2.0 / 3.0), "got {}", out[2][0]);
        assert!(approx(out[3][0], 1.0 / 3.0), "got {}", out[3][0]);
        assert_eq!(out[2][3], 1.0);
    }

    #[test]
    fn bc1_punchthrough_makes_the_fourth_index_transparent() {
        // c0 <= c1 selects the three-colour ordering.
        let mut block = [0u8; 8];
        block[0..2].copy_from_slice(&0x0000u16.to_le_bytes());
        block[2..4].copy_from_slice(&0xFFFFu16.to_le_bytes());
        block[4] = 0b11_10_01_00;
        let out = decode_bc1(&block);
        assert_eq!(out[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(out[1], [1.0, 1.0, 1.0, 1.0]);
        assert!(approx(out[2][0], 0.5), "midpoint, got {}", out[2][0]);
        assert_eq!(out[3], [0.0, 0.0, 0.0, 0.0], "index 3 is transparent black");
    }

    #[test]
    fn bc2_reads_four_bit_alpha_straight_out_of_the_block() {
        let mut block = [0u8; 16];
        // Texel 0 alpha = 0x0, texel 1 = 0xF, texel 2 = 0x8.
        block[0] = 0xF0;
        block[1] = 0x08;
        block[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes());
        block[10..12].copy_from_slice(&0xFFFFu16.to_le_bytes());
        let out = decode_bc2(&block);
        assert_eq!(out[0][3], 0.0);
        assert_eq!(out[1][3], 1.0);
        assert!(approx(out[2][3], 8.0 / 15.0));
    }

    #[test]
    fn bc3_alpha_uses_both_endpoint_orderings() {
        // a0 > a1: eight interpolated values, no exact 0 or 255 in the palette.
        let eight = scalar_palette(255, 0, 0, 255, false);
        assert_eq!(eight[0], 255);
        assert_eq!(eight[1], 0);
        assert_eq!(eight[2], 6 * 255 / 7);
        // a0 <= a1: six interpolated plus the exact extremes of the range.
        let six = scalar_palette(0, 255, 0, 255, true);
        assert_eq!(six[6], 0);
        assert_eq!(six[7], 255);
        assert_eq!(six[2], 255 / 5);
    }

    #[test]
    fn bc4_puts_its_one_channel_in_red_and_bc5_adds_green() {
        let mut block = [0u8; 8];
        block[0] = 255;
        block[1] = 0;
        let out = decode_bc4(&block, false);
        assert_eq!(out[0], [1.0, 0.0, 0.0, 1.0], "index 0 is the first endpoint");

        let mut wide = [0u8; 16];
        wide[0] = 255;
        wide[1] = 0;
        wide[8] = 0;
        wide[9] = 255;
        let out = decode_bc5(&wide, false);
        assert_eq!(out[0][0], 1.0);
        assert_eq!(out[0][1], 0.0);
        assert_eq!(out[0][3], 1.0, "BC5 has no alpha of its own");
    }

    #[test]
    fn bc4_signed_keeps_the_range_symmetric() {
        let mut block = [0u8; 8];
        block[0] = (-127i8) as u8;
        block[1] = 127u8;
        let out = decode_bc4(&block, true);
        assert_eq!(out[0][0], -1.0);
        // -128 is clamped onto -127 so that it still means exactly -1.
        block[0] = (-128i8) as u8;
        let out = decode_bc4(&block, true);
        assert_eq!(out[0][0], -1.0);
    }

    /// Mode 6 is the single-subset, 4-bit-index mode a BC7 encoder reaches for
    /// most often, and it has no partition or rotation to obscure the result:
    /// 1 mode bit, then 7+1 bits per channel per endpoint, then the indices.
    #[test]
    fn bc7_mode_6_interpolates_a_single_subset() {
        let mut bits: u128 = 0;
        let mut at = 0;
        let mut put = |value: u128, count: u32, at: &mut u32| {
            bits |= value << *at;
            *at += count;
        };
        put(0, 6, &mut at); // mode 6 is a unary prefix: six zeroes...
        put(1, 1, &mut at); // ...then the terminating one
        // Endpoint 0 = black, endpoint 1 = white, channel-major R0 R1 G0 G1...
        for _ in 0..4 {
            put(0, 7, &mut at); // endpoint 0
            put(0x7F, 7, &mut at); // endpoint 1
        }
        put(0, 1, &mut at); // P-bit of endpoint 0
        put(1, 1, &mut at); // P-bit of endpoint 1
        put(0, 3, &mut at); // texel 0 is the anchor: 3 bits, not 4
        put(15, 4, &mut at); // texel 1 = the far endpoint
        put(8, 4, &mut at); // texel 2 = just past the middle

        let block = bits.to_le_bytes();
        let out = decode_bc7(&block);
        assert_eq!(out[0], [0.0, 0.0, 0.0, 0.0], "index 0 is endpoint 0, alpha included");
        assert_eq!(out[1], [1.0, 1.0, 1.0, 1.0], "index 15 is endpoint 1");
        assert!(approx(out[2][0], 34.0 / 64.0), "got {}", out[2][0]);
    }

    #[test]
    fn a_bc7_block_with_no_mode_bit_is_transparent_black() {
        assert_eq!(decode_bc7(&[0u8; 16]), [[0.0; 4]; 16]);
    }

    #[test]
    fn every_codec_reports_the_size_it_reads() {
        for codec in [Codec::Bc1, Codec::Bc4Unorm, Codec::Bc4Snorm] {
            assert_eq!(codec.bytes_per_block(), 8);
        }
        for codec in [Codec::Bc2, Codec::Bc3, Codec::Bc5Unorm, Codec::Bc5Snorm, Codec::Bc7] {
            assert_eq!(codec.bytes_per_block(), 16);
        }
        // A short block is refused rather than read past its end.
        assert!(decode(Codec::Bc7, &[0u8; 8]).is_err());
        assert!(decode(Codec::Bc1, &[0u8; 8]).is_ok());
    }
}
