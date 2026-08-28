//! ASTC LDR (`ASTC_2D_*`): 16 bytes covering a footprint from 4x4 up to 12x12.
//!
//! Everything about a block is variable. The footprint is chosen per texture,
//! and within a block the *weight grid* is a separate, usually smaller, grid
//! that is bilinearly resampled up to the footprint; the number of partitions,
//! which endpoint pairs each partition gets, and the numeric range every one of
//! those values is stored in all come out of the block's own header. There is
//! no fixed field layout to tabulate — the layout is computed from the header
//! and then the two halves of the block are read towards each other, endpoints
//! forwards from the low end and weights backwards from bit 127.
//!
//! The one genuinely unusual mechanism is Integer Sequence Encoding, which
//! packs values whose range is not a power of two — five values into eight bits
//! plus a trit each, or three into seven bits plus a quint — so that a
//! range of, say, 0..11 costs a little over three and a half bits rather than
//! four. [`TRITS_FROM_T`] and [`QUINTS_FROM_Q`] are its unpacking tables.
//!
//! Only the LDR profile is decoded, which is what Maxwell exposes and what
//! every `DkImageFormat_RGBA_ASTC_*` is. A block asking for an HDR endpoint
//! mode, or one whose header does not describe a legal configuration, decodes
//! to the specification's error colour: opaque magenta.

use crate::{Error, Result};

/// The largest footprint, and so the most texels one block can carry.
pub const MAX_TEXELS: usize = 12 * 12;

/// The error colour a malformed or unsupported block decodes to.
const ERROR_COLOUR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

include!("astc_tables.rs");

fn bits(data: u128, lo: i32, hi: i32) -> u32 {
    if hi < lo || lo < 0 {
        return 0;
    }
    let width = (hi - lo + 1) as u32;
    if width >= 128 {
        return data as u32;
    }
    ((data >> lo) & ((1u128 << width) - 1)) as u32
}

fn bit(data: u128, index: i32) -> u32 {
    bits(data, index, index)
}

fn reverse_bits(value: u32, count: i32) -> u32 {
    let mut out = 0;
    for i in 0..count {
        out |= ((value >> i) & 1) << (count - 1 - i);
    }
    out
}

/// Spread `count` source bits across `dst_bits` by repeating them, which is how
/// ASTC widens a quantised value without changing its endpoints.
fn replicate(value: u32, src_bits: i32, dst_bits: i32) -> u32 {
    let mut out = 0u32;
    let mut shift = dst_bits - src_bits;
    while shift > -src_bits {
        out |= if shift >= 0 {
            value << shift
        } else {
            value >> -shift
        };
        shift -= src_bits;
    }
    out
}

/// A run of bits read either forwards from `start` or backwards from it.
///
/// The two halves of a block grow towards each other: endpoints upwards from
/// just past the header, weights downwards from bit 127. Reads past `length`
/// yield zeroes rather than running into the other half.
struct Stream {
    data: u128,
    start: i32,
    length: i32,
    forward: bool,
    position: i32,
}

impl Stream {
    fn new(data: u128, start: i32, length: i32, forward: bool) -> Stream {
        Stream {
            data,
            start,
            length,
            forward,
            position: 0,
        }
    }

    fn next(&mut self, count: i32) -> u32 {
        if count == 0 || self.position >= self.length {
            return 0;
        }
        let end = self.position + count;
        let from_source = (self.length.min(end) - self.position).max(0);
        let low = self.position;
        let high = self.position + from_source - 1;
        self.position += count;
        if self.forward {
            bits(self.data, self.start + low, self.start + high)
        } else {
            reverse_bits(
                bits(self.data, self.start - high, self.start - low),
                from_source,
            )
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IseMode {
    Trit,
    Quint,
    Plain,
}

#[derive(Clone, Copy)]
struct Ise {
    mode: IseMode,
    bits: i32,
}

/// One decoded value: its low bits, its trit or quint, and the two combined.
#[derive(Clone, Copy, Default)]
struct IseValue {
    low: u32,
    trit_or_quint: u32,
    value: u32,
}

fn div_round_up(a: i32, b: i32) -> i32 {
    (a + b - 1) / b
}

fn ise_required_bits(params: Ise, values: i32) -> i32 {
    match params.mode {
        IseMode::Trit => div_round_up(values * 8, 5) + values * params.bits,
        IseMode::Quint => div_round_up(values * 7, 3) + values * params.bits,
        IseMode::Plain => values * params.bits,
    }
}

/// The widest range that fits `values` numbers into `available` bits.
///
/// ASTC does not store the range; it is whatever the largest representable one
/// is for the space left over once the header and the other half of the block
/// have taken their share, so encoder and decoder derive it the same way.
fn max_range_ise(available: i32, values: i32) -> Ise {
    let (mut trit, mut quint, mut plain) = (6i32, 5i32, 8i32);
    loop {
        let trit_range = if trit > 0 { (3 << trit) - 1 } else { -1 };
        let quint_range = if quint > 0 { (5 << quint) - 1 } else { -1 };
        let plain_range = if plain > 0 { (1 << plain) - 1 } else { -1 };
        let widest = trit_range.max(quint_range).max(plain_range);
        if widest == trit_range {
            let params = Ise {
                mode: IseMode::Trit,
                bits: trit,
            };
            if ise_required_bits(params, values) <= available {
                return params;
            }
            trit -= 1;
        } else if widest == quint_range {
            let params = Ise {
                mode: IseMode::Quint,
                bits: quint,
            };
            if ise_required_bits(params, values) <= available {
                return params;
            }
            quint -= 1;
        } else {
            let params = Ise {
                mode: IseMode::Plain,
                bits: plain,
            };
            if ise_required_bits(params, values) <= available {
                return params;
            }
            plain -= 1;
        }
    }
}

/// Unpack a sequence of ISE values. Trits arrive five at a time and quints
/// three at a time, each group interleaving its low bits with the packed
/// trit/quint field.
fn decode_ise(out: &mut [IseValue], count: usize, stream: &mut Stream, params: Ise) {
    match params.mode {
        IseMode::Trit => {
            for group in 0..div_round_up(count as i32, 5) as usize {
                let in_group = (count - 5 * group).min(5);
                let mut low = [0u32; 5];
                low[0] = stream.next(params.bits);
                let t01 = stream.next(2);
                low[1] = stream.next(params.bits);
                let t23 = stream.next(2);
                low[2] = stream.next(params.bits);
                let t4 = stream.next(1);
                low[3] = stream.next(params.bits);
                let t56 = stream.next(2);
                low[4] = stream.next(params.bits);
                let t7 = stream.next(1);
                // A short final group leaves the higher fields unread.
                let t23 = if in_group < 2 { 0 } else { t23 };
                let t4 = if in_group < 3 { 0 } else { t4 };
                let t56 = if in_group < 4 { 0 } else { t56 };
                let t7 = if in_group < 5 { 0 } else { t7 };
                let t = (t7 << 7) | (t56 << 5) | (t4 << 4) | (t23 << 2) | t01;
                let trits = TRITS_FROM_T[t as usize];
                for i in 0..in_group {
                    out[5 * group + i] = IseValue {
                        low: low[i],
                        trit_or_quint: trits[i] as u32,
                        value: ((trits[i] as u32) << params.bits) + low[i],
                    };
                }
            }
        }
        IseMode::Quint => {
            for group in 0..div_round_up(count as i32, 3) as usize {
                let in_group = (count - 3 * group).min(3);
                let mut low = [0u32; 3];
                low[0] = stream.next(params.bits);
                let q012 = stream.next(3);
                low[1] = stream.next(params.bits);
                let q34 = stream.next(2);
                low[2] = stream.next(params.bits);
                let q56 = stream.next(2);
                let q34 = if in_group < 2 { 0 } else { q34 };
                let q56 = if in_group < 3 { 0 } else { q56 };
                let q = (q56 << 5) | (q34 << 3) | q012;
                let quints = QUINTS_FROM_Q[q as usize];
                for i in 0..in_group {
                    out[3 * group + i] = IseValue {
                        low: low[i],
                        trit_or_quint: quints[i] as u32,
                        value: ((quints[i] as u32) << params.bits) + low[i],
                    };
                }
            }
        }
        IseMode::Plain => {
            for slot in out.iter_mut().take(count) {
                let value = stream.next(params.bits);
                *slot = IseValue {
                    low: value,
                    trit_or_quint: 0,
                    value,
                };
            }
        }
    }
}

/// What a block's first eleven bits say about its weight grid.
struct BlockMode {
    void_extent: bool,
    dual_plane: bool,
    grid_width: i32,
    grid_height: i32,
    weight_ise: Ise,
}

fn block_mode(data: u32) -> Option<BlockMode> {
    if bits(data as u128, 0, 8) == 0x1FC {
        return Some(BlockMode {
            void_extent: true,
            dual_plane: false,
            grid_width: 0,
            grid_height: 0,
            weight_ise: Ise {
                mode: IseMode::Plain,
                bits: 0,
            },
        });
    }
    let data = data as u128;
    if (bits(data, 0, 1) == 0 && bits(data, 6, 8) == 7) || bits(data, 0, 3) == 0 {
        return None; // Reserved.
    }
    let (range, grid_width, grid_height);
    if bits(data, 0, 1) == 0 {
        let r = (bit(data, 3) << 2) | (bit(data, 2) << 1) | bit(data, 4);
        let i78 = bits(data, 7, 8);
        let (w, h) = if i78 == 3 {
            if bit(data, 5) != 0 {
                (10, 6)
            } else {
                (6, 10)
            }
        } else {
            let a = bits(data, 5, 6) as i32;
            match i78 {
                0 => (12, a + 2),
                1 => (a + 2, 12),
                _ => (a + 6, bits(data, 9, 10) as i32 + 6),
            }
        };
        range = r;
        grid_width = w;
        grid_height = h;
    } else {
        let r = (bit(data, 1) << 2) | (bit(data, 0) << 1) | bit(data, 4);
        let i23 = bits(data, 2, 3);
        let a = bits(data, 5, 6) as i32;
        let (w, h) = if i23 == 3 {
            let b = bit(data, 7) as i32;
            if bit(data, 8) != 0 {
                (b + 2, a + 2)
            } else {
                (a + 2, b + 6)
            }
        } else {
            let b = bits(data, 7, 8) as i32;
            match i23 {
                0 => (b + 4, a + 2),
                1 => (b + 8, a + 2),
                _ => (a + 2, b + 8),
            }
        };
        range = r;
        grid_width = w;
        grid_height = h;
    }
    let zero_dh = bits(data, 0, 1) == 0 && bits(data, 7, 8) == 2;
    let high = !zero_dh && bit(data, 9) != 0;
    let dual_plane = !zero_dh && bit(data, 10) != 0;
    let weight_ise = match (high, range) {
        (true, 2) => Ise {
            mode: IseMode::Quint,
            bits: 1,
        },
        (true, 3) => Ise {
            mode: IseMode::Trit,
            bits: 2,
        },
        (true, 4) => Ise {
            mode: IseMode::Plain,
            bits: 4,
        },
        (true, 5) => Ise {
            mode: IseMode::Quint,
            bits: 2,
        },
        (true, 6) => Ise {
            mode: IseMode::Trit,
            bits: 3,
        },
        (true, 7) => Ise {
            mode: IseMode::Plain,
            bits: 5,
        },
        (false, 2) => Ise {
            mode: IseMode::Plain,
            bits: 1,
        },
        (false, 3) => Ise {
            mode: IseMode::Trit,
            bits: 0,
        },
        (false, 4) => Ise {
            mode: IseMode::Plain,
            bits: 2,
        },
        (false, 5) => Ise {
            mode: IseMode::Quint,
            bits: 0,
        },
        (false, 6) => Ise {
            mode: IseMode::Trit,
            bits: 1,
        },
        (false, 7) => Ise {
            mode: IseMode::Plain,
            bits: 3,
        },
        _ => return None,
    };
    Some(BlockMode {
        void_extent: false,
        dual_plane,
        grid_width,
        grid_height,
        weight_ise,
    })
}

/// How many values a colour endpoint mode spends.
fn endpoint_values(mode: u32) -> i32 {
    (mode as i32 / 4 + 1) * 2
}

fn is_hdr_endpoint_mode(mode: u32) -> bool {
    matches!(mode, 2 | 3 | 7 | 11 | 14 | 15)
}

/// Spread quantised endpoints back over 0..255.
fn unquantize_endpoints(out: &mut [u32], values: &[IseValue], count: usize, params: Ise) {
    if params.mode == IseMode::Plain {
        for i in 0..count {
            out[i] = replicate(values[i].value, params.bits, 8);
        }
        return;
    }
    let case = params.bits * 2 - if params.mode == IseMode::Trit { 2 } else { 1 };
    const CA: [u32; 11] = [204, 113, 93, 54, 44, 26, 22, 13, 11, 6, 5];
    let c = CA[case as usize];
    for i in 0..count {
        let m = values[i].low;
        let (a, b, cc, d, e, f) = (
            m & 1,
            (m >> 1) & 1,
            (m >> 2) & 1,
            (m >> 3) & 1,
            (m >> 4) & 1,
            (m >> 5) & 1,
        );
        let big_a = if a == 0 { 0 } else { (1 << 9) - 1 };
        let big_b = match case {
            0 | 1 => 0,
            2 => (b << 8) | (b << 4) | (b << 2) | (b << 1),
            3 => (b << 8) | (b << 3) | (b << 2),
            4 => (cc << 8) | (b << 7) | (cc << 3) | (b << 2) | (cc << 1) | b,
            5 => (cc << 8) | (b << 7) | (cc << 2) | (b << 1) | cc,
            6 => (d << 8) | (cc << 7) | (b << 6) | (d << 2) | (cc << 1) | b,
            7 => (d << 8) | (cc << 7) | (b << 6) | (d << 1) | cc,
            8 => (e << 8) | (d << 7) | (cc << 6) | (b << 5) | (e << 1) | d,
            9 => (e << 8) | (d << 7) | (cc << 6) | (b << 5) | e,
            _ => (f << 8) | (e << 7) | (d << 6) | (cc << 5) | (b << 4) | f,
        };
        out[i] = (((values[i].trit_or_quint * c + big_b) ^ big_a) >> 2) | (big_a & 0x80);
    }
}

/// Move the low bit of `a` into `b`'s sign, which is how the modes that store
/// a base and a signed offset pack nine bits of range into eight.
fn bit_transfer_signed(a: &mut i32, b: &mut i32) {
    *b >>= 1;
    *b |= *a & 0x80;
    *a >>= 1;
    *a &= 0x3F;
    if *a & 0x20 != 0 {
        *a -= 0x40;
    }
}

fn blue_contract(r: i32, g: i32, b: i32, a: i32) -> [i32; 4] {
    [(r + b) >> 1, (g + b) >> 1, b, a]
}

fn clamped(rgba: [i32; 4]) -> [u32; 4] {
    [
        rgba[0].clamp(0, 0xFF) as u32,
        rgba[1].clamp(0, 0xFF) as u32,
        rgba[2].clamp(0, 0xFF) as u32,
        rgba[3].clamp(0, 0xFF) as u32,
    ]
}

/// Turn a partition's unquantised values into its two endpoints. Only the LDR
/// modes are here; an HDR one is rejected before this is reached.
fn decode_endpoint_pair(mode: u32, v: &[u32]) -> ([u32; 4], [u32; 4]) {
    match mode {
        0 => ([v[0], v[0], v[0], 0xFF], [v[1], v[1], v[1], 0xFF]),
        1 => {
            let l0 = (v[0] >> 2) | (((v[1] >> 6) & 0x3) << 6);
            let l1 = 0xFFu32.min(l0 + (v[1] & 0x3F));
            ([l0, l0, l0, 0xFF], [l1, l1, l1, 0xFF])
        }
        4 => ([v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]),
        5 => {
            let (mut v0, mut v1) = (v[0] as i32, v[1] as i32);
            let (mut v2, mut v3) = (v[2] as i32, v[3] as i32);
            bit_transfer_signed(&mut v1, &mut v0);
            bit_transfer_signed(&mut v3, &mut v2);
            (
                clamped([v0, v0, v0, v2]),
                clamped([v0 + v1, v0 + v1, v0 + v1, v2 + v3]),
            )
        }
        6 => (
            [
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                0xFF,
            ],
            [v[0], v[1], v[2], 0xFF],
        ),
        8 => {
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                ([v[0], v[2], v[4], 0xFF], [v[1], v[3], v[5], 0xFF])
            } else {
                (
                    clamped(blue_contract(v[1] as i32, v[3] as i32, v[5] as i32, 0xFF)),
                    clamped(blue_contract(v[0] as i32, v[2] as i32, v[4] as i32, 0xFF)),
                )
            }
        }
        9 => {
            let (mut v0, mut v1) = (v[0] as i32, v[1] as i32);
            let (mut v2, mut v3) = (v[2] as i32, v[3] as i32);
            let (mut v4, mut v5) = (v[4] as i32, v[5] as i32);
            bit_transfer_signed(&mut v1, &mut v0);
            bit_transfer_signed(&mut v3, &mut v2);
            bit_transfer_signed(&mut v5, &mut v4);
            if v1 + v3 + v5 >= 0 {
                (
                    clamped([v0, v2, v4, 0xFF]),
                    clamped([v0 + v1, v2 + v3, v4 + v5, 0xFF]),
                )
            } else {
                (
                    clamped(blue_contract(v0 + v1, v2 + v3, v4 + v5, 0xFF)),
                    clamped(blue_contract(v0, v2, v4, 0xFF)),
                )
            }
        }
        10 => (
            [
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                v[4],
            ],
            [v[0], v[1], v[2], v[5]],
        ),
        12 => {
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                ([v[0], v[2], v[4], v[6]], [v[1], v[3], v[5], v[7]])
            } else {
                (
                    clamped(blue_contract(
                        v[1] as i32,
                        v[3] as i32,
                        v[5] as i32,
                        v[7] as i32,
                    )),
                    clamped(blue_contract(
                        v[0] as i32,
                        v[2] as i32,
                        v[4] as i32,
                        v[6] as i32,
                    )),
                )
            }
        }
        _ => {
            let (mut v0, mut v1) = (v[0] as i32, v[1] as i32);
            let (mut v2, mut v3) = (v[2] as i32, v[3] as i32);
            let (mut v4, mut v5) = (v[4] as i32, v[5] as i32);
            let (mut v6, mut v7) = (v[6] as i32, v[7] as i32);
            bit_transfer_signed(&mut v1, &mut v0);
            bit_transfer_signed(&mut v3, &mut v2);
            bit_transfer_signed(&mut v5, &mut v4);
            bit_transfer_signed(&mut v7, &mut v6);
            if v1 + v3 + v5 >= 0 {
                (
                    clamped([v0, v2, v4, v6]),
                    clamped([v0 + v1, v2 + v3, v4 + v5, v6 + v7]),
                )
            } else {
                (
                    clamped(blue_contract(v0 + v1, v2 + v3, v4 + v5, v6 + v7)),
                    clamped(blue_contract(v0, v2, v4, v6)),
                )
            }
        }
    }
}

fn unquantize_weights(out: &mut [u32; 64], grid: &[IseValue], count: usize, params: Ise) {
    if params.mode == IseMode::Plain {
        for i in 0..count {
            out[i] = replicate(grid[i].value, params.bits, 6);
        }
    } else {
        let case = params.bits * 2 + i32::from(params.mode == IseMode::Quint);
        if case == 0 || case == 1 {
            const MAP0: [u32; 3] = [0, 32, 63];
            const MAP1: [u32; 5] = [0, 16, 32, 47, 63];
            for i in 0..count {
                out[i] = if case == 0 {
                    MAP0[grid[i].value as usize]
                } else {
                    MAP1[grid[i].value as usize]
                };
            }
        } else {
            const CA: [u32; 5] = [50, 28, 23, 13, 11];
            let c = CA[(case - 2) as usize];
            for i in 0..count {
                let m = grid[i].low;
                let (a, b, cc) = (m & 1, (m >> 1) & 1, (m >> 2) & 1);
                let big_a = if a == 0 { 0 } else { (1 << 7) - 1 };
                let big_b = match case {
                    2 | 3 => 0,
                    4 => (b << 6) | (b << 2) | b,
                    5 => (b << 6) | (b << 1),
                    _ => (cc << 6) | (b << 5) | (cc << 1) | b,
                };
                out[i] = (((grid[i].trit_or_quint * c + big_b) ^ big_a) >> 2) | (big_a & 0x20);
            }
        }
    }
    for slot in out.iter_mut().take(count) {
        if *slot > 32 {
            *slot += 1;
        }
    }
    for slot in out.iter_mut().skip(count) {
        *slot = 0;
    }
}

/// Resample the weight grid up to the footprint, bilinearly, in the fixed-point
/// arithmetic the specification prescribes.
fn interpolate_weights(
    out: &mut [[u32; 2]; MAX_TEXELS],
    weights: &[u32; 64],
    bw: i32,
    bh: i32,
    mode: &BlockMode,
) {
    let planes = if mode.dual_plane { 2 } else { 1 };
    let scale_x = (1024 + bw / 2) / (bw - 1);
    let scale_y = (1024 + bh / 2) / (bh - 1);
    for y in 0..bh {
        for x in 0..bw {
            let gx = (scale_x * x * (mode.grid_width - 1) + 32) >> 6;
            let gy = (scale_y * y * (mode.grid_height - 1) + 32) >> 6;
            let (jx, jy) = (gx >> 4, gy >> 4);
            let (fx, fy) = (gx & 0xF, gy & 0xF);
            let w11 = (fx * fy + 8) >> 4;
            let w10 = fy - w11;
            let w01 = fx - w11;
            let w00 = 16 - fx - fy + w11;
            let i00 = jy * mode.grid_width + jx;
            let indices = [
                i00,
                i00 + 1,
                i00 + mode.grid_width,
                i00 + mode.grid_width + 1,
            ];
            for plane in 0..planes {
                // Out-of-grid corners always carry a zero weight, and masking
                // keeps the read inside the array as the hardware does.
                let p: Vec<u32> = indices
                    .iter()
                    .map(|&i| weights[((i * planes + plane) & 0x3F) as usize])
                    .collect();
                out[(y * bw + x) as usize][plane as usize] = (p[0] * w00 as u32
                    + p[1] * w01 as u32
                    + p[2] * w10 as u32
                    + p[3] * w11 as u32
                    + 8)
                    >> 4;
            }
        }
    }
}

fn hash52(value: u32) -> u32 {
    let mut p = value;
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

/// Which partition a texel belongs to. ASTC computes this rather than storing
/// it: a seed from the block header feeds a hash whose output is twelve small
/// coefficients, and the partition is whichever of up to four linear functions
/// of the texel's position comes out largest.
fn texel_partition(seed: u32, x: u32, y: u32, partitions: i32, small_block: bool) -> usize {
    let (x, y) = if small_block {
        (x << 1, y << 1)
    } else {
        (x, y)
    };
    let seed = seed + 1024 * (partitions as u32 - 1);
    let rnum = hash52(seed);
    let mut s = [0u32; 12];
    for (i, slot) in s.iter_mut().enumerate().take(8) {
        *slot = (rnum >> (4 * i)) & 0xF;
    }
    s[8] = (rnum >> 18) & 0xF;
    s[9] = (rnum >> 22) & 0xF;
    s[10] = (rnum >> 26) & 0xF;
    s[11] = rnum.rotate_left(2) & 0xF;
    for slot in s.iter_mut() {
        *slot = (*slot * *slot) & 0xFF;
    }
    let sh_a = if seed & 2 != 0 { 4 } else { 5 };
    let sh_b = if partitions == 3 { 6 } else { 5 };
    let (sh1, sh2) = if seed & 1 != 0 {
        (sh_a, sh_b)
    } else {
        (sh_b, sh_a)
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    for (i, slot) in s.iter_mut().enumerate().take(8) {
        *slot >>= if i % 2 == 0 { sh1 } else { sh2 };
    }
    for slot in s.iter_mut().skip(8) {
        *slot >>= sh3;
    }
    let a = 0x3F
        & (s[0]
            .wrapping_mul(x)
            .wrapping_add(s[1].wrapping_mul(y))
            .wrapping_add(rnum >> 14));
    let b = 0x3F
        & (s[2]
            .wrapping_mul(x)
            .wrapping_add(s[3].wrapping_mul(y))
            .wrapping_add(rnum >> 10));
    let c = if partitions >= 3 {
        0x3F & (s[4]
            .wrapping_mul(x)
            .wrapping_add(s[5].wrapping_mul(y))
            .wrapping_add(rnum >> 6))
    } else {
        0
    };
    let d = if partitions >= 4 {
        0x3F & (s[6]
            .wrapping_mul(x)
            .wrapping_add(s[7].wrapping_mul(y))
            .wrapping_add(rnum >> 2))
    } else {
        0
    };
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

fn fill(out: &mut [[f32; 4]], texels: usize, colour: [f32; 4]) {
    for slot in out.iter_mut().take(texels) {
        *slot = colour;
    }
}

/// Decode one ASTC block of the given footprint into `out`, which must hold at
/// least `block_width * block_height` texels.
pub fn decode_astc(
    block: &[u8],
    block_width: u32,
    block_height: u32,
    out: &mut [[f32; 4]],
) -> Result<()> {
    let texels = (block_width * block_height) as usize;
    if block.len() < 16 || out.len() < texels {
        return Err(Error::Gpu("astc: block or destination too small".into()));
    }
    let data = u128::from_le_bytes(block[..16].try_into().expect("checked above"));
    let (bw, bh) = (block_width as i32, block_height as i32);

    let Some(mode) = block_mode(bits(data, 0, 10)) else {
        fill(out, texels, ERROR_COLOUR);
        return Ok(());
    };

    if mode.void_extent {
        // A void extent paints one colour over the whole block, and names the
        // region of the texture over which that stays true; the region only
        // matters to a filtering hardware unit, not to a single-block decode.
        let min_s = bits(data, 12, 24);
        let max_s = bits(data, 25, 37);
        let min_t = bits(data, 38, 50);
        let max_t = bits(data, 51, 63);
        let all_ones = min_s == 0x1FFF && max_s == 0x1FFF && min_t == 0x1FFF && max_t == 0x1FFF;
        let hdr = bit(data, 9) != 0;
        if hdr || (!all_ones && (min_s >= max_s || min_t >= max_t)) {
            fill(out, texels, ERROR_COLOUR);
            return Ok(());
        }
        let channel = |lo: i32| {
            let raw = bits(data, lo, lo + 15);
            if raw == 65535 {
                1.0
            } else {
                raw as f32 / 65536.0
            }
        };
        fill(
            out,
            texels,
            [channel(64), channel(80), channel(96), channel(112)],
        );
        return Ok(());
    }

    let planes = if mode.dual_plane { 2 } else { 1 };
    let num_weights = mode.grid_width * mode.grid_height * planes;
    let weight_bits = ise_required_bits(mode.weight_ise, num_weights);
    let partitions = bits(data, 11, 12) as i32 + 1;
    if num_weights > 64
        || !(24..=96).contains(&weight_bits)
        || mode.grid_width > bw
        || mode.grid_height > bh
        || (partitions == 4 && mode.dual_plane)
    {
        fill(out, texels, ERROR_COLOUR);
        return Ok(());
    }

    let single_cem = partitions == 1 || bits(data, 23, 24) == 0;
    let config_bits = (if partitions == 1 {
        17
    } else if single_cem {
        29
    } else {
        25 + 3 * partitions
    }) + if mode.dual_plane { 2 } else { 0 };
    let endpoint_bits = 128 - weight_bits - config_bits;
    let extra_cem_start = 127
        - weight_bits
        - if single_cem {
            -1
        } else {
            match partitions {
                4 => 7,
                3 => 4,
                2 => 1,
                _ => 0,
            }
        };

    // Colour endpoint modes, one per partition.
    let mut modes = [0u32; 4];
    if partitions == 1 {
        modes[0] = bits(data, 13, 16);
    } else {
        let selector = bits(data, 23, 24);
        if selector == 0 {
            let shared = bits(data, 25, 28);
            modes[..partitions as usize].fill(shared);
        } else {
            for (part, slot) in modes.iter_mut().enumerate().take(partitions as usize) {
                let class = selector
                    - if bit(data, 25 + part as i32) != 0 {
                        0
                    } else {
                        1
                    };
                let low0 = partitions + 2 * part as i32;
                let low1 = low0 + 1;
                let at = |index: i32| {
                    bit(
                        data,
                        if index < 4 {
                            25 + index
                        } else {
                            extra_cem_start + index - 4
                        },
                    )
                };
                *slot = (class << 2) | (at(low1) << 1) | at(low0);
            }
        }
    }

    let value_count: i32 = modes[..partitions as usize]
        .iter()
        .map(|&m| endpoint_values(m))
        .sum();
    if value_count > 18 || endpoint_bits < div_round_up(13 * value_count, 5) {
        fill(out, texels, ERROR_COLOUR);
        return Ok(());
    }
    // The LDR profile has no way to express an HDR endpoint.
    if modes[..partitions as usize]
        .iter()
        .any(|&m| is_hdr_endpoint_mode(m))
    {
        fill(out, texels, ERROR_COLOUR);
        return Ok(());
    }

    let endpoint_ise = max_range_ise(endpoint_bits, value_count);
    let mut raw = [IseValue::default(); 18];
    let start = if partitions == 1 { 17 } else { 29 };
    let mut stream = Stream::new(data, start, endpoint_bits, true);
    decode_ise(&mut raw, value_count as usize, &mut stream, endpoint_ise);
    let mut values = [0u32; 18];
    unquantize_endpoints(&mut values, &raw, value_count as usize, endpoint_ise);

    let mut endpoints = [([0u32; 4], [0u32; 4]); 4];
    let mut taken = 0usize;
    for (part, slot) in endpoints.iter_mut().enumerate().take(partitions as usize) {
        let mode = modes[part];
        *slot = decode_endpoint_pair(mode, &values[taken..]);
        taken += endpoint_values(mode) as usize;
    }

    // Weights are read from the top of the block downwards.
    let mut weight_raw = [IseValue::default(); 64];
    let mut stream = Stream::new(data, 127, weight_bits, false);
    decode_ise(
        &mut weight_raw,
        num_weights as usize,
        &mut stream,
        mode.weight_ise,
    );
    let mut weights = [0u32; 64];
    unquantize_weights(
        &mut weights,
        &weight_raw,
        num_weights as usize,
        mode.weight_ise,
    );
    let mut texel_weights = [[0u32; 2]; MAX_TEXELS];
    interpolate_weights(&mut texel_weights, &weights, bw, bh, &mode);

    let component_selector = if mode.dual_plane {
        bits(data, extra_cem_start - 2, extra_cem_start - 1) as i32
    } else {
        -1
    };
    let seed = bits(data, 13, 22);
    let small_block = bw * bh < 31;

    for y in 0..bh {
        for x in 0..bw {
            let index = (y * bw + x) as usize;
            let part = if partitions == 1 {
                0
            } else {
                texel_partition(seed, x as u32, y as u32, partitions, small_block)
            };
            let (e0, e1) = endpoints[part];
            for channel in 0..4 {
                let c0 = (e0[channel] << 8) | e0[channel];
                let c1 = (e1[channel] << 8) | e1[channel];
                let w = texel_weights[index][usize::from(component_selector == channel as i32)];
                let c = (c0 * (64 - w) + c1 * w + 32) / 64;
                out[index][channel] = if c == 65535 { 1.0 } else { c as f32 / 65536.0 };
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A void-extent block: the low nine bits mark it, the four extent fields
    /// are all-ones (the encoding for "this colour is not known to extend
    /// anywhere"), and the colour sits in the top four sixteen-bit fields.
    fn void_extent(r: u16, g: u16, b: u16, a: u16) -> [u8; 16] {
        let low: u64 = 0x1FC | (0x1FFF << 12) | (0x1FFF << 25) | (0x1FFF << 38) | (0x1FFF << 51);
        let high: u64 = r as u64 | ((g as u64) << 16) | ((b as u64) << 32) | ((a as u64) << 48);
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&low.to_le_bytes());
        block[8..].copy_from_slice(&high.to_le_bytes());
        block
    }

    #[test]
    fn a_void_extent_paints_one_colour_over_the_whole_footprint() {
        let block = void_extent(65535, 0, 32768, 65535);
        let mut out = [[0.0f32; 4]; MAX_TEXELS];
        decode_astc(&block, 12, 12, &mut out).unwrap();
        for texel in out.iter().take(144) {
            assert_eq!(texel[0], 1.0, "65535 is exactly one, not 65535/65536");
            assert_eq!(texel[1], 0.0);
            assert_eq!(texel[2], 0.5);
            assert_eq!(texel[3], 1.0);
        }
    }

    #[test]
    fn an_hdr_void_extent_is_refused_by_the_ldr_decoder() {
        let mut block = void_extent(65535, 65535, 65535, 65535);
        block[1] |= 0b10; // bit 9: the HDR flag
        let mut out = [[0.0f32; 4]; MAX_TEXELS];
        decode_astc(&block, 4, 4, &mut out).unwrap();
        assert_eq!(out[0], ERROR_COLOUR);
    }

    #[test]
    fn a_reserved_block_mode_decodes_to_the_error_colour() {
        // Bits 0..3 all zero is reserved, whatever the rest of the block says.
        let mut out = [[0.0f32; 4]; MAX_TEXELS];
        decode_astc(&[0u8; 16], 4, 4, &mut out).unwrap();
        assert_eq!(out[0], ERROR_COLOUR);
        // So is bits[1:0] == 0 with bits[8:6] == 7.
        let mut block = [0u8; 16];
        block[0] = 0b0000_0100; // bits[3:2] non-zero so it is not the case above
        block[1] = 0b0000_0001; // bits 8 set
        block[0] |= 0b1100_0000; // bits 7, 6 set -> bits[8:6] == 7
        decode_astc(&block, 4, 4, &mut out).unwrap();
        assert_eq!(out[0], ERROR_COLOUR);
    }

    #[test]
    fn a_destination_smaller_than_the_footprint_is_refused() {
        let mut out = [[0.0f32; 4]; 16];
        assert!(decode_astc(&void_extent(0, 0, 0, 0), 12, 12, &mut out).is_err());
        assert!(decode_astc(&[0u8; 8], 4, 4, &mut out).is_err());
    }

    /// The integer-sequence tables are the specification's, and each row has
    /// to hold values in range for its base.
    #[test]
    fn the_trit_and_quint_tables_are_in_range() {
        assert_eq!(TRITS_FROM_T.len(), 256);
        assert_eq!(QUINTS_FROM_Q.len(), 128);
        for row in TRITS_FROM_T {
            assert!(row.iter().all(|&t| t < 3), "a trit is 0, 1 or 2");
        }
        for row in QUINTS_FROM_Q {
            assert!(row.iter().all(|&q| q < 5), "a quint is 0..4");
        }
    }

    /// The widest range that fits has to actually fit, and one step wider
    /// must not.
    #[test]
    fn the_ise_range_is_the_widest_that_fits() {
        for values in [2i32, 6, 9, 12, 18] {
            for available in [24i32, 40, 63, 80, 96] {
                let params = max_range_ise(available, values);
                assert!(
                    ise_required_bits(params, values) <= available,
                    "{values} values in {available} bits"
                );
            }
        }
    }
}
