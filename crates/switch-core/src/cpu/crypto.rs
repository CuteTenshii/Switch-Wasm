//! The ARMv8 cryptographic extension: AES, SHA-1, SHA-256 and the polynomial
//! multiplies. The Tegra X1's A57 implements all of it, so guest code is free
//! to use it.
//!
//! The AES steps are the ones [`crate::crypto`] already decrypts NCAs with —
//! same column-major state, so the instructions are just a different way in.

use super::Cpu;
use crate::crypto::{
    inv_mix_columns, inv_shift_rows, inv_sub_bytes, mix_columns, shift_rows, sub_bytes,
};
use crate::Result;

#[inline]
fn elem32(v: u128, i: u32) -> u32 {
    (v >> (32 * i)) as u32
}

#[inline]
fn pack32(e0: u32, e1: u32, e2: u32, e3: u32) -> u128 {
    u128::from(e0) | (u128::from(e1) << 32) | (u128::from(e2) << 64) | (u128::from(e3) << 96)
}

fn sha_choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}

fn sha_parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

fn sha_majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

fn sigma0_upper(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

fn sigma1_upper(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

fn sigma0_lower(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

fn sigma1_lower(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// Four SHA-1 rounds. `f` is the round function the opcode picks, and the
/// 160-bit `Y:X` rotate at the end of each round is what walks the state.
fn sha1_rounds(mut x: u128, mut y: u32, w: u128, f: fn(u32, u32, u32) -> u32) -> u128 {
    for e in 0..4 {
        let (x0, x1, x2, x3) = (elem32(x, 0), elem32(x, 1), elem32(x, 2), elem32(x, 3));
        let t = f(x1, x2, x3);
        y = y
            .wrapping_add(x0.rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(elem32(w, e));
        x = pack32(y, x0, x1.rotate_left(30), x2);
        y = x3;
    }
    x
}

/// Four SHA-256 rounds over the two halves of the state. `part1` selects which
/// half the instruction keeps.
fn sha256_rounds(mut x: u128, mut y: u128, w: u128, part1: bool) -> u128 {
    for e in 0..4 {
        let (y0, y1, y2, y3) = (elem32(y, 0), elem32(y, 1), elem32(y, 2), elem32(y, 3));
        let (x0, x1, x2, x3) = (elem32(x, 0), elem32(x, 1), elem32(x, 2), elem32(x, 3));
        let t = y3
            .wrapping_add(sigma1_upper(y0))
            .wrapping_add(sha_choose(y0, y1, y2))
            .wrapping_add(elem32(w, e));
        let x3n = t.wrapping_add(x3);
        let y3n = t
            .wrapping_add(sigma0_upper(x0))
            .wrapping_add(sha_majority(x0, x1, x2));
        x = pack32(y3n, x0, x1, x2);
        y = pack32(x3n, y0, y1, y2);
    }
    if part1 {
        x
    } else {
        y
    }
}

/// Carry-less multiply of two `bits`-wide operands, for PMULL/PMULL2.
pub(super) fn poly_mul(a: u64, b: u64, bits: u32) -> u128 {
    let mut out: u128 = 0;
    for i in 0..bits {
        if (b >> i) & 1 == 1 {
            out ^= u128::from(a) << i;
        }
    }
    out
}

impl Cpu {
    /// AES and SHA. These sit in the Advanced SIMD encoding space but share
    /// bits with the copy group, so [`Cpu::try_simd`] has to offer them here
    /// first.
    pub(super) fn try_crypto(&mut self, insn: u32) -> Result<bool> {
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rm = ((insn >> 16) & 0x1F) as u8;

        // AES: `0100 1110 00 10100 opcode(5) 10 Rn Rd`.
        if (insn >> 24) & 0xFF == 0x4E
            && (insn >> 22) & 0b11 == 0b00
            && (insn >> 17) & 0x1F == 0b10100
            && (insn >> 10) & 0b11 == 0b10
        {
            let opcode = (insn >> 12) & 0x1F;
            let (n, d) = (self.vregs[rn as usize], self.vregs[rd as usize]);
            // AESE/AESD fold in Vd EOR Vn; the mix-columns step is a separate
            // instruction so a full round is built from the pair.
            let mut state = (if opcode < 0b00110 { d ^ n } else { n }).to_le_bytes();
            match opcode {
                0b00100 => {
                    shift_rows(&mut state);
                    sub_bytes(&mut state);
                }
                0b00101 => {
                    inv_shift_rows(&mut state);
                    inv_sub_bytes(&mut state);
                }
                0b00110 => mix_columns(&mut state),
                0b00111 => inv_mix_columns(&mut state),
                _ => return Ok(false),
            }
            self.vregs[rd as usize] = u128::from_le_bytes(state);
            return Ok(true);
        }

        // Three-register SHA: `0101 1110 000 Rm 0 opcode(3) 00 Rn Rd`.
        if (insn >> 24) & 0xFF == 0x5E
            && (insn >> 21) & 0b111 == 0b000
            && (insn >> 15) & 1 == 0
            && (insn >> 10) & 0b11 == 0b00
        {
            let opcode = (insn >> 12) & 0b111;
            let (d, n, m) = (
                self.vregs[rd as usize],
                self.vregs[rn as usize],
                self.vregs[rm as usize],
            );
            let result = match opcode {
                0b000 => sha1_rounds(d, n as u32, m, sha_choose),
                0b001 => sha1_rounds(d, n as u32, m, sha_parity),
                0b010 => sha1_rounds(d, n as u32, m, sha_majority),
                // SHA1SU0: the schedule's three-way XOR, over a window that
                // straddles two of the message vectors.
                0b011 => {
                    let shifted = (d >> 64) | (n << 64);
                    shifted ^ d ^ m
                }
                0b100 => sha256_rounds(d, n, m, true),
                0b101 => sha256_rounds(n, d, m, false),
                // SHA256SU1
                0b110 => {
                    let t0 = (n >> 32) | (m << 96);
                    let mut out: u128 = 0;
                    for e in 0..4u32 {
                        let src = if e < 2 {
                            elem32(m, e + 2)
                        } else {
                            elem32(out, e - 2)
                        };
                        let v = sigma1_lower(src)
                            .wrapping_add(elem32(d, e))
                            .wrapping_add(elem32(t0, e));
                        out |= u128::from(v) << (32 * e);
                    }
                    out
                }
                _ => return Ok(false),
            };
            self.vregs[rd as usize] = result;
            return Ok(true);
        }

        // Two-register SHA: `0101 1110 00 10100 opcode(5) 10 Rn Rd`.
        if (insn >> 24) & 0xFF == 0x5E
            && (insn >> 22) & 0b11 == 0b00
            && (insn >> 17) & 0x1F == 0b10100
            && (insn >> 10) & 0b11 == 0b10
        {
            let opcode = (insn >> 12) & 0x1F;
            let (d, n) = (self.vregs[rd as usize], self.vregs[rn as usize]);
            let result = match opcode {
                // SHA1H writes a scalar, so the rest of the register clears.
                0b00000 => u128::from((n as u32).rotate_left(30)),
                // SHA1SU1
                0b00001 => {
                    let t = d ^ (n >> 32);
                    let r = [0, 1, 2, 3].map(|e| elem32(t, e).rotate_left(1));
                    pack32(r[0], r[1], r[2], r[3] ^ elem32(t, 0).rotate_left(2))
                }
                // SHA256SU0
                0b00010 => {
                    let t = (d >> 32) | (n << 96);
                    let mut out: u128 = 0;
                    for e in 0..4u32 {
                        let v = sigma0_lower(elem32(t, e)).wrapping_add(elem32(d, e));
                        out |= u128::from(v) << (32 * e);
                    }
                    out
                }
                _ => return Ok(false),
            };
            self.vregs[rd as usize] = result;
            return Ok(true);
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a whole SHA-256 block through the instruction helpers exactly as
    /// the ARM-optimised implementations sequence them, and check the digest.
    #[test]
    fn the_sha256_helpers_hash_abc() {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        // "abc" padded into one 512-bit block, as four big-endian word vectors.
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[62..].copy_from_slice(&24u16.to_be_bytes());
        let word = |i: usize| u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        let mut w = [0u128; 4];
        for (v, chunk) in w.iter_mut().zip(0..4) {
            *v = pack32(
                word(chunk * 4),
                word(chunk * 4 + 1),
                word(chunk * 4 + 2),
                word(chunk * 4 + 3),
            );
        }

        let mut abcd = pack32(0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a);
        let mut efgh = pack32(0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19);
        let (abcd0, efgh0) = (abcd, efgh);

        for round in 0..16usize {
            let i = round % 4;
            if round >= 4 {
                // The schedule runs one group ahead of the rounds that use it.
                w[i] = {
                    let su0 = {
                        let d = w[i];
                        let n = w[(i + 1) % 4];
                        let t = (d >> 32) | (n << 96);
                        let mut out: u128 = 0;
                        for e in 0..4u32 {
                            let v = sigma0_lower(elem32(t, e)).wrapping_add(elem32(d, e));
                            out |= u128::from(v) << (32 * e);
                        }
                        out
                    };
                    let (n, m) = (w[(i + 2) % 4], w[(i + 3) % 4]);
                    let t0 = (n >> 32) | (m << 96);
                    let mut out: u128 = 0;
                    for e in 0..4u32 {
                        let src = if e < 2 { elem32(m, e + 2) } else { elem32(out, e - 2) };
                        let v = sigma1_lower(src)
                            .wrapping_add(elem32(su0, e))
                            .wrapping_add(elem32(t0, e));
                        out |= u128::from(v) << (32 * e);
                    }
                    out
                };
            }
            let wk = pack32(
                elem32(w[i], 0).wrapping_add(K[round * 4]),
                elem32(w[i], 1).wrapping_add(K[round * 4 + 1]),
                elem32(w[i], 2).wrapping_add(K[round * 4 + 2]),
                elem32(w[i], 3).wrapping_add(K[round * 4 + 3]),
            );
            let saved = abcd;
            abcd = sha256_rounds(abcd, efgh, wk, true);
            efgh = sha256_rounds(saved, efgh, wk, false);
        }

        let mut digest = [0u8; 32];
        for e in 0..4u32 {
            let a = elem32(abcd, e).wrapping_add(elem32(abcd0, e));
            let b = elem32(efgh, e).wrapping_add(elem32(efgh0, e));
            digest[e as usize * 4..e as usize * 4 + 4].copy_from_slice(&a.to_be_bytes());
            digest[16 + e as usize * 4..16 + e as usize * 4 + 4].copy_from_slice(&b.to_be_bytes());
        }
        let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// The same for SHA-1, whose round function changes every twenty rounds.
    #[test]
    fn the_sha1_helpers_hash_abc() {
        const K: [u32; 4] = [0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6];
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[62..].copy_from_slice(&24u16.to_be_bytes());
        let word = |i: usize| u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        let mut w = [0u128; 4];
        for (v, chunk) in w.iter_mut().zip(0..4) {
            *v = pack32(
                word(chunk * 4),
                word(chunk * 4 + 1),
                word(chunk * 4 + 2),
                word(chunk * 4 + 3),
            );
        }

        let mut abcd = pack32(0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476);
        let mut e_state: u32 = 0xc3d2e1f0;
        let (abcd0, e0) = (abcd, e_state);

        for round in 0..20usize {
            let i = round % 4;
            if round >= 4 {
                // SHA1SU0 then SHA1SU1 extend the schedule by four words.
                let su0 = {
                    let (d, n, m) = (w[i], w[(i + 1) % 4], w[(i + 2) % 4]);
                    ((d >> 64) | (n << 64)) ^ d ^ m
                };
                let t = su0 ^ (w[(i + 3) % 4] >> 32);
                let r = [0, 1, 2, 3].map(|e| elem32(t, e).rotate_left(1));
                w[i] = pack32(r[0], r[1], r[2], r[3] ^ elem32(t, 0).rotate_left(2));
            }
            let k = K[round / 5];
            let wk = pack32(
                elem32(w[i], 0).wrapping_add(k),
                elem32(w[i], 1).wrapping_add(k),
                elem32(w[i], 2).wrapping_add(k),
                elem32(w[i], 3).wrapping_add(k),
            );
            let next_e = (elem32(abcd, 0)).rotate_left(30);
            let f = match round / 5 {
                0 => sha_choose as fn(u32, u32, u32) -> u32,
                2 => sha_majority,
                _ => sha_parity,
            };
            abcd = sha1_rounds(abcd, e_state, wk, f);
            e_state = next_e;
        }

        let mut digest = [0u8; 20];
        for e in 0..4u32 {
            let v = elem32(abcd, e).wrapping_add(elem32(abcd0, e));
            digest[e as usize * 4..e as usize * 4 + 4].copy_from_slice(&v.to_be_bytes());
        }
        digest[16..].copy_from_slice(&e_state.wrapping_add(e0).to_be_bytes());
        let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn the_polynomial_multiply_carries_nothing() {
        assert_eq!(poly_mul(0b11, 0b11, 8), 0b101);
        assert_eq!(poly_mul(0xFF, 0x01, 8), 0xFF);
        assert_eq!(poly_mul(1 << 63, 1 << 63, 64), 1u128 << 126);
    }
}
