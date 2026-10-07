//! NEON/AdvSIMD instruction decode and execution.

use super::bits::*;
use super::crypto::poly_mul;
use super::fp;
use super::Cpu;
use crate::{Error, Result};

mod lanes;
mod misc;
mod rest;

impl Cpu {
    pub(super) fn try_simd(&mut self, insn: u32) -> Result<bool> {
        // AES and SHA: decoded before scalar DUP, which shares bits[28:21].
        if self.try_crypto(insn)? {
            return Ok(true);
        }

        // Narrowing shifts (SHRN/RSHRN/SQSHRN/...), checked before MOVI.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01111
            && ((insn >> 23) & 1) == 0
            && ((insn >> 10) & 1) == 1
        {
            let op = (insn >> 11) & 0x1F;
            if matches!(op, 0b10000..=0b10011) {
                let q = (insn >> 30) & 1 == 1;
                let u = (insn >> 29) & 1;
                let rd = (insn & 0x1F) as u8;
                let rn = ((insn >> 5) & 0x1F) as u8;
                // Destination element size comes from all of `immh`, not bit22.
                let immh = (insn >> 19) & 0xF;
                // `immh == 0` is MOVI.
                let dest_esize = match immh {
                    0b0001 => Some(8u32),
                    0b0010 | 0b0011 => Some(16),
                    0b0100..=0b0111 => Some(32),
                    _ => None,
                };
                let dest_esize = dest_esize.unwrap_or(0);
                let shift_field = (insn >> 16) & 0x7F;
                let shift = (2 * dest_esize).saturating_sub(shift_field);
                if shift > 0 && shift <= dest_esize {
                    let rounding = op & 1 == 1; // RSHRN/SQRSHRN/UQRSHRN round
                    let (signed_src, to_unsigned) = match (u, op) {
                        (0, 0b10000) => (false, false), // SHRN
                        (0, 0b10001) => (false, false), // RSHRN
                        (1, 0b10000) => (true, true),   // SQSHRUN
                        (1, 0b10001) => (true, true),   // SQRSHRUN
                        (0, 0b10010) => (true, false),  // SQSHRN
                        (0, 0b10011) => (true, false),  // SQRSHRN
                        (1, 0b10010) => (false, false), // UQSHRN
                        (1, 0b10011) => (false, false), // UQRSHRN
                        _ => unreachable!(),
                    };
                    let saturating = op & 0b10 != 0 || to_unsigned;
                    self.simd_shrn(
                        rd,
                        rn,
                        q,
                        dest_esize,
                        shift,
                        rounding,
                        signed_src,
                        to_unsigned,
                        saturating,
                    );
                    return Ok(true);
                }
            }
        }

        // Shift by immediate (SSHR/USHR/SHL/SLI/SRI/SSHLL/...), vector and scalar.
        let scalar_shift = ((insn >> 30) & 0b11) == 0b01 && ((insn >> 23) & 0x3F) == 0b111110;
        let vector_shift = ((insn >> 31) & 1) == 0 && ((insn >> 23) & 0x3F) == 0b011110;
        if (vector_shift || scalar_shift) && ((insn >> 10) & 1) == 1 && ((insn >> 19) & 0xF) != 0 {
            let opcode = (insn >> 11) & 0x1F;
            if !matches!(opcode, 0b10000..=0b10011) {
                let q = vector_shift && (insn >> 30) & 1 == 1;
                let u = (insn >> 29) & 1 == 1;
                let immh = (insn >> 19) & 0xF;
                let imm = (insn >> 16) & 0x7F;
                let rd = (insn & 0x1F) as u8;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let esize = match immh {
                    0b0001 => 8,
                    0b0010 | 0b0011 => 16,
                    0b0100..=0b0111 => 32,
                    _ => 64,
                };
                // SCVTF/UCVTF and FCVTZS/FCVTZU with fixed-point fraction bits.
                if matches!(opcode, 0b11100 | 0b11111) {
                    if vector_shift && !q && esize == 64 {
                        return Err(Error::Cpu(format!(
                            "unallocated SIMD fixed-point convert {insn:#010x}: one 64-bit lane in a 64-bit vector"
                        )));
                    }
                    let lanes = match (scalar_shift, q) {
                        (true, _) => 1,
                        (false, true) => 128 / esize,
                        (false, false) => 64 / esize,
                    };
                    return self
                        .simd_fixed_convert(
                            rd,
                            rn,
                            lanes,
                            esize,
                            2 * esize - imm,
                            opcode == 0b11111,
                            u,
                        )
                        .map(|()| true);
                }
                return self
                    .simd_shift_imm(rd, rn, q, u, opcode, esize, imm)
                    .map(|()| true);
            }
        }

        // MOVI/MVNI: imm8 is split across bits 18:16 and 9:5, cmode in 15:12, op in bit29.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 23) & 0x3F) == 0b011110
            && ((insn >> 19) & 0b1111) == 0b0000
        {
            let q = (insn >> 30) & 1;
            let op = (insn >> 29) & 1;
            let rd = (insn & 0x1F) as u8;
            let imm8 = (((insn >> 16) & 0b111) << 5) | ((insn >> 5) & 0x1F);
            let cmode = (insn >> 12) & 0b1111;
            let imm64 = simd_imm_const(imm8, cmode, op);
            self.vregs[rd as usize] = if q == 1 {
                imm64 as u128 | ((imm64 as u128) << 64)
            } else {
                imm64 as u128
            };
            return Ok(true);
        }

        // Permute (ZIP/UZP/TRN); bit29 set would be EXT.
        let perm = (insn >> 10) & 0b111111;
        if ((insn >> 31) & 1) == 0
            && ((insn >> 29) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01110
            && ((insn >> 21) & 1) == 0
            && matches!(
                perm,
                0b000110 | 0b010110 | 0b001010 | 0b011010 | 0b001110 | 0b011110
            )
        {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let esize = 8u32 << ((insn >> 22) & 0b11);
            self.simd_permute(rd, rn, rm, q, esize, perm);
            return Ok(true);
        }

        // Three different (widening/narrowing): bits[11:10] = 00.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0b11) == 0b00
        {
            let q = (insn >> 30) & 1 == 1;
            let u = (insn >> 29) & 1;
            let size = (insn >> 22) & 0b11;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let opcode = (insn >> 12) & 0xF;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            // PMULL has a 64-bit source element, so it precedes the size check.
            if opcode == 0b1110 && u == 0 {
                let half = if q { 64 } else { 0 };
                let a = (self.vregs[rn as usize] >> half) as u64;
                let b = (self.vregs[rm as usize] >> half) as u64;
                self.vregs[rd as usize] = match size {
                    0b00 => (0..8u32).fold(0u128, |acc, i| {
                        let lane = poly_mul((a >> (8 * i)) & 0xFF, (b >> (8 * i)) & 0xFF, 8);
                        acc | (lane << (16 * i))
                    }),
                    0b11 => poly_mul(a, b, 64),
                    _ => return Ok(false), // 16- and 32-bit sources are undefined
                };
                return Ok(true);
            }
            if size == 0b11 {
                return Ok(false); // no 128-bit destination elements
            }
            let esize = 8u32 << size;
            let wide = esize * 2;
            let elements = 128 / wide;
            let half = if q { elements * esize } else { 0 };
            let signed = u == 0;
            let narrow = |v: u128, base: u32, i: u32| {
                let raw = ((v >> (base + esize * i)) & elem_mask(esize)) as u64;
                if signed {
                    sext_u64(raw, esize) as i64 as i128
                } else {
                    i128::from(raw)
                }
            };
            let a_reg = self.vregs[rn as usize];
            let b_reg = self.vregs[rm as usize];
            let d_reg = self.vregs[rd as usize];
            let mut out: u128 = 0;
            match opcode {
                // ADDHN/SUBHN and rounding variants: wide operands, narrow result.
                0b0100 | 0b0110 => {
                    let subtract = opcode == 0b0110;
                    let rounding = u == 1;
                    let mut packed: u128 = 0;
                    for i in 0..elements {
                        let a = ((a_reg >> (wide * i)) & elem_mask(wide)) as u64;
                        let b = ((b_reg >> (wide * i)) & elem_mask(wide)) as u64;
                        let mut value = if subtract {
                            a.wrapping_sub(b)
                        } else {
                            a.wrapping_add(b)
                        };
                        if rounding {
                            value = value.wrapping_add(1 << (esize - 1));
                        }
                        let narrowed = (value >> esize) & (elem_mask(esize) as u64);
                        packed |= u128::from(narrowed) << (esize * i);
                    }
                    self.vregs[rd as usize] = if q {
                        (d_reg & elem_mask(64)) | (packed << 64)
                    } else {
                        packed
                    };
                    return Ok(true);
                }
                _ => {}
            }
            for i in 0..elements {
                let b = narrow(b_reg, half, i);
                let a = if matches!(opcode, 0b0001 | 0b0011) {
                    let raw = ((a_reg >> (wide * i)) & elem_mask(wide)) as u64;
                    if signed {
                        sext_u64(raw, wide) as i64 as i128
                    } else {
                        i128::from(raw)
                    }
                } else {
                    narrow(a_reg, half, i)
                };
                let acc = {
                    let raw = ((d_reg >> (wide * i)) & elem_mask(wide)) as u64;
                    if signed {
                        sext_u64(raw, wide) as i64 as i128
                    } else {
                        i128::from(raw)
                    }
                };
                let value = match opcode {
                    0b0000 | 0b0001 => a + b,      // SADDL/W, UADDL/W
                    0b0010 | 0b0011 => a - b,      // SSUBL/W, USUBL/W
                    0b0101 => acc + (a - b).abs(), // SABAL, UABAL
                    0b0111 => (a - b).abs(),       // SABDL, UABDL
                    0b1000 => acc + a * b,         // SMLAL, UMLAL
                    0b1010 => acc - a * b,         // SMLSL, UMLSL
                    0b1100 => a * b,               // SMULL, UMULL
                    _ => {
                        return Err(Error::Cpu(format!(
                            "unimplemented SIMD three-different u={} opcode={:#06b} at {:#x}",
                            u, opcode, self.pc
                        )))
                    }
                };
                out |= ((value as u128) & elem_mask(wide)) << (wide * i);
            }
            self.vregs[rd as usize] = out;
            return Ok(true);
        }

        // By-element multiplies, scalar: result in the bottom lane, rest zeroed.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11111
            && ((insn >> 10) & 1) == 0
        {
            let u = (insn >> 29) & 1;
            let size = (insn >> 22) & 0b11;
            let l = (insn >> 21) & 1;
            let m = (insn >> 20) & 1;
            let rm_low = (insn >> 16) & 0xF;
            let opcode = (insn >> 12) & 0xF;
            let h = (insn >> 11) & 1;
            let rn = ((insn >> 5) & 0x1F) as usize;
            let rd = (insn & 0x1F) as usize;
            // Index is H:L:M; halfword elements only use v0-v15.
            let (esize, index, rm) = match size {
                0b01 => (16u32, (h << 2) | (l << 1) | m, rm_low as usize),
                0b10 => (32, (h << 1) | l, ((m << 4) | rm_low) as usize),
                0b11 => (64, h, ((m << 4) | rm_low) as usize),
                _ => return Ok(false),
            };
            let key = 16 * u + opcode;
            let elem = ((self.vregs[rm] >> (index * esize)) & elem_mask(esize)) as u64;
            let a = (self.vregs[rn] & elem_mask(esize)) as u64;
            let signed = |v: u64, bits: u32| -> i128 { sext_u64(v, bits) as i64 as i128 };
            let lane =
                |reg: usize, bits: u32| -> u64 { (self.vregs[reg] & elem_mask(bits)) as u64 };

            let sat = |value: i128, bits: u32| -> u64 {
                let max = (1i128 << (bits - 1)) - 1;
                let min = -(1i128 << (bits - 1));
                (value.clamp(min, max) as u64) & (elem_mask(bits) as u64)
            };
            let doubled = signed(a, esize) * signed(elem, esize) * 2;
            let high_half = |rounding: bool| -> i128 {
                let product = if rounding {
                    doubled + (1i128 << (esize - 1))
                } else {
                    doubled
                };
                product >> esize
            };

            let (value, width) = match key {
                // FMLA / FMLS / FMUL / FMULX (no half precision).
                0x01 | 0x05 | 0x09 | 0x19 => {
                    if esize == 16 {
                        return Ok(false);
                    }
                    let d = lane(rd, esize);
                    let bits = if esize == 64 {
                        let (mut x, y, acc) =
                            (f64::from_bits(a), f64::from_bits(elem), f64::from_bits(d));
                        if key == 0x05 {
                            x = -x;
                        }
                        let r = match key {
                            0x01 | 0x05 => x.mul_add(y, acc),
                            0x19 => fmulx(x, y),
                            _ => x * y,
                        };
                        r.to_bits()
                    } else {
                        let (mut x, y, acc) = (
                            f32::from_bits(a as u32),
                            f32::from_bits(elem as u32),
                            f32::from_bits(d as u32),
                        );
                        if key == 0x05 {
                            x = -x;
                        }
                        // Fused at source width to round once.
                        let r = match key {
                            0x01 | 0x05 => x.mul_add(y, acc),
                            0x19 => fmulx(f64::from(x), f64::from(y)) as f32,
                            _ => x * y,
                        };
                        u64::from(r.to_bits())
                    };
                    (bits, esize)
                }
                // SQDMULL
                0x0b => (sat(doubled, esize * 2), esize * 2),
                // SQDMLAL / SQDMLSL: product and sum both saturate.
                0x03 | 0x07 => {
                    let wide = esize * 2;
                    let acc = signed(lane(rd, wide), wide);
                    let product = signed(sat(doubled, wide), wide);
                    let sum = if key == 0x03 {
                        acc + product
                    } else {
                        acc - product
                    };
                    (sat(sum, wide), wide)
                }
                // SQDMULH / SQRDMULH
                0x0c | 0x0d => (sat(high_half(key == 0x0d), esize), esize),
                // SQRDMLAH / SQRDMLSH
                0x1d | 0x1f => {
                    let acc = signed(lane(rd, esize), esize);
                    let product = high_half(true);
                    let sum = if key == 0x1d {
                        acc + product
                    } else {
                        acc - product
                    };
                    (sat(sum, esize), esize)
                }
                _ => {
                    return Err(Error::Cpu(format!(
                        "unimplemented SIMD scalar by-element u={} opcode={:#06b} at {:#x}",
                        u, opcode, self.pc
                    )))
                }
            };
            self.vregs[rd] = u128::from(value) & elem_mask(width);
            return Ok(true);
        }

        // By-element multiplies, vector.
        if ((insn >> 31) & 1) == 0 && ((insn >> 24) & 0x1F) == 0b01111 && ((insn >> 10) & 1) == 0 {
            let q = (insn >> 30) & 1 == 1;
            let u = (insn >> 29) & 1;
            let size = (insn >> 22) & 0b11;
            let l = (insn >> 21) & 1;
            let m = (insn >> 20) & 1;
            let rm_low = (insn >> 16) & 0xF;
            let opcode = (insn >> 12) & 0xF;
            let h = (insn >> 11) & 1;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            // Index is H:L:M; halfword elements only use v0-v15.
            let (esize, index, rm) = match size {
                0b01 => (16u32, (h << 2) | (l << 1) | m, rm_low as u8),
                0b10 => (32, (h << 1) | l, ((m << 4) | rm_low) as u8),
                0b11 => (64, h, ((m << 4) | rm_low) as u8),
                _ => return Ok(false),
            };
            let key = 16 * u + opcode;
            let elem = ((self.vregs[rm as usize] >> (index * esize)) & elem_mask(esize)) as u64;
            let widening = matches!(key, 0x02 | 0x06 | 0x0a | 0x12 | 0x16 | 0x1a);
            if widening {
                // SMULL/UMULL and accumulating forms: `Q` selects the Vn half, lanes widen.
                let signed = u == 0;
                let dest_esize = esize * 2;
                let elements = 128 / dest_esize;
                let base = if q { elements * esize } else { 0 };
                let src = self.vregs[rn as usize];
                let dest = self.vregs[rd as usize];
                let b = if signed {
                    sext_u64(elem, esize) as i64 as i128
                } else {
                    elem as i128
                };
                let mut out: u128 = 0;
                for i in 0..elements {
                    let raw = ((src >> (base + esize * i)) & elem_mask(esize)) as u64;
                    let a = if signed {
                        sext_u64(raw, esize) as i64 as i128
                    } else {
                        raw as i128
                    };
                    let product = a * b;
                    let acc = ((dest >> (dest_esize * i)) & elem_mask(dest_esize)) as u64;
                    let value = match key & 0xF {
                        0x02 => (acc as i128).wrapping_add(product), // SMLAL/UMLAL
                        0x06 => (acc as i128).wrapping_sub(product), // SMLSL/UMLSL
                        _ => product,                                // SMULL/UMULL
                    };
                    out |= ((value as u128) & elem_mask(dest_esize)) << (dest_esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            match key {
                // MUL / MLA / MLS
                0x08 | 0x10 | 0x14 => {
                    let mode = key;
                    self.simd_elem_acc(rd, rn, rd, q, esize, move |a, _, d| match mode {
                        0x08 => a.wrapping_mul(elem),
                        0x10 => d.wrapping_add(a.wrapping_mul(elem)),
                        _ => d.wrapping_sub(a.wrapping_mul(elem)),
                    });
                    Ok(true)
                }
                // SQDMULH / SQRDMULH: doubled high half, saturated.
                0x0c | 0x0d => {
                    let rounding = key == 0x0d;
                    let b = sext_u64(elem, esize) as i64 as i128;
                    self.simd_elem(rd, rn, rn, q, esize, move |a, _| {
                        let a = sext_u64(a, esize) as i64 as i128;
                        let mut product = 2 * a * b;
                        if rounding {
                            product += 1i128 << (esize - 1);
                        }
                        let shifted = product >> esize;
                        let max = (1i128 << (esize - 1)) - 1;
                        let min = -(1i128 << (esize - 1));
                        shifted.clamp(min, max) as u64
                    });
                    Ok(true)
                }
                // FMUL / FMULX / FMLA / FMLS
                0x09 | 0x19 | 0x01 | 0x05 => {
                    if esize == 16 {
                        return Ok(false); // half precision: out of scope
                    }
                    let subtract = key == 0x05;
                    let accumulate = key == 0x01 || key == 0x05;
                    let double = esize == 64;
                    self.simd_elem_acc(rd, rn, rd, q, esize, move |a, _, d| {
                        if double {
                            let (x, y, acc) =
                                (f64::from_bits(a), f64::from_bits(elem), f64::from_bits(d));
                            let x = if subtract { -x } else { x };
                            let r = if accumulate { x.mul_add(y, acc) } else { x * y };
                            r.to_bits()
                        } else {
                            let x = f32::from_bits(a as u32);
                            let y = f32::from_bits(elem as u32);
                            let acc = f32::from_bits(d as u32);
                            let x = if subtract { -x } else { x };
                            let r = if accumulate { x.mul_add(y, acc) } else { x * y };
                            u64::from(r.to_bits())
                        }
                    });
                    Ok(true)
                }
                _ => Err(Error::Cpu(format!(
                    "unimplemented SIMD by-element u={} opcode={:#06b} at {:#x}",
                    u, opcode, self.pc
                ))),
            }
        } else {
            self.try_simd_rest(insn)
        }
    }
}
