//! NEON/AdvSIMD instruction decode and execution.

use super::bits::*;
use super::crypto::poly_mul;
use super::Cpu;
use crate::{Error, Result};

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

    fn try_simd_rest(&mut self, insn: u32) -> Result<bool> {
        // EXT: bytes from Vm:Vn starting at imm4.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x3F) == 0b101110
            && ((insn >> 22) & 0b11) == 0
            && ((insn >> 21) & 1) == 0
            && ((insn >> 15) & 1) == 0
            && ((insn >> 10) & 1) == 0
        {
            let q = (insn >> 30) & 1 == 1;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let imm4 = (insn >> 11) & 0xF;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            if !q && imm4 & 0b1000 != 0 {
                return Ok(false); // an 8-byte result can't start 8+ bytes in
            }
            let shift = imm4 * 8;
            let n = self.vregs[rn as usize];
            let m = self.vregs[rm as usize];
            self.vregs[rd as usize] = if q {
                if shift == 0 {
                    n
                } else {
                    (n >> shift) | (m << (128 - shift))
                }
            } else {
                let low = elem_mask(64);
                (((n & low) | ((m & low) << 64)) >> shift) & low
            };
            return Ok(true);
        }

        // Two-register misc (vector).
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01110
            && ((insn >> 17) & 0x1F) == 0b10000
            && ((insn >> 10) & 0b11) == 0b10
        {
            return self.simd_two_reg_misc(insn, false);
        }
        // Scalar FP three-same.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && ((insn >> 11) & 0x1F) >= 0b11000
        {
            return self.simd_fp_three_same(insn, true);
        }
        // Scalar integer three-same: variable shifts.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && (0b01000..=0b01011).contains(&((insn >> 11) & 0x1F))
        {
            let op = (insn >> 11) & 0x1F;
            let size = (insn >> 22) & 0b11;
            let saturating = op & 1 == 1;
            if !saturating && size != 0b11 {
                return Ok(false); // SSHL/SRSHL have no narrow scalar form
            }
            let esize = 8u32 << size;
            let unsigned = (insn >> 29) & 1 == 1;
            let rounding = op & 0b10 != 0;
            let a = self.vregs[((insn >> 5) & 0x1F) as usize] as u64;
            let b = self.vregs[((insn >> 16) & 0x1F) as usize] as u64;
            let v = shift_by_reg(a, b, esize, unsigned, rounding, saturating);
            self.vregs[(insn & 0x1F) as usize] = u128::from(v);
            return Ok(true);
        }
        // Scalar pairwise: ADDP, FADDP, FMAXP, FMINP, FMAXNMP, FMINNMP.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 17) & 0x1F) == 0b11000
            && ((insn >> 10) & 0b11) == 0b10
        {
            let unsigned = (insn >> 29) & 1 == 1;
            let opcode = (insn >> 12) & 0x1F;
            let a = self.vregs[((insn >> 5) & 0x1F) as usize];
            let rd = (insn & 0x1F) as usize;
            if !unsigned {
                if opcode != 0b11011 || (insn >> 22) & 0b11 != 0b11 {
                    return Ok(false); // ADDP only, and only on doublewords
                }
                self.vregs[rd] = u128::from((a as u64).wrapping_add((a >> 64) as u64));
                return Ok(true);
            }
            let double = (insn >> 22) & 1 == 1;
            let min = (insn >> 23) & 1 == 1;
            let op = |x: f64, y: f64| -> Option<f64> {
                Some(match (opcode, min) {
                    (0b01100, false) => fp_maxnum(x, y),
                    (0b01100, true) => fp_minnum(x, y),
                    (0b01101, false) => x + y,
                    (0b01111, false) => fp_max(x, y),
                    (0b01111, true) => fp_min(x, y),
                    _ => return None,
                })
            };
            self.vregs[rd] = if double {
                let lane = |i: u32| f64::from_bits((a >> (64 * i)) as u64);
                let Some(r) = op(lane(0), lane(1)) else {
                    return Ok(false);
                };
                u128::from(r.to_bits())
            } else {
                let lane = |i: u32| f32::from_bits((a >> (32 * i)) as u32);
                // Computed via widening to match the vector pairwise forms.
                let Some(r) = op(f64::from(lane(0)), f64::from(lane(1))) else {
                    return Ok(false);
                };
                u128::from((r as f32).to_bits())
            };
            return Ok(true);
        }
        // Scalar integer three-same: compares and ADD/SUB on one doubleword.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && matches!((insn >> 11) & 0x1F, 0b00110 | 0b00111 | 0b10000 | 0b10001)
        {
            if (insn >> 22) & 0b11 != 0b11 {
                return Ok(false); // no byte, half or word scalar form
            }
            let unsigned = (insn >> 29) & 1 == 1;
            let a = self.vregs[((insn >> 5) & 0x1F) as usize] as u64;
            let b = self.vregs[((insn >> 16) & 0x1F) as usize] as u64;
            let all = |holds: bool| if holds { u64::MAX } else { 0 };
            let v = match ((insn >> 11) & 0x1F, unsigned) {
                (0b00110, false) => all((a as i64) > (b as i64)), // CMGT
                (0b00110, true) => all(a > b),                    // CMHI
                (0b00111, false) => all((a as i64) >= (b as i64)), // CMGE
                (0b00111, true) => all(a >= b),                   // CMHS
                (0b10000, false) => a.wrapping_add(b),            // ADD
                (0b10000, true) => a.wrapping_sub(b),             // SUB
                (0b10001, false) => all(a & b != 0),              // CMTST
                _ => all(a == b),                                 // CMEQ
            };
            self.vregs[(insn & 0x1F) as usize] = u128::from(v);
            return Ok(true);
        }
        // Two-register misc, scalar forms.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 17) & 0x1F) == 0b10000
            && ((insn >> 10) & 0b11) == 0b10
        {
            return self.simd_two_reg_misc(insn, true);
        }

        // Integer three-same / compare / logical (vector).
        let grp = (insn >> 24) & 0x1F;
        // Copy group (DUP/INS/UMOV/SMOV) is told apart from three-same by bit21 == 0.
        let copy_group = ((insn >> 21) & 0xFF) == 0b01110000 && ((insn >> 31) & 1) == 0;
        if ((insn >> 31) & 1) == 0 && grp == 0b01110 && !copy_group {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rm = ((insn >> 16) & 0x1F) as u8;
            let sz = (insn >> 22) & 0b11;
            let u = (insn >> 29) & 1; // 0 → 0x4e group, 1 → 0x6e group
            let op = (insn >> 11) & 0x1F;
            let b10 = (insn >> 10) & 1;
            let esize = match sz {
                0 => 8u32,
                1 => 16,
                2 => 32,
                _ => 64,
            };
            // Across lanes (SMAXV etc.): bits[21:17] = 11000, bit10 = 0.
            if b10 == 0 && ((insn >> 17) & 0x1F) == 0b11000 {
                return self.simd_across_lanes(insn);
            }
            // Opcodes from 0b11000 up are FP.
            if b10 == 1 && ((insn >> 21) & 1) == 1 && op >= 0b11000 {
                return self.simd_fp_three_same(insn, false);
            }
            if b10 == 1 {
                match op {
                    0b00000 => {
                        // SHADD / UHADD
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 + b as i128) >> 1) as u64
                            } else {
                                a.wrapping_add(b) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00001 => {
                        // SQADD (signed group) / UQADD (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            saturating_add(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b00010 => {
                        // SRHADD / URHADD: rounding halving add.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 + b as i128 + 1) >> 1) as u64
                            } else {
                                a.wrapping_add(b).wrapping_add(1) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00100 => {
                        // SHSUB / UHSUB: halving subtract.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                ((a as i128 - b as i128) >> 1) as u64
                            } else {
                                a.wrapping_sub(b) >> 1
                            }
                        });
                        return Ok(true);
                    }
                    0b00101 => {
                        // SQSUB / UQSUB: saturating subtract.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            saturating_sub(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    // Variable shifts: opcode low bits are the saturate and round flags.
                    0b01000..=0b01011 => {
                        let saturating = op & 1 == 1;
                        let rounding = op & 0b10 != 0;
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            shift_by_reg(a, b, esize, u != 0, rounding, saturating)
                        });
                        return Ok(true);
                    }
                    0b10000 => {
                        // ADD (signed group) / SUB (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                a.wrapping_add(b)
                            } else {
                                a.wrapping_sub(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b10001 => {
                        // CMTST (signed group) / CMEQ (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if a & b != 0 {
                                    u64::MAX
                                } else {
                                    0
                                }
                            } else if a == b {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b00111 => {
                        // CMGE (signed group) / CMHS (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            let ge = if u == 0 {
                                Self::sge(a, b, esize)
                            } else {
                                a >= b
                            };
                            if ge {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b00110 => {
                        // CMGT (signed group) / CMHI (unsigned group).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            let gt = if u == 0 {
                                Self::sge(a, b, esize) && a != b
                            } else {
                                a > b
                            };
                            if gt {
                                u64::MAX
                            } else {
                                0
                            }
                        });
                        return Ok(true);
                    }
                    0b01100 => {
                        // SMAX / UMAX.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    a
                                } else {
                                    b
                                }
                            } else {
                                a.max(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b01101 => {
                        // SMIN / UMIN.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    b
                                } else {
                                    a
                                }
                            } else {
                                a.min(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b01110 => {
                        // SABD / UABD: absolute difference.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            simd_abs_diff(a, b, esize, u != 0)
                        });
                        return Ok(true);
                    }
                    0b01111 => {
                        // SABA / UABA: accumulate the absolute difference.
                        self.simd_elem_acc(rd, rn, rm, q, esize, |a, b, d| {
                            d.wrapping_add(simd_abs_diff(a, b, esize, u != 0))
                        });
                        return Ok(true);
                    }
                    0b10010 => {
                        // MLA (signed group) / MLS (unsigned group).
                        self.simd_elem_acc(rd, rn, rm, q, esize, |a, b, d| {
                            let product = a.wrapping_mul(b);
                            if u == 0 {
                                d.wrapping_add(product)
                            } else {
                                d.wrapping_sub(product)
                            }
                        });
                        return Ok(true);
                    }
                    0b10011 if u == 0 => {
                        // MUL (PMUL, U=1, is 8-bit polynomial)
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| a.wrapping_mul(b));
                        return Ok(true);
                    }
                    0b10110 => {
                        // SQDMULH / SQRDMULH
                        let rounding = u == 1;
                        self.simd_elem(rd, rn, rm, q, esize, move |a, b| {
                            let a = sext_u64(a, esize) as i64 as i128;
                            let b = sext_u64(b, esize) as i64 as i128;
                            let mut product = 2 * a * b;
                            if rounding {
                                product += 1i128 << (esize - 1);
                            }
                            let max = (1i128 << (esize - 1)) - 1;
                            let min = -(1i128 << (esize - 1));
                            ((product >> esize).clamp(min, max)) as u64
                        });
                        return Ok(true);
                    }
                    0b10111 if u == 0 => {
                        // ADDP: pairwise addition.
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| a.wrapping_add(b));
                        return Ok(true);
                    }
                    0b10100 => {
                        // SMAXP (signed group) / UMAXP (unsigned group).
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    a
                                } else {
                                    b
                                }
                            } else {
                                a.max(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b10101 => {
                        // SMINP (signed group) / UMINP (unsigned group).
                        self.simd_pairwise(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) {
                                    b
                                } else {
                                    a
                                }
                            } else {
                                a.min(b)
                            }
                        });
                        return Ok(true);
                    }
                    0b00011 => {
                        // Bitwise logicals: selector in bits[23:21].
                        let sub = (insn >> 21) & 0b111;
                        let a = self.vregs[rn as usize];
                        let b = self.vregs[rm as usize];
                        let full = if q { u128::MAX } else { (1u128 << 64) - 1 };
                        let d = self.vregs[rd as usize];
                        let r = match (u, sub) {
                            (0, 0b001) => a & b,  // AND
                            (0, 0b011) => a & !b, // BIC
                            (0, 0b101) => a | b,  // ORR
                            (0, 0b111) => a | !b, // ORN
                            (1, 0b001) => a ^ b,  // EOR
                            // BSL masks with Vd; BIT and BIF with Vm.
                            (1, 0b011) => (a & d) | (b & !d), // BSL
                            (1, 0b101) => (a & b) | (d & !b), // BIT
                            (1, 0b111) => (a & !b) | (d & b), // BIF
                            _ => return Ok(false),
                        };
                        self.vregs[rd as usize] = r & full;
                        return Ok(true);
                    }
                    _ => {}
                }
            } else if op == 0b10011 && rm == 0 && u == 0 {
                // CMEQ <Vd>.<T>, <Vn>.<T>, #0 (compare against zero).
                self.simd_elem(
                    rd,
                    rn,
                    rm,
                    q,
                    esize,
                    |a, _| {
                        if a == 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    },
                );
                return Ok(true);
            }
            return Err(Error::Cpu(format!(
                "unimplemented SIMD three-same u={} op={:#b} sz={} at {:#x}",
                u, op, sz, self.pc
            )));
        }

        // Scalar DUP (element): lane to the bottom, rest zeroed. Matched before the vector copy group.
        if ((insn >> 21) & 0xFF) == 0b11110000
            && ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 10) & 0b111111) == 0b000001
        {
            let imm5 = (insn >> 16) & 0x1F;
            let lsb = imm5.trailing_zeros();
            // Reserved imm5 encodings.
            if imm5 == 0 || lsb > 3 {
                return Ok(false);
            }
            let esize = 8u32 << lsb;
            let index = imm5 >> (lsb + 1);
            let rd = (insn & 0x1F) as usize;
            let rn = ((insn >> 5) & 0x1F) as usize;
            let mask = (1u128 << esize) - 1;
            self.vregs[rd] = (self.vregs[rn] >> (index * esize)) & mask;
            return Ok(true);
        }

        // Copy / element moves and table lookup (bits[28:21] == 0111 0000).
        if ((insn >> 21) & 0xFF) != 0b01110000 || ((insn >> 31) & 1) != 0 {
            return Ok(false);
        }
        let op = (insn >> 29) & 1;

        // TBL / TBX: split off from the copy group by bits[11:10] = 00.
        if op == 0 && ((insn >> 15) & 1) == 0 && ((insn >> 10) & 0b11) == 0 {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as usize;
            let rn = ((insn >> 5) & 0x1F) as usize;
            let rm = ((insn >> 16) & 0x1F) as usize;
            let len = ((insn >> 13) & 0b11) as usize + 1;
            // Out-of-range index: TBL writes zero, TBX keeps Vd.
            let keep_on_miss = (insn >> 12) & 1 == 1;
            let indices = self.vregs[rm].to_le_bytes();
            let mut out = self.vregs[rd].to_le_bytes();
            let lanes = if q { 16 } else { 8 };
            for (i, slot) in out.iter_mut().enumerate().take(lanes) {
                let idx = indices[i] as usize;
                if idx < len * 16 {
                    // The table wraps past v31.
                    *slot = self.vregs[(rn + idx / 16) % 32].to_le_bytes()[idx % 16];
                } else if !keep_on_miss {
                    *slot = 0;
                }
            }
            out[lanes..].fill(0);
            self.vregs[rd] = u128::from_le_bytes(out);
            return Ok(true);
        }

        let q = (insn >> 30) & 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let imm5 = (insn >> 16) & 0x1F;

        if op == 1 {
            // INS (element): imm5 gives size and destination index, imm4 the source index.
            if q == 0 || imm5 == 0 || ((insn >> 15) & 1) != 0 || ((insn >> 10) & 1) == 0 {
                return Ok(false);
            }
            let lsb = imm5.trailing_zeros();
            if lsb > 3 {
                return Ok(false);
            }
            let esize = 8u32 << lsb;
            let dst_index = imm5 >> (lsb + 1);
            let src_index = ((insn >> 11) & 0xF) >> lsb;
            let mask = (1u128 << esize) - 1;
            let val = (self.vregs[rn as usize] >> (src_index * esize)) & mask;
            let shift = dst_index * esize;
            let v = self.vregs[rd as usize];
            // INS leaves other lanes, including the top half, untouched.
            self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
            return Ok(true);
        }

        match (insn >> 10) & 0b111111 {
            0b000111 => {
                // INS (general): esize = 8 << ctz(imm5), index = imm5 >> (ctz + 1).
                let lsb = imm5.trailing_zeros();
                if lsb > 3 {
                    return Ok(false);
                }
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = index * esize;
                let mask = (1u128 << esize) - 1;
                let v = self.vregs[rd as usize];
                let val = (self.read_zr(rn) as u128) & mask;
                self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
                Ok(true)
            }
            0b000011 if imm5 != 0 => {
                // DUP (general): esize = 8 << ctz(imm5).
                let esize = 8u32 << imm5.trailing_zeros();
                let elements = if q == 1 { 128 / esize } else { 64 / esize };
                let val = (self.read_zr(rn) as u128) & ((1u128 << esize) - 1);
                let mut v: u128 = 0;
                for i in 0..elements {
                    v |= val << (i * esize);
                }
                self.vregs[rd as usize] = v;
                Ok(true)
            }
            0b000001 if imm5 != 0 => {
                // DUP (element)
                let lsb = imm5.trailing_zeros();
                if lsb > 3 {
                    return Ok(false);
                }
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let elements = if q == 1 { 128 / esize } else { 64 / esize };
                let mask = (1u128 << esize) - 1;
                let val = (self.vregs[rn as usize] >> (index * esize)) & mask;
                let mut v: u128 = 0;
                for i in 0..elements {
                    v |= val << (i * esize);
                }
                self.vregs[rd as usize] = v;
                Ok(true)
            }
            0b001111 => {
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = index * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                self.write_zr(rd, val as u64);
                Ok(true)
            }
            0b001011 => {
                // SMOV: sign-extended lane extract.
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = index * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                let val = sext_u64(val as u64, esize);
                self.write_zr(rd, val);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    // Scalar floating point. FP exception flags are not modelled.

    /// Element-wise binary op over `esize`-bit lanes; `q` selects 128 vs 64 bits.
    pub(super) fn simd_elem<F: Fn(u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
        esize: u32,
        f: F,
    ) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            out = set_lane(out, esize, i, f(lane(a, esize, i), lane(b, esize, i)));
        }
        self.vregs[rd as usize] = out;
    }

    pub(super) fn simd_elem_n<F: Fn(u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        lanes: u32,
        esize: u32,
        f: F,
    ) {
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            out = set_lane(out, esize, i, f(lane(a, esize, i), lane(b, esize, i)));
        }
        self.vregs[rd as usize] = out;
    }

    /// ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2 over `esize`-bit lanes.
    pub(super) fn simd_permute(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, op: u32) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = elem_mask(esize);
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let get = |r: u128, i: u32| (r >> (esize * i)) & mask;
        let mut out: u128 = 0;
        match op {
            // UZP1 (even elements) / UZP2 (odd) of Vn:Vm.
            0b000110 | 0b010110 => {
                let start = u32::from(op == 0b010110);
                for i in 0..lanes {
                    let index = start + 2 * (i % half);
                    let v = if i < half {
                        get(a, index)
                    } else {
                        get(b, index)
                    };
                    out |= v << (esize * i);
                }
            }
            // TRN1 (even elements) / TRN2 (odd) of both, interleaved.
            0b001010 | 0b011010 => {
                let odd = u32::from(op == 0b011010);
                for i in 0..half {
                    out |= get(a, 2 * i + odd) << (esize * 2 * i);
                    out |= get(b, 2 * i + odd) << (esize * (2 * i + 1));
                }
            }
            // ZIP1 (low halves) / ZIP2 (high halves), interleaved.
            _ => {
                let base = if op == 0b011110 { half } else { 0 };
                for i in 0..half {
                    out |= get(a, base + i) << (esize * 2 * i);
                    out |= get(b, base + i) << (esize * (2 * i + 1));
                }
            }
        }
        self.vregs[rd as usize] = out;
    }

    /// Lanewise op that also reads the destination lane (MLA/MLS, SABA/UABA).
    pub(super) fn simd_elem_acc<F: Fn(u64, u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
        esize: u32,
        f: F,
    ) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let d = self.vregs[rd as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let position = esize * i;
            let va = ((a >> position) & mask) as u64;
            let vb = ((b >> position) & mask) as u64;
            let vd = ((d >> position) & mask) as u64;
            out |= (f(va, vb, vd) as u128 & mask) << position;
        }
        self.vregs[rd as usize] = out;
    }

    /// Pairwise op: low half pairs Vn's lanes, high half Vm's.
    pub(super) fn simd_pairwise<F: Fn(u64, u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        q: bool,
        esize: u32,
        f: F,
    ) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..half {
            let a0 = ((a >> (esize * 2 * i)) & mask) as u64;
            let a1 = ((a >> (esize * (2 * i + 1))) & mask) as u64;
            let b0 = ((b >> (esize * 2 * i)) & mask) as u64;
            let b1 = ((b >> (esize * (2 * i + 1))) & mask) as u64;
            out |= (f(a0, a1) as u128 & mask) << (esize * i);
            out |= (f(b0, b1) as u128 & mask) << (esize * (i + half));
        }
        self.vregs[rd as usize] = out;
    }
    pub(super) fn sge(a: u64, b: u64, bits: u32) -> bool {
        let shift = 64 - bits;
        ((a << shift) as i64) >= ((b << shift) as i64)
    }

    /// Across lanes, integer forms: reduce Vn into one scalar lane.
    fn simd_across_lanes(&mut self, insn: u32) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let size = (insn >> 22) & 0b11;
        let opcode = (insn >> 12) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        // FMAXNMV / FMINNMV / FMAXV / FMINV: reduced in pairs, (0 op 1) op (2 op 3).
        if u == 1 && matches!(opcode, 0b01100 | 0b01111) {
            if !q || (insn >> 22) & 1 != 0 {
                return Ok(false); // two lanes, or doubles: unallocated
            }
            let min = (insn >> 23) & 1 == 1;
            let op = |x: f64, y: f64| match (opcode == 0b01100, min) {
                (true, false) => fp_maxnum(x, y),
                (true, true) => fp_minnum(x, y),
                (false, false) => fp_max(x, y),
                (false, true) => fp_min(x, y),
            };
            let a = self.vregs[rn as usize];
            let lane = |i: u32| f64::from(f32::from_bits((a >> (32 * i)) as u32));
            let reduced = op(op(lane(0), lane(1)), op(lane(2), lane(3))) as f32;
            self.vregs[rd as usize] = u128::from(reduced.to_bits());
            return Ok(true);
        }
        if size == 0b11 || !matches!(opcode, 0b00011 | 0b01010 | 0b11010 | 0b11011) {
            return Ok(false);
        }
        let esize = 8u32 << size;
        let lanes = if q { 128 / esize } else { 64 / esize };
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let elem = |i: u32| ((a >> (esize * i)) & mask) as u64;
        let signed = u == 0;
        let result = match opcode {
            0b00011 => {
                // SADDLV / UADDLV: sum widened to double the element size.
                let mut sum: i128 = 0;
                for i in 0..lanes {
                    let v = elem(i);
                    sum += if signed {
                        sext_u64(v, esize) as i64 as i128
                    } else {
                        v as i128
                    };
                }
                (sum as u128) & ((1u128 << (esize * 2)) - 1)
            }
            0b01010 | 0b11010 => {
                // SMAXV / UMAXV (0b01010), SMINV / UMINV (0b11010).
                let want_max = opcode == 0b01010;
                let mut best = elem(0);
                for i in 1..lanes {
                    let v = elem(i);
                    let v_wins = if signed {
                        if want_max {
                            !Self::sge(best, v, esize)
                        } else {
                            Self::sge(best, v, esize) && best != v
                        }
                    } else if want_max {
                        v > best
                    } else {
                        v < best
                    };
                    if v_wins {
                        best = v;
                    }
                }
                best as u128
            }
            0b11011 => {
                // ADDV: wrapping sum, no widening.
                let mut sum: u128 = 0;
                for i in 0..lanes {
                    sum = sum.wrapping_add(elem(i) as u128);
                }
                sum & mask
            }
            _ => unreachable!(),
        };
        self.vregs[rd as usize] = result;
        Ok(true)
    }

    /// Fixed-point SCVTF/UCVTF (`to_int` false) and FCVTZS/FCVTZU (`to_int` true).
    #[allow(clippy::too_many_arguments)]
    fn simd_fixed_convert(
        &mut self,
        rd: u8,
        rn: u8,
        lanes: u32,
        esize: u32,
        fbits: u32,
        to_int: bool,
        unsigned: bool,
    ) -> Result<()> {
        if esize == 16 {
            return Err(Error::Cpu(
                "unimplemented half-precision SIMD fixed-point convert".to_owned(),
            ));
        }
        let mask = if esize == 64 {
            u64::MAX
        } else {
            (1u64 << esize) - 1
        };
        let scale = Self::pow2(fbits);
        let src = self.vregs[rn as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let raw = (src >> (esize * i)) as u64 & mask;
            let value = if to_int {
                let f = if esize == 64 {
                    f64::from_bits(raw)
                } else {
                    f64::from(f32::from_bits(raw as u32))
                };
                round_to_int_sized(f * scale, Rounding::TowardZero, !unsigned, esize) & mask
            } else {
                let int = match (unsigned, esize == 64) {
                    (false, true) => raw as i64 as f64,
                    (false, false) => f64::from(raw as u32 as i32),
                    (true, true) => raw as f64,
                    (true, false) => f64::from(raw as u32),
                };
                if esize == 64 {
                    (int / scale).to_bits()
                } else {
                    u64::from(((int / scale) as f32).to_bits())
                }
            };
            out |= u128::from(value) << (esize * i);
        }
        self.vregs[rd as usize] = out;
        Ok(())
    }

    /// Shift by immediate: right shift is `2*esize - imm`, left is `imm - esize`.
    pub(super) fn simd_shift_imm(
        &mut self,
        rd: u8,
        rn: u8,
        q: bool,
        u: bool,
        opcode: u32,
        esize: u32,
        imm: u32,
    ) -> Result<()> {
        let mask: u128 = if esize >= 128 {
            u128::MAX
        } else {
            (1u128 << esize) - 1
        };
        let src = self.vregs[rn as usize];
        let dst = self.vregs[rd as usize];

        // SSHLL/USHLL (and SXTL/UXTL): widen from one half of Vn.
        if opcode == 0b10100 {
            let shift = imm - esize;
            let wide = 2 * esize;
            let wide_mask: u128 = if wide >= 128 {
                u128::MAX
            } else {
                (1u128 << wide) - 1
            };
            let lanes = 64 / esize;
            let base = if q { lanes } else { 0 };
            let mut out: u128 = 0;
            for i in 0..lanes {
                let raw = ((src >> (esize * (i + base))) & mask) as u64;
                let extended = if u { raw } else { sext_u64(raw, esize) };
                let value = ((extended as u128) << shift) & wide_mask;
                out |= value << (wide * i);
            }
            self.vregs[rd as usize] = out;
            return Ok(());
        }

        let lanes = if q { 128 / esize } else { 64 / esize };
        let left = matches!(opcode, 0b01010 | 0b01100 | 0b01110);
        let shift = if left { imm - esize } else { 2 * esize - imm };
        let mut out: u128 = 0;
        for i in 0..lanes {
            let position = esize * i;
            let raw = ((src >> position) & mask) as u64;
            let old = ((dst >> position) & mask) as u64;
            let value: u64 = match opcode {
                // SSHR / USHR, SSRA / USRA, SRSHR / URSHR, SRSRA / URSRA.
                0b00000 | 0b00010 | 0b00100 | 0b00110 => {
                    let rounding = opcode & 0b00100 != 0;
                    let round = if rounding && shift > 0 {
                        1u64 << (shift - 1)
                    } else {
                        0
                    };
                    let shifted = if u {
                        raw.wrapping_add(round) >> shift.min(63)
                    } else {
                        let signed = sext_u64(raw, esize) as i64;
                        (signed.wrapping_add(round as i64) >> shift.min(63)) as u64
                    };
                    if opcode & 0b00010 != 0 {
                        old.wrapping_add(shifted)
                    } else {
                        shifted
                    }
                }
                // SRI: shift right and insert, keeping the high bits of Vd.
                0b01000 => {
                    let keep = if shift >= esize {
                        mask as u64
                    } else {
                        !(mask as u64 >> shift)
                    };
                    (old & keep) | ((raw >> shift.min(63)) & !keep)
                }
                // SHL, or SLI when the insert variant is selected.
                0b01010 => {
                    let shifted = raw.wrapping_shl(shift);
                    if u {
                        let keep = if shift == 0 { 0 } else { (1u64 << shift) - 1 };
                        (old & keep) | (shifted & !keep)
                    } else {
                        shifted
                    }
                }
                other => {
                    return Err(Error::Cpu(format!(
                        "unimplemented SIMD shift-by-immediate opcode {:#07b}",
                        other
                    )))
                }
            };
            out |= ((value as u128) & mask) << position;
        }
        self.vregs[rd as usize] = out;
        Ok(())
    }

    /// Lanewise unary op on raw lane bits.
    pub(super) fn simd_lane_unary_n<F: Fn(u64) -> u64>(
        &mut self,
        rd: u8,
        rn: u8,
        lanes: u32,
        esize: u32,
        f: F,
    ) {
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let v = ((a >> (esize * i)) & mask) as u64;
            out |= (f(v) as u128 & mask) << (esize * i);
        }
        self.vregs[rd as usize] = out;
    }

    /// Two-register misc. FP forms are keyed by `(U, size<1>, opcode)`.
    fn simd_two_reg_misc(&mut self, insn: u32, scalar: bool) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let size = (insn >> 22) & 0b11;
        let opcode = (insn >> 12) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let esize = 8u32 << size;
        let lanes = |esize: u32| {
            if scalar {
                1
            } else if q {
                128 / esize
            } else {
                64 / esize
            }
        };

        // Integer forms.
        match (u, opcode) {
            // REV64 / REV32 / REV16
            (0, 0b00000) | (1, 0b00000) | (0, 0b00001) => {
                if scalar {
                    return Ok(false);
                }
                let container = match (u, opcode) {
                    (0, 0b00000) => 64u32,
                    (1, 0b00000) => 32,
                    _ => 16,
                };
                if esize >= container {
                    return Ok(false);
                }
                let lanes = if q { 128 / esize } else { 64 / esize };
                let per_container = container / esize;
                let mask = (1u128 << esize) - 1;
                let a = self.vregs[rn as usize];
                let mut out: u128 = 0;
                for i in 0..lanes {
                    let group = i / per_container;
                    let within = i % per_container;
                    let src = group * per_container + (per_container - 1 - within);
                    out |= ((a >> (esize * src)) & mask) << (esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            // CLS / CLZ: count leading sign bits / leading zeros.
            (_, 0b00100) => {
                if scalar {
                    return Ok(false);
                }
                let signed = u == 0;
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    let shifted = v << (64 - esize);
                    if signed {
                        let inverted = if shifted >> 63 == 1 {
                            !shifted
                        } else {
                            shifted
                        };
                        u64::from((inverted << 1).leading_zeros().min(esize - 1))
                    } else {
                        u64::from(shifted.leading_zeros().min(esize))
                    }
                });
                return Ok(true);
            }
            // CNT (population count per byte) and NOT / RBIT.
            (0, 0b00101) => {
                if scalar {
                    return Ok(false);
                }
                if esize != 8 {
                    return Ok(false);
                }
                self.simd_lane_unary_n(rd, rn, lanes(8), 8, |v| u64::from(v.count_ones()));
                return Ok(true);
            }
            (1, 0b00101) => {
                if scalar {
                    return Ok(false);
                }
                let full = if q { u128::MAX } else { (1u128 << 64) - 1 };
                match size {
                    0b00 => self.vregs[rd as usize] = !self.vregs[rn as usize] & full,
                    0b01 => {
                        self.simd_lane_unary_n(rd, rn, lanes(8), 8, |v| {
                            u64::from((v as u8).reverse_bits())
                        });
                    }
                    _ => return Ok(false),
                }
                return Ok(true);
            }
            // SADDLP / UADDLP and SADALP / UADALP
            (_, 0b00010) | (_, 0b00110) => {
                if scalar {
                    return Ok(false);
                }
                let accumulate = opcode == 0b00110;
                let signed = u == 0;
                if esize == 64 {
                    return Ok(false);
                }
                let dest_esize = esize * 2;
                let lanes = if q { 128 / dest_esize } else { 64 / dest_esize };
                let src_mask = (1u128 << esize) - 1;
                let dest_mask = (1u128 << dest_esize) - 1;
                let a = self.vregs[rn as usize];
                let d = self.vregs[rd as usize];
                let mut out: u128 = 0;
                for i in 0..lanes {
                    let lo = ((a >> (esize * 2 * i)) & src_mask) as u64;
                    let hi = ((a >> (esize * (2 * i + 1))) & src_mask) as u64;
                    let sum = if signed {
                        (sext_u64(lo, esize) as i64).wrapping_add(sext_u64(hi, esize) as i64) as u64
                    } else {
                        lo.wrapping_add(hi)
                    };
                    let sum = if accumulate {
                        let acc = ((d >> (dest_esize * i)) & dest_mask) as u64;
                        sum.wrapping_add(acc)
                    } else {
                        sum
                    };
                    out |= (sum as u128 & dest_mask) << (dest_esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            // ABS / NEG, and the compares against zero.
            (0, 0b01011) => {
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    (sext_u64(v, esize) as i64).wrapping_abs() as u64
                });
                return Ok(true);
            }
            (1, 0b01011) => {
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, |v| {
                    (v as i64).wrapping_neg() as u64
                });
                return Ok(true);
            }
            (0, 0b01000) | (0, 0b01001) | (0, 0b01010) | (1, 0b01000) | (1, 0b01001) => {
                let kind = (u, opcode);
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    let signed = sext_u64(v, esize) as i64;
                    let holds = match kind {
                        (0, 0b01000) => signed > 0,  // CMGT #0
                        (0, 0b01001) => signed == 0, // CMEQ #0
                        (0, 0b01010) => signed < 0,  // CMLT #0
                        (1, 0b01000) => signed >= 0, // CMGE #0
                        _ => signed <= 0,            // CMLE #0
                    };
                    if holds {
                        u64::MAX
                    } else {
                        0
                    }
                });
                return Ok(true);
            }
            // XTN / SQXTN / UQXTN / SQXTUN: `size` is the destination width.
            (0, 0b10010) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, false, false, false);
                return Ok(true);
            }
            (0, 0b10100) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, true, false, true);
                return Ok(true);
            }
            (1, 0b10010) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, true, true, true);
                return Ok(true);
            }
            (1, 0b10100) => {
                self.simd_shrn(rd, rn, q, esize, 0, false, false, false, true);
                return Ok(true);
            }
            // SHLL: widen each lane and shift it left by the element width.
            (1, 0b10011) => {
                if scalar {
                    return Ok(false);
                }
                if esize == 64 {
                    return Ok(false);
                }
                let dest_esize = esize * 2;
                let lanes = 64 / esize;
                let src_mask = (1u128 << esize) - 1;
                let src = self.vregs[rn as usize];
                let base = if q {
                    u128::from(lanes) * u128::from(esize)
                } else {
                    0
                };
                let mut out: u128 = 0;
                for i in 0..lanes {
                    let shift = esize * i + base as u32;
                    let v = (src >> shift) & src_mask;
                    out |= (v << esize) << (dest_esize * i);
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            // FCVTL / FCVTN: half <-> single and single <-> double.
            (0, 0b10111) if size <= 0b01 => {
                if scalar {
                    return Ok(false);
                }
                let src = self.vregs[rn as usize];
                let base = if q { 64 } else { 0 };
                let mut out: u128 = 0;
                if size == 0b00 {
                    for i in 0..4u32 {
                        let h = ((src >> (base + 16 * i)) & 0xFFFF) as u16;
                        out |= u128::from(f16_to_f32(h).to_bits()) << (32 * i);
                    }
                } else {
                    for i in 0..2u32 {
                        let bits = ((src >> (base + 32 * i)) & 0xFFFF_FFFF) as u32;
                        let wide = f64::from(f32::from_bits(bits)).to_bits();
                        out |= u128::from(wide) << (64 * i);
                    }
                }
                self.vregs[rd as usize] = out;
                return Ok(true);
            }
            (0, 0b10110) if size <= 0b01 => {
                if scalar {
                    return Ok(false);
                }
                let src = self.vregs[rn as usize];
                let mut narrowed: u128 = 0;
                if size == 0b00 {
                    // Via double so the half is rounded once.
                    for i in 0..4u32 {
                        let bits = ((src >> (32 * i)) & 0xFFFF_FFFF) as u32;
                        let h = f64_to_f16(f64::from(f32::from_bits(bits)));
                        narrowed |= u128::from(h) << (16 * i);
                    }
                } else {
                    for i in 0..2u32 {
                        let bits = (src >> (64 * i)) as u64;
                        let narrow = (f64::from_bits(bits) as f32).to_bits();
                        narrowed |= u128::from(narrow) << (32 * i);
                    }
                }
                self.vregs[rd as usize] = if q {
                    (self.vregs[rd as usize] & ((1u128 << 64) - 1)) | (narrowed << 64)
                } else {
                    narrowed
                };
                return Ok(true);
            }
            _ => {}
        }

        // FP forms.
        let double = size & 1 == 1;
        if double && !q && !scalar {
            return Ok(false); // a single 64-bit lane isn't a vector form
        }
        let key = (u << 6) | ((size >> 1) << 5) | opcode;
        let esize = if double { 64 } else { 32 };
        if matches!(key, 0x2c | 0x2d | 0x2e | 0x6c | 0x6d) {
            self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                let a = if double {
                    f64::from_bits(v)
                } else {
                    f64::from(f32::from_bits(v as u32))
                };
                let holds = match key {
                    0x2c => a > 0.0,
                    0x2d => a == 0.0,
                    0x2e => a < 0.0,
                    0x6c => a >= 0.0,
                    _ => a <= 0.0,
                };
                if holds {
                    u64::MAX
                } else {
                    0
                }
            });
            return Ok(true);
        }
        // Float -> integer converts, with the rounding mode in the opcode.
        if matches!(
            key,
            0x1a | 0x1b | 0x1c | 0x3a | 0x3b | 0x5a | 0x5b | 0x5c | 0x7a | 0x7b
        ) {
            let signed = u == 0;
            let rounding = match key & 0x1F {
                0b11010 if size >> 1 == 0 => Rounding::TiesEven, // FCVTNS/FCVTNU
                0b11010 => Rounding::TowardPos,                  // FCVTPS/FCVTPU
                0b11011 if size >> 1 == 0 => Rounding::TowardNeg, // FCVTMS/FCVTMU
                0b11011 => Rounding::TowardZero,                 // FCVTZS/FCVTZU
                _ => Rounding::TiesAway,                         // FCVTAS/FCVTAU
            };
            self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                let a = if double {
                    f64::from_bits(v)
                } else {
                    f64::from(f32::from_bits(v as u32))
                };
                round_to_int_sized(a, rounding, signed, esize)
            });
            return Ok(true);
        }
        // Integer -> float, the FP unary ops and the reciprocal estimates.
        match key {
            // SCVTF / UCVTF
            0x1d | 0x5d => {
                let signed = u == 0;
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    if double {
                        let f = if signed { (v as i64) as f64 } else { v as f64 };
                        f.to_bits()
                    } else {
                        let f = if signed {
                            (v as i32) as f32
                        } else {
                            (v as u32) as f32
                        };
                        u64::from(f.to_bits())
                    }
                });
                Ok(true)
            }
            // FABS / FNEG: sign-bit operations, so no float round-trip.
            0x2f | 0x6f => {
                let negate = u == 1;
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    let sign = 1u64 << (esize - 1);
                    if negate {
                        v ^ sign
                    } else {
                        v & !sign
                    }
                });
                Ok(true)
            }
            // FSQRT, FRINTx and the reciprocal estimates.
            0x7f | 0x18 | 0x19 | 0x38 | 0x39 | 0x58 | 0x59 | 0x79 | 0x3d | 0x7d => {
                let mode = fpcr_rounding(self.fpcr);
                self.simd_lane_unary_n(rd, rn, lanes(esize), esize, move |v| {
                    if double {
                        let a = f64::from_bits(v);
                        let r = match key {
                            0x7f => a.sqrt(),
                            0x18 => a.round_ties_even(), // FRINTN
                            0x19 => a.floor(),           // FRINTM
                            0x38 => a.ceil(),            // FRINTP
                            0x39 => a.trunc(),           // FRINTZ
                            0x58 => a.round(),           // FRINTA
                            0x59 | 0x79 => round_to_integral(a, mode), // FRINTX / FRINTI
                            0x3d => {
                                return super::fp::recip_estimate_bits(v, 64);
                            }
                            0x7d => {
                                return super::fp::rsqrt_estimate_bits(v, 64);
                            }
                            _ => unreachable!("unhandled two-register misc key"),
                        };
                        r.to_bits()
                    } else {
                        let a = f32::from_bits(v as u32);
                        let r = match key {
                            0x7f => a.sqrt(),
                            0x18 => a.round_ties_even(),
                            0x19 => a.floor(),
                            0x38 => a.ceil(),
                            0x39 => a.trunc(),
                            0x58 => a.round(),
                            0x59 | 0x79 => round_to_integral(f64::from(a), mode) as f32,
                            0x3d => {
                                return super::fp::recip_estimate_bits(v & 0xFFFF_FFFF, 32);
                            }
                            0x7d => {
                                return super::fp::rsqrt_estimate_bits(v & 0xFFFF_FFFF, 32);
                            }
                            _ => unreachable!("unhandled two-register misc key"),
                        };
                        u64::from(r.to_bits())
                    }
                });
                Ok(true)
            }
            // URECPE / URSQRTE: the unsigned-integer estimates.
            0x3c | 0x7c => {
                let sqrt = key == 0x7c;
                self.simd_lane_unary_n(rd, rn, lanes(32), 32, move |v| {
                    let a = v as u32;
                    if a == 0 {
                        return u64::from(u32::MAX);
                    }
                    let f = a as f64 / 4_294_967_296.0;
                    let r = if sqrt { 1.0 / f.sqrt() } else { 1.0 / f };
                    (r * 4_294_967_296.0).min(f64::from(u32::MAX)) as u64
                });
                Ok(true)
            }
            _ => Err(Error::Cpu(format!(
                "unimplemented SIMD two-register misc u={} size={} opcode={:#07b} at {:#x}",
                u, size, opcode, self.pc
            ))),
        }
    }

    /// Three-same, floating point (opcode >= 0b11000).
    fn simd_fp_three_same(&mut self, insn: u32, scalar: bool) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let a = (insn >> 23) & 1;
        let double = (insn >> 22) & 1 == 1;
        let rm = ((insn >> 16) & 0x1F) as u8;
        let opcode = (insn >> 11) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        if double && !q && !scalar {
            return Ok(false); // a single 64-bit lane isn't a vector form
        }
        let esize = if double { 64u32 } else { 32 };
        let lanes = if scalar {
            1
        } else if q {
            128 / esize
        } else {
            64 / esize
        };
        let key = (u << 6) | (a << 5) | opcode;
        let arith = |x: f64, y: f64| -> f64 {
            match key {
                0x1a => x + y,            // FADD
                0x3a => x - y,            // FSUB
                0x5b => x * y,            // FMUL
                0x1b => fmulx(x, y),      // FMULX
                0x5f => x / y,            // FDIV
                0x7a => (x - y).abs(),    // FABD
                0x1e => fp_max(x, y),     // FMAX
                0x3e => fp_min(x, y),     // FMIN
                0x18 => fp_maxnum(x, y),  // FMAXNM
                0x38 => fp_minnum(x, y),  // FMINNM
                0x1f => 2.0 - x * y,      // FRECPS
                _ => (3.0 - x * y) / 2.0, // FRSQRTS
            }
        };
        if matches!(
            key,
            0x1a | 0x3a | 0x5b | 0x1b | 0x5f | 0x7a | 0x1e | 0x3e | 0x18 | 0x38 | 0x1f | 0x3f
        ) {
            if double {
                self.simd_elem_n(rd, rn, rm, lanes, esize, |x, y| {
                    arith(f64::from_bits(x), f64::from_bits(y)).to_bits()
                });
            } else {
                self.simd_elem_n(rd, rn, rm, lanes, esize, |x, y| {
                    let r = arith(
                        f64::from(f32::from_bits(x as u32)),
                        f64::from(f32::from_bits(y as u32)),
                    ) as f32;
                    u64::from(r.to_bits())
                });
            }
            return Ok(true);
        }
        match key {
            // FMLA / FMLS: fused multiply-accumulate into Vd.
            0x19 | 0x39 if !scalar => {
                let subtract = a == 1;
                self.simd_elem_acc(rd, rn, rm, q, esize, move |x, y, d| {
                    if double {
                        let n = f64::from_bits(x);
                        let n = if subtract { -n } else { n };
                        n.mul_add(f64::from_bits(y), f64::from_bits(d)).to_bits()
                    } else {
                        let n = f32::from_bits(x as u32);
                        let n = if subtract { -n } else { n };
                        u64::from(
                            n.mul_add(f32::from_bits(y as u32), f32::from_bits(d as u32))
                                .to_bits(),
                        )
                    }
                });
                Ok(true)
            }
            // The compares, including the absolute-value forms (FACGE/FACGT).
            0x1c | 0x5c | 0x7c | 0x5d | 0x7d => {
                self.simd_elem_n(rd, rn, rm, lanes, esize, move |x, y| {
                    let (mut fx, mut fy) = if double {
                        (f64::from_bits(x), f64::from_bits(y))
                    } else {
                        (
                            f64::from(f32::from_bits(x as u32)),
                            f64::from(f32::from_bits(y as u32)),
                        )
                    };
                    if key == 0x5d || key == 0x7d {
                        fx = fx.abs();
                        fy = fy.abs();
                    }
                    let holds = match key {
                        0x1c => fx == fy,        // FCMEQ
                        0x5c | 0x5d => fx >= fy, // FCMGE / FACGE
                        _ => fx > fy,            // FCMGT / FACGT
                    };
                    if holds {
                        u64::MAX
                    } else {
                        0
                    }
                });
                Ok(true)
            }
            // The pairwise reductions.
            0x5a | 0x5e | 0x7e | 0x58 | 0x78 if !scalar => {
                self.simd_pairwise(rd, rn, rm, q, esize, move |x, y| {
                    let (fx, fy) = if double {
                        (f64::from_bits(x), f64::from_bits(y))
                    } else {
                        (
                            f64::from(f32::from_bits(x as u32)),
                            f64::from(f32::from_bits(y as u32)),
                        )
                    };
                    let r = match key {
                        0x5a => fx + fy,           // FADDP
                        0x5e => fp_max(fx, fy),    // FMAXP
                        0x7e => fp_min(fx, fy),    // FMINP
                        0x58 => fp_maxnum(fx, fy), // FMAXNMP
                        _ => fp_minnum(fx, fy),    // FMINNMP
                    };
                    if double {
                        r.to_bits()
                    } else {
                        u64::from((r as f32).to_bits())
                    }
                });
                Ok(true)
            }
            _ => Err(Error::Cpu(format!(
                "unimplemented SIMD FP three-same u={} a={} opcode={:#07b} at {:#x}",
                u, a, opcode, self.pc
            ))),
        }
    }

    /// SHRN/RSHRN and saturating forms; `Q=1` targets the high half.
    pub(super) fn simd_shrn(
        &mut self,
        rd: u8,
        rn: u8,
        q: bool,
        dest_esize: u32,
        shift: u32,
        rounding: bool,
        signed_src: bool,
        to_unsigned: bool,
        saturating: bool,
    ) {
        let src_esize = 2 * dest_esize;
        let src_elements = 128 / src_esize;
        let dest_mask = (1u128 << dest_esize) - 1;
        let src = self.vregs[rn as usize];
        let round_add = if rounding { 1i64 << (shift - 1) } else { 0 };
        let mut narrowed = [0u64; 16];
        for i in 0..src_elements {
            let raw = lane(src, src_esize, i);
            let shifted = if signed_src {
                let v = (raw as i64) << (64 - src_esize) >> (64 - src_esize);
                v.wrapping_add(round_add) >> shift
            } else {
                (raw.wrapping_add(round_add as u64) >> shift) as i64
            };
            let mut v = shifted;
            if saturating || to_unsigned {
                let (min, max) = if to_unsigned || !signed_src {
                    (0i64, dest_mask as i64)
                } else {
                    (-(1i64 << (dest_esize - 1)), (1i64 << (dest_esize - 1)) - 1)
                };
                v = v.clamp(min, max);
            }
            narrowed[i as usize] = (v as u64) & (dest_mask as u64);
        }
        let mut out: u128 = 0;
        for i in 0..src_elements {
            out = set_lane(out, dest_esize, i, narrowed[i as usize]);
        }
        if q {
            self.vregs[rd as usize] = (self.vregs[rd as usize] & ((1u128 << 64) - 1)) | (out << 64);
        } else {
            self.vregs[rd as usize] = out & ((1u128 << 64) - 1);
        }
    }
}
