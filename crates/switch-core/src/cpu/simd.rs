//! NEON/AdvSIMD: the vector instruction subset compiled homebrew uses
//! (three-same integer ops, logicals, permutes, compares, shifts and the
//! element moves libnx's string and memory routines are built from).

use super::bits::*;
use super::crypto::poly_mul;
use super::Cpu;
use crate::{Error, Result};

impl Cpu {
    /// Minimal SIMD data-processing for the vector registers — just enough for
    /// the libnx `memset`/`memcpy` hot path (Phase 1 keeps NEON out of scope).
    ///
    /// Handled forms (fixed bits `[30:23] == 10_011100`, `[22:20] == 000`):
    /// * `DUP <Vd>.<T>, <Rn>`  — replicate the low element of a GPR across the
    ///   vector (`bits[15:10] == 000011`).
    /// * `MOV <Xd>, <Vn>.D[<m>]` (UMOV) — copy a 64-bit lane to a GPR
    ///   (`bits[15:10] == 001111`, 64-bit lane form `imm5 == 01000 | m`).
    pub(super) fn try_simd(&mut self, insn: u32) -> Result<bool> {
        // ---- AES and SHA ----
        // The three-register SHA forms share bits[28:21] with the scalar DUP
        // below, so the crypto group has to get first look at the encoding.
        if self.try_crypto(insn)? {
            return Ok(true);
        }

        // ---- narrowing shifts (SHRN / RSHRN / SQSHRN / ...) ----
        // bits[31]=0, bits[28:24]=01111, bit23=0. These share the group with
        // MOVI (bits[28:23]=011110), so they must be checked first; the
        // opcode lives in bits[15:11] and the shift in bits[22:16].
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01111
            && ((insn >> 23) & 1) == 0
            && ((insn >> 10) & 1) == 1
        {
            let op = (insn >> 11) & 0x1F;
            if matches!(op, 0b10000 | 0b10001 | 0b10010 | 0b10011) {
                let q = (insn >> 30) & 1 == 1;
                let u = (insn >> 29) & 1;
                let rd = (insn & 0x1F) as u8;
                let rn = ((insn >> 5) & 0x1F) as u8;
                // The destination element size comes from `immh` (bits[22:19]),
                // not bit22 alone: 0001 narrows to bytes, 001x to halfwords,
                // 01xx to words. Reading one bit made every form but the
                // byte-destination one fall through as unimplemented
                // (`shrn v2.4h, v18.4s, #16`).
                let immh = (insn >> 19) & 0xF;
                // `immh == 0` is MOVI sharing this space, so fall through to it
                // rather than reporting the instruction as unimplemented.
                let dest_esize = match immh {
                    0b0001 => Some(8u32),
                    0b0010 | 0b0011 => Some(16),
                    0b0100..=0b0111 => Some(32),
                    _ => None,
                };
                let dest_esize = dest_esize.unwrap_or(0);
                let shift_field = (insn >> 16) & 0x7F;
                let shift = (2 * dest_esize).saturating_sub(shift_field as u32);
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

        // ---- shift by immediate (SSHR/USHR/SHL/SLI/SRI/SSHLL/...) ----
        // Vector: `0 Q U 011110 immh immb opcode 1 Rn Rd` (the same group as
        // MOVI and the narrowing shifts; `immh` is zero only for MOVI, and the
        // narrowing opcodes are handled above). Scalar: `01 U 111110 …`, which
        // differs only in bit28 and always works on one 64-bit lane —
        // `ushr d30, d31, #32` = 0x7f6007fe.
        let scalar_shift = ((insn >> 30) & 0b11) == 0b01 && ((insn >> 23) & 0x3F) == 0b111110;
        let vector_shift = ((insn >> 31) & 1) == 0 && ((insn >> 23) & 0x3F) == 0b011110;
        if (vector_shift || scalar_shift) && ((insn >> 10) & 1) == 1 && ((insn >> 19) & 0xF) != 0 {
            let opcode = (insn >> 11) & 0x1F;
            if !matches!(opcode, 0b10000 | 0b10001 | 0b10010 | 0b10011) {
                let q = vector_shift && (insn >> 30) & 1 == 1;
                let u = (insn >> 29) & 1 == 1;
                let immh = (insn >> 19) & 0xF;
                let imm = ((insn >> 16) & 0x7F) as u32;
                let rd = (insn & 0x1F) as u8;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let esize = match immh {
                    0b0001 => 8,
                    0b0010 | 0b0011 => 16,
                    0b0100..=0b0111 => 32,
                    _ => 64,
                };
                return self
                    .simd_shift_imm(rd, rn, q, u, opcode, esize, imm)
                    .map(|()| true);
            }
        }

        // MOVI/MVNI (modified immediate): bits[28:23] == 011110, bits[22:19]==0.
        // The 8-bit immediate is NOT contiguous: `abcdefgh` sits at bits 18:16
        // (a:b:c) and 9:5 (d:e:f:g:h), with bits 15:12 = cmode, bit 29 = op
        // (0 = MOVI, 1 = MVNI/bitwise). Cross-checked against QEMU.
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
            // q=0 writes only the low 64 bits (upper half cleared).
            self.vregs[rd as usize] = if q == 1 {
                imm64 as u128 | ((imm64 as u128) << 64)
            } else {
                imm64 as u128
            };
            return Ok(true);
        }

        // ---- permute (ZIP/UZP/TRN) ----
        // bit31=0, q=bit30, bits[29:24]=001110, bit21=0, opcode in bits[15:10]
        // (UZP1/TRN1/ZIP1 = 000110/001010/001110, UZP2/TRN2/ZIP2 = 010110/
        // 011010/011110). The copy-group guard above must not swallow these.
        // bit29 has to be 0: with it set the same bits are EXT, and
        // `ext v3.8b, v4.8b, v5.8b, #3` was being executed as `uzp1`.
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

        // ---- three different (widening / narrowing) ----
        // `0 Q U 01110 size 1 Rm opcode(4) 00 Rn Rd`: the results are twice
        // (or half) the source width. Distinguished from three-same by
        // bits[11:10] = 00.
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
            // PMULL/PMULL2 is the one form with a 64-bit source element, so it
            // is decoded before the size check the rest of the group needs.
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
            // The `2` variants (Q=1) read the top half of their narrow sources.
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
                // ADDHN / SUBHN (and the rounding variants): both operands are
                // already wide and the result is narrow, into the Q-selected
                // half of Vd.
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
                // The W forms take Vn at the destination width already.
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

        // ---- by-element multiplies (scalar x indexed element) ----
        // `01 U 11111 size L M Rm opcode(4) H 0 Rn Rd`: one lane of Vn times
        // one selected lane of Vm, with the result written as a *scalar* --
        // the bottom element, everything above it zeroed. The vector form
        // below is the same arithmetic across every lane, and differs only in
        // bits[28:24], 01111 against 11111.
        //
        // `fmul s3, s4, v3.s[0]` = 0x5f839083.
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
            // The index is spread over H:L:M, exactly as in the vector form: a
            // halfword element can only come from the low 16 vector registers,
            // because M is part of the index rather than of Rm.
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

            // Saturate to a signed field of `bits`, which every form here but
            // the floating-point four ends with.
            let sat = |value: i128, bits: u32| -> u64 {
                let max = (1i128 << (bits - 1)) - 1;
                let min = -(1i128 << (bits - 1));
                (value.clamp(min, max) as u64) & (elem_mask(bits) as u64)
            };
            // `2 * a * b`, the doubling every saturating form in this group is
            // built on. Only the most negative input squared can overflow it.
            let doubled = signed(a, esize) * signed(elem, esize) * 2;
            // ...and its high half, rounded or truncated back to the source
            // width.
            let high_half = |rounding: bool| -> i128 {
                let product = if rounding {
                    doubled + (1i128 << (esize - 1))
                } else {
                    doubled
                };
                product >> esize
            };

            let (value, width) = match key {
                // FMLA / FMLS / FMUL / FMULX. Half precision is out of scope
                // here for the same reason it is in the vector form.
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
                        // Fused at the source width: FMLA rounds once, so
                        // widening to f64 and back would round twice.
                        let r = match key {
                            0x01 | 0x05 => x.mul_add(y, acc),
                            0x19 => fmulx(f64::from(x), f64::from(y)) as f32,
                            _ => x * y,
                        };
                        u64::from(r.to_bits())
                    };
                    (bits, esize)
                }
                // SQDMULL: the doubled product, kept at twice the width.
                0x0b => (sat(doubled, esize * 2), esize * 2),
                // SQDMLAL / SQDMLSL: the same, accumulated into the wide
                // destination. Both saturations are real -- the product is
                // clamped before the accumulate and the sum after it.
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
                // SQDMULH / SQRDMULH: the doubled product's high half, the
                // second rounded rather than truncated.
                0x0c | 0x0d => (sat(high_half(key == 0x0d), esize), esize),
                // SQRDMLAH / SQRDMLSH: SQRDMULH accumulated into Rd.
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
            // A scalar destination: the result at the bottom, everything above
            // it zeroed.
            self.vregs[rd] = u128::from(value) & elem_mask(width);
            return Ok(true);
        }

        // ---- by-element multiplies (vector x indexed element) ----
        // `0 Q U 01111 size L M Rm opcode(4) H 0 Rn Rd`: every lane of Vn times
        // one selected lane of Vm. Shares bits[28:24] with MOVI and the
        // immediate shifts, which all have bit10 = 1.
        // `smull2 v19.4s, v18.8h, v0.h[2]` = 0x4f60a253.
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
            // The index is spread over H:L:M, and a halfword element can only
            // come from the low 16 vector registers (M is part of the index).
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
                // SMULL/UMULL and the accumulating forms: `Q` selects which half
                // of Vn is read, and the destination lanes are twice as wide.
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
            // Same-width forms.
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

    /// The rest of the AdvSIMD decode, split out so the by-element group above
    /// can `return` without nesting everything below it.
    fn try_simd_rest(&mut self, insn: u32) -> Result<bool> {
        // ---- EXT (vector extract) ----
        // `0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd`: take `datasize` bits out of the
        // concatenation Vm:Vn starting `imm4` bytes in. Shares bits[28:24] with
        // three-same, but has bit10 = 0.
        // `ext v31.16b, v31.16b, v31.16b, #8` = 0x6e1f43ff.
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

        // ---- two-register misc (Advanced SIMD) ----
        // `0 Q U 01110 size 10000 opcode(5) 10 Rn Rd`: the one-operand vector
        // ops — REV/CLS/CLZ/CNT/NOT/RBIT/ABS/NEG, the compares against zero,
        // the narrowing and lengthening moves, and the whole FP rounding and
        // integer<->float convert set. `scvtf v28.4s, v31.4s` = 0x4e21dbfc.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 24) & 0x1F) == 0b01110
            && ((insn >> 17) & 0x1F) == 0b10000
            && ((insn >> 10) & 0b11) == 0b10
        {
            return self.simd_two_reg_misc(insn, false);
        }
        // Scalar FP three-same: `01 U 11110 sz 1 Rm opcode(5) 1 Rn Rd` — one
        // lane of the vector group above. `fabd d31, d0, d31` = 0x7effd41f.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 1) == 1
            && ((insn >> 11) & 0x1F) >= 0b11000
        {
            return self.simd_fp_three_same(insn, true);
        }
        // Scalar integer three-same, the variable shifts: same encoding, an
        // opcode below the FP ones. SSHL and SRSHL are doubleword-only; the
        // saturating pair carries its own element size because saturation
        // needs to know the width.
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
        // The same group's scalar forms: `01 U 11110 size 10000 opcode(5) 10`.
        // One lane, and the rest of the register is zeroed.
        // `ucvtf s13, s13` = 0x7e21d9ad.
        if ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 24) & 0x1F) == 0b11110
            && ((insn >> 17) & 0x1F) == 0b10000
            && ((insn >> 10) & 0b11) == 0b10
        {
            return self.simd_two_reg_misc(insn, true);
        }

        // ---- integer three-same / compare / logical (Advanced SIMD) ----
        // bit31=0 with bits[29:24]=001110 (signed group, bit29=0) or 011110
        // (unsigned group, bit29=1). The opcode is in bits[15:11] with
        // bit10=1; the only bit10=0 form handled here is CMEQ #0.
        let grp = (insn >> 24) & 0x1F;
        // Vector three-same always has bits[28:24] == 01110 (bit28=0);
        // bits[28:24] == 11110 is the scalar-FP group, handled by try_fp.
        // Copy group (DUP/INS/UMOV/SMOV, 0{q}{op} 0111 0000): q (bit30) and
        // **op (bit29)** are both free, and bit20 is part of imm5 (so it may
        // be set for 64-bit lanes). What separates the group from three-same
        // is bit21 == 0, not bit29 — matching on bits[29:21] here excluded
        // every `op == 1` encoding, i.e. the whole of INS (element), which
        // then fell through to the three-same decoder and was executed as an
        // unrelated arithmetic op. See [`Cpu::try_simd_copy`].
        let copy_group = ((insn >> 21) & 0xFF) == 0b01110000 && ((insn >> 31) & 1) == 0;
        if ((insn >> 31) & 1) == 0 && grp == 0b01110 && !copy_group {
            // (copy_group == the DUP/MOV/INS encodings, which also live in the
            // 0x4e group with bits[23:21] == 000 and are handled below.)
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
            // AdvSIMD across lanes: `0 Q U 01110 size 11000 opcode 10 Rn Rd`
            // — a horizontal reduce across a vector into a single scalar
            // lane. Shares bits[28:24] with three-same, but bits[21:17] are
            // the fixed group selector 11000 rather than a free Rm, and
            // bit10 = 0 where three-same always has bit10 = 1.
            // `smaxv s28, v28.4s` = 0x4eb0ab9c.
            if b10 == 0 && ((insn >> 17) & 0x1F) == 0b11000 {
                return self.simd_across_lanes(insn);
            }
            // Opcodes from 0b11000 up in the three-same group are the FP ops,
            // where bits[23:22] are `a`:`sz` rather than an element size.
            if b10 == 1 && ((insn >> 21) & 1) == 1 && op >= 0b11000 {
                return self.simd_fp_three_same(insn, false);
            }
            if b10 == 1 {
                match op {
                    0b00000 => {
                        // SHADD (signed group) / UHADD (unsigned group):
                        // halving add, (a+b) >> 1.
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
                    // The variable-shift family. The opcode's low two bits
                    // are the saturating and rounding flags: SSHL/USHL,
                    // SQSHL/UQSHL, SRSHL/URSHL, SQRSHL/UQRSHL.
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
                        // MUL: lanewise multiply (PMUL, the U=1 form, is
                        // polynomial and only defined for 8-bit lanes).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| a.wrapping_mul(b));
                        return Ok(true);
                    }
                    0b10110 => {
                        // SQDMULH (signed group) / SQRDMULH (unsigned group):
                        // the doubled high half of the product, saturated.
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
                        // Bitwise logicals (the selector lives in bits[23:21];
                        // it doubles as `sz`, so no sz guard here).
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
                            // The insert/select trio differ only in which
                            // register is the mask: BSL selects with Vd, BIT
                            // and BIF with Vm (BIF taking Vn where the mask bit
                            // is clear). Getting the mask wrong made newlib's
                            // vectorised `strchr` miss the ':' in
                            // "romfs:/assets.zip".
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

        // ---- scalar copy: DUP (element) ----
        // `01 0 11110000 imm5 0 0000 1 Rn Rd`. The scalar copy group holds
        // exactly one instruction -- lifting one lane of a vector into a
        // scalar register, which is what `mov s1, v0.s[1]` assembles to -- and
        // it differs from the vector copy group below only in bits[28:21],
        // 1111 0000 against 0111 0000. So it has to be matched before that
        // check rejects it.
        //
        // Unlike the vector `DUP` further down, the destination is a *scalar*:
        // the lane is written at the bottom and the rest of the register is
        // zeroed, rather than the lane being replicated across it.
        // `dup s1, v0.s[1]` = 0x5e0c0401.
        if ((insn >> 21) & 0xFF) == 0b11110000
            && ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 10) & 0b111111) == 0b000001
        {
            let imm5 = (insn >> 16) & 0x1F;
            let lsb = imm5.trailing_zeros();
            // imm5 == 0, and any imm5 whose lowest set bit is above bit 3, is
            // reserved rather than a 128-bit element.
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

        // ---- copy / element moves, and table lookup ----
        // bits[28:21] == 0111 0000 (bit21 = 0 is what separates this whole
        // space from three-same/three-different/two-reg-misc); bit30 is Q and
        // bit29 is `op`, both free. EXT and the ZIP/UZP/TRN permutes share the
        // bit21 = 0 space but are matched earlier in `try_simd`/`try_simd_rest`,
        // so they never reach here.
        if ((insn >> 21) & 0xFF) != 0b01110000 || ((insn >> 31) & 1) != 0 {
            return Ok(false);
        }
        let op = (insn >> 29) & 1;

        // TBL / TBX: `0 Q 001110 00 0 Rm 0 len op 00 Rn Rd`. Rebuild a vector
        // by picking bytes out of a table held in `len+1` consecutive vector
        // registers, one index per byte of `Vm` — how a compiler spells an
        // arbitrary byte shuffle when no fixed permute (ZIP/UZP/TRN/EXT)
        // matches. It shares bits[28:21] with the copy group below, so it has
        // to be split off first: the copy encodings all set bit10, where
        // table lookup has bit15 = 0 and bits[11:10] = 00. Table lookup has no
        // `op` bit of its own — bits[29:24] are fixed at 001110 — so it is
        // only ever the `op == 0` half.
        // `tbl v31.16b, {v29.16b}, v28.16b` = 0x4e1c03bf.
        if op == 0 && ((insn >> 15) & 1) == 0 && ((insn >> 10) & 0b11) == 0 {
            let q = (insn >> 30) & 1 == 1;
            let rd = (insn & 0x1F) as usize;
            let rn = ((insn >> 5) & 0x1F) as usize;
            let rm = ((insn >> 16) & 0x1F) as usize;
            let len = ((insn >> 13) & 0b11) as usize + 1;
            // The only difference between the two: for an index past the end
            // of the table TBL writes zero, TBX leaves the destination byte
            // alone (so a second lookup can fill in what the first missed).
            let keep_on_miss = (insn >> 12) & 1 == 1;
            let indices = self.vregs[rm].to_le_bytes();
            let mut out = self.vregs[rd].to_le_bytes();
            let lanes = if q { 16 } else { 8 };
            for (i, slot) in out.iter_mut().enumerate().take(lanes) {
                let idx = indices[i] as usize;
                if idx < len * 16 {
                    // The table wraps past v31, so {v30, v31, v0, v1} is a
                    // legal four-register table.
                    *slot = self.vregs[(rn + idx / 16) % 32].to_le_bytes()[idx % 16];
                } else if !keep_on_miss {
                    *slot = 0;
                }
            }
            // The 8-byte form zeroes the top half like every other AdvSIMD
            // `Q == 0` encoding, including TBX.
            out[lanes..].fill(0);
            self.vregs[rd] = u128::from_le_bytes(out);
            return Ok(true);
        }

        let q = (insn >> 30) & 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let imm5 = (insn >> 16) & 0x1F;

        if op == 1 {
            // INS <Vd>.<Ts>[<index1>], <Vn>.<Ts>[<index2>] — move one lane to
            // another lane, the only `op == 1` encoding in the copy group.
            // `imm5` gives the element size and the *destination* index the
            // same way UMOV/SMOV/INS-general do; `imm4` gives the *source*
            // index, shifted down by the same `lsb` (imm4<3:size>).
            //
            // This is how a compiler assembles a short string in a register
            // without touching memory: libnx's `smEncodeName` builds the
            // 8-byte `SmServiceName` with one `ldr b<n>, [str, #i]` per
            // character and then a chain of `ins v31.b[i], v<n>.b[0]`.
            // Without this, every such name reached `sm::GetService` as
            // eight zero bytes — Checkpoint asked for `ns:am2`, got a session
            // bound to "", and panicked once it used it.
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
            // INS leaves every other lane of Vd alone — including the top
            // half, which is why there is no `Q == 0` zeroing here.
            self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
            return Ok(true);
        }

        match (insn >> 10) & 0b111111 {
            0b000111 => {
                // INS <Vd>.<T>[<index>], <Rn> — insert a GPR lane. Same
                // imm5 → (esize, index) scheme as UMOV/SMOV: esize = 8<<ctz,
                // index = imm5 >> (ctz+1).
                let lsb = imm5.trailing_zeros();
                if lsb > 3 {
                    return Ok(false);
                }
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = (index as u32) * esize;
                let mask = (1u128 << esize) - 1;
                let v = self.vregs[rd as usize];
                let val = (self.read_zr(rn) as u128) & mask;
                self.vregs[rd as usize] = (v & !(mask << shift)) | (val << shift);
                Ok(true)
            }
            0b000011 if imm5 != 0 => {
                // DUP <Vd>.<T>, <Rn>: element size is `8 << ctz(imm5)` (imm5 =
                // 1/2/4/8 for 8/16/32/64-bit; the low bits hold the element
                // index, which the general-register form ignores).
                let esize = 8u32 << imm5.trailing_zeros();
                let elements = if q == 1 { 128 / esize } else { 64 / esize };
                let val = (self.read_zr(rn) as u128) & ((1u128 << esize) - 1);
                let mut v: u128 = 0;
                for i in 0..elements {
                    v |= (val as u128) << (i as u32 * esize);
                }
                self.vregs[rd as usize] = v;
                Ok(true)
            }
            0b000001 if imm5 != 0 => {
                // DUP <Vd>.<T>, <Vn>.<Ts>[<index>]: replicate one lane of a
                // vector, rather than a GPR. `dup v1.4s, v0.s[0]` = 0x4e040401.
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
                let shift = (index as u32) * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                self.write_zr(rd, val as u64);
                Ok(true)
            }
            0b001011 => {
                // SMOV <Xd/Wd>, <Vn>.B/H/S[<index>] — extract a lane,
                // sign-extended (8/16-bit → Wd, 32-bit → Xd).
                let lsb = imm5.trailing_zeros();
                let esize = 8u32 << lsb;
                let index = imm5 >> (lsb + 1);
                let shift = (index as u32) * esize;
                let val = (self.vregs[rn as usize] >> shift) & ((1u128 << esize) - 1);
                let val = sext_u64(val as u64, esize);
                self.write_zr(rd, val);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    // ---------------- scalar floating point ----------------
    //
    // The scalar FP subset hbmenu's UI/drawing code needs: FMOV, the common
    // arithmetic (FADD/FSUB/FMUL/FDIV/FNMUL/FMAX/FMIN/FMAXNM/FMINNM), the
    // unary ops (FABS/FNEG/FSQRT/FRINTx/FCVT between single and double),
    // fused multiply-add (FMADD/FMSUB/FNMADD/FNMSUB), compares (FCMP/
    // FCMPE/FCCMP), FCSEL, and the integer<->float conversions. NaN, infinity
    // and rounding come straight from Rust's IEEE f32/f64 (round-to-nearest,
    // the FPCR default); FP exception flags are not modelled.

    /// Element-wise SIMD binary op over `esize`-bit lanes (little-endian lane
    /// order), `q` selects 128-bit vs 64-bit registers.
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

    /// [`Cpu::simd_elem`] over an explicit lane count, for the scalar forms.
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
    ///
    /// The three families place their results differently, and conflating them
    /// scrambles a matrix transpose: TRN takes the even (or odd) elements of
    /// *both* operands and interleaves them, ZIP interleaves one half of each,
    /// and UZP packs every other element of Vn into the low half of the result
    /// and Vm's into the high half. `trn1` picking Vm's odd elements is what
    /// left hbmenu's NEON JPEG decoder (its icon) spinning.
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

    /// Lanewise SIMD op that also reads the destination lane, for the
    /// accumulating forms (MLA/MLS, SABA/UABA).
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

    /// Pairwise SIMD binary op (ADDP/SMAXP/UMAXP): the destination's first
    /// half pairs up Vn's lanes, the second half Vm's.
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
    /// Signed `a >= b` for `bits`-wide lanes.
    pub(super) fn sge(a: u64, b: u64, bits: u32) -> bool {
        let shift = 64 - bits;
        ((a << shift) as i64) >= ((b << shift) as i64)
    }

    /// AdvSIMD across lanes (integer forms): `0 Q U 01110 size 11000
    /// opcode(5) 10 Rn Rd`. A horizontal reduce over every lane of Vn into
    /// a single scalar written to Vd, zeroing the rest of the register —
    /// SADDLV/UADDLV, SMAXV/UMAXV, SMINV/UMINV and ADDV.
    fn simd_across_lanes(&mut self, insn: u32) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let size = (insn >> 22) & 0b11;
        let opcode = (insn >> 12) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
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
                // ADDV: sum of all lanes wrapped to the element size — unlike
                // SADDLV/UADDLV there is no widening, and U is always 0 (a
                // same-width wraparound sum doesn't care about signedness).
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

    /// AdvSIMD shift-by-immediate.
    ///
    /// The encoding packs the element size into `immh` and the shift amount
    /// into `immh:immb`: a right shift is `2*esize - imm`, a left shift is
    /// `imm - esize`. `opcode` selects the operation, `u` its signed/unsigned
    /// (or, for SHL, its insert) variant.
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

        // Widening left shift (SSHLL/USHLL, and SXTL/UXTL when the shift is 0):
        // the destination lanes are twice as wide, taken from one half of Vn.
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

    /// Lanewise unary op over `lanes` lanes of `esize` bits each, on the raw
    /// lane bits so the same helper serves the integer and floating-point
    /// forms. A scalar form is the same thing with one lane.
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

    /// AdvSIMD two-register misc: `0 Q U 01110 size 10000 opcode(5) 10 Rn Rd`.
    ///
    /// The FP forms are identified by `(U, size<1>, opcode)` together — e.g.
    /// opcode 11101 is SCVTF with `U=0, size<1>=0` but FRECPE with
    /// `U=0, size<1>=1` — and their element size is `size<0>` (0 = single,
    /// 1 = double). The integer forms use `size` as the element size instead.
    fn simd_two_reg_misc(&mut self, insn: u32, scalar: bool) -> Result<bool> {
        let q = (insn >> 30) & 1 == 1;
        let u = (insn >> 29) & 1;
        let size = (insn >> 22) & 0b11;
        let opcode = (insn >> 12) & 0x1F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let esize = 8u32 << size;
        // A scalar form is one lane; the vector forms fill the register.
        let lanes = |esize: u32| {
            if scalar {
                1
            } else if q {
                128 / esize
            } else {
                64 / esize
            }
        };

        // Integer forms first: opcodes below 0b01100 plus the narrowing and
        // lengthening moves. The lane-shuffling and width-changing ones have no
        // scalar form.
        match (u, opcode) {
            // REV64 / REV32 / REV16: reverse groups of bytes within a
            // container of 64/32/16 bits.
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
                        // CLS counts the sign bits after the first one.
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
            // SADDLP / UADDLP and SADALP / UADALP: add adjacent lanes into
            // double-width lanes, accumulating into Vd for the ADALP forms.
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
            // XTN / SQXTN / UQXTN / SQXTUN: narrow to half-width lanes (Q
            // targets the high half). `size` is the destination width here.
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
            // FCVTL / FCVTN: half <-> single (size 00) and single <-> double
            // (size 01). Q selects which half of the narrow vector is used.
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
                    // Promoting to double before narrowing is exact, so the
                    // half is rounded once rather than once per step.
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

        // The FP forms: `(U, size<1>, opcode)` selects the operation and
        // `size<0>` the element width.
        let double = size & 1 == 1;
        if double && !q && !scalar {
            return Ok(false); // a single 64-bit lane isn't a vector form
        }
        let key = (u << 6) | ((size >> 1) << 5) | opcode;
        let esize = if double { 64 } else { 32 };
        // Comparisons against zero produce a lane mask rather than a float.
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
            // FSQRT, the FRINTx family and the reciprocal estimates. FRECPE
            // and FRSQRTE are the architecture's 8-bit estimates, in
            // `fp::recip_estimate_bits` / `fp::rsqrt_estimate_bits`: they used
            // to be a division and a reciprocal square root here, which is a
            // different number in every low bit.
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

    /// AdvSIMD three-same, floating-point: `0 Q U 01110 a sz 1 Rm opcode(5) 1
    /// Rn Rd` for opcodes from 0b11000 up. `a` (bit23) picks between the pairs
    /// that share an opcode (FADD/FSUB, FMAX/FMIN, ...) and `sz` (bit22) the
    /// element width.
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
        // Arithmetic and the min/max family, over the raw lane bits.
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

    /// Shift-right-and-narrow a vector (SHRN/RSHRN and the saturating forms).
    /// Every `2*dest_esize`-bit lane of `Vn` is shifted right by `shift`
    /// (optionally rounding) and narrowed to `dest_esize` bits; `Q=1` targets
    /// the high half (`SHRN2`), `Q=0` the low half.
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
