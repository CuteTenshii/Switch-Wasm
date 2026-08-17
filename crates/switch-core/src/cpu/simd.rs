//! NEON/AdvSIMD: the vector instruction subset compiled homebrew uses
//! (three-same integer ops, logicals, permutes, compares, shifts and the
//! element moves libnx's string and memory routines are built from).

use super::bits::*;
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
                let size = (insn >> 22) & 1;
                let dest_esize = 8u32 << size;
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
                    self.simd_shrn(rd, rn, q, dest_esize, shift, rounding, signed_src, to_unsigned, saturating);
                    return Ok(true);
                }
            }
        }

        // ---- shift by immediate (SSHR/USHR/SHL/SLI/SRI/SSHLL/...) ----
        // Same group as MOVI and the narrowing shifts (bits[28:23] == 011110,
        // bit10 == 1); `immh` (bits[22:19]) is zero only for MOVI, and the
        // narrowing opcodes are handled above.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 23) & 0x3F) == 0b011110
            && ((insn >> 10) & 1) == 1
            && ((insn >> 19) & 0xF) != 0
        {
            let opcode = (insn >> 11) & 0x1F;
            if !matches!(opcode, 0b10000 | 0b10001 | 0b10010 | 0b10011) {
                let q = (insn >> 30) & 1 == 1;
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
                return self.simd_shift_imm(rd, rn, q, u, opcode, esize, imm).map(|()| true);
            }
        }

        // MOVI/MVNI (modified immediate): bits[28:23] == 011110, bits[22:19]==0.
        // The 8-bit immediate is NOT contiguous: `abcdefgh` sits at bits 18:16
        // (a:b:c) and 9:5 (d:e:f:g:h), with bits 15:12 = cmode, bit 29 = op
        // (0 = MOVI, 1 = MVNI/bitwise). Cross-checked against QEMU.
        if ((insn >> 31) & 1) == 0 && ((insn >> 23) & 0x3F) == 0b011110 && ((insn >> 19) & 0b1111) == 0b0000 {
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
        let perm = (insn >> 10) & 0b111111;
        if ((insn >> 31) & 1) == 0
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

        // ---- integer three-same / compare / logical (Advanced SIMD) ----
        // bit31=0 with bits[29:24]=001110 (signed group, bit29=0) or 011110
        // (unsigned group, bit29=1). The opcode is in bits[15:11] with
        // bit10=1; the only bit10=0 form handled here is CMEQ #0.
        let grp = (insn >> 24) & 0x1F;
        // Vector three-same always has bits[28:24] == 01110 (bit28=0);
        // bits[28:24] == 11110 is the scalar-FP group, handled by try_fp.
        // Copy group (DUP/INS/UMOV/SMOV, 0{q}00 1110 000): q (bit30) is free,
        // and bit20 is part of imm5 (so it may be set for 64-bit lanes).
        let copy_group = ((insn >> 21) & 0x1FF) == 0b001110000 && ((insn >> 31) & 1) == 0;
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
                    0b01000 => {
                        // SSHL / USHL: shift left by register (negative shift
                        // amounts shift right).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            shift_by_reg(a, b, esize, u != 0)
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
                                if a & b != 0 { u64::MAX } else { 0 }
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
                            if ge { u64::MAX } else { 0 }
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
                            if gt { u64::MAX } else { 0 }
                        });
                        return Ok(true);
                    }
                    0b01100 => {
                        // SMAX / UMAX.
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| {
                            if u == 0 {
                                if Self::sge(a, b, esize) { a } else { b }
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
                                if Self::sge(a, b, esize) { b } else { a }
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
                            if u == 0 { d.wrapping_add(product) } else { d.wrapping_sub(product) }
                        });
                        return Ok(true);
                    }
                    0b10011 if u == 0 => {
                        // MUL: lanewise multiply (PMUL, the U=1 form, is
                        // polynomial and only defined for 8-bit lanes).
                        self.simd_elem(rd, rn, rm, q, esize, |a, b| a.wrapping_mul(b));
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
                                if Self::sge(a, b, esize) { a } else { b }
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
                                if Self::sge(a, b, esize) { b } else { a }
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
                            (0, 0b001) => a & b,        // AND
                            (0, 0b011) => a & !b,        // BIC
                            (0, 0b101) => a | b,         // ORR
                            (0, 0b111) => a | !b,        // ORN
                            (1, 0b001) => a ^ b,         // EOR
                            (1, 0b011) => (d & a) | (b & !a), // BSL: mask = Vn
                            (1, 0b101) => (b & a) | (d & !a), // BIT: mask = Vn
                            (1, 0b111) => (b & d) | (a & !d), // BIF: mask = Vd
                            _ => return Ok(false),
                        };
                        self.vregs[rd as usize] = r & full;
                        return Ok(true);
                    }
                    _ => {}
                }
            } else if op == 0b10011 && rm == 0 && u == 0 {
                // CMEQ <Vd>.<T>, <Vn>.<T>, #0 (compare against zero).
                self.simd_elem(rd, rn, rm, q, esize, |a, _| {
                    if a == 0 { u64::MAX } else { 0 }
                });
                return Ok(true);
            }
            return Err(Error::Cpu(format!(
                "unimplemented SIMD three-same u={} op={:#b} sz={} at {:#x}",
                u, op, sz, self.pc
            )));
        }

        // ---- copy / element moves ----
        if ((insn >> 21) & 0x1FF) != 0b001110000 || ((insn >> 31) & 1) != 0 {
            return Ok(false);
        }
        let q = (insn >> 30) & 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let imm5 = (insn >> 16) & 0x1F;
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
    pub(super) fn simd_elem<F: Fn(u64, u64) -> u64>(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, f: F) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let mut out: u128 = 0;
        for i in 0..lanes {
            let av = ((a >> (esize * i)) & mask) as u64;
            let bv = ((b >> (esize * i)) & mask) as u64;
            out |= (f(av, bv) as u128 & mask) << (esize * i);
        }
        self.vregs[rd as usize] = out;
    }

    /// ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2 over `esize`-bit lanes.
    pub(super) fn simd_permute(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, op: u32) {
        let lanes = if q { 128 / esize } else { 64 / esize };
        let half = lanes / 2;
        let mask = (1u128 << esize) - 1;
        let a = self.vregs[rn as usize];
        let b = self.vregs[rm as usize];
        let get = |r: u128, i: u32| ((r >> (esize * i)) & mask) as u64;
        let mut out: u128 = 0;
        for i in 0..half {
            let (n0, m0) = match op {
                // ZIP1: interleave the low halves; ZIP2: the high halves.
                0b001110 => (get(a, i), get(b, i)),
                0b011110 => (get(a, half + i), get(b, half + i)),
                // UZP1: even lanes; UZP2: odd lanes.
                0b000110 => (get(a, 2 * i), get(b, 2 * i)),
                0b010110 => (get(a, 2 * i + 1), get(b, 2 * i + 1)),
                // TRN1/TRN2: transpose even/odd lanes.
                0b001010 => (get(a, 2 * i), get(b, 2 * i + 1)),
                _ => (get(a, 2 * i + 1), get(b, 2 * i)),
            };
            out |= ((n0 as u128) & mask) << (esize * 2 * i);
            out |= ((m0 as u128) & mask) << (esize * (2 * i + 1));
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
    pub(super) fn simd_pairwise<F: Fn(u64, u64) -> u64>(&mut self, rd: u8, rn: u8, rm: u8, q: bool, esize: u32, f: F) {
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
        let mask: u128 = if esize >= 128 { u128::MAX } else { (1u128 << esize) - 1 };
        let src = self.vregs[rn as usize];
        let dst = self.vregs[rd as usize];

        // Widening left shift (SSHLL/USHLL, and SXTL/UXTL when the shift is 0):
        // the destination lanes are twice as wide, taken from one half of Vn.
        if opcode == 0b10100 {
            let shift = imm - esize;
            let wide = 2 * esize;
            let wide_mask: u128 = if wide >= 128 { u128::MAX } else { (1u128 << wide) - 1 };
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
                    let round = if rounding && shift > 0 { 1u64 << (shift - 1) } else { 0 };
                    let shifted = if u {
                        raw.wrapping_add(round) >> shift.min(63)
                    } else {
                        let signed = sext_u64(raw, esize) as i64;
                        (signed.wrapping_add(round as i64) >> shift.min(63)) as u64
                    };
                    if opcode & 0b00010 != 0 { old.wrapping_add(shifted) } else { shifted }
                }
                // SRI: shift right and insert, keeping the high bits of Vd.
                0b01000 => {
                    let keep = if shift >= esize { mask as u64 } else { !(mask as u64 >> shift) };
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
        let src_mask = (1u128 << src_esize) - 1;
        let dest_mask = (1u128 << dest_esize) - 1;
        let src = self.vregs[rn as usize];
        let round_add = if rounding { 1i64 << (shift - 1) } else { 0 };
        let mut narrowed = [0u64; 16];
        for i in 0..src_elements {
            let raw = ((src >> (src_esize * i)) & src_mask) as u64;
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
            out |= (narrowed[i as usize] as u128) << (dest_esize * i);
        }
        if q {
            self.vregs[rd as usize] = (self.vregs[rd as usize] & ((1u128 << 64) - 1)) | (out << 64);
        } else {
            self.vregs[rd as usize] = out & ((1u128 << 64) - 1);
        }
    }
}
