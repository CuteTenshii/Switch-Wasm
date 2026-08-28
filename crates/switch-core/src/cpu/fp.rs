//! Scalar floating point: FMOV/arithmetic/compare/convert on the S and D
//! views of the vector registers, using the host's IEEE-754 semantics.

use super::bits::*;
use super::Cpu;

/// Which of the scalar floating-point forms an encoding is. See
/// [`Cpu::fp_form`], which is the only thing that produces one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FpForm {
    /// `FMOV` (immediate).
    MovImm,
    /// The 1-source group: `FCVT`, `FMOV` register-to-register, `FABS`, ...
    OneSource,
    /// `FMOV` between a general-purpose register and a vector lane.
    MovReg,
    /// Conversion between floating point and integer.
    IntConv,
    /// Conversion between floating point and fixed point.
    FixedConv,
    /// The scalar integer compare-to-zero forms.
    CmpZero,
    /// The 3-source fused multiply-adds.
    ThreeSource,
    /// The 2-source arithmetic, the compares and the conditional forms.
    DataProc,
    /// Nothing here claims the encoding.
    None,
}
use crate::Result;

impl Cpu {
    #[inline]
    pub(super) fn fp_get_f32(&self, r: u8) -> f32 {
        f32::from_bits(self.vregs[r as usize] as u32)
    }

    #[inline]
    pub(super) fn fp_get_f64(&self, r: u8) -> f64 {
        f64::from_bits(self.vregs[r as usize] as u64)
    }

    /// Write Sn. A scalar FP write is a 32-bit write to the vector register, so
    /// it zeroes the other 96 bits rather than leaving whatever was there.
    #[inline]
    pub(super) fn fp_set_f32(&mut self, r: u8, v: f32) {
        self.vregs[r as usize] = u128::from(v.to_bits());
    }

    #[inline]
    pub(super) fn fp_set_f64(&mut self, r: u8, v: f64) {
        self.vregs[r as usize] = v.to_bits() as u128;
    }

    /// Which scalar floating-point form an encoding is, and so which handler
    /// below owns it — decided from the encoding alone, with no side effects.
    ///
    /// The forms are tested in the order [`Cpu::try_fp`] used to test them
    /// inline, because that order is load-bearing: the fixed-point conversions
    /// have to be recognised before the `sf`-inclusive guard that follows
    /// them, and the 3-source group before the `00011110` space.
    ///
    /// Separating classification from execution is what lets the block
    /// translator settle the form once ([`super::jit::Op::Fp`]) instead of
    /// walking eight guards on every execution — `scvtf`, `fcvt`, `fadd` and
    /// `fcmpe` sit in four different ones, and together they are most of the
    /// floating point hbmenu runs.
    pub(super) fn fp_form(insn: u32) -> FpForm {
        // FMOV (immediate): bits[31:24] = 00011110, bit21 = 1,
        // bits[12:10] = 100, bits[9:5] = 0, imm8 = bits[20:13], type in
        // bits[23:22] (00 = S, 01 = D). The value is VFPExpandImm() —
        // `fmov s0, #1.0` = 0x1E2E1002 (sdl-hello's float env-var helper).
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0b111) == 0b100
            && ((insn >> 5) & 0x1F) == 0 {
            return FpForm::MovImm;
        }
        // Scalar FP 1-source: bits[31:24] = 00011110, bit21 = 1,
        // bits[14:10] = 10000, opcode in bits[20:15], `type` in bits[23:22]
        // (00 = S, 01 = D). Note the opcode's low bit lands in bits[15], so
        // matching on bits[15:10] as a unit misses half the group — which is
        // how `fmov s0, s15` (opcode 0) came out unimplemented.
        // `fcvt s0, d0` = 0x1E624000, `fmov s0, s15` = 0x1E2041E0.
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0x1F) == 0b10000 {
            return FpForm::OneSource;
        }
        // FMOV (register): move between GPR and a vector lane. bits[30:24] =
        // 0011110, bits[15:10] = 000000, bits[21:16] select direction/size.
        if ((insn >> 24) & 0x7F) == 0b0011110
            && ((insn >> 10) & 0x3F) == 0
            && matches!((insn >> 16) & 0x3F, 0b100110 | 0b100111) {
            return FpForm::MovReg;
        }
        // Conversion between floating-point and integer: bits[30:24] =
        // 0011110 (sf at bit31), `type` in bits[23:22], bit21 = 1,
        // bits[15:10] = 0 (non-zero there is the fixed-point scale, a separate
        // class). The operation is `rmode` (bits[20:19]) and `opcode`
        // (bits[18:16]) — treating bits[21:16] as one 6-bit field folds the
        // fixed bit21 into the value, which made `ucvtf d0, x1` decode as
        // FCVTMU and write x0 instead of d0 (NX-Shell then dereferenced the
        // clobbered pointer).
        if ((insn >> 24) & 0x7F) == 0b0011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0x3F) == 0 {
            return FpForm::IntConv;
        }
        // Floating-point <-> fixed-point conversion: bits[30:24] = 0011110
        // with bit21 = 0. Same rmode/opcode split as the integer conversions,
        // with the binary point `64 - scale` bits in from bits[15:10]. `sf` is
        // bit31, so this has to be matched before the bit31-inclusive guard
        // below (which is only correct for the forms that have no `sf`).
        if ((insn >> 24) & 0x7F) == 0b0011110 && ((insn >> 21) & 1) == 0 {
            return FpForm::FixedConv;
        }
        // Scalar integer compare-to-zero: CMGE/CMGT/CMLE/CMLT <Dd>, <Dn>, #0.
        // bits[31:30] = 01 (D), bit29 = U, bits[28:25] = 1110,
        // bits[24:21] = 0111, bits[20:16] = 00000 (the zero operand),
        // op = bits[15:10]. The result is an all-ones/all-zeros mask (used as
        // a predicate by NX-Shell: `cmge d31, d31, #0` then `fmov x2, d31`).
        if ((insn >> 31) & 1) == 0
            && ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 25) & 0b1111) == 0b1111
            && ((insn >> 21) & 0xF) == 0b0111
            && ((insn >> 16) & 0x1F) == 0 {
            return FpForm::CmpZero;
        }
        // 3-source fused ops: bits[31:24] = 00011111, `type` in bits[23:22],
        // o1 = bit21, o0 = bit15, Ra in bits[14:10]. This group has its own
        // top byte, so it has to be matched before the 00011110 space below.
        // `fmadd d0, d31, d26, d0` = 0x1F5A03E0.
        if ((insn >> 24) & 0xFF) == 0b00011111 {
            return FpForm::ThreeSource;
        }
        // Scalar FP data processing: bits[31:24] = 00011110 (single/double;
        // bit23 = 1 selects half precision, which is out of scope).
        if ((insn >> 24) & 0xFF) == 0b00011110 && ((insn >> 23) & 1) == 0 {
            return FpForm::DataProc;
        }
        FpForm::None
    }

    /// Run a scalar floating-point instruction whose form is already known.
    /// Returns whether the form's handler claimed it — a handler may still
    /// decline (half precision, an unallocated opcode), in which case nothing
    /// else here gets a look, exactly as the guard chain behaved.
    pub(super) fn run_fp(&mut self, form: FpForm, insn: u32) -> Result<bool> {
        match form {
            FpForm::MovImm => self.fp_mov_imm(insn),
            FpForm::OneSource => self.fp_one_source(insn),
            FpForm::MovReg => self.fp_mov_reg(insn),
            FpForm::IntConv => self.fp_int_conv(insn),
            FpForm::FixedConv => self.fp_fixed_conv(insn),
            FpForm::CmpZero => self.fp_int_cmp_zero(insn),
            FpForm::ThreeSource => self.fp_three_source(insn),
            FpForm::DataProc => self.fp_data_proc(insn),
            FpForm::None => Ok(false),
        }
    }

    pub(super) fn try_fp(&mut self, insn: u32) -> Result<bool> {
        self.run_fp(Cpu::fp_form(insn), insn)
    }

    pub(super) fn fp_mov_imm(&mut self, insn: u32) -> Result<bool> {
        let imm8 = ((insn >> 13) & 0xFF) as u8;
        let ftype = (insn >> 22) & 0b11;
        let rd = (insn & 0x1F) as u8;
        let sign = if (imm8 >> 7) & 1 == 1 { 0x8000u32 } else { 0 };
        return match ftype {
            0b00 => {
                let imm = (sign
                    | if (imm8 >> 6) & 1 == 1 { 0x3E00 } else { 0x4000 }
                    | (((imm8 & 0x3F) as u32) << 3))
                    << 16;
                self.fp_set_f32(rd, f32::from_bits(imm));
                Ok(true)
            }
            0b01 => {
                let imm = (sign as u64
                    | if (imm8 >> 6) & 1 == 1 { 0x3FC0 } else { 0x4000 }
                    | (imm8 & 0x3F) as u64)
                    << 48;
                self.fp_set_f64(rd, f64::from_bits(imm));
                Ok(true)
            }
            _ => Ok(false), // half precision: out of scope
        };
    }

    pub(super) fn fp_one_source(&mut self, insn: u32) -> Result<bool> {
        let op = (insn >> 15) & 0x3F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let ftype = (insn >> 22) & 0b11;
        // FCVT is the one form whose destination width differs from its
        // source, so it can't share the write-back below — and the only
        // scalar form that reaches half precision at all.
        let half = |v: &Self| f16_to_f32(v.vregs[rn as usize] as u16);
        match (op, ftype) {
            (0b000100, 0b01) => {
                self.fp_set_f32(rd, self.fp_get_f64(rn) as f32);
                return Ok(true);
            }
            (0b000100, 0b11) => {
                self.fp_set_f32(rd, half(self));
                return Ok(true);
            }
            (0b000101, 0b00) => {
                self.fp_set_f64(rd, f64::from(self.fp_get_f32(rn)));
                return Ok(true);
            }
            (0b000101, 0b11) => {
                self.fp_set_f64(rd, f64::from(half(self)));
                return Ok(true);
            }
            // FCVT Hd, Sn/Dn. A half is a 16-bit write, so it clears the
            // rest of the register like every other scalar FP write.
            // Singles go via a double, which is exact, so they round once.
            (0b000111, 0b00) => {
                let v = f64::from(self.fp_get_f32(rn));
                self.vregs[rd as usize] = u128::from(f64_to_f16(v));
                return Ok(true);
            }
            (0b000111, 0b01) => {
                self.vregs[rd as usize] = u128::from(f64_to_f16(self.fp_get_f64(rn)));
                return Ok(true);
            }
            _ => {}
        }
        let double = match ftype {
            0b00 => false,
            0b01 => true,
            // Half-precision *arithmetic* is ARMv8.2 and not on the A57.
            _ => return Ok(false),
        };
        if op == 0 {
            // FMOV Sd/Dd, Sn/Dn: a bit-exact copy, so it must not go
            // through a float conversion (that can canonicalize NaNs).
            let bits = self.vregs[rn as usize];
            self.vregs[rd as usize] = if double {
                u128::from(bits as u64)
            } else {
                u128::from(bits as u32)
            };
            return Ok(true);
        }
        // FABS/FNEG are bit operations on the sign, and single-precision
        // FSQRT/FRINTx round once — computing them in f64 and narrowing
        // would round twice.
        let mode = fpcr_rounding(self.fpcr);
        if op == 0b000011 && self.fp_sqrt_is_invalid(rn, double) {
            self.fpsr |= FPSR_IOC;
        }
        if double {
            let a = self.fp_get_f64(rn);
            let r = match op {
                0b000001 => f64::from_bits(a.to_bits() & !(1 << 63)),
                0b000010 => f64::from_bits(a.to_bits() ^ (1 << 63)),
                0b000011 => a.sqrt(),
                0b001000 => a.round_ties_even(), // FRINTN
                0b001001 => a.ceil(),            // FRINTP
                0b001010 => a.floor(),           // FRINTM
                0b001011 => a.trunc(),           // FRINTZ
                0b001100 => a.round(),           // FRINTA (ties away)
                // FRINTX/FRINTI are the two that round to whatever mode
                // FPCR currently selects.
                0b001110 | 0b001111 => round_to_integral(a, mode),
                _ => return Ok(false),
            };
            self.fp_set_f64(rd, r);
        } else {
            let a = self.fp_get_f32(rn);
            let r = match op {
                0b000001 => f32::from_bits(a.to_bits() & !(1 << 31)),
                0b000010 => f32::from_bits(a.to_bits() ^ (1 << 31)),
                0b000011 => a.sqrt(),
                0b001000 => a.round_ties_even(),
                0b001001 => a.ceil(),
                0b001010 => a.floor(),
                0b001011 => a.trunc(),
                0b001100 => a.round(),
                0b001110 | 0b001111 => round_to_integral(f64::from(a), mode) as f32,
                _ => return Ok(false),
            };
            self.fp_set_f32(rd, r);
        }
        return Ok(true);
    }

    pub(super) fn fp_mov_reg(&mut self, insn: u32) -> Result<bool> {
        let sel = (insn >> 16) & 0x3F;
        let double = ((insn >> 22) & 1) == 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        match sel {
            0b100110 => {
                // FMOV Xd/Wd, Dn/Sn — move the FP bit pattern to a GPR.
                let val = if double {
                    self.fp_get_f64(rn).to_bits()
                } else {
                    self.fp_get_f32(rn).to_bits() as u64
                };
                self.write_zr(rd, val);
            }
            0b100111 => {
                // FMOV Vd.D/S, Xn/Wn — move a GPR bit pattern to FP.
                if double {
                    self.fp_set_f64(rd, f64::from_bits(self.read_zr(rn)));
                } else {
                    self.fp_set_f32(rd, f32::from_bits(self.read_zr(rn) as u32));
                }
            }
            _ => return Ok(false),
        }
        return Ok(true);
    }

    pub(super) fn fp_int_conv(&mut self, insn: u32) -> Result<bool> {
        let sf = (insn >> 31) & 1;
        let ftype = (insn >> 22) & 0b11; // 00 = single, 01 = double
        if ftype > 0b01 {
            return Ok(false); // half precision: out of scope
        }
        let use_double = ftype == 0b01;
        let rmode = (insn >> 19) & 0b11;
        let opcode = (insn >> 16) & 0b111;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let wide = sf != 0;
        return match (rmode, opcode) {
            // SCVTF / UCVTF: integer → float. `sf` gives the source width,
            // `type` the destination's, and they are independent.
            (0b00, 0b010) | (0b00, 0b011) => {
                let signed = opcode == 0b010;
                let v = self.read_zr(rn);
                let f = match (signed, wide) {
                    (true, true) => v as i64 as f64,
                    (true, false) => f64::from(v as i32),
                    (false, true) => v as f64,
                    (false, false) => f64::from(v as u32),
                };
                if use_double {
                    self.fp_set_f64(rd, f);
                } else {
                    self.fp_set_f32(rd, f as f32);
                }
                Ok(true)
            }
            // FMOV between a GPR and an FP register is handled above.
            (0b00, 0b110) | (0b00, 0b111) => Ok(false),
            // Float → integer. `opcode` picks signed/unsigned and `rmode`
            // the rounding: 00 = nearest-even (FCVTNS/NU), 01 = +inf
            // (FCVTPS/PU), 10 = -inf (FCVTMS/MU), 11 = zero (FCVTZS/ZU).
            // rmode 00 with opcode 100/101 is FCVTAS/FCVTAU (ties away).
            (_, 0b000) | (_, 0b001) | (0b00, 0b100) | (0b00, 0b101) => {
                let signed = opcode & 1 == 0;
                let rounding = if opcode >= 0b100 {
                    Rounding::TiesAway
                } else {
                    match rmode {
                        0b00 => Rounding::TiesEven,
                        0b01 => Rounding::TowardPos,
                        0b10 => Rounding::TowardNeg,
                        _ => Rounding::TowardZero,
                    }
                };
                let f = if use_double {
                    self.fp_get_f64(rn)
                } else {
                    f64::from(self.fp_get_f32(rn))
                };
                let width = if wide { 64 } else { 32 };
                self.note_convert_exceptions(f, rounding, signed, width);
                let r = round_to_int_sized(f, rounding, signed, width);
                self.write_zr(rd, r);
                Ok(true)
            }
            _ => Ok(false),
        };
    }

    pub(super) fn fp_fixed_conv(&mut self, insn: u32) -> Result<bool> {
        let sf = (insn >> 31) & 1;
        let ftype = (insn >> 22) & 0b11;
        if ftype > 0b01 {
            return Ok(false); // half precision: out of scope
        }
        let double = ftype == 0b01;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let rmode = (insn >> 19) & 0b11;
        let opcode = (insn >> 16) & 0b111;
        let fbits = 64 - ((insn >> 10) & 0x3F);
        let wide = sf != 0;
        let scale = Self::pow2(fbits);
        return match (rmode, opcode) {
            // SCVTF / UCVTF: fixed-point → float.
            (0b00, 0b010) | (0b00, 0b011) => {
                let signed = opcode == 0b010;
                let v = self.read_zr(rn);
                let raw = match (signed, wide) {
                    (true, true) => v as i64 as f64,
                    (true, false) => f64::from(v as i32),
                    (false, true) => v as f64,
                    (false, false) => f64::from(v as u32),
                };
                if double {
                    self.fp_set_f64(rd, raw / scale);
                } else {
                    self.fp_set_f32(rd, (raw / scale) as f32);
                }
                Ok(true)
            }
            // FCVTZS / FCVTZU: float → fixed-point, rounding toward zero.
            (0b11, 0b000) | (0b11, 0b001) => {
                let signed = opcode == 0b000;
                let f = if double {
                    self.fp_get_f64(rn)
                } else {
                    f64::from(self.fp_get_f32(rn))
                };
                let r = round_to_int_sized(
                    f * scale,
                    Rounding::TowardZero,
                    signed,
                    if wide { 64 } else { 32 },
                );
                self.write_zr(rd, r);
                Ok(true)
            }
            _ => Ok(false),
        };
    }

    pub(super) fn fp_int_cmp_zero(&mut self, insn: u32) -> Result<bool> {
        let u = (insn >> 29) & 1;
        let op = (insn >> 10) & 0x3F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let v = (self.vregs[rn as usize] as u64) as i64;
        let cond = match (u, op) {
            (1, 0b100010) => v >= 0, // CMGE
            (0, 0b100010) => v > 0,  // CMGT
            (1, 0b100110) => v <= 0, // CMLE
            (0, 0b101010) => v < 0,  // CMLT
            _ => return Ok(false),
        };
        self.fp_set_f64(rd, f64::from_bits(if cond { u64::MAX } else { 0 }));
        return Ok(true);
    }

    pub(super) fn fp_three_source(&mut self, insn: u32) -> Result<bool> {
        let double = match (insn >> 22) & 0b11 {
            0b00 => false,
            0b01 => true,
            _ => return Ok(false), // half precision: out of scope
        };
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let rm = ((insn >> 16) & 0x1F) as u8;
        let ra = ((insn >> 10) & 0x1F) as u8;
        let o0 = (insn >> 15) & 1;
        let o1 = (insn >> 21) & 1;
        // o1 negates the accumulator, o1 != o0 the product:
        // 00 FMADD, 01 FMSUB, 10 FNMADD, 11 FNMSUB.
        let neg_a = o1 == 1;
        let neg_n = o1 != o0;
        // These are fused: one rounding for the whole multiply-add.
        if double {
            let mut fa = self.fp_get_f64(ra);
            let mut fnn = self.fp_get_f64(rn);
            let fm = self.fp_get_f64(rm);
            if neg_a {
                fa = -fa;
            }
            if neg_n {
                fnn = -fnn;
            }
            self.fp_set_f64(rd, fnn.mul_add(fm, fa));
        } else {
            let mut fa = self.fp_get_f32(ra);
            let mut fnn = self.fp_get_f32(rn);
            let fm = self.fp_get_f32(rm);
            if neg_a {
                fa = -fa;
            }
            if neg_n {
                fnn = -fnn;
            }
            self.fp_set_f32(rd, fnn.mul_add(fm, fa));
        }
        return Ok(true);
    }

    pub(super) fn fp_data_proc(&mut self, insn: u32) -> Result<bool> {
    let double = ((insn >> 22) & 1) == 1;
    let rn = ((insn >> 5) & 0x1F) as u8;
    let rd = (insn & 0x1F) as u8;
    let rm = ((insn >> 16) & 0x1F) as u8;
    // bit21 == 1. The 1-source group is handled above; bits[11:10] split the
    // rest: 01 = FCCMP, 11 = FCSEL, 10 = the 2-source ops, 00 = FCMP. Both
    // conditional forms have bit21 SET — testing for 0 made them dead code,
    // so `fcsel s30, s31, s30, gt` came out unimplemented.
    let cond = ((insn >> 12) & 0xF) as u8;
    match (insn >> 10) & 0b11 {
        0b01 => {
            // FCCMP: compare, or set NZCV from the immediate when the
            // condition fails.
            if self.condition_holds(cond) {
                self.fp_cmp(rn, rm, double);
            } else {
                self.nzcv = ((insn & 0xF) as u32) << 28;
            }
            return Ok(true);
        }
        0b11 => {
            // FCSEL: select Vn or Vm on the condition.
            let v = if self.condition_holds(cond) { rn } else { rm };
            if double {
                let f = self.fp_get_f64(v);
                self.fp_set_f64(rd, f);
            } else {
                let f = self.fp_get_f32(v);
                self.fp_set_f32(rd, f);
            }
            return Ok(true);
        }
        _ => {}
    }
    let fixed = (insn >> 10) & 0x3F;
    if fixed == 0b001000 {
        // FCMP / FCMPE. `opcode2` is bits[4:0]: bit3 selects the
        // compare-with-zero form and bit4 the signalling (E) variant, which
        // only differs in which NaNs raise an exception - not modelled.
        // Reading them from bits[9:8] took them out of Rn instead, so
        // `fcmp d0, #0.0` compared d0 with d0 and `fcmp d8, #0.0` compared
        // against whatever v0 held.
        let z = (insn >> 3) & 1;
        if z == 1 {
            self.fp_cmp_zero(rn, double);
        } else {
            self.fp_cmp(rn, rm, double);
        }
        return Ok(true);
    }
    // 2-source: opcode in bits[15:11] (its low bit is the fixed 1 of
    // bits[11:10] = 10). Single precision is computed in f32 rather than in
    // f64 and narrowed, which would round twice.
    let op = (insn >> 11) & 0x1F;
    if double {
        let a = self.fp_get_f64(rn);
        let b = self.fp_get_f64(rm);
        if op == 3 {
            self.note_divide_exceptions(a, b);
        }
        let r = match op {
            1 => a * b,            // FMUL
            3 => a / b,            // FDIV
            5 => a + b,            // FADD
            7 => a - b,            // FSUB
            9 => fp_max(a, b),     // FMAX
            11 => fp_min(a, b),    // FMIN
            13 => fp_maxnum(a, b), // FMAXNM
            15 => fp_minnum(a, b), // FMINNM
            17 => -(a * b),        // FNMUL
            _ => return Ok(false),
        };
        self.fp_set_f64(rd, r);
    } else {
        let a = self.fp_get_f32(rn);
        let b = self.fp_get_f32(rm);
        if op == 3 {
            self.note_divide_exceptions(f64::from(a), f64::from(b));
        }
        let r = match op {
            1 => a * b,
            3 => a / b,
            5 => a + b,
            7 => a - b,
            9 => fp_max(f64::from(a), f64::from(b)) as f32,
            11 => fp_min(f64::from(a), f64::from(b)) as f32,
            13 => fp_maxnum(f64::from(a), f64::from(b)) as f32,
            15 => fp_minnum(f64::from(a), f64::from(b)) as f32,
            17 => -(a * b),
            _ => return Ok(false),
        };
        self.fp_set_f32(rd, r);
    }
    Ok(true)
    }

    /// `2^n` as an `f64`, built rather than computed.
    ///
    /// `f64::powi` is a libcall in wasm (`__powidf2`, 1.4% of a translated
    /// frame), and every `fcvtzs` went through two of them for its saturation
    /// bounds. A power of two is exact: the exponent field is `1023 + n` and
    /// the mantissa is zero.
    #[inline(always)]
    fn pow2(n: u32) -> f64 {
        debug_assert!(n <= 1023);
        f64::from_bits(u64::from(1023 + n) << 52)
    }

    /// The Invalid and Inexact flags a float-to-integer convert raises: a NaN
    /// or a result the destination cannot hold is Invalid (and saturates), and
    /// anything that lost a fraction is Inexact.
    fn note_convert_exceptions(&mut self, v: f64, r: Rounding, signed: bool, bits: u32) {
        if v.is_nan() {
            self.fpsr |= FPSR_IOC;
            return;
        }
        let rounded = round_to_integral(v, r);
        let (min, upper) = if signed {
            let edge = Self::pow2(bits - 1);
            (-edge, edge)
        } else {
            (0.0, Self::pow2(bits))
        };
        if !rounded.is_finite() || rounded < min || rounded >= upper {
            self.fpsr |= FPSR_IOC;
        } else if rounded != v {
            self.fpsr |= FPSR_IXC;
        }
    }

    /// The square root of a negative has no real answer, which is Invalid.
    /// Negative zero is not negative for this purpose.
    fn fp_sqrt_is_invalid(&self, rn: u8, double: bool) -> bool {
        let v = if double { self.fp_get_f64(rn) } else { f64::from(self.fp_get_f32(rn)) };
        v < 0.0
    }

    /// Division raises Divide-by-zero for a finite numerator over zero, and
    /// Invalid for the two forms with no answer at all.
    fn note_divide_exceptions(&mut self, a: f64, b: f64) {
        if (a == 0.0 && b == 0.0) || (a.is_infinite() && b.is_infinite()) {
            self.fpsr |= FPSR_IOC;
        } else if b == 0.0 && a.is_finite() && !a.is_nan() {
            self.fpsr |= FPSR_DZC;
        }
    }

    /// Compare two FP values and set NZCV.
    pub(super) fn fp_cmp(&mut self, rn: u8, rm: u8, double: bool) {
        let a = if double {
            self.fp_get_f64(rn)
        } else {
            self.fp_get_f32(rn) as f64
        };
        let b = if double {
            self.fp_get_f64(rm)
        } else {
            self.fp_get_f32(rm) as f64
        };
        self.set_fp_flags(a, b);
    }

    pub(super) fn fp_cmp_zero(&mut self, rn: u8, double: bool) {
        let a = if double {
            self.fp_get_f64(rn)
        } else {
            self.fp_get_f32(rn) as f64
        };
        self.set_fp_flags(a, 0.0);
    }

    pub(super) fn set_fp_flags(&mut self, a: f64, b: f64) {
        let (n, z, c, v) = if a.is_nan() || b.is_nan() {
            (0, 0, 1, 1)
        } else if a < b {
            (1, 0, 0, 0)
        } else if a == b {
            (0, 1, 1, 0)
        } else {
            (0, 0, 1, 0)
        };
        self.nzcv = (n << 31) | (z << 30) | (c << 29) | (v << 28);
    }
}
