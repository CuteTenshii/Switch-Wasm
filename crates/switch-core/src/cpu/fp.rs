//! Scalar floating point: FMOV/arithmetic/compare/convert on the S and D
//! views of the vector registers, using the host's IEEE-754 semantics.

use super::bits::*;
use super::Cpu;

/// Which scalar floating-point form an encoding is, from [`Cpu::fp_form`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FpForm {
    MovImm,
    /// The 1-source group: `FCVT`, `FMOV` register-to-register, `FABS`, ...
    OneSource,
    /// `FMOV` between a general-purpose register and a vector lane.
    MovReg,
    IntConv,
    FixedConv,
    /// The scalar integer compare-to-zero forms.
    CmpZero,
    /// The 3-source fused multiply-adds.
    ThreeSource,
    /// The 2-source arithmetic, the compares and the conditional forms.
    DataProc,
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

    /// Write Sn, zeroing the other 96 bits of the vector register.
    #[inline]
    pub(super) fn fp_set_f32(&mut self, r: u8, v: f32) {
        self.vregs[r as usize] = u128::from(v.to_bits());
    }

    #[inline]
    pub(super) fn fp_set_f64(&mut self, r: u8, v: f64) {
        self.vregs[r as usize] = v.to_bits() as u128;
    }

    /// Classify a scalar FP encoding with no side effects. The test order matters:
    /// fixed-point conversions before the `sf`-inclusive guard, 3-source before `00011110`.
    pub(super) fn fp_form(insn: u32) -> FpForm {
        // FMOV (immediate): imm8 = bits[20:13], type in bits[23:22].
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0b111) == 0b100
            && ((insn >> 5) & 0x1F) == 0
        {
            return FpForm::MovImm;
        }
        // 1-source: opcode in bits[20:15], so bits[15:10] is not one field.
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0x1F) == 0b10000
        {
            return FpForm::OneSource;
        }
        // FMOV (register) between GPR and vector lane; bits[21:16] select direction/size.
        if ((insn >> 24) & 0x7F) == 0b0011110
            && ((insn >> 10) & 0x3F) == 0
            && (matches!((insn >> 16) & 0x3F, 0b100110 | 0b100111)
                || (insn >> 22) & 0x3FF == 0b10_0111_1010 && (insn >> 17) & 0x1F == 0b10111)
        {
            return FpForm::MovReg;
        }
        // FP <-> integer: rmode bits[20:19] and opcode bits[18:16] are separate fields
        // (folding in the fixed bit21 misdecodes e.g. `ucvtf d0, x1`).
        if ((insn >> 24) & 0x7F) == 0b0011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0x3F) == 0
        {
            return FpForm::IntConv;
        }
        // FP <-> fixed-point (bit21 = 0); must precede the bit31-inclusive guard below.
        if ((insn >> 24) & 0x7F) == 0b0011110 && ((insn >> 21) & 1) == 0 {
            return FpForm::FixedConv;
        }
        // CMGE/CMGT/CMLE/CMLT <Dd>, <Dn>, #0, producing an all-ones/all-zeros mask.
        if ((insn >> 31) & 1) == 0
            && ((insn >> 30) & 0b11) == 0b01
            && ((insn >> 25) & 0b1111) == 0b1111
            && ((insn >> 21) & 0xF) == 0b0111
            && ((insn >> 16) & 0x1F) == 0
        {
            return FpForm::CmpZero;
        }
        // 3-source fused ops have their own top byte `00011111`.
        if ((insn >> 24) & 0xFF) == 0b00011111 {
            return FpForm::ThreeSource;
        }
        // 2-source data processing; bit23 = 1 (half precision) is out of scope.
        if ((insn >> 24) & 0xFF) == 0b00011110 && ((insn >> 23) & 1) == 0 {
            return FpForm::DataProc;
        }
        FpForm::None
    }

    /// Run an FP instruction whose form is known; a handler may still decline it.
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
        match ftype {
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
        }
    }

    pub(super) fn fp_one_source(&mut self, insn: u32) -> Result<bool> {
        let op = (insn >> 15) & 0x3F;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let ftype = (insn >> 22) & 0b11;
        // FCVT's destination width differs from its source, so it has its own write-back.
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
            // FCVT Hd, Sn/Dn. Singles go via an exact double, so they round once.
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
            // Half-precision arithmetic is ARMv8.2, not on the A57.
            _ => return Ok(false),
        };
        if op == 0 {
            // Bit-exact copy: a float conversion could canonicalize NaNs.
            let bits = self.vregs[rn as usize];
            self.vregs[rd as usize] = if double {
                u128::from(bits as u64)
            } else {
                u128::from(bits as u32)
            };
            return Ok(true);
        }
        // Single precision is computed in f32 to avoid double rounding.
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
                // FRINTX/FRINTI round with the FPCR mode.
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
        Ok(true)
    }

    pub(super) fn fp_mov_reg(&mut self, insn: u32) -> Result<bool> {
        let sel = (insn >> 16) & 0x3F;
        // `FMOV Xd, Vn.D[1]` and `FMOV Vd.D[1], Xn`: the top half, the bottom kept.
        if (insn >> 22) & 0b11 == 0b10 {
            let rd = (insn & 0x1F) as usize;
            let rn = ((insn >> 5) & 0x1F) as u8;
            match sel {
                0b101110 => self.write_zr(rd as u8, (self.vregs[rn as usize] >> 64) as u64),
                0b101111 => {
                    let low = self.vregs[rd] & u128::from(u64::MAX);
                    self.vregs[rd] = low | u128::from(self.read_zr(rn)) << 64;
                }
                _ => return Ok(false),
            }
            return Ok(true);
        }
        let double = ((insn >> 22) & 1) == 1;
        let rd = (insn & 0x1F) as u8;
        let rn = ((insn >> 5) & 0x1F) as u8;
        match sel {
            0b100110 => {
                let val = if double {
                    self.fp_get_f64(rn).to_bits()
                } else {
                    self.fp_get_f32(rn).to_bits() as u64
                };
                self.write_zr(rd, val);
            }
            0b100111 => {
                if double {
                    self.fp_set_f64(rd, f64::from_bits(self.read_zr(rn)));
                } else {
                    self.fp_set_f32(rd, f32::from_bits(self.read_zr(rn) as u32));
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
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
        match (rmode, opcode) {
            // SCVTF / UCVTF: `sf` gives the source width, `type` the destination's.
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
            // Float to integer. rmode: 00 nearest-even, 01 +inf, 10 -inf, 11 zero;
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
        }
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
        match (rmode, opcode) {
            // SCVTF / UCVTF: fixed-point to float.
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
            // FCVTZS / FCVTZU: float to fixed-point, rounding toward zero.
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
        }
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
        Ok(true)
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
        // o1 negates the accumulator, o1 != o0 the product.
        let neg_a = o1 == 1;
        let neg_n = o1 != o0;
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
        Ok(true)
    }

    pub(super) fn fp_data_proc(&mut self, insn: u32) -> Result<bool> {
        let double = ((insn >> 22) & 1) == 1;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let rm = ((insn >> 16) & 0x1F) as u8;
        // bits[11:10]: 01 FCCMP, 11 FCSEL, 10 2-source, 00 FCMP (all bit21 = 1).
        let cond = ((insn >> 12) & 0xF) as u8;
        match (insn >> 10) & 0b11 {
            0b01 => {
                // FCCMP: when the condition fails, NZCV comes from the immediate.
                if self.condition_holds(cond) {
                    self.fp_cmp(rn, rm, double);
                } else {
                    self.nzcv = (insn & 0xF) << 28;
                }
                return Ok(true);
            }
            0b11 => {
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
            // FCMP / FCMPE: `opcode2` bit3 compares with zero, bit4 signals (unmodelled).
            let z = (insn >> 3) & 1;
            if z == 1 {
                self.fp_cmp_zero(rn, double);
            } else {
                self.fp_cmp(rn, rm, double);
            }
            return Ok(true);
        }
        // 2-source: opcode in bits[15:11]. Single precision stays in f32 to avoid double rounding.
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

    /// `2^n` built from the exponent field; `f64::powi` is a libcall in wasm.
    #[inline(always)]
    pub(super) fn pow2(n: u32) -> f64 {
        debug_assert!(n <= 1023);
        f64::from_bits(u64::from(1023 + n) << 52)
    }

    /// Raise Invalid (NaN or out of range, saturating) and Inexact for a float-to-int convert.
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

    /// Square root of a negative (not -0) is Invalid.
    fn fp_sqrt_is_invalid(&self, rn: u8, double: bool) -> bool {
        let v = if double {
            self.fp_get_f64(rn)
        } else {
            f64::from(self.fp_get_f32(rn))
        };
        v < 0.0
    }

    /// Divide-by-zero for finite / 0; Invalid for 0/0 and inf/inf.
    fn note_divide_exceptions(&mut self, a: f64, b: f64) {
        if (a == 0.0 && b == 0.0) || (a.is_infinite() && b.is_infinite()) {
            self.fpsr |= FPSR_IOC;
        } else if b == 0.0 && a.is_finite() && !a.is_nan() {
            self.fpsr |= FPSR_DZC;
        }
    }

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

/// One unpacked float: `value = 1.mantissa x 2^exponent`, point at [`NORMALIZED_POINT`].
struct Unpacked {
    sign: bool,
    exponent: i32,
    mantissa: u64,
}

/// High enough to fit a double's 52 bits with room to normalise a subnormal.
const NORMALIZED_POINT: u32 = 62;

fn format(esize: u32) -> (u32, i32) {
    if esize == 64 {
        (52, 1023)
    } else {
        (23, 127)
    }
}

fn unpack(bits: u64, esize: u32) -> Unpacked {
    let (width, bias) = format(esize);
    let sign = (bits >> (esize - 1)) & 1 == 1;
    let field = ((bits >> width) & ((1 << (esize - 1 - width)) - 1)) as i32;
    let frac = bits & ((1u64 << width) - 1);
    if field == 0 {
        // Subnormal: normalise the leading one, paying an exponent per place.
        let mut mantissa = frac << (NORMALIZED_POINT - width);
        let mut exponent = 1 - bias;
        while mantissa != 0 && mantissa & (1 << NORMALIZED_POINT) == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        Unpacked {
            sign,
            exponent,
            mantissa,
        }
    } else {
        Unpacked {
            sign,
            exponent: field - bias,
            mantissa: (1 << NORMALIZED_POINT) | (frac << (NORMALIZED_POINT - width)),
        }
    }
}

fn is_nan(bits: u64, esize: u32) -> bool {
    let (width, _) = format(esize);
    let field = (bits >> width) & ((1 << (esize - 1 - width)) - 1);
    field == ((1 << (esize - 1 - width)) - 1) && bits & ((1u64 << width) - 1) != 0
}

fn is_infinite(bits: u64, esize: u32) -> bool {
    let (width, _) = format(esize);
    let field = (bits >> width) & ((1 << (esize - 1 - width)) - 1);
    field == ((1 << (esize - 1 - width)) - 1) && bits & ((1u64 << width) - 1) == 0
}

fn quiet_nan(bits: u64, esize: u32) -> u64 {
    let (width, _) = format(esize);
    bits | (1 << (width - 1))
}

fn default_nan(esize: u32) -> u64 {
    if esize == 64 {
        0x7FF8_0000_0000_0000
    } else {
        0x7FC0_0000
    }
}

fn infinity(sign: bool, esize: u32) -> u64 {
    let (width, bias) = format(esize);
    let field = ((2 * bias) + 1) as u64;
    ((sign as u64) << (esize - 1)) | (field << width)
}

fn zero(sign: bool, esize: u32) -> u64 {
    (sign as u64) << (esize - 1)
}

/// ARM's `RecipEstimate`: a u0.9 input in [0.5, 1) to a u0.8 output.
fn recip_estimate(scaled: u64) -> u8 {
    let a = (scaled - 256) + 256;
    let a = a * 2 + 1;
    let b = (1u64 << 19) / a;
    b.div_ceil(2) as u8
}

/// ARM's `RecipSqrtEstimate`: a u0.9 input in [0.25, 1) to u0.8, cached as a table.
fn recip_sqrt_estimate(scaled: u64) -> u8 {
    static TABLE: std::sync::OnceLock<[u8; 512]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u8; 512];
        for (index, slot) in table.iter_mut().enumerate().skip(128) {
            // To u.10 with 8 significant bits, forced odd.
            let a = if index < 256 {
                (index as u64) * 2 + 1
            } else {
                ((index as u64) | 1) * 2
            };
            let mut b = 512u64;
            while a * (b + 1) * (b + 1) < (1 << 28) {
                b += 1;
            }
            *slot = b.div_ceil(2) as u8;
        }
        table
    });
    table[scaled as usize]
}

/// `FRECPE`: the architectural 8-bit estimate, not a division.
pub(super) fn recip_estimate_bits(bits: u64, esize: u32) -> u64 {
    let (width, bias) = format(esize);
    if is_nan(bits, esize) {
        return quiet_nan(bits, esize);
    }
    let value = unpack(bits, esize);
    if is_infinite(bits, esize) {
        return zero(value.sign, esize);
    }
    if value.mantissa == 0 {
        return infinity(value.sign, esize);
    }
    let exponent_min = 1 - bias;
    if value.exponent < exponent_min - 2 {
        // Round-to-nearest takes a too-small input to infinity.
        return infinity(value.sign, esize);
    }
    let scaled = value.mantissa >> (NORMALIZED_POINT - 8);
    let mut estimate = u64::from(recip_estimate(scaled)) << (width - 8);
    let mut result_exponent = -(value.exponent + 1);
    if result_exponent < exponent_min {
        // Subnormal reciprocal: restore the implicit one and shift it into the fraction.
        let implicit = 1u64 << width;
        if result_exponent == exponent_min - 1 {
            estimate = (estimate | implicit) >> 1;
        } else {
            estimate = (estimate | implicit) >> 2;
            result_exponent += 1;
        }
    }
    let field = (result_exponent + bias) as u64;
    ((value.sign as u64) << (esize - 1)) | (field << width) | (estimate & ((1u64 << width) - 1))
}

/// `FRSQRTE`; a negative input gives the default NaN.
pub(super) fn rsqrt_estimate_bits(bits: u64, esize: u32) -> u64 {
    let (width, bias) = format(esize);
    if is_nan(bits, esize) {
        return quiet_nan(bits, esize);
    }
    let value = unpack(bits, esize);
    if value.mantissa == 0 {
        return infinity(value.sign, esize);
    }
    if value.sign {
        return default_nan(esize);
    }
    if is_infinite(bits, esize) {
        return zero(false, esize);
    }
    let result_exponent = -(value.exponent + 1) >> 1;
    // An odd exponent becomes an extra bit of table input.
    let odd = value.exponent.rem_euclid(2) == 0;
    let scaled = value.mantissa >> (NORMALIZED_POINT - if odd { 7 } else { 8 });
    let estimate = u64::from(recip_sqrt_estimate(scaled)) << (width - 8);
    let field = (result_exponent + bias) as u64;
    (field << width) | (estimate & ((1u64 << width) - 1))
}

#[cfg(test)]
mod tests {
    use super::{recip_estimate_bits, rsqrt_estimate_bits};

    /// `qemu-aarch64` results for `frecpe v3.4s` over (1.5, -2.25, 3.0, 0.5).
    #[test]
    fn the_reciprocal_estimate_is_eight_bits_wide_like_the_hardware() {
        for (input, expected) in [
            (1.5f32, 0x3F2A_8000u32),
            (-2.25, 0xBEE3_0000),
            (3.0, 0x3EAA_8000),
            (0.5, 0x3FFF_8000),
        ] {
            let got = recip_estimate_bits(u64::from(input.to_bits()), 32) as u32;
            assert_eq!(got, expected, "frecpe {input}: {got:#010x}");
            assert_ne!(f32::from_bits(got), 1.0 / input);
        }
        // The double form reads the same table.
        assert_eq!(
            recip_estimate_bits(1.5f64.to_bits(), 64),
            0x3FE5_5000_0000_0000
        );
    }

    #[test]
    fn the_reciprocal_square_root_of_a_negative_is_the_default_nan() {
        for (input, expected) in [
            (1.5f32, 0x3F51_0000u32),
            (-2.25, 0x7FC0_0000),
            (3.0, 0x3F13_8000),
            (0.5, 0x3FB4_8000),
        ] {
            let got = rsqrt_estimate_bits(u64::from(input.to_bits()), 32) as u32;
            assert_eq!(got, expected, "frsqrte {input}: {got:#010x}");
        }
        assert_eq!(rsqrt_estimate_bits((-1.0f64).to_bits(), 64) >> 63, 0);
    }

    #[test]
    fn the_estimates_answer_the_edges_the_way_a_divide_cannot() {
        // Zero and infinity keep their sign, except a negative square root.
        assert_eq!(
            recip_estimate_bits(0.0f32.to_bits().into(), 32),
            0x7F80_0000
        );
        assert_eq!(
            recip_estimate_bits((-0.0f32).to_bits().into(), 32),
            0xFF80_0000
        );
        assert_eq!(
            recip_estimate_bits(f32::INFINITY.to_bits().into(), 32),
            0x0000_0000
        );
        assert_eq!(
            recip_estimate_bits(f32::NEG_INFINITY.to_bits().into(), 32),
            0x8000_0000
        );
        assert_eq!(
            rsqrt_estimate_bits(f32::INFINITY.to_bits().into(), 32),
            0x0000_0000
        );
        assert_eq!(
            rsqrt_estimate_bits(0.0f32.to_bits().into(), 32),
            0x7F80_0000
        );
        // A signalling NaN comes back quiet, and keeps its payload.
        let snan = 0x7F80_0001u32;
        assert_eq!(recip_estimate_bits(snan.into(), 32) as u32, 0x7FC0_0001);
    }
}
