//! Scalar floating point as wasm `f32`/`f64`, for the forms whose interpreter
//! arms are plain IEEE arithmetic and leave FPSR alone. Anything else answers
//! `false` and keeps its block on the interpreter.

use super::*;
use crate::cpu::bits::{FPSR_DZC, FPSR_IOC, FPSR_IXC};
use crate::cpu::fp::FpForm;

const F32X4_EXTRACT: u32 = 0x1F;
const F32X4_REPLACE: u32 = 0x20;
const F64X2_EXTRACT: u32 = 0x21;
const F64X2_REPLACE: u32 = 0x22;

/// Wasm scalar float opcodes, single then double.
const MUL: [u8; 2] = [0x94, 0xA2];
const ADD: [u8; 2] = [0x92, 0xA0];
const SUB: [u8; 2] = [0x93, 0xA1];
const MIN: [u8; 2] = [0x96, 0xA4];
const MAX: [u8; 2] = [0x97, 0xA5];
const NEG: [u8; 2] = [0x8C, 0x9A];
const ABS: [u8; 2] = [0x8B, 0x99];
const SQRT: [u8; 2] = [0x91, 0x9F];
const EQ: [u8; 2] = [0x5B, 0x61];
const LT: [u8; 2] = [0x5D, 0x63];
const NE: [u8; 2] = [0x5C, 0x62];
/// FRINTN, FRINTP, FRINTM and FRINTZ: nearest (ties to even), ceil, floor, trunc.
const ROUND: [[u8; 2]; 4] = [[0x90, 0x9E], [0x8D, 0x9B], [0x8E, 0x9C], [0x8F, 0x9D]];

const F64_PROMOTE_F32: u8 = 0xBB;
const F32_DEMOTE_F64: u8 = 0xB6;
const F64_ADD: u8 = 0xA0;
const F64_SUB: u8 = 0xA1;
const F64_MUL: u8 = 0xA2;
const F64_DIV: u8 = 0xA3;
const F64_LT: u8 = 0x63;
const F64_GT: u8 = 0x64;
const F64_GE: u8 = 0x66;
const F64_EQ: u8 = 0x61;
const I64_REINTERPRET_F64: u8 = 0xBD;
const F64_REINTERPRET_I64: u8 = 0xBF;

impl Emitter<'_> {
    /// Push `Sn` or `Dn`.
    fn fget(&mut self, r: u32, double: bool) {
        self.vget(r);
        self.f
            .simd_lane(if double { F64X2_EXTRACT } else { F32X4_EXTRACT }, 0);
    }

    /// Pop into `Sd` or `Dd`, zeroing the rest of the vector register.
    pub(super) fn fset(&mut self, r: u32, double: bool) {
        let local = if double { L_F64 } else { L_F32 };
        self.f.local_set(local);
        self.f.v128_const([0; 16]);
        self.f.local_get(local);
        self.f
            .simd_lane(if double { F64X2_REPLACE } else { F32X4_REPLACE }, 0);
        self.vset(r);
    }

    /// Push the NZCV word of comparing `Sn`/`Dn` with `Rm`, or with zero when `against_rm` is false.
    fn fcmp_flags(&mut self, rn: u32, against_rm: bool, rm: u32, double: bool) {
        let d = usize::from(double);
        let (a, b) = if double {
            (L_F64, L_F64B)
        } else {
            (L_F32, L_F32B)
        };
        self.fget(rn, double);
        self.f.local_set(a);
        if against_rm {
            self.fget(rm, double);
        } else if double {
            self.f.f64_const(0.0);
        } else {
            self.f.f32_const(0.0);
        }
        self.f.local_set(b);
        // NZCV: 0011 unordered, 1000 less, 0110 equal, 0010 greater.
        // `select` keeps its first operand when the condition holds.
        self.f.i32_const(0x6000_0000);
        self.f.i32_const(0x2000_0000);
        self.f.local_get(a);
        self.f.local_get(b);
        self.f.op_raw(EQ[d]);
        self.f.select();
        self.f.local_set(L_C);
        self.f.i32_const(0x8000_0000u32 as i32);
        self.f.local_get(L_C);
        self.f.local_get(a);
        self.f.local_get(b);
        self.f.op_raw(LT[d]);
        self.f.select();
        self.f.local_set(L_C);
        // Unordered: either side unequal to itself.
        self.f.i32_const(0x3000_0000);
        self.f.local_get(L_C);
        self.f.local_get(a);
        self.f.local_get(a);
        self.f.op_raw(NE[d]);
        self.f.local_get(b);
        self.f.local_get(b);
        self.f.op_raw(NE[d]);
        self.f.i32_or();
        self.f.select();
    }

    /// FMAX and FMIN: a NaN operand is the result, the first one first.
    /// FMAXNM and FMINNM (`number`): a NaN loses to a number; two NaNs give the first.
    fn fminmax(&mut self, rd: u32, rn: u32, rm: u32, max: bool, number: bool, double: bool) {
        let d = usize::from(double);
        let (a, b) = if double {
            (L_F64, L_F64B)
        } else {
            (L_F32, L_F32B)
        };
        self.fget(rn, double);
        self.f.local_set(a);
        self.fget(rm, double);
        self.f.local_set(b);
        self.f.local_get(a);
        self.f.local_get(b);
        self.f.local_get(a);
        self.f.local_get(b);
        self.f.op_raw(if max { MAX[d] } else { MIN[d] });
        let (first, second) = if number { (a, b) } else { (b, a) };
        self.f.local_get(first);
        self.f.local_get(first);
        self.f.op_raw(NE[d]);
        self.f.select();
        self.f.local_get(second);
        self.f.local_get(second);
        self.f.op_raw(NE[d]);
        self.f.select();
        self.fset(rd, double);
    }

    /// One scalar floating-point instruction of `form`, or `false`.
    pub(super) fn fp(&mut self, insn: u32, form: FpForm) -> bool {
        if !self.simd {
            return false;
        }
        let ftype = (insn >> 22) & 3;
        if ftype > 1 && !matches!(form, FpForm::MovReg | FpForm::IntConv) {
            return false;
        }
        let double = ftype == 1;
        let d = usize::from(double);
        let rm = (insn >> 16) & 0x1F;
        let rn = (insn >> 5) & 0x1F;
        let rd = insn & 0x1F;
        match form {
            FpForm::DataProc => match (insn >> 10) & 3 {
                // FCSEL.
                0b11 => {
                    self.fget(rn, double);
                    self.fget(rm, double);
                    self.cond_holds(((insn >> 12) & 0xF) as u8);
                    self.f.select();
                    self.fset(rd, double);
                }
                // FCMP and FCMPE, against Rm or zero.
                0b00 if (insn >> 10) & 0x3F == 0b001000 => {
                    self.fcmp_flags(rn, (insn >> 3) & 1 == 0, rm, double);
                    self.store_nzcv();
                }
                // FCCMP and FCCMPE: the comparison when the condition holds, else the immediate.
                0b01 => {
                    self.fcmp_flags(rn, true, rm, double);
                    self.f.i32_const(((insn & 0xF) << 28) as i32);
                    self.cond_holds(((insn >> 12) & 0xF) as u8);
                    self.f.select();
                    self.store_nzcv();
                }
                0b10 if (insn >> 11) & 0x1F == 3 => self.fdiv(insn, double),
                0b10 => {
                    let opcode = match (insn >> 11) & 0x1F {
                        op @ (9 | 11 | 13 | 15) => {
                            self.fminmax(rd, rn, rm, op & 2 == 0, op >= 13, double);
                            return true;
                        }
                        1 => MUL[d],
                        5 => ADD[d],
                        7 => SUB[d],
                        // FNMUL.
                        17 => MUL[d],
                        _ => return false,
                    };
                    self.fget(rn, double);
                    self.fget(rm, double);
                    self.f.op_raw(opcode);
                    if (insn >> 11) & 0x1F == 17 {
                        self.f.op_raw(NEG[d]);
                    }
                    self.fset(rd, double);
                }
                _ => return false,
            },
            FpForm::OneSource => match (insn >> 15) & 0x3F {
                // FMOV (register) copies bits; FABS and FNEG only touch the sign.
                0..=2 => {
                    self.fget(rn, double);
                    match (insn >> 15) & 0x3F {
                        1 => self.f.op_raw(ABS[d]),
                        2 => self.f.op_raw(NEG[d]),
                        _ => {}
                    }
                    self.fset(rd, double);
                }
                // FSQRT: Invalid for a negative operand, not -0 or a NaN.
                3 => {
                    self.f.i32_const(FPSR_IOC as i32);
                    self.f.i32_const(0);
                    self.fget(rn, double);
                    if double {
                        self.f.f64_const(0.0);
                    } else {
                        self.f.f32_const(0.0);
                    }
                    self.f.op_raw(LT[d]);
                    self.f.select();
                    self.or_fpsr();
                    self.fget(rn, double);
                    self.f.op_raw(SQRT[d]);
                    self.fset(rd, double);
                }
                // FCVT Sd, Dn and Dd, Sn.
                4 if double => {
                    self.fget(rn, true);
                    self.f.op_raw(0xB6);
                    self.fset(rd, false);
                }
                5 if !double => {
                    self.fget(rn, false);
                    self.f.op_raw(0xBB);
                    self.fset(rd, true);
                }
                op @ 8..=11 => {
                    self.fget(rn, double);
                    self.f.op_raw(ROUND[op as usize - 8][d]);
                    self.fset(rd, double);
                }
                _ => return false,
            },
            FpForm::ThreeSource if !double => self.fmadd_single(insn),
            FpForm::FixedConv => return self.fixed_conv(insn, double),
            FpForm::MovImm => {
                let imm8 = (insn >> 13) & 0xFF;
                let sign = (imm8 >> 7) & 1;
                let b6 = (imm8 >> 6) & 1;
                let low = imm8 & 0x3F;
                if double {
                    let exp = if b6 == 1 { 0x3FC } else { 0x400 };
                    let bits = (u64::from(sign) << 63)
                        | (u64::from(exp | (low >> 4)) << 52)
                        | (u64::from(low & 0xF) << 48);
                    self.f.f64_const(f64::from_bits(bits));
                } else {
                    let exp = if b6 == 1 { 0x7C } else { 0x80 };
                    let bits = (sign << 31) | ((exp | (low >> 4)) << 23) | ((low & 0xF) << 19);
                    self.f.f32_const(f32::from_bits(bits));
                }
                self.fset(rd, double);
            }
            FpForm::MovReg => {
                let sf = insn >> 31 == 1;
                match ((insn >> 16) & 0x3F, ftype, sf) {
                    // FMOV Wd, Sn and Xd, Dn.
                    (0b100110, 0, false) | (0b100110, 1, true) => {
                        self.vget(rn);
                        if double {
                            self.f.simd_lane(0x1D, 0);
                        } else {
                            self.f.simd_lane(0x1B, 0);
                            self.f.i64_extend_i32_u();
                        }
                        self.store_reg(gp_dst(rd));
                    }
                    // FMOV Sd, Wn and Dd, Xn.
                    (0b100111, 0, false) | (0b100111, 1, true) => {
                        self.f.v128_const([0; 16]);
                        self.read_reg_raw(rn as u8);
                        if double {
                            self.f.simd_lane(0x1E, 0);
                        } else {
                            self.f.i32_wrap_i64();
                            self.f.simd_lane(0x1C, 0);
                        }
                        self.vset(rd);
                    }
                    _ => return false,
                }
            }
            // FCVTNS, FCVTPS, FCVTMS, FCVTZS and their unsigned forms.
            FpForm::IntConv if ftype <= 1 && (insn >> 17) & 3 == 0 => {
                self.convert_to_int(insn, double)
            }
            // SCVTF/UCVTF, through a double as the interpreter does.
            FpForm::IntConv if ftype <= 1 && (insn >> 19) & 3 == 0 => {
                let signed = match (insn >> 16) & 7 {
                    0b010 => true,
                    0b011 => false,
                    _ => return false,
                };
                self.read_reg_raw(rn as u8);
                let wide = insn >> 31 == 1;
                if !wide {
                    self.f.i32_wrap_i64();
                }
                self.f.op_raw(match (wide, signed) {
                    (false, true) => 0xB7,
                    (false, false) => 0xB8,
                    (true, true) => 0xB9,
                    (true, false) => 0xBA,
                });
                if !double {
                    self.f.op_raw(0xB6);
                }
                self.fset(rd, double);
            }
            _ => return false,
        }
        true
    }

    /// FMADD, FMSUB, FNMADD and FNMSUB on singles, as the interpreter's
    /// `f32::mul_add` computes them: the sum in double precision, nudged by
    /// one unit when it lies exactly halfway between two singles but is inexact.
    fn fmadd_single(&mut self, insn: u32) {
        let (rm, ra, rn, rd) = (
            (insn >> 16) & 0x1F,
            (insn >> 10) & 0x1F,
            (insn >> 5) & 0x1F,
            insn & 0x1F,
        );
        let o1 = (insn >> 21) & 1 == 1;
        let o0 = (insn >> 15) & 1 == 1;
        let (xy, z, sum) = (L_F64, L_F64B, L_T);
        self.fget(rn, false);
        if o1 != o0 {
            self.f.op_raw(NEG[0]);
        }
        self.f.op_raw(F64_PROMOTE_F32);
        self.fget(rm, false);
        self.f.op_raw(F64_PROMOTE_F32);
        self.f.op_raw(F64_MUL);
        self.f.local_set(xy);
        self.fget(ra, false);
        if o1 {
            self.f.op_raw(NEG[0]);
        }
        self.f.op_raw(F64_PROMOTE_F32);
        self.f.local_set(z);
        self.f.local_get(xy);
        self.f.local_get(z);
        self.f.op_raw(F64_ADD);
        self.f.op_raw(I64_REINTERPRET_F64);
        self.f.local_set(sum);
        let result = |e: &mut Self| {
            e.f.local_get(sum);
            e.f.op_raw(F64_REINTERPRET_I64);
        };
        // Halfway between two singles, and neither NaN nor infinite.
        self.f.local_get(sum);
        self.f.i64_const(0x1FFF_FFFF);
        self.f.i64_and();
        self.f.i64_const(0x1000_0000);
        self.f.i64_eq();
        self.f.local_get(sum);
        self.f.i64_const(0x7FF0_0000_0000_0000);
        self.f.i64_and();
        self.f.i64_const(0x7FF0_0000_0000_0000);
        self.f.i64_ne();
        self.f.i32_and();
        // And inexact: `sum - xy == z && sum - z == xy` fails.
        result(self);
        self.f.local_get(xy);
        self.f.op_raw(F64_SUB);
        self.f.local_get(z);
        self.f.op_raw(F64_EQ);
        result(self);
        self.f.local_get(z);
        self.f.op_raw(F64_SUB);
        self.f.local_get(xy);
        self.f.op_raw(F64_EQ);
        self.f.i32_and();
        self.f.i32_eqz();
        self.f.i32_and();
        self.f.if_void();
        {
            // The rounding error, as the interpreter orders it.
            let negative = |e: &mut Self| {
                e.f.local_get(sum);
                e.f.i64_const(63);
                e.f.i64_shr_u();
                e.f.i32_wrap_i64();
            };
            self.f.local_get(sum);
            self.f.i64_const(1);
            self.f.i64_add();
            self.f.local_get(sum);
            self.f.i64_const(1);
            self.f.i64_sub();
            negative(self);
            // err = neg == (z > xy) ? xy - sum + z : z - sum + xy
            self.f.local_get(xy);
            result(self);
            self.f.op_raw(F64_SUB);
            self.f.local_get(z);
            self.f.op_raw(F64_ADD);
            self.f.local_get(z);
            result(self);
            self.f.op_raw(F64_SUB);
            self.f.local_get(xy);
            self.f.op_raw(F64_ADD);
            negative(self);
            self.f.local_get(z);
            self.f.local_get(xy);
            self.f.op_raw(F64_GT);
            self.f.i32_eq();
            self.f.select();
            self.f.f64_const(0.0);
            self.f.op_raw(F64_LT);
            self.f.i32_eq();
            self.f.select();
            self.f.local_set(sum);
        }
        self.f.end();
        result(self);
        self.f.op_raw(F32_DEMOTE_F64);
        self.fset(rd, false);
    }

    /// SCVTF, UCVTF, FCVTZS and FCVTZU between a general register and fixed point.
    fn fixed_conv(&mut self, insn: u32, double: bool) -> bool {
        let wide = insn >> 31 == 1;
        let rn = (insn >> 5) & 0x1F;
        let rd = insn & 0x1F;
        let fbits = 64 - ((insn >> 10) & 0x3F);
        let scale = f64::from_bits(u64::from(1023 + fbits) << 52);
        match ((insn >> 19) & 3, (insn >> 16) & 7) {
            (0b00, opcode @ (0b010 | 0b011)) => {
                let signed = opcode == 0b010;
                self.read_reg_raw(rn as u8);
                if !wide {
                    self.f.i32_wrap_i64();
                }
                self.f.op_raw(match (wide, signed) {
                    (false, true) => 0xB7,
                    (false, false) => 0xB8,
                    (true, true) => 0xB9,
                    (true, false) => 0xBA,
                });
                self.f.f64_const(scale);
                self.f.op_raw(F64_DIV);
                if !double {
                    self.f.op_raw(F32_DEMOTE_F64);
                }
                self.fset(rd, double);
            }
            (0b11, opcode @ (0b000 | 0b001)) => {
                let signed = opcode == 0b000;
                self.fget(rn, double);
                if !double {
                    self.f.op_raw(F64_PROMOTE_F32);
                }
                self.f.f64_const(scale);
                self.f.op_raw(F64_MUL);
                // i32/i64.trunc_sat_f64_s/u: NaN is zero and the rest saturates.
                self.f.trunc_sat(match (wide, signed) {
                    (false, true) => 2,
                    (false, false) => 3,
                    (true, true) => 6,
                    (true, false) => 7,
                });
                if !wide {
                    self.f.i64_extend_i32_u();
                }
                self.store_reg(gp_dst(rd));
            }
            _ => return false,
        }
        true
    }

    /// OR the `i32` on the stack into FPSR.
    fn or_fpsr(&mut self) {
        self.f.local_set(L_C);
        self.f.local_get(STATE);
        self.f.local_get(STATE);
        self.f.i32_load(ALIGN_4, self.layout.fpsr);
        self.f.local_get(L_C);
        self.f.i32_or();
        self.f.i32_store(ALIGN_4, self.layout.fpsr);
    }

    /// Float to integer under the rounding `rmode` names, saturating, with the
    /// interpreter's Invalid and Inexact flags.
    fn convert_to_int(&mut self, insn: u32, double: bool) {
        let wide = insn >> 31 == 1;
        let signed = (insn >> 16) & 1 == 0;
        let rd = insn & 0x1F;
        let (value, rounded) = (L_F64, L_F64B);
        self.fget((insn >> 5) & 0x1F, double);
        if !double {
            self.f.op_raw(F64_PROMOTE_F32);
        }
        self.f.local_tee(value);
        self.f.op_raw(ROUND[((insn >> 19) & 3) as usize][1]);
        self.f.local_set(rounded);
        let bits = if wide { 64 } else { 32 };
        let (min, upper) = if signed {
            (-(2f64.powi(bits - 1)), 2f64.powi(bits - 1))
        } else {
            (0.0, 2f64.powi(bits))
        };
        // Invalid unless min <= rounded < upper, which a NaN never is.
        self.f.i32_const(FPSR_IOC as i32);
        self.f.i32_const(FPSR_IXC as i32);
        self.f.i32_const(0);
        self.f.local_get(rounded);
        self.f.local_get(value);
        self.f.op_raw(NE[1]);
        self.f.select();
        self.f.local_get(rounded);
        self.f.f64_const(min);
        self.f.op_raw(F64_GE);
        self.f.local_get(rounded);
        self.f.f64_const(upper);
        self.f.op_raw(F64_LT);
        self.f.i32_and();
        self.f.i32_eqz();
        self.f.select();
        self.or_fpsr();
        self.f.local_get(rounded);
        self.f.trunc_sat(match (wide, signed) {
            (false, true) => 2,
            (false, false) => 3,
            (true, true) => 6,
            (true, false) => 7,
        });
        if !wide {
            self.f.i64_extend_i32_u();
        }
        self.store_reg(gp_dst(rd));
    }

    /// FDIV, with the interpreter's Invalid and Divide-by-zero flags.
    fn fdiv(&mut self, insn: u32, double: bool) {
        let (rm, rn, rd) = ((insn >> 16) & 0x1F, (insn >> 5) & 0x1F, insn & 0x1F);
        let (a, b) = (L_F64, L_F64B);
        for (r, local) in [(rn, a), (rm, b)] {
            self.fget(r, double);
            if !double {
                self.f.op_raw(F64_PROMOTE_F32);
            }
            self.f.local_set(local);
        }
        let zero = |e: &mut Self, local: u32| {
            e.f.local_get(local);
            e.f.f64_const(0.0);
            e.f.op_raw(F64_EQ);
        };
        let magnitude = |e: &mut Self, local: u32| {
            e.f.local_get(local);
            e.f.op_raw(ABS[1]);
            e.f.f64_const(f64::INFINITY);
        };
        self.f.i32_const(FPSR_IOC as i32);
        self.f.i32_const(FPSR_DZC as i32);
        self.f.i32_const(0);
        // Divide by zero: a finite dividend over zero.
        zero(self, b);
        magnitude(self, a);
        self.f.op_raw(F64_LT);
        self.f.i32_and();
        self.f.select();
        // Invalid: zero over zero, or infinity over infinity.
        zero(self, a);
        zero(self, b);
        self.f.i32_and();
        magnitude(self, a);
        self.f.op_raw(F64_EQ);
        magnitude(self, b);
        self.f.op_raw(F64_EQ);
        self.f.i32_and();
        self.f.i32_or();
        self.f.select();
        self.or_fpsr();
        // The quotient from the registers themselves, so a single NaN is not widened.
        self.fget(rn, double);
        self.fget(rm, double);
        self.f.op_raw(if double { F64_DIV } else { 0x95 });
        self.fset(rd, double);
    }
}
