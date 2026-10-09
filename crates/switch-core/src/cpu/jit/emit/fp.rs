//! Scalar floating point as wasm `f32`/`f64`, for the forms whose interpreter
//! arms are plain IEEE arithmetic and leave FPSR alone. Anything else answers
//! `false` and keeps its block on the interpreter.

use super::*;
use crate::cpu::fp::FpForm;

const F32X4_EXTRACT: u32 = 0x1F;
const F32X4_REPLACE: u32 = 0x20;
const F64X2_EXTRACT: u32 = 0x21;
const F64X2_REPLACE: u32 = 0x22;

/// Wasm scalar float opcodes, single then double.
const MUL: [u8; 2] = [0x94, 0xA2];
const ADD: [u8; 2] = [0x92, 0xA0];
const SUB: [u8; 2] = [0x93, 0xA1];
const NEG: [u8; 2] = [0x8C, 0x9A];
const ABS: [u8; 2] = [0x8B, 0x99];
const EQ: [u8; 2] = [0x5B, 0x61];
const LT: [u8; 2] = [0x5D, 0x63];
const NE: [u8; 2] = [0x5C, 0x62];

impl Emitter<'_> {
    /// Push `Sn` or `Dn`.
    fn fget(&mut self, r: u32, double: bool) {
        self.vget(r);
        self.f
            .simd_lane(if double { F64X2_EXTRACT } else { F32X4_EXTRACT }, 0);
    }

    /// Pop into `Sd` or `Dd`, zeroing the rest of the vector register.
    fn fset(&mut self, r: u32, double: bool) {
        let local = if double { L_F64 } else { L_F32 };
        self.f.local_set(local);
        self.f.v128_const([0; 16]);
        self.f.local_get(local);
        self.f
            .simd_lane(if double { F64X2_REPLACE } else { F32X4_REPLACE }, 0);
        self.vset(r);
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
                    let zero = (insn >> 3) & 1 == 1;
                    let operand = |e: &mut Self, r: u32| {
                        if r == u32::MAX {
                            if double {
                                e.f.f64_const(0.0);
                            } else {
                                e.f.f32_const(0.0);
                            }
                        } else {
                            e.fget(r, double);
                        }
                    };
                    let b = if zero { u32::MAX } else { rm };
                    // NZCV: 0011 unordered, 1000 less, 0110 equal, 0010 greater.
                    operand(self, rn);
                    self.f.local_set(if double { L_F64 } else { L_F32 });
                    operand(self, b);
                    self.f.local_set(if double { L_F64B } else { L_F32B });
                    let (a, b) = if double {
                        (L_F64, L_F64B)
                    } else {
                        (L_F32, L_F32B)
                    };
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
                    self.store_nzcv();
                }
                0b10 => {
                    let opcode = match (insn >> 11) & 0x1F {
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
                _ => return false,
            },
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
}
