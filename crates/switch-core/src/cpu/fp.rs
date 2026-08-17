//! Scalar floating point: FMOV/arithmetic/compare/convert on the S and D
//! views of the vector registers, using the host's IEEE-754 semantics.

use super::bits::*;
use super::Cpu;
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

    #[inline]
    pub(super) fn fp_set_f32(&mut self, r: u8, v: f32) {
        let bits = v.to_bits() as u128;
        self.vregs[r as usize] = (self.vregs[r as usize] & !0xFFFF_FFFF) | bits;
    }

    #[inline]
    pub(super) fn fp_set_f64(&mut self, r: u8, v: f64) {
        self.vregs[r as usize] = v.to_bits() as u128;
    }

    pub(super) fn try_fp(&mut self, insn: u32) -> Result<bool> {
        let sf = (insn >> 31) & 1;
        // FMOV (immediate): bits[31:24] = 00011110, bit21 = 1,
        // bits[12:10] = 100, bits[9:5] = 0, imm8 = bits[20:13], type in
        // bits[23:22] (00 = S, 01 = D). The value is VFPExpandImm() —
        // `fmov s0, #1.0` = 0x1E2E1002 (sdl-hello's float env-var helper).
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0b111) == 0b100
            && ((insn >> 5) & 0x1F) == 0
        {
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

        // FCVT (float <-> float): bits[31:24] = 00011110, bit21 = 1,
        // bits[20:15] = 000100 (Dn -> Sd) / 000101 (Sn -> Dd),
        // bits[14:10] = 10000. `fcvt s0, d0` = 0x1E624000.
        if ((insn >> 24) & 0xFF) == 0b00011110
            && ((insn >> 21) & 1) == 1
            && ((insn >> 10) & 0x1F) == 0b10000
        {
            let op = (insn >> 15) & 0x3F;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let rd = (insn & 0x1F) as u8;
            return match op {
                0b000100 => {
                    // double -> single
                    self.fp_set_f32(rd, self.fp_get_f64(rn) as f32);
                    Ok(true)
                }
                0b000101 => {
                    // single -> double
                    self.fp_set_f64(rd, self.fp_get_f32(rn) as f64);
                    Ok(true)
                }
                _ => Ok(false),
            };
        }
        // FMOV (register): move between GPR and a vector lane. bits[30:24] =
        // 0011110, bits[15:10] = 000000, bits[21:16] select direction/size.
        if ((insn >> 24) & 0x7F) == 0b0011110
            && ((insn >> 10) & 0x3F) == 0
            && matches!((insn >> 16) & 0x3F, 0b100110 | 0b100111)
        {
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

        // Integer <-> float conversions: bits[30:24] = 0011110 (sf at bit31),
        // `type` in bits[23:22] picks single/double, opc in bits[21:16].
        // The pure integer forms have bits[15:10] = 0; non-zero there means a
        // fixed-point scale (or, with bit21=1, a 2-source FP op — FADD etc.).
        if ((insn >> 24) & 0x7F) == 0b0011110 && ((insn >> 10) & 0x3F) == 0 {
            let ftype = (insn >> 22) & 0b11; // 00 → S, 01 → D
            let opc = (insn >> 16) & 0x3F;
            let rd = (insn & 0x1F) as u8;
            let rn = ((insn >> 5) & 0x1F) as u8;
            let fbits = (insn >> 10) & 0x3F;
            // type 00 = single, 01 = double (10/11 = half, out of scope). The
            // float size is independent of `sf`; the old `&& sf == 1` clause
            // made `fcvtzs w, d1` / `scvtf d0, w1` pick the wrong register size.
            let use_double = ftype == 0b01;
            return match opc {
                0b000010 | 0b000011 => {
                    // SCVTF / UCVTF: integer → float.
                    let signed = opc == 0b000010;
                    let v = self.read_zr(rn);
                    if use_double {
                        let f = if signed {
                            (v as i64) as f64
                        } else {
                            v as f64
                        };
                        self.fp_set_f64(rd, f);
                    } else {
                        let f = if signed {
                            (v as i32) as f32
                        } else {
                            v as f32
                        };
                        self.fp_set_f32(rd, f);
                    }
                    Ok(true)
                }
                 0b011000 | 0b011001 | 0b101000 | 0b111000 | 0b101001 | 0b111001 => {
                    // FCVTZS / FCVTZU: float → integer, round toward zero.
                    // opc bit16 = 0 for ZS (signed), 1 for ZU (unsigned);
                    // bit20 (and `type`) selects single vs double source.
                    // `fcvtzs w24, d1` = 0x1e780038, `fcvtzu x8, d1` = 0x9e790028.
                    let signed = (opc & 1) == 0;
                    let f = if use_double {
                        self.fp_get_f64(rn)
                    } else {
                        self.fp_get_f32(rn) as f64
                    };
                    let r = if signed {
                        if sf != 0 {
                            f as i64 as u64
                        } else {
                            f as i32 as u32 as u64
                        }
                    } else if sf != 0 {
                        f as u64
                    } else {
                        f as u32 as u64
                    };
                    self.write_zr(rd, r);
                    Ok(true)
                }
                0b100000..=0b100111 if fbits == 0 => {
                    // Float → integer with explicit rounding mode (opc 1000xx).
                    let (signed, rounding) = match opc {
                        0b100000 => (true, Rounding::TiesEven),
                        0b100001 => (false, Rounding::TiesEven),
                        0b100010 => (true, Rounding::TowardNeg),
                        0b100011 => (false, Rounding::TowardNeg),
                        0b100100 => (true, Rounding::TowardPos),
                        0b100101 => (false, Rounding::TowardPos),
                        0b100110 => (true, Rounding::TiesAway),
                        0b100111 => (false, Rounding::TiesAway),
                        _ => unreachable!(),
                    };
                    let f = if use_double {
                        self.fp_get_f64(rn)
                    } else {
                        self.fp_get_f32(rn) as f64
                    };
                    let r = round_to_int(f, rounding, signed);
                    self.write_zr(rd, r & Self::mask(sf != 0));
                    Ok(true)
                }
                _ => Ok(false),
            }
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
            && ((insn >> 16) & 0x1F) == 0
        {
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

        // Scalar FP data processing: bits[31:24] = 00011110 (single/double;
        // bit23 = 1 selects half precision, which is out of scope).
        if ((insn >> 24) & 0xFF) != 0b00011110 || ((insn >> 23) & 1) == 1 {
            return Ok(false);
        }
        let double = ((insn >> 22) & 1) == 1;
        let rn = ((insn >> 5) & 0x1F) as u8;
        let rd = (insn & 0x1F) as u8;
        let rm = ((insn >> 16) & 0x1F) as u8;
        // 3-source fused ops: bits[31:24] = 00011111.
        if ((insn >> 24) & 0xFF) == 0b00011111 {
            let ra = ((insn >> 10) & 0x1F) as u8;
            let o3 = (insn >> 15) & 1;
            let o1 = (insn >> 21) & 1;
            // o1/o3 → negate-accumulator / negate-product (QEMU do_fmadd):
            // 00 FMADD, 01 FMSUB, 10 FNMADD, 11 FNMSUB.
            let neg_a = o1 == 1;
            let neg_n = o1 != o3;
            let fa = if double {
                self.fp_get_f64(ra)
            } else {
                self.fp_get_f32(ra) as f64
            };
            let fn_ = if double {
                self.fp_get_f64(rn)
            } else {
                self.fp_get_f32(rn) as f64
            };
            let fm = if double {
                self.fp_get_f64(rm)
            } else {
                self.fp_get_f32(rm) as f64
            };
            let fa = if neg_a { -fa } else { fa };
            let fn_ = if neg_n { -fn_ } else { fn_ };
            let r = fn_ * fm + fa;
            if double {
                self.fp_set_f64(rd, r);
            } else {
                self.fp_set_f32(rd, r as f32);
            }
            return Ok(true);
        }
        // bit21 == 0 → the compare/select encodings (FCSEL/FCCMP live here,
        // distinguished by bits[11:10]).
        if ((insn >> 21) & 1) == 0 {
            let sel_lo = (insn >> 10) & 0b11;
            let cond = ((insn >> 12) & 0xF) as u8;
            if sel_lo == 0b01 {
                // FCCMP: compare, else set nzcv from the immediate.
                let nzcv = (insn & 0xF) as u32;
                if self.condition_holds(cond) {
                    self.fp_cmp(rn, rm, double);
                } else {
                    self.nzcv = nzcv << 28;
                }
                return Ok(true);
            }
            if sel_lo == 0b11 {
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
            return Ok(false);
        }
        // bit21 == 1: 1-source (bits[14:10] == 10000), 2-source, FCMP.
        let fixed = (insn >> 10) & 0x3F;
        if fixed == 0b100000 {
            // 1-source: opcode in bits[20:15].
            let op = (insn >> 15) & 0x3F;
            let a = if double { self.fp_get_f64(rn) } else { self.fp_get_f32(rn) as f64 };
            let r = match op {
                1 => a.abs(),
                2 => -a,
                3 => a.sqrt(),
                4 if double => a as f32 as f64, // FCVT Sd←Dn
                4 => a,                          // FCVT Sh←Sn (half, unsupported)
                5 if !double => a as f64,        // FCVT Dd←Sn
                5 => a as f32 as f64,            // FCVT Hd←Dn (half, unsupported)
                8 => a.round_ties_even(),        // FRINTN
                9 => a.ceil(),                   // FRINTP
                10 => a.floor(),                 // FRINTM
                11 => a.trunc(),                 // FRINTZ
                12 => a.round(),                 // FRINTA (ties away)
                14 | 15 => a,                    // FRINTX/I (already rounded)
                _ => return Ok(false),
            };
            if double {
                self.fp_set_f64(rd, r);
            } else if op == 4 && !double {
                // FCVT to half — out of scope.
                return Ok(false);
            } else {
                self.fp_set_f32(rd, r as f32);
            }
            return Ok(true);
        }
        if fixed == 0b001000 {
            // FCMP / FCMPE (with or without zero).
            let e = (insn >> 9) & 1;
            let z = (insn >> 8) & 1;
            let _ = e;
            if z == 1 {
                self.fp_cmp_zero(rn, double);
            } else {
                self.fp_cmp(rn, rm, double);
            }
            return Ok(true);
        }
        // 2-source: opcode in bits[15:11].
        let op = (insn >> 11) & 0x1F;
        let a = if double { self.fp_get_f64(rn) } else { self.fp_get_f32(rn) as f64 };
        let b = if double { self.fp_get_f64(rm) } else { self.fp_get_f32(rm) as f64 };
        let r = match op {
            1 => a * b,    // FMUL
            3 => a / b,    // FDIV
            5 => a + b,    // FADD
            7 => a - b,    // FSUB
            9 => fp_max(a, b),     // FMAX
            11 => fp_min(a, b),    // FMIN
            13 => fp_maxnum(a, b), // FMAXNM
            15 => fp_minnum(a, b), // FMINNM
            17 => -(a * b),        // FNMUL
            _ => return Ok(false),
        };
        if double {
            self.fp_set_f64(rd, r);
        } else {
            self.fp_set_f32(rd, r as f32);
        }
        Ok(true)
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
