//! VFP: the floating-point coprocessor a 32-bit title does its arithmetic in.
//!
//! Measured across Mario Kart 8 Deluxe's eight modules, the VFP data
//! processing, load/store and register-transfer encodings are 5% of the
//! binary, small next to the integer core, and load-bearing: `rtld` reaches
//! its first `vpush {d0-d7}` 8192 instructions in.
//!
//! # The registers are A64's, seen differently
//!
//! There is no separate file. AArch32's `D0`..`D31` are the low halves of
//! AArch64's `V0`..`V15`: `D(2n)` is the bottom 64 bits of `V(n)` and
//! `D(2n+1)` the top, and `S(2n)`/`S(2n+1)` split each `D` the same way. So
//! [`crate::cpu::Cpu::vregs`] backs both states and a context switch already
//! carries the floating-point state without knowing which one wrote it.
//!
//! # FPSCR
//!
//! AArch32 keeps in one register what A64 splits between `FPCR` and `FPSR`,
//! and the halves line up: the rounding mode is bits 23:22 in both, so
//! [`crate::cpu::bits::fpcr_rounding`] reads either. The one part with nowhere
//! to go is FPSCR's own N/Z/C/V, which `VCMP` writes and `VMRS APSR_nzcv`
//! copies into the condition flags: a comparison does *not* set the condition
//! flags directly, the way A64's `FCMP` does.

use crate::cpu::Cpu;
use crate::{Error, Result};

/// Expand a VFP 8-bit immediate to a double, as `VFPExpandImm` defines it:
/// the sign, then bit 6 inverted, then bit 6 repeated to fill the exponent,
/// then the low six bits as the top of the mantissa.
fn expand_imm_f64(imm8: u32) -> u64 {
    let sign = u64::from((imm8 >> 7) & 1) << 63;
    let top = u64::from(!(imm8 >> 6) & 1) << 62;
    let fill = if (imm8 >> 6) & 1 != 0 { 0xFFu64 } else { 0 } << 54;
    let mantissa = u64::from(imm8 & 0x3F) << 48;
    sign | top | fill | mantissa
}

/// The same for a single, whose exponent is five bits shorter.
fn expand_imm_f32(imm8: u32) -> u32 {
    let sign = ((imm8 >> 7) & 1) << 31;
    let top = (!(imm8 >> 6) & 1) << 30;
    let fill = if (imm8 >> 6) & 1 != 0 { 0x1Fu32 } else { 0 } << 25;
    let mantissa = (imm8 & 0x3F) << 19;
    sign | top | fill | mantissa
}

impl Cpu {
    /// `Sn`, which is one quarter of a vector register.
    #[inline]
    pub(super) fn vfp_s(&self, n: u8) -> u32 {
        (self.vregs[(n >> 2) as usize] >> (32 * u32::from(n & 3))) as u32
    }

    #[inline]
    pub(super) fn set_vfp_s(&mut self, n: u8, val: u32) {
        let shift = 32 * u32::from(n & 3);
        let slot = &mut self.vregs[(n >> 2) as usize];
        *slot = (*slot & !(u128::from(u32::MAX) << shift)) | (u128::from(val) << shift);
    }

    /// `Dn`, which is one half of a vector register.
    #[inline]
    pub(super) fn vfp_d(&self, n: u8) -> u64 {
        (self.vregs[(n >> 1) as usize] >> (64 * u32::from(n & 1))) as u64
    }

    #[inline]
    pub(super) fn set_vfp_d(&mut self, n: u8, val: u64) {
        let shift = 64 * u32::from(n & 1);
        let slot = &mut self.vregs[(n >> 1) as usize];
        *slot = (*slot & !(u128::from(u64::MAX) << shift)) | (u128::from(val) << shift);
    }

    #[inline]
    fn vfp_f32(&self, n: u8) -> f32 {
        f32::from_bits(self.vfp_s(n))
    }

    #[inline]
    fn vfp_f64(&self, n: u8) -> f64 {
        f64::from_bits(self.vfp_d(n))
    }

    /// `VLDR`, `VSTR`, `VLDM`, `VSTM` and the core-register pair transfers,
    /// which all live in the coprocessor load/store space.
    pub(super) fn a32_coproc_load_store(&mut self, insn: u32) -> Result<()> {
        let pre = (insn >> 24) & 1 != 0;
        let add = (insn >> 23) & 1 != 0;
        let d_bit = ((insn >> 22) & 1) as u8;
        let writeback = (insn >> 21) & 1 != 0;
        let load = (insn >> 20) & 1 != 0;
        let rn = ((insn >> 16) & 0xF) as u8;
        let vd = ((insn >> 12) & 0xF) as u8;
        let double = (insn >> 8) & 0xF == 0xB;
        let imm8 = insn & 0xFF;

        // P and W both clear is not an addressing mode at all: it is the pair
        // of core registers moving to or from one D register (or two S).
        if !pre && !writeback {
            return self.a32_vfp_core_pair(insn, load, double);
        }

        if pre && !writeback {
            // VLDR/VSTR: a single register at base ± (imm8 * 4).
            let offset = imm8 * 4;
            let base = self.r32(rn);
            let addr = if add {
                base.wrapping_add(offset)
            } else {
                base.wrapping_sub(offset)
            };
            if double {
                let dd = (d_bit << 4) | vd;
                if load {
                    let lo = self.mem.read_u32(addr)?;
                    let hi = self.mem.read_u32(addr.wrapping_add(4))?;
                    self.set_vfp_d(dd, u64::from(lo) | (u64::from(hi) << 32));
                } else {
                    let val = self.vfp_d(dd);
                    self.mem.write_u32(addr, val as u32)?;
                    self.mem
                        .write_u32(addr.wrapping_add(4), (val >> 32) as u32)?;
                }
            } else {
                let sd = (vd << 1) | d_bit;
                if load {
                    let val = self.mem.read_u32(addr)?;
                    self.set_vfp_s(sd, val);
                } else {
                    let val = self.vfp_s(sd);
                    self.mem.write_u32(addr, val)?;
                }
            }
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }

        // VLDM/VSTM, and so VPUSH and VPOP: `imm8` counts *words*, so a list
        // of doubles is half as long as it is wide.
        let count = if double { imm8 / 2 } else { imm8 };
        let bytes = imm8 * 4;
        let base = self.r32(rn);
        // The decrementing form addresses below the base; both run upwards
        // from wherever they start.
        let mut addr = if add { base } else { base.wrapping_sub(bytes) };
        for i in 0..count as u8 {
            if double {
                let dd = ((d_bit << 4) | vd).wrapping_add(i) & 0x1F;
                if load {
                    let lo = self.mem.read_u32(addr)?;
                    let hi = self.mem.read_u32(addr.wrapping_add(4))?;
                    self.set_vfp_d(dd, u64::from(lo) | (u64::from(hi) << 32));
                } else {
                    let val = self.vfp_d(dd);
                    self.mem.write_u32(addr, val as u32)?;
                    self.mem
                        .write_u32(addr.wrapping_add(4), (val >> 32) as u32)?;
                }
                addr = addr.wrapping_add(8);
            } else {
                let sd = (((vd << 1) | d_bit).wrapping_add(i)) & 0x1F;
                if load {
                    let val = self.mem.read_u32(addr)?;
                    self.set_vfp_s(sd, val);
                } else {
                    let val = self.vfp_s(sd);
                    self.mem.write_u32(addr, val)?;
                }
                addr = addr.wrapping_add(4);
            }
        }
        if writeback {
            let end = if add {
                base.wrapping_add(bytes)
            } else {
                base.wrapping_sub(bytes)
            };
            self.set_r32(rn, end);
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `VMOV` between a pair of core registers and one `D` register, or two
    /// consecutive `S` registers.
    fn a32_vfp_core_pair(&mut self, insn: u32, load: bool, double: bool) -> Result<()> {
        let rt = ((insn >> 12) & 0xF) as u8;
        let rt2 = ((insn >> 16) & 0xF) as u8;
        let m_bit = ((insn >> 5) & 1) as u8;
        let vm = (insn & 0xF) as u8;
        if double {
            let dm = (m_bit << 4) | vm;
            if load {
                let val = self.vfp_d(dm);
                self.set_r32(rt, val as u32);
                self.set_r32(rt2, (val >> 32) as u32);
            } else {
                let val = u64::from(self.r32(rt)) | (u64::from(self.r32(rt2)) << 32);
                self.set_vfp_d(dm, val);
            }
        } else {
            let sm = (vm << 1) | m_bit;
            if load {
                let lo = self.vfp_s(sm);
                let hi = self.vfp_s(sm.wrapping_add(1) & 0x1F);
                self.set_r32(rt, lo);
                self.set_r32(rt2, hi);
            } else {
                let lo = self.r32(rt);
                let hi = self.r32(rt2);
                self.set_vfp_s(sm, lo);
                self.set_vfp_s(sm.wrapping_add(1) & 0x1F, hi);
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// VFP data processing, and the single-register transfers that share its
    /// encoding space.
    pub(super) fn a32_vfp_data(&mut self, insn: u32) -> Result<()> {
        if insn & 0x10 != 0 {
            return self.a32_vfp_transfer(insn);
        }
        let double = (insn >> 8) & 0xF == 0xB;
        let d_bit = ((insn >> 22) & 1) as u8;
        let n_bit = ((insn >> 7) & 1) as u8;
        let m_bit = ((insn >> 5) & 1) as u8;
        let vd = ((insn >> 12) & 0xF) as u8;
        let vn = ((insn >> 16) & 0xF) as u8;
        let vm = (insn & 0xF) as u8;
        // A double numbers its top bit last and a single numbers it first,
        // which is the whole of the difference between the two register
        // encodings.
        let (rd, rn, rm) = if double {
            ((d_bit << 4) | vd, (n_bit << 4) | vn, (m_bit << 4) | vm)
        } else {
            ((vd << 1) | d_bit, (vn << 1) | n_bit, (vm << 1) | m_bit)
        };
        let negate = (insn >> 6) & 1 != 0;

        match ((insn >> 23) & 1, (insn >> 20) & 0b11) {
            (0, 0b00) | (0, 0b01) | (1, 0b01) | (1, 0b10) => {
                self.a32_vfp_multiply_accumulate(insn, double, rd, rn, rm)
            }
            (0, 0b10) => {
                // VMUL, and VNMUL which negates the product.
                self.vfp_write(
                    double,
                    rd,
                    |a, b| a * b,
                    self.vfp_pair(double, rn, rm),
                    negate,
                )
            }
            (0, 0b11) => {
                // VADD and VSUB.
                let (a, b) = self.vfp_pair(double, rn, rm);
                let result = if negate { a - b } else { a + b };
                self.vfp_store(double, rd, result);
                Ok(())
            }
            (1, 0b00) => {
                let (a, b) = self.vfp_pair(double, rn, rm);
                self.vfp_store(double, rd, a / b);
                Ok(())
            }
            _ => self.a32_vfp_extended(insn, double, rd, rm),
        }
        .map(|()| self.pc = self.pc.wrapping_add(4))
    }

    /// The two source operands of a data-processing instruction, widened to
    /// `f64` so one routine serves both precisions.
    #[inline]
    fn vfp_pair(&self, double: bool, rn: u8, rm: u8) -> (f64, f64) {
        if double {
            (self.vfp_f64(rn), self.vfp_f64(rm))
        } else {
            (f64::from(self.vfp_f32(rn)), f64::from(self.vfp_f32(rm)))
        }
    }

    #[inline]
    fn vfp_store(&mut self, double: bool, rd: u8, val: f64) {
        if double {
            self.set_vfp_d(rd, val.to_bits());
        } else {
            self.set_vfp_s(rd, (val as f32).to_bits());
        }
    }

    #[inline]
    fn vfp_write(
        &mut self,
        double: bool,
        rd: u8,
        op: impl Fn(f64, f64) -> f64,
        (a, b): (f64, f64),
        negate: bool,
    ) -> Result<()> {
        let result = op(a, b);
        self.vfp_store(double, rd, if negate { -result } else { result });
        Ok(())
    }

    /// The multiply-accumulate family, fused and unfused. Which of the eight
    /// it is comes from three bits: the operation group, whether the product
    /// is negated, and whether the accumulator is.
    fn a32_vfp_multiply_accumulate(
        &mut self,
        insn: u32,
        double: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    ) -> Result<()> {
        let (a, b) = self.vfp_pair(double, rn, rm);
        let acc = if double {
            self.vfp_f64(rd)
        } else {
            f64::from(self.vfp_f32(rd))
        };
        let op2 = (insn >> 6) & 1 != 0;
        let fused = (insn >> 23) & 1 != 0;
        let (product, accumulator) = match ((insn >> 20) & 0b11, op2) {
            // VMLA / VFMA: the accumulator and the product both as they are.
            (0b00, false) | (0b10, false) => (a * b, acc),
            // VMLS / VFMS: the product negated.
            (0b00, true) | (0b10, true) => (-(a * b), acc),
            // VNMLS / VFNMS: the accumulator negated.
            (0b01, false) => (a * b, -acc),
            // VNMLA / VFNMA: both negated.
            _ => (-(a * b), -acc),
        };
        // The fused forms round once, which is the whole point of them, so the
        // product must not be materialised first.
        let result = if fused {
            let signed_a = if product.is_sign_negative() != (a * b).is_sign_negative() {
                -a
            } else {
                a
            };
            signed_a.mul_add(b, accumulator)
        } else {
            product + accumulator
        };
        self.vfp_store(double, rd, result);
        Ok(())
    }

    /// The `opc1 == 1x11` corner of the data-processing space: the moves, the
    /// one-operand arithmetic, the comparisons and the conversions.
    fn a32_vfp_extended(&mut self, insn: u32, double: bool, rd: u8, rm: u8) -> Result<()> {
        let opc2 = (insn >> 16) & 0xF;
        let opc3 = (insn >> 6) & 0b11;
        // An even opc3 is the immediate move; everything else is an operation.
        if opc3 & 1 == 0 {
            let imm8 = ((insn >> 12) & 0xF0) | (insn & 0xF);
            if double {
                self.set_vfp_d(rd, expand_imm_f64(imm8));
            } else {
                self.set_vfp_s(rd, expand_imm_f32(imm8));
            }
            return Ok(());
        }
        match (opc2, opc3) {
            // VMOV register, and VABS.
            (0b0000, 0b01) => {
                if double {
                    let v = self.vfp_d(rm);
                    self.set_vfp_d(rd, v);
                } else {
                    let v = self.vfp_s(rm);
                    self.set_vfp_s(rd, v);
                }
                Ok(())
            }
            (0b0000, _) => {
                // Clearing the sign bit, not `f64::abs`: the sign of a NaN is
                // architectural here.
                if double {
                    let v = self.vfp_d(rm) & !(1 << 63);
                    self.set_vfp_d(rd, v);
                } else {
                    let v = self.vfp_s(rm) & !(1 << 31);
                    self.set_vfp_s(rd, v);
                }
                Ok(())
            }
            // VNEG, which flips the sign bit for the same reason.
            (0b0001, 0b01) => {
                if double {
                    let v = self.vfp_d(rm) ^ (1 << 63);
                    self.set_vfp_d(rd, v);
                } else {
                    let v = self.vfp_s(rm) ^ (1 << 31);
                    self.set_vfp_s(rd, v);
                }
                Ok(())
            }
            (0b0001, _) => {
                let v = if double {
                    self.vfp_f64(rm)
                } else {
                    f64::from(self.vfp_f32(rm))
                };
                self.vfp_store(double, rd, v.sqrt());
                Ok(())
            }
            // VCMP and VCMPE, against another register or against zero.
            (0b0100 | 0b0101, _) => {
                let a = if double {
                    self.vfp_f64(rd)
                } else {
                    f64::from(self.vfp_f32(rd))
                };
                let b = if opc2 == 0b0101 {
                    0.0
                } else if double {
                    self.vfp_f64(rm)
                } else {
                    f64::from(self.vfp_f32(rm))
                };
                self.set_fpscr_flags(a, b);
                Ok(())
            }
            // VCVT between the two precisions. `sz` names the *operand's*
            // width, so the destination is numbered by the other rule: a
            // double is `D:Vd` and a single `Vd:D`, and using one rule for
            // both sends the result to a register 16 away.
            (0b0111, _) => {
                let vd = ((insn >> 12) & 0xF) as u8;
                let d_bit = ((insn >> 22) & 1) as u8;
                if double {
                    let v = self.vfp_f64(rm) as f32;
                    self.set_vfp_s((vd << 1) | d_bit, v.to_bits());
                } else {
                    let v = f64::from(self.vfp_f32(rm));
                    self.set_vfp_d((d_bit << 4) | vd, v.to_bits());
                }
                Ok(())
            }
            // VCVT from an integer, which always arrives in an S register
            // however wide the result is, so the operand is numbered
            // `Vm:M` even when `sz` says double.
            (0b1000, _) => {
                let sm = ((insn & 0xF) as u8) << 1 | ((insn >> 5) & 1) as u8;
                let signed = (insn >> 7) & 1 != 0;
                let bits = self.vfp_s(sm);
                let v = if signed {
                    f64::from(bits as i32)
                } else {
                    f64::from(bits)
                };
                self.vfp_store(double, rd, v);
                Ok(())
            }
            // VCVT to an integer, which always lands in an S register. The
            // architecture saturates rather than wrapping, which is what a
            // float-to-int `as` cast in Rust already does.
            (0b1100 | 0b1101, _) => {
                let v = if double {
                    self.vfp_f64(rm)
                } else {
                    f64::from(self.vfp_f32(rm))
                };
                let bits = if opc2 & 1 != 0 {
                    (v as i32) as u32
                } else {
                    v as u32
                };
                let vd = ((insn >> 12) & 0xF) as u8;
                let d_bit = ((insn >> 22) & 1) as u8;
                self.set_vfp_s((vd << 1) | d_bit, bits);
                Ok(())
            }
            _ => Err(Error::Cpu(format!(
                "unimplemented VFP operation {:#010x} at pc={:#010x}",
                insn, self.pc
            ))),
        }
    }

    /// A comparison's result, which goes to FPSCR rather than to the condition
    /// flags: AArch32 needs a `VMRS` to move it across.
    fn set_fpscr_flags(&mut self, a: f64, b: f64) {
        let (n, z, c, v) = if a.is_nan() || b.is_nan() {
            (0, 0, 1, 1)
        } else if a < b {
            (1, 0, 0, 0)
        } else if a == b {
            (0, 1, 1, 0)
        } else {
            (0, 0, 1, 0)
        };
        self.fpscr_nzcv = (n << 31) | (z << 30) | (c << 29) | (v << 28);
    }

    /// `VMOV` between one core register and one `S`, and `VMRS`/`VMSR`.
    fn a32_vfp_transfer(&mut self, insn: u32) -> Result<()> {
        let to_arm = (insn >> 20) & 1 != 0;
        let rt = ((insn >> 12) & 0xF) as u8;
        // VMSR/VMRS name a system register in the field a VMOV uses for Vn.
        if (insn >> 21) & 0b111 == 0b111 {
            let reg = (insn >> 16) & 0xF;
            if reg != 1 {
                return Err(Error::Cpu(format!(
                    "unimplemented VFP system register {reg} at pc={:#010x}",
                    self.pc
                )));
            }
            if to_arm {
                // Rt == 15 is `VMRS APSR_nzcv`, which is how a comparison
                // reaches the condition flags at all.
                let fpscr = self.fpscr();
                if rt == 15 {
                    self.nzcv = fpscr & 0xF000_0000;
                } else {
                    self.set_r32(rt, fpscr);
                }
            } else {
                let val = self.r32(rt);
                self.set_fpscr(val);
            }
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }
        let sn = ((((insn >> 16) & 0xF) as u8) << 1) | ((insn >> 7) & 1) as u8;
        if to_arm {
            let val = self.vfp_s(sn);
            self.set_r32(rt, val);
        } else {
            let val = self.r32(rt);
            self.set_vfp_s(sn, val);
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// FPSCR assembled from the halves A64 keeps it in, plus the comparison
    /// flags that belong to neither.
    fn fpscr(&self) -> u32 {
        self.fpscr_nzcv | (self.fpcr & 0x07C0_0000) | (self.fpsr & 0x0000_009F)
    }

    fn set_fpscr(&mut self, val: u32) {
        self.fpscr_nzcv = val & 0xF000_0000;
        self.fpcr = val & 0x07C0_0000;
        self.fpsr = val & 0x0000_009F;
    }
}

impl Cpu {
    /// The ARMv8 additions to AArch32's floating point, which live in the
    /// unconditional encoding space because they carry their own condition or
    /// rounding mode rather than the instruction's.
    ///
    /// `VSEL` is 2,672 of the 2,915 such instructions in Mario Kart 8 Deluxe,
    /// a compiler emitting branchless `min`, `max` and ternaries.
    pub(super) fn a32_vfp_v8(&mut self, insn: u32) -> Result<()> {
        let double = (insn >> 8) & 0xF == 0xB;
        let d_bit = ((insn >> 22) & 1) as u8;
        let n_bit = ((insn >> 7) & 1) as u8;
        let m_bit = ((insn >> 5) & 1) as u8;
        let vd = ((insn >> 12) & 0xF) as u8;
        let vn = ((insn >> 16) & 0xF) as u8;
        let vm = (insn & 0xF) as u8;
        let (rd, rn, rm) = if double {
            ((d_bit << 4) | vd, (n_bit << 4) | vn, (m_bit << 4) | vm)
        } else {
            ((vd << 1) | d_bit, (vn << 1) | n_bit, (vm << 1) | m_bit)
        };

        if (insn >> 23) & 1 == 0 {
            // VSEL, whose two condition bits name four of the sixteen
            // conditions rather than being one of them.
            let cond = match (insn >> 20) & 0b11 {
                0b00 => 0x0, // EQ
                0b01 => 0x6, // VS
                0b10 => 0xA, // GE
                _ => 0xC,    // GT
            };
            let take = if self.condition_holds(cond) { rn } else { rm };
            if double {
                let v = self.vfp_d(take);
                self.set_vfp_d(rd, v);
            } else {
                let v = self.vfp_s(take);
                self.set_vfp_s(rd, v);
            }
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }

        match (insn >> 20) & 0b11 {
            // VMAXNM and VMINNM, which differ from VMAX/VMIN in returning the
            // number when one side is a NaN.
            0b00 => {
                let (a, b) = self.vfp_pair(double, rn, rm);
                let minimum = (insn >> 6) & 1 != 0;
                let result = if minimum { a.min(b) } else { a.max(b) };
                self.vfp_store(double, rd, result);
            }
            // VRINT and VCVT with the rounding mode in the instruction rather
            // than in FPSCR.
            0b11 => {
                let rounding = (insn >> 16) & 0b11;
                let v = if double {
                    self.vfp_f64(rm)
                } else {
                    f64::from(self.vfp_f32(rm))
                };
                let rounded = match rounding {
                    0b00 => v.round(), // A: ties away from zero
                    0b01 => round_ties_even(v),
                    0b10 => v.ceil(), // P
                    _ => v.floor(),   // M
                };
                if (insn >> 18) & 1 != 0 {
                    // VCVT: the result is an integer in an S register.
                    let signed = (insn >> 7) & 1 != 0;
                    let bits = if signed {
                        (rounded as i32) as u32
                    } else {
                        rounded as u32
                    };
                    self.set_vfp_s((vd << 1) | d_bit, bits);
                } else {
                    self.vfp_store(double, rd, rounded);
                }
            }
            _ => {
                return Err(Error::Cpu(format!(
                    "unimplemented ARMv8 VFP instruction {:#010x} at pc={:#010x}",
                    insn, self.pc
                )))
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }
}

/// Round half to even, which Rust's `f64::round` does not do.
fn round_ties_even(v: f64) -> f64 {
    let rounded = v.round();
    if (v - v.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - v.signum()
    } else {
        rounded
    }
}

/// Name a VFP encoding for a trace, which is worth having even where the
/// operation is not implemented: "cop p11" says nothing about what stopped a
/// run.
pub(super) fn vfp_mnemonic(insn: u32, cond: &str) -> String {
    let width = if (insn >> 8) & 0xF == 10 {
        "f32"
    } else {
        "f64"
    };
    if (insn >> 25) & 0x7 == 0b110 {
        let load = (insn >> 20) & 1 != 0;
        let base = (insn >> 16) & 0xF;
        let count = insn & 0xFF;
        let pre = (insn >> 24) & 1 != 0;
        let writeback = (insn >> 21) & 1 != 0;
        return match (pre, writeback, base) {
            (false, false, _) => format!("vmov{cond} (core pair)"),
            (true, false, _) => format!(
                "v{}r{cond}.{width} d{}, [r{base}]",
                if load { "ld" } else { "st" },
                (insn >> 12) & 0xF
            ),
            (true, true, 13) if !load => format!("vpush{cond} ({count} words)"),
            (false, true, 13) if load => format!("vpop{cond} ({count} words)"),
            _ => format!(
                "v{}m{cond} r{base}, ({count} words)",
                if load { "ld" } else { "st" }
            ),
        };
    }
    if insn & 0x10 != 0 {
        if (insn >> 21) & 0b111 == 0b111 {
            return format!(
                "vm{}s{cond} fpscr",
                if (insn >> 20) & 1 != 0 { "r" } else { "" }
            );
        }
        return format!("vmov{cond} (core)");
    }
    let name = match ((insn >> 23) & 1, (insn >> 20) & 0b11, (insn >> 6) & 1) {
        (0, 0b00, 0) => "vmla",
        (0, 0b00, 1) => "vmls",
        (0, 0b01, 0) => "vnmls",
        (0, 0b01, 1) => "vnmla",
        (0, 0b10, 0) => "vmul",
        (0, 0b10, 1) => "vnmul",
        (0, 0b11, 0) => "vadd",
        (0, 0b11, 1) => "vsub",
        (1, 0b00, _) => "vdiv",
        (1, 0b01 | 0b10, _) => "vfma",
        _ => match ((insn >> 16) & 0xF, (insn >> 6) & 0b11) {
            (_, 0b00 | 0b10) => "vmov (imm)",
            (0b0000, 0b01) => "vmov",
            (0b0000, _) => "vabs",
            (0b0001, 0b01) => "vneg",
            (0b0001, _) => "vsqrt",
            (0b0100 | 0b0101, _) => "vcmp",
            (0b0111, _) => "vcvt",
            (0b1000 | 0b1100 | 0b1101, _) => "vcvt",
            _ => "vfp",
        },
    };
    format!("{name}{cond}.{width}")
}
