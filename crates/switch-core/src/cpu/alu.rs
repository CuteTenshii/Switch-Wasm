//! Data processing: the immediate and register forms of the integer ALU,
//! bitfield, shift, multiply/divide, conditional-select and compare groups.

use super::bits::*;
use super::Cpu;
use crate::{Error, Result};

impl Cpu {
    pub(super) fn try_data_proc_imm(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        let grp = (insn >> 24) & 0x1F;
        let sf = (insn >> 31) & 1 == 1;
        match grp {
            0b10000 => {
                // ADR/ADRP (handled earlier, defensive)
                Ok(true)
            }
            0b10001 => {
                // ADD/SUB immediate
                if ((insn >> 23) & 1) == 1 {
                    return Err(Error::Cpu(format!(
                        "unimplemented ADDG/SUBG at {:#x}",
                        self.pc
                    )));
                }
                let op = (insn >> 29) & 0b11;
                let sh = (insn >> 22) & 1;
                let imm12 = ((insn >> 10) & 0xFFF) as u64;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let imm = if sh == 1 { imm12 << 12 } else { imm12 };
                // op bit1 selects ADD/SUB, bit0 selects the S (flags) form:
                // ADD=00, ADDS=01, SUB=10, SUBS=11.
                let sub = (op >> 1) == 1;
                let set_flags = (op & 1) == 1;
                self.add_sub(rd, rn, imm, set_flags, sub, sf, true);
                Ok(true)
            }
            0b10010 => {
                if ((insn >> 23) & 1) == 1 {
                    // MOVN/MOVZ/MOVK
                    let opc = (insn >> 29) & 0b11;
                    let rd = (insn & 0x1F) as u8;
                    let imm16 = ((insn >> 5) & 0xFFFF) as u64;
                    let hw = if sf {
                        (insn >> 21) & 0b11
                    } else {
                        // 32-bit forms encode the shift in bit 21 (bit 22 is
                        // part of the fixed 100101 pattern).
                        (insn >> 21) & 1
                    };
                    let shift = hw * 16;
                    match opc {
                        0b00 => {
                            // MOVN
                            let v = !(imm16 << shift) & Self::mask(sf);
                            self.write_zr(rd, v);
                        }
                        0b10 => {
                            // MOVZ
                            self.write_zr(rd, (imm16 << shift) & Self::mask(sf));
                        }
                        0b11 => {
                            self.movk(Self::zr_write_slot(rd), shift as u8, imm16 as u16, sf);
                        }
                        _ => {
                            return Err(Error::Cpu(format!(
                                "unimplemented MOV wide opc {} at {:#x}",
                                opc, self.pc
                            )))
                        }
                    }
                    Ok(true)
                } else {
                    // Logical immediate
                    let opc = (insn >> 29) & 0b11;
                    let n = (insn >> 22) & 1;
                    let immr = (insn >> 16) & 0x3F;
                    let imms = (insn >> 10) & 0x3F;
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let mask = decode_bit_mask(sf, n, immr, imms).ok_or_else(|| {
                        Error::Cpu(format!("unallocated logical immediate at {:#x}", self.pc))
                    })?;
                    // `Rd == 31` is **SP** for AND/ORR/EOR, and the zero
                    // register only for ANDS -- the immediate forms are among
                    // the handful where the two differ, unlike the
                    // shifted-register forms right below where 31 is always
                    // ZR. `write_zr` for all four threw away every
                    // `and sp, xN, #imm`, which is how LLVM aligns a stack
                    // frame it has just made room in:
                    //
                    //     sub x9, sp, #0x260
                    //     and sp, x9, #0xffffffffffffffc0
                    //
                    // With the second one discarded the frame is never
                    // allocated, and every local the function then writes
                    // lands 0x260 bytes too high -- straight over the
                    // register save area it just filled in.
                    let rd = if opc == 0b11 {
                        Self::zr_write_slot(rd)
                    } else {
                        Self::x_slot(rd)
                    };
                    self.logical(rd, rn, mask, opc as u8, sf);
                    Ok(true)
                }
            }
            0b10011 => {
                if ((insn >> 23) & 1) == 0 {
                    // Bitfield move
                    let opc = (insn >> 29) & 0b11;
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let (immr, imms) = if sf {
                        if ((insn >> 22) & 1) != 1 {
                            return Err(Error::Cpu(format!(
                                "unallocated bitfield N at {:#x}",
                                self.pc
                            )));
                        }
                        (((insn >> 16) & 0x3F), ((insn >> 10) & 0x3F))
                    } else {
                        if ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1 {
                            return Err(Error::Cpu(format!(
                                "unallocated 32-bit bitfield at {:#x}",
                                self.pc
                            )));
                        }
                        (((insn >> 16) & 0x1F), ((insn >> 10) & 0x1F))
                    };
                    self.bitfield(
                        Self::zr_write_slot(rd),
                        rn,
                        opc as u8,
                        immr as u8,
                        imms as u8,
                        sf,
                    );
                    Ok(true)
                } else {
                    // EXTR
                    let rn = ((insn >> 5) & 0x1F) as u8;
                    let rd = (insn & 0x1F) as u8;
                    let rm = ((insn >> 16) & 0x1F) as u8;
                    let (imm, ok) = if sf {
                        if ((insn >> 22) & 1) != 1 || ((insn >> 21) & 1) == 1 {
                            (0, false)
                        } else {
                            (((insn >> 10) & 0x3F), true)
                        }
                    } else {
                        if ((insn >> 22) & 1) == 1
                            || ((insn >> 21) & 1) == 1
                            || ((insn >> 15) & 1) == 1
                        {
                            (0, false)
                        } else {
                            (((insn >> 10) & 0x1F), true)
                        }
                    };
                    if !ok {
                        return Err(Error::Cpu(format!("unallocated EXTR at {:#x}", self.pc)));
                    }
                    self.extr(Self::zr_write_slot(rd), rn, rm, imm as u8, sf);
                    Ok(true)
                }
            }
            _ => Ok(false),
        }
    }

    // ---------- data processing: register ----------

    #[allow(clippy::too_many_lines)]
    pub(super) fn try_data_proc_reg(&mut self, insn: u32, _next_pc: &mut u32) -> Result<bool> {
        let grp = (insn >> 24) & 0x1F;
        let sf = (insn >> 31) & 1 == 1;
        match grp {
            0b01010 => {
                // Logical shifted register
                let opc = (insn >> 29) & 0b11;
                let st = (insn >> 22) & 0b11;
                let invert = ((insn >> 21) & 1) == 1;
                let rm = ((insn >> 16) & 0x1F) as u8;
                let sa = (insn >> 10) & 0x3F;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let b = shift_reg(self.read_zr(rm) & Self::mask(sf), st, sa, sf);
                // `BIC`/`ORN`/`EON` invert the shifted operand, not the
                // register: `ir.Not(ShiftReg(...))` in dynarmic, and the same
                // order in the ARM ARM. Inverting first only agreed with that
                // when the shift amount was zero.
                let b = if invert { !b & Self::mask(sf) } else { b };
                self.logical(Self::zr_write_slot(rd), rn, b, opc as u8, sf);
                Ok(true)
            }
            0b01011 => {
                // ADD/SUB shifted or extended. op bit1 selects ADD/SUB,
                // bit0 the S (flags) form: ADD=00, ADDS=01, SUB=10, SUBS=11.
                let op = (insn >> 29) & 0b11;
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let rm = ((insn >> 16) & 0x1F) as u8;
                let sub = (op >> 1) == 1;
                let set_flags = (op & 1) == 1;
                if ((insn >> 21) & 0b111) == 0b001 {
                    // Extended register
                    let option = ((insn >> 13) & 0b111) as u8;
                    let shift = (insn >> 10) & 0b111;
                    let v = extend_reg(self.read_zr(rm), option, sf) & Self::mask(sf);
                    let v = v.wrapping_shl(shift) & Self::mask(sf);
                    self.add_sub(rd, rn, v, set_flags, sub, sf, true);
                } else {
                    // Shifted register
                    let st = (insn >> 22) & 0b11;
                    let sa = (insn >> 10) & 0x3F;
                    let v = shift_reg(self.read_zr(rm) & Self::mask(sf), st, sa, sf);
                    self.add_sub(rd, rn, v, set_flags, sub, sf, false);
                }
                Ok(true)
            }
            0b11010 => {
                if ((insn >> 22) & 1) == 1 {
                    if ((insn >> 23) & 1) == 1 {
                        // 2-source or 1-source (bits[28:21]=11010110)
                        let opcode2 = (insn >> 10) & 0x3F;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        if ((insn >> 29) & 0b11) == 0b00 {
                            // 2-source (bits[30:29]=00)
                            let a = self.read_zr(rn) & Self::mask(sf);
                            let b = self.read_zr(rm) & Self::mask(sf);
                            let r = match opcode2 {
                                0b000010 => {
                                    // UDIV. Division by zero gives 0 (no trap).
                                    a.checked_div(b).unwrap_or(0) & Self::mask(sf)
                                }
                                0b000011 => {
                                    // SDIV. The operands have to be sign-extended
                                    // from *their own* width — using the masked
                                    // 32-bit values as positive i64 turned
                                    // `sdiv w9, w10, w11` into an unsigned
                                    // divide. INT_MIN / -1 wraps rather than
                                    // trapping.
                                    let size = if sf { 64 } else { 32 };
                                    let x = sext_u64(a, size) as i64;
                                    let y = sext_u64(b, size) as i64;
                                    let q = if y == 0 { 0 } else { x.wrapping_div(y) };
                                    (q as u64) & Self::mask(sf)
                                }
                                0b001000 => shift_var(a, b, 0, sf),
                                0b001001 => shift_var(a, b, 1, sf),
                                0b001010 => shift_var(a, b, 2, sf),
                                0b001011 => {
                                    // RORV
                                    let size = if sf { 64 } else { 32 };
                                    let amt = (b % size) as u32;
                                    if sf {
                                        a.rotate_right(amt)
                                    } else {
                                        (a as u32).rotate_right(amt) as u64
                                    }
                                }
                                0b010000..=0b010111 => {
                                    // CRC32/CRC32C. The accumulator and the
                                    // result are always 32-bit; only the
                                    // doubleword form reads a full 64-bit Rm,
                                    // and it is the only one encoded with sf
                                    // set.
                                    let sz = opcode2 & 0b11;
                                    if (sz == 0b11) != sf {
                                        return Err(Error::Cpu(format!(
                                            "malformed CRC32 operand size at {:#x}",
                                            self.pc
                                        )));
                                    }
                                    let castagnoli = ((opcode2 >> 2) & 1) == 1;
                                    u64::from(crc32(a as u32, b, 8 << sz, castagnoli))
                                }
                                _ => {
                                    return Err(Error::Cpu(format!(
                                        "unimplemented 2-source opcode {} at {:#x}",
                                        opcode2, self.pc
                                    )))
                                }
                            };
                            self.write_zr(rd, r);
                        } else if ((insn >> 29) & 0b11) == 0b10 {
                            // 1-source (bits[30:29]=10)
                            let a = self.read_zr(rn) & Self::mask(sf);
                            let size = if sf { 64 } else { 32 };
                            let r = match opcode2 {
                                0b000000 => reverse_bits(a, size),     // RBIT
                                0b000001 => reverse_16_lanes(a, size), // REV16
                                0b000010 => reverse_32_lanes(a, size), // REV32
                                0b000011 => {
                                    // REV64 (64-bit only)
                                    a.swap_bytes()
                                }
                                0b000100 => clz(a, size),
                                0b000101 => cls(a, size),
                                0b000110 => ctz(a, size),
                                _ => {
                                    return Err(Error::Cpu(format!(
                                        "unimplemented 1-source opcode {} at {:#x}",
                                        opcode2, self.pc
                                    )))
                                }
                            };
                            self.write_zr(rd, r & Self::mask(sf));
                        } else {
                            return Err(Error::Cpu(format!(
                                "unimplemented data-processing op at {:#x}",
                                self.pc
                            )));
                        }
                        Ok(true)
                    } else {
                        // CCMP / CCMN
                        // Bit 30 selects CCMP (subtract) over CCMN (add),
                        // and bit 11 the immediate form. Rm and the immediate
                        // are the same field, read either as a register or as
                        // the value itself.
                        let field = ((insn >> 16) & 0x1F) as u8;
                        self.cond_cmp(
                            ((insn >> 5) & 0x1F) as u8,
                            field,
                            field,
                            ((insn >> 12) & 0xF) as u8,
                            (insn & 0xF) as u8,
                            ((insn >> 30) & 1) == 1,
                            ((insn >> 11) & 1) == 1,
                            sf,
                        );
                        Ok(true)
                    }
                } else {
                    if ((insn >> 23) & 1) == 1 {
                        // CSEL family: csel / csinc / csinv / csneg.
                        // The invert/increment are part of the *else* value,
                        // not applied to the selected value.
                        let else_inv = ((insn >> 30) & 1) == 1;
                        let else_inc = ((insn >> 10) & 1) == 1;
                        let cond = ((insn >> 12) & 0xF) as u8;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        self.cond_sel(
                            Self::zr_write_slot(rd),
                            rn,
                            rm,
                            cond,
                            else_inv,
                            else_inc,
                            sf,
                        );
                    } else {
                        // ADC / ADCS / SBC / SBCS
                        let op = (insn >> 29) & 0b11;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        let rd = (insn & 0x1F) as u8;
                        let rm = ((insn >> 16) & 0x1F) as u8;
                        let carry_in = ((self.nzcv >> 29) & 1) as u64;
                        let a = self.read_zr(rn) & Self::mask(sf);
                        let b = self.read_zr(rm) & Self::mask(sf);
                        // bit30 = subtract (SBC), bit29 = S. Reading them the
                        // other way round made `adcs` subtract and `ngc` negate
                        // the wrong operand.
                        let _ = op;
                        let sub = ((insn >> 30) & 1) == 1;
                        let set_flags = ((insn >> 29) & 1) == 1;
                        let (result, carry, overflow) = if sub {
                            Self::add_carry_overflow(a, !b, carry_in, sf)
                        } else {
                            Self::add_carry_overflow(a, b, carry_in, sf)
                        };
                        if set_flags {
                            self.set_nzcv_from_alu(result, sf, carry, overflow);
                        }
                        self.write_zr(rd, result);
                    }
                    Ok(true)
                }
            }
            0b11011 => {
                // Data processing (3-source) / multiply.
                let rn = ((insn >> 5) & 0x1F) as u8;
                let rd = (insn & 0x1F) as u8;
                let rm = ((insn >> 16) & 0x1F) as u8;
                let ra = ((insn >> 10) & 0x1F) as u8;
                let o0 = ((insn >> 15) & 1) == 1;
                let rd = Self::zr_write_slot(rd);
                match (insn >> 21) & 0xFF {
                    // MADD / MSUB (bits[28:21] == 11011000), 32- and 64-bit.
                    0b11011000 => self.madd(rd, rn, rm, ra, o0, sf),
                    // SMADDL / SMSUBL: the multiplicands are the low 32 bits
                    // of Rn/Rm, sign-extended — not the whole register.
                    0b11011001 => self.madd_long(rd, rn, rm, ra, o0, true),
                    // UMADDL / UMSUBL: the low 32 bits of Rn/Rm, zero-extended.
                    0b11011101 => self.madd_long(rd, rn, rm, ra, o0, false),
                    // SMULH: top 64 bits of the signed 128-bit product.
                    0b11011010 => self.mulh(rd, rn, rm, true),
                    // UMULH: top 64 bits of the unsigned 128-bit product.
                    0b11011110 => self.mulh(rd, rn, rm, false),
                    _ => {
                        return Err(Error::Cpu(format!(
                            "unimplemented multiply-long at {:#x}",
                            self.pc
                        )));
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    // ---- one implementation per instruction ----

    // The bodies below are the instructions themselves, with their operands
    // already resolved to register-file slots (see `Cpu::x_slot` and
    // `Cpu::zr_write_slot`). The interpreter resolves them from the encoding
    // on every execution and the block translator resolves them once when it
    // builds an `Op`, but from here down there is one implementation — so the
    // two engines cannot compute an instruction differently, which is the
    // drift `examples/jit_difftest.rs` exists to find.
    //
    // A destination is always the *write* slot, so an instruction that reads
    // its destination back (`MOVK`, `BFM`) reads through that same slot. For
    // register 31 that is the discard slot rather than the zero one, which is
    // unobservable: the result is discarded too.

    /// `MOVK`: replace the 16-bit field at `shift`, leaving the rest alone.
    ///
    /// The field itself never reaches above bit 31 in a 32-bit form, but the
    /// register it merges into does — and a write to a W register zeroes bits
    /// 63:32. Without the narrowing, `movk w0, #0x1234` over an all-ones
    /// register left `ffffffffffff1234` where hardware gives `00000000ffff1234`
    /// (`tools/difftest.py --scalar`).
    #[inline(always)]
    pub(super) fn movk(&mut self, rd: u8, shift: u8, val: u16, sf: bool) {
        let mask = 0xFFFFu64 << shift;
        let cur = self.reg_at(rd) & !mask;
        self.set_reg_at(rd, (cur | (u64::from(val) << shift)) & Self::mask(sf));
    }

    /// `AND`/`ORR`/`EOR`/`ANDS`. The immediate, shifted-register and plain
    /// register forms differ only in how `b` is formed, so they share this.
    /// `ANDS` is the one that writes flags, and it leaves C and V alone.
    #[inline(always)]
    pub(super) fn logical(&mut self, rd: u8, rn: u8, b: u64, opc: u8, sf: bool) {
        let a = self.reg_at(rn) & Self::mask(sf);
        let r = match opc {
            0b00 => a & b,
            0b01 => a | b,
            0b10 => a ^ b,
            _ => {
                let r = a & b;
                let n = (r >> (if sf { 63 } else { 31 })) & 1;
                let z = u64::from(r == 0);
                let c = (self.nzcv >> 29) & 1;
                let v = (self.nzcv >> 28) & 1;
                self.nzcv = ((n as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                r
            }
        };
        self.set_reg_at(rd, r);
    }

    /// `SBFM`/`BFM`/`UBFM` and the aliases built on them.
    #[inline(always)]
    pub(super) fn bitfield(&mut self, rd: u8, rn: u8, opc: u8, immr: u8, imms: u8, sf: bool) {
        let val = self.reg_at(rn) & Self::mask(sf);
        let cur = self.reg_at(rd) & Self::mask(sf);
        let r = bitfield_apply(
            u32::from(opc),
            val,
            cur,
            u32::from(immr),
            u32::from(imms),
            sf,
        );
        self.set_reg_at(rd, r);
    }

    /// `EXTR`: the low `size` bits of `Rn:Rm >> imm`, so Rn is the *high*
    /// half. Having them the other way round extracts from the wrong operand.
    #[inline(always)]
    pub(super) fn extr(&mut self, rd: u8, rn: u8, rm: u8, imm: u8, sf: bool) {
        let size = if sf { 64u32 } else { 32 };
        let a = self.reg_at(rn) & Self::mask(sf);
        let b = self.reg_at(rm) & Self::mask(sf);
        let imm = u32::from(imm);
        let r = if imm == 0 {
            b
        } else {
            ((b >> imm) | a.wrapping_shl(size.wrapping_sub(imm))) & Self::mask(sf)
        };
        self.set_reg_at(rd, r);
    }

    /// `CSEL`/`CSINC`/`CSINV`/`CSNEG`. The invert and the increment are part
    /// of the *else* value, not applied to the selected one.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cond_sel(
        &mut self,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: u8,
        else_inv: bool,
        else_inc: bool,
        sf: bool,
    ) {
        let a = self.reg_at(rn) & Self::mask(sf);
        let b = self.reg_at(rm) & Self::mask(sf);
        let take_a = self.condition_holds(cond);
        let mut else_val = b;
        if else_inv {
            else_val = !else_val;
        }
        if else_inc {
            else_val = else_val.wrapping_add(1);
        }
        let r = if take_a { a } else { else_val };
        self.set_reg_at(rd, r & Self::mask(sf));
    }

    /// `CCMP`/`CCMN`, register and immediate forms. `sub` is CCMP, whose
    /// borrow implies the carry-in; the 5-bit immediate is unsigned for both
    /// forms (QEMU-verified). When the condition fails the instruction just
    /// installs the flags it carries.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cond_cmp(
        &mut self,
        rn: u8,
        rm: u8,
        imm: u8,
        cond: u8,
        nzcv: u8,
        sub: bool,
        is_imm: bool,
        sf: bool,
    ) {
        if self.condition_holds(cond) {
            let a = self.reg_at(rn) & Self::mask(sf);
            let b = if is_imm {
                u64::from(imm)
            } else {
                self.reg_at(rm)
            };
            self.set_nzcv_from_compare(a, b, sub, u64::from(sub), sf);
        } else {
            self.nzcv = u32::from(nzcv) << 28;
        }
    }

    /// `MADD`/`MSUB`.
    #[inline(always)]
    pub(super) fn madd(&mut self, rd: u8, rn: u8, rm: u8, ra: u8, sub: bool, sf: bool) {
        let mask = Self::mask(sf);
        let product = (self.reg_at(rn) & mask).wrapping_mul(self.reg_at(rm) & mask);
        let c = self.reg_at(ra) & mask;
        let r = if sub {
            c.wrapping_sub(product)
        } else {
            c.wrapping_add(product)
        };
        self.set_reg_at(rd, r & mask);
    }

    /// `SMADDL`/`SMSUBL`/`UMADDL`/`UMSUBL`: the multiplicands are the low 32
    /// bits of Rn/Rm, not the whole register. A 32x32 product fits in 64 bits,
    /// so this does not need the 128-bit arithmetic wasm has to synthesize.
    #[inline(always)]
    pub(super) fn madd_long(&mut self, rd: u8, rn: u8, rm: u8, ra: u8, sub: bool, signed: bool) {
        let a = self.reg_at(rn);
        let b = self.reg_at(rm);
        let product = if signed {
            i64::from(a as u32 as i32).wrapping_mul(i64::from(b as u32 as i32)) as u64
        } else {
            u64::from(a as u32).wrapping_mul(u64::from(b as u32))
        };
        let c = self.reg_at(ra);
        let r = if sub {
            c.wrapping_sub(product)
        } else {
            c.wrapping_add(product)
        };
        self.set_reg_at(rd, r);
    }

    /// `SMULH`/`UMULH`: the top 64 bits of the 128-bit product.
    #[inline(always)]
    pub(super) fn mulh(&mut self, rd: u8, rn: u8, rm: u8, signed: bool) {
        let a = self.reg_at(rn);
        let b = self.reg_at(rm);
        let r = if signed {
            (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64
        } else {
            ((u128::from(a) * u128::from(b)) >> 64) as u64
        };
        self.set_reg_at(rd, r);
    }
}
