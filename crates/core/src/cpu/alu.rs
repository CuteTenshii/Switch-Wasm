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
                // op: bit1 = SUB, bit0 = S (flags).
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
                        // 32-bit forms encode the shift in bit 21.
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
                    // Rd == 31 is SP for AND/ORR/EOR, and ZR only for ANDS.
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
                if sa == 0 && opc == 1 && !invert && rn == 31 {
                    self.copy_reg(Self::zr_write_slot(rd), rm, sf);
                    return Ok(true);
                }
                let b = shift_reg(self.read_zr(rm) & Self::mask(sf), st, sa, sf);
                // BIC/ORN/EON invert the shifted operand, not the register.
                let b = if invert { !b & Self::mask(sf) } else { b };
                self.logical(Self::zr_write_slot(rd), rn, b, opc as u8, sf);
                Ok(true)
            }
            0b01011 => {
                // ADD/SUB shifted or extended: bit1 = SUB, bit0 = S.
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
                            let rd_slot = Self::zr_write_slot(rd);
                            match opcode2 {
                                0b000010 | 0b000011 => {
                                    self.divide(rd_slot, rn, rm, opcode2 & 1 == 1, sf)
                                }
                                0b001000..=0b001011 => {
                                    self.shift_by_reg(rd_slot, rn, rm, (opcode2 & 0b11) as u8, sf)
                                }
                                0b010000..=0b010111 => {
                                    // CRC32/CRC32C; only the X form reads a 64-bit Rm.
                                    let sz = opcode2 & 0b11;
                                    if (sz == 0b11) != sf {
                                        return Err(Error::Cpu(format!(
                                            "malformed CRC32 operand size at {:#x}",
                                            self.pc
                                        )));
                                    }
                                    self.crc(rd_slot, rn, rm, sz as u8, ((opcode2 >> 2) & 1) == 1);
                                }
                                _ => {
                                    return Err(Error::Cpu(format!(
                                        "unimplemented 2-source opcode {} at {:#x}",
                                        opcode2, self.pc
                                    )))
                                }
                            }
                        } else if ((insn >> 29) & 0b11) == 0b10 {
                            // 1-source (bits[30:29]=10)
                            if opcode2 > 0b000110 {
                                return Err(Error::Cpu(format!(
                                    "unimplemented 1-source opcode {} at {:#x}",
                                    opcode2, self.pc
                                )));
                            }
                            self.one_source(Self::zr_write_slot(rd), rn, opcode2 as u8, sf);
                        } else {
                            return Err(Error::Cpu(format!(
                                "unimplemented data-processing op at {:#x}",
                                self.pc
                            )));
                        }
                        Ok(true)
                    } else {
                        // CCMP / CCMN: bit 30 = CCMP, bit 11 = immediate form.
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
                        // CSEL family: invert/increment apply to the else value.
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
                        // ADC / ADCS / SBC / SBCS: bit 30 subtracts, bit 29 sets flags.
                        self.adc(
                            Self::zr_write_slot((insn & 0x1F) as u8),
                            ((insn >> 5) & 0x1F) as u8,
                            ((insn >> 16) & 0x1F) as u8,
                            ((insn >> 30) & 1) == 1,
                            ((insn >> 29) & 1) == 1,
                            sf,
                        );
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
                    // SMADDL / SMSUBL: low 32 bits of Rn/Rm, sign-extended.
                    0b11011001 => self.madd_long(rd, rn, rm, ra, o0, true),
                    // UMADDL / UMSUBL: low 32 bits of Rn/Rm, zero-extended.
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

    // Shared instruction bodies, operands already resolved to register-file slots.

    /// `MOVK`: replace the 16-bit field at `shift`, leaving the rest alone.
    #[inline(always)]
    pub(super) fn movk(&mut self, rd: u8, shift: u8, val: u16, sf: bool) {
        let mask = 0xFFFFu64 << shift;
        let cur = self.reg_at(rd) & !mask;
        self.set_reg_at(rd, (cur | (u64::from(val) << shift)) & Self::mask(sf));
    }

    /// `AND`/`ORR`/`EOR`/`ANDS`; `ANDS` leaves C and V alone.
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

    /// `SBFM`/`BFM`/`UBFM`; the unallocated `opc` writes zero.
    #[inline(always)]
    pub(super) fn bitfield(&mut self, rd: u8, rn: u8, opc: u8, immr: u8, imms: u8, sf: bool) {
        let (immr, imms) = (u32::from(immr), u32::from(imms));
        if let Some(extract) = Extract::of(u32::from(opc), immr, imms, sf) {
            self.extract(rd, rn, extract, sf);
            return;
        }
        let r = if opc == 0b01 {
            let val = self.reg_at(rn) & Self::mask(sf);
            let cur = self.reg_at(rd) & Self::mask(sf);
            bitfield_insert(val, cur, immr, imms, sf)
        } else {
            0
        };
        self.set_reg_at(rd, r);
    }

    #[inline(always)]
    pub(super) fn extract(&mut self, rd: u8, rn: u8, extract: Extract, sf: bool) {
        let r = extract.apply(self.reg_at(rn), sf);
        self.set_reg_at(rd, r);
    }

    /// `LSLV`/`LSRV`/`ASRV`/`RORV`: `kind` is the shift type, 3 = rotate.
    #[inline(always)]
    pub(super) fn shift_by_reg(&mut self, rd: u8, rn: u8, rm: u8, kind: u8, sf: bool) {
        let a = self.reg_at(rn) & Self::mask(sf);
        let b = self.reg_at(rm) & Self::mask(sf);
        self.set_reg_at(rd, shift_var(a, b, u32::from(kind), sf));
    }

    #[inline(always)]
    pub(super) fn copy_reg(&mut self, rd: u8, rn: u8, sf: bool) {
        self.set_reg_at(rd, self.reg_at(rn) & Self::mask(sf));
    }

    /// `RBIT`/`REV16`/`REV32`/`REV`/`CLZ`/`CLS`/`CTZ` by one-source `opcode`.
    #[inline(always)]
    pub(super) fn one_source(&mut self, rd: u8, rn: u8, opcode: u8, sf: bool) {
        let size = if sf { 64 } else { 32 };
        let a = self.reg_at(rn) & Self::mask(sf);
        let r = match opcode {
            0b000 => reverse_bits(a, size),
            0b001 => reverse_16_lanes(a, size),
            0b010 => reverse_32_lanes(a, size),
            0b011 => a.swap_bytes(),
            0b100 => clz(a, size),
            0b101 => cls(a, size),
            _ => ctz(a, size),
        };
        self.set_reg_at(rd, r & Self::mask(sf));
    }

    /// `CRC32`/`CRC32C` over `8 << sz` bits of Rm.
    #[inline(always)]
    pub(super) fn crc(&mut self, rd: u8, rn: u8, rm: u8, sz: u8, castagnoli: bool) {
        let acc = self.reg_at(rn) as u32;
        let r = crc32(acc, self.reg_at(rm), 8 << sz, castagnoli);
        self.set_reg_at(rd, u64::from(r));
    }

    #[inline(always)]
    pub(super) fn adc(&mut self, rd: u8, rn: u8, rm: u8, sub: bool, set_flags: bool, sf: bool) {
        let carry = u64::from((self.nzcv >> 29) & 1);
        let a = self.reg_at(rn) & Self::mask(sf);
        let b = self.reg_at(rm) & Self::mask(sf);
        let (result, c, v) = if sub {
            Self::add_carry_overflow(a, !b, carry, sf)
        } else {
            Self::add_carry_overflow(a, b, carry, sf)
        };
        if set_flags {
            self.set_nzcv_from_alu(result, sf, c, v);
        }
        self.set_reg_at(rd, result);
    }

    /// `UDIV`/`SDIV`: division by zero gives 0, `INT_MIN / -1` wraps.
    #[inline(always)]
    pub(super) fn divide(&mut self, rd: u8, rn: u8, rm: u8, signed: bool, sf: bool) {
        let a = self.reg_at(rn) & Self::mask(sf);
        let b = self.reg_at(rm) & Self::mask(sf);
        let q = if signed {
            // Operands are sign-extended from their own width.
            let size = if sf { 64 } else { 32 };
            let x = sext_u64(a, size) as i64;
            let y = sext_u64(b, size) as i64;
            if y == 0 {
                0
            } else {
                x.wrapping_div(y) as u64
            }
        } else {
            a.checked_div(b).unwrap_or(0)
        };
        self.set_reg_at(rd, q & Self::mask(sf));
    }

    /// `EXTR`: low `size` bits of `Rn:Rm >> imm` (Rn is the high half).
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

    /// `CCMP`/`CCMN`; when the condition fails the flags come from the immediate.
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

    /// `SMADDL`/`SMSUBL`/`UMADDL`/`UMSUBL` on the low 32 bits of Rn/Rm.
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
