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
                            // MOVK
                            let mask = (0xFFFFu64 << shift) & Self::mask(sf);
                            let cur = self.read_zr(rd) & !mask;
                            self.write_zr(rd, cur | ((imm16 << shift) & mask));
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
                    let a = self.read_zr(rn) & Self::mask(sf);
                    let r = match opc {
                        0b00 => a & mask,         // AND
                        0b01 => a | mask,         // ORR
                        0b10 => a ^ mask,         // EOR
                        _ => {
                            let r = a & mask;
                            let nbit = (r >> (if sf { 63 } else { 31 })) & 1;
                            let z = (r == 0) as u64;
                            let c = (self.nzcv >> 29) & 1;
                            let v = (self.nzcv >> 28) & 1;
                            self.nzcv =
                                ((nbit as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                            r
                        }
                    };
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
                    if opc == 0b11 {
                        self.write_zr(rd, r);
                    } else {
                        self.write_x(rd, r);
                    }
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
                    let val = self.read_zr(rn) & Self::mask(sf);
                    let cur = self.read_zr(rd) & Self::mask(sf);
                    let r = bitfield_apply(opc, val, cur, immr, imms, sf);
                    self.write_zr(rd, r);
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
                        if ((insn >> 22) & 1) == 1 || ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1
                        {
                            (0, false)
                        } else {
                            (((insn >> 10) & 0x1F), true)
                        }
                    };
                    if !ok {
                        return Err(Error::Cpu(format!("unallocated EXTR at {:#x}", self.pc)));
                    }
                    let size = if sf { 64 } else { 32 };
                    let a = self.read_zr(rn) & Self::mask(sf);
                    let b = self.read_zr(rm) & Self::mask(sf);
                    // EXTR takes the low `size` bits of `Rn:Rm >> imm`, so Rn
                    // is the *high* half. Having them the other way round made
                    // every `extr` extract from the wrong operand.
                    let r = if imm == 0 {
                        b
                    } else {
                        ((b >> imm) | (a.wrapping_shl((size as u32).wrapping_sub(imm)))) & Self::mask(sf)
                    };
                    self.write_zr(rd, r);
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
                let a = self.read_zr(rn) & Self::mask(sf);
                let mut b = self.read_zr(rm) & Self::mask(sf);
                if invert {
                    b = !b & Self::mask(sf);
                }
                let b = shift_reg(b, st, sa, sf);
                let r = match opc {
                    0b00 => a & b,
                    0b01 => a | b,
                    0b10 => a ^ b,
                    _ => {
                        let r = a & b;
                        let nbit = (r >> (if sf { 63 } else { 31 })) & 1;
                        let z = (r == 0) as u64;
                        let c = (self.nzcv >> 29) & 1;
                        let v = (self.nzcv >> 28) & 1;
                        self.nzcv = ((nbit as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                        r
                    }
                };
                self.write_zr(rd, r);
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
                                0b000000 => reverse_bits(a, size),   // RBIT
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
                        let op = (insn >> 30) & 1;
                        let imm_flag = (insn >> 11) & 1;
                        let cond = ((insn >> 12) & 0xF) as u8;
                        let nzcv = insn & 0xF;
                        let rn = ((insn >> 5) & 0x1F) as u8;
                        if !self.condition_holds(cond) {
                            self.nzcv = nzcv << 28;
                        } else {
                            let a = self.read_zr(rn) & Self::mask(sf);
                            // Bit 30: 1 = CCMP (subtract), 0 = CCMN (add). The
                            // 5-bit immediate is unsigned for both forms
                            // (QEMU-verified). The subtract needs the
                            // +carry_in the borrow implies, so carry_in is 1
                            // for CCMP and 0 for CCMN.
                            let b = if imm_flag == 1 {
                                ((insn >> 16) & 0x1F) as u64
                            } else {
                                self.read_zr(((insn >> 16) & 0x1F) as u8)
                            };
                            self.set_nzcv_from_compare(a, b, op == 1, op as u64, sf);
                        }
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
                        let a = self.read_zr(rn) & Self::mask(sf);
                        let b = self.read_zr(rm) & Self::mask(sf);
                        let take_a = self.condition_holds(cond);
                        let mut else_val = b;
                        if else_inv {
                            else_val = !else_val;
                        }
                        if else_inc {
                            else_val = else_val.wrapping_add(1);
                        }
                        let r = if take_a { a } else { else_val };
                        self.write_zr(rd, r & Self::mask(sf));
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
                let a = self.read_zr(rn);
                let b = self.read_zr(rm);
                match (insn >> 21) & 0xFF {
                    // MADD / MSUB (bits[28:21] == 11011000), 32- and 64-bit.
                    0b11011000 => {
                        let sf = (insn >> 31) & 1;
                        let mask = Self::mask(sf != 0);
                        let a = a & mask;
                        let b = b & mask;
                        let c = self.read_zr(ra) & mask;
                        let product = a.wrapping_mul(b);
                        let r = if o0 {
                            c.wrapping_sub(product)
                        } else {
                            c.wrapping_add(product)
                        };
                        self.write_zr(rd, r & mask);
                    }
                    // SMADDL / SMSUBL: the multiplicands are the low 32 bits
                    // of Rn/Rm, sign-extended — not the whole register.
                    0b11011001 => {
                        let product =
                            ((i128::from(a as u32 as i32)) * (i128::from(b as u32 as i32))) as u64;
                        let c = self.read_zr(ra);
                        let r = if o0 {
                            c.wrapping_sub(product)
                        } else {
                            c.wrapping_add(product)
                        };
                        self.write_zr(rd, r);
                    }
                    // UMADDL / UMSUBL: the low 32 bits of Rn/Rm, zero-extended.
                    0b11011101 => {
                        let product = (u128::from(a as u32) * u128::from(b as u32)) as u64;
                        let c = self.read_zr(ra);
                        let r = if o0 {
                            c.wrapping_sub(product)
                        } else {
                            c.wrapping_add(product)
                        };
                        self.write_zr(rd, r);
                    }
                    // SMULH: top 64 bits of the signed 128-bit product.
                    0b11011010 => {
                        let product = ((a as i64 as i128) * (b as i64 as i128)) >> 64;
                        self.write_zr(rd, product as u64);
                    }
                    // UMULH: top 64 bits of the unsigned 128-bit product.
                    0b11011110 => {
                        let product = (a as u128 * b as u128) >> 64;
                        self.write_zr(rd, product as u64);
                    }
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
}
