//! A32 data processing: the shifter-fed ALU, the multiplies, and the
//! miscellaneous branch/status group that shares their encoding space.

use super::shift::{decode_imm_shift, expand_imm_c, shift_c};
use crate::cpu::Cpu;
use crate::{Error, Result};

impl Cpu {
    /// The `op1 == 00x` encoding space: data processing with an immediate or a
    /// shifted register, and the multiply, status-register and interworking
    /// forms that share it.
    pub(super) fn a32_data_processing(&mut self, insn: u32) -> Result<()> {
        let immediate = (insn >> 25) & 1 != 0;
        // Bits 24:23 == 10 with S clear is not an ALU operation at all: the
        // architecture reuses the four opcodes that would be TST/TEQ/CMP/CMN
        // without flags for the status-register, interworking and saturating
        // group.
        let misc_slot = (insn >> 23) & 0b11 == 0b10 && (insn >> 20) & 1 == 0;
        if !immediate {
            if insn & 0x90 == 0x90 {
                // With bits 6:5 set this is a halfword or doubleword access;
                // otherwise bit 24 alone tells a multiply from a
                // synchronisation primitive. `LDREX` is 0001 1001 and `STREX`
                // 0001 1000, so both sit outside the 10xx0 slot the
                // miscellaneous group uses and testing for that sends every
                // exclusive pair to the multiplier.
                return if insn & 0x60 != 0 {
                    self.a32_extra_load_store(insn)
                } else if (insn >> 24) & 1 != 0 {
                    self.a32_sync(insn)
                } else {
                    self.a32_multiply(insn)
                };
            }
            if misc_slot {
                // Bit 7 splits the two, not bit 4: the miscellaneous group
                // runs `op2` 0000..0111, so `BX` (0001) and `BLX` (0011) have
                // bit 4 set and are still misc. Testing bit 4 sends every
                // `bx lr` to the halfword multiplier, which returns nowhere
                // and falls into the next function.
                return if insn & 0x80 == 0 {
                    self.a32_misc(insn)
                } else {
                    self.a32_halfword_multiply(insn)
                };
            }
        } else if misc_slot {
            // Bit 21 splits the wide moves from the status-register writes:
            // `MOVW` is 0011 0000 and `MOVT` 0011 0100, against `MSR`
            // immediate's 0011 x010.
            return if (insn >> 21) & 1 == 0 {
                self.a32_move_wide(insn)
            } else {
                self.a32_msr_imm_or_hint(insn)
            };
        }

        let carry_in = self.carry_flag();
        let (operand, shifter_carry) = if immediate {
            expand_imm_c(insn & 0xFFF, carry_in)
        } else {
            let rm = self.r32((insn & 0xF) as u8);
            if (insn >> 4) & 1 == 0 {
                let (ty, amount) =
                    decode_imm_shift(((insn >> 5) & 0b11) as u8, ((insn >> 7) & 0x1F) as u8);
                shift_c(rm, ty, amount, carry_in)
            } else {
                // Shift by register: only the bottom byte counts, and a zero
                // shift leaves both the value and the carry alone. `Rm` reads
                // as pc+12 here, since this is the one form with a fifth
                // register operand, but r15 is unpredictable as any operand
                // of it, so the ordinary pc+8 is what the read above gave.
                let amount = self.r32(((insn >> 8) & 0xF) as u8) & 0xFF;
                if amount == 0 {
                    (rm, carry_in)
                } else {
                    shift_c(rm, ((insn >> 5) & 0b11) as u8, amount, carry_in)
                }
            }
        };

        let op = (insn >> 21) & 0xF;
        let set_flags = (insn >> 20) & 1 != 0;
        let rn = ((insn >> 16) & 0xF) as u8;
        let rd = ((insn >> 12) & 0xF) as u8;
        let a = self.r32(rn);

        // The four comparisons write no register; everything else does.
        let (result, writes) = match op {
            0x0 => (a & operand, true),
            0x1 => (a ^ operand, true),
            0x2 => (self.add32_flags(a, !operand, true, set_flags), true),
            0x3 => (self.add32_flags(!a, operand, true, set_flags), true),
            0x4 => (self.add32_flags(a, operand, false, set_flags), true),
            0x5 => (self.add32_flags(a, operand, carry_in, set_flags), true),
            0x6 => (self.add32_flags(a, !operand, carry_in, set_flags), true),
            0x7 => (self.add32_flags(!a, operand, carry_in, set_flags), true),
            0x8 => (a & operand, false),
            0x9 => (a ^ operand, false),
            0xA => (self.add32_flags(a, !operand, true, true), false),
            0xB => (self.add32_flags(a, operand, false, true), false),
            0xC => (a | operand, true),
            0xD => (operand, true),
            0xE => (a & !operand, true),
            _ => (!operand, true),
        };

        // The arithmetic opcodes set every flag inside the adder; the logical
        // ones take C from the shifter and leave V alone.
        let logical = matches!(op, 0x0 | 0x1 | 0x8 | 0x9 | 0xC | 0xD | 0xE | 0xF);
        if logical && set_flags {
            self.set_nzc32(result, shifter_carry);
        }

        if writes {
            if rd == 15 {
                return self.a32_write_pc(result);
            }
            self.set_r32(rd, result);
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `MRS`, `MSR` (register form), `BX`, `BLX`, `CLZ` and the saturating
    /// add/subtract pairs.
    fn a32_misc(&mut self, insn: u32) -> Result<()> {
        let op = (insn >> 21) & 0b11;
        let rd = ((insn >> 12) & 0xF) as u8;
        let rm = (insn & 0xF) as u8;
        match (insn >> 4) & 0x7 {
            0b000 => {
                if (insn >> 21) & 1 == 0 {
                    // MRS: the guest reads APSR. Only the condition flags and
                    // GE are visible at EL0; the mode bits read as User.
                    let apsr = (self.nzcv & 0xF000_0000)
                        | (u32::from(self.cpsr_q) << 27)
                        | (u32::from(self.cpsr_ge) << 16)
                        | 0x10;
                    self.set_r32(rd, apsr);
                } else {
                    // MSR: a write to APSR, which at EL0 can only reach the
                    // flags and GE: the mask says which.
                    let mask = (insn >> 16) & 0xF;
                    let val = self.r32(rm);
                    if mask & 0b1000 != 0 {
                        self.nzcv = (self.nzcv & 0x0FFF_FFFF) | (val & 0xF000_0000);
                        self.cpsr_q = (val >> 27) & 1 != 0;
                    }
                    if mask & 0b0100 != 0 {
                        self.cpsr_ge = ((val >> 16) & 0xF) as u8;
                    }
                }
                self.pc = self.pc.wrapping_add(4);
                Ok(())
            }
            0b001 if op == 0b01 => {
                // BX
                let target = self.r32(rm);
                self.a32_write_pc(target)
            }
            0b001 if op == 0b11 => {
                // CLZ
                let val = self.r32(rm);
                self.set_r32(rd, val.leading_zeros());
                self.pc = self.pc.wrapping_add(4);
                Ok(())
            }
            0b011 if op == 0b01 => {
                // BLX register: the return address is the next instruction.
                let target = self.r32(rm);
                self.regs[14] = u64::from(self.pc.wrapping_add(4));
                self.a32_write_pc(target)
            }
            0b101 => {
                // QADD/QSUB/QDADD/QDSUB: saturating, and they set Q rather
                // than the condition flags.
                let a = self.r32(rm) as i32;
                let b = self.r32(((insn >> 16) & 0xF) as u8) as i32;
                // The doubling in QDADD/QDSUB saturates on its own, and
                // either half setting Q is enough.
                let (doubled, doubled_sat) = {
                    let (wrapped, sat) = b.overflowing_add(b);
                    (
                        if sat {
                            Self::sat_edge(wrapped)
                        } else {
                            wrapped
                        },
                        sat,
                    )
                };
                let (value, saturated) = match op {
                    0b00 => a.overflowing_add(b),
                    0b01 => a.overflowing_sub(b),
                    0b10 => a.overflowing_add(doubled),
                    _ => a.overflowing_sub(doubled),
                };
                let value = if saturated {
                    Self::sat_edge(value)
                } else {
                    value
                };
                if saturated || (doubled_sat && op & 0b10 != 0) {
                    self.cpsr_q = true;
                }
                self.set_r32(rd, value as u32);
                self.pc = self.pc.wrapping_add(4);
                Ok(())
            }
            0b111 => Err(Error::Cpu(format!(
                "BKPT {:#06x} at pc={:#010x}",
                ((insn >> 4) & 0xFFF0) | (insn & 0xF),
                self.pc
            ))),
            _ => Err(Error::Cpu(format!(
                "unimplemented A32 miscellaneous instruction {:#010x} at pc={:#010x}",
                insn, self.pc
            ))),
        }
    }

    /// The edge an overflowing operation saturates to, from the sign of the
    /// value it wrapped to: an addition that overflowed upwards wraps
    /// negative, and saturates to `INT_MAX`.
    #[inline]
    fn sat_edge(wrapped: i32) -> i32 {
        if wrapped < 0 {
            i32::MAX
        } else {
            i32::MIN
        }
    }

    /// `MOVW` and `MOVT`: the two halves a compiler builds a 32-bit constant
    /// out of, since no rotated 8-bit immediate can hold one.
    fn a32_move_wide(&mut self, insn: u32) -> Result<()> {
        let rd = ((insn >> 12) & 0xF) as u8;
        let imm16 = ((insn >> 4) & 0xF000) | (insn & 0xFFF);
        let value = if (insn >> 22) & 1 != 0 {
            (self.r32(rd) & 0xFFFF) | (imm16 << 16)
        } else {
            imm16
        };
        self.set_r32(rd, value);
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `MSR` with an immediate, and the hint space (`NOP`, `YIELD`, `WFE`,
    /// `WFI`, `SEV`, `DBG`) that shares its encoding.
    fn a32_msr_imm_or_hint(&mut self, insn: u32) -> Result<()> {
        let mask = (insn >> 16) & 0xF;
        if mask == 0 {
            // The hints. `YIELD` is the only one worth acting on, and the
            // scheduler already switches between instructions, so all of them
            // retire.
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }
        let (val, _) = expand_imm_c(insn & 0xFFF, self.carry_flag());
        if mask & 0b1000 != 0 {
            self.nzcv = (self.nzcv & 0x0FFF_FFFF) | (val & 0xF000_0000);
            self.cpsr_q = (val >> 27) & 1 != 0;
        }
        if mask & 0b0100 != 0 {
            self.cpsr_ge = ((val >> 16) & 0xF) as u8;
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `MUL`, `MLA`, `MLS`, `UMAAL` and the four long multiplies.
    fn a32_multiply(&mut self, insn: u32) -> Result<()> {
        let set_flags = (insn >> 20) & 1 != 0;
        let rd_hi = ((insn >> 16) & 0xF) as u8;
        let rd_lo = ((insn >> 12) & 0xF) as u8;
        let rs = self.r32(((insn >> 8) & 0xF) as u8);
        let rm = self.r32((insn & 0xF) as u8);
        match (insn >> 21) & 0x7 {
            0b000 => {
                let result = rm.wrapping_mul(rs);
                self.set_r32(rd_hi, result);
                if set_flags {
                    self.set_nz32(result);
                }
            }
            0b001 => {
                let result = rm.wrapping_mul(rs).wrapping_add(self.r32(rd_lo));
                self.set_r32(rd_hi, result);
                if set_flags {
                    self.set_nz32(result);
                }
            }
            0b011 => {
                // MLS, which has no flag-setting form.
                let result = self.r32(rd_lo).wrapping_sub(rm.wrapping_mul(rs));
                self.set_r32(rd_hi, result);
            }
            0b010 => {
                // UMAAL: two independent accumulates into one 64-bit product.
                let product = u64::from(rm) * u64::from(rs)
                    + u64::from(self.r32(rd_lo))
                    + u64::from(self.r32(rd_hi));
                self.set_r32(rd_lo, product as u32);
                self.set_r32(rd_hi, (product >> 32) as u32);
            }
            other => {
                let signed = other & 0b010 != 0;
                let accumulate = other & 0b001 != 0;
                let product = if signed {
                    ((rm as i32 as i64) * (rs as i32 as i64)) as u64
                } else {
                    u64::from(rm) * u64::from(rs)
                };
                let product = if accumulate {
                    let acc = (u64::from(self.r32(rd_hi)) << 32) | u64::from(self.r32(rd_lo));
                    product.wrapping_add(acc)
                } else {
                    product
                };
                self.set_r32(rd_lo, product as u32);
                self.set_r32(rd_hi, (product >> 32) as u32);
                if set_flags {
                    let n = u32::from(product >> 63 != 0) << 31;
                    let z = u32::from(product == 0) << 30;
                    self.nzcv = (self.nzcv & 0x3000_0000) | n | z;
                }
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// The `SMLA<x><y>` family: 16-bit halves multiplied into a 32- or 64-bit
    /// accumulator.
    fn a32_halfword_multiply(&mut self, insn: u32) -> Result<()> {
        let rd = ((insn >> 16) & 0xF) as u8;
        let ra = ((insn >> 12) & 0xF) as u8;
        let rs = self.r32(((insn >> 8) & 0xF) as u8);
        let rm = self.r32((insn & 0xF) as u8);
        let n_high = (insn >> 5) & 1 != 0;
        let m_high = (insn >> 6) & 1 != 0;
        let half = |v: u32, high: bool| -> i32 {
            if high {
                (v >> 16) as i16 as i32
            } else {
                v as i16 as i32
            }
        };
        match (insn >> 21) & 0b11 {
            // SMLABB/BT/TB/TT
            0b00 => {
                let product = half(rm, n_high).wrapping_mul(half(rs, m_high));
                let acc = self.r32(ra) as i32;
                let (result, overflow) = product.overflowing_add(acc);
                if overflow {
                    self.cpsr_q = true;
                }
                self.set_r32(rd, result as u32);
            }
            // SMLAW<y> and SMULW<y>: a 32-bit operand by a halfword, keeping
            // the top 32 bits of the 48-bit product.
            0b01 => {
                let product = ((rm as i32 as i64) * i64::from(half(rs, m_high))) >> 16;
                if n_high {
                    self.set_r32(rd, product as u32);
                } else {
                    let acc = self.r32(ra) as i32;
                    let (result, overflow) = (product as i32).overflowing_add(acc);
                    if overflow {
                        self.cpsr_q = true;
                    }
                    self.set_r32(rd, result as u32);
                }
            }
            // SMLAL<x><y>
            0b10 => {
                let product = i64::from(half(rm, n_high).wrapping_mul(half(rs, m_high)));
                let acc = ((u64::from(self.r32(rd)) << 32) | u64::from(self.r32(ra))) as i64;
                let result = acc.wrapping_add(product) as u64;
                self.set_r32(ra, result as u32);
                self.set_r32(rd, (result >> 32) as u32);
            }
            // SMUL<x><y>
            _ => {
                let product = half(rm, n_high).wrapping_mul(half(rs, m_high));
                self.set_r32(rd, product as u32);
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }
}
