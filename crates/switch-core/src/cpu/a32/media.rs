//! The ARMv6 media instructions: the extends, the reverses, the bitfield
//! moves and the saturations a compiler emits from ordinary C.
//!
//! They share their `op1` with each other in ways that matter — `SSAT` is
//! `0110 101x` and `SXTB16`/`SXTH` are `0110 1000`/`0110 1011`, so bit 5 of
//! `op2` is what tells a saturation from an extend, not the opcode field.

use super::shift::{decode_imm_shift, shift_c};
use crate::cpu::Cpu;
use crate::{Error, Result};

impl Cpu {
    /// The ARMv6 media space: the extends, the reverses, the bitfield
    /// instructions and the saturations a compiler emits from ordinary C.
    pub(super) fn a32_media(&mut self, insn: u32) -> Result<()> {
        let rd = ((insn >> 12) & 0xF) as u8;
        let rn = ((insn >> 16) & 0xF) as u8;
        let rm = (insn & 0xF) as u8;
        let op1 = (insn >> 20) & 0x1F;
        let op2 = (insn >> 5) & 0x7;
        match (op1, op2) {
            // SBFX / UBFX
            (0x1A | 0x1B | 0x1E | 0x1F, 0b010 | 0b110) => {
                let lsb = (insn >> 7) & 0x1F;
                let width = ((insn >> 16) & 0x1F) + 1;
                let value = self.r32(rm);
                let field = if lsb + width > 32 {
                    return Err(Error::Cpu(format!(
                        "bitfield extract past the end of a word: {:#010x} at pc={:#010x}",
                        insn, self.pc
                    )));
                } else {
                    (value >> lsb) & (u32::MAX >> (32 - width))
                };
                let signed = op1 & 0b00100 == 0;
                let result = if signed && width < 32 && (field >> (width - 1)) & 1 != 0 {
                    field | !(u32::MAX >> (32 - width))
                } else {
                    field
                };
                self.set_r32(rd, result);
            }
            // BFC / BFI
            (0x1C | 0x1D, 0b000 | 0b100) => {
                let lsb = (insn >> 7) & 0x1F;
                let msb = (insn >> 16) & 0x1F;
                if msb < lsb {
                    return Err(Error::Cpu(format!(
                        "bitfield insert with msb below lsb: {:#010x} at pc={:#010x}",
                        insn, self.pc
                    )));
                }
                let mask = (u32::MAX >> (31 - (msb - lsb))) << lsb;
                // Rm == 15 is BFC, which clears the field instead of taking one.
                let source = if rm == 15 { 0 } else { self.r32(rm) << lsb };
                let result = (self.r32(rd) & !mask) | (source & mask);
                self.set_r32(rd, result);
            }
            // The extends, with and without an addend: (S|U)XT(B|H|B16)(A).
            (0x0A | 0x0B | 0x0E | 0x0F, 0b011 | 0b111) => {
                let rotate = ((insn >> 10) & 0b11) * 8;
                let value = self.r32(rm).rotate_right(rotate);
                let signed = op1 & 0b00100 == 0;
                let halfword = op1 & 0b00011 == 0b00011;
                let extended = match (signed, halfword) {
                    (true, false) => value as u8 as i8 as i32 as u32,
                    (true, true) => value as u16 as i16 as i32 as u32,
                    (false, false) => value & 0xFF,
                    (false, true) => value & 0xFFFF,
                };
                // Rn == 15 selects the plain extend; anything else adds.
                let result = if rn == 15 {
                    extended
                } else {
                    self.r32(rn).wrapping_add(extended)
                };
                self.set_r32(rd, result);
            }
            // REV / REV16 / RBIT / REVSH
            (0x0B | 0x0F, 0b001 | 0b101) => {
                let value = self.r32(rm);
                let result = match (op1, op2) {
                    (0x0B, 0b001) => value.swap_bytes(),
                    (0x0B, 0b101) => ((value & 0x00FF_00FF) << 8) | ((value >> 8) & 0x00FF_00FF),
                    (0x0F, 0b001) => value.reverse_bits(),
                    _ => i32::from((value as u16).swap_bytes() as i16) as u32,
                };
                self.set_r32(rd, result);
            }
            // SSAT / USAT
            (0x0A | 0x0B | 0x0E | 0x0F, _) if op2 & 0b001 == 0 => {
                let signed = op1 & 0b00100 == 0;
                let bits = if signed {
                    ((insn >> 16) & 0x1F) + 1
                } else {
                    (insn >> 16) & 0x1F
                };
                let (ty, amount) =
                    decode_imm_shift(((insn >> 5) & 0b10) as u8, ((insn >> 7) & 0x1F) as u8);
                let (value, _) = shift_c(self.r32(rm), ty, amount, self.carry_flag());
                let value = value as i32;
                let (lo, hi) = if signed {
                    (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
                } else {
                    (0, (1i64 << bits) - 1)
                };
                let clamped = i64::from(value).clamp(lo, hi);
                if clamped != i64::from(value) {
                    self.cpsr_q = true;
                }
                self.set_r32(rd, clamped as u32);
            }
            // The dual signed multiplies: two halfword products added or
            // subtracted, optionally accumulated. `X` swaps the second
            // operand's halves.
            (0x10, 0b000..=0b011) => {
                let d = rn;
                let a = self.r32(rm) as i32;
                let b = self.r32(((insn >> 8) & 0xF) as u8);
                let b = if (insn >> 5) & 1 != 0 {
                    b.rotate_right(16) as i32
                } else {
                    b as i32
                };
                let lo = i32::from(a as i16).wrapping_mul(i32::from(b as i16));
                let hi = i32::from((a >> 16) as i16).wrapping_mul(i32::from((b >> 16) as i16));
                let dual = if op2 & 0b010 != 0 {
                    lo.wrapping_sub(hi)
                } else {
                    lo.wrapping_add(hi)
                };
                // Ra == 15 is the non-accumulating form.
                let result = if rd == 15 {
                    dual
                } else {
                    let (sum, overflow) = dual.overflowing_add(self.r32(rd) as i32);
                    if overflow {
                        self.cpsr_q = true;
                    }
                    sum
                };
                self.set_r32(d, result as u32);
            }
            // SMMUL, SMMLA and SMMLS: the top 32 bits of a 64-bit product,
            // optionally rounded and accumulated.
            (0x15, 0b000 | 0b001 | 0b110 | 0b111) => {
                let d = rn;
                let a = i64::from(self.r32(rm) as i32);
                let b = i64::from(self.r32(((insn >> 8) & 0xF) as u8) as i32);
                let acc = if rd == 15 {
                    0
                } else {
                    i64::from(self.r32(rd) as i32) << 32
                };
                let product = a.wrapping_mul(b);
                let sum = if op2 & 0b100 != 0 {
                    acc.wrapping_sub(product)
                } else {
                    acc.wrapping_add(product)
                };
                // The R bit rounds by adding half before the truncation.
                let rounded = if (insn >> 5) & 1 != 0 {
                    sum.wrapping_add(0x8000_0000)
                } else {
                    sum
                };
                self.set_r32(d, (rounded >> 32) as u32);
            }
            // SDIV and UDIV, which the media space files under the signed
            // multiplies. The destination is bits 19:16 here, not 15:12 —
            // that field holds the 0b1111 that says there is no accumulator.
            (0x11 | 0x13, 0b000) => {
                let n = self.r32(rm);
                let m = self.r32(((insn >> 8) & 0xF) as u8);
                // Horizon leaves integer division-by-zero untrapped, so it
                // gives zero rather than faulting.
                let result = if m == 0 {
                    0
                } else if op1 == 0x11 {
                    (n as i32).wrapping_div(m as i32) as u32
                } else {
                    n / m
                };
                self.set_r32(rn, result);
            }
            _ => {
                return Err(Error::Cpu(format!(
                    "unimplemented A32 media instruction {:#010x} at pc={:#010x}",
                    insn, self.pc
                )))
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }
}
