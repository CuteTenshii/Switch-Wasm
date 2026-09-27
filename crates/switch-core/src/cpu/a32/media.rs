//! The ARMv6 media instructions: the extends, the reverses, the bitfield
//! moves and the saturations a compiler emits from ordinary C.
//!
//! They share their `op1` with each other in ways that matter: `SSAT` is
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
            // The parallel adds and subtracts: SADD16 through UHSUB8.
            (0x01..=0x03 | 0x05..=0x07, 0b000..=0b100 | 0b111) => {
                let result = self.a32_parallel(op1, op2, self.r32(rn), self.r32(rm));
                self.set_r32(rd, result);
            }
            // SEL: each byte from Rn where its GE flag is set, else from Rm.
            (0x08, 0b101) => {
                let (n, m) = (self.r32(rn), self.r32(rm));
                let result = (0..4).fold(0u32, |acc, byte| {
                    let from = if self.cpsr_ge >> byte & 1 != 0 { n } else { m };
                    acc | from & (0xFF << (byte * 8))
                });
                self.set_r32(rd, result);
            }
            // PKHBT / PKHTB: one halfword of Rn and one of the shifted Rm.
            // The top-from-Rn form shifts arithmetically right, and a shift
            // of 0 there means 32.
            (0x08, 0b000 | 0b010 | 0b100 | 0b110) => {
                let amount = (insn >> 7) & 0x1F;
                let (n, m) = (self.r32(rn), self.r32(rm));
                let result = if (insn >> 6) & 1 == 0 {
                    n & 0xFFFF | (m << amount) & 0xFFFF_0000
                } else {
                    let shifted = ((m as i32) >> if amount == 0 { 31 } else { amount }) as u32;
                    n & 0xFFFF_0000 | shifted & 0xFFFF
                };
                self.set_r32(rd, result);
            }
            // SSAT16 / USAT16: each signed halfword saturated on its own.
            (0x0A | 0x0E, 0b001) => {
                let bits = (insn >> 16) & 0xF;
                let (lo, hi) = if op1 == 0x0A {
                    (-(1i32 << bits), (1i32 << bits) - 1)
                } else {
                    (0, (1i32 << bits) - 1)
                };
                let value = self.r32(rm);
                let mut result = 0u32;
                for lane in 0..2 {
                    let half = i32::from((value >> (16 * lane)) as u16 as i16);
                    let clamped = half.clamp(lo, hi);
                    if clamped != half {
                        self.cpsr_q = true;
                    }
                    result |= (clamped as u32 & 0xFFFF) << (16 * lane);
                }
                self.set_r32(rd, result);
            }
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
            // (S|U)XTB16(A): bytes 0 and 2 of the rotated source, each
            // extended to a halfword; the addend form adds lane by lane, and
            // a carry out of the low lane does not reach the high one.
            (0x08 | 0x0C, 0b011) => {
                let rotate = ((insn >> 10) & 0b11) * 8;
                let value = self.r32(rm).rotate_right(rotate);
                let lane = |byte: u32| -> u16 {
                    if op1 == 0x08 {
                        byte as u8 as i8 as i16 as u16
                    } else {
                        byte as u8 as u16
                    }
                };
                let (mut low, mut high) = (lane(value), lane(value >> 16));
                if rn != 15 {
                    let addend = self.r32(rn);
                    low = low.wrapping_add(addend as u16);
                    high = high.wrapping_add((addend >> 16) as u16);
                }
                self.set_r32(rd, u32::from(low) | u32::from(high) << 16);
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
            // SMLALD / SMLSLD: the dual products, summed or differenced, into
            // a 64-bit accumulator in RdHi:RdLo. `X` swaps Rm's halves.
            (0x14, 0b000..=0b011) => {
                let (lo_reg, hi_reg) = (rd, rn);
                let n = self.r32(rm);
                let m = self.r32(((insn >> 8) & 0xF) as u8);
                let m = if (insn >> 5) & 1 != 0 {
                    m.rotate_right(16)
                } else {
                    m
                };
                let low = i64::from(n as i16) * i64::from(m as i16);
                let high = i64::from((n >> 16) as i16) * i64::from((m >> 16) as i16);
                let dual = if op2 & 0b010 != 0 {
                    low - high
                } else {
                    low + high
                };
                let acc = (u64::from(self.r32(hi_reg)) << 32 | u64::from(self.r32(lo_reg))) as i64;
                let sum = acc.wrapping_add(dual) as u64;
                self.set_r32(lo_reg, sum as u32);
                self.set_r32(hi_reg, (sum >> 32) as u32);
            }
            // USAD8 / USADA8: the sum of the four bytes' absolute differences,
            // plus Ra unless it is 15. The destination is bits 19:16.
            (0x18, 0b000) => {
                let n = self.r32(rm);
                let m = self.r32(((insn >> 8) & 0xF) as u8);
                let sum = (0..4).fold(0u32, |acc, byte| {
                    let a = (n >> (8 * byte)) & 0xFF;
                    let b = (m >> (8 * byte)) & 0xFF;
                    acc + a.abs_diff(b)
                });
                let result = if rd == 15 {
                    sum
                } else {
                    self.r32(rd).wrapping_add(sum)
                };
                self.set_r32(rn, result);
            }
            // SDIV and UDIV, which the media space files under the signed
            // multiplies. The destination is bits 19:16 here, not 15:12,
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

    /// One parallel add or subtract. `op1`'s low two bits are the kind: 1
    /// modular, which sets GE; 2 saturating; 3 halving. Its bit 2 says
    /// unsigned. `op2` is the operation: ADD16, ASX, SAX, SUB16, ADD8 or
    /// SUB8. Each lane is computed at full precision first, and the flags,
    /// the saturation and the halving are all taken from that.
    fn a32_parallel(&mut self, op1: u32, op2: u32, n: u32, m: u32) -> u32 {
        let signed = op1 < 0x04;
        let kind = op1 & 0b11;
        // (subtract, Rn lane, Rm lane) for each result lane, low first.
        let (width, lanes): (u32, &[(bool, u32, u32)]) = match op2 {
            0b000 => (16, &[(false, 0, 0), (false, 1, 1)]),
            0b001 => (16, &[(true, 0, 1), (false, 1, 0)]),
            0b010 => (16, &[(false, 0, 1), (true, 1, 0)]),
            0b011 => (16, &[(true, 0, 0), (true, 1, 1)]),
            0b100 => (
                8,
                &[(false, 0, 0), (false, 1, 1), (false, 2, 2), (false, 3, 3)],
            ),
            _ => (8, &[(true, 0, 0), (true, 1, 1), (true, 2, 2), (true, 3, 3)]),
        };
        let mask = (1u32 << width) - 1;
        let lane_value = |word: u32, lane: u32| -> i32 {
            let raw = (word >> (lane * width)) & mask;
            if signed {
                ((raw << (32 - width)) as i32) >> (32 - width)
            } else {
                raw as i32
            }
        };
        let flags_per_lane = 4 / lanes.len() as u32;
        let mut result = 0u32;
        let mut ge = 0u8;
        for (index, &(subtract, from_n, from_m)) in lanes.iter().enumerate() {
            let (a, b) = (lane_value(n, from_n), lane_value(m, from_m));
            let full = if subtract { a - b } else { a + b };
            let value = match kind {
                // GE says which lanes did not go negative, for a signed lane
                // or an unsigned subtract, and which carried out, for an
                // unsigned add.
                0b01 => {
                    let set = if signed || subtract {
                        full >= 0
                    } else {
                        full > mask as i32
                    };
                    if set {
                        ge |= ((1u8 << flags_per_lane) - 1) << (index as u32 * flags_per_lane);
                    }
                    full
                }
                0b10 if signed => full.clamp(-(1 << (width - 1)), (1 << (width - 1)) - 1),
                0b10 => full.clamp(0, mask as i32),
                _ => full >> 1,
            };
            result |= (value as u32 & mask) << (index as u32 * width);
        }
        if kind == 0b01 {
            self.cpsr_ge = ge;
        }
        result
    }
}
