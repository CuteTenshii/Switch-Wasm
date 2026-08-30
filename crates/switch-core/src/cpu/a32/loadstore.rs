//! A32 memory access: the single load/store forms, the halfword and
//! doubleword "extra" forms, the block transfers, and the exclusive pairs.

use super::shift::{decode_imm_shift, shift_c};
use crate::cpu::Cpu;
use crate::{Error, Result};

impl Cpu {
    /// Apply an addressing mode and hand back the address to access and the
    /// value the base register should end up with.
    ///
    /// Pre-indexed and offset forms access `base ± offset`; post-indexed forms
    /// access `base` and write the sum back. A post-indexed form always writes
    /// back, which is what the `W` bit means only when `P` is set.
    #[inline]
    fn a32_address(&self, base: u32, offset: u32, insn: u32) -> (u32, Option<u32>) {
        let pre = (insn >> 24) & 1 != 0;
        let add = (insn >> 23) & 1 != 0;
        let writeback = (insn >> 21) & 1 != 0;
        let moved = if add {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
        if pre {
            (moved, if writeback { Some(moved) } else { None })
        } else {
            (base, Some(moved))
        }
    }

    /// `LDR`/`STR`/`LDRB`/`STRB` with a 12-bit immediate offset.
    pub(super) fn a32_load_store_imm(&mut self, insn: u32) -> Result<()> {
        self.a32_load_store(insn, insn & 0xFFF)
    }

    /// The same, with a shifted register offset.
    pub(super) fn a32_load_store_reg(&mut self, insn: u32) -> Result<()> {
        let (ty, amount) = decode_imm_shift(((insn >> 5) & 0b11) as u8, ((insn >> 7) & 0x1F) as u8);
        let (offset, _) = shift_c(self.r32((insn & 0xF) as u8), ty, amount, self.carry_flag());
        self.a32_load_store(insn, offset)
    }

    fn a32_load_store(&mut self, insn: u32, offset: u32) -> Result<()> {
        let rn = ((insn >> 16) & 0xF) as u8;
        let rt = ((insn >> 12) & 0xF) as u8;
        let byte = (insn >> 22) & 1 != 0;
        let load = (insn >> 20) & 1 != 0;
        let (addr, writeback) = self.a32_address(self.r32(rn), offset, insn);

        if load {
            let value = if byte {
                u32::from(self.mem.read_u8(addr)?)
            } else {
                self.mem.read_u32(addr)?
            };
            // The base is written back before the loaded value lands, so
            // `ldr r0, [r0], #4` keeps the load and not the increment.
            if let Some(value) = writeback {
                self.set_r32(rn, value);
            }
            if rt == 15 {
                return self.a32_write_pc(value);
            }
            self.set_r32(rt, value);
        } else {
            let value = self.r32(rt);
            if byte {
                self.mem.write_u8(addr, value as u8)?;
            } else {
                self.mem.write_u32(addr, value)?;
            }
            if let Some(value) = writeback {
                self.set_r32(rn, value);
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// The halfword, signed-byte and doubleword forms, which the architecture
    /// puts in the data-processing space rather than with the loads.
    pub(super) fn a32_extra_load_store(&mut self, insn: u32) -> Result<()> {
        let rn = ((insn >> 16) & 0xF) as u8;
        let rt = ((insn >> 12) & 0xF) as u8;
        let immediate = (insn >> 22) & 1 != 0;
        let offset = if immediate {
            ((insn >> 4) & 0xF0) | (insn & 0xF)
        } else {
            self.r32((insn & 0xF) as u8)
        };
        let (addr, writeback) = self.a32_address(self.r32(rn), offset, insn);
        let load = (insn >> 20) & 1 != 0;

        match ((insn >> 5) & 0b11, load) {
            (0b01, true) => {
                let value = u32::from(self.mem.read_u16(addr)?);
                self.set_r32(rt, value);
            }
            (0b01, false) => self.mem.write_u16(addr, self.r32(rt) as u16)?,
            (0b10, true) => {
                let value = self.mem.read_u8(addr)? as i8 as i32 as u32;
                self.set_r32(rt, value);
            }
            (0b11, true) => {
                let value = self.mem.read_u16(addr)? as i16 as i32 as u32;
                self.set_r32(rt, value);
            }
            // LDRD and STRD, which move `Rt` and `Rt + 1` together.
            (0b10, false) => {
                let lo = self.mem.read_u32(addr)?;
                let hi = self.mem.read_u32(addr.wrapping_add(4))?;
                self.set_r32(rt, lo);
                self.set_r32(rt + 1, hi);
            }
            _ => {
                self.mem.write_u32(addr, self.r32(rt))?;
                self.mem.write_u32(addr.wrapping_add(4), self.r32(rt + 1))?;
            }
        }
        if let Some(value) = writeback {
            self.set_r32(rn, value);
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `LDM`/`STM`: the block transfers a function prologue and epilogue are
    /// built from.
    ///
    /// The list is always transferred lowest register to lowest address
    /// whichever direction the addressing runs, so both directions are handled
    /// by computing the lowest address first and walking up.
    pub(super) fn a32_load_store_multiple(&mut self, insn: u32) -> Result<()> {
        if (insn >> 22) & 1 != 0 {
            return Err(Error::Cpu(format!(
                "LDM/STM with banked or PSR transfer is privileged: {:#010x} at pc={:#010x}",
                insn, self.pc
            )));
        }
        let rn = ((insn >> 16) & 0xF) as u8;
        let list = insn & 0xFFFF;
        let count = list.count_ones();
        let base = self.r32(rn);
        let pre = (insn >> 24) & 1 != 0;
        let add = (insn >> 23) & 1 != 0;
        let bytes = count * 4;

        // Where the lowest-numbered register goes, for each of the four
        // addressing modes.
        let start = match (pre, add) {
            (false, true) => base,                      // IA
            (true, true) => base.wrapping_add(4),       // IB
            (false, false) => base.wrapping_sub(bytes), // DA
            (true, false) => base.wrapping_sub(bytes),  // DB
        };
        let end = if add {
            base.wrapping_add(bytes)
        } else {
            base.wrapping_sub(bytes)
        };

        let load = (insn >> 20) & 1 != 0;
        let writeback = (insn >> 21) & 1 != 0;
        // A load that names its own base in the list keeps what it loaded, not
        // the writeback. A store transfers the base's *original* value, which
        // is why the writeback happens after the loop rather than before it.
        let base_in_list = list & (1 << rn) != 0;

        let mut addr = start;
        let mut branch_to = None;
        for reg in 0..16u8 {
            if list & (1 << reg) == 0 {
                continue;
            }
            if load {
                let value = self.mem.read_u32(addr)?;
                if reg == 15 {
                    branch_to = Some(value);
                } else {
                    self.set_r32(reg, value);
                }
            } else {
                // Storing r15 stores pc+8, which `r32` already gives.
                self.mem.write_u32(addr, self.r32(reg))?;
            }
            addr = addr.wrapping_add(4);
        }

        if writeback && !(load && base_in_list) {
            self.set_r32(rn, end);
        }

        if let Some(target) = branch_to {
            return self.a32_write_pc(target);
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `LDREX`/`STREX` and the deprecated `SWP`, which share an encoding.
    pub(super) fn a32_sync(&mut self, insn: u32) -> Result<()> {
        let rn = ((insn >> 16) & 0xF) as u8;
        let rd = ((insn >> 12) & 0xF) as u8;
        let rt = (insn & 0xF) as u8;
        let addr = self.r32(rn);
        match (insn >> 20) & 0xF {
            // SWP / SWPB: an unconditional read-modify-write, and the only
            // form here that does not use the monitor.
            0b0000 | 0b0100 => {
                let byte = (insn >> 22) & 1 != 0;
                let value = self.r32(rt);
                if byte {
                    let old = self.mem.read_u8(addr)?;
                    self.mem.write_u8(addr, value as u8)?;
                    self.set_r32(rd, u32::from(old));
                } else {
                    let old = self.mem.read_u32(addr)?;
                    self.mem.write_u32(addr, value)?;
                    self.set_r32(rd, old);
                }
            }
            // STREX{,D,B,H}: `rd` takes 0 on success, 1 when the monitor was
            // lost. Same monitor the A64 pairs use, so a mode switch inside a
            // sequence cannot make one succeed across the other's writes.
            0b1000 | 0b1010 | 0b1100 | 0b1110 => {
                if self.exclusive.take() == Some(addr) {
                    let value = self.r32(rt);
                    match (insn >> 21) & 0b11 {
                        0b00 => self.mem.write_u32(addr, value)?,
                        0b01 => {
                            self.mem.write_u32(addr, value)?;
                            self.mem.write_u32(addr.wrapping_add(4), self.r32(rt + 1))?;
                        }
                        0b10 => self.mem.write_u8(addr, value as u8)?,
                        _ => self.mem.write_u16(addr, value as u16)?,
                    }
                    self.set_r32(rd, 0);
                } else {
                    self.set_r32(rd, 1);
                }
            }
            // LDREX{,D,B,H}. `rd` is the destination here, not a status word.
            0b1001 | 0b1011 | 0b1101 | 0b1111 => {
                match (insn >> 21) & 0b11 {
                    0b00 => {
                        let value = self.mem.read_u32(addr)?;
                        self.set_r32(rd, value);
                    }
                    0b01 => {
                        let lo = self.mem.read_u32(addr)?;
                        let hi = self.mem.read_u32(addr.wrapping_add(4))?;
                        self.set_r32(rd, lo);
                        self.set_r32(rd + 1, hi);
                    }
                    0b10 => {
                        let value = u32::from(self.mem.read_u8(addr)?);
                        self.set_r32(rd, value);
                    }
                    _ => {
                        let value = u32::from(self.mem.read_u16(addr)?);
                        self.set_r32(rd, value);
                    }
                }
                self.exclusive = Some(addr);
            }
            _ => {
                return Err(Error::Cpu(format!(
                    "unimplemented A32 synchronisation instruction {:#010x} at pc={:#010x}",
                    insn, self.pc
                )))
            }
        }
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }
}
