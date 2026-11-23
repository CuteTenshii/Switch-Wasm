//! The T32 execution state: the 16-bit and 32-bit Thumb encodings, IT blocks and
//! interworking. Instructions with an A32 twin of the same shape are rewritten as
//! it and run by the A32 decoder; the rest call the helpers the A32 decoder uses.

use super::loadstore::Access;
use super::shift::{decode_imm_shift, shift_by_register, shift_c};
use crate::cpu::Cpu;
use crate::{Error, Result};

/// The A32 condition field of an instruction that always runs.
const AL: u32 = 0xE000_0000;

/// `ThumbExpandImm_C`: the 12-bit modified immediate and the carry it gives.
fn expand_imm_c(imm12: u32, carry_in: bool) -> (u32, bool) {
    let imm8 = imm12 & 0xFF;
    if imm12 >> 10 == 0 {
        let value = match (imm12 >> 8) & 3 {
            0 => imm8,
            1 => imm8 << 16 | imm8,
            2 => imm8 << 24 | imm8 << 8,
            _ => imm8 * 0x0101_0101,
        };
        (value, carry_in)
    } else {
        let value = (0x80 | (imm12 & 0x7F)).rotate_right(imm12 >> 7);
        (value, value >> 31 != 0)
    }
}

/// Sign-extend the low `bits` of `value`.
fn sext(value: u32, bits: u32) -> u32 {
    ((value << (32 - bits)) as i32 >> (32 - bits)) as u32
}

impl Cpu {
    /// Fetch and run the instruction at the PC, which is in T32 state; the caller
    /// has counted it against the slice.
    pub(crate) fn step_thumb(&mut self) -> Result<()> {
        let pc = self.pc;
        let first = match self.mem.read_u16(u64::from(pc)) {
            Ok(h) => u32::from(h),
            Err(e) => {
                self.record_fault(&e, pc, 0);
                return Err(e);
            }
        };
        let wide = first >> 11 >= 0b11101;
        let (insn, size) = if wide {
            match self.mem.read_u16(u64::from(pc.wrapping_add(2))) {
                Ok(second) => (first << 16 | u32::from(second), 4),
                Err(e) => {
                    self.record_fault(&e, pc, first);
                    return Err(e);
                }
            }
        } else {
            (first, 2)
        };
        self.record_run(pc, 1);
        let result = self.execute_thumb(insn, size);
        if self.trace_enabled {
            self.trace_line(&format!("{pc:08x}: {insn:08x}  .thumb\n"));
        }
        if let Err(e) = &result {
            self.record_fault(e, pc, insn);
        }
        self.retire();
        result
    }

    /// Run `insn`, `size` bytes long: a halfword, or both halfwords with the first on top.
    fn execute_thumb(&mut self, insn: u32, size: u32) -> Result<()> {
        let it = self.it_state;
        let in_block = it & 0xF != 0;
        let result = if in_block && !self.condition_holds(it >> 4) {
            self.pc = self.pc.wrapping_add(size);
            Ok(())
        } else if size == 2 {
            self.thumb16(insn)
        } else {
            self.thumb32(insn >> 16, insn & 0xFFFF)
        };
        // ITAdvance, for an instruction that ran inside the block; `IT` itself sets the state.
        if in_block {
            let it = self.it_state;
            self.it_state = if it & 7 == 0 {
                0
            } else {
                (it & 0xE0) | ((it << 1) & 0x1F)
            };
        }
        result
    }

    fn undefined(&self, insn: u32) -> Error {
        Error::Cpu(format!(
            "unimplemented T32 instruction {insn:#x} at pc={:#010x}",
            self.pc
        ))
    }

    /// Whether the 16-bit forms that set flags outside an `IT` block set them now.
    fn flags_now(&self) -> bool {
        self.it_state & 0xF == 0
    }

    /// A branch that keeps the state: bit 0 is ignored.
    fn thumb_branch(&mut self, target: u32) {
        self.pc = target & !1;
    }

    /// Write `r`, taking the PC as a branch.
    fn thumb_set(&mut self, r: u8, value: u32) {
        if r == 15 {
            self.thumb_branch(value);
        } else {
            self.set_r32(r, value);
        }
    }

    /// Run the A32 instruction `arm`, which has no PC operand, as this one of `size` bytes.
    fn thumb_as_arm(&mut self, arm: u32, size: u32) -> Result<()> {
        let fall = self.pc.wrapping_add(size);
        self.execute_a32(AL | arm)?;
        self.pc = fall;
        Ok(())
    }

    fn thumb16(&mut self, hw: u32) -> Result<()> {
        let fall = self.pc.wrapping_add(2);
        let pc4 = self.pc.wrapping_add(4);
        let rd3 = (hw & 7) as u8;
        let rn3 = ((hw >> 3) & 7) as u8;
        match hw >> 11 {
            // Shift by immediate.
            0..=2 => {
                let (ty, amount) = decode_imm_shift((hw >> 11) as u8, ((hw >> 6) & 0x1F) as u8);
                let carry = self.carry_flag();
                let (result, carry) = shift_c(self.r32(rn3), ty, amount, carry);
                self.set_r32(rd3, result);
                if self.flags_now() {
                    self.set_nzc32(result, carry);
                }
            }
            // ADD and SUB, register or 3-bit immediate.
            3 => {
                let operand = if hw & 0x400 != 0 {
                    (hw >> 6) & 7
                } else {
                    self.r32(((hw >> 6) & 7) as u8)
                };
                let a = self.r32(rn3);
                let flags = self.flags_now();
                let result = if hw & 0x200 != 0 {
                    self.add32_flags(a, !operand, true, flags)
                } else {
                    self.add32_flags(a, operand, false, flags)
                };
                self.set_r32(rd3, result);
            }
            // MOV, CMP, ADD, SUB with an 8-bit immediate.
            4..=7 => {
                let rd = ((hw >> 8) & 7) as u8;
                let imm = hw & 0xFF;
                let a = self.r32(rd);
                let flags = self.flags_now();
                match (hw >> 11) & 3 {
                    0 => {
                        self.set_r32(rd, imm);
                        if flags {
                            let carry = self.carry_flag();
                            self.set_nzc32(imm, carry);
                        }
                    }
                    1 => {
                        self.add32_flags(a, !imm, true, true);
                    }
                    2 => {
                        let result = self.add32_flags(a, imm, false, flags);
                        self.set_r32(rd, result);
                    }
                    _ => {
                        let result = self.add32_flags(a, !imm, true, flags);
                        self.set_r32(rd, result);
                    }
                }
            }
            8 if hw & 0x400 == 0 => self.thumb_alu(hw),
            // Special data instructions and BX.
            8 => {
                let rm = ((hw >> 3) & 0xF) as u8;
                let rdn = (hw & 7 | (hw >> 4) & 8) as u8;
                match (hw >> 8) & 3 {
                    0 => {
                        let result = self.r32(rdn).wrapping_add(self.r32(rm));
                        self.thumb_set(rdn, result);
                        if rdn == 15 {
                            return Ok(());
                        }
                    }
                    1 => {
                        let (a, b) = (self.r32(rdn), self.r32(rm));
                        self.add32_flags(a, !b, true, true);
                    }
                    2 => {
                        let value = self.r32(rm);
                        self.thumb_set(rdn, value);
                        if rdn == 15 {
                            return Ok(());
                        }
                    }
                    _ => {
                        let target = self.r32(rm);
                        if hw & 0x80 != 0 {
                            self.regs[14] = u64::from(fall | 1);
                        }
                        return self.a32_write_pc(target);
                    }
                }
            }
            // LDR (literal).
            9 => {
                let addr = (pc4 & !3).wrapping_add((hw & 0xFF) * 4);
                let value = self.mem.read_u32(u64::from(addr))?;
                self.set_r32(((hw >> 8) & 7) as u8, value);
            }
            // Load and store with a register offset.
            10 | 11 => {
                let addr = self.r32(rn3).wrapping_add(self.r32(((hw >> 6) & 7) as u8));
                match (hw >> 9) & 7 {
                    0 => self.mem.write_u32(u64::from(addr), self.r32(rd3))?,
                    1 => self.mem.write_u16(u64::from(addr), self.r32(rd3) as u16)?,
                    2 => self.mem.write_u8(u64::from(addr), self.r32(rd3) as u8)?,
                    3 => {
                        let v = self.mem.read_u8(u64::from(addr))? as i8 as i32 as u32;
                        self.set_r32(rd3, v);
                    }
                    4 => {
                        let v = self.mem.read_u32(u64::from(addr))?;
                        self.set_r32(rd3, v);
                    }
                    5 => {
                        let v = u32::from(self.mem.read_u16(u64::from(addr))?);
                        self.set_r32(rd3, v);
                    }
                    6 => {
                        let v = u32::from(self.mem.read_u8(u64::from(addr))?);
                        self.set_r32(rd3, v);
                    }
                    _ => {
                        let v = self.mem.read_u16(u64::from(addr))? as i16 as i32 as u32;
                        self.set_r32(rd3, v);
                    }
                }
            }
            // Load and store with a 5-bit immediate, word, byte and halfword.
            12..=17 => {
                let imm5 = (hw >> 6) & 0x1F;
                let base = self.r32(rn3);
                let load = hw & 0x800 != 0;
                match hw >> 12 {
                    6 => {
                        let addr = base.wrapping_add(imm5 * 4);
                        if load {
                            let v = self.mem.read_u32(u64::from(addr))?;
                            self.set_r32(rd3, v);
                        } else {
                            self.mem.write_u32(u64::from(addr), self.r32(rd3))?;
                        }
                    }
                    7 => {
                        let addr = base.wrapping_add(imm5);
                        if load {
                            let v = u32::from(self.mem.read_u8(u64::from(addr))?);
                            self.set_r32(rd3, v);
                        } else {
                            self.mem.write_u8(u64::from(addr), self.r32(rd3) as u8)?;
                        }
                    }
                    _ => {
                        let addr = base.wrapping_add(imm5 * 2);
                        if load {
                            let v = u32::from(self.mem.read_u16(u64::from(addr))?);
                            self.set_r32(rd3, v);
                        } else {
                            self.mem.write_u16(u64::from(addr), self.r32(rd3) as u16)?;
                        }
                    }
                }
            }
            // Load and store relative to SP.
            18 | 19 => {
                let rt = ((hw >> 8) & 7) as u8;
                let addr = self.r32(13).wrapping_add((hw & 0xFF) * 4);
                if hw & 0x800 != 0 {
                    let v = self.mem.read_u32(u64::from(addr))?;
                    self.set_r32(rt, v);
                } else {
                    self.mem.write_u32(u64::from(addr), self.r32(rt))?;
                }
            }
            // ADR and ADD to SP.
            20 | 21 => {
                let base = if hw & 0x800 != 0 {
                    self.r32(13)
                } else {
                    pc4 & !3
                };
                self.set_r32(((hw >> 8) & 7) as u8, base.wrapping_add((hw & 0xFF) * 4));
            }
            22 | 23 => return self.thumb_misc(hw),
            // STM and LDM, always writing the base back (an LDM naming it keeps the load).
            24 | 25 => {
                let access = Access {
                    rn: ((hw >> 8) & 7) as u8,
                    rt: 0,
                    load: hw & 0x800 != 0,
                    pre: false,
                    add: true,
                    wb: true,
                };
                self.a32_block_transfer(access, (hw & 0xFF) as u16)?;
            }
            // Conditional branch and SVC.
            26 | 27 => {
                let cond = (hw >> 8) & 0xF;
                match cond {
                    0xE => return Err(self.undefined(hw)),
                    0xF => {
                        self.pc = fall;
                        return self.syscall((hw & 0xFF) as u16);
                    }
                    _ if self.condition_holds(cond as u8) => {
                        self.pc = pc4.wrapping_add(sext(hw & 0xFF, 8) << 1);
                        return Ok(());
                    }
                    _ => {}
                }
            }
            28 => {
                self.pc = pc4.wrapping_add(sext(hw & 0x7FF, 11) << 1);
                return Ok(());
            }
            _ => return Err(self.undefined(hw)),
        }
        self.pc = fall;
        Ok(())
    }

    /// The register data-processing group: `010000 op Rm Rdn`.
    fn thumb_alu(&mut self, hw: u32) {
        let rm = ((hw >> 3) & 7) as u8;
        let rdn = (hw & 7) as u8;
        let (a, b) = (self.r32(rdn), self.r32(rm));
        let carry = self.carry_flag();
        let flags = self.flags_now();
        let op = (hw >> 6) & 0xF;
        let shift = |cpu: &mut Cpu, ty: u8| {
            let (result, carry) = shift_by_register(a, ty, b, carry);
            cpu.set_r32(rdn, result);
            if flags {
                cpu.set_nzc32(result, carry);
            }
        };
        match op {
            2 => shift(self, 0),
            3 => shift(self, 1),
            4 => shift(self, 2),
            7 => shift(self, 3),
            // RSB Rd, Rm, #0.
            9 => {
                let result = self.add32_flags(!b, 0, true, flags);
                self.set_r32(rdn, result);
            }
            13 => {
                let result = a.wrapping_mul(b);
                self.set_r32(rdn, result);
                if flags {
                    self.set_nz32(result);
                }
            }
            _ => {
                // The A32 opcode for each, compares always setting the flags.
                const OPCODE: [u32; 16] = [
                    0x0, 0x1, 0, 0, 0, 0x5, 0x6, 0, 0x8, 0, 0xA, 0xB, 0xC, 0, 0xE, 0xF,
                ];
                let opcode = OPCODE[op as usize];
                let set_flags = flags || (0x8..=0xB).contains(&opcode);
                if let Some(result) = self.a32_alu(opcode, set_flags, a, b, carry, carry) {
                    self.set_r32(rdn, result);
                }
            }
        }
    }

    /// `1011` miscellaneous 16-bit instructions.
    fn thumb_misc(&mut self, hw: u32) -> Result<()> {
        let fall = self.pc.wrapping_add(2);
        let rd = (hw & 7) as u8;
        let rm = ((hw >> 3) & 7) as u8;
        match (hw >> 8) & 0xF {
            0 => {
                let sp = self.r32(13);
                let delta = (hw & 0x7F) * 4;
                let sp = if hw & 0x80 != 0 {
                    sp.wrapping_sub(delta)
                } else {
                    sp.wrapping_add(delta)
                };
                self.set_r32(13, sp);
            }
            // CBZ and CBNZ.
            1 | 3 | 9 | 11 => {
                let offset = ((hw >> 9) & 1) << 6 | ((hw >> 3) & 0x1F) << 1;
                let nonzero = hw & 0x800 != 0;
                if (self.r32(rd) == 0) != nonzero {
                    self.pc = self.pc.wrapping_add(4).wrapping_add(offset);
                    return Ok(());
                }
            }
            2 => {
                let value = self.r32(rm);
                let result = match (hw >> 6) & 3 {
                    0 => value as u16 as i16 as i32 as u32,
                    1 => value as u8 as i8 as i32 as u32,
                    2 => value & 0xFFFF,
                    _ => value & 0xFF,
                };
                self.set_r32(rd, result);
            }
            // PUSH and POP, the extra bit being LR and PC.
            4 | 5 => {
                let list = (hw & 0xFF) | (hw & 0x100) << 6;
                let access = Access {
                    rn: 13,
                    rt: 0,
                    load: false,
                    pre: true,
                    add: false,
                    wb: true,
                };
                self.a32_block_transfer(access, list as u16)?;
            }
            12 | 13 => {
                let list = (hw & 0xFF) | (hw & 0x100) << 7;
                let access = Access {
                    rn: 13,
                    rt: 0,
                    load: true,
                    pre: false,
                    add: true,
                    wb: true,
                };
                self.a32_block_transfer(access, list as u16)?;
                if hw & 0x100 != 0 {
                    return Ok(());
                }
            }
            10 => {
                let value = self.r32(rm);
                let result = match (hw >> 6) & 3 {
                    0 => value.swap_bytes(),
                    1 => ((value & 0x00FF_00FF) << 8) | ((value >> 8) & 0x00FF_00FF),
                    3 => i32::from((value as u16).swap_bytes() as i16) as u32,
                    _ => return Err(self.undefined(hw)),
                };
                self.set_r32(rd, result);
            }
            14 => return Err(Error::Cpu(format!("BKPT at pc={:#010x}", self.pc))),
            // Hints and IT.
            15 => {
                if hw & 0xF == 0 {
                    // NOP, YIELD, WFE, WFI and SEV all retire as no-ops.
                } else {
                    self.it_state = (hw & 0xFF) as u8;
                }
            }
            // CPS is privileged and does nothing at EL0.
            6 => {}
            _ => return Err(self.undefined(hw)),
        }
        self.pc = fall;
        Ok(())
    }
}

impl Cpu {
    /// A 32-bit instruction; each form leaves the PC where it goes next.
    fn thumb32(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        match (hw1 >> 11) & 3 {
            1 if hw1 & 0x400 != 0 => self.t32_coprocessor(hw1, hw2),
            1 if hw1 & 0x200 != 0 => self.t32_shifted(hw1, hw2),
            1 if hw1 & 0x40 == 0 => self.t32_block(hw1, hw2),
            1 => self.t32_dual(hw1, hw2),
            2 if hw2 & 0x8000 != 0 => self.t32_branch(hw1, hw2),
            2 if hw1 & 0x200 == 0 => self.t32_modified_immediate(hw1, hw2),
            2 => self.t32_plain_immediate(hw1, hw2),
            _ => match (hw1 >> 4) & 0x7F {
                0x00..=0x1F => self.t32_load_store(hw1, hw2),
                0x20..=0x2F => self.t32_register(hw1, hw2),
                0x30..=0x37 => self.t32_multiply(hw1, hw2),
                0x38..=0x3F => self.t32_long_multiply(hw1, hw2),
                _ => self.t32_coprocessor(hw1, hw2),
            },
        }
    }

    /// Advance to the next instruction after a form that cannot branch.
    fn t32_next(&mut self) -> Result<()> {
        self.pc = self.pc.wrapping_add(4);
        Ok(())
    }

    /// `LDM` and `STM`, increment-after or decrement-before.
    fn t32_block(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let (pre, add) = match (hw1 >> 7) & 3 {
            1 => (false, true),
            2 => (true, false),
            _ => return Err(self.undefined(hw1 << 16 | hw2)),
        };
        let load = hw1 & 0x10 != 0;
        let access = Access {
            rn: (hw1 & 0xF) as u8,
            rt: 0,
            load,
            pre,
            add,
            wb: hw1 & 0x20 != 0,
        };
        let fall = self.pc.wrapping_add(4);
        self.a32_block_transfer(access, hw2 as u16)?;
        if !(load && hw2 & 0x8000 != 0) {
            self.pc = fall;
        }
        Ok(())
    }

    /// `LDRD`, `STRD`, the exclusives and table branches.
    fn t32_dual(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let rn = (hw1 & 0xF) as u8;
        let (p, u, w, l) = (
            hw1 & 0x100 != 0,
            hw1 & 0x80 != 0,
            hw1 & 0x20 != 0,
            hw1 & 0x10 != 0,
        );
        let rt = (hw2 >> 12) as u8;
        let rt2 = ((hw2 >> 8) & 0xF) as u8;
        let base = self.base32(rn);
        if !p && !w {
            if !u {
                // STREX and LDREX with a word offset.
                let addr = base.wrapping_add((hw2 & 0xFF) * 4);
                if l {
                    let value = self.mem.read_u32(u64::from(addr))?;
                    self.set_r32(rt, value);
                    self.exclusive.arm(u64::from(addr), [u64::from(value), 0]);
                } else {
                    let value = u64::from(self.r32(rt));
                    let stored = self.store_exclusive_at(u64::from(addr), 4, [value, 0])?;
                    self.set_r32(rt2, u32::from(!stored));
                }
                return self.t32_next();
            }
            let op = (hw2 >> 4) & 0xF;
            let rm = (hw2 & 0xF) as u8;
            return match (l, op) {
                // TBB and TBH.
                (true, 0 | 1) => {
                    // The PC is the address 4 ahead as it is, not word-aligned.
                    let (table, index) = (self.r32(rn), self.r32(rm));
                    let offset = if op == 0 {
                        u32::from(self.mem.read_u8(u64::from(table.wrapping_add(index)))?)
                    } else {
                        u32::from(
                            self.mem
                                .read_u16(u64::from(table.wrapping_add(index << 1)))?,
                        )
                    };
                    self.pc = self.pc.wrapping_add(4).wrapping_add(offset << 1);
                    Ok(())
                }
                (false, 4 | 5 | 7) => {
                    let (size, value) = match op {
                        4 => (1, u64::from(self.r32(rt)) & 0xFF),
                        5 => (2, u64::from(self.r32(rt)) & 0xFFFF),
                        _ => (8, u64::from(self.r32(rt)) | u64::from(self.r32(rt2)) << 32),
                    };
                    let stored = self.store_exclusive_at(u64::from(base), size, [value, 0])?;
                    self.set_r32(rm, u32::from(!stored));
                    self.t32_next()
                }
                (true, 4 | 5 | 7) => {
                    let value = match op {
                        4 => u64::from(self.mem.read_u8(u64::from(base))?),
                        5 => u64::from(self.mem.read_u16(u64::from(base))?),
                        _ => {
                            let lo = self.mem.read_u32(u64::from(base))?;
                            let hi = self.mem.read_u32(u64::from(base.wrapping_add(4)))?;
                            self.set_r32(rt2, hi);
                            u64::from(lo) | u64::from(hi) << 32
                        }
                    };
                    self.set_r32(rt, value as u32);
                    self.exclusive.arm(u64::from(base), [value, 0]);
                    self.t32_next()
                }
                _ => Err(self.undefined(insn)),
            };
        }
        // LDRD and STRD with an immediate offset.
        let moved = if u {
            base.wrapping_add((hw2 & 0xFF) * 4)
        } else {
            base.wrapping_sub((hw2 & 0xFF) * 4)
        };
        let addr = if p { moved } else { base };
        if l {
            let lo = self.mem.read_u32(u64::from(addr))?;
            let hi = self.mem.read_u32(u64::from(addr.wrapping_add(4)))?;
            self.set_r32(rt, lo);
            self.set_r32(rt2, hi);
        } else {
            self.mem.write_u32(u64::from(addr), self.r32(rt))?;
            self.mem
                .write_u32(u64::from(addr.wrapping_add(4)), self.r32(rt2))?;
        }
        if w {
            self.set_r32(rn, moved);
        }
        self.t32_next()
    }

    /// The A32 data-processing opcode of the Thumb `op4`; with `Rd` the PC and `S` set,
    /// the four that have one are the compares.
    fn t32_opcode(op4: u32, rd: u8, rn: u8, set_flags: bool) -> Option<u32> {
        let compare = rd == 15 && set_flags;
        Some(match op4 {
            0b0000 if compare => 0x8,
            0b0000 => 0x0,
            0b0001 => 0xE,
            0b0010 if rn == 15 => 0xD,
            0b0010 => 0xC,
            0b0100 if compare => 0x9,
            0b0100 => 0x1,
            0b1000 if compare => 0xB,
            0b1000 => 0x4,
            0b1010 => 0x5,
            0b1011 => 0x6,
            0b1101 if compare => 0xA,
            0b1101 => 0x2,
            0b1110 => 0x3,
            _ => return None,
        })
    }

    /// Finish a data-processing form: the ALU step, the destination and the PC.
    fn t32_alu(
        &mut self,
        op4: u32,
        set_flags: bool,
        rd: u8,
        rn: u8,
        operand: u32,
        shifter_carry: bool,
    ) -> Result<()> {
        let carry = self.carry_flag();
        let a = self.r32(rn);
        // ORN and MVN are an ORR or a MOV of the inverted operand.
        let (opcode, operand) = match (op4, rn) {
            (0b0011, 15) => (0xF, operand),
            (0b0011, _) => (0xC, !operand),
            _ => match Self::t32_opcode(op4, rd, rn, set_flags) {
                Some(opcode) => (opcode, operand),
                None => return Err(self.undefined(op4)),
            },
        };
        if let Some(result) = self.a32_alu(opcode, set_flags, a, operand, carry, shifter_carry) {
            if rd == 15 {
                self.thumb_branch(result);
                return Ok(());
            }
            self.set_r32(rd, result);
        }
        self.t32_next()
    }

    fn t32_modified_immediate(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let imm12 = (hw1 >> 10 & 1) << 11 | (hw2 >> 12 & 7) << 8 | hw2 & 0xFF;
        let (operand, carry) = expand_imm_c(imm12, self.carry_flag());
        let (op4, rd, rn) = ((hw1 >> 5) & 0xF, (hw2 >> 8) & 0xF, hw1 & 0xF);
        self.t32_alu(op4, hw1 & 0x10 != 0, rd as u8, rn as u8, operand, carry)
    }

    fn t32_shifted(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let (op4, rd, rn) = ((hw1 >> 5) & 0xF, (hw2 >> 8) & 0xF, hw1 & 0xF);
        let imm5 = (hw2 >> 12 & 7) << 2 | hw2 >> 6 & 3;
        if op4 == 0b0110 {
            // PKHBT and PKHTB.
            let arm =
                0x0680_0010 | rn << 16 | rd << 12 | imm5 << 7 | (hw2 >> 5 & 1) << 6 | hw2 & 0xF;
            return self.thumb_as_arm(arm, 4);
        }
        let (ty, amount) = decode_imm_shift(((hw2 >> 4) & 3) as u8, imm5 as u8);
        let (operand, carry) = shift_c(self.r32((hw2 & 0xF) as u8), ty, amount, self.carry_flag());
        self.t32_alu(op4, hw1 & 0x10 != 0, rd as u8, rn as u8, operand, carry)
    }

    fn t32_plain_immediate(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let rn = hw1 & 0xF;
        let rd = (hw2 >> 8) & 0xF;
        let imm12 = (hw1 >> 10 & 1) << 11 | (hw2 >> 12 & 7) << 8 | hw2 & 0xFF;
        let lsb = (hw2 >> 12 & 7) << 2 | hw2 >> 6 & 3;
        let field = (hw2 & 0x1F) << 16 | rd << 12 | lsb << 7 | rn;
        match (hw1 >> 4) & 0x1F {
            op @ (0b00000 | 0b01010) => {
                // ADDW and SUBW; with the PC as the base they are ADR.
                let base = self.base32(rn as u8);
                let result = if op == 0 {
                    base.wrapping_add(imm12)
                } else {
                    base.wrapping_sub(imm12)
                };
                self.set_r32(rd as u8, result);
                self.t32_next()
            }
            0b00100 | 0b01100 => {
                let imm16 = rn << 12 | imm12;
                let value = if hw1 & 0x80 != 0 {
                    (self.r32(rd as u8) & 0xFFFF) | imm16 << 16
                } else {
                    imm16
                };
                self.set_r32(rd as u8, value);
                self.t32_next()
            }
            // SSAT, SSAT16, USAT, USAT16: the 16-bit forms are the shifted ones with no shift.
            op @ (0b10000 | 0b10010 | 0b11000 | 0b11010) => {
                let unsigned = op & 0b01000 != 0;
                let arm = if op & 0b10 != 0 && lsb == 0 {
                    (if unsigned { 0x06E0_0F30 } else { 0x06A0_0F30 })
                        | (hw2 & 0xF) << 16
                        | rd << 12
                        | rn
                } else {
                    (if unsigned { 0x06E0_0010 } else { 0x06A0_0010 }) | (op >> 1 & 1) << 6 | field
                };
                self.thumb_as_arm(arm, 4)
            }
            0b10100 => self.thumb_as_arm(0x07A0_0050 | field, 4),
            0b10110 => self.thumb_as_arm(0x07C0_0010 | field, 4),
            0b11100 => self.thumb_as_arm(0x07E0_0050 | field, 4),
            _ => Err(self.undefined(insn)),
        }
    }

    /// `B`, `BL`, `BLX` and the miscellaneous control group.
    fn t32_branch(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let pc4 = self.pc.wrapping_add(4);
        let s = (hw1 >> 10) & 1;
        let (j1, j2) = ((hw2 >> 13) & 1, (hw2 >> 11) & 1);
        let long_offset = |imm10: u32, imm11: u32| {
            let (i1, i2) = (!(j1 ^ s) & 1, !(j2 ^ s) & 1);
            sext(s << 24 | i1 << 23 | i2 << 22 | imm10 << 12 | imm11 << 1, 25)
        };
        match (hw2 >> 12) & 0b101 {
            0b000 if (hw1 >> 7) & 7 != 7 => {
                if self.condition_holds(((hw1 >> 6) & 0xF) as u8) {
                    let imm =
                        s << 20 | j2 << 19 | j1 << 18 | (hw1 & 0x3F) << 12 | (hw2 & 0x7FF) << 1;
                    self.pc = pc4.wrapping_add(sext(imm, 21));
                    return Ok(());
                }
                self.t32_next()
            }
            0b000 => self.t32_control(hw1, hw2),
            0b001 => {
                self.pc = pc4.wrapping_add(long_offset(hw1 & 0x3FF, hw2 & 0x7FF));
                Ok(())
            }
            0b101 => {
                self.regs[14] = u64::from(pc4 | 1);
                self.pc = pc4.wrapping_add(long_offset(hw1 & 0x3FF, hw2 & 0x7FF));
                Ok(())
            }
            0b100 if hw2 & 1 == 0 => {
                self.regs[14] = u64::from(pc4 | 1);
                self.thumb = false;
                self.pc = (pc4 & !3).wrapping_add(long_offset(hw1 & 0x3FF, hw2 & 0x7FF));
                Ok(())
            }
            _ => Err(self.undefined(insn)),
        }
    }

    /// `MSR`, `MRS`, hints and barriers.
    fn t32_control(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        match hw1 {
            0xF380..=0xF38F if hw2 & 0xF3FF == 0x8000 => {
                self.thumb_as_arm(0x0120_F000 | ((hw2 >> 10) & 3) << 18 | (hw1 & 0xF), 4)
            }
            0xF3EF if hw2 & 0xF0FF == 0x8000 => {
                self.thumb_as_arm(0x010F_0000 | ((hw2 >> 8) & 0xF) << 12, 4)
            }
            0xF3AF if hw2 & 0xFF00 == 0x8000 => self.t32_next(),
            0xF3BF if hw2 & 0xFF00 == 0x8F00 => {
                match (hw2 >> 4) & 0xF {
                    0x2 => self.exclusive.clear(),
                    0x4..=0x6 => std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst),
                    _ => return Err(self.undefined(insn)),
                }
                self.t32_next()
            }
            _ => Err(self.undefined(insn)),
        }
    }

    /// Loads and stores of one register, with an immediate or register offset.
    fn t32_load_store(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let size = (hw1 >> 5) & 3;
        let signed = hw1 & 0x100 != 0;
        let load = hw1 & 0x10 != 0;
        let (rn, rt) = ((hw1 & 0xF) as u8, (hw2 >> 12) as u8);
        if size == 3 || (signed && !load) || (signed && size == 2) {
            return Err(self.undefined(insn));
        }
        // A load into the PC of a byte or halfword is a hint (PLD, PLI), which does nothing.
        if load && rt == 15 && size != 2 {
            return self.t32_next();
        }
        let mut access = Access {
            rn,
            rt,
            load,
            pre: true,
            add: true,
            wb: false,
        };
        let offset = if rn == 15 && load {
            access.add = hw1 & 0x80 != 0;
            hw2 & 0xFFF
        } else if hw1 & 0x80 != 0 {
            hw2 & 0xFFF
        } else if hw2 & 0x800 == 0 {
            self.r32((hw2 & 0xF) as u8) << ((hw2 >> 4) & 3)
        } else {
            access.pre = hw2 & 0x400 != 0;
            access.add = hw2 & 0x200 != 0;
            access.wb = hw2 & 0x100 != 0;
            hw2 & 0xFF
        };
        let fall = self.pc.wrapping_add(4);
        if size == 1 || signed {
            let sh = match (signed, size) {
                (false, _) => 0b01,
                (true, 0) => 0b10,
                _ => 0b11,
            };
            self.a32_extra(access, sh, offset)?;
        } else {
            self.a32_single(access, size == 0, offset)?;
        }
        // Only a word load into the PC branches, and it has.
        if !(load && rt == 15) {
            self.pc = fall;
        }
        Ok(())
    }

    /// Shifts by register, extends, parallel arithmetic and the miscellaneous operations.
    fn t32_register(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let (rn, rd, rm) = (hw1 & 0xF, (hw2 >> 8) & 0xF, hw2 & 0xF);
        if hw1 & 0x80 == 0 {
            if hw2 & 0xF0 == 0 {
                // LSL, LSR, ASR and ROR by a register.
                let carry = self.carry_flag();
                let (result, carry) = shift_by_register(
                    self.r32(rn as u8),
                    ((hw1 >> 5) & 3) as u8,
                    self.r32(rm as u8),
                    carry,
                );
                self.set_r32(rd as u8, result);
                if hw1 & 0x10 != 0 {
                    self.set_nzc32(result, carry);
                }
                return self.t32_next();
            }
            if hw2 & 0xF0C0 != 0xF080 {
                return Err(self.undefined(insn));
            }
            // SXTAH through UXTAB, which are the plain extends when Rn is the PC.
            let arm = match (hw1 >> 4) & 7 {
                0 => 0x06B0_0070,
                1 => 0x06F0_0070,
                2 => 0x0680_0070,
                3 => 0x06C0_0070,
                4 => 0x06A0_0070,
                _ => 0x06E0_0070,
            };
            return self.thumb_as_arm(arm | rn << 16 | rd << 12 | ((hw2 >> 4) & 3) << 10 | rm, 4);
        }
        if hw2 & 0xF080 == 0xF000 {
            // Parallel addition and subtraction.
            let kind = match (hw2 >> 4) & 7 {
                0 => 0b001,
                1 => 0b010,
                2 => 0b011,
                4 => 0b101,
                5 => 0b110,
                6 => 0b111,
                _ => return Err(self.undefined(insn)),
            };
            let op2 = match (hw1 >> 4) & 7 {
                1 => 0b000,
                2 => 0b001,
                6 => 0b010,
                5 => 0b011,
                0 => 0b100,
                4 => 0b111,
                _ => return Err(self.undefined(insn)),
            };
            return self.thumb_as_arm(
                0x0600_0F10 | kind << 20 | rn << 16 | rd << 12 | op2 << 5 | rm,
                4,
            );
        }
        if hw1 & 0x40 != 0 || hw2 & 0xF0C0 != 0xF080 {
            return Err(self.undefined(insn));
        }
        let arm = match ((hw1 >> 4) & 3, (hw2 >> 4) & 3) {
            // QADD, QDADD, QSUB and QDSUB, in the A32 order.
            (0, op) => {
                [0x0100_0050, 0x0140_0050, 0x0120_0050, 0x0160_0050][op as usize]
                    | rn << 16
                    | rd << 12
                    | rm
            }
            (1, op) => {
                [0x06BF_0F30, 0x06BF_0FB0, 0x06FF_0F30, 0x06FF_0FB0][op as usize] | rd << 12 | rm
            }
            (2, 0) => 0x0680_0FB0 | rn << 16 | rd << 12 | rm,
            (3, 0) => 0x016F_0F10 | rd << 12 | rm,
            _ => return Err(self.undefined(insn)),
        };
        self.thumb_as_arm(arm, 4)
    }

    /// Multiplies and the media multiplies.
    fn t32_multiply(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let (rn, ra, rd, rm) = (hw1 & 0xF, hw2 >> 12, (hw2 >> 8) & 0xF, hw2 & 0xF);
        let (n, m) = ((hw2 >> 5) & 1, (hw2 >> 4) & 1);
        let fields = rd << 16 | ra << 12 | rm << 8 | rn;
        let no_acc = rd << 16 | rm << 8 | rn;
        let arm = match ((hw1 >> 4) & 7, (hw2 >> 4) & 3) {
            (0, 0) if ra == 15 => 0x0000_0090 | no_acc,
            (0, 0) => 0x0020_0090 | fields,
            (0, 1) => 0x0060_0090 | fields,
            (1, _) if ra == 15 => 0x0160_0080 | no_acc | m << 6 | n << 5,
            (1, _) => 0x0100_0080 | fields | m << 6 | n << 5,
            (2, op) => 0x0700_0010 | fields | (op & 1) << 5,
            (3, op) if ra == 15 => 0x0120_00A0 | no_acc | (op & 1) << 6,
            (3, op) => 0x0120_0080 | fields | (op & 1) << 6,
            (4, op) => 0x0700_0050 | fields | (op & 1) << 5,
            (5, op) => 0x0750_0010 | fields | (op & 1) << 5,
            (6, op) => 0x0750_00D0 | fields | (op & 1) << 5,
            (7, 0) => 0x0780_0010 | fields,
            _ => return Err(self.undefined(insn)),
        };
        self.thumb_as_arm(arm, 4)
    }

    /// Long multiplies, `SDIV` and `UDIV`.
    fn t32_long_multiply(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let (rn, lo, hi, rm) = (hw1 & 0xF, hw2 >> 12, (hw2 >> 8) & 0xF, hw2 & 0xF);
        let op2 = (hw2 >> 4) & 0xF;
        let fields = hi << 16 | lo << 12 | rm << 8 | rn;
        let arm = match ((hw1 >> 4) & 7, op2) {
            (0, 0) => 0x00C0_0090 | fields,
            (1, 0xF) => 0x0710_F010 | hi << 16 | rm << 8 | rn,
            (2, 0) => 0x0080_0090 | fields,
            (3, 0xF) => 0x0730_F010 | hi << 16 | rm << 8 | rn,
            (4, 0) => 0x00E0_0090 | fields,
            (4, 0x8..=0xB) => 0x0140_0080 | fields | (op2 & 1) << 6 | (op2 >> 1 & 1) << 5,
            (4, 0xC | 0xD) => 0x0740_0010 | fields | (op2 & 1) << 5,
            (5, 0xC | 0xD) => 0x0740_0050 | fields | (op2 & 1) << 5,
            (6, 0) => 0x00A0_0090 | fields,
            (6, 6) => 0x0040_0090 | fields,
            _ => return Err(self.undefined(insn)),
        };
        self.thumb_as_arm(arm, 4)
    }

    /// VFP, Advanced SIMD and the coprocessor transfers: their A32 encodings under another prefix.
    fn t32_coprocessor(&mut self, hw1: u32, hw2: u32) -> Result<()> {
        let insn = hw1 << 16 | hw2;
        let low = (hw1 & 0xFF) << 16 | hw2;
        let arm = match hw1 >> 8 {
            0xEC..=0xEE => 0xE000_0000 | (hw1 & 0xFFF) << 16 | hw2,
            0xEF | 0xFF => 0xF200_0000 | (hw1 >> 12 & 1) << 24 | low,
            0xF9 => 0xF400_0000 | low,
            0xFE => 0xFE00_0000 | low,
            _ => return Err(self.undefined(insn)),
        };
        let fall = self.pc.wrapping_add(4);
        let result = self.execute_a32(arm);
        self.pc = fall;
        result
    }
}
