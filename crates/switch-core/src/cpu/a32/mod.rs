//! The AArch32 (A32) execution state.
//!
//! `r0`..`r14` alias the low halves of `X0`..`X14`, with `r13` as SP and `r14` as LR;
//! `r15` is not stored and reads as `pc + 8`. N/Z/C/V are shared with A64. T32 is not implemented.

mod branch;
mod dataproc;
mod disasm;
mod loadstore;
mod media;
mod neon;
mod shift;
mod vfp;

use super::Cpu;
use crate::{Error, Result};

pub use disasm::disassemble_a32;
use vfp::vfp_mnemonic;

/// Which instruction set the current thread is executing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    A64,
    A32,
}
impl Cpu {
    /// `r0`..`r14`, and `r15` as the address of the current instruction plus 8.
    #[inline(always)]
    pub(super) fn r32(&self, r: u8) -> u32 {
        if r == 15 {
            self.pc.wrapping_add(8)
        } else {
            self.regs[(r & 0xF) as usize] as u32
        }
    }

    /// Write `r0`..`r14`; writes to `r15` go through [`Cpu::a32_write_pc`].
    #[inline(always)]
    pub(super) fn set_r32(&mut self, r: u8, val: u32) {
        debug_assert!(r != 15, "a write to r15 is a branch, not a register write");
        self.regs[(r & 0xF) as usize] = u64::from(val);
    }

    #[inline(always)]
    pub(super) fn carry_flag(&self) -> bool {
        (self.nzcv >> 29) & 1 != 0
    }

    /// Set N and Z from a result, leaving C and V alone.
    #[inline(always)]
    pub(super) fn set_nz32(&mut self, result: u32) {
        let n = u32::from(result >> 31 != 0) << 31;
        let z = u32::from(result == 0) << 30;
        self.nzcv = (self.nzcv & 0x3000_0000) | n | z;
    }

    /// Set N and Z from a result and C from the shifter, leaving V alone.
    #[inline(always)]
    pub(super) fn set_nzc32(&mut self, result: u32, carry: bool) {
        let n = u32::from(result >> 31 != 0) << 31;
        let z = u32::from(result == 0) << 30;
        let c = u32::from(carry) << 29;
        self.nzcv = (self.nzcv & 0x1000_0000) | n | z | c;
    }

    /// The 32-bit adder; subtraction is addition of the inverted operand with carry in 1.
    #[inline(always)]
    pub(super) fn add32_flags(&mut self, a: u32, b: u32, carry_in: bool, set_flags: bool) -> u32 {
        let sum = u64::from(a) + u64::from(b) + u64::from(carry_in);
        let result = sum as u32;
        if set_flags {
            let carry = sum >> 32 != 0;
            let overflow = ((a ^ result) & (b ^ result)) >> 31 != 0;
            let n = u32::from(result >> 31 != 0) << 31;
            let z = u32::from(result == 0) << 30;
            self.nzcv = n | z | (u32::from(carry) << 29) | (u32::from(overflow) << 28);
        }
        result
    }

    /// Branch; an interworking switch to Thumb is reported as an error.
    #[inline]
    pub(super) fn a32_write_pc(&mut self, target: u32) -> Result<()> {
        if target & 1 != 0 {
            return Err(Error::Cpu(format!(
                "branch to Thumb code at {:#010x} from pc={:#010x}; T32 is not implemented",
                target, self.pc
            )));
        }
        self.pc = target & !3;
        Ok(())
    }

    pub(super) fn execute_a32(&mut self, insn: u32) -> Result<()> {
        let cond = (insn >> 28) & 0xF;
        if cond == 0xF {
            return self.a32_unconditional(insn);
        }
        // A skipped instruction still advances the pc.
        if !self.condition_holds(cond as u8) {
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }
        match (insn >> 25) & 0x7 {
            0b000 | 0b001 => self.a32_data_processing(insn),
            0b010 => self.a32_load_store_imm(insn),
            0b011 => {
                if insn & 0x10 != 0 {
                    self.a32_media(insn)
                } else {
                    self.a32_load_store_reg(insn)
                }
            }
            0b100 => self.a32_load_store_multiple(insn),
            0b101 => self.a32_branch(insn),
            0b110 => self.a32_coproc_load_store(insn),
            _ => {
                if (insn >> 24) & 1 != 0 {
                    let imm = insn & 0x00FF_FFFF;
                    self.pc = self.pc.wrapping_add(4);
                    self.syscall(imm as u16)
                } else {
                    self.a32_coproc(insn)
                }
            }
        }
    }
}

impl Cpu {
    pub fn mode(&self) -> ExecMode {
        self.mode
    }

    /// Call before the entry point runs.
    pub fn set_mode(&mut self, mode: ExecMode) {
        if mode == self.mode {
            return;
        }
        match mode {
            ExecMode::A32 => {
                self.regs[13] = self.regs[super::SP_SLOT];
                self.regs[14] = self.regs[30];
                // The trampolines `bootstrap` wrote are A64; re-assemble them as A32.
                let _ = self
                    .mem
                    .write_u32(super::SELF_RETURN_TRAMPOLINE, 0xEF00_0007); // svc #7
                let _ = self
                    .mem
                    .write_u32(super::SELF_RETURN_TRAMPOLINE + 4, 0xEAFF_FFFE); // b .
                let _ = self
                    .mem
                    .write_u32(super::THREAD_EXIT_TRAMPOLINE, 0xEF00_000A); // svc #0xa
                let _ = self
                    .mem
                    .write_u32(super::THREAD_EXIT_TRAMPOLINE + 4, 0xEAFF_FFFE); // b .
            }
            ExecMode::A64 => {
                self.regs[super::SP_SLOT] = self.regs[13];
                self.regs[30] = self.regs[14];
            }
        }
        self.mode = mode;
        if let Some(thread) = self.threads.get_mut(self.current_thread) {
            thread.mode = mode;
        }
    }

    pub(super) fn disassemble_for_mode(&self, insn: u32) -> String {
        match self.mode {
            ExecMode::A64 => crate::disasm::disassemble(insn),
            ExecMode::A32 => disassemble_a32(insn),
        }
    }
}

/// The AArch32 syscall ABI, which splits 64-bit arguments across register pairs
/// (mappings from Eden's `SvcWrap_*64From32`). Accessors, not a register shuffle,
/// because a blocking syscall is reissued with the same registers.
impl Cpu {
    /// A 64-bit syscall argument: one register in A64, the pair `lo:hi` in AArch32.
    pub(super) fn svc_arg64(&self, a64: u8, lo: u8, hi: u8) -> u64 {
        match self.mode {
            ExecMode::A64 => self.reg_at(a64),
            ExecMode::A32 => u64::from(self.r32(lo)) | (u64::from(self.r32(hi)) << 32),
        }
    }

    /// A 64-bit syscall result, split across a register pair in AArch32.
    pub(super) fn svc_out64(&mut self, a64: u8, lo: u8, hi: u8, val: u64) {
        match self.mode {
            ExecMode::A64 => self.set_reg(a64, val),
            ExecMode::A32 => {
                self.set_r32(lo, val as u32);
                self.set_r32(hi, (val >> 32) as u32);
            }
        }
    }
}
