//! The AArch32 (A32) execution state.
//!
//! Horizon runs some retail titles in AArch32 — Mario Kart 8 Deluxe
//! (`0100152000022000`) is one, and its `main.npdm` says so in bit 0 of the
//! flags byte at 0x0C. Such a title's `rtld` opens with the 32-bit module
//! prologue, `ea000000` (`b #+8`) followed by the offset to `MOD0`; run
//! through the A64 decoder that first word is a valid `ANDS x0, x0, x0`, so
//! execution falls into the offset word and faults on data.
//!
//! # One CPU, two states
//!
//! Everything above the instruction set — the syscalls, IPC, the services,
//! the GPU — is the same Horizon in either state, so there is one [`Cpu`] and
//! one register file, with [`ExecMode`] saying how to read it. `r0`..`r14`
//! alias the low halves of `X0`..`X14`; `r13` is SP and `r14` is LR, which is
//! why neither of A64's separate [`super::SP_SLOT`] nor `X30` is used in this
//! state. `r15` is not stored at all: reading it yields `pc + 8`, the value
//! ARM's pipeline made architectural, and writing it branches.
//!
//! N/Z/C/V share [`Cpu::nzcv`] with A64 — the bit positions and the condition
//! encoding are identical, so [`Cpu::condition_holds`] serves both. Q and GE
//! are AArch32's alone and live in [`Cpu::cpsr_q`] and [`Cpu::cpsr_ge`].
//!
//! # No Thumb
//!
//! T32 is not implemented, and measurement says it is not needed: across the
//! 4.8M instruction words of Mario Kart 8 Deluxe's eight modules there is
//! exactly one `BLX` immediate — the only encoding that statically switches
//! to Thumb — which at that rate is a literal pool word, not a call. An
//! interworking branch to an odd address is therefore a diagnosable error
//! rather than a silent wrong-mode execution; see [`Cpu::a32_write_pc`].

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
pub(self) use vfp::vfp_mnemonic;

/// Which instruction set the current thread is executing.
///
/// Per thread rather than per process: `svcCreateThread` in a 32-bit process
/// makes 32-bit threads, but the field travels with the context either way,
/// so the two can never disagree about a thread that was switched away from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    A64,
    A32,
}
impl Cpu {
    /// `r0`..`r14`, and `r15` as the pipeline exposes it: the address of the
    /// instruction being executed plus 8.
    #[inline(always)]
    pub(super) fn r32(&self, r: u8) -> u32 {
        if r == 15 {
            self.pc.wrapping_add(8)
        } else {
            self.regs[(r & 0xF) as usize] as u32
        }
    }

    /// Write `r0`..`r14`. Writes to `r15` are branches and do not come here —
    /// see [`Cpu::a32_write_pc`].
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

    /// Set N and Z from a result and C from the shifter, leaving V alone —
    /// what the logical operations do.
    #[inline(always)]
    pub(super) fn set_nzc32(&mut self, result: u32, carry: bool) {
        let n = u32::from(result >> 31 != 0) << 31;
        let z = u32::from(result == 0) << 30;
        let c = u32::from(carry) << 29;
        self.nzcv = (self.nzcv & 0x1000_0000) | n | z | c;
    }

    /// The 32-bit adder every arithmetic data-processing instruction shares.
    /// Subtraction arrives here as addition of the inverted operand with a
    /// carry in of one, exactly as the architecture defines it.
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

    /// Branch, honouring the interworking rule that bit 0 selects Thumb.
    ///
    /// Nothing implements T32, so rather than execute ARM words at a Thumb
    /// address — which produces a fault somewhere else entirely, with nothing
    /// to say the state was wrong — a switch is reported where it happens.
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

    /// Execute one A32 instruction. The caller has already fetched it.
    pub(super) fn execute_a32(&mut self, insn: u32) -> Result<()> {
        let cond = (insn >> 28) & 0xF;
        if cond == 0xF {
            return self.a32_unconditional(insn);
        }
        // Every A32 instruction is conditional, and one that does not run
        // still costs its own advance.
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
    /// Which instruction set the running thread executes.
    pub fn mode(&self) -> ExecMode {
        self.mode
    }

    /// Put the core into AArch32 and lay the register file out the way a
    /// 32-bit process expects: `r13` is the stack pointer rather than A64's
    /// separate [`super::SP_SLOT`], and `r14` the link register rather than
    /// `X30`. Call it before the entry point runs.
    pub fn set_mode(&mut self, mode: ExecMode) {
        if mode == self.mode {
            return;
        }
        match mode {
            ExecMode::A32 => {
                self.regs[13] = self.regs[super::SP_SLOT];
                self.regs[14] = self.regs[30];
                // The two trampolines `bootstrap` wrote are A64 words. A
                // 32-bit process returns into them just the same, so they have
                // to be re-assembled in the state that will execute them, or
                // `main` returning lands on a `.word` fault instead of a clean
                // exit.
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

    /// Disassemble with whichever decoder the core is actually running, so a
    /// fault trace of 32-bit code is not annotated with A64 mnemonics.
    pub(super) fn disassemble_for_mode(&self, insn: u32) -> String {
        match self.mode {
            ExecMode::A64 => crate::disasm::disassemble(insn),
            ExecMode::A32 => disassemble_a32(insn),
        }
    }
}

/// The AArch32 syscall ABI.
///
/// Horizon numbers its syscalls the same in both execution states and passes
/// the arguments in the same low registers, so most of `svc.rs` needs no help:
/// `X0`..`X7` and `r0`..`r7` are the same slots of the same register file, and
/// a 32-bit argument zero-extends into a 64-bit one by itself.
///
/// What does not carry over is the arguments that *are* 64 bits. AArch32 has
/// no register wide enough, so the kernel splits each across a pair — and the
/// pairs are not always adjacent, nor are the remaining arguments always in
/// the same positions. `svcWaitSynchronization`'s timeout is `r0:r3` while its
/// handle list stays in `r1`; `svcCreateThread` takes its priority in `r0`
/// where A64 takes it in `X4`. The mappings below are Eden's
/// `SvcWrap_*64From32` wrappers in `core/hle/kernel/svc.cpp`, which are
/// generated from the kernel's own definitions.
///
/// These are accessors rather than a shuffle of the register file around the
/// dispatch, because a blocking syscall rewinds onto its own `svc` and is
/// reissued: anything that rewrote the argument registers on the way out would
/// corrupt the arguments the next attempt reads.
impl Cpu {
    /// A 64-bit syscall argument: one register in A64, the pair `lo:hi` in
    /// AArch32, low half first.
    pub(super) fn svc_arg64(&self, a64: u8, lo: u8, hi: u8) -> u64 {
        match self.mode {
            ExecMode::A64 => self.reg_at(a64),
            ExecMode::A32 => u64::from(self.r32(lo)) | (u64::from(self.r32(hi)) << 32),
        }
    }

    /// A 64-bit syscall result, scattered across a register pair in AArch32.
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
