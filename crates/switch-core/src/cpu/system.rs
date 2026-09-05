//! System instructions: MRS/MSR, barriers, hints and the cache maintenance
//! operations (`DC ZVA`) that memset implementations rely on.
//!
//! Reading one is two steps: [`SysOp::of`] says what the encoding names, and
//! the `Cpu` methods below carry it out. [`Cpu::system`] runs the two back to
//! back, which is what the interpreter needs; the block translator runs the
//! classification once and keeps the answer in its op, so a hot `MRS` walks
//! the table when it is translated rather than every time it executes.
//! `MRS TPIDRRO_EL0` (how every `nnSdk` thread finds its own TLS) sits near
//! the end of that table, which is what made the difference worth having.

use super::bits::{FPCR_MASK, FPSR_MASK};
use super::power::CLOCK_RATES_HZ;
use super::Cpu;
use crate::{Error, Result};

/// The generic timer's rate. `nn::os::GetSystemTickFrequency` returns it
/// from a constant of its own, so the counter has to be on this scale
/// whatever the CPU is clocked at.
pub(super) const TICK_HZ: u32 = 19_200_000;

/// The system register an `MRS`/`MSR` names, resolved from its
/// `op0:op1:CRn:CRm:op2` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SysReg {
    Nzcv,
    /// `TPIDR_EL0`, which guest code may write.
    Tpidr,
    /// `TPIDRRO_EL0`, the kernel-fixed TLS base. Read-only at EL0, so an
    /// `MSR` to it is ignored rather than refused.
    TpidrRo,
    Fpcr,
    Fpsr,
    /// `CNTPCT_EL0`, and `CNTVCT_EL0` for the same count: EL0 has no virtual
    /// offset here. **This is the clock, not `svcGetSystemTick`**:
    /// `nn::os::GetSystemTick` is `mrs x0, cntpct_el0; ret`, so a retail
    /// title's frame timing never reaches a syscall at all.
    SystemTick,
    /// A register that always reads the same value: the two Cortex-A57
    /// constants this emulator reports, and zero for everything it does not
    /// model. Writes to these are dropped. Both constants fit in 32 bits,
    /// which keeps [`super::jit::ir::Op`] to one 64-bit word.
    Fixed(u32),
}

impl SysReg {
    // The literals below are grouped as op0_op1_CRn_CRm_op2, the encoding the
    // comments name in `3:3:13:0:2` form.
    fn of(insn: u32) -> SysReg {
        let op0 = (insn >> 19) & 0b11;
        let op1 = (insn >> 16) & 0b111;
        let crn = (insn >> 12) & 0xF;
        let crm = (insn >> 8) & 0xF;
        let op2 = (insn >> 5) & 0b111;
        match (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2 {
            0b11_011_0100_0010_000 => SysReg::Nzcv,
            // TPIDR_EL0 (3:3:13:0:2): freely writable by guest code.
            0b11_011_1101_0000_010 => SysReg::Tpidr,
            0b11_011_1101_0000_011 => SysReg::TpidrRo,
            // FPCR (3:3:4:4:0) and FPSR (3:3:4:4:1). The op1 field is 3 at
            // EL0, not 0: reading it as 3:0:... meant a guest's `mrs x0,
            // fpcr` fell through to the catch-all zero.
            0b11_011_0100_0100_000 => SysReg::Fpcr,
            0b11_011_0100_0100_001 => SysReg::Fpsr,
            // DCZID_EL0: BS=4, a 64-byte `DC ZVA` block. musl/newlib memset
            // strides the cache-zero loop with `4 << BS`; BS=0 runs away.
            0b11_011_0000_0000_111 => SysReg::Fixed(4),
            // CTR_EL0: the Cortex-A57 value, 64-byte I- and D-cache lines,
            // 64-byte ERG/CWG. Cache-flush loops stride by `4 << DminLine`,
            // so reporting 0 made NX-Shell's flush walk its buffers 4 bytes
            // at a time.
            0b11_011_0000_0000_001 => SysReg::Fixed(0x8444_C004),
            // CNTFRQ_EL0 (3:3:14:0:0), CNTPCT_EL0 (…:1) and CNTVCT_EL0 (…:2).
            0b11_011_1110_0000_000 => SysReg::Fixed(TICK_HZ),
            0b11_011_1110_0000_001 | 0b11_011_1110_0000_010 => SysReg::SystemTick,
            _ => SysReg::Fixed(0),
        }
    }
}

/// What a system instruction does, once its fields have been read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SysOp {
    /// A hint, barrier, cache maintenance operation or PSTATE-immediate write
    /// that retires with no effect. There are no caches to maintain here,
    /// memory is always coherent, and libnx flushes the data cache around
    /// every buffer it hands to the GPU.
    Nop,
    /// `MRS Xt, <sysreg>`. `rd` is a resolved register-file slot.
    Mrs { rd: u8, reg: SysReg },
    /// `MSR <sysreg>, Xt`.
    Msr { rt: u8, reg: SysReg },
    /// The one `MSR` immediate form that has an effect.
    MsrNzcvImm { imm: u8 },
    /// `DC ZVA Xt`: zero the 64-byte block Xt points into.
    DcZva { rt: u8 },
    /// `CLREX`: clear the local exclusive monitor.
    ///
    /// It sits in the barrier group and is not a barrier: a `STXR` after one
    /// must fail. It was retiring as a hint, so a guest that abandoned a
    /// read-modify-write kept its reservation, and the store it had given up
    /// on could still land.
    ClearExclusive,
    /// An encoding this does not place. The caller reports it.
    Unhandled,
}

impl SysOp {
    pub(super) fn of(insn: u32) -> SysOp {
        // HINT (incl. NOP) and the barriers. CLREX shares the group and is
        // the one member of it with an architectural effect: CRn == 0011 with
        // op2 == 010, where DMB/DSB/ISB take op2 100/101/110.
        if (insn >> 16) & 0xFFFF == 0xD503 {
            if (insn >> 12) & 0xF == 0b0011 && (insn >> 5) & 0b111 == 0b010 {
                return SysOp::ClearExclusive;
            }
            return SysOp::Nop;
        }
        let l = (insn >> 21) & 1;
        let op0 = (insn >> 19) & 0b11;
        let op1 = (insn >> 16) & 0b111;
        let crn = (insn >> 12) & 0xF;
        let crm = (insn >> 8) & 0xF;
        let op2 = (insn >> 5) & 0b111;
        let rt = (insn & 0x1F) as u8;

        if l == 1 {
            return SysOp::Mrs {
                rd: Cpu::zr_write_slot(rt),
                reg: SysReg::of(insn),
            };
        }
        if op0 == 0 {
            // MSR (immediate). Only the PSTATE write has an effect; DAIF,
            // SPSel and the rest retire with none.
            return match (op1, crn, crm, op2) {
                (0b010 | 0b011, 0b0100, 0b0010, 0b000) => SysOp::MsrNzcvImm {
                    imm: ((insn >> 8) & 0xF) as u8,
                },
                _ => SysOp::Nop,
            };
        }
        if op0 == 1 && crn == 7 {
            if op1 == 3 && crm == 4 && op2 == 1 {
                return SysOp::DcZva { rt };
            }
            return SysOp::Nop;
        }
        if op0 == 2 || op0 == 3 {
            return SysOp::Msr {
                rt,
                reg: SysReg::of(insn),
            };
        }
        SysOp::Unhandled
    }
}

impl Cpu {
    /// Decode and run one system instruction, retiring it to `next_pc`.
    pub(super) fn system(&mut self, insn: u32, next_pc: u32) -> Result<()> {
        let op = SysOp::of(insn);
        if op == SysOp::Unhandled {
            return Err(Error::Cpu(format!(
                "unimplemented system instruction 0x{:08x} at {:#x}",
                insn, self.pc
            )));
        }
        self.exec_sys(op)?;
        self.pc = next_pc;
        Ok(())
    }

    /// The 19.2 MHz generic-timer count, which both `CNTPCT_EL0` and
    /// `svcGetSystemTick` report. One emulated instruction stands for one
    /// cycle of the CPU `apm` reports, so a tick is worth about 53 of them.
    pub(super) fn system_tick(&self) -> u64 {
        (u128::from(self.cycles) * u128::from(TICK_HZ) / u128::from(CLOCK_RATES_HZ[0])) as u64
    }

    /// Carry out an already-classified system instruction. `Unhandled` is the
    /// caller's to report, and does nothing here.
    #[inline(always)]
    pub(super) fn exec_sys(&mut self, op: SysOp) -> Result<()> {
        match op {
            SysOp::Nop | SysOp::Unhandled => {}
            SysOp::Mrs { rd, reg } => {
                let val = match reg {
                    SysReg::Nzcv => u64::from(self.nzcv),
                    SysReg::Tpidr => self.tpidr_rw,
                    SysReg::TpidrRo => self.tpidr,
                    SysReg::Fpcr => u64::from(self.fpcr),
                    SysReg::Fpsr => u64::from(self.fpsr),
                    SysReg::SystemTick => self.system_tick(),
                    SysReg::Fixed(v) => u64::from(v),
                };
                self.set_reg_at(rd, val);
            }
            SysOp::Msr { rt, reg } => match reg {
                SysReg::Nzcv => self.nzcv = self.reg_at(rt) as u32,
                // Only the bits the architecture defines stick, so a guest
                // that reads back what it wrote sees the same value.
                SysReg::Fpcr => self.fpcr = self.reg_at(rt) as u32 & FPCR_MASK,
                SysReg::Fpsr => self.fpsr = self.reg_at(rt) as u32 & FPSR_MASK,
                SysReg::Tpidr => self.tpidr_rw = self.reg_at(rt),
                // TPIDRRO_EL0 is read-only at EL0, so a guest write to it is
                // ignored rather than refused.
                SysReg::TpidrRo | SysReg::SystemTick | SysReg::Fixed(_) => {}
            },
            SysOp::MsrNzcvImm { imm } => self.nzcv = u32::from(imm),
            SysOp::ClearExclusive => self.exclusive = None,
            SysOp::DcZva { rt } => {
                // Eight doubleword stores rather than sixty-four byte ones,
                // which is eight page lookups instead of sixty-four. (Not
                // `fill_le`: it stamps a 512-byte pattern before copying,
                // which a block this small never amortizes.)
                let addr = self.reg_at(rt) as u32 & !0x3F;
                for i in 0..8u32 {
                    self.mem.write_u64(addr.wrapping_add(i * 8), 0)?;
                }
            }
        }
        Ok(())
    }
}
