//! System instructions: MRS/MSR, barriers, hints and `DC ZVA`.
//!
//! [`SysOp::of`] classifies an encoding; the `Cpu` methods below execute it.

use super::bits::{FPCR_MASK, FPSR_MASK};
use super::power::CLOCK_RATES_HZ;
use super::Cpu;
use crate::{Error, Result};

/// The generic timer's rate, which `nn::os::GetSystemTickFrequency` hardcodes.
pub(super) const TICK_HZ: u32 = 19_200_000;

/// The system register an `MRS`/`MSR` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SysReg {
    Nzcv,
    Tpidr,
    /// `TPIDRRO_EL0`, the TLS base. An EL0 `MSR` to it is ignored.
    TpidrRo,
    Fpcr,
    Fpsr,
    /// `CNTPCT_EL0` and `CNTVCT_EL0`: the clock `nn::os::GetSystemTick` reads.
    SystemTick,
    /// A constant register; writes are dropped. Values fit in 32 bits.
    Fixed(u32),
}

impl SysReg {
    // Literals are grouped as op0_op1_CRn_CRm_op2.
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
            // FPCR (3:3:4:4:0) and FPSR (3:3:4:4:1).
            0b11_011_0100_0100_000 => SysReg::Fpcr,
            0b11_011_0100_0100_001 => SysReg::Fpsr,
            // DCZID_EL0: BS=4, a 64-byte `DC ZVA` block.
            0b11_011_0000_0000_111 => SysReg::Fixed(4),
            // CTR_EL0: the Cortex-A57 value, 64-byte cache lines and ERG/CWG.
            0b11_011_0000_0000_001 => SysReg::Fixed(0x8444_C004),
            // CNTFRQ_EL0 (3:3:14:0:0), CNTPCT_EL0 (…:1) and CNTVCT_EL0 (…:2).
            0b11_011_1110_0000_000 => SysReg::Fixed(TICK_HZ),
            0b11_011_1110_0000_001 | 0b11_011_1110_0000_010 => SysReg::SystemTick,
            _ => SysReg::Fixed(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SysOp {
    /// A hint, barrier, cache op or PSTATE-immediate write with no effect.
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
    ClearExclusive,
    /// An encoding this does not place. The caller reports it.
    Unhandled,
}

impl SysOp {
    pub(super) fn of(insn: u32) -> SysOp {
        // HINT and barriers. CLREX is CRn == 0011, op2 == 010.
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
            // MSR (immediate). Only the PSTATE write has an effect.
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

    /// The 19.2 MHz generic-timer count, read by `CNTPCT_EL0` and `svcGetSystemTick`.
    pub(crate) fn system_tick(&self) -> u64 {
        (u128::from(self.cycles) * u128::from(TICK_HZ) / u128::from(CLOCK_RATES_HZ[0])) as u64
    }

    /// Execute a classified system instruction. `Unhandled` is the caller's to report.
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
                // Only the architecturally defined bits stick.
                SysReg::Fpcr => self.fpcr = self.reg_at(rt) as u32 & FPCR_MASK,
                SysReg::Fpsr => self.fpsr = self.reg_at(rt) as u32 & FPSR_MASK,
                SysReg::Tpidr => self.tpidr_rw = self.reg_at(rt),
                SysReg::TpidrRo | SysReg::SystemTick | SysReg::Fixed(_) => {}
            },
            SysOp::MsrNzcvImm { imm } => self.nzcv = u32::from(imm),
            SysOp::ClearExclusive => self.exclusive = None,
            SysOp::DcZva { rt } => {
                // Eight doubleword stores rather than sixty-four byte ones.
                let addr = self.reg_at(rt) as u32 & !0x3F;
                for i in 0..8u32 {
                    self.mem.write_u64(addr.wrapping_add(i * 8), 0)?;
                }
            }
        }
        Ok(())
    }
}
