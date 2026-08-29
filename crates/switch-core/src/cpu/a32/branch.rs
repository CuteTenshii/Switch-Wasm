//! A32 control flow and the coprocessor space: the immediate branches, the
//! `cond == 0xF` encodings that are unconditional by construction, and CP15 —
//! which at EL0 is only the thread-pointer pair.

use crate::cpu::Cpu;
use crate::{Error, Result};

impl Cpu {
    /// `B` and `BL`.
    pub(super) fn a32_branch(&mut self, insn: u32) -> Result<()> {
        let imm = ((insn & 0x00FF_FFFF) << 8) as i32 >> 6;
        let target = self.pc.wrapping_add(8).wrapping_add(imm as u32);
        if (insn >> 24) & 1 != 0 {
            self.regs[14] = u64::from(self.pc.wrapping_add(4));
        }
        self.pc = target;
        Ok(())
    }

    /// The `cond == 0xF` encoding space, which is unconditional by
    /// construction: the memory barriers and preload hints a guest issues, and
    /// the one branch that would switch to Thumb.
    pub(super) fn a32_unconditional(&mut self, insn: u32) -> Result<()> {
        // DSB/DMB/ISB, CLREX, and the PLD/PLI preload hints: architectural
        // no-ops here, since there is one core and no cache to maintain.
        let barrier = (insn & 0xFFFF_FFF0) == 0xF57F_F040
            || (insn & 0xFFFF_FFF0) == 0xF57F_F050
            || (insn & 0xFFFF_FFF0) == 0xF57F_F060;
        let clrex = insn == 0xF57F_F01F;
        let preload = (insn & 0xFD30_F000) == 0xF510_F000;
        if barrier || preload {
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }
        if clrex {
            self.exclusive = None;
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }
        if (insn & 0xFE00_0000) == 0xFA00_0000 {
            let h = (insn >> 24) & 1;
            let imm = ((insn & 0x00FF_FFFF) << 8) as i32 >> 6;
            let target = self
                .pc
                .wrapping_add(8)
                .wrapping_add(imm as u32)
                .wrapping_add(h << 1);
            self.regs[14] = u64::from(self.pc.wrapping_add(4));
            return Err(Error::Cpu(format!(
                "BLX to Thumb code at {:#010x} from pc={:#010x}; T32 is not implemented",
                target, self.pc
            )));
        }
        Err(Error::Cpu(format!(
            "unimplemented unconditional A32 instruction {:#010x} at pc={:#010x}",
            insn, self.pc
        )))
    }

    /// `MRC`/`MCR` and the rest of the coprocessor space. The only coprocessor
    /// a Horizon guest reaches at EL0 is CP15's thread-pointer pair, which is
    /// AArch32's spelling of `TPIDRRO_EL0` and `TPIDR_EL0`; the emulator
    /// already keeps those apart, and aliasing them breaks IPC exactly as it
    /// does in A64.
    pub(super) fn a32_coproc(&mut self, insn: u32) -> Result<()> {
        let coproc = (insn >> 8) & 0xF;
        if coproc == 10 || coproc == 11 {
            return self.a32_vfp_data(insn);
        }
        let is_mrc = (insn >> 20) & 1 != 0;
        let opc1 = (insn >> 21) & 0x7;
        let crn = (insn >> 16) & 0xF;
        let crm = insn & 0xF;
        let opc2 = (insn >> 5) & 0x7;
        let rt = ((insn >> 12) & 0xF) as u8;
        if coproc == 15 && opc1 == 0 && crn == 13 && crm == 0 {
            match (is_mrc, opc2) {
                // TPIDRURW, the guest's own per-thread pointer.
                (true, 2) => {
                    let val = self.tpidr_rw as u32;
                    self.set_r32(rt, val);
                }
                (false, 2) => self.tpidr_rw = u64::from(self.r32(rt)),
                // TPIDRURO: the kernel-set, read-only thread pointer the IPC
                // message buffer hangs off.
                (true, 3) => {
                    let val = self.tpidr as u32;
                    self.set_r32(rt, val);
                }
                _ => {
                    return Err(Error::Cpu(format!(
                        "unimplemented CP15 c13 access {:#010x} at pc={:#010x}",
                        insn, self.pc
                    )))
                }
            }
            self.pc = self.pc.wrapping_add(4);
            return Ok(());
        }
        Err(Error::Cpu(format!(
            "unimplemented coprocessor access {:#010x} (p{coproc}) at pc={:#010x}",
            insn, self.pc
        )))
    }
}
