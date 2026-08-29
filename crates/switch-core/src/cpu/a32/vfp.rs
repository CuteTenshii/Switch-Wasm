//! VFP: the floating-point coprocessor a 32-bit title does its arithmetic in.
//!
//! Measured across Mario Kart 8 Deluxe's eight modules, the VFP data
//! processing, load/store and register-transfer encodings are 5% of the
//! binary — small next to the integer core, and load-bearing: `rtld` reaches
//! its first `vpush {d0-d7}` 8192 instructions in.

use crate::cpu::Cpu;
use crate::{Error, Result};

impl Cpu {
    /// VFP data processing and the register transfers that share its space.
    /// Stage 3; until then a guest that reaches one says so by name.
    pub(super) fn a32_vfp_data(&mut self, insn: u32) -> Result<()> {
        Err(Error::Cpu(format!(
            "unimplemented VFP instruction {:#010x} at pc={:#010x}",
            insn, self.pc
        )))
    }

    /// `VLDR`/`VSTR`/`VLDM`/`VSTM`, which live in the coprocessor load/store
    /// space. Stage 3.
    pub(super) fn a32_coproc_load_store(&mut self, insn: u32) -> Result<()> {
        Err(Error::Cpu(format!(
            "unimplemented VFP load/store {:#010x} at pc={:#010x}",
            insn, self.pc
        )))
    }
}

/// Name a VFP encoding for a trace. The arithmetic is Stage 3; naming what a
/// fault landed on is worth having before then, because "cop p11" says
/// nothing about which operation stopped a run.
pub(super) fn vfp_mnemonic(insn: u32, cond: &str) -> String {
    let single = (insn >> 8) & 0xF == 10;
    let width = if single { "s" } else { "d" };
    // The load/store space (bits 27:25 = 110) is the transfers; 111 is the
    // arithmetic and the register moves.
    if (insn >> 25) & 0x7 == 0b110 {
        let load = (insn >> 20) & 1 != 0;
        let base = (insn >> 16) & 0xF;
        let list = insn & 0xFF;
        return match ((insn >> 23) & 0b11, (insn >> 21) & 1, base) {
            // The pre-decrement and post-increment forms with SP as the base
            // are how a prologue and epilogue spell themselves.
            (0b10, 1, 13) if !load => format!("vpush{cond} {{{list} {width}-regs}}"),
            (0b01, 1, 13) if load => format!("vpop{cond} {{{list} {width}-regs}}"),
            (0b01 | 0b10, 1, _) => format!(
                "v{}m{cond} r{base}!, {{{list} {width}-regs}}",
                if load { "ld" } else { "st" }
            ),
            _ => format!(
                "v{}r{cond}.{width} d{}, [r{base}]",
                if load { "ld" } else { "st" },
                (insn >> 12) & 0xF
            ),
        };
    }
    if insn & 0x10 != 0 {
        let to_arm = (insn >> 20) & 1 != 0;
        return format!("v{}{cond} r{}", if to_arm { "mov (to arm)" } else { "mov" }, (insn >> 12) & 0xF);
    }
    let op = (insn >> 20) & 0xF;
    let name = match (op, (insn >> 6) & 1) {
        (0b0010 | 0b0011, _) => "vmul",
        (0b0000 | 0b0001, _) => "vmla",
        (0b0110 | 0b0111, 0) => "vadd",
        (0b0110 | 0b0111, 1) => "vsub",
        (0b1000 | 0b1001, _) => "vdiv",
        (0b1011, _) => "vmov/vcmp/vcvt",
        _ => "vfp",
    };
    format!("{name}{cond}.{width}")
}
