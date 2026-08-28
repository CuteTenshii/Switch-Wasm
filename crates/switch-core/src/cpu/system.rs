//! System instructions: MRS/MSR, barriers, hints and the cache maintenance
//! operations (`DC ZVA`) that memset implementations rely on.

use super::bits::{FPCR_MASK, FPSR_MASK};
use super::Cpu;
use crate::{Error, Result};

impl Cpu {
    pub(super) fn system(&mut self, insn: u32, next_pc: u32) -> Result<()> {
        let l = (insn >> 21) & 1;
        let op0 = (insn >> 19) & 0b11;
        let op1 = (insn >> 16) & 0b111;
        let crn = (insn >> 12) & 0xF;
        let crm = (insn >> 8) & 0xF;
        let op2 = (insn >> 5) & 0b111;
        let rt = (insn & 0x1F) as u8;

        if (insn >> 16) & 0xFFFF == 0xD503 {
            // HINT (incl. NOP), barriers, MSR immediate etc all no-op here.
            self.pc = next_pc;
            return Ok(());
        }

        if l == 1 {
            // MRS Xt, <sysreg>
            let sysreg = (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
            let val = match sysreg {
                // NZCV: 3:3:4:2:0
                0b11_011_0100_0010_000 => self.nzcv as u64,
                // TPIDR_EL0 (3:3:13:0:2): freely writable by guest code.
                0b11_011_1101_0000_010 => self.tpidr_rw,
                // TPIDRRO_EL0 (3:3:13:0:3): the kernel-fixed TLS base —
                // read-only at EL0 on real hardware, so there is no
                // corresponding MSR case below.
                0b11_011_1101_0000_011 => self.tpidr,
                // FPCR (3:3:4:4:0) and FPSR (3:3:4:4:1). The op1 field is 3
                // at EL0, not 0 — reading it as 3:0:... meant a guest's
                // `mrs x0, fpcr` fell through to the catch-all zero.
                0b11_011_0100_0100_000 => u64::from(self.fpcr),
                0b11_011_0100_0100_001 => u64::from(self.fpsr),
                // DCZID_EL0: 3:3:0:0:7 — report the Cortex-A57 DC ZVA block
                // size (BS=4 → 64 bytes). musl/newlib memset strides the
                // cache-zero loop with `4 << BS`; BS=0 makes it run away.
                0b11_011_0000_0000_111 => 4,
                // CTR_EL0: 3:3:0:0:1 — the Cortex-A57 value (0x8444C004:
                // 64-byte I- and D-cache lines, 64-byte ERG/CWG). Cache-flush
                // loops stride by `4 << DminLine`, so reporting 0 made
                // NX-Shell's flush walk its buffers 4 bytes at a time.
                0b11_011_0000_0000_001 => 0x8444_C004,
                _ => 0,
            };
            self.write_zr(rt, val);
            self.pc = next_pc;
            return Ok(());
        }

        if op0 == 0 {
            // MSR (immediate) or MSR (register) to PSTATE/special
            match (op1, crn, crm, op2) {
                // MSR NZCV, #imm (best-effort; also DAIFSET/CLEAR as no-op)
                (0b010, 0b0100, 0b0010, 0b000) | (0b011, 0b0100, 0b0010, 0b000) => {
                    let imm = (insn >> 8) & 0xF;
                    self.nzcv = imm;
                }
                _ => {
                    // All other MSR immediate forms (DAIF, SPSel, ...) are no-ops.
                }
            }
            self.pc = next_pc;
            return Ok(());
        }

        if l == 0 && op0 == 1 && op1 == 3 && crn == 7 && crm == 4 && op2 == 1 {
            // DC ZVA Xt: zero the 64-byte block at Xt (A57 dczid BS=4).
            // musl/newlib memset uses this to clear aligned blocks, so it is
            // hot — eight doubleword stores rather than sixty-four byte ones,
            // which is eight page lookups instead of sixty-four. (Not
            // `fill_le`: it stamps a 512-byte pattern before copying, which a
            // block this small never amortizes.)
            let addr = self.read_zr(rt) as u32 & !0x3F;
            for i in 0..8u32 {
                self.mem.write_u64(addr.wrapping_add(i * 8), 0)?;
            }
            self.pc = next_pc;
            return Ok(());
        }

        if l == 0 && op0 == 1 && crn == 7 {
            // The other cache-maintenance operations (`DC IVAC/CVAC/CVAU/
            // CIVAC`, `IC IALLU/IVAU`). libnx flushes the data cache around
            // every buffer it hands to the GPU; there are no caches to
            // maintain here — memory is always coherent — so these retire
            // with no effect.
            self.pc = next_pc;
            return Ok(());
        }

        if l == 0 && (op0 == 2 || op0 == 3) {
            // MSR (register): write sysreg from Xt. We observe NZCV and
            // TPIDR_EL0 — not TPIDRRO_EL0, which real hardware makes
            // read-only at EL0, so a guest write to it is simply ignored
            // (matches the `_ => {}` fallthrough).
            let sysreg = (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
            match sysreg {
                0b11_011_0100_0010_000 => self.nzcv = self.read_zr(rt) as u32,
                // Only the bits the architecture defines stick, so a guest
                // that reads back what it wrote sees the same value.
                0b11_011_0100_0100_000 => self.fpcr = self.read_zr(rt) as u32 & FPCR_MASK,
                0b11_011_0100_0100_001 => self.fpsr = self.read_zr(rt) as u32 & FPSR_MASK,
                0b11_011_1101_0000_010 => self.tpidr_rw = self.read_zr(rt),
                _ => {}
            }
            self.pc = next_pc;
            return Ok(());
        }

        Err(Error::Cpu(format!(
            "unimplemented system instruction 0x{:08x} at {:#x}",
            insn, self.pc
        )))
    }

    // ---------- loads & stores ----------
}
