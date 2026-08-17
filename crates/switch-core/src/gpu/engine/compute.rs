//! MAXWELL_COMPUTE_B (class 0xB1C0).
//!
//! A compute dispatch is described by a QMD (Queue Meta Data) structure in
//! memory; the channel kicks it off by writing the QMD's address (shifted
//! right by 8) to `SendPcasA` and then `SendSignalingPcasB`. Running the
//! dispatch needs the shader core, so for now the engine records the request —
//! the register file and QMD address are real, the warp execution is not.

use crate::gpu::engine::Registers;
use crate::gpu::exec::ExecCtx;
use crate::Result;

const SEND_PCAS_A: u32 = 0x0AD;
const SEND_SIGNALING_PCAS_B: u32 = 0x0AF;

/// A dispatch the engine was asked to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dispatch {
    /// GPU address of the QMD (already un-shifted).
    pub qmd_addr: u64,
}

#[derive(Debug, Default)]
pub struct EngineCompute {
    pub regs: Registers,
    pub last_dispatch: Option<Dispatch>,
    pub dispatches: u64,
}

impl EngineCompute {
    pub fn new() -> EngineCompute {
        EngineCompute { regs: Registers::new(), last_dispatch: None, dispatches: 0 }
    }

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        if method == SEND_SIGNALING_PCAS_B {
            let qmd_addr = (self.regs.get(SEND_PCAS_A) as u64) << 8;
            self.last_dispatch = Some(Dispatch { qmd_addr });
            self.dispatches += 1;
            if ctx.trace {
                eprintln!("[gpu] compute dispatch qmd={:#x}", qmd_addr);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::AddressSpace;
    use crate::mem::Memory;

    #[test]
    fn dispatch_unshifts_the_qmd_address() {
        let mut mem = Memory::new();
        let vmm = AddressSpace::new();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();
        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };

        let mut engine = EngineCompute::new();
        engine.write(SEND_PCAS_A, 0x0012_3456, &mut ctx).unwrap();
        engine.write(SEND_SIGNALING_PCAS_B, 0, &mut ctx).unwrap();
        assert_eq!(engine.last_dispatch, Some(Dispatch { qmd_addr: 0x1234_5600 }));
        assert_eq!(engine.dispatches, 1);
    }
}
