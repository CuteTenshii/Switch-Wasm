//! MAXWELL_COMPUTE_B (class 0xB1C0).
//!
//! Almost none of a launch is in this register file. Writing the QMD's address
//! (shifted right by 8) to `SendPcasA` and then `SendSignalingPcasB` starts it,
//! and everything about the grid comes out of the [`crate::gpu::qmd`] in
//! memory. What is here is the state a QMD refers to rather than carries: the
//! program region its offset is relative to, and the descriptor pools its
//! textures are drawn from.
//!
//! Method numbers are from NVIDIA's generated `clb1c0.h`. The pools and the
//! program region sit at the same methods as the 3D class's, which is why
//! [`crate::gpu::texture`] serves both unchanged.

use crate::gpu::engine::Registers;
use crate::gpu::exec::ExecCtx;
use crate::Result;

const SEND_PCAS_A: u32 = 0x0AD;

/// The launch trigger. Public because the channel flushes the 3D backend
/// before one: a dispatch reads and writes guest memory that a GPU-resident
/// render target may still be holding.
pub const SEND_SIGNALING_PCAS_B: u32 = 0x0AF;

const SET_TEX_SAMPLER_POOL: u32 = 0x557;
const SET_TEX_HEADER_POOL: u32 = 0x55D;
const SET_PROGRAM_REGION: u32 = 0x582;
const SET_BINDLESS_TEXTURE: u32 = 0x982;

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

    /// Base a QMD's `program_offset` is measured from.
    pub fn program_region(&self) -> u64 {
        self.regs.iova(SET_PROGRAM_REGION)
    }

    pub fn tex_header_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_HEADER_POOL)
    }

    pub fn tex_sampler_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_SAMPLER_POOL)
    }

    /// Which constant bank a `texs`'s immediate indexes for its handle —
    /// `SetBindlessTexture`, the compute class's `TexCbIndex`.
    pub fn tex_cb_index(&self) -> u8 {
        self.regs.field(SET_BINDLESS_TEXTURE, 0, 4) as u8
    }

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        if method == SEND_SIGNALING_PCAS_B {
            let qmd_addr = (self.regs.get(SEND_PCAS_A) as u64) << 8;
            self.last_dispatch = Some(Dispatch { qmd_addr });
            self.dispatches += 1;
            ctx.stats.dispatches += 1;
            if ctx.trace {
                eprintln!("[gpu] compute dispatch qmd={:#x}", qmd_addr);
            }
            self.dispatch_or_log(ctx);
        }
        Ok(())
    }

    /// Run the launch, or report why it did not run.
    ///
    /// A refused dispatch is counted rather than propagated, exactly as a
    /// refused draw is: one kernel the interpreter cannot follow should cost
    /// that kernel, not the pushbuffer it arrived in.
    fn dispatch_or_log(&mut self, ctx: &mut ExecCtx) {
        if let Err(e) = crate::gpu::compute::dispatch(self, ctx) {
            ctx.stats.dispatches_skipped += 1;
            if ctx.trace {
                eprintln!("[gpu] compute: {e}");
            }
        }
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
        // The QMD is at an address nothing has mapped, so the launch is
        // refused — and counted, rather than taking the pushbuffer with it.
        assert_eq!(stats.dispatches_skipped, 1);
    }

    #[test]
    fn the_program_region_and_pools_read_back_as_written() {
        let mut engine = EngineCompute::new();
        engine.regs.set(SET_PROGRAM_REGION, 0x11);
        engine.regs.set(SET_PROGRAM_REGION + 1, 0x2233_4455);
        engine.regs.set(SET_TEX_HEADER_POOL, 0);
        engine.regs.set(SET_TEX_HEADER_POOL + 1, 0x8000);
        engine.regs.set(SET_BINDLESS_TEXTURE, 2);
        assert_eq!(engine.program_region(), 0x11_2233_4455);
        assert_eq!(engine.tex_header_pool(), 0x8000);
        assert_eq!(engine.tex_cb_index(), 2);
    }
}
