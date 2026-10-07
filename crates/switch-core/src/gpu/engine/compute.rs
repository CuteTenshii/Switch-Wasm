//! MAXWELL_COMPUTE_B (class 0xB1C0). Launches come from the QMD in memory;
//! this holds the program region and texture pools, at the 3D class's methods.
//! Method numbers are from NVIDIA's `clb1c0.h`.

use crate::gpu::engine::Registers;
use crate::gpu::exec::ExecCtx;
use crate::Result;

const SEND_PCAS_A: u32 = 0x0AD;

/// The launch trigger; the channel flushes the 3D backend before it.
pub const SEND_SIGNALING_PCAS_B: u32 = 0x0AF;

const SET_TEX_SAMPLER_POOL: u32 = 0x557;
const SET_TEX_HEADER_POOL: u32 = 0x55D;
const SET_PROGRAM_REGION: u32 = 0x582;
const SET_BINDLESS_TEXTURE: u32 = 0x982;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dispatch {
    /// GPU address of the QMD, already un-shifted.
    pub qmd_addr: u64,
}

#[derive(Debug, Default)]
pub struct EngineCompute {
    pub regs: Registers,
    pub last_dispatch: Option<Dispatch>,
    pub dispatches: u64,
    /// Inline upload at 0x60..0x6D, which drivers use to write QMDs before launch.
    pub inline: crate::gpu::engine::inline::EngineInline,
    /// Dispatches refused, by reason.
    pub activity: crate::gpu::activity::GpuActivity,
}

impl EngineCompute {
    pub fn new() -> EngineCompute {
        EngineCompute {
            regs: Registers::new(),
            last_dispatch: None,
            dispatches: 0,
            inline: crate::gpu::engine::inline::EngineInline::new(),
            activity: Default::default(),
        }
    }

    pub fn program_region(&self) -> u64 {
        self.regs.iova(SET_PROGRAM_REGION)
    }

    pub fn tex_header_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_HEADER_POOL)
    }

    pub fn tex_sampler_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_SAMPLER_POOL)
    }

    /// `SetBindlessTexture`: the constant bank a `texs` handle indexes.
    pub fn tex_cb_index(&self) -> u8 {
        self.regs.field(SET_BINDLESS_TEXTURE, 0, 4) as u8
    }

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        if crate::gpu::engine::inline::METHOD_RANGE.contains(&method) {
            return self.inline.write(method, arg, ctx);
        }
        if method == SEND_SIGNALING_PCAS_B {
            let qmd_addr = (self.regs.get(SEND_PCAS_A) as u64) << 8;
            self.last_dispatch = Some(Dispatch { qmd_addr });
            self.dispatches += 1;
            ctx.stats.dispatches += 1;
            if ctx.trace {
                crate::traceln!("[gpu] compute dispatch qmd={:#x}", qmd_addr);
            }
            self.dispatch_or_log(ctx);
        }
        Ok(())
    }

    /// Run the launch, or count the refusal instead of failing the pushbuffer.
    fn dispatch_or_log(&mut self, ctx: &mut ExecCtx) {
        if let Err(e) = crate::gpu::compute::dispatch(self, ctx) {
            ctx.stats.dispatches_skipped += 1;
            if ctx.trace {
                crate::traceln!("[gpu] compute: {e}");
            }
            self.activity
                .refuse(crate::gpu::activity::Kind::Dispatch, e.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    #[test]
    fn an_inline_upload_on_the_compute_class_reaches_memory() {
        use crate::gpu::engine::inline::{
            LAUNCH_DMA, LINE_COUNT, LINE_LENGTH_IN, LOAD_INLINE_DATA, OFFSET_OUT, PITCH_OUT,
        };
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm
            .map(0x3000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
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
        for (method, arg) in [
            (OFFSET_OUT, (base >> 32) as u32),
            (OFFSET_OUT + 1, base as u32),
            (LINE_LENGTH_IN, 8),
            (LINE_COUNT, 1),
            (PITCH_OUT, 8),
            (LAUNCH_DMA, 1),
            (LOAD_INLINE_DATA, 0x1122_3344),
            (LOAD_INLINE_DATA, 0x5566_7788),
        ] {
            engine.write(method, arg, &mut ctx).unwrap();
        }
        assert_eq!(mem.read_u32(0x3000_0000).unwrap(), 0x1122_3344);
        assert_eq!(mem.read_u32(0x3000_0004).unwrap(), 0x5566_7788);
    }

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
        assert_eq!(
            engine.last_dispatch,
            Some(Dispatch {
                qmd_addr: 0x1234_5600
            })
        );
        assert_eq!(engine.dispatches, 1);
        // The QMD is unmapped, so the launch is refused and counted.
        assert_eq!(stats.dispatches_skipped, 1);
    }
}
