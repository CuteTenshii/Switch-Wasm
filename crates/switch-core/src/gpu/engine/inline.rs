//! KEPLER_INLINE_TO_MEMORY_B (class 0xA140).
//!
//! Uploads data that travels *inside* the pushbuffer straight into memory —
//! deko3d uses it for small buffer updates where a DMA round-trip would cost
//! more than the words themselves. The 3D class implements the same methods,
//! and deko3d sends them on the 3D subchannel, so the channel routes both.

use crate::gpu::engine::{field, Registers};
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::Layout;
use crate::{Error, Result};

pub const LINE_LENGTH_IN: u32 = 0x60;
pub const LINE_COUNT: u32 = 0x61;
pub const OFFSET_OUT: u32 = 0x62;
pub const PITCH_OUT: u32 = 0x64;
pub const SET_DST_BLOCK_SIZE: u32 = 0x65;
pub const SET_DST_WIDTH: u32 = 0x66;
pub const SET_DST_HEIGHT: u32 = 0x67;
pub const SET_ORIGIN_BYTES_X: u32 = 0x6A;
pub const SET_ORIGIN_SAMPLES_Y: u32 = 0x6B;
pub const LAUNCH_DMA: u32 = 0x6C;
pub const LOAD_INLINE_DATA: u32 = 0x6D;

/// The method range this class owns.
pub const METHOD_RANGE: std::ops::RangeInclusive<u32> = LINE_LENGTH_IN..=LOAD_INLINE_DATA;

#[derive(Debug, Default)]
pub struct EngineInline {
    pub regs: Registers,
    /// Bytes written since the last `LaunchDma`.
    written: u32,
}

impl EngineInline {
    pub fn new() -> EngineInline {
        EngineInline { regs: Registers::new(), written: 0 }
    }

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        match method {
            LAUNCH_DMA => {
                self.written = 0;
                if ctx.trace {
                    eprintln!(
                        "[gpu] inline launch dst={:#x} line_len={} lines={} pitch={} flags={arg:#x}",
                        self.regs.iova(OFFSET_OUT), self.regs.get(LINE_LENGTH_IN),
                        self.regs.get(LINE_COUNT), self.regs.get(PITCH_OUT)
                    );
                }
            }
            LOAD_INLINE_DATA => self.load_inline_data(arg, ctx)?,
            _ => {}
        }
        Ok(())
    }

    fn load_inline_data(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        let pitch_layout = self.regs.bit(LAUNCH_DMA, 0);
        let line_length = self.regs.get(LINE_LENGTH_IN);
        if line_length == 0 {
            return Err(Error::Gpu(
                "inline: LoadInlineData with a zero-length line".into(),
            ));
        }
        let base = self.regs.iova(OFFSET_OUT);
        let (layout, width_bytes) = if pitch_layout {
            (Layout::Pitch { pitch: self.regs.get(PITCH_OUT) }, self.regs.get(PITCH_OUT))
        } else {
            let block = self.regs.get(SET_DST_BLOCK_SIZE);
            (
                Layout::BlockLinear { block_height_gobs: 1 << field(block, 4, 7) },
                self.regs.get(SET_DST_WIDTH),
            )
        };
        let origin_x = self.regs.get(SET_ORIGIN_BYTES_X);
        let origin_y = self.regs.get(SET_ORIGIN_SAMPLES_Y);

        for byte in 0..4u32 {
            let position = self.written + byte;
            let line = position / line_length;
            let column = position % line_length;
            let offset = layout.offset(origin_x + column, origin_y + line, width_bytes);
            ctx.vmm_write_u8(base + offset as u64, (arg >> (8 * byte)) as u8)?;
        }
        self.written += 4;
        Ok(())
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
    fn inline_upload_writes_consecutive_words() {
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm.map(0x3000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();

        let mut engine = EngineInline::new();
        engine.regs.set(OFFSET_OUT, (base >> 32) as u32);
        engine.regs.set(OFFSET_OUT + 1, base as u32);
        engine.regs.set(LINE_LENGTH_IN, 64);
        engine.regs.set(LINE_COUNT, 1);
        engine.regs.set(PITCH_OUT, 64);

        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        engine.write(LAUNCH_DMA, 1, &mut ctx).unwrap(); // pitch destination
        engine.write(LOAD_INLINE_DATA, 0x1122_3344, &mut ctx).unwrap();
        engine.write(LOAD_INLINE_DATA, 0x5566_7788, &mut ctx).unwrap();

        assert_eq!(mem.read_u32(0x3000_0000).unwrap(), 0x1122_3344);
        assert_eq!(mem.read_u32(0x3000_0004).unwrap(), 0x5566_7788);
    }

    #[test]
    fn inline_upload_wraps_to_the_next_line() {
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm.map(0x3000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();

        let mut engine = EngineInline::new();
        engine.regs.set(OFFSET_OUT, (base >> 32) as u32);
        engine.regs.set(OFFSET_OUT + 1, base as u32);
        engine.regs.set(LINE_LENGTH_IN, 4);
        engine.regs.set(LINE_COUNT, 2);
        engine.regs.set(PITCH_OUT, 32);

        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        engine.write(LAUNCH_DMA, 1, &mut ctx).unwrap();
        engine.write(LOAD_INLINE_DATA, 0xAABB_CCDD, &mut ctx).unwrap();
        engine.write(LOAD_INLINE_DATA, 0x1122_3344, &mut ctx).unwrap();

        assert_eq!(mem.read_u32(0x3000_0000).unwrap(), 0xAABB_CCDD);
        assert_eq!(mem.read_u32(0x3000_0020).unwrap(), 0x1122_3344);
    }
}
