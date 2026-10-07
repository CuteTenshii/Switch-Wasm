//! MAXWELL_DMA_COPY_A (class 0xB0B5): copies rectangles between pitch and
//! block-linear surfaces, with a component remap that doubles as buffer fill.

use crate::gpu::engine::{field, Registers};
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::Layout;
use crate::{Error, Result};

/// The launch method, so a GPU backend can flush its surfaces before guest memory is read.
pub const LAUNCH_DMA: u32 = 0x0C0;
const OFFSET_IN: u32 = 0x100;
const OFFSET_OUT: u32 = 0x102;
const PITCH_IN: u32 = 0x104;
const PITCH_OUT: u32 = 0x105;
const LINE_LENGTH_IN: u32 = 0x106;
const LINE_COUNT: u32 = 0x107;
const SET_REMAP_CONST: u32 = 0x1C0;
const SET_REMAP_COMPONENTS: u32 = 0x1C2;
const SET_DST_BLOCK_SIZE: u32 = 0x1C3;
const SET_DST_WIDTH: u32 = 0x1C4;
const SET_DST_ORIGIN: u32 = 0x1C8;
const SET_SRC_BLOCK_SIZE: u32 = 0x1CA;
const SET_SRC_WIDTH: u32 = 0x1CB;
const SET_SRC_ORIGIN: u32 = 0x1CF;

const REMAP_SRC_X: u32 = 0;
const REMAP_CONST_0: u32 = 4;
const REMAP_CONST_1: u32 = 5;
const REMAP_NO_WRITE: u32 = 6;

#[derive(Debug, Default)]
pub struct EngineCopy {
    pub regs: Registers,
    /// Reused source run buffer.
    run: Vec<u8>,
    pub activity: crate::gpu::activity::GpuActivity,
}

impl EngineCopy {
    pub fn new() -> EngineCopy {
        EngineCopy {
            regs: Registers::new(),
            run: Vec::new(),
            activity: Default::default(),
        }
    }

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        if method == LAUNCH_DMA {
            self.launch(arg, ctx)?;
        }
        Ok(())
    }

    fn launch(&mut self, flags: u32, ctx: &mut ExecCtx) -> Result<()> {
        let src_pitch_layout = field(flags, 7, 7) != 0;
        let dst_pitch_layout = field(flags, 8, 8) != 0;
        let multi_line = field(flags, 9, 9) != 0;
        let remap = field(flags, 10, 10) != 0;

        let src_base = self.regs.iova(OFFSET_IN);
        let dst_base = self.regs.iova(OFFSET_OUT);
        let line_length = self.regs.get(LINE_LENGTH_IN);
        let line_count = if multi_line {
            self.regs.get(LINE_COUNT).max(1)
        } else {
            1
        };
        let layout = |pitch: bool| if pitch { "pitch" } else { "block" };
        let (src_cpu, dst_cpu) = (ctx.span(src_base, 1), ctx.span(dst_base, 1));
        self.activity.note(
            crate::gpu::activity::Kind::Copy,
            src_base,
            dst_base,
            u64::from(line_length) * u64::from(line_count),
            false,
            || {
                let at = |cpu: Option<u32>| match cpu {
                    Some(cpu) => format!("cpu {cpu:#x}"),
                    None => "unmapped".to_owned(),
                };
                format!(
                    "{src_base:#x} ({}, {}) -> {dst_base:#x} ({}, {}), \
                     {line_length:#x} x {line_count} lines, remap {remap}",
                    at(src_cpu),
                    layout(src_pitch_layout),
                    at(dst_cpu),
                    layout(dst_pitch_layout)
                )
            },
        );
        if ctx.trace || crate::trace::enabled(crate::trace::Trace::Copy) {
            crate::traceln!(
                "[gpu] dma copy {src_base:#x} ({}) -> {dst_base:#x} ({}) \
                 {line_length:#x} x {line_count} lines remap={remap} cpu src {:x?} dst {:x?}",
                layout(src_pitch_layout),
                layout(dst_pitch_layout),
                ctx.vmm.translate(src_base).map(|(c, _)| c),
                ctx.vmm.translate(dst_base).map(|(c, _)| c),
            );
        }

        if !multi_line && !remap {
            self.run.resize(line_length as usize, 0);
            ctx.read_run(src_base, &mut self.run)?;
            ctx.write_run(dst_base, &self.run)?;
            ctx.stats.copies += 1;
            return Ok(());
        }

        let (src_element, dst_element, components) = if remap {
            self.remap_shape()?
        } else {
            (1, 1, RemapComponents::identity())
        };

        let src = SurfaceWalk::new(
            &self.regs,
            src_pitch_layout,
            SET_SRC_BLOCK_SIZE,
            SET_SRC_WIDTH,
            SET_SRC_ORIGIN,
            PITCH_IN,
            src_element,
        )?;
        let dst = SurfaceWalk::new(
            &self.regs,
            dst_pitch_layout,
            SET_DST_BLOCK_SIZE,
            SET_DST_WIDTH,
            SET_DST_ORIGIN,
            PITCH_OUT,
            dst_element,
        )?;

        if !remap {
            self.copy_runs(
                ctx,
                &src.based_at(src_base),
                &dst.based_at(dst_base),
                line_length,
                line_count,
            )?;
            ctx.stats.copies += 1;
            return Ok(());
        }

        let consts = [
            self.regs.get(SET_REMAP_CONST),
            self.regs.get(SET_REMAP_CONST + 1),
        ];
        let component_size = components.component_size;

        for line in 0..line_count {
            for element in 0..line_length {
                let src_off = src.offset(element, line);
                let dst_off = dst.offset(element, line);
                let mut source = [0u32; 4];
                for (i, s) in source
                    .iter_mut()
                    .enumerate()
                    .take(components.num_src as usize)
                {
                    let mut v = 0u32;
                    for b in 0..component_size {
                        let addr = src_base + (src_off + i as u32 * component_size + b) as u64;
                        v |= (ctx.vmm_read_u8(addr)? as u32) << (8 * b);
                    }
                    *s = v;
                }
                for i in 0..components.num_dst {
                    let selector = components.dst[i as usize];
                    if selector == REMAP_NO_WRITE {
                        continue;
                    }
                    let value = match selector {
                        REMAP_SRC_X..=3 => source[selector as usize],
                        REMAP_CONST_0 => consts[0],
                        REMAP_CONST_1 => consts[1],
                        other => {
                            return Err(Error::Gpu(format!(
                                "copy: unknown remap selector {}",
                                other
                            )))
                        }
                    };
                    for b in 0..component_size {
                        let addr = dst_base + (dst_off + i * component_size + b) as u64;
                        ctx.vmm_write_u8(addr, (value >> (8 * b)) as u8)?;
                    }
                }
            }
        }
        ctx.stats.copies += 1;
        Ok(())
    }

    /// The remap-off copy (one-byte elements), a contiguous run at a time.
    fn copy_runs(
        &mut self,
        ctx: &mut ExecCtx,
        src: &SurfaceWalk,
        dst: &SurfaceWalk,
        line_length: u32,
        line_count: u32,
    ) -> Result<()> {
        for line in 0..line_count {
            let mut x = 0;
            while x < line_length {
                let (src_at, src_run) = src.run_at(x, line);
                // A pitch shorter than the line reports no run; still advance one byte.
                let take = src_run.min(line_length - x).max(1);
                self.run.resize(take as usize, 0);
                ctx.read_run(src_at, &mut self.run)?;

                let mut done = 0usize;
                while done < take as usize {
                    let (dst_at, dst_run) = dst.run_at(x + done as u32, line);
                    let n = (dst_run as usize).min(take as usize - done).max(1);
                    ctx.write_run(dst_at, &self.run[done..done + n])?;
                    done += n;
                }
                x += take;
            }
        }
        Ok(())
    }

    /// `(source element bytes, destination element bytes, layout)` from `SetRemapComponents`.
    fn remap_shape(&self) -> Result<(u32, u32, RemapComponents)> {
        let raw = self.regs.get(SET_REMAP_COMPONENTS);
        let component_size = field(raw, 16, 17) + 1;
        let num_src = field(raw, 20, 21) + 1;
        let num_dst = field(raw, 24, 25) + 1;
        let components = RemapComponents {
            dst: [
                field(raw, 0, 2),
                field(raw, 4, 6),
                field(raw, 8, 10),
                field(raw, 12, 14),
            ],
            component_size,
            num_src,
            num_dst,
        };
        Ok((
            num_src * component_size,
            num_dst * component_size,
            components,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct RemapComponents {
    dst: [u32; 4],
    component_size: u32,
    num_src: u32,
    num_dst: u32,
}

impl RemapComponents {
    fn identity() -> RemapComponents {
        RemapComponents {
            dst: [0, 1, 2, 3],
            component_size: 1,
            num_src: 1,
            num_dst: 1,
        }
    }
}

#[derive(Clone, Copy)]
struct SurfaceWalk {
    layout: Layout,
    base: u64,
    /// Row length in bytes: `Set{Src,Dst}Width` count elements, which the remap
    /// can make wider than a byte.
    width_bytes: u32,
    origin_x_bytes: u32,
    origin_y: u32,
    element_bytes: u32,
}

impl SurfaceWalk {
    fn new(
        regs: &Registers,
        pitch: bool,
        block_size_reg: u32,
        width_reg: u32,
        origin_reg: u32,
        pitch_reg: u32,
        element_bytes: u32,
    ) -> Result<SurfaceWalk> {
        if pitch {
            return Ok(SurfaceWalk {
                layout: Layout::Pitch {
                    pitch: regs.get(pitch_reg),
                },
                base: 0,
                width_bytes: regs.get(pitch_reg),
                origin_x_bytes: 0,
                origin_y: 0,
                element_bytes,
            });
        }
        let block = regs.get(block_size_reg);
        if field(block, 0, 3) != 0 {
            return Err(Error::Gpu(
                "copy: block-linear surfaces wider than one GOB are not a Maxwell layout".into(),
            ));
        }
        Ok(SurfaceWalk {
            layout: Layout::BlockLinear {
                block_height_gobs: 1 << field(block, 4, 7),
            },
            base: 0,
            width_bytes: regs.get(width_reg).saturating_mul(element_bytes),
            origin_x_bytes: regs.field(origin_reg, 0, 15).saturating_mul(element_bytes),
            origin_y: regs.field(origin_reg, 16, 31),
            element_bytes,
        })
    }

    fn offset(&self, element: u32, line: u32) -> u32 {
        let x = self.origin_x_bytes + element * self.element_bytes;
        self.layout
            .offset(x, self.origin_y + line, self.width_bytes)
    }

    fn based_at(&self, base: u64) -> SurfaceWalk {
        SurfaceWalk { base, ..*self }
    }

    /// GPU address of `(element, line)` and how many bytes from there are contiguous.
    fn run_at(&self, element: u32, line: u32) -> (u64, u32) {
        let x = self.origin_x_bytes + element * self.element_bytes;
        let (offset, run) = self
            .layout
            .run_at(x, self.origin_y + line, self.width_bytes);
        (self.base + u64::from(offset), run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    struct Harness {
        mem: Memory,
        vmm: AddressSpace,
        host1x: Host1x,
        stats: GpuStats,
        base: u64,
    }

    impl Harness {
        fn new() -> Harness {
            let mut mem = Memory::new();
            mem.map_zero(0x3000_0000, 0x8000).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm
                .map(0x3000_0000, 0x8000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
                .unwrap();
            Harness {
                mem,
                vmm,
                host1x: Host1x::new(),
                stats: GpuStats::default(),
                base,
            }
        }

        fn ctx(&mut self) -> ExecCtx<'_> {
            ExecCtx {
                mem: &mut self.mem,
                vmm: &self.vmm,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            }
        }
    }

    fn set_iova(engine: &mut EngineCopy, method: u32, va: u64) {
        engine.regs.set(method, (va >> 32) as u32);
        engine.regs.set(method + 1, va as u32);
    }

    #[test]
    fn flat_copy_moves_bytes() {
        let mut h = Harness::new();
        for i in 0..16u32 {
            h.mem.write_u8(0x3000_0000 + i, i as u8).unwrap();
        }
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_IN, h.base);
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x100);
        engine.regs.set(LINE_LENGTH_IN, 16);

        let mut ctx = h.ctx();
        engine.write(LAUNCH_DMA, 0, &mut ctx).unwrap();

        for i in 0..16u32 {
            assert_eq!(h.mem.read_u8(0x3000_0100 + i).unwrap(), i as u8);
        }
        assert_eq!(h.stats.copies, 1);
    }

    #[test]
    fn multiline_pitch_copy_respects_both_pitches() {
        let mut h = Harness::new();
        for y in 0..4u32 {
            for x in 0..8u32 {
                h.mem
                    .write_u8(0x3000_0000 + y * 16 + x, (y * 8 + x) as u8)
                    .unwrap();
            }
        }
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_IN, h.base);
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x400);
        engine.regs.set(PITCH_IN, 16);
        engine.regs.set(PITCH_OUT, 8);
        engine.regs.set(LINE_LENGTH_IN, 8);
        engine.regs.set(LINE_COUNT, 4);

        let mut ctx = h.ctx();
        engine
            .write(LAUNCH_DMA, (1 << 7) | (1 << 8) | (1 << 9), &mut ctx)
            .unwrap();

        for y in 0..4u32 {
            for x in 0..8u32 {
                assert_eq!(
                    h.mem.read_u8(0x3000_0400 + y * 8 + x).unwrap(),
                    (y * 8 + x) as u8
                );
            }
        }
    }

    #[test]
    fn remap_from_a_constant_fills_the_destination() {
        let mut h = Harness::new();
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_IN, h.base);
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x400);
        engine.regs.set(PITCH_OUT, 64);
        engine.regs.set(LINE_LENGTH_IN, 4);
        engine.regs.set(LINE_COUNT, 1);
        engine.regs.set(SET_REMAP_CONST, 0xDEAD_BEEF);
        // One 4-byte destination component from RemapConst[0].
        engine.regs.set(
            SET_REMAP_COMPONENTS,
            REMAP_CONST_0 | (3 << 16) | (0 << 20) | (0 << 24),
        );

        let mut ctx = h.ctx();
        engine
            .write(
                LAUNCH_DMA,
                (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10),
                &mut ctx,
            )
            .unwrap();

        for i in 0..4u32 {
            assert_eq!(h.mem.read_u32(0x3000_0400 + i * 4).unwrap(), 0xDEAD_BEEF);
        }
    }

    #[test]
    fn block_linear_destination_uses_the_swizzle() {
        let mut h = Harness::new();
        for i in 0..64u32 {
            h.mem.write_u8(0x3000_0000 + i, (i + 1) as u8).unwrap();
        }
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_IN, h.base);
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x1000);
        engine.regs.set(PITCH_IN, 64);
        engine.regs.set(SET_DST_BLOCK_SIZE, 0); // one GOB per block
        engine.regs.set(SET_DST_WIDTH, 64);
        engine.regs.set(LINE_LENGTH_IN, 64);
        engine.regs.set(LINE_COUNT, 1);

        let mut ctx = h.ctx();
        engine
            .write(LAUNCH_DMA, (1 << 7) | (1 << 9), &mut ctx)
            .unwrap();

        // Row 0 of a GOB lands at 0..16, 32..48, 256..272, 288..304.
        assert_eq!(h.mem.read_u8(0x3000_1000).unwrap(), 1);
        assert_eq!(h.mem.read_u8(0x3000_1000 + 32).unwrap(), 17);
        assert_eq!(h.mem.read_u8(0x3000_1000 + 256).unwrap(), 33);
        assert_eq!(h.mem.read_u8(0x3000_1000 + 288).unwrap(), 49);
    }
}
