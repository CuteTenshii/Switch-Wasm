//! MAXWELL_DMA_COPY_A (class 0xB0B5): copies rectangles between pitch and
//! block-linear surfaces, with a component remap that doubles as buffer fill.

use crate::gpu::engine::{field, Registers};
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::Layout;
use crate::{Error, Result};

/// The launch method, so a GPU backend can flush its surfaces before guest memory is read.
pub const LAUNCH_DMA: u32 = 0x0C0;
const SET_SEMAPHORE_A: u32 = 0x090;
const SET_SEMAPHORE_PAYLOAD: u32 = 0x092;
pub(crate) const OFFSET_IN: u32 = 0x100;
const OFFSET_OUT: u32 = 0x102;
const PITCH_IN: u32 = 0x104;
const PITCH_OUT: u32 = 0x105;
pub(crate) const LINE_LENGTH_IN: u32 = 0x106;
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

/// `LaunchDma`'s `DATA_TRANSFER_TYPE`: a launch that only releases a semaphore.
const DATA_TRANSFER_NONE: u32 = 0;
/// `LaunchDma`'s `SEMAPHORE_TYPE`.
const RELEASE_ONE_WORD_SEMAPHORE: u32 = 1;
const RELEASE_FOUR_WORD_SEMAPHORE: u32 = 2;

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
            self.launch(arg, ctx).map_err(|e| match e {
                Error::Gpu(why) => Error::Gpu(format!("{}: {why}", self.describe(arg))),
                other => other,
            })?;
        }
        Ok(())
    }

    /// The launch and the registers it reads, for an error that names them.
    fn describe(&self, flags: u32) -> String {
        let r = &self.regs;
        format!(
            "dma copy {flags:#x} from {:#x} (pitch {:#x}, origin {:#x}, width {:#x}, block {:#x}) \
             to {:#x} (pitch {:#x}, origin {:#x}, width {:#x}, block {:#x}), \
             {:#x} bytes x {} lines",
            r.iova(OFFSET_IN),
            r.get(PITCH_IN),
            r.get(SET_SRC_ORIGIN),
            r.get(SET_SRC_WIDTH),
            r.get(SET_SRC_BLOCK_SIZE),
            r.iova(OFFSET_OUT),
            r.get(PITCH_OUT),
            r.get(SET_DST_ORIGIN),
            r.get(SET_DST_WIDTH),
            r.get(SET_DST_BLOCK_SIZE),
            r.get(LINE_LENGTH_IN),
            r.get(LINE_COUNT),
        )
    }

    /// The source and destination a launch with `flags` touches, when it is one plain run
    /// of bytes; other shapes do not say so without walking them.
    pub fn plain_ranges(&self, flags: u32) -> Option<[std::ops::Range<u64>; 2]> {
        let (multi_line, remap) = (field(flags, 9, 9) != 0, field(flags, 10, 10) != 0);
        if multi_line || remap || field(flags, 0, 1) == DATA_TRANSFER_NONE {
            return None;
        }
        let len = u64::from(self.regs.get(LINE_LENGTH_IN));
        let (src, dst) = (self.regs.iova(OFFSET_IN), self.regs.iova(OFFSET_OUT));
        Some([src..src + len, dst..dst + len])
    }

    /// The copy `flags` asks for, if any, then its semaphore release.
    fn launch(&mut self, flags: u32, ctx: &mut ExecCtx) -> Result<()> {
        if field(flags, 0, 1) != DATA_TRANSFER_NONE {
            self.copy(flags, ctx)?;
        }
        let addr = self.regs.iova(SET_SEMAPHORE_A);
        let payload = self.regs.get(SET_SEMAPHORE_PAYLOAD);
        match field(flags, 3, 4) {
            RELEASE_ONE_WORD_SEMAPHORE => ctx.write_u32(addr, payload),
            RELEASE_FOUR_WORD_SEMAPHORE => {
                ctx.write_u64(addr, u64::from(payload))?;
                ctx.write_u64(addr + 8, ctx.stats.submissions)
            }
            _ => Ok(()),
        }
    }

    fn copy(&mut self, flags: u32, ctx: &mut ExecCtx) -> Result<()> {
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
        let dst_cpu = ctx.span(dst_base, 1);
        self.activity.note(
            crate::gpu::activity::Kind::Copy,
            0,
            dst_base,
            u64::from(line_length) * u64::from(line_count),
            false,
            || match dst_cpu {
                Some(cpu) => format!(
                    "into {dst_base:#x} (cpu {cpu:#x}, {})",
                    layout(dst_pitch_layout)
                ),
                None => format!(
                    "into {dst_base:#x} (unmapped, {})",
                    layout(dst_pitch_layout)
                ),
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
        // A fill from the constants has no source, and its address may be unmapped.
        let reads_source = components.dst[..components.num_dst as usize]
            .iter()
            .any(|&selector| selector < REMAP_CONST_0);
        let num_src = if reads_source { components.num_src } else { 0 };

        for line in 0..line_count {
            for element in 0..line_length {
                let src_off = src.offset(element, line);
                let dst_off = dst.offset(element, line);
                let mut source = [0u32; 4];
                for (i, s) in source.iter_mut().enumerate().take(num_src as usize) {
                    let mut v = 0u32;
                    for b in 0..component_size {
                        let addr = at(src_base, src_off + i64::from(i as u32 * component_size + b));
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
                        let addr = at(dst_base, dst_off + i64::from(i * component_size + b));
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

/// `base` moved by a signed `offset`.
fn at(base: u64, offset: i64) -> u64 {
    base.wrapping_add_signed(offset)
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
            let pitch = regs.get(pitch_reg);
            return Ok(SurfaceWalk {
                layout: Layout::Pitch { pitch },
                base: 0,
                width_bytes: (pitch as i32).unsigned_abs(),
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

    /// Bytes from the surface's base to `(element, line)`; negative for a pitch layout whose
    /// pitch is, writing the rows bottom to top.
    fn offset(&self, element: u32, line: u32) -> i64 {
        let x = self.origin_x_bytes + element * self.element_bytes;
        let y = self.origin_y + line;
        match self.layout {
            Layout::Pitch { pitch } => i64::from(y) * i64::from(pitch as i32) + i64::from(x),
            layout => i64::from(layout.offset(x, y, self.width_bytes)),
        }
    }

    fn based_at(&self, base: u64) -> SurfaceWalk {
        SurfaceWalk { base, ..*self }
    }

    /// GPU address of `(element, line)` and how many bytes from there are contiguous.
    fn run_at(&self, element: u32, line: u32) -> (u64, u32) {
        let x = self.origin_x_bytes + element * self.element_bytes;
        let run = match self.layout {
            Layout::Pitch { .. } => self.width_bytes.saturating_sub(x),
            layout => layout.run_at(x, self.origin_y + line, self.width_bytes).1,
        };
        (at(self.base, self.offset(element, line)), run)
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
            let mem = Memory::new();
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
                mem: &self.mem,
                vmm: &self.vmm,
                as_id: 0,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            }
        }
    }

    /// `LaunchDma`'s `DATA_TRANSFER_TYPE_NON_PIPELINED`.
    const NON_PIPELINED: u32 = 2;

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
        engine.write(LAUNCH_DMA, NON_PIPELINED, &mut ctx).unwrap();

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
            .write(
                LAUNCH_DMA,
                NON_PIPELINED | (1 << 7) | (1 << 8) | (1 << 9),
                &mut ctx,
            )
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
    fn a_negative_destination_pitch_writes_the_rows_bottom_to_top() {
        let mut h = Harness::new();
        for i in 0..32u32 {
            h.mem.write_u8(0x3000_0000 + i, i as u8).unwrap();
        }
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_IN, h.base);
        // The first row goes to the last of four 8-byte rows.
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x400 + 3 * 8);
        engine.regs.set(PITCH_IN, 8);
        engine.regs.set(PITCH_OUT, (-8i32) as u32);
        engine.regs.set(LINE_LENGTH_IN, 8);
        engine.regs.set(LINE_COUNT, 4);

        let mut ctx = h.ctx();
        engine
            .write(
                LAUNCH_DMA,
                NON_PIPELINED | (1 << 7) | (1 << 8) | (1 << 9),
                &mut ctx,
            )
            .unwrap();

        for y in 0..4u32 {
            for x in 0..8u32 {
                assert_eq!(
                    h.mem.read_u8(0x3000_0400 + (3 - y) * 8 + x).unwrap(),
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
                NON_PIPELINED | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10),
                &mut ctx,
            )
            .unwrap();

        for i in 0..4u32 {
            assert_eq!(h.mem.read_u32(0x3000_0400 + i * 4).unwrap(), 0xDEAD_BEEF);
        }
    }

    #[test]
    fn a_launch_that_transfers_nothing_only_releases_its_semaphore() {
        let mut h = Harness::new();
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_IN, h.base + 0x200);
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x400);
        engine.regs.set(LINE_LENGTH_IN, 8);
        set_iova(&mut engine, SET_SEMAPHORE_A, h.base + 0x100);
        engine.regs.set(SET_SEMAPHORE_PAYLOAD, 0x1234);
        h.stats.submissions = 7;
        h.mem.write_u32(0x3000_0200, 0xABCD).unwrap();

        let mut ctx = h.ctx();
        // No transfer, flush, four-word semaphore.
        engine.write(LAUNCH_DMA, 0x14, &mut ctx).unwrap();
        assert_eq!(h.mem.read_u32(0x3000_0400).unwrap(), 0, "nothing copied");
        assert_eq!(h.mem.read_u64(0x3000_0100).unwrap(), 0x1234);
        assert_eq!(h.mem.read_u64(0x3000_0108).unwrap(), 7, "timestamp");

        engine.regs.set(SET_SEMAPHORE_PAYLOAD, 0x5678);
        let mut ctx = h.ctx();
        // A one-word release after a plain copy.
        engine.write(LAUNCH_DMA, 0x0a, &mut ctx).unwrap();
        assert_eq!(h.mem.read_u32(0x3000_0400).unwrap(), 0xABCD, "copied");
        assert_eq!(h.mem.read_u32(0x3000_0100).unwrap(), 0x5678);
        assert_eq!(h.mem.read_u64(0x3000_0108).unwrap(), 7, "untouched");
    }

    #[test]
    fn a_constant_fill_reads_no_source() {
        let mut h = Harness::new();
        let mut engine = EngineCopy::new();
        set_iova(&mut engine, OFFSET_OUT, h.base + 0x400);
        engine.regs.set(LINE_LENGTH_IN, 2);
        engine.regs.set(SET_REMAP_CONST, 0x1111_2222);
        engine.regs.set(SET_REMAP_CONST + 1, 0x3333_4444);
        // Two 4-byte destination components, RemapConst[0] then RemapConst[1].
        engine.regs.set(
            SET_REMAP_COMPONENTS,
            REMAP_CONST_0 | (REMAP_CONST_1 << 4) | (3 << 16) | (1 << 24),
        );

        let mut ctx = h.ctx();
        // Pipelined, pitch to pitch, one line, remapped; the source address is 0.
        engine.write(LAUNCH_DMA, 0x581, &mut ctx).unwrap();

        for i in 0..2u32 {
            let at = 0x3000_0400 + i * 8;
            assert_eq!(h.mem.read_u32(at).unwrap(), 0x1111_2222);
            assert_eq!(h.mem.read_u32(at + 4).unwrap(), 0x3333_4444);
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
            .write(LAUNCH_DMA, NON_PIPELINED | (1 << 7) | (1 << 9), &mut ctx)
            .unwrap();

        // Row 0 of a GOB lands at 0..16, 32..48, 256..272, 288..304.
        assert_eq!(h.mem.read_u8(0x3000_1000).unwrap(), 1);
        assert_eq!(h.mem.read_u8(0x3000_1000 + 32).unwrap(), 17);
        assert_eq!(h.mem.read_u8(0x3000_1000 + 256).unwrap(), 33);
        assert_eq!(h.mem.read_u8(0x3000_1000 + 288).unwrap(), 49);
    }
}
