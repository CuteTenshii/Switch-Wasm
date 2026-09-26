//! FERMI_TWOD_A (class 0x902D), the 2D blitter.
//!
//! `PixelsFromMemory` scales a source rectangle into a destination rectangle
//! using 32.32 fixed-point stepping, with point or bilinear sampling. deko3d
//! routes `dkCmdBufBlitImage` here whenever the copy engine cannot express the
//! operation (scaling, format conversion, filtering).

use crate::gpu::engine::Registers;
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::{ColorFormat, Layout, Surface};
use crate::{Error, Result};

const SET_DST_FORMAT: u32 = 0x080;
const SET_DST_MEMORY_LAYOUT: u32 = 0x081;
const SET_DST_BLOCK_SIZE: u32 = 0x082;
const SET_DST_PITCH: u32 = 0x085;
const SET_DST_WIDTH: u32 = 0x086;
const SET_DST_HEIGHT: u32 = 0x087;
const SET_DST_OFFSET: u32 = 0x088;
const SET_SRC_FORMAT: u32 = 0x08C;
const SET_SRC_MEMORY_LAYOUT: u32 = 0x08D;
const SET_SRC_BLOCK_SIZE: u32 = 0x08E;
const SET_SRC_PITCH: u32 = 0x091;
const SET_SRC_WIDTH: u32 = 0x092;
const SET_SRC_HEIGHT: u32 = 0x093;
const SET_SRC_OFFSET: u32 = 0x094;
const SET_OPERATION: u32 = 0x0AB;
const SAMPLE_MODE: u32 = 0x223;
const DST_X0: u32 = 0x22C;
const DST_Y0: u32 = 0x22D;
const DST_WIDTH: u32 = 0x22E;
const DST_HEIGHT: u32 = 0x22F;
const DU_DX_FRAC: u32 = 0x230;
const DU_DX_INT: u32 = 0x231;
const DV_DY_FRAC: u32 = 0x232;
const DV_DY_INT: u32 = 0x233;
const SRC_X0_FRAC: u32 = 0x234;
const SRC_X0_INT: u32 = 0x235;
const SRC_Y0_FRAC: u32 = 0x236;
/// Writing this register triggers the blit.
const SRC_Y0_INT: u32 = 0x237;

const MEMORY_LAYOUT_PITCH: u32 = 1;
const OPERATION_SRC_COPY: u32 = 3;
const FILTER_BILINEAR: u32 = 1;

/// The tile a byte-exact copy walks the destination in: a GOB column of
/// 32-bit texels at a halving step, and one 16-GOB block of rows.
const TILE_W: usize = 8;
const TILE_H: usize = 64;

#[derive(Debug, Default)]
pub struct Engine2D {
    pub regs: Registers,
    /// Both surfaces of the last staged copy, kept between blits so a title
    /// that resolves every frame does not allocate and zero 18 MB each time.
    /// See [`Engine2D::blit_staged`].
    source: Vec<u8>,
    target: Vec<u8>,
    /// The texels the last staged copy read out of its source, in the order
    /// it wrote them, and where they came from. A copy of the same texels
    /// from a source nothing has written since reuses them rather than
    /// reading the source again.
    resolved: Vec<u8>,
    resolved_from: Option<ResolvedFrom>,
    /// Blits by source and destination: see [`crate::gpu::activity`].
    pub activity: crate::gpu::activity::GpuActivity,
}

impl Engine2D {
    pub fn new() -> Engine2D {
        Engine2D {
            regs: Registers::new(),
            source: Vec::new(),
            target: Vec::new(),
            resolved: Vec::new(),
            resolved_from: None,
            activity: Default::default(),
        }
    }

    /// Which method write launches a blit, so a caller can hand a GPU
    /// backend's surfaces back before the copy reads guest memory.
    pub const LAUNCHES_BLIT: u32 = SRC_Y0_INT;

    pub fn write(&mut self, method: u32, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        self.regs.set(method, arg);
        if method == Engine2D::LAUNCHES_BLIT {
            self.blit(ctx)?;
        }
        Ok(())
    }

    fn surface(&self, dst: bool) -> Result<Surface> {
        let (format, layout_reg, block_reg, pitch_reg, width_reg, height_reg, offset_reg) = if dst {
            (
                SET_DST_FORMAT,
                SET_DST_MEMORY_LAYOUT,
                SET_DST_BLOCK_SIZE,
                SET_DST_PITCH,
                SET_DST_WIDTH,
                SET_DST_HEIGHT,
                SET_DST_OFFSET,
            )
        } else {
            (
                SET_SRC_FORMAT,
                SET_SRC_MEMORY_LAYOUT,
                SET_SRC_BLOCK_SIZE,
                SET_SRC_PITCH,
                SET_SRC_WIDTH,
                SET_SRC_HEIGHT,
                SET_SRC_OFFSET,
            )
        };
        let format = ColorFormat::from_raw(self.regs.get(format))?;
        let pitch_linear = self.regs.get(layout_reg) == MEMORY_LAYOUT_PITCH;
        let layout = if pitch_linear {
            Layout::Pitch {
                pitch: self.regs.get(pitch_reg),
            }
        } else {
            Layout::BlockLinear {
                block_height_gobs: 1 << self.regs.field(block_reg, 4, 6),
            }
        };
        Ok(Surface {
            addr: self.regs.iova(offset_reg),
            width: self.regs.get(width_reg),
            height: self.regs.get(height_reg),
            format,
            layout,
        })
    }

    fn blit(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        let operation = self.regs.get(SET_OPERATION);
        if operation != OPERATION_SRC_COPY {
            return Err(Error::Gpu(format!(
                "2d: blit operation {} is not implemented (only SrcCopy)",
                operation
            )));
        }
        let src = self.surface(false)?;
        let dst = self.surface(true)?;
        let dst_x0 = self.regs.get(DST_X0);
        let dst_y0 = self.regs.get(DST_Y0);
        let dst_w = self.regs.get(DST_WIDTH);
        let dst_h = self.regs.get(DST_HEIGHT);
        let du_dx = fixed(self.regs.get(DU_DX_INT), self.regs.get(DU_DX_FRAC));
        let dv_dy = fixed(self.regs.get(DV_DY_INT), self.regs.get(DV_DY_FRAC));
        let src_x0 = fixed(self.regs.get(SRC_X0_INT), self.regs.get(SRC_X0_FRAC));
        let src_y0 = fixed(self.regs.get(SRC_Y0_INT), self.regs.get(SRC_Y0_FRAC));
        let bilinear = self.regs.field(SAMPLE_MODE, 4, 4) == FILTER_BILINEAR;

        // A filter whose taps land exactly on texel centres is not a filter.
        // `bilinear` weights its four taps by the fractional part of
        // `(u, v) - 0.5`, so a step and an origin that leave that zero for
        // every pixel give weights of 1, 0, 0, 0, `c00` plus three terms
        // multiplied by zero. Just Dance 2019 resolves its 2x2 MSAA target
        // with `du_dx=2, dv_dy=2, src0=(0.5,0.5)`, which is exactly that, and
        // paid four fetches and twelve lerps per pixel to copy one texel.
        let centred = |origin: f64, step: f64| (origin - 0.5).fract() == 0.0 && step.fract() == 0.0;
        let filtered = bilinear && !(centred(src_x0, du_dx) && centred(src_y0, dv_dy));

        {
            use crate::gpu::activity::surface_text;
            let (src_cpu, dst_cpu) = (ctx.span(src.addr, 1), ctx.span(dst.addr, 1));
            self.activity.note(
                crate::gpu::activity::Kind::Blit,
                src.addr,
                dst.addr,
                u64::from(dst_w) * u64::from(dst_h),
                false,
                || {
                    format!(
                        "{} -> {}, {dst_w}x{dst_h} at ({dst_x0},{dst_y0}), {}",
                        surface_text(src.addr, src_cpu, src.width, src.height, src.format.raw),
                        surface_text(dst.addr, dst_cpu, dst.width, dst.height, dst.format.raw),
                        if filtered {
                            "filtered"
                        } else {
                            "point-sampled"
                        }
                    )
                },
            );
        }
        if ctx.trace || crate::trace::enabled(crate::trace::Trace::Copy) {
            crate::traceln!(
                "[gpu] 2d blit src {:#x} {}x{} fmt={:#x} -> dst {:#x} ({},{}) {}x{} fmt={:#x} \
                 du_dx={du_dx} dv_dy={dv_dy} src0=({src_x0},{src_y0}) bilinear={bilinear} \
                 filtered={filtered} layout={:?}/{:?} cpu src {:x?} dst {:x?}",
                src.addr,
                src.width,
                src.height,
                src.format.raw,
                dst.addr,
                dst_x0,
                dst_y0,
                dst_w,
                dst_h,
                dst.format.raw,
                src.layout,
                dst.layout,
                ctx.vmm.translate(src.addr).map(|(c, _)| c),
                ctx.vmm.translate(dst.addr).map(|(c, _)| c),
            );
        }

        // Point-sampling between two surfaces of one byte-exact format is a
        // move, not a conversion: `encode(decode(x))` is `x` there, and the
        // pair costs a round trip through linear light for every pixel of the
        // destination. A title that resolves its render target this way does
        // 921,600 of them a frame.
        let byte_exact =
            !filtered && src.format.raw == dst.format.raw && dst.format.is_byte_exact();
        let bpp = dst.format.bytes_per_pixel;

        // A copy walks both surfaces whole, so their GPU addresses are worth
        // translating once here rather than once per texel. `Gpu::present`
        // has always worked this way; the blitter went through `ExecCtx` per
        // texel and paid an address-space lookup for every pixel it read and
        // every pixel it wrote, which measured 40% of the copy.
        //
        // Only when each surface lies in one mapping: a copy that would walk
        // off the end of one is left to the general path below, which asks
        // per texel and so cannot.
        if byte_exact {
            if let (Some(src_base), Some(dst_base)) = (mapped(&src, ctx), mapped(&dst, ctx)) {
                let (src_width, dst_width) = (src.width_bytes(), dst.width_bytes());
                // A column's half of the swizzle depends on `x` alone, and the
                // same 1280 columns are walked for every one of 720 rows, so
                // they are worked out once here instead of 921,600 times, and
                // the source step's fixed-point arithmetic goes with them.
                let columns: Vec<(u32, u32)> = (0..dst_w)
                    .map(|x| {
                        let u = src_x0 + du_dx * x as f64;
                        let sx = (u.max(0.0) as u32).min(src.width.saturating_sub(1));
                        (
                            src.layout.column_offset(sx * bpp),
                            dst.layout.column_offset((dst_x0 + x) * bpp),
                        )
                    })
                    .collect();
                // The same for a row's half, once per row rather than 1280
                // times.
                let rows: Vec<(u32, u32)> = (0..dst_h)
                    .map(|y| {
                        let v = src_y0 + dv_dy * y as f64;
                        // `Surface::texel_raw` clamps to the surface; reading
                        // the addresses directly means doing that here
                        // instead.
                        let sy = (v.max(0.0) as u32).min(src.height.saturating_sub(1));
                        (
                            src.layout.row_offset(sy, src_width),
                            dst.layout.row_offset(dst_y0 + y, dst_width),
                        )
                    })
                    .collect();
                // Walked a destination row at a time, a block-linear source is
                // read in the one order its layout makes slowest: eight texels
                // of a GOB, then a jump of a whole column of GOBs to the next,
                // a new cache line every few texels and in no pattern a
                // prefetcher follows. In tiles, each one's source is a few
                // contiguous blocks.
                //
                // Only the order changes, which nothing can see unless the
                // copy reads what it has already written, so overlapping
                // surfaces keep going a row at a time. The hardware does not
                // promise an order either.
                let disjoint = u64::from(src_base) + u64::from(src.size()) <= u64::from(dst_base)
                    || u64::from(dst_base) + u64::from(dst.size()) <= u64::from(src_base);
                if disjoint
                    && self.blit_staged(ctx, (&src, src_base), (&dst, dst_base), &rows, &columns)?
                {
                    ctx.stats.copies += 1;
                    return Ok(());
                }
                let (tile_w, tile_h) = if disjoint {
                    (TILE_W, TILE_H)
                } else {
                    (columns.len().max(1), 1)
                };
                for band in rows.chunks(tile_h) {
                    for strip in columns.chunks(tile_w) {
                        for &(src_row, dst_row) in band {
                            let (src_row, dst_row) = (
                                src_base.wrapping_add(src_row),
                                dst_base.wrapping_add(dst_row),
                            );
                            for &(from, to) in strip {
                                let raw = ctx.mem.read_le(src_row.wrapping_add(from), bpp)?;
                                ctx.mem.write_le(dst_row.wrapping_add(to), bpp, raw)?;
                            }
                        }
                    }
                }
                ctx.stats.copies += 1;
                return Ok(());
            }
        }

        for y in 0..dst_h {
            let v = src_y0 + dv_dy * y as f64;
            for x in 0..dst_w {
                let u = src_x0 + du_dx * x as f64;
                let va = dst.addr + dst.offset(dst_x0 + x, dst_y0 + y) as u64;
                if byte_exact {
                    let raw = src.texel_raw(u.max(0.0) as u32, v.max(0.0) as u32, ctx)?;
                    ctx.write_pixel(va, bpp, raw)?;
                    continue;
                }
                let color = if filtered {
                    src.sample_bilinear(u, v, ctx)?
                } else {
                    src.sample_point(u, v, ctx)?
                };
                ctx.write_pixel(va, bpp, dst.format.encode(color)?)?;
            }
        }
        ctx.stats.copies += 1;
        Ok(())
    }

    /// A byte-exact copy between two disjoint surfaces, done in host memory:
    /// the source read whole and its texels gathered into one buffer, the
    /// target read whole, the texels put in it, and the target written back.
    /// `rows` and `columns` are the two halves of every texel's offset into
    /// its surface, source first. Reports whether it did the copy; when it
    /// did not, nothing has been written.
    ///
    /// Just Dance 2019 resolves a 2560x1440 target this way every frame, and
    /// through guest memory each of the 921,600 texels it reads was a
    /// dependent load out of whichever of 3,600 separately allocated pages
    /// held it, so the copy was a chain of cache misses: 13 ms of a frame.
    /// Staged, the misses are page-sized sequential copies instead. And the
    /// gathered texels are kept, with the source's pages watched, so a copy
    /// of the same texels from a source nothing has stored to since skips
    /// the source altogether; that target is one Just Dance 2019 stopped
    /// drawing to when it finished loading.
    ///
    /// Only where that is exactly the per-texel copy: every offset inside its
    /// surface, no watchpoint over either and no protected byte in the
    /// target, so skipping the per-access checks skips nothing. Writing back
    /// the bytes between texels, the padding of a block-linear surface,
    /// stores what was just read from them.
    fn blit_staged(
        &mut self,
        ctx: &mut ExecCtx,
        (src, src_base): (&Surface, u32),
        (dst, dst_base): (&Surface, u32),
        rows: &[(u32, u32)],
        columns: &[(u32, u32)],
    ) -> Result<bool> {
        let bpp = dst.format.bytes_per_pixel;
        let furthest =
            |pick: fn(&(u32, u32)) -> u32, rows: &[(u32, u32)], columns: &[(u32, u32)]| {
                u64::from(rows.iter().map(pick).max().unwrap_or(0))
                    + u64::from(columns.iter().map(pick).max().unwrap_or(0))
                    + u64::from(bpp)
            };
        let inside = furthest(|&(s, _)| s, rows, columns) <= u64::from(src.size())
            && furthest(|&(_, d)| d, rows, columns) <= u64::from(dst.size());
        if !inside
            || !matches!(bpp, 1 | 2 | 4 | 8 | 16)
            || !ctx.mem.plainly_readable(src_base, src.size())
            || !ctx.mem.plainly_writable(dst_base, dst.size())
        {
            return Ok(false);
        }
        let texels = rows.len() * columns.len() * bpp as usize;
        // The source's texels are what they were at the last copy if it read
        // the same ones and nothing has been stored to its pages since. Just
        // Dance 2019 resolves a target it has not drawn to since it loaded,
        // so this is every frame of it.
        let unwritten = !ctx.mem.take_copy_written();
        let current = self
            .resolved_from
            .as_ref()
            .is_some_and(|from| unwritten && from.is(src_base, src.size(), bpp, rows, columns));
        if !current {
            let mut source = std::mem::take(&mut self.source);
            source.resize(src.size() as usize, 0);
            let read = ctx.mem.read_into(src_base, &mut source);
            if read.is_ok() {
                self.resolved.resize(texels, 0);
                match bpp {
                    1 => gather::<1>(&source, &mut self.resolved, rows, columns),
                    2 => gather::<2>(&source, &mut self.resolved, rows, columns),
                    4 => gather::<4>(&source, &mut self.resolved, rows, columns),
                    8 => gather::<8>(&source, &mut self.resolved, rows, columns),
                    _ => gather::<16>(&source, &mut self.resolved, rows, columns),
                }
                ctx.mem.mark_copy_range(src_base, src.size());
                self.resolved_from =
                    Some(ResolvedFrom::new(src_base, src.size(), bpp, rows, columns));
            } else {
                self.resolved_from = None;
            }
            self.source = source;
            if read.is_err() {
                return Ok(false);
            }
        }
        let mut target = std::mem::take(&mut self.target);
        target.resize(dst.size() as usize, 0);
        let result = if ctx.mem.read_into(dst_base, &mut target).is_ok() {
            match bpp {
                1 => scatter::<1>(&self.resolved, &mut target, rows, columns),
                2 => scatter::<2>(&self.resolved, &mut target, rows, columns),
                4 => scatter::<4>(&self.resolved, &mut target, rows, columns),
                8 => scatter::<8>(&self.resolved, &mut target, rows, columns),
                _ => scatter::<16>(&self.resolved, &mut target, rows, columns),
            }
            ctx.mem.write_from(dst_base, &target).map(|()| true)
        } else {
            Ok(false)
        };
        self.target = target;
        result
    }
}

/// What the texels in [`Engine2D::resolved`] were read from: the source's
/// place and size, the texel width, and the source's half of every row and
/// column offset. Two copies that agree on all of it read the same texels.
#[derive(Debug)]
struct ResolvedFrom {
    base: u32,
    size: u32,
    bpp: u32,
    rows: Vec<u32>,
    columns: Vec<u32>,
}

impl ResolvedFrom {
    fn new(base: u32, size: u32, bpp: u32, rows: &[(u32, u32)], columns: &[(u32, u32)]) -> Self {
        ResolvedFrom {
            base,
            size,
            bpp,
            rows: rows.iter().map(|&(from, _)| from).collect(),
            columns: columns.iter().map(|&(from, _)| from).collect(),
        }
    }

    fn is(
        &self,
        base: u32,
        size: u32,
        bpp: u32,
        rows: &[(u32, u32)],
        columns: &[(u32, u32)],
    ) -> bool {
        self.base == base
            && self.size == size
            && self.bpp == bpp
            && self
                .rows
                .iter()
                .copied()
                .eq(rows.iter().map(|&(from, _)| from))
            && self
                .columns
                .iter()
                .copied()
                .eq(columns.iter().map(|&(from, _)| from))
    }
}

/// Read every texel a staged blit copies out of `source`, `N` bytes each,
/// into `resolved` in destination order, a row of columns at a time; a
/// tile at a time, so the source is read a few blocks at a time too. A
/// width known here makes each move one load and one store rather than a
/// call to `memcpy`.
fn gather<const N: usize>(
    source: &[u8],
    resolved: &mut [u8],
    rows: &[(u32, u32)],
    columns: &[(u32, u32)],
) {
    let width = columns.len();
    for (band_at, band) in rows.chunks(TILE_H).enumerate() {
        for (strip_at, strip) in columns.chunks(TILE_W).enumerate() {
            for (r, &(src_row, _)) in band.iter().enumerate() {
                let row = band_at * TILE_H + r;
                for (c, &(from, _)) in strip.iter().enumerate() {
                    let at = (src_row + from) as usize;
                    let texel: [u8; N] = source[at..at + N].try_into().unwrap();
                    let slot = (row * width + strip_at * TILE_W + c) * N;
                    resolved[slot..slot + N].copy_from_slice(&texel);
                }
            }
        }
    }
}

/// Write the texels [`gather`] collected into `target`, each where the
/// destination's half of its row and column offsets puts it.
/// [`Engine2D::blit_staged`] has already established that the furthest any
/// row and column offset can add up to is inside `target`, and that `resolved`
/// holds exactly one texel per row and column. Neither index here can be out
/// of range, and saying so is what leaves the walk without a bounds check and
/// without the panic path behind it: the destination offset was being spilled
/// to the stack on the way past every one of 921,600 texels a frame.
fn scatter<const N: usize>(
    resolved: &[u8],
    target: &mut [u8],
    rows: &[(u32, u32)],
    columns: &[(u32, u32)],
) {
    let row_bytes = columns.len() * N;
    let Some(last) = target.len().checked_sub(N) else {
        return;
    };
    if row_bytes == 0 {
        return;
    }
    for (&(_, dst_row), row_texels) in rows.iter().zip(resolved.chunks_exact(row_bytes)) {
        for (&(_, to), texel) in columns.iter().zip(row_texels.as_chunks::<N>().0) {
            let at = ((dst_row + to) as usize).min(last);
            target[at..at + N].copy_from_slice(texel);
        }
    }
}

/// Recombine the 32.32 fixed-point pairs the engine takes.
fn fixed(int_part: u32, frac: u32) -> f64 {
    int_part as i32 as f64 + frac as f64 / 4_294_967_296.0
}

/// Where a surface begins in guest memory, if the whole of it is one mapping.
///
/// `None` means some of it is not, and the caller has to go on asking per
/// texel, a translation that covered less than the surface would hand out an
/// address past the end of the mapping for the rest of it.
fn mapped(surface: &Surface, ctx: &ExecCtx) -> Option<u32> {
    match ctx.vmm.translate(surface.addr) {
        Some((cpu, left)) if left >= u64::from(surface.size()) => Some(cpu),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::exec::GpuStats;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    fn set_iova(engine: &mut Engine2D, method: u32, va: u64) {
        engine.regs.set(method, (va >> 32) as u32);
        engine.regs.set(method + 1, va as u32);
    }

    #[test]
    fn one_to_one_blit_copies_pixels() {
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x2000).unwrap();
        for i in 0..16u32 {
            mem.write_u32(0x3000_0000 + i * 4, 0xFF00_0000 | i).unwrap();
        }
        let mut vmm = AddressSpace::new();
        let base = vmm
            .map(0x3000_0000, 0x2000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();

        let mut engine = Engine2D::new();
        engine.regs.set(SET_SRC_FORMAT, 0xD5);
        engine.regs.set(SET_SRC_MEMORY_LAYOUT, MEMORY_LAYOUT_PITCH);
        engine.regs.set(SET_SRC_PITCH, 16);
        engine.regs.set(SET_SRC_WIDTH, 4);
        engine.regs.set(SET_SRC_HEIGHT, 4);
        set_iova(&mut engine, SET_SRC_OFFSET, base);

        engine.regs.set(SET_DST_FORMAT, 0xD5);
        engine.regs.set(SET_DST_MEMORY_LAYOUT, MEMORY_LAYOUT_PITCH);
        engine.regs.set(SET_DST_PITCH, 16);
        engine.regs.set(SET_DST_WIDTH, 4);
        engine.regs.set(SET_DST_HEIGHT, 4);
        set_iova(&mut engine, SET_DST_OFFSET, base + 0x1000);

        engine.regs.set(SET_OPERATION, OPERATION_SRC_COPY);
        engine.regs.set(DST_WIDTH, 4);
        engine.regs.set(DST_HEIGHT, 4);
        engine.regs.set(DU_DX_INT, 1);
        engine.regs.set(DV_DY_INT, 1);

        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        engine.write(SRC_Y0_INT, 0, &mut ctx).unwrap();

        for i in 0..16u32 {
            assert_eq!(
                mem.read_u32(0x3000_1000 + i * 4).unwrap(),
                0xFF00_0000 | i,
                "pixel {}",
                i
            );
        }
        assert_eq!(stats.copies, 1);
    }

    /// The staged copy is only allowed where it is the texel-by-texel one, so
    /// it has to leave memory byte for byte as that one does: every texel,
    /// and the padding a block-linear surface carries past its edges. A write
    /// watchpoint over the target is what sends a copy the other way, so the
    /// same resolve is run both ways and the results compared.
    #[test]
    fn a_staged_blit_leaves_memory_as_the_texel_walk_does() {
        const SRC: u32 = 0x3000_0000;
        const DST: u32 = 0x3001_0000;
        const BLOCK_16_GOBS: u32 = 4 << 4;
        let resolve = |watch: bool| {
            let mut mem = Memory::new();
            mem.map_zero(SRC, 0x2_0000).unwrap();
            for i in 0..0x2_0000 / 4 {
                mem.write_u32(SRC + i * 4, i.wrapping_mul(0x9E37_79B9))
                    .unwrap();
            }
            if watch {
                mem.watch_writes(DST + 0x5FF0, 4);
            }
            let mut vmm = AddressSpace::new();
            let base = vmm.map(SRC, 0x2_0000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
            let mut host1x = Host1x::new();
            let mut stats = GpuStats::default();
            let mut engine = Engine2D::new();
            // Neither size a whole number of GOBs, so both have padding.
            for (format, layout, block, width, height, offset, w, h, at) in [
                (
                    SET_SRC_FORMAT,
                    SET_SRC_MEMORY_LAYOUT,
                    SET_SRC_BLOCK_SIZE,
                    SET_SRC_WIDTH,
                    SET_SRC_HEIGHT,
                    SET_SRC_OFFSET,
                    74,
                    90,
                    base,
                ),
                (
                    SET_DST_FORMAT,
                    SET_DST_MEMORY_LAYOUT,
                    SET_DST_BLOCK_SIZE,
                    SET_DST_WIDTH,
                    SET_DST_HEIGHT,
                    SET_DST_OFFSET,
                    37,
                    45,
                    base + u64::from(DST - SRC),
                ),
            ] {
                engine.regs.set(format, 0xD5);
                engine.regs.set(layout, 0);
                engine.regs.set(block, BLOCK_16_GOBS);
                engine.regs.set(width, w);
                engine.regs.set(height, h);
                set_iova(&mut engine, offset, at);
            }
            engine.regs.set(SET_OPERATION, OPERATION_SRC_COPY);
            engine.regs.set(DST_WIDTH, 37);
            engine.regs.set(DST_HEIGHT, 45);
            engine.regs.set(DU_DX_INT, 2);
            engine.regs.set(DV_DY_INT, 2);
            let mut ctx = ExecCtx {
                mem: &mut mem,
                vmm: &vmm,
                host1x: &mut host1x,
                stats: &mut stats,
                trace: false,
            };
            engine.write(SRC_Y0_INT, 0, &mut ctx).unwrap();
            assert_eq!(stats.copies, 1);
            mem.dump(SRC, 0x2_0000).unwrap()
        };
        let (staged, walked) = (resolve(false), resolve(true));
        let differs = staged.iter().zip(&walked).position(|(a, b)| a != b);
        assert_eq!(differs, None, "first differing byte, from {SRC:#x}");
    }

    /// A copy that reuses the texels it gathered last time has to notice
    /// that the source changed under it, whether a store or a new mapping
    /// did it.
    #[test]
    fn a_repeated_blit_sees_what_was_written_to_its_source() {
        const SRC: u32 = 0x3000_0000;
        const DST: u32 = 0x3001_0000;
        let mut mem = Memory::new();
        mem.map_zero(SRC, 0x2_0000).unwrap();
        for i in 0..64 * 64 {
            mem.write_u32(SRC + i * 4, i).unwrap();
        }
        let mut vmm = AddressSpace::new();
        let base = vmm.map(SRC, 0x2_0000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();
        let mut engine = Engine2D::new();
        for (format, layout, pitch, width, height, offset, w, at) in [
            (
                SET_SRC_FORMAT,
                SET_SRC_MEMORY_LAYOUT,
                SET_SRC_PITCH,
                SET_SRC_WIDTH,
                SET_SRC_HEIGHT,
                SET_SRC_OFFSET,
                64,
                base,
            ),
            (
                SET_DST_FORMAT,
                SET_DST_MEMORY_LAYOUT,
                SET_DST_PITCH,
                SET_DST_WIDTH,
                SET_DST_HEIGHT,
                SET_DST_OFFSET,
                32,
                base + u64::from(DST - SRC),
            ),
        ] {
            engine.regs.set(format, 0xD5);
            engine.regs.set(layout, MEMORY_LAYOUT_PITCH);
            engine.regs.set(pitch, w * 4);
            engine.regs.set(width, w);
            engine.regs.set(height, w);
            set_iova(&mut engine, offset, at);
        }
        engine.regs.set(SET_OPERATION, OPERATION_SRC_COPY);
        engine.regs.set(DST_WIDTH, 32);
        engine.regs.set(DST_HEIGHT, 32);
        engine.regs.set(DU_DX_INT, 2);
        engine.regs.set(DV_DY_INT, 2);
        let mut blit = |mem: &mut Memory| {
            let mut ctx = ExecCtx {
                mem,
                vmm: &vmm,
                host1x: &mut host1x,
                stats: &mut stats,
                trace: false,
            };
            engine.write(SRC_Y0_INT, 0, &mut ctx).unwrap();
        };
        // Destination texel (5, 7) comes from source texel (10, 14).
        let (from, to) = (SRC + (14 * 64 + 10) * 4, DST + (7 * 32 + 5) * 4);
        blit(&mut mem);
        assert_eq!(mem.read_u32(to).unwrap(), 14 * 64 + 10);
        mem.write_u32(to, 0).unwrap();
        blit(&mut mem);
        assert_eq!(
            mem.read_u32(to).unwrap(),
            14 * 64 + 10,
            "unchanged source, copied again"
        );
        mem.write_u32(from, 0xABCD).unwrap();
        blit(&mut mem);
        assert_eq!(mem.read_u32(to).unwrap(), 0xABCD, "a store to the source");
        mem.map(from & !0xFFF, &[0x5A; 0x1000]).unwrap();
        blit(&mut mem);
        assert_eq!(
            mem.read_u32(to).unwrap(),
            0x5A5A_5A5A,
            "a new mapping of the source"
        );
    }

    /// A halving copy bigger than one tile in both directions, so the tiled
    /// walk has to get every band and every strip right, the ragged last ones
    /// included.
    #[test]
    fn a_blit_larger_than_a_tile_lands_every_texel() {
        const SRC_W: u32 = 2 * 21;
        const SRC_H: u32 = 2 * 70;
        const DST_W: u32 = SRC_W / 2;
        const DST_H: u32 = SRC_H / 2;
        const SRC: u32 = 0x3000_0000;
        const DST: u32 = 0x3001_0000;
        let mut mem = Memory::new();
        mem.map_zero(SRC, 0x2_0000).unwrap();
        for i in 0..SRC_W * SRC_H {
            mem.write_u32(SRC + i * 4, i).unwrap();
        }
        let mut vmm = AddressSpace::new();
        let base = vmm.map(SRC, 0x2_0000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        let mut host1x = Host1x::new();
        let mut stats = GpuStats::default();

        let mut engine = Engine2D::new();
        engine.regs.set(SET_SRC_FORMAT, 0xD5);
        engine.regs.set(SET_SRC_MEMORY_LAYOUT, MEMORY_LAYOUT_PITCH);
        engine.regs.set(SET_SRC_PITCH, SRC_W * 4);
        engine.regs.set(SET_SRC_WIDTH, SRC_W);
        engine.regs.set(SET_SRC_HEIGHT, SRC_H);
        set_iova(&mut engine, SET_SRC_OFFSET, base);

        engine.regs.set(SET_DST_FORMAT, 0xD5);
        engine.regs.set(SET_DST_MEMORY_LAYOUT, MEMORY_LAYOUT_PITCH);
        engine.regs.set(SET_DST_PITCH, DST_W * 4);
        engine.regs.set(SET_DST_WIDTH, DST_W);
        engine.regs.set(SET_DST_HEIGHT, DST_H);
        set_iova(&mut engine, SET_DST_OFFSET, base + u64::from(DST - SRC));

        engine.regs.set(SET_OPERATION, OPERATION_SRC_COPY);
        engine.regs.set(DST_WIDTH, DST_W);
        engine.regs.set(DST_HEIGHT, DST_H);
        engine.regs.set(DU_DX_INT, 2);
        engine.regs.set(DV_DY_INT, 2);

        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        engine.write(SRC_Y0_INT, 0, &mut ctx).unwrap();

        for y in 0..DST_H {
            for x in 0..DST_W {
                assert_eq!(
                    mem.read_u32(DST + (y * DST_W + x) * 4).unwrap(),
                    (2 * y) * SRC_W + 2 * x,
                    "texel ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn fixed_point_conversion() {
        assert_eq!(fixed(1, 0), 1.0);
        assert_eq!(fixed(0, 0x8000_0000), 0.5);
        assert_eq!(fixed(2, 0x8000_0000), 2.5);
    }
}
