//! FERMI_TWOD_A (class 0x902D), the 2D blitter: `PixelsFromMemory` scales a
//! source rectangle into a destination with 32.32 fixed-point stepping and
//! point or bilinear sampling.

use crate::gpu::engine::Registers;
use crate::gpu::exec::ExecCtx;
use crate::gpu::renderer::Blit;
use crate::gpu::surface::{blend, taps, ColorFormat, Layout, Surface, GOB_HEIGHT, GOB_WIDTH};
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

/// Destination tile for a byte-exact copy: a GOB column of 32-bit texels at a
/// halving step, by one 16-GOB block of rows.
const TILE_W: usize = 8;
const TILE_H: usize = 64;

#[derive(Debug, Default)]
pub struct Engine2D {
    pub regs: Registers,
    /// Both surfaces of the last staged copy, reused between blits.
    source: Vec<u8>,
    target: Vec<u8>,
    /// The texels the last staged copy read, and their source, for reuse.
    resolved: Vec<u8>,
    resolved_from: Option<ResolvedFrom>,
    /// A filtered blit's source and destination rectangles as plain rows.
    rows_in: Vec<u8>,
    rows_out: Vec<u8>,
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
            rows_in: Vec::new(),
            rows_out: Vec::new(),
            activity: Default::default(),
        }
    }

    /// The method that launches a blit.
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

    /// The source copy the registers describe.
    pub fn request(&self) -> Result<Blit> {
        let operation = self.regs.get(SET_OPERATION);
        if operation != OPERATION_SRC_COPY {
            return Err(Error::Gpu(format!(
                "2d: blit operation {} is not implemented (only SrcCopy)",
                operation
            )));
        }
        let du_dx = fixed(self.regs.get(DU_DX_INT), self.regs.get(DU_DX_FRAC));
        let dv_dy = fixed(self.regs.get(DV_DY_INT), self.regs.get(DV_DY_FRAC));
        let src_x0 = fixed(self.regs.get(SRC_X0_INT), self.regs.get(SRC_X0_FRAC));
        let src_y0 = fixed(self.regs.get(SRC_Y0_INT), self.regs.get(SRC_Y0_FRAC));
        let bilinear = self.regs.field(SAMPLE_MODE, 4, 4) == FILTER_BILINEAR;
        // Taps exactly on texel centres make bilinear a point sample.
        let centred = |origin: f64, step: f64| (origin - 0.5).fract() == 0.0 && step.fract() == 0.0;
        Ok(Blit {
            src: self.surface(false)?,
            dst: self.surface(true)?,
            dst_rect: (
                self.regs.get(DST_X0),
                self.regs.get(DST_Y0),
                self.regs.get(DST_WIDTH),
                self.regs.get(DST_HEIGHT),
            ),
            src_origin: (src_x0, src_y0),
            step: (du_dx, dv_dy),
            filtered: bilinear && !(centred(src_x0, du_dx) && centred(src_y0, dv_dy)),
        })
    }

    /// Count a blit, whichever side ran it.
    pub fn note(&mut self, blit: &Blit, ctx: &ExecCtx) {
        use crate::gpu::activity::surface_text;
        let (src, dst) = (&blit.src, &blit.dst);
        let (dst_x0, dst_y0, dst_w, dst_h) = blit.dst_rect;
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
                    if blit.filtered {
                        "filtered"
                    } else {
                        "point-sampled"
                    }
                )
            },
        );
    }

    fn blit(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        let request = self.request()?;
        let Blit {
            src,
            dst,
            dst_rect: (dst_x0, dst_y0, dst_w, dst_h),
            src_origin: (src_x0, src_y0),
            step: (du_dx, dv_dy),
            filtered,
        } = request;
        let bilinear = self.regs.field(SAMPLE_MODE, 4, 4) == FILTER_BILINEAR;
        self.note(&request, ctx);
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

        // Point-sampling between surfaces of one byte-exact format is a move.
        let byte_exact =
            !filtered && src.format.raw == dst.format.raw && dst.format.is_byte_exact();
        let bpp = dst.format.bytes_per_pixel;

        // Translate each surface once, when it lies in a single mapping.
        if byte_exact {
            if let (Some(src_base), Some(dst_base)) = (mapped(&src, ctx), mapped(&dst, ctx)) {
                let (src_width, dst_width) = (src.width_bytes(), dst.width_bytes());
                // A column's half of the swizzle depends on `x` alone.
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
                let rows: Vec<(u32, u32)> = (0..dst_h)
                    .map(|y| {
                        let v = src_y0 + dv_dy * y as f64;
                        // `Surface::texel_raw` would clamp; do it here.
                        let sy = (v.max(0.0) as u32).min(src.height.saturating_sub(1));
                        (
                            src.layout.row_offset(sy, src_width),
                            dst.layout.row_offset(dst_y0 + y, dst_width),
                        )
                    })
                    .collect();
                // Walk in tiles so a block-linear source is read a few blocks
                // at a time; overlapping surfaces keep row order.
                let disjoint = disjoint((&src, src_base), (&dst, dst_base));
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

        let inside = u64::from(dst_x0) + u64::from(dst_w) <= u64::from(dst.width)
            && u64::from(dst_y0) + u64::from(dst_h) <= u64::from(dst.height);
        if filtered && inside {
            if let (Some(src_base), Some(dst_base)) = (mapped(&src, ctx), mapped(&dst, ctx)) {
                if disjoint((&src, src_base), (&dst, dst_base))
                    && self.blit_filtered(
                        ctx,
                        (&src, src_base),
                        (&dst, dst_base),
                        (dst_x0, dst_y0, dst_w, dst_h),
                        (src_x0, src_y0),
                        (du_dx, dv_dy),
                    )?
                {
                    ctx.stats.copies += 1;
                    return Ok(());
                }
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

    /// A byte-exact copy between disjoint surfaces done in host memory, reusing
    /// the gathered texels when the source has not been written since.
    /// `rows` and `columns` are the two halves of each texel's offset, source
    /// first. Only used where it equals the per-texel copy (offsets in bounds,
    /// no watchpoints or protected bytes); returns false having written nothing.
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
        // Reuse last copy's texels if the same ones were read and the source is unchanged.
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

    /// Filtered blit in host memory, decoding each source row once. Surfaces
    /// must be disjoint and each in one mapping, with the destination rectangle
    /// inside its surface. Returns false having written nothing where memory
    /// needs per-access handling.
    fn blit_filtered(
        &mut self,
        ctx: &mut ExecCtx,
        (src, src_base): (&Surface, u32),
        (dst, dst_base): (&Surface, u32),
        (dst_x0, dst_y0, dst_w, dst_h): (u32, u32, u32, u32),
        (src_x0, src_y0): (f64, f64),
        (du_dx, dv_dy): (f64, f64),
    ) -> Result<bool> {
        if !ctx.mem.plainly_readable(src_base, src.size())
            || !ctx.mem.plainly_writable(dst_base, dst.size())
        {
            return Ok(false);
        }
        let last_x = src.width.saturating_sub(1);
        let last_y = src.height.saturating_sub(1);
        let columns: Vec<(u32, u32, f32)> = (0..dst_w)
            .map(|x| {
                let (x0, fx) = taps(src_x0 + du_dx * x as f64);
                (x0.min(last_x), x0.saturating_add(1).min(last_x), fx)
            })
            .collect();
        let Some(lo) = columns.iter().map(|c| c.0).min() else {
            return Ok(true);
        };
        let hi = columns.iter().map(|c| c.1).max().unwrap_or(lo);
        let (src_bpp, dst_bpp) = (src.format.bytes_per_pixel, dst.format.bytes_per_pixel);
        let (source, target) = (src.format.codec(), dst.format.codec());

        let mut from = std::mem::take(&mut self.source);
        let mut to = std::mem::take(&mut self.target);
        let mut rows_in = std::mem::take(&mut self.rows_in);
        let mut rows_out = std::mem::take(&mut self.rows_out);
        from.resize(src.size() as usize, 0);
        to.resize(dst.size() as usize, 0);
        let in_bytes = ((hi - lo + 1) * src_bpp) as usize;
        let out_bytes = (dst_w * dst_bpp) as usize;
        rows_in.resize(in_bytes * src.height as usize, 0);
        rows_out.resize(out_bytes * dst_h as usize, 0);
        let mut inside = true;
        let staged = ctx.mem.read_into(src_base, &mut from).is_ok()
            && ctx.mem.read_into(dst_base, &mut to).is_ok()
            && {
                for_each_run(
                    src.layout,
                    src.width_bytes(),
                    (lo * src_bpp, (hi + 1) * src_bpp),
                    (0, src.height),
                    |at, linear, len| match from.get(at..at + len) {
                        Some(run) => copy_run(&mut rows_in[linear..linear + len], run),
                        None => inside = false,
                    },
                );
                inside
            }
            && {
                let (src_bpp, dst_bpp) = (src_bpp as usize, dst_bpp as usize);
                let decode_row = |y: u32, row: &mut Vec<[f32; 4]>| -> Result<()> {
                    row.clear();
                    let at = y as usize * in_bytes;
                    for texel in rows_in[at..at + in_bytes].chunks_exact(src_bpp) {
                        row.push(source.decode(le(texel))?);
                    }
                    Ok(())
                };
                let mut rows: [(Option<u32>, Vec<[f32; 4]>); 2] = Default::default();
                for (y, out) in (0..dst_h).zip(rows_out.chunks_exact_mut(out_bytes)) {
                    let (y0, fy) = taps(src_y0 + dv_dy * y as f64);
                    let (top, bottom) = (y0.min(last_y), y0.saturating_add(1).min(last_y));
                    for want in [top, bottom] {
                        if rows.iter().all(|(held, _)| *held != Some(want)) {
                            let keep = rows[0].0 == Some(top) || rows[0].0 == Some(bottom);
                            let slot = usize::from(keep);
                            decode_row(want, &mut rows[slot].1)?;
                            rows[slot].0 = Some(want);
                        }
                    }
                    let upper = &rows[usize::from(rows[1].0 == Some(top))].1;
                    let lower = &rows[usize::from(rows[1].0 == Some(bottom))].1;
                    for (&(a, b, fx), texel) in columns.iter().zip(out.chunks_exact_mut(dst_bpp)) {
                        let (a, b) = ((a - lo) as usize, (b - lo) as usize);
                        let color = blend(upper[a], upper[b], lower[a], lower[b], fx, fy);
                        put_le(texel, target.encode(color)?);
                    }
                }
                for_each_run(
                    dst.layout,
                    dst.width_bytes(),
                    (dst_x0 * dst_bpp as u32, (dst_x0 + dst_w) * dst_bpp as u32),
                    (dst_y0, dst_y0 + dst_h),
                    |at, linear, len| match to.get_mut(at..at + len) {
                        Some(run) => copy_run(run, &rows_out[linear..linear + len]),
                        None => inside = false,
                    },
                );
                inside
            };
        let written = if staged {
            ctx.mem.write_from(dst_base, &to).map(|()| true)
        } else {
            Ok(false)
        };
        self.source = from;
        self.target = to;
        self.rows_in = rows_in;
        self.rows_out = rows_out;
        written
    }
}

/// What [`Engine2D::resolved`] was read from; equal keys mean the same texels.
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

/// Gather a staged blit's source texels into `resolved` in destination order,
/// a tile at a time.
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

/// Scatter the texels [`gather`] collected into `target`. The caller has
/// established that every index is in range.
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

/// Call `visit(offset, linear, len)` for each contiguous run of the rectangle
/// `x_bytes` by `rows`: `offset` in the surface, `linear` in plain rows of the
/// rectangle. A block-linear surface is walked a GOB column of a block at a
/// time, which is its memory order.
fn for_each_run(
    layout: Layout,
    width_bytes: u32,
    (x0, x1): (u32, u32),
    (y0, y1): (u32, u32),
    mut visit: impl FnMut(usize, usize, usize),
) {
    let row_bytes = (x1 - x0) as usize;
    let row_offsets: Vec<usize> = (y0..y1)
        .map(|y| layout.row_offset(y, width_bytes) as usize)
        .collect();
    // Runs of one row, by column group: (x into the rectangle, length, column offset).
    let mut groups: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    let band = match layout {
        Layout::Pitch { .. } => {
            groups.push(vec![(0, row_bytes, layout.column_offset(x0) as usize)]);
            1
        }
        Layout::BlockLinear { block_height_gobs } => {
            let mut x = x0;
            while x < x1 {
                let (_, run) = layout.run_at(x, 0, width_bytes);
                let len = run.min(x1 - x);
                if groups.is_empty() || x % GOB_WIDTH == 0 {
                    groups.push(Vec::new());
                }
                if let Some(group) = groups.last_mut() {
                    group.push((
                        (x - x0) as usize,
                        len as usize,
                        layout.column_offset(x) as usize,
                    ));
                }
                x += len;
            }
            GOB_HEIGHT * block_height_gobs.max(1)
        }
    };
    let mut band_y = y0;
    while band_y < y1 {
        let band_end = (band_y - band_y % band + band).min(y1);
        for group in &groups {
            for y in band_y..band_end {
                let i = (y - y0) as usize;
                for &(x, len, column) in group {
                    visit(row_offsets[i] + column, i * row_bytes + x, len);
                }
            }
        }
        band_y = band_end;
    }
}

/// A texel of `bytes.len()` bytes as a little-endian value.
#[inline(always)]
fn le(bytes: &[u8]) -> u128 {
    match *bytes {
        [a, b, c, d] => u128::from(u32::from_le_bytes([a, b, c, d])),
        [a, b, c, d, e, f, g, h] => u128::from(u64::from_le_bytes([a, b, c, d, e, f, g, h])),
        _ => bytes
            .iter()
            .rev()
            .fold(0, |value, &byte| (value << 8) | u128::from(byte)),
    }
}

/// Store `value` little-endian in all of `out`.
#[inline(always)]
fn put_le(out: &mut [u8], value: u128) {
    match out.len() {
        4 => out.copy_from_slice(&(value as u32).to_le_bytes()),
        8 => out.copy_from_slice(&(value as u64).to_le_bytes()),
        len => out.copy_from_slice(&value.to_le_bytes()[..len]),
    }
}

/// `from` into `to`, as one 16-byte move when it is a whole block-linear run.
#[inline(always)]
fn copy_run(to: &mut [u8], from: &[u8]) {
    match (
        <&mut [u8; 16]>::try_from(&mut *to),
        <&[u8; 16]>::try_from(from),
    ) {
        (Ok(to), Ok(from)) => *to = *from,
        _ => to.copy_from_slice(from),
    }
}

fn disjoint((a, a_base): (&Surface, u32), (b, b_base): (&Surface, u32)) -> bool {
    u64::from(a_base) + u64::from(a.size()) <= u64::from(b_base)
        || u64::from(b_base) + u64::from(b.size()) <= u64::from(a_base)
}

fn fixed(int_part: u32, frac: u32) -> f64 {
    int_part as i32 as f64 + frac as f64 / 4_294_967_296.0
}

/// Where a surface begins in guest memory, if the whole of it is one mapping.
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
            as_id: 0,
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

    /// The staged copy must leave memory byte for byte as the per-texel copy
    /// does, padding included; a write watchpoint forces the per-texel path.
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
                as_id: 0,
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

    /// A filtered blit matches sampling each destination pixel on its own.
    #[test]
    fn a_filtered_blit_matches_per_pixel_sampling() {
        const SRC: u32 = 0x3000_0000;
        const DST: u32 = 0x3004_0000;
        const BLOCK_16_GOBS: u32 = 4 << 4;
        let fixed_pair = |v: f64| {
            (
                v.floor() as i32 as u32,
                (v.fract() * 4_294_967_296.0) as u32,
            )
        };
        // Block heights of 16 GOBs, of 1 (a band every 8 rows) and pitch rows.
        let layouts = [(0, BLOCK_16_GOBS), (0, 0), (MEMORY_LAYOUT_PITCH, 0)];
        for ((step, origin, (w, h)), (memory_layout, block_size)) in [
            ((2.0, 1.0), (0.5, 0.0), (37, 45)),
            ((1.5, 0.75), (0.25, 0.1), (60, 70)),
        ]
        .into_iter()
        .flat_map(|blit| layouts.map(|layout| (blit, layout)))
        {
            let mut mem = Memory::new();
            mem.map_zero(SRC, 0x8_0000).unwrap();
            for i in 0..0x4_0000 / 4 {
                mem.write_u32(SRC + i * 4, i.wrapping_mul(0x9E37_79B9))
                    .unwrap();
            }
            let mut vmm = AddressSpace::new();
            let base = vmm.map(SRC, 0x8_0000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
            let mut host1x = Host1x::new();
            let mut stats = GpuStats::default();
            let mut engine = Engine2D::new();
            for (format, layout, block, pitch, width, height, offset, sw, sh, at) in [
                (
                    SET_SRC_FORMAT,
                    SET_SRC_MEMORY_LAYOUT,
                    SET_SRC_BLOCK_SIZE,
                    SET_SRC_PITCH,
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
                    SET_DST_PITCH,
                    SET_DST_WIDTH,
                    SET_DST_HEIGHT,
                    SET_DST_OFFSET,
                    w,
                    h,
                    base + u64::from(DST - SRC),
                ),
            ] {
                engine.regs.set(format, 0xD5);
                engine.regs.set(layout, memory_layout);
                engine.regs.set(block, block_size);
                engine.regs.set(pitch, sw * 4 + 12);
                engine.regs.set(width, sw);
                engine.regs.set(height, sh);
                set_iova(&mut engine, offset, at);
            }
            engine.regs.set(SET_OPERATION, OPERATION_SRC_COPY);
            engine.regs.set(SAMPLE_MODE, FILTER_BILINEAR << 4);
            engine.regs.set(DST_WIDTH, w);
            engine.regs.set(DST_HEIGHT, h);
            for (int_reg, frac_reg, value) in [
                (DU_DX_INT, DU_DX_FRAC, step.0),
                (DV_DY_INT, DV_DY_FRAC, step.1),
                (SRC_X0_INT, SRC_X0_FRAC, origin.0),
            ] {
                let (int_part, frac) = fixed_pair(value);
                engine.regs.set(int_reg, int_part);
                engine.regs.set(frac_reg, frac);
            }
            let (y0_int, y0_frac) = fixed_pair(origin.1);
            engine.regs.set(SRC_Y0_FRAC, y0_frac);
            let (src, dst) = (
                engine.surface(false).unwrap(),
                engine.surface(true).unwrap(),
            );
            let mut ctx = ExecCtx {
                mem: &mut mem,
                vmm: &vmm,
                as_id: 0,
                host1x: &mut host1x,
                stats: &mut stats,
                trace: false,
            };
            engine.write(SRC_Y0_INT, y0_int, &mut ctx).unwrap();
            let as_engine = |v: f64| {
                let (int_part, frac) = fixed_pair(v);
                fixed(int_part, frac)
            };
            for y in 0..h {
                for x in 0..w {
                    let u = as_engine(origin.0) + as_engine(step.0) * x as f64;
                    let v = as_engine(origin.1) + as_engine(step.1) * y as f64;
                    let want = dst
                        .format
                        .encode(src.sample_bilinear(u, v, &ctx).unwrap())
                        .unwrap();
                    let got = ctx
                        .read_pixel(dst.addr + dst.offset(x, y) as u64, 4)
                        .unwrap();
                    assert_eq!(got, want, "pixel ({x}, {y}) of a {step:?} step, layout {memory_layout} block {block_size:#x}");
                }
            }
        }
    }

    /// Reused texels must be invalidated by a store or a new mapping.
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
        let mut blit = |mem: &Memory| {
            let mut ctx = ExecCtx {
                mem,
                vmm: &vmm,
                as_id: 0,
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

    /// A halving copy larger than one tile both ways, ragged edges included.
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
            as_id: 0,
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
