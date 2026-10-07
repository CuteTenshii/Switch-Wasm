//! Clears, and the renderer they reach.

use super::{
    depth_format_layout, DepthFill, Engine3D, ScissorRect, CLEAR_BUFFER_FLAGS, CLEAR_COLOR,
    CLEAR_DEPTH, CLEAR_STENCIL, DEPTH_TARGET_ADDR, DEPTH_TARGET_FORMAT, DEPTH_TARGET_HORIZONTAL,
    DEPTH_TARGET_TILE_MODE, DEPTH_TARGET_VERTICAL, SCISSOR_BASE, SCREEN_SCISSOR_HORIZONTAL,
    SCREEN_SCISSOR_VERTICAL, VIEWPORT_BASE,
};
use crate::gpu::engine::field;
use crate::gpu::exec::ExecCtx;
use crate::gpu::renderer::{Renderer, Software};
use crate::gpu::surface::{Layout, GOB_HEIGHT, GOB_SIZE, GOB_WIDTH};
use crate::Result;

impl Engine3D {
    /// The pixel rectangle a clear covers (screen scissor, optionally scissor and viewport).
    fn clear_rect(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let rect = self.clear_rectangle(width, height);
        (rect.x0, rect.y0, rect.x1, rect.y1)
    }

    /// [`Engine3D::clear_rect`] as a [`ScissorRect`].
    pub fn clear_rectangle(&self, width: u32, height: u32) -> ScissorRect {
        let (x0, y0, x1, y1) = self.clear_rect_bounds(width, height);
        ScissorRect { x0, y0, x1, y1 }
    }

    /// The colour a `ClearBuffers` writes, as linear floats.
    pub fn clear_color_value(&self) -> [f32; 4] {
        [
            self.regs.float(CLEAR_COLOR),
            self.regs.float(CLEAR_COLOR + 1),
            self.regs.float(CLEAR_COLOR + 2),
            self.regs.float(CLEAR_COLOR + 3),
        ]
    }

    /// The depth a `ClearBuffers` writes, in `0.0..=1.0`.
    pub fn clear_depth_value(&self) -> f32 {
        self.regs.float(CLEAR_DEPTH).clamp(0.0, 1.0)
    }

    /// The stencil byte a `ClearBuffers` writes.
    pub fn clear_stencil_value(&self) -> u32 {
        self.regs.get(CLEAR_STENCIL) & 0xFF
    }

    fn clear_rect_bounds(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let mut x0 = self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 0, 15);
        let mut y0 = self.regs.field(SCREEN_SCISSOR_VERTICAL, 0, 15);
        let mut x1 = x0 + self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 16, 31);
        let mut y1 = y0 + self.regs.field(SCREEN_SCISSOR_VERTICAL, 16, 31);

        let flags = self.regs.get(CLEAR_BUFFER_FLAGS);
        if field(flags, 8, 8) != 0 && self.regs.get(SCISSOR_BASE) != 0 {
            let (sy0, sy1) = self.scissor_y();
            x0 = x0.max(self.regs.field(SCISSOR_BASE + 1, 0, 15));
            x1 = x1.min(self.regs.field(SCISSOR_BASE + 1, 16, 31));
            y0 = y0.max(sy0);
            y1 = y1.min(sy1);
        }
        if field(flags, 12, 12) != 0 {
            let vx = self.regs.field(VIEWPORT_BASE, 0, 15);
            let vy = self.regs.field(VIEWPORT_BASE + 1, 0, 15);
            x0 = x0.max(vx);
            x1 = x1.min(vx + self.regs.field(VIEWPORT_BASE, 16, 31));
            y0 = y0.max(vy);
            y1 = y1.min(vy + self.regs.field(VIEWPORT_BASE + 1, 16, 31));
        }
        // An unprogrammed screen scissor means "the whole target".
        if x1 <= x0 {
            x0 = 0;
            x1 = width;
        }
        if y1 <= y0 {
            y0 = 0;
            y1 = height;
        }
        (x0, y0, x1.min(width), y1.min(height))
    }

    /// Run `f` against the bound renderer, taken out of the engine for the call.
    pub(super) fn with_renderer<T>(
        &mut self,
        ctx: &mut ExecCtx,
        f: impl FnOnce(&mut dyn Renderer, &Engine3D, &mut ExecCtx) -> T,
    ) -> T {
        let mut renderer = std::mem::replace(&mut self.renderer, Box::new(Software));
        let out = f(renderer.as_mut(), self, ctx);
        self.renderer = renderer;
        out
    }

    pub(super) fn clear_buffers(&mut self, arg: u32, ctx: &mut ExecCtx) -> Result<()> {
        ctx.stats.clears += 1;
        let clear_depth = field(arg, 0, 0) != 0;
        let clear_stencil = field(arg, 1, 1) != 0;
        let channels = [
            field(arg, 2, 2) != 0,
            field(arg, 3, 3) != 0,
            field(arg, 4, 4) != 0,
            field(arg, 5, 5) != 0,
        ];
        let target = field(arg, 6, 9);
        let layer = field(arg, 10, 20);

        let trace_clear = ctx.trace || crate::trace::enabled(crate::trace::Trace::Draw);
        if channels.iter().any(|&c| c) {
            if let Ok(Some(rt)) = self.render_target(self.render_target_slot(target)) {
                let cpu = ctx.span(rt.addr, 4);
                self.activity.note(
                    crate::gpu::activity::Kind::Clear,
                    rt.addr,
                    0,
                    0,
                    false,
                    || {
                        format!(
                            "colour {}",
                            crate::gpu::activity::surface_text(
                                rt.addr,
                                cpu,
                                rt.width,
                                rt.height,
                                rt.format.raw
                            )
                        )
                    },
                );
            }
            if trace_clear {
                let colour = [
                    self.regs.float(CLEAR_COLOR),
                    self.regs.float(CLEAR_COLOR + 1),
                    self.regs.float(CLEAR_COLOR + 2),
                    self.regs.float(CLEAR_COLOR + 3),
                ];
                match self.render_target(target) {
                    Ok(Some(rt)) => crate::traceln!(
                        "[gpu] clear target={target} addr={:#x} {}x{} texels colour={colour:?}",
                        rt.addr,
                        rt.width,
                        rt.height
                    ),
                    other => crate::traceln!("[gpu] clear target={target} -> {other:x?}"),
                }
            }
            self.with_renderer(ctx, |renderer, engine, ctx| {
                renderer.clear_color(engine, ctx, target, layer, channels)
            })?;
        }
        if clear_depth || clear_stencil {
            if let Ok(Some(zt)) = self.depth_target() {
                let cpu = ctx.span(zt.addr, 4);
                self.activity.note(crate::gpu::activity::Kind::Clear, zt.addr, 1, 0, false, || {
                    let at = match cpu {
                        Some(cpu) => format!(" (cpu {cpu:#x})"),
                        None => String::new(),
                    };
                    format!(
                        "depth/stencil {:#x}{at} {}x{} (depth {clear_depth}, stencil {clear_stencil})",
                        zt.addr, zt.width, zt.height
                    )
                });
            }
            if trace_clear {
                let depth = self.regs.float(CLEAR_DEPTH);
                let stencil = self.regs.get(CLEAR_STENCIL) & 0xFF;
                match self.depth_target() {
                    Ok(Some(zt)) => crate::traceln!(
                        "[gpu] clear depth={clear_depth}/{depth} stencil={clear_stencil}/{stencil} \
                         addr={:#x} {}x{} texels",
                        zt.addr,
                        zt.width,
                        zt.height
                    ),
                    other => crate::traceln!(
                        "[gpu] clear depth={clear_depth} stencil={clear_stencil} -> {other:x?}"
                    ),
                }
            }
            self.with_renderer(ctx, |renderer, engine, ctx| {
                renderer.clear_depth_stencil(engine, ctx, clear_depth, clear_stencil)
            })?;
        }
        Ok(())
    }

    pub(crate) fn clear_color(
        &self,
        target: u32,
        layer: u32,
        channels: [bool; 4],
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let slot = self.render_target_slot(target);
        let rt = match self.render_target(slot)? {
            Some(rt) => rt,
            None => return Ok(()),
        };
        let color = [
            self.regs.float(CLEAR_COLOR),
            self.regs.float(CLEAR_COLOR + 1),
            self.regs.float(CLEAR_COLOR + 2),
            self.regs.float(CLEAR_COLOR + 3),
        ];
        let raw = rt.format.encode(color)?;
        let bpp = rt.format.bytes_per_pixel;
        let all_channels = channels.iter().all(|&c| c);
        let base = rt.addr + (layer as u64) * (rt.layer_stride as u64) * 4;
        let grid = self.sample_grid()?;
        let (width, height) = grid.pixels(rt.width, rt.height);
        let (x0, y0, x1, y1) = self.clear_rect(width, height);
        if ctx.trace {
            crate::traceln!(
                "[gpu] clear color rt{} {:#x} {width}x{height}px {}x{} samples fmt={:#x} \
                 rect=({x0},{y0})..({x1},{y1}) rgba={color:?}",
                slot,
                rt.addr,
                grid.samples_x,
                grid.samples_y,
                rt.format.raw
            );
        }
        // Clearing whole pixels clears their texel rectangle, so fill runs of texels.
        if all_channels {
            let (tx0, ty0) = (x0 * grid.samples_x, y0 * grid.samples_y);
            let (tx1, ty1) = (x1 * grid.samples_x, y1 * grid.samples_y);
            // A GOB is 512 contiguous bytes, so a fully covered GOB is one fill.
            let gob_texels = GOB_WIDTH / bpp;
            let whole_gobs = matches!(rt.layout, Layout::BlockLinear { .. }) && gob_texels > 0;
            let mut ty = ty0;
            while ty < ty1 {
                let gob_row = ty - ty % GOB_HEIGHT;
                let row_whole = whole_gobs && ty == gob_row && ty + GOB_HEIGHT <= ty1;
                let mut tx = tx0;
                while tx < tx1 {
                    let gob_col = tx - tx % gob_texels;
                    if row_whole && tx == gob_col && tx + gob_texels <= tx1 {
                        let (offset, _) = rt.texel_run(tx, ty);
                        ctx.fill_pixels(base + offset as u64, bpp, raw, GOB_SIZE / bpp)?;
                        tx += gob_texels;
                        continue;
                    }
                    let (offset, run) = rt.texel_run(tx, ty);
                    let count = run.min(tx1 - tx);
                    ctx.fill_pixels(base + offset as u64, bpp, raw, count)?;
                    tx += count;
                }
                ty += if row_whole { GOB_HEIGHT } else { 1 };
            }
            return Ok(());
        }

        for y in y0..y1 {
            for x in x0..x1 {
                for sample in 0..grid.count() {
                    let (tx, ty) = grid.texel(x, y, sample);
                    let va = base + rt.texel_offset(tx, ty) as u64;
                    {
                        let old = rt.format.decode(ctx.read_pixel(va, bpp)?)?;
                        let mut merged = old;
                        for (i, &enabled) in channels.iter().enumerate() {
                            if enabled {
                                merged[i] = color[i];
                            }
                        }
                        ctx.write_pixel(va, bpp, rt.format.encode(merged)?)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn clear_depth_stencil(
        &self,
        clear_depth: bool,
        clear_stencil: bool,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let addr = self.regs.iova(DEPTH_TARGET_ADDR);
        let raw_format = self.regs.get(DEPTH_TARGET_FORMAT);
        if addr == 0 || raw_format == 0 {
            return Ok(());
        }
        let format = depth_format_layout(raw_format)?;
        let bytes = format.bytes;
        let texels_x = self.regs.get(DEPTH_TARGET_HORIZONTAL);
        let texels_y = self.regs.get(DEPTH_TARGET_VERTICAL);
        let tile_mode = self.regs.get(DEPTH_TARGET_TILE_MODE);
        let layout = Layout::BlockLinear {
            block_height_gobs: 1 << field(tile_mode, 4, 7),
        };
        let depth = self.regs.float(CLEAR_DEPTH).clamp(0.0, 1.0);
        let stencil = self.regs.get(CLEAR_STENCIL) & 0xFF;
        let grid = self.sample_grid()?;
        let (width, height) = grid.pixels(texels_x, texels_y);
        let (x0, y0, x1, y1) = self.clear_rect(width, height);
        let width_bytes = texels_x * bytes;
        if ctx.trace {
            crate::traceln!(
                "[gpu] clear depth {addr:#x} {width}x{height}px {}x{} samples fmt={raw_format:#x} \
                 rect=({x0},{y0})..({x1},{y1}) depth={clear_depth}/{depth} stencil={clear_stencil}/{stencil}",
                grid.samples_x, grid.samples_y
            );
        }

        // The masked value this clear writes to every texel.
        let mut written = 0u128;
        let mut value = 0u128;
        if clear_depth {
            let mask = format.depth_mask();
            written |= mask;
            value |= format.encode_depth(depth) & mask;
        }
        if clear_stencil {
            if let Some(shift) = format.stencil_shift {
                written |= 0xFFu128 << shift;
                value |= u128::from(stencil) << shift;
            }
        }
        if written == 0 {
            return Ok(());
        }

        // Skip a repeat of the last clear over bytes nothing has written since.
        let stale = ctx.mem.take_fill_written();
        let fill = DepthFill {
            addr,
            span: 0,
            value,
            written,
            bytes,
            rect: (x0, y0, x1, y1),
            tile_mode,
            width_bytes,
        };
        if !stale {
            if let Some(last) = *self.depth_fill.borrow() {
                if (DepthFill { span: 0, ..last }) == fill {
                    ctx.stats.clears_elided += 1;
                    return Ok(());
                }
            }
        }
        // Invalidate the record until this clear has finished.
        *self.depth_fill.borrow_mut() = None;

        // Same GOB walk as `clear_color`; `span` grows to cover every byte touched.
        let mut span = 0u64;
        let (tx0, ty0) = (x0 * grid.samples_x, y0 * grid.samples_y);
        let (tx1, ty1) = (x1 * grid.samples_x, y1 * grid.samples_y);
        let gob_texels = GOB_WIDTH / bytes;
        let mut ty = ty0;
        while ty < ty1 {
            let gob_row = ty - ty % GOB_HEIGHT;
            let row_whole = gob_texels > 0 && ty == gob_row && ty + GOB_HEIGHT <= ty1;
            let mut tx = tx0;
            while tx < tx1 {
                let gob_col = if gob_texels > 0 {
                    tx - tx % gob_texels
                } else {
                    tx
                };
                if row_whole && tx == gob_col && tx + gob_texels <= tx1 {
                    let (offset, _) = layout.run_at(tx * bytes, ty, width_bytes);
                    let va = addr + offset as u64;
                    span = span.max(offset as u64 + u64::from(GOB_SIZE));
                    ctx.merge_pixels(va, bytes, value, written, GOB_SIZE / bytes)?;
                    tx += gob_texels;
                    continue;
                }
                let (offset, run) = layout.run_at(tx * bytes, ty, width_bytes);
                let count = (run / bytes).max(1).min(tx1 - tx);
                span = span.max(offset as u64 + u64::from(count) * u64::from(bytes));
                ctx.merge_pixels(addr + offset as u64, bytes, value, written, count)?;
                tx += count;
            }
            ty += if row_whole { GOB_HEIGHT } else { 1 };
        }

        // Arm the watch after the walk so its own stores are not reported.
        if let Some(cpu) = ctx.span(addr, span) {
            if let Ok(len) = u32::try_from(span) {
                ctx.mem.mark_fill_range(cpu, len);
                ctx.mem.take_fill_written();
                *self.depth_fill.borrow_mut() = Some(DepthFill { span: len, ..fill });
            }
        }
        Ok(())
    }
}
