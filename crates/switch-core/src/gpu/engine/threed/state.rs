//! Register state decoded for the renderers.

use super::{
    depth_format_layout, BlendTarget, CullState, DepthState, DepthTarget, Engine3D, ProgramBinding,
    RenderTarget, ScissorRect, ShaderStage, VertexArray, VertexAttrib, ViewportTransform,
    WindowOrigin, BLEND_CONSTANT, BLEND_EQUATION_ALPHA, BLEND_EQUATION_RGB, BLEND_FUNC_DST_ALPHA,
    BLEND_FUNC_DST_RGB, BLEND_FUNC_SRC_ALPHA, BLEND_FUNC_SRC_RGB, COLOR_BLEND_ENABLE, COLOR_MASK,
    COLOR_MASK_COMMON, COLOR_TARGETS, DEPTH_TARGET_ADDR, DEPTH_TARGET_FORMAT,
    DEPTH_TARGET_HORIZONTAL, DEPTH_TARGET_TILE_MODE, DEPTH_TARGET_VERTICAL, DEPTH_TEST_ENABLE,
    DEPTH_TEST_FUNC, DEPTH_WRITE_ENABLE, INDEPENDENT_BLEND, INDEPENDENT_BLEND_ENABLE,
    INDEPENDENT_BLEND_STRIDE, INDEX_ARRAY_START, MULTISAMPLE_CONTROL, MULTISAMPLE_ENABLE,
    MULTISAMPLE_MODE, MULTISAMPLE_SAMPLE_LOCATIONS, MULTISAMPLE_SAMPLE_MASK, OGL_SET_CULL,
    OGL_SET_CULL_FACE, OGL_SET_FRONT_FACE, RENDER_TARGET_BASE, RENDER_TARGET_CONTROL,
    RENDER_TARGET_STRIDE, SCISSOR_BASE, SCREEN_SCISSOR_HORIZONTAL, SCREEN_SCISSOR_VERTICAL,
    SET_PROGRAM, SET_PROGRAM_REGION, SET_PROGRAM_STRIDE, SET_TEX_HEADER_POOL, SET_TEX_SAMPLER_POOL,
    TEX_CB_INDEX, VERTEX_ARRAY, VERTEX_ARRAY_LIMIT, VERTEX_ARRAY_PER_INSTANCE, VERTEX_ARRAY_STRIDE,
    VERTEX_ATTRIB_STATE, VIEWPORT_BASE, VIEWPORT_TRANSFORM_BASE, WINDOW_ORIGIN,
};
use crate::gpu::engine::field;
use crate::gpu::exec::ExecCtx;
use crate::gpu::renderer::Renderer;
use crate::gpu::surface::{ColorFormat, Layout, SampleGrid, MAX_SAMPLES};
use crate::{Error, Result};

impl Engine3D {
    /// Resolve `stage`'s bound program, if enabled.
    pub fn program(&self, stage: ShaderStage) -> Option<ProgramBinding> {
        let base = SET_PROGRAM + stage.index() * SET_PROGRAM_STRIDE;
        // VertexB is always active; drivers need not set its enable bit.
        if stage != ShaderStage::VertexB && field(self.regs.get(base), 0, 0) == 0 {
            return None;
        }
        let offset = self.regs.get(base + 1);
        let num_registers = self.regs.get(base + 3);
        let addr = self
            .regs
            .iova(SET_PROGRAM_REGION)
            .wrapping_add(u64::from(offset));
        Some(ProgramBinding {
            addr,
            num_registers,
        })
    }

    /// Resolve `VertexAttribState[i]`.
    pub fn vertex_attrib(&self, i: u32) -> VertexAttrib {
        let raw = self.regs.get(VERTEX_ATTRIB_STATE + i);
        VertexAttrib {
            buffer_id: field(raw, 0, 4),
            is_fixed: field(raw, 6, 6) != 0,
            offset: field(raw, 7, 20),
            size: field(raw, 21, 26),
            ty: field(raw, 27, 29),
            is_bgra: field(raw, 31, 31) != 0,
        }
    }

    /// Resolve `VertexArray[i]` plus its `VertexArrayLimit[i]`.
    pub fn vertex_array(&self, i: u32) -> VertexArray {
        let base = VERTEX_ARRAY + i * VERTEX_ARRAY_STRIDE;
        let config = self.regs.get(base);
        VertexArray {
            enabled: field(config, 12, 12) != 0,
            stride: field(config, 0, 11),
            start: self.regs.iova(base + 1),
            limit: self.regs.iova(VERTEX_ARRAY_LIMIT + i * 2),
            // A stream's frequency only applies when it is per-instance.
            divisor: if self.regs.get(VERTEX_ARRAY_PER_INSTANCE + i) & 1 != 0 {
                self.regs.get(base + 3)
            } else {
                0
            },
        }
    }

    /// Base of the `SetTexHeaderPool` descriptor pool.
    pub fn tex_header_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_HEADER_POOL)
    }

    /// Base of the `SetTexSamplerPool` descriptor pool.
    pub fn tex_sampler_pool(&self) -> u64 {
        self.regs.iova(SET_TEX_SAMPLER_POOL)
    }

    /// The `(addr, size)` bound to `stage`'s constant bank `bank`.
    pub fn bound_constbuf(&self, stage: ShaderStage, bank: u32) -> Option<(u64, u32)> {
        *self
            .bound_constbufs
            .get(stage.bind_slot() as usize)?
            .get(bank as usize)?
    }

    /// Resolve `IndependentBlend[index]` plus its `ColorBlendEnable[index]` bit.
    pub fn blend_target(&self, index: u32) -> BlendTarget {
        let enabled = self.regs.get(COLOR_BLEND_ENABLE + index) != 0;
        if self.independent_blend_enabled() {
            let base = INDEPENDENT_BLEND + index * INDEPENDENT_BLEND_STRIDE;
            BlendTarget {
                enabled,
                equation_rgb: self.regs.get(base + 1),
                func_rgb_src: self.regs.get(base + 2),
                func_rgb_dst: self.regs.get(base + 3),
                equation_alpha: self.regs.get(base + 4),
                func_alpha_src: self.regs.get(base + 5),
                func_alpha_dst: self.regs.get(base + 6),
            }
        } else {
            BlendTarget {
                enabled,
                equation_rgb: self.regs.get(BLEND_EQUATION_RGB),
                func_rgb_src: self.regs.get(BLEND_FUNC_SRC_RGB),
                func_rgb_dst: self.regs.get(BLEND_FUNC_DST_RGB),
                equation_alpha: self.regs.get(BLEND_EQUATION_ALPHA),
                func_alpha_src: self.regs.get(BLEND_FUNC_SRC_ALPHA),
                func_alpha_dst: self.regs.get(BLEND_FUNC_DST_ALPHA),
            }
        }
    }

    /// Whether blend targets are independent; otherwise all use `blend_target(0)`.
    pub fn independent_blend_enabled(&self) -> bool {
        self.regs.get(INDEPENDENT_BLEND_ENABLE) != 0
    }

    pub fn blend_constant(&self) -> [f32; 4] {
        [
            self.regs.float(BLEND_CONSTANT),
            self.regs.float(BLEND_CONSTANT + 1),
            self.regs.float(BLEND_CONSTANT + 2),
            self.regs.float(BLEND_CONSTANT + 3),
        ]
    }

    pub fn depth_state(&self) -> DepthState {
        DepthState {
            test_enabled: self.regs.get(DEPTH_TEST_ENABLE) != 0,
            write_enabled: self.regs.get(DEPTH_WRITE_ENABLE) != 0,
            func: self.regs.get(DEPTH_TEST_FUNC),
        }
    }

    /// Resolve the depth/stencil render target from the register file.
    pub fn depth_target(&self) -> Result<Option<DepthTarget>> {
        let addr = self.regs.iova(DEPTH_TARGET_ADDR);
        let raw_format = self.regs.get(DEPTH_TARGET_FORMAT);
        if addr == 0 || raw_format == 0 {
            return Ok(None);
        }
        let format = depth_format_layout(raw_format)?;
        let tile_mode = self.regs.get(DEPTH_TARGET_TILE_MODE);
        Ok(Some(DepthTarget {
            addr,
            width: self.regs.get(DEPTH_TARGET_HORIZONTAL),
            height: self.regs.get(DEPTH_TARGET_VERTICAL),
            layout: Layout::BlockLinear {
                block_height_gobs: 1 << field(tile_mode, 4, 7),
            },
            format,
        }))
    }

    /// Viewport 0's `(x, y, width, height)` in pixels.
    pub fn viewport(&self) -> (f32, f32, f32, f32) {
        (
            self.regs.field(VIEWPORT_BASE, 0, 15) as f32,
            self.regs.field(VIEWPORT_BASE + 1, 0, 15) as f32,
            self.regs.field(VIEWPORT_BASE, 16, 31) as f32,
            self.regs.field(VIEWPORT_BASE + 1, 16, 31) as f32,
        )
    }

    /// What `SET_WINDOW_ORIGIN_MODE` says about y and about winding.
    pub fn window_origin(&self) -> WindowOrigin {
        WindowOrigin {
            lower_left: self.regs.bit(WINDOW_ORIGIN, 0),
            flip_y: self.regs.bit(WINDOW_ORIGIN, 4),
        }
    }

    /// The height a bottom-left window origin is measured against (the surface clip's).
    pub(crate) fn surface_clip_height(&self) -> u32 {
        self.regs.field(SCREEN_SCISSOR_VERTICAL, 16, 31)
    }

    /// Reflect a viewport about the surface clip when the window origin is at the bottom.
    fn flip_window_y(&self, mut vt: ViewportTransform) -> ViewportTransform {
        if !self.window_origin().lower_left {
            return vt;
        }
        // Register: top = translate - scale, height = 2 scale; flipped: top + clip_height, -height.
        vt.translate[1] += self.surface_clip_height() as f32 - 2.0 * vt.scale[1];
        vt.scale[1] = -vt.scale[1];
        vt
    }

    pub fn viewport_transform(&self) -> ViewportTransform {
        let f = |i: u32| f32::from_bits(self.regs.get(VIEWPORT_TRANSFORM_BASE + i));
        let scale = [f(0), f(1), f(2)];
        if scale[0] != 0.0 || scale[1] != 0.0 {
            return self.flip_window_y(ViewportTransform {
                scale,
                translate: [f(3), f(4), f(5)],
            });
        }
        // Unwritten transform: fall back to the viewport rectangle with the window-system flip.
        let (vx, vy, vw, vh) = self.viewport();
        self.flip_window_y(ViewportTransform {
            scale: [vw / 2.0, -vh / 2.0, 0.5],
            translate: [vx + vw / 2.0, vy + vh / 2.0, 0.5],
        })
    }

    /// Constant bank a `texs` immediate indexes for its texture handle (`TexCbIndex`).
    pub fn tex_cb_index(&self) -> u8 {
        self.regs.field(TEX_CB_INDEX, 0, 4) as u8
    }

    /// Replace the backend this engine draws and clears through.
    pub fn set_renderer(&mut self, renderer: Box<dyn Renderer>) {
        self.renderer = renderer;
    }

    /// Exchange this engine's backend with the caller's.
    pub fn swap_renderer(&mut self, other: &mut Box<dyn Renderer>) {
        std::mem::swap(&mut self.renderer, other);
    }

    /// See [`Renderer::report_json`].
    pub fn renderer_report(&self) -> String {
        self.renderer.report_json()
    }

    /// See [`Renderer::lost`].
    pub fn renderer_lost(&self) -> bool {
        self.renderer.lost()
    }

    /// Tell the backend something outside it is about to read a render target.
    pub fn flush_renderer(&mut self, ctx: &mut ExecCtx) -> Result<crate::gpu::renderer::Flush> {
        self.with_renderer(ctx, |renderer, _, ctx| renderer.flush(ctx))
    }

    /// `gl_InstanceID` for [`Engine3D::last_draw`].
    pub fn instance_id(&self) -> u32 {
        self.instance_id
    }

    /// Set the current instance, for tests.
    pub fn set_instance_id(&mut self, instance_id: u32) {
        self.instance_id = instance_id;
    }

    pub fn index_array_start(&self) -> u64 {
        self.regs.iova(INDEX_ARRAY_START)
    }

    /// The scissor's y bounds in window coordinates, flipped for a bottom-left origin.
    pub(super) fn scissor_y(&self) -> (u32, u32) {
        let y0 = self.regs.field(SCISSOR_BASE + 2, 0, 15);
        let y1 = self.regs.field(SCISSOR_BASE + 2, 16, 31);
        if self.window_origin().lower_left {
            let height = self.surface_clip_height();
            (height.saturating_sub(y1), height.saturating_sub(y0))
        } else {
            (y0, y1)
        }
    }

    pub fn apply_scissor(&self, rect: ScissorRect) -> ScissorRect {
        let mut out = rect;
        let screen_w = self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 16, 31);
        let screen_h = self.regs.field(SCREEN_SCISSOR_VERTICAL, 16, 31);
        if screen_w != 0 && screen_h != 0 {
            let x0 = self.regs.field(SCREEN_SCISSOR_HORIZONTAL, 0, 15);
            let y0 = self.regs.field(SCREEN_SCISSOR_VERTICAL, 0, 15);
            out.x0 = out.x0.max(x0);
            out.y0 = out.y0.max(y0);
            out.x1 = out.x1.min(x0 + screen_w);
            out.y1 = out.y1.min(y0 + screen_h);
        }
        if self.regs.get(SCISSOR_BASE) != 0 {
            let (y0, y1) = self.scissor_y();
            out.x0 = out.x0.max(self.regs.field(SCISSOR_BASE + 1, 0, 15));
            out.x1 = out.x1.min(self.regs.field(SCISSOR_BASE + 1, 16, 31));
            out.y0 = out.y0.max(y0);
            out.y1 = out.y1.min(y1);
        }
        ScissorRect {
            x0: out.x0,
            y0: out.y0,
            x1: out.x1.max(out.x0),
            y1: out.y1.max(out.y0),
        }
    }

    /// Face culling from `OGL_SET_CULL`/`_FRONT_FACE`/`_CULL_FACE`.
    pub fn cull_state(&self) -> CullState {
        let face = self.regs.get(OGL_SET_CULL_FACE);
        CullState {
            enabled: field(self.regs.get(OGL_SET_CULL), 0, 0) != 0,
            // `flip_y` swaps the front winding independently of the viewport.
            front_ccw: (self.regs.get(OGL_SET_FRONT_FACE) != 0x900) != self.window_origin().flip_y,
            cull_front: face == 0x404 || face == 0x408,
            cull_back: face == 0x405 || face == 0x408,
        }
    }

    /// Sample layout of the bound surfaces: `MsaaMode` sets the grid, `AntiAliasEnable` only coverage.
    pub fn sample_grid(&self) -> Result<SampleGrid> {
        let mut locations = [0u8; MAX_SAMPLES];
        for (i, byte) in locations.iter_mut().enumerate() {
            let word = self.regs.get(MULTISAMPLE_SAMPLE_LOCATIONS + (i / 4) as u32);
            *byte = (word >> (8 * (i % 4))) as u8;
        }
        let grid = SampleGrid::new(self.regs.get(MULTISAMPLE_MODE), &locations)?;
        if !self.multisample_enabled() {
            return Ok(grid.per_pixel_coverage());
        }
        Ok(grid)
    }

    /// Whether coverage is tested per sample (`AntiAliasEnable`).
    pub fn multisample_enabled(&self) -> bool {
        self.regs.get(MULTISAMPLE_ENABLE) != 0
    }

    /// Which samples a draw may write; an unwritten (zero) mask means all.
    pub fn sample_mask(&self) -> u32 {
        match self.regs.get(MULTISAMPLE_SAMPLE_MASK) {
            0 => u32::MAX,
            mask => mask,
        }
    }

    /// Whether a fragment's alpha turns into coverage (`MultisampleControl`).
    pub fn alpha_to_coverage(&self) -> bool {
        self.regs.bit(MULTISAMPLE_CONTROL, 0)
    }

    /// The physical slot `RenderTargetControl` maps colour target `index` onto.
    pub fn render_target_slot(&self, index: u32) -> u32 {
        let count = self.regs.field(RENDER_TARGET_CONTROL, 0, 3);
        if index < count {
            self.regs
                .field(RENDER_TARGET_CONTROL, 4 + index * 3, 6 + index * 3)
        } else {
            index
        }
    }

    /// Which channels of colour target `index` a draw may write.
    pub fn color_mask(&self, index: u32) -> [bool; 4] {
        let slot = if self.regs.get(COLOR_MASK_COMMON) != 0 {
            0
        } else {
            index
        };
        let raw = self.regs.get(COLOR_MASK + slot.min(COLOR_TARGETS - 1));
        [
            field(raw, 0, 0) != 0,
            field(raw, 4, 4) != 0,
            field(raw, 8, 8) != 0,
            field(raw, 12, 12) != 0,
        ]
    }

    /// Resolve colour render target `index` from the register file.
    pub fn render_target(&self, index: u32) -> Result<Option<RenderTarget>> {
        let base = RENDER_TARGET_BASE + index * RENDER_TARGET_STRIDE;
        let addr = self.regs.iova(base);
        let raw_format = self.regs.get(base + 4);
        if addr == 0 || raw_format == 0 {
            return Ok(None);
        }
        // A depth format bound as a colour target: report nothing bound.
        if depth_format_layout(raw_format).is_ok() {
            return Ok(None);
        }
        let format = ColorFormat::from_raw(raw_format)?;
        let tile_mode = self.regs.get(base + 5);
        let horizontal = self.regs.get(base + 2);
        let vertical = self.regs.get(base + 3);
        let is_linear = tile_mode >> 12 & 1 != 0;
        let (layout, width) = if is_linear {
            (
                Layout::Pitch { pitch: horizontal },
                horizontal / format.bytes_per_pixel.max(1),
            )
        } else {
            let block_width_gobs = field(tile_mode, 0, 3);
            if block_width_gobs != 0 {
                return Err(Error::Gpu(format!(
                    "3d: render target {} uses a {}-GOB-wide block, which Maxwell does not have",
                    index,
                    1 << block_width_gobs
                )));
            }
            let block_height_gobs = 1 << field(tile_mode, 4, 7);
            (Layout::BlockLinear { block_height_gobs }, horizontal)
        };
        Ok(Some(RenderTarget {
            addr,
            width,
            height: vertical,
            format,
            layout,
            layers: self.regs.field(base + 6, 0, 15).max(1),
            layer_stride: self.regs.get(base + 7),
        }))
    }
}
