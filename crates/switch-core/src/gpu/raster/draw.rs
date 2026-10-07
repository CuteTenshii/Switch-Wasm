//! The draw entry point: index fetch, culling, clipping and the pixel loop.

use super::fragment::{quad_pixel, shade_fragment, shade_quad, with_fragment_env, Fragments, QUAD};
use super::setup::TriangleSetup;
use super::vertex::{shade_vertex, to_screen, ShadedVertex, MAX_VERTEX_ATTRIBS, NUM_VARYINGS};
use super::{assemble, Bounds, Primitive, ScreenVertex};
use crate::gpu::engine::threed::{
    BlendTarget, CullState, Engine3D, ScissorRect, ShaderStage, VertexArray, VertexAttrib,
};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::interp::{Invocation, MemoryConstants, MemoryGlobal};
use crate::gpu::shader::{decode_program_from_memory, wgsl, Op, Program};
use crate::gpu::surface::MAX_SAMPLES;
use crate::{Error, Result};

/// Whether `cull` throws this triangle away, judged by its winding in window
/// space (y down) after the viewport transform, as Eden does.
pub(super) fn culls(cull: CullState, v: [ScreenVertex; 3]) -> bool {
    if !cull.enabled {
        return false;
    }
    let area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
    if area == 0.0 {
        return true;
    }
    let front = (area < 0.0) == cull.front_ccw;
    if front {
        cull.cull_front
    } else {
        cull.cull_back
    }
}

/// Read the `i`th index of an indexed draw out of the bound index buffer.
fn read_index(ctx: &ExecCtx, base: u64, format: u32, i: u32) -> Result<u32> {
    Ok(match format {
        0 => u32::from(ctx.vmm_read_u8(base + u64::from(i))?),
        1 => {
            let at = base + u64::from(i) * 2;
            u32::from(ctx.vmm_read_u8(at)?) | (u32::from(ctx.vmm_read_u8(at + 1)?) << 8)
        }
        2 => ctx.read_u32(base + u64::from(i) * 4)?,
        other => {
            return Err(Error::Gpu(format!("raster: unknown index format {other}")));
        }
    })
}

/// A vertex after clipping: clip-space position plus varyings.
#[derive(Debug, Clone, Copy)]
pub(super) struct ClipVertex {
    pub(super) clip: [f32; 4],
    pub(super) varyings: [[f32; 4]; NUM_VARYINGS],
}

impl ClipVertex {
    fn lerp(a: &ClipVertex, b: &ClipVertex, t: f32) -> ClipVertex {
        let mut out = ClipVertex {
            clip: [0.0; 4],
            varyings: [[0.0; 4]; NUM_VARYINGS],
        };
        for c in 0..4 {
            out.clip[c] = a.clip[c] + (b.clip[c] - a.clip[c]) * t;
        }
        for slot in 0..NUM_VARYINGS {
            for c in 0..4 {
                out.varyings[slot][c] =
                    a.varyings[slot][c] + (b.varyings[slot][c] - a.varyings[slot][c]) * t;
            }
        }
        out
    }
}

/// Clip a triangle against the near plane (`w > epsilon`), so nothing
/// behind the eye is divided by.
pub(super) fn clip_near(tri: [ClipVertex; 3]) -> Vec<[ClipVertex; 3]> {
    /// Far enough from zero that the reciprocal stays finite.
    const NEAR_W: f32 = 1e-6;

    let inside: Vec<bool> = tri.iter().map(|v| v.clip[3] > NEAR_W).collect();
    let count = inside.iter().filter(|&&i| i).count();
    if count == 3 {
        return vec![tri];
    }
    if count == 0 {
        return Vec::new();
    }
    // Walk the edges, emitting kept vertices and the crossings between them.
    let mut poly: Vec<ClipVertex> = Vec::with_capacity(4);
    for i in 0..3 {
        let j = (i + 1) % 3;
        let (a, b) = (tri[i], tri[j]);
        if inside[i] {
            poly.push(a);
        }
        if inside[i] != inside[j] {
            let t = (NEAR_W - a.clip[3]) / (b.clip[3] - a.clip[3]);
            poly.push(ClipVertex::lerp(&a, &b, t));
        }
    }
    // A triangle clipped by one plane is a triangle or a quad; fan it.
    (1..poly.len().saturating_sub(1))
        .map(|i| [poly[0], poly[i], poly[i + 1]])
        .collect()
}

/// Run `engine.last_draw` into the bound render targets.
pub fn draw(engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
    let call = engine.last_draw;
    // Logical target 0, mapped through RenderTargetControl as a clear is.
    let rt = engine.render_target(engine.render_target_slot(0))?;

    let vs_binding = engine
        .program(ShaderStage::VertexB)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound vertex program".into()))?;
    let fs_binding = engine
        .program(ShaderStage::Fragment)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound fragment program".into()))?;

    let vs_program = decode_program_from_memory(&*ctx, vs_binding.addr, &|bank: u8| {
        engine.bound_constbuf(ShaderStage::VertexB, bank as u32)
    })?;
    let fs_program = decode_program_from_memory(&*ctx, fs_binding.addr, &|bank: u8| {
        engine.bound_constbuf(ShaderStage::Fragment, bank as u32)
    })?;
    // Only when a diagnostic asked; see `examples/shader_coverage.rs`.
    crate::gpu::shader::uses::note(wgsl::Stage::Vertex, vs_binding.addr, &vs_program);
    crate::gpu::shader::uses::note(wgsl::Stage::Fragment, fs_binding.addr, &fs_program);

    let attribs: Vec<VertexAttrib> = (0..MAX_VERTEX_ATTRIBS)
        .map(|i| engine.vertex_attrib(i))
        .collect();
    let arrays: Vec<VertexArray> = (0..MAX_VERTEX_ATTRIBS)
        .map(|i| engine.vertex_array(i))
        .collect();
    let viewport = engine.viewport_transform();
    let grid = engine.sample_grid()?;
    let sample_mask = engine.sample_mask();
    let alpha_to_coverage = engine.alpha_to_coverage();
    let depth = engine.depth_target()?;
    // Draw bounds are in pixels, target sizes in texels. A depth-only pass
    // takes its extent from whichever target is bound.
    let Some((target_width, target_height)) = crate::gpu::engine::threed::draw_extent(
        rt.map(|rt| (rt.width, rt.height)),
        depth.map(|dt| (dt.width, dt.height)),
        engine.depth_state(),
    ) else {
        return Err(Error::Gpu(
            "raster: draw with neither a colour nor a depth target".into(),
        ));
    };
    let (rt_width, rt_height) = grid.pixels(target_width, target_height);
    let clip = engine.apply_scissor(ScissorRect {
        x0: 0,
        y0: 0,
        x1: rt_width,
        y1: rt_height,
    });
    let bounds = Bounds {
        x0: clip.x0,
        y0: clip.y0,
        x1: clip.x1,
        y1: clip.y1,
    };
    let depth_state = engine.depth_state();
    let blend_target = engine.blend_target(0);
    let blend_constant = engine.blend_constant();
    // A mask with nothing set writes depth only.
    let color_mask = engine.color_mask(engine.render_target_slot(0));
    let writes_all_channels = color_mask == [true; 4];
    let writes_any_channel = color_mask.iter().any(|&channel| channel);
    let cull = engine.cull_state();

    let index_base = if call.indexed {
        engine.index_array_start()
    } else {
        0
    };
    let instance_id = engine.instance_id();
    let primitive = Primitive::from_raw(call.primitive)?;
    // Report point and line topologies rather than silently drawing nothing.
    if matches!(
        primitive,
        Primitive::Points | Primitive::Lines | Primitive::LineLoop | Primitive::LineStrip
    ) {
        return Err(Error::Gpu(format!(
            "raster: {primitive:?} is not rasterized"
        )));
    }
    let triangles = assemble(primitive, call.count);
    let mut tally = DrawTally::new(&fs_program);
    // Vertex shading is cached per index.
    let mut cache: crate::IdMap<u32, ShadedVertex> = crate::IdMap::default();
    // One fragment invocation for the whole draw, reset per pixel.
    let mut fragment = Invocation::new();
    let fragments = Fragments {
        grid,
        sample_mask,
        depth,
        depth_state,
        rt,
        blend_target,
        blend_constant,
        color_mask,
        writes_all_channels,
        writes_any_channel,
        alpha_to_coverage,
    };
    // Parsed TIC/TSC pairs and decoded blocks, shared by the draw.
    let descriptors = std::cell::RefCell::new(crate::IdMap::default());
    let blocks = std::cell::RefCell::new(crate::gpu::texture::BlockCache::default());
    let vs_const_cache = std::cell::RefCell::new(crate::gpu::shader::interp::ConstCache::default());
    let fs_const_cache = std::cell::RefCell::new(crate::gpu::shader::interp::ConstCache::default());
    // Lower both programs: resolve branch targets, fold bound constants.
    let vs_program = {
        let consts = MemoryConstants {
            ctx: &*ctx,
            bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::VertexB, bank as u32),
            cache: &vs_const_cache,
        };
        Compiled::with_constants(&vs_program, &consts)
    };
    let fs_program = {
        let consts = MemoryConstants {
            ctx: &*ctx,
            bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::Fragment, bank as u32),
            cache: &fs_const_cache,
        };
        Compiled::with_constants(&fs_program, &consts)
    };
    // Shaders with warp shuffles run in 2x2 quads.
    let mut quad: Option<Box<[Invocation; QUAD]>> = fs_program
        .ops()
        .iter()
        .any(|op| matches!(op, Op::Shfl { .. } | Op::Fswzadd { .. }))
        .then(|| Box::new(std::array::from_fn(|_| Invocation::new())));

    // `TRACE_PIPELINE=1`: this draw's fixed-function state, or why it is
    // not describable.
    if crate::trace::enabled(crate::trace::Trace::Pipeline) {
        // `Viewport::flip_y` and the window origin are separate claims.
        crate::traceln!(
            "[pipe] {:?} {:?} clip_height={}",
            engine.viewport_transform(),
            engine.window_origin(),
            engine.surface_clip_height()
        );
        match crate::gpu::pipeline::Pipeline::of(engine) {
            Ok(pipeline) => crate::traceln!("[pipe] {pipeline:?}"),
            Err(e) => crate::traceln!("[pipe] undescribable: {e}"),
        }
    }
    // `TRACE_UPLOAD=1`: the guest bytes this draw would upload to a device.
    if crate::trace::enabled(crate::trace::Trace::Upload) {
        match crate::gpu::pipeline::Pipeline::of(engine)
            .map_err(|e| Error::Gpu(format!("pipeline: {e}")))
            .and_then(|p| {
                // The two stages index different constant buffers per slot.
                let mut slots: Vec<(ShaderStage, crate::gpu::texture::TextureSlot)> = Vec::new();
                for (stage, program) in [
                    (ShaderStage::VertexB, &vs_program),
                    (ShaderStage::Fragment, &fs_program),
                ] {
                    if let Ok(translated) = crate::gpu::shader::wgsl::translate(program) {
                        slots.extend(
                            translated
                                .textures
                                .iter()
                                .map(|&(slot, _, _)| (stage, slot)),
                        );
                    }
                }
                crate::gpu::upload::Uploads::of(
                    engine,
                    &p,
                    &*ctx,
                    crate::gpu::upload::Banks::Bound,
                    &slots,
                )
            }) {
            Ok(uploads) => crate::traceln!(
                "[up] {} bytes: {} vertex ({}), {} index, {} constant ({} banks), \
                 {} texture ({})",
                uploads.len(),
                uploads.vertex.iter().map(|v| v.bytes.len()).sum::<usize>(),
                uploads.vertex.len(),
                uploads.index.as_ref().map_or(0, |i| i.bytes.len()),
                uploads
                    .constants
                    .iter()
                    .map(|c| c.bytes.len())
                    .sum::<usize>(),
                uploads.constants.len(),
                uploads
                    .textures
                    .iter()
                    .map(|t| t.bytes.len())
                    .sum::<usize>(),
                uploads.textures.len(),
            ),

            Err(e) => crate::traceln!("[up] cannot resolve: {e:?}"),
        }
        // The other direction: what a device-side backend must hand back.
        match crate::gpu::upload::Targets::of(engine) {
            Ok(targets) => crate::traceln!(
                "[rt] {} bytes back: colour {:?}, depth {:?}",
                targets.len(),
                targets
                    .color
                    .map(|t| (t.format, t.width, t.height, t.len())),
                targets
                    .depth
                    .map(|t| (t.format, t.width, t.height, t.len())),
            ),
            Err(e) => crate::traceln!("[rt] cannot resolve: {e:?}"),
        }
    }
    // `TRACE_CFG=1`: each shader's control flow, as a translator sees it.
    if crate::trace::enabled(crate::trace::Trace::Cfg) {
        for (stage, addr, program) in [
            ("vs", vs_binding.addr, &vs_program),
            ("fs", fs_binding.addr, &fs_program),
        ] {
            crate::traceln!(
                "[cfg] {stage}@{addr:#x} {}",
                crate::gpu::shader::cfg::Cfg::new(program).describe()
            );
        }
    }
    // `TRACE_WGSL=1`: whether each shader translates. `TRACE_WGSL=dir`
    // writes each module to `dir/<stage>_<addr>.wgsl` for `naga --validate`.
    if crate::trace::enabled(crate::trace::Trace::Wgsl) {
        // No directory in a browser, so only the summary there.
        let where_to = match std::env::var("TRACE_WGSL").unwrap_or_default() {
            dir if dir.is_empty() || dir == "1" => None,
            dir => Some(dir),
        };
        use crate::gpu::shader::wgsl::{self, Layout, Stage};
        for (name, stage, addr, program) in [
            ("vs", Stage::Vertex, vs_binding.addr, &vs_program),
            ("fs", Stage::Fragment, fs_binding.addr, &fs_program),
        ] {
            let translated = match wgsl::translate(program) {
                Ok(translated) => translated,
                Err(e) => {
                    crate::traceln!("[wgsl] {name}@{addr:#x} untranslated: {e}");
                    continue;
                }
            };
            let mut layout = Layout::of(&translated, stage);
            // Both corrections come from the viewport transform.
            if let Ok(pipeline) = crate::gpu::pipeline::Pipeline::of(engine) {
                layout.flip_y = pipeline.viewport.flip_y;
                layout.depth_minus_one_to_one = pipeline.viewport.depth_minus_one_to_one();
            }
            match wgsl::module(&translated, stage, &layout) {
                Ok(_) if where_to.is_none() => crate::traceln!(
                    "[wgsl] {name}@{addr:#x} {} regs, {} attribs, {} varyings, \
                     {} banks, {} textures",
                    translated.registers.len(),
                    layout.attributes.len(),
                    layout.varyings.len(),
                    layout.const_banks.len(),
                    layout.textures.len()
                ),
                Ok(module) => {
                    let dir = where_to.as_deref().unwrap_or_default();
                    let path = format!("{dir}/{name}_{addr:x}.wgsl");
                    match std::fs::write(&path, module) {
                        Ok(()) => crate::traceln!("[wgsl] {name}@{addr:#x} -> {path}"),
                        Err(e) => {
                            crate::traceln!("[wgsl] {name}@{addr:#x} cannot write {path}: {e}")
                        }
                    }
                }
                Err(e) => crate::traceln!("[wgsl] {name}@{addr:#x} no module: {e}"),
            }
        }
    }

    // What the draw's shaders store to global memory, landed after each
    // shading step: see `MemoryGlobal`.
    let global_stores = std::cell::RefCell::new(Vec::new());
    for tri in triangles {
        let mut shaded: Vec<ShadedVertex> = Vec::with_capacity(3);
        for &ordinal in &tri {
            let index = if call.indexed {
                read_index(&*ctx, index_base, call.index_format, call.first + ordinal)?
            } else {
                call.first + ordinal
            };
            if let Some(v) = cache.get(&index) {
                shaded.push(*v);
                continue;
            }
            let vs_consts = MemoryConstants {
                ctx: &*ctx,
                bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::VertexB, bank as u32),
                cache: &vs_const_cache,
            };
            let v = shade_vertex(
                &vs_program,
                &attribs,
                &arrays,
                (index, instance_id),
                &*ctx,
                &vs_consts,
                engine.window_origin().lower_left,
                &global_stores,
            )?;
            MemoryGlobal::land(ctx, &global_stores)?;
            cache.insert(index, v);
            shaded.push(v);
        }

        let unclipped = [
            ClipVertex {
                clip: shaded[0].clip,
                varyings: shaded[0].varyings,
            },
            ClipVertex {
                clip: shaded[1].clip,
                varyings: shaded[1].varyings,
            },
            ClipVertex {
                clip: shaded[2].clip,
                varyings: shaded[2].varyings,
            },
        ];
        for piece in clip_near(unclipped) {
            let shaded: [ShadedVertex; 3] = [
                ShadedVertex {
                    clip: piece[0].clip,
                    varyings: piece[0].varyings,
                },
                ShadedVertex {
                    clip: piece[1].clip,
                    varyings: piece[1].varyings,
                },
                ShadedVertex {
                    clip: piece[2].clip,
                    varyings: piece[2].varyings,
                },
            ];
            let projected: Vec<(ScreenVertex, f32, f32)> =
                shaded.iter().map(|v| to_screen(v.clip, viewport)).collect();
            let screen = [projected[0].0, projected[1].0, projected[2].0];
            let inv_w = [projected[0].1, projected[1].1, projected[2].1];
            let window_z = [projected[0].2, projected[1].2, projected[2].2];

            tally.triangles += 1;
            tally.geometry(screen);
            if culls(cull, screen) {
                tally.culled += 1;
                continue;
            }

            let Some(tri) = TriangleSetup::new(screen[0], screen[1], screen[2]) else {
                tally.degenerate += 1;
                continue;
            };
            let (min_x, max_x, min_y, max_y) = tri.bbox(bounds);
            if min_x >= max_x || min_y >= max_y {
                tally.outside(screen);
                continue;
            }
            let Some(quad) = quad.as_mut() else {
                let mut sample_z = [0.0f32; MAX_SAMPLES];
                for y in min_y..max_y {
                    for x in min_x..max_x {
                        let covered =
                            fragments.coverage(&tri, window_z, (x, y), &mut sample_z, ctx)?;
                        if covered == 0 {
                            tally.uncovered += 1;
                            continue;
                        }
                        tally.covered += 1;

                        let weights = tri.weights(x as f32 + 0.5, y as f32 + 0.5);
                        let color = with_fragment_env(
                            engine,
                            &*ctx,
                            &fs_const_cache,
                            &descriptors,
                            &blocks,
                            &global_stores,
                            |env| {
                                shade_fragment(
                                    &mut fragment,
                                    &fs_program,
                                    &shaded,
                                    inv_w,
                                    weights,
                                    env,
                                )
                            },
                        )?;
                        MemoryGlobal::land(ctx, &global_stores)?;
                        // `kil` skips the depth write too, so it waits for shading.
                        let Some(color) = color else {
                            tally.killed += 1;
                            continue;
                        };
                        fragments.write((x, y), covered, &sample_z, color, ctx, &mut tally)?;
                    }
                }
                continue;
            };

            // The quad walk, aligned to even pixels. Pixels outside the box
            // are shaded as helpers but never tested or written.
            let mut sample_z = [[0.0f32; MAX_SAMPLES]; QUAD];
            for y in ((min_y & !1)..max_y).step_by(2) {
                for x in ((min_x & !1)..max_x).step_by(2) {
                    let mut covered = [0u32; QUAD];
                    let mut weights = [[0.0f32; 3]; QUAD];
                    let mut any_covered = false;
                    for lane in 0..QUAD {
                        let (px, py) = quad_pixel(x, y, lane);
                        weights[lane] = tri.weights(px as f32 + 0.5, py as f32 + 0.5);
                        if px < min_x || px >= max_x || py < min_y || py >= max_y {
                            continue;
                        }
                        let mask = fragments.coverage(
                            &tri,
                            window_z,
                            (px, py),
                            &mut sample_z[lane],
                            ctx,
                        )?;
                        covered[lane] = mask;
                        if mask == 0 {
                            tally.uncovered += 1;
                        } else {
                            tally.covered += 1;
                            any_covered = true;
                        }
                    }
                    if !any_covered {
                        continue;
                    }

                    let colors = with_fragment_env(
                        engine,
                        &*ctx,
                        &fs_const_cache,
                        &descriptors,
                        &blocks,
                        &global_stores,
                        |env| shade_quad(quad, &fs_program, &shaded, inv_w, weights, env),
                    )?;
                    MemoryGlobal::land(ctx, &global_stores)?;
                    for lane in 0..QUAD {
                        if covered[lane] == 0 {
                            continue;
                        }
                        let Some(color) = colors[lane] else {
                            tally.killed += 1;
                            continue;
                        };
                        fragments.write(
                            quad_pixel(x, y, lane),
                            covered[lane],
                            &sample_z[lane],
                            color,
                            ctx,
                            &mut tally,
                        )?;
                    }
                }
            }
        }
    }
    tally.report(&call, primitive, blend_target, bounds);
    Ok(())
}

/// Per-draw fragment accounting for `TRACE_DRAW=1`: which stage fragments died at.
pub(super) struct DrawTally {
    enabled: bool,
    fs_len: usize,
    triangles: u64,
    culled: u64,
    degenerate: u64,
    /// Triangles whose box holds no pixel of the target (off it, or non-finite).
    outside: u64,
    nonfinite: u64,
    uncovered: u64,
    covered: u64,
    killed: u64,
    pub(super) alpha_killed: u64,
    written: u64,
    first_shaded: Option<[f32; 4]>,
    first_written: Option<[f32; 4]>,
    first_screen: Option<[ScreenVertex; 3]>,
}

impl DrawTally {
    fn new(fs_program: &Program) -> DrawTally {
        DrawTally {
            enabled: crate::trace::enabled(crate::trace::Trace::Draw),
            fs_len: fs_program.insns.len(),
            triangles: 0,
            culled: 0,
            degenerate: 0,
            outside: 0,
            nonfinite: 0,
            uncovered: 0,
            covered: 0,
            killed: 0,
            alpha_killed: 0,
            written: 0,
            first_shaded: None,
            first_written: None,
            first_screen: None,
        }
    }

    fn geometry(&mut self, screen: [ScreenVertex; 3]) {
        if self.enabled && self.first_screen.is_none() {
            self.first_screen = Some(screen);
        }
    }

    fn outside(&mut self, screen: [ScreenVertex; 3]) {
        self.outside += 1;
        if screen.iter().any(|v| !v.x.is_finite() || !v.y.is_finite()) {
            self.nonfinite += 1;
        }
    }

    pub(super) fn shaded(&mut self, color: [f32; 4]) {
        if self.enabled && self.first_shaded.is_none() {
            self.first_shaded = Some(color);
        }
    }

    pub(super) fn wrote(&mut self, color: [f32; 4]) {
        self.written += 1;
        if self.enabled && self.first_written.is_none() {
            self.first_written = Some(color);
        }
    }

    fn report(
        &self,
        call: &crate::gpu::engine::threed::DrawCall,
        primitive: Primitive,
        blend: BlendTarget,
        bounds: Bounds,
    ) {
        if !self.enabled {
            return;
        }
        let blend = if blend.enabled {
            format!(
                "{:#x},{:#x},{:#x}/{:#x},{:#x},{:#x}",
                blend.equation_rgb,
                blend.func_rgb_src,
                blend.func_rgb_dst,
                blend.equation_alpha,
                blend.func_alpha_src,
                blend.func_alpha_dst
            )
        } else {
            "off".to_string()
        };
        let bounds = (bounds.x0, bounds.y0, bounds.x1, bounds.y1);
        crate::traceln!(
            "[draw] {primitive:?} count={} indexed={} fs_ops={} bounds={bounds:?} \
             blend={blend} tris={} culled={} degen={} outside={} nonfinite={} covered={} \
             uncovered={} kil={} a2c={} wrote={} shaded={:?} out={:?} screen={:?}",
            call.count,
            call.indexed,
            self.fs_len,
            self.triangles,
            self.culled,
            self.degenerate,
            self.outside,
            self.nonfinite,
            self.covered,
            self.uncovered,
            self.killed,
            self.alpha_killed,
            self.written,
            self.first_shaded,
            self.first_written,
            self.first_screen.map(|s| s.map(|v| (v.x, v.y))),
        );
    }
}
