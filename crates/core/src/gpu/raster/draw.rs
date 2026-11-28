//! The draw entry point: index fetch, culling, clipping and the pixel loop.

use super::fragment::{
    quad_pixel, shade_fragment, shade_quad, with_fragment_env, Fragments, OutputTarget, QUAD,
};
use super::setup::TriangleSetup;
use super::vertex::{shade_vertex, to_screen, ShadedVertex, MAX_VERTEX_ATTRIBS, NUM_VARYINGS};
use super::{assemble, Bounds, Primitive, ScreenVertex};
use crate::gpu::engine::threed::{
    BlendTarget, CullState, DepthTarget, DrawCall, Engine3D, RenderTarget, ScissorRect,
    ShaderStage, VertexArray, VertexAttrib, ViewportTransform,
};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::decode_program_from_memory;
use crate::gpu::shader::sources::{
    ConstCache, MemoryConstants, MemoryGlobal, MemoryTextures, SurfaceWrite,
};
use crate::gpu::surface::{SampleGrid, MAX_SAMPLES};
use crate::gpu::texture::{BlockCache, DescriptorCache};
use crate::{Error, Result};
use std::cell::RefCell;
use switch_shader::compiled::Compiled;
use switch_shader::interp::Invocation;
use switch_shader::{wgsl, Op, Program};

/// Whether `cull` throws this triangle away, judged by its winding in window
/// space (y down) after the viewport transform, as Eden does.
pub(super) fn culls(cull: CullState, v: [ScreenVertex; 3]) -> bool {
    if !cull.enabled {
        return false;
    }
    match front_facing(cull, v) {
        None => true,
        Some(true) => cull.cull_front,
        Some(false) => cull.cull_back,
    }
}

/// Whether the triangle faces front by `cull`'s winding, judged as [`culls`] judges it;
/// `None` when it has no area.
pub(super) fn front_facing(cull: CullState, v: [ScreenVertex; 3]) -> Option<bool> {
    let area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
    (area != 0.0).then_some((area < 0.0) == cull.front_ccw)
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
/// Whether a program can write global memory.
fn stores_globally(program: &Program) -> bool {
    program
        .insns
        .iter()
        .any(|insn| matches!(insn.op, Op::Stg { .. } | Op::Atom { .. }))
}

/// Whether [`draw`] would only run the vertex shader for what it stores, sampling
/// no texture: a point or line draw, which nothing here rasterizes. Such a draw
/// reads no render target, so it can run here while a device holds them.
pub fn only_shades_vertices(engine: &Engine3D, ctx: &ExecCtx) -> Result<bool> {
    if Primitive::from_raw(engine.last_draw.primitive)?.is_rasterized() {
        return Ok(false);
    }
    let Some(binding) = engine.program(ShaderStage::VertexB) else {
        return Ok(false);
    };
    let program = decode_program_from_memory(ctx, binding.addr, &|bank: u8| {
        engine.bound_constbuf(ShaderStage::VertexB, bank as u32)
    })?;
    let samples = program.insns.iter().any(|insn| {
        matches!(
            insn.op,
            Op::Texs { .. }
                | Op::Tlds { .. }
                | Op::Tld { .. }
                | Op::Tex { .. }
                | Op::Txq { .. }
                | Op::Tld4 { .. }
                | Op::Tld4s { .. }
                | Op::Suld { .. }
        )
    });
    Ok(stores_globally(&program) && !samples)
}

pub fn draw(engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
    let call = engine.last_draw;
    // Logical target 0, mapped through RenderTargetControl as a clear is.
    let rt = engine.draw_target(engine.render_target_slot(0))?;

    let vs_binding = engine
        .program(ShaderStage::VertexB)
        .ok_or_else(|| Error::Gpu("raster: draw with no bound vertex program".into()))?;
    if crate::gpu::pipeline::writes_no_surface(engine) {
        return Ok(());
    }
    let fs_binding = engine.program(ShaderStage::Fragment);
    let fs_addr = fs_binding.map_or(0, |binding| binding.addr);

    let vs_decoded = decode_program_from_memory(&*ctx, vs_binding.addr, &|bank: u8| {
        engine.bound_constbuf(ShaderStage::VertexB, bank as u32)
    })?;
    let fs_decoded = match fs_binding {
        Some(binding) => decode_program_from_memory(&*ctx, binding.addr, &|bank: u8| {
            engine.bound_constbuf(ShaderStage::Fragment, bank as u32)
        })?,
        None => switch_shader::no_fragment_program(),
    };
    // Only when a diagnostic asked; see `examples/shader_coverage.rs`.
    switch_shader::uses::note(wgsl::Stage::Vertex, vs_binding.addr, &vs_decoded);
    if fs_binding.is_some() {
        switch_shader::uses::note(wgsl::Stage::Fragment, fs_addr, &fs_decoded);
    }
    trace_wgsl(
        engine,
        (vs_binding.addr, &vs_decoded),
        (fs_addr, &fs_decoded),
    );

    let attribs: Vec<VertexAttrib> = (0..MAX_VERTEX_ATTRIBS)
        .map(|i| engine.vertex_attrib(i))
        .collect();
    let arrays: Vec<VertexArray> = (0..MAX_VERTEX_ATTRIBS)
        .map(|i| engine.vertex_array(i))
        .collect();
    let index_base = if call.indexed {
        engine.index_array_start()
    } else {
        0
    };
    let primitive = Primitive::from_raw(call.primitive)?;
    // Report point and line topologies rather than silently drawing nothing.
    if !primitive.is_rasterized() {
        // What the vertex shader stores still lands: Nintendo Switch Sports resets
        // records in memory with a points draw that only writes through pointers.
        if stores_globally(&vs_decoded) {
            shade_for_stores(engine, ctx, &vs_decoded, (&attribs, &arrays), index_base)?;
        }
        return Err(Error::Gpu(format!(
            "raster: {primitive:?} is not rasterized"
        )));
    }
    let viewport = engine.viewport_transform();
    let grid = engine.sample_grid()?;
    let sample_mask = engine.sample_mask();
    let alpha_to_coverage = engine.alpha_to_coverage();
    let depth = engine.depth_target()?;
    // A draw with no fragment program writes depth only, as does one whose mask is empty.
    let outputs = engine.color_outputs();
    let targets = if fs_binding.is_some() {
        color_targets(engine, outputs)?
    } else {
        Vec::new()
    };
    let bounds = draw_bounds(engine, rt.or(targets.first().map(|t| t.rt)), depth, grid)?;
    let depth_state = engine.depth_state();
    let blend_target = engine.blend_target(0);
    let blend_constant = engine.blend_constant();
    let cull = engine.cull_state();

    let triangles = assemble(primitive, call.count);
    let mut tally = DrawTally::new(&fs_decoded);
    // Vertex shading is cached per index.
    let mut cache: crate::IdMap<u32, ShadedVertex> = crate::IdMap::default();
    // One fragment invocation for the whole draw, reset per pixel.
    let mut fragment = Invocation::new();
    let fragments = Fragments {
        grid,
        sample_mask,
        depth,
        depth_state,
        targets,
        blend_constant,
        alpha_to_coverage,
    };
    // Parsed TIC/TSC pairs and decoded blocks, shared by the draw.
    let descriptors = RefCell::new(crate::IdMap::default());
    let blocks = RefCell::new(BlockCache::default());
    let vs_const_cache = RefCell::new(ConstCache::default());
    let fs_const_cache = RefCell::new(ConstCache::default());
    let vs_program = lower(
        engine,
        &*ctx,
        ShaderStage::VertexB,
        &vs_decoded,
        &vs_const_cache,
    );
    let fs_program = lower(
        engine,
        &*ctx,
        ShaderStage::Fragment,
        &fs_decoded,
        &fs_const_cache,
    );
    let mut quad: Option<Box<[Invocation; QUAD]>> =
        runs_in_quads(&fs_program).then(|| Box::new(std::array::from_fn(|_| Invocation::new())));

    trace_pipeline(engine);
    trace_uploads(engine, &*ctx, &vs_program, &fs_program);
    trace_cfg((vs_binding.addr, &vs_program), (fs_addr, &fs_program));
    // What the draw's shaders store to global memory, landed after each
    // shading step: see `MemoryGlobal`.
    let global_stores = RefCell::new(Vec::new());
    let surface_stores = RefCell::new(Vec::new());
    let vertices = VertexFetch {
        engine,
        call,
        attribs: &attribs,
        arrays: &arrays,
        index_base,
        instance_id: engine.instance_id(),
        program: &vs_program,
        consts: &vs_const_cache,
        descriptors: &descriptors,
        blocks: &blocks,
        stores: &global_stores,
    };
    let raster = Raster {
        engine,
        program: &fs_program,
        fragments: &fragments,
        outputs,
        consts: &fs_const_cache,
        descriptors: &descriptors,
        blocks: &blocks,
        stores: &global_stores,
        surface_stores: &surface_stores,
    };
    for tri in triangles {
        let unclipped = vertices.triangle(ctx, &mut cache, tri)?;
        for clipped in clip_near(unclipped) {
            let Some(piece) = Piece::new(clipped, viewport, cull, bounds, &mut tally) else {
                continue;
            };
            match quad.as_deref_mut() {
                None => raster.pixels(&piece, &mut fragment, ctx, &mut tally)?,
                Some(quad) => raster.quads(&piece, quad, ctx, &mut tally)?,
            }
        }
    }
    tally.report(&call, primitive, blend_target, bounds);
    Ok(())
}

/// `TRACE_WGSL`: whether each shader translates, and each module as
/// `<stage>_<addr>.wgsl` for `naga --validate` when a dump sink wants it.
fn trace_wgsl(engine: &Engine3D, vs: (u64, &Program), fs: (u64, &Program)) {
    if !crate::trace::enabled(crate::trace::Trace::Wgsl) {
        return;
    }
    let keep = crate::dump::wants(crate::dump::Artefact::RasterWgsl);
    use switch_shader::wgsl::{Layout, Stage};
    // Unfolded, as the device translates it: folding turns the constant
    // operands a `ldg` descriptor is traced through into immediates.
    for (name, stage, (addr, program)) in [("vs", Stage::Vertex, vs), ("fs", Stage::Fragment, fs)] {
        let translated = match wgsl::translate(&Compiled::new(program)) {
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
            layout.window_coordinates = pipeline.viewport.window_coordinates;
        }
        match wgsl::module(&translated, stage, &layout) {
            Ok(_) if !keep => crate::traceln!(
                "[wgsl] {name}@{addr:#x} {} regs, {} attribs, {} varyings, \
                 {} banks, {} textures",
                translated.registers.len(),
                layout.attributes.len(),
                layout.varyings.len(),
                layout.const_banks.len(),
                layout.textures.len()
            ),
            Ok(module) => {
                let file = format!("{name}_{addr:x}.wgsl");
                crate::dump::keep(crate::dump::Artefact::RasterWgsl, &file, module.as_bytes());
                crate::traceln!("[wgsl] {name}@{addr:#x} -> {file}");
            }
            Err(e) => crate::traceln!("[wgsl] {name}@{addr:#x} no module: {e}"),
        }
    }
}

/// Lower `program`: resolve branch targets, fold the constants bound to `stage`.
fn lower(
    engine: &Engine3D,
    ctx: &ExecCtx,
    stage: ShaderStage,
    program: &Program,
    cache: &RefCell<ConstCache>,
) -> Compiled {
    let consts = MemoryConstants {
        ctx,
        bindings: &|bank: u8| engine.bound_constbuf(stage, bank as u32),
        cache,
    };
    Compiled::with_constants(program, &consts)
}

/// Shaders with warp shuffles or votes, or that read their lane, run in 2x2 quads.
fn runs_in_quads(program: &Compiled) -> bool {
    program.ops().iter().any(|op| {
        matches!(op, Op::Shfl { .. } | Op::Vote { .. } | Op::Fswzadd { .. })
            || matches!(op, Op::S2r { sr, .. } if switch_shader::reads_lane(*sr))
    })
}

/// Run the vertex shader of a draw nothing here rasterizes, once per vertex, for what it stores.
fn shade_for_stores(
    engine: &Engine3D,
    ctx: &mut ExecCtx,
    vs_decoded: &Program,
    (attribs, arrays): (&[VertexAttrib], &[VertexArray]),
    index_base: u64,
) -> Result<()> {
    let call = engine.last_draw;
    let vs_const_cache = RefCell::new(ConstCache::default());
    let vs_program = lower(
        engine,
        &*ctx,
        ShaderStage::VertexB,
        vs_decoded,
        &vs_const_cache,
    );
    let global_stores = RefCell::new(Vec::new());
    let descriptors = RefCell::new(crate::IdMap::default());
    let blocks = RefCell::new(BlockCache::default());
    let vertices = VertexFetch {
        engine,
        call,
        attribs,
        arrays,
        index_base,
        instance_id: engine.instance_id(),
        program: &vs_program,
        consts: &vs_const_cache,
        descriptors: &descriptors,
        blocks: &blocks,
        stores: &global_stores,
    };
    let mut shaded = std::collections::BTreeSet::new();
    for ordinal in 0..call.count {
        let index = vertices.index(&*ctx, ordinal)?;
        if !shaded.insert(index) {
            continue;
        }
        vertices.shade(ctx, index)?;
    }
    Ok(())
}

/// The colour targets the fragment program's `outputs` land in, skipping unbound and fully masked ones.
fn color_targets(engine: &Engine3D, outputs: u32) -> Result<Vec<OutputTarget>> {
    let mut targets = Vec::new();
    for output in 0..outputs {
        let slot = engine.render_target_slot(output);
        let Some(rt) = engine.draw_target(slot)? else {
            continue;
        };
        let mask = engine.color_mask(slot);
        if !mask.contains(&true) {
            continue;
        }
        targets.push(OutputTarget {
            output: output as usize,
            rt,
            blend: engine.blend_target(output),
            mask,
        });
    }
    Ok(targets)
}

/// The scissored box a draw covers, from the extent of its colour target `rt` or its depth target.
fn draw_bounds(
    engine: &Engine3D,
    rt: Option<RenderTarget>,
    depth: Option<DepthTarget>,
    grid: SampleGrid,
) -> Result<Bounds> {
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
    Ok(Bounds {
        x0: clip.x0,
        y0: clip.y0,
        x1: clip.x1,
        y1: clip.y1,
    })
}

/// `--trace pipeline`: this draw's fixed-function state, or why it is
/// not describable.
fn trace_pipeline(engine: &Engine3D) {
    if !crate::trace::enabled(crate::trace::Trace::Pipeline) {
        return;
    }
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

/// `--trace upload`: the guest bytes this draw would upload to a device.
fn trace_uploads(engine: &Engine3D, ctx: &ExecCtx, vs_program: &Compiled, fs_program: &Compiled) {
    if !crate::trace::enabled(crate::trace::Trace::Upload) {
        return;
    }
    match crate::gpu::pipeline::Pipeline::of(engine)
        .map_err(|e| Error::Gpu(format!("pipeline: {e}")))
        .and_then(|p| {
            // The two stages index different constant buffers per slot.
            let mut slots: Vec<(ShaderStage, switch_shader::texture::TextureSlot)> = Vec::new();
            for (stage, program) in [
                (ShaderStage::VertexB, vs_program),
                (ShaderStage::Fragment, fs_program),
            ] {
                if let Ok(translated) = switch_shader::wgsl::translate(program) {
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
                ctx,
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
                .map(|t| t.bytes().map_or(0, |b| b.len()))
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

/// `--trace cfg`: each shader's control flow, as a translator sees it.
fn trace_cfg(vs: (u64, &Compiled), fs: (u64, &Compiled)) {
    if !crate::trace::enabled(crate::trace::Trace::Cfg) {
        return;
    }
    for (stage, (addr, program)) in [("vs", vs), ("fs", fs)] {
        crate::traceln!(
            "[cfg] {stage}@{addr:#x} {}",
            switch_shader::cfg::Cfg::new(program).describe()
        );
    }
}

/// What shading one of a draw's vertices reads.
struct VertexFetch<'a> {
    engine: &'a Engine3D,
    call: DrawCall,
    attribs: &'a [VertexAttrib],
    arrays: &'a [VertexArray],
    index_base: u64,
    instance_id: u32,
    program: &'a Compiled,
    consts: &'a RefCell<ConstCache>,
    descriptors: &'a RefCell<DescriptorCache>,
    blocks: &'a RefCell<BlockCache>,
    stores: &'a RefCell<Vec<(u64, u32)>>,
}

impl VertexFetch<'_> {
    /// The vertex the draw's `ordinal`th index names.
    #[inline]
    fn index(&self, ctx: &ExecCtx, ordinal: u32) -> Result<u32> {
        let call = self.call;
        Ok(if call.indexed {
            call.vertex_of(read_index(
                ctx,
                self.index_base,
                call.index_format,
                call.first + ordinal,
            )?)
        } else {
            call.first + ordinal
        })
    }

    /// Shade the vertices of the draw's `tri`, through `cache`, which is per index.
    #[inline]
    fn triangle(
        &self,
        ctx: &mut ExecCtx,
        cache: &mut crate::IdMap<u32, ShadedVertex>,
        tri: [u32; 3],
    ) -> Result<[ClipVertex; 3]> {
        let mut shaded: Vec<ShadedVertex> = Vec::with_capacity(3);
        for &ordinal in &tri {
            let index = self.index(&*ctx, ordinal)?;
            if let Some(v) = cache.get(&index) {
                shaded.push(*v);
                continue;
            }
            let v = self.shade(ctx, index)?;
            cache.insert(index, v);
            shaded.push(v);
        }

        Ok([
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
        ])
    }

    /// Shade vertex `index` and land what it stores.
    #[inline]
    fn shade(&self, ctx: &mut ExecCtx, index: u32) -> Result<ShadedVertex> {
        let engine = self.engine;
        let vs_consts = MemoryConstants {
            ctx: &*ctx,
            bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::VertexB, bank as u32),
            cache: self.consts,
        };
        let vs_textures = MemoryTextures {
            ctx: &*ctx,
            tex_header_pool: engine.tex_header_pool(),
            tex_sampler_pool: engine.tex_sampler_pool(),
            descriptors: self.descriptors,
            blocks: self.blocks,
            surface_stores: None,
        };
        let v = shade_vertex(
            self.program,
            self.attribs,
            self.arrays,
            (index, self.instance_id),
            &*ctx,
            &vs_consts,
            (&vs_textures, engine.tex_cb_index()),
            engine.window_origin().lower_left,
            self.stores,
        )?;
        MemoryGlobal::land(ctx, self.stores)?;
        Ok(v)
    }
}

/// One near-clipped triangle that survived culling and has pixels in the draw's bounds.
struct Piece {
    tri: TriangleSetup,
    shaded: [ShadedVertex; 3],
    inv_w: [f32; 3],
    window_z: [f32; 3],
    front: bool,
    /// `(min_x, max_x, min_y, max_y)`, half-open.
    bbox: (u32, u32, u32, u32),
}

impl Piece {
    /// Project `piece` to the screen, or count why it draws nothing.
    #[inline]
    fn new(
        piece: [ClipVertex; 3],
        viewport: ViewportTransform,
        cull: CullState,
        bounds: Bounds,
        tally: &mut DrawTally,
    ) -> Option<Piece> {
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
            return None;
        }
        let front = front_facing(cull, screen).unwrap_or(true);

        let Some(tri) = TriangleSetup::new(screen[0], screen[1], screen[2]) else {
            tally.degenerate += 1;
            return None;
        };
        let (min_x, max_x, min_y, max_y) = tri.bbox(bounds);
        if min_x >= max_x || min_y >= max_y {
            tally.outside(screen);
            return None;
        }
        Some(Piece {
            tri,
            shaded,
            inv_w,
            window_z,
            front,
            bbox: (min_x, max_x, min_y, max_y),
        })
    }
}

/// What shading and writing a draw's fragments reads.
struct Raster<'a> {
    engine: &'a Engine3D,
    program: &'a Compiled,
    fragments: &'a Fragments,
    outputs: u32,
    consts: &'a RefCell<ConstCache>,
    descriptors: &'a RefCell<DescriptorCache>,
    blocks: &'a RefCell<BlockCache>,
    stores: &'a RefCell<Vec<(u64, u32)>>,
    surface_stores: &'a RefCell<Vec<SurfaceWrite>>,
}

impl Raster<'_> {
    /// Shade and write `piece` a pixel at a time.
    #[inline]
    fn pixels(
        &self,
        piece: &Piece,
        fragment: &mut Invocation,
        ctx: &mut ExecCtx,
        tally: &mut DrawTally,
    ) -> Result<()> {
        let (tri, shaded) = (&piece.tri, &piece.shaded);
        let (inv_w, window_z, front) = (piece.inv_w, piece.window_z, piece.front);
        let (min_x, max_x, min_y, max_y) = piece.bbox;
        let fragments = self.fragments;
        let mut sample_z = [0.0f32; MAX_SAMPLES];
        for y in min_y..max_y {
            for x in min_x..max_x {
                let covered = fragments.coverage(tri, window_z, (x, y), &mut sample_z, ctx)?;
                if covered == 0 {
                    tally.uncovered += 1;
                    continue;
                }
                tally.covered += 1;

                let weights = tri.weights(x as f32 + 0.5, y as f32 + 0.5);
                let colors = with_fragment_env(
                    self.engine,
                    &*ctx,
                    self.consts,
                    self.descriptors,
                    self.blocks,
                    self.stores,
                    self.surface_stores,
                    |env| {
                        shade_fragment(
                            &mut *fragment,
                            self.program,
                            shaded,
                            inv_w,
                            weights,
                            (x, y),
                            front,
                            env,
                            self.outputs,
                        )
                    },
                )?;
                MemoryGlobal::land(ctx, self.stores)?;
                MemoryTextures::land(ctx, self.surface_stores)?;
                // `kil` skips the depth write too, so it waits for shading.
                let Some(colors) = colors else {
                    tally.killed += 1;
                    continue;
                };
                fragments.write((x, y), covered, &sample_z, &colors, ctx, tally)?;
            }
        }
        Ok(())
    }

    /// Shade and write `piece` a 2x2 quad at a time, for shaders that talk across lanes.
    #[inline]
    fn quads(
        &self,
        piece: &Piece,
        quad: &mut [Invocation; QUAD],
        ctx: &mut ExecCtx,
        tally: &mut DrawTally,
    ) -> Result<()> {
        let (tri, shaded) = (&piece.tri, &piece.shaded);
        let (inv_w, window_z, front) = (piece.inv_w, piece.window_z, piece.front);
        let (min_x, max_x, min_y, max_y) = piece.bbox;
        let fragments = self.fragments;
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
                    let mask =
                        fragments.coverage(tri, window_z, (px, py), &mut sample_z[lane], ctx)?;
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
                    self.engine,
                    &*ctx,
                    self.consts,
                    self.descriptors,
                    self.blocks,
                    self.stores,
                    self.surface_stores,
                    |env| {
                        shade_quad(
                            quad,
                            self.program,
                            shaded,
                            inv_w,
                            weights,
                            (x, y),
                            front,
                            env,
                            self.outputs,
                        )
                    },
                )?;
                MemoryGlobal::land(ctx, self.stores)?;
                MemoryTextures::land(ctx, self.surface_stores)?;
                for lane in 0..QUAD {
                    if covered[lane] == 0 {
                        continue;
                    }
                    let Some(lane_colors) = &colors[lane] else {
                        tally.killed += 1;
                        continue;
                    };
                    fragments.write(
                        quad_pixel(x, y, lane),
                        covered[lane],
                        &sample_z[lane],
                        lane_colors,
                        ctx,
                        tally,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Per-draw fragment accounting for `--trace draw`: which stage fragments died at.
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
