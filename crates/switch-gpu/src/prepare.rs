//! Checking and preparing a draw, and translating its shaders.

use std::sync::Arc;

use crate::convert::{held_attachment_format, Widen};
use crate::held::{spans_overlap, surface_span};
use crate::{
    CachedShader, Checked, Gpu, Prepared, Render, PAGE_BITS, SEQUENTIAL_INDEX_LISTS,
    SHADER_CACHE_ENTRIES,
};
use switch_core::gpu::engine::threed::{Engine3D, ShaderStage};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state, AttributeBase, Pipeline};
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Coverage, Layout, Stage, Translation};
use switch_core::gpu::texture::TextureSlot;
use switch_core::gpu::upload::{Banks, Target, Targets, TexturePools, Uploads};

impl Gpu {
    /// Decide whether a draw can run here, without uploading anything.
    pub(super) fn check(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<Checked, String> {
        let mut state = Pipeline::of(engine).map_err(|e| e.to_string())?;
        let targets = Targets::of(engine).map_err(|e| format!("{e:?}"))?;
        // With its fragment stage disabled a draw writes no colour.
        let color = targets
            .color
            .filter(|_| engine.program(ShaderStage::Fragment).is_some());
        // Attach depth only for draws that test or write it.
        let uses_depth = state
            .depth
            .is_some_and(|d| d.write_enabled || d.compare != state::Compare::Always);
        let depth = targets.depth.filter(|_| uses_depth);
        // A target every channel of which is masked off takes nothing, so a draw with depth to
        // render to leaves it unattached, free for its fragment stage to store to as an image.
        let masked = state
            .target
            .is_some_and(|t| t.write_mask.iter().all(|&on| !on));
        let color = color.filter(|_| !(masked && depth.is_some()));

        // Unfolded, so a module depends only on the shader binary.
        let vs = timed!(
            self,
            translate,
            self.translate(engine, ctx, ShaderStage::VertexB)
        );
        let fs = timed!(
            self,
            translate,
            self.translate(engine, ctx, ShaderStage::Fragment)
        );
        let ((vs, vs_serial), (fs, fs_serial)) = (vs?, fs?);
        let extra_colors = extra_attachments(engine, &state, &fs)?;
        let extras = extra_colors.iter().any(Option::is_some);
        // The first attached colour sets the pass's size, which every other one must have.
        let sizing = color.or_else(|| extra_colors.iter().flatten().next().copied());
        if let Some(sizing) = sizing {
            if let Some(other) = std::iter::once(color)
                .chain(extra_colors.iter().copied())
                .flatten()
                .find(|t| (t.width, t.height) != (sizing.width, sizing.height))
            {
                return Err(format!(
                    "colour outputs of different sizes, {}x{} at {:#x} beside {}x{} at {:#x}",
                    other.width, other.height, other.addr, sizing.width, sizing.height, sizing.addr
                ));
            }
        }
        // WebGPU refuses one texture attached twice to a pass.
        let attached: Vec<u64> = std::iter::once(color)
            .chain(extra_colors.iter().copied())
            .chain(std::iter::once(depth))
            .flatten()
            .map(|t| t.addr)
            .collect();
        for (at, addr) in attached.iter().enumerate() {
            if attached[at + 1..].contains(addr) {
                return Err(format!("one surface at {addr:#x} attached twice to a draw"));
            }
        }

        // Attachments must match in size, so the pass covers the intersection. A depth surface is
        // cropped along each axis it is larger; a colour target larger along either draws into a
        // scratch texture copied back afterwards.
        let mut depth = depth;
        let mut color_scratch = None;
        if let (Some(full_color), Some(full_depth)) = (sizing, depth) {
            let (cw, ch) = (full_color.width, full_color.height);
            let (dw, dh) = (full_depth.width, full_depth.height);
            if dw >= cw && dh >= ch {
                depth = Some(crate::crop_depth(full_depth, Some(full_color)));
            } else if extras {
                return Err("several colour outputs larger than their depth surface".into());
            } else {
                let (width, height) = (cw.min(dw), ch.min(dh));
                depth = Some(crate::crop_depth(full_depth, Some(full_color)));
                color_scratch = Some((width, height));
                // The scissor is confined to the depth surface only for draws that reach it.
                let (pixels_x, pixels_y) = state.grid.pixels(width, height);
                state.scissor.x1 = state.scissor.x1.min(pixels_x);
                state.scissor.y1 = state.scissor.y1.min(pixels_y);
                state.scissor.x0 = state.scissor.x0.min(state.scissor.x1);
                state.scissor.y0 = state.scissor.y0.min(state.scissor.y1);
            }
        }
        if sizing.is_none() && depth.is_none() {
            return Err("a draw into neither a colour nor a depth surface".into());
        }
        let render = self.route(&state, sizing, depth)?;
        if color_scratch.is_some() && matches!(render, Render::Companion(_)) {
            return Err(
                "a colour target larger than its depth surface, drawn through a multisample companion"
                    .into(),
            );
        }
        if extras && render != Render::Direct {
            return Err("several colour outputs drawn multisampled".into());
        }
        Ok(Checked {
            state,
            render,
            color,
            extra_colors,
            color_scratch,
            depth,
            vs,
            fs,
            serials: [vs_serial, fs_serial],
        })
    }

    pub(super) fn prepare(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<Prepared, String> {
        let Checked {
            state,
            render,
            color,
            extra_colors,
            color_scratch,
            depth,
            vs,
            fs,
            serials,
        } = self.check(engine, ctx)?;

        let mut vs_layout = Layout::of(&vs, Stage::Vertex);
        let mut fs_layout = Layout::of(&fs, Stage::Fragment);
        bind_fetch_levels(engine, ctx, &mut vs_layout, ShaderStage::VertexB)?;
        bind_fetch_levels(engine, ctx, &mut fs_layout, ShaderStage::Fragment)?;
        let storage_limit = self
            .device_limits
            .clone()
            .max_storage_buffers_per_shader_stage as usize;
        let limits = self.device_limits.clone();
        for (stage, layout) in [("vertex", &vs_layout), ("fragment", &fs_layout)] {
            if layout.textures.len() > limits.max_sampled_textures_per_shader_stage as usize {
                return Err(format!(
                    "{} textures in the {stage} stage, past the device's limit",
                    layout.textures.len()
                ));
            }
            let buffers = usize::from(!layout.const_banks.is_empty()) + layout.globals.len();
            if buffers > storage_limit {
                return Err(format!(
                    "{buffers} storage buffers in the {stage} stage, past the device's {storage_limit}"
                ));
            }
        }
        let origin = engine.window_origin();
        fs_layout.window_height = origin.lower_left.then(|| engine.surface_clip_height());
        // A depth-only pass has no colour output; past output 0, each output up to the last
        // attached one is declared, the unattached ones landing nowhere.
        let attached_outputs = extra_colors
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |last| last as u32 + 1);
        fs_layout.targets = match (color, attached_outputs) {
            (None, 0) => 0,
            _ => 1 + attached_outputs,
        };
        let outputs = std::iter::once(color).chain(extra_colors.iter().copied());
        for (output, target) in outputs.enumerate().take(fs_layout.targets as usize) {
            let Some(target) = target else {
                continue;
            };
            let output = output as u32;
            // An integer target takes raw words.
            if let Some(max) = target.format.uint_max() {
                fs_layout.uint_targets.push((output, max));
            }
            if held_attachment_format(self.features(), target.format)
                .is_ok_and(|(_, widen)| widen == Widen::Unorm16)
            {
                fs_layout.unorm16_targets.push(output);
            }
        }
        // Both stages must name the same varyings; missing ones read as zero.
        let mut varyings = vs_layout.varyings.clone();
        varyings.extend(fs_layout.varyings.iter().copied());
        varyings.sort_unstable();
        varyings.dedup();
        vs_layout.varyings = varyings.clone();
        fs_layout.varyings = varyings;
        // Only the fragment program says which varyings are centroid.
        vs_layout.centroid_varyings = fs_layout.centroid_varyings.clone();
        // Negated because WebGPU mirrors y itself. See `Layout::flip_y`.
        vs_layout.flip_y = !state.viewport.flip_y;
        vs_layout.depth_minus_one_to_one = state.viewport.depth_minus_one_to_one();
        vs_layout.window_coordinates = state.viewport.window_coordinates;
        // On the expanded route the shader applies the sample mask and alpha-to-coverage.
        if render == Render::Expanded {
            fs_layout.coverage = Some(Coverage {
                samples_x: state.grid.samples_x,
                samples_y: state.grid.samples_y,
                sample_of_slot: state.grid.sample_of_slot()[..state.samples as usize].to_vec(),
                sample_mask: state.sample_mask,
                alpha_to_coverage: state.alpha_to_coverage,
                per_sample_pass: !state.grid.samples_at_texel_centres(),
            });
        }
        // Integer attributes come from the draw's registers.
        vs_layout.integer_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.base() != AttributeBase::Float)
            .map(|a| (a.location as usize, a.base()))
            .collect();
        // WebGPU has no BGRA vertex format, so the entry point swaps.
        vs_layout.bgra_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.is_bgra)
            .map(|a| a.location as usize)
            .collect();
        // And which arrive as one 10-10-10-2 word to unpack.
        vs_layout.packed_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter_map(|a| match a.format {
                state::VertexFormat::Packed1010102(packing) => Some((a.location as usize, packing)),
                _ => None,
            })
            .collect();
        // One bind group per stage; see `Layout::group`.
        vs_layout.group = 0;
        fs_layout.group = 1;
        let drawn_surfaces: Vec<Target> = std::iter::once(color)
            .chain(extra_colors.iter().copied())
            .chain(std::iter::once(depth))
            .flatten()
            .collect();
        let images = self.draw_images(engine, ctx, &vs_layout, &mut fs_layout, &drawn_surfaces)?;

        // The layouts' textures: the translations' and the levels bound beside them.
        let mut slots: Vec<(ShaderStage, TextureSlot)> = Vec::new();
        slots.extend(
            vs_layout
                .textures
                .iter()
                .map(|b| (ShaderStage::VertexB, b.slot)),
        );
        slots.extend(
            fs_layout
                .textures
                .iter()
                .map(|b| (ShaderStage::Fragment, b.slot)),
        );
        let mut banks: Vec<(ShaderStage, u32, u32)> = Vec::new();
        for (stage, layout) in [(ShaderStage::VertexB, &vs), (ShaderStage::Fragment, &fs)] {
            banks.extend(
                layout
                    .const_banks
                    .iter()
                    .zip(&layout.const_extents)
                    .map(|(&b, &extent)| (stage, u32::from(b), extent)),
            );
        }
        // Taken out of `self` because the closure holds it while `timed!` borrows `self`.
        let cache = std::mem::take(&mut self.space.texture_cache);
        let mut hits = 0u64;
        let uploads = timed!(self, upload, {
            Uploads::of_cached(
                engine,
                &state,
                ctx,
                Banks::Read(&banks),
                &slots,
                &mut |key| {
                    let hit = cache.get(key).cloned();
                    hits += u64::from(hit.is_some());
                    hit
                },
            )
        });
        self.space.texture_cache = cache;
        self.texture_hits += hits;
        let uploads = uploads.map_err(|e| format!("{e:?}"))?;
        for upload in &uploads.textures {
            self.hold_at_sampled_size(upload, ctx)
                .map_err(|e| format!("{e:?}"))?;
        }
        // Vertices and constants are copied from where compute launches wrote them on the
        // device; indices also bound the vertices uploaded, which stale ones would get wrong.
        if uploads.index.is_some() {
            let call = engine.last_draw;
            let width = match call.index_format {
                0 => 1,
                1 => 2,
                _ => 4,
            };
            let start = engine.index_array_start() + u64::from(call.first) * width;
            let indices = start..start + u64::from(call.count) * width;
            if !self.no_global_in(std::slice::from_ref(&indices)) {
                return Err("reads indices a compute launch wrote on the device".into());
            }
        }
        let constant_patches = uploads
            .constants
            .iter()
            .map(|constants| {
                let Some((addr, _)) = engine.bound_constbuf(constants.stage, constants.bank) else {
                    return Ok(Vec::new());
                };
                // `pack_banks` keeps whole words.
                let len = constants.bytes.len() as u64 & !3;
                self.patches_over(addr..addr + len, "constants")
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        self.uploaded.add_but_textures(&uploads);
        self.keep_textures(ctx, &uploads.textures);

        self.fill_texture_bindings(&mut vs_layout, ShaderStage::VertexB, &uploads.textures)?;
        self.fill_texture_bindings(&mut fs_layout, ShaderStage::Fragment, &uploads.textures)?;
        let drawn: Vec<u64> = drawn_surfaces.iter().map(|t| t.addr).collect();
        for (layout, stage, translation) in [
            (&mut vs_layout, ShaderStage::VertexB, &vs),
            (&mut fs_layout, ShaderStage::Fragment, &fs),
        ] {
            for binding in &mut layout.textures {
                if let Some(upload) = uploads
                    .textures
                    .iter()
                    .find(|t| t.stage == stage && t.slot == binding.slot)
                {
                    binding.crop =
                        self.crop_of_held(upload, binding, &translation.read_otherwise, &drawn);
                }
            }
        }
        for (layout, stage, name) in [
            (&mut vs_layout, ShaderStage::VertexB, "vertex"),
            (&mut fs_layout, ShaderStage::Fragment, "fragment"),
        ] {
            self.share_samplers(layout, stage, &uploads.textures);
            let samplers = layout
                .textures
                .iter()
                .filter(|b| b.sampler.is_none())
                .count();
            if samplers > limits.max_samplers_per_shader_stage as usize {
                return Err(format!(
                    "{samplers} distinct samplers in the {name} stage, past the device's limit"
                ));
            }
        }

        // Fans, quad strips and polygons become triangle lists.
        let assembled = match (state.expand, &uploads.index) {
            // Indexed: triples index the index list; base vertex is the lowest index.
            (Some(primitive), Some(index)) => {
                let triangles =
                    switch_core::gpu::raster::assemble(primitive, engine.last_draw.count);
                let list = index.indices();
                let mut bytes = Vec::with_capacity(triangles.len() * 12);
                for triangle in triangles {
                    for at in triangle {
                        let index = list.get(at as usize).ok_or_else(|| {
                            format!("assembling {primitive:?}: index {at} is past the list")
                        })?;
                        bytes.extend_from_slice(&index.to_le_bytes());
                    }
                }
                Some((Arc::from(bytes), -(index.lowest as i32)))
            }
            // Sequential: triples are vertex ordinals, already relative to the upload, so the
            // list depends only on the primitive and the count.
            (Some(primitive), None) => {
                let key = (primitive, engine.last_draw.count);
                let bytes = match self.sequential_indices.get(&key) {
                    Some(bytes) => bytes.clone(),
                    None => {
                        let bytes: Arc<[u8]> = switch_core::gpu::raster::assemble(primitive, key.1)
                            .into_iter()
                            .flatten()
                            .flat_map(u32::to_le_bytes)
                            .collect();
                        if self.sequential_indices.len() >= SEQUENTIAL_INDEX_LISTS {
                            self.sequential_indices.clear();
                        }
                        self.sequential_indices.insert(key, bytes.clone());
                        bytes
                    }
                };
                Some((bytes, 0))
            }
            (None, _) => None,
        };

        let mut globals = self.global_uploads(&vs_layout, ShaderStage::VertexB, &uploads, ctx)?;
        globals.extend(self.global_uploads(&fs_layout, ShaderStage::Fragment, &uploads, ctx)?);
        // WebGPU refuses a buffer bound read-only where the same draw writes it, so a fragment
        // read of one binds it writable too, and a vertex read, which cannot, reads a copy.
        let written: Vec<wgpu::Buffer> = globals
            .iter()
            .filter(|g| g.writes)
            .filter_map(|g| g.held.clone())
            .collect();
        for global in &mut globals {
            let shares = global.held.as_ref().is_some_and(|h| written.contains(h));
            if !shares || global.writes {
                continue;
            }
            if global.stage != Some(ShaderStage::Fragment) {
                global.snapshot = true;
                continue;
            }
            global.writes = true;
            fs_layout.globals[global.slot as usize].written = true;
        }
        for global in &globals {
            let layout = match global.stage {
                Some(ShaderStage::Fragment) => &mut fs_layout,
                _ => &mut vs_layout,
            };
            layout.globals[global.slot as usize].base = global.base;
        }

        if switch_core::trace::enabled(switch_core::trace::Trace::GpuTex) {
            // Trace what the draw renders into.
            switch_core::traceln!(
                "[gpu-draw] colour={} depth={} state={:?} viewport={:?} scissor={:?} \
                 topology={:?} call={:?} vertex_buffers={} cull={:?} front={:?} buffers={:?}",
                color.map_or("none".to_string(), |c| format!(
                    "{:#x} {:?} {}x{}",
                    c.addr, c.format, c.width, c.height
                )),
                depth.map_or("none".to_string(), |d| format!("{:#x}", d.addr)),
                state.target,
                state.viewport,
                state.scissor,
                state.topology,
                engine.last_draw,
                state.vertex_buffers.len(),
                state.cull,
                state.front_face,
                state.vertex_buffers
            );
            // A buffer over a held surface would read stale memory too.
            let held_at = |addr: u64| {
                self.space
                    .held
                    .values()
                    .find(|h| (h.target.addr..h.target.addr + h.target.len()).contains(&addr))
                    .map(|h| h.target.addr)
            };
            for c in &uploads.constants {
                if let Some((addr, _)) = engine.bound_constbuf(c.stage, c.bank) {
                    if let Some(surface) = held_at(addr) {
                        switch_core::traceln!(
                            "[gpu-tex] constant bank {} of {:?} at {addr:#x} is inside the held \
                             surface at {surface:#x}",
                            c.bank,
                            c.stage
                        );
                    }
                }
            }
            for t in &uploads.textures {
                switch_core::traceln!(
                    "[gpu-tex] {:?} {:?} {}x{} swizzle={:?} sampler={:?} addr={:#x}{}",
                    t.slot,
                    t.format,
                    t.width,
                    t.height,
                    t.swizzle,
                    t.sampler,
                    t.key.addr,
                    // Sampling a held surface would read stale guest memory.
                    match self.space.held.values().find(|h| {
                        (h.target.addr..h.target.addr + h.target.len()).contains(&t.key.addr)
                    }) {
                        Some(h) if h.target.addr == t.key.addr => {
                            " (held on the device)".to_string()
                        }
                        Some(h) => format!(" (inside the held surface at {:#x})", h.target.addr),
                        None if self
                            .space
                            .pending
                            .iter()
                            .any(|p| p.target().is_some_and(|s| s.addr == t.key.addr)) =>
                        {
                            " (being read back)".to_string()
                        }
                        None => String::new(),
                    }
                );
            }
        }
        Ok(Prepared {
            state,
            render,
            color,
            extra_colors,
            color_scratch,
            depth,
            vs,
            fs,
            serials,
            vs_layout,
            fs_layout,
            uploads,
            constant_patches,
            globals,
            count: match &assembled {
                Some((indices, _)) => (indices.len() / 4) as u32,
                None => engine.last_draw.count,
            },
            assembled,
            instance: engine.instance_id(),
            images,
        })
    }

    /// The images a draw's fragment stage stores to, held as storage textures, with their
    /// formats filled into `fs_layout`. WebGPU lets no vertex stage store to one.
    fn draw_images(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        vs_layout: &Layout,
        fs_layout: &mut Layout,
        drawn: &[Target],
    ) -> std::result::Result<Vec<crate::DrawImage>, String> {
        if !vs_layout.images.is_empty() {
            return Err("a vertex stage stores to an image, which WebGPU cannot".into());
        }
        if fs_layout.images.is_empty() {
            return Ok(Vec::new());
        }
        let bank = |bank: u8| engine.bound_constbuf(ShaderStage::Fragment, u32::from(bank));
        let pools = TexturePools {
            stage: ShaderStage::Fragment,
            tex_cb_index: engine.tex_cb_index(),
            header_pool: engine.tex_header_pool(),
            sampler_pool: engine.tex_sampler_pool(),
            bank: &bank,
        };
        let targets = self.storage_images(fs_layout, pools, ctx)?;
        // A texture bound for storage cannot also be an attachment of the pass.
        for image in targets.iter().flatten() {
            for surface in drawn {
                if spans_overlap(&surface_span(surface), &surface_span(image)) {
                    let state = Pipeline::of(engine).map_err(|e| e.to_string())?;
                    let what = if surface.depth.is_some() {
                        "depth"
                    } else {
                        "colour"
                    };
                    return Err(format!(
                        "stores to an image at {:#x} ({}x{}) over the {what} target at {:#x} \
                         ({}x{}); colour writes {:?}, depth {:?}",
                        image.addr,
                        image.width,
                        image.height,
                        surface.addr,
                        surface.width,
                        surface.height,
                        state.target.map(|t| t.write_mask),
                        state.depth
                    ));
                }
            }
        }
        let textures = self.storage_textures(&targets, fs_layout, ctx)?;
        Ok(targets.into_iter().zip(textures).collect())
    }

    /// What a stage's texture bindings take from the descriptors: the swizzle, which WebGPU has
    /// no per-texture form of, whether the image is integer, and how a tall one is folded.
    pub(super) fn fill_texture_bindings(
        &self,
        layout: &mut Layout,
        stage: ShaderStage,
        textures: &[switch_core::gpu::upload::TextureUpload],
    ) -> std::result::Result<(), String> {
        for binding in &mut layout.textures {
            let Some(upload) = textures
                .iter()
                .find(|t| t.stage == stage && t.slot == binding.slot)
            else {
                continue;
            };
            binding.swizzle = upload.swizzle;
            binding.uint = upload.format.is_uint();
            binding.shrink = upload.shrink;
            if binding.uint && upload.shrink > 0 {
                return Err("an integer image past the upload cap, read as a smaller level".into());
            }
            if binding.uint {
                // Read texel by texel, the raw words through the swizzle.
                if binding.compare {
                    return Err("an integer image sampled through a comparison".into());
                }
                continue;
            }
            binding.fold = self.fold(upload, binding)?;
        }
        Ok(())
    }

    /// Past the device's sampler limit, point each texture at the first earlier one bound with
    /// the same sampler, so only distinct samplers are bound. Under it nothing is shared, so a
    /// module does not change with the samplers its draws happen to use.
    pub(super) fn share_samplers(
        &self,
        layout: &mut Layout,
        stage: ShaderStage,
        textures: &[switch_core::gpu::upload::TextureUpload],
    ) {
        if layout.textures.len() <= self.device_limits.max_samplers_per_shader_stage as usize {
            return;
        }
        let keys: Vec<_> = layout
            .textures
            .iter()
            .map(|binding| {
                textures
                    .iter()
                    .find(|t| t.stage == stage && t.slot == binding.slot)
                    .map(|upload| {
                        let folded = binding.fold.is_some();
                        let key =
                            crate::bind::sampler_key(upload, binding.compare, binding.uint, folded);
                        // The binding type follows from these, so they must agree too.
                        (key, binding.compare, binding.uint)
                    })
            })
            .collect();
        for index in 0..layout.textures.len() {
            let Some(key) = keys[index] else {
                continue;
            };
            layout.textures[index].sampler = (0..index)
                .find(|&earlier| {
                    keys[earlier] == Some(key) && layout.textures[earlier].sampler.is_none()
                })
                .map(|earlier| earlier as u32);
        }
    }

    /// How a 2D image taller than the device allows is folded into layers, if it is one.
    /// Folding keeps two rows filtered together in one tile, repeating or not, but not a mirror.
    fn fold(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
        binding: &wgsl::TextureBinding,
    ) -> std::result::Result<Option<wgsl::Fold>, String> {
        use switch_core::gpu::shader::isa::TexDim;
        use switch_core::gpu::texture::Wrap;
        let limits = self.device_limits.clone();
        let plain_2d = matches!(binding.dim, TexDim::T2d | TexDim::T2dArray)
            && !binding.compare
            && upload.layers <= 1
            // Compressed rows are blocks, which a texel-row tile would cut through.
            && upload.rows == upload.height
            && upload.width <= limits.max_texture_dimension_2d;
        if !plain_2d {
            return Ok(None);
        }
        let Some(fold) = wgsl::Fold::of(
            upload.height,
            limits.max_texture_dimension_2d,
            limits.max_texture_array_layers,
            upload.sampler.wrap_v == Wrap::Repeat,
        ) else {
            return Ok(None);
        };
        if upload.sampler.wrap_v == Wrap::Mirror {
            return Err(format!(
                "a {}x{} image, taller than the device allows, sampled with a mirrored v wrap \
                 that folding it into layers cannot keep",
                upload.width, upload.height
            ));
        }
        Ok(Some(fold))
    }

    fn translate(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        stage: ShaderStage,
    ) -> std::result::Result<(Arc<Translation>, u64), String> {
        let Some(binding) = engine.program(stage) else {
            if stage == ShaderStage::Fragment {
                // Always the same program, so always the same serial.
                if let Some(translation) = &self.no_fragment {
                    return Ok((translation.clone(), 0));
                }
                let program = switch_core::gpu::shader::no_fragment_program();
                let translation = Arc::new(
                    wgsl::translate_for(&Compiled::new(&program), self.wgsl_caps(false))
                        .map_err(|e| e.to_string())?,
                );
                self.no_fragment = Some(translation.clone());
                return Ok((translation, 0));
            }
            return Err(format!("no {stage:?} program"));
        };
        self.translate_program(
            (binding.addr, Some(stage)),
            &|bank: u8| engine.bound_constbuf(stage, u32::from(bank)),
            ctx,
        )
    }

    /// The program at `key`'s address, translated or from the cache; `bank` resolves the
    /// constant banks a `brx` jump table is read from.
    pub(super) fn translate_program(
        &mut self,
        key: crate::ShaderKey,
        bank: &dyn Fn(u8) -> Option<(u64, u32)>,
        ctx: &ExecCtx,
    ) -> std::result::Result<(Arc<Translation>, u64), String> {
        // Validate hits: a `brx` jump table read from a constant buffer may have changed.
        if let Some(cached) = self.space.shader_cache.get(&key) {
            if cached.reads.constants_unchanged(ctx) {
                self.shader_hits += 1;
                return Ok((cached.translation.clone(), cached.serial));
            }
        }
        self.shader_misses += 1;
        let (program, reads) =
            switch_core::gpu::shader::decode_program_from_memory_recording(ctx, key.0, bank)
                .map_err(|e| format!("{e:?}"))?;
        let translation = Arc::new(
            wgsl::translate_for(&Compiled::new(&program), self.wgsl_caps(key.1.is_none()))
                .map_err(|e| e.to_string())?,
        );
        // Watch the CPU page behind each virtual page the decode read.
        for &page in &reads.pages {
            if let Some((cpu, _)) = ctx.vmm.translate(page << PAGE_BITS) {
                self.shader_to_watch.push(((cpu >> PAGE_BITS) as u32, key));
            }
        }
        // Whole-cache eviction, as for textures.
        if self.space.shader_cache.len() >= SHADER_CACHE_ENTRIES {
            self.space.shader_cache.clear();
            self.space.shader_pages.clear();
        }
        self.translations += 1;
        let serial = self.translations;
        self.space.shader_cache.insert(
            key,
            CachedShader {
                translation: translation.clone(),
                reads,
                serial,
            },
        );
        Ok((translation, serial))
    }

    /// What the translation of a draw stage's program, or a `compute` one's, may use.
    fn wgsl_caps(&self, compute: bool) -> wgsl::Caps {
        wgsl::Caps {
            subgroups: self.features().contains(wgpu::Features::SUBGROUP),
            // A browser wants the directive; naga rejects it.
            subgroup_enable: cfg!(target_arch = "wasm32"),
            warp: compute && self.warp_subgroups,
        }
    }

    /// Drop every translation a written page held program words for.
    pub(super) fn evict_shaders(&mut self, pages: &[u32]) {
        for page in pages {
            let Some(keys) = self.space.shader_pages.remove(page) else {
                continue;
            };
            for key in keys {
                self.space.shader_cache.remove(&key);
            }
        }
    }
}

/// The surfaces colour outputs 1 onward are attached to: each one bound, written by at least
/// one channel, and named by the fragment program's output map. A headerless program writes
/// output 0 alone.
fn extra_attachments(
    engine: &Engine3D,
    state: &Pipeline,
    fs: &Translation,
) -> std::result::Result<Vec<Option<Target>>, String> {
    if engine.program(ShaderStage::Fragment).is_none() {
        return Ok(Vec::new());
    }
    let header = fs.header.filter(|h| h.writes_any_color());
    let surfaces = Targets::extra_colors(engine).map_err(|e| format!("{e:?}"))?;
    Ok(surfaces
        .into_iter()
        .zip(&state.extra_targets)
        .enumerate()
        .map(|(index, (surface, target))| {
            let written = header.is_some_and(|h| h.writes_target(index as u32 + 1));
            let unmasked = target.is_some_and(|t| t.write_mask.contains(&true));
            surface.filter(|_| written && unmasked)
        })
        .collect())
}

/// The most levels a `tlds` naming its level binds; each is a texture binding of its own.
const MAX_FETCH_LEVELS: u32 = 16;

/// Bind every level of each texture a `stage` program fetches from a level register.
pub(super) fn bind_fetch_levels(
    engine: &Engine3D,
    ctx: &ExecCtx,
    layout: &mut Layout,
    stage: ShaderStage,
) -> std::result::Result<(), String> {
    if layout.fetch_levels.is_empty() {
        return Ok(());
    }
    let bank = |bank: u8| engine.bound_constbuf(stage, u32::from(bank));
    let pools = TexturePools {
        stage,
        tex_cb_index: engine.tex_cb_index(),
        header_pool: engine.tex_header_pool(),
        sampler_pool: engine.tex_sampler_pool(),
        bank: &bank,
    };
    bind_levels_from(pools, ctx, layout)
}

/// [`bind_fetch_levels`] against `pools`, for draws and launches alike.
pub(super) fn bind_levels_from(
    pools: TexturePools<'_>,
    ctx: &ExecCtx,
    layout: &mut Layout,
) -> std::result::Result<(), String> {
    let bases: Vec<TextureSlot> = layout.fetch_levels.iter().map(|&(slot, _)| slot).collect();
    for base in bases {
        let levels = switch_core::gpu::upload::texture_levels(pools, ctx, base)
            .map_err(|e| format!("{e:?}"))?;
        if levels > MAX_FETCH_LEVELS {
            return Err(format!(
                "fetches from an image of {levels} levels, past the {MAX_FETCH_LEVELS} bound"
            ));
        }
        layout.bind_fetch_levels(base, levels);
    }
    Ok(())
}
