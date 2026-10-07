//! Checking and preparing a draw, and translating its shaders.

use crate::{CachedShader, Checked, Gpu, Prepared, Render, PAGE_BITS, SHADER_CACHE_ENTRIES};
use switch_core::gpu::engine::threed::{Engine3D, ShaderStage};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state, AttributeBase, Pipeline};
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Coverage, Layout, Stage, Translation};
use switch_core::gpu::texture::TextureSlot;
use switch_core::gpu::upload::{Banks, Target, Targets, Uploads};

impl Gpu {
    /// Decide whether a draw can run here, without uploading anything.
    pub(super) fn check(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<Checked, String> {
        let mut state = Pipeline::of(engine).map_err(|e| e.to_string())?;
        let targets = Targets::of(engine).map_err(|e| format!("{e:?}"))?;
        let color = targets.color;
        // Attach depth only for draws that test or write it.
        let uses_depth = state
            .depth
            .is_some_and(|d| d.write_enabled || d.compare != state::Compare::Always);
        let depth = targets.depth.filter(|_| uses_depth);
        // Attachments must match in size, so the pass covers the intersection. A larger depth surface is
        // cropped; a larger colour target draws into a scratch texture copied back afterwards.
        let mut depth = depth;
        let mut color_scratch = None;
        if let (Some(full_color), Some(full_depth)) = (color, depth) {
            let (cw, ch) = (full_color.width, full_color.height);
            let (dw, dh) = (full_depth.width, full_depth.height);
            if dw >= cw && dh >= ch {
                depth = Some(Target {
                    width: cw,
                    height: ch,
                    rows: ch,
                    ..full_depth
                });
            } else if dw <= cw && dh <= ch {
                color_scratch = Some((dw, dh));
                // The scissor is confined to the depth surface only for draws that reach it.
                let (pixels_x, pixels_y) = state.grid.pixels(dw, dh);
                state.scissor.x1 = state.scissor.x1.min(pixels_x);
                state.scissor.y1 = state.scissor.y1.min(pixels_y);
                state.scissor.x0 = state.scissor.x0.min(state.scissor.x1);
                state.scissor.y0 = state.scissor.y0.min(state.scissor.y1);
            } else {
                return Err(format!(
                    "a {cw}x{ch} colour target beside a {dw}x{dh} depth one, each larger one way"
                ));
            }
        }
        if color.is_none() && depth.is_none() {
            return Err("a draw into neither a colour nor a depth surface".into());
        }
        let render = self.route(&state, color, depth)?;
        if color_scratch.is_some() && matches!(render, Render::Companion(_)) {
            return Err(
                "a colour target larger than its depth surface, drawn through a multisample companion"
                    .into(),
            );
        }

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
        let (vs, fs) = (vs?, fs?);
        Ok(Checked {
            state,
            render,
            color,
            color_scratch,
            depth,
            vs,
            fs,
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
            color_scratch,
            depth,
            vs,
            fs,
        } = self.check(engine, ctx)?;

        let mut vs_layout = Layout::of(&vs, Stage::Vertex);
        let mut fs_layout = Layout::of(&fs, Stage::Fragment);
        // A depth-only pass has no colour output.
        fs_layout.targets = u32::from(color.is_some());
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
        // On the expanded route the shader applies the sample mask and alpha-to-coverage.
        if render == Render::Expanded {
            fs_layout.coverage = Some(Coverage {
                samples_x: state.grid.samples_x,
                samples_y: state.grid.samples_y,
                sample_of_slot: state.grid.sample_of_slot()[..state.samples as usize].to_vec(),
                sample_mask: state.sample_mask,
                alpha_to_coverage: state.alpha_to_coverage,
            });
        }
        // Integer attributes come from the draw's registers.
        vs_layout.integer_attributes = state
            .vertex_buffers
            .iter()
            .flat_map(|buffer| &buffer.attributes)
            .filter(|a| a.format.base() != AttributeBase::Float)
            .map(|a| (a.location as usize, a.format.base()))
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

        let mut slots: Vec<(ShaderStage, TextureSlot)> = Vec::new();
        slots.extend(
            vs.textures
                .iter()
                .map(|&(slot, _, _)| (ShaderStage::VertexB, slot)),
        );
        slots.extend(
            fs.textures
                .iter()
                .map(|&(slot, _, _)| (ShaderStage::Fragment, slot)),
        );
        let mut banks: Vec<(ShaderStage, u32)> = Vec::new();
        banks.extend(
            vs.const_banks
                .iter()
                .map(|&b| (ShaderStage::VertexB, u32::from(b))),
        );
        banks.extend(
            fs.const_banks
                .iter()
                .map(|&b| (ShaderStage::Fragment, u32::from(b))),
        );
        // Taken out of `self` because the closure holds it while `timed!` borrows `self`.
        let cache = std::mem::take(&mut self.texture_cache);
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
        self.texture_cache = cache;
        self.texture_hits += hits;
        let uploads = uploads.map_err(|e| format!("{e:?}"))?;
        self.uploaded.add_but_textures(&uploads);
        for upload in &uploads.textures {
            if !self.texture_cache.contains_key(&upload.key) {
                self.uploaded.add_texture(upload.bytes.len());
                self.to_remember
                    .push((upload.key, upload.bytes.clone(), upload.source_len));
            }
        }

        // The texture swizzle is in the descriptor, and WebGPU has no per-texture swizzle.
        for (layout, stage) in [
            (&mut vs_layout, ShaderStage::VertexB),
            (&mut fs_layout, ShaderStage::Fragment),
        ] {
            for binding in &mut layout.textures {
                if let Some(upload) = uploads
                    .textures
                    .iter()
                    .find(|t| t.stage == stage && t.slot == binding.slot)
                {
                    binding.swizzle = upload.swizzle;
                }
            }
        }

        // Fans, quad strips and polygons become triangle lists.
        let assembled = match state.expand {
            Some(primitive) => {
                let triangles =
                    switch_core::gpu::raster::assemble(primitive, engine.last_draw.count);
                let mut indices = Vec::with_capacity(triangles.len() * 3);
                match &uploads.index {
                    // Indexed: triples index the index list; base vertex is the lowest index.
                    Some(index) => {
                        let list = index.indices();
                        for triangle in triangles {
                            for at in triangle {
                                indices.push(*list.get(at as usize).ok_or_else(|| {
                                    format!("assembling {primitive:?}: index {at} is past the list")
                                })?);
                            }
                        }
                        Some((indices, -(index.lowest as i32)))
                    }
                    // Sequential: triples are vertex ordinals, already relative to the upload.
                    None => {
                        for triangle in triangles {
                            indices.extend_from_slice(&triangle);
                        }
                        Some((indices, 0))
                    }
                }
            }
            None => None,
        };

        let mut globals = self.global_uploads(&vs_layout, ShaderStage::VertexB, &uploads, ctx)?;
        globals.extend(self.global_uploads(&fs_layout, ShaderStage::Fragment, &uploads, ctx)?);

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
                self.held
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
                    match self.held.values().find(|h| {
                        (h.target.addr..h.target.addr + h.target.len()).contains(&t.key.addr)
                    }) {
                        Some(h) if h.target.addr == t.key.addr => {
                            " (held on the device)".to_string()
                        }
                        Some(h) => format!(" (inside the held surface at {:#x})", h.target.addr),
                        None if self.evicted.iter().any(|h| h.target.addr == t.key.addr) => {
                            " (evicted, not yet written back)".to_string()
                        }
                        None if self.pending.iter().any(|p| p.target.addr == t.key.addr) => {
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
            color_scratch,
            depth,
            vs,
            fs,
            vs_layout,
            fs_layout,
            uploads,
            globals,
            count: match &assembled {
                Some((indices, _)) => indices.len() as u32,
                None => engine.last_draw.count,
            },
            assembled,
            instance: engine.instance_id(),
        })
    }

    fn translate(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        stage: ShaderStage,
    ) -> std::result::Result<Translation, String> {
        let binding = engine
            .program(stage)
            .ok_or_else(|| format!("no {stage:?} program"))?;
        let key = (binding.addr, stage);
        // Validate hits: a `brx` jump table read from a constant buffer may have changed.
        if let Some(cached) = self.shader_cache.get(&key) {
            if cached.reads.constants_unchanged(ctx) {
                self.shader_hits += 1;
                return Ok(cached.translation.clone());
            }
        }
        self.shader_misses += 1;
        let (program, reads) = switch_core::gpu::shader::decode_program_from_memory_recording(
            ctx,
            binding.addr,
            &|bank: u8| engine.bound_constbuf(stage, u32::from(bank)),
        )
        .map_err(|e| format!("{e:?}"))?;
        let caps = wgsl::Caps {
            subgroups: self.features().contains(wgpu::Features::SUBGROUP),
            // A browser wants the directive; naga rejects it.
            subgroup_enable: cfg!(target_arch = "wasm32"),
        };
        let translation =
            wgsl::translate_for(&Compiled::new(&program), caps).map_err(|e| e.to_string())?;
        // Watch the CPU page behind each virtual page the decode read.
        for &page in &reads.pages {
            if let Some((cpu, _)) = ctx.vmm.translate(page << PAGE_BITS) {
                self.shader_to_watch
                    .push(((u64::from(cpu) >> PAGE_BITS) as u32, key));
            }
        }
        // Whole-cache eviction, as for textures.
        if self.shader_cache.len() >= SHADER_CACHE_ENTRIES {
            self.shader_cache.clear();
            self.shader_pages.clear();
        }
        self.shader_cache.insert(
            key,
            CachedShader {
                translation: translation.clone(),
                reads,
            },
        );
        Ok(translation)
    }

    /// Drop every translation a written page held program words for.
    pub(super) fn evict_shaders(&mut self, pages: &[u32]) {
        for page in pages {
            let Some(keys) = self.shader_pages.remove(page) else {
                continue;
            };
            for key in keys {
                self.shader_cache.remove(&key);
            }
        }
    }
}
