//! The draw pass: pipeline, encoding, and attachments.

use crate::convert::{
    blend, compare, depth_texture_format, held_attachment_format, index_format, topology,
    vertex_format, write_mask, Widen,
};
use crate::readback::Scratch;
use crate::{
    can_blend, dump_wgsl, fit_vertex_buffers, Bound, Feed, Gpu, PipelineKey, Prepared, Render,
    Shape, ABSENT_ATTRIBUTE, ATTRIBUTE_DEFAULTS, DEFAULT_ATTRIBUTE,
};
use switch_core::gpu::engine::threed::ShaderStage;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state};
use switch_core::gpu::shader::wgsl::{self, Stage};

impl Gpu {
    /// Build (or reuse) the pipeline and run the pass.
    pub(super) fn render(
        &mut self,
        p: &Prepared,
        ctx: &mut ExecCtx,
    ) -> std::result::Result<(), String> {
        // No vertices, so nothing is uploaded and nothing is drawn.
        if p.count == 0 {
            return Ok(());
        }
        let target = match p.color {
            Some(color) => Some(
                held_attachment_format(self.features(), color.format)
                    .map_err(|e| format!("{e:?}"))?,
            ),
            None => None,
        };
        let target_format = target.map(|(format, _)| format);
        // A blend the device cannot do must fall back here; a rejected pipeline silently draws nothing.
        if let Some((format, widen)) = target {
            let blends = p.state.target.is_some_and(|t| t.blend.is_some());
            if blends && widen == Widen::Unorm16 {
                return Err(format!(
                    "blending into a 16-bit unorm target held as {format:?}, which would not \
                     clamp as the surface does"
                ));
            }
            if blends && !can_blend(format, self.features()) {
                return Err(format!(
                    "blending into a {format:?} target, which this device cannot blend"
                ));
            }
        }
        let depth_format = p
            .depth
            .and_then(|d| d.depth_kind())
            .map(depth_texture_format);
        // Sample mask and alpha-to-coverage are the device's only on the companion route.
        let multisample = match p.render {
            Render::Companion(Shape::Multisampled(count)) => wgpu::MultisampleState {
                count,
                mask: u64::from(p.state.sample_mask),
                alpha_to_coverage_enabled: p.state.alpha_to_coverage,
            },
            _ => wgpu::MultisampleState::default(),
        };
        if let Some(e) = self.device_error() {
            return Err(format!("the device rejected an earlier draw: {e}"));
        }
        let ((vs_key, vs_module), (fs_key, fs_module)) = timed!(self, modules, {
            if switch_core::dump::wants(switch_core::dump::Artefact::DeviceWgsl) {
                let vs_source = wgsl::module(&p.vs, Stage::Vertex, &p.vs_layout);
                let fs_source = wgsl::module(&p.fs, Stage::Fragment, &p.fs_layout);
                if let (Ok(vs), Ok(fs)) = (vs_source, fs_source) {
                    dump_wgsl(&vs, &fs);
                }
            }
            let [vs_serial, fs_serial] = p.serials;
            let vs = self.built_module("vertex", (&p.vs, vs_serial), Stage::Vertex, &p.vs_layout);
            let fs = self.built_module(
                "fragment",
                (&p.fs, fs_serial),
                Stage::Fragment,
                &p.fs_layout,
            );
            vs.and_then(|vs| fs.map(|fs| (vs, fs)))
        })?;

        // Vertex buffers the shader reads; the stride travels with each buffer.
        let mut feeds: Vec<Feed> = Vec::new();
        for buffer in &p.state.vertex_buffers {
            let attributes: Vec<wgpu::VertexAttribute> = buffer
                .attributes
                .iter()
                .filter(|a| p.vs_layout.attributes.contains(&(a.location as usize)))
                .map(|a| wgpu::VertexAttribute {
                    format: vertex_format(a.format),
                    offset: u64::from(a.offset),
                    shader_location: a.location,
                })
                .collect();
            if attributes.is_empty() {
                continue;
            }
            let upload = p
                .uploads
                .vertex
                .iter()
                .find(|v| v.array == buffer.index)
                .ok_or("a bound vertex array with no bytes")?;
            // An instanced array uploads only this instance's element, so its stride is zero.
            let stride = match buffer.step {
                state::StepMode::Instance => 0,
                state::StepMode::Vertex => u64::from(buffer.stride),
            };
            // Metal drops a vertex whose stride runs past the end of the buffer.
            let mut bytes = std::borrow::Cow::Borrowed(&upload.bytes[..]);
            let whole = bytes.len().next_multiple_of(stride.max(1) as usize);
            if whole != bytes.len() {
                bytes.to_mut().resize(whole, 0);
            }
            let step = match buffer.step {
                state::StepMode::Instance => wgpu::VertexStepMode::Instance,
                state::StepMode::Vertex => wgpu::VertexStepMode::Vertex,
            };
            let read = upload.addr..upload.addr + upload.inside;
            for (bytes, attributes, shift, part_stride) in
                crate::within_stride(bytes, attributes, stride)
            {
                let (bytes, widened) = crate::aligned_stride(bytes, part_stride);
                // Widening moves every element but the first.
                let origin = shift
                    .filter(|_| widened == stride || bytes.len() <= stride as usize)
                    .map(|shift| (upload.addr + shift, upload.inside.saturating_sub(shift)));
                feeds.push(Feed {
                    bytes,
                    origin,
                    read: read.clone(),
                    attributes,
                    stride: widened,
                    step,
                });
            }
        }
        // Unbound locations read a constant buffer of the two default vectors.
        let fed: Vec<usize> = feeds
            .iter()
            .flat_map(|f| f.attributes.iter().map(|a| a.shader_location as usize))
            .collect();
        let unfed: Vec<wgpu::VertexAttribute> = p
            .vs_layout
            .attributes
            .iter()
            .filter(|l| !fed.contains(l))
            .map(|&location| wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: if p.state.fixed_attributes.contains(&(location as u32)) {
                    DEFAULT_ATTRIBUTE
                } else {
                    ABSENT_ATTRIBUTE
                },
                shader_location: location as u32,
            })
            .collect();
        if !unfed.is_empty() {
            feeds.push(Feed {
                bytes: std::borrow::Cow::Borrowed(&ATTRIBUTE_DEFAULTS),
                origin: None,
                read: 0..0,
                attributes: unfed,
                stride: 0,
                step: wgpu::VertexStepMode::Vertex,
            });
        }
        // What compute launches wrote on the device is copied over the stale guest bytes.
        let patches: Vec<Vec<crate::Patch>> = feeds
            .iter()
            .map(|feed| self.vertex_patches(feed))
            .collect::<std::result::Result<_, String>>()?;
        // Folding rearranges bytes on the CPU, so arrays with patches keep buffers of their own
        // and only the rest are folded into what the limit leaves.
        let (patched, plain): (Vec<_>, Vec<_>) = feeds
            .into_iter()
            .zip(patches)
            .partition(|(_, patches)| !patches.is_empty());
        let room = (self.vertex_buffer_limit as usize).saturating_sub(patched.len());
        let plain: Vec<_> = plain.into_iter().map(|(feed, _)| feed).collect();
        let folded = fit_vertex_buffers(plain, room as u32);
        if patched.len() + folded.len() > self.vertex_buffer_limit as usize {
            return Err("folds a vertex array a compute launch wrote on the device".into());
        }
        let feeds = patched
            .into_iter()
            .chain(folded.into_iter().map(|feed| (feed, Vec::new())));
        let bound: Vec<Bound> = feeds
            .map(|(f, patches)| {
                // The patches are copied into it, so it is this draw's alone.
                let buffer = if patches.is_empty() {
                    self.shared_buffer("vertex", &f.bytes, wgpu::BufferUsages::VERTEX)
                } else {
                    self.buffer("vertex", &f.bytes, wgpu::BufferUsages::VERTEX)
                };
                // Kept so the next instances can be laid out beside it; see `Gpu::emit_pending`.
                let element = (f.step == wgpu::VertexStepMode::Instance && patches.is_empty())
                    .then(|| f.bytes.to_vec());
                Bound {
                    buffer,
                    patches,
                    attributes: f.attributes,
                    stride: f.stride,
                    step: f.step,
                    element,
                }
            })
            .collect();
        let (vs_group_layout, vs_group, vs_offsets, vs_constants) =
            timed!(self, bind, self.bind_group(p, ShaderStage::VertexB, 0))?;
        let (fs_group_layout, fs_group, fs_offsets, fs_constants) =
            timed!(self, bind, self.bind_group(p, ShaderStage::Fragment, 1))?;
        let groups = Groups {
            vs: vs_group,
            fs: fs_group,
            offsets: [vs_offsets, fs_offsets],
            patched: vs_constants.into_iter().chain(fs_constants).collect(),
        };
        // An assembled topology is always `u32` indices.
        let draw_index_format = match &p.assembled {
            Some(_) => Some(wgpu::IndexFormat::Uint32),
            None => p.uploads.index.as_ref().map(|i| index_format(i.format)),
        };
        // WebGPU requires the strip index format on strip pipelines and forbids it otherwise.
        let strip_index_format = match p.state.topology {
            state::Topology::LineStrip | state::Topology::TriangleStrip => draw_index_format,
            _ => None,
        };
        let recipe = Recipe {
            key: PipelineKey {
                vs: vs_key,
                fs: fs_key,
                target: target_format,
                depth: depth_format
                    .zip(p.state.depth)
                    .map(|(format, d)| (format, d.write_enabled, d.compare)),
                samples: multisample.count,
                sample_mask: multisample.mask,
                alpha_to_coverage: multisample.alpha_to_coverage_enabled,
                sample_pass: None,
                blend: p.state.target.and_then(|t| t.blend),
                write_mask: p.state.target.map_or([true; 4], |t| t.write_mask),
                topology: p.state.topology,
                strip_index_format,
                front_face: p.state.front_face,
                cull: p.state.cull,
                buffers: Vec::new(),
            },
            modules: [vs_module, fs_module],
            group_layouts: [vs_group_layout, fs_group_layout],
            primitive: wgpu::PrimitiveState {
                topology: topology(p.state.topology),
                strip_index_format,
                front_face: match p.state.front_face {
                    state::FrontFace::Ccw => wgpu::FrontFace::Ccw,
                    state::FrontFace::Cw => wgpu::FrontFace::Cw,
                },
                cull_mode: match p.state.cull {
                    state::Cull::None => None,
                    state::Cull::Front => Some(wgpu::Face::Front),
                    state::Cull::Back => Some(wgpu::Face::Back),
                },
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            multisample,
            // Empty for a depth-only pass.
            colour_targets: target_format
                .map(|format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: p.state.target.and_then(|t| t.blend).map(blend),
                        write_mask: p
                            .state
                            .target
                            .map_or(wgpu::ColorWrites::ALL, |t| write_mask(t.write_mask)),
                    })
                })
                .into_iter()
                .collect(),
            buffers: Vec::new(),
        }
        .with_buffers(
            bound
                .iter()
                .map(|b| (b.stride, b.step, b.attributes.clone()))
                .collect(),
        );
        // A draw whose samples sit off their texel centres is drawn once per sample.
        let passes: Vec<Option<u32>> =
            if p.render == Render::Expanded && !p.state.grid.samples_at_texel_centres() {
                (0..p.state.samples).map(Some).collect()
            } else {
                vec![None]
            };
        let pipelines: Vec<wgpu::RenderPipeline> = passes
            .into_iter()
            .map(|sample_pass| self.pipeline(&recipe, sample_pass))
            .collect();
        self.encode(p, recipe, &pipelines, &groups, &bound, ctx)
    }

    /// The pipeline `recipe` describes, drawing only `sample_pass` when given; built once.
    pub(super) fn pipeline(
        &mut self,
        recipe: &Recipe,
        sample_pass: Option<u32>,
    ) -> wgpu::RenderPipeline {
        // A recipe's own key is the single-pass one, which most draws look up without a copy.
        let key = match sample_pass {
            None => std::borrow::Cow::Borrowed(&recipe.key),
            Some(_) => std::borrow::Cow::Owned(PipelineKey {
                sample_pass,
                ..recipe.key.clone()
            }),
        };
        if let Some(pipeline) = self.pipelines.get(&*key) {
            return pipeline.clone();
        }
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("draw"),
                bind_group_layouts: &[
                    Some(&recipe.group_layouts[0]),
                    Some(&recipe.group_layouts[1]),
                ],
                immediate_size: 0,
            });
        let buffers: Vec<Option<wgpu::VertexBufferLayout>> = recipe
            .buffers
            .iter()
            .map(|(stride, step, attributes)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: *stride,
                    step_mode: *step,
                    attributes,
                })
            })
            .collect();
        let constants: Vec<(&str, f64)> = sample_pass
            .map(|k| ("pass_sample", f64::from(k)))
            .into_iter()
            .collect();
        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &recipe.modules[0],
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: recipe.primitive,
            depth_stencil: key.depth.map(|(format, write_enabled, test)| {
                wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(write_enabled),
                    depth_compare: Some(compare(test)),
                    // Neither renderer tests stencil.
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }
            }),
            multisample: recipe.multisample,
            fragment: Some(wgpu::FragmentState {
                module: &recipe.modules[1],
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                targets: &recipe.colour_targets,
            }),
            multiview_mask: None,
            cache: None,
        };
        let pipeline = timed!(
            self,
            pipeline,
            self.device.create_render_pipeline(&descriptor)
        );
        self.pipelines.insert(key.into_owned(), pipeline.clone());
        pipeline
    }

    /// Record and submit one draw against a built pipeline.
    fn encode(
        &mut self,
        p: &Prepared,
        recipe: Recipe,
        pipelines: &[wgpu::RenderPipeline],
        groups: &Groups,
        bound: &[Bound],
        ctx: &mut ExecCtx,
    ) -> std::result::Result<(), String> {
        // Held across the frame.
        let colour_target = match p.color {
            Some(color) => {
                self.hold(&color, ctx).map_err(|e| format!("{e:?}"))?;
                Some(self.attachment(p, color.addr, true)?)
            }
            None => None,
        };
        // See [`Prepared::color_scratch`].
        let scratch = match (p.color, p.color_scratch) {
            (Some(color), Some((width, height))) => {
                let held = self
                    .space
                    .held
                    .get(&color.addr)
                    .ok_or("the surface was not held")?
                    .texture
                    .clone();
                let size = wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                };
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("colour scratch"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: held.format(),
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("colour scratch in"),
                        });
                encoder.copy_texture_to_texture(
                    held.as_image_copy(),
                    texture.as_image_copy(),
                    size,
                );
                self.submit_with_batch(encoder);
                Some((held, texture, size))
            }
            _ => None,
        };
        let colour_target = match &scratch {
            Some((_, texture, _)) => Some(texture.clone()),
            None => colour_target,
        };
        let depth_target = match p.depth {
            Some(depth) => {
                self.hold(&depth, ctx).map_err(|e| format!("{e:?}"))?;
                // A depth test without writes leaves the surface clean.
                let writes = p.state.depth.is_some_and(|d| d.write_enabled);
                Some(self.attachment(p, depth.addr, writes)?)
            }
            None => None,
        };
        let index = match &p.assembled {
            // An assembled topology is always drawn indexed.
            Some((bytes, base)) => Some((
                self.shared_buffer("assembled", bytes, wgpu::BufferUsages::INDEX),
                wgpu::IndexFormat::Uint32,
                *base,
            )),
            None => p.uploads.index.as_ref().map(|index| {
                (
                    self.shared_buffer("index", &index.bytes, wgpu::BufferUsages::INDEX),
                    index_format(index.format),
                    -(index.lowest as i32),
                )
            }),
        };

        // Copies cannot be recorded inside a pass, so a draw needing one opens its own.
        let patched = bound.iter().any(|b| !b.patches.is_empty()) || !groups.patched.is_empty();
        // A pass is one usage scope: a buffer written in it cannot also be bound read-only.
        let held_globals: Vec<(wgpu::Buffer, bool)> = p
            .globals
            .iter()
            .filter(|g| !g.snapshot)
            .filter_map(|g| g.held.clone().map(|buffer| (buffer, g.writes)))
            .collect();
        // A pass is one usage scope, and an image stored to is in no other use within its own.
        let stores = !p.images.is_empty();
        let joins = !patched
            && !stores
            && scratch.is_none()
            && self.pass.as_ref().is_some_and(|open| {
                open.colour == colour_target
                    && open.depth == depth_target
                    && !held_globals.iter().any(|(buffer, writes)| {
                        open.globals
                            .iter()
                            .any(|(bound, was)| bound == buffer && was != writes)
                    })
            });
        if joins {
            self.passes_joined += 1;
            if let Some(open) = self.pass.as_mut() {
                for global in held_globals {
                    if !open.globals.contains(&global) {
                        open.globals.push(global);
                    }
                }
            }
        } else {
            self.end_pass();
            let mut encoder = self.batch.take().unwrap_or_else(|| {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("draws"),
                    })
            });
            for (_, layers) in &p.images {
                for layer in layers {
                    layer.copy_in(&mut encoder);
                }
            }
            let vertex_patches = bound.iter().map(|b| (&b.buffer, &b.patches));
            let group_patches = groups.patched.iter().map(|(b, p)| (b, p));
            for (buffer, patches) in vertex_patches.chain(group_patches) {
                for patch in patches {
                    encoder.copy_buffer_to_buffer(
                        &patch.source,
                        patch.from,
                        buffer,
                        patch.to,
                        patch.len,
                    );
                }
            }
            let colour_view = colour_target.as_ref().map(|t| self.view(t, None));
            let depth_view = depth_target.as_ref().map(|t| self.view(t, None));
            let colour: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_view
                .as_ref()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Loaded, never cleared: a clear is its own method.
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .into_iter()
                .collect();
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draws"),
                color_attachments: &colour,
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Kept open beside its encoder, which stays locked until `end_pass`.
            let pass = pass.forget_lifetime();
            self.batch = Some(encoder);
            self.pass = Some(crate::OpenPass {
                pass,
                colour: colour_target,
                depth: depth_target,
                globals: held_globals,
                set: crate::PassState::default(),
                pending: None,
            });
        }
        {
            // In pixels, except on the expanded route where the attachment is in texels.
            let (sx, sy) = match p.render {
                Render::Expanded => (p.state.grid.samples_x, p.state.grid.samples_y),
                _ => (1, 1),
            };
            let viewport = &p.state.viewport;
            let scissor = p.state.scissor;
            let scissor = [
                scissor.x0 * sx,
                scissor.y0 * sy,
                scissor.x1.saturating_sub(scissor.x0) * sx,
                scissor.y1.saturating_sub(scissor.y0) * sy,
            ];
            let blend_constant = p
                .state
                .target
                .and_then(|t| t.blend)
                .map(|_| p.state.blend_constant);
            let index_buffer = index.as_ref().map(|(buffer, format, _)| (buffer, *format));
            // One pipeline, or one per sample with the geometry moved onto it.
            let passes: Vec<(&wgpu::RenderPipeline, [f32; 6])> = pipelines
                .iter()
                .enumerate()
                .map(|(sample, pipeline)| {
                    let (shift_x, shift_y) = match pipelines.len() {
                        1 => (0.0, 0.0),
                        _ => p.state.grid.texel_shift(sample as u32),
                    };
                    let rect = [
                        viewport.x * sx as f32 + shift_x,
                        viewport.y * sy as f32 + shift_y,
                        viewport.width * sx as f32,
                        viewport.height * sy as f32,
                        viewport.min_depth.clamp(0.0, 1.0),
                        viewport.max_depth.clamp(0.0, 1.0),
                    ];
                    (pipeline, rect)
                })
                .collect();
            // One instance, numbered so `instance_index` matches `gl_InstanceID`.
            // One instance, numbered so `instance_index` matches `gl_InstanceID`.
            let mut call = crate::DrawCall {
                count: p.count,
                base_vertex: index.as_ref().map(|(_, _, base)| *base),
                instances: p.instance..p.instance + 1,
                instanced: Vec::new(),
                recipe: None,
            };
            // The next instance of the draw before, in the state it left, widens that draw, its
            // instanced arrays' elements laid out after the earlier instances'.
            let max_stride = u64::from(self.device_limits.max_vertex_buffer_array_stride);
            let mut instanced = Vec::new();
            let mut mergeable = passes.len() == 1;
            for (slot, b) in bound.iter().enumerate() {
                if b.step != wgpu::VertexStepMode::Instance {
                    continue;
                }
                let Some(element) = b.element.as_deref() else {
                    mergeable = false;
                    continue;
                };
                let stride = crate::instance_stride(element.len());
                mergeable &= !element.is_empty()
                    && stride <= max_stride
                    && b.attributes
                        .iter()
                        .all(|a| a.offset + a.format.size() <= stride);
                instanced.push((slot, element));
            }
            let merges = mergeable
                && self.pass.as_ref().is_some_and(|open| {
                    open.pending
                        .as_ref()
                        .is_some_and(|d| d.continued_by(&call, &instanced))
                        && open.set.holds(
                            [&groups.vs, &groups.fs],
                            &groups.offsets,
                            bound.iter().map(|b| {
                                (b.step == wgpu::VertexStepMode::Vertex).then_some(&b.buffer)
                            }),
                            scissor,
                            blend_constant,
                            index_buffer,
                            passes[0],
                        )
                });
            if merges {
                let open = self.pass.as_mut().expect("a pass was opened or joined");
                if let Some(pending) = open.pending.as_mut() {
                    pending.instances.end += 1;
                    for ((_, _, elements), (_, element)) in
                        pending.instanced.iter_mut().zip(&instanced)
                    {
                        elements.extend_from_slice(element);
                    }
                }
                self.instances_merged += 1;
            } else {
                self.emit_pending();
                if mergeable && !instanced.is_empty() {
                    call.instanced = instanced
                        .iter()
                        .map(|(slot, element)| (*slot, element.len(), element.to_vec()))
                        .collect();
                    call.recipe = Some(Box::new(recipe));
                }
                let open = self.pass.as_mut().expect("a pass was opened or joined");
                let (pass, set) = (&mut open.pass, &mut open.set);
                let mut skipped = 0u64;
                let mut fresh = |changed: bool| {
                    skipped += u64::from(!changed);
                    changed
                };
                for (slot, (group, offsets)) in [&groups.vs, &groups.fs]
                    .into_iter()
                    .zip(&groups.offsets)
                    .enumerate()
                {
                    let now = Some((group.clone(), offsets.clone()));
                    if fresh(set.groups[slot] != now) {
                        pass.set_bind_group(slot as u32, group, offsets);
                        set.groups[slot] = now;
                    }
                }
                for (slot, b) in bound.iter().enumerate() {
                    if set.vertex.len() <= slot {
                        set.vertex.resize(slot + 1, None);
                    }
                    if fresh(set.vertex[slot].as_ref() != Some(&b.buffer)) {
                        pass.set_vertex_buffer(slot as u32, b.buffer.slice(..));
                        set.vertex[slot] = Some(b.buffer.clone());
                    }
                }
                if fresh(set.scissor != Some(scissor)) {
                    pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
                    set.scissor = Some(scissor);
                }
                if let Some(constant) = blend_constant {
                    if fresh(set.blend_constant != Some(constant)) {
                        let [r, g, b, a] = constant;
                        pass.set_blend_constant(wgpu::Color {
                            r: f64::from(r),
                            g: f64::from(g),
                            b: f64::from(b),
                            a: f64::from(a),
                        });
                        set.blend_constant = Some(constant);
                    }
                }
                if let Some((buffer, format)) = index_buffer {
                    let now = Some((buffer.clone(), format));
                    if fresh(set.index != now) {
                        pass.set_index_buffer(buffer.slice(..), format);
                        set.index = now;
                    }
                }
                for &(pipeline, rect) in &passes {
                    if fresh(set.pipeline.as_ref() != Some(pipeline)) {
                        pass.set_pipeline(pipeline);
                        set.pipeline = Some(pipeline.clone());
                    }
                    if fresh(set.viewport != Some(rect)) {
                        pass.set_viewport(rect[0], rect[1], rect[2], rect[3], rect[4], rect[5]);
                        set.viewport = Some(rect);
                    }
                    if !mergeable {
                        call.record(pass);
                    }
                }
                self.pass_calls_skipped += skipped;
                if mergeable {
                    open.pending = Some(call);
                }
            }
        }
        if stores {
            self.end_pass();
            if let Some(encoder) = self.batch.as_mut() {
                for (_, layers) in &p.images {
                    for layer in layers {
                        layer.copy_out(encoder);
                    }
                }
            }
            for image in p.images.iter().flat_map(|(layers, _)| layers) {
                let held = self
                    .space
                    .held
                    .get_mut(&image.addr)
                    .ok_or("an image that is no longer held")?;
                held.dirty = true;
                held.writes += 1;
                held.overwritten.clear();
            }
        }
        if let Some((held, texture, size)) = &scratch {
            self.end_pass();
            if let Some(encoder) = self.batch.as_mut() {
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    held.as_image_copy(),
                    *size,
                );
            }
            self.scratch.push(Scratch::Texture(texture.clone()));
        }
        self.later_instances += u64::from(p.instance > 0);
        self.batched_draws += 1;
        if self.batched_draws >= crate::MAX_BATCHED_DRAWS {
            self.submit_batch();
        }
        Ok(())
    }

    /// The view a draw renders one of its surfaces through, with any companion in place.
    fn attachment(
        &mut self,
        p: &Prepared,
        addr: u64,
        writes: bool,
    ) -> std::result::Result<wgpu::Texture, String> {
        match p.render {
            Render::Companion(shape) => self.companion(addr, shape, p.state.grid)?,
            // Put back whatever a companion holds first.
            Render::Direct | Render::Expanded => self.resolve_companion(addr)?,
        }
        let held = self
            .space
            .held
            .get_mut(&addr)
            .ok_or("the surface was not held")?;
        held.dirty |= writes;
        held.writes += u64::from(writes);
        // What the device draws now is newer than what the guest wrote over the surface.
        if writes {
            held.overwritten.clear();
        }
        Ok(match &held.companion {
            Some(companion) => companion.texture.clone(),
            None => held.texture.clone(),
        })
    }

    /// End the render pass draws share, so the encoder can record something else.
    pub(super) fn end_pass(&mut self) {
        self.emit_pending();
        self.pass = None;
    }

    /// Record the draw the open pass holds back. Instances merged into it read their instanced
    /// arrays from one buffer of their elements, each at `instance_index` strides in, through a
    /// pipeline reading those arrays at that stride.
    pub(super) fn emit_pending(&mut self) {
        let Some(call) = self.pass.as_mut().and_then(|open| open.pending.take()) else {
            return;
        };
        if call.instances.len() > 1 {
            if let Some(recipe) = &call.recipe {
                let strides: Vec<(usize, u64)> = call
                    .instanced
                    .iter()
                    .map(|&(slot, len, _)| (slot, crate::instance_stride(len)))
                    .collect();
                let pipeline = self.pipeline(&recipe.with_strides(&strides), None);
                let buffers: Vec<(usize, wgpu::Buffer)> = call
                    .instanced
                    .iter()
                    .zip(&strides)
                    .map(|((slot, len, elements), &(_, stride))| {
                        let stride = stride as usize;
                        // The instances before the first are never read.
                        let mut bytes = vec![0u8; call.instances.start as usize * stride];
                        for element in elements.chunks_exact(*len) {
                            bytes.extend_from_slice(element);
                            bytes.resize(bytes.len() + stride - len, 0);
                        }
                        (
                            *slot,
                            self.buffer("instances", &bytes, wgpu::BufferUsages::VERTEX),
                        )
                    })
                    .collect();
                let open = self.pass.as_mut().expect("the pass the draw was held in");
                open.pass.set_pipeline(&pipeline);
                open.set.pipeline = Some(pipeline);
                for (slot, buffer) in buffers {
                    open.pass.set_vertex_buffer(slot as u32, buffer.slice(..));
                    open.set.vertex[slot] = Some(buffer);
                }
            }
        }
        if let Some(open) = self.pass.as_mut() {
            call.record(&mut open.pass);
        }
    }
}

/// A draw's bind groups, and the buffers of its own that copies fill before its pass: constant
/// buffers with what compute launches wrote over them, and copies of written globals.
struct Groups {
    vs: wgpu::BindGroup,
    fs: wgpu::BindGroup,
    /// Each group's dynamic offsets: where its constants are in the arena.
    offsets: [Vec<u32>; 2],
    patched: Vec<(wgpu::Buffer, Vec<crate::Patch>)>,
}

/// Everything a draw's pipeline is built from, kept with a held-back draw so the pipeline can
/// be rebuilt with its instanced arrays read at a real stride. See [`Gpu::emit_pending`].
#[derive(Debug, Clone)]
pub(crate) struct Recipe {
    /// What the pipeline is cached by; its buffers follow [`Recipe::buffers`].
    key: PipelineKey,
    modules: [wgpu::ShaderModule; 2],
    group_layouts: [wgpu::BindGroupLayout; 2],
    primitive: wgpu::PrimitiveState,
    multisample: wgpu::MultisampleState,
    colour_targets: Vec<Option<wgpu::ColorTargetState>>,
    /// Each vertex buffer's stride, step and attributes, in binding order.
    buffers: Vec<(u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)>,
}

impl Recipe {
    fn with_buffers(
        mut self,
        buffers: Vec<(u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)>,
    ) -> Recipe {
        self.key.buffers = buffers
            .iter()
            .map(|(stride, step, attributes)| {
                let attributes = attributes
                    .iter()
                    .map(|a| (a.format, a.offset, a.shader_location))
                    .collect();
                (
                    *stride as u32,
                    *step == wgpu::VertexStepMode::Instance,
                    attributes,
                )
            })
            .collect();
        self.buffers = buffers;
        self
    }

    /// The same pipeline with each of `strides`' slots, an instanced array, read at its stride.
    pub(crate) fn with_strides(&self, strides: &[(usize, u64)]) -> Recipe {
        let mut buffers = self.buffers.clone();
        for &(slot, stride) in strides {
            buffers[slot].0 = stride;
        }
        self.clone().with_buffers(buffers)
    }
}
