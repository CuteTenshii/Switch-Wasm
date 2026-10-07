//! The draw pass: pipeline, encoding, and attachments.

use crate::convert::{
    blend, compare, depth_texture_format, device_attachment_format, index_format, topology,
    vertex_format, write_mask,
};
use crate::readback::Scratch;
use crate::{
    can_blend, dump_wgsl, Bound, Gpu, PipelineKey, Prepared, Render, Shape, ABSENT_ATTRIBUTE,
    ATTRIBUTE_DEFAULTS, DEFAULT_ATTRIBUTE,
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
        let target_format = match p.color {
            Some(color) => Some(
                device_attachment_format(self.features(), color.format)
                    .map_err(|e| format!("{e:?}"))?,
            ),
            None => None,
        };
        // A blend the device cannot do must fall back here; a rejected pipeline silently draws nothing.
        if let Some(format) = target_format {
            let blends = p.state.target.is_some_and(|t| t.blend.is_some());
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
            let vs_source = wgsl::module(&p.vs, Stage::Vertex, &p.vs_layout);
            let fs_source = wgsl::module(&p.fs, Stage::Fragment, &p.fs_layout);
            match (vs_source, fs_source) {
                (Ok(vs), Ok(fs)) => {
                    if let Ok(dir) = std::env::var("GPU_DUMP_WGSL") {
                        dump_wgsl(&dir, &vs, &fs);
                    }
                    Ok((self.module("vertex", &vs), self.module("fragment", &fs)))
                }
                (Err(e), _) | (_, Err(e)) => Err(format!("module: {e}")),
            }
        })?;

        // Vertex buffers the shader reads; the stride travels with each buffer.
        let mut bound: Vec<Bound> = Vec::new();
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
            bound.push(Bound {
                buffer: self.buffer("vertex", &bytes, wgpu::BufferUsages::VERTEX),
                attributes,
                stride,
                step: match buffer.step {
                    state::StepMode::Instance => wgpu::VertexStepMode::Instance,
                    state::StepMode::Vertex => wgpu::VertexStepMode::Vertex,
                },
            });
        }
        // Unbound locations read a constant buffer of the two default vectors.
        let fed: Vec<usize> = bound
            .iter()
            .flat_map(|b| b.attributes.iter().map(|a| a.shader_location as usize))
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
            bound.push(Bound {
                buffer: self.buffer("defaults", &ATTRIBUTE_DEFAULTS, wgpu::BufferUsages::VERTEX),
                attributes: unfed,
                stride: 0,
                step: wgpu::VertexStepMode::Vertex,
            });
        }
        let layouts: Vec<Option<wgpu::VertexBufferLayout>> = bound
            .iter()
            .map(|b| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: b.stride,
                    step_mode: b.step,
                    attributes: &b.attributes,
                })
            })
            .collect();

        let (vs_group_layout, vs_group) =
            timed!(self, pipeline, self.bind_group(p, ShaderStage::VertexB, 0))?;
        let (fs_group_layout, fs_group) =
            timed!(self, pipeline, self.bind_group(p, ShaderStage::Fragment, 1))?;
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
        let key = PipelineKey {
            vs: vs_key,
            fs: fs_key,
            target: target_format,
            depth: depth_format
                .zip(p.state.depth)
                .map(|(format, d)| (format, d.write_enabled, d.compare)),
            samples: multisample.count,
            sample_mask: multisample.mask,
            alpha_to_coverage: multisample.alpha_to_coverage_enabled,
            blend: p.state.target.and_then(|t| t.blend),
            write_mask: p.state.target.map_or([true; 4], |t| t.write_mask),
            topology: p.state.topology,
            strip_index_format,
            front_face: p.state.front_face,
            cull: p.state.cull,
            buffers: bound
                .iter()
                .map(|b| {
                    let attributes = b
                        .attributes
                        .iter()
                        .map(|a| (a.format, a.offset, a.shader_location))
                        .collect();
                    (
                        b.stride as u32,
                        b.step == wgpu::VertexStepMode::Instance,
                        attributes,
                    )
                })
                .collect(),
        };
        if let Some(pipeline) = self.pipelines.get(&key) {
            return self.encode(p, &pipeline.clone(), &vs_group, &fs_group, &bound, ctx);
        }
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("draw"),
                bind_group_layouts: &[Some(&vs_group_layout), Some(&fs_group_layout)],
                immediate_size: 0,
            });
        // Empty for a depth-only pass.
        let colour_targets: Vec<Option<wgpu::ColorTargetState>> = target_format
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
            .collect();
        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &layouts,
            },
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
            multisample,
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &colour_targets,
            }),
            multiview_mask: None,
            cache: None,
        };
        let pipeline = timed!(
            self,
            pipeline,
            self.device.create_render_pipeline(&descriptor)
        );
        self.pipelines.insert(key, pipeline.clone());
        self.encode(p, &pipeline, &vs_group, &fs_group, &bound, ctx)
    }

    /// Record and submit one draw against a built pipeline.
    fn encode(
        &mut self,
        p: &Prepared,
        pipeline: &wgpu::RenderPipeline,
        vs_group: &wgpu::BindGroup,
        fs_group: &wgpu::BindGroup,
        bound: &[Bound],
        ctx: &mut ExecCtx,
    ) -> std::result::Result<(), String> {
        // Held across the frame.
        let colour_view = match p.color {
            Some(color) => {
                self.hold(&color, ctx).map_err(|e| format!("{e:?}"))?;
                Some(self.attachment(p, color.addr, true)?)
            }
            None => None,
        };
        // See [`Prepared::color_scratch`].
        let scratch = match (p.color, p.color_scratch) {
            (Some(color), Some((width, height))) => {
                let held = &self
                    .held
                    .get(&color.addr)
                    .ok_or("the surface was not held")?
                    .texture;
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
                self.queue.submit([encoder.finish()]);
                Some((held.clone(), texture, size))
            }
            _ => None,
        };
        let colour_view = match &scratch {
            Some((_, texture, _)) => {
                Some(texture.create_view(&wgpu::TextureViewDescriptor::default()))
            }
            None => colour_view,
        };
        let depth_view = match p.depth {
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
            Some((indices, base)) => {
                let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                Some((
                    self.buffer("assembled", &bytes, wgpu::BufferUsages::INDEX),
                    wgpu::IndexFormat::Uint32,
                    *base,
                ))
            }
            None => p.uploads.index.as_ref().map(|index| {
                (
                    self.buffer("index", &index.bytes, wgpu::BufferUsages::INDEX),
                    index_format(index.format),
                    -(index.lowest as i32),
                )
            }),
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("draw"),
            });
        {
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, vs_group, &[]);
            pass.set_bind_group(1, fs_group, &[]);
            for (slot, b) in bound.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, b.buffer.slice(..));
            }
            // In pixels, except on the expanded route where the attachment is in texels.
            let (sx, sy) = match p.render {
                Render::Expanded => (p.state.grid.samples_x, p.state.grid.samples_y),
                _ => (1, 1),
            };
            let viewport = &p.state.viewport;
            pass.set_viewport(
                viewport.x * sx as f32,
                viewport.y * sy as f32,
                viewport.width * sx as f32,
                viewport.height * sy as f32,
                viewport.min_depth.clamp(0.0, 1.0),
                viewport.max_depth.clamp(0.0, 1.0),
            );
            let scissor = p.state.scissor;
            pass.set_scissor_rect(
                scissor.x0 * sx,
                scissor.y0 * sy,
                scissor.x1.saturating_sub(scissor.x0) * sx,
                scissor.y1.saturating_sub(scissor.y0) * sy,
            );
            if let Some(constant) = p.state.target.and_then(|t| t.blend) {
                let _ = constant;
                let [r, g, b, a] = p.state.blend_constant;
                pass.set_blend_constant(wgpu::Color {
                    r: f64::from(r),
                    g: f64::from(g),
                    b: f64::from(b),
                    a: f64::from(a),
                });
            }
            // One instance, numbered so `instance_index` matches `gl_InstanceID`.
            let instances = p.instance..p.instance + 1;
            match &index {
                Some((buffer, format, base)) => {
                    pass.set_index_buffer(buffer.slice(..), *format);
                    // The vertex buffer starts at the draw's lowest index.
                    pass.draw_indexed(0..p.count, *base, instances);
                }
                None => pass.draw(0..p.count, instances),
            }
        }
        if let Some((held, texture, size)) = &scratch {
            encoder.copy_texture_to_texture(texture.as_image_copy(), held.as_image_copy(), *size);
            self.scratch.push(Scratch::Texture(texture.clone()));
        }
        timed!(self, encode, self.queue.submit([encoder.finish()]));
        Ok(())
    }

    /// The view a draw renders one of its surfaces through, with any companion in place.
    fn attachment(
        &mut self,
        p: &Prepared,
        addr: u64,
        writes: bool,
    ) -> std::result::Result<wgpu::TextureView, String> {
        match p.render {
            Render::Companion(shape) => self.companion(addr, shape, p.state.grid)?,
            // Put back whatever a companion holds first.
            Render::Direct | Render::Expanded => self.resolve_companion(addr)?,
        }
        let held = self.held.get_mut(&addr).ok_or("the surface was not held")?;
        held.dirty |= writes;
        let texture = match &held.companion {
            Some(companion) => &companion.texture,
            None => &held.texture,
        };
        Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }
}
