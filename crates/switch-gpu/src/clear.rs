//! Clears on the device.

use crate::builtin::CLEAR_RECT_WGSL;
use crate::convert::{depth_texture_format, device_attachment_format, write_mask};
use crate::readback::Held;
use crate::{ClearKey, Gpu};
use switch_core::gpu::engine::threed::Engine3D;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state};
use switch_core::gpu::upload::Target;
use switch_core::Result;

impl Gpu {
    /// Clear part or all of the surfaces a `ClearBuffers` names; whole clears skip the upload.
    fn clear_on_device(
        &mut self,
        color: Option<(Target, [f32; 4], [bool; 4])>,
        depth: Option<(Target, f32)>,
        rect: state::ScissorRect,
        ctx: &ExecCtx,
    ) -> std::result::Result<(), String> {
        let extent = color.map(|(t, _, _)| t).or(depth.map(|(t, _)| t));
        let Some(extent) = extent else { return Ok(()) };
        let whole = rect.x0 == 0
            && rect.y0 == 0
            && rect.x1 >= extent.width
            && rect.y1 >= extent.height
            && color.is_none_or(|(_, _, channels)| channels.iter().all(|&c| c));
        if rect.x1 <= rect.x0 || rect.y1 <= rect.y0 {
            return Ok(());
        }

        let mut views = Vec::new();
        for (target, blank) in [color.map(|(t, _, _)| t), depth.map(|(t, _)| t)]
            .into_iter()
            .flatten()
            .map(|t| (t, whole))
        {
            // A surface about to be written whole need not be uploaded.
            if blank {
                self.hold_blank(&target).map_err(|e| format!("{e:?}"))?;
                // The companion is about to be overwritten, so drop it.
                self.discard_companion(target.addr);
            } else {
                self.hold(&target, ctx).map_err(|e| format!("{e:?}"))?;
                // A partial clear keeps what the companion holds outside the rectangle.
                self.resolve_companion(target.addr)?;
            }
            let held = self
                .held
                .get_mut(&target.addr)
                .ok_or("the surface was not held")?;
            held.dirty = true;
            views.push(
                held.texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
        }
        let mut view = views.into_iter();
        let colour_view = color.map(|_| view.next().expect("a colour view"));
        let depth_view = depth.map(|_| view.next().expect("a depth view"));

        let key = ClearKey {
            color: match color {
                Some((target, _, _)) => Some(
                    device_attachment_format(self.features(), target.format)
                        .map_err(|e| format!("{e:?}"))?,
                ),
                None => None,
            },
            depth: depth
                .and_then(|(target, _)| target.depth_kind())
                .map(depth_texture_format),
            write_mask: color.map_or([true; 4], |(_, _, channels)| channels),
        };
        // Only a partial clear needs its value in a uniform.
        let uniform = (!whole).then(|| {
            let [r, g, b, a] = color.map_or([0.0; 4], |(_, colour, _)| colour);
            let mut bytes = Vec::new();
            for value in [r, g, b, a, depth.map_or(0.0, |(_, d)| d)] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            // A uniform binding is a multiple of sixteen bytes.
            bytes.resize(32, 0);
            self.buffer("clear", &bytes, wgpu::BufferUsages::UNIFORM)
        });
        let pipeline = match &uniform {
            Some(_) => Some(self.clear_pipeline(key)?),
            None => None,
        };
        let group = pipeline
            .as_ref()
            .zip(uniform.as_ref())
            .map(|(pipeline, buffer)| {
                let layout = pipeline.get_bind_group_layout(0);
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("clear"),
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                })
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clear"),
            });
        {
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_view
                .as_ref()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: match (whole, color) {
                                (true, Some((_, [r, g, b, a], _))) => {
                                    wgpu::LoadOp::Clear(wgpu::Color {
                                        r: f64::from(r),
                                        g: f64::from(g),
                                        b: f64::from(b),
                                        a: f64::from(a),
                                    })
                                }
                                _ => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .into_iter()
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &attachments,
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: match (whole, depth) {
                                (true, Some((_, value))) => wgpu::LoadOp::Clear(value),
                                _ => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some(pipeline), Some(group)) = (&pipeline, &group) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.set_scissor_rect(rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Clear colour target `target`, on the device.
    pub(super) fn clear_color_here(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> std::result::Result<(), String> {
        if layer != 0 {
            // Surfaces are held by address only, so layers are unsupported.
            return Err(format!("a clear of layer {layer}"));
        }
        let slot = engine.render_target_slot(target);
        let Some(surface) = Target::color(engine, slot).map_err(|e| format!("{e:?}"))? else {
            // Nothing bound is nothing to clear, which is what the rasterizer answers too.
            return Ok(());
        };
        let rect = self.clear_texels(engine, &surface)?;
        self.clear_on_device(
            Some((surface, engine.clear_color_value(), channels)),
            None,
            rect,
            ctx,
        )
    }

    /// Clear the depth surface, on the device.
    pub(super) fn clear_depth_here(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<(), String> {
        let Some(surface) = Target::depth_surface(engine).map_err(|e| format!("{e:?}"))? else {
            return Ok(());
        };
        let rect = self.clear_texels(engine, &surface)?;
        self.clear_on_device(None, Some((surface, engine.clear_depth_value())), rect, ctx)
    }

    /// The clear rectangle in texels: pixels scaled by the sample tile.
    fn clear_texels(
        &self,
        engine: &Engine3D,
        surface: &Target,
    ) -> std::result::Result<state::ScissorRect, String> {
        let grid = engine.sample_grid().map_err(|e| format!("{e:?}"))?;
        let (width, height) = grid.pixels(surface.width, surface.height);
        let rect = engine.clear_rectangle(width, height);
        Ok(state::ScissorRect {
            x0: rect.x0 * grid.samples_x,
            y0: rect.y0 * grid.samples_y,
            x1: rect.x1 * grid.samples_x,
            y1: rect.y1 * grid.samples_y,
        })
    }

    /// Hold a surface without reading it, for a clear that writes every texel.
    fn hold_blank(&mut self, target: &Target) -> Result<()> {
        match self.held.get(&target.addr) {
            // Already held: clear it in place.
            Some(held) if held.target == *target => return Ok(()),
            Some(_) => {
                if let Some(held) = self.held.remove(&target.addr) {
                    self.evicted.push(held);
                }
            }
            None => {}
        }
        let texture = self.blank_target(target)?;
        self.held.insert(
            target.addr,
            Held {
                texture,
                target: *target,
                dirty: false,
                companion: None,
            },
        );
        Ok(())
    }

    /// A device texture for a surface, with nothing in it.
    fn blank_target(&self, target: &Target) -> Result<wgpu::Texture> {
        let (format, usage) = match target.depth_kind() {
            Some(kind) => (
                depth_texture_format(kind),
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            None => (
                device_attachment_format(self.features(), target.format)?,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
        };
        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cleared target"),
            size: wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }))
    }

    /// The rectangle clear pipeline for these formats, built once. See [`CLEAR_RECT_WGSL`].
    pub(super) fn clear_pipeline(
        &mut self,
        key: ClearKey,
    ) -> std::result::Result<wgpu::RenderPipeline, String> {
        if let Some(pipeline) = self.clear_pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let (_, module) = self.module("clear", CLEAR_RECT_WGSL);
        let targets: Vec<Option<wgpu::ColorTargetState>> = key
            .color
            .map(|format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: write_mask(key.write_mask),
                })
            })
            .into_iter()
            .collect();
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("clear"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: key.depth.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(if key.color.is_some() {
                        "fs_color"
                    } else {
                        "fs_depth"
                    }),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            });
        self.clear_pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}
