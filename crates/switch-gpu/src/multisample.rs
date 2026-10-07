//! Routing multisampled draws, companions, and resampling passes.

use crate::builtin::{grid_bytes, resample_wgsl, ResampleKey};
use crate::convert::{depth_texture_format, device_attachment_format};
use crate::readback::Companion;
use crate::{Gpu, Render, Shape};
use switch_core::gpu::pipeline::Pipeline;
use switch_core::gpu::surface::SampleGrid;
use switch_core::gpu::upload::Target;

impl Gpu {
    /// Decide how a draw reaches its surfaces. The expanded route is the default because
    /// it reproduces Maxwell's texel-centre sample positions exactly.
    pub(super) fn route(
        &self,
        state: &Pipeline,
        color: Option<Target>,
        depth: Option<Target>,
    ) -> std::result::Result<Render, String> {
        if state.grid.is_single() {
            return Ok(Render::Direct);
        }
        if state.per_pixel_coverage {
            // Every texel of a pixel's tile takes the same value, so a partial mask cannot apply.
            let all = (1u64 << state.samples) - 1;
            if u64::from(state.sample_mask) & all != all || state.alpha_to_coverage {
                return Err("a draw with coverage per pixel and a mask that is per sample".into());
            }
            return Ok(Render::Companion(Shape::PerPixel));
        }
        let formats = [
            match color {
                Some(color) => Some(
                    device_attachment_format(self.features(), color.format)
                        .map_err(|e| format!("{e:?}"))?,
                ),
                None => None,
            },
            depth.and_then(|d| d.depth_kind()).map(depth_texture_format),
        ];
        // All attachments of a pass share one sample count.
        let offered = self.device_msaa
            && formats
                .into_iter()
                .flatten()
                .all(|format| self.samples_supported(format, state.samples));
        if offered {
            return Ok(Render::Companion(Shape::Multisampled(state.samples)));
        }
        // Samples moved off texel centres cannot be expressed by either route.
        if !state.grid.samples_at_texel_centres() {
            return Err("a draw with programmed sample locations".into());
        }
        Ok(Render::Expanded)
    }

    /// Whether this adapter will render `samples` samples into `format`.
    pub(super) fn samples_supported(&self, format: wgpu::TextureFormat, samples: u32) -> bool {
        // The adapter's answer applies only with `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`.
        let features = self.device.features();
        let flags = if features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            self._adapter.get_texture_format_features(format).flags
        } else {
            format.guaranteed_format_features(features).flags
        };
        flags.sample_count_supported(samples)
    }

    /// Give the surface at `addr` a companion of `shape`, resolving any previous one first.
    pub(super) fn companion(
        &mut self,
        addr: u64,
        shape: Shape,
        grid: SampleGrid,
    ) -> std::result::Result<(), String> {
        match self.held.get(&addr).and_then(|h| h.companion.as_ref()) {
            Some(have) if have.shape == shape && have.grid == grid => return Ok(()),
            Some(_) => self.resolve_companion(addr)?,
            None => {}
        }
        let held = self.held.get(&addr).ok_or("the surface was not held")?;
        let (width, height) = grid.pixels(held.target.width, held.target.height);
        let samples = match shape {
            Shape::Multisampled(n) => n,
            Shape::PerPixel => 1,
        };
        let depth = held.target.depth_kind();
        let format = held.texture.format();
        let companion = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("companion"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            // A multisampled texture accepts no copy usage.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // Fill the companion from the surface.
        let source = self
            .held
            .get(&addr)
            .ok_or("the surface was not held")?
            .texture
            .clone();
        self.resample(
            &companion,
            &source,
            ResampleKey {
                entry: if samples > 1 {
                    "fs_gather"
                } else {
                    "fs_gather_flat"
                },
                dst: format,
                samples,
                ms_source: false,
                depth: depth.is_some(),
            },
            grid,
        )?;
        let held = self.held.get_mut(&addr).ok_or("the surface was not held")?;
        held.companion = Some(Companion {
            shape,
            texture: companion,
            grid,
        });
        Ok(())
    }

    /// Drop the companion without resolving it, for a caller about to overwrite the surface.
    pub(super) fn discard_companion(&mut self, addr: u64) {
        if let Some(held) = self.held.get_mut(&addr) {
            if let Some(companion) = held.companion.take() {
                companion.texture.destroy();
            }
        }
    }

    /// Resolve the companion back into the surface and drop it.
    pub(super) fn resolve_companion(&mut self, addr: u64) -> std::result::Result<(), String> {
        let Some(held) = self.held.get_mut(&addr) else {
            return Ok(());
        };
        let Some(companion) = held.companion.take() else {
            return Ok(());
        };
        let surface = held.texture.clone();
        let depth = held.target.depth_kind().is_some();
        self.resolve_into(&surface, &companion, depth)
    }

    /// Scatter a companion back into the expanded surface it stands in for.
    pub(super) fn resolve_into(
        &mut self,
        surface: &wgpu::Texture,
        companion: &Companion,
        depth: bool,
    ) -> std::result::Result<(), String> {
        self.resample(
            surface,
            &companion.texture,
            ResampleKey {
                entry: "fs_scatter",
                dst: surface.format(),
                samples: 1,
                ms_source: matches!(companion.shape, Shape::Multisampled(_)),
                depth,
            },
            companion.grid,
        )?;
        companion.texture.destroy();
        Ok(())
    }

    /// Run one resampling pass from `src` into `dst`.
    fn resample(
        &mut self,
        dst: &wgpu::Texture,
        src: &wgpu::Texture,
        key: ResampleKey,
        grid: SampleGrid,
    ) -> std::result::Result<(), String> {
        let pipeline = self.resample_pipeline(key)?;
        let layout = pipeline.get_bind_group_layout(0);
        let buffer = self.buffer("grid", &grid_bytes(grid), wgpu::BufferUsages::STORAGE);
        let source = src.create_view(&wgpu::TextureViewDescriptor::default());
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resample"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source),
                },
            ],
        });
        let view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resample"),
            });
        {
            let colour: Vec<Option<wgpu::RenderPassColorAttachment>> = (!key.depth)
                .then_some({
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Every texel of the destination is written.
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .into_iter()
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resample"),
                color_attachments: &colour,
                depth_stencil_attachment: key.depth.then_some({
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// The pipeline for one resampling direction, built once per shape.
    pub(super) fn resample_pipeline(
        &mut self,
        key: ResampleKey,
    ) -> std::result::Result<wgpu::RenderPipeline, String> {
        if let Some(pipeline) = self.resample_pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let (sampled, load) = match (key.depth, key.ms_source, key.entry) {
            (false, false, "fs_scatter") => ("texture_2d<f32>", "textureLoad(src, pixel, 0)"),
            (false, false, _) => ("texture_2d<f32>", "textureLoad(src, texel, 0)"),
            (false, true, _) => (
                "texture_multisampled_2d<f32>",
                "textureLoad(src, pixel, sample)",
            ),
            (true, false, "fs_scatter") => ("texture_depth_2d", "textureLoad(src, pixel, 0)"),
            (true, false, _) => ("texture_depth_2d", "textureLoad(src, texel, 0)"),
            (true, true, _) => (
                "texture_depth_multisampled_2d",
                "textureLoad(src, pixel, sample)",
            ),
        };
        let (_, module) = {
            let source = resample_wgsl(sampled, load, key.depth);
            self.module("resample", &source)
        };
        let targets: Vec<Option<wgpu::ColorTargetState>> = (!key.depth)
            .then_some({
                Some(wgpu::ColorTargetState {
                    format: key.dst,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .into_iter()
            .collect();
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("resample"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: key.depth.then(|| wgpu::DepthStencilState {
                    format: key.dst,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: key.samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(key.entry),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            });
        self.resample_pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}
