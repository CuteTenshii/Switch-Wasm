//! 2D engine blits between surfaces held on the device, so a resolve does not send a
//! rendered frame through guest memory and back.

use crate::builtin::BLIT_WGSL;
use crate::convert::device_attachment_format;
use crate::held::{spans_overlap as overlap, surface_span as span};
use crate::Gpu;
use std::ops::Range;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::Format;
use switch_core::gpu::renderer::{Blit, BlitRoute};
use switch_core::gpu::surface::Surface;
use switch_core::gpu::upload::Target;

/// The scale a channel of `format` is rounded at, as `ColorFormat::encode` rounds it; a device's
/// unorm conversion rounds a half down, and a two-tap resolve lands on halves all the time.
/// `None` for a format the blit shader cannot carry unchanged.
fn rounding(format: Format) -> Option<f32> {
    match format {
        Format::R8Unorm | Format::Rg8Unorm | Format::Rgba8Unorm | Format::Bgra8Unorm => Some(255.0),
        Format::Rgb10a2Unorm
        | Format::R16Unorm
        | Format::Rg16Unorm
        | Format::Rgba16Unorm
        | Format::R16Float
        | Format::Rg16Float
        | Format::Rgba16Float
        | Format::R32Float
        | Format::Rg32Float
        | Format::Rgba32Float
        | Format::Rg11b10Ufloat => Some(0.0),
        _ => None,
    }
}

/// Past this many destination texels, filtering on the CPU every frame costs more than
/// uploading the source once; smaller blits are mostly texture uploads, cheap where they are.
const UPLOAD_BLIT_TEXELS: u64 = 256 * 256;

pub(super) fn worth_uploading(blit: &Blit) -> bool {
    u64::from(blit.dst_rect.2) * u64::from(blit.dst_rect.3) >= UPLOAD_BLIT_TEXELS
}

impl Gpu {
    /// Whether guest memory is current for both surfaces and stays so if the engine writes it.
    pub(super) fn blit_touches_nothing_held(&self, blit: &Blit) -> bool {
        let surface_span = |surface: &Surface| {
            let len = u64::from(surface.size());
            surface.addr..surface.addr + len
        };
        self.touches_nothing_held(&[surface_span(&blit.src), surface_span(&blit.dst)])
    }

    /// Whether nothing held, rebound or on its way back shares memory with `ranges`.
    pub(super) fn touches_nothing_held(&self, ranges: &[Range<u64>]) -> bool {
        let clear = |target: &Target| {
            let held = span(target);
            !ranges.iter().any(|range| overlap(&held, range))
        };
        self.space.held.values().all(|held| clear(&held.target))
            && self.space.evicted.iter().all(|held| clear(&held.target))
            && self
                .space
                .pending
                .iter()
                .all(|pending| clear(&pending.target))
    }

    /// Run `blit` on the device if its source is, or can be held there as the plain colour
    /// surface the registers name; anything else is for the engine, after a flush.
    pub(super) fn blit_on_device(&mut self, blit: &Blit, ctx: &ExecCtx) -> BlitRoute {
        match self.try_blit_on_device(blit, ctx) {
            Some(()) => BlitRoute::Done,
            None => BlitRoute::Flush,
        }
    }

    fn try_blit_on_device(&mut self, blit: &Blit, ctx: &ExecCtx) -> Option<()> {
        let (x0, y0, width, height) = blit.dst_rect;
        let source = Target::of_surface(&blit.src).ok()?;
        let target = Target::of_surface(&blit.dst).ok()?;
        let round = rounding(target.format)?;
        let (x1, y1) = (x0.checked_add(width)?, y0.checked_add(height)?);
        if source.format != target.format
            || width == 0
            || height == 0
            || x1 > target.width
            || y1 > target.height
            || overlap(&span(&source), &span(&target))
        {
            return None;
        }
        match self.space.held.get(&source.addr) {
            Some(held) if held.target != source => return None,
            Some(_) => {}
            None => self.hold(&source, ctx).ok()?,
        }
        let format = device_attachment_format(self.features(), target.format).ok()?;
        self.resolve_companion(source.addr).ok()?;
        // A surface about to be written whole need not be uploaded.
        if (x0, y0, x1, y1) == (0, 0, target.width, target.height) {
            self.hold_blank(&target).ok()?;
            self.discard_companion(target.addr);
        } else {
            self.hold(&target, ctx).ok()?;
            self.resolve_companion(target.addr).ok()?;
        }

        let pipeline = self.blit_pipeline(format, blit.filtered);
        let last = (
            source.width.saturating_sub(1) as f32,
            source.height.saturating_sub(1) as f32,
        );
        let mut bytes = Vec::with_capacity(48);
        for value in [
            blit.src_origin.0 as f32,
            blit.src_origin.1 as f32,
            blit.step.0 as f32,
            blit.step.1 as f32,
            x0 as f32,
            y0 as f32,
            last.0,
            last.1,
            round,
            0.0,
            0.0,
            0.0,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let parameters = self.buffer("blit", &bytes, wgpu::BufferUsages::UNIFORM);
        let source_view = self
            .space
            .held
            .get(&source.addr)?
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let held = self.space.held.get_mut(&target.addr)?;
        held.dirty = true;
        let target_view = held
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let layout = pipeline.get_bind_group_layout(0);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_scissor_rect(x0, y0, width, height);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        Some(())
    }

    fn blit_pipeline(
        &mut self,
        format: wgpu::TextureFormat,
        filtered: bool,
    ) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.blit_pipelines.get(&(format, filtered)) {
            return pipeline.clone();
        }
        let (_, module) = self.module("blit", BLIT_WGSL);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(if filtered { "fs_filtered" } else { "fs_point" }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.blit_pipelines
            .insert((format, filtered), pipeline.clone());
        pipeline
    }
}
