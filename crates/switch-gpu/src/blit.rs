//! 2D engine blits between surfaces held on the device, so a resolve does not send a
//! rendered frame through guest memory and back.

use crate::builtin::{BLIT_WGSL, PACKED_DEPTH_WGSL};
use crate::convert::device_attachment_format;
use crate::held::{spans_overlap as overlap, surface_span as span};
use crate::Gpu;
use crate::{Held, Scratch};
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
        Format::R8Unorm
        | Format::Rg8Unorm
        | Format::Rgba8Unorm
        | Format::Bgra8Unorm
        | Format::Rgba8UnormSrgb
        | Format::Bgra8UnormSrgb => Some(255.0),
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

/// Whether the device texture for `format` stores an alpha channel, which a guest format
/// without alpha leaves as whatever was rendered.
fn stores_alpha(format: Format) -> bool {
    !matches!(
        format,
        Format::R8Unorm
            | Format::Rg8Unorm
            | Format::R16Unorm
            | Format::Rg16Unorm
            | Format::R16Float
            | Format::Rg16Float
            | Format::R32Float
            | Format::Rg32Float
            | Format::Rg11b10Ufloat
    )
}

fn is_srgb(format: Format) -> bool {
    matches!(format, Format::Rgba8UnormSrgb | Format::Bgra8UnormSrgb)
}

/// Whether the shader converts `blit`'s source into its target as the engine's decode and
/// encode do: only into 8-bit targets, which it rounds itself, and with alpha read as stored.
fn converts(blit: &Blit, source: Format, target: Format) -> bool {
    rounding(source).is_some()
        && rounding(target) == Some(255.0)
        && (blit.src.format.has_alpha() || !stores_alpha(source))
        && (blit.dst.format.has_alpha() || !stores_alpha(target))
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

    /// Whether a surface held at another address shares memory with the blit's surfaces, which
    /// only guest memory tells apart.
    pub(super) fn blit_aliases_held(&self, blit: &Blit) -> bool {
        [&blit.src, &blit.dst].into_iter().any(|surface| {
            let range = surface.addr..surface.addr + u64::from(surface.size());
            self.space.held.values().any(|held| {
                held.target.addr != surface.addr && overlap(&span(&held.target), &range)
            })
        })
    }

    /// Whether a surface rebound or on its way back to guest memory shares memory with `range`,
    /// so guest memory is not yet what the device would be given.
    fn in_flight_overlaps(&self, range: &Range<u64>) -> bool {
        self.space
            .pending
            .iter()
            .any(|pending| pending.target().is_some_and(|t| overlap(&span(t), range)))
    }

    /// Whether nothing held, rebound or on its way back shares memory with `ranges`.
    pub(super) fn touches_nothing_held(&self, ranges: &[Range<u64>]) -> bool {
        self.touches_no_surface(ranges) && self.no_global_in(ranges)
    }

    /// Whether `ranges` miss every surface held on the device or on its way back.
    pub(super) fn touches_no_surface(&self, ranges: &[Range<u64>]) -> bool {
        let clear = |target: &Target| {
            let held = span(target);
            !ranges.iter().any(|range| overlap(&held, range))
        };
        self.space.held.values().all(|held| {
            clear(&held.target) && held.other.as_ref().is_none_or(|other| clear(&other.target))
        }) && self
            .space
            .pending
            .iter()
            .all(|pending| pending.target().is_none_or(clear))
    }

    /// Whether `ranges` miss every buffer shaders wrote on the device, held or on its way back.
    pub(super) fn no_global_in(&self, ranges: &[Range<u64>]) -> bool {
        let clear = |span: Range<u64>| !ranges.iter().any(|range| overlap(&span, range));
        self.space
            .globals
            .iter()
            .all(|(&at, global)| clear(at..at + global.original.len() as u64))
            && self
                .space
                .pending
                .iter()
                .filter_map(|pending| pending.global_span())
                .all(clear)
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
        let (x1, y1) = (x0.checked_add(width)?, y0.checked_add(height)?);
        if width == 0
            || height == 0
            || x1 > target.width
            || y1 > target.height
            || overlap(&span(&source), &span(&target))
        {
            return None;
        }
        let packed = self
            .space
            .held
            .get(&source.addr)
            .is_some_and(|held| held.target != source && holds_words_of(held, &source));
        // The last parameter: the rounding scale and sRGB flag, or whether the target is BGRA.
        let (entry, tail) = if packed {
            if !moves_words(blit, &target) {
                return None;
            }
            let bgra = u32::from(target.format == Format::Bgra8Unorm);
            ((PACKED_DEPTH_WGSL, "fs_blit"), [bgra, 0])
        } else {
            let round = rounding(target.format)?;
            // The device's sRGB decode is not the engine's.
            if is_srgb(source.format)
                || (source.format != target.format && !converts(blit, source.format, target.format))
            {
                return None;
            }
            match self.space.held.get(&source.addr) {
                Some(held) if held.target != source => return None,
                Some(_) => {}
                None if self.in_flight_overlaps(&span(&source)) => return None,
                None => self.hold(&source, ctx).ok()?,
            }
            self.resolve_companion(source.addr).ok()?;
            let entry = if blit.filtered {
                "fs_filtered"
            } else {
                "fs_point"
            };
            let entry = (BLIT_WGSL, entry);
            (entry, [round.to_bits(), u32::from(is_srgb(target.format))])
        };
        let format = device_attachment_format(self.features(), target.format).ok()?;
        let stencil = if packed {
            Some(self.stencil_words(&source, ctx)?)
        } else {
            None
        };
        // A surface about to be written whole need not be uploaded.
        if (x0, y0, x1, y1) == (0, 0, target.width, target.height) {
            self.hold_blank(&target).ok()?;
            self.discard_companion(target.addr);
        } else {
            if !self.space.held.contains_key(&target.addr)
                && self.in_flight_overlaps(&span(&target))
            {
                return None;
            }
            self.hold(&target, ctx).ok()?;
            self.resolve_companion(target.addr).ok()?;
        }

        let pipeline = self.blit_pipeline(format, entry);
        let mut bytes = Vec::with_capacity(48);
        for value in [
            blit.src_origin.0 as f32,
            blit.src_origin.1 as f32,
            blit.step.0 as f32,
            blit.step.1 as f32,
            x0 as f32,
            y0 as f32,
            source.width.saturating_sub(1) as f32,
            source.height.saturating_sub(1) as f32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in tail {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.resize(48, 0);
        let parameters = self.buffer("blit", &bytes, wgpu::BufferUsages::UNIFORM);
        let source_texture = &self.space.held.get(&source.addr)?.texture;
        let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor {
            aspect: if packed {
                wgpu::TextureAspect::DepthOnly
            } else {
                wgpu::TextureAspect::All
            },
            ..Default::default()
        });
        let stencil_view = stencil
            .as_ref()
            .map(|words| words.create_view(&wgpu::TextureViewDescriptor::default()));
        let held = self.space.held.get_mut(&target.addr)?;
        held.dirty = true;
        held.writes += 1;
        held.overwritten.clear();
        let target_view = held
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&source_view),
        }];
        // `PACKED_DEPTH_WGSL` binds the words second.
        if let Some(view) = &stencil_view {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: entries.len() as u32,
            resource: parameters.as_entire_binding(),
        });
        let layout = pipeline.get_bind_group_layout(0);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &layout,
            entries: &entries,
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
        self.submit_with_batch(encoder);
        if let Some(words) = stencil {
            self.scratch.push(Scratch::Texture(words));
        }
        Some(())
    }

    /// `source`'s words as guest memory holds them, for their stencil bytes.
    fn stencil_words(&mut self, source: &Target, ctx: &ExecCtx) -> Option<wgpu::Texture> {
        let bytes = source.read(ctx).ok()?;
        let size = wgpu::Extent3d {
            width: source.width,
            height: source.height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stencil words"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(source.row_bytes),
                rows_per_image: Some(source.rows),
            },
            size,
        );
        Some(texture)
    }

    /// The pipeline running `entry` of `source` into `format`; entry points are unique across sources.
    fn blit_pipeline(
        &mut self,
        format: wgpu::TextureFormat,
        (source, entry): (&'static str, &'static str),
    ) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.blit_pipelines.get(&(format, entry)) {
            return pipeline.clone();
        }
        let (_, module) = self.module("blit", source);
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
                    entry_point: Some(entry),
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
            .insert((format, entry), pipeline.clone());
        pipeline
    }
}

/// Whether `held` is an `S8D24` depth surface on the same texels as `source`, so each of
/// `source`'s words is its stencil byte and its depth in unorm.
fn holds_words_of(held: &Held, source: &Target) -> bool {
    let target = &held.target;
    target.depth.is_some_and(|depth| {
        (
            depth.bytes,
            depth.depth_bits,
            depth.depth_shift,
            depth.stencil_shift,
        ) == (4, 24, 8, Some(0))
    }) && (
        target.width,
        target.height,
        target.layout,
        target.row_bytes,
        target.rows,
    ) == (
        source.width,
        source.height,
        source.layout,
        source.row_bytes,
        source.rows,
    ) && held.texture.format() == wgpu::TextureFormat::Depth32Float
        && held.texture.sample_count() == 1
        && (held.texture.width(), held.texture.height()) == (source.width, source.height)
}

/// Whether the engine runs `blit` as a move of whole words into `target`, which holds bytes.
fn moves_words(blit: &Blit, target: &Target) -> bool {
    !blit.filtered
        && blit.src.format.raw == blit.dst.format.raw
        && blit.dst.format.is_byte_exact()
        && matches!(target.format, Format::Rgba8Unorm | Format::Bgra8Unorm)
}
