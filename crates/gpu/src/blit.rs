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
        [(&blit.src, true), (&blit.dst, false)]
            .into_iter()
            .any(|(surface, only_read)| {
                let range = surface.addr..surface.addr + u64::from(surface.size());
                self.space.held.values().any(|held| {
                    held.target.addr != surface.addr && overlap(&span(&held.target), &range)
                }) || self.space.retired.iter().any(|retired| {
                    // A clean one owes guest memory nothing, and the source is only read. Pokémon
                    // Legends: Z-A otherwise flushed its depth every frame over its own colour view.
                    (retired.held.dirty || !only_read)
                        && overlap(&span(&retired.held.target), &range)
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
            .retired
            .iter()
            .all(|retired| clear(&retired.held.target))
            && self
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
    pub(super) fn blit_on_device(&mut self, blit: &Blit, ctx: &mut ExecCtx) -> BlitRoute {
        match self.try_blit_on_device(blit, ctx) {
            Some(()) => BlitRoute::Done,
            None => BlitRoute::Flush,
        }
    }

    fn try_blit_on_device(&mut self, blit: &Blit, ctx: &mut ExecCtx) -> Option<()> {
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
        self.uncrop_for_words(&source)?;
        let packed = self.holds_words_at(&source);
        // The last parameter: the rounding scale and sRGB flag, or whether the target is BGRA.
        let (entry, tail) = if packed {
            if !moves_words(blit, &target) {
                return None;
            }
            self.uncrop_for_words(&target)?;
            if self.holds_words_at(&target) {
                return self.blit_words_into_depth(blit, &source, &target, ctx);
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
        let parameters = self.buffer(
            "blit",
            &blit_parameters(blit, &source, tail),
            wgpu::BufferUsages::UNIFORM,
        );
        if !self.space.held.contains_key(&source.addr)
            || !self.space.held.contains_key(&target.addr)
        {
            return None;
        }
        // Made once nothing can turn the blit away, which would drop it undestroyed. Source
        // and target do not overlap, so holding the target left the source's words as they were.
        let stencil = if packed {
            Some(self.picked_stencil_texture(blit, ctx)?)
        } else {
            None
        };
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
            let timed = self.time_pass(|| "blit".into());
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
                timestamp_writes: self.render_timestamps(timed),
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

    /// A depth surface held as a crop of `surface`, with `surface`'s size kept beside it, goes
    /// back to that size on the device so its words are moved there. Pokémon Legends: Z-A moves
    /// words of its 1920x1080 depth after drawing it as a 960x540 crop, and the flush that took
    /// instead read the depth back and uploaded it twice a frame.
    fn uncrop_for_words(&mut self, surface: &Target) -> Option<()> {
        let Some(held) = self.space.held.get(&surface.addr) else {
            return Some(());
        };
        if held.target == *surface || held.target.depth.is_none() {
            return Some(());
        }
        let Some(whole) = held
            .other
            .as_deref()
            .filter(|other| holds_words_of(other, surface))
            .map(|other| other.target)
        else {
            return Some(());
        };
        self.switch_crop(&whole, true).ok().map(|_| ())
    }

    /// Whether `surface`'s words are an `S8D24` depth surface held on the device.
    fn holds_words_at(&self, surface: &Target) -> bool {
        self.space
            .held
            .get(&surface.addr)
            .is_some_and(|held| held.target != *surface && holds_words_of(held, surface))
    }

    /// The engine's move of `S8D24` words between two depth surfaces held on the device: the
    /// depth is drawn there, and the stencil bytes, which the device never holds, are moved in
    /// guest memory. Pokémon Legends: Z-A copies its depth whole and at half size every frame.
    fn blit_words_into_depth(
        &mut self,
        blit: &Blit,
        source: &Target,
        target: &Target,
        ctx: &mut ExecCtx,
    ) -> Option<()> {
        // A readback landing later would write over the stencil bytes moved now.
        if self.in_flight_overlaps(&span(source)) || self.in_flight_overlaps(&span(target)) {
            return None;
        }
        let (_, src_bytes) = swizzled_bytes(&blit.src, ctx)?;
        let picked = picked_stencil(blit, &src_bytes)?;
        let (dst_cpu, mut dst_bytes) = swizzled_bytes(&blit.dst, ctx)?;
        if !place_stencil(blit, &picked, &mut dst_bytes) {
            return None;
        }
        self.writing_guest_memory(ctx, |_, ctx| ctx.write_span(dst_cpu, &dst_bytes))
            .ok()?;

        let pipeline = self.blit_depth_pipeline();
        let parameters = self.buffer(
            "blit",
            &blit_parameters(blit, source, [0, 0]),
            wgpu::BufferUsages::UNIFORM,
        );
        let source_view =
            self.space
                .held
                .get(&source.addr)?
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                });
        let held = self.space.held.get_mut(&target.addr)?;
        held.dirty = true;
        held.writes += 1;
        held.overwritten.clear();
        let target_view = held
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let layout = pipeline.get_bind_group_layout(0);
        // `fs_blit_depth` reads no words, so the layout made from it has no binding 1.
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit depth"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parameters.as_entire_binding(),
                },
            ],
        });
        let (x0, y0, width, height) = blit.dst_rect;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit depth"),
            });
        {
            let timed = self.time_pass(|| "blit depth".into());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.render_timestamps(timed),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_scissor_rect(x0, y0, width, height);
            pass.draw(0..3, 0..1);
        }
        self.submit_with_batch(encoder);
        Some(())
    }

    /// The pipeline drawing `fs_blit_depth` into a `depth32float` surface.
    fn blit_depth_pipeline(&mut self) -> wgpu::RenderPipeline {
        let key = (wgpu::TextureFormat::Depth32Float, "fs_blit_depth");
        if let Some(pipeline) = self.blit_pipelines.get(&key) {
            return pipeline.clone();
        }
        let (_, module) = self.module("blit", PACKED_DEPTH_WGSL);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit depth"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: key.0,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(key.1),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.blit_pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// The stencil bytes `blit` moves as guest memory holds them, one texel for each in its
    /// destination rectangle. Reading the whole source for them was 13% of Pokémon Legends:
    /// Z-A's frames, most of whose moves take a few texels of a 1920x1080 depth surface.
    fn picked_stencil_texture(&mut self, blit: &Blit, ctx: &ExecCtx) -> Option<wgpu::Texture> {
        let (_, src_bytes) = swizzled_bytes(&blit.src, ctx)?;
        let picked = picked_stencil(blit, &src_bytes)?;
        let (_, _, width, height) = blit.dst_rect;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stencil bytes"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
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
            &picked,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
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

/// The uniform `BLIT_WGSL` and `PACKED_DEPTH_WGSL` read, `tail` being its last field.
fn blit_parameters(blit: &Blit, source: &Target, tail: [u32; 2]) -> Vec<u8> {
    let (x0, y0, _, _) = blit.dst_rect;
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
    bytes
}

/// A surface's bytes as it lies in guest memory, and their CPU address, when one mapping
/// holds them all.
fn swizzled_bytes(surface: &Surface, ctx: &ExecCtx) -> Option<(u64, Vec<u8>)> {
    let len = surface.size();
    let cpu = ctx.span(surface.addr, u64::from(len))?;
    let mut bytes = vec![0u8; len as usize];
    ctx.read_span(cpu, &mut bytes).ok()?;
    Some((cpu, bytes))
}

/// The row and column offsets of each texel of `blit`'s destination rectangle in `surface`,
/// whose texel is picked from `pick`'s coordinates; the source as the engine samples it, the
/// destination where it writes.
fn offsets(
    surface: &Surface,
    bpp: u32,
    pick: impl Fn(u32, bool) -> u32,
    (width, height): (u32, u32),
) -> (Vec<usize>, Vec<usize>) {
    let rows = (0..height)
        .map(|y| {
            surface
                .layout
                .row_offset(pick(y, false), surface.width_bytes()) as usize
        })
        .collect();
    let columns = (0..width)
        .map(|x| surface.layout.column_offset(pick(x, true) * bpp) as usize)
        .collect();
    (rows, columns)
}

/// Whether every texel the offsets name lies within `len` bytes.
fn within((rows, columns): &(Vec<usize>, Vec<usize>), len: usize) -> bool {
    let furthest = |of: &[usize]| of.iter().copied().max().unwrap_or(0);
    furthest(rows) + furthest(columns) < len
}

/// The stencil byte, the first of an `S8D24` word, of each word `blit` moves, row by row over
/// its destination rectangle, picking source texels as the engine does. `None` when a word lies
/// outside `src_bytes`, the source's swizzled bytes.
fn picked_stencil(blit: &Blit, src_bytes: &[u8]) -> Option<Vec<u8>> {
    let src = &blit.src;
    let (_, _, width, height) = blit.dst_rect;
    let pick = |at: u32, column: bool| {
        let (origin, step, size) = if column {
            (blit.src_origin.0, blit.step.0, src.width)
        } else {
            (blit.src_origin.1, blit.step.1, src.height)
        };
        ((origin + step * f64::from(at)).max(0.0) as u32).min(size.saturating_sub(1))
    };
    let picked = offsets(src, src.format.bytes_per_pixel, pick, (width, height));
    if !within(&picked, src_bytes.len()) {
        return None;
    }
    let (rows, columns) = picked;
    Some(
        rows.iter()
            .flat_map(|row| columns.iter().map(move |column| src_bytes[row + column]))
            .collect(),
    )
}

/// Write `picked`, from [`picked_stencil`], over the stencil bytes of `blit`'s destination
/// rectangle in `dst_bytes`, the destination's swizzled bytes. False, having written nothing,
/// when a word lies outside them.
fn place_stencil(blit: &Blit, picked: &[u8], dst_bytes: &mut [u8]) -> bool {
    let dst = &blit.dst;
    let (x0, y0, width, height) = blit.dst_rect;
    let at = |at: u32, column: bool| if column { x0 + at } else { y0 + at };
    let placed = offsets(dst, dst.format.bytes_per_pixel, at, (width, height));
    if !within(&placed, dst_bytes.len()) {
        return false;
    }
    let (rows, columns) = placed;
    for (row, picked) in rows.iter().zip(picked.chunks_exact(width as usize)) {
        for (column, &byte) in columns.iter().zip(picked) {
            dst_bytes[row + column] = byte;
        }
    }
    true
}

/// Whether the engine runs `blit` as a move of whole words into `target`, which holds bytes.
fn moves_words(blit: &Blit, target: &Target) -> bool {
    !blit.filtered
        && blit.src.format.raw == blit.dst.format.raw
        && blit.dst.format.is_byte_exact()
        && matches!(target.format, Format::Rgba8Unorm | Format::Bgra8Unorm)
}
