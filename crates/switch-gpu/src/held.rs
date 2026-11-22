//! Surfaces held on the device: uploads, depth loads, readbacks, and flushes.

use crate::builtin::LOAD_DEPTH_WGSL;
use crate::convert::{depth_texture_format, held_attachment_format, narrow, widen, Widen};
use crate::readback::{Held, Pending, Readback, MAP_FAILED, MAP_READY, MAP_WAITING};
use crate::resources::guest_pages;
use crate::{Gpu, Space, COPY_ALIGNMENT, MAX_FRAMES_IN_FLIGHT};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::renderer::{Flush, Submit};
use switch_core::gpu::upload::{DepthKind, Target};
use switch_core::mem::PAGE_BITS;
use switch_core::{Error, Result};

/// Past this much held across a present, it all goes back to guest memory and is freed.
const MAX_HELD_BYTES: u64 = 512 << 20;

/// Guest memory a target's texels occupy; block-linear rows pad out to whole blocks.
pub(super) fn surface_span(target: &Target) -> std::ops::Range<u64> {
    let len =
        u64::from(target.layout.layer_stride(target.row_bytes, target.rows)).max(target.len());
    target.addr..target.addr + len
}

/// Whether two targets are one depth surface, one cropped from the other.
fn crop_of_each_other(a: &Target, b: &Target) -> bool {
    let contains = |x: &Target, y: &Target| x.width >= y.width && x.height >= y.height;
    a.depth_kind().is_some()
        && Target {
            width: b.width,
            height: b.height,
            rows: b.rows,
            ..*a
        } == *b
        && (contains(a, b) || contains(b, a))
}

/// The memory a held surface owes, which a crop's target alone understates.
fn held_span(held: &Held) -> std::ops::Range<u64> {
    let span = surface_span(&held.target);
    match &held.other {
        Some(other) => span.start..span.end.max(surface_span(&other.target).end),
        None => span,
    }
}

pub(super) fn spans_overlap(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

impl Gpu {
    /// Make `as_id`'s surfaces and caches current, parking the previous address space's.
    pub(super) fn enter_space(&mut self, as_id: u32) {
        if as_id == self.space_id {
            return;
        }
        let space = self.parked.remove(&as_id).unwrap_or_default();
        let previous = std::mem::replace(&mut self.space, space);
        self.parked.insert(self.space_id, previous);
        self.space_id = as_id;
    }

    /// Destroy an address space's device textures; nothing is owed to guest memory that is gone.
    pub(super) fn drop_space(&mut self, space: Space) {
        self.submit_batch();
        for held in space.held.into_values().chain(space.evicted) {
            held.texture.destroy();
            if let Some(other) = &held.other {
                other.texture.destroy();
            }
            if let Some(companion) = &held.companion {
                companion.texture.destroy();
            }
        }
        for pending in space.pending {
            pending.destroy();
        }
        for copy in space.held_copies.into_values() {
            copy.texture.destroy();
        }
        for held in space.globals.into_values() {
            held.buffer.destroy();
        }
        for input in space.inputs.into_values() {
            input.buffer.destroy();
        }
        for (key, made) in space.gpu_textures {
            let bytes = space.texture_cache.get(&key).map_or(0, |b| b.len() as u64);
            for (_, texture) in made {
                texture.destroy();
                self.gpu_texture_bytes = self.gpu_texture_bytes.saturating_sub(bytes);
            }
        }
        let cached: u64 = space.texture_cache.values().map(|b| b.len() as u64).sum();
        self.cached_bytes = self.cached_bytes.saturating_sub(cached);
    }

    /// The device texture for a surface, uploaded on first use.
    pub(super) fn hold(&mut self, target: &Target, ctx: &ExecCtx) -> Result<()> {
        if self.switch_crop(target, true)? {
            return Ok(());
        }
        match self.space.held.get(&target.addr) {
            // The same surface as last time.
            Some(held) if held.target == *target => return Ok(()),
            // A different surface at the same address: the guest rebound it; write the old one back later.
            Some(_) => {
                if let Some(held) = self.space.held.remove(&target.addr) {
                    self.space.evicted.push(held);
                }
            }
            None => {}
        }
        self.note_aliases(target);
        let texture = self.upload_target(target, ctx)?;
        self.space.held.insert(
            target.addr,
            Held {
                texture,
                target: *target,
                dirty: false,
                writes: 0,
                companion: None,
                watched: Vec::new(),
                overwritten: Vec::new(),
                other: None,
            },
        );
        Ok(())
    }

    /// A depth surface drawn beside colour targets of different sizes, cropped for the smaller,
    /// stays on the device at both sizes. Going through guest memory cost two readbacks and an
    /// upload a frame.
    pub(super) fn switch_crop(&mut self, target: &Target, keep_contents: bool) -> Result<bool> {
        let eligible = self.space.held.get(&target.addr).is_some_and(|held| {
            held.companion.is_none()
                && held.target != *target
                && crop_of_each_other(&held.target, target)
        });
        if !eligible {
            return Ok(false);
        }
        let Some(mut old) = self.space.held.remove(&target.addr) else {
            return Ok(false);
        };
        let mut new = match old.other.take() {
            Some(other) if other.target == *target => *other,
            earlier => {
                // A third size: the one before last may hold texels neither of the others does,
                // so the two are merged rather than it dropped. Tears of the Kingdom crops its
                // 384x376 shadow cascade to 384x360 and then to 320x180.
                if let Some(earlier) = earlier {
                    old.other = Some(earlier);
                    old = self.merge_crop(old);
                }
                Held {
                    texture: self.blank_target(target)?,
                    target: *target,
                    dirty: false,
                    writes: 0,
                    companion: None,
                    watched: Vec::new(),
                    overwritten: Vec::new(),
                    other: None,
                }
            }
        };
        if keep_contents {
            self.copy_depth(&old.texture, &new.texture);
            new.writes += 1;
        }
        self.crop_switches += 1;
        new.dirty |= old.dirty;
        // Pages are the larger surface's, whichever is current.
        if old.watched.len() > new.watched.len() {
            new.watched = old.watched.clone();
        }
        new.overwritten.extend(std::mem::take(&mut old.overwritten));
        new.other = Some(Box::new(old));
        self.space.held.insert(target.addr, new);
        Ok(true)
    }

    /// A depth surface held as crops narrower colour targets drew it at, sampled at a larger
    /// size with the same rows, is held at that size instead: the crop already held at it or
    /// one that holds it, or guest memory with the crops drawn over its corner, the current one
    /// last. Each layer of an array is its own surface: Tears of the Kingdom's post-processing
    /// draws beside the last shadow cascade still bound, so that layer alone is held cropped.
    pub(super) fn hold_at_sampled_size(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        ctx: &ExecCtx,
    ) -> Result<()> {
        // A volume's slices interleave; none is a surface of its own size.
        if upload.key.block_depth_gobs > 1 {
            return Ok(());
        }
        for layer in 0..upload.layers.max(1) {
            let addr = upload.key.addr + u64::from(layer) * u64::from(upload.key.layer_stride);
            self.hold_layer_at_sampled_size(addr, upload, ctx)?;
        }
        Ok(())
    }

    /// [`Gpu::hold_at_sampled_size`] for the layer held at `addr`.
    fn hold_layer_at_sampled_size(
        &mut self,
        addr: u64,
        upload: &switch_core::gpu::upload::TextureUpload,
        ctx: &ExecCtx,
    ) -> Result<()> {
        let Some(held) = self.space.held.get(&addr) else {
            return Ok(());
        };
        let sampled = (upload.width, upload.height);
        let size = |target: &Target| (target.width, target.height);
        if held.target.depth_kind().is_none()
            || held.companion.is_some()
            || size(&held.target) == sampled
        {
            return Ok(());
        }
        // A padded surface is sampled out of its corner, which the shadow map pass draws.
        let holds_sampled = |target: &Target| {
            size(target) == sampled
                || (target.width >= sampled.0
                    && target.height >= sampled.1
                    && crate::same_rows(target, &upload.key))
        };
        if let Some(other) = held.other.as_ref().filter(|o| holds_sampled(&o.target)) {
            let target = other.target;
            return self.switch_crop(&target, true).map(|_| ());
        }
        let full = Target {
            width: upload.width,
            height: upload.height,
            rows: upload.rows,
            ..held.target
        };
        let within = |target: &Target| target.width <= full.width && target.height <= full.height;
        if !crate::same_rows(&held.target, &upload.key)
            || !crop_of_each_other(&full, &held.target)
            || !within(&held.target)
            || held.other.as_ref().is_some_and(|o| !within(&o.target))
        {
            return Ok(());
        }
        let texture = self.upload_target(&full, ctx)?;
        let Some(mut crop) = self.space.held.remove(&addr) else {
            return Ok(());
        };
        let mut dirty = crop.dirty;
        if let Some(other) = crop.other.take() {
            self.copy_depth(&other.texture, &texture);
            dirty |= other.dirty;
            self.submit_batch();
            other.texture.destroy();
        }
        self.copy_depth(&crop.texture, &texture);
        self.submit_batch();
        crop.texture.destroy();
        self.crop_switches += 1;
        self.space.held.insert(
            full.addr,
            Held {
                texture,
                target: full,
                dirty,
                writes: crop.writes + 1,
                companion: None,
                watched: Vec::new(),
                other: None,
                overwritten: Vec::new(),
            },
        );
        Ok(())
    }

    /// The whole surface for a readback, with what the crop holds in it.
    fn merge_crop(&mut self, mut held: Held) -> Held {
        let Some(other) = held.other.take() else {
            return held;
        };
        let mut other = *other;
        self.submit_batch();
        if held.target.width >= other.target.width && held.target.height >= other.target.height {
            held.dirty |= other.dirty;
            held.overwritten.append(&mut other.overwritten);
            other.texture.destroy();
            held
        } else {
            self.copy_depth(&held.texture, &other.texture);
            other.dirty |= held.dirty;
            other.overwritten.append(&mut held.overwritten);
            held.texture.destroy();
            other
        }
    }

    /// Draw the part of one depth texture that fits into the top left of another.
    fn copy_depth(&mut self, from: &wgpu::Texture, to: &wgpu::Texture) {
        let pipeline = self.depth_loader(to.format());
        let layout = pipeline.get_bind_group_layout(0);
        let source = from.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth crop"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&source),
            }],
        });
        let view = to.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("depth crop"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth crop"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            // Past the smaller texture there is nothing to read.
            pass.set_scissor_rect(
                0,
                0,
                from.width().min(to.width()),
                from.height().min(to.height()),
            );
            pass.draw(0..3, 0..1);
        }
        self.submit_with_batch(encoder);
    }

    /// Write one held surface back into guest memory and stop holding it.
    fn flush_one(&mut self, addr: u64) {
        let Some(held) = self.space.held.remove(&addr) else {
            return;
        };
        self.ask_for(held);
    }

    /// Ask for a surface back and keep it held, clean: it is what guest memory will say once the
    /// readback lands, so the next draw into it uploads nothing. A swap chain's buffers come back
    /// round to round.
    fn send_back_kept(&mut self, addr: u64) {
        let Some(held) = self.space.held.get(&addr) else {
            return;
        };
        if held.companion.is_some() || held.other.is_some() || held.target.depth_kind().is_some() {
            self.flush_one(addr);
            return;
        }
        let (target, texture, dirty) = (held.target, held.texture.clone(), held.dirty);
        let kept = kept_pages(held);
        if dirty {
            self.submit_batch();
            let pending = self.start_read_back(&target, &texture, kept);
            self.space.pending.push(pending);
        }
        if let Some(held) = self.space.held.get_mut(&addr) {
            held.dirty = false;
            held.watched.clear();
            held.overwritten.clear();
        }
    }

    /// Ask for a surface back without waiting; [`Gpu::flush`] collects it.
    fn ask_for(&mut self, held: Held) {
        self.submit_batch();
        let held = self.merge_crop(held);
        // A companion's contents must be in the surface before it is copied.
        if let Some(companion) = &held.companion {
            let depth = held.target.depth_kind().is_some();
            if let Err(why) = self.resolve_into(&held.texture, companion, depth) {
                self.fall_back(format!("putting a companion surface back: {why}"));
            }
        }
        // A surface nothing drew into is already what guest memory says.
        if held.dirty {
            let pending = self.start_read_back(&held.target, &held.texture, kept_pages(&held));
            self.space.pending.push(pending);
        }
        // Destroyed rather than dropped, so a browser frees it without waiting for GC.
        held.texture.destroy();
        if let Some(companion) = &held.companion {
            companion.texture.destroy();
        }
    }

    /// Whether `target` is held as 32-bit floats for its 16-bit unorm channels; see
    /// [`held_attachment_format`].
    pub(super) fn holds_widened(&self, target: &Target) -> bool {
        target.depth_kind().is_none()
            && held_attachment_format(self.features(), target.format)
                .is_ok_and(|(_, widening)| widening == Widen::Unorm16)
    }

    fn upload_target(&mut self, target: &Target, ctx: &ExecCtx) -> Result<wgpu::Texture> {
        if let Some(kind) = target.depth_kind() {
            return self.upload_depth_target(target, kind, ctx);
        }
        let (format, widening) = held_attachment_format(self.features(), target.format)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render target"),
            size: wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut rows = target.read(ctx)?;
        let mut row_bytes = target.row_bytes;
        if widening == Widen::Unorm16 {
            rows = widen(&rows, widening);
            row_bytes *= 2;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rows,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(target.rows),
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }

    /// Upload a depth surface via an `r32float` staging image and a pass. See [`LOAD_DEPTH_WGSL`].
    fn upload_depth_target(
        &mut self,
        target: &Target,
        kind: DepthKind,
        ctx: &ExecCtx,
    ) -> Result<wgpu::Texture> {
        let format = depth_texture_format(kind);
        let size = wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // A cropped target takes the left of each row.
        let values = target.read_depth(ctx, target.width)?;
        // The staging image is always `r32float`.
        let staging = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth upload"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let floats: Vec<u8> = match kind {
            DepthKind::Float32 => values,
            DepthKind::Unorm16 => values
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|v| {
                    let stored = u16::from_le_bytes(*v);
                    (f32::from(stored) / 65535.0).to_le_bytes()
                })
                .collect(),
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &staging,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &floats,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.width * 4),
                rows_per_image: Some(target.height),
            },
            size,
        );
        self.load_depth(&texture, &staging, format)?;
        staging.destroy();
        Ok(texture)
    }

    /// Draw `staging` into `texture`'s depth.
    pub(super) fn load_depth(
        &mut self,
        texture: &wgpu::Texture,
        staging: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) -> Result<()> {
        for layer in 0..texture.depth_or_array_layers() {
            self.load_depth_layer(texture, layer, staging, layer, format)?;
        }
        Ok(())
    }

    /// One layer at a time: a depth attachment is a single layer.
    pub(super) fn load_depth_layer(
        &mut self,
        texture: &wgpu::Texture,
        layer: u32,
        staging: &wgpu::Texture,
        staging_layer: u32,
        format: wgpu::TextureFormat,
    ) -> Result<()> {
        let one_layer = |base_array_layer: u32| wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer,
            array_layer_count: Some(1),
            ..Default::default()
        };
        let pipeline = self.depth_loader(format);
        let layout = pipeline.get_bind_group_layout(0);
        let view = staging.create_view(&one_layer(staging_layer));
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth upload"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let target = texture.create_view(&one_layer(layer));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("depth upload"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth upload"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.submit_with_batch(encoder);
        Ok(())
    }

    /// The depth loader for this format, built once.
    pub(super) fn depth_loader(&mut self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.depth_loaders.get(&format) {
            return pipeline.clone();
        }
        let (_, module) = self.module("load depth", LOAD_DEPTH_WGSL);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("load depth"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.depth_loaders.insert(format, pipeline.clone());
        pipeline
    }

    /// Start copying a surface off the device, with rows padded to 256 bytes.
    fn start_read_back(
        &mut self,
        target: &Target,
        texture: &wgpu::Texture,
        kept: Vec<u32>,
    ) -> Pending {
        let widened = self.holds_widened(target);
        // Device row width, which for depth or a widened target is not the guest's.
        let row_bytes = match target.depth_kind() {
            Some(kind) => target.width * kind.unit(),
            None if widened => target.row_bytes * 2,
            None => target.row_bytes,
        };
        let aspect = match target.depth_kind() {
            Some(_) => wgpu::TextureAspect::DepthOnly,
            None => wgpu::TextureAspect::All,
        };
        let padded = row_bytes.div_ceil(COPY_ALIGNMENT) * COPY_ALIGNMENT;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded) * u64::from(target.rows),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(target.rows),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        self.submit_with_batch(encoder);

        // Asked for, not waited on.
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(MAP_WAITING));
        let sink = state.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let done = if r.is_ok() { MAP_READY } else { MAP_FAILED };
            sink.store(done, std::sync::atomic::Ordering::Release);
        });
        Pending {
            staging,
            what: Readback::Surface {
                target: *target,
                row_bytes,
                padded,
                widened,
                kept,
            },
            generation: self.space.asked,
            state,
        }
    }

    /// Ask for the buffers atomics wrote back, without waiting; [`Gpu::flush`] collects them.
    fn ask_for_globals(&mut self) {
        self.ask_for_globals_where(|_| true);
    }

    /// Send back the held buffers whose span `wanted` picks.
    fn ask_for_globals_where(&mut self, wanted: impl Fn(&std::ops::Range<u64>) -> bool) {
        let (now, later) = std::mem::take(&mut self.space.globals)
            .into_iter()
            .partition(|(addr, held)| wanted(&(*addr..*addr + held.original.len() as u64)));
        self.space.globals = later;
        let now: Vec<(u64, crate::readback::HeldGlobal)> = now.into_iter().collect();
        for (addr, held) in now {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: held.original.len() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("readback"),
                });
            encoder.copy_buffer_to_buffer(&held.buffer, 0, &staging, 0, held.original.len() as u64);
            self.submit_with_batch(encoder);
            let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(MAP_WAITING));
            let sink = state.clone();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let done = if r.is_ok() { MAP_READY } else { MAP_FAILED };
                sink.store(done, std::sync::atomic::Ordering::Release);
            });
            self.space.pending.push(Pending {
                staging,
                what: Readback::Global {
                    addr,
                    original: held.original,
                    source: held.buffer,
                },
                generation: self.space.asked,
                state,
            });
        }
    }

    /// Copy a finished readback into guest memory, dropping the row padding.
    fn land(&self, pending: &Pending, ctx: &mut ExecCtx) -> Result<()> {
        let slice = pending.staging.slice(..);
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapping the readback: {e}")))?;
        let outcome = match &pending.what {
            // The guest unmapped the surface while its readback was in flight.
            Readback::Surface { target, .. } if ctx.vmm.translate(target.addr).is_none() => Ok(()),
            // Colour writes straight from the padded mapping; depth repacks.
            Readback::Surface {
                target,
                row_bytes,
                padded,
                widened,
                kept,
            } => {
                let page = 1usize << PAGE_BITS;
                let mut guest = vec![0u8; kept.len() * page];
                for (bytes, &at) in guest.chunks_exact_mut(page).zip(kept) {
                    ctx.mem.read_into(at << PAGE_BITS, bytes)?;
                }
                let unpadded = || {
                    let mut rows = Vec::with_capacity((row_bytes * target.rows) as usize);
                    for y in 0..target.rows {
                        let at = (y * padded) as usize;
                        rows.extend_from_slice(&mapped[at..at + *row_bytes as usize]);
                    }
                    rows
                };
                match target.depth_kind() {
                    None if *widened => target.write(ctx, &narrow(&unpadded())),
                    None => target.write_strided(ctx, &mapped, *padded),
                    Some(_) => Self::land_depth(target, &unpadded(), ctx),
                }
                .and_then(|()| {
                    guest
                        .chunks_exact(page)
                        .zip(kept)
                        .try_for_each(|(bytes, &at)| ctx.mem.write_bytes(at << PAGE_BITS, bytes))
                })
            }
            Readback::Global { addr, original, .. } => {
                let words = mapped
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(original.as_chunks::<4>().0);
                (0u64..)
                    .zip(words)
                    .filter(|(_, (now, was))| now != was)
                    .try_for_each(|(word, (now, _))| {
                        let value = u32::from_le_bytes(*now);
                        ctx.write_u32(addr + word * 4, value)
                    })
            }
        };
        drop(mapped);
        pending.staging.unmap();
        // Destroyed rather than dropped; see [`Gpu::scratch`].
        pending.destroy();
        outcome
    }

    /// Put a depth readback back, repacked. A cropped target holds the left of each row.
    fn land_depth(target: &Target, rows: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        target.write_depth(ctx, rows, target.width)
    }

    /// A new surface sharing memory with another at a different address only sees it through
    /// guest memory, which has to catch up.
    pub(super) fn note_aliases(&mut self, target: &Target) {
        let new = surface_span(target);
        let overlapped: Vec<u64> = self
            .space
            .held
            .values()
            .filter(|held| held.target.addr != target.addr && spans_overlap(&held_span(held), &new))
            .map(|held| held.target.addr)
            .collect();
        if !overlapped.is_empty() {
            self.syncs[3] += 1;
            self.space.aliased.extend(overlapped);
            self.space.aliased.push(target.addr);
        }
    }

    /// Whether a surface being drawn into shares memory with what is about to be read from it.
    /// What the guest wrote over since is current in guest memory.
    pub(super) fn note_read_of(&mut self, read: std::ops::Range<u64>) {
        if self.space.held.values().any(|held| {
            held.dirty && spans_overlap(&held_span(held), &read) && !overwritten(held, &read)
        }) {
            self.syncs[4] += 1;
            self.space.sync_all = true;
        }
    }

    /// Start watching the pages of surfaces no readback is about to write, and send back the
    /// surfaces whose watched pages were written, which only the guest can have done.
    pub(super) fn watch_held(&mut self, ctx: &ExecCtx, written: &[u32]) {
        if !written.is_empty() {
            let written: std::collections::HashSet<u32> = written.iter().copied().collect();
            let mut hit = Vec::new();
            for held in self.space.held.values_mut() {
                if !held.watched.iter().any(|page| written.contains(page)) {
                    continue;
                }
                hit.push(held.target.addr);
                // Only pages wholly the surface's: other data shares the rest, and the guest
                // writing that says nothing about the surface's texels.
                let span = held_span(held);
                for (range, page) in gpu_pages(ctx, span) {
                    let whole = range.end - range.start == 1 << PAGE_BITS;
                    if whole
                        && written.contains(&page)
                        && !held.overwritten.iter().any(|(_, p)| *p == page)
                    {
                        held.overwritten.push((range, page));
                    }
                }
            }
            if !hit.is_empty() {
                self.syncs[5] += 1;
                self.space.aliased.extend(hit);
            }
        }
        let pending: Vec<_> = self
            .space
            .pending
            .iter()
            .filter_map(|pending| pending.target().map(surface_span))
            .collect();
        for held in self.space.held.values_mut() {
            let span = held_span(held);
            if !held.watched.is_empty() || pending.iter().any(|p| spans_overlap(p, &span)) {
                continue;
            }
            held.watched = guest_pages(ctx, span.start, span.end - span.start);
            for &page in &held.watched {
                ctx.mem.mark_gpu_page(page << PAGE_BITS);
            }
        }
    }

    /// Ask for only the surface the display reads, with whatever the guest rebound.
    fn ask_for_scanout(&mut self, scanout: Option<u64>) {
        timed!(self, flush_ask, {
            self.space.asked += 1;
            for held in std::mem::take(&mut self.space.evicted) {
                self.ask_for(held);
            }
            if let Some(addr) = scanout {
                self.send_back_kept(addr);
            }
            for addr in std::mem::take(&mut self.space.aliased) {
                self.flush_one(addr);
            }
            self.ask_for_globals();
        });
    }

    /// Ask for every held and evicted surface as one generation.
    fn ask_for_all(&mut self) {
        timed!(self, flush_ask, {
            self.space.sync_all = false;
            self.space.aliased.clear();
            self.space.asked += 1;
            for held in std::mem::take(&mut self.space.evicted) {
                self.ask_for(held);
            }
            let addresses: Vec<u64> = self.space.held.keys().copied().collect();
            for addr in addresses {
                self.flush_one(addr);
            }
            self.ask_for_globals();
        });
    }

    /// Check on the readbacks, then land the oldest ones that have arrived.
    fn land_ready(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        use std::sync::atomic::Ordering;
        let polled = if self.defer_readbacks {
            self.device.poll(wgpu::PollType::Poll)
        } else {
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
        };
        let _ = polled;
        while let Some(first) = self.space.pending.first() {
            match first.state.load(Ordering::Acquire) {
                MAP_WAITING => break,
                MAP_FAILED => {
                    self.space.pending.remove(0).destroy();
                    return Err(Error::Gpu(match self.device_error() {
                        Some(e) => format!("the readback was not mapped: {e}"),
                        None => "the readback was not mapped".into(),
                    }));
                }
                _ => {
                    let pending = self.space.pending.remove(0);
                    self.land(&pending, ctx)?;
                }
            }
        }
        Ok(())
    }

    /// The newest ask whose readbacks have all landed.
    fn landed(&self) -> u64 {
        self.space
            .pending
            .first()
            .map_or(self.space.asked, |p| p.generation - 1)
    }

    /// Start a presented frame's readbacks while earlier frames' are still on their way.
    ///
    /// Only the surface the display reads goes back; the rest stay on the device, since
    /// a readback and a re-upload per surface per frame was most of a frame's time. Whatever
    /// reads guest memory instead flushes first, or asks for this to do it all ([`Space::sync_all`]).
    /// A guest reading a render target's memory directly, with no copy, is not seen.
    pub(super) fn submit_frame_inner(
        &mut self,
        ctx: &mut ExecCtx,
        scanout: Option<u32>,
    ) -> Result<Submit> {
        if self.give_up() {
            return Ok(Submit::Done);
        }
        self.submit_batch();
        self.release_scratch();
        let held_bytes: u64 = self.space.held.values().map(|h| h.target.len()).sum();
        let everything = scanout.is_none()
            || self.sync_every_present
            || self.space.sync_all
            || held_bytes > MAX_HELD_BYTES;
        for (cause, hit) in [
            scanout.is_none(),
            held_bytes > MAX_HELD_BYTES,
            self.space.sync_all,
        ]
        .into_iter()
        .enumerate()
        {
            self.syncs[cause] += u64::from(hit);
        }
        let shown = scanout.and_then(|cpu| {
            self.space
                .held
                .values()
                .map(|held| &held.target)
                .find_map(|target| {
                    (ctx.vmm.translate(target.addr).map(|(at, _)| at) == Some(cpu))
                        .then_some(target.addr)
                })
        });
        let owed = !self.space.globals.is_empty()
            || if everything {
                !self.space.held.is_empty()
            } else {
                shown.is_some()
                    || self
                        .space
                        .aliased
                        .iter()
                        .any(|addr| self.space.held.contains_key(addr))
            };
        if owed || !self.space.evicted.is_empty() {
            let mut asks: Vec<u64> = self.space.pending.iter().map(|p| p.generation).collect();
            asks.dedup();
            if asks.len() >= MAX_FRAMES_IN_FLIGHT {
                return Ok(Submit::Full);
            }
            if everything {
                self.ask_for_all();
            } else {
                self.ask_for_scanout(shown);
            }
        }
        self.space.aliased.clear();
        timed!(self, flush_wait, self.land_ready(ctx))?;
        if self.space.pending.is_empty() {
            return Ok(Submit::Done);
        }
        // A readback outlived the call that asked for it. See [`Gpu::deferred_readbacks`].
        self.deferred_readbacks = true;
        Ok(Submit::Queued(self.space.asked))
    }

    pub(super) fn frames_landed_inner(&mut self, ctx: &mut ExecCtx) -> Result<u64> {
        if self.give_up() {
            return Ok(u64::MAX);
        }
        timed!(self, flush_wait, self.land_ready(ctx))?;
        Ok(self.landed())
    }

    pub(super) fn flush_inner(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        // After a loss, report rather than error: a flush also runs inside GPU submissions.
        if self.give_up() {
            return Ok(Flush::Done);
        }
        self.submit_batch();
        self.release_scratch();
        // Nothing held, owed or in flight: nothing to do (and no poll).
        if self.space.held.is_empty()
            && self.space.evicted.is_empty()
            && self.space.globals.is_empty()
            && self.space.pending.is_empty()
        {
            return Ok(Flush::Done);
        }
        // Only once per frame.
        if self.space.pending.is_empty() {
            self.ask_for_all();
        }
        self.wait_and_land(ctx)
    }

    /// [`Gpu::flush_inner`] for an engine about to read `read` and overwrite `written`: only what
    /// is held there goes back. A surface only read stays held, as a scanned-out one does.
    pub(super) fn sync_ranges_inner(
        &mut self,
        read: &[std::ops::Range<u64>],
        written: &[std::ops::Range<u64>],
        ctx: &mut ExecCtx,
    ) -> Result<Flush> {
        if self.give_up() {
            return Ok(Flush::Done);
        }
        self.submit_batch();
        self.release_scratch();
        let touches = |span: &std::ops::Range<u64>, ranges: &[std::ops::Range<u64>]| {
            ranges.iter().any(|range| spans_overlap(span, range))
        };
        timed!(self, flush_ask, {
            self.space.asked += 1;
            let (now, later): (Vec<Held>, Vec<Held>) = std::mem::take(&mut self.space.evicted)
                .into_iter()
                .partition(|held| {
                    let span = held_span(held);
                    touches(&span, read) || touches(&span, written)
                });
            self.space.evicted = later;
            for held in now {
                self.ask_for(held);
            }
            let mut owed: Vec<(u64, bool)> = self
                .space
                .held
                .iter()
                .filter_map(|(&addr, held)| {
                    let span = held_span(held);
                    let overwritten = touches(&span, written);
                    (overwritten || touches(&span, read)).then_some((addr, overwritten))
                })
                .collect();
            owed.sort_unstable();
            for (addr, overwritten) in owed {
                if overwritten {
                    self.flush_one(addr);
                } else {
                    self.send_back_kept(addr);
                }
            }
            self.ask_for_globals_where(|span| touches(span, read) || touches(span, written));
        });
        self.wait_and_land(ctx)
    }

    /// Wait for the readbacks asked for and put them in guest memory.
    fn wait_and_land(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        if self.space.pending.is_empty() {
            return Ok(Flush::Done);
        }
        // `Wait` blocks natively and does nothing on the web, where the present waits for a later slice.
        // `--defer-readbacks` skips it to reproduce the browser natively.
        let _ = timed!(self, flush_wait, {
            if self.defer_readbacks {
                self.device.poll(wgpu::PollType::Poll)
            } else {
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(5)),
                })
            }
        });
        use std::sync::atomic::Ordering;
        if self
            .space
            .pending
            .iter()
            .any(|p| p.state.load(Ordering::Acquire) == MAP_WAITING)
        {
            // A readback outlived its flush. See [`Gpu::deferred_readbacks`].
            self.deferred_readbacks = true;
            return Ok(Flush::Pending);
        }
        timed!(self, flush_land, {
            for pending in std::mem::take(&mut self.space.pending) {
                if pending.state.load(Ordering::Acquire) == MAP_FAILED {
                    pending.destroy();
                    // Include the device's reason if it left one.
                    return Err(Error::Gpu(match self.device_error() {
                        Some(e) => format!("the readback was not mapped: {e}"),
                        None => "the readback was not mapped".into(),
                    }));
                }
                self.land(&pending, ctx)?;
            }
        });
        Ok(Flush::Done)
    }
}

/// The CPU pages a readback of `held` must leave as the guest wrote them.
fn kept_pages(held: &Held) -> Vec<u32> {
    held.overwritten.iter().map(|&(_, page)| page).collect()
}

/// Whether the guest wrote over every page of `read` that `held` covers since watching began.
pub(super) fn overwritten(held: &Held, read: &std::ops::Range<u64>) -> bool {
    let span = held_span(held);
    let (start, end) = (read.start.max(span.start), read.end.min(span.end));
    let mut at = start;
    while at < end {
        let Some((range, _)) = held
            .overwritten
            .iter()
            .find(|(range, _)| range.contains(&at))
        else {
            return false;
        };
        at = range.end;
    }
    true
}

/// Each guest page under a GPU span, with the part of the span it backs.
fn gpu_pages(ctx: &ExecCtx, span: std::ops::Range<u64>) -> Vec<(std::ops::Range<u64>, u32)> {
    let page = 1u64 << PAGE_BITS;
    let mut pages = Vec::new();
    let mut at = span.start;
    while at < span.end {
        let Some((cpu, run)) = ctx.vmm.translate(at) else {
            break;
        };
        // To the end of this CPU page, the mapped run, or the span, whichever is first.
        let in_page = page - u64::from(cpu) % page;
        let take = in_page.min(run).min(span.end - at);
        if take == 0 {
            break;
        }
        pages.push((at..at + take, cpu >> PAGE_BITS));
        at += take;
    }
    pages
}
