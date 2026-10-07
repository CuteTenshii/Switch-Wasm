//! Surfaces held on the device: uploads, depth loads, readbacks, and flushes.

use crate::builtin::LOAD_DEPTH_WGSL;
use crate::convert::{depth_texture_format, device_attachment_format};
use crate::readback::{Held, Pending, MAP_FAILED, MAP_READY, MAP_WAITING};
use crate::{crop_rows, Gpu, COPY_ALIGNMENT};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::renderer::Flush;
use switch_core::gpu::upload::{DepthKind, Target};
use switch_core::{Error, Result};

impl Gpu {
    /// The device texture for a surface, uploaded on first use.
    pub(super) fn hold(&mut self, target: &Target, ctx: &ExecCtx) -> Result<()> {
        match self.held.get(&target.addr) {
            // The same surface as last time.
            Some(held) if held.target == *target => return Ok(()),
            // A different surface at the same address: the guest rebound it; write the old one back later.
            Some(_) => {
                if let Some(held) = self.held.remove(&target.addr) {
                    self.evicted.push(held);
                }
            }
            None => {}
        }
        let texture = self.upload_target(target, ctx)?;
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

    /// Write one held surface back into guest memory and stop holding it.
    fn flush_one(&mut self, addr: u64) {
        let Some(held) = self.held.remove(&addr) else {
            return;
        };
        self.ask_for(held);
    }

    /// Ask for a surface back without waiting; [`Gpu::flush`] collects it.
    fn ask_for(&mut self, held: Held) {
        // A companion's contents must be in the surface before it is copied.
        if let Some(companion) = &held.companion {
            let depth = held.target.depth_kind().is_some();
            if let Err(why) = self.resolve_into(&held.texture, companion, depth) {
                self.fall_back(format!("putting a companion surface back: {why}"));
            }
        }
        // A surface nothing drew into is already what guest memory says.
        if held.dirty {
            let pending = self.start_read_back(&held.target, &held.texture);
            self.pending.push(pending);
        }
        // Destroyed rather than dropped, so a browser frees it without waiting for GC.
        held.texture.destroy();
        if let Some(companion) = &held.companion {
            companion.texture.destroy();
        }
    }

    fn upload_target(&mut self, target: &Target, ctx: &ExecCtx) -> Result<wgpu::Texture> {
        if let Some(kind) = target.depth_kind() {
            return self.upload_depth_target(target, kind, ctx);
        }
        let format = device_attachment_format(self.features(), target.format)?;
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
        let rows = target.read(ctx)?;
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
                bytes_per_row: Some(target.row_bytes),
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
        let values = target.read_depth(ctx)?;
        let surface_texels = (target.row_bytes / target.unit.max(1)) as usize;
        let values = crop_rows(
            values,
            surface_texels,
            target.width as usize,
            kind.unit() as usize,
        );
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
        self.queue.submit([encoder.finish()]);
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
    fn start_read_back(&self, target: &Target, texture: &wgpu::Texture) -> Pending {
        // Device row width, which for depth is not the guest's texel width.
        let row_bytes = match target.depth_kind() {
            Some(kind) => target.width * kind.unit(),
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
        self.queue.submit([encoder.finish()]);

        // Asked for, not waited on.
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(MAP_WAITING));
        let sink = state.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let done = if r.is_ok() { MAP_READY } else { MAP_FAILED };
            sink.store(done, std::sync::atomic::Ordering::Release);
        });
        Pending {
            staging,
            target: *target,
            row_bytes,
            padded,
            state,
        }
    }

    /// Copy a finished readback into guest memory, dropping the row padding.
    fn land(&self, pending: &Pending, ctx: &mut ExecCtx) -> Result<()> {
        let slice = pending.staging.slice(..);
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapping the readback: {e}")))?;
        let target = &pending.target;
        // Colour writes straight from the padded mapping; depth repacks.
        let outcome = match target.depth_kind() {
            None => target.write_strided(ctx, &mapped, pending.padded),
            Some(kind) => {
                let mut rows = Vec::with_capacity((pending.row_bytes * target.rows) as usize);
                for y in 0..target.rows {
                    let at = (y * pending.padded) as usize;
                    rows.extend_from_slice(&mapped[at..at + pending.row_bytes as usize]);
                }
                Self::land_depth(target, kind, &rows, ctx)
            }
        };
        drop(mapped);
        pending.staging.unmap();
        // Destroyed rather than dropped; see [`Gpu::scratch`].
        pending.staging.destroy();
        outcome
    }

    /// Put a depth readback back, repacked.
    fn land_depth(target: &Target, kind: DepthKind, rows: &[u8], ctx: &mut ExecCtx) -> Result<()> {
        // A cropped depth target holds the left of each row; the rest is preserved.
        let surface_texels = (target.row_bytes / target.unit.max(1)) as usize;
        let window = target.width as usize;
        if window >= surface_texels {
            return target.write_depth(ctx, rows);
        }
        let unit = kind.unit() as usize;
        let mut full = target.read_depth(ctx)?;
        for y in 0..target.rows as usize {
            let from = y * window * unit;
            let to = y * surface_texels * unit;
            let len = window * unit;
            if to + len <= full.len() && from + len <= rows.len() {
                full[to..to + len].copy_from_slice(&rows[from..from + len]);
            }
        }
        target.write_depth(ctx, &full)
    }

    pub(super) fn flush_inner(&mut self, ctx: &mut ExecCtx) -> Result<Flush> {
        // After a loss, report rather than error: a flush also runs inside GPU submissions.
        if self.give_up() {
            return Ok(Flush::Done);
        }
        // Nothing held, owed or in flight: nothing to do (and no poll).
        if self.held.is_empty() && self.evicted.is_empty() && self.pending.is_empty() {
            return Ok(Flush::Done);
        }
        // Only once per frame.
        if self.pending.is_empty() {
            timed!(self, flush_ask, {
                for held in std::mem::take(&mut self.evicted) {
                    self.ask_for(held);
                }
                let addresses: Vec<u64> = self.held.keys().copied().collect();
                for addr in addresses {
                    self.flush_one(addr);
                }
            });
        }
        // `Wait` blocks natively and does nothing on the web, where the present waits for a later slice.
        // `GPU_DEFER_READBACKS=1` skips it to reproduce the browser natively.
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
            .pending
            .iter()
            .any(|p| p.state.load(Ordering::Acquire) == MAP_WAITING)
        {
            // A readback outlived its flush. See [`Gpu::deferred_readbacks`].
            self.deferred_readbacks = true;
            return Ok(Flush::Pending);
        }
        timed!(self, flush_land, {
            for pending in std::mem::take(&mut self.pending) {
                if pending.state.load(Ordering::Acquire) == MAP_FAILED {
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
