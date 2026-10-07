//! Bind groups and the textures and samplers they bind.

use crate::convert::{sampled_texture_format, widen, Widen};
use crate::readback::Scratch;
use crate::{
    same_rows, GlobalUpload, Gpu, HeldLayer, Prepared, Red, Resource, SamplerKey, MAX_GLOBAL,
    TEXTURE_BINDING,
};
use switch_core::gpu::engine::threed::ShaderStage;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::shader::wgsl::{self, Layout};
use switch_core::gpu::upload::Uploads;

impl Gpu {
    /// One stage's bindings: constant banks, and textures with their samplers.
    pub(super) fn bind_group(
        &mut self,
        p: &Prepared,
        stage: ShaderStage,
        group: u32,
    ) -> std::result::Result<(wgpu::BindGroupLayout, wgpu::BindGroup), String> {
        // The layout's dimensionality, as the module declared it.
        let declared = if stage == ShaderStage::VertexB {
            &p.vs_layout
        } else {
            &p.fs_layout
        };
        let mut entries = Vec::new();
        let mut resources: Vec<Resource> = Vec::new();

        for upload in p.uploads.constants.iter().filter(|c| c.stage == stage) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: upload.bank,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            resources.push(Resource::Buffer(
                upload.bank,
                self.buffer("constants", &upload.bytes, wgpu::BufferUsages::STORAGE),
            ));
        }

        for (index, upload) in p
            .uploads
            .textures
            .iter()
            .filter(|t| t.stage == stage)
            .enumerate()
        {
            use switch_core::gpu::shader::isa::TexDim;
            let declared_texture = declared
                .textures
                .iter()
                .find(|b| b.slot == upload.slot)
                .ok_or("a texture the module never declared")?;
            let compare = declared_texture.compare;
            let dim = declared_texture.dim;
            let view_dimension = match dim {
                TexDim::T2dArray => wgpu::TextureViewDimension::D2Array,
                TexDim::T3d => wgpu::TextureViewDimension::D3,
                // Six faces of a 2D texture, viewed as one cube.
                TexDim::TCube => wgpu::TextureViewDimension::Cube,
                // Six faces to a cube, as many cubes as the array holds.
                TexDim::TCubeArray => wgpu::TextureViewDimension::CubeArray,
                _ => wgpu::TextureViewDimension::D2,
            };
            let binding = TEXTURE_BINDING + 2 * index as u32;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: if compare {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float { filterable: true }
                    },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding + 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(if compare {
                    wgpu::SamplerBindingType::Comparison
                } else {
                    wgpu::SamplerBindingType::Filtering
                }),
                count: None,
            });
            // A size-only query binds as 2D, and the device refuses a 2D view of a layered texture.
            if view_dimension == wgpu::TextureViewDimension::D2 && upload.layers > 1 {
                return Err(format!(
                    "binds a {}-layer texture where the program declared a 2D one",
                    upload.layers
                ));
            }
            let held = self.held_layers(upload, compare, view_dimension)?;
            let texture = if !held.is_empty() {
                self.texture_over_held(upload, view_dimension, &held, compare)?
            } else if compare {
                self.shadow_texture(upload)?
            } else {
                self.texture(upload, view_dimension)?
            };
            resources.push(Resource::Texture(binding, texture, view_dimension));
            resources.push(Resource::Sampler(
                binding + 1,
                self.sampler(upload, compare),
            ));
        }

        for global in p.globals.iter().filter(|g| g.stage == stage) {
            let binding = wgsl::GLOBAL_BINDING + global.slot;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            resources.push(Resource::Buffer(
                binding,
                self.buffer("global", &global.bytes, wgpu::BufferUsages::STORAGE),
            ));
        }

        let layout = match self.group_layouts.get(&entries) {
            Some(layout) => layout.clone(),
            None => {
                let layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("stage"),
                            entries: &entries,
                        });
                self.group_layouts.insert(entries, layout.clone());
                layout
            }
        };
        // The views have to outlive the descriptor that borrows them.
        let views: Vec<Option<wgpu::TextureView>> = resources
            .iter()
            .map(|r| match r {
                Resource::Texture(_, texture, dimension) => {
                    Some(texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(*dimension),
                        ..Default::default()
                    }))
                }
                _ => None,
            })
            .collect();
        let bindings: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .zip(&views)
            .map(|(resource, view)| match resource {
                Resource::Buffer(binding, buffer) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                },
                Resource::Texture(binding, _, _) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(view.as_ref().expect("a view")),
                },
                Resource::Sampler(binding, sampler) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            })
            .collect();
        let group_name = format!("group {group}");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&group_name),
            layout: &layout,
            entries: &bindings,
        });
        Ok((layout, bind_group))
    }

    /// Layers of a sampled texture that are surfaces still held on the device, whose guest
    /// memory is stale. One that cannot be copied is refused.
    fn held_layers(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<Vec<(u32, HeldLayer)>, String> {
        let mut layers = Vec::new();
        for layer in 0..upload.layers.max(1) {
            let addr = upload.key.addr + u64::from(layer) * u64::from(upload.key.layer_stride);
            let Some(held) = self.held.get(&addr) else {
                continue;
            };
            // A held `ZF32` is copied through a buffer into an `R32` view; a depth texture cannot copy to colour.
            if let Some(depth) = held.target.depth {
                let float =
                    depth.bytes == 4 && depth.depth_bits == 0 && depth.stencil_shift.is_none();
                let (format, _) = sampled_texture_format(self.features(), upload.format)
                    .map_err(|e| format!("{e:?}"))?;
                // A shadow map is copied depth to depth.
                if compare {
                    let (width, height) = (held.texture.width(), held.texture.height());
                    let depth32 = held.texture.format() == wgpu::TextureFormat::Depth32Float;
                    if depth32 && (width, height) == (upload.width, upload.height) {
                        layers.push((layer, HeldLayer::Shadow(held.texture.clone())));
                        continue;
                    }
                    // A padded depth surface sampled at its drawn size goes through the shadow map pass.
                    if depth32
                        && upload.width <= width
                        && upload.height <= height
                        && same_rows(&held.target, &upload.key)
                    {
                        layers.push((layer, HeldLayer::ShadowCorner(held.texture.clone())));
                        continue;
                    }
                    return Err(format!(
                        "samples a depth surface held on the device as a shadow map, \
                         held {:?} {width}x{height} as {:?} and sampled {}x{} with {} \
                         layer(s) as {:?}",
                        held.texture.format(),
                        held.target.layout,
                        upload.width,
                        upload.height,
                        upload.layers,
                        upload.key.layout
                    ));
                }
                let refused = if !float {
                    Some(format!(
                        "packed {depth:?}, which is not the float it would be read as"
                    ))
                } else if format != wgpu::TextureFormat::R32Float
                    || held.texture.format() != wgpu::TextureFormat::Depth32Float
                {
                    Some(format!(
                        "held as {:?} and sampled as {format:?}",
                        held.texture.format()
                    ))
                } else if (held.texture.width(), held.texture.height())
                    != (upload.width, upload.height)
                {
                    Some(format!(
                        "held {}x{} and sampled {}x{}",
                        held.texture.width(),
                        held.texture.height(),
                        upload.width,
                        upload.height
                    ))
                } else {
                    None
                };
                if let Some(why) = refused {
                    return Err(format!("samples a depth surface held on the device, {why}"));
                }
                layers.push((layer, HeldLayer::Depth(held.texture.clone())));
                continue;
            }
            if held.companion.is_some() {
                return Err("samples a multisampled surface held on the device".into());
            }
            // Deep-block volume slices interleave, so none is a surface of its own.
            if view == wgpu::TextureViewDimension::D3 && upload.key.block_depth_gobs > 1 {
                return Err(
                    "samples a surface held on the device as a slice of an interleaved volume"
                        .into(),
                );
            }
            // A texture may be the top-left corner of a larger surface with the same rows.
            let size = held.texture.size();
            if !same_rows(&held.target, &upload.key)
                || upload.width > size.width
                || upload.height > size.height
            {
                return Err(format!(
                    "samples a {}x{} image out of a {}x{} surface held on the device \
                     that lays its rows out differently (layer {layer} of {} at {:#x})",
                    upload.width,
                    upload.height,
                    size.width,
                    size.height,
                    upload.layers,
                    upload.key.addr
                ));
            }
            let (format, widening) = sampled_texture_format(self.features(), upload.format)
                .map_err(|e| format!("{e:?}"))?;
            // A copy may change nothing but whether the format is sRGB.
            if widening != Widen::None
                || format.remove_srgb_suffix() != held.texture.format().remove_srgb_suffix()
            {
                return Err(format!(
                    "samples a {:?} surface held on the device as {format:?}",
                    held.texture.format()
                ));
            }
            layers.push((layer, HeldLayer::Colour(held.texture.clone())));
        }
        Ok(layers)
    }

    /// Upload a texture, then copy the held layers over it; never cached.
    fn texture_over_held(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
        held: &[(u32, HeldLayer)],
        compare: bool,
    ) -> std::result::Result<wgpu::Texture, String> {
        let texture = if compare {
            self.shadow_texture(upload)?
        } else {
            let (texture, _) = self.upload_texture(upload, view)?;
            self.scratch.push(Scratch::Texture(texture.clone()));
            texture
        };
        for (layer, surface) in held {
            if let HeldLayer::ShadowCorner(surface) = surface {
                self.draw_shadow_corner(&texture, *layer, surface)?;
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("held layers"),
            });
        let extent = wgpu::Extent3d {
            width: upload.width,
            height: upload.height,
            depth_or_array_layers: 1,
        };
        for (layer, surface) in held {
            let into = wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: *layer,
                },
                aspect: wgpu::TextureAspect::All,
            };
            match surface {
                HeldLayer::Colour(surface) | HeldLayer::Shadow(surface) => {
                    encoder.copy_texture_to_texture(surface.as_image_copy(), into, extent);
                }
                // Drawn above, before these copies were encoded.
                HeldLayer::ShadowCorner(_) => {}
                HeldLayer::Depth(surface) => {
                    // Four bytes a texel, rows padded for a texture-buffer copy.
                    let row = (upload.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                    let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("held depth"),
                        size: u64::from(row) * u64::from(upload.height),
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let layout = wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(upload.height),
                    };
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: surface,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::DepthOnly,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &staging,
                            layout,
                        },
                        extent,
                    );
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: &staging,
                            layout,
                        },
                        into,
                        extent,
                    );
                    self.scratch.push(Scratch::Buffer(staging));
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        Ok(texture)
    }

    /// Draw the top-left corner of a held `depth32float` into a smaller shadow map layer, via `r32float`.
    fn draw_shadow_corner(
        &mut self,
        shadow: &wgpu::Texture,
        layer: u32,
        surface: &wgpu::Texture,
    ) -> std::result::Result<(), String> {
        let size = surface.size();
        let row = (size.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("held shadow"),
            size: u64::from(row) * u64::from(size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("held shadow"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: Some(size.height),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("held shadow"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: surface,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout,
            },
            size,
        );
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout,
            },
            staging.as_image_copy(),
            size,
        );
        self.queue.submit([encoder.finish()]);
        let drawn = self
            .load_depth_layer(
                shadow,
                layer,
                &staging,
                0,
                wgpu::TextureFormat::Depth32Float,
            )
            .map_err(|e| format!("{e:?}"));
        staging.destroy();
        self.scratch.push(Scratch::Buffer(buffer));
        drawn
    }

    fn texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<wgpu::Texture, String> {
        if let Some(made) = self.gpu_textures.get(&upload.key) {
            if let Some((_, texture)) = made.iter().find(|(v, _)| *v == view) {
                return Ok(texture.clone());
            }
        }
        let (texture, len) = self.upload_texture(upload, view)?;
        // Kept only alongside its cached bytes, which a guest write evicts.
        if self.texture_cache.contains_key(&upload.key) {
            self.gpu_texture_bytes += len as u64;
            self.gpu_textures
                .entry(upload.key)
                .or_default()
                .push((view, texture.clone()));
        } else {
            self.scratch.push(Scratch::Texture(texture.clone()));
        }
        Ok(texture)
    }

    /// A texture made from an upload's bytes, and how many bytes went into it.
    fn upload_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<(wgpu::Texture, usize), String> {
        let (format, widening) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        // Formats the device lacks are widened first. See [`convert::Widen`].
        let widened = (widening != Widen::None).then(|| widen(&upload.bytes, widening));
        let (bytes, row_bytes) = match &widened {
            Some(bytes) => (bytes.as_slice(), upload.row_bytes * 2),
            None => (&*upload.bytes, upload.row_bytes),
        };
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        // A 3D image must be created as one so sampling filters between slices.
        let dimension = match view {
            wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
            _ => wgpu::TextureDimension::D2,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
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
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(upload.rows),
            },
            size,
        );
        Ok((texture, bytes.len()))
    }

    /// The buffers a stage's `ldg`s read, one per descriptor traced back to a constant bank.
    pub(super) fn global_uploads(
        &self,
        layout: &Layout,
        stage: ShaderStage,
        uploads: &Uploads,
        ctx: &ExecCtx,
    ) -> std::result::Result<Vec<GlobalUpload>, String> {
        let mut out = Vec::new();
        for (slot, &(bank, offset)) in layout.globals.iter().enumerate() {
            let held = uploads
                .constants
                .iter()
                .find(|c| c.stage == stage && c.bank == u32::from(bank))
                .ok_or("a `ldg` descriptor in a bank the draw never bound")?;
            let word = |at: usize| -> Option<u32> {
                held.bytes
                    .get(at..at + 4)
                    .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
            };
            let at = usize::from(offset);
            let (Some(lo), Some(hi)) = (word(at), word(at + 4)) else {
                return Err(format!(
                    "a `ldg` descriptor at c{bank}[{offset:#x}], past the bank's end"
                ));
            };
            let address = (u64::from(hi) << 32) | u64::from(lo);
            if switch_core::trace::enabled(switch_core::trace::Trace::GpuTex) {
                if let Some(h) = self
                    .held
                    .values()
                    .find(|h| (h.target.addr..h.target.addr + h.target.len()).contains(&address))
                {
                    switch_core::traceln!(
                        "[gpu-tex] ldg buffer of {stage:?} at {address:#x} is inside the held \
                         surface at {:#x}",
                        h.target.addr
                    );
                }
            }
            let mapping = ctx.vmm.mapping_at(address).ok_or_else(|| {
                format!("a `ldg` descriptor naming {address:#x}, which is unmapped")
            })?;
            let len = (mapping.gpu_va + mapping.size - address).min(MAX_GLOBAL) as usize;
            let mut bytes = vec![0u8; len];
            ctx.vmm
                .read_into(ctx.mem, address, &mut bytes)
                .map_err(|e| format!("{e:?}"))?;
            out.push(GlobalUpload {
                stage,
                slot: slot as u32,
                bytes,
            });
        }
        Ok(out)
    }

    /// A sampled depth image, drawn through [`Gpu::load_depth`] since it cannot be copied in.
    fn shadow_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<wgpu::Texture, String> {
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        let format = wgpu::TextureFormat::Depth32Float;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Copied into as well as drawn into.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // The staging image `load_depth` reads: `r32float`.
        let depths = self.shadow_depths(upload)?;
        let staging = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow upload"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &staging,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &depths,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * 4),
                rows_per_image: Some(size.height),
            },
            size,
        );
        self.load_depth(&texture, &staging, format)
            .map_err(|e| format!("{e:?}"))?;
        staging.destroy();
        self.scratch.push(Scratch::Texture(texture.clone()));
        Ok(texture)
    }

    /// One `f32` per texel: the decoded red channel, which a comparison reads.
    fn shadow_depths(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<Vec<u8>, String> {
        use switch_core::gpu::pipeline::Format;
        // (bytes per texel, where red starts in one, how wide red is).
        let (unit, at, red) = match upload.format {
            Format::R32Float => return Ok(upload.bytes.to_vec()),
            Format::R16Unorm => (2, 0, Red::Unorm16),
            Format::R16Float => (2, 0, Red::Float16),
            Format::R8Unorm => (1, 0, Red::Unorm8),
            Format::Rg8Unorm => (2, 0, Red::Unorm8),
            Format::Rgba8Unorm => (4, 0, Red::Unorm8),
            Format::Bgra8Unorm => (4, 2, Red::Unorm8),
            Format::Rgba16Float => (8, 0, Red::Float16),
            Format::Rgba32Float => (8 * 2, 0, Red::Float32),
            other => return Err(format!("a shadow map stored as {other:?}")),
        };
        let mut out = Vec::with_capacity(upload.bytes.len() / unit * 4);
        for texel in upload.bytes.chunks_exact(unit) {
            let b = &texel[at..];
            let value = match red {
                Red::Unorm8 => f32::from(b[0]) / 255.0,
                Red::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
                Red::Float16 => {
                    switch_core::gpu::surface::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                }
                Red::Float32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            };
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    /// `compare` is the binding's question, not the descriptor's.
    fn sampler(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
    ) -> wgpu::Sampler {
        use switch_core::gpu::texture::Wrap;
        let wrap = |w: Wrap| match w {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
            Wrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            // WebGPU has no border mode, and the rasterizer also samples border as edge.
            Wrap::ClampToBorder => wgpu::AddressMode::ClampToEdge,
        };
        let filter = |linear: bool| {
            if linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            }
        };
        use switch_core::gpu::texture::Compare;
        let compare = compare
            .then(|| upload.sampler.compare.unwrap_or(Compare::Always))
            .map(|c| match c {
                Compare::Never => wgpu::CompareFunction::Never,
                Compare::Less => wgpu::CompareFunction::Less,
                Compare::Equal => wgpu::CompareFunction::Equal,
                Compare::LessEqual => wgpu::CompareFunction::LessEqual,
                Compare::Greater => wgpu::CompareFunction::Greater,
                Compare::NotEqual => wgpu::CompareFunction::NotEqual,
                Compare::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
                Compare::Always => wgpu::CompareFunction::Always,
            });
        let key = SamplerKey {
            compare,
            wrap_u: wrap(upload.sampler.wrap_u),
            wrap_v: wrap(upload.sampler.wrap_v),
            mag: filter(upload.sampler.mag_linear),
            min: filter(upload.sampler.min_linear),
        };
        let device = &self.device;
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("sampler"),
                    compare: key.compare,
                    address_mode_u: key.wrap_u,
                    address_mode_v: key.wrap_v,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: key.mag,
                    min_filter: key.min,
                    ..Default::default()
                })
            })
            .clone()
    }
}
