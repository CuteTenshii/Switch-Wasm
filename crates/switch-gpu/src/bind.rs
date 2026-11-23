//! Bind groups and the textures and samplers they bind.

use crate::builtin::{DEPTH_TO_COLOUR_WGSL, PACKED_DEPTH_WGSL};
use crate::convert::{can_filter, sampled_texture_format, widen, Widen};
use crate::held::{overwritten, spans_overlap};
use crate::readback::{HeldGlobal, Scratch};
use crate::{
    same_rows, CachedBinding, GlobalUpload, Gpu, HeldCopy, HeldLayer, Patch, Prepared, Red,
    Resource, SamplerKey, MAX_GLOBAL, TEXTURE_BINDING,
};
use switch_core::gpu::engine::threed::ShaderStage;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::shader::wgsl::{self, Layout};
use switch_core::gpu::surface::{
    block_linear_volume_offset, Layout as SurfaceLayout, GOB_HEIGHT, GOB_WIDTH,
};
use switch_core::gpu::upload::TextureKey;
use switch_core::gpu::upload::{ConstantUpload, Uploads};

/// A sampled texture's held layers, and each one's surface's write count.
type HeldLayers = (Vec<(u32, HeldLayer)>, Vec<u64>);

/// `a == b` for byte slices, a word at a time: wasm's `memcmp` goes a byte at a time, and a cache
/// hit compares tens of kilobytes of constants.
pub(super) fn same_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let (a_blocks, a_rest) = a.as_chunks::<32>();
    let (b_blocks, b_rest) = b.as_chunks::<32>();
    let word = |block: &[u8; 32], at: usize| {
        u64::from_ne_bytes(block[at..at + 8].try_into().expect("eight bytes"))
    };
    a_blocks.iter().zip(b_blocks).all(|(x, y)| {
        (word(x, 0) ^ word(y, 0))
            | (word(x, 8) ^ word(y, 8))
            | (word(x, 16) ^ word(y, 16))
            | (word(x, 24) ^ word(y, 24))
            == 0
    }) && a_rest.iter().zip(b_rest).all(|(x, y)| x == y)
}

const K: [u64; 4] = [
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
    0x27D4_EB2F_1656_67C5,
];

/// Mix `bytes` into four lanes; four keep the multiplies from waiting on each other.
fn mix_bytes(lanes: &mut [u64; 4], bytes: &[u8]) {
    let (blocks, rest) = bytes.as_chunks::<32>();
    for block in blocks {
        for (lane, word) in lanes.iter_mut().zip(block.as_chunks::<8>().0) {
            *lane = (*lane ^ u64::from_le_bytes(*word))
                .wrapping_mul(K[1])
                .rotate_left(31);
        }
    }
    for &byte in rest {
        lanes[0] = (lanes[0] ^ u64::from(byte)).wrapping_mul(K[2]);
    }
}

fn fold_lanes(lanes: [u64; 4]) -> u64 {
    lanes.iter().zip(K).fold(0, |acc, (lane, k)| {
        (acc ^ lane).wrapping_mul(k).rotate_left(27)
    })
}

/// A fast fingerprint of a buffer's bytes; see [`fingerprint`].
pub(super) fn bytes_fingerprint(bytes: &[u8]) -> u64 {
    let mut lanes = [bytes.len() as u64, 1, 2, 3];
    mix_bytes(&mut lanes, bytes);
    fold_lanes(lanes)
}

/// A fast fingerprint of what a stage binds. Constants run to tens of kilobytes a draw, which
/// the default hasher is too slow for.
fn fingerprint(entries: &[wgpu::BindGroupLayoutEntry], resources: &[Resource]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut small = std::collections::hash_map::DefaultHasher::new();
    entries.hash(&mut small);
    let mut lanes = [small.finish(), 1, 2, 3];
    for resource in resources {
        let Resource::Bytes(binding, bytes) = resource else {
            let mut other = std::collections::hash_map::DefaultHasher::new();
            resource.hash(&mut other);
            lanes[0] = (lanes[0] ^ other.finish())
                .wrapping_mul(K[0])
                .rotate_left(29);
            continue;
        };
        lanes[0] = (lanes[0] ^ u64::from(*binding) ^ (bytes.len() as u64) << 32)
            .wrapping_mul(K[0])
            .rotate_left(29);
        mix_bytes(&mut lanes, bytes);
    }
    fold_lanes(lanes)
}

/// A stage's banks as the table `cbRead` indexes, then the banks themselves, whole words.
pub(super) fn pack_banks<'a>(banks: impl Iterator<Item = &'a ConstantUpload>) -> Vec<u8> {
    pack_banks_at(banks).0
}

/// [`pack_banks`], and where each bank's bytes start in the result; `None` for one left out.
fn pack_banks_at<'a>(
    banks: impl Iterator<Item = &'a ConstantUpload>,
) -> (Vec<u8>, Vec<Option<u64>>) {
    let table = wgsl::CONSTANT_BANKS as usize * 2;
    let mut bytes = vec![0u8; table * 4];
    let mut offsets = Vec::new();
    for bank in banks {
        if bank.bank >= wgsl::CONSTANT_BANKS {
            offsets.push(None);
            continue;
        }
        let at = bank.bank as usize * 4 * 2;
        let words = bank.bytes.len() / 4;
        let start = (bytes.len() / 4) as u32;
        bytes[at..at + 4].copy_from_slice(&start.to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&(words as u32).to_le_bytes());
        offsets.push(Some(bytes.len() as u64));
        bytes.extend_from_slice(&bank.bytes[..words * 4]);
    }
    (bytes, offsets)
}

/// A stage's bind group and its layout, and its constant buffer when compute launches wrote
/// over it, with the copies that put their bytes there.
type StageBinding = (
    wgpu::BindGroupLayout,
    wgpu::BindGroup,
    Vec<u32>,
    Vec<(wgpu::Buffer, Vec<Patch>)>,
);

impl Gpu {
    /// One stage's bindings: constant banks, and textures with their samplers. With them, the
    /// group's dynamic offsets, and the constant buffer and the copies of what compute launches
    /// wrote, to record before the draw.
    pub(super) fn bind_group(
        &mut self,
        p: &Prepared,
        stage: ShaderStage,
        group: u32,
    ) -> std::result::Result<StageBinding, String> {
        // The layout's dimensionality, as the module declared it.
        let declared = if stage == ShaderStage::VertexB {
            &p.vs_layout
        } else {
            &p.fs_layout
        };
        let mut entries = Vec::new();
        let mut resources: Vec<Resource> = Vec::new();
        // Whether everything it binds outlives the draw, so the group can serve the next one.
        let mut persistent = true;
        // Why it is not: a held surface, a shadow map, a texture the cache does not keep, a global.
        let mut why = [false; 4];
        let mut constant_patches = Vec::new();
        let mut dynamic_offsets = Vec::new();
        let mut patched = None;
        // Buffers of this draw's own that copies fill before its pass.
        let mut snapshots = Vec::new();
        // Only its own stage reads a stage's buffers, and each stage has a storage buffer limit.
        let own = if stage == ShaderStage::VertexB {
            wgpu::ShaderStages::VERTEX
        } else {
            wgpu::ShaderStages::FRAGMENT
        };

        if p.uploads.constants.iter().any(|c| c.stage == stage) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: wgsl::CONSTANTS_BINDING,
                visibility: own,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            });
            let (banks, patches): (Vec<_>, Vec<_>) = p
                .uploads
                .constants
                .iter()
                .zip(&p.constant_patches)
                .filter(|(c, _)| c.stage == stage)
                .unzip();
            let (packed, offsets) = pack_banks_at(banks.into_iter());
            for (offset, patches) in offsets.into_iter().zip(patches) {
                let Some(offset) = offset else {
                    continue;
                };
                constant_patches.extend(patches.iter().map(|patch| Patch {
                    to: patch.to + offset,
                    ..patch.clone()
                }));
            }
            // The copies land in it, so it is this draw's alone.
            let staged = if constant_patches.is_empty() {
                self.stage_constants(&packed)
            } else {
                None
            };
            let (buffer, offset, size) = match staged {
                Some((offset, size)) => (self.constant_arena(), offset, size),
                None => {
                    persistent = false;
                    let buffer = self.buffer("constants", &packed, wgpu::BufferUsages::STORAGE);
                    let size = buffer.size();
                    if !constant_patches.is_empty() {
                        patched = Some((buffer.clone(), std::mem::take(&mut constant_patches)));
                    }
                    (buffer, 0, size)
                }
            };
            dynamic_offsets.push(offset);
            resources.push(Resource::Window(wgsl::CONSTANTS_BINDING, buffer, size));
        }

        let drawn: Vec<u64> = [p.color, p.depth]
            .iter()
            .flatten()
            .chain(p.images.iter().flat_map(|(layers, _)| layers))
            .map(|t| t.addr)
            .collect();
        for (index, upload) in p
            .uploads
            .textures
            .iter()
            .filter(|t| t.stage == stage)
            .enumerate()
        {
            let declared_texture = declared
                .textures
                .iter()
                .find(|b| b.slot == upload.slot)
                .ok_or("a texture the module never declared")?;
            if let Some(reason) = self.bind_texture(
                upload,
                declared_texture,
                index,
                own,
                &drawn,
                (&mut entries, &mut resources),
            )? {
                persistent = false;
                why[reason] = true;
            }
        }

        for global in p.globals.iter().filter(|g| g.stage == Some(stage)) {
            let binding = wgsl::GLOBAL_BINDING + global.slot;
            let written = global.writes;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: own,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: !written,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            // What a shader writes goes back to the guest, so it is never shared.
            let shared = global.held.is_none() && !written;
            persistent &= shared;
            why[3] |= !shared;
            resources.push(match &global.held {
                Some(held) if global.snapshot => {
                    let copy = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("global snapshot"),
                        size: held.size(),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.scratch.push(Scratch::Buffer(copy.clone()));
                    let whole = Patch {
                        source: held.clone(),
                        from: 0,
                        to: 0,
                        len: held.size(),
                    };
                    snapshots.push((copy.clone(), vec![whole]));
                    Resource::Buffer(binding, copy)
                }
                Some(held) => Resource::Buffer(binding, held.clone()),
                None => Resource::Bytes(binding, global.bytes.to_vec()),
            });
        }

        if stage == ShaderStage::Fragment {
            for (index, (_, layers)) in p.images.iter().enumerate() {
                for (layer, bound) in layers.iter().enumerate() {
                    let binding = declared.image_binding(index, layer as u32);
                    entries.push(bound.entry(binding, own));
                    resources.push(match bound {
                        crate::compute::ImageLayer::Texture(texture) => Resource::Texture(
                            binding,
                            texture.clone(),
                            wgpu::TextureViewDimension::D2,
                        ),
                        crate::compute::ImageLayer::Words { buffer, .. } => {
                            Resource::Buffer(binding, buffer.clone())
                        }
                    });
                    persistent = false;
                    why[0] = true;
                }
            }
        }

        let key = persistent.then(|| fingerprint(&entries, &resources));
        if let Some(cached) = key
            .and_then(|key| self.bind_groups.get(&key))
            .filter(|cached| cached.entries == entries && cached.resources == resources)
        {
            self.bind_hits += 1;
            return Ok((
                cached.layout.clone(),
                cached.group.clone(),
                dynamic_offsets,
                Vec::new(),
            ));
        }
        let remembered = key.map(|key| (key, entries.clone(), resources.clone()));
        if key.is_some() {
            self.bind_misses += 1;
        } else {
            self.bind_uncached += 1;
            for (count, why) in self.bind_uncached_why.iter_mut().zip(why) {
                *count += u64::from(why);
            }
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
        let mut owned = Vec::new();
        let resources: Vec<Resource> = resources
            .into_iter()
            .map(|resource| match resource {
                Resource::Bytes(binding, bytes) => {
                    let buffer = if persistent {
                        let buffer =
                            self.kept_buffer("global", &bytes, wgpu::BufferUsages::STORAGE);
                        owned.push(buffer.clone());
                        buffer
                    } else {
                        self.buffer("global", &bytes, wgpu::BufferUsages::STORAGE)
                    };
                    Resource::Buffer(binding, buffer)
                }
                other => other,
            })
            .collect();
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
                Resource::Window(binding, buffer, size) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(*size),
                    }),
                },
                Resource::Texture(binding, _, _) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(view.as_ref().expect("a view")),
                },
                Resource::Sampler(binding, sampler) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                Resource::Bytes(..) => unreachable!("made into buffers above"),
            })
            .collect();
        let group_name = format!("group {group}");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&group_name),
            layout: &layout,
            entries: &bindings,
        });
        if let Some((key, entries, resources)) = remembered {
            self.bind_groups.insert(
                key,
                CachedBinding {
                    entries,
                    resources,
                    layout: layout.clone(),
                    group: bind_group.clone(),
                    buffers: owned,
                },
            );
        }
        Ok((
            layout,
            bind_group,
            dynamic_offsets,
            patched.into_iter().chain(snapshots).collect(),
        ))
    }

    /// Bind one texture and its sampler at `TEXTURE_BINDING + 2 * index`, never sampling in
    /// place a surface in `drawn`. Answers why the binding does not outlive the work, if it
    /// does not, as an index into [`Gpu::bind_uncached_why`].
    pub(super) fn bind_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        declared_texture: &wgsl::TextureBinding,
        index: usize,
        own: wgpu::ShaderStages,
        drawn: &[u64],
        (entries, resources): (&mut Vec<wgpu::BindGroupLayoutEntry>, &mut Vec<Resource>),
    ) -> std::result::Result<Option<usize>, String> {
        use switch_core::gpu::shader::isa::TexDim;
        let compare = declared_texture.compare;
        let uint = declared_texture.uint;
        // An integer image is read texel by texel, which needs no filter.
        if !compare && !uint {
            let (format, _) = sampled_texture_format(self.features(), upload.format)
                .map_err(|e| format!("{e:?}"))?;
            if !can_filter(format, self.features()) {
                return Err(format!(
                    "samples a {format:?} texture, which this device cannot filter"
                ));
            }
        }
        let dim = declared_texture.dim;
        let fold = declared_texture.fold;
        let view_dimension = match dim {
            // Folded tiles are layers, whatever the program samples the image as.
            _ if fold.is_some() => wgpu::TextureViewDimension::D2Array,
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
            visibility: own,
            ty: wgpu::BindingType::Texture {
                sample_type: if compare {
                    wgpu::TextureSampleType::Depth
                } else if uint {
                    wgpu::TextureSampleType::Uint
                } else {
                    wgpu::TextureSampleType::Float { filterable: true }
                },
                view_dimension,
                multisampled: false,
            },
            count: None,
        });
        // A texture sharing an earlier one's sampler binds none of its own.
        let own_sampler = declared_texture.sampler.is_none();
        if own_sampler {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding + 1,
                visibility: own,
                ty: wgpu::BindingType::Sampler(if compare {
                    wgpu::SamplerBindingType::Comparison
                } else if uint {
                    wgpu::SamplerBindingType::NonFiltering
                } else {
                    wgpu::SamplerBindingType::Filtering
                }),
                count: None,
            });
        }
        // A null image reads zero through any view.
        let null_cube;
        let upload = if upload.is_null() && view_dimension == wgpu::TextureViewDimension::Cube {
            null_cube = upload.null_layers(6);
            &null_cube
        } else {
            upload
        };
        // A size-only query binds as 2D, and the device refuses a 2D view of a layered texture.
        if view_dimension == wgpu::TextureViewDimension::D2 && upload.layers > 1 {
            return Err(format!(
                "binds a {}-layer texture where the program declared a 2D one",
                upload.layers
            ));
        }
        // A cube view takes exactly six layers and a cube array a multiple of six; anything
        // else invalidates the whole command buffer.
        let layers = upload.layers.max(1);
        let cube_layers_fit = match view_dimension {
            wgpu::TextureViewDimension::Cube => layers == 6,
            wgpu::TextureViewDimension::CubeArray => layers.is_multiple_of(6),
            _ => true,
        };
        if !cube_layers_fit {
            return Err(format!(
                "binds a {layers}-layer texture where the program declared a {view_dimension:?}"
            ));
        }
        // Nothing the device holds is as tall as an image that needs folding.
        // `prepare` settled that the held surface's corner is sampled in place.
        let cropped = match declared_texture.crop {
            Some(_) => Some(
                self.space
                    .held
                    .get(&upload.key.addr)
                    .map(|held| held.texture.clone())
                    .ok_or("a cropped surface that is no longer held")?,
            ),
            None => None,
        };
        // A size query reads no texels, so guest memory's serve however stale.
        let (held, writes) = match (fold, &cropped) {
            (None, None) if !declared_texture.sized_only => {
                self.held_layers(upload, compare, view_dimension)?
            }
            _ => (Vec::new(), Vec::new()),
        };
        let direct = match cropped {
            Some(surface) => {
                self.cropped_binds += 1;
                Some(surface)
            }
            None => self.directly_sampled(drawn, upload, view_dimension, compare, &held)?,
        };
        let mut reason = None;
        let texture = if let Some(surface) = direct {
            surface
        } else if !held.is_empty() {
            reason = Some(0);
            self.texture_over_held(upload, view_dimension, &held, &writes, compare)?
        } else if compare {
            reason = Some(1);
            self.shadow_texture(upload)?
        } else {
            let texture =
                self.texture(upload, view_dimension, fold, !declared_texture.sized_only)?;
            let kept = self
                .space
                .gpu_textures
                .get(&upload.key)
                .is_some_and(|made| made.iter().any(|(_, kept)| *kept == texture));
            if !kept {
                reason = Some(2);
            }
            texture
        };
        resources.push(Resource::Texture(binding, texture, view_dimension));
        if own_sampler {
            resources.push(Resource::Sampler(
                binding + 1,
                self.sampler(upload, compare, uint, fold.is_some()),
            ));
        }
        Ok(reason)
    }

    /// Layers of a sampled texture that are surfaces still held on the device, whose guest
    /// memory is stale, each with its surface's [`Held::writes`](crate::readback::Held). One that
    /// cannot be copied is refused.
    fn held_layers(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<HeldLayers, String> {
        let mut layers = Vec::new();
        let mut writes = Vec::new();
        // A deep-block volume's slices interleave inside each block.
        let deep_volume = view == wgpu::TextureViewDimension::D3 && upload.key.block_depth_gobs > 1;
        for layer in 0..upload.layers.max(1) {
            let addr = if deep_volume {
                volume_slice_addr(&upload.key, layer)
            } else {
                upload.key.addr + u64::from(layer) * u64::from(upload.key.layer_stride)
            };
            let Some(held) = self.space.held.get(&addr) else {
                continue;
            };
            // The guest wrote this layer over the surface since; guest memory has it.
            let len = if upload.layers > 1 {
                u64::from(upload.key.layer_stride)
            } else {
                upload.source_len
            };
            if overwritten(held, &(addr..addr + len)) {
                continue;
            }
            writes.push(held.writes);
            // A held `ZF32` or `ZF32_X24S8` is copied through a buffer into an `R32` view; a depth
            // texture cannot copy to colour. `S8` is the other zero-bit depth, and has no depth.
            if let Some(depth) = held.target.depth {
                let float = depth.depth_bits == 0 && depth.bytes >= 4;
                let (format, _) = sampled_texture_format(self.features(), upload.format)
                    .map_err(|e| format!("{e:?}"))?;
                // A shadow map is copied depth to depth.
                if compare {
                    let (width, height) = (held.texture.width(), held.texture.height());
                    let depth32 = held.texture.format() == wgpu::TextureFormat::Depth32Float;
                    // A 16-bit one is copied depth to depth too, into a texture of its own format.
                    let depth16 = held.texture.format() == wgpu::TextureFormat::Depth16Unorm;
                    if (depth32 || depth16) && (width, height) == (upload.width, upload.height) {
                        layers.push((layer, HeldLayer::Shadow(held.texture.clone())));
                        continue;
                    }
                    // A padded depth surface sampled at its drawn size goes through the shadow map pass.
                    if (depth32 || depth16)
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
                // `S8D24` sampled as itself takes its depth from the device; see [`PACKED_DEPTH_WGSL`].
                let packed = (
                    depth.bytes,
                    depth.depth_bits,
                    depth.depth_shift,
                    depth.stencil_shift,
                ) == (4, 24, 8, Some(0));
                // `Z16` sampled as `R16` takes the unorm depth the device holds.
                let unorm16 = (
                    depth.bytes,
                    depth.depth_bits,
                    depth.depth_shift,
                    depth.stencil_shift,
                ) == (2, 16, 0, None)
                    && upload.format == switch_core::gpu::pipeline::Format::R16Unorm;
                let (wanted, held_as) = if packed {
                    (
                        wgpu::TextureFormat::Rg32Float,
                        wgpu::TextureFormat::Depth32Float,
                    )
                } else if unorm16 {
                    (format, wgpu::TextureFormat::Depth16Unorm)
                } else {
                    (
                        wgpu::TextureFormat::R32Float,
                        wgpu::TextureFormat::Depth32Float,
                    )
                };
                let refused = if !float && !packed && !unorm16 {
                    Some(format!(
                        "packed {depth:?}, which is not the float it would be read as"
                    ))
                } else if format != wanted
                    || held.texture.format() != held_as
                    || ((packed || unorm16) && held.texture.sample_count() != 1)
                {
                    Some(format!(
                        "held as {:?} and sampled as {format:?}",
                        held.texture.format()
                    ))
                } else if upload.width > held.texture.width()
                    || upload.height > held.texture.height()
                    || ((held.texture.width(), held.texture.height())
                        != (upload.width, upload.height)
                        && !same_rows(&held.target, &upload.key))
                {
                    Some(format!(
                        "held {}x{} ({:?}, {}) and sampled {}x{} ({:?}, {})",
                        held.texture.width(),
                        held.texture.height(),
                        held.target.layout,
                        match &held.other {
                            Some(other) => format!(
                                "also held at {}x{}",
                                other.target.width, other.target.height
                            ),
                            None => "at no other size".to_string(),
                        },
                        upload.width,
                        upload.height,
                        upload.key.layout,
                        if same_rows(&held.target, &upload.key) {
                            "the same rows"
                        } else {
                            "other rows"
                        }
                    ))
                } else {
                    None
                };
                if let Some(why) = refused {
                    return Err(format!("samples a depth surface held on the device, {why}"));
                }
                let surface = held.texture.clone();
                layers.push((
                    layer,
                    if packed {
                        HeldLayer::PackedDepth(surface)
                    } else if unorm16 {
                        HeldLayer::UnormDepth(surface)
                    } else {
                        HeldLayer::Depth(surface)
                    },
                ));
                continue;
            }
            if held.companion.is_some() {
                return Err("samples a multisampled surface held on the device".into());
            }
            // A slice drawn as a target of the volume's block depth is laid out as the volume
            // lays it; within one GOB any slice is addressed as a plain surface is.
            let slice = deep_volume.then(|| slice_layout(&upload.key));
            let rows = match slice {
                Some(layout) if held.target.layout == layout => TextureKey {
                    layout,
                    ..upload.key
                },
                _ => upload.key,
            };
            if deep_volume && rows.layout != held.target.layout && !slice_fits_one_gob(&upload.key)
            {
                return Err(format!(
                    "samples a surface held on the device as a slice of an interleaved volume \
                     (slice {layer} of {}x{}x{} {:?} at {:#x}, {:?}, {} GOBs deep; held {}x{} {:?})",
                    upload.width,
                    upload.height,
                    upload.layers,
                    upload.format,
                    upload.key.addr,
                    upload.key.layout,
                    upload.key.block_depth_gobs,
                    held.target.width,
                    held.target.height,
                    held.target.layout
                ));
            }
            // A texture may be the top-left corner of a larger surface with the same rows.
            let size = held.texture.size();
            if !same_rows(&held.target, &rows)
                || upload.width > size.width
                || upload.height > size.height
            {
                return Err(format!(
                    "samples a {}x{} image out of a {}x{} surface held on the device \
                     that lays its rows out differently (layer {layer} of {} at {:#x}): \
                     sampled {:?} {:?} stride {:#x}, held {:?} {:?} dirty {}",
                    upload.width,
                    upload.height,
                    size.width,
                    size.height,
                    upload.layers,
                    upload.key.addr,
                    upload.format,
                    upload.key.layout,
                    upload.key.layer_stride,
                    held.texture.format(),
                    held.target.layout,
                    held.dirty
                ));
            }
            let (format, widening) = sampled_texture_format(self.features(), upload.format)
                .map_err(|e| format!("{e:?}"))?;
            // A float colour surface sampled as a shadow map is drawn into it as depth.
            if compare {
                if held.texture.format() != wgpu::TextureFormat::R32Float
                    || format != wgpu::TextureFormat::R32Float
                {
                    return Err(format!(
                        "samples a {:?} surface held on the device as a shadow map, as {:?}",
                        held.texture.format(),
                        upload.format
                    ));
                }
                layers.push((layer, HeldLayer::ShadowCorner(held.texture.clone())));
                continue;
            }
            // A surface held widened holds the texels the widened image would.
            if self.holds_widened(&held.target) {
                if widening == Widen::Unorm16 && format == held.texture.format() {
                    layers.push((layer, HeldLayer::Colour(held.texture.clone())));
                    continue;
                }
                return Err(format!(
                    "samples a {:?} surface held on the device as {:?}, as {:?}",
                    held.target.format,
                    held.texture.format(),
                    upload.format
                ));
            }
            // A copy may change nothing but whether the format is sRGB.
            if widening != Widen::None
                || format.remove_srgb_suffix() != held.texture.format().remove_srgb_suffix()
            {
                if !compare && same_bytes_per_texel(held.texture.format(), format, widening, upload)
                {
                    layers.push((layer, HeldLayer::Bytes(held.texture.clone())));
                    continue;
                }
                return Err(format!(
                    "samples a {:?} surface held on the device as {format:?} \
                     (layer {layer} of {} at {:#x}): sampled {}x{} {:?} {:?}, held {}x{} {:?}",
                    held.texture.format(),
                    upload.layers,
                    upload.key.addr,
                    upload.width,
                    upload.height,
                    upload.format,
                    upload.key.layout,
                    size.width,
                    size.height,
                    held.target.layout
                ));
            }
            layers.push((layer, HeldLayer::Colour(held.texture.clone())));
        }
        Ok((layers, writes))
    }

    /// The corner of a held colour surface a padded image can be sampled from in place, when
    /// that filters exactly as the copy [`Gpu::texture_over_held`] would make: clamped to the
    /// edge on both axes, and only ever sampled, never gathered, fetched or measured.
    pub(super) fn crop_of_held(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
        binding: &wgsl::TextureBinding,
        read_otherwise: &[u32],
        drawn: [Option<u64>; 2],
    ) -> Option<wgsl::Crop> {
        use switch_core::gpu::shader::isa::TexDim;
        use switch_core::gpu::texture::Wrap;
        let edge = |wrap: Wrap| matches!(wrap, Wrap::ClampToEdge | Wrap::ClampToBorder);
        if binding.compare
            || binding.uint
            || binding.fold.is_some()
            || binding.dim != TexDim::T2d
            || upload.layers > 1
            || read_otherwise.contains(&binding.slot.key())
            || matches!(
                binding.slot,
                switch_core::gpu::texture::TextureSlot::Level { .. }
            )
            || !edge(upload.sampler.wrap_u)
            || !edge(upload.sampler.wrap_v)
            || drawn.contains(&Some(upload.key.addr))
        {
            return None;
        }
        let held = self.space.held.get(&upload.key.addr)?;
        let surface = &held.texture;
        let (full_width, full_height) = (surface.width(), surface.height());
        let padded = (full_width, full_height) != (upload.width, upload.height);
        if held.target.depth.is_some()
            || held.companion.is_some()
            || !held.overwritten.is_empty()
            || !padded
            || upload.width > full_width
            || upload.height > full_height
            || surface.sample_count() != 1
            || surface.mip_level_count() != 1
            || surface.dimension() != wgpu::TextureDimension::D2
            || surface.depth_or_array_layers() != 1
            || !same_rows(&held.target, &upload.key)
        {
            return None;
        }
        let (format, widening) = sampled_texture_format(self.features(), upload.format).ok()?;
        let as_held = widening == Widen::None
            || (widening == Widen::Unorm16 && self.holds_widened(&held.target));
        (as_held && format == surface.format()).then_some(wgsl::Crop {
            width: upload.width,
            height: upload.height,
            full_width,
            full_height,
        })
    }

    /// The held colour surface itself, when sampling it is the same as sampling the copy that
    /// [`Gpu::texture_over_held`] would make: one layer of the same size and format, which this
    /// draw does not also render to.
    fn directly_sampled(
        &self,
        drawn: &[u64],
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
        compare: bool,
        held: &[(u32, HeldLayer)],
    ) -> std::result::Result<Option<wgpu::Texture>, String> {
        let [(0, HeldLayer::Colour(surface))] = held else {
            return Ok(None);
        };
        let size = surface.size();
        if compare
            || view != wgpu::TextureViewDimension::D2
            || upload.layers.max(1) != 1
            || (size.width, size.height, size.depth_or_array_layers)
                != (upload.width, upload.height, 1)
            || surface.sample_count() != 1
            || surface.mip_level_count() != 1
            || surface.dimension() != wgpu::TextureDimension::D2
        {
            return Ok(None);
        }
        let (format, widening) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        // `held_layers` stands a held surface in for a widened image only when it is held widened.
        if !matches!(widening, Widen::None | Widen::Unorm16) || format != surface.format() {
            return Ok(None);
        }
        // A pass cannot sample what it draws to.
        let target = |addr: &u64| self.space.held.get(addr).map(|held| &held.texture);
        if drawn.iter().any(|addr| target(addr) == Some(surface)) {
            return Ok(None);
        }
        Ok(Some(surface.clone()))
    }

    /// Upload a texture, then copy the held layers over it. When the held layers are the whole
    /// texture, guest memory is not read and the copy is kept until a source is written: a
    /// title samples one shadow map in hundreds of draws. A kept copy whose sources were only
    /// written or added to takes just those layers: Breath of the Wild draws an array layer by
    /// layer, sampling it after every draw.
    pub(super) fn texture_over_held(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
        held: &[(u32, HeldLayer)],
        writes: &[u64],
        compare: bool,
    ) -> std::result::Result<wgpu::Texture, String> {
        let sixteen = |surface: &HeldLayer| {
            matches!(surface, HeldLayer::Shadow(t)
                if t.format() == wgpu::TextureFormat::Depth16Unorm)
        };
        // A 16-bit shadow map is copied together only from 16-bit layers, every one held; beside
        // others they are drawn into the float map the rest fill, which holds their depths
        // exactly. Tears of the Kingdom holds some of its four cascades on the device.
        let whole_sixteen =
            held.len() == upload.layers.max(1) as usize && held.iter().all(|(_, s)| sixteen(s));
        let drawn: Vec<(u32, HeldLayer)>;
        let held = if compare && !whole_sixteen && held.iter().any(|(_, s)| sixteen(s)) {
            drawn = held
                .iter()
                .map(|(layer, surface)| match surface {
                    HeldLayer::Shadow(texture) if sixteen(surface) => {
                        (*layer, HeldLayer::ShadowCorner(texture.clone()))
                    }
                    other => (*layer, other.clone()),
                })
                .collect();
            &drawn[..]
        } else {
            held
        };
        // Packed depth keeps its stencil from guest memory.
        let covered = held.len() == upload.layers.max(1) as usize
            && !held
                .iter()
                .any(|(_, surface)| matches!(surface, HeldLayer::PackedDepth(_)));
        let key = (upload.key, view, compare);
        let sources: Vec<(u32, wgpu::Texture, u64)> = held
            .iter()
            .zip(writes)
            .map(|((layer, surface), &writes)| (*layer, surface.surface().clone(), writes))
            .collect();
        let guest = (!covered).then(|| upload.bytes.clone());
        let depths: Vec<(u32, wgpu::Texture)> = held
            .iter()
            .filter_map(|(layer, surface)| match surface {
                HeldLayer::PackedDepth(surface) => Some((*layer, surface.clone())),
                _ => None,
            })
            .collect();
        let (_, widening) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        // Packed depth is converted from the whole image at once, so no layer can be redone alone.
        let by_layer = depths.is_empty() && widening != Widen::PackedDepth;
        let mut update = None;
        if let Some(copy) = self.space.held_copies.get(&key) {
            let same_guest = match (&copy.guest, &guest) {
                (None, None) => true,
                (Some(was), Some(now)) => std::sync::Arc::ptr_eq(was, now),
                _ => false,
            };
            if same_guest && copy.sources == sources {
                self.held_copies_reused += 1;
                return Ok(copy.texture.clone());
            }
            if same_guest && by_layer {
                update = stale_layers(&copy.sources, &sources, compare)
                    .map(|layers| (copy.texture.clone(), layers));
            }
        }
        if let Some((texture, layers)) = update {
            let stale: Vec<(u32, HeldLayer)> = held
                .iter()
                .filter(|(layer, _)| layers.contains(layer))
                .cloned()
                .collect();
            // Draws recorded before this are submitted first, so they still sample the old layers.
            if let Err(why) = self.fill_held_layers(&texture, upload, &stale, false) {
                if let Some(copy) = self.space.held_copies.remove(&key) {
                    self.scratch.push(Scratch::Texture(copy.texture));
                }
                return Err(why);
            }
            self.held_copies_updated += 1;
            if let Some(copy) = self.space.held_copies.get_mut(&key) {
                copy.sources = sources;
            }
            return Ok(texture);
        }
        self.held_copies_made += 1;
        // `held_layers` lets these through only as the single layer of a 2D image.
        let packed_words = match held {
            [(0, HeldLayer::Bytes(surface))] if widening == Widen::PackedDepth => Some(surface),
            _ => None,
        };
        if packed_words.is_some() && view == wgpu::TextureViewDimension::D3 {
            return Err("samples a packed depth image as a volume".into());
        }
        let texture = if compare {
            if whole_sixteen {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("held 16-bit shadow map"),
                    size: wgpu::Extent3d {
                        width: upload.width.max(1),
                        height: upload.height.max(1),
                        depth_or_array_layers: upload.layers.max(1),
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth16Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            } else if covered {
                self.blank_shadow_texture(upload)
            } else {
                self.unkept_shadow_texture(upload)?
            }
        } else if let Some(surface) = packed_words {
            self.packed_depth_from_held(upload, surface)?
        } else if covered {
            self.blank_texture(upload, view)?
        } else {
            self.upload_texture(upload, view, None, &depths)?.0
        };
        if let Err(why) = self.fill_held_layers(&texture, upload, held, packed_words.is_some()) {
            self.scratch.push(Scratch::Texture(texture));
            return Err(why);
        }
        if self.space.held_copies.len() >= crate::MAX_HELD_COPIES {
            let stale: Vec<HeldCopy> = self
                .space
                .held_copies
                .drain()
                .map(|(_, copy)| copy)
                .collect();
            self.scratch
                .extend(stale.into_iter().map(|copy| Scratch::Texture(copy.texture)));
        }
        let copy = HeldCopy {
            texture: texture.clone(),
            sources,
            guest,
        };
        if let Some(stale) = self.space.held_copies.insert(key, copy) {
            // A draw recorded before this one may still sample it.
            self.scratch.push(Scratch::Texture(stale.texture));
        }
        Ok(texture)
    }

    /// Fill `held`'s layers of `texture` from their surfaces, after the draws recorded so far.
    /// `packed_words` leaves a packed depth image's words to [`Gpu::packed_depth_from_held`].
    fn fill_held_layers(
        &mut self,
        texture: &wgpu::Texture,
        upload: &switch_core::gpu::upload::TextureUpload,
        held: &[(u32, HeldLayer)],
        packed_words: bool,
    ) -> std::result::Result<(), String> {
        let mut unorm_depths = Vec::new();
        for (layer, surface) in held {
            match surface {
                HeldLayer::ShadowCorner(surface) => {
                    self.draw_shadow_corner(texture, *layer, surface)?;
                }
                HeldLayer::UnormDepth(surface) => {
                    unorm_depths.push(self.depth_to_colour(surface, texture.format()));
                }
                _ => {}
            }
        }
        let mut unorm_depths = unorm_depths.into_iter();
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
                texture,
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
                HeldLayer::ShadowCorner(_) | HeldLayer::PackedDepth(_) => {}
                HeldLayer::UnormDepth(_) => {
                    let colour = unorm_depths.next().expect("drawn above");
                    encoder.copy_texture_to_texture(colour.as_image_copy(), into, extent);
                    self.scratch.push(Scratch::Texture(colour));
                }
                HeldLayer::Bytes(_) if packed_words => {}
                HeldLayer::Bytes(surface) => {
                    self.copy_through_buffer(&mut encoder, surface, into, extent);
                }
                HeldLayer::Depth(surface) => {
                    // A depth copy takes the whole surface; the sampled corner is taken from the buffer.
                    let (width, height) = (surface.width(), surface.height());
                    // Four bytes a texel, rows padded for a texture-buffer copy.
                    let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                    let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("held depth"),
                        size: u64::from(row) * u64::from(height),
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let layout = wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(height),
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
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
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
        self.submit_with_batch(encoder);
        Ok(())
    }

    /// Copy the top-left `extent` of a colour surface into `into` byte for byte, whatever the
    /// two formats are, through a buffer; texture copies need matching formats.
    fn copy_through_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        surface: &wgpu::Texture,
        into: wgpu::TexelCopyTextureInfo,
        extent: wgpu::Extent3d,
    ) {
        let texel = surface.format().block_copy_size(None).unwrap_or(4);
        let row = (extent.width * texel).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("held bytes"),
            size: u64::from(row) * u64::from(extent.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: Some(extent.height),
        };
        encoder.copy_texture_to_buffer(
            surface.as_image_copy(),
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

    /// An `S8D24` image made of a held colour surface's bytes, converted as an upload of the
    /// same words would be.
    fn packed_depth_from_held(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        surface: &wgpu::Texture,
    ) -> std::result::Result<wgpu::Texture, String> {
        let (format, _) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = |label, format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | usage,
                view_formats: &[],
            })
        };
        let words = texture(
            "held words",
            wgpu::TextureFormat::R32Uint,
            wgpu::TextureUsages::COPY_DST,
        );
        let converted = texture("held copy", format, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("held words"),
            });
        self.copy_through_buffer(&mut encoder, surface, words.as_image_copy(), size);
        self.submit_with_batch(encoder);
        self.convert_packed_depth(&words, &converted, &[]);
        self.scratch.push(Scratch::Texture(words));
        Ok(converted)
    }

    /// What [`Gpu::upload_texture`] would make for this image, with nothing written into it.
    fn blank_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
    ) -> std::result::Result<wgpu::Texture, String> {
        let (format, _) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        let dimension = match view {
            wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
            _ => wgpu::TextureDimension::D2,
        };
        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("held copy"),
            size: wgpu::Extent3d {
                width: upload.width.max(1),
                height: upload.height.max(1),
                depth_or_array_layers: upload.layers.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }))
    }

    /// Convert every layer of an `S8D24` image's `words` into `into`; a layer in `depths` takes its
    /// depth from that held `depth32float` instead of from its words. See [`PACKED_DEPTH_WGSL`].
    pub(super) fn convert_packed_depth(
        &mut self,
        words: &wgpu::Texture,
        into: &wgpu::Texture,
        depths: &[(u32, wgpu::Texture)],
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("packed depth"),
            });
        for layer in 0..words.depth_or_array_layers() {
            let held = depths.iter().find(|(at, _)| *at == layer).map(|(_, d)| d);
            let pipeline = self.packed_depth_pipeline(held.is_some());
            let layer_view = |texture: &wgpu::Texture| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            };
            let words = layer_view(words);
            let depth = held.map(|surface| {
                surface.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                })
            });
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&words),
            }];
            if let Some(depth) = &depth {
                entries.push(wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                });
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("packed depth"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let target = layer_view(into);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("packed depth"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
            pass.draw(0..3, 0..1);
        }
        self.submit_with_batch(encoder);
    }

    /// The pipeline of [`Gpu::convert_packed_depth`], built once per source of depth.
    fn packed_depth_pipeline(&mut self, held: bool) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.packed_depth.get(&held) {
            return pipeline.clone();
        }
        let (_, module) = self.module("packed depth", PACKED_DEPTH_WGSL);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("packed depth"),
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
                    entry_point: Some(if held { "fs_held" } else { "fs_words" }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.packed_depth.insert(held, pipeline.clone());
        pipeline
    }

    /// Draw the top-left corner of a held depth surface into a smaller shadow map layer.
    fn draw_shadow_corner(
        &mut self,
        shadow: &wgpu::Texture,
        layer: u32,
        surface: &wgpu::Texture,
    ) -> std::result::Result<(), String> {
        self.load_depth_layer(shadow, layer, surface, 0, wgpu::TextureFormat::Depth32Float)
            .map_err(|e| format!("{e:?}"))
    }

    /// A texture of `format` the size of a held depth surface, holding its depth in red.
    fn depth_to_colour(
        &mut self,
        surface: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        let colour = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("held depth as colour"),
            size: wgpu::Extent3d {
                width: surface.width(),
                height: surface.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pipeline = self.depth_reader(format);
        let depth = surface.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("held depth as colour"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth),
            }],
        });
        let target = colour.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("held depth as colour"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("held depth as colour"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
            pass.draw(0..3, 0..1);
        }
        self.submit_with_batch(encoder);
        colour
    }

    /// The pipeline of [`Gpu::depth_to_colour`], built once per colour format.
    fn depth_reader(&mut self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.depth_readers.get(&format) {
            return pipeline.clone();
        }
        let (_, module) = self.module("depth to colour", DEPTH_TO_COLOUR_WGSL);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth to colour"),
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
                    entry_point: Some("fs_main"),
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
        self.depth_readers.insert(format, pipeline.clone());
        pipeline
    }

    fn texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        view: wgpu::TextureViewDimension,
        fold: Option<wgsl::Fold>,
        reads_texels: bool,
    ) -> std::result::Result<wgpu::Texture, String> {
        if let Some(made) = self.space.gpu_textures.get(&upload.key) {
            if let Some((_, texture)) = made.iter().find(|(v, _)| *v == view) {
                return Ok(texture.clone());
            }
        }
        // Stale texels answer a size query as well as current ones.
        if reads_texels {
            self.note_read_of(upload.key.addr..upload.key.addr + upload.bytes.len() as u64);
        }
        let (texture, len) = self.upload_texture(upload, view, fold, &[])?;
        // Kept only alongside its cached bytes, which a guest write evicts, or as a null image.
        if upload.is_null() || self.space.texture_cache.contains_key(&upload.key) {
            self.gpu_texture_bytes += len as u64;
            self.space
                .gpu_textures
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
        fold: Option<wgsl::Fold>,
        depths: &[(u32, wgpu::Texture)],
    ) -> std::result::Result<(wgpu::Texture, usize), String> {
        let (format, widening) =
            sampled_texture_format(self.features(), upload.format).map_err(|e| format!("{e:?}"))?;
        let packed = widening == Widen::PackedDepth;
        if packed && view == wgpu::TextureViewDimension::D3 {
            return Err("samples a packed depth image as a volume".into());
        }
        // Formats the device lacks are widened first. See [`convert::Widen`].
        let widened = (!matches!(widening, Widen::None | Widen::PackedDepth))
            .then(|| widen(&upload.bytes, widening));
        let (bytes, row_bytes) = match &widened {
            Some(bytes) => (bytes.as_slice(), upload.row_bytes * 2),
            None => (&*upload.bytes, upload.row_bytes),
        };
        // Each tile's rows, the image's last repeated past its end. See [`wgsl::Fold`].
        let folded = fold.map(|fold| {
            let row = |at: u32| {
                let at = at.min(upload.rows.saturating_sub(1)) as usize * row_bytes as usize;
                bytes.get(at..at + row_bytes as usize).unwrap_or(&[])
            };
            let mut tiles =
                Vec::with_capacity(row_bytes as usize * (fold.tile_rows * fold.layers) as usize);
            for layer in 0..fold.layers {
                for at in 0..fold.tile_rows {
                    tiles.extend_from_slice(row(fold.image_row(layer, at)));
                }
            }
            tiles
        });
        let (bytes, rows) = match &folded {
            Some(tiles) => (tiles.as_slice(), fold.map_or(upload.rows, |f| f.tile_rows)),
            None => (bytes, upload.rows),
        };
        let size = match fold {
            Some(fold) => wgpu::Extent3d {
                width: upload.width.max(1),
                height: fold.tile_rows,
                depth_or_array_layers: fold.layers,
            },
            None => wgpu::Extent3d {
                width: upload.width.max(1),
                height: upload.height.max(1),
                depth_or_array_layers: upload.layers.max(1),
            },
        };
        // A 3D image must be created as one so sampling filters between slices.
        let dimension = match view {
            wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
            _ => wgpu::TextureDimension::D2,
        };
        // Past a limit the device makes an invalid texture that poisons every bind group using it.
        let limits = self.device_limits.clone();
        let (largest, deepest) = match dimension {
            wgpu::TextureDimension::D3 => (
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
            ),
            _ => (
                limits.max_texture_dimension_2d,
                limits.max_texture_array_layers,
            ),
        };
        if size.width > largest || size.height > largest || size.depth_or_array_layers > deepest {
            return Err(format!(
                "a {}x{}x{} {dimension:?} texture at {:#x} ({:?}, {:?}), past the device's \
                 {largest} texels or {deepest} layers",
                size.width,
                size.height,
                size.depth_or_array_layers,
                upload.key.addr,
                upload.format,
                upload.key.layout
            ));
        }
        // Packed depth words are written as they are, then converted into the sampled texture.
        let stored = if packed {
            wgpu::TextureFormat::R32Uint
        } else {
            format
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format: stored,
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
                rows_per_image: Some(rows),
            },
            size,
        );
        if packed {
            let converted = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.convert_packed_depth(&texture, &converted, depths);
            self.scratch.push(Scratch::Texture(texture));
            return Ok((converted, bytes.len() * 2));
        }
        Ok((texture, bytes.len()))
    }

    /// The buffers a draw stage's `ldg`s and atomics reach, one per descriptor traced back to a
    /// constant bank. A written one stays on the device until the next flush.
    pub(super) fn global_uploads(
        &mut self,
        layout: &Layout,
        stage: ShaderStage,
        uploads: &Uploads,
        ctx: &ExecCtx,
    ) -> std::result::Result<Vec<GlobalUpload>, String> {
        let bank = |bank: u8| {
            uploads
                .constants
                .iter()
                .find(|c| c.stage == stage && c.bank == u32::from(bank))
                .map(|c| c.bytes.as_slice())
        };
        self.globals_through(layout, Some(stage), &bank, ctx)
    }

    /// [`Gpu::global_uploads`] for any program: `bank` gives a bound constant bank's bytes,
    /// and `stage` is the draw stage, `None` for a compute launch.
    pub(super) fn globals_through<'b>(
        &mut self,
        layout: &Layout,
        stage: Option<ShaderStage>,
        bank_bytes: &dyn Fn(u8) -> Option<&'b [u8]>,
        ctx: &ExecCtx,
    ) -> std::result::Result<Vec<GlobalUpload>, String> {
        let mut out = Vec::new();
        for (slot, global) in layout.globals.iter().enumerate() {
            let (bank, offset) = (global.bank, global.offset);
            let held =
                bank_bytes(bank).ok_or("a `ldg` descriptor in a bank the program never bound")?;
            let word = |at: usize| -> Option<u32> {
                held.get(at..at + 4)
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
                if let Some(h) =
                    self.space.held.values().find(|h| {
                        (h.target.addr..h.target.addr + h.target.len()).contains(&address)
                    })
                {
                    switch_core::traceln!(
                        "[gpu-tex] ldg buffer of {stage:?} at {address:#x} is inside the held \
                         surface at {:#x}",
                        h.target.addr
                    );
                }
            }
            // A null descriptor is a buffer the draw never reaches, as an unbound one is.
            if address == 0 && !global.written {
                out.push(GlobalUpload {
                    stage,
                    slot: slot as u32,
                    addr: 0,
                    len: 0,
                    bytes: Vec::new(),
                    held: None,
                    writes: false,
                    base: 0,
                    snapshot: false,
                });
                continue;
            }
            let mapping = ctx.vmm.mapping_at(address).ok_or_else(|| {
                format!("a `ldg` descriptor naming {address:#x}, which is unmapped")
            })?;
            let mapped = (mapping.gpu_va + mapping.size - address).min(MAX_GLOBAL);
            // NVN's descriptor carries the buffer's size after its address; without it the
            // buffer runs to the end of its mapping.
            let len = match word(at + 8) {
                Some(size) if size != 0 => u64::from(size).min(mapped),
                _ => mapped,
            } as usize;
            // Atomics work on whole words, so a written buffer covers every word it touches.
            let words = len.next_multiple_of(4).min(mapped as usize & !3);
            let len = if global.written { words } else { len };
            let span = address..address + len as u64;
            // What a device draw wrote is read from where it is, until the flush puts it back.
            if let Some(held) = self.space.globals.get(&address) {
                if held.original.len() != words {
                    return Err(format!(
                        "a buffer at {address:#x} used at two lengths before a flush"
                    ));
                }
                out.push(GlobalUpload {
                    stage,
                    slot: slot as u32,
                    addr: address,
                    len: len as u64,
                    bytes: Vec::new(),
                    held: Some(held.buffer.clone()),
                    writes: global.written,
                    base: 0,
                    snapshot: false,
                });
                continue;
            }
            // Written buffers are sub-allocations of one pool, so their spans overlap by
            // construction. A read inside one, and touching no other, is read from it on the
            // device; any other read of memory a device draw wrote waits for the flush.
            if !global.written {
                let mut touched =
                    self.space.globals.iter().filter(|(&at, g)| {
                        spans_overlap(&(at..at + g.original.len() as u64), &span)
                    });
                if let (Some((&at, held)), None) = (touched.next(), touched.next()) {
                    let inside = at <= address && span.end <= at + held.original.len() as u64;
                    if inside && (address - at).is_multiple_of(4) {
                        out.push(GlobalUpload {
                            stage,
                            slot: slot as u32,
                            addr: address,
                            len: len as u64,
                            bytes: Vec::new(),
                            held: Some(held.buffer.clone()),
                            writes: false,
                            base: ((address - at) / 4) as u32,
                            snapshot: false,
                        });
                        continue;
                    }
                }
            }
            let reads_written = !global.written
                && self
                    .space
                    .globals
                    .iter()
                    .any(|(&at, g)| spans_overlap(&(at..at + g.original.len() as u64), &span));
            if reads_written {
                let held: Vec<String> = self
                    .space
                    .globals
                    .iter()
                    .map(|(at, g)| format!("{at:#x}+{:#x}", g.original.len()))
                    .collect();
                return Err(format!(
                    "a read of {address:#x}+{len:#x} over memory written on the device and not \
                     yet put back (held: {})",
                    held.join(", ")
                ));
            }
            if stage.is_none() && !global.written {
                out.push(GlobalUpload {
                    stage,
                    slot: slot as u32,
                    addr: address,
                    len: len as u64,
                    bytes: Vec::new(),
                    held: Some(self.compute_input(address, len, ctx)?),
                    writes: false,
                    base: 0,
                    snapshot: false,
                });
                continue;
            }
            let mut bytes = vec![0u8; len];
            ctx.vmm
                .read_into(ctx.mem, address, &mut bytes)
                .map_err(|e| format!("{e:?}"))?;
            let held = global.written.then(|| {
                let buffer = self.hold_global(address, &bytes);
                bytes.clear();
                buffer
            });
            out.push(GlobalUpload {
                stage,
                slot: slot as u32,
                addr: address,
                len: len as u64,
                bytes,
                held,
                writes: global.written,
                base: 0,
                snapshot: false,
            });
        }
        Ok(out)
    }

    /// Put guest memory on the device for atomics to work on, owed back at the next flush.
    fn hold_global(&mut self, address: u64, bytes: &[u8]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("written global"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        self.space.globals.insert(
            address,
            HeldGlobal {
                buffer: buffer.clone(),
                original: bytes.to_vec(),
            },
        );
        buffer
    }

    /// A sampled depth image, drawn through [`Gpu::load_depth`] since it cannot be copied in;
    /// destroyed after the draw.
    fn shadow_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<wgpu::Texture, String> {
        let texture = self.unkept_shadow_texture(upload)?;
        self.scratch.push(Scratch::Texture(texture.clone()));
        Ok(texture)
    }

    /// The `depth32float` a shadow map is sampled from, with nothing in it.
    fn blank_shadow_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d {
                width: upload.width.max(1),
                height: upload.height.max(1),
                depth_or_array_layers: upload.layers.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            // Copied into as well as drawn into.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// [`Gpu::shadow_texture`] without handing it to the scratch list.
    fn unkept_shadow_texture(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<wgpu::Texture, String> {
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        let format = wgpu::TextureFormat::Depth32Float;
        let texture = self.blank_shadow_texture(upload);
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
            Format::PackedDepth => (4, 0, Red::StencilBits),
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
                Red::StencilBits => f32::from_bits(u32::from(b[0])),
            };
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    /// `compare` is the binding's question, not the descriptor's.
    /// A folded image wraps in the shader, so its tiles are clamped here, and an integer
    /// image's sampler only addresses, since its texels cannot be filtered.
    fn sampler(
        &mut self,
        upload: &switch_core::gpu::upload::TextureUpload,
        compare: bool,
        integer: bool,
        folded: bool,
    ) -> wgpu::Sampler {
        let key = sampler_key(upload, compare, integer, folded);
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

/// Where slice `z` of a deep-block volume starts: slices interleave inside each block.
pub(crate) fn volume_slice_addr(key: &TextureKey, z: u32) -> u64 {
    let block_height_gobs = match key.layout {
        SurfaceLayout::BlockLinear {
            block_height_gobs, ..
        } => block_height_gobs,
        SurfaceLayout::Pitch { .. } => 1,
    };
    key.addr
        + u64::from(block_linear_volume_offset(
            0,
            0,
            z,
            key.row_bytes,
            key.rows,
            block_height_gobs,
            key.block_depth_gobs,
        ))
}

/// How a slice of a deep-block volume is laid out, drawn as a surface of its own.
pub(crate) fn slice_layout(key: &TextureKey) -> SurfaceLayout {
    match key.layout {
        SurfaceLayout::BlockLinear {
            block_height_gobs, ..
        } => SurfaceLayout::BlockLinear {
            block_height_gobs,
            block_depth_gobs: key.block_depth_gobs,
        },
        pitch @ SurfaceLayout::Pitch { .. } => pitch,
    }
}

/// Whether a volume's slice lies within one GOB, where its texels are addressed as a
/// plain surface's are, so a held surface can stand in for it.
pub(crate) fn slice_fits_one_gob(key: &TextureKey) -> bool {
    key.row_bytes <= GOB_WIDTH && key.rows <= GOB_HEIGHT
}

/// The layers of a kept copy whose sources `now` has written or added, when filling just those
/// makes what a fresh copy would. Every surface it was made from must still be on its layer,
/// and a shadow map gains none: its texture is chosen by what all its layers are.
fn stale_layers(
    was: &[(u32, wgpu::Texture, u64)],
    now: &[(u32, wgpu::Texture, u64)],
    compare: bool,
) -> Option<Vec<u32>> {
    let kept = |(layer, surface, _): &(u32, wgpu::Texture, u64)| {
        now.iter()
            .any(|(at, other, _)| at == layer && other == surface)
    };
    if !was.iter().all(kept) {
        return None;
    }
    let mut stale = Vec::new();
    for (layer, _, writes) in now {
        match was.iter().find(|(at, _, _)| at == layer) {
            Some((_, _, before)) if before == writes => {}
            Some(_) => stale.push(*layer),
            None if compare => return None,
            None => stale.push(*layer),
        }
    }
    Some(stale)
}

/// Whether a held colour surface's texels are the sampled image's texels in another format of
/// the same size, so a byte copy is exact: a 16-bit norm widening would need converting, and
/// an `S8D24` image's words are converted only as the one layer of a 2D image.
fn same_bytes_per_texel(
    held: wgpu::TextureFormat,
    sampled: wgpu::TextureFormat,
    widening: Widen,
    upload: &switch_core::gpu::upload::TextureUpload,
) -> bool {
    let stored = match widening {
        Widen::None => sampled,
        Widen::PackedDepth if upload.layers.max(1) == 1 => wgpu::TextureFormat::R32Uint,
        _ => return false,
    };
    let plain = |format: wgpu::TextureFormat| {
        format.block_dimensions() == (1, 1)
            && format.has_color_aspect()
            && !format.has_depth_aspect()
    };
    plain(held)
        && plain(stored)
        && held.block_copy_size(None).is_some()
        && held.block_copy_size(None) == stored.block_copy_size(None)
}

/// What [`Gpu::sampler`] makes for an image, and caches it by.
pub(crate) fn sampler_key(
    upload: &switch_core::gpu::upload::TextureUpload,
    compare: bool,
    integer: bool,
    folded: bool,
) -> SamplerKey {
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
    SamplerKey {
        compare,
        wrap_u: wrap(upload.sampler.wrap_u),
        wrap_v: if folded {
            wgpu::AddressMode::ClampToEdge
        } else {
            wrap(upload.sampler.wrap_v)
        },
        mag: filter(upload.sampler.mag_linear && !integer),
        min: filter(upload.sampler.min_linear && !integer),
    }
}
