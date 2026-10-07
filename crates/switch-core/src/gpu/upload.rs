//! The bytes a draw moves between guest memory and a device: vertices, indices,
//! constants and textures read out, render targets written back. Uploads are bounded
//! by what the draw touches and capped at [`MAX_UPLOAD`].

use crate::gpu::bcn::Codec;
use crate::gpu::engine::threed::{DepthLayout, Engine3D, ShaderStage};
use crate::gpu::exec::ExecCtx;
use crate::gpu::pipeline::{Format, Pipeline, StepMode, VertexBuffer};
use crate::gpu::surface::{ColorFormat, Layout};
use crate::gpu::texture::{self, Sampler, SwizzleSource, TexelKind, Texture, TextureSlot};
use crate::{Error, Result};

/// Which constant banks to resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Banks<'a> {
    /// Every bank the draw has bound.
    Bound,
    /// Only the banks the shaders read, per stage.
    Read(&'a [(ShaderStage, u32)]),
}

/// The most one buffer will be read into memory: 64 MiB.
pub const MAX_UPLOAD: u64 = 64 << 20;

const CONSTBUF_BANKS: u32 = 32;

/// Index width handed to a backend; 8-bit indices are widened to 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// Bytes `count` elements reach: whole strides, except the last, which is
/// only as long as its attributes (as WebGPU sizes it).
fn vertex_span(count: u32, buffer: &VertexBuffer) -> u64 {
    let last = buffer
        .attributes
        .iter()
        .map(|a| a.offset + a.format.size())
        .max()
        .unwrap_or(buffer.stride);
    u64::from(count.saturating_sub(1)) * u64::from(buffer.stride) + u64::from(last)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexUpload {
    /// Matches [`crate::gpu::pipeline::VertexBuffer::index`].
    pub array: u32,
    /// The element these bytes start at; a backend must offset by it.
    pub first: u32,
    pub stride: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexUpload {
    pub format: IndexFormat,
    pub bytes: Vec<u8>,
    /// Lowest and highest index used, which bound the vertex uploads.
    pub lowest: u32,
    pub highest: u32,
}

impl IndexUpload {
    /// The indices, widened, for backends that rewrite fans or quads.
    pub fn indices(&self) -> Vec<u32> {
        match self.format {
            IndexFormat::Uint16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            IndexFormat::Uint32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantUpload {
    pub stage: ShaderStage,
    pub bank: u32,
    pub bytes: Vec<u8>,
}

/// What decides a texture upload's bytes, all from the TIC: a rewritten
/// descriptor is a new key, and texel writes are caught by watched pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub layer_stride: u32,
    pub row_bytes: u32,
    pub rows: u32,
    pub block_depth_gobs: u32,
    pub layout: Layout,
    pub format: Format,
    pub srgb: bool,
}

/// One texture, deswizzled into linear rows; compressed blocks stay encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureUpload {
    /// The stage whose constant buffer named this texture.
    pub stage: ShaderStage,
    pub slot: TextureSlot,
    pub handle: u32,
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    /// Bytes per row of the linear image; a row of blocks when compressed.
    pub row_bytes: u32,
    /// Rows per layer, in texels or blocks.
    pub rows: u32,
    /// The layers back to back, shared so a cache hit costs a refcount.
    pub bytes: std::sync::Arc<[u8]>,
    pub key: TextureKey,
    /// How far past `key.addr` the read reached, for page watching.
    pub source_len: u64,
    /// Channel swizzle, applied by the backend when sampling.
    pub swizzle: [SwizzleSource; 4],
    pub sampler: Sampler,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Uploads {
    pub vertex: Vec<VertexUpload>,
    pub index: Option<IndexUpload>,
    pub constants: Vec<ConstantUpload>,
    pub textures: Vec<TextureUpload>,
}

impl Uploads {
    pub fn len(&self) -> usize {
        self.vertex.iter().map(|v| v.bytes.len()).sum::<usize>()
            + self.index.as_ref().map_or(0, |i| i.bytes.len())
            + self.constants.iter().map(|c| c.bytes.len()).sum::<usize>()
            + self.textures.iter().map(|t| t.bytes.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve what [`Engine3D::last_draw`] reads. `slots` are each stage's
    /// texture handle slots, from the shader translation.
    pub fn of(
        engine: &Engine3D,
        pipeline: &Pipeline,
        ctx: &ExecCtx,
        banks: Banks<'_>,
        slots: &[(ShaderStage, TextureSlot)],
    ) -> Result<Uploads> {
        Uploads::of_cached(engine, pipeline, ctx, banks, slots, &mut |_| None)
    }

    /// [`Uploads::of`], with `cached` answering for textures already read.
    pub fn of_cached(
        engine: &Engine3D,
        pipeline: &Pipeline,
        ctx: &ExecCtx,
        banks: Banks<'_>,
        slots: &[(ShaderStage, TextureSlot)],
        cached: &mut dyn FnMut(&TextureKey) -> Option<std::sync::Arc<[u8]>>,
    ) -> Result<Uploads> {
        let call = engine.last_draw;
        let index = if call.indexed {
            Some(read_indices(
                ctx,
                engine.index_array_start(),
                call.first,
                call.count,
                call.index_format,
            )?)
        } else {
            None
        };

        let mut vertex = Vec::new();
        for buffer in &pipeline.vertex_buffers {
            let array = engine.vertex_array(buffer.index);
            // One instance per draw, so an instanced array reads one element.
            let (first, count) = match buffer.step {
                StepMode::Instance => (engine.instance_id(), 1),
                StepMode::Vertex => match &index {
                    Some(index) => (index.lowest, index.highest - index.lowest + 1),
                    None => (call.first, call.count),
                },
            };
            if count == 0 || buffer.stride == 0 {
                continue;
            }
            let length = vertex_span(count, buffer);
            let start = array.start + u64::from(first) * u64::from(buffer.stride);
            // `limit` is the last valid byte; reads past it are zero, as on hardware.
            let inside = match array.limit {
                0 => length,
                limit => (limit + 1).saturating_sub(start).min(length),
            };
            let mut bytes = if inside == 0 {
                Vec::new()
            } else {
                read_range(ctx, start, inside, "vertex array")?
            };
            bytes.resize(length as usize, 0);
            vertex.push(VertexUpload {
                array: buffer.index,
                first,
                stride: buffer.stride,
                bytes,
            });
        }

        let mut constants = Vec::new();
        for stage in [ShaderStage::VertexB, ShaderStage::Fragment] {
            for bank in 0..CONSTBUF_BANKS {
                if let Banks::Read(wanted) = banks {
                    if !wanted.contains(&(stage, bank)) {
                        continue;
                    }
                }
                let Some((addr, size)) = engine.bound_constbuf(stage, bank) else {
                    continue;
                };
                if size == 0 {
                    continue;
                }
                constants.push(ConstantUpload {
                    stage,
                    bank,
                    bytes: read_range(ctx, addr, u64::from(size), "constant bank")?,
                });
            }
        }

        let mut textures = Vec::new();
        for &(stage, slot) in slots {
            if textures
                .iter()
                .any(|t: &TextureUpload| t.stage == stage && t.slot == slot)
            {
                continue;
            }
            textures.push(read_texture(engine, ctx, stage, slot, cached)?);
        }

        Ok(Uploads {
            vertex,
            index,
            constants,
            textures,
        })
    }
}

/// A surface a draw renders into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub format: Format,
    pub addr: u64,
    /// In texels, which on a multisampled surface is not pixels.
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    pub row_bytes: u32,
    pub rows: u32,
    /// Bytes per texel.
    pub unit: u32,
    /// Depth/stencil packing, for converting to the device format and back.
    pub depth: Option<DepthLayout>,
}

impl Target {
    pub fn len(&self) -> u64 {
        u64::from(self.row_bytes) * u64::from(self.rows)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read the surface out as linear rows.
    pub fn read(&self, ctx: &ExecCtx) -> Result<Vec<u8>> {
        if self.len() > MAX_UPLOAD {
            return Err(Error::Gpu(format!(
                "upload: a {}x{} target is {} bytes, past the {MAX_UPLOAD}-byte cap",
                self.width,
                self.height,
                self.len()
            )));
        }
        let mut out = Vec::with_capacity(self.len() as usize);
        deswizzle(
            ctx,
            self.addr,
            self.layout,
            self.row_bytes,
            self.rows,
            self.unit,
            &mut out,
        )?;
        Ok(out)
    }

    /// Write linear rows back, swizzled.
    pub fn write(&self, ctx: &mut ExecCtx, rows: &[u8]) -> Result<()> {
        self.write_strided(ctx, rows, self.row_bytes)
    }

    /// [`Target::write`] from rows `stride` bytes apart, as a readback pads them.
    pub fn write_strided(&self, ctx: &mut ExecCtx, rows: &[u8], stride: u32) -> Result<()> {
        let want = (stride * self.rows) as usize;
        if rows.len() < want {
            return Err(Error::Gpu(format!(
                "upload: writing back {} bytes of a {want}-byte target",
                rows.len()
            )));
        }
        let per_row = self.row_bytes / self.unit.max(1);
        // Patched over the existing bytes: block-linear padding is not ours to zero.
        if let Some((cpu, mut raw)) = self.mapped(ctx)? {
            // A contiguous run at a time, as `run_at` reports it.
            let width = per_row * self.unit;
            for y in 0..self.rows {
                let mut x = 0;
                while x < width {
                    let (at, run) = self.layout.run_at(x, y, self.row_bytes);
                    let take = run.min(width - x) as usize;
                    if take == 0 {
                        break;
                    }
                    let at = at as usize;
                    let from = ((y * stride) + x) as usize;
                    raw[at..at + take].copy_from_slice(&rows[from..from + take]);
                    x += take as u32;
                }
            }
            return ctx.write_span(cpu, &raw);
        }
        for y in 0..self.rows {
            for x in 0..per_row {
                let offset = self.layout.offset(x * self.unit, y, self.row_bytes);
                let at = self.addr + u64::from(offset);
                let from = ((y * stride) + x * self.unit) as usize;
                let mut value = 0u128;
                for (i, &byte) in rows[from..from + self.unit as usize].iter().enumerate() {
                    value |= u128::from(byte) << (8 * i);
                }
                ctx.write_pixel(at, self.unit, value)?;
            }
        }
        Ok(())
    }

    /// The swizzled bytes and their address, when one mapping holds them all.
    fn mapped(&self, ctx: &ExecCtx) -> Result<Option<(u32, Vec<u8>)>> {
        let swizzled = u64::from(self.layout.layer_stride(self.row_bytes, self.rows));
        let Some(cpu) = ctx.span(self.addr, swizzled) else {
            return Ok(None);
        };
        let mut raw = vec![0u8; swizzled as usize];
        ctx.read_span(cpu, &mut raw)?;
        Ok(Some((cpu, raw)))
    }

    pub fn depth_kind(&self) -> Option<DepthKind> {
        self.depth.map(DepthKind::of)
    }

    /// Read a depth surface as device rows, one [`DepthKind::unit`] per texel, stencil dropped.
    pub fn read_depth(&self, ctx: &ExecCtx) -> Result<Vec<u8>> {
        let (layout, kind) = self.depth_parts()?;
        let rows = self.read(ctx)?;
        let unit = self.unit as usize;
        let mut out = Vec::with_capacity(rows.len() / unit * kind.unit() as usize);
        for texel in rows.chunks_exact(unit) {
            let mut pixel = 0u128;
            for (i, &byte) in texel.iter().enumerate() {
                pixel |= u128::from(byte) << (8 * i);
            }
            out.extend_from_slice(&kind.encode(layout.decode_depth(pixel))[..kind.unit() as usize]);
        }
        Ok(out)
    }

    /// Write device depth rows back in the guest packing, keeping stencil bytes.
    pub fn write_depth(&self, ctx: &mut ExecCtx, values: &[u8]) -> Result<()> {
        let (layout, kind) = self.depth_parts()?;
        let unit = kind.unit() as usize;
        let per_row = self.row_bytes / self.unit.max(1);
        let want = per_row as usize * self.rows as usize * unit;
        if values.len() < want {
            return Err(Error::Gpu(format!(
                "upload: writing back {} bytes of a {want}-byte depth target",
                values.len()
            )));
        }
        // One translation when one mapping holds it; the stencil byte is then in `raw`.
        if let Some((cpu, mut raw)) = self.mapped(ctx)? {
            let texel = self.unit as usize;
            for y in 0..self.rows {
                for x in 0..per_row {
                    let at = self.layout.offset(x * self.unit, y, self.row_bytes) as usize;
                    let from = (y * per_row + x) as usize * unit;
                    let depth = kind.decode(&values[from..from + unit]);
                    let value = if layout.packs_stencil() {
                        let mut old = 0u128;
                        for (i, &byte) in raw[at..at + texel].iter().enumerate() {
                            old |= u128::from(byte) << (8 * i);
                        }
                        layout.with_depth(old, depth)
                    } else {
                        layout.encode_depth(depth)
                    };
                    raw[at..at + texel].copy_from_slice(&value.to_le_bytes()[..texel]);
                }
            }
            return ctx.write_span(cpu, &raw);
        }
        for y in 0..self.rows {
            for x in 0..per_row {
                let at =
                    self.addr + u64::from(self.layout.offset(x * self.unit, y, self.row_bytes));
                let from = (y * per_row + x) as usize * unit;
                let depth = kind.decode(&values[from..from + unit]);
                let value = if layout.packs_stencil() {
                    layout.with_depth(ctx.read_pixel(at, self.unit)?, depth)
                } else {
                    layout.encode_depth(depth)
                };
                ctx.write_pixel(at, self.unit, value)?;
            }
        }
        Ok(())
    }

    /// The colour surface bound at `slot`, or `None`.
    pub fn color(engine: &Engine3D, slot: u32) -> Result<Option<Target>> {
        let Some(rt) = engine.render_target(slot)? else {
            return Ok(None);
        };
        let unit = rt.format.bytes_per_pixel;
        // A disabled target reads back as format 0.
        if unit == 0 {
            return Ok(None);
        }
        Ok(Some(Target {
            format: crate::gpu::pipeline::color_format(rt.format)
                .map_err(|e| Error::Gpu(format!("upload: colour target: {e}")))?,
            addr: rt.addr,
            width: rt.width,
            height: rt.height,
            layout: rt.layout,
            row_bytes: rt.width * unit,
            rows: rt.height,
            unit,
            depth: None,
        }))
    }

    pub fn depth_surface(engine: &Engine3D) -> Result<Option<Target>> {
        let Some(dt) = engine.depth_target()? else {
            return Ok(None);
        };
        Ok(Some(Target {
            format: crate::gpu::pipeline::depth_format(dt.format)
                .map_err(|e| Error::Gpu(format!("upload: depth target: {e}")))?,
            addr: dt.addr,
            width: dt.width,
            height: dt.height,
            layout: dt.layout,
            row_bytes: dt.width * dt.format.bytes,
            rows: dt.height,
            unit: dt.format.bytes,
            depth: Some(dt.format),
        }))
    }

    fn depth_parts(&self) -> Result<(DepthLayout, DepthKind)> {
        let layout = self
            .depth
            .ok_or_else(|| Error::Gpu("upload: a colour target has no depth packing".into()))?;
        Ok((layout, DepthKind::of(layout)))
    }
}

/// A device depth format: `Z16` stays `depth16unorm`, everything else is
/// `depth32float` (lossless for 24-bit depth). WebGPU copies cannot write `depth32float`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthKind {
    Unorm16,
    Float32,
}

impl DepthKind {
    pub fn unit(self) -> u32 {
        match self {
            DepthKind::Unorm16 => 2,
            DepthKind::Float32 => 4,
        }
    }

    pub fn of(layout: DepthLayout) -> DepthKind {
        match (layout.bytes, layout.depth_bits, layout.stencil_shift) {
            (2, 16, None) => DepthKind::Unorm16,
            _ => DepthKind::Float32,
        }
    }

    /// `depth` in `0.0..=1.0` as device texel bytes.
    fn encode(self, depth: f32) -> [u8; 4] {
        match self {
            DepthKind::Unorm16 => {
                let stored = (f64::from(depth.clamp(0.0, 1.0)) * 65535.0 + 0.5) as u16;
                let [a, b] = stored.to_le_bytes();
                [a, b, 0, 0]
            }
            DepthKind::Float32 => depth.to_le_bytes(),
        }
    }

    fn decode(self, bytes: &[u8]) -> f32 {
        match self {
            DepthKind::Unorm16 => f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0,
            DepthKind::Float32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Targets {
    /// Colour target 0, or `None` for a depth-only pass.
    pub color: Option<Target>,
    pub depth: Option<Target>,
}

impl Targets {
    pub fn of(engine: &Engine3D) -> Result<Targets> {
        Ok(Targets {
            color: Target::color(engine, engine.render_target_slot(0))?,
            depth: Target::depth_surface(engine)?,
        })
    }

    /// Bytes of both surfaces: the per-frame round-trip cost.
    pub fn len(&self) -> u64 {
        self.color.map_or(0, |t| t.len()) + self.depth.map_or(0, |t| t.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One GOB, the block-linear unit rows are padded to.
const GOB_BYTES: u64 = 512;

/// Resolve a texture slot and copy it out. A `texs` immediate indexes the
/// `TexCbIndex` constant bank for a bindless handle into the TIC/TSC pools.
fn read_texture(
    engine: &Engine3D,
    ctx: &ExecCtx,
    stage: ShaderStage,
    slot: TextureSlot,
    cached: &mut dyn FnMut(&TextureKey) -> Option<std::sync::Arc<[u8]>>,
) -> Result<TextureUpload> {
    let (bank, offset) = slot.handle_at(engine.tex_cb_index());
    let bank = u32::from(bank);
    let (addr, size) = engine.bound_constbuf(stage, bank).ok_or_else(|| {
        Error::Gpu(format!(
            "upload: {stage:?}'s texture bank {bank} is not bound"
        ))
    })?;
    let offset = u32::from(offset);
    if offset + 4 > size {
        return Err(Error::Gpu(format!(
            "upload: texture handle at c{bank}[{offset:#x}] is past the bound buffer's {size:#x}"
        )));
    }
    let handle = ctx.read_u32(addr + u64::from(offset))?;
    let descriptors = texture::read_descriptors(
        ctx,
        engine.tex_header_pool(),
        engine.tex_sampler_pool(),
        handle,
    )?;
    let image = descriptors.texture;

    let copy = image_copy(&image)?;
    let (format, row_bytes, rows) = copy.shape(&image);
    let layers = image.layers.max(1);
    let total = u64::from(row_bytes) * u64::from(rows) * u64::from(layers);
    if total > MAX_UPLOAD {
        return Err(Error::Gpu(format!(
            "upload: texture {}x{} is {total} bytes, past the {MAX_UPLOAD}-byte cap",
            image.width, image.height
        )));
    }
    let key = TextureKey {
        addr: image.addr,
        width: image.width,
        height: image.height,
        layers,
        layer_stride: image.layer_stride,
        row_bytes,
        rows,
        block_depth_gobs: image.block_depth_gobs,
        layout: image.layout,
        format,
        srgb: image.srgb,
    };
    // How far the read reaches: the last row's last byte plus a GOB, or the
    // dense size, whichever is larger.
    let dense = u64::from(row_bytes) * u64::from(rows) * u64::from(layers);
    let last = u64::from(image.layout.offset(
        row_bytes.saturating_sub(1),
        rows.saturating_sub(1),
        row_bytes,
    ));
    let strided =
        last + GOB_BYTES + u64::from(image.layer_stride) * u64::from(layers.saturating_sub(1));
    let source_len = dense.max(strided);

    if let Some(bytes) = cached(&key) {
        return Ok(TextureUpload {
            stage,
            slot,
            handle,
            format,
            width: image.width,
            height: image.height,
            layers,
            row_bytes,
            rows,
            bytes,
            key,
            source_len,
            swizzle: image.swizzle,
            sampler: descriptors.sampler,
        });
    }

    let mut bytes = Vec::with_capacity(total as usize);
    for layer in 0..layers {
        // A volume's slices interleave inside a block, so read through `Texture::texel`.
        if image.block_depth_gobs > 1 {
            let unit = match copy {
                Copy::Raw { unit } => unit,
                Copy::Decode { .. } => {
                    return Err(Error::Gpu(
                        "upload: a compressed 3D image is not decoded".into(),
                    ))
                }
            };
            for y in 0..rows {
                for x in 0..row_bytes / unit {
                    let at = crate::gpu::surface::block_linear_volume_offset(
                        x * unit,
                        y,
                        layer,
                        row_bytes,
                        rows,
                        match image.layout {
                            Layout::BlockLinear { block_height_gobs } => block_height_gobs,
                            Layout::Pitch { .. } => 1,
                        },
                        image.block_depth_gobs,
                    );
                    let word = ctx.read_pixel(image.addr + u64::from(at), unit)?;
                    bytes.extend_from_slice(&word.to_le_bytes()[..unit as usize]);
                }
            }
            continue;
        }
        let base = image.addr + u64::from(layer) * u64::from(image.layer_stride);
        match copy {
            Copy::Raw { unit } => {
                deswizzle(ctx, base, image.layout, row_bytes, rows, unit, &mut bytes)?
            }
            Copy::Decode { codec } => decode_blocks(ctx, &image, base, codec, &mut bytes)?,
        }
    }
    Ok(TextureUpload {
        stage,
        slot,
        handle,
        format,
        width: image.width,
        height: image.height,
        layers,
        row_bytes,
        rows,
        bytes: std::sync::Arc::from(bytes),
        key,
        source_len,
        swizzle: image.swizzle,
        sampler: descriptors.sampler,
    })
}

#[derive(Debug, Clone, Copy)]
enum Copy {
    /// Deswizzled in the surface's own units, BC codecs staying compressed.
    Raw { unit: u32 },
    /// Decoded to `Rgba8Unorm`, for codecs WebGPU cannot name.
    Decode { codec: Codec },
}

impl Copy {
    fn shape(self, image: &Texture) -> (Format, u32, u32) {
        match self {
            Copy::Raw { unit } => match image.kind {
                TexelKind::Plain(plain) => (
                    plain_format(plain, image.srgb),
                    image.width * unit,
                    image.height,
                ),
                TexelKind::Block(codec) => {
                    let (block_w, block_h) = codec.block_size();
                    (
                        block_format(codec, image.srgb),
                        image.width.div_ceil(block_w) * unit,
                        image.height.div_ceil(block_h),
                    )
                }
                TexelKind::Depth(_) => unreachable!("a depth texel has no WebGPU format"),
            },
            Copy::Decode { .. } => {
                let format = if image.srgb {
                    Format::Rgba8UnormSrgb
                } else {
                    Format::Rgba8Unorm
                };
                (format, image.width * 4, image.height)
            }
        }
    }
}

fn image_copy(image: &Texture) -> Result<Copy> {
    Ok(match image.kind {
        TexelKind::Plain(plain) => {
            crate::gpu::pipeline::color_format(plain)
                .map_err(|e| Error::Gpu(format!("upload: texture format: {e}")))?;
            Copy::Raw {
                unit: plain.bytes_per_pixel,
            }
        }
        // WebGPU fills `depth32float` only from a texture copy, so the rasterizer handles these.
        TexelKind::Depth(depth) => {
            return Err(Error::Gpu(format!(
                "upload: {depth:?} is a depth surface, which cannot be uploaded as a texture"
            )))
        }
        // Desktop browsers lack `texture-compression-astc`, so ASTC is decoded.
        TexelKind::Block(codec @ Codec::Astc { .. }) => Copy::Decode { codec },
        TexelKind::Block(codec) => {
            let (block_w, block_h) = codec.block_size();
            // WebGPU needs whole blocks; partial ones (the Home Menu's 1x1 BC4/BC5) are decoded.
            if image.width.is_multiple_of(block_w) && image.height.is_multiple_of(block_h) {
                Copy::Raw {
                    unit: codec.bytes_per_block(),
                }
            } else {
                Copy::Decode { codec }
            }
        }
    })
}

/// `srgb` is the TIC's flag. Infallible: [`image_copy`] refused unnamed formats.
fn plain_format(plain: ColorFormat, srgb: bool) -> Format {
    let format = crate::gpu::pipeline::color_format(plain).unwrap_or(Format::Rgba8Unorm);
    match (format, srgb) {
        (Format::Rgba8Unorm, true) => Format::Rgba8UnormSrgb,
        (Format::Bgra8Unorm, true) => Format::Bgra8UnormSrgb,
        (format, _) => format,
    }
}

/// Infallible: ASTC is decoded rather than named.
fn block_format(codec: Codec, srgb: bool) -> Format {
    match (codec, srgb) {
        (Codec::Bc1, false) => Format::Bc1RgbaUnorm,
        (Codec::Bc1, true) => Format::Bc1RgbaUnormSrgb,
        (Codec::Bc2, false) => Format::Bc2RgbaUnorm,
        (Codec::Bc2, true) => Format::Bc2RgbaUnormSrgb,
        (Codec::Bc3, false) => Format::Bc3RgbaUnorm,
        (Codec::Bc3, true) => Format::Bc3RgbaUnormSrgb,
        (Codec::Bc4Unorm, _) => Format::Bc4RUnorm,
        (Codec::Bc4Snorm, _) => Format::Bc4RSnorm,
        (Codec::Bc5Unorm, _) => Format::Bc5RgUnorm,
        (Codec::Bc5Snorm, _) => Format::Bc5RgSnorm,
        (Codec::Bc6hUf16, _) => Format::Bc6hRgbUfloat,
        (Codec::Bc6hSf16, _) => Format::Bc6hRgbFloat,
        (Codec::Bc7, false) => Format::Bc7RgbaUnorm,
        (Codec::Bc7, true) => Format::Bc7RgbaUnormSrgb,
        (Codec::Astc { .. }, true) => Format::Rgba8UnormSrgb,
        (Codec::Astc { .. }, false) => Format::Rgba8Unorm,
    }
}

/// Decode a compressed surface to `Rgba8Unorm`, keeping its sRGB encoding.
fn decode_blocks(
    ctx: &ExecCtx,
    image: &Texture,
    base: u64,
    codec: Codec,
    out: &mut Vec<u8>,
) -> Result<()> {
    let (block_w, block_h) = codec.block_size();
    let bytes = codec.bytes_per_block();
    let blocks_wide = image.width.div_ceil(block_w);
    let width_bytes = match image.layout {
        Layout::Pitch { pitch } => pitch,
        Layout::BlockLinear { .. } => blocks_wide * bytes,
    };
    // A row of blocks at a time, into buffers reused across blocks.
    let mut strip: Vec<[f32; 4]> = vec![[0.0; 4]; (blocks_wide * block_w * block_h) as usize];
    let mut block = [[0.0f32; 4]; crate::gpu::bcn::MAX_TEXELS];
    for block_y in 0..image.height.div_ceil(block_h) {
        for block_x in 0..blocks_wide {
            let at = base + u64::from(image.layout.offset(block_x * bytes, block_y, width_bytes));
            let raw = ctx.read_pixel(at, bytes)?.to_le_bytes();
            crate::gpu::bcn::decode_into(codec, &raw[..bytes as usize], &mut block)?;
            for y in 0..block_h {
                for x in 0..block_w {
                    let into = (y * blocks_wide * block_w + block_x * block_w + x) as usize;
                    strip[into] = block[(y * block_w + x) as usize];
                }
            }
        }
        for y in 0..block_h {
            if block_y * block_h + y >= image.height {
                break;
            }
            for x in 0..image.width {
                let texel = strip[(y * blocks_wide * block_w + x) as usize];
                for channel in texel {
                    out.push((channel.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
            }
        }
    }
    Ok(())
}

/// Walk a swizzled surface once and write it out as rows of `unit`s
/// (a texel, or a whole block when compressed).
fn deswizzle(
    ctx: &ExecCtx,
    base: u64,
    layout: Layout,
    row_bytes: u32,
    rows: u32,
    unit: u32,
    out: &mut Vec<u8>,
) -> Result<()> {
    if unit == 0 {
        return Err(Error::Gpu(
            "upload: a texture with no bytes per texel".into(),
        ));
    }
    let per_row = row_bytes / unit;
    // One translation when one mapping holds the surface. See [`ExecCtx::span`].
    let swizzled = u64::from(layout.layer_stride(row_bytes, rows));
    if let Some(cpu) = ctx.span(base, swizzled) {
        let mut raw = vec![0u8; swizzled as usize];
        ctx.read_span(cpu, &mut raw)?;
        // The run-at-a-time walk of `Target::write`, reversed.
        let width = per_row * unit;
        for y in 0..rows {
            let mut x = 0;
            while x < width {
                let (at, run) = layout.run_at(x, y, row_bytes);
                let take = run.min(width - x) as usize;
                if take == 0 {
                    break;
                }
                let at = at as usize;
                out.extend_from_slice(&raw[at..at + take]);
                x += take as u32;
            }
        }
        return Ok(());
    }
    for y in 0..rows {
        for x in 0..per_row {
            let at = base + u64::from(layout.offset(x * unit, y, row_bytes));
            let value = ctx.read_pixel(at, unit)?;
            out.extend_from_slice(&value.to_le_bytes()[..unit as usize]);
        }
    }
    Ok(())
}

/// Read a draw's indices, widening the 8-bit form WebGPU does not have.
fn read_indices(
    ctx: &ExecCtx,
    base: u64,
    first: u32,
    count: u32,
    format: u32,
) -> Result<IndexUpload> {
    let (width, out_format) = match format {
        0 => (1u64, IndexFormat::Uint16),
        1 => (2, IndexFormat::Uint16),
        2 => (4, IndexFormat::Uint32),
        other => return Err(Error::Gpu(format!("upload: unknown index format {other}"))),
    };
    let out_width = if out_format == IndexFormat::Uint16 {
        2
    } else {
        4
    };
    if u64::from(count) * out_width > MAX_UPLOAD {
        return Err(Error::Gpu(format!(
            "upload: {count} indices is past the {MAX_UPLOAD}-byte cap"
        )));
    }

    let mut bytes = Vec::with_capacity(count as usize * out_width as usize);
    let mut lowest = u32::MAX;
    let mut highest = 0u32;
    for ordinal in 0..count {
        let at = base + u64::from(first + ordinal) * width;
        let value = match width {
            1 => u32::from(ctx.vmm_read_u8(at)?),
            2 => u32::from(ctx.vmm_read_u8(at)?) | (u32::from(ctx.vmm_read_u8(at + 1)?) << 8),
            _ => ctx.read_u32(at)?,
        };
        lowest = lowest.min(value);
        highest = highest.max(value);
        match out_format {
            IndexFormat::Uint16 => bytes.extend_from_slice(&(value as u16).to_le_bytes()),
            IndexFormat::Uint32 => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }
    if count == 0 {
        lowest = 0;
    }
    Ok(IndexUpload {
        format: out_format,
        bytes,
        lowest,
        highest,
    })
}

/// `len` bytes from a GPU virtual address, a word at a time where possible.
fn read_range(ctx: &ExecCtx, gpu_va: u64, len: u64, what: &str) -> Result<Vec<u8>> {
    if len > MAX_UPLOAD {
        return Err(Error::Gpu(format!(
            "upload: {what} at {gpu_va:#x} is {len} bytes, past the {MAX_UPLOAD}-byte cap"
        )));
    }
    let mut out = Vec::with_capacity(len as usize);
    let mut at = gpu_va;
    let end = gpu_va + len;
    while at < end {
        if at.is_multiple_of(4) && end - at >= 4 {
            out.extend_from_slice(&ctx.read_u32(at)?.to_le_bytes());
            at += 4;
        } else {
            out.push(ctx.vmm_read_u8(at)?);
            at += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::vmm::AddressSpace;
    use crate::gpu::{GpuStats, Host1x};
    use crate::mem::Memory;

    struct Harness {
        mem: Memory,
        vmm: AddressSpace,
        host1x: Host1x,
        stats: GpuStats,
        base: u64,
    }

    impl Harness {
        fn new(size: u32) -> Harness {
            let mut mem = Memory::new();
            mem.map_zero(0x3000_0000, size as usize).unwrap();
            let mut vmm = AddressSpace::new();
            let base = vmm
                .map(0x3000_0000, size as u64, 1, 0, 0x1000, 0, 0)
                .unwrap();
            Harness {
                mem,
                vmm,
                host1x: Host1x::new(),
                stats: GpuStats::default(),
                base,
            }
        }

        fn ctx(&mut self) -> ExecCtx<'_> {
            ExecCtx {
                mem: &mut self.mem,
                vmm: &self.vmm,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            }
        }

        fn write(&mut self, offset: u64, bytes: &[u8]) {
            for (i, &byte) in bytes.iter().enumerate() {
                self.mem
                    .write_u8(0x3000_0000 + offset as u32 + i as u32, byte)
                    .unwrap();
            }
        }
    }

    #[test]
    fn an_eight_bit_index_is_widened_because_webgpu_has_no_such_format() {
        let mut h = Harness::new(0x1000);
        h.write(0, &[3, 1, 2]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 0, 3, 0).unwrap();
        assert_eq!(indices.format, IndexFormat::Uint16);
        assert_eq!(indices.bytes, vec![3, 0, 1, 0, 2, 0]);
    }

    #[test]
    fn the_last_vertex_reaches_only_as_far_as_its_attributes() {
        use crate::gpu::pipeline::{VertexAttribute, VertexFormat};
        let attribute = |offset, format| VertexAttribute {
            format,
            offset,
            location: 0,
            is_bgra: false,
        };
        // Two 32-byte elements using 16 bytes each reach 48 bytes.
        let buffer = VertexBuffer {
            index: 0,
            stride: 32,
            step: StepMode::Vertex,
            attributes: vec![
                attribute(0, VertexFormat::Float32x2),
                attribute(8, VertexFormat::Unorm16x4),
            ],
        };
        assert_eq!(vertex_span(2, &buffer), 32 + 16);
        assert_eq!(vertex_span(1, &buffer), 16);
        // With nothing read, the element is taken whole.
        let bare = VertexBuffer {
            attributes: Vec::new(),
            ..buffer
        };
        assert_eq!(vertex_span(2, &bare), 64);
    }

    #[test]
    fn the_index_range_is_what_bounds_a_vertex_upload() {
        // The index range bounds an indexed draw's vertex reads.
        let mut h = Harness::new(0x1000);
        h.write(0, &[9, 0, 5, 0, 7, 0]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 0, 3, 1).unwrap();
        assert_eq!((indices.lowest, indices.highest), (5, 9));
    }

    #[test]
    fn a_thirty_two_bit_index_is_passed_through() {
        let mut h = Harness::new(0x1000);
        h.write(0, &1u32.to_le_bytes());
        h.write(4, &0x1234_5678u32.to_le_bytes());
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 0, 2, 2).unwrap();
        assert_eq!(indices.format, IndexFormat::Uint32);
        assert_eq!(indices.lowest, 1);
        assert_eq!(indices.highest, 0x1234_5678);
    }

    #[test]
    fn the_first_index_is_an_offset_into_the_index_buffer() {
        // For an indexed draw `first` counts indices.
        let mut h = Harness::new(0x1000);
        h.write(0, &[0, 0, 0, 42, 0, 0]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 3, 1, 0).unwrap();
        assert_eq!((indices.lowest, indices.highest), (42, 42));
    }

    #[test]
    fn an_index_count_past_the_ceiling_is_reported_rather_than_allocated() {
        let mut h = Harness::new(0x1000);
        let base = h.base;
        assert!(read_indices(&h.ctx(), base, 0, u32::MAX, 2).is_err());
    }

    #[test]
    fn a_range_past_the_ceiling_is_reported_before_it_is_read() {
        let mut h = Harness::new(0x1000);
        let base = h.base;
        assert!(read_range(&h.ctx(), base, MAX_UPLOAD + 1, "test").is_err());
    }

    #[test]
    fn a_range_reads_the_same_bytes_however_it_is_aligned() {
        // The word and byte paths must agree.
        let mut h = Harness::new(0x1000);
        let bytes: Vec<u8> = (0..32u8).collect();
        h.write(0, &bytes);
        let base = h.base;
        let ctx = h.ctx();
        assert_eq!(read_range(&ctx, base, 32, "test").unwrap(), bytes);
        assert_eq!(
            read_range(&ctx, base + 1, 30, "test").unwrap(),
            bytes[1..31]
        );
        assert_eq!(read_range(&ctx, base + 3, 5, "test").unwrap(), bytes[3..8]);
        assert_eq!(read_range(&ctx, base, 0, "test").unwrap(), Vec::<u8>::new());
    }

    fn image(kind: TexelKind, width: u32, height: u32, srgb: bool) -> Texture {
        Texture {
            addr: 0,
            width,
            height,
            layout: Layout::Pitch { pitch: 0 },
            kind,
            srgb,
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            layer_stride: 0,
            layers: 1,
            block_depth_gobs: 1,
        }
    }

    #[test]
    fn a_bc_texture_stays_compressed() {
        let bc1 = image(TexelKind::Block(Codec::Bc1), 64, 64, false);
        let copy = image_copy(&bc1).unwrap();
        assert!(matches!(copy, Copy::Raw { unit: 8 }), "{copy:?}");
        // 16 blocks of 8 bytes across, 16 rows of blocks down.
        assert_eq!(copy.shape(&bc1), (Format::Bc1RgbaUnorm, 128, 16));
    }

    #[test]
    fn a_compressed_texture_that_is_not_whole_blocks_is_decoded() {
        // A partial block is decoded rather than rounded up.
        let stub = image(TexelKind::Block(Codec::Bc4Unorm), 1, 1, false);
        assert!(matches!(image_copy(&stub).unwrap(), Copy::Decode { .. }));
        assert_eq!(
            image_copy(&stub).unwrap().shape(&stub),
            (Format::Rgba8Unorm, 4, 1)
        );
        let whole = image(TexelKind::Block(Codec::Bc4Unorm), 8, 8, false);
        assert!(matches!(image_copy(&whole).unwrap(), Copy::Raw { unit: 8 }));
    }

    #[test]
    fn an_astc_texture_is_decoded_because_no_desktop_browser_can_sample_one() {
        // ASTC is decoded.
        let astc = image(
            TexelKind::Block(Codec::Astc {
                width: 4,
                height: 4,
            }),
            64,
            64,
            false,
        );
        let copy = image_copy(&astc).unwrap();
        assert!(matches!(copy, Copy::Decode { .. }), "{copy:?}");
        assert_eq!(copy.shape(&astc), (Format::Rgba8Unorm, 256, 64));
    }

    #[test]
    fn the_tics_srgb_flag_picks_the_format_not_the_format_code() {
        let srgb = image(TexelKind::Block(Codec::Bc7), 8, 8, true);
        assert_eq!(
            image_copy(&srgb).unwrap().shape(&srgb).0,
            Format::Bc7RgbaUnormSrgb
        );
        let linear = image(TexelKind::Block(Codec::Bc7), 8, 8, false);
        assert_eq!(
            image_copy(&linear).unwrap().shape(&linear).0,
            Format::Bc7RgbaUnorm
        );
        // A decoded image keeps its sRGB encoding.
        let astc = image(
            TexelKind::Block(Codec::Astc {
                width: 4,
                height: 4,
            }),
            8,
            8,
            true,
        );
        assert_eq!(
            image_copy(&astc).unwrap().shape(&astc).0,
            Format::Rgba8UnormSrgb
        );
    }

    #[test]
    fn a_texture_format_nothing_can_sample_is_reported() {
        let unknown = image(
            TexelKind::Plain(ColorFormat::from_raw(0xE8).unwrap()),
            8,
            8,
            false,
        );
        assert!(image_copy(&unknown).is_err(), "B5G6R5 has no WebGPU format");
    }

    #[test]
    fn a_pitch_surface_comes_out_as_the_rows_it_already_was() {
        let mut h = Harness::new(0x1000);
        let bytes: Vec<u8> = (0..48u8).collect();
        h.write(0, &bytes);
        let base = h.base;
        let mut out = Vec::new();
        // Four 4-byte texels per row, three rows, 16 bytes apart.
        deswizzle(
            &h.ctx(),
            base,
            Layout::Pitch { pitch: 16 },
            16,
            3,
            4,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, bytes);
    }

    #[test]
    fn a_deswizzled_surface_reads_the_same_texels_the_rasterizer_samples() {
        // The two walks must agree.
        let mut h = Harness::new(0x4000);
        let bytes: Vec<u8> = (0..=255u8).cycle().take(0x2000).collect();
        h.write(0, &bytes);
        let base = h.base;
        let layout = Layout::BlockLinear {
            block_height_gobs: 2,
        };
        let texture = Texture {
            addr: base,
            width: 16,
            height: 16,
            layout,
            kind: TexelKind::Plain(ColorFormat::from_raw(0xD5).unwrap()),
            srgb: false,
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            layer_stride: 0,
            layers: 1,
            block_depth_gobs: 1,
        };
        let mut out = Vec::new();
        deswizzle(&h.ctx(), base, layout, 16 * 4, 16, 4, &mut out).unwrap();
        let ctx = h.ctx();
        for y in 0..16u32 {
            for x in 0..16u32 {
                let sampled = texture.texel_cached(x, y, 0, &ctx).unwrap();
                let at = ((y * 16 + x) * 4) as usize;
                let copied = [
                    out[at] as f32 / 255.0,
                    out[at + 1] as f32 / 255.0,
                    out[at + 2] as f32 / 255.0,
                    out[at + 3] as f32 / 255.0,
                ];
                for c in 0..4 {
                    assert!(
                        (sampled[c] - copied[c]).abs() < 1.0 / 255.0,
                        "texel ({x}, {y}) channel {c}: sampled {sampled:?}, copied {copied:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_surface_survives_a_round_trip_through_linear_rows() {
        // Read and write must be inverse walks.
        let mut h = Harness::new(0x8000);
        let original: Vec<u8> = (0..=255u8).cycle().take(16 * 16 * 4).collect();
        let target = Target {
            format: Format::Rgba8Unorm,
            addr: h.base,
            width: 16,
            height: 16,
            layout: Layout::BlockLinear {
                block_height_gobs: 2,
            },
            row_bytes: 16 * 4,
            rows: 16,
            unit: 4,
            depth: None,
        };
        target.write(&mut h.ctx(), &original).unwrap();
        assert_eq!(target.read(&h.ctx()).unwrap(), original);
        // The bytes really were swizzled.
        let mut linear = Vec::new();
        let base = h.base;
        deswizzle(
            &h.ctx(),
            base,
            Layout::Pitch { pitch: 64 },
            64,
            16,
            4,
            &mut linear,
        )
        .unwrap();
        assert_ne!(linear, original, "a block-linear surface is not rows");
    }

    fn packed_depth_target(addr: u64) -> Target {
        let format = DepthLayout {
            bytes: 4,
            depth_bits: 24,
            depth_shift: 8,
            stencil_shift: Some(0),
        };
        Target {
            format: Format::Depth24PlusStencil8,
            addr,
            width: 4,
            height: 4,
            layout: Layout::Pitch { pitch: 16 },
            row_bytes: 16,
            rows: 4,
            unit: 4,
            depth: Some(format),
        }
    }

    #[test]
    fn a_packed_depth_surface_round_trips_through_a_device_format() {
        // 24-bit depth survives the f32 round trip exactly.
        let mut h = Harness::new(0x1000);
        let target = packed_depth_target(h.base);
        let mut original = Vec::new();
        for i in 0..16u32 {
            let depth = i * 0x11_1111;
            original.extend_from_slice(&((depth << 8) | (u32::from(i as u8) + 1)).to_le_bytes());
        }
        target.write(&mut h.ctx(), &original).unwrap();
        let device = target.read_depth(&h.ctx()).unwrap();
        assert_eq!(device.len(), 16 * 4);
        target.write_depth(&mut h.ctx(), &device).unwrap();
        assert_eq!(target.read(&h.ctx()).unwrap(), original);
    }

    #[test]
    fn writing_depth_back_leaves_the_stencil_byte_alone() {
        // The stencil byte is preserved.
        let mut h = Harness::new(0x1000);
        let target = packed_depth_target(h.base);
        let original: Vec<u8> = (0..16)
            .flat_map(|i: u32| (0x00AB_CD00 | (i + 1)).to_le_bytes())
            .collect();
        target.write(&mut h.ctx(), &original).unwrap();
        target.write_depth(&mut h.ctx(), &[0u8; 16 * 4]).unwrap();
        let after = target.read(&h.ctx()).unwrap();
        for (i, pixel) in after.as_chunks::<4>().0.iter().enumerate() {
            let value = u32::from_le_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
            assert_eq!(value >> 8, 0, "texel {i} kept a depth it was told to clear");
            assert_eq!(value & 0xFF, i as u32 + 1, "texel {i} lost its stencil");
        }
    }

    #[test]
    fn a_sixteen_bit_depth_surface_stays_sixteen_bit_on_a_device() {
        // Z16 is the one packing a copy may write into the device.
        let format = DepthLayout {
            bytes: 2,
            depth_bits: 16,
            depth_shift: 0,
            stencil_shift: None,
        };
        assert_eq!(DepthKind::of(format), DepthKind::Unorm16);
        assert_eq!(DepthKind::Unorm16.unit(), 2);
        let mut h = Harness::new(0x1000);
        let target = Target {
            format: Format::Depth16Unorm,
            addr: h.base,
            width: 4,
            height: 2,
            layout: Layout::Pitch { pitch: 8 },
            row_bytes: 8,
            rows: 2,
            unit: 2,
            depth: Some(format),
        };
        let original: Vec<u8> = (0..8u16).flat_map(|i| (i * 0x1234).to_le_bytes()).collect();
        target.write(&mut h.ctx(), &original).unwrap();
        let device = target.read_depth(&h.ctx()).unwrap();
        assert_eq!(
            device, original,
            "a Z16 texel is already what the device holds"
        );
        target.write_depth(&mut h.ctx(), &device).unwrap();
        assert_eq!(target.read(&h.ctx()).unwrap(), original);
    }

    #[test]
    fn asking_a_colour_target_for_its_depth_packing_is_reported() {
        let mut h = Harness::new(0x1000);
        let target = Target {
            format: Format::Rgba8Unorm,
            addr: h.base,
            width: 4,
            height: 4,
            layout: Layout::Pitch { pitch: 16 },
            row_bytes: 16,
            rows: 4,
            unit: 4,
            depth: None,
        };
        assert!(target.depth_kind().is_none());
        assert!(target.read_depth(&h.ctx()).is_err());
    }

    #[test]
    fn writing_back_less_than_a_surface_is_reported() {
        let mut h = Harness::new(0x1000);
        let target = Target {
            format: Format::Rgba8Unorm,
            addr: h.base,
            width: 4,
            height: 4,
            layout: Layout::Pitch { pitch: 16 },
            row_bytes: 16,
            rows: 4,
            unit: 4,
            depth: None,
        };
        assert_eq!(target.len(), 64);
        assert!(target.write(&mut h.ctx(), &[0; 32]).is_err());
    }

    #[test]
    fn the_total_is_every_buffer_a_draw_would_move() {
        let uploads = Uploads {
            vertex: vec![VertexUpload {
                array: 0,
                first: 0,
                stride: 8,
                bytes: vec![0; 32],
            }],
            index: Some(IndexUpload {
                format: IndexFormat::Uint16,
                bytes: vec![0; 12],
                lowest: 0,
                highest: 3,
            }),
            constants: vec![ConstantUpload {
                stage: ShaderStage::VertexB,
                bank: 1,
                bytes: vec![0; 256],
            }],
            textures: Vec::new(),
        };
        assert_eq!(uploads.len(), 300);
        assert!(!uploads.is_empty());
    }
}
