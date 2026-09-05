//! The bytes a draw moves between guest memory and a device.
//!
//! [`crate::gpu::shader::wgsl`] says how to shade a draw and
//! [`crate::gpu::pipeline`] says what state it runs under. Neither says what
//! it draws. That is in guest memory, vertices at a GPU virtual address,
//! indices at another, constants in whatever the driver bound, and a device
//! cannot read guest memory. Somebody has to translate the addresses, bound
//! the ranges and hand over bytes, and this is that.
//!
//! The software rasterizer never needed this. It reads a vertex attribute at
//! a time, through the GPU MMU, exactly when a vertex shader asks for it, and
//! a buffer that a draw does not touch costs nothing. A GPU backend has to
//! decide up front what to upload, which turns "read this word" into "how
//! much of this buffer is this draw actually going to look at", a question
//! the register file does not answer directly.
//!
//! # Bounding what a draw touches
//!
//! A vertex array says where it starts and where it ends, and the end is
//! often the end of a heap rather than the end of the mesh. What bounds an
//! upload is the draw: `first` and `count` for a sequential draw, and for an
//! indexed one the lowest and highest index in the index buffer, which has to
//! be read to be known. Doing that here is not wasted work: the indices have
//! to be uploaded anyway.
//!
//! # And what it writes
//!
//! A render target lives in guest memory too. `present` deswizzles
//! block-linear pixels straight out of it, the 2D blitter copies out of it,
//! and a shader can sample it, so a backend that keeps its surfaces on the
//! device owns the question of when to write them back. [`Targets`] says
//! where they are and [`Target::write`] is the walk that puts them back,
//! which is [`Target::read`] run backwards.
//!
//! # It has a ceiling, on purpose
//!
//! A stride and a count that multiply to something absurd are not a reason to
//! allocate it. [`MAX_UPLOAD`] is the point at which this reports rather than
//! tries, because the failure mode of not having one is a machine in swap.

use crate::gpu::bcn::Codec;
use crate::gpu::engine::threed::{DepthLayout, Engine3D, ShaderStage};
use crate::gpu::exec::ExecCtx;
use crate::gpu::pipeline::{Format, Pipeline, StepMode};
use crate::gpu::surface::{ColorFormat, Layout};
use crate::gpu::texture::{self, Sampler, SwizzleSource, TexelKind, Texture};
use crate::{Error, Result};

/// Which constant banks to resolve.
///
/// The distinction is not fussiness. A bank is up to 64 KiB and the Home Menu
/// binds eight of them per draw while its shaders read two, so the difference
/// between these two answers is 190 KiB a draw and 60 KiB a draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Banks<'a> {
    /// Every bank the draw has bound, which is a fact about the draw.
    Bound,
    /// Only these, which is a fact about the shaders: the `const_banks` of
    /// each stage's [`crate::gpu::shader::wgsl::Translation`], paired with
    /// the stage it came from.
    Read(&'a [(ShaderStage, u32)]),
}

/// The most one buffer will be read into memory: 64 MiB.
///
/// Larger than any mesh a draw addresses and far smaller than the heap a
/// vertex array's limit usually points at the end of.
pub const MAX_UPLOAD: u64 = 64 << 20;

/// How many constant banks a bind slot has.
const CONSTBUF_BANKS: u32 = 32;

/// The index width a backend is handed. Maxwell also has an 8-bit form and
/// WebGPU does not, so [`Uploads::of`] widens that to 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// One vertex array's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexUpload {
    /// Which of Maxwell's vertex arrays this is, matching
    /// [`crate::gpu::pipeline::VertexBuffer::index`].
    pub array: u32,
    /// The element these bytes start at. A backend either offsets the buffer
    /// binding by `first * stride` or adds `first` to its base vertex; what
    /// it must not do is assume element zero, since a draw that starts at
    /// vertex 900 uploads from there.
    pub first: u32,
    pub stride: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexUpload {
    pub format: IndexFormat,
    pub bytes: Vec<u8>,
    /// The lowest and highest index the draw uses, which is what bounds the
    /// vertex uploads.
    pub lowest: u32,
    pub highest: u32,
}

impl IndexUpload {
    /// The indices themselves, widened.
    ///
    /// A backend that has to *rewrite* the list, which assembling a fan or a
    /// quad into triangles is: needs the values rather than the bytes, and
    /// the widening is the same one `read_indices` already does for an
    /// eight-bit list.
    pub fn indices(&self) -> Vec<u32> {
        match self.format {
            IndexFormat::Uint16 => self
                .bytes
                .chunks_exact(2)
                .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            IndexFormat::Uint32 => self
                .bytes
                .chunks_exact(4)
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

/// One texture, deswizzled into the linear rows a device copies from.
///
/// A surface in guest memory is *block-linear*: rows are interleaved through
/// 512-byte GOBs so that a 2D neighbourhood is contiguous, which is what
/// makes a texture cache work and what makes the bytes unreadable to anything
/// that expects rows. `WriteTexture` wants rows. So this walks the swizzle
/// once and writes them out, in the same units the surface addresses, texels
/// for a plain format, whole blocks for a compressed one.
///
/// The blocks of a compressed texture are *not* decoded. WebGPU has the BC
/// formats natively, and decoding them here would turn 4 bits a texel into
/// 32 for no reason and then ask the device to sample the result.
/// What decides a texture upload's bytes: where they are and how they are
/// read.
///
/// Two draws with the same key read the same bytes out of the same memory, so
/// one of them can be given the other's, which is the whole point, because a
/// title samples a handful of images over and over and 96.5% of everything
/// [`Uploads::of`] lifts is texture bytes.
///
/// Every field comes off the TIC, so a descriptor the guest rewrites produces
/// a *different key* rather than a stale hit; only a write to the texels
/// themselves needs reporting, which is what [`crate::mem::Memory`]'s watched
/// pages are for. Not the swizzle or the sampler: both are applied when the
/// texture is sampled rather than when it is copied, so they change the draw
/// and not the bytes.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureUpload {
    /// The stage whose constant buffer named this texture. The same
    /// immediate in the other stage is a different texture.
    pub stage: ShaderStage,
    /// The `texs` immediate this was resolved for.
    pub immediate: u16,
    /// The bindless handle that immediate named.
    pub handle: u32,
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    /// Bytes per row of the linear image, a row of *blocks* for a
    /// compressed format, which is not `width * bytes_per_texel`.
    pub row_bytes: u32,
    /// Rows per layer: the height in texels, or in blocks when compressed.
    pub rows: u32,
    /// The layers back to back, each `row_bytes * rows` long.
    ///
    /// Shared rather than owned so that a cache hand-back costs a refcount
    /// instead of copying 1.76 MiB, which is what an average draw reads.
    pub bytes: std::sync::Arc<[u8]>,
    /// What these bytes were read from. See [`TextureKey`].
    pub key: TextureKey,
    /// How far past `key.addr` the read reached, so a caller holding these
    /// bytes knows which pages to watch.
    pub source_len: u64,
    /// How the shader expects the channels rearranged. WebGPU has no
    /// per-texture component swizzle, so a backend applies this itself,
    /// in the sampling hook, where it costs a shuffle rather than a copy.
    pub swizzle: [SwizzleSource; 4],
    pub sampler: Sampler,
}

/// Everything a draw reads, resolved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Uploads {
    pub vertex: Vec<VertexUpload>,
    pub index: Option<IndexUpload>,
    pub constants: Vec<ConstantUpload>,
    pub textures: Vec<TextureUpload>,
}

impl Uploads {
    /// How many bytes this draw would move to a device.
    pub fn len(&self) -> usize {
        self.vertex.iter().map(|v| v.bytes.len()).sum::<usize>()
            + self.index.as_ref().map_or(0, |i| i.bytes.len())
            + self.constants.iter().map(|c| c.bytes.len()).sum::<usize>()
            + self.textures.iter().map(|t| t.bytes.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve what [`Engine3D::last_draw`] reads.
    ///
    /// `pipeline` supplies the vertex layout, so that this and the pipeline
    /// description cannot disagree about which arrays a draw binds or how
    /// they step.
    /// `immediates` are the `texs` immediates the draw's shaders sample
    /// with, each paired with the stage it came from, a
    /// [`crate::gpu::shader::wgsl::Translation`]'s `textures`. They are not
    /// in the register file: only the shader knows which texture units it
    /// reaches, and the two stages index *different* constant buffers with
    /// the same immediate.
    pub fn of(
        engine: &Engine3D,
        pipeline: &Pipeline,
        ctx: &ExecCtx,
        banks: Banks<'_>,
        immediates: &[(ShaderStage, u16)],
    ) -> Result<Uploads> {
        Uploads::of_cached(engine, pipeline, ctx, banks, immediates, &mut |_| None)
    }

    /// [`Uploads::of`], letting the caller answer for a texture it has
    /// already read.
    ///
    /// `cached` is asked once per distinct texture a draw samples, with the
    /// key that decides its bytes; answering `Some` skips the deswizzle
    /// entirely. The caller is the one that can know the answer is still
    /// good, because it is the one that can watch the pages the bytes came
    /// from. See `TextureUpload::source_len` and `Memory::mark_gpu_page`.
    pub fn of_cached(
        engine: &Engine3D,
        pipeline: &Pipeline,
        ctx: &ExecCtx,
        banks: Banks<'_>,
        immediates: &[(ShaderStage, u16)],
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
            // What the draw reaches: an instanced array advances once per
            // instance, and the engine issues one instance per draw, so the
            // element is the instance id and there is exactly one of it.
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
            let length = u64::from(count) * u64::from(buffer.stride);
            let start = array.start + u64::from(first) * u64::from(buffer.stride);
            // The array's own limit is the real end of the mapping, and it is
            // the address of the *last valid byte* rather than one past it,
            // a 32-byte array at `0x204730000` has a limit of `0x20473001f`.
            // A draw that runs past it is reading something else's memory,
            // and saying so is better than uploading it.
            if array.limit != 0 && start + length > array.limit + 1 {
                return Err(Error::Gpu(format!(
                    "upload: vertex array {} reads {start:#x}..{:#x}, past its limit {:#x}",
                    buffer.index,
                    start + length,
                    array.limit
                )));
            }
            vertex.push(VertexUpload {
                array: buffer.index,
                first,
                stride: buffer.stride,
                bytes: read_range(ctx, start, length, "vertex array")?,
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
        for &(stage, immediate) in immediates {
            if textures
                .iter()
                .any(|t: &TextureUpload| t.stage == stage && t.immediate == immediate)
            {
                continue;
            }
            textures.push(read_texture(engine, ctx, stage, immediate, cached)?);
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
///
/// The same shape as a [`TextureUpload`], because it is the same question
/// asked of a different register: where the bytes are, what format they are
/// in, and how many rows of what length come out once the swizzle is undone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub format: Format,
    pub addr: u64,
    /// Extent in *texels*, which on a multisampled surface is not its extent
    /// in pixels: one pixel is a grid of texels there.
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    /// Bytes per row of the linear image.
    pub row_bytes: u32,
    pub rows: u32,
    /// Bytes per texel, which is what the layout addresses.
    pub unit: u32,
    /// How a depth surface packs its depth and its stencil, for the target
    /// that is one. A device holds depth in a format of its own, no shading
    /// API exposes Maxwell's packings, so a backend needs this to convert,
    /// and to put the stencil byte back untouched.
    pub depth: Option<DepthLayout>,
}

impl Target {
    /// How many bytes one copy of this surface is.
    pub fn len(&self) -> u64 {
        u64::from(self.row_bytes) * u64::from(self.rows)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read the surface out as linear rows: what a backend needs before a
    /// draw that blends, or tests depth, against what is already there.
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

    /// Put linear rows back, swizzled: the walk of [`Target::read`] run
    /// backwards, and the thing a backend does before the guest looks at
    /// what it drew.
    pub fn write(&self, ctx: &mut ExecCtx, rows: &[u8]) -> Result<()> {
        self.write_strided(ctx, rows, self.row_bytes)
    }

    /// [`Target::write`], from rows that are `stride` bytes apart rather than
    /// packed.
    ///
    /// A device readback pads its rows out to an alignment, and repacking
    /// them first is a copy of the whole surface, 3.7 MB a frame at 720p,
    /// for a walk that is about to read every byte anyway and can as easily
    /// read them where they are.
    pub fn write_strided(&self, ctx: &mut ExecCtx, rows: &[u8], stride: u32) -> Result<()> {
        let want = (stride * self.rows) as usize;
        if rows.len() < want {
            return Err(Error::Gpu(format!(
                "upload: writing back {} bytes of a {want}-byte target",
                rows.len()
            )));
        }
        let per_row = self.row_bytes / self.unit.max(1);
        // The same one-translation walk [`deswizzle`] makes, run backwards.
        // The surface is read first and patched rather than built from
        // nothing, because a block-linear surface is padded out to whole
        // blocks and those bytes are not this surface's to zero.
        if let Some((cpu, mut raw)) = self.mapped(ctx)? {
            // A run at a time, not a texel: `run_at` answers where a byte
            // lands *and* how far from there is contiguous, which inside a GOB
            // is 16 bytes (four pixels at 32 bits) and a whole row for a
            // pitch surface. This is the walk `Gpu::present` already makes,
            // and it is the one that runs every frame: a readback lands here.
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

    /// The surface's swizzled bytes, and where they live, when one mapping
    /// holds all of them.
    fn mapped(&self, ctx: &ExecCtx) -> Result<Option<(u32, Vec<u8>)>> {
        let swizzled = u64::from(self.layout.layer_stride(self.row_bytes, self.rows));
        let Some(cpu) = ctx.span(self.addr, swizzled) else {
            return Ok(None);
        };
        let mut raw = vec![0u8; swizzled as usize];
        ctx.read_span(cpu, &mut raw)?;
        Ok(Some((cpu, raw)))
    }

    /// How a device would hold this surface, for the target that is a depth
    /// one.
    pub fn depth_kind(&self) -> Option<DepthKind> {
        self.depth.map(DepthKind::of)
    }

    /// Read a depth surface out as the linear rows a device texture wants:
    /// one [`DepthKind::unit`] per texel, the stencil byte dropped.
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

    /// Put a device's depth rows back into guest memory, in the guest's own
    /// packing.
    ///
    /// A packed pixel's stencil byte is read back and kept: a depth pass
    /// writes depth, and the byte beside it belongs to whoever wrote it last
    ///, which is what `raster`'s `Fragments::write` does per fragment.
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
        // The whole surface in one translation where one mapping holds it,
        // which also makes the stencil byte free: it is already in `raw`.
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
                // Reading the pixel back is only worth its cost where
                // something else lives in it.
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

    /// The colour surface bound at `slot`, or `None` for a slot with no
    /// surface in it.
    ///
    /// Split out of [`Targets::of`] because a clear names its own target and
    /// a draw does not: `ClearBuffers` carries the index, and reading it
    /// through a second walk of the register file is a second place for the
    /// two to disagree about what a surface is.
    pub fn color(engine: &Engine3D, slot: u32) -> Result<Option<Target>> {
        let Some(rt) = engine.render_target(slot)? else {
            return Ok(None);
        };
        let unit = rt.format.bytes_per_pixel;
        // A disabled target reads back as format 0, which is no pixel at all
        // rather than a pixel of no bytes.
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

    /// The depth surface the engine has bound, or `None`.
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

/// How a device holds a guest depth surface.
///
/// A shading API has no Maxwell packing in it, so a depth surface is
/// converted rather than copied, and there are only two formats worth
/// converting *to*: WebGPU lets a copy read `depth32float` but never write
/// it, lets a copy do both to `depth16unorm`, and lets a copy touch
/// `depth24plus` in neither direction. So `Z16` stays what it is and
/// everything else becomes a float, including the 24-bit packings, which
/// survive the round trip bit-for-bit because 24 bits is exactly what an
/// `f32` mantissa holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthKind {
    Unorm16,
    Float32,
}

impl DepthKind {
    /// Bytes one texel takes on the device.
    pub fn unit(self) -> u32 {
        match self {
            DepthKind::Unorm16 => 2,
            DepthKind::Float32 => 4,
        }
    }

    /// The kind a guest packing converts to.
    pub fn of(layout: DepthLayout) -> DepthKind {
        match (layout.bytes, layout.depth_bits, layout.stencil_shift) {
            (2, 16, None) => DepthKind::Unorm16,
            _ => DepthKind::Float32,
        }
    }

    /// `depth`, in `0.0..=1.0`, as the bytes a device texel holds.
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

    /// Inverse of [`DepthKind::encode`], from `unit` bytes of a device texel.
    fn decode(self, bytes: &[u8]) -> f32 {
        match self {
            DepthKind::Unorm16 => f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0,
            DepthKind::Float32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

/// Where a draw's surfaces are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Targets {
    /// Colour target 0, or `None` for a depth-only pass, which is a real
    /// thing a title does: Just Dance 2017 renders every pass that way.
    pub color: Option<Target>,
    pub depth: Option<Target>,
}

impl Targets {
    /// Resolve the surfaces the engine has bound.
    pub fn of(engine: &Engine3D) -> Result<Targets> {
        Ok(Targets {
            color: Target::color(engine, engine.render_target_slot(0))?,
            depth: Target::depth_surface(engine)?,
        })
    }

    /// How many bytes both surfaces are, which is what a round trip through a
    /// device costs per frame.
    pub fn len(&self) -> u64 {
        self.color.map_or(0, |t| t.len()) + self.depth.map_or(0, |t| t.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Resolve a `texs` immediate to a texture and copy it out.
///
/// The immediate is not a handle. It indexes, in dwords, the constant bank
/// `TexCbIndex` names, a register the driver programs, to 15 under nouveau
/// and 0 under deko3d, and *that* holds the bindless handle, which in turn
/// indexes the TIC and TSC pools. Three levels, none of them optional.
/// One GOB: the block-linear unit a surface's rows are padded up to.
const GOB_BYTES: u64 = 512;

fn read_texture(
    engine: &Engine3D,
    ctx: &ExecCtx,
    stage: ShaderStage,
    immediate: u16,
    cached: &mut dyn FnMut(&TextureKey) -> Option<std::sync::Arc<[u8]>>,
) -> Result<TextureUpload> {
    let bank = u32::from(engine.tex_cb_index());
    let (addr, size) = engine.bound_constbuf(stage, bank).ok_or_else(|| {
        Error::Gpu(format!(
            "upload: {stage:?}'s texture bank {bank} is not bound"
        ))
    })?;
    let offset = u32::from(texture::handle_offset(immediate));
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
    // How far the read reaches, so a caller holding the bytes knows which
    // pages a guest write to them would land on. Block-linear pads a surface
    // up, so the dense size is a floor rather than the answer: the offset of
    // the last row's last byte is what the walk actually reaches, plus a GOB
    // of slack, plus the stride to the last layer. Whichever is larger, since
    // a volume interleaves its slices and addresses through neither.
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
            immediate,
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
        // A volume's slices interleave inside a block, so there is no base a
        // slice starts at, `Texture::texel` addresses each one instead, and
        // that is also the only path that reads the depth of a block.
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
        immediate,
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

/// How a surface's bytes get to a device.
#[derive(Debug, Clone, Copy)]
enum Copy {
    /// Deswizzled and handed over in the surface's own units, which is what
    /// happens whenever WebGPU has a format for them, including every BC
    /// codec, which stays compressed.
    Raw { unit: u32 },
    /// Decoded to `Rgba8Unorm` first, because WebGPU has no format for the
    /// codec. Turning 1 byte a texel into 4 is a real cost, and the
    /// alternative is not sampling the texture at all.
    Decode { codec: Codec },
}

impl Copy {
    /// The WebGPU format, and the linear image's row length and count.
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
                // `image_copy` refuses these before a `Copy` exists.
                TexelKind::Depth(_) => unreachable!("a depth texel has no WebGPU format"),
            },
            // A decoded image is texels again, whatever it was stored as.
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

/// Whether a surface can be handed over as it is.
fn image_copy(image: &Texture) -> Result<Copy> {
    Ok(match image.kind {
        TexelKind::Plain(plain) => {
            // A format `pipeline` cannot name is one nothing can sample.
            crate::gpu::pipeline::color_format(plain)
                .map_err(|e| Error::Gpu(format!("upload: texture format: {e}")))?;
            Copy::Raw {
                unit: plain.bytes_per_pixel,
            }
        }
        // A depth surface handed over as a texture would have to be a
        // `depth32float`, and WebGPU fills one only from another texture of
        // the same format: the same rule that keeps a shadow map off the
        // device (see `wgsl`'s `Unsupported::DepthCompare`). So a draw that
        // samples one is the rasterizer's.
        TexelKind::Depth(depth) => {
            return Err(Error::Gpu(format!(
                "upload: {depth:?} is a depth surface, which cannot be uploaded as a texture"
            )))
        }
        // WebGPU has ASTC only behind the `texture-compression-astc` feature,
        // which a desktop browser does not offer, and the Home Menu's real
        // textures are ASTC 4x4, so refusing them would mean refusing the
        // draws that matter.
        TexelKind::Block(codec @ Codec::Astc { .. }) => Copy::Decode { codec },
        TexelKind::Block(codec) => {
            let (block_w, block_h) = codec.block_size();
            // WebGPU will not make a compressed texture whose extent is not a
            // whole number of blocks, and Maxwell will: the Home Menu binds
            // 1x1 BC4 and BC5 images as the default texture for its
            // untextured quads. Rounding the extent up would change what a
            // normalized coordinate samples: one texel becomes sixteen,
            // so the partial ones are decoded instead, which
            // `decode_blocks` already clips to the real extent.
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

/// The `srgb` flag is the TIC's, not the format code's: the same raw format
/// is sampled either way depending on it, which is why this takes both.
///
/// Infallible: [`image_copy`] has already refused a format with no name.
fn plain_format(plain: ColorFormat, srgb: bool) -> Format {
    let format = crate::gpu::pipeline::color_format(plain).unwrap_or(Format::Rgba8Unorm);
    match (format, srgb) {
        (Format::Rgba8Unorm, true) => Format::Rgba8UnormSrgb,
        (Format::Bgra8Unorm, true) => Format::Bgra8UnormSrgb,
        (format, _) => format,
    }
}

/// Infallible for the same reason as [`plain_format`]: ASTC, the one codec
/// with no WebGPU format, is decoded rather than named.
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

/// Decode a compressed surface to `Rgba8Unorm`, block by block.
///
/// The values a codec yields are what the texture *stores*, so an sRGB image
/// stays sRGB-encoded here and the format says so, the device applies the
/// transfer function, exactly as `Texture::texel` applies it for the
/// rasterizer.
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
    // One row of blocks at a time: a decoded block covers `block_h` output
    // rows, so the whole strip is decoded before any of it is written.
    //
    // Both buffers are made once and written over. `MAX_TEXELS` is a 12x12
    // footprint, so a fresh `block` per block zeroed 2304 bytes to fill the
    // 256 a 4x4 needs, and the strip is written in full every row before
    // anything reads it: the clearing was 24% of an ASTC title's upload.
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
        // The last block of a row or column hangs outside the extent.
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

/// Walk a swizzled surface once and write it out as rows.
///
/// `unit` is what the layout addresses: one texel for a plain format, one
/// whole block for a compressed one. Reading a compressed surface in texels
/// instead is the mistake that shreds an image into diagonal ribbons, because
/// the row stride comes out a block too wide.
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
    // One translation for the whole surface where one mapping holds it, which
    // a render target's does. The walk is then arithmetic over a slice
    // instead of 3.7 million address translations. See [`ExecCtx::span`].
    let swizzled = u64::from(layout.layer_stride(row_bytes, rows));
    if let Some(cpu) = ctx.span(base, swizzled) {
        let mut raw = vec![0u8; swizzled as usize];
        ctx.read_span(cpu, &mut raw)?;
        // The same run-at-a-time walk `Target::write` makes, in the other
        // direction: `out` is built in order, so a run appends.
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
        // Widened, not passed through: a backend has nowhere to put an 8-bit
        // index, and the alternative is every backend widening it itself.
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

/// `len` bytes from a GPU virtual address.
///
/// A word at a time where the range allows it: the address translation and
/// the page lookup are per access, not per byte, and a mesh read a byte at a
/// time pays for both eight times over.
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

    /// Guest memory with one page mapped, and the GPU address it is at.
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
    fn the_index_range_is_what_bounds_a_vertex_upload() {
        // Nothing else says how much of a vertex array an indexed draw
        // reaches: the array's own limit is usually the end of a heap.
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
        // For an indexed draw `first` counts indices, not vertices, the
        // vertex it lands on is whatever the index there says.
        let mut h = Harness::new(0x1000);
        h.write(0, &[0, 0, 0, 42, 0, 0]);
        let base = h.base;
        let indices = read_indices(&h.ctx(), base, 3, 1, 0).unwrap();
        assert_eq!((indices.lowest, indices.highest), (42, 42));
    }

    #[test]
    fn an_index_count_past_the_ceiling_is_reported_rather_than_allocated() {
        // The failure mode of having no ceiling is a machine in swap.
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
        // Words where the range allows and bytes at the edges: a mesh read a
        // byte at a time pays for an address translation eight times over,
        // and the two paths have to agree.
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
        // WebGPU has the BC formats natively. Decoding them here would turn
        // 4 bits a texel into 32 and then ask the device to sample that.
        let bc1 = image(TexelKind::Block(Codec::Bc1), 64, 64, false);
        let copy = image_copy(&bc1).unwrap();
        assert!(matches!(copy, Copy::Raw { unit: 8 }), "{copy:?}");
        // 16 blocks of 8 bytes across, 16 rows of blocks down.
        assert_eq!(copy.shape(&bc1), (Format::Bc1RgbaUnorm, 128, 16));
    }

    #[test]
    fn a_compressed_texture_that_is_not_whole_blocks_is_decoded() {
        // WebGPU will not make one, and Maxwell will: the Home Menu binds
        // 1x1 BC4 and BC5 images as the default texture for its untextured
        // quads. Rounding the extent up to a block would turn one texel into
        // sixteen and change what a normalized coordinate samples.
        let stub = image(TexelKind::Block(Codec::Bc4Unorm), 1, 1, false);
        assert!(matches!(image_copy(&stub).unwrap(), Copy::Decode { .. }));
        assert_eq!(
            image_copy(&stub).unwrap().shape(&stub),
            (Format::Rgba8Unorm, 4, 1)
        );
        // A whole number of blocks still goes over compressed.
        let whole = image(TexelKind::Block(Codec::Bc4Unorm), 8, 8, false);
        assert!(matches!(image_copy(&whole).unwrap(), Copy::Raw { unit: 8 }));
    }

    #[test]
    fn an_astc_texture_is_decoded_because_no_desktop_browser_can_sample_one() {
        // WebGPU has ASTC behind `texture-compression-astc`, which desktop
        // browsers do not offer, and the Home Menu's real textures are ASTC
        // 4x4, so refusing them would be refusing the draws that matter.
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
        // Texels again, whatever it was stored as.
        assert_eq!(copy.shape(&astc), (Format::Rgba8Unorm, 256, 64));
    }

    #[test]
    fn the_tics_srgb_flag_picks_the_format_not_the_format_code() {
        // The same raw code is sampled either way depending on the flag, so
        // a format named without it would be a whole transfer function out.
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
        // A decoded image keeps the encoding it was stored in; the device
        // applies the transfer function.
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
        // The simplest layout there is, and the one that says whether the
        // walk writes rows in the right order at all.
        let mut h = Harness::new(0x1000);
        let bytes: Vec<u8> = (0..48u8).collect();
        h.write(0, &bytes);
        let base = h.base;
        let mut out = Vec::new();
        // Four texels of four bytes per row, three rows, in a surface whose
        // rows are 16 bytes apart.
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
        // The two walks have to agree, or a GPU backend draws a different
        // image from the one it is compared against.
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
        // Read and write are the same walk in opposite directions, and a
        // backend that keeps its surfaces on the device does both every time
        // the guest is about to look at what it drew. If they disagree, the
        // frame comes back scrambled in a way that looks like a rendering
        // bug.
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
        // And the bytes really did get swizzled on the way in, rather than
        // both walks agreeing to write rows.
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

    /// A `Z24S8` target, four by four, pitch-linear.
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
        // 24 bits of depth is exactly what an f32 mantissa holds, so the
        // conversion a device forces is lossless, which is the whole reason
        // `depth32float` is what a 24-bit packing is held in. A round trip
        // that lost a bit would fail every `Equal` depth test in the frame
        // after it.
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
        // A depth pass writes depth. The byte beside it belongs to whoever
        // wrote it last, and flattening it to zero is how a stencil buffer
        // gets cleared by a pass that never mentioned it.
        let mut h = Harness::new(0x1000);
        let target = packed_depth_target(h.base);
        let original: Vec<u8> = (0..16)
            .flat_map(|i: u32| (0x00AB_CD00 | (i + 1)).to_le_bytes())
            .collect();
        target.write(&mut h.ctx(), &original).unwrap();
        target.write_depth(&mut h.ctx(), &[0u8; 16 * 4]).unwrap();
        let after = target.read(&h.ctx()).unwrap();
        for (i, pixel) in after.chunks_exact(4).enumerate() {
            let value = u32::from_le_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
            assert_eq!(value >> 8, 0, "texel {i} kept a depth it was told to clear");
            assert_eq!(value & 0xFF, i as u32 + 1, "texel {i} lost its stencil");
        }
    }

    #[test]
    fn a_sixteen_bit_depth_surface_stays_sixteen_bit_on_a_device() {
        // `depth16unorm` is the one depth format a copy may write *into*, so
        // Z16 is the one packing that reaches a device without a pass to put
        // it there.
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
    fn a_drawing_with_nothing_in_it_moves_no_bytes() {
        let uploads = Uploads::default();
        assert!(uploads.is_empty());
        assert_eq!(uploads.len(), 0);
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
    }
}
