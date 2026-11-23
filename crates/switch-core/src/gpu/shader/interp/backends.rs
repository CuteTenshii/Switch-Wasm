//! Constant, global memory and texture sources an invocation reads.

use super::*;

/// Resolves a `cN[offset]` operand to its raw 32 bits. `bank` is a bind slot,
/// not an address; see [`MemoryConstants`].
pub trait ConstantSource {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32>;
}

impl ConstantSource for HashMap<(u8, u16), f32> {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32> {
        Ok(self.get(&(bank, offset)).copied().unwrap_or(0.0).to_bits())
    }
}

/// Reads `cN[offset]` from GPU memory; `bindings` resolves a bank to `(address, size)`.
pub struct MemoryConstants<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub bindings: &'a dyn Fn(u8) -> Option<(u64, u32)>,
    /// See [`ConstCache`].
    pub cache: &'a std::cell::RefCell<ConstCache>,
}

/// Direct-mapped, so a power of two.
const CONST_CACHE_SLOTS: usize = 512;

/// Constants already read this draw; a buffer can't change mid-draw.
pub struct ConstCache {
    /// `(key, value)`, key packing bank and offset; `u32::MAX` is empty.
    slots: Box<[(u32, u32); CONST_CACHE_SLOTS]>,
}

impl Default for ConstCache {
    fn default() -> ConstCache {
        ConstCache {
            slots: Box::new([(u32::MAX, 0); CONST_CACHE_SLOTS]),
        }
    }
}

impl ConstCache {
    #[inline]
    pub(crate) fn key(bank: u8, offset: u16) -> u32 {
        (bank as u32) << 16 | offset as u32
    }

    #[inline]
    pub(crate) fn get(&self, key: u32) -> Option<u32> {
        let slot = self.slots[key as usize % CONST_CACHE_SLOTS];
        (slot.0 == key).then_some(slot.1)
    }

    #[inline]
    pub(crate) fn insert(&mut self, key: u32, value: u32) {
        self.slots[key as usize % CONST_CACHE_SLOTS] = (key, value);
    }
}

impl ConstantSource for MemoryConstants<'_, '_> {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32> {
        let key = ConstCache::key(bank, offset);
        if let Some(value) = self.cache.borrow().get(key) {
            return Ok(value);
        }
        let (addr, size) = (self.bindings)(bank)
            .ok_or_else(|| Error::Gpu(format!("shader: read from unbound constant bank {bank}")))?;
        if offset as u32 + 4 > size {
            return Err(fault(format!(
                "shader: constant read c{bank}[{offset:#x}] is past the bound buffer's size {size:#x}"
            )));
        }
        let value = self.ctx.read_u32(addr + offset as u64)?;
        self.cache.borrow_mut().insert(key, value);
        Ok(value)
    }
}

/// `ldg`'s global memory, by 64-bit GPU virtual address. Stores (and atomics)
/// queue in `stores` and land after each shading step; see [`MemoryGlobal::land`].
pub struct MemoryGlobal<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub stores: &'a std::cell::RefCell<Vec<(u64, u32)>>,
}

impl MemoryGlobal<'_, '_> {
    /// Write out queued stores oldest first, so a later one to the same word wins.
    pub fn land(ctx: &mut ExecCtx, stores: &std::cell::RefCell<Vec<(u64, u32)>>) -> Result<()> {
        for (addr, value) in stores.borrow_mut().drain(..) {
            ctx.write_u32(addr, value)?;
        }
        Ok(())
    }
}

impl GlobalMemory for MemoryGlobal<'_, '_> {
    fn read_u32(&self, addr: u64) -> ShaderResult<u32> {
        let held = self
            .stores
            .borrow()
            .iter()
            .rev()
            .find(|&&(at, _)| at == addr)
            .map(|&(_, value)| value);
        match held {
            Some(value) => Ok(value),
            None => Ok(self.ctx.read_u32(addr)?),
        }
    }

    fn write_u32(&self, addr: u64, value: u32) -> ShaderResult<()> {
        self.stores.borrow_mut().push((addr, value));
        Ok(())
    }
}

/// Resolves a `texs` sample. `handle` is the packed `imageId | samplerId << 20`,
/// already read via [`ConstantSource`] by [`Invocation::execute`].
pub trait TextureSource {
    /// Sample `handle` at `(u, v)` of array layer `layer` (0 if not an array).
    fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]>;

    /// Sample a 3D image, whose third coordinate is normalized.
    fn sample_3d(&self, handle: u32, _u: f32, _v: f32, _w: f32) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: 3D sample of handle {handle:#x} with no 3D source bound"
        )))
    }

    /// Sample a cubemap, whose three coordinates are a direction.
    fn sample_cube(&self, handle: u32, _s: f32, _t: f32, _r: f32) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: cube sample of handle {handle:#x} with no cube source bound"
        )))
    }

    /// Sample cube `cube` of a cube array in the given direction.
    fn sample_cube_array(
        &self,
        handle: u32,
        _s: f32,
        _t: f32,
        _r: f32,
        _cube: u32,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: cube-array sample of handle {handle:#x} with no cube source bound"
        )))
    }

    /// One texel in normalized coordinates, `(1/width, 1/height)`, for scaling
    /// `tex.aoffi` offsets (added before clamping, unlike hardware).
    fn texel_step(&self, handle: u32) -> ShaderResult<(f32, f32)> {
        Err(fault(format!(
            "shader: texel offset on handle {handle:#x} with no texture source bound"
        )))
    }

    /// Width, height, depth or layer count, and mip level count, at level 0.
    fn dimensions(&self, handle: u32) -> ShaderResult<[u32; 4]> {
        Err(fault(format!(
            "shader: size query of handle {handle:#x} with no texture source bound"
        )))
    }

    /// Channel `component` of a bilinear footprint; see [`crate::gpu::texture::gather_with`].
    fn gather(
        &self,
        handle: u32,
        _u: f32,
        _v: f32,
        _layer: u32,
        _component: usize,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: gather of handle {handle:#x} with no texture source bound"
        )))
    }

    /// [`TextureSource::gather`] of the face of cube `cube` that `direction` hits.
    fn gather_cube(
        &self,
        handle: u32,
        _direction: [f32; 3],
        _cube: u32,
        _component: usize,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: cube gather of handle {handle:#x} with no cube source bound"
        )))
    }

    /// [`TextureSource::gather_compare`] of the face of cube `cube` that `direction` hits.
    fn gather_compare_cube(
        &self,
        handle: u32,
        _direction: [f32; 3],
        _cube: u32,
        _reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: cube shadow gather of handle {handle:#x} with no depth source bound"
        )))
    }

    /// The texel at integer `at` (x, y, layer or slice) of level 0, through the descriptor's
    /// swizzle; zero outside.
    fn fetch(&self, handle: u32, _at: [u32; 3]) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: texel fetch of handle {handle:#x} with no texture source bound"
        )))
    }

    /// The four texels of a gather, each compared against `reference` as 1.0 or 0.0.
    fn gather_compare(
        &self,
        handle: u32,
        _u: f32,
        _v: f32,
        _layer: u32,
        _reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: shadow gather of handle {handle:#x} with no depth source bound"
        )))
    }

    /// A shadow sample as `[c, c, c, 1.0]`.
    fn sample_compare(
        &self,
        handle: u32,
        _u: f32,
        _v: f32,
        _layer: u32,
        _reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: shadow sample of handle {handle:#x} with no depth source bound"
        )))
    }

    /// The registers `suld` reads from texel `at` (x, y, layer or slice).
    fn surface_load(
        &self,
        handle: u32,
        _at: [u32; 3],
        _data: SurfaceData,
    ) -> ShaderResult<[u32; 4]> {
        Err(fault(format!(
            "shader: surface load of handle {handle:#x} with no surface source bound"
        )))
    }

    /// Write `sust`'s registers to texel `at` of the image `handle` names.
    fn surface_store(
        &self,
        handle: u32,
        _at: [u32; 3],
        _data: SurfaceData,
        _regs: [u32; 4],
    ) -> ShaderResult<()> {
        Err(fault(format!(
            "shader: surface store to handle {handle:#x} with no surface source bound"
        )))
    }

    /// `sured`: combine `value` into the 32-bit texel `at` of the image `handle` names.
    fn surface_reduce(
        &self,
        handle: u32,
        _at: [u32; 3],
        _op: AtomOp,
        _ty: AtomType,
        _value: u32,
    ) -> ShaderResult<()> {
        Err(fault(format!(
            "shader: surface reduction on handle {handle:#x} with no surface source bound"
        )))
    }
}

/// What `sured` leaves in a texel holding `old`. Only 32-bit texels take a reduction.
pub fn surface_reduced(
    texel_bytes: u32,
    old: u32,
    op: AtomOp,
    ty: AtomType,
    value: u32,
) -> ShaderResult<u32> {
    if texel_bytes != 4 {
        return Err(fault(format!(
            "shader: a surface reduction on a {texel_bytes}-byte texel"
        )));
    }
    Ok(atom_apply(op, ty, u64::from(old), u64::from(value), 0)? as u32)
}

/// No texture backend: every `texs` is an error.
pub struct NoTextures;

impl TextureSource for NoTextures {
    fn sample(&self, handle: u32, _u: f32, _v: f32, _layer: u32) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: texture sample of handle {handle:#x} with no texture source bound"
        )))
    }
}

/// Samples from the TIC/TSC descriptor pools in GPU memory.
pub struct MemoryTextures<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub tex_header_pool: u64,
    pub tex_sampler_pool: u64,
    /// Owned by the caller so this struct can be rebuilt per fragment.
    pub descriptors: &'a std::cell::RefCell<crate::gpu::texture::DescriptorCache>,
    /// Decoded compressed blocks, shared across the draw.
    pub blocks: &'a std::cell::RefCell<crate::gpu::texture::BlockCache>,
    /// Where `sust` queues its writes, landed after each shading step as global stores are;
    /// `None` where a stage cannot store to an image.
    pub surface_stores: Option<&'a std::cell::RefCell<Vec<SurfaceWrite>>>,
}

/// A queued `sust`: the texel's address, its size in bytes and its bytes.
pub type SurfaceWrite = (u64, u32, u128);

impl MemoryTextures<'_, '_> {
    /// Write out queued surface stores oldest first, so a later one to the same texel wins.
    pub fn land(
        ctx: &mut ExecCtx,
        stores: &std::cell::RefCell<Vec<SurfaceWrite>>,
    ) -> crate::Result<()> {
        for (addr, bytes, raw) in stores.borrow_mut().drain(..) {
            ctx.write_pixel(addr, bytes, raw)?;
        }
        Ok(())
    }

    fn surface(&self, handle: u32) -> ShaderResult<Option<crate::gpu::texture::Texture>> {
        let at = self.tex_header_pool + u64::from(crate::gpu::texture::image_id(handle)) * 32;
        Ok(crate::gpu::texture::read_bound_image(self.ctx, at)?)
    }
}

impl MemoryTextures<'_, '_> {
    /// `None` for a null image, which every read here answers with zeros.
    fn descriptors_for(
        &self,
        handle: u32,
    ) -> ShaderResult<Option<crate::gpu::texture::Descriptors>> {
        if let Some(d) = self.descriptors.borrow().get(&handle).copied() {
            return Ok(d);
        }
        let d = crate::gpu::texture::read_descriptors(
            self.ctx,
            self.tex_header_pool,
            self.tex_sampler_pool,
            handle,
        )?;
        self.descriptors.borrow_mut().insert(handle, d);
        Ok(d)
    }
}

impl TextureSource for MemoryTextures<'_, '_> {
    fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::sample_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            layer,
            self.blocks,
        )?)
    }

    fn texel_step(&self, handle: u32) -> ShaderResult<(f32, f32)> {
        let (width, height) = self
            .descriptors_for(handle)?
            .map_or((0, 0), |d| (d.texture.width, d.texture.height));
        Ok((1.0 / width.max(1) as f32, 1.0 / height.max(1) as f32))
    }

    // One level, all either renderer gives a texture; a null image has none.
    fn dimensions(&self, handle: u32) -> ShaderResult<[u32; 4]> {
        Ok(self.descriptors_for(handle)?.map_or([0; 4], |d| {
            [
                d.texture.width,
                d.texture.height,
                d.texture.layers.max(1),
                1,
            ]
        }))
    }

    fn gather(
        &self,
        handle: u32,
        u: f32,
        v: f32,
        layer: u32,
        component: usize,
    ) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::gather_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            layer,
            component,
            self.blocks,
        )?)
    }

    fn gather_cube(
        &self,
        handle: u32,
        direction: [f32; 3],
        cube: u32,
        component: usize,
    ) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::gather_cube_with(
            self.ctx,
            &descriptors,
            direction.map(f64::from),
            cube,
            component,
            self.blocks,
        )?)
    }

    fn gather_compare_cube(
        &self,
        handle: u32,
        direction: [f32; 3],
        cube: u32,
        reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::gather_compare_cube_with(
            self.ctx,
            &descriptors,
            direction.map(f64::from),
            cube,
            reference,
            self.blocks,
        )?)
    }

    fn sample_3d(&self, handle: u32, u: f32, v: f32, w: f32) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::sample_3d_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            w as f64,
            self.blocks,
        )?)
    }

    fn sample_cube(&self, handle: u32, s: f32, t: f32, r: f32) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::sample_cube_with(
            self.ctx,
            &descriptors,
            s as f64,
            t as f64,
            r as f64,
            0,
            self.blocks,
        )?)
    }

    fn sample_cube_array(
        &self,
        handle: u32,
        s: f32,
        t: f32,
        r: f32,
        cube: u32,
    ) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::sample_cube_with(
            self.ctx,
            &descriptors,
            s as f64,
            t as f64,
            r as f64,
            cube,
            self.blocks,
        )?)
    }

    fn fetch(&self, handle: u32, at: [u32; 3]) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::fetch_with(
            self.ctx,
            &descriptors,
            at,
            self.blocks,
        )?)
    }

    fn gather_compare(
        &self,
        handle: u32,
        u: f32,
        v: f32,
        layer: u32,
        reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::gather_compare_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            layer,
            reference,
            self.blocks,
        )?)
    }

    fn sample_compare(
        &self,
        handle: u32,
        u: f32,
        v: f32,
        layer: u32,
        reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(crate::gpu::texture::NULL_TEXEL);
        };
        Ok(crate::gpu::texture::sample_compare_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            layer,
            reference,
            self.blocks,
        )?)
    }

    fn surface_load(&self, handle: u32, at: [u32; 3], data: SurfaceData) -> ShaderResult<[u32; 4]> {
        let Some(image) = self.surface(handle)? else {
            return Ok([0; 4]);
        };
        let Some((addr, bytes)) = image.surface_texel(at)? else {
            return Ok([0; 4]);
        };
        // A store this step queued is newer than memory.
        let queued = self.surface_stores.and_then(|stores| {
            stores
                .borrow()
                .iter()
                .rev()
                .find(|&&(to, len, _)| to == addr && len == bytes)
                .map(|&(_, _, raw)| raw)
        });
        let raw = match queued {
            Some(raw) => raw,
            None => self.ctx.read_pixel(addr, bytes)?,
        };
        Ok(crate::gpu::texture::surface_registers(
            &image, bytes, raw, data,
        )?)
    }

    fn surface_store(
        &self,
        handle: u32,
        at: [u32; 3],
        data: SurfaceData,
        regs: [u32; 4],
    ) -> ShaderResult<()> {
        let Some(stores) = self.surface_stores else {
            return Err(fault(format!(
                "shader: surface store to handle {handle:#x} from a stage that cannot store"
            )));
        };
        let Some(image) = self.surface(handle)? else {
            return Ok(());
        };
        if let Some(write) = crate::gpu::texture::surface_write(&image, at, data, regs)? {
            stores.borrow_mut().push(write);
        }
        Ok(())
    }

    fn surface_reduce(
        &self,
        handle: u32,
        at: [u32; 3],
        op: AtomOp,
        ty: AtomType,
        value: u32,
    ) -> ShaderResult<()> {
        let Some(stores) = self.surface_stores else {
            return Err(fault(format!(
                "shader: surface reduction on handle {handle:#x} from a stage that cannot store"
            )));
        };
        let Some(image) = self.surface(handle)? else {
            return Ok(());
        };
        let Some((addr, bytes)) = image.surface_texel(at)? else {
            return Ok(());
        };
        let queued = stores
            .borrow()
            .iter()
            .rev()
            .find(|&&(to, len, _)| to == addr && len == bytes)
            .map(|&(_, _, raw)| raw);
        let old = match queued {
            Some(raw) => raw as u32,
            None => self.ctx.read_pixel(addr, bytes)? as u32,
        };
        let new = surface_reduced(bytes, old, op, ty, value)?;
        stores.borrow_mut().push((addr, bytes, u128::from(new)));
        Ok(())
    }
}

/// A shader's global (`ldg`/`stg`/`atom`) address space. Writes take `&self`
/// since a dispatch shares it across threads; they default to an error.
pub trait GlobalMemory {
    fn read_u32(&self, addr: u64) -> ShaderResult<u32>;

    fn read_u8(&self, addr: u64) -> ShaderResult<u8> {
        Ok((self.read_u32(addr & !3)? >> ((addr % 4) * 8)) as u8)
    }

    fn write_u32(&self, addr: u64, _value: u32) -> ShaderResult<()> {
        Err(fault(format!(
            "shader: a global store to {addr:#x} from a stage whose memory is read-only"
        )))
    }

    fn write_u8(&self, addr: u64, value: u8) -> ShaderResult<()> {
        let word = addr & !3;
        let shift = (addr % 4) * 8;
        let old = self.read_u32(word)?;
        self.write_u32(word, (old & !(0xFF << shift)) | (u32::from(value) << shift))
    }
}
