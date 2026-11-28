//! The interpreter's constant, global memory and texture sources, reading guest memory.

use crate::gpu::exec::ExecCtx;
use crate::gpu::texture;
use crate::Result;
use std::cell::RefCell;
use switch_shader::interp::{
    surface_reduced, ConstantSource, GlobalMemory, ShaderResult, TextureSource,
};
use switch_shader::isa::{AtomOp, AtomType, SurfaceData};

fn fault(message: String) -> Box<switch_shader::Error> {
    Box::new(switch_shader::Error::Fault(message))
}

/// Reads `cN[offset]` from GPU memory; `bindings` resolves a bank to `(address, size)`.
pub struct MemoryConstants<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub bindings: &'a dyn Fn(u8) -> Option<(u64, u32)>,
    /// See [`ConstCache`].
    pub cache: &'a RefCell<ConstCache>,
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
            .ok_or_else(|| fault(format!("shader: read from unbound constant bank {bank}")))?;
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
    pub stores: &'a RefCell<Vec<(u64, u32)>>,
}

impl MemoryGlobal<'_, '_> {
    /// Write out queued stores oldest first, so a later one to the same word wins.
    pub fn land(ctx: &mut ExecCtx, stores: &RefCell<Vec<(u64, u32)>>) -> Result<()> {
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

/// Samples from the TIC/TSC descriptor pools in GPU memory.
pub struct MemoryTextures<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub tex_header_pool: u64,
    pub tex_sampler_pool: u64,
    /// Owned by the caller so this struct can be rebuilt per fragment.
    pub descriptors: &'a RefCell<texture::DescriptorCache>,
    /// Decoded compressed blocks, shared across the draw.
    pub blocks: &'a RefCell<texture::BlockCache>,
    /// Where `sust` queues its writes, landed after each shading step as global stores are;
    /// `None` where a stage cannot store to an image.
    pub surface_stores: Option<&'a RefCell<Vec<SurfaceWrite>>>,
}

/// A queued `sust`: the texel's address, its size in bytes and its bytes.
pub type SurfaceWrite = (u64, u32, u128);

impl MemoryTextures<'_, '_> {
    /// Write out queued surface stores oldest first, so a later one to the same texel wins.
    pub fn land(ctx: &mut ExecCtx, stores: &RefCell<Vec<SurfaceWrite>>) -> Result<()> {
        for (addr, bytes, raw) in stores.borrow_mut().drain(..) {
            ctx.write_pixel(addr, bytes, raw)?;
        }
        Ok(())
    }

    fn surface(&self, handle: u32) -> ShaderResult<Option<texture::Texture>> {
        let at = self.tex_header_pool + u64::from(texture::image_id(handle)) * 32;
        Ok(texture::read_bound_image(self.ctx, at)?)
    }
}

impl MemoryTextures<'_, '_> {
    /// `None` for a null image, which every read here answers with zeros.
    fn descriptors_for(&self, handle: u32) -> ShaderResult<Option<texture::Descriptors>> {
        if let Some(d) = self.descriptors.borrow().get(&handle).copied() {
            return Ok(d);
        }
        let d = texture::read_descriptors(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::sample_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::gather_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::gather_cube_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::gather_compare_cube_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::sample_3d_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::sample_cube_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::sample_cube_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::fetch_with(
            self.ctx,
            &descriptors,
            at,
            self.blocks,
        )?)
    }

    fn fetch_level(&self, handle: u32, at: [u32; 3], level: u32) -> ShaderResult<[f32; 4]> {
        let Some(descriptors) = self.descriptors_for(handle)? else {
            return Ok(texture::NULL_TEXEL);
        };
        if level == 0 {
            return self.fetch(handle, at);
        }
        // Past the chain there is no texel, so the fetch reads zero as outside the image does.
        if level >= descriptors.texture.levels {
            return Ok([0.0; 4]);
        }
        let Some(texture) = descriptors.texture.level(level) else {
            return Err(fault(format!(
                "shader: texel fetch of level {level} of a pitch or volume image, whose smaller \
                 levels are not addressed"
            )));
        };
        Ok(texture::fetch_with(
            self.ctx,
            &texture::Descriptors {
                texture,
                ..descriptors
            },
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::gather_compare_with(
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
            return Ok(texture::NULL_TEXEL);
        };
        Ok(texture::sample_compare_with(
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
        Ok(texture::surface_registers(&image, bytes, raw, data)?)
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
        if let Some(write) = texture::surface_write(&image, at, data, regs)? {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_constants_reads_a_real_bound_buffer_out_of_gpu_memory() {
        use crate::gpu::syncpt::Host1x;
        use crate::gpu::vmm::AddressSpace;
        use crate::mem::Memory;

        let mut mem = Memory::new();
        mem.map_zero(0x5000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let gpu_va = vmm
            .map(
                0x5000_0000,
                0x1000,
                1,
                0,
                crate::gpu::vmm::SMALL_PAGE_SIZE,
                0,
                0,
            )
            .unwrap();
        vmm.write_u32(&mem, gpu_va + 0x10, 42.5f32.to_bits())
            .unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            as_id: 0,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };

        let bindings = |bank: u8| {
            if bank == 2 {
                Some((gpu_va, 0x1000))
            } else {
                None
            }
        };
        let cache = RefCell::new(ConstCache::default());
        let source = MemoryConstants {
            ctx: &ctx,
            bindings: &bindings,
            cache: &cache,
        };

        assert_eq!(f32::from_bits(source.read_const(2, 0x10).unwrap()), 42.5);
        let unbound = source.read_const(3, 0x10).unwrap_err();
        assert!(unbound.to_string().contains("unbound"), "got {unbound}");
        // The page past the buffer is unmapped too, so only the message says which check refused.
        let past = source.read_const(2, 0x1000).unwrap_err();
        assert!(
            past.to_string().contains("past the bound buffer"),
            "got {past}"
        );
    }
}
