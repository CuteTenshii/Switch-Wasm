//! Constant, global memory and texture sources an invocation reads.

use super::*;

/// Resolves a `cN[offset]` operand to its raw 32 bits. `bank` is a bind slot,
/// not an address; the host resolves it to the buffer bound there.
pub trait ConstantSource {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32>;
}

impl ConstantSource for HashMap<(u8, u16), f32> {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32> {
        Ok(self.get(&(bank, offset)).copied().unwrap_or(0.0).to_bits())
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

    /// Channel `component` of a bilinear footprint; see switch-core's `texture::gather_with`.
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

    /// [`TextureSource::fetch`] from mip level `level`.
    fn fetch_level(&self, handle: u32, at: [u32; 3], level: u32) -> ShaderResult<[f32; 4]> {
        if level == 0 {
            return self.fetch(handle, at);
        }
        Err(fault(format!(
            "shader: texel fetch of level {level} of handle {handle:#x}, but only level 0 is held"
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
