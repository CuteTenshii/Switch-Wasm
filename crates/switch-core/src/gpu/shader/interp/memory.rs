//! Executing memory, atomic, texture and surface ops.

use super::*;

impl Invocation {
    /// `ld`/`st a[r + imm]`: the index register is a byte offset.
    pub(super) fn attr_index(&self, idx: u8) -> u16 {
        self.reg(idx) as u16
    }

    pub(super) fn reg64(&self, r: u8) -> u64 {
        u64::from(self.reg(r)) | (u64::from(self.reg(r.wrapping_add(1))) << 32)
    }

    fn reg_wide(&self, r: u8, width: usize) -> u64 {
        if width == 8 {
            self.reg64(r)
        } else {
            u64::from(self.reg(r))
        }
    }

    fn set_reg_wide(&mut self, r: u8, width: usize, value: u64) {
        self.set_reg(r, value as u32);
        if width == 8 {
            self.set_reg(r.wrapping_add(1), (value >> 32) as u32);
        }
    }

    pub(super) fn store_value(&self, src: u8, size: MemSize) -> ([u8; 16], usize) {
        let mut out = [0u8; 16];
        for i in 0..size.regs() as usize {
            let word = self.reg(src.wrapping_add(i as u8)).to_le_bytes();
            out[i * 4..i * 4 + 4].copy_from_slice(&word);
        }
        (out, size.bytes() as usize)
    }

    /// `atom`/`atoms`/`red`: read-modify-write, old value to `dst`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_atom(
        &mut self,
        dst: u8,
        addr: u8,
        offset: i32,
        src: u8,
        op: AtomOp,
        ty: AtomType,
        space: AtomSpace,
        env: &Env,
    ) -> ShaderResult<()> {
        let width = match ty {
            AtomType::U128 => {
                return Err(fault("shader: a 128-bit atomic is not implemented".into()))
            }
            AtomType::U64 | AtomType::S64 => 8usize,
            _ => 4usize,
        };
        // `cas` compares against `src` and stores the register after it.
        let operand = self.reg_wide(src, width);
        let stored = self.reg_wide(src.wrapping_add((width / 4) as u8), width);

        let old = match space {
            AtomSpace::Shared => {
                let shared = env.shared.ok_or_else(|| {
                    Error::Gpu("shader: a shared atomic with no shared memory bound".into())
                })?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let block = shared.borrow();
                let words = read_scratch(&block, base, wide_size(width));
                drop(block);
                let old = pack(&words, width);
                let new = atom_apply(op, ty, old, operand, stored)?;
                let mut block = shared.borrow_mut();
                let cap = block.len();
                write_scratch(&mut block, cap, base, &unpack(new, width)[..width]);
                old
            }
            AtomSpace::Global => {
                let mem = env.memory.ok_or_else(|| {
                    Error::Gpu("shader: a global atomic with no global memory bound".into())
                })?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                let mut old = u64::from(mem.read_u32(base)?);
                if width == 8 {
                    old |= u64::from(mem.read_u32(base + 4)?) << 32;
                }
                let new = atom_apply(op, ty, old, operand, stored)?;
                mem.write_u32(base, new as u32)?;
                if width == 8 {
                    mem.write_u32(base + 4, (new >> 32) as u32)?;
                }
                old
            }
        };
        self.set_reg_wide(dst, width, old);
        Ok(())
    }

    /// `texs` results are deferred: compiled code reads the destinations' old values
    /// between the fetch and its first consumer. Each is landed just before its first
    /// reader, or at the next branch or `exit`.
    pub(super) fn run_texs(
        &mut self,
        program: &Compiled,
        pc: usize,
        op: Op,
        env: &Env,
        pending: &mut Vec<(usize, u8, u32)>,
    ) -> ShaderResult<()> {
        let Op::Texs {
            coords,
            dref,
            handle,
            dim,
            ..
        } = op
        else {
            unreachable!("run_texs called with {op:?}");
        };
        // The bindless handle lives in the driver's constant bank; see `texture::handle_offset`.
        let handle = env
            .consts
            .read_const(env.tex_cb_index, crate::gpu::texture::handle_offset(handle))?;
        let u = self.reg_f32(coords[0]);
        let v = self.reg_f32(coords[1]);
        // An array layer is an integer in the low half of its register.
        let layer = match dim {
            TexDim::T2dArray => self.reg(coords[2]) & 0xffff,
            _ => 0,
        };
        let color = match (dref, dim) {
            (Some(reg), _) => {
                env.textures
                    .sample_compare(handle, u, v, layer, self.reg_f32(reg))?
            }
            // A 3D image's third coordinate is normalized.
            (None, TexDim::T3d) => env
                .textures
                .sample_3d(handle, u, v, self.reg_f32(coords[2]))?,
            // A cubemap's three are a direction.
            (None, TexDim::TCube) => {
                env.textures
                    .sample_cube(handle, u, v, self.reg_f32(coords[2]))?
            }
            (None, _) => env.textures.sample(handle, u, v, layer)?,
        };
        self.land_texture(program, pc, color, pending);
        Ok(())
    }

    /// The general `tex`: LOD, offset and shadow reference share one register, and
    /// an array's layer comes before the coordinates.
    pub(super) fn run_tex(
        &mut self,
        program: &Compiled,
        pc: usize,
        op: Op,
        env: &Env,
        pending: &mut Vec<(usize, u8, u32)>,
    ) -> ShaderResult<()> {
        let Op::Tex {
            coords,
            layer,
            dref,
            offset,
            handle,
            handle_reg,
            dim,
            ..
        } = op
        else {
            unreachable!("run_tex called with {op:?}");
        };
        // A bindless sample holds the handle in a register.
        let handle = match handle_reg {
            Some(reg) => self.reg(reg),
            None => env
                .consts
                .read_const(env.tex_cb_index, crate::gpu::texture::handle_offset(handle))?,
        };
        let mut u = self.reg_f32(coords[0]);
        // A 1D image has one coordinate.
        let mut v = match dim {
            TexDim::T1d => 0.0,
            _ => self.reg_f32(coords[1]),
        };
        // `.AOFFI` packs a signed four-bit offset per axis into one register.
        if let Some(reg) = offset {
            let packed = self.reg(reg);
            let axis = |shift: u32| ((packed >> shift) as i32) << 28 >> 28;
            let (du, dv) = env.textures.texel_step(handle)?;
            u += axis(0) as f32 * du;
            v += axis(4) as f32 * dv;
        }
        let layer = layer.map_or(0, |reg| self.reg(reg) & 0xffff);
        let color = match (dref, dim) {
            (Some(reg), _) => {
                env.textures
                    .sample_compare(handle, u, v, layer, self.reg_f32(reg))?
            }
            (None, TexDim::T3d) => env
                .textures
                .sample_3d(handle, u, v, self.reg_f32(coords[2]))?,
            (None, TexDim::TCube) => {
                env.textures
                    .sample_cube(handle, u, v, self.reg_f32(coords[2]))?
            }
            (None, TexDim::TCubeArray) => {
                env.textures
                    .sample_cube_array(handle, u, v, self.reg_f32(coords[2]), layer)?
            }
            (None, _) => env.textures.sample(handle, u, v, layer)?,
        };
        self.land_texture(program, pc, color, pending);
        Ok(())
    }

    /// `txq`: texture size at the `lod` register's level, as integers; layers don't shrink.
    pub(super) fn run_txq(
        &mut self,
        program: &Compiled,
        pc: usize,
        op: Op,
        env: &Env,
        pending: &mut Vec<(usize, u8, u32)>,
    ) -> ShaderResult<()> {
        let Op::Txq { lod, handle, .. } = op else {
            unreachable!("run_txq called with {op:?}");
        };
        let handle = env
            .consts
            .read_const(env.tex_cb_index, crate::gpu::texture::handle_offset(handle))?;
        let [width, height, depth, levels] = env.textures.dimensions(handle)?;
        let level = self.reg(lod);
        let at = |size: u32| size.checked_shr(level).unwrap_or(0).max(1);
        let size = [at(width), at(height), depth, levels];
        self.land_texture(program, pc, size.map(f32::from_bits), pending);
        Ok(())
    }

    /// `tld4`: one channel of a bilinear footprint, offset by `.AOFFI`.
    pub(super) fn run_tld4(
        &mut self,
        program: &Compiled,
        pc: usize,
        op: Op,
        env: &Env,
        pending: &mut Vec<(usize, u8, u32)>,
    ) -> ShaderResult<()> {
        let Op::Tld4 {
            coords,
            layer,
            offset,
            handle,
            component,
            ..
        } = op
        else {
            unreachable!("run_tld4 called with {op:?}");
        };
        let handle = env
            .consts
            .read_const(env.tex_cb_index, crate::gpu::texture::handle_offset(handle))?;
        let mut u = self.reg_f32(coords[0]);
        let mut v = self.reg_f32(coords[1]);
        if let Some(reg) = offset {
            let packed = self.reg(reg);
            let axis = |shift: u32| ((packed >> shift) as i32) << 28 >> 28;
            let (du, dv) = env.textures.texel_step(handle)?;
            u += axis(0) as f32 * du;
            v += axis(4) as f32 * dv;
        }
        let layer = layer.map_or(0, |reg| self.reg(reg) & 0xffff);
        let texels = env
            .textures
            .gather(handle, u, v, layer, usize::from(component))?;
        self.land_texture(program, pc, texels, pending);
        Ok(())
    }

    /// A surface instruction's handle: from its register if bindless, else the texture bank.
    pub(super) fn surface_handle(
        &self,
        handle: u16,
        handle_reg: Option<u8>,
        env: &Env,
    ) -> ShaderResult<u32> {
        match handle_reg {
            Some(reg) => Ok(self.reg(reg)),
            None => env
                .consts
                .read_const(env.tex_cb_index, crate::gpu::texture::handle_offset(handle)),
        }
    }

    /// A surface instruction's x, y and layer (low half of its register).
    pub(super) fn surface_coords(&self, coords: u8, dim: SurfaceDim) -> [u32; 3] {
        let at = |i: u8| self.reg(coords.wrapping_add(i));
        match dim {
            SurfaceDim::D1 | SurfaceDim::Buffer1d => [at(0), 0, 0],
            SurfaceDim::Array1d => [at(0), 0, at(1) & 0xffff],
            SurfaceDim::D2 => [at(0), at(1), 0],
            SurfaceDim::Array2d => [at(0), at(1), at(2) & 0xffff],
            SurfaceDim::D3 => [at(0), at(1), at(2)],
        }
    }

    /// Queue a sample's channels, each due before the first instruction that reads it.
    pub(super) fn land_texture(
        &self,
        program: &Compiled,
        pc: usize,
        color: [f32; 4],
        pending: &mut Vec<(usize, u8, u32)>,
    ) {
        for &(reg, store, due) in program.texs_writes(pc) {
            let raw = match store {
                isa::TexsStore::Float(channel) => color[channel].to_bits(),
                // Low half first; an odd channel count pads with zero.
                isa::TexsStore::Halves(low, high) => {
                    let pack = |c: Option<usize>| {
                        u32::from(f32_to_f16(c.map_or(0.0, |channel| color[channel])))
                    };
                    pack(Some(low)) | pack(high) << 16
                }
            };
            pending.retain(|&(_, r, _)| r != reg);
            pending.push((due, reg, raw));
        }
    }
}
