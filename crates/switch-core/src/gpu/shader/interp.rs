//! Executing decoded Maxwell instructions.
//!
//! [`Invocation`] is deliberately rasterizer-oblivious: it doesn't know
//! whether it's a vertex or fragment shader, or where `attr_in`/constants
//! came from. Real GPU-memory-backed constant buffers and vertex fetch are
//! later stages' job (see `gpu/engine/threed.rs`'s module docs for the
//! staging); this module only needs values placed in its maps directly,
//! which is what makes it independently testable.

use crate::gpu::exec::ExecCtx;
use crate::{Error, Result};
use std::collections::HashMap;

use super::isa::{Instruction, MemSize, Operand};

/// Resolves a `cN[offset]` operand to a value. `bank` is whatever the ISA's
/// `Operand::Const` carries — for real programs that's a constant-buffer
/// *bind slot* (`Bind[]`'s index, not a raw GPU address), so a real source
/// still needs its own way to turn that into bytes; see [`MemoryConstants`].
/// Reads are fallible because a real one touches guest memory.
pub trait ConstantSource {
    fn read_const(&self, bank: u8, offset: u16) -> Result<f32>;
}

impl ConstantSource for HashMap<(u8, u16), f32> {
    fn read_const(&self, bank: u8, offset: u16) -> Result<f32> {
        Ok(self.get(&(bank, offset)).copied().unwrap_or(0.0))
    }
}

/// Reads `cN[offset]` straight out of GPU memory. `bindings` resolves a bank
/// index to the `(address, size)` a real constant buffer was bound to —
/// `Engine3D::bound_constbuf` for the real integration, anything else for
/// tests — so this module stays decoupled from `engine::threed`.
pub struct MemoryConstants<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub bindings: &'a dyn Fn(u8) -> Option<(u64, u32)>,
}

impl ConstantSource for MemoryConstants<'_, '_> {
    fn read_const(&self, bank: u8, offset: u16) -> Result<f32> {
        let (addr, size) = (self.bindings)(bank).ok_or_else(|| {
            Error::Gpu(format!("shader: read from unbound constant bank {}", bank))
        })?;
        if offset as u32 + 4 > size {
            return Err(Error::Gpu(format!(
                "shader: constant read c{}[{:#x}] is past the bound buffer's size {:#x}",
                bank, offset, size
            )));
        }
        let bits = self.ctx.read_u32(addr + offset as u64)?;
        Ok(f32::from_bits(bits))
    }
}

/// Resolves a `texs` sample. `handle` is the packed `imageId | samplerId <<
/// 20` value a real one reads out of the driver's reserved constant bank
/// (see `gpu::texture`'s module docs) — `Invocation::execute` does that
/// read itself via `ConstantSource` before calling this, so this trait only
/// needs to turn a resolved handle plus UVs into a colour.
pub trait TextureSource {
    fn sample(&self, handle: u32, u: f32, v: f32) -> Result<[f32; 4]>;
}

/// No texture backend at all — every `texs` is an error. Correct for vertex
/// shading (this ISA subset never samples textures in a vertex stage) and
/// for tests that don't exercise `texs`.
pub struct NoTextures;

impl TextureSource for NoTextures {
    fn sample(&self, _handle: u32, _u: f32, _v: f32) -> Result<[f32; 4]> {
        Err(Error::Gpu("shader: texs with no texture backend bound".into()))
    }
}

/// Samples straight out of GPU memory via the bound TIC/TSC pools —
/// `Engine3D::tex_header_pool`/`tex_sampler_pool` for the real integration.
pub struct MemoryTextures<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub tex_header_pool: u64,
    pub tex_sampler_pool: u64,
}

impl TextureSource for MemoryTextures<'_, '_> {
    fn sample(&self, handle: u32, u: f32, v: f32) -> Result<[f32; 4]> {
        crate::gpu::texture::sample(
            self.ctx,
            self.tex_header_pool,
            self.tex_sampler_pool,
            handle,
            u as f64,
            v as f64,
        )
    }
}

/// Hardware's zero register: reads as 0, writes are discarded.
const RZ: u8 = 0xff;

/// Per-vertex/per-fragment machine state: 255 general-purpose registers
/// (`r0`..`r254`; `r255` is [`RZ`]) plus the `a[]` attribute-space input and
/// output maps, keyed by the same byte offset the ISA uses (`a[0x7c]`
/// becomes key `0x7c`).
#[derive(Debug)]
pub struct Invocation {
    gpr: [u32; 255],
    pub attr_in: HashMap<u16, f32>,
    pub attr_out: HashMap<u16, f32>,
}

impl Default for Invocation {
    fn default() -> Self {
        Invocation {
            gpr: [0; 255],
            attr_in: HashMap::new(),
            attr_out: HashMap::new(),
        }
    }
}

impl Invocation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reg_f32(&self, r: u8) -> f32 {
        f32::from_bits(self.reg(r))
    }

    pub fn set_reg_f32(&mut self, r: u8, v: f32) {
        self.set_reg(r, v.to_bits());
    }

    fn reg(&self, r: u8) -> u32 {
        if r == RZ {
            0
        } else {
            self.gpr[r as usize]
        }
    }

    fn set_reg(&mut self, r: u8, v: u32) {
        if r != RZ {
            self.gpr[r as usize] = v;
        }
    }

    fn operand(&self, op: Operand, consts: &dyn ConstantSource) -> Result<f32> {
        match op {
            Operand::Reg(r) => Ok(self.reg_f32(r)),
            Operand::Const { bank, offset } => consts.read_const(bank, offset),
        }
    }

    fn attr_slots(size: MemSize) -> u8 {
        match size {
            MemSize::B32 => 1,
            MemSize::B64 => 2,
            MemSize::B96 => 3,
            MemSize::B128 => 4,
        }
    }

    /// Execute `program` to completion. `program` must end in
    /// [`Instruction::Exit`] — every path [`super::decode_program`] returns
    /// does — so falling off the end without one is this function's own
    /// bug, not a guest error.
    pub fn execute(
        &mut self,
        program: &[Instruction],
        consts: &dyn ConstantSource,
        textures: &dyn TextureSource,
    ) -> Result<()> {
        for insn in program {
            match *insn {
                Instruction::Exit => return Ok(()),

                Instruction::Ld { dst, offset, size } => {
                    for i in 0..Self::attr_slots(size) {
                        let v = self
                            .attr_in
                            .get(&(offset + i as u16 * 4))
                            .copied()
                            .unwrap_or(0.0);
                        self.set_reg_f32(dst.wrapping_add(i), v);
                    }
                }

                Instruction::St { offset, src, size } => {
                    for i in 0..Self::attr_slots(size) {
                        let v = self.reg_f32(src.wrapping_add(i));
                        self.attr_out.insert(offset + i as u16 * 4, v);
                    }
                }

                Instruction::Ipa { dst, offset, mul, perspective } => {
                    let mut v = self.attr_in.get(&offset).copied().unwrap_or(0.0);
                    if perspective {
                        if let Some(m) = mul {
                            v *= self.reg_f32(m);
                        }
                    }
                    self.set_reg_f32(dst, v);
                }

                Instruction::MufuRcp { dst, src } => {
                    self.set_reg_f32(dst, 1.0 / self.reg_f32(src));
                }

                Instruction::Fmul { dst, a, b, .. } => {
                    let v = self.reg_f32(a) * self.operand(b, consts)?;
                    self.set_reg_f32(dst, v);
                }

                Instruction::Fadd { dst, a, b, .. } => {
                    let v = self.reg_f32(a) + self.operand(b, consts)?;
                    self.set_reg_f32(dst, v);
                }

                Instruction::Mov32i { dst, imm } => {
                    self.set_reg_f32(dst, f32::from_bits(imm));
                }

                Instruction::Ffma { dst, a, b, c, .. } => {
                    let v = self.reg_f32(a) * self.operand(b, consts)? + self.reg_f32(c);
                    self.set_reg_f32(dst, v);
                }

                Instruction::Texs { dst, coords, handle, mask, .. } => {
                    // The bindless handle lives in the driver's reserved
                    // constant bank at the shader's own immediate offset —
                    // see `gpu::texture`'s module docs for how that was
                    // confirmed. `t2d`'s two coordinates are REG_00 (u) and
                    // REG_20 (v); REG_08 (coords[1]) is unused for a plain
                    // 2D sample (it's a 3rd-dimension/array slot on other
                    // `TexDim`s this decoder doesn't need yet).
                    let handle_bits = self
                        .operand(Operand::Const { bank: crate::gpu::texture::DRIVER_CONSTBUF_BANK, offset: handle }, consts)?
                        .to_bits();
                    let u = self.reg_f32(coords[0]);
                    let v = self.reg_f32(coords[2]);
                    let color = textures.sample(handle_bits, u, v)?;
                    let mut r = dst;
                    for (channel, &enabled) in mask.iter().enumerate() {
                        if enabled {
                            self.set_reg_f32(r, color[channel]);
                            r = r.wrapping_add(1);
                        }
                    }
                }

                Instruction::Unimplemented { raw } => {
                    return Err(Error::Gpu(format!(
                        "shader: unimplemented instruction {:#018x}",
                        raw
                    )));
                }
            }
        }
        Err(Error::Gpu("shader: program has no exit".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shader::decode_program;
    use crate::gpu::shader::isa::TexDim;
    use std::cell::RefCell;

    fn no_consts() -> HashMap<(u8, u16), f32> {
        HashMap::new()
    }

    /// Records the `(handle, u, v)` it was asked to sample and always
    /// returns the same colour, so a test can check both what the
    /// interpreter computed and what it fed the texture backend.
    struct RecordingTextures {
        calls: RefCell<Vec<(u32, f32, f32)>>,
        color: [f32; 4],
    }

    impl TextureSource for RecordingTextures {
        fn sample(&self, handle: u32, u: f32, v: f32) -> Result<[f32; 4]> {
            self.calls.borrow_mut().push((handle, u, v));
            Ok(self.color)
        }
    }

    #[test]
    fn a_hand_written_alu_program_produces_the_expected_registers() {
        // r2 = r0 * r1; r3 = r2 * r1 + r0. Register-register forms only, so
        // no constant source is exercised — this is purely the interpreter's
        // execute loop, independent of the decoder and of any real shader.
        let program = vec![
            Instruction::Fmul { dst: 2, a: 0, b: Operand::Reg(1), ftz: true },
            Instruction::Ffma { dst: 3, a: 2, b: Operand::Reg(1), c: 0, ftz: true },
            Instruction::Exit,
        ];
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 2.0);
        inv.set_reg_f32(1, 3.0);

        inv.execute(&program, &no_consts(), &NoTextures).unwrap();

        assert_eq!(inv.reg_f32(2), 6.0);
        assert_eq!(inv.reg_f32(3), 20.0);
    }

    #[test]
    fn rz_reads_as_zero_and_discards_writes() {
        let program = vec![
            Instruction::Fmul { dst: 0xff, a: 0, b: Operand::Reg(1), ftz: true },
            Instruction::Ffma { dst: 2, a: 0xff, b: Operand::Reg(1), c: 5, ftz: true },
            Instruction::Exit,
        ];
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 99.0);
        inv.set_reg_f32(1, 3.0);
        inv.set_reg_f32(5, 7.0);

        inv.execute(&program, &no_consts(), &NoTextures).unwrap();

        // dst=RZ: the write to r255 is discarded, not aliased to some slot.
        assert_eq!(inv.reg_f32(2), 0.0 * 3.0 + 7.0);
    }

    #[test]
    fn texs_resolves_its_handle_from_the_driver_constant_bank_and_writes_the_masked_channels() {
        // tex.frag's real shape: "texs $r2 $r0 $r0 $r1 <handle_offset> t2d rgba".
        let program = vec![
            Instruction::Texs {
                dst: 2,
                coords: [0, 1, 3], // u=r0, [unused for t2d]=r1, v=r3
                handle: 0x20,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
            },
            Instruction::Exit,
        ];
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 0.25); // u
        inv.set_reg_f32(3, 0.75); // v

        let mut consts = HashMap::new();
        let handle = 7u32 | (2u32 << 20); // imageId=7, samplerId=2
        consts.insert((crate::gpu::texture::DRIVER_CONSTBUF_BANK, 0x20), f32::from_bits(handle));

        let textures = RecordingTextures {
            calls: RefCell::new(Vec::new()),
            color: [0.1, 0.2, 0.3, 0.4],
        };

        inv.execute(&program, &consts, &textures).unwrap();

        assert_eq!(textures.calls.borrow().as_slice(), &[(handle, 0.25, 0.75)]);
        assert_eq!(inv.reg_f32(2), 0.1);
        assert_eq!(inv.reg_f32(3), 0.2);
        assert_eq!(inv.reg_f32(4), 0.3);
        assert_eq!(inv.reg_f32(5), 0.4);
    }

    #[test]
    fn solid_color_fragment_shader_reproduces_the_perspective_corrected_color() {
        // solid.frag: `oColor = vColor;` — a fixture from the same envydis
        // capture `isa`'s module docs cite, run end to end through the real
        // decoder. The rasterizer normally supplies attr_in already divided
        // by clip-w plus 1/w itself at a[0x7c]; we inject that directly here
        // since Stage 3 is scoped to the interpreter, not vertex fetch.
        let w = 2.0f32;
        let color = [0.25f32, 0.5, 0.75, 1.0];

        fn word(low: u32, high: u32) -> [u8; 8] {
            (((high as u64) << 32) | low as u64).to_le_bytes()
        }
        fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
            let mut out = Vec::with_capacity(32);
            out.extend_from_slice(&word(sched.0, sched.1));
            out.extend_from_slice(&word(a.0, a.1));
            out.extend_from_slice(&word(b.0, b.1));
            out.extend_from_slice(&word(c.0, c.1));
            out
        }
        let mut bytes = block(
            (0xe1a0070f, 0x00240401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
            (0x00470003, 0x50800000), // mufu rcp $r3 $r0
            (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
        );
        bytes.extend(block(
            (0xb0400341, 0x055c8400),
            (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
            (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
            (0xc037ff03, 0xe043ff88), // ipa $r3 a[0x8c] $r3 0x0 0x1
        ));
        bytes.extend(block(
            (0xffe1ffef, 0x001f8000),
            (0x0007000f, 0xe3000000), // exit
            (0xff87000f, 0xe2400fff),
            (0x00070f00, 0x50b00000),
        ));

        let program = decode_program(&bytes).unwrap();

        let mut inv = Invocation::new();
        inv.attr_in.insert(0x7c, 1.0 / w);
        inv.attr_in.insert(0x80, color[0] / w);
        inv.attr_in.insert(0x84, color[1] / w);
        inv.attr_in.insert(0x88, color[2] / w);
        inv.attr_in.insert(0x8c, color[3] / w);

        inv.execute(&program, &no_consts(), &NoTextures).unwrap();

        // Fragment output RT0 is registers r0-r3.
        assert_eq!(inv.reg_f32(0), color[0]);
        assert_eq!(inv.reg_f32(1), color[1]);
        assert_eq!(inv.reg_f32(2), color[2]);
        assert_eq!(inv.reg_f32(3), color[3]);
    }

    #[test]
    fn mvp_vertex_shader_transforms_a_known_position_via_a_fake_constant_buffer() {
        // mvp.vert: `gl_Position = uMVP * aPosition; vColor = aColor;` — the
        // Stage 0 fixture cited in `isa`'s module docs, run end to end
        // through the real decoder with a hand-picked matrix standing in for
        // a real bound constant buffer (real GPU-memory wiring is
        // `MemoryConstants`, exercised separately below).
        fn word(low: u32, high: u32) -> [u8; 8] {
            (((high as u64) << 32) | low as u64).to_le_bytes()
        }
        fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
            let mut out = Vec::with_capacity(32);
            out.extend_from_slice(&word(sched.0, sched.1));
            out.extend_from_slice(&word(a.0, a.1));
            out.extend_from_slice(&word(b.0, b.1));
            out.extend_from_slice(&word(c.0, c.1));
            out
        }
        let mut bytes = block(
            (0xfc20070f, 0x081f8441),
            (0x0807ff00, 0xefd9ff80), // ld b128 $r0 a[0x80] 0x0
            (0x00070004, 0x4c681008), // fmul ftz $r4 $r0 c2[0x0]
            (0x00170005, 0x4c681008), // fmul ftz $r5 $r0 c2[0x4]
        );
        bytes.extend(block(
            (0xfc6207e1, 0x081f8400),
            (0x00270006, 0x4c681008), // fmul ftz $r6 $r0 c2[0x8]
            (0x00370000, 0x4c681008), // fmul ftz $r0 $r0 c2[0xc]
            (0x00470104, 0x49a00208), // ffma ftz $r4 $r1 c2[0x10] $r4
        ));
        bytes.extend(block(
            (0xfc2207e1, 0x001f8c40),
            (0x00570105, 0x49a00288), // ffma ftz $r5 $r1 c2[0x14] $r5
            (0x00670106, 0x49a00308), // ffma ftz $r6 $r1 c2[0x18] $r6
            (0x00770100, 0x49a00008), // ffma ftz $r0 $r1 c2[0x1c] $r0
        ));
        bytes.extend(block(
            (0xfc2207e1, 0x081f8440),
            (0x00870201, 0x49a00208), // ffma ftz $r1 $r2 c2[0x20] $r4
            (0x00970204, 0x49a00288), // ffma ftz $r4 $r2 c2[0x24] $r5
            (0x00a70205, 0x49a00308), // ffma ftz $r5 $r2 c2[0x28] $r6
        ));
        bytes.extend(block(
            (0xfc2007e3, 0x081f8440),
            (0x00b70206, 0x49a00008), // ffma ftz $r6 $r2 c2[0x2c] $r0
            (0x00c70300, 0x49a00088), // ffma ftz $r0 $r3 c2[0x30] $r1
            (0x00d70301, 0x49a00208), // ffma ftz $r1 $r3 c2[0x34] $r4
        ));
        bytes.extend(block(
            (0xfcc207e1, 0x00038800),
            (0x00e70302, 0x49a00288), // ffma ftz $r2 $r3 c2[0x38] $r5
            (0x00f70303, 0x49a00308), // ffma ftz $r3 $r3 c2[0x3c] $r6
            (0x0707ff00, 0xeff1ff80), // st b128 a[0x70] $r0 0x0
        ));
        bytes.extend(block(
            (0x1c200f0f, 0x07ffbc01),
            (0x0907ff00, 0xefd9ff80), // ld b128 $r0 a[0x90] 0x0
            (0x0807ff00, 0xeff1ff80), // st b128 a[0x80] $r0 0x0
            (0x0007000f, 0xe3000000), // exit
        ));
        let program = decode_program(&bytes).unwrap();

        // A std140 mat4 is column-major: column c's four rows sit at bytes
        // [c*16, c*16+16). m[row][col] is the usual math notation.
        let m: [[f32; 4]; 4] = [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut consts: HashMap<(u8, u16), f32> = HashMap::new();
        for col in 0..4 {
            for row in 0..4 {
                consts.insert((2, (col * 16 + row * 4) as u16), m[row][col]);
            }
        }

        let pos = [10.0f32, 20.0, 30.0, 1.0];
        let color = [0.1f32, 0.2, 0.3, 0.4];
        let mut inv = Invocation::new();
        inv.attr_in.insert(0x80, pos[0]);
        inv.attr_in.insert(0x84, pos[1]);
        inv.attr_in.insert(0x88, pos[2]);
        inv.attr_in.insert(0x8c, pos[3]);
        inv.attr_in.insert(0x90, color[0]);
        inv.attr_in.insert(0x94, color[1]);
        inv.attr_in.insert(0x98, color[2]);
        inv.attr_in.insert(0x9c, color[3]);

        inv.execute(&program, &consts, &NoTextures).unwrap();

        let expected = [
            (0..4).map(|c| m[0][c] * pos[c]).sum::<f32>(),
            (0..4).map(|c| m[1][c] * pos[c]).sum::<f32>(),
            (0..4).map(|c| m[2][c] * pos[c]).sum::<f32>(),
            (0..4).map(|c| m[3][c] * pos[c]).sum::<f32>(),
        ];
        assert_eq!(inv.attr_out[&0x70], expected[0]);
        assert_eq!(inv.attr_out[&0x74], expected[1]);
        assert_eq!(inv.attr_out[&0x78], expected[2]);
        assert_eq!(inv.attr_out[&0x7c], expected[3]);

        // vColor = aColor passthrough.
        assert_eq!(inv.attr_out[&0x80], color[0]);
        assert_eq!(inv.attr_out[&0x84], color[1]);
        assert_eq!(inv.attr_out[&0x88], color[2]);
        assert_eq!(inv.attr_out[&0x8c], color[3]);
    }

    #[test]
    fn memory_constants_reads_a_real_bound_buffer_out_of_gpu_memory() {
        use crate::gpu::syncpt::Host1x;
        use crate::gpu::vmm::AddressSpace;
        use crate::mem::Memory;

        let mut mem = Memory::new();
        mem.map_zero(0x5000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let gpu_va = vmm
            .map(0x5000_0000, 0x1000, 1, 0, crate::gpu::vmm::SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        vmm.write_u32(&mut mem, gpu_va + 0x10, 42.5f32.to_bits())
            .unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };

        let bindings = |bank: u8| if bank == 2 { Some((gpu_va, 0x1000)) } else { None };
        let source = MemoryConstants { ctx: &ctx, bindings: &bindings };

        assert_eq!(source.read_const(2, 0x10).unwrap(), 42.5);
        assert!(source.read_const(3, 0x10).is_err()); // unbound bank
        assert!(source.read_const(2, 0x1000).is_err()); // past the buffer's size
    }
}
