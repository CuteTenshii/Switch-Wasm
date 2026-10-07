use super::*;
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::decode_program;
use crate::gpu::shader::isa::{FMod, FmulScale, Instruction, TexDim};
use crate::gpu::testing::{block, solid_fragment_shader};

fn no_consts() -> HashMap<(u8, u16), f32> {
    HashMap::new()
}

/// A straight-line program at real 32-byte-block byte offsets.
fn prog(ops: &[Op]) -> Compiled {
    let mut p = crate::gpu::shader::Program::default();
    for (i, &op) in ops.iter().enumerate() {
        p.insns.push(Instruction::always(op));
        p.offsets
            .push(crate::gpu::shader::ENTRY_OFFSET + i as u32 * 8);
    }
    Compiled::new(&p)
}
use std::cell::RefCell;

/// Records what it was asked to sample and returns a fixed colour.
struct RecordingTextures {
    calls: RefCell<Vec<(u32, f32, f32, u32)>>,
    color: [f32; 4],
}

impl TextureSource for RecordingTextures {
    fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]> {
        self.calls.borrow_mut().push((handle, u, v, layer));
        Ok(self.color)
    }
}

/// A flat byte-addressed global memory.
#[derive(Default)]
struct FlatMemory {
    bytes: RefCell<Vec<u8>>,
}

impl FlatMemory {
    fn with(size: usize) -> FlatMemory {
        FlatMemory {
            bytes: RefCell::new(vec![0; size]),
        }
    }
}

impl GlobalMemory for FlatMemory {
    fn read_u32(&self, addr: u64) -> ShaderResult<u32> {
        let bytes = self.bytes.borrow();
        let at = addr as usize;
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes[at..at + 4]);
        Ok(u32::from_le_bytes(word))
    }

    fn read_u8(&self, addr: u64) -> ShaderResult<u8> {
        Ok(self.bytes.borrow()[addr as usize])
    }

    fn write_u32(&self, addr: u64, value: u32) -> ShaderResult<()> {
        let at = addr as usize;
        self.bytes.borrow_mut()[at..at + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_u8(&self, addr: u64, value: u8) -> ShaderResult<()> {
        self.bytes.borrow_mut()[addr as usize] = value;
        Ok(())
    }
}

#[test]
fn texs_resolves_its_handle_from_the_driver_constant_bank_and_writes_the_masked_channels() {
    // tex.frag: destinations REG_00 and REG_28, coordinates REG_08 and REG_20.
    let program = prog(&[
        Op::Texs {
            dst: 2,
            dst2: 4,
            coords: [0, 3, RZ], // u=r0, v=r3
            dref: None,
            handle: 0x20,
            dim: TexDim::T2d,
            mask: [true, true, true, true],
            f16: false,
        },
        Op::Exit,
    ]);
    let mut inv = Invocation::new();
    inv.set_reg_f32(0, 0.25); // u
    inv.set_reg_f32(3, 0.75); // v

    let mut consts = HashMap::new();
    let handle = 7u32 | (2u32 << 20); // imageId=7, samplerId=2
                                      // 0x20 is a dword index: byte 0x80.
    consts.insert(
        (crate::gpu::texture::NOUVEAU_TEX_CB_INDEX, 0x80),
        f32::from_bits(handle),
    );
    consts.insert(
        (crate::gpu::texture::NOUVEAU_TEX_CB_INDEX, 0x20),
        f32::from_bits(99),
    );

    let textures = RecordingTextures {
        calls: RefCell::new(Vec::new()),
        color: [0.1, 0.2, 0.3, 0.4],
    };

    inv.execute(&program, &Env::new(&consts, &textures))
        .unwrap();

    assert_eq!(
        textures.calls.borrow().as_slice(),
        &[(handle, 0.25, 0.75, 0)]
    );
    assert_eq!(inv.reg_f32(2), 0.1);
    assert_eq!(inv.reg_f32(3), 0.2);
    assert_eq!(inv.reg_f32(4), 0.3);
    assert_eq!(inv.reg_f32(5), 0.4);
}

#[test]
fn solid_color_fragment_shader_reproduces_the_perspective_corrected_color() {
    // solid.frag (`oColor = vColor;`) through the real decoder, with attr_in
    // already divided by clip-w and 1/w at a[0x7c].
    let w = 2.0f32;
    let color = [0.25f32, 0.5, 0.75, 1.0];

    let program = Compiled::new(&decode_program(&solid_fragment_shader()).unwrap());

    let mut inv = Invocation::new();
    inv.attr_in.set(0x7c, 1.0 / w);
    inv.attr_in.set(0x80, color[0] / w);
    inv.attr_in.set(0x84, color[1] / w);
    inv.attr_in.set(0x88, color[2] / w);
    inv.attr_in.set(0x8c, color[3] / w);

    inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
        .unwrap();

    // Fragment output RT0 is r0-r3.
    assert_eq!(inv.reg_f32(0), color[0]);
    assert_eq!(inv.reg_f32(1), color[1]);
    assert_eq!(inv.reg_f32(2), color[2]);
    assert_eq!(inv.reg_f32(3), color[3]);
}

#[test]
fn mvp_vertex_shader_transforms_a_known_position_via_a_fake_constant_buffer() {
    // mvp.vert (`gl_Position = uMVP * aPosition; vColor = aColor;`) through the real decoder.
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
    let program = Compiled::new(&decode_program(&bytes).unwrap());

    // A std140 mat4 is column-major.
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 3.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut consts: HashMap<(u8, u16), f32> = HashMap::new();
    for (row, values) in m.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            consts.insert((2, (col * 16 + row * 4) as u16), *value);
        }
    }

    let pos = [10.0f32, 20.0, 30.0, 1.0];
    let color = [0.1f32, 0.2, 0.3, 0.4];
    let mut inv = Invocation::new();
    inv.attr_in.set(0x80, pos[0]);
    inv.attr_in.set(0x84, pos[1]);
    inv.attr_in.set(0x88, pos[2]);
    inv.attr_in.set(0x8c, pos[3]);
    inv.attr_in.set(0x90, color[0]);
    inv.attr_in.set(0x94, color[1]);
    inv.attr_in.set(0x98, color[2]);
    inv.attr_in.set(0x9c, color[3]);

    inv.execute(&program, &Env::new(&consts, &NoTextures))
        .unwrap();

    let expected = [
        (0..4).map(|c| m[0][c] * pos[c]).sum::<f32>(),
        (0..4).map(|c| m[1][c] * pos[c]).sum::<f32>(),
        (0..4).map(|c| m[2][c] * pos[c]).sum::<f32>(),
        (0..4).map(|c| m[3][c] * pos[c]).sum::<f32>(),
    ];
    assert_eq!(inv.attr_out.get(0x70), expected[0]);
    assert_eq!(inv.attr_out.get(0x74), expected[1]);
    assert_eq!(inv.attr_out.get(0x78), expected[2]);
    assert_eq!(inv.attr_out.get(0x7c), expected[3]);

    assert_eq!(inv.attr_out.get(0x80), color[0]);
    assert_eq!(inv.attr_out.get(0x84), color[1]);
    assert_eq!(inv.attr_out.get(0x88), color[2]);
    assert_eq!(inv.attr_out.get(0x8c), color[3]);
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
    vmm.write_u32(&mut mem, gpu_va + 0x10, 42.5f32.to_bits())
        .unwrap();

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
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
    let cache = std::cell::RefCell::new(ConstCache::default());
    let source = MemoryConstants {
        ctx: &ctx,
        bindings: &bindings,
        cache: &cache,
    };

    assert_eq!(f32::from_bits(source.read_const(2, 0x10).unwrap()), 42.5);
    assert!(source.read_const(3, 0x10).is_err()); // unbound bank
    assert!(source.read_const(2, 0x1000).is_err()); // past the buffer's size
}

#[test]
fn textured_fragment_shader_multiplies_the_real_sample_by_vertex_colour() {
    // tex.frag (`oColor = texture(uTex, vTexCoord) * vColor;`). With a white vertex
    // colour the output is exactly the sampled colour.
    let mut bytes = block(
        (0xe1a0070f, 0x003c0401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470004, 0x50800000), // mufu rcp $r4 $r0
        (0x0047ff00, 0xe043ff89), // ipa $r0 a[0x90] $r4 0x0 0x1  (u)
    );
    bytes.extend(block(
        (0xe020072f, 0x001cbc03),
        (0x4047ff01, 0xe043ff89), // ipa $r1 a[0x94] $r4 0x0 0x1  (v)
        (0x20170000, 0xd8301a40), // texs $r2 $r0 $r0 $r1 0x1a4 t2d rgba
        (0x0047ff05, 0xe043ff88), // ipa $r5 a[0x80] $r4 0x0 0x1
    ));
    bytes.extend(block(
        (0xe1e01ff0, 0x003fc000),
        (0x00570000, 0x5c681000), // fmul ftz $r0 $r0 $r5
        (0x4047ff05, 0xe043ff88), // ipa $r5 a[0x84] $r4 0x0 0x1
        (0x00570101, 0x5c681000), // fmul ftz $r1 $r1 $r5
    ));
    bytes.extend(block(
        (0xfe00070f, 0x001c3c01),
        (0x8047ff05, 0xe043ff88), // ipa $r5 a[0x88] $r4 0x0 0x1
        (0x00570202, 0x5c681000), // fmul ftz $r2 $r2 $r5
        (0xc047ff04, 0xe043ff88), // ipa $r4 a[0x8c] $r4 0x0 0x1
    ));
    bytes.extend(block(
        (0xfde00ff0, 0x001ffc3f),
        (0x00470303, 0x5c681000), // fmul ftz $r3 $r3 $r4
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
    ));
    let program = Compiled::new(&decode_program(&bytes).unwrap());

    struct StubTex;
    impl TextureSource for StubTex {
        fn sample(&self, _handle: u32, _u: f32, _v: f32, _layer: u32) -> ShaderResult<[f32; 4]> {
            Ok([0.2, 0.4, 0.6, 0.8])
        }
    }

    let w = 2.0f32;
    let color = [1.0f32, 1.0, 1.0, 1.0];
    let mut inv = Invocation::new();
    inv.attr_in.set(0x7c, 1.0 / w);
    inv.attr_in.set(0x90, 0.5 / w); // u
    inv.attr_in.set(0x94, 0.5 / w); // v
    inv.attr_in.set(0x80, color[0] / w);
    inv.attr_in.set(0x84, color[1] / w);
    inv.attr_in.set(0x88, color[2] / w);
    inv.attr_in.set(0x8c, color[3] / w);

    let no_consts: HashMap<(u8, u16), f32> = HashMap::new();
    inv.execute(&program, &Env::new(&no_consts, &StubTex))
        .unwrap();

    assert_eq!(inv.reg_f32(0), 0.2);
    assert_eq!(inv.reg_f32(1), 0.4);
    assert_eq!(inv.reg_f32(2), 0.6);
    assert_eq!(inv.reg_f32(3), 0.8);
}

#[test]
fn an_f16_texs_lands_its_channels_packed_as_halves() {
    // Asphalt 9's splash shader: packed `texs`, read back as half pairs by `h*2` ops.
    let sample = [0.25f32, 0.5, 0.75, 1.0];
    struct StubTex([f32; 4]);
    impl TextureSource for StubTex {
        fn sample(&self, _handle: u32, _u: f32, _v: f32, _layer: u32) -> ShaderResult<[f32; 4]> {
            Ok(self.0)
        }
    }

    // dst = $r1, dst2 = $r0, all four channels, precision bit clear.
    let texs = Op::Texs {
        dst: 1,
        dst2: 0,
        coords: [4, 5, RZ],
        dref: None,
        handle: 0,
        dim: TexDim::T2d,
        mask: [true, true, true, true],
        f16: true,
    };
    let program = Compiled::new(&super::super::Program {
        insns: vec![
            Instruction {
                pred: Pred {
                    reg: 7,
                    negate: false,
                },
                op: texs,
            },
            Instruction {
                pred: Pred {
                    reg: 7,
                    negate: false,
                },
                op: Op::Exit,
            },
        ],
        offsets: vec![8, 0x10],
        ..Default::default()
    });

    let no_consts: HashMap<(u8, u16), f32> = HashMap::new();
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts, &StubTex(sample)))
        .unwrap();

    // r1 holds (r, g) and r0 holds (b, a).
    assert_eq!(inv.reg(1), halves(sample[0], sample[1]));
    assert_eq!(inv.reg(0), halves(sample[2], sample[3]));
}

#[test]
fn an_odd_channel_count_pads_its_second_half_with_zero() {
    struct StubTex;
    impl TextureSource for StubTex {
        fn sample(&self, _handle: u32, _u: f32, _v: f32, _layer: u32) -> ShaderResult<[f32; 4]> {
            Ok([0.25, 0.5, 0.75, 1.0])
        }
    }
    let program = Compiled::new(&super::super::Program {
        insns: vec![
            Instruction {
                pred: Pred {
                    reg: 7,
                    negate: false,
                },
                op: Op::Texs {
                    dst: 2,
                    dst2: 4,
                    coords: [4, 5, RZ],
                    dref: None,
                    handle: 0,
                    dim: TexDim::T2d,
                    mask: [true, true, true, false],
                    f16: true,
                },
            },
            Instruction {
                pred: Pred {
                    reg: 7,
                    negate: false,
                },
                op: Op::Exit,
            },
        ],
        offsets: vec![8, 0x10],
        ..Default::default()
    });

    let no_consts: HashMap<(u8, u16), f32> = HashMap::new();
    let mut inv = Invocation::new();
    inv.execute(&program, &Env::new(&no_consts, &StubTex))
        .unwrap();
    assert_eq!(inv.reg(2), halves(0.25, 0.5));
    assert_eq!(inv.reg(4), halves(0.75, 0.0));
}

fn halves(low: f32, high: f32) -> u32 {
    u32::from(f32_to_f16(low)) | (u32::from(f32_to_f16(high)) << 16)
}

fn lanes(bits: u32) -> [f32; 2] {
    half_lanes(bits, HSwizzle::H1H0)
}

fn hadd2(dst: u8, a: u8, b: u8, asw: HSwizzle, bsw: HSwizzle, merge: HMerge) -> Op {
    Op::Hadd2 {
        dst,
        a,
        am: FMod::NONE,
        asw,
        b: Operand::Reg(b),
        bm: FMod::NONE,
        bsw,
        merge,
        ftz: false,
        sat: false,
    }
}

fn run_half(setup: &[(u8, u32)], ops: &[Op]) -> Invocation {
    let consts = no_consts();
    let env = Env::new(&consts, &NoTextures);
    let mut inv = Invocation::new();
    for &(reg, value) in setup {
        inv.set_reg(reg, value);
    }
    let mut program: Vec<Op> = ops.to_vec();
    program.push(Op::Exit);
    inv.execute(&prog(&program), &env).unwrap();
    inv
}

/// `tex.aoffi` offsets are in texels, scaled by the level's size.
#[test]
fn a_tex_texel_offset_moves_the_sample_by_one_texel() {
    fn word(lo: u32, hi: u32) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&lo.to_le_bytes());
        out[4..].copy_from_slice(&hi.to_le_bytes());
        out
    }
    // `tex.aoffi $r1 $r4 $r7 0x8 t2d r`; coordinates in $r4/$r5, offset in $r7.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&word(0, 0)); // sched
    bytes.extend_from_slice(&word(0xa0770401, 0xc07a0080));
    bytes.extend_from_slice(&word(0x0007000f, 0xe3000000)); // exit
    bytes.extend_from_slice(&word(0, 0));
    let program = Compiled::new(&super::super::decode_program(&bytes).unwrap());

    struct Probe(std::cell::Cell<(f32, f32)>);
    impl TextureSource for Probe {
        fn sample(&self, _h: u32, u: f32, v: f32, _l: u32) -> ShaderResult<[f32; 4]> {
            self.0.set((u, v));
            Ok([1.0, 0.0, 0.0, 1.0])
        }
        fn texel_step(&self, _h: u32) -> ShaderResult<(f32, f32)> {
            Ok((1.0 / 64.0, 1.0 / 32.0))
        }
    }

    let mut consts: HashMap<(u8, u16), f32> = HashMap::new();
    consts.insert(
        (
            crate::gpu::texture::NOUVEAU_TEX_CB_INDEX,
            crate::gpu::texture::handle_offset(8),
        ),
        f32::from_bits(1),
    );
    // +1 texel in x, -1 in y.
    for (packed, want) in [
        (0x00u32, (0.5, 0.5)),
        (0x01, (0.5 + 1.0 / 64.0, 0.5)),
        (0xF0, (0.5, 0.5 - 1.0 / 32.0)),
    ] {
        let probe = Probe(std::cell::Cell::new((0.0, 0.0)));
        let mut inv = Invocation::new();
        inv.set_reg_f32(4, 0.5);
        inv.set_reg_f32(5, 0.5);
        inv.set_reg(7, packed);
        inv.execute(&program, &Env::new(&consts, &probe)).unwrap();
        assert_eq!(probe.0.get(), want, "offset {packed:#04x}");
    }
}

/// `tex` of a cube array: direction after the cube, channels from `$r0`.
#[test]
fn a_cube_array_tex_samples_the_cube_its_layer_register_names() {
    fn word(lo: u32, hi: u32) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&lo.to_le_bytes());
        out[4..].copy_from_slice(&hi.to_le_bytes());
        out
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&word(0, 0)); // sched
    bytes.extend_from_slice(&word(0xfff70400, 0xc03a0087));
    bytes.extend_from_slice(&word(0x0007000f, 0xe3000000)); // exit
    bytes.extend_from_slice(&word(0, 0));
    let program = Compiled::new(&super::super::decode_program(&bytes).unwrap());

    struct Probe(std::cell::Cell<(f32, f32, f32, u32)>);
    impl TextureSource for Probe {
        fn sample(&self, _h: u32, _u: f32, _v: f32, _l: u32) -> ShaderResult<[f32; 4]> {
            Err(fault(
                "a cube array is not sampled as a 2D image".to_owned(),
            ))
        }
        fn sample_cube_array(
            &self,
            _h: u32,
            s: f32,
            t: f32,
            r: f32,
            cube: u32,
        ) -> ShaderResult<[f32; 4]> {
            self.0.set((s, t, r, cube));
            Ok([0.25, 0.5, 0.75, 1.0])
        }
    }

    let mut consts: HashMap<(u8, u16), f32> = HashMap::new();
    consts.insert(
        (
            crate::gpu::texture::NOUVEAU_TEX_CB_INDEX,
            crate::gpu::texture::handle_offset(8),
        ),
        f32::from_bits(1),
    );
    let probe = Probe(std::cell::Cell::new((0.0, 0.0, 0.0, 0)));
    let mut inv = Invocation::new();
    // The cube is the low half.
    inv.set_reg(4, 0x0001_0003);
    inv.set_reg_f32(5, -1.0);
    inv.set_reg_f32(6, 0.5);
    inv.set_reg_f32(7, 0.25);
    inv.execute(&program, &Env::new(&consts, &probe)).unwrap();
    assert_eq!(probe.0.get(), (-1.0, 0.5, 0.25, 3));
    assert_eq!([0, 1, 2, 3].map(|r| inv.reg_f32(r)), [0.25, 0.5, 0.75, 1.0]);
}

/// Convert into RZ to set condition codes, then `csetp.neu` ("not zero").
#[test]
fn csetp_neu_after_an_i2i_cc_asks_whether_the_value_was_zero() {
    let program = [
        isa::decode(0x5ce0800000170aff).op,
        isa::decode(0x50a0038000070d07).op,
        Op::Exit,
    ];
    let consts = no_consts();
    for (value, not_zero) in [(0u32, false), (5, true), ((-3i32) as u32, true)] {
        let mut inv = Invocation::new();
        inv.set_reg(1, value);
        inv.execute(&prog(&program), &Env::new(&consts, &NoTextures))
            .unwrap();
        assert_eq!(inv.pred(0), not_zero, "r1 = {value:#x}");
    }
}

/// `txq` halves width and height per level, never below one.
#[test]
fn txq_reports_the_size_at_the_level_it_is_asked_about() {
    struct Sized;
    impl TextureSource for Sized {
        fn sample(&self, handle: u32, _u: f32, _v: f32, _l: u32) -> ShaderResult<[f32; 4]> {
            Err(fault(format!("no sample of {handle:#x} here")))
        }
        fn dimensions(&self, handle: u32) -> ShaderResult<[u32; 4]> {
            assert_eq!(handle, 0x1234, "the handle slot 8 names");
            Ok([100, 50, 3, 1])
        }
    }
    // `txq $r0 $r8 dimension 0x8 0x3`
    let txq = isa::decode(0xdf48008180470800).op;
    let mut consts = no_consts();
    consts.insert(
        (crate::gpu::texture::NOUVEAU_TEX_CB_INDEX, 0x20),
        f32::from_bits(0x1234),
    );
    for (level, want) in [(0, [100, 50]), (2, [25, 12]), (7, [1, 1]), (40, [1, 1])] {
        let mut inv = Invocation::new();
        inv.set_reg(8, level);
        inv.set_reg(2, 0xdead);
        inv.execute(&prog(&[txq, Op::Exit]), &Env::new(&consts, &Sized))
            .unwrap();
        assert_eq!([inv.reg(0), inv.reg(1)], want, "level {level}");
        assert_eq!(inv.reg(2), 0xdead, "a channel the mask leaves out");
    }
}

/// A bindless `tex.b` samples the handle in its register.
#[test]
fn a_bindless_tex_samples_the_handle_its_register_holds() {
    struct Probe(std::cell::Cell<u32>);
    impl TextureSource for Probe {
        fn sample(&self, handle: u32, _u: f32, _v: f32, _l: u32) -> ShaderResult<[f32; 4]> {
            self.0.set(handle);
            Ok([0.25, 0.5, 0.75, 1.0])
        }
    }
    let op = isa::decode(0xdeba0007a0270000).op;
    let probe = Probe(std::cell::Cell::new(0));
    let consts = no_consts();
    let mut program: Vec<Op> = vec![op];
    program.push(Op::Exit);
    let mut inv = Invocation::new();
    inv.set_reg_f32(0, 0.5);
    inv.set_reg_f32(1, 0.5);
    inv.set_reg(2, 0x0030_0007);
    inv.execute(&prog(&program), &Env::new(&consts, &probe))
        .unwrap();
    assert_eq!(probe.0.get(), 0x0030_0007);
    assert_eq!([0, 1, 2, 3].map(|r| inv.reg_f32(r)), [0.25, 0.5, 0.75, 1.0]);
}

mod alu;
mod compute;
