//! Backend tests.

use switch_core::gpu::testing::Harness;

/// The one device every test shares, since opening one takes most of a test's time.
static SHARED: std::sync::OnceLock<
    Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), String>,
> = std::sync::OnceLock::new();

/// Held for a test's whole use of the device: concurrent tests on one software Vulkan device fail at random.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A fresh backend on the shared device, with the lease that keeps other tests off it until
/// dropped; `None` with a notice when there is no device.
pub(crate) fn device() -> Option<(std::sync::MutexGuard<'static, ()>, super::Gpu)> {
    // A test that panicked while holding the lease leaves nothing to repair.
    let serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match SHARED.get_or_init(super::Gpu::open_parts) {
        Ok((instance, adapter, device, queue)) => {
            let mut gpu = super::Gpu::with_device(
                instance.clone(),
                adapter.clone(),
                device.clone(),
                queue.clone(),
            );
            gpu.device_is_shared = true;
            Some((serial, gpu))
        }
        Err(why) => {
            eprintln!("[gpu] skipped: {why}");
            None
        }
    }
}

/// naga rejects `shader::wgsl`'s dispatch function without its unreachable trailing
/// `return false;` (Tint warns about it); this fails once naga stops requiring it.
#[test]
fn naga_still_needs_a_return_after_a_loop_that_cannot_fall_through() {
    let Some((_serial, gpu)) = device() else {
        return;
    };
    let module = |name: &str, trailing: &str| {
        let src = [
            "fn f() -> bool {",
            "  var pc: u32 = 0u;",
            "  loop {",
            "    switch (pc) {",
            "      case 0u: { return false; }",
            "      default: { return false; }",
            "    }",
            "  }",
            trailing,
            "}",
            "@fragment fn fs_main() -> @location(0) vec4<f32> {",
            "  if (f()) { discard; }",
            "  return vec4<f32>(0.0);",
            "}",
        ]
        .join("\n");
        let _m = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        gpu.failed.lock().ok().and_then(|mut e| e.fresh.take())
    };
    assert!(
        module("with", "  return false;").is_none(),
        "naga rejected the form `shader::wgsl` actually emits"
    );
    assert!(
        module("without", "").is_some(),
        "naga now accepts a function whose loop cannot fall through: drop the \
         trailing `return false;` from `shader::wgsl`, and Tint stops warning"
    );
}

/// The browser's derivative-based quad swap (`Caps::NONE`) passes validation.
#[test]
fn the_quad_swap_a_browser_gets_is_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{Instruction, Operand, Pred, ShflMode};
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let mut program = Program::default();
    for (index, op) in [
        Op::Shfl {
            dst: 1,
            pred: 0,
            src: 2,
            index: Operand::Imm(1),
            mask: Operand::Imm(0x1c),
            mode: ShflMode::Bfly,
        },
        Op::Exit,
    ]
    .into_iter()
    .enumerate()
    {
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
        .expect("a shuffle translates without the device's quad operations");
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad swap"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
}

/// Mario Kart 8 Deluxe's `tld4s`, on a float or an integer image, and an integer image read into
/// an integer target, are valid WGSL.
#[test]
fn mario_karts_tld4s_and_an_integer_target_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let mut program = Program::default();
    let ops = [0xdf020080c0a7010e].map(|word| isa::decode(word).op);
    for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
        assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
        .expect("a tld4s translates");
    for uint in [false, true] {
        let mut layout = Layout::of(&translated, Stage::Fragment);
        layout.textures[0].uint = uint;
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tld4s"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
    }
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(
        rejected.is_none(),
        "naga rejected the tld4s module: {rejected:?}"
    );

    // A `texs` of an integer image, written to an integer target.
    let mut program = Program::default();
    let texs = Op::Texs {
        dst: 0,
        dst2: 2,
        coords: [4, 5, isa::RZ],
        dref: None,
        handle: 8,
        dim: isa::TexDim::T2d,
        mask: [true; 4],
        f16: false,
    };
    for (index, op) in [texs, Op::Exit].into_iter().enumerate() {
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated =
        wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE).expect("a texs translates");
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.textures[0].uint = true;
    // `RGBA32_UINT` and `RGBA16_UINT`.
    for max in [u32::MAX, 0xFFFF] {
        layout.uint_targets = vec![(0, max)];
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("integer texs"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
        assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    }
}

/// The `.fmz`, `.cc`, `.x`, `.brev` and `lop3` predicate forms are valid WGSL.
#[test]
fn condition_code_extended_and_fmz_forms_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{
        BoolOp, FCmp, FMod, FmulScale, Half16, ICmp, Iadd3Shift, Instruction, LogicOp, LopTest,
        Operand, Pred,
    };
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let reg = Operand::Reg;
    let none = FMod::NONE;
    let ops = [
        Op::Fmul {
            dst: 1,
            a: 2,
            b: reg(3),
            bm: none,
            ftz: false,
            fmz: true,
            sat: false,
            scale: FmulScale::None,
        },
        Op::Ffma {
            dst: 1,
            a: 2,
            b: reg(3),
            bneg: false,
            c: reg(4),
            cneg: false,
            ftz: false,
            fmz: true,
            sat: false,
        },
        Op::Iadd3 {
            dst: 1,
            a: 2,
            aneg: false,
            b: reg(3),
            bneg: false,
            c: reg(4),
            cneg: false,
            x: false,
            cc: true,
            halves: [Half16::All; 3],
            shift: Iadd3Shift::None,
        },
        Op::Iadd3 {
            dst: 1,
            a: 2,
            aneg: false,
            b: reg(3),
            bneg: false,
            c: reg(4),
            cneg: false,
            x: true,
            cc: false,
            halves: [Half16::All; 3],
            shift: Iadd3Shift::None,
        },
        Op::Isetp {
            p0: 0,
            p1: 1,
            a: 2,
            b: reg(3),
            cmp: ICmp::Gt,
            signed: false,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            x: true,
        },
        Op::Iset {
            dst: 1,
            a: 2,
            b: reg(3),
            cmp: ICmp::Le,
            signed: true,
            bop: BoolOp::Or,
            src: Pred::ALWAYS,
            bf: true,
            x: true,
            cc: false,
        },
        Op::Iset {
            dst: 1,
            a: 2,
            b: reg(3),
            cmp: ICmp::Eq,
            signed: true,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            bf: false,
            x: false,
            cc: true,
        },
        Op::Lop {
            dst: 1,
            a: 2,
            ainv: false,
            b: reg(3),
            binv: false,
            op: LogicOp::Xor,
            pred: Some((2, LopTest::Zero)),
            cc: true,
        },
        Op::Lop3 {
            dst: 1,
            a: 2,
            b: reg(3),
            c: reg(4),
            lut: 0xe8,
            pred: Some((3, LopTest::NonZero)),
        },
        Op::Bfe {
            dst: 1,
            a: 2,
            b: reg(3),
            signed: true,
            brev: false,
            cc: true,
        },
        Op::Pset {
            dst: 1,
            a: Pred::ALWAYS,
            b: Pred::ALWAYS,
            c: Pred::ALWAYS,
            op1: BoolOp::And,
            op2: BoolOp::Or,
            float: true,
            cc: true,
        },
        Op::Fset {
            dst: 1,
            a: 2,
            am: none,
            b: reg(3),
            bm: none,
            cmp: FCmp::Lt,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            bf: false,
            cc: true,
        },
        Op::R2p {
            src: 1,
            mask: Operand::Imm(0xf),
            byte: 0,
            flags: true,
        },
        Op::Shr {
            dst: 1,
            a: 2,
            b: reg(3),
            signed: true,
            wrap: false,
            brev: true,
        },
        Op::Exit,
    ];
    let mut program = Program::default();
    for (index, op) in ops.into_iter().enumerate() {
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated =
        wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE).expect("translates");
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flags"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
}

/// Persona 5 Royal's `tlds` and shadow `tld4s` are valid WGSL, folded images included.
#[test]
fn persona_5_royals_tlds_and_shadow_tld4s_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    for words in [[0xda4600aff0d70404u64], [0xdf0601e040480202]] {
        let mut program = Program::default();
        for (index, op) in words
            .map(|word| isa::decode(word).op)
            .into_iter()
            .chain([Op::Exit])
            .enumerate()
        {
            assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
            program.insns.push(Instruction {
                pred: Pred::ALWAYS,
                op,
            });
            program.offsets.push(index as u32 * 8);
        }
        let translated =
            wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE).expect("translates");
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
        let _ = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("persona"),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
        assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    }
}

/// `ops` as a fragment module with `tweak` applied to its layout, which the device validates.
fn validated_module(
    gpu: &super::Gpu,
    label: &str,
    ops: &[switch_core::gpu::shader::Op],
    tweak: impl FnOnce(&mut super::Layout),
) -> String {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let mut program = Program::default();
    for (index, op) in ops.iter().copied().chain([Op::Exit]).enumerate() {
        assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
        .unwrap_or_else(|e| panic!("{label} translates: {e}"));
    let mut layout = Layout::of(&translated, Stage::Fragment);
    tweak(&mut layout);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
    source
}

/// Every `tlds` shape: an offset, an array layer, a volume slice, and a folded image's rows.
#[test]
fn every_tlds_shape_is_wgsl_naga_accepts() {
    use switch_core::gpu::shader::isa::{TexDim, RZ};
    use switch_core::gpu::shader::Op;

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let tlds = |dim, coords, layer, offset, handle| Op::Tlds {
        dst: 0,
        dst2: RZ,
        coords,
        layer,
        lod: None,
        offset,
        handle,
        dim,
        mask: [true, true, false, false],
        f16: false,
    };
    let ops = [
        tlds(TexDim::T2d, [4, 5, RZ], None, Some(6), 8),
        tlds(TexDim::T2dArray, [4, 5, RZ], Some(7), None, 9),
        tlds(TexDim::T3d, [4, 5, 6], None, None, 10),
    ];
    let source = validated_module(&gpu, "tlds shapes", &ops, |_| {});
    assert!(
        source.contains("textureLoad(tex2, vec3<u32>(x, y, z), 0)"),
        "{source}"
    );

    let folded = validated_module(&gpu, "folded tlds", &ops[..1], |layout| {
        layout.textures[0].fold = super::wgsl::Fold::of(100, 16, 64, false);
        assert!(layout.textures[0].fold.is_some());
    });
    assert!(folded.contains("fk"), "{folded}");
}

/// An image bound as a smaller level answers size queries, fetches and texel offsets in the
/// guest's texels, in WGSL naga accepts.
#[test]
fn a_smaller_level_reads_in_the_guests_texels() {
    use switch_core::gpu::shader::isa::{TexDim, RZ};
    use switch_core::gpu::shader::Op;

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let ops = [
        Op::Mov32i { dst: 7, imm: 0xf1 },
        Op::Tex {
            dst: 0,
            coords: [4, 5, RZ],
            layer: Some(6),
            dref: None,
            offset: Some(7),
            lod: None,
            handle: 8,
            handle_reg: None,
            dim: TexDim::T2dArray,
            mask: [true; 4],
        },
        Op::Tlds {
            dst: 0,
            dst2: RZ,
            coords: [4, 5, RZ],
            layer: None,
            lod: None,
            offset: None,
            handle: 9,
            dim: TexDim::T2d,
            mask: [true, true, false, false],
            f16: false,
        },
        Op::Txq {
            dst: 0,
            lod: RZ,
            handle: 10,
            handle_reg: None,
            mask: [true, true, false, false],
        },
    ];
    let source = validated_module(&gpu, "smaller level", &ops, |layout| {
        for binding in &mut layout.textures {
            binding.shrink = 2;
        }
    });
    for shrunk in [
        "let x = x >> 2u; let y = y >> 2u;",
        "(textureDimensions(tex2) << vec2<u32>(2u))",
        "u + f32(1) / d.x, v + f32(-1) / d.y",
    ] {
        assert!(source.contains(shrunk), "{shrunk} in {source}");
    }
}

/// A gather whose offset is in a register, folded images included, is valid WGSL.
#[test]
fn a_gather_with_a_register_offset_is_wgsl_naga_accepts() {
    use switch_core::gpu::shader::isa::{TexDim, RZ};
    use switch_core::gpu::shader::Op;

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let ops = [
        Op::Tld4 {
            dst: 0,
            coords: [4, 5, 5],
            layer: None,
            offset: Some(7),
            ptp: None,
            dref: None,
            handle: 8,
            handle_reg: None,
            sparse: 7,
            dim: TexDim::T2d,
            component: 1,
            mask: [true; 4],
        },
        Op::Tld4s {
            dst: 10,
            dst2: 12,
            coords: [4, 5],
            offset: Some(7),
            dref: None,
            handle: 9,
            component: 0,
            f16: false,
        },
        Op::Tld4 {
            dst: 20,
            coords: [4, 5, 5],
            layer: Some(RZ),
            offset: Some(7),
            ptp: None,
            dref: None,
            handle: 10,
            handle_reg: None,
            sparse: 7,
            dim: TexDim::T2dArray,
            component: 0,
            mask: [true; 4],
        },
    ];
    let source = validated_module(&gpu, "register gather offsets", &ops, |_| {});
    assert_eq!(
        source
            .matches("extractBits(bitcast<i32>(r7), 8u, 6u)")
            .count(),
        3
    );

    let folded = validated_module(&gpu, "folded gather offset", &ops[..1], |layout| {
        layout.textures[0].fold = super::wgsl::Fold::of(100, 16, 64, false);
        assert!(layout.textures[0].fold.is_some());
    });
    assert!(
        folded.contains("return vec2<f32>(1.0 / f32(textureDimensions(tex0).x), 1.0 / f32(100u))"),
        "{folded}"
    );
}

/// Every `tld4` form, shadow, cube and per-texel offsets included, is valid WGSL.
#[test]
fn every_tld4_form_is_wgsl_naga_accepts() {
    use switch_core::gpu::shader::isa::{Operand, TexDim};
    use switch_core::gpu::shader::Op;

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let tld4 = |dim, layer, ptp, dref, handle, handle_reg, sparse| Op::Tld4 {
        dst: 0,
        coords: [4, 5, 6],
        layer,
        offset: None,
        ptp,
        dref,
        handle,
        handle_reg,
        sparse,
        dim,
        component: 1,
        mask: [true; 4],
    };
    let ops = [
        tld4(TexDim::T2d, None, None, Some(10), 8, None, 7),
        tld4(TexDim::T2dArray, Some(2), None, Some(10), 9, None, 7),
        tld4(TexDim::TCube, None, None, None, 10, None, 7),
        tld4(TexDim::TCube, None, None, Some(10), 11, None, 7),
        tld4(TexDim::TCubeArray, Some(2), None, None, 12, None, 7),
        tld4(TexDim::TCubeArray, Some(2), None, Some(10), 13, None, 7),
        tld4(TexDim::T2d, None, Some([7, 8]), None, 14, None, 7),
        tld4(TexDim::T2d, None, Some([7, 8]), Some(10), 15, None, 7),
        Op::Mov {
            dst: 20,
            src: Operand::Const {
                bank: 3,
                offset: 0x20,
            },
        },
        tld4(TexDim::T2d, None, None, None, 0, Some(20), 7),
        // Its sparse predicate is cleared.
        tld4(TexDim::T2d, None, None, None, 16, None, 3),
    ];
    let source = validated_module(&gpu, "tld4 forms", &ops, |_| {});
    for texture in [
        "texture_depth_2d;",
        "texture_depth_2d_array",
        "texture_cube<f32>",
        "texture_depth_cube;",
        "texture_cube_array<f32>",
        "texture_depth_cube_array",
    ] {
        assert!(source.contains(texture), "{texture}: {source}");
    }
}

/// Every global store, narrow load and atomic form translates to valid WGSL.
#[test]
fn global_stores_loads_and_atomics_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{
        AtomOp, AtomSpace, AtomType, Instruction, MemSize, Pred, RZ,
    };
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let atom = |op, ty| Op::Atom {
        dst: 3,
        addr: 0,
        offset: -4,
        src: 2,
        op,
        ty,
        space: AtomSpace::Global,
    };
    let mut ops = vec![Op::Ldc {
        dst: 0,
        bank: 1,
        offset: 0x40,
        idx: RZ,
        size: MemSize::B64,
    }];
    for op in [
        AtomOp::Add,
        AtomOp::Min,
        AtomOp::Max,
        AtomOp::And,
        AtomOp::Or,
        AtomOp::Xor,
        AtomOp::Exch,
        AtomOp::Cas,
    ] {
        for ty in [AtomType::U32, AtomType::S32] {
            ops.push(atom(op, ty));
        }
    }
    ops.push(atom(AtomOp::Add, AtomType::F32));
    for size in [
        MemSize::U8,
        MemSize::S8,
        MemSize::U16,
        MemSize::S16,
        MemSize::B32,
        MemSize::B64,
        MemSize::B128,
    ] {
        ops.push(Op::Stg {
            addr: 0,
            offset: 5,
            src: 4,
            size,
        });
        ops.push(Op::Ldg {
            dst: 8,
            addr: 0,
            offset: -3,
            size,
        });
    }
    ops.push(Op::Exit);
    let mut program = Program::default();
    for (index, op) in ops.into_iter().enumerate() {
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated =
        wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE).expect("translates");
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("globals"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
}

/// Tomodachi Life's cube-array `tex` forms and its `vmnmx` translate to valid WGSL.
#[test]
fn tomodachi_lifes_cube_array_and_video_min_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let mut program = Program::default();
    let words = [0xc03a0087fff70400, 0xc1ba0087f0970400, 0x3a2c03e060c70907];
    let ops = words.map(|word| isa::decode(word).op);
    for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
        assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
        .expect("a cube-array tex and a vmnmx translate");
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    assert!(source.contains("texture_cube_array<f32>"), "{source}");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cube array and vmnmx"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
}

#[test]
fn a_float32_target_blends_only_where_the_device_offers_it() {
    use super::{can_blend, wgpu};
    let none = wgpu::Features::empty();
    let offered = wgpu::Features::FLOAT32_BLENDABLE;
    for format in [
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Rg32Float,
        wgpu::TextureFormat::Rgba32Float,
    ] {
        assert!(!can_blend(format, none), "{format:?}");
        assert!(can_blend(format, offered), "{format:?}");
    }
    assert!(can_blend(wgpu::TextureFormat::Rgba16Float, none));
    assert!(can_blend(wgpu::TextureFormat::Rgba8Unorm, none));
}

/// A lost device falls back without failing the flush.
#[test]
fn a_lost_device_keeps_its_cause_rather_than_failing_the_flush() {
    use switch_core::gpu::renderer::Renderer;
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = Harness::new();
    h.triangle([1.0, 0.0, 1.0, 1.0]);
    h.draw_with(&mut gpu).expect("the draw");
    *gpu.lost.lock().unwrap() = Some("Out of memory".into());
    // `flush_with` panics on an error.
    h.flush_with(&mut gpu);
    h.flush_with(&mut gpu);
    assert!(
        gpu.lost().is_some_and(|why| why.contains("Out of memory")),
        "the loss names its cause: {:?}",
        gpu.lost()
    );
}

/// A rejection the backend never asks about is still counted.
#[test]
fn a_rejection_nothing_asked_about_is_still_counted() {
    let Some((_serial, gpu)) = device() else {
        return;
    };
    assert_eq!(gpu.device_errors(), (0, Vec::new()), "nothing rejected yet");

    // `bool` is not a valid fragment output.
    let _m = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rejected"),
            source: wgpu::ShaderSource::Wgsl(
                "@fragment fn fs_main() -> @location(0) bool { return true; }".into(),
            ),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);

    let (count, distinct) = gpu.device_errors();
    assert!(
        count >= 1,
        "the device rejected the module and said nothing"
    );
    assert_eq!(distinct.len(), 1, "one rejection, one distinct message");
    // Not drained.
    assert_eq!(gpu.device_errors(), (count, distinct.clone()));
}

/// Nintendo Switch Sports' `txq` and `tld4` translate to a module the device accepts.
#[test]
fn nintendo_switch_sports_txq_and_tld4_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let mut program = Program::default();
    let words = [0xdf48008180470800, 0xc83a0086aff70208];
    let ops = words.map(|word| isa::decode(word).op);
    for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
        assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
        .expect("a txq of a texture the program gathers from translates");
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    assert!(source.contains("textureGather(0, tex0"), "{source}");
    assert!(source.contains("textureNumLevels(tex0)"), "{source}");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("txq and tld4"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
}

/// Nintendo Switch Sports' `i2i.cc` and `csetp.neu` translate to a module the device accepts.
#[test]
fn nintendo_switch_sports_csetp_is_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some((_serial, gpu)) = device() else {
        return;
    };
    let mut program = Program::default();
    let words = [0x5ce0800000170aff, 0x50a0038000070d07];
    let ops = words.map(|word| isa::decode(word).op);
    for (index, op) in ops.into_iter().chain([Op::Exit]).enumerate() {
        assert!(!matches!(op, Op::Unimplemented { .. }), "{op:?}");
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
        program.offsets.push(index as u32 * 8);
    }
    let translated = wgsl::translate_for(&Compiled::new(&program), wgsl::Caps::NONE)
        .expect("an i2i.cc and a csetp translate");
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = wgsl::module(&translated, Stage::Fragment, &layout).expect("a module");
    assert!(source.contains("var ccZ: bool"), "{source}");
    let _ = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("csetp"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    let rejected = gpu.failed.lock().ok().and_then(|mut e| e.fresh.take());
    assert!(rejected.is_none(), "naga rejected {source}\n{rejected:?}");
}

#[test]
fn every_pass_the_backend_builds_for_itself_compiles() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    // A multisampled colour format and the two readback depth formats.
    let colour = wgpu::TextureFormat::Bgra8Unorm;
    let depths = [
        wgpu::TextureFormat::Depth16Unorm,
        wgpu::TextureFormat::Depth32Float,
    ];

    for depth in depths {
        let _ = gpu.depth_loader(depth);
        gpu.clear_pipeline(super::ClearKey {
            color: None,
            depth: Some(depth),
            write_mask: [true; 4],
        })
        .expect("a depth clear pipeline");
    }
    for write_mask in [[true; 4], [true, false, true, false]] {
        gpu.clear_pipeline(super::ClearKey {
            color: Some(colour),
            depth: None,
            write_mask,
        })
        .expect("a colour clear pipeline");
    }

    // Four samples is the count core WebGPU guarantees.
    let samples = 4;
    assert!(
        gpu.samples_supported(colour, samples),
        "an adapter that will not multisample {colour:?} four ways"
    );
    for (dst, is_depth) in [(colour, false), (depths[0], true), (depths[1], true)] {
        if is_depth && !gpu.samples_supported(dst, samples) {
            continue;
        }
        for key in [
            // Into a device multisample companion, and into a per-pixel one.
            super::ResampleKey {
                entry: "fs_gather",
                dst,
                samples,
                ms_source: false,
                depth: is_depth,
            },
            super::ResampleKey {
                entry: "fs_gather_flat",
                dst,
                samples: 1,
                ms_source: false,
                depth: is_depth,
            },
            // And back out of each.
            super::ResampleKey {
                entry: "fs_scatter",
                dst,
                samples: 1,
                ms_source: true,
                depth: is_depth,
            },
            super::ResampleKey {
                entry: "fs_scatter",
                dst,
                samples: 1,
                ms_source: false,
                depth: is_depth,
            },
        ] {
            gpu.resample_pipeline(key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
        }
    }

    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(
        gpu.device_error(),
        None,
        "the device rejected one of its own passes"
    );
}

/// The grid tables the resampling passes read.
#[test]
fn the_grid_a_resampling_pass_reads_is_the_one_the_rasterizer_uses() {
    use switch_core::gpu::surface::SampleGrid;
    let grid = SampleGrid::new(2, &[0; 16]).expect("a 2x2 grid");
    assert_eq!(grid.count(), 4);
    let bytes = super::grid_bytes(grid);
    let word = |i: usize| {
        u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ])
    };
    assert_eq!(bytes.len(), 8 + 3 * 16 * 4);
    assert_eq!((word(0), word(1)), (2, 2), "the tile a pixel owns");
    for sample in 0..grid.count() {
        let (x, y) = grid.slot(sample);
        assert_eq!(word(2 + sample as usize), x);
        assert_eq!(word(18 + sample as usize), y);
        // The inverse maps the slot back to this sample.
        assert_eq!(word(34 + (y * 2 + x) as usize), sample);
    }
}

/// A deep-block volume's slices start inside one block, a block's height of GOBs apart,
/// and only a slice that fits one GOB is laid out as a plain surface.
#[test]
fn a_volume_slice_is_looked_for_where_the_volume_puts_it() {
    use crate::bind::{slice_fits_one_gob, volume_slice_addr};
    use switch_core::gpu::pipeline::Format;
    use switch_core::gpu::surface::{Layout, GOB_SIZE};
    use switch_core::gpu::upload::TextureKey;

    // Nintendo Switch Sports' colour volume: 16x8 RGBA8 slices, 8 GOBs deep.
    let key = |row_bytes: u32, rows: u32, block_height_gobs: u32| TextureKey {
        addr: 0x3_1b30_a000,
        width: row_bytes / 4,
        height: rows,
        layers: 8,
        layer_stride: 0x400,
        row_bytes,
        rows,
        block_depth_gobs: 8,
        layout: Layout::BlockLinear {
            block_height_gobs,
            block_depth_gobs: 1,
        },
        format: Format::Rgba8Unorm,
        srgb: false,
    };
    let small = key(64, 8, 1);
    assert_eq!(volume_slice_addr(&small, 0), small.addr);
    assert_eq!(
        volume_slice_addr(&small, 1),
        small.addr + u64::from(GOB_SIZE)
    );
    let tall_blocks = key(64, 8, 2);
    assert_eq!(
        volume_slice_addr(&tall_blocks, 3),
        tall_blocks.addr + 3 * 2 * u64::from(GOB_SIZE)
    );
    assert!(slice_fits_one_gob(&small));
    assert!(!slice_fits_one_gob(&key(128, 8, 1)), "two GOBs wide");
    assert!(!slice_fits_one_gob(&key(64, 16, 2)), "two GOBs tall");
}

/// Tears of the Kingdom draws its volumes slice by slice, each slice a target of its own whose
/// tile mode carries the volume's block depth, then samples the volume. Cleared that way, a
/// volume of slices two GOBs tall samples each slice's colour on the device, from the slices
/// it holds, as the rasterizer does from guest memory.
#[test]
fn a_volume_cleared_slice_by_slice_samples_each_slice_on_both_renderers() {
    use switch_core::gpu::renderer::{Renderer, Software};
    use switch_core::gpu::surface::block_linear_volume_offset;
    use switch_core::gpu::testing::{self, Harness};

    const SIDE: u32 = 16;
    const SLICES: u32 = 4;
    const CLEAR_COLOR: u32 = 0x360;
    let colours = [
        [1.0f32, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let packed = |[r, g, b, a]: [f32; 4]| {
        [r, g, b, a].iter().enumerate().fold(0u32, |word, (i, &c)| {
            word | (((c * 255.0) as u32) << (8 * i))
        })
    };
    let sampled = |renderer: &mut dyn Renderer| -> Vec<u32> {
        let mut h = Harness::with_fragment_shader(testing::offset_volume_fragment_shader(0));
        h.bindless_volume_texture();
        // Handles come from the bank the volume's handle is in.
        h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
        let volume = h.base + 0x2000;
        // The harness's own target, put back once the slices are drawn.
        let kept: Vec<(u32, u32)> = (0x200..0x208)
            .chain([0x300, 0x301, 0x3FD, 0x3FE])
            .map(|reg| (reg, h.engine.regs.get(reg)))
            .collect();
        for (z, colour) in colours.iter().enumerate() {
            let slice = volume
                + u64::from(block_linear_volume_offset(
                    0,
                    0,
                    z as u32,
                    SIDE * 4,
                    SIDE,
                    1,
                    SLICES,
                ));
            for (reg, value) in [
                (0x200, (slice >> 32) as u32),
                (0x201, slice as u32),
                (0x202, SIDE),
                (0x203, SIDE),
                (0x204, 0xD5),
                // A volume's slice, one GOB tall per block and four deep.
                (0x205, 1 << 16 | 2 << 8),
                (0x206, 1),
                (0x300, SIDE << 16),
                (0x301, SIDE << 16),
                (0x3FD, SIDE << 16),
                (0x3FE, SIDE << 16),
            ] {
                h.engine.regs.set(reg, value);
            }
            for (channel, value) in colour.iter().enumerate() {
                h.engine
                    .regs
                    .set(CLEAR_COLOR + channel as u32, value.to_bits());
            }
            h.clear_with(renderer, [true; 4]).expect("the clear");
        }
        for (reg, value) in kept {
            h.engine.regs.set(reg, value);
        }
        // TIC 1 as the volume: block-linear A8B8G8R8, one GOB tall per block and four deep.
        let tic = h.base + 0x1400 + 32;
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        let mut ctx = h.ctx();
        for (word, value) in [
            0x08 | (2 << 7) | identity,
            volume as u32,
            (volume >> 32) as u32 | (3 << 21),
            2 << 6,
            (SIDE - 1) | (2 << 23),
            (SIDE - 1) | ((SLICES - 1) << 16),
            0,
            0,
        ]
        .into_iter()
        .enumerate()
        {
            ctx.write_u32(tic + word as u64 * 4, value).unwrap();
        }
        (0..SLICES)
            .map(|z| {
                // The middle of the slice's second row of blocks, then its first.
                let w = (z as f32 + 0.5) / SLICES as f32;
                let mut texels = Vec::new();
                for v in [0.75, 0.25] {
                    h.triangle([0.5, v, w, 1.0]);
                    h.draw_with(renderer).expect("the draw");
                    h.flush_with(renderer);
                    texels.push(h.texel(0, 0));
                }
                assert_eq!(
                    texels[0], texels[1],
                    "slice {z}'s two rows of blocks differ"
                );
                texels[0]
            })
            .collect()
    };
    let want: Vec<u32> = colours.iter().map(|&c| packed(c)).collect();
    assert_eq!(sampled(&mut Software), want, "the rasterizer");
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let drawn = sampled(&mut gpu);
    assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
    assert_eq!(drawn, want, "the device");
}

#[test]
fn vertex_arrays_past_the_limit_fold_into_one_interleaved_buffer_and_one_shared_element() {
    use super::{fit_vertex_buffers, Feed};
    use std::borrow::Cow;
    let attribute = |format, offset, shader_location| wgpu::VertexAttribute {
        format,
        offset,
        shader_location,
    };
    let feeds = vec![
        // Two vertices of a 12-byte position in a 16-byte stride.
        Feed {
            bytes: Cow::Owned((0..32).collect()),
            origin: None,
            read: 0..0,
            attributes: vec![attribute(wgpu::VertexFormat::Float32x3, 0, 0)],
            stride: 16,
            step: wgpu::VertexStepMode::Vertex,
        },
        // Two vertices of a byte pair at offset 2 of a 4-byte stride.
        Feed {
            bytes: Cow::Owned(vec![0, 0, 0xa0, 0xa1, 0, 0, 0xb0, 0xb1]),
            origin: None,
            read: 0..0,
            attributes: vec![attribute(wgpu::VertexFormat::Unorm8x2, 2, 1)],
            stride: 4,
            step: wgpu::VertexStepMode::Vertex,
        },
        // One instance's element.
        Feed {
            bytes: Cow::Owned(vec![0xc0, 0xc1, 0xc2, 0xc3]),
            origin: None,
            read: 0..0,
            attributes: vec![attribute(wgpu::VertexFormat::Uint32, 0, 2)],
            stride: 0,
            step: wgpu::VertexStepMode::Instance,
        },
    ];
    let folded = fit_vertex_buffers(feeds, 1);
    assert_eq!(folded.len(), 2);

    let per_vertex = &folded[0];
    // 12 bytes of position, the pair at 12, padded to 16.
    assert_eq!(per_vertex.stride, 16);
    assert_eq!(
        per_vertex.attributes,
        vec![
            attribute(wgpu::VertexFormat::Float32x3, 0, 0),
            attribute(wgpu::VertexFormat::Unorm8x2, 12, 1),
        ]
    );
    let mut want: Vec<u8> = (0..12).collect();
    want.extend([0xa0, 0xa1, 0, 0]);
    want.extend(16..28);
    want.extend([0xb0, 0xb1, 0, 0]);
    assert_eq!(&per_vertex.bytes[..], &want[..]);

    let shared = &folded[1];
    assert_eq!(shared.stride, 0, "every vertex reads the one element");
    assert_eq!(shared.step, wgpu::VertexStepMode::Vertex);
    assert_eq!(
        shared.attributes,
        vec![attribute(wgpu::VertexFormat::Uint32, 0, 2)]
    );
    assert_eq!(&shared.bytes[..], &[0xc0, 0xc1, 0xc2, 0xc3]);

    // Within the limit nothing changes.
    let within = fit_vertex_buffers(
        vec![Feed {
            bytes: Cow::Owned(vec![0; 16]),
            origin: None,
            read: 0..0,
            attributes: vec![attribute(wgpu::VertexFormat::Float32x4, 0, 0)],
            stride: 16,
            step: wgpu::VertexStepMode::Vertex,
        }],
        8,
    );
    assert_eq!(within.len(), 1);
    assert_eq!(within[0].stride, 16);
}

/// Rows of `Rg32Float` texels read back from `image`, `width` wide and as tall.
fn read_rg32(gpu: &super::Gpu, image: &wgpu::Texture, width: u32) -> Vec<u8> {
    let size = wgpu::Extent3d {
        width,
        height: width,
        depth_or_array_layers: 1,
    };
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(width * width * 8),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        image.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(width),
            },
        },
        size,
    );
    gpu.queue.submit([encoder.finish()]);
    out.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("the readback"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device");
    let read = out
        .slice(..)
        .get_mapped_range()
        .expect("the mapped readback")
        .to_vec();
    read
}

/// `width` x `width` words in an `R32Uint` texture, and an `Rg32Float` one to convert them into.
fn packed_words(gpu: &super::Gpu, words: &[u32], width: u32) -> (wgpu::Texture, wgpu::Texture) {
    let size = wgpu::Extent3d {
        width,
        height: width,
        depth_or_array_layers: 1,
    };
    let texture = |format, usage| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let source = texture(
        wgpu::TextureFormat::R32Uint,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let image = texture(
        wgpu::TextureFormat::Rg32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    gpu.queue.write_texture(
        source.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(width),
        },
        size,
    );
    (source, image)
}

fn xorshift(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
}

/// An uploaded `S8D24` word converts to the stencil's bits and the unorm depth the rasterizer reads.
#[test]
fn the_packed_depth_pass_converts_uploaded_words_bit_for_bit() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let width = 256u32;
    let mut x = 0x9E37_79B9u32;
    let mut words: Vec<u32> = vec![0, 0xff, 0xffff_ff00, u32::MAX, 0x0100, 0x8000_0000];
    while words.len() < (width * width) as usize {
        words.push(xorshift(&mut x));
    }
    let (source, image) = packed_words(&gpu, &words, width);
    gpu.convert_packed_depth(&source, &image, &[]);
    gpu.submit_batch();
    let read = read_rg32(&gpu, &image, width);
    for (i, &word) in words.iter().enumerate() {
        let at = i * 8;
        let red = u32::from_le_bytes(read[at..at + 4].try_into().unwrap());
        let green = f32::from_le_bytes(read[at + 4..at + 8].try_into().unwrap());
        let want = (word >> 8) as f32 / 16_777_215.0;
        assert_eq!(red, word & 0xff, "stencil of {word:#010x}");
        assert_eq!(green.to_bits(), want.to_bits(), "depth of {word:#010x}");
    }
}

/// The packed-depth pass converts every held depth exactly as guest memory and the rasterizer would.
#[test]
fn the_packed_depth_pass_converts_depths_bit_for_bit() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut depths: Vec<f32> = vec![
        0.0,
        -0.5,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        2f32.powi(-25),
        2f32.powi(-24),
        2f32.powi(-24) * 1.5,
        0.25,
        0.5,
        f32::from_bits(0.5f32.to_bits() - 1),
        f32::from_bits(0.5f32.to_bits() + 1),
        f32::from_bits(1f32.to_bits() - 1),
        1.0,
    ];
    let width = 256u32;
    let mut x = 0x1234_5678u32;
    while depths.len() < (width * width) as usize {
        // Uniform in [0, 1) and log-spread small values, from a fixed xorshift.
        let x = xorshift(&mut x);
        depths.push(if x & 1 == 0 {
            (x >> 8) as f32 / 16_777_216.0
        } else {
            f32::from_bits(x % 0x3f80_0000)
        });
    }
    let size = wgpu::Extent3d {
        width,
        height: width,
        depth_or_array_layers: 1,
    };
    let surface = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let staging = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    gpu.queue.write_texture(
        staging.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(width),
        },
        size,
    );
    gpu.load_depth(&surface, &staging, wgpu::TextureFormat::Depth32Float)
        .expect("the depth upload");
    // The image's own depth is overwritten; only its stencil byte survives.
    let words: Vec<u32> = (0..width * width)
        .map(|i| 0xffff_ff00 | (i & 0xff))
        .collect();
    let (source, image) = packed_words(&gpu, &words, width);
    gpu.convert_packed_depth(&source, &image, &[(0, surface)]);
    gpu.submit_batch();
    let read = read_rg32(&gpu, &image, width);

    let layout = switch_core::gpu::engine::threed::DepthLayout {
        bytes: 4,
        depth_bits: 24,
        depth_shift: 8,
        stencil_shift: Some(0),
    };
    for (i, &d) in depths.iter().enumerate() {
        // Depth32Float holds what the upload wrote; read it as the device does.
        let stored = (layout.encode_depth(d) >> 8) as u32;
        let want = stored as f32 / 16_777_215.0;
        let at = i * 8;
        let red = u32::from_le_bytes(read[at..at + 4].try_into().unwrap());
        let got = f32::from_le_bytes(read[at + 4..at + 8].try_into().unwrap());
        assert_eq!(red, i as u32 & 0xff, "stencil under depth {d:e}");
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "depth {d:e} ({:#x}): stored {stored:#x}",
            d.to_bits()
        );
    }
}

#[test]
fn an_attribute_past_its_stride_reads_from_a_later_buffer() {
    let attribute = |offset, shader_location| wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset,
        shader_location,
    };
    let bytes: Vec<u8> = (0..48).collect();
    let split = super::within_stride(
        std::borrow::Cow::Borrowed(&bytes[..]),
        vec![attribute(0, 0), attribute(32, 1)],
        8,
    );
    assert_eq!(split.len(), 2);
    assert_eq!((split[0].3, split[1].3), (8, 8), "both fit the stride");
    assert_eq!(split[0].0.as_ref(), &bytes[..]);
    assert_eq!(split[0].1, [attribute(0, 0)]);
    assert_eq!(split[1].0.len(), bytes.len(), "a draw must still fit it");
    assert_eq!(split[1].0[0], 32, "the first element's bytes");
    assert_eq!(split[1].0[47], 0, "past the array reads zero");
    assert_eq!(split[1].1, [attribute(0, 1)]);
}

/// Tears of the Kingdom's 8-byte half-float attribute on a 6-byte stride reads two bytes of the
/// next vertex, as Maxwell does, from a copy laid out 8 bytes apart.
#[test]
fn an_attribute_running_into_the_next_element_reads_its_bytes() {
    let wide = wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float16x4,
        offset: 0,
        shader_location: 0,
    };
    let bytes: Vec<u8> = (1..=18).collect();
    let parts = super::within_stride(std::borrow::Cow::Borrowed(&bytes[..]), vec![wide], 6);
    assert_eq!(parts.len(), 1);
    let (restrided, attributes, shift, stride) = &parts[0];
    assert_eq!((*shift, *stride), (None, 8));
    assert_eq!(attributes, &[wide]);
    assert_eq!(
        restrided.as_ref(),
        [1, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 18, 0, 0],
        "each element is the eight bytes at six apart, zeros past the end"
    );
}

/// Tears of the Kingdom's attribute at offset 10, which WebGPU refuses for a 4-byte format,
/// reads the same bytes from a copy that puts it on a word.
#[test]
fn an_attribute_off_its_alignment_reads_from_a_copy_on_it() {
    let attribute = |format, offset, shader_location| wgpu::VertexAttribute {
        format,
        offset,
        shader_location,
    };
    let bytes: Vec<u8> = (1..=28).collect();
    let parts = super::within_stride(
        std::borrow::Cow::Borrowed(&bytes[..]),
        vec![
            attribute(wgpu::VertexFormat::Unorm8x2, 0, 0),
            attribute(wgpu::VertexFormat::Float16x2, 10, 3),
        ],
        14,
    );
    assert_eq!(parts.len(), 1);
    let (copied, attributes, shift, stride) = &parts[0];
    assert_eq!((*shift, *stride), (None, 8));
    assert_eq!(
        attributes,
        &[
            attribute(wgpu::VertexFormat::Unorm8x2, 0, 0),
            attribute(wgpu::VertexFormat::Float16x2, 4, 3),
        ]
    );
    assert_eq!(
        copied.as_ref(),
        [1, 2, 0, 0, 11, 12, 13, 14, 15, 16, 0, 0, 25, 26, 27, 28]
    );
}

#[test]
fn a_stride_off_a_word_is_widened_to_one() {
    let bytes = [1u8, 2, 3, 4, 5, 6];
    let (widened, stride) = super::aligned_stride(std::borrow::Cow::Borrowed(&bytes[..]), 2);
    assert_eq!(stride, 4);
    assert_eq!(widened.as_ref(), [1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0]);
    let (same, stride) = super::aligned_stride(std::borrow::Cow::Borrowed(&bytes[..4]), 4);
    assert_eq!((same.as_ref(), stride), (&bytes[..4], 4));
}

#[test]
fn same_bytes_agrees_with_slice_equality() {
    use super::bind::same_bytes;
    let base: Vec<u8> = (0..200u32).map(|i| (i * 7 + 3) as u8).collect();
    assert!(same_bytes(&base, &base.clone()));
    assert!(same_bytes(&[], &[]));
    assert!(!same_bytes(&base, &base[..199]));
    for at in 0..base.len() {
        let mut other = base.clone();
        other[at] ^= 0x40;
        assert!(!same_bytes(&base, &other), "a difference at byte {at}");
    }
}

/// `width` x `height` RGBA8 texels of `image`, without the row padding a copy needs.
fn read_rgba8(gpu: &super::Gpu, image: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
    let padded = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        image.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    out.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("the readback"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device");
    let mapped = out
        .slice(..)
        .get_mapped_range()
        .expect("the mapped readback");
    let mut rows = Vec::new();
    for row in 0..height {
        let at = (row * padded) as usize;
        rows.extend_from_slice(&mapped[at..at + (width * 4) as usize]);
    }
    rows
}

#[test]
fn the_scan_out_pass_crops_flips_and_drops_alpha_without_touching_the_bytes() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = (7u32, 5u32);
    let size = |w, h| wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    fn texture(
        gpu: &super::Gpu,
        format: wgpu::TextureFormat,
        w: u32,
        h: u32,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }
    // Every byte differs, and alpha is not opaque.
    let texels: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            [
                (i * 3 + 1) as u8,
                (i * 7 + 90) as u8,
                (i * 11 + 200) as u8,
                (i * 5) as u8,
            ]
        })
        .collect();
    for (source_format, target_format) in [
        (
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ),
        (
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ),
    ] {
        let source = texture(
            &gpu,
            source_format,
            width,
            height,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue.write_texture(
            source.as_image_copy(),
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size(width, height),
        );
        for (flip_h, flip_v) in [(false, false), (true, false), (false, true), (true, true)] {
            let (x, y, w, h) = (2u32, 1u32, 4u32, 3u32);
            let target = texture(
                &gpu,
                target_format,
                w,
                h,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            gpu.scan_out_pass(
                &source,
                (x, y, w, h),
                (flip_h, flip_v),
                &view,
                target_format,
            );
            let mut expected = Vec::new();
            for row in 0..h {
                for column in 0..w {
                    let sx = x + if flip_h { w - 1 - column } else { column };
                    let sy = y + if flip_v { h - 1 - row } else { row };
                    let at = ((sy * width + sx) * 4) as usize;
                    expected.extend_from_slice(&[texels[at], texels[at + 1], texels[at + 2], 255]);
                }
            }
            assert_eq!(
                read_rgba8(&gpu, &target, w, h),
                expected,
                "{source_format:?} flipped h {flip_h} v {flip_v}"
            );
        }
    }
}

/// A compute launch run on the device leaves guest memory alone until a flush, which then
/// writes what the software dispatch writes.
#[test]
fn a_compute_launch_on_the_device_writes_what_the_software_dispatch_writes() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{indexed_store_program, ComputeHarness};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let program = indexed_store_program();
    let (grid, block) = ([3, 1, 1], [4, 1, 1]);

    let mut software = ComputeHarness::new(&program, grid, block, 12);
    let engine = software.engine();
    switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
    let expected = software.output(12);
    assert_eq!(expected, (0..12).map(|i| i * 4 + 7).collect::<Vec<u32>>());

    let mut device = ComputeHarness::new(&program, grid, block, 12);
    let engine = device.engine();
    let ran = gpu.dispatch(&engine, &mut device.ctx()).unwrap();
    assert!(ran, "refused: {:?}", gpu.compute_refusals);
    assert_eq!(
        device.output(12),
        vec![0; 12],
        "held on the device until a flush"
    );
    let mut flushes = 0;
    while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    assert_eq!(device.output(12), expected);
}

/// Two surfaces at one address, as a rebinding guest draws them: the first, dirty, and a
/// second over the same 128 bytes laid out differently. The first's texture holds `seed`.
fn rebound_pair(
    gpu: &mut super::Gpu,
    h: &mut switch_core::gpu::testing::ComputeHarness,
    seed: u8,
) -> (
    switch_core::gpu::upload::Target,
    switch_core::gpu::upload::Target,
) {
    use switch_core::gpu::pipeline::Format;
    use switch_core::gpu::surface::Layout;
    use switch_core::gpu::upload::Target;
    let surface = |width: u32, height: u32| Target {
        format: Format::Rgba8Unorm,
        addr: h.base + 0x2000,
        width,
        height,
        layout: Layout::Pitch { pitch: width * 4 },
        row_bytes: width * 4,
        rows: height,
        unit: 4,
        depth: None,
    };
    let (first, second) = (surface(8, 4), surface(4, 8));
    gpu.hold(&first, &h.ctx()).unwrap();
    let held = gpu.space.held.get_mut(&first.addr).unwrap();
    held.dirty = true;
    let texels: Vec<u8> = (0..128).map(|i| seed.wrapping_add(i)).collect();
    gpu.queue.write_texture(
        held.texture.as_image_copy(),
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(32),
            rows_per_image: Some(4),
        },
        held.texture.size(),
    );
    (first, second)
}

/// The 128 guest bytes at `addr`.
fn guest_bytes(h: &mut switch_core::gpu::testing::ComputeHarness, addr: u64) -> Vec<u8> {
    let ctx = h.ctx();
    (0..128)
        .map(|i| ctx.vmm_read_u8(addr + i).unwrap())
        .collect()
}

/// A surface the guest rebinds stays on the device rather than going back, and drawing into it
/// again brings back its own texture and texels, with nothing read back on the way.
#[test]
fn a_rebound_surface_held_again_comes_back_without_a_readback() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::ComputeHarness;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = ComputeHarness::new(&[], [1, 1, 1], [1, 1, 1], 0);
    let (first, second) = rebound_pair(&mut gpu, &mut h, 7);
    let texture = gpu.space.held[&first.addr].texture.clone();
    gpu.hold(&second, &h.ctx()).unwrap();
    assert_eq!(gpu.space.retired.len(), 1);
    assert!(
        gpu.space.pending.is_empty(),
        "the rebound surface was read back"
    );
    gpu.hold(&first, &h.ctx()).unwrap();
    let held = &gpu.space.held[&first.addr];
    assert_eq!(held.target, first);
    assert_eq!(held.texture, texture, "held again from a new texture");
    assert!(held.dirty);
    assert!(
        gpu.space.pending.is_empty(),
        "held again through a readback"
    );
    let mut flushes = 0;
    while gpu.flush(&mut h.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    let expected: Vec<u8> = (0..128).map(|i| 7u8.wrapping_add(i)).collect();
    assert_eq!(guest_bytes(&mut h, first.addr), expected);
}

/// An engine about to read memory a retired surface covers gets the surface's texels there.
#[test]
fn a_read_of_a_retired_surface_sends_it_back() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::ComputeHarness;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = ComputeHarness::new(&[], [1, 1, 1], [1, 1, 1], 0);
    let (first, second) = rebound_pair(&mut gpu, &mut h, 40);
    gpu.hold(&second, &h.ctx()).unwrap();
    let span = first.addr..first.addr + first.len();
    assert!(!gpu.guest_memory_is_current(std::slice::from_ref(&span), &mut h.ctx()));
    let mut synced = 0;
    while gpu
        .sync_ranges(std::slice::from_ref(&span), &[], &mut h.ctx())
        .unwrap()
        == Flush::Pending
    {
        synced += 1;
        assert!(synced < 100, "the readback never landed");
    }
    assert!(gpu.space.retired.is_empty());
    let expected: Vec<u8> = (0..128).map(|i| 40u8.wrapping_add(i)).collect();
    assert_eq!(guest_bytes(&mut h, first.addr), expected);
}

/// A retired surface nothing holds again goes back once enough presents pass without it.
#[test]
fn a_retired_surface_goes_back_after_presents_without_it() {
    use switch_core::gpu::testing::ComputeHarness;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = ComputeHarness::new(&[], [1, 1, 1], [1, 1, 1], 0);
    let (_, second) = rebound_pair(&mut gpu, &mut h, 1);
    gpu.hold(&second, &h.ctx()).unwrap();
    gpu.age_retired();
    assert_eq!(gpu.space.retired.len(), 1, "sent back before any present");
    let mut presents = 0;
    while !gpu.space.retired.is_empty() {
        gpu.frames_presented += 1;
        gpu.age_retired();
        presents += 1;
        assert!(presents < 100, "never sent back");
    }
    assert!(presents > 1, "sent back at the first present");
    assert_eq!(gpu.space.pending.len(), 1, "sent back without a readback");
}

/// A sync for an engine's ranges sends back only the held surfaces over them: one it reads
/// stays held, one it overwrites is let go, and one beside them stays on the device unsent.
#[test]
fn a_sync_for_ranges_sends_back_only_the_surfaces_over_them() {
    use switch_core::gpu::pipeline::Format;
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::surface::Layout;
    use switch_core::gpu::testing::ComputeHarness;
    use switch_core::gpu::upload::Target;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = ComputeHarness::new(&[], [1, 1, 1], [1, 1, 1], 0);
    let surface = |offset: u64| Target {
        format: Format::Rgba8Unorm,
        addr: h.base + offset,
        width: 8,
        height: 4,
        layout: Layout::Pitch { pitch: 32 },
        row_bytes: 32,
        rows: 4,
        unit: 4,
        depth: None,
    };
    let (read, written, beside) = (surface(0x2000), surface(0x2400), surface(0x2800));
    let texels = |seed: u8| -> Vec<u8> { (0..128).map(|i| seed.wrapping_add(i)).collect() };
    for (target, seed) in [(read, 1), (written, 2), (beside, 3)] {
        gpu.hold(&target, &h.ctx()).unwrap();
        let held = gpu.space.held.get_mut(&target.addr).unwrap();
        held.dirty = true;
        gpu.queue.write_texture(
            held.texture.as_image_copy(),
            &texels(seed),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(32),
                rows_per_image: Some(4),
            },
            held.texture.size(),
        );
    }
    let span = |t: &Target| t.addr..t.addr + t.len();
    let mut synced = 0;
    while gpu
        .sync_ranges(&[span(&read)], &[span(&written)], &mut h.ctx())
        .unwrap()
        == Flush::Pending
    {
        synced += 1;
        assert!(synced < 100, "the readback never landed");
    }
    let guest = |h: &mut ComputeHarness, t: &Target| -> Vec<u8> {
        let ctx = h.ctx();
        (0..128)
            .map(|i| ctx.vmm_read_u8(t.addr + i).unwrap())
            .collect()
    };
    assert_eq!(guest(&mut h, &read), texels(1));
    assert_eq!(guest(&mut h, &written), texels(2));
    assert_eq!(guest(&mut h, &beside), vec![0; 128], "sent back unasked");
    assert!(gpu
        .space
        .held
        .get(&read.addr)
        .is_some_and(|held| !held.dirty));
    assert!(!gpu.space.held.contains_key(&written.addr));
    assert!(gpu
        .space
        .held
        .get(&beside.addr)
        .is_some_and(|held| held.dirty));
}

/// A launch that samples a texture and stores to an image runs on the device, and once
/// flushed the image holds what the software dispatch stores.
#[test]
fn a_compute_launch_stores_sampled_texels_to_an_image_on_the_device() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{
        image_copy_program, image_copy_texel, ComputeHarness, IMAGE_COPY_SIZE,
    };

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];
    let program = image_copy_program();
    // Sampled at 2^x / 16 and 2^y / 2 of the texture's width and height.
    let column = |x: u32| ((1u32 << x) / 16).min(width - 1);
    let row = |y: u32| ((1u32 << y) / 2).min(height - 1);
    let expected: Vec<u32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| image_copy_texel(column(x), row(y))))
        .collect();

    let mut software = ComputeHarness::new(&program, [1, 1, 1], block, 0);
    let engine = software.image_copy();
    switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
    assert_eq!(software.image(), expected);

    let mut device = ComputeHarness::new(&program, [1, 1, 1], block, 0);
    let engine = device.image_copy();
    assert!(
        gpu.dispatch(&engine, &mut device.ctx()).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    assert_eq!(
        device.image(),
        vec![0; expected.len()],
        "held until a flush"
    );
    let mut flushes = 0;
    while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    assert_eq!(device.image(), expected);
}

/// A 2D-array store to a one-layer image lands in layer 0 and is dropped past it, on the device
/// as in the software dispatch.
#[test]
fn a_compute_array_store_reaches_only_the_layer_the_image_has() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{
        image_copy_array_program, image_copy_program, ComputeHarness, IMAGE_COPY_SIZE,
    };

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];
    let mut plain = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    let engine = plain.image_copy();
    switch_core::gpu::compute::dispatch(&engine, &mut plain.ctx()).unwrap();
    let flat = plain.image();
    assert!(
        flat.iter().any(|&texel| texel != 0),
        "the 2D store wrote nothing"
    );
    for layer in [0, 1] {
        let mut software = ComputeHarness::new(&image_copy_array_program(), [1, 1, 1], block, 0);
        let engine = software.image_copy();
        software.set_image_layer(layer);
        switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
        let want = if layer == 0 {
            flat.clone()
        } else {
            vec![0; flat.len()]
        };
        assert_eq!(
            software.image(),
            want,
            "layer {layer}: the software dispatch"
        );

        let mut device = ComputeHarness::new(&image_copy_array_program(), [1, 1, 1], block, 0);
        let engine = device.image_copy();
        device.set_image_layer(layer);
        assert!(
            gpu.dispatch(&engine, &mut device.ctx()).unwrap(),
            "layer {layer}: refused: {:?}",
            gpu.compute_refusals
        );
        let mut flushes = 0;
        while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
            flushes += 1;
            assert!(flushes < 100, "the readback never landed");
        }
        assert_eq!(device.image(), want, "layer {layer}: the device");
    }
}

/// A 2D-array store to a three-layer image lands in the layer it names, on the device as in
/// the software dispatch, and one past the last layer is dropped.
#[test]
fn a_compute_array_store_lands_in_the_layer_it_names() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{
        image_copy_array_program, image_copy_program, ComputeHarness, IMAGE_COPY_SIZE,
    };

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];
    let mut plain = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    let engine = plain.image_copy();
    switch_core::gpu::compute::dispatch(&engine, &mut plain.ctx()).unwrap();
    let stored = plain.image();
    let blank = vec![0; stored.len()];
    for layer in 0..4 {
        let layers = |h: &mut ComputeHarness| (0..3).map(|l| h.image_layer(l)).collect::<Vec<_>>();
        let want: Vec<Vec<u32>> = (0..3)
            .map(|l| {
                if l == layer {
                    stored.clone()
                } else {
                    blank.clone()
                }
            })
            .collect();

        let mut software = ComputeHarness::new(&image_copy_array_program(), [1, 1, 1], block, 0);
        let engine = software.image_copy();
        software.image_copy_layers(3);
        software.set_image_layer(layer);
        switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
        assert_eq!(
            layers(&mut software),
            want,
            "layer {layer}: the software dispatch"
        );

        let mut device = ComputeHarness::new(&image_copy_array_program(), [1, 1, 1], block, 0);
        let engine = device.image_copy();
        device.image_copy_layers(3);
        device.set_image_layer(layer);
        assert!(
            gpu.dispatch(&engine, &mut device.ctx()).unwrap(),
            "layer {layer}: refused: {:?}",
            gpu.compute_refusals
        );
        let mut flushes = 0;
        while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
            flushes += 1;
            assert!(flushes < 100, "the readback never landed");
        }
        assert_eq!(layers(&mut device), want, "layer {layer}: the device");
    }
}

/// A store to a `B10G11R11_FLOAT` image, which WebGPU cannot bind for storage, packs the
/// texels the software dispatch packs, through a buffer of the image's words.
#[test]
fn a_compute_store_to_a_packed_float_image_packs_what_software_packs() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{image_copy_program, ComputeHarness, IMAGE_COPY_SIZE};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];
    let mut software = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    let engine = software.image_copy();
    software.image_copy_packed_float();
    switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
    let want = software.image();
    assert!(want.iter().any(|&texel| texel != 0), "nothing was stored");

    let mut device = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    let engine = device.image_copy();
    device.image_copy_packed_float();
    assert!(
        gpu.dispatch(&engine, &mut device.ctx()).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    let mut flushes = 0;
    while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the launch");
    assert_eq!(device.image(), want);
}

/// A launch reducing into an image, which WebGPU has no texture atomics for, runs on the device
/// through a buffer of the image's words and leaves what the software dispatch does.
#[test]
fn a_compute_reduction_into_an_image_runs_on_the_device() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{image_reduce_program, ComputeHarness, IMAGE_COPY_SIZE};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];
    let threads = width * height;
    let mut want = vec![0; threads as usize];
    // `threads` increments wrapping past two land on `threads % 3`.
    want[..3].copy_from_slice(&[threads, width - 1, threads % 3]);

    let mut software = ComputeHarness::new(&image_reduce_program(), [1, 1, 1], block, 0);
    let engine = software.image_copy();
    software.image_copy_words();
    switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
    assert_eq!(software.image(), want, "the software dispatch");

    let mut device = ComputeHarness::new(&image_reduce_program(), [1, 1, 1], block, 0);
    let engine = device.image_copy();
    device.image_copy_words();
    assert!(
        gpu.dispatch(&engine, &mut device.ctx()).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    let mut flushes = 0;
    while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the launch");
    assert_eq!(device.image(), want, "the device");
}

/// What a launch makes for its storage images goes to the scratch list, which destroys it:
/// a browser frees a dropped resource only on GC, and Firefox ran out of device memory first.
#[test]
fn a_compute_launch_hands_what_its_images_made_to_the_scratch_list() {
    use crate::readback::Scratch;
    use switch_core::gpu::renderer::Renderer;
    use switch_core::gpu::testing::{
        image_copy_program, image_reduce_program, ComputeHarness, IMAGE_COPY_SIZE,
    };

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];

    // A reduction stores through a buffer of the image's words.
    let mut reduce = ComputeHarness::new(&image_reduce_program(), [1, 1, 1], block, 0);
    let engine = reduce.image_copy();
    reduce.image_copy_words();
    assert!(
        gpu.dispatch(&engine, &mut reduce.ctx()).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    let words = u64::from(crate::compute::word_row_bytes(width)) * u64::from(height);
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    assert!(
        gpu.scratch
            .iter()
            .any(|made| matches!(made, Scratch::Buffer(buffer)
            if buffer.size() == words && buffer.usage().contains(storage))),
        "the image's words buffer was not handed over"
    );

    // A held surface made writable by a shader: the texture it held before is replaced. The
    // two harnesses share addresses, so the first one's caches go with its address space.
    gpu.submit_batch();
    gpu.release_scratch();
    gpu.address_space_closed(gpu.space_id);
    let mut copy = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    let engine = copy.image_copy();
    assert!(
        gpu.dispatch(&engine, &mut copy.ctx()).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    let stored = gpu
        .space
        .held
        .values()
        .map(|held| held.texture.clone())
        .find(|texture| {
            texture
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        })
        .expect("the image is held as a storage texture");
    assert!(
        gpu.scratch
            .iter()
            .any(|made| matches!(made, Scratch::Texture(old)
            if *old != stored && old.size() == stored.size() && old.format() == stored.format())),
        "the texture the storage image replaced was not handed over"
    );
}

/// The distinct subgroup sizes one 64-invocation workgroup runs with.
fn subgroup_sizes(gpu: &super::Gpu) -> Vec<u32> {
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> sizes: array<u32, 64>;\n\
                 @compute @workgroup_size(64)\n\
                 fn main(@builtin(subgroup_size) size: u32, \
                 @builtin(local_invocation_index) index: u32) { sizes[index] = size; }"
                    .into(),
            ),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let sizes = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage,
        mapped_at_creation: false,
    });
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: sizes.as_entire_binding(),
        }],
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&sizes, 0, &out, 0, 256);
    gpu.queue.submit([encoder.finish()]);
    out.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("the readback"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device");
    let mapped = out
        .slice(..)
        .get_mapped_range()
        .expect("the mapped readback");
    let mut seen: Vec<u32> = mapped
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen
}

/// A launch that only asks an image's size binds it without copying it off the device, even
/// while it stores to that image, and stores where the software dispatch does.
#[test]
fn a_size_query_of_an_image_held_on_the_device_copies_nothing() {
    use switch_core::gpu::engine::compute::EngineCompute;
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{
        image_copy_program, image_size_program, ComputeHarness, IMAGE_COPY_SIZE,
    };

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let block = [width, height, 1];
    type Run<'a> = dyn FnMut(&mut ComputeHarness, &EngineCompute) + 'a;
    let run_both = |h: &mut ComputeHarness, run: &mut Run| {
        let engine = h.image_copy();
        run(h, &engine);
        h.set_program(&image_size_program());
        run(h, &engine);
    };

    let mut software = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    run_both(&mut software, &mut |h, engine| {
        switch_core::gpu::compute::dispatch(engine, &mut h.ctx()).unwrap();
    });
    let expected = software.image();
    // Level 1 of 8x4 is 4x2, and 1/128 of full scale rounds to 2.
    assert_eq!(expected[(2 * width + 4) as usize], 0x0202_0202);

    let mut device = ComputeHarness::new(&image_copy_program(), [1, 1, 1], block, 0);
    run_both(&mut device, &mut |h, engine| {
        assert!(
            gpu.dispatch(engine, &mut h.ctx()).unwrap(),
            "refused: {:?}",
            gpu.compute_refusals
        );
    });
    assert_eq!(gpu.held_copies_made, 0);
    assert_eq!(
        gpu.syncs[4], 0,
        "a size query counted as a read of the held image"
    );
    let mut flushes = 0;
    while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    assert_eq!(device.image(), expected);
}

/// A null image's stand-in is kept like any texture, so launches after the first that sample
/// one find it.
#[test]
fn a_null_texture_is_read_once_across_launches() {
    use switch_core::gpu::renderer::Renderer;
    use switch_core::gpu::testing::{image_copy_program, ComputeHarness, IMAGE_COPY_SIZE};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = IMAGE_COPY_SIZE;
    let mut h = ComputeHarness::new(&image_copy_program(), [1, 1, 1], [width, height, 1], 0);
    let engine = h.image_copy();
    h.null_texture();
    for _ in 0..3 {
        assert!(
            gpu.dispatch(&engine, &mut h.ctx()).unwrap(),
            "refused: {:?}",
            gpu.compute_refusals
        );
    }
    assert_eq!(gpu.texture_misses, 1);
}

/// Warp and quad instructions in a launch run on 32-lane subgroups where the device has them,
/// and write what the software dispatch writes.
#[test]
fn a_compute_launch_shuffles_and_votes_across_its_warps_on_the_device() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{warp_program, ComputeHarness, WARP_THREADS};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    // The adapter may report a range wider than the subgroups this device runs.
    if !gpu.features().contains(wgpu::Features::SUBGROUP) || subgroup_sizes(&gpu) != [32] {
        return;
    }
    gpu.warp_subgroups = true;
    let program = warp_program();
    let (grid, block) = ([1, 1, 1], [WARP_THREADS, 1, 1]);
    let words = WARP_THREADS * 3;
    let mut expected: Vec<u32> = (0..WARP_THREADS).map(|tid| tid ^ 3).collect();
    expected.extend((0..WARP_THREADS).map(|tid| if tid < 32 { u32::MAX } else { 0xf }));
    let signed = [-4.0f32, 2.0, -2.0, -1.0];
    expected.extend((0..WARP_THREADS).map(|tid| signed[tid as usize & 3].to_bits()));

    let mut software = ComputeHarness::new(&program, grid, block, words);
    let engine = software.engine();
    switch_core::gpu::compute::dispatch(&engine, &mut software.ctx()).unwrap();
    assert_eq!(software.output(words as usize), expected);

    let mut device = ComputeHarness::new(&program, grid, block, words);
    let engine = device.engine();
    assert!(
        gpu.dispatch(&engine, &mut device.ctx()).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    let mut flushes = 0;
    while gpu.flush(&mut device.ctx()).unwrap() == Flush::Pending {
        flushes += 1;
        assert!(flushes < 100, "the readback never landed");
    }
    assert_eq!(device.output(words as usize), expected);
}

/// What a compute launch only reads stays on the device for the next launch, until the guest
/// writes over it.
#[test]
fn a_compute_input_is_uploaded_again_only_after_the_guest_writes_it() {
    use switch_core::gpu::renderer::{Flush, Renderer};
    use switch_core::gpu::testing::{copy_program, write_launch, ComputeHarness};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    const WORDS: u32 = 4;
    let mut h = ComputeHarness::new(&copy_program(), [1, 1, 1], [WORDS, 1, 1], WORDS);
    let (base, output) = (h.base, h.output_addr());
    // Beside the bank, which is laid out once, so only the guest's own stores write its page.
    let input = base + 0x3800;
    let engine = write_launch(
        &mut h.ctx(),
        base,
        &copy_program(),
        [1, 1, 1],
        [WORDS, 1, 1],
        &[
            (0x40, output as u32),
            (0x44, (output >> 32) as u32),
            (0x48, WORDS * 4),
            (0x60, input as u32),
            (0x64, (input >> 32) as u32),
            (0x68, WORDS * 4),
        ],
    );
    let run = |h: &mut ComputeHarness, gpu: &mut crate::Gpu, words: [u32; 4]| {
        let mut ctx = h.ctx();
        for (i, word) in words.iter().enumerate() {
            if ctx.read_u32(input + i as u64 * 4).unwrap() != *word {
                ctx.write_u32(input + i as u64 * 4, *word).unwrap();
            }
        }
        assert!(
            gpu.dispatch(&engine, &mut ctx).unwrap(),
            "refused: {:?}",
            gpu.compute_refusals
        );
        let mut flushes = 0;
        while gpu.flush(&mut ctx).unwrap() == Flush::Pending {
            flushes += 1;
            assert!(flushes < 100, "the readback never landed");
        }
        assert_eq!(h.output(WORDS as usize), words);
    };

    run(&mut h, &mut gpu, [1, 2, 3, 4]);
    run(&mut h, &mut gpu, [1, 2, 3, 4]);
    assert_eq!((gpu.inputs_uploaded, gpu.inputs_reused), (1, 1));
    run(&mut h, &mut gpu, [5, 6, 7, 8]);
    assert_eq!((gpu.inputs_uploaded, gpu.inputs_reused), (2, 1));
}

/// A draw reads vertices a compute launch wrote on the device from where the launch left them,
/// before guest memory has them, and draws what it would have drawn from guest memory.
#[test]
fn a_draw_reads_vertices_a_compute_launch_wrote_on_the_device() {
    use switch_core::gpu::renderer::Renderer;
    use switch_core::gpu::testing::{copy_program, write_launch, Harness};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let colour = [0.25, 0.5, 1.0, 1.0];

    let mut direct = Harness::new();
    direct.as_id = 1;
    direct.triangle(colour);
    direct.draw_with(&mut gpu).unwrap();
    direct.flush_with(&mut gpu);
    let expected = direct.target();
    assert!(
        expected.iter().any(|&texel| texel != 0),
        "the reference drew"
    );

    let mut h = Harness::new();
    h.as_id = 2;
    h.triangle(colour);
    let (vertices, staged, launch) = (h.base + 0x400, h.base + 0x1000, h.base + 0x800);
    const WORDS: u32 = 3 * 8;
    let mut ctx = h.ctx();
    for word in 0..u64::from(WORDS) {
        let value = ctx.read_u32(vertices + word * 4).unwrap();
        ctx.write_u32(staged + word * 4, value).unwrap();
        ctx.write_u32(vertices + word * 4, 0).unwrap();
    }
    let engine = write_launch(
        &mut ctx,
        launch,
        &copy_program(),
        [1, 1, 1],
        [WORDS, 1, 1],
        &[
            (0x40, vertices as u32),
            (0x44, (vertices >> 32) as u32),
            (0x48, WORDS * 4),
            (0x60, staged as u32),
            (0x64, (staged >> 32) as u32),
            (0x68, WORDS * 4),
        ],
    );
    assert!(
        gpu.dispatch(&engine, &mut ctx).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    assert_eq!(
        ctx.read_u32(vertices).unwrap(),
        0,
        "the copy is still on the device"
    );

    h.draw_with(&mut gpu).unwrap();
    assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
    h.flush_with(&mut gpu);
    assert_eq!(h.target(), expected);
    let ctx = h.ctx();
    assert_eq!(
        ctx.read_u32(vertices).unwrap(),
        ctx.read_u32(staged).unwrap(),
        "the flush put the copy back"
    );
}

/// A draw reads constants a compute launch wrote on the device from where the launch left
/// them, before guest memory has them.
#[test]
fn a_draw_reads_constants_a_compute_launch_wrote_on_the_device() {
    use switch_core::gpu::renderer::Renderer;
    use switch_core::gpu::testing::{alu_fragment_shader, copy_program, write_launch, Harness};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    // mov rN, c0[N * 4], so the colour is the bank's first four words.
    let mov_c0 = |dst: u64| 0x4c98u64 << 48 | 0xf << 39 | dst << 20 | 7 << 16 | dst;
    let shader = || alu_fragment_shader(&[], &[mov_c0(0), mov_c0(1), mov_c0(2), mov_c0(3)]);
    let colour = [0.25f32, 0.5, 1.0, 1.0].map(f32::to_bits);
    const WORDS: u32 = 4;
    let bind_bank = |h: &mut Harness, constants: u64| {
        let mut ctx = switch_core::gpu::exec::ExecCtx {
            mem: &h.mem,
            vmm: &h.vmm,
            as_id: h.as_id,
            host1x: &mut h.host1x,
            stats: &mut h.stats,
            trace: false,
        };
        for (method, arg) in [
            (0x8E0, 0x100),
            (0x8E1, (constants >> 32) as u32),
            (0x8E2, constants as u32),
            // BindGroup[Fragment]: bank 0, valid.
            (0x900 + 4 * 8 + 4, 1),
        ] {
            h.engine.write(method, arg, true, &mut ctx).unwrap();
        }
    };

    let mut direct = Harness::with_fragment_shader(shader());
    direct.as_id = 1;
    direct.triangle([0.0; 4]);
    let constants = direct.base + 0x1500;
    let mut ctx = direct.ctx();
    for (i, word) in colour.iter().enumerate() {
        ctx.write_u32(constants + i as u64 * 4, *word).unwrap();
    }
    bind_bank(&mut direct, constants);
    direct.draw_with(&mut gpu).unwrap();
    direct.flush_with(&mut gpu);
    let expected = direct.target();
    assert!(
        expected.iter().any(|&texel| texel != 0),
        "the reference drew"
    );

    let mut h = Harness::with_fragment_shader(shader());
    h.as_id = 2;
    h.triangle([0.0; 4]);
    let (constants, staged, launch) = (h.base + 0x1500, h.base + 0x1000, h.base + 0x800);
    let mut ctx = h.ctx();
    for (i, word) in colour.iter().enumerate() {
        ctx.write_u32(staged + i as u64 * 4, *word).unwrap();
    }
    let engine = write_launch(
        &mut ctx,
        launch,
        &copy_program(),
        [1, 1, 1],
        [WORDS, 1, 1],
        &[
            (0x40, constants as u32),
            (0x44, (constants >> 32) as u32),
            (0x48, WORDS * 4),
            (0x60, staged as u32),
            (0x64, (staged >> 32) as u32),
            (0x68, WORDS * 4),
        ],
    );
    assert!(
        gpu.dispatch(&engine, &mut ctx).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    assert_eq!(
        ctx.read_u32(constants).unwrap(),
        0,
        "the copy is still on the device"
    );
    bind_bank(&mut h, constants);

    h.draw_with(&mut gpu).unwrap();
    assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
    h.flush_with(&mut gpu);
    assert_eq!(h.target(), expected);
}

/// An indexed constant read past the end of its bank is zero on the device, though the next
/// bank sits right behind it in the buffer the device reads.
#[test]
fn an_indexed_constant_read_past_its_bank_is_zero() {
    use switch_core::gpu::testing::{alu_fragment_shader, Harness};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    const ONE: u32 = 0x3f80_0000;
    const BANK_BYTES: u32 = 0x100;
    // ldc.b32 dst, c<bank>[idx + offset].
    let ldc = |dst: u64, bank: u64, idx: u64, offset: u64| {
        0xef90u64 << 48 | 4 << 48 | bank << 36 | (offset & 0xffff) << 20 | 7 << 16 | idx << 8 | dst
    };
    // Alpha comes from bank 2, so the device has that bank's words in its buffer.
    let shader = alu_fragment_shader(
        &[(4, 0), (5, BANK_BYTES)],
        &[
            ldc(0, 0, 4, 0),
            ldc(1, 0, 5, 0),
            ldc(2, 0, 5, 0x7c),
            ldc(3, 2, 4, 0),
        ],
    );
    let mut h = Harness::with_fragment_shader(shader);
    h.triangle([0.0; 4]);
    for (bank, at) in [(0u32, 0x2800u64), (2, 0x2a00)] {
        let address = h.base + at;
        let mut ctx = h.ctx();
        for word in 0..u64::from(BANK_BYTES) / 4 {
            ctx.write_u32(address + word * 4, ONE).unwrap();
        }
        let mut ctx = switch_core::gpu::exec::ExecCtx {
            mem: &h.mem,
            vmm: &h.vmm,
            as_id: h.as_id,
            host1x: &mut h.host1x,
            stats: &mut h.stats,
            trace: false,
        };
        for (method, arg) in [
            (0x8E0, BANK_BYTES),
            (0x8E1, (address >> 32) as u32),
            (0x8E2, address as u32),
            // BindGroup[Fragment]: this bank, valid.
            (0x900 + 4 * 8 + 4, 1 | bank << 4),
        ] {
            h.engine.write(method, arg, true, &mut ctx).unwrap();
        }
    }

    h.draw_with(&mut gpu).unwrap();
    assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
    h.flush_with(&mut gpu);
    assert_eq!(
        h.texel(1, 1),
        0xff00_00ff,
        "red read inside the bank, green and blue past it"
    );
}

/// Timed per pass, two draws into one target share a pass and a row; timed per draw, each
/// draw is a pass of its own, named by its shaders.
#[test]
fn device_time_counts_passes_or_draws() {
    use switch_core::gpu::renderer::{DeviceTiming, Renderer};
    use switch_core::gpu::testing::Harness;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    if !gpu
        .device_features
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        eprintln!("[gpu] skipped: the adapter has no timestamp queries");
        return;
    }
    for (timing, passes) in [(DeviceTiming::Passes, 1), (DeviceTiming::Draws, 2)] {
        gpu.set_device_timing(timing);
        let mut h = Harness::new();
        h.triangle([0.0; 4]);
        h.draw_with(&mut gpu).unwrap();
        h.draw_with(&mut gpu).unwrap();
        assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
        h.flush_with(&mut gpu);
        gpu.submit_batch();
        gpu.wait_for_times();

        let (rows, untimed) = gpu.device_time_rows();
        let draws: Vec<_> = rows
            .iter()
            .filter(|(what, _)| what.starts_with("draws "))
            .collect();
        assert_eq!(untimed, 0, "{timing:?}");
        assert_eq!(draws.len(), 1, "{timing:?}: {rows:?}");
        let (what, total) = draws[0];
        assert_eq!(total.passes, passes, "{timing:?}: {what}");
        assert!(
            what.contains(": vs ") && what.contains(" fs "),
            "{timing:?}: {what}"
        );
        assert!(total.max_ns <= total.ns, "{timing:?}: {total:?}");
    }
    gpu.set_device_timing(DeviceTiming::Off);
    assert!(gpu.device_time_rows().0.is_empty());
}

/// Past the vertex buffer limit, an array a compute launch wrote keeps a buffer of its own
/// for the copy and the others are folded.
#[test]
fn a_draw_folds_the_arrays_beside_one_a_compute_launch_wrote() {
    use switch_core::gpu::renderer::Renderer;
    use switch_core::gpu::testing::{block, copy_program, write_launch, Harness};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    // gl_Position = aPosition; vColor = vec4(aColor.rgb, aExtra.x).
    let mut shader = block(
        (0xfc20070f, 0x081f8441),
        (0x0807ff00, 0xefd9ff80), // ld b128 $r0 a[0x80] 0x0  (aPosition)
        (0x0707ff00, 0xeff1ff80), // st b128 a[0x70] $r0 0x0  (gl_Position)
        (0x0907ff00, 0xefd9ff80), // ld b128 $r0 a[0x90] 0x0  (aColor)
    );
    shader.extend(block(
        (0xfc2207e1, 0x001f8c40),
        (0x0a07ff03, 0xefd87f80), // ld b32 $r3 a[0xa0] 0x0  (aExtra.x)
        (0x0807ff00, 0xeff1ff80), // st b128 a[0x80] $r0 0x0  (vColor)
        (0x0007000f, 0xe3000000), // exit
    ));
    let alphas = [0.2f32, 0.6, 0.4];
    // Array 2, 16 bytes a vertex, holding each vertex's alpha.
    let set_up = |h: &mut Harness| {
        h.set_vertex_shader(&shader);
        h.triangle([0.25, 0.5, 1.0, 1.0]);
        let extra = h.base + 0x600;
        h.engine.regs.set(0x458 + 2, 2 | (0x01 << 21) | (7 << 27));
        h.engine.regs.set(0x708, 16 | (1 << 12));
        h.engine.regs.set(0x709, (extra >> 32) as u32);
        h.engine.regs.set(0x70a, extra as u32);
        h.engine.regs.set(0x7C4, (extra >> 32) as u32);
        h.engine.regs.set(0x7C5, extra as u32 + 3 * 16 - 1);
        extra
    };

    // Read from guest memory, unfolded: interpolation rounds as the folded draw's does.
    let mut direct = Harness::new();
    direct.as_id = 2;
    let extra = set_up(&mut direct);
    let mut ctx = direct.ctx();
    for (vertex, alpha) in alphas.iter().enumerate() {
        ctx.write_u32(extra + vertex as u64 * 16, alpha.to_bits())
            .unwrap();
    }
    direct.draw_with(&mut gpu).unwrap();
    direct.flush_with(&mut gpu);
    let expected = direct.target();
    assert!(
        expected
            .iter()
            .any(|&texel| texel >> 24 != 0 && texel >> 24 != 0xff),
        "the reference drew the alphas: {expected:x?}"
    );

    let mut h = Harness::new();
    h.as_id = 3;
    let extra = set_up(&mut h);
    let (staged, launch) = (h.base + 0x1000, h.base + 0x800);
    const WORDS: u32 = 3 * 4;
    let mut ctx = h.ctx();
    for (vertex, alpha) in alphas.iter().enumerate() {
        ctx.write_u32(staged + vertex as u64 * 16, alpha.to_bits())
            .unwrap();
    }
    let engine = write_launch(
        &mut ctx,
        launch,
        &copy_program(),
        [1, 1, 1],
        [WORDS, 1, 1],
        &[
            (0x40, extra as u32),
            (0x44, (extra >> 32) as u32),
            (0x48, WORDS * 4),
            (0x60, staged as u32),
            (0x64, (staged >> 32) as u32),
            (0x68, WORDS * 4),
        ],
    );
    assert!(
        gpu.dispatch(&engine, &mut ctx).unwrap(),
        "refused: {:?}",
        gpu.compute_refusals
    );
    assert_eq!(
        ctx.read_u32(extra).unwrap(),
        0,
        "the copy is still on the device"
    );
    // Three arrays against a limit of two: the copied one alone, the other two folded.
    gpu.vertex_buffer_limit = 2;
    h.draw_with(&mut gpu).unwrap();
    assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
    h.flush_with(&mut gpu);
    assert_eq!(h.target(), expected);
}

/// A draw lands in the layer `SET_RT_LAYER` picks on the device too, and layer 0 keeps its
/// texels: Minecraft's title panorama is six such draws into a cube map's faces.
#[test]
fn a_draw_into_the_layer_set_rt_layer_picks_lands_there_on_the_device() {
    use switch_core::gpu::exec::ExecCtx;
    use switch_core::gpu::testing::Harness;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = Harness::new();
    let mut ctx = ExecCtx {
        mem: &h.mem,
        vmm: &h.vmm,
        as_id: h.as_id,
        host1x: &mut h.host1x,
        stats: &mut h.stats,
        trace: false,
    };
    // Two layers 0x2000 bytes apart; the array pitch counts words.
    for (method, arg) in [(0x206, 2), (0x207, 0x800), (0x573, 1)] {
        h.engine.write(method, arg, true, &mut ctx).unwrap();
    }
    // Off the halfway points between unorm steps, where devices may round either way.
    let colour = [0.2, 0.6, 1.0, 1.0];
    h.triangle(colour);
    h.draw_with(&mut gpu).unwrap();
    assert!(gpu.reasons.is_empty(), "fell back: {:?}", gpu.reasons);
    h.flush_with(&mut gpu);
    let rt = h.engine.render_target(0).unwrap().unwrap();
    let expected = rt.format.encode(colour).unwrap() as u32;
    let ctx = h.ctx();
    assert_eq!(ctx.read_u32(rt.addr + 0x2000).unwrap(), expected);
    assert_eq!(ctx.read_u32(rt.addr).unwrap(), 0);
}

/// Constants placed before a submit are found again after it by their bytes, whichever stage
/// places them, and bytes that differ anywhere are placed anew.
#[test]
fn constants_placed_before_a_submit_are_found_again_after_it() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let first = vec![1u8; 512];
    let second = vec![2u8; 512];
    let (at_first, _) = gpu.stage_constants(0, &first).expect("placed");
    let (at_second, _) = gpu.stage_constants(1, &second).expect("placed");
    assert_ne!(at_first, at_second);
    // Recorded and submitted, as a draw would be.
    gpu.constants_unrecorded = false;
    gpu.submit_batch();
    assert_eq!(
        gpu.stage_constants(0, &second).map(|(at, _)| at),
        Some(at_second)
    );
    assert_eq!(
        gpu.stage_constants(1, &first).map(|(at, _)| at),
        Some(at_first)
    );
    let mut changed = first.clone();
    changed[300] = 9;
    let (at_changed, _) = gpu.stage_constants(0, &changed).expect("placed");
    assert!(at_changed != at_first && at_changed != at_second);
}

/// Copies made for unsubmitted draws are submitted and freed once they pass
/// `MAX_SCRATCH_BYTES`, rather than piling up until the frame ends.
#[test]
fn unsubmitted_draws_are_submitted_once_their_scratch_passes_the_budget() {
    use crate::readback::Scratch;

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let array = |gpu: &crate::Gpu| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scratch test"),
            size: wgpu::Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 8,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let copy = Scratch::Texture(array(&gpu));
    assert_eq!(copy.bytes(), 32 << 20);
    gpu.batch = Some(
        gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default()),
    );
    gpu.scratch.push(copy);
    gpu.release_scratch();
    assert!(gpu.batch.is_some(), "submitted under the budget");
    assert_eq!(gpu.scratch.len(), 1);
    while gpu.scratch.len() as u64 * (32 << 20) < crate::MAX_SCRATCH_BYTES {
        gpu.scratch.push(Scratch::Texture(array(&gpu)));
    }
    gpu.release_scratch();
    assert!(gpu.batch.is_none());
    assert!(gpu.scratch.is_empty());
}

/// A mipmapped block-compressed texture counts each level in whole blocks.
#[test]
fn scratch_counts_every_level_of_a_compressed_texture() {
    let Some((_serial, gpu)) = device() else {
        return;
    };
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    {
        return;
    }
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scratch test"),
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 4,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    // 8x8, 4x4, 2x2 and 1x1 are four blocks, then one each, at 8 bytes a block.
    assert_eq!(
        crate::readback::Scratch::Texture(texture).bytes(),
        (4 + 1 + 1 + 1) * 8
    );
}

/// A draw that renders to the surface it samples keeps sampling its copy as it was until the
/// guest writes a texture barrier or a frame is presented; any other draw takes the update.
#[test]
fn a_draw_sampling_its_own_target_updates_the_copy_only_past_a_barrier() {
    use crate::HeldLayer;
    use switch_core::gpu::engine::threed::ShaderStage;
    use switch_core::gpu::pipeline::Format;
    use switch_core::gpu::surface::Layout;
    use switch_core::gpu::texture::{Sampler, SwizzleSource, TextureSlot, Wrap};
    use switch_core::gpu::upload::{TextureKey, TextureUpload};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    const SIDE: u32 = 4;
    let format = Format::Rgba8Unorm;
    let upload = TextureUpload {
        stage: ShaderStage::Fragment,
        slot: TextureSlot::Bound(0),
        handle: 0,
        format,
        width: SIDE,
        height: SIDE,
        layers: 1,
        row_bytes: SIDE * 4,
        rows: SIDE,
        texels: switch_core::gpu::upload::Texels::Read(std::sync::Arc::from(vec![
            0u8;
            (SIDE * SIDE * 4)
                as usize
        ])),
        key: TextureKey {
            addr: 0x1_0000,
            width: SIDE,
            height: SIDE,
            layers: 1,
            layer_stride: SIDE * SIDE * 4,
            row_bytes: SIDE * 4,
            rows: SIDE,
            block_depth_gobs: 1,
            layout: Layout::Pitch { pitch: SIDE * 4 },
            format,
            srgb: false,
        },
        source_len: u64::from(SIDE * SIDE * 4),
        swizzle: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
        sampler: Sampler {
            wrap_u: Wrap::ClampToEdge,
            wrap_v: Wrap::ClampToEdge,
            wrap_w: Wrap::ClampToEdge,
            mag_linear: false,
            min_linear: false,
            compare: None,
        },
        shrink: 0,
    };
    let surface = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("held target"),
        size: wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let held = [(0, HeldLayer::Colour(surface))];
    let view = wgpu::TextureViewDimension::D2;
    let copy = gpu
        .texture_over_held(&upload, view, &held, &[1], false, true)
        .expect("the first copy");

    // Drawn again with no barrier since: the copy stays as it was.
    let kept = gpu
        .texture_over_held(&upload, view, &held, &[2], false, true)
        .expect("the kept copy");
    assert!(kept == copy, "a new copy was made");
    assert_eq!((gpu.held_copies_kept, gpu.held_copies_updated), (1, 0));

    // A draw that does not render to it sees every draw before it.
    gpu.texture_over_held(&upload, view, &held, &[2], false, false)
        .expect("the update");
    assert_eq!((gpu.held_copies_kept, gpu.held_copies_updated), (1, 1));

    // Past a barrier, so does one that does.
    gpu.texture_barriers += 1;
    gpu.texture_over_held(&upload, view, &held, &[3], false, true)
        .expect("the update past a barrier");
    assert_eq!((gpu.held_copies_kept, gpu.held_copies_updated), (1, 2));
    gpu.texture_over_held(&upload, view, &held, &[4], false, true)
        .expect("the copy kept again");
    assert_eq!((gpu.held_copies_kept, gpu.held_copies_updated), (2, 2));

    // And past a presented frame.
    gpu.frames_presented += 1;
    gpu.texture_over_held(&upload, view, &held, &[5], false, true)
        .expect("the update past a frame");
    assert_eq!((gpu.held_copies_kept, gpu.held_copies_updated), (2, 3));
}

/// A kept copy of a held array, one of whose layers was drawn again, takes only that layer, after
/// the work recorded before it, which still reads the layer as it was.
#[test]
fn a_kept_copy_of_a_held_array_takes_only_the_layer_written_since() {
    use crate::HeldLayer;
    use switch_core::gpu::engine::threed::ShaderStage;
    use switch_core::gpu::pipeline::Format;
    use switch_core::gpu::surface::Layout;
    use switch_core::gpu::texture::{Sampler, SwizzleSource, TextureSlot, Wrap};
    use switch_core::gpu::upload::{TextureKey, TextureUpload};

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    const SIDE: u32 = 4;
    // A texture-buffer copy pads each row to this.
    const ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let format = Format::Rgba8Unorm;
    let upload = TextureUpload {
        stage: ShaderStage::Fragment,
        slot: TextureSlot::Bound(0),
        handle: 0,
        format,
        width: SIDE,
        height: SIDE,
        layers: 2,
        row_bytes: SIDE * 4,
        rows: SIDE,
        texels: switch_core::gpu::upload::Texels::Read(std::sync::Arc::from(vec![
            0u8;
            (SIDE * SIDE * 4 * 2)
                as usize
        ])),
        key: TextureKey {
            addr: 0x1_0000,
            width: SIDE,
            height: SIDE,
            layers: 2,
            layer_stride: SIDE * SIDE * 4,
            row_bytes: SIDE * 4,
            rows: SIDE,
            block_depth_gobs: 1,
            layout: Layout::Pitch { pitch: SIDE * 4 },
            format,
            srgb: false,
        },
        source_len: u64::from(SIDE * SIDE * 4 * 2),
        swizzle: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
        sampler: Sampler {
            wrap_u: Wrap::ClampToEdge,
            wrap_v: Wrap::ClampToEdge,
            wrap_w: Wrap::ClampToEdge,
            mag_linear: false,
            min_linear: false,
            compare: None,
        },
        shrink: 0,
    };
    let extent = wgpu::Extent3d {
        width: SIDE,
        height: SIDE,
        depth_or_array_layers: 1,
    };
    let fill = |gpu: &crate::Gpu, texture: &wgpu::Texture, byte: u8| {
        gpu.queue.write_texture(
            texture.as_image_copy(),
            &vec![byte; (SIDE * SIDE * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIDE * 4),
                rows_per_image: Some(SIDE),
            },
            extent,
        );
    };
    let surface = |gpu: &crate::Gpu, byte: u8| {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("held layer"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        fill(gpu, &texture, byte);
        texture
    };
    let readback = |gpu: &crate::Gpu| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer readback"),
            size: u64::from(ROW),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    // Read a layer's first texel as a draw sampling the copy would: the copy is never a copy source.
    let copy_layer = |gpu: &crate::Gpu,
                      encoder: &mut wgpu::CommandEncoder,
                      texture: &wgpu::Texture,
                      layer: u32,
                      buffer: &wgpu::Buffer| {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("layer texel"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "@group(0) @binding(0) var image: texture_2d_array<f32>;
                         @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
                           let corners = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                           return vec4(corners[i], 0.0, 1.0);
                         }}
                         @fragment fn fs() -> @location(0) vec4<f32> {{
                           return textureLoad(image, vec2(0i, 0i), {layer}i, 0i);
                         }}"
                    )
                    .into(),
                ),
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("layer texel"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer texel"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let pixel = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer texel"),
            size: pixel,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("layer texel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: Some(1),
                },
            },
            pixel,
        );
    };
    let first_texel = |gpu: &crate::Gpu, buffer: &wgpu::Buffer| {
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("the readback"));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device");
        let texel = buffer
            .slice(..)
            .get_mapped_range()
            .expect("the mapped readback")[..4]
            .to_vec();
        texel
    };

    let (below, drawn) = (surface(&gpu, 0x11), surface(&gpu, 0x22));
    let held = [
        (0, HeldLayer::Colour(below.clone())),
        (1, HeldLayer::Colour(drawn.clone())),
    ];
    let view = wgpu::TextureViewDimension::D2Array;
    let copy = gpu
        .texture_over_held(&upload, view, &held, &[1, 1], false, false)
        .expect("the first copy");
    let made = gpu.held_copies_made;

    // Work recorded before the layer is drawn again, as an earlier draw sampling the copy is.
    let mut earlier = gpu.take_batch();
    let before = readback(&gpu);
    copy_layer(&gpu, &mut earlier, &copy, 1, &before);
    gpu.restore_batch(earlier);
    // Drawn again by work recorded after it, as the backend's draws into a held surface are.
    let redraw = surface(&gpu, 0x33);
    let mut encoder = gpu.take_batch();
    encoder.copy_texture_to_texture(redraw.as_image_copy(), drawn.as_image_copy(), extent);
    gpu.restore_batch(encoder);
    let again = gpu
        .texture_over_held(&upload, view, &held, &[1, 2], false, false)
        .expect("the update");
    assert!(again == copy, "a new copy was made for one written layer");
    assert_eq!(gpu.held_copies_made, made);
    assert_eq!(gpu.held_copies_updated, 1);
    // The update is recorded after the earlier work, so `before` below still sees the old layer.
    gpu.submit_batch();

    let after = [readback(&gpu), readback(&gpu)];
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (layer, buffer) in after.iter().enumerate() {
        copy_layer(&gpu, &mut encoder, &copy, layer as u32, buffer);
    }
    gpu.queue.submit([encoder.finish()]);
    assert_eq!(
        first_texel(&gpu, &before),
        [0x22; 4],
        "earlier work saw the update"
    );
    assert_eq!(
        first_texel(&gpu, &after[0]),
        [0x11; 4],
        "the unwritten layer changed"
    );
    assert_eq!(
        first_texel(&gpu, &after[1]),
        [0x33; 4],
        "the written layer is stale"
    );

    // The same sources again are the same copy, untouched.
    let reused = gpu.held_copies_reused;
    gpu.texture_over_held(&upload, view, &held, &[1, 2], false, false)
        .expect("the reuse");
    assert_eq!(gpu.held_copies_reused, reused + 1);
}

/// The WGSL forms of `i2f.f64` and `f2f.f32.f64` give the interpreter's bits for every edge
/// of the conversions: rounding ties, carries into the exponent, single denormals, overflow,
/// infinities and NaNs, and a spread of arbitrary words.
#[test]
fn the_double_conversion_helpers_match_the_interpreter_on_the_device() {
    use switch_core::gpu::shader::interp::{f64_to_f32_bits, int_to_f64_bits};
    use switch_core::gpu::shader::wgsl::helper_source;
    let Some((_serial, gpu)) = device() else {
        return;
    };
    let mut inputs: Vec<u64> = vec![
        0,
        1,
        2,
        (1 << 52) - 1,
        1 << 52,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        (1 << 53) + 2,
        (1 << 53) + 3,
        (1 << 54) + 2,
        (1 << 54) + 6,
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) + 1,
        i64::MAX as u64,
        (-1i64) as u64,
        (-(1i64 << 53) - 1) as u64,
    ];
    let doubles = [
        0.0f64,
        -0.0,
        1.0,
        -1.5,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * (1.0 + f64::EPSILON),
        // Half a single's top ulp past its largest: a tie that rounds to infinity.
        f64::from(f32::MAX) + 2f64.powi(103),
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(1)),
        f64::from(f32::from_bits(1)) * 0.5,
        f64::from(f32::from_bits(1)) * 0.5000001,
        f64::from(f32::from_bits(1)) * 1.5,
        f64::from(f32::from_bits(3)) * 0.5,
        f64::from(f32::from_bits(0x007f_ffff)) * (1.0 + 1e-9),
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0 + f64::from(f32::EPSILON) * 0.5,
        1.0 + f64::from(f32::EPSILON) * 1.5,
        1.0 + f64::from(f32::EPSILON) * 0.5 * (1.0 + f64::EPSILON),
        16_777_217.0,
        16_777_219.0,
    ];
    inputs.extend(doubles.iter().map(|d| d.to_bits()));
    inputs.extend(doubles.iter().map(|d| (-d).to_bits()));
    // Every single's exponent band, each with a tie and a near-tie below it.
    for exponent in 0..2048u64 {
        let base = exponent << 52;
        inputs.extend([
            base,
            base | 1 << 28,
            base | (1 << 29) | (1 << 28),
            base | 0x000f_ffff_ffff_ffff,
        ]);
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..8192 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        inputs.push(state);
        inputs.push(state >> (state % 64));
    }

    let source = format!(
        "{}\n{}\n{}\n\
         @group(0) @binding(0) var<storage, read> inputs: array<vec2<u32>>;\n\
         @group(0) @binding(1) var<storage, read_write> doubles: array<vec4<u32>>;\n\
         @group(0) @binding(2) var<storage, read_write> singles: array<vec2<u32>>;\n\
         @compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
           let i = id.x;\n\
           if (i >= arrayLength(&inputs)) {{ return; }}\n\
           let v = inputs[i];\n\
           let u = i2dBits(v.x, v.y, false);\n\
           let s = i2dBits(v.x, v.y, true);\n\
           doubles[i] = vec4<u32>(u.x, u.y, s.x, s.y);\n\
           singles[i] = vec2<u32>(d2fBits(v.x, v.y, false), d2fBits(v.x, v.y, true));\n\
         }}\n",
        helper_source("u64").expect("the u64 helpers"),
        helper_source("i2d").expect("the i2d helper"),
        helper_source("d2f").expect("the d2f helper"),
    );
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let count = inputs.len() as u64;
    let words: Vec<u8> = inputs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: count * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&input, 0, &words);
    let storage = |size: u64| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let (doubles_out, singles_out) = (storage(count * 16), storage(count * 8));
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: count * 24,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: doubles_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: singles_out.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&doubles_out, 0, &readback, 0, count * 16);
    encoder.copy_buffer_to_buffer(&singles_out, 0, &readback, count * 16, count * 8);
    gpu.queue.submit([encoder.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("the readback"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("the mapped readback");
    let word = |at: u64| {
        let at = at as usize * 4;
        u32::from_le_bytes(mapped[at..at + 4].try_into().unwrap())
    };
    let wide = |at: u64| u64::from(word(at)) | u64::from(word(at + 1)) << 32;
    for (i, &v) in inputs.iter().enumerate() {
        let i = i as u64;
        assert_eq!(wide(i * 4), int_to_f64_bits(v, false), "u64 {v:#x}");
        assert_eq!(wide(i * 4 + 2), int_to_f64_bits(v, true), "s64 {v:#x}");
        let singles = count * 4 + i * 2;
        assert_eq!(word(singles), f64_to_f32_bits(v, false), "f64 {v:#x}");
        assert_eq!(
            word(singles + 1),
            f64_to_f32_bits(v, true),
            "f64 {v:#x}, ftz"
        );
    }
}
