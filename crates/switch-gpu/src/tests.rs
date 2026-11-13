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
    layout.uint_targets = vec![0];
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

/// The `.fmz`, `.cc`, `.x`, `.brev` and `lop3` predicate forms are valid WGSL.
#[test]
fn condition_code_extended_and_fmz_forms_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{
        BoolOp, FCmp, FMod, FmulScale, ICmp, Instruction, LogicOp, LopTest, Operand, Pred,
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
        layout: Layout::BlockLinear { block_height_gobs },
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
            attributes: vec![attribute(wgpu::VertexFormat::Float32x3, 0, 0)],
            stride: 16,
            step: wgpu::VertexStepMode::Vertex,
        },
        // Two vertices of a byte pair at offset 2 of a 4-byte stride.
        Feed {
            bytes: Cow::Owned(vec![0, 0, 0xa0, 0xa1, 0, 0, 0xb0, 0xb1]),
            attributes: vec![attribute(wgpu::VertexFormat::Unorm8x2, 2, 1)],
            stride: 4,
            step: wgpu::VertexStepMode::Vertex,
        },
        // One instance's element.
        Feed {
            bytes: Cow::Owned(vec![0xc0, 0xc1, 0xc2, 0xc3]),
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
    )
    .expect("both fit a stride");
    assert_eq!(split.len(), 2);
    assert_eq!(split[0].0.as_ref(), &bytes[..]);
    assert_eq!(split[0].1, [attribute(0, 0)]);
    assert_eq!(split[1].0.len(), bytes.len(), "a draw must still fit it");
    assert_eq!(split[1].0[0], 32, "the first element's bytes");
    assert_eq!(split[1].0[47], 0, "past the array reads zero");
    assert_eq!(split[1].1, [attribute(0, 1)]);
}

#[test]
fn an_attribute_wider_than_its_stride_is_refused_before_the_device_sees_it() {
    let wide = wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 4,
        shader_location: 0,
    };
    let refused = super::within_stride(std::borrow::Cow::Borrowed(&[0u8; 16][..]), vec![wide], 8);
    assert!(refused.is_err());
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
