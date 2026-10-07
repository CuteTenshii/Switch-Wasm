//! Backend tests.

use switch_core::gpu::testing::Harness;

/// The device, or `None` with a notice; with `REQUIRE_GPU` set, a panic.
pub(crate) fn device() -> Option<super::Gpu> {
    match super::Gpu::open() {
        Ok(gpu) => Some(gpu),
        Err(why) if std::env::var_os("REQUIRE_GPU").is_some_and(|v| !v.is_empty()) => {
            panic!("REQUIRE_GPU is set and there is no device: {why}")
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
    let Some(gpu) = device() else { return };
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

    let Some(gpu) = device() else { return };
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

/// Tomodachi Life's cube-array `tex` forms and its `vmnmx` translate to valid WGSL.
#[test]
fn tomodachi_lifes_cube_array_and_video_min_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some(gpu) = device() else { return };
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
fn a_lost_device_is_reported_rather_than_failing_the_flush() {
    use switch_core::gpu::renderer::Renderer;
    let Some(mut gpu) = device() else { return };
    let mut h = Harness::new();
    h.triangle([1.0, 0.0, 1.0, 1.0]);
    h.draw_with(&mut gpu).expect("the draw");
    *gpu.lost.lock().unwrap() = Some("Out of memory".into());
    // `flush_with` panics on an error.
    h.flush_with(&mut gpu);
    h.flush_with(&mut gpu);
    let json = gpu.report_json();
    assert!(json.contains("\"gaveUp\":true"), "{json}");
    assert!(json.contains("Out of memory"), "{json}");
}

/// A rejection the backend never asks about still reaches the report.
#[test]
fn a_rejection_nothing_asked_about_is_still_counted_and_reported() {
    let Some(gpu) = device() else { return };
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

    use switch_core::gpu::renderer::Renderer;
    let json = gpu.report_json();
    assert!(
        json.contains(&format!("\"deviceErrorCount\":{count}")),
        "the count is missing from the report: {json}"
    );
    assert!(
        json.contains("\"deviceErrors\":[\""),
        "the message is missing from the report: {json}"
    );
}

/// Nintendo Switch Sports' `txq` and `tld4` translate to a module the device accepts.
#[test]
fn nintendo_switch_sports_txq_and_tld4_are_wgsl_naga_accepts() {
    use super::{wgpu, wgsl, Compiled, Layout, Stage};
    use switch_core::gpu::shader::isa::{self, Instruction, Pred};
    use switch_core::gpu::shader::{Op, Program};

    let Some(gpu) = device() else { return };
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

    let Some(gpu) = device() else { return };
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
    let Some(mut gpu) = device() else { return };
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

/// Fallback reasons are escaped so `JSON.parse` accepts the report.
#[test]
fn a_reason_with_json_punctuation_in_it_stays_one_string() {
    assert_eq!(super::json_string("plain"), "\"plain\"");
    assert_eq!(
        super::json_string(r#"no WGSL form for Ldg { at: 3 } "x" \ y"#),
        r#""no WGSL form for Ldg { at: 3 } \"x\" \\ y""#
    );
    assert_eq!(
        super::json_string("two\nlines\ttabbed"),
        r#""two\nlines\ttabbed""#
    );
    // A control character has no literal form at all.
    assert_eq!(super::json_string("\u{1}"), r#""\u0001""#);
}
