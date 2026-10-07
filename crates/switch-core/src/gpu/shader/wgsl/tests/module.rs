//! Layout and module tests.

use super::super::layout::{GENERIC_BASE, GENERIC_STRIDE};
use super::*;
use crate::gpu::pipeline::AttributeBase;
use crate::gpu::texture::{SwizzleSource, TextureSlot};

/// The smallest vertex/fragment pair with an interface.
fn pair(slot: usize) -> (Compiled, Compiled) {
    let offset = (GENERIC_BASE + slot * GENERIC_STRIDE) as u16;
    let vs = program(&[
        (
            Op::Ld {
                dst: 1,
                offset,
                idx: RZ,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (
            Op::St {
                offset,
                idx: RZ,
                src: 1,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let fs = program(&[
        (
            Op::Ipa {
                dst: 0,
                offset,
                mul: None,
                perspective: true,
                sat: false,
                centroid: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    (vs, fs)
}

#[test]
fn a_layout_is_read_off_what_translating_the_program_touched() {
    let (vs, fs) = pair(3);
    let vs = translate(&vs).unwrap();
    let fs = translate(&fs).unwrap();
    assert_eq!(Layout::of(&vs, Stage::Vertex).attributes, vec![3]);
    assert_eq!(Layout::of(&vs, Stage::Vertex).varyings, vec![3]);
    assert_eq!(
        Layout::of(&fs, Stage::Fragment).attributes,
        Vec::<usize>::new()
    );
    assert_eq!(Layout::of(&fs, Stage::Fragment).varyings, vec![3]);
}

#[test]
fn a_layout_names_every_binding_the_text_calls_through() {
    let p = program(&[
        (
            Op::Ldc {
                dst: 1,
                bank: 5,
                offset: 0x10,
                idx: RZ,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (
            Op::Mov {
                dst: 2,
                src: Operand::Const {
                    bank: 1,
                    offset: 0x40,
                },
            },
            ALWAYS,
        ),
        (
            Op::Texs {
                dst: 4,
                dst2: 6,
                coords: [7, 8, RZ],
                dref: None,
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    assert_eq!(layout.const_banks, vec![1, 5]);
    assert_eq!(
        layout.textures,
        vec![TextureBinding {
            slot: TextureSlot::Bound(0x1a4),
            dim: TexDim::T2d,
            compare: false,
            swizzle: IDENTITY_SWIZZLE
        }]
    );
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("var<storage, read> cb1:"), "{source}");
    assert!(source.contains("var<storage, read> cb5:"), "{source}");
    assert!(
        source.contains("case 420u: { return textureSampleLevel(tex0, smp0"),
        "{source}"
    );
}

#[test]
fn a_swizzled_texture_is_rearranged_where_it_is_sampled() {
    // Descriptor swizzles are applied.
    let p = program(&[
        (
            Op::Texs {
                dst: 0,
                dst2: 2,
                coords: [4, 5, RZ],
                dref: None,
                handle: 8,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.textures[0].swizzle = [
        SwizzleSource::R,
        SwizzleSource::R,
        SwizzleSource::R,
        SwizzleSource::One,
    ];
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("return vec4<f32>(sampled.x, sampled.x, sampled.x, 1.0);"),
        "{source}"
    );
    layout.textures[0].swizzle = IDENTITY_SWIZZLE;
    let plain = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(!plain.contains("let sampled ="), "{plain}");
}

#[test]
fn both_stages_name_the_same_location_for_a_varying() {
    // Both stages must agree on locations.
    let (vs, fs) = pair(7);
    let vs = translate(&vs).unwrap();
    let fs = translate(&fs).unwrap();
    let vs = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
    let fs = module(&fs, Stage::Fragment, &Layout::of(&fs, Stage::Fragment)).unwrap();
    assert!(
        vs.contains("@location(7) @interpolate(linear) vary7"),
        "{vs}"
    );
    assert!(
        fs.contains("@location(7) @interpolate(linear) vary7"),
        "{fs}"
    );
}

/// The vertex stage is told which varyings are centroid.
#[test]
fn a_centroid_varying_is_qualified_the_same_way_in_both_stages() {
    let offset = (GENERIC_BASE + 5 * GENERIC_STRIDE) as u16;
    let (vs, _) = pair(5);
    let fs = program(&[
        (
            Op::Ipa {
                dst: 0,
                offset,
                mul: None,
                perspective: true,
                sat: false,
                centroid: true,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let vs = translate(&vs).unwrap();
    let fs = translate(&fs).unwrap();
    let fs_layout = Layout::of(&fs, Stage::Fragment);
    assert_eq!(fs_layout.centroid_varyings, vec![5]);
    let mut vs_layout = Layout::of(&vs, Stage::Vertex);
    assert_eq!(
        vs_layout.centroid_varyings,
        Vec::<usize>::new(),
        "a vertex shader has no ipa"
    );
    vs_layout.centroid_varyings = fs_layout.centroid_varyings.clone();

    let vs = module(&vs, Stage::Vertex, &vs_layout).unwrap();
    let fs = module(&fs, Stage::Fragment, &fs_layout).unwrap();
    assert!(
        vs.contains("@location(5) @interpolate(linear, centroid) vary5"),
        "{vs}"
    );
    assert!(
        fs.contains("@location(5) @interpolate(linear, centroid) vary5"),
        "{fs}"
    );
}

#[test]
fn varyings_interpolate_linearly_because_the_shader_divides_by_w_itself() {
    // Varyings are linear, carrying value/w.
    let (vs, _) = pair(0);
    let vs = translate(&vs).unwrap();
    let source = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
    assert!(source.contains("@interpolate(linear)"), "{source}");
    assert!(!source.contains("perspective"), "{source}");
    assert!(
        source.contains("let over_w = 1.0 / out.position.w;"),
        "{source}"
    );
    assert!(source.contains("* over_w;"), "{source}");
}

#[test]
fn a_clip_position_no_store_writes_is_the_one_the_rasterizer_defaults_to() {
    let (vs, _) = pair(0);
    let vs = translate(&vs).unwrap();
    let source = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
    assert!(source.contains("attr_out[31u] = 1.0;"), "{source}");
}

#[test]
fn an_integer_attribute_is_declared_as_one_and_carried_as_its_bits() {
    // Integer attributes are declared integer and bitcast.
    let (vs, _) = pair(3);
    let vs = translate(&vs).unwrap();
    let mut layout = Layout::of(&vs, Stage::Vertex);
    layout.integer_attributes = vec![(3, AttributeBase::Sint)];
    let source = module(&vs, Stage::Vertex, &layout).unwrap();
    assert!(source.contains("@location(3) attr3: vec4<i32>"), "{source}");
    assert!(source.contains("bitcast<f32>(input.attr3.x)"), "{source}");
    assert!(braces_balance(&source), "{source}");

    let plain = Layout::of(&vs, Stage::Vertex);
    let source = module(&vs, Stage::Vertex, &plain).unwrap();
    assert!(source.contains("@location(3) attr3: vec4<f32>"), "{source}");
    assert!(!source.contains("bitcast<f32>(input."), "{source}");
}

#[test]
fn a_depth_only_pass_writes_no_colour_and_still_shades() {
    // Depth-only fragments return nothing but still run for `kil`.
    let p = program(&[
        (
            Op::Mov {
                dst: 0,
                src: Operand::Imm(0x3f80_0000),
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.targets = 0;
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("fn fs_main(input: FragmentInput) {"),
        "{source}"
    );
    assert!(!source.contains("@location(0) vec4<f32>"), "{source}");
    assert!(source.contains("if (run()) { discard; }"), "{source}");
    let entry = source
        .split("@fragment")
        .nth(1)
        .expect("a fragment entry point");
    assert!(!entry.contains("return"), "{source}");
    assert!(braces_balance(&source), "{source}");

    layout.coverage = Some(quad_coverage(u32::MAX, true));
    let shaded = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(shaded.contains("let target0 = vec4<f32>("), "{shaded}");
    assert!(
        shaded.contains("if (sample >= u32(floor(clamp(target0.w"),
        "{shaded}"
    );
    assert!(!shaded.contains("return target0;"), "{shaded}");
    assert!(braces_balance(&shaded), "{shaded}");
}

#[test]
fn a_bgra_attribute_is_swapped_where_it_is_read() {
    // BGRA swaps as `fetch_attribute` does.
    let (vs, _) = pair(2);
    let vs = translate(&vs).unwrap();
    let mut layout = Layout::of(&vs, Stage::Vertex);
    layout.bgra_attributes = vec![2];
    let source = module(&vs, Stage::Vertex, &layout).unwrap();
    assert!(source.contains("attr_in[40u] = input.attr2.z;"), "{source}");
    assert!(source.contains("attr_in[42u] = input.attr2.x;"), "{source}");
    assert!(source.contains("attr_in[41u] = input.attr2.y;"), "{source}");
    assert!(source.contains("attr_in[43u] = input.attr2.w;"), "{source}");

    let plain = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
    assert!(plain.contains("attr_in[40u] = input.attr2.x;"), "{plain}");
}

/// A 2x2 grid in raster order.
fn quad_coverage(sample_mask: u32, alpha_to_coverage: bool) -> Coverage {
    Coverage {
        samples_x: 2,
        samples_y: 2,
        sample_of_slot: vec![0, 1, 2, 3],
        sample_mask,
        alpha_to_coverage,
    }
}

#[test]
fn a_sample_mask_discards_the_texels_it_excludes() {
    // Per-texel multisample rendering applies the sample mask by discarding.
    let p = program(&[(Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.coverage = Some(quad_coverage(0b0001, false));
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("var sample_of_slot = array<u32, 4>(0u, 1u, 2u, 3u);"),
        "{source}"
    );
    assert!(source.contains("u32(input.position.x) % 2u"), "{source}");
    assert!(
        source.contains("if ((1u >> sample) & 1u) == 0u { discard; }"),
        "{source}"
    );
    assert!(braces_balance(&source), "{source}");

    // An all-ones mask emits no branch.
    layout.coverage = Some(quad_coverage(u32::MAX, false));
    let open = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(!open.contains("& 1u) == 0u"), "{open}");
}

#[test]
fn alpha_to_coverage_keeps_the_same_prefix_the_rasterizer_keeps() {
    // Alpha-to-coverage keeps a prefix of `round(alpha * count)` samples.
    let p = program(&[(Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.coverage = Some(quad_coverage(u32::MAX, true));
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("let target0 = vec4<f32>("), "{source}");
    assert!(
        source.contains("if (sample >= u32(floor(clamp(target0.w, 0.0, 1.0) * 4.0 + 0.5)))"),
        "{source}"
    );
    assert!(source.contains("return target0;"), "{source}");
    assert!(braces_balance(&source), "{source}");

    layout.coverage = Some(quad_coverage(u32::MAX, false));
    let plain = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(!plain.contains("floor(clamp("), "{plain}");
    assert!(!plain.contains("let target0"), "{plain}");
}

#[test]
fn a_single_sample_draw_carries_no_coverage_at_all() {
    // Only for grids with more than one sample.
    let p = program(&[(Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    assert!(layout.coverage.is_none());
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(!source.contains("sample_of_slot"), "{source}");
}

#[test]
fn a_vertex_shader_with_no_attributes_declares_no_input_struct() {
    // WGSL has no empty struct.
    let p = program(&[(Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Vertex);
    assert!(layout.attributes.is_empty());
    let source = module(&translated, Stage::Vertex, &layout).unwrap();
    assert!(!source.contains("struct VertexInput"), "{source}");
    assert!(!source.contains("input: VertexInput"), "{source}");
}

#[test]
fn a_colour_channel_the_shader_never_wrote_is_zero_not_a_missing_register() {
    let p = program(&[
        (
            Op::Mov {
                dst: 0,
                src: Operand::Imm(0x3f80_0000),
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("return vec4<f32>(bitcast<f32>(r0), 0.0, 0.0, 0.0);"),
        "{source}"
    );
}

#[test]
fn each_extra_colour_target_takes_the_next_four_registers() {
    let mut entries: Vec<(Op, Pred)> = (0..8u8)
        .map(|r| {
            (
                Op::Mov {
                    dst: r,
                    src: Operand::Imm(0),
                },
                ALWAYS,
            )
        })
        .collect();
    entries.push((Op::Exit, ALWAYS));
    let translated = translate(&program(&entries)).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.targets = 2;
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("@location(1) target1: vec4<f32>"),
        "{source}"
    );
    assert!(
        source.contains(
            "out.target1 = vec4<f32>(bitcast<f32>(r4), bitcast<f32>(r5), \
                         bitcast<f32>(r6), bitcast<f32>(r7));"
        ),
        "{source}"
    );
}

#[test]
fn a_texture_dimension_with_no_binding_is_reported() {
    // Unbindable dimensions are reported.
    let p = program(&[
        (
            Op::Texs {
                dst: 0,
                dst2: 2,
                coords: [4, 5, 6],
                dref: None,
                handle: 1,
                // 1D does not bind.
                dim: TexDim::T1d,
                mask: [true, true, true, true],
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    assert_eq!(
        module(&translated, Stage::Fragment, &layout).unwrap_err(),
        Unsupported::TextureDimension { dim: TexDim::T1d }
    );
}

const TEX_B: u64 = 0xdeba0007a0270000;

#[test]
fn a_bindless_tex_binds_the_constant_word_its_handle_was_loaded_from() {
    let tex = crate::gpu::shader::isa::decode(TEX_B).op;
    let p = program(&[
        (
            Op::Ldc {
                dst: 2,
                bank: 3,
                offset: 0x10,
                idx: RZ,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (tex, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let slot = TextureSlot::Bindless {
        bank: 3,
        offset: 0x10,
    };
    assert_eq!(translated.textures, vec![(slot, TexDim::T2d, false)]);
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    let key = slot.key();
    assert!(
        source.contains(&format!(
            "case {key}u: {{ return textureSampleLevel(tex0, smp0"
        )),
        "{source}"
    );
    assert!(
        translated.source.contains(&format!("texSample({key}u,")),
        "{}",
        translated.source
    );
}

#[test]
fn a_bindless_handle_is_traced_through_a_wide_load_and_across_blocks() {
    // A sole write in another block still counts.
    let tex = crate::gpu::shader::isa::decode(TEX_B).op;
    let p = program(&[
        (
            Op::Ldc {
                dst: 1,
                bank: 4,
                offset: 0x20,
                idx: RZ,
                size: MemSize::B64,
            },
            ALWAYS,
        ),
        (Op::Ssy { target: at(3) }, ALWAYS),
        (Op::Sync, ALWAYS),
        (tex, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    assert_eq!(
        translated.textures,
        vec![(
            TextureSlot::Bindless {
                bank: 4,
                offset: 0x24
            },
            TexDim::T2d,
            false
        )]
    );
}

#[test]
fn a_bindless_handle_that_is_not_a_constant_is_refused() {
    let tex = crate::gpu::shader::isa::decode(TEX_B).op;
    let computed = |pred| {
        program(&[
            (
                Op::Mov {
                    dst: 2,
                    src: Operand::Const {
                        bank: 3,
                        offset: 0x10,
                    },
                },
                pred,
            ),
            (tex, ALWAYS),
            (Op::Exit, ALWAYS),
        ])
    };
    // A guarded load could leave any handle.
    let guarded = Pred {
        reg: 0,
        negate: false,
    };
    assert_eq!(
        translate(&computed(guarded)).unwrap_err(),
        Unsupported::UntracedHandle { at: 1 }
    );
    assert!(translate(&computed(ALWAYS)).is_ok());

    let indexed = program(&[
        (
            Op::Ldc {
                dst: 2,
                bank: 3,
                offset: 0x10,
                idx: 5,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (tex, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    assert_eq!(
        translate(&indexed).unwrap_err(),
        Unsupported::UntracedHandle { at: 1 }
    );
}

#[test]
fn a_global_load_through_a_descriptor_loaded_whole_reads_that_descriptor() {
    let ldg = Op::Ldg {
        dst: 2,
        addr: 0,
        offset: 0,
        size: MemSize::B64,
    };
    let through_ldc = program(&[
        (
            Op::Ldc {
                dst: 0,
                bank: 1,
                offset: 0x40,
                idx: RZ,
                size: MemSize::B64,
            },
            ALWAYS,
        ),
        (ldg, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    assert_eq!(translate(&through_ldc).unwrap().globals, vec![(1, 0x40)]);

    let mov = |dst, offset| Op::Mov {
        dst,
        src: Operand::Const { bank: 1, offset },
    };
    let through_movs = program(&[
        (mov(0, 0x40), ALWAYS),
        (mov(1, 0x44), ALWAYS),
        (ldg, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    assert_eq!(translate(&through_movs).unwrap().globals, vec![(1, 0x40)]);

    let mismatched = program(&[
        (mov(0, 0x40), ALWAYS),
        (mov(1, 0x48), ALWAYS),
        (ldg, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    assert!(translate(&mismatched).is_err());
}

/// Nintendo Switch Sports: a size-only texture query and an RZ-cleared offset.
#[test]
fn a_size_query_alone_and_a_zeroed_offset_translate() {
    let txq = crate::gpu::shader::isa::decode(0xdf48008180470800).op;
    let query_only = program(&[(txq, ALWAYS), (Op::Exit, ALWAYS)]);
    let translated = translate(&query_only).unwrap();
    assert_eq!(
        translated.textures,
        vec![(TextureSlot::Bound(8), TexDim::T2d, false)]
    );

    let tex = crate::gpu::shader::isa::decode(0xc0780083a0a70808).op;
    let zeroed = program(&[
        (
            Op::Mov {
                dst: 10,
                src: Operand::Reg(RZ),
            },
            ALWAYS,
        ),
        (tex, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    assert_eq!(translate(&zeroed).unwrap().texture_offsets, vec![(0, 0)]);
}

#[test]
fn a_module_is_complete_enough_to_stand_on_its_own() {
    // All four `HOST_INTERFACE` hooks are defined in the module.
    let (vs, fs) = pair(2);
    for (program, stage) in [(vs, Stage::Vertex), (fs, Stage::Fragment)] {
        let translated = translate(&program).unwrap();
        let layout = Layout::of(&translated, stage);
        let source = module(&translated, stage, &layout).unwrap();
        for hook in ["fn attrIn(", "fn attrOut(", "fn cbRead(", "fn texSample("] {
            assert!(
                source.contains(hook),
                "{stage:?} module has no {hook}:\n{source}"
            );
        }
        assert!(braces_balance(&source), "{stage:?}:\n{source}");
    }
}
