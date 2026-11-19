//! Layout and module tests.

use super::super::layout::{atom_code, GENERIC_BASE, GENERIC_STRIDE};
use super::*;
use crate::gpu::pipeline::AttributeBase;
use crate::gpu::shader::isa::{AtomOp, AtomSpace, AtomType};
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

/// A fixed-function texture coordinate (`a[0x300]`) is a varying on both renderers, slot 40
/// counting on from the generic ones. The Vulkan driver's blits pass their coordinates there;
/// dropped, every texel read was (0, 0) and Minecraft's panorama faces came out one colour.
#[test]
fn a_fixed_function_texture_coordinate_is_a_varying() {
    let texcoord = 0x300;
    let vs = program(&[
        (
            Op::Ld {
                dst: 1,
                offset: GENERIC_BASE as u16,
                idx: RZ,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (
            Op::St {
                offset: texcoord,
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
                offset: texcoord,
                mul: None,
                perspective: false,
                sat: false,
                centroid: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    assert_eq!(fs.interpolated_slots(), [40]);
    let vs = Layout::of(&translate(&vs).unwrap(), Stage::Vertex);
    assert_eq!((vs.attributes, vs.varyings), (vec![0], vec![40]));
    assert_eq!(
        Layout::of(&translate(&fs).unwrap(), Stage::Fragment).varyings,
        vec![40]
    );
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
            swizzle: IDENTITY_SWIZZLE,
            fold: None,
            uint: false,
            crop: None,
            sampler: None,
            sized_only: false,
        }]
    );
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("var<storage, read> cbs:"), "{source}");
    assert!(
        source.contains("case 420u: { return textureSampleLevel(tex0, smp0"),
        "{source}"
    );
}

#[test]
fn a_shadow_tld4s_gathers_comparisons_through_the_comparison_sampler() {
    let p = program(&[
        (
            Op::Tld4s {
                dst: 2,
                dst2: 4,
                coords: [2, 3],
                offset: None,
                dref: Some(4),
                handle: 8,
                component: 0,
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    assert!(layout.textures[0].compare);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("texGatherCompare(8u, "), "{source}");
    assert!(
        source.contains("textureGatherCompare(tex0, smp0"),
        "{source}"
    );
}

#[test]
fn a_tld4s_gathers_into_texss_registers() {
    let p = program(&[
        (
            Op::Tld4s {
                dst: 4,
                dst2: 6,
                coords: [7, 8],
                offset: None,
                dref: None,
                handle: 8,
                component: 2,
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    assert_eq!(layout.textures.len(), 1);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("texGather(8u, 2u"), "{source}");
    // Four channels, two registers each side.
    for reg in ["r4", "r5", "r6", "r7"] {
        assert!(
            source.contains(&format!("{reg} = ")),
            "{reg} unwritten: {source}"
        );
    }
}

#[test]
fn an_integer_image_is_read_by_texel_and_an_integer_target_takes_raw_words() {
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
    layout.textures[0].uint = true;
    layout.uint_targets = vec![(0, u32::MAX)];
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("var tex0: texture_2d<u32>"), "{source}");
    assert!(
        source.contains("textureLoad(tex0, at, 0)"),
        "an integer image is not filtered: {source}"
    );
    assert!(source.contains("-> @location(0) vec4<u32>"), "{source}");
    assert!(
        source.contains("return vec4<u32>(r0, r1, r2, r3);"),
        "{source}"
    );
    layout.uint_targets = vec![(0, 0xFFFF)];
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("return vec4<u32>(min(r0, 0xffffu), min(r1, 0xffffu), min(r2, 0xffffu), min(r3, 0xffffu));"),
        "a 16-bit integer target saturates: {source}"
    );

    // A gather reads an integer image's footprint as raw words.
    let gather = program(&[
        (
            Op::Tld4s {
                dst: 0,
                dst2: 2,
                coords: [4, 5],
                offset: None,
                dref: None,
                handle: 8,
                component: 0,
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&gather).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.textures[0].uint = true;
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    for channel in 0..4 {
        assert!(
            source.contains(&format!(
                "case {channel}u: {{ return bitcast<vec4<f32>>(textureGather({channel}, tex0, \
                 smp0, vec2<f32>(u, v))); }}"
            )),
            "{source}"
        );
    }
    assert!(source.contains("var tex0: texture_2d<u32>"), "{source}");

    // An offset sample moves the nearest texel the way `interp` moves the coordinate.
    let offset = program(&[
        (Op::Mov32i { dst: 7, imm: 0xf1 }, ALWAYS),
        (
            Op::Tex {
                dst: 0,
                coords: [4, 5, RZ],
                layer: None,
                dref: None,
                offset: Some(7),
                lod: None,
                handle: 8,
                handle_reg: None,
                dim: TexDim::T2d,
                mask: [true; 4],
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&offset).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.textures[0].uint = true;
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains(
            "vec2<f32>(u + f32(1) * (1.0 / f32(size.x)), v + f32(-1) * (1.0 / f32(size.y)))"
        ),
        "{source}"
    );
    assert!(source.contains("textureLoad(tex0, at, 0)"), "{source}");
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
        vs.contains("@location(0) @interpolate(linear) vary7"),
        "{vs}"
    );
    assert!(
        fs.contains("@location(0) @interpolate(linear) vary7"),
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
        vs.contains("@location(0) @interpolate(linear, centroid) vary5"),
        "{vs}"
    );
    assert!(
        fs.contains("@location(0) @interpolate(linear, centroid) vary5"),
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
fn a_scaled_attribute_is_declared_integer_and_converted_to_its_value() {
    let (vs, _) = pair(3);
    let vs = translate(&vs).unwrap();
    let mut layout = Layout::of(&vs, Stage::Vertex);
    layout.integer_attributes = vec![(3, AttributeBase::Uscaled)];
    let source = module(&vs, Stage::Vertex, &layout).unwrap();
    assert!(source.contains("@location(3) attr3: vec4<u32>"), "{source}");
    assert!(source.contains("f32(input.attr3.x)"), "{source}");
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
        per_sample_pass: false,
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
fn a_per_sample_pass_keeps_only_the_texels_of_its_sample() {
    let p = program(&[(Op::Exit, ALWAYS)]);
    let translated = translate(&p).unwrap();
    let mut layout = Layout::of(&translated, Stage::Fragment);
    layout.coverage = Some(Coverage {
        per_sample_pass: true,
        ..quad_coverage(u32::MAX, false)
    });
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains("override pass_sample: u32 = 0u;"),
        "{source}"
    );
    assert!(
        source.contains("if (sample != pass_sample) { discard; }"),
        "{source}"
    );
    assert!(braces_balance(&source), "{source}");

    layout.coverage = Some(quad_coverage(u32::MAX, false));
    let plain = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(!plain.contains("pass_sample"), "{plain}");
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

/// Just Dance 2023's split handle: `mov r2, c2[0x168]; lop.or r2, r2, <second>; tex.b ..., r2`.
fn split_handle(second: Operand) -> Vec<(Op, Pred)> {
    use crate::gpu::shader::isa::LogicOp;
    let tex = crate::gpu::shader::isa::decode(TEX_B).op;
    vec![
        (
            Op::Mov {
                dst: 2,
                src: Operand::Const {
                    bank: 2,
                    offset: 0x168,
                },
            },
            ALWAYS,
        ),
        (
            Op::Lop {
                dst: 2,
                a: 2,
                ainv: false,
                b: second,
                binv: false,
                op: LogicOp::Or,
                pred: None,
                cc: false,
            },
            ALWAYS,
        ),
        (tex, ALWAYS),
    ]
}

fn sampler_word(offset: u16) -> Operand {
    Operand::Const { bank: 2, offset }
}

#[test]
fn a_split_handle_set_up_in_the_entry_block_reaches_a_later_block() {
    // Just Dance 2023 loads the handle first and samples it across the shader.
    let mut ops = split_handle(sampler_word(0x578));
    let tex = ops.pop().unwrap();
    ops.push((Op::Ssy { target: at(4) }, ALWAYS));
    ops.push((Op::Sync, ALWAYS));
    ops.push(tex);
    ops.push((Op::Exit, ALWAYS));
    assert_eq!(
        translate(&program(&ops)).unwrap().textures,
        vec![(
            TextureSlot::BindlessPair {
                bank: 2,
                offset: 0x168,
                other_bank: 2,
                other_offset: 0x578,
            },
            TexDim::T2d,
            false
        )]
    );
}

/// `split_handle`, then a branch over `skipped` to the sample and `after` it.
fn split_handle_across_a_branch(skipped: Op, after: Op) -> Vec<(Op, Pred)> {
    let mut ops = split_handle(sampler_word(0x578));
    let tex = ops.pop().unwrap();
    let guard = Pred {
        reg: 0,
        negate: false,
    };
    ops.push((Op::Bra { target: at(4) }, guard));
    ops.push((skipped, ALWAYS));
    ops.push(tex);
    ops.push((after, ALWAYS));
    ops.push((Op::Exit, ALWAYS));
    ops
}

#[test]
fn a_register_reused_after_the_sample_does_not_hide_the_handle() {
    let reuse = Op::Mov {
        dst: 2,
        src: sampler_word(0x20),
    };
    let unrelated = Op::Mov {
        dst: 9,
        src: sampler_word(0x20),
    };
    let ops = split_handle_across_a_branch(unrelated, reuse);
    assert!(translate(&program(&ops)).is_ok());
}

#[test]
fn a_handle_that_differs_by_path_stays_refused() {
    let other = Op::Mov {
        dst: 2,
        src: sampler_word(0x20),
    };
    let unrelated = Op::Mov {
        dst: 9,
        src: sampler_word(0x20),
    };
    let ops = split_handle_across_a_branch(other, unrelated);
    assert!(translate(&program(&ops)).is_err());
}

#[test]
fn a_handle_ored_from_two_constant_words_binds_both() {
    let mut ops = split_handle(sampler_word(0x568));
    ops.push((Op::Exit, ALWAYS));
    let translated = translate(&program(&ops)).unwrap();
    let slot = TextureSlot::BindlessPair {
        bank: 2,
        offset: 0x168,
        other_bank: 2,
        other_offset: 0x568,
    };
    assert_eq!(translated.textures, vec![(slot, TexDim::T2d, false)]);
    let layout = Layout::of(&translated, Stage::Fragment);
    module(&translated, Stage::Fragment, &layout).unwrap();
}

#[test]
fn a_handle_ored_with_a_register_is_refused() {
    let mut ops = split_handle(Operand::Reg(5));
    ops.push((Op::Exit, ALWAYS));
    assert_eq!(
        translate(&program(&ops)).unwrap_err(),
        Unsupported::UntracedHandle { at: 2 }
    );
}

#[test]
fn two_handles_keyed_by_one_first_word_are_refused() {
    // The same image word with two sampler words would share one sampling case.
    let mut ops = split_handle(sampler_word(0x568));
    ops.extend(split_handle(sampler_word(0x56c)));
    ops.push((Op::Exit, ALWAYS));
    assert_eq!(
        translate(&program(&ops)).unwrap_err(),
        Unsupported::SharedHandleKey { at: 5 }
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
    assert_eq!(
        translate(&through_ldc).unwrap().globals,
        vec![Global {
            bank: 1,
            offset: 0x40,
            written: false
        }]
    );

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
    assert_eq!(
        translate(&through_movs).unwrap().globals,
        vec![Global {
            bank: 1,
            offset: 0x40,
            written: false
        }]
    );

    let mismatched = program(&[
        (mov(0, 0x40), ALWAYS),
        (mov(1, 0x48), ALWAYS),
        (ldg, ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    assert!(translate(&mismatched).is_err());
}

#[test]
fn an_atomic_on_a_descriptor_makes_its_buffer_read_write_and_only_for_fragments() {
    let atom = |ty, space| Op::Atom {
        dst: RZ,
        addr: 0,
        offset: 4,
        src: 2,
        op: AtomOp::Or,
        ty,
        space,
    };
    let load = Op::Ldc {
        dst: 0,
        bank: 1,
        offset: 0x40,
        idx: RZ,
        size: MemSize::B64,
    };
    let translated = translate(&program(&[
        (load, ALWAYS),
        (atom(AtomType::U32, AtomSpace::Global), ALWAYS),
        (Op::Exit, ALWAYS),
    ]))
    .unwrap();
    assert_eq!(
        translated.globals,
        vec![Global {
            bank: 1,
            offset: 0x40,
            written: true
        }]
    );
    let layout = Layout::of(&translated, Stage::Fragment);
    let fragment = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(fragment.contains("var<storage, read_write> g0: array<atomic<u32>>"));
    assert!(fragment.contains("atomicOr(&g0[word], value)"));
    assert_eq!(
        module(&translated, Stage::Vertex, &layout).unwrap_err(),
        Unsupported::VertexWrite
    );

    for (ty, space) in [
        (AtomType::U64, AtomSpace::Global),
        (AtomType::S64, AtomSpace::Global),
        (AtomType::U128, AtomSpace::Global),
        (AtomType::U32, AtomSpace::Shared),
    ] {
        let refused = translate(&program(&[
            (load, ALWAYS),
            (atom(ty, space), ALWAYS),
            (Op::Exit, ALWAYS),
        ]));
        assert!(
            matches!(refused, Err(Unsupported::Op { at: 1, .. })),
            "{ty:?} {space:?}"
        );
    }
}

fn through_descriptor(op: Op) -> Result<Translation, Unsupported> {
    translate(&program(&[
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
        (op, ALWAYS),
        (Op::Exit, ALWAYS),
    ]))
}

fn fragment_module(translated: &Translation) -> String {
    let layout = Layout::of(translated, Stage::Fragment);
    module(translated, Stage::Fragment, &layout).unwrap()
}

#[test]
fn an_atomic_runs_the_row_for_its_operation_and_type() {
    let run = |op, ty| {
        let translated = through_descriptor(Op::Atom {
            dst: 3,
            addr: 0,
            offset: -4,
            src: 2,
            op,
            ty,
            space: AtomSpace::Global,
        })
        .unwrap();
        let call = translated
            .source
            .lines()
            .find(|line| line.contains("gAtom("))
            .expect("a gAtom call")
            .to_string();
        (call, fragment_module(&translated))
    };
    let code = |op, ty| atom_code(op, ty).expect("a row");

    // Signed minimum is a loop over signed words; unsigned minimum is the direct atomic.
    let (call, source) = run(AtomOp::Min, AtomType::S32);
    assert!(call.contains(&format!("gAtom({}u,", code(AtomOp::Min, AtomType::S32))));
    assert!(source.contains("min(bitcast<i32>(was), bitcast<i32>(value))"));
    assert_ne!(
        code(AtomOp::Min, AtomType::S32),
        code(AtomOp::Min, AtomType::U32)
    );

    // A type without a row of its own runs the unsigned row.
    assert_eq!(
        code(AtomOp::Xor, AtomType::S32),
        code(AtomOp::Xor, AtomType::U32)
    );
    assert_eq!(
        code(AtomOp::Add, AtomType::S32),
        code(AtomOp::Add, AtomType::U32)
    );

    // Float add is a compare-exchange loop on the float's bits.
    let (_, source) = run(AtomOp::Add, AtomType::F32);
    assert!(source.contains("bitcast<f32>(was) + bitcast<f32>(value)"));
    assert!(source.contains("atomicCompareExchangeWeak(&g0[word], was,"));

    // `cas` passes the register after its comparand as the new word; the others pass zero.
    let (call, _) = run(AtomOp::Cas, AtomType::U32);
    assert!(call.ends_with(", r3);"), "{call}");
    let (call, source) = run(AtomOp::Add, AtomType::U32);
    assert!(call.ends_with(", 0u);"), "{call}");

    // A negative offset wraps rather than printing a negative unsigned literal.
    assert!(source.contains("+ 4294967292u"), "{source}");
}

#[test]
fn only_the_atomics_eden_applies_have_a_row() {
    use AtomOp::*;
    for op in [Add, Min, Max, And, Or, Xor, Exch, Cas] {
        for ty in [AtomType::U32, AtomType::S32] {
            assert!(atom_code(op, ty).is_some(), "{op:?} {ty:?}");
        }
    }
    assert!(atom_code(Add, AtomType::F32).is_some());
    for (op, ty) in [
        (Inc, AtomType::U32),
        (Dec, AtomType::S32),
        (SafeAdd, AtomType::U32),
        (SafeAdd, AtomType::F32),
        (Min, AtomType::F32),
        (Max, AtomType::F32),
        (Xor, AtomType::F32),
        (Exch, AtomType::F32),
    ] {
        assert_eq!(atom_code(op, ty), None, "{op:?} {ty:?}");
    }
    for op in [Add, Min, Cas, Xor] {
        for ty in [AtomType::U64, AtomType::S64, AtomType::U128] {
            assert_eq!(atom_code(op, ty), None, "{op:?} {ty:?}");
        }
    }
}

#[test]
fn an_atomic_without_a_row_is_refused() {
    for (op, ty) in [
        (AtomOp::Inc, AtomType::U32),
        (AtomOp::SafeAdd, AtomType::U32),
        (AtomOp::Min, AtomType::F32),
    ] {
        let refused = through_descriptor(Op::Atom {
            dst: 3,
            addr: 0,
            offset: 0,
            src: 2,
            op,
            ty,
            space: AtomSpace::Global,
        });
        assert!(
            matches!(refused, Err(Unsupported::Op { at: 1, .. })),
            "{op:?} {ty:?}"
        );
    }
}

#[test]
fn a_store_through_a_descriptor_makes_its_buffer_read_write_and_only_for_fragments() {
    let stg = |size| Op::Stg {
        addr: 0,
        offset: 8,
        src: 2,
        size,
    };
    let translated = through_descriptor(stg(MemSize::B64)).unwrap();
    assert_eq!(
        translated.globals,
        vec![Global {
            bank: 1,
            offset: 0x40,
            written: true
        }]
    );
    let source = fragment_module(&translated);
    assert!(source.contains("var<storage, read_write> g0: array<atomic<u32>>"));
    assert!(source.contains("atomicStore(&g0[word], low)"));
    assert_eq!(translated.source.matches("gWrite(0u,").count(), 2);
    assert!(translated.source.contains("0xffffffffu);"));
    let layout = Layout::of(&translated, Stage::Fragment);
    assert_eq!(
        module(&translated, Stage::Vertex, &layout).unwrap_err(),
        Unsupported::VertexWrite
    );

    // Narrow stores merge one byte at a time under a mask.
    let narrow = through_descriptor(stg(MemSize::U16)).unwrap();
    assert_eq!(narrow.source.matches("gWrite(0u,").count(), 2);
    assert!(narrow.source.contains(", 0xffu);"));
    assert!(!narrow.source.contains("0xffffffffu"));
    assert!(fragment_module(&narrow).contains(
        "atomicCompareExchangeWeak(&g0[word], was, (was & ~low_mask) | (low & low_mask))"
    ));
}

#[test]
fn a_narrow_global_load_reads_its_bytes_and_a_signed_one_extends_them() {
    let ldg = |size| Op::Ldg {
        dst: 2,
        addr: 0,
        offset: 6,
        size,
    };
    let signed = through_descriptor(ldg(MemSize::S16)).unwrap();
    assert_eq!(signed.source.matches("gRead(0u,").count(), 2);
    assert!(signed.source.contains("sext("));
    let unsigned = through_descriptor(ldg(MemSize::U8)).unwrap();
    assert_eq!(unsigned.source.matches("gRead(0u,").count(), 1);
    assert!(!unsigned.source.contains("sext("));
    let wide = through_descriptor(ldg(MemSize::B64)).unwrap();
    assert_eq!(wide.source.matches("gRead(0u,").count(), 2);
}

#[test]
fn a_store_without_a_traced_descriptor_is_refused() {
    let untraced = translate(&program(&[
        (
            Op::Stg {
                addr: 4,
                offset: 0,
                src: 2,
                size: MemSize::B32,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]));
    assert!(matches!(untraced, Err(Unsupported::Op { at: 0, .. })));
}

#[test]
fn shared_memory_and_barriers_are_refused_even_through_a_traced_descriptor() {
    let traced = Op::Stg {
        addr: 0,
        offset: 0,
        src: 2,
        size: MemSize::B32,
    };
    assert!(through_descriptor(traced).is_ok());
    for op in [
        Op::Lds {
            dst: 2,
            addr: 0,
            offset: 0,
            size: MemSize::B32,
        },
        Op::Sts {
            addr: 0,
            offset: 0,
            src: 2,
            size: MemSize::B32,
        },
        Op::Bar {
            mode: crate::gpu::shader::isa::BarMode::Sync,
        },
    ] {
        assert!(
            matches!(through_descriptor(op), Err(Unsupported::Op { at: 1, .. })),
            "{op:?}"
        );
    }
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
    assert_eq!(translate(&zeroed).unwrap().texture_offsets, vec![(0, 0, 0)]);
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

/// Tiles overlap by a row, so rows `r` and `r + 1` are always in tile `r / (tile_rows - 1)`;
/// a repeating image's wrap tile holds the last row and then the first.
#[test]
fn an_image_folds_into_as_few_overlapping_tiles_as_hold_it() {
    use crate::gpu::shader::wgsl::Fold;
    // Nintendo Switch Sports' glyph atlas under a browser's limits.
    let atlas = Fold::of(32768, 8192, 256, false).expect("folds");
    assert_eq!((atlas.tile_rows, atlas.layers), (8192, 5));
    for row in [0, 8190, 8191, 16381, 16382, 32766] {
        let tile = (row / (atlas.tile_rows - 1)).min(atlas.tiles() - 1);
        let at = row - atlas.image_row(tile, 0);
        assert!(
            at + 1 < atlas.tile_rows,
            "row {row} is the last of tile {tile}"
        );
        assert_eq!(atlas.image_row(tile, at), row);
        assert_eq!(atlas.image_row(tile, at + 1), row + 1);
    }
    // The last tile runs past the image, repeating its last row.
    assert_eq!(atlas.image_row(4, atlas.tile_rows - 1), 32767);

    let repeating = Fold::of(32768, 8192, 256, true).expect("folds");
    assert_eq!((repeating.tiles(), repeating.layers), (5, 6));
    assert_eq!(repeating.image_row(5, 0), 32767);
    assert_eq!(repeating.image_row(5, 1), 0);

    assert_eq!(Fold::of(8192, 8192, 256, false), None, "fits as it is");
    assert_eq!(Fold::of(8193, 8192, 256, false).map(|f| f.layers), Some(2));
    assert_eq!(
        Fold::of(32768, 8192, 5, true),
        None,
        "more tiles than layers"
    );
}

#[test]
fn a_tlds_loads_the_texel_and_lands_it_like_texs() {
    let tlds = |dim, coords, layer, offset| {
        program(&[
            (
                Op::Tlds {
                    dst: 4,
                    dst2: RZ,
                    coords,
                    layer,
                    lod: None,
                    offset,
                    handle: 10,
                    dim,
                    mask: [false, true, false, false],
                    f16: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ])
    };
    let source = |program: &Compiled| {
        let translated = translate(program).unwrap();
        let layout = Layout::of(&translated, Stage::Fragment);
        module(&translated, Stage::Fragment, &layout).unwrap()
    };

    let plain = source(&tlds(TexDim::T2d, [2, 3, RZ], None, None));
    assert!(plain.contains("texFetch(10u, r2, r3, 0u)"), "{plain}");
    assert!(
        plain.contains("textureLoad(tex0, vec2<u32>(x, y), 0)"),
        "{plain}"
    );

    let offset = source(&tlds(TexDim::T2d, [2, 3, RZ], None, Some(6)));
    assert!(
        offset.contains(
            "texFetch(10u, (r2 + bitcast<u32>(extractBits(bitcast<i32>(r6), 0u, 4u))), \
             (r3 + bitcast<u32>(extractBits(bitcast<i32>(r6), 4u, 4u))), 0u)"
        ),
        "{offset}"
    );

    let array = source(&tlds(TexDim::T2dArray, [5, 6, RZ], Some(2), None));
    assert!(
        array.contains("texFetch(10u, r5, r6, (r2 & 0xffffu))"),
        "{array}"
    );
    assert!(
        array.contains("textureLoad(tex0, vec2<u32>(x, y), z, 0)"),
        "{array}"
    );
    assert!(array.contains("textureNumLayers(tex0)"), "{array}");

    let volume = source(&tlds(TexDim::T3d, [2, 3, 4], None, None));
    assert!(volume.contains("texFetch(10u, r2, r3, r4)"), "{volume}");
    assert!(
        volume.contains("textureLoad(tex0, vec3<u32>(x, y, z), 0)"),
        "{volume}"
    );
}

#[test]
fn a_tlds_of_a_level_register_is_refused_unless_it_is_rz() {
    let tlds = |lod| {
        program(&[
            (
                Op::Tlds {
                    dst: 4,
                    dst2: RZ,
                    coords: [2, 3, RZ],
                    layer: None,
                    lod,
                    offset: None,
                    handle: 10,
                    dim: TexDim::T2d,
                    mask: [true, false, false, false],
                    f16: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ])
    };
    assert!(translate(&tlds(Some(RZ))).is_ok());
    assert!(matches!(
        translate(&tlds(Some(6))),
        Err(Unsupported::Op { at: 0, .. })
    ));
}

#[test]
fn a_gather_offset_register_moves_the_coordinates_before_the_gather() {
    let p = program(&[
        (
            Op::Tld4s {
                dst: 0,
                dst2: 2,
                coords: [4, 5],
                offset: Some(7),
                dref: None,
                handle: 8,
                component: 0,
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(source.contains("= texStep(8u);"), "{source}");
    assert!(
        source.contains("f32(extractBits(bitcast<i32>(r7), 0u, 6u)) *"),
        "{source}"
    );
    assert!(
        source.contains("f32(extractBits(bitcast<i32>(r7), 8u, 6u)) *"),
        "{source}"
    );
    assert!(source.contains("texGather(8u, 0u, "), "{source}");
    assert!(
        source.contains("return vec2<f32>(1.0 / f32(textureDimensions(tex0).x)"),
        "{source}"
    );
}

fn gather(
    dim: TexDim,
    layer: Option<u8>,
    ptp: Option<[u8; 2]>,
    dref: Option<u8>,
    handle_reg: Option<u8>,
) -> Op {
    Op::Tld4 {
        dst: 0,
        coords: [4, 5, 6],
        layer,
        offset: None,
        ptp,
        dref,
        handle: 8,
        handle_reg,
        sparse: 7,
        dim,
        component: 1,
        mask: [true; 4],
    }
}

fn gather_module(ops: &[Op]) -> String {
    let entries: Vec<(Op, Pred)> = ops
        .iter()
        .copied()
        .chain([Op::Exit])
        .map(|op| (op, ALWAYS))
        .collect();
    let translated = translate(&program(&entries)).unwrap();
    let layout = Layout::of(&translated, Stage::Fragment);
    module(&translated, Stage::Fragment, &layout).unwrap()
}

#[test]
fn a_cube_gathers_through_the_direction_and_a_cube_array_through_its_cube() {
    let source = gather_module(&[gather(TexDim::TCube, None, None, None, None)]);
    assert!(source.contains("var tex0: texture_cube<f32>"), "{source}");
    assert!(source.contains("texGather(8u, 1u, "), "{source}");
    assert!(
        source.contains("textureGather(1, tex0, smp0, vec3<f32>(u, v, w))"),
        "{source}"
    );

    let source = gather_module(&[gather(TexDim::TCubeArray, Some(2), None, None, None)]);
    assert!(
        source.contains("textureGather(1, tex0, smp0, vec3<f32>(u, v, w), layer)"),
        "{source}"
    );
}

#[test]
fn a_shadow_gather_compares_through_a_depth_texture_of_its_dimension() {
    for (dim, layer, texture, call) in [
        (
            TexDim::T2dArray,
            Some(2),
            "texture_depth_2d_array",
            "textureGatherCompare(tex0, smp0, vec2<f32>(u, v), layer, dref)",
        ),
        (
            TexDim::TCube,
            None,
            "texture_depth_cube",
            "textureGatherCompare(tex0, smp0, vec3<f32>(u, v, w), dref)",
        ),
        (
            TexDim::TCubeArray,
            Some(2),
            "texture_depth_cube_array",
            "textureGatherCompare(tex0, smp0, vec3<f32>(u, v, w), layer, dref)",
        ),
    ] {
        let source = gather_module(&[gather(dim, layer, None, Some(10), None)]);
        assert!(source.contains(&format!("var tex0: {texture}")), "{source}");
        assert!(source.contains(call), "{source}");
        assert!(source.contains("texGatherCompare(8u, "), "{source}");
    }
}

#[test]
fn a_cube_sample_cannot_compare() {
    let p = program(&[
        (
            Op::Texs {
                dst: 0,
                dst2: 2,
                coords: [4, 5, 6],
                dref: Some(7),
                handle: 8,
                dim: TexDim::TCube,
                mask: [true; 4],
                f16: false,
            },
            ALWAYS,
        ),
        (Op::Exit, ALWAYS),
    ]);
    assert!(matches!(
        translate(&p),
        Err(Unsupported::TextureDimension { dim: TexDim::TCube })
    ));
}

#[test]
fn a_ptp_gather_keeps_each_texel_of_the_gather_its_own_offset_moved() {
    let source = gather_module(&[gather(TexDim::T2d, None, Some([7, 8]), None, None)]);
    let lines: Vec<&str> = source.lines().map(str::trim).collect();
    let line_of = |name: &str| {
        let head = format!("let {name} = ");
        *lines
            .iter()
            .find(|line| line.starts_with(&head))
            .unwrap_or_else(|| panic!("no {name}: {source}"))
    };
    let gathers: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| line.contains("= texGather(8u, 1u, "))
        .collect();
    assert_eq!(gathers.len(), 4, "{source}");
    // Gather `i` moves by the offsets in `[7, 8][i / 2]`, `16 * (i % 2)` bits up, `y` a byte
    // above `x`, and the result keeps lane `i` of it.
    let mut names = Vec::new();
    for (i, gather) in gathers.iter().enumerate() {
        let (reg, shift) = (7 + i / 2, 16 * (i % 2));
        let name = gather["let ".len()..].split(' ').next().unwrap();
        let args: Vec<&str> = gather
            .split("texGather(8u, 1u, ")
            .nth(1)
            .unwrap()
            .split(", ")
            .collect();
        for (arg, bits) in [(args[0], shift), (args[1], shift + 8)] {
            let moved = format!("extractBits(bitcast<i32>(r{reg}), {bits}u, 6u)");
            assert!(line_of(arg).contains(&moved), "gather {i}: {source}");
        }
        names.push(name);
    }
    let kept = format!(
        "vec4<f32>({}.x, {}.y, {}.z, {}.w)",
        names[0], names[1], names[2], names[3]
    );
    assert!(source.contains(&kept), "{source}");
}

#[test]
fn a_bindless_gather_keys_its_slot_by_the_constant_the_handle_came_from() {
    let p = program(&[
        (
            Op::Mov {
                dst: 12,
                src: Operand::Const {
                    bank: 3,
                    offset: 0x20,
                },
            },
            ALWAYS,
        ),
        (gather(TexDim::T2d, None, None, None, Some(12)), ALWAYS),
        (Op::Exit, ALWAYS),
    ]);
    let translated = translate(&p).unwrap();
    let slot = TextureSlot::Bindless {
        bank: 3,
        offset: 0x20,
    };
    assert_eq!(translated.textures, [(slot, TexDim::T2d, false)]);
    let layout = Layout::of(&translated, Stage::Fragment);
    let source = module(&translated, Stage::Fragment, &layout).unwrap();
    assert!(
        source.contains(&format!("texGather({}u, 1u, ", slot.key())),
        "{source}"
    );
}

#[test]
fn a_tld4_clears_the_predicate_its_sparse_field_names() {
    let tld4 = |sparse| Op::Tld4 {
        dst: 0,
        coords: [4, 5, 6],
        layer: None,
        offset: None,
        ptp: None,
        dref: None,
        handle: 8,
        handle_reg: None,
        sparse,
        dim: TexDim::T2d,
        component: 1,
        mask: [true; 4],
    };
    assert!(gather_module(&[tld4(3)]).contains("p3 = false;"));
    let other = gather_module(&[tld4(7)]);
    let cleared = |source: &str| {
        source
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with('p') && line.ends_with(" = false;"))
            .count()
    };
    assert_eq!(cleared(&other), 0, "{other}");
}

#[test]
fn a_global_load_store_or_atomic_outside_its_buffer_reads_zero_and_changes_nothing() {
    let load = fragment_module(
        &through_descriptor(Op::Ldg {
            dst: 2,
            addr: 0,
            offset: 0,
            size: MemSize::B32,
        })
        .unwrap(),
    );
    assert!(
        load.contains("if (word >= arrayLength(&g0)) { return 0u; }"),
        "{load}"
    );

    let store = fragment_module(
        &through_descriptor(Op::Stg {
            addr: 0,
            offset: 0,
            src: 2,
            size: MemSize::B32,
        })
        .unwrap(),
    );
    assert!(store.contains("if (word < arrayLength(&g0))"), "{store}");
    assert!(store.contains("word + 1u < arrayLength(&g0)"), "{store}");

    let atom = fragment_module(
        &through_descriptor(Op::Atom {
            dst: 3,
            addr: 0,
            offset: 0,
            src: 2,
            op: AtomOp::Or,
            ty: AtomType::U32,
            space: AtomSpace::Global,
        })
        .unwrap(),
    );
    assert!(atom.contains("if (word < arrayLength(&g0))"), "{atom}");
}

/// Persona 5 Royal's skinning builds each address into other registers and copies the pair
/// into place; only an unconditional copy of an unchanged source counts.
#[test]
fn an_address_copied_into_place_still_reads_its_descriptor() {
    let iadd = |dst, a, offset, cin, cout| Op::Iadd {
        dst,
        a,
        aneg: false,
        b: Operand::Const { bank: 0, offset },
        bneg: false,
        cin,
        cout,
    };
    let mov = |dst, src| Op::Mov {
        dst,
        src: Operand::Reg(src),
    };
    let ldg = Op::Ldg {
        dst: 23,
        addr: 28,
        offset: 0,
        size: MemSize::B32,
    };
    let copied = |mov_pred: Pred, clobber: bool| {
        let mut entries = vec![
            (iadd(23, 23, 0x330, false, true), ALWAYS),
            (iadd(14, RZ, 0x334, true, false), ALWAYS),
            (mov(29, 14), ALWAYS),
            (mov(28, 23), mov_pred),
        ];
        if clobber {
            entries.insert(2, (Op::Mov32i { dst: 23, imm: 0x40 }, ALWAYS));
        }
        entries.extend([(ldg, ALWAYS), (Op::Exit, ALWAYS)]);
        translate(&program(&entries))
    };
    assert_eq!(
        copied(ALWAYS, false).unwrap().globals,
        vec![Global {
            bank: 0,
            offset: 0x330,
            written: false
        }]
    );
    assert!(copied(IF_P0, false).is_err(), "a copy that may not happen");
    assert!(
        copied(ALWAYS, true).is_err(),
        "a source rewritten before the copy"
    );
}

#[test]
fn window_coordinates_are_turned_into_clip_space_before_y_is_flipped() {
    let (vs, _) = pair(3);
    let translated = translate(&vs).unwrap();
    let mut layout = Layout::of(&translated, Stage::Vertex);
    layout.flip_y = true;
    let plain = module(&translated, Stage::Vertex, &layout).unwrap();
    assert!(!plain.contains("- out.position.w"));

    layout.window_coordinates = Some((1280, 720));
    let source = module(&translated, Stage::Vertex, &layout).unwrap();
    let x = source
        .find("out.position.x = out.position.x * 0.0015625 - out.position.w;")
        .expect("x scaled by 2/1280");
    let y = source
        .find("out.position.y = out.position.y * 0.0027777778 - out.position.w;")
        .expect("y scaled by 2/720");
    let flip = source.find("out.position.y = -out.position.y;").unwrap();
    assert!(x < flip && y < flip, "converted, then mirrored");
}
