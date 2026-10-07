//! Full-pipeline draw tests.

use super::super::draw::{clip_near, culls, ClipVertex};
use super::super::setup::alpha_coverage;
use super::super::vertex::NUM_VARYINGS;
use super::*;
use crate::gpu::engine::threed::{CullState, Engine3D};
use crate::gpu::exec::ExecCtx;

// Full-pipeline integration tests.

/// [`Harness`] split into the pieces these tests borrow separately.
pub(super) fn pipeline_harness() -> (Memory, AddressSpace, Engine3D) {
    pipeline_harness_with(solid_fragment_shader())
}

fn pipeline_harness_with(fragment_shader: Vec<u8>) -> (Memory, AddressSpace, Engine3D) {
    let h = Harness::with_fragment_shader(fragment_shader);
    (h.mem, h.vmm, h.engine)
}

#[test]
fn a_solid_colour_triangle_matches_a_clear_color_equivalent_fill() {
    let (mut mem, vmm, engine) = pipeline_harness();
    let vbuf_addr = engine.vertex_array(0).start;
    let color = [0.2f32, 0.4, 0.6, 1.0];
    // Screen (0,0)-(16,0)-(0,8), w = 1: the upper triangle half.
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: true,
    };
    draw(&engine, &mut ctx).unwrap();

    let rt = engine.render_target(0).unwrap().unwrap();
    let expected = rt.format.encode(color).unwrap();
    // (12,6) is in the untouched half.
    assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);
    assert_eq!(
        ctx.read_u32(rt.addr + rt.layout.offset(2 * 4, 2, 16 * 4) as u64)
            .unwrap() as u128,
        expected
    );
    assert_eq!(
        ctx.read_u32(rt.addr + rt.layout.offset(12 * 4, 6, 16 * 4) as u64)
            .unwrap(),
        0
    );
}

/// A warp shuffle reads the neighbouring lane's register.
#[test]
fn a_shuffling_fragment_shader_reads_the_pixel_beside_it() {
    let (mut mem, vmm, engine) = pipeline_harness_with(derivative_fragment_shader());
    let vbuf_addr = engine.vertex_array(0).start;
    // Red is `(x + 0.5) / 16`, so neighbours differ by 1/16.
    write_vertex(
        &mut mem,
        &vmm,
        vbuf_addr,
        0,
        [-1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    );
    write_vertex(
        &mut mem,
        &vmm,
        vbuf_addr,
        1,
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    );
    write_vertex(
        &mut mem,
        &vmm,
        vbuf_addr,
        2,
        [-1.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    );

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    draw(&engine, &mut ctx).unwrap();

    let rt = engine.render_target(0).unwrap().unwrap();
    let pixel = |ctx: &mut ExecCtx, x: u32, y: u32| {
        let raw = ctx
            .read_pixel(rt.addr + rt.texel_offset(x, y) as u64, 4)
            .unwrap();
        rt.format.decode(raw).unwrap()
    };
    let close = |got: f32, want: f32| (got - want).abs() < 1.0 / 255.0;

    // Lane 0 reads lane 1, half a pixel further along in each direction.
    let left = pixel(&mut ctx, 0, 0);
    assert!(close(left[0], 1.0 / 16.0), "dFdx at (0,0): {left:?}");
    assert!(
        close(left[1], 1.5 / 16.0),
        "the neighbour's own value: {left:?}"
    );
    // Lane 1 differences the other way; negative clamps to zero.
    let right = pixel(&mut ctx, 1, 0);
    assert!(close(right[0], 0.0), "dFdx at (1,0): {right:?}");
    assert!(
        close(right[1], 0.5 / 16.0),
        "the neighbour's own value: {right:?}"
    );
}

/// The colour write mask keeps the channels it turns off, and a mask with
/// nothing set writes no colour at all.
#[test]
fn a_masked_channel_keeps_what_the_target_already_held() {
    let full = [0.2f32, 0.4, 0.6, 0.25];
    // Whatever the target held before the draw, in the same format.
    let before = [1.0f32, 1.0, 1.0, 1.0];

    // `0x1111` is every channel, `0x0111` drops alpha, `0` drops all four.
    for (mask, expected) in [
        (0x1111u32, Some([full[0], full[1], full[2], full[3]])),
        (0x0111, Some([full[0], full[1], full[2], before[3]])),
        (0x1000, Some([before[0], before[1], before[2], full[3]])),
        (0, None),
    ] {
        let (mut mem, vmm, mut engine) = pipeline_harness();
        engine.regs.set(0x680, mask);
        let vbuf_addr = engine.vertex_array(0).start;
        for (i, pos) in [
            [-1.0f32, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        {
            write_vertex(&mut mem, &vmm, vbuf_addr, i as u32, pos, full);
        }

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        let rt = engine.render_target(0).unwrap().unwrap();
        let held = rt.format.encode(before).unwrap();
        ctx.write_pixel(rt.addr, rt.format.bytes_per_pixel, held)
            .unwrap();

        draw(&engine, &mut ctx).unwrap();

        let want = match expected {
            Some(colour) => rt.format.encode(colour).unwrap(),
            // Nothing written: the target still holds what it did.
            None => held,
        };
        assert_eq!(
            ctx.read_u32(rt.addr).unwrap() as u128,
            want,
            "mask {mask:#06x}"
        );
    }
}

/// Zero mask registers (never written) must still draw.
#[test]
fn an_unwritten_write_mask_lets_every_channel_through() {
    assert_eq!(Engine3D::new().color_mask(0), [true; 4]);
    assert_eq!(Engine3D::new().color_mask(7), [true; 4]);
}

#[test]
fn a_multisampled_edge_covers_some_samples_of_a_pixel_and_not_others() {
    // 8x4 pixels of 2x2 samples; the hypotenuse crosses pixel (7, 0).
    let (mut mem, vmm, mut engine) = pipeline_harness();
    engine.regs.set(0x300, 8 << 16); // viewport width, in pixels
    engine.regs.set(0x301, 4 << 16); // viewport height, in pixels
    engine.regs.set(0x54D, 1); // MultisampleEnable
    engine.regs.set(0x574, 2); // MultisampleMode = 2x2
    let vbuf_addr = engine.vertex_array(0).start;
    let color = [1.0f32, 1.0, 1.0, 1.0];
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    draw(&engine, &mut ctx).unwrap();

    let rt = engine.render_target(0).unwrap().unwrap();
    let texel = |ctx: &ExecCtx, x: u32, y: u32| {
        ctx.read_u32(rt.addr + rt.texel_offset(x, y) as u64)
            .unwrap()
    };
    // Pixel (0, 0) is wholly inside: all four of its texels are written.
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert_ne!(texel(&ctx, x, y), 0, "texel ({x}, {y}) of a covered pixel");
    }
    // Pixel (7, 0) straddles the edge; only texel (14, 0) is inside.
    assert_ne!(texel(&ctx, 14, 0), 0, "the covered sample of pixel (7, 0)");
    for (x, y) in [(15, 0), (14, 1), (15, 1)] {
        assert_eq!(texel(&ctx, x, y), 0, "texel ({x}, {y}) is outside the edge");
    }
    // Pixel (7, 3) is wholly outside.
    for (x, y) in [(14, 6), (15, 6), (14, 7), (15, 7)] {
        assert_eq!(
            texel(&ctx, x, y),
            0,
            "texel ({x}, {y}) of an uncovered pixel"
        );
    }
}

/// 8x4 pixels of 2x2 samples, pixel (0, 0) wholly covered.
fn multisampled_harness() -> (Memory, AddressSpace, Engine3D) {
    let (mut mem, vmm, mut engine) = pipeline_harness();
    engine.regs.set(0x300, 8 << 16); // viewport width, in pixels
    engine.regs.set(0x301, 4 << 16); // viewport height, in pixels
    engine.regs.set(0x54D, 1); // MultisampleEnable
    engine.regs.set(0x574, 2); // MultisampleMode = 2x2
    let vbuf_addr = engine.vertex_array(0).start;
    let color = [1.0f32, 1.0, 1.0, 1.0];
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);
    (mem, vmm, engine)
}

#[test]
fn a_sample_mask_keeps_only_the_samples_it_names() {
    let (mut mem, vmm, mut engine) = multisampled_harness();
    engine.regs.set(0x3EF, 0b0001); // MultisampleSampleMask: sample 0 only

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    draw(&engine, &mut ctx).unwrap();

    let rt = engine.render_target(0).unwrap().unwrap();
    let texel = |ctx: &ExecCtx, x: u32, y: u32| {
        ctx.read_u32(rt.addr + rt.texel_offset(x, y) as u64)
            .unwrap()
    };
    assert_ne!(texel(&ctx, 0, 0), 0, "sample 0 is in the mask");
    for (x, y) in [(1, 0), (0, 1), (1, 1)] {
        assert_eq!(texel(&ctx, x, y), 0, "texel ({x}, {y}) is masked off");
    }
}

#[test]
fn alpha_to_coverage_turns_half_alpha_into_half_the_samples() {
    let (mut mem, vmm, mut engine) = multisampled_harness();
    engine.regs.set(0x54F, 1); // MultisampleControl: AlphaToCoverage
    let vbuf_addr = engine.vertex_array(0).start;
    let color = [1.0f32, 1.0, 1.0, 0.5];
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    draw(&engine, &mut ctx).unwrap();

    // Half alpha keeps two of four samples.
    let rt = engine.render_target(0).unwrap().unwrap();
    let texel = |ctx: &ExecCtx, x: u32, y: u32| {
        ctx.read_u32(rt.addr + rt.texel_offset(x, y) as u64)
            .unwrap()
    };
    assert_ne!(texel(&ctx, 0, 0), 0);
    assert_ne!(texel(&ctx, 1, 0), 0);
    assert_eq!(texel(&ctx, 0, 1), 0);
    assert_eq!(texel(&ctx, 1, 1), 0);
}

#[test]
fn alpha_to_coverage_spans_none_to_all_of_the_samples() {
    assert_eq!(alpha_coverage(0.0, 4), 0);
    assert_eq!(alpha_coverage(0.25, 4), 0b0001);
    assert_eq!(alpha_coverage(0.5, 4), 0b0011);
    assert_eq!(alpha_coverage(1.0, 4), u32::MAX);
    // Out-of-range alpha clamps rather than shifting past the sample count.
    assert_eq!(alpha_coverage(2.0, 4), u32::MAX);
    assert_eq!(alpha_coverage(-1.0, 4), 0);
    assert_eq!(alpha_coverage(1.0, 16), u32::MAX);
}

/// A draw honours RenderTargetControl's mapping, as a clear does.
#[test]
fn a_draw_follows_the_render_target_control_mapping() {
    let (mut mem, vmm, mut engine) = pipeline_harness();
    let slot0 = engine.render_target(0).unwrap().unwrap().addr;
    let slot1 = slot0 + 0x800;
    // Bind a second target in physical slot 1, same 16x8 pitch-linear form.
    engine.regs.set(0x210, (slot1 >> 32) as u32);
    engine.regs.set(0x211, slot1 as u32);
    engine.regs.set(0x212, 16 * 4);
    engine.regs.set(0x213, 8);
    engine.regs.set(0x214, 0xD5);
    engine.regs.set(0x215, 1 << 12);
    engine.regs.set(0x216, 1);
    // One target in use, and logical 0 maps onto physical slot 1.
    engine.regs.set(0x487, 1 | (1 << 4));
    assert_eq!(engine.render_target_slot(0), 1);

    let vbuf_addr = engine.vertex_array(0).start;
    let color = [1.0f32, 1.0, 1.0, 1.0];
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    draw(&engine, &mut ctx).unwrap();

    let expected = engine
        .render_target(0)
        .unwrap()
        .unwrap()
        .format
        .encode(color);
    assert_eq!(
        ctx.read_u32(slot1).unwrap() as u128,
        expected.unwrap(),
        "the draw belongs in slot 1"
    );
    assert_eq!(
        ctx.read_u32(slot0).unwrap(),
        0,
        "slot 0 is not the mapped target"
    );
}

#[test]
fn an_indexed_draw_reads_its_vertices_through_the_index_buffer() {
    // The same half, vertices reversed and put back by an index buffer.
    let (mut mem, vmm, mut engine) = pipeline_harness();
    let vbuf_addr = engine.vertex_array(0).start;
    let color = [0.2f32, 0.4, 0.6, 1.0];
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, -1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, 1.0, 0.0, 1.0], color);

    // A u16 index buffer of [2, 1, 0], right after the vertex data.
    let ibuf_addr = vbuf_addr + 3 * 32;
    {
        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        ctx.write_u32(ibuf_addr, 2 | (1 << 16)).unwrap();
        ctx.write_u32(ibuf_addr + 4, 0).unwrap();
    }
    engine.regs.set(0x5F2, (ibuf_addr >> 32) as u32);
    engine.regs.set(0x5F3, ibuf_addr as u32);
    engine.last_draw = DrawCall {
        primitive: 4,
        first: 0,
        count: 3,
        indexed: true,
        index_format: 1,
    };

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: true,
    };
    draw(&engine, &mut ctx).unwrap();

    let rt = engine.render_target(0).unwrap().unwrap();
    let expected = rt.format.encode(color).unwrap();
    assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);
    assert_eq!(
        ctx.read_u32(rt.addr + rt.layout.offset(2 * 4, 2, 16 * 4) as u64)
            .unwrap() as u128,
        expected
    );
    assert_eq!(
        ctx.read_u32(rt.addr + rt.layout.offset(12 * 4, 6, 16 * 4) as u64)
            .unwrap(),
        0
    );
}

#[test]
fn back_face_culling_drops_the_wrongly_wound_triangle() {
    let (mut mem, vmm, mut engine) = pipeline_harness();
    let vbuf_addr = engine.vertex_array(0).start;
    let color = [0.2f32, 0.4, 0.6, 1.0];
    // Clockwise in window space through the mirroring viewport: the back face.
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], color);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], color);

    // OGL_SET_CULL enable, front = CCW, cull = BACK.
    engine.regs.set(0x646, 1);
    engine.regs.set(0x647, 0x901);
    engine.regs.set(0x648, 0x405);

    let mut host1x = Host1x::new();
    let mut stats = Default::default();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: true,
    };
    draw(&engine, &mut ctx).unwrap();
    let rt = engine.render_target(0).unwrap().unwrap();
    assert_eq!(
        ctx.read_u32(rt.addr).unwrap(),
        0,
        "a back face must not be drawn"
    );

    // Culling the front face instead draws it.
    engine.regs.set(0x648, 0x404);
    draw(&engine, &mut ctx).unwrap();
    assert_eq!(
        ctx.read_u32(rt.addr).unwrap() as u128,
        rt.format.encode(color).unwrap()
    );
}

/// A face is judged by its winding in window space, whatever the viewport did.
#[test]
fn a_face_is_judged_by_its_winding_in_window_space() {
    let cull = |front_ccw| CullState {
        enabled: true,
        front_ccw,
        cull_front: false,
        cull_back: true,
    };
    // Counter-clockwise as the target holds it: down the left edge, then
    // across the top.
    let ccw = [
        ScreenVertex { x: 0.0, y: 0.0 },
        ScreenVertex { x: 0.0, y: 720.0 },
        ScreenVertex { x: 1280.0, y: 0.0 },
    ];
    // Clockwise: across the top, then down to the bottom-left corner.
    let cw = [ccw[0], ccw[2], ccw[1]];
    // Tomodachi Life: front=CCW, and its quad lands counter-clockwise.
    assert!(!culls(cull(true), ccw), "a front face survives");
    assert!(culls(cull(true), cw));
    // Echoes of Wisdom: front=CCW reversed by FlipY, so clockwise is
    // front, and its quad lands clockwise.
    assert!(!culls(cull(false), cw), "a front face survives");
    assert!(culls(cull(false), ccw));
}

#[test]
fn a_triangle_crossing_the_near_plane_is_clipped_not_projected() {
    // One vertex behind the eye.
    let far = ClipVertex {
        clip: [-1.0, 1.0, 0.0, 1.0],
        varyings: [[0.0; 4]; NUM_VARYINGS],
    };
    let also_far = ClipVertex {
        clip: [1.0, 1.0, 0.0, 1.0],
        varyings: [[0.0; 4]; NUM_VARYINGS],
    };
    let behind = ClipVertex {
        clip: [0.0, -1.0, 0.0, -1.0],
        varyings: [[0.0; 4]; NUM_VARYINGS],
    };

    let pieces = clip_near([far, also_far, behind]);
    assert_eq!(pieces.len(), 2, "a triangle cut by one plane fans into two");
    for piece in &pieces {
        for v in piece {
            assert!(
                v.clip[3] > 0.0,
                "no vertex may survive at or behind the eye"
            );
        }
    }

    // Wholly in front: untouched. Wholly behind: gone.
    assert_eq!(clip_near([far, also_far, far]).len(), 1);
    assert!(clip_near([behind, behind, behind]).is_empty());
}

#[test]
fn the_other_triangle_topologies_assemble() {
    assert_eq!(
        assemble(Primitive::TriangleFan, 5),
        vec![[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    );
    assert_eq!(assemble(Primitive::Quads, 4), vec![[0, 1, 2], [0, 2, 3]]);
    assert_eq!(
        assemble(Primitive::QuadStrip, 4),
        vec![[0, 1, 2], [2, 1, 3]]
    );
    assert!(assemble(Primitive::Lines, 6).is_empty());
    assert!(assemble(Primitive::Points, 6).is_empty());
}
