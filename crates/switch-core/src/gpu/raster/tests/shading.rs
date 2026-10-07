//! Interpolation, depth, blend and vertex shader tests.

use super::super::attrib::ATTRIB_TYPE_FLOAT;
use super::super::fragment::{
    blend, blend_equation, blend_factor, depth_test_passes, source_color,
};
use super::super::vertex::{
    shade_vertex, CLIP_POS_OFFSET, INSTANCE_ID_OFFSET, VARYING_BASE, VERTEX_ID_OFFSET,
};
use super::attrib::harness;
use super::draw::pipeline_harness;
use super::*;
use crate::gpu::engine::threed::{BlendTarget, VertexArray, VertexAttrib};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::interp::MemoryGlobal;
use crate::gpu::shader::Program;
use crate::gpu::surface::ColorFormat;

#[test]
fn three_vertex_colours_interpolate_correctly_at_a_known_interior_point() {
    let (mut mem, vmm, engine) = pipeline_harness();
    let vbuf_addr = engine.vertex_array(0).start;
    let red = [1.0f32, 0.0, 0.0, 1.0];
    let green = [0.0f32, 1.0, 0.0, 1.0];
    let blue = [0.0f32, 0.0, 1.0, 1.0];
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.0, 1.0], red);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.0, 1.0], green);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.0, 1.0], blue);

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
    // Pixel (4,2), centre (4.5,2.5): barycentric weights against
    // (0,0)-(16,0)-(0,8) are (0.40625, 0.28125, 0.3125).
    let expected_color = [
        0.40625 * red[0] + 0.28125 * green[0] + 0.3125 * blue[0],
        0.40625 * red[1] + 0.28125 * green[1] + 0.3125 * blue[1],
        0.40625 * red[2] + 0.28125 * green[2] + 0.3125 * blue[2],
        1.0,
    ];
    let expected = rt.format.encode(expected_color).unwrap();
    let va = rt.addr + rt.layout.offset(4 * 4, 2, 16 * 4) as u64;
    assert_eq!(ctx.read_u32(va).unwrap() as u128, expected);
}

#[test]
fn depth_test_passes_matches_gl_comparison_op() {
    assert!(!depth_test_passes(0x0200, 0.1, 0.5)); // Never
    assert!(depth_test_passes(0x0201, 0.1, 0.5)); // Less
    assert!(!depth_test_passes(0x0201, 0.5, 0.5));
    assert!(depth_test_passes(0x0202, 0.5, 0.5)); // Equal
    assert!(depth_test_passes(0x0203, 0.5, 0.5)); // Lequal
    assert!(depth_test_passes(0x0204, 0.6, 0.5)); // Greater
    assert!(depth_test_passes(0x0205, 0.6, 0.5)); // NotEqual
    assert!(depth_test_passes(0x0206, 0.5, 0.5)); // Gequal
    assert!(depth_test_passes(0x0207, 0.9, 0.1)); // Always
}

#[test]
fn blend_factor_reads_src_and_dst_color_and_alpha() {
    let src = [1.0, 0.5, 0.25, 0.75];
    let dst = [0.0, 1.0, 0.0, 0.2];
    let constant = [0.1, 0.2, 0.3, 0.4];
    assert_eq!(blend_factor(0x4000, src, dst, constant), [0.0; 4]); // Zero
    assert_eq!(blend_factor(0x4001, src, dst, constant), [1.0; 4]); // One
    assert_eq!(blend_factor(0x4300, src, dst, constant), src); // SrcColor
    assert_eq!(blend_factor(0x4302, src, dst, constant), [0.75; 4]); // SrcAlpha
    assert_eq!(blend_factor(0x4303, src, dst, constant), [0.25; 4]); // OneMinusSrcAlpha
    assert_eq!(blend_factor(0x4306, src, dst, constant), dst); // DstColor
    assert_eq!(blend_factor(0xc001, src, dst, constant), constant); // ConstantColor
}

#[test]
fn a_fixed_point_target_clamps_the_blend_source_and_a_float_one_does_not() {
    // A NaN source must not reach a fixed-point target through a zero alpha.
    let unorm = ColorFormat::from_raw(0xD5).unwrap(); // RGBA8Unorm
    let float = ColorFormat::from_raw(0xCA).unwrap(); // RGBA16Float
    let color = [f32::NAN, 2.0, -1.0, 0.5];
    assert_eq!(source_color(color, unorm), [0.0, 1.0, 0.0, 0.5]);
    let through = source_color(color, float);
    assert!(through[0].is_nan());
    assert_eq!(&through[1..], &[2.0, -1.0, 0.5]);

    // A transparent NaN leaves an opaque background alone.
    let target = BlendTarget {
        enabled: true,
        equation_rgb: 0x8006, // FuncAdd
        func_rgb_src: 0x4302, // SrcAlpha
        func_rgb_dst: 0x4303, // OneMinusSrcAlpha
        equation_alpha: 0x8006,
        func_alpha_src: 0x4302,
        func_alpha_dst: 0x4303,
    };
    let dst = [0.9, 0.9, 0.9, 1.0];
    let src = source_color([f32::NAN, f32::NAN, f32::NAN, 0.0], unorm);
    assert_eq!(blend(target, [0.0; 4], src, dst), dst);
}

/// A vertex shader's stores land in guest memory, visible to later loads.
#[test]
fn a_vertex_shader_stores_to_global_memory() {
    use crate::gpu::shader::isa::{Instruction, MemSize, Op, Pred, RZ};

    let (mut mem, vmm, base) = harness();
    let ops = [
        Op::Ld {
            dst: 0,
            offset: VERTEX_ID_OFFSET,
            idx: RZ,
            size: MemSize::B32,
        },
        Op::Mov32i {
            dst: 2,
            imm: base as u32,
        },
        Op::Mov32i {
            dst: 3,
            imm: (base >> 32) as u32,
        },
        Op::Stg {
            addr: 2,
            offset: 0,
            src: 0,
            size: MemSize::B32,
        },
        Op::Ldg {
            dst: 4,
            addr: 2,
            offset: 0,
            size: MemSize::B32,
        },
        Op::Stg {
            addr: 2,
            offset: 4,
            src: 4,
            size: MemSize::B32,
        },
        Op::Exit,
    ];
    let mut program = Program::default();
    for (i, op) in ops.into_iter().enumerate() {
        program.offsets.push(8 + i as u32 * 8);
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
    }
    let program = Compiled::new(&program);

    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let mut ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let consts: std::collections::HashMap<(u8, u16), f32> = Default::default();
    let stores = std::cell::RefCell::new(Vec::new());
    shade_vertex(&program, &[], &[], (7, 0), &ctx, &consts, false, &stores).unwrap();
    MemoryGlobal::land(&mut ctx, &stores).unwrap();
    assert_eq!(ctx.read_u32(base).unwrap(), 7);
    assert_eq!(ctx.read_u32(base + 4).unwrap(), 7, "the load saw the store");
}

#[test]
fn a_vertex_shader_reads_its_vertex_and_instance_ids() {
    // `gl_InstanceID` and `gl_VertexID` reach the register as integer bits.
    use crate::gpu::shader::isa::{Instruction, MemSize, Op, Pred, RZ};

    let mut program = Program::default();
    for (at, op) in [
        (
            8u32,
            Op::Ld {
                dst: 0,
                offset: INSTANCE_ID_OFFSET,
                idx: RZ,
                size: MemSize::B32,
            },
        ),
        (
            16,
            Op::Ld {
                dst: 1,
                offset: VERTEX_ID_OFFSET,
                idx: RZ,
                size: MemSize::B32,
            },
        ),
        (
            24,
            Op::St {
                offset: CLIP_POS_OFFSET,
                idx: RZ,
                src: 0,
                size: MemSize::B32,
            },
        ),
        (
            32,
            Op::St {
                offset: CLIP_POS_OFFSET + 4,
                idx: RZ,
                src: 1,
                size: MemSize::B32,
            },
        ),
        (40, Op::Exit),
    ] {
        program.offsets.push(at);
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
    }

    let (mut mem, vmm, _) = harness();
    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let consts: std::collections::HashMap<(u8, u16), f32> = Default::default();

    let program = Compiled::new(&program);
    let stores = std::cell::RefCell::new(Vec::new());
    let v = shade_vertex(&program, &[], &[], (7, 42), &ctx, &consts, false, &stores).unwrap();
    assert_eq!(v.clip[0].to_bits(), 42, "gl_InstanceID");
    assert_eq!(v.clip[1].to_bits(), 7, "gl_VertexID");
}

#[test]
fn an_instanced_vertex_array_advances_with_the_instance_not_the_vertex() {
    let (mut mem, vmm, base) = harness();
    // Four consecutive floats, one per element of a divisor-2 array.
    for (i, v) in [10.0f32, 20.0, 30.0, 40.0].iter().enumerate() {
        vmm.write_u32(&mut mem, base + i as u64 * 4, v.to_bits())
            .unwrap();
    }
    use crate::gpu::shader::isa::{Instruction, MemSize, Op, Pred, RZ};

    // Copy attribute 0 into the clip position's x.
    let mut program = Program::default();
    for (at, op) in [
        (
            8u32,
            Op::Ld {
                dst: 0,
                offset: VARYING_BASE,
                idx: RZ,
                size: MemSize::B32,
            },
        ),
        (
            16,
            Op::St {
                offset: CLIP_POS_OFFSET,
                idx: RZ,
                src: 0,
                size: MemSize::B32,
            },
        ),
        (24, Op::Exit),
    ] {
        program.offsets.push(at);
        program.insns.push(Instruction {
            pred: Pred::ALWAYS,
            op,
        });
    }
    let program = Compiled::new(&program);
    let consts: std::collections::HashMap<(u8, u16), f32> = Default::default();
    let stores = std::cell::RefCell::new(Vec::new());

    let mut stats = Default::default();
    let mut host1x = Host1x::new();
    let ctx = ExecCtx {
        mem: &mut mem,
        vmm: &vmm,
        host1x: &mut host1x,
        stats: &mut stats,
        trace: false,
    };
    let attrib = VertexAttrib {
        buffer_id: 0,
        is_fixed: false,
        offset: 0,
        size: 0x12, // 1x32
        ty: ATTRIB_TYPE_FLOAT,
        is_bgra: false,
    };
    let array = VertexArray {
        enabled: true,
        stride: 4,
        start: base,
        limit: base + 0x1000,
        divisor: 2,
    };

    // Instances 0 and 1 share element 0; instances 2 and 3 share element 1,
    // whatever the vertex.
    for (instance, expected) in [(0u32, 10.0f32), (1, 10.0), (2, 20.0), (3, 20.0)] {
        let v = shade_vertex(
            &program,
            &[attrib],
            &[array],
            (3, instance),
            &ctx,
            &consts,
            false,
            &stores,
        )
        .unwrap();
        assert_eq!(v.clip[0], expected, "instance {instance}");
    }
}

#[test]
fn the_same_blend_state_composites_the_same_in_either_numbering() {
    // The D3D blend numbering.
    let gl = BlendTarget {
        enabled: true,
        equation_rgb: 0x8006,
        func_rgb_src: 0x4302,
        func_rgb_dst: 0x4303,
        equation_alpha: 0x8006,
        func_alpha_src: 0x4001,
        func_alpha_dst: 0x4000,
    };
    let d3d = BlendTarget {
        enabled: true,
        equation_rgb: 1,   // Add
        func_rgb_src: 5,   // SrcAlpha
        func_rgb_dst: 6,   // OneMinusSrcAlpha
        equation_alpha: 1, // Add
        func_alpha_src: 2, // One
        func_alpha_dst: 1, // Zero
    };
    let src = [1.0, 0.0, 0.0, 0.5];
    let dst = [0.0, 0.0, 1.0, 1.0];
    assert_eq!(blend(d3d, [0.0; 4], src, dst), [0.5, 0.0, 0.5, 0.5]);
    assert_eq!(
        blend(d3d, [0.0; 4], src, dst),
        blend(gl, [0.0; 4], src, dst)
    );
}

#[test]
fn blend_factor_reads_the_d3d_numbering_too() {
    let src = [1.0, 0.5, 0.25, 0.75];
    let dst = [0.0, 1.0, 0.0, 0.2];
    let constant = [0.1, 0.2, 0.3, 0.4];
    for (d3d, gl) in [
        (0x01u32, 0x4000u32), // Zero
        (0x02, 0x4001),       // One
        (0x03, 0x4300),       // SrcColor
        (0x04, 0x4301),       // OneMinusSrcColor
        (0x05, 0x4302),       // SrcAlpha
        (0x06, 0x4303),       // OneMinusSrcAlpha
        (0x07, 0x4304),       // DstAlpha
        (0x08, 0x4305),       // OneMinusDstAlpha
        (0x09, 0x4306),       // DstColor
        (0x0a, 0x4307),       // OneMinusDstColor
        (0x0b, 0x4308),       // SrcAlphaSaturate
        (0x61, 0xc001),       // ConstantColor
        (0x62, 0xc002),       // OneMinusConstantColor
        (0x63, 0xc003),       // ConstantAlpha
        (0x64, 0xc004),       // OneMinusConstantAlpha
    ] {
        assert_eq!(
            blend_factor(d3d, src, dst, constant),
            blend_factor(gl, src, dst, constant),
            "factor {d3d:#x} and {gl:#x} name the same thing"
        );
    }
    // SrcAlphaSaturate is min(srcA, 1 - dstA) on colour and 1 on alpha.
    assert_eq!(
        blend_factor(0x0b, src, dst, constant),
        [0.75, 0.75, 0.75, 1.0]
    );
}

#[test]
fn blend_equation_reads_the_d3d_numbering_too() {
    for (d3d, gl) in [
        (1u32, 0x8006u32),
        (2, 0x800a),
        (3, 0x800b),
        (4, 0x8007),
        (5, 0x8008),
    ] {
        assert_eq!(
            blend_equation(d3d, 0.75, 0.25),
            blend_equation(gl, 0.75, 0.25),
            "equation {d3d:#x} and {gl:#x} name the same thing"
        );
    }
}

#[test]
fn a_depth_only_pass_draws_without_a_colour_target() {
    // A depth surface bound as colour target 0: the draw must still write depth.
    let (mut mem, vmm, mut engine) = pipeline_harness();
    let vbuf_addr = engine.vertex_array(0).start;
    let rt_addr = engine.regs.iova(0x200);
    let depth_addr = vbuf_addr + 0x200; // past the vertex buffer, still mapped.

    engine.regs.set(0x204, 0x14); // colour target 0 in Z24S8, as the title binds it
    engine.regs.set(0x3F8, (depth_addr >> 32) as u32);
    engine.regs.set(0x3F9, depth_addr as u32);
    engine.regs.set(0x3FA, 0x0A); // Z32Float
    engine.regs.set(0x3FB, 0); // block_height_gobs = 1
    engine.regs.set(0x48A, 16);
    engine.regs.set(0x48B, 8);
    engine.regs.set(0x4B3, 1); // DepthTestEnable
    engine.regs.set(0x4BA, 1); // DepthWriteEnable
    engine.regs.set(0x4C3, 0x0201); // DepthTestFunc = GL_LESS

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
        engine.regs.set(0x364, 1.0f32.to_bits()); // CLEAR_DEPTH = far
        engine.write(0x674, 0b1, true, &mut ctx).unwrap();
    }

    let color = [0.2f32, 0.8, 0.2, 1.0];
    for (i, pos) in [
        [-1.0f32, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    {
        write_vertex(&mut mem, &vmm, vbuf_addr, i as u32, pos, color);
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
    assert!(
        engine.render_target(0).unwrap().is_none(),
        "no colour target is bound"
    );
    draw(&engine, &mut ctx).unwrap();

    // NDC z = 0 is window z = 0.5 through this viewport, and it passes
    // GL_LESS against the 1.0 the clear left.
    assert_eq!(f32::from_bits(ctx.read_u32(depth_addr).unwrap()), 0.5);
    // And nothing was written where a colour target would have been.
    assert_eq!(ctx.read_u32(rt_addr).unwrap(), 0);
}

#[test]
fn depth_test_keeps_the_nearer_of_two_overlapping_triangles() {
    let (mut mem, vmm, mut engine) = pipeline_harness();
    let vbuf_addr = engine.vertex_array(0).start;
    let depth_addr = vbuf_addr + 0x200; // past the vertex buffer, still inside the mapped region.

    engine.regs.set(0x3F8, (depth_addr >> 32) as u32);
    engine.regs.set(0x3F9, depth_addr as u32);
    engine.regs.set(0x3FA, 0x0A); // Z32Float
    engine.regs.set(0x3FB, 0); // block_height_gobs = 1
    engine.regs.set(0x48A, 16);
    engine.regs.set(0x48B, 8);
    engine.regs.set(0x4B3, 1); // DepthTestEnable
    engine.regs.set(0x4BA, 1); // DepthWriteEnable
    engine.regs.set(0x4C3, 0x0201); // DepthTestFunc = GL_LESS

    {
        // Clear depth to 1.0 first; a zeroed buffer rejects every draw.
        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let mut ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };
        engine.regs.set(0x364, 1.0f32.to_bits()); // CLEAR_DEPTH
        engine.write(0x674, 0b1, true, &mut ctx).unwrap(); // CLEAR_BUFFERS: clear_depth
    }

    let near = [0.2f32, 0.8, 0.2, 1.0];
    let far = [0.8f32, 0.2, 0.2, 1.0];

    // Far triangle first, at NDC z = 0.5 (covers the whole target).
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.5, 1.0], far);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.5, 1.0], far);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.5, 1.0], far);
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
        draw(&engine, &mut ctx).unwrap();
    }
    // Near triangle second, at NDC z = -0.5 (closer, smaller depth).
    write_vertex(&mut mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, -0.5, 1.0], near);
    write_vertex(&mut mem, &vmm, vbuf_addr, 1, [1.0, 1.0, -0.5, 1.0], near);
    write_vertex(&mut mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, -0.5, 1.0], near);
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
    let expected = rt.format.encode(near).unwrap();
    assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);

    // A third far draw must not overwrite the nearer surface.
    write_vertex(ctx.mem, &vmm, vbuf_addr, 0, [-1.0, 1.0, 0.5, 1.0], far);
    write_vertex(ctx.mem, &vmm, vbuf_addr, 1, [1.0, 1.0, 0.5, 1.0], far);
    write_vertex(ctx.mem, &vmm, vbuf_addr, 2, [-1.0, -1.0, 0.5, 1.0], far);
    draw(&engine, &mut ctx).unwrap();
    assert_eq!(ctx.read_u32(rt.addr).unwrap() as u128, expected);
}
