//! Device output against the software rasterizer.

use crate::tests::device;

use switch_core::gpu::renderer::{Renderer, Software, Submit};
use switch_core::gpu::testing::{self, Harness};

/// A solid white triangle over a multisample mode, by both renderers; only coverage can differ.
fn compare(mode: u32, samples_x: u32, samples_y: u32, set_up: impl Fn(&mut Harness)) {
    let Some(mut gpu) = device() else { return };
    let colour = [1.0f32, 1.0, 1.0, 1.0];

    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::new();
        h.multisample(mode, samples_x, samples_y);
        set_up(&mut h);
        h.triangle(colour);
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        h.target()
    };
    let want = build(None);
    let before = gpu.fallbacks;
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, before,
        "the draw did not run on the device: {:?}",
        gpu.last_fallback
    );
    // Surface any device rejection before comparing.
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(
        got, want,
        "mode {mode} ({samples_x}x{samples_y}) came out differently on the device"
    );
}

/// One draw, rendered by both renderers, colour and depth.
fn agrees(set_up: impl Fn(&mut Harness)) {
    agrees_shading(Harness::new, |_| {}, set_up);
}

/// [`agrees`] with a custom fragment shader, and `tune` applied to the device.
fn agrees_shading(
    new: impl Fn() -> Harness,
    tune: impl Fn(&mut super::Gpu),
    set_up: impl Fn(&mut Harness),
) {
    let Some(mut gpu) = device() else { return };
    tune(&mut gpu);
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = new();
        set_up(&mut h);
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        (h.target(), h.depth())
    };
    let want = build(None);
    let before = gpu.fallbacks;
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, before,
        "the draw did not run on the device: {:?}",
        gpu.last_fallback
    );
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(got.0, want.0, "the colour surface differs");
    assert_eq!(got.1, want.1, "the depth surface differs");
}

/// A 10-10-10-2 colour attribute renders the same on both renderers.
#[test]
fn a_10_10_10_2_colour_reaches_the_same_pixels_on_both_renderers() {
    const UNORM: u32 = 2;
    const SNORM: u32 = 1;
    for (ty, word) in [
        // Magenta, opaque: red and blue at their largest, alpha 3 of 3.
        (UNORM, 0x3ff | 0x3ff << 20 | 0b11 << 30),
        // Through snorm: 511 is 1, -512 clamps to -1 then 0, and a two-bit 1 is 1.
        (SNORM, 0x1ff | 0x200 << 10 | 0x1ff << 20 | 0b01 << 30),
    ] {
        let set_up = move |h: &mut Harness| {
            h.triangle([0.0; 4]);
            // Attribute 1, the colour: offset 16, size 0x30.
            h.engine
                .regs
                .set(0x458 + 1, (16 << 7) | (0x30 << 21) | (ty << 27));
            let vertices = h.base + 0x400;
            for vertex in 0..3u64 {
                h.vmm
                    .write_u32(&h.mem, vertices + vertex * 32 + 16, word)
                    .unwrap();
            }
        };
        agrees(set_up);
        let mut h = Harness::new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_eq!(h.texel(1, 1), 0xffff_00ff, "type {ty}: opaque magenta");
    }
}

/// A colour through local memory (`st.64 l[RZ + 0x10]` and back) renders the same on both.
#[test]
fn a_colour_through_local_memory_reaches_the_same_pixels_on_both_renderers() {
    use switch_core::gpu::shader::isa::{self, MemSize};
    use switch_core::gpu::shader::Op;
    use switch_core::gpu::testing::block;
    const ALWAYS: u64 = 7 << 16;
    const RZ: u64 = 0xff << 8;
    let at = 0x10u64 << 20;
    let stl = ((0xef50u64 | 5) << 48) | at | ALWAYS | RZ;
    let ldl = ((0xef40u64 | 5) << 48) | at | ALWAYS | RZ;
    assert_eq!(
        isa::decode(stl).op,
        Op::Stl {
            addr: 0xff,
            offset: 0x10,
            src: 0,
            size: MemSize::B64
        }
    );
    assert_eq!(
        isa::decode(ldl).op,
        Op::Ldl {
            dst: 0,
            addr: 0xff,
            offset: 0x10,
            size: MemSize::B64
        }
    );
    let split = |w: u64| (w as u32, (w >> 32) as u32);
    let shader = move || {
        let mut bytes = block(
            (0xe1a0070f, 0x00240401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c]
            (0x00470003, 0x50800000), // mufu rcp $r3 $r0
            (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3
        );
        bytes.extend(block(
            (0xb0400341, 0x055c8400),
            (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3
            (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3
            (0xc037ff03, 0xe043ff88), // ipa $r3 a[0x8c] $r3
        ));
        bytes.extend(block(
            (0xffe1ffef, 0x001f8000),
            split(stl),
            split(ldl),
            (0x0007000f, 0xe3000000), // exit
        ));
        bytes
    };
    let set_up = |h: &mut Harness| h.triangle([1.0, 0.0, 1.0, 1.0]);
    agrees_shading(
        move || Harness::with_fragment_shader(shader()),
        |_| {},
        set_up,
    );
    let mut h = Harness::with_fragment_shader(shader());
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.texel(1, 1), 0xffff_00ff, "red survived the round trip");
}

/// `lea.hi`'s 64-bit negate and shift, done on two words in WGSL, gives what the interpreter
/// gives: each channel is 1.0 only when its `lea` produced the expected word.
#[test]
fn lea_computes_the_same_words_on_both_renderers() {
    use switch_core::gpu::shader::isa::{self, ICmp};
    use switch_core::gpu::shader::Op;
    use switch_core::gpu::testing::block;
    const PT: u64 = 7 << 16;
    let mov32i = |dst: u64, imm: u32| 0x0100_0000_0000_0000 | (u64::from(imm) << 20) | PT | dst;
    let lea_hi = |dst: u64, a: u64, b: u64, hi: u64, shift: u64, neg: u64| {
        0x5bdfu64 << 48 | hi << 39 | neg << 37 | shift << 28 | b << 20 | PT | a << 8 | dst
    };
    let lea_neg = |dst: u64, a: u64, b: u64, shift: u64| {
        0x5bd7u64 << 48 | 1 << 45 | shift << 39 | b << 20 | PT | a << 8 | dst
    };
    // iset.eq.u32.and.bf dst, a, b, pt
    let iset = |dst: u64, a: u64, b: u64| {
        0x5b54u64 << 48 | 1 << 44 | 7 << 39 | b << 20 | PT | a << 8 | dst
    };
    let leas = [
        // A borrow out of the low word: (0:2) negated, shifted right by one.
        lea_hi(6, 8, 3, 9, 31, 1),
        // No borrow: (1:0) negated.
        lea_hi(7, 10, 3, 11, 31, 1),
        // A shift of zero takes the high word whole.
        lea_hi(12, 4, 3, 5, 0, 0),
        lea_neg(13, 14, 3, 4),
    ];
    for word in leas {
        assert!(matches!(isa::decode(word).op, Op::Lea { .. }), "{word:#x}");
    }
    assert!(matches!(
        isa::decode(iset(0, 6, 16)).op,
        Op::Iset {
            cmp: ICmp::Eq,
            bf: true,
            ..
        }
    ));
    let words = [
        mov32i(3, 0x10),
        mov32i(8, 2),
        mov32i(9, 0),
        mov32i(10, 0),
        mov32i(11, 1),
        mov32i(4, 0x1234),
        mov32i(5, 7),
        mov32i(14, 3),
        leas[0],
        leas[1],
        leas[2],
        leas[3],
        mov32i(16, 0xf),
        mov32i(17, 0x8000_0010),
        mov32i(18, 0x17),
        mov32i(19, 0xffff_ffe0),
        iset(0, 6, 16),
        iset(1, 7, 17),
        iset(2, 12, 18),
        iset(3, 13, 19),
        0xe300_0000_0007_000f, // exit
    ];
    let split = |w: u64| (w as u32, (w >> 32) as u32);
    let nop = (0x00070f00, 0x50b00000);
    let shader = move || {
        words
            .chunks(3)
            .flat_map(|slots| {
                let slot = |i: usize| slots.get(i).map_or(nop, |&w| split(w));
                block((0xffe1ffef, 0x001f8000), slot(0), slot(1), slot(2))
            })
            .collect::<Vec<u8>>()
    };
    let set_up = |h: &mut Harness| h.triangle([0.0; 4]);
    agrees_shading(
        move || Harness::with_fragment_shader(shader()),
        |_| {},
        set_up,
    );
    let mut h = Harness::with_fragment_shader(shader());
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.texel(1, 1), 0xffff_ffff, "every lea matched");
}

/// A `ldg` through a constant-bank descriptor reads the address the add made, even
/// when the shader rewrites the register the offset came from before loading.
/// Echoes of Wisdom's controller applet builds its glyph quads this way, and the
/// device read the wrong record words, so none of its text showed.
#[test]
fn a_global_load_reads_where_the_address_points_after_its_offset_register_changes() {
    use switch_core::gpu::shader::isa::{self, MemSize};
    use switch_core::gpu::shader::Op;
    use switch_core::gpu::testing::block;
    const PT: u64 = 7 << 16;
    const RZ: u64 = 0xff;
    const ONE: u32 = 0x3f80_0000;
    const HALF: u32 = 0x3f00_0000;
    let mov32i = |dst: u64, imm: u32| 0x0100_0000_0000_0000 | (u64::from(imm) << 20) | PT | dst;
    // iadd dst, a, c0[word * 4], carrying out of the low half or into the high one.
    let iadd_c0 = |dst: u64, a: u64, word: u64, cin: u64, cout: u64| {
        0x4c10u64 << 48 | cout << 47 | cin << 43 | word << 20 | PT | a << 8 | dst
    };
    let ldg = |dst: u64, addr: u64| 0xeed0u64 << 48 | 4 << 48 | PT | addr << 8 | dst;
    let words = [
        mov32i(7, 16),
        iadd_c0(10, 7, 0x110 / 4, 0, 1),
        iadd_c0(11, RZ, 0x114 / 4, 1, 0),
        // The offset register changes; the address pair does not.
        mov32i(7, 0),
        ldg(0, 10),
        mov32i(12, 20),
        // The offset register is the address register itself.
        iadd_c0(12, 12, 0x110 / 4, 0, 1),
        iadd_c0(13, RZ, 0x114 / 4, 1, 0),
        ldg(1, 12),
        mov32i(2, 0),
        mov32i(3, ONE),
        0xe300_0000_0007_000f, // exit
    ];
    assert_eq!(
        isa::decode(ldg(0, 10)).op,
        Op::Ldg {
            dst: 0,
            addr: 10,
            offset: 0,
            size: MemSize::B32
        }
    );
    assert!(matches!(
        isa::decode(iadd_c0(10, 7, 0x110 / 4, 0, 1)).op,
        Op::Iadd {
            dst: 10,
            a: 7,
            cout: true,
            cin: false,
            ..
        }
    ));
    let split = |w: u64| (w as u32, (w >> 32) as u32);
    let nop = (0x00070f00, 0x50b00000);
    let shader = move || {
        words
            .chunks(3)
            .flat_map(|slots| {
                let slot = |i: usize| slots.get(i).map_or(nop, |&w| split(w));
                block((0xffe1ffef, 0x001f8000), slot(0), slot(1), slot(2))
            })
            .collect::<Vec<u8>>()
    };
    let set_up = |h: &mut Harness| {
        h.triangle([0.0; 4]);
        let constants = h.base + 0x1500;
        let buffer = h.base + 0x1a00;
        // Every word but the two the offsets name reads as one half.
        for word in 0..16u64 {
            let value = if word == 4 || word == 5 { ONE } else { HALF };
            h.vmm.write_u32(&h.mem, buffer + word * 4, value).unwrap();
        }
        h.vmm
            .write_u32(&h.mem, constants + 0x110, buffer as u32)
            .unwrap();
        h.vmm
            .write_u32(&h.mem, constants + 0x114, (buffer >> 32) as u32)
            .unwrap();
        // The fragment stage's bank 0.
        let mut ctx = switch_core::gpu::exec::ExecCtx {
            mem: &h.mem,
            vmm: &h.vmm,
            as_id: h.as_id,
            host1x: &mut h.host1x,
            stats: &mut h.stats,
            trace: false,
        };
        for (method, arg) in [
            (0x8E0, 0x200),
            (0x8E1, (constants >> 32) as u32),
            (0x8E2, constants as u32),
            (0x900 + 4 * 8 + 4, 1),
        ] {
            h.engine.write(method, arg, true, &mut ctx).unwrap();
        }
    };
    agrees_shading(
        move || Harness::with_fragment_shader(shader()),
        |_| {},
        set_up,
    );
    let mut h = Harness::with_fragment_shader(shader());
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.texel(1, 1), 0xff00_ffff, "both loads read one");
}

/// A shader reading its quad neighbour's register reads the same on both renderers,
/// with native quad operations and with the browser's `QUAD_SWAP`.
#[test]
fn a_shuffling_fragment_shader_reads_the_same_neighbour_the_rasterizer_reads() {
    for web_limits in [false, true] {
        agrees_shading(
            || Harness::with_fragment_shader(testing::derivative_fragment_shader()),
            move |gpu| gpu.set_web_limits(web_limits),
            |h| {
                h.depth_target(0x0207);
                // Red ramps across the target, so neighbours differ by a sixteenth.
                h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
                h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
                h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            },
        );
    }
}

/// A `tld4` gathers the same four texels in the same order on both renderers.
#[test]
fn a_gather_reads_the_texels_the_rasterizer_reads() {
    for component in [0, 1] {
        let set_up = |h: &mut Harness| {
            h.bindless_texture();
            // `TexCbIndex`: the bank the bindless fixture writes its handle into.
            h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        };
        let new = move || Harness::with_fragment_shader(testing::gather_fragment_shader(component));

        // The reference must be the image's channel, or two empty renders would agree.
        let mut h = new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        let channel = |texel: u32| (texel >> (8 * component)) & 0xff;
        let size = testing::BINDLESS_TEXTURE_SIZE;
        let values: Vec<u32> = (0..size * size)
            .map(|i| channel(testing::bindless_texel(i % size, i / size)))
            .collect();
        let drawn: Vec<u32> = h.target().into_iter().filter(|&c| c != 0).collect();
        assert!(
            !drawn.is_empty(),
            "component {component}: nothing was drawn"
        );
        for pixel in &drawn {
            for byte in pixel.to_le_bytes() {
                assert!(
                    values.contains(&u32::from(byte)),
                    "component {component}: {pixel:#010x} holds {byte:#x}, no texel's value"
                );
            }
        }

        agrees_shading(new, |_| {}, set_up);
    }
}

/// A `tex.aoffi` with an immediate offset samples the same texel on both renderers.
#[test]
fn a_constant_texel_offset_samples_what_the_rasterizer_samples() {
    let set_up = |h: &mut Harness| {
        h.bindless_texture();
        h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
    };
    let new = || Harness::with_fragment_shader(testing::offset_fragment_shader());

    let mut h = new();
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    let size = testing::BINDLESS_TEXTURE_SIZE;
    let texels: Vec<u32> = (0..size * size)
        .map(|i| testing::bindless_texel(i % size, i / size))
        .collect();
    let drawn: std::collections::BTreeSet<u32> =
        h.target().into_iter().filter(|&c| c != 0).collect();
    assert!(drawn.len() >= 8, "only {drawn:x?} was drawn");
    assert!(drawn.iter().all(|c| texels.contains(c)), "{drawn:x?}");

    agrees_shading(new, |_| {}, set_up);
}

/// Vertex arrays past the device's buffer count are interleaved into one buffer and draw alike.
#[test]
fn vertex_arrays_past_the_device_limit_are_folded_and_draw_alike() {
    let Some(mut gpu) = device() else { return };
    // Off the eight-bit half-way points, so both renderers round alike.
    let colour = [0.2f32, 0.4, 0.8, 1.0];
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::new();
        // The colour comes from array 1, 16 bytes a vertex, apart from array 0's positions.
        let colours = h.base + 0x600;
        h.engine.regs.set(0x458 + 1, 1 | (0x01 << 21) | (7 << 27));
        h.engine.regs.set(0x704, 16 | (1 << 12));
        h.engine.regs.set(0x705, (colours >> 32) as u32);
        h.engine.regs.set(0x706, colours as u32);
        h.engine.regs.set(0x7C2, (colours >> 32) as u32);
        h.engine.regs.set(0x7C3, colours as u32 + 3 * 16 - 1);
        // Array 0 still carries a colour, which must not be the one drawn.
        h.triangle([1.0, 1.0, 1.0, 1.0]);
        let mut ctx = h.ctx();
        for vertex in 0..3u64 {
            for (i, c) in colour.iter().enumerate() {
                ctx.write_u32(colours + vertex * 16 + i as u64 * 4, c.to_bits())
                    .unwrap();
            }
        }
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        h.target()
    };
    let want = build(None);
    assert!(want.contains(&0xffcc_6633), "the reference: {want:x?}");
    // Two arrays against a limit of one forces the fold.
    gpu.vertex_buffer_limit = 1;
    let fallbacks = gpu.fallbacks;
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(got, want, "the colour surface differs");
}

/// A held float depth surface sampled as a float texture reads the device's depth, not stale
/// memory, whether or not a stencil byte shares its texels.
#[test]
fn a_held_float_depth_surface_samples_as_the_depth_it_holds() {
    let Some(mut gpu) = device() else { return };
    for (zt_format, tic_format, name) in [(0x0A, 0x2f, "ZF32"), (0x19, 0x30, "ZF32_X24S8")] {
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
            h.bindless_texture();
            h.depth_target(0x0207);
            // Neither tested nor written, so the draw samples it without attaching it.
            h.engine.regs.set(0x3FA, zt_format);
            h.engine.regs.set(testing::DEPTH_TEST_ENABLE, 0);
            h.engine.regs.set(testing::DEPTH_WRITE_ENABLE, 0);
            h.engine.regs.set(0x364, 0.25f32.to_bits());
            // The image descriptor, pointed at the depth surface.
            let (tic, depth) = (h.base + 0x1400 + 32, h.base + 0x1000);
            let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
            let mut ctx = h.ctx();
            for (word, value) in [
                (0, tic_format | (7 << 7) | identity),
                (1, depth as u32),
                (2, (depth >> 32) as u32 | (3 << 21)),
                (3, 0),
                (4, (testing::TARGET_WIDTH - 1) | (1 << 23)),
                (5, testing::TARGET_HEIGHT - 1),
            ] {
                ctx.write_u32(tic + word * 4, value).unwrap();
            }
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
            match gpu {
                Some(gpu) => {
                    h.clear_depth_with(gpu).expect("the clear");
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => {
                    h.clear_depth_with(&mut Software).expect("the clear");
                    h.draw_with(&mut Software).expect("the draw");
                }
            }
            h.target()
        };
        let want = build(None);
        assert!(
            want.contains(&0xff00_0040),
            "{name}: the reference did not draw the cleared depth: {want:x?}"
        );
        let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, fallbacks,
            "{name}: the draw fell back: {:?}",
            gpu.last_fallback
        );
        assert_eq!(
            gpu.drawn,
            drawn + 1,
            "{name}: the draw did not run on the device"
        );
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(
            gpu.device_error(),
            None,
            "{name}: the device rejected the pass"
        );
        assert_eq!(got, want, "{name}: the colour surface differs");
    }
}

/// A `ZF32_X24S8` image in guest memory samples as its depth, the stencil half of each texel left out.
#[test]
fn a_float_depth_image_with_stencil_samples_as_its_depth() {
    let Some(mut gpu) = device() else { return };
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
        h.bindless_texture();
        let size = testing::BINDLESS_TEXTURE_SIZE;
        let (tic, image) = (h.base + 0x1400 + 32, h.base + 0x1000);
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        let mut ctx = h.ctx();
        for y in 0..size {
            for x in 0..size {
                let at = image + u64::from((y * size + x) * 8);
                let depth = ((y * size + x) * 4) as f32 / 255.0;
                ctx.write_u32(at, depth.to_bits()).unwrap();
                ctx.write_u32(at + 4, 0xA5A5_A5A5).unwrap();
            }
        }
        // Pitch-linear, rows of eight 8-byte texels.
        for (word, value) in [
            (0, 0x30 | (7 << 7) | identity),
            (1, image as u32),
            (2, (image >> 32) as u32 | (2 << 21)),
            (3, size * 8 / 32),
            (4, (size - 1) | (1 << 23)),
            (5, size - 1),
        ] {
            ctx.write_u32(tic + word * 4, value).unwrap();
        }
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        h.target()
    };
    let want = build(None);
    let reds: std::collections::BTreeSet<u32> = want
        .iter()
        .filter(|&&c| c != 0)
        .map(|&c| c & 0xff)
        .collect();
    assert!(
        reds.len() >= 8,
        "the reference drew too few depths: {reds:x?}"
    );
    assert!(
        want.iter().all(|&c| c == 0 || c & 0x00ff_ff00 == 0),
        "{want:x?}"
    );
    let fallbacks = gpu.fallbacks;
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(got, want, "the colour surface differs");
}

/// An `S8D24` image in guest memory samples as the rasterizer reads it: unorm depth in green.
#[test]
fn a_packed_depth_image_samples_its_depth_in_green() {
    let Some(mut gpu) = device() else { return };
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
        h.bindless_texture();
        let size = testing::BINDLESS_TEXTURE_SIZE;
        let (tic, image) = (h.base + 0x1400 + 32, h.base + 0x1000);
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        let mut ctx = h.ctx();
        for y in 0..size {
            for x in 0..size {
                let at = image + u64::from((y * size + x) * 4);
                let depth = (y * size + x) * 4 * 0x01_0101;
                ctx.write_u32(at, depth << 8 | 0xA5).unwrap();
            }
        }
        // Pitch-linear, rows of eight 4-byte texels, every channel unsigned.
        for (word, value) in [
            (
                0,
                0x29 | (4 << 7) | (4 << 10) | (4 << 13) | (4 << 16) | identity,
            ),
            (1, image as u32),
            (2, (image >> 32) as u32 | (2 << 21)),
            (3, size * 4 / 32),
            (4, (size - 1) | (1 << 23)),
            (5, size - 1),
        ] {
            ctx.write_u32(tic + word * 4, value).unwrap();
        }
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        h.target()
    };
    let want = build(None);
    let greens: std::collections::BTreeSet<u32> = want
        .iter()
        .filter(|&&c| c != 0)
        .map(|&c| c >> 8 & 0xff)
        .collect();
    assert!(
        greens.len() >= 8,
        "the reference drew too few depths: {greens:x?}"
    );
    let fallbacks = gpu.fallbacks;
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(got, want, "the colour surface differs");
}

/// A held depth surface sampled as a shadow map, whole and as a padded surface's corner.
#[test]
fn a_held_depth_surface_is_the_shadow_map_the_rasterizer_compares_against() {
    let Some(mut gpu) = device() else { return };
    for width in [testing::TARGET_WIDTH, testing::TARGET_WIDTH / 2] {
        let build = |gpu: Option<&mut super::Gpu>, clear: f32| {
            let mut h = Harness::with_fragment_shader(testing::shadow_fragment_shader());
            h.bindless_texture();
            h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
            h.depth_target(0x0207);
            h.engine.regs.set(0x3FA, 0x0A); // ZF32
            h.engine.regs.set(testing::DEPTH_TEST_ENABLE, 0);
            h.engine.regs.set(testing::DEPTH_WRITE_ENABLE, 0);
            h.engine.regs.set(0x364, clear.to_bits());
            let tic = h.base + 0x1400 + 32;
            let tsc = h.base + 0x1480 + 32;
            let depth = h.base + 0x1000;
            let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
            let mut ctx = h.ctx();
            for (word, value) in [
                (0, 0x2f | (7 << 7) | identity),
                (1, depth as u32),
                (2, (depth >> 32) as u32 | (3 << 21)),
                (3, 0),
                (4, (width - 1) | (1 << 23)),
                (5, testing::TARGET_HEIGHT - 1),
            ] {
                ctx.write_u32(tic + word * 4, value).unwrap();
            }
            // The fixture's sampler, comparing with Less.
            ctx.write_u32(tsc, 2 | (2 << 3) | (2 << 6) | (1 << 9) | (1 << 10))
                .unwrap();
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
            match gpu {
                Some(gpu) => {
                    h.clear_depth_with(gpu).expect("the clear");
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => {
                    h.clear_depth_with(&mut Software).expect("the clear");
                    h.draw_with(&mut Software).expect("the draw");
                }
            }
            h.target()
        };
        // A stale read would compare against zero, a different picture.
        let want = build(None, 0.75);
        assert_ne!(want, build(None, 0.0), "{width} wide: 0.75 reads as 0");
        let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
        let got = build(Some(&mut gpu), 0.75);
        assert_eq!(
            gpu.fallbacks, fallbacks,
            "{width} wide: the draw fell back: {:?}",
            gpu.last_fallback
        );
        assert_eq!(gpu.drawn, drawn + 1, "{width} wide: not on the device");
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "{width} wide: rejected");
        assert_eq!(got, want, "{width} wide: the colour surface differs");
    }
}

/// A texture that is the corner of a held surface is copied from the device.
#[test]
fn a_held_surface_stands_in_for_a_smaller_texture_laid_out_the_same_way() {
    let Some(mut gpu) = device() else { return };
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
        h.bindless_texture();
        // The fixture's image descriptor, pointed at the colour target.
        let tic = h.base + 0x1400 + 32;
        let target = h.base;
        let mut ctx = h.ctx();
        ctx.write_u32(tic + 4, target as u32).unwrap();
        ctx.write_u32(tic + 8, (target >> 32) as u32 | (2 << 21))
            .unwrap();
        ctx.write_u32(tic + 12, testing::TARGET_WIDTH * 4 / 32)
            .unwrap();
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        // No half anywhere: see the clear test.
        for (i, value) in [0.0f32, 0.2, 0.6, 1.0].into_iter().enumerate() {
            h.engine.regs.set(0x360 + i as u32, value.to_bits());
        }
        match gpu {
            Some(gpu) => {
                h.clear_with(gpu, [true; 4]).expect("the clear");
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => {
                h.clear_with(&mut Software, [true; 4]).expect("the clear");
                h.draw_with(&mut Software).expect("the draw");
            }
        }
        h.target()
    };
    let want = build(None);
    let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    assert_eq!(gpu.drawn, drawn + 1, "the draw did not run on the device");
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(got, want, "the colour surface differs");
}

/// Culling matches on both renderers whether or not the viewport mirrors y.
#[test]
fn culling_keeps_the_faces_the_rasterizer_keeps() {
    const VIEWPORT_TRANSFORM: u32 = 0x280;
    const CULL: u32 = 0x646;
    const FRONT_FACE: u32 = 0x647;
    const CULL_FACE: u32 = 0x648;
    const CW: u32 = 0x900;
    const CCW: u32 = 0x901;
    const BACK: u32 = 0x405;
    let set_up = |mirrored: bool, front: u32| {
        move |h: &mut Harness| {
            h.triangle([0.0, 1.0, 0.0, 1.0]);
            let (w, height) = (
                testing::TARGET_WIDTH as f32 / 2.0,
                testing::TARGET_HEIGHT as f32 / 2.0,
            );
            let scale_y = if mirrored { -height } else { height };
            for (i, value) in [w, scale_y, 0.5, w, height, 0.5].into_iter().enumerate() {
                h.engine
                    .regs
                    .set(VIEWPORT_TRANSFORM + i as u32, value.to_bits());
            }
            h.engine.regs.set(CULL, 1);
            h.engine.regs.set(FRONT_FACE, front);
            h.engine.regs.set(CULL_FACE, BACK);
        }
    };
    for mirrored in [true, false] {
        // One winding keeps the triangle and the other culls it.
        let drawn = [CW, CCW].map(|front| {
            let mut h = Harness::new();
            set_up(mirrored, front)(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            h.target().iter().any(|&c| c != 0)
        });
        assert_ne!(drawn[0], drawn[1], "mirrored={mirrored}: {drawn:?}");
        for front in [CW, CCW] {
            agrees(set_up(mirrored, front));
        }
    }
}

/// A bindless `tex.b` resolves its handle from the constant word the shader loaded.
#[test]
fn a_bindless_texture_is_the_one_the_rasterizer_samples() {
    // Pixel centres fall a quarter or three quarters into a texel.
    let set_up = |h: &mut Harness| {
        h.bindless_texture();
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
    };
    let new = || Harness::with_fragment_shader(testing::bindless_fragment_shader());

    // The reference must be the image first.
    let mut h = new();
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    let size = testing::BINDLESS_TEXTURE_SIZE;
    let texels: Vec<u32> = (0..size * size)
        .map(|i| testing::bindless_texel(i % size, i / size))
        .collect();
    let drawn: std::collections::BTreeSet<u32> =
        h.target().into_iter().filter(|&c| c != 0).collect();
    assert!(drawn.len() >= 8, "only {drawn:x?} was drawn");
    assert!(
        drawn.iter().all(|c| texels.contains(c)),
        "{drawn:x?} is not all texels"
    );

    agrees_shading(new, |_| {}, set_up);
}

#[test]
fn a_depth_tested_draw_writes_the_same_depth_the_rasterizer_writes() {
    // The depth functions titles use. Colours are ones and zeros since `mufu rcp` and
    // WGSL division differ by a rounding step.
    for (func, passes) in [
        (0x0201, false),
        (0x0203, false),
        (0x0204, true),
        (0x0207, true),
    ] {
        let set_up = move |h: &mut Harness| {
            h.depth_target(func);
            h.triangle([1.0, 0.0, 1.0, 1.0]);
        };
        agrees(set_up);
        // The surface starts at 0 and the triangle sits at 0.5 (Z24 0x800000).
        let mut h = Harness::new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        let (colour, depth) = if passes {
            (0xffff_00ff, 0x8000_0000)
        } else {
            (0, 0)
        };
        assert_eq!(h.texel(1, 1), colour, "func {func:#x}: colour");
        assert_eq!(h.depth()[17], depth, "func {func:#x}: depth");
    }
}

/// A depth surface smaller than the colour target confines the draw on both renderers.
#[test]
fn a_depth_surface_smaller_than_the_colour_target_confines_the_draw() {
    // `Always` with writes on still reaches the depth surface.
    let set_up = |h: &mut Harness| {
        h.depth_target_sized(0x0207, 8, 4);
        h.triangle([1.0, 0.0, 1.0, 1.0]);
    };
    agrees(set_up);

    // (9, 1) is inside the triangle and outside the depth surface, (1, 1) inside both.
    let mut h = Harness::new();
    let (inside, outside) = (h.texel(1, 1), h.texel(9, 1));
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_ne!(h.texel(1, 1), inside, "drawn where both surfaces exist");
    assert_eq!(h.texel(9, 1), outside, "untouched past the depth surface");
}

/// A clear and a draw on a depth surface larger than the colour target share one cropped
/// hold on the device, and what lies outside the colour target's extent stays as it was.
#[test]
fn a_depth_clear_and_draw_beside_a_smaller_colour_target_agree() {
    let Some(mut gpu) = device() else { return };
    let (width, height) = (testing::TARGET_WIDTH, testing::TARGET_HEIGHT);
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::new();
        h.depth_target_sized(0x0201, width * 2, height * 2);
        h.engine.regs.set(0x364, 0.75f32.to_bits());
        // The clear covers the colour target's extent, not the whole depth surface.
        h.engine.regs.set(0x3FD, width << 16);
        h.engine.regs.set(0x3FE, height << 16);
        h.triangle([1.0, 0.0, 1.0, 1.0]);
        let layout = h.engine.depth_target().unwrap().expect("bound").layout;
        let addr = h.base + 0x1000;
        let mut ctx = h.ctx();
        for y in 0..height * 2 {
            for x in 0..width * 2 {
                let at = addr + u64::from(layout.offset(x * 4, y, width * 2 * 4));
                let seed = y
                    .wrapping_mul(0x101)
                    .wrapping_add(x.wrapping_mul(0x1357_9bdf));
                ctx.write_u32(at, seed | 0x11).unwrap();
            }
        }
        match gpu {
            Some(gpu) => {
                h.clear_depth_with(gpu).expect("the clear");
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => {
                h.clear_depth_with(&mut Software).expect("the clear");
                h.draw_with(&mut Software).expect("the draw");
            }
        }
        (h.target(), h.depth())
    };
    let want = build(None);
    let before = gpu.fallbacks;
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, before,
        "a clear or the draw fell back: {:?}",
        gpu.last_fallback
    );
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected a pass");
    assert_eq!(got.0, want.0, "the colour surface differs");
    assert_eq!(got.1, want.1, "the depth surface differs");
}

#[test]
fn a_depth_only_pass_still_writes_depth() {
    // No colour target at all; the fragment shader still runs.
    let set_up = |h: &mut Harness| {
        h.depth_target(0x0207);
        // Unbind colour target 0: an address of zero is no surface.
        h.engine.regs.set(0x200, 0);
        h.engine.regs.set(0x201, 0);
        h.triangle([1.0, 1.0, 1.0, 1.0]);
    };
    agrees(set_up);
    let mut h = Harness::new();
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.depth()[17], 0x8000_0000, "depth 0.5 at (1, 1)");
}

#[test]
fn a_triangle_fan_is_assembled_the_way_the_rasterizer_assembles_one() {
    // Fans go through `raster::assemble` on both renderers.
    for primitive in [6, 9] {
        // A quad as a four-vertex fan: a list or strip would leave (0, 7) bare.
        let set_up = move |h: &mut Harness| {
            h.engine.last_draw.primitive = primitive;
            h.engine.last_draw.count = 4;
            let limit = h.vertices() as u32 + 4 * 32 - 1;
            h.engine.regs.set(0x7C1, limit);
            h.depth_target(0x0207);
            let colour = [1.0, 0.0, 1.0, 1.0];
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], colour);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], colour);
            h.write_vertex(2, [1.0, -1.0, 0.0, 1.0], colour);
            h.write_vertex(3, [-1.0, -1.0, 0.0, 1.0], colour);
        };
        agrees(set_up);
        let mut h = Harness::new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_eq!(h.texel(0, 7), 0xffff_00ff, "primitive {primitive}");
    }
}

#[test]
fn an_instanced_array_reads_this_instance_and_not_the_first() {
    // An instanced array's single uploaded element needs a zero stride.
    for instance in [0, 1, 2] {
        agrees(move |h| {
            h.depth_target(0x0207);
            h.triangle([0.0, 0.0, 0.0, 1.0]);
            h.instanced_colour(
                instance,
                &[
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
            );
        });
    }
}

#[test]
fn a_bgra_attribute_is_swapped_the_way_the_rasterizer_swaps_one() {
    // Red and blue tell a BGRA swap apart.
    let set_up = |h: &mut Harness| {
        h.depth_target(0x0207);
        h.triangle([1.0, 0.0, 0.0, 1.0]);
        let raw = h.engine.regs.get(0x459);
        h.engine.regs.set(0x459, raw | 1 << 31);
    };
    agrees(set_up);
    let mut h = Harness::new();
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.texel(1, 1), 0xffff_0000, "red arrives as blue");
}

/// With late readbacks, the frame after a fallback and those following go to the rasterizer.
#[test]
fn a_frame_the_device_cannot_finish_is_a_frame_it_does_not_start() {
    let Some(mut gpu) = device() else { return };
    // What a browser teaches it on its first present.
    gpu.deferred_readbacks = true;
    let colour = [1.0f32, 0.0, 1.0, 1.0];

    let mut h = Harness::new();
    h.triangle(colour);
    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    assert!(!gpu.software_frame, "nothing has fallen back yet");
    // A line loop must fall back.
    h.engine.last_draw.primitive = 2;
    // The rasterizer refuses it too; the fallback is what is under test.
    let _ = h.draw_with(&mut gpu);
    assert!(
        gpu.fell_back_this_frame,
        "a line loop should not have been expressible"
    );

    // The next frame's clear is where that becomes a decision.
    h.engine.last_draw.primitive = 4;
    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    assert!(
        gpu.software_frame,
        "the frame after a fallback is the rasterizer's"
    );
    let drawn = gpu.drawn;
    h.draw_with(&mut gpu).expect("the draw");
    assert_eq!(
        gpu.drawn, drawn,
        "a draw ran on the device in a rasterizer's frame"
    );

    // Read before anything clears it again.
    let got = h.target();
    let mut want = Harness::new();
    want.triangle(colour);
    want.clear_with(&mut Software, [true; 4])
        .expect("the clear");
    want.draw_with(&mut Software).expect("the draw");
    assert_eq!(got, want.target());

    // One clean frame releases the latch the first time.
    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    assert!(
        !gpu.software_frame,
        "a clean frame did not release the latch"
    );
    assert_eq!(gpu.unlatched, 1);
}

/// The latch holds through unrunnable frames, releases after a clean one, and doubles each relatch.
#[test]
fn the_latch_lets_go_after_clean_frames_and_waits_longer_each_time() {
    let Some(mut gpu) = device() else { return };
    gpu.deferred_readbacks = true;
    let mut h = Harness::new();
    h.triangle([1.0, 0.0, 1.0, 1.0]);
    // A line loop has no pipeline, so the device and the check refuse it.
    let line_loop = |h: &mut Harness, gpu: &mut super::Gpu| {
        h.engine.last_draw.primitive = 2;
        let _ = h.draw_with(gpu);
        h.engine.last_draw.primitive = 4;
    };
    let frame = |h: &mut Harness, gpu: &mut super::Gpu| {
        h.clear_with(gpu, [true; 4]).expect("the clear");
    };

    frame(&mut h, &mut gpu);
    line_loop(&mut h, &mut gpu);
    frame(&mut h, &mut gpu);
    assert!(gpu.software_frame, "a fallback did not latch");

    line_loop(&mut h, &mut gpu);
    frame(&mut h, &mut gpu);
    assert!(
        gpu.software_frame,
        "released after a frame the device could not have drawn"
    );

    // Nothing drawn is nothing learned.
    frame(&mut h, &mut gpu);
    assert!(gpu.software_frame, "released after a frame with no draws");

    h.draw_with(&mut gpu).expect("the draw");
    frame(&mut h, &mut gpu);
    assert!(!gpu.software_frame, "a clean frame did not release it");

    // Falling back again closes it, and the next release waits for two.
    line_loop(&mut h, &mut gpu);
    frame(&mut h, &mut gpu);
    assert!(gpu.software_frame, "a second fallback did not latch");
    assert_eq!(gpu.clean_frames_needed, 2);
    h.draw_with(&mut gpu).expect("the draw");
    frame(&mut h, &mut gpu);
    assert!(gpu.software_frame, "released after one of two clean frames");
    h.draw_with(&mut gpu).expect("the draw");
    frame(&mut h, &mut gpu);
    assert!(!gpu.software_frame, "two clean frames did not release it");
    assert_eq!(gpu.unlatched, 2);
}

/// A points draw that only stores runs its vertex shader on the rasterizer, which reads
/// no render target, so it neither counts as a fallback nor hands the frames over.
#[test]
fn a_store_only_points_draw_lands_without_latching_the_frame() {
    let Some(mut gpu) = device() else { return };
    gpu.deferred_readbacks = true;
    let mut h = Harness::new();
    let dest = h.base + 0x1000;
    h.set_vertex_shader(&testing::vertex_id_store_shader(dest));
    h.engine.last_draw.primitive = 0;

    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    let fallbacks = gpu.fallbacks;
    // The points themselves are still refused.
    let _ = h.draw_with(&mut gpu);
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "counted as a fallback: {:?}",
        gpu.last_fallback
    );
    assert_eq!(
        h.vmm.read_u32(&h.mem, dest).unwrap(),
        2,
        "the last vertex's store"
    );
    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    assert!(
        !gpu.software_frame,
        "the store-only draw latched the frames"
    );
}

/// A tall image sampled over eight rows starting at `first` (past the end where it
/// repeats), by both renderers: they must agree and the device must draw it.
fn folded_image_agrees(repeat: bool, first: impl Fn(u32) -> u32, wanted: impl Fn(u32) -> [u32; 2]) {
    let Some(mut gpu) = device() else { return };
    let max = gpu.device.limits().max_texture_dimension_2d;
    let (width, height) = (8u32, max + 8);
    let first = first(max);
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
        h.bindless_texture();
        let bytes = width * height * 4;
        h.mem
            .map_zero(0x7100_0000, bytes as usize)
            .expect("room for the image");
        let image = h
            .vmm
            .map(0x7100_0000, u64::from(bytes), 2, 0, 0x1000, 0, 0)
            .expect("a GPU mapping");
        let tic = h.base + 0x1400 + 32;
        let tsc = h.base + 0x1480 + 32;
        let mut ctx = h.ctx();
        for y in 0..height {
            for x in 0..width {
                // The row in the low half, so a pixel says which row it sampled.
                let texel = (y & 0xffff) | (x * 30) << 16 | 0xff << 24;
                ctx.write_u32(image + u64::from((y * width + x) * 4), texel)
                    .unwrap();
            }
        }
        // The fixture's descriptor, pointed at the tall image.
        ctx.write_u32(tic + 4, image as u32).unwrap();
        ctx.write_u32(tic + 8, (image >> 32) as u32 | (2 << 21))
            .unwrap();
        ctx.write_u32(tic + 12, width * 4 / 32).unwrap();
        ctx.write_u32(tic + 16, (width - 1) | (1 << 23)).unwrap();
        ctx.write_u32(tic + 20, height - 1).unwrap();
        // Clamped to the edge on u and p; v repeats or clamps.
        let wrap_v = if repeat { 0 } else { 2 };
        ctx.write_u32(tsc, 2 | (wrap_v << 3) | (2 << 6)).unwrap();
        // Pixel row p samples image row first + p.
        let (top, bottom) = (
            first as f32 / height as f32,
            (first + 8) as f32 / height as f32,
        );
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, top, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, top, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, bottom, 0.0, 1.0]);
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        h.target()
    };
    let want = build(None);
    let (fallbacks, drawn) = (gpu.fallbacks, gpu.drawn);
    let got = build(Some(&mut gpu));
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    assert_eq!(gpu.drawn, drawn + 1, "the draw did not run on the device");
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    let rows: std::collections::BTreeSet<u32> = want
        .iter()
        .filter(|&&c| c != 0)
        .map(|c| c & 0xffff)
        .collect();
    for row in wanted(max) {
        assert!(rows.contains(&row), "row {row} unsampled: {rows:?}");
    }
    assert_eq!(got, want, "the folded image sampled differently");
}

/// An image taller than the device allows is folded into layers and samples what the
/// rasterizer samples, across the seam between two tiles too (Nintendo Switch Sports'
/// 1024x32768 glyph atlas).
#[test]
fn an_image_taller_than_the_device_allows_samples_what_the_rasterizer_samples() {
    // The seam is row `max - 1`, the first tile's last and the second's first.
    folded_image_agrees(false, |max| max - 4, |max| [max - 2, max]);
}

/// A repeating tall image wraps from its last row to its first, as the glyph atlas does.
#[test]
fn a_repeating_tall_image_wraps_from_its_last_row_to_its_first() {
    // The image is `max + 8` rows tall; eight rows from four before its end.
    folded_image_agrees(true, |max| max + 4, |max| [max + 7, 0]);
}

/// A clear of a target the device cannot hold, overlapping nothing it holds, runs on
/// the rasterizer without counting as a fallback or handing the frames over.
#[test]
fn a_clear_the_device_cannot_hold_does_not_latch_the_frames() {
    let Some(mut gpu) = device() else { return };
    gpu.deferred_readbacks = true;
    // No browser renders into `R16_UNORM`.
    gpu.set_web_limits(true);
    let r16_unorm = |h: &mut Harness| {
        h.engine.regs.set(0x204, 0xEE);
        for (i, value) in [0.6f32, 0.0, 0.0, 0.0].into_iter().enumerate() {
            h.engine.regs.set(0x360 + i as u32, value.to_bits());
        }
    };
    let mut h = Harness::new();
    r16_unorm(&mut h);
    let fallbacks = gpu.fallbacks;
    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    assert_eq!(
        gpu.fallbacks, fallbacks,
        "counted as a fallback: {:?}",
        gpu.last_fallback
    );

    let mut want = Harness::new();
    r16_unorm(&mut want);
    want.clear_with(&mut Software, [true; 4])
        .expect("the clear");
    assert_eq!(h.target(), want.target());

    h.clear_with(&mut gpu, [true; 4])
        .expect("the next frame's clear");
    assert!(!gpu.software_frame, "the clear latched the frames");
}

#[test]
fn a_clear_writes_what_the_rasterizer_would_have_written() {
    let Some(mut gpu) = device() else { return };
    for channels in [[true; 4], [true, false, true, false], [false; 4]] {
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::new();
            // A channel in each of the four so a masked clear has one to leave alone.
            // No 0.5: the two renderers round 127.5 differently.
            h.engine.regs.set(0x360, 0.0f32.to_bits());
            h.engine.regs.set(0x361, 0.2f32.to_bits());
            h.engine.regs.set(0x362, 0.6f32.to_bits());
            h.engine.regs.set(0x363, 1.0f32.to_bits());
            match gpu {
                Some(gpu) => {
                    h.clear_with(gpu, channels).expect("the clear");
                    h.flush_with(gpu);
                }
                None => h.clear_with(&mut Software, channels).expect("the clear"),
            }
            h.target()
        };
        let want = build(None);
        let before = gpu.fallbacks;
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, before,
            "the clear fell back: {:?}",
            gpu.last_fallback
        );
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the clear");
        assert_eq!(got, want, "a clear of channels {channels:?}");
    }
}

#[test]
fn an_attribute_the_draw_binds_nothing_to_reads_what_the_rasterizer_reads() {
    // Fixed attributes get a constant buffer rather than falling back.
    agrees(|h| {
        h.depth_target(0x0207);
        h.triangle([1.0, 1.0, 1.0, 1.0]);
        // VertexAttribState[1], the colour: fixed, so no buffer feeds it.
        let raw = h.engine.regs.get(0x459);
        h.engine.regs.set(0x459, raw | 1 << 6);
    });
}

/// Every multisample mode; `4x4` takes the expanded route and `2x2` the device's.
#[test]
fn a_multisampled_draw_reaches_the_same_texels_the_rasterizer_reaches() {
    for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (3, 4, 2), (6, 4, 4)] {
        compare(mode, x, y, |_| {});
    }
}

/// The device multisampling route: fully covered pixels match, edges may differ.
#[test]
fn the_device_route_agrees_wherever_an_edge_is_not() {
    let Some(mut gpu) = device() else { return };
    gpu.set_device_msaa(true);
    let colour = [1.0f32, 1.0, 1.0, 1.0];
    let mut ran = 0;
    for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (3, 4, 2), (6, 4, 4)] {
        let build = |gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::new();
            h.multisample(mode, x, y);
            h.triangle(colour);
            match gpu {
                Some(gpu) => {
                    h.draw_with(gpu).expect("the draw");
                    h.flush_with(gpu);
                }
                None => h.draw_with(&mut Software).expect("the draw"),
            }
            h.target()
        };
        let want = build(None);
        let before = gpu.multisampled;
        let got = build(Some(&mut gpu));
        if gpu.multisampled == before {
            // Not offered, so it went the expanded way, already covered.
            continue;
        }
        ran += 1;
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "the device rejected the pass");
        let width = switch_core::gpu::testing::TARGET_WIDTH;
        for py in 0..switch_core::gpu::testing::TARGET_HEIGHT / y {
            for px in 0..width / x {
                let tile: Vec<u32> = (0..y)
                    .flat_map(|dy| (0..x).map(move |dx| (dx, dy)))
                    .map(|(dx, dy)| want[((py * y + dy) * width + px * x + dx) as usize])
                    .collect();
                if tile.iter().any(|&t| t != tile[0]) {
                    continue;
                }
                for (dx, dy) in (0..y).flat_map(|dy| (0..x).map(move |dx| (dx, dy))) {
                    let at = ((py * y + dy) * width + px * x + dx) as usize;
                    assert_eq!(
                        got[at], tile[0],
                        "mode {mode}: pixel ({px}, {py}) is not on an edge and differs"
                    );
                }
            }
        }
    }
    assert!(
        ran > 0,
        "this device offered none of the sample counts under test"
    );
}

#[test]
fn a_sample_mask_keeps_the_same_samples_on_the_device() {
    for (mode, x, y) in [(2, 2, 2), (6, 4, 4)] {
        for mask in [0b0001, 0b1010, 0b0110] {
            compare(mode, x, y, move |h| {
                h.engine.regs.set(testing::MULTISAMPLE_SAMPLE_MASK, mask);
            });
        }
    }
}

#[test]
fn alpha_to_coverage_keeps_the_same_samples_on_the_device() {
    // Alpha-to-coverage on the expanded route at 4x4, against the reference.
    for alpha in [0.0f32, 0.25, 0.5, 1.0] {
        compare(6, 4, 4, move |h| {
            h.engine.regs.set(testing::MULTISAMPLE_CONTROL, 1);
            let colour = [1.0, 1.0, 1.0, alpha];
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], colour);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], colour);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], colour);
        });
    }
}

/// `AntiAliasEnable` off over a multisampled surface: whole-pixel coverage.
#[test]
fn coverage_per_pixel_covers_whole_pixels_on_the_device_too() {
    for (mode, x, y) in [(1, 2, 1), (2, 2, 2), (6, 4, 4)] {
        compare(mode, x, y, |h| {
            h.engine.regs.set(testing::MULTISAMPLE_ENABLE, 0);
        });
    }
}

/// Every sample programmed to the pixel centre covers alike, which is whole-pixel coverage.
#[test]
fn samples_programmed_to_the_pixel_centre_run_on_the_device() {
    for (mode, x, y) in [(1, 2, 1), (2, 2, 2)] {
        compare(mode, x, y, |h| {
            for word in 0..4 {
                h.engine
                    .regs
                    .set(testing::MULTISAMPLE_SAMPLE_LOCATIONS + word, 0x8888_8888);
            }
        });
    }
}

/// Samples away from their texel centres: one draw per sample, the geometry moved onto it.
#[test]
fn programmed_sample_locations_reach_the_same_texels_the_rasterizer_reaches() {
    // `x | (y << 4)` in sixteenths of a pixel, four samples to a register.
    let rotated: &[u32] = &[0xEAA2_6E26];
    // The 4x2 pattern A Short Hike programs.
    let jittered: &[u32] = &[0xB735_9D59, 0x1FFB_71D3];
    for (mode, x, y, words) in [(2, 2, 2, rotated), (3, 4, 2, jittered)] {
        compare(mode, x, y, |h| {
            for (i, &word) in words.iter().enumerate() {
                h.engine
                    .regs
                    .set(testing::MULTISAMPLE_SAMPLE_LOCATIONS + i as u32, word);
            }
        });
    }
}

#[test]
fn two_address_spaces_with_a_surface_at_one_address_keep_their_own() {
    let Some(mut gpu) = device() else { return };
    let mut below = Harness::new();
    below.as_id = 1;
    below.triangle([1.0, 0.0, 0.0, 1.0]);
    let mut above = Harness::new();
    above.as_id = 2;
    above.triangle([0.0, 0.0, 1.0, 1.0]);
    let want = |h: &mut Harness| {
        h.draw_with(&mut Software).expect("the draw");
        h.target()
    };
    let (red, blue) = {
        let mut r = Harness::new();
        r.triangle([1.0, 0.0, 0.0, 1.0]);
        let mut b = Harness::new();
        b.triangle([0.0, 0.0, 1.0, 1.0]);
        (want(&mut r), want(&mut b))
    };

    below.draw_with(&mut gpu).expect("the first draw");
    above.draw_with(&mut gpu).expect("the second draw");
    below.flush_with(&mut gpu);
    above.flush_with(&mut gpu);
    assert_eq!(
        below.target(),
        red,
        "the first address space lost its surface"
    );
    assert_eq!(
        above.target(),
        blue,
        "the second address space got the first's"
    );
}

/// Two presented frames drawn into one surface land in order, the second's pixels last.
#[test]
fn presented_frames_land_in_the_order_they_were_submitted() {
    let Some(mut gpu) = device() else { return };
    gpu.deferred_readbacks = true;
    gpu.defer_readbacks = true;
    let mut h = Harness::new();
    let submit = |h: &mut Harness, gpu: &mut super::Gpu, colour: [f32; 4]| {
        h.triangle(colour);
        h.clear_with(gpu, [true; 4]).expect("the clear");
        h.draw_with(gpu).expect("the draw");
        gpu.submit_frame(&mut h.ctx()).expect("the submit")
    };
    let first = submit(&mut h, &mut gpu, [1.0, 0.0, 0.0, 1.0]);
    let second = submit(&mut h, &mut gpu, [0.0, 0.0, 1.0, 1.0]);
    let generation = |s| match s {
        Submit::Queued(g) => g,
        _ => 0,
    };
    assert!(generation(second) >= generation(first));

    let mut landed = 0;
    for _ in 0..10_000 {
        let now = gpu.frames_landed(&mut h.ctx()).expect("the landing");
        assert!(now >= landed, "a landed generation went back");
        landed = now;
        if landed >= generation(second) && landed > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(landed >= generation(second), "the readbacks never landed");
    let mut want = Harness::new();
    want.triangle([0.0, 0.0, 1.0, 1.0]);
    want.clear_with(&mut Software, [true; 4])
        .expect("the clear");
    want.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.target(), want.target());
}
