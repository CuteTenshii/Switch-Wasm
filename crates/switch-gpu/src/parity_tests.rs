//! Device output against the software rasterizer.

use crate::tests::device;

use switch_core::gpu::renderer::{Blit, BlitRoute, Renderer, Software, Submit};
use switch_core::gpu::surface::{blend, taps, ColorFormat, Layout, Surface};
use switch_core::gpu::testing::{self, Harness};

/// A solid white triangle over a multisample mode, by both renderers; only coverage can differ.
fn compare(mode: u32, samples_x: u32, samples_y: u32, set_up: impl Fn(&mut Harness)) {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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

/// Put `words` in guest memory and name them in the fragment stage's bank 0 at `c0[0x110]`.
fn bind_global(h: &mut Harness, words: &[u32]) {
    let constants = h.base + 0x1500;
    let buffer = h.base + 0x1a00;
    for (word, &value) in words.iter().enumerate() {
        h.vmm
            .write_u32(&h.mem, buffer + word as u64 * 4, value)
            .unwrap();
    }
    h.vmm
        .write_u32(&h.mem, constants + 0x110, buffer as u32)
        .unwrap();
    h.vmm
        .write_u32(&h.mem, constants + 0x114, (buffer >> 32) as u32)
        .unwrap();
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
        // Every word but the two the offsets name reads as one half.
        let words: Vec<u32> = (0..16)
            .map(|word| if word == 4 || word == 5 { ONE } else { HALF })
            .collect();
        bind_global(h, &words);
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

/// `red` on a descriptor's buffer lands in guest memory as the rasterizer leaves it: one
/// add per covered fragment, and an or of a bit. Echoes of Wisdom's visibility pass marks
/// a buffer this way, and the device refused every draw of it.
#[test]
fn a_global_reduction_leaves_the_same_words_in_guest_memory_on_both_renderers() {
    use switch_core::gpu::testing::block;
    const PT: u64 = 7 << 16;
    const RZ: u64 = 0xff;
    let mov32i = |dst: u64, imm: u32| 0x0100_0000_0000_0000 | (u64::from(imm) << 20) | PT | dst;
    let iadd_c0 = |dst: u64, a: u64, word: u64, cin: u64, cout: u64| {
        0x4c10u64 << 48 | cout << 47 | cin << 43 | word << 20 | PT | a << 8 | dst
    };
    // red.<op>.u32 [addr + offset], src with the operation in bits 23..26.
    let red = |op: u64, offset: u64, addr: u64, src: u64| {
        0xebf8u64 << 48 | offset << 28 | op << 23 | PT | addr << 8 | src
    };
    // iscadd.cc dst, a, c0[word * 4], shift: the scaled index added to the descriptor.
    let iscadd_c0 = |dst: u64, a: u64, word: u64, shift: u64| {
        0x4c18u64 << 48 | 1 << 47 | shift << 39 | word << 20 | PT | a << 8 | dst
    };
    assert!(matches!(
        switch_core::gpu::shader::isa::decode(iscadd_c0(10, 7, 0x110 / 4, 2)).op,
        switch_core::gpu::shader::Op::Iscadd {
            cout: true,
            shift: 2,
            ..
        }
    ));
    let words = [
        mov32i(7, 1),
        iscadd_c0(10, 7, 0x110 / 4, 2),
        iadd_c0(11, RZ, 0x114 / 4, 1, 0),
        mov32i(2, 1),
        mov32i(3, 4),
        red(0, 0, 10, 2),
        red(6, 4, 10, 3),
        mov32i(0, 0),
        mov32i(1, 0),
        mov32i(2, 0),
        mov32i(3, 0x3f80_0000),
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
    let set_up = |h: &mut Harness| {
        h.triangle([0.0; 4]);
        bind_global(h, &[100, 1, 3]);
    };
    let words_after = |h: &Harness| -> Vec<u32> {
        (0..3u64)
            .map(|word| h.vmm.read_u32(&h.mem, h.base + 0x1a00 + word * 4).unwrap())
            .collect()
    };
    let mut want = Harness::with_fragment_shader(shader());
    set_up(&mut want);
    want.draw_with(&mut Software).expect("the draw");
    let covered = words_after(&want)[1] - 1;
    assert!(covered > 0, "the triangle covers fragments");
    assert_eq!(words_after(&want)[0], 100, "a word left alone");
    assert_eq!(words_after(&want)[2], 7, "the or");

    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut got = Harness::with_fragment_shader(shader());
    set_up(&mut got);
    let before = gpu.fallbacks;
    got.draw_with(&mut gpu).expect("the draw");
    assert_eq!(
        gpu.fallbacks, before,
        "the draw did not run on the device: {:?}",
        gpu.last_fallback
    );
    assert_eq!(
        words_after(&got),
        [100, 1, 3],
        "nothing lands before the flush"
    );
    // A second draw before the flush works on the words the first left on the device.
    got.draw_with(&mut gpu).expect("the second draw");
    assert_eq!(gpu.fallbacks, before, "the second draw fell back");
    got.flush_with(&mut gpu);
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    let twice = words_after(&got);
    assert_eq!(twice[1] - 1, 2 * covered, "both draws counted");
    assert_eq!(twice[2], 7);
}

const PT: u64 = 7 << 16;
const RZ: u64 = 0xff;

fn mov32i(dst: u64, imm: u32) -> u64 {
    0x0100_0000_0000_0000 | (u64::from(imm) << 20) | PT | dst
}

/// `body` between a prologue that points `r10` at the global buffer and an `exit`, drawn by
/// both renderers over `initial` words; returns the words the rasterizer leaves. The program
/// has to fit the harness's 0x100 bytes before its vertex buffer.
fn global_words_after_both(body: &[u64], initial: &[u32]) -> Vec<u32> {
    use switch_core::gpu::testing::block;
    let iadd_c0 = |dst: u64, word: u64, cin: u64, cout: u64| {
        0x4c10u64 << 48 | cout << 47 | cin << 43 | word << 20 | PT | RZ << 8 | dst
    };
    let mut words = vec![iadd_c0(10, 0x110 / 4, 0, 1), iadd_c0(11, 0x114 / 4, 1, 0)];
    words.extend_from_slice(body);
    words.extend([mov32i(3, 0x3f80_0000), 0xe300_0000_0007_000f]);
    assert!(words.len() <= 24, "{} instructions", words.len());
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
        bind_global(h, initial);
    };
    let words_after = |h: &Harness| -> Vec<u32> {
        (0..initial.len() as u64)
            .map(|word| h.vmm.read_u32(&h.mem, h.base + 0x1a00 + word * 4).unwrap())
            .collect()
    };
    let mut want = Harness::with_fragment_shader(shader());
    set_up(&mut want);
    want.draw_with(&mut Software).expect("the draw");
    let want = words_after(&want);

    let Some((_serial, mut gpu)) = device() else {
        return want;
    };
    let mut got = Harness::with_fragment_shader(shader());
    set_up(&mut got);
    let before = gpu.fallbacks;
    got.draw_with(&mut gpu).expect("the draw");
    assert_eq!(
        gpu.fallbacks, before,
        "the draw did not run on the device: {:?}",
        gpu.last_fallback
    );
    got.flush_with(&mut gpu);
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    assert_eq!(gpu.device_error(), None, "the device rejected the pass");
    assert_eq!(words_after(&got), want, "the device left other words");
    want
}

/// Every global atomic the device runs lands the word the rasterizer leaves, signed and float
/// reductions and the compare-exchange forms included. Each is applied once per covered fragment.
#[test]
fn global_atomics_leave_the_same_words_on_both_renderers() {
    use switch_core::gpu::shader::isa::{self, AtomOp, AtomSpace, AtomType};
    use switch_core::gpu::shader::Op;
    let red = |op: u64, ty: u64, word: u64, src: u64| {
        0xebf8u64 << 48 | (word * 4) << 28 | op << 23 | ty << 20 | PT | 10 << 8 | src
    };
    let atom = |op: u64, word: u64, src: u64| {
        0xedu64 << 56 | op << 52 | (word * 4) << 28 | src << 20 | PT | 10 << 8 | RZ
    };
    let cas =
        |word: u64, src: u64| 0xeef0u64 << 48 | (word * 4) << 28 | src << 20 | PT | 10 << 8 | RZ;
    let (add, min, max, xor, exch) = (0, 1, 2, 7, 8);
    let (u32_, s32, f32_) = (0, 1, 3);
    for (word, op, ty) in [
        (red(add, u32_, 0, 2), AtomOp::Add, AtomType::U32),
        (red(min, s32, 1, 2), AtomOp::Min, AtomType::S32),
        (red(add, f32_, 5, 2), AtomOp::Add, AtomType::F32),
        (atom(exch, 9, 2), AtomOp::Exch, AtomType::U32),
        (cas(10, 2), AtomOp::Cas, AtomType::U32),
    ] {
        let Op::Atom {
            op: got_op,
            ty: got_ty,
            space: AtomSpace::Global,
            ..
        } = isa::decode(word).op
        else {
            panic!("{word:#x} is not a global atomic");
        };
        assert_eq!((got_op, got_ty), (op, ty), "{word:#x}");
    }
    let one = 1.0f32.to_bits();
    let initial = [
        5,
        10,
        0xffff_fff0,
        0,
        3,
        one,
        one,
        (-2.0f32).to_bits(),
        0xff,
        1,
        7,
    ];
    let counted = global_words_after_both(
        &[
            mov32i(2, 3),
            red(add, u32_, 0, 2),
            mov32i(2, 0xffff_fffe),
            red(min, s32, 1, 2),
            mov32i(2, 4),
            red(max, s32, 2, 2),
            mov32i(2, 0.5f32.to_bits()),
            red(add, f32_, 5, 2),
        ],
        &initial,
    );
    let covered = (f32::from_bits(counted[5]) - 1.0) * 2.0;
    assert!(covered >= 1.0, "the triangle covers fragments");
    let covered = covered as u32;
    assert_eq!(counted[0], 5 + 3 * covered, "add");
    assert_eq!(counted[1], 0xffff_fffe, "signed min");
    assert_eq!(counted[2], 4, "signed max");

    let rest = global_words_after_both(
        &[
            mov32i(2, 0x0f),
            red(xor, u32_, 8, 2),
            mov32i(2, 0x77),
            atom(exch, 9, 2),
            mov32i(2, 7),
            mov32i(3, 9),
            cas(10, 2),
        ],
        &initial,
    );
    assert_eq!(
        rest[8],
        0xff ^ if covered % 2 == 1 { 0x0f } else { 0 },
        "xor"
    );
    assert_eq!(rest[9], 0x77, "exch");
    assert_eq!(rest[10], 9, "cas");
}

/// Global stores of every size, and narrow loads read back through them, land the words the
/// rasterizer leaves, including a 16-bit store straddling two words.
#[test]
fn global_stores_and_narrow_loads_leave_the_same_words_on_both_renderers() {
    use switch_core::gpu::shader::isa::{self, MemSize};
    use switch_core::gpu::shader::Op;
    let stg = |size: u64, byte: u64, src: u64| {
        0xeed8u64 << 48 | size << 48 | byte << 20 | PT | 10 << 8 | src
    };
    let ldg = |size: u64, byte: u64, dst: u64| {
        0xeed0u64 << 48 | size << 48 | byte << 20 | PT | 10 << 8 | dst
    };
    let (u8_, s8, u16_, s16, b32, b64) = (0, 1, 2, 3, 4, 5);
    assert_eq!(
        isa::decode(stg(u16_, 7, 2)).op,
        Op::Stg {
            addr: 10,
            offset: 7,
            src: 2,
            size: MemSize::U16
        }
    );
    assert_eq!(
        isa::decode(ldg(s16, 7, 4)).op,
        Op::Ldg {
            dst: 4,
            addr: 10,
            offset: 7,
            size: MemSize::S16
        }
    );
    let mut initial = vec![0u32; 12];
    initial[1] = 0x0102_0304;
    initial[2] = 0xffff_ffff;
    initial[3] = 0xffff_ffff;
    initial[6] = 0xaaaa_aaaa;
    initial[7] = 0xbbbb_bbbb;
    initial[8] = 0x8081_7f00;
    initial[9] = 0xab;
    let stored = global_words_after_both(
        &[
            mov32i(2, 0x1122_3344),
            stg(b32, 0, 2),
            mov32i(2, 0xaa),
            stg(u8_, 4 + 1, 2),
            mov32i(2, 0xbeef),
            stg(u16_, 8 + 3, 2),
            mov32i(2, 0xa0a0_a0a0),
            mov32i(3, 0xb0b0_b0b0),
            stg(b64, 16, 2),
            mov32i(2, 0x1122_3344),
            stg(b32, 6 * 4 + 2, 2),
        ],
        &initial,
    );
    assert_eq!(stored[0], 0x1122_3344, "b32");
    assert_eq!(stored[1], 0x0102_aa04, "u8 beside its neighbours");
    assert_eq!(stored[2], 0xefff_ffff, "u16 straddling, low byte");
    assert_eq!(stored[3], 0xffff_ffbe, "u16 straddling, high byte");
    assert_eq!(stored[4..6], [0xa0a0_a0a0, 0xb0b0_b0b0], "b64");
    assert_eq!(
        stored[6..8],
        [0x3344_aaaa, 0xbbbb_1122],
        "b32 off the word boundary"
    );

    let loaded = global_words_after_both(
        &[
            ldg(s8, 8 * 4 + 2, 4),
            stg(b32, 0, 4),
            ldg(u16_, 8 * 4 + 3, 4),
            stg(b32, 4, 4),
            ldg(s16, 8 * 4 + 3, 4),
            stg(b32, 8, 4),
            ldg(u8_, 8 * 4 + 1, 4),
            stg(b32, 12, 4),
        ],
        &initial,
    );
    assert_eq!(loaded[0], 0xffff_ff81, "s8");
    assert_eq!(loaded[1], 0xab80, "u16 straddling");
    assert_eq!(loaded[2], 0xffff_ab80, "s16 straddling");
    assert_eq!(loaded[3], 0x7f, "u8");
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

/// A `vote` counts the quad's four lanes on both renderers, ballot bits in lane order.
#[test]
fn a_quad_vote_gives_the_same_verdicts_and_ballot_on_both_renderers() {
    use switch_core::gpu::shader::isa::{self, ICmp, VoteMode};
    use switch_core::gpu::shader::Op;
    use switch_core::gpu::testing::block;
    const PT: u64 = 7 << 16;
    const RZ: u64 = 0xff;
    let mov32i = |dst: u64, imm: u32| 0x0100_0000_0000_0000 | (u64::from(imm) << 20) | PT | dst;
    // `@guard mov32i`.
    let mov32i_if = |guard: u64, dst: u64, imm: u32| {
        0x0100_0000_0000_0000 | (u64::from(imm) << 20) | guard << 16 | dst
    };
    // isetp.<cmp>.u32 p<p0>, pt, r<a>, r<b>
    let isetp = |p0: u64, cmp: u64, a: u64, b: u64| {
        0x5b60u64 << 48 | cmp << 49 | 7 << 39 | b << 20 | PT | a << 8 | p0 << 3 | 7
    };
    // vote.<mode> r<dst>, p<pred>, p0
    let vote = |mode: u64, pred: u64, dst: u64| (0x50d8u64 | mode) << 48 | pred << 45 | dst;
    const LT: u64 = 1;
    const EQ: u64 = 2;
    assert!(matches!(
        isa::decode(isetp(0, LT, 0, 5)).op,
        Op::Isetp {
            cmp: ICmp::Lt,
            p0: 0,
            ..
        }
    ));
    assert!(matches!(
        isa::decode(isetp(3, EQ, 4, 6)).op,
        Op::Isetp {
            cmp: ICmp::Eq,
            p0: 3,
            ..
        }
    ));
    assert!(matches!(
        isa::decode(vote(1, 1, 4)).op,
        Op::Vote {
            mode: VoteMode::Any,
            pred: 1,
            dst: 4,
            ..
        }
    ));
    const ONE: u32 = 0x3f80_0000;
    // `vColor.r`, a ramp across the target, as `derivative_fragment_shader` reads it.
    let words = [
        0xe003ff87_cff7ff00u64,
        0x5080_0000_0047_0003,
        0xe043ff88_0037ff00,
        mov32i(5, 0x3f00_0000),
        isetp(0, LT, 0, 5),
        vote(1, 1, 4),
        vote(0, 2, RZ),
        mov32i(6, 5),
        isetp(3, EQ, 4, 6),
        mov32i(0, 0),
        mov32i(1, 0),
        mov32i(2, 0),
        mov32i(3, ONE),
        mov32i_if(1, 0, ONE),
        mov32i_if(2, 1, ONE),
        mov32i_if(3, 2, ONE),
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
    let set_up = |h: &mut Harness| {
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
    };
    for web_limits in [false, true] {
        agrees_shading(
            || Harness::with_fragment_shader(shader()),
            move |gpu| gpu.set_web_limits(web_limits),
            set_up,
        );
    }
    let mut h = Harness::with_fragment_shader(shader());
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    let seen: std::collections::BTreeSet<u32> = h.target().into_iter().collect();
    assert!(
        seen.len() >= 3,
        "quads that agree and quads that split both occur: {seen:x?}"
    );
}

/// `bfe.brev` reverses the source before extracting, the same on both renderers.
/// Echoes of Wisdom's shaders use it, and the decoder refused it.
#[test]
fn a_reversing_bit_field_extract_gives_the_same_word_on_both_renderers() {
    use switch_core::gpu::shader::isa;
    use switch_core::gpu::shader::Op;
    use switch_core::gpu::testing::block;
    const PT: u64 = 7 << 16;
    const ONE: u32 = 0x3f80_0000;
    let mov32i = |dst: u64, imm: u32| 0x0100_0000_0000_0000 | (u64::from(imm) << 20) | PT | dst;
    let mov32i_if = |guard: u64, dst: u64, imm: u32| {
        0x0100_0000_0000_0000 | (u64::from(imm) << 20) | guard << 16 | dst
    };
    // bfe.brev r<dst>, r<a>, 32 << 8 (the whole word from bit 0)
    let bfe_brev = |dst: u64, a: u64| 0x3800u64 << 48 | 1 << 40 | 0x2000 << 20 | PT | a << 8 | dst;
    let isetp_eq = |p0: u64, a: u64, b: u64| {
        0x5b60u64 << 48 | 2 << 49 | 7 << 39 | b << 20 | PT | a << 8 | p0 << 3 | 7
    };
    assert!(matches!(
        isa::decode(bfe_brev(5, 4)).op,
        Op::Bfe {
            dst: 5,
            a: 4,
            brev: true,
            signed: false,
            ..
        }
    ));
    let words = [
        mov32i(4, 1),
        mov32i(6, 0x8000_0000),
        bfe_brev(5, 4),
        isetp_eq(0, 5, 6),
        mov32i(0, 0),
        mov32i(1, 0),
        mov32i(2, 0),
        mov32i(3, ONE),
        mov32i_if(0, 0, ONE),
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
    assert_eq!(h.texel(1, 1), 0xff00_00ff, "bit 0 became bit 31");
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

/// A `tld4.aoffi` whose offset sits in a register moves the gather by the same texels on both
/// renderers.
#[test]
fn a_register_gather_offset_reads_the_texels_the_rasterizer_reads() {
    let set_up = |h: &mut Harness| {
        h.bindless_texture();
        h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
        h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
    };
    for component in [0, 1] {
        let plain = {
            let mut h = Harness::with_fragment_shader(testing::gather_fragment_shader(component));
            set_up(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            h.target()
        };
        let new = move || {
            Harness::with_fragment_shader(testing::gather_offset_fragment_shader(component))
        };
        let mut h = new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        assert_ne!(
            h.target(),
            plain,
            "component {component}: the offset moved nothing"
        );

        agrees_shading(new, |_| {}, set_up);
    }
}

/// The bindless fixture's triangle with `corners` as the colour attribute each vertex carries.
fn colour_triangle(h: &mut Harness, corners: [[f32; 4]; 3]) {
    h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
    h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], corners[0]);
    h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], corners[1]);
    h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], corners[2]);
}

/// `tlds` forms with integer coordinates: a plain fetch, one moved by an offset register, an array
/// layer, a volume slice and a fetch outside the image read what the rasterizer reads.
#[test]
fn a_tlds_fetches_the_texel_the_rasterizer_fetches() {
    use testing::{bindless_texel, tlds_fragment_shader};
    type SetUp = fn(&mut Harness);
    type Form<'a> = (&'a str, u64, &'a [(u8, u32)], SetUp, u32);
    // The packed offset `0xe1` is +1 in x and -2 in y; a layer sits in its register's low half.
    let forms: [Form; 5] = [
        (
            "plain",
            2,
            &[(4, 3), (6, 5)],
            Harness::bindless_texture,
            bindless_texel(3, 5),
        ),
        (
            "offset",
            4,
            &[(4, 2), (5, 6), (6, 0xe1)],
            Harness::bindless_texture,
            bindless_texel(3, 4),
        ),
        (
            "array layer",
            8,
            &[(4, 0xabcd_0002), (6, 4), (7, 1)],
            Harness::bindless_array_texture,
            bindless_texel(6, 1),
        ),
        (
            "volume slice",
            7,
            &[(4, 1), (5, 2), (6, 2)],
            Harness::bindless_volume_texture,
            bindless_texel(3, 2),
        ),
        (
            "outside the image",
            4,
            &[(4, 0), (5, 3), (6, 0xf)],
            Harness::bindless_texture,
            0,
        ),
    ];
    for (name, encoding, set, image, texel) in forms {
        let set_up = move |h: &mut Harness| {
            image(h);
            colour_triangle(h, [[0.0; 4]; 3]);
        };
        let new = || Harness::with_fragment_shader(tlds_fragment_shader(encoding, set));
        let mut h = new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        let drawn: std::collections::BTreeSet<u32> = h.target().into_iter().collect();
        let want: std::collections::BTreeSet<u32> = [0, texel].into();
        assert_eq!(drawn, want, "{name}: the texel the registers name");

        agrees_shading(new, |_| {}, set_up);
    }
}

/// `tld4` forms beyond the plain gather: a shadow gather, a gather with an offset per texel and a
/// cube gather read what the rasterizer reads.
#[test]
fn the_other_gather_forms_read_the_texels_the_rasterizer_reads() {
    // Texture coordinates over the image, and a direction `(x, y, 1)` that stays in the +z
    // face, clear of its edges and of the 0.125 steps where a footprint changes texel.
    const UV: [[f32; 4]; 3] = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ];
    const DIRECTION: [[f32; 4]; 3] = [
        [-0.43, -0.43, 1.0, 1.0],
        [0.43, -0.43, 1.0, 1.0],
        [-0.43, 0.43, 1.0, 1.0],
    ];
    // Directions `(x, y, 1)` whose gather footprint runs over the +z face's left and right edges
    // onto -x and +x, short of a corner.
    const SEAM: [[f32; 4]; 3] = [
        [-0.95, -0.43, 1.0, 1.0],
        [0.95, -0.43, 1.0, 1.0],
        [-0.95, 0.43, 1.0, 1.0],
    ];
    type Shader = fn() -> Vec<u8>;
    type SetUp = fn(&mut Harness);
    let forms: [(&str, Shader, SetUp, Option<Shader>); 7] = [
        (
            "shadow gather",
            testing::gather_compare_fragment_shader,
            |h| {
                h.bindless_depth_texture();
                colour_triangle(h, UV);
            },
            None,
        ),
        (
            "per-texel offsets",
            || testing::gather_ptp_fragment_shader(1),
            |h| {
                h.bindless_texture();
                colour_triangle(h, UV);
            },
            Some(|| testing::gather_fragment_shader(1)),
        ),
        (
            "cube shadow gather",
            testing::gather_cube_compare_fragment_shader,
            |h| {
                h.bindless_depth_cube_texture();
                colour_triangle(h, DIRECTION);
            },
            None,
        ),
        (
            "cube gather",
            || testing::gather_cube_fragment_shader(2),
            |h| {
                h.bindless_cube_texture();
                colour_triangle(h, DIRECTION);
            },
            None,
        ),
        (
            "cube gather over a seam",
            || testing::gather_cube_fragment_shader(2),
            |h| {
                h.bindless_cube_texture();
                colour_triangle(h, SEAM);
            },
            Some(|| testing::gather_cube_fragment_shader(1)),
        ),
        (
            "cube shadow gather over a seam",
            testing::gather_cube_compare_fragment_shader,
            |h| {
                h.bindless_depth_cube_texture();
                colour_triangle(h, SEAM);
            },
            None,
        ),
        (
            "cube array gather of the second cube",
            || testing::gather_cube_array_fragment_shader(2, 1),
            |h| {
                h.bindless_cube_array_texture();
                colour_triangle(h, DIRECTION);
            },
            Some(|| testing::gather_cube_array_fragment_shader(2, 0)),
        ),
    ];
    for (name, shader, set_up, plain) in forms {
        let software = |shader: Shader| {
            let mut h = Harness::with_fragment_shader(shader());
            set_up(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            h.target()
        };
        let drawn = software(shader);
        let distinct: std::collections::BTreeSet<u32> =
            drawn.iter().copied().filter(|&c| c != 0).collect();
        assert!(distinct.len() >= 4, "{name}: only {distinct:x?} was drawn");
        if let Some(plain) = plain {
            assert_ne!(drawn, software(plain), "{name}: the variant drew the same");
        }

        agrees_shading(|| Harness::with_fragment_shader(shader()), |_| {}, set_up);
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

/// An integer image's words, gathered and offset-sampled, reach the same pixels on both renderers
/// and the pixels its float twin draws.
#[test]
fn an_integer_image_gathers_and_offset_samples_the_words_the_rasterizer_reads() {
    let set_up = |integer: bool| {
        move |h: &mut Harness| {
            if integer {
                h.bindless_uint_texture();
            } else {
                h.bindless_texture();
            }
            h.engine.regs.set(0x982, testing::BINDLESS_HANDLE_BANK);
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        }
    };
    type Shader = fn() -> Vec<u8>;
    let shaders: [(&str, Shader); 3] = [
        ("gather", || testing::gather_fragment_shader(1)),
        ("register gather offset", || {
            testing::gather_offset_fragment_shader(2)
        }),
        ("constant sample offset", testing::offset_fragment_shader),
    ];
    for (name, shader) in shaders {
        let software = |integer: bool| {
            let mut h = Harness::with_fragment_shader(shader());
            set_up(integer)(&mut h);
            h.draw_with(&mut Software).expect("the draw");
            h.target()
        };
        let (float, integer) = (software(false), software(true));
        assert!(integer.iter().any(|&c| c != 0), "{name}: nothing was drawn");
        assert_eq!(integer, float, "{name}: the words are not the texels");

        agrees_shading(
            || Harness::with_fragment_shader(shader()),
            |_| {},
            set_up(true),
        );
    }
}

/// Vertex arrays past the device's buffer count are interleaved into one buffer and draw alike.
#[test]
fn vertex_arrays_past_the_device_limit_are_folded_and_draw_alike() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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

/// A held `Z24S8` surface sampled as `S8D24` reads the device's depth as guest memory would hold it.
#[test]
fn a_held_packed_depth_surface_samples_its_depth_in_green() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
        h.bindless_texture();
        h.depth_target(0x0207);
        // Neither tested nor written, so the draw samples it without attaching it.
        h.engine.regs.set(testing::DEPTH_TEST_ENABLE, 0);
        h.engine.regs.set(testing::DEPTH_WRITE_ENABLE, 0);
        h.engine.regs.set(0x364, 0.25f32.to_bits());
        let (tic, depth) = (h.base + 0x1400 + 32, h.base + 0x1000);
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        let mut ctx = h.ctx();
        for (word, value) in [
            (
                0,
                0x29 | (4 << 7) | (4 << 10) | (4 << 13) | (4 << 16) | identity,
            ),
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
        want.contains(&0xff00_4000),
        "the reference did not draw the cleared depth in green: {want:x?}"
    );
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

/// A `ZF32_X24S8` image in guest memory samples as its depth, the stencil half of each texel left out.
#[test]
fn a_float_depth_image_with_stencil_samples_as_its_depth() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
fn a_zero_stride_array_feeds_every_vertex_its_one_element() {
    let set_up = |h: &mut Harness| {
        h.depth_target(0x0207);
        h.triangle([1.0, 0.0, 0.0, 1.0]);
        h.shared_colour([0.0, 1.0, 0.0, 1.0]);
    };
    agrees(set_up);
    let mut h = Harness::new();
    set_up(&mut h);
    h.draw_with(&mut Software).expect("the draw");
    assert_eq!(h.texel(1, 1), 0xff00_ff00, "green from the shared element");
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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

/// Forcing the rasterizer holds it through clean frames, and the same frame matches the reference.
#[test]
fn forced_software_draws_the_rasterizers_frame_however_clean() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let colour = [0.0f32, 1.0, 0.0, 1.0];
    let mut h = Harness::new();
    h.triangle(colour);
    Renderer::force_software(&mut gpu, true);
    for _ in 0..3 {
        h.clear_with(&mut gpu, [true; 4]).expect("the clear");
        let drawn = gpu.drawn;
        h.draw_with(&mut gpu).expect("the draw");
        assert_eq!(gpu.drawn, drawn, "a draw ran on the device while forced");
        assert!(gpu.software_frame, "a clean frame released a forced latch");
    }
    let got = h.target();
    let mut want = Harness::new();
    want.triangle(colour);
    want.clear_with(&mut Software, [true; 4])
        .expect("the clear");
    want.draw_with(&mut Software).expect("the draw");
    assert_eq!(got, want.target());

    Renderer::force_software(&mut gpu, false);
    h.clear_with(&mut gpu, [true; 4]).expect("the clear");
    h.draw_with(&mut gpu).expect("the draw");
    assert!(!gpu.software_frame);
    assert_eq!(gpu.drawn, 1, "the device has the frames back");
}

/// The latch holds through unrunnable frames, releases after a clean one, and doubles each relatch.
#[test]
fn the_latch_lets_go_after_clean_frames_and_waits_longer_each_time() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
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
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    gpu.deferred_readbacks = true;
    gpu.defer_readbacks = true;
    let mut h = Harness::new();
    let submit = |h: &mut Harness, gpu: &mut super::Gpu, colour: [f32; 4]| {
        h.triangle(colour);
        h.clear_with(gpu, [true; 4]).expect("the clear");
        h.draw_with(gpu).expect("the draw");
        gpu.submit_frame(&mut h.ctx(), None).expect("the submit")
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

const RGBA8: u32 = 0xD5;

/// Where the blits below write: past everything else in the harness's mapping.
const BLIT_TARGET: u64 = 0x1000;

/// A shaded target of `Harness`'s, drawn by `renderer`.
fn shaded(renderer: &mut dyn Renderer) -> Harness {
    let mut h = Harness::new();
    h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.25, 1.0]);
    h.write_vertex(1, [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]);
    h.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]);
    h.clear_with(renderer, [true; 4]).expect("the clear");
    h.draw_with(renderer).expect("the draw");
    h
}

fn colour_surface(addr: u64, width: u32, height: u32) -> Surface {
    Surface {
        addr,
        width,
        height,
        format: ColorFormat::from_raw(RGBA8).expect("a format"),
        layout: Layout::Pitch { pitch: width * 4 },
    }
}

fn downscale(h: &Harness, filtered: bool) -> Blit {
    Blit {
        src: colour_surface(h.base, testing::TARGET_WIDTH, testing::TARGET_HEIGHT),
        dst: colour_surface(
            h.base + BLIT_TARGET,
            testing::TARGET_WIDTH / 2,
            testing::TARGET_HEIGHT / 2,
        ),
        dst_rect: (0, 0, testing::TARGET_WIDTH / 2, testing::TARGET_HEIGHT / 2),
        src_origin: (1.0, 1.0),
        step: (2.0, 2.0),
        filtered,
    }
}

/// What `Engine2D` computes for a blit of `source`, `Harness::target`'s texels.
fn expected_blit(blit: &Blit, source: &[u32]) -> Vec<u32> {
    let (width, height) = (blit.src.width, blit.src.height);
    let texel = |x: u32, y: u32| {
        let raw = source[(y.min(height - 1) * width + x.min(width - 1)) as usize];
        std::array::from_fn::<f32, 4, _>(|i| ((raw >> (8 * i)) & 0xFF) as f32 / 255.0)
    };
    let mut out = Vec::new();
    for y in 0..blit.dst_rect.3 {
        for x in 0..blit.dst_rect.2 {
            let u = blit.src_origin.0 + blit.step.0 * f64::from(x);
            let v = blit.src_origin.1 + blit.step.1 * f64::from(y);
            let colour = if blit.filtered {
                let (x0, fx) = taps(u);
                let (y0, fy) = taps(v);
                blend(
                    texel(x0, y0),
                    texel(x0 + 1, y0),
                    texel(x0, y0 + 1),
                    texel(x0 + 1, y0 + 1),
                    fx,
                    fy,
                )
            } else {
                texel(u as u32, v as u32)
            };
            out.push(colour.iter().enumerate().fold(0u32, |word, (i, channel)| {
                word | (((channel * 255.0 + 0.5).floor() as u32) << (8 * i))
            }));
        }
    }
    out
}

fn blit_on_device_agrees_with_the_engine(filtered: bool) {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = shaded(&mut gpu);
    let blit = downscale(&h, filtered);
    let route = gpu.blit(&blit, &mut h.ctx()).expect("the blit");
    assert_eq!(route, BlitRoute::Done, "both surfaces can be on the device");
    gpu.flush(&mut h.ctx()).expect("the flush");

    let base = h.base;
    let (width, height) = (blit.dst_rect.2, blit.dst_rect.3);
    let ctx = h.ctx();
    let got: Vec<u32> = (0..u64::from(width * height))
        .map(|i| ctx.read_u32(base + BLIT_TARGET + i * 4).expect("a texel"))
        .collect();

    let mut reference = shaded(&mut Software);
    assert_eq!(got, expected_blit(&blit, &reference.target()));
}

#[test]
fn a_filtered_blit_on_the_device_agrees_with_the_engine() {
    blit_on_device_agrees_with_the_engine(true);
}

#[test]
fn a_point_sampled_blit_on_the_device_agrees_with_the_engine() {
    blit_on_device_agrees_with_the_engine(false);
}

#[test]
fn a_blit_away_from_held_surfaces_stays_in_guest_memory() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = shaded(&mut gpu);
    let blit = Blit {
        src: colour_surface(h.base + BLIT_TARGET, 4, 4),
        dst: colour_surface(h.base + BLIT_TARGET + 0x100, 4, 4),
        dst_rect: (0, 0, 4, 4),
        src_origin: (0.0, 0.0),
        step: (1.0, 1.0),
        filtered: false,
    };
    let route = gpu.blit(&blit, &mut h.ctx()).expect("the blit");
    assert_eq!(route, BlitRoute::GuestMemory);
}

#[test]
fn a_blit_out_of_a_surface_on_another_format_flushes() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = shaded(&mut gpu);
    let mut blit = downscale(&h, false);
    blit.src.format = ColorFormat::from_raw(0xCA).expect("a format");
    let route = gpu.blit(&blit, &mut h.ctx()).expect("the blit");
    assert_eq!(route, BlitRoute::Flush);
}

#[test]
fn a_copy_away_from_held_surfaces_needs_no_flush() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = shaded(&mut gpu);
    let base = h.base;
    let away = base + BLIT_TARGET;
    let upload = away..away + 48;
    let run = std::slice::from_ref(&upload);
    assert!(gpu.guest_memory_is_current(run, &mut h.ctx()));
    assert!(
        !gpu.guest_memory_is_current(&[away..away + 48, base + 4..base + 8], &mut h.ctx()),
        "the harness's target is held"
    );
}

#[test]
fn a_blit_out_of_guest_memory_uploads_its_source_and_agrees_with_the_engine() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let mut h = Harness::new();
    let base = h.base;
    let source = base + BLIT_TARGET;
    let texels: Vec<u32> = (0..16 * 8u32)
        .map(|i| 0xFF00_0000 | ((i * 0x0003_0507) & 0x00FF_FFFF))
        .collect();
    {
        let mut ctx = h.ctx();
        for (i, texel) in texels.iter().enumerate() {
            ctx.write_u32(source + i as u64 * 4, *texel)
                .expect("a texel");
        }
    }
    let blit = Blit {
        src: colour_surface(source, 16, 8),
        dst: colour_surface(source + 0x400, 8, 4),
        dst_rect: (0, 0, 8, 4),
        src_origin: (1.0, 1.0),
        step: (2.0, 2.0),
        filtered: true,
    };
    let route = gpu.blit_on_device(&blit, &h.ctx());
    assert_eq!(route, BlitRoute::Done);
    gpu.flush(&mut h.ctx()).expect("the flush");
    let ctx = h.ctx();
    let got: Vec<u32> = (0..8 * 4u64)
        .map(|i| ctx.read_u32(source + 0x400 + i * 4).expect("a texel"))
        .collect();
    assert_eq!(got, expected_blit(&blit, &texels));
}

/// Minecraft's world pass: each surface is larger one way, and the draw lands in their overlap.
#[test]
fn a_depth_surface_alternating_with_its_crop_stays_on_the_device_and_agrees() {
    const COLOUR_ADDRESS_LOW: u32 = 0x201;
    const COLOUR_PITCH: u32 = 0x202;
    const NARROW_TARGET: u64 = 0x3000;
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let build = |renderer: &mut dyn Renderer| {
        let mut h = Harness::new();
        h.depth_target(0x0203);
        h.engine.regs.set(0x364, 0.75f32.to_bits());
        let wide = h.base as u32;
        let narrow = h.base + NARROW_TARGET;
        for _ in 0..2 {
            h.engine.regs.set(COLOUR_ADDRESS_LOW, wide);
            h.engine.regs.set(COLOUR_PITCH, testing::TARGET_WIDTH * 4);
            h.triangle([1.0, 0.0, 1.0, 1.0]);
            h.clear_with(renderer, [true; 4]).expect("the clear");
            h.clear_depth_with(renderer).expect("the depth clear");
            h.draw_with(renderer).expect("the full draw");
            // A narrower target beside the same depth, which keeps what was drawn above.
            h.engine.regs.set(COLOUR_ADDRESS_LOW, narrow as u32);
            h.engine.regs.set(COLOUR_PITCH, testing::TARGET_WIDTH * 2);
            h.triangle([0.0, 1.0, 0.0, 1.0]);
            h.draw_with(renderer).expect("the cropped draw");
        }
        h.flush_with(renderer);
        let narrow_row: Vec<u32> = {
            let ctx = h.ctx();
            (0..testing::TARGET_WIDTH / 2)
                .map(|x| ctx.read_u32(narrow + u64::from(x) * 4).unwrap())
                .collect()
        };
        h.engine.regs.set(COLOUR_ADDRESS_LOW, wide);
        h.engine.regs.set(COLOUR_PITCH, testing::TARGET_WIDTH * 4);
        (h.target(), narrow_row, h.depth())
    };
    let want = build(&mut Software);
    let before = gpu.fallbacks;
    let got = build(&mut gpu);
    assert_eq!(
        gpu.fallbacks, before,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    assert!(
        gpu.crop_switches >= 3,
        "switched {} times",
        gpu.crop_switches
    );
    assert_eq!(got.0, want.0, "wide colour");
    assert_eq!(got.1, want.1, "narrow colour");
    assert_eq!(got.2, want.2, "depth");
}

#[test]
fn a_draw_beside_a_depth_surface_larger_only_one_way_agrees() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let (width, height) = (testing::TARGET_WIDTH * 2, testing::TARGET_HEIGHT / 2);
    let build = |renderer: &mut dyn Renderer| {
        let mut h = Harness::new();
        h.depth_target_sized(0x0201, width, height);
        h.engine.regs.set(0x364, 0.75f32.to_bits());
        h.triangle([1.0, 0.0, 1.0, 1.0]);
        h.clear_with(renderer, [true; 4]).expect("the clear");
        h.clear_depth_with(renderer).expect("the depth clear");
        h.draw_with(renderer).expect("the draw");
        h.flush_with(renderer);
        (h.target(), h.depth())
    };
    let want = build(&mut Software);
    let before = gpu.fallbacks;
    let got = build(&mut gpu);
    assert_eq!(
        gpu.fallbacks, before,
        "the draw fell back: {:?}",
        gpu.last_fallback
    );
    assert_eq!(got.0, want.0, "colour");
    assert_eq!(got.1, want.1, "depth");
}

/// An `RGBA32_UINT` target takes the fragment shader's register words as they are, on both renderers.
#[test]
fn an_integer_target_takes_the_shader_words_on_both_renderers() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    let colour = [1.0f32, 2.0, 3.0, 4.0];
    let build = |gpu: Option<&mut super::Gpu>| {
        let mut h = Harness::new();
        // 4x8 texels of sixteen bytes, which fit before the shaders.
        h.engine.regs.set(0x202, 4 * 16);
        h.engine.regs.set(0x204, 0xC2);
        h.engine.regs.set(0x300, 4 << 16);
        h.engine.regs.set(0x301, 8 << 16);
        h.triangle(colour);
        match gpu {
            Some(gpu) => {
                h.draw_with(gpu).expect("the draw");
                h.flush_with(gpu);
            }
            None => h.draw_with(&mut Software).expect("the draw"),
        }
        let base = h.base;
        let ctx = h.ctx();
        (0..4 * 8 * 4)
            .map(|word| ctx.read_u32(base + word * 4).unwrap())
            .collect::<Vec<u32>>()
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
    assert_eq!(got, want);
    // The top-left texel is the colour's own words, not integers made of them.
    assert_eq!(got[..4], colour.map(f32::to_bits));
}

mod condition_codes {
    use super::*;
    use testing::{alu_fragment_shader, encode};

    const RZ: u64 = 255;
    const ALWAYS: u64 = 7;
    const LT: u64 = 1;
    const EQ: u64 = 2;
    const LE: u64 = 3;
    const GT: u64 = 4;
    const NE: u64 = 5;
    const GE: u64 = 6;
    /// `csetp` tests that read one flag: EQU is Z, SFT is S, HS is C and OFT is O.
    const FLAG_TESTS: [u64; 4] = [10, 21, 22, 23];

    /// `iadd3 dst, a, b, c` with the flags written when `cc`, the carry read when `x`, and `c` negated when `neg_c`.
    fn iadd3(dst: u64, [a, b, c]: [u64; 3], cc: bool, x: bool, neg_c: bool) -> u64 {
        encode(
            0x5cc0,
            &[
                (0, 8, dst),
                (8, 8, a),
                (20, 8, b),
                (39, 8, c),
                (47, 1, u64::from(cc)),
                (48, 1, u64::from(x)),
                (49, 1, u64::from(neg_c)),
            ],
        )
    }

    /// `iset.bf dst, a, b` under `cmp`, 1.0 or 0.0 by the compare and the predicate `src`.
    fn iset(dst: u64, a: u64, b: u64, cmp: u64, signed: bool, x: bool, src: u64) -> u64 {
        encode(
            0x5b50,
            &[
                (0, 8, dst),
                (8, 8, a),
                (20, 8, b),
                (39, 3, src),
                (43, 1, u64::from(x)),
                (44, 1, 1),
                (48, 1, u64::from(signed)),
                (49, 3, cmp),
            ],
        )
    }

    fn csetp(p: u64, test: u64) -> u64 {
        encode(0x50a0, &[(0, 3, 7), (3, 3, p), (8, 5, test), (39, 3, 7)])
    }

    /// r0 to r3 take Z, S, C and O, as `csetp` reads them into p0 to p3.
    fn read_flags() -> Vec<u64> {
        let mut ops: Vec<u64> = (0..4).map(|p| csetp(p, FLAG_TESTS[p as usize])).collect();
        ops.extend((0..4).map(|p| iset(p, RZ, RZ, ALWAYS, false, false, p)));
        ops
    }

    /// Pixel (1, 1)'s channels from the rasterizer, after checking the device draws the same.
    fn channels(set: &[(u8, u32)], ops: &[u64]) -> [u8; 4] {
        let shader = alu_fragment_shader(set, ops);
        let new = || Harness::with_fragment_shader(shader.clone());
        let set_up = |h: &mut Harness| h.triangle([0.0; 4]);
        agrees_shading(new, |_| {}, set_up);
        let mut h = new();
        set_up(&mut h);
        h.draw_with(&mut Software).expect("the draw");
        h.texel(1, 1).to_le_bytes()
    }

    fn lit(held: bool) -> u8 {
        if held {
            255
        } else {
            0
        }
    }

    /// A 64-bit compare done as `iadd3.cc` on the low words and `iset.x` on the high ones.
    #[test]
    fn an_extended_compare_matches_the_64_bit_compare_on_both_renderers() {
        let compares = [LT, EQ, LE, GT, NE, GE]
            .into_iter()
            .flat_map(|cmp| [(cmp, false), (cmp, true)])
            .collect::<Vec<_>>();
        // The high words differ by less than half the range: Eden reads the sign of their 32-bit
        // difference. Low words never zero: `iadd3` negates `b` as `!b + 1`, which carries wrongly for zero.
        let words = [
            (5, 5, 7, 7),
            (1, 2, 7, 7),
            (2, 1, 7, 7),
            (5, 5, 1, 2),
            (5, 5, 0xffff_fffe, 1),
            (1, 2, 1, 0xffff_fffe),
            (1, 2, 0xffff_ffff, 0),
            (u32::MAX, 1, 3, 3),
            (1, u32::MAX, 3, 3),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (lo_a, lo_b, hi_a, hi_b) in words {
            let set = [(4, lo_a), (5, lo_b), (6, hi_a), (7, hi_b)];
            let a = u64::from(hi_a) << 32 | u64::from(lo_a);
            let b = u64::from(hi_b) << 32 | u64::from(lo_b);
            for group in compares.chunks(4) {
                let mut ops = vec![iadd3(10, [RZ, 4, 5], true, false, true)];
                ops.extend(
                    group
                        .iter()
                        .enumerate()
                        .map(|(i, &(cmp, signed))| iset(i as u64, 6, 7, cmp, signed, true, 7)),
                );
                let got = channels(&set, &ops);
                for (i, &(cmp, signed)) in group.iter().enumerate() {
                    let order = if signed {
                        (a as i64).cmp(&(b as i64))
                    } else {
                        a.cmp(&b)
                    };
                    let held = match cmp {
                        LT => order.is_lt(),
                        EQ => order.is_eq(),
                        LE => order.is_le(),
                        GT => order.is_gt(),
                        NE => order.is_ne(),
                        _ => order.is_ge(),
                    };
                    assert_eq!(
                        got[i],
                        lit(held),
                        "{a:#x} vs {b:#x}, cmp {cmp}, signed {signed}"
                    );
                    seen.insert((cmp, held));
                }
            }
        }
        assert!(seen.iter().any(|&(_, held)| held) && seen.iter().any(|&(_, held)| !held));
    }

    /// `iadd3.cc` then `csetp` on each flag; Z, S and C have an independent answer, O the interpreter's.
    #[test]
    fn iadd3_cc_flags_read_the_same_on_both_renderers() {
        let triples = [
            [1, 2, 3],
            [u32::MAX, 1, 0],
            [0, 0, 0],
            [0x7fff_ffff, 1, 0],
            [0x8000_0000, 0x8000_0000, 0],
            [u32::MAX, 0, 1],
            [3, u32::MAX, 5],
        ];
        for [a, b, c] in triples {
            let set = [(4, a), (5, b), (6, c)];
            let mut ops = vec![iadd3(10, [4, 5, 6], true, false, false)];
            ops.extend(read_flags());
            let got = channels(&set, &ops);
            let lhs = a.wrapping_add(b);
            let (sum, carry) = lhs.overflowing_add(c);
            let want = [lit(sum == 0), lit((sum as i32) < 0), lit(carry)];
            assert_eq!(got[..3], want, "{a:#x} + {b:#x} + {c:#x}");
        }
    }

    /// `iadd.cc` and `iscadd.cc` write Z, S and C of their sum, and O as the interpreter does.
    #[test]
    fn iadd_cc_and_iscadd_cc_flags_read_the_same_on_both_renderers() {
        let cases = [
            (1, 2),
            (u32::MAX, 1),
            (0x7fff_ffff, 1),
            (0x8000_0000, 0x8000_0000),
            (0, 0),
        ];
        for (a, b) in cases {
            let set = [(4, a), (5, b)];
            let iadd = encode(0x5c10, &[(0, 8, 10), (8, 8, 4), (20, 8, 5), (47, 1, 1)]);
            let iscadd = encode(
                0x5c18,
                &[(0, 8, 10), (8, 8, 4), (20, 8, 5), (39, 5, 3), (47, 1, 1)],
            );
            for (op, scaled) in [(iadd, a), (iscadd, a << 3)] {
                let mut ops = vec![op];
                ops.extend(read_flags());
                let got = channels(&set, &ops);
                let (sum, carry) = scaled.overflowing_add(b);
                let want = [lit(sum == 0), lit((sum as i32) < 0), lit(carry)];
                assert_eq!(got[..3], want, "{scaled:#x} + {b:#x}");
            }
        }
    }

    /// `iadd3.x` adds the carry an earlier `iadd3.cc` left.
    #[test]
    fn iadd3_x_adds_the_carry_the_flags_hold() {
        for (a, b, carry) in [
            (u32::MAX, 1, true),
            (1, 2, false),
            (0x8000_0000, 0x8000_0000, true),
        ] {
            let (hi_a, hi_b) = (40, 2);
            let want = hi_a + hi_b + u32::from(carry);
            let set = [(4, a), (5, b), (6, hi_a), (7, hi_b), (8, want)];
            let ops = [
                iadd3(10, [RZ, 4, 5], true, false, false),
                iadd3(11, [6, 7, RZ], false, true, false),
                iset(0, 11, 8, EQ, false, false, 7),
            ];
            let got = channels(&set, &ops);
            assert_eq!(got[0], 255, "{a:#x} + {b:#x} carries {carry}");
        }
    }

    /// `r2p.cc` writes the flags from the low nibble, as `csetp` reads them back.
    #[test]
    fn r2p_cc_loads_the_flags_the_same_on_both_renderers() {
        for bits in [0b0101u32, 0b1010, 0b1111, 0b0001, 0b1000, 0b0110] {
            let r2p = encode(0x38f0, &[(8, 8, 4), (20, 19, 0xf), (40, 1, 1), (41, 2, 0)]);
            let mut ops = vec![r2p];
            ops.extend(read_flags());
            let got = channels(&[(4, bits)], &ops);
            let want: Vec<u8> = (0..4).map(|i| lit(bits >> i & 1 != 0)).collect();
            assert_eq!(got.to_vec(), want, "{bits:#06b}");
        }
    }
}

/// A padded held surface sampled out of its corner, clamped to the edge, is sampled in place
/// and filters exactly as its copy does, nearest or linear, inside and past the edges.
#[test]
fn a_padded_held_surface_is_sampled_in_place_as_its_copy_would_be() {
    let Some((_serial, mut gpu)) = device() else {
        return;
    };
    for linear in [false, true] {
        let build = |mut gpu: Option<&mut super::Gpu>| {
            let mut h = Harness::with_fragment_shader(testing::bindless_fragment_shader());
            h.bindless_texture();
            let (first, second) = (h.base, h.base + 0x2000);
            let mut renderer = |h: &mut Harness, flush: bool| match gpu.as_deref_mut() {
                Some(gpu) => {
                    h.draw_with(gpu).expect("the draw");
                    if flush {
                        h.flush_with(gpu);
                    }
                }
                None => h.draw_with(&mut Software).expect("the draw"),
            };
            // The first surface, drawn from the fixture's image.
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
            h.write_vertex(1, [3.0, 1.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -3.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]);
            renderer(&mut h, false);
            // The second samples the first's 12x6 corner, with the same rows.
            h.engine.regs.set(0x200, (second >> 32) as u32);
            h.engine.regs.set(0x201, second as u32);
            let (tic, tsc) = (h.base + 0x1400 + 32, h.base + 0x1480 + 32);
            let mut ctx = h.ctx();
            ctx.write_u32(tic + 4, first as u32).unwrap();
            ctx.write_u32(tic + 8, (first >> 32) as u32 | (2 << 21))
                .unwrap();
            ctx.write_u32(tic + 12, testing::TARGET_WIDTH * 4 / 32)
                .unwrap();
            let shape = ctx.read_u32(tic + 16).unwrap() & !0xffff;
            ctx.write_u32(tic + 16, shape | (12 - 1)).unwrap();
            let rows = ctx.read_u32(tic + 20).unwrap() & !0xffff;
            ctx.write_u32(tic + 20, rows | (6 - 1)).unwrap();
            let filter = if linear { 2 | (2 << 4) } else { 0 };
            ctx.write_u32(tsc + 4, filter).unwrap();
            h.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], [-0.25, -0.25, 0.0, 1.0]);
            h.write_vertex(1, [3.0, 1.0, 0.0, 1.0], [2.25, -0.25, 0.0, 1.0]);
            h.write_vertex(2, [-1.0, -3.0, 0.0, 1.0], [-0.25, 2.25, 0.0, 1.0]);
            renderer(&mut h, true);
            h.target()
        };
        let want = build(None);
        let (fallbacks, drawn, cropped) = (gpu.fallbacks, gpu.drawn, gpu.cropped_binds);
        let got = build(Some(&mut gpu));
        assert_eq!(
            gpu.fallbacks, fallbacks,
            "linear {linear}: fell back: {:?}",
            gpu.last_fallback
        );
        assert_eq!(gpu.drawn, drawn + 2, "linear {linear}: not on the device");
        assert_eq!(
            gpu.cropped_binds,
            cropped + 1,
            "linear {linear}: not sampled in place"
        );
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert_eq!(gpu.device_error(), None, "linear {linear}: rejected");
        assert_eq!(got, want, "linear {linear}: the sampled corner differs");
    }
}
