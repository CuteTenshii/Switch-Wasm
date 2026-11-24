//! A minimal drawable engine shared by the software and wgpu renderer tests:
//! a 16x8 pitch-linear RGBA8 target, two captured shaders, and a three-vertex array.

use crate::gpu::engine::threed::{DrawCall, Engine3D};
use crate::gpu::exec::{ExecCtx, GpuStats};
use crate::gpu::renderer::{Flush, Renderer};
use crate::gpu::syncpt::Host1x;
use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
use crate::mem::Memory;

pub fn word(low: u32, high: u32) -> [u8; 8] {
    (((high as u64) << 32) | low as u64).to_le_bytes()
}

/// One 32-byte scheduling block: the sched word and three instructions.
pub fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(&word(sched.0, sched.1));
    out.extend_from_slice(&word(a.0, a.1));
    out.extend_from_slice(&word(b.0, b.1));
    out.extend_from_slice(&word(c.0, c.1));
    out
}

/// `gl_Position = aPosition; vColor = aColor;`
pub fn passthrough_vertex_shader() -> Vec<u8> {
    // Sched words are never all-zero: an all-zero first word means a Mesa header.
    let mut bytes = block(
        (0xfc20070f, 0x081f8441),
        (0x0807ff00, 0xefd9ff80), // ld b128 $r0 a[0x80] 0x0  (aPosition)
        (0x0707ff00, 0xeff1ff80), // st b128 a[0x70] $r0 0x0  (gl_Position)
        (0x0907ff00, 0xefd9ff80), // ld b128 $r0 a[0x90] 0x0  (aColor)
    );
    bytes.extend(block(
        (0xfc2207e1, 0x001f8c40),
        (0x0807ff00, 0xeff1ff80), // st b128 a[0x80] $r0 0x0  (vColor)
        (0x0007000f, 0xe3000000), // exit
        (0, 0),
    ));
    bytes
}

/// `oColor = vColor;`
pub fn solid_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
        (0xc037ff03, 0xe043ff88), // ipa $r3 a[0x8c] $r3 0x0 0x1
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra 0x50 (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `*(u32 *)dest = gl_VertexID;` and nothing else: no position, only a store.
pub fn vertex_id_store_shader(dest: u64) -> Vec<u8> {
    let mov32i = |dst: u64, imm: u32| {
        let raw = 0x0100_0000_0007_f000 | u64::from(imm) << 20 | dst;
        (raw as u32, (raw >> 32) as u32)
    };
    let mut bytes = block(
        (0xfc20070f, 0x081f8441),
        (0x2fc7ff02, 0xefd87f80), // ld b32 $r2 a[0x2fc]  (gl_VertexID)
        mov32i(0, dest as u32),   // mov32i $r0 dest.lo
        mov32i(1, (dest >> 32) as u32),
    );
    bytes.extend(block(
        (0xfc2207e1, 0x001f8c40),
        (0x00070002, 0xeedc2000), // stg e b32 [$r0] $r2
        (0x0007000f, 0xe3000000), // exit
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `oColor = (dFdx(vColor.r), neighbour, 0, 1)` via `shfl.bfly`.
pub fn derivative_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0xf0170001, 0xef100070), // shfl.bfly $p0 $r1 $r0 0x1 0x1c
        (0x00170000, 0x5c590000), // fadd $r0 -$r0 $r1
        (0x0007000f, 0xe3000000), // exit
    ));
    bytes
}

/// `oColor = texture(bindless, vColor.xy)`, handle from `c3[0x10]`.
pub fn bindless_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x0107ff02, 0xef940030), // ld b32 $r2 c3[0x10]
        (0xa0270000, 0xdeba0007), // tex b nodep $r0 $r0 $r2 0x0 t2d 0xf
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra 0x50 (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `oColor = texture(bindless, vec3(vColor.xy, 0))`: layer 0 of an array, handle from `c3[0x10]`.
pub fn bindless_array_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff01, 0xe043ff88), // ipa $r1 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff02, 0xe043ff88), // ipa $r2 a[0x84] $r3 0x0 0x1
        (0x0107ff04, 0xef940030), // ld b32 $r4 c3[0x10]
        mov32i(0, 0),
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        BINDLESS_ARRAY_TEX,       // tex b nodep $r0 $r0 $r4 0x0 a2d 0xf
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
    ));
    bytes
}

/// `oColor = vec4(gl_FragCoord.xy, 0, 1/w)`: the window position `a[0x70]` and `a[0x74]` hold.
pub fn frag_coord_fragment_shader() -> Vec<u8> {
    let sched = (0xe1a0070f, 0x00240401);
    let mut bytes = block(
        sched,
        (0x0ff7ff00, 0xe003ff87), // ipa pass $r0 a[0x70] 0x0 0x0 0x1
        (0x4ff7ff01, 0xe003ff87), // ipa pass $r1 a[0x74] 0x0 0x0 0x1
        (0xcff7ff03, 0xe003ff87), // ipa pass $r3 a[0x7c] 0x0 0x0 0x1
    );
    bytes.extend(block(
        sched,
        mov32i(2, 0),
        (0x0007000f, 0xe3000000), // exit
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `imageStore(bindless, uvec2(vColor.xy), uvec4(uvec2(vColor.xy), 0, 0))`, the image's
/// handle from `c3[0x10]`: each fragment stores its own coordinates, so the order fragments
/// run in cannot change what lands.
pub fn image_store_fragment_shader() -> Vec<u8> {
    use crate::gpu::shader::isa::{FMod, FRound, Operand, SurfaceDim};
    use crate::gpu::shader::Op;
    let split = |w: u64| (w as u32, (w >> 32) as u32);
    let f2i_u32 = |reg: u8| {
        split(checked(
            0x5cb0u64 << 48
                | 3 << 39
                | u64::from(reg) << 20
                | PT
                | 2 << 10
                | 2 << 8
                | u64::from(reg),
            Op::F2i {
                dst: reg,
                src: Operand::Reg(reg),
                sm: FMod {
                    neg: false,
                    abs: false,
                },
                dst_bytes: 4,
                dst_signed: false,
                round: FRound::Trunc,
                ftz: false,
                src_bits: 32,
                hi: false,
            },
        ))
    };
    let sched = (0xe1a0070f, 0x00240401);
    let mut bytes = block(
        sched,
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff04, 0xe043ff88), // ipa $r4 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        sched,
        (0x4037ff05, 0xe043ff88), // ipa $r5 a[0x84] $r3 0x0 0x1
        f2i_u32(4),
        f2i_u32(5),
    ));
    bytes.extend(block(
        sched,
        (0x0107ff08, 0xef940030), // ld b32 $r8 c3[0x10]
        split(sust_p(SurfaceDim::D2, 4, 4, 8)),
        (0x0007000f, 0xe3000000), // exit
    ));
    bytes
}

/// [`bindless_array_fragment_shader`]'s sample: the layer in `r0`, then `u` and `v`.
pub const BINDLESS_ARRAY_TEX: (u32, u32) = (0xb0470000, 0xdeba0007);

/// `oColor = textureGather(texture, vColor.xy, component)` from bound slot 4.
pub fn gather_fragment_shader(component: u32) -> Vec<u8> {
    let tld4_high = 0xc83a0047 | (component & 3) << 24;
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0xaff70000, tld4_high),  // tld4 r|g nodep $r0 $r0 0x0 0x4 t2d 0xf
        (0x0007000f, 0xe3000000), // exit
    ));
    bytes
}

/// `oColor = textureGather(texture, vColor.xy + texel * ivec2(1, -1), component)` from slot 4,
/// the offset in a register.
pub fn gather_offset_fragment_shader(component: u32) -> Vec<u8> {
    let tld4_high = 0xc83a0047 | (component & 3) << 24 | 1 << 22;
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0xf017000a, 0x01000003), // mov32i $r10 0x3f01: x +1, y -1
        (0xa0a70000, tld4_high),  // tld4.aoffi r|g nodep $r0 $r0 $r10 0x4 t2d 0xf
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `oColor = textureGather(texture, vColor.xy, component)` with each of the four texels moved
/// by its own offset (`tld4.ptp`), from slot 4.
pub fn gather_ptp_fragment_shader(component: u32) -> Vec<u8> {
    let tld4_high = 0xc83a0047 | (component & 3) << 24 | 2 << 22;
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0xf017000a, 0x010033e3), // mov32i $r10 0x033e3f01: (+1, -1) and (-2, +3)
        (0x2007000b, 0x010003f0), // mov32i $r11 0x003f0200: (0, +2) and (-1, 0)
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0xa0a70000, tld4_high), // tld4.ptp r|g nodep $r0 $r0 $r10 0x4 t2d 0xf
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
    ));
    bytes
}

/// `oColor = textureGather(shadowMap, vColor.xy, 0.5)` from slot 4: each texel compared.
pub fn gather_compare_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88),           // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x0007f002, 0x0103f000),           // mov32i $r2 0x3f000000: 0.5
        (0xa0270000, 0xc83a0047 | 1 << 18), // tld4.dc nodep $r0 $r0 $r2 0x4 t2d 0xf
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `oColor = textureGather(shadowCube, vColor.xyz, 0.5)` from slot 4: each texel compared.
pub fn gather_cube_compare_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
        (0x0007f004, 0x0103f000), // mov32i $r4 0x3f000000: 0.5
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0xe0470000, 0xc83a0047 | 1 << 18), // tld4.dc nodep $r0 $r0 $r4 0x4 cube 0xf
        (0x0007000f, 0xe3000000),           // exit
        (0xff87000f, 0xe2400fff),           // bra (padding, never reached)
    ));
    bytes
}

/// `oColor = textureGather(cube, vColor.xyz, component)` from slot 4.
pub fn gather_cube_fragment_shader(component: u32) -> Vec<u8> {
    let tld4_high = 0xc83a0047 | (component & 3) << 24;
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
        (0xeff70000, tld4_high),  // tld4 r|g nodep $r0 $r0 0x0 0x4 cube 0xf
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `body` behind a Shader Program Header whose output map is `omap_target`, a nibble of
/// components per render target, as Mesa and nvn lay fragment programs out.
pub fn with_output_map(omap_target: u32, body: Vec<u8>) -> Vec<u8> {
    let mut program = vec![0u8; 0x50];
    program[18 * 4..19 * 4].copy_from_slice(&omap_target.to_le_bytes());
    program.extend(body);
    program
}

/// A fragment program body setting `r0..r7` to two opaque colours, red then blue; behind
/// [`with_output_map`] each output's components take them in order.
pub fn two_colour_fragment_body() -> Vec<u8> {
    let one = 1.0f32.to_bits();
    alu_fragment_shader(
        &[
            (0, one),
            (1, 0),
            (2, 0),
            (3, one),
            (4, 0),
            (5, 0),
            (6, one),
            (7, one),
        ],
        &[],
    )
}

/// `mov32i $reg, imm`.
fn mov32i(reg: u8, imm: u32) -> (u32, u32) {
    (
        (imm & 0xfff) << 20 | 0x0007_f000 | u32::from(reg),
        0x0100_0000 | imm >> 12,
    )
}

/// One instruction: `opcode` in the top 16 bits, predicate `PT`, and `(position, width, value)` fields.
pub fn encode(opcode: u16, fields: &[(u32, u32, u64)]) -> u64 {
    let mut word = u64::from(opcode) << 48 | 0x7 << 16;
    for &(position, width, value) in fields {
        word |= (value & ((1u64 << width) - 1)) << position;
    }
    word
}

/// A fragment shader that loads each `(register, value)` of `set`, runs `ops` and exits;
/// its colour is r0 to r3.
pub fn alu_fragment_shader(set: &[(u8, u32)], ops: &[u64]) -> Vec<u8> {
    const NOP: (u32, u32) = (0x00070f00, 0x50b00000);
    const EXIT: (u32, u32) = (0x0007000f, 0xe3000000);
    let mut words: Vec<(u32, u32)> = set.iter().map(|&(reg, value)| mov32i(reg, value)).collect();
    words.extend(ops.iter().map(|&op| (op as u32, (op >> 32) as u32)));
    words.push(EXIT);
    while !words.len().is_multiple_of(3) {
        words.push(NOP);
    }
    let mut bytes = Vec::new();
    for group in words.as_chunks::<3>().0 {
        bytes.extend(block(
            (0xe1a0070f, 0x00240401),
            group[0],
            group[1],
            group[2],
        ));
    }
    bytes
}

/// `oColor = texelFetch(...)` from slot 4: a `tlds` of the given `encoding` of Eden's
/// `texture_load_swizzled.cpp`, reading `a` as r4 and `b` as r6, after each `(register, value)`
/// of `set` (at most three) is loaded.
pub fn tlds_fragment_shader(encoding: u64, set: &[(u8, u32)]) -> Vec<u8> {
    const NOP: (u32, u32) = (0x00070f00, 0x50b00000);
    let tlds = (0xda46_00af_f0d7_0404u64
        & !(0xff | 0xff << 8 | 0xff << 20 | 0xff << 28 | 0x1fff << 36 | 0x7 << 50 | 0xf << 53))
        | 2 << 28
        | 4 << 8
        | 6 << 20
        | 4 << 36
        | 4 << 50
        | encoding << 53;
    let mut loads = set.iter().map(|&(reg, value)| mov32i(reg, value));
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        loads.next().unwrap_or(NOP),
        loads.next().unwrap_or(NOP),
        loads.next().unwrap_or(NOP),
    );
    assert!(loads.next().is_none(), "three registers at most");
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (tlds as u32, (tlds >> 32) as u32), // tlds nodep $r0 $r2 $r4 $r6 0x4 0xf
        (0x0007000f, 0xe3000000),           // exit
        (0xff87000f, 0xe2400fff),           // bra (padding, never reached)
    ));
    bytes
}

/// `oColor = textureGather(cubeArray, vec4(vColor.xyz, layer), component)` from slot 4.
pub fn gather_cube_array_fragment_shader(component: u32, layer: u32) -> Vec<u8> {
    let tld4_high = 0xc83a0047 | (component & 3) << 24;
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff05, 0xe043ff88), // ipa $r5 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff06, 0xe043ff88), // ipa $r6 a[0x84] $r3 0x0 0x1
        (0x8037ff07, 0xe043ff88), // ipa $r7 a[0x88] $r3 0x0 0x1
        mov32i(4, layer),
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0xfff70400, tld4_high), // tld4 r|g nodep $r0 $r4 0x0 0x4 cube array 0xf
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
    ));
    bytes
}

/// `oColor = textureOffset(texture, vColor.xy, ivec2(1, -1))` from slot 4.
pub fn offset_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x0f17000a, 0x01000000), // mov32i $r10 0xf1: x +1, y -1
        (0xa0a70000, 0xc07a0047), // tex aoffi nodep $r0 $r0 $r10 0x4 t2d 0xf
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra 0x50 (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `oColor = textureOffset(volume, vColor.xyz, offset)` from slot 4, `offset` packed as `tex.aoffi`
/// packs it: a signed nibble per axis, x lowest.
pub fn offset_volume_fragment_shader(offset: u32) -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x8037ff02, 0xe043ff88), // ipa $r2 a[0x88] $r3 0x0 0x1
        mov32i(10, offset),
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0xc0a70000, 0xc07a0047), // tex aoffi nodep $r0 $r0 $r10 0x4 3d 0xf
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
    ));
    bytes
}

/// `oColor = texture(shadowMap, vec3(vColor.xy, 0.5))` from slot 4.
pub fn shadow_fragment_shader() -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff00, 0xe043ff88), // ipa $r0 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff01, 0xe043ff88), // ipa $r1 a[0x84] $r3 0x0 0x1
        (0x0007f002, 0x0103f000), // mov32i $r2 0x3f000000: 0.5
        (0xa0270000, 0xc03e0047), // tex dc nodep $r0 $r0 $r2 0x4 t2d 0xf
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra 0x50 (padding, never reached)
        (0x00070f00, 0x50b00000), // nop (padding, never reached)
    ));
    bytes
}

/// `oColor = texture(shadowMaps, vec4(vColor.xy, layer, 0.5))` from slot 4: an array shadow map.
pub fn shadow_array_fragment_shader(layer: u32) -> Vec<u8> {
    let mut bytes = block(
        (0xe1a0070f, 0x00240401),
        (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
        (0x00470003, 0x50800000), // mufu rcp $r3 $r0
        (0x0037ff01, 0xe043ff88), // ipa $r1 a[0x80] $r3 0x0 0x1
    );
    bytes.extend(block(
        (0xb0400341, 0x055c8400),
        (0x4037ff02, 0xe043ff88), // ipa $r2 a[0x84] $r3 0x0 0x1
        mov32i(4, 0x3f00_0000),   // 0.5
        mov32i(0, layer),
    ));
    bytes.extend(block(
        (0xffe1ffef, 0x001f8000),
        SHADOW_ARRAY_TEX,         // tex dc nodep $r0 $r0 $r4 0x4 a2d 0xf
        (0x0007000f, 0xe3000000), // exit
        (0xff87000f, 0xe2400fff), // bra (padding, never reached)
    ));
    bytes
}

/// [`shadow_array_fragment_shader`]'s sample: the layer in `r0`, then `u` and `v`, the
/// reference in `r4`.
pub const SHADOW_ARRAY_TEX: (u32, u32) = (0xb0470000, 0xc03e0047);

pub const BINDLESS_HANDLE_BANK: u32 = 3;
pub const BINDLESS_HANDLE_OFFSET: u32 = 0x10;

pub const BINDLESS_TEXTURE_SIZE: u32 = 8;

pub const BINDLESS_LAYERS: u32 = 3;

/// What `Harness::bindless_image` binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindlessImage {
    Rgba8,
    Uint,
    Depth,
    Cube,
    DepthCube,
    CubeArray,
    Array,
    Volume,
}

/// Depth at `(x, y)` of `Harness::bindless_depth_texture`: eight values in `(0, 1)` that
/// neighbouring texels rarely share, so a shadow gather of 0.5 sees every pattern of four.
pub fn bindless_depth(x: u32, y: u32) -> f32 {
    (((x * 3 + y * 5) % 8) as f32 + 0.5) / 8.0
}

/// Distinct RGBA8 texel at `(x, y)` of [`Harness::bindless_texture`].
pub fn bindless_texel(x: u32, y: u32) -> u32 {
    let (r, g, b) = (x * 30 + 10, y * 30 + 10, (y * 8 + x) * 3);
    r | g << 8 | b << 16 | 0xff << 24
}

pub const MULTISAMPLE_ENABLE: u32 = 0x54D;
pub const MULTISAMPLE_SAMPLE_LOCATIONS: u32 = 0x478;
pub const MULTISAMPLE_CONTROL: u32 = 0x54F;
pub const MULTISAMPLE_MODE: u32 = 0x574;
pub const MULTISAMPLE_SAMPLE_MASK: u32 = 0x3EF;
pub const DEPTH_TEST_ENABLE: u32 = 0x4B3;
pub const DEPTH_WRITE_ENABLE: u32 = 0x4BA;
pub const DEPTH_TEST_FUNC: u32 = 0x4C3;

/// Write one vertex: `vec4` position at offset 0, `vec4` colour at 16, stride 32.
pub fn write_vertex(
    mem: &Memory,
    vmm: &AddressSpace,
    base: u64,
    index: u32,
    pos: [f32; 4],
    color: [f32; 4],
) {
    let addr = base + index as u64 * 32;
    for (i, v) in pos.iter().enumerate() {
        vmm.write_u32(mem, addr + i as u64 * 4, v.to_bits())
            .unwrap();
    }
    for (i, v) in color.iter().enumerate() {
        vmm.write_u32(mem, addr + 16 + i as u64 * 4, v.to_bits())
            .unwrap();
    }
}

pub struct Harness {
    pub mem: Memory,
    pub vmm: AddressSpace,
    pub engine: Engine3D,
    pub host1x: Host1x,
    pub stats: GpuStats,
    /// Mapping start, also the colour target's address.
    pub base: u64,
    /// The address space the harness stands for; harnesses with different ids are
    /// different processes to a backend.
    pub as_id: u32,
}

pub const TARGET_WIDTH: u32 = 16;
/// Where [`Harness::second_target`] puts its surface, past every other harness surface.
const SECOND_TARGET: u64 = 0x3800;
pub const TARGET_HEIGHT: u32 = 8;

impl Harness {
    pub fn new() -> Harness {
        Harness::with_fragment_shader(solid_fragment_shader())
    }

    pub fn with_fragment_shader(fragment_shader: Vec<u8>) -> Harness {
        let mut mem = Memory::new();
        mem.map_zero(0x7000_0000, 0x4000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm
            .map(0x7000_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();

        let rt_addr = base;
        let vs_addr = base + 0x200;
        let fs_addr = base + 0x300;
        let vbuf_addr = base + 0x400;

        {
            let mut host1x = Host1x::new();
            let mut stats = GpuStats::default();
            let mut ctx = ExecCtx {
                mem: &mut mem,
                vmm: &vmm,
                as_id: 0,
                host1x: &mut host1x,
                stats: &mut stats,
                trace: false,
            };
            for (words, addr) in [
                (passthrough_vertex_shader(), vs_addr),
                (fragment_shader, fs_addr),
            ] {
                for (i, chunk) in words.as_chunks::<4>().0.iter().enumerate() {
                    let word = u32::from_le_bytes(*chunk);
                    ctx.write_u32(addr + i as u64 * 4, word).unwrap();
                }
            }
        }

        let mut engine = Engine3D::new();
        // Render target: 16x8 pitch-linear RGBA8.
        engine.regs.set(0x200, (rt_addr >> 32) as u32);
        engine.regs.set(0x201, rt_addr as u32);
        engine.regs.set(0x202, TARGET_WIDTH * 4);
        engine.regs.set(0x203, TARGET_HEIGHT);
        engine.regs.set(0x204, 0xD5); // RGBA8Unorm
        engine.regs.set(0x205, 1 << 12); // IsLinear
        engine.regs.set(0x206, 1);
        // Viewport 0: x=0, y=0, w=16, h=8.
        engine.regs.set(0x300, TARGET_WIDTH << 16);
        engine.regs.set(0x301, TARGET_HEIGHT << 16);
        engine.regs.set(0x582, (base >> 32) as u32);
        engine.regs.set(0x583, base as u32);
        // SetProgram[VertexB] (StageId 1): enabled, offset 0x200.
        engine.regs.set(0x800 + 0x10, 1 | (1 << 4));
        engine.regs.set(0x800 + 0x11, 0x200);
        engine.regs.set(0x800 + 0x13, 8);
        // SetProgram[Fragment] (StageId 5): enabled, offset 0x300.
        engine.regs.set(0x800 + 5 * 0x10, 1 | (5 << 4));
        engine.regs.set(0x800 + 5 * 0x10 + 1, 0x300);
        engine.regs.set(0x800 + 5 * 0x10 + 3, 8);
        // VertexAttribState[0] = aPosition: buffer 0, offset 0, 4x32 float.
        engine.regs.set(0x458, 0x01 << 21 | 7 << 27);
        // VertexAttribState[1] = aColor: buffer 0, offset 16, 4x32 float.
        engine
            .regs
            .set(0x458 + 1, (16 << 7) | (0x01 << 21) | (7 << 27));
        // VertexArray[0]: stride 32, enabled.
        engine.regs.set(0x700, 32 | (1 << 12));
        engine.regs.set(0x701, (vbuf_addr >> 32) as u32);
        engine.regs.set(0x702, vbuf_addr as u32);
        engine.regs.set(0x7C0, (vbuf_addr >> 32) as u32);
        engine.regs.set(0x7C1, vbuf_addr as u32 + 3 * 32);

        engine.last_draw = DrawCall {
            primitive: 4,
            first: 0,
            count: 3,
            indexed: false,
            index_format: 0,
            base_vertex: 0,
        };
        Harness {
            mem,
            vmm,
            engine,
            host1x: Host1x::new(),
            stats: GpuStats::default(),
            base,
            as_id: 0,
        }
    }

    /// Replace the captured vertex shader.
    pub fn set_vertex_shader(&mut self, words: &[u8]) {
        let mut ctx = ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: self.as_id,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        };
        for (i, chunk) in words.as_chunks::<4>().0.iter().enumerate() {
            let word = u32::from_le_bytes(*chunk);
            ctx.write_u32(self.base + 0x200 + i as u64 * 4, word)
                .expect("the shader fits its mapping");
        }
    }

    pub fn draw_with(&mut self, renderer: &mut dyn Renderer) -> crate::Result<()> {
        let engine = &self.engine;
        let mut ctx = ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: self.as_id,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        };
        renderer.draw(engine, &mut ctx)
    }

    pub fn clear_with(
        &mut self,
        renderer: &mut dyn Renderer,
        channels: [bool; 4],
    ) -> crate::Result<()> {
        let engine = &self.engine;
        let mut ctx = ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: self.as_id,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        };
        renderer.clear_color(engine, &mut ctx, 0, 0, channels)
    }

    /// Clear the depth surface to `ClearDepth` (0x364).
    pub fn clear_depth_with(&mut self, renderer: &mut dyn Renderer) -> crate::Result<()> {
        let engine = &self.engine;
        let mut ctx = ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: self.as_id,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        };
        renderer.clear_depth_stencil(engine, &mut ctx, true, false)
    }

    /// Flush `renderer` until it reports nothing pending.
    pub fn flush_with(&mut self, renderer: &mut dyn Renderer) {
        for _ in 0..64 {
            let mut ctx = ExecCtx {
                mem: &self.mem,
                vmm: &self.vmm,
                as_id: self.as_id,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: false,
            };
            match renderer.flush(&mut ctx) {
                Ok(Flush::Done) => return,
                Ok(Flush::Pending) => continue,
                Err(e) => panic!("flushing: {e:?}"),
            }
        }
        panic!("a renderer never finished handing its surfaces back");
    }

    pub fn ctx(&mut self) -> ExecCtx<'_> {
        ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: self.as_id,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        }
    }

    pub fn vertices(&self) -> u64 {
        self.engine.vertex_array(0).start
    }

    pub fn write_vertex(&mut self, index: u32, pos: [f32; 4], color: [f32; 4]) {
        let base = self.vertices();
        write_vertex(&self.mem, &self.vmm, base, index, pos, color);
    }

    /// A one-colour triangle covering the upper-left half of the target.
    pub fn triangle(&mut self, color: [f32; 4]) {
        self.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], color);
        self.write_vertex(1, [1.0, 1.0, 0.0, 1.0], color);
        self.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], color);
    }

    /// Read the target texel by texel, row-major.
    pub fn target(&mut self) -> Vec<u32> {
        let addr = self.base;
        let Some(rt) = self.engine.render_target(0).unwrap() else {
            return Vec::new();
        };
        let (width, height) = (rt.width, rt.height);
        let ctx = self.ctx();
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                out.push(ctx.read_u32(addr + u64::from(y * width + x) * 4).unwrap());
            }
        }
        out
    }

    /// Read the target as `samples_x` by `samples_y` separate texels per pixel.
    pub fn texel(&mut self, x: u32, y: u32) -> u32 {
        let addr = self.base;
        let width = self.engine.render_target(0).unwrap().unwrap().width;
        self.ctx()
            .read_u32(addr + u64::from(y * width + x) * 4)
            .unwrap()
    }

    /// Move the colour attribute to instanced vertex array 1, filled with `colours`.
    pub fn instanced_colour(&mut self, instance: u32, colours: &[[f32; 4]]) {
        let addr = self.base + 0x800;
        for (i, colour) in colours.iter().enumerate() {
            for (c, v) in colour.iter().enumerate() {
                let at = addr + i as u64 * 16 + c as u64 * 4;
                self.vmm.write_u32(&self.mem, at, v.to_bits()).unwrap();
            }
        }
        // VertexAttribState[1] = aColor: buffer 1, offset 0, 4x32 float.
        self.engine.regs.set(0x459, 1 | (0x01 << 21) | (7 << 27));
        // VertexArray[1]: stride 16, enabled, one element per instance.
        self.engine.regs.set(0x704, 16 | (1 << 12));
        self.engine.regs.set(0x705, (addr >> 32) as u32);
        self.engine.regs.set(0x706, addr as u32);
        self.engine.regs.set(0x707, 1); // divisor

        // VertexArrayPerInstance[1]: without it the engine ignores the divisor.
        self.engine.regs.set(0x621, 1);
        self.engine.regs.set(0x7C2, (addr >> 32) as u32);
        self.engine
            .regs
            .set(0x7C3, addr as u32 + colours.len() as u32 * 16 - 1);
        self.engine.set_instance_id(instance);
    }

    /// Move the colour attribute to vertex array 1 with a zero stride: every vertex reads `colour`.
    pub fn shared_colour(&mut self, colour: [f32; 4]) {
        let addr = self.base + 0x800;
        for (c, v) in colour.iter().enumerate() {
            self.vmm
                .write_u32(&self.mem, addr + c as u64 * 4, v.to_bits())
                .unwrap();
        }
        // VertexAttribState[1] = aColor: buffer 1, offset 0, 4x32 float.
        self.engine.regs.set(0x459, 1 | (0x01 << 21) | (7 << 27));
        // VertexArray[1]: stride 0, enabled, per vertex.
        self.engine.regs.set(0x704, 1 << 12);
        self.engine.regs.set(0x705, (addr >> 32) as u32);
        self.engine.regs.set(0x706, addr as u32);
        self.engine.regs.set(0x7C2, (addr >> 32) as u32);
        self.engine.regs.set(0x7C3, addr as u32 + 15);
    }

    /// Bind a `Z24S8` depth surface after the colour target and enable the depth test.
    pub fn depth_target(&mut self, func: u32) {
        self.depth_target_sized(func, TARGET_WIDTH, TARGET_HEIGHT);
    }

    /// [`Harness::depth_target`] with its own extent.
    pub fn depth_target_sized(&mut self, func: u32, width: u32, height: u32) {
        let addr = self.base + 0x1000;
        self.engine.regs.set(0x3F8, (addr >> 32) as u32);
        self.engine.regs.set(0x3F9, addr as u32);
        self.engine.regs.set(0x3FA, 0x14); // Z24S8
        self.engine.regs.set(0x3FB, 0); // one GOB per block
        self.engine.regs.set(0x48A, width);
        self.engine.regs.set(0x48B, height);
        self.engine.regs.set(DEPTH_TEST_ENABLE, 1);
        self.engine.regs.set(DEPTH_WRITE_ENABLE, 1);
        self.engine.regs.set(DEPTH_TEST_FUNC, func);
    }

    /// Read the depth surface texel by texel, empty if none is bound.
    pub fn depth(&mut self) -> Vec<u32> {
        let addr = self.base + 0x1000;
        let Some(target) = self.engine.depth_target().unwrap() else {
            return Vec::new();
        };
        let (width, height, layout) = (target.width, target.height, target.layout);
        let ctx = self.ctx();
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let offset = layout.offset(x * 4, y, width * 4);
                out.push(ctx.read_u32(addr + u64::from(offset)).unwrap());
            }
        }
        out
    }

    /// Bind a second 16x8 pitch-linear RGBA8 target at slot 1, every texel `fill`, and have
    /// draws write `outputs` colour outputs, each to the slot of its own number.
    pub fn second_target(&mut self, fill: u32, outputs: u32) {
        let addr = self.base + SECOND_TARGET;
        let base = 0x200 + 0x10;
        self.engine.regs.set(base, (addr >> 32) as u32);
        self.engine.regs.set(base + 1, addr as u32);
        self.engine.regs.set(base + 2, TARGET_WIDTH * 4);
        self.engine.regs.set(base + 3, TARGET_HEIGHT);
        self.engine.regs.set(base + 4, 0xD5); // RGBA8Unorm
        self.engine.regs.set(base + 5, 1 << 12); // IsLinear
        self.engine.regs.set(base + 6, 1);
        let identity: u32 = (0..8).map(|slot| slot << (4 + 3 * slot)).sum();
        self.engine.regs.set(0x487, outputs | identity);
        let mut ctx = self.ctx();
        for texel in 0..TARGET_WIDTH * TARGET_HEIGHT {
            ctx.write_u32(addr + u64::from(texel) * 4, fill)
                .expect("the target fits its mapping");
        }
    }

    /// Read [`Harness::second_target`]'s surface texel by texel, row-major.
    pub fn second_target_texels(&mut self) -> Vec<u32> {
        let addr = self.base + SECOND_TARGET;
        let ctx = self.ctx();
        (0..TARGET_WIDTH * TARGET_HEIGHT)
            .map(|texel| ctx.read_u32(addr + u64::from(texel) * 4).unwrap())
            .collect()
    }

    /// Bind an 8x8 RGBA8 image for [`bindless_fragment_shader`] as image 1 and sampler 1,
    /// so a backend reading a zero handle samples the wrong descriptor.
    pub fn bindless_texture(&mut self) {
        self.bindless_image(BindlessImage::Rgba8);
    }

    /// [`Harness::bindless_texture`] as an `RGBA32Uint` image whose words are the bits of
    /// each RGBA8 texel's channels as floats, so a shader that passes them on draws the same colours.
    pub fn bindless_uint_texture(&mut self) {
        self.bindless_image(BindlessImage::Uint);
    }

    /// [`Harness::bindless_texture`] as a shadow map: `ZF32` depths [`bindless_depth`] under a
    /// `Less` comparison.
    pub fn bindless_depth_texture(&mut self) {
        self.bindless_image(BindlessImage::Depth);
    }

    /// [`Harness::bindless_texture`] as a cubemap whose face `f` holds [`bindless_texel`]
    /// shifted `f` texels along.
    pub fn bindless_cube_texture(&mut self) {
        self.bindless_image(BindlessImage::Cube);
    }

    /// [`Harness::bindless_depth_texture`] as a cubemap whose face `f` holds [`bindless_depth`]
    /// shifted `f` texels along.
    pub fn bindless_depth_cube_texture(&mut self) {
        self.bindless_image(BindlessImage::DepthCube);
    }

    /// [`Harness::bindless_texture`] as a cubemap array of two cubes: layer `l` holds
    /// [`bindless_texel`] shifted `l` texels along.
    pub fn bindless_cube_array_texture(&mut self) {
        self.bindless_image(BindlessImage::CubeArray);
    }

    /// [`Harness::bindless_texture`] as a 2D array of [`BINDLESS_LAYERS`] layers, layer `l`
    /// holding [`bindless_texel`] shifted `l` texels along.
    pub fn bindless_array_texture(&mut self) {
        self.bindless_image(BindlessImage::Array);
    }

    /// [`Harness::bindless_array_texture`] as a 3D image of [`BINDLESS_LAYERS`] slices.
    pub fn bindless_volume_texture(&mut self) {
        self.bindless_image(BindlessImage::Volume);
    }

    fn bindless_image(&mut self, kind: BindlessImage) {
        let header_pool = self.base + 0x1400;
        let sampler_pool = self.base + 0x1480;
        let constants = self.base + 0x1500;
        let image = self.base + 0x1800;
        let size = BINDLESS_TEXTURE_SIZE;
        let texel_bytes = if kind == BindlessImage::Uint { 16 } else { 4 };
        let depth = matches!(kind, BindlessImage::Depth | BindlessImage::DepthCube);
        // TIC `TextureType`, the layers stored, and `DEPTH_MINUS_ONE`.
        let (texture_type, faces, depth_minus_one) = match kind {
            BindlessImage::Cube | BindlessImage::DepthCube => (3, 6, 0),
            BindlessImage::CubeArray => (8, 12, 1),
            BindlessImage::Array => (5, BINDLESS_LAYERS, BINDLESS_LAYERS - 1),
            BindlessImage::Volume => (2, BINDLESS_LAYERS, BINDLESS_LAYERS - 1),
            _ => (1, 1, 0),
        };
        let layer_bytes = size * size * texel_bytes;
        let mut ctx = self.ctx();
        for face in 0..faces {
            for y in 0..size {
                for x in 0..size {
                    let at = image + u64::from(face * layer_bytes + (y * size + x) * texel_bytes);
                    let texel = bindless_texel((x + face) % size, y);
                    match kind {
                        BindlessImage::Uint => {
                            for channel in 0..4u32 {
                                let value = ((texel >> (8 * channel)) & 0xff) as f32 / 255.0;
                                ctx.write_u32(at + 4 * u64::from(channel), value.to_bits())
                                    .unwrap();
                            }
                        }
                        BindlessImage::Depth | BindlessImage::DepthCube => {
                            let depth = bindless_depth((x + face) % size, y);
                            ctx.write_u32(at, depth.to_bits()).unwrap();
                        }
                        _ => {
                            ctx.write_u32(at, texel).unwrap();
                        }
                    }
                }
            }
        }
        // TIC 1, pitch-linear with the identity swizzle: A8B8G8R8 UNORM, R32_G32_B32_A32 UINT,
        // ZF32 or a cube of A8B8G8R8.
        let tic = header_pool + 32;
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        let format = match kind {
            BindlessImage::Uint => 0x01 | (4 << 7),
            BindlessImage::Depth | BindlessImage::DepthCube => 0x2f | (7 << 7),
            _ => 0x08 | (2 << 7),
        };
        ctx.write_u32(tic, format | identity).unwrap();
        ctx.write_u32(tic + 4, image as u32).unwrap();
        ctx.write_u32(tic + 8, (image >> 32) as u32 | (2 << 21))
            .unwrap();
        ctx.write_u32(tic + 12, size * texel_bytes / 32).unwrap();
        ctx.write_u32(tic + 16, (size - 1) | (texture_type << 23))
            .unwrap();
        ctx.write_u32(tic + 20, (size - 1) | (depth_minus_one << 16))
            .unwrap();
        // TSC 1: ClampToEdge on all three axes, nearest filtering, and a `Less` comparison
        // for a shadow map.
        let compare = if depth { 1 << 9 | 1 << 10 } else { 0 };
        ctx.write_u32(sampler_pool + 32, 2 | (2 << 3) | (2 << 6) | compare)
            .unwrap();
        let handle = 1 | (1 << 20);
        ctx.write_u32(constants + u64::from(BINDLESS_HANDLE_OFFSET), handle)
            .unwrap();

        // SetTexHeaderPool and SetTexSamplerPool, address high then low.
        self.engine.regs.set(0x55D, (header_pool >> 32) as u32);
        self.engine.regs.set(0x55E, header_pool as u32);
        self.engine.regs.set(0x557, (sampler_pool >> 32) as u32);
        self.engine.regs.set(0x558, sampler_pool as u32);
        // Bind the constant buffer to the fragment stage's slot 4.
        let mut ctx = ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: self.as_id,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        };
        for (method, arg) in [
            (0x8E0, 0x100),
            (0x8E1, (constants >> 32) as u32),
            (0x8E2, constants as u32),
            (0x900 + 4 * 8 + 4, 1 | (BINDLESS_HANDLE_BANK << 4)),
        ] {
            self.engine.write(method, arg, true, &mut ctx).unwrap();
        }
    }

    /// [`Harness::bindless_texture`] made block-linear with a second, 4x4 level whose texel
    /// `(x, y)` is [`bindless_texel`]`(7 - x, 7 - y)`.
    pub fn bindless_mipmapped_texture(&mut self) {
        self.bindless_texture();
        let tic = self.base + 0x1400 + 32;
        let image = self.base + 0x1800;
        let size = BINDLESS_TEXTURE_SIZE;
        let mut ctx = self.ctx();
        // BLOCKLINEAR, one GOB high, two levels.
        ctx.write_u32(tic + 8, (image >> 32) as u32 | (3 << 21))
            .unwrap();
        ctx.write_u32(tic + 12, 1 << 28).unwrap();
        let texture = crate::gpu::texture::read_image(&ctx, tic).unwrap();
        let level = texture.level(1).unwrap();
        for (surface, extent, texel) in [
            (
                &texture,
                size,
                (|x, y| bindless_texel(x, y)) as fn(u32, u32) -> u32,
            ),
            (&level, size / 2, |x, y| bindless_texel(7 - x, 7 - y)),
        ] {
            for y in 0..extent {
                for x in 0..extent {
                    let at = surface.layout.offset(x * 4, y, extent * 4);
                    ctx.write_u32(surface.addr + u64::from(at), texel(x, y))
                        .unwrap();
                }
            }
        }
    }

    /// Multisample the target: on Maxwell the pixel extent shrinks, the surface does not.
    pub fn multisample(&mut self, mode: u32, samples_x: u32, samples_y: u32) {
        self.engine.regs.set(MULTISAMPLE_ENABLE, 1);
        self.engine.regs.set(MULTISAMPLE_MODE, mode);
        self.engine
            .regs
            .set(0x300, (TARGET_WIDTH / samples_x) << 16);
        self.engine
            .regs
            .set(0x301, (TARGET_HEIGHT / samples_y) << 16);
    }
}

impl Default for Harness {
    fn default() -> Harness {
        Harness::new()
    }
}

/// A compute launch in memory of its own: its QMD, its program, constant bank 0 holding an
/// NVN storage-buffer descriptor for the output, and the output buffer.
pub struct ComputeHarness {
    pub mem: Memory,
    pub vmm: AddressSpace,
    pub host1x: Host1x,
    pub stats: GpuStats,
    pub base: u64,
}

const COMPUTE_QMD_AT: u64 = 0;
const COMPUTE_PROGRAM_AT: u64 = 0x1000;
const COMPUTE_OUTPUT_AT: u64 = 0x2000;
const COMPUTE_BANK_AT: u64 = 0x3000;
const COMPUTE_BANK_SIZE: u32 = 0x100;
/// Where bank 0 holds the output's descriptor: address, then size.
const COMPUTE_DESCRIPTOR: u32 = 0x40;
/// Where bank 0 holds the constant [`indexed_store_program`] adds.
const COMPUTE_ADDEND: u32 = 0x50;

impl ComputeHarness {
    /// `program` launched as `grid` CTAs of `block` threads, with an output of `output_words`.
    pub fn new(program: &[u64], grid: [u32; 3], block: [u32; 3], output_words: u32) -> Self {
        let mem = Memory::new();
        mem.map_zero(0x7100_0000, 0x4000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm
            .map(0x7100_0000, 0x4000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let mut harness = ComputeHarness {
            mem,
            vmm,
            host1x: Host1x::new(),
            stats: GpuStats::default(),
            base,
        };
        let output = base + COMPUTE_OUTPUT_AT;
        let mut ctx = harness.ctx();
        write_launch(
            &mut ctx,
            base + COMPUTE_QMD_AT,
            program,
            grid,
            block,
            &[
                (COMPUTE_DESCRIPTOR, output as u32),
                (COMPUTE_DESCRIPTOR + 4, (output >> 32) as u32),
                (COMPUTE_DESCRIPTOR + 8, output_words * 4),
                (COMPUTE_ADDEND, 7),
            ],
        );
        harness
    }

    pub fn ctx(&mut self) -> ExecCtx<'_> {
        ExecCtx {
            mem: &self.mem,
            vmm: &self.vmm,
            as_id: 0,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        }
    }

    /// The compute engine, its program region set and the launch recorded.
    pub fn engine(&self) -> crate::gpu::engine::compute::EngineCompute {
        let mut engine = crate::gpu::engine::compute::EngineCompute::new();
        let program = self.base + COMPUTE_PROGRAM_AT;
        engine.regs.set(0x582, (program >> 32) as u32);
        engine.regs.set(0x583, program as u32);
        engine.last_dispatch = Some(crate::gpu::engine::compute::Dispatch {
            qmd_addr: self.base + COMPUTE_QMD_AT,
        });
        engine
    }

    pub fn output(&mut self, words: usize) -> Vec<u32> {
        let at = self.base + COMPUTE_OUTPUT_AT;
        let ctx = self.ctx();
        (0..words)
            .map(|i| ctx.read_u32(at + i as u64 * 4).unwrap())
            .collect()
    }

    /// The output's GPU address.
    pub fn output_addr(&self) -> u64 {
        self.base + COMPUTE_OUTPUT_AT
    }

    /// Lay out what [`image_copy_program`] reads in the output's place: an
    /// [`IMAGE_COPY_SIZE`] pitch-linear `A8B8G8R8` texture of [`image_copy_texel`]s and an
    /// image of the same shape, cleared. Answers the engine with their pools set.
    pub fn image_copy(&mut self) -> crate::gpu::engine::compute::EngineCompute {
        let (width, height) = IMAGE_COPY_SIZE;
        let tic = self.base + IMAGE_COPY_TIC_AT;
        let tsc = self.base + IMAGE_COPY_TSC_AT;
        let texture = self.base + IMAGE_COPY_TEXTURE_AT;
        let image = self.image_copy_addr();
        let bank = self.base + COMPUTE_BANK_AT;
        let mut ctx = self.ctx();
        for y in 0..height {
            for x in 0..width {
                let at = u64::from((y * width + x) * 4);
                ctx.write_u32(texture + at, image_copy_texel(x, y)).unwrap();
                ctx.write_u32(image + at, 0).unwrap();
            }
        }
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        for (entry, addr) in [(1u64, texture), (2, image)] {
            let at = tic + entry * 32;
            ctx.write_u32(at, 0x08 | (2 << 7) | identity).unwrap();
            ctx.write_u32(at + 4, addr as u32).unwrap();
            ctx.write_u32(at + 8, (addr >> 32) as u32 | (2 << 21))
                .unwrap();
            ctx.write_u32(at + 12, width * 4 / 32).unwrap();
            ctx.write_u32(at + 16, (width - 1) | (1 << 23)).unwrap();
            ctx.write_u32(at + 20, height - 1).unwrap();
        }
        // TSC 1: clamped to the edge on every axis, nearest filtering.
        ctx.write_u32(tsc + 32, 2 | (2 << 3) | (2 << 6)).unwrap();
        for (offset, word) in [
            (IMAGE_COPY_TEXTURE, 1 | (1 << 20)),
            (IMAGE_COPY_IMAGE, 2),
            (IMAGE_COPY_U, (1.0f32 / 128.0).to_bits()),
            (IMAGE_COPY_V, (1.0f32 / 8.0).to_bits()),
            (IMAGE_SIZE_LEVEL, 1),
            (IMAGE_SIZE_HANDLE, 2),
            (IMAGE_COPY_LAYER, 0),
        ] {
            ctx.write_u32(bank + u64::from(offset), word).unwrap();
        }
        let mut engine = self.engine();
        engine.regs.set(0x55D, (tic >> 32) as u32);
        engine.regs.set(0x55E, tic as u32);
        engine.regs.set(0x557, (tsc >> 32) as u32);
        engine.regs.set(0x558, tsc as u32);
        engine
    }

    /// Point [`image_copy_program`]'s texture handle at TIC 3, which is all zero: a null image.
    pub fn null_texture(&mut self) {
        let at = self.base + COMPUTE_BANK_AT + u64::from(IMAGE_COPY_TEXTURE);
        self.ctx().write_u32(at, 3 | (1 << 20)).unwrap();
    }

    /// Make [`Self::image_copy`]'s image `R32_UINT`.
    pub fn image_copy_words(&mut self) {
        let tic = self.base + IMAGE_COPY_TIC_AT + 2 * 32;
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        self.ctx()
            .write_u32(tic, 0x0f | (4 << 7) | identity)
            .unwrap();
    }

    /// Make [`Self::image_copy`]'s image `B10G11R11_FLOAT`, which WebGPU cannot store to.
    pub fn image_copy_packed_float(&mut self) {
        let tic = self.base + IMAGE_COPY_TIC_AT + 2 * 32;
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        self.ctx()
            .write_u32(tic, 0x21 | (7 << 7) | identity)
            .unwrap();
    }

    /// Make [`Self::image_copy`]'s image a 2D array of `layers` cleared layers, each
    /// [`IMAGE_COPY_SIZE`] and one after another.
    pub fn image_copy_layers(&mut self, layers: u32) {
        let (width, height) = IMAGE_COPY_SIZE;
        let tic = self.base + IMAGE_COPY_TIC_AT + 2 * 32;
        let image = self.image_copy_addr();
        let mut ctx = self.ctx();
        for word in 0..layers * width * height {
            ctx.write_u32(image + u64::from(word) * 4, 0).unwrap();
        }
        // `TextureType` 5, a 2D array, `DEPTH_MINUS_ONE` layers after the first.
        ctx.write_u32(tic + 16, (width - 1) | (5 << 23)).unwrap();
        ctx.write_u32(tic + 20, (height - 1) | ((layers - 1) << 16))
            .unwrap();
    }

    /// Layer `layer` of [`Self::image_copy_layers`]'s image, row by row.
    pub fn image_layer(&mut self, layer: u32) -> Vec<u32> {
        let (width, height) = IMAGE_COPY_SIZE;
        let at = self.image_copy_addr() + u64::from(layer * width * height * 4);
        let ctx = self.ctx();
        (0..width * height)
            .map(|i| ctx.read_u32(at + u64::from(i) * 4).unwrap())
            .collect()
    }

    /// Set the layer [`image_copy_array_program`] stores to, after [`Self::image_copy`].
    pub fn set_image_layer(&mut self, layer: u32) {
        let at = self.base + COMPUTE_BANK_AT + u64::from(IMAGE_COPY_LAYER);
        self.ctx().write_u32(at, layer).unwrap();
    }

    /// Replace the launch's program, leaving the rest as it is.
    pub fn set_program(&mut self, program: &[u64]) {
        let at = self.base + COMPUTE_PROGRAM_AT;
        let mut ctx = self.ctx();
        for (i, word) in laid_out(program).iter().enumerate() {
            ctx.write_u32(at + i as u64 * 4, *word).unwrap();
        }
    }

    /// The image [`ComputeHarness::image_copy`] lays out.
    pub fn image_copy_addr(&self) -> u64 {
        self.base + IMAGE_COPY_IMAGE_AT
    }

    /// The image's texels, row by row.
    pub fn image(&mut self) -> Vec<u32> {
        let (width, height) = IMAGE_COPY_SIZE;
        let at = self.image_copy_addr();
        let ctx = self.ctx();
        (0..width * height)
            .map(|i| ctx.read_u32(at + u64::from(i) * 4).unwrap())
            .collect()
    }
}

/// The texture and image [`ComputeHarness::image_copy`] lays out, and the launch's block.
pub const IMAGE_COPY_SIZE: (u32, u32) = (8, 4);
const IMAGE_COPY_TIC_AT: u64 = 0x2000;
const IMAGE_COPY_TSC_AT: u64 = 0x2080;
const IMAGE_COPY_TEXTURE_AT: u64 = 0x2100;
const IMAGE_COPY_IMAGE_AT: u64 = 0x2200;

/// Texel `(x, y)` of the texture [`image_copy_program`] samples, every byte distinct.
pub fn image_copy_texel(x: u32, y: u32) -> u32 {
    u32::from_le_bytes([
        (x * 30 + 1) as u8,
        (y * 60 + 2) as u8,
        (x + y * 8 + 100) as u8,
        255,
    ])
}

/// Lay a launch out from `at`: its QMD, its program at `at + 0x1000` and constant bank 0 at
/// `at + 0x3000` holding `bank_words` (byte offset, word). Answers the compute engine set to
/// launch it.
pub fn write_launch(
    ctx: &mut ExecCtx,
    at: u64,
    program: &[u64],
    grid: [u32; 3],
    block: [u32; 3],
    bank_words: &[(u32, u32)],
) -> crate::gpu::engine::compute::EngineCompute {
    let program_at = at + COMPUTE_PROGRAM_AT;
    let bank_at = at + COMPUTE_BANK_AT;
    let mut write = |from: u64, words: &[u32]| {
        for (i, word) in words.iter().enumerate() {
            ctx.write_u32(from + i as u64 * 4, *word).unwrap();
        }
    };
    write(at, &qmd_words(grid, block, bank_at));
    write(program_at, &laid_out(program));
    for &(offset, word) in bank_words {
        write(bank_at + u64::from(offset), &[word]);
    }
    let mut engine = crate::gpu::engine::compute::EngineCompute::new();
    engine.regs.set(0x582, (program_at >> 32) as u32);
    engine.regs.set(0x583, program_at as u32);
    engine.last_dispatch = Some(crate::gpu::engine::compute::Dispatch { qmd_addr: at });
    engine
}

/// A program's words as memory holds them: a scheduling word before every three.
fn laid_out(program: &[u64]) -> Vec<u32> {
    program
        .chunks(3)
        .flat_map(|chunk| std::iter::once(0).chain(chunk.iter().copied()))
        .flat_map(|w: u64| [w as u32, (w >> 32) as u32])
        .collect()
}

/// A version 2.x QMD launching `grid` CTAs of `block` threads with bank 0 bound at `bank`.
fn qmd_words(grid: [u32; 3], block: [u32; 3], bank: u64) -> Vec<u32> {
    let mut words = [0u32; crate::gpu::qmd::QMD_WORDS];
    let mut set = |lo: u32, hi: u32, value: u32| {
        for bit in 0..=(hi - lo) {
            let at = (lo + bit) as usize;
            if value >> bit & 1 != 0 {
                words[at / 32] |= 1 << (at % 32);
            }
        }
    };
    set(576, 579, 6);
    set(384, 415, grid[0]);
    set(416, 431, grid[1]);
    set(432, 447, grid[2]);
    set(592, 607, block[0]);
    set(608, 623, block[1]);
    set(624, 639, block[2]);
    set(640, 640, 1);
    set(928, 959, bank as u32);
    set(960, 967, (bank >> 32) as u32);
    set(975, 991, COMPUTE_BANK_SIZE);
    words.to_vec()
}

/// Thread `tid` copies word `tid` of the buffer bank 0 describes at `0x60` into the one it
/// describes at `0x40`.
pub fn copy_program() -> Vec<u64> {
    use crate::gpu::shader::isa::{self, MemSize, Operand, RZ};
    use crate::gpu::shader::Op;
    const PT: u64 = 7 << 16;
    let check = |word: u64, expected: Op| {
        assert_eq!(isa::decode(word).op, expected, "encoded {word:#018x}");
        word
    };
    let iadd = |dst: u8, a: u8, offset: u16, cin: bool, cout: bool| {
        check(
            0x4c10u64 << 48
                | u64::from(offset >> 2) << 20
                | u64::from(cin) << 43
                | u64::from(cout) << 47
                | PT
                | u64::from(a) << 8
                | u64::from(dst),
            Op::Iadd {
                dst,
                a,
                aneg: false,
                b: Operand::Const { bank: 0, offset },
                bneg: false,
                cin,
                cout,
            },
        )
    };
    let memory = |opcode: u64, data: u8, addr: u8| {
        (opcode | 4) << 48 | PT | u64::from(addr) << 8 | u64::from(data)
    };
    vec![
        check(
            0xf0c8u64 << 48 | 0x21 << 20 | PT,
            Op::S2r { dst: 0, sr: 0x21 },
        ),
        check(
            0x5c18u64 << 48 | u64::from(RZ) << 20 | 2 << 39 | PT | 5,
            Op::Iscadd {
                dst: 5,
                a: 0,
                aneg: false,
                b: Operand::Reg(RZ),
                bneg: false,
                shift: 2,
                cout: false,
            },
        ),
        iadd(6, 5, 0x60, false, true),
        iadd(7, RZ, 0x64, true, false),
        check(
            memory(0xeed0, 8, 6),
            Op::Ldg {
                dst: 8,
                addr: 6,
                offset: 0,
                size: MemSize::B32,
            },
        ),
        iadd(10, 5, COMPUTE_DESCRIPTOR as u16, false, true),
        iadd(11, RZ, COMPUTE_DESCRIPTOR as u16 + 4, true, false),
        check(
            memory(0xeed8, 8, 10),
            Op::Stg {
                addr: 10,
                offset: 0,
                src: 8,
                size: MemSize::B32,
            },
        ),
        check(0xe300_0000_0000_0000 | PT | 0xF, Op::Exit),
    ]
}

/// Every thread of a launch of four-thread CTAs stores `index * 4 + c0[0x50]` at word
/// `index`, `index` being its CTA's base plus its thread id, through bank 0's descriptor.
pub fn indexed_store_program() -> Vec<u64> {
    use crate::gpu::shader::isa::{self, MemSize, Operand, RZ};
    use crate::gpu::shader::Op;
    const PT: u64 = 7 << 16;
    let check = |word: u64, expected: Op| {
        assert_eq!(isa::decode(word).op, expected, "encoded {word:#018x}");
        word
    };
    let s2r = |dst: u8, sr: u8| {
        check(
            0xf0c8u64 << 48 | u64::from(sr) << 20 | PT | u64::from(dst),
            Op::S2r { dst, sr },
        )
    };
    // `(a << shift) + b`, `b` a register or a bank word.
    let iscadd = |dst: u8, a: u8, b: Operand, shift: u8| {
        let form = match b {
            Operand::Reg(r) => 0x5c18u64 << 48 | u64::from(r) << 20,
            Operand::Const { bank, offset } => {
                0x4c18u64 << 48 | u64::from(bank) << 34 | u64::from(offset >> 2) << 20
            }
            Operand::Imm(_) => unreachable!("not used here"),
        };
        check(
            form | u64::from(shift) << 39 | PT | u64::from(a) << 8 | u64::from(dst),
            Op::Iscadd {
                dst,
                a,
                aneg: false,
                b,
                bneg: false,
                shift,
                cout: false,
            },
        )
    };
    // `a + c0[offset]`, carrying out or in.
    let iadd = |dst: u8, a: u8, offset: u16, cin: bool, cout: bool| {
        check(
            0x4c10u64 << 48
                | u64::from(offset >> 2) << 20
                | u64::from(cin) << 43
                | u64::from(cout) << 47
                | PT
                | u64::from(a) << 8
                | u64::from(dst),
            Op::Iadd {
                dst,
                a,
                aneg: false,
                b: Operand::Const { bank: 0, offset },
                bneg: false,
                cin,
                cout,
            },
        )
    };
    let descriptor = COMPUTE_DESCRIPTOR as u16;
    vec![
        s2r(0, 0x21),
        s2r(1, 0x25),
        iscadd(0, 1, Operand::Reg(0), 2),
        iscadd(5, 0, Operand::Reg(RZ), 2),
        iadd(6, 5, descriptor, false, true),
        iadd(7, RZ, descriptor + 4, true, false),
        iscadd(
            8,
            0,
            Operand::Const {
                bank: 0,
                offset: COMPUTE_ADDEND as u16,
            },
            2,
        ),
        check(
            (0xeed8u64 | 4) << 48 | PT | 6 << 8 | 8,
            Op::Stg {
                addr: 6,
                offset: 0,
                src: 8,
                size: MemSize::B32,
            },
        ),
        check(0xe300_0000_0000_0000 | PT | 0xF, Op::Exit),
    ]
}

/// Bank 0 words of [`image_copy_program`]: the texture's and the image's handles, then the
/// float bits that scale each thread's x and y into a texture coordinate.
pub const IMAGE_COPY_TEXTURE: u32 = 0x60;
pub const IMAGE_COPY_IMAGE: u32 = 0x64;
pub const IMAGE_COPY_U: u32 = 0x68;
pub const IMAGE_COPY_V: u32 = 0x6c;
/// The layer [`image_copy_array_program`] stores to; zero unless a test sets it.
pub const IMAGE_COPY_LAYER: u32 = 0x74;

/// Assert `word` decodes as `expected`.
fn checked(word: u64, expected: crate::gpu::shader::Op) -> u64 {
    assert_eq!(
        crate::gpu::shader::isa::decode(word).op,
        expected,
        "encoded {word:#018x}"
    );
    word
}

const PT: u64 = 7 << 16;

/// `ldc dst, c0[offset]`.
fn ldc0(dst: u8, offset: u32) -> u64 {
    use crate::gpu::shader::isa::{MemSize, RZ};
    checked(
        0xef94u64 << 48 | u64::from(offset) << 20 | PT | u64::from(RZ) << 8 | u64::from(dst),
        crate::gpu::shader::Op::Ldc {
            dst,
            bank: 0,
            offset: offset as i32,
            idx: RZ,
            size: MemSize::B32,
        },
    )
}

/// `sust.p.2d.rgba [coords], src, handle_reg`.
fn sust_p_2d(src: u8, coords: u8, handle_reg: u8) -> u64 {
    sust_p(
        crate::gpu::shader::isa::SurfaceDim::D2,
        src,
        coords,
        handle_reg,
    )
}

/// `sust.p.rgba [coords], src, handle_reg` of `dim`, a 2D image or array.
fn sust_p(dim: crate::gpu::shader::isa::SurfaceDim, src: u8, coords: u8, handle_reg: u8) -> u64 {
    use crate::gpu::shader::isa::{SurfaceData, SurfaceDim};
    let code = match dim {
        SurfaceDim::D2 => 3,
        SurfaceDim::Array2d => 4,
        other => panic!("no encoding here for a {other:?} store"),
    };
    checked(
        0xeb20u64 << 48
            | u64::from(handle_reg) << 39
            | code << 33
            | 0xf << 20
            | PT
            | u64::from(coords) << 8
            | u64::from(src),
        crate::gpu::shader::Op::Sust {
            src,
            coords,
            handle: 0,
            handle_reg: Some(handle_reg),
            dim,
            data: SurfaceData::Formatted([true; 4]),
        },
    )
}

fn exit_word() -> u64 {
    checked(
        0xe300_0000_0000_0000 | PT | 0xF,
        crate::gpu::shader::Op::Exit,
    )
}

/// Thread `(x, y)` samples the bindless texture at `(2^x * u, 2^y * v)` and stores the texel
/// to texel `(x, y)` of the bindless image.
pub fn image_copy_program() -> Vec<u64> {
    image_copy_program_as(crate::gpu::shader::isa::SurfaceDim::D2)
}

/// [`image_copy_program`], storing as a 2D array to the layer bank word
/// [`IMAGE_COPY_LAYER`] holds, as Pokémon Legends: Z-A does.
pub fn image_copy_array_program() -> Vec<u64> {
    image_copy_program_as(crate::gpu::shader::isa::SurfaceDim::Array2d)
}

fn image_copy_program_as(dim: crate::gpu::shader::isa::SurfaceDim) -> Vec<u64> {
    use crate::gpu::shader::isa::{Operand, TexDim};
    use crate::gpu::shader::Op;
    let s2r = |dst: u8, sr: u8| {
        checked(
            0xf0c8u64 << 48 | u64::from(sr) << 20 | PT | u64::from(dst),
            Op::S2r { dst, sr },
        )
    };
    // `(a << 23) + c0[offset]`: a float's exponent raised by `a`.
    let scale = |dst: u8, a: u8, offset: u32| {
        let b = Operand::Const {
            bank: 0,
            offset: offset as u16,
        };
        checked(
            0x4c18u64 << 48
                | u64::from(offset >> 2) << 20
                | 23 << 39
                | PT
                | u64::from(a) << 8
                | u64::from(dst),
            Op::Iscadd {
                dst,
                a,
                aneg: false,
                b,
                bneg: false,
                shift: 23,
                cout: false,
            },
        )
    };
    vec![
        s2r(8, 0x21),
        s2r(9, 0x22),
        scale(0, 8, IMAGE_COPY_U),
        scale(1, 9, IMAGE_COPY_V),
        ldc0(2, IMAGE_COPY_TEXTURE),
        checked(
            0xdeba_0007_a027_0000,
            Op::Tex {
                dst: 0,
                coords: [0, 1, 2],
                layer: None,
                dref: None,
                offset: None,
                lod: None,
                handle: 0,
                handle_reg: Some(2),
                dim: TexDim::T2d,
                mask: [true; 4],
            },
        ),
        ldc0(10, IMAGE_COPY_LAYER),
        ldc0(39, IMAGE_COPY_IMAGE),
        sust_p(dim, 0, 8, 39),
        exit_word(),
    ]
}

/// Each thread queries the image's size through bound texture 8, at the level bank word
/// [`IMAGE_SIZE_LEVEL`] holds, as Echoes of Wisdom's cubemap filter does, and stores `u` in
/// every channel to the texel at that size.
pub fn image_size_program() -> Vec<u64> {
    use crate::gpu::shader::Op;
    vec![
        ldc0(8, IMAGE_SIZE_LEVEL),
        checked(
            0xdf48_0081_8047_0800,
            Op::Txq {
                dst: 0,
                lod: 8,
                handle: 8,
                mask: [true, true, false, false],
            },
        ),
        ldc0(4, IMAGE_COPY_U),
        ldc0(5, IMAGE_COPY_U),
        ldc0(6, IMAGE_COPY_U),
        ldc0(7, IMAGE_COPY_U),
        ldc0(39, IMAGE_COPY_IMAGE),
        sust_p_2d(4, 0, 39),
        exit_word(),
    ]
}

/// Each of [`IMAGE_COPY_SIZE`]'s threads reduces into the bindless image, made `R32_UINT` by
/// [`ComputeHarness::image_copy_words`]: it adds one to texel `(0, 0)`, raises `(1, 0)` to its
/// x and increments `(2, 0)` wrapping past two, so only the thread count decides what lands.
pub fn image_reduce_program() -> Vec<u64> {
    use crate::gpu::shader::isa::{AtomOp, AtomType, SurfaceDim};
    use crate::gpu::shader::Op;
    let sured = |op: AtomOp, code: u64, coords: u8, src: u8| {
        checked(
            0xeb50u64 << 48
                | 39 << 39
                | 3 << 33
                | code << 24
                | PT
                | u64::from(coords) << 8
                | u64::from(src),
            Op::Sured {
                src,
                coords,
                handle: 0,
                handle_reg: Some(39),
                dim: SurfaceDim::D2,
                op,
                ty: AtomType::U32,
            },
        )
    };
    let tid_x = checked(
        0xf0c8u64 << 48 | 0x21 << 20 | PT,
        Op::S2r { dst: 0, sr: 0x21 },
    );
    // Bank words holding 0, 1 and 2.
    let (zero, one, two) = (IMAGE_COPY_LAYER, IMAGE_SIZE_LEVEL, IMAGE_SIZE_HANDLE);
    vec![
        tid_x,
        ldc0(39, IMAGE_COPY_IMAGE),
        ldc0(20, zero),
        ldc0(21, zero),
        ldc0(5, one),
        sured(AtomOp::Add, 0, 20, 5),
        ldc0(22, one),
        ldc0(23, zero),
        sured(AtomOp::Max, 2, 22, 0),
        ldc0(24, two),
        ldc0(25, zero),
        ldc0(6, two),
        sured(AtomOp::Inc, 3, 24, 6),
        exit_word(),
    ]
}

/// Bank 0's word holding the level [`image_size_program`] queries, and the word bound
/// texture 8's handle is read from.
pub const IMAGE_SIZE_LEVEL: u32 = 0x70;
const IMAGE_SIZE_HANDLE: u32 = 8 * 4;

/// The threads [`warp_program`] runs as: a full warp and four lanes of another.
pub const WARP_THREADS: u32 = 36;

/// Thread `tid` stores what lane `tid ^ 3` held, by two butterfly shuffles, to output word
/// `tid`, and its warp's `vote.all` ballot to word [`WARP_THREADS`]` + tid`.
/// Word `2 * WARP_THREADS + tid` gets `fswzadd`'s identity swizzle of 3.0 and 1.0, whose signs
/// follow the lane's place in its quad.
pub fn warp_program() -> Vec<u64> {
    use crate::gpu::shader::isa::{self, MemSize, Operand, ShflMode, RZ};
    use crate::gpu::shader::Op;
    const PT: u64 = 7 << 16;
    let check = |word: u64, expected: Op| {
        assert_eq!(isa::decode(word).op, expected, "encoded {word:#018x}");
        word
    };
    let shfl_bfly = |dst: u8, src: u8, index: u32| {
        check(
            0xef10u64 << 48
                | 0x1f << 34
                | 3 << 30
                | 1 << 29
                | 1 << 28
                | u64::from(index) << 20
                | PT
                | u64::from(src) << 8
                | u64::from(dst),
            Op::Shfl {
                dst,
                pred: 0,
                src,
                index: Operand::Imm(index),
                mask: Operand::Imm(0x1f),
                mode: ShflMode::Bfly,
            },
        )
    };
    let iadd = |dst: u8, a: u8, offset: u16, cin: bool, cout: bool| {
        check(
            0x4c10u64 << 48
                | u64::from(offset >> 2) << 20
                | u64::from(cin) << 43
                | u64::from(cout) << 47
                | PT
                | u64::from(a) << 8
                | u64::from(dst),
            Op::Iadd {
                dst,
                a,
                aneg: false,
                b: Operand::Const { bank: 0, offset },
                bneg: false,
                cin,
                cout,
            },
        )
    };
    let stg = |offset: u32, src: u8| {
        check(
            (0xeed8u64 | 4) << 48 | u64::from(offset) << 20 | PT | 6 << 8 | u64::from(src),
            Op::Stg {
                addr: 6,
                offset: offset as i32,
                src,
                size: MemSize::B32,
            },
        )
    };
    let descriptor = COMPUTE_DESCRIPTOR as u16;
    vec![
        check(
            0xf0c8u64 << 48 | 0x21 << 20 | PT,
            Op::S2r { dst: 0, sr: 0x21 },
        ),
        shfl_bfly(1, 0, 1),
        shfl_bfly(1, 1, 2),
        check(
            0x50d8_e380_0000_0000 | PT | 2,
            Op::Vote {
                dst: 2,
                pred: isa::Pred::PT,
                src: isa::Pred::ALWAYS,
                mode: isa::VoteMode::All,
            },
        ),
        check(
            0x5c18u64 << 48 | u64::from(RZ) << 20 | 2 << 39 | PT | 5,
            Op::Iscadd {
                dst: 5,
                a: 0,
                aneg: false,
                b: Operand::Reg(RZ),
                bneg: false,
                shift: 2,
                cout: false,
            },
        ),
        check(
            0x010u64 << 52 | u64::from(3.0f32.to_bits()) << 20 | 3,
            Op::Mov32i {
                dst: 3,
                imm: 3.0f32.to_bits(),
            },
        ),
        check(
            0x010u64 << 52 | u64::from(1.0f32.to_bits()) << 20 | 4,
            Op::Mov32i {
                dst: 4,
                imm: 1.0f32.to_bits(),
            },
        ),
        check(
            0x50f8u64 << 48 | 0xe4 << 28 | 4 << 20 | PT | 3 << 8 | 3,
            Op::Fswzadd {
                dst: 3,
                a: 3,
                b: 4,
                swizzle: 0xe4,
                ftz: false,
            },
        ),
        iadd(6, 5, descriptor, false, true),
        iadd(7, RZ, descriptor + 4, true, false),
        stg(0, 1),
        stg(WARP_THREADS * 4, 2),
        stg(WARP_THREADS * 8, 3),
        check(0xe300_0000_0000_0000 | PT | 0xF, Op::Exit),
    ]
}
