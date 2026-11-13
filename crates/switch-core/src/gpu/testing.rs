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

/// `mov32i $reg, imm`.
fn mov32i(reg: u8, imm: u32) -> (u32, u32) {
    (
        (imm & 0xfff) << 20 | 0x0007_f000 | u32::from(reg),
        0x0100_0000 | imm >> 12,
    )
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
