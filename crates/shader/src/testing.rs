//! Captured and hand-assembled programs for tests, here and in the crates that run them.

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
