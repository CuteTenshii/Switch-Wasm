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

/// Distinct RGBA8 texel at `(x, y)` of [`Harness::bindless_texture`].
pub fn bindless_texel(x: u32, y: u32) -> u32 {
    let (r, g, b) = (x * 30 + 10, y * 30 + 10, (y * 8 + x) * 3);
    r | g << 8 | b << 16 | 0xff << 24
}

pub const MULTISAMPLE_ENABLE: u32 = 0x54D;
pub const MULTISAMPLE_CONTROL: u32 = 0x54F;
pub const MULTISAMPLE_MODE: u32 = 0x574;
pub const MULTISAMPLE_SAMPLE_MASK: u32 = 0x3EF;
pub const DEPTH_TEST_ENABLE: u32 = 0x4B3;
pub const DEPTH_WRITE_ENABLE: u32 = 0x4BA;
pub const DEPTH_TEST_FUNC: u32 = 0x4C3;

/// Write one vertex: `vec4` position at offset 0, `vec4` colour at 16, stride 32.
pub fn write_vertex(
    mem: &mut Memory,
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
        }
    }

    pub fn draw_with(&mut self, renderer: &mut dyn Renderer) -> crate::Result<()> {
        let engine = &self.engine;
        let mut ctx = ExecCtx {
            mem: &mut self.mem,
            vmm: &self.vmm,
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
            mem: &mut self.mem,
            vmm: &self.vmm,
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
            mem: &mut self.mem,
            vmm: &self.vmm,
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
                mem: &mut self.mem,
                vmm: &self.vmm,
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
            mem: &mut self.mem,
            vmm: &self.vmm,
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
        write_vertex(&mut self.mem, &self.vmm, base, index, pos, color);
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
                self.vmm.write_u32(&mut self.mem, at, v.to_bits()).unwrap();
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
        let header_pool = self.base + 0x1400;
        let sampler_pool = self.base + 0x1480;
        let constants = self.base + 0x1500;
        let image = self.base + 0x1800;
        let size = BINDLESS_TEXTURE_SIZE;
        let mut ctx = self.ctx();
        for y in 0..size {
            for x in 0..size {
                ctx.write_u32(image + u64::from((y * size + x) * 4), bindless_texel(x, y))
                    .unwrap();
            }
        }
        // TIC 1: A8B8G8R8 UNORM, the identity swizzle, pitch-linear, 2D.
        let tic = header_pool + 32;
        let identity = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        ctx.write_u32(tic, 0x08 | (2 << 7) | identity).unwrap();
        ctx.write_u32(tic + 4, image as u32).unwrap();
        ctx.write_u32(tic + 8, (image >> 32) as u32 | (2 << 21))
            .unwrap();
        ctx.write_u32(tic + 12, size * 4 / 32).unwrap();
        ctx.write_u32(tic + 16, (size - 1) | (1 << 23)).unwrap();
        ctx.write_u32(tic + 20, size - 1).unwrap();
        // TSC 1: ClampToEdge on all three axes, nearest filtering.
        ctx.write_u32(sampler_pool + 32, 2 | (2 << 3) | (2 << 6))
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
            mem: &mut self.mem,
            vmm: &self.vmm,
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
