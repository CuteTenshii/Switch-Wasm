//! A drawable engine, for tests that need one.
//!
//! The software rasterizer is the reference every other path must agree with,
//! and agreeing is something only a *comparison* establishes. That needs both
//! renderers driven over the same [`Engine3D`] — which lives here rather than
//! in `raster`'s own test module, because `switch-gpu` is a separate crate and
//! cannot reach into one. The same reason `ipc::testing` exists.
//!
//! It is the smallest complete draw: a 16x8 pitch-linear RGBA8 target, two
//! real shaders decoded from captured SASS, and a vertex array of three
//! positions and three colours. Everything a test varies — the multisample
//! mode, the sample mask, the depth state — it varies by writing the register
//! the guest would have written.

use crate::gpu::engine::threed::{DrawCall, Engine3D};
use crate::gpu::exec::{ExecCtx, GpuStats};
use crate::gpu::renderer::{Flush, Renderer};
use crate::gpu::syncpt::Host1x;
use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
use crate::mem::Memory;

/// Two instruction words, as the eight bytes a shader binary holds.
fn word(low: u32, high: u32) -> [u8; 8] {
    (((high as u64) << 32) | low as u64).to_le_bytes()
}

/// One 32-byte scheduling block: the sched word and the three instructions it
/// schedules.
fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(&word(sched.0, sched.1));
    out.extend_from_slice(&word(a.0, a.1));
    out.extend_from_slice(&word(b.0, b.1));
    out.extend_from_slice(&word(c.0, c.1));
    out
}

/// `gl_Position = aPosition; vColor = aColor;` — composed from the same real,
/// oracle-verified `ld`/`st` b128 attribute-space words `mvp.vert`'s fixture
/// uses (see `isa`'s module docs), so no bit-level guessing is needed for a
/// passthrough.
pub fn passthrough_vertex_shader() -> Vec<u8> {
    // Sched words are placeholders reused from mvp.vert's real capture —
    // never all-zero, since `decode_program_from_memory` treats an all-zero
    // first word as "this binary has a Mesa header".
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

/// `oColor = vColor;` — the same real capture `isa`'s module docs and
/// `shader::interp`'s tests use.
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
        (0, 0),
        (0, 0),
    ));
    bytes
}

/// The register the multisample mode lives in, and the ones a test that
/// varies coverage reaches for. Named because a test that writes `0x574`
/// says nothing about what it is doing.
pub const MULTISAMPLE_ENABLE: u32 = 0x54D;
pub const MULTISAMPLE_CONTROL: u32 = 0x54F;
pub const MULTISAMPLE_MODE: u32 = 0x574;
pub const MULTISAMPLE_SAMPLE_MASK: u32 = 0x3EF;
pub const DEPTH_TEST_ENABLE: u32 = 0x4B3;
pub const DEPTH_WRITE_ENABLE: u32 = 0x4BA;
pub const DEPTH_TEST_FUNC: u32 = 0x4C3;

/// A memory, an address space and an engine set up to issue one draw.
pub struct Harness {
    pub mem: Memory,
    pub vmm: AddressSpace,
    pub engine: Engine3D,
    pub host1x: Host1x,
    pub stats: GpuStats,
    /// Where the mapping starts, which is also the colour target's address.
    pub base: u64,
}

/// How wide the target is, in texels.
pub const TARGET_WIDTH: u32 = 16;
/// How tall the target is, in texels.
pub const TARGET_HEIGHT: u32 = 8;

impl Harness {
    /// The 16x8 RGBA8 target, both shaders, and a three-vertex array — with
    /// nothing drawn yet.
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
                for (i, chunk) in words.chunks_exact(4).enumerate() {
                    let word = u32::from_le_bytes(chunk.try_into().unwrap());
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
        // SetProgramRegion.
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

    /// Issue the draw through `renderer`.
    ///
    /// The two halves are borrowed separately because a renderer wants the
    /// engine and the memory at once, and they are fields of the same thing.
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

    /// Clear the colour target through `renderer`.
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

    /// Ask `renderer` for whatever it is holding, until it has handed it all
    /// back. A backend that keeps its surfaces on a device answers
    /// [`Flush::Pending`] while a readback is in flight.
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

    /// Where the vertex array starts.
    pub fn vertices(&self) -> u64 {
        self.engine.vertex_array(0).start
    }

    pub fn write_vertex(&mut self, index: u32, pos: [f32; 4], color: [f32; 4]) {
        let addr = self.vertices() + index as u64 * 32;
        for (i, v) in pos.iter().enumerate() {
            self.vmm
                .write_u32(&mut self.mem, addr + i as u64 * 4, v.to_bits())
                .unwrap();
        }
        for (i, v) in color.iter().enumerate() {
            self.vmm
                .write_u32(&mut self.mem, addr + 16 + i as u64 * 4, v.to_bits())
                .unwrap();
        }
    }

    /// The three vertices of a triangle covering the upper-left half of the
    /// target, all of one colour.
    pub fn triangle(&mut self, color: [f32; 4]) {
        self.write_vertex(0, [-1.0, 1.0, 0.0, 1.0], color);
        self.write_vertex(1, [1.0, 1.0, 0.0, 1.0], color);
        self.write_vertex(2, [-1.0, -1.0, 0.0, 1.0], color);
    }

    /// Read the target as it stands, texel by texel, left to right and top to
    /// bottom. What a comparison between two renderers compares.
    pub fn target(&mut self) -> Vec<u32> {
        let addr = self.base;
        // A depth-only pass has no colour surface, and there is nothing to
        // read rather than nothing to say about it.
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

    /// Read the target as a multisampled surface: `samples_x` by `samples_y`
    /// texels per pixel, and every one of them separate.
    pub fn texel(&mut self, x: u32, y: u32) -> u32 {
        let addr = self.base;
        let width = self.engine.render_target(0).unwrap().unwrap().width;
        self.ctx()
            .read_u32(addr + u64::from(y * width + x) * 4)
            .unwrap()
    }

    /// Move the colour attribute onto vertex array 1, stepped once per
    /// instance, and put `colours[instance]` in it.
    ///
    /// An instanced array is what the upload path reads a single element of —
    /// the one this instance reaches — so a backend has to bind that element
    /// as though every instance read it.
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

    /// Bind a `Z24S8` depth surface beside the colour one, the same extent,
    /// and turn the depth test on.
    ///
    /// It goes after the colour target in the same mapping, which is why the
    /// harness maps more than it needs for one surface.
    pub fn depth_target(&mut self, func: u32) {
        let addr = self.base + 0x1000;
        self.engine.regs.set(0x3F8, (addr >> 32) as u32);
        self.engine.regs.set(0x3F9, addr as u32);
        self.engine.regs.set(0x3FA, 0x14); // Z24S8
        self.engine.regs.set(0x3FB, 0); // one GOB per block
        self.engine.regs.set(0x48A, TARGET_WIDTH);
        self.engine.regs.set(0x48B, TARGET_HEIGHT);
        self.engine.regs.set(DEPTH_TEST_ENABLE, 1);
        self.engine.regs.set(DEPTH_WRITE_ENABLE, 1);
        self.engine.regs.set(DEPTH_TEST_FUNC, func);
    }

    /// Read the depth surface as it stands, texel by texel.
    pub fn depth(&mut self) -> Vec<u32> {
        let addr = self.base + 0x1000;
        let target = self.engine.depth_target().unwrap().unwrap();
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

    /// Turn the target into a `samples_x` by `samples_y` multisampled one,
    /// which on Maxwell means the *pixel* extent shrinks and the surface
    /// stays the size it was.
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
