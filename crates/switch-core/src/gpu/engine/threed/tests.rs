//! 3D engine tests.

use super::*;
use crate::gpu::exec::ExecCtx;

#[test]
fn a_draw_is_confined_to_a_depth_surface_only_where_it_reaches_it() {
    let state = |test_enabled, write_enabled, func| DepthState {
        test_enabled,
        write_enabled,
        func,
    };
    let (color, depth) = (Some((1280, 720)), Some((128, 128)));
    let less = state(true, false, 0x0201);
    assert_eq!(draw_extent(color, depth, less), Some((128, 128)));
    let always_writing = state(true, true, 8);
    assert_eq!(draw_extent(color, depth, always_writing), Some((128, 128)));
    // A disabled test touches nothing, nor does a non-writing `Always`.
    for untouched in [state(false, true, 0x0201), state(true, false, 0x0207)] {
        assert!(!untouched.reaches_surface());
        assert_eq!(draw_extent(color, depth, untouched), Some((1280, 720)));
    }
    assert_eq!(draw_extent(None, depth, less), Some((128, 128)));
    assert_eq!(draw_extent(None, None, less), None);
}
use crate::gpu::exec::GpuStats;
use crate::gpu::syncpt::Host1x;
use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
use crate::mem::Memory;

struct Harness {
    mem: Memory,
    vmm: AddressSpace,
    host1x: Host1x,
    stats: GpuStats,
    base: u64,
}

impl Harness {
    fn new(size: u32) -> Harness {
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, size as usize).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm
            .map(0x3000_0000, size as u64, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        Harness {
            mem,
            vmm,
            host1x: Host1x::new(),
            stats: GpuStats::default(),
            base,
        }
    }

    fn ctx(&mut self) -> ExecCtx<'_> {
        ExecCtx {
            mem: &mut self.mem,
            vmm: &self.vmm,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: false,
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Recorded {
    draws: u32,
    color_clears: Vec<(u32, u32, [bool; 4])>,
    depth_clears: Vec<(bool, bool)>,
}

/// A renderer that records what it was asked to do and writes nothing.
#[derive(Debug, Default)]
struct Recorder(std::rc::Rc<std::cell::RefCell<Recorded>>);

impl Renderer for Recorder {
    fn draw(&mut self, _: &Engine3D, _: &mut ExecCtx) -> Result<()> {
        self.0.borrow_mut().draws += 1;
        Ok(())
    }

    fn clear_color(
        &mut self,
        _: &Engine3D,
        _: &mut ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> Result<()> {
        self.0
            .borrow_mut()
            .color_clears
            .push((target, layer, channels));
        Ok(())
    }

    fn clear_depth_stencil(
        &mut self,
        _: &Engine3D,
        _: &mut ExecCtx,
        depth: bool,
        stencil: bool,
    ) -> Result<()> {
        self.0.borrow_mut().depth_clears.push((depth, stencil));
        Ok(())
    }
}

/// Program a 16x8 pitch-linear RGBA8 render target.
fn setup_pitch_target(engine: &mut Engine3D, base: u64, width: u32, height: u32) {
    engine.regs.set(0x200, (base >> 32) as u32);
    engine.regs.set(0x201, base as u32);
    engine.regs.set(0x202, width * 4); // pitch in bytes
    engine.regs.set(0x203, height);
    engine.regs.set(0x204, 0xD5); // RGBA8Unorm
    engine.regs.set(0x205, 1 << 12); // IsLinear
    engine.regs.set(0x206, 1);
    engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, width << 16);
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, height << 16);
}

/// Program a single-pixel block-linear depth target in `format`.
fn setup_depth_target(engine: &mut Engine3D, base: u64, format: u32) {
    engine.regs.set(DEPTH_TARGET_ADDR, (base >> 32) as u32);
    engine.regs.set(DEPTH_TARGET_ADDR + 1, base as u32);
    engine.regs.set(DEPTH_TARGET_FORMAT, format);
    engine.regs.set(DEPTH_TARGET_TILE_MODE, 0);
    engine.regs.set(DEPTH_TARGET_HORIZONTAL, 1);
    engine.regs.set(DEPTH_TARGET_VERTICAL, 1);
    engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 1 << 16);
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, 1 << 16);
}

#[test]
fn every_pixel_the_engine_produces_goes_through_the_bound_renderer() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let log = std::rc::Rc::new(std::cell::RefCell::new(Recorded::default()));
    engine.set_renderer(Box::new(Recorder(log.clone())));
    setup_pitch_target(&mut engine, h.base, 16, 8);
    {
        let mut ctx = h.ctx();
        // Every way this engine makes pixels.
        engine
            .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
            .unwrap();
        engine.write(CLEAR_BUFFERS, 0b11, true, &mut ctx).unwrap();
        engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap();
        engine.write(DRAW_ARRAYS_COUNT, 3, true, &mut ctx).unwrap();
    }

    let log = log.borrow();
    assert_eq!(log.draws, 1, "the draw reached the renderer");
    assert_eq!(
        log.color_clears,
        vec![(0, 0, [true; 4])],
        "and the colour clear"
    );
    assert_eq!(log.depth_clears, vec![(true, true)], "and the depth clear");

    // The recorder wrote nothing, so the target is untouched.
    let target = h.mem.dump(0x3000_0000, 16 * 8 * 4).unwrap();
    assert!(
        target.iter().all(|&b| b == 0),
        "no pixel was written past the renderer"
    );
}

#[test]
fn clear_fills_a_pitch_render_target() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    setup_pitch_target(&mut engine, h.base, 16, 8);
    engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits());
    engine.regs.set(CLEAR_COLOR + 1, 0.0f32.to_bits());
    engine.regs.set(CLEAR_COLOR + 2, 0.0f32.to_bits());
    engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());

    let mut ctx = h.ctx();
    // Clear all four colour channels of target 0.
    engine
        .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
        .unwrap();

    assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0xFF00_00FF);
    assert_eq!(h.mem.read_u32(0x3000_0000 + 15 * 4).unwrap(), 0xFF00_00FF);
    assert_eq!(h.mem.read_u32(0x3000_0000 + 7 * 64).unwrap(), 0xFF00_00FF);
    // One past the last row must be untouched.
    assert_eq!(h.mem.read_u32(0x3000_0000 + 8 * 64).unwrap(), 0);
    assert_eq!(h.stats.clears, 1);
}

#[test]
fn clear_of_a_multisampled_target_fills_every_sample() {
    // 16x8 texel target, 8x4 pixel scissor, 2x2 multisampling.
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    setup_pitch_target(&mut engine, h.base, 16, 8);
    engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 8 << 16);
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, 4 << 16);
    engine.regs.set(MULTISAMPLE_ENABLE, 1);
    engine.regs.set(MULTISAMPLE_MODE, 2); // 2x2
    for i in 0..4 {
        engine
            .regs
            .set(MULTISAMPLE_SAMPLE_LOCATIONS + i, 0xEAA2_6E26);
    }
    engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits());
    engine.regs.set(CLEAR_COLOR + 1, 1.0f32.to_bits());
    engine.regs.set(CLEAR_COLOR + 2, 1.0f32.to_bits());
    engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());

    let mut ctx = h.ctx();
    engine
        .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
        .unwrap();

    // Every one of the 16x8 texels, not just the 8x4 the scissor names.
    for y in 0..8u32 {
        for x in 0..16u32 {
            assert_eq!(
                h.mem.read_u32(0x3000_0000 + y * 64 + x * 4).unwrap(),
                0xFFFF_FFFF,
                "texel ({x}, {y})"
            );
        }
    }
    // One past the last row is still outside the target.
    assert_eq!(h.mem.read_u32(0x3000_0000 + 8 * 64).unwrap(), 0);
}

/// `MsaaMode` alone sets texels per pixel; `AntiAliasEnable` only where coverage is tested.
#[test]
fn the_msaa_mode_sizes_the_surface_and_the_enable_bit_only_moves_coverage() {
    let mut engine = Engine3D::new();
    engine.regs.set(MULTISAMPLE_MODE, 2); // 2x2
    let off = engine.sample_grid().unwrap();
    assert_eq!(off.count(), 4, "the surface is multisampled either way");
    assert_eq!((off.samples_x, off.samples_y), (2, 2));
    // With antialiasing off every sample tests the pixel's centre.
    assert!((0..off.count()).all(|s| off.position(s) == [0.5, 0.5]));
    // Its samples still land in texels of their own.
    assert_eq!(off.texel(3, 5, 0), (6, 10));
    assert_ne!(off.texel(3, 5, 3), off.texel(3, 5, 0));

    engine.regs.set(MULTISAMPLE_ENABLE, 1);
    let on = engine.sample_grid().unwrap();
    assert_eq!(on.count(), 4);
    assert!(
        (0..on.count()).any(|s| on.position(s) != [0.5, 0.5]),
        "coverage is per sample"
    );
}

/// A guest that never touches either register is not multisampled.
#[test]
fn an_unprogrammed_multisample_mode_is_one_sample_a_pixel() {
    assert!(Engine3D::new().sample_grid().unwrap().is_single());
}

#[test]
fn clear_respects_the_scissor() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    setup_pitch_target(&mut engine, h.base, 16, 8);
    engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());
    engine.regs.set(CLEAR_BUFFER_FLAGS, 1 << 8);
    engine.regs.set(SCISSOR_BASE, 1); // enable
    engine.regs.set(SCISSOR_BASE + 1, 4 | (8 << 16)); // x in [4, 8)
    engine.regs.set(SCISSOR_BASE + 2, 0 | (8 << 16));

    let mut ctx = h.ctx();
    engine
        .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
        .unwrap();

    assert_eq!(h.mem.read_u32(0x3000_0000 + 3 * 4).unwrap(), 0);
    assert_eq!(h.mem.read_u32(0x3000_0000 + 4 * 4).unwrap(), 0xFF00_0000);
    assert_eq!(h.mem.read_u32(0x3000_0000 + 7 * 4).unwrap(), 0xFF00_0000);
    assert_eq!(h.mem.read_u32(0x3000_0000 + 8 * 4).unwrap(), 0);
}

#[test]
fn clear_with_a_channel_mask_preserves_the_others() {
    let mut h = Harness::new(0x1000);
    h.mem.write_u32(0x3000_0000, 0x1122_3344).unwrap();
    let mut engine = Engine3D::new();
    setup_pitch_target(&mut engine, h.base, 1, 1);
    engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits()); // red = 1.0

    let mut ctx = h.ctx();
    // Only the red channel.
    engine.write(CLEAR_BUFFERS, 0b100, true, &mut ctx).unwrap();

    assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1122_33FF);
}

#[test]
fn a_colour_target_holding_a_depth_format_is_not_a_colour_target() {
    // One Z24S8 surface bound as both Z target and colour target 0; the clear must do nothing.
    let mut h = Harness::new(0x1000);
    h.mem.write_u32(0x3000_0000, 0x1122_3344).unwrap();
    let mut engine = Engine3D::new();
    setup_pitch_target(&mut engine, h.base, 1, 1);
    engine.regs.set(0x204, 0x14); // Z24S8, where a colour format goes
    engine.regs.set(CLEAR_COLOR, 1.0f32.to_bits());
    {
        let mut ctx = h.ctx();
        engine
            .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
            .unwrap();
    }

    assert_eq!(engine.render_target(0).unwrap(), None);
    assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1122_3344);
}

#[test]
fn a_colour_target_in_a_format_that_is_neither_is_still_an_error() {
    // A value that is neither a colour nor a depth format is still an error.
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    setup_pitch_target(&mut engine, h.base, 1, 1);
    engine.regs.set(0x204, 0x77);
    let mut ctx = h.ctx();
    assert!(engine
        .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
        .is_err());
}

#[test]
fn z24s8_keeps_its_stencil_low_and_s8z24_keeps_it_high() {
    // NVIDIA names fields most significant first.
    for (format, expected) in [(0x14u32, 0xFFFF_FFAB_u32), (0x16, 0xABFF_FFFF)] {
        let mut h = Harness::new(0x1000);
        let mut engine = Engine3D::new();
        setup_depth_target(&mut engine, h.base, format);
        engine.regs.set(CLEAR_DEPTH, 1.0f32.to_bits());
        engine.regs.set(CLEAR_STENCIL, 0xAB);
        {
            let mut ctx = h.ctx();
            engine.write(CLEAR_BUFFERS, 0b11, true, &mut ctx).unwrap();
        }

        assert_eq!(
            h.mem.read_u32(0x3000_0000).unwrap(),
            expected,
            "format {format:#x}"
        );
    }
}

#[test]
fn writing_depth_leaves_a_packed_stencil_byte_alone() {
    let z24s8 = depth_format_layout(0x14).unwrap();
    assert!(z24s8.packs_stencil());
    assert_eq!(z24s8.with_depth(0x0000_00AB, 1.0), 0xFFFF_FFAB);
    assert!((z24s8.decode_depth(z24s8.with_depth(0x0000_00AB, 0.5)) - 0.5).abs() < 1e-6);

    // Z32Float owns its whole pixel, so a depth write need not read first.
    let zf32 = depth_format_layout(0x0A).unwrap();
    assert!(!zf32.packs_stencil());
    assert_eq!(zf32.decode_depth(zf32.encode_depth(0.25)), 0.25);
}

/// The run-based depth clear must match a per-texel one and keep `Z24S8`'s stencil byte.
#[test]
fn a_depth_clear_writes_runs_where_a_per_texel_clear_would() {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 16;
    const BYTES: u32 = 4;
    /// A distinct starting stencil byte per texel.
    fn seeded(tx: u32, ty: u32) -> u32 {
        0xDEAD_0000 | (tx << 8) | ((tx * 7 + ty) & 0xFF)
    }

    // Whole GOBs and a partial rectangle, covering both the run path and its edges.
    for &(rx, rw, ry, rh) in &[(0u32, WIDTH, 0u32, HEIGHT), (4, 20, 3, 9)] {
        let mut h = Harness::new(0x10000);
        let layout = Layout::BlockLinear {
            block_height_gobs: 1,
        };
        let width_bytes = WIDTH * BYTES;
        for ty in 0..HEIGHT {
            for tx in 0..WIDTH {
                let at = 0x3000_0000 + layout.offset(tx * BYTES, ty, width_bytes);
                h.mem.write_u32(at, seeded(tx, ty)).unwrap();
            }
        }

        let mut engine = Engine3D::new();
        setup_depth_target(&mut engine, h.base, 0x14);
        engine.regs.set(DEPTH_TARGET_HORIZONTAL, WIDTH);
        engine.regs.set(DEPTH_TARGET_VERTICAL, HEIGHT);
        engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, rx | (rw << 16));
        engine.regs.set(SCREEN_SCISSOR_VERTICAL, ry | (rh << 16));
        engine.regs.set(CLEAR_DEPTH, 0.5f32.to_bits());
        {
            let mut ctx = h.ctx();
            engine.write(CLEAR_BUFFERS, 0b01, true, &mut ctx).unwrap();
        }

        let format = depth_format_layout(0x14).unwrap();
        for ty in 0..HEIGHT {
            for tx in 0..WIDTH {
                let at = 0x3000_0000 + layout.offset(tx * BYTES, ty, width_bytes);
                let old = seeded(tx, ty);
                let inside = (rx..rx + rw).contains(&tx) && (ry..ry + rh).contains(&ty);
                let want = if inside {
                    format.with_depth(u128::from(old), 0.5) as u32
                } else {
                    old
                };
                assert_eq!(
                    h.mem.read_u32(at).unwrap(),
                    want,
                    "texel ({tx},{ty}) of rect ({rx},{ry}) {rw}x{rh}"
                );
                if inside {
                    assert_eq!(want & 0xFF, old & 0xFF, "the stencil byte survived");
                }
            }
        }
    }
}

#[test]
fn clear_of_a_block_linear_target_uses_the_swizzle() {
    let mut h = Harness::new(0x10000);
    let mut engine = Engine3D::new();
    engine.regs.set(0x200, (h.base >> 32) as u32);
    engine.regs.set(0x201, h.base as u32);
    engine.regs.set(0x202, 16); // 16 pixels wide
    engine.regs.set(0x203, 8);
    engine.regs.set(0x204, 0xD5);
    engine.regs.set(0x205, 0); // block-linear, one GOB per block
    engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 16 << 16);
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, 8 << 16);
    engine.regs.set(CLEAR_COLOR + 3, 1.0f32.to_bits());

    let mut ctx = h.ctx();
    engine
        .write(CLEAR_BUFFERS, 0b11_1100, true, &mut ctx)
        .unwrap();

    // A 16x8 RGBA8 surface is exactly one GOB; every byte of it is written.
    for i in 0..512u32 / 4 {
        assert_eq!(
            h.mem.read_u32(0x3000_0000 + i * 4).unwrap(),
            0xFF00_0000,
            "word {}",
            i
        );
    }
}

#[test]
fn report_semaphore_release_writes_the_payload() {
    let mut h = Harness::new(0x1000);
    let base = h.base;
    let mut engine = Engine3D::new();
    engine
        .regs
        .set(REPORT_SEMAPHORE_OFFSET, (base >> 32) as u32);
    engine.regs.set(REPORT_SEMAPHORE_OFFSET + 1, base as u32);
    engine.regs.set(REPORT_SEMAPHORE_PAYLOAD, 0x1234_5678);

    let mut ctx = h.ctx();
    // Release, one-word structure.
    engine
        .write(REPORT_SEMAPHORE, 1 << 28, true, &mut ctx)
        .unwrap();
    assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0x1234_5678);
}

#[test]
fn syncpt_action_increments_the_counter() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine
        .write(SYNCPT_ACTION, 9 | (1 << 20), true, &mut ctx)
        .unwrap();
    assert_eq!(h.host1x.read(9).unwrap(), 1);
}

#[test]
fn constbuf_upload_walks_the_cursor() {
    let mut h = Harness::new(0x1000);
    let base = h.base;
    let mut engine = Engine3D::new();
    engine.regs.set(CONSTBUF_SELECTOR_SIZE, 0x100);
    engine.regs.set(CONSTBUF_SELECTOR_ADDR, (base >> 32) as u32);
    engine.regs.set(CONSTBUF_SELECTOR_ADDR + 1, base as u32);

    let mut ctx = h.ctx();
    engine
        .write(LOAD_CONSTBUF_OFFSET, 0, true, &mut ctx)
        .unwrap();
    engine
        .write(LOAD_CONSTBUF_DATA, 0xAAAA_AAAA, false, &mut ctx)
        .unwrap();
    engine
        .write(LOAD_CONSTBUF_DATA, 0xBBBB_BBBB, true, &mut ctx)
        .unwrap();

    assert_eq!(h.mem.read_u32(0x3000_0000).unwrap(), 0xAAAA_AAAA);
    assert_eq!(h.mem.read_u32(0x3000_0004).unwrap(), 0xBBBB_BBBB);
}

#[test]
fn constbuf_upload_past_the_end_is_rejected() {
    let mut h = Harness::new(0x1000);
    let base = h.base;
    let mut engine = Engine3D::new();
    engine.regs.set(CONSTBUF_SELECTOR_SIZE, 4);
    engine.regs.set(CONSTBUF_SELECTOR_ADDR, (base >> 32) as u32);
    engine.regs.set(CONSTBUF_SELECTOR_ADDR + 1, base as u32);

    let mut ctx = h.ctx();
    engine
        .write(LOAD_CONSTBUF_DATA, 1, false, &mut ctx)
        .unwrap();
    assert!(engine.write(LOAD_CONSTBUF_DATA, 2, true, &mut ctx).is_err());
}

#[test]
fn draw_arrays_records_the_call() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap(); // Triangles
    engine.write(0x35D, 6, true, &mut ctx).unwrap(); // first
    engine.write(DRAW_ARRAYS_COUNT, 3, true, &mut ctx).unwrap();
    assert_eq!(
        engine.last_draw,
        DrawCall {
            primitive: 4,
            first: 6,
            count: 3,
            indexed: false,
            index_format: 0
        }
    );
    assert_eq!(h.stats.draws, 1);

    // No target is bound, and the report says so.
    let activity = engine.activity.take();
    assert_eq!(activity.len(), 1, "{activity:?}");
    let (kind, tally) = &activity[0];
    assert_eq!(*kind, crate::gpu::activity::Kind::Draw);
    assert_eq!(tally.label, "no colour target bound");
    assert_eq!((tally.count, tally.amount), (1, 3));
}

#[test]
fn instance_next_steps_the_instance_counter_and_a_plain_begin_resets_it() {
    // Each Begin/End pair after the first carries `InstanceNext`.
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();

    engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap();
    assert_eq!(engine.instance_id(), 0);
    for expected in 1..=3 {
        engine
            .write(
                VERTEX_BEGIN_GL,
                4 | VERTEX_BEGIN_INSTANCE_NEXT,
                true,
                &mut ctx,
            )
            .unwrap();
        assert_eq!(engine.instance_id(), expected);
    }

    // The next draw's first instance starts over.
    engine.write(VERTEX_BEGIN_GL, 4, true, &mut ctx).unwrap();
    assert_eq!(engine.instance_id(), 0);
}

#[test]
fn instance_next_does_not_disturb_the_primitive() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine
        .write(
            VERTEX_BEGIN_GL,
            4 | VERTEX_BEGIN_INSTANCE_NEXT,
            true,
            &mut ctx,
        )
        .unwrap();
    engine.write(0x5F7, 0, true, &mut ctx).unwrap();
    engine
        .write(DRAW_ELEMENTS_COUNT, 6, true, &mut ctx)
        .unwrap();
    assert_eq!(
        engine.last_draw.primitive, 4,
        "Triangles, not the raw argument"
    );
}

#[test]
fn the_vertex_b_stage_is_bound_without_its_enable_bit() {
    // VertexB is active without its `Config.Enable` bit.
    let mut h = Harness::new(0x1000);
    let base_addr = h.base;
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine
        .write(SET_PROGRAM_REGION, (base_addr >> 32) as u32, true, &mut ctx)
        .unwrap();
    engine
        .write(SET_PROGRAM_REGION + 1, base_addr as u32, true, &mut ctx)
        .unwrap();

    // Offset and register count, and no Config write at all.
    let base = SET_PROGRAM + ShaderStage::VertexB.index() * SET_PROGRAM_STRIDE;
    engine.write(base + 1, 0x200, true, &mut ctx).unwrap();
    engine.write(base + 3, 0xd, true, &mut ctx).unwrap();
    assert_eq!(
        engine.program(ShaderStage::VertexB),
        Some(ProgramBinding {
            addr: base_addr + 0x200,
            num_registers: 0xd
        }),
    );

    // Every other stage still needs the bit.
    let a = SET_PROGRAM + ShaderStage::VertexA.index() * SET_PROGRAM_STRIDE;
    engine.write(a, 0, true, &mut ctx).unwrap();
    engine.write(a + 1, 0x300, true, &mut ctx).unwrap();
    assert_eq!(engine.program(ShaderStage::VertexA), None);
    assert_eq!(engine.program(ShaderStage::Geometry), None);
}

#[test]
fn program_binding_resolves_relative_to_its_region() {
    let mut h = Harness::new(0x1000);
    let base_addr = h.base;
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine
        .write(SET_PROGRAM_REGION, (base_addr >> 32) as u32, true, &mut ctx)
        .unwrap();
    engine
        .write(SET_PROGRAM_REGION + 1, base_addr as u32, true, &mut ctx)
        .unwrap();
    // Fragment (StageId 5) enabled, at +0x100, using 4 registers.
    let base = SET_PROGRAM + 5 * SET_PROGRAM_STRIDE;
    engine.write(base, 1 | (5 << 4), true, &mut ctx).unwrap();
    engine.write(base + 1, 0x100, true, &mut ctx).unwrap();
    engine.write(base + 3, 4, true, &mut ctx).unwrap();

    assert_eq!(
        engine.program(ShaderStage::Fragment),
        Some(ProgramBinding {
            addr: base_addr + 0x100,
            num_registers: 4
        })
    );
    assert_eq!(engine.program(ShaderStage::Geometry), None);
}

#[test]
fn vertex_attrib_state_decodes_its_bit_fields() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    // BufferId=2, IsFixed=0, Offset=0x10, Size=0x1F, Type=3, IsBgra=1.
    let raw = 2 | (0x10 << 7) | (0x1F << 21) | (3 << 27) | (1 << 31);
    engine
        .write(VERTEX_ATTRIB_STATE + 3, raw, true, &mut ctx)
        .unwrap();

    let attrib = engine.vertex_attrib(3);
    assert_eq!(attrib.buffer_id, 2);
    assert!(!attrib.is_fixed);
    assert_eq!(attrib.offset, 0x10);
    assert_eq!(attrib.size, 0x1F);
    assert_eq!(attrib.ty, 3);
    assert!(attrib.is_bgra);
}

#[test]
fn vertex_array_resolves_start_and_limit() {
    let mut h = Harness::new(0x1000);
    let base_addr = h.base;
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    let base = VERTEX_ARRAY + 2 * VERTEX_ARRAY_STRIDE;
    engine
        .write(base, 0x20 | (1 << 12), true, &mut ctx)
        .unwrap(); // stride 0x20, enabled
    engine
        .write(base + 1, (base_addr >> 32) as u32, true, &mut ctx)
        .unwrap();
    engine
        .write(base + 2, base_addr as u32, true, &mut ctx)
        .unwrap();
    engine.write(base + 3, 5, true, &mut ctx).unwrap(); // divisor
    engine
        .write(
            VERTEX_ARRAY_LIMIT + 2 * 2,
            (base_addr >> 32) as u32,
            true,
            &mut ctx,
        )
        .unwrap();
    engine
        .write(
            VERTEX_ARRAY_LIMIT + 2 * 2 + 1,
            base_addr as u32 + 0x1000,
            true,
            &mut ctx,
        )
        .unwrap();

    let va = engine.vertex_array(2);
    assert!(va.enabled);
    assert_eq!(va.stride, 0x20);
    assert_eq!(va.start, base_addr);
    assert_eq!(va.limit, base_addr + 0x1000);
    // A frequency on an array that is not per-instance divides nothing.
    assert_eq!(va.divisor, 0);
    engine
        .write(VERTEX_ARRAY_PER_INSTANCE + 2, 1, true, &mut ctx)
        .unwrap();
    assert_eq!(engine.vertex_array(2).divisor, 5);
}

#[test]
fn binding_a_constbuf_snapshots_the_current_selector() {
    let mut h = Harness::new(0x1000);
    let base_addr = h.base;
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine
        .write(CONSTBUF_SELECTOR_SIZE, 0x40, true, &mut ctx)
        .unwrap();
    engine
        .write(
            CONSTBUF_SELECTOR_ADDR,
            (base_addr >> 32) as u32,
            true,
            &mut ctx,
        )
        .unwrap();
    engine
        .write(CONSTBUF_SELECTOR_ADDR + 1, base_addr as u32, true, &mut ctx)
        .unwrap();

    // Fragment's bind slot (4), bank 2, valid.
    let base = BIND + 4 * BIND_STRIDE;
    engine
        .write(base + BIND_CONSTBUF_OFFSET, 1 | (2 << 4), true, &mut ctx)
        .unwrap();

    assert_eq!(
        engine.bound_constbuf(ShaderStage::Fragment, 2),
        Some((base_addr, 0x40))
    );
    assert_eq!(engine.bound_constbuf(ShaderStage::Fragment, 3), None);
    // Vertex shares Fragment's data source but not its bank slot.
    assert_eq!(engine.bound_constbuf(ShaderStage::VertexB, 2), None);

    // A later selector change must not retroactively affect an already
    // bound bank: binding really does snapshot, not alias.
    engine
        .write(
            CONSTBUF_SELECTOR_ADDR + 1,
            base_addr as u32 + 0x40,
            true,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(
        engine.bound_constbuf(ShaderStage::Fragment, 2),
        Some((base_addr, 0x40))
    );

    // Unbinding forgets it.
    engine
        .write(base + BIND_CONSTBUF_OFFSET, 0 | (2 << 4), true, &mut ctx)
        .unwrap();
    assert_eq!(engine.bound_constbuf(ShaderStage::Fragment, 2), None);
}

#[test]
fn blend_target_uses_the_shared_registers_when_independent_blend_is_off() {
    // `IndependentBlendEnable` off reads the shared block, not `IndependentBlend[0]`.
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine.write(COLOR_BLEND_ENABLE, 1, true, &mut ctx).unwrap();
    engine.write(BLEND_EQUATION_RGB, 1, true, &mut ctx).unwrap(); // Add
    engine.write(BLEND_FUNC_SRC_RGB, 5, true, &mut ctx).unwrap(); // SrcAlpha
    engine.write(BLEND_FUNC_DST_RGB, 6, true, &mut ctx).unwrap(); // InvSrcAlpha
    engine
        .write(BLEND_EQUATION_ALPHA, 1, true, &mut ctx)
        .unwrap();
    engine
        .write(BLEND_FUNC_SRC_ALPHA, 2, true, &mut ctx)
        .unwrap(); // One
    engine
        .write(BLEND_FUNC_DST_ALPHA, 1, true, &mut ctx)
        .unwrap(); // Zero

    assert!(!engine.independent_blend_enabled());
    let bt = engine.blend_target(0);
    assert!(bt.enabled);
    assert_eq!(
        (
            bt.equation_rgb,
            bt.func_rgb_src,
            bt.func_rgb_dst,
            bt.equation_alpha,
            bt.func_alpha_src,
            bt.func_alpha_dst
        ),
        (1, 5, 6, 1, 2, 1)
    );
}

#[test]
fn a_lower_left_window_origin_flips_the_viewport() {
    // nnSdk uses a positive y scale and asks for the flip via the window origin.
    let mut engine = Engine3D::new();
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, 720 << 16);
    engine.regs.set(VIEWPORT_TRANSFORM_BASE, 640.0f32.to_bits());
    engine
        .regs
        .set(VIEWPORT_TRANSFORM_BASE + 1, 360.0f32.to_bits());
    engine
        .regs
        .set(VIEWPORT_TRANSFORM_BASE + 3, 640.0f32.to_bits());
    engine
        .regs
        .set(VIEWPORT_TRANSFORM_BASE + 4, 360.0f32.to_bits());

    let upper = engine.viewport_transform();
    assert_eq!(upper.scale[1], 360.0, "untouched, so no flip");

    engine.regs.set(WINDOW_ORIGIN, 1);
    let lower = engine.viewport_transform();
    assert_eq!(lower.scale[1], -360.0);
    assert_eq!(lower.translate[1], 360.0);
    // NDC +1 is the top of the screen once flipped, which is row 0.
    assert_eq!(lower.scale[1] + lower.translate[1], 0.0);
}

#[test]
fn a_flipped_viewport_keeps_its_distance_from_the_far_edge() {
    // The bottom 240 rows of a 720-row surface become the top 240.
    let mut engine = Engine3D::new();
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, 720 << 16);
    engine.regs.set(WINDOW_ORIGIN, 1);
    engine.regs.set(VIEWPORT_TRANSFORM_BASE, 640.0f32.to_bits());
    engine
        .regs
        .set(VIEWPORT_TRANSFORM_BASE + 1, 120.0f32.to_bits());
    engine
        .regs
        .set(VIEWPORT_TRANSFORM_BASE + 3, 640.0f32.to_bits());
    engine
        .regs
        .set(VIEWPORT_TRANSFORM_BASE + 4, 120.0f32.to_bits());

    let vt = engine.viewport_transform();
    // NDC -1 lands on 720 and +1 on 480.
    assert_eq!(vt.translate[1] - vt.scale[1], 720.0);
    assert_eq!(vt.translate[1] + vt.scale[1], 480.0);
}

#[test]
fn a_lower_left_window_origin_flips_the_scissor() {
    let mut engine = Engine3D::new();
    engine.regs.set(SCREEN_SCISSOR_VERTICAL, 720 << 16);
    engine.regs.set(SCREEN_SCISSOR_HORIZONTAL, 1280 << 16);
    engine.regs.set(SCISSOR_BASE, 1);
    engine.regs.set(SCISSOR_BASE + 1, 1280 << 16);
    engine.regs.set(SCISSOR_BASE + 2, 100 | (200 << 16));
    let full = ScissorRect {
        x0: 0,
        y0: 0,
        x1: 1280,
        y1: 720,
    };

    assert_eq!(engine.apply_scissor(full).y0, 100);
    assert_eq!(engine.apply_scissor(full).y1, 200);

    engine.regs.set(WINDOW_ORIGIN, 1);
    // Measured from the bottom, rows 100..200 are rows 520..620 down.
    assert_eq!(engine.apply_scissor(full).y0, 520);
    assert_eq!(engine.apply_scissor(full).y1, 620);
}

#[test]
fn flip_y_swaps_which_winding_is_front() {
    // Bit 4 only changes the front face.
    let mut engine = Engine3D::new();
    engine.regs.set(OGL_SET_FRONT_FACE, 0x901); // counter-clockwise
    assert!(engine.cull_state().front_ccw);
    assert_eq!(engine.viewport_transform().scale[1], 0.0);

    engine.regs.set(WINDOW_ORIGIN, 1 << 4);
    assert!(!engine.cull_state().front_ccw);
    assert_eq!(
        engine.viewport_transform().scale[1],
        0.0,
        "the viewport is untouched"
    );
}

#[test]
fn blend_and_depth_state_resolve_from_registers() {
    let mut h = Harness::new(0x1000);
    let mut engine = Engine3D::new();
    let mut ctx = h.ctx();
    engine
        .write(COLOR_BLEND_ENABLE + 1, 1, true, &mut ctx)
        .unwrap();
    let base = INDEPENDENT_BLEND + 1 * INDEPENDENT_BLEND_STRIDE;
    engine.write(base + 1, 1, true, &mut ctx).unwrap(); // EquationRgb = FUNC_ADD
    engine.write(base + 2, 1, true, &mut ctx).unwrap(); // FuncRgbSrc = ONE
    engine.write(base + 3, 0, true, &mut ctx).unwrap(); // FuncRgbDst = ZERO
    engine
        .write(INDEPENDENT_BLEND_ENABLE, 1, true, &mut ctx)
        .unwrap();
    engine.write(DEPTH_TEST_ENABLE, 1, true, &mut ctx).unwrap();
    engine.write(DEPTH_WRITE_ENABLE, 1, true, &mut ctx).unwrap();
    engine.write(DEPTH_TEST_FUNC, 4, true, &mut ctx).unwrap(); // Lequal

    let bt = engine.blend_target(1);
    assert!(bt.enabled);
    assert_eq!(
        (bt.equation_rgb, bt.func_rgb_src, bt.func_rgb_dst),
        (1, 1, 0)
    );
    assert!(engine.independent_blend_enabled());
    assert_eq!(
        engine.depth_state(),
        DepthState {
            test_enabled: true,
            write_enabled: true,
            func: 4
        }
    );
}
