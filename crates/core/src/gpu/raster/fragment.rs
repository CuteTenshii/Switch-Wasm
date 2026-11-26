//! Depth test, blending and fragment shading.

use super::draw::DrawTally;
use super::setup::{alpha_coverage, TriangleSetup};
use super::vertex::{ShadedVertex, INV_W_OFFSET, VARYING_BASE, VARYING_STRIDE};
use crate::gpu::engine::threed::{
    BlendTarget, DepthState, DepthTarget, Engine3D, RenderTarget, ShaderStage,
};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::interp::{
    resolve_warp, Env, Halt, Invocation, MemoryConstants, MemoryGlobal, MemoryTextures,
};
use crate::gpu::surface::{ColorFormat, SampleGrid, MAX_SAMPLES};
use crate::{Error, Result};

/// `DEPTH_TEST_FUNC` takes either the GL enums (`0x200..=0x207`, as Mesa
/// writes) or the D3D numbering (`1..=8`, as Eden's `ComparisonOp` lists).
pub(super) fn depth_test_passes(func: u32, new: f32, old: f32) -> bool {
    match func {
        1 | 0x0200 => false,
        2 | 0x0201 => new < old,
        3 | 0x0202 => new == old,
        4 | 0x0203 => new <= old,
        5 | 0x0204 => new > old,
        6 | 0x0205 => new != old,
        7 | 0x0206 => new >= old,
        _ => true, // Always (8 or 0x0207), and any unrecognised code.
    }
}

/// `BLEND_FUNC_*` values: the GL blend-factor enums (`G80_BLEND_FACTOR`,
/// `0x4000`+, `0xc000`+ for constant colour) or the D3D numbering.
pub(super) fn blend_factor(
    code: u32,
    src: [f32; 4],
    dst: [f32; 4],
    constant: [f32; 4],
) -> [f32; 4] {
    match code {
        // The D3D numbering (deko3d, nvn).
        0x01 => [0.0; 4],                  // Zero
        0x02 => [1.0; 4],                  // One
        0x03 => src,                       // SrcColor
        0x04 => src.map(|c| 1.0 - c),      // OneMinusSrcColor
        0x05 => [src[3]; 4],               // SrcAlpha
        0x06 => [1.0 - src[3]; 4],         // OneMinusSrcAlpha
        0x07 => [dst[3]; 4],               // DstAlpha
        0x08 => [1.0 - dst[3]; 4],         // OneMinusDstAlpha
        0x09 => dst,                       // DstColor
        0x0a => dst.map(|c| 1.0 - c),      // OneMinusDstColor
        0x61 => constant,                  // ConstantColor
        0x62 => constant.map(|c| 1.0 - c), // OneMinusConstantColor
        0x63 => [constant[3]; 4],          // ConstantAlpha
        0x64 => [1.0 - constant[3]; 4],    // OneMinusConstantAlpha

        // The GL numbering.
        0x4000 => [0.0; 4],                  // Zero
        0x4300 => src,                       // SrcColor
        0x4301 => src.map(|c| 1.0 - c),      // OneMinusSrcColor
        0x4302 => [src[3]; 4],               // SrcAlpha
        0x4303 => [1.0 - src[3]; 4],         // OneMinusSrcAlpha
        0x4304 => [dst[3]; 4],               // DstAlpha
        0x4305 => [1.0 - dst[3]; 4],         // OneMinusDstAlpha
        0x4306 => dst,                       // DstColor
        0x4307 => dst.map(|c| 1.0 - c),      // OneMinusDstColor
        0xc001 => constant,                  // ConstantColor
        0xc002 => constant.map(|c| 1.0 - c), // OneMinusConstantColor
        0xc003 => [constant[3]; 4],          // ConstantAlpha
        0xc004 => [1.0 - constant[3]; 4],    // OneMinusConstantAlpha

        // SrcAlphaSaturate, in both numberings. Alpha's factor is 1.
        0x0b | 0x4308 => {
            let f = src[3].min(1.0 - dst[3]);
            [f, f, f, 1.0]
        }

        _ => [1.0; 4], // One (0x4001), and anything unrecognised.
    }
}

/// `BLEND_EQUATION_*` values: the GL enums (`GL_FUNC_ADD` 0x8006 ..=
/// `GL_FUNC_REVERSE_SUBTRACT` 0x800b) or the D3D numbering.
pub(super) fn blend_equation(op: u32, src: f32, dst: f32) -> f32 {
    match op {
        0x2 | 0x800a => src - dst,    // FuncSubtract
        0x3 | 0x800b => dst - src,    // FuncReverseSubtract
        0x4 | 0x8007 => src.min(dst), // Min
        0x5 | 0x8008 => src.max(dst), // Max
        _ => src + dst,               // FuncAdd (1, 0x8006), and anything unrecognised.
    }
}

/// The shader's output colour as the blend unit sees it: clamped to the
/// range of a fixed-point target (NaN to zero), unchanged for a float one.
pub(super) fn source_color(color: [f32; 4], format: ColorFormat) -> [f32; 4] {
    let Some((low, high)) = format.source_clamp() else {
        return color;
    };
    // NaN floors at zero, not at the SNORM bottom of -1.
    color.map(|c| if c.is_nan() { 0.0 } else { c.clamp(low, high) })
}

pub(super) fn blend(
    target: BlendTarget,
    constant: [f32; 4],
    src: [f32; 4],
    dst: [f32; 4],
) -> [f32; 4] {
    // Min and max take the source and destination as they are, whatever the factors say.
    let unweighted = |equation: u32| matches!(equation, 0x4 | 0x5 | 0x8007 | 0x8008);
    let factor = |code: u32, unweighted: bool| match unweighted {
        true => [1.0; 4],
        false => blend_factor(code, src, dst, constant),
    };
    let rgb_unweighted = unweighted(target.equation_rgb);
    let alpha_unweighted = unweighted(target.equation_alpha);
    let src_rgb = factor(target.func_rgb_src, rgb_unweighted);
    let dst_rgb = factor(target.func_rgb_dst, rgb_unweighted);
    let src_a = factor(target.func_alpha_src, alpha_unweighted)[3];
    let dst_a = factor(target.func_alpha_dst, alpha_unweighted)[3];
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        out[i] = blend_equation(
            target.equation_rgb,
            src[i] * src_rgb[i],
            dst[i] * dst_rgb[i],
        );
    }
    out[3] = blend_equation(target.equation_alpha, src[3] * src_a, dst[3] * dst_a);
    out
}

/// Put one fragment's interpolated inputs in place; `inv` is reused across
/// the draw.
fn seed_fragment(
    inv: &mut Invocation,
    program: &Compiled,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [f32; 3],
    (x, y, window_height): (u32, u32, Option<u32>),
    front: bool,
) {
    inv.reset();
    // All ones on a front face, as the hardware's `a[0x3fc]` reads; Eden's `FrontFace`.
    inv.attr_in.set(
        FRONT_FACING,
        f32::from_bits(if front { u32::MAX } else { 0 }),
    );
    // `a[0x70]` and `a[0x74]`: the pixel's centre in window coordinates.
    inv.attr_in.set(FRAG_COORD_X, x as f32 + 0.5);
    let y = y as f32 + 0.5;
    inv.attr_in.set(
        FRAG_COORD_Y,
        window_height.map_or(y, |height| height as f32 - y),
    );
    let interp_inv_w = weights[0] * inv_w[0] + weights[1] * inv_w[1] + weights[2] * inv_w[2];
    inv.attr_in.set(INV_W_OFFSET, interp_inv_w);
    for &slot in program.interpolated_slots() {
        let base = VARYING_BASE + slot as u16 * VARYING_STRIDE;
        for c in 0..4 {
            let over_w = weights[0] * verts[0].varyings[slot][c] * inv_w[0]
                + weights[1] * verts[1].varyings[slot][c] * inv_w[1]
                + weights[2] * verts[2].varyings[slot][c] * inv_w[2];
            inv.attr_in.set(base + c as u16 * 4, over_w);
        }
    }
}

/// The most colour outputs a fragment program writes.
pub const MAX_OUTPUTS: usize = 8;

/// Each colour output's value, `None` where the program leaves the output alone.
pub type Colors = [Option<[f32; 4]>; MAX_OUTPUTS];

/// The colours an invocation that has run to `exit` leaves behind for its first `outputs`
/// outputs, or `None` if `kil` discarded it. The program header maps registers to components;
/// headerless programs write output 0 from `r0..r3`.
fn fragment_colors(inv: &Invocation, program: &Compiled, outputs: u32) -> Option<Colors> {
    if inv.discarded {
        return None;
    }
    let mut colors: Colors = [None; MAX_OUTPUTS];
    let Some(header) = program.header().filter(|h| h.writes_any_color()) else {
        colors[0] = Some([
            inv.reg_f32(0),
            inv.reg_f32(1),
            inv.reg_f32(2),
            inv.reg_f32(3),
        ]);
        return Some(colors);
    };
    for (output, color) in colors.iter_mut().enumerate().take(outputs as usize) {
        let output = output as u32;
        // Output 0 is always written; a later one the map leaves out keeps what its target holds.
        if output > 0 && !header.writes_target(output) {
            continue;
        }
        *color = Some(std::array::from_fn(|component| {
            match header.fragment_output_reg(output, component as u32) {
                Some(reg) => inv.reg_f32(reg),
                // Unwritten: colour zero, alpha opaque.
                None => (component == 3) as u32 as f32,
            }
        }));
    }
    Some(colors)
}

/// Where a fragment's window coordinates are read from.
const FRAG_COORD_X: u16 = 0x70;
const FRAG_COORD_Y: u16 = 0x74;
const FRONT_FACING: u16 = 0x3fc;

/// Shade one covered pixel.
#[allow(clippy::too_many_arguments)]
pub(super) fn shade_fragment(
    inv: &mut Invocation,
    program: &Compiled,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [f32; 3],
    (x, y): (u32, u32),
    front: bool,
    env: &Env,
    outputs: u32,
) -> Result<Option<Colors>> {
    seed_fragment(
        inv,
        program,
        verts,
        inv_w,
        weights,
        (x, y, env.special.window_height),
        front,
    );
    inv.execute(program, env)?;
    Ok(fragment_colors(inv, program, outputs))
}

/// The four pixels hardware shades together, in lane order: `(x, y)`,
/// `(x + 1, y)`, `(x, y + 1)`, `(x + 1, y + 1)`.
pub const QUAD: usize = 4;

/// Where lane `lane` of a quad based at `(x, y)` sits.
pub(super) fn quad_pixel(x: u32, y: u32, lane: usize) -> (u32, u32) {
    (x + lane as u32 % 2, y + lane as u32 / 2)
}

/// Shade a 2x2 quad in lock-step so warp shuffles can read neighbours.
/// Uncovered lanes run as helpers; diverged lanes read wherever the other
/// lane is.
#[allow(clippy::too_many_arguments)]
pub(super) fn shade_quad(
    lanes: &mut [Invocation; QUAD],
    program: &Compiled,
    verts: &[ShadedVertex; 3],
    inv_w: [f32; 3],
    weights: [[f32; 3]; QUAD],
    (x, y): (u32, u32),
    front: bool,
    env: &mut Env,
    outputs: u32,
) -> Result<[Option<Colors>; QUAD]> {
    for (lane, invocation) in lanes.iter_mut().enumerate() {
        let (px, py) = quad_pixel(x, y, lane);
        let at = (px, py, env.special.window_height);
        seed_fragment(invocation, program, verts, inv_w, weights[lane], at, front);
        invocation.begin();
    }
    let mut running = [true; QUAD];
    loop {
        let mut shuffled = false;
        for (lane, invocation) in lanes.iter_mut().enumerate() {
            if !running[lane] {
                continue;
            }
            env.special.lane = lane as u32;
            match invocation.resume(program, env)? {
                Halt::Exited => running[lane] = false,
                Halt::Warp => shuffled = true,
                Halt::Barrier => {
                    return Err(Error::Gpu(
                        "raster: bar in a fragment shader, where there is no CTA to \
                         synchronise with"
                            .into(),
                    ))
                }
            }
        }
        if !shuffled {
            break;
        }
        resolve_warp(lanes);
    }
    Ok(std::array::from_fn(|lane| {
        fragment_colors(&lanes[lane], program, outputs)
    }))
}

/// One colour output a draw writes, with the target it lands in and how it lands.
#[derive(Debug, Clone, Copy)]
pub(super) struct OutputTarget {
    /// The fragment program's output, an index into [`Colors`].
    pub(super) output: usize,
    pub(super) rt: RenderTarget,
    pub(super) blend: BlendTarget,
    /// Channels written; at least one is.
    pub(super) mask: [bool; 4],
}

/// The per-pixel half of a draw, shared by the pixel and quad walks: sample
/// coverage, depth test, and target writes.
pub(super) struct Fragments {
    pub(super) grid: SampleGrid,
    pub(super) sample_mask: u32,
    pub(super) depth: Option<DepthTarget>,
    pub(super) depth_state: DepthState,
    /// Empty for a depth-only pass.
    pub(super) targets: Vec<OutputTarget>,
    pub(super) blend_constant: [f32; 4],
    pub(super) alpha_to_coverage: bool,
}

impl Fragments {
    /// Which samples of pixel `(x, y)` this triangle covers and the depth test
    /// passes, with each one's depth in `sample_z` (others left alone).
    pub(super) fn coverage(
        &self,
        tri: &TriangleSetup,
        window_z: [f32; 3],
        (x, y): (u32, u32),
        sample_z: &mut [f32; MAX_SAMPLES],
        ctx: &mut ExecCtx,
    ) -> Result<u32> {
        let mut covered = 0u32;
        for sample in 0..self.grid.count() {
            if self.sample_mask >> sample & 1 == 0 {
                continue;
            }
            let [offset_x, offset_y] = self.grid.position(sample);
            let Some(w) = tri.coverage(x as f32 + offset_x, y as f32 + offset_y) else {
                continue;
            };
            let z = w[0] * window_z[0] + w[1] * window_z[1] + w[2] * window_z[2];
            if let (true, Some(dt)) = (self.depth_state.test_enabled, self.depth) {
                let (tx, ty) = self.grid.texel(x, y, sample);
                let bytes = dt.format.bytes;
                let dva = dt.addr + dt.layout.offset(tx * bytes, ty, dt.width * bytes) as u64;
                let old = dt.format.decode_depth(ctx.read_pixel(dva, bytes)? as u64);
                if !depth_test_passes(self.depth_state.func, z, old) {
                    continue;
                }
            }
            covered |= 1 << sample;
            sample_z[sample as usize] = z;
        }
        Ok(covered)
    }

    /// Put a shaded pixel into the targets, for every sample still covered.
    pub(super) fn write(
        &self,
        (x, y): (u32, u32),
        covered: u32,
        sample_z: &[f32; MAX_SAMPLES],
        colors: &Colors,
        ctx: &mut ExecCtx,
        tally: &mut DrawTally,
    ) -> Result<()> {
        if let Some(color) = colors.iter().flatten().next() {
            tally.shaded(*color);
        }

        // Alpha-to-coverage narrows the mask after shading, from output 0's alpha.
        let mut covered = covered;
        if self.alpha_to_coverage {
            let alpha = colors[0].map_or(1.0, |color| color[3]);
            covered &= alpha_coverage(alpha, self.grid.count());
            if covered == 0 {
                tally.alpha_killed += 1;
                return Ok(());
            }
        }

        for sample in 0..self.grid.count() {
            if covered & (1 << sample) == 0 {
                continue;
            }
            let (tx, ty) = self.grid.texel(x, y, sample);
            if self.depth_state.test_enabled && self.depth_state.write_enabled {
                if let Some(dt) = self.depth {
                    let bytes = dt.format.bytes;
                    let dva = dt.addr + dt.layout.offset(tx * bytes, ty, dt.width * bytes) as u64;
                    let z = sample_z[sample as usize];
                    // Merge into a packed depth-stencil pixel's stencil byte.
                    let value = if dt.format.packs_stencil() {
                        dt.format.with_depth(ctx.read_pixel(dva, bytes)? as u64, z)
                    } else {
                        dt.format.encode_depth(z)
                    };
                    ctx.write_pixel(dva, bytes, u128::from(value))?;
                }
            }

            for target in &self.targets {
                let Some(color) = colors[target.output] else {
                    continue;
                };
                let rt = target.rt;
                let bpp = rt.format.bytes_per_pixel;
                let va = rt.addr + rt.texel_offset(tx, ty) as u64;
                // Blending and masking both need the existing value.
                let dst = if target.blend.enabled || target.mask != [true; 4] {
                    Some(rt.format.decode(ctx.read_pixel(va, bpp)?)?)
                } else {
                    None
                };
                let mut out = color;
                if let Some(dst) = dst {
                    if target.blend.enabled {
                        let src = source_color(color, rt.format);
                        out = blend(target.blend, self.blend_constant, src, dst);
                    }
                    for (channel, keep) in target.mask.iter().enumerate() {
                        if !keep {
                            out[channel] = dst[channel];
                        }
                    }
                }
                tally.wrote(out);
                ctx.write_pixel(va, bpp, rt.format.encode(out)?)?;
            }
        }
        Ok(())
    }
}

/// Build the environment a fragment shader runs under and hand it to `f`,
/// so it cannot outlive one shading step.
pub(super) fn with_fragment_env<T>(
    engine: &Engine3D,
    ctx: &ExecCtx,
    consts: &std::cell::RefCell<crate::gpu::shader::interp::ConstCache>,
    descriptors: &std::cell::RefCell<crate::gpu::texture::DescriptorCache>,
    blocks: &std::cell::RefCell<crate::gpu::texture::BlockCache>,
    stores: &std::cell::RefCell<Vec<(u64, u32)>>,
    surface_stores: &std::cell::RefCell<Vec<crate::gpu::shader::interp::SurfaceWrite>>,
    f: impl FnOnce(&mut Env) -> Result<T>,
) -> Result<T> {
    let fs_consts = MemoryConstants {
        ctx,
        bindings: &|bank: u8| engine.bound_constbuf(ShaderStage::Fragment, bank as u32),
        cache: consts,
    };
    let fs_textures = MemoryTextures {
        ctx,
        tex_header_pool: engine.tex_header_pool(),
        tex_sampler_pool: engine.tex_sampler_pool(),
        descriptors,
        blocks,
        surface_stores: Some(surface_stores),
    };
    let fs_global = MemoryGlobal { ctx, stores };
    let mut env = Env::with_tex_cb_index(&fs_consts, &fs_textures, engine.tex_cb_index());
    env.memory = Some(&fs_global);
    env.special.y_negate = engine.window_origin().lower_left;
    env.special.window_height = env.special.y_negate.then(|| engine.surface_clip_height());
    f(&mut env)
}
