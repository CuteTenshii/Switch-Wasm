//! Vertex shading and the viewport transform.

use super::attrib::fetch_attribute;
use super::ScreenVertex;
use crate::gpu::engine::threed::{VertexArray, VertexAttrib, ViewportTransform};
use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::compiled::Compiled;
use crate::gpu::shader::interp::{ConstantSource, Env, Invocation, MemoryGlobal, NoTextures};
use crate::Result;

/// The `a[]` offset of the clip position, and the fixed interpolated-`1/w` slot.
pub(super) const CLIP_POS_OFFSET: u16 = 0x70;
pub(super) const INV_W_OFFSET: u16 = 0x7c;
/// Generic varying `i`'s `a[]` slot, the same for vertex outputs and
/// fragment inputs.
pub(super) const VARYING_BASE: u16 = 0x80;
pub(super) const VARYING_STRIDE: u16 = 0x10;

/// `gl_InstanceID`'s and `gl_VertexID`'s slots in a vertex shader's `a[]`
/// input space, holding integer bits.
pub(super) const INSTANCE_ID_OFFSET: u16 = 0x2f8;
pub(super) const VERTEX_ID_OFFSET: u16 = 0x2fc;
/// How many generic varying slots exist: Maxwell's `a[0x80]..a[0x280)`.
/// [`Program::interpolated_slots`] narrows it to what the shader reads.
pub(super) const NUM_VARYINGS: usize = 32;
/// Vertex attribute slots scanned, so shaders need not declare a count.
pub(super) const MAX_VERTEX_ATTRIBS: u32 = 16;
/// One vertex after the vertex shader: clip position and every varying.
#[derive(Clone, Copy)]
pub(super) struct ShadedVertex {
    pub(super) clip: [f32; 4],
    pub(super) varyings: [[f32; 4]; NUM_VARYINGS],
}

pub(super) fn shade_vertex(
    program: &Compiled,
    attribs: &[VertexAttrib],
    arrays: &[VertexArray],
    // The two ordinals that pick this invocation's data out of the arrays.
    (vertex_index, instance_id): (u32, u32),
    ctx: &ExecCtx,
    consts: &dyn ConstantSource,
    y_negate: bool,
    stores: &std::cell::RefCell<Vec<(u64, u32)>>,
) -> Result<ShadedVertex> {
    let mut inv = Invocation::new();
    inv.attr_in
        .set(VERTEX_ID_OFFSET, f32::from_bits(vertex_index));
    inv.attr_in
        .set(INSTANCE_ID_OFFSET, f32::from_bits(instance_id));
    for (i, attrib) in attribs.iter().enumerate() {
        // Size 0 is an unconfigured slot: unused, not unsupported.
        if attrib.size == 0 {
            continue;
        }
        let array = arrays[attrib.buffer_id as usize];
        if !array.enabled {
            continue;
        }
        // A nonzero divisor advances the array once per `divisor` instances.
        let element = instance_id
            .checked_div(array.divisor)
            .unwrap_or(vertex_index);
        let v = fetch_attribute(*attrib, array, element, ctx)?;
        let base = VARYING_BASE + i as u16 * VARYING_STRIDE;
        for (c, &component) in v.iter().enumerate() {
            inv.attr_in.set(base + c as u16 * 4, component);
        }
    }
    let global = MemoryGlobal { ctx, stores };
    let mut env = Env::new(consts, &NoTextures);
    env.memory = Some(&global);
    env.special.y_negate = y_negate;
    inv.execute(program, &env)?;

    let mut clip = [0.0, 0.0, 0.0, 1.0];
    for (c, slot) in clip.iter_mut().enumerate() {
        if let Some(v) = inv.attr_out.written(CLIP_POS_OFFSET + c as u16 * 4) {
            *slot = v;
        }
    }
    let mut varyings = [[0.0f32; 4]; NUM_VARYINGS];
    for (i, varying) in varyings.iter_mut().enumerate() {
        let base = VARYING_BASE + i as u16 * VARYING_STRIDE;
        for (c, slot) in varying.iter_mut().enumerate() {
            if let Some(v) = inv.attr_out.written(base + c as u16 * 4) {
                *slot = v;
            }
        }
    }
    Ok(ShadedVertex { clip, varyings })
}

/// Clip position to window space: perspective divide, then the guest's
/// viewport transform. Also returns `1/w` and window depth, both affine in
/// screen space.
pub(super) fn to_screen(clip: [f32; 4], vt: ViewportTransform) -> (ScreenVertex, f32, f32) {
    let inv_w = 1.0 / clip[3];
    let screen = ScreenVertex {
        x: clip[0] * inv_w * vt.scale[0] + vt.translate[0],
        y: clip[1] * inv_w * vt.scale[1] + vt.translate[1],
    };
    (
        screen,
        inv_w,
        clip[2] * inv_w * vt.scale[2] + vt.translate[2],
    )
}
