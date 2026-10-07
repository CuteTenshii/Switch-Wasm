//! Translates a lowered program to WGSL. Control flow is a `switch` over a
//! program counter inside a `loop`, with an explicit reconvergence stack, so
//! nothing is restructured. Registers are untyped `u32`s. [`translate`] emits the
//! body against [`HOST_INTERFACE`]; [`module`] wraps it into a complete module.
//! `TRACE_WGSL=<dir>` dumps every module a run uses, for checking with `naga`.

use super::compiled::{Compiled, NO_TARGET};
use super::isa::{Op, TexDim};
use super::{interp, isa};
use crate::gpu::texture::TextureSlot;
use std::collections::BTreeSet;
use std::fmt;

mod alu;
mod blocks;
mod convert;
mod emitter;
mod float;
mod helpers;
mod integer;
mod layout;
mod module;
#[cfg(test)]
mod tests;
mod texture;

use emitter::Emitter;
pub use layout::{Coverage, Layout, Stage, TextureBinding, GLOBAL_BINDING, IDENTITY_SWIZZLE};
pub use module::module;

/// Why a program could not be translated; the caller falls back to the rasterizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// An opcode with no WGSL form here.
    Op { at: usize, op: Op },
    /// A branch target that was never decoded.
    UndecodedTarget { at: usize },
    /// A `brx` whose jump table could not be read.
    IndirectBranch { at: usize },
    /// A texture dimensionality [`module`] cannot bind.
    TextureDimension { dim: TexDim },
    /// An instruction that needs the 2x2 quad outside a fragment shader.
    Quad { at: usize },
    /// A shadow sample: a guest shadow map cannot become a `texture_depth_*`.
    DepthCompare { at: usize },
    /// A bindless `tex.b` whose handle is not loaded straight from a constant bank.
    UntracedHandle { at: usize },
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsupported::Op { at, op } => {
                write!(f, "instruction {at}: no WGSL form for {op:?}")
            }
            Unsupported::Quad { at } => {
                write!(
                    f,
                    "instruction {at}: a quad operation outside a fragment shader"
                )
            }
            Unsupported::DepthCompare { at } => {
                write!(
                    f,
                    "instruction {at}: a shadow sample needs a depth texture and a comparison sampler"
                )
            }
            Unsupported::UndecodedTarget { at } => {
                write!(
                    f,
                    "instruction {at}: branches to a target that was never decoded"
                )
            }
            Unsupported::IndirectBranch { at } => {
                write!(f, "instruction {at}: brx with no known targets")
            }
            Unsupported::TextureDimension { dim } => {
                write!(f, "no binding for a {dim:?} texture")
            }
            Unsupported::UntracedHandle { at } => {
                write!(
                    f,
                    "instruction {at}: a bindless handle not loaded from a constant bank"
                )
            }
        }
    }
}

/// The functions the emitted text calls, as compilable stubs; [`module`]
/// supplies the real ones. `texSample` takes a [`TextureSlot::key`]; `dim` is [`tex_dim_code`].
pub const HOST_INTERFACE: &str = "\
fn attrIn(offset: u32) -> f32 { return 0.0; }
fn attrOut(offset: u32, value: f32) { }
fn cbRead(bank: u32, offset: u32) -> u32 { return 0u; }
fn gRead(slot: u32, offset: u32) -> u32 { return 0u; }
fn texSample(imm: u32, dim: u32, u: f32, v: f32, layer: u32, w: f32) -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
fn texSampleCompare(imm: u32, dim: u32, u: f32, v: f32, layer: u32, dref: f32) -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
fn texDims(imm: u32, lod: u32) -> vec4<u32> {
  return vec4<u32>(1u, 1u, 1u, 1u);
}
fn texSampleOffset(imm: u32, offset: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
fn texGather(imm: u32, component: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
";

/// Emitted reconvergence stack depth, Maxwell's own.
const RECONVERGENCE_DEPTH: usize = 16;

/// The `dim` code [`HOST_INTERFACE`]'s `texSample` receives.
pub fn tex_dim_code(dim: TexDim) -> u32 {
    match dim {
        TexDim::T1d => 0,
        TexDim::T2d => 1,
        TexDim::T2dArray => 2,
        TexDim::T3d => 3,
        TexDim::TCube => 4,
        TexDim::TCubeArray => 5,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translation {
    /// WGSL source text; needs [`HOST_INTERFACE`] in front to compile.
    pub source: String,
    /// Declared registers, ascending; `var<private>` because a fragment's colour is `r0`..`r3`.
    pub registers: Vec<u8>,
    /// Generic `a[]` slots read, ascending.
    pub loads: Vec<usize>,
    /// Generic `a[]` slots written, ascending.
    pub stores: Vec<usize>,
    /// The [`Translation::loads`] read with `ipa.centroid`, ascending.
    pub centroid_loads: Vec<usize>,
    /// Constant banks read, ascending.
    pub const_banks: Vec<u8>,
    /// Textures sampled, in first-mention order, with dimension and shadow flag.
    pub textures: Vec<(TextureSlot, TexDim, bool)>,
    /// Distinct `tex.aoffi` offsets `(x, y)`, named by index since WGSL needs constants.
    pub texture_offsets: Vec<(i32, i32)>,
    /// The first instruction asking which quad lane it is.
    pub quad: Option<usize>,
    /// The first instruction reading another lane.
    pub quad_swap: Option<usize>,
    /// Whether that uses the device's quad operations rather than [`QUAD_SWAP`].
    pub subgroups: bool,
    pub subgroup_enable: bool,
    /// `(bank, offset)` of each 64-bit `ldg` descriptor, in binding order.
    pub globals: Vec<(u8, u16)>,
}

/// Translate into a WGSL function `run`, returning whether it hit `kil`.
pub fn translate(program: &Compiled) -> Result<Translation, Unsupported> {
    translate_for(program, Caps::NONE)
}

/// Device features WGSL does not guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Caps {
    /// WGSL quad operations; without them [`QUAD_SWAP`] stands in.
    pub subgroups: bool,
    /// Whether to write `enable subgroups;`: browsers require it, naga rejects it.
    pub subgroup_enable: bool,
}

impl Caps {
    pub const NONE: Caps = Caps {
        subgroups: false,
        subgroup_enable: false,
    };
}

pub fn translate_for(program: &Compiled, caps: Caps) -> Result<Translation, Unsupported> {
    let leaders = leaders(program)?;
    let mut emitter = Emitter::new(program);
    emitter.emit_blocks(&leaders)?;
    // Nintendo Switch Sports queries texture sizes without sampling.
    for &slot in &emitter.queried {
        if !emitter.textures.iter().any(|&(seen, _, _)| seen == slot) {
            emitter.textures.push((slot, TexDim::T2d, false));
        }
    }
    Ok(Translation {
        source: emitter.finish(&leaders),
        registers: emitter.regs.iter().copied().collect(),
        loads: emitter.loads.iter().copied().collect(),
        stores: emitter.stores.iter().copied().collect(),
        centroid_loads: emitter.centroid_loads.iter().copied().collect(),
        const_banks: emitter.banks.iter().copied().collect(),
        textures: emitter.textures.clone(),
        texture_offsets: emitter.texture_offsets.clone(),
        quad: emitter.quad,
        quad_swap: emitter.quad_swap,
        subgroups: caps.subgroups,
        subgroup_enable: caps.subgroup_enable,
        globals: emitter.globals.clone(),
    })
}

/// Whether an instruction ends its block. `ssy`/`pbk`/`pcnt` push and fall through.
fn is_terminator(op: Op) -> bool {
    matches!(
        op,
        Op::Bra { .. } | Op::Brx { .. } | Op::Exit | Op::Kil | Op::Sync | Op::Brk | Op::Cont
    )
}

/// Block starts: the entry, branch targets, and instructions after terminators.
fn leaders(program: &Compiled) -> Result<Vec<usize>, Unsupported> {
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    leaders.insert(0);
    for at in 0..program.len() {
        let op = program.op(at);
        match op {
            Op::Bra { .. } | Op::Ssy { .. } | Op::Pbk { .. } | Op::Pcnt { .. } => {
                let target = program.target(at);
                if target == NO_TARGET {
                    return Err(Unsupported::UndecodedTarget { at });
                }
                leaders.insert(target as usize);
            }
            Op::Brx { .. } => match program.indirect_targets(at) {
                Some(targets) => leaders.extend(targets.iter().map(|&t| t as usize)),
                None => return Err(Unsupported::IndirectBranch { at }),
            },
            _ => {}
        }
        if is_terminator(op) && at + 1 < program.len() {
            leaders.insert(at + 1);
        }
    }
    Ok(leaders.into_iter().collect())
}
