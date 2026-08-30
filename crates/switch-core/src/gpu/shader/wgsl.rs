//! Translating a lowered program to WGSL.
//!
//! This is the half of a GPU backend that does not need a GPU: it turns a
//! [`Compiled`] program into WGSL source text and nothing else. What binds
//! that text to real buffers, textures and render targets is a separate
//! problem, and deliberately not this module's.
//!
//! # Why the shape below, and not structured control flow
//!
//! Maxwell has no `if`/`else`/`loop`. It has a reconvergence stack — `ssy`
//! pushes an address, `sync` pops one and jumps there — plus ordinary
//! branches and `brx`, an indexed jump into a table. [`super::cfg`] can prove
//! that the pushes and pops of a given program pair up statically, which is
//! what a *structured* translation would need, and every Home Menu shader
//! does. But a proof that structure exists is not the structure, and two
//! things in these programs resist nesting directly: a backward `bra` is a
//! loop whose head is wherever it points, and `brx` is a multi-way jump.
//!
//! So this emits the form that is correct for all of it: the program becomes
//! a `switch` over a program counter inside a `loop`, one `case` per basic
//! block, with the reconvergence stack as an explicit array — the same
//! machine [`super::interp::Invocation`] runs, written in WGSL. Every branch
//! is an assignment to `pc`. Nothing about the control flow can be
//! mistranslated because nothing about it is *re*structured.
//!
//! That form is slower on a GPU than nested blocks, because the shader
//! compiler cannot see the loop structure. Recovering that structure where
//! [`super::cfg`] says it is safe is worth doing, and is a change to this
//! module with the state machine as the fallback — not a change to anything
//! else.
//!
//! # What the register file looks like
//!
//! Maxwell's registers are untyped 32 bits and an instruction decides how to
//! read them, so every register here is a `u32` and float operations go
//! through `bitcast<f32>`. That is not a translation artefact: a shader
//! genuinely computes an address with integer instructions in the same
//! register it later loads a float into, and typing the register file would
//! be wrong rather than tidier.
//!
//! Only the registers a program touches are declared, which is why the
//! emitter records them as it goes rather than scanning first — a second pass
//! over the opcodes is a second place to get the list wrong.
//!
//! # Two layers
//!
//! [`translate`] produces the shader's *body*: a function `run` that reaches
//! outside itself through four calls it does not define, because attribute
//! space, constant banks and textures live in guest memory that only a
//! backend knows how to address. [`HOST_INTERFACE`] is their signatures.
//!
//! [`module`] wraps that in everything it needs to be a shader module — the
//! bindings, the attribute storage, real implementations of those four calls,
//! and the `@vertex` or `@fragment` entry point that fills attribute space in
//! and takes the result out. What it needs to know to do that is a
//! [`Layout`], and a layout can be read off the translation itself, because
//! Maxwell's attribute space *is* the interface: a vertex shader's inputs are
//! the `a[]` offsets it loads from and its outputs are the ones it stores to.
//!
//! # Checking a translation
//!
//! Nothing in this crate can parse WGSL, and a translation that is merely
//! plausible is worth very little. `TRACE_WGSL=<dir>` on a real run writes
//! every shader a title uses to that directory as a complete module, and
//! `naga` then says whether it is one — `--validate 31` for the front end
//! Firefox compiles WGSL with, or an output path to compile the whole way
//! down to SPIR-V, HLSL, MSL or GLSL, which is what a pipeline actually does
//! with it. Both are development steps; this crate still has no
//! dependencies.

use super::compiled::{Compiled, NO_TARGET};
use super::isa::{
    BoolOp, FCmp, FMod, FRound, HMerge, HPrecision, HSwizzle, ICmp, LogicOp, LopTest, MufuOp, Op,
    Operand, Pred, ShflMode, TexDim, TexsStore, XmadC, RZ,
};
use crate::gpu::pipeline::AttributeBase;
use crate::gpu::texture::SwizzleSource;
use std::collections::BTreeSet;
use std::fmt;

/// Why a program could not be translated.
///
/// Separate from [`crate::Error`] because these are statements about the
/// program rather than failures of the translation: a caller that gets one
/// falls back to the software rasterizer for that draw, which is a normal
/// thing to do and not an error to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// An opcode with no WGSL form here. `Ldg`/`Stg`/`Ldl`/`Stl` are the
    /// deliberate ones: global and local memory need storage buffers, which
    /// is a resource-binding question rather than a translation one.
    Op { at: usize, op: Op },
    /// A branch whose target was never decoded, so there is no block to jump
    /// to. The interpreter raises this where the branch is taken; a
    /// translation has to know before it emits anything.
    UndecodedTarget { at: usize },
    /// A `brx` whose jump table the decoder could not read, so its arms are
    /// unknown; see the jump-table walk in this module's parent.
    IndirectBranch { at: usize },
    /// A texture dimensionality [`module`] cannot bind. The software
    /// rasterizer samples every dimension as though it were 2D, which is
    /// what a `texture_2d` binding would then be lying about.
    TextureDimension { dim: TexDim },
    /// An instruction that needs the 2x2 quad, in a stage that has none.
    ///
    /// Only a fragment shader has one here: its entry point reads the lane
    /// out of `position`, and there is nothing to read it out of anywhere
    /// else. A `shfl` also needs a way to reach the lane it names, which is
    /// `quadSwapX`/`Y`/`Diagonal` where the device has them and [`QUAD_SWAP`]
    /// where it does not — and derivatives are a fragment shader's too.
    Quad { at: usize },
    /// A shadow sample. WGSL compares depth only through
    /// `textureSampleCompare` on a `texture_depth_*` bound beside a
    /// `sampler_comparison`, and a guest shadow map cannot become one:
    /// WebGPU allows a copy into a `depth32float` texture *only from another
    /// texture of the same format* (§26.1.2.2), so an uploaded buffer cannot
    /// reach it. Getting there means a render pass per shadow map writing
    /// `frag_depth`, which is the workaround the specification itself names.
    ///
    /// Until then the rasterizer takes these draws, and it does implement the
    /// comparison — see `texture::sample_compare_with`.
    DepthCompare { at: usize },
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
        }
    }
}

/// The functions the emitted text calls and does not define.
///
/// Given here as compilable WGSL with bodies that answer nothing, so that a
/// bare [`translate`] result can be parsed and checked on its own.
/// [`module`] supplies the real ones: `attrIn`/`attrOut` reach the attribute
/// space its entry point fills, `cbRead` a bound constant buffer, and
/// `texSample` a bound texture.
///
/// `texSample` takes the *immediate* a `texs` carries rather than a texture
/// handle, because turning one into the other means reading the driver's
/// reserved constant bank at an offset only the engine knows — see
/// [`crate::gpu::texture`]. `dim` is [`tex_dim_code`].
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
";

/// How deep the emitted reconvergence stack is.
///
/// Maxwell's own is 16 entries and a program that overflows it is broken on
/// hardware too; the deepest any shader put through this has needed is 3.
const RECONVERGENCE_DEPTH: usize = 16;

/// The `dim` code [`HOST_INTERFACE`]'s `texSample` receives.
pub fn tex_dim_code(dim: TexDim) -> u32 {
    match dim {
        TexDim::T1d => 0,
        TexDim::T2d => 1,
        TexDim::T2dArray => 2,
        TexDim::T3d => 3,
        TexDim::TCube => 4,
    }
}

/// A translated program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translation {
    /// WGSL source text — not a module: it needs [`HOST_INTERFACE`], or a
    /// backend's replacement for it, in front of it to compile.
    pub source: String,
    /// The registers the program declares, ascending.
    ///
    /// They are `var<private>`, and not because anything in the translation
    /// needs them to be: a **fragment shader's colour is its registers**.
    /// Maxwell has no output attribute for it — the rasterizer reads `r0` to
    /// `r3` after the invocation ends, and a GPU backend has to do the same,
    /// so the register file has to outlive the call. A vertex shader's
    /// outputs go the other way, through `attrOut`.
    pub registers: Vec<u8>,
    /// Generic `a[]` slots the program reads, ascending — a vertex shader's
    /// vertex attributes, a fragment shader's varyings.
    pub loads: Vec<usize>,
    /// Generic `a[]` slots the program writes, ascending: a vertex shader's
    /// varyings.
    pub stores: Vec<usize>,
    /// The subset of [`Translation::loads`] a fragment shader interpolates
    /// with `ipa.centroid`, ascending.
    pub centroid_loads: Vec<usize>,
    /// The constant banks it reads, ascending.
    pub const_banks: Vec<u8>,
    /// The textures it samples, in the order it first mentions them: the
    /// `texs` immediate and the dimensionality sampled with.
    /// Each `texs` immediate, what it samples as, and whether it is
    /// sampled as a shadow map — a depth image compared against a reference
    /// rather than read.
    pub textures: Vec<(u16, TexDim, bool)>,
    /// The first instruction that asks which lane of the 2x2 quad it is, if
    /// any.
    pub quad: Option<usize>,
    /// The first instruction that reads *another* lane, if any. `fswzadd`
    /// asks which lane this is without reading one, so it sets [`quad`] and
    /// leaves this `None`.
    ///
    /// [`quad`]: Translation::quad
    pub quad_swap: Option<usize>,
    /// Whether that read is the device's own quad operations rather than
    /// [`QUAD_SWAP`].
    pub subgroups: bool,
    /// Whether that needs `enable subgroups;` in front of the module.
    pub subgroup_enable: bool,
    /// The `(bank, offset)` of each 64-bit descriptor a `ldg` reads memory
    /// through, in the order the module binds them.
    pub globals: Vec<(u8, u16)>,
}

/// Translate `program` into a WGSL function `run`, which returns whether the
/// invocation discarded itself with `kil`.
pub fn translate(program: &Compiled) -> Result<Translation, Unsupported> {
    translate_for(program, Caps::NONE)
}

/// What a device can do that WGSL does not guarantee, and so that the
/// translation has to be told rather than assume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Caps {
    /// WGSL's quad operations, which are what a warp shuffle is. Without
    /// them a fragment shader reads its neighbour through [`QUAD_SWAP`]
    /// instead, and nothing else can read one at all — see
    /// [`Unsupported::Quad`].
    pub subgroups: bool,
    /// Whether to write `enable subgroups;` in front of the module.
    ///
    /// The standard requires the directive, and a browser compiling the text
    /// enforces that. Naga recognises the built-ins on their own and rejects
    /// the directive as not yet implemented, so a native device wants the
    /// module without it. Whoever compiles the text decides.
    pub subgroup_enable: bool,
}

impl Caps {
    /// Only what every WebGPU device has.
    pub const NONE: Caps = Caps {
        subgroups: false,
        subgroup_enable: false,
    };
}

/// [`translate`], told what the device it is being built for can do.
pub fn translate_for(program: &Compiled, caps: Caps) -> Result<Translation, Unsupported> {
    let leaders = leaders(program)?;
    let mut emitter = Emitter::new(program);
    emitter.emit_blocks(&leaders)?;
    Ok(Translation {
        source: emitter.finish(&leaders),
        registers: emitter.regs.iter().copied().collect(),
        loads: emitter.loads.iter().copied().collect(),
        stores: emitter.stores.iter().copied().collect(),
        centroid_loads: emitter.centroid_loads.iter().copied().collect(),
        const_banks: emitter.banks.iter().copied().collect(),
        textures: emitter.textures.clone(),
        quad: emitter.quad,
        quad_swap: emitter.quad_swap,
        subgroups: caps.subgroups,
        subgroup_enable: caps.subgroup_enable,
        globals: emitter.globals.clone(),
    })
}

/// Whether an instruction ends its basic block: everything that can move the
/// program counter other than by one.
///
/// `ssy`/`pbk`/`pcnt` are absent on purpose. They push a target and fall
/// through, so they are ordinary statements — it is the address they push
/// that starts a block, not the push.
fn is_terminator(op: Op) -> bool {
    matches!(
        op,
        Op::Bra { .. } | Op::Brx { .. } | Op::Exit | Op::Kil | Op::Sync | Op::Brk | Op::Cont
    )
}

/// The instruction indices that start a basic block: the entry, everything a
/// branch can reach, and everything after a terminator.
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

/// The WGSL definitions of the helpers the emitter can call, in an order that
/// satisfies their dependencies on each other. Only the ones a program
/// reaches are emitted, so a translation carries no code it does not run.
const HELPERS: &[(&str, &str)] = &[
    (
        "ftz",
        "\
fn ftz(v: f32) -> f32 {
  // `.ftz` flushes a subnormal to a zero of the same sign.
  if (v != 0.0 && abs(v) < 1.1754943508222875e-38) {
    return bitcast<f32>(bitcast<u32>(v) & 0x80000000u);
  }
  return v;
}",
    ),
    (
        "fsat",
        "\
fn fsat(v: f32) -> f32 {
  // A saturating instruction answers 0 for NaN rather than propagating it.
  if (v != v) { return 0.0; }
  return clamp(v, 0.0, 1.0);
}",
    ),
    (
        "hftz",
        "\
fn hftz(v: vec2<f32>) -> vec2<f32> {
  // A half instruction's `.ftz` flushes a subnormal *half* to a zero of the
  // same sign, which starts four orders of magnitude above an f32's.
  let signs = bitcast<vec2<f32>>(bitcast<vec2<u32>>(v) & vec2<u32>(0x80000000u));
  return select(v, signs, abs(v) < vec2<f32>(6.103515625e-5));
}",
    ),
    (
        "ftz2",
        "\
fn ftz2(v: vec2<f32>) -> vec2<f32> {
  let signs = bitcast<vec2<f32>>(bitcast<vec2<u32>>(v) & vec2<u32>(0x80000000u));
  return select(v, signs, abs(v) < vec2<f32>(1.1754943508222875e-38));
}",
    ),
    (
        "fsat2",
        "\
fn fsat2(v: vec2<f32>) -> vec2<f32> {
  // A saturating instruction answers 0 for NaN rather than propagating it.
  return clamp(select(v, vec2<f32>(0.0), v != v), vec2<f32>(0.0), vec2<f32>(1.0));
}",
    ),
    (
        "shl32",
        "\
fn shl32(a: u32, n: u32) -> u32 {
  if (n >= 32u) { return 0u; }
  return a << n;
}",
    ),
    (
        "shr32",
        "\
fn shr32(a: u32, n: u32) -> u32 {
  if (n >= 32u) { return 0u; }
  return a >> n;
}",
    ),
    (
        "sar32",
        "\
fn sar32(a: u32, n: u32) -> u32 {
  let x = bitcast<i32>(a);
  if (n >= 32u) { return bitcast<u32>(x >> 31u); }
  return bitcast<u32>(x >> n);
}",
    ),
    (
        "shf",
        "\
fn shf(lo: u32, hi: u32, count: u32, left: bool, hi_out: bool) -> u32 {
  // A 64-bit shift of a register pair, in a language with no 64-bit integer.
  let n = count & 63u;
  var rlo = lo;
  var rhi = hi;
  if (left) {
    if (n >= 32u) { rhi = shl32(lo, n - 32u); rlo = 0u; }
    else if (n > 0u) { rhi = (hi << n) | (lo >> (32u - n)); rlo = lo << n; }
  } else {
    if (n >= 32u) { rlo = shr32(hi, n - 32u); rhi = 0u; }
    else if (n > 0u) { rlo = (lo >> n) | (hi << (32u - n)); rhi = hi >> n; }
  }
  if (hi_out) { return rhi; }
  return rlo;
}",
    ),
    (
        "bfe",
        "\
fn bfe(v: u32, start0: u32, width0: u32, signed: bool) -> u32 {
  if (width0 == 0u) { return 0u; }
  let start = min(start0, 31u);
  let width = min(width0, 32u - start);
  let raw = (v >> start) & (0xffffffffu >> (32u - width));
  if (signed && width < 32u && (raw & (1u << (width - 1u))) != 0u) {
    return raw | ~(0xffffffffu >> (32u - width));
  }
  return raw;
}",
    ),
    (
        "bfi",
        "\
fn bfi(insert: u32, src: u32, base: u32) -> u32 {
  // `src` carries the field's offset in its low byte and its width in the
  // next. An offset past the word leaves the base alone, and a width that
  // would run off the end is clamped to what is left.
  let offset = src & 0xffu;
  if (offset >= 32u) { return base; }
  let count = min((src >> 8u) & 0xffu, 32u - offset);
  var mask = 0xffffffffu;
  if (count < 32u) { mask = ((1u << count) - 1u) << offset; }
  return (base & ~mask) | ((insert << offset) & mask);
}",
    ),
    (
        "lop3",
        "\
fn lop3(a: u32, b: u32, c: u32, lut: u32) -> u32 {
  // Bit n of the truth table is the result for the input combination whose
  // bits are (a, b, c) read as a three-bit number.
  var out = 0u;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if ((lut & (1u << i)) == 0u) { continue; }
    var m = 0xffffffffu;
    if ((i & 4u) != 0u) { m = m & a; } else { m = m & ~a; }
    if ((i & 2u) != 0u) { m = m & b; } else { m = m & ~b; }
    if ((i & 1u) != 0u) { m = m & c; } else { m = m & ~c; }
    out = out | m;
  }
  return out;
}",
    ),
    (
        "flo",
        "\
fn flo(v0: u32, signed: bool, shift: bool) -> u32 {
  // The highest set bit, counting from bit 0; a signed search ignores the
  // sign bits at the top.
  var v = v0;
  if (signed && bitcast<i32>(v) < 0) { v = ~v; }
  if (v == 0u) { return 0xffffffffu; }
  let index = 31u - countLeadingZeros(v);
  if (shift) { return 31u - index; }
  return index;
}",
    ),
    (
        "mulhi_u",
        "\
fn mulhi_u(a: u32, b: u32) -> u32 {
  let a0 = a & 0xffffu; let a1 = a >> 16u;
  let b0 = b & 0xffffu; let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}",
    ),
    (
        "mulhi_s",
        "\
fn mulhi_s(a: u32, b: u32) -> u32 {
  var hi = mulhi_u(a, b);
  if (bitcast<i32>(a) < 0) { hi = hi - b; }
  if (bitcast<i32>(b) < 0) { hi = hi - a; }
  return hi;
}",
    ),
    (
        "sext",
        "\
fn sext(v: u32, bytes: u32) -> u32 {
  if (bytes == 1u) { return bitcast<u32>(bitcast<i32>(v << 24u) >> 24u); }
  if (bytes == 2u) { return bitcast<u32>(bitcast<i32>(v << 16u) >> 16u); }
  return v;
}",
    ),
    (
        "truncw",
        "\
fn truncw(v: u32, bytes: u32) -> u32 {
  if (bytes == 1u) { return v & 0xffu; }
  if (bytes == 2u) { return v & 0xffffu; }
  return v;
}",
    ),
    (
        "f2i_s",
        "\
fn f2i_s(v: f32, bytes: u32) -> u32 {
  // Out of range saturates and NaN is zero, and the result is the value
  // sign-extended to 32 bits however narrow the destination was.
  if (v != v) { return 0u; }
  let bits = bytes * 8u;
  let limit = exp2(f32(bits - 1u));
  if (v >= limit) { return (1u << (bits - 1u)) - 1u; }
  if (v <= -limit) { return bitcast<u32>(-(1i << (bits - 1u))); }
  return bitcast<u32>(i32(v));
}",
    ),
    (
        "f2i_u",
        "\
fn f2i_u(v: f32, bytes: u32) -> u32 {
  if (v != v) { return 0u; }
  if (v <= 0.0) { return 0u; }
  let bits = bytes * 8u;
  if (v >= exp2(f32(bits))) {
    if (bits >= 32u) { return 0xffffffffu; }
    return (1u << bits) - 1u;
  }
  return u32(v);
}",
    ),
];

/// Builds the text, recording what it used as it goes.
struct Emitter<'a> {
    program: &'a Compiled,
    body: String,
    indent: usize,
    /// Registers, predicates and helpers the emitted text refers to.
    /// Collected while emitting rather than by a pass beforehand: a second
    /// walk over the opcodes would be a second place to get the list wrong,
    /// and a register missing from it is a compile error in the output.
    regs: BTreeSet<u8>,
    preds: BTreeSet<u8>,
    helpers: BTreeSet<&'static str>,
    uses_carry: bool,
    uses_stack: bool,
    /// The first instruction that asks which lane of its quad it is.
    quad: Option<usize>,
    /// The first instruction that reads another lane, if any.
    quad_swap: Option<usize>,
    /// The interface the emitted text reaches through, recorded as it is
    /// emitted so that a binding it calls cannot be left out of the module.
    loads: BTreeSet<usize>,
    stores: BTreeSet<usize>,
    centroid_loads: BTreeSet<usize>,
    banks: BTreeSet<u8>,
    textures: Vec<(u16, TexDim, bool)>,
    /// The constant-bank descriptors a `ldg` reads memory through.
    globals: Vec<(u8, u16)>,
    /// Names `let` bindings apart. WGSL scopes them to their block, but one
    /// counter across the whole function is simpler than reasoning about it.
    temps: usize,
}

impl<'a> Emitter<'a> {
    fn new(program: &'a Compiled) -> Emitter<'a> {
        Emitter {
            program,
            body: String::new(),
            indent: 4,
            regs: BTreeSet::new(),
            preds: BTreeSet::new(),
            helpers: BTreeSet::new(),
            uses_carry: false,
            uses_stack: false,
            quad: None,
            quad_swap: None,
            loads: BTreeSet::new(),
            stores: BTreeSet::new(),
            centroid_loads: BTreeSet::new(),
            banks: BTreeSet::new(),
            textures: Vec::new(),
            globals: Vec::new(),
            temps: 0,
        }
    }

    /// Where a `ldg`'s address comes from, if this program builds it the way
    /// every compiler does: a 64-bit base in a constant bank, plus a
    /// per-invocation byte offset in a register.
    ///
    /// ```text
    /// iadd.cout r10, r7, c0[0x110]      // low half, carrying out
    /// iadd.cin  r11, RZ, c0[0x114]      // high half, carrying in
    /// ldg       r10, [r10]
    /// ```
    ///
    /// The base is a *descriptor*, not an address the shader computed, so the
    /// backend can read it out of the same bank it uploads and bind the
    /// memory it points at. Eden calls the same pass
    /// `global_memory_to_storage_buffer`.
    fn global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        let (mut lo, mut hi) = (None, None);
        for i in (0..at).rev() {
            match self.program.op(i) {
                // The low half: the index register plus the descriptor.
                Op::Iadd {
                    dst,
                    a,
                    b: Operand::Const { bank, offset },
                    aneg: false,
                    bneg: false,
                    cout: true,
                    ..
                } if dst == addr && lo.is_none() => lo = Some((bank, offset, a)),
                // The high half, which adds nothing but the carry.
                Op::Iadd {
                    dst,
                    a: RZ,
                    b: Operand::Const { bank, offset },
                    aneg: false,
                    bneg: false,
                    cin: true,
                    ..
                } if dst == addr.wrapping_add(1) && hi.is_none() => hi = Some((bank, offset)),
                // Anything else writing either half means the pattern does
                // not hold, and a guess at the address is a read of whatever
                // guest memory happens to be there.
                other => {
                    let writes = super::interp::writes(&other);
                    if writes.contains(&addr) && lo.is_none() {
                        return None;
                    }
                    if writes.contains(&addr.wrapping_add(1)) && hi.is_none() {
                        return None;
                    }
                }
            }
            if lo.is_some() && hi.is_some() {
                break;
            }
        }
        let ((bank, offset, index), (hi_bank, hi_offset)) = (lo?, hi?);
        // The two halves must be the two words of one descriptor.
        if hi_bank != bank || hi_offset != offset.wrapping_add(4) {
            return None;
        }
        Some((bank, offset, index))
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.body.push_str("  ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn need(&mut self, helper: &'static str) {
        self.helpers.insert(helper);
        if helper == "mulhi_s" {
            self.helpers.insert("mulhi_u");
        }
        if helper == "shf" {
            self.helpers.insert("shl32");
            self.helpers.insert("shr32");
        }
    }

    /// A `let` holding `value`, for when it is read more than once.
    fn bind(&mut self, value: &str) -> String {
        self.temps += 1;
        let name = format!("t{}", self.temps);
        self.line(&format!("let {name} = {value};"));
        name
    }

    // ---- operands ----

    fn r(&mut self, reg: u8) -> String {
        if reg == RZ {
            return "0u".to_string();
        }
        self.regs.insert(reg);
        format!("r{reg}")
    }

    fn f(&mut self, reg: u8) -> String {
        let value = self.r(reg);
        format!("bitcast<f32>({value})")
    }

    fn operand(&mut self, operand: Operand) -> String {
        match operand {
            Operand::Reg(reg) => self.r(reg),
            Operand::Imm(value) => format!("{value}u"),
            Operand::Const { bank, offset } => {
                self.banks.insert(bank);
                format!("cbRead({bank}u, {offset}u)")
            }
        }
    }

    fn operand_f(&mut self, operand: Operand) -> String {
        let value = self.operand(operand);
        format!("bitcast<f32>({value})")
    }

    fn p(&mut self, pred: u8) -> String {
        if pred >= 7 {
            return "true".to_string();
        }
        self.preds.insert(pred);
        format!("p{pred}")
    }

    /// Whether a guard or source predicate holds.
    fn holds(&mut self, pred: Pred) -> String {
        if pred.reg >= 7 {
            return if pred.negate { "false" } else { "true" }.to_string();
        }
        let name = self.p(pred.reg);
        if pred.negate {
            format!("!{name}")
        } else {
            name
        }
    }

    // ---- destinations ----

    /// Write a register. `RZ` discards, so nothing is emitted — every
    /// operation whose result is not its only effect binds the value first.
    fn set_r(&mut self, dst: u8, value: &str) {
        if dst == RZ {
            return;
        }
        self.regs.insert(dst);
        self.line(&format!("r{dst} = {value};"));
    }

    fn set_f(&mut self, dst: u8, value: &str) {
        self.set_r(dst, &format!("bitcast<u32>({value})"));
    }

    /// Write a predicate. `PT` and above are not writable, exactly as
    /// [`super::interp::Invocation`]'s `set_pred` ignores them.
    fn set_p(&mut self, dst: u8, value: &str) {
        if dst >= 7 {
            return;
        }
        self.preds.insert(dst);
        self.line(&format!("p{dst} = {value};"));
    }

    // ---- expression builders ----

    fn fmod(&mut self, modifier: FMod, value: String) -> String {
        let value = if modifier.abs {
            format!("abs({value})")
        } else {
            value
        };
        if modifier.neg {
            format!("-({value})")
        } else {
            value
        }
    }

    fn flush(&mut self, ftz: bool, value: String) -> String {
        if ftz {
            self.need("ftz");
            format!("ftz({value})")
        } else {
            value
        }
    }

    fn saturate(&mut self, sat: bool, value: String) -> String {
        if sat {
            self.need("fsat");
            format!("fsat({value})")
        } else {
            value
        }
    }

    // ---- half-precision ----
    //
    // A pair of halves is a `vec2<f32>` here and `pack2x16float` puts it back,
    // rounding to nearest with ties to even exactly as `f32_to_f16` does — so
    // the two backends agree on every finite result. They can differ on one
    // that overflows the half range, which WGSL leaves indeterminate and the
    // interpreter answers with an infinity.

    /// One source's two lanes, flushed and modified.
    fn half_source(&mut self, bits: String, m: FMod, sw: HSwizzle, ftz: bool) -> String {
        let lanes = match sw {
            HSwizzle::H1H0 => format!("unpack2x16float({bits})"),
            HSwizzle::H0H0 => format!("unpack2x16float({bits}).xx"),
            HSwizzle::H1H1 => format!("unpack2x16float({bits}).yy"),
            // Not a pair at all: one f32 that both lanes read.
            HSwizzle::F32 => format!("vec2<f32>(bitcast<f32>({bits}))"),
        };
        let lanes = if !ftz {
            lanes
        } else if sw == HSwizzle::F32 {
            self.need("ftz2");
            format!("ftz2({lanes})")
        } else {
            self.need("hftz");
            format!("hftz({lanes})")
        };
        self.fmod(m, lanes)
    }

    fn half_saturate(&mut self, sat: bool, value: String) -> String {
        if sat {
            self.need("fsat2");
            format!("fsat2({value})")
        } else {
            value
        }
    }

    /// The u32 a half op's two lanes leave in its destination.
    fn half_merge(&mut self, dst: u8, lanes: &str, merge: HMerge) -> String {
        match merge {
            HMerge::H1H0 => format!("pack2x16float({lanes})"),
            HMerge::F32 => format!("bitcast<u32>(({lanes}).x)"),
            HMerge::MrgH0 => {
                let kept = self.r(dst);
                format!("(({kept} & 0xffff0000u) | (pack2x16float({lanes}) & 0x0000ffffu))")
            }
            HMerge::MrgH1 => {
                let kept = self.r(dst);
                format!("(({kept} & 0x0000ffffu) | (pack2x16float({lanes}) & 0xffff0000u))")
            }
        }
    }

    /// `.fmz`'s "anything times zero is zero", as a lane-wise condition.
    /// Whether it applies at all is [`HPrecision::zeroes_products`].
    fn half_zeroed(&mut self, a: &str, b: &str) -> String {
        format!("(({a} == vec2<f32>(0.0)) | ({b} == vec2<f32>(0.0)))")
    }

    /// Each lane's comparison, combined with the source predicate — the half
    /// of `hset2` and `hsetp2` that is the same instruction.
    #[allow(clippy::too_many_arguments)]
    fn half_compare(
        &mut self,
        a: u8,
        am: FMod,
        asw: HSwizzle,
        b: Operand,
        bm: FMod,
        bsw: HSwizzle,
        cmp: FCmp,
        bop: BoolOp,
        src: Pred,
        ftz: bool,
    ) -> (String, String) {
        let x = self.r(a);
        let x = self.half_source(x, am, asw, ftz);
        let x = self.bind(&x);
        let y = self.operand(b);
        let y = self.half_source(y, bm, bsw, ftz);
        let y = self.bind(&y);
        let guard = self.holds(src);
        let guard = self.bind(&guard);
        let low = self.float_compare(cmp, &format!("{x}.x"), &format!("{y}.x"));
        let low = self.combine(bop, &low, &guard);
        let low = self.bind(&low);
        let high = self.float_compare(cmp, &format!("{x}.y"), &format!("{y}.y"));
        let high = self.combine(bop, &high, &guard);
        let high = self.bind(&high);
        (low, high)
    }

    fn ineg(&mut self, neg: bool, value: String) -> String {
        if neg {
            format!("(0u - ({value}))")
        } else {
            value
        }
    }

    fn inv(&mut self, invert: bool, value: String) -> String {
        if invert {
            format!("(~({value}))")
        } else {
            value
        }
    }

    fn float_compare(&mut self, cmp: FCmp, a: &str, b: &str) -> String {
        // WGSL has no `isNan`, and `x != x` is the form every backend
        // recognises for it.
        let unordered = format!("(({a}) != ({a}) || ({b}) != ({b}))");
        match cmp {
            FCmp::Never => "false".to_string(),
            FCmp::Lt => format!("(({a}) < ({b}))"),
            FCmp::Eq => format!("(({a}) == ({b}))"),
            FCmp::Le => format!("(({a}) <= ({b}))"),
            FCmp::Gt => format!("(({a}) > ({b}))"),
            FCmp::Ge => format!("(({a}) >= ({b}))"),
            FCmp::Ne => format!("(!{unordered} && ({a}) != ({b}))"),
            FCmp::Num => format!("(!{unordered})"),
            FCmp::Nan => unordered,
            FCmp::LtU => format!("({unordered} || ({a}) < ({b}))"),
            FCmp::EqU => format!("({unordered} || ({a}) == ({b}))"),
            FCmp::LeU => format!("({unordered} || ({a}) <= ({b}))"),
            FCmp::GtU => format!("({unordered} || ({a}) > ({b}))"),
            FCmp::GeU => format!("({unordered} || ({a}) >= ({b}))"),
            FCmp::NeU => format!("({unordered} || ({a}) != ({b}))"),
            FCmp::Always => "true".to_string(),
        }
    }

    fn int_compare(&mut self, cmp: ICmp, a: &str, b: &str, signed: bool) -> String {
        let (a, b) = if signed {
            (format!("bitcast<i32>({a})"), format!("bitcast<i32>({b})"))
        } else {
            (a.to_string(), b.to_string())
        };
        match cmp {
            ICmp::Never => "false".to_string(),
            ICmp::Lt => format!("(({a}) < ({b}))"),
            ICmp::Eq => format!("(({a}) == ({b}))"),
            ICmp::Le => format!("(({a}) <= ({b}))"),
            ICmp::Gt => format!("(({a}) > ({b}))"),
            ICmp::Ne => format!("(({a}) != ({b}))"),
            ICmp::Ge => format!("(({a}) >= ({b}))"),
            ICmp::Always => "true".to_string(),
        }
    }

    fn combine(&mut self, op: BoolOp, a: &str, b: &str) -> String {
        match op {
            BoolOp::And => format!("({a} && {b})"),
            BoolOp::Or => format!("({a} || {b})"),
            BoolOp::Xor => format!("({a} != {b})"),
        }
    }

    /// A `set`'s register result: all ones as a bit mask, or 1.0f with `.bf`.
    fn set_result(&mut self, taken: &str, bf: bool) -> String {
        let one = if bf { "0x3f800000u" } else { "0xffffffffu" };
        format!("select(0u, {one}, {taken})")
    }

    fn round(&mut self, mode: FRound, value: String) -> String {
        // WGSL's `round` breaks ties to even, which is the mode Maxwell's
        // `.rn` means.
        match mode {
            FRound::Nearest => format!("round({value})"),
            FRound::Floor => format!("floor({value})"),
            FRound::Ceil => format!("ceil({value})"),
            FRound::Trunc => format!("trunc({value})"),
        }
    }
}

impl Emitter<'_> {
    /// Everything that is not control flow.
    fn emit_alu(&mut self, at: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            // ---- attribute space ----
            Op::Ld {
                dst,
                offset,
                idx,
                size,
            } => {
                self.loads.extend(generic_slot(offset));
                let base = self.attr_base(offset, idx);
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    self.set_f(dst.wrapping_add(i), &format!("attrIn({base} + {word}u)"));
                }
            }
            Op::St {
                offset,
                idx,
                src,
                size,
            } => {
                self.stores.extend(generic_slot(offset));
                let base = self.attr_base(offset, idx);
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    let value = self.f(src.wrapping_add(i));
                    self.line(&format!("attrOut({base} + {word}u, {value});"));
                }
            }
            Op::Ipa {
                dst,
                offset,
                mul,
                perspective,
                sat,
                centroid,
            } => {
                self.loads.extend(generic_slot(offset));
                if centroid {
                    self.centroid_loads.extend(generic_slot(offset));
                }
                let mut value = format!("attrIn({offset}u)");
                if perspective {
                    if let Some(mul) = mul {
                        let factor = self.f(mul);
                        value = format!("({value} * {factor})");
                    }
                }
                let value = self.saturate(sat, value);
                self.set_f(dst, &value);
            }

            // ---- float ----
            Op::Fadd {
                dst,
                a,
                am,
                b,
                bm,
                ftz,
                sat,
            } => {
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = self.fmod(bm, y);
                let value = self.saturate(sat, format!("({x} + {y})"));
                self.set_f(dst, &value);
            }
            Op::Fmul {
                dst,
                a,
                b,
                bm,
                ftz,
                sat,
                scale,
            } => {
                // The pre-scale multiplies the first operand, before the
                // multiply proper.
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let factor = scale.factor();
                let x = if factor == 1.0 {
                    x
                } else {
                    format!("({x} * {factor:?})")
                };
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = self.fmod(bm, y);
                let value = self.saturate(sat, format!("({x} * {y})"));
                self.set_f(dst, &value);
            }
            Op::Ffma {
                dst,
                a,
                b,
                bneg,
                c,
                cneg,
                ftz,
                sat,
            } => {
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = if bneg { format!("-({y})") } else { y };
                let z = self.operand_f(c);
                let z = self.flush(ftz, z);
                let z = if cneg { format!("-({z})") } else { z };
                let value = self.saturate(sat, format!("fma({x}, {y}, {z})"));
                self.set_f(dst, &value);
            }
            Op::Fmnmx {
                dst,
                a,
                am,
                b,
                bm,
                pred,
                ftz,
            } => {
                let x = self.f(a);
                let x = self.flush(ftz, x);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.flush(ftz, y);
                let y = self.fmod(bm, y);
                // The predicate picks which end: true is the minimum, which
                // is why `fmnmx ... !pt` is a compiler's `max`. WGSL leaves
                // min/max with a NaN operand to the implementation, where the
                // interpreter answers with the operand that is not NaN.
                let take_min = self.holds(pred);
                let value = format!("select(max({x}, {y}), min({x}, {y}), {take_min})");
                self.set_f(dst, &value);
            }
            Op::Mufu {
                dst,
                src,
                sm,
                op: mufu,
                sat,
            } => {
                let x = self.f(src);
                let x = self.fmod(sm, x);
                let value = match mufu {
                    MufuOp::Cos => format!("cos({x})"),
                    MufuOp::Sin => format!("sin({x})"),
                    MufuOp::Ex2 => format!("exp2({x})"),
                    MufuOp::Lg2 => format!("log2({x})"),
                    MufuOp::Rcp => format!("(1.0 / {x})"),
                    MufuOp::Rsq => format!("(1.0 / sqrt({x}))"),
                    MufuOp::Sqrt => format!("sqrt({x})"),
                };
                let value = self.saturate(sat, value);
                self.set_f(dst, &value);
            }
            // ---- half-precision ----
            Op::Hadd2 {
                dst,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                merge,
                ftz,
                sat,
            } => {
                let x = self.r(a);
                let x = self.half_source(x, am, asw, ftz);
                let y = self.operand(b);
                let y = self.half_source(y, bm, bsw, ftz);
                let lanes = self.half_saturate(sat, format!("({x} + {y})"));
                let value = self.half_merge(dst, &lanes, merge);
                self.set_r(dst, &value);
            }
            Op::Hmul2 {
                dst,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                merge,
                prec,
                sat,
            } => {
                let ftz = prec == HPrecision::Ftz;
                let x = self.r(a);
                let x = self.half_source(x, am, asw, ftz);
                let y = self.operand(b);
                let y = self.half_source(y, bm, bsw, ftz);
                let lanes = if prec.zeroes_products(sat) {
                    // Bound, because `.fmz` reads both operands twice.
                    let x = self.bind(&x);
                    let y = self.bind(&y);
                    let zeroed = self.half_zeroed(&x, &y);
                    format!("select({x} * {y}, vec2<f32>(0.0), {zeroed})")
                } else {
                    format!("({x} * {y})")
                };
                let lanes = self.half_saturate(sat, lanes);
                let value = self.half_merge(dst, &lanes, merge);
                self.set_r(dst, &value);
            }
            Op::Hfma2 {
                dst,
                a,
                asw,
                b,
                bneg,
                bsw,
                c,
                cneg,
                csw,
                merge,
                prec,
                sat,
            } => {
                let ftz = prec == HPrecision::Ftz;
                let x = self.r(a);
                let x = self.half_source(x, FMod::NONE, asw, ftz);
                let y = self.operand(b);
                let y = self.half_source(
                    y,
                    FMod {
                        neg: bneg,
                        abs: false,
                    },
                    bsw,
                    ftz,
                );
                let z = self.operand(c);
                let z = self.half_source(
                    z,
                    FMod {
                        neg: cneg,
                        abs: false,
                    },
                    csw,
                    ftz,
                );
                let lanes = if prec.zeroes_products(sat) {
                    // A zeroed product leaves the addend, not zero.
                    let x = self.bind(&x);
                    let y = self.bind(&y);
                    let z = self.bind(&z);
                    let zeroed = self.half_zeroed(&x, &y);
                    format!("select(fma({x}, {y}, {z}), {z}, {zeroed})")
                } else {
                    format!("fma({x}, {y}, {z})")
                };
                let lanes = self.half_saturate(sat, lanes);
                let value = self.half_merge(dst, &lanes, merge);
                self.set_r(dst, &value);
            }
            Op::Hset2 {
                dst,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                cmp,
                bop,
                src,
                bf,
                ftz,
            } => {
                let (low, high) = self.half_compare(a, am, asw, b, bm, bsw, cmp, bop, src, ftz);
                // Each lane's answer fills its own half of the register:
                // 1.0h with `.bf`, all ones without.
                let taken = if bf { "0x3c00u" } else { "0x0000ffffu" };
                self.set_r(
                    dst,
                    &format!("(select(0u, {taken}, {low}) | select(0u, {taken} << 16u, {high}))"),
                );
            }
            Op::Hsetp2 {
                p0,
                p1,
                a,
                am,
                asw,
                b,
                bm,
                bsw,
                cmp,
                bop,
                src,
                and,
                ftz,
            } => {
                let (low, high) = self.half_compare(a, am, asw, b, bm, bsw, cmp, bop, src, ftz);
                if and {
                    let both = self.bind(&format!("({low} && {high})"));
                    self.set_p(p0, &both);
                    self.set_p(p1, &format!("!{both}"));
                } else {
                    self.set_p(p0, &low);
                    self.set_p(p1, &high);
                }
            }

            Op::Fsetp {
                p0,
                p1,
                a,
                am,
                b,
                bm,
                cmp,
                bop,
                src,
            } => {
                let x = self.f(a);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.fmod(bm, y);
                let taken = self.float_compare(cmp, &x, &y);
                let taken = self.bind(&taken);
                let guard = self.holds(src);
                let guard = self.bind(&guard);
                let set = self.combine(bop, &taken, &guard);
                self.set_p(p0, &set);
                let clear = self.combine(bop, &format!("!{taken}"), &guard);
                self.set_p(p1, &clear);
            }
            Op::Fset {
                dst,
                a,
                am,
                b,
                bm,
                cmp,
                bop,
                src,
                bf,
            } => {
                let x = self.f(a);
                let x = self.fmod(am, x);
                let y = self.operand_f(b);
                let y = self.fmod(bm, y);
                let taken = self.float_compare(cmp, &x, &y);
                let guard = self.holds(src);
                let taken = self.combine(bop, &taken, &guard);
                let value = self.set_result(&taken, bf);
                self.set_r(dst, &value);
            }

            // ---- integer ----
            Op::Iadd {
                dst,
                a,
                aneg,
                b,
                bneg,
                cin,
                cout,
            } => {
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let x = self.bind(&x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                // Two adds rather than one, because the carry out is whether
                // either of them wrapped and WGSL has no wider integer to see
                // it fall off the top of.
                let sum = self.bind(&format!("{x} + ({y})"));
                let carry_in = if cin {
                    self.uses_carry = true;
                    "select(0u, 1u, carry)".to_string()
                } else {
                    "0u".to_string()
                };
                let total = self.bind(&format!("{sum} + {carry_in}"));
                self.set_r(dst, &total);
                if cout {
                    self.uses_carry = true;
                    self.line(&format!("carry = ({sum} < {x}) || ({total} < {sum});"));
                }
            }
            Op::Iadd3 {
                dst,
                a,
                aneg,
                b,
                bneg,
                c,
                cneg,
            } => {
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                let z = self.operand(c);
                let z = self.ineg(cneg, z);
                self.set_r(dst, &format!("{x} + ({y}) + ({z})"));
            }
            Op::Iscadd {
                dst,
                a,
                aneg,
                b,
                bneg,
                shift,
            } => {
                let x = self.r(a);
                let x = self.ineg(aneg, x);
                let y = self.operand(b);
                let y = self.ineg(bneg, y);
                let shift = u32::from(shift) & 31;
                self.set_r(dst, &format!("(({x}) << {shift}u) + ({y})"));
            }
            Op::Imnmx {
                dst,
                a,
                b,
                pred,
                signed,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let take_min = self.holds(pred);
                let value = if signed {
                    format!(
                        "bitcast<u32>(select(max(bitcast<i32>({x}), bitcast<i32>({y})), \
                         min(bitcast<i32>({x}), bitcast<i32>({y})), {take_min}))"
                    )
                } else {
                    format!("select(max({x}, {y}), min({x}, {y}), {take_min})")
                };
                self.set_r(dst, &value);
            }
            Op::Imul {
                dst,
                a,
                b,
                signed,
                hi,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let value = match (hi, signed) {
                    (false, _) => format!("{x} * ({y})"),
                    (true, true) => {
                        self.need("mulhi_s");
                        format!("mulhi_s({x}, {y})")
                    }
                    (true, false) => {
                        self.need("mulhi_u");
                        format!("mulhi_u({x}, {y})")
                    }
                };
                self.set_r(dst, &value);
            }
            Op::Xmad {
                dst,
                a,
                ah,
                asigned,
                b,
                bh,
                bsigned,
                c,
                cmode,
                psl,
                mrg,
            } => {
                let x = self.r(a);
                let x = self.half(&x, ah, asigned);
                let raw_b = self.operand(b);
                let raw_b = self.bind(&raw_b);
                let y = self.half(&raw_b, bh, bsigned);
                let product = self.bind(&format!("({x}) * ({y})"));
                let product = if psl {
                    self.bind(&format!("{product} << 16u"))
                } else {
                    product
                };
                let raw_c = self.operand(c);
                let z = match cmode {
                    XmadC::Full => raw_c,
                    XmadC::Lo => format!("(({raw_c}) & 0xffffu)"),
                    XmadC::Hi => format!("(({raw_c}) >> 16u)"),
                    XmadC::Bcc => format!("(({raw_b} << 16u) + ({raw_c}))"),
                };
                let sum = self.bind(&format!("{product} + ({z})"));
                // `.mrg` replaces the result's high half with `b`'s low one
                // rather than adding anything there.
                let value = if mrg {
                    format!("({sum} & 0xffffu) | ({raw_b} << 16u)")
                } else {
                    sum
                };
                self.set_r(dst, &value);
            }
            Op::Isetp {
                p0,
                p1,
                a,
                b,
                cmp,
                signed,
                bop,
                src,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.int_compare(cmp, &x, &y, signed);
                let taken = self.bind(&taken);
                let guard = self.holds(src);
                let guard = self.bind(&guard);
                let set = self.combine(bop, &taken, &guard);
                self.set_p(p0, &set);
                let clear = self.combine(bop, &format!("!{taken}"), &guard);
                self.set_p(p1, &clear);
            }
            Op::Iset {
                dst,
                a,
                b,
                cmp,
                signed,
                bop,
                src,
                bf,
            } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.int_compare(cmp, &x, &y, signed);
                let guard = self.holds(src);
                let taken = self.combine(bop, &taken, &guard);
                let value = self.set_result(&taken, bf);
                self.set_r(dst, &value);
            }
            Op::Icmp {
                dst,
                a,
                b,
                c,
                cmp,
                signed,
            } => {
                // `icmp dst, a, b, c` is "dst = compare(c, 0) ? a : b".
                let selector = self.r(c);
                let taken = self.int_compare(cmp, &selector, "0u", signed);
                let x = self.r(a);
                let y = self.operand(b);
                self.set_r(dst, &format!("select({y}, {x}, {taken})"));
            }
            Op::Bfi {
                dst,
                insert,
                src,
                base,
            } => {
                self.need("bfi");
                let insert = self.r(insert);
                let src = self.operand(src);
                let base = self.operand(base);
                self.set_r(dst, &format!("bfi({insert}, {src}, {base})"));
            }
            Op::R2p { src, mask, byte } => {
                // One statement per predicate: they are separate variables,
                // so there is nothing to index.
                let bits = self.r(src);
                let shift = u32::from(byte) * 8;
                let bits = self.bind(&format!("{bits} >> {shift}u"));
                let mask = self.operand(mask);
                let mask = self.bind(&mask);
                for index in 0..7u8 {
                    let bit = 1u32 << index;
                    let value = format!("(({bits} & {bit}u) != 0u)");
                    self.line(&format!("if (({mask} & {bit}u) != 0u) {{"));
                    self.indent += 1;
                    self.set_p(index, &value);
                    self.indent -= 1;
                    self.line("}");
                }
            }
            Op::Lop {
                dst,
                a,
                ainv,
                b,
                binv,
                op: logic,
                pred,
            } => {
                let x = self.r(a);
                let x = self.inv(ainv, x);
                let y = self.operand(b);
                let y = self.inv(binv, y);
                let value = match logic {
                    LogicOp::And => format!("({x}) & ({y})"),
                    LogicOp::Or => format!("({x}) | ({y})"),
                    LogicOp::Xor => format!("({x}) ^ ({y})"),
                    LogicOp::PassB => y,
                };
                let value = self.bind(&value);
                self.set_r(dst, &value);
                if let Some((p, test)) = pred {
                    let bit = match test {
                        LopTest::True => "true".to_string(),
                        LopTest::Zero => format!("({value} == 0u)"),
                        LopTest::NonZero => format!("({value} != 0u)"),
                    };
                    self.set_p(p, &bit);
                }
            }
            Op::Lop3 { dst, a, b, c, lut } => {
                self.need("lop3");
                let x = self.r(a);
                let y = self.operand(b);
                let z = self.operand(c);
                self.set_r(dst, &format!("lop3({x}, {y}, {z}, {lut}u)"));
            }
            Op::Shl { dst, a, b, wrap } => {
                self.need("shl32");
                let x = self.r(a);
                let n = self.shift_count(b, wrap);
                self.set_r(dst, &format!("shl32({x}, {n})"));
            }
            Op::Shr {
                dst,
                a,
                b,
                signed,
                wrap,
            } => {
                let x = self.r(a);
                let n = self.shift_count(b, wrap);
                if signed {
                    self.need("sar32");
                    self.set_r(dst, &format!("sar32({x}, {n})"));
                } else {
                    self.need("shr32");
                    self.set_r(dst, &format!("shr32({x}, {n})"));
                }
            }
            Op::Shf {
                dst,
                lo,
                shift,
                hi,
                left,
                wrap,
                hi_out,
            } => {
                self.need("shf");
                let low = self.r(lo);
                let high = self.r(hi);
                let count = self.operand(shift);
                let count = if wrap {
                    format!("(({count}) & 63u)")
                } else {
                    count
                };
                self.set_r(
                    dst,
                    &format!("shf({low}, {high}, {count}, {left}, {hi_out})"),
                );
            }
            Op::Bfe { dst, a, b, signed } => {
                self.need("bfe");
                let x = self.r(a);
                let desc = self.operand(b);
                let desc = self.bind(&desc);
                self.set_r(
                    dst,
                    &format!("bfe({x}, {desc} & 0xffu, ({desc} >> 8u) & 0xffu, {signed})"),
                );
            }
            Op::Popc { dst, b, inv } => {
                let value = self.operand(b);
                let value = self.inv(inv, value);
                self.set_r(dst, &format!("countOneBits({value})"));
            }
            Op::Flo {
                dst,
                b,
                signed,
                shift,
                inv,
            } => {
                self.need("flo");
                let value = self.operand(b);
                let value = self.inv(inv, value);
                self.set_r(dst, &format!("flo({value}, {signed}, {shift})"));
            }
            Op::Sel { dst, a, b, pred } => {
                let x = self.r(a);
                let y = self.operand(b);
                let taken = self.holds(pred);
                self.set_r(dst, &format!("select({y}, {x}, {taken})"));
            }

            // ---- conversions ----
            Op::I2f {
                dst,
                src,
                sm,
                src_bytes,
                src_signed,
                sel,
            } => {
                let raw = self.operand(src);
                let raw = self.narrow(&raw, sel, src_bytes, src_signed);
                let value = if src_signed {
                    format!("f32(bitcast<i32>({raw}))")
                } else {
                    format!("f32({raw})")
                };
                let value = self.fmod(sm, value);
                self.set_f(dst, &value);
            }
            Op::F2i {
                dst,
                src,
                sm,
                dst_bytes,
                dst_signed,
                round,
                ftz,
            } => {
                let x = self.operand_f(src);
                let x = self.flush(ftz, x);
                let x = self.fmod(sm, x);
                let x = self.round(round, x);
                let value = if dst_signed {
                    self.need("f2i_s");
                    format!("f2i_s({x}, {dst_bytes}u)")
                } else {
                    self.need("f2i_u");
                    format!("f2i_u({x}, {dst_bytes}u)")
                };
                self.set_r(dst, &value);
            }
            Op::F2f {
                dst,
                src,
                sm,
                round,
                sat,
                ftz,
                src_bits,
                dst_bits,
                hi,
            } => {
                let x = if src_bits == 16 {
                    let raw = self.operand(src);
                    let lane = if hi { "y" } else { "x" };
                    format!("unpack2x16float({raw}).{lane}")
                } else {
                    self.operand_f(src)
                };
                let x = self.flush(ftz, x);
                let x = self.fmod(sm, x);
                let x = match round {
                    Some(round) => self.round(round, x),
                    None => x,
                };
                let value = self.saturate(sat, x);
                if dst_bits == 16 {
                    // `pack2x16float` rounds to nearest-even exactly as
                    // `f32_to_f16` does, so the two backends agree.
                    let packed = format!("pack2x16float(vec2<f32>({value}, 0.0))");
                    self.set_r(dst, &packed);
                } else {
                    self.set_f(dst, &value);
                }
            }
            Op::I2i {
                dst,
                src,
                sm,
                src_bytes,
                src_signed,
                dst_signed,
                sat,
                sel,
            } => {
                let raw = self.operand(src);
                let value = self.narrow(&raw, sel, src_bytes, src_signed);
                let value = if sm.neg {
                    format!("(0u - ({value}))")
                } else {
                    value
                };
                let value = if sm.abs {
                    let bound = self.bind(&value);
                    format!("select({bound}, 0u - {bound}, bitcast<i32>({bound}) < 0)")
                } else {
                    value
                };
                let value = if sat && !dst_signed {
                    let bound = self.bind(&value);
                    format!("select({bound}, 0u, bitcast<i32>({bound}) < 0)")
                } else {
                    value
                };
                self.set_r(dst, &value);
            }

            // ---- moves ----
            Op::Mov { dst, src } => {
                let value = self.operand(src);
                self.set_r(dst, &value);
            }
            Op::Mov32i { dst, imm } => self.set_r(dst, &format!("{imm}u")),
            Op::S2r { dst, .. } => {
                // Nothing here runs a warp or more than one invocation at a
                // time, so every lane and thread identity is zero — the same
                // answer the interpreter gives.
                self.set_r(dst, "0u");
            }
            Op::Psetp {
                p0,
                p1,
                a,
                b,
                c,
                op1,
                op2,
            } => {
                let x = self.holds(a);
                let y = self.holds(b);
                let first = self.combine(op1, &x, &y);
                let z = self.holds(c);
                let value = self.combine(op2, &first, &z);
                let value = self.bind(&value);
                self.set_p(p0, &value);
                self.set_p(p1, &format!("!{value}"));
            }

            // ---- memory ----
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                self.banks.insert(bank);
                let index = self.r(idx);
                let base = self.bind(&format!("{}u + {index}", offset as u32));
                for i in 0..size.regs() {
                    let word = i as u32 * 4;
                    self.set_r(
                        dst.wrapping_add(i),
                        &format!("cbRead({bank}u, ({base} + {word}u) & 0xffffu)"),
                    );
                }
            }

            // ---- texture ----
            Op::Texs {
                coords,
                dref,
                handle,
                dim,
                ..
            } => {
                let compare = dref.is_some();
                match self.textures.iter().find(|&&(imm, _, _)| imm == handle) {
                    // One binding cannot be both a colour image and a depth
                    // one — they are different WGSL types — so a program that
                    // reads the same immediate each way is the rasterizer's.
                    Some(&(_, _, was)) if was != compare => {
                        return Err(Unsupported::DepthCompare { at })
                    }
                    Some(_) => {}
                    None => self.textures.push((handle, dim, compare)),
                }
                let u = self.f(coords[0]);
                let v = self.f(coords[1]);
                // An array's layer is an integer in the low half of its
                // register, not a float like the coordinates beside it.
                let layer = match dim {
                    TexDim::T2dArray => {
                        let reg = self.r(coords[2]);
                        format!("({reg} & 0xffffu)")
                    }
                    _ => "0u".to_string(),
                };
                // A 3D image's third coordinate is normalized like the other
                // two, where an array's is the layer number — so they travel
                // in separate arguments rather than one that means both.
                let w = match dim {
                    // A cubemap's three coordinates are a direction, and a
                    // 3D image's third is normalized like the other two.
                    TexDim::T3d | TexDim::TCube => self.f(coords[2]),
                    _ => "0.0".to_string(),
                };
                let code = tex_dim_code(dim);
                let color = match dref {
                    // A shadow sample answers with one value, and every
                    // channel a `texs` asks for gets it except alpha — which
                    // is what `sample_compare_with` returns too, so the
                    // destinations below are stored the same way either way.
                    Some(reg) => {
                        let reference = self.f(reg);
                        self.bind(&format!(
                            "texSampleCompare({handle}u, {code}u, {u}, {v}, {layer}, {reference})"
                        ))
                    }
                    None => self.bind(&format!(
                        "texSample({handle}u, {code}u, {u}, {v}, {layer}, {w})"
                    )),
                };
                // The interpreter lands these results *late*, at the first
                // instruction that reads the destination, because that is
                // where hardware's scoreboard would have waited. Writing them
                // now is equivalent wherever that deferral was built to
                // matter: `first_use_after` finds the first read, so nothing
                // between here and there reads or writes the register. Where
                // the two differ is a destination overwritten before any read
                // — the interpreter still lands the sample afterwards, and
                // hardware does not.
                let writes = self.program.texs_writes(at).to_vec();
                for (reg, store, _) in writes {
                    const COMPONENT: [&str; 4] = ["x", "y", "z", "w"];
                    match store {
                        TexsStore::Float(channel) => {
                            self.set_f(reg, &format!("{color}.{}", COMPONENT[channel]));
                        }
                        // The `.F16` form packs two channels into the register
                        // as halves, which is what the `h*2` ops that read it
                        // back are expecting to unpack.
                        TexsStore::Halves(low, high) => {
                            let half = |c: Option<usize>| match c {
                                Some(channel) => format!("{color}.{}", COMPONENT[channel]),
                                None => "0.0".to_string(),
                            };
                            self.set_r(
                                reg,
                                &format!(
                                    "pack2x16float(vec2<f32>({}, {}))",
                                    half(Some(low)),
                                    half(high)
                                ),
                            );
                        }
                    }
                }
            }

            // `shfl` reads the value of another lane of the 2x2 quad, which
            // is the whole warp the rasterizer models — so `quadSwapX`/`Y`/
            // `Diagonal` are that instruction exactly, and the lane it names
            // picks between them. `interp::shuffle_source` is the arithmetic
            // mirrored here. Where the device has no quad operations, the
            // module defines the three out of derivatives; see [`QUAD_SWAP`].
            Op::Shfl {
                dst,
                pred,
                src,
                index,
                mask,
                mode,
            } => {
                self.quad.get_or_insert(at);
                self.quad_swap.get_or_insert(at);
                let value = self.r(src);
                let here = self.bind(&value);
                let x = self.bind(&format!("quadSwapX({here})"));
                let y = self.bind(&format!("quadSwapY({here})"));
                let d = self.bind(&format!("quadSwapDiagonal({here})"));
                let lane = self.bind("i32(quadLane())");
                let index = self.operand(index);
                let index = self.bind(&format!("i32({index})"));
                let mask = self.operand(mask);
                let mask = self.bind(&format!("i32({mask})"));
                let clamp = self.bind(&format!("({mask} & 31)"));
                let segment = self.bind(&format!("(({mask} >> 8) & 31)"));
                let floor = self.bind(&format!("({lane} & {segment})"));
                let ceiling = self.bind(&format!("({floor} | ({clamp} & ~{segment}))"));
                let from = match mode {
                    ShflMode::Idx => format!("(({index} & ~{segment}) | {floor})"),
                    ShflMode::Up => format!("({lane} - {index})"),
                    ShflMode::Down => format!("({lane} + {index})"),
                    ShflMode::Bfly => format!("({lane} ^ {index})"),
                };
                let from = self.bind(&from);
                // `up` is the one mode whose bound holds from below.
                let within = match mode {
                    ShflMode::Up => format!("({from} >= {ceiling})"),
                    _ => format!("({from} <= {ceiling})"),
                };
                let ok = self.bind(&format!("({within} && {from} >= 0)"));
                let sel = self.bind(&format!("u32(({from} ^ {lane}) & 3)"));
                let peer = format!(
                    "select(select(select({here}, {x}, {sel} == 1u), {y}, {sel} == 2u), {d}, {sel} == 3u)"
                );
                // A lane outside the quad has no value to read, and the
                // rasterizer leaves this one's own there.
                let reachable = format!("({ok} && {from} < 4)");
                self.set_r(dst, &format!("select({here}, {peer}, {reachable})"));
                self.set_p(pred, &ok);
            }

            // `fswzadd` is not a cross-lane read at all: each lane of the
            // quad combines its *own* two operands with the pair of signs its
            // two bits of the swizzle name. Only which lane this is comes
            // from outside — which the module answers from `position`, so
            // this needs nothing of the device.
            Op::Fswzadd {
                dst,
                a,
                b,
                swizzle,
                ftz,
            } => {
                self.quad.get_or_insert(at);
                let x = self.r(a);
                let x = self.flush(ftz, format!("bitcast<f32>({x})"));
                let x = self.bind(&x);
                let y = self.r(b);
                let y = self.flush(ftz, format!("bitcast<f32>({y})"));
                let y = self.bind(&y);
                let code = self.bind(&format!(
                    "((({swizzle}u) >> ((quadLane() & 3u) * 2u)) & 3u)"
                ));
                // `FSWZ_SIGNS` in `super::interp`, arm for arm.
                let ka = self.bind(&format!(
                    "select(select(-1.0, 1.0, {code} == 1u), 0.0, {code} == 3u)"
                ));
                let kb = self.bind(&format!("select(-1.0, 1.0, {code} == 2u)"));
                self.set_f(dst, &format!("{ka} * {x} + {kb} * {y}"));
            }

            Op::Nop | Op::Inert => {}

            // Global, local and shared memory need storage buffers, which is
            // a question about resource binding rather than translation. A
            // barrier has no meaning in the graphics stages this translates.
            //
            // A global load whose address is a descriptor in a constant bank
            // plus an index: the backend binds the memory that descriptor
            // points at, and the read becomes an indexed one of that buffer.
            // Any other address is guest memory this cannot reach — see
            // `Unsupported::Op`.
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let Some((bank, at, index)) = self.global_base(at, addr) else {
                    return Err(Unsupported::Op { at, op });
                };
                let slot = match self.globals.iter().position(|g| *g == (bank, at)) {
                    Some(slot) => slot,
                    None => {
                        self.globals.push((bank, at));
                        self.globals.len() - 1
                    }
                };
                let index = self.r(index);
                let base = self.bind(&format!("({index} + {offset}u)"));
                for word in 0..size.regs() {
                    let byte = u32::from(word) * 4;
                    self.set_r(
                        dst.wrapping_add(word),
                        &format!("gRead({slot}u, {base} + {byte}u)"),
                    );
                }
            }

            Op::Stg { .. }
            | Op::Ldl { .. }
            | Op::Stl { .. }
            | Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Atom { .. }
            | Op::Bar { .. }
            | Op::Unimplemented { .. } => return Err(Unsupported::Op { at, op }),

            // Handled by `emit_terminator`, and `ssy`/`pbk`/`pcnt` by
            // `emit_instruction` before it gets here.
            Op::Bra { .. }
            | Op::Brx { .. }
            | Op::Ssy { .. }
            | Op::Pbk { .. }
            | Op::Pcnt { .. }
            | Op::Sync
            | Op::Brk
            | Op::Cont
            | Op::Exit
            | Op::Kil => unreachable!("control flow is emitted by emit_instruction"),
            // The general `tex` needs the operands `texs` has no room for —
            // an explicit level, a texel offset — and each of those is a
            // separate WGSL builtin. Until they are emitted, a shader with
            // one is the rasterizer's.
            Op::Tex { .. } => return Err(Unsupported::Op { at, op }),
        }
        Ok(())
    }

    /// `a[offset + Rn]`'s byte address. The index register contributes a
    /// 16-bit byte offset, and the sum wraps within that width.
    fn attr_base(&mut self, offset: u16, idx: u8) -> String {
        let index = self.r(idx);
        self.bind(&format!("({offset}u + ({index} & 0xffffu)) & 0xffffu"))
    }

    /// One 16-bit half of a register, as `xmad` reads it.
    fn half(&mut self, value: &str, high: bool, signed: bool) -> String {
        let half = if high {
            format!("(({value}) >> 16u)")
        } else {
            format!("(({value}) & 0xffffu)")
        };
        if signed {
            self.need("sext");
            format!("sext({half}, 2u)")
        } else {
            half
        }
    }

    /// A shift instruction's count, masked when the encoding says to wrap.
    fn shift_count(&mut self, operand: Operand, wrap: bool) -> String {
        let count = self.operand(operand);
        if wrap {
            format!("(({count}) & 31u)")
        } else {
            count
        }
    }

    /// A conversion's source: the selected byte lane, narrowed to the source
    /// width and sign- or zero-extended back to 32 bits.
    fn narrow(&mut self, raw: &str, sel: u8, bytes: u8, signed: bool) -> String {
        let shift = u32::from(sel) * 8;
        let shifted = if shift == 0 {
            raw.to_string()
        } else {
            format!("(({raw}) >> {shift}u)")
        };
        if signed {
            self.need("sext");
            format!("sext({shifted}, {bytes}u)")
        } else {
            self.need("truncw");
            format!("truncw({shifted}, {bytes}u)")
        }
    }
}

impl Emitter<'_> {
    /// One `case` per basic block, in order.
    fn emit_blocks(&mut self, leaders: &[usize]) -> Result<(), Unsupported> {
        for (n, &start) in leaders.iter().enumerate() {
            let end = leaders.get(n + 1).copied().unwrap_or(self.program.len());
            self.indent = 3;
            self.line(&format!("case {start}u: {{"));
            self.indent = 4;
            for at in start..end {
                self.emit_instruction(at, end)?;
            }
            // A block that does not end in a terminator falls into the next.
            if !is_terminator(self.program.op(end - 1)) {
                self.line(&format!("pc = {end}u;"));
            }
            self.indent = 3;
            self.line("}");
        }
        Ok(())
    }

    fn emit_instruction(&mut self, at: usize, fallthrough: usize) -> Result<(), Unsupported> {
        let op = self.program.op(at);
        let guard = self.program.pred(at);
        // A push falls through, so it is an ordinary statement. The decoder
        // gives these `PT`: the bits every other instruction keeps its guard
        // in are part of their target.
        if matches!(op, Op::Ssy { .. } | Op::Pbk { .. } | Op::Pcnt { .. }) {
            let target = self.program.target(at);
            if target == NO_TARGET {
                return Err(Unsupported::UndecodedTarget { at });
            }
            self.uses_stack = true;
            self.line(&format!("stack[sp] = {target}u;"));
            self.line("sp = sp + 1;");
            return Ok(());
        }
        if is_terminator(op) {
            return self.emit_terminator(at, op, guard, fallthrough);
        }
        if guard.is_always() {
            return self.emit_alu(at, op);
        }
        let cond = self.holds(guard);
        self.line(&format!("if ({cond}) {{"));
        self.indent += 1;
        let result = self.emit_alu(at, op);
        self.indent -= 1;
        self.line("}");
        result
    }

    /// A guarded terminator has to say where control goes when the guard does
    /// not hold, because falling out of the `case` would re-enter the block.
    fn emit_terminator(
        &mut self,
        at: usize,
        op: Op,
        guard: Pred,
        fallthrough: usize,
    ) -> Result<(), Unsupported> {
        if guard.is_always() {
            return self.emit_jump(at, op);
        }
        let cond = self.holds(guard);
        self.line(&format!("if ({cond}) {{"));
        self.indent += 1;
        let result = self.emit_jump(at, op);
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line(&format!("pc = {fallthrough}u;"));
        self.indent -= 1;
        self.line("}");
        result
    }

    fn emit_jump(&mut self, at: usize, op: Op) -> Result<(), Unsupported> {
        match op {
            Op::Bra { .. } => {
                let target = self.program.target(at);
                if target == NO_TARGET {
                    return Err(Unsupported::UndecodedTarget { at });
                }
                self.line(&format!("pc = {target}u;"));
            }
            Op::Exit => self.line("return false;"),
            Op::Kil => self.line("return true;"),
            Op::Sync | Op::Brk | Op::Cont => {
                self.uses_stack = true;
                self.line("sp = sp - 1;");
                self.line("pc = stack[sp];");
            }
            Op::Brx { base, reg } => {
                let targets: Vec<u32> = match self.program.indirect_targets(at) {
                    Some(targets) => targets.to_vec(),
                    None => return Err(Unsupported::IndirectBranch { at }),
                };
                let selector = self.r(reg);
                let raw = self.bind(&format!("{base}u + {selector}"));
                // A target landing on the `sched` word that starts a 32-byte
                // block means that block's first real instruction, which is
                // what `align_slot` resolves it to.
                let slot = self.bind(&format!("select({raw}, {raw} + 8u, ({raw} & 31u) == 0u)"));
                self.line(&format!("switch ({slot}) {{"));
                self.indent += 1;
                for target in targets {
                    let offset = self.program.offset(target as usize);
                    self.line(&format!("case {offset}u: {{ pc = {target}u; }}"));
                }
                // An arm outside the table is an address the decoder never
                // saw. WGSL has no way to raise the error the interpreter
                // raises there, so the invocation ends instead.
                self.line("default: { return false; }");
                self.indent -= 1;
                self.line("}");
            }
            _ => unreachable!("emit_jump called with {op:?}"),
        }
        Ok(())
    }

    /// The helpers, the declarations and the dispatch loop around the blocks.
    fn finish(&self, leaders: &[usize]) -> String {
        let mut out = format!(
            "// {} instructions in {} blocks\n\n",
            self.program.len(),
            leaders.len()
        );
        for (name, source) in HELPERS {
            if self.helpers.contains(name) {
                out.push_str(source);
                out.push_str("\n\n");
            }
        }
        // Registers outlive the call; everything else is invocation state
        // the caller has no business reading.
        for reg in &self.regs {
            out.push_str(&format!("var<private> r{reg}: u32 = 0u;\n"));
        }
        if !self.regs.is_empty() {
            out.push('\n');
        }
        out.push_str("fn run() -> bool {\n");
        for pred in &self.preds {
            out.push_str(&format!("  var p{pred}: bool = false;\n"));
        }
        if self.uses_carry {
            out.push_str("  var carry: bool = false;\n");
        }
        if self.uses_stack {
            out.push_str(&format!(
                "  var stack: array<u32, {RECONVERGENCE_DEPTH}>;\n"
            ));
            out.push_str("  var sp: i32 = 0;\n");
        }
        out.push_str("  var pc: u32 = 0u;\n");
        out.push_str("  loop {\n");
        out.push_str("    switch (pc) {\n");
        out.push_str(&self.body);
        // Reached by a fall-through past the last block, which is a program
        // with no `exit` — the decoder rejects those, so this is the arm WGSL
        // requires rather than one control gets to.
        out.push_str("      default: { return false; }\n");
        out.push_str("    }\n");
        out.push_str("  }\n");
        // Unreachable: the loop has no `break`, and every path out of the
        // switch either assigns `pc` or returns. WGSL still wants a function
        // with a return type to return at the end of its body, and a
        // validator will not take the loop's word for it.
        out.push_str("  return false;\n");
        out.push_str("}\n");
        out
    }
}

/// Which pipeline stage a module is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
}

/// Everything a complete module has to be wired to, as slot and bank numbers.
///
/// All of it can be read off the program — see [`Layout::of`] — because
/// Maxwell's attribute space *is* the interface: a vertex shader's inputs are
/// the `a[]` offsets it loads from, its outputs are the ones it stores to,
/// and a fragment shader's inputs are the ones it interpolates. What cannot
/// be read off the program is how those map to memory, which is a question
/// about the draw rather than about the shader.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// Generic slots a vertex shader loads its inputs from. The slot number
    /// is the `@location`, and the backend feeds each one four floats —
    /// whatever the vertex format was, `raster::fetch_attribute` has already
    /// widened it to that.
    pub attributes: Vec<usize>,
    /// The slots of [`Layout::attributes`] whose vertex format carries
    /// integers instead, and which of the two.
    ///
    /// WebGPU makes a shader input's base type part of the match, so an
    /// integer attribute has to be declared `vec4<i32>`/`vec4<u32>` and
    /// bitcast into `a[]` — which is the bit pattern `fetch_attribute`
    /// leaves there for one as well, so the two stages still agree. Nothing
    /// the program says: the backend fills it in from the draw, the same way
    /// it does [`Layout::flip_y`].
    pub integer_attributes: Vec<(usize, AttributeBase)>,
    /// Generic slots passed from the vertex stage to the fragment stage.
    /// Both stages have to name the same ones, so a pair of modules is built
    /// from one layout rather than two.
    pub varyings: Vec<usize>,
    /// The subset of [`Layout::varyings`] the fragment shader reads with
    /// `ipa.centroid`, which both stages must qualify the same way — an
    /// `@interpolate` that differs between them is a pipeline that will not
    /// build. Only the fragment program says so; the vertex stage is told.
    pub centroid_varyings: Vec<usize>,
    /// Constant banks the program reads, by bind slot.
    pub const_banks: Vec<u8>,
    /// The textures the module binds.
    pub textures: Vec<TextureBinding>,
    /// The `(bank, offset)` of each `ldg` descriptor, in binding order.
    pub globals: Vec<(u8, u16)>,
    /// How many colour targets a fragment shader writes. Each takes four
    /// consecutive registers from `r0`, so target `n` is `r[4n..4n+4]`.
    ///
    /// Zero is a depth-only pass, which is a real thing a title does rather
    /// than a gap — Just Dance 2017 renders every pass that way. The entry
    /// point then returns nothing at all, because a fragment shader that
    /// names `@location(0)` with no colour attachment behind it is a pipeline
    /// that will not build. It still runs: `kil` and alpha-to-coverage are
    /// both reasons a depth-only fragment is shaded.
    pub targets: u32,
    /// Which bind group this module's bindings live in.
    ///
    /// The two stages must not share one. They index *different* constant
    /// buffers with the same bank number and different textures with the
    /// same `texs` immediate, so a single group would have two bindings
    /// claiming the same slot. One group per stage is the only arrangement
    /// where a binding number means one thing.
    pub group: u32,
    /// Whether the vertex entry point negates `position.y`.
    ///
    /// **Not the same as the viewport transform mirroring y.** WebGPU's own
    /// NDC has y pointing up and its framebuffer has y pointing down, so it
    /// mirrors already: `ndc_y = +1` is row 0 whatever the viewport says,
    /// and its height cannot be negative to say otherwise.
    ///
    /// Maxwell mirrors only when the guest wrote a negative `scale_y` —
    /// which a driver does for the window, to reconcile GL's bottom-left
    /// origin with a target whose row 0 is at the top. So the two agree
    /// exactly when the guest's transform *does* mirror, and the shader has
    /// to negate when it does **not**, which is the offscreen case.
    ///
    /// Setting this from `pipeline::Viewport::flip_y` directly flips twice.
    /// It looks almost right, because a full-screen quad is symmetric about
    /// the centre and so is most of a UI — the Home Menu came out 94.87%
    /// correct that way, with one off-centre band mirrored onto the other
    /// side of the screen.
    pub flip_y: bool,
    /// Whether the vertex entry point remaps `position.z` from `-w..w` onto
    /// `0..w`.
    ///
    /// Maxwell clips z the way GL does and WebGPU clips it the way Vulkan
    /// does, so a shader whose z is left alone has everything in the near
    /// half of the frustum clipped away entirely. Read it off the viewport
    /// transform's z scale, which is 0.5 for a guest using GL's range.
    pub depth_minus_one_to_one: bool,
    /// Attribute slots whose fetch swaps the first and third components.
    ///
    /// WebGPU has no BGRA vertex format, so the swap happens in the entry
    /// point — which is where `raster::fetch_attribute` does it too. Nothing
    /// the program says: the draw's registers say it, and the backend fills
    /// it in the way it does [`Layout::flip_y`].
    pub bgra_attributes: Vec<usize>,
    /// What a fragment has to work out for itself when a backend is
    /// rendering a multisampled surface one texel at a time, rather than
    /// through a device's own multisampling.
    pub coverage: Option<Coverage>,
}

/// Per-sample coverage, for a backend rendering an expanded multisample
/// surface at texel resolution.
///
/// A Maxwell multisample surface stores its samples spatially — a pixel owns
/// a `samples_x` by `samples_y` tile of texels — so a backend can render it
/// by treating every texel as its own fragment. What that arrangement does
/// *not* get for free is the two things a device's multisample state would
/// have done: the sample mask, and alpha-to-coverage. Both are a question
/// about which sample this fragment is, and the answer is in its position.
///
/// Nothing here is anything the program says. It is the draw's
/// [`crate::gpu::surface::SampleGrid`] and its multisample registers, filled
/// in by the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    pub samples_x: u32,
    pub samples_y: u32,
    /// Which sample each texel of a pixel's tile holds, indexed by
    /// `dy * samples_x + dx`. The inverse of
    /// [`crate::gpu::surface::SampleGrid::texel`], which is what a fragment
    /// needs: it knows where it is, and has to work out which sample that
    /// makes it.
    pub sample_of_slot: Vec<u32>,
    /// Which samples the draw may write, as a bit per sample.
    pub sample_mask: u32,
    /// Whether the fragment's alpha narrows that mask further.
    pub alpha_to_coverage: bool,
}

/// One texture a module samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureBinding {
    /// The `texs` immediate that names it.
    pub immediate: u16,
    /// What the instruction samples it as, which decides the binding's type.
    pub dim: TexDim,
    /// How the descriptor says the channels are rearranged on the way out.
    ///
    /// Not something the program says — it is in the TIC, which is guest
    /// memory the draw points at — so [`Layout::of`] leaves it as the
    /// identity and a backend fills it in from
    /// [`crate::gpu::upload::TextureUpload::swizzle`]. WebGPU has no
    /// per-texture component swizzle, so it happens in the sampling hook,
    /// where it costs a shuffle rather than a copy.
    ///
    /// It is not decoration: two thirds of the Home Menu's draws sample a
    /// single-channel image swizzled to `[R, R, R, One]`, and reading it as
    /// the identity gets every one of them wrong.
    pub swizzle: [SwizzleSource; 4],
    /// Whether this is a shadow map: bound as a `texture_depth_*` beside a
    /// `sampler_comparison` and read with `textureSampleCompareLevel`.
    ///
    /// A depth image cannot be uploaded — WebGPU allows a copy into a
    /// `depth32float` only from another texture of the same format
    /// (§26.1.2.2) — so a backend gets one there by *drawing* into it, which
    /// is the workaround the specification itself names.
    pub compare: bool,
}

/// The swizzle that rearranges nothing.
pub const IDENTITY_SWIZZLE: [SwizzleSource; 4] = [
    SwizzleSource::R,
    SwizzleSource::G,
    SwizzleSource::B,
    SwizzleSource::A,
];

/// `a[]`'s word count: a ten-bit byte address, one `f32` per word.
const ATTRIBUTE_WORDS: usize = 0x400 / 4;
/// The `a[]` byte offset of generic slot `n`, and of its component `c`,
/// is `GENERIC_BASE + n * GENERIC_STRIDE + c * 4`.
const GENERIC_BASE: usize = 0x80;
const GENERIC_STRIDE: usize = 0x10;
const GENERIC_SLOTS: usize = 32;
/// Clip position, four floats. Its `w` slot is also where a fragment shader
/// reads `1/w`, which is not a collision: one is a vertex output and the
/// other a fragment input.
const POSITION: usize = 0x70;
/// `InstanceId` and `VertexId`, in that order — the instance is the lower.
const INSTANCE_ID: usize = 0x2f8;
const VERTEX_ID: usize = 0x2fc;

/// Constant bank `b` binds at `b`, and texture `i` at `TEXTURE_BINDING + 2i`
/// with its sampler beside it. Fixed rather than packed so that a backend can
/// work out a binding number without re-deriving the layout.
const TEXTURE_BINDING: u32 = 32;

/// Where a module's `ldg` buffers start binding, past the textures.
pub const GLOBAL_BINDING: u32 = 96;

impl Layout {
    /// Read a program's interface off what translating it touched.
    ///
    /// Derived from the [`Translation`] rather than from a second walk over
    /// the opcodes, for the same reason the register list is: a pass that
    /// re-derives what the emitter already knows is a second place for the
    /// two to disagree, and here they would disagree by leaving out a
    /// binding the emitted text calls.
    pub fn of(translated: &Translation, stage: Stage) -> Layout {
        let (attributes, varyings) = match stage {
            // A vertex shader loads its inputs and stores its outputs.
            Stage::Vertex => (translated.loads.clone(), translated.stores.clone()),
            // A fragment shader's inputs are what it interpolates; anything
            // it stores to `a[]` goes nowhere, since its output is `r0` on.
            Stage::Fragment => (Vec::new(), translated.loads.clone()),
        };
        Layout {
            attributes,
            integer_attributes: Vec::new(),
            varyings,
            // A vertex shader has no `ipa` to say it with; the backend copies
            // the fragment stage's answer over before building the pair.
            centroid_varyings: match stage {
                Stage::Vertex => Vec::new(),
                Stage::Fragment => translated.centroid_loads.clone(),
            },
            const_banks: translated.const_banks.clone(),
            textures: translated
                .textures
                .iter()
                .map(|&(immediate, dim, compare)| TextureBinding {
                    immediate,
                    dim,
                    compare,
                    swizzle: IDENTITY_SWIZZLE,
                })
                .collect(),
            globals: translated.globals.clone(),
            targets: 1,
            group: 0,
            // Neither is anything the program says; both come from the
            // draw's viewport transform.
            flip_y: false,
            depth_minus_one_to_one: false,
            bgra_attributes: Vec::new(),
            coverage: None,
        }
    }

    /// The sampling half of a varying's `@interpolate`, as the text that
    /// follows `linear` — `", centroid"` or nothing.
    ///
    /// WGSL's default is `center`, and in a single-sampled pass the two are
    /// the same point. It is still said, because the expanded-multisample
    /// route is not the only one the backend has: with `GPU_DEVICE_MSAA` the
    /// pass is genuinely multisampled, and there centroid is what keeps a
    /// varying from being extrapolated outside the primitive at a partly
    /// covered edge.
    fn sampling(&self, slot: usize) -> &'static str {
        if self.centroid_varyings.contains(&slot) {
            ", centroid"
        } else {
            ""
        }
    }

    /// What slot `slot` arrives as. Float unless the draw said otherwise:
    /// only the integer formats are recorded, since they are the only ones
    /// that change how the input is declared.
    pub fn attribute_base(&self, slot: usize) -> AttributeBase {
        self.integer_attributes
            .iter()
            .find(|&&(at, _)| at == slot)
            .map_or(AttributeBase::Float, |&(_, base)| base)
    }
}

/// The WGSL scalar an attribute of this base type is read as.
fn attribute_scalar(base: AttributeBase) -> &'static str {
    match base {
        AttributeBase::Float => "f32",
        AttributeBase::Sint => "i32",
        AttributeBase::Uint => "u32",
    }
}

/// The generic slot an `a[]` byte offset names, if it is one.
fn generic_slot(offset: u16) -> Option<usize> {
    let offset = usize::from(offset);
    if (GENERIC_BASE..GENERIC_BASE + GENERIC_SLOTS * GENERIC_STRIDE).contains(&offset) {
        Some((offset - GENERIC_BASE) / GENERIC_STRIDE)
    } else {
        None
    }
}

/// Wrap a translation in everything it needs to be a shader module: the
/// bindings it reads through, the attribute space it reads and writes, and
/// the entry point that fills one in and takes the other out.
///
/// # Why the varyings are `@interpolate(linear)`
///
/// This is the one place where doing the obvious thing is silently wrong.
/// Maxwell's `ipa` does not receive a perspective-correct value: it receives
/// `value/w`, linearly interpolated in screen space, and the shader finishes
/// the job itself by multiplying by `rcp(a[0x7c])`. Declaring the varyings
/// `perspective` — which is WGSL's default — would have the hardware divide
/// as well, and every textured surface would be wrong in a way that looks
/// like a texture-coordinate bug.
///
/// So the vertex stage multiplies each varying by `1/w` on the way out and
/// the fragment stage interpolates linearly, which reproduces exactly what
/// `raster::shade_fragment` puts in `attr_in`. `a[0x7c]` is then WGSL's own
/// `position.w`, which in a fragment shader is already `1/w` interpolated the
/// same way.
/// `quadSwapX`/`Y`/`Diagonal` for a fragment shader on a device that has no
/// quad operations of its own, which in a browser is every device: wgpu's web
/// backend can neither request WebGPU's `subgroups` nor report it.
///
/// A fine derivative *is* the difference between the two lanes of a pair, so
/// the neighbour is this lane's value plus or minus it depending on which of
/// the two this is. Adding an `f32` difference back to an `f32` does not in
/// general land on the other lane's bits — but it does when both are
/// integers small enough that an `f32` holds them without rounding, and a
/// 32-bit register split into halves is two such integers. The difference of
/// two values in `0..=65535` is exact, and so is adding it back, so this
/// recovers the neighbour's word exactly rather than approximately.
///
/// The clamp is for the derivative the language promises rather than the one
/// the hardware computes: `dpdxFine` is specified as an approximation, and a
/// half that came back a fraction off would otherwise truncate to the wrong
/// integer or convert out of range.
const QUAD_SWAP: &str = "\
fn quadSwapHalves(low: f32, high: f32, d_low: f32, d_high: f32, sign: f32) -> u32 {
  let other_low = clamp(round(low + sign * d_low), 0.0, 65535.0);
  let other_high = clamp(round(high + sign * d_high), 0.0, 65535.0);
  return (u32(other_high) << 16u) | u32(other_low);
}

fn quadSwapX(v: u32) -> u32 {
  // `quad_lane`'s low bit is the column parity: the left lane of the pair
  // adds the difference, the right one subtracts it.
  let low = f32(v & 0xffffu);
  let high = f32(v >> 16u);
  let sign = select(1.0, -1.0, (quad_lane & 1u) == 1u);
  return quadSwapHalves(low, high, dpdxFine(low), dpdxFine(high), sign);
}

fn quadSwapY(v: u32) -> u32 {
  // And its second bit is the row parity, which `dpdyFine` runs down just as
  // `position.y` does.
  let low = f32(v & 0xffffu);
  let high = f32(v >> 16u);
  let sign = select(1.0, -1.0, (quad_lane & 2u) == 2u);
  return quadSwapHalves(low, high, dpdyFine(low), dpdyFine(high), sign);
}

fn quadSwapDiagonal(v: u32) -> u32 {
  // The lane across the quad is the one across the row from the one across
  // the column, and the first swap's answer is an exact word to take the
  // second derivative of.
  return quadSwapY(quadSwapX(v));
}";

pub fn module(
    translated: &Translation,
    stage: Stage,
    layout: &Layout,
) -> Result<String, Unsupported> {
    let mut out = String::new();
    // Only the fragment entry point knows which lane it is, so a quad
    // anywhere else is a program to hand back rather than one to run with the
    // lane left at zero.
    if let Some(at) = translated.quad {
        if stage != Stage::Fragment {
            return Err(Unsupported::Quad { at });
        }
    }
    // Directives come before every declaration, and the lane index after
    // them: `quadSwap*` say which value, and this says which lane is asking.
    if translated.quad_swap.is_some() {
        if translated.subgroups {
            if translated.subgroup_enable {
                out.push_str("enable subgroups;\n\n");
            }
        } else {
            // The emulation's derivatives sit wherever the shuffle did,
            // which is inside the dispatch loop's arm for its block. The
            // device's own quad operations are read from there too and the
            // language says nothing about them; turning the diagnostic off
            // is what makes the two agree rather than one of them refuse to
            // compile.
            out.push_str("diagnostic(off, derivative_uniformity);\n\n");
        }
    }
    if translated.quad.is_some() {
        out.push_str("var<private> quad_lane: u32;\n");
        out.push_str("fn quadLane() -> u32 { return quad_lane; }\n\n");
    }
    if translated.quad_swap.is_some() && !translated.subgroups {
        out.push_str(QUAD_SWAP);
        out.push_str("\n\n");
    }
    for bank in &layout.const_banks {
        out.push_str(&format!(
            "@group({}) @binding({bank}) var<storage, read> cb{bank}: array<u32>;\n",
            layout.group
        ));
    }
    for (index, texture) in layout.textures.iter().enumerate() {
        let binding = TEXTURE_BINDING + 2 * index as u32;
        out.push_str(&format!(
            "@group({}) @binding({binding}) var tex{index}: {};\n",
            layout.group,
            texture_type(texture.dim, texture.compare)?
        ));
        out.push_str(&format!(
            "@group({}) @binding({}) var smp{index}: {};\n",
            layout.group,
            binding + 1,
            if texture.compare {
                "sampler_comparison"
            } else {
                "sampler"
            }
        ));
    }
    for slot in 0..layout.globals.len() {
        out.push_str(&format!(
            "@group({}) @binding({}) var<storage, read> g{slot}: array<u32>;\n",
            layout.group,
            GLOBAL_BINDING + slot as u32
        ));
    }
    if !layout.const_banks.is_empty() || !layout.textures.is_empty() || !layout.globals.is_empty() {
        out.push('\n');
    }
    if !layout.globals.is_empty() {
        out.push_str("fn gRead(slot: u32, offset: u32) -> u32 {\n  switch (slot) {\n");
        for slot in 0..layout.globals.len() {
            out.push_str(&format!(
                "    case {slot}u: {{ return g{slot}[offset >> 2u]; }}\n"
            ));
        }
        // Out of range reads as zero, which is what an unwritten word does
        // and what the rasterizer's `MemoryGlobal` reports for an address
        // the mapping does not cover.
        out.push_str("    default: { return 0u; }\n  }\n}\n\n");
    }

    // `a[]` is a ten-bit byte address holding one f32 per word. The two
    // halves are separate because a vertex shader's inputs and its outputs
    // occupy the same offsets and must not alias — `Invocation` keeps them
    // apart for the same reason.
    out.push_str(&format!(
        "var<private> attr_in: array<f32, {ATTRIBUTE_WORDS}>;\n"
    ));
    out.push_str(&format!(
        "var<private> attr_out: array<f32, {ATTRIBUTE_WORDS}>;\n\n"
    ));
    let mask = ATTRIBUTE_WORDS - 1;
    out.push_str(&format!(
        "fn attrIn(offset: u32) -> f32 {{ return attr_in[(offset >> 2u) & {mask}u]; }}\n"
    ));
    out.push_str(&format!(
        "fn attrOut(offset: u32, value: f32) {{ attr_out[(offset >> 2u) & {mask}u] = value; }}\n\n"
    ));

    out.push_str("fn cbRead(bank: u32, offset: u32) -> u32 {\n  switch (bank) {\n");
    for bank in &layout.const_banks {
        out.push_str(&format!(
            "    case {bank}u: {{ return cb{bank}[offset >> 2u]; }}\n"
        ));
    }
    // A bank the program reads but the draw never bound. The interpreter
    // raises an error there; a shader has nowhere to raise one, and zero is
    // what an unwritten constant already reads as.
    out.push_str("    default: { return 0u; }\n  }\n}\n\n");

    // `dim` is unused: which of these a call reaches is decided by the
    // immediate, and the dimensionality then belongs to the binding's type.
    // It stays in the signature because it is part of `HOST_INTERFACE`, and
    // a backend that binds textures some other way may want it.
    out.push_str(
        "fn texSample(imm: u32, dim: u32, u: f32, v: f32, layer: u32, w: f32) -> vec4<f32> {\n\
         \x20 switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        if texture.compare {
            continue;
        }
        let coords = match texture.dim {
            TexDim::T2d => "vec2<f32>(u, v), 0.0",
            TexDim::T2dArray => "vec2<f32>(u, v), layer, 0.0",
            TexDim::T3d | TexDim::TCube => "vec3<f32>(u, v, w), 0.0",
            other => return Err(Unsupported::TextureDimension { dim: other }),
        };
        let imm = texture.immediate;
        let sample = format!("textureSampleLevel(tex{index}, smp{index}, {coords})");
        if texture.swizzle == IDENTITY_SWIZZLE {
            out.push_str(&format!("    case {imm}u: {{ return {sample}; }}\n"));
            continue;
        }
        let channels: Vec<&str> = texture
            .swizzle
            .iter()
            .map(|source| match source {
                SwizzleSource::Zero => "0.0",
                SwizzleSource::R => "sampled.x",
                SwizzleSource::G => "sampled.y",
                SwizzleSource::B => "sampled.z",
                SwizzleSource::A => "sampled.w",
                SwizzleSource::One => "1.0",
            })
            .collect();
        out.push_str(&format!("    case {imm}u: {{\n"));
        out.push_str(&format!("      let sampled = {sample};\n"));
        out.push_str(&format!(
            "      return vec4<f32>({});\n    }}\n",
            channels.join(", ")
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }\n  }\n}\n\n");

    // The shadow half. `textureSampleCompareLevel` compares each texel it
    // fetches and filters the results, which is what
    // `texture::sample_compare_with` does texel by texel — so the two
    // renderers answer a soft shadow edge the same way. Alpha is one and the
    // other three channels are the comparison, exactly as Eden's `Extract`
    // hands a shadow sample back.
    out.push_str(
        "fn texSampleCompare(imm: u32, dim: u32, u: f32, v: f32, layer: u32, dref: f32)\
         \x20-> vec4<f32> {\n\
         \x20 switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        if !texture.compare {
            continue;
        }
        let coords = match texture.dim {
            TexDim::T2d => "vec2<f32>(u, v)",
            TexDim::T2dArray => "vec2<f32>(u, v), layer",
            other => return Err(Unsupported::TextureDimension { dim: other }),
        };
        let imm = texture.immediate;
        out.push_str(&format!(
            "    case {imm}u: {{\n      let c = textureSampleCompareLevel(tex{index}, \
             smp{index}, {coords}, dref);\n      return vec4<f32>(c, c, c, 1.0);\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n  }\n}\n\n");

    out.push_str(&translated.source);
    out.push('\n');
    out.push_str(&match stage {
        Stage::Vertex => vertex_entry(layout),
        Stage::Fragment => fragment_entry(translated, layout),
    });
    Ok(out)
}

/// The WGSL type a `texs` of this dimensionality samples.
fn texture_type(dim: TexDim, compare: bool) -> Result<&'static str, Unsupported> {
    match (dim, compare) {
        (TexDim::T2d, false) => Ok("texture_2d<f32>"),
        (TexDim::T2dArray, false) => Ok("texture_2d_array<f32>"),
        (TexDim::T3d, false) => Ok("texture_3d<f32>"),
        (TexDim::TCube, false) => Ok("texture_cube<f32>"),
        (TexDim::T2d, true) => Ok("texture_depth_2d"),
        (TexDim::T2dArray, true) => Ok("texture_depth_2d_array"),
        (dim, _) => Err(Unsupported::TextureDimension { dim }),
    }
}

/// The `attr_in`/`attr_out` word index of generic slot `slot`'s component
/// `component`.
fn generic_word(slot: usize, component: usize) -> usize {
    (GENERIC_BASE + slot * GENERIC_STRIDE) / 4 + component
}

/// `vec4<f32>(a[base + 0], .., a[base + 3])` out of one of the halves.
fn gather(array: &str, base: usize) -> String {
    let words: Vec<String> = (0..4).map(|c| format!("{array}[{}u]", base + c)).collect();
    format!("vec4<f32>({})", words.join(", "))
}

fn vertex_entry(layout: &Layout) -> String {
    let mut out = String::new();
    if !layout.attributes.is_empty() {
        out.push_str("struct VertexInput {\n");
        for slot in &layout.attributes {
            out.push_str(&format!(
                "  @location({slot}) attr{slot}: vec4<{}>,\n",
                attribute_scalar(layout.attribute_base(*slot))
            ));
        }
        out.push_str("}\n\n");
    }
    out.push_str("struct VertexOutput {\n  @builtin(position) position: vec4<f32>,\n");
    for slot in &layout.varyings {
        out.push_str(&format!(
            "  @location({slot}) @interpolate(linear{}) vary{slot}: vec4<f32>,\n",
            layout.sampling(*slot)
        ));
    }
    out.push_str("}\n\n@vertex\nfn vs_main(\n");
    if !layout.attributes.is_empty() {
        out.push_str("  input: VertexInput,\n");
    }
    out.push_str(
        "  @builtin(vertex_index) vertex: u32,\n\
         \x20 @builtin(instance_index) instance: u32,\n\
         ) -> VertexOutput {\n",
    );
    out.push_str(&format!(
        "  attr_in[{}u] = bitcast<f32>(instance);\n  attr_in[{}u] = bitcast<f32>(vertex);\n",
        INSTANCE_ID / 4,
        VERTEX_ID / 4
    ));
    for slot in &layout.attributes {
        // An integer attribute is carried as its bits, the way `shade_vertex`
        // carries one and the way `fetch_attribute` writes one into `a[]`.
        let integer = layout.attribute_base(*slot) != AttributeBase::Float;
        // A BGRA attribute is read the same way and stored with its first and
        // third components swapped, which is `fetch_attribute`'s `swap(0, 2)`.
        let axes = if layout.bgra_attributes.contains(slot) {
            ["z", "y", "x", "w"]
        } else {
            ["x", "y", "z", "w"]
        };
        for (component, axis) in axes.iter().enumerate() {
            let read = format!("input.attr{slot}.{axis}");
            let read = if integer {
                format!("bitcast<f32>({read})")
            } else {
                read
            };
            out.push_str(&format!(
                "  attr_in[{}u] = {read};\n",
                generic_word(*slot, component)
            ));
        }
    }
    // A clip position no `st` writes is (0, 0, 0, 1), which `shade_vertex`
    // gets from `Attributes::written` answering `None`.
    out.push_str(&format!("  attr_out[{}u] = 1.0;\n", POSITION / 4 + 3));
    out.push_str("  run();\n  var out: VertexOutput;\n");
    out.push_str(&format!(
        "  out.position = {};\n",
        gather("attr_out", POSITION / 4)
    ));
    if layout.depth_minus_one_to_one {
        out.push_str("  // Maxwell clips z from -w to w and WebGPU clips it from 0 to w;\n");
        out.push_str("  // left alone, the near half of the frustum is clipped away.\n");
        out.push_str("  out.position.z = (out.position.z + out.position.w) * 0.5;\n");
    }
    if layout.flip_y {
        out.push_str("  // The viewport transform mirrors y, and WebGPU has no\n");
        out.push_str("  // negative viewport height to mirror it with.\n");
        out.push_str("  out.position.y = -out.position.y;\n");
    }
    if !layout.varyings.is_empty() {
        // See this module's note on `@interpolate(linear)`: what the fragment
        // stage has to receive is value/w, not value.
        out.push_str("  let over_w = 1.0 / out.position.w;\n");
        for slot in &layout.varyings {
            out.push_str(&format!(
                "  out.vary{slot} = {} * over_w;\n",
                gather("attr_out", generic_word(*slot, 0))
            ));
        }
    }
    out.push_str("  return out;\n}\n");
    out
}

fn fragment_entry(translated: &Translation, layout: &Layout) -> String {
    let colour = |target: u32| -> String {
        let components: Vec<String> = (0..4)
            .map(|c| {
                let reg = (target * 4 + c) as u8;
                if translated.registers.contains(&reg) {
                    format!("bitcast<f32>(r{reg})")
                } else {
                    // A channel the shader never wrote. `Invocation`'s
                    // register file starts at zero and `shade_fragment` reads
                    // it out regardless.
                    "0.0".to_string()
                }
            })
            .collect();
        format!("vec4<f32>({})", components.join(", "))
    };

    let mut out =
        String::from("struct FragmentInput {\n  @builtin(position) position: vec4<f32>,\n");
    for slot in &layout.varyings {
        out.push_str(&format!(
            "  @location({slot}) @interpolate(linear{}) vary{slot}: vec4<f32>,\n",
            layout.sampling(*slot)
        ));
    }
    out.push_str("}\n\n");
    let targets = layout.targets;
    if targets > 1 {
        out.push_str("struct FragmentOutput {\n");
        for target in 0..targets {
            out.push_str(&format!(
                "  @location({target}) target{target}: vec4<f32>,\n"
            ));
        }
        out.push_str("}\n\n");
    }
    let returns = match targets {
        0 => String::new(),
        1 => " -> @location(0) vec4<f32>".to_string(),
        _ => " -> FragmentOutput".to_string(),
    };
    out.push_str(&format!(
        "@fragment\nfn fs_main(input: FragmentInput){returns} {{\n"
    ));
    // Coverage first, because the sample mask is coverage: a sample the mask
    // excludes is one the fragment never had, and `Fragments::coverage`
    // applies it before anything is shaded.
    if let Some(coverage) = &layout.coverage {
        out.push_str(&sample_index(coverage));
        if coverage.sample_mask != u32::MAX {
            out.push_str(&format!(
                "  if (({}u >> sample) & 1u) == 0u {{ discard; }}\n",
                coverage.sample_mask
            ));
        }
    }
    // Which lane of the 2x2 quad this is, numbered as `raster::quad_pixel`
    // numbers them: the row parity above the column parity.
    if translated.quad.is_some() {
        out.push_str(
            "  quad_lane = (u32(input.position.y) & 1u) * 2u + (u32(input.position.x) & 1u);\n",
        );
    }
    // WGSL's fragment `position.w` is `1/w` interpolated linearly, which is
    // exactly what `a[0x7c]` holds.
    out.push_str(&format!(
        "  attr_in[{}u] = input.position.w;\n",
        POSITION / 4 + 3
    ));
    for slot in &layout.varyings {
        for (component, axis) in ["x", "y", "z", "w"].iter().enumerate() {
            out.push_str(&format!(
                "  attr_in[{}u] = input.vary{slot}.{axis};\n",
                generic_word(*slot, component)
            ));
        }
    }
    out.push_str("  if (run()) { discard; }\n");
    // Alpha-to-coverage narrows the mask *after* shading, since it is the
    // shaded alpha it turns into coverage — `Fragments::write`'s ordering.
    let alpha_to_coverage = layout.coverage.as_ref().filter(|c| c.alpha_to_coverage);
    if targets > 1 {
        out.push_str("  var out: FragmentOutput;\n");
        for target in 0..targets {
            out.push_str(&format!("  out.target{target} = {};\n", colour(target)));
        }
        if let Some(coverage) = alpha_to_coverage {
            out.push_str(&alpha_coverage(coverage, "out.target0.w"));
        }
        out.push_str("  return out;\n");
    } else if let Some(coverage) = alpha_to_coverage {
        // Shaded even with nowhere to put it: a depth-only pass still turns
        // its alpha into coverage, which is what `Fragments::write` does.
        out.push_str(&format!("  let target0 = {};\n", colour(0)));
        out.push_str(&alpha_coverage(coverage, "target0.w"));
        if targets == 1 {
            out.push_str("  return target0;\n");
        }
    } else if targets == 1 {
        out.push_str(&format!("  return {};\n", colour(0)));
    }
    out.push_str("}\n");
    out
}

/// Work out which sample of its pixel this fragment is.
///
/// A texel's place inside its pixel's tile is its position modulo the grid,
/// and which sample lives in that place is the table hardware fixes per mode
/// — so this is a lookup, not an arithmetic identity. The table is a local
/// `var` rather than a `const` because it is indexed by a value only known at
/// run time, and that is the form every WGSL implementation accepts.
fn sample_index(coverage: &Coverage) -> String {
    let slots: Vec<String> = coverage
        .sample_of_slot
        .iter()
        .map(|s| format!("{s}u"))
        .collect();
    let count = slots.len();
    format!(
        "  var sample_of_slot = array<u32, {count}>({});\n\
         \x20 let tile = vec2<u32>(u32(input.position.x) % {}u, u32(input.position.y) % {}u);\n\
         \x20 let sample = sample_of_slot[tile.y * {}u + tile.x];\n",
        slots.join(", "),
        coverage.samples_x,
        coverage.samples_y,
        coverage.samples_x,
    )
}

/// The samples an alpha of `alpha` keeps, as `raster::alpha_coverage` keeps
/// them: a prefix of `round(alpha * count)` of them, so every pixel of equal
/// alpha keeps the same fraction.
fn alpha_coverage(coverage: &Coverage, alpha: &str) -> String {
    let count = coverage.samples_x * coverage.samples_y;
    // `floor(x + 0.5)` rather than `round(x)`: WGSL's `round` breaks a tie
    // towards the even whole number and Rust's breaks it away from zero, so
    // an alpha of an eighth over four samples keeps one sample there and none
    // here. This is the arithmetic `alpha_coverage` actually performs.
    format!(
        "  if (sample >= u32(floor(clamp({alpha}, 0.0, 1.0) * {}.0 + 0.5))) {{ discard; }}\n",
        count
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shader::isa::{FmulScale, Instruction, MemSize, ShflMode};
    use crate::gpu::shader::{next_slot, Program, ENTRY_OFFSET};
    use std::collections::BTreeMap;

    const ALWAYS: Pred = Pred::ALWAYS;
    /// `@p0` — the guard a two-armed branch is built out of.
    const IF_P0: Pred = Pred {
        reg: 0,
        negate: false,
    };
    const NO_MOD: FMod = FMod::NONE;

    /// The byte offset instruction `index` lands at in a real 32-byte-block
    /// layout, so branch targets resolve the way they do in a decoded shader.
    fn at(index: usize) -> u32 {
        let mut offset = ENTRY_OFFSET;
        for _ in 0..index {
            offset = next_slot(offset);
        }
        offset
    }

    fn program(entries: &[(Op, Pred)]) -> Compiled {
        build(entries, BTreeMap::new())
    }

    fn build(entries: &[(Op, Pred)], indirect: BTreeMap<u32, Vec<u32>>) -> Compiled {
        let mut p = Program {
            indirect,
            ..Program::default()
        };
        for (index, &(op, pred)) in entries.iter().enumerate() {
            p.insns.push(Instruction { pred, op });
            p.offsets.push(at(index));
        }
        Compiled::new(&p)
    }

    /// The braces the emitted text opens and closes must balance, or nothing
    /// downstream will parse it. Cheap, and it catches every emitter that
    /// returns early with a block still open.
    fn braces_balance(source: &str) -> bool {
        let mut depth = 0i32;
        for c in source.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return false;
            }
        }
        depth == 0
    }

    /// One of every opcode the Home Menu's twelve shaders use, measured over
    /// a live qlaunch frame. A translator that cannot emit one of these
    /// cannot render the Home Menu, so this is the list that decides whether
    /// the front half of a GPU backend is finished.
    fn home_menu_opcodes() -> Vec<Op> {
        vec![
            Op::Ffma {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                bneg: false,
                c: Operand::Imm(0),
                cneg: false,
                ftz: true,
                sat: false,
            },
            Op::Fadd {
                dst: 1,
                a: 2,
                am: NO_MOD,
                b: Operand::Reg(3),
                bm: NO_MOD,
                ftz: true,
                sat: false,
            },
            Op::Fmul {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                bm: NO_MOD,
                ftz: true,
                sat: false,
                scale: FmulScale::None,
            },
            Op::Mov {
                dst: 1,
                src: Operand::Reg(2),
            },
            Op::Fsetp {
                p0: 0,
                p1: 7,
                a: 1,
                am: NO_MOD,
                b: Operand::Reg(2),
                bm: NO_MOD,
                cmp: FCmp::Lt,
                bop: BoolOp::And,
                src: ALWAYS,
            },
            Op::Isetp {
                p0: 0,
                p1: 7,
                a: 1,
                b: Operand::Imm(3),
                cmp: ICmp::Eq,
                signed: true,
                bop: BoolOp::And,
                src: ALWAYS,
            },
            Op::Mov32i {
                dst: 1,
                imm: 0x3f80_0000,
            },
            Op::Iadd {
                dst: 1,
                a: 2,
                aneg: false,
                b: Operand::Imm(1),
                bneg: false,
                cin: false,
                cout: true,
            },
            Op::Lop {
                dst: 1,
                a: 2,
                ainv: false,
                b: Operand::Imm(0xff),
                binv: false,
                op: LogicOp::And,
                pred: Some((1, LopTest::NonZero)),
            },
            Op::Mufu {
                dst: 1,
                src: 2,
                sm: NO_MOD,
                op: MufuOp::Rcp,
                sat: false,
            },
            Op::Shr {
                dst: 1,
                a: 2,
                b: Operand::Imm(4),
                signed: false,
                wrap: false,
            },
            Op::F2i {
                dst: 1,
                src: Operand::Reg(2),
                sm: NO_MOD,
                dst_bytes: 4,
                dst_signed: true,
                round: FRound::Trunc,
                ftz: true,
            },
            Op::Iscadd {
                dst: 1,
                a: 2,
                aneg: false,
                b: Operand::Reg(3),
                bneg: false,
                shift: 2,
            },
            Op::Iset {
                dst: 1,
                a: 2,
                b: Operand::Imm(3),
                cmp: ICmp::Eq,
                signed: true,
                bop: BoolOp::And,
                src: ALWAYS,
                bf: false,
            },
            Op::Ipa {
                dst: 1,
                offset: 0x80,
                mul: Some(2),
                perspective: true,
                sat: false,
                centroid: false,
            },
            Op::Fmnmx {
                dst: 1,
                a: 2,
                am: NO_MOD,
                b: Operand::Reg(3),
                bm: NO_MOD,
                pred: ALWAYS,
                ftz: true,
            },
            Op::Ldc {
                dst: 1,
                bank: 1,
                offset: 0x14,
                idx: 2,
                size: MemSize::B32,
            },
            Op::St {
                offset: 0x70,
                idx: RZ,
                src: 1,
                size: MemSize::B32,
            },
            Op::I2f {
                dst: 1,
                src: Operand::Reg(2),
                sm: NO_MOD,
                src_bytes: 4,
                src_signed: true,
                sel: 0,
            },
            Op::Shl {
                dst: 1,
                a: 2,
                b: Operand::Imm(2),
                wrap: false,
            },
            Op::Bfi {
                dst: 1,
                insert: 2,
                src: Operand::Reg(3),
                base: Operand::Reg(4),
            },
            Op::Imnmx {
                dst: 1,
                a: 2,
                b: Operand::Imm(4),
                pred: ALWAYS,
                signed: false,
            },
            Op::Fset {
                dst: 1,
                a: 2,
                am: NO_MOD,
                b: Operand::Reg(3),
                bm: NO_MOD,
                cmp: FCmp::Ge,
                bop: BoolOp::And,
                src: ALWAYS,
                bf: true,
            },
            Op::R2p {
                src: 1,
                mask: Operand::Imm(0x7f),
                byte: 0,
            },
            Op::Ld {
                offset: 0x80,
                idx: RZ,
                dst: 1,
                size: MemSize::B32,
            },
            Op::Texs {
                dst: 1,
                dst2: 3,
                coords: [4, 5, RZ],
                dref: None,
                handle: 0x1a4,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
                f16: false,
            },
            Op::Icmp {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                c: 4,
                cmp: ICmp::Ne,
                signed: true,
            },
            Op::Iadd3 {
                dst: 1,
                a: 2,
                aneg: false,
                b: Operand::Reg(3),
                bneg: false,
                c: Operand::Reg(4),
                cneg: false,
            },
            Op::Bfe {
                dst: 1,
                a: 2,
                b: Operand::Imm(0x0810),
                signed: false,
            },
        ]
    }

    #[test]
    fn every_opcode_the_home_menu_uses_translates() {
        // The control-flow opcodes in that set — bra, ssy, sync, pbk, brk,
        // brx, exit — are covered by the tests below, which need programs
        // shaped around them rather than a straight line.
        for op in home_menu_opcodes() {
            let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
            let wgsl = translate(&p)
                .unwrap_or_else(|e| panic!("{op:?}: {e}"))
                .source;
            assert!(braces_balance(&wgsl), "{op:?} left a block open:\n{wgsl}");
        }
    }

    #[test]
    fn a_guard_becomes_a_conditional_rather_than_a_dropped_instruction() {
        let p = program(&[
            (
                Op::Mov {
                    dst: 1,
                    src: Operand::Imm(7),
                },
                IF_P0,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("if (p0) {"), "{wgsl}");
        assert!(wgsl.contains("r1 = 7u;"), "{wgsl}");
    }

    #[test]
    fn a_guarded_branch_says_where_control_goes_when_it_is_not_taken() {
        // Without the `else`, control would fall out of the `case` with `pc`
        // unchanged and the block would run again forever.
        let p = program(&[
            (Op::Bra { target: at(2) }, IF_P0),
            (Op::Nop, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("pc = 2u;"), "the taken edge:\n{wgsl}");
        assert!(wgsl.contains("} else {"), "the not-taken edge:\n{wgsl}");
        assert!(wgsl.contains("pc = 1u;"), "which falls through:\n{wgsl}");
    }

    #[test]
    fn reconvergence_becomes_an_explicit_stack() {
        let p = program(&[
            (Op::Ssy { target: at(3) }, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Sync, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("stack[sp] = 3u;"), "the push:\n{wgsl}");
        assert!(wgsl.contains("sp = sp - 1;"), "the pop:\n{wgsl}");
        assert!(
            wgsl.contains("pc = stack[sp];"),
            "and where it goes:\n{wgsl}"
        );
    }

    #[test]
    fn a_brx_becomes_a_switch_over_the_arms_its_table_names() {
        // Byte offsets in, indices out: the emitted switch compares the
        // address the branch computes and assigns the block that address is.
        let arms = vec![at(3), at(4)];
        let mut indirect = BTreeMap::new();
        indirect.insert(at(0), arms.clone());
        let p = build(
            &[
                (Op::Brx { base: 0, reg: 16 }, ALWAYS),
                (Op::Nop, ALWAYS),
                (Op::Nop, ALWAYS),
                (Op::Exit, ALWAYS),
                (Op::Exit, ALWAYS),
            ],
            indirect,
        );
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("0u + r16"), "the computed address:\n{wgsl}");
        assert!(
            wgsl.contains("& 31u) == 0u"),
            "rounded onto a slot:\n{wgsl}"
        );
        assert!(
            wgsl.contains(&format!("case {}u: {{ pc = 3u; }}", at(3))),
            "arm 0:\n{wgsl}"
        );
        assert!(
            wgsl.contains(&format!("case {}u: {{ pc = 4u; }}", at(4))),
            "arm 1:\n{wgsl}"
        );
    }

    #[test]
    fn a_brx_with_no_known_arms_is_reported_rather_than_guessed() {
        let p = program(&[(Op::Brx { base: 0, reg: 16 }, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(
            translate(&p).unwrap_err(),
            Unsupported::IndirectBranch { at: 0 }
        );
    }

    #[test]
    fn global_memory_is_reported_rather_than_mistranslated() {
        // `ldg` needs a storage buffer, which is a question about binding
        // resources rather than about translating instructions. Saying so is
        // what lets a caller fall back for that draw.
        let op = Op::Ldg {
            dst: 1,
            addr: 2,
            offset: 0,
            size: MemSize::B32,
        };
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(translate(&p).unwrap_err(), Unsupported::Op { at: 0, op });
    }

    /// A `ldg` whose address is a descriptor in a constant bank plus an
    /// index, which is how every compiler builds one — A Short Hike's own
    /// fragment shader does exactly this, twelve times over.
    #[test]
    fn a_global_load_through_a_descriptor_binds_the_memory_it_names() {
        let base = |dst, a, offset, cin, cout| Op::Iadd {
            dst,
            a,
            aneg: false,
            b: Operand::Const { bank: 0, offset },
            bneg: false,
            cin,
            cout,
        };
        let p = program(&[
            // r10 = r7 + c0[0x110], carrying out; r11 = RZ + c0[0x114] with it.
            (base(10, 7, 0x110, false, true), ALWAYS),
            (base(11, RZ, 0x114, true, false), ALWAYS),
            (
                Op::Ldg {
                    dst: 10,
                    addr: 10,
                    offset: 0,
                    size: MemSize::B32,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        assert_eq!(translated.globals, vec![(0, 0x110)]);
        assert!(
            translated.source.contains("gRead(0u,"),
            "{}",
            translated.source
        );
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(
            source.contains("var<storage, read> g0: array<u32>"),
            "{source}"
        );
        assert!(
            source.contains("case 0u: { return g0[offset >> 2u]; }"),
            "{source}"
        );

        // An address the shader computed some other way is memory this
        // cannot reach, and is reported rather than read from somewhere.
        let op = Op::Ldg {
            dst: 1,
            addr: 2,
            offset: 0,
            size: MemSize::B32,
        };
        let loose = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(
            translate(&loose).unwrap_err(),
            Unsupported::Op { at: 0, op }
        );
    }

    #[test]
    fn a_warp_shuffle_is_a_quad_operation_where_the_device_has_them() {
        // A `shfl` reads another lane of the 2x2 quad, which is exactly what
        // `quadSwapX`/`Y`/`Diagonal` do — and a quad is the whole warp the
        // rasterizer models.
        let op = Op::Shfl {
            dst: 1,
            pred: 0,
            src: 2,
            index: Operand::Imm(1),
            mask: Operand::Imm(0x1c),
            mode: ShflMode::Bfly,
        };
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        let translated = translate_for(
            &p,
            Caps {
                subgroups: true,
                subgroup_enable: true,
            },
        )
        .unwrap();
        assert_eq!(translated.quad, Some(0));
        assert_eq!(translated.quad_swap, Some(0));
        for wanted in ["quadSwapX(", "quadSwapY(", "quadSwapDiagonal("] {
            assert!(
                translated.source.contains(wanted),
                "{wanted} missing from {}",
                translated.source
            );
        }
        // The enable and the lane the arithmetic asks about are the module's,
        // and the device's operations are taken as they are rather than
        // defined.
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(source.starts_with("enable subgroups;"), "{source}");
        assert!(!source.contains("dpdxFine"), "{source}");
        assert!(
            source.contains("quad_lane = (u32(input.position.y)"),
            "{source}"
        );
    }

    #[test]
    fn a_warp_shuffle_without_quad_operations_is_a_fine_derivative() {
        // Every browser device is this one: wgpu's web backend can neither
        // request WebGPU's `subgroups` nor report it, so a `shfl` that fell
        // back here latched the whole session onto the rasterizer. The
        // module defines the three operations out of derivatives instead —
        // see `QUAD_SWAP` for why that recovers the neighbour's bits exactly.
        let op = Op::Shfl {
            dst: 1,
            pred: 0,
            src: 2,
            index: Operand::Imm(1),
            mask: Operand::Imm(0x1c),
            mode: ShflMode::Bfly,
        };
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        assert_eq!(translated.quad_swap, Some(0));

        let layout = Layout::of(&translated, Stage::Fragment);
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        // The diagnostic is a directive and so comes before every
        // declaration; nothing asks the device for a feature it does not
        // have.
        assert!(
            source.starts_with("diagnostic(off, derivative_uniformity);"),
            "{source}"
        );
        assert!(!source.contains("enable subgroups"), "{source}");
        for wanted in ["fn quadSwapX(", "dpdxFine(", "dpdyFine("] {
            assert!(source.contains(wanted), "{wanted} missing from {source}");
        }

        // Only a fragment shader has derivatives, so only a fragment shader
        // has this route.
        let layout = Layout::of(&translated, Stage::Vertex);
        assert_eq!(
            module(&translated, Stage::Vertex, &layout).unwrap_err(),
            Unsupported::Quad { at: 0 }
        );
    }

    #[test]
    fn fswzadd_asks_which_lane_it_is_and_nothing_of_the_device() {
        // It reads no other lane at all — each lane combines its *own* two
        // operands with the signs its two bits of the swizzle name — so
        // rejecting it without the device's quad operations turned a draw
        // that needed nothing into a fallback.
        let op = Op::Fswzadd {
            dst: 1,
            a: 2,
            b: 3,
            swizzle: 0x99,
            ftz: false,
        };
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        assert_eq!(translated.quad, Some(0));
        assert_eq!(translated.quad_swap, None);

        let layout = Layout::of(&translated, Stage::Fragment);
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(source.contains("fn quadLane()"), "{source}");
        assert!(!source.contains("fn quadSwapX("), "{source}");

        // A vertex shader still cannot have it: nothing there knows which
        // lane it is, and a lane index left at zero would give every vertex
        // the signs of the first lane rather than its own.
        let layout = Layout::of(&translated, Stage::Vertex);
        assert_eq!(
            module(&translated, Stage::Vertex, &layout).unwrap_err(),
            Unsupported::Quad { at: 0 }
        );
    }

    #[test]
    fn an_undecoded_branch_target_is_reported_before_anything_is_emitted() {
        // `target` past the end never resolved to an index, so there is no
        // block to jump to. The interpreter raises this where the branch is
        // taken; a translation has to know first.
        let p = program(&[(Op::Bra { target: 0x9999 }, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(
            translate(&p).unwrap_err(),
            Unsupported::UndecodedTarget { at: 0 }
        );
    }

    #[test]
    fn a_block_starts_at_every_branch_target() {
        // Instruction 2 is only reachable by the branch, so it has to be its
        // own case — a translation that folded it into the block above would
        // run it on the fall-through path as well.
        let p = program(&[
            (Op::Bra { target: at(2) }, ALWAYS),
            (Op::Nop, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        for leader in ["case 0u: {", "case 1u: {", "case 2u: {"] {
            assert!(wgsl.contains(leader), "missing {leader}:\n{wgsl}");
        }
    }

    #[test]
    fn only_the_registers_a_program_touches_are_declared() {
        let p = program(&[
            (
                Op::Mov {
                    dst: 9,
                    src: Operand::Reg(4),
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("var<private> r4: u32 = 0u;"), "{wgsl}");
        assert!(wgsl.contains("var<private> r9: u32 = 0u;"), "{wgsl}");
        assert!(
            !wgsl.contains(" r5:"),
            "declared a register nothing uses:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("var carry"),
            "declared a carry nothing sets:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("var stack"),
            "declared a stack nothing pushes:\n{wgsl}"
        );
    }

    #[test]
    fn a_fragment_shaders_colour_is_readable_after_the_call() {
        // Maxwell has no output attribute for a fragment's colour: the
        // rasterizer reads r0 to r3 once the invocation ends. Registers that
        // did not outlive `run` would leave a backend with nothing to write
        // to the render target.
        let p = program(&[
            (
                Op::Mov {
                    dst: 0,
                    src: Operand::Imm(0x3f80_0000),
                },
                ALWAYS,
            ),
            (
                Op::Mov {
                    dst: 3,
                    src: Operand::Imm(0),
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        assert_eq!(translated.registers, vec![0, 3]);
        for reg in &translated.registers {
            assert!(
                translated
                    .source
                    .contains(&format!("var<private> r{reg}: u32")),
                "r{reg} does not outlive the call:\n{}",
                translated.source
            );
        }
    }

    #[test]
    fn the_zero_register_reads_as_zero_and_discards_what_is_written_to_it() {
        let p = program(&[
            (
                Op::Mov {
                    dst: RZ,
                    src: Operand::Reg(RZ),
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(!wgsl.contains("r255"), "RZ is not a register:\n{wgsl}");
    }

    #[test]
    fn the_function_returns_on_every_path() {
        // The dispatch loop has no `break`, so control cannot fall out of it
        // — but WGSL requires a function with a return type to return at the
        // end of its body regardless, and `naga` rejects one that does not.
        let p = program(&[(Op::Exit, ALWAYS)]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.trim_end().ends_with("return false;\n}"), "{wgsl}");
    }

    #[test]
    fn only_the_helpers_a_program_reaches_are_emitted() {
        let p = program(&[
            (
                Op::Shl {
                    dst: 1,
                    a: 2,
                    b: Operand::Imm(2),
                    wrap: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("fn shl32("), "{wgsl}");
        assert!(
            !wgsl.contains("fn lop3("),
            "carried a helper it never calls:\n{wgsl}"
        );
    }

    #[test]
    fn a_helper_never_arrives_without_the_one_it_calls() {
        // `mulhi_s` corrects `mulhi_u`'s result; emitting it alone would not
        // compile.
        let p = program(&[
            (
                Op::Imul {
                    dst: 1,
                    a: 2,
                    b: Operand::Reg(3),
                    signed: true,
                    hi: true,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(wgsl.contains("fn mulhi_u("), "{wgsl}");
        assert!(
            wgsl.find("fn mulhi_u(") < wgsl.find("fn mulhi_s("),
            "a helper must be defined before it is called:\n{wgsl}"
        );
    }

    #[test]
    fn the_host_interface_is_what_the_emitted_text_calls() {
        // Every hook the emitter can emit a call to has to be in the list a
        // backend is told to supply, or a translation using it will not link.
        let mut wgsl = String::new();
        for op in home_menu_opcodes() {
            let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
            wgsl.push_str(&translate(&p).unwrap().source);
        }
        for hook in ["attrIn(", "attrOut(", "cbRead(", "texSample("] {
            assert!(wgsl.contains(hook), "nothing emits a call to {hook}");
            assert!(
                HOST_INTERFACE.contains(hook),
                "{hook} is not in HOST_INTERFACE"
            );
        }
    }

    /// A vertex shader that reads attribute `slot` and writes varying `slot`,
    /// and a fragment shader that interpolates it — the smallest pair that
    /// has an interface at all.
    fn pair(slot: usize) -> (Compiled, Compiled) {
        let offset = (GENERIC_BASE + slot * GENERIC_STRIDE) as u16;
        let vs = program(&[
            (
                Op::Ld {
                    dst: 1,
                    offset,
                    idx: RZ,
                    size: MemSize::B32,
                },
                ALWAYS,
            ),
            (
                Op::St {
                    offset,
                    idx: RZ,
                    src: 1,
                    size: MemSize::B32,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let fs = program(&[
            (
                Op::Ipa {
                    dst: 0,
                    offset,
                    mul: None,
                    perspective: true,
                    sat: false,
                    centroid: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        (vs, fs)
    }

    #[test]
    fn a_layout_is_read_off_what_translating_the_program_touched() {
        let (vs, fs) = pair(3);
        let vs = translate(&vs).unwrap();
        let fs = translate(&fs).unwrap();
        assert_eq!(Layout::of(&vs, Stage::Vertex).attributes, vec![3]);
        assert_eq!(Layout::of(&vs, Stage::Vertex).varyings, vec![3]);
        // A fragment shader has no vertex attributes, and what it reads out
        // of `a[]` is a varying.
        assert_eq!(
            Layout::of(&fs, Stage::Fragment).attributes,
            Vec::<usize>::new()
        );
        assert_eq!(Layout::of(&fs, Stage::Fragment).varyings, vec![3]);
    }

    #[test]
    fn a_layout_names_every_binding_the_text_calls_through() {
        let p = program(&[
            (
                Op::Ldc {
                    dst: 1,
                    bank: 5,
                    offset: 0x10,
                    idx: RZ,
                    size: MemSize::B32,
                },
                ALWAYS,
            ),
            (
                Op::Mov {
                    dst: 2,
                    src: Operand::Const {
                        bank: 1,
                        offset: 0x40,
                    },
                },
                ALWAYS,
            ),
            (
                Op::Texs {
                    dst: 4,
                    dst2: 6,
                    coords: [7, 8, RZ],
                    dref: None,
                    handle: 0x1a4,
                    dim: TexDim::T2d,
                    mask: [true, true, true, true],
                    f16: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        let layout = Layout::of(&translated, Stage::Fragment);
        assert_eq!(layout.const_banks, vec![1, 5]);
        assert_eq!(
            layout.textures,
            vec![TextureBinding {
                immediate: 0x1a4,
                dim: TexDim::T2d,
                compare: false,
                swizzle: IDENTITY_SWIZZLE
            }]
        );
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(source.contains("var<storage, read> cb1:"), "{source}");
        assert!(source.contains("var<storage, read> cb5:"), "{source}");
        assert!(
            source.contains("case 420u: { return textureSampleLevel(tex0, smp0"),
            "{source}"
        );
    }

    #[test]
    fn a_swizzled_texture_is_rearranged_where_it_is_sampled() {
        // WebGPU has no per-texture component swizzle, and the descriptor
        // does: two thirds of the Home Menu's draws sample a single-channel
        // image as `[R, R, R, One]`, and sampling it as the identity gets
        // every one of them wrong.
        let p = program(&[
            (
                Op::Texs {
                    dst: 0,
                    dst2: 2,
                    coords: [4, 5, RZ],
                    dref: None,
                    handle: 8,
                    dim: TexDim::T2d,
                    mask: [true, true, true, true],
                    f16: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        let mut layout = Layout::of(&translated, Stage::Fragment);
        // What a backend fills in from the TIC.
        layout.textures[0].swizzle = [
            SwizzleSource::R,
            SwizzleSource::R,
            SwizzleSource::R,
            SwizzleSource::One,
        ];
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(
            source.contains("return vec4<f32>(sampled.x, sampled.x, sampled.x, 1.0);"),
            "{source}"
        );
        // The identity costs nothing.
        layout.textures[0].swizzle = IDENTITY_SWIZZLE;
        let plain = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!plain.contains("let sampled ="), "{plain}");
    }

    #[test]
    fn both_stages_name_the_same_location_for_a_varying() {
        // A pipeline whose stages disagree about a location does not link,
        // and nothing but the layout keeps them together.
        let (vs, fs) = pair(7);
        let vs = translate(&vs).unwrap();
        let fs = translate(&fs).unwrap();
        let vs = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
        let fs = module(&fs, Stage::Fragment, &Layout::of(&fs, Stage::Fragment)).unwrap();
        assert!(
            vs.contains("@location(7) @interpolate(linear) vary7"),
            "{vs}"
        );
        assert!(
            fs.contains("@location(7) @interpolate(linear) vary7"),
            "{fs}"
        );
    }

    /// Only the fragment program's `ipa` says a varying is sampled at the
    /// centroid, and an `@interpolate` the two stages spell differently is a
    /// pipeline that will not build — so the vertex stage is told, the way
    /// `Gpu::pipeline` tells it.
    #[test]
    fn a_centroid_varying_is_qualified_the_same_way_in_both_stages() {
        let offset = (GENERIC_BASE + 5 * GENERIC_STRIDE) as u16;
        let (vs, _) = pair(5);
        let fs = program(&[
            (
                Op::Ipa {
                    dst: 0,
                    offset,
                    mul: None,
                    perspective: true,
                    sat: false,
                    centroid: true,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let vs = translate(&vs).unwrap();
        let fs = translate(&fs).unwrap();
        let fs_layout = Layout::of(&fs, Stage::Fragment);
        assert_eq!(fs_layout.centroid_varyings, vec![5]);
        let mut vs_layout = Layout::of(&vs, Stage::Vertex);
        assert_eq!(
            vs_layout.centroid_varyings,
            Vec::<usize>::new(),
            "a vertex shader has no ipa"
        );
        vs_layout.centroid_varyings = fs_layout.centroid_varyings.clone();

        let vs = module(&vs, Stage::Vertex, &vs_layout).unwrap();
        let fs = module(&fs, Stage::Fragment, &fs_layout).unwrap();
        assert!(
            vs.contains("@location(5) @interpolate(linear, centroid) vary5"),
            "{vs}"
        );
        assert!(
            fs.contains("@location(5) @interpolate(linear, centroid) vary5"),
            "{fs}"
        );
    }

    #[test]
    fn varyings_interpolate_linearly_because_the_shader_divides_by_w_itself() {
        // Maxwell's `ipa` is handed value/w and finishes the perspective
        // divide with `rcp(a[0x7c])`. Letting the hardware correct as well
        // would divide twice, which looks like a texture-coordinate bug and
        // is not one.
        let (vs, _) = pair(0);
        let vs = translate(&vs).unwrap();
        let source = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
        assert!(source.contains("@interpolate(linear)"), "{source}");
        assert!(!source.contains("perspective"), "{source}");
        assert!(
            source.contains("let over_w = 1.0 / out.position.w;"),
            "{source}"
        );
        assert!(source.contains("* over_w;"), "{source}");
    }

    #[test]
    fn a_clip_position_no_store_writes_is_the_one_the_rasterizer_defaults_to() {
        // `shade_vertex` reads (0, 0, 0, 1) when `Attributes::written`
        // answers `None`; zero-initialised storage gets three of those four
        // right on its own.
        let (vs, _) = pair(0);
        let vs = translate(&vs).unwrap();
        let source = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
        assert!(source.contains("attr_out[31u] = 1.0;"), "{source}");
    }

    #[test]
    fn an_integer_attribute_is_declared_as_one_and_carried_as_its_bits() {
        // WebGPU makes the base type part of the match between a vertex
        // format and the input it feeds, so a `Sint8x4` attribute fed to a
        // `vec4<f32>` input is a pipeline that will not build. The bitcast is
        // what keeps the slot holding the same bits `fetch_attribute` puts
        // there for an integer attribute.
        let (vs, _) = pair(3);
        let vs = translate(&vs).unwrap();
        let mut layout = Layout::of(&vs, Stage::Vertex);
        layout.integer_attributes = vec![(3, AttributeBase::Sint)];
        let source = module(&vs, Stage::Vertex, &layout).unwrap();
        assert!(source.contains("@location(3) attr3: vec4<i32>"), "{source}");
        assert!(source.contains("bitcast<f32>(input.attr3.x)"), "{source}");
        assert!(braces_balance(&source), "{source}");

        // A float attribute is read straight through, and is what a slot the
        // draw said nothing about is assumed to be.
        let plain = Layout::of(&vs, Stage::Vertex);
        let source = module(&vs, Stage::Vertex, &plain).unwrap();
        assert!(source.contains("@location(3) attr3: vec4<f32>"), "{source}");
        assert!(!source.contains("bitcast<f32>(input."), "{source}");
    }

    #[test]
    fn a_depth_only_pass_writes_no_colour_and_still_shades() {
        // A fragment shader naming `@location(0)` with no colour attachment
        // behind it is a pipeline that will not build, and a pass that
        // skipped the shader entirely would lose `kil` — which is a reason a
        // fragment is not there, and so a reason depth is not written.
        let p = program(&[
            (
                Op::Mov {
                    dst: 0,
                    src: Operand::Imm(0x3f80_0000),
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        let mut layout = Layout::of(&translated, Stage::Fragment);
        layout.targets = 0;
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(
            source.contains("fn fs_main(input: FragmentInput) {"),
            "{source}"
        );
        assert!(!source.contains("@location(0) vec4<f32>"), "{source}");
        assert!(source.contains("if (run()) { discard; }"), "{source}");
        let entry = source
            .split("@fragment")
            .nth(1)
            .expect("a fragment entry point");
        assert!(!entry.contains("return"), "{source}");
        assert!(braces_balance(&source), "{source}");

        // Alpha-to-coverage still shades it, because the alpha is what the
        // coverage is made of.
        layout.coverage = Some(quad_coverage(u32::MAX, true));
        let shaded = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(shaded.contains("let target0 = vec4<f32>("), "{shaded}");
        assert!(
            shaded.contains("if (sample >= u32(floor(clamp(target0.w"),
            "{shaded}"
        );
        assert!(!shaded.contains("return target0;"), "{shaded}");
        assert!(braces_balance(&shaded), "{shaded}");
    }

    #[test]
    fn a_bgra_attribute_is_swapped_where_it_is_read() {
        // WebGPU has no BGRA vertex format, and `fetch_attribute` ends with
        // `out.swap(0, 2)` — so the swap has to happen here or the two
        // renderers disagree about which channel a packed colour's blue is.
        let (vs, _) = pair(2);
        let vs = translate(&vs).unwrap();
        let mut layout = Layout::of(&vs, Stage::Vertex);
        layout.bgra_attributes = vec![2];
        let source = module(&vs, Stage::Vertex, &layout).unwrap();
        assert!(source.contains("attr_in[40u] = input.attr2.z;"), "{source}");
        assert!(source.contains("attr_in[42u] = input.attr2.x;"), "{source}");
        // y and w are not part of the swap.
        assert!(source.contains("attr_in[41u] = input.attr2.y;"), "{source}");
        assert!(source.contains("attr_in[43u] = input.attr2.w;"), "{source}");

        let plain = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
        assert!(plain.contains("attr_in[40u] = input.attr2.x;"), "{plain}");
    }

    /// A 2x2 grid in raster order, which is what an unprogrammed sample
    /// location table gives.
    fn quad_coverage(sample_mask: u32, alpha_to_coverage: bool) -> Coverage {
        Coverage {
            samples_x: 2,
            samples_y: 2,
            sample_of_slot: vec![0, 1, 2, 3],
            sample_mask,
            alpha_to_coverage,
        }
    }

    #[test]
    fn a_sample_mask_discards_the_texels_it_excludes() {
        // Rendering an expanded multisample surface a texel at a time gets
        // per-sample coverage for nothing and the sample mask for nothing at
        // all — a device's multisample state is what would have applied it,
        // and there is no multisample state here. So the fragment works out
        // which sample it is and discards itself.
        let p = program(&[(Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        let mut layout = Layout::of(&translated, Stage::Fragment);
        layout.coverage = Some(quad_coverage(0b0001, false));
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(
            source.contains("var sample_of_slot = array<u32, 4>(0u, 1u, 2u, 3u);"),
            "{source}"
        );
        assert!(source.contains("u32(input.position.x) % 2u"), "{source}");
        assert!(
            source.contains("if ((1u >> sample) & 1u) == 0u { discard; }"),
            "{source}"
        );
        assert!(braces_balance(&source), "{source}");

        // An all-ones mask excludes nothing, and is what an unprogrammed
        // register reads as — so it must not cost a branch.
        layout.coverage = Some(quad_coverage(u32::MAX, false));
        let open = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!open.contains("& 1u) == 0u"), "{open}");
    }

    #[test]
    fn alpha_to_coverage_keeps_the_same_prefix_the_rasterizer_keeps() {
        // `raster::alpha_coverage` keeps `round(alpha * count)` samples as a
        // prefix rather than as a dither, so half an alpha is samples 0 and 1
        // of four. The comparison here is that rule, sample by sample.
        let p = program(&[(Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        let mut layout = Layout::of(&translated, Stage::Fragment);
        layout.coverage = Some(quad_coverage(u32::MAX, true));
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(source.contains("let target0 = vec4<f32>("), "{source}");
        assert!(
            source.contains("if (sample >= u32(floor(clamp(target0.w, 0.0, 1.0) * 4.0 + 0.5)))"),
            "{source}"
        );
        assert!(source.contains("return target0;"), "{source}");
        assert!(braces_balance(&source), "{source}");

        // And nothing of it is emitted for a draw that does not ask.
        layout.coverage = Some(quad_coverage(u32::MAX, false));
        let plain = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!plain.contains("floor(clamp("), "{plain}");
        assert!(!plain.contains("let target0"), "{plain}");
    }

    #[test]
    fn a_single_sample_draw_carries_no_coverage_at_all() {
        // The whole of it is conditional on a grid with more than one sample
        // in it, which is what every draw into an ordinary surface has.
        let p = program(&[(Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        let layout = Layout::of(&translated, Stage::Fragment);
        assert!(layout.coverage.is_none());
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!source.contains("sample_of_slot"), "{source}");
    }

    #[test]
    fn a_vertex_shader_with_no_attributes_declares_no_input_struct() {
        // WGSL has no empty struct, so one has to be left out rather than
        // emitted empty.
        let p = program(&[(Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        let layout = Layout::of(&translated, Stage::Vertex);
        assert!(layout.attributes.is_empty());
        let source = module(&translated, Stage::Vertex, &layout).unwrap();
        assert!(!source.contains("struct VertexInput"), "{source}");
        assert!(!source.contains("input: VertexInput"), "{source}");
    }

    #[test]
    fn a_colour_channel_the_shader_never_wrote_is_zero_not_a_missing_register() {
        // `shade_fragment` reads r0 to r3 whatever the shader touched, and a
        // module that named a register it never declared would not compile.
        let p = program(&[
            (
                Op::Mov {
                    dst: 0,
                    src: Operand::Imm(0x3f80_0000),
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(
            source.contains("return vec4<f32>(bitcast<f32>(r0), 0.0, 0.0, 0.0);"),
            "{source}"
        );
    }

    #[test]
    fn each_extra_colour_target_takes_the_next_four_registers() {
        let mut entries: Vec<(Op, Pred)> = (0..8u8)
            .map(|r| {
                (
                    Op::Mov {
                        dst: r,
                        src: Operand::Imm(0),
                    },
                    ALWAYS,
                )
            })
            .collect();
        entries.push((Op::Exit, ALWAYS));
        let translated = translate(&program(&entries)).unwrap();
        let mut layout = Layout::of(&translated, Stage::Fragment);
        layout.targets = 2;
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(
            source.contains("@location(1) target1: vec4<f32>"),
            "{source}"
        );
        assert!(
            source.contains(
                "out.target1 = vec4<f32>(bitcast<f32>(r4), bitcast<f32>(r5), \
                             bitcast<f32>(r6), bitcast<f32>(r7));"
            ),
            "{source}"
        );
    }

    #[test]
    fn a_texture_dimension_with_no_binding_is_reported() {
        // The software rasterizer samples every dimension as though it were
        // 2D. A `texture_2d` binding for a 3D texture would be a lie the
        // pipeline finds out about, so this says so instead.
        let p = program(&[
            (
                Op::Texs {
                    dst: 0,
                    dst2: 2,
                    coords: [4, 5, 6],
                    dref: None,
                    handle: 1,
                    // 2D, 2D-array, 3D and cube all bind. A 1D image does
                    // not: `texture_1d` takes no sampler in WebGPU, and the
                    // TIC decoder refuses the type before a draw gets here.
                    dim: TexDim::T1d,
                    mask: [true, true, true, true],
                    f16: false,
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        let layout = Layout::of(&translated, Stage::Fragment);
        assert_eq!(
            module(&translated, Stage::Fragment, &layout).unwrap_err(),
            Unsupported::TextureDimension { dim: TexDim::T1d }
        );
    }

    #[test]
    fn a_module_is_complete_enough_to_stand_on_its_own() {
        // Nothing here can parse WGSL; what this can check is that the four
        // hooks `HOST_INTERFACE` describes are all defined in the module
        // rather than left to a caller, since a module is meant to be handed
        // to a device as it is.
        let (vs, fs) = pair(2);
        for (program, stage) in [(vs, Stage::Vertex), (fs, Stage::Fragment)] {
            let translated = translate(&program).unwrap();
            let layout = Layout::of(&translated, stage);
            let source = module(&translated, stage, &layout).unwrap();
            for hook in ["fn attrIn(", "fn attrOut(", "fn cbRead(", "fn texSample("] {
                assert!(
                    source.contains(hook),
                    "{stage:?} module has no {hook}:\n{source}"
                );
            }
            assert!(braces_balance(&source), "{stage:?}:\n{source}");
        }
    }

    /// Every half-precision op, in the modes that change what is emitted.
    fn half_opcodes() -> Vec<Op> {
        let pair = |asw, bsw, merge| Op::Hadd2 {
            dst: 1,
            a: 2,
            am: NO_MOD,
            asw,
            b: Operand::Reg(3),
            bm: NO_MOD,
            bsw,
            merge,
            ftz: true,
            sat: true,
        };
        vec![
            pair(HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0),
            pair(HSwizzle::H0H0, HSwizzle::F32, HMerge::F32),
            pair(HSwizzle::H1H1, HSwizzle::H1H0, HMerge::MrgH0),
            pair(HSwizzle::H1H0, HSwizzle::H1H0, HMerge::MrgH1),
            Op::Hmul2 {
                dst: 1,
                a: 2,
                am: FMod {
                    neg: true,
                    abs: true,
                },
                asw: HSwizzle::H1H0,
                b: Operand::Const {
                    bank: 1,
                    offset: 0x10,
                },
                bm: NO_MOD,
                bsw: HSwizzle::F32,
                merge: HMerge::H1H0,
                prec: HPrecision::Fmz,
                sat: false,
            },
            Op::Hfma2 {
                dst: 1,
                a: 2,
                asw: HSwizzle::H1H0,
                b: Operand::Reg(3),
                bneg: true,
                bsw: HSwizzle::H1H0,
                c: Operand::Reg(4),
                cneg: false,
                csw: HSwizzle::H1H0,
                merge: HMerge::H1H0,
                prec: HPrecision::Fmz,
                sat: false,
            },
            Op::Hfma2 {
                dst: 1,
                a: 2,
                asw: HSwizzle::H1H0,
                b: Operand::Imm(0x3c00_3c00),
                bneg: false,
                bsw: HSwizzle::H1H0,
                c: Operand::Reg(1),
                cneg: true,
                csw: HSwizzle::H1H0,
                merge: HMerge::H1H0,
                prec: HPrecision::Ftz,
                sat: true,
            },
            Op::Hset2 {
                dst: 1,
                a: 2,
                am: NO_MOD,
                asw: HSwizzle::H1H0,
                b: Operand::Reg(3),
                bm: NO_MOD,
                bsw: HSwizzle::H1H0,
                cmp: FCmp::Gt,
                bop: BoolOp::And,
                src: ALWAYS,
                bf: true,
                ftz: false,
            },
            Op::Hsetp2 {
                p0: 0,
                p1: 1,
                a: 2,
                am: NO_MOD,
                asw: HSwizzle::H1H0,
                b: Operand::Reg(3),
                bm: NO_MOD,
                bsw: HSwizzle::H1H0,
                cmp: FCmp::Lt,
                bop: BoolOp::Or,
                src: IF_P0,
                and: true,
                ftz: true,
            },
            Op::Hsetp2 {
                p0: 0,
                p1: 1,
                a: 2,
                am: NO_MOD,
                asw: HSwizzle::H1H0,
                b: Operand::Reg(3),
                bm: NO_MOD,
                bsw: HSwizzle::H1H0,
                cmp: FCmp::Ne,
                bop: BoolOp::And,
                src: ALWAYS,
                and: false,
                ftz: false,
            },
        ]
    }

    #[test]
    fn every_half_opcode_translates() {
        for op in half_opcodes() {
            let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
            let wgsl = translate(&p)
                .unwrap_or_else(|e| panic!("{op:?}: {e}"))
                .source;
            assert!(braces_balance(&wgsl), "{op:?} left a block open:\n{wgsl}");
        }
    }

    /// A pair of halves is a `vec2<f32>` here, and the pack that puts it back
    /// rounds the way `f32_to_f16` does — so the two backends agree on every
    /// finite result rather than only on the ones that round the same way.
    #[test]
    fn a_half_op_unpacks_its_lanes_and_a_merge_keeps_the_other_one() {
        let ops: Vec<(Op, Pred)> = half_opcodes()
            .into_iter()
            .map(|op| (op, ALWAYS))
            .chain([(Op::Exit, ALWAYS)])
            .collect();
        let wgsl = translate(&program(&ops)).unwrap().source;
        assert!(
            wgsl.contains("pack2x16float(fsat2((hftz(unpack2x16float(r2))"),
            "{wgsl}"
        );
        // The three swizzles that are not the plain pair.
        assert!(wgsl.contains("unpack2x16float(r2).xx"), "{wgsl}");
        assert!(wgsl.contains("unpack2x16float(r2).yy"), "{wgsl}");
        assert!(wgsl.contains("vec2<f32>(bitcast<f32>(r3))"), "{wgsl}");
        // A merge writes one half and leaves the other where it was.
        assert!(wgsl.contains("(r1 & 0xffff0000u) |"), "{wgsl}");
        assert!(wgsl.contains("(r1 & 0x0000ffffu) |"), "{wgsl}");
        // `.fmz` reads both operands twice, so both are bound first.
        assert!(wgsl.contains("== vec2<f32>(0.0)) | ("), "{wgsl}");
        // An f32 lane keeps the f32 flush threshold, not the half one.
        assert!(wgsl.contains("ftz2(vec2<f32>(bitcast<f32>(r3)))"), "{wgsl}");
    }
}
