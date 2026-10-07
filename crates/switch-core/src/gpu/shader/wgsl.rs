//! Translates a lowered program to WGSL. Control flow is a `switch` over a
//! program counter inside a `loop`, with an explicit reconvergence stack, so
//! nothing is restructured. Registers are untyped `u32`s. [`translate`] emits the
//! body against [`HOST_INTERFACE`]; [`module`] wraps it into a complete module.
//! `TRACE_WGSL=<dir>` dumps every module a run uses, for checking with `naga`.

use super::compiled::{Compiled, NO_TARGET};
use super::isa::{
    BoolOp, FCmp, FMod, FRound, HMerge, HPrecision, HSwizzle, ICmp, LogicOp, LopTest, MemSize,
    MufuOp, Op, Operand, Pred, ShflMode, TexDim, TexsStore, XmadC, RZ,
};
use crate::gpu::pipeline::{AttributeBase, Packed1010102};
use crate::gpu::texture::{SwizzleSource, TextureSlot};
use std::collections::BTreeSet;
use std::fmt;

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

/// WGSL helpers, in dependency order; only reached ones are emitted.
const HELPERS: &[(&str, &str)] = &[
    // Local memory `l[]`, a byte-addressed private array; out-of-range bytes read zero.
    (
        "local",
        "\
var<private> local_mem: array<u32, 256>;

fn localByte(i: u32) -> u32 {
  if (i >= 1024u) { return 0u; }
  return (local_mem[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn localWord(i: u32) -> u32 {
  if (i >= 1024u) { return 0u; }
  return localByte(i) | (localByte(i + 1u) << 8u) | (localByte(i + 2u) << 16u)
    | (localByte(i + 3u) << 24u);
}

fn setLocalByte(i: u32, v: u32) {
  let shift = (i & 3u) * 8u;
  local_mem[i >> 2u] = (local_mem[i >> 2u] & ~(0xffu << shift)) | ((v & 0xffu) << shift);
}",
    ),
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

struct Emitter<'a> {
    program: &'a Compiled,
    body: String,
    indent: usize,
    /// Collected while emitting, not by a separate pass.
    regs: BTreeSet<u8>,
    preds: BTreeSet<u8>,
    helpers: BTreeSet<&'static str>,
    uses_carry: bool,
    /// Whether the zero, sign or overflow flags are used.
    uses_flags: bool,
    uses_stack: bool,
    quad: Option<usize>,
    quad_swap: Option<usize>,
    /// The interface the emitted text reaches through.
    loads: BTreeSet<usize>,
    stores: BTreeSet<usize>,
    centroid_loads: BTreeSet<usize>,
    banks: BTreeSet<u8>,
    textures: Vec<(TextureSlot, TexDim, bool)>,
    globals: Vec<(u8, u16)>,
    /// Counter naming `let` bindings.
    temps: usize,
    block: usize,
    /// Textures only `txq`'d, bound as 2D.
    queried: Vec<TextureSlot>,
    texture_offsets: Vec<(i32, i32)>,
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
            uses_flags: false,
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
            block: 0,
            queried: Vec::new(),
            texture_offsets: Vec::new(),
        }
    }

    /// A `ldg` base descriptor from a constant bank plus a register offset:
    ///
    /// ```text
    /// iadd.cout r10, r7, c0[0x110]      // low half, carrying out
    /// iadd.cin  r11, RZ, c0[0x114]      // high half, carrying in
    /// ldg       r10, [r10]
    /// ```
    fn global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        self.indexed_global_base(at, addr)
            .or_else(|| self.direct_global_base(at, addr))
    }

    /// A descriptor loaded whole into the address pair, read with no index:
    ///
    /// ```text
    /// ldc.64 r0, c1[0x40]
    /// ldg.64 r2, [r0]
    /// ```
    fn direct_global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        let lo = self.sole_writer(at, addr)?;
        let hi = self.sole_writer(at, addr.wrapping_add(1))?;
        match (self.program.op(lo), self.program.op(hi)) {
            (
                Op::Ldc {
                    dst,
                    bank,
                    offset,
                    idx: RZ,
                    size: MemSize::B64,
                },
                _,
            ) if lo == hi && dst == addr => Some((bank, (offset as u32 & 0xffff) as u16, RZ)),
            (
                Op::Mov {
                    src: Operand::Const { bank, offset },
                    ..
                },
                Op::Mov {
                    src:
                        Operand::Const {
                            bank: hi_bank,
                            offset: hi_offset,
                        },
                    ..
                },
            ) if hi_bank == bank && hi_offset == offset.wrapping_add(4) => Some((bank, offset, RZ)),
            _ => None,
        }
    }

    /// The indexed form of [`Emitter::global_base`].
    fn indexed_global_base(&self, at: usize, addr: u8) -> Option<(u8, u16, u8)> {
        let (mut lo, mut hi) = (None, None);
        for i in (0..at).rev() {
            match self.program.op(i) {
                Op::Iadd {
                    dst,
                    a,
                    b: Operand::Const { bank, offset },
                    aneg: false,
                    bneg: false,
                    cout: true,
                    ..
                } if dst == addr && lo.is_none() => lo = Some((bank, offset, a)),
                // The high half adds only the carry.
                Op::Iadd {
                    dst,
                    a: RZ,
                    b: Operand::Const { bank, offset },
                    aneg: false,
                    bneg: false,
                    cin: true,
                    ..
                } if dst == addr.wrapping_add(1) && hi.is_none() => hi = Some((bank, offset)),
                // Any other write breaks the pattern.
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
        if hi_bank != bank || hi_offset != offset.wrapping_add(4) {
            return None;
        }
        Some((bank, offset, index))
    }

    /// The constant word a bindless `tex.b` reads its handle from:
    ///
    /// ```text
    /// ldc   r2, c3[0x10]
    /// tex.b r0, r4, r2, 0x2, 2D, 0xf
    /// ```
    fn bindless_slot(&self, at: usize, reg: u8) -> Option<TextureSlot> {
        let writer = self.sole_writer(at, reg)?;
        match self.program.op(writer) {
            Op::Mov {
                src: Operand::Const { bank, offset },
                ..
            } => Some(TextureSlot::Bindless { bank, offset }),
            // Wide loads fill consecutive registers from consecutive words.
            Op::Ldc {
                dst,
                bank,
                offset,
                idx: RZ,
                size,
            } if size.bytes() >= 4 => {
                let word = u32::from(reg.wrapping_sub(dst)) * 4;
                let offset = (offset as u32).wrapping_add(word) & 0xffff;
                Some(TextureSlot::Bindless {
                    bank,
                    offset: offset as u16,
                })
            }
            _ => None,
        }
    }

    /// The unguarded instruction whose value `reg` holds at `at`: the nearest
    /// earlier write in the block, else the program's only write.
    fn sole_writer(&self, at: usize, reg: u8) -> Option<usize> {
        let writes_reg = |i: usize| super::interp::writes(&self.program.op(i)).contains(&reg);
        let writer = match (self.block..at).rev().find(|&i| writes_reg(i)) {
            Some(writer) => writer,
            None => {
                let mut writers = (0..self.program.len()).filter(|&i| i != at && writes_reg(i));
                let only = writers.next()?;
                if writers.next().is_some() {
                    return None;
                }
                only
            }
        };
        (self.program.pred(writer) == Pred::ALWAYS).then_some(writer)
    }

    /// The immediate `reg` holds at `at`. See [`Emitter::sole_writer`].
    fn constant_in(&self, at: usize, reg: u8) -> Option<u32> {
        match self.program.op(self.sole_writer(at, reg)?) {
            Op::Mov32i { imm, .. } => Some(imm),
            Op::Mov {
                src: Operand::Imm(imm),
                ..
            } => Some(imm),
            // A copy of RZ is a zero.
            Op::Mov {
                src: Operand::Reg(RZ),
                ..
            } => Some(0),
            _ => None,
        }
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
        let value = self.runtime_if_non_finite(value, |bits| f32::from_bits(bits).is_finite());
        format!("bitcast<f32>({value})")
    }

    /// Bind non-finite literals to a `let`: WGSL rejects them as constant expressions.
    fn runtime_if_non_finite(&mut self, bits: String, finite: fn(u32) -> bool) -> String {
        match bits.strip_suffix('u').and_then(|n| n.parse::<u32>().ok()) {
            Some(value) if !finite(value) => self.bind(&bits),
            _ => bits,
        }
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

    /// Write a register; `RZ` discards.
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

    /// Write a predicate; `PT` and above are not writable.
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

    /// One source's two lanes, flushed and modified.
    fn half_source(&mut self, bits: String, m: FMod, sw: HSwizzle, ftz: bool) -> String {
        let bits = match sw {
            HSwizzle::F32 => {
                self.runtime_if_non_finite(bits, |bits| f32::from_bits(bits).is_finite())
            }
            _ => self.runtime_if_non_finite(bits, |bits| {
                [bits, bits >> 16]
                    .iter()
                    .all(|half| (half >> 10) & 0x1f != 0x1f)
            }),
        };
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

    /// `.fmz` zeroing as a lane-wise condition.
    fn half_zeroed(&mut self, a: &str, b: &str) -> String {
        format!("(({a} == vec2<f32>(0.0)) | ({b} == vec2<f32>(0.0)))")
    }

    /// Each lane's comparison combined with the source predicate.
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
        // WGSL has no `isNan`.
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

    fn set_result(&mut self, taken: &str, bf: bool) -> String {
        let one = if bf { "0x3f800000u" } else { "0xffffffffu" };
        format!("select(0u, {one}, {taken})")
    }

    fn round(&mut self, mode: FRound, value: String) -> String {
        // WGSL's `round` breaks ties to even, matching `.rn`.
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
            Op::Rro { dst, src, sm } => {
                let x = self.operand_f(src);
                let x = self.fmod(sm, x);
                self.set_f(dst, &x);
            }
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
                // The pre-scale multiplies the first operand.
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
                // True picks the minimum. NaN handling matches the interpreter.
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
                // Each lane fills its half: 1.0h with `.bf`, all ones without.
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
                // Two adds: the carry is whether either wrapped.
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
            Op::Vmnmx {
                dst,
                a,
                b,
                c,
                max,
                then_max,
                signed,
                then_signed,
            } => {
                let pick = |x: &str, y: &str, max: bool, signed: bool| {
                    let op = if max { "max" } else { "min" };
                    if signed {
                        format!("bitcast<u32>({op}(bitcast<i32>({x}), bitcast<i32>({y})))")
                    } else {
                        format!("{op}({x}, {y})")
                    }
                };
                let (x, y, z) = (self.r(a), self.r(b), self.r(c));
                let first = self.bind(&pick(&x, &y, max, signed));
                self.set_r(dst, &pick(&first, &z, then_max, then_signed));
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
                // `.mrg` replaces the high half with `b`'s low half.
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
                    // Rounds as `f32_to_f16` does.
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
                cc,
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
                // Bound first: the flags read it and `dst` may be `RZ`.
                let value = self.bind(&value);
                self.set_r(dst, &value);
                if cc {
                    self.uses_carry = true;
                    self.uses_flags = true;
                    self.line(&format!("ccZ = ({value} == 0u);"));
                    self.line(&format!("ccS = (bitcast<i32>({value}) < 0);"));
                    self.line("carry = false;");
                    self.line("ccO = false;");
                }
            }

            // ---- moves ----
            Op::Mov { dst, src } => {
                let value = self.operand(src);
                self.set_r(dst, &value);
            }
            Op::Mov32i { dst, imm } => self.set_r(dst, &format!("{imm}u")),
            Op::S2r { dst, .. } => {
                // Lane and thread identities are zero, as in the interpreter.
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
            Op::Csetp {
                p0,
                p1,
                test,
                src,
                op: bop,
            } => {
                self.uses_carry = true;
                self.uses_flags = true;
                let flags = ["ccZ", "ccS", "carry", "ccO"].map(str::to_string);
                let Some(passed) = super::isa::flow_test(test, flags, &TextLogic) else {
                    return Err(Unsupported::Op { at, op });
                };
                let passed = self.bind(&passed);
                let source = self.holds(src);
                let a = self.combine(bop, &passed, &source);
                let b = self.combine(bop, &format!("!{passed}"), &source);
                self.set_p(p0, &a);
                self.set_p(p1, &b);
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
            // A `texs` array keeps its layer in the third coordinate register.
            Op::Texs {
                coords,
                dref,
                handle,
                dim,
                ..
            } => {
                let layer = (dim == TexDim::T2dArray).then_some(coords[2]);
                let slot = TextureSlot::Bound(handle);
                self.sample_texture(at, slot, dim, dref, coords, layer)?;
            }
            // `tex` keeps the layer before the coordinates; `.LL`/`.LB` sample the one level.
            Op::Tex {
                coords,
                layer,
                dref,
                offset,
                handle,
                handle_reg,
                dim,
                ..
            } => {
                let slot = match handle_reg {
                    None => TextureSlot::Bound(handle),
                    Some(reg) => self
                        .bindless_slot(at, reg)
                        .ok_or(Unsupported::UntracedHandle { at })?,
                };
                match offset {
                    None => self.sample_texture(at, slot, dim, dref, coords, layer)?,
                    Some(reg) => {
                        self.sample_offset(at, slot, dim, dref, coords, layer, reg, op)?;
                    }
                }
            }
            Op::Txq { lod, handle, .. } => {
                self.query_texture(at, TextureSlot::Bound(handle), lod);
            }
            // WGSL needs a constant gather offset.
            Op::Tld4 {
                coords,
                layer,
                offset: None,
                handle,
                dim,
                component,
                ..
            } => self.gather_texture(
                at,
                TextureSlot::Bound(handle),
                dim,
                coords,
                layer,
                component,
            )?,

            // `shfl` maps onto `quadSwapX`/`Y`/`Diagonal`, mirroring `interp::shuffle_source`.
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
                // Out-of-quad lanes keep their own value.
                let reachable = format!("({ok} && {from} < 4)");
                self.set_r(dst, &format!("select({here}, {peer}, {reachable})"));
                self.set_p(pred, &ok);
            }

            // `fswzadd` needs only this lane's index, not another lane's value.
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

            // `ldg` from a constant bank descriptor binds a buffer; other global
            // and shared memory, and barriers, are unsupported.
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

            Op::Ldl {
                dst,
                addr,
                offset,
                size,
            } => {
                self.helpers.insert("local");
                let base = self.local_address(addr, offset);
                let value = |word: u32| format!("localWord({base} + {}u)", word * 4);
                match size {
                    MemSize::U8 => self.set_r(dst, &format!("localByte({base})")),
                    MemSize::S8 => self.set_r(
                        dst,
                        &format!(
                            "bitcast<u32>(extractBits(bitcast<i32>(localByte({base})), 0u, 8u))"
                        ),
                    ),
                    MemSize::U16 | MemSize::S16 => {
                        let half = format!("(localByte({base}) | (localByte({base} + 1u) << 8u))");
                        let half = if size == MemSize::S16 {
                            format!("bitcast<u32>(extractBits(bitcast<i32>({half}), 0u, 16u))")
                        } else {
                            half
                        };
                        self.set_r(dst, &half);
                    }
                    _ => {
                        for word in 0..u32::from(size.regs()) {
                            self.set_r(dst.wrapping_add(word as u8), &value(word));
                        }
                    }
                }
            }
            Op::Stl {
                addr,
                offset,
                src,
                size,
            } => {
                self.helpers.insert("local");
                let base = self.local_address(addr, offset);
                let len = size.bytes();
                let words: Vec<String> = (0..size.regs())
                    .map(|i| self.r(src.wrapping_add(i)))
                    .collect();
                // Out of range drops the whole store, as the interpreter does.
                self.line(&format!("if ({base} + {len}u <= 1024u) {{"));
                self.indent += 1;
                for byte in 0..len {
                    let word = &words[(byte / 4) as usize];
                    let shift = (byte % 4) * 8;
                    self.line(&format!(
                        "setLocalByte({base} + {byte}u, {word} >> {shift}u);"
                    ));
                }
                self.indent -= 1;
                self.line("}");
            }
            Op::Stg { .. }
            | Op::Lds { .. }
            | Op::Sts { .. }
            | Op::Atom { .. }
            | Op::Bar { .. }
            // A ballot needs the warp, which is only a quad here.
            | Op::Vote { .. }
            | Op::Suld { .. }
            | Op::Sust { .. }
            | Op::Unimplemented { .. } => return Err(Unsupported::Op { at, op }),

            // Handled by `emit_terminator` and `emit_instruction`.
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
            // `tex.aoffi` with a non-constant offset is the rasterizer's.
            Op::Tld4 { .. } => return Err(Unsupported::Op { at, op }),
        }
        Ok(())
    }

    /// `a[offset + Rn]`'s byte address, wrapping at 16 bits.
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

    /// A conversion's source byte lane, narrowed and extended back to 32 bits.
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

const COMPONENT: [&str; 4] = ["x", "y", "z", "w"];

impl Emitter<'_> {
    /// Record that the program samples `slot` as `dim`, a shadow map or not.
    fn bind_texture(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        compare: bool,
    ) -> Result<(), Unsupported> {
        match self.textures.iter().find(|&&(seen, _, _)| seen == slot) {
            // A slot sampled both as colour and depth cannot be one binding.
            Some(&(_, _, was)) if was != compare => Err(Unsupported::DepthCompare { at }),
            Some(_) => Ok(()),
            None => {
                self.textures.push((slot, dim, compare));
                Ok(())
            }
        }
    }

    /// `tex.aoffi`: a sample offset by an immediate texel offset (signed nibbles, x low).
    #[allow(clippy::too_many_arguments)]
    fn sample_offset(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        dref: Option<u8>,
        coords: [u8; 3],
        layer: Option<u8>,
        offset: u8,
        op: Op,
    ) -> Result<(), Unsupported> {
        let refuse = Unsupported::Op { at, op };
        if dref.is_some() || !matches!(dim, TexDim::T2d | TexDim::T2dArray) {
            return Err(refuse);
        }
        let packed = self.constant_in(at, offset).ok_or(refuse)?;
        let axis = |shift: u32| ((packed >> shift) as i32) << 28 >> 28;
        let texel = (axis(0), axis(4));
        let index = match self.texture_offsets.iter().position(|&seen| seen == texel) {
            Some(index) => index,
            None => {
                self.texture_offsets.push(texel);
                self.texture_offsets.len() - 1
            }
        };
        self.bind_texture(at, slot, dim, false)?;
        let u = self.f(coords[0]);
        let v = self.f(coords[1]);
        let layer = match layer {
            Some(reg) => {
                let reg = self.r(reg);
                format!("({reg} & 0xffffu)")
            }
            None => "0u".to_string(),
        };
        let color = self.bind(&format!(
            "texSampleOffset({}u, {index}u, {u}, {v}, {layer})",
            slot.key()
        ));
        for (reg, store, _) in self.program.texs_writes(at).to_vec() {
            if let TexsStore::Float(channel) = store {
                self.set_f(reg, &format!("{color}.{}", COMPONENT[channel]));
            }
        }
        Ok(())
    }

    /// `txq`: texture size as integers.
    fn query_texture(&mut self, at: usize, slot: TextureSlot, lod: u8) {
        self.queried.push(slot);
        let lod = self.r(lod);
        let size = self.bind(&format!("texDims({}u, {lod})", slot.key()));
        for (reg, store, _) in self.program.texs_writes(at).to_vec() {
            if let TexsStore::Float(channel) = store {
                self.set_r(reg, &format!("{size}.{}", COMPONENT[channel]));
            }
        }
    }

    /// `tld4`: one channel of the four bilinear texels, as `textureGather`.
    fn gather_texture(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        coords: [u8; 3],
        layer: Option<u8>,
        component: u8,
    ) -> Result<(), Unsupported> {
        self.bind_texture(at, slot, dim, false)?;
        let u = self.f(coords[0]);
        let v = self.f(coords[1]);
        let layer = match layer {
            Some(reg) => {
                let reg = self.r(reg);
                format!("({reg} & 0xffffu)")
            }
            None => "0u".to_string(),
        };
        let texels = self.bind(&format!(
            "texGather({}u, {component}u, {u}, {v}, {layer})",
            slot.key()
        ));
        for (reg, store, _) in self.program.texs_writes(at).to_vec() {
            if let TexsStore::Float(channel) = store {
                self.set_f(reg, &format!("{texels}.{}", COMPONENT[channel]));
            }
        }
        Ok(())
    }

    /// Sample and store the channels, for `texs` and `tex`.
    fn sample_texture(
        &mut self,
        at: usize,
        slot: TextureSlot,
        dim: TexDim,
        dref: Option<u8>,
        coords: [u8; 3],
        layer: Option<u8>,
    ) -> Result<(), Unsupported> {
        let compare = dref.is_some();
        self.bind_texture(at, slot, dim, compare)?;
        let key = slot.key();
        let u = self.f(coords[0]);
        let v = match dim {
            TexDim::T1d => "0.0".to_string(),
            _ => self.f(coords[1]),
        };
        // The layer is an integer in the register's low half.
        let layer = match layer {
            Some(reg) => {
                let reg = self.r(reg);
                format!("({reg} & 0xffffu)")
            }
            None => "0u".to_string(),
        };
        // A 3D third coordinate is normalized; an array's is a layer.
        let w = match dim {
            TexDim::T3d | TexDim::TCube | TexDim::TCubeArray => self.f(coords[2]),
            _ => "0.0".to_string(),
        };
        let code = tex_dim_code(dim);
        let color = match dref {
            // A shadow sample fills every requested channel except alpha.
            Some(reg) => {
                let reference = self.f(reg);
                self.bind(&format!(
                    "texSampleCompare({key}u, {code}u, {u}, {v}, {layer}, {reference})"
                ))
            }
            None => self.bind(&format!(
                "texSample({key}u, {code}u, {u}, {v}, {layer}, {w})"
            )),
        };
        // Stored now rather than at first use like the interpreter; equivalent
        // unless the destination is overwritten before being read.
        let writes = self.program.texs_writes(at).to_vec();
        for (reg, store, _) in writes {
            match store {
                TexsStore::Float(channel) => {
                    self.set_f(reg, &format!("{color}.{}", COMPONENT[channel]));
                }
                // `.F16` packs two channels as halves.
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
        Ok(())
    }
}

impl Emitter<'_> {
    /// One `case` per basic block, in order.
    fn emit_blocks(&mut self, leaders: &[usize]) -> Result<(), Unsupported> {
        for (n, &start) in leaders.iter().enumerate() {
            let end = leaders.get(n + 1).copied().unwrap_or(self.program.len());
            self.block = start;
            self.indent = 3;
            self.line(&format!("case {start}u: {{"));
            self.indent = 4;
            for at in start..end {
                self.emit_instruction(at, end)?;
            }
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
        // A push falls through; its `PT` guard bits hold the target.
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

    /// A guarded terminator must say where control goes when the guard fails.
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
                // A target on a `sched` word means the block's first instruction.
                let slot = self.bind(&format!("select({raw}, {raw} + 8u, ({raw} & 31u) == 0u)"));
                self.line(&format!("switch ({slot}) {{"));
                self.indent += 1;
                for target in targets {
                    let offset = self.program.offset(target as usize);
                    self.line(&format!("case {offset}u: {{ pc = {target}u; }}"));
                }
                // No way to raise the interpreter's error, so end the invocation.
                self.line("default: { return false; }");
                self.indent -= 1;
                self.line("}");
            }
            _ => unreachable!("emit_jump called with {op:?}"),
        }
        Ok(())
    }

    /// A local-memory byte address; `0xffffffffu` marks offsets past 2^31.
    fn local_address(&mut self, addr: u8, offset: i32) -> String {
        let reg = self.r(addr);
        self.bind(&format!(
            "select(0xffffffffu, bitcast<u32>(bitcast<i32>({reg}) + ({offset})), {reg} < 0x80000000u)"
        ))
    }

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
        // Only registers are visible to the caller.
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
        if self.uses_flags {
            out.push_str(
                "  var ccZ: bool = false;\n  var ccS: bool = false;\n  var ccO: bool = false;\n",
            );
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
        // Required by WGSL; the decoder rejects programs with no `exit`.
        out.push_str("      default: { return false; }\n");
        out.push_str("    }\n");
        out.push_str("  }\n");
        // Unreachable, but WGSL requires a final return.
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

/// Module wiring as slot and bank numbers, mostly read off the program.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// Generic slots a vertex shader loads, by `@location`, each four floats.
    pub attributes: Vec<usize>,
    /// Integer-format attribute slots, declared `vec4<i32>`/`vec4<u32>`; filled from the draw.
    pub integer_attributes: Vec<(usize, AttributeBase)>,
    /// Generic slots passed from vertex to fragment.
    pub varyings: Vec<usize>,
    /// Varyings read with `ipa.centroid`; both stages must agree.
    pub centroid_varyings: Vec<usize>,
    pub const_banks: Vec<u8>,
    pub textures: Vec<TextureBinding>,
    /// See [`Translation::texture_offsets`].
    pub texture_offsets: Vec<(i32, i32)>,
    pub globals: Vec<(u8, u16)>,
    /// Colour targets written, from `r0` in fours. Zero is a depth-only pass.
    pub targets: u32,
    /// This module's bind group; each stage needs its own.
    pub group: u32,
    /// Whether the vertex entry negates `position.y`: when the guest's viewport
    /// does *not* mirror y, since WebGPU's NDC-to-framebuffer already does.
    ///
    /// Setting this from `pipeline::Viewport::flip_y` directly flips twice.
    /// It looks almost right, because a full-screen quad is symmetric about
    /// the centre and so is most of a UI: the Home Menu came out 94.87%
    /// correct that way, with one off-centre band mirrored onto the other
    /// side of the screen.
    pub flip_y: bool,
    /// Whether to remap z from GL's `-w..w` onto WebGPU's `0..w`.
    pub depth_minus_one_to_one: bool,
    /// Attribute slots fetched as BGRA; filled from the draw.
    pub bgra_attributes: Vec<usize>,
    /// Attribute slots packed as 10-10-10-2; filled from the draw.
    pub packed_attributes: Vec<(usize, Packed1010102)>,
    /// Coverage for a backend rendering multisampled surfaces per texel.
    pub coverage: Option<Coverage>,
}

/// Per-sample coverage for an expanded multisample surface: sample mask and
/// alpha-to-coverage, from the draw's [`crate::gpu::surface::SampleGrid`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    pub samples_x: u32,
    pub samples_y: u32,
    /// The sample each texel of a pixel's tile holds, by `dy * samples_x + dx`.
    pub sample_of_slot: Vec<u32>,
    pub sample_mask: u32,
    pub alpha_to_coverage: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureBinding {
    pub slot: TextureSlot,
    /// What the instruction samples it as.
    pub dim: TexDim,
    /// The descriptor's channel swizzle, filled from the TIC by the backend.
    pub swizzle: [SwizzleSource; 4],
    /// A shadow map, bound as `texture_depth_*` with a `sampler_comparison`.
    pub compare: bool,
}

pub const IDENTITY_SWIZZLE: [SwizzleSource; 4] = [
    SwizzleSource::R,
    SwizzleSource::G,
    SwizzleSource::B,
    SwizzleSource::A,
];

const ATTRIBUTE_WORDS: usize = 0x400 / 4;
/// `GENERIC_BASE + n * GENERIC_STRIDE + c * 4` addresses slot `n`, component `c`.
const GENERIC_BASE: usize = 0x80;
const GENERIC_STRIDE: usize = 0x10;
const GENERIC_SLOTS: usize = 32;
/// Clip position. Its `w` is also the fragment shader's `1/w` input.
const POSITION: usize = 0x70;
/// `InstanceId` then `VertexId`.
const INSTANCE_ID: usize = 0x2f8;
const VERTEX_ID: usize = 0x2fc;

/// Bank `b` binds at `b`; texture `i` at `TEXTURE_BINDING + 2i`, sampler beside it.
const TEXTURE_BINDING: u32 = 32;

pub const GLOBAL_BINDING: u32 = 96;

impl Layout {
    /// Read a program's interface off its translation.
    pub fn of(translated: &Translation, stage: Stage) -> Layout {
        let (attributes, varyings) = match stage {
            Stage::Vertex => (translated.loads.clone(), translated.stores.clone()),
            // Fragment `a[]` stores go nowhere.
            Stage::Fragment => (Vec::new(), translated.loads.clone()),
        };
        Layout {
            attributes,
            integer_attributes: Vec::new(),
            varyings,
            // Copied from the fragment stage by the backend.
            centroid_varyings: match stage {
                Stage::Vertex => Vec::new(),
                Stage::Fragment => translated.centroid_loads.clone(),
            },
            const_banks: translated.const_banks.clone(),
            textures: translated
                .textures
                .iter()
                .map(|&(slot, dim, compare)| TextureBinding {
                    slot,
                    dim,
                    compare,
                    swizzle: IDENTITY_SWIZZLE,
                })
                .collect(),
            texture_offsets: translated.texture_offsets.clone(),
            globals: translated.globals.clone(),
            targets: 1,
            group: 0,
            // Both from the draw's viewport.
            flip_y: false,
            depth_minus_one_to_one: false,
            bgra_attributes: Vec::new(),
            packed_attributes: Vec::new(),
            coverage: None,
        }
    }

    /// The sampling half of a varying's `@interpolate`: `", centroid"` or nothing.
    fn sampling(&self, slot: usize) -> &'static str {
        if self.centroid_varyings.contains(&slot) {
            ", centroid"
        } else {
            ""
        }
    }

    /// How slot `slot` is packed, if as one 10-10-10-2 word.
    pub fn packing(&self, slot: usize) -> Option<Packed1010102> {
        self.packed_attributes
            .iter()
            .find(|&&(at, _)| at == slot)
            .map(|&(_, packing)| packing)
    }

    /// What slot `slot` arrives as; float unless the draw recorded an integer format.
    pub fn attribute_base(&self, slot: usize) -> AttributeBase {
        self.integer_attributes
            .iter()
            .find(|&&(at, _)| at == slot)
            .map_or(AttributeBase::Float, |&(_, base)| base)
    }
}

/// The four `a[]` words a 10-10-10-2 attribute unpacks to, as `raster::fetch_attribute` does.
fn unpack_1010102(word: &str, packing: Packed1010102) -> [String; 4] {
    let fields = [(0u32, 10u32), (10, 10), (20, 10), (30, 2)];
    fields.map(|(offset, bits)| {
        let signed = format!("extractBits(bitcast<i32>({word}), {offset}u, {bits}u)");
        let unsigned = format!("extractBits({word}, {offset}u, {bits}u)");
        let largest = (1u32 << bits) - 1;
        match packing {
            Packed1010102::Snorm => {
                let positive = (1u32 << (bits - 1)) - 1;
                format!("max(f32({signed}) / {positive}.0, -1.0)")
            }
            Packed1010102::Unorm => format!("f32({unsigned}) / {largest}.0"),
            Packed1010102::Sint => format!("bitcast<f32>({signed})"),
            Packed1010102::Uint => format!("bitcast<f32>({unsigned})"),
        }
    })
}

fn attribute_scalar(base: AttributeBase) -> &'static str {
    match base {
        AttributeBase::Float => "f32",
        AttributeBase::Sint => "i32",
        AttributeBase::Uint => "u32",
    }
}

fn generic_slot(offset: u16) -> Option<usize> {
    let offset = usize::from(offset);
    if (GENERIC_BASE..GENERIC_BASE + GENERIC_SLOTS * GENERIC_STRIDE).contains(&offset) {
        Some((offset - GENERIC_BASE) / GENERIC_STRIDE)
    } else {
        None
    }
}

/// `quadSwapX`/`Y`/`Diagonal` from fine derivatives, for devices without quad
/// operations (every browser). Exact because 16-bit halves survive the f32 round trip.
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

/// Wrap a translation into a complete shader module: bindings, attribute
/// space and entry point. Varyings are `@interpolate(linear)` carrying value/w,
/// because Maxwell's `ipa` finishes the perspective divide itself.
pub fn module(
    translated: &Translation,
    stage: Stage,
    layout: &Layout,
) -> Result<String, Unsupported> {
    let mut out = String::new();
    // A quad outside the fragment stage is unsupported.
    if let Some(at) = translated.quad {
        if stage != Stage::Fragment {
            return Err(Unsupported::Quad { at });
        }
    }
    // Directives first, then the lane index.
    if translated.quad_swap.is_some() {
        if translated.subgroups {
            if translated.subgroup_enable {
                out.push_str("enable subgroups;\n\n");
            }
        } else {
            // Derivatives sit inside the dispatch loop, so silence uniformity analysis.
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
        // Out of range reads zero.
        out.push_str("    default: { return 0u; }\n  }\n}\n\n");
    }

    // `a[]` in and out are separate so they do not alias.
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
    // An unbound bank reads zero.
    out.push_str("    default: { return 0u; }\n  }\n}\n\n");

    // `dim` is unused here but part of `HOST_INTERFACE`.
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
            TexDim::TCubeArray => "vec3<f32>(u, v, w), layer, 0.0",
            other => return Err(Unsupported::TextureDimension { dim: other }),
        };
        let key = texture.slot.key();
        let sample = format!("textureSampleLevel(tex{index}, smp{index}, {coords})");
        out.push_str(&format!(
            "    case {key}u: {{ {} }}\n",
            return_swizzled(&sample, texture.swizzle)
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }\n  }\n}\n\n");

    // Shadow samples: alpha one, the comparison in the rest, as Eden's `Extract`.
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
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      let c = textureSampleCompareLevel(tex{index}, \
             smp{index}, {coords}, dref);\n      return vec4<f32>(c, c, c, 1.0);\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n  }\n}\n\n");

    // One case per binding and offset: offsets must be constants.
    out.push_str(
        "fn texSampleOffset(imm: u32, offset: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v), 0.0",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer, 0.0",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!("    case {key}u: {{\n      switch (offset) {{\n"));
        for (n, (x, y)) in layout.texture_offsets.iter().enumerate() {
            let sample = format!(
                "textureSampleLevel(tex{index}, smp{index}, {coords}, vec2<i32>({x}, {y}))"
            );
            out.push_str(&format!(
                "        case {n}u: {{ {} }}\n",
                return_swizzled(&sample, texture.swizzle)
            ));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    // `txq` sizes, as `interp`'s `run_txq` answers.
    out.push_str("fn texDims(imm: u32, lod: u32) -> vec4<u32> {\n  switch (imm) {\n");
    for (index, texture) in layout.textures.iter().enumerate() {
        let depth = match texture.dim {
            TexDim::T2dArray => format!("textureNumLayers(tex{index})"),
            TexDim::T3d => "s.z".to_string(),
            TexDim::TCube => "6u".to_string(),
            TexDim::TCubeArray => format!("textureNumLayers(tex{index}) * 6u"),
            _ => "1u".to_string(),
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      let s = textureDimensions(tex{index});\n      \
             return vec4<u32>(select(max(s.x >> lod, 1u), 1u, lod >= 32u), \
             select(max(s.y >> lod, 1u), 1u, lod >= 32u), {depth}, \
             textureNumLevels(tex{index}));\n    }}\n"
        ));
    }
    out.push_str("    default: { return vec4<u32>(1u, 1u, 1u, 1u); }\n  }\n}\n\n");

    // One call per gather channel, through the descriptor's swizzle.
    out.push_str(
        "fn texGather(imm: u32, component: u32, u: f32, v: f32, layer: u32) -> vec4<f32> {\n  \
         switch (imm) {\n",
    );
    for (index, texture) in layout.textures.iter().enumerate() {
        let coords = match (texture.dim, texture.compare) {
            (TexDim::T2d, false) => "vec2<f32>(u, v)",
            (TexDim::T2dArray, false) => "vec2<f32>(u, v), layer",
            _ => continue,
        };
        let key = texture.slot.key();
        out.push_str(&format!(
            "    case {key}u: {{\n      switch (component) {{\n"
        ));
        for (channel, source) in texture.swizzle.iter().enumerate() {
            let texels = match source {
                SwizzleSource::R => format!("textureGather(0, tex{index}, smp{index}, {coords})"),
                SwizzleSource::G => format!("textureGather(1, tex{index}, smp{index}, {coords})"),
                SwizzleSource::B => format!("textureGather(2, tex{index}, smp{index}, {coords})"),
                SwizzleSource::A => format!("textureGather(3, tex{index}, smp{index}, {coords})"),
                SwizzleSource::Zero => "vec4<f32>(0.0)".to_string(),
                SwizzleSource::One => "vec4<f32>(1.0)".to_string(),
            };
            out.push_str(&format!(
                "        case {channel}u: {{ return {texels}; }}\n"
            ));
        }
        out.push_str("        default: { return vec4<f32>(0.0); }\n      }\n    }\n");
    }
    out.push_str("    default: { return vec4<f32>(0.0); }\n  }\n}\n\n");

    out.push_str(&translated.source);
    out.push('\n');
    out.push_str(&match stage {
        Stage::Vertex => vertex_entry(layout),
        Stage::Fragment => fragment_entry(translated, layout),
    });
    Ok(out)
}

/// [`super::isa::flow_test`] as WGSL.
struct TextLogic;

impl super::isa::FlowLogic<String> for TextLogic {
    fn constant(&self, value: bool) -> String {
        value.to_string()
    }
    fn not(&self, a: String) -> String {
        format!("!({a})")
    }
    fn and(&self, a: String, b: String) -> String {
        format!("({a} && {b})")
    }
    fn or(&self, a: String, b: String) -> String {
        format!("({a} || {b})")
    }
    fn xor(&self, a: String, b: String) -> String {
        format!("({a} != {b})")
    }
}

/// `return` a sample rearranged by a descriptor swizzle.
fn return_swizzled(sample: &str, swizzle: [SwizzleSource; 4]) -> String {
    if swizzle == IDENTITY_SWIZZLE {
        return format!("return {sample};");
    }
    let channels: Vec<&str> = swizzle
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
    format!(
        "let sampled = {sample}; return vec4<f32>({});",
        channels.join(", ")
    )
}

/// The WGSL type a `texs` of this dimensionality samples.
fn texture_type(dim: TexDim, compare: bool) -> Result<&'static str, Unsupported> {
    match (dim, compare) {
        (TexDim::T2d, false) => Ok("texture_2d<f32>"),
        (TexDim::T2dArray, false) => Ok("texture_2d_array<f32>"),
        (TexDim::T3d, false) => Ok("texture_3d<f32>"),
        (TexDim::TCube, false) => Ok("texture_cube<f32>"),
        (TexDim::TCubeArray, false) => Ok("texture_cube_array<f32>"),
        (TexDim::T2d, true) => Ok("texture_depth_2d"),
        (TexDim::T2dArray, true) => Ok("texture_depth_2d_array"),
        (dim, _) => Err(Unsupported::TextureDimension { dim }),
    }
}

fn generic_word(slot: usize, component: usize) -> usize {
    (GENERIC_BASE + slot * GENERIC_STRIDE) / 4 + component
}

fn gather(array: &str, base: usize) -> String {
    let words: Vec<String> = (0..4).map(|c| format!("{array}[{}u]", base + c)).collect();
    format!("vec4<f32>({})", words.join(", "))
}

fn vertex_entry(layout: &Layout) -> String {
    let mut out = String::new();
    if !layout.attributes.is_empty() {
        out.push_str("struct VertexInput {\n");
        for slot in &layout.attributes {
            if layout.packing(*slot).is_some() {
                out.push_str(&format!("  @location({slot}) attr{slot}: u32,\n"));
                continue;
            }
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
        if let Some(packing) = layout.packing(*slot) {
            for (component, value) in unpack_1010102(&format!("input.attr{slot}"), packing)
                .iter()
                .enumerate()
            {
                out.push_str(&format!(
                    "  attr_in[{}u] = {value};\n",
                    generic_word(*slot, component)
                ));
            }
            continue;
        }
        // Integer attributes are carried as bits.
        let integer = layout.attribute_base(*slot) != AttributeBase::Float;
        // BGRA swaps the first and third components.
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
    // An unwritten clip position is (0, 0, 0, 1).
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
        // The fragment stage receives value/w.
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
                    // An unwritten channel is zero.
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
    // Coverage first: the sample mask applies before shading.
    if let Some(coverage) = &layout.coverage {
        out.push_str(&sample_index(coverage));
        if coverage.sample_mask != u32::MAX {
            out.push_str(&format!(
                "  if (({}u >> sample) & 1u) == 0u {{ discard; }}\n",
                coverage.sample_mask
            ));
        }
    }
    // Quad lane numbered as `raster::quad_pixel`: row parity above column parity.
    if translated.quad.is_some() {
        out.push_str(
            "  quad_lane = (u32(input.position.y) & 1u) * 2u + (u32(input.position.x) & 1u);\n",
        );
    }
    // Fragment `position.w` is `1/w`, as `a[0x7c]` holds.
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
    // Alpha-to-coverage applies after shading.
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
        // Shaded even depth-only, for alpha-to-coverage.
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

/// Which sample of its pixel this fragment is, by table lookup.
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

/// Samples kept for `alpha`: a prefix of `round(alpha * count)`.
fn alpha_coverage(coverage: &Coverage, alpha: &str) -> String {
    let count = coverage.samples_x * coverage.samples_y;
    // `floor(x + 0.5)`: WGSL's `round` ties to even, Rust's away from zero.
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
    /// `@p0`: the guard a two-armed branch is built out of.
    const IF_P0: Pred = Pred {
        reg: 0,
        negate: false,
    };
    const NO_MOD: FMod = FMod::NONE;

    /// The byte offset of instruction `index` in a 32-byte-block layout.
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

    /// Non-finite immediates go through a `let`.
    #[test]
    fn a_non_finite_immediate_is_converted_at_run_time() {
        let fadd = |bits: u32| Op::Fadd {
            dst: 1,
            a: 2,
            am: NO_MOD,
            b: Operand::Imm(bits),
            bm: NO_MOD,
            ftz: false,
            sat: false,
        };
        let source = |op: Op| {
            translate(&program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]))
                .unwrap()
                .source
        };
        for bits in [0x7f80_0000u32, 0xff80_0000, 0x7fc0_0000] {
            let wgsl = source(fadd(bits));
            assert!(!wgsl.contains(&format!("bitcast<f32>({bits}u)")), "{wgsl}");
            assert!(wgsl.contains(&format!(" = {bits}u;")), "{wgsl}");
        }
        let one = source(fadd(0x3f80_0000));
        assert!(one.contains("bitcast<f32>(1065353216u)"), "{one}");
    }

    /// Whether braces balance.
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

    /// One of every opcode the Home Menu's shaders use.
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
        // Control-flow opcodes are covered by the tests below.
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
        // Without the `else`, the block would loop forever.
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
        // Byte offsets in, block indices out.
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
        // `ldg` without a traceable descriptor is unsupported.
        let op = Op::Ldg {
            dst: 1,
            addr: 2,
            offset: 0,
            size: MemSize::B32,
        };
        let p = program(&[(op, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(translate(&p).unwrap_err(), Unsupported::Op { at: 0, op });
    }

    /// A `ldg` from a constant bank descriptor plus an index.
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

        // Any other address is unsupported.
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
        // `shfl` maps onto quad swaps.
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
        // With device quad operations, only the enable and lane are added.
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
        // Without them the module defines the swaps from derivatives.
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
        assert!(
            source.starts_with("diagnostic(off, derivative_uniformity);"),
            "{source}"
        );
        assert!(!source.contains("enable subgroups"), "{source}");
        for wanted in ["fn quadSwapX(", "dpdxFine(", "dpdyFine("] {
            assert!(source.contains(wanted), "{wanted} missing from {source}");
        }

        // Only fragment shaders have derivatives.
        let layout = Layout::of(&translated, Stage::Vertex);
        assert_eq!(
            module(&translated, Stage::Vertex, &layout).unwrap_err(),
            Unsupported::Quad { at: 0 }
        );
    }

    #[test]
    fn fswzadd_asks_which_lane_it_is_and_nothing_of_the_device() {
        // `fswzadd` reads no other lane, so needs no device support.
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

        // A vertex shader has no lane index.
        let layout = Layout::of(&translated, Stage::Vertex);
        assert_eq!(
            module(&translated, Stage::Vertex, &layout).unwrap_err(),
            Unsupported::Quad { at: 0 }
        );
    }

    #[test]
    fn an_undecoded_branch_target_is_reported_before_anything_is_emitted() {
        let p = program(&[(Op::Bra { target: 0x9999 }, ALWAYS), (Op::Exit, ALWAYS)]);
        assert_eq!(
            translate(&p).unwrap_err(),
            Unsupported::UndecodedTarget { at: 0 }
        );
    }

    #[test]
    fn a_block_starts_at_every_branch_target() {
        // A branch-only target gets its own case.
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
        // Fragment colour is `r0`..`r3`, so registers must outlive `run`.
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
                    dst: 1,
                    src: Operand::Reg(RZ),
                },
                ALWAYS,
            ),
            (
                Op::Mov {
                    dst: RZ,
                    src: Operand::Reg(2),
                },
                ALWAYS,
            ),
            (Op::Exit, ALWAYS),
        ]);
        let wgsl = translate(&p).unwrap().source;
        assert!(!wgsl.contains("r255"), "RZ is not a register:\n{wgsl}");
        assert!(wgsl.contains("r1 = 0u;"), "RZ reads as zero:\n{wgsl}");
        assert!(
            !wgsl.contains("= r2;"),
            "a write to RZ is discarded:\n{wgsl}"
        );
    }

    #[test]
    fn the_function_returns_on_every_path() {
        // WGSL requires a final return.
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
        // `mulhi_s` depends on `mulhi_u`.
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
        // Every hook the emitter calls must be in `HOST_INTERFACE`.
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

    /// The smallest vertex/fragment pair with an interface.
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
                slot: TextureSlot::Bound(0x1a4),
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
        // Descriptor swizzles are applied.
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
        layout.textures[0].swizzle = IDENTITY_SWIZZLE;
        let plain = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!plain.contains("let sampled ="), "{plain}");
    }

    #[test]
    fn both_stages_name_the_same_location_for_a_varying() {
        // Both stages must agree on locations.
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

    /// The vertex stage is told which varyings are centroid.
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
        // Varyings are linear, carrying value/w.
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
        let (vs, _) = pair(0);
        let vs = translate(&vs).unwrap();
        let source = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
        assert!(source.contains("attr_out[31u] = 1.0;"), "{source}");
    }

    #[test]
    fn an_integer_attribute_is_declared_as_one_and_carried_as_its_bits() {
        // Integer attributes are declared integer and bitcast.
        let (vs, _) = pair(3);
        let vs = translate(&vs).unwrap();
        let mut layout = Layout::of(&vs, Stage::Vertex);
        layout.integer_attributes = vec![(3, AttributeBase::Sint)];
        let source = module(&vs, Stage::Vertex, &layout).unwrap();
        assert!(source.contains("@location(3) attr3: vec4<i32>"), "{source}");
        assert!(source.contains("bitcast<f32>(input.attr3.x)"), "{source}");
        assert!(braces_balance(&source), "{source}");

        let plain = Layout::of(&vs, Stage::Vertex);
        let source = module(&vs, Stage::Vertex, &plain).unwrap();
        assert!(source.contains("@location(3) attr3: vec4<f32>"), "{source}");
        assert!(!source.contains("bitcast<f32>(input."), "{source}");
    }

    #[test]
    fn a_depth_only_pass_writes_no_colour_and_still_shades() {
        // Depth-only fragments return nothing but still run for `kil`.
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
        // BGRA swaps as `fetch_attribute` does.
        let (vs, _) = pair(2);
        let vs = translate(&vs).unwrap();
        let mut layout = Layout::of(&vs, Stage::Vertex);
        layout.bgra_attributes = vec![2];
        let source = module(&vs, Stage::Vertex, &layout).unwrap();
        assert!(source.contains("attr_in[40u] = input.attr2.z;"), "{source}");
        assert!(source.contains("attr_in[42u] = input.attr2.x;"), "{source}");
        assert!(source.contains("attr_in[41u] = input.attr2.y;"), "{source}");
        assert!(source.contains("attr_in[43u] = input.attr2.w;"), "{source}");

        let plain = module(&vs, Stage::Vertex, &Layout::of(&vs, Stage::Vertex)).unwrap();
        assert!(plain.contains("attr_in[40u] = input.attr2.x;"), "{plain}");
    }

    /// A 2x2 grid in raster order.
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
        // Per-texel multisample rendering applies the sample mask by discarding.
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

        // An all-ones mask emits no branch.
        layout.coverage = Some(quad_coverage(u32::MAX, false));
        let open = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!open.contains("& 1u) == 0u"), "{open}");
    }

    #[test]
    fn alpha_to_coverage_keeps_the_same_prefix_the_rasterizer_keeps() {
        // Alpha-to-coverage keeps a prefix of `round(alpha * count)` samples.
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

        layout.coverage = Some(quad_coverage(u32::MAX, false));
        let plain = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!plain.contains("floor(clamp("), "{plain}");
        assert!(!plain.contains("let target0"), "{plain}");
    }

    #[test]
    fn a_single_sample_draw_carries_no_coverage_at_all() {
        // Only for grids with more than one sample.
        let p = program(&[(Op::Exit, ALWAYS)]);
        let translated = translate(&p).unwrap();
        let layout = Layout::of(&translated, Stage::Fragment);
        assert!(layout.coverage.is_none());
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        assert!(!source.contains("sample_of_slot"), "{source}");
    }

    #[test]
    fn a_vertex_shader_with_no_attributes_declares_no_input_struct() {
        // WGSL has no empty struct.
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
        // Unbindable dimensions are reported.
        let p = program(&[
            (
                Op::Texs {
                    dst: 0,
                    dst2: 2,
                    coords: [4, 5, 6],
                    dref: None,
                    handle: 1,
                    // 1D does not bind.
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

    const TEX_B: u64 = 0xdeba0007a0270000;

    #[test]
    fn a_bindless_tex_binds_the_constant_word_its_handle_was_loaded_from() {
        let tex = crate::gpu::shader::isa::decode(TEX_B).op;
        let p = program(&[
            (
                Op::Ldc {
                    dst: 2,
                    bank: 3,
                    offset: 0x10,
                    idx: RZ,
                    size: MemSize::B32,
                },
                ALWAYS,
            ),
            (tex, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        let slot = TextureSlot::Bindless {
            bank: 3,
            offset: 0x10,
        };
        assert_eq!(translated.textures, vec![(slot, TexDim::T2d, false)]);
        let layout = Layout::of(&translated, Stage::Fragment);
        let source = module(&translated, Stage::Fragment, &layout).unwrap();
        let key = slot.key();
        assert!(
            source.contains(&format!(
                "case {key}u: {{ return textureSampleLevel(tex0, smp0"
            )),
            "{source}"
        );
        assert!(
            translated.source.contains(&format!("texSample({key}u,")),
            "{}",
            translated.source
        );
    }

    #[test]
    fn a_bindless_handle_is_traced_through_a_wide_load_and_across_blocks() {
        // A sole write in another block still counts.
        let tex = crate::gpu::shader::isa::decode(TEX_B).op;
        let p = program(&[
            (
                Op::Ldc {
                    dst: 1,
                    bank: 4,
                    offset: 0x20,
                    idx: RZ,
                    size: MemSize::B64,
                },
                ALWAYS,
            ),
            (Op::Ssy { target: at(3) }, ALWAYS),
            (Op::Sync, ALWAYS),
            (tex, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        let translated = translate(&p).unwrap();
        assert_eq!(
            translated.textures,
            vec![(
                TextureSlot::Bindless {
                    bank: 4,
                    offset: 0x24
                },
                TexDim::T2d,
                false
            )]
        );
    }

    #[test]
    fn a_bindless_handle_that_is_not_a_constant_is_refused() {
        let tex = crate::gpu::shader::isa::decode(TEX_B).op;
        let computed = |pred| {
            program(&[
                (
                    Op::Mov {
                        dst: 2,
                        src: Operand::Const {
                            bank: 3,
                            offset: 0x10,
                        },
                    },
                    pred,
                ),
                (tex, ALWAYS),
                (Op::Exit, ALWAYS),
            ])
        };
        // A guarded load could leave any handle.
        let guarded = Pred {
            reg: 0,
            negate: false,
        };
        assert_eq!(
            translate(&computed(guarded)).unwrap_err(),
            Unsupported::UntracedHandle { at: 1 }
        );
        assert!(translate(&computed(ALWAYS)).is_ok());

        let indexed = program(&[
            (
                Op::Ldc {
                    dst: 2,
                    bank: 3,
                    offset: 0x10,
                    idx: 5,
                    size: MemSize::B32,
                },
                ALWAYS,
            ),
            (tex, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert_eq!(
            translate(&indexed).unwrap_err(),
            Unsupported::UntracedHandle { at: 1 }
        );
    }

    #[test]
    fn a_global_load_through_a_descriptor_loaded_whole_reads_that_descriptor() {
        let ldg = Op::Ldg {
            dst: 2,
            addr: 0,
            offset: 0,
            size: MemSize::B64,
        };
        let through_ldc = program(&[
            (
                Op::Ldc {
                    dst: 0,
                    bank: 1,
                    offset: 0x40,
                    idx: RZ,
                    size: MemSize::B64,
                },
                ALWAYS,
            ),
            (ldg, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert_eq!(translate(&through_ldc).unwrap().globals, vec![(1, 0x40)]);

        let mov = |dst, offset| Op::Mov {
            dst,
            src: Operand::Const { bank: 1, offset },
        };
        let through_movs = program(&[
            (mov(0, 0x40), ALWAYS),
            (mov(1, 0x44), ALWAYS),
            (ldg, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert_eq!(translate(&through_movs).unwrap().globals, vec![(1, 0x40)]);

        let mismatched = program(&[
            (mov(0, 0x40), ALWAYS),
            (mov(1, 0x48), ALWAYS),
            (ldg, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert!(translate(&mismatched).is_err());
    }

    /// Nintendo Switch Sports: a size-only texture query and an RZ-cleared offset.
    #[test]
    fn a_size_query_alone_and_a_zeroed_offset_translate() {
        let txq = crate::gpu::shader::isa::decode(0xdf48008180470800).op;
        let query_only = program(&[(txq, ALWAYS), (Op::Exit, ALWAYS)]);
        let translated = translate(&query_only).unwrap();
        assert_eq!(
            translated.textures,
            vec![(TextureSlot::Bound(8), TexDim::T2d, false)]
        );

        let tex = crate::gpu::shader::isa::decode(0xc0780083a0a70808).op;
        let zeroed = program(&[
            (
                Op::Mov {
                    dst: 10,
                    src: Operand::Reg(RZ),
                },
                ALWAYS,
            ),
            (tex, ALWAYS),
            (Op::Exit, ALWAYS),
        ]);
        assert_eq!(translate(&zeroed).unwrap().texture_offsets, vec![(0, 0)]);
    }

    #[test]
    fn a_module_is_complete_enough_to_stand_on_its_own() {
        // All four `HOST_INTERFACE` hooks are defined in the module.
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

    /// Halves round as `f32_to_f16` does.
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
        assert!(wgsl.contains("unpack2x16float(r2).xx"), "{wgsl}");
        assert!(wgsl.contains("unpack2x16float(r2).yy"), "{wgsl}");
        assert!(wgsl.contains("vec2<f32>(bitcast<f32>(r3))"), "{wgsl}");
        assert!(wgsl.contains("(r1 & 0xffff0000u) |"), "{wgsl}");
        assert!(wgsl.contains("(r1 & 0x0000ffffu) |"), "{wgsl}");
        assert!(wgsl.contains("== vec2<f32>(0.0)) | ("), "{wgsl}");
        assert!(wgsl.contains("ftz2(vec2<f32>(bitcast<f32>(r3)))"), "{wgsl}");
    }
}
