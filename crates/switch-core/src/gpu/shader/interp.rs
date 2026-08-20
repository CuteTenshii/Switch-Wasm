//! Executing decoded Maxwell instructions.
//!
//! [`Invocation`] is one shader invocation — one vertex or one fragment —
//! run to completion on a scalar machine: 255 general-purpose 32-bit
//! registers, seven predicate registers, a program counter, and the
//! reconvergence stack `ssy`/`sync` and `pbk`/`brk` push onto.
//!
//! Being scalar is what makes control flow simple here. Real hardware runs a
//! warp of 32 invocations in lockstep and needs an execution mask to handle
//! threads that take different branches; one invocation at a time just
//! follows its own branches, so `ssy`/`sync` reduce to an ordinary
//! push/pop of a return address.
//!
//! The register file is untyped, exactly as on hardware: a register holds 32
//! bits and the instruction decides whether they are a float, a signed
//! integer or a bit pattern. That matters — a shader routinely computes an
//! address with integer ops in the same registers it later loads floats
//! into.
//!
//! [`Invocation`] is deliberately rasterizer-oblivious: it doesn't know
//! whether it's a vertex or fragment shader, or where `attr_in`/constants
//! came from, which is what makes it independently testable.

use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::Program;
use crate::{Error, Result};
use std::collections::HashMap;

use super::isa::{
    BoolOp, FCmp, FRound, ICmp, LogicOp, MufuOp, Op, Operand, Pred, RZ,
};

/// Resolves a `cN[offset]` operand to its raw 32 bits. `bank` is whatever the
/// ISA's `Operand::Const` carries — for real programs that's a constant-buffer
/// *bind slot* (`Bind[]`'s index, not a raw GPU address), so a real source
/// still needs its own way to turn that into bytes; see [`MemoryConstants`].
/// Reads are fallible because a real one touches guest memory.
pub trait ConstantSource {
    fn read_const(&self, bank: u8, offset: u16) -> Result<u32>;
}

impl ConstantSource for HashMap<(u8, u16), f32> {
    fn read_const(&self, bank: u8, offset: u16) -> Result<u32> {
        Ok(self.get(&(bank, offset)).copied().unwrap_or(0.0).to_bits())
    }
}

/// Reads `cN[offset]` straight out of GPU memory. `bindings` resolves a bank
/// index to the `(address, size)` a real constant buffer was bound to —
/// `Engine3D::bound_constbuf` for the real integration, anything else for
/// tests — so this module stays decoupled from `engine::threed`.
pub struct MemoryConstants<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub bindings: &'a dyn Fn(u8) -> Option<(u64, u32)>,
}

impl ConstantSource for MemoryConstants<'_, '_> {
    fn read_const(&self, bank: u8, offset: u16) -> Result<u32> {
        let (addr, size) = (self.bindings)(bank).ok_or_else(|| {
            Error::Gpu(format!("shader: read from unbound constant bank {bank}"))
        })?;
        if offset as u32 + 4 > size {
            return Err(Error::Gpu(format!(
                "shader: constant read c{bank}[{offset:#x}] is past the bound buffer's size {size:#x}"
            )));
        }
        self.ctx.read_u32(addr + offset as u64)
    }
}

/// Resolves a `texs` sample. `handle` is the packed `imageId | samplerId <<
/// 20` value a real one reads out of the driver's reserved constant bank
/// (see `gpu::texture`'s module docs) — [`Invocation::execute`] does that
/// read itself via [`ConstantSource`] before calling this, so this trait only
/// needs to turn a resolved handle plus UVs into a colour.
pub trait TextureSource {
    fn sample(&self, handle: u32, u: f32, v: f32) -> Result<[f32; 4]>;
}

/// No texture backend at all — every `texs` is an error. Correct for vertex
/// shading and for tests that don't exercise `texs`.
pub struct NoTextures;

impl TextureSource for NoTextures {
    fn sample(&self, handle: u32, _u: f32, _v: f32) -> Result<[f32; 4]> {
        Err(Error::Gpu(format!(
            "shader: texture sample of handle {handle:#x} with no texture source bound"
        )))
    }
}

/// Samples a texture out of the real TIC/TSC descriptor pools in GPU memory.
pub struct MemoryTextures<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub tex_header_pool: u64,
    pub tex_sampler_pool: u64,
}

impl TextureSource for MemoryTextures<'_, '_> {
    fn sample(&self, handle: u32, u: f32, v: f32) -> Result<[f32; 4]> {
        crate::gpu::texture::sample(
            self.ctx,
            self.tex_header_pool,
            self.tex_sampler_pool,
            handle,
            u as f64,
            v as f64,
        )
    }
}

/// Reads a shader's global (`ldg`) address space. Optional: a program that
/// never issues one doesn't need a backend, and a program that does without
/// one gets an error naming the address rather than a silent zero.
pub trait GlobalMemory {
    fn read_u32(&self, addr: u64) -> Result<u32>;
}

/// The upper bound on instructions one invocation may execute. A shader with
/// a loop whose exit condition this interpreter gets wrong must fail rather
/// than hang the emulator.
const MAX_STEPS: usize = 1 << 20;

/// How many bytes of per-thread local scratch (`l[]`) an invocation gets.
const LOCAL_MEMORY_BYTES: usize = 1024;

/// Everything an invocation can read that isn't its own registers.
pub struct Env<'a> {
    pub consts: &'a dyn ConstantSource,
    pub textures: &'a dyn TextureSource,
    pub memory: Option<&'a dyn GlobalMemory>,
}

impl<'a> Env<'a> {
    pub fn new(consts: &'a dyn ConstantSource, textures: &'a dyn TextureSource) -> Env<'a> {
        Env { consts, textures, memory: None }
    }
}

/// Per-vertex/per-fragment machine state.
#[derive(Debug)]
/// The `a[]` attribute space — a shader's interpolated inputs on the way in,
/// its outputs on the way out — addressed by the byte offset the ISA uses
/// (`a[0x7c]` is offset `0x7c`).
///
/// Flat rather than a map, because a fragment shader runs *once per covered
/// pixel*: the `HashMap<u16, f32>` this replaces cost a hash per component
/// plus a heap allocation on each invocation's first insert, and those
/// together were most of the time in a shaded pixel. `ld`/`st`/`ipa` address
/// `a[]` with a ten-bit field, so the whole space is `0x000..0x400` — 256
/// words — and an offset past that (only reachable by adding an indexing
/// register) is outside attribute space entirely: it reads zero and a write
/// to it is dropped.
///
/// The written-mask is what makes "never written" distinguishable from
/// "written zero", which matters for outputs: a vertex shader that leaves
/// `clip.w` alone must get the default 1.0, not 0.0. It also makes
/// [`Attributes::clear`] a 32-byte wipe instead of a 1 KiB one.
#[derive(Clone)]
pub struct Attributes {
    words: [f32; Attributes::WORDS],
    written: [u64; Attributes::WORDS / 64],
}

impl Attributes {
    /// `a[]` is a ten-bit byte address, one `f32` per word.
    const WORDS: usize = 0x400 / 4;

    /// The value at `offset`, or 0.0 if nothing wrote it — what a read of an
    /// absent key gave before.
    pub fn get(&self, offset: u16) -> f32 {
        self.written(offset).unwrap_or(0.0)
    }

    /// The value at `offset`, or `None` if nothing wrote it.
    pub fn written(&self, offset: u16) -> Option<f32> {
        let word = offset as usize / 4;
        if word >= Self::WORDS || self.written[word / 64] & (1 << (word % 64)) == 0 {
            return None;
        }
        Some(self.words[word])
    }

    pub fn set(&mut self, offset: u16, value: f32) {
        let word = offset as usize / 4;
        if word >= Self::WORDS {
            return;
        }
        self.words[word] = value;
        self.written[word / 64] |= 1 << (word % 64);
    }

    /// Forget everything. Only the mask has to be cleared — a stale word is
    /// unreachable once it reads as unwritten.
    pub fn clear(&mut self) {
        self.written = [0; Self::WORDS / 64];
    }
}

impl Default for Attributes {
    fn default() -> Self {
        Attributes { words: [0.0; Attributes::WORDS], written: [0; Attributes::WORDS / 64] }
    }
}

pub struct Invocation {
    gpr: [u32; 255],
    /// `p0`..`p6`. `p7` is `PT`, which always reads true and can't be
    /// written, so it isn't stored.
    pred: [bool; 7],
    /// `a[]` input and output.
    pub attr_in: Attributes,
    pub attr_out: Attributes,
    /// Set by `kil`: this fragment must not be written.
    pub discarded: bool,
    /// `ssy`/`pbk`/`pcnt` push a resume address; `sync`/`brk`/`cont` pop it.
    stack: Vec<u32>,
    local: Vec<u8>,
}

impl Default for Invocation {
    fn default() -> Self {
        Invocation {
            gpr: [0; 255],
            pred: [false; 7],
            attr_in: Attributes::default(),
            attr_out: Attributes::default(),
            discarded: false,
            stack: Vec::new(),
            local: Vec::new(),
        }
    }
}

impl Invocation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Put this invocation back to its initial state so one of them can serve
    /// a whole draw. Building a fresh `Invocation` per fragment meant a 1 KiB
    /// register-file wipe and two map allocations for every covered pixel;
    /// this is the same state, without the allocations.
    pub fn reset(&mut self) {
        self.gpr = [0; 255];
        self.pred = [false; 7];
        self.attr_in.clear();
        self.attr_out.clear();
        self.discarded = false;
        self.stack.clear();
        self.local.clear();
    }

    pub fn reg_f32(&self, r: u8) -> f32 {
        f32::from_bits(self.reg(r))
    }

    pub fn set_reg_f32(&mut self, r: u8, v: f32) {
        self.set_reg(r, v.to_bits());
    }

    pub fn reg(&self, r: u8) -> u32 {
        if r == RZ {
            0
        } else {
            self.gpr[r as usize]
        }
    }

    pub fn set_reg(&mut self, r: u8, v: u32) {
        if r != RZ {
            self.gpr[r as usize] = v;
        }
    }

    pub fn pred(&self, p: u8) -> bool {
        if p >= 7 {
            true // PT
        } else {
            self.pred[p as usize]
        }
    }

    fn set_pred(&mut self, p: u8, v: bool) {
        if p < 7 {
            self.pred[p as usize] = v;
        }
    }

    /// Whether a guard or source predicate holds.
    fn holds(&self, p: Pred) -> bool {
        self.pred(p.reg) != p.negate
    }

    fn operand(&self, op: Operand, env: &Env) -> Result<u32> {
        match op {
            Operand::Reg(r) => Ok(self.reg(r)),
            Operand::Imm(v) => Ok(v),
            Operand::Const { bank, offset } => env.consts.read_const(bank, offset),
        }
    }

    fn operand_f32(&self, op: Operand, env: &Env) -> Result<f32> {
        Ok(f32::from_bits(self.operand(op, env)?))
    }

    /// Execute `program` from its entry point until it exits.
    pub fn execute(&mut self, program: &Program, env: &Env) -> Result<()> {
        if program.is_empty() {
            return Err(Error::Gpu("shader: executing an empty program".into()));
        }
        // Texture results land late; see `run_texs`.
        let mut pending: Vec<(usize, u8, f32)> = Vec::new();
        let mut pc = 0usize;
        let mut steps = 0usize;

        loop {
            if pc >= program.len() {
                return Err(Error::Gpu(
                    "shader: ran off the end of the program without an exit".into(),
                ));
            }
            steps += 1;
            if steps > MAX_STEPS {
                return Err(Error::Gpu(format!(
                    "shader: did not terminate within {MAX_STEPS} instructions"
                )));
            }
            pending.retain(|&(due, reg, val)| {
                if due == pc {
                    self.set_reg_f32(reg, val);
                    false
                } else {
                    true
                }
            });

            let insn = program.insns[pc];
            if !self.holds(insn.pred) {
                pc += 1;
                continue;
            }

            // Anything that moves the pc other than by one flushes the
            // deferred texture writes first: their landing place was found by
            // scanning forward in program order, which a jump invalidates.
            let jump = |target: u32, pending: &mut Vec<(usize, u8, f32)>, inv: &mut Self| {
                for (_, reg, val) in pending.drain(..) {
                    inv.set_reg_f32(reg, val);
                }
                program.index_of(target).ok_or_else(|| {
                    Error::Gpu(format!("shader: branch to {target:#x}, which was never decoded"))
                })
            };

            match insn.op {
                Op::Exit => {
                    for (_, reg, val) in pending.drain(..) {
                        self.set_reg_f32(reg, val);
                    }
                    return Ok(());
                }
                Op::Kil => {
                    self.discarded = true;
                    return Ok(());
                }
                Op::Nop | Op::Inert => {}
                Op::Bra { target } => {
                    pc = jump(target, &mut pending, self)?;
                    continue;
                }
                Op::Ssy { target } | Op::Pbk { target } | Op::Pcnt { target } => {
                    self.stack.push(target);
                }
                Op::Sync | Op::Brk | Op::Cont => {
                    let target = self.stack.pop().ok_or_else(|| {
                        Error::Gpu("shader: sync/brk/cont with an empty reconvergence stack".into())
                    })?;
                    pc = jump(target, &mut pending, self)?;
                    continue;
                }
                Op::Texs { .. } => {
                    self.run_texs(program, pc, insn.op, env, &mut pending)?;
                }
                other => self.run_alu(other, env)?,
            }
            pc += 1;
        }
    }

    /// Everything that isn't control flow or a texture fetch.
    fn run_alu(&mut self, op: Op, env: &Env) -> Result<()> {
        match op {
            // ---- attribute space ----
            Op::Ld { dst, offset, idx, size } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.attr_in.get(base + i as u16 * 4);
                    self.set_reg_f32(dst.wrapping_add(i), v);
                }
            }
            Op::St { offset, idx, src, size } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.reg_f32(src.wrapping_add(i));
                    self.attr_out.set(base + i as u16 * 4, v);
                }
            }
            Op::Ipa { dst, offset, mul, perspective, sat } => {
                let mut v = self.attr_in.get(offset);
                if perspective {
                    if let Some(m) = mul {
                        v *= self.reg_f32(m);
                    }
                }
                if sat {
                    v = v.clamp(0.0, 1.0);
                }
                self.set_reg_f32(dst, v);
            }

            // ---- float ----
            Op::Fadd { dst, a, am, b, bm, ftz, sat } => {
                let x = am.apply(flush(self.reg_f32(a), ftz));
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                self.set_reg_f32(dst, saturate(x + y, sat));
            }
            Op::Fmul { dst, a, b, bm, ftz, sat } => {
                let x = flush(self.reg_f32(a), ftz);
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                self.set_reg_f32(dst, saturate(x * y, sat));
            }
            Op::Ffma { dst, a, b, bneg, c, cneg, ftz, sat } => {
                let x = flush(self.reg_f32(a), ftz);
                let y = neg_if(flush(self.operand_f32(b, env)?, ftz), bneg);
                let z = neg_if(flush(self.operand_f32(c, env)?, ftz), cneg);
                self.set_reg_f32(dst, saturate(x.mul_add(y, z), sat));
            }
            Op::Fmnmx { dst, a, am, b, bm, pred, ftz } => {
                let x = am.apply(flush(self.reg_f32(a), ftz));
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                // The predicate selects which end: true picks the minimum,
                // which is why `fmnmx ... !pt` is the compiler's `max`.
                let v = if self.holds(pred) { x.min(y) } else { x.max(y) };
                self.set_reg_f32(dst, v);
            }
            Op::Mufu { dst, src, sm, op, sat } => {
                let x = sm.apply(self.reg_f32(src));
                let v = match op {
                    MufuOp::Cos => x.cos(),
                    MufuOp::Sin => x.sin(),
                    MufuOp::Ex2 => x.exp2(),
                    MufuOp::Lg2 => x.log2(),
                    MufuOp::Rcp => 1.0 / x,
                    MufuOp::Rsq => 1.0 / x.sqrt(),
                    MufuOp::Sqrt => x.sqrt(),
                };
                self.set_reg_f32(dst, saturate(v, sat));
            }
            Op::Fsetp { p0, p1, a, am, b, bm, cmp, bop, src } => {
                let x = am.apply(self.reg_f32(a));
                let y = bm.apply(self.operand_f32(b, env)?);
                let r = float_compare(cmp, x, y);
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
            }
            Op::Fset { dst, a, am, b, bm, cmp, bop, src, bf } => {
                let x = am.apply(self.reg_f32(a));
                let y = bm.apply(self.operand_f32(b, env)?);
                let r = combine(bop, float_compare(cmp, x, y), self.holds(src));
                self.set_reg(dst, set_result(r, bf));
            }

            // ---- integer ----
            Op::Iadd { dst, a, aneg, b, bneg } => {
                let x = ineg_if(self.reg(a), aneg);
                let y = ineg_if(self.operand(b, env)?, bneg);
                self.set_reg(dst, x.wrapping_add(y));
            }
            Op::Iadd3 { dst, a, aneg, b, bneg, c, cneg } => {
                let x = ineg_if(self.reg(a), aneg);
                let y = ineg_if(self.operand(b, env)?, bneg);
                let z = ineg_if(self.operand(c, env)?, cneg);
                self.set_reg(dst, x.wrapping_add(y).wrapping_add(z));
            }
            Op::Iscadd { dst, a, aneg, b, bneg, shift } => {
                let x = ineg_if(self.reg(a), aneg).wrapping_shl(shift as u32);
                let y = ineg_if(self.operand(b, env)?, bneg);
                self.set_reg(dst, x.wrapping_add(y));
            }
            Op::Imnmx { dst, a, b, pred, signed } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let take_min = self.holds(pred);
                let v = if signed {
                    let (x, y) = (x as i32, y as i32);
                    (if take_min { x.min(y) } else { x.max(y) }) as u32
                } else if take_min {
                    x.min(y)
                } else {
                    x.max(y)
                };
                self.set_reg(dst, v);
            }
            Op::Imul { dst, a, b, signed, hi } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let full = if signed {
                    ((x as i32 as i64) * (y as i32 as i64)) as u64
                } else {
                    (x as u64) * (y as u64)
                };
                self.set_reg(dst, if hi { (full >> 32) as u32 } else { full as u32 });
            }
            Op::Xmad { dst, a, ah, asigned, b, bh, bsigned, c, psl, mrg } => {
                let av = half(self.reg(a), ah, asigned);
                let bv = half(self.operand(b, env)?, bh, bsigned);
                let mut product = (av.wrapping_mul(bv)) as u32;
                if psl {
                    product <<= 16;
                }
                let cv = self.operand(c, env)?;
                let mut v = product.wrapping_add(cv);
                if mrg {
                    // `.mrg` keeps the product's low half in the result's
                    // high half instead of adding it there.
                    v = (v & 0xffff) | (product << 16);
                }
                self.set_reg(dst, v);
            }
            Op::Isetp { p0, p1, a, b, cmp, signed, bop, src } => {
                let r = int_compare(cmp, self.reg(a), self.operand(b, env)?, signed);
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
            }
            Op::Iset { dst, a, b, cmp, signed, bop, src, bf } => {
                let r = int_compare(cmp, self.reg(a), self.operand(b, env)?, signed);
                let r = combine(bop, r, self.holds(src));
                self.set_reg(dst, set_result(r, bf));
            }
            Op::Icmp { dst, a, b, c, cmp, signed } => {
                // `icmp dst, a, b, c` is "dst = compare(c, 0) ? a : b".
                let taken = int_compare(cmp, self.reg(c), 0, signed);
                let v = if taken { self.reg(a) } else { self.operand(b, env)? };
                self.set_reg(dst, v);
            }
            Op::Lop { dst, a, ainv, b, binv, op } => {
                let x = inv_if(self.reg(a), ainv);
                let y = inv_if(self.operand(b, env)?, binv);
                let v = match op {
                    LogicOp::And => x & y,
                    LogicOp::Or => x | y,
                    LogicOp::Xor => x ^ y,
                    LogicOp::PassB => y,
                };
                self.set_reg(dst, v);
            }
            Op::Lop3 { dst, a, b, c, lut } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let z = self.operand(c, env)?;
                self.set_reg(dst, lop3(x, y, z, lut));
            }
            Op::Shl { dst, a, b, wrap } => {
                let n = self.operand(b, env)?;
                let n = if wrap { n & 31 } else { n };
                self.set_reg(dst, if n >= 32 { 0 } else { self.reg(a) << n });
            }
            Op::Shr { dst, a, b, signed, wrap } => {
                let n = self.operand(b, env)?;
                let n = if wrap { n & 31 } else { n };
                let x = self.reg(a);
                let v = if signed {
                    if n >= 32 {
                        ((x as i32) >> 31) as u32
                    } else {
                        ((x as i32) >> n) as u32
                    }
                } else if n >= 32 {
                    0
                } else {
                    x >> n
                };
                self.set_reg(dst, v);
            }
            Op::Shf { dst, lo, shift, hi, left, wrap, hi_out } => {
                let n = self.operand(shift, env)?;
                let n = if wrap { n & 63 } else { n };
                let pair = ((self.reg(hi) as u64) << 32) | self.reg(lo) as u64;
                let shifted = if left { pair.wrapping_shl(n) } else { pair.wrapping_shr(n) };
                self.set_reg(dst, if hi_out { (shifted >> 32) as u32 } else { shifted as u32 });
            }
            Op::Bfe { dst, a, b, signed } => {
                let desc = self.operand(b, env)?;
                let start = (desc & 0xff) as u32;
                let width = ((desc >> 8) & 0xff) as u32;
                self.set_reg(dst, bitfield_extract(self.reg(a), start, width, signed));
            }
            Op::Popc { dst, b, inv } => {
                let v = inv_if(self.operand(b, env)?, inv);
                self.set_reg(dst, v.count_ones());
            }
            Op::Flo { dst, b, signed, shift, inv } => {
                let v = inv_if(self.operand(b, env)?, inv);
                // The highest set bit, counting from bit 0; for a signed
                // search the sign bits at the top don't count.
                let v = if signed && (v as i32) < 0 { !v } else { v };
                let idx = if v == 0 { 0xffff_ffff } else { 31 - v.leading_zeros() };
                self.set_reg(dst, if shift && v != 0 { 31 - idx } else { idx });
            }
            Op::Sel { dst, a, b, pred } => {
                let v = if self.holds(pred) { self.reg(a) } else { self.operand(b, env)? };
                self.set_reg(dst, v);
            }

            // ---- conversions ----
            Op::I2f { dst, src, sm, src_bytes, src_signed, sel } => {
                let raw = self.operand(src, env)?;
                let raw = raw >> (sel as u32 * 8);
                let v = if src_signed {
                    sign_extend(raw, src_bytes) as i32 as f32
                } else {
                    truncate(raw, src_bytes) as f32
                };
                self.set_reg_f32(dst, sm.apply(v));
            }
            Op::F2i { dst, src, sm, dst_bytes, dst_signed, round, ftz } => {
                let x = sm.apply(flush(self.operand_f32(src, env)?, ftz));
                let r = apply_round(x, round);
                let v = if dst_signed {
                    let lo = -(2f64.powi(dst_bytes as i32 * 8 - 1)) as f32;
                    let hi = (2f64.powi(dst_bytes as i32 * 8 - 1) - 1.0) as f32;
                    if r.is_nan() { 0 } else { r.clamp(lo, hi) as i32 as u32 }
                } else {
                    let hi = (2f64.powi(dst_bytes as i32 * 8) - 1.0) as f32;
                    if r.is_nan() { 0 } else { r.clamp(0.0, hi) as u32 }
                };
                self.set_reg(dst, v);
            }
            Op::F2f { dst, src, sm, round, sat, ftz } => {
                let x = sm.apply(flush(self.operand_f32(src, env)?, ftz));
                self.set_reg_f32(dst, saturate(apply_round(x, round), sat));
            }
            Op::I2i { dst, src, sm, src_bytes, src_signed, dst_signed, sat, sel } => {
                let raw = self.operand(src, env)? >> (sel as u32 * 8);
                let mut v = if src_signed {
                    sign_extend(raw, src_bytes)
                } else {
                    truncate(raw, src_bytes)
                };
                if sm.neg {
                    v = (v as i32).wrapping_neg() as u32;
                }
                if sm.abs {
                    v = (v as i32).unsigned_abs();
                }
                if sat && !dst_signed {
                    v = (v as i32).max(0) as u32;
                }
                self.set_reg(dst, v);
            }

            // ---- moves ----
            Op::Mov { dst, src } => {
                let v = self.operand(src, env)?;
                self.set_reg(dst, v);
            }
            Op::Mov32i { dst, imm } => self.set_reg(dst, imm),
            Op::S2r { dst, sr } => {
                // Nothing here runs a warp, a CTA or more than one
                // invocation at a time, so every lane/thread identity is 0.
                // `invocation_info` and the mask registers are the only ones
                // whose zero is not obviously right, and no shader this
                // executes has asked for them.
                let _ = sr;
                self.set_reg(dst, 0);
            }
            Op::Psetp { p0, p1, a, b, c, op1, op2 } => {
                let first = combine(op1, self.holds(a), self.holds(b));
                let r = combine(op2, first, self.holds(c));
                self.set_pred(p0, r);
                self.set_pred(p1, !r);
            }

            // ---- memory ----
            Op::Ldc { dst, bank, offset, idx, size } => {
                let base = offset.wrapping_add(self.reg(idx) as i32);
                for i in 0..size.regs() {
                    let at = base.wrapping_add(i as i32 * 4);
                    let v = env.consts.read_const(bank, at as u16)?;
                    self.set_reg(dst.wrapping_add(i), v);
                }
            }
            Op::Ldg { dst, addr, offset, size } => {
                let mem = env.memory.ok_or_else(|| {
                    Error::Gpu("shader: ldg with no global memory bound".into())
                })?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                for i in 0..size.regs() {
                    let v = mem.read_u32(base.wrapping_add(i as u64 * 4))?;
                    self.set_reg(dst.wrapping_add(i), v);
                }
            }
            Op::Ldl { dst, addr, offset, size } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                for i in 0..size.regs() {
                    let at = base + i as usize * 4;
                    let mut word = [0u8; 4];
                    if at + 4 <= self.local.len() {
                        word.copy_from_slice(&self.local[at..at + 4]);
                    }
                    self.set_reg(dst.wrapping_add(i), u32::from_le_bytes(word));
                }
            }
            Op::Stl { addr, offset, src, size } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                if self.local.len() < LOCAL_MEMORY_BYTES {
                    self.local.resize(LOCAL_MEMORY_BYTES, 0);
                }
                for i in 0..size.regs() {
                    let at = base + i as usize * 4;
                    if at + 4 <= self.local.len() {
                        let v = self.reg(src.wrapping_add(i)).to_le_bytes();
                        self.local[at..at + 4].copy_from_slice(&v);
                    }
                }
            }
            Op::Stg { .. } => {
                return Err(Error::Gpu("shader: stg (global store) is not implemented".into()))
            }

            Op::Unimplemented { raw } => {
                return Err(Error::Gpu(format!("shader: unimplemented instruction {raw:#018x}")))
            }
            // Handled by `execute`.
            Op::Exit
            | Op::Kil
            | Op::Nop
            | Op::Inert
            | Op::Bra { .. }
            | Op::Ssy { .. }
            | Op::Pbk { .. }
            | Op::Pcnt { .. }
            | Op::Sync
            | Op::Brk
            | Op::Cont
            | Op::Texs { .. } => unreachable!("control flow is dispatched in execute"),
        }
        Ok(())
    }

    /// `ld`/`st a[r + imm]`: the index register holds a byte offset, and `RZ`
    /// (the common case) contributes nothing.
    fn attr_index(&self, idx: u8) -> u16 {
        self.reg(idx) as u16
    }

    /// A 64-bit address held in a register pair.
    fn reg64(&self, r: u8) -> u64 {
        u64::from(self.reg(r)) | (u64::from(self.reg(r.wrapping_add(1))) << 32)
    }

    /// Real Maxwell issues `texs` asynchronously: the compiler interleaves
    /// unrelated instructions between the fetch and its first real consumer,
    /// relying on the texture unit's latency to hide them, and those
    /// instructions still see whatever the destination registers held
    /// before the fetch. A synchronous write at the `texs` itself breaks
    /// that (see `gpu::texture`'s module docs for how this was caught
    /// against real content), so each destination's value is queued and
    /// applied immediately before the instruction that first reads it —
    /// or flushed at the next branch or at `exit`, whichever comes first.
    fn run_texs(
        &mut self,
        program: &Program,
        pc: usize,
        op: Op,
        env: &Env,
        pending: &mut Vec<(usize, u8, f32)>,
    ) -> Result<()> {
        let Op::Texs { coords, handle, .. } = op else {
            unreachable!("run_texs called with {op:?}");
        };
        // The bindless handle lives in the driver's reserved constant bank
        // at the shader's own immediate offset — see `gpu::texture`'s module
        // docs for how that was confirmed.
        let handle =
            env.consts.read_const(crate::gpu::texture::DRIVER_CONSTBUF_BANK, handle)?;
        let u = self.reg_f32(coords[1]);
        let v = self.reg_f32(coords[2]);
        let color = env.textures.sample(handle, u, v)?;

        // Where each channel lands was worked out at decode time.
        for &(channel, reg, due) in program.texs_writes(pc) {
            pending.retain(|&(_, r, _)| r != reg);
            pending.push((due, reg, color[channel]));
        }
        Ok(())
    }
}

/// Work out, for every `texs` in `program`, where each of its results lands.
/// Called once per decode by [`super::decode_program_with`]; see
/// [`super::Program::texs_writes`] for why it is not done per invocation.
pub(super) fn texs_writes_for(program: &Program) -> Vec<super::TexsWrites> {
    let mut out = Vec::new();
    for (pc, insn) in program.insns.iter().enumerate() {
        let Op::Texs { dst, mask, .. } = insn.op else {
            continue;
        };
        let mut writes = Vec::new();
        let mut reg = dst;
        for (channel, &enabled) in mask.iter().enumerate() {
            if !enabled {
                continue;
            }
            let due = first_use_after(program, pc + 1, reg).unwrap_or(program.len() - 1);
            writes.push((channel, reg, due));
            reg = reg.wrapping_add(1);
        }
        out.push(super::TexsWrites { at: pc, writes });
    }
    out
}

/// Where `reg`'s pending write should land: right before the first later
/// instruction that reads it (the real dependency point), or dropped
/// entirely if something overwrites it first. A program that never touches
/// it again lands it before the last instruction, so a shader that hands a
/// `texs` result straight to its output register still ends with the value
/// hardware would eventually have written.
fn first_use_after(program: &Program, start: usize, reg: u8) -> Option<usize> {
    for (idx, insn) in program.insns.iter().enumerate().skip(start) {
        if reads(&insn.op).contains(&reg) {
            return Some(idx);
        }
        if writes(&insn.op).contains(&reg) {
            return None;
        }
    }
    program.len().checked_sub(1)
}

fn operand_reg(op: Operand) -> Option<u8> {
    match op {
        Operand::Reg(r) if r != RZ => Some(r),
        _ => None,
    }
}

/// Registers `op` reads as a source (never [`RZ`], which is always zero).
fn reads(op: &Op) -> Vec<u8> {
    let mut out: Vec<u8> = match *op {
        Op::St { src, size, idx, .. } => {
            let mut v: Vec<u8> = (0..size.regs()).map(|i| src.wrapping_add(i)).collect();
            v.push(idx);
            v
        }
        Op::Ld { idx, .. } => vec![idx],
        Op::Ipa { mul: Some(m), .. } => vec![m],
        Op::Mufu { src, .. } => vec![src],
        Op::Fadd { a, b, .. } | Op::Fmul { a, b, .. } | Op::Fmnmx { a, b, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v
        }
        Op::Ffma { a, b, c, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v
        }
        Op::Iadd { a, b, .. }
        | Op::Imnmx { a, b, .. }
        | Op::Imul { a, b, .. }
        | Op::Lop { a, b, .. }
        | Op::Shl { a, b, .. }
        | Op::Shr { a, b, .. }
        | Op::Bfe { a, b, .. }
        | Op::Sel { a, b, .. }
        | Op::Iset { a, b, .. }
        | Op::Isetp { a, b, .. }
        | Op::Fset { a, b, .. }
        | Op::Fsetp { a, b, .. }
        | Op::Iscadd { a, b, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v
        }
        Op::Iadd3 { a, b, c, .. } | Op::Xmad { a, b, c, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v
        }
        Op::Lop3 { a, b, c, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v
        }
        Op::Icmp { a, b, c, .. } => {
            let mut v = vec![a, c];
            v.extend(operand_reg(b));
            v
        }
        Op::Shf { lo, shift, hi, .. } => {
            let mut v = vec![lo, hi];
            v.extend(operand_reg(shift));
            v
        }
        Op::Popc { b, .. } | Op::Flo { b, .. } => operand_reg(b).into_iter().collect(),
        Op::Mov { src, .. } => operand_reg(src).into_iter().collect(),
        Op::I2f { src, .. } | Op::F2i { src, .. } | Op::F2f { src, .. } | Op::I2i { src, .. } => {
            operand_reg(src).into_iter().collect()
        }
        Op::Ldc { idx, .. } => vec![idx],
        Op::Ldg { addr, .. } | Op::Ldl { addr, .. } => vec![addr, addr.wrapping_add(1)],
        Op::Stg { addr, src, size, .. } | Op::Stl { addr, src, size, .. } => {
            let mut v = vec![addr, addr.wrapping_add(1)];
            v.extend((0..size.regs()).map(|i| src.wrapping_add(i)));
            v
        }
        Op::Texs { coords, .. } => coords.to_vec(),
        _ => Vec::new(),
    };
    out.retain(|&r| r != RZ);
    out
}

/// Registers `op` writes as a destination.
fn writes(op: &Op) -> Vec<u8> {
    match *op {
        Op::Ld { dst, size, .. } | Op::Ldg { dst, size, .. } | Op::Ldl { dst, size, .. }
        | Op::Ldc { dst, size, .. } => {
            (0..size.regs()).map(|i| dst.wrapping_add(i)).collect()
        }
        Op::Ipa { dst, .. }
        | Op::Mufu { dst, .. }
        | Op::Fadd { dst, .. }
        | Op::Fmul { dst, .. }
        | Op::Ffma { dst, .. }
        | Op::Fmnmx { dst, .. }
        | Op::Fset { dst, .. }
        | Op::Mov { dst, .. }
        | Op::Mov32i { dst, .. }
        | Op::S2r { dst, .. }
        | Op::Iadd { dst, .. }
        | Op::Iadd3 { dst, .. }
        | Op::Imnmx { dst, .. }
        | Op::Imul { dst, .. }
        | Op::Xmad { dst, .. }
        | Op::Iscadd { dst, .. }
        | Op::Iset { dst, .. }
        | Op::Icmp { dst, .. }
        | Op::Lop { dst, .. }
        | Op::Lop3 { dst, .. }
        | Op::Shl { dst, .. }
        | Op::Shr { dst, .. }
        | Op::Shf { dst, .. }
        | Op::Bfe { dst, .. }
        | Op::Popc { dst, .. }
        | Op::Flo { dst, .. }
        | Op::Sel { dst, .. }
        | Op::I2f { dst, .. }
        | Op::F2i { dst, .. }
        | Op::F2f { dst, .. }
        | Op::I2i { dst, .. } => vec![dst],
        Op::Texs { dst, mask, .. } => {
            let mut r = dst;
            let mut v = Vec::new();
            for &enabled in mask.iter() {
                if enabled {
                    v.push(r);
                    r = r.wrapping_add(1);
                }
            }
            v
        }
        _ => Vec::new(),
    }
}

fn flush(v: f32, ftz: bool) -> f32 {
    if ftz && v.is_subnormal() {
        0.0f32.copysign(v)
    } else {
        v
    }
}

fn saturate(v: f32, sat: bool) -> f32 {
    if sat {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 1.0)
        }
    } else {
        v
    }
}

fn neg_if(v: f32, neg: bool) -> f32 {
    if neg {
        -v
    } else {
        v
    }
}

fn ineg_if(v: u32, neg: bool) -> u32 {
    if neg {
        (v as i32).wrapping_neg() as u32
    } else {
        v
    }
}

fn inv_if(v: u32, inv: bool) -> u32 {
    if inv {
        !v
    } else {
        v
    }
}

fn apply_round(v: f32, round: FRound) -> f32 {
    match round {
        FRound::Nearest => v.round_ties_even(),
        FRound::Floor => v.floor(),
        FRound::Ceil => v.ceil(),
        FRound::Trunc => v.trunc(),
    }
}

/// A `set`'s register result: all-ones as a bit mask, or 1.0f with `.bf`.
fn set_result(r: bool, bf: bool) -> u32 {
    match (r, bf) {
        (false, _) => 0,
        (true, true) => 1.0f32.to_bits(),
        (true, false) => u32::MAX,
    }
}

fn combine(op: BoolOp, a: bool, b: bool) -> bool {
    match op {
        BoolOp::And => a && b,
        BoolOp::Or => a || b,
        BoolOp::Xor => a != b,
    }
}

fn float_compare(cmp: FCmp, a: f32, b: f32) -> bool {
    let unordered = a.is_nan() || b.is_nan();
    match cmp {
        FCmp::Never => false,
        FCmp::Lt => a < b,
        FCmp::Eq => a == b,
        FCmp::Le => a <= b,
        FCmp::Gt => a > b,
        FCmp::Ne => !unordered && a != b,
        FCmp::Ge => a >= b,
        FCmp::Num => !unordered,
        FCmp::Nan => unordered,
        FCmp::LtU => unordered || a < b,
        FCmp::EqU => unordered || a == b,
        FCmp::LeU => unordered || a <= b,
        FCmp::GtU => unordered || a > b,
        FCmp::NeU => unordered || a != b,
        FCmp::GeU => unordered || a >= b,
        FCmp::Always => true,
    }
}

fn int_compare(cmp: ICmp, a: u32, b: u32, signed: bool) -> bool {
    let ord = if signed {
        (a as i32).cmp(&(b as i32))
    } else {
        a.cmp(&b)
    };
    match cmp {
        ICmp::Never => false,
        ICmp::Lt => ord.is_lt(),
        ICmp::Eq => ord.is_eq(),
        ICmp::Le => ord.is_le(),
        ICmp::Gt => ord.is_gt(),
        ICmp::Ne => ord.is_ne(),
        ICmp::Ge => ord.is_ge(),
        ICmp::Always => true,
    }
}

/// `lop3`'s truth table: bit `n` of `lut` is the result for the input
/// combination whose bits are `(a, b, c)` read as a 3-bit number.
fn lop3(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    let mut out = 0u32;
    for i in 0..8u32 {
        if lut & (1 << i) == 0 {
            continue;
        }
        let mask = mask_for(a, i & 4 != 0) & mask_for(b, i & 2 != 0) & mask_for(c, i & 1 != 0);
        out |= mask;
    }
    out
}

fn mask_for(v: u32, want_set: bool) -> u32 {
    if want_set {
        v
    } else {
        !v
    }
}

fn bitfield_extract(v: u32, start: u32, width: u32, signed: bool) -> u32 {
    if width == 0 {
        return 0;
    }
    let start = start.min(31);
    let width = width.min(32 - start);
    let raw = (v >> start) & (u32::MAX >> (32 - width));
    if signed && width < 32 && raw & (1 << (width - 1)) != 0 {
        raw | !(u32::MAX >> (32 - width))
    } else {
        raw
    }
}

fn sign_extend(v: u32, bytes: u8) -> u32 {
    match bytes {
        1 => v as u8 as i8 as i32 as u32,
        2 => v as u16 as i16 as i32 as u32,
        _ => v,
    }
}

fn truncate(v: u32, bytes: u8) -> u32 {
    match bytes {
        1 => v & 0xff,
        2 => v & 0xffff,
        _ => v,
    }
}

/// One 16-bit half of a register, as `xmad` reads it.
fn half(v: u32, high: bool, signed: bool) -> u32 {
    let h = if high { v >> 16 } else { v & 0xffff };
    if signed {
        h as u16 as i16 as i32 as u32
    } else {
        h
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shader::decode_program;
    use crate::gpu::shader::isa::{FMod, Instruction, MufuOp, TexDim};
    use crate::gpu::shader::Program;

    fn no_consts() -> HashMap<(u8, u16), f32> {
        HashMap::new()
    }

    /// Build a straight-line program out of unpredicated ops, at the byte
    /// offsets a real 32-byte-block layout would put them at.
    fn prog(ops: &[Op]) -> Program {
        let mut p = Program::default();
        for (i, &op) in ops.iter().enumerate() {
            p.insns.push(Instruction::always(op));
            p.offsets.push(crate::gpu::shader::ENTRY_OFFSET + i as u32 * 8);
        }
        p
    }
    use std::cell::RefCell;

    /// Records the `(handle, u, v)` it was asked to sample and always
    /// returns the same colour, so a test can check both what the
    /// interpreter computed and what it fed the texture backend.
    struct RecordingTextures {
        calls: RefCell<Vec<(u32, f32, f32)>>,
        color: [f32; 4],
    }

    impl TextureSource for RecordingTextures {
        fn sample(&self, handle: u32, u: f32, v: f32) -> Result<[f32; 4]> {
            self.calls.borrow_mut().push((handle, u, v));
            Ok(self.color)
        }
    }

    #[test]
    fn a_hand_written_alu_program_produces_the_expected_registers() {
        // r2 = r0 * r1; r3 = r2 * r1 + r0. Register-register forms only, so
        // no constant source is exercised — this is purely the interpreter's
        // execute loop, independent of the decoder and of any real shader.
        let program = prog(&[
            Op::Fmul { dst: 2, a: 0, b: Operand::Reg(1), bm: FMod::NONE, ftz: true, sat: false },
            Op::Ffma {
                dst: 3,
                a: 2,
                b: Operand::Reg(1),
                bneg: false,
                c: Operand::Reg(0),
                cneg: false,
                ftz: true,
                sat: false,
            },
            Op::Exit,
        ]);
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 2.0);
        inv.set_reg_f32(1, 3.0);

        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();

        assert_eq!(inv.reg_f32(2), 6.0);
        assert_eq!(inv.reg_f32(3), 20.0);
    }

    /// A program with real byte offsets, so branch targets resolve. Each
    /// entry is `(op, predicate)`; offsets follow the 32-byte block layout
    /// (slot 0 of every block is a `sched` word, so it is skipped).
    fn prog_at(entries: &[(Op, Pred)]) -> Program {
        let mut p = Program::default();
        let mut offset = crate::gpu::shader::ENTRY_OFFSET;
        for &(op, pred) in entries {
            p.insns.push(Instruction { pred, op });
            p.offsets.push(offset);
            offset = crate::gpu::shader::next_slot(offset);
        }
        p
    }

    #[test]
    fn a_guard_predicate_skips_the_instruction() {
        // r1 = 1.0 always; r2 = 2.0 only if p0; r3 = 3.0 only if !p0.
        let program = prog_at(&[
            (Op::Mov32i { dst: 1, imm: 1.0f32.to_bits() }, Pred::ALWAYS),
            (Op::Mov32i { dst: 2, imm: 2.0f32.to_bits() }, Pred { reg: 0, negate: false }),
            (Op::Mov32i { dst: 3, imm: 3.0f32.to_bits() }, Pred { reg: 0, negate: true }),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        // p0 starts false.
        assert_eq!(inv.reg_f32(1), 1.0);
        assert_eq!(inv.reg(2), 0, "a false guard must skip the write");
        assert_eq!(inv.reg_f32(3), 3.0);
    }

    #[test]
    fn isetp_then_a_predicated_branch_takes_the_right_path() {
        // if (r0 < r1) r2 = 10 else r2 = 20 — the shape every `if` in a real
        // shader compiles to, and the whole reason the decoder had to stop
        // treating a predicated instruction as unsupported.
        let program = prog_at(&[
            (
                Op::Isetp {
                    p0: 0,
                    p1: 7,
                    a: 0,
                    b: Operand::Reg(1),
                    cmp: ICmp::Lt,
                    signed: true,
                    bop: BoolOp::And,
                    src: Pred::ALWAYS,
                },
                Pred::ALWAYS,
            ),
            // @!p0 bra else
            (Op::Bra { target: 0x30 }, Pred { reg: 0, negate: true }),
            (Op::Mov32i { dst: 2, imm: 10 }, Pred::ALWAYS),
            (Op::Bra { target: 0x38 }, Pred::ALWAYS), // skip the else
            (Op::Mov32i { dst: 2, imm: 20 }, Pred::ALWAYS), // else, at 0x30
            (Op::Exit, Pred::ALWAYS),                       // at 0x38
        ]);
        // Offset 0x20 is a `sched` control word, not an instruction slot.
        assert_eq!(program.offsets, vec![0x08, 0x10, 0x18, 0x28, 0x30, 0x38]);

        let mut taken = Invocation::new();
        taken.set_reg(0, 1);
        taken.set_reg(1, 2);
        taken.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert_eq!(taken.reg(2), 10);

        let mut not_taken = Invocation::new();
        not_taken.set_reg(0, 5);
        not_taken.set_reg(1, 2);
        not_taken.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert_eq!(not_taken.reg(2), 20);
    }

    #[test]
    fn a_backward_branch_runs_a_real_loop() {
        // r1 = 0; do { r1 += 1 } while (r1 < 4)
        let program = prog_at(&[
            (Op::Mov32i { dst: 1, imm: 0 }, Pred::ALWAYS),
            // loop body, at 0x10
            (
                Op::Iadd { dst: 1, a: 1, aneg: false, b: Operand::Imm(1), bneg: false },
                Pred::ALWAYS,
            ),
            (
                Op::Isetp {
                    p0: 0,
                    p1: 7,
                    a: 1,
                    b: Operand::Imm(4),
                    cmp: ICmp::Lt,
                    signed: true,
                    bop: BoolOp::And,
                    src: Pred::ALWAYS,
                },
                Pred::ALWAYS,
            ),
            (Op::Bra { target: 0x10 }, Pred { reg: 0, negate: false }),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert_eq!(inv.reg(1), 4);
    }

    #[test]
    fn ssy_and_sync_reconverge() {
        let program = prog_at(&[
            (Op::Ssy { target: 0x28 }, Pred::ALWAYS),
            (Op::Mov32i { dst: 1, imm: 7 }, Pred::ALWAYS),
            (Op::Sync, Pred::ALWAYS),
            (Op::Mov32i { dst: 2, imm: 9 }, Pred::ALWAYS), // at 0x28
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert_eq!(inv.reg(1), 7);
        assert_eq!(inv.reg(2), 9);
    }

    #[test]
    fn a_program_that_never_exits_fails_instead_of_hanging() {
        let program = prog_at(&[(Op::Bra { target: 0x08 }, Pred::ALWAYS)]);
        let mut inv = Invocation::new();
        assert!(inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).is_err());
    }

    #[test]
    fn kil_discards_the_fragment() {
        let program = prog_at(&[(Op::Kil, Pred::ALWAYS), (Op::Exit, Pred::ALWAYS)]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert!(inv.discarded);
    }

    #[test]
    fn integer_ops_use_the_registers_as_integers_not_floats() {
        // The register file is untyped; the same bits are an address here
        // and a float three instructions later, so an integer op must not
        // round-trip through f32.
        let program = prog_at(&[
            (Op::Mov32i { dst: 0, imm: 0x1234_5678 }, Pred::ALWAYS),
            (Op::Shr { dst: 1, a: 0, b: Operand::Imm(16), signed: false, wrap: false }, Pred::ALWAYS),
            (
                Op::Lop {
                    dst: 2,
                    a: 0,
                    ainv: false,
                    b: Operand::Imm(0xffff),
                    binv: false,
                    op: LogicOp::And,
                },
                Pred::ALWAYS,
            ),
            (Op::Iadd { dst: 3, a: 1, aneg: false, b: Operand::Reg(2), bneg: false }, Pred::ALWAYS),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert_eq!(inv.reg(1), 0x1234);
        assert_eq!(inv.reg(2), 0x5678);
        assert_eq!(inv.reg(3), 0x1234 + 0x5678);
    }

    #[test]
    fn lop3_evaluates_its_truth_table() {
        // lut 0xe8 is majority(a, b, c): true where at least two inputs are.
        assert_eq!(lop3(0b1100, 0b1010, 0b0110, 0xe8), 0b1110);
        // lut 0xf0 is "just a", 0xcc "just b", 0xaa "just c".
        assert_eq!(lop3(0xdead, 0xbeef, 0x1234, 0xf0), 0xdead);
        assert_eq!(lop3(0xdead, 0xbeef, 0x1234, 0xcc), 0xbeef);
        assert_eq!(lop3(0xdead, 0xbeef, 0x1234, 0xaa), 0x1234);
    }

    #[test]
    fn conversions_round_the_way_the_instruction_asks() {
        let program = prog_at(&[
            (Op::Mov32i { dst: 0, imm: (-2.5f32).to_bits() }, Pred::ALWAYS),
            (
                Op::F2i {
                    dst: 1,
                    src: Operand::Reg(0),
                    sm: FMod::NONE,
                    dst_bytes: 4,
                    dst_signed: true,
                    round: FRound::Trunc,
                    ftz: false,
                },
                Pred::ALWAYS,
            ),
            (
                Op::F2i {
                    dst: 2,
                    src: Operand::Reg(0),
                    sm: FMod::NONE,
                    dst_bytes: 4,
                    dst_signed: true,
                    round: FRound::Floor,
                    ftz: false,
                },
                Pred::ALWAYS,
            ),
            (
                Op::I2f {
                    dst: 3,
                    src: Operand::Reg(1),
                    sm: FMod::NONE,
                    src_bytes: 4,
                    src_signed: true,
                    sel: 0,
                },
                Pred::ALWAYS,
            ),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();
        assert_eq!(inv.reg(1) as i32, -2);
        assert_eq!(inv.reg(2) as i32, -3);
        assert_eq!(inv.reg_f32(3), -2.0);
    }

    #[test]
    fn rz_reads_as_zero_and_discards_writes() {
        let program = prog(&[
            Op::Fmul { dst: 0xff, a: 0, b: Operand::Reg(1), bm: FMod::NONE, ftz: true, sat: false },
            Op::Ffma {
                dst: 2,
                a: 0xff,
                b: Operand::Reg(1),
                bneg: false,
                c: Operand::Reg(5),
                cneg: false,
                ftz: true,
                sat: false,
            },
            Op::Exit,
        ]);
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 99.0);
        inv.set_reg_f32(1, 3.0);
        inv.set_reg_f32(5, 7.0);

        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();

        // dst=RZ: the write to r255 is discarded, not aliased to some slot.
        assert_eq!(inv.reg_f32(2), 0.0 * 3.0 + 7.0);
    }

    #[test]
    fn texs_resolves_its_handle_from_the_driver_constant_bank_and_writes_the_masked_channels() {
        // tex.frag's real shape, with the roles `isa`'s `decodes_texs` test
        // documents: dst is REG_00, coords are [REG_28 (unused), REG_08 (u),
        // REG_20 (v)].
        let program = prog(&[
            Op::Texs {
                dst: 2,
                dst2: 9,
                coords: [9, 0, 3], // coords[0] unused for t2d; u=r0, v=r3
                handle: 0x20,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
            },
            Op::Exit,
        ]);
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 0.25); // u
        inv.set_reg_f32(3, 0.75); // v

        let mut consts = HashMap::new();
        let handle = 7u32 | (2u32 << 20); // imageId=7, samplerId=2
        consts.insert((crate::gpu::texture::DRIVER_CONSTBUF_BANK, 0x20), f32::from_bits(handle));

        let textures = RecordingTextures {
            calls: RefCell::new(Vec::new()),
            color: [0.1, 0.2, 0.3, 0.4],
        };

        inv.execute(&program, &Env::new(&consts, &textures)).unwrap();

        assert_eq!(textures.calls.borrow().as_slice(), &[(handle, 0.25, 0.75)]);
        assert_eq!(inv.reg_f32(2), 0.1);
        assert_eq!(inv.reg_f32(3), 0.2);
        assert_eq!(inv.reg_f32(4), 0.3);
        assert_eq!(inv.reg_f32(5), 0.4);
    }

    #[test]
    fn solid_color_fragment_shader_reproduces_the_perspective_corrected_color() {
        // solid.frag: `oColor = vColor;` — a fixture from the same envydis
        // capture `isa`'s module docs cite, run end to end through the real
        // decoder. The rasterizer normally supplies attr_in already divided
        // by clip-w plus 1/w itself at a[0x7c]; we inject that directly here
        // since Stage 3 is scoped to the interpreter, not vertex fetch.
        let w = 2.0f32;
        let color = [0.25f32, 0.5, 0.75, 1.0];

        fn word(low: u32, high: u32) -> [u8; 8] {
            (((high as u64) << 32) | low as u64).to_le_bytes()
        }
        fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
            let mut out = Vec::with_capacity(32);
            out.extend_from_slice(&word(sched.0, sched.1));
            out.extend_from_slice(&word(a.0, a.1));
            out.extend_from_slice(&word(b.0, b.1));
            out.extend_from_slice(&word(c.0, c.1));
            out
        }
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
            (0xff87000f, 0xe2400fff),
            (0x00070f00, 0x50b00000),
        ));

        let program = decode_program(&bytes).unwrap();

        let mut inv = Invocation::new();
        inv.attr_in.set(0x7c, 1.0 / w);
        inv.attr_in.set(0x80, color[0] / w);
        inv.attr_in.set(0x84, color[1] / w);
        inv.attr_in.set(0x88, color[2] / w);
        inv.attr_in.set(0x8c, color[3] / w);

        inv.execute(&program, &Env::new(&no_consts(), &NoTextures)).unwrap();

        // Fragment output RT0 is registers r0-r3.
        assert_eq!(inv.reg_f32(0), color[0]);
        assert_eq!(inv.reg_f32(1), color[1]);
        assert_eq!(inv.reg_f32(2), color[2]);
        assert_eq!(inv.reg_f32(3), color[3]);
    }

    #[test]
    fn mvp_vertex_shader_transforms_a_known_position_via_a_fake_constant_buffer() {
        // mvp.vert: `gl_Position = uMVP * aPosition; vColor = aColor;` — the
        // Stage 0 fixture cited in `isa`'s module docs, run end to end
        // through the real decoder with a hand-picked matrix standing in for
        // a real bound constant buffer (real GPU-memory wiring is
        // `MemoryConstants`, exercised separately below).
        fn word(low: u32, high: u32) -> [u8; 8] {
            (((high as u64) << 32) | low as u64).to_le_bytes()
        }
        fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
            let mut out = Vec::with_capacity(32);
            out.extend_from_slice(&word(sched.0, sched.1));
            out.extend_from_slice(&word(a.0, a.1));
            out.extend_from_slice(&word(b.0, b.1));
            out.extend_from_slice(&word(c.0, c.1));
            out
        }
        let mut bytes = block(
            (0xfc20070f, 0x081f8441),
            (0x0807ff00, 0xefd9ff80), // ld b128 $r0 a[0x80] 0x0
            (0x00070004, 0x4c681008), // fmul ftz $r4 $r0 c2[0x0]
            (0x00170005, 0x4c681008), // fmul ftz $r5 $r0 c2[0x4]
        );
        bytes.extend(block(
            (0xfc6207e1, 0x081f8400),
            (0x00270006, 0x4c681008), // fmul ftz $r6 $r0 c2[0x8]
            (0x00370000, 0x4c681008), // fmul ftz $r0 $r0 c2[0xc]
            (0x00470104, 0x49a00208), // ffma ftz $r4 $r1 c2[0x10] $r4
        ));
        bytes.extend(block(
            (0xfc2207e1, 0x001f8c40),
            (0x00570105, 0x49a00288), // ffma ftz $r5 $r1 c2[0x14] $r5
            (0x00670106, 0x49a00308), // ffma ftz $r6 $r1 c2[0x18] $r6
            (0x00770100, 0x49a00008), // ffma ftz $r0 $r1 c2[0x1c] $r0
        ));
        bytes.extend(block(
            (0xfc2207e1, 0x081f8440),
            (0x00870201, 0x49a00208), // ffma ftz $r1 $r2 c2[0x20] $r4
            (0x00970204, 0x49a00288), // ffma ftz $r4 $r2 c2[0x24] $r5
            (0x00a70205, 0x49a00308), // ffma ftz $r5 $r2 c2[0x28] $r6
        ));
        bytes.extend(block(
            (0xfc2007e3, 0x081f8440),
            (0x00b70206, 0x49a00008), // ffma ftz $r6 $r2 c2[0x2c] $r0
            (0x00c70300, 0x49a00088), // ffma ftz $r0 $r3 c2[0x30] $r1
            (0x00d70301, 0x49a00208), // ffma ftz $r1 $r3 c2[0x34] $r4
        ));
        bytes.extend(block(
            (0xfcc207e1, 0x00038800),
            (0x00e70302, 0x49a00288), // ffma ftz $r2 $r3 c2[0x38] $r5
            (0x00f70303, 0x49a00308), // ffma ftz $r3 $r3 c2[0x3c] $r6
            (0x0707ff00, 0xeff1ff80), // st b128 a[0x70] $r0 0x0
        ));
        bytes.extend(block(
            (0x1c200f0f, 0x07ffbc01),
            (0x0907ff00, 0xefd9ff80), // ld b128 $r0 a[0x90] 0x0
            (0x0807ff00, 0xeff1ff80), // st b128 a[0x80] $r0 0x0
            (0x0007000f, 0xe3000000), // exit
        ));
        let program = decode_program(&bytes).unwrap();

        // A std140 mat4 is column-major: column c's four rows sit at bytes
        // [c*16, c*16+16). m[row][col] is the usual math notation.
        let m: [[f32; 4]; 4] = [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut consts: HashMap<(u8, u16), f32> = HashMap::new();
        for col in 0..4 {
            for row in 0..4 {
                consts.insert((2, (col * 16 + row * 4) as u16), m[row][col]);
            }
        }

        let pos = [10.0f32, 20.0, 30.0, 1.0];
        let color = [0.1f32, 0.2, 0.3, 0.4];
        let mut inv = Invocation::new();
        inv.attr_in.set(0x80, pos[0]);
        inv.attr_in.set(0x84, pos[1]);
        inv.attr_in.set(0x88, pos[2]);
        inv.attr_in.set(0x8c, pos[3]);
        inv.attr_in.set(0x90, color[0]);
        inv.attr_in.set(0x94, color[1]);
        inv.attr_in.set(0x98, color[2]);
        inv.attr_in.set(0x9c, color[3]);

        inv.execute(&program, &Env::new(&consts, &NoTextures)).unwrap();

        let expected = [
            (0..4).map(|c| m[0][c] * pos[c]).sum::<f32>(),
            (0..4).map(|c| m[1][c] * pos[c]).sum::<f32>(),
            (0..4).map(|c| m[2][c] * pos[c]).sum::<f32>(),
            (0..4).map(|c| m[3][c] * pos[c]).sum::<f32>(),
        ];
        assert_eq!(inv.attr_out.get(0x70), expected[0]);
        assert_eq!(inv.attr_out.get(0x74), expected[1]);
        assert_eq!(inv.attr_out.get(0x78), expected[2]);
        assert_eq!(inv.attr_out.get(0x7c), expected[3]);

        // vColor = aColor passthrough.
        assert_eq!(inv.attr_out.get(0x80), color[0]);
        assert_eq!(inv.attr_out.get(0x84), color[1]);
        assert_eq!(inv.attr_out.get(0x88), color[2]);
        assert_eq!(inv.attr_out.get(0x8c), color[3]);
    }

    #[test]
    fn memory_constants_reads_a_real_bound_buffer_out_of_gpu_memory() {
        use crate::gpu::syncpt::Host1x;
        use crate::gpu::vmm::AddressSpace;
        use crate::mem::Memory;

        let mut mem = Memory::new();
        mem.map_zero(0x5000_0000, 0x1000).unwrap();
        let mut vmm = AddressSpace::new();
        let gpu_va = vmm
            .map(0x5000_0000, 0x1000, 1, 0, crate::gpu::vmm::SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        vmm.write_u32(&mut mem, gpu_va + 0x10, 42.5f32.to_bits())
            .unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };

        let bindings = |bank: u8| if bank == 2 { Some((gpu_va, 0x1000)) } else { None };
        let source = MemoryConstants { ctx: &ctx, bindings: &bindings };

        assert_eq!(f32::from_bits(source.read_const(2, 0x10).unwrap()), 42.5);
        assert!(source.read_const(3, 0x10).is_err()); // unbound bank
        assert!(source.read_const(2, 0x1000).is_err()); // past the buffer's size
    }

    #[test]
    fn textured_fragment_shader_multiplies_the_real_sample_by_vertex_colour() {
        // tex.frag in full (the same real capture `isa`'s module docs and
        // `decodes_texs`'s test cite): `oColor = texture(uTex, vTexCoord) *
        // vColor;`. This is also the test that caught `texs`'s real
        // dst/coordinate roles (see `isa::decodes_texs`'s doc comment) —
        // with a solid vertex colour of (1,1,1,1) the expected output is
        // exactly the sampled texture colour, letting a wrong register
        // mapping surface immediately as a wrong result instead of a
        // plausible-looking wash of white.
        fn word(low: u32, high: u32) -> [u8; 8] {
            (((high as u64) << 32) | low as u64).to_le_bytes()
        }
        fn block(sched: (u32, u32), a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> Vec<u8> {
            let mut out = Vec::with_capacity(32);
            out.extend_from_slice(&word(sched.0, sched.1));
            out.extend_from_slice(&word(a.0, a.1));
            out.extend_from_slice(&word(b.0, b.1));
            out.extend_from_slice(&word(c.0, c.1));
            out
        }
        let mut bytes = block(
            (0xe1a0070f, 0x003c0401),
            (0xcff7ff00, 0xe003ff87), // ipa pass $r0 a[0x7c] 0x0 0x0 0x1
            (0x00470004, 0x50800000), // mufu rcp $r4 $r0
            (0x0047ff00, 0xe043ff89), // ipa $r0 a[0x90] $r4 0x0 0x1  (u)
        );
        bytes.extend(block(
            (0xe020072f, 0x001cbc03),
            (0x4047ff01, 0xe043ff89), // ipa $r1 a[0x94] $r4 0x0 0x1  (v)
            (0x20170000, 0xd8301a40), // texs $r2 $r0 $r0 $r1 0x1a4 t2d rgba
            (0x0047ff05, 0xe043ff88), // ipa $r5 a[0x80] $r4 0x0 0x1
        ));
        bytes.extend(block(
            (0xe1e01ff0, 0x003fc000),
            (0x00570000, 0x5c681000), // fmul ftz $r0 $r0 $r5
            (0x4047ff05, 0xe043ff88), // ipa $r5 a[0x84] $r4 0x0 0x1
            (0x00570101, 0x5c681000), // fmul ftz $r1 $r1 $r5
        ));
        bytes.extend(block(
            (0xfe00070f, 0x001c3c01),
            (0x8047ff05, 0xe043ff88), // ipa $r5 a[0x88] $r4 0x0 0x1
            (0x00570202, 0x5c681000), // fmul ftz $r2 $r2 $r5
            (0xc047ff04, 0xe043ff88), // ipa $r4 a[0x8c] $r4 0x0 0x1
        ));
        bytes.extend(block(
            (0xfde00ff0, 0x001ffc3f),
            (0x00470303, 0x5c681000), // fmul ftz $r3 $r3 $r4
            (0x0007000f, 0xe3000000), // exit
            (0xff87000f, 0xe2400fff), // bra (padding, never reached)
        ));
        let program = decode_program(&bytes).unwrap();

        struct StubTex;
        impl TextureSource for StubTex {
            fn sample(&self, _handle: u32, _u: f32, _v: f32) -> Result<[f32; 4]> {
                Ok([0.2, 0.4, 0.6, 0.8])
            }
        }

        let w = 2.0f32;
        let color = [1.0f32, 1.0, 1.0, 1.0];
        let mut inv = Invocation::new();
        inv.attr_in.set(0x7c, 1.0 / w);
        inv.attr_in.set(0x90, 0.5 / w); // u
        inv.attr_in.set(0x94, 0.5 / w); // v
        inv.attr_in.set(0x80, color[0] / w);
        inv.attr_in.set(0x84, color[1] / w);
        inv.attr_in.set(0x88, color[2] / w);
        inv.attr_in.set(0x8c, color[3] / w);

        let no_consts: HashMap<(u8, u16), f32> = HashMap::new();
        inv.execute(&program, &Env::new(&no_consts, &StubTex)).unwrap();

        assert_eq!(inv.reg_f32(0), 0.2);
        assert_eq!(inv.reg_f32(1), 0.4);
        assert_eq!(inv.reg_f32(2), 0.6);
        assert_eq!(inv.reg_f32(3), 0.8);
    }
}
