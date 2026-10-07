//! Executes decoded Maxwell instructions. [`Invocation`] is one vertex or
//! fragment run to completion on a scalar machine (255 untyped 32-bit GPRs, seven
//! predicates, the `ssy`/`pbk` reconvergence stack), so divergence needs no mask.

mod alu;
mod backends;
mod math;
mod memory;
mod regs;
#[cfg(test)]
mod tests;
mod warp;

pub use backends::*;
use math::*;
use regs::*;
pub use warp::*;

use crate::gpu::exec::ExecCtx;
use crate::gpu::shader::compiled::{Compiled, NO_TARGET};
use crate::{Error, Result};
use std::collections::HashMap;

use super::isa::{
    self, AtomOp, AtomSpace, AtomType, BarMode, BoolOp, FCmp, FMod, FRound, HMerge, HPrecision,
    HSwizzle, ICmp, LogicOp, LopTest, MemSize, MufuOp, Op, Operand, Pred, ShflMode, SurfaceData,
    SurfaceDim, TexDim, VoteMode, XmadC, RZ,
};
use crate::gpu::surface::{f16_to_f32, f32_to_f16};

/// Boxed so the `Result` fits in a register pair: [`Error`] is 56 bytes and this
/// is returned per operand per pixel.
pub type ShaderResult<T> = std::result::Result<T, Box<Error>>;

fn fault(message: String) -> Box<Error> {
    Box::new(Error::Gpu(message))
}

/// A CTA's shared memory (`s[]`), owned by the scheduler.
pub type SharedMemory = std::cell::RefCell<Vec<u8>>;

/// What `s2r` reads; all zero for a draw.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecialRegs {
    /// Lane within the warp (a fragment quad's pixel); what shuffles are relative to.
    pub lane: u32,
    pub tid: [u32; 3],
    pub ctaid: [u32; 3],
    pub shared_size: u32,
    pub local_size: u32,
    /// Window y grows upward, so `SR_Y_DIRECTION` reads -1.0. Follows `SET_WINDOW_ORIGIN_MODE`.
    pub y_negate: bool,
}

impl SpecialRegs {
    /// `SR_TID`/`SR_NTID` packed forms: refused, as the packing is unconfirmed.
    const PACKED: [u8; 2] = [0x20, 0x28];

    /// The value of special register `sr`, or `None` if not modelled (read as zero).
    pub fn read(&self, sr: u8) -> Option<u32> {
        Some(match sr {
            0x00 => self.lane,
            0x21..=0x23 => self.tid[(sr - 0x21) as usize],
            0x25..=0x27 => self.ctaid[(sr - 0x25) as usize],
            0x32 => self.shared_size,
            // `SR_Y_DIRECTION` is a float sign.
            0x12 => {
                if self.y_negate {
                    (-1.0f32).to_bits()
                } else {
                    1.0f32.to_bits()
                }
            }
            0x36 => self.local_size,
            _ => return None,
        })
    }
}

/// Instruction cap per invocation, so a mis-executed loop fails instead of hanging.
const MAX_STEPS: usize = 1 << 20;

const LOCAL_MEMORY_BYTES: usize = 1024;

pub struct Env<'a> {
    pub consts: &'a dyn ConstantSource,
    pub textures: &'a dyn TextureSource,
    pub memory: Option<&'a dyn GlobalMemory>,
    /// The CTA's shared memory; a draw has none.
    pub shared: Option<&'a SharedMemory>,
    pub special: SpecialRegs,
    /// The constant bank `texs` handles come from: `TexCbIndex`
    /// ([`crate::gpu::engine::threed::Engine3D::tex_cb_index`]).
    pub tex_cb_index: u8,
}

impl<'a> Env<'a> {
    /// Uses nouveau's texture bank, as the fixtures do; see [`Env::with_tex_cb_index`].
    pub fn new(consts: &'a dyn ConstantSource, textures: &'a dyn TextureSource) -> Env<'a> {
        Env::with_tex_cb_index(consts, textures, crate::gpu::texture::NOUVEAU_TEX_CB_INDEX)
    }

    pub fn with_tex_cb_index(
        consts: &'a dyn ConstantSource,
        textures: &'a dyn TextureSource,
        tex_cb_index: u8,
    ) -> Env<'a> {
        Env {
            consts,
            textures,
            memory: None,
            shared: None,
            special: SpecialRegs::default(),
            tex_cb_index,
        }
    }
}

#[derive(Debug)]
/// The `a[]` attribute space, flat and addressed by byte offset (`0x000..0x400`).
/// Offsets past it read zero and drop writes. The written-mask distinguishes
/// "unwritten" from zero (unwritten `clip.w` defaults to 1.0).
#[derive(Clone)]
pub struct Attributes {
    words: [f32; Attributes::WORDS],
    written: [u64; Attributes::WORDS / 64],
}

impl Attributes {
    const WORDS: usize = 0x400 / 4;

    /// The value at `offset`, or 0.0 if unwritten.
    pub fn get(&self, offset: u16) -> f32 {
        self.written(offset).unwrap_or(0.0)
    }

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

    /// Only the mask needs clearing.
    pub fn clear(&mut self) {
        self.written = [0; Self::WORDS / 64];
    }
}

impl Default for Attributes {
    fn default() -> Self {
        Attributes {
            words: [0.0; Attributes::WORDS],
            written: [0; Attributes::WORDS / 64],
        }
    }
}

pub struct Invocation {
    /// 256 so `RZ` has a slot (kept zero) and `u8` indexing needs no bounds check.
    gpr: [u32; 256],
    /// `p0`..`p6`; `p7` is `PT`, always true.
    pred: [bool; 7],
    pub attr_in: Attributes,
    pub attr_out: Attributes,
    /// The carry `iadd.cc` sets and `iadd.x` reads.
    carry: bool,
    /// Zero, sign and overflow condition codes; see [`isa::flow_test`].
    zero: bool,
    sign: bool,
    overflow: bool,
    /// Set by `kil`.
    pub discarded: bool,
    /// `ssy`/`pbk`/`pcnt` push a resume address; `sync`/`brk`/`cont` pop it.
    stack: Vec<u32>,
    local: Vec<u8>,
    /// Size of `l[]`; a dispatch sets it from the QMD.
    local_bytes: usize,
    /// Kept here so a `bar` can suspend mid-program.
    pc: usize,
    /// Instructions retired against [`MAX_STEPS`], across suspensions.
    steps: usize,
    /// Texture results not yet landed; see `run_texs`.
    pending: Vec<(usize, u8, u32)>,
    /// The shuffle or vote awaiting the rest of the warp; see [`resolve_warp`].
    exchange: Option<Exchange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exchange {
    Shuffle(Shuffle),
    Vote(Vote),
}

/// A `vote` with its source predicate read, waiting for the warp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Vote {
    mode: VoteMode,
    dst: u8,
    pred: u8,
    holds: bool,
}

/// A decoded `shfl` with operands read, waiting for its source lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shuffle {
    mode: ShflMode,
    dst: u8,
    pred: u8,
    src: u8,
    index: u32,
    mask: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Halt {
    /// It ran to `exit`, or `kil` discarded it.
    Exited,
    /// It reached a `bar` and is waiting for the rest of its CTA.
    Barrier,
    /// It reached a `shfl` or `vote`; [`resolve_warp`] releases it.
    Warp,
}

impl Default for Invocation {
    fn default() -> Self {
        Invocation {
            gpr: [0; 256],
            pred: [false; 7],
            carry: false,
            zero: false,
            sign: false,
            overflow: false,
            attr_in: Attributes::default(),
            attr_out: Attributes::default(),
            discarded: false,
            stack: Vec::new(),
            local: Vec::new(),
            local_bytes: LOCAL_MEMORY_BYTES,
            pc: 0,
            steps: 0,
            pending: Vec::new(),
            exchange: None,
        }
    }
}

impl Invocation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to the initial state so one invocation can serve a whole draw.
    pub fn reset(&mut self) {
        self.gpr = [0; 256];
        self.pred = [false; 7];
        self.attr_in.clear();
        self.attr_out.clear();
        self.discarded = false;
        self.stack.clear();
        self.local.clear();
        self.pc = 0;
        self.steps = 0;
        self.pending.clear();
        self.exchange = None;
    }

    pub fn set_local_bytes(&mut self, bytes: usize) {
        self.local_bytes = bytes.max(LOCAL_MEMORY_BYTES);
        self.local.clear();
    }

    pub fn reg_f32(&self, r: u8) -> f32 {
        f32::from_bits(self.reg(r))
    }

    pub fn set_reg_f32(&mut self, r: u8, v: f32) {
        self.set_reg(r, v.to_bits());
    }

    pub fn reg(&self, r: u8) -> u32 {
        self.gpr[r as usize]
    }

    pub fn set_reg(&mut self, r: u8, v: u32) {
        self.gpr[r as usize] = v;
        // An unconditional store is cheaper than a branch.
        self.gpr[RZ as usize] = 0;
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

    fn holds(&self, p: Pred) -> bool {
        self.pred(p.reg) != p.negate
    }

    #[inline(always)]
    fn operand(&self, op: Operand, env: &Env) -> ShaderResult<u32> {
        match op {
            Operand::Reg(r) => Ok(self.reg(r)),
            Operand::Imm(v) => Ok(v),
            Operand::Const { bank, offset } => env.consts.read_const(bank, offset),
        }
    }

    #[inline(always)]
    fn operand_f32(&self, op: Operand, env: &Env) -> ShaderResult<f32> {
        Ok(f32::from_bits(self.operand(op, env)?))
    }

    /// Execute `program` until it exits; a `bar` is an error (see [`Invocation::resume`]).
    pub fn execute(&mut self, program: &Compiled, env: &Env) -> Result<()> {
        self.begin();
        match self.resume(program, env)? {
            Halt::Exited => Ok(()),
            Halt::Barrier => Err(Error::Gpu(format!(
                "shader: bar at {:#x} outside a compute dispatch, where there is no CTA to \
                 synchronise with",
                program.offset(self.pc.saturating_sub(1))
            ))),
            Halt::Warp => Err(Error::Gpu(format!(
                "shader: the warp instruction at {:#x} reads another lane, and this \
                 invocation is running on its own",
                program.offset(self.pc.saturating_sub(1))
            ))),
        }
    }

    /// Back to the entry point, keeping registers.
    pub fn begin(&mut self) {
        self.pc = 0;
        self.steps = 0;
        self.pending.clear();
        self.exchange = None;
    }

    /// Run until exit or a barrier, continuing from the last stop.
    pub fn resume(&mut self, program: &Compiled, env: &Env) -> Result<Halt> {
        if program.is_empty() {
            return Err(Error::Gpu("shader: executing an empty program".into()));
        }
        // Moved out so the loop can hold `&mut self`; restored before every return.
        let mut pending = std::mem::take(&mut self.pending);
        let out = self.run(program, env, &mut pending);
        self.pending = pending;
        // Unbox at the boundary; everything below is per instruction.
        out.map_err(|boxed| *boxed)
    }

    fn run(
        &mut self,
        program: &Compiled,
        env: &Env,
        pending: &mut Vec<(usize, u8, u32)>,
    ) -> ShaderResult<Halt> {
        let mut pc = self.pc;
        let mut steps = self.steps;
        // Sliced to one length so one bounds check covers both reads.
        let len = program.len();
        let ops = &program.ops()[..len];
        let preds = &program.preds()[..len];

        loop {
            if pc >= len {
                return Err(fault(
                    "shader: ran off the end of the program without an exit".into(),
                ));
            }
            steps += 1;
            self.steps = steps;
            if steps > MAX_STEPS {
                return Err(fault(format!(
                    "shader: did not terminate within {MAX_STEPS} instructions"
                )));
            }
            // `retain` is a real call even when empty.
            if !pending.is_empty() {
                pending.retain(|&(due, reg, val)| {
                    if due == pc {
                        self.set_reg(reg, val);
                        false
                    } else {
                        true
                    }
                });
            }

            if !self.holds(preds[pc]) {
                pc += 1;
                continue;
            }
            let op = ops[pc];

            // A jump flushes deferred texture writes, placed assuming program order.
            self.pc = pc;
            let jump = |index: u32, pending: &mut Vec<(usize, u8, u32)>, inv: &mut Self| {
                for (_, reg, val) in pending.drain(..) {
                    inv.set_reg(reg, val);
                }
                if index == NO_TARGET {
                    return Err(Error::Gpu(format!(
                        "shader: branch at {:#x} goes somewhere that was never decoded",
                        program.offset(pc)
                    )));
                }
                Ok(index as usize)
            };

            match op {
                Op::Exit => {
                    for (_, reg, val) in pending.drain(..) {
                        self.set_reg(reg, val);
                    }
                    return Ok(Halt::Exited);
                }
                Op::Kil => {
                    self.discarded = true;
                    return Ok(Halt::Exited);
                }
                // Advance past the barrier before suspending.
                Op::Bar { mode } => match mode {
                    BarMode::Sync | BarMode::Arrive => {
                        self.pc = pc + 1;
                        return Ok(Halt::Barrier);
                    }
                    other => {
                        return Err(fault(format!(
                            "shader: bar.{other:?} at {:#x} reduces a value across a warp's \
                             lanes, which a scalar interpreter has none of",
                            program.offset(pc)
                        )))
                    }
                },
                // Advance past it like a barrier; deferred texture writes stay deferred.
                Op::Shfl {
                    dst,
                    pred,
                    src,
                    index,
                    mask,
                    mode,
                } => {
                    self.exchange = Some(Exchange::Shuffle(Shuffle {
                        mode,
                        dst,
                        pred,
                        src,
                        index: self.operand(index, env)?,
                        mask: self.operand(mask, env)?,
                    }));
                    self.pc = pc + 1;
                    return Ok(Halt::Warp);
                }
                Op::Vote {
                    dst,
                    pred,
                    src,
                    mode,
                } => {
                    self.exchange = Some(Exchange::Vote(Vote {
                        mode,
                        dst,
                        pred,
                        holds: self.holds(src),
                    }));
                    self.pc = pc + 1;
                    return Ok(Halt::Warp);
                }
                Op::Nop | Op::Inert => {}
                Op::Bra { .. } => {
                    pc = jump(program.target(pc), pending, self)?;
                    continue;
                }
                // The target is a register value, so it needs a lookup.
                Op::Brx { base, reg } => {
                    let at = super::align_slot(base.wrapping_add(self.reg(reg)));
                    let index = program.index_of(at).map(|i| i as u32).unwrap_or(NO_TARGET);
                    if index == NO_TARGET {
                        return Err(fault(format!(
                            "shader: branch to {at:#x}, which was never decoded"
                        )));
                    }
                    pc = jump(index, pending, self)?;
                    continue;
                }
                Op::Ssy { .. } | Op::Pbk { .. } | Op::Pcnt { .. } => {
                    self.stack.push(program.target(pc));
                }
                Op::Sync | Op::Brk | Op::Cont => {
                    let target = self.stack.pop().ok_or_else(|| {
                        Error::Gpu(format!(
                            "shader: sync/brk/cont at {:#x} with an empty reconvergence stack",
                            program.offset(pc)
                        ))
                    })?;
                    pc = jump(target, pending, self)?;
                    continue;
                }
                Op::Texs { .. } => {
                    self.run_texs(program, pc, op, env, pending)?;
                }
                Op::Tex { .. } => {
                    self.run_tex(program, pc, op, env, pending)?;
                }
                Op::Txq { .. } => {
                    self.run_txq(program, pc, op, env, pending)?;
                }
                Op::Tld4 { .. } => {
                    self.run_tld4(program, pc, op, env, pending)?;
                }
                Op::Suld {
                    coords,
                    handle,
                    handle_reg,
                    dim,
                    data,
                    ..
                } => {
                    let handle = self.surface_handle(handle, handle_reg, env)?;
                    let at = self.surface_coords(coords, dim);
                    let regs = env.textures.surface_load(handle, at, data)?;
                    self.land_texture(program, pc, regs.map(f32::from_bits), pending);
                }
                Op::Sust {
                    src,
                    coords,
                    handle,
                    handle_reg,
                    dim,
                    data,
                } => {
                    let handle = self.surface_handle(handle, handle_reg, env)?;
                    let at = self.surface_coords(coords, dim);
                    let words = surface_source_words(data);
                    let regs = std::array::from_fn(|i| {
                        if i < words {
                            self.reg(src.wrapping_add(i as u8))
                        } else {
                            0
                        }
                    });
                    env.textures.surface_store(handle, at, data, regs)?;
                }
                other => self.run_alu(other, env)?,
            }
            pc += 1;
        }
    }
}

/// Where each `texs` result in `ops` lands; see [`super::Program::texs_writes`].
pub(super) fn texs_writes_for(ops: &[Op]) -> Vec<super::TexsWrites> {
    let mut out = Vec::new();
    for (pc, op) in ops.iter().enumerate() {
        let destinations = match *op {
            Op::Texs {
                dst,
                dst2,
                mask,
                f16,
                ..
            } => isa::texs_destinations(dst, dst2, mask, f16),
            // `tex` channels land in consecutive registers from `dst`, one per mask bit.
            Op::Tex { dst, mask, .. } | Op::Txq { dst, mask, .. } | Op::Tld4 { dst, mask, .. } => {
                consecutive_destinations(dst, mask)
            }
            // A surface load lands the same way.
            Op::Suld { dst, data, .. } => consecutive_destinations(dst, data.channels()),
            _ => continue,
        };
        let writes = destinations
            .into_iter()
            .map(|(reg, store)| {
                let due = first_use_after(ops, pc + 1, reg).unwrap_or(ops.len() - 1);
                (reg, store, due)
            })
            .collect();
        out.push(super::TexsWrites { at: pc, writes });
    }
    out
}

pub(super) fn writes(op: &Op) -> Vec<u8> {
    match *op {
        Op::Ld { dst, size, .. }
        | Op::Ldg { dst, size, .. }
        | Op::Ldl { dst, size, .. }
        | Op::Ldc { dst, size, .. } => (0..size.regs()).map(|i| dst.wrapping_add(i)).collect(),
        Op::Ipa { dst, .. }
        | Op::Mufu { dst, .. }
        | Op::Rro { dst, .. }
        | Op::Fadd { dst, .. }
        | Op::Fmul { dst, .. }
        | Op::Ffma { dst, .. }
        | Op::Fmnmx { dst, .. }
        | Op::Fset { dst, .. }
        | Op::Hadd2 { dst, .. }
        | Op::Hmul2 { dst, .. }
        | Op::Hfma2 { dst, .. }
        | Op::Hset2 { dst, .. }
        | Op::Mov { dst, .. }
        | Op::Mov32i { dst, .. }
        | Op::S2r { dst, .. }
        | Op::Iadd { dst, .. }
        | Op::Iadd3 { dst, .. }
        | Op::Imnmx { dst, .. }
        | Op::Vmnmx { dst, .. }
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
        | Op::I2i { dst, .. }
        | Op::Shfl { dst, .. }
        | Op::Vote { dst, .. }
        | Op::Fswzadd { dst, .. } => vec![dst],
        Op::Texs {
            dst,
            dst2,
            mask,
            f16,
            ..
        } => isa::texs_destinations(dst, dst2, mask, f16)
            .into_iter()
            .map(|(reg, _)| reg)
            .collect(),
        // One register per set mask bit, consecutive from `dst`.
        Op::Tex { dst, mask, .. } | Op::Txq { dst, mask, .. } | Op::Tld4 { dst, mask, .. } => (0
            ..mask.iter().filter(|&&m| m).count() as u8)
            .map(|i| dst.wrapping_add(i))
            .collect(),
        Op::Suld { dst, data, .. } => (0..data.channels().iter().filter(|&&m| m).count() as u8)
            .map(|i| dst.wrapping_add(i))
            .collect(),
        _ => Vec::new(),
    }
}
