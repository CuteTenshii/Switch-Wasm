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
use crate::gpu::shader::compiled::{Compiled, NO_TARGET};
use crate::{Error, Result};
use std::collections::HashMap;

use super::isa::{
    self, AtomOp, AtomSpace, AtomType, BarMode, BoolOp, FCmp, FMod, FRound, HMerge, HPrecision,
    HSwizzle, ICmp, LogicOp, LopTest, MemSize, MufuOp, Op, Operand, Pred, ShflMode, TexDim, XmadC,
    RZ,
};
use crate::gpu::surface::{f16_to_f32, f32_to_f16};

/// What everything the interpreter runs per instruction returns.
///
/// The error is boxed because [`Error`] is 56 bytes, which is too big to come
/// back in registers: a `Result<u32, Error>` is written to the stack through a
/// hidden pointer, and an operand read returns one of those *per source, per
/// instruction, per covered pixel*. That was 8% of a Home Menu frame. Boxed,
/// the whole `Result` is a register pair, and nothing allocates until a draw
/// is about to stop. `?` converts an [`Error`] into one on its own.
pub type ShaderResult<T> = std::result::Result<T, Box<Error>>;

/// A shader fault, boxed for [`ShaderResult`].
fn fault(message: String) -> Box<Error> {
    Box::new(Error::Gpu(message))
}

/// Resolves a `cN[offset]` operand to its raw 32 bits. `bank` is whatever the
/// ISA's `Operand::Const` carries — for real programs that's a constant-buffer
/// *bind slot* (`Bind[]`'s index, not a raw GPU address), so a real source
/// still needs its own way to turn that into bytes; see [`MemoryConstants`].
/// Reads are fallible because a real one touches guest memory.
pub trait ConstantSource {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32>;
}

impl ConstantSource for HashMap<(u8, u16), f32> {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32> {
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
    /// Values already read, shared across the draw — see [`ConstCache`].
    pub cache: &'a std::cell::RefCell<ConstCache>,
}

/// How many distinct constants [`ConstCache`] holds. Direct-mapped, so this is
/// a power of two and the index is the low bits of the key.
const CONST_CACHE_SLOTS: usize = 512;

/// The constants a draw has already read.
///
/// A constant buffer cannot change while a draw is running — the GPU processes
/// methods in order, and a draw is one method — so every pixel of a draw reads
/// the same handful of values out of the same buffers, and each read otherwise
/// costs a bank lookup, a bounds check, a GPU address translation and a guest
/// memory access. Shading a full-screen quad paid that 921 600 times per
/// constant.
pub struct ConstCache {
    /// `(key, value)`, where the key packs bank and offset. `u32::MAX` is
    /// empty: a real key is at most `31 << 16 | 0xffff`.
    slots: Box<[(u32, u32); CONST_CACHE_SLOTS]>,
}

impl Default for ConstCache {
    fn default() -> ConstCache {
        ConstCache {
            slots: Box::new([(u32::MAX, 0); CONST_CACHE_SLOTS]),
        }
    }
}

impl ConstCache {
    #[inline]
    pub(crate) fn key(bank: u8, offset: u16) -> u32 {
        (bank as u32) << 16 | offset as u32
    }

    #[inline]
    pub(crate) fn get(&self, key: u32) -> Option<u32> {
        let slot = self.slots[key as usize % CONST_CACHE_SLOTS];
        (slot.0 == key).then_some(slot.1)
    }

    #[inline]
    pub(crate) fn insert(&mut self, key: u32, value: u32) {
        self.slots[key as usize % CONST_CACHE_SLOTS] = (key, value);
    }
}

impl ConstantSource for MemoryConstants<'_, '_> {
    fn read_const(&self, bank: u8, offset: u16) -> ShaderResult<u32> {
        let key = ConstCache::key(bank, offset);
        if let Some(value) = self.cache.borrow().get(key) {
            return Ok(value);
        }
        let (addr, size) = (self.bindings)(bank)
            .ok_or_else(|| Error::Gpu(format!("shader: read from unbound constant bank {bank}")))?;
        if offset as u32 + 4 > size {
            return Err(fault(format!(
                "shader: constant read c{bank}[{offset:#x}] is past the bound buffer's size {size:#x}"
            )));
        }
        let value = self.ctx.read_u32(addr + offset as u64)?;
        self.cache.borrow_mut().insert(key, value);
        Ok(value)
    }
}

/// Reads `ldg`'s global memory straight out of the channel's address space.
///
/// A Maxwell shader addresses global memory by full 64-bit GPU virtual
/// address, which it builds itself out of a constant-buffer pair with
/// `iadd.cc`/`iadd.x` — so there is nothing to bind and no window to set up.
/// Translating the address is the whole implementation, and it is the same
/// translation every other GPU read goes through.
pub struct MemoryGlobal<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
}

impl GlobalMemory for MemoryGlobal<'_, '_> {
    fn read_u32(&self, addr: u64) -> ShaderResult<u32> {
        Ok(self.ctx.read_u32(addr)?)
    }
}

/// Resolves a `texs` sample. `handle` is the packed `imageId | samplerId <<
/// 20` value a real one reads out of the driver's reserved constant bank
/// (see `gpu::texture`'s module docs) — [`Invocation::execute`] does that
/// read itself via [`ConstantSource`] before calling this, so this trait only
/// needs to turn a resolved handle plus UVs into a colour.
pub trait TextureSource {
    /// Sample `handle` at `(u, v)` of array layer `layer` — 0 for everything
    /// that is not a 2D array.
    fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]>;

    /// Sample a 3D image, whose third coordinate is normalized rather than a
    /// layer index. Defaulted, like the two below, so that a source with
    /// none of them stays a one-method implementation.
    fn sample_3d(&self, handle: u32, _u: f32, _v: f32, _w: f32) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: 3D sample of handle {handle:#x} with no 3D source bound"
        )))
    }

    /// Sample a cubemap, whose three coordinates are a direction.
    fn sample_cube(&self, handle: u32, _s: f32, _t: f32, _r: f32) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: cube sample of handle {handle:#x} with no cube source bound"
        )))
    }

    /// A shadow sample: how `reference` compares against the depth there,
    /// as `[c, c, c, 1.0]`. Defaulted to an error so that a source with no
    /// depth textures behind it stays a one-method implementation.
    fn sample_compare(
        &self,
        handle: u32,
        _u: f32,
        _v: f32,
        _layer: u32,
        _reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: shadow sample of handle {handle:#x} with no depth source bound"
        )))
    }
}

/// No texture backend at all — every `texs` is an error. Correct for vertex
/// shading and for tests that don't exercise `texs`.
pub struct NoTextures;

impl TextureSource for NoTextures {
    fn sample(&self, handle: u32, _u: f32, _v: f32, _layer: u32) -> ShaderResult<[f32; 4]> {
        Err(fault(format!(
            "shader: texture sample of handle {handle:#x} with no texture source bound"
        )))
    }
}

/// Samples a texture out of the real TIC/TSC descriptor pools in GPU memory.
pub struct MemoryTextures<'a, 'b> {
    pub ctx: &'a ExecCtx<'b>,
    pub tex_header_pool: u64,
    pub tex_sampler_pool: u64,
    /// The descriptors already parsed for this draw, keyed by handle.
    ///
    /// A TIC and a TSC are eight `u32` reads through the GPU address space
    /// that decode to the same thing for every pixel of a draw, so parsing
    /// them per sample was pure repetition: a full-screen textured quad paid
    /// for it 921 600 times. The cache lives in the caller so that this
    /// struct can still be rebuilt per fragment — it borrows `ctx`, which
    /// the pixel loop needs mutably between shading calls.
    pub descriptors: &'a std::cell::RefCell<crate::IdMap<u32, crate::gpu::texture::Descriptors>>,
    /// Decoded compressed blocks, shared by every fragment of the draw for the
    /// same reason `descriptors` is.
    pub blocks: &'a std::cell::RefCell<crate::gpu::texture::BlockCache>,
}

impl MemoryTextures<'_, '_> {
    /// The TIC/TSC pair `handle` resolves to, parsed once per draw.
    fn descriptors_for(&self, handle: u32) -> ShaderResult<crate::gpu::texture::Descriptors> {
        if let Some(d) = self.descriptors.borrow().get(&handle).copied() {
            return Ok(d);
        }
        let d = crate::gpu::texture::read_descriptors(
            self.ctx,
            self.tex_header_pool,
            self.tex_sampler_pool,
            handle,
        )?;
        self.descriptors.borrow_mut().insert(handle, d);
        Ok(d)
    }
}

impl TextureSource for MemoryTextures<'_, '_> {
    fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]> {
        let descriptors = self.descriptors_for(handle)?;
        Ok(crate::gpu::texture::sample_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            layer,
            self.blocks,
        )?)
    }

    fn sample_3d(&self, handle: u32, u: f32, v: f32, w: f32) -> ShaderResult<[f32; 4]> {
        let descriptors = self.descriptors_for(handle)?;
        Ok(crate::gpu::texture::sample_3d_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            w as f64,
            self.blocks,
        )?)
    }

    fn sample_cube(&self, handle: u32, s: f32, t: f32, r: f32) -> ShaderResult<[f32; 4]> {
        let descriptors = self.descriptors_for(handle)?;
        Ok(crate::gpu::texture::sample_cube_with(
            self.ctx,
            &descriptors,
            s as f64,
            t as f64,
            r as f64,
            self.blocks,
        )?)
    }

    fn sample_compare(
        &self,
        handle: u32,
        u: f32,
        v: f32,
        layer: u32,
        reference: f32,
    ) -> ShaderResult<[f32; 4]> {
        let descriptors = self.descriptors_for(handle)?;
        Ok(crate::gpu::texture::sample_compare_with(
            self.ctx,
            &descriptors,
            u as f64,
            v as f64,
            layer,
            reference,
            self.blocks,
        )?)
    }
}

/// A shader's global (`ldg`/`stg`/`atom`) address space. Optional: a program
/// that never issues one doesn't need a backend, and a program that does
/// without one gets an error naming the address rather than a silent zero.
///
/// Writes take `&self` because a compute dispatch shares one of these across
/// every thread of the grid while the interpreter holds it — the backend owns
/// whatever interior mutability that needs. They default to an error so that
/// a read-only source (the rasterizer's) stays a one-method implementation.
pub trait GlobalMemory {
    fn read_u32(&self, addr: u64) -> ShaderResult<u32>;

    fn read_u8(&self, addr: u64) -> ShaderResult<u8> {
        Ok((self.read_u32(addr & !3)? >> ((addr % 4) * 8)) as u8)
    }

    fn write_u32(&self, addr: u64, _value: u32) -> ShaderResult<()> {
        Err(fault(format!(
            "shader: a global store to {addr:#x} from a stage whose memory is read-only"
        )))
    }

    fn write_u8(&self, addr: u64, value: u8) -> ShaderResult<()> {
        let word = addr & !3;
        let shift = (addr % 4) * 8;
        let old = self.read_u32(word)?;
        self.write_u32(word, (old & !(0xFF << shift)) | (u32::from(value) << shift))
    }
}

/// A CTA's shared memory (`s[]`). Every thread of one CTA sees the same
/// bytes, which is the entire point of it, so the scheduler owns it and hands
/// each thread a reference.
pub type SharedMemory = std::cell::RefCell<Vec<u8>>;

/// What `s2r` reads.
///
/// Zero for every field is right for a draw: a vertex or fragment invocation
/// has no thread or CTA identity, which is what this returned before compute
/// gave the registers meaning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecialRegs {
    /// This invocation's lane within its warp — which of a fragment quad's
    /// four pixels it is shading, and the one thing a warp shuffle is
    /// relative to. Zero for anything run on its own, which is what every
    /// caller but the quad shader and a cooperative dispatch does.
    pub lane: u32,
    /// This thread's position in its CTA.
    pub tid: [u32; 3],
    /// This CTA's position in the grid.
    pub ctaid: [u32; 3],
    /// Bytes of shared memory the CTA was given.
    pub shared_size: u32,
    /// Bytes of local memory the thread was given.
    pub local_size: u32,
    /// Whether window y grows upward, which is what `SR_Y_DIRECTION` reports
    /// as -1.0 rather than +1.0. Follows `SET_WINDOW_ORIGIN_MODE`.
    pub y_negate: bool,
}

impl SpecialRegs {
    /// `SR_TID`/`SR_NTID`, the forms that pack all three dimensions into one
    /// register. Compilers emit the per-dimension registers instead, and the
    /// packing here has not been confirmed against hardware, so reading one
    /// is refused rather than answered with a layout that might be wrong.
    const PACKED: [u8; 2] = [0x20, 0x28];

    /// The value of special register `sr`, or `None` if this doesn't model it
    /// — which the caller answers with zero, as every one of them was
    /// answered before.
    pub fn read(&self, sr: u8) -> Option<u32> {
        Some(match sr {
            0x00 => self.lane,
            0x21..=0x23 => self.tid[(sr - 0x21) as usize],
            0x25..=0x27 => self.ctaid[(sr - 0x25) as usize],
            0x32 => self.shared_size,
            // `SR_Y_DIRECTION` is a float, and the sign is the whole content:
            // a shader that derives a screen-space direction multiplies by it,
            // so the zero every unmodelled special register used to read
            // collapsed that direction to nothing rather than reversing it.
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
    /// The CTA's shared memory, for the `lds`/`sts`/`atoms` a compute
    /// dispatch issues. A draw has none.
    pub shared: Option<&'a SharedMemory>,
    pub special: SpecialRegs,
    /// Which constant bank a `texs`'s immediate indexes for its texture
    /// handle — `TexCbIndex`, which the driver programs (see
    /// [`crate::gpu::engine::threed::Engine3D::tex_cb_index`]).
    pub tex_cb_index: u8,
}

impl<'a> Env<'a> {
    /// An environment whose `texs` handles come from nouveau's bank, which
    /// is what the fixtures in this module's tests are captured from. A real
    /// draw uses [`Env::with_tex_cb_index`] with the register's value.
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
        Attributes {
            words: [0.0; Attributes::WORDS],
            written: [0; Attributes::WORDS / 64],
        }
    }
}

pub struct Invocation {
    /// 256 rather than 255 so `RZ` has a slot of its own: indexing by a `u8`
    /// is then provably in bounds, which takes the check *and* the branch off
    /// the hottest read in the interpreter. The slot is kept at zero.
    gpr: [u32; 256],
    /// `p0`..`p6`. `p7` is `PT`, which always reads true and can't be
    /// written, so it isn't stored.
    pred: [bool; 7],
    /// `a[]` input and output.
    pub attr_in: Attributes,
    pub attr_out: Attributes,
    /// The carry `iadd.cc` leaves behind and `iadd.x` reads. One flag, not one
    /// per thread lane: this interpreter runs a single invocation at a time.
    carry: bool,
    /// Set by `kil`: this fragment must not be written.
    pub discarded: bool,
    /// `ssy`/`pbk`/`pcnt` push a resume address; `sync`/`brk`/`cont` pop it.
    stack: Vec<u32>,
    local: Vec<u8>,
    /// How much `l[]` this invocation gets. A dispatch sets it from the QMD.
    local_bytes: usize,
    /// Where execution is. In the struct rather than in [`Invocation::resume`]
    /// because a `bar` suspends an invocation mid-program and the scheduler
    /// resumes it once every other thread of the CTA has arrived.
    pc: usize,
    /// Instructions retired, against [`MAX_STEPS`]. Spans suspensions, or a
    /// program that loops around a barrier would get a fresh budget each time.
    steps: usize,
    /// Texture results not yet landed; see `run_texs`.
    pending: Vec<(usize, u8, u32)>,
    /// The shuffle this invocation is suspended on, waiting for the rest of
    /// its warp to reach one too — see [`resolve_shuffles`].
    shuffle: Option<Shuffle>,
}

/// A `shfl` that has been decoded and had its operands read, and is waiting
/// for the lane it names to be readable.
///
/// The operands are resolved here rather than at resolution time because
/// only the invocation can read its own registers and constants; what is
/// left is a question about lanes, which only the warp can answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shuffle {
    mode: ShflMode,
    dst: u8,
    pred: u8,
    src: u8,
    index: u32,
    mask: u32,
}

/// Why an invocation stopped running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Halt {
    /// It ran to `exit`, or `kil` discarded it.
    Exited,
    /// It reached a `bar` and is waiting for the rest of its CTA.
    Barrier,
    /// It reached a `shfl` and is waiting for the rest of its warp, which
    /// [`resolve_shuffles`] releases it from.
    Shuffle,
}

impl Default for Invocation {
    fn default() -> Self {
        Invocation {
            gpr: [0; 256],
            pred: [false; 7],
            carry: false,
            attr_in: Attributes::default(),
            attr_out: Attributes::default(),
            discarded: false,
            stack: Vec::new(),
            local: Vec::new(),
            local_bytes: LOCAL_MEMORY_BYTES,
            pc: 0,
            steps: 0,
            pending: Vec::new(),
            shuffle: None,
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
        self.shuffle = None;
    }

    /// Give this invocation `bytes` of `l[]`, as a launch's QMD asks for.
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
        // Cheaper than not writing it: a store that is always taken beats a
        // branch that is almost never taken.
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

    /// Whether a guard or source predicate holds.
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

    /// Execute `program` from its entry point until it exits.
    ///
    /// A `bar` has no meaning outside a CTA, so one reached here is an error
    /// rather than a suspension — see [`Invocation::resume`], which is what a
    /// compute dispatch drives instead.
    pub fn execute(&mut self, program: &Compiled, env: &Env) -> Result<()> {
        self.begin();
        match self.resume(program, env)? {
            Halt::Exited => Ok(()),
            Halt::Barrier => Err(Error::Gpu(format!(
                "shader: bar at {:#x} outside a compute dispatch, where there is no CTA to \
                 synchronise with",
                program.offset(self.pc.saturating_sub(1))
            ))),
            Halt::Shuffle => Err(Error::Gpu(format!(
                "shader: shfl at {:#x} reads a register of another lane, and this invocation \
                 is running on its own",
                program.offset(self.pc.saturating_sub(1))
            ))),
        }
    }

    /// Put execution back at the entry point, leaving the register file alone.
    pub fn begin(&mut self) {
        self.pc = 0;
        self.steps = 0;
        self.pending.clear();
        self.shuffle = None;
    }

    /// Run until the program exits or reaches a barrier, continuing from
    /// wherever the last call stopped.
    pub fn resume(&mut self, program: &Compiled, env: &Env) -> Result<Halt> {
        if program.is_empty() {
            return Err(Error::Gpu("shader: executing an empty program".into()));
        }
        // Moved out for the duration so the loop can hold `&mut self`; put
        // back before every return, since a barrier may suspend with texture
        // results still in flight.
        let mut pending = std::mem::take(&mut self.pending);
        let out = self.run(program, env, &mut pending);
        self.pending = pending;
        // The boundary of the boxed error: everything below runs per
        // instruction and pays for `Error`'s size, everything above does not.
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
        // Sliced to one length so the bounds check on each is the same check
        // the loop already makes: the two indexed reads per instruction were
        // 7% of a Home Menu frame as `Vec` accesses through `&self`.
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
            // Guarded: a texture result is pending for a handful of steps out
            // of a program, and `Vec::retain` is a real call even over an
            // empty vector — one per instruction, per covered pixel.
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

            // The guard first, out of its own dense array: an instruction
            // whose predicate is false never touches the 32-byte operation.
            if !self.holds(preds[pc]) {
                pc += 1;
                continue;
            }
            let op = ops[pc];

            // Anything that moves the pc other than by one flushes the
            // deferred texture writes first: their landing place was found by
            // scanning forward in program order, which a jump invalidates.
            //
            // The target is already an index: resolving it used to mean a
            // binary search over the program's byte offsets on every taken
            // branch, which the lowering does once instead.
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
                // The pc moves past the barrier before suspending, so the
                // resume lands after it rather than on it.
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
                // Like a barrier, and for the same reason: the pc moves
                // past it before suspending, so the resume lands after the
                // shuffle rather than on it. The deferred texture writes stay
                // deferred — nothing has jumped, so where they land is still
                // the place the lowering found.
                Op::Shfl {
                    dst,
                    pred,
                    src,
                    index,
                    mask,
                    mode,
                } => {
                    self.shuffle = Some(Shuffle {
                        mode,
                        dst,
                        pred,
                        src,
                        index: self.operand(index, env)?,
                        mask: self.operand(mask, env)?,
                    });
                    self.pc = pc + 1;
                    return Ok(Halt::Shuffle);
                }
                Op::Nop | Op::Inert => {}
                Op::Bra { .. } => {
                    pc = jump(program.target(pc), pending, self)?;
                    continue;
                }
                // The one branch whose target is not known until it runs: it
                // is a register value, so it still costs a lookup.
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
                other => self.run_alu(other, env)?,
            }
            pc += 1;
        }
    }

    /// Everything that isn't control flow or a texture fetch.
    fn run_alu(&mut self, op: Op, env: &Env) -> ShaderResult<()> {
        match op {
            // ---- attribute space ----
            Op::Ld {
                dst,
                offset,
                idx,
                size,
            } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.attr_in.get(base + i as u16 * 4);
                    self.set_reg_f32(dst.wrapping_add(i), v);
                }
            }
            Op::St {
                offset,
                idx,
                src,
                size,
            } => {
                let base = offset.wrapping_add(self.attr_index(idx));
                for i in 0..size.regs() {
                    let v = self.reg_f32(src.wrapping_add(i));
                    self.attr_out.set(base + i as u16 * 4, v);
                }
            }
            // `centroid` is not read here, and that is exact rather than a
            // gap: this rasterizer shades one invocation per *sample*, at
            // that sample's own centre, and an invocation only runs where
            // its sample is covered. The centroid of the area this
            // invocation covers is therefore where it is already sampling.
            Op::Ipa {
                dst,
                offset,
                mul,
                perspective,
                sat,
                centroid: _,
            } => {
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
            Op::Fadd {
                dst,
                a,
                am,
                b,
                bm,
                ftz,
                sat,
            } => {
                let x = am.apply(flush(self.reg_f32(a), ftz));
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                self.set_reg_f32(dst, saturate(x + y, sat));
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
                // The pre-scale multiplies the *first* operand, before the
                // multiply proper — a constant halving or doubling folded into
                // a multiply the shader was doing anyway.
                let x = flush(self.reg_f32(a), ftz) * scale.factor();
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                self.set_reg_f32(dst, saturate(x * y, sat));
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
                let x = flush(self.reg_f32(a), ftz);
                let y = neg_if(flush(self.operand_f32(b, env)?, ftz), bneg);
                let z = neg_if(flush(self.operand_f32(c, env)?, ftz), cneg);
                self.set_reg_f32(dst, saturate(x.mul_add(y, z), sat));
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
                let x = am.apply(flush(self.reg_f32(a), ftz));
                let y = bm.apply(flush(self.operand_f32(b, env)?, ftz));
                // The predicate selects which end: true picks the minimum,
                // which is why `fmnmx ... !pt` is the compiler's `max`.
                let v = if self.holds(pred) { x.min(y) } else { x.max(y) };
                self.set_reg_f32(dst, v);
            }
            Op::Mufu {
                dst,
                src,
                sm,
                op,
                sat,
            } => {
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
            // The signs are a property of *which pixel of the quad this
            // is*: two lanes of a derivative subtract and the other two add,
            // which is how one instruction serves both halves of a
            // difference without a branch.
            Op::Fswzadd {
                dst,
                a,
                b,
                swizzle,
                ftz,
            } => {
                let code = (swizzle >> ((env.special.lane & 3) * 2)) & 3;
                let x = flush(self.reg_f32(a), ftz);
                let y = flush(self.reg_f32(b), ftz);
                let (ka, kb) = FSWZ_SIGNS[code as usize];
                self.set_reg_f32(dst, ka * x + kb * y);
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
                let x = am.apply(self.reg_f32(a));
                let y = bm.apply(self.operand_f32(b, env)?);
                let r = float_compare(cmp, x, y);
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
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
                let x = am.apply(self.reg_f32(a));
                let y = bm.apply(self.operand_f32(b, env)?);
                let r = combine(bop, float_compare(cmp, x, y), self.holds(src));
                self.set_reg(dst, set_result(r, bf));
            }

            // ---- half-precision ----
            // Both lanes are computed in f32 and rounded once, at the merge —
            // which is what a half instruction does: it rounds its result,
            // not its arithmetic. So [`HMerge::F32`], whose result is a float,
            // rounds nowhere, even where one of its sources was a half.
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
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let lanes = [saturate(x[0] + y[0], sat), saturate(x[1] + y[1], sat)];
                let v = half_pack(self.reg(dst), lanes, merge);
                self.set_reg(dst, v);
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
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let mut lanes = [x[0] * y[0], x[1] * y[1]];
                for lane in 0..2 {
                    if fmz_zeroes(prec, sat, x[lane], y[lane]) {
                        lanes[lane] = 0.0;
                    }
                    lanes[lane] = saturate(lanes[lane], sat);
                }
                let v = half_pack(self.reg(dst), lanes, merge);
                self.set_reg(dst, v);
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
                let x = half_source(self.reg(a), FMod::NONE, asw, ftz);
                let y = half_source(
                    self.operand(b, env)?,
                    FMod {
                        neg: bneg,
                        abs: false,
                    },
                    bsw,
                    ftz,
                );
                let z = half_source(
                    self.operand(c, env)?,
                    FMod {
                        neg: cneg,
                        abs: false,
                    },
                    csw,
                    ftz,
                );
                let mut lanes = [x[0].mul_add(y[0], z[0]), x[1].mul_add(y[1], z[1])];
                for lane in 0..2 {
                    // A zeroed product leaves the addend, not zero.
                    if fmz_zeroes(prec, sat, x[lane], y[lane]) {
                        lanes[lane] = z[lane];
                    }
                    lanes[lane] = saturate(lanes[lane], sat);
                }
                let v = half_pack(self.reg(dst), lanes, merge);
                self.set_reg(dst, v);
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
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let s = self.holds(src);
                // Each lane's answer fills its own half of the register:
                // 1.0h with `.bf`, all ones without.
                let taken = if bf { 0x3C00u32 } else { 0xFFFF };
                let mut out = 0u32;
                if combine(bop, float_compare(cmp, x[0], y[0]), s) {
                    out |= taken;
                }
                if combine(bop, float_compare(cmp, x[1], y[1]), s) {
                    out |= taken << 16;
                }
                self.set_reg(dst, out);
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
                let x = half_source(self.reg(a), am, asw, ftz);
                let y = half_source(self.operand(b, env)?, bm, bsw, ftz);
                let s = self.holds(src);
                let low = combine(bop, float_compare(cmp, x[0], y[0]), s);
                let high = combine(bop, float_compare(cmp, x[1], y[1]), s);
                if and {
                    self.set_pred(p0, low && high);
                    self.set_pred(p1, !(low && high));
                } else {
                    self.set_pred(p0, low);
                    self.set_pred(p1, high);
                }
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
                let x = ineg_if(self.reg(a), aneg);
                let y = ineg_if(self.operand(b, env)?, bneg);
                // Widened, so the carry out is the bit that falls off the top.
                // A negated operand is already two's-complement here, so a
                // subtraction carries exactly when it does not borrow — which
                // is the convention `iadd.x` expects on the high half.
                let sum = u64::from(x) + u64::from(y) + u64::from(cin && self.carry);
                self.set_reg(dst, sum as u32);
                if cout {
                    self.carry = sum > u64::from(u32::MAX);
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
                let x = ineg_if(self.reg(a), aneg);
                let y = ineg_if(self.operand(b, env)?, bneg);
                let z = ineg_if(self.operand(c, env)?, cneg);
                self.set_reg(dst, x.wrapping_add(y).wrapping_add(z));
            }
            Op::Iscadd {
                dst,
                a,
                aneg,
                b,
                bneg,
                shift,
            } => {
                let x = ineg_if(self.reg(a), aneg).wrapping_shl(shift as u32);
                let y = ineg_if(self.operand(b, env)?, bneg);
                self.set_reg(dst, x.wrapping_add(y));
            }
            Op::Imnmx {
                dst,
                a,
                b,
                pred,
                signed,
            } => {
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
            Op::Imul {
                dst,
                a,
                b,
                signed,
                hi,
            } => {
                let x = self.reg(a);
                let y = self.operand(b, env)?;
                let full = if signed {
                    ((x as i32 as i64) * (y as i32 as i64)) as u64
                } else {
                    (x as u64) * (y as u64)
                };
                self.set_reg(dst, if hi { (full >> 32) as u32 } else { full as u32 });
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
                let raw_b = self.operand(b, env)?;
                let av = half(self.reg(a), ah, asigned);
                let bv = half(raw_b, bh, bsigned);
                let mut product = (av.wrapping_mul(bv)) as u32;
                if psl {
                    product <<= 16;
                }
                let raw_c = self.operand(c, env)?;
                let cv = match cmode {
                    XmadC::Full => raw_c,
                    XmadC::Lo => raw_c & 0xffff,
                    XmadC::Hi => raw_c >> 16,
                    XmadC::Bcc => (raw_b << 16).wrapping_add(raw_c),
                };
                let mut v = product.wrapping_add(cv);
                if mrg {
                    // `.mrg` replaces the result's high half with `b`'s low
                    // one rather than adding anything there.
                    v = (v & 0xffff) | (raw_b << 16);
                }
                self.set_reg(dst, v);
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
                let r = int_compare(cmp, self.reg(a), self.operand(b, env)?, signed);
                let s = self.holds(src);
                self.set_pred(p0, combine(bop, r, s));
                self.set_pred(p1, combine(bop, !r, s));
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
                let r = int_compare(cmp, self.reg(a), self.operand(b, env)?, signed);
                let r = combine(bop, r, self.holds(src));
                self.set_reg(dst, set_result(r, bf));
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
                let taken = int_compare(cmp, self.reg(c), 0, signed);
                let v = if taken {
                    self.reg(a)
                } else {
                    self.operand(b, env)?
                };
                self.set_reg(dst, v);
            }
            Op::Bfi {
                dst,
                insert,
                src,
                base,
            } => {
                let src = self.operand(src, env)?;
                let base = self.operand(base, env)?;
                let offset = src & 0xff;
                let count = (src >> 8) & 0xff;
                // Hardware's edge cases, not tidiness: an offset past the word
                // leaves the base alone, and a width that would run off the end
                // is clamped to what is left rather than wrapping.
                let v = if offset >= 32 {
                    base
                } else {
                    let count = count.min(32 - offset);
                    let mask = if count >= 32 {
                        !0
                    } else {
                        ((1u32 << count) - 1) << offset
                    };
                    (base & !mask) | ((self.reg(insert) << offset) & mask)
                };
                self.set_reg(dst, v);
            }
            Op::R2p { src, mask, byte } => {
                let bits = self.reg(src) >> (u32::from(byte) * 8);
                let mask = self.operand(mask, env)?;
                for index in 0..7u8 {
                    if mask & (1 << index) != 0 {
                        self.set_pred(index, bits & (1 << index) != 0);
                    }
                }
            }
            Op::Lop {
                dst,
                a,
                ainv,
                b,
                binv,
                op,
                pred,
            } => {
                let x = inv_if(self.reg(a), ainv);
                let y = inv_if(self.operand(b, env)?, binv);
                let v = match op {
                    LogicOp::And => x & y,
                    LogicOp::Or => x | y,
                    LogicOp::Xor => x ^ y,
                    LogicOp::PassB => y,
                };
                self.set_reg(dst, v);
                if let Some((p, test)) = pred {
                    let bit = match test {
                        LopTest::True => true,
                        LopTest::Zero => v == 0,
                        LopTest::NonZero => v != 0,
                    };
                    self.set_pred(p, bit);
                }
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
            Op::Shr {
                dst,
                a,
                b,
                signed,
                wrap,
            } => {
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
            Op::Shf {
                dst,
                lo,
                shift,
                hi,
                left,
                wrap,
                hi_out,
            } => {
                let n = self.operand(shift, env)?;
                let n = if wrap { n & 63 } else { n };
                let pair = ((self.reg(hi) as u64) << 32) | self.reg(lo) as u64;
                let shifted = if left {
                    pair.wrapping_shl(n)
                } else {
                    pair.wrapping_shr(n)
                };
                self.set_reg(
                    dst,
                    if hi_out {
                        (shifted >> 32) as u32
                    } else {
                        shifted as u32
                    },
                );
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
            Op::Flo {
                dst,
                b,
                signed,
                shift,
                inv,
            } => {
                let v = inv_if(self.operand(b, env)?, inv);
                // The highest set bit, counting from bit 0; for a signed
                // search the sign bits at the top don't count.
                let v = if signed && (v as i32) < 0 { !v } else { v };
                let idx = if v == 0 {
                    0xffff_ffff
                } else {
                    31 - v.leading_zeros()
                };
                self.set_reg(dst, if shift && v != 0 { 31 - idx } else { idx });
            }
            Op::Sel { dst, a, b, pred } => {
                let v = if self.holds(pred) {
                    self.reg(a)
                } else {
                    self.operand(b, env)?
                };
                self.set_reg(dst, v);
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
                let raw = self.operand(src, env)?;
                let raw = raw >> (sel as u32 * 8);
                let v = if src_signed {
                    sign_extend(raw, src_bytes) as i32 as f32
                } else {
                    truncate(raw, src_bytes) as f32
                };
                self.set_reg_f32(dst, sm.apply(v));
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
                let x = sm.apply(flush(self.operand_f32(src, env)?, ftz));
                let r = apply_round(x, round);
                let v = if dst_signed {
                    let lo = -(2f64.powi(dst_bytes as i32 * 8 - 1)) as f32;
                    let hi = (2f64.powi(dst_bytes as i32 * 8 - 1) - 1.0) as f32;
                    if r.is_nan() {
                        0
                    } else {
                        r.clamp(lo, hi) as i32 as u32
                    }
                } else {
                    let hi = (2f64.powi(dst_bytes as i32 * 8) - 1.0) as f32;
                    if r.is_nan() {
                        0
                    } else {
                        r.clamp(0.0, hi) as u32
                    }
                };
                self.set_reg(dst, v);
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
                    let raw = self.operand(src, env)?;
                    f16_to_f32((raw >> if hi { 16 } else { 0 }) as u16)
                } else {
                    self.operand_f32(src, env)?
                };
                let x = sm.apply(flush(x, ftz));
                let x = match round {
                    Some(round) => apply_round(x, round),
                    None => x,
                };
                let x = saturate(x, sat);
                if dst_bits == 16 {
                    // The half lands in the low half and the rest is cleared,
                    // which is `PackFloat2x16` against a zero.
                    self.set_reg(dst, u32::from(f32_to_f16(x)));
                } else {
                    self.set_reg_f32(dst, x);
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
                if SpecialRegs::PACKED.contains(&sr) {
                    return Err(fault(format!(
                        "shader: s2r of the packed special register {sr:#x}, whose field \
                         layout is not confirmed"
                    )));
                }
                // A register this doesn't model still reads zero, which is
                // what every one of them read before compute gave the thread
                // and CTA registers meaning.
                self.set_reg(dst, env.special.read(sr).unwrap_or(0));
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
                let first = combine(op1, self.holds(a), self.holds(b));
                let r = combine(op2, first, self.holds(c));
                self.set_pred(p0, r);
                self.set_pred(p1, !r);
            }

            // ---- memory ----
            Op::Ldc {
                dst,
                bank,
                offset,
                idx,
                size,
            } => {
                let base = offset.wrapping_add(self.reg(idx) as i32);
                for i in 0..size.regs() {
                    let at = base.wrapping_add(i as i32 * 4);
                    let v = env.consts.read_const(bank, at as u16)?;
                    self.set_reg(dst.wrapping_add(i), v);
                }
            }
            Op::Ldg {
                dst,
                addr,
                offset,
                size,
            } => {
                let mem = env
                    .memory
                    .ok_or_else(|| Error::Gpu("shader: ldg with no global memory bound".into()))?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                if let Some(raw) = narrow_load(size, |i| mem.read_u8(base + i as u64))? {
                    self.set_reg(dst, raw);
                } else {
                    for i in 0..size.regs() {
                        let v = mem.read_u32(base.wrapping_add(u64::from(i) * 4))?;
                        self.set_reg(dst.wrapping_add(i), v);
                    }
                }
            }
            Op::Stg {
                addr,
                offset,
                src,
                size,
            } => {
                let mem = env
                    .memory
                    .ok_or_else(|| Error::Gpu("shader: stg with no global memory bound".into()))?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                let (bytes, len) = self.store_value(src, size);
                if len < 4 {
                    for (i, byte) in bytes[..len].iter().enumerate() {
                        mem.write_u8(base + i as u64, *byte)?;
                    }
                } else {
                    for i in 0..size.regs() {
                        let v = self.reg(src.wrapping_add(i));
                        mem.write_u32(base.wrapping_add(u64::from(i) * 4), v)?;
                    }
                }
            }
            Op::Ldl {
                dst,
                addr,
                offset,
                size,
            } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let values = read_scratch(&self.local, base, size);
                for i in 0..size.regs() {
                    self.set_reg(dst.wrapping_add(i), values[i as usize]);
                }
            }
            Op::Stl {
                addr,
                offset,
                src,
                size,
            } => {
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let (bytes, len) = self.store_value(src, size);
                let cap = self.local_bytes;
                write_scratch(&mut self.local, cap, base, &bytes[..len]);
            }
            Op::Lds {
                dst,
                addr,
                offset,
                size,
            } => {
                let shared = env
                    .shared
                    .ok_or_else(|| Error::Gpu("shader: lds with no shared memory bound".into()))?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let values = read_scratch(&shared.borrow(), base, size);
                for i in 0..size.regs() {
                    self.set_reg(dst.wrapping_add(i), values[i as usize]);
                }
            }
            Op::Sts {
                addr,
                offset,
                src,
                size,
            } => {
                let shared = env
                    .shared
                    .ok_or_else(|| Error::Gpu("shader: sts with no shared memory bound".into()))?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let (bytes, len) = self.store_value(src, size);
                let mut block = shared.borrow_mut();
                let cap = block.len();
                write_scratch(&mut block, cap, base, &bytes[..len]);
            }
            Op::Atom {
                dst,
                addr,
                offset,
                src,
                op,
                ty,
                space,
            } => {
                self.run_atom(dst, addr, offset, src, op, ty, space, env)?;
            }

            Op::Unimplemented { raw } => {
                return Err(fault(format!(
                    "shader: unimplemented instruction {raw:#018x}"
                )))
            }
            // Handled by `execute`.
            Op::Exit
            | Op::Kil
            | Op::Nop
            | Op::Inert
            | Op::Bra { .. }
            | Op::Brx { .. }
            | Op::Ssy { .. }
            | Op::Pbk { .. }
            | Op::Pcnt { .. }
            | Op::Sync
            | Op::Brk
            | Op::Cont
            | Op::Bar { .. }
            | Op::Shfl { .. }
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

    /// A value `width` bytes wide, from `r` and the register after it.
    fn reg_wide(&self, r: u8, width: usize) -> u64 {
        if width == 8 {
            self.reg64(r)
        } else {
            u64::from(self.reg(r))
        }
    }

    fn set_reg_wide(&mut self, r: u8, width: usize, value: u64) {
        self.set_reg(r, value as u32);
        if width == 8 {
            self.set_reg(r.wrapping_add(1), (value >> 32) as u32);
        }
    }

    /// The bytes a store of `size` moves, taken from `src` upwards.
    fn store_value(&self, src: u8, size: MemSize) -> ([u8; 16], usize) {
        let mut out = [0u8; 16];
        for i in 0..size.regs() as usize {
            let word = self.reg(src.wrapping_add(i as u8)).to_le_bytes();
            out[i * 4..i * 4 + 4].copy_from_slice(&word);
        }
        (out, size.bytes() as usize)
    }

    /// `atom`/`atoms`/`red`: read one location, combine, write back, and hand
    /// the *old* value to `dst`. Nothing here runs two threads at once, so
    /// the read-modify-write is atomic by construction.
    #[allow(clippy::too_many_arguments)]
    fn run_atom(
        &mut self,
        dst: u8,
        addr: u8,
        offset: i32,
        src: u8,
        op: AtomOp,
        ty: AtomType,
        space: AtomSpace,
        env: &Env,
    ) -> ShaderResult<()> {
        let width = match ty {
            AtomType::U128 => {
                return Err(fault("shader: a 128-bit atomic is not implemented".into()))
            }
            AtomType::U64 | AtomType::S64 => 8usize,
            _ => 4usize,
        };
        // `cas` compares against `src` and stores the register after it;
        // every other operation takes the one operand.
        let operand = self.reg_wide(src, width);
        let stored = self.reg_wide(src.wrapping_add((width / 4) as u8), width);

        let old = match space {
            AtomSpace::Shared => {
                let shared = env.shared.ok_or_else(|| {
                    Error::Gpu("shader: a shared atomic with no shared memory bound".into())
                })?;
                let base = (self.reg(addr) as i64).wrapping_add(offset as i64) as usize;
                let block = shared.borrow();
                let words = read_scratch(&block, base, wide_size(width));
                drop(block);
                let old = pack(&words, width);
                let new = atom_apply(op, ty, old, operand, stored)?;
                let mut block = shared.borrow_mut();
                let cap = block.len();
                write_scratch(&mut block, cap, base, &unpack(new, width)[..width]);
                old
            }
            AtomSpace::Global => {
                let mem = env.memory.ok_or_else(|| {
                    Error::Gpu("shader: a global atomic with no global memory bound".into())
                })?;
                let base = (self.reg64(addr) as i64).wrapping_add(offset as i64) as u64;
                let mut old = u64::from(mem.read_u32(base)?);
                if width == 8 {
                    old |= u64::from(mem.read_u32(base + 4)?) << 32;
                }
                let new = atom_apply(op, ty, old, operand, stored)?;
                mem.write_u32(base, new as u32)?;
                if width == 8 {
                    mem.write_u32(base + 4, (new >> 32) as u32)?;
                }
                old
            }
        };
        self.set_reg_wide(dst, width, old);
        Ok(())
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
        program: &Compiled,
        pc: usize,
        op: Op,
        env: &Env,
        pending: &mut Vec<(usize, u8, u32)>,
    ) -> ShaderResult<()> {
        let Op::Texs {
            coords,
            dref,
            handle,
            dim,
            ..
        } = op
        else {
            unreachable!("run_texs called with {op:?}");
        };
        // The bindless handle lives in the driver's reserved constant bank,
        // indexed by the shader's own immediate — see `gpu::texture`'s
        // module docs and `texture::handle_offset`.
        let handle = env
            .consts
            .read_const(env.tex_cb_index, crate::gpu::texture::handle_offset(handle))?;
        let u = self.reg_f32(coords[0]);
        let v = self.reg_f32(coords[1]);
        // An array's layer is an integer in the low half of its register, not
        // a float like the coordinates beside it.
        let layer = match dim {
            TexDim::T2dArray => self.reg(coords[2]) & 0xffff,
            _ => 0,
        };
        let color = match (dref, dim) {
            (Some(reg), _) => {
                env.textures
                    .sample_compare(handle, u, v, layer, self.reg_f32(reg))?
            }
            // A 3D image's third coordinate is normalized like the other two,
            // where an array's is the layer number.
            (None, TexDim::T3d) => env
                .textures
                .sample_3d(handle, u, v, self.reg_f32(coords[2]))?,
            // A cubemap's three are a direction, and the face comes out of it.
            (None, TexDim::TCube) => {
                env.textures
                    .sample_cube(handle, u, v, self.reg_f32(coords[2]))?
            }
            (None, _) => env.textures.sample(handle, u, v, layer)?,
        };

        // Where each channel lands was worked out at decode time.
        for &(reg, store, due) in program.texs_writes(pc) {
            let raw = match store {
                isa::TexsStore::Float(channel) => color[channel].to_bits(),
                // Low half first. An odd channel count pads with zero, which
                // is what hardware leaves in the unused half.
                isa::TexsStore::Halves(low, high) => {
                    let pack = |c: Option<usize>| {
                        u32::from(f32_to_f16(c.map_or(0.0, |channel| color[channel])))
                    };
                    pack(Some(low)) | pack(high) << 16
                }
            };
            pending.retain(|&(_, r, _)| r != reg);
            pending.push((due, reg, raw));
        }
        Ok(())
    }
}

/// Work out, for every `texs` in `insns`, where each of its results lands.
/// Called once per decode, and once per lowering; see
/// [`super::Program::texs_writes`] for why it is not done per invocation.
pub(super) fn texs_writes_for(ops: &[Op]) -> Vec<super::TexsWrites> {
    let mut out = Vec::new();
    for (pc, op) in ops.iter().enumerate() {
        let Op::Texs {
            dst,
            dst2,
            mask,
            f16,
            ..
        } = *op
        else {
            continue;
        };
        let writes = isa::texs_destinations(dst, dst2, mask, f16)
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

/// Where `reg`'s pending write should land: right before the first later
/// instruction that reads it (the real dependency point), or dropped
/// entirely if something overwrites it first. A program that never touches
/// it again lands it before the last instruction, so a shader that hands a
/// `texs` result straight to its output register still ends with the value
/// hardware would eventually have written.
fn first_use_after(ops: &[Op], start: usize, reg: u8) -> Option<usize> {
    for (idx, op) in ops.iter().enumerate().skip(start) {
        if reads(op).contains(&reg) {
            return Some(idx);
        }
        if writes(op).contains(&reg) {
            return None;
        }
    }
    ops.len().checked_sub(1)
}

fn operand_reg(op: Operand) -> Option<u8> {
    match op {
        Operand::Reg(r) if r != RZ => Some(r),
        _ => None,
    }
}

/// The destination register, where a half op's merge mode keeps half of what
/// is already in it and so reads it back.
fn half_merge_reads(dst: u8, merge: HMerge) -> Option<u8> {
    match merge {
        HMerge::MrgH0 | HMerge::MrgH1 if dst != RZ => Some(dst),
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
        // A merging half op keeps the half of its destination it does not
        // write, which makes the destination a source as well.
        Op::Hfma2 {
            dst,
            a,
            b,
            c,
            merge,
            ..
        } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(operand_reg(c));
            v.extend(half_merge_reads(dst, merge));
            v
        }
        Op::Hadd2 {
            dst, a, b, merge, ..
        }
        | Op::Hmul2 {
            dst, a, b, merge, ..
        } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
            v.extend(half_merge_reads(dst, merge));
            v
        }
        Op::Hset2 { a, b, .. } | Op::Hsetp2 { a, b, .. } => {
            let mut v = vec![a];
            v.extend(operand_reg(b));
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
        Op::Stg {
            addr, src, size, ..
        }
        | Op::Stl {
            addr, src, size, ..
        } => {
            let mut v = vec![addr, addr.wrapping_add(1)];
            v.extend((0..size.regs()).map(|i| src.wrapping_add(i)));
            v
        }
        Op::Texs { coords, .. } => coords.to_vec(),
        Op::Shfl {
            src, index, mask, ..
        } => {
            let mut v = vec![src];
            v.extend(operand_reg(index));
            v.extend(operand_reg(mask));
            v
        }
        Op::Fswzadd { a, b, .. } => vec![a, b],
        _ => Vec::new(),
    };
    out.retain(|&r| r != RZ);
    out
}

/// Registers `op` writes as a destination.
pub(super) fn writes(op: &Op) -> Vec<u8> {
    match *op {
        Op::Ld { dst, size, .. }
        | Op::Ldg { dst, size, .. }
        | Op::Ldl { dst, size, .. }
        | Op::Ldc { dst, size, .. } => (0..size.regs()).map(|i| dst.wrapping_add(i)).collect(),
        Op::Ipa { dst, .. }
        | Op::Mufu { dst, .. }
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
        _ => Vec::new(),
    }
}

/// `fswzadd`'s two multipliers per two-bit swizzle code — the constant
/// tables Eden's GLSL backend emits as `FSWZ_A`/`FSWZ_B`
/// (`glsl_emit_context.cpp`).
const FSWZ_SIGNS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (0.0, -1.0)];

/// How many lanes a warp shuffle is bounded by on hardware. A fragment quad
/// is four of them and a CTA's threads are grouped into warps of this many;
/// either way the clamp and segment mask a `shfl` carries are written against
/// this width, so the arithmetic below has to be done in it.
pub const WARP_LANES: usize = 32;

/// Complete every shuffle the lanes of one warp are suspended on.
///
/// `warp` is the warp in lane order, so an invocation's index is its lane.
/// Every source register is read before any destination is written: a shuffle
/// is an exchange between lanes of one instruction, so no lane may see
/// another's result.
///
/// A lane the clamp allows but this warp does not hold — a quad is four lanes
/// of a hardware warp's thirty-two — reads as the requesting lane's own
/// value, which is what an inactive lane gives on hardware. The predicate
/// still reports what the clamp said, since that is a property of the lane
/// numbers rather than of who is running.
pub fn resolve_shuffles(warp: &mut [Invocation]) {
    let requests: Vec<Option<Shuffle>> = warp
        .iter_mut()
        .map(|invocation| invocation.shuffle.take())
        .collect();
    let sources: Vec<Option<(u8, u8, u32, bool)>> = requests
        .iter()
        .enumerate()
        .map(|(lane, request)| {
            let &Some(shuffle) = request else { return None };
            let (from, in_bounds) = shuffle_source(shuffle, lane as u32);
            let value = match warp.get(from as usize).filter(|_| in_bounds) {
                Some(peer) => peer.reg(shuffle.src),
                None => warp[lane].reg(shuffle.src),
            };
            Some((shuffle.dst, shuffle.pred, value, in_bounds))
        })
        .collect();
    for (invocation, source) in warp.iter_mut().zip(sources) {
        let Some((dst, pred, value, in_bounds)) = source else {
            continue;
        };
        invocation.set_reg(dst, value);
        invocation.set_pred(pred, in_bounds);
    }
}

/// Which lane `shuffle` reads when `lane` executes it, and whether that lane
/// was within the bound its clamp and segment mask describe.
///
/// The clamp bounds the lanes this one may reach and the segment mask splits
/// the warp into independent groups; every mode composes them the same way,
/// differing only in where it looks. Signed, because a lane below the segment
/// (`shfl.up` at its bottom) must read as out of bounds rather than wrapping
/// around to the top of the warp.
fn shuffle_source(shuffle: Shuffle, lane: u32) -> (u32, bool) {
    let clamp = (shuffle.mask & 0x1f) as i32;
    let segment = ((shuffle.mask >> 8) & 0x1f) as i32;
    let lane = lane as i32;
    let index = shuffle.index as i32;
    // The bottom of this lane's segment, and how far up it may reach.
    let floor = lane & segment;
    let ceiling = floor | (clamp & !segment);
    let (from, in_bounds) = match shuffle.mode {
        ShflMode::Idx => {
            let from = (index & !segment) | floor;
            (from, from <= ceiling)
        }
        // `up` is the one mode the bound holds from below: it reads towards
        // the bottom of the segment, so it is the floor it must not cross.
        ShflMode::Up => {
            let from = lane - index;
            (from, from >= ceiling)
        }
        ShflMode::Down => {
            let from = lane + index;
            (from, from <= ceiling)
        }
        ShflMode::Bfly => {
            let from = lane ^ index;
            (from, from <= ceiling)
        }
    };
    (from.max(0) as u32, in_bounds && from >= 0)
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

/// The smallest half that is not subnormal, which is what a half
/// instruction's `.ftz` flushes towards zero.
const SMALLEST_NORMAL_HALF: f32 = 6.103_515_6e-5;

/// One source of a half-precision op: its two lanes, flushed and modified.
///
/// The modifier comes second, exactly as it does for `fadd` — `abs` of a
/// flushed subnormal is a flushed subnormal, not a subnormal made positive.
fn half_source(bits: u32, m: FMod, sw: HSwizzle, ftz: bool) -> [f32; 2] {
    let mut lanes = half_lanes(bits, sw);
    for lane in lanes.iter_mut() {
        *lane = m.apply(half_flush(*lane, sw, ftz));
    }
    lanes
}

/// A source register's two lanes, widened to f32.
fn half_lanes(bits: u32, sw: HSwizzle) -> [f32; 2] {
    let low = f16_to_f32(bits as u16);
    let high = f16_to_f32((bits >> 16) as u16);
    match sw {
        HSwizzle::H1H0 => [low, high],
        HSwizzle::H0H0 => [low, low],
        HSwizzle::H1H1 => [high, high],
        // Not a pair at all: one f32 that both lanes read.
        HSwizzle::F32 => [f32::from_bits(bits); 2],
    }
}

/// `.ftz` against whichever precision the lane actually came from: an f32
/// operand of a half instruction is still an f32, and flushing it at the
/// half threshold would swallow four orders of magnitude of real numbers.
fn half_flush(v: f32, sw: HSwizzle, ftz: bool) -> f32 {
    if !ftz {
        return v;
    }
    if sw == HSwizzle::F32 {
        return flush(v, true);
    }
    if v != 0.0 && v.abs() < SMALLEST_NORMAL_HALF {
        0.0f32.copysign(v)
    } else {
        v
    }
}

/// Write a half-precision op's two lanes into `dst`'s current value.
fn half_pack(dst: u32, lanes: [f32; 2], merge: HMerge) -> u32 {
    let half = |v: f32| u32::from(f32_to_f16(v));
    match merge {
        HMerge::H1H0 => half(lanes[0]) | (half(lanes[1]) << 16),
        HMerge::F32 => lanes[0].to_bits(),
        HMerge::MrgH0 => (dst & 0xFFFF_0000) | half(lanes[0]),
        HMerge::MrgH1 => (dst & 0x0000_FFFF) | (half(lanes[1]) << 16),
    }
}

/// Whether `.fmz` forces this lane's product to zero: D3D9's rule that
/// anything times zero is zero, NaN and infinity included. Whether the mode
/// applies at all is [`HPrecision::zeroes_products`].
fn fmz_zeroes(prec: HPrecision, sat: bool, a: f32, b: f32) -> bool {
    prec.zeroes_products(sat) && (a == 0.0 || b == 0.0)
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
/// A load narrower than a register: the raw bytes, sign-extended for the
/// signed forms. `None` means `size` moves whole registers instead.
fn narrow_load(
    size: MemSize,
    mut byte: impl FnMut(usize) -> ShaderResult<u8>,
) -> ShaderResult<Option<u32>> {
    let width = size.bytes() as usize;
    if size.regs() != 1 || width >= 4 {
        return Ok(None);
    }
    let mut raw = 0u32;
    for i in 0..width {
        raw |= u32::from(byte(i)?) << (i * 8);
    }
    let signed = matches!(size, MemSize::S8 | MemSize::S16);
    Ok(Some(if signed {
        sign_extend(raw, size.bytes() as u8)
    } else {
        raw
    }))
}

/// The registers a load of `size` from byte-addressed scratch produces. Past
/// the end reads zero, which is what an out-of-range local access gave before
/// sub-word sizes were honoured.
fn read_scratch(bytes: &[u8], base: usize, size: MemSize) -> [u32; 4] {
    let mut out = [0u32; 4];
    if let Ok(Some(raw)) = narrow_load(size, |i| Ok(bytes.get(base + i).copied().unwrap_or(0))) {
        out[0] = raw;
        return out;
    }
    for (i, word) in out.iter_mut().enumerate().take(size.regs() as usize) {
        let mut raw = [0u8; 4];
        for (j, b) in raw.iter_mut().enumerate() {
            *b = bytes.get(base + i * 4 + j).copied().unwrap_or(0);
        }
        *word = u32::from_le_bytes(raw);
    }
    out
}

/// Write `value` into byte-addressed scratch, growing it to `cap` first. A
/// store that would run past the end is dropped rather than growing it.
fn write_scratch(bytes: &mut Vec<u8>, cap: usize, base: usize, value: &[u8]) {
    if bytes.len() < cap {
        bytes.resize(cap, 0);
    }
    let end = base + value.len();
    if end <= bytes.len() {
        bytes[base..end].copy_from_slice(value);
    }
}

/// The [`MemSize`] that moves `width` bytes as whole registers.
fn wide_size(width: usize) -> MemSize {
    if width == 8 {
        MemSize::B64
    } else {
        MemSize::B32
    }
}

fn pack(words: &[u32; 4], width: usize) -> u64 {
    if width == 8 {
        u64::from(words[0]) | (u64::from(words[1]) << 32)
    } else {
        u64::from(words[0])
    }
}

fn unpack(value: u64, width: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&(value as u32).to_le_bytes());
    if width == 8 {
        out[4..].copy_from_slice(&((value >> 32) as u32).to_le_bytes());
    }
    out
}

/// What an atomic leaves in memory, given what was there.
fn atom_apply(op: AtomOp, ty: AtomType, old: u64, b: u64, stored: u64) -> ShaderResult<u64> {
    let wide = matches!(ty, AtomType::U64 | AtomType::S64);
    let trim = |v: u64| if wide { v } else { v & 0xFFFF_FFFF };
    let float = matches!(ty, AtomType::F32);
    Ok(match op {
        AtomOp::Add | AtomOp::SafeAdd if float => (f32::from_bits(old as u32)
            + f32::from_bits(b as u32))
        .to_bits()
        .into(),
        AtomOp::Add | AtomOp::SafeAdd => trim(old.wrapping_add(b)),
        AtomOp::Min => {
            if atom_less(ty, b, old) {
                b
            } else {
                old
            }
        }
        AtomOp::Max => {
            if atom_less(ty, old, b) {
                b
            } else {
                old
            }
        }
        // Wrapping counters: `inc` rolls to zero once it reaches the operand,
        // `dec` rolls back up to it. Both are unsigned however the type reads.
        AtomOp::Inc => {
            if old >= b {
                0
            } else {
                trim(old + 1)
            }
        }
        AtomOp::Dec => {
            if old == 0 || old > b {
                b
            } else {
                old - 1
            }
        }
        AtomOp::And => old & b,
        AtomOp::Or => old | b,
        AtomOp::Xor => old ^ b,
        AtomOp::Exch => b,
        AtomOp::Cas => {
            if old == b {
                stored
            } else {
                old
            }
        }
    })
}

/// `x < y` under the atomic's type, which is what `min`/`max` turn on.
fn atom_less(ty: AtomType, x: u64, y: u64) -> bool {
    match ty {
        AtomType::F32 => f32::from_bits(x as u32) < f32::from_bits(y as u32),
        AtomType::S32 => (x as u32 as i32) < (y as u32 as i32),
        AtomType::S64 => (x as i64) < (y as i64),
        _ => x < y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::shader::compiled::Compiled;
    use crate::gpu::shader::decode_program;
    use crate::gpu::shader::isa::{FMod, FmulScale, Instruction, TexDim};

    fn no_consts() -> HashMap<(u8, u16), f32> {
        HashMap::new()
    }

    /// Build a straight-line program out of unpredicated ops, at the byte
    /// offsets a real 32-byte-block layout would put them at.
    fn prog(ops: &[Op]) -> Compiled {
        let mut p = crate::gpu::shader::Program::default();
        for (i, &op) in ops.iter().enumerate() {
            p.insns.push(Instruction::always(op));
            p.offsets
                .push(crate::gpu::shader::ENTRY_OFFSET + i as u32 * 8);
        }
        Compiled::new(&p)
    }
    use std::cell::RefCell;

    /// Records the `(handle, u, v)` it was asked to sample and always
    /// returns the same colour, so a test can check both what the
    /// interpreter computed and what it fed the texture backend.
    struct RecordingTextures {
        calls: RefCell<Vec<(u32, f32, f32, u32)>>,
        color: [f32; 4],
    }

    impl TextureSource for RecordingTextures {
        fn sample(&self, handle: u32, u: f32, v: f32, layer: u32) -> ShaderResult<[f32; 4]> {
            self.calls.borrow_mut().push((handle, u, v, layer));
            Ok(self.color)
        }
    }

    /// A flat byte-addressed global address space, so the memory ops can be
    /// checked without a GPU address space under them.
    #[derive(Default)]
    struct FlatMemory {
        bytes: RefCell<Vec<u8>>,
    }

    impl FlatMemory {
        fn with(size: usize) -> FlatMemory {
            FlatMemory {
                bytes: RefCell::new(vec![0; size]),
            }
        }
    }

    impl GlobalMemory for FlatMemory {
        fn read_u32(&self, addr: u64) -> ShaderResult<u32> {
            let bytes = self.bytes.borrow();
            let at = addr as usize;
            let mut word = [0u8; 4];
            word.copy_from_slice(&bytes[at..at + 4]);
            Ok(u32::from_le_bytes(word))
        }

        fn read_u8(&self, addr: u64) -> ShaderResult<u8> {
            Ok(self.bytes.borrow()[addr as usize])
        }

        fn write_u32(&self, addr: u64, value: u32) -> ShaderResult<()> {
            let at = addr as usize;
            self.bytes.borrow_mut()[at..at + 4].copy_from_slice(&value.to_le_bytes());
            Ok(())
        }

        fn write_u8(&self, addr: u64, value: u8) -> ShaderResult<()> {
            self.bytes.borrow_mut()[addr as usize] = value;
            Ok(())
        }
    }

    #[test]
    fn sr_y_direction_reads_a_sign_and_never_a_zero() {
        // A shader multiplies a screen-space direction by this. Answering the
        // zero an unmodelled special register gets does not flip anything --
        // it deletes it.
        let up = SpecialRegs {
            y_negate: false,
            ..SpecialRegs::default()
        };
        assert_eq!(f32::from_bits(up.read(0x12).unwrap()), 1.0);
        let down = SpecialRegs {
            y_negate: true,
            ..SpecialRegs::default()
        };
        assert_eq!(f32::from_bits(down.read(0x12).unwrap()), -1.0);
    }

    #[test]
    fn s2r_answers_the_thread_and_cta_registers_a_dispatch_set() {
        let consts = no_consts();
        let mut env = Env::new(&consts, &NoTextures);
        env.special.tid = [3, 4, 5];
        env.special.ctaid = [6, 7, 8];
        env.special.shared_size = 0x400;
        let mut inv = Invocation::new();
        inv.execute(
            &prog(&[
                Op::S2r { dst: 0, sr: 0x21 },
                Op::S2r { dst: 1, sr: 0x23 },
                Op::S2r { dst: 2, sr: 0x25 },
                Op::S2r { dst: 3, sr: 0x27 },
                Op::S2r { dst: 4, sr: 0x32 },
                // Not modelled, and still zero rather than an error: that is
                // what every one of these read before compute existed.
                Op::S2r { dst: 5, sr: 0x1d },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap();
        assert_eq!([inv.reg(0), inv.reg(1)], [3, 5]);
        assert_eq!([inv.reg(2), inv.reg(3)], [6, 8]);
        assert_eq!(inv.reg(4), 0x400);
        assert_eq!(inv.reg(5), 0);
    }

    #[test]
    fn the_packed_thread_register_is_refused_rather_than_guessed_at() {
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let err = Invocation::new()
            .execute(&prog(&[Op::S2r { dst: 0, sr: 0x20 }, Op::Exit]), &env)
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("packed special register"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_global_store_narrower_than_a_word_touches_only_its_own_bytes() {
        // The byte and halfword forms used to load a whole word, and had no
        // store at all. A kernel writing a byte array would have scribbled
        // over three of its neighbours.
        let memory = FlatMemory::with(16);
        memory.write_u32(0, 0xAABB_CCDD).unwrap();
        let consts = no_consts();
        let mut env = Env::new(&consts, &NoTextures);
        env.memory = Some(&memory);

        let mut inv = Invocation::new();
        inv.set_reg(4, 1);
        inv.set_reg(5, 0);
        inv.set_reg(6, 0x77);
        inv.execute(
            &prog(&[
                Op::Stg {
                    addr: 4,
                    offset: 0,
                    src: 6,
                    size: MemSize::U8,
                },
                Op::Ldg {
                    dst: 0,
                    addr: 4,
                    offset: 0,
                    size: MemSize::U8,
                },
                Op::Ldg {
                    dst: 1,
                    addr: 4,
                    offset: 0,
                    size: MemSize::S8,
                },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap();
        assert_eq!(memory.read_u32(0).unwrap(), 0xAABB_77DD);
        assert_eq!(inv.reg(0), 0x77);
        assert_eq!(inv.reg(1), 0x77);
    }

    #[test]
    fn a_signed_narrow_load_extends_its_sign() {
        let memory = FlatMemory::with(16);
        memory.write_u32(0, 0x0000_FF80).unwrap();
        let consts = no_consts();
        let mut env = Env::new(&consts, &NoTextures);
        env.memory = Some(&memory);

        let mut inv = Invocation::new();
        inv.execute(
            &prog(&[
                Op::Ldg {
                    dst: 0,
                    addr: 4,
                    offset: 0,
                    size: MemSize::S16,
                },
                Op::Ldg {
                    dst: 1,
                    addr: 4,
                    offset: 0,
                    size: MemSize::U16,
                },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap();
        assert_eq!(inv.reg(0), 0xFFFF_FF80);
        assert_eq!(inv.reg(1), 0x0000_FF80);
    }

    #[test]
    fn a_global_atomic_returns_the_old_value_and_leaves_the_new_one() {
        let memory = FlatMemory::with(16);
        memory.write_u32(0, 10).unwrap();
        let consts = no_consts();
        let mut env = Env::new(&consts, &NoTextures);
        env.memory = Some(&memory);

        let mut inv = Invocation::new();
        inv.set_reg(6, 7);
        inv.execute(
            &prog(&[
                Op::Atom {
                    dst: 0,
                    addr: 4,
                    offset: 0,
                    src: 6,
                    op: AtomOp::Add,
                    ty: AtomType::U32,
                    space: AtomSpace::Global,
                },
                Op::Exit,
            ]),
            &env,
        )
        .unwrap();
        assert_eq!(inv.reg(0), 10, "the old value");
        assert_eq!(memory.read_u32(0).unwrap(), 17);
    }

    #[test]
    fn every_atomic_operation_computes_what_its_name_says() {
        use AtomOp::*;
        let u32s = AtomType::U32;
        assert_eq!(atom_apply(Add, u32s, 5, 3, 0).unwrap(), 8);
        assert_eq!(atom_apply(Min, u32s, 5, 3, 0).unwrap(), 3);
        assert_eq!(atom_apply(Max, u32s, 5, 3, 0).unwrap(), 5);
        assert_eq!(atom_apply(And, u32s, 0b110, 0b011, 0).unwrap(), 0b010);
        assert_eq!(atom_apply(Or, u32s, 0b110, 0b011, 0).unwrap(), 0b111);
        assert_eq!(atom_apply(Xor, u32s, 0b110, 0b011, 0).unwrap(), 0b101);
        assert_eq!(atom_apply(Exch, u32s, 5, 3, 0).unwrap(), 3);
        // A signed minimum is the whole reason the type is carried.
        let negative = (-4i32) as u32 as u64;
        assert_eq!(
            atom_apply(Min, AtomType::S32, negative, 3, 0).unwrap(),
            negative
        );
        assert_eq!(atom_apply(Min, u32s, negative, 3, 0).unwrap(), 3);
        // `inc` wraps to zero at the operand, `dec` wraps back up to it.
        assert_eq!(atom_apply(Inc, u32s, 2, 4, 0).unwrap(), 3);
        assert_eq!(atom_apply(Inc, u32s, 4, 4, 0).unwrap(), 0);
        assert_eq!(atom_apply(Dec, u32s, 0, 4, 0).unwrap(), 4);
        assert_eq!(atom_apply(Dec, u32s, 3, 4, 0).unwrap(), 2);
        // `cas` stores the register after its comparand, and only on a match.
        assert_eq!(atom_apply(Cas, u32s, 5, 5, 9).unwrap(), 9);
        assert_eq!(atom_apply(Cas, u32s, 5, 4, 9).unwrap(), 5);
        let one = 1.0f32.to_bits().into();
        let two = 2.0f32.to_bits().into();
        assert_eq!(
            atom_apply(Add, AtomType::F32, one, two, 0).unwrap(),
            3.0f32.to_bits().into()
        );
    }

    #[test]
    fn a_barrier_suspends_where_it_stands_and_resumes_after_it() {
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let program = prog(&[
            Op::Mov32i { dst: 0, imm: 1 },
            Op::Bar {
                mode: BarMode::Sync,
            },
            Op::Mov32i { dst: 1, imm: 2 },
            Op::Exit,
        ]);

        let mut inv = Invocation::new();
        inv.begin();
        assert_eq!(inv.resume(&program, &env).unwrap(), Halt::Barrier);
        assert_eq!(inv.reg(0), 1);
        assert_eq!(inv.reg(1), 0, "nothing past the barrier has run");
        assert_eq!(inv.resume(&program, &env).unwrap(), Halt::Exited);
        assert_eq!(inv.reg(1), 2);
    }

    #[test]
    fn a_barrier_in_a_draw_is_an_error_rather_than_a_silent_no_op() {
        // It used to decode to `Inert`, which is right for `membar` and wrong
        // for this: outside a CTA there is nothing to synchronise with.
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let err = Invocation::new()
            .execute(
                &prog(&[
                    Op::Bar {
                        mode: BarMode::Sync,
                    },
                    Op::Exit,
                ]),
                &env,
            )
            .unwrap_err();
        assert!(format!("{err:?}").contains("no CTA"), "got {err:?}");
    }

    /// Four lanes exchanging a register is the whole point of a quad: this
    /// is `dFdx`'s fetch, where each lane reads the value its horizontal
    /// neighbour holds at the same instruction.
    #[test]
    fn a_shuffle_suspends_until_the_rest_of_its_warp_can_answer_it() {
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let program = prog(&[
            Op::Shfl {
                dst: 1,
                pred: 0,
                src: 0,
                index: Operand::Imm(1),
                mask: Operand::Imm(0x1c),
                mode: ShflMode::Bfly,
            },
            Op::Exit,
        ]);

        // `begin` rather than `reset`: the seeded registers are the values
        // the lanes are exchanging.
        let mut warp: [Invocation; 4] = std::array::from_fn(|_| Invocation::new());
        for (lane, invocation) in warp.iter_mut().enumerate() {
            invocation.set_reg(0, 10 + lane as u32);
            invocation.begin();
        }

        for invocation in warp.iter_mut() {
            assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Shuffle);
            assert_eq!(invocation.reg(1), 0, "nothing has been exchanged yet");
        }
        resolve_shuffles(&mut warp);
        for invocation in warp.iter_mut() {
            assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Exited);
        }

        // `bfly 1` pairs the lanes whose numbers differ in the low bit.
        assert_eq!(warp.each_ref().map(|lane| lane.reg(1)), [11, 10, 13, 12]);
        assert!(
            warp.iter().all(|lane| lane.pred(0)),
            "every lane was in bounds"
        );
    }

    /// A lane the clamp puts out of reach keeps its own value, and says so in
    /// the predicate. `shfl.up` at the bottom of a segment is the case that
    /// happens: there is nothing below it to read.
    #[test]
    fn a_shuffle_that_reaches_past_its_segment_keeps_the_lane_s_own_value() {
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let program = prog(&[
            Op::Shfl {
                dst: 1,
                pred: 0,
                src: 0,
                index: Operand::Imm(1),
                mask: Operand::Imm(0),
                mode: ShflMode::Up,
            },
            Op::Exit,
        ]);

        let mut warp: [Invocation; 2] = std::array::from_fn(|_| Invocation::new());
        for (lane, invocation) in warp.iter_mut().enumerate() {
            invocation.set_reg(0, 10 + lane as u32);
            invocation.begin();
        }
        for invocation in warp.iter_mut() {
            assert_eq!(invocation.resume(&program, &env).unwrap(), Halt::Shuffle);
        }
        resolve_shuffles(&mut warp);

        assert_eq!(warp[0].reg(1), 10, "lane 0 has nothing below it to read");
        assert!(!warp[0].pred(0));
        assert_eq!(warp[1].reg(1), 10);
        assert!(warp[1].pred(0));
    }

    #[test]
    fn a_shuffle_outside_a_warp_is_an_error_rather_than_a_lane_reading_itself() {
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let program = prog(&[
            Op::Shfl {
                dst: 1,
                pred: 0,
                src: 0,
                index: Operand::Imm(1),
                mask: Operand::Imm(0x1c),
                mode: ShflMode::Bfly,
            },
            Op::Exit,
        ]);
        let err = Invocation::new().execute(&program, &env).unwrap_err();
        assert!(format!("{err:?}").contains("another lane"), "got {err:?}");
    }

    /// The other half of a derivative: which sign each lane adds with is a
    /// property of where it sits in the quad, and `sr0` is how a shader asks
    /// where that is.
    #[test]
    fn the_per_lane_add_takes_its_signs_from_the_lane_it_runs_on() {
        let consts = no_consts();
        let program = prog(&[
            Op::S2r { dst: 5, sr: 0x00 },
            // 0xe4 is the identity swizzle: lane n takes code n.
            Op::Fswzadd {
                dst: 0,
                a: 1,
                b: 2,
                swizzle: 0xe4,
                ftz: false,
            },
            Op::Exit,
        ]);

        // (-a - b), (a - b), (-a + b), (0 - b) — the four codes in order.
        for (lane, expected) in [-4.0f32, 2.0, -2.0, -1.0].into_iter().enumerate() {
            let mut env = Env::new(&consts, &NoTextures);
            env.special.lane = lane as u32;
            let mut invocation = Invocation::new();
            invocation.set_reg_f32(1, 3.0);
            invocation.set_reg_f32(2, 1.0);
            invocation.execute(&program, &env).unwrap();
            assert_eq!(invocation.reg_f32(0), expected, "lane {lane}");
            assert_eq!(invocation.reg(5), lane as u32);
        }
    }

    #[test]
    fn shared_memory_is_addressed_in_bytes_and_shared_between_invocations() {
        let consts = no_consts();
        let shared: SharedMemory = RefCell::new(vec![0u8; 64]);
        let mut env = Env::new(&consts, &NoTextures);
        env.shared = Some(&shared);

        let mut writer = Invocation::new();
        writer.set_reg(4, 8);
        writer.set_reg(5, 0xDEAD);
        writer
            .execute(
                &prog(&[
                    Op::Sts {
                        addr: 4,
                        offset: 4,
                        src: 5,
                        size: MemSize::B32,
                    },
                    Op::Exit,
                ]),
                &env,
            )
            .unwrap();

        let mut reader = Invocation::new();
        reader
            .execute(
                &prog(&[
                    Op::Lds {
                        dst: 0,
                        addr: RZ,
                        offset: 12,
                        size: MemSize::B32,
                    },
                    Op::Exit,
                ]),
                &env,
            )
            .unwrap();
        assert_eq!(reader.reg(0), 0xDEAD);
    }

    #[test]
    fn a_hand_written_alu_program_produces_the_expected_registers() {
        // r2 = r0 * r1; r3 = r2 * r1 + r0. Register-register forms only, so
        // no constant source is exercised — this is purely the interpreter's
        // execute loop, independent of the decoder and of any real shader.
        let program = prog(&[
            Op::Fmul {
                dst: 2,
                a: 0,
                b: Operand::Reg(1),
                bm: FMod::NONE,
                ftz: true,
                sat: false,
                scale: FmulScale::None,
            },
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

        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();

        assert_eq!(inv.reg_f32(2), 6.0);
        assert_eq!(inv.reg_f32(3), 20.0);
    }

    /// A program with real byte offsets, so branch targets resolve. Each
    /// entry is `(op, predicate)`; offsets follow the 32-byte block layout
    /// (slot 0 of every block is a `sched` word, so it is skipped).
    fn prog_at(entries: &[(Op, Pred)]) -> Compiled {
        let mut p = crate::gpu::shader::Program::default();
        let mut offset = crate::gpu::shader::ENTRY_OFFSET;
        for &(op, pred) in entries {
            p.insns.push(Instruction { pred, op });
            p.offsets.push(offset);
            offset = crate::gpu::shader::next_slot(offset);
        }
        Compiled::new(&p)
    }

    #[test]
    fn a_guard_predicate_skips_the_instruction() {
        // r1 = 1.0 always; r2 = 2.0 only if p0; r3 = 3.0 only if !p0.
        let program = prog_at(&[
            (
                Op::Mov32i {
                    dst: 1,
                    imm: 1.0f32.to_bits(),
                },
                Pred::ALWAYS,
            ),
            (
                Op::Mov32i {
                    dst: 2,
                    imm: 2.0f32.to_bits(),
                },
                Pred {
                    reg: 0,
                    negate: false,
                },
            ),
            (
                Op::Mov32i {
                    dst: 3,
                    imm: 3.0f32.to_bits(),
                },
                Pred {
                    reg: 0,
                    negate: true,
                },
            ),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
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
            (
                Op::Bra { target: 0x30 },
                Pred {
                    reg: 0,
                    negate: true,
                },
            ),
            (Op::Mov32i { dst: 2, imm: 10 }, Pred::ALWAYS),
            (Op::Bra { target: 0x38 }, Pred::ALWAYS), // skip the else
            (Op::Mov32i { dst: 2, imm: 20 }, Pred::ALWAYS), // else, at 0x30
            (Op::Exit, Pred::ALWAYS),                 // at 0x38
        ]);
        // Offset 0x20 is a `sched` control word, not an instruction slot.
        let offsets: Vec<u32> = (0..program.len()).map(|i| program.offset(i)).collect();
        assert_eq!(offsets, vec![0x08, 0x10, 0x18, 0x28, 0x30, 0x38]);

        let mut taken = Invocation::new();
        taken.set_reg(0, 1);
        taken.set_reg(1, 2);
        taken
            .execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
        assert_eq!(taken.reg(2), 10);

        let mut not_taken = Invocation::new();
        not_taken.set_reg(0, 5);
        not_taken.set_reg(1, 2);
        not_taken
            .execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
        assert_eq!(not_taken.reg(2), 20);
    }

    #[test]
    fn a_backward_branch_runs_a_real_loop() {
        // r1 = 0; do { r1 += 1 } while (r1 < 4)
        let program = prog_at(&[
            (Op::Mov32i { dst: 1, imm: 0 }, Pred::ALWAYS),
            // loop body, at 0x10
            (
                Op::Iadd {
                    dst: 1,
                    a: 1,
                    aneg: false,
                    b: Operand::Imm(1),
                    bneg: false,
                    cin: false,
                    cout: false,
                },
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
            (
                Op::Bra { target: 0x10 },
                Pred {
                    reg: 0,
                    negate: false,
                },
            ),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
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
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
        assert_eq!(inv.reg(1), 7);
        assert_eq!(inv.reg(2), 9);
    }

    #[test]
    fn a_program_that_never_exits_fails_instead_of_hanging() {
        let program = prog_at(&[(Op::Bra { target: 0x08 }, Pred::ALWAYS)]);
        let mut inv = Invocation::new();
        assert!(inv
            .execute(&program, &Env::new(&no_consts(), &NoTextures))
            .is_err());
    }

    #[test]
    fn kil_discards_the_fragment() {
        let program = prog_at(&[(Op::Kil, Pred::ALWAYS), (Op::Exit, Pred::ALWAYS)]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
        assert!(inv.discarded);
    }

    #[test]
    fn integer_ops_use_the_registers_as_integers_not_floats() {
        // The register file is untyped; the same bits are an address here
        // and a float three instructions later, so an integer op must not
        // round-trip through f32.
        let program = prog_at(&[
            (
                Op::Mov32i {
                    dst: 0,
                    imm: 0x1234_5678,
                },
                Pred::ALWAYS,
            ),
            (
                Op::Shr {
                    dst: 1,
                    a: 0,
                    b: Operand::Imm(16),
                    signed: false,
                    wrap: false,
                },
                Pred::ALWAYS,
            ),
            (
                Op::Lop {
                    dst: 2,
                    a: 0,
                    ainv: false,
                    b: Operand::Imm(0xffff),
                    binv: false,
                    op: LogicOp::And,
                    pred: None,
                },
                Pred::ALWAYS,
            ),
            (
                Op::Iadd {
                    dst: 3,
                    a: 1,
                    aneg: false,
                    b: Operand::Reg(2),
                    bneg: false,
                    cin: false,
                    cout: false,
                },
                Pred::ALWAYS,
            ),
            (Op::Exit, Pred::ALWAYS),
        ]);
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
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
            (
                Op::Mov32i {
                    dst: 0,
                    imm: (-2.5f32).to_bits(),
                },
                Pred::ALWAYS,
            ),
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
        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();
        assert_eq!(inv.reg(1) as i32, -2);
        assert_eq!(inv.reg(2) as i32, -3);
        assert_eq!(inv.reg_f32(3), -2.0);
    }

    #[test]
    fn rz_reads_as_zero_and_discards_writes() {
        let program = prog(&[
            Op::Fmul {
                dst: 0xff,
                a: 0,
                b: Operand::Reg(1),
                bm: FMod::NONE,
                ftz: true,
                sat: false,
                scale: FmulScale::None,
            },
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

        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();

        // dst=RZ: the write to r255 is discarded, not aliased to some slot.
        assert_eq!(inv.reg_f32(2), 0.0 * 3.0 + 7.0);
    }

    #[test]
    fn texs_resolves_its_handle_from_the_driver_constant_bank_and_writes_the_masked_channels() {
        // tex.frag's real shape, with the roles `isa`'s `decodes_texs` test
        // documents: the destinations are REG_00 and REG_28, the coordinates
        // REG_08 and REG_20.
        let program = prog(&[
            Op::Texs {
                dst: 2,
                dst2: 4,
                coords: [0, 3, RZ], // u=r0, v=r3
                dref: None,
                handle: 0x20,
                dim: TexDim::T2d,
                mask: [true, true, true, true],
                f16: false,
            },
            Op::Exit,
        ]);
        let mut inv = Invocation::new();
        inv.set_reg_f32(0, 0.25); // u
        inv.set_reg_f32(3, 0.75); // v

        let mut consts = HashMap::new();
        let handle = 7u32 | (2u32 << 20); // imageId=7, samplerId=2
                                          // The immediate 0x20 is a dword index, so the handle is 0x80 bytes in
                                          // — putting it at 0x20 instead is what made every draw in a page of
                                          // text resolve to the same texture.
        consts.insert(
            (crate::gpu::texture::NOUVEAU_TEX_CB_INDEX, 0x80),
            f32::from_bits(handle),
        );
        consts.insert(
            (crate::gpu::texture::NOUVEAU_TEX_CB_INDEX, 0x20),
            f32::from_bits(99),
        );

        let textures = RecordingTextures {
            calls: RefCell::new(Vec::new()),
            color: [0.1, 0.2, 0.3, 0.4],
        };

        inv.execute(&program, &Env::new(&consts, &textures))
            .unwrap();

        assert_eq!(
            textures.calls.borrow().as_slice(),
            &[(handle, 0.25, 0.75, 0)]
        );
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

        let program = Compiled::new(&decode_program(&bytes).unwrap());

        let mut inv = Invocation::new();
        inv.attr_in.set(0x7c, 1.0 / w);
        inv.attr_in.set(0x80, color[0] / w);
        inv.attr_in.set(0x84, color[1] / w);
        inv.attr_in.set(0x88, color[2] / w);
        inv.attr_in.set(0x8c, color[3] / w);

        inv.execute(&program, &Env::new(&no_consts(), &NoTextures))
            .unwrap();

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
        let program = Compiled::new(&decode_program(&bytes).unwrap());

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

        inv.execute(&program, &Env::new(&consts, &NoTextures))
            .unwrap();

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
            .map(
                0x5000_0000,
                0x1000,
                1,
                0,
                crate::gpu::vmm::SMALL_PAGE_SIZE,
                0,
                0,
            )
            .unwrap();
        vmm.write_u32(&mut mem, gpu_va + 0x10, 42.5f32.to_bits())
            .unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx {
            mem: &mut mem,
            vmm: &vmm,
            host1x: &mut host1x,
            stats: &mut stats,
            trace: false,
        };

        let bindings = |bank: u8| {
            if bank == 2 {
                Some((gpu_va, 0x1000))
            } else {
                None
            }
        };
        let cache = std::cell::RefCell::new(ConstCache::default());
        let source = MemoryConstants {
            ctx: &ctx,
            bindings: &bindings,
            cache: &cache,
        };

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
        let program = Compiled::new(&decode_program(&bytes).unwrap());

        struct StubTex;
        impl TextureSource for StubTex {
            fn sample(
                &self,
                _handle: u32,
                _u: f32,
                _v: f32,
                _layer: u32,
            ) -> ShaderResult<[f32; 4]> {
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
        inv.execute(&program, &Env::new(&no_consts, &StubTex))
            .unwrap();

        assert_eq!(inv.reg_f32(0), 0.2);
        assert_eq!(inv.reg_f32(1), 0.4);
        assert_eq!(inv.reg_f32(2), 0.6);
        assert_eq!(inv.reg_f32(3), 0.8);
    }

    #[test]
    fn an_f16_texs_lands_its_channels_packed_as_halves() {
        // Asphalt 9's splash shader: `texs` in the packed form, whose result
        // the `h*2` ops after it read back as half pairs. Landing four floats
        // in four registers instead left every colour after the sample being
        // read as a pair of halves of an f32 bit pattern -- a red car came out
        // green.
        let sample = [0.25f32, 0.5, 0.75, 1.0];
        struct StubTex([f32; 4]);
        impl TextureSource for StubTex {
            fn sample(
                &self,
                _handle: u32,
                _u: f32,
                _v: f32,
                _layer: u32,
            ) -> ShaderResult<[f32; 4]> {
                Ok(self.0)
            }
        }

        // dst = $r1, dst2 = $r0, all four channels, precision bit clear.
        let texs = Op::Texs {
            dst: 1,
            dst2: 0,
            coords: [4, 5, RZ],
            dref: None,
            handle: 0,
            dim: TexDim::T2d,
            mask: [true, true, true, true],
            f16: true,
        };
        let program = Compiled::new(&super::super::Program {
            insns: vec![
                Instruction {
                    pred: Pred {
                        reg: 7,
                        negate: false,
                    },
                    op: texs,
                },
                Instruction {
                    pred: Pred {
                        reg: 7,
                        negate: false,
                    },
                    op: Op::Exit,
                },
            ],
            offsets: vec![8, 0x10],
            ..Default::default()
        });

        let no_consts: HashMap<(u8, u16), f32> = HashMap::new();
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts, &StubTex(sample)))
            .unwrap();

        // Two registers, not four: r1 holds (r, g) and r0 holds (b, a).
        assert_eq!(inv.reg(1), halves(sample[0], sample[1]));
        assert_eq!(inv.reg(0), halves(sample[2], sample[3]));
    }

    #[test]
    fn an_odd_channel_count_pads_its_second_half_with_zero() {
        struct StubTex;
        impl TextureSource for StubTex {
            fn sample(
                &self,
                _handle: u32,
                _u: f32,
                _v: f32,
                _layer: u32,
            ) -> ShaderResult<[f32; 4]> {
                Ok([0.25, 0.5, 0.75, 1.0])
            }
        }
        let program = Compiled::new(&super::super::Program {
            insns: vec![
                Instruction {
                    pred: Pred {
                        reg: 7,
                        negate: false,
                    },
                    op: Op::Texs {
                        dst: 2,
                        dst2: 4,
                        coords: [4, 5, RZ],
                        dref: None,
                        handle: 0,
                        dim: TexDim::T2d,
                        mask: [true, true, true, false],
                        f16: true,
                    },
                },
                Instruction {
                    pred: Pred {
                        reg: 7,
                        negate: false,
                    },
                    op: Op::Exit,
                },
            ],
            offsets: vec![8, 0x10],
            ..Default::default()
        });

        let no_consts: HashMap<(u8, u16), f32> = HashMap::new();
        let mut inv = Invocation::new();
        inv.execute(&program, &Env::new(&no_consts, &StubTex))
            .unwrap();
        assert_eq!(inv.reg(2), halves(0.25, 0.5));
        assert_eq!(inv.reg(4), halves(0.75, 0.0));
    }

    /// Pack two f32s into the pair of halves a register holds.
    fn halves(low: f32, high: f32) -> u32 {
        u32::from(f32_to_f16(low)) | (u32::from(f32_to_f16(high)) << 16)
    }

    fn lanes(bits: u32) -> [f32; 2] {
        half_lanes(bits, HSwizzle::H1H0)
    }

    fn hadd2(dst: u8, a: u8, b: u8, asw: HSwizzle, bsw: HSwizzle, merge: HMerge) -> Op {
        Op::Hadd2 {
            dst,
            a,
            am: FMod::NONE,
            asw,
            b: Operand::Reg(b),
            bm: FMod::NONE,
            bsw,
            merge,
            ftz: false,
            sat: false,
        }
    }

    fn run_half(setup: &[(u8, u32)], ops: &[Op]) -> Invocation {
        let consts = no_consts();
        let env = Env::new(&consts, &NoTextures);
        let mut inv = Invocation::new();
        for &(reg, value) in setup {
            inv.set_reg(reg, value);
        }
        let mut program: Vec<Op> = ops.to_vec();
        program.push(Op::Exit);
        inv.execute(&prog(&program), &env).unwrap();
        inv
    }

    #[test]
    fn a_half_op_computes_both_lanes_at_once() {
        let inv = run_half(
            &[(1, halves(1.0, 2.0)), (2, halves(0.5, -4.0))],
            &[hadd2(0, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0)],
        );
        assert_eq!(lanes(inv.reg(0)), [1.5, -2.0]);
    }

    /// Every swizzle but `H1_H0` reads one lane twice — and `F32` reads the
    /// register as a single float rather than as a pair at all, which is how
    /// a shader multiplies a `half2` by a `float`.
    #[test]
    fn a_half_swizzle_chooses_which_lanes_a_source_offers() {
        let a = halves(1.0, 2.0);
        let b = halves(10.0, 20.0);
        let inv = run_half(
            &[(1, a), (2, b)],
            &[
                hadd2(3, 1, 2, HSwizzle::H0H0, HSwizzle::H1H0, HMerge::H1H0),
                hadd2(4, 1, 2, HSwizzle::H1H1, HSwizzle::H1H0, HMerge::H1H0),
                hadd2(5, 1, 2, HSwizzle::H1H0, HSwizzle::H0H0, HMerge::H1H0),
            ],
        );
        assert_eq!(lanes(inv.reg(3)), [11.0, 21.0]);
        assert_eq!(lanes(inv.reg(4)), [12.0, 22.0]);
        assert_eq!(lanes(inv.reg(5)), [11.0, 12.0]);
    }

    /// `hadd2.f32` — swizzles and merge all `F32` — is a plain float add
    /// issued on the half unit, and it is what most of "A Short Hike"'s
    /// skipped draws stopped on.
    #[test]
    fn a_half_op_in_f32_mode_is_an_ordinary_float_op() {
        let inv = run_half(
            &[(1, 1.5f32.to_bits()), (2, 2.25f32.to_bits())],
            &[hadd2(0, 1, 2, HSwizzle::F32, HSwizzle::F32, HMerge::F32)],
        );
        assert_eq!(f32::from_bits(inv.reg(0)), 3.75);
    }

    /// A merging write leaves the other half of the destination alone, which
    /// also makes the destination one of the instruction's sources.
    #[test]
    fn a_merging_half_op_keeps_the_half_it_does_not_write() {
        let inv = run_half(
            &[
                (0, halves(7.0, 9.0)),
                (1, halves(1.0, 2.0)),
                (2, halves(0.5, -4.0)),
            ],
            &[hadd2(
                0,
                1,
                2,
                HSwizzle::H1H0,
                HSwizzle::H1H0,
                HMerge::MrgH0,
            )],
        );
        assert_eq!(lanes(inv.reg(0)), [1.5, 9.0]);

        let inv = run_half(
            &[
                (0, halves(7.0, 9.0)),
                (1, halves(1.0, 2.0)),
                (2, halves(0.5, -4.0)),
            ],
            &[hadd2(
                0,
                1,
                2,
                HSwizzle::H1H0,
                HSwizzle::H1H0,
                HMerge::MrgH1,
            )],
        );
        assert_eq!(lanes(inv.reg(0)), [7.0, -2.0]);

        let merging = hadd2(3, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::MrgH1);
        assert!(
            reads(&merging).contains(&3),
            "a merge reads its destination back"
        );
        let whole = hadd2(3, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0);
        assert!(!reads(&whole).contains(&3), "a full write does not");
    }

    #[test]
    fn a_half_multiply_add_runs_per_lane() {
        let inv = run_half(
            &[
                (1, halves(2.0, 3.0)),
                (2, halves(4.0, 5.0)),
                (3, halves(1.0, -1.0)),
            ],
            &[Op::Hfma2 {
                dst: 0,
                a: 1,
                asw: HSwizzle::H1H0,
                b: Operand::Reg(2),
                bneg: false,
                bsw: HSwizzle::H1H0,
                c: Operand::Reg(3),
                cneg: false,
                csw: HSwizzle::H1H0,
                merge: HMerge::H1H0,
                prec: HPrecision::None,
                sat: false,
            }],
        );
        assert_eq!(lanes(inv.reg(0)), [9.0, 14.0]);
    }

    /// `.fmz` is D3D9's rule that anything times zero is zero — infinity and
    /// NaN included, which an ordinary multiply answers with a NaN.
    #[test]
    fn fmz_makes_anything_times_zero_zero() {
        let hmul = |prec| Op::Hmul2 {
            dst: 0,
            a: 1,
            am: FMod::NONE,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(2),
            bm: FMod::NONE,
            bsw: HSwizzle::H1H0,
            merge: HMerge::H1H0,
            prec,
            sat: false,
        };
        let operands = [(1, halves(0.0, 2.0)), (2, halves(f32::INFINITY, 3.0))];
        let plain = run_half(&operands, &[hmul(HPrecision::None)]);
        assert!(lanes(plain.reg(0))[0].is_nan());
        let fmz = run_half(&operands, &[hmul(HPrecision::Fmz)]);
        assert_eq!(lanes(fmz.reg(0)), [0.0, 6.0]);
    }

    /// `hsetp2` writes one predicate per lane, unlike `fsetp`'s result and
    /// its inverse — until `.h_and`, which ands the lanes and then does write
    /// the inverse.
    #[test]
    fn hsetp2_writes_one_predicate_per_lane_until_h_and() {
        let setp = |and| Op::Hsetp2 {
            p0: 0,
            p1: 1,
            a: 1,
            am: FMod::NONE,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(2),
            bm: FMod::NONE,
            bsw: HSwizzle::H1H0,
            cmp: FCmp::Gt,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            and,
            ftz: false,
        };
        // Lane 0 compares true, lane 1 false.
        let operands = [(1, halves(5.0, 1.0)), (2, halves(2.0, 8.0))];
        let split = run_half(&operands, &[setp(false)]);
        assert!(split.pred(0) && !split.pred(1));
        let anded = run_half(&operands, &[setp(true)]);
        assert!(!anded.pred(0) && anded.pred(1));
    }

    /// `hset2` puts each lane's answer in its own half of the register.
    #[test]
    fn hset2_fills_each_half_with_its_own_lane() {
        let set = |bf| Op::Hset2 {
            dst: 0,
            a: 1,
            am: FMod::NONE,
            asw: HSwizzle::H1H0,
            b: Operand::Reg(2),
            bm: FMod::NONE,
            bsw: HSwizzle::H1H0,
            cmp: FCmp::Gt,
            bop: BoolOp::And,
            src: Pred::ALWAYS,
            bf,
            ftz: false,
        };
        let operands = [(1, halves(5.0, 1.0)), (2, halves(2.0, 8.0))];
        assert_eq!(run_half(&operands, &[set(false)]).reg(0), 0x0000_FFFF);
        // `.bf` answers with 1.0h rather than a mask.
        assert_eq!(run_half(&operands, &[set(true)]).reg(0), halves(1.0, 0.0));
    }

    /// A half instruction's `.ftz` flushes at the *half* threshold, four
    /// orders of magnitude above an f32's — but only for lanes that are
    /// halves.
    #[test]
    fn ftz_flushes_a_subnormal_half_but_not_a_small_float() {
        let subnormal = f16_to_f32(0x0001);
        let mut add = hadd2(0, 1, 2, HSwizzle::H1H0, HSwizzle::H1H0, HMerge::H1H0);
        let Op::Hadd2 { ftz, .. } = &mut add else {
            unreachable!()
        };
        *ftz = true;
        let inv = run_half(
            &[(1, halves(subnormal, 1.0)), (2, halves(0.0, 0.0))],
            &[add],
        );
        assert_eq!(lanes(inv.reg(0)), [0.0, 1.0]);

        // The same instruction reading an f32 lane leaves that value alone.
        let mut add = hadd2(0, 1, 2, HSwizzle::F32, HSwizzle::F32, HMerge::F32);
        let Op::Hadd2 { ftz, .. } = &mut add else {
            unreachable!()
        };
        *ftz = true;
        let inv = run_half(&[(1, subnormal.to_bits()), (2, 0.0f32.to_bits())], &[add]);
        assert_eq!(f32::from_bits(inv.reg(0)), subnormal);
    }
}
