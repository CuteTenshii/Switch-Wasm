//! Turning a region of translated blocks into wasm.
//!
//! A region function takes the address of the [`crate::cpu::Cpu`] in linear
//! memory and reaches guest state through baked [`Layout`] offsets. Guest
//! registers and NZCV live in locals, loaded on entry and stored on the way
//! out. Guest accesses walk the page table inline and make exactly the checks
//! of [`crate::mem::Memory::peek`]/[`crate::mem::Memory::poke`], handing the
//! instruction back to the interpreter when they decline. Branches between the
//! region's blocks stay inside the function while the fuel lasts. `run`
//! returns how many instructions the block it stopped in retired, that block's
//! index above [`MEMBER_SHIFT`], and [`LEFT`] when it left at a taken branch or
//! a terminator [`writes_term`] accepts. A block with anything the emitter
//! cannot write is [`Refused`]. Chained regions leave by tail-calling the next
//! compiled region when the chain table has it and the fuel allows.

use super::cache::{Chain, Jit, CHAIN_SLOTS};
use super::decode::{decode, translate, Decoded, MAX_BLOCK_OPS};
use super::ir::{Block, Exit, Op, Term};
use super::wasm::{Func, Module, F32, F64, I32, I64, V128};
use crate::cpu::bits::mask_of_width;
use crate::cpu::loadstore::{Acc, Ext, PairKind, Wb};
use crate::cpu::power::CLOCK_RATES_HZ;
use crate::cpu::system::{SysOp, SysReg, TICK_HZ};
use crate::cpu::{Cpu, CONDITION_MASKS, REG_SLOTS, SELF_RETURN_TRAMPOLINE};
use crate::mem::{PAGE_BITS, PAGE_SIZE};

mod a32;
mod a32_neon;
mod a32_vfp;
mod alu;
mod dispatch;
mod flow;
mod fp;
mod mem;
mod simd;

pub use a32::a32_emits;
pub(super) use a32::emit_a32;
use mem::access_bytes;

/// Whether the emitter has a way to write `insn` out as wasm. Block-ending and
/// branching instructions answer `false`: those are exits, not ops.
pub fn emits(insn: u32) -> bool {
    // Any aligned address works for PC-relative forms.
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    let Decoded::Op(op) = decode(insn, REPRESENTATIVE_PC) else {
        return false;
    };
    let mut f = scratch_func();
    Emitter::new(&mut f, SCRATCH, false, Vec::new()).op(&op, 0)
}

/// Whether `insn` is a branch the emitter writes as a block exit or terminator.
pub fn writes_exit(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    match decode(insn, REPRESENTATIVE_PC) {
        Decoded::Op(_) => false,
        Decoded::Exit(_) => true,
        Decoded::Term(term) => writes_term(&term),
    }
}

/// Whether the emitted form of `insn` can decline and hand the instruction
/// back to the interpreter (every guest access can). Derived by emitting the
/// instruction under two retired counts and comparing the bodies.
pub fn defers(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    let Decoded::Op(op) = decode(insn, REPRESENTATIVE_PC) else {
        return false;
    };
    let body = |retired: usize| {
        let mut f = scratch_func();
        Emitter::new(&mut f, SCRATCH, false, Vec::new()).op(&op, retired);
        f.code().to_vec()
    };
    body(0) != body(1)
}

/// A function body with the locals every emitted region declares.
fn scratch_func() -> Func {
    let mut f = Func::new();
    f.locals(1, 6, I64);
    f.locals(1, 10, I32);
    f.locals(1, 3, V128);
    f.locals(1, 2, F32);
    f.locals(1, 2, F64);
    f.locals(1, ENV_SLOTS, I32);
    f
}

/// Offsets for code that is inspected, never run.
const SCRATCH: Layout = Layout {
    regs: 0,
    nzcv: 0,
    fpsr: 0,
    fpcr: 0,
    fpscr_nzcv: 0,
    pc: 0,
    fuel: 0,
    memory: 0,
    pages: 0,
    read_watch_lo: 0,
    read_watch_hi: 0,
    watch_lo: 0,
    watch_hi: 0,
    watched: 0,
    pair_lock: 0,
    vregs: 0,
    exclusive: 0,
    tpidr: 0,
    tpidr_rw: 0,
    core: 0,
    clock_at: 0,
    origin: 0,
};

/// Byte offsets of the guest state an emitted block touches, from the pointer
/// it is handed. Taken from `offset_of!`, since `Cpu` is not `#[repr(C)]`.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    /// Start of the `[u64; REG_FILE]` register file.
    pub regs: u32,
    /// The packed NZCV word, in its architectural bit positions.
    pub nzcv: u32,
    /// FPSR, whose cumulative exception bits some conversions and divides set.
    pub fpsr: u32,
    /// FPCR, whose rounding mode FRINTX and FRINTI follow.
    pub fpcr: u32,
    /// The flags `VCMP` leaves in the FPSCR, in NZCV's bit positions.
    pub fpscr_nzcv: u32,
    /// The guest program counter, written only by a taken branch leaving the region.
    pub pc: u32,
    /// Instructions blocks left through a branch may still retire, as an `i32`;
    /// a branch that takes it below zero leaves the region.
    pub fuel: u32,
    /// Start of the `[u128; 32]` SIMD&FP register file.
    pub vregs: u32,
    /// Where the running process's `Memory` address is kept; the fields below are
    /// offsets from that address.
    pub memory: u32,
    /// Where the page table's pointer is kept (the table itself is boxed).
    pub pages: u32,
    /// The read watchpoint, as the two `u32`s of its `[start, end)`.
    pub read_watch_lo: u32,
    pub read_watch_hi: u32,
    /// The write watchpoint, likewise.
    pub watch_lo: u32,
    pub watch_hi: u32,
    /// Where the pointer to the cached-pages bitmap is kept; null until
    /// something watches a page. Write-protected pages are set in it too.
    pub watched: u32,
    /// Where the byte that serialises 16-byte exclusive stores is kept.
    pub pair_lock: u32,
    /// The exclusive monitor, `Monitor`'s `addr`, `armed` and `value` at fixed offsets from here.
    pub exclusive: u32,
    /// `TPIDRRO_EL0` and `TPIDR_EL0`, the thread's TLS base and its software register.
    pub tpidr: u32,
    pub tpidr_rw: u32,
    /// Which core this is, as a `u32`; core 0 keeps the clock the others read.
    pub core: u32,
    /// Where the pointer to core 0's published clock, a `u64`, is kept.
    pub clock_at: u32,
    /// The `u64` clock origin of the chain being run; see `Chain::origin`.
    pub origin: u32,
}

impl Layout {
    /// Where that state sits inside a real [`Cpu`].
    pub const fn of_cpu() -> Layout {
        let m = crate::mem::Offsets::OF_MEMORY;
        Layout {
            regs: std::mem::offset_of!(Cpu, regs) as u32,
            nzcv: std::mem::offset_of!(Cpu, nzcv) as u32,
            fpsr: std::mem::offset_of!(Cpu, fpsr) as u32,
            fpcr: std::mem::offset_of!(Cpu, fpcr) as u32,
            fpscr_nzcv: std::mem::offset_of!(Cpu, fpscr_nzcv) as u32,
            pc: std::mem::offset_of!(Cpu, pc) as u32,
            fuel: (std::mem::offset_of!(Cpu, jit)
                + std::mem::offset_of!(Jit, chain)
                + std::mem::offset_of!(Chain, fuel)) as u32,
            vregs: std::mem::offset_of!(Cpu, vregs) as u32,
            memory: std::mem::offset_of!(Cpu, mem) as u32 + crate::mem::AddressSpace::MEMORY_AT,
            pages: m.pages,
            read_watch_lo: m.read_watch_lo,
            read_watch_hi: m.read_watch_hi,
            watch_lo: m.watch_lo,
            watch_hi: m.watch_hi,
            watched: m.watched,
            pair_lock: m.pair_lock,
            exclusive: std::mem::offset_of!(Cpu, exclusive) as u32,
            tpidr: std::mem::offset_of!(Cpu, tpidr) as u32,
            tpidr_rw: std::mem::offset_of!(Cpu, tpidr_rw) as u32,
            core: std::mem::offset_of!(Cpu, core) as u32,
            clock_at: std::mem::offset_of!(Cpu, clock_at) as u32,
            origin: (std::mem::offset_of!(Cpu, jit)
                + std::mem::offset_of!(Jit, chain)
                + std::mem::offset_of!(Chain, origin)) as u32,
        }
    }
}

/// Log2 bytes per page-table entry, an `AtomicPtr` that is null when unmapped; only `wasm32` runs emitted code.
const PAGE_ENTRY_SHIFT: i32 = 2;
#[cfg(target_arch = "wasm32")]
const _: () = assert!(
    std::mem::size_of::<std::sync::atomic::AtomicPtr<[u8; PAGE_SIZE]>>() == 1 << PAGE_ENTRY_SHIFT,
    "emitted code indexes the page table by this shift, so it has to be the entry's size"
);

/// Alignment hints, as log2. Guest addresses promise no alignment.
const ALIGN_4: u8 = 2;
const ALIGN_8: u8 = 3;
const UNALIGNED: u8 = 0;
/// `v128.load` and `v128.store`, after the `0xFD` prefix.
const V128_LOAD: u32 = 0x00;
const V128_STORE: u32 = 0x0B;

/// wasm `i64` comparisons that `i64_eq`, `i64_ne` and `i64_lt_u` do not cover.
const I64_LT_S: u8 = 0x53;
const I64_GT_S: u8 = 0x55;
const I64_GT_U: u8 = 0x56;
const I64_LE_S: u8 = 0x57;
const I64_LE_U: u8 = 0x58;
const I64_GE_S: u8 = 0x59;
const I64_GE_U: u8 = 0x5A;

/// Why a block was not written out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// An exit the walk over the body never reached.
    ControlFlow,
    /// An op with no emitter, as the instruction word it was decoded from.
    Op(u32),
    /// The body grew past [`MAX_BODY_BYTES`].
    TooLong,
}

/// Set in `run`'s result when the region left at a taken branch; the guest `pc`
/// holds the target. A block retires at most `MAX_WHOLE`, so counts fit `COUNT_MASK`.
pub const LEFT: u32 = 1 << 31;

/// The retired count in `run`'s result.
pub(super) const COUNT_MASK: u32 = 0xFFFF;

/// Where `run`'s result keeps the index of the block it stopped in.
pub(super) const MEMBER_SHIFT: u32 = 16;

/// The parameter every block function takes.
const STATE: u32 = 0;

/// Offsets inside `Monitor`, which is `#[repr(C)]`.
const MONITOR_ADDR: u32 = 0;
const MONITOR_ARMED: u32 = 4;
const MONITOR_VALUE: u32 = 8;
const _: () = {
    assert!(std::mem::offset_of!(crate::cpu::Monitor, addr) as u32 == MONITOR_ADDR);
    assert!(std::mem::offset_of!(crate::cpu::Monitor, armed) as u32 == MONITOR_ARMED);
    assert!(std::mem::offset_of!(crate::cpu::Monitor, value) as u32 == MONITOR_VALUE);
};

/// Scratch locals.
const L_A: u32 = 1;
const L_B: u32 = 2;
const L_T: u32 = 3;
const L_R: u32 = 4;
/// The duplicate operand of a 32-bit rotate.
const L_S: u32 = 5;
/// The base register's writeback value, computed before the access.
const L_W: u32 = 6;
const L_C: u32 = 7;
/// The guest address of an access.
const L_ADDR: u32 = 8;
/// The host address of the guest page that address is on.
const L_PAGE: u32 = 9;
/// The guest page index, for the page table and the watched-pages bitmap.
const L_IDX: u32 = 10;
/// The watched-pages bitmap.
const L_WP: u32 = 11;
/// `run`'s result, set before branching to the epilogue.
const L_RET: u32 = 12;
/// The block a branch to the loop head goes to.
const L_NEXT: u32 = 13;
const L_FUEL: u32 = 14;
/// Where a region that left is going.
const L_TARGET: u32 = 15;
const L_NZCV: u32 = 16;
/// Vector scratch.
const L_V0: u32 = 17;
const L_V1: u32 = 18;
const L_V2: u32 = 19;
/// Float scratch, two of each width.
const L_F32: u32 = 20;
const L_F32B: u32 = 21;
const L_F64: u32 = 22;
const L_F64B: u32 = 23;
/// Memory state only the host changes, loaded once per run instead of per access.
const L_ENV: u32 = 24;
const ENV_SLOTS: u32 = 8;
/// The first register local; slots get theirs in the order the region touches them.
const L_REGS: u32 = L_ENV + ENV_SLOTS;

/// Reads as zero and is never written.
const ZR_SLOT: u8 = 31;

/// Register 31 as a base register is SP.
fn base_slot(rn: u32) -> u8 {
    if rn == 31 {
        crate::cpu::regs::SP_SLOT as u8
    } else {
        rn as u8
    }
}

/// Register 31 as a destination is the zero register, whose writes land in the discard slot.
fn gp_dst(rd: u32) -> u8 {
    if rd == 31 {
        DISCARD
    } else {
        rd as u8
    }
}
/// Takes writes to the zero register; not guest state, so never loaded or stored.
const DISCARD: u8 = crate::cpu::DISCARD_SLOT as u8;

/// A block bigger than this is not emitted, to bound module compile time.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// Instructions in a PLT stub, which a folded call retires after the `BL`.
pub(super) const PLT_STUB: u32 = 4;

/// The most a block can retire, terminator and stub included: fuel keeps this
/// much budget for the next.
pub(super) const MAX_WHOLE: i64 = MAX_BLOCK_OPS as i64 + 1 + PLT_STUB as i64;

/// Offsets from the `Cpu` of what a jump between regions touches.
struct ChainLayout {
    hops: u32,
    last: u32,
    slots: u32,
}

impl ChainLayout {
    const OF_CPU: ChainLayout = {
        let chain = std::mem::offset_of!(Cpu, jit) + std::mem::offset_of!(Jit, chain);
        ChainLayout {
            hops: (chain + std::mem::offset_of!(Chain, hops)) as u32,
            last: (chain + std::mem::offset_of!(Chain, last)) as u32,
            slots: (chain + std::mem::offset_of!(Chain, slots)) as u32,
        }
    };
}

/// The immediate of a fused `CMP` inverted at emit time, as
/// [`Emitter::invert_if`] does at run time.
fn invert_if_const(v: u64, carry: u8) -> u64 {
    v ^ 0u64.wrapping_sub(u64::from(carry))
}

/// The mask an operation of this width applies to its operands and result.
fn width_mask(sf: bool) -> i64 {
    if sf {
        -1
    } else {
        0xFFFF_FFFF
    }
}

/// The bit an operation of this width calls the sign.
fn sign_bit(sf: bool) -> i64 {
    if sf {
        i64::MIN
    } else {
        0x8000_0000u32 as i64
    }
}

/// How wide an operation of this width is.
fn size_of(sf: bool) -> u32 {
    if sf {
        64
    } else {
        32
    }
}

struct Emitter<'a> {
    f: &'a mut Func,
    layout: Layout,
    chain: bool,
    /// Whether the engine has SIMD128, for the Advanced SIMD forms.
    simd: bool,
    /// Entry addresses of the region's blocks a branch may stay inside for.
    targets: Vec<u32>,
    /// The block being emitted, as an index into the region.
    member: usize,
    /// The op being emitted, as an index into its block.
    at: usize,
    epilogue: u32,
    /// The loop head that dispatches on `L_NEXT`, when a branch stays inside.
    top: Option<u32>,
    /// Per member, the label whose end its code follows, while that is open.
    starts: Vec<u32>,
    /// Register slots the region touches, loaded on entry.
    used: u64,
    /// Register slots it writes, stored on the way out.
    written: u64,
    /// Each slot's local, or 0 before it has one.
    locals: [u32; REG_SLOTS],
    /// Each vector register's local, or 0 before it has one.
    vlocals: [u32; 32],
    /// Vector registers the region touches, loaded on entry.
    vused: u32,
    /// Vector registers it writes, stored on the way out.
    vwritten: u32,
    /// The type of each local from `L_REGS` on, in index order.
    local_types: Vec<u8>,
    next_local: u32,
    /// The operands of a compare whose NZCV is not built yet, and which form
    /// it is (0 when none is held, else 1 for 32-bit and 2 for 64-bit), once
    /// the region has one. [`Emitter::epilogue`] builds the word if a run
    /// leaves while one is held.
    pend_a: u32,
    pend_b: u32,
    pend_kind: u32,
    /// The forms of compare the region holds, as bit 0 for 32-bit and bit 1 for 64-bit.
    pend_forms: u8,
    /// Whether the compare being emitted is overwritten before anything reads its flags.
    flags_dead: bool,
    /// Whether a watchpoint was armed when the region was emitted.
    watches: bool,
    /// Where the region's calls return to: what a `RET` in it is compared with.
    returns: Vec<u32>,
    flags_used: bool,
    flags_written: bool,
    fuel_used: bool,
    /// Whether any exit leaves with [`LEFT`].
    leaves: bool,
    /// [`Emitter::env`] offsets, by local, that the region reads.
    env: [Option<u32>; ENV_SLOTS as usize],
}

impl<'a> Emitter<'a> {
    /// Opens the block every exit branches out of to reach the epilogue.
    fn new(f: &'a mut Func, layout: Layout, chain: bool, targets: Vec<u32>) -> Emitter<'a> {
        let epilogue = f.block_void();
        Emitter {
            f,
            layout,
            chain,
            simd: true,
            watches: true,
            returns: Vec::new(),
            targets,
            member: 0,
            at: 0,
            epilogue,
            top: None,
            starts: Vec::new(),
            used: 0,
            written: 0,
            locals: [0; REG_SLOTS],
            vlocals: [0; 32],
            vused: 0,
            vwritten: 0,
            local_types: Vec::new(),
            next_local: L_REGS,
            pend_a: 0,
            pend_b: 0,
            pend_kind: 0,
            pend_forms: 0,
            flags_dead: false,
            flags_used: false,
            flags_written: false,
            fuel_used: false,
            leaves: false,
            env: [None; ENV_SLOTS as usize],
        }
    }
}

impl Emitter<'_> {
    /// Push `regs[slot]` unnarrowed, for ops that shift its high bits out.
    fn read_reg_raw(&mut self, slot: u8) {
        if slot == ZR_SLOT {
            self.f.i64_const(0);
            return;
        }
        self.used |= 1 << slot;
        let local = self.local_of(slot);
        self.f.local_get(local);
    }

    /// A new local of type `ty`.
    fn fresh_local(&mut self, ty: u8) -> u32 {
        self.local_types.push(ty);
        self.next_local += 1;
        self.next_local - 1
    }

    /// The local holding `slot`, assigned on first use.
    fn local_of(&mut self, slot: u8) -> u32 {
        if self.locals[usize::from(slot)] == 0 {
            self.locals[usize::from(slot)] = self.fresh_local(I64);
        }
        self.locals[usize::from(slot)]
    }

    /// The local holding `V<r>`, assigned on first use.
    fn vlocal_of(&mut self, r: u32) -> u32 {
        if self.vlocals[r as usize] == 0 {
            self.vlocals[r as usize] = self.fresh_local(V128);
        }
        self.vlocals[r as usize]
    }

    /// Narrow the value on the stack to the operation's width.
    fn mask_to(&mut self, sf: bool) {
        if !sf {
            self.f.i64_const(width_mask(false));
            self.f.i64_and();
        }
    }

    /// Sign-extend the value on the stack from the operation's width.
    fn sext_to(&mut self, sf: bool) {
        if !sf {
            self.f.i64_extend32_s();
        }
    }

    /// Push `regs[slot]`, already narrowed to the operation's width.
    fn read_reg(&mut self, slot: u8, sf: bool) {
        self.read_reg_raw(slot);
        self.mask_to(sf);
    }

    /// Push the `u32` at `offset` from the `Cpu`, which only the host writes,
    /// from the local the prologue loads it into.
    fn env(&mut self, offset: u32) {
        let slot = match self.env.iter().position(|&o| o == Some(offset)) {
            Some(slot) => slot,
            None => {
                let slot = self
                    .env
                    .iter()
                    .position(Option::is_none)
                    .expect("more host-written words than ENV_SLOTS");
                self.env[slot] = Some(offset);
                slot
            }
        };
        self.f.local_get(L_ENV + slot as u32);
    }

    /// Write the value on the stack to `regs[slot]`.
    fn store_reg(&mut self, slot: u8) {
        if slot != DISCARD {
            self.used |= 1 << slot;
            self.written |= 1 << slot;
        }
        let local = self.local_of(slot);
        self.f.local_set(local);
    }
}

pub(super) fn writes_term(term: &Term) -> bool {
    matches!(
        term,
        Term::B { .. }
            | Term::Bl { .. }
            | Term::Br { .. }
            | Term::Blr { .. }
            | Term::Ret { .. }
            | Term::BlPlt { .. }
            | Term::BPlt { .. }
    )
}

/// Instructions a written `term` retires: a folded PLT call runs its stub too.
pub(super) fn term_retires(term: &Term) -> usize {
    match term {
        Term::BlPlt { .. } | Term::BPlt { .. } => 1 + PLT_STUB as usize,
        _ => 1,
    }
}

/// A module that validates only if the engine has SIMD128, for the embedder
/// to set [`super::JitHost::simd`].
pub fn simd_probe() -> Vec<u8> {
    let mut f = Func::new();
    f.v128_const([0; 16]);
    f.drop_value();
    f.local_get(STATE);
    f.end();
    let mut m = Module::new();
    let ty = m.add_type(vec![I32], vec![I32]);
    m.add_func(ty, f);
    m.finish()
}

/// A module that validates only if the engine has the tail calls
/// chained regions use; for the embedder to set [`super::JitHost::tail_calls`].
pub fn tail_call_probe() -> Vec<u8> {
    let mut f = Func::new();
    f.local_get(STATE);
    f.local_get(STATE);
    f.return_call_indirect(0);
    f.end();
    let mut m = Module::new();
    m.import_table();
    let ty = m.add_type(vec![I32], vec![I32]);
    m.add_func(ty, f);
    m.finish()
}

/// Emit `block` alone as a module exporting `run`, leaving at every branch.
pub(super) fn emit_block(block: &Block, layout: Layout, watches: bool) -> Result<Vec<u8>, Refused> {
    emit_region(&[block], layout, false, false, true, watches).map_err(|(_, refused)| refused)
}

/// Emit the region `blocks`, entered at the first, as a module exporting
/// `run`. `inside` lets branches between them stay in the function; `chain`
/// jumps into compiled successors through the function table it imports;
/// `simd` writes Advanced SIMD as SIMD128; `watches` tests the watchpoints. A
/// refusal names the block that caused it.
pub(super) fn emit_region(
    blocks: &[&Block],
    layout: Layout,
    chain: bool,
    inside: bool,
    simd: bool,
    watches: bool,
) -> Result<Vec<u8>, (usize, Refused)> {
    let targets: Vec<u32> = if inside {
        blocks.iter().map(|b| b.start).collect()
    } else {
        Vec::new()
    };
    let looped = blocks
        .iter()
        .any(|b| b.region_successors().iter().any(|pc| targets.contains(pc)));
    let n = blocks.len();
    let mut f = scratch_func();
    let mut e = Emitter::new(&mut f, layout, chain, targets);
    e.simd = simd;
    e.watches = watches;
    e.returns = blocks
        .iter()
        .filter_map(|b| match b.term {
            Some(
                Term::Bl { ret_pc, .. } | Term::Blr { ret_pc, .. } | Term::BlPlt { ret_pc, .. },
            ) => Some(ret_pc),
            _ => None,
        })
        .collect();
    if looped {
        e.top = Some(e.f.loop_void());
        if n > 1 {
            e.starts = vec![0; n];
            for j in (0..n).rev() {
                e.starts[j] = e.f.block_void();
            }
            e.f.local_get(L_NEXT);
            let table: Vec<u32> = (0..n as u32).collect();
            e.f.br_table(&table, 0);
        }
    }
    for (j, block) in blocks.iter().enumerate() {
        if looped && n > 1 {
            e.f.end();
        }
        e.member = j;
        e.block(block).map_err(|refused| (j, refused))?;
    }
    if looped {
        e.f.end();
    }
    e.f.end();
    e.epilogue();
    let prologue = e.prologue();
    let local_types = std::mem::take(&mut e.local_types);
    f.prepend(&prologue);
    for run in local_types.chunk_by(|a, b| a == b) {
        f.locals(1, run.len() as u32, run[0]);
    }
    f.end();

    let mut m = Module::new();
    if chain {
        m.import_table();
    }
    let ty = m.add_type(vec![I32], vec![I32]);
    let idx = m.add_func(ty, f);
    m.export("run", idx);
    Ok(m.finish())
}

impl Cpu {
    /// The same for the A32 block at `pc`, with how many instructions it can retire.
    pub fn emit_a32_block_at(&self, pc: u32, layout: Layout) -> Result<(Vec<u8>, usize), Refused> {
        let block = super::a32::translate(&self.mem, pc);
        let emitted = emit_a32(&block, layout, false, self.mem.watchpoints_armed())?;
        Ok((emitted.module, emitted.retires))
    }

    /// Translate the block at `pc` and emit it against `layout`, for
    /// `examples/emit_difftest.rs`. Also reports the address of each covered
    /// instruction and the one after.
    pub fn emit_block_at(&self, pc: u32, layout: Layout) -> Result<(Vec<u8>, Vec<u32>), Refused> {
        let block = translate(&self.mem, pc);
        let bytes = emit_block(&block, layout, self.mem.watchpoints_armed())?;
        let mut jumps = block.exits.iter().filter_map(|branch| match branch.exit {
            Exit::Jump { target } => Some((branch.at as usize + branch.span as usize, target)),
            _ => None,
        });
        let mut next_jump = jumps.next();
        let mut path = Vec::with_capacity(block.ops.len() + 1);
        let mut at = pc;
        for i in 0..=block.ops.len() {
            if let Some((_, target)) = next_jump.filter(|&(after, _)| after == i) {
                at = target;
                next_jump = jumps.next();
            }
            path.push(at);
            at = at.wrapping_add(4);
        }
        // A written terminator is covered too; no branch target is `u32::MAX`. A
        // folded PLT call reads its GOT slot, which difftests never map, so it is not.
        let covered = |term: &Term| {
            writes_term(term) && !matches!(term, Term::BlPlt { .. } | Term::BPlt { .. })
        };
        if block.term.as_ref().is_some_and(covered) {
            path.push(u32::MAX);
        }
        Ok((bytes, path))
    }

    /// The thread registers `MRS` reads, for the difftests to seed.
    pub fn set_thread_pointers(&mut self, tpidr: u64, tpidr_rw: u64) {
        self.tpidr = tpidr;
        self.tpidr_rw = tpidr_rw;
    }

    /// `TPIDRRO_EL0` and `TPIDR_EL0`, for the difftests to carry into a case.
    pub fn thread_pointers(&self) -> (u64, u64) {
        (self.tpidr, self.tpidr_rw)
    }

    /// Arm the exclusive monitor at `addr` as having read `value`, or disarm it.
    pub fn set_exclusive(&mut self, armed: bool, addr: u32, value: [u64; 2]) {
        if armed {
            self.exclusive.arm(addr, value);
        } else {
            self.exclusive = Default::default();
        }
    }

    /// The monitor as `(address, armed, doublewords)`, which the difftests compare.
    pub fn exclusive_state(&self) -> (u32, bool, [u64; 2]) {
        (
            self.exclusive.addr,
            self.exclusive.armed != 0,
            self.exclusive.value,
        )
    }

    /// The register file by slot, including the three register 31 resolves to.
    pub fn reg_slots(&self) -> [u64; crate::cpu::REG_SLOTS] {
        let mut out = [0u64; crate::cpu::REG_SLOTS];
        out.copy_from_slice(&self.regs[..crate::cpu::REG_SLOTS]);
        out
    }
}

#[cfg(test)]
mod tests;
