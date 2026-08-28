//! A block-translating JIT for the A64 core.
//!
//! The interpreter re-derives everything about an instruction every time it
//! runs it: which of the eight top-level groups it belongs to, which of that
//! group's forms it is, where its operand fields sit, what its immediate
//! decodes to. In a loop body that runs a million times, that work is done a
//! million times and produces the same answer every time.
//!
//! This module does it once. The first time control reaches an address, the
//! translator walks forward from it decoding instructions into [`Op`]s — the
//! operation with its operands already extracted, its immediates already
//! decoded, its PC-relative addresses already resolved, and every field the
//! interpreter re-reads per execution (a load's width and direction, a
//! register offset's extension, whether an add is really a subtract, which
//! system register an `MRS` names, which floating-point form an encoding is)
//! already resolved to the one thing the instruction does. That run of ops
//! plus its terminator is a [`Block`], cached by entry address, and every
//! later visit executes it with no decoding at all.
//!
//! A block does not end at the first branch. The three conditional branches —
//! `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ` — have the following instruction as
//! their not-taken path, so translation carries on through them and each
//! becomes an [`Exit`] the body is checked against on the way past. Only an
//! instruction that *always* leaves ends a block. `b.cond` alone is 12% of an
//! hbmenu frame, so this is the difference between blocks that average seven
//! instructions and blocks that average thirteen.
//!
//! What this removes is decode, not dispatch — three quarters of where the
//! interpreter's time went (see [`super::Cpu::execute`]'s note on the group
//! dispatcher). It does not generate code.
//!
//! # Why not generate wasm
//!
//! Emitting a wasm module per translation unit and handing it to
//! `WebAssembly.Module` is a real JIT, and it is the next step for speed
//! rather than something the browser forbids. What stops it being *this*
//! step is the memory model, not the platform.
//!
//! A generated module can only address its own linear memory directly. Guest
//! memory cannot be that: this emulator is itself a wasm32 module, whose
//! linear memory caps at 4 GiB, and the guest address space is 4 GiB reaching
//! up to [`super::GUEST_SPACE_END`] — so an identity map does not fit, and
//! would give up the lazy soft-mapping that makes a multi-gigabyte heap cost
//! nothing until it is touched. [`crate::mem::Memory`] is a page table of
//! boxed 4 KiB pages with soft regions, read-only ranges and watchpoints over
//! it. Generated code would have to reach all of that through imported host
//! calls, one per guest load and store — which is most of what a block does,
//! and exactly what the codegen was supposed to make cheap.
//!
//! Two smaller things point the same way: compiling a module goes out through
//! JS, so a basic block is far too small a unit to pay for one, and generated
//! code would not exist in host builds — leaving the test suite and
//! `examples/jit_difftest.rs` covering only one of the two engines. Flattening
//! the guest address space behind a base-plus-bounds check is the change that
//! has to come first.
//!
//! # Fidelity
//!
//! Every op executes the same helper the interpreter's decoder would have
//! called with the same arguments, so translated and interpreted execution are
//! the same computation. Anything the translator does not have an op for — the
//! exclusive accessors, the divides and variable shifts, the encodings the
//! interpreter rejects as unallocated — becomes [`Op::Interpret`], which hands
//! the raw instruction word straight back to [`super::Cpu::execute`]. That
//! makes the translator's coverage a performance question rather than a
//! correctness one: a form it does not know is slower, never wrong.
//!
//! SIMD and floating point are half-way between. They have no ops of their
//! own, but which decoder owns an encoding — and, for the scalar forms, which
//! of [`super::fp::FpForm`]'s eight groups it belongs to — is settled at
//! translation time, so [`Op::Fp`] enters the right handler directly instead
//! of walking a guard chain. The classification lives in
//! [`super::Cpu::fp_form`] and the interpreter asks the same function, so the
//! two cannot drift.
//!
//! An op may only be a mid-block [`Op::Interpret`] if it cannot move the PC
//! anywhere but to the following instruction. Every A64 instruction that can
//! is in the branch/exception/system group (bits 28:25 = `101x`), so that
//! group is decoded as a terminator — except the `D503xxxx` hints and
//! barriers, which are no-ops, and the `MRS`/`MSR`/cache-maintenance forms,
//! which [`super::Cpu::system`] always retires to `next_pc`.
//!
//! # Staleness
//!
//! A block is only valid while the instructions it was built from are still
//! there. [`crate::mem::Memory`] records which pages have been translated out
//! of and reports the ones a store has landed on; [`Cpu::jit_block_at`] drains
//! that list before every lookup and drops the blocks translated from those
//! pages. A block never spans a page, so one page's worth of invalidation is
//! exact.

use super::bits::*;
use super::fp::FpForm;
use super::{Cpu, Result, RunReport, SELF_RETURN_TRAMPOLINE, SP_SLOT, TIME_SLICE, ZR_DISCARD};
use crate::mem::{Memory, PAGE_SIZE};
use crate::IdMap;
use std::rc::Rc;

/// Longest run of instructions one block may cover. Longer blocks amortize the
/// per-block bookkeeping over more work, but they also make the step budget
/// coarser. Not the binding limit in practice: raising it to 160 moved
/// hbmenu's block entries by 0.2%, because what actually ends a block is an
/// unconditional branch or the end of the page.
const MAX_BLOCK_OPS: usize = 64;

/// How many blocks the cache holds before it is dropped wholesale. A retail
/// title's hot code is a few thousand blocks; the cap is only there so a
/// program that walks endlessly over fresh code cannot grow the cache without
/// bound.
const MAX_BLOCKS: usize = 64 * 1024;

/// What a load/store does to its base register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Wb {
    /// No writeback: the access is at `base + offset`.
    None,
    /// Pre-index: the access is at `base + offset`, which is also written back.
    Pre,
    /// Post-index: the access is at `base`, and `base + offset` is written back.
    Post,
}

/// What a load or store actually does, decided when the block is translated.
///
/// [`super::Cpu::ld_st_opc`] derives this from the `size` and `opc` fields
/// every time it runs: the `PRFM` test, the load/store test, the sign-extend
/// test, and then a second match inside `load_by_size` to pick the width, and
/// a third to pick the sign-extension width. All of it is constant per
/// instruction, so all of it belongs here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Acc {
    Store8,
    Store16,
    Store32,
    Store64,
    Load8,
    Load16,
    Load32,
    Load64,
    LoadS8,
    LoadS16,
    LoadS32,
    /// `PRFM`, a hint with no architectural effect. Still an op, because the
    /// addressing mode's writeback happens whether or not the access does.
    Prefetch,
}

impl Acc {
    /// Whether the access writes `Rt`. Decides which of register 31's two
    /// meanings the `Rt` field names, and so which slot the translator bakes.
    fn writes_rt(self) -> bool {
        !matches!(
            self,
            Acc::Store8 | Acc::Store16 | Acc::Store32 | Acc::Store64 | Acc::Prefetch
        )
    }

    /// The access a `size`:`opc` pair selects, reading them exactly as
    /// [`super::Cpu::ld_st_opc`] does.
    fn of(sz: u8, opc: u8) -> Acc {
        if sz == 0b11 && opc >= 0b10 {
            return Acc::Prefetch;
        }
        match (opc, sz) {
            (0b00, 0b00) => Acc::Store8,
            (0b00, 0b01) => Acc::Store16,
            (0b00, 0b10) => Acc::Store32,
            (0b00, _) => Acc::Store64,
            (0b01, 0b00) => Acc::Load8,
            (0b01, 0b01) => Acc::Load16,
            (0b01, 0b10) => Acc::Load32,
            (0b01, _) => Acc::Load64,
            (_, 0b00) => Acc::LoadS8,
            (_, 0b01) => Acc::LoadS16,
            (_, _) => Acc::LoadS32,
        }
    }
}

/// How a register-offset load/store extends its index register. The four
/// encodings A64 defines collapse to three behaviours; the other four are
/// undefined and never reach a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ext {
    /// `UXTW`: the low 32 bits, zero-extended.
    Uxtw,
    /// `SXTW`: the low 32 bits, sign-extended.
    Sxtw,
    /// `LSL`, `UXTX` and `SXTX`, all of which take the register as it stands.
    None,
}

/// The system register a translated `MRS`/`MSR` names, resolved from the
/// `op0:op1:CRn:CRm:op2` fields once instead of on every execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SysReg {
    Nzcv,
    /// `TPIDR_EL0`, which guest code may write.
    Tpidr,
    /// `TPIDRRO_EL0`, the kernel-fixed TLS base. Read-only at EL0, so an
    /// `MSR` to it is ignored rather than refused.
    TpidrRo,
    Fpcr,
    Fpsr,
    /// A register that always reads the same value: the two Cortex-A57
    /// constants [`super::Cpu::system`] reports, and zero for everything it
    /// does not model. Writes to these are dropped. Both constants fit in 32
    /// bits, which keeps [`Op`] to one 64-bit word.
    Fixed(u32),
}

/// What a load/store pair moves, and which way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PairKind {
    Load32,
    /// `LDPSW`: two words, each sign-extended to 64 bits.
    Load32Sext,
    Load64,
    Store32,
    Store64,
}

impl PairKind {
    /// Whether the pair writes its two registers.
    fn loads(self) -> bool {
        matches!(
            self,
            PairKind::Load32 | PairKind::Load32Sext | PairKind::Load64
        )
    }

    /// The distance between the two registers' addresses.
    #[inline(always)]
    fn stride(self) -> u32 {
        match self {
            PairKind::Load64 | PairKind::Store64 => 8,
            _ => 4,
        }
    }
}

/// One translated instruction: what it does, with its operands already pulled
/// out of the encoding.
#[derive(Debug, Clone, Copy)]
pub(super) enum Op {
    /// A hint, barrier or PSTATE-immediate write the interpreter also retires
    /// with no effect.
    Nop,
    /// Not translated: run the original instruction through the interpreter.
    Interpret { insn: u32 },
    /// SIMD and floating point, handed to the decoder that owns it instead of
    /// back through [`super::Cpu::execute`]'s group match. `scalar` is the
    /// same top-byte test `execute` makes to decide which of the two decoders
    /// gets first look, and `form` is which of the scalar forms it is — both
    /// decided once here rather than on every execution.
    Fp {
        insn: u32,
        scalar: bool,
        form: FpForm,
    },
    /// A system instruction [`decode_system`] could not place. Straight to
    /// [`super::Cpu::system`], which is where its error comes from.
    System { insn: u32 },
    /// `MRS Xt, <sysreg>`.
    Mrs { rd: u8, reg: SysReg },
    /// `MSR <sysreg>, Xt`.
    Msr { rt: u8, reg: SysReg },
    /// The one `MSR` immediate form that has an effect.
    MsrNzcvImm { imm: u8 },
    /// `DC ZVA Xt`: zero the 64-byte block Xt points into.
    DcZva { rt: u8 },

    /// A value the translator already computed: `MOVZ`/`MOVN`, and the
    /// PC-relative `ADR`/`ADRP` whose result depends only on where the
    /// instruction is.
    MovConst { rd: u8, val: u64 },
    /// `MOVK`: replace the 16-bit field at `shift` with `val`. Held as a
    /// shift and a halfword rather than a mask and a placed value so the
    /// variant needs one 64-bit word instead of two — which is what decides
    /// [`Op`]'s size, and so a block body's whole cache footprint.
    MovK { rd: u8, shift: u8, val: u16 },

    /// `ADD`/`SUB`/`ADDS`/`SUBS` against a constant.
    ///
    /// `rhs` arrives already inverted for the subtractions, with `carry` set
    /// to match, so which direction the operation runs in does not survive to
    /// run time. `rn_sp`/`rd_sp` are the two places register 31 means the
    /// stack pointer rather than the zero register, also decided here.
    AddSubImm {
        rd: u8,
        rn: u8,
        rhs: u64,
        carry: u8,
        set_flags: bool,
        sf: bool,
    },
    /// The shifted-register form, where both `Rd` and `Rn` are always the zero
    /// register. `carry` is 1 for the subtractions, and doubles as the mask
    /// that inverts the operand.
    AddSubShifted {
        rd: u8,
        rn: u8,
        rm: u8,
        st: u8,
        sa: u8,
        carry: u8,
        set_flags: bool,
        sf: bool,
    },
    /// The same with no shift at all — `add x0, x1, x2` — which is most of
    /// them, and skips [`shift_reg`] entirely.
    AddSubReg {
        rd: u8,
        rn: u8,
        rm: u8,
        carry: u8,
        set_flags: bool,
        sf: bool,
    },
    AddSubExtended {
        rd: u8,
        rn: u8,
        rm: u8,
        option: u8,
        shift: u8,
        carry: u8,
        set_flags: bool,
        sf: bool,
    },

    /// `AND`/`ORR`/`EOR`/`ANDS` with the bitmask immediate already decoded.
    LogicalImm {
        rd: u8,
        rn: u8,
        imm: u64,
        opc: u8,
        sf: bool,
    },
    LogicalShifted {
        rd: u8,
        rn: u8,
        rm: u8,
        st: u8,
        sa: u8,
        opc: u8,
        invert: bool,
        sf: bool,
    },
    /// The unshifted form, which covers every `mov xd, xm` and `mvn xd, xm`.
    LogicalReg {
        rd: u8,
        rn: u8,
        rm: u8,
        opc: u8,
        invert: bool,
        sf: bool,
    },

    /// `SBFM`/`BFM`/`UBFM` and the aliases built on them.
    Bitfield {
        rd: u8,
        rn: u8,
        opc: u8,
        immr: u8,
        imms: u8,
        sf: bool,
    },
    Extr {
        rd: u8,
        rn: u8,
        rm: u8,
        imm: u8,
        sf: bool,
    },

    /// `CSEL`/`CSINC`/`CSINV`/`CSNEG`.
    CondSel {
        rd: u8,
        rn: u8,
        rm: u8,
        cond: u8,
        else_inv: bool,
        else_inc: bool,
        sf: bool,
    },
    /// `CCMP`/`CCMN`, register and immediate forms.
    CondCmp {
        rn: u8,
        rm: u8,
        imm: u8,
        cond: u8,
        nzcv: u8,
        sub: bool,
        is_imm: bool,
        sf: bool,
    },

    /// `MADD`/`MSUB`.
    Madd {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
        sub: bool,
        sf: bool,
    },
    /// `SMADDL`/`SMSUBL`/`UMADDL`/`UMSUBL`: the 32x32 widening multiplies.
    MaddLong {
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
        sub: bool,
        signed: bool,
    },
    /// `SMULH`/`UMULH`.
    Mulh {
        rd: u8,
        rn: u8,
        rm: u8,
        signed: bool,
    },

    LoadStoreImm {
        rt: u8,
        rn: u8,
        acc: Acc,
        wb: Wb,
        offset: i64,
    },
    LoadStoreReg {
        rt: u8,
        rn: u8,
        rm: u8,
        ext: Ext,
        shift: u8,
        acc: Acc,
    },
    /// `LDP`/`STP`/`LDPSW`.
    Pair {
        rt: u8,
        rt2: u8,
        rn: u8,
        offset: i64,
        kind: PairKind,
        wb: Wb,
    },
    /// `LDR <t>, label`, with the literal's address already resolved.
    LoadLiteral { rt: u8, addr: u32, acc: Acc },
}

/// The instruction a block ends on: one that always moves the PC somewhere
/// other than the following instruction. The conditional branches, whose
/// not-taken path *is* the following instruction, are [`Exit`]s instead and do
/// not end a block. A block with no terminator ran into the block-length or
/// page limit and simply falls through.
#[derive(Debug, Clone, Copy)]
pub(super) enum Term {
    /// `B #imm`.
    B { target: u32 },
    /// `BL #imm`.
    Bl { target: u32, ret_pc: u32 },
    /// `BR Xn`.
    Br { rn: u8 },
    /// `BLR Xn`.
    Blr { rn: u8, ret_pc: u32 },
    /// `RET Xn`.
    Ret { rn: u8 },
    /// `SVC #imm`.
    Svc { imm: u16, next: u32 },
    /// A control instruction with no op of its own: the interpreter decodes it
    /// and sets the PC itself.
    Interpret { insn: u32, next: u32 },
    /// The instruction could not be read when the block was translated. Try
    /// again at run time, so the fault is raised against the state the guest
    /// is actually in.
    Fetch,
}

/// A conditional branch *inside* a block: control leaves at this instruction
/// if the condition holds, and otherwise carries straight on to the next one.
///
/// These are the only three A64 branches whose not-taken path is the following
/// instruction, which is what lets a block continue past them at all.
#[derive(Debug, Clone, Copy)]
pub(super) enum Exit {
    /// `B.cond`.
    Cond { cond: u8, target: u32 },
    /// `CBZ`/`CBNZ`.
    Cbz {
        rt: u8,
        sf: bool,
        nz: bool,
        target: u32,
    },
    /// `TBZ`/`TBNZ`.
    Tbz {
        rt: u8,
        bit: u8,
        nz: bool,
        target: u32,
    },
    /// A `CMP`/`CMN` against a constant, fused with the `B.cond` that reads
    /// its flags — the commonest pair in compiled code, and one that only
    /// became fusable when blocks started running through conditional
    /// branches. `rhs` and `carry` arrive as they do for any other
    /// subtraction; the destination was the zero register, so nothing but
    /// NZCV is written.
    CmpImm {
        rn: u8,
        rhs: u64,
        carry: u8,
        sf: bool,
        cond: u8,
        target: u32,
    },
    /// The same against a register.
    CmpReg {
        rn: u8,
        rm: u8,
        carry: u8,
        sf: bool,
        cond: u8,
        target: u32,
    },
}

impl Exit {
    /// How many instructions the exit covers — two once a compare has been
    /// folded into it.
    #[inline(always)]
    fn span(self) -> usize {
        match self {
            Exit::CmpImm { .. } | Exit::CmpReg { .. } => 2,
            _ => 1,
        }
    }
}

/// A run of instructions with a single entry point, translated once.
#[derive(Debug)]
pub(super) struct Block {
    /// Guest address of the first instruction.
    start: u32,
    /// One entry per instruction the block covers before its terminator, so
    /// `ops[i]` is the instruction at `start + 4 * i`. The slots that hold a
    /// conditional branch carry [`Op::Nop`] as filler — the branch itself is
    /// in `exits`, and keeping the indexing exact is worth one dead slot per
    /// exit.
    ops: Vec<Op>,
    /// The original instruction words, body then terminator, kept so a fault
    /// inside a block leaves the same run-up trail an interpreted one does.
    words: Vec<u32>,
    /// The conditional branches the block runs through, as
    /// `(index into ops, branch)`, in ascending order.
    exits: Vec<(u32, Exit)>,
    term: Option<Term>,
}

/// How many entries the direct-mapped lookup in front of the block map holds.
/// A hash of the entry address is a large share of what entering a short
/// block costs, and guest code is dense enough that indexing by the address
/// itself nearly always hits.
const LOOKUP_SLOTS: usize = 4096;

/// The translation cache.
#[derive(Debug)]
pub(super) struct Jit {
    /// The most recent block to land in each slot, or `None`. Only a hint:
    /// the entry address is checked against the block's own, and `blocks` is
    /// what actually owns the cache.
    lookup: Vec<Option<Rc<Block>>>,
    blocks: IdMap<u32, Rc<Block>>,
    /// Entry addresses translated out of each page, so a store to that page
    /// drops exactly the blocks that read it.
    by_page: IdMap<u32, Vec<u32>>,
    translated: u64,
    executed: u64,
    invalidated: u64,
    interpreted: u64,
}

/// What the translator has been doing, for host-side diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitStats {
    /// Blocks currently held in the cache.
    pub blocks: usize,
    /// Blocks translated since the cache was created.
    pub translated: u64,
    /// Blocks entered.
    pub executed: u64,
    /// Blocks dropped because the memory they came from was written.
    pub invalidated: u64,
    /// Instructions that reached the interpreter's dispatcher anyway, because
    /// the translator had no op for them.
    ///
    /// Against `Cpu::run`'s step count this is the share of a run the
    /// translator did not actually translate — the one number here that says
    /// where the next block of speed is, and the same number on any target.
    pub interpreted: u64,
}

/// Whether the block translator has a real op for `insn`, or hands it back to
/// the interpreter to decode again on every execution.
///
/// Exact, and the same answer on every target — which is why it is worth
/// asking. The alternative is to time two engines against each other and call
/// a class translated when they come out within noise of one another, and
/// noise on a desktop is not evidence about the browser.
///
/// PC-relative forms decode against an address; which one makes no difference
/// to whether the encoding is translated, so any aligned address answers.
pub fn translates(insn: u32) -> bool {
    const REPRESENTATIVE_PC: u32 = 0x0800_0000;
    !matches!(
        decode(insn, REPRESENTATIVE_PC),
        Decoded::Op(Op::Interpret { .. }) | Decoded::Term(Term::Interpret { .. })
    )
}

impl Default for Jit {
    fn default() -> Jit {
        Jit {
            lookup: vec![None; LOOKUP_SLOTS],
            blocks: IdMap::default(),
            by_page: IdMap::default(),
            translated: 0,
            executed: 0,
            invalidated: 0,
            interpreted: 0,
        }
    }
}

impl Jit {
    #[inline(always)]
    fn slot(pc: u32) -> usize {
        (pc >> 2) as usize & (LOOKUP_SLOTS - 1)
    }

    /// The block entered at `pc`, if it is already translated.
    #[inline(always)]
    fn get(&mut self, pc: u32) -> Option<Rc<Block>> {
        let slot = Self::slot(pc);
        if let Some(block) = &self.lookup[slot] {
            if block.start == pc {
                return Some(block.clone());
            }
        }
        let block = self.blocks.get(&pc)?.clone();
        self.lookup[slot] = Some(block.clone());
        Some(block)
    }

    fn insert(&mut self, block: Rc<Block>) {
        if self.blocks.len() >= MAX_BLOCKS {
            self.blocks.clear();
            self.by_page.clear();
            self.drop_lookup();
        }
        let page = block.start >> 12;
        self.by_page.entry(page).or_default().push(block.start);
        self.lookup[Self::slot(block.start)] = Some(block.clone());
        self.blocks.insert(block.start, block);
    }

    /// Forget every lookup hint. Called whenever a block is dropped: a hint
    /// outliving the block it points at would keep running stale code.
    fn drop_lookup(&mut self) {
        for slot in &mut self.lookup {
            *slot = None;
        }
    }

    fn invalidate(&mut self, pages: &[u32]) {
        let mut dropped = false;
        for &page in pages {
            if let Some(starts) = self.by_page.remove(&page) {
                for start in starts {
                    if self.blocks.remove(&start).is_some() {
                        self.invalidated += 1;
                        dropped = true;
                    }
                }
            }
        }
        if dropped {
            self.drop_lookup();
        }
    }

    fn clear(&mut self) {
        self.invalidated += self.blocks.len() as u64;
        self.blocks.clear();
        self.by_page.clear();
        self.drop_lookup();
    }

    fn stats(&self) -> JitStats {
        JitStats {
            blocks: self.blocks.len(),
            translated: self.translated,
            executed: self.executed,
            invalidated: self.invalidated,
            interpreted: self.interpreted,
        }
    }
}

/// One decoded instruction: part of a block's body, a conditional branch the
/// block can run through, or the terminator that ends it.
enum Decoded {
    Op(Op),
    Exit(Exit),
    Term(Term),
}

/// Translate the block starting at `start`.
///
/// Stops at the first instruction that can move the PC, at [`MAX_BLOCK_OPS`],
/// or at the end of the page — never past it, so one page's invalidation
/// covers a block completely.
fn translate(mem: &Memory, start: u32) -> Block {
    let page_room = (PAGE_SIZE - (start as usize & (PAGE_SIZE - 1))) / 4;
    let limit = MAX_BLOCK_OPS.min(page_room.max(1));
    let mut ops = Vec::with_capacity(limit);
    let mut words = Vec::with_capacity(limit);
    let mut exits = Vec::new();
    for i in 0..limit {
        let pc = start.wrapping_add(4 * i as u32);
        let insn = match mem.fetch(pc) {
            Ok(insn) => insn,
            Err(_) => {
                fuse_compares(&mut ops, &mut exits);
                return Block {
                    start,
                    ops,
                    words,
                    exits,
                    term: Some(Term::Fetch),
                };
            }
        };
        match decode(insn, pc) {
            Decoded::Term(term) => {
                words.push(insn);
                fuse_compares(&mut ops, &mut exits);
                return Block {
                    start,
                    ops,
                    words,
                    exits,
                    term: Some(term),
                };
            }
            // A conditional branch does not end the block: its not-taken path
            // is the next instruction, so translation carries on there and the
            // branch becomes an early exit. This is the whole reason blocks
            // are longer than the six or seven instructions a basic block runs
            // to — `b.cond` alone is 12% of hbmenu's frame.
            Decoded::Exit(exit) => {
                exits.push((i as u32, exit));
                ops.push(Op::Nop);
                words.push(insn);
            }
            Decoded::Op(op) => {
                ops.push(op);
                words.push(insn);
            }
        }
    }
    fuse_compares(&mut ops, &mut exits);
    Block {
        start,
        ops,
        words,
        exits,
        term: None,
    }
}

/// Fold every `CMP`/`CMN` that feeds the conditional branch immediately after
/// it into that branch.
///
/// The pair is the shape of every bounds check and loop condition compiled
/// code emits, and until blocks ran through conditional branches the two were
/// never in the same block to fold. Only a compare whose destination is the
/// zero register qualifies, so the fused op writes nothing but NZCV and the
/// rewrite cannot lose a result.
fn fuse_compares(ops: &mut [Op], exits: &mut [(u32, Exit)]) {
    for (at, exit) in exits.iter_mut() {
        let Exit::Cond { cond, target } = *exit else {
            continue;
        };
        if *at == 0 {
            continue;
        }
        let prev = (*at - 1) as usize;
        let fused = match ops[prev] {
            Op::AddSubImm {
                rd,
                rn,
                rhs,
                carry,
                set_flags: true,
                sf,
            } if rd == ZR_DISCARD as u8 => Exit::CmpImm {
                rn,
                rhs,
                carry,
                sf,
                cond,
                target,
            },
            Op::AddSubReg {
                rd,
                rn,
                rm,
                carry,
                set_flags: true,
                sf,
            } if rd == ZR_DISCARD as u8 => Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            },
            _ => continue,
        };
        // The compare's slot becomes filler and the exit moves onto it, so the
        // pair is one dispatch covering two instructions.
        ops[prev] = Op::Nop;
        *at -= 1;
        *exit = fused;
    }
}

/// Classify one instruction the way [`super::Cpu::execute`] does — by bits
/// 28:25, the architecture's own first decode table — and translate it.
fn decode(insn: u32, pc: u32) -> Decoded {
    match (insn >> 25) & 0xF {
        0x8 | 0x9 => Decoded::Op(decode_data_proc_imm(insn, pc)),
        0x5 | 0xD => Decoded::Op(decode_data_proc_reg(insn)),
        0x4 | 0x6 | 0xC | 0xE => Decoded::Op(decode_load_store(insn, pc)),
        // Advanced SIMD and scalar floating point. No ops of their own, but
        // which of the two decoders owns the encoding is decided here rather
        // than on every execution.
        0x7 | 0xF => Decoded::Op(Op::Fp {
            insn,
            scalar: matches!((insn >> 24) & 0xFF, 0x1E | 0x1F | 0x9E | 0x9F),
            form: Cpu::fp_form(insn),
        }),
        0xA | 0xB => decode_branch_or_system(insn, pc),
        // Reserved and SVE. The interpreter rejects them; let it say so.
        _ => Decoded::Op(Op::Interpret { insn }),
    }
}

/// The branch, exception-generation and system group. Everything here either
/// ends the block or is one of the two system forms that provably retire to
/// the following instruction.
fn decode_branch_or_system(insn: u32, pc: u32) -> Decoded {
    let next = pc.wrapping_add(4);
    match (insn >> 24) & 0xFF {
        // B.cond
        0x54 => Decoded::Exit(Exit::Cond {
            cond: (insn & 0xF) as u8,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2),
        }),
        // B #imm
        0x14..=0x17 => Decoded::Term(Term::B {
            target: branch_target(pc, sext_u64(insn & 0x3FF_FFFF, 26) << 2),
        }),
        // TBZ / TBNZ
        0x36 | 0x37 | 0xB6 | 0xB7 => Decoded::Exit(Exit::Tbz {
            rt: (insn & 0x1F) as u8,
            bit: ((((insn >> 31) & 1) << 5) | ((insn >> 19) & 0x1F)) as u8,
            nz: ((insn >> 24) & 1) == 1,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x3FFF, 14) << 2),
        }),
        // CBZ / CBNZ
        0x34 | 0x35 | 0xB4 | 0xB5 => Decoded::Exit(Exit::Cbz {
            rt: (insn & 0x1F) as u8,
            sf: ((insn >> 31) & 1) == 1,
            nz: ((insn >> 24) & 1) == 1,
            target: branch_target(pc, sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2),
        }),
        // BL #imm
        0x94..=0x97 => Decoded::Term(Term::Bl {
            target: branch_target(pc, sext_u64(insn & 0x3FF_FFFF, 26) << 2),
            ret_pc: next,
        }),
        // BR / BLR / RET
        0xD6 | 0xD7 => {
            let rn = ((insn >> 5) & 0x1F) as u8;
            if ((insn >> 16) & 0x1F) != 0x1F || ((insn >> 10) & 0x3F) != 0 {
                return Decoded::Term(Term::Interpret { insn, next });
            }
            match (insn >> 21) & 0xF {
                0b0000 => Decoded::Term(Term::Br { rn }),
                0b0001 => Decoded::Term(Term::Blr { rn, ret_pc: next }),
                0b0010 => Decoded::Term(Term::Ret { rn }),
                _ => Decoded::Term(Term::Interpret { insn, next }),
            }
        }
        // SVC, and the other exception-generating forms the interpreter faults on.
        0xD4 => {
            if ((insn >> 21) & 0b111) == 0 && (insn & 0x1F) == 0b00001 {
                Decoded::Term(Term::Svc {
                    imm: ((insn >> 5) & 0xFFFF) as u16,
                    next,
                })
            } else {
                Decoded::Term(Term::Interpret { insn, next })
            }
        }
        // MSR/MRS, barriers and hints. `system` retires all of them to the
        // next instruction, so they stay inside the block.
        0xD5 => {
            if (insn >> 16) & 0xFFFF == 0xD503 {
                Decoded::Op(Op::Nop)
            } else if ((insn >> 22) & 0x3FF) == 0b1101010100 {
                // The same guard `try_branch_or_system` applies before handing
                // an encoding to `system`. Anything else falls through to the
                // whole-space chain, so it stays an `Interpret`.
                Decoded::Op(decode_system(insn))
            } else {
                Decoded::Op(Op::Interpret { insn })
            }
        }
        _ => Decoded::Term(Term::Interpret { insn, next }),
    }
}

/// The system register `op0:op1:CRn:CRm:op2` names, read exactly as
/// [`super::Cpu::system`] reads it.
fn decode_sysreg(insn: u32) -> SysReg {
    let op0 = (insn >> 19) & 0b11;
    let op1 = (insn >> 16) & 0b111;
    let crn = (insn >> 12) & 0xF;
    let crm = (insn >> 8) & 0xF;
    let op2 = (insn >> 5) & 0b111;
    match (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2 {
        0b11_011_0100_0010_000 => SysReg::Nzcv,
        0b11_011_1101_0000_010 => SysReg::Tpidr,
        0b11_011_1101_0000_011 => SysReg::TpidrRo,
        0b11_011_0100_0100_000 => SysReg::Fpcr,
        0b11_011_0100_0100_001 => SysReg::Fpsr,
        // DCZID_EL0: BS=4, a 64-byte `DC ZVA` block.
        0b11_011_0000_0000_111 => SysReg::Fixed(4),
        // CTR_EL0: the Cortex-A57 value.
        0b11_011_0000_0000_001 => SysReg::Fixed(0x8444_C004),
        _ => SysReg::Fixed(0),
    }
}

/// The `MRS`/`MSR`/cache-maintenance group, resolved to the single thing the
/// instruction does.
///
/// [`super::Cpu::system`] re-derives six fields and walks a chain of guards to
/// reach the same answer every time it runs one, and `MRS TPIDRRO_EL0` — how
/// every `nnSdk` thread finds its own TLS — is near the end of that chain.
fn decode_system(insn: u32) -> Op {
    let l = (insn >> 21) & 1;
    let op0 = (insn >> 19) & 0b11;
    let op1 = (insn >> 16) & 0b111;
    let crn = (insn >> 12) & 0xF;
    let crm = (insn >> 8) & 0xF;
    let op2 = (insn >> 5) & 0b111;
    let rt = (insn & 0x1F) as u8;

    if l == 1 {
        return Op::Mrs {
            rd: zr_write(insn),
            reg: decode_sysreg(insn),
        };
    }
    if op0 == 0 {
        // MSR (immediate). Only the PSTATE write has an effect; DAIF, SPSel
        // and the rest retire with none.
        return match (op1, crn, crm, op2) {
            (0b010 | 0b011, 0b0100, 0b0010, 0b000) => Op::MsrNzcvImm {
                imm: ((insn >> 8) & 0xF) as u8,
            },
            _ => Op::Nop,
        };
    }
    if op0 == 1 && crn == 7 {
        if op1 == 3 && crm == 4 && op2 == 1 {
            return Op::DcZva { rt };
        }
        // The other cache maintenance operations: this memory is always
        // coherent, so there is nothing for them to do.
        return Op::Nop;
    }
    if op0 == 2 || op0 == 3 {
        return Op::Msr {
            rt,
            reg: decode_sysreg(insn),
        };
    }
    Op::System { insn }
}

/// The register-file slot a five-bit field names when 31 means `XZR` and the
/// field is *written*. Reads need no mapping — slot 31 holds zero and nothing
/// writes it — so only the destinations are resolved here.
#[inline]
fn zr_write(n: u32) -> u8 {
    let n = (n & 0x1F) as u8;
    if n == 31 {
        ZR_DISCARD as u8
    } else {
        n
    }
}

/// The slot a five-bit field names when 31 means `SP`, read or written.
#[inline]
fn sp_form(n: u32) -> u8 {
    let n = (n & 0x1F) as u8;
    if n == 31 {
        SP_SLOT as u8
    } else {
        n
    }
}

/// The slot a load or store's `Rt` field names, which depends on whether the
/// access reads it or writes it.
#[inline]
fn rt_slot(rt: u32, acc: Acc) -> u8 {
    if acc.writes_rt() {
        zr_write(rt)
    } else {
        (rt & 0x1F) as u8
    }
}

/// The operand an addition needs to compute a subtraction. `carry` is 1
/// exactly when the instruction subtracts, so it doubles as the mask that
/// inverts the operand — no branch, and nothing left to decide at run time.
#[inline(always)]
fn invert_if(v: u64, carry: u8) -> u64 {
    v ^ 0u64.wrapping_sub(u64::from(carry))
}

/// The target of a PC-relative branch. `imm` is already the sign-extended
/// byte displacement.
#[inline]
fn branch_target(pc: u32, imm: u64) -> u32 {
    (pc as i64).wrapping_add(imm as i64) as u32
}

fn decode_data_proc_imm(insn: u32, pc: u32) -> Op {
    let sf = (insn >> 31) & 1 == 1;
    match (insn >> 24) & 0x1F {
        // ADR / ADRP: the result depends only on where the instruction is, so
        // the translator computes it and the block just moves a constant.
        0b10000 => {
            let rd = zr_write(insn);
            let immhi = u64::from((insn >> 5) & 0x7_FFFF);
            let immlo = u64::from((insn >> 29) & 0b11);
            let imm = sext_u64((immhi << 2) | immlo, 21);
            let val = if (insn >> 31) & 1 == 1 {
                (u64::from(pc & !0xFFF)).wrapping_add(imm.wrapping_shl(12))
            } else {
                u64::from(pc).wrapping_add(imm)
            };
            Op::MovConst { rd, val }
        }
        // ADD/SUB immediate. Bit 23 is the ADDG/SUBG tag arithmetic the
        // interpreter rejects.
        0b10001 => {
            if ((insn >> 23) & 1) == 1 {
                return Op::Interpret { insn };
            }
            let op = (insn >> 29) & 0b11;
            let imm12 = u64::from((insn >> 10) & 0xFFF);
            let imm = if ((insn >> 22) & 1) == 1 {
                imm12 << 12
            } else {
                imm12
            };
            let set_flags = (op & 1) == 1;
            let sub = (op >> 1) == 1;
            Op::AddSubImm {
                // Both operands are the SP form here, and the destination is
                // only SP when the instruction does not set flags.
                rd: if set_flags {
                    zr_write(insn)
                } else {
                    sp_form(insn)
                },
                rn: sp_form(insn >> 5),
                // Subtraction is addition of the inverted operand with a carry
                // in, and both are known now.
                rhs: if sub { !imm } else { imm },
                carry: u8::from(sub),
                set_flags,
                sf,
            }
        }
        0b10010 => {
            if ((insn >> 23) & 1) == 1 {
                // MOVN / MOVZ / MOVK
                let rd = zr_write(insn);
                let imm16 = u64::from((insn >> 5) & 0xFFFF);
                let hw = if sf {
                    (insn >> 21) & 0b11
                } else {
                    (insn >> 21) & 1
                };
                let shift = hw * 16;
                match (insn >> 29) & 0b11 {
                    0b00 => Op::MovConst {
                        rd,
                        val: !(imm16 << shift) & Cpu::mask(sf),
                    },
                    0b10 => Op::MovConst {
                        rd,
                        val: (imm16 << shift) & Cpu::mask(sf),
                    },
                    // A 32-bit MOVK never selects a field above bit 31, so
                    // the operation-size mask cannot narrow it.
                    0b11 => Op::MovK {
                        rd,
                        shift: shift as u8,
                        val: imm16 as u16,
                    },
                    _ => Op::Interpret { insn },
                }
            } else {
                // Logical immediate: the bitmask decodes to a constant.
                let immr = (insn >> 16) & 0x3F;
                let imms = (insn >> 10) & 0x3F;
                match decode_bit_mask(sf, (insn >> 22) & 1, immr, imms) {
                    Some(imm) => {
                        let opc = ((insn >> 29) & 0b11) as u8;
                        Op::LogicalImm {
                            // `ANDS` is the only one of the four whose Rd is
                            // the zero register rather than SP.
                            rd: if opc == 0b11 {
                                zr_write(insn)
                            } else {
                                sp_form(insn)
                            },
                            rn: ((insn >> 5) & 0x1F) as u8,
                            imm,
                            opc,
                            sf,
                        }
                    }
                    None => Op::Interpret { insn },
                }
            }
        }
        0b10011 => {
            let rd = zr_write(insn);
            let rn = ((insn >> 5) & 0x1F) as u8;
            if ((insn >> 23) & 1) == 0 {
                // Bitfield move. The unallocated encodings go to the
                // interpreter so it raises the same error.
                let (immr, imms) = if sf {
                    if ((insn >> 22) & 1) != 1 {
                        return Op::Interpret { insn };
                    }
                    ((insn >> 16) & 0x3F, (insn >> 10) & 0x3F)
                } else {
                    if ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1 {
                        return Op::Interpret { insn };
                    }
                    ((insn >> 16) & 0x1F, (insn >> 10) & 0x1F)
                };
                Op::Bitfield {
                    rd,
                    rn,
                    opc: ((insn >> 29) & 0b11) as u8,
                    immr: immr as u8,
                    imms: imms as u8,
                    sf,
                }
            } else {
                // EXTR
                let imm = if sf {
                    if ((insn >> 22) & 1) != 1 || ((insn >> 21) & 1) == 1 {
                        return Op::Interpret { insn };
                    }
                    (insn >> 10) & 0x3F
                } else {
                    if ((insn >> 22) & 1) == 1 || ((insn >> 21) & 1) == 1 || ((insn >> 15) & 1) == 1
                    {
                        return Op::Interpret { insn };
                    }
                    (insn >> 10) & 0x1F
                };
                Op::Extr {
                    rd,
                    rn,
                    rm: ((insn >> 16) & 0x1F) as u8,
                    imm: imm as u8,
                    sf,
                }
            }
        }
        _ => Op::Interpret { insn },
    }
}

fn decode_data_proc_reg(insn: u32) -> Op {
    let sf = (insn >> 31) & 1 == 1;
    // Every form in this group reads Rn and Rm as the zero register, and all
    // but the extended ADD/SUB write Rd as one too.
    let rd = zr_write(insn);
    let rn = ((insn >> 5) & 0x1F) as u8;
    let rm = ((insn >> 16) & 0x1F) as u8;
    match (insn >> 24) & 0x1F {
        // Logical shifted register.
        0b01010 => {
            let st = ((insn >> 22) & 0b11) as u8;
            let sa = ((insn >> 10) & 0x3F) as u8;
            let opc = ((insn >> 29) & 0b11) as u8;
            let invert = ((insn >> 21) & 1) == 1;
            if sa == 0 {
                // `mov xd, xm` and `mvn xd, xm` are both this.
                Op::LogicalReg {
                    rd,
                    rn,
                    rm,
                    opc,
                    invert,
                    sf,
                }
            } else {
                Op::LogicalShifted {
                    rd,
                    rn,
                    rm,
                    st,
                    sa,
                    opc,
                    invert,
                    sf,
                }
            }
        }
        // ADD/SUB, shifted or extended register.
        0b01011 => {
            let op = (insn >> 29) & 0b11;
            let set_flags = (op & 1) == 1;
            let sub = (op >> 1) == 1;
            let carry = u8::from(sub);
            if ((insn >> 21) & 0b111) == 0b001 {
                Op::AddSubExtended {
                    // The extended form is the other place register 31 is SP.
                    rd: if set_flags { rd } else { sp_form(insn) },
                    rn: sp_form(insn >> 5),
                    rm,
                    option: ((insn >> 13) & 0b111) as u8,
                    shift: ((insn >> 10) & 0b111) as u8,
                    carry,
                    set_flags,
                    sf,
                }
            } else {
                let st = ((insn >> 22) & 0b11) as u8;
                let sa = ((insn >> 10) & 0x3F) as u8;
                if sa == 0 {
                    // Every shift kind is the identity at zero, and this is
                    // the common form.
                    Op::AddSubReg {
                        rd,
                        rn,
                        rm,
                        carry,
                        set_flags,
                        sf,
                    }
                } else {
                    Op::AddSubShifted {
                        rd,
                        rn,
                        rm,
                        st,
                        sa,
                        carry,
                        set_flags,
                        sf,
                    }
                }
            }
        }
        0b11010 => match (((insn >> 23) & 1), ((insn >> 22) & 1)) {
            // Conditional compare.
            (0, 1) => Op::CondCmp {
                rn,
                rm,
                imm: ((insn >> 16) & 0x1F) as u8,
                cond: ((insn >> 12) & 0xF) as u8,
                nzcv: (insn & 0xF) as u8,
                sub: ((insn >> 30) & 1) == 1,
                is_imm: ((insn >> 11) & 1) == 1,
                sf,
            },
            // Conditional select.
            (1, 0) => Op::CondSel {
                rd,
                rn,
                rm,
                cond: ((insn >> 12) & 0xF) as u8,
                else_inv: ((insn >> 30) & 1) == 1,
                else_inc: ((insn >> 10) & 1) == 1,
                sf,
            },
            // The one- and two-source group (divides, variable shifts, CRC32,
            // bit counts) and ADC/SBC: left to the interpreter.
            _ => Op::Interpret { insn },
        },
        // Three-source: the multiplies.
        0b11011 => {
            let ra = ((insn >> 10) & 0x1F) as u8;
            let sub = ((insn >> 15) & 1) == 1;
            // Ra is read, Rd written; `rd` above already resolved.

            match (insn >> 21) & 0xFF {
                0b11011000 => Op::Madd {
                    rd,
                    rn,
                    rm,
                    ra,
                    sub,
                    sf,
                },
                0b11011001 => Op::MaddLong {
                    rd,
                    rn,
                    rm,
                    ra,
                    sub,
                    signed: true,
                },
                0b11011101 => Op::MaddLong {
                    rd,
                    rn,
                    rm,
                    ra,
                    sub,
                    signed: false,
                },
                0b11011010 => Op::Mulh {
                    rd,
                    rn,
                    rm,
                    signed: true,
                },
                0b11011110 => Op::Mulh {
                    rd,
                    rn,
                    rm,
                    signed: false,
                },
                _ => Op::Interpret { insn },
            }
        }
        _ => Op::Interpret { insn },
    }
}

/// The load/store group, in the order [`super::Cpu::execute`] tries it: the
/// literal forms first, then everything [`super::Cpu::try_load_store`] claims.
fn decode_load_store(insn: u32, pc: u32) -> Op {
    // LDR <t>, label — the address is fixed by where the instruction is.
    if ((insn >> 27) & 0b111) == 0b011 && ((insn >> 26) & 1) == 0 && ((insn >> 24) & 0b11) == 0b00 {
        let imm = sext_u64((insn >> 5) & 0x7_FFFF, 19) << 2;
        // The literal forms encode the width in `opc` alone; `Acc::of` reads
        // it as a size:opc pair, so pass the size the width implies.
        let acc = match (insn >> 30) & 0b11 {
            0b00 => Acc::Load32,
            0b01 => Acc::Load64,
            0b10 => Acc::LoadS32,
            _ => Acc::Prefetch,
        };
        return Op::LoadLiteral {
            rt: rt_slot(insn, acc),
            addr: (pc as i64).wrapping_add(imm as i64) as u32,
            acc,
        };
    }

    // The exclusive accessors touch the local monitor, and the V=1 forms are
    // SIMD: both stay with the interpreter.
    let grp_excl = (insn >> 21) & 0x1FF;
    if (0b001000000..=0b001000011).contains(&grp_excl)
        || grp_excl == 0b001000100
        || grp_excl == 0b001000110
    {
        return Op::Interpret { insn };
    }
    if ((insn >> 26) & 1) == 1 {
        return Op::Interpret { insn };
    }

    let sz = ((insn >> 30) & 0b11) as u8;
    let opc = ((insn >> 22) & 0b11) as u8;
    let acc = Acc::of(sz, opc);
    // Rt is read by a store and written by a load, and Rn is always the SP
    // form — the base of an addressing mode is never the zero register.
    let rt = rt_slot(insn, acc);
    let rn = sp_form(insn >> 5);

    // Register offset.
    if ((insn >> 27) & 0b111) == 0b111 && ((insn >> 24) & 0b11) == 0b00 && ((insn >> 21) & 1) == 1 {
        // Only four of the eight extend encodings are defined here; the rest
        // are undefined, and the interpreter faults on them.
        let ext = match (insn >> 13) & 0b111 {
            0b010 => Ext::Uxtw,
            0b110 => Ext::Sxtw,
            0b011 | 0b111 => Ext::None,
            _ => return Op::Interpret { insn },
        };
        // `S` scales by log2 of the access size, not by the byte count.
        let shift = if ((insn >> 12) & 1) == 1 { sz } else { 0 };
        return Op::LoadStoreReg {
            rt,
            rn,
            rm: ((insn >> 16) & 0x1F) as u8,
            ext,
            shift,
            acc,
        };
    }

    // Immediate offset forms.
    if ((insn >> 27) & 0b111) == 0b111 {
        let mode = (insn >> 24) & 0b11;
        if mode == 0b01 {
            // Unsigned offset, scaled by the access size.
            let scale = 1i64 << sz;
            return Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb: Wb::None,
                offset: i64::from((insn >> 10) & 0xFFF) * scale,
            };
        }
        if mode == 0b00 && ((insn >> 21) & 1) == 0 {
            let offset = sext_u64((insn >> 12) & 0x1FF, 9) as i64;
            let wb = match (insn >> 10) & 0b11 {
                0b01 => Wb::Post,
                0b11 => Wb::Pre,
                // Unscaled (`LDUR`/`STUR`) and the unprivileged forms.
                _ => Wb::None,
            };
            return Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            };
        }
    }

    // Load/store pair.
    if ((insn >> 27) & 0b111) == 0b101 && ((insn >> 25) & 1) == 0 {
        let pair_opc = (insn >> 30) & 0b11;
        let load = ((insn >> 22) & 1) == 1;
        // The tagged store-pair and the reserved addressing mode are errors
        // the interpreter reports.
        if (pair_opc == 0b01 && !load) || pair_opc == 0b11 {
            return Op::Interpret { insn };
        }
        let wide = pair_opc == 0b10;
        let scale: i64 = if wide { 8 } else { 4 };
        let wb = match (insn >> 23) & 0b11 {
            0b01 => Wb::Post,
            0b11 => Wb::Pre,
            _ => Wb::None,
        };
        let kind = match (load, wide, pair_opc == 0b01) {
            (true, _, true) => PairKind::Load32Sext,
            (true, true, _) => PairKind::Load64,
            (true, false, _) => PairKind::Load32,
            (false, true, _) => PairKind::Store64,
            (false, false, _) => PairKind::Store32,
        };
        // The pair forms have their own Rt/Rt2 slots: `kind`, not `acc`,
        // says whether they are read or written.
        let pair_slot = |n: u32| {
            if kind.loads() {
                zr_write(n)
            } else {
                (n & 0x1F) as u8
            }
        };
        return Op::Pair {
            rt: pair_slot(insn),
            rt2: pair_slot(insn >> 10),
            rn,
            offset: (sext_u64((insn >> 15) & 0x7F, 7) as i64).wrapping_mul(scale),
            kind,
            wb,
        };
    }

    Op::Interpret { insn }
}

impl Cpu {
    /// Whether [`Cpu::run`] goes through the translator.
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled
    }

    /// Turn the translator on or off. Turning it off drops the cache, so a
    /// caller that switches back gets a translator with no stale state.
    pub fn set_jit_enabled(&mut self, on: bool) {
        if !on {
            self.jit.clear();
        }
        self.jit_enabled = on;
    }

    /// What the translator has been doing since the CPU was created.
    pub fn jit_stats(&self) -> JitStats {
        self.jit.stats()
    }

    /// Drop every translated block. Callers that rewrite guest code behind
    /// [`crate::mem::Memory`]'s back need this; ordinary guest stores are
    /// noticed on their own.
    pub fn jit_flush(&mut self) {
        self.jit.clear();
    }

    /// [`Cpu::run`] over translated blocks.
    ///
    /// The step budget is honoured exactly. A block whose remaining
    /// instructions do not fit is entered anyway and left part-way through,
    /// with `pc` on the instruction that would have come next — the block is
    /// a cache, not a unit of execution, so stopping inside one is no
    /// different from stopping between two interpreted instructions.
    pub(super) fn run_jit(&mut self, max_steps: u64) -> Result<RunReport> {
        let mut steps = 0u64;
        // The block last executed, kept across iterations. A loop that branches
        // back to its own head — which is most of what a hot loop is — then
        // costs neither the cache lookup nor the reference count, because the
        // handle is moved out and back rather than cloned.
        let mut held: Option<Rc<Block>> = None;
        while steps < max_steps && !self.halted {
            // The scheduler's preemption point. The interpreter takes it
            // between any two instructions; here it is between blocks, which
            // is the same thing at the scale of a 20,000-instruction slice.
            if self.slice_used >= TIME_SLICE {
                self.slice_used = 0;
                self.yield_thread();
            }
            self.sweep_timed_waits();
            let block = match held.take() {
                Some(block) if block.start == self.pc && !self.mem.has_dirty_code() => {
                    self.jit.executed += 1;
                    block
                }
                _ => self.jit_block_at(self.pc),
            };
            let ran = self.exec_block(&block, max_steps - steps)?;
            held = Some(block);
            self.slice_used += ran;
            steps += ran;
        }
        Ok(RunReport {
            steps,
            halted: self.halted,
        })
    }

    /// The block entered at `pc`, translating it if this is the first visit.
    ///
    /// Drains the pages guest stores have landed on first, so a block is never
    /// handed out after the instructions behind it have changed.
    fn jit_block_at(&mut self, pc: u32) -> Rc<Block> {
        if self.mem.has_dirty_code() {
            let dirty = self.mem.dirty_code_pages();
            self.jit.invalidate(&dirty);
        }
        self.jit.executed += 1;
        if let Some(block) = self.jit.get(pc) {
            return block;
        }
        let block = Rc::new(translate(&self.mem, pc));
        self.mem.mark_code_page(block.start);
        self.jit.translated += 1;
        self.jit.insert(block.clone());
        block
    }

    /// Run at most `budget` of a block's instructions, reporting how many
    /// retired.
    ///
    /// `self.pc` tracks the instruction being executed throughout, exactly as
    /// it does in the interpreter, so a fault inside a block reports the same
    /// address and the same register state an interpreted one would.
    fn exec_block(&mut self, block: &Block, budget: u64) -> Result<u64> {
        let body = (block.ops.len() as u64).min(budget) as usize;
        let mut i = 0usize;
        let mut pc = block.start;
        let mut next_exit = 0usize;
        loop {
            // Run straight through to the next conditional branch, or to the
            // end of what the budget allows. Taking the segment as a slice
            // keeps this the same bounds-check-free walk it was when a block
            // had no interior exits at all.
            let stop = match block.exits.get(next_exit) {
                Some(&(at, _)) if (at as usize) < body => at as usize,
                _ => body,
            };
            for (k, &op) in block.ops[i..stop].iter().enumerate() {
                if let Err(e) = self.exec_op(op, pc) {
                    // The clock, the step counter, the trail and `pc` are all
                    // settled here rather than maintained per instruction:
                    // nothing inside a block reads any of them, and a fault is
                    // the only thing that ever does. The faulting instruction
                    // counts, exactly as it does in the interpreter.
                    let at = i + k;
                    self.retire_run(block.start, at as u64 + 1);
                    self.pc = pc;
                    self.record_fault(&e, pc, block.words[at]);
                    return Err(e);
                }
                pc = pc.wrapping_add(4);
            }
            i = stop;
            if stop == body {
                break;
            }
            let (_, exit) = block.exits[next_exit];
            let span = exit.span();
            if i + span > body {
                // The budget splits a fused pair. Run the compare's half of it
                // and stop on the branch, which is a valid entry point with the
                // flags already set. Doing nothing here instead would return no
                // progress at all when the pair starts the block, and
                // [`Cpu::run_jit`] would spin on it forever.
                if i < body {
                    self.apply_compare(exit);
                    i += 1;
                    pc = pc.wrapping_add(4);
                }
                break;
            }
            i += span;
            pc = pc.wrapping_add(4 * span as u32);
            if self.take_exit(exit) {
                // Taken: the branch is the last instruction of this visit, and
                // `take_exit` has already put the target in `pc`.
                self.retire_run(block.start, i as u64);
                return Ok(i as u64);
            }
            next_exit += 1;
        }
        self.retire_run(block.start, i as u64);
        let mut ran = i as u64;
        match block.term {
            Some(term) if i == block.ops.len() && ran < budget => {
                self.pc = pc;
                self.record_run(pc, 1);
                let result = self.exec_term(term, pc);
                // After the terminator, not before, and whether or not it
                // faulted — which is what `step_inner` does. An `SVC` is the
                // one instruction that reads the clock while it runs, so
                // retiring it early made the JIT hand every syscall a tick the
                // interpreter had not spent yet: sdl-hello ended 1 cycle apart
                // between the two engines, because a sleep deadline is
                // computed from the value the syscall saw.
                self.retire();
                if let Err(e) = result {
                    self.record_fault(&e, pc, block.words[i]);
                    return Err(e);
                }
                ran += 1;
            }
            // Either the budget ran out inside the block, or it covered the
            // body of a block that has no terminator. Both leave `pc` on the
            // next instruction to run.
            _ => self.pc = pc,
        }
        Ok(ran)
    }

    /// The flag-setting half of a fused compare-and-branch, for the one case
    /// that cannot run both: a step budget that ends between them.
    #[inline(always)]
    fn apply_compare(&mut self, exit: Exit) {
        let (a, b, carry, sf) = match exit {
            Exit::CmpImm {
                rn, rhs, carry, sf, ..
            } => (self.reg_at(rn) & Cpu::mask(sf), rhs, carry, sf),
            Exit::CmpReg {
                rn, rm, carry, sf, ..
            } => (
                self.reg_at(rn) & Cpu::mask(sf),
                invert_if(self.reg_at(rm) & Cpu::mask(sf), carry),
                carry,
                sf,
            ),
            // Nothing else spans two instructions.
            _ => return,
        };
        let (result, c, v) = Cpu::add_carry_overflow(a, b, u64::from(carry), sf);
        self.set_nzcv_from_alu(result, sf, c, v);
    }

    /// Evaluate a conditional branch inside a block. Returns whether it was
    /// taken — in which case `pc` is where control went and the block is over,
    /// and otherwise the block carries on at the following instruction with
    /// `pc` still to be settled by the caller.
    #[inline(always)]
    fn take_exit(&mut self, exit: Exit) -> bool {
        let (taken, target) = match exit {
            Exit::Cond { cond, target } => (self.condition_holds(cond), target),
            Exit::CmpImm {
                rn,
                rhs,
                carry,
                sf,
                cond,
                target,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let (result, c, v) = Cpu::add_carry_overflow(a, rhs, u64::from(carry), sf);
                self.set_nzcv_from_alu(result, sf, c, v);
                (self.condition_holds(cond), target)
            }
            Exit::CmpReg {
                rn,
                rm,
                carry,
                sf,
                cond,
                target,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = invert_if(self.reg_at(rm) & Cpu::mask(sf), carry);
                let (result, c, v) = Cpu::add_carry_overflow(a, b, u64::from(carry), sf);
                self.set_nzcv_from_alu(result, sf, c, v);
                (self.condition_holds(cond), target)
            }
            Exit::Cbz { rt, sf, nz, target } => {
                let val = self.read_zr(rt);
                let is_zero = if sf { val == 0 } else { (val as u32) == 0 };
                (is_zero == !nz, target)
            }
            Exit::Tbz {
                rt,
                bit,
                nz,
                target,
            } => {
                let set = (self.read_zr(rt) >> bit) & 1 == 1;
                (set == nz, target)
            }
        };
        if taken {
            self.pc = target;
        }
        taken
    }

    /// Execute one body op. Every arm does what the interpreter's decoder
    /// would have done once it finished decoding.
    #[inline(always)]
    fn exec_op(&mut self, op: Op, pc: u32) -> Result<()> {
        match op {
            Op::Nop => {}
            // The three arms that re-enter the interpreter are the only ones
            // that need `pc` in the register file: `execute` resolves
            // PC-relative forms from it, and every fault message names it.
            Op::Interpret { insn } => {
                self.pc = pc;
                self.jit.interpreted += 1;
                self.execute(insn, pc.wrapping_add(4))?;
            }
            Op::Fp { insn, scalar, form } => {
                self.pc = pc;
                let next = pc.wrapping_add(4);
                let claimed = if scalar {
                    self.run_fp(form, insn)? || self.try_simd(insn)?
                } else {
                    self.try_simd(insn)? || self.run_fp(form, insn)?
                };
                if claimed {
                    self.pc = next;
                } else {
                    self.execute_chain(insn, next)?;
                }
            }
            Op::System { insn } => {
                self.pc = pc;
                self.system(insn, pc.wrapping_add(4))?;
            }
            Op::Mrs { rd, reg } => {
                let val = match reg {
                    SysReg::Nzcv => u64::from(self.nzcv),
                    SysReg::Tpidr => self.tpidr_rw,
                    SysReg::TpidrRo => self.tpidr,
                    SysReg::Fpcr => u64::from(self.fpcr),
                    SysReg::Fpsr => u64::from(self.fpsr),
                    SysReg::Fixed(v) => u64::from(v),
                };
                self.set_reg_at(rd, val);
            }
            Op::Msr { rt, reg } => match reg {
                SysReg::Nzcv => self.nzcv = self.reg_at(rt) as u32,
                // Only the bits the architecture defines stick, so a guest
                // that reads back what it wrote sees the same value.
                SysReg::Fpcr => self.fpcr = self.reg_at(rt) as u32 & FPCR_MASK,
                SysReg::Fpsr => self.fpsr = self.reg_at(rt) as u32 & FPSR_MASK,
                SysReg::Tpidr => self.tpidr_rw = self.reg_at(rt),
                SysReg::TpidrRo | SysReg::Fixed(_) => {}
            },
            Op::MsrNzcvImm { imm } => self.nzcv = u32::from(imm),
            Op::DcZva { rt } => {
                let addr = self.reg_at(rt) as u32 & !0x3F;
                for i in 0..8u32 {
                    self.mem.write_u64(addr.wrapping_add(i * 8), 0)?;
                }
            }

            Op::MovConst { rd, val } => self.set_reg_at(rd, val),
            Op::MovK { rd, shift, val } => {
                // Read and write through the same slot: a `MOVK` to `XZR`
                // reads the discard slot and writes it back, which is as
                // unobservable as discarding the write was.
                let mask = 0xFFFFu64 << shift;
                let cur = self.reg_at(rd) & !mask;
                self.set_reg_at(rd, cur | (u64::from(val) << shift));
            }

            Op::AddSubImm {
                rd,
                rn,
                rhs,
                carry,
                set_flags,
                sf,
            } => {
                self.add_sub_pre(rd, rn, rhs, carry, set_flags, sf);
            }
            Op::AddSubReg {
                rd,
                rn,
                rm,
                carry,
                set_flags,
                sf,
            } => {
                let v = self.reg_at(rm) & Cpu::mask(sf);
                self.add_sub_pre(rd, rn, invert_if(v, carry), carry, set_flags, sf);
            }
            Op::AddSubShifted {
                rd,
                rn,
                rm,
                st,
                sa,
                carry,
                set_flags,
                sf,
            } => {
                let v = shift_reg(
                    self.reg_at(rm) & Cpu::mask(sf),
                    u32::from(st),
                    u32::from(sa),
                    sf,
                );
                self.add_sub_pre(rd, rn, invert_if(v, carry), carry, set_flags, sf);
            }
            Op::AddSubExtended {
                rd,
                rn,
                rm,
                option,
                shift,
                carry,
                set_flags,
                sf,
            } => {
                let v = extend_reg(self.reg_at(rm), option, sf) & Cpu::mask(sf);
                let v = v.wrapping_shl(u32::from(shift)) & Cpu::mask(sf);
                self.add_sub_pre(rd, rn, invert_if(v, carry), carry, set_flags, sf);
            }

            Op::LogicalImm {
                rd,
                rn,
                imm,
                opc,
                sf,
            } => {
                // Rd == 31 is SP for AND/ORR/EOR and the zero register only
                // for ANDS — one of the few places the two differ, and one
                // the translator has already settled into `rd`.
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let r = self.logical(a, imm, opc, sf);
                self.set_reg_at(rd, r);
            }
            Op::LogicalShifted {
                rd,
                rn,
                rm,
                st,
                sa,
                opc,
                invert,
                sf,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = shift_reg(
                    self.reg_at(rm) & Cpu::mask(sf),
                    u32::from(st),
                    u32::from(sa),
                    sf,
                );
                // `BIC`/`ORN`/`EON` invert the *shifted* operand, not the
                // register: `ir.Not(ShiftReg(...))` in dynarmic, and the same
                // order in the ARM ARM's pseudocode.
                let b = if invert { !b & Cpu::mask(sf) } else { b };
                let r = self.logical(a, b, opc, sf);
                self.set_reg_at(rd, r);
            }
            Op::LogicalReg {
                rd,
                rn,
                rm,
                opc,
                invert,
                sf,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = self.reg_at(rm) & Cpu::mask(sf);
                let b = if invert { !b & Cpu::mask(sf) } else { b };
                let r = self.logical(a, b, opc, sf);
                self.set_reg_at(rd, r);
            }

            Op::Bitfield {
                rd,
                rn,
                opc,
                immr,
                imms,
                sf,
            } => {
                let val = self.reg_at(rn) & Cpu::mask(sf);
                let cur = self.reg_at(rd) & Cpu::mask(sf);
                let r = bitfield_apply(
                    u32::from(opc),
                    val,
                    cur,
                    u32::from(immr),
                    u32::from(imms),
                    sf,
                );
                self.set_reg_at(rd, r);
            }
            Op::Extr {
                rd,
                rn,
                rm,
                imm,
                sf,
            } => {
                let size = if sf { 64u32 } else { 32 };
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = self.reg_at(rm) & Cpu::mask(sf);
                let imm = u32::from(imm);
                // Rn is the high half of the Rn:Rm pair the field is taken from.
                let r = if imm == 0 {
                    b
                } else {
                    ((b >> imm) | a.wrapping_shl(size.wrapping_sub(imm))) & Cpu::mask(sf)
                };
                self.set_reg_at(rd, r);
            }

            Op::CondSel {
                rd,
                rn,
                rm,
                cond,
                else_inv,
                else_inc,
                sf,
            } => {
                let a = self.reg_at(rn) & Cpu::mask(sf);
                let b = self.reg_at(rm) & Cpu::mask(sf);
                let take_a = self.condition_holds(cond);
                let mut else_val = b;
                if else_inv {
                    else_val = !else_val;
                }
                if else_inc {
                    else_val = else_val.wrapping_add(1);
                }
                let r = if take_a { a } else { else_val };
                self.set_reg_at(rd, r & Cpu::mask(sf));
            }
            Op::CondCmp {
                rn,
                rm,
                imm,
                cond,
                nzcv,
                sub,
                is_imm,
                sf,
            } => {
                if self.condition_holds(cond) {
                    let a = self.read_zr(rn) & Cpu::mask(sf);
                    let b = if is_imm {
                        u64::from(imm)
                    } else {
                        self.read_zr(rm)
                    };
                    self.set_nzcv_from_compare(a, b, sub, u64::from(sub), sf);
                } else {
                    self.nzcv = u32::from(nzcv) << 28;
                }
            }

            Op::Madd {
                rd,
                rn,
                rm,
                ra,
                sub,
                sf,
            } => {
                let mask = Cpu::mask(sf);
                let product = (self.reg_at(rn) & mask).wrapping_mul(self.reg_at(rm) & mask);
                let c = self.reg_at(ra) & mask;
                let r = if sub {
                    c.wrapping_sub(product)
                } else {
                    c.wrapping_add(product)
                };
                self.set_reg_at(rd, r & mask);
            }
            Op::MaddLong {
                rd,
                rn,
                rm,
                ra,
                sub,
                signed,
            } => {
                let a = self.reg_at(rn);
                let b = self.reg_at(rm);
                // The multiplicands are the low 32 bits of Rn/Rm, not the
                // whole register.
                // A 32x32 product fits in 64 bits, so this does not need the
                // 128-bit arithmetic wasm has to synthesize.
                let product = if signed {
                    i64::from(a as u32 as i32).wrapping_mul(i64::from(b as u32 as i32)) as u64
                } else {
                    u64::from(a as u32).wrapping_mul(u64::from(b as u32))
                };
                let c = self.reg_at(ra);
                let r = if sub {
                    c.wrapping_sub(product)
                } else {
                    c.wrapping_add(product)
                };
                self.set_reg_at(rd, r);
            }
            Op::Mulh { rd, rn, rm, signed } => {
                let a = self.reg_at(rn);
                let b = self.reg_at(rm);
                let r = if signed {
                    (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64
                } else {
                    ((u128::from(a) * u128::from(b)) >> 64) as u64
                };
                self.set_reg_at(rd, r);
            }

            Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            } => {
                let base = self.reg_at(rn);
                let (addr, wb_val) = Self::indexed(base, offset, wb);
                self.access(addr as u32, rt, acc)?;
                if let Some(v) = wb_val {
                    self.set_reg_at(rn, v);
                }
            }
            Op::LoadStoreReg {
                rt,
                rn,
                rm,
                ext,
                shift,
                acc,
            } => {
                let index = match ext {
                    Ext::Uxtw => u64::from(self.reg_at(rm) as u32),
                    Ext::Sxtw => sext_u64(self.reg_at(rm), 32),
                    Ext::None => self.reg_at(rm),
                };
                let offset = index.wrapping_shl(u32::from(shift)) as i64;
                let addr = (self.reg_at(rn) as i64).wrapping_add(offset) as u32;
                self.access(addr, rt, acc)?;
            }
            Op::Pair {
                rt,
                rt2,
                rn,
                offset,
                kind,
                wb,
            } => {
                let base = self.reg_at(rn);
                let (addr, wb_val) = Self::indexed(base, offset, wb);
                let addr = addr as u32;
                let second = addr.wrapping_add(kind.stride());
                match kind {
                    // Both halves are read before either register is written,
                    // so `ldp x0, x1, [x0]` sees the memory it was pointed at.
                    PairKind::Load64 => {
                        let v0 = self.mem.read_u64(addr)?;
                        let v1 = self.mem.read_u64(second)?;
                        self.set_reg_at(rt, v0);
                        self.set_reg_at(rt2, v1);
                    }
                    PairKind::Load32 => {
                        let v0 = u64::from(self.mem.read_u32(addr)?);
                        let v1 = u64::from(self.mem.read_u32(second)?);
                        self.set_reg_at(rt, v0);
                        self.set_reg_at(rt2, v1);
                    }
                    PairKind::Load32Sext => {
                        let v0 = u64::from(self.mem.read_u32(addr)?);
                        let v1 = u64::from(self.mem.read_u32(second)?);
                        self.set_reg_at(rt, sext_u64(v0, 32));
                        self.set_reg_at(rt2, sext_u64(v1, 32));
                    }
                    PairKind::Store64 => {
                        self.mem.write_u64(addr, self.reg_at(rt))?;
                        self.mem.write_u64(second, self.reg_at(rt2))?;
                    }
                    PairKind::Store32 => {
                        self.mem.write_u32(addr, self.reg_at(rt) as u32)?;
                        self.mem.write_u32(second, self.reg_at(rt2) as u32)?;
                    }
                }
                if let Some(v) = wb_val {
                    self.set_reg_at(rn, v);
                }
            }
            Op::LoadLiteral { rt, addr, acc } => self.access(addr, rt, acc)?,
        }
        Ok(())
    }

    /// Perform one load or store whose width, direction and sign-extension
    /// were all settled when the block was translated.
    #[inline(always)]
    fn access(&mut self, addr: u32, rt: u8, acc: Acc) -> Result<()> {
        match acc {
            Acc::Load8 => {
                let v = u64::from(self.mem.read_u8(addr)?);
                self.set_reg_at(rt, v);
            }
            Acc::Load16 => {
                let v = u64::from(self.mem.read_u16(addr)?);
                self.set_reg_at(rt, v);
            }
            Acc::Load32 => {
                let v = u64::from(self.mem.read_u32(addr)?);
                self.set_reg_at(rt, v);
            }
            Acc::Load64 => {
                let v = self.mem.read_u64(addr)?;
                self.set_reg_at(rt, v);
            }
            Acc::LoadS8 => {
                let v = u64::from(self.mem.read_u8(addr)?);
                self.set_reg_at(rt, sext_u64(v, 8));
            }
            Acc::LoadS16 => {
                let v = u64::from(self.mem.read_u16(addr)?);
                self.set_reg_at(rt, sext_u64(v, 16));
            }
            Acc::LoadS32 => {
                let v = u64::from(self.mem.read_u32(addr)?);
                self.set_reg_at(rt, sext_u64(v, 32));
            }
            Acc::Store8 => self.mem.write_u8(addr, self.reg_at(rt) as u8)?,
            Acc::Store16 => self.mem.write_u16(addr, self.reg_at(rt) as u16)?,
            Acc::Store32 => self.mem.write_u32(addr, self.reg_at(rt) as u32)?,
            Acc::Store64 => self.mem.write_u64(addr, self.reg_at(rt))?,
            // PRFM: a hint. The addressing mode's writeback still happens.
            Acc::Prefetch => {}
        }
        Ok(())
    }

    /// The `ADD`/`SUB` core, with the direction already folded into `rhs` and
    /// `carry` and register 31's two meanings already resolved.
    #[inline(always)]
    fn add_sub_pre(&mut self, rd: u8, rn: u8, rhs: u64, carry: u8, set_flags: bool, sf: bool) {
        let a = self.reg_at(rn) & Cpu::mask(sf);
        let (result, c, v) = Cpu::add_carry_overflow(a, rhs, u64::from(carry), sf);
        if set_flags {
            self.set_nzcv_from_alu(result, sf, c, v);
        }
        self.set_reg_at(rd, result);
    }

    /// The address a load/store touches and the value (if any) to write back
    /// into its base register.
    #[inline(always)]
    fn indexed(base: u64, offset: i64, wb: Wb) -> (u64, Option<u64>) {
        match wb {
            Wb::None => (base.wrapping_add(offset as u64), None),
            Wb::Pre => {
                let addr = base.wrapping_add(offset as u64);
                (addr, Some(addr))
            }
            Wb::Post => (base, Some(base.wrapping_add(offset as u64))),
        }
    }

    /// The `AND`/`ORR`/`EOR`/`ANDS` core, shared by the immediate and
    /// shifted-register forms. `ANDS` is the only one that writes flags, and
    /// it leaves C and V alone.
    #[inline(always)]
    fn logical(&mut self, a: u64, b: u64, opc: u8, sf: bool) -> u64 {
        match opc {
            0b00 => a & b,
            0b01 => a | b,
            0b10 => a ^ b,
            _ => {
                let r = a & b;
                let n = (r >> (if sf { 63 } else { 31 })) & 1;
                let z = u64::from(r == 0);
                let c = (self.nzcv >> 29) & 1;
                let v = (self.nzcv >> 28) & 1;
                self.nzcv = ((n as u32) << 31) | ((z as u32) << 30) | (c << 29) | (v << 28);
                r
            }
        }
    }

    /// Execute the instruction a block ends on, leaving `self.pc` wherever
    /// control goes next.
    #[inline(always)]
    fn exec_term(&mut self, term: Term, pc: u32) -> Result<()> {
        match term {
            Term::B { target } => self.pc = target,
            Term::Bl { target, ret_pc } => {
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Br { rn } => self.pc = self.read_zr(rn) as u32,
            Term::Blr { rn, ret_pc } => {
                // Read the target before linking: `blr x30` is a
                // return-and-relink, and writing x30 first makes it jump to
                // itself.
                let target = self.read_zr(rn) as u32;
                self.write_zr(30, u64::from(ret_pc));
                self.pc = target;
            }
            Term::Ret { rn } => {
                // A return to address 0 is homebrew's exit path; the boot
                // model routes it through the exit trampoline instead of
                // fetching from NULL.
                let target = self.read_zr(rn) as u32;
                self.pc = if target == 0 {
                    SELF_RETURN_TRAMPOLINE
                } else {
                    target
                };
            }
            Term::Svc { imm, next } => {
                // Retire the SVC before dispatching it: a syscall that
                // switches threads installs the incoming thread's PC, and the
                // outgoing one has to resume after its own SVC.
                self.pc = next;
                self.syscall(imm)?;
            }
            Term::Interpret { insn, next } => {
                self.jit.interpreted += 1;
                self.execute(insn, next)?;
            }
            Term::Fetch => {
                let insn = self.mem.fetch(pc)?;
                self.jit.interpreted += 1;
                self.execute(insn, pc.wrapping_add(4))?;
            }
        }
        Ok(())
    }
}
