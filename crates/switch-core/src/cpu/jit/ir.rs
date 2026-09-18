//! The translated form of an instruction: what a [`Block`] is made of.
//!
//! Every field an [`Op`] carries was extracted from the encoding when the
//! block was built, so executing one asks nothing about the instruction it
//! came from. The load/store vocabulary these share with the interpreter
//! ([`Acc`], [`Ext`], [`PairKind`], [`Wb`]) lives in
//! [`crate::cpu::loadstore`], and the system-register one in
//! [`crate::cpu::system`].

use crate::cpu::bits::Extract;
use crate::cpu::fp::FpForm;
use crate::cpu::loadstore::{Acc, Ext, PairKind, Wb};
use crate::cpu::system::SysOp;

/// One translated instruction: what it does, with its operands already pulled
/// out of the encoding.
#[derive(Debug, Clone, Copy)]
pub(in crate::cpu) enum Op {
    /// A hint, barrier or PSTATE-immediate write the interpreter also retires
    /// with no effect.
    Nop,
    /// Not translated: run the original instruction through the interpreter.
    Interpret {
        insn: u32,
    },
    /// SIMD and floating point, handed to the decoder that owns it instead of
    /// back through [`crate::cpu::Cpu::execute`]'s group match. `scalar` is the
    /// same top-byte test `execute` makes to decide which of the two decoders
    /// gets first look, and `form` is which of the scalar forms it is, both
    /// decided once here rather than on every execution.
    Fp {
        insn: u32,
        scalar: bool,
        form: FpForm,
    },
    /// A system instruction [`SysOp::of`] could not place, its
    /// [`SysOp::Unhandled`]. Straight to [`crate::cpu::Cpu::system`], which is
    /// where its error comes from.
    System {
        insn: u32,
    },
    /// `MRS`, `MSR` and `DC ZVA`, already resolved to the register they name.
    Sys {
        op: SysOp,
    },

    /// A value the translator already computed: `MOVZ`/`MOVN`, and the
    /// PC-relative `ADR`/`ADRP` whose result depends only on where the
    /// instruction is.
    MovConst {
        rd: u8,
        val: u64,
    },
    /// `MOVK`: replace the 16-bit field at `shift` with `val`. Held as a
    /// shift and a halfword rather than a mask and a placed value so the
    /// variant needs one 64-bit word instead of two, which is what decides
    /// [`Op`]'s size, and so a block body's whole cache footprint.
    MovK {
        rd: u8,
        shift: u8,
        val: u16,
        sf: bool,
    },

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
    /// The same with no shift at all (`add x0, x1, x2`) which is most of
    /// them, and skips [`crate::cpu::bits::shift_reg`] entirely.
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

    /// `SBFM`/`UBFM` and every alias of them, already decoded to the shifts
    /// they are. 3.9% of a retail frame, most of it `LSL`/`LSR`/`UBFX`.
    Extract {
        rd: u8,
        rn: u8,
        extract: Extract,
        sf: bool,
    },
    /// `BFM`, the one form that keeps bits of Rd, and the unallocated `opc`.
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
    /// `LSLV`/`LSRV`/`ASRV`/`RORV`, `kind` being the shift type.
    ShiftVar {
        rd: u8,
        rn: u8,
        rm: u8,
        kind: u8,
        sf: bool,
    },
    /// `UDIV`/`SDIV`.
    Divide {
        rd: u8,
        rn: u8,
        rm: u8,
        signed: bool,
        sf: bool,
    },

    LoadStoreImm {
        rt: u8,
        rn: u8,
        acc: Acc,
        wb: Wb,
        offset: i64,
    },
    /// [`Op::LoadStoreImm`] for the six accesses a retail frame makes almost
    /// all of its single-register loads and stores with, the access folded
    /// into the variant. Build them through [`Op::load_store_imm`].
    ///
    /// The general form dispatches twice, once on the op and again on `acc`,
    /// and under V8 each of the two is an indirect jump that mispredicts on
    /// its own. The samples of the general arm sat on the instructions just
    /// past those jumps rather than on the memory access, so the second one
    /// goes where the first already is.
    Load64 {
        rt: u8,
        rn: u8,
        wb: Wb,
        offset: i64,
    },
    Store64 {
        rt: u8,
        rn: u8,
        wb: Wb,
        offset: i64,
    },
    Load32 {
        rt: u8,
        rn: u8,
        wb: Wb,
        offset: i64,
    },
    Store32 {
        rt: u8,
        rn: u8,
        wb: Wb,
        offset: i64,
    },
    Load8 {
        rt: u8,
        rn: u8,
        wb: Wb,
        offset: i64,
    },
    Store8 {
        rt: u8,
        rn: u8,
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
    /// [`Op::Pair`] for `LDP`/`STP` of X registers, which is what every
    /// prologue and epilogue saves and restores through, with the kind folded
    /// into the variant for the same reason as [`Op::Load64`]. Build them
    /// through [`Op::pair`].
    PairLoad64 {
        rt: u8,
        rt2: u8,
        rn: u8,
        offset: i64,
        wb: Wb,
    },
    PairStore64 {
        rt: u8,
        rt2: u8,
        rn: u8,
        offset: i64,
        wb: Wb,
    },
    /// `LDR <t>, label`, with the literal's address already resolved.
    LoadLiteral {
        rt: u8,
        addr: u32,
        acc: Acc,
    },
    /// `LDXR`/`LDAXR`, one register. The lock word of every `nn::os` mutex
    /// goes through this and [`Op::StoreExclusive`].
    LoadExclusive {
        rt: u8,
        rn: u8,
        sz: u8,
    },
    /// `STXR`/`STLXR`, one register; `rs` takes the status.
    StoreExclusive {
        rs: u8,
        rt: u8,
        rn: u8,
        sz: u8,
    },
}

impl Op {
    /// A single-register load or store with an immediate offset, as the
    /// variant that has its access built in when there is one.
    pub(super) fn load_store_imm(rt: u8, rn: u8, acc: Acc, wb: Wb, offset: i64) -> Op {
        match acc {
            Acc::Load64 => Op::Load64 { rt, rn, wb, offset },
            Acc::Store64 => Op::Store64 { rt, rn, wb, offset },
            Acc::Load32 => Op::Load32 { rt, rn, wb, offset },
            Acc::Store32 => Op::Store32 { rt, rn, wb, offset },
            Acc::Load8 => Op::Load8 { rt, rn, wb, offset },
            Acc::Store8 => Op::Store8 { rt, rn, wb, offset },
            _ => Op::LoadStoreImm {
                rt,
                rn,
                acc,
                wb,
                offset,
            },
        }
    }

    /// A load or store pair, as the variant that has its kind built in when
    /// there is one.
    pub(super) fn pair(rt: u8, rt2: u8, rn: u8, offset: i64, kind: PairKind, wb: Wb) -> Op {
        match kind {
            PairKind::Load64 => Op::PairLoad64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            },
            PairKind::Store64 => Op::PairStore64 {
                rt,
                rt2,
                rn,
                offset,
                wb,
            },
            _ => Op::Pair {
                rt,
                rt2,
                rn,
                offset,
                kind,
                wb,
            },
        }
    }
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
    /// `BL` to a PLT stub, with the stub run as part of it: see
    /// [`super::decode`]'s `plt_slot`. `got` is the slot the stub loads its
    /// target from, and `stub` where the stub is, for the fallback.
    BlPlt { got: u32, stub: u32, ret_pc: u32 },
    /// `B` to a PLT stub, a tail call into another module, folded the same
    /// way.
    BPlt { got: u32, stub: u32 },
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
    /// its flags, the commonest pair in compiled code, and one that only
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
    /// A `B #imm` the translator followed: always taken, and the block goes
    /// on at `target` rather than ending. The ops after it are the ones at
    /// `target`.
    Jump { target: u32 },
}

impl Exit {
    /// How many instructions the exit covers: two once a compare has been
    /// folded into it.
    fn span(&self) -> u8 {
        match self {
            Exit::CmpImm { .. } | Exit::CmpReg { .. } => 2,
            _ => 1,
        }
    }
}

/// Where a conditional branch sits in a block, and how much of it the branch
/// speaks for.
///
/// The span is [`Exit::span`] resolved once, when the block is built. Asking
/// the branch itself meant loading its discriminant and testing it before the
/// exit could even be evaluated, on the same pass that then dispatches on that
/// discriminant again; the padding in this record was already there to hold
/// it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Branch {
    /// Index into [`Block::ops`]: the instruction the branch is checked at.
    pub(super) at: u32,
    /// Instructions the branch covers, two once a compare has been fused into
    /// it.
    pub(super) span: u8,
    pub(super) exit: Exit,
}

impl Branch {
    pub(super) fn new(at: u32, exit: Exit) -> Branch {
        Branch {
            at,
            span: exit.span(),
            exit,
        }
    }
}

/// The address an empty link slot holds. Unaligned, so no block starts there
/// and an empty slot can never match.
const NO_LINK: u32 = 1;

/// How many successors a block remembers.
///
/// A block that runs through a conditional branch has two, the branch's target
/// and wherever its terminator goes, and one slot evicted one for the other
/// every time the branch changed its mind. A function's `RET` has as many as
/// it has callers. On a Just Dance 2019 frame one slot linked 73.3% of block
/// entries, two 79.8% and four 82.5%, each step taking 2.2% and then 1.4% off
/// the frame in the wasm build. Eight linked 82.6%: what still misses is `RET`
/// from functions with more callers than any small cache holds.
const LINKS: usize = 4;

/// A run of instructions with a single entry point, translated once.
#[derive(Debug)]
pub(super) struct Block {
    /// The last [`LINKS`] places control went when this block was left, and
    /// the blocks it found there: an inline cache filled on the way past, most
    /// recent first.
    ///
    /// A retail frame enters a block every 6.1 instructions, so what a block
    /// boundary costs is charged against six instructions rather than against
    /// a whole loop body. Most of those boundaries go somewhere they have been
    /// before: a loop alternating between two blocks, a `RET` to the site that
    /// called it, a `BLR` through a call site that is monomorphic in practice.
    ///
    /// Held [`Weak`] so it cannot keep a block alive. That is not only about
    /// the A-to-B-to-A cycle leaking: a block dropped because a guest store
    /// landed on its page is *gone* from the cache, and a link that still
    /// upgraded would be running code the guest has overwritten. Failing to
    /// upgrade is exactly the right answer, and it needs no invalidation pass
    /// of its own.
    pub(super) link: std::cell::RefCell<[(u32, std::rc::Weak<Block>); LINKS]>,
    /// Guest address of the first instruction.
    pub(super) start: u32,
    /// One entry per instruction the block covers before its terminator, in
    /// the order they run. Consecutive entries are consecutive instructions
    /// except across an [`Exit::Jump`], after which they are the ones at its
    /// target. The slots that hold a branch carry [`Op::Nop`]
    /// as filler, the branch itself is in `exits`, and keeping one slot per
    /// instruction is worth one dead slot per exit.
    pub(super) ops: Vec<Op>,
    /// The original instruction words, body then terminator, kept so a fault
    /// inside a block leaves the same run-up trail an interpreted one does.
    pub(super) words: Vec<u32>,
    /// The conditional branches the block runs through, in ascending order of
    /// where they sit.
    pub(super) exits: Vec<Branch>,
    pub(super) term: Option<Term>,
    /// Every page the block's instructions were read from, as page numbers.
    /// One for most blocks; more once it follows a `B`, runs off the end of
    /// a page, or folds in a PLT stub that lives on another. A store to any
    /// of them has to drop the block.
    pub(super) pages: Vec<u32>,
}

impl Block {
    /// A block with nothing linked to it yet, from the parts [`super::decode`]
    /// builds.
    ///
    /// The translator sizes `ops` and `words` for the longest block a page
    /// allows, 1.25 KiB between them, and a block is a handful of
    /// instructions, so both are trimmed here. Kept at that size, a full cache
    /// was ~80 MiB of mostly empty heap, in a wasm32 address space the guest
    /// may claim 3.2 GiB of.
    pub(super) fn new(
        start: u32,
        mut ops: Vec<Op>,
        mut words: Vec<u32>,
        exits: Vec<Branch>,
        term: Option<Term>,
        mut pages: Vec<u32>,
    ) -> Block {
        ops.shrink_to_fit();
        words.shrink_to_fit();
        pages.shrink_to_fit();
        Block {
            link: std::cell::RefCell::new(std::array::from_fn(|_| (NO_LINK, std::rc::Weak::new()))),
            start,
            ops,
            words,
            exits,
            term,
            pages,
        }
    }

    /// The block at `pc`, if that is one of the places this one went recently
    /// and it is still translated.
    #[inline(always)]
    pub(super) fn successor(&self, pc: u32) -> Option<std::rc::Rc<Block>> {
        let links = self.link.borrow();
        links.iter().find(|l| l.0 == pc).and_then(|l| l.1.upgrade())
    }

    /// Remember that control went to `block` at `pc`, forgetting the oldest
    /// place it remembered before.
    #[inline(always)]
    pub(super) fn link_to(&self, pc: u32, block: &std::rc::Rc<Block>) {
        let mut links = self.link.borrow_mut();
        links.rotate_right(1);
        links[0] = (pc, std::rc::Rc::downgrade(block));
    }
}

#[cfg(test)]
mod tests {
    use super::{Branch, Exit, Op, SysOp};

    /// A block body is an array of [`Op`], so its size is that body's whole
    /// cache footprint, which is why [`Op::MovK`] holds a shift and a
    /// halfword rather than a mask and a placed value, and why
    /// [`super::SysReg::Fixed`] is a `u32`. One 64-bit payload plus its
    /// discriminant is the budget those choices were made against.
    #[test]
    fn an_op_still_costs_one_word_and_its_tag() {
        assert_eq!(std::mem::size_of::<Op>(), 16);
        // Folding the four system-instruction variants into one `Op::Sys`
        // only stays free while `SysOp` fits inside that budget.
        assert!(std::mem::size_of::<SysOp>() <= 16);
    }

    /// [`Branch::span`] is free only while it fits in the padding an
    /// `(index, Exit)` pair already carried. Grow [`Exit`] past this and the
    /// span is worth re-deriving instead.
    #[test]
    fn a_branch_costs_no_more_than_the_pair_it_replaced() {
        assert_eq!(
            std::mem::size_of::<Branch>(),
            std::mem::size_of::<(u32, Exit)>()
        );
    }
}
