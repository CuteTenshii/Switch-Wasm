//! The translated form of an instruction: what a [`Block`] is made of.
//!
//! Every field an [`Op`] carries was extracted from the encoding when the
//! block was built, so executing one asks nothing about the instruction it
//! came from. The load/store vocabulary these share with the interpreter
//! ([`Acc`], [`Ext`], [`PairKind`], [`Wb`]) lives in
//! [`crate::cpu::loadstore`], and the system-register one in
//! [`crate::cpu::system`].

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
    Interpret { insn: u32 },
    /// SIMD and floating point, handed to the decoder that owns it instead of
    /// back through [`crate::cpu::Cpu::execute`]'s group match. `scalar` is the
    /// same top-byte test `execute` makes to decide which of the two decoders
    /// gets first look, and `form` is which of the scalar forms it is — both
    /// decided once here rather than on every execution.
    Fp {
        insn: u32,
        scalar: bool,
        form: FpForm,
    },
    /// A system instruction [`SysOp::of`] could not place — its
    /// [`SysOp::Unhandled`]. Straight to [`crate::cpu::Cpu::system`], which is
    /// where its error comes from.
    System { insn: u32 },
    /// `MRS`, `MSR` and `DC ZVA`, already resolved to the register they name.
    Sys { op: SysOp },

    /// A value the translator already computed: `MOVZ`/`MOVN`, and the
    /// PC-relative `ADR`/`ADRP` whose result depends only on where the
    /// instruction is.
    MovConst { rd: u8, val: u64 },
    /// `MOVK`: replace the 16-bit field at `shift` with `val`. Held as a
    /// shift and a halfword rather than a mask and a placed value so the
    /// variant needs one 64-bit word instead of two — which is what decides
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
    /// The same with no shift at all — `add x0, x1, x2` — which is most of
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
    pub(super) fn span(self) -> usize {
        match self {
            Exit::CmpImm { .. } | Exit::CmpReg { .. } => 2,
            _ => 1,
        }
    }
}

/// A run of instructions with a single entry point, translated once.
#[derive(Debug)]
pub(super) struct Block {
    /// Where control went the last time this block was left, and the block it
    /// found there — an inline cache of one entry, filled on the way past.
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
    pub(super) link: std::cell::RefCell<Option<(u32, std::rc::Weak<Block>)>>,
    /// Guest address of the first instruction.
    pub(super) start: u32,
    /// One entry per instruction the block covers before its terminator, so
    /// `ops[i]` is the instruction at `start + 4 * i`. The slots that hold a
    /// conditional branch carry [`Op::Nop`] as filler — the branch itself is
    /// in `exits`, and keeping the indexing exact is worth one dead slot per
    /// exit.
    pub(super) ops: Vec<Op>,
    /// The original instruction words, body then terminator, kept so a fault
    /// inside a block leaves the same run-up trail an interpreted one does.
    pub(super) words: Vec<u32>,
    /// The conditional branches the block runs through, as
    /// `(index into ops, branch)`, in ascending order.
    pub(super) exits: Vec<(u32, Exit)>,
    pub(super) term: Option<Term>,
}

impl Block {
    /// A block with nothing linked to it yet, from the parts [`super::decode`]
    /// builds.
    pub(super) fn new(
        start: u32,
        ops: Vec<Op>,
        words: Vec<u32>,
        exits: Vec<(u32, Exit)>,
        term: Option<Term>,
    ) -> Block {
        Block {
            link: std::cell::RefCell::new(None),
            start,
            ops,
            words,
            exits,
            term,
        }
    }

    /// The block at `pc`, if that is where this one went last time and it is
    /// still translated.
    #[inline(always)]
    pub(super) fn successor(&self, pc: u32) -> Option<std::rc::Rc<Block>> {
        match &*self.link.borrow() {
            Some((at, block)) if *at == pc => block.upgrade(),
            _ => None,
        }
    }

    /// Remember that control went to `block` at `pc`.
    #[inline(always)]
    pub(super) fn link_to(&self, pc: u32, block: &std::rc::Rc<Block>) {
        *self.link.borrow_mut() = Some((pc, std::rc::Rc::downgrade(block)));
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, SysOp};

    /// A block body is an array of [`Op`], so its size is that body's whole
    /// cache footprint — which is why [`Op::MovK`] holds a shift and a
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
}
