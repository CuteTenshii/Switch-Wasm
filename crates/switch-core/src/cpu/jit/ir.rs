//! The translated form of an instruction: what a [`Block`] is made of.
//! Operands are extracted from the encoding when the block is built.

use crate::cpu::bits::Extract;
use crate::cpu::fp::FpForm;
use crate::cpu::loadstore::{Acc, Ext, PairKind, Wb};
use crate::cpu::system::SysOp;

#[derive(Debug, Clone, Copy)]
pub(in crate::cpu) enum Op {
    /// A hint, barrier or PSTATE-immediate write, retired with no effect.
    Nop,
    /// Not translated: run the original instruction through the interpreter.
    Interpret {
        insn: u32,
    },
    /// SIMD and floating point, sent straight to the owning decoder.
    Fp {
        insn: u32,
        scalar: bool,
        form: FpForm,
    },
    /// A system instruction [`SysOp::of`] could not place.
    System {
        insn: u32,
    },
    /// `MRS`, `MSR` and `DC ZVA`, resolved to the register they name.
    Sys {
        op: SysOp,
    },
    /// A SIMD&FP load or store, sent straight to the V=1 decoder.
    SimdLoadStore {
        insn: u32,
    },

    /// A precomputed value: `MOVZ`/`MOVN`, `ADR`/`ADRP`.
    MovConst {
        rd: u8,
        val: u64,
    },
    /// `MOV` between registers (`ORR` with the zero register).
    Mov32 {
        rd: u8,
        rn: u8,
    },
    Mov64 {
        rd: u8,
        rn: u8,
    },
    /// `MOVK`: replace the 16-bit field at `shift` with `val`.
    MovK {
        rd: u8,
        shift: u8,
        val: u16,
        sf: bool,
    },

    /// `ADD`/`SUB`/`ADDS`/`SUBS` against a constant; `rhs` is pre-inverted for
    /// subtractions with `carry` set to match.
    AddSubImm {
        rd: u8,
        rn: u8,
        rhs: u64,
        carry: u8,
        set_flags: bool,
        sf: bool,
    },
    /// Shifted-register form; `carry` is 1 for subtractions and doubles as the inversion mask.
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
    /// The same with no shift.
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

    /// `AND`/`ORR`/`EOR`/`ANDS` with the bitmask immediate decoded.
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
    /// The unshifted form (`mov xd, xm`, `mvn xd, xm`).
    LogicalReg {
        rd: u8,
        rn: u8,
        rm: u8,
        opc: u8,
        invert: bool,
        sf: bool,
    },

    /// `SBFM`/`UBFM` and their aliases, decoded to shifts.
    Extract {
        rd: u8,
        rn: u8,
        extract: Extract,
        sf: bool,
    },
    /// `BFM` and the unallocated `opc`.
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

    /// `ADC`/`ADCS`/`SBC`/`SBCS`.
    Adc {
        rd: u8,
        rn: u8,
        rm: u8,
        sub: bool,
        set_flags: bool,
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
    /// `RBIT`/`REV16`/`REV32`/`REV`/`CLZ`/`CLS`/`CTZ`, by opcode field.
    OneSource {
        rd: u8,
        rn: u8,
        opcode: u8,
        sf: bool,
    },
    /// `CRC32`/`CRC32C` over `8 << sz` bits of Rm.
    Crc {
        rd: u8,
        rn: u8,
        rm: u8,
        sz: u8,
        castagnoli: bool,
    },

    LoadStoreImm {
        rt: u8,
        rn: u8,
        acc: Acc,
        wb: Wb,
        offset: i64,
    },
    /// [`Op::LoadStoreImm`] specialized for the six commonest accesses.
    /// Build through [`Op::load_store_imm`].
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
    /// [`Op::Pair`] specialized for X-register `LDP`/`STP`. Build through [`Op::pair`].
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
    /// `LDR <t>, label`, with the literal's address resolved.
    LoadLiteral {
        rt: u8,
        addr: u32,
        acc: Acc,
    },
    /// `LDXR`/`LDAXR`, one register.
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

/// The instruction a block ends on. Conditional branches are [`Exit`]s instead;
/// a block with no terminator falls through at the length or page limit.
#[derive(Debug, Clone, Copy)]
pub(super) enum Term {
    /// `B #imm`.
    B { target: u32 },
    /// `BL #imm`.
    Bl { target: u32, ret_pc: u32 },
    /// `BL` to a PLT stub, run inline; `got` is the slot it loads from, `stub` the fallback.
    BlPlt { got: u32, stub: u32, ret_pc: u32 },
    /// `B` to a PLT stub (a tail call), folded the same way.
    BPlt { got: u32, stub: u32 },
    /// `BR Xn`.
    Br { rn: u8 },
    /// `BLR Xn`.
    Blr { rn: u8, ret_pc: u32 },
    /// `RET Xn`.
    Ret { rn: u8 },
    /// `SVC #imm`.
    Svc { imm: u16, next: u32 },
    /// A control instruction the interpreter decodes, setting the PC itself.
    Interpret { insn: u32, next: u32 },
    /// Unreadable at translation time; retried at run time so the fault is current.
    Fetch,
}

/// A conditional branch inside a block: leave if the condition holds, else continue.
/// Carries its own tag byte for cheaper dispatch.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
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
    /// `CMP`/`CMN` against a constant fused with the following `B.cond`.
    /// `imm` is as encoded; `carry` is 1 for `CMP`.
    CmpImm {
        rn: u8,
        imm: u32,
        carry: u8,
        sf: bool,
        cond: u8,
        target: u32,
    },
    /// A flagless `ADD`/`SUB` of a constant fused ahead of an [`Exit::CmpImm`].
    UpdateCmpImm {
        rd: u8,
        source: u8,
        step: PackedImm,
        rn: u8,
        imm: PackedImm,
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
    /// A `B #imm` the translator followed; the ops after it are at `target`.
    Jump { target: u32 },
}

impl Exit {
    /// Instructions the exit covers, including fused compare and update.
    fn span(&self) -> u8 {
        match self {
            Exit::CmpImm { .. } | Exit::CmpReg { .. } => 2,
            Exit::UpdateCmpImm { .. } => 3,
            _ => 1,
        }
    }
}

/// An `ADD`/`SUB`/`CMP`/`CMN` immediate packed into sixteen bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PackedImm(u16);

impl PackedImm {
    const SHIFTED: u16 = 1 << 12;
    const SUB: u16 = 1 << 13;
    const SF: u16 = 1 << 14;

    pub(super) fn new(imm: u32, sub: bool, sf: bool) -> Option<PackedImm> {
        let low = if imm & !0xFFF == 0 {
            imm as u16
        } else if imm & !0xFF_F000 == 0 {
            (imm >> 12) as u16 | Self::SHIFTED
        } else {
            return None;
        };
        let sub = if sub { Self::SUB } else { 0 };
        let sf = if sf { Self::SF } else { 0 };
        Some(PackedImm(low | sub | sf))
    }

    /// The constant, uninverted.
    pub(super) fn imm(self) -> u64 {
        let low = u64::from(self.0 & 0xFFF);
        if self.0 & Self::SHIFTED != 0 {
            low << 12
        } else {
            low
        }
    }

    /// 1 for a subtraction, which is also the carry in.
    pub(super) fn carry(self) -> u8 {
        u8::from(self.0 & Self::SUB != 0)
    }

    pub(super) fn sf(self) -> bool {
        self.0 & Self::SF != 0
    }
}

/// Where a conditional branch sits in a block, with its span precomputed.
#[derive(Debug, Clone, Copy)]
pub(super) struct Branch {
    /// Index into [`Block::ops`].
    pub(super) at: u32,
    /// Instructions the branch covers.
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

/// An empty link slot; unaligned, so it never matches a block.
const NO_LINK: u32 = 1;

/// How many successors a block remembers.
const LINKS: usize = 4;

/// Whether a block has been emitted as wasm. Blocks start cold and are counted up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Code {
    /// Entered this many times, not emitted yet.
    Cold(u32),
    /// Emitted: its entry point and how many entries handed back without retiring anything.
    Ready {
        entry: super::host::Entry,
        misses: u16,
    },
    /// Never to be emitted.
    Never,
}

/// A run of instructions with a single entry point, translated once.
#[derive(Debug)]
pub(super) struct Block {
    /// The last [`LINKS`] successors, most recent first. [`Weak`] so a dropped
    /// (invalidated) block cannot be reached through a link.
    pub(super) link: std::cell::RefCell<[(u32, std::rc::Weak<Block>); LINKS]>,
    pub(super) code: std::cell::Cell<Code>,
    pub(super) start: u32,
    /// One entry per instruction before the terminator; branch slots hold [`Op::Nop`].
    pub(super) ops: Vec<Op>,
    /// Original instruction words, kept for fault run-up trails.
    pub(super) words: Vec<u32>,
    /// Conditional branches, in ascending order.
    pub(super) exits: Vec<Branch>,
    /// Whether any exit is a followed [`Exit::Jump`].
    pub(super) follows: bool,
    pub(super) term: Option<Term>,
    /// Page numbers the block was read from; a store to any drops the block.
    pub(super) pages: Vec<u32>,
}

impl Block {
    /// Trims `ops` and `words`, which the translator sizes for the longest block.
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
        let follows = exits.iter().any(|b| matches!(b.exit, Exit::Jump { .. }));
        Block {
            link: std::cell::RefCell::new(std::array::from_fn(|_| (NO_LINK, std::rc::Weak::new()))),
            code: std::cell::Cell::new(Code::Cold(0)),
            start,
            ops,
            words,
            exits,
            follows,
            term,
            pages,
        }
    }

    #[inline(always)]
    pub(super) fn successor(&self, pc: u32) -> Option<std::rc::Rc<Block>> {
        let links = self.link.borrow();
        links.iter().find(|l| l.0 == pc).and_then(|l| l.1.upgrade())
    }

    #[inline(always)]
    pub(super) fn link_to(&self, pc: u32, block: &std::rc::Rc<Block>) {
        let mut links = self.link.borrow_mut();
        links.rotate_right(1);
        links[0] = (pc, std::rc::Rc::downgrade(block));
    }

    /// Release the emitted form's table slot; the block goes on being interpreted.
    pub(super) fn drop_code(&self) {
        if let Code::Ready { entry, .. } = self.code.replace(Code::Never) {
            super::host::release(entry);
        }
    }
}

/// Free the function table slot when a block is dropped.
impl Drop for Block {
    fn drop(&mut self) {
        self.drop_code();
    }
}

#[cfg(test)]
mod tests {
    use super::{Branch, Exit, Op, SysOp};

    #[test]
    fn an_op_still_costs_one_word_and_its_tag() {
        assert_eq!(std::mem::size_of::<Op>(), 16);
        assert!(std::mem::size_of::<SysOp>() <= 16);
    }

    #[test]
    fn a_branch_costs_no_more_than_the_pair_it_replaced() {
        assert_eq!(std::mem::size_of::<Branch>(), 24);
    }

    #[test]
    fn an_exit_with_its_own_tag_still_costs_two_words() {
        assert_eq!(std::mem::size_of::<Exit>(), 16);
    }
}
