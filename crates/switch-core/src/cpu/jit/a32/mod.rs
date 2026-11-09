//! The A32 translator: blocks of pre-decoded ARM instructions and their cache.
//!
//! The same shape as the A64 side, without the emitted form: [`decode`] turns
//! each instruction into an [`Op`] that [`exec`] runs through the helpers the
//! interpreter uses, and a block ends at the first instruction that can write
//! the PC.

pub(in crate::cpu::jit) mod decode;
mod exec;

pub use decode::a32_translates;
pub(in crate::cpu::jit) use decode::translate;

use super::host;
use super::ir::Code;
use crate::cpu::a32::loadstore::Access;
use crate::IdMap;
use std::cell::Cell;
use std::rc::Rc;

/// The condition field of an instruction that always runs.
pub(super) const ALWAYS: u8 = 0xE;

/// Blocks cached before the cache starts over.
pub(super) const MAX_BLOCKS: usize = 32 * 1024;

/// Direct-mapped lookup slots, indexed by entry address.
const LOOKUP_SLOTS: usize = 4096;

/// The second operand of a data-processing instruction, with the shifter's
/// immediate cases settled at translation.
#[derive(Debug, Clone, Copy)]
pub(super) enum Operand {
    /// A rotated immediate; C comes from its top bit.
    Imm(u32),
    /// An unrotated immediate, which leaves C alone.
    ImmKeep(u32),
    /// `rm` shifted by a constant: `ty` is the shift type, 4 for `RRX`.
    RegImm { rm: u8, ty: u8, amount: u8 },
    /// `rm` shifted by the bottom byte of `rs`.
    RegReg { rm: u8, ty: u8, rs: u8 },
}

/// The offset of a load or store.
#[derive(Debug, Clone, Copy)]
pub(super) enum Offset {
    Imm(u16),
    /// `rm` shifted by a constant.
    Reg {
        rm: u8,
        ty: u8,
        amount: u8,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Kind {
    Nop,
    /// An instruction the translator has no op for; it never writes the PC.
    Interpret {
        insn: u32,
    },
    Dp {
        opcode: u8,
        set_flags: bool,
        rd: u8,
        rn: u8,
        operand: Operand,
    },
    /// `LDR`, `STR`, `LDRB` and `STRB`.
    Single {
        access: Access,
        byte: bool,
        offset: Offset,
    },
    /// The halfword, signed and doubleword forms; `sh` is bits 6:5.
    Extra {
        access: Access,
        sh: u8,
        offset: Offset,
    },
    /// `MOVW` and `MOVT`: the halfword into the low half, or the high half over the low.
    MoveWide {
        rd: u8,
        imm: u16,
        top: bool,
    },
    /// `LDM` and `STM`, with no PC in the list.
    Block {
        access: Access,
        list: u16,
    },
}

/// One instruction of a block body, run only if its condition holds.
#[derive(Debug, Clone, Copy)]
pub(super) struct Op {
    pub(super) cond: u8,
    pub(super) kind: Kind,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum TermKind {
    B {
        target: u32,
    },
    Bl {
        target: u32,
    },
    Bx {
        rm: u8,
    },
    Svc {
        imm: u16,
    },
    /// An instruction that can write the PC, run by the interpreter.
    Interpret {
        insn: u32,
    },
    /// Unreadable at translation time; retried at run time so the fault is current.
    Fetch,
}

/// The instruction a block ends on. A skipped one falls through to the next address.
#[derive(Debug, Clone, Copy)]
pub(super) struct Term {
    pub(super) cond: u8,
    pub(super) kind: TermKind,
}

/// A run of instructions with a single entry point, translated once.
#[derive(Debug)]
pub(super) struct Block {
    pub(super) code: Cell<Code>,
    /// How many leading ops the emitted form covers.
    pub(super) emitted: Cell<usize>,
    pub(super) start: u32,
    pub(super) ops: Vec<Op>,
    /// Original instruction words, kept for fault reports.
    pub(super) words: Vec<u32>,
    /// `None` when the block was cut at the length limit.
    pub(super) term: Option<Term>,
    /// Page numbers the block was read from; a store to any drops it.
    pub(super) pages: Vec<u32>,
}

/// Free the function table slot when a block is dropped.
impl Drop for Block {
    fn drop(&mut self) {
        self.drop_code();
    }
}

/// Translated blocks by entry address, and the pages they were read from.
#[derive(Debug)]
pub(super) struct Cache {
    /// Last block per slot; a hint checked against the block's entry address.
    lookup: Vec<Option<Rc<Block>>>,
    blocks: IdMap<u32, Rc<Block>>,
    by_page: IdMap<u32, Vec<u32>>,
}

impl Default for Cache {
    fn default() -> Cache {
        Cache {
            lookup: vec![None; LOOKUP_SLOTS],
            blocks: IdMap::default(),
            by_page: IdMap::default(),
        }
    }
}

impl Block {
    /// Release the emitted form's table slot; the block goes on being walked.
    pub(super) fn drop_code(&self) {
        if let Code::Ready { entry, .. } = self.code.replace(Code::Never) {
            host::release(entry);
        }
    }
}

impl Cache {
    fn slot(pc: u32) -> usize {
        (pc >> 2) as usize & (LOOKUP_SLOTS - 1)
    }

    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }

    pub(super) fn get(&mut self, pc: u32) -> Option<Rc<Block>> {
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

    pub(super) fn insert(&mut self, block: Rc<Block>) {
        for &page in &block.pages {
            self.by_page.entry(page).or_default().push(block.start);
        }
        self.lookup[Self::slot(block.start)] = Some(block.clone());
        self.blocks.insert(block.start, block);
    }

    /// Drop every block read from one of `pages`, returning them so their
    /// compiled forms can be unchained before they are released.
    pub(super) fn invalidate(&mut self, pages: &[u32]) -> Vec<Rc<Block>> {
        let mut dropped = Vec::new();
        for page in pages {
            for start in self.by_page.remove(page).into_iter().flatten() {
                dropped.extend(self.blocks.remove(&start));
            }
        }
        if !dropped.is_empty() {
            self.lookup.fill(None);
        }
        dropped
    }

    pub(super) fn clear(&mut self) -> usize {
        let dropped = self.blocks.len();
        self.blocks.clear();
        self.by_page.clear();
        self.lookup.fill(None);
        dropped
    }
}
