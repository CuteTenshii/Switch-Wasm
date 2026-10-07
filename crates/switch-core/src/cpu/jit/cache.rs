//! Translation cache: block lookup, generations, and store invalidation.

use super::ir::{Block, Code};
use crate::IdMap;
use std::rc::Rc;

/// Blocks per generation; two are live, bounding the cache at twice this.
const MAX_BLOCKS: usize = 32 * 1024;

/// Direct-mapped lookup slots, indexed by entry address.
const LOOKUP_SLOTS: usize = 4096;

/// Chain-table slots, indexed like the lookup; retail titles compile some ten thousand blocks.
pub(super) const CHAIN_SLOTS: usize = 64 * 1024;

/// What emitted blocks read and write to jump straight into the next one.
#[derive(Debug)]
pub(super) struct Chain {
    /// Instructions jumped-from blocks may still retire; a jump that would take it below zero returns.
    pub(super) fuel: i32,
    /// Jumps since `run_jit` last entered a block.
    pub(super) hops: u32,
    /// Entry address of the last block jumped to.
    pub(super) last: u32,
    /// `[entry address, table entry]` of compiled blocks, by [`Chain::slot`].
    pub(super) slots: Box<[[u32; 2]; CHAIN_SLOTS]>,
}

impl Chain {
    #[inline(always)]
    pub(super) fn slot(pc: u32) -> usize {
        (pc >> 2) as usize & (CHAIN_SLOTS - 1)
    }

    /// An address that hashes to another slot, so no lookup can match it.
    fn vacant(slot: usize) -> u32 {
        (((slot + 1) & (CHAIN_SLOTS - 1)) << 2) as u32
    }

    fn reset(&mut self) {
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            *entry = [Chain::vacant(slot), 0];
        }
    }
}

impl Default for Chain {
    fn default() -> Chain {
        let mut chain = Chain {
            fuel: -1,
            hops: 0,
            last: 0,
            slots: vec![[0; 2]; CHAIN_SLOTS]
                .into_boxed_slice()
                .try_into()
                .expect("CHAIN_SLOTS long by construction"),
        };
        chain.reset();
        chain
    }
}

#[derive(Debug)]
pub(in crate::cpu) struct Jit {
    /// Last block per slot; a hint checked against the block's entry address.
    pub(super) lookup: Vec<Option<Rc<Block>>>,
    pub(super) blocks: IdMap<u32, Rc<Block>>,
    /// The previous generation; entering a block here promotes it back.
    pub(super) older: IdMap<u32, Rc<Block>>,
    /// Entry addresses per page read, so a store drops the blocks that read it.
    pub(super) by_page: IdMap<u32, Vec<u32>>,
    pub(super) translated: u64,
    pub(super) executed: u64,
    pub(super) linked: u64,
    pub(super) invalidated: u64,
    pub(super) interpreted: u64,
    pub(super) interpreted_groups: [u64; 16],
    pub(super) emitted: u64,
    pub(super) entered_emitted: u64,
    pub(super) chained: u64,
    pub(super) chain: Chain,
}

/// What the translator has been doing, for host-side diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitStats {
    pub blocks: usize,
    pub translated: u64,
    pub executed: u64,
    /// Entries reached through the previous block's link rather than a lookup.
    pub linked: u64,
    pub invalidated: u64,
    /// Instructions the translator had no op for, run by the interpreter.
    pub interpreted: u64,
    /// One in 1024 of `interpreted`, by encoding group (bits 28:25).
    pub interpreted_groups: [u64; 16],
    /// Blocks compiled to wasm by the host; always zero on host builds.
    pub emitted: u64,
    /// Entries that ran compiled code rather than the op walk.
    pub entered_emitted: u64,
    /// Entries a compiled block made by jumping straight into the next.
    pub chained: u64,
}

impl Default for Jit {
    fn default() -> Jit {
        Jit {
            lookup: vec![None; LOOKUP_SLOTS],
            blocks: IdMap::default(),
            older: IdMap::default(),
            by_page: IdMap::default(),
            translated: 0,
            executed: 0,
            linked: 0,
            invalidated: 0,
            interpreted: 0,
            interpreted_groups: [0; 16],
            emitted: 0,
            entered_emitted: 0,
            chained: 0,
            chain: Chain::default(),
        }
    }
}

impl Jit {
    #[inline(always)]
    fn slot(pc: u32) -> usize {
        (pc >> 2) as usize & (LOOKUP_SLOTS - 1)
    }

    #[inline(always)]
    pub(super) fn note_interpreted(&mut self, insn: u32) {
        self.interpreted += 1;
        if self.interpreted & 1023 == 0 {
            self.interpreted_groups[((insn >> 25) & 0xF) as usize] += 1;
        }
    }

    #[inline(always)]
    pub(super) fn get(&mut self, pc: u32) -> Option<Rc<Block>> {
        let slot = Self::slot(pc);
        if let Some(block) = &self.lookup[slot] {
            if block.start == pc {
                return Some(block.clone());
            }
        }
        let block = match self.blocks.get(&pc) {
            Some(block) => block.clone(),
            // Promote out of `older`; moving keeps the generations disjoint.
            None => {
                let block = self.older.remove(&pc)?;
                self.blocks.insert(pc, block.clone());
                block
            }
        };
        self.lookup[slot] = Some(block.clone());
        Some(block)
    }

    /// Whether `block` is the cached block for its address.
    pub(super) fn holds(&self, block: &Block) -> bool {
        let same = |held: &Rc<Block>| std::ptr::eq(Rc::as_ptr(held), block);
        self.blocks.get(&block.start).is_some_and(same)
            || self.older.get(&block.start).is_some_and(same)
    }

    /// Let emitted blocks jump to `block`'s compiled form at `entry`.
    pub(super) fn chain_to(&mut self, block: &Block, entry: u32) {
        self.chain.slots[Chain::slot(block.start)] = [block.start, entry];
    }

    /// Stop emitted blocks jumping to `block`, before its compiled form is released.
    pub(super) fn unchain(&mut self, block: &Block) {
        let Code::Ready { entry, .. } = block.code.get() else {
            return;
        };
        let slot = Chain::slot(block.start);
        if self.chain.slots[slot] == [block.start, entry as u32] {
            self.chain.slots[slot] = [Chain::vacant(slot), 0];
        }
    }

    pub(super) fn insert(&mut self, block: Rc<Block>) {
        self.rotate_if_full();
        for &page in &block.pages {
            self.by_page.entry(page).or_default().push(block.start);
        }
        self.lookup[Self::slot(block.start)] = Some(block.clone());
        self.blocks.insert(block.start, block);
    }

    /// Rotate when full, dropping blocks not entered since the last rotation.
    fn rotate_if_full(&mut self) {
        if self.blocks.len() < MAX_BLOCKS {
            return;
        }
        let dropped = std::mem::replace(&mut self.older, std::mem::take(&mut self.blocks));
        for block in dropped.values() {
            self.unchain(block);
        }
        // Rebuild `by_page` from the surviving generations.
        self.by_page.clear();
        for block in self.older.values() {
            for &page in &block.pages {
                self.by_page.entry(page).or_default().push(block.start);
            }
        }
        self.drop_lookup();
    }

    /// Called whenever a block is dropped, so no hint outlives its block.
    fn drop_lookup(&mut self) {
        for slot in &mut self.lookup {
            *slot = None;
        }
    }

    pub(super) fn invalidate(&mut self, pages: &[u32]) {
        let mut dropped = false;
        for &page in pages {
            if let Some(starts) = self.by_page.remove(&page) {
                for start in starts {
                    // Check both generations: `get` still reaches blocks in `older`.
                    let newer = self.blocks.remove(&start);
                    let older = self.older.remove(&start);
                    for block in newer.iter().chain(older.iter()) {
                        self.unchain(block);
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

    pub(super) fn clear(&mut self) {
        self.invalidated += (self.blocks.len() + self.older.len()) as u64;
        self.blocks.clear();
        self.older.clear();
        self.by_page.clear();
        self.chain.reset();
        self.drop_lookup();
    }

    pub(super) fn stats(&self) -> JitStats {
        JitStats {
            blocks: self.blocks.len() + self.older.len(),
            translated: self.translated,
            executed: self.executed,
            linked: self.linked,
            invalidated: self.invalidated,
            interpreted: self.interpreted,
            interpreted_groups: self.interpreted_groups,
            emitted: self.emitted,
            entered_emitted: self.entered_emitted,
            chained: self.chained,
        }
    }
}
