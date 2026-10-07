//! Translation cache: block lookup, generations, and store invalidation.

use super::ir::Block;
use crate::IdMap;
use std::rc::Rc;

/// Blocks per generation; two are live, bounding the cache at twice this.
const MAX_BLOCKS: usize = 32 * 1024;

/// Direct-mapped lookup slots, indexed by entry address.
const LOOKUP_SLOTS: usize = 4096;

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
        self.older = std::mem::take(&mut self.blocks);
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
                    let newer = self.blocks.remove(&start).is_some();
                    let older = self.older.remove(&start).is_some();
                    if newer || older {
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
        }
    }
}
