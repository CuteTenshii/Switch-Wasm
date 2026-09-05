//! The translation cache: which blocks exist, how one is found, and when a
//! guest store takes one away again.

use super::ir::Block;
use crate::IdMap;
use std::rc::Rc;

/// How many blocks the cache holds before it is dropped wholesale. A retail
/// title's hot code is a few thousand blocks; the cap is only there so a
/// program that walks endlessly over fresh code cannot grow the cache without
/// bound.
const MAX_BLOCKS: usize = 64 * 1024;

/// How many entries the direct-mapped lookup in front of the block map holds.
/// A hash of the entry address is a large share of what entering a short
/// block costs, and guest code is dense enough that indexing by the address
/// itself nearly always hits.
const LOOKUP_SLOTS: usize = 4096;

/// The translation cache.
#[derive(Debug)]
pub(in crate::cpu) struct Jit {
    /// The most recent block to land in each slot, or `None`. Only a hint:
    /// the entry address is checked against the block's own, and `blocks` is
    /// what actually owns the cache.
    pub(super) lookup: Vec<Option<Rc<Block>>>,
    pub(super) blocks: IdMap<u32, Rc<Block>>,
    /// Entry addresses translated out of each page, so a store to that page
    /// drops exactly the blocks that read it.
    pub(super) by_page: IdMap<u32, Vec<u32>>,
    pub(super) translated: u64,
    pub(super) executed: u64,
    pub(super) linked: u64,
    pub(super) invalidated: u64,
    pub(super) interpreted: u64,
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
    /// Of those, the ones reached through the previous block's own link rather
    /// than through a lookup.
    ///
    /// Against `executed` this is the link cache's hit rate, the one number
    /// that says whether chaining is doing anything, and the same number on
    /// any target.
    pub linked: u64,
    /// Blocks dropped because the memory they came from was written.
    pub invalidated: u64,
    /// Instructions that reached the interpreter's dispatcher anyway, because
    /// the translator had no op for them.
    ///
    /// Against `Cpu::run`'s step count this is the share of a run the
    /// translator did not actually translate: the one number here that says
    /// where the next block of speed is, and the same number on any target.
    pub interpreted: u64,
}

impl Default for Jit {
    fn default() -> Jit {
        Jit {
            lookup: vec![None; LOOKUP_SLOTS],
            blocks: IdMap::default(),
            by_page: IdMap::default(),
            translated: 0,
            executed: 0,
            linked: 0,
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

    pub(super) fn invalidate(&mut self, pages: &[u32]) {
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

    pub(super) fn clear(&mut self) {
        self.invalidated += self.blocks.len() as u64;
        self.blocks.clear();
        self.by_page.clear();
        self.drop_lookup();
    }

    pub(super) fn stats(&self) -> JitStats {
        JitStats {
            blocks: self.blocks.len(),
            translated: self.translated,
            executed: self.executed,
            linked: self.linked,
            invalidated: self.invalidated,
            interpreted: self.interpreted,
        }
    }
}
