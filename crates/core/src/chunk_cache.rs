//! Which fixed-size chunks of the host's files are held and in which slot of the host's pool,
//! with LRU eviction and read-ahead on sequential misses. The bytes themselves live with the host.

use crate::IdMap;

/// After this many sequential misses in a row a miss reads `1 << MAX_RUN` chunks.
const MAX_RUN: u32 = 4;

struct Held {
    slot: usize,
    used: u64,
}

struct Streak {
    next: u64,
    run: u32,
}

pub struct ChunkIndex {
    capacity: usize,
    held: IdMap<u64, Held>,
    free: Vec<usize>,
    unused: usize,
    tick: u64,
    streaks: IdMap<u32, Streak>,
}

fn key(file: u32, chunk: u64) -> u64 {
    u64::from(file) << 40 | chunk
}

impl ChunkIndex {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            held: IdMap::default(),
            free: Vec::new(),
            unused: 0,
            tick: 0,
            streaks: IdMap::default(),
        }
    }

    /// The slot holding `chunk` of `file`, marked as just used.
    pub fn get(&mut self, file: u32, chunk: u64) -> Option<usize> {
        self.tick += 1;
        let held = self.held.get_mut(&key(file, chunk))?;
        held.used = self.tick;
        Some(held.slot)
    }

    /// Slots for the chunks the host should read now, starting at `chunk`, which is missing.
    /// The span grows while misses run in sequence and stops at the file's last chunk.
    pub fn miss(&mut self, file: u32, chunk: u64, chunks_in_file: u64) -> Vec<usize> {
        let streak = self.streaks.entry(file).or_insert(Streak {
            next: u64::MAX,
            run: 0,
        });
        streak.run = if chunk == streak.next {
            (streak.run + 1).min(MAX_RUN)
        } else {
            0
        };
        let count = (1u64 << streak.run)
            .min(chunks_in_file.saturating_sub(chunk))
            .min(self.capacity as u64)
            .max(1);
        streak.next = chunk + count;
        (0..count).map(|i| self.claim(file, chunk + i)).collect()
    }

    fn claim(&mut self, file: u32, chunk: u64) -> usize {
        let slot = if self.held.len() >= self.capacity {
            self.evict()
        } else if let Some(slot) = self.free.pop() {
            slot
        } else {
            self.unused += 1;
            self.unused - 1
        };
        self.tick += 1;
        let used = self.tick;
        self.held.insert(key(file, chunk), Held { slot, used });
        slot
    }

    fn evict(&mut self) -> usize {
        let oldest = self
            .held
            .iter()
            .min_by_key(|(_, held)| held.used)
            .map(|(&k, _)| k);
        oldest
            .and_then(|k| self.held.remove(&k))
            .map_or(self.unused, |held| held.slot)
    }

    /// `chunk` was claimed but its bytes never arrived.
    pub fn forget(&mut self, file: u32, chunk: u64) {
        if let Some(held) = self.held.remove(&key(file, chunk)) {
            self.free.push(held.slot);
        }
    }

    /// Like [`forget`](Self::forget), for a slot the host cannot back: it is never handed out again.
    pub fn retire(&mut self, file: u32, chunk: u64) {
        if self.held.remove(&key(file, chunk)).is_some() {
            self.capacity = self.capacity.saturating_sub(1);
        }
    }

    /// Drop everything held for `file`, such as when another file takes its index.
    pub fn forget_file(&mut self, file: u32) {
        let lo = key(file, 0);
        let hi = key(file + 1, 0);
        let gone: Vec<u64> = self
            .held
            .keys()
            .copied()
            .filter(|k| (lo..hi).contains(k))
            .collect();
        for k in gone {
            if let Some(held) = self.held.remove(&k) {
                self.free.push(held.slot);
            }
        }
        self.streaks.remove(&file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missed_chunk_is_found_until_evicted() {
        let mut index = ChunkIndex::new(2);
        assert_eq!(index.get(0, 5), None);
        let slots = index.miss(0, 5, 100);
        assert_eq!(slots.len(), 1);
        assert_eq!(index.get(0, 5), Some(slots[0]));
        assert_eq!(index.get(1, 5), None);
    }

    #[test]
    fn sequential_misses_read_ahead_up_to_the_cap() {
        let mut index = ChunkIndex::new(1000);
        let mut at = 0;
        let mut counts = Vec::new();
        for _ in 0..7 {
            let n = index.miss(0, at, 1000).len() as u64;
            counts.push(n);
            at += n;
        }
        assert_eq!(counts, [1, 2, 4, 8, 16, 16, 16]);
        assert_eq!(index.miss(0, 900, 1000).len(), 1);
    }

    #[test]
    fn read_ahead_stops_at_the_end_of_the_file() {
        let mut index = ChunkIndex::new(1000);
        for at in [0, 1, 3] {
            index.miss(0, at, 7);
        }
        assert_eq!(index.miss(0, 7, 7).len(), 1);
        let mut index = ChunkIndex::new(1000);
        for at in [0, 1, 3] {
            index.miss(0, at, 9);
        }
        assert_eq!(index.miss(0, 7, 9).len(), 2);
    }

    #[test]
    fn the_least_recently_used_chunk_goes_first() {
        let mut index = ChunkIndex::new(2);
        let a = index.miss(0, 10, 100)[0];
        index.miss(0, 20, 100);
        index.get(0, 10);
        let c = index.miss(0, 30, 100)[0];
        assert_ne!(c, a);
        assert!(index.get(0, 10).is_some());
        assert_eq!(index.get(0, 20), None);
        assert!(index.get(0, 30).is_some());
    }

    #[test]
    fn a_span_never_exceeds_the_capacity() {
        let mut index = ChunkIndex::new(3);
        let mut at = 0;
        for _ in 0..5 {
            let slots = index.miss(0, at, 100);
            assert!(slots.len() <= 3);
            at += slots.len() as u64;
        }
        let held = (0..at).filter(|&c| index.get(0, c).is_some()).count();
        assert_eq!(held, 3);
    }

    #[test]
    fn a_forgotten_slot_is_reused_and_a_retired_one_is_not() {
        let mut index = ChunkIndex::new(2);
        let a = index.miss(0, 1, 100)[0];
        index.forget(0, 1);
        assert_eq!(index.get(0, 1), None);
        assert_eq!(index.miss(0, 50, 100)[0], a);

        let b = index.miss(0, 60, 100)[0];
        index.retire(0, 60);
        let c = index.miss(0, 70, 100)[0];
        assert_ne!(c, b);
        assert_eq!(index.get(0, 70), Some(c));
        assert_eq!(index.get(0, 50), None, "capacity shrank to one");
    }

    #[test]
    fn dropping_a_file_leaves_the_others() {
        let mut index = ChunkIndex::new(8);
        index.miss(0, 3, 100);
        index.miss(1, 3, 100);
        index.forget_file(0);
        assert_eq!(index.get(0, 3), None);
        assert!(index.get(1, 3).is_some());
    }
}
