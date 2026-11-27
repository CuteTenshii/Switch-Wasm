use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

/// Samples the pc every `interval` instructions, per thread: by 4 KiB page,
/// by return address (which names the caller of a hot leaf like `memcpy`),
/// and optionally by call stack. Needs a [`super::Pace::Instructions`] tick.
pub struct Profile {
    interval: u64,
    /// One call stack every this many samples; 0 takes none.
    stack_every: u64,
    sampled: u64,
    pages: BTreeMap<(u64, u32), u64>,
    callers: BTreeMap<(u64, u32), u64>,
    stacks: BTreeMap<(u64, Vec<u32>), u64>,
}

/// Rows per table in the report.
const ROWS: usize = 20;
const STACK_ROWS: usize = 8;
const STACK_DEPTH: usize = 10;

impl Profile {
    /// An `interval` of 0 samples nothing.
    pub fn new(interval: u64, stack_every: u64) -> Profile {
        Profile {
            interval,
            stack_every,
            sampled: 0,
            pages: BTreeMap::new(),
            callers: BTreeMap::new(),
            stacks: BTreeMap::new(),
        }
    }

    pub fn is_on(&self) -> bool {
        self.interval > 0
    }

    pub fn samples(&self) -> u64 {
        self.sampled
    }

    /// Call once per instruction with the count retired so far.
    pub fn tick(&mut self, cpu: &Cpu, done: u64) {
        if self.interval == 0 || !done.is_multiple_of(self.interval) {
            return;
        }
        let thread = cpu.current_thread_handle();
        *self
            .pages
            .entry((thread, cpu.get_pc() & !0xFFF))
            .or_default() += 1;
        *self
            .callers
            .entry((thread, cpu.read_x(30) as u32))
            .or_default() += 1;
        if self.stack_every > 0 && self.sampled.is_multiple_of(self.stack_every) {
            *self
                .stacks
                .entry((thread, cpu.backtrace(STACK_DEPTH)))
                .or_default() += 1;
        }
        self.sampled += 1;
    }

    pub fn report(&self) {
        if self.sampled == 0 {
            return;
        }
        let share = |count: u64, of: u64| count as f64 * 100.0 / of as f64;
        println!("--- profile: {} samples ---", self.sampled);
        let mut by_thread: BTreeMap<u64, u64> = BTreeMap::new();
        for ((thread, _), count) in &self.pages {
            *by_thread.entry(*thread).or_default() += count;
        }
        for (count, thread) in ranked(&by_thread).into_iter().take(STACK_ROWS) {
            println!("  thread {thread:#x}: {:.1}%", share(count, self.sampled));
        }

        println!("--- by page ---");
        for (count, (thread, page)) in ranked(&self.pages).into_iter().take(ROWS) {
            println!(
                "  {:5.1}%  thread {thread:#x}  {page:#010x}..{:#010x}",
                share(count, self.sampled),
                page + 0x1000,
            );
        }

        println!("--- by return address ---");
        for (count, (thread, at)) in ranked(&self.callers).into_iter().take(ROWS) {
            println!(
                "  {:5.1}%  thread {thread:#x}  lr {at:#010x}",
                share(count, self.sampled)
            );
        }

        if self.stacks.is_empty() {
            return;
        }
        let taken: u64 = self.stacks.values().sum();
        println!("--- by call stack ({taken} stacks) ---");
        for (count, (thread, stack)) in ranked(&self.stacks).into_iter().take(STACK_ROWS) {
            let frames: Vec<String> = stack.iter().map(|pc| format!("{pc:#010x}")).collect();
            println!(
                "  {:5.1}%  thread {thread:#x}  {}",
                share(count, taken),
                frames.join(" <- ")
            );
        }
    }
}

/// `(count, key)`, most frequent first.
fn ranked<K: Clone + Ord>(counts: &BTreeMap<K, u64>) -> Vec<(u64, K)> {
    let mut rows: Vec<(u64, K)> = counts.iter().map(|(k, &n)| (n, k.clone())).collect();
    rows.sort_unstable_by(|a, b| b.cmp(a));
    rows
}
