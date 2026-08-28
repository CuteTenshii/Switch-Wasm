//! Where a title that presents frames but draws nothing spends its time once
//! it has stopped getting anywhere:
//! `steady_state <nsp> <prod.keys> [title.keys] [frame] [steps]`.
//!
//! `boot_nsp`'s `PROFILE=` samples a whole run, and for a title that spends a
//! billion instructions booting and then stalls, that is a profile of the part
//! that *worked*: Just Dance 2019's came back 23% zlib `adler32`, which turned
//! out to be four calls that all finished before its first frame. The stall
//! itself was a rounding error in the same numbers.
//!
//! So this skips the boot — running it through the block translator, which is
//! the only fast way to get there — and starts sampling at the Nth presented
//! frame. Everything it reports happened after the title stopped making
//! progress, which is the only part that says why.
//!
//! Reported per thread, per 4 KiB page and per return address. A page rather
//! than an address because a loop is a run of instructions and one bucket each
//! turns a profile into a list; a return address as well because the hot page
//! of a spin is usually a leaf that says nothing about who is spinning.
mod common;

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

const USAGE: &str = "steady_state <nsp> <prod.keys> [title.keys] [frame] [steps]";

/// How many instructions to sample once the frame is reached. Enough to cover
/// a few seconds of a stalled title's own loop.
const DEFAULT_STEPS: u64 = 200_000_000;
/// One sample every this many instructions.
const INTERVAL: u64 = 64;
/// One call stack every this many samples.
const STACK_EVERY: u64 = 512;
/// How much of the boot to allow before giving up on reaching the frame.
const BOOT_BUDGET: u64 = 20_000_000_000;

/// Print one ranked table of samples, as percentages of `total`.
fn report(title: &str, counts: &BTreeMap<(u64, u32), u64>, total: u64, rows: usize, label: &str) {
    let mut ranked: Vec<_> = counts
        .iter()
        .map(|(&(thread, at), &count)| (count, thread, at))
        .collect();
    ranked.sort_unstable_by(|a, b| b.cmp(a));
    println!("--- {title} ---");
    for (count, thread, at) in ranked.iter().take(rows) {
        println!(
            "  {:5.1}%  thread {thread:#x}  {label} {at:#010x}",
            *count as f64 * 100.0 / total as f64
        );
    }
}

fn main() {
    let args = common::container_args(USAGE);
    let title = args.open();
    let want_frame = args.rest_num(0).unwrap_or(60);
    let steps = args.rest_num(1).unwrap_or(DEFAULT_STEPS);

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    common::register_firmware(&mut cpu, &title.keys);
    title.boot(&mut cpu);

    // Nothing is sampled here, so the boot runs through the block translator
    // rather than the interpreter — the difference is 1.8x on real code, and
    // a title's boot is measured in billions of instructions.
    let boot = common::run_to(&mut cpu, BOOT_BUDGET, |cpu| cpu.nv.gpu.frames >= want_frame);
    if cpu.nv.gpu.frames < want_frame {
        println!(
            "never reached frame {want_frame}: stopped at {} after {} steps",
            cpu.nv.gpu.frames, boot.steps
        );
        common::report(&cpu, &boot);
        return;
    }
    println!(
        "frame {want_frame} at step {}; sampling the next {steps} instructions",
        boot.steps
    );
    let before = cpu.nv.gpu.stats.clone();

    let mut pages: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    let mut callers: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    // Whole call stacks, sampled far more rarely than the pc: walking frame
    // pointers costs more than reading a register, and what a stack answers is
    // "which loop is this" rather than "how hot is it" — a question a few
    // thousand samples settle as well as a few million.
    let mut stacks: BTreeMap<(u64, Vec<u32>), u64> = BTreeMap::new();
    let mut sampled = 0u64;
    // Per-instruction pacing, because a sample has to be taken *between* two
    // instructions to read the machine at all.
    let run = common::drive(&mut cpu, Pace::Instructions, steps, |cpu, done| {
        if done % INTERVAL == 0 {
            let thread = cpu.current_thread_handle();
            *pages.entry((thread, cpu.get_pc() & !0xFFF)).or_default() += 1;
            *callers.entry((thread, cpu.read_x(30) as u32)).or_default() += 1;
            if sampled % STACK_EVERY == 0 {
                *stacks.entry((thread, cpu.backtrace(10))).or_default() += 1;
            }
            sampled += 1;
        }
        Flow::Continue
    });

    let mut by_thread: BTreeMap<u64, u64> = BTreeMap::new();
    for ((thread, _), count) in &pages {
        *by_thread.entry(*thread).or_default() += count;
    }
    let mut threads: Vec<_> = by_thread.into_iter().map(|(t, c)| (c, t)).collect();
    threads.sort_unstable_by(|a, b| b.cmp(a));
    println!("--- {sampled} samples over {} instructions ---", run.steps);
    for (count, thread) in threads.iter().take(8) {
        println!(
            "  thread {thread:#x}: {:.1}%",
            *count as f64 * 100.0 / sampled as f64
        );
    }
    report("by page", &pages, sampled, 20, "");
    report("by return address", &callers, sampled, 20, "lr");

    let taken: u64 = stacks.values().sum();
    let mut ranked: Vec<_> = stacks
        .iter()
        .map(|((thread, stack), count)| (count, thread, stack))
        .collect();
    ranked.sort_unstable_by(|a, b| b.0.cmp(a.0));
    println!("--- by call stack ({taken} stacks) ---");
    for (count, thread, stack) in ranked.iter().take(8) {
        println!(
            "  {:5.1}%  thread {thread:#x}  {}",
            **count as f64 * 100.0 / taken as f64,
            stack
                .iter()
                .map(|pc| format!("{pc:#010x}"))
                .collect::<Vec<_>>()
                .join(" <- ")
        );
    }

    // What the GPU did *during the window*, which is the whole question a
    // stalled title raises: frames that carry no draw are frames the title is
    // not drawing, however many of them there are.
    let after = &cpu.nv.gpu.stats;
    let frames = cpu.nv.gpu.frames - want_frame;
    println!(
        "window: frames +{frames}, draws +{}, clears +{}, copies +{}",
        after.draws - before.draws,
        after.clears - before.clears,
        after.copies - before.copies,
    );
    // What the window cost is not reported here. It would be a wall clock on
    // this host, and the frontend's frame rate is made in a browser, out of
    // the same work run through a different compiler — `tools/wasm_bench.mjs`
    // times that one. What a stall costs in *work* is above, and
    // `examples/frame_work.rs` reports it per frame.
    print!("{}", cpu.thread_dump());
}
