//! Profile a title that presents frames but draws nothing, sampling only after
//! the Nth presented frame:
//! `steady_state <container> <prod.keys> [title.keys] [frame] [steps]`.
//!
//! Reports samples per thread, per 4 KiB page, and per return address.
mod common;

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

const USAGE: &str = "steady_state <container> <prod.keys> [title.keys] [frame] [steps]";

/// Instructions to sample once the frame is reached.
const DEFAULT_STEPS: u64 = 200_000_000;
/// One sample every this many instructions.
const INTERVAL: u64 = 64;
/// One call stack every this many samples.
const STACK_EVERY: u64 = 512;
/// Boot budget for reaching the frame.
const BOOT_BUDGET: u64 = 20_000_000_000;

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
    let before = cpu.nv.gpu.stats;

    let mut pages: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    let mut callers: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    let mut stacks: BTreeMap<(u64, Vec<u32>), u64> = BTreeMap::new();
    let mut sampled = 0u64;
    let run = common::drive(&mut cpu, Pace::Instructions, steps, |cpu, done| {
        if done % INTERVAL == 0 {
            let thread = cpu.current_thread_handle();
            *pages.entry((thread, cpu.get_pc() & !0xFFF)).or_default() += 1;
            *callers.entry((thread, cpu.read_x(30) as u32)).or_default() += 1;
            if sampled.is_multiple_of(STACK_EVERY) {
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

    let after = &cpu.nv.gpu.stats;
    let frames = cpu.nv.gpu.frames - want_frame;
    println!(
        "window: frames +{frames}, draws +{}, clears +{}, copies +{}",
        after.draws - before.draws,
        after.clears - before.clears,
        after.copies - before.copies,
    );
    print!("{}", cpu.thread_dump());
}
