//! Boot an `.nsp`, `.xci` or Program `.nca` from the command line.
//!
//! Usage: cargo run -p switch-core --example boot_nsp -- <container> <prod.keys> [title.keys] [max_steps]
//!
//! `PROFILE=<interval>` samples the pc by thread, page and caller. `CORES=<n>` runs
//! n more guest cores on host threads; `STOP_FRAME=<n>` stops at frame n; `NO_DRAW=1`
//! draws nothing.
//! `SHOT=<out.ppm>` writes the last presented frame.
//! `PRESS=<buttons>@<interval>` taps buttons periodically, e.g. `PRESS=A+DOWN@300000000`.
//! `DUMP=`, `TRAP_WRITE=`, `TRAP_READ=` and `WATCH_PC=` are described in [`common::Debug`].
mod common;

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

const USAGE: &str = "boot_nsp <container> <prod.keys> [title.keys] [max_steps]";

fn report_profile(
    pages: &BTreeMap<(u64, u32), u64>,
    callers: &BTreeMap<(u64, u32), u64>,
    sampled: u64,
) {
    let share = |count: u64| count as f64 * 100.0 / sampled as f64;
    let mut ranked: Vec<_> = pages.iter().map(|(&key, &count)| (count, key)).collect();
    ranked.sort_unstable_by(|a, b| b.cmp(a));

    println!("--- profile: {sampled} samples ---");
    let mut by_thread: BTreeMap<u64, u64> = BTreeMap::new();
    for (count, (thread, _)) in &ranked {
        *by_thread.entry(*thread).or_default() += count;
    }
    let mut threads: Vec<_> = by_thread.into_iter().map(|(t, c)| (c, t)).collect();
    threads.sort_unstable_by(|a, b| b.cmp(a));
    for (count, thread) in threads.iter().take(8) {
        println!("  thread {thread:#x}: {:.1}%", share(*count));
    }
    for (count, (thread, page)) in ranked.iter().take(16) {
        println!(
            "  {:5.1}%  thread {thread:#x}  {page:#010x}..{:#010x}",
            share(*count),
            page + 0x1000,
        );
    }

    let mut by_caller: Vec<_> = callers.iter().map(|(&key, &count)| (count, key)).collect();
    by_caller.sort_unstable_by(|a, b| b.cmp(a));
    println!("--- by return address ---");
    for (count, (thread, at)) in by_caller.iter().take(16) {
        println!(
            "  {:5.1}%  thread {thread:#x}  lr {at:#010x}",
            share(*count)
        );
    }
}

fn main() {
    let args = common::container_args(USAGE);
    let max_steps = args.rest_num(0).unwrap_or(2_000_000);
    let title = args.open();
    println!(
        "program {:016x}: {} file(s) in the ExeFS",
        title.nca.program_id,
        title.exefs_pfs0.files.len()
    );

    let mut cpu = Cpu::new();
    cpu.bootstrap();

    // The save-data quota comes from the Control NCA's NACP.
    match title.control() {
        Ok(control) => {
            let quota = switch_core::cpu::SaveDataQuota::from(&control.nacp);
            println!(
                "save data: {} bytes (+{} journal), extendable to {} (+{}); \
                 cache storage: {} x {} bytes, all from the NACP",
                quota.size,
                quota.journal_size,
                quota.size_max,
                quota.journal_size_max,
                quota.cache_storage_index_max,
                quota.cache_storage_size_max,
            );
            cpu.set_save_data_quota(quota);
            cpu.set_add_on_content_base_id(control.nacp.add_on_content_base_id);
        }
        Err(e) => println!("no control data ({e}): using default save sizes"),
    }

    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    let registered = common::register_firmware(&mut cpu, &title.keys);
    if registered > 0 {
        println!("registered {registered} system data archive(s)");
    }
    for module in title.boot(&mut cpu) {
        println!(
            "module base={:#010x} entry={:#010x}",
            module.base, module.entry
        );
    }

    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
    let profile = common::env_u64("PROFILE", 0);
    let mut pages: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    // Keyed by return address, which names the caller of a hot leaf like `memcpy`.
    let mut callers: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    let mut sampled = 0u64;
    let mut press = common::Press::from_env();
    // Without sampling or watchpoints the run uses the block translator, like the frontend.
    let pace = if debug.stepwise() || profile > 0 {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    common::apply_no_draw(&mut cpu);
    let cores = common::start_cores(&mut cpu);
    let stop_frame = common::env_u64("STOP_FRAME", u64::MAX);
    let started = std::time::Instant::now();
    let mut frames_seen = 0;
    let run = common::drive(&mut cpu, pace, max_steps, |cpu, done| {
        debug.tick(cpu, done);
        if cpu.nv.gpu.frames != frames_seen {
            frames_seen = cpu.nv.gpu.frames;
            println!(
                "frame {frames_seen} at {:.2}s, {} steps, {} on core 0",
                started.elapsed().as_secs_f64(),
                cpu.steps,
                cpu.steps - cpu.other_core_steps()
            );
            if frames_seen >= stop_frame {
                return Flow::Stop;
            }
        }
        if let Some(press) = press.as_mut() {
            press.tick(cpu, done);
        }
        if profile > 0 && done % profile == 0 {
            let thread = cpu.current_thread_handle();
            *pages.entry((thread, cpu.get_pc() & !0xFFF)).or_default() += 1;
            *callers.entry((thread, cpu.read_x(30) as u32)).or_default() += 1;
            sampled += 1;
        }
        Flow::Continue
    });

    let wall = started.elapsed();
    common::stop_cores(&mut cpu, cores);
    common::report(&cpu, &run);
    println!("ran for {:.1}s", wall.as_secs_f64());
    debug.report();
    debug.stop_state(&cpu);
    print!("{}", cpu.thread_dump());
    let (threads, _, _) = cpu.take_thread_report();
    for thread in threads {
        println!(
            "  thread {} ({}, runs {}, priority {}): {}; {}",
            thread.index,
            thread.name.as_deref().unwrap_or("unnamed"),
            thread.entry,
            thread.priority,
            thread.state,
            thread.at
        );
    }

    if let Ok(out) = std::env::var("SHOT") {
        if !cpu.nv.gpu.framebuffer.is_empty() {
            common::write_ppm(&out, &cpu.nv.gpu.framebuffer);
        }
    }
    if sampled > 0 {
        report_profile(&pages, &callers, sampled);
    }
    println!("--- program console output ({} bytes) ---", cpu.out.len());
    for line in String::from_utf8_lossy(&cpu.out).lines().take(80) {
        println!("  {line}");
    }
}
