//! Boot an `.nsp`, `.xci` or Program `.nca` from the command line.
//!
//! Usage: cargo run -p switch-core --example boot_nsp -- <container> <prod.keys> [title.keys] [max_steps]
//!
//! `PROFILE=<interval>` samples the pc by thread, page and caller.
//! `SHOT=<out.ppm>` writes the last presented frame.
//! `PRESS=<buttons>@<interval>` taps buttons periodically, e.g. `PRESS=A+DOWN@300000000`.
//! `DUMP=`, `TRAP_WRITE=`, `TRAP_READ=` and `WATCH_PC=` are described in [`common::Debug`].
mod common;

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

const USAGE: &str = "boot_nsp <container> <prod.keys> [title.keys] [max_steps]";

/// Three frames at 1.02 GHz and 60 Hz, so a once-per-frame pad poll sees the press.
const PRESS_HOLD: u64 = 3 * 17_000_000;

/// In Horizon's `HidNpadButton` order.
const BUTTONS: [(&str, u64); 16] = [
    ("A", 1 << 0),
    ("B", 1 << 1),
    ("X", 1 << 2),
    ("Y", 1 << 3),
    ("LSTICK", 1 << 4),
    ("RSTICK", 1 << 5),
    ("L", 1 << 6),
    ("R", 1 << 7),
    ("ZL", 1 << 8),
    ("ZR", 1 << 9),
    ("PLUS", 1 << 10),
    ("MINUS", 1 << 11),
    ("LEFT", 1 << 12),
    ("UP", 1 << 13),
    ("RIGHT", 1 << 14),
    ("DOWN", 1 << 15),
];

/// Exits on an unreadable spelling rather than running without the input.
fn press_from_env() -> Option<(u64, u64)> {
    let raw = std::env::var("PRESS").ok()?;
    let fail = |why: &str| -> ! {
        eprintln!("PRESS={raw}: {why}; expected <buttons>@<interval>, e.g. A@500000000");
        std::process::exit(2)
    };
    let (names, interval) = raw.split_once('@').unwrap_or_else(|| fail("no interval"));
    let interval: u64 = interval
        .parse()
        .unwrap_or_else(|_| fail("the interval is not a number"));
    if interval <= PRESS_HOLD {
        fail("the interval must be longer than a press");
    }
    let mut mask = 0;
    for name in names.split('+') {
        let upper = name.trim().to_ascii_uppercase();
        match BUTTONS.iter().find(|(button, _)| *button == upper) {
            Some((_, bit)) => mask |= bit,
            None => fail(&format!("no button named {name:?}")),
        }
    }
    Some((mask, interval))
}

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
    let press = press_from_env();
    let mut held = false;
    // Without sampling or watchpoints the run uses the block translator, like the frontend.
    let pace = if debug.stepwise() || profile > 0 {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    let run = common::drive(&mut cpu, pace, max_steps, |cpu, done| {
        debug.tick(cpu, done);
        if let Some((buttons, interval)) = press {
            // The first tap waits a whole interval so it does not land during boot.
            let down = done >= interval && done % interval < PRESS_HOLD;
            if down != held {
                held = down;
                cpu.set_gamepad_state(if down { buttons } else { 0 }, 0, 0, 0, 0);
            }
        }
        if profile > 0 && done % profile == 0 {
            let thread = cpu.current_thread_handle();
            *pages.entry((thread, cpu.get_pc() & !0xFFF)).or_default() += 1;
            *callers.entry((thread, cpu.read_x(30) as u32)).or_default() += 1;
            sampled += 1;
        }
        Flow::Continue
    });

    common::report(&cpu, &run);
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
