//! Boot a real game from its container, an `.nsp`, an `.xci` or a bare
//! Program `.nca`: find the Program NCA, decrypt the ExeFS, load every module
//! and run it. The CLI equivalent of the browser's panel "Launch" button,
//! useful for debugging without a browser.
//!
//! Usage: cargo run -p switch-core --example boot_nsp -- <container> <prod.keys> [title.keys] [max_steps]
//!
//! `PROFILE=<interval>` samples the pc every `interval` steps and reports
//! where the run spent itself, by thread and by 4 KiB page. A title that runs
//! for billions of instructions without reaching a frame is not stuck
//! anywhere a backtrace can be taken; it is *somewhere*, and this is what
//! says where.
//!
//! `SHOT=<out.ppm>` writes whatever was presented last.
//!
//! `PRESS=<buttons>@<interval>` taps buttons every `interval` steps, such as
//! `PRESS=A@500000000`, or `PRESS=A+DOWN@300000000` for two at once. Much of
//! what a title does only happens once someone gets it past a title screen or
//! a menu, which a run with no input never does.
//!
//! `DUMP=`, `TRAP_WRITE=`, `TRAP_READ=` and `WATCH_PC=` are the debugging
//! knobs every runner here shares. See [`common::Debug`] for their spelling.
mod common;

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

const USAGE: &str = "boot_nsp <container> <prod.keys> [title.keys] [max_steps]";

/// How long a `PRESS=` tap holds its buttons down: three frames of the 1.02
/// GHz CPU at 60 Hz. A title that samples its pad once a frame misses a press
/// shorter than a frame, and one that waits for a release needs one.
const PRESS_HOLD: u64 = 3 * 17_000_000;

/// The buttons `PRESS=` names, in Horizon's `HidNpadButton` order.
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

/// `PRESS=`'s buttons and interval, or `None` when it is unset. Exits with
/// a message on a spelling it cannot read, rather than running without the
/// input that was asked for.
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

/// Where `PROFILE=` found the run: by thread, by page, and by return address.
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

    // The title's save-data quota, which `IApplicationFunctions::GetSaveDataSize`
    // reports. It is declared in the NACP, and the NACP is in the *Control*
    // NCA rather than the Program one booted below, so it has to be read
    // separately, and a container without one leaves the CPU's default in
    // place rather than reporting a size this title never asked for.
    match title.control() {
        Ok(control) => {
            let quota = switch_core::cpu::SaveDataQuota::from(&control.nacp);
            println!(
                "save data: {} bytes (+{} journal), extendable to {} (+{}); \
                 cache storage: {} x {} bytes — all from the NACP",
                quota.size,
                quota.journal_size,
                quota.size_max,
                quota.journal_size_max,
                quota.cache_storage_index_max,
                quota.cache_storage_size_max,
            );
            cpu.set_save_data_quota(quota);
            // The id this title's DLC is numbered from, when its NACP names
            // one rather than leaving it to be derived. Before the boot, which
            // is where add-on content gets mounted.
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
    // Samples per (thread handle, pc page). A page rather than an address
    // because a hot loop is a run of instructions, not one of them, and one
    // bucket per instruction turns a profile into a list.
    let mut pages: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    // The same samples keyed by the return address instead. The hot page of a
    // run that spends itself in `memcpy` says nothing on its own, every
    // caller in the process shares it, and for a leaf like that the link
    // register *is* the caller.
    let mut callers: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    let mut sampled = 0u64;
    let press = press_from_env();
    let mut held = false;
    // Sampling and the watchpoints both read the machine between two
    // instructions. With neither armed the run goes through the block
    // translator, which is the engine the frontend uses.
    let pace = if debug.stepwise() || profile > 0 {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    let run = common::drive(&mut cpu, pace, max_steps, |cpu, done| {
        debug.tick(cpu, done);
        if let Some((buttons, interval)) = press {
            // The first tap waits a whole interval: a press during boot
            // lands before anything is reading the pad.
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
    // A run that stops on its step budget rather than on a fault has almost
    // always stopped making progress, and where each *thread* is says more
    // about why than where the one running thread is.
    print!("{}", cpu.thread_dump());

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
