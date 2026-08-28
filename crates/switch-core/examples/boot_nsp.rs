//! Boot a real game from its container: find the Program NCA, decrypt the
//! ExeFS, load every module and run it — the CLI equivalent of the browser's
//! NSP/NCA panel "Launch" button, useful for debugging without a browser.
//!
//! Usage: cargo run -p switch-core --example boot_nsp -- <path.nsp|path.nca> <prod.keys> [title.keys] [max_steps]
//!
//! `PROFILE=<interval>` samples the pc every `interval` steps and reports
//! where the run spent itself, by thread and by 4 KiB page. A title that runs
//! for billions of instructions without reaching a frame is not stuck
//! anywhere a backtrace can be taken; it is *somewhere*, and this is what
//! says where.
//!
//! `SHOT=<out.ppm>` writes whatever was presented last.
//!
//! `DUMP=`, `TRAP_WRITE=`, `TRAP_READ=` and `WATCH_PC=` are the debugging
//! knobs every runner here shares — see [`common::Debug`] for their spelling.
mod common;

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

const USAGE: &str = "boot_nsp <path.nsp|path.nca> <prod.keys> [title.keys] [max_steps]";

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
    // NCA rather than the Program one booted below — so it has to be read
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
    // run that spends itself in `memcpy` says nothing on its own — every
    // caller in the process shares it — and for a leaf like that the link
    // register *is* the caller.
    let mut callers: BTreeMap<(u64, u32), u64> = BTreeMap::new();
    let mut sampled = 0u64;
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
