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

use common::profile::Profile;
use common::Flow;

const USAGE: &str = "boot_nsp <container> <prod.keys> [title.keys] [max_steps]";

fn main() {
    let args = common::container_args(USAGE);
    let max_steps = args.rest_num(0).unwrap_or(2_000_000);
    let title = args.open();
    println!(
        "program {:016x}: {} file(s) in the ExeFS",
        title.nca.program_id,
        title.exefs_pfs0.files.len()
    );

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    let booted = title.boot(&mut cpu);
    if let Some(quota) = booted.quota {
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
    }
    if booted.archives > 0 {
        println!("registered {} system data archive(s)", booted.archives);
    }
    for module in &booted.modules {
        println!(
            "module base={:#010x} entry={:#010x}",
            module.base, module.entry
        );
    }

    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
    let mut profile = Profile::new(common::env_u64("PROFILE", 0), 0);
    let mut press = common::Press::from_env();
    let pace = debug.pace(profile.is_on());
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
        profile.tick(cpu, done);
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
    profile.report();
    common::print_console(&cpu, 80);
}
