//! Boot an `.nro`, `.nsp`, `.xci` or Program `.nca` from the command line.
//!
//! Usage: cargo run -p switch-core --example boot -- <target> [--keys <prod.keys>] [flags]
mod common;

use clap::Parser;
use common::profile::Profile;
use common::Flow;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    #[command(flatten)]
    boot: common::Boot,
    #[command(flatten)]
    report: common::Report,
    #[command(flatten)]
    play: common::Play,
    #[command(flatten)]
    probe: common::Probe,
    /// Instructions to run (default: unlimited for an `.nro`, 2 million for a title).
    #[arg(long, value_name = "N")]
    max_steps: Option<u64>,
    /// Copy this host directory onto the SD card; an NRO inside it is launched from its SD path,
    /// as the page's Files dialog does, so it finds the files beside it.
    #[arg(long, value_name = "DIR")]
    sd: Option<String>,
    /// Print this SD card file when the run ends (a homebrew's own log).
    #[arg(long, value_name = "SD PATH")]
    show: Option<String>,
    /// Sample the pc by thread, page and caller every this many instructions.
    #[arg(long, default_value_t = 0, value_name = "INTERVAL")]
    profile: u64,
    /// Stop at this presented frame.
    #[arg(short = 'e', long, default_value_t = u64::MAX, value_name = "N")]
    stop_frame: u64,
    /// Write the last presented frame here.
    #[arg(long, value_name = "OUT.PPM")]
    shot: Option<String>,
}

fn main() {
    let cli = common::parse::<Cli>(|cli| {
        common::Flags::default()
            .with(&cli.shared)
            .with(&cli.boot)
            .with(&cli.report)
            .with(&cli.play)
            .with(&cli.probe)
    });
    let program = cli.target.open_program();
    let max_steps = cli.max_steps.unwrap_or(if program.is_homebrew() {
        u64::MAX
    } else {
        2_000_000
    });

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    let sd_path = common::copy_sd(&mut cpu, &program, cli.sd.as_deref(), &cli.target);
    let booted = program.boot(&mut cpu, sd_path.as_deref());
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

    let mut debug = common::Debug::from_flags();
    debug.arm(&mut cpu);
    let mut profile = Profile::new(cli.profile, 0);
    let mut press = common::Press::from_flags();
    let pace = debug.pace(profile.is_on());
    let mut draw_gate =
        common::DrawGate::install(&mut cpu, Box::new(switch_core::gpu::renderer::Software));
    let cores = common::start_cores(&mut cpu);
    let started = std::time::Instant::now();
    let mut frames_seen = 0;
    let run = common::drive(&mut cpu, pace, max_steps, |cpu, done| {
        debug.tick(cpu, done);
        if let Some(gate) = draw_gate.as_mut() {
            gate.tick(cpu);
        }
        if cpu.nv.gpu.frames != frames_seen {
            frames_seen = cpu.nv.gpu.frames;
            println!(
                "frame {frames_seen} at {:.2}s, {} steps, {} on core 0",
                started.elapsed().as_secs_f64(),
                cpu.steps,
                cpu.steps - cpu.other_core_steps()
            );
            if frames_seen >= cli.stop_frame {
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
            thread.entry.as_deref().unwrap_or("main"),
            thread.priority,
            thread.state,
            thread.at
        );
    }

    if let Some(out) = &cli.shot {
        if !cpu.nv.gpu.framebuffer.is_empty() {
            common::write_ppm(out, &cpu.nv.gpu.framebuffer);
        }
    }
    profile.report();
    common::print_console(&cpu, 80);
    if let Some(show) = &cli.show {
        match cpu.fs.file(show) {
            Some(bytes) => {
                println!("--- {show} ({} bytes) ---", bytes.len());
                print!("{}", String::from_utf8_lossy(bytes));
            }
            None => println!("--- {show}: not on the SD card ---"),
        }
    }
}
