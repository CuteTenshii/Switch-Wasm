//! Profile a title that presents frames but draws nothing, sampling only after
//! the Nth presented frame:
//! `steady_state <container> --keys <prod.keys> [--frame <n>] [--steps <n>]`.
//!
//! Reports samples per thread, per 4 KiB page, per return address and per call stack.
mod common;

use clap::Parser;
use common::profile::Profile;
use common::{Flow, Pace};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    /// Presented frame to start sampling after.
    #[arg(long, default_value_t = 60)]
    frame: u64,
    /// Instructions to sample once the frame is reached.
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: u64,
}

/// Instructions to sample once the frame is reached.
const DEFAULT_STEPS: u64 = 200_000_000;
/// One sample every this many instructions.
const INTERVAL: u64 = 64;
/// One call stack every this many samples.
const STACK_EVERY: u64 = 512;
/// Boot budget for reaching the frame.
const BOOT_BUDGET: u64 = 20_000_000_000;

fn main() {
    let cli = common::parse::<Cli>(|cli| &cli.shared);
    let title = cli.target.open();
    let (want_frame, steps) = (cli.frame, cli.steps);

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    title.boot(&mut cpu);

    let Some(boot) = common::reach_frame(&mut cpu, want_frame, BOOT_BUDGET) else {
        return;
    };
    println!(
        "frame {want_frame} at step {}; sampling the next {steps} instructions",
        boot.steps
    );
    let before = cpu.nv.gpu.stats;

    let mut profile = Profile::new(INTERVAL, STACK_EVERY);
    let run = common::drive(&mut cpu, Pace::Instructions, steps, |cpu, done| {
        profile.tick(cpu, done);
        Flow::Continue
    });
    println!(
        "{} samples over {} instructions",
        profile.samples(),
        run.steps
    );
    profile.report();

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
