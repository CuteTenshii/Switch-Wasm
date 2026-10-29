//! Profile a title that presents frames but draws nothing, sampling only after
//! the Nth presented frame:
//! `steady_state <container> <prod.keys> [title.keys] [frame] [steps]`.
//!
//! Reports samples per thread, per 4 KiB page, per return address and per call stack.
mod common;

use common::profile::Profile;
use common::{Flow, Pace};
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

fn main() {
    let args = common::container_args(USAGE);
    let title = args.open();
    let want_frame = args.rest_num(0).unwrap_or(60);
    let steps = args.rest_num(1).unwrap_or(DEFAULT_STEPS);

    let mut cpu = Cpu::new();
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
