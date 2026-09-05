//! What one frame costs, counted rather than timed:
//! `frame_work <target> [prod.keys] [title.keys] [frames] [font.ttf]`.
//!
//! A frame's cost is how much work it asks for multiplied by what that work
//! costs on the machine running it. Only the first half is a fact about this
//! emulator: the second is a fact about a compiler and a CPU, and this project
//! ships to a browser, where both are different and the ratio between them is
//! not a constant anyone can divide out. So this reports the first half.
//! Instructions retired, blocks entered, ops handed back to the interpreter,
//! methods dispatched, draws, clears, copies, pixels scanned out. Every one of
//! those is the same number under rustc's x86-64 backend and under V8, and
//! every optimisation worth making moves one of them.
//!
//! Read it against `tools/wasm_bench.mjs`, which times the artefact the
//! browser actually runs. Work counts say what to fix and prove a fix landed;
//! the wasm timing says what it was worth. A change that moves no count here
//! but looks faster on the host made this host faster, which is not the
//! project.
//!
//! Startup is skipped: the first frames of a program are its loader, its
//! allocator and its first upload, and averaging those into a steady frame
//! describes neither.
//!
//! The target is a homebrew `.nro` or a retail container, an `.nsp`, an
//! `.xci` or a bare Program `.nca`, which needs its keys after it. An NRO is
//! not the workload a retail title is, so a ranking taken from one does not
//! transfer.
mod common;

const USAGE: &str = "frame_work <target> [prod.keys] [title.keys] [frames] [font.ttf]";

use switch_core::cpu::Cpu;
use switch_core::gpu::exec::GpuStats;

/// Frames of startup to run through before the window opens.
const WARMUP_FRAMES: u64 = 2;
/// How long to allow for reaching a frame before giving up.
const FRAME_BUDGET: u64 = 2_000_000_000;

/// Every counter this reports, at one instant.
struct Snapshot {
    steps: u64,
    entered: u64,
    linked: u64,
    translated: u64,
    invalidated: u64,
    interpreted: u64,
    gpu: GpuStats,
    pixels: u64,
}

impl Snapshot {
    fn of(cpu: &Cpu, steps: u64) -> Snapshot {
        let jit = cpu.jit_stats();
        let fb = &cpu.nv.gpu.framebuffer;
        Snapshot {
            steps,
            entered: jit.executed,
            linked: jit.linked,
            translated: jit.translated,
            invalidated: jit.invalidated,
            interpreted: jit.interpreted,
            gpu: cpu.nv.gpu.stats,
            pixels: u64::from(fb.width) * u64::from(fb.height),
        }
    }
}

/// What happened between two snapshots.
struct Delta {
    steps: u64,
    entered: u64,
    linked: u64,
    translated: u64,
    invalidated: u64,
    interpreted: u64,
    submissions: u64,
    methods: u64,
    inert_methods: u64,
    draws: u64,
    draws_skipped: u64,
    clears: u64,
    copies: u64,
    macros: u64,
    dispatches: u64,
    dispatches_skipped: u64,
    pixels: u64,
}

impl Delta {
    fn between(before: &Snapshot, after: &Snapshot) -> Delta {
        Delta {
            steps: after.steps - before.steps,
            entered: after.entered - before.entered,
            linked: after.linked - before.linked,
            translated: after.translated - before.translated,
            invalidated: after.invalidated - before.invalidated,
            interpreted: after.interpreted - before.interpreted,
            submissions: after.gpu.submissions - before.gpu.submissions,
            methods: after.gpu.methods - before.gpu.methods,
            inert_methods: after.gpu.inert_methods - before.gpu.inert_methods,
            draws: after.gpu.draws - before.gpu.draws,
            draws_skipped: after.gpu.draws_skipped - before.gpu.draws_skipped,
            clears: after.gpu.clears - before.gpu.clears,
            copies: after.gpu.copies - before.gpu.copies,
            macros: after.gpu.macros - before.gpu.macros,
            dispatches: after.gpu.dispatches - before.gpu.dispatches,
            dispatches_skipped: after.gpu.dispatches_skipped - before.gpu.dispatches_skipped,
            // The frame that was just presented, not a difference: scan-out
            // walks the whole surface every time whatever the last one held.
            pixels: after.pixels,
        }
    }

    fn add(&mut self, other: &Delta) {
        self.steps += other.steps;
        self.entered += other.entered;
        self.linked += other.linked;
        self.translated += other.translated;
        self.invalidated += other.invalidated;
        self.interpreted += other.interpreted;
        self.submissions += other.submissions;
        self.methods += other.methods;
        self.inert_methods += other.inert_methods;
        self.draws += other.draws;
        self.draws_skipped += other.draws_skipped;
        self.clears += other.clears;
        self.copies += other.copies;
        self.macros += other.macros;
        self.dispatches += other.dispatches;
        self.dispatches_skipped += other.dispatches_skipped;
        self.pixels += other.pixels;
    }

    fn zero() -> Delta {
        Delta {
            steps: 0,
            entered: 0,
            linked: 0,
            translated: 0,
            invalidated: 0,
            interpreted: 0,
            submissions: 0,
            methods: 0,
            inert_methods: 0,
            draws: 0,
            draws_skipped: 0,
            clears: 0,
            copies: 0,
            macros: 0,
            dispatches: 0,
            dispatches_skipped: 0,
            pixels: 0,
        }
    }
}

/// Run until one more frame is presented, and report the counters it moved.
fn frame(cpu: &mut Cpu, total_steps: &mut u64) -> Option<Delta> {
    let before = Snapshot::of(cpu, *total_steps);
    let target = cpu.nv.gpu.frames + 1;
    let run = common::run_to(cpu, FRAME_BUDGET, |cpu| cpu.nv.gpu.frames >= target);
    *total_steps += run.steps;
    if cpu.nv.gpu.frames < target {
        return None;
    }
    Some(Delta::between(&before, &Snapshot::of(cpu, *total_steps)))
}

fn main() {
    let args = common::program_args(USAGE);
    let program = args.open_program();
    let frames = args.rest_num(0).unwrap_or(30).max(1);

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    program.boot(&mut cpu);

    let boot = common::run_to(&mut cpu, FRAME_BUDGET, |cpu| {
        cpu.nv.gpu.frames >= WARMUP_FRAMES
    });
    if cpu.nv.gpu.frames < WARMUP_FRAMES {
        println!(
            "never presented {WARMUP_FRAMES} frames: stopped at {} after {} steps",
            cpu.nv.gpu.frames, boot.steps
        );
        common::report(&cpu, &boot);
        return;
    }
    println!(
        "startup: {} instructions to frame {WARMUP_FRAMES}",
        boot.steps
    );
    println!(
        "{:>5}  {:>11}  {:>10}  {:>8}  {:>11}  {:>6}  {:>6}  {:>6}  {:>9}",
        "frame", "insns", "entries", "new", "interpreted", "draws", "clears", "copies", "methods"
    );

    let mut total = Delta::zero();
    let mut counted = 0u64;
    let mut steps = boot.steps;
    for n in 0..frames {
        let Some(delta) = frame(&mut cpu, &mut steps) else {
            println!("stopped after {counted} frames: no further frame was presented");
            break;
        };
        println!(
            "{:>5}  {:>11}  {:>10}  {:>8}  {:>11}  {:>6}  {:>6}  {:>6}  {:>9}",
            WARMUP_FRAMES + n,
            delta.steps,
            delta.entered,
            delta.translated,
            delta.interpreted,
            delta.draws,
            delta.clears,
            delta.copies,
            delta.methods,
        );
        total.add(&delta);
        counted += 1;
    }
    if counted == 0 {
        return;
    }

    let mean = |n: u64| n as f64 / counted as f64;
    let share = |n: u64, of: u64| {
        if of == 0 {
            0.0
        } else {
            n as f64 * 100.0 / of as f64
        }
    };
    println!("--- per frame, over {counted} frames ---");
    println!(
        "  cpu:    {:.0} instructions, {:.0} block entries ({:.1} instructions each), {:.0} newly translated, {:.0} invalidated",
        mean(total.steps),
        mean(total.entered),
        total.steps as f64 / total.entered.max(1) as f64,
        mean(total.translated),
        mean(total.invalidated),
    );
    println!(
        "  cpu:    {:.0} of those entries ({:.1}%) came from the previous block's link",
        mean(total.linked),
        share(total.linked, total.entered),
    );
    // The share the translator did not translate. It is the same share in the
    // browser, and it is the largest CPU-side number here that a change can
    // actually move.
    println!(
        "  cpu:    {:.0} instructions fell back to the interpreter ({:.2}% of the frame)",
        mean(total.interpreted),
        share(total.interpreted, total.steps),
    );
    println!(
        "  gpu:    {:.1} submissions, {:.0} methods ({:.1}% inert), {:.1} macros",
        mean(total.submissions),
        mean(total.methods),
        share(total.inert_methods, total.methods),
        mean(total.macros),
    );
    println!(
        "  gpu:    {:.1} draws ({:.0} skipped), {:.1} clears, {:.1} copies, {:.1} dispatches ({:.0} skipped)",
        mean(total.draws),
        mean(total.draws_skipped),
        mean(total.clears),
        mean(total.copies),
        mean(total.dispatches),
        mean(total.dispatches_skipped),
    );
    // Scan-out runs whatever the frame drew, so a title issuing no draws at
    // all still pays this one in full. See `examples/present_work.rs`.
    println!("  scan-out: {:.0} pixels", mean(total.pixels));

    let used = cpu.mem.mapped_bytes();
    println!(
        "  memory: {:.1} MiB of {} MiB backed",
        used as f64 / (1024.0 * 1024.0),
        cpu.mem.max_mapped_bytes() / (1024 * 1024),
    );
}
