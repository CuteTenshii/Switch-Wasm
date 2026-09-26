//! The block translator against the interpreter, in lockstep, on a retail
//! title: find the first slice of instructions after which the two machines
//! disagree.
//!
//! Usage: jit_bisect <container> <prod.keys> [title.keys] [max_steps] [window]
//!
//! `jit_difftest` compares the two engines once, at the end, on homebrew.
//! That is the wrong shape for a retail title, where a translation bug shows
//! up a billion instructions later as something else entirely: a thread
//! branching into the process-exit stub, a title that quits by itself. This
//! boots the same container twice, the translator on in one and off in the
//! other, runs both in the same 4096-instruction slices `boot_nsp` runs in,
//! and compares them every `window` instructions (a million by default). At
//! the first window that disagrees it boots both again, runs them to the
//! start of that window, and compares after every slice, which names the
//! slice the translator got wrong and prints where each machine stood.
//!
//! What is compared is the running thread's registers, its pc, which thread
//! is running, the clock, and every thread's saved state: a wrong memory
//! write is not compared directly, and is found when something reads it back.
//! The interpreter is the reference.
//!
//! **Exact only until the first thread switch.** The interpreter can be
//! preempted between any two instructions and the translator only between
//! blocks, so once a title has several runnable threads the two machines
//! interleave them differently, and both are correct. A disagreement past
//! that point says the interleavings differ and nothing more; it is the tool
//! for a single-threaded stretch, or for proving a divergence starts before
//! the threads do.
mod common;

use switch_core::cpu::Cpu;

const USAGE: &str = "jit_bisect <container> <prod.keys> [title.keys] [max_steps] [window]";

/// The slice both machines run in, the one `boot_nsp` uses.
const SLICE: u64 = 4096;

/// One machine, booted the way `boot_nsp` boots it.
fn boot(title: &common::Title, jit: bool) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.set_jit_enabled(jit);
    if let Ok(control) = title.control() {
        cpu.set_save_data_quota(switch_core::cpu::SaveDataQuota::from(&control.nacp));
        cpu.set_add_on_content_base_id(control.nacp.add_on_content_base_id);
    }
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    common::register_firmware(&mut cpu, &title.keys);
    title.boot(&mut cpu);
    cpu
}

/// Run `steps` instructions in [`SLICE`]s. Returns false if the machine
/// stopped first, by halting or faulting.
fn advance(cpu: &mut Cpu, steps: u64) -> bool {
    let mut done = 0;
    while done < steps && !cpu.halted {
        match cpu.run(SLICE.min(steps - done)) {
            Ok(report) if report.steps == 0 => return false,
            Ok(report) => done += report.steps,
            Err(e) => {
                println!("  fault after {} steps: {e}", cpu.steps);
                return false;
            }
        }
    }
    !cpu.halted
}

/// Everything compared between the two machines, one fact per line so that
/// a disagreement prints as the lines that differ.
fn state(cpu: &Cpu) -> Vec<String> {
    let mut lines = vec![
        format!(
            "steps={} cycles={} halted={}",
            cpu.steps, cpu.cycles, cpu.halted
        ),
        format!(
            "pc={} thread={:#x}",
            cpu.locate(cpu.get_pc()),
            cpu.current_thread_handle()
        ),
    ];
    for i in 0..=30u8 {
        lines.push(format!("x{i}={:#x}", cpu.read_x(i)));
    }
    lines.extend(cpu.thread_dump().lines().map(str::to_owned));
    lines
}

/// The lines on which the two machines disagree, or none.
fn differences(reference: &[String], translated: &[String]) -> Vec<String> {
    let longest = reference.len().max(translated.len());
    (0..longest)
        .filter_map(|i| {
            let a = reference.get(i).map_or("(missing)", String::as_str);
            let b = translated.get(i).map_or("(missing)", String::as_str);
            (a != b).then(|| format!("  interpreter: {a}\n  translator:  {b}"))
        })
        .collect()
}

fn main() {
    let args = common::container_args(USAGE);
    let max_steps = args.rest_num(0).unwrap_or(2_000_000_000);
    let window = args.rest_num(1).unwrap_or(1_000_000).max(SLICE);
    let title = args.open();

    println!("pass 1: comparing every {window} instructions, up to {max_steps}");
    let mut reference = boot(&title, false);
    let mut translated = boot(&title, true);
    let mut at = 0u64;
    let diverged = loop {
        if at >= max_steps {
            println!("no disagreement in {at} instructions");
            return;
        }
        let a = advance(&mut reference, window);
        let b = advance(&mut translated, window);
        let diff = differences(&state(&reference), &state(&translated));
        if !diff.is_empty() {
            break at;
        }
        at += window;
        if !a || !b {
            println!("both machines stopped at {at} instructions, in agreement");
            return;
        }
    };
    println!(
        "the machines disagree after the window starting at {diverged}; narrowing to one slice"
    );

    println!("pass 2: booting both again and comparing every {SLICE} instructions from {diverged}");
    let mut reference = boot(&title, false);
    let mut translated = boot(&title, true);
    advance(&mut reference, diverged);
    advance(&mut translated, diverged);
    let mut at = diverged;
    loop {
        let before = state(&translated);
        let a = advance(&mut reference, SLICE);
        let b = advance(&mut translated, SLICE);
        let diff = differences(&state(&reference), &state(&translated));
        if !diff.is_empty() {
            println!(
                "first disagreement: the slice of {SLICE} instructions starting at {at}, \
                 entered at {}",
                before[1]
            );
            for line in diff {
                println!("{line}");
            }
            return;
        }
        at += SLICE;
        if !a || !b || at >= diverged + window {
            println!("pass 2 found no disagreement, which pass 1 did: the slicing differs");
            return;
        }
    }
}
