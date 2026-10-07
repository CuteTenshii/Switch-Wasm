//! Run a retail title with the JIT on and off in lockstep and report the first
//! slice after which the two machines disagree. The interpreter is the
//! reference.
//!
//! Usage: jit_bisect <container> <prod.keys> [title.keys] [max_steps] [window]
//!
//! Exact only until the first thread switch: the two engines can interleave
//! threads differently and both be correct.
mod common;

use switch_core::cpu::Cpu;

const USAGE: &str = "jit_bisect <container> <prod.keys> [title.keys] [max_steps] [window]";

const SLICE: u64 = 4096;

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

/// Returns false if the machine halted or faulted first.
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

/// One fact per line, so a disagreement prints as the differing lines.
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
