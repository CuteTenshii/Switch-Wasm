//! Run a program with the JIT on and off in lockstep and report the first
//! slice after which the two machines disagree. The interpreter is the
//! reference.
//!
//! Usage: jit_bisect <target> [prod.keys] [title.keys] [max_steps] [window] [font.ttf]
//!
//! The target is an `.nro`, `.nsp`, `.xci` or Program `.nca` (retail needs keys).
//! Exact only until the first thread switch: the two engines can interleave
//! threads differently and both be correct.
mod common;

use switch_core::cpu::Cpu;

const USAGE: &str = "jit_bisect <target> [prod.keys] [title.keys] [max_steps] [window] [font.ttf]";

const SLICE: u64 = 4096;

fn boot(program: &common::Program, jit: bool) -> Cpu {
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    cpu.set_jit_enabled(jit);
    program.boot(&mut cpu);
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

/// FNV-1a, so bulk state fits on one comparable line.
fn checksum(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xCBF2_9CE4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// One fact per line, so a disagreement prints as the differing lines.
fn state(cpu: &Cpu) -> Vec<String> {
    let fb = &cpu.nv.gpu.framebuffer;
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
        format!("nzcv={:#x} sp={:#x}", cpu.nzcv(), cpu.sp()),
        format!(
            "console: {} bytes, checksum {:#x}",
            cpu.out.len(),
            checksum(cpu.out.iter().copied())
        ),
        format!(
            "frames={} framebuffer {}x{} checksum {:#x}",
            cpu.nv.gpu.frames,
            fb.width,
            fb.height,
            checksum(fb.pixels.iter().flat_map(|p| p.to_le_bytes()))
        ),
    ];
    for i in 0..=30u8 {
        lines.push(format!("x{i}={:#x}", cpu.read_x(i)));
    }
    for i in 0..32u8 {
        lines.push(format!("v{i}={:#x}", cpu.read_vreg(i)));
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

/// The translator's work counters over the run so far.
fn print_jit_stats(cpu: &Cpu) {
    let stats = cpu.jit_stats();
    let per = |n: u64, of: u64| if of == 0 { 0.0 } else { n as f64 / of as f64 };
    println!(
        "translator: {} blocks, {} translated, {} entered ({:.0}x each), {} invalidated; \
         {} instructions fell back to the interpreter ({:.2}% of the run)",
        stats.blocks,
        stats.translated,
        stats.executed,
        per(stats.executed, stats.translated),
        stats.invalidated,
        stats.interpreted,
        per(stats.interpreted, cpu.steps) * 100.0,
    );
}

fn main() {
    let args = common::program_args(USAGE);
    let max_steps = args.rest_num(0).unwrap_or(2_000_000_000);
    let window = args.rest_num(1).unwrap_or(1_000_000).max(SLICE);
    let program = args.open_program();

    println!("pass 1: comparing every {window} instructions, up to {max_steps}");
    let mut reference = boot(&program, false);
    let mut translated = boot(&program, true);
    let mut at = 0u64;
    let diverged = loop {
        if at >= max_steps {
            println!("no disagreement in {at} instructions");
            print_jit_stats(&translated);
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
            print_jit_stats(&translated);
            return;
        }
    };
    println!(
        "the machines disagree after the window starting at {diverged}; narrowing to one slice"
    );

    println!("pass 2: booting both again and comparing every {SLICE} instructions from {diverged}");
    let mut reference = boot(&program, false);
    let mut translated = boot(&program, true);
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
