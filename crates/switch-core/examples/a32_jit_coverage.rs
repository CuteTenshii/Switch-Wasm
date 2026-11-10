//! Lists the instructions a 32-bit title runs that the A32 emitter has no writer for,
//! weighted by a window of the running title's instruction mix:
//! `a32_jit_coverage <target> [--keys <prod.keys>] [--skip <steps>] [--window <steps>]`.
mod common;

use clap::Parser;
use common::{Flow, Pace};
use std::collections::HashMap;
use switch_core::cpu::{a32_emits, a32_translates};

const ROWS: usize = 25;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    /// Instructions to run before measuring.
    #[arg(long, default_value_t = 5_000_000_000)]
    skip: u64,
    /// Instructions to measure.
    #[arg(long, default_value_t = 100_000_000)]
    window: u64,
}

/// The condition and register fields cleared, so one instruction class is one key.
fn class(word: u32) -> u32 {
    word & 0x0FF0_00F0
}

fn main() {
    let cli = common::parse::<Cli>(|cli| &cli.shared);
    let program = cli.target.open_program();
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    program.boot(&mut cpu);
    let mut next_report = 0;
    common::drive(&mut cpu, Pace::Blocks, cli.skip, |_, steps| {
        if steps >= next_report {
            eprintln!("{steps} steps");
            next_report = steps + 500_000_000;
        }
        Flow::Continue
    });

    let mut seen: HashMap<u32, u64> = HashMap::new();
    let run = common::drive(&mut cpu, Pace::Instructions, cli.window, |cpu, _| {
        if let Ok(word) = cpu.mem.read_u32(cpu.get_pc()) {
            *seen.entry(word).or_default() += 1;
        }
        Flow::Continue
    });

    let total = run.steps;
    let share = |count: u64| count as f64 * 100.0 / total as f64;
    let interpreted: u64 = seen
        .iter()
        .filter(|(&word, _)| !a32_translates(word))
        .map(|(_, &count)| count)
        .sum();
    let unwritten: u64 = seen
        .iter()
        .filter(|(&word, _)| !a32_emits(word))
        .map(|(_, &count)| count)
        .sum();
    println!(
        "window = {total} instructions, {} distinct words",
        seen.len()
    );
    println!(
        "{interpreted} ({:.2}%) decode to the interpreter, {unwritten} ({:.2}%) have no emitter writer",
        share(interpreted),
        share(unwritten)
    );

    let mut classes: HashMap<u32, (u64, u32)> = HashMap::new();
    for (&word, &count) in seen.iter().filter(|(&word, _)| !a32_emits(word)) {
        let entry = classes.entry(class(word)).or_insert((0, word));
        entry.0 += count;
        if count > seen[&entry.1] {
            entry.1 = word;
        }
    }
    let mut rows: Vec<_> = classes.into_iter().collect();
    rows.sort_by_key(|&(key, (count, _))| (std::cmp::Reverse(count), key));
    println!("--- hottest classes with no writer (bits 27:20 and 7:4) ---");
    for (key, (count, example)) in rows.iter().take(ROWS) {
        println!(
            "  {key:#010x}  {count:>10}  {:5.2}%  e.g. {example:#010x}",
            share(*count)
        );
    }
}
