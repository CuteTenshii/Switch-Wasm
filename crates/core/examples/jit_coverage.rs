//! Lists the instructions a program runs that the block emitter has no writer for,
//! weighted by one frame's instruction mix:
//! `jit_coverage <target> [--keys <prod.keys>] [--from-frame <n>]`.
mod common;

use clap::Parser;
use common::{Flow, Pace};
use std::collections::HashMap;
use switch_core::cpu::{emits, writes_exit};
use switch_core::disasm::disassemble;

const ROWS: usize = 20;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    #[command(flatten)]
    play: common::Play,
    #[command(flatten)]
    boot: common::Boot,
    #[command(flatten)]
    report: common::Report,
    /// The presented frame to sample; the default skips startup to a steady frame.
    #[arg(short = 's', long, default_value_t = 2, value_name = "N")]
    from_frame: u64,
}

fn written(insn: u32) -> bool {
    emits(insn) || writes_exit(insn)
}

fn percent(part: u64, whole: u64) -> f64 {
    part as f64 * 100.0 / whole as f64
}

/// Executed encodings of the frame after `frame`, with how often each ran, and the frame's length.
fn sample_frame(cpu: &mut switch_core::cpu::Cpu, frame: u64) -> Option<(HashMap<u32, u64>, u64)> {
    common::reach_frame(cpu, frame, u64::MAX)?;

    let mut seen: HashMap<u32, u64> = HashMap::new();
    let start = cpu.nv.gpu.frames;
    let run = common::drive(cpu, Pace::Instructions, u64::MAX, |cpu, _| {
        if cpu.nv.gpu.frames != start {
            return Flow::Stop;
        }
        if let Ok(insn) = cpu.mem.fetch(cpu.get_pc()) {
            *seen.entry(insn).or_default() += 1;
        }
        Flow::Continue
    });
    Some((seen, run.steps))
}

fn report_encodings(missing: &mut [(u64, u32)], total: u64) {
    missing.sort_by_key(|&(count, insn)| (std::cmp::Reverse(count), insn));
    println!("--- hottest encodings with no writer ---");
    for &(count, insn) in missing.iter().take(ROWS) {
        println!(
            "  {insn:#010x}  {count:>10}  {:5.2}%  {}",
            percent(count, total),
            disassemble(insn),
        );
    }
}

/// Rolled up the way `examples/hotspots.rs` reports the mix.
fn report_groups(seen: &HashMap<u32, u64>, missing: &[(u64, u32)], total: u64) {
    let mut ran = [0u64; 256];
    let mut gaps = [0u64; 256];
    for (&insn, &count) in seen {
        ran[(insn >> 24) as usize] += count;
    }
    for &(count, insn) in missing {
        gaps[(insn >> 24) as usize] += count;
    }
    let mut groups: Vec<(u64, u64, usize)> = (0..256)
        .filter(|&top| gaps[top] > 0)
        .map(|top| (gaps[top], ran[top], top))
        .collect();
    groups.sort_by_key(|&(gap, ran, top)| (std::cmp::Reverse(gap), ran, top));
    println!("--- by encoding group (bits 31:24) ---");
    for &(gap, ran, top) in groups.iter().take(ROWS) {
        println!(
            "  {top:#04x}  {gap:>10} of {ran:>10} run  ({:5.1}% of the group, {:5.2}% of the frame)",
            percent(gap, ran),
            percent(gap, total),
        );
    }
}

fn main() {
    let cli = common::parse::<Cli>(|cli| {
        common::Flags::default()
            .with(&cli.shared)
            .with(&cli.play)
            .with(&cli.boot)
            .with(&cli.report)
    });
    let program = cli.target.open_program();
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    program.boot(&mut cpu, None);
    common::start_play(&mut cpu);

    let Some((seen, total)) = sample_frame(&mut cpu, cli.from_frame) else {
        return;
    };
    let mut missing: Vec<(u64, u32)> = seen
        .iter()
        .filter(|(&insn, _)| !written(insn))
        .map(|(&insn, &count)| (count, insn))
        .collect();
    let fallbacks: u64 = missing.iter().map(|&(count, _)| count).sum();

    println!(
        "one frame = {total} instructions, {} distinct encodings",
        seen.len()
    );
    println!(
        "{fallbacks} of them ({:.2}%) have no emitter writer, over {} distinct encodings",
        percent(fallbacks, total),
        missing.len(),
    );
    report_encodings(&mut missing, total);
    report_groups(&seen, &missing, total);
}
