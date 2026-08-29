//! Where a frame's emulated instructions go:
//! `hotspots <target> [prod.keys] [title.keys] [font.ttf]`.
//!
//! Skips startup, then counts every instruction of one steady-state frame by
//! address, by top-level encoding byte, and by the group the block translator
//! dispatches on. The address histogram says which guest function to blame
//! (hbmenu spends most of a frame in its own software gradient fill, not in
//! the emulator); the encoding histograms say which decoder paths are worth
//! optimising.
//!
//! The group table is the one that ranks work inside the translator, because
//! its rows are exactly the arms of `jit::decode`: loads and stores are the
//! ops that walk the page table, and SIMD/FP are the ops that still carry a
//! raw instruction word and re-derive their operands on every execution.
//! `jit_coverage` measures the instructions the translator has *no* op for,
//! which on both a homebrew and a retail title is under 1.5%; this measures
//! the ones it has an op for and still charges for.
//!
//! The target is a homebrew `.nro` or a retail container — an `.nsp`, an
//! `.xci` or a bare Program `.nca`, which needs its keys after it. An NRO is
//! not the workload a retail title is, so a ranking taken from one does not
//! transfer.
mod common;

const USAGE: &str = "hotspots <target> [prod.keys] [title.keys] [font.ttf]";

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

/// Bytes of guest code per row of the address histogram. A page rather than an
/// instruction because a hot loop is a run of instructions and one bucket each
/// turns a profile into a list.
const BUCKET: u32 = 4096;

/// The top-level groups [`switch_core::cpu`]'s translator decodes on, keyed by
/// bits 28:25 of the encoding. Naming them here rather than counting raw bytes
/// is what makes the table say which part of the translator to work on.
fn group_of(insn: u32) -> &'static str {
    match (insn >> 25) & 0xF {
        0x8 | 0x9 => "data-proc immediate",
        0x5 | 0xD => "data-proc register",
        0x4 | 0x6 | 0xC | 0xE => "loads and stores",
        0x7 | 0xF => "SIMD and floating point",
        0xA | 0xB => "branch, exception, system",
        _ => "reserved and SVE",
    }
}

fn main() {
    let args = common::program_args(USAGE);
    let program = args.open_program();
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let booted = program.boot(&mut cpu);

    // Two frames of startup, so what follows is a steady-state frame. Nothing
    // is sampled here, so it runs through the block translator.
    common::run_to(&mut cpu, u64::MAX, |cpu| cpu.nv.gpu.frames >= 2);

    // Keyed by page rather than indexed into a fixed window: a retail title's
    // modules land wherever the loader put them and run to far more than the
    // 16 MiB an NRO image was assumed to fit in. Gating the *mix* on that
    // window as well is what would make this report a retail frame as almost
    // no instructions at all.
    let mut by_page: BTreeMap<u32, u64> = BTreeMap::new();
    let mut by_top = [0u64; 256];
    let mut by_group: BTreeMap<&'static str, u64> = BTreeMap::new();
    let start = cpu.nv.gpu.frames;
    // Every instruction is counted, so this half is necessarily stepwise.
    let run = common::drive(&mut cpu, Pace::Instructions, u64::MAX, |cpu, _| {
        if cpu.nv.gpu.frames != start {
            return Flow::Stop;
        }
        let pc = cpu.get_pc();
        *by_page.entry(pc / BUCKET * BUCKET).or_default() += 1;
        if let Ok(insn) = cpu.mem.read_u32(pc) {
            by_top[((insn >> 24) & 0xFF) as usize] += 1;
            *by_group.entry(group_of(insn)).or_default() += 1;
        }
        Flow::Continue
    });
    let total = run.steps;
    println!("one frame = {total} instructions");
    for module in &booted.modules {
        println!(
            "module base={:#010x} text at {:#010x}, {} bytes",
            module.base, module.text.mem_addr, module.text.file_size
        );
    }

    let mut buckets: Vec<(u64, u32)> = by_page.iter().map(|(&at, &n)| (n, at)).collect();
    buckets.sort_unstable_by_key(|&(count, _)| std::cmp::Reverse(count));
    println!("--- hottest guest code (4 KiB buckets) ---");
    for (count, addr) in buckets.iter().take(10) {
        println!("{addr:#010x}  {count:>12}  {:5.2}%", pct(*count, total));
    }

    let mut groups: Vec<(u64, &str)> = by_group.iter().map(|(&name, &n)| (n, name)).collect();
    groups.sort_unstable_by_key(|&(count, _)| std::cmp::Reverse(count));
    println!("--- by translator group (bits 28:25) ---");
    for (count, name) in &groups {
        println!("{count:>12}  {:5.2}%  {name}", pct(*count, total));
    }

    let mut tops: Vec<(u64, usize)> = by_top
        .iter()
        .copied()
        .zip(0..256)
        .filter(|(n, _)| *n > 0)
        .collect();
    tops.sort_unstable_by_key(|&(count, _)| std::cmp::Reverse(count));
    println!("--- instruction mix (bits 31:24) ---");
    for (count, top) in tops.iter().take(16) {
        println!("{top:#04x}  {count:>12}  {:5.2}%", pct(*count, total));
    }
}

fn pct(count: u64, total: u64) -> f64 {
    count as f64 * 100.0 / total as f64
}
