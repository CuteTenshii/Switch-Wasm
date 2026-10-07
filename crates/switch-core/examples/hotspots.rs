//! Profiles one steady-state frame by guest address, encoding byte and
//! translator group: `hotspots <target> [prod.keys] [title.keys] [font.ttf]`.
//!
//! The target is an `.nro`, `.nsp`, `.xci` or Program `.nca` (keys required for
//! retail). Add `--instructions` to list the hottest individual instructions.
mod common;

const USAGE: &str = "hotspots <target> [prod.keys] [title.keys] [font.ttf]";

use common::{Flow, Pace};
use std::collections::BTreeMap;
use switch_core::cpu::Cpu;

/// Bytes of guest code per row of the address histogram.
const BUCKET: u32 = 4096;

/// The translator's top-level groups, keyed by bits 28:25 of the encoding.
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

    // Two frames of startup, so what follows is a steady-state frame.
    common::run_to(&mut cpu, u64::MAX, |cpu| cpu.nv.gpu.frames >= 2);

    let detail = std::env::args().any(|arg| arg == "--instructions");
    let mut by_pc: BTreeMap<u32, u64> = BTreeMap::new();
    let mut by_page: BTreeMap<u32, u64> = BTreeMap::new();
    let mut by_top = [0u64; 256];
    let mut by_group: BTreeMap<&'static str, u64> = BTreeMap::new();
    let start = cpu.nv.gpu.frames;
    let run = common::drive(&mut cpu, Pace::Instructions, u64::MAX, |cpu, _| {
        if cpu.nv.gpu.frames != start {
            return Flow::Stop;
        }
        let pc = cpu.get_pc();
        *by_page.entry(pc / BUCKET * BUCKET).or_default() += 1;
        if detail {
            *by_pc.entry(pc).or_default() += 1;
        }
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

    if detail {
        let mut instructions: Vec<_> = by_pc.into_iter().collect();
        instructions.sort_unstable_by_key(|&(pc, count)| (std::cmp::Reverse(count), pc));
        println!("--- hottest individual instructions ---");
        for (pc, count) in instructions.iter().take(48) {
            let insn = cpu.mem.read_u32(*pc).unwrap();
            println!(
                "{pc:#010x} {count:>12} {insn:08x} {}",
                switch_core::disasm::disassemble(insn)
            );
        }
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
