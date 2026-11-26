//! Profiles frames by guest address, encoding byte and translator group:
//! `hotspots <target> [--keys <prod.keys>] [--instructions] [--simd] [--from-frame <n>] [--frames <n>]`.
//!
//! The target is an `.nro`, `.nsp`, `.xci` or Program `.nca` (keys required for
//! retail).
mod common;

use clap::Parser;
use common::{Flow, Pace};
use std::collections::{BTreeMap, HashMap};

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
    /// List the hottest individual instructions.
    #[arg(long)]
    instructions: bool,
    /// Rank SIMD&FP instructions by mnemonic, each address counted at most `HOT` times:
    /// roughly what the block translator runs uncompiled.
    #[arg(long)]
    simd: bool,
    /// The presented frame to start at; the default skips startup to a steady frame.
    #[arg(long, default_value_t = 2)]
    from_frame: u64,
    /// Frames to profile.
    #[arg(long, default_value_t = 1)]
    frames: u64,
}

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

/// Advanced SIMD, scalar FP, and the loads and stores with the V bit set.
fn is_simd(insn: u32) -> bool {
    matches!((insn >> 25) & 0xF, 0x7 | 0xF | 0x6 | 0xE)
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
    let booted = program.boot(&mut cpu, None);
    common::start_play(&mut cpu);

    if common::reach_frame(&mut cpu, cli.from_frame, u64::MAX).is_none() {
        return;
    }

    let detail = cli.instructions || cli.simd;
    let mut by_pc: HashMap<u32, u64> = HashMap::new();
    let mut by_page: HashMap<u32, u64> = HashMap::new();
    let mut by_top = [0u64; 256];
    let mut by_group: BTreeMap<&'static str, u64> = BTreeMap::new();
    let stop = cpu.nv.gpu.frames + cli.frames;
    let run = common::drive(&mut cpu, Pace::Instructions, u64::MAX, |cpu, _| {
        if cpu.nv.gpu.frames >= stop {
            return Flow::Stop;
        }
        let pc = cpu.get_pc();
        *by_page.entry(pc / BUCKET * BUCKET).or_default() += 1;
        if detail {
            *by_pc.entry(pc).or_default() += 1;
        }
        if let Ok(insn) = cpu.mem.fetch(pc) {
            by_top[((insn >> 24) & 0xFF) as usize] += 1;
            *by_group.entry(group_of(insn)).or_default() += 1;
        }
        Flow::Continue
    });
    let total = run.steps;
    println!(
        "{} frame(s) from frame {} = {total} instructions",
        cli.frames, cli.from_frame
    );
    for module in &booted.modules {
        println!(
            "module base={:#010x} text at {:#010x}, {} bytes",
            module.base, module.text.mem_addr, module.text.file_size
        );
    }

    let mut buckets: Vec<(u64, u32)> = by_page.iter().map(|(&at, &n)| (n, at)).collect();
    buckets.sort_unstable_by_key(|&(count, addr)| (std::cmp::Reverse(count), addr));
    println!("--- hottest guest code (4 KiB buckets) ---");
    for (count, addr) in buckets.iter().take(10) {
        println!("{addr:#010x}  {count:>12}  {:5.2}%", pct(*count, total));
    }

    if cli.simd {
        report_simd(&cpu, &by_pc, total);
    }

    if cli.instructions {
        let mut instructions: Vec<_> = by_pc.into_iter().collect();
        instructions.sort_unstable_by_key(|&(pc, count)| (std::cmp::Reverse(count), pc));
        println!("--- hottest individual instructions ---");
        for (pc, count) in instructions.iter().take(48) {
            let insn = cpu.mem.fetch(*pc).unwrap();
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

fn report_simd(cpu: &switch_core::cpu::Cpu, by_pc: &HashMap<u32, u64>, total: u64) {
    let hot = u64::from(switch_core::cpu::HOT);
    let mut by_mnemonic: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let (mut all, mut capped) = (0u64, 0u64);
    for (&pc, &count) in by_pc {
        let Ok(insn) = cpu.mem.fetch(pc) else {
            continue;
        };
        if !is_simd(insn) {
            continue;
        }
        let text = switch_core::disasm::disassemble(insn);
        // The disassembler has no vector data processing; key those by encoding, registers masked.
        let mnemonic = match text.split_whitespace().next() {
            Some(".word") | None => format!("{:#010x}", insn & 0xFFE0_FC00),
            Some(word) => word.to_owned(),
        };
        let row = by_mnemonic.entry(mnemonic).or_default();
        row.0 += count;
        row.1 += count.min(hot);
        all += count;
        capped += count.min(hot);
    }
    println!(
        "--- SIMD&FP: {all} run ({:.2}%), {capped} below HOT ({:.2}%) ---",
        pct(all, total),
        pct(capped, total)
    );
    let mut rows: Vec<_> = by_mnemonic.into_iter().collect();
    rows.sort_unstable_by_key(|(name, (_, cold))| (std::cmp::Reverse(*cold), name.clone()));
    for (name, (count, cold)) in rows.iter().take(40) {
        println!(
            "{name:<10} {cold:>12} below HOT ({:5.1}%)  {count:>12} run",
            pct(*cold, capped)
        );
    }
}

fn pct(count: u64, total: u64) -> f64 {
    count as f64 * 100.0 / total as f64
}
