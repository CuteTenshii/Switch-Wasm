//! Break on guest PCs and dump registers, or profile where a run spends its
//! time: `trace <path.nro> [--pc <pc,...>]`. With no PCs, prints the hottest at the end.
mod common;

use clap::Parser;
use common::{Flow, Pace};
use std::collections::{HashMap, HashSet};

const SAMPLE_EVERY: u64 = 64;

#[derive(Parser)]
struct Cli {
    nro: String,
    #[command(flatten)]
    shared: common::Shared,
    /// Hexadecimal pcs to break on.
    #[arg(long, value_delimiter = ',', value_parser = common::hex32, value_name = "PC")]
    pc: Vec<u32>,
    /// Instructions to run.
    #[arg(long, default_value_t = 60_000_000)]
    steps: u64,
}

fn main() {
    let cli = common::parse::<Cli>(|cli| &cli.shared);
    let watch: HashSet<u32> = cli.pc.iter().copied().collect();

    let data = common::read(&cli.nro);
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    cpu.boot_homebrew(&data).expect("boot");

    let mut hot: HashMap<u32, u64> = HashMap::new();
    let run = common::drive(&mut cpu, Pace::Instructions, cli.steps, |cpu, steps| {
        let pc = cpu.get_pc();
        if watch.contains(&pc) {
            print!("{steps} pc={pc:#x}");
            for reg in [0u8, 1, 2, 3, 19, 20, 21, 22, 23] {
                print!(" x{reg}={:#x}", cpu.read_x(reg));
            }
            println!(" lr={:#x}", cpu.read_x(30));
        }
        if watch.is_empty() && steps % SAMPLE_EVERY == 0 {
            *hot.entry(pc).or_default() += 1;
        }
        Flow::Continue
    });
    if let Some(fault) = &run.fault {
        println!("FAULT at {}: {fault} pc={:#x}", run.steps, cpu.get_pc());
    }
    println!(
        "stopped after {} steps, halted={} pc={:#x}",
        run.steps,
        cpu.halted,
        cpu.get_pc()
    );

    if !hot.is_empty() {
        let mut top: Vec<_> = hot.into_iter().collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        println!("--- hottest PCs ---");
        for (pc, n) in top.into_iter().take(20) {
            println!("  {pc:#010x}  {n}");
        }
    }
    common::print_console(&cpu, usize::MAX);
}
