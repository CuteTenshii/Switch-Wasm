//! Break on guest PCs and dump registers, or profile where a run spends its
//! time: `trace <path.nro> [pc...]`. With no PCs, prints the hottest at the end.
mod common;

use common::{Flow, Pace};
use std::collections::{HashMap, HashSet};
use switch_core::cpu::Cpu;

/// How often to sample the PC when profiling.
const SAMPLE_EVERY: u64 = 64;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        common::usage("trace <path.nro> [pc...]")
    };
    let watch: HashSet<u32> = args.map(|a| common::hex(&a)).collect();

    let data = common::read(&path);
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.boot_homebrew(&data).expect("boot");

    let mut hot: HashMap<u32, u64> = HashMap::new();
    let run = common::drive(
        &mut cpu,
        // Both modes below look at the PC of the instruction about to run.
        Pace::Instructions,
        common::env_u64("STEPS", 60_000_000),
        |cpu, steps| {
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
        },
    );
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
    println!("--- console ---\n{}", String::from_utf8_lossy(&cpu.out));
}
