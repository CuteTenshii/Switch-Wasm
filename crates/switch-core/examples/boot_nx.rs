//! Boot an NRO and print what it wrote to the console:
//! `boot_nx <path.nro>`.
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        common::usage("boot_nx <path.nro>")
    };
    let data = common::read(&path);
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let loaded = cpu.boot_homebrew(&data).expect("boot nro");
    println!("entry = {:#010x}", loaded.entry);

    let run = common::drive(
        &mut cpu,
        Pace::Blocks,
        common::env_u64("STEPS", u64::MAX),
        |_, _| Flow::Continue,
    );
    if let Some(fault) = &run.fault {
        println!("FAULT at step {}: {fault}", run.steps);
    }
    if run.halted {
        println!(
            "HALTED at step {} pc={:#x} x0={:#x}",
            run.steps,
            cpu.get_pc(),
            cpu.read_x(0)
        );
    }

    println!("--- program console output ({} bytes) ---", cpu.out.len());
    let out = String::from_utf8_lossy(&cpu.out);
    for line in out.lines().take(60) {
        println!("  {line}");
    }
}
