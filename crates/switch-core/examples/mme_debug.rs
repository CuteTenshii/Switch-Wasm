//! Boot an NRO and run until the first fault, printing it — which, for an MME
//! timeout, includes the macro disassembly:
//! `mme_debug <path.nro> [max_steps]`.
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;

fn main() {
    let data = common::read(common::arg(1, "mme_debug <path.nro> [max_steps]"));
    let budget = common::opt_num(2).unwrap_or(500_000_000);
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    common::load_fallback_font(&mut cpu);
    let loaded = cpu.boot_homebrew(&data).expect("boot nro");
    println!(
        "entry = {:#010x}, env = {:#x}",
        loaded.entry, loaded.env_addr
    );

    let run = common::drive(&mut cpu, Pace::Blocks, budget, |_, _| Flow::Continue);
    match &run.fault {
        Some(fault) => println!("FAULT at step {}: {fault}", run.steps),
        None if run.halted => println!("HALTED at step {}, pc={:#x}", run.steps, cpu.get_pc()),
        None => println!("budget exhausted ({budget}), pc={:#x}", cpu.get_pc()),
    }
}
