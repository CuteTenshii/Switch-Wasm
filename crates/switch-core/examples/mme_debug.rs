//! Boot an NRO (via `boot_homebrew`, like the browser) and run until the first
//! fault, printing the fault (which, for an MME timeout, includes the macro
//! disassembly). Usage:
//! `cargo run -p switch-core --example mme_debug -- <path> [max_steps]`.
use std::env;
use std::fs;
use switch_core::cpu::Cpu;

fn main() {
    let path = env::args().nth(1).expect("usage: mme_debug <path> [max_steps]");
    let max_steps: u64 = env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(500_000_000);
    let data = fs::read(&path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    if let Ok(font) = fs::read("web/font.ttf") {
        cpu.set_shared_font(font);
    }
    let loaded = cpu.boot_homebrew(&data).expect("boot nro");
    println!("entry = {:#010x}, env = {:#x}", loaded.entry, loaded.env_addr);

    for step in 0..max_steps {
        if let Err(e) = cpu.step() {
            println!("FAULT at step {step}: {e}");
            return;
        }
        if cpu.halted {
            println!("HALTED at step {step}, pc={:#x}", cpu.get_pc());
            cpu.trace_enabled = true;
            return;
        }
    }
    println!("budget exhausted ({max_steps}), pc={:#x}", cpu.get_pc());
}
