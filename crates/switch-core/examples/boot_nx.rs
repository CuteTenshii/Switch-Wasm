use std::fs;
use switch_core::cpu::{Cpu, SyscallMode};
use switch_core::nro::load_nro;
fn main() {
    let path = std::env::args().nth(1).expect("usage: boot_nx <path-to-.nro>");
    let data = fs::read(&path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.syscall_mode = SyscallMode::Horizon;
    let loaded = load_nro(&mut cpu.mem, &data).expect("load nro");
    for i in 0..=30u8 { cpu.set_reg(i, 0); }
    cpu.set_reg(1, 1);
    cpu.set_pc(loaded.entry);
    println!("entry = {:#010x}", loaded.entry);
    let budget: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    let mut done = 0u64;
    loop {
        if done >= budget { println!("BUDGET exhausted at step {done} pc={:#x}", cpu.get_pc()); break; }
        match cpu.step() {
            Ok(()) => {}
            Err(e) => { println!("FAULT at step {done}: {e}"); break; }
        }
        if cpu.halted {
            println!("HALTED at step {done} pc={:#x} x0={:#x}", cpu.get_pc(), cpu.read_x(0));
            break;
        }
        done += 1;
    }
    println!("--- program console output ({} bytes) ---", cpu.out.len());
    let out = String::from_utf8_lossy(&cpu.out);
    for line in out.lines().take(60) { println!("  {line}"); }
}
