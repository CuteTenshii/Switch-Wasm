use std::fs;
use switch_core::cpu::{Cpu, SyscallMode};
fn main() {
    let path = std::env::args().nth(1).expect("usage: boot_nx <path-to-.nro>");
    let data = fs::read(&path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.syscall_mode = SyscallMode::Horizon;
    let loaded = cpu.boot_homebrew(&data).expect("boot nro");
    println!("entry = {:#010x}", loaded.entry);
    let mut done = 0u64;
    loop {
        match cpu.step() {
            Ok(()) => {}
            Err(e) => {
                println!("FAULT at step {done}: {e}");
                break;
            }
        }
        if cpu.halted {
            println!("HALTED at step {done} pc={:#x} x0={:#x}", cpu.get_pc(), cpu.read_x(0));
            break;
        }
        done += 1;
    }
    println!("--- program console output ({} bytes) ---", cpu.out.len());
    let out = String::from_utf8_lossy(&cpu.out);
    for line in out.lines().take(60) {
        println!("  {line}");
    }
}
