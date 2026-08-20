//! Dev tool for tracing a real NRO (e.g. hbmenu.nro) boot under the Horizon
//! syscall stubs. Usage: `cargo run -p switch-core --example boot_hbmenu -- <path>`.
use std::env;
use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nro::load_nro;

fn main() {
    let path = env::args().nth(1).expect("usage: boot_hbmenu <path-to-.nro>");
    let data = fs::read(&path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let loaded = load_nro(&mut cpu.mem, &data).expect("load nro");
    println!("entry = {:#010x}", loaded.entry);
    for i in 0..=30u8 {
        cpu.set_reg(i, 0);
    }
    cpu.set_reg(1, 1); // boot_entry_regs: x0=0, x1=1
    cpu.set_pc(loaded.entry);
    cpu.trace_enabled = true;

    let steps = 5_000_000u64;
    let mut done = 0u64;
    loop {
        if done >= steps {
            println!("budget exhausted");
            break;
        }
        let before = cpu.mem.read_u32(0x0825_3fb8).unwrap_or(0xFFFF);
        match cpu.step() {
            Ok(()) => {}
            Err(e) => {
                println!("FAULT at step {done}: {e}");
                break;
            }
        }
        let after = cpu.mem.read_u32(0x0825_3fb8).unwrap_or(0xFFFF);
        if before != after {
            println!(
                "step {done}: [0x823b000] {:#x} -> {:#x} (pc was {:#x})",
                before, after, cpu.get_pc()
            );
        }
        if cpu.halted {
            println!("HALTED at step {done} (pc before halt was {:#x})", cpu.get_pc());
            break;
        }
        done += 1;
    }
    println!("stopped at step {done}, pc={:#x}", cpu.get_pc());
    let trace = String::from_utf8_lossy(&cpu.trace);
    println!("{}", trace);
}
