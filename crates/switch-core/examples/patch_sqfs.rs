use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nro::{load_nro, symbol_value};

fn main() {
    let path = std::env::args().nth(1).expect("usage: patch_sqfs <nro>");
    let data = fs::read(&path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    load_nro(&mut cpu.mem, &data).expect("load nro");

    let sqfs_init = symbol_value(&data, "sqfs_init").expect("sqfs_init") as u32 + 0x8000000;
    println!("patching sqfs_init at {:#x} to return 0", sqfs_init);
    // mov x0, #0 -> 0xd2800000; ret -> 0xd65f03c0
    let _ = cpu.mem.write_u32(sqfs_init, 0xd2800000);
    let _ = cpu.mem.write_u32(sqfs_init + 4, 0xd65f03c0);

    for i in 0..=30u8 { cpu.set_reg(i, 0); }
    cpu.set_reg(1, 1);
    cpu.set_pc(0x8000000);
    let budget: u64 = 5_000_000;
    let mut done = 0u64;
    loop {
        if done >= budget { println!("BUDGET exhausted"); break; }
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
    for line in out.lines().take(80) { println!("  {line}"); }
}
