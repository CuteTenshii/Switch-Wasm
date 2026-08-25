//! Boot an NRO with libtransistor's `sqfs_init` stubbed out to return 0, to
//! see how far it gets without its embedded filesystem:
//! `patch_sqfs <path.nro>`.
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;
use switch_core::nro::{load_nro, symbol_value};

/// Where `load_nro` puts an NRO's first byte.
const BASE: u32 = 0x0800_0000;

fn main() {
    let data = common::read(common::arg(1, "patch_sqfs <path.nro>"));
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    load_nro(&mut cpu.mem, &data).expect("load nro");

    let sqfs_init = symbol_value(&data, "sqfs_init").expect("sqfs_init") as u32 + BASE;
    println!("patching sqfs_init at {sqfs_init:#x} to return 0");
    let _ = cpu.mem.write_u32(sqfs_init, 0xd280_0000); // mov x0, #0
    let _ = cpu.mem.write_u32(sqfs_init + 4, 0xd65f_03c0); // ret

    for reg in 0..=30u8 {
        cpu.set_reg(reg, 0);
    }
    cpu.set_reg(1, 1); // boot_entry_regs: x0 = 0, x1 = 1
    cpu.set_pc(BASE);

    let run = common::drive(&mut cpu, Pace::Blocks, common::env_u64("STEPS", 5_000_000), |_, _| {
        Flow::Continue
    });
    if let Some(fault) = &run.fault {
        println!("FAULT at step {}: {fault}", run.steps);
    } else if run.halted {
        println!("HALTED at step {} pc={:#x} x0={:#x}", run.steps, cpu.get_pc(), cpu.read_x(0));
    } else {
        println!("BUDGET exhausted at step {}", run.steps);
    }

    println!("--- program console output ({} bytes) ---", cpu.out.len());
    for line in String::from_utf8_lossy(&cpu.out).lines().take(80) {
        println!("  {line}");
    }
}
