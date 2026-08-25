//! Trace an NRO's boot under the Horizon syscall stubs, watching one word for
//! changes: `boot_hbmenu <path.nro> [addr]`.
mod common;

use common::{Flow, Pace};
use switch_core::cpu::Cpu;
use switch_core::nro::load_nro;

/// The word this was written to watch, kept as the default so the tool still
/// does what it did with no second argument.
const DEFAULT_WATCH: u32 = 0x0825_3fb8;

fn main() {
    let data = common::read(common::arg(1, "boot_hbmenu <path.nro> [addr]"));
    let watch = common::opt_arg(2).map(|a| common::hex(&a)).unwrap_or(DEFAULT_WATCH);

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    let loaded = load_nro(&mut cpu.mem, &data).expect("load nro");
    println!("entry = {:#010x}, watching {watch:#x}", loaded.entry);
    for reg in 0..=30u8 {
        cpu.set_reg(reg, 0);
    }
    cpu.set_reg(1, 1); // boot_entry_regs: x0 = 0, x1 = 1
    cpu.set_pc(loaded.entry);
    cpu.trace_enabled = true;

    let mut last = cpu.mem.read_u32(watch).unwrap_or(0xFFFF);
    let run = common::drive(
        &mut cpu,
        Pace::Instructions,
        common::env_u64("STEPS", 5_000_000),
        |cpu, steps| {
            let now = cpu.mem.read_u32(watch).unwrap_or(0xFFFF);
            if now != last {
                println!("step {steps}: [{watch:#x}] {last:#x} -> {now:#x} (pc {:#x})", cpu.get_pc());
                last = now;
            }
            Flow::Continue
        },
    );
    if let Some(fault) = &run.fault {
        println!("FAULT at step {}: {fault}", run.steps);
    } else if run.halted {
        println!("HALTED at step {} (pc {:#x})", run.steps, cpu.get_pc());
    } else {
        println!("budget exhausted");
    }
    println!("stopped at step {}, pc={:#x}", run.steps, cpu.get_pc());
    println!("{}", String::from_utf8_lossy(&cpu.trace));
}
