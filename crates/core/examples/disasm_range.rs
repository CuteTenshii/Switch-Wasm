//! Disassemble a run of instructions from a loaded NRO:
//! `disasm_range <path.nro> --addr <addr> [--count <n>]`.
mod common;

use clap::Parser;
use switch_core::disasm::disassemble;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

#[derive(Parser)]
struct Cli {
    nro: String,
    /// Hexadecimal address to start at.
    #[arg(long, value_parser = common::hex32)]
    addr: u32,
    /// Instructions to print.
    #[arg(long, default_value_t = 64)]
    count: u32,
}

fn main() {
    let cli = Cli::parse();
    let data = common::read(&cli.nro);
    let (start, count) = (cli.addr, cli.count);
    let mem = Memory::new();
    load_nro(&mem, &data).expect("load nro");
    for i in 0..count {
        let at = start.wrapping_add(i * 4);
        let insn = mem.fetch(at).unwrap_or(0);
        println!("{at:#010x}: {insn:#010x} {}", disassemble(insn));
    }
}
