//! Disassemble a single A64 instruction word: `disasm_insn <word>`.
mod common;

use clap::Parser;
use switch_core::disasm::disassemble;

#[derive(Parser)]
struct Cli {
    /// The instruction word, in hexadecimal.
    #[arg(value_parser = common::hex32)]
    word: u32,
}

fn main() {
    let insn = Cli::parse().word;
    println!("{insn:#010x}: {}", disassemble(insn));
}
