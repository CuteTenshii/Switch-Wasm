//! Disassemble a flat image dumped by `dump_exefs` at its real load address:
//! `disasm_flat <file.bin> --base <base> --addr <addr> [--count <n>]`.
mod common;

use clap::Parser;
use switch_core::disasm::disassemble;

#[derive(Parser)]
struct Cli {
    file: String,
    /// Hexadecimal load address of the image.
    #[arg(long, value_parser = common::hex32)]
    base: u32,
    /// Hexadecimal address to start at.
    #[arg(long, value_parser = common::hex32)]
    addr: u32,
    /// Instructions to print.
    #[arg(long, default_value_t = 48)]
    count: u32,
}

fn main() {
    let cli = Cli::parse();
    let data = common::read(&cli.file);
    let (base, addr, count) = (cli.base, cli.addr, cli.count);
    for i in 0..count {
        let va = addr.wrapping_add(i * 4);
        let Some(word) = data
            .get((va.wrapping_sub(base)) as usize..)
            .and_then(|s| s.first_chunk::<4>())
        else {
            break;
        };
        let insn = u32::from_le_bytes(*word);
        println!("{va:#010x}: {insn:08x}  {}", disassemble(insn));
    }
}
