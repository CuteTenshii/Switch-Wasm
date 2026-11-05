//! Hex-dump guest memory after loading an NRO:
//! `dump_mem <path.nro> --addr <addr> [--len <n>]`.
mod common;

use clap::Parser;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

#[derive(Parser)]
struct Cli {
    nro: String,
    /// Hexadecimal address to start at.
    #[arg(long, value_parser = common::hex32)]
    addr: u32,
    /// Bytes to print.
    #[arg(long, default_value_t = 64)]
    len: u32,
}

fn main() {
    let cli = Cli::parse();
    let data = common::read(&cli.nro);
    let (addr, len) = (cli.addr, cli.len);
    let mem = Memory::new();
    load_nro(&mem, &data).expect("load nro");
    for i in 0..len {
        print!("{:02x}", mem.read_u8(addr.wrapping_add(i)).unwrap_or(0));
    }
    println!();
}
