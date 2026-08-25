//! Hex-dump guest memory after loading an NRO:
//! `dump_mem <path.nro> <addr> [len]`.
mod common;

use switch_core::mem::Memory;
use switch_core::nro::load_nro;

const USAGE: &str = "dump_mem <path.nro> <addr> [len]";

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let addr = common::hex(&common::arg(2, USAGE));
    let len = common::opt_num(3).unwrap_or(64) as u32;
    let mut mem = Memory::new();
    load_nro(&mut mem, &data).expect("load nro");
    for i in 0..len {
        print!("{:02x}", mem.read_u8(addr.wrapping_add(i)).unwrap_or(0));
    }
    println!();
}
