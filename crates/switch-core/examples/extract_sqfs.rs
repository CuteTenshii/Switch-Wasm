//! Extract libtransistor's embedded squashfs image out of a loaded NRO:
//! `extract_sqfs <path.nro> <out.bin>`.
mod common;

use switch_core::cpu::Cpu;
use switch_core::nro::{load_nro, symbol_value};

const USAGE: &str = "extract_sqfs <path.nro> <out.bin>";
/// Where `load_nro` puts an NRO's first byte.
const BASE: u32 = 0x0800_0000;

fn main() {
    let data = common::read(common::arg(1, USAGE));
    let out = common::arg(2, USAGE);
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    load_nro(&mut cpu.mem, &data).expect("load nro");

    let start = symbol_value(&data, "_libtransistor_squashfs_image").expect("start symbol") as u32;
    let end = symbol_value(&data, "_libtransistor_squashfs_image_end").expect("end symbol") as u32;
    let size = end.wrapping_sub(start) as usize;
    println!("squashfs at {:#x} size {size:#x}", BASE + start);
    let bytes = cpu.mem.dump(BASE + start, size).expect("dump");
    std::fs::write(&out, &bytes).expect("write");
    println!("wrote {out}");
}
