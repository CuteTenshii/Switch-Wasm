use std::fs;
use switch_core::mem::Memory;
use switch_core::nro::load_nro;

fn main() {
    let path = std::env::args().nth(1).expect("nro");
    let addr = u32::from_str_radix(std::env::args().nth(2).expect("addr").trim_start_matches("0x"), 16).unwrap();
    let len = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let data = fs::read(&path).expect("read");
    let mut mem = Memory::new();
    load_nro(&mut mem, &data).expect("load");
    for i in 0..len {
        print!("{:02x}", mem.read_u8(addr.wrapping_add(i)).unwrap_or(0));
    }
    println!();
}
