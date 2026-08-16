use std::fs;
use switch_core::cpu::{Cpu, SyscallMode};
use switch_core::nro::{load_nro, symbol_value};

fn main() {
    let nro_path = std::env::args().nth(1).expect("usage: extract_sqfs <nro> <out.bin>");
    let out_path = std::env::args().nth(2).expect("usage: extract_sqfs <nro> <out.bin>");
    let data = fs::read(&nro_path).expect("read nro");
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    cpu.syscall_mode = SyscallMode::Horizon;
    load_nro(&mut cpu.mem, &data).expect("load nro");

    let start = symbol_value(&data, "_libtransistor_squashfs_image").expect("start symbol") as u32;
    let end = symbol_value(&data, "_libtransistor_squashfs_image_end").expect("end symbol") as u32;
    let size = end.wrapping_sub(start) as usize;
    println!("squashfs at {:#x} size {:#x}", 0x8000000u64 + start as u64, size);
    let bytes = cpu.mem.dump(start + 0x8000000, size).expect("dump");
    fs::write(&out_path, &bytes).expect("write");
    println!("wrote {}", out_path);
}
