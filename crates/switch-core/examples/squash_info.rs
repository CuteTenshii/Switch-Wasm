use std::fs;
use switch_core::nro::symbol_value;

fn main() {
    let path = std::env::args().nth(1).expect("usage: squash_info <nro>");
    let data = fs::read(&path).expect("read nro");
    let base = 0x8000000u64;
    let names = [
        "_libtransistor_squashfs_image",
        "_libtransistor_squashfs_image_end",
    ];
    for name in names {
        if let Some(v) = symbol_value(&data, name) {
            println!("{} = {:#x}", name, base + v);
        } else {
            println!("{} = not found", name);
        }
    }
}
