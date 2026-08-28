//! Print where libtransistor's embedded squashfs image sits in an NRO:
//! `squash_info <path.nro>`.
mod common;

use switch_core::nro::symbol_value;

/// Where `load_nro` puts an NRO's first byte.
const BASE: u64 = 0x0800_0000;

fn main() {
    let data = common::read(common::arg(1, "squash_info <path.nro>"));
    for name in [
        "_libtransistor_squashfs_image",
        "_libtransistor_squashfs_image_end",
    ] {
        match symbol_value(&data, name) {
            Some(value) => println!("{name} = {:#x}", BASE + value),
            None => println!("{name} = not found"),
        }
    }
}
