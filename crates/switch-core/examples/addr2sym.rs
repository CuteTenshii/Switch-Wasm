//! Map an address back to the symbol it falls in, using an NRO's own dynamic
//! symbol table: `addr2sym <path.nro> <addr>`.
mod common;

fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}
fn read_u64(data: &[u8], at: usize) -> u64 {
    let low = read_u32(data, at) as u64;
    let high = read_u32(data, at + 4) as u64;
    low | (high << 32)
}

fn find_mod0(data: &[u8]) -> Option<usize> {
    let magic = 0x30444f4du32.to_le_bytes();
    data.windows(4).position(|w| w == &magic[..])
}

fn read_cstr(data: &[u8], off: usize) -> String {
    let mut s = Vec::new();
    for i in off..data.len() {
        if data[i] == 0 { break; }
        s.push(data[i]);
    }
    String::from_utf8_lossy(&s).into_owned()
}

fn main() {
    const USAGE: &str = "addr2sym <path.nro> <addr>";
    let data = common::read(common::arg(1, USAGE));
    let target = common::hex(&common::arg(2, USAGE));
    let mod0 = find_mod0(&data).expect("mod0");
    let dyn_off = mod0.wrapping_add(read_u32(&data, mod0 + 4) as usize);
    let mut symtab = 0u64;
    let mut strtab = 0u64;
    let mut off = dyn_off;
    while off + 16 <= data.len() {
        let tag = read_u64(&data, off);
        let val = read_u64(&data, off + 8);
        off += 16;
        if tag == 0 { break; }
        match tag {
            0x06 => symtab = val,
            0x05 => strtab = val,
            _ => {}
        }
    }
    println!("symtab={:#x} strtab={:#x}", symtab, strtab);
    if symtab == 0 || strtab == 0 { return; }
    const BASE: u32 = 0x0800_0000;
    let symtab = (symtab as u32) as usize;
    let strtab = (strtab as u32).wrapping_sub(BASE) as usize;
    let mut best: Option<(u32, u32, String)> = None;
    for i in 0.. {
        let sym_off = symtab + i * 24;
        if sym_off + 24 > data.len() { break; }
        let name_off = read_u32(&data, sym_off) as usize;
        if name_off == 0 { continue; }
        let name = read_cstr(&data, strtab + name_off);
        let value = read_u32(&data, sym_off + 8);
        let size = read_u32(&data, sym_off + 16);
        if i < 8 {
            println!("sym {} name={:?} off={:#x} value={:#x} size={:#x}", i, name, name_off, value, size);
        }
        if value == 0 { continue; }
        if target >= value && target < value + size {
            println!("{:#x} is in {}+{:#x} (size {:#x})", target, name, target - value, size);
            return;
        }
        if value < target {
            if best.as_ref().map_or(true, |(bv, _, _)| value > *bv) {
                best = Some((value, size, name));
            }
        }
    }
    if let Some((v, _, name)) = best {
        println!("{:#x} after {}+{:#x}", target, name, target - v);
    } else {
        println!("{:#x} not found", target);
    }
}
