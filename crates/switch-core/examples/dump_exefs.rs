//! Dump a retail Program ExeFS as flat module images plus symbol maps, at
//! `boot_retail_program` addresses: `dump_exefs <container> <prod.keys> [title.keys] <out_dir>`.
mod common;

use std::fs;

const USAGE: &str = "dump_exefs <container> <prod.keys> [title.keys] <out_dir>";

fn main() {
    let args = common::container_args(USAGE);
    let out_dir = args.need(0).to_string();
    let title = args.open();
    fs::create_dir_all(&out_dir).expect("create the output directory");

    for file in &title.exefs_pfs0.files {
        let start = file.offset as usize;
        let raw = &title.exefs[start..start + file.size as usize];
        fs::write(format!("{out_dir}/raw_{}", file.name), raw).expect("write the raw module");
    }

    let mem = switch_core::mem::Memory::new();
    let mut base = switch_core::nso::NSO_BASE;
    let mut syms: Vec<(u32, u32, String, String)> = Vec::new();
    for (name, nso) in title.modules() {
        let m = switch_core::nso::load_nso(&mem, nso, base).expect("load the module");
        let image_end = m.data.mem_addr + m.data.file_size + m.bss_size;
        let mut flat = Vec::with_capacity((image_end - base) as usize);
        for a in base..image_end {
            flat.push(mem.read_u8(a).unwrap_or(0));
        }
        let path = format!("{out_dir}/{name}.bin");
        fs::write(&path, &flat).expect("write the flat image");
        println!(
            "{name}: base={:#010x} entry={:#010x} text={:#x}+{:#x} ro={:#x}+{:#x} data={:#x}+{:#x} bss={:#x} end={:#010x} -> {path}",
            m.base, m.entry,
            m.text.mem_addr, m.text.file_size,
            m.ro.mem_addr, m.ro.file_size,
            m.data.mem_addr, m.data.file_size,
            m.bss_size, image_end
        );

        let found = common::symbols::read(&flat);
        if !found.is_empty() {
            println!("  {} symbols", found.len());
        }
        for sym in found {
            syms.push((
                base + sym.value as u32,
                sym.size as u32,
                sym.name,
                name.to_string(),
            ));
        }
        base = (image_end + 0xfff) & !0xfff;
    }
    syms.sort();
    let mut out = String::new();
    for (a, s, n, m) in &syms {
        out.push_str(&format!("{a:#010x} {s:#x} {m} {n}\n"));
    }
    fs::write(format!("{out_dir}/symbols.txt"), out).expect("write the symbol map");
    println!("wrote {} symbols", syms.len());
}
