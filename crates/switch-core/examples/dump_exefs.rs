//! Decrypt a retail NSP's Program ExeFS and dump every module as a flat
//! image plus a symbol map, at the exact addresses `boot_retail_program`
//! lays them out at. That makes a backtrace from a real run nameable:
//! `dump_exefs <nsp> <prod.keys> [title.keys] <out_dir>`.

use std::env;
use std::fs;

fn read_u32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}
fn read_u64(d: &[u8], at: usize) -> u64 {
    read_u32(d, at) as u64 | ((read_u32(d, at + 4) as u64) << 32)
}
fn read_cstr(d: &[u8], off: usize) -> String {
    let end = d[off..].iter().position(|&b| b == 0).unwrap_or(0) + off;
    String::from_utf8_lossy(&d[off..end]).into_owned()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let nsp_path = &args[1];
    let prod_path = &args[2];
    let title_path = &args[3];
    let out_dir = &args[4];
    fs::create_dir_all(out_dir).unwrap();

    let nsp_data = fs::read(nsp_path).unwrap();
    let pfs0 = switch_core::nsp::Pfs0::parse(&nsp_data).unwrap();
    let prod_text = fs::read_to_string(prod_path).unwrap();
    let mut keys = switch_core::keys::keyset_from_prod(&switch_core::keys::parse_keys_file(&prod_text));
    let title_text = fs::read_to_string(title_path).unwrap();
    keys.title_keys = switch_core::keys::keyset_from_title(&switch_core::keys::parse_keys_file(&title_text));

    let mut program = None;
    for f in &pfs0.files {
        if !f.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }
        let raw = &nsp_data[f.offset as usize..(f.offset + f.size) as usize];
        if let Ok(nca) = switch_core::nca::Nca::parse_with_keys(raw, Some(&keys)) {
            if nca.content_type == switch_core::nca::ContentType::Program {
                program = Some(f);
            }
        }
    }
    let f = program.unwrap();
    let raw = &nsp_data[f.offset as usize..(f.offset + f.size) as usize];
    let nca = switch_core::nca::Nca::parse_with_keys(raw, Some(&keys)).unwrap();
    if nca.has_rights_id() && keys.title_key(&nca.rights_id).is_none() {
        let tk = switch_core::ticket::find_and_decrypt_title_key(&nca.rights_id, &pfs0.files, &nsp_data, &keys).unwrap();
        keys.title_keys.push((nca.rights_id, tk));
    }
    let idx = nca.exefs_section_index().unwrap();
    let exefs = nca.decrypt_pfs0_section(raw, &keys, idx).unwrap();
    let exefs_pfs0 = switch_core::nsp::Pfs0::parse(&exefs).unwrap();

    for ef in &exefs_pfs0.files {
        let b = &exefs[ef.offset as usize..(ef.offset + ef.size) as usize];
        fs::write(format!("{out_dir}/raw_{}", ef.name), b).unwrap();
    }

    const ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5",
        "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk",
    ];
    let mut mem = switch_core::mem::Memory::new();
    let mut base = switch_core::nso::NSO_BASE;
    let mut syms: Vec<(u32, u32, String, String)> = Vec::new();
    for &name in ORDER {
        let Some(ef) = exefs_pfs0.find(name) else { continue };
        let nso = &exefs[ef.offset as usize..(ef.offset + ef.size) as usize];
        let m = switch_core::nso::load_nso(&mut mem, nso, base).unwrap();
        let image_end = m.data.mem_addr + m.data.file_size + m.bss_size;
        // Flat image, base..image_end, straight out of the mapped memory.
        let mut flat = Vec::with_capacity((image_end - base) as usize);
        for a in base..image_end {
            flat.push(mem.read_u8(a).unwrap_or(0));
        }
        let path = format!("{out_dir}/{name}.bin");
        fs::write(&path, &flat).unwrap();
        println!(
            "{name}: base={:#010x} entry={:#010x} text={:#x}+{:#x} ro={:#x}+{:#x} data={:#x}+{:#x} bss={:#x} end={:#010x} -> {path}",
            m.base, m.entry,
            m.text.mem_addr, m.text.file_size,
            m.ro.mem_addr, m.ro.file_size,
            m.data.mem_addr, m.data.file_size,
            m.bss_size, image_end
        );

        // Dynamic symbol table, addressed the same way the loaded image is.
        let d = &flat[..];
        let magic = 0x3044_4f4du32.to_le_bytes();
        if let Some(mod0) = d.windows(4).position(|w| w == magic) {
            let dyn_off = mod0 + read_u32(d, mod0 + 4) as usize;
            let (mut symtab, mut strtab, mut hash) = (0u64, 0u64, 0u64);
            let mut off = dyn_off;
            while off + 16 <= d.len() {
                let tag = read_u64(d, off);
                let val = read_u64(d, off + 8);
                off += 16;
                if tag == 0 { break; }
                match tag {
                    0x06 => symtab = val,
                    0x05 => strtab = val,
                    0x04 => hash = val,
                    _ => {}
                }
            }
            if symtab != 0 && strtab != 0 && hash != 0 {
                let nchain = read_u32(d, hash as usize + 4) as usize;
                for i in 0..nchain {
                    let so = symtab as usize + i * 24;
                    if so + 24 > d.len() { break; }
                    let n = read_u32(d, so) as usize;
                    if n == 0 { continue; }
                    let sname = read_cstr(d, strtab as usize + n);
                    let value = read_u64(d, so + 8) as u32;
                    let size = read_u64(d, so + 16) as u32;
                    if value == 0 { continue; }
                    syms.push((base + value, size, sname, name.to_string()));
                }
                println!("  {} symbols ({} chain entries)", syms.len(), nchain);
            }
        }
        base = (image_end + 0xfff) & !0xfff;
    }
    syms.sort();
    let mut out = String::new();
    for (a, s, n, m) in &syms {
        out.push_str(&format!("{a:#010x} {s:#x} {m} {n}\n"));
    }
    fs::write(format!("{out_dir}/symbols.txt"), out).unwrap();
    println!("wrote {} symbols", syms.len());
}
