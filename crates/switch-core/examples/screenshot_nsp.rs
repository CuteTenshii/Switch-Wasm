//! Boot a retail NSP and write the Nth presented frame to a PPM:
//! `screenshot_nsp <path.nsp> <prod.keys> <title.keys> <out.ppm> [frame]`.
//!
//! The counterpart to `screenshot` for a title rather than an NRO, and the
//! difference from `boot_nsp SHOT=` is that this stops *at* the frame rather
//! than at a step budget. That matters more than it sounds: a title needs
//! **seconds** of console time before its first frame, which is billions of
//! steps, and picking a budget that lands after it is guesswork — "A Short
//! Hike" reaches frame 30 at step 3.3 billion.
//!
//! `SWITCH_FIRMWARE=<dir>` registers the system data archives, as `boot_nsp`
//! does.
use std::env;
use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nsp::Pfs0;

fn main() {
    let a: Vec<String> = env::args().collect();
    let nsp = fs::read(&a[1]).unwrap();
    let pfs0 = Pfs0::parse(&nsp).unwrap();
    let mut keys = switch_core::keys::keyset_from_prod(&switch_core::keys::parse_keys_file(
        &fs::read_to_string(&a[2]).unwrap()));
    keys.title_keys = switch_core::keys::keyset_from_title(&switch_core::keys::parse_keys_file(
        &fs::read_to_string(&a[3]).unwrap()));
    let out = a[4].clone();
    let want: u64 = a[5].parse().unwrap();
    let f = pfs0.files.iter().find(|f| {
        if !f.name.to_ascii_lowercase().ends_with(".nca") { return false; }
        let end = (f.offset + f.size) as usize;
        if end > nsp.len() { return false; }
        matches!(switch_core::nca::Nca::parse_with_keys(&nsp[f.offset as usize..end], Some(&keys)),
            Ok(n) if n.content_type == switch_core::nca::ContentType::Program)
    }).expect("no Program NCA");
    let raw = &nsp[f.offset as usize..(f.offset + f.size) as usize];
    let nca = switch_core::nca::Nca::parse_with_keys(raw, Some(&keys)).unwrap();
    if nca.has_rights_id() && keys.title_key(&nca.rights_id).is_none() {
        let tk = switch_core::ticket::find_and_decrypt_title_key(&nca.rights_id, &pfs0.files, &nsp, &keys).unwrap();
        keys.title_keys.push((nca.rights_id, tk));
    }
    let exefs = nca.decrypt_pfs0_section(raw, &keys, nca.exefs_section_index().unwrap()).unwrap();
    let pf = Pfs0::parse(&exefs).unwrap();
    const ORDER: &[&str] = &["rtld","main","subsdk0","subsdk1","subsdk2","subsdk3","subsdk4","sdk"];
    let modules: Vec<(&str,&[u8])> = ORDER.iter().filter_map(|&n| {
        let e = pf.find(n)?;
        Some((n, &exefs[e.offset as usize..(e.offset + e.size) as usize]))
    }).collect();
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    if let Some(i) = nca.romfs_section_index() {
        if let Ok(r) = nca.decrypt_romfs_section(raw, &keys, i) { cpu.set_romfs(r); }
    }
    let font = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/assets/font.ttf");
    if let Ok(b) = fs::read(font) { cpu.set_shared_font(b); }
    if let Ok(dir) = env::var("SWITCH_FIRMWARE") {
        for entry in fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("nca") { continue; }
            let Ok(src) = switch_core::source::FileSource::open(&path) else { continue };
            let Ok(ar) = switch_core::nca::Nca::parse_source(&src, Some(&keys)) else { continue };
            use switch_core::nca::ContentType;
            if !matches!(ar.content_type, ContentType::Data | ContentType::PublicData) { continue; }
            let Some(idx) = ar.romfs_section_index() else { continue };
            if let Ok(r) = ar.romfs_source(src, &keys, idx) { cpu.add_data_archive(ar.title_id, Box::new(r)); }
        }
    }
    cpu.set_program_id(nca.program_id);
    cpu.boot_retail_program(&modules).unwrap();
    let mut done = 0u64;
    while !cpu.halted && cpu.nv.gpu.frames < want && done < 40_000_000_000 {
        if cpu.step().is_err() { break; }
        done += 1;
    }
    println!("steps={done} frames={} stats={:?}", cpu.nv.gpu.frames, cpu.nv.gpu.stats);
    let fb = &cpu.nv.gpu.framebuffer;
    if fb.is_empty() { println!("no frame"); return; }
    let mut ppm = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    for px in &fb.pixels {
        ppm.extend_from_slice(&[*px as u8, (*px >> 8) as u8, (*px >> 16) as u8]);
    }
    fs::write(&out, ppm).unwrap();
    let lit = fb.pixels.iter().filter(|p| **p & 0x00FF_FFFF != 0).count();
    println!("wrote {out}: {}x{}, {lit}/{} non-black", fb.width, fb.height, fb.pixels.len());
}
