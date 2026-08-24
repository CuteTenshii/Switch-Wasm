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
    if nca.has_rights_id() && keys.resolved_title_key(&nca.rights_id).is_none() {
        let tk = switch_core::ticket::find_and_decrypt_title_key(&nca.rights_id, &pfs0.files, &nsp, &keys).unwrap();
        keys.add_resolved_title_key(nca.rights_id, tk);
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
    // The address space this title gets is chosen by its own manifest — see
    // `MemoryLayout`. Must precede `boot_retail_program`.
    cpu.set_system_resource_size(switch_core::npdm::Npdm::system_resource_size_of(&pf, &exefs));
    cpu.set_program_id(nca.program_id);
    cpu.boot_retail_program(&modules).unwrap();
    // `WATCH_MEM=<hex addr>` reports the first step at which a 4 KiB window
    // there stops being all zeroes. A GPU reading zeroes is either looking at
    // the wrong memory or at memory nothing has filled yet, and this is what
    // tells the two apart.
    let watch: Option<u32> = env::var("WATCH_MEM").ok()
        .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok());
    let trap: Option<(u32, u32)> = env::var("TRAP_WRITE").ok().and_then(|v| {
        let (a, n) = v.split_once(':')?;
        let a = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
        let n = u32::from_str_radix(n.trim().trim_start_matches("0x"), 16).ok()?;
        Some((a, a + n))
    });
    // `POKE_TRI=<cpu addr>` fills a 3-vertex, 60-byte-stride array with a
    // full-screen triangle. A buffer the guest binds but never writes leaves
    // two possibilities open -- the upload is missing, or the draw would draw
    // nothing anyway -- and putting real geometry there separates them.
    let poke: Option<u32> = env::var("POKE_TRI").ok()
        .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok());
    if let Some((lo, hi)) = trap {
        cpu.mem.watch_writes(lo, hi - lo);
    }
    let mut share = [0u64; 32];
    let mut traps = 0u32;
    let mut seen_nonzero = false;
    let mut done = 0u64;
    while !cpu.halted && cpu.nv.gpu.frames < want && done < 40_000_000_000 {
        // `TRAP_WRITE=<addr>:<size>` reports the guest PC of each of the first
        // writes into a region -- which code owns a buffer, rather than just
        // whether it was touched.
        if let Some(at) = cpu.mem.take_watch_hit() {
            if traps < 24 {
                let v = cpu.mem.read_u32(at & !3).unwrap_or(0);
                let src = cpu.reg(1) as u32;
                let peek: Vec<u32> = (0..4).map(|k| cpu.mem.read_u32(src.wrapping_add(k * 4)).unwrap_or(0)).collect();
                println!("[trap] wrote {at:#x} = {v:#010x} at step {done} pc={:#x} x0={:#x} x1={src:#x} x2={:#x} src[..4]={peek:08x?} bt={:x?}",
                    cpu.get_pc(), cpu.reg(0), cpu.reg(2), cpu.backtrace(4));
                traps += 1;
            }
        }
        if let Some(w) = watch {
            if !seen_nonzero && done % 4096 == 0
                && (0..0x1000u32).step_by(4).any(|k| cpu.mem.read_u32(w + k).unwrap_or(0) != 0)
            {
                println!("[watch-mem] {w:#x} first non-zero at step {done}");
                seen_nonzero = true;
            }
        }
        if let Some(base) = poke {
            if done % (1 << 22) == 0 {
                const POS: [[f32; 3]; 3] = [[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]];
                const UV: [[f32; 2]; 3] = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
                for v in 0..3u32 {
                    let at = base + v * 60;
                    for c in 0..3u32 {
                        let _ = cpu.mem.write_u32(at + c * 4, POS[v as usize][c as usize].to_bits());
                    }
                    for c in 0..2u32 {
                        let bits = UV[v as usize][c as usize].to_bits();
                        let _ = cpu.mem.write_u32(at + 0x2c + c * 4, bits);
                        let _ = cpu.mem.write_u32(at + 0x34 + c * 4, bits);
                    }
                }
            }
        }
        if done % 4096 == 0 {
            let t = cpu.current_thread_index();
            if t < share.len() { share[t] += 1; }
        }
        if cpu.step().is_err() { break; }
        done += 1;
    }
    println!("[threads] sampled share = {:?}", &share[..]);
    print!("{}", cpu.thread_dump());
    if watch.is_some() && !seen_nonzero {
        println!("[watch-mem] never non-zero");
    }
    println!("steps={done} frames={} stats={:?}", cpu.nv.gpu.frames, cpu.nv.gpu.stats);
    // `DUMP_MEM=<addr>[,<addr>]` reads three 60-byte rows there as floats,
    // which is the quickest way to tell a vertex buffer from whatever else
    // happened to be nearby: real positions are ordinary numbers, and a
    // structure reinterpreted as float is a wall of denormals.
    // `SCAN_MEM=<addr>:<size>` lists the non-zero spans in a region, which is
    // how you find the buffer you meant among the ones you did not.
    if let Ok(v) = env::var("SCAN_MEM") {
        let (a, n) = v.split_once(':').unwrap_or((v.as_str(), "10000"));
        let at = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).unwrap_or(0);
        let len = u32::from_str_radix(n.trim().trim_start_matches("0x"), 16).unwrap_or(0x10000);
        let mut spans = Vec::new();
        let mut run: Option<u32> = None;
        for off in (0..len).step_by(4) {
            let nz = cpu.mem.read_u32(at.wrapping_add(off)).unwrap_or(0) != 0;
            match (nz, run) {
                (true, None) => run = Some(off),
                (false, Some(st)) => { spans.push((at + st, off - st)); run = None; }
                _ => {}
            }
        }
        if let Some(st) = run { spans.push((at + st, len - st)); }
        println!("  {} non-zero spans in {at:#x}+{len:#x}", spans.len());
        for (a, l) in spans.iter().take(40) { println!("    {a:#x} .. +{l:#x}"); }
    }
    if let Ok(v) = env::var("DUMP_MEM") {
        for spec in v.split(',') {
            let at = u32::from_str_radix(spec.trim().trim_start_matches("0x"), 16).unwrap_or(0);
            let f: Vec<f32> = (0..45u32).map(|k| f32::from_bits(cpu.mem.read_u32(at + k * 4).unwrap_or(0))).collect();
            println!("  {at:#x} as f32: {:?}", &f[..15]);
            println!("  {at:#x} +60    : {:?}", &f[15..30]);
            println!("  {at:#x} +120   : {:?}", &f[30..45]);
        }
    }
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
