//! Boot a bare Program NCA — a system applet such as the Home Menu, which
//! ships inside firmware rather than in an NSP — and write the Nth presented
//! frame to a PPM:
//! `screenshot_nca <path.nca> <prod.keys> <title.keys> <out.ppm> [frame]`.
//!
//! The counterpart to `screenshot_nsp`. An applet is the case that matters for
//! the system UI and the one an NSP-only runner cannot reach, because there is
//! no PFS0 around it to find a Program NCA in.
//!
//! `SWITCH_FIRMWARE=<dir>` registers the system data archives; an applet needs
//! them far more than a title does, since its fonts, icons and settings all
//! live there.
use std::collections::HashMap;
use std::env;
use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nsp::Pfs0;

fn main() {
    let a: Vec<String> = env::args().collect();
    let raw = fs::read(&a[1]).unwrap();
    let mut keys = switch_core::keys::keyset_from_prod(&switch_core::keys::parse_keys_file(
        &fs::read_to_string(&a[2]).unwrap(),
    ));
    keys.title_keys = switch_core::keys::keyset_from_title(&switch_core::keys::parse_keys_file(
        &fs::read_to_string(&a[3]).unwrap(),
    ));
    let out = a[4].clone();
    let want: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    let nca = switch_core::nca::Nca::parse_with_keys(&raw, Some(&keys)).unwrap();
    let exefs = nca
        .decrypt_pfs0_section(&raw, &keys, nca.exefs_section_index().unwrap())
        .unwrap();
    let pf = Pfs0::parse(&exefs).unwrap();
    const ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "sdk",
    ];
    let modules: Vec<(&str, &[u8])> = ORDER
        .iter()
        .filter_map(|&n| {
            let e = pf.find(n)?;
            Some((n, &exefs[e.offset as usize..(e.offset + e.size) as usize]))
        })
        .collect();

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    match nca.romfs_section_index() {
        Some(i) => match nca.decrypt_romfs_section(&raw, &keys, i) {
            Ok(r) => {
                println!("[romfs] section {i}: {} bytes", r.len());
                cpu.set_romfs(r);
            }
            Err(e) => println!("[romfs] section {i} failed to decrypt: {e}"),
        },
        None => println!("[romfs] no romfs section"),
    }
    let font = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/assets/font.ttf");
    if let Ok(b) = fs::read(font) {
        cpu.set_shared_font(b);
    }
    if let Ok(dir) = env::var("SWITCH_FIRMWARE") {
        for entry in fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("nca") {
                continue;
            }
            let Ok(src) = switch_core::source::FileSource::open(&path) else { continue };
            let Ok(ar) = switch_core::nca::Nca::parse_source(&src, Some(&keys)) else { continue };
            use switch_core::nca::ContentType;
            if !matches!(ar.content_type, ContentType::Data | ContentType::PublicData) {
                continue;
            }
            let Some(idx) = ar.romfs_section_index() else { continue };
            if let Ok(r) = ar.romfs_source(src, &keys, idx) {
                cpu.add_data_archive(ar.title_id, Box::new(r));
            }
        }
    }
    cpu.set_program_id(nca.program_id);
    cpu.boot_retail_program(&modules).unwrap();

    let trap: Option<(u32, u32)> = env::var("TRAP_WRITE").ok().and_then(|v| {
        let (s, n) = v.split_once(':')?;
        let s = u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()?;
        let n = u32::from_str_radix(n.trim().trim_start_matches("0x"), 16).ok()?;
        Some((s, s + n))
    });
    if let Some((lo, hi)) = trap {
        cpu.mem.watch_writes(lo, hi - lo);
    }
    // `WATCH_PC=<addr>[,...]` reports the guest backtrace the first few times
    // execution reaches an address. Finding who calls a thin IPC stub is not a
    // static question here -- they are reached through vtables, so nothing in
    // the image points at them.
    let watch_pc: Vec<u32> = env::var("WATCH_PC")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|a| u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default();
    let mut watch_hits = 0u32;
    // `COVER=<lo>:<hi>` records which instructions in a range ever execute.
    // Chasing a call chain statically stops the moment a function has no
    // reference anywhere in the image; this answers "which of these ran" for a
    // whole class at once.
    let cover: Option<(u32, u32)> = env::var("COVER").ok().and_then(|v| {
        let (a, b) = v.split_once(':')?;
        let a = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
        let b = u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?;
        Some((a, b))
    });
    let mut covered: Vec<bool> =
        cover.map(|(a, b)| vec![false; ((b - a) / 4) as usize]).unwrap_or_default();
    // `START_THREADS=<step>` makes every created-but-never-started thread
    // runnable once, at that step.
    let start_threads: Option<u64> = env::var("START_THREADS").ok().and_then(|v| v.parse().ok());
    let mut started_threads = false;
    let mut share = [0u64; 32];
    let mut hot: HashMap<(usize, u32), u64> = HashMap::new();
    let mut traps = 0u32;
    let mut done = 0u64;
    let budget: u64 = env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(40_000_000_000);
    while !cpu.halted && cpu.nv.gpu.frames < want && done < budget {
        if let Some(at) = cpu.mem.take_watch_hit() {
            let v = cpu.mem.read_u32(at & !3).unwrap_or(0);
            if traps < 24 && v != 0 {
                println!(
                    "[trap] wrote {at:#x} = {v:#010x} at step {done} pc={:#x} bt={:x?}",
                    cpu.get_pc(),
                    cpu.backtrace(6)
                );
                traps += 1;
            }
        }
        if done % 4096 == 0 {
            let t = cpu.current_thread_index();
            if t < share.len() {
                share[t] += 1;
            }
            *hot.entry((t, cpu.get_pc())).or_insert(0u64) += 1;
        }
        if watch_hits < 8 && !watch_pc.is_empty() && watch_pc.contains(&cpu.get_pc()) {
            println!(
                "[watch-pc] {:#x} at step {done} bt={:x?}",
                cpu.get_pc(),
                cpu.backtrace(10)
            );
            watch_hits += 1;
        }
        if let Some(at) = start_threads {
            if !started_threads && done >= at {
                started_threads = true;
                println!("[threads] force-started {}", cpu.start_created_threads());
            }
        }
        if let Some((lo, hi)) = cover {
            let pc = cpu.get_pc();
            if pc >= lo && pc < hi {
                covered[((pc - lo) / 4) as usize] = true;
            }
        }
        if cpu.step().is_err() {
            break;
        }
        done += 1;
    }
    if let Some((lo, _)) = cover {
        let mut run: Option<u32> = None;
        for (i, &hit) in covered.iter().enumerate() {
            let at = lo + i as u32 * 4;
            match (hit, run) {
                (true, None) => run = Some(at),
                (false, Some(start)) => {
                    println!("[cover] {start:#x}..{at:#x}");
                    run = None;
                }
                _ => {}
            }
        }
        if let Some(start) = run {
            println!("[cover] {start:#x}..end");
        }
    }
    println!("[threads] sampled share = {:?}", &share[..]);
    println!("[bt] {:#x} <- {:x?}", cpu.get_pc(), cpu.backtrace(12));
    let hot_snapshot = hot.clone();
    let mut top: Vec<_> = hot.into_iter().collect();
    top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let mut by_page: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for (&(_, pc), &n) in &hot_snapshot {
        *by_page.entry(pc & !0xFFF).or_insert(0) += n;
    }
    let mut pages: Vec<_> = by_page.into_iter().collect();
    pages.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    for (page, n) in pages.into_iter().take(20) {
        println!("[hot-page] {page:#x} {n}");
    }
    for ((t, pc), n) in top.into_iter().take(10) {
        println!("[hot] thread {t} pc={pc:#x} {n}");
    }
    print!("{}", cpu.thread_dump());
    println!("steps={done} frames={} stats={:?}", cpu.nv.gpu.frames, cpu.nv.gpu.stats);
    let fb = &cpu.nv.gpu.framebuffer;
    if fb.is_empty() {
        println!("no frame");
        return;
    }
    let mut ppm = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    for px in &fb.pixels {
        ppm.extend_from_slice(&[*px as u8, (*px >> 8) as u8, (*px >> 16) as u8]);
    }
    fs::write(&out, ppm).unwrap();
    let lit = fb.pixels.iter().filter(|p| **p & 0x00FF_FFFF != 0).count();
    println!("wrote {out}: {}x{}, {lit}/{} non-black", fb.width, fb.height, fb.pixels.len());
}
