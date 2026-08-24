//! Boot a real game from an NSP: find its Program NCA, decrypt the ExeFS,
//! extract `main` and run it in the interpreter — the CLI equivalent of the
//! browser's NSP/NCA panel "Launch" button, useful for debugging without a
//! browser.
//!
//! Usage: cargo run -p switch-core --example boot_nsp -- <path.nsp> <prod.keys> [title.keys] [max_steps]

use std::env;
use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, FileSource, Window};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: boot_nsp <path.nsp> <prod.keys> [title.keys] [max_steps]");
        std::process::exit(1);
    }
    let nsp_path = &args[1];
    let prod_path = &args[2];
    let title_path = args.get(3).filter(|s| !s.chars().all(|c| c.is_ascii_digit()));
    let max_steps: u64 = args
        .iter()
        .find_map(|s| s.parse::<u64>().ok())
        .unwrap_or(2_000_000);

    // Read the container off disk rather than into memory. A retail Program
    // NCA is the whole game — Just Dance 2019's is 7.2 GiB — and reading it in
    // and then materialising its RomFS beside it wants more RAM than the title
    // could ever touch, on a machine that also has to hold the guest. The
    // browser has always streamed this; there is no reason the CLI should be
    // the build that cannot boot a big title.
    let src = FileSource::open(nsp_path).expect("open nsp");
    println!("NSP: {} bytes", src.len());
    let pfs0 = Pfs0::read_from(&src).expect("parse nsp");
    println!("{} files in NSP", pfs0.files.len());

    let prod_text = fs::read_to_string(prod_path).expect("read prod.keys");
    let prod_entries = switch_core::keys::parse_keys_file(&prod_text);
    let mut keys = switch_core::keys::keyset_from_prod(&prod_entries);
    if let Some(tp) = title_path {
        let title_text = fs::read_to_string(tp).expect("read title.keys");
        let title_entries = switch_core::keys::parse_keys_file(&title_text);
        keys.title_keys = switch_core::keys::keyset_from_title(&title_entries);
    }

    // Find the Program NCA: parse every .nca's header and pick the one whose
    // content type is Program. (cnmt.nca entries are tiny metadata records,
    // control.nca is the icon/nacp, so this reliably picks the real one.)
    let mut program: Option<usize> = None;
    for (i, f) in pfs0.files.iter().enumerate() {
        if !f.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }
        let Ok(window) = pfs0.file_source(&src, i) else {
            continue;
        };
        match switch_core::nca::Nca::parse_source(&window, Some(&keys)) {
            Ok(nca) => {
                println!(
                    "{}: content_type={:?} title_id={:016x} size={}",
                    f.name, nca.content_type, nca.title_id, f.size
                );
                if nca.content_type == switch_core::nca::ContentType::Program {
                    program = Some(i);
                }
            }
            Err(e) => println!("{}: parse failed: {}", f.name, e),
        }
    }

    let Some(program_index) = program else {
        println!("no Program NCA found");
        return;
    };
    let program_file = &pfs0.files[program_index];
    println!("--- decrypting Program NCA: {} ---", program_file.name);
    let program_window = pfs0
        .file_source(&src, program_index)
        .expect("window over the program nca");
    let nca =
        switch_core::nca::Nca::parse_source(&program_window, Some(&keys)).expect("parse program nca");

    // Title-key crypto: no key-area unlock needed, but the title key itself
    // has to come from somewhere. Scene NSP releases bundle the ticket right
    // next to the content, so try that before giving up.
    if nca.has_rights_id() && keys.resolved_title_key(&nca.rights_id).is_none() {
        match switch_core::ticket::find_and_decrypt_title_key_from(
            &nca.rights_id,
            &pfs0.files,
            &src,
            &keys,
        ) {
            Ok(title_key) => {
                println!(
                    "resolved title key from bundled ticket: {}",
                    title_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
                );
                keys.add_resolved_title_key(nca.rights_id, title_key);
            }
            Err(e) => println!("ticket resolution failed: {}", e),
        }
    }

    let exefs_index = nca.exefs_section_index().expect("no exefs section");
    let exefs = match nca.read_pfs0_section(&program_window, &keys, exefs_index) {
        Ok(v) => v,
        Err(e) => {
            println!("read_pfs0_section FAILED: {}", e);
            return;
        }
    };
    println!("ExeFS decrypted + hash-verified: {} bytes", exefs.len());
    let exefs_pfs0 = Pfs0::parse(&exefs).expect("parse exefs pfs0");
    for ef in &exefs_pfs0.files {
        println!("  exefs file: {} ({} bytes)", ef.name, ef.size);
    }
    const MODULE_ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5",
        "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk",
    ];
    let modules: Vec<(&str, &[u8])> = MODULE_ORDER
        .iter()
        .filter_map(|&name| {
            let f = exefs_pfs0.find(name)?;
            let start = f.offset as usize;
            let end = start + f.size as usize;
            Some((name, &exefs[start..end]))
        })
        .collect();
    println!("--- booting {} modules: {:?} ---", modules.len(), modules.iter().map(|(n, _)| *n).collect::<Vec<_>>());

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    // The title's save-data quota, which `IApplicationFunctions::GetSaveDataSize`
    // reports. It is declared in the NACP, and the NACP is in the *Control*
    // NCA rather than the Program one booted above — so it has to be read
    // separately, and a container without one leaves the CPU's default in
    // place rather than reporting a size this title never asked for.
    match switch_core::control::find_control_nca(&pfs0.files, &src, &keys) {
        Some((index, _)) => {
            let control_window = pfs0
                .file_source(&src, index)
                .expect("window over the control nca");
            match switch_core::control::Control::from_source(control_window, &keys) {
                Ok(control) => {
                    let quota = switch_core::cpu::SaveDataQuota::from(&control.nacp);
                    println!(
                        "save data: {} bytes (+{} journal), extendable to {} (+{}); \
                         cache storage: {} x {} bytes — all from the NACP",
                        quota.size,
                        quota.journal_size,
                        quota.size_max,
                        quota.journal_size_max,
                        quota.cache_storage_index_max,
                        quota.cache_storage_size_max,
                    );
                    cpu.set_save_data_quota(quota);
                }
                Err(e) => println!("Control NCA unreadable, using default save sizes: {}", e),
            }
        }
        None => println!("no Control NCA in this container: using default save sizes"),
    }
    // RomFS is optional (Meta/Control-only content, or a title with no
    // assets of its own, has none) and a failure to decrypt it shouldn't
    // block booting — the title just won't have its asset storage mounted.
    // Handed to the CPU as a source rather than decrypted up front, for the
    // same reason the container is streamed: the guest reads its RomFS a range
    // at a time through `IStorage`, so there is nothing to gain by holding the
    // whole game in memory and a title's worth of RAM to lose. This needs its
    // own handle on the file, since the CPU keeps the source for the whole run
    // and cannot borrow the one the scan above is using.
    if let Some(romfs_index) = nca.romfs_section_index() {
        let owned = FileSource::open(nsp_path).expect("reopen nsp for romfs");
        let owned_window = Window::new(owned, program_file.offset, program_file.size, "program nca")
            .expect("window over the program nca");
        match nca.romfs_source(owned_window, &keys, romfs_index) {
            Ok(romfs) => {
                println!("RomFS: {} bytes, streamed", romfs.len());
                cpu.set_romfs_source(Box::new(romfs));
            }
            Err(e) => println!("romfs_source FAILED: {}", e),
        }
    } else {
        println!("no RomFS section in this NCA");
    }
    // The system fonts `pl:u` hands out. Without them a title that draws
    // text waits for a font that never arrives — the browser stages one at
    // startup, so a native run that skips it fails in a way the real
    // frontend never would.
    let font = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/assets/font.ttf");
    match fs::read(font) {
        Ok(bytes) => cpu.set_shared_font(bytes),
        Err(e) => println!("no font at {font} ({e}): text will not render"),
    }
    // System data archives, if the host pointed at a firmware dump. A title
    // mounts these by data id for content that is not its own — an applet's
    // shared assets, the Mii and amiibo models. Each is another Data NCA, and
    // is served straight off disk rather than read in.
    if let Ok(dir) = env::var("SWITCH_FIRMWARE") {
        let mut registered = 0;
        let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir}: {e}"));
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("nca") {
                continue;
            }
            let Ok(src) = switch_core::source::FileSource::open(&path) else {
                continue;
            };
            let Ok(archive) = switch_core::nca::Nca::parse_source(&src, Some(&keys)) else {
                continue;
            };
            use switch_core::nca::ContentType;
            if !matches!(archive.content_type, ContentType::Data | ContentType::PublicData) {
                continue;
            }
            let Some(idx) = archive.romfs_section_index() else {
                continue;
            };
            if let Ok(romfs) = archive.romfs_source(src, &keys, idx) {
                cpu.add_data_archive(archive.title_id, Box::new(romfs));
                registered += 1;
            }
        }
        println!("registered {registered} system data archive(s) from {dir}");
    }

    // The address space this title gets, which its own manifest decides — a
    // title that declares no system resource keeps the plain heap and the
    // larger total memory. Must precede `boot_retail_program`, since
    // `nn::init` reads the resulting figures as soon as it runs.
    let system_resource = switch_core::npdm::Npdm::system_resource_size_of(&exefs_pfs0, &exefs);
    println!(
        "NPDM system resource: {system_resource:#x} — {}",
        if system_resource == 0 { "plain heap" } else { "virtual address memory" }
    );
    cpu.set_system_resource_size(system_resource);

    cpu.set_program_id(nca.program_id);
    let loaded = cpu.boot_retail_program(&modules).expect("boot modules");
    for m in &loaded {
        println!("module base={:#010x} entry={:#010x}", m.base, m.entry);
    }

    let mut done = 0u64;
    let watch: std::collections::HashSet<u32> = std::env::var("WATCH")
        .map(|s| {
            s.split(',')
                .filter_map(|x| u32::from_str_radix(x.trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default();
    loop {
        if done >= max_steps {
            println!("STOPPED at step budget {done} pc={:#x}", cpu.get_pc());
            // A run that stops on the budget rather than on a fault has almost
            // always stopped making progress, and where each *thread* is says
            // more about why than where the one running thread is.
            print!("{}", cpu.thread_dump());
            break;
        }
        let pc = cpu.get_pc();
        if watch.contains(&pc) {
            println!(
                "[watch] step={done} pc={pc:#x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x4={:#x} x5={:#x} x6={:#x} x7={:#x} bt={:?}",
                cpu.read_x(0),
                cpu.read_x(1),
                cpu.read_x(2),
                cpu.read_x(3),
                cpu.read_x(4),
                cpu.read_x(5),
                cpu.read_x(6),
                cpu.read_x(7),
                cpu.backtrace(16)
            );
            if pc == 0xce6ab00 {
                let f = cpu.read_x(0);
                let mut s = String::new();
                for i in 0..48u64 {
                    match cpu.mem.read_u64((f as u32).wrapping_add(i as u32)) {
                        Ok(0) => break,
                        Ok(b) => s.push((b & 0xff) as u8 as char),
                        Err(_) => break,
                    }
                }
                println!("  ** nn_result_abort msg@{f:#x} = {:?}", s);
            }
        }
        match cpu.step() {
            Ok(()) => {}
            Err(e) => {
                println!("FAULT at step {done} pc={:#x}: {e}", cpu.get_pc());
                break;
            }
        }
        if cpu.halted {
            println!("HALTED at step {done} pc={:#x} x0={:#x}", cpu.get_pc(), cpu.read_x(0));
            if std::env::var("DUMP_REGS").is_ok() {
                println!("{}", cpu.reg_dump());
                for a in [0xdffb790u32, 0xdffdaf0, 0xdffd080, 0xdffdaa0, 0xdffc2a8, 0xdffc298, 0xdffc2c0, 0xdffc2d0, 0xdffd028, 0xdffd018, 0xdffd020, 0xdffd010] {
                    if let Ok(v) = cpu.mem.read_u64(a) {
                        println!("data@{a:#x} -> fx {v:#x}");
                    }
                }
                if let Ok(v) = cpu.mem.read_u32(0x0e06e2ac) {
                    println!("data@0x0e06e2ac (result) = {v:#010x}");
                }
                let tls = cpu.tls_base();
                if let Ok(tp) = cpu.mem.read_u64(tls + 0x1f8) {
                    println!("TLS tls_tp = {tp:#x}");
                    if let Ok(succ) = cpu.mem.read_u32(tp as u32 + 0x1b0) {
                        println!("expected success code @tls_tp+0x1b0 = {succ:#010x}");
                    }
                }
            }
            break;
        }
        done += 1;
    }
    if let Ok(out) = std::env::var("SHOT") {
        let fb = &cpu.nv.gpu.framebuffer;
        if !fb.is_empty() {
            let mut ppm = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
            for px in &fb.pixels {
                ppm.extend_from_slice(&[*px as u8, (*px >> 8) as u8, (*px >> 16) as u8]);
            }
            std::fs::write(&out, ppm).expect("write ppm");
            let lit = fb.pixels.iter().filter(|p| **p & 0x00FF_FFFF != 0).count();
            println!("wrote {out}: {}x{}, {lit}/{} non-black", fb.width, fb.height, fb.pixels.len());
        }
    }
    println!("guest RAM touched: {} MiB", cpu.mem.mapped_bytes() / (1024 * 1024));
    println!("frames presented: {}", cpu.nv.gpu.frames);
    println!("gpu stats: {:?}", cpu.nv.gpu.stats);
    println!("--- program console output ({} bytes) ---", cpu.out.len());
    let out = String::from_utf8_lossy(&cpu.out);
    for line in out.lines().take(80) {
        println!("  {line}");
    }
}
