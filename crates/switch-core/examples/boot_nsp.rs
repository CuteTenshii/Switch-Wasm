//! Boot a real game from an NSP: find its Program NCA, decrypt the ExeFS,
//! extract `main` and run it in the interpreter — the CLI equivalent of the
//! browser's NSP/NCA panel "Launch" button, useful for debugging without a
//! browser.
//!
//! Usage: cargo run -p switch-core --example boot_nsp -- <path.nsp> <prod.keys> [title.keys] [max_steps]
//!
//! `DUMP=<base>[+<hex>][:<hex length>][,...]` hex-dumps guest memory wherever
//! the run stops, alongside the registers and the call stack. The base may be
//! a register, so the object a fault was holding can be dumped without
//! knowing its address first: `DUMP=x23+0x1830:0x40`.
//!
//! `TRAP_WRITE=<addr>:<hex size>` names the code that writes into a region —
//! the pc and call stack of the first writes into it, which is how a buffer
//! nobody admits to owning gets an owner. Same spelling as `screenshot_nsp`'s.

mod common;

use std::env;
use switch_core::cpu::Cpu;
use switch_core::nsp::Pfs0;
use switch_core::source::{ByteSource, FileSource, Window};

/// Where one `DUMP=` entry starts. A fault's interesting address is usually
/// one the faulting code was holding rather than one known before the run —
/// the object a null came out of is whatever `x23` happened to be — so a base
/// may name a register as well as an absolute address.
enum DumpBase {
    Absolute(u32),
    Register(u8),
    StackPointer,
    ProgramCounter,
}

/// One region to hex-dump once the run has stopped:
/// `DUMP=<base>[+<hex>][:<hex length>][,...]`, where `<base>` is `x0`..`x30`,
/// `sp`, `pc` or a hex address — `DUMP=x23+0x1830:0x40,0x10c2e870`.
struct DumpSpec {
    label: String,
    base: DumpBase,
    offset: i64,
    len: u32,
}

fn parse_hex(text: &str) -> Option<u64> {
    let text = text.trim();
    u64::from_str_radix(text.trim_start_matches("0x"), 16).ok()
}

fn parse_dump_specs(spec: &str) -> Vec<DumpSpec> {
    /// Enough to see a small object and its first few pointers. A region
    /// worth more than this is one the caller knows the size of.
    const DEFAULT_LEN: u32 = 0x40;
    spec.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let (addr, len) = match entry.split_once(':') {
                Some((addr, len)) => (addr, parse_hex(len)? as u32),
                None => (entry, DEFAULT_LEN),
            };
            // The sign has to be found from the right: `x23+0x10` splits on
            // the `+`, but a bare `0x...` address must not be split on a `-`
            // that is part of nothing at all.
            let (base, offset) = match addr.rfind(['+', '-']).filter(|&i| i > 0) {
                Some(i) => {
                    let value = parse_hex(&addr[i + 1..])? as i64;
                    (&addr[..i], if addr.as_bytes()[i] == b'-' { -value } else { value })
                }
                None => (addr, 0),
            };
            let base = match base.trim() {
                "sp" => DumpBase::StackPointer,
                "pc" => DumpBase::ProgramCounter,
                name if name.starts_with('x') => DumpBase::Register(name[1..].parse().ok()?),
                absolute => DumpBase::Absolute(parse_hex(absolute)? as u32),
            };
            Some(DumpSpec { label: entry.to_string(), base, offset, len })
        })
        .collect()
}

/// Hex-dump every `DUMP=` region, four words and their ASCII to a line.
///
/// Words rather than bytes because what is being read here is almost always a
/// structure: a null in a field is the thing being looked for, and a run of
/// pointers is what says a table was populated. The ASCII column is there
/// because the other half of what turns up in guest memory is names.
fn dump_regions(cpu: &Cpu, specs: &[DumpSpec]) {
    for spec in specs {
        let base = match spec.base {
            DumpBase::Absolute(addr) => u64::from(addr),
            DumpBase::Register(reg) => cpu.read_x(reg),
            DumpBase::StackPointer => cpu.sp(),
            DumpBase::ProgramCounter => u64::from(cpu.get_pc()),
        };
        let at = (base as i64).wrapping_add(spec.offset) as u32;
        println!("[dump] {} = {at:#010x} ({:#x} bytes)", spec.label, spec.len);
        for line in (0..spec.len).step_by(16) {
            let addr = at.wrapping_add(line);
            let mut words = String::new();
            let mut ascii = String::new();
            for word in 0..4u32 {
                let value = cpu.mem.read_u32(addr.wrapping_add(word * 4)).unwrap_or(0);
                words.push_str(&format!(" {value:08x}"));
                for byte in value.to_le_bytes() {
                    ascii.push(if (0x20..0x7f).contains(&byte) { byte as char } else { '.' });
                }
            }
            println!("  {addr:#010x}:{words}  {ascii}");
        }
    }
}

/// Everything worth knowing about where a run stopped: the registers, the
/// call stack, and whatever `DUMP=` asked for.
///
/// Printed for a fault, a halt and an exhausted step budget alike. A run that
/// stops any of those three ways stops somewhere, and the address a fault was
/// holding is no easier to guess than the one a hang was.
fn dump_stop_state(cpu: &Cpu, specs: &[DumpSpec]) {
    if specs.is_empty() {
        return;
    }
    print!("{}", cpu.reg_dump());
    println!(
        "backtrace: {}",
        cpu.backtrace(24).iter().map(|pc| format!("{pc:#010x}")).collect::<Vec<_>>().join(" <- ")
    );
    dump_regions(cpu, specs);
}

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

    let mut keys = common::keys(prod_path, title_path);

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
    common::load_fallback_font(&mut cpu);
    // System data archives, if the host pointed at a firmware dump. A title
    // mounts these by data id for content that is not its own — an applet's
    // shared assets, the Mii and amiibo models. Each is another Data NCA, and
    // is served straight off disk rather than read in.
    let registered = common::register_firmware(&mut cpu, &keys);
    if registered > 0 {
        println!("registered {registered} system data archive(s)");
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
    let dump_specs = std::env::var("DUMP").map(|s| parse_dump_specs(&s)).unwrap_or_default();
    let trap_write = std::env::var("TRAP_WRITE").ok().and_then(|v| {
        let (addr, size) = v.split_once(':')?;
        Some((parse_hex(addr)? as u32, parse_hex(size)? as u32))
    });
    if let Some((addr, size)) = trap_write {
        cpu.mem.watch_writes(addr, size);
    }
    // Only the first few: what is wanted is which code reached a region
    // first, and a region being written to at all is usually a loop.
    let mut traps = 0u32;
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
            dump_stop_state(&cpu, &dump_specs);
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
            Ok(()) => {
                if let Some(at) = cpu.mem.take_watch_hit() {
                    if traps < 24 {
                        println!(
                            "[trap] wrote {at:#010x} = {:#010x} at step {done} pc={pc:#010x} bt={}",
                            cpu.mem.read_u32(at & !3).unwrap_or(0),
                            cpu.backtrace(12)
                                .iter()
                                .map(|pc| format!("{pc:#010x}"))
                                .collect::<Vec<_>>()
                                .join(" <- "),
                        );
                        traps += 1;
                    }
                }
            }
            Err(e) => {
                println!("FAULT at step {done} pc={:#x}: {e}", cpu.get_pc());
                dump_stop_state(&cpu, &dump_specs);
                break;
            }
        }
        if cpu.halted {
            println!("HALTED at step {done} pc={:#x} x0={:#x}", cpu.get_pc(), cpu.read_x(0));
            dump_stop_state(&cpu, &dump_specs);
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
        if !cpu.nv.gpu.framebuffer.is_empty() {
            common::write_ppm(&out, &cpu.nv.gpu.framebuffer);
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
