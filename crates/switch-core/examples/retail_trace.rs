//! Boot a retail NSP like `boot_nsp`, but keep a ring buffer of the last N
//! executed instructions and dump it when the guest halts — the fastest way
//! to see how an `nnSdk` abort was reached without tracing 117M steps.
//!
//! Usage: retail_trace <nsp> <prod.keys> <title.keys> [tail_len]
//!   RING_FROM=<hex pc>  start recording only once this pc is first hit.

use std::env;
use std::fs;
use switch_core::cpu::Cpu;
use switch_core::nsp::Pfs0;

fn main() {
    let args: Vec<String> = env::args().collect();
    let nsp_data = fs::read(&args[1]).unwrap();
    let pfs0 = Pfs0::parse(&nsp_data).unwrap();
    let mut keys = switch_core::keys::keyset_from_prod(&switch_core::keys::parse_keys_file(
        &fs::read_to_string(&args[2]).unwrap(),
    ));
    keys.title_keys = switch_core::keys::keyset_from_title(&switch_core::keys::parse_keys_file(
        &fs::read_to_string(&args[3]).unwrap(),
    ));
    let tail: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4000);

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
        let tk = switch_core::ticket::find_and_decrypt_title_key(
            &nca.rights_id, &pfs0.files, &nsp_data, &keys,
        )
        .unwrap();
        keys.title_keys.push((nca.rights_id, tk));
    }
    let exefs = nca
        .decrypt_pfs0_section(raw, &keys, nca.exefs_section_index().unwrap())
        .unwrap();
    let exefs_pfs0 = Pfs0::parse(&exefs).unwrap();
    const ORDER: &[&str] = &[
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5",
        "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk",
    ];
    let modules: Vec<(&str, &[u8])> = ORDER
        .iter()
        .filter_map(|&n| {
            let f = exefs_pfs0.find(n)?;
            Some((n, &exefs[f.offset as usize..(f.offset + f.size) as usize]))
        })
        .collect();

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    if let Some(i) = nca.romfs_section_index() {
        if let Ok(r) = nca.decrypt_romfs_section(raw, &keys, i) {
            cpu.set_romfs(r);
        }
    }
    cpu.boot_retail_program(&modules).unwrap();

    let ring_from = env::var("RING_FROM")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
    let mut recording = ring_from.is_none();
    // Skip whole address ranges (rtld's lazy-binding resolver runs hundreds
    // of steps per call and would otherwise fill the whole ring).
    let ring_min = env::var("RING_MIN")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    let mut ring: std::collections::VecDeque<(u64, u32, [u64; 8])> =
        std::collections::VecDeque::with_capacity(tail + 1);

    // Stop this many steps after recording starts, so the ring holds the
    // *beginning* of a function rather than the last N steps before the halt.
    let stop_after = env::var("RING_STOP_AFTER").ok().and_then(|s| s.parse::<u64>().ok());
    let mut recorded = 0u64;
    let mut done = 0u64;
    while !cpu.halted && done < 400_000_000 {
        let pc = cpu.get_pc();
        if !recording && Some(pc) == ring_from {
            recording = true;
            // Whatever this function was called with: dump any argument that
            // points at a printable C string, which is how a path or a mount
            // name gets read out of a stuck `nn::fs` call — or the condition,
            // file, function and message of an `nn::diag` assertion, which sit
            // as far out as x5.
            for r in 0..8u8 {
                let addr = cpu.read_x(r) as u32;
                let mut sbuf = String::new();
                for i in 0..128u32 {
                    match cpu.mem.read_u8(addr.wrapping_add(i)) {
                        Ok(0) => break,
                        Ok(b) if (0x20..0x7f).contains(&b) => sbuf.push(b as char),
                        _ => {
                            sbuf.clear();
                            break;
                        }
                    }
                }
                if sbuf.len() >= 2 {
                    println!("x{r} = {addr:#x} -> {sbuf:?}");
                }
            }
        }
        if recording && pc >= ring_min {
            recorded += 1;
            if stop_after.is_some_and(|n| recorded > n) {
                println!("stopped {recorded} steps after RING_FROM");
                break;
            }
            if ring.len() == tail {
                ring.pop_front();
            }
            ring.push_back((
                done,
                pc,
                [cpu.read_x(0), cpu.read_x(1), cpu.read_x(2), cpu.read_x(3), cpu.read_x(8), cpu.read_x(19), cpu.read_x(30), cpu.sp()],
            ));
        }
        if let Err(e) = cpu.step() {
            println!("FAULT step {done} pc={:#x}: {e}", cpu.get_pc());
            break;
        }
        done += 1;
    }
    println!("halted at step {done} pc={:#x}", cpu.get_pc());
    println!("--- last {} steps ---", ring.len());
    for (s, pc, r) in &ring {
        println!(
            "{s} {pc:#010x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x8={:#x} x19={:#x} lr={:#x} sp={:#x}",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        );
    }
    println!("--- out ---\n{}", String::from_utf8_lossy(&cpu.out));
}
