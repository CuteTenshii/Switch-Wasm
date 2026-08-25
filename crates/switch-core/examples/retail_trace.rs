//! Boot a retail NSP like `boot_nsp`, but keep a ring buffer of the last N
//! executed instructions and dump it when the guest halts — the fastest way
//! to see how an `nnSdk` abort was reached without tracing 117M steps.
//!
//! Usage: retail_trace <nsp> <prod.keys> <title.keys> [tail_len]
//!   RING_FROM=<hex pc>  start recording only once this pc is first hit.
//!   MARK=<pc>[=name][,...]  print a line each time one of these pcs runs.
//!   MARK_DUMP=<reg>,<byte offset>,<words>  also dump memory at each mark.
mod common;


use std::env;
use switch_core::cpu::Cpu;

const USAGE: &str = "retail_trace <nsp> <prod.keys> <title.keys> [tail_len]";

fn main() {
    let title = common::Title::open_nsp(
        common::arg(1, USAGE),
        common::arg(2, USAGE),
        Some(common::arg(3, USAGE)),
    );
    let tail: usize = common::opt_num(4).unwrap_or(4000) as usize;

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    title.mount_romfs(&mut cpu);
    // The system fonts `pl:u` hands out. Without them a title that draws
    // text waits for a font that never arrives — the browser stages one at
    // startup, so a native run that skips it fails in a way the real
    // frontend never would.
    common::load_fallback_font(&mut cpu);
    title.boot(&mut cpu);

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
    // Print a line every time one of these addresses is executed. Pass a
    // comma-separated list of hex pcs (a function's entry, say) to watch a
    // whole API get called in order without recording every step in between.
    let marks: std::collections::HashMap<u32, String> = env::var("MARK")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|entry| {
                    let (pc, name) = entry.split_once('=').unwrap_or((entry, entry));
                    let pc = u32::from_str_radix(pc.trim().trim_start_matches("0x"), 16).ok()?;
                    Some((pc, name.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    // With MARK, also dump memory: `MARK_DUMP=<reg>,<signed byte offset>,<words>`
    // — the reply struct a marked function is about to read, say.
    let mark_dump: Option<(u8, i64, u32)> = env::var("MARK_DUMP").ok().and_then(|v| {
        let mut parts = v.split(',');
        Some((
            parts.next()?.trim().parse().ok()?,
            parts.next()?.trim().parse().ok()?,
            parts.next()?.trim().parse().ok()?,
        ))
    });
    let mut done = 0u64;
    while !cpu.halted && done < 400_000_000 {
        let pc = cpu.get_pc();
        if let Some(name) = marks.get(&pc) {
            println!(
                "[mark] {done} {name} x0={:#x} x1={:#x} x2={:#x} x3={:#x} lr={:#x}",
                cpu.read_x(0), cpu.read_x(1), cpu.read_x(2), cpu.read_x(3), cpu.read_x(30)
            );
            if let Some((reg, off, len)) = mark_dump {
                let at = (cpu.read_x(reg) as i64 + off) as u32;
                let mut line = String::new();
                for i in 0..len {
                    let _ = std::fmt::Write::write_fmt(
                        &mut line,
                        format_args!(" {:08x}", cpu.mem.read_u32(at + i * 4).unwrap_or(0)),
                    );
                }
                println!("[mark]   x{reg}{off:+} = {at:#x}:{line}");
            }
        }
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
