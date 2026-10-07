//! Boot a retail container and dump the last N executed instructions when the
//! guest halts.
//!
//! Usage: retail_trace <container> <prod.keys> [title.keys] [tail_len] [max_steps]
//!   RING_FROM=<hex pc>  start recording only once this pc is first hit.
//!   RING_MIN=<hex pc>  skip pcs below this.
//!   RING_STOP_AFTER=<n>  stop n steps after recording starts.
//!   MARK=<pc>[=name][,...]  print a line each time one of these pcs runs.
//!   MARK_DUMP=<reg>,<byte offset>,<words>  also dump memory at each mark.
mod common;

use std::env;
use switch_core::cpu::Cpu;

const USAGE: &str = "retail_trace <container> <prod.keys> [title.keys] [tail_len] [max_steps]";

fn main() {
    let args = common::container_args(USAGE);
    let title = args.open();
    let tail: usize = args.rest_num(0).unwrap_or(4000) as usize;
    let budget = args.rest_num(1).unwrap_or(400_000_000);

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    title.boot(&mut cpu);

    let ring_from = env::var("RING_FROM")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
    let mut recording = ring_from.is_none();
    let ring_min = env::var("RING_MIN")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    let mut ring: std::collections::VecDeque<(u64, u32, [u64; 8])> =
        std::collections::VecDeque::with_capacity(tail + 1);

    let stop_after = env::var("RING_STOP_AFTER")
        .ok()
        .and_then(|s| s.parse::<u64>().ok());
    let mut recorded = 0u64;
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
    let mark_dump: Option<(u8, i64, u32)> = env::var("MARK_DUMP").ok().and_then(|v| {
        let mut parts = v.split(',');
        Some((
            parts.next()?.trim().parse().ok()?,
            parts.next()?.trim().parse().ok()?,
            parts.next()?.trim().parse().ok()?,
        ))
    });
    let mut done = 0u64;
    while !cpu.halted && done < budget {
        let pc = cpu.get_pc();
        if let Some(name) = marks.get(&pc) {
            println!(
                "[mark] {done} {name} x0={:#x} x1={:#x} x2={:#x} x3={:#x} lr={:#x}",
                cpu.read_x(0),
                cpu.read_x(1),
                cpu.read_x(2),
                cpu.read_x(3),
                cpu.read_x(30)
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
            // Dump any argument that points at a printable C string.
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
                [
                    cpu.read_x(0),
                    cpu.read_x(1),
                    cpu.read_x(2),
                    cpu.read_x(3),
                    cpu.read_x(8),
                    cpu.read_x(19),
                    cpu.read_x(30),
                    cpu.sp(),
                ],
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
