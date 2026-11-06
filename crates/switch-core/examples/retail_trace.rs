//! Boot a retail container and dump the last N executed instructions when the
//! guest halts.
//!
//! Usage: retail_trace <container> --keys <prod.keys> [flags]
mod common;

use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    /// Instructions kept in the ring.
    #[arg(long, default_value_t = 4000)]
    tail: usize,
    /// Instructions to run.
    #[arg(long, default_value_t = 400_000_000)]
    max_steps: u64,
    /// Start recording only once this pc is first hit.
    #[arg(long, value_parser = common::hex32, value_name = "PC")]
    ring_from: Option<u32>,
    /// Run this many steps at full speed before recording; the ring covers what follows.
    #[arg(long, value_name = "N")]
    ring_from_step: Option<u64>,
    /// Skip pcs below this.
    #[arg(long, value_parser = common::hex32, default_value = "0", value_name = "PC")]
    ring_min: u32,
    /// Stop this many steps after recording starts.
    #[arg(long, value_name = "N")]
    ring_stop_after: Option<u64>,
    /// Print a line each time one of these pcs runs: `<pc>[=name],...`.
    #[arg(long, value_delimiter = ',', value_parser = mark, value_name = "PC[=NAME]")]
    mark: Vec<(u32, String)>,
    /// Also dump memory at each mark: `<reg>,<byte offset>,<words>`.
    #[arg(long, value_parser = mark_dump, value_name = "REG,OFFSET,WORDS")]
    mark_dump: Option<(u8, i64, u32)>,
}

fn mark(entry: &str) -> Result<(u32, String), String> {
    let (pc, name) = entry.split_once('=').unwrap_or((entry, entry));
    Ok((common::hex32(pc)?, name.to_string()))
}

fn mark_dump(spec: &str) -> Result<(u8, i64, u32), String> {
    let bad = || format!("{spec:?} is not <reg>,<byte offset>,<words>");
    let mut parts = spec.split(',');
    let mut next = || parts.next().map(str::trim).ok_or_else(bad);
    Ok((
        next()?.parse().map_err(|_| bad())?,
        next()?.parse().map_err(|_| bad())?,
        next()?.parse().map_err(|_| bad())?,
    ))
}

fn main() {
    let cli = common::parse::<Cli>(|cli| &cli.shared);
    let title = cli.target.open();
    let (tail, budget) = (cli.tail, cli.max_steps);

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    title.boot(&mut cpu);
    common::apply_no_draw(&mut cpu);

    let ring_from = cli.ring_from;
    let mut recording = ring_from.is_none();
    let ring_min = cli.ring_min;
    let mut ring: std::collections::VecDeque<(u64, u32, [u64; 8])> =
        std::collections::VecDeque::with_capacity(tail + 1);

    let stop_after = cli.ring_stop_after;
    let mut recorded = 0u64;
    let marks: std::collections::HashMap<u32, String> = cli.mark.iter().cloned().collect();
    let mark_dump = cli.mark_dump;
    let mut done = 0u64;
    if let Some(from) = cli.ring_from_step {
        let run = common::run_to(&mut cpu, from.min(budget), |_| false);
        done = run.steps;
        if let Some(fault) = run.fault {
            println!("FAULT before --ring-from-step, step {done}: {fault}");
            return;
        }
    }
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
                println!("stopped {recorded} steps after --ring-from");
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
    common::print_console(&cpu, usize::MAX);
}
