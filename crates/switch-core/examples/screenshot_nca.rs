//! Boot a bare Program NCA — a system applet such as the Home Menu, which
//! ships inside firmware rather than in an NSP — and write the Nth presented
//! frame to a PPM:
//! `screenshot_nca <path.nca> <prod.keys> [title.keys] <out.ppm> [frame]`.
//!
//! The counterpart to `screenshot_nsp`. An applet is the case that matters for
//! the system UI and the one an NSP-only runner cannot reach, because there is
//! no PFS0 around it to find a Program NCA in.
//!
//! `SWITCH_FIRMWARE=<dir>` registers the system data archives; an applet needs
//! them far more than a title does, since its fonts, icons and settings all
//! live there.
mod common;

use common::{Flow, Pace};
use std::collections::HashMap;
use std::env;
use switch_core::cpu::Cpu;

const USAGE: &str = "screenshot_nca <path.nca> <prod.keys> [title.keys] <out.ppm> [frame]";

fn main() {
    let args = common::container_args(USAGE);
    let title = args.open();
    let out = args.need(0).to_string();
    let want = args.rest_num(1).unwrap_or(1);

    let mut cpu = Cpu::new();
    cpu.bootstrap();
    title.mount_romfs(&mut cpu);
    common::load_fallback_font(&mut cpu);
    common::register_firmware(&mut cpu, &title.keys);
    title.boot(&mut cpu);

    // `TRAP_WRITE`, `TRAP_READ`, `WATCH_PC` and `DUMP`, which every runner
    // here shares.
    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);
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
    let mut covered: Vec<bool> = cover
        .map(|(a, b)| vec![false; ((b - a) / 4) as usize])
        .unwrap_or_default();
    // `START_THREADS=<step>` makes every created-but-never-started thread
    // runnable once, at that step.
    let start_threads: Option<u64> = env::var("START_THREADS").ok().and_then(|v| v.parse().ok());
    let mut started_threads = false;
    let poke_at: Option<u64> = env::var("POKE_AT").ok().and_then(|v| v.parse().ok());
    // `GATE_SNIFF=1` finds the applet framework's "reasons to skip this frame"
    // mask without knowing the object's address: the frame loop reads it with
    // `ldr w8, [xN, #0x3e8]`, so the first time that instruction executes, the
    // word it names is the gate. Every system applet is built on the same
    // framework, so this locates it in any of them.
    let gate_sniff = env::var("GATE_SNIFF").is_ok();
    let mut gate: Option<u32> = None;
    // `WAKE_ALL=<period>` makes every blocked thread runnable that often.
    let wake_every: u64 = env::var("WAKE_ALL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // `POKE_U32=<addr>:<value>` writes a word into guest memory on every
    // sampling tick once the run is under way. A latched state flag is only a
    // theory until you clear it and see what the guest does.
    let poke: Option<(u32, u32)> = env::var("POKE_U32").ok().and_then(|v| {
        let (a, b) = v.split_once(':')?;
        let a = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
        let b = u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?;
        Some((a, b))
    });
    let mut share = [0u64; 32];
    let mut hot: HashMap<(usize, u32), u64> = HashMap::new();
    // `STACKS=1` counts how often each return address appears on the stack, so
    // a frame loop's whole call tree comes out of one run instead of one
    // backtrace at a time.
    let stacks = env::var("STACKS").is_ok();
    let mut frames: HashMap<u32, (u64, usize)> = HashMap::new();
    let budget = common::env_u64("STEPS", 40_000_000_000);
    // Every hook below reads the machine between two instructions, which is
    // what `Pace::Instructions` is for — and about half the speed. With none
    // of them armed the run goes through the block translator instead, which
    // is the engine the frontend uses.
    let pace = if debug.stepwise()
        || poke.is_some()
        || start_threads.is_some()
        || wake_every > 0
        || cover.is_some()
        || gate_sniff
        || stacks
    {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    let run = common::drive(&mut cpu, pace, budget, |cpu, done| {
        if cpu.nv.gpu.frames >= want {
            return Flow::Stop;
        }
        let t = cpu.current_thread_index();
        if t < share.len() {
            share[t] += 1;
        }
        debug.tick(cpu, done);
        if done % 4096 == 0 {
            let t = cpu.current_thread_index();
            if t < share.len() {
                share[t] += 1;
            }
            *hot.entry((t, cpu.get_pc())).or_insert(0u64) += 1;
            if stacks {
                for (depth, frame) in cpu.backtrace(14).into_iter().enumerate() {
                    let slot = frames.entry(frame).or_insert((0u64, depth));
                    slot.0 += 1;
                    slot.1 = slot.1.min(depth);
                }
            }
        }
        if let Some((addr, value)) = poke {
            match poke_at {
                // One shot: does the guest keep the value, or put it back?
                Some(at) => {
                    if done == at {
                        let _ = cpu.mem.write_u32(addr, value);
                        println!("[poke] {addr:#x} = {value:#x} at step {done}");
                    }
                }
                None => {
                    if done % 4096 == 0 {
                        let _ = cpu.mem.write_u32(addr, value);
                    }
                }
            }
        }
        if let Some(at) = start_threads {
            if !started_threads && done >= at {
                started_threads = true;
                println!("[threads] force-started {}", cpu.start_created_threads());
            }
        }
        if wake_every > 0 && done % wake_every == 0 {
            cpu.wake_all_blocked();
        }
        if let Some((lo, hi)) = cover {
            let pc = cpu.get_pc();
            if pc >= lo && pc < hi {
                covered[((pc - lo) / 4) as usize] = true;
            }
        }
        if gate_sniff && gate.is_none() {
            let pc = cpu.get_pc();
            if let Ok(insn) = cpu.mem.read_u32(pc) {
                if insn & 0xFFFF_FC1F == 0xB943_E808 {
                    let base = cpu.reg(((insn >> 5) & 0x1F) as usize) as u32;
                    let at = base.wrapping_add(0x3e8);
                    println!("[gate] found at {at:#x} via pc={pc:#x} step {done}");
                    gate = Some(at);
                }
            }
        }
        if let Some(at) = gate {
            if done % 4096 == 0 {
                let _ = cpu.mem.write_u32(at, 0);
            }
        }
        Flow::Continue
    });
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
    // `FIND_MAGIC=SARC` scans guest memory for a four-byte magic. A layout
    // archive arrives Yaz0-compressed and is decompressed by the guest; if the
    // decompressed form is nowhere in memory, the decompression produced
    // nothing and the UI has no panes to draw.
    if let Ok(magic) = env::var("FIND_MAGIC") {
        let want = u32::from_le_bytes(
            magic
                .as_bytes()
                .first_chunk::<4>()
                .copied()
                .unwrap_or([0; 4]),
        );
        let mut hits = 0u32;
        let mut at = 0u32;
        while at < 0x8000_0000 {
            if cpu.mem.read_u32(at) == Ok(want) {
                println!("[find] {magic} at {at:#x}");
                hits += 1;
                if hits >= 12 {
                    break;
                }
            }
            at += 4;
        }
        println!("[find] {magic}: {hits} hit(s)");
    }
    debug.report();
    println!(
        "[mem] {} MiB mapped",
        cpu.mem.mapped_bytes() / (1024 * 1024)
    );
    println!("[threads] sampled share = {:?}", &share[..]);
    println!("[bt] {:#x} <- {:x?}", cpu.get_pc(), cpu.backtrace(12));
    let hot_snapshot = hot.clone();
    let mut top: Vec<_> = hot.into_iter().collect();
    top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    if stacks {
        let mut fs: Vec<_> = frames.into_iter().collect();
        fs.sort_by_key(|&(_, (n, _))| std::cmp::Reverse(n));
        for (addr, (n, min_depth)) in fs.into_iter().take(34) {
            println!("[stack] {addr:#x} {n} (shallowest depth {min_depth})");
        }
    }
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
    common::report(&cpu, &run);
    debug.stop_state(&cpu);
    let fb = &cpu.nv.gpu.framebuffer;
    if fb.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(&out, fb);
}
