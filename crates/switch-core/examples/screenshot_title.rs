//! Boot a retail title, an NSP, a cartridge image or a bare Program NCA,
//! decided by the container's header, and write the Nth presented frame to a
//! PPM:
//! `screenshot_title <container> <prod.keys> [title.keys] <out.ppm> [frame]`.
//!
//! The counterpart to `screenshot` for a title rather than an NRO. The
//! difference from `boot_nsp SHOT=` is that this stops *at* the frame rather
//! than at a step budget, which matters more than it sounds: a title needs
//! **seconds** of console time before its first frame, which is billions of
//! steps, and picking a budget that lands after it is guesswork, "A Short
//! Hike" reaches frame 30 at step 3.3 billion.
//!
//! This was `screenshot_nsp` and `screenshot_nca`. They differed in which
//! container each could open, and once that was decided by header instead,
//! they differed only in which investigation's knobs each had accumulated,
//! neither set having anything to do with the container kind.
//!
//! `SWITCH_FIRMWARE=<dir>` registers the system data archives. A system applet
//! needs them far more than a game does, since its fonts, icons and settings
//! all live there.
//!
//! Beyond the knobs every runner shares (see [`common::Debug`]):
//!
//! - `STEPS=<n>` caps the run; it otherwise goes until the frame arrives.
//! - `PROFILE=<interval>` samples the pc that often, hot pcs, hot 4 KiB pages
//!   and which thread is running. Same spelling as `boot_nsp`'s. `STACKS=1`
//!   adds a return-address histogram at the same interval, so a frame loop's
//!   whole call tree comes out of one run instead of one backtrace at a time.
//! - `COVER=<lo>:<hi>` records which instructions in a range ever execute.
//!   Chasing a call chain statically stops the moment a function has no
//!   reference anywhere in the image; this answers "which of these ran" for a
//!   whole class at once.
//! - `WATCH_MEM=<addr>` reports the first step at which a 4 KiB window there
//!   stops being all zeroes. A GPU reading zeroes is either looking at the
//!   wrong memory or at memory nothing has filled yet, and this tells the two
//!   apart. `SCAN_MEM=<addr>:<size>` lists a region's non-zero spans once the
//!   run has stopped, which is how you find the buffer you meant among the
//!   ones you did not, and `DUMP_VERTS=<addr>[,...]` reads three 60-byte rows
//!   as floats: real positions are ordinary numbers, and a structure
//!   reinterpreted as float is a wall of denormals.
//! - `POKE_U32=<addr>:<value>` writes a word every sampling tick, or once at
//!   `POKE_AT=<step>`. A latched state flag is only a theory until you clear
//!   it and see what the guest does.
//! - `START_THREADS=<step>` makes every created-but-never-started thread
//!   runnable, and `WAKE_ALL=<period>` makes every blocked one runnable that
//!   often.
//! - `GATE_SNIFF=1` finds the applet framework's "reasons to skip this frame"
//!   mask without knowing the object's address: the frame loop reads it with
//!   `ldr w8, [xN, #0x3e8]`, so the first time that instruction executes, the
//!   word it names is the gate. Then it holds that word at zero. **Zeroing it
//!   is not free**: on 18.0.1's qlaunch the sniffer finds the gate at step
//!   26M, and the run then reaches its frame with 0 draws instead of 8.
//! - `FIND_MAGIC=SARC` scans guest memory for a four-byte magic. A layout
//!   archive arrives Yaz0-compressed and is decompressed by the guest; if the
//!   decompressed form is nowhere in memory, the decompression produced
//!   nothing and the UI has no panes to draw.
mod common;

use common::{Flow, Pace};
use std::collections::HashMap;
use std::env;
use switch_core::cpu::Cpu;

const USAGE: &str = "screenshot_title <container> <prod.keys> [title.keys] <out.ppm> [frame]";

/// The sampling interval `STACKS=1` implies when `PROFILE=` names none. Short
/// enough that a hot loop still resolves.
const DEFAULT_INTERVAL: u64 = 4096;

/// A hex `<lo>:<hi>` pair, both ends given, unlike [`common::env_span`]'s
/// address and length.
fn env_bounds(name: &str) -> Option<(u32, u32)> {
    let raw = env::var(name).ok()?;
    let (lo, hi) = raw.split_once(':')?;
    Some((common::hex(lo), common::hex(hi)))
}

/// What `PROFILE=` and `STACKS=` collected.
#[derive(Default)]
struct Profile {
    /// Samples per (thread index, pc).
    hot: HashMap<(usize, u32), u64>,
    /// How often each return address was on the stack, and the shallowest
    /// depth it appeared at.
    frames: HashMap<u32, (u64, usize)>,
    /// Samples per thread index.
    share: [u64; 32],
}

impl Profile {
    fn sample(&mut self, cpu: &Cpu, stacks: bool) {
        let thread = cpu.current_thread_index();
        if thread < self.share.len() {
            self.share[thread] += 1;
        }
        *self.hot.entry((thread, cpu.get_pc())).or_default() += 1;
        if stacks {
            for (depth, frame) in cpu.backtrace(14).into_iter().enumerate() {
                let slot = self.frames.entry(frame).or_insert((0, depth));
                slot.0 += 1;
                slot.1 = slot.1.min(depth);
            }
        }
    }

    fn report(&self) {
        println!("[threads] sampled share = {:?}", &self.share[..]);
        let mut stacks: Vec<_> = self.frames.iter().collect();
        stacks.sort_by_key(|&(_, &(count, _))| std::cmp::Reverse(count));
        for (at, (count, shallowest)) in stacks.into_iter().take(34) {
            println!("[stack] {at:#x} {count} (shallowest depth {shallowest})");
        }

        let mut by_page: HashMap<u32, u64> = HashMap::new();
        for (&(_, pc), &count) in &self.hot {
            *by_page.entry(pc & !0xFFF).or_default() += count;
        }
        let mut pages: Vec<_> = by_page.into_iter().collect();
        pages.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        for (page, count) in pages.into_iter().take(20) {
            println!("[hot-page] {page:#x} {count}");
        }

        let mut top: Vec<_> = self.hot.iter().collect();
        top.sort_by_key(|&(_, &count)| std::cmp::Reverse(count));
        for (&(thread, pc), count) in top.into_iter().take(10) {
            println!("[hot] thread {thread} pc={pc:#x} {count}");
        }
    }
}

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

    let mut debug = common::Debug::from_env();
    debug.arm(&mut cpu);

    let stacks = env::var("STACKS").is_ok();
    // A stack histogram is profiling by another name, so asking for one
    // without an interval still samples.
    let interval = match (common::env_u64("PROFILE", 0), stacks) {
        (0, true) => DEFAULT_INTERVAL,
        (given, _) => given,
    };
    let mut profile = Profile::default();

    let cover = env_bounds("COVER");
    let mut covered: Vec<bool> = cover
        .map(|(lo, hi)| vec![false; ((hi - lo) / 4) as usize])
        .unwrap_or_default();
    let watch_mem = common::env_hex("WATCH_MEM");
    let mut seen_nonzero = false;
    let poke = env_bounds("POKE_U32");
    let poke_at: Option<u64> = env::var("POKE_AT").ok().and_then(|v| v.parse().ok());
    let start_threads: Option<u64> = env::var("START_THREADS").ok().and_then(|v| v.parse().ok());
    let mut started_threads = false;
    let wake_every = common::env_u64("WAKE_ALL", 0);
    let gate_sniff = env::var("GATE_SNIFF").is_ok();
    let mut gate: Option<u32> = None;

    // Every hook below reads the machine between two instructions, which is
    // what `Pace::Instructions` is for, and about half the speed. With none
    // of them armed the run goes through the block translator instead, which
    // is the engine the frontend uses. `screenshot_nsp` used to be stepwise
    // unconditionally, so every run it ever timed was of the interpreter.
    let pace = if debug.stepwise()
        || interval > 0
        || cover.is_some()
        || poke.is_some()
        || start_threads.is_some()
        || wake_every > 0
        || gate_sniff
    {
        Pace::Instructions
    } else {
        Pace::Blocks
    };
    let run = common::drive(
        &mut cpu,
        pace,
        common::env_u64("STEPS", u64::MAX),
        |cpu, done| {
            if cpu.nv.gpu.frames >= want {
                return Flow::Stop;
            }
            debug.tick(cpu, done);
            if interval > 0 && done % interval == 0 {
                profile.sample(cpu, stacks);
            }
            if let Some(at) = watch_mem {
                if !seen_nonzero
                    && done % DEFAULT_INTERVAL == 0
                    && (0..0x1000u32)
                        .step_by(4)
                        .any(|k| cpu.mem.read_u32(at + k).unwrap_or(0) != 0)
                {
                    println!("[watch-mem] {at:#x} first non-zero at step {done}");
                    seen_nonzero = true;
                }
            }
            if let Some((addr, value)) = poke {
                match poke_at {
                    // One shot: does the guest keep the value, or put it back?
                    Some(at) if done == at => {
                        let _ = cpu.mem.write_u32(addr, value);
                        println!("[poke] {addr:#x} = {value:#x} at step {done}");
                    }
                    None if done % DEFAULT_INTERVAL == 0 => {
                        let _ = cpu.mem.write_u32(addr, value);
                    }
                    _ => {}
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
                        let at =
                            (cpu.read_x(((insn >> 5) & 0x1F) as u8) as u32).wrapping_add(0x3e8);
                        println!("[gate] found at {at:#x} via pc={pc:#x} step {done}");
                        gate = Some(at);
                    }
                }
            }
            if let Some(at) = gate {
                if done % DEFAULT_INTERVAL == 0 {
                    let _ = cpu.mem.write_u32(at, 0);
                }
            }
            Flow::Continue
        },
    );

    if let Some((lo, _)) = cover {
        let mut span: Option<u32> = None;
        for (i, &hit) in covered.iter().enumerate() {
            let at = lo + i as u32 * 4;
            match (hit, span) {
                (true, None) => span = Some(at),
                (false, Some(start)) => {
                    println!("[cover] {start:#x}..{at:#x}");
                    span = None;
                }
                _ => {}
            }
        }
        if let Some(start) = span {
            println!("[cover] {start:#x}..end");
        }
    }
    if watch_mem.is_some() && !seen_nonzero {
        println!("[watch-mem] never non-zero");
    }
    if let Ok(magic) = env::var("FIND_MAGIC") {
        let wanted = u32::from_le_bytes(
            magic
                .as_bytes()
                .first_chunk::<4>()
                .copied()
                .unwrap_or([0; 4]),
        );
        let mut hits = 0u32;
        let mut at = 0u32;
        while at < 0x8000_0000 && hits < 12 {
            if cpu.mem.read_u32(at) == Ok(wanted) {
                println!("[find] {magic} at {at:#x}");
                hits += 1;
            }
            at += 4;
        }
        println!("[find] {magic}: {hits} hit(s)");
    }
    if let Some((lo, hi)) = common::env_span("SCAN_MEM") {
        let mut spans = Vec::new();
        let mut span: Option<u32> = None;
        for at in (lo..hi).step_by(4) {
            let nonzero = cpu.mem.read_u32(at).unwrap_or(0) != 0;
            match (nonzero, span) {
                (true, None) => span = Some(at),
                (false, Some(start)) => {
                    spans.push((start, at - start));
                    span = None;
                }
                _ => {}
            }
        }
        if let Some(start) = span {
            spans.push((start, hi - start));
        }
        println!("  {} non-zero spans in {lo:#x}+{:#x}", spans.len(), hi - lo);
        for (at, len) in spans.iter().take(40) {
            println!("    {at:#x} .. +{len:#x}");
        }
    }
    if let Ok(list) = env::var("DUMP_VERTS") {
        for spec in list.split(',') {
            let at = common::hex(spec);
            let f: Vec<f32> = (0..45u32)
                .map(|k| f32::from_bits(cpu.mem.read_u32(at + k * 4).unwrap_or(0)))
                .collect();
            println!("  {at:#x} as f32: {:?}", &f[..15]);
            println!("  {at:#x} +60    : {:?}", &f[15..30]);
            println!("  {at:#x} +120   : {:?}", &f[30..45]);
        }
    }

    if interval > 0 {
        profile.report();
    }
    println!("[bt] {:#x} <- {:x?}", cpu.get_pc(), cpu.backtrace(12));
    print!("{}", cpu.thread_dump());
    common::report(&cpu, &run);
    debug.report();
    debug.stop_state(&cpu);

    let fb = &cpu.nv.gpu.framebuffer;
    if fb.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(&out, fb);
}
