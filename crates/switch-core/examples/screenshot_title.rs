//! Boot a title (NSP, XCI or Program NCA) and write its Nth presented frame to a PPM:
//! `screenshot_title <container> <prod.keys> [title.keys] <out.ppm> [frame]`.
//!
//! Knobs beyond [`common::Debug`]: `SWITCH_FIRMWARE`, `STEPS`, `PROFILE`, `STACKS`,
//! `COVER`, `WATCH_MEM`, `SCAN_MEM`, `DUMP_VERTS`, `DUMP_SURFACE`, `POKE_U32`,
//! `POKE_AT`, `START_THREADS`, `WAKE_ALL`, `GATE_SNIFF`, `FIND_MAGIC`.
mod common;

use common::{Flow, Pace};
use std::collections::HashMap;
use std::env;
use switch_core::cpu::Cpu;

const USAGE: &str = "screenshot_title <container> <prod.keys> [title.keys] <out.ppm> [frame]";

const DEFAULT_INTERVAL: u64 = 4096;

/// A hex `<lo>:<hi>` pair.
fn env_bounds(name: &str) -> Option<(u32, u32)> {
    let raw = env::var(name).ok()?;
    let (lo, hi) = raw.split_once(':')?;
    Some((common::hex(lo), common::hex(hi)))
}

#[derive(Default)]
struct Profile {
    /// Samples per (thread index, pc).
    hot: HashMap<(usize, u32), u64>,
    /// Return address to (count, shallowest depth).
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
    for (lo, hi) in common::env_spans("SCAN_MEM") {
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

    if let Ok(list) = env::var("DUMP_SURFACE") {
        for spec in list.split(',') {
            dump_surface(&cpu, spec, &out);
        }
    }

    let fb = &cpu.nv.gpu.framebuffer;
    if fb.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(&out, fb);
}

/// One `DUMP_SURFACE` entry: `<addr>:<w>x<h>:<format>[:<block height>]`.
fn dump_surface(cpu: &switch_core::cpu::Cpu, spec: &str, out: &str) {
    use switch_core::gpu::surface::{block_linear_offset, ColorFormat};
    let fields: Vec<&str> = spec.split(':').collect();
    let parsed = (|| {
        let addr = common::hex(fields.first()?);
        let (w, h) = fields.get(1)?.split_once('x')?;
        let (w, h): (u32, u32) = (w.parse().ok()?, h.parse().ok()?);
        let format = ColorFormat::from_raw(common::hex(fields.get(2)?)).ok()?;
        let block_height = fields.get(3).and_then(|v| v.parse().ok()).unwrap_or(16);
        Some((addr, w, h, format, block_height))
    })();
    let Some((addr, w, h, format, block_height)) = parsed else {
        println!("[surface] cannot read {spec:?}: <addr>:<w>x<h>:<format>[:<block height>]");
        return;
    };
    let bpp = format.bytes_per_pixel;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    let mut brightest = 0.0f32;
    for y in 0..h {
        for x in 0..w {
            let at = addr + block_linear_offset(x * bpp, y, w * bpp, block_height);
            let raw = (0..bpp).fold(0u128, |acc, i| {
                acc | u128::from(cpu.mem.read_u8(at + i).unwrap_or(0)) << (8 * i)
            });
            let rgba = format.decode(raw).unwrap_or([0.0; 4]);
            brightest = rgba[..3].iter().fold(brightest, |m, &c| m.max(c));
            let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u32;
            pixels.push(byte(rgba[0]) | byte(rgba[1]) << 8 | byte(rgba[2]) << 16 | 0xFF << 24);
        }
    }
    let path = format!("{out}.{addr:x}.ppm");
    let lit = common::write_ppm(
        &path,
        &switch_core::gpu::Framebuffer {
            width: w,
            height: h,
            pixels,
        },
    );
    println!("[surface] {addr:#x} {w}x{h} -> {path}: {lit} lit, brightest channel {brightest}");
}
