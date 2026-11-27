//! Boot a program and write its Nth presented frame to a PPM:
//! `screenshot <target> --out <out.ppm> [--keys <prod.keys>] [--stop-frame <n>] [--software]`.
//!
//! The target is an `.nro`, `.nsp`, `.xci` or Program `.nca` (retail needs keys). Draws go to
//! the GPU backend, as in the browser, falling back to the rasterizer where it does; without an
//! adapter, or with `--software`, every draw is the rasterizer's. `cmp` a `--software` frame with
//! a device frame to check the backend against that reference.
#[path = "../../core/examples/common/mod.rs"]
mod common;

use clap::Parser;
use common::profile::Profile;
use common::Flow;
use switch_core::gpu::renderer::{Renderer, Software};
use switch_core::mem::PAGE_SIZE;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    target: common::Target,
    #[command(flatten)]
    shared: common::Shared,
    #[command(flatten)]
    boot: common::Boot,
    #[command(flatten)]
    report: common::Report,
    #[command(flatten)]
    play: common::Play,
    #[command(flatten)]
    shots: common::Shots,
    #[command(flatten)]
    probe: common::Probe,
    /// Where the frame is written.
    #[arg(short = 'o', long, value_name = "OUT.PPM")]
    out: String,
    /// The presented frame to write; the run stops there.
    #[arg(short = 'e', long, default_value_t = 1, value_name = "N")]
    stop_frame: u64,
    /// Copy this host directory onto the SD card; an NRO inside it is launched from its SD path,
    /// as the page's Files dialog does, so it finds the files beside it.
    #[arg(long, value_name = "DIR")]
    sd: Option<String>,
    /// Instructions to run; unlimited for a retail title, 200 million for an `.nro`.
    #[arg(long, value_name = "N")]
    max_steps: Option<u64>,
    /// Draw everything with the software rasterizer, the reference the device is checked against.
    #[arg(long, conflicts_with_all = DEVICE_FLAGS)]
    software: bool,
    #[command(flatten)]
    device: Device,
    #[command(flatten)]
    memory: Memory,
}

/// The ids of [`Device`]'s flags, which `--software` leaves nothing to apply to.
const DEVICE_FLAGS: [&str; 8] = [
    "defer_readbacks",
    "software_compute",
    "sync_every_present",
    "interleave",
    "web_limits",
    "device_msaa",
    "times",
    "only",
];

/// The device's diagnostic switches.
#[derive(clap::Args)]
struct Device {
    /// Never wait for readbacks, as in a browser.
    #[arg(long)]
    defer_readbacks: bool,
    /// Run every compute launch in software, to compare against the device's.
    #[arg(long)]
    software_compute: bool,
    /// Bring every surface back at every present, to compare against keeping them held.
    #[arg(long)]
    sync_every_present: bool,
    /// Interleave fallback draws into device frames even with late readbacks.
    #[arg(long)]
    interleave: bool,
    /// Hide native-only features, so a native run takes the browser's routes.
    #[arg(long)]
    web_limits: bool,
    /// Let the device multisample where it can.
    #[arg(long)]
    device_msaa: bool,
    /// Print phase timings.
    #[arg(long)]
    times: bool,
    /// Draw only this draw index, or the half-open range `a..b`.
    #[arg(long, value_parser = draw_range, value_name = "N|A..B")]
    only: Option<std::ops::Range<u32>>,
}

/// Probes of guest memory and execution, read while the title runs or once it stops.
#[derive(clap::Args)]
struct Memory {
    /// Sample the pc every this many instructions.
    #[arg(long, default_value_t = 0, value_name = "INTERVAL")]
    profile: u64,
    /// Sample call stacks too.
    #[arg(long)]
    stacks: bool,
    /// Report which instructions in this `<lo>:<hi>` pc range ran.
    #[arg(long, value_parser = bounds, value_name = "LO:HI")]
    cover: Option<(u32, u32)>,
    /// Report the step at which 4 KiB at this address first holds a non-zero word.
    #[arg(long, value_parser = common::hex64, value_name = "ADDR")]
    watch_mem: Option<u64>,
    /// Report the non-zero spans of these `<addr>:<size>` regions at the end.
    #[arg(long, value_delimiter = ',', value_parser = common::span, value_name = "ADDR:SIZE")]
    scan_mem: Vec<(u64, u64)>,
    /// Print three vertices' worth of f32s at each address at the end.
    #[arg(long, value_delimiter = ',', value_parser = common::hex64, value_name = "ADDR")]
    dump_verts: Vec<u64>,
    /// Write block-linear surfaces as PPMs: `<addr>:<w>x<h>:<format>[:<block height>]`.
    #[arg(long, value_delimiter = ',', value_name = "SPEC")]
    dump_surface: Vec<String>,
    /// Keep writing this 32-bit `<addr>:<value>` word, or with `@STEP` write it once at that step.
    #[arg(long, value_parser = poke, value_name = "ADDR:VALUE[@STEP]")]
    poke: Option<Poke>,
    /// Find the title's gate word and keep it zero.
    #[arg(long)]
    gate_sniff: bool,
    /// List the instructions the decoder leaves unimplemented in every program the run decodes.
    #[arg(long)]
    undecoded: bool,
    /// Print where these four bytes sit in guest memory.
    #[arg(long, value_name = "TEXT")]
    find_magic: Option<String>,
}

/// Steps between the periodic memory checks and pokes, and the `--stacks` sample interval.
const DEFAULT_INTERVAL: u64 = 4096;

/// The default `--max-steps` for an `.nro`.
const HOMEBREW_STEPS: u64 = 200_000_000;

fn draw_range(spec: &str) -> Result<std::ops::Range<u32>, String> {
    switch_gpu::draw_range(spec).ok_or_else(|| format!("{spec:?} is not N or A..B"))
}

/// A hex `<lo>:<hi>` pair.
fn bounds(text: &str) -> Result<(u32, u32), String> {
    let (lo, hi) = text
        .split_once(':')
        .ok_or_else(|| format!("{text:?} is not <hex>:<hex>"))?;
    Ok((common::hex32(lo)?, common::hex32(hi)?))
}

/// `--poke`: a guest address, the word to keep there, and the one step to write it at if any.
#[derive(Clone, Copy)]
struct Poke {
    addr: u64,
    value: u32,
    at: Option<u64>,
}

/// A hex `<addr>:<value>` pair, then an optional decimal `@<step>`.
fn poke(text: &str) -> Result<Poke, String> {
    let (word, at) = match text.split_once('@') {
        Some((word, step)) => {
            let step = step
                .parse()
                .map_err(|_| format!("{step:?} is not a step number"))?;
            (word, Some(step))
        }
        None => (text, None),
    };
    let (addr, value) = word
        .split_once(':')
        .ok_or_else(|| format!("{text:?} is not <hex addr>:<hex value>[@<step>]"))?;
    Ok(Poke {
        addr: common::hex64(addr)?,
        value: common::hex32(value)?,
        at,
    })
}

fn main() {
    let cli = common::parse::<Cli>(|cli| {
        common::Flags::default()
            .with(&cli.shared)
            .with(&cli.boot)
            .with(&cli.report)
            .with(&cli.play)
            .with(&cli.shots)
            .with(&cli.probe)
    });
    let program = cli.target.open_program();
    let (out, want) = (&cli.out, cli.stop_frame);
    let renderer = open_renderer(cli.software, &cli.device);

    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    let sd_path = common::copy_sd(&mut cpu, &program, cli.sd.as_deref(), &cli.target);
    program.boot(&mut cpu, sd_path.as_deref());

    let mut last_shot = 0;
    let mut draw_gate = common::DrawGate::install(&mut cpu, renderer);
    let mut debug = common::Debug::from_flags();
    debug.arm(&mut cpu);
    let mut press = common::Press::from_flags();

    let memory = &cli.memory;
    if memory.undecoded {
        switch_core::gpu::shader::uses::record_decoded();
    }
    let interval = match (memory.profile, memory.stacks) {
        (0, true) => DEFAULT_INTERVAL,
        (given, _) => given,
    };
    let mut profile = Profile::new(interval, u64::from(memory.stacks));
    let mut covered: Vec<bool> = memory
        .cover
        .map(|(lo, hi)| vec![false; ((hi - lo) / 4) as usize])
        .unwrap_or_default();
    let mut seen_nonzero = false;
    let mut watch_due = 0u64;
    let mut gate: Option<u64> = None;

    // Homebrew that never presents the frame stops; a retail title needs billions of steps.
    let default_steps = if program.is_homebrew() {
        HOMEBREW_STEPS
    } else {
        u64::MAX
    };
    let pace = debug.pace(
        profile.is_on() || memory.cover.is_some() || memory.poke.is_some() || memory.gate_sniff,
    );
    let cores = common::start_cores(&mut cpu);
    let run = common::drive(
        &mut cpu,
        pace,
        cli.max_steps.unwrap_or(default_steps),
        |cpu, done| {
            if cpu.nv.gpu.frames >= want {
                return Flow::Stop;
            }
            debug.tick(cpu, done);
            if let Some(gate) = draw_gate.as_mut() {
                gate.tick(cpu);
            }
            common::shoot(cpu, out, &mut last_shot);
            if let Some(press) = press.as_mut() {
                press.tick(cpu, done);
            }
            profile.tick(cpu, done);
            if let Some(at) = memory.watch_mem {
                // A block slice can end short, so `done` need not land on a multiple.
                if !seen_nonzero && done >= watch_due {
                    watch_due = done + DEFAULT_INTERVAL;
                    if read_guest(cpu, at, 0x1000).iter().any(|&b| b != 0) {
                        println!("[watch-mem] {at:#x} first non-zero at step {done}");
                        seen_nonzero = true;
                    }
                }
            }
            if let Some(Poke { addr, value, at }) = memory.poke {
                match at {
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
            if let Some((lo, hi)) = memory.cover {
                let pc = cpu.get_pc();
                if pc >= lo && pc < hi {
                    covered[((pc - lo) / 4) as usize] = true;
                }
            }
            if memory.gate_sniff && gate.is_none() {
                let pc = cpu.get_pc();
                if let Ok(insn) = cpu.mem.fetch(pc) {
                    if insn & 0xFFFF_FC1F == 0xB943_E808 {
                        let at = cpu.read_x(((insn >> 5) & 0x1F) as u8).wrapping_add(0x3e8);
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

    if let Some((lo, _)) = memory.cover {
        report_cover(lo, &covered);
    }
    if memory.watch_mem.is_some() && !seen_nonzero {
        println!("[watch-mem] never non-zero");
    }
    if let Some(magic) = &memory.find_magic {
        find_magic(&cpu, magic);
    }
    for &(lo, hi) in &memory.scan_mem {
        scan_mem(&cpu, lo, hi);
    }
    for &at in &memory.dump_verts {
        let f: Vec<f32> = (0..45u64)
            .map(|k| f32::from_bits(cpu.mem.read_u32(at + k * 4).unwrap_or(0)))
            .collect();
        println!("  {at:#x} as f32: {:?}", &f[..15]);
        println!("  {at:#x} +60    : {:?}", &f[15..30]);
        println!("  {at:#x} +120   : {:?}", &f[30..45]);
    }

    profile.report();
    println!("[bt] {:#x} <- {:x?}", cpu.get_pc(), cpu.backtrace(12));
    print!("{}", cpu.thread_dump());
    common::stop_cores(&mut cpu, cores);
    common::report(&cpu, &run);
    debug.report();
    debug.stop_state(&cpu);
    // The browser's Rendering panel: device draws and fallbacks.
    println!("rendering: {}", cpu.nv.gpu.renderer_report());

    if memory.undecoded {
        report_undecoded(switch_core::gpu::shader::uses::take_decoded());
    }
    for spec in &memory.dump_surface {
        dump_surface(&cpu, spec, out);
    }

    let fb = &cpu.nv.gpu.framebuffer;
    if fb.is_empty() {
        println!("no frame");
        return;
    }
    common::write_ppm(out, fb);
}

/// The GPU backend configured from `device`, or the rasterizer when asked for or when no adapter opens.
fn open_renderer(software: bool, device: &Device) -> Box<dyn Renderer> {
    if software {
        println!("[gpu] software");
        return Box::new(Software);
    }
    match switch_gpu::Gpu::open() {
        Ok(mut gpu) => {
            gpu.set_defer_readbacks(device.defer_readbacks);
            gpu.set_software_compute(device.software_compute);
            gpu.set_sync_every_present(device.sync_every_present);
            gpu.set_interleave(device.interleave);
            gpu.set_web_limits(device.web_limits);
            gpu.set_device_msaa(device.device_msaa);
            gpu.set_times(device.times);
            gpu.set_only(device.only.clone());
            println!("[gpu] installed");
            Box::new(gpu)
        }
        Err(why) => {
            eprintln!("no GPU backend ({why}): drawing in software");
            Box::new(Software)
        }
    }
}

/// Print the `--cover` range's runs of executed instructions.
fn report_cover(lo: u32, covered: &[bool]) {
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

/// Print up to 12 word-aligned places in the low 2 GiB holding `magic`'s first four bytes.
fn find_magic(cpu: &switch_core::cpu::Cpu, magic: &str) {
    let wanted = u32::from_le_bytes(
        magic
            .as_bytes()
            .first_chunk::<4>()
            .copied()
            .unwrap_or([0; 4]),
    );
    const BLOCK: u64 = 0x20_0000;
    let mut hits = 0u32;
    // Only backed pages hold anything; untouched blocks of the 39-bit space are skipped whole.
    'search: for block in (0..switch_core::mem::ADDRESS_SPACE_SIZE).step_by(BLOCK as usize) {
        if cpu.mem.mapped_bytes_in(block, block + BLOCK) == 0 {
            continue;
        }
        for page in (block..block + BLOCK).step_by(PAGE_SIZE) {
            if !cpu.mem.page_mapped(page) {
                continue;
            }
            let words = read_guest(cpu, page, PAGE_SIZE);
            for (at, word) in (page..).step_by(4).zip(words.as_chunks::<4>().0) {
                if *word == wanted.to_le_bytes() {
                    println!("[find] {magic} at {at:#x}");
                    hits += 1;
                    if hits == 12 {
                        break 'search;
                    }
                }
            }
        }
    }
    println!("[find] {magic}: {hits} hit(s)");
}

/// `len` bytes of guest memory from `addr`, a page at a time; an unbacked page reads as zeros.
fn read_guest(cpu: &switch_core::cpu::Cpu, addr: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    let mut done = 0;
    while done < len {
        let at = addr + done as u64;
        let n = (PAGE_SIZE - (at as usize % PAGE_SIZE)).min(len - done);
        // A failed read copies nothing, leaving that page's zeros.
        let _ = cpu.mem.read_into(at, &mut bytes[done..done + n]);
        done += n;
    }
    bytes
}

/// Print the non-zero spans of `lo..hi`.
fn scan_mem(cpu: &switch_core::cpu::Cpu, lo: u64, hi: u64) {
    let mut spans = Vec::new();
    let mut span: Option<u64> = None;
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

/// `--undecoded`: unimplemented instructions by their top 16 opcode bits, the most common first.
fn report_undecoded(programs: std::collections::BTreeMap<u64, switch_core::gpu::shader::Program>) {
    use switch_core::gpu::shader::Op;
    // Per opcode: instructions, programs, and one word to decode by hand.
    let mut seen: std::collections::BTreeMap<u16, (u64, u64, u64)> = Default::default();
    for program in programs.values() {
        let mut here = std::collections::BTreeSet::new();
        for insn in &program.insns {
            if let Op::Unimplemented { raw } = insn.op {
                let opcode = (raw >> 48) as u16;
                let row = seen.entry(opcode).or_insert((0, 0, raw));
                row.0 += 1;
                if here.insert(opcode) {
                    row.1 += 1;
                }
            }
        }
    }
    println!(
        "[undecoded] {} program(s) decoded, {} opcode(s) left unimplemented",
        programs.len(),
        seen.len()
    );
    let mut rows: Vec<_> = seen.into_iter().collect();
    rows.sort_by_key(|&(opcode, (insns, programs, _))| {
        (
            std::cmp::Reverse(programs),
            std::cmp::Reverse(insns),
            opcode,
        )
    });
    for (opcode, (insns, programs, raw)) in rows {
        println!("[undecoded] {opcode:#06x}  {programs:>4} program(s)  {insns:>5} instruction(s)  e.g. {raw:016x}");
    }
}

/// One `--dump-surface` entry: `<addr>:<w>x<h>:<format>[:<block height>]`.
fn dump_surface(cpu: &switch_core::cpu::Cpu, spec: &str, out: &str) {
    use switch_core::gpu::surface::{ColorFormat, Layout, Surface};
    let fields: Vec<&str> = spec.split(':').collect();
    let parsed = (|| {
        let addr = common::parse_hex(fields.first()?)?;
        let (w, h) = fields.get(1)?.split_once('x')?;
        let (w, h): (u32, u32) = (w.parse().ok()?, h.parse().ok()?);
        let format = ColorFormat::from_raw(common::parse_hex32(fields.get(2)?)?).ok()?;
        let block_height = fields.get(3).and_then(|v| v.parse().ok()).unwrap_or(16);
        Some((addr, w, h, format, block_height))
    })();
    let Some((addr, w, h, format, block_height)) = parsed else {
        println!("[surface] cannot read {spec:?}: <addr>:<w>x<h>:<format>[:<block height>]");
        return;
    };
    let surface = Surface {
        addr,
        width: w,
        height: h,
        format,
        layout: Layout::BlockLinear {
            block_height_gobs: block_height,
            block_depth_gobs: 1,
        },
    };
    let bytes = read_guest(cpu, addr, surface.size() as usize);
    let bpp = format.bytes_per_pixel as usize;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    let mut brightest = 0.0f32;
    for y in 0..h {
        for x in 0..w {
            let at = surface.offset(x, y) as usize;
            let texel = bytes.get(at..at + bpp).unwrap_or_default();
            let raw = texel
                .iter()
                .rev()
                .fold(0u128, |acc, &b| acc << 8 | u128::from(b));
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
