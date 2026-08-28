//! Scaffolding every example shares: argument handling, key loading, booting,
//! driving the machine, and writing a frame out.
//!
//! Each example used to carry its own copy of all of this, and the copies
//! drifted. That is not only untidy — it is how `screenshot` and
//! `screenshot_nca` both ended up driving the CPU one instruction at a time
//! through [`switch_core::cpu::Cpu::step`], which is the *interpreter*: only
//! `Cpu::run` reaches the block translator, and it is 1.8x faster on real
//! code. Every measurement either tool produced was of the wrong engine, and
//! nothing said so, because there was no one place where "how an example runs
//! the machine" was written down. [`drive`] is that place now.
//!
//! Not every example needs every part of this. `#![allow(dead_code)]` is the
//! usual way to share a module between Cargo examples: each one compiles the
//! whole module and uses the piece it needs.
#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;
use switch_core::cpu::Cpu;
use switch_core::gpu::Framebuffer;
use switch_core::keys::KeySet;
use switch_core::source::ByteSource;

/// The font `pl:u` serves as every shared font type when no firmware fonts
/// are registered. A guest with no font at all draws no text.
pub const FALLBACK_FONT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/font.ttf");

/// Print a usage line and exit. Examples are dev tools: a wrong invocation
/// should say what the right one is, not panic with a backtrace.
pub fn usage(line: &str) -> ! {
    eprintln!("usage: {line}");
    std::process::exit(1)
}

/// Positional argument `n`, counting the first after the program name as 1,
/// or the usage line if it is not there.
pub fn arg(n: usize, line: &str) -> String {
    match env::args().nth(n) {
        Some(arg) => arg,
        None => usage(line),
    }
}

/// Positional argument `n`, if it is there.
pub fn opt_arg(n: usize) -> Option<String> {
    env::args().nth(n)
}

/// Positional argument `n` parsed as a `u64`, if it is there and parses.
pub fn opt_num(n: usize) -> Option<u64> {
    env::args().nth(n)?.parse().ok()
}

/// Parse a hexadecimal address (`0x` optional), or exit saying what failed.
pub fn hex(text: &str) -> u32 {
    match u32::from_str_radix(text.trim().trim_start_matches("0x"), 16) {
        Ok(value) => value,
        Err(_) => {
            eprintln!("{text:?} is not a hexadecimal address");
            std::process::exit(1)
        }
    }
}

/// Read a file, or exit saying which one could not be read.
pub fn read(path: impl AsRef<Path>) -> Vec<u8> {
    let path = path.as_ref();
    match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("cannot read {}: {e}", path.display());
            std::process::exit(1)
        }
    }
}

/// Load `prod.keys`, and `title.keys` if one was given.
pub fn keys(prod: impl AsRef<Path>, title: Option<impl AsRef<Path>>) -> KeySet {
    let prod = prod.as_ref();
    let text = match fs::read_to_string(prod) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("cannot read {}: {e}", prod.display());
            std::process::exit(1)
        }
    };
    let mut set = switch_core::keys::keyset_from_prod(&switch_core::keys::parse_keys_file(&text));
    if let Some(title) = title {
        let title = title.as_ref();
        match fs::read_to_string(title) {
            Ok(text) => {
                set.title_keys = switch_core::keys::keyset_from_title(
                    &switch_core::keys::parse_keys_file(&text),
                );
            }
            Err(e) => eprintln!(
                "cannot read {}: {e} (continuing without title keys)",
                title.display()
            ),
        }
    }
    set
}

/// The `<container> <prod.keys> [title.keys]` triple every tool that reads a
/// retail container takes, resolved once.
///
/// The optional third argument had been spelled four ways — "not all digits",
/// "does not start with a digit", "ends with `.keys`", and taken blind — and
/// five more tools made it *mandatory* by passing `Some(arg(3, USAGE))`, so a
/// container whose keys are all in `prod.keys` could not be opened without
/// naming a `title.keys` that does not exist. One rule now: argument 3 is the
/// title keys when it is named like a keys file. Everything after the triple
/// is the tool's own, and is reached through [`Args::rest`].
pub struct Args {
    pub container: String,
    pub prod: String,
    pub title: Option<String>,
    rest: Vec<String>,
    line: String,
}

/// Resolve the container triple, or print `line` and exit.
pub fn container_args(line: &str) -> Args {
    let mut positional = env::args().skip(1);
    let (Some(container), Some(prod)) = (positional.next(), positional.next()) else {
        usage(line)
    };
    let mut rest: Vec<String> = positional.collect();
    let title = rest
        .first()
        .is_some_and(|arg| arg.to_ascii_lowercase().ends_with(".keys"))
        .then(|| rest.remove(0));
    Args {
        container,
        prod,
        title,
        rest,
        line: line.to_string(),
    }
}

impl Args {
    /// Open the container, whichever kind it turns out to be.
    pub fn open(&self) -> Title {
        Title::open(&self.container, &self.prod, self.title.as_ref())
    }

    /// The keys alone, for a tool that reads the container itself rather than
    /// booting what is in it.
    pub fn keys(&self) -> KeySet {
        keys(&self.prod, self.title.as_ref())
    }

    /// Argument `n` after the triple, counting from 0.
    pub fn rest(&self, n: usize) -> Option<&str> {
        self.rest.get(n).map(String::as_str)
    }

    /// Argument `n` after the triple, or the usage line if it is not there.
    pub fn need(&self, n: usize) -> &str {
        match self.rest.get(n) {
            Some(arg) => arg,
            None => usage(&self.line),
        }
    }

    /// Argument `n` after the triple, if it is there and parses as a `u64`.
    pub fn rest_num(&self, n: usize) -> Option<u64> {
        self.rest.get(n)?.parse().ok()
    }
}

/// A `u64` from the environment, or `default`.
pub fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A hexadecimal `u32` from the environment (`0x` optional).
pub fn env_hex(name: &str) -> Option<u32> {
    let raw = env::var(name).ok()?;
    u32::from_str_radix(raw.trim().trim_start_matches("0x"), 16).ok()
}

/// A `lo:len` pair of hexadecimal addresses, as `(lo, lo + len)`.
pub fn env_span(name: &str) -> Option<(u32, u32)> {
    let raw = env::var(name).ok()?;
    let (lo, len) = raw.split_once(':')?;
    let lo = u32::from_str_radix(lo.trim().trim_start_matches("0x"), 16).ok()?;
    let len = u32::from_str_radix(len.trim().trim_start_matches("0x"), 16).ok()?;
    Some((lo, lo.saturating_add(len)))
}

/// A comma-separated list of hexadecimal `u32`s from the environment.
pub fn env_hex_list(name: &str) -> Vec<u32> {
    env::var(name)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Give the guest the fallback shared font, if it is where it should be.
///
/// Quiet when the firmware fonts will be registered anyway
/// ([`register_firmware`] overrides this), and loud otherwise, because a
/// missing font is invisible until you notice a UI with no text in it.
pub fn load_fallback_font(cpu: &mut Cpu) {
    match fs::read(FALLBACK_FONT) {
        Ok(bytes) => cpu.set_shared_font(bytes),
        Err(e) => eprintln!("no font at {FALLBACK_FONT} ({e}): text will not render"),
    }
}

/// Register every system data archive in `SWITCH_FIRMWARE` — the shared
/// fonts among them, which is what makes an applet render real text.
///
/// Does nothing when the variable is unset, so an example that calls this
/// still works without a firmware dump.
pub fn register_firmware(cpu: &mut Cpu, keys: &KeySet) -> usize {
    let Ok(dir) = env::var("SWITCH_FIRMWARE") else {
        return 0;
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        eprintln!("SWITCH_FIRMWARE={dir} cannot be read");
        return 0;
    };
    let mut registered = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("nca") {
            continue;
        }
        let Ok(src) = switch_core::source::FileSource::open(&path) else {
            continue;
        };
        let Ok(archive) = switch_core::nca::Nca::parse_source(&src, Some(keys)) else {
            continue;
        };
        use switch_core::nca::ContentType;
        if !matches!(
            archive.content_type,
            ContentType::Data | ContentType::PublicData
        ) {
            continue;
        }
        let Some(section) = archive.romfs_section_index() else {
            continue;
        };
        if let Ok(romfs) = archive.romfs_source(src, keys, section) {
            cpu.add_data_archive(archive.title_id, Box::new(romfs));
            registered += 1;
        }
    }
    registered
}

/// How [`drive`] advances the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// In slices, through `Cpu::run` — which is the only thing that reaches
    /// the block translator, and so the only honest way to measure or to wait
    /// out a long boot. Use this unless the answer requires looking at the
    /// machine *between* two instructions.
    Blocks,
    /// One instruction at a time, through `Cpu::step`. Necessary for
    /// watchpoints, coverage, PC watches and anything else that samples state
    /// per instruction — and about half the speed.
    Instructions,
}

/// How many instructions [`Pace::Blocks`] runs between two `tick` calls.
/// Short enough that a sampling `tick` keeps useful resolution.
const SLICE: u64 = 4096;

/// Whether [`drive`] should keep going.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// What a run ended up doing.
#[derive(Debug, Default)]
pub struct Run {
    pub steps: u64,
    /// The fault that stopped it, if one did.
    pub fault: Option<String>,
    /// Whether the machine reached a halt trap.
    pub halted: bool,
}

/// What each presented frame cost in wall clock, printed when `FRAME_TIMES=1`
/// is set.
///
/// Timing two whole processes and subtracting them does not measure a frame
/// here. A retail title spends most of a run booting — Just Dance 2019 presents
/// its first frame at step 900,710,775 — and that boot swings by seconds
/// between runs when anything else on the machine is busy, which is larger than
/// the frames being measured. Min-of-N does not rescue it either: the minimum
/// of the boot and the minimum of the boot-plus-frames come from different runs,
/// so their difference is not any run's frame cost and can even go negative.
///
/// Sampling the frame counter *inside* one run has no such term.
struct FrameTimes {
    on: bool,
    last_count: u64,
    last_at: Instant,
    deltas: Vec<f64>,
}

impl FrameTimes {
    fn new(cpu: &Cpu) -> FrameTimes {
        FrameTimes {
            on: env::var("FRAME_TIMES").is_ok(),
            last_count: cpu.nv.gpu.frames,
            last_at: Instant::now(),
            deltas: Vec::new(),
        }
    }

    fn sample(&mut self, cpu: &Cpu) {
        if !self.on || cpu.nv.gpu.frames == self.last_count {
            return;
        }
        let now = Instant::now();
        // A slice can carry more than one present; share its time out evenly
        // rather than charge the whole slice to the last of them.
        let presented = cpu.nv.gpu.frames - self.last_count;
        let each = now.duration_since(self.last_at).as_secs_f64() / presented as f64;
        self.deltas
            .extend(std::iter::repeat_n(each, presented as usize));
        self.last_count = cpu.nv.gpu.frames;
        self.last_at = now;
    }

    /// Report the frames after the first. The first one's "cost" is the whole
    /// boot, which is not a frame cost and would dominate any average.
    fn report(&self) {
        if !self.on || self.deltas.len() < 2 {
            return;
        }
        let steady = &self.deltas[1..];
        let mut sorted = steady.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let ms = |v: f64| v * 1000.0;
        println!(
            "[frames] {} after the first: mean {:.1} ms  min {:.1} ms  median {:.1} ms",
            steady.len(),
            ms(steady.iter().sum::<f64>() / steady.len() as f64),
            ms(sorted[0]),
            ms(sorted[sorted.len() / 2]),
        );
    }
}

/// Run the machine until `tick` says to stop, it halts, it faults, or `budget`
/// instructions have retired.
///
/// `tick` is called before each instruction under [`Pace::Instructions`] and
/// before each slice under [`Pace::Blocks`], with the instructions retired so
/// far. Returning [`Flow::Stop`] ends the run.
///
/// `FRAME_TIMES=1` reports what each presented frame cost — see [`FrameTimes`]
/// for why that has to be measured from in here rather than around the process.
pub fn drive(
    cpu: &mut Cpu,
    pace: Pace,
    budget: u64,
    mut tick: impl FnMut(&mut Cpu, u64) -> Flow,
) -> Run {
    let mut run = Run::default();
    let mut frames = FrameTimes::new(cpu);
    while run.steps < budget && !cpu.halted {
        frames.sample(cpu);
        if tick(cpu, run.steps) == Flow::Stop {
            break;
        }
        match pace {
            Pace::Instructions => {
                if let Err(e) = cpu.step() {
                    run.fault = Some(format!("{e:?}"));
                    break;
                }
                run.steps += 1;
            }
            Pace::Blocks => match cpu.run(SLICE.min(budget - run.steps)) {
                // No progress and not halted: nothing more will happen.
                Ok(report) if report.steps == 0 => break,
                Ok(report) => run.steps += report.steps,
                Err(e) => {
                    run.fault = Some(format!("{e:?}"));
                    break;
                }
            },
        }
    }
    frames.sample(cpu);
    frames.report();
    run.halted = cpu.halted;
    run
}

/// Run with no per-step work at all, at full speed.
pub fn run_to(cpu: &mut Cpu, budget: u64, mut until: impl FnMut(&Cpu) -> bool) -> Run {
    drive(cpu, Pace::Blocks, budget, |cpu, _| {
        if until(cpu) {
            Flow::Stop
        } else {
            Flow::Continue
        }
    })
}

/// Write a framebuffer out as a binary PPM, and report how many of its pixels
/// are not black.
///
/// That count is the cheapest regression test this project has: a frame that
/// renders correctly lights a stable number of pixels, and a change that
/// breaks rendering moves it immediately.
pub fn write_ppm(path: impl AsRef<Path>, fb: &Framebuffer) -> usize {
    let path = path.as_ref();
    let mut ppm = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    let mut lit = 0usize;
    for px in &fb.pixels {
        let (r, g, b) = (*px as u8, (*px >> 8) as u8, (*px >> 16) as u8);
        if r != 0 || g != 0 || b != 0 {
            lit += 1;
        }
        ppm.extend_from_slice(&[r, g, b]);
    }
    if let Err(e) = fs::write(path, ppm) {
        eprintln!("cannot write {}: {e}", path.display());
        std::process::exit(1);
    }
    println!(
        "wrote {}: {}x{}, {lit}/{} non-black",
        path.display(),
        fb.width,
        fb.height,
        fb.pixels.len()
    );
    lit
}

/// Report a finished run the way every example that boots one does.
pub fn report(cpu: &Cpu, run: &Run) {
    if let Some(fault) = &run.fault {
        println!(
            "[fault] at step {} pc={:#x}: {fault}",
            run.steps,
            cpu.get_pc()
        );
    }
    println!(
        "steps={} frames={} stats={:?}",
        run.steps, cpu.nv.gpu.frames, cpu.nv.gpu.stats
    );
    // How near the run came to the RAM cap. Reaching it fails an allocation
    // the title may not check, which is a symptom that never names memory —
    // so the number belongs in every run's report rather than in whichever
    // tool someone thought to add it to.
    let used = cpu.mem.mapped_bytes();
    let cap = cpu.mem.max_mapped_bytes();
    println!(
        "memory: {:.1} MiB of {} MiB backed ({:.0}%)",
        used as f64 / (1024.0 * 1024.0),
        cap / (1024 * 1024),
        used as f64 * 100.0 / cap as f64
    );
}

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

/// One region to hex-dump once the run has stopped.
struct DumpSpec {
    label: String,
    base: DumpBase,
    offset: i64,
    len: u32,
}

fn parse_dump_specs(spec: &str) -> Vec<DumpSpec> {
    /// Enough to see a small object and its first few pointers. A region
    /// worth more than this is one the caller knows the size of.
    const DEFAULT_LEN: u32 = 0x40;
    let parse = |text: &str| u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
    spec.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let (addr, len) = match entry.split_once(':') {
                Some((addr, len)) => (addr, parse(len)? as u32),
                None => (entry, DEFAULT_LEN),
            };
            // The sign has to be found from the right: `x23+0x10` splits on
            // the `+`, but a bare `0x...` address must not be split on a `-`
            // that is part of nothing at all.
            let (base, offset) = match addr.rfind(['+', '-']).filter(|&i| i > 0) {
                Some(i) => {
                    let value = parse(&addr[i + 1..])? as i64;
                    let signed = if addr.as_bytes()[i] == b'-' {
                        -value
                    } else {
                        value
                    };
                    (&addr[..i], signed)
                }
                None => (addr, 0),
            };
            let base = match base.trim() {
                "sp" => DumpBase::StackPointer,
                "pc" => DumpBase::ProgramCounter,
                name if name.starts_with('x') => DumpBase::Register(name[1..].parse().ok()?),
                absolute => DumpBase::Absolute(parse(absolute)? as u32),
            };
            Some(DumpSpec {
                label: entry.to_string(),
                base,
                offset,
                len,
            })
        })
        .collect()
}

/// How many times a watchpoint or a pc watch reports before going quiet. What
/// is wanted is which code reached a region *first*, and a region being
/// touched at all is usually a loop.
const MAX_HITS: u32 = 24;

/// `x0`..`x7`, which are a call's arguments — what a trapped write or a
/// watched pc was reached *with*, as opposed to where it was reached from.
fn arguments(cpu: &Cpu) -> String {
    (0..8)
        .map(|r| format!(" x{r}={:#x}", cpu.read_x(r)))
        .collect()
}

fn backtrace(cpu: &Cpu, depth: usize) -> String {
    cpu.backtrace(depth)
        .iter()
        .map(|pc| format!("{pc:#010x}"))
        .collect::<Vec<_>>()
        .join(" <- ")
}

/// The debugging knobs a runner needs, read from the environment once.
///
/// `boot_nsp`, `screenshot_nsp` and `screenshot_nca` each grew their own parse
/// of the same `<addr>:<hex size>` spelling, and then drifted apart in name:
/// the pc watch was `WATCH` in one and `WATCH_PC` in another, memory dumping
/// was `DUMP` in one and `DUMP_MEM` in another, and no tool had all of them. A
/// knob worth having in one runner is worth having in every runner.
///
/// - `TRAP_WRITE=<addr>:<hex size>` — the pc and call stack of the first
///   writes into a region, which is how a buffer nobody admits to owning gets
///   an owner.
/// - `TRAP_READ=<addr>:<hex size>` — every distinct pc that reads a region,
///   counted.
/// - `TRAP_ZERO=1` — keep writes of zero as well. Off by default, since a
///   field being cleared is usually the noise; on when clearing is the hunt.
/// - `WATCH_PC=<addr>[,...]` — the argument registers and the call stack the
///   first few times execution reaches an address. Who calls a thin IPC stub
///   is not a static question here: they are reached through vtables, so
///   nothing in the image points at them.
/// - `DUMP=<base>[+<hex>][:<hex length>][,...]` — hex-dump guest memory
///   wherever the run stopped, where `<base>` is `x0`..`x30`, `sp`, `pc` or an
///   address: `DUMP=x23+0x1830:0x40,0x10c2e870`.
pub struct Debug {
    write_trap: Option<(u32, u32)>,
    read_trap: Option<(u32, u32)>,
    trap_zero: bool,
    watch_pc: Vec<u32>,
    dumps: Vec<DumpSpec>,
    traps: u32,
    watch_hits: u32,
    /// Pcs that read the `TRAP_READ` region, and how often each did.
    readers: std::collections::BTreeMap<u32, u64>,
    /// The read watchpoint reports on the step *after* the one that tripped
    /// it, so the pc that did the reading has to be carried across a tick.
    reader_pc: u32,
}

impl Debug {
    pub fn from_env() -> Debug {
        Debug {
            write_trap: env_span("TRAP_WRITE"),
            read_trap: env_span("TRAP_READ"),
            trap_zero: env::var("TRAP_ZERO").is_ok(),
            watch_pc: env_hex_list("WATCH_PC"),
            dumps: env::var("DUMP")
                .map(|spec| parse_dump_specs(&spec))
                .unwrap_or_default(),
            traps: 0,
            watch_hits: 0,
            readers: std::collections::BTreeMap::new(),
            reader_pc: 0,
        }
    }

    /// Whether anything armed here reads the machine between two
    /// instructions, and so needs [`Pace::Instructions`]. With none of them
    /// set the run goes through the block translator instead, which is the
    /// engine the frontend uses.
    pub fn stepwise(&self) -> bool {
        self.write_trap.is_some() || self.read_trap.is_some() || !self.watch_pc.is_empty()
    }

    /// Install the watchpoints. After the title has booted, since booting
    /// lays out the address space they are set in.
    pub fn arm(&self, cpu: &mut Cpu) {
        if let Some((lo, hi)) = self.write_trap {
            cpu.mem.watch_writes(lo, hi - lo);
        }
        if let Some((lo, hi)) = self.read_trap {
            cpu.mem.watch_reads(lo, hi - lo);
        }
    }

    /// Report whatever tripped since the last instruction. Call from a
    /// [`drive`] tick running at [`Pace::Instructions`].
    pub fn tick(&mut self, cpu: &mut Cpu, done: u64) {
        if self.read_trap.is_some() && cpu.mem.take_read_hit().is_some() {
            *self.readers.entry(self.reader_pc).or_default() += 1;
        }
        self.reader_pc = cpu.get_pc();
        if let Some(at) = cpu.mem.take_watch_hit() {
            let value = cpu.mem.read_u32(at & !3).unwrap_or(0);
            if self.traps < MAX_HITS && (value != 0 || self.trap_zero) {
                println!(
                    "[trap] wrote {at:#010x} = {value:#010x} at step {done} pc={:#010x}{} bt={}",
                    cpu.get_pc(),
                    arguments(cpu),
                    backtrace(cpu, 12),
                );
                self.traps += 1;
            }
        }
        if self.watch_hits < MAX_HITS && self.watch_pc.contains(&cpu.get_pc()) {
            println!(
                "[watch-pc] {:#010x} at step {done}{} bt={}",
                cpu.get_pc(),
                arguments(cpu),
                backtrace(cpu, 12),
            );
            self.watch_hits += 1;
        }
    }

    /// Everything worth knowing about where a run stopped: the registers, the
    /// call stack, and whatever `DUMP=` asked for.
    ///
    /// Printed for a fault, a halt and an exhausted step budget alike. A run
    /// that stops any of those three ways stops somewhere, and the address a
    /// fault was holding is no easier to guess than the one a hang was.
    pub fn stop_state(&self, cpu: &Cpu) {
        if self.dumps.is_empty() {
            return;
        }
        print!("{}", cpu.reg_dump());
        println!("backtrace: {}", backtrace(cpu, 24));
        for spec in &self.dumps {
            let base = match spec.base {
                DumpBase::Absolute(addr) => u64::from(addr),
                DumpBase::Register(reg) => cpu.read_x(reg),
                DumpBase::StackPointer => cpu.sp(),
                DumpBase::ProgramCounter => u64::from(cpu.get_pc()),
            };
            let at = (base as i64).wrapping_add(spec.offset) as u32;
            println!("[dump] {} = {at:#010x} ({:#x} bytes)", spec.label, spec.len);
            // Words rather than bytes because what is being read here is
            // almost always a structure: a null in a field is the thing being
            // looked for, and a run of pointers is what says a table was
            // populated. The ASCII column is there because the other half of
            // what turns up in guest memory is names.
            for line in (0..spec.len).step_by(16) {
                let addr = at.wrapping_add(line);
                let mut words = String::new();
                let mut ascii = String::new();
                for word in 0..4u32 {
                    let value = cpu.mem.read_u32(addr.wrapping_add(word * 4)).unwrap_or(0);
                    words.push_str(&format!(" {value:08x}"));
                    for byte in value.to_le_bytes() {
                        ascii.push(if (0x20..0x7f).contains(&byte) {
                            byte as char
                        } else {
                            '.'
                        });
                    }
                }
                println!("  {addr:#010x}:{words}  {ascii}");
            }
        }
    }

    /// What the run collected, once it has stopped.
    pub fn report(&self) {
        for (pc, count) in &self.readers {
            println!("[reader] {pc:#010x} {count}");
        }
    }
}

/// The order the modules of an ExeFS load in.
///
/// One list, because three examples carried their own and two of them
/// stopped at `subsdk4` — a title with a `subsdk5` booted differently
/// depending on which tool you ran it with.
const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

/// What a container turned out to be, decided by looking at it.
///
/// `PFS0` sits at offset 0 and the NCA magic at 0x200. An NCA straight off the
/// CDN keeps its header encrypted, so its magic is invisible until `prod.keys`
/// decrypts it — those fall back to the extension, which is what the frontend
/// does with them too (`web/main/filetype.ts`).
enum Kind {
    Pfs0,
    Nca,
}

fn container_kind(path: &Path) -> Kind {
    let mut head = [0u8; 0x204];
    // The whole window or nothing: a short read would leave the magic
    // positions as zeros and quietly look like "not an NCA".
    let read = open_source(path).read_exact_at(0, &mut head).is_ok();
    if read && &head[..4] == b"PFS0" {
        return Kind::Pfs0;
    }
    let is_nca = read && matches!(&head[0x200..0x204], b"NCA3" | b"NCA2" | b"NCA0");
    let by_name = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nca"));
    if is_nca || by_name {
        Kind::Nca
    } else {
        Kind::Pfs0
    }
}

/// A container's control data — the NACP and the icon that name a title, and
/// the save-data sizes it declares.
///
/// A free function rather than a method on [`Title`] because a Control NCA is
/// a container in its own right: `title_info` is handed one directly, and it
/// has no ExeFS for a `Title` to be built around. Any bundled ticket's title
/// key goes into `keys`, since the Control NCA is encrypted under the same
/// title key as the program beside it.
pub fn open_control(
    container: impl AsRef<Path>,
    keys: &mut KeySet,
) -> Result<switch_core::control::Control, switch_core::Error> {
    let path = container.as_ref();
    let src = open_source(path);
    match container_kind(path) {
        Kind::Nca => switch_core::control::Control::from_source(&src, keys),
        Kind::Pfs0 => {
            let pfs0 = switch_core::nsp::Pfs0::read_from(&src)?;
            let Some((index, nca)) =
                switch_core::control::find_control_nca(&pfs0.files, &src, keys)
            else {
                return Err(switch_core::Error::Nca(format!(
                    "no Control NCA in {}",
                    path.display()
                )));
            };
            if let Err(e) =
                switch_core::ticket::load_bundled_title_key(keys, &nca, &pfs0.files, &src)
            {
                eprintln!("no title key for the Control NCA: {e}");
            }
            let window = pfs0.file_source(&src, index)?;
            switch_core::control::Control::from_source(window, keys)
        }
    }
}

/// A title's Program NCA, opened without reading the container in.
///
/// A retail container is the whole game. Just Dance 2017's NSP is 13.4 GB,
/// and reading it into a `Vec` and then decrypting its RomFS beside it wants
/// 21 GB on a machine that also has to hold the guest — which is not slow, it
/// is a machine that stops responding. `boot_nsp` learned that and streams;
/// every other example kept its own copy of the loading code and kept
/// slurping.
///
/// So opening a container lives here, and the only part held in memory is the
/// ExeFS, which is the executable and is tens of megabytes at most. The RomFS
/// is handed to the CPU as a source, because the guest reads it a range at a
/// time through `IStorage` and there is nothing to gain by holding the rest.
pub struct Title {
    pub nca: switch_core::nca::Nca,
    pub keys: KeySet,
    /// The ExeFS, decrypted and hash-verified.
    pub exefs: Vec<u8>,
    pub exefs_pfs0: switch_core::nsp::Pfs0,
    /// The container, and where the Program NCA sits inside it. Kept rather
    /// than a handle because each read wants its own: the CPU holds the
    /// RomFS source for the whole run and cannot borrow one this shares.
    path: std::path::PathBuf,
    program: (u64, u64),
    /// The base game, when the NCA above is an *update's* Program NCA. An
    /// update carries no game of its own, so its RomFS only reads over this
    /// one. See [`Update`].
    base: Option<(std::path::PathBuf, (u64, u64), switch_core::nca::Nca)>,
}

impl Title {
    /// Open whichever kind of container this is, by looking at it.
    ///
    /// A tool that only takes one kind is a tool that cannot be pointed at the
    /// title you want to debug: `screenshot_gpu` took a bare NCA, so the
    /// backend it exists to measure could not be run against a retail game at
    /// all. The web frontend already decides this by header rather than by
    /// name (`web/main/filetype.ts`); this is the same rule.
    pub fn open(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        match container_kind(&path) {
            Kind::Nca => Title::open_nca(path, prod, title),
            Kind::Pfs0 => Title::open_nsp(path, prod, title),
        }
    }

    /// Open the Program NCA inside an NSP, resolving its title key from a
    /// bundled ticket if the container has one.
    pub fn open_nsp(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        let mut keys = keys(prod, title);
        let (program, nca) = open_program(&path, &mut keys);
        // `UPDATE=<path.nsp>` runs the title as patched: the update's own
        // Program NCA supplies every module, and this container stays open
        // for the RomFS underneath it.
        match Update::from_env(&mut keys) {
            Some(update) => {
                let base = Some((path, program, nca));
                Title::finish(update.path, update.program, update.nca, keys, base)
            }
            None => Title::finish(path, program, nca, keys, None),
        }
    }

    /// Open a bare Program NCA — a system applet, which ships inside firmware
    /// rather than in an NSP and so has no PFS0 around it.
    pub fn open_nca(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        let src = open_source(&path);
        let size = switch_core::source::ByteSource::len(&src);
        let keys = keys(prod, title);
        let nca = switch_core::nca::Nca::parse_source(&src, Some(&keys))
            .unwrap_or_else(|e| die(&format!("{} is not an NCA: {e}", path.display())));
        Title::finish(path, (0, size), nca, keys, None)
    }

    /// Read the ExeFS, which is the one part of a container worth holding.
    fn finish(
        path: std::path::PathBuf,
        program: (u64, u64),
        nca: switch_core::nca::Nca,
        keys: KeySet,
        base: Option<(std::path::PathBuf, (u64, u64), switch_core::nca::Nca)>,
    ) -> Title {
        let index = nca
            .exefs_section_index()
            .unwrap_or_else(|| die(&format!("{} has no ExeFS section", path.display())));
        let exefs = nca
            .read_pfs0_section(program_window(&path, program.0, program.1), &keys, index)
            .unwrap_or_else(|e| die(&format!("reading the ExeFS: {e}")));
        let exefs_pfs0 = switch_core::nsp::Pfs0::parse(&exefs)
            .unwrap_or_else(|e| die(&format!("the ExeFS is not a PFS0: {e}")));
        Title {
            nca,
            keys,
            exefs,
            exefs_pfs0,
            path,
            program,
            base,
        }
    }

    /// The modules to boot, in [`MODULE_ORDER`].
    pub fn modules(&self) -> Vec<(&str, &[u8])> {
        MODULE_ORDER
            .iter()
            .filter_map(|&name| {
                let file = self.exefs_pfs0.find(name)?;
                let start = file.offset as usize;
                Some((name, &self.exefs[start..start + file.size as usize]))
            })
            .collect()
    }

    /// Give the guest this title's RomFS, streamed off disk.
    ///
    /// Optional: Meta and Control content has none, and a title without one
    /// boots fine and simply has no asset storage mounted.
    pub fn mount_romfs(&self, cpu: &mut Cpu) {
        match self.romfs_source() {
            Some(Ok(romfs)) => cpu.set_romfs_source(Box::new(romfs)),
            Some(Err(e)) => eprintln!("this title's RomFS could not be opened: {e}"),
            None => {}
        }
    }

    /// The same source [`Title::mount_romfs`] hands the guest, for a tool that
    /// reads the image itself rather than running the title.
    ///
    /// `None` when the NCA has no RomFS section at all, which is a different
    /// thing from one that would not open. Boxed because a patched RomFS —
    /// two containers read as one — is a different type from a plain one, and
    /// every caller wants the same thing from either.
    pub fn romfs_source(&self) -> Option<Result<Box<dyn ByteSource>, switch_core::Error>> {
        let window = program_window(&self.path, self.program.0, self.program.1);
        match &self.base {
            Some((base_path, base_program, base_nca)) => Some(
                switch_core::bktr::patched_romfs_source(
                    &self.nca,
                    window,
                    base_nca,
                    program_window(base_path, base_program.0, base_program.1),
                    &self.keys,
                )
                .map(|romfs| Box::new(romfs) as Box<dyn ByteSource>),
            ),
            None => {
                let index = self.nca.romfs_section_index()?;
                Some(
                    self.nca
                        .romfs_source(window, &self.keys, index)
                        .map(|romfs| Box::new(romfs) as Box<dyn ByteSource>),
                )
            }
        }
    }

    /// Boot the title: its address space, its program id, and its modules.
    /// Returns where each module landed.
    ///
    /// The system resource size comes from the title's own manifest and has
    /// to be set before the modules load, since `nn::init` reads the
    /// resulting figures as soon as it runs.
    pub fn boot(&self, cpu: &mut Cpu) -> Vec<switch_core::nso::LoadedNso> {
        // `DOCKED=1` boots as a docked console — a 1080p display, which is
        // what the frontend's dock toggle sets and so what a browser session
        // is usually looking at. The default here is handheld, and a title
        // told 720p while its swapchain is 1080p composites its frame into a
        // corner of it. That is not a rendering bug, it is the two halves
        // disagreeing, and measuring a retail title in a mode nobody runs it
        // in is how a corner gets mistaken for one.
        if env::var("DOCKED").is_ok() {
            cpu.set_operation_mode(switch_core::cpu::OperationMode::Docked);
        }
        cpu.set_system_resource_size(switch_core::npdm::Npdm::system_resource_size_of(
            &self.exefs_pfs0,
            &self.exefs,
        ));
        cpu.set_program_id(self.nca.program_id);
        let modules = self.modules();
        let loaded = cpu
            .boot_retail_program(&modules)
            .unwrap_or_else(|e| die(&format!("booting {} modules: {e:?}", modules.len())));
        // After the program id, which is what add-on content is numbered
        // against, and after the modules, which is where the browser has to
        // do it — booting clears the diagnostics it reads them through.
        mount_add_on_content(cpu, &self.keys);
        loaded
    }

    /// The container the *game* is in, which is the base one when an update
    /// is stacked over it. An update NSP carries no Control NCA of its own.
    pub fn container(&self) -> &Path {
        match &self.base {
            Some((path, _, _)) => path,
            None => &self.path,
        }
    }

    /// This title's control data, or the reason it could not be read.
    pub fn control(&self) -> Result<switch_core::control::Control, switch_core::Error> {
        open_control(self.container(), &mut self.keys.clone())
    }
}

/// An update container, which `UPDATE=<path.nsp>` names.
///
/// An update NSP holds no game. Its Program NCA carries the patched modules
/// in full — so an update runs by booting *its* ExeFS — and a RomFS section
/// holding only the ranges the update changed, which reads over the base
/// container's RomFS and nowhere else. The browser pairs the two files the
/// user picked; this is the same pairing for a run without one.
pub struct Update {
    pub nca: switch_core::nca::Nca,
    path: std::path::PathBuf,
    program: (u64, u64),
}

impl Update {
    /// The update `UPDATE=<path.nsp>` names, or `None` when it named none.
    ///
    /// Its title key goes into `keys`: an update is signed and ticketed
    /// separately from the game it patches, so the base container's key does
    /// not open it.
    pub fn from_env(keys: &mut KeySet) -> Option<Update> {
        let path = std::path::PathBuf::from(env::var("UPDATE").ok()?);
        let (program, nca) = open_program(&path, keys);
        if !nca.is_update() {
            die(&format!(
                "{} is not an update: its RomFS is a title's own, not a patch over one",
                path.display()
            ));
        }
        Some(Update { nca, path, program })
    }

    /// A fresh window over this update's Program NCA.
    pub fn program_window(&self) -> switch_core::source::Window<switch_core::source::FileSource> {
        program_window(&self.path, self.program.0, self.program.1)
    }
}

/// Give the running title the add-on content `DLC=<a.nsp>,<b.nsp>` names.
///
/// A DLC container is nothing like an update: no program, no patch, no base to
/// read over. Each is one Data NCA with an ordinary RomFS whose title id is
/// the title's add-on base plus an index, and a title mounts it by that id —
/// so all this does is decrypt each one and hand it over. Content belonging to
/// another title is reported and skipped, since an index the title cannot
/// number is one nothing will ever ask for.
///
/// Reads its own keys out of each container: a DLC is bought, and so ticketed,
/// separately from the game and from every other piece of it.
pub fn mount_add_on_content(cpu: &mut Cpu, keys: &KeySet) {
    let Ok(list) = env::var("DLC") else { return };
    for path in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let path = Path::new(path);
        let src = open_source(path);
        let pfs0 = match switch_core::nsp::Pfs0::read_from(&src) {
            Ok(pfs0) => pfs0,
            Err(e) => {
                eprintln!("{} is not a PFS0: {e}", path.display());
                continue;
            }
        };
        let mut keys = keys.clone();
        for (index, file) in pfs0.files.iter().enumerate() {
            if !file.name.to_ascii_lowercase().ends_with(".nca") {
                continue;
            }
            let Ok(window) = pfs0.file_source(&src, index) else {
                continue;
            };
            let Ok(nca) = switch_core::nca::Nca::parse_source(&window, Some(&keys)) else {
                continue;
            };
            use switch_core::nca::ContentType;
            if !matches!(
                nca.content_type,
                ContentType::Data | ContentType::PublicData
            ) {
                continue;
            }
            if let Err(e) =
                switch_core::ticket::load_bundled_title_key(&mut keys, &nca, &pfs0.files, &src)
            {
                eprintln!("no title key for {}: {e}", file.name);
            }
            let Some(section) = nca.romfs_section_index() else {
                continue;
            };
            // Its own handle on the file: the CPU keeps every archive for the
            // whole run and cannot borrow the one this scan is using.
            let owned = match switch_core::source::Window::new(
                open_source(path),
                file.offset,
                file.size,
                "add-on content nca",
            ) {
                Ok(owned) => owned,
                Err(e) => {
                    eprintln!("{}: {e}", file.name);
                    continue;
                }
            };
            match nca.romfs_source(owned, &keys, section) {
                Ok(romfs) => {
                    let size = romfs.len();
                    match cpu.add_add_on_content(nca.title_id, Box::new(romfs)) {
                        Some(index) => println!(
                            "add-on content {:016x} mounted as index {index}, {size:#x} bytes",
                            nca.title_id
                        ),
                        None => println!(
                            "add-on content {:016x} is not this title's — not mounted",
                            nca.title_id
                        ),
                    }
                }
                Err(e) => println!("add-on content {:016x} unreadable: {e}", nca.title_id),
            }
        }
    }
}

/// The Program NCA in a container: where it sits, its parsed header, and any
/// bundled ticket's title key added to `keys`.
///
/// The *last* Program NCA rather than the first: a container that carries
/// more than one is an update, and the later entry is the one it updates to.
fn open_program(path: &Path, keys: &mut KeySet) -> ((u64, u64), switch_core::nca::Nca) {
    let src = open_source(path);
    let pfs0 = switch_core::nsp::Pfs0::read_from(&src)
        .unwrap_or_else(|e| die(&format!("{} is not a PFS0: {e}", path.display())));
    let mut found = None;
    for (index, file) in pfs0.files.iter().enumerate() {
        if !file.name.to_ascii_lowercase().ends_with(".nca") {
            continue;
        }
        let Ok(window) = pfs0.file_source(&src, index) else {
            continue;
        };
        match switch_core::nca::Nca::parse_source(&window, Some(&*keys)) {
            Ok(nca) if nca.content_type == switch_core::nca::ContentType::Program => {
                found = Some((index, file.offset, file.size));
            }
            _ => {}
        }
    }
    let Some((index, offset, size)) = found else {
        die(&format!("no Program NCA in {}", path.display()))
    };
    let window = pfs0
        .file_source(&src, index)
        .unwrap_or_else(|e| die(&format!("window over the program nca: {e}")));
    let nca = switch_core::nca::Nca::parse_source(&window, Some(&*keys))
        .unwrap_or_else(|e| die(&format!("parsing the program nca: {e}")));
    // Title-key crypto needs the key itself from somewhere. Scene releases
    // bundle the ticket next to the content.
    if let Err(e) = switch_core::ticket::load_bundled_title_key(keys, &nca, &pfs0.files, &src) {
        eprintln!("no title key for {}: {e}", path.display());
    }
    ((offset, size), nca)
}

fn open_source(path: &Path) -> switch_core::source::FileSource {
    switch_core::source::FileSource::open(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {e}", path.display())))
}

/// A fresh window over the Program NCA's bytes. Each reader gets its own,
/// since [`switch_core::nca::Nca`]'s section readers take a source by value.
fn program_window(
    path: &Path,
    offset: u64,
    size: u64,
) -> switch_core::source::Window<switch_core::source::FileSource> {
    switch_core::source::Window::new(open_source(path), offset, size, "program nca")
        .unwrap_or_else(|e| die(&format!("window over the program nca: {e}")))
}

fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
}
