//! Scaffolding shared by the examples: argument handling, key loading,
//! booting, driving the machine, and writing a frame out.
#![allow(dead_code)]

/// Image metadata tables, for tools that speak in file names.
pub mod romfs;

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;
use switch_core::cpu::Cpu;
use switch_core::gpu::Framebuffer;
use switch_core::keys::KeySet;
use switch_core::source::ByteSource;

/// The font `pl:u` serves when no firmware fonts are registered.
pub const FALLBACK_FONT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/font.ttf");

/// Print a usage line and exit.
pub fn usage(line: &str) -> ! {
    eprintln!("usage: {line}");
    std::process::exit(1)
}

/// Positional argument `n` (1-based), or exit with the usage line.
pub fn arg(n: usize, line: &str) -> String {
    match env::args().nth(n) {
        Some(arg) => arg,
        None => usage(line),
    }
}

pub fn opt_arg(n: usize) -> Option<String> {
    env::args().nth(n)
}

pub fn opt_num(n: usize) -> Option<u64> {
    env::args().nth(n)?.parse().ok()
}

/// Parse a hexadecimal address (`0x` optional), or exit.
pub fn hex(text: &str) -> u32 {
    match u32::from_str_radix(text.trim().trim_start_matches("0x"), 16) {
        Ok(value) => value,
        Err(_) => {
            eprintln!("{text:?} is not a hexadecimal address");
            std::process::exit(1)
        }
    }
}

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

/// The `<container> <prod.keys> [title.keys]` arguments. Argument 3 is the
/// title keys only when named like a keys file; the rest is [`Args::rest`].
pub struct Args {
    pub container: String,
    pub prod: String,
    pub title: Option<String>,
    font: Option<String>,
    rest: Vec<String>,
    line: String,
}

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
        font: None,
        rest,
        line: line.to_string(),
    }
}

/// Arguments of a tool that runs a `.nro` or a retail container with its
/// keys. A `.ttf` anywhere in the tail is the shared font.
pub fn program_args(line: &str) -> Args {
    let mut positional = env::args().skip(1);
    let Some(container) = positional.next() else {
        usage(line)
    };
    let mut rest: Vec<String> = positional.collect();
    let take_named = |rest: &mut Vec<String>, suffix: &str| {
        let at = rest
            .iter()
            .position(|arg| arg.to_ascii_lowercase().ends_with(suffix))?;
        Some(rest.remove(at))
    };
    let font = take_named(&mut rest, ".ttf");
    // Only a retail target needs keys.
    let (prod, title) = match target_kind(Path::new(&container)) {
        Kind::Nro => (String::new(), None),
        _ => {
            let Some(prod) = (!rest.is_empty()).then(|| rest.remove(0)) else {
                usage(line)
            };
            (prod, take_named(&mut rest, ".keys"))
        }
    };
    Args {
        container,
        prod,
        title,
        font,
        rest,
        line: line.to_string(),
    }
}

pub struct Program {
    form: Form,
    /// The font named on the command line, served as `pl:u`.
    font: Option<Vec<u8>>,
}

enum Form {
    Homebrew(Vec<u8>),
    /// An `.nsp`, an `.xci`, or a bare Program `.nca`.
    Retail(Box<Title>),
}

pub struct Booted {
    pub modules: Vec<switch_core::nso::LoadedNso>,
    /// System data archives registered from `SWITCH_FIRMWARE`.
    pub archives: usize,
}

impl Program {
    /// Boot into an already bootstrapped `cpu`.
    pub fn boot(&self, cpu: &mut Cpu) -> Booted {
        if let Form::Retail(title) = &self.form {
            title.mount_romfs(cpu);
        }
        match &self.font {
            Some(bytes) => cpu.set_shared_font(bytes.clone()),
            None => load_fallback_font(cpu),
        }
        match &self.form {
            Form::Homebrew(image) => {
                cpu.boot_homebrew(image)
                    .unwrap_or_else(|e| die(&format!("booting the NRO: {e:?}")));
                Booted {
                    modules: Vec::new(),
                    archives: 0,
                }
            }
            Form::Retail(title) => {
                // After the font (firmware fonts override it), before the modules.
                let archives = register_firmware(cpu, &title.keys);
                Booted {
                    modules: title.boot(cpu),
                    archives,
                }
            }
        }
    }

    /// The retail title behind this program; `None` for homebrew.
    pub fn title(&self) -> Option<&Title> {
        match &self.form {
            Form::Retail(title) => Some(title),
            Form::Homebrew(_) => None,
        }
    }
}

impl Args {
    pub fn open(&self) -> Title {
        Title::open(&self.container, &self.prod, self.title.as_ref())
    }

    pub fn open_program(&self) -> Program {
        let form = match target_kind(Path::new(&self.container)) {
            Kind::Nro => Form::Homebrew(read(&self.container)),
            _ => Form::Retail(Box::new(self.open())),
        };
        Program {
            form,
            font: self.font.as_ref().map(read),
        }
    }

    /// The keys alone, for tools that read the container themselves.
    pub fn keys(&self) -> KeySet {
        keys(&self.prod, self.title.as_ref())
    }

    /// Argument `n` after the triple, counting from 0.
    pub fn rest(&self, n: usize) -> Option<&str> {
        self.rest.get(n).map(String::as_str)
    }

    pub fn need(&self, n: usize) -> &str {
        match self.rest.get(n) {
            Some(arg) => arg,
            None => usage(&self.line),
        }
    }

    pub fn rest_num(&self, n: usize) -> Option<u64> {
        self.rest.get(n)?.parse().ok()
    }
}

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
    parse_span(&env::var(name).ok()?)
}

/// Comma-separated `lo:len` pairs; unparsable pairs are skipped.
pub fn env_spans(name: &str) -> Vec<(u32, u32)> {
    env::var(name)
        .map(|raw| raw.split(',').filter_map(parse_span).collect())
        .unwrap_or_default()
}

fn parse_span(raw: &str) -> Option<(u32, u32)> {
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

/// Give the guest the fallback shared font, warning if it is missing and no
/// firmware fonts will replace it.
pub fn load_fallback_font(cpu: &mut Cpu) {
    match fs::read(FALLBACK_FONT) {
        Ok(bytes) => cpu.set_shared_font(bytes),
        Err(e) => eprintln!("no font at {FALLBACK_FONT} ({e}): text will not render"),
    }
}

/// Register every system data archive in `SWITCH_FIRMWARE`, if it is set.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// Slices through `Cpu::run`, which reaches the block translator.
    Blocks,
    /// One instruction at a time through `Cpu::step`, for per-instruction work.
    Instructions,
}

/// Instructions [`Pace::Blocks`] runs between two `tick` calls.
const SLICE: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

#[derive(Debug, Default)]
pub struct Run {
    pub steps: u64,
    pub fault: Option<String>,
    pub halted: bool,
}

/// Per-frame wall-clock cost, printed when `FRAME_TIMES=1`.
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
        // Share a multi-present slice's time out evenly.
        let presented = cpu.nv.gpu.frames - self.last_count;
        let each = now.duration_since(self.last_at).as_secs_f64() / presented as f64;
        self.deltas
            .extend(std::iter::repeat_n(each, presented as usize));
        self.last_count = cpu.nv.gpu.frames;
        self.last_at = now;
    }

    /// Report the frames after the first, whose cost is the whole boot.
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

/// Run until `tick` returns [`Flow::Stop`], the machine halts or faults, or
/// `budget` instructions retire.
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

pub fn run_to(cpu: &mut Cpu, budget: u64, mut until: impl FnMut(&Cpu) -> bool) -> Run {
    drive(cpu, Pace::Blocks, budget, |cpu, _| {
        if until(cpu) {
            Flow::Stop
        } else {
            Flow::Continue
        }
    })
}

/// Write a framebuffer as a binary PPM and report its lit and opaque pixel
/// counts.
pub fn write_ppm(path: impl AsRef<Path>, fb: &Framebuffer) -> usize {
    let path = path.as_ref();
    let mut ppm = format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
    let mut lit = 0usize;
    let mut opaque = 0usize;
    for px in &fb.pixels {
        let (r, g, b) = (*px as u8, (*px >> 8) as u8, (*px >> 16) as u8);
        if r != 0 || g != 0 || b != 0 {
            lit += 1;
        }
        if (*px >> 24) as u8 == 0xFF {
            opaque += 1;
        }
        ppm.extend_from_slice(&[r, g, b]);
    }
    if let Err(e) = fs::write(path, ppm) {
        eprintln!("cannot write {}: {e}", path.display());
        std::process::exit(1);
    }
    println!(
        "wrote {}: {}x{}, {lit}/{} non-black, {opaque}/{} opaque",
        path.display(),
        fb.width,
        fb.height,
        fb.pixels.len(),
        fb.pixels.len()
    );
    lit
}

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
    // Peak RAM use against the cap.
    let used = cpu.mem.mapped_bytes();
    let cap = cpu.mem.max_mapped_bytes();
    println!(
        "memory: {:.1} MiB of {} MiB backed ({:.0}%)",
        used as f64 / (1024.0 * 1024.0),
        cap / (1024 * 1024),
        used as f64 * 100.0 / cap as f64
    );
}

/// Where one `DUMP=` entry starts: a register or an absolute address.
enum DumpBase {
    Absolute(u32),
    Register(u8),
    StackPointer,
    ProgramCounter,
}

struct DumpSpec {
    label: String,
    base: DumpBase,
    /// `*<base>`: follow the pointer stored at the base.
    deref: bool,
    offset: i64,
    len: u32,
}

fn parse_dump_specs(spec: &str) -> Vec<DumpSpec> {
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
            // Search from the right so `x23+0x10` splits on the sign.
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
            let (deref, base) = match base.trim().strip_prefix('*') {
                Some(pointer) => (true, pointer),
                None => (false, base),
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
                deref,
                offset,
                len,
            })
        })
        .collect()
}

fn dump_regions(cpu: &Cpu, specs: &[DumpSpec]) -> String {
    let mut out = String::new();
    for spec in specs {
        let base = match spec.base {
            DumpBase::Absolute(addr) => u64::from(addr),
            DumpBase::Register(reg) => cpu.read_x(reg),
            DumpBase::StackPointer => cpu.sp(),
            DumpBase::ProgramCounter => u64::from(cpu.get_pc()),
        };
        let base = if spec.deref {
            let at = base as u32;
            let low = cpu.mem.read_u32(at).unwrap_or(0);
            let high = cpu.mem.read_u32(at.wrapping_add(4)).unwrap_or(0);
            u64::from(high) << 32 | u64::from(low)
        } else {
            base
        };
        let at = (base as i64).wrapping_add(spec.offset) as u32;
        out.push_str(&format!(
            "[dump] {} = {at:#010x} ({:#x} bytes)\n",
            spec.label, spec.len
        ));
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
            out.push_str(&format!("  {addr:#010x}:{words}  {ascii}\n"));
        }
    }
    out
}

/// How many times a watchpoint or pc watch reports before going quiet.
const MAX_HITS: u32 = 24;

/// `x0`..`x7`, a call's arguments.
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

/// Debugging knobs, read from the environment once:
///
/// - `TRAP_WRITE=<addr>:<hex size>`: pc and call stack of the first writes
///   into a region.
/// - `TRAP_READ=<addr>:<hex size>`: every distinct pc that reads a region,
///   counted.
/// - `TRAP_LAST=<n>`: print the last `n` trapped writes when the run stops.
/// - `TRAP_ZERO=1`: also trap writes of zero.
/// - `WATCH_PC=<addr>[,...]`: argument registers and call stack the first few
///   times execution reaches an address.
/// - `WATCH_LAST=<n>`: print the last `n` `WATCH_PC` hits when the run stops.
/// - `WATCH_DUMP=<spec>`: hex-dump memory (`DUMP` syntax) at each `WATCH_PC` hit.
/// - `WATCH_REGS=1`: print every register at each `WATCH_PC` hit.
/// - `DUMP=<base>[+<hex>][:<hex length>][,...]`: hex-dump memory where the run
///   stopped. `<base>` is `x0`..`x30`, `sp`, `pc` or an address; a leading `*`
///   follows the pointer stored there.
pub struct Debug {
    write_trap: Option<(u32, u32)>,
    read_trap: Option<(u32, u32)>,
    trap_zero: bool,
    /// `TRAP_LAST`: capacity and the latest trapped writes.
    trap_last: usize,
    last_traps: std::collections::VecDeque<String>,
    watch_pc: Vec<u32>,
    watch_regs: bool,
    watch_dumps: Vec<DumpSpec>,
    /// `WATCH_LAST`: capacity and the latest watch hits.
    watch_last: usize,
    last_watches: std::collections::VecDeque<String>,
    dumps: Vec<DumpSpec>,
    traps: u32,
    watch_hits: u32,
    /// Pcs that read the `TRAP_READ` region, with counts.
    readers: std::collections::BTreeMap<u32, u64>,
    /// Read watchpoints report a step late, so the reading pc is carried over.
    reader_pc: u32,
}

impl Debug {
    pub fn from_env() -> Debug {
        Debug {
            write_trap: env_span("TRAP_WRITE"),
            read_trap: env_span("TRAP_READ"),
            trap_zero: env::var("TRAP_ZERO").is_ok(),
            trap_last: env_u64("TRAP_LAST", 0) as usize,
            last_traps: std::collections::VecDeque::new(),
            watch_pc: env_hex_list("WATCH_PC"),
            watch_regs: env::var("WATCH_REGS").is_ok(),
            watch_dumps: env::var("WATCH_DUMP")
                .map(|spec| parse_dump_specs(&spec))
                .unwrap_or_default(),
            watch_last: env_u64("WATCH_LAST", 0) as usize,
            last_watches: std::collections::VecDeque::new(),
            dumps: env::var("DUMP")
                .map(|spec| parse_dump_specs(&spec))
                .unwrap_or_default(),
            traps: 0,
            watch_hits: 0,
            readers: std::collections::BTreeMap::new(),
            reader_pc: 0,
        }
    }

    /// Whether any knob needs [`Pace::Instructions`].
    pub fn stepwise(&self) -> bool {
        self.write_trap.is_some() || self.read_trap.is_some() || !self.watch_pc.is_empty()
    }

    /// Install the watchpoints; call after boot.
    pub fn arm(&self, cpu: &mut Cpu) {
        if let Some((lo, hi)) = self.write_trap {
            cpu.mem.watch_writes(lo, hi - lo);
        }
        if let Some((lo, hi)) = self.read_trap {
            cpu.mem.watch_reads(lo, hi - lo);
        }
    }

    /// Report what tripped since the last instruction, from a
    /// [`Pace::Instructions`] tick.
    pub fn tick(&mut self, cpu: &mut Cpu, done: u64) {
        if self.read_trap.is_some() && cpu.mem.take_read_hit().is_some() {
            *self.readers.entry(self.reader_pc).or_default() += 1;
        }
        self.reader_pc = cpu.get_pc();
        if let Some(at) = cpu.mem.take_watch_hit() {
            let value = cpu.mem.read_u32(at & !3).unwrap_or(0);
            let wanted = value != 0 || self.trap_zero;
            let line = || {
                format!(
                    "[trap] wrote {at:#010x} = {value:#010x} at step {done} pc={:#010x} \
                     thread={:#x}{} bt={}",
                    cpu.get_pc(),
                    cpu.current_thread_handle(),
                    arguments(cpu),
                    backtrace(cpu, 12),
                )
            };
            if wanted && self.trap_last > 0 {
                if self.last_traps.len() == self.trap_last {
                    self.last_traps.pop_front();
                }
                self.last_traps.push_back(line());
            } else if wanted && self.traps < MAX_HITS {
                println!("{}", line());
                self.traps += 1;
            }
        }
        if self.watch_pc.contains(&cpu.get_pc()) {
            let mut line = format!(
                "[watch-pc] {:#010x} at step {done} thread={:#x}{} bt={}\n",
                cpu.get_pc(),
                cpu.current_thread_handle(),
                arguments(cpu),
                backtrace(cpu, 12),
            );
            if self.watch_regs {
                line.push_str(&cpu.reg_dump());
            }
            line.push_str(&dump_regions(cpu, &self.watch_dumps));
            if self.watch_last > 0 {
                if self.last_watches.len() == self.watch_last {
                    self.last_watches.pop_front();
                }
                self.last_watches.push_back(line);
            } else if self.watch_hits < MAX_HITS {
                print!("{line}");
                self.watch_hits += 1;
            }
        }
    }

    /// Print the registers, call stack, and `DUMP=` regions where the run stopped.
    pub fn stop_state(&self, cpu: &Cpu) {
        if self.dumps.is_empty() {
            return;
        }
        print!("{}", cpu.reg_dump());
        println!("backtrace: {}", backtrace(cpu, 24));
        print!("{}", dump_regions(cpu, &self.dumps));
    }

    pub fn report(&self) {
        for line in &self.last_traps {
            println!("{line}");
        }
        for line in &self.last_watches {
            print!("{line}");
        }
        for (pc, count) in &self.readers {
            println!("[reader] {pc:#010x} {count}");
        }
    }
}

/// The load order of an ExeFS's modules.
const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

/// A container's kind, detected from its magic, or from the extension for an
/// NCA whose header is still encrypted.
enum Kind {
    Container,
    Nca,
    Nro,
}

fn target_kind(path: &Path) -> Kind {
    let mut head = [0u8; 0x204];
    // A short read would leave zeros and look like "not an NCA".
    let read = open_source(path).read_exact_at(0, &mut head).is_ok();
    if read && (&head[..4] == b"PFS0" || &head[0x100..0x104] == b"HEAD") {
        return Kind::Container;
    }
    // A crt0 stub may precede the NRO magic, so search rather than index.
    if read
        && head[..0x100]
            .windows(4)
            .any(|w| w == switch_core::nro::NRO0_MAGIC.to_le_bytes())
    {
        return Kind::Nro;
    }
    let is_nca = read && matches!(&head[0x200..0x204], b"NCA3" | b"NCA2" | b"NCA0");
    let named = |ext: &str| {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
    };
    if is_nca || named("nca") {
        Kind::Nca
    } else if named("nro") {
        Kind::Nro
    } else {
        Kind::Container
    }
}

/// A container's control data: NACP, icon, and declared save-data sizes. Any
/// bundled ticket's title key goes into `keys`.
pub fn open_control(
    container: impl AsRef<Path>,
    keys: &mut KeySet,
) -> Result<switch_core::control::Control, switch_core::Error> {
    let path = container.as_ref();
    let src = open_source(path);
    match target_kind(path) {
        Kind::Nca => switch_core::control::Control::from_source(&src, keys),
        // A homebrew NRO carries its own NACP, but not as a Control NCA.
        Kind::Nro => Err(switch_core::Error::BadMagic {
            what: "control data (this is a homebrew NRO)".to_string(),
            found: 0,
        }),
        Kind::Container => {
            let pfs0 = switch_core::xci::read_container(&src)?;
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

/// A title's Program NCA, read by range. Only the ExeFS is held in memory.
pub struct Title {
    pub nca: switch_core::nca::Nca,
    pub keys: KeySet,
    /// The ExeFS, decrypted and hash-verified.
    pub exefs: Vec<u8>,
    pub exefs_pfs0: switch_core::nsp::Pfs0,
    /// The container and the Program NCA's extent; each reader opens its own handle.
    path: std::path::PathBuf,
    program: (u64, u64),
    /// The base game, when the NCA above belongs to an [`Update`].
    base: Option<(std::path::PathBuf, (u64, u64), switch_core::nca::Nca)>,
}

impl Title {
    /// Open whichever kind of container this is, by looking at it.
    pub fn open(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        match target_kind(&path) {
            Kind::Nca => Title::open_nca(path, prod, title),
            Kind::Container => Title::open_container(path, prod, title),
            Kind::Nro => die(&format!(
                "{} is a homebrew NRO, not a retail container",
                path.display()
            )),
        }
    }

    /// Open the Program NCA in an `.nsp` or `.xci`, with any bundled ticket's
    /// title key.
    pub fn open_container(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        let mut keys = keys(prod, title);
        let (program, nca) = open_program(&path, &mut keys);
        // `UPDATE=<path.nsp>` boots the update's modules over this container's RomFS.
        match Update::from_env(&mut keys) {
            Some(update) => {
                let base = Some((path, program, nca));
                Title::finish(update.path, update.program, update.nca, keys, base)
            }
            None => Title::finish(path, program, nca, keys, None),
        }
    }

    /// Open a bare Program NCA, such as a system applet.
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

    /// Mount this title's RomFS, streamed off disk, if it has one.
    pub fn mount_romfs(&self, cpu: &mut Cpu) {
        match self.romfs_source() {
            Some(Ok(romfs)) => cpu.set_romfs_source(Box::new(romfs)),
            Some(Err(e)) => eprintln!("this title's RomFS could not be opened: {e}"),
            None => {}
        }
    }

    /// The RomFS source [`Title::mount_romfs`] uses; `None` if the NCA has no
    /// RomFS section.
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

    /// Open this title's RomFS and read its metadata tables, or exit.
    pub fn romfs(&self, usage_line: &str) -> (Box<dyn ByteSource>, romfs::Image) {
        let source = match self.romfs_source() {
            Some(Ok(source)) => source,
            Some(Err(e)) => usage(&format!("this title's RomFS could not be opened: {e}")),
            None => usage("this NCA has no RomFS section"),
        };
        match romfs::read(&*source) {
            Ok(image) => (source, image),
            Err(why) => usage(&format!("{why} ({usage_line})")),
        }
    }

    /// Boot the title and return where each module landed.
    pub fn boot(&self, cpu: &mut Cpu) -> Vec<switch_core::nso::LoadedNso> {
        // `DOCKED=1` boots docked (1080p), as the frontend's dock toggle does.
        if env::var("DOCKED").is_ok() {
            cpu.set_operation_mode(switch_core::cpu::OperationMode::Docked);
        }
        cpu.set_system_resource_size(switch_core::npdm::Npdm::system_resource_size_of(
            &self.exefs_pfs0,
            &self.exefs,
        ));
        cpu.set_program_id(self.nca.program_id);
        if let Some(priority) =
            switch_core::npdm::Npdm::main_thread_priority_of(&self.exefs_pfs0, &self.exefs)
        {
            cpu.set_main_thread_priority(priority);
        }
        if let Some(core) =
            switch_core::npdm::Npdm::main_thread_core_of(&self.exefs_pfs0, &self.exefs)
        {
            cpu.set_main_thread_core(core);
        }
        if let Some(mask) = switch_core::npdm::Npdm::core_mask_of(&self.exefs_pfs0, &self.exefs) {
            cpu.set_process_core_mask(mask);
        }
        // Before the boot, which lays out the entry ABI per instruction set.
        if !switch_core::npdm::Npdm::is_64_bit_of(&self.exefs_pfs0, &self.exefs) {
            eprintln!("[npdm] AArch32 title: running the A32 interpreter");
            cpu.set_mode(switch_core::cpu::ExecMode::A32);
        }
        let modules = self.modules();
        let loaded = cpu
            .boot_retail_program(&modules)
            .unwrap_or_else(|e| die(&format!("booting {} modules: {e:?}", modules.len())));
        // After the program id and the modules: booting clears the diagnostics.
        mount_add_on_content(cpu, &self.keys);
        loaded
    }

    /// The container holding the game: the base one when an update is stacked.
    pub fn container(&self) -> &Path {
        match &self.base {
            Some((path, _, _)) => path,
            None => &self.path,
        }
    }

    pub fn control(&self) -> Result<switch_core::control::Control, switch_core::Error> {
        open_control(self.container(), &mut self.keys.clone())
    }
}

/// An update container, named by `UPDATE=<path.nsp>`. Its RomFS reads over
/// the base container's.
pub struct Update {
    pub nca: switch_core::nca::Nca,
    path: std::path::PathBuf,
    program: (u64, u64),
}

impl Update {
    /// The update `UPDATE=<path.nsp>` names, if any. Its title key goes into
    /// `keys`.
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

/// Give the running title the add-on content `DLC=<a.nsp>,<b.nsp>` names,
/// skipping content that belongs to another title.
pub fn mount_add_on_content(cpu: &mut Cpu, keys: &KeySet) {
    let Ok(list) = env::var("DLC") else { return };
    for path in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let path = Path::new(path);
        let src = open_source(path);
        let pfs0 = match switch_core::xci::read_container(&src) {
            Ok(pfs0) => pfs0,
            Err(e) => {
                eprintln!("{} is not a container this reads: {e}", path.display());
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
            // Its own handle: the CPU keeps every archive for the whole run.
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
                            "add-on content {:016x} is not this title's; not mounted",
                            nca.title_id
                        ),
                    }
                }
                Err(e) => println!("add-on content {:016x} unreadable: {e}", nca.title_id),
            }
        }
    }
}

/// The last Program NCA in a container (an update's target), its header, and
/// any bundled ticket's title key added to `keys`.
fn open_program(path: &Path, keys: &mut KeySet) -> ((u64, u64), switch_core::nca::Nca) {
    let src = open_source(path);
    let pfs0 = switch_core::xci::read_container(&src).unwrap_or_else(|e| {
        die(&format!(
            "{} is not a container this reads: {e}",
            path.display()
        ))
    });
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
    // Scene releases bundle the ticket next to the content.
    if let Err(e) = switch_core::ticket::load_bundled_title_key(keys, &nca, &pfs0.files, &src) {
        eprintln!("no title key for {}: {e}", path.display());
    }
    ((offset, size), nca)
}

fn open_source(path: &Path) -> switch_core::source::FileSource {
    switch_core::source::FileSource::open(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {e}", path.display())))
}

/// A fresh window over the Program NCA's bytes, one per reader.
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
