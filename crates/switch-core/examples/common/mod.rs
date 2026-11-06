//! Scaffolding shared by the examples: flag handling, key loading,
//! booting, driving the machine, and writing a frame out.
// Every example compiles its own copy of this module and uses only part of it.
#![allow(dead_code)]

/// The guest-state layout emitted blocks are recorded against.
pub mod emit;
/// Sampling where a run spends its instructions.
pub mod profile;
/// A module's dynamic symbol table.
pub mod symbols;

use clap::{Args as ClapArgs, Parser};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;
use switch_core::cpu::Cpu;
use switch_core::gpu::Framebuffer;
use switch_core::keys::KeySet;
use switch_core::source::ByteSource;
use switch_core::trace::Trace;

/// The font `pl:u` serves when no firmware fonts are registered.
const FALLBACK_FONT: &[u8] = include_bytes!("../../../../web/font.ttf");

// The flags a tool that boots a machine takes: what a run traces and dumps, and how it is driven.
#[derive(ClapArgs, Clone, Default)]
pub struct Shared {
    /// Trace channels to switch on (`svc`, `ipc`, ...).
    #[arg(long, value_delimiter = ',', value_parser = channel)]
    trace: Vec<Trace>,
    /// Run the interpreter only.
    #[arg(long)]
    no_jit: bool,
    /// Present as fast as the guest asks instead of pacing to 60 Hz.
    #[arg(long)]
    no_vsync_throttle: bool,
    /// Write every decoded texture into this directory.
    #[arg(long, value_name = "DIR")]
    dump_tex: Option<String>,
    /// Write every translated shader's WGSL into this directory.
    #[arg(long, value_name = "DIR")]
    dump_wgsl: Option<String>,
    /// Write the WGSL the device was handed into this directory.
    #[arg(long, value_name = "DIR")]
    dump_device_wgsl: Option<String>,
    /// Boot docked (1080p).
    #[arg(long)]
    docked: bool,
    /// Register the system data archives of this firmware directory.
    #[arg(long, value_name = "DIR")]
    firmware: Option<String>,
    /// An update container whose RomFS reads over the base title's.
    #[arg(long, value_name = "NSP")]
    update: Option<String>,
    /// Add-on content containers to mount.
    #[arg(long, value_delimiter = ',', value_name = "NSP")]
    dlc: Vec<String>,
    /// Tap buttons every interval steps, e.g. `A+DOWN@300000000`.
    #[arg(long, value_parser = Press::parse, value_name = "BUTTONS@INTERVAL")]
    press: Option<Press>,
    /// Present frames without drawing them.
    #[arg(long)]
    no_draw: bool,
    /// Extra guest cores, each on a host thread.
    #[arg(long, default_value_t = 0, value_name = "N")]
    cores: u64,
    /// Print each frame's wall-clock cost.
    #[arg(long)]
    frame_times: bool,
    /// Report the pc and call stack of the first writes into a region.
    #[arg(long, value_parser = span, value_name = "ADDR:SIZE")]
    trap_write: Option<(u32, u32)>,
    /// Count every distinct pc that reads a region.
    #[arg(long, value_parser = span, value_name = "ADDR:SIZE")]
    trap_read: Option<(u32, u32)>,
    /// Trap writes of zero too.
    #[arg(long)]
    trap_zero: bool,
    /// Print the last N trapped writes when the run stops.
    #[arg(long, default_value_t = 0, value_name = "N")]
    trap_last: usize,
    /// Report argument registers and the call stack the first few times execution reaches these addresses.
    #[arg(long, value_delimiter = ',', value_parser = hex32, value_name = "ADDR")]
    watch_pc: Vec<u32>,
    /// Print every register at each `--watch-pc` hit.
    #[arg(long)]
    watch_regs: bool,
    /// Hex-dump memory (`--dump` syntax) at each `--watch-pc` hit.
    #[arg(long, value_name = "SPEC")]
    watch_dump: Option<String>,
    /// Print the last N `--watch-pc` hits when the run stops.
    #[arg(long, default_value_t = 0, value_name = "N")]
    watch_last: usize,
    /// Hex-dump memory where the run stopped: `<base>[+<hex>][:<hex length>][,...]`, where
    /// `<base>` is `x0`..`x30`, `sp`, `pc` or an address and a leading `*` follows the pointer
    /// stored there.
    #[arg(long, value_name = "SPEC")]
    dump: Option<String>,
}

static SHARED: OnceLock<Shared> = OnceLock::new();

fn shared() -> &'static Shared {
    SHARED.get_or_init(Shared::default)
}

/// Parse the command line, `shared_of` naming where the tool flattened [`Shared`], and hand the
/// core what the flags ask for: the trace channels and the dump directories. The core itself
/// reads no environment.
pub fn parse<C: Parser>(shared_of: impl FnOnce(&C) -> &Shared) -> C {
    let cli = C::parse();
    let flags = shared_of(&cli).clone();
    let mut channels = flags.trace.clone();
    if flags.dump_wgsl.is_some() && !channels.contains(&Trace::Wgsl) {
        channels.push(Trace::Wgsl);
    }
    switch_core::trace::set_mask(&channels);
    switch_core::dump::install(Box::new(DumpDirs {
        texture: flags.dump_tex.clone(),
        raster_wgsl: flags.dump_wgsl.clone(),
        device_wgsl: flags.dump_device_wgsl.clone(),
    }));
    SHARED.set(flags).ok();
    cli
}

fn channel(name: &str) -> Result<Trace, String> {
    Trace::from_name(&format!("TRACE_{}", name.to_ascii_uppercase()))
        .ok_or_else(|| format!("no trace channel named {name:?}"))
}

/// A machine with the `--no-jit` and `--no-vsync-throttle` flags applied.
pub fn new_cpu() -> Cpu {
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(!shared().no_jit);
    cpu.set_vsync_throttle(!shared().no_vsync_throttle);
    cpu
}

/// Where each dumped artefact is written.
struct DumpDirs {
    texture: Option<String>,
    raster_wgsl: Option<String>,
    device_wgsl: Option<String>,
}

impl DumpDirs {
    fn dir(&self, what: switch_core::dump::Artefact) -> Option<&str> {
        use switch_core::dump::Artefact;
        match what {
            Artefact::Texture => self.texture.as_deref(),
            Artefact::RasterWgsl => self.raster_wgsl.as_deref(),
            Artefact::DeviceWgsl => self.device_wgsl.as_deref(),
        }
    }
}

impl switch_core::dump::Sink for DumpDirs {
    fn wants(&self, what: switch_core::dump::Artefact) -> bool {
        self.dir(what).is_some()
    }

    fn keep(&self, what: switch_core::dump::Artefact, name: &str, bytes: &[u8]) {
        let Some(dir) = self.dir(what) else {
            return;
        };
        let path = Path::new(dir).join(name);
        if let Err(e) = fs::create_dir_all(dir).and_then(|()| fs::write(&path, bytes)) {
            eprintln!("cannot write {}: {e}", path.display());
        }
    }
}

/// A hexadecimal number, `0x` optional.
pub fn parse_hex(text: &str) -> Option<u64> {
    u64::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok()
}

/// A hexadecimal number that fits a guest address.
pub fn parse_hex32(text: &str) -> Option<u32> {
    u32::try_from(parse_hex(text)?).ok()
}

/// A flag value that is a hexadecimal number.
pub fn hex64(text: &str) -> Result<u64, String> {
    parse_hex(text).ok_or_else(|| format!("{text:?} is not a hexadecimal number"))
}

/// A flag value that is a hexadecimal guest address.
pub fn hex32(text: &str) -> Result<u32, String> {
    parse_hex32(text).ok_or_else(|| format!("{text:?} is not a hexadecimal address"))
}

/// A flag value that is `<addr>:<hex size>`, as `(addr, addr + size)`.
pub fn span(text: &str) -> Result<(u32, u32), String> {
    parse_span(text).ok_or_else(|| format!("{text:?} is not <hex addr>:<hex size>"))
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

// What a tool runs: an `.nro`, or a retail container and the keys to open it.
#[derive(ClapArgs)]
pub struct Target {
    /// An `.nro`, or a retail container (`.nsp`, `.xci` or a program `.nca`).
    pub target: String,
    /// `prod.keys`; a retail container needs it.
    #[arg(long, value_name = "FILE")]
    keys: Option<String>,
    /// `title.keys`.
    #[arg(long, value_name = "FILE")]
    title_keys: Option<String>,
    /// A TTF/OTF to serve as the shared font instead of the built-in one.
    #[arg(long, value_name = "TTF")]
    font: Option<String>,
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
    /// System data archives registered from `--firmware`.
    pub archives: usize,
    /// The save quota from the title's NACP; `None` for homebrew or without control data.
    pub quota: Option<switch_core::cpu::SaveDataQuota>,
}

impl Program {
    pub fn is_homebrew(&self) -> bool {
        matches!(self.form, Form::Homebrew(_))
    }

    /// Boot into an already bootstrapped `cpu`.
    pub fn boot(&self, cpu: &mut Cpu) -> Booted {
        match &self.form {
            Form::Retail(title) => title.boot_with_font(cpu, self.font.as_deref()),
            Form::Homebrew(image) => {
                apply_docked(cpu);
                load_font(cpu, self.font.as_deref());
                cpu.boot_homebrew(image, None)
                    .unwrap_or_else(|e| die(&format!("booting the NRO: {e:?}")));
                Booted {
                    modules: Vec::new(),
                    archives: 0,
                    quota: None,
                }
            }
        }
    }
}

/// `--docked` boots docked (1080p), as the frontend's dock toggle does.
fn apply_docked(cpu: &mut Cpu) {
    if shared().docked {
        cpu.set_operation_mode(switch_core::cpu::OperationMode::Docked);
    }
}

/// The font named on the command line, or the fallback one.
fn load_font(cpu: &mut Cpu, font: Option<&[u8]>) {
    match font {
        Some(bytes) => cpu.set_shared_font(bytes.to_vec()),
        None => load_fallback_font(cpu),
    }
}

impl Target {
    /// The keys path a retail container needs.
    fn prod(&self) -> &str {
        match &self.keys {
            Some(keys) => keys,
            None => die("a retail container needs --keys <prod.keys>"),
        }
    }

    pub fn open(&self) -> Title {
        Title::open(&self.target, self.prod(), self.title_keys.as_ref())
    }

    pub fn open_program(&self) -> Program {
        let form = match target_kind(Path::new(&self.target)) {
            Kind::Nro => Form::Homebrew(read(&self.target)),
            _ => Form::Retail(Box::new(self.open())),
        };
        Program {
            form,
            font: self.font.as_ref().map(read),
        }
    }

    /// The keys alone, for tools that read the container themselves.
    pub fn keys(&self) -> KeySet {
        keys(self.prod(), self.title_keys.as_ref())
    }
}

/// A renderer that draws nothing, for `--no-draw`: frames still present, so
/// the run times the CPU alone, as with a GPU doing the drawing.
#[derive(Debug)]
struct NoDraw;

impl switch_core::gpu::renderer::Renderer for NoDraw {
    fn draw(
        &mut self,
        _: &switch_core::gpu::engine::threed::Engine3D,
        _: &mut switch_core::gpu::exec::ExecCtx,
    ) -> switch_core::Result<()> {
        Ok(())
    }

    fn clear_color(
        &mut self,
        _: &switch_core::gpu::engine::threed::Engine3D,
        _: &mut switch_core::gpu::exec::ExecCtx,
        _: u32,
        _: u32,
        _: [bool; 4],
    ) -> switch_core::Result<()> {
        Ok(())
    }

    fn clear_depth_stencil(
        &mut self,
        _: &switch_core::gpu::engine::threed::Engine3D,
        _: &mut switch_core::gpu::exec::ExecCtx,
        _: bool,
        _: bool,
    ) -> switch_core::Result<()> {
        Ok(())
    }
}

/// Three frames at 1.02 GHz and 60 Hz, so a once-per-frame pad poll sees the press.
const PRESS_HOLD: u64 = 3 * 17_000_000;

/// In Horizon's `HidNpadButton` order.
const BUTTONS: [(&str, u64); 16] = [
    ("A", 1 << 0),
    ("B", 1 << 1),
    ("X", 1 << 2),
    ("Y", 1 << 3),
    ("LSTICK", 1 << 4),
    ("RSTICK", 1 << 5),
    ("L", 1 << 6),
    ("R", 1 << 7),
    ("ZL", 1 << 8),
    ("ZR", 1 << 9),
    ("PLUS", 1 << 10),
    ("MINUS", 1 << 11),
    ("LEFT", 1 << 12),
    ("UP", 1 << 13),
    ("RIGHT", 1 << 14),
    ("DOWN", 1 << 15),
];

/// `--press <buttons>@<interval>`: tap buttons every `interval` steps, e.g.
/// `--press A+DOWN@300000000`. The first tap waits a whole interval so it does
/// not land during boot.
#[derive(Clone)]
pub struct Press {
    buttons: u64,
    interval: u64,
    held: bool,
}

impl Press {
    /// The `--press` flag, if given.
    pub fn from_flags() -> Option<Press> {
        shared().press.clone()
    }

    fn parse(raw: &str) -> Result<Press, String> {
        let fail =
            |why: &str| format!("{raw}: {why}; expected <buttons>@<interval>, e.g. A@500000000");
        let (names, interval) = raw.split_once('@').ok_or_else(|| fail("no interval"))?;
        let interval: u64 = interval
            .parse()
            .map_err(|_| fail("the interval is not a number"))?;
        if interval <= PRESS_HOLD {
            return Err(fail("the interval must be longer than a press"));
        }
        let mut buttons = 0;
        for name in names.split('+') {
            let upper = name.trim().to_ascii_uppercase();
            match BUTTONS.iter().find(|(button, _)| *button == upper) {
                Some((_, bit)) => buttons |= bit,
                None => return Err(fail(&format!("no button named {name:?}"))),
            }
        }
        Ok(Press {
            buttons,
            interval,
            held: false,
        })
    }

    /// Press or release for the step count `done`; call once per tick.
    pub fn tick(&mut self, cpu: &mut Cpu, done: u64) {
        let down = done >= self.interval && done % self.interval < PRESS_HOLD;
        if down != self.held {
            self.held = down;
            cpu.set_gamepad_state(if down { self.buttons } else { 0 }, 0, 0, 0, 0);
        }
    }
}

/// Swap in [`NoDraw`] for `--no-draw`.
pub fn apply_no_draw(cpu: &mut Cpu) {
    if shared().no_draw {
        cpu.nv.gpu.set_renderer(Box::new(NoDraw));
    }
}

/// `--cores <n>` more guest cores, each served on a host thread, for [`stop_cores`].
pub fn start_cores(cpu: &mut Cpu) -> Vec<std::thread::JoinHandle<()>> {
    (0..shared().cores)
        .map(|_| {
            let slot = cpu.add_core();
            std::thread::spawn(move || switch_core::cpu::serve(slot))
        })
        .collect()
}

pub fn stop_cores(cpu: &mut Cpu, cores: Vec<std::thread::JoinHandle<()>>) {
    cpu.stop_cores();
    for core in cores {
        core.join().expect("a core thread panicked");
    }
}

fn parse_span(raw: &str) -> Option<(u32, u32)> {
    let (lo, len) = raw.split_once(':')?;
    let lo = parse_hex32(lo)?;
    Some((lo, lo.saturating_add(parse_hex32(len)?)))
}

/// Give the guest the fallback shared font; firmware fonts replace it.
pub fn load_fallback_font(cpu: &mut Cpu) {
    cpu.set_shared_font(FALLBACK_FONT.to_vec());
}

/// Register every system data archive in the `--firmware` directory, if one is given, and install
/// its programs as the browser's NAND does, so library applets run. Returns the
/// number of data archives.
pub fn register_firmware(cpu: &mut Cpu, keys: &KeySet) -> usize {
    let Some(dir) = &shared().firmware else {
        return 0;
    };
    let Ok(entries) = fs::read_dir(dir) else {
        eprintln!("--firmware {dir} cannot be read");
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
        if archive.content_type == ContentType::Program {
            install_firmware_program(cpu, keys, &path, archive);
            continue;
        }
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

/// `switch_add_applet`'s install, over a file; a program that does not open is skipped.
fn install_firmware_program(cpu: &mut Cpu, keys: &KeySet, path: &Path, nca: switch_core::nca::Nca) {
    use std::rc::Rc;
    use switch_core::source::FileSource;
    let Some(exefs_index) = nca.exefs_section_index() else {
        return;
    };
    let Ok(exefs) = FileSource::open(path)
        .map_err(|e| switch_core::Error::Io(e.to_string()))
        .and_then(|file| nca.section_source(file, keys, exefs_index))
    else {
        return;
    };
    let romfs = nca.romfs_section_index().and_then(|index| {
        let file = FileSource::open(path).ok()?;
        let romfs = nca.romfs_source(file, keys, index).ok()?;
        Some(Rc::new(romfs) as Rc<dyn ByteSource>)
    });
    cpu.install_program(switch_core::cpu::Installed {
        nca,
        exefs: Rc::new(exefs),
        exefs_index,
        romfs,
    });
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

/// Per-frame wall-clock cost, printed for `--frame-times`.
struct FrameTimes {
    on: bool,
    last_count: u64,
    last_at: Instant,
    deltas: Vec<f64>,
}

impl FrameTimes {
    fn new(cpu: &Cpu) -> FrameTimes {
        FrameTimes {
            on: shared().frame_times,
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
                Ok(report) if report.steps == 0 => {
                    println!("[drive] a slice ran nothing at step {}", run.steps);
                    break;
                }
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

/// Run until frame `frame` is presented; `None`, after saying why, if it never is.
pub fn reach_frame(cpu: &mut Cpu, frame: u64, budget: u64) -> Option<Run> {
    let run = run_to(cpu, budget, |cpu| cpu.nv.gpu.frames >= frame);
    if cpu.nv.gpu.frames >= frame {
        return Some(run);
    }
    println!(
        "never reached frame {frame}: stopped at {} after {} steps",
        cpu.nv.gpu.frames, run.steps
    );
    report(cpu, &run);
    None
}

/// The guest's console output, up to `lines` lines.
pub fn print_console(cpu: &Cpu, lines: usize) {
    println!("--- program console output ({} bytes) ---", cpu.out.len());
    for line in String::from_utf8_lossy(&cpu.out).lines().take(lines) {
        println!("  {line}");
    }
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
    if cpu.other_core_steps() > 0 {
        println!(
            "other cores retired {} of {} steps",
            cpu.other_core_steps(),
            cpu.steps
        );
    }
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

/// Where one `--dump` entry starts: a register or an absolute address.
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
    spec.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let (addr, len) = match entry.split_once(':') {
                Some((addr, len)) => (addr, parse_hex(len)? as u32),
                None => (entry, DEFAULT_LEN),
            };
            // Search from the right so `x23+0x10` splits on the sign.
            let (base, offset) = match addr.rfind(['+', '-']).filter(|&i| i > 0) {
                Some(i) => {
                    let value = parse_hex(&addr[i + 1..])? as i64;
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
                absolute => DumpBase::Absolute(parse_hex(absolute)? as u32),
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

/// The debugging flags of [`Shared`]: `--trap-write`, `--trap-read`, `--trap-last`, `--trap-zero`,
/// `--watch-pc`, `--watch-last`, `--watch-dump`, `--watch-regs` and `--dump`.
///
/// - `--trap-write`: pc and call stack of the first writes into a region.
/// - `--trap-read`: every distinct pc that reads a region, counted.
/// - `--watch-pc`: argument registers and call stack the first few times execution reaches an
///   address.
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
    pub fn from_flags() -> Debug {
        let flags = shared();
        Debug {
            write_trap: flags.trap_write,
            read_trap: flags.trap_read,
            trap_zero: flags.trap_zero,
            trap_last: flags.trap_last,
            last_traps: std::collections::VecDeque::new(),
            watch_pc: flags.watch_pc.clone(),
            watch_regs: flags.watch_regs,
            watch_dumps: flags
                .watch_dump
                .as_deref()
                .map(parse_dump_specs)
                .unwrap_or_default(),
            watch_last: flags.watch_last,
            last_watches: std::collections::VecDeque::new(),
            dumps: flags
                .dump
                .as_deref()
                .map(parse_dump_specs)
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

    /// [`Pace::Instructions`] when a knob or the caller's own per-instruction
    /// work needs it; otherwise the block translator, as in the frontend.
    pub fn pace(&self, caller_stepwise: bool) -> Pace {
        if caller_stepwise || self.stepwise() {
            Pace::Instructions
        } else {
            Pace::Blocks
        }
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

    /// Print the registers, call stack, and `--dump` regions where the run stopped.
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
    fn open_container(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        let mut keys = keys(prod, title);
        let (program, nca) = open_program(&path, &mut keys);
        // `--update <path.nsp>` boots the update's modules over this container's RomFS.
        match Update::from_env(&mut keys) {
            Some(update) => {
                let base = Some((path, program, nca));
                Title::finish(update.path, update.program, update.nca, keys, base)
            }
            None => Title::finish(path, program, nca, keys, None),
        }
    }

    /// Open a bare Program NCA, such as a system applet.
    fn open_nca(
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
    fn mount_romfs(&self, cpu: &mut Cpu) {
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
    pub fn romfs(&self) -> (Box<dyn ByteSource>, switch_core::romfs::RomFsIndex) {
        let source = match self.romfs_source() {
            Some(Ok(source)) => source,
            Some(Err(e)) => die(&format!("this title's RomFS could not be opened: {e}")),
            None => die("this NCA has no RomFS section"),
        };
        match switch_core::romfs::RomFsIndex::read(&*source) {
            Ok(index) => (source, index),
            Err(e) => die(&format!(
                "this title's RomFS metadata could not be read: {e}"
            )),
        }
    }

    /// Boot the title as the frontend does: save quota, RomFS, font, system
    /// archives, modules, add-on content.
    pub fn boot(&self, cpu: &mut Cpu) -> Booted {
        self.boot_with_font(cpu, None)
    }

    fn boot_with_font(&self, cpu: &mut Cpu, font: Option<&[u8]>) -> Booted {
        let quota = match self.control() {
            Ok(control) => {
                let quota = switch_core::cpu::SaveDataQuota::from(&control.nacp);
                cpu.set_save_data_quota(quota);
                cpu.set_add_on_content_base_id(control.nacp.add_on_content_base_id);
                Some(quota)
            }
            Err(e) => {
                eprintln!("no control data ({e}): using default save sizes");
                None
            }
        };
        self.mount_romfs(cpu);
        load_font(cpu, font);
        // After the font (firmware fonts override it), before the modules.
        let archives = register_firmware(cpu, &self.keys);
        Booted {
            modules: self.boot_modules(cpu),
            archives,
            quota,
        }
    }

    fn boot_modules(&self, cpu: &mut Cpu) -> Vec<switch_core::nso::LoadedNso> {
        apply_docked(cpu);
        let program = switch_core::cpu::Program::parse(self.nca.program_id, self.exefs.clone())
            .unwrap_or_else(|e| die(&format!("the ExeFS is not a PFS0: {e}")));
        let loaded = cpu
            .boot_program(&program)
            .unwrap_or_else(|e| die(&format!("booting the program: {e:?}")));
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

/// An update container, named by `--update <path.nsp>`. Its RomFS reads over
/// the base container's.
struct Update {
    nca: switch_core::nca::Nca,
    path: std::path::PathBuf,
    program: (u64, u64),
}

impl Update {
    /// The update `--update <path.nsp>` names, if any. Its title key goes into
    /// `keys`.
    fn from_env(keys: &mut KeySet) -> Option<Update> {
        let path = std::path::PathBuf::from(shared().update.as_ref()?);
        let (program, nca) = open_program(&path, keys);
        if !nca.is_update() {
            die(&format!(
                "{} is not an update: its RomFS is a title's own, not a patch over one",
                path.display()
            ));
        }
        Some(Update { nca, path, program })
    }
}

/// Give the running title the add-on content `--dlc <a.nsp>,<b.nsp>` names,
/// skipping content that belongs to another title.
pub fn mount_add_on_content(cpu: &mut Cpu, keys: &KeySet) {
    for path in &shared().dlc {
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
