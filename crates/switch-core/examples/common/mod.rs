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
                set.title_keys =
                    switch_core::keys::keyset_from_title(&switch_core::keys::parse_keys_file(&text));
            }
            Err(e) => eprintln!("cannot read {}: {e} (continuing without title keys)", title.display()),
        }
    }
    set
}

/// A `u64` from the environment, or `default`.
pub fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
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
    let Ok(dir) = env::var("SWITCH_FIRMWARE") else { return 0 };
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
        let Ok(src) = switch_core::source::FileSource::open(&path) else { continue };
        let Ok(archive) = switch_core::nca::Nca::parse_source(&src, Some(keys)) else { continue };
        use switch_core::nca::ContentType;
        if !matches!(archive.content_type, ContentType::Data | ContentType::PublicData) {
            continue;
        }
        let Some(section) = archive.romfs_section_index() else { continue };
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
        self.deltas.extend(std::iter::repeat_n(each, presented as usize));
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
        println!("[fault] at step {} pc={:#x}: {fault}", run.steps, cpu.get_pc());
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

/// The order the modules of an ExeFS load in.
///
/// One list, because three examples carried their own and two of them
/// stopped at `subsdk4` — a title with a `subsdk5` booted differently
/// depending on which tool you ran it with.
const MODULE_ORDER: &[&str] = &[
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

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
    /// Open the Program NCA inside an NSP, resolving its title key from a
    /// bundled ticket if the container has one.
    /// Open whichever kind of container this is, by looking at it.
    ///
    /// A tool that only takes one kind is a tool that cannot be pointed at the
    /// title you want to debug: `screenshot_gpu` took a bare NCA, so the
    /// backend it exists to measure could not be run against a retail game at
    /// all. The web frontend already decides this by header rather than by
    /// name (`web/main/filetype.ts`); this is the same rule.
    ///
    /// `PFS0` sits at offset 0 and the NCA magic at 0x200. An NCA straight off
    /// the CDN keeps its header encrypted, so its magic is invisible until
    /// `prod.keys` decrypts it — those fall back to the extension, which is
    /// what the frontend does with them too.
    pub fn open(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        let mut head = [0u8; 0x204];
        // The whole window or nothing: a short read would leave the magic
        // positions as zeros and quietly look like "not an NCA".
        let read = {
            let src = open_source(&path);
            switch_core::source::ByteSource::read_exact_at(&src, 0, &mut head).is_ok()
        };
        let is_pfs0 = read && &head[..4] == b"PFS0";
        let is_nca = read
            && matches!(&head[0x200..0x204], b"NCA3" | b"NCA2" | b"NCA0");
        let by_name = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("nca"));
        if is_pfs0 {
            Title::open_nsp(path, prod, title)
        } else if is_nca || by_name {
            Title::open_nca(path, prod, title)
        } else {
            Title::open_nsp(path, prod, title)
        }
    }

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
        Title { nca, keys, exefs, exefs_pfs0, path, program, base }
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
    ///
    /// The system resource size comes from the title's own manifest and has
    /// to be set before the modules load, since `nn::init` reads the
    /// resulting figures as soon as it runs.
    pub fn boot(&self, cpu: &mut Cpu) {
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
        if let Err(e) = cpu.boot_retail_program(&modules) {
            die(&format!("booting {} modules: {e:?}", modules.len()))
        }
        // After the program id, which is what add-on content is numbered
        // against, and after the modules, which is where the browser has to
        // do it — booting clears the diagnostics it reads them through.
        mount_add_on_content(cpu, &self.keys);
    }
}

/// A handle on the container, or exit saying which one could not be opened.
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
            let Ok(window) = pfs0.file_source(&src, index) else { continue };
            let Ok(nca) = switch_core::nca::Nca::parse_source(&window, Some(&keys)) else {
                continue;
            };
            use switch_core::nca::ContentType;
            if !matches!(nca.content_type, ContentType::Data | ContentType::PublicData) {
                continue;
            }
            if let Err(e) =
                switch_core::ticket::load_bundled_title_key(&mut keys, &nca, &pfs0.files, &src)
            {
                eprintln!("no title key for {}: {e}", file.name);
            }
            let Some(section) = nca.romfs_section_index() else { continue };
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
        let Ok(window) = pfs0.file_source(&src, index) else { continue };
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
