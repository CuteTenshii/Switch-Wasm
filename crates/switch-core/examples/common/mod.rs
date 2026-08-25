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
use switch_core::cpu::Cpu;
use switch_core::gpu::Framebuffer;
use switch_core::keys::KeySet;

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

/// Run the machine until `tick` says to stop, it halts, it faults, or `budget`
/// instructions have retired.
///
/// `tick` is called before each instruction under [`Pace::Instructions`] and
/// before each slice under [`Pace::Blocks`], with the instructions retired so
/// far. Returning [`Flow::Stop`] ends the run.
pub fn drive(
    cpu: &mut Cpu,
    pace: Pace,
    budget: u64,
    mut tick: impl FnMut(&mut Cpu, u64) -> Flow,
) -> Run {
    let mut run = Run::default();
    while run.steps < budget && !cpu.halted {
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
}

impl Title {
    /// Open the Program NCA inside an NSP, resolving its title key from a
    /// bundled ticket if the container has one.
    pub fn open_nsp(
        container: impl AsRef<Path>,
        prod: impl AsRef<Path>,
        title: Option<impl AsRef<Path>>,
    ) -> Title {
        let path = container.as_ref().to_path_buf();
        let src = open_source(&path);
        let pfs0 = switch_core::nsp::Pfs0::read_from(&src)
            .unwrap_or_else(|e| die(&format!("{} is not a PFS0: {e}", path.display())));
        let mut keys = keys(prod, title);

        // The last Program NCA rather than the first: a container that
        // carries more than one is an update, and the later entry is the one
        // it updates to. `boot_nsp` picks the same one.
        let mut found = None;
        for (index, file) in pfs0.files.iter().enumerate() {
            if !file.name.to_ascii_lowercase().ends_with(".nca") {
                continue;
            }
            let Ok(window) = pfs0.file_source(&src, index) else { continue };
            match switch_core::nca::Nca::parse_source(&window, Some(&keys)) {
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
        let nca = switch_core::nca::Nca::parse_source(&window, Some(&keys))
            .unwrap_or_else(|e| die(&format!("parsing the program nca: {e}")));
        // Title-key crypto needs the key itself from somewhere. Scene
        // releases bundle the ticket next to the content.
        if nca.has_rights_id() && keys.resolved_title_key(&nca.rights_id).is_none() {
            match switch_core::ticket::find_and_decrypt_title_key_from(
                &nca.rights_id,
                &pfs0.files,
                &src,
                &keys,
            ) {
                Ok(key) => keys.add_resolved_title_key(nca.rights_id, key),
                Err(e) => eprintln!("no title key for this container: {e}"),
            }
        }
        Title::finish(path, offset, size, nca, keys)
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
        Title::finish(path, 0, size, nca, keys)
    }

    /// Read the ExeFS, which is the one part of a container worth holding.
    fn finish(
        path: std::path::PathBuf,
        offset: u64,
        size: u64,
        nca: switch_core::nca::Nca,
        keys: KeySet,
    ) -> Title {
        let index = nca
            .exefs_section_index()
            .unwrap_or_else(|| die(&format!("{} has no ExeFS section", path.display())));
        let exefs = nca
            .read_pfs0_section(program_window(&path, offset, size), &keys, index)
            .unwrap_or_else(|e| die(&format!("reading the ExeFS: {e}")));
        let exefs_pfs0 = switch_core::nsp::Pfs0::parse(&exefs)
            .unwrap_or_else(|e| die(&format!("the ExeFS is not a PFS0: {e}")));
        Title { nca, keys, exefs, exefs_pfs0, path, program: (offset, size) }
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
    /// thing from one that would not open.
    pub fn romfs_source(
        &self,
    ) -> Option<Result<
        switch_core::source::Window<
            switch_core::nca::SectionSource<
                switch_core::source::Window<switch_core::source::FileSource>,
            >,
        >,
        switch_core::Error,
    >> {
        let index = self.nca.romfs_section_index()?;
        let window = program_window(&self.path, self.program.0, self.program.1);
        Some(self.nca.romfs_source(window, &self.keys, index))
    }

    /// Boot the title: its address space, its program id, and its modules.
    ///
    /// The system resource size comes from the title's own manifest and has
    /// to be set before the modules load, since `nn::init` reads the
    /// resulting figures as soon as it runs.
    pub fn boot(&self, cpu: &mut Cpu) {
        cpu.set_system_resource_size(switch_core::npdm::Npdm::system_resource_size_of(
            &self.exefs_pfs0,
            &self.exefs,
        ));
        cpu.set_program_id(self.nca.program_id);
        let modules = self.modules();
        if let Err(e) = cpu.boot_retail_program(&modules) {
            die(&format!("booting {} modules: {e:?}", modules.len()))
        }
    }
}

/// A handle on the container, or exit saying which one could not be opened.
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
