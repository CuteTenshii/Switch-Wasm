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
