//! The block translator against the interpreter, on real homebrew:
//! `jit_difftest <nro> [instructions] [font.ttf]`.
//!
//! Boots the same program twice, once with [`Cpu::set_jit_enabled`] off, once
//! with it on: runs the same number of instructions through both, and reports
//! every way the two machines ended up disagreeing.
//!
//! This is a correctness tool, not a benchmark, and it used to be both. The
//! benchmark half reported what each engine managed per second *on this host*,
//! and the browser is what this project runs in: a ratio between two engines
//! measured through rustc's x86-64 backend is not the ratio between them under
//! a browser's, which recompiles the same wasm with its own register
//! allocator, its own inlining and a bounds check on every guest load. That
//! number belongs to `tools/wasm_bench.mjs`, which runs the artefact the
//! browser runs.
//!
//! What survives here is the part that was never about speed. The translator
//! resolves at translation time what the interpreter re-derives per execution,
//! so the two have to agree on every register, the flags, memory, the console
//! and the framebuffer, and if they ever do not, the translated run is wrong
//! however fast it was. The interpreter is the reference.
//!
//! The work counters below say the same thing the timing tried to, without a
//! clock: how many blocks were translated, how often each was re-entered, and
//! how much of the run the translator handed straight back to the interpreter
//! because it had no op for it. Those numbers are identical on every target.
mod common;

use switch_core::cpu::Cpu;

fn boot(nro: &[u8], font: &Option<Vec<u8>>, jit: bool) -> Cpu {
    let mut cpu = Cpu::new();
    cpu.set_jit_enabled(jit);
    cpu.bootstrap();
    if let Some(font) = font {
        cpu.set_shared_font(font.clone());
    }
    cpu.boot_homebrew(nro).expect("boot");
    cpu
}

/// Run up to `want` instructions, in slices, and report how many retired.
///
/// Sliced because that is how the frontend drives a session, a frame's worth
/// per call, and re-entering blocks across those calls is part of what is
/// being checked.
fn drive(cpu: &mut Cpu, want: u64) -> u64 {
    const SLICE: u64 = 1_000_000;
    let mut done = 0u64;
    while done < want && !cpu.halted {
        match cpu.run(SLICE.min(want - done)) {
            Ok(report) => done += report.steps,
            Err(e) => {
                println!("FAULT after {done}: {e}");
                break;
            }
        }
    }
    done
}

/// Report every difference between the two machines rather than only the
/// first, since a divergence in one register usually shows up in several.
fn compare(a: &Cpu, b: &Cpu) -> usize {
    let mut bad = 0;
    let mut differs = |what: String| {
        println!("  MISMATCH: {what}");
        bad += 1;
    };
    for i in 0..32u8 {
        if a.read_x(i) != b.read_x(i) {
            let name = if i == 31 {
                String::from("sp")
            } else {
                format!("x{i}")
            };
            differs(format!("{name} {:#x} vs {:#x}", a.read_x(i), b.read_x(i)));
        }
    }
    for i in 0..32u8 {
        if a.read_vreg(i) != b.read_vreg(i) {
            differs(format!(
                "v{i} {:#x} vs {:#x}",
                a.read_vreg(i),
                b.read_vreg(i)
            ));
        }
    }
    if a.get_pc() != b.get_pc() {
        differs(format!("pc {:#x} vs {:#x}", a.get_pc(), b.get_pc()));
    }
    if a.nzcv() != b.nzcv() {
        differs(format!("nzcv {:#x} vs {:#x}", a.nzcv(), b.nzcv()));
    }
    if a.cycles != b.cycles {
        differs(format!("cycles {} vs {}", a.cycles, b.cycles));
    }
    if a.halted != b.halted {
        differs(format!("halted {} vs {}", a.halted, b.halted));
    }
    if a.out != b.out {
        differs(String::from("console output"));
    }
    if a.nv.gpu.frames != b.nv.gpu.frames {
        differs(format!(
            "frames presented {} vs {}",
            a.nv.gpu.frames, b.nv.gpu.frames
        ));
    }
    if a.nv.gpu.framebuffer.pixels != b.nv.gpu.framebuffer.pixels {
        let differing =
            a.nv.gpu
                .framebuffer
                .pixels
                .iter()
                .zip(&b.nv.gpu.framebuffer.pixels)
                .filter(|(x, y)| x != y)
                .count();
        differs(format!("{differing} framebuffer pixels"));
    }
    bad
}

fn main() {
    let nro = common::read(common::arg(
        1,
        "jit_difftest <path.nro> [instructions] [font.ttf]",
    ));
    let want = common::opt_num(2).unwrap_or(40_000_000);
    let font = std::fs::read(
        common::opt_arg(3)
            .as_deref()
            .unwrap_or(common::FALLBACK_FONT),
    )
    .ok();

    let mut interpreted = boot(&nro, &font, false);
    let mut translated = boot(&nro, &font, true);
    let steps_i = drive(&mut interpreted, want);
    let steps_t = drive(&mut translated, want);

    println!("interpreted  {steps_i:>10} steps");
    println!("translated   {steps_t:>10} steps");

    let stats = translated.jit_stats();
    let per = |n: u64, of: u64| if of == 0 { 0.0 } else { n as f64 / of as f64 };
    println!(
        "  {} blocks, {} translated, {} entered ({:.0}x each), {} invalidated",
        stats.blocks,
        stats.translated,
        stats.executed,
        per(stats.executed, stats.translated),
        stats.invalidated,
    );
    // The share of the run the translator did not translate. Every one of
    // these re-derived the instruction's group, form and fields on the way
    // past, which is the work the translator exists to do once, so this is
    // where the next block of speed is, and it reads the same under any
    // compiler on any target.
    println!(
        "  {} instructions fell back to the interpreter ({:.2}% of the run)",
        stats.interpreted,
        per(stats.interpreted, steps_t) * 100.0,
    );

    if steps_i != steps_t {
        println!("  MISMATCH: {steps_i} instructions interpreted, {steps_t} translated");
    }
    match compare(&interpreted, &translated) {
        0 => println!("  the two machines agree"),
        n => println!("  {n} differences"),
    }
}
