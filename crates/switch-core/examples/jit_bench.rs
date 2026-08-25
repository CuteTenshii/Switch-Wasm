//! The block translator against the interpreter, on real homebrew:
//! `jit_bench <nro> [instructions] [font.ttf]`.
//!
//! Boots the same program twice — once with [`Cpu::set_jit_enabled`] off, once
//! with it on — runs the same number of instructions through both, and reports
//! what each managed per second along with whether the two machines ended up
//! in the same state.
//!
//! `examples/bench.rs` measures one instruction class at a time, which says
//! what a decode costs but not what a program costs: real code is a mix, its
//! blocks are entered thousands of times each, and a good part of a frame is
//! spent in the GPU rather than in the CPU at all. This is the number that
//! decides whether translating was worth it.
mod common;

use std::time::Instant;
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

/// Run up to `want` instructions, in slices, and report the rate. Sliced
/// because that is how the frontend drives a session — a frame's worth per
/// call — and re-entering blocks is part of what is being measured.
fn drive(cpu: &mut Cpu, want: u64) -> (u64, f64) {
    const SLICE: u64 = 1_000_000;
    let start = Instant::now();
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
    (done, done as f64 / start.elapsed().as_secs_f64() / 1e6)
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
            let name = if i == 31 { String::from("sp") } else { format!("x{i}") };
            differs(format!("{name} {:#x} vs {:#x}", a.read_x(i), b.read_x(i)));
        }
    }
    for i in 0..32u8 {
        if a.read_vreg(i) != b.read_vreg(i) {
            differs(format!("v{i} {:#x} vs {:#x}", a.read_vreg(i), b.read_vreg(i)));
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
        differs(format!("frames presented {} vs {}", a.nv.gpu.frames, b.nv.gpu.frames));
    }
    if a.nv.gpu.framebuffer.pixels != b.nv.gpu.framebuffer.pixels {
        let differing = a
            .nv
            .gpu
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
    let nro = common::read(common::arg(1, "jit_bench <path.nro> [instructions] [font.ttf]"));
    let want = common::opt_num(2).unwrap_or(40_000_000);
    let font = std::fs::read(common::opt_arg(3).as_deref().unwrap_or(common::FALLBACK_FONT)).ok();

    let mut interpreted = boot(&nro, &font, false);
    let mut translated = boot(&nro, &font, true);
    let (steps_i, rate_i) = drive(&mut interpreted, want);
    let (steps_t, rate_t) = drive(&mut translated, want);

    println!("interpreted  {steps_i:>10} steps  {rate_i:>6.1} M/s");
    println!("translated   {steps_t:>10} steps  {rate_t:>6.1} M/s  ({:.2}x)", rate_t / rate_i);
    let stats = translated.jit_stats();
    println!(
        "  {} blocks, {} translated, {} entered ({:.0}x each), {} invalidated",
        stats.blocks,
        stats.translated,
        stats.executed,
        if stats.translated == 0 { 0.0 } else { stats.executed as f64 / stats.translated as f64 },
        stats.invalidated,
    );

    if steps_i != steps_t {
        println!("  MISMATCH: {steps_i} instructions interpreted, {steps_t} translated");
    }
    match compare(&interpreted, &translated) {
        0 => println!("  the two machines agree"),
        n => println!("  {n} differences"),
    }
}
