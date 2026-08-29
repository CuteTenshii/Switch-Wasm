//! Which instructions a program actually runs that the block translator has
//! no op for:
//! `jit_coverage <target> [prod.keys] [title.keys] [font.ttf]`.
//!
//! The translator resolves an instruction's group, form, fields and immediates
//! once, at translation time, and every later execution of that block does
//! none of it. An encoding it has no op for becomes `Op::Interpret` instead:
//! the block still runs, but that instruction is decoded from scratch on every
//! pass, which is the work the translator exists to remove. In a loop body
//! entered ten thousand times, one such instruction costs ten thousand
//! decodes.
//!
//! This is what `examples/bench.rs` was for, and it answered the question the
//! wrong way round. It ran sixteen copies of one encoding in a loop, timed it
//! on the host, and called the class untranslated when the two engines came
//! out "within noise of each other" — inferring an exact, static property of
//! the decoder from a wall-clock measurement on a machine this emulator does
//! not run on. [`switch_core::cpu::translates`] answers it directly, and
//! weighting by a real frame's instruction mix says which of the gaps is worth
//! anything.
//!
//! Counted per instruction, so this half necessarily runs the interpreter. The
//! mix does not depend on which engine produced it — `examples/jit_difftest.rs`
//! is the check that the two execute the same instructions. It measures the
//! same property the `interpreted` counter does, over one steady frame rather
//! than over whatever window that run was given, so the two shares only match
//! when the windows do.
//!
//! The target is a homebrew `.nro` or a retail container — an `.nsp`, an
//! `.xci` or a bare Program `.nca`, which needs its keys after it. An NRO is
//! not the workload a retail title is, so a ranking taken from one does not
//! transfer.
mod common;

const USAGE: &str = "jit_coverage <target> [prod.keys] [title.keys] [font.ttf]";

use common::{Flow, Pace};
use std::collections::HashMap;
use switch_core::cpu::Cpu;
use switch_core::disasm::disassemble;

/// How many distinct untranslated encodings to name.
const ROWS: usize = 20;

fn main() {
    let args = common::program_args(USAGE);
    let program = args.open_program();
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    program.boot(&mut cpu);

    // Two frames of startup, so what follows is a steady-state frame. Nothing
    // is sampled here, so it runs through the block translator.
    common::run_to(&mut cpu, u64::MAX, |cpu| cpu.nv.gpu.frames >= 2);

    let mut seen: HashMap<u32, u64> = HashMap::new();
    let start = cpu.nv.gpu.frames;
    let run = common::drive(&mut cpu, Pace::Instructions, u64::MAX, |cpu, _| {
        if cpu.nv.gpu.frames != start {
            return Flow::Stop;
        }
        let pc = cpu.get_pc();
        if let Ok(insn) = cpu.mem.read_u32(pc) {
            *seen.entry(insn).or_default() += 1;
        }
        Flow::Continue
    });

    let total = run.steps;
    let mut missing: Vec<(u64, u32)> = seen
        .iter()
        .filter(|(&insn, _)| !switch_core::cpu::translates(insn))
        .map(|(&insn, &count)| (count, insn))
        .collect();
    let fallbacks: u64 = missing.iter().map(|(count, _)| count).sum();

    println!(
        "one frame = {total} instructions, {} distinct encodings",
        seen.len()
    );
    println!(
        "{fallbacks} of them ({:.2}%) fall back to the interpreter, over {} distinct encodings",
        fallbacks as f64 * 100.0 / total as f64,
        missing.len(),
    );

    missing.sort_by_key(|&(count, insn)| (std::cmp::Reverse(count), insn));
    println!("--- hottest untranslated encodings ---");
    for (count, insn) in missing.iter().take(ROWS) {
        println!(
            "  {insn:#010x}  {count:>10}  {:5.2}%  {}",
            *count as f64 * 100.0 / total as f64,
            disassemble(*insn),
        );
    }

    // Rolled up the way `examples/hotspots.rs` reports the mix, so the two
    // tables can be read against each other: hot group with a large share
    // untranslated is where a new op pays for itself.
    let mut by_top = [(0u64, 0u64); 256];
    for (&insn, &count) in &seen {
        let top = ((insn >> 24) & 0xFF) as usize;
        by_top[top].0 += count;
        if !switch_core::cpu::translates(insn) {
            by_top[top].1 += count;
        }
    }
    let mut groups: Vec<(u64, u64, usize)> = by_top
        .iter()
        .zip(0..256)
        .filter(|((_, missing), _)| *missing > 0)
        .map(|(&(ran, missing), top)| (missing, ran, top))
        .collect();
    groups.sort_by_key(|&(missing, ran, top)| (std::cmp::Reverse(missing), ran, top));
    println!("--- by encoding group (bits 31:24) ---");
    for (missing, ran, top) in groups.iter().take(ROWS) {
        println!(
            "  {top:#04x}  {missing:>10} of {ran:>10} run  ({:5.1}% of the group, {:5.2}% of the frame)",
            *missing as f64 * 100.0 / *ran as f64,
            *missing as f64 * 100.0 / total as f64,
        );
    }
}
