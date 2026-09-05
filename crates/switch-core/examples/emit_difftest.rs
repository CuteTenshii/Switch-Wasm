//! The emitted wasm against the interpreter, on real guest code:
//! `emit_difftest <target> [prod.keys] [title.keys] [font.ttf]`.
//!
//! `jit_difftest` runs the two engines side by side because both are in this
//! binary. The emitter's output is not: it is a wasm module, and nothing in
//! `switch-core` can run one (the crate has no dependencies, and on the host
//! there is no engine at all). So the comparison is split in two.
//!
//! This half finds real blocks the emitter can write, records the guest state
//! going in, steps the **interpreter** over exactly those instructions, and
//! records the state coming out. It writes each module beside its case, and
//! `tools/emit_difftest.mjs` runs them under V8 and reports any register or
//! NZCV that came out different.
//!
//! The blocks are the ones a title actually executes, not encodings chosen
//! here: a difference the emitter has only shows up on the operand values and
//! flag states real code produces.
//!
//! ```text
//! cargo run --profile quick --example emit_difftest -- <nro> [-- <outdir>]
//! node tools/emit_difftest.mjs <outdir>
//! ```
mod common;

const USAGE: &str = "emit_difftest <target> [prod.keys] [title.keys] [font.ttf]";

use common::{Flow, Pace};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use switch_core::cpu::Cpu;

/// Where the harness puts guest state in the memory it hands a module. The
/// register file at zero and NZCV clear of it; an emitted block only ever
/// reaches these two, so nothing else has to be modelled.
const REGS_AT: u32 = 0;
const NZCV_AT: u32 = 4096;

/// How many distinct block entry points to try.
const CANDIDATES: usize = 4000;

/// Instructions to run before sampling, so the addresses are code the title
/// really reaches rather than its loader.
const WARMUP: u64 = 40_000_000;

fn main() {
    let args = common::program_args(USAGE);
    let out_dir = std::env::var("OUT").unwrap_or_else(|_| "target/emit-difftest".into());
    let program = args.open_program();
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    program.boot(&mut cpu);

    // Collect addresses the guest actually branches to. Sampling the pc every
    // instruction would give mostly mid-block addresses, which are legal entry
    // points but over-represent the middle of long runs; taking it once per
    // slice spreads the sample over the whole frame instead.
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    common::drive(&mut cpu, Pace::Instructions, WARMUP, |cpu, steps| {
        if steps % 97 == 0 {
            seen.insert(cpu.get_pc());
        }
        if seen.len() >= CANDIDATES * 8 {
            return Flow::Stop;
        }
        Flow::Continue
    });

    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");
    let mut manifest = String::new();
    let mut cases = 0usize;
    let mut refused = 0usize;
    let mut skipped_fault = 0usize;

    for &pc in seen.iter() {
        if cases >= CANDIDATES {
            break;
        }
        let Some((module, ops)) = cpu.emit_block_at(pc, REGS_AT, NZCV_AT) else {
            refused += 1;
            continue;
        };
        // A block of one op is nearly always a lone `MOV`; it would pass
        // without saying anything about the operand handling.
        if ops < 2 {
            continue;
        }

        let before = cpu.reg_slots();
        let nzcv_before = cpu.nzcv();
        cpu.set_pc(pc);
        let mut faulted = false;
        for _ in 0..ops {
            if cpu.step().is_err() {
                faulted = true;
                break;
            }
        }
        if faulted {
            skipped_fault += 1;
            continue;
        }
        let after = cpu.reg_slots();
        let nzcv_after = cpu.nzcv();

        let name = format!("case{cases:04}");
        std::fs::write(format!("{out_dir}/{name}.wasm"), &module).expect("cannot write a module");
        let _ = write!(
            manifest,
            "{name} {pc:#010x} {ops} {nzcv_before:#010x} {nzcv_after:#010x}"
        );
        for v in before {
            let _ = write!(manifest, " {v:016x}");
        }
        manifest.push_str(" |");
        for v in after {
            let _ = write!(manifest, " {v:016x}");
        }
        manifest.push('\n');
        cases += 1;
    }

    let header = format!(
        "regs_at {REGS_AT}\nnzcv_at {NZCV_AT}\nslots {}\n",
        before_len()
    );
    std::fs::write(format!("{out_dir}/manifest.txt"), header + &manifest)
        .expect("cannot write the manifest");

    println!(
        "{cases} cases written to {out_dir}/ ({refused} blocks the emitter refused, \
         {skipped_fault} that faulted under the interpreter)"
    );
    println!("now run: node tools/emit_difftest.mjs {out_dir}");
}

/// How many register slots a case carries, taken from the snapshot itself so
/// the manifest and the reader cannot disagree about the width.
fn before_len() -> usize {
    Cpu::new().reg_slots().len()
}
