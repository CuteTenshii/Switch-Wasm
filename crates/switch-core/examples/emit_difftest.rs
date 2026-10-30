//! Differential test of emitted wasm blocks against the interpreter on real guest code.
//!
//! This half records cases; `tools/emit_difftest.mjs` runs the modules under V8.
//!
//! ```text
//! cargo run --release --example emit_difftest -- <nro> [-- <outdir>]
//! FROM_FRAME=<n> cargo run --release --example emit_difftest -- <title> <prod.keys>
//! node tools/emit_difftest.mjs <outdir>
//! ```
mod common;

const USAGE: &str = "emit_difftest <target> [prod.keys] [title.keys] [font.ttf]";

use common::emit::{self, LAYOUT, TABLE_AT, TABLE_BYTES};
use common::{Flow, Pace};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use switch_core::cpu::{defers, Refused};
use switch_core::disasm::disassemble;

/// Encodings listed in the refusal report.
const ROWS: usize = 20;

const CANDIDATES: usize = 4000;

/// Instructions to run before sampling, to skip the loader.
const WARMUP: u64 = 40_000_000;

fn main() {
    let args = common::program_args(USAGE);
    let out_dir = std::env::var("OUT").unwrap_or_else(|_| "target/emit-difftest".into());
    let program = args.open_program();
    let mut cpu = common::new_cpu();
    cpu.bootstrap();
    program.boot(&mut cpu);

    // `FROM_FRAME=<n>` runs to that frame first, to sample a title's later code.
    if let Some(frame) = std::env::var("FROM_FRAME")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        common::run_to(&mut cpu, u64::MAX, |cpu| cpu.nv.gpu.frames >= frame);
    }
    // Sample the pc once per slice to get branch targets, not mid-block addresses.
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
    let mut refused_flow = 0usize;
    let mut branched = 0usize;
    let mut refused_long = 0usize;
    let mut skipped_fault = 0usize;
    let mut skipped_switch = 0usize;
    // Encodings that kept a block off the emitted path, by blocks lost.
    let mut unwritable: HashMap<u32, u64> = HashMap::new();

    for &pc in seen.iter() {
        if cases >= CANDIDATES {
            break;
        }
        let (module, path) = match cpu.emit_block_at(pc, LAYOUT) {
            Ok(emitted) => emitted,
            Err(Refused::ControlFlow) => {
                refused_flow += 1;
                continue;
            }
            Err(Refused::TooLong) => {
                refused_long += 1;
                continue;
            }
            Err(Refused::Op(insn)) => {
                *unwritable.entry(insn).or_default() += 1;
                continue;
            }
        };
        // No guest memory is mapped, so a block runs up to its first memory access.
        let whole = path.len() - 1;
        let ops = (0..whole)
            .find(|&i| cpu.mem.read_u32(path[i]).is_ok_and(defers))
            .unwrap_or(whole);
        if ops < 2 {
            continue;
        }

        // Step the interpreter until control leaves the straight-line path.
        for i in 0..32 {
            cpu.set_vreg(i, emit::vreg_seed(i));
        }
        let before = cpu.reg_slots();
        let nzcv_before = cpu.nzcv();
        let fpsr_before = cpu.fpsr();
        let (tpidr, tpidr_rw) = cpu.thread_pointers();
        // The fuel is zero, so the origin is the clock itself.
        let origin = cpu.cycles;
        cpu.set_pc(pc);
        let mut faulted = false;
        let mut switched = false;
        let mut retired = 0usize;
        let mut left = None;
        let thread = cpu.current_thread_handle();
        for i in 0..ops {
            if cpu.step().is_err() {
                faulted = true;
                break;
            }
            // A step may preempt into another thread, whose pc says nothing about this block.
            if cpu.current_thread_handle() != thread {
                switched = true;
                break;
            }
            retired = i + 1;
            let next = cpu.get_pc();
            // A written terminator's slot is `u32::MAX`, which a jump to -1 would match.
            if next != path[retired] || path[retired] == u32::MAX {
                left = Some(next);
                break;
            }
        }
        if faulted {
            skipped_fault += 1;
            continue;
        }
        if switched {
            skipped_switch += 1;
            continue;
        }
        if left.is_some() {
            branched += 1;
        }
        let mode = match left {
            Some(target) => format!("left:{target:#010x}"),
            None => "exact".into(),
        };
        let after = cpu.reg_slots();
        let nzcv_after = cpu.nzcv();

        let name = format!("case{cases:04}");
        std::fs::write(format!("{out_dir}/{name}.wasm"), &module).expect("cannot write a module");
        let _ = write!(
            manifest,
            "{name} {pc:#010x} {retired} {mode} {nzcv_before:#010x} {nzcv_after:#010x}"
        );
        for v in before {
            let _ = write!(manifest, " {v:016x}");
        }
        manifest.push_str(" |");
        for v in after {
            let _ = write!(manifest, " {v:016x}");
        }
        // No memory delta, but the marker is still required.
        manifest.push_str(" !");
        manifest.push_str(&emit::vreg_delta(&cpu, emit::vreg_seed));
        let _ = write!(manifest, " fpsr:{fpsr_before:x}:{:x}", cpu.fpsr());
        let _ = write!(manifest, " tpidr:{tpidr:x}:{tpidr_rw:x} clock:{origin:x}");
        manifest.push('\n');
        cases += 1;
    }

    let header = emit::header()
        + &format!(
            "wasm_pages {}\n",
            (TABLE_AT + TABLE_BYTES).div_ceil(64 * 1024)
        );
    std::fs::write(format!("{out_dir}/manifest.txt"), header + &manifest)
        .expect("cannot write the manifest");

    let refused_op: u64 = unwritable.values().sum();
    println!(
        "{cases} cases written to {out_dir}/, {branched} of them leaving at a branch they took \
         ({skipped_fault} blocks faulted under the interpreter, {skipped_switch} switched threads)"
    );
    println!(
        "refused: {refused_flow} for control flow, {refused_op} for an op with no emitter, \
         {refused_long} for length"
    );

    // Ranked by blocks lost, not by execution count.
    let mut ranked: Vec<(u64, u32)> = unwritable.iter().map(|(&i, &n)| (n, i)).collect();
    ranked.sort_by_key(|&(count, insn)| (std::cmp::Reverse(count), insn));
    println!("--- encodings that cost the most blocks ---");
    for (count, insn) in ranked.iter().take(ROWS) {
        println!("  {insn:#010x}  {count:>6}  {}", disassemble(*insn));
    }

    println!("now run: node tools/emit_difftest.mjs {out_dir}");
}
