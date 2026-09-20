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
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use switch_core::cpu::{defers, Cpu, Layout, Refused};
use switch_core::disasm::disassemble;

/// Where the harness puts guest state in the memory it hands a module.
const REGS_AT: u32 = 0;
const NZCV_AT: u32 = 4096;
const READ_WATCH_LO_AT: u32 = 4100;
const READ_WATCH_HI_AT: u32 = 4104;
const WATCH_LO_AT: u32 = 4108;
const WATCH_HI_AT: u32 = 4112;
const READONLY_LO_AT: u32 = 4116;
const READONLY_HI_AT: u32 = 4120;
const WATCHED_AT: u32 = 4124;
const PAGES_AT: u32 = 4128;

/// Where the page table goes, and how much memory that needs: one four-byte
/// entry per 4 KiB of the guest's 4 GiB.
const TABLE_AT: u32 = 0x0010_0000;
const TABLE_BYTES: u32 = (1 << 20) * 4;

const LAYOUT: Layout = Layout {
    regs: REGS_AT,
    nzcv: NZCV_AT,
    pages: PAGES_AT,
    read_watch_lo: READ_WATCH_LO_AT,
    read_watch_hi: READ_WATCH_HI_AT,
    watch_lo: WATCH_LO_AT,
    watch_hi: WATCH_HI_AT,
    readonly_lo: READONLY_LO_AT,
    readonly_hi: READONLY_HI_AT,
    watched: WATCHED_AT,
};

/// How many encodings to name in the refusal report.
const ROWS: usize = 20;

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
    let mut refused_flow = 0usize;
    let mut refused_long = 0usize;
    let mut skipped_fault = 0usize;
    // The encodings that took a block out of the emitted path, by how many
    // blocks each one cost. This is the list that says what to write next.
    let mut unwritable: HashMap<u32, u64> = HashMap::new();

    for &pc in seen.iter() {
        if cases >= CANDIDATES {
            break;
        }
        let (module, ops) = match cpu.emit_block_at(pc, LAYOUT) {
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
        // How much of the block will run. A title's guest memory is hundreds
        // of megabytes and the harness hands a module a bare buffer, so every
        // page here is unmapped and every access hands its instruction back:
        // the block runs as far as its first one and reports that. Which is
        // still the whole of most blocks, and it is real code with real
        // operands, which is what this half is for. What a *mapped* access
        // does is `emit_selftest`'s to check, against a window it owns.
        let ops = (0..ops)
            .find(|i| cpu.mem.read_u32(pc + 4 * *i as u32).is_ok_and(defers))
            .unwrap_or(ops);
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
            "{name} {pc:#010x} {ops} exact {nzcv_before:#010x} {nzcv_after:#010x}"
        );
        for v in before {
            let _ = write!(manifest, " {v:016x}");
        }
        manifest.push_str(" |");
        for v in after {
            let _ = write!(manifest, " {v:016x}");
        }
        // No window is mapped, so nothing this block did can have reached
        // guest memory and the delta is empty. The marker is still owed: it
        // is what separates the two snapshots from it.
        manifest.push_str(" !");
        manifest.push('\n');
        cases += 1;
    }

    // No `page` lines, so every page-table entry stays zero and an emitted
    // access finds nothing at its address. No watchpoints and no protected
    // ranges either, which leaves all three disarmed, and no `watched_at`
    // bitmap, which is a `Memory` nothing is caching a page of.
    let header = format!(
        "regs_at {REGS_AT}\n\
         nzcv_at {NZCV_AT}\n\
         read_watch_lo_at {READ_WATCH_LO_AT}\n\
         read_watch_hi_at {READ_WATCH_HI_AT}\n\
         watch_lo_at {WATCH_LO_AT}\n\
         watch_hi_at {WATCH_HI_AT}\n\
         readonly_lo_at {READONLY_LO_AT}\n\
         readonly_hi_at {READONLY_HI_AT}\n\
         watched_at {WATCHED_AT}\n\
         pages_at {PAGES_AT}\n\
         table_at {TABLE_AT}\n\
         slots {}\n\
         wasm_pages {}\n",
        before_len(),
        (TABLE_AT + TABLE_BYTES).div_ceil(64 * 1024),
    );
    std::fs::write(format!("{out_dir}/manifest.txt"), header + &manifest)
        .expect("cannot write the manifest");

    let refused_op: u64 = unwritable.values().sum();
    println!(
        "{cases} cases written to {out_dir}/ ({skipped_fault} blocks faulted under the \
         interpreter)"
    );
    println!(
        "refused: {refused_flow} for control flow, {refused_op} for an op with no emitter, \
         {refused_long} for length"
    );

    // Ranked by blocks cost rather than by how often the encoding runs: one
    // instruction with no emitter takes its whole block with it, so this is
    // what writing that one op would buy.
    let mut ranked: Vec<(u64, u32)> = unwritable.iter().map(|(&i, &n)| (n, i)).collect();
    ranked.sort_by_key(|&(count, insn)| (std::cmp::Reverse(count), insn));
    println!("--- encodings that cost the most blocks ---");
    for (count, insn) in ranked.iter().take(ROWS) {
        println!("  {insn:#010x}  {count:>6}  {}", disassemble(*insn));
    }

    println!("now run: node tools/emit_difftest.mjs {out_dir}");
}

/// How many register slots a case carries, taken from the snapshot itself so
/// the manifest and the reader cannot disagree about the width.
fn before_len() -> usize {
    Cpu::new().reg_slots().len()
}
