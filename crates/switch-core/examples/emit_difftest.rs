//! Differential test of emitted wasm blocks against the interpreter on real guest code.
//!
//! This half records cases; `tools/emit_difftest.mjs` runs the modules under V8.
//!
//! ```text
//! cargo run --release --example emit_difftest -- <nro> [-- <outdir>]
//! node tools/emit_difftest.mjs <outdir>
//! ```
mod common;

const USAGE: &str = "emit_difftest <target> [prod.keys] [title.keys] [font.ttf]";

use common::{Flow, Pace};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use switch_core::cpu::{defers, Cpu, Layout, Refused, DISCARD_SLOT};
use switch_core::disasm::disassemble;

/// Guest state offset in the module's memory.
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
/// Guest `pc` offset, written by a taken branch.
const PC_AT: u32 = 4132;

/// Page table offset: one four-byte entry per 4 KiB page.
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
    pc: PC_AT,
};

/// Encodings listed in the refusal report.
const ROWS: usize = 20;

const CANDIDATES: usize = 4000;

/// Instructions to run before sampling, to skip the loader.
const WARMUP: u64 = 40_000_000;

fn main() {
    let args = common::program_args(USAGE);
    let out_dir = std::env::var("OUT").unwrap_or_else(|_| "target/emit-difftest".into());
    let program = args.open_program();
    let mut cpu = Cpu::new();
    cpu.bootstrap();
    program.boot(&mut cpu);

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
        let before = cpu.reg_slots();
        let nzcv_before = cpu.nzcv();
        cpu.set_pc(pc);
        let mut faulted = false;
        let mut retired = 0usize;
        let mut left = None;
        for i in 0..ops {
            if cpu.step().is_err() {
                faulted = true;
                break;
            }
            retired = i + 1;
            let next = cpu.get_pc();
            if next != path[retired] {
                left = Some(next);
                break;
            }
        }
        if faulted {
            skipped_fault += 1;
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
        // Empty delta, but the marker is still required.
        manifest.push_str(" !");
        manifest.push('\n');
        cases += 1;
    }

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
         discard_slot {DISCARD_SLOT}\n\
         pc_at {PC_AT}\n\
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
        "{cases} cases written to {out_dir}/, {branched} of them leaving at a branch they took \
         ({skipped_fault} blocks faulted under the interpreter)"
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

fn before_len() -> usize {
    Cpu::new().reg_slots().len()
}
