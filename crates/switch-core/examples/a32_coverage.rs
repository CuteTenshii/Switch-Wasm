//! Which of a 32-bit title's NEON and VFP encodings the interpreter refuses:
//! `a32_coverage <image> <base> <text length> [<image> <base> <text length>...]`.
//!
//! The images are `dump_exefs` output, each a module laid out at its load
//! address, with the base and text length it prints. Every word of each text
//! section in the NEON and VFP encoding space is run on its own, with every
//! register pointing at mapped scratch memory so a load or store has somewhere
//! to go, and the refusals are grouped by encoding with the register fields
//! masked out. A title stops on one missing instruction at a time, often an
//! hour into a run; this lists all of them in seconds.
//!
//! Literal pools sit in text sections too, so a group seen once or twice may
//! be data that happens to fall in the space. The counts say which is which.
mod common;

use std::collections::BTreeMap;
use switch_core::cpu::{Cpu, ExecMode};

const USAGE: &str = "a32_coverage <image> <base> <text length> [<image> <base> <text length>...]";

const CODE: u32 = 0x1000;
const SCRATCH: u32 = 0x10_0000;
const SCRATCH_SIZE: usize = 0x10_0000;

/// NEON data processing (`1111 001x`), element and structure loads and
/// stores (`1111 0100 xxx0`), and VFP (`cccc 110x`/`1110`, coprocessor 10 or
/// 11).
fn in_scope(word: u32) -> bool {
    let neon = word >> 25 == 0b111_1001 || word >> 24 == 0xF4 && (word >> 20) & 1 == 0;
    let vfp = word >> 28 != 0xF && (word >> 25) & 0b111 >= 0b110 && (word >> 9) & 0b111 == 0b101;
    neon || vfp
}

/// The encoding with its register fields cleared, so one instruction class
/// is one key: Vd (22, 15:12), Vn (7, 19:16) and Vm (5, 3:0) for NEON, and
/// the same positions for VFP's D and S register numbers.
fn class(word: u32) -> u32 {
    word & !(0x0040_F000 | 0x000F_0080 | 0x0000_002F)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || !args.len().is_multiple_of(3) {
        eprintln!("usage: {USAGE}");
        std::process::exit(2);
    }
    // Refused classes: how many words, one example, and the reason.
    let mut refused: BTreeMap<u32, (usize, u32, u32, String)> = BTreeMap::new();
    let mut scanned = 0usize;
    for module in args.chunks(3) {
        let image = common::read(module[0].clone());
        let base = common::hex(&module[1]);
        let text = common::hex(&module[2]) as usize;
        for offset in (0..text.min(image.len())).step_by(4) {
            let word = u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
            if !in_scope(word) {
                continue;
            }
            scanned += 1;
            let mut cpu = Cpu::new();
            cpu.mem.map_zero(SCRATCH, SCRATCH_SIZE).unwrap();
            cpu.mem.map(CODE, &word.to_le_bytes()).unwrap();
            cpu.set_mode(ExecMode::A32);
            cpu.set_pc_and_sp(CODE, u64::from(SCRATCH) + SCRATCH_SIZE as u64 / 2);
            for reg in 0..13 {
                cpu.set_reg(reg, u64::from(SCRATCH) + SCRATCH_SIZE as u64 / 2);
            }
            let Err(error) = cpu.run(1) else {
                continue;
            };
            let reason = error.to_string();
            if !reason.contains("unimplemented") && !reason.contains("undefined") {
                continue;
            }
            let entry =
                refused
                    .entry(class(word))
                    .or_insert((0, word, base + offset as u32, reason));
            entry.0 += 1;
        }
    }
    let mut rows: Vec<_> = refused.into_iter().collect();
    rows.sort_by_key(|(_, (count, ..))| std::cmp::Reverse(*count));
    println!("{scanned} words in scope, {} classes refused", rows.len());
    for (key, (count, example, at, reason)) in rows {
        println!("{count:>6}  {key:#010x}  e.g. {example:#010x} at {at:#010x}: {reason}");
    }
}
