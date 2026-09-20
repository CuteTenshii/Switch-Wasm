//! The emitted wasm against the interpreter, on encodings chosen to break it:
//! `emit_selftest`.
//!
//! `emit_difftest` is the same comparison on blocks a real title executes, and
//! it only ever covers what that title runs. hbmenu never divides by zero,
//! never rotates by a variable amount and never runs a `CCMP`, so the arms
//! that write those were written against nothing. This half supplies the
//! operands a title will not: zero and minus one into every divisor, `INT_MIN`
//! into every dividend, shift amounts sitting exactly on 32 and 64, and all
//! sixteen settings of NZCV under every condition code.
//!
//! It needs no target and no keys, so it runs in CI where `emit_difftest`
//! cannot.
//!
//! # Nothing here is assembled by hand
//!
//! An encoding written out in this file and quietly wrong tests some other
//! instruction and passes anyway, which is the one failure a difftest cannot
//! see. So the instructions come from sweeping the encoding space and asking
//! [`switch_core::cpu::emits`] which words the emitter has an arm for. What
//! each one *is* comes from the disassembler, and is used only to group them
//! and to name them in the report: the comparison itself does not care, since
//! the interpreter is the reference for whatever the word turns out to mean.
//!
//! ```text
//! cargo run --profile quick --example emit_selftest
//! node tools/emit_difftest.mjs target/emit-selftest
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use switch_core::cpu::{emits, Cpu};
use switch_core::disasm::disassemble;

/// Where the harness puts guest state in the memory it hands a module, the
/// same two offsets `emit_difftest` uses, so one manifest reader serves both.
const REGS_AT: u32 = 0;
const NZCV_AT: u32 = 4096;

/// Where the test program is mapped. One page, and a block never spans one.
const CODE_AT: u32 = 0x1000;
const CODE_BYTES: u32 = 0x1000;

/// `MSR NZCV, X0`, the one instruction of the preamble: NZCV has no setter on
/// [`Cpu`], and seeding it matters more here than anything else does, because
/// the conditional forms are most of what this file exists to reach.
const MSR_NZCV_X0: u32 = 0xD51B_4200;

/// `RET`, which ends the translated block exactly at the last instruction
/// under test rather than letting translation run on into whatever follows.
const RET: u32 = 0xD65F_03C0;

/// How many instructions of one form go into a block. They share a block
/// rather than getting one each because a block is what the emitter refuses or
/// accepts, and because the manifest carries a whole register file per case.
const PER_FORM: usize = 16;

/// An instruction's *form*: its disassembly with every register number and
/// every immediate replaced, which is what a bucket is keyed on.
///
/// The mnemonic alone is too coarse, and the mnemonic with bit 31 beside it,
/// which is what this used to be, is too coarse in exactly one place. Bit 31
/// is `sf` in the data-processing forms and splitting on it separates the two
/// widths there; in a load or a store it is the top bit of `size`, so `ldr w`
/// and `ldr x` answer to the same name *and* the same bit. The sweep fills a
/// bucket from the low encoding prefixes, so it filled that one with 32-bit
/// loads and never wrote a 64-bit one.
///
/// The shape separates all of them, and every other pair a width, a sign or an
/// addressing mode decides, without this file having to know which field does
/// it in which group.
fn form(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'0' && i + 1 < b.len() && b[i + 1] | 0x20 == b'x' {
            i += 2;
            while i < b.len() && b[i].is_ascii_hexdigit() {
                i += 1;
            }
            out.push('_');
        } else if b[i].is_ascii_digit() {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            out.push('_');
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// Random fills per encoding-group prefix during the sweep. Every A64
/// data-processing group is named by bits 31:21, so filling the remaining 21
/// bits under each of the 2048 prefixes reaches every form there is an arm
/// for; this is how many tries each prefix gets.
const FILLS: usize = 256;

fn main() {
    let out_dir = std::env::var("OUT").unwrap_or_else(|_| "target/emit-selftest".into());
    let buckets = survey();

    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");
    let mut manifest = String::new();
    let mut cases = 0usize;
    let mut faulted = 0usize;
    let mut refused = 0usize;

    for (form, encodings) in &buckets {
        let mut program = vec![MSR_NZCV_X0];
        program.extend_from_slice(encodings);
        program.push(RET);

        let mut cpu = Cpu::new();
        cpu.mem
            .map_zero(CODE_AT, CODE_BYTES as usize)
            .expect("cannot map the code page");
        let mut bytes = Vec::with_capacity(program.len() * 4);
        for insn in &program {
            bytes.extend_from_slice(&insn.to_le_bytes());
        }
        cpu.mem.map(CODE_AT, &bytes).expect("cannot write the code");
        let block_at = CODE_AT + 4;

        for set in 0..SEEDS {
            for nzcv in 0u8..16 {
                seed(&mut cpu, set, nzcv);

                let before = cpu.reg_slots();
                let nzcv_before = cpu.nzcv();
                let Ok((module, ops)) = cpu.emit_block_at(block_at, REGS_AT, NZCV_AT) else {
                    refused += 1;
                    continue;
                };

                let mut fault = false;
                for _ in 0..ops {
                    if cpu.step().is_err() {
                        fault = true;
                        break;
                    }
                }
                if fault {
                    faulted += 1;
                    continue;
                }

                let name = format!("case{cases:05}");
                std::fs::write(format!("{out_dir}/{name}.wasm"), &module)
                    .expect("cannot write a module");
                let _ = write!(
                    manifest,
                    "{name} {block_at:#010x} {ops} {nzcv_before:#010x} {:#010x}",
                    cpu.nzcv()
                );
                for v in before {
                    let _ = write!(manifest, " {v:016x}");
                }
                manifest.push_str(" |");
                for v in cpu.reg_slots() {
                    let _ = write!(manifest, " {v:016x}");
                }
                manifest.push('\n');
                cases += 1;
            }
        }
        println!(
            "  {form:<34} {:>3} encodings   {}",
            encodings.len(),
            disassemble(encodings[0])
        );
    }

    let header = format!(
        "regs_at {REGS_AT}\nnzcv_at {NZCV_AT}\nslots {}\n",
        Cpu::new().reg_slots().len()
    );
    std::fs::write(format!("{out_dir}/manifest.txt"), header + &manifest)
        .expect("cannot write the manifest");

    println!(
        "{cases} cases over {} forms written to {out_dir}/ \
         ({refused} blocks the emitter refused, {faulted} that faulted)",
        buckets.len()
    );
    println!("now run: node tools/emit_difftest.mjs {out_dir}");
}

/// The encodings the emitter can write, grouped by [`form`].
///
/// Swept rather than assembled: see this file's note on why. Bits 31:21 name
/// the group in every data-processing form, so every prefix gets its own tries
/// and no group can be missed because random words rarely land in it.
fn survey() -> BTreeMap<String, Vec<u32>> {
    let mut rng = 0x243F_6A88_85A3_08D3u64;
    let mut buckets: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for prefix in 0u32..(1 << 11) {
        for _ in 0..FILLS {
            let insn = (prefix << 21) | (next(&mut rng) as u32 & 0x1F_FFFF);
            if !emits(insn) {
                continue;
            }
            let slot = buckets.entry(form(&disassemble(insn))).or_default();
            if slot.len() < PER_FORM {
                slot.push(insn);
            }
        }
    }
    buckets
}

/// How many register-content sets each block is run under.
const SEEDS: usize = 10;

/// Put set `set` into the register file and `nzcv` into the flags.
///
/// The flags go in first, through the preamble instruction, because writing
/// them means running guest code that clobbers `X0`; the register file is
/// seeded after it for that reason and not before.
fn seed(cpu: &mut Cpu, set: usize, nzcv: u8) {
    cpu.set_reg(0, u64::from(nzcv) << 28);
    cpu.set_pc(CODE_AT);
    cpu.step().expect("the preamble cannot fault");
    assert_eq!(
        cpu.nzcv() >> 28,
        u32::from(nzcv),
        "the preamble did not set the flags, so every conditional form here is \
         being tested under one setting"
    );
    for i in 0..31u8 {
        cpu.set_reg(i, seed_value(set, u64::from(i)));
    }
    cpu.set_pc_and_sp(CODE_AT + 4, seed_value(set, 31));
}

/// What register `i` holds under set `set`.
///
/// The sets are adversarial rather than random, and pair up on purpose: with
/// the even registers holding `i64::MIN` and the odd ones minus one, any
/// `SDIV` whose operands land that way is the division wasm traps on and A64
/// wraps. The three ramps do the same for shift amounts, which A64 takes
/// modulo the operand width and wasm takes modulo its own: they put the
/// amounts either side of 32 and of 64, where the two disagree.
fn seed_value(set: usize, i: u64) -> u64 {
    let even = i.is_multiple_of(2);
    match set {
        0 => 0,
        1 => u64::MAX,
        2 => {
            if even {
                1 << 63
            } else {
                u64::MAX
            }
        }
        3 => {
            if even {
                0x8000_0000
            } else {
                0xFFFF_FFFF
            }
        }
        4 => i,
        5 => 32 + i,
        6 => 63 + i,
        7 => splitmix(0x9E37_79B9_7F4A_7C15 ^ i),
        8 => splitmix(0xD1B5_4A32_D192_ED03 ^ i),
        _ => {
            if even {
                splitmix(i)
            } else {
                i % 65
            }
        }
    }
}

/// SplitMix64, so the sweep and the pseudorandom seed sets are the same on
/// every run and a failure can be looked at again.
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix(*state)
}
