//! Emitted wasm against the interpreter on adversarial operands and memory:
//! `emit_selftest`. Needs no title or keys, so it runs in CI.
//!
//! Encodings are swept from [`switch_core::cpu::emits`], never hand-assembled.
//!
//! ```text
//! cargo run --release --example emit_selftest
//! node tools/emit_difftest.mjs target/emit-selftest
//! ```

mod common;

use common::emit::{self, LAYOUT, TABLE_AT, TABLE_BYTES};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use switch_core::cpu::{defers, emits, Cpu};
use switch_core::disasm::disassemble;

/// Where the mapped guest pages live, one 4 KiB slot each.
const POOL_AT: u32 = TABLE_AT + TABLE_BYTES;

/// Bitmap of pages with cached contents, one bit per guest page.
const BITMAP_AT: u32 = POOL_AT + WINDOW_BYTES;
const BITMAP_BYTES: u32 = (1 << 20) / 8;

/// Slot of the `n`th window page, scattered so page-boundary checks are exercised.
/// [`WINDOW_PAGES`] is prime, so the stepping reaches every slot.
const SCATTER: u32 = 7;
fn pool_slot(page: u32) -> u32 {
    (page * SCATTER) % WINDOW_PAGES
}

/// The guest window, starting at zero so small register seeds are addresses.
const WINDOW_AT: u32 = 0;
const WINDOW_PAGES: u32 = 17;
const PAGE_BYTES: u32 = 4096;
const WINDOW_BYTES: u32 = WINDOW_PAGES * PAGE_BYTES;

/// Where the test program is mapped, inside that window.
const CODE_AT: u32 = 0x1000;

/// The read watchpoint, armed for every case.
const READ_WATCH_AT: u32 = 0x40;
const READ_WATCH_BYTES: u32 = 0x40;

/// The write watchpoint, over a range distinct from the read one.
const WATCH_AT: u32 = 0xC0;
const WATCH_BYTES: u32 = 0x40;

/// The cached page; stores there must hand back. Re-armed per case.
const WATCHED_PAGE_AT: u32 = 3 * PAGE_BYTES;

/// Two write-protected pages with writable pages between, which stores see
/// through the cached-pages bitmap.
const READONLY_ONE: (u32, u32) = (5 * PAGE_BYTES, 6 * PAGE_BYTES);
const READONLY_TWO: (u32, u32) = (8 * PAGE_BYTES, 9 * PAGE_BYTES);

/// What `MRS` of the thread registers reads; small enough for the manifest's numbers.
const TPIDR_SEED: u64 = 0x1234_5678_9ABC;
const TPIDR_RW_SEED: u64 = 0x0FED_CBA9_8765;

/// The clock at the block's start, by seed set: the edges of the tick
/// conversion's quotient and remainder, and values whose 64-bit product overflows.
const CLOCK_SEEDS: [u64; 6] = [
    0,
    1_019_999_999,
    1_020_000_000,
    0x2000_0000_0000_0123,
    0xFFFF_FFFF_FFFF_F000,
    0x7FFF_FFFF_FFFF_FFFF,
];

/// `MSR NZCV, X0`: the preamble that seeds the flags.
const MSR_NZCV_X0: u32 = 0xD51B_4200;

/// `RET`, ending the translated block after the last instruction under test.
const RET: u32 = 0xD65F_03C0;

/// Instructions of one form per block.
const PER_FORM: usize = 16;

/// An instruction's form: its disassembly with registers and immediates
/// replaced, which keys a bucket.
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

/// Random fills per 11-bit encoding-group prefix (bits 31:21) during the sweep.
const FILLS: usize = 256;

fn main() {
    let out_dir = std::env::var("OUT").unwrap_or_else(|_| "target/emit-selftest".into());
    let buckets = survey();

    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");
    std::fs::write(format!("{out_dir}/guest.bin"), window_bytes())
        .expect("cannot write the window");

    let mut manifest = String::new();
    let mut cases = 0usize;
    let mut faulted = 0usize;
    let mut refused = 0usize;
    let mut rewrote = 0usize;
    let mut handed_back = 0usize;
    let mut pinned = 0usize;
    let all_flags: Vec<u8> = (0u8..16).collect();

    for (form, encodings) in &buckets {
        // Instructions that can hand back get their own block.
        let bucket_hands_back = encodings.iter().copied().any(defers);
        let bodies: Vec<&[u32]> = if bucket_hands_back {
            encodings.iter().map(std::slice::from_ref).collect()
        } else {
            vec![encodings.as_slice()]
        };
        // NZCV does not affect memory accesses, so those run under one setting.
        let flags: &[u8] = if bucket_hands_back {
            &all_flags[..1]
        } else {
            &all_flags
        };

        for body in bodies {
            let hands_back = body.iter().copied().any(defers);
            let mut program = vec![MSR_NZCV_X0];
            program.extend_from_slice(body);
            program.push(RET);
            let code = code_bytes(&program);
            let start = baseline(&code);
            let mut cpu = mapped_cpu(&code);
            let block_at = CODE_AT + 4;

            // The code page is part of each case's starting state.
            manifest.push_str("code ");
            for b in &code {
                let _ = write!(manifest, "{b:02x}");
            }
            manifest.push('\n');

            for set in 0..SEEDS {
                for &nzcv in flags {
                    // A case before this one may have stored over the preamble.
                    cpu.mem
                        .map(WINDOW_AT, &start)
                        .expect("the window is mapped");
                    seed(&mut cpu, set, nzcv);
                    for i in 0..32 {
                        cpu.set_vreg(i, vreg_seed(i, set == SPECIALS_SET));
                    }

                    cpu.set_fpsr(0);
                    cpu.set_thread_pointers(TPIDR_SEED, TPIDR_RW_SEED);
                    cpu.cycles = CLOCK_SEEDS[set % CLOCK_SEEDS.len()];
                    // The fuel is zero, so the origin is the clock itself.
                    let origin = cpu.cycles;
                    let before = cpu.reg_slots();
                    let nzcv_before = cpu.nzcv();
                    let Ok((module, path)) = cpu.emit_block_at(block_at, LAYOUT) else {
                        refused += 1;
                        continue;
                    };
                    let ops = path.len() - 1;

                    // Reset the window through `map` (ignoring read-only ranges) and re-arm the
                    // cached page before each block.
                    cpu.mem
                        .map(WINDOW_AT, &start)
                        .expect("the window is mapped");
                    cpu.mem.take_read_hit();
                    cpu.mem.take_watch_hit();
                    cpu.mem.mark_code_page(WATCHED_PAGE_AT);
                    cpu.mem.dirty_code_pages(0);
                    arm_monitor(&mut cpu, body[0], set);
                    let monitor_before = cpu.exclusive_state();
                    let mut fault = false;
                    for _ in 0..ops {
                        if cpu.step().is_err() {
                            fault = true;
                            break;
                        }
                    }
                    if fault && !hands_back {
                        faulted += 1;
                        continue;
                    }
                    // A store over the block's own terminator; real code pages hand back.
                    let left = path.last() == Some(&u32::MAX);
                    if left && cpu.mem.read_u32(path[ops - 1]).ok() != Some(RET) {
                        rewrote += 1;
                        continue;
                    }

                    // Faults, watched reads and writes, and cached-page stores must all hand
                    // back with no state change.
                    let must_hand_back = fault
                        || cpu.mem.take_read_hit().is_some()
                        || cpu.mem.take_watch_hit().is_some()
                        || cpu
                            .mem
                            .dirty_code_pages(0)
                            .is_none_or(|pages| !pages.is_empty());

                    let name = format!("case{cases:05}");
                    std::fs::write(format!("{out_dir}/{name}.wasm"), &module)
                        .expect("cannot write a module");
                    // A written terminator leaves with the interpreter's pc.
                    let mode = match (hands_back, must_hand_back, left) {
                        (false, _, false) => "exact".into(),
                        (false, _, true) => format!("left:{:#010x}", cpu.get_pc()),
                        (true, false, false) => "maybe".into(),
                        (true, false, true) => format!("maybe-left:{:#010x}", cpu.get_pc()),
                        (true, true, _) => {
                            pinned += 1;
                            "none".into()
                        }
                    };
                    let _ = write!(
                        manifest,
                        "{name} {block_at:#010x} {ops} {mode} {nzcv_before:#010x} {:#010x}",
                        cpu.nzcv()
                    );
                    for v in before {
                        let _ = write!(manifest, " {v:016x}");
                    }
                    manifest.push_str(" |");
                    for v in cpu.reg_slots() {
                        let _ = write!(manifest, " {v:016x}");
                    }
                    manifest.push_str(" !");
                    manifest.push_str(&memory_delta(&cpu, &start));
                    manifest.push_str(&emit::vreg_delta(&cpu, |i| {
                        vreg_seed(i, set == SPECIALS_SET)
                    }));
                    if set == SPECIALS_SET {
                        manifest.push_str(" specials");
                    }
                    let _ = write!(manifest, " fpsr:0:{:x} clock:{origin:x}", cpu.fpsr());
                    let monitor_after = cpu.exclusive_state();
                    let _ = write!(
                        manifest,
                        " excl:{:x}:{}:{:x}:{:x}:{}:{:x}",
                        monitor_before.0,
                        u8::from(monitor_before.1),
                        monitor_before.2,
                        monitor_after.0,
                        u8::from(monitor_after.1),
                        monitor_after.2,
                    );
                    manifest.push('\n');
                    cases += 1;
                }
            }
        }
        if bucket_hands_back {
            handed_back += 1;
        }
        println!(
            "  {form:<34} {:>3} encodings   {}",
            encodings.len(),
            disassemble(encodings[0])
        );
    }

    let header = emit::header()
        + &format!(
            "tpidr_values {TPIDR_SEED} {TPIDR_RW_SEED}\n\
         bitmap_at {BITMAP_AT}\n\
         code_at {CODE_AT}\n\
         read_watch {READ_WATCH_AT} {}\n\
         watch {WATCH_AT} {}\n\
         watched_page {WATCHED_PAGE_AT} {} {}\n\
         wasm_pages {}\n\
         {}",
            READ_WATCH_AT + READ_WATCH_BYTES,
            WATCH_AT + WATCH_BYTES,
            READONLY_ONE.0,
            READONLY_TWO.0,
            (BITMAP_AT + BITMAP_BYTES).div_ceil(64 * 1024),
            page_lines(),
        );
    std::fs::write(format!("{out_dir}/manifest.txt"), header + &manifest)
        .expect("cannot write the manifest");

    println!(
        "{cases} cases over {} forms written to {out_dir}/ \
         ({handed_back} forms able to hand an instruction back, \
         {pinned} cases where it has to, \
         {refused} blocks the emitter refused, {faulted} that faulted, \
         {rewrote} that stored over their terminator)",
        buckets.len()
    );
    println!("now run: node tools/emit_difftest.mjs {out_dir}");
}

/// [`emit::vreg_seed`], or IEEE special values for the specials set.
fn vreg_seed(i: u8, specials: bool) -> u128 {
    if specials {
        special_floats(i)
    } else {
        emit::vreg_seed(i)
    }
}

/// Signed zeros, infinities, denormals and unit values: as doubles in the even registers, as
/// pairs of singles in the odd ones. Only the doubles hold a NaN; the emitted fused multiply-add
/// of singles does not keep a NaN's sign.
fn special_floats(i: u8) -> u128 {
    const SINGLES: [u32; 8] = [
        0,
        0x8000_0000,
        0x0080_0000,
        0x0000_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x3F80_0000,
        0xBF80_0000,
    ];
    const DOUBLES: [u64; 8] = [
        0,
        1 << 63,
        0x7FF8_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
    ];
    let pick = usize::from(i / 2);
    let half = |at: usize| {
        if i.is_multiple_of(2) {
            DOUBLES[at % 8]
        } else {
            u64::from(SINGLES[(at + 3) % 8]) << 32 | u64::from(SINGLES[at % 8])
        }
    };
    u128::from(half(pick + 5)) << 64 | u128::from(half(pick))
}

/// The byte at guest address `a`, distinct across load widths and signs.
fn pattern(a: u32) -> u8 {
    (splitmix(u64::from(a) ^ 0xA5A5_5A5A_1234_4321) >> 27) as u8
}

/// The guest window's starting contents, in guest order.
fn window_bytes() -> Vec<u8> {
    (0..WINDOW_BYTES).map(|a| pattern(WINDOW_AT + a)).collect()
}

/// Where each window page goes, in `guest.bin` order.
fn page_lines() -> String {
    let mut slots: Vec<u32> = (0..WINDOW_PAGES).map(pool_slot).collect();
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(
        slots.len(),
        WINDOW_PAGES as usize,
        "the scatter has to be a permutation or two guest pages share a slot"
    );
    let mut out = String::new();
    for page in 0..WINDOW_PAGES {
        let guest = WINDOW_AT + page * PAGE_BYTES;
        let wasm = POOL_AT + pool_slot(page) * PAGE_BYTES;
        let _ = writeln!(out, "page {guest} {wasm}");
    }
    out
}

fn code_bytes(program: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(program.len() * 4);
    for insn in program {
        out.extend_from_slice(&insn.to_le_bytes());
    }
    out
}

/// A `Cpu` with the window mapped and filled, `code` loaded and the watchpoint armed.
fn mapped_cpu(code: &[u8]) -> Cpu {
    let cpu = Cpu::new();
    cpu.mem
        .map_zero(WINDOW_AT, WINDOW_BYTES as usize)
        .expect("cannot map the guest window");
    cpu.mem
        .write_bytes(WINDOW_AT, &window_bytes())
        .expect("cannot fill the guest window");
    cpu.mem.map(CODE_AT, code).expect("cannot write the code");
    cpu.mem.mark_readonly(READONLY_ONE.0, READONLY_ONE.1);
    cpu.mem.mark_readonly(READONLY_TWO.0, READONLY_TWO.1);
    cpu.mem.watch_reads(READ_WATCH_AT, READ_WATCH_BYTES);
    cpu.mem.watch_writes(WATCH_AT, WATCH_BYTES);
    cpu
}

/// The window as a case starts: the shared pattern with this body's code.
fn baseline(code: &[u8]) -> Vec<u8> {
    let mut out = window_bytes();
    let at = (CODE_AT - WINDOW_AT) as usize;
    out[at..at + code.len()].copy_from_slice(code);
    out
}

/// The window bytes the block changed, as `offset:hex` runs.
fn memory_delta(cpu: &Cpu, baseline: &[u8]) -> String {
    let mut now = vec![0u8; WINDOW_BYTES as usize];
    cpu.mem
        .read_into(WINDOW_AT, &mut now)
        .expect("the window is mapped");
    let mut out = String::new();
    let mut at = 0usize;
    while at < now.len() {
        if now[at] == baseline[at] {
            at += 1;
            continue;
        }
        let start = at;
        while at < now.len() && now[at] != baseline[at] {
            at += 1;
        }
        let _ = write!(out, " {start}:");
        for b in &now[start..at] {
            let _ = write!(out, "{b:02x}");
        }
    }
    out
}

/// The encodings the emitter can write, grouped by [`form`].
fn survey() -> BTreeMap<String, Vec<u32>> {
    let mut rng = 0x243F_6A88_85A3_08D3u64;
    let mut buckets: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for prefix in 0u32..(1 << 11) {
        for _ in 0..FILLS {
            let insn = (prefix << 21) | (next(&mut rng) as u32 & 0x1F_FFFF);
            if !emits(insn) {
                continue;
            }
            // The disassembler names no Advanced SIMD form; key those by encoding.
            let text = disassemble(insn);
            let key = if text.starts_with(".word") {
                format!(".word {:08x}", insn & !(0x1F | 0x1F << 5 | 0x1F << 16))
            } else {
                form(&text)
            };
            let slot = buckets.entry(key).or_default();
            if slot.len() < PER_FORM {
                slot.push(insn);
            }
        }
    }
    // Random fills almost never land on one system register; sweep the generic
    // timer's (`MRS Xt, S3_3_C14_C0_<op2>`), which read the clock, a form each.
    for op2 in 0..8u32 {
        for rt in 0..32u32 {
            let insn = 0xD53B_E000 | op2 << 5 | rt;
            if !emits(insn) {
                continue;
            }
            let slot = buckets
                .entry(format!("mrs x_, s3_3_c14_c0_{op2}"))
                .or_default();
            if slot.len() < PER_FORM {
                slot.push(insn);
            }
        }
    }
    buckets
}

/// How many register-content sets each block is run under.
const SEEDS: usize = 12;

/// The set whose vector registers hold zeros, infinities and NaNs instead of noise.
const SPECIALS_SET: usize = SEEDS - 1;

/// Arm the monitor for an exclusive load or store at the address its base register holds:
/// not at all, with what memory holds, or with something else, by seed set.
fn arm_monitor(cpu: &mut Cpu, insn: u32, set: usize) {
    let exclusive = matches!((insn >> 21) & 0x1FF, 0b001000000 | 0b001000010);
    let rn = ((insn >> 5) & 0x1F) as usize;
    if !exclusive || rn == 31 || set.is_multiple_of(3) {
        cpu.set_exclusive(false, 0, 0);
        return;
    }
    let addr = cpu.reg_slots()[rn] as u32;
    let held = match (insn >> 30) & 0b11 {
        0 => cpu.mem.read_u8(addr).map(u64::from),
        1 => cpu.mem.read_u16(addr).map(u64::from),
        2 => cpu.mem.read_u32(addr).map(u64::from),
        _ => cpu.mem.read_u64(addr),
    };
    match held {
        Ok(value) => cpu.set_exclusive(true, addr, if set % 3 == 1 { value } else { value ^ 1 }),
        Err(_) => cpu.set_exclusive(false, 0, 0),
    }
}

/// Seed flags first (the preamble clobbers `X0`), then the register file.
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

/// Register `i` under set `set`: adversarial seeds for `SDIV` overflow and
/// shift amounts around 32 and 64.
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
        7 | 11 => splitmix(0x9E37_79B9_7F4A_7C15 ^ i),
        8 => splitmix(0xD1B5_4A32_D192_ED03 ^ i),
        9 => {
            if even {
                splitmix(i)
            } else {
                i % 65
            }
        }
        // The last bytes of any page but the window's last, so accesses straddle
        // into a mapped page.
        _ => {
            let page = i % u64::from(WINDOW_PAGES - 1);
            (page + 1) * u64::from(PAGE_BYTES) - i % 9
        }
    }
}

/// SplitMix64, so runs are reproducible.
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
