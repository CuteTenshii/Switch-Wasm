//! Emitted wasm against the interpreter on adversarial operands and memory:
//! `emit_selftest`. Needs no title or keys, so it runs in CI.
//!
//! Encodings are swept from [`switch_core::cpu::emits`], never hand-assembled.
//!
//! ```text
//! cargo run --release --example emit_selftest
//! node tools/emit_difftest.mjs target/emit-selftest
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use switch_core::cpu::{defers, emits, Cpu, Layout, DISCARD_SLOT};
use switch_core::disasm::disassemble;

/// Guest state offsets in module memory, shared with `emit_difftest`.
const REGS_AT: u32 = 0;
const NZCV_AT: u32 = 0x1000;
const READ_WATCH_LO_AT: u32 = 0x1004;
const READ_WATCH_HI_AT: u32 = 0x1008;
/// Where the pointer to the page table is (a real `Memory` boxes its table).
const PAGES_AT: u32 = 0x100C;
const WATCH_LO_AT: u32 = 0x1010;
const WATCH_HI_AT: u32 = 0x1014;
const READONLY_LO_AT: u32 = 0x1018;
const READONLY_HI_AT: u32 = 0x101C;
/// Where the pointer to the watched-page bitmap is.
const WATCHED_AT: u32 = 0x1020;
/// Where the guest `pc` is; unused here but required by `Layout`.
const PC_AT: u32 = 0x1024;

/// The page table: one four-byte entry per 4 KiB of guest space, zero if unmapped.
const TABLE_AT: u32 = 0x0010_0000;
const TABLE_BYTES: u32 = (1 << 20) * 4;

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

/// Two write-protected ranges with a writable gap, to exercise the envelope test.
const READONLY_ONE: (u32, u32) = (5 * PAGE_BYTES, 6 * PAGE_BYTES);
const READONLY_TWO: (u32, u32) = (8 * PAGE_BYTES, 9 * PAGE_BYTES);

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
                    seed(&mut cpu, set, nzcv);

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
                    cpu.mem.dirty_code_pages();
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
                        || !cpu.mem.dirty_code_pages().is_empty();

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
         bitmap_at {BITMAP_AT}\n\
         slots {}\n\
         code_at {CODE_AT}\n\
         read_watch {READ_WATCH_AT} {}\n\
         watch {WATCH_AT} {}\n\
         readonly {} {}\n\
         watched_page {WATCHED_PAGE_AT}\n\
         wasm_pages {}\n\
         {}",
        Cpu::new().reg_slots().len(),
        READ_WATCH_AT + READ_WATCH_BYTES,
        WATCH_AT + WATCH_BYTES,
        READONLY_ONE.0,
        READONLY_TWO.1,
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
    let mut cpu = Cpu::new();
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
            let slot = buckets.entry(form(&disassemble(insn))).or_default();
            if slot.len() < PER_FORM {
                slot.push(insn);
            }
        }
    }
    buckets
}

/// How many register-content sets each block is run under.
const SEEDS: usize = 11;

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
        7 => splitmix(0x9E37_79B9_7F4A_7C15 ^ i),
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
