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
//! # Guest memory
//!
//! It also owns the only guest memory either difftest has. `emit_difftest`
//! runs a title, whose memory is hundreds of megabytes that cannot be handed
//! to a module, so every page there is unmapped and every access hands its
//! instruction back. Here the window is seventeen pages this file lays out
//! itself, put low so that the register seeds are addresses in it, filled with
//! a pattern that differs at every width and sign a load comes in, and watched
//! over a range of it. So the three things an emitted access declines are all
//! reachable: a page that is not there, an access with half of itself on the
//! next page, and one the watchpoint covers.
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
use switch_core::cpu::{defers, emits, Cpu, Layout};
use switch_core::disasm::disassemble;

/// Where the harness puts guest state in the memory it hands a module, the
/// same offsets `emit_difftest` uses, so one manifest reader serves both.
const REGS_AT: u32 = 0;
const NZCV_AT: u32 = 0x1000;
const READ_WATCH_LO_AT: u32 = 0x1004;
const READ_WATCH_HI_AT: u32 = 0x1008;
/// Where the *pointer* to the page table is, which is what an emitted access
/// loads before it can index one: a real `Memory` holds its table behind a
/// `Box`, so the harness has to present it the same way.
const PAGES_AT: u32 = 0x100C;
const WATCH_LO_AT: u32 = 0x1010;
const WATCH_HI_AT: u32 = 0x1014;
const READONLY_LO_AT: u32 = 0x1018;
const READONLY_HI_AT: u32 = 0x101C;
/// Where the pointer to the watched-page bitmap is.
const WATCHED_AT: u32 = 0x1020;

/// The page table itself: one four-byte entry per 4 KiB of the guest's 4 GiB,
/// holding where in this memory that page lives and zero where it is not
/// mapped. Four megabytes of mostly zeroes, exactly as the emulator's own is.
const TABLE_AT: u32 = 0x0010_0000;
const TABLE_BYTES: u32 = (1 << 20) * 4;

/// Where the mapped guest pages live, one 4 KiB slot each.
const POOL_AT: u32 = TABLE_AT + TABLE_BYTES;

/// The bitmap of pages something has cached the contents of: one bit per page
/// of the guest's 4 GiB, which a store tests before it can be written inline.
const BITMAP_AT: u32 = POOL_AT + WINDOW_BYTES;
const BITMAP_BYTES: u32 = (1 << 20) / 8;

/// Which slot the `n`th page of the window goes in.
///
/// Scattered rather than laid out in guest order, because a real `Memory`'s
/// pages are separate allocations and nothing puts the page above one next to
/// it. In guest order an access that runs off the end of its page finds its
/// guest neighbour there anyway and reads exactly the bytes it should have, so
/// the boundary check every emitted access makes could be deleted and not one
/// case would notice. [`WINDOW_PAGES`] is prime, so stepping through the slots
/// like this reaches every one of them.
const SCATTER: u32 = 7;
fn pool_slot(page: u32) -> u32 {
    (page * SCATTER) % WINDOW_PAGES
}

/// The guest window every case runs against: a single run of pages starting at
/// zero, which is what makes the existing seed sets useful as addresses. A
/// register holding a small integer is then a valid guest address, and one
/// holding `u64::MAX` or a random word still is not.
const WINDOW_AT: u32 = 0;
const WINDOW_PAGES: u32 = 17;
const PAGE_BYTES: u32 = 4096;
const WINDOW_BYTES: u32 = WINDOW_PAGES * PAGE_BYTES;

/// Where the test program is mapped, inside that window. One page, and a block
/// never spans one.
const CODE_AT: u32 = 0x1000;

/// The read watchpoint, armed for every case rather than disarmed.
///
/// A watched read is one of the three things an emitted access declines and
/// leaves to the interpreter, and it is the only one of the three that a
/// disarmed harness can never reach: an unmapped page and a straddled boundary
/// both fall out of the seeds on their own. Sitting it low in the window puts
/// it where the small-integer seed sets address.
const READ_WATCH_AT: u32 = 0x40;
const READ_WATCH_BYTES: u32 = 0x40;

/// The write watchpoint, armed over a different range so a store and a load to
/// the same address do not both hand back for the same reason.
const WATCH_AT: u32 = 0xC0;
const WATCH_BYTES: u32 = 0x40;

/// The page whose contents something has cached, so a store there owes a
/// report and cannot be written inline. Re-armed before every case, because
/// the interpreter clears the bit as it makes that report.
const WATCHED_PAGE_AT: u32 = 3 * PAGE_BYTES;

/// Two write-protected ranges with a page of writable memory between them.
///
/// Two and not one on purpose. An emitted store tests the *envelope* of the
/// protected ranges and hands back anything inside it, where the interpreter
/// walks the list and finds the gap writable, so the gap is the only place
/// that test can be seen being made at all: a store to a range that really is
/// protected faults under the interpreter and has no answer to compare with.
const READONLY_ONE: (u32, u32) = (5 * PAGE_BYTES, 6 * PAGE_BYTES);
const READONLY_TWO: (u32, u32) = (8 * PAGE_BYTES, 9 * PAGE_BYTES);

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
/// loads and never wrote a 64-bit one: the widest load there is went untested,
/// and so did every sign-extending load with a 32-bit destination.
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
    std::fs::write(format!("{out_dir}/guest.bin"), window_bytes())
        .expect("cannot write the window");

    let mut manifest = String::new();
    let mut cases = 0usize;
    let mut faulted = 0usize;
    let mut refused = 0usize;
    let mut handed_back = 0usize;
    let mut pinned = 0usize;
    let all_flags: Vec<u8> = (0u8..16).collect();

    for (form, encodings) in &buckets {
        // An instruction that can hand itself back gets a block to itself, so
        // that what `run` reports is either the whole block or none of it and
        // the two register files the manifest already carries are the only two
        // answers to check it against. The rest keep sharing a block, which is
        // what makes an emitted body long enough to be worth compiling.
        let bucket_hands_back = encodings.iter().copied().any(defers);
        let bodies: Vec<&[u32]> = if bucket_hands_back {
            encodings.iter().map(std::slice::from_ref).collect()
        } else {
            vec![encodings.as_slice()]
        };
        // NZCV decides nothing about an access, and a block of one instruction
        // would otherwise be written out sixteen times over.
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

            // The code page is part of the window a load can read, so what is
            // in it belongs to the state a case starts from rather than to the
            // one copy of the window every case shares.
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
                    let Ok((module, ops)) = cpu.emit_block_at(block_at, LAYOUT) else {
                        refused += 1;
                        continue;
                    };

                    // Everything the seeding and the last case left behind, so
                    // that what is there afterwards belongs to this block. The
                    // window goes back first, because a `Cpu` outlives the
                    // cases run on it and a store from one of them would
                    // otherwise still be sitting in guest memory, where it
                    // would be read as this block's doing. Through `map`,
                    // which is the loader's path and so pays no attention to
                    // the ranges marked read-only below.
                    //
                    // The cached page is armed again as part of the same
                    // reset: the interpreter clears its bit on the way to
                    // reporting it, and a case that found it already cleared
                    // would be testing nothing.
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

                    // Four of the things an emitted access hands back for
                    // leave the guest state the interpreter would have left
                    // anyway, so nothing compared so far would notice if the
                    // emitted one went ahead instead. Each leaves a mark,
                    // though, and each mark comes from the mechanism itself
                    // rather than from a second copy of its test.
                    //
                    // A fault is the widest of them. Everything the
                    // interpreter cannot finish an access for -- a page with
                    // no storage, a write-protected address -- is something an
                    // emitted access hands back for, so a case the interpreter
                    // faulted on is one where the answer is known exactly: the
                    // block retired none of itself and changed nothing. Those
                    // used to be thrown away, and they are the only cases that
                    // can see the protection test being made at all: a store
                    // inside the protected ranges is the one address where
                    // going ahead anyway would write where nothing may, and
                    // everywhere else the two engines agree whether the test
                    // is there or not.
                    //
                    // The other three are a watched read, a watched write, and
                    // a store landing on a page whose contents something has
                    // cached, which is a report an emitted store cannot make.
                    let must_hand_back = fault
                        || cpu.mem.take_read_hit().is_some()
                        || cpu.mem.take_watch_hit().is_some()
                        || !cpu.mem.dirty_code_pages().is_empty();

                    let name = format!("case{cases:05}");
                    std::fs::write(format!("{out_dir}/{name}.wasm"), &module)
                        .expect("cannot write a module");
                    let mode = match (hands_back, must_hand_back) {
                        (false, _) => "exact",
                        (true, false) => "maybe",
                        (true, true) => {
                            pinned += 1;
                            "none"
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
         {refused} blocks the emitter refused, {faulted} that faulted)",
        buckets.len()
    );
    println!("now run: node tools/emit_difftest.mjs {out_dir}");
}

/// Where the harness has put guest state in the memory it hands a module.
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

/// What the byte at guest address `a` holds.
///
/// Anything a load reads has to differ at every width and sign a load comes
/// in. Against a window of zeroes an `LDRB` and an `LDRSW` of the wrong
/// address agree with the right one, and so does a load the emitter wrote one
/// size too narrow.
fn pattern(a: u32) -> u8 {
    (splitmix(u64::from(a) ^ 0xA5A5_5A5A_1234_4321) >> 27) as u8
}

/// The guest window's contents, in guest order, which every case starts from
/// and only its code page differs in.
fn window_bytes() -> Vec<u8> {
    (0..WINDOW_BYTES).map(|a| pattern(WINDOW_AT + a)).collect()
}

/// Where each page of the window goes, in guest order, which is the order
/// `guest.bin` holds them in.
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

/// A program as the bytes it is mapped as.
fn code_bytes(program: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(program.len() * 4);
    for insn in program {
        out.extend_from_slice(&insn.to_le_bytes());
    }
    out
}

/// A `Cpu` with the guest window mapped and filled, `code` in its code page
/// and the read watchpoint armed: the state the harness mirrors into the
/// memory it hands a module.
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

/// The window as a case starts from it: the shared pattern with this body's
/// code laid over it.
fn baseline(code: &[u8]) -> Vec<u8> {
    let mut out = window_bytes();
    let at = (CODE_AT - WINDOW_AT) as usize;
    out[at..at + code.len()].copy_from_slice(code);
    out
}

/// The bytes of the window the block changed, as `offset:hex` runs.
///
/// A store writes at most eight bytes, so this is a token or two, and it is
/// what says a store landed where the interpreter put it rather than merely
/// leaving the registers right.
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
const SEEDS: usize = 11;

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
        9 => {
            if even {
                splitmix(i)
            } else {
                i % 65
            }
        }
        // The last few bytes of a page, which is where an access finds the
        // rest of itself somewhere the page table has to be asked about
        // separately. Nothing else reaches it: an address is only in the top
        // eight bytes of its page seven times in four thousand, and left to
        // the other sets a whole sweep straddled a boundary once.
        //
        // Below the *last* page of the window, so the page the access runs
        // into is one that exists and the interpreter has an answer to compare
        // against rather than a fault.
        _ => {
            let page = i % u64::from(WINDOW_PAGES - 1);
            (page + 1) * u64::from(PAGE_BYTES) - i % 9
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
