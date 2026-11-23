//! `emit_selftest` and `emit_difftest` record cases against this layout;
//! `tools/emit_difftest.mjs` reads it back from the manifest header.

use std::fmt::Write as _;
use switch_core::cpu::{Cpu, Layout, DISCARD_SLOT};

pub const REGS_AT: u32 = 0;
pub const NZCV_AT: u32 = 0x1000;
/// Where the pointer to the top-level page table is (a real `Memory` boxes its table).
pub const BLOCKS_AT: u32 = 0x100C;
/// The read and write watchpoints, each a `u64` `[start, end)`.
pub const READ_WATCH_LO_AT: u32 = 0x1090;
pub const READ_WATCH_HI_AT: u32 = 0x1098;
pub const WATCH_LO_AT: u32 = 0x10A0;
pub const WATCH_HI_AT: u32 = 0x10A8;
/// The byte that serialises 16-byte exclusive stores, zero while free.
pub const PAIR_LOCK_AT: u32 = 0x1080;
/// Guest `pc`, written by a taken branch.
pub const PC_AT: u32 = 0x1024;
/// Fuel, read only by branches between blocks, which a single block never takes.
pub const FUEL_AT: u32 = 0x1028;
pub const FPSR_AT: u32 = 0x102C;
/// FPCR, whose rounding mode FRINTX and FRINTI read.
pub const FPCR_AT: u32 = 0x1084;
/// The flags `VCMP` leaves in the FPSCR.
pub const FPSCR_NZCV_AT: u32 = 0x1088;
/// A zero word: the `Memory` address, so the offsets above are absolute.
pub const MEMORY_AT: u32 = 0x1030;
/// The exclusive monitor: the `u64` address, the armed flag, then two doublewords at 16.
pub const EXCLUSIVE_AT: u32 = 0x1040;
/// `TPIDRRO_EL0` and `TPIDR_EL0`, past the monitor.
pub const TPIDR_AT: u32 = 0x1060;
pub const TPIDR_RW_AT: u32 = 0x1068;
/// The core number (zero, so the clock is the `Cpu`'s own), the published
/// clock's pointer and the chain's clock origin.
pub const CORE_AT: u32 = 0x1070;
pub const CLOCK_AT_AT: u32 = 0x1074;
pub const ORIGIN_AT: u32 = 0x1078;
/// The `[u128; 32]` vector register file.
pub const VREGS_AT: u32 = 0x2000;

/// The top-level page table: one four-byte entry per 2 MiB block, never zero.
pub const TABLE_AT: u32 = 0x0010_0000;
pub const TABLE_BYTES: u32 = (1 << 18) * 4;
/// A block's page entries (four bytes per 4 KiB page, zero if unmapped) and its cached-pages
/// bitmap, at a real block's wasm32 offsets; blocks are laid out a stride apart.
pub const BLOCK_PAGES: u32 = 0;
pub const BLOCK_WATCHED: u32 = 512 * 4;
pub const BLOCK_STRIDE: u32 = 0x1000;
/// The block every entry starts at, and the one the cases' window, below 2 MiB, is in.
pub const EMPTY_BLOCK_AT: u32 = TABLE_AT + TABLE_BYTES;
pub const WINDOW_BLOCK_AT: u32 = EMPTY_BLOCK_AT + BLOCK_STRIDE;
/// Where the harnesses put mapped guest pages, one 4 KiB slot each.
pub const POOL_AT: u32 = WINDOW_BLOCK_AT + BLOCK_STRIDE;

pub const LAYOUT: Layout = Layout {
    regs: REGS_AT,
    nzcv: NZCV_AT,
    fpsr: FPSR_AT,
    fpcr: FPCR_AT,
    fpscr_nzcv: FPSCR_NZCV_AT,
    memory: MEMORY_AT,
    blocks: BLOCKS_AT,
    block_pages: BLOCK_PAGES,
    block_watched: BLOCK_WATCHED,
    read_watch_lo: READ_WATCH_LO_AT,
    read_watch_hi: READ_WATCH_HI_AT,
    watch_lo: WATCH_LO_AT,
    watch_hi: WATCH_HI_AT,
    pair_lock: PAIR_LOCK_AT,
    pc: PC_AT,
    fuel: FUEL_AT,
    vregs: VREGS_AT,
    exclusive: EXCLUSIVE_AT,
    tpidr: TPIDR_AT,
    tpidr_rw: TPIDR_RW_AT,
    core: CORE_AT,
    clock_at: CLOCK_AT_AT,
    origin: ORIGIN_AT,
};

/// Vector registers start each case at these, so a wrong register or half shows.
pub fn vreg_seed(i: u8) -> u128 {
    let lo = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(2 * u64::from(i) + 1);
    let hi = 0xC2B2_AE3D_27D4_EB4Fu64.wrapping_mul(2 * u64::from(i) + 2);
    u128::from(hi) << 64 | u128::from(lo)
}

/// [`vreg_seed`], IEEE special values for the specials set, or ordinary doubles for the
/// doubles set.
pub fn kind_seed(i: u8, kind: Option<&str>) -> u128 {
    match kind {
        Some("specials") => special_floats(i, true),
        Some("infinities") => special_floats(i, false),
        Some("doubles") => {
            u128::from(mixed_double(2 * u64::from(i) + 1)) << 64
                | u128::from(mixed_double(2 * u64::from(i)))
        }
        _ => vreg_seed(i),
    }
}

/// A double with few significant bits and a small exponent, so that sums and products of
/// several of them land on exact ties and near-cancellations where noise never would.
fn mixed_double(n: u64) -> u64 {
    let mut h = (n + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    let bits = 1 + (h >> 1) % 53;
    let significand = ((h >> 8) & ((1 << bits) - 1)) | (1 << (bits - 1));
    let scale = f64::from_bits((1023 - 60 + (h >> 40) % 121) << 52);
    (significand as f64 * scale).to_bits() | (h & 1) << 63
}

/// Signed zeros, infinities, denormals and unit values: as doubles in the even registers, as
/// pairs of singles in the odd ones. Only the doubles hold a NaN, and not in `infinities`; the emitted fused multiply-add
/// of singles does not keep a NaN's sign.
fn special_floats(i: u8, nan: bool) -> u128 {
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
    let doubles: [u64; 8] = [
        0,
        1 << 63,
        if nan {
            0x7FF8_0000_0000_0000
        } else {
            0x7FEF_FFFF_FFFF_FFFF
        },
        0x0000_0000_0000_0001,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
    ];
    let pick = usize::from(i / 2);
    let half = |at: usize| {
        if i.is_multiple_of(2) {
            doubles[at % 8]
        } else {
            u64::from(SINGLES[(at + 3) % 8]) << 32 | u64::from(SINGLES[at % 8])
        }
    };
    u128::from(half(pick + 5)) << 64 | u128::from(half(pick))
}

/// ` vN:hex` for each vector register the case changed from its seed.
pub fn vreg_delta(cpu: &Cpu, seed: impl Fn(u8) -> u128) -> String {
    let mut out = String::new();
    for i in 0..32 {
        let v = cpu.read_vreg(i);
        if v != seed(i) {
            let _ = write!(out, " v{i}:{v:032x}");
        }
    }
    out
}

/// The manifest header lines both harnesses write.
pub fn header() -> String {
    format!(
        "regs_at {REGS_AT}\n\
         nzcv_at {NZCV_AT}\n\
         fpsr_at {FPSR_AT}\n\
         fpcr_at {FPCR_AT}\n\
         fpscr_nzcv_at {FPSCR_NZCV_AT}\n\
         read_watch_lo_at {READ_WATCH_LO_AT}\n\
         read_watch_hi_at {READ_WATCH_HI_AT}\n\
         watch_lo_at {WATCH_LO_AT}\n\
         watch_hi_at {WATCH_HI_AT}\n\
         blocks_at {BLOCKS_AT}\n\
         block_pages {BLOCK_PAGES}\n\
         block_watched {BLOCK_WATCHED}\n\
         empty_block_at {EMPTY_BLOCK_AT}\n\
         window_block_at {WINDOW_BLOCK_AT}\n\
         discard_slot {DISCARD_SLOT}\n\
         pc_at {PC_AT}\n\
         vregs_at {VREGS_AT}\n\
         exclusive_at {EXCLUSIVE_AT}\n\
         tpidr_at {TPIDR_AT}\n\
         tpidr_rw_at {TPIDR_RW_AT}\n\
         origin_at {ORIGIN_AT}\n\
         table_at {TABLE_AT}\n\
         slots {}\n",
        Cpu::new().reg_slots().len(),
    )
}
