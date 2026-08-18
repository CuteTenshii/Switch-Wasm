//! Nintendo homebrew NRO loader.
//!
//! NRO is the binary format produced by devkitA64 for Switch homebrew. It is
//! a simple container: a 0x40/0x50 byte header followed by three segments
//! (`.text`, `.rodata`, `.data`) that are loaded contiguously, plus a BSS
//! region zero-filled at the end.
//!
//! Header layout (offsets relative to the file start):
//!
//! ```text
//! 0x00  magic "NRO0"
//! 0x04  version (0, 1 or 2)
//! 0x08  total NRO size
//! 0x0C  flags
//! 0x10  text: u32 offset, u32 size
//! 0x18  ro:   u32 offset, u32 size
//! 0x20  data: u32 offset, u32 size
//! 0x28  bss size (u32)
//! 0x2C  reserved
//! 0x30  build id [0x20]
//! 0x50  (version 2 only) "NRO2" header
//! ```

use crate::mem::Memory;
use crate::nsp::read_u32;
use crate::{Error, Result};

pub const NRO0_MAGIC: u32 = 0x304f524e; // "NRO0"
pub const NRO2_MAGIC: u32 = 0x32524f4e; // "NRO2"
/// Base address where the NRO image is mapped.
///
/// Real homebrew (devkitA64/libnx) is linked against `0x08000000`, the load
/// address the Homebrew Loader (HBL) uses, so baked-in absolute pointers in
/// the image assume that base. Loading anywhere else would make those
/// pointers dangle.
pub const NRO_BASE: u32 = 0x0800_0000;
const HEADER_MIN: usize = 0x50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NroHeader {
    /// File offset where the "NRO0" magic lives (may be nonzero after a boot
    /// stub preamble).
    pub magic_offset: u32,
    pub version: u32,
    pub nro_size: u32,
    pub flags: u32,
    pub text_offset: u32,
    pub text_size: u32,
    pub ro_offset: u32,
    pub ro_size: u32,
    pub data_offset: u32,
    pub data_size: u32,
    pub bss_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadedNro {
    pub base: u32,
    pub entry: u32,
    pub text: Segment,
    pub ro: Segment,
    pub data: Segment,
    pub bss_size: u32,
    pub build_id: [u8; 0x20],
    pub is_64bit: bool,
    /// Address of the synthesized homebrew environment block to pass as the
    /// crt0's `x0` (0 for NROs whose crt0 doesn't parse one).
    pub env_addr: u32,
}

/// Where [`setup_env_block`] maps the environment block. Kept out of the
/// loaded image so the crt0's BSS zeroing never touches it.
pub const ENV_BLOCK_ADDR: u32 = 0x0010_0000;

/// Path the loaded NRO is presented at on the emulated SD card. libnx's
/// `romfsMountSelf` re-opens the running NRO by `argv[0]` to read the RomFS
/// appended after its image, so the file has to exist and the path has to
/// match what the environment block advertises.
pub const HOMEBREW_NRO_PATH: &str = "sdmc:/switch/homebrew.nro";

/// Entry keys of the libnx homebrew ABI (`nx/source/runtime/env.h`).
const ENTRY_MAIN_THREAD_HANDLE: u32 = 1;
const ENTRY_NEXT_LOAD_PATH: u32 = 2;
const ENTRY_ARGV: u32 = 5;
const ENTRY_HOS_VERSION: u32 = 16;
const ENTRY_END_OF_LIST: u32 = 0;

/// Where the environment block keeps the "next NRO to run" buffers that
/// `EntryType_NextLoadPath` points at. A menu writes the path it wants launched
/// here and exits; hbmenu's `launchInit()` refuses to start without them.
pub const NEXT_LOAD_PATH_ADDR: u32 = ENV_BLOCK_ADDR + 0x400;
pub const NEXT_LOAD_ARGV_ADDR: u32 = ENV_BLOCK_ADDR + 0x800;
/// Size of each of those buffers, as hbloader sizes them.
pub const NEXT_LOAD_BUFFER_SIZE: usize = 0x300;

/// Write a minimal homebrew ABI environment block so `envSetup` in the crt0
/// populates its runtime globals. libnx's `EntryType_HosVersion` handler
/// stores `Value[0]` as the host version and, when `Value[1]` is the
/// `'ATMOSPHR'` magic, keeps it as-is. `0xFFFFFFFF` reads as "current
/// firmware", which the version gates accept. Returns [`ENV_BLOCK_ADDR`].
///
/// The `ConfigEntry` here is the 24-byte form used by the linked crt0
/// (`u32 Key, u32 Flags, u64 Value[2]`).
pub fn setup_env_block(mem: &mut Memory) -> u32 {
    let a = ENV_BLOCK_ADDR;
    // MainThreadHandle (Key 1): required by __libnx_init_thread.
    let _ = mem.write_u32(a, ENTRY_MAIN_THREAD_HANDLE);
    let _ = mem.write_u32(a + 4, 0);
    let _ = mem.write_u64(a + 8, 1);
    let _ = mem.write_u64(a + 16, 0);
    // HosVersion (Key 16 in this crt0): { version = 0xFFFFFFFF, "ATMOSPHR" }.
    let _ = mem.write_u32(a + 24, ENTRY_HOS_VERSION);
    let _ = mem.write_u32(a + 28, 0);
    let _ = mem.write_u64(a + 32, 0xFFFF_FFFF);
    let _ = mem.write_u64(a + 40, 0x4154_4D4F_5350_4852); // "ATMOSPHR"
    // Argv (Key 5): Value[1] points at the command line, which libnx splits
    // into argv. argv[0] is how `romfsMountSelf` finds the running NRO.
    const ARGV_STRING_OFFSET: u32 = 0x100;
    let _ = mem.write_u32(a + 48, ENTRY_ARGV);
    let _ = mem.write_u32(a + 52, 0);
    let _ = mem.write_u64(a + 56, 0);
    let _ = mem.write_u64(a + 64, (a + ARGV_STRING_OFFSET) as u64);
    // NextLoadPath (Key 2): { path buffer, argv buffer }. A homebrew menu is
    // expected to write what it wants launched next into these and exit — and
    // hbmenu's `launchInit()` fails outright when the loader doesn't offer them.
    let _ = mem.write_u32(a + 72, ENTRY_NEXT_LOAD_PATH);
    let _ = mem.write_u32(a + 76, 0);
    let _ = mem.write_u64(a + 80, u64::from(NEXT_LOAD_PATH_ADDR));
    let _ = mem.write_u64(a + 88, u64::from(NEXT_LOAD_ARGV_ADDR));
    let _ = mem.map_zero(NEXT_LOAD_PATH_ADDR, NEXT_LOAD_BUFFER_SIZE);
    let _ = mem.map_zero(NEXT_LOAD_ARGV_ADDR, NEXT_LOAD_BUFFER_SIZE);
    // EndOfList
    let _ = mem.write_u32(a + 96, ENTRY_END_OF_LIST);
    let _ = mem.write_u32(a + 100, 0);
    let _ = mem.write_u64(a + 104, 0);
    let _ = mem.write_u64(a + 112, 0);

    let path = HOMEBREW_NRO_PATH.as_bytes();
    for (i, &byte) in path.iter().enumerate() {
        let _ = mem.write_u8(a + ARGV_STRING_OFFSET + i as u32, byte);
    }
    let _ = mem.write_u8(a + ARGV_STRING_OFFSET + path.len() as u32, 0);
    a
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub file_offset: u32,
    pub file_size: u32,
    pub mem_addr: u32,
}

impl NroHeader {
    /// Parse an NRO header. The `NRO0` magic may appear after a small
    /// preamble (some builds prepend a boot stub, e.g. hbmenu); we scan the
    /// first 0x100 bytes for it.
    pub fn parse(data: &[u8]) -> Result<NroHeader> {
        if data.len() < HEADER_MIN {
            return Err(Error::Truncated {
                what: "NRO header".into(),
                expected: HEADER_MIN,
                got: data.len(),
            });
        }
        let magic_at = find_magic(data, NRO0_MAGIC).ok_or(Error::BadMagic {
            what: "NRO".into(),
            found: crate::nsp::read_u32(data, 0),
        })?;
        let h = magic_at;
        let magic = crate::nsp::read_u32(data, h);
        if magic != NRO0_MAGIC {
            return Err(Error::BadMagic {
                what: "NRO".into(),
                found: magic,
            });
        }
        let version = crate::nsp::read_u32(data, h + 0x04);
        if version > 2 {
            return Err(Error::Nro(format!("unsupported NRO version {}", version)));
        }
        let text_offset = crate::nsp::read_u32(data, h + 0x10);
        let text_size = crate::nsp::read_u32(data, h + 0x14);
        let ro_offset = crate::nsp::read_u32(data, h + 0x18);
        let ro_size = crate::nsp::read_u32(data, h + 0x1C);
        let data_offset = crate::nsp::read_u32(data, h + 0x20);
        let data_size = crate::nsp::read_u32(data, h + 0x24);

        let nro_size = crate::nsp::read_u32(data, h + 0x08);
        if nro_size as usize > data.len() {
            return Err(Error::Nro(format!(
                "header claims {} bytes but file has {}",
                nro_size,
                data.len()
            )));
        }
        for (name, off, size) in [
            ("text", text_offset, text_size),
            ("ro", ro_offset, ro_size),
            ("data", data_offset, data_size),
        ] {
            if off as usize + size as usize > nro_size as usize {
                return Err(Error::Nro(format!(
                    "{} segment [{:#x}, {:#x}) out of bounds (nro_size {:#x})",
                    name, off, off + size, nro_size
                )));
            }
        }

        Ok(NroHeader {
            magic_offset: h as u32,
            version,
            nro_size,
            flags: crate::nsp::read_u32(data, h + 0x0C),
            text_offset,
            text_size,
            ro_offset,
            ro_size,
            data_offset,
            data_size,
            bss_size: crate::nsp::read_u32(data, h + 0x28),
        })
    }

    pub fn is_64bit(&self, data: &[u8]) -> bool {
        if self.version >= 2 && data.len() >= self.magic_offset as usize + 0x64 {
            let is_64 = data[self.magic_offset as usize + 0x60];
            if is_64 == 0 {
                return false;
            }
        }
        true
    }

    pub fn build_id(&self, data: &[u8]) -> [u8; 0x20] {
        let mut id = [0u8; 0x20];
        let start = self.magic_offset as usize + 0x30;
        let end = (start + 0x20).min(data.len());
        id[..end - start].copy_from_slice(&data[start..end]);
        id
    }
}

fn find_magic(data: &[u8], magic: u32) -> Option<usize> {
    let magic_le = magic.to_le_bytes();
    data[..data.len().min(0x100)]
        .windows(4)
        .position(|w| w == &magic_le[..])
}

fn read_u64(data: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    let n = (data.len().saturating_sub(off)).min(8);
    b[..n].copy_from_slice(&data[off..off + n]);
    u64::from_le_bytes(b)
}

fn read_cstr(data: &[u8], off: usize) -> &str {
    if off >= data.len() {
        return "";
    }
    let end = data[off..]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(data.len() - off);
    std::str::from_utf8(&data[off..off + end]).unwrap_or("")
}

/// ELF hash used by the legacy DT_HASH symbol table.
fn elf_hash(name: &str) -> u32 {
    let mut h: u32 = 0;
    for &b in name.as_bytes() {
        h = (h << 4).wrapping_add(b as u32);
        let g = h & 0xF000_0000;
        if g != 0 {
            h ^= g >> 24;
            h &= !g;
        }
    }
    h
}

/// Look up a symbol in the NRO's DT_HASH / DT_SYMTAB / DT_STRTAB dynamic
/// tables. Returns the symbol's `st_value` (a file-offset relative to the
/// load base) when found. This is intentionally simple: it only supports the
/// legacy DT_HASH layout used by devkitA64/libtransistor NROs.
pub fn symbol_value(data: &[u8], name: &str) -> Option<u64> {
    let mod0 = find_mod0(data)?;
    let dyn_off = mod0.wrapping_add(read_u32(data, mod0 + 4) as usize);
    let mut symtab = 0u64;
    let mut strtab = 0u64;
    let mut hash_off = 0u64;
    let mut off = dyn_off;
    while off + 16 <= data.len() {
        let tag = read_u64(data, off);
        let val = read_u64(data, off + 8);
        off += 16;
        if tag == 0 {
            break;
        }
        match tag {
            0x06 => symtab = val, // DT_SYMTAB
            0x05 => strtab = val, // DT_STRTAB
            0x04 => hash_off = val, // DT_HASH
            _ => {}
        }
    }
    if symtab == 0 || strtab == 0 || hash_off == 0 {
        return None;
    }
    let symtab = symtab as usize;
    let strtab = strtab as usize;
    let hash_off = hash_off as usize;
    if hash_off + 8 > data.len() {
        return None;
    }
    let nbucket = read_u32(data, hash_off) as usize;
    let nchain = read_u32(data, hash_off + 4) as usize;
    let buckets_off = hash_off + 8;
    let chains_off = buckets_off.checked_add(4 * nbucket)?;
    if buckets_off.checked_add(4 * nbucket)? > data.len()
        || chains_off.checked_add(4 * nchain)? > data.len()
    {
        return None;
    }

    let h = elf_hash(name) as usize;
    let mut idx = read_u32(data, buckets_off + 4 * (h % nbucket)) as usize;
    while idx != 0 {
        let sym_off = symtab.checked_add(idx * 24)?;
        if sym_off + 24 > data.len() {
            break;
        }
        let name_off = read_u32(data, sym_off) as usize;
        if read_cstr(data, strtab + name_off) == name {
            return Some(read_u64(data, sym_off + 8));
        }
        idx = read_u32(data, chains_off + 4 * idx) as usize;
    }
    // Bucket 0 may legitimately point to symbol index 0, so check it too.
    if idx == 0 {
        let sym_off = symtab;
        if sym_off + 24 <= data.len() {
            let name_off = read_u32(data, sym_off) as usize;
            if read_cstr(data, strtab + name_off) == name {
                return Some(read_u64(data, sym_off + 8));
            }
        }
    }
    None
}

/// libtransistor NROs can ship with `_trn_runconf_heap_mode` set to OVERRIDE
/// with a tiny or zero-sized heap. Without a real loader config that leaves
/// `_sbrk_r` returning NULL on the first allocation. We force the runtime into
/// NORMAL heap mode so it calls `svcSetHeapSize`, which the emulator stubs.
///
/// The active `_trn_runconf_heap_mode` is not always the weak symbol exported
/// in the dynamic table; the main executable may define a strong copy that
/// `_sbrk_r` actually reads. We therefore decode `_sbrk_r` to find the live
/// mode pointer and patch that.
fn patch_libtransistor_runconf(mem: &mut Memory, data: &[u8], base: u32, text_end: u32) -> Result<()> {
    if let Some(off) = symbol_value(data, "_trn_runconf_heap_mode") {
        let addr = base.wrapping_add(off as u32);
        let _ = mem.write_u32(addr, 1);
    }
    let _ = patch_sbrk_runconf_via_code(mem, base, text_end);
    Ok(())
}

fn read_insn(mem: &Memory, addr: u32) -> Option<u32> {
    mem.read_u32(addr).ok()
}

fn is_svc(insn: u32, imm: u16) -> bool {
    insn == (0xD4000001 | ((imm as u32) << 5))
}

fn decode_bl_target(pc: u32, insn: u32) -> Option<u32> {
    if insn & 0xFC000000 != 0x94000000 {
        return None;
    }
    let imm26 = insn & 0x03FFFFFF;
    let offset = if imm26 < 0x02000000 {
        imm26 as i32
    } else {
        (imm26 as i32) - 0x04000000
    };
    Some(pc.wrapping_add((offset * 4) as u32))
}

fn decode_adrp_target(pc: u32, insn: u32) -> Option<u32> {
    if insn & 0x9F000000 != 0x90000000 {
        return None;
    }
    let immhi = (insn >> 5) & 0x7FFFF;
    let immlo = (insn >> 29) & 0x3;
    let imm = ((immhi << 2) | immlo) as i32;
    let imm = if imm >= (1 << 20) {
        imm - (1 << 21)
    } else {
        imm
    };
    let page = (pc & !0xFFFu32).wrapping_add((imm << 12) as u32);
    Some(page)
}

fn decode_ldr_x_imm_offset(insn: u32) -> Option<u32> {
    // ldr Xt, [Xn, #imm12]: offset in units of 8 bytes.
    if insn & 0xFFC00000 == 0xF9400000 {
        Some(((insn >> 10) & 0xFFF) * 8)
    } else {
        None
    }
}

fn decode_cmp_w_imm(insn: u32) -> Option<(u8, u16)> {
    // subs wzr, wn, #imm  →  top 9 bits 0b011100010, Rd=31.
    if insn & 0xFF80001F != 0x7100001F {
        return None;
    }
    let rn = ((insn >> 5) & 0x1F) as u8;
    let imm = ((insn >> 10) & 0xFFF) as u16;
    Some((rn, imm))
}

fn is_b_cond(insn: u32, cond: u8) -> bool {
    insn & 0xFF00000F == (0x54000000 | (cond as u32))
}

/// Find the `_sbrk_r` function by locating `svc #1` (SetHeapSize) and the
/// `bl` to it, then decode the live `_trn_runconf_heap_mode` pointer and set
/// it to NORMAL.
fn patch_sbrk_runconf_via_code(mem: &mut Memory, base: u32, text_end: u32) -> Result<()> {
    // Locate svc #1 inside the text segment.
    let mut svc_addr = None;
    let mut addr = base;
    while addr < text_end {
        if let Some(insn) = read_insn(mem, addr) {
            if is_svc(insn, 1) {
                svc_addr = Some(addr);
                break;
            }
        }
        addr = addr.wrapping_add(4);
    }
    let svc_addr = match svc_addr {
        Some(a) => a,
        None => return Ok(()),
    };

    // Find a bl that calls it (this is inside _sbrk_r).
    let mut bl_addr = None;
    addr = base;
    while addr < text_end {
        if let Some(insn) = read_insn(mem, addr) {
            if let Some(tgt) = decode_bl_target(addr, insn) {
                if tgt == svc_addr {
                    bl_addr = Some(addr);
                    break;
                }
            }
        }
        addr = addr.wrapping_add(4);
    }
    let bl_addr = match bl_addr {
        Some(a) => a,
        None => return Ok(()),
    };

    // Walk backwards to find the function start (previous ret or prologue).
    let mut func_start = base;
    addr = bl_addr;
    while addr > base {
        addr = addr.wrapping_sub(4);
        if let Some(insn) = read_insn(mem, addr) {
            if insn & 0xFFFFFC1F == 0xD65F0000 {
                func_start = addr.wrapping_add(4);
                break;
            }
            // stp x29, x30, [sp, #imm]! prologue
            if insn & 0xFFE07FFF == 0xA9007BFD {
                func_start = addr;
                break;
            }
        }
    }

    // Scan the function for the mode-test pattern:
    //   adrp  xN, ...
    //   ldr   xN, [xN, #off]      ; load pointer to mode
    //   ldr   wN, [xN]            ; load mode value
    //   cmp   wN, #2
    //   b.eq  override_path
    //   cmp   wN, #1
    //   b.ne  error_path
    addr = func_start;
    while addr < bl_addr {
        let window = [
            read_insn(mem, addr),
            read_insn(mem, addr.wrapping_add(4)),
            read_insn(mem, addr.wrapping_add(8)),
            read_insn(mem, addr.wrapping_add(12)),
            read_insn(mem, addr.wrapping_add(16)),
            read_insn(mem, addr.wrapping_add(20)),
            read_insn(mem, addr.wrapping_add(24)),
        ];
        if let [Some(i0), Some(i1), Some(i2), Some(i3), Some(i4), Some(i5), Some(i6)] = window {
            if let Some(page) = decode_adrp_target(addr, i0) {
                if let Some(off) = decode_ldr_x_imm_offset(i1) {
                    let ptr_addr = page.wrapping_add(off);
                    let reg = i1 & 0x1F;
                    // i2: ldr wreg, [xreg, #0]
                    let i2_is_ldrw = i2 & 0xFFC00000 == 0xB9400000;
                    let i2_reg = i2 & 0x1F;
                    let i2_base = (i2 >> 5) & 0x1F;
                    if i2_is_ldrw && i2_base == reg && ((i2 >> 10) & 0xFFF) == 0 {
                        if let Some((cmp_reg, imm)) = decode_cmp_w_imm(i3) {
                            if cmp_reg == i2_reg as u8 && imm == 2 && is_b_cond(i4, 0) {
                                if let Some((cmp_reg2, imm2)) = decode_cmp_w_imm(i5) {
                                    if cmp_reg2 == i2_reg as u8 && imm2 == 1 && is_b_cond(i6, 1) {
                                        return mem.write_u32(ptr_addr, 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        addr = addr.wrapping_add(4);
    }

    Ok(())
}

/// Find the MOD0 header in the NRO image. It is usually close to the end of
/// the file (just before the dynamic section), so the search covers the whole
/// image rather than only the first page.
fn find_mod0(data: &[u8]) -> Option<usize> {
    let magic = 0x30444f4du32.to_le_bytes(); // "MOD0"
    data.windows(4).position(|w| w == &magic[..])
}

/// Read the `.init_array` function addresses (relative to [`NRO_BASE`]) from
/// the image's dynamic section, returning them as absolute vaddrs. Self-
/// relocating NROs run this table in their crt0's `__libnx_init`; when the
/// crt0 skips that step the loader (HBL / the emulator boot) must run it so
/// C++ static constructors (std::string globals, ...) actually run. Returns
/// an empty list when the image has no constructors.
pub fn init_array_entries(data: &[u8]) -> Vec<u32> {
    let mod0 = match find_mod0(data) {
        Some(m) => m,
        None => return Vec::new(),
    };
    // The dynamic offset is relative to the MOD0 header itself.
    let dyn_rel = crate::nsp::read_u32(data, mod0 + 4);
    let mut dynp = mod0.wrapping_add(dyn_rel as usize);
    let mut init_arr: Option<u32> = None;
    let mut init_sz: u32 = 0;
    for _ in 0..512 {
        let tag = read_u64(data, dynp);
        if tag == 0 {
            break; // DT_NULL
        }
        let val = read_u64(data, dynp + 8);
        if tag == 0x19 {
            init_arr = Some(val as u32);
        } else if tag == 0x1b {
            init_sz = val as u32;
        }
        dynp += 16;
    }
    let (Some(arr_rel), n) = (init_arr, init_sz as usize) else {
        return Vec::new();
    };
    if n == 0 || n % 8 != 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n / 8);
    for i in 0..n / 8 {
        out.push(NRO_BASE.wrapping_add(read_u64(data, arr_rel as usize + i * 8) as u32));
    }
    out
}

/// Apply RELR packed relative relocations. Every 64-bit entry is either an
/// address (bit 0 clear) or a bitmap (bit 0 set); each relocated word gets the
/// load base added, turning stored file offsets into runtime addresses.
///
/// The chain is strictly monotonic (addresses only increase), so processing
/// stops as soon as an entry would move backwards or past `end_addr` — this
/// also guards against trailing garbage after the real RELR data.
fn apply_relr(mem: &mut Memory, base: u32, end_addr: u32, relr: &[u8]) -> Result<()> {
    let mut addr: u32 = 0;
    let mut last: u32 = 0;
    for chunk in relr.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        let word = u64::from_le_bytes(bytes);
        if word & 1 == 0 {
            addr = base.wrapping_add(word as u32);
            if addr < last || addr >= end_addr {
                return Ok(());
            }
            last = addr;
            let cur = mem.read_u64(addr)?;
            mem.write_u64(addr, cur.wrapping_add(base as u64))?;
        } else {
            if addr >= end_addr {
                return Ok(());
            }
            for bit in 1..64u32 {
                if word & (1u64 << bit) != 0 {
                    let a = addr.wrapping_add(8 * (bit - 1));
                    if a < last || a >= end_addr {
                        return Ok(());
                    }
                    last = a;
                    let cur = mem.read_u64(a)?;
                    mem.write_u64(a, cur.wrapping_add(base as u64))?;
                }
            }
            addr = addr.wrapping_add(8 * 63);
        }
    }
    Ok(())
}

/// Apply the RELR relocations described by the image's MOD0/dynamic headers,
/// if present. NROs are linked against `NRO_BASE`, so absolute pointers in
/// .data/.rodata only become valid once the base is added.
fn apply_nro_relocations(mem: &mut Memory, data: &[u8], image_end: u32) -> Result<()> {
    let mod0 = match find_mod0(data) {
        Some(m) => m,
        None => return Ok(()),
    };
    let dyn_off = mod0.wrapping_add(read_u32(data, mod0 + 4) as usize);
    if dyn_off + 16 > data.len() {
        return Ok(());
    }
    let mut off = dyn_off;
    let mut relr_off = 0u32;
    let mut relr_size = 0u32;
    let mut relr_count = 0u64;
    loop {
        if off + 16 > data.len() {
            break;
        }
        let tag = read_u64(data, off);
        let val = read_u64(data, off + 8);
        off += 16;
        if tag == 0 {
            break;
        }
        match tag {
            0x24 => relr_off = val as u32,   // DT_RELR
            0x23 => relr_count = val,        // DT_RELRCOUNT
            0x25 => relr_size = val as u32,  // DT_RELRSZ
            _ => {}
        }
    }
    // Some builds report a bogus DT_RELRSZ (e.g. hbmenu writes the byte size
    // into DT_RELRCOUNT instead); take whichever field is larger, in bytes.
    let size = relr_size.max(relr_count as u32);
    let start = relr_off as usize;
    let end = start.saturating_add(size as usize);
    if size > 0 && end <= data.len() {
        apply_relr(mem, NRO_BASE, image_end, &data[start..end])?;
    }
    Ok(())
}

/// Whether the NRO carries the "HOME BREW" self-relocating crt0 (the `b` +
/// "HOME" "BREW" preamble with `NRO0` at offset 0x10). Such images run their
/// own RELR relocator during startup, so applying RELR here too would add the
/// load base a second time and corrupt every relocated pointer.
fn has_self_relocating_crt0(data: &[u8]) -> bool {
    data.len() >= 0x10
        && read_u32(data, 0x08) == 0x454d_4f48 // "HOME"
        && read_u32(data, 0x0C) == 0x5745_5242 // "BREW"
}

/// Load an NRO into `mem` at [`NRO_BASE`], returning its entry point.
pub fn load_nro(mem: &mut Memory, data: &[u8]) -> Result<LoadedNro> {
    let h = NroHeader::parse(data)?;
    let build_id = h.build_id(data);
    let is_64bit = h.is_64bit(data);

    let base = NRO_BASE;
    let text_addr = base;
    let ro_addr = text_addr.wrapping_add(h.text_size);
    let data_addr = ro_addr.wrapping_add(h.ro_size);
    let end_addr = data_addr.wrapping_add(h.data_size);

    copy_segment(mem, data, h.text_offset, h.text_size, text_addr)?;
    copy_segment(mem, data, h.ro_offset, h.ro_size, ro_addr)?;
    copy_segment(mem, data, h.data_offset, h.data_size, data_addr)?;
    mem.map_zero(end_addr, h.bss_size as usize)?;
    let image_end = end_addr.wrapping_add(h.bss_size);
    // Self-relocating NROs apply RELR themselves in their crt0; running it
    // here as well would relocate every pointer twice. Plain NROs (e.g. the
    // sdl demo) rely on the loader to do it.
    if !has_self_relocating_crt0(data) {
        apply_nro_relocations(mem, data, image_end)?;
    }
    // libtransistor NROs may hardcode a tiny OVERRIDE heap. Force NORMAL so
    // the runtime uses svcSetHeapSize instead of faulting on the first malloc.
    let text_end = text_addr.wrapping_add(h.text_size);
    let _ = patch_libtransistor_runconf(mem, data, base, text_end);
    // .text is never a legitimate relocation target (position-independent
    // code needs no runtime patches to its own instructions), so it can be
    // locked down now: a wild guest write through a stray/null pointer
    // faults immediately instead of silently corrupting the running image.
    // `.rodata` is left writable — a self-relocating crt0 may still need to
    // patch RELR entries living in `.data.rel.ro` there.
    mem.mark_readonly(text_addr, ro_addr);

    let env_addr = if has_self_relocating_crt0(data) {
        setup_env_block(mem)
    } else {
        0
    };

    Ok(LoadedNro {
        base,
        entry: text_addr,
        text: Segment {
            file_offset: h.text_offset,
            file_size: h.text_size,
            mem_addr: text_addr,
        },
        ro: Segment {
            file_offset: h.ro_offset,
            file_size: h.ro_size,
            mem_addr: ro_addr,
        },
        data: Segment {
            file_offset: h.data_offset,
            file_size: h.data_size,
            mem_addr: data_addr,
        },
        bss_size: h.bss_size,
        build_id,
        is_64bit,
        env_addr,
    })
}

fn copy_segment(mem: &mut Memory, data: &[u8], off: u32, size: u32, addr: u32) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    let start = off as usize;
    let end = start + size as usize;
    if end > data.len() {
        return Err(Error::Nro("segment exceeds input data".into()));
    }
    mem.map(addr, &data[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_nro(text: &[u8], data: &[u8]) -> Vec<u8> {
        let text_size = align4(text.len());
        let data_off = 0x50 + text_size;
        let data_size = align4(data.len());
        let mut out = vec![0u8; data_off + data_size];
        out[0..4].copy_from_slice(&NRO0_MAGIC.to_le_bytes());
        out[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
        let total = out.len() as u32;
        out[8..12].copy_from_slice(&total.to_le_bytes());
        out[0x10..0x14].copy_from_slice(&0x50u32.to_le_bytes());
        out[0x14..0x18].copy_from_slice(&(text_size as u32).to_le_bytes());
        out[0x18..0x1C].copy_from_slice(&0u32.to_le_bytes());
        out[0x1C..0x20].copy_from_slice(&0u32.to_le_bytes());
        out[0x20..0x24].copy_from_slice(&(data_off as u32).to_le_bytes());
        out[0x24..0x28].copy_from_slice(&(data_size as u32).to_le_bytes());
        out[0x28..0x2C].copy_from_slice(&0x100u32.to_le_bytes()); // bss
        out[0x50..0x50 + text.len()].copy_from_slice(text);
        out[data_off..data_off + data.len()].copy_from_slice(data);
        out
    }

    fn align4(n: usize) -> usize {
        (n + 3) & !3
    }

    #[test]
    fn loads_and_lays_out_segments() {
        let text = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let nro = build_nro(&text, &data);
        let mut mem = Memory::new();
        let loaded = load_nro(&mut mem, &nro).unwrap();
        assert_eq!(loaded.entry, NRO_BASE);
        assert_eq!(mem.read_u32(NRO_BASE).unwrap(), 0x01);
        assert_eq!(mem.read_u32(NRO_BASE + 4).unwrap(), 0x02);
        assert_eq!(
            mem.read_u32(loaded.data.mem_addr).unwrap(),
            0xEFBE_ADDE
        );
        // bss zero-filled
        assert_eq!(
            mem.read_u8(loaded.data.mem_addr + loaded.data.file_size).unwrap(),
            0
        );
        assert!(loaded.is_64bit);
        assert_eq!(loaded.build_id, [0u8; 0x20]);
    }

    #[test]
    fn loaded_text_is_read_only_but_data_is_not() {
        let text = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let nro = build_nro(&text, &data);
        let mut mem = Memory::new();
        let loaded = load_nro(&mut mem, &nro).unwrap();
        // A wild write into .text (what corrupted the running image before
        // this was locked down) now faults instead of silently succeeding.
        assert!(mem.write_u32(loaded.text.mem_addr, 0xDEAD_BEEF).is_err());
        assert_eq!(mem.read_u32(loaded.text.mem_addr).unwrap(), 0x01);
        // .data stays writable — globals still work.
        mem.write_u32(loaded.data.mem_addr, 0x1234).unwrap();
        assert_eq!(mem.read_u32(loaded.data.mem_addr).unwrap(), 0x1234);
    }

    #[test]
    fn apply_relr_relocates_and_stops_at_backward_jump() {
        let mut mem = Memory::new();
        mem.map_zero(NRO_BASE, 0x2000).unwrap();
        mem.write_u64(NRO_BASE + 0x1000, 0x1234).unwrap();
        mem.write_u64(NRO_BASE + 0x1008, 0x5678).unwrap();
        // RELR: address entry [0x1000], a bitmap flagging the next slot, then
        // a backward address (the trailing garbage hbmenu's section has).
        let mut relr = Vec::new();
        relr.extend_from_slice(&0x1000u64.to_le_bytes());
        relr.extend_from_slice(&0b101u64.to_le_bytes()); // bitmap: bit0=1, bit2 → +8
        relr.extend_from_slice(&0u64.to_le_bytes());     // address 0 → backwards → stop
        apply_relr(&mut mem, NRO_BASE, NRO_BASE + 0x2000, &relr).unwrap();
        assert_eq!(mem.read_u64(NRO_BASE + 0x1000).unwrap(), 0x1234 + NRO_BASE as u64);
        assert_eq!(mem.read_u64(NRO_BASE + 0x1008).unwrap(), 0x5678 + NRO_BASE as u64);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut nro = build_nro(&[0u8; 4], &[]);
        nro[0] = b'X';
        assert!(matches!(NroHeader::parse(&nro), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_truncated() {
        assert!(matches!(
            NroHeader::parse(&[0u8; 0x20]),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn rejects_segment_out_of_bounds() {
        let mut nro = build_nro(&[0u8; 4], &[]);
        nro[0x14..0x18].copy_from_slice(&0xFFFFu32.to_le_bytes());
        assert!(matches!(
            NroHeader::parse(&nro),
            Err(Error::Nro(_))
        ));
    }
}
