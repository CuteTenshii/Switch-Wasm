//! switch-core: a from-scratch Nintendo Switch emulation core targeting the
//! browser (WASM) and the host for testing.
//!
//! Phase 0 provides container/format parsers: PFS0 (`.nsp`), NCA headers and
//! the homebrew NRO/ELF loaders.
//!
//! Phase 1 provides a full AArch64 integer interpreter ([`cpu::Cpu`]) that can
//! boot hand-assembled and simple compiled homebrew.
//!
//! Commercial game content (NCA) can be decrypted and its main executable
//! (NSO0) loaded when the caller supplies `prod.keys`/`title.keys` — see
//! [`nca`] and [`nso`]. That only gets a real title as far as its own crt0;
//! actually running one needs the Horizon service surface a retail SDK
//! program expects, which is a much larger undertaking than homebrew ever
//! needed and is tracked separately in `PROGRESS.md`.

/// Whether an environment switch such as `TRACE_SVC` is set, read once and
/// remembered.
///
/// `std::env::var` is a linear scan of the whole environment on every call,
/// and these switches sit in per-syscall, per-IPC and per-draw paths —
/// `getenv` was **37% of a Home Menu run**, more than the shader interpreter
/// and the rasterizer put together. Nothing changes the environment under a
/// running emulator, so asking once is the same answer for less than a
/// thousandth of the cost.
///
/// One `OnceLock` per call site rather than a map: the name is a literal, so
/// the lookup is a load and a branch with nothing to hash.
#[macro_export]
macro_rules! env_flag {
    ($name:literal) => {{
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var($name).is_ok())
    }};
}

/// A hasher for keys that are already well-distributed integers — kernel
/// handles, guest addresses, object ids.
///
/// `HashMap`'s default is SipHash-1-3, built to survive keys chosen by an
/// attacker and costing about as much as the lookup it protects. Every key
/// here is one this emulator minted itself, and `horizon_syscall` looks
/// several up per guest syscall: SipHash was **18.7%** of a Home Menu run,
/// more than the shader interpreter.
///
/// Iteration order cannot be depended on either way — the default hasher
/// seeds itself randomly per process, so anything that could notice the
/// difference was already nondeterministic between two runs.
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl IdHasher {
    /// fxhash's constant: the 64-bit inverse of the golden ratio, whose bits
    /// are spread evenly enough that one multiply carries a change in any
    /// input bit across the whole word.
    const SEED: u64 = 0x517c_c1b7_2722_0a95;

    #[inline]
    fn mix(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
    }
}

impl std::hash::Hasher for IdHasher {
    /// Folding the high half down is not optional. One multiply moves entropy
    /// *upward* only, and `HashMap` picks its bucket with the **low** bits —
    /// so without this every page-aligned guest address hashed to the same
    /// bucket: 4096 of them shared one, against 2938 buckets with it.
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }

    /// The general case, for a key that is not a single word. Correct for any
    /// bytes; it is just not what this exists for.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.mix(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.mix(u64::from(n));
    }
    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.mix(u64::from(n));
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.mix(u64::from(n));
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.mix(n);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.mix(n as u64);
    }
}

/// A [`std::collections::HashMap`] keyed by an integer this emulator minted;
/// see [`IdHasher`] for why it does not need a cryptographic hash.
pub type IdMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<IdHasher>>;

pub mod bktr;
pub mod control;
pub mod cpu;
pub mod crypto;
pub mod disasm;
pub mod display;
pub mod elf;
pub mod error;
pub mod gpu;
pub mod keys;
pub mod lz4;
pub mod mem;
pub mod nca;
pub mod nro;
pub mod nso;
pub mod npdm;
pub mod opus;
pub mod nsp;
pub mod romfs;
pub mod source;
pub mod ticket;
pub mod vfs;

pub use error::{Error, Result};

/// Memory-mapped framebuffer (modelled on the Switch GPU's): fixed-size,
/// little-endian RGBA. Homebrew writes pixels here and the host renders it.
///
/// It and [`INPUT_ADDR`] sit above every region a Horizon process is given —
/// see `cpu::GUEST_SPACE_END`. They used to live at 0x3F00_0000, immediately
/// after a 240 MiB heap region; the heap now needs the address space they
/// were standing in.
pub const FB_BASE: u32 = 0xF400_0000;
pub const FB_WIDTH: u32 = 640;
pub const FB_HEIGHT: u32 = 360;
pub const FB_STRIDE: u32 = FB_WIDTH * 4;
/// Memory-mapped input register: the host writes an ASCII key here and
/// homebrew acknowledges (writes 0) when consumed.
pub const INPUT_ADDR: u32 = 0xF410_0000;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, BuildHasherDefault};

    /// The keys this hashes are handles and addresses: dense runs, and runs
    /// spaced by a page or a sector. Every one of those has to spread across
    /// the **low** bits, because that is where `HashMap` picks its bucket —
    /// and the low bits of a page-aligned address are all the same.
    #[test]
    fn the_id_hasher_spreads_the_keys_it_exists_for() {
        let build = BuildHasherDefault::<IdHasher>::default();
        for (name, keys) in [
            ("handles", (0x1000u64..0x1000 + 4096).collect::<Vec<_>>()),
            ("page-aligned", (0..4096u64).map(|i| i * 0x1000).collect()),
            ("sector-aligned", (0..4096u64).map(|i| 0x8000_0000 + i * 0x200).collect()),
        ] {
            let hashes: std::collections::HashSet<u64> =
                keys.iter().map(|k| build.hash_one(k)).collect();
            assert_eq!(hashes.len(), keys.len(), "{name}: every key hashed to its own value");
            // 4096 keys into 4096 buckets is a birthday problem: about
            // 4096 * (1 - 1/e) = 2589 of them end up occupied however good the
            // hash is, so this asks for that and not for perfection.
            let buckets: std::collections::HashSet<u64> =
                keys.iter().map(|k| build.hash_one(k) & 0xFFF).collect();
            assert!(buckets.len() > 2400, "{name}: {} buckets of 4096", buckets.len());
        }
    }

    /// It is a hasher, so it has to answer for any bytes, not just words.
    #[test]
    fn the_id_hasher_handles_a_key_that_is_not_one_word() {
        let build = BuildHasherDefault::<IdHasher>::default();
        assert_ne!(build.hash_one("hbmenu"), build.hash_one("qlaunch"));
        assert_eq!(build.hash_one("hbmenu"), build.hash_one("hbmenu"));
        assert_ne!(build.hash_one((1u64, 2u64)), build.hash_one((2u64, 1u64)));
    }
}
