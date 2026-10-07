//! switch-core: a from-scratch Nintendo Switch emulation core targeting the
//! browser (WASM) and the host for testing.

/// Whether an environment switch such as `SWITCH_NO_JIT` is set, read once
/// and remembered.
#[macro_export]
macro_rules! env_flag {
    ($name:literal) => {{
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var($name).is_ok())
    }};
}

/// A fast non-cryptographic hasher for integer keys this emulator minted:
/// kernel handles, guest addresses, object ids.
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl IdHasher {
    /// fxhash's constant.
    const SEED: u64 = 0x517c_c1b7_2722_0a95;

    #[inline]
    fn mix(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
    }
}

impl std::hash::Hasher for IdHasher {
    /// Folds the high half down: `HashMap` buckets by the low bits.
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }

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

/// A [`std::collections::HashMap`] keyed by an integer this emulator minted.
pub type IdMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<IdHasher>>;

pub mod bktr;
pub mod bucket;
pub mod compressed;
pub mod control;
pub mod cpu;
pub mod crypto;
pub mod disasm;
pub mod display;
pub mod elf;
pub mod error;
pub mod gpu;
pub mod kernel;
pub mod keys;
pub mod lz4;
pub mod mem;
pub mod nca;
pub mod npdm;
pub mod nro;
pub mod nso;
pub mod nsp;
pub mod opus;
pub mod romfs;
pub mod services;
pub mod source;
pub mod sparse;
pub mod ticket;
pub mod trace;
pub mod vfs;
pub mod xci;

pub use error::{Error, Result};

/// A fixed RGBA framebuffer in guest memory, presented when nothing has drawn
/// through the GPU. Not a console facility; for programs with no graphics stack.
pub const FB_BASE: u32 = 0xFE00_0000;
pub const FB_WIDTH: u32 = 640;
pub const FB_HEIGHT: u32 = 360;
pub const FB_STRIDE: u32 = FB_WIDTH * 4;
/// Memory-mapped pad state (buttons then four stick axes), the input
/// counterpart to [`FB_BASE`].
pub const INPUT_ADDR: u32 = 0xFE10_0000;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, BuildHasherDefault};

    #[test]
    fn the_id_hasher_spreads_the_keys_it_exists_for() {
        let build = BuildHasherDefault::<IdHasher>::default();
        for (name, keys) in [
            ("handles", (0x1000u64..0x1000 + 4096).collect::<Vec<_>>()),
            ("page-aligned", (0..4096u64).map(|i| i * 0x1000).collect()),
            (
                "sector-aligned",
                (0..4096u64).map(|i| 0x8000_0000 + i * 0x200).collect(),
            ),
        ] {
            let hashes: std::collections::HashSet<u64> =
                keys.iter().map(|k| build.hash_one(k)).collect();
            assert_eq!(
                hashes.len(),
                keys.len(),
                "{name}: every key hashed to its own value"
            );
            // A birthday problem: about 4096 * (1 - 1/e) = 2589 occupied at best.
            let buckets: std::collections::HashSet<u64> =
                keys.iter().map(|k| build.hash_one(k) & 0xFFF).collect();
            assert!(
                buckets.len() > 2400,
                "{name}: {} buckets of 4096",
                buckets.len()
            );
        }
    }

    #[test]
    fn the_id_hasher_handles_a_key_that_is_not_one_word() {
        let build = BuildHasherDefault::<IdHasher>::default();
        assert_ne!(build.hash_one("hbmenu"), build.hash_one("qlaunch"));
        assert_eq!(build.hash_one("hbmenu"), build.hash_one("hbmenu"));
        assert_ne!(build.hash_one((1u64, 2u64)), build.hash_one((2u64, 1u64)));
    }
}
