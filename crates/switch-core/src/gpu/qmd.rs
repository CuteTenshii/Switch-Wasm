//! The Queue Meta Data a compute dispatch is described by.
//!
//! A launch carries almost none of its state in the class's register file:
//! the channel writes one address and the grid, the block, the constant
//! buffers and the shared-memory size all come out of a 256-byte structure in
//! memory. The opposite of the 3D class, where the register file *is* the
//! state.
//!
//! Field positions are transcribed from NVIDIA's generated `clb1c0qmd.h`.
//! The class defines two QMD versions, `V00_06` and `V01_07`; they differ only
//! in fields nothing here reads, so one parser serves both.

use crate::gpu::engine::field;
use crate::{Error, Result};

/// A QMD is 64 words (256 bytes) however few of them a launch fills in.
pub const QMD_WORDS: usize = 64;

/// How many constant buffers a QMD can bind. The bind slot *is* the index:
/// entry `i` is what the shader reads as `c[i]`.
pub const CONSTANT_BUFFERS: usize = 8;

/// Hardware's ceiling on threads per CTA. A block past it is a misparsed QMD,
/// and the alternative to failing on it is a dispatch that runs for hours.
pub const MAX_CTA_THREADS: u32 = 1024;

/// One of the QMD's bound constant buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantBuffer {
    pub addr: u64,
    pub size: u32,
}

/// A semaphore the launch releases when it completes, immediately, since a
/// dispatch retires inside its own method. It still has to be written, or a
/// guest waiting on it rather than on a syncpoint waits forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Release {
    pub addr: u64,
    pub payload: u32,
    /// The one-word form writes just the payload; the four-word form writes
    /// the payload and a timestamp, as `SetReportSemaphore` does.
    pub one_word: bool,
}

/// A parsed launch descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qmd {
    /// `(major, version)`, kept because it is what says this parse applies.
    pub version: (u32, u32),
    /// Byte offset of the program within the class's program region.
    pub program_offset: u32,
    /// The grid: how many CTAs, in each dimension.
    pub cta_raster: [u32; 3],
    /// The block: how many threads per CTA, in each dimension.
    pub cta_threads: [u32; 3],
    /// Bytes of `s[]` a CTA gets.
    pub shared_memory_size: u32,
    /// Bytes of `l[]` a thread gets.
    pub local_memory_size: u32,
    /// Registers per thread, as the compiler allocated them.
    pub register_count: u32,
    /// How many named barriers the program uses.
    pub barrier_count: u32,
    /// Entry `i` is `c[i]`, or `None` where the valid bit is clear.
    pub constant_buffers: [Option<ConstantBuffer>; CONSTANT_BUFFERS],
    pub releases: [Option<Release>; 2],
}

impl Qmd {
    pub fn threads_per_cta(&self) -> u32 {
        self.cta_threads[0] * self.cta_threads[1] * self.cta_threads[2]
    }

    /// Total CTAs in the grid.
    pub fn cta_count(&self) -> u64 {
        let [x, y, z] = self.cta_raster;
        u64::from(x) * u64::from(y) * u64::from(z)
    }

    /// Whether this launch has any work in it at all. A zero in any dimension
    /// is a legal launch of nothing, not an error.
    pub fn is_empty(&self) -> bool {
        self.cta_count() == 0 || self.threads_per_cta() == 0
    }

    /// Parse the 64 words of a QMD.
    pub fn parse(words: &[u32; QMD_WORDS]) -> Result<Qmd> {
        let version = (mw(words, 580, 583), mw(words, 576, 579));
        // The fields below sit at the same bits in both versions. Anything
        // else is a structure this parser has never seen, and a grid read out
        // of one launches whatever the misread said.
        if !matches!(version, (0, 6) | (1, 7)) {
            return Err(Error::Gpu(format!(
                "qmd: version {}.{} is not a Maxwell compute QMD (expected 0.6 or 1.7)",
                version.0, version.1
            )));
        }

        let cta_threads = [
            mw(words, 592, 607),
            mw(words, 608, 623),
            mw(words, 624, 639),
        ];
        let threads = cta_threads[0] * cta_threads[1] * cta_threads[2];
        if threads > MAX_CTA_THREADS {
            return Err(Error::Gpu(format!(
                "qmd: a CTA of {}x{}x{} is {} threads, past the {} hardware allows",
                cta_threads[0], cta_threads[1], cta_threads[2], threads, MAX_CTA_THREADS
            )));
        }

        let mut constant_buffers = [None; CONSTANT_BUFFERS];
        for (i, slot) in constant_buffers.iter_mut().enumerate() {
            let bit = 640 + i as u32;
            if mw(words, bit, bit) == 0 {
                continue;
            }
            let base = i as u32 * 64;
            let lower = mw(words, 928 + base, 959 + base);
            let upper = mw(words, 960 + base, 967 + base);
            *slot = Some(ConstantBuffer {
                addr: (u64::from(upper) << 32) | u64::from(lower),
                size: mw(words, 975 + base, 991 + base),
            });
        }

        let releases = [release(words, 0), release(words, 1)];

        Ok(Qmd {
            version,
            program_offset: mw(words, 256, 287),
            cta_raster: [
                mw(words, 384, 415),
                mw(words, 416, 431),
                mw(words, 432, 447),
            ],
            cta_threads,
            shared_memory_size: mw(words, 544, 561),
            local_memory_size: mw(words, 1440, 1463),
            register_count: mw(words, 1496, 1503),
            barrier_count: mw(words, 1467, 1471),
            constant_buffers,
            releases,
        })
    }
}

/// One of the two release semaphores, or `None` where its enable bit is clear.
fn release(words: &[u32; QMD_WORDS], which: u32) -> Option<Release> {
    let enable = 202 + which;
    if mw(words, enable, enable) == 0 {
        return None;
    }
    let base = which * 96;
    let lower = mw(words, 736 + base, 767 + base);
    let upper = mw(words, 768 + base, 775 + base);
    Some(Release {
        addr: (u64::from(upper) << 32) | u64::from(lower),
        payload: mw(words, 800 + base, 831 + base),
        one_word: mw(words, 799 + base, 799 + base) == 1,
    })
}

/// Bits `lo..=hi` of the structure: the `MW(hi:lo)` the header names a field
/// by. Reads a word pair, because a field may straddle the boundary (every
/// constant buffer's size does).
fn mw(words: &[u32; QMD_WORDS], lo: u32, hi: u32) -> u32 {
    debug_assert!(hi >= lo && hi - lo < 32, "MW({hi}:{lo}) is not a u32 field");
    let word = (lo / 32) as usize;
    let low = u64::from(words.get(word).copied().unwrap_or(0));
    let high = u64::from(words.get(word + 1).copied().unwrap_or(0));
    let pair = low | (high << 32);
    field((pair >> (lo % 32)) as u32, 0, hi - lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A QMD with just the version stamped, which is the floor every other
    /// test builds on.
    fn blank() -> [u32; QMD_WORDS] {
        let mut words = [0u32; QMD_WORDS];
        set(&mut words, 576, 579, 6);
        set(&mut words, 580, 583, 0);
        words
    }

    /// Write `value` into bits `lo..=hi`, the inverse of [`mw`].
    fn set(words: &mut [u32; QMD_WORDS], lo: u32, hi: u32, value: u32) {
        for bit in 0..=(hi - lo) {
            let at = (lo + bit) as usize;
            let mask = 1u32 << (at % 32);
            if value >> bit & 1 != 0 {
                words[at / 32] |= mask;
            } else {
                words[at / 32] &= !mask;
            }
        }
    }

    #[test]
    fn a_field_that_straddles_a_word_boundary_reads_whole() {
        // Constant buffer 0's size is MW(991:975), 15 bits into word 30 and
        // ending in word 31.
        let mut words = blank();
        set(&mut words, 975, 991, 0x1_2345);
        assert_eq!(mw(&words, 975, 991), 0x1_2345);
    }

    #[test]
    fn the_grid_and_the_block_come_out_as_written() {
        let mut words = blank();
        set(&mut words, 384, 415, 40);
        set(&mut words, 416, 431, 23);
        set(&mut words, 432, 447, 2);
        set(&mut words, 592, 607, 32);
        set(&mut words, 608, 623, 4);
        set(&mut words, 624, 639, 1);
        let qmd = Qmd::parse(&words).unwrap();
        assert_eq!(qmd.cta_raster, [40, 23, 2]);
        assert_eq!(qmd.cta_threads, [32, 4, 1]);
        assert_eq!(qmd.threads_per_cta(), 128);
        assert_eq!(qmd.cta_count(), 40 * 23 * 2);
        assert!(!qmd.is_empty());
    }

    #[test]
    fn only_the_valid_constant_buffers_are_bound() {
        let mut words = blank();
        // c1: valid, at 0x1_0000_2000, 0x400 bytes.
        set(&mut words, 641, 641, 1);
        set(&mut words, 928 + 64, 959 + 64, 0x0000_2000);
        set(&mut words, 960 + 64, 967 + 64, 1);
        set(&mut words, 975 + 64, 991 + 64, 0x400);
        // c2 is filled in with its valid bit clear, the way a reused QMD
        // leaves a stale address behind.
        set(&mut words, 928 + 128, 959 + 128, 0xDEAD_0000);
        let qmd = Qmd::parse(&words).unwrap();
        assert_eq!(qmd.constant_buffers[0], None);
        assert_eq!(
            qmd.constant_buffers[1],
            Some(ConstantBuffer {
                addr: 0x1_0000_2000,
                size: 0x400
            })
        );
        assert_eq!(qmd.constant_buffers[2], None);
    }

    #[test]
    fn both_release_semaphores_are_read_from_their_own_fields() {
        let mut words = blank();
        set(&mut words, 202, 202, 1);
        set(&mut words, 736, 767, 0x1000);
        set(&mut words, 768, 775, 0);
        set(&mut words, 799, 799, 1);
        set(&mut words, 800, 831, 0x11);
        set(&mut words, 203, 203, 1);
        set(&mut words, 832, 863, 0x2000);
        set(&mut words, 896, 927, 0x22);
        let qmd = Qmd::parse(&words).unwrap();
        assert_eq!(
            qmd.releases[0],
            Some(Release {
                addr: 0x1000,
                payload: 0x11,
                one_word: true
            })
        );
        assert_eq!(
            qmd.releases[1],
            Some(Release {
                addr: 0x2000,
                payload: 0x22,
                one_word: false
            })
        );
    }

    #[test]
    fn a_disabled_release_is_not_written() {
        let mut words = blank();
        set(&mut words, 736, 767, 0x1000);
        set(&mut words, 800, 831, 0x11);
        assert_eq!(Qmd::parse(&words).unwrap().releases, [None, None]);
    }

    #[test]
    fn the_other_qmd_version_this_class_defines_parses_too() {
        let mut words = blank();
        set(&mut words, 576, 579, 7);
        set(&mut words, 580, 583, 1);
        set(&mut words, 592, 607, 64);
        assert_eq!(Qmd::parse(&words).unwrap().version, (1, 7));
    }

    #[test]
    fn a_structure_that_is_not_a_qmd_is_refused() {
        // Zeroed memory reads as version 0.0, what a wrong QMD address
        // hands us.
        let err = Qmd::parse(&[0u32; QMD_WORDS]).unwrap_err();
        assert!(
            format!("{err:?}").contains("not a Maxwell compute QMD"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_block_bigger_than_hardware_allows_is_refused() {
        let mut words = blank();
        set(&mut words, 592, 607, 1024);
        set(&mut words, 608, 623, 2);
        set(&mut words, 624, 639, 1);
        let err = Qmd::parse(&words).unwrap_err();
        assert!(format!("{err:?}").contains("past the 1024"), "got {err:?}");
    }

    #[test]
    fn an_empty_grid_is_a_launch_of_nothing_rather_than_an_error() {
        let mut words = blank();
        set(&mut words, 592, 607, 32);
        let qmd = Qmd::parse(&words).unwrap();
        assert!(qmd.is_empty());
    }

    #[test]
    fn the_scalar_fields_land_where_the_header_says() {
        let mut words = blank();
        set(&mut words, 256, 287, 0x1A0);
        set(&mut words, 544, 561, 0x2000);
        set(&mut words, 1440, 1463, 0x100);
        set(&mut words, 1496, 1503, 40);
        set(&mut words, 1467, 1471, 3);
        let qmd = Qmd::parse(&words).unwrap();
        assert_eq!(qmd.program_offset, 0x1A0);
        assert_eq!(qmd.shared_memory_size, 0x2000);
        assert_eq!(qmd.local_memory_size, 0x100);
        assert_eq!(qmd.register_count, 40);
        assert_eq!(qmd.barrier_count, 3);
    }
}
