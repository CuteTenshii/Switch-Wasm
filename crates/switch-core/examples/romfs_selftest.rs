//! Read a title's RomFS twice and check it said the same thing both times:
//! `romfs_selftest <container> <prod.keys> [title.keys] [samples]`.
//!
//! A RomFS is served through a stack — `HostSource` → the NCA window →
//! AES-CTR → the compression layer and its block cache — and every layer of
//! it answers a *range*. Nothing verifies what comes out: the ExeFS is
//! hash-checked against the NCA header, but full IVFC verification of a RomFS
//! is not implemented (`nca.rs` says so), so a byte that decrypts or
//! decompresses wrongly is served to the guest and believed. It surfaces as a
//! title behaving oddly hundreds of millions of instructions later, with
//! nothing anywhere near the reader to say the bytes were wrong.
//!
//! There is no reference image to compare against — but a correct reader has
//! a property that does not need one: **the bytes of a range do not depend on
//! how the range was asked for**. Read it whole, read it in pieces, read the
//! pieces backwards, read something far away in between, read it again: a
//! stack with a block boundary off by one, a cache that keys on the wrong
//! thing, or a decompressor that carries state between calls disagrees with
//! itself, and that disagreement is a real bug in every case. That is the
//! whole test.
//!
//! Ranges are drawn where such a bug would show first: at each file's first
//! and last bytes, and straddling the compression layer's block boundaries —
//! `CACHED_BLOCKS` is 4, so a window a few blocks wide also makes the cache
//! evict while the sample is still being read.
//!
//! `SEED=<n>` picks the sample set — a failing run names its seed, and that
//! seed reproduces it exactly. `WINDOW=<hex>` is how many bytes each sample
//! compares (default 0x9000, a little over two 16 KiB blocks).
//!
//! `INJECT=1` puts a deliberate boundary bug in front of the real reader and
//! expects to be told about it. A consistency test that has never failed is
//! indistinguishable from one that cannot fail, and "the RomFS is fine" is
//! exactly the kind of answer that has to be worth something: run it once
//! with the canary and the same command reports what a real bug would look
//! like.
mod common;

use switch_core::source::ByteSource;

const USAGE: &str = "romfs_selftest <container> <prod.keys> [title.keys] [samples]";

/// The block size the compression layer's LZ4 entries usually cover. Only a
/// hint for where to aim a sample: the table is the authority and it is not
/// visible from out here, so samples straddle this *and* every file boundary.
const BLOCK_HINT: u64 = 0x1_0000;

/// xorshift64*, so a seed names a sample set exactly. Sampling has to be
/// reproducible or a failure cannot be shown to anyone else.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next() % bound
        }
    }
}

/// A reader with a bug of the shape this test exists to find: a short read
/// that starts on a block boundary comes back with its first byte wrong,
/// which is what a cache keyed on the wrong thing or an entry lookup off by
/// one does. A whole-window read is unaffected, so only reading the range
/// both ways finds it.
#[derive(Debug)]
struct Flaky<S>(S);

impl<S: ByteSource> ByteSource for Flaky<S> {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, switch_core::Error> {
        let got = self.0.read_at(offset, out)?;
        if got > 0 && offset % BLOCK_HINT == 0 && out.len() < 0x1000 {
            out[0] ^= 0x01;
        }
        Ok(got)
    }
}

/// One range to check, and why it was picked — a failure reports the reason,
/// because "the first byte of a file" and "across a block boundary" send you
/// to different code.
struct Sample {
    at: u64,
    len: u64,
    why: String,
}

/// Where the two readings first differ.
struct Mismatch {
    at: u64,
    whole: u8,
    piecewise: u8,
}

/// Compare `reference` against the same range read in `chunk`-sized pieces,
/// optionally back to front, and optionally with a distant read between every
/// piece to make the block cache evict.
fn read_piecewise(
    source: &dyn ByteSource,
    at: u64,
    len: u64,
    chunk: u64,
    backwards: bool,
    thrash: Option<u64>,
) -> Result<Vec<u8>, switch_core::Error> {
    let mut out = vec![0u8; len as usize];
    let mut offsets: Vec<u64> = (0..len).step_by(chunk as usize).collect();
    if backwards {
        offsets.reverse();
    }
    let mut scratch = [0u8; 64];
    for start in offsets {
        let end = (start + chunk).min(len);
        source.read_exact_at(at + start, &mut out[start as usize..end as usize])?;
        if let Some(far) = thrash {
            // Deliberately ignored: the point is the side effect on the cache,
            // and a far read that lands past the end of the image is not a
            // failure of the sample being taken.
            let _ = source.read_at(far, &mut scratch);
        }
    }
    Ok(out)
}

fn first_difference(reference: &[u8], other: &[u8], at: u64) -> Option<Mismatch> {
    reference
        .iter()
        .zip(other)
        .position(|(a, b)| a != b)
        .map(|i| Mismatch {
            at: at + i as u64,
            whole: reference[i],
            piecewise: other[i],
        })
}

/// The ranges to check, drawn where a range-addressed stack goes wrong: the
/// edges of what the guest asks for, and the edges of what the layers below
/// store.
fn samples(image: &common::romfs::Image, window: u64, wanted: usize, rng: &mut Rng) -> Vec<Sample> {
    let mut out = Vec::new();
    if image.files.is_empty() {
        return out;
    }
    let clamp = |at: u64, len: u64| -> Option<(u64, u64)> {
        let at = at.min(image.len.saturating_sub(1));
        let len = len.min(image.len - at);
        (len > 0).then_some((at, len))
    };
    while out.len() < wanted {
        let file = &image.files[rng.below(image.files.len() as u64) as usize];
        if file.size == 0 {
            continue;
        }
        let end = file.start + file.size;
        let picks = [
            (file.start, "a file's first bytes"),
            (end.saturating_sub(window), "a file's last bytes"),
            // Straddling the boundary the compression layer most likely has an
            // entry edge on, and a boundary inside the file wherever the
            // sample lands.
            (
                (file.start + BLOCK_HINT) & !(BLOCK_HINT - 1),
                "across a block boundary",
            ),
            (file.start + rng.below(file.size), "somewhere inside a file"),
        ];
        for (at, why) in picks {
            if out.len() == wanted {
                break;
            }
            // A range past this file is still a range the stack must serve
            // consistently, but a sample that says "a file's last bytes" and
            // reads the next file's is a confusing thing to report, so the
            // window is trimmed to the file it was drawn from.
            let limit = window.min(end.saturating_sub(at).max(1));
            if let Some((at, len)) = clamp(at, limit) {
                out.push(Sample {
                    at,
                    len,
                    why: format!("{why}: {}", file.path),
                });
            }
        }
    }
    out
}

fn main() {
    let args = common::container_args(USAGE);
    let wanted = args.rest_num(0).unwrap_or(200) as usize;
    let seed = common::env_u64("SEED", 1);
    let window = u64::from(common::env_hex("WINDOW").unwrap_or(0x9000));

    let title = args.open();
    let (real, image) = title.romfs(USAGE);
    let canary = common::env_u64("INJECT", 0) != 0;
    let source: Box<dyn ByteSource> = if canary {
        println!(
            "INJECT=1: a boundary bug sits in front of the reader — it must be reported \
             below, and this run exits 0 only if it was"
        );
        Box::new(Flaky(real))
    } else {
        real
    };
    println!(
        "RomFS: {:#x} bytes, {} files, data at {:#x}",
        image.len,
        image.files.len(),
        image.data_offset
    );

    // Before reading a byte: a file whose extent leaves the image is a
    // metadata or geometry fault rather than a reader one, and it would
    // otherwise turn up as an unreadable sample and be blamed on the stack.
    let overrunning: Vec<&common::romfs::Entry> = image
        .files
        .iter()
        .filter(|f| f.start.saturating_add(f.size) > image.len)
        .collect();
    for file in &overrunning {
        println!(
            "  extent past the end of the image: {} at {:#x} +{:#x}",
            file.path, file.start, file.size
        );
    }

    let mut rng = Rng(seed);
    let samples = samples(&image, window, wanted, &mut rng);
    let mut compared = 0u64;
    let mut failures = 0usize;
    for sample in &samples {
        let reference = match source.read_vec(sample.at, sample.len) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!(
                    "  UNREADABLE {:#x} +{:#x} ({}): {e}",
                    sample.at, sample.len, sample.why
                );
                failures += 1;
                continue;
            }
        };
        // Chunk sizes that put a boundary everywhere it can be: inside a
        // machine word, either side of a page, and a whole page — plus the
        // orders and the eviction that a stateful stack disagrees with itself
        // over.
        let far = (sample.at + image.len / 2) % image.len;
        let readings: [(&str, u64, bool, Option<u64>); 8] = [
            ("1-byte pieces", 1, false, None),
            ("3-byte pieces", 3, false, None),
            ("0xfff-byte pieces", 0xfff, false, None),
            ("page-sized pieces", 0x1000, false, None),
            ("0x1001-byte pieces", 0x1001, false, None),
            ("pages, back to front", 0x1000, true, None),
            ("pages, cache evicted between", 0x1000, false, Some(far)),
            ("the whole range again", sample.len.max(1), false, None),
        ];
        for (how, chunk, backwards, thrash) in readings {
            // A one-byte walk over a wide window is thousands of reads; the
            // narrow chunk sizes are what catch an off-by-one, so they run on
            // the first part of the range rather than not at all.
            let len = if chunk < 8 {
                sample.len.min(0x800)
            } else {
                sample.len
            };
            let other = match read_piecewise(&*source, sample.at, len, chunk, backwards, thrash) {
                Ok(bytes) => bytes,
                Err(e) => {
                    println!(
                        "  UNREADABLE {:#x} +{len:#x} as {how} ({}): {e}",
                        sample.at, sample.why
                    );
                    failures += 1;
                    continue;
                }
            };
            compared += len;
            if let Some(bad) = first_difference(&reference[..len as usize], &other, sample.at) {
                println!(
                    "  MISMATCH at {:#x}: whole read {:#04x}, {how} {:#04x}\n    \
                     sample {:#x} +{:#x} — {}",
                    bad.at, bad.whole, bad.piecewise, sample.at, sample.len, sample.why
                );
                failures += 1;
                break;
            }
        }
    }

    println!(
        "{} samples, {:.1} MiB compared, seed {seed}, window {window:#x}",
        samples.len(),
        compared as f64 / (1024.0 * 1024.0),
    );
    if canary {
        // The canary inverts the verdict: what is being checked is the test.
        println!(
            "{}",
            match failures {
                0 => "CANARY MISSED: the injected boundary bug went unreported",
                _ => "canary caught: the injected boundary bug was reported above",
            }
        );
        std::process::exit((failures == 0) as i32);
    }
    if failures == 0 && overrunning.is_empty() {
        println!("consistent: every range read the same however it was asked for");
        return;
    }
    println!(
        "{failures} inconsistent sample(s), {} file(s) whose extent leaves the image — \
         rerun with SEED={seed} to get the same set",
        overrunning.len()
    );
    std::process::exit(1);
}
