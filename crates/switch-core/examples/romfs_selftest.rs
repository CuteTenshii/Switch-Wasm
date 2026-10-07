//! Read a title's RomFS ranges in different ways and check the bytes agree:
//! `romfs_selftest <container> <prod.keys> [title.keys] [samples]`.
//!
//! `SEED=<n>` picks the sample set, `WINDOW=<hex>` the bytes per sample
//! (default 0x9000), and `INJECT=1` adds a deliberate boundary bug that the
//! test must report.
mod common;

use switch_core::source::ByteSource;

const USAGE: &str = "romfs_selftest <container> <prod.keys> [title.keys] [samples]";

/// Usual LZ4 block size of the compression layer; only a hint for sampling.
const BLOCK_HINT: u64 = 0x1_0000;

/// xorshift64*, so a seed reproduces a sample set exactly.
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

/// Canary reader: a short read starting on a block boundary gets its first
/// byte wrong.
#[derive(Debug)]
struct Flaky<S>(S);

impl<S: ByteSource> ByteSource for Flaky<S> {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<usize, switch_core::Error> {
        let got = self.0.read_at(offset, out)?;
        if got > 0 && offset.is_multiple_of(BLOCK_HINT) && out.len() < 0x1000 {
            out[0] ^= 0x01;
        }
        Ok(got)
    }
}

struct Sample {
    at: u64,
    len: u64,
    why: String,
}

struct Mismatch {
    at: u64,
    whole: u8,
    piecewise: u8,
}

/// Compare `reference` against the range read in `chunk`-sized pieces,
/// optionally reversed and with evicting far reads in between.
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
            // Only the cache side effect matters.
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

/// Ranges at file edges and around compression block boundaries.
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
            // Trim the window to the file it was drawn from.
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
            "INJECT=1: a boundary bug sits in front of the reader; it must be reported \
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

    // A file extending past the image is a metadata fault, not a reader one.
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
            // One-byte walks only cover the start of the range.
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
                     sample {:#x} +{:#x}: {}",
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
        // The canary inverts the verdict.
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
        "{failures} inconsistent sample(s), {} file(s) whose extent leaves the image; \
         rerun with SEED={seed} to get the same set",
        overrunning.len()
    );
    std::process::exit(1);
}
