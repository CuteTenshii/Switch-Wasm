//! Hand every frame of each `x.ivf` in a directory to the rebuilder as nvdec would get it (the
//! setup, the probabilities the frame decodes with, the tile data and its surfaces) and check the
//! rebuilt stream decodes to the same pictures, hidden ones included:
//! `vp9_rebuild_check <dir> [--out <dir>]`.
//!
//! `--out` writes `x.rebuilt.ivf` and the `x.rebuilt.yuv` it must decode to, so libvpx can check
//! it as well: `vpxdec --i420 -o got.yuv x.rebuilt.ivf && cmp got.yuv x.rebuilt.yuv`.
use clap::Parser;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use switch_vp9::decoder::split_superframe;
use switch_vp9::nvdec::{Dimensions, PicSetup};
use switch_vp9::rebuild::Rebuilder;
use switch_vp9::{Decoder, Frame};

#[derive(Parser)]
struct Cli {
    dir: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
}

/// More surfaces than the eight slots, so the rebuilder sees a surface written over.
const SURFACES: u64 = 11;

fn dimensions(width: u32, height: u32) -> Dimensions {
    Dimensions {
        width: width as u16,
        height: height as u16,
        luma_pitch: width as u16,
        chroma_pitch: width.div_ceil(2) as u16,
    }
}

fn visible(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::new();
    for plane in &frame.planes {
        for y in 0..plane.height {
            out.extend_from_slice(&plane.data[y * plane.stride..][..plane.width]);
        }
    }
    out
}

struct Ivf {
    bytes: Vec<u8>,
    frames: u32,
}

impl Ivf {
    fn new(width: u32, height: u32) -> Ivf {
        let mut bytes = b"DKIF".to_vec();
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&32u16.to_le_bytes());
        bytes.extend_from_slice(b"VP90");
        bytes.extend_from_slice(&(width as u16).to_le_bytes());
        bytes.extend_from_slice(&(height as u16).to_le_bytes());
        bytes.extend_from_slice(&30u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 8]);
        Ivf { bytes, frames: 0 }
    }

    fn push(&mut self, data: &[u8]) {
        self.bytes
            .extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.bytes
            .extend_from_slice(&u64::from(self.frames).to_le_bytes());
        self.bytes.extend_from_slice(data);
        self.frames += 1;
    }

    fn finish(mut self) -> Vec<u8> {
        self.bytes[24..28].copy_from_slice(&self.frames.to_le_bytes());
        self.bytes
    }
}

fn check(path: &Path, out: Option<&Path>) {
    let ivf = std::fs::read(path).expect("read clip");
    let (width, height) = (
        u32::from(u16::from_le_bytes([ivf[12], ivf[13]])),
        u32::from(u16::from_le_bytes([ivf[14], ivf[15]])),
    );
    let mut original = Decoder::new();
    let mut host = Decoder::new();
    let mut rebuilder = Rebuilder::default();
    let mut rebuilt = Ivf::new(width, height);
    let mut expected = Vec::new();
    // The surface the driver keeps in each of the stream's slots.
    let mut slots = [u64::MAX; 8];
    let mut previous: Option<(bool, bool, Dimensions)> = None;
    let (mut frames, mut fillers, mut waiting) = (0, 0, false);
    let (mut segmented, mut compound, mut hidden) = (0, 0, 0);
    let mut refused: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut at = 32;
    while at + 12 <= ivf.len() {
        let size = u32::from_le_bytes(ivf[at..at + 4].try_into().unwrap()) as usize;
        at += 12;
        let packet = &ivf[at..at + size];
        at += size;
        for data in split_superframe(packet) {
            let mut given = None;
            let decoded = original
                .decode_frame_inspecting(data, &mut |input, tiles| {
                    given = Some((
                        input.hdr.clone(),
                        input.fc,
                        input.seg.clone(),
                        input.lf.clone(),
                        tiles.to_vec(),
                    ))
                })
                .expect("the original stream decodes");
            // A frame shown again never reaches nvdec.
            let Some((hdr, fc, seg, lf, tiles)) = given else {
                continue;
            };
            frames += 1;
            segmented += u32::from(seg.enabled);
            compound += u32::from(!hdr.is_intra() && hdr.reference_mode != 0);
            hidden += u32::from(!decoded.shown);
            let key = hdr.key_frame;
            let surface = (0..SURFACES)
                .find(|s| !slots.contains(s))
                .expect("a surface no slot holds");
            let refs = [0, 1, 2].map(|i| slots[hdr.ref_frame_idx[i]]);
            let size = dimensions(hdr.width, hdr.height);
            let (last_shown, last_key, last_size) =
                previous.unwrap_or((false, false, Dimensions::default()));
            let flags = u32::from(key)
                | u32::from(last_key) << 1
                | u32::from(hdr.frame_parallel) << 2
                | u32::from(hdr.error_resilient) << 3
                | u32::from(last_shown) << 4
                | u32::from(hdr.intra_only) << 5;
            let setup = PicSetup {
                bitstream_size: tiles.len() as u32,
                last_size,
                golden_size: size,
                alt_size: size,
                cur_size: size,
                flags,
                header: hdr.clone(),
                seg,
                lf,
            };
            previous = Some((decoded.shown, key, size));
            for (i, slot) in slots.iter_mut().enumerate() {
                if hdr.refresh_frame_flags >> i & 1 != 0 {
                    *slot = surface;
                }
            }

            if waiting && !key {
                continue;
            }
            waiting = false;
            let chunk =
                match rebuilder.frame(&setup, &fc, &tiles, [refs[0], refs[1], refs[2], surface]) {
                    Ok(chunk) => chunk,
                    Err(why) => {
                        *refused.entry(why).or_default() += 1;
                        // As on the host: nothing more goes there until the next key frame.
                        (rebuilder, host, waiting) = (Rebuilder::default(), Decoder::new(), true);
                        continue;
                    }
                };
            let parts = split_superframe(&chunk.data).len();
            fillers += usize::from(parts == 2 && !hdr.intra_only);
            let shown = host
                .decode_packet(&chunk.data)
                .unwrap_or_else(|e| panic!("{}: frame {frames}: {e:?}", path.display()));
            let [frame] = shown.as_slice() else {
                panic!(
                    "{}: frame {frames} showed {} frames",
                    path.display(),
                    shown.len()
                );
            };
            let (got, want) = (visible(frame), visible(&decoded.frame));
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                panic!(
                    "{}: frame {frames} differs at byte {i}: got {}, want {}",
                    path.display(),
                    got[i],
                    want[i]
                );
            }
            rebuilt.push(&chunk.data);
            expected.extend_from_slice(&want);
        }
    }
    println!(
        "{}: {} of {frames} frames rebuilt and matching, {fillers} behind a filler; \
         {hidden} hidden, {segmented} segmented, {compound} with compound prediction; \
         refused: {refused:?}",
        path.display(),
        rebuilt.frames
    );
    if let Some(out) = out {
        let stem = path.file_stem().unwrap().to_string_lossy();
        std::fs::write(out.join(format!("{stem}.rebuilt.ivf")), rebuilt.finish())
            .expect("write ivf");
        std::fs::write(out.join(format!("{stem}.rebuilt.yuv")), expected).expect("write yuv");
    }
}

fn main() {
    let cli = Cli::parse();
    let mut clips: Vec<_> = std::fs::read_dir(&cli.dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "ivf"))
        .filter(|p| !p.to_string_lossy().ends_with(".rebuilt.ivf"))
        .collect();
    clips.sort();
    assert!(!clips.is_empty(), "no .ivf in {}", cli.dir.display());
    for path in clips {
        check(&path, cli.out.as_deref());
    }
}
