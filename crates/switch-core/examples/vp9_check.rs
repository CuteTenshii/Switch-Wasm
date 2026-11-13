//! Decode every `x.ivf` in a directory and compare each shown frame with the `x.yuv` ffmpeg wrote
//! beside it (`ffmpeg -i x.ivf -f rawvideo -pix_fmt yuv420p x.yuv`): `vp9_check <dir>`.
use clap::Parser;
use switch_core::vp9::Decoder;

#[derive(Parser)]
struct Cli {
    dir: std::path::PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let mut clips: Vec<_> = std::fs::read_dir(&cli.dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "ivf"))
        .collect();
    clips.sort();
    assert!(!clips.is_empty(), "no .ivf in {}", cli.dir.display());
    for path in clips {
        let ivf = std::fs::read(&path).expect("read clip");
        let expect = std::fs::read(path.with_extension("yuv")).expect("read yuv");
        let (width, height) = (
            u16::from_le_bytes([ivf[12], ivf[13]]) as usize,
            u16::from_le_bytes([ivf[14], ivf[15]]) as usize,
        );
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let frame_bytes = width * height + 2 * cw * ch;
        let mut decoder = Decoder::new();
        let mut at = 32;
        let mut shown = 0;
        while at + 12 <= ivf.len() {
            let size = u32::from_le_bytes(ivf[at..at + 4].try_into().unwrap()) as usize;
            at += 12;
            let packet = &ivf[at..at + size];
            at += size;
            for frame in decoder.decode_packet(packet).expect("decode") {
                let reference = &expect[shown * frame_bytes..(shown + 1) * frame_bytes];
                let mut got = Vec::with_capacity(frame_bytes);
                for (plane, (w, h)) in
                    frame
                        .planes
                        .iter()
                        .zip([(width, height), (cw, ch), (cw, ch)])
                {
                    for y in 0..h {
                        got.extend_from_slice(&plane.data[y * plane.stride..y * plane.stride + w]);
                    }
                }
                if let Some(i) = got.iter().zip(reference).position(|(a, b)| a != b) {
                    let (plane, offset) = if i < width * height {
                        (0, i)
                    } else if i < width * height + cw * ch {
                        (1, i - width * height)
                    } else {
                        (2, i - width * height - cw * ch)
                    };
                    let w = if plane == 0 { width } else { cw };
                    panic!(
                        "{}: frame {shown}, plane {plane}, pixel ({}, {}): got {}, want {}",
                        path.display(),
                        offset % w,
                        offset / w,
                        got[i],
                        reference[i]
                    );
                }
                shown += 1;
            }
        }
        assert_eq!(
            shown * frame_bytes,
            expect.len(),
            "{}: frame count",
            path.display()
        );
        println!("{}: {shown} frames match", path.display());
    }
}
