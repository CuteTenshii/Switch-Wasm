//! Decode the first captured frame of an nvdec stream and write it as a PPM:
//! `vp9_capture <dir> <out.ppm>`, `<dir>` holding `c10_001_{setup,bits,probs}.bin`.
use clap::Parser;
use switch_vp9::decoder::{decode_pixels, FrameInput};
use switch_vp9::nvdec::{parse_prob_table, PicSetup};

#[derive(Parser)]
struct Cli {
    dir: std::path::PathBuf,
    out: std::path::PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let read = |name: &str| std::fs::read(cli.dir.join(name)).expect("read capture");
    let setup = PicSetup::parse(&read("c10_001_setup.bin"));
    let bits = read("c10_001_bits.bin");
    let fc = parse_prob_table(&read("c10_001_probs.bin"));
    assert!(setup.is_key_frame());
    let mi = setup.header.mi_cols() * setup.header.mi_rows();
    let input = FrameInput {
        hdr: setup.header.clone(),
        fc,
        seg: setup.seg.clone(),
        lf: setup.lf.clone(),
        refs: Default::default(),
        prev_mvs: None,
        use_prev_frame_mvs: false,
        last_seg_map: vec![0; mi],
        cur_seg_map: vec![0; mi],
        count: true,
    };
    let output = decode_pixels(input, &bits[..setup.bitstream_size as usize]);
    output.result.expect("decode");
    let frame = output.frame;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            let luma = f32::from(frame.planes[0].data[y * frame.planes[0].stride + x]);
            let u =
                f32::from(frame.planes[1].data[(y / 2) * frame.planes[1].stride + x / 2]) - 128.0;
            let v =
                f32::from(frame.planes[2].data[(y / 2) * frame.planes[2].stride + x / 2]) - 128.0;
            let r = luma + 1.402 * v;
            let g = luma - 0.344 * u - 0.714 * v;
            let b = luma + 1.772 * u;
            ppm.extend([r, g, b].map(|c| c.clamp(0.0, 255.0) as u8));
        }
    }
    std::fs::write(&cli.out, ppm).expect("write ppm");
}
