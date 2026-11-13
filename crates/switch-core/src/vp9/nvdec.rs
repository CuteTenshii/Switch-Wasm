//! nvdec's view of a VP9 frame: the picture setup struct, the probability table and the counters
//! it shares with the driver, mapped onto the decoder's own types.

use super::frame::Frame;
use super::header::{FrameHeader, LoopFilterParams, SegFeature, Segmentation};
use super::probs::FrameContext;

fn u16_at(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([data[at], data[at + 1]])
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

/// A frame size as the picture setup states it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dimensions {
    pub width: u16,
    pub height: u16,
    pub luma_pitch: u16,
    pub chroma_pitch: u16,
}

#[derive(Clone, Debug)]
pub struct PicSetup {
    pub bitstream_size: u32,
    pub last_size: Dimensions,
    pub golden_size: Dimensions,
    pub alt_size: Dimensions,
    pub cur_size: Dimensions,
    pub flags: u32,
    pub header: FrameHeader,
    pub seg: Segmentation,
    pub lf: LoopFilterParams,
}

impl PicSetup {
    pub fn is_key_frame(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn last_frame_is_key(&self) -> bool {
        self.flags >> 1 & 1 != 0
    }

    pub fn last_show_frame(&self) -> bool {
        self.flags >> 4 & 1 != 0
    }

    /// Whether the previous frame's motion vectors predict this frame's.
    pub fn use_prev_frame_mvs(&self) -> bool {
        !self.header.error_resilient
            && self.last_size == self.cur_size
            && self.last_size.width != 0
            && !self.last_frame_is_key()
            && self.last_show_frame()
    }

    pub fn parse(data: &[u8]) -> PicSetup {
        let dims = |at: usize| Dimensions {
            width: u16_at(data, at),
            height: u16_at(data, at + 2),
            luma_pitch: u16_at(data, at + 4),
            chroma_pitch: u16_at(data, at + 6),
        };
        let flags = u32_at(data, 0x68);
        let cur_size = dims(0x60);
        let mut header = FrameHeader {
            key_frame: flags & 1 != 0,
            error_resilient: flags >> 3 & 1 != 0,
            frame_parallel: flags >> 2 & 1 != 0,
            intra_only: flags >> 5 & 1 != 0,
            width: u32::from(cur_size.width),
            height: u32::from(cur_size.height),
            ..Default::default()
        };
        for i in 0..4 {
            header.sign_bias[i] = data[0x6c + i] != 0;
        }
        let lf_level = data[0x70];
        let lf_sharpness = data[0x71];
        header.base_qindex = i32::from(data[0x72]);
        header.y_dc_delta_q = i32::from(data[0x73] as i8);
        header.uv_ac_delta_q = i32::from(data[0x74] as i8);
        header.uv_dc_delta_q = i32::from(data[0x75] as i8);
        header.lossless = data[0x76] != 0;
        header.tx_mode = data[0x77];
        header.allow_hp = data[0x78] != 0;
        header.interp_filter = data[0x79];
        header.reference_mode = data[0x7a];
        header.comp_fixed_ref = data[0x7b] as i8;
        header.comp_var_ref = [data[0x7c] as i8, data[0x7d] as i8];
        header.log2_tile_cols = u32::from(data[0x7e]);
        header.log2_tile_rows = u32::from(data[0x7f]);

        let mut seg = Segmentation {
            enabled: data[0x80] != 0,
            update_map: data[0x81] != 0,
            temporal_update: data[0x82] != 0,
            abs_delta: data[0x83] != 0,
            ..Default::default()
        };
        for segment in 0..8 {
            let mask = u32_at(data, 0x84 + segment * 4);
            for feature in 0..4 {
                seg.features[segment][feature] = SegFeature {
                    enabled: mask >> feature & 1 != 0,
                    data: u16_at(data, 0xa4 + segment * 8 + feature * 2) as i16,
                };
            }
        }

        let lf = LoopFilterParams {
            level: lf_level,
            sharpness: lf_sharpness,
            delta_enabled: data[0xe4] != 0,
            ref_deltas: [
                data[0xe5] as i8,
                data[0xe6] as i8,
                data[0xe7] as i8,
                data[0xe8] as i8,
            ],
            mode_deltas: [data[0xe9] as i8, data[0xea] as i8],
        };
        PicSetup {
            bitstream_size: u32_at(data, 0x30),
            last_size: dims(0x48),
            golden_size: dims(0x50),
            alt_size: dims(0x58),
            cur_size,
            flags,
            header,
            seg,
            lf,
        }
    }
}

/// Where each table lives in nvdec's probability buffer. Rows are padded to four bytes where the
/// libvpx tables have three, and the intra mode tables keep their ninth probability apart.
mod table {
    pub const INTER_MODE: usize = 1024;
    pub const INTRA_INTER: usize = 1052;
    pub const UV_MODE_FIRST8: usize = 1056;
    pub const TX: usize = 1136;
    pub const Y_MODE_NINTH: usize = 1148;
    pub const Y_MODE_FIRST8: usize = 1152;
    pub const PARTITION: usize = 1248;
    pub const UV_MODE_NINTH: usize = 1312;
    pub const SWITCHABLE_INTERP: usize = 1322;
    pub const COMP_INTER: usize = 1330;
    pub const SKIP: usize = 1335;
    pub const MV_JOINTS: usize = 1339;
    pub const MV_SIGN: usize = 1342;
    pub const MV_CLASS0: usize = 1344;
    pub const MV_FP: usize = 1346;
    pub const MV_CLASS0_HP: usize = 1352;
    pub const MV_HP: usize = 1354;
    pub const MV_CLASSES: usize = 1356;
    pub const MV_CLASS0_FP: usize = 1376;
    pub const MV_BITS: usize = 1388;
    pub const SINGLE_REF: usize = 1408;
    pub const COMP_REF: usize = 1418;
    pub const COEF: usize = 1440;
    pub const SEG_TREE: usize = 903;
    pub const SEG_PRED: usize = 910;
}

pub fn parse_prob_table(p: &[u8]) -> FrameContext {
    let mut fc = FrameContext::defaults();
    for i in 0..7 {
        fc.inter_mode[i].copy_from_slice(&p[table::INTER_MODE + i * 4..][..3]);
    }
    fc.intra_inter
        .copy_from_slice(&p[table::INTRA_INTER..][..4]);
    for i in 0..10 {
        fc.uv_mode[i][..8].copy_from_slice(&p[table::UV_MODE_FIRST8 + i * 8..][..8]);
        fc.uv_mode[i][8] = p[table::UV_MODE_NINTH + i];
    }
    fc.tx8 = [[p[table::TX]], [p[table::TX + 1]]];
    fc.tx16 = [
        [p[table::TX + 2], p[table::TX + 3]],
        [p[table::TX + 4], p[table::TX + 5]],
    ];
    fc.tx32 = [
        [p[table::TX + 6], p[table::TX + 7], p[table::TX + 8]],
        [p[table::TX + 9], p[table::TX + 10], p[table::TX + 11]],
    ];
    for i in 0..4 {
        fc.y_mode[i][..8].copy_from_slice(&p[table::Y_MODE_FIRST8 + i * 8..][..8]);
        fc.y_mode[i][8] = p[table::Y_MODE_NINTH + i];
    }
    for i in 0..16 {
        fc.partition[i].copy_from_slice(&p[table::PARTITION + i * 4..][..3]);
    }
    for i in 0..4 {
        fc.switchable_interp[i].copy_from_slice(&p[table::SWITCHABLE_INTERP + i * 2..][..2]);
    }
    fc.comp_inter.copy_from_slice(&p[table::COMP_INTER..][..5]);
    fc.skip.copy_from_slice(&p[table::SKIP..][..3]);
    fc.mv.joints.copy_from_slice(&p[table::MV_JOINTS..][..3]);
    for c in 0..2 {
        let comp = &mut fc.mv.comps[c];
        comp.sign = p[table::MV_SIGN + c];
        comp.class0 = [p[table::MV_CLASS0 + c]];
        comp.fp.copy_from_slice(&p[table::MV_FP + c * 3..][..3]);
        comp.class0_hp = p[table::MV_CLASS0_HP + c];
        comp.hp = p[table::MV_HP + c];
        comp.classes
            .copy_from_slice(&p[table::MV_CLASSES + c * 10..][..10]);
        for j in 0..2 {
            comp.class0_fp[j].copy_from_slice(&p[table::MV_CLASS0_FP + c * 6 + j * 3..][..3]);
        }
        comp.bits
            .copy_from_slice(&p[table::MV_BITS + c * 10..][..10]);
    }
    for i in 0..5 {
        fc.single_ref[i].copy_from_slice(&p[table::SINGLE_REF + i * 2..][..2]);
    }
    fc.comp_ref.copy_from_slice(&p[table::COMP_REF..][..5]);
    for tx in 0..4 {
        for plane in 0..2 {
            for r in 0..2 {
                for band in 0..6 {
                    for ctx in 0..6 {
                        let at =
                            table::COEF + (((tx * 2 + plane) * 2 + r) * 6 + band) * 24 + ctx * 4;
                        fc.coef[tx][plane][r][band][ctx].copy_from_slice(&p[at..at + 3]);
                    }
                }
            }
        }
    }
    fc
}

/// The decoder state that lives as long as an nvdec channel: the previous frame and the
/// segmentation maps the driver keeps in its own buffers.
#[derive(Debug, Default)]
pub struct Channel {
    prev: Option<std::sync::Arc<Frame>>,
    last_seg_map: Vec<u8>,
    cur_seg_map: Vec<u8>,
}

impl Channel {
    /// Decodes one frame from what an nvdec `EXECUTE` names: the picture setup, the tile data, the
    /// probability table and the three reference frames (`None` where the driver has none).
    pub fn decode(
        &mut self,
        setup: &[u8],
        bitstream: &[u8],
        probs: &[u8],
        refs: [Option<std::sync::Arc<Frame>>; 3],
    ) -> Result<std::sync::Arc<Frame>, super::Error> {
        let setup = PicSetup::parse(setup);
        let header = setup.header.clone();
        let maps = header.mi_cols() * header.mi_rows();
        if self.last_seg_map.len() != maps {
            self.last_seg_map = vec![0; maps];
            self.cur_seg_map = vec![0; maps];
        }
        let size = (setup.bitstream_size as usize).min(bitstream.len());
        let refs = if setup.is_key_frame() || header.intra_only {
            Default::default()
        } else {
            refs.map(|frame| frame.filter(|f| f.width == header.width && f.height == header.height))
        };
        if !(setup.is_key_frame() || header.intra_only) && refs.iter().any(Option::is_none) {
            return Err(super::Error::Corrupt(
                "a reference frame is missing or differs in size",
            ));
        }
        let input = super::decoder::FrameInput {
            hdr: header,
            fc: parse_prob_table(probs),
            seg: {
                let mut seg = setup.seg.clone();
                seg.tree_probs
                    .copy_from_slice(&probs[table::SEG_TREE..table::SEG_PRED]);
                seg.pred_probs
                    .copy_from_slice(&probs[table::SEG_PRED..table::SEG_PRED + 3]);
                seg
            },
            lf: setup.lf.clone(),
            refs,
            prev_mvs: self.prev.clone(),
            use_prev_frame_mvs: setup.use_prev_frame_mvs() && self.prev.is_some(),
            last_seg_map: std::mem::take(&mut self.last_seg_map),
            cur_seg_map: std::mem::take(&mut self.cur_seg_map),
            count: false,
        };
        let output = super::decoder::decode_pixels(input, &bitstream[..size]);
        self.last_seg_map = output.last_seg_map;
        self.cur_seg_map = output.cur_seg_map;
        output.result?;
        if setup.seg.enabled {
            std::mem::swap(&mut self.last_seg_map, &mut self.cur_seg_map);
        }
        let frame = std::sync::Arc::new(output.frame);
        self.prev = Some(frame.clone());
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_bytes() -> Vec<u8> {
        let mut d = vec![0u8; 0x100];
        let put32 =
            |d: &mut Vec<u8>, at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put32(&mut d, 0x30, 26618);
        put32(&mut d, 0x60, 0x02d0_0500);
        put32(&mut d, 0x64, 0x0500_0500);
        put32(&mut d, 0x68, 0x13);
        put32(&mut d, 0x70, 0x0053_000e);
        put32(&mut d, 0x74, 0x0300_0000);
        put32(&mut d, 0x78, 0x0302_0401);
        put32(&mut d, 0x7c, 0x0002_0201);
        d[0xe4..0xeb].copy_from_slice(&[1, 1, 0, 0xff, 0xff, 0, 0]);
        d
    }

    #[test]
    fn a_picture_setup_names_the_frame_and_its_coding_tools() {
        let setup = PicSetup::parse(&setup_bytes());
        let h = &setup.header;
        assert_eq!((h.width, h.height), (1280, 720));
        assert_eq!(setup.bitstream_size, 26618);
        assert!(setup.is_key_frame() && setup.last_frame_is_key() && setup.last_show_frame());
        assert_eq!((setup.lf.level, h.base_qindex, h.tx_mode), (14, 0x53, 3));
        assert_eq!(
            (
                h.allow_hp,
                h.interp_filter,
                h.reference_mode,
                h.comp_fixed_ref
            ),
            (true, 4, 2, 3)
        );
        assert_eq!((h.comp_var_ref, h.log2_tile_cols), ([1, 2], 2));
        assert_eq!(setup.lf.ref_deltas, [1, 0, -1, -1]);
    }

    #[test]
    fn previous_motion_vectors_need_a_shown_non_key_previous_frame_of_the_same_size() {
        let mut bytes = setup_bytes();
        bytes[0x48..0x50].copy_from_slice(&[0, 5, 0xd0, 2, 0, 5, 0, 5]);
        let mut setup = PicSetup::parse(&bytes);
        assert!(
            !setup.use_prev_frame_mvs(),
            "the previous frame was a key frame"
        );
        setup.flags = 0x10;
        assert!(setup.use_prev_frame_mvs());
        setup.flags = 0;
        assert!(!setup.use_prev_frame_mvs(), "the previous frame was hidden");
    }

    #[test]
    fn the_probability_table_is_read_from_its_documented_offsets() {
        let mut p = vec![0u8; 3744];
        p[table::SKIP..table::SKIP + 3].copy_from_slice(&[242, 128, 64]);
        p[table::MV_JOINTS..table::MV_JOINTS + 3].copy_from_slice(&[32, 64, 96]);
        p[table::MV_CLASS0 + 1] = 208;
        p[table::Y_MODE_FIRST8 + 8] = 7;
        p[table::Y_MODE_NINTH + 1] = 99;
        p[table::COEF + 24 + 8] = 55;
        let fc = parse_prob_table(&p);
        assert_eq!(fc.skip, [242, 128, 64]);
        assert_eq!(fc.mv.joints, [32, 64, 96]);
        assert_eq!(fc.mv.comps[1].class0, [208]);
        assert_eq!((fc.y_mode[1][0], fc.y_mode[1][8]), (7, 99));
        assert_eq!(fc.coef[0][0][0][1][2], [55, 0, 0]);
    }
}
