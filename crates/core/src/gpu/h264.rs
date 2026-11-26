//! nvdec's H.264 picture setup (`nvdec_h264_pic_s`) turned back into the SPS and PPS it was
//! filled from: the driver keeps a stream's parameter sets and hands nvdec only slice data.

/// The bytes of the setup this reads, up to and including the lossless flags.
pub const SETUP_BYTES: usize = 0x2d4;

const STREAM_LEN: usize = 0x48;
const LOG2_MAX_POC_LSB_MINUS4: usize = 0x58;
const DELTA_PIC_ORDER_ALWAYS_ZERO: usize = 0x5c;
const FRAME_MBS_ONLY: usize = 0x60;
const WIDTH_IN_MBS: usize = 0x64;
const HEIGHT_IN_MBS: usize = 0x68;
const ENTROPY_CODING_MODE: usize = 0x70;
const PIC_ORDER_PRESENT: usize = 0x74;
const NUM_REF_IDX_L0_MINUS1: usize = 0x78;
const NUM_REF_IDX_L1_MINUS1: usize = 0x7c;
const DEBLOCKING_FILTER_CONTROL_PRESENT: usize = 0x80;
const REDUNDANT_PIC_CNT_PRESENT: usize = 0x84;
const TRANSFORM_8X8_MODE: usize = 0x88;
/// `MbaffFrameFlag` up to `second_chroma_qp_index_offset`, then `weighted_bipred_idc` and on.
const PICTURE_FLAGS: usize = 0xb0;
const PICTURE_FLAGS_2: usize = 0xb4;
/// `WeightScale[6][4][4]` then `WeightScale8x8[2][8][8]`, in raster order.
const WEIGHT_SCALE: usize = 0x1c0;
const WEIGHT_SCALE_8X8: usize = 0x220;
const LOSSLESS_FLAGS: usize = 0x2d0;

/// The fields of a setup the parameter sets need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub stream_len: u32,
    pub current_index: u32,
    log2_max_poc_lsb: u32,
    delta_pic_order_always_zero: bool,
    frame_mbs_only: bool,
    width_mbs: u32,
    height_mbs: u32,
    cabac: bool,
    pic_order_present: bool,
    num_ref_idx_l0_minus1: u32,
    num_ref_idx_l1_minus1: u32,
    deblocking_filter_control_present: bool,
    redundant_pic_cnt_present: bool,
    transform_8x8_mode: bool,
    mbaff: bool,
    direct_8x8_inference: bool,
    weighted_pred: bool,
    constrained_intra_pred: bool,
    log2_max_frame_num: u32,
    chroma_format_idc: u32,
    poc_type: u32,
    pic_init_qp_minus26: i32,
    chroma_qp_index_offset: i32,
    second_chroma_qp_index_offset: i32,
    weighted_bipred_idc: u32,
    transform_bypass: bool,
    scaling_4x4: [[u8; 16]; 6],
    scaling_8x8: [[u8; 64]; 2],
}

fn signed(value: u32, bits: u32) -> i32 {
    ((value << (32 - bits)) as i32) >> (32 - bits)
}

impl Picture {
    pub fn parse(setup: &[u8]) -> Picture {
        let word = |at: usize| {
            setup
                .get(at..at + 4)
                .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let flags = word(PICTURE_FLAGS);
        let flags_2 = word(PICTURE_FLAGS_2);
        let bits = |word: u32, at: u32, len: u32| word >> at & ((1 << len) - 1);
        let byte = |at: usize| setup.get(at).copied().unwrap_or(16);
        Picture {
            stream_len: word(STREAM_LEN),
            current_index: bits(flags_2, 2, 7),
            log2_max_poc_lsb: word(LOG2_MAX_POC_LSB_MINUS4) + 4,
            delta_pic_order_always_zero: word(DELTA_PIC_ORDER_ALWAYS_ZERO) != 0,
            frame_mbs_only: word(FRAME_MBS_ONLY) != 0,
            width_mbs: word(WIDTH_IN_MBS),
            height_mbs: word(HEIGHT_IN_MBS),
            cabac: word(ENTROPY_CODING_MODE) != 0,
            pic_order_present: word(PIC_ORDER_PRESENT) != 0,
            num_ref_idx_l0_minus1: word(NUM_REF_IDX_L0_MINUS1),
            num_ref_idx_l1_minus1: word(NUM_REF_IDX_L1_MINUS1),
            deblocking_filter_control_present: word(DEBLOCKING_FILTER_CONTROL_PRESENT) != 0,
            redundant_pic_cnt_present: word(REDUNDANT_PIC_CNT_PRESENT) != 0,
            transform_8x8_mode: word(TRANSFORM_8X8_MODE) != 0,
            mbaff: bits(flags, 0, 1) != 0,
            direct_8x8_inference: bits(flags, 1, 1) != 0,
            weighted_pred: bits(flags, 2, 1) != 0,
            constrained_intra_pred: bits(flags, 3, 1) != 0,
            log2_max_frame_num: bits(flags, 8, 4) + 4,
            chroma_format_idc: bits(flags, 12, 2),
            poc_type: bits(flags, 14, 2),
            pic_init_qp_minus26: signed(bits(flags, 16, 6), 6),
            chroma_qp_index_offset: signed(bits(flags, 22, 5), 5),
            second_chroma_qp_index_offset: signed(bits(flags, 27, 5), 5),
            weighted_bipred_idc: bits(flags_2, 0, 2),
            transform_bypass: bits(word(LOSSLESS_FLAGS), 1, 1) != 0,
            scaling_4x4: std::array::from_fn(|list| {
                std::array::from_fn(|i| byte(WEIGHT_SCALE + list * 16 + i))
            }),
            scaling_8x8: std::array::from_fn(|list| {
                std::array::from_fn(|i| byte(WEIGHT_SCALE_8X8 + list * 64 + i))
            }),
        }
    }
}

/// Bits most significant first, written out as an escaped NAL unit.
struct Writer {
    bytes: Vec<u8>,
    current: u8,
    used: u32,
}

impl Writer {
    fn new(header: u8) -> Writer {
        let mut w = Writer {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        };
        w.u(u32::from(header), 8);
        w
    }

    fn bit(&mut self, bit: bool) {
        self.current = self.current << 1 | u8::from(bit);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    fn u(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.bit(value >> i & 1 != 0);
        }
    }

    fn ue(&mut self, value: u32) {
        let coded = u64::from(value) + 1;
        let len = 64 - coded.leading_zeros();
        self.u(0, len - 1);
        for i in (0..len).rev() {
            self.bit(coded >> i & 1 != 0);
        }
    }

    fn se(&mut self, value: i32) {
        let coded = if value > 0 {
            2 * value.unsigned_abs() - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(coded);
    }

    /// The trailing bits, emulation prevention and a start code: the NAL as a stream holds it.
    fn finish(mut self, out: &mut Vec<u8>) {
        self.bit(true);
        while self.used != 0 {
            self.bit(false);
        }
        out.extend_from_slice(&[0, 0, 0, 1, self.bytes[0]]);
        let mut zeros = 0;
        for &byte in &self.bytes[1..] {
            if zeros >= 2 && byte <= 3 {
                out.push(3);
                zeros = 0;
            }
            out.push(byte);
            zeros = if byte == 0 { zeros + 1 } else { 0 };
        }
    }
}

/// Raster position of each coefficient in zigzag order.
const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `scaling_list()`: each entry as a delta from the one before it, in zigzag order.
fn scaling_list(w: &mut Writer, raster: &[u8], zigzag: &[usize]) {
    let mut last = 8i32;
    for &at in zigzag {
        let next = i32::from(raster[at]);
        w.se(((next - last + 128) & 0xff) - 128);
        last = next;
    }
}

/// The largest frame buffer and reference count a level allows, which the setup does not carry.
const MAX_REFERENCES: u32 = 16;

/// The SPS and PPS `picture` was filled from, as Annex B NAL units, the PPS numbered `pps_id`.
/// The SPS asks for no reordering, so a decoder hands each picture back as soon as it has it;
/// nvdec writes every picture to its surface at once and has no display order.
pub fn parameter_sets(picture: &Picture, pps_id: u32) -> Result<Vec<u8>, String> {
    let p = picture;
    if p.poc_type == 1 {
        return Err("picture order count type 1 keeps its cycle in the SPS alone".into());
    }
    if p.chroma_format_idc != 1 {
        return Err(format!("chroma format {}", p.chroma_format_idc));
    }
    if p.width_mbs == 0 || p.height_mbs < 2 {
        return Err(format!("{}x{} macroblocks", p.width_mbs, p.height_mbs));
    }
    let mut out = Vec::new();

    let mut sps = Writer::new(0x67);
    sps.u(100, 8); // profile_idc: High
    sps.u(0, 8); // constraint flags
    sps.u(51, 8); // level_idc
    sps.ue(0); // seq_parameter_set_id
    sps.ue(p.chroma_format_idc);
    sps.ue(0); // bit_depth_luma_minus8
    sps.ue(0); // bit_depth_chroma_minus8
    sps.bit(p.transform_bypass);
    sps.bit(false); // seq_scaling_matrix_present_flag: the PPS carries the matrices
    sps.ue(p.log2_max_frame_num - 4);
    sps.ue(p.poc_type);
    if p.poc_type == 0 {
        sps.ue(p.log2_max_poc_lsb - 4);
    }
    sps.ue(MAX_REFERENCES);
    sps.bit(false); // gaps_in_frame_num_value_allowed_flag
    sps.ue(p.width_mbs - 1);
    let map_units = if p.frame_mbs_only {
        p.height_mbs
    } else {
        p.height_mbs / 2
    };
    sps.ue(map_units - 1);
    sps.bit(p.frame_mbs_only);
    if !p.frame_mbs_only {
        sps.bit(p.mbaff);
    }
    sps.bit(p.direct_8x8_inference);
    sps.bit(false); // frame_cropping_flag
    sps.bit(true); // vui_parameters_present_flag
    for _ in 0..8 {
        // aspect ratio, overscan, video signal, chroma location, timing, both HRDs, pic_struct
        sps.bit(false);
    }
    sps.bit(true); // bitstream_restriction_flag
    sps.bit(true); // motion_vectors_over_pic_boundaries_flag
    sps.ue(0); // max_bytes_per_pic_denom
    sps.ue(0); // max_bits_per_mb_denom
    sps.ue(15); // log2_max_mv_length_horizontal
    sps.ue(15); // log2_max_mv_length_vertical
    sps.ue(0); // max_num_reorder_frames
    sps.ue(MAX_REFERENCES); // max_dec_frame_buffering
    sps.finish(&mut out);

    let mut pps = Writer::new(0x68);
    pps.ue(pps_id);
    pps.ue(0); // seq_parameter_set_id
    pps.bit(p.cabac);
    pps.bit(p.pic_order_present);
    pps.ue(0); // num_slice_groups_minus1
    pps.ue(p.num_ref_idx_l0_minus1);
    pps.ue(p.num_ref_idx_l1_minus1);
    pps.bit(p.weighted_pred);
    pps.u(p.weighted_bipred_idc, 2);
    pps.se(p.pic_init_qp_minus26);
    pps.se(0); // pic_init_qs_minus26: SP and SI slices only
    pps.se(p.chroma_qp_index_offset);
    pps.bit(p.deblocking_filter_control_present);
    pps.bit(p.constrained_intra_pred);
    pps.bit(p.redundant_pic_cnt_present);
    pps.bit(p.transform_8x8_mode);
    let flat = p.scaling_4x4.iter().flatten().all(|&v| v == 16)
        && p.scaling_8x8.iter().flatten().all(|&v| v == 16);
    pps.bit(!flat);
    if !flat {
        let lists = 6 + if p.transform_8x8_mode { 2 } else { 0 };
        for list in 0..lists {
            pps.bit(true); // pic_scaling_list_present_flag
            match p.scaling_4x4.get(list) {
                Some(raster) => scaling_list(&mut pps, raster, &ZIGZAG_4X4),
                None => scaling_list(&mut pps, &p.scaling_8x8[list - 6], &ZIGZAG_8X8),
            }
        }
    }
    pps.se(p.second_chroma_qp_index_offset);
    pps.finish(&mut out);
    Ok(out)
}

/// The slices of one picture as nvdec reads them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnit {
    pub bytes: Vec<u8>,
    /// An IDR slice, which a decoder can start from.
    pub key: bool,
    pub pps_id: u32,
}

/// The slice NAL units of `bitstream`, without the end-of-stream units nvdec's driver appends.
pub fn access_unit(bitstream: &[u8]) -> Option<AccessUnit> {
    let starts: Vec<usize> = (0..bitstream.len().saturating_sub(2))
        .filter(|&i| bitstream[i] == 0 && bitstream[i + 1] == 0 && bitstream[i + 2] == 1)
        .collect();
    let mut bytes = Vec::with_capacity(bitstream.len());
    let (mut key, mut pps_id) = (false, None);
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(bitstream.len());
        let payload = bitstream.get(start + 3..end)?;
        let kind = payload.first()? & 0x1f;
        if !matches!(kind, 1 | 5) {
            continue;
        }
        key |= kind == 5;
        if pps_id.is_none() {
            pps_id = slice_pps_id(&payload[1..]);
        }
        bytes.extend_from_slice(&[0, 0, 1]);
        bytes.extend_from_slice(payload);
    }
    Some(AccessUnit {
        bytes,
        key,
        pps_id: pps_id?,
    })
}

/// `pic_parameter_set_id`, the third field of a slice header.
fn slice_pps_id(header: &[u8]) -> Option<u32> {
    let mut at = 0usize;
    let mut bit = || {
        let b = header.get(at / 8)? >> (7 - at % 8) & 1;
        at += 1;
        Some(b)
    };
    let mut ue = || -> Option<u32> {
        let mut zeros = 0;
        while bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let mut value = 0u32;
        for _ in 0..zeros {
            value = value << 1 | u32::from(bit()?);
        }
        Some((1 << zeros) - 1 + value)
    };
    ue()?; // first_mb_in_slice
    ue()?; // slice_type
    ue()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads back what [`Writer`] wrote, emulation prevention removed.
    struct Reader {
        bits: Vec<bool>,
        at: usize,
    }

    impl Reader {
        /// The NAL unit after the start code at `at`, past its header byte.
        fn nal(stream: &[u8], index: usize) -> (u8, Reader) {
            let starts: Vec<usize> = (0..stream.len() - 3)
                .filter(|&i| stream[i..i + 4] == [0, 0, 0, 1])
                .collect();
            let start = starts[index] + 4;
            let end = starts.get(index + 1).copied().unwrap_or(stream.len());
            let mut rbsp = Vec::new();
            let mut zeros = 0;
            for &byte in &stream[start + 1..end] {
                if zeros >= 2 && byte == 3 {
                    zeros = 0;
                    continue;
                }
                rbsp.push(byte);
                zeros = if byte == 0 { zeros + 1 } else { 0 };
            }
            let bits = rbsp
                .iter()
                .flat_map(|byte| (0..8).rev().map(move |i| byte >> i & 1 != 0))
                .collect();
            (stream[start], Reader { bits, at: 0 })
        }

        fn u(&mut self, n: usize) -> u32 {
            let v = self.bits[self.at..self.at + n]
                .iter()
                .fold(0, |v, &b| v << 1 | u32::from(b));
            self.at += n;
            v
        }

        fn ue(&mut self) -> u32 {
            let mut zeros = 0;
            while !self.bits[self.at] {
                zeros += 1;
                self.at += 1;
            }
            self.at += 1;
            (1 << zeros) - 1 + self.u(zeros)
        }

        fn se(&mut self) -> i32 {
            let k = self.ue() as i32;
            if k & 1 == 1 {
                (k + 1) / 2
            } else {
                -k / 2
            }
        }
    }

    /// Mario Kart 8 Deluxe's first menu movie picture: setup words `+0x40` to `+0xbf`.
    fn mario_kart_setup() -> Vec<u8> {
        let words: [u32; 32] = [
            0,
            0,
            0xa8fe,
            1,
            0xd00,
            0,
            1,
            0,
            1,
            0x20,
            0x26,
            0,
            1,
            0,
            5,
            0,
            1,
            0,
            0,
            0x200,
            0x200,
            0,
            0,
            0,
            0,
            0,
            0,
            0x52,
            0xe73f_1016,
            2,
            0,
            0,
        ];
        let mut setup = vec![0u8; SETUP_BYTES];
        for (i, word) in words.iter().enumerate() {
            setup[0x40 + i * 4..0x44 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        setup[WEIGHT_SCALE..LOSSLESS_FLAGS].fill(16);
        setup
    }

    #[test]
    fn the_parameter_sets_carry_what_the_setup_says_in_the_order_the_standard_reads_them() {
        let sets = parameter_sets(&Picture::parse(&mario_kart_setup()), 3).unwrap();

        let (header, mut sps) = Reader::nal(&sets, 0);
        assert_eq!(header, 0x67);
        assert_eq!(sps.u(8), 100, "profile_idc");
        sps.u(16);
        let fields = [sps.ue(), sps.ue(), sps.ue(), sps.ue()];
        assert_eq!(fields, [0, 1, 0, 0], "id, chroma format, bit depths");
        assert_eq!((sps.u(1), sps.u(1)), (0, 0), "no bypass, no SPS matrices");
        let fields = [sps.ue(), sps.ue(), sps.ue(), sps.ue()];
        assert_eq!(
            fields,
            [0, 0, 1, MAX_REFERENCES],
            "frame_num bits, POC type, POC bits, references"
        );
        sps.u(1);
        assert_eq!((sps.ue(), sps.ue()), (31, 37), "512x608 in macroblocks");
        assert_eq!(
            (sps.u(1), sps.u(1), sps.u(1)),
            (1, 1, 0),
            "frames, 8x8 inference, no crop"
        );
        assert_eq!(sps.u(1), 1, "VUI");
        sps.u(8);
        assert_eq!((sps.u(1), sps.u(1)), (1, 1), "bitstream restriction");
        let vui = [sps.ue(), sps.ue(), sps.ue(), sps.ue(), sps.ue(), sps.ue()];
        assert_eq!(vui, [0, 0, 15, 15, 0, MAX_REFERENCES], "no reordering");

        let (header, mut pps) = Reader::nal(&sets, 1);
        assert_eq!(header, 0x68);
        assert_eq!((pps.ue(), pps.ue()), (3, 0), "PPS and SPS ids");
        assert_eq!((pps.u(1), pps.u(1)), (1, 0), "CABAC, no bottom field POC");
        assert_eq!(
            (pps.ue(), pps.ue(), pps.ue()),
            (0, 5, 0),
            "slice groups, references"
        );
        assert_eq!((pps.u(1), pps.u(2)), (1, 2), "weighted P, implicit bipred");
        assert_eq!(
            (pps.se(), pps.se(), pps.se()),
            (-1, 0, -4),
            "QP 25, QS, chroma offset"
        );
        assert_eq!(
            (pps.u(1), pps.u(1), pps.u(1)),
            (1, 0, 0),
            "deblocking control"
        );
        assert_eq!((pps.u(1), pps.u(1)), (0, 0), "no 8x8, no matrices");
        assert_eq!(pps.se(), -4, "second chroma offset");
    }

    #[test]
    fn a_scaling_matrix_reaches_the_pps_in_zigzag_order() {
        let mut setup = mario_kart_setup();
        setup[TRANSFORM_8X8_MODE] = 1;
        for (i, v) in setup[WEIGHT_SCALE_8X8..WEIGHT_SCALE_8X8 + 64]
            .iter_mut()
            .enumerate()
        {
            *v = 6 + i as u8 * 3;
        }
        let sets = parameter_sets(&Picture::parse(&setup), 0).unwrap();
        let (_, mut pps) = Reader::nal(&sets, 1);
        // Past every field before transform_8x8_mode_flag.
        for _ in 0..2 {
            pps.ue();
        }
        pps.u(2);
        for _ in 0..3 {
            pps.ue();
        }
        pps.u(3);
        for _ in 0..3 {
            pps.se();
        }
        pps.u(3);
        assert_eq!((pps.u(1), pps.u(1)), (1, 1), "8x8, matrices");
        let mut read_list = |len: usize| {
            assert_eq!(pps.u(1), 1, "list present");
            let mut last = 8i32;
            (0..len)
                .map(|_| {
                    last = (last + pps.se() + 256) % 256;
                    last as u8
                })
                .collect::<Vec<u8>>()
        };
        for _ in 0..6 {
            assert_eq!(read_list(16), vec![16; 16]);
        }
        let zigzag = read_list(64);
        let raster: Vec<u8> = (0..64)
            .map(|at| zigzag[ZIGZAG_8X8.iter().position(|&z| z == at).unwrap()])
            .collect();
        assert_eq!(raster, (0..64).map(|i| 6 + i * 3).collect::<Vec<u8>>());
    }

    #[test]
    fn an_access_unit_keeps_its_slices_and_drops_the_end_of_stream() {
        // An IDR slice whose header is first_mb 0, slice type 7 and PPS 2, then the driver's tail.
        let mut bitstream = vec![0, 0, 1, 0x65, 0b1000_1000, 0b0110_0000, 0xab];
        bitstream.extend_from_slice(&[0, 0, 1, 0x0b, 0, 0, 0, 0, 0, 0, 1, 0x0b, 0, 0, 0, 0]);
        let unit = access_unit(&bitstream).unwrap();
        assert_eq!(unit.bytes, bitstream[..7]);
        assert!(unit.key);
        assert_eq!(unit.pps_id, 2);
    }
}
