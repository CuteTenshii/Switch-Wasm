//! The uncompressed and compressed frame headers.

use super::bool_decoder::{BitReader, BoolDecoder};
use super::decoder::Decoder;
use super::probs::FrameContext;
use super::tables::INV_MAP_TABLE;
use super::types::*;
use super::Error;

pub const MAX_SEGMENTS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegFeature {
    pub enabled: bool,
    pub data: i16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub update_data: bool,
    pub abs_delta: bool,
    pub temporal_update: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    pub features: [[SegFeature; 4]; MAX_SEGMENTS],
}

impl Default for Segmentation {
    fn default() -> Segmentation {
        Segmentation {
            enabled: false,
            update_map: false,
            update_data: false,
            abs_delta: false,
            temporal_update: false,
            tree_probs: [255; 7],
            pred_probs: [255; 3],
            features: Default::default(),
        }
    }
}

impl Segmentation {
    pub fn active(&self, segment: usize, feature: usize) -> bool {
        self.enabled && self.features[segment][feature].enabled
    }

    pub fn data(&self, segment: usize, feature: usize) -> i32 {
        i32::from(self.features[segment][feature].data)
    }

    /// The quantizer index of `segment`, given the frame's base index.
    pub fn qindex(&self, segment: usize, base: i32) -> i32 {
        if self.active(segment, SEG_LVL_ALT_Q) {
            let data = self.data(segment, SEG_LVL_ALT_Q);
            (if self.abs_delta { data } else { base + data }).clamp(0, 255)
        } else {
            base
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFilterParams {
    pub level: u8,
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilterParams {
    fn default() -> LoopFilterParams {
        LoopFilterParams {
            level: 0,
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show: usize,
    pub key_frame: bool,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [usize; 3],
    pub sign_bias: [bool; 4],
    pub width: u32,
    pub height: u32,
    pub allow_hp: bool,
    pub interp_filter: u8,
    pub refresh_frame_context: bool,
    pub frame_parallel: bool,
    pub frame_context_idx: usize,
    pub base_qindex: i32,
    pub y_dc_delta_q: i32,
    pub uv_dc_delta_q: i32,
    pub uv_ac_delta_q: i32,
    pub lossless: bool,
    pub log2_tile_cols: u32,
    pub log2_tile_rows: u32,
    pub tx_mode: u8,
    pub reference_mode: u8,
    pub comp_fixed_ref: i8,
    pub comp_var_ref: [i8; 2],
    pub compressed_header_size: usize,
    pub uncompressed_header_size: usize,
}

impl FrameHeader {
    pub fn is_intra(&self) -> bool {
        self.key_frame || self.intra_only
    }

    pub fn mi_cols(&self) -> usize {
        (self.width as usize).div_ceil(8)
    }

    pub fn mi_rows(&self) -> usize {
        (self.height as usize).div_ceil(8)
    }
}

fn read_delta_q(rb: &mut BitReader) -> Result<i32, Error> {
    if rb.read_bit()? {
        rb.read_signed(4)
    } else {
        Ok(0)
    }
}

fn read_frame_size(rb: &mut BitReader) -> Result<(u32, u32), Error> {
    Ok((rb.read(16)? + 1, rb.read(16)? + 1))
}

fn read_sync_code(rb: &mut BitReader) -> Result<(), Error> {
    if rb.read(8)? == 0x49 && rb.read(8)? == 0x83 && rb.read(8)? == 0x42 {
        Ok(())
    } else {
        Err(Error::Corrupt("invalid frame sync code"))
    }
}

/// Profile 0 only: 8-bit 4:2:0.
fn read_color_config(rb: &mut BitReader, profile: u8) -> Result<(), Error> {
    if profile != 0 {
        return Err(Error::Unsupported("only profile 0 is supported"));
    }
    let color_space = rb.read(3)?;
    if color_space == 7 {
        return Err(Error::Unsupported("4:4:4 is not supported in profile 0"));
    }
    rb.read_bit()?;
    Ok(())
}

impl Decoder {
    /// Parses the uncompressed header at the start of `data`, applying its side effects on the
    /// decoder's persistent state (loop filter deltas, segmentation, probability contexts).
    pub(super) fn read_uncompressed_header(&mut self, data: &[u8]) -> Result<FrameHeader, Error> {
        let mut rb = BitReader::new(data);
        let mut hdr = FrameHeader::default();
        self.last_frame_key = self.frame_key;
        self.last_intra_only = self.intra_only;

        if rb.read(2)? != 2 {
            return Err(Error::Corrupt("invalid frame marker"));
        }
        let profile = rb.read(1)? as u8 | (rb.read(1)? as u8) << 1;
        if profile > 2 {
            rb.read_bit()?;
        }
        if profile != 0 {
            return Err(Error::Unsupported("only profile 0 is supported"));
        }

        hdr.show_existing_frame = rb.read_bit()?;
        if hdr.show_existing_frame {
            hdr.frame_to_show = rb.read(3)? as usize;
            hdr.show_frame = true;
            hdr.uncompressed_header_size = rb.bytes_read();
            return Ok(hdr);
        }

        hdr.key_frame = !rb.read_bit()?;
        hdr.show_frame = rb.read_bit()?;
        hdr.error_resilient = rb.read_bit()?;

        if hdr.key_frame {
            read_sync_code(&mut rb)?;
            read_color_config(&mut rb, profile)?;
            hdr.refresh_frame_flags = 0xff;
            (hdr.width, hdr.height) = read_frame_size(&mut rb)?;
            self.read_render_size(&mut rb)?;
        } else {
            hdr.intra_only = if hdr.show_frame {
                false
            } else {
                rb.read_bit()?
            };
            hdr.reset_frame_context = if hdr.error_resilient {
                0
            } else {
                rb.read(2)? as u8
            };
            if hdr.intra_only {
                read_sync_code(&mut rb)?;
                hdr.refresh_frame_flags = rb.read(8)? as u8;
                (hdr.width, hdr.height) = read_frame_size(&mut rb)?;
                self.read_render_size(&mut rb)?;
            } else {
                hdr.refresh_frame_flags = rb.read(8)? as u8;
                for i in 0..3 {
                    hdr.ref_frame_idx[i] = rb.read(3)? as usize;
                    hdr.sign_bias[1 + i] = rb.read_bit()?;
                    if self.ref_frame_map[hdr.ref_frame_idx[i]].is_none() {
                        return Err(Error::Corrupt("invalid reference frame index"));
                    }
                }
                let mut size = None;
                for i in 0..3 {
                    if rb.read_bit()? {
                        let frame = self.ref_frame_map[hdr.ref_frame_idx[i]].as_ref().unwrap();
                        size = Some((frame.width, frame.height));
                        break;
                    }
                }
                (hdr.width, hdr.height) = match size {
                    Some(size) => size,
                    None => read_frame_size(&mut rb)?,
                };
                self.read_render_size(&mut rb)?;
                hdr.allow_hp = rb.read_bit()?;
                hdr.interp_filter = if rb.read_bit()? {
                    SWITCHABLE
                } else {
                    [EIGHTTAP_SMOOTH, EIGHTTAP, EIGHTTAP_SHARP, BILINEAR][rb.read(2)? as usize]
                };
            }
        }

        if hdr.error_resilient {
            hdr.refresh_frame_context = false;
            hdr.frame_parallel = true;
        } else {
            hdr.refresh_frame_context = rb.read_bit()?;
            hdr.frame_parallel = rb.read_bit()?;
        }
        hdr.frame_context_idx = rb.read(2)? as usize;

        self.frame_key = hdr.key_frame;
        self.intra_only = if hdr.key_frame {
            self.intra_only
        } else {
            hdr.intra_only
        };

        if hdr.is_intra() || hdr.error_resilient {
            self.setup_past_independence(&mut hdr);
        }

        self.read_loop_filter(&mut rb, &mut hdr)?;

        hdr.base_qindex = rb.read(8)? as i32;
        hdr.y_dc_delta_q = read_delta_q(&mut rb)?;
        hdr.uv_dc_delta_q = read_delta_q(&mut rb)?;
        hdr.uv_ac_delta_q = read_delta_q(&mut rb)?;
        hdr.lossless = hdr.base_qindex == 0
            && hdr.y_dc_delta_q == 0
            && hdr.uv_dc_delta_q == 0
            && hdr.uv_ac_delta_q == 0;

        self.read_segmentation(&mut rb)?;

        let sb_cols = hdr.mi_cols().div_ceil(8);
        let mut min_log2 = 0u32;
        while (64usize << min_log2) < sb_cols {
            min_log2 += 1;
        }
        let mut max_log2 = 1u32;
        while (sb_cols >> max_log2) >= 4 {
            max_log2 += 1;
        }
        max_log2 -= 1;
        hdr.log2_tile_cols = min_log2;
        for _ in 0..max_log2.saturating_sub(min_log2) {
            if !rb.read_bit()? {
                break;
            }
            hdr.log2_tile_cols += 1;
        }
        hdr.log2_tile_rows = u32::from(rb.read_bit()?);
        if hdr.log2_tile_rows != 0 {
            hdr.log2_tile_rows += u32::from(rb.read_bit()?);
        }

        hdr.compressed_header_size = rb.read(16)? as usize;
        if hdr.compressed_header_size == 0 {
            return Err(Error::Corrupt("invalid header size"));
        }
        hdr.uncompressed_header_size = rb.bytes_read();
        Ok(hdr)
    }

    fn read_render_size(&mut self, rb: &mut BitReader) -> Result<(), Error> {
        if rb.read_bit()? {
            read_frame_size(rb)?;
        }
        Ok(())
    }

    fn setup_past_independence(&mut self, hdr: &mut FrameHeader) {
        self.seg.features = Default::default();
        self.seg.abs_delta = false;
        self.lf = LoopFilterParams::default();
        self.last_seg_map.fill(0);
        self.current_seg_map.fill(0);
        self.fc = FrameContext::defaults();
        if hdr.key_frame || hdr.error_resilient || hdr.reset_frame_context == 3 {
            self.frame_contexts = [self.fc; 4];
        } else if hdr.reset_frame_context == 2 {
            self.frame_contexts[hdr.frame_context_idx] = self.fc;
        }
        hdr.sign_bias = [false; 4];
        hdr.frame_context_idx = 0;
    }

    fn read_loop_filter(&mut self, rb: &mut BitReader, hdr: &mut FrameHeader) -> Result<(), Error> {
        let _ = hdr;
        self.lf.level = rb.read(6)? as u8;
        self.lf.sharpness = rb.read(3)? as u8;
        self.lf.delta_enabled = rb.read_bit()?;
        if self.lf.delta_enabled && rb.read_bit()? {
            for i in 0..4 {
                if rb.read_bit()? {
                    self.lf.ref_deltas[i] = rb.read_signed(6)? as i8;
                }
            }
            for i in 0..2 {
                if rb.read_bit()? {
                    self.lf.mode_deltas[i] = rb.read_signed(6)? as i8;
                }
            }
        }
        Ok(())
    }

    fn read_segmentation(&mut self, rb: &mut BitReader) -> Result<(), Error> {
        const DATA_MAX: [i32; 4] = [255, 63, 3, 0];
        const BITS: [u32; 4] = [8, 6, 2, 0];
        const SIGNED: [bool; 4] = [true, true, false, false];
        let seg = &mut self.seg;
        seg.update_map = false;
        seg.update_data = false;
        seg.enabled = rb.read_bit()?;
        if !seg.enabled {
            return Ok(());
        }
        seg.update_map = rb.read_bit()?;
        if seg.update_map {
            for prob in &mut seg.tree_probs {
                *prob = if rb.read_bit()? {
                    rb.read(8)? as u8
                } else {
                    255
                };
            }
            seg.temporal_update = rb.read_bit()?;
            for prob in &mut seg.pred_probs {
                *prob = if seg.temporal_update && rb.read_bit()? {
                    rb.read(8)? as u8
                } else {
                    255
                };
            }
        }
        seg.update_data = rb.read_bit()?;
        if seg.update_data {
            seg.abs_delta = rb.read_bit()?;
            seg.features = Default::default();
            for segment in 0..MAX_SEGMENTS {
                for feature in 0..4 {
                    let mut data = 0;
                    let enabled = rb.read_bit()?;
                    if enabled {
                        data = (rb.read(BITS[feature])? as i32).min(DATA_MAX[feature]);
                        if SIGNED[feature] && rb.read_bit()? {
                            data = -data;
                        }
                    }
                    seg.features[segment][feature] = SegFeature {
                        enabled,
                        data: data as i16,
                    };
                }
            }
        }
        Ok(())
    }

    /// Applies the compressed header's probability updates to `self.fc` and returns the modes it
    /// chose.
    pub(super) fn read_compressed_header(
        &mut self,
        hdr: &mut FrameHeader,
        data: &[u8],
    ) -> Result<(), Error> {
        let mut r = BoolDecoder::new(data)?;
        let fc = &mut self.fc;

        hdr.tx_mode = if hdr.lossless {
            ONLY_4X4
        } else {
            let mode = r.read_literal(2) as u8;
            if mode == ALLOW_32X32 {
                mode + u8::from(r.read_bit())
            } else {
                mode
            }
        };
        if hdr.tx_mode == TX_MODE_SELECT {
            for i in 0..2 {
                diff_update(&mut r, &mut fc.tx8[i][0]);
            }
            for i in 0..2 {
                for j in 0..2 {
                    diff_update(&mut r, &mut fc.tx16[i][j]);
                }
            }
            for i in 0..2 {
                for j in 0..3 {
                    diff_update(&mut r, &mut fc.tx32[i][j]);
                }
            }
        }

        for tx in 0..=usize::from(TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(hdr.tx_mode)]) {
            if r.read_bit() {
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..6 {
                            for l in 0..(if k == 0 { 3 } else { 6 }) {
                                for m in 0..3 {
                                    diff_update(&mut r, &mut fc.coef[tx][i][j][k][l][m]);
                                }
                            }
                        }
                    }
                }
            }
        }

        for prob in &mut fc.skip {
            diff_update(&mut r, prob);
        }

        if !hdr.is_intra() {
            for row in &mut fc.inter_mode {
                for prob in row {
                    diff_update(&mut r, prob);
                }
            }
            if hdr.interp_filter == SWITCHABLE {
                for row in &mut fc.switchable_interp {
                    for prob in row {
                        diff_update(&mut r, prob);
                    }
                }
            }
            for prob in &mut fc.intra_inter {
                diff_update(&mut r, prob);
            }

            let compound_allowed = (1..3).any(|i| hdr.sign_bias[i + 1] != hdr.sign_bias[1]);
            hdr.reference_mode = if compound_allowed && r.read_bit() {
                if r.read_bit() {
                    REFERENCE_MODE_SELECT
                } else {
                    COMPOUND_REFERENCE
                }
            } else {
                SINGLE_REFERENCE
            };
            if hdr.reference_mode != SINGLE_REFERENCE {
                setup_compound_reference_mode(hdr);
            }
            if hdr.reference_mode == REFERENCE_MODE_SELECT {
                for prob in &mut fc.comp_inter {
                    diff_update(&mut r, prob);
                }
            }
            if hdr.reference_mode != COMPOUND_REFERENCE {
                for row in &mut fc.single_ref {
                    diff_update(&mut r, &mut row[0]);
                    diff_update(&mut r, &mut row[1]);
                }
            }
            if hdr.reference_mode != SINGLE_REFERENCE {
                for prob in &mut fc.comp_ref {
                    diff_update(&mut r, prob);
                }
            }
            for row in &mut fc.y_mode {
                for prob in row {
                    diff_update(&mut r, prob);
                }
            }
            for row in &mut fc.partition {
                for prob in row {
                    diff_update(&mut r, prob);
                }
            }
            read_mv_probs(&mut r, fc, hdr.allow_hp);
        }
        if r.overran() {
            return Err(Error::Corrupt("the compressed header ran out of data"));
        }
        Ok(())
    }
}

pub fn setup_compound_reference_mode(hdr: &mut FrameHeader) {
    let bias = &hdr.sign_bias;
    if bias[1] == bias[2] {
        hdr.comp_fixed_ref = ALTREF_FRAME;
        hdr.comp_var_ref = [LAST_FRAME, GOLDEN_FRAME];
    } else if bias[1] == bias[3] {
        hdr.comp_fixed_ref = GOLDEN_FRAME;
        hdr.comp_var_ref = [LAST_FRAME, ALTREF_FRAME];
    } else {
        hdr.comp_fixed_ref = LAST_FRAME;
        hdr.comp_var_ref = [GOLDEN_FRAME, ALTREF_FRAME];
    }
}

fn update_mv_probs(r: &mut BoolDecoder, probs: &mut [u8]) {
    for prob in probs {
        if r.read(252) {
            *prob = ((r.read_literal(7) << 1) | 1) as u8;
        }
    }
}

fn read_mv_probs(r: &mut BoolDecoder, fc: &mut FrameContext, allow_hp: bool) {
    let mv = &mut fc.mv;
    update_mv_probs(r, &mut mv.joints);
    for comp in &mut mv.comps {
        update_mv_probs(r, std::slice::from_mut(&mut comp.sign));
        update_mv_probs(r, &mut comp.classes);
        update_mv_probs(r, &mut comp.class0);
        update_mv_probs(r, &mut comp.bits);
    }
    for comp in &mut mv.comps {
        for row in &mut comp.class0_fp {
            update_mv_probs(r, row);
        }
        update_mv_probs(r, &mut comp.fp);
    }
    if allow_hp {
        for comp in &mut mv.comps {
            update_mv_probs(r, std::slice::from_mut(&mut comp.class0_hp));
            update_mv_probs(r, std::slice::from_mut(&mut comp.hp));
        }
    }
}

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

fn decode_term_subexp(r: &mut BoolDecoder) -> i32 {
    if !r.read_bit() {
        return r.read_literal(4) as i32;
    }
    if !r.read_bit() {
        return r.read_literal(4) as i32 + 16;
    }
    if !r.read_bit() {
        return r.read_literal(5) as i32 + 32;
    }
    let v = r.read_literal(7) as i32;
    let uniform = if v < 65 {
        v
    } else {
        (v << 1) - 65 + i32::from(r.read_bit())
    };
    uniform + 64
}

/// `vp9_diff_update_prob`.
pub fn diff_update(r: &mut BoolDecoder, prob: &mut u8) {
    if r.read(252) {
        let delta = decode_term_subexp(r) as usize;
        let v = i32::from(INV_MAP_TABLE[delta.min(254)]);
        let m = i32::from(*prob) - 1;
        *prob = if (m << 1) <= 255 {
            1 + inv_recenter_nonneg(v, m)
        } else {
            255 - inv_recenter_nonneg(v, 254 - m)
        } as u8;
    }
}
