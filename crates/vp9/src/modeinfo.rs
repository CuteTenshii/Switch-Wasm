//! One tile's decode state and the mode info parser for its blocks.

use super::bool_decoder::BoolDecoder;
use super::frame::*;
use super::tables::*;
use super::types::*;

const MV_JOINT_ZERO: u8 = 0;
const MV_JOINT_HNZVZ: u8 = 1;
const MV_JOINT_HZVNZ: u8 = 2;
const MV_JOINT_HNZVNZ: u8 = 3;
const MV_BORDER: i32 = 16 << 3;
const MV_LOW: i32 = -(1 << 14);
const MV_UPP: i32 = (1 << 14) - 1;
const COMPANDED_MVREF_THRESH: i32 = 8;
const IDX_N_COLUMN_TO_SUBBLOCK: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

/// The position and extent of the block being decoded.
#[derive(Clone, Copy, Default)]
pub struct Blk {
    pub mi_row: usize,
    pub mi_col: usize,
    pub bsize: usize,
    /// Size in 8x8 units, possibly reaching past the frame.
    pub bw: usize,
    pub bh: usize,
    /// The part of it inside the frame.
    pub x_mis: usize,
    pub y_mis: usize,
    pub mb_to_left_edge: i32,
    pub mb_to_right_edge: i32,
    pub mb_to_top_edge: i32,
    pub mb_to_bottom_edge: i32,
    /// Pool index of the block's mode info.
    pub mi: usize,
    pub above: Option<usize>,
    pub left: Option<usize>,
    /// log2 of the 4x4 sub-blocks a sub-8x8 block is split into, wide and high.
    pub bmode_blocks_wl: usize,
    pub bmode_blocks_hl: usize,
    /// log2 of the block's width and height in 4x4 units, a sub-8x8 block counting as 8x8.
    pub bwl: usize,
    pub bhl: usize,
}

pub struct TileDecoder<'a> {
    pub r: BoolDecoder<'a>,
    pub st: &'a mut FrameState,
    pub mi_col_start: usize,
    pub mi_col_end: usize,
    pub left_ctx: [[u8; 16]; 3],
    pub left_seg_ctx: [u8; 8],
    pub dqcoeff: Box<[i32; 1024]>,
}

fn use_mv_hp(mv: Mv) -> bool {
    (i32::from(mv.row).abs() >> 3) < COMPANDED_MVREF_THRESH
        && (i32::from(mv.col).abs() >> 3) < COMPANDED_MVREF_THRESH
}

fn lower_mv_precision(mv: &mut Mv, allow_hp: bool) {
    if !(allow_hp && use_mv_hp(*mv)) {
        if mv.row & 1 != 0 {
            mv.row += if mv.row > 0 { -1 } else { 1 };
        }
        if mv.col & 1 != 0 {
            mv.col += if mv.col > 0 { -1 } else { 1 };
        }
    }
}

fn mv_class_base(class: usize) -> i32 {
    if class != 0 {
        2 << (class + 2)
    } else {
        0
    }
}

fn mv_class_and_offset(z: i32) -> (usize, i32) {
    let class = if z >= 2 * 4096 {
        10
    } else {
        let n = (z >> 3) as u32;
        if n == 0 {
            0
        } else {
            (31 - n.leading_zeros()) as usize
        }
    };
    (class, z - mv_class_base(class))
}

impl<'a> TileDecoder<'a> {
    fn mi(&self, index: usize) -> &ModeInfo {
        &self.st.mi_pool[index]
    }

    pub fn counting(&self) -> bool {
        self.st.counts.is_some()
    }

    fn count<F: FnOnce(&mut super::probs::Counts)>(&mut self, f: F) {
        if let Some(counts) = self.st.counts.as_mut() {
            f(counts);
        }
    }

    fn read_segment_id(&mut self) -> u8 {
        self.r.read_tree(&SEGMENT_TREE, &self.st.seg.tree_probs)
    }

    fn segment_region_min(&self, map: &[u8], b: &Blk) -> u8 {
        let mut id = u8::MAX;
        for y in 0..b.y_mis {
            for x in 0..b.x_mis {
                id = id.min(map[(b.mi_row + y) * self.st.mi_cols + b.mi_col + x]);
            }
        }
        id
    }

    fn set_segment_id(&mut self, b: &Blk, id: u8) {
        for y in 0..b.y_mis {
            for x in 0..b.x_mis {
                self.st.cur_seg_map[(b.mi_row + y) * self.st.mi_cols + b.mi_col + x] = id;
            }
        }
    }

    fn copy_segment_id(&mut self, b: &Blk) {
        for y in 0..b.y_mis {
            for x in 0..b.x_mis {
                let at = (b.mi_row + y) * self.st.mi_cols + b.mi_col + x;
                self.st.cur_seg_map[at] = self.st.last_seg_map[at];
            }
        }
    }

    fn read_intra_segment_id(&mut self, b: &Blk) -> u8 {
        if !self.st.seg.enabled {
            return 0;
        }
        if !self.st.seg.update_map {
            self.copy_segment_id(b);
            return 0;
        }
        let id = self.read_segment_id();
        self.set_segment_id(b, id);
        id
    }

    fn read_inter_segment_id(&mut self, b: &Blk) -> u8 {
        if !self.st.seg.enabled {
            return 0;
        }
        let predicted = self.segment_region_min(&self.st.last_seg_map, b);
        if !self.st.seg.update_map {
            self.copy_segment_id(b);
            return predicted;
        }
        let id = if self.st.seg.temporal_update {
            let ctx = b
                .above
                .map_or(0, |i| usize::from(self.mi(i).seg_id_predicted))
                + b.left
                    .map_or(0, |i| usize::from(self.mi(i).seg_id_predicted));
            let prob = self.st.seg.pred_probs[ctx];
            let flag = self.r.read(prob);
            self.st.mi_pool[b.mi].seg_id_predicted = flag;
            if flag {
                predicted
            } else {
                self.read_segment_id()
            }
        } else {
            self.read_segment_id()
        };
        self.set_segment_id(b, id);
        id
    }

    fn read_skip(&mut self, b: &Blk, segment_id: u8) -> bool {
        if self.st.seg.active(usize::from(segment_id), SEG_LVL_SKIP) {
            return true;
        }
        let above = b.above.map_or(0, |i| usize::from(self.mi(i).skip));
        let left = b.left.map_or(0, |i| usize::from(self.mi(i).skip));
        let ctx = above + left;
        let skip = self.r.read(self.st.fc.skip[ctx]);
        self.count(|c| c.skip[ctx][usize::from(skip)] += 1);
        skip
    }

    fn tx_size_context(&self, b: &Blk) -> usize {
        let max_tx = MAX_TX_SIZE[b.bsize];
        let above = b.above.map(|i| self.mi(i));
        let left = b.left.map(|i| self.mi(i));
        let mut above_ctx = match above {
            Some(mi) if !mi.skip => mi.tx_size,
            _ => max_tx,
        };
        let mut left_ctx = match left {
            Some(mi) if !mi.skip => mi.tx_size,
            _ => max_tx,
        };
        if left.is_none() {
            left_ctx = above_ctx;
        }
        if above.is_none() {
            above_ctx = left_ctx;
        }
        usize::from(above_ctx + left_ctx > max_tx)
    }

    fn read_tx_size(&mut self, b: &Blk, allow_select: bool) -> u8 {
        let max_tx = MAX_TX_SIZE[b.bsize];
        let tx_mode = self.st.hdr.tx_mode;
        if !(allow_select && tx_mode == TX_MODE_SELECT && b.bsize >= BLOCK_8X8) {
            return max_tx.min(TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(tx_mode)]);
        }
        let ctx = self.tx_size_context(b);
        let fc = &self.st.fc;
        let probs: &[u8] = match max_tx {
            TX_8X8 => &fc.tx8[ctx],
            TX_16X16 => &fc.tx16[ctx],
            _ => &fc.tx32[ctx],
        };
        let probs: Vec<u8> = probs.to_vec();
        let mut tx = u8::from(self.r.read(probs[0]));
        if tx != TX_4X4 && max_tx >= TX_16X16 {
            tx += u8::from(self.r.read(probs[1]));
            if tx != TX_8X8 && max_tx >= TX_32X32 {
                tx += u8::from(self.r.read(probs[2]));
            }
        }
        self.count(|c| match max_tx {
            TX_8X8 => c.tx8[ctx][usize::from(tx)] += 1,
            TX_16X16 => c.tx16[ctx][usize::from(tx)] += 1,
            _ => c.tx32[ctx][usize::from(tx)] += 1,
        });
        tx
    }

    fn above_block_mode(&self, cur: &ModeInfo, above: Option<usize>, block: usize) -> u8 {
        if block < 2 {
            match above.map(|i| self.mi(i)) {
                Some(mi) if !mi.is_inter() => mi.y_mode(block + 2),
                _ => DC_PRED,
            }
        } else {
            cur.bmi_mode[block - 2]
        }
    }

    fn left_block_mode(&self, cur: &ModeInfo, left: Option<usize>, block: usize) -> u8 {
        if block == 0 || block == 2 {
            match left.map(|i| self.mi(i)) {
                Some(mi) if !mi.is_inter() => mi.y_mode(block + 1),
                _ => DC_PRED,
            }
        } else {
            cur.bmi_mode[block - 1]
        }
    }

    fn read_kf_y_mode(&mut self, b: &Blk, block: usize) -> u8 {
        let cur = *self.mi(b.mi);
        let above = self.above_block_mode(&cur, b.above, block);
        let left = self.left_block_mode(&cur, b.left, block);
        let probs = KF_Y_MODE_PROBS[usize::from(above)][usize::from(left)];
        self.r.read_tree(&INTRA_MODE_TREE, &probs)
    }

    pub fn read_mode_info(&mut self, b: &Blk) {
        if self.st.hdr.is_intra() {
            self.read_intra_frame_mode_info(b);
        } else {
            self.read_inter_frame_mode_info(b);
            let mi = *self.mi(b.mi);
            for y in 0..b.y_mis {
                for x in 0..b.x_mis {
                    self.st.cur.mvs[(b.mi_row + y) * self.st.mi_cols + b.mi_col + x] = MvRef {
                        ref_frame: mi.ref_frame,
                        mv: mi.mv,
                    };
                }
            }
        }
    }

    fn read_intra_frame_mode_info(&mut self, b: &Blk) {
        let segment_id = self.read_intra_segment_id(b);
        let skip = self.read_skip(b, segment_id);
        let tx_size = self.read_tx_size(b, true);
        {
            let mi = &mut self.st.mi_pool[b.mi];
            mi.segment_id = segment_id;
            mi.skip = skip;
            mi.tx_size = tx_size;
            mi.ref_frame = [INTRA_FRAME, NO_REF];
        }
        match b.bsize {
            BLOCK_4X4 => {
                for i in 0..4 {
                    let mode = self.read_kf_y_mode(b, i);
                    self.st.mi_pool[b.mi].bmi_mode[i] = mode;
                }
                self.st.mi_pool[b.mi].mode = self.st.mi_pool[b.mi].bmi_mode[3];
            }
            BLOCK_4X8 => {
                let m0 = self.read_kf_y_mode(b, 0);
                {
                    let mi = &mut self.st.mi_pool[b.mi];
                    mi.bmi_mode[0] = m0;
                    mi.bmi_mode[2] = m0;
                }
                let m1 = self.read_kf_y_mode(b, 1);
                let mi = &mut self.st.mi_pool[b.mi];
                mi.bmi_mode[1] = m1;
                mi.bmi_mode[3] = m1;
                mi.mode = m1;
            }
            BLOCK_8X4 => {
                let m0 = self.read_kf_y_mode(b, 0);
                {
                    let mi = &mut self.st.mi_pool[b.mi];
                    mi.bmi_mode[0] = m0;
                    mi.bmi_mode[1] = m0;
                }
                let m2 = self.read_kf_y_mode(b, 2);
                let mi = &mut self.st.mi_pool[b.mi];
                mi.bmi_mode[2] = m2;
                mi.bmi_mode[3] = m2;
                mi.mode = m2;
            }
            _ => {
                let mode = self.read_kf_y_mode(b, 0);
                self.st.mi_pool[b.mi].mode = mode;
            }
        }
        let mode = self.st.mi_pool[b.mi].mode;
        let uv = self
            .r
            .read_tree(&INTRA_MODE_TREE, &KF_UV_MODE_PROBS[usize::from(mode)]);
        self.st.mi_pool[b.mi].uv_mode = uv;
    }

    fn read_intra_mode_y(&mut self, size_group: usize) -> u8 {
        let probs = self.st.fc.y_mode[size_group];
        let mode = self.r.read_tree(&INTRA_MODE_TREE, &probs);
        self.count(|c| c.y_mode[size_group][usize::from(mode)] += 1);
        mode
    }

    fn read_intra_block_mode_info(&mut self, b: &Blk) {
        match b.bsize {
            BLOCK_4X4 => {
                for i in 0..4 {
                    let mode = self.read_intra_mode_y(0);
                    self.st.mi_pool[b.mi].bmi_mode[i] = mode;
                }
                self.st.mi_pool[b.mi].mode = self.st.mi_pool[b.mi].bmi_mode[3];
            }
            BLOCK_4X8 => {
                let m0 = self.read_intra_mode_y(0);
                let m1 = self.read_intra_mode_y(0);
                let mi = &mut self.st.mi_pool[b.mi];
                mi.bmi_mode = [m0, m1, m0, m1];
                mi.mode = m1;
            }
            BLOCK_8X4 => {
                let m0 = self.read_intra_mode_y(0);
                let m2 = self.read_intra_mode_y(0);
                let mi = &mut self.st.mi_pool[b.mi];
                mi.bmi_mode = [m0, m0, m2, m2];
                mi.mode = m2;
            }
            _ => {
                let mode = self.read_intra_mode_y(usize::from(SIZE_GROUP[b.bsize]));
                self.st.mi_pool[b.mi].mode = mode;
            }
        }
        let y_mode = usize::from(self.st.mi_pool[b.mi].mode);
        let probs = self.st.fc.uv_mode[y_mode];
        let uv = self.r.read_tree(&INTRA_MODE_TREE, &probs);
        self.count(|c| c.uv_mode[y_mode][usize::from(uv)] += 1);
        let mi = &mut self.st.mi_pool[b.mi];
        mi.uv_mode = uv;
        mi.interp_filter = 3;
        mi.ref_frame = [INTRA_FRAME, NO_REF];
    }

    fn read_mv_component(&mut self, comp: usize, use_hp: bool) -> i32 {
        let probs = self.st.fc.mv.comps[comp];
        let sign = self.r.read(probs.sign);
        let class = usize::from(self.r.read_tree(&MV_CLASS_TREE, &probs.classes));
        let class0 = class == 0;
        let (d, mut mag);
        if class0 {
            d = i32::from(self.r.read(probs.class0[0]));
            mag = 0;
        } else {
            let mut bits = 0;
            for i in 0..class {
                bits |= i32::from(self.r.read(probs.bits[i])) << i;
            }
            d = bits;
            mag = 2 << (class + 2);
        }
        let fp_probs = if class0 {
            probs.class0_fp[d as usize]
        } else {
            probs.fp
        };
        let fr = i32::from(self.r.read_tree(&MV_FP_TREE, &fp_probs));
        let hp = if use_hp {
            i32::from(self.r.read(if class0 { probs.class0_hp } else { probs.hp }))
        } else {
            1
        };
        mag += ((d << 3) | (fr << 1) | hp) + 1;
        if sign {
            -mag
        } else {
            mag
        }
    }

    fn inc_mv_component(&mut self, comp: usize, v: i32) {
        if self.st.counts.is_none() {
            return;
        }
        let s = usize::from(v < 0);
        let z = (if v < 0 { -v } else { v }) - 1;
        let (class, offset) = mv_class_and_offset(z);
        let d = (offset >> 3) as usize;
        let f = ((offset >> 1) & 3) as usize;
        let e = (offset & 1) as usize;
        let c = &mut self.st.counts.as_mut().unwrap().mv_comps[comp];
        c.sign[s] += 1;
        c.classes[class] += 1;
        if class == 0 {
            c.class0[d] += 1;
            c.class0_fp[d][f] += 1;
            c.class0_hp[e] += 1;
        } else {
            for i in 0..class {
                c.bits[i][(d >> i) & 1] += 1;
            }
            c.fp[f] += 1;
            c.hp[e] += 1;
        }
    }

    fn read_mv(&mut self, reference: Mv, allow_hp: bool) -> Mv {
        let joints = self.st.fc.mv.joints;
        let joint = self.r.read_tree(&MV_JOINT_TREE, &joints);
        let use_hp = allow_hp && use_mv_hp(reference);
        let mut diff = (0i32, 0i32);
        if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
            diff.0 = self.read_mv_component(0, use_hp);
        }
        if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
            diff.1 = self.read_mv_component(1, use_hp);
        }
        if let Some(counts) = self.st.counts.as_mut() {
            counts.mv_joints[usize::from(joint)] += 1;
        }
        if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
            self.inc_mv_component(0, diff.0);
        }
        if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
            self.inc_mv_component(1, diff.1);
        }
        let _ = MV_JOINT_ZERO;
        Mv {
            row: reference.row.wrapping_add(diff.0 as i16),
            col: reference.col.wrapping_add(diff.1 as i16),
        }
    }

    fn comp_inter_context(&self, b: &Blk) -> usize {
        let fixed = self.st.hdr.comp_fixed_ref;
        let above = b.above.map(|i| *self.mi(i));
        let left = b.left.map(|i| *self.mi(i));
        match (above, left) {
            (Some(a), Some(l)) => {
                if !a.has_second_ref() && !l.has_second_ref() {
                    usize::from((a.ref_frame[0] == fixed) ^ (l.ref_frame[0] == fixed))
                } else if !a.has_second_ref() {
                    2 + usize::from(a.ref_frame[0] == fixed || !a.is_inter())
                } else if !l.has_second_ref() {
                    2 + usize::from(l.ref_frame[0] == fixed || !l.is_inter())
                } else {
                    4
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.has_second_ref() {
                    usize::from(e.ref_frame[0] == fixed)
                } else {
                    3
                }
            }
            (None, None) => 1,
        }
    }

    fn comp_ref_context(&self, b: &Blk) -> usize {
        let hdr = &self.st.hdr;
        let fix_ref_idx = usize::from(hdr.sign_bias[hdr.comp_fixed_ref as usize]);
        let var_ref_idx = 1 - fix_ref_idx;
        let (fixed, var0, var1) = (hdr.comp_fixed_ref, hdr.comp_var_ref[0], hdr.comp_var_ref[1]);
        let above = b.above.map(|i| *self.mi(i));
        let left = b.left.map(|i| *self.mi(i));
        match (above, left) {
            (Some(a), Some(l)) => {
                let (above_intra, left_intra) = (!a.is_inter(), !l.is_inter());
                if above_intra && left_intra {
                    2
                } else if above_intra || left_intra {
                    let edge = if above_intra { l } else { a };
                    if !edge.has_second_ref() {
                        1 + 2 * usize::from(edge.ref_frame[0] != var1)
                    } else {
                        1 + 2 * usize::from(edge.ref_frame[var_ref_idx] != var1)
                    }
                } else {
                    let (l_sg, a_sg) = (!l.has_second_ref(), !a.has_second_ref());
                    let vrfa = if a_sg {
                        a.ref_frame[0]
                    } else {
                        a.ref_frame[var_ref_idx]
                    };
                    let vrfl = if l_sg {
                        l.ref_frame[0]
                    } else {
                        l.ref_frame[var_ref_idx]
                    };
                    if vrfa == vrfl && var1 == vrfa {
                        0
                    } else if l_sg && a_sg {
                        if (vrfa == fixed && vrfl == var0) || (vrfl == fixed && vrfa == var0) {
                            4
                        } else if vrfa == vrfl {
                            3
                        } else {
                            1
                        }
                    } else if l_sg || a_sg {
                        let vrfc = if l_sg { vrfa } else { vrfl };
                        let rfs = if a_sg { vrfa } else { vrfl };
                        if vrfc == var1 && rfs != var1 {
                            1
                        } else if rfs == var1 && vrfc != var1 {
                            2
                        } else {
                            4
                        }
                    } else if vrfa == vrfl {
                        4
                    } else {
                        2
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter() {
                    2
                } else if e.has_second_ref() {
                    4 * usize::from(e.ref_frame[var_ref_idx] != var1)
                } else {
                    3 * usize::from(e.ref_frame[0] != var1)
                }
            }
            (None, None) => 2,
        }
    }

    fn single_ref_p1_context(&self, b: &Blk) -> usize {
        let above = b.above.map(|i| *self.mi(i));
        let left = b.left.map(|i| *self.mi(i));
        match (above, left) {
            (Some(a), Some(l)) => {
                let (above_intra, left_intra) = (!a.is_inter(), !l.is_inter());
                if above_intra && left_intra {
                    2
                } else if above_intra || left_intra {
                    let edge = if above_intra { l } else { a };
                    if !edge.has_second_ref() {
                        4 * usize::from(edge.ref_frame[0] == LAST_FRAME)
                    } else {
                        1 + usize::from(
                            edge.ref_frame[0] == LAST_FRAME || edge.ref_frame[1] == LAST_FRAME,
                        )
                    }
                } else {
                    let (a_second, l_second) = (a.has_second_ref(), l.has_second_ref());
                    let (a0, a1, l0, l1) = (
                        a.ref_frame[0],
                        a.ref_frame[1],
                        l.ref_frame[0],
                        l.ref_frame[1],
                    );
                    if a_second && l_second {
                        1 + usize::from(
                            a0 == LAST_FRAME
                                || a1 == LAST_FRAME
                                || l0 == LAST_FRAME
                                || l1 == LAST_FRAME,
                        )
                    } else if a_second || l_second {
                        let rfs = if !a_second { a0 } else { l0 };
                        let crf1 = if a_second { a0 } else { l0 };
                        let crf2 = if a_second { a1 } else { l1 };
                        if rfs == LAST_FRAME {
                            3 + usize::from(crf1 == LAST_FRAME || crf2 == LAST_FRAME)
                        } else {
                            usize::from(crf1 == LAST_FRAME || crf2 == LAST_FRAME)
                        }
                    } else {
                        2 * usize::from(a0 == LAST_FRAME) + 2 * usize::from(l0 == LAST_FRAME)
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter() {
                    2
                } else if !e.has_second_ref() {
                    4 * usize::from(e.ref_frame[0] == LAST_FRAME)
                } else {
                    1 + usize::from(e.ref_frame[0] == LAST_FRAME || e.ref_frame[1] == LAST_FRAME)
                }
            }
            (None, None) => 2,
        }
    }

    fn single_ref_p2_context(&self, b: &Blk) -> usize {
        let above = b.above.map(|i| *self.mi(i));
        let left = b.left.map(|i| *self.mi(i));
        match (above, left) {
            (Some(a), Some(l)) => {
                let (above_intra, left_intra) = (!a.is_inter(), !l.is_inter());
                if above_intra && left_intra {
                    2
                } else if above_intra || left_intra {
                    let edge = if above_intra { l } else { a };
                    if !edge.has_second_ref() {
                        if edge.ref_frame[0] == LAST_FRAME {
                            3
                        } else {
                            4 * usize::from(edge.ref_frame[0] == GOLDEN_FRAME)
                        }
                    } else {
                        1 + 2 * usize::from(
                            edge.ref_frame[0] == GOLDEN_FRAME || edge.ref_frame[1] == GOLDEN_FRAME,
                        )
                    }
                } else {
                    let (a_second, l_second) = (a.has_second_ref(), l.has_second_ref());
                    let (a0, a1, l0, l1) = (
                        a.ref_frame[0],
                        a.ref_frame[1],
                        l.ref_frame[0],
                        l.ref_frame[1],
                    );
                    if a_second && l_second {
                        if a0 == l0 && a1 == l1 {
                            3 * usize::from(
                                a0 == GOLDEN_FRAME
                                    || a1 == GOLDEN_FRAME
                                    || l0 == GOLDEN_FRAME
                                    || l1 == GOLDEN_FRAME,
                            )
                        } else {
                            2
                        }
                    } else if a_second || l_second {
                        let rfs = if !a_second { a0 } else { l0 };
                        let crf1 = if a_second { a0 } else { l0 };
                        let crf2 = if a_second { a1 } else { l1 };
                        if rfs == GOLDEN_FRAME {
                            3 + usize::from(crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME)
                        } else if rfs == ALTREF_FRAME {
                            usize::from(crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME)
                        } else {
                            1 + 2 * usize::from(crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME)
                        }
                    } else if a0 == LAST_FRAME && l0 == LAST_FRAME {
                        3
                    } else if a0 == LAST_FRAME || l0 == LAST_FRAME {
                        let edge0 = if a0 == LAST_FRAME { l0 } else { a0 };
                        4 * usize::from(edge0 == GOLDEN_FRAME)
                    } else {
                        2 * usize::from(a0 == GOLDEN_FRAME) + 2 * usize::from(l0 == GOLDEN_FRAME)
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter() || (e.ref_frame[0] == LAST_FRAME && !e.has_second_ref()) {
                    2
                } else if !e.has_second_ref() {
                    4 * usize::from(e.ref_frame[0] == GOLDEN_FRAME)
                } else {
                    3 * usize::from(
                        e.ref_frame[0] == GOLDEN_FRAME || e.ref_frame[1] == GOLDEN_FRAME,
                    )
                }
            }
            (None, None) => 2,
        }
    }

    fn read_ref_frames(&mut self, b: &Blk, segment_id: u8) -> [i8; 2] {
        let seg = usize::from(segment_id);
        if self.st.seg.active(seg, SEG_LVL_REF_FRAME) {
            return [self.st.seg.data(seg, SEG_LVL_REF_FRAME) as i8, NO_REF];
        }
        let mode = if self.st.hdr.reference_mode == REFERENCE_MODE_SELECT {
            let ctx = self.comp_inter_context(b);
            let bit = self.r.read(self.st.fc.comp_inter[ctx]);
            self.count(|c| c.comp_inter[ctx][usize::from(bit)] += 1);
            u8::from(bit)
        } else {
            self.st.hdr.reference_mode
        };
        let mut ref_frame = [INTRA_FRAME, NO_REF];
        if mode == COMPOUND_REFERENCE {
            let idx = usize::from(self.st.hdr.sign_bias[self.st.hdr.comp_fixed_ref as usize]);
            let ctx = self.comp_ref_context(b);
            let bit = self.r.read(self.st.fc.comp_ref[ctx]);
            self.count(|c| c.comp_ref[ctx][usize::from(bit)] += 1);
            ref_frame[idx] = self.st.hdr.comp_fixed_ref;
            ref_frame[1 - idx] = self.st.hdr.comp_var_ref[usize::from(bit)];
        } else {
            let ctx0 = self.single_ref_p1_context(b);
            let bit0 = self.r.read(self.st.fc.single_ref[ctx0][0]);
            self.count(|c| c.single_ref[ctx0][0][usize::from(bit0)] += 1);
            if bit0 {
                let ctx1 = self.single_ref_p2_context(b);
                let bit1 = self.r.read(self.st.fc.single_ref[ctx1][1]);
                self.count(|c| c.single_ref[ctx1][1][usize::from(bit1)] += 1);
                ref_frame[0] = if bit1 { ALTREF_FRAME } else { GOLDEN_FRAME };
            } else {
                ref_frame[0] = LAST_FRAME;
            }
        }
        ref_frame
    }

    fn switchable_interp_context(&self, b: &Blk) -> usize {
        let left = b.left.map_or(3, |i| self.mi(i).interp_filter);
        let above = b.above.map_or(3, |i| self.mi(i).interp_filter);
        usize::from(if left == above {
            left
        } else if left == 3 {
            above
        } else if above == 3 {
            left
        } else {
            3
        })
    }

    fn read_inter_mode(&mut self, ctx: usize) -> u8 {
        let probs = self.st.fc.inter_mode[ctx];
        let mode = self.r.read_tree(&INTER_MODE_TREE, &probs);
        self.count(|c| c.inter_mode[ctx][usize::from(mode)] += 1);
        NEARESTMV + mode
    }

    fn is_inside(&self, b: &Blk, pos: (i32, i32)) -> bool {
        let row = b.mi_row as i32 + pos.0;
        let col = b.mi_col as i32 + pos.1;
        !(row < 0
            || col < self.mi_col_start as i32
            || row >= self.st.mi_rows as i32
            || col >= self.mi_col_end as i32)
    }

    fn candidate(&self, b: &Blk, pos: (i32, i32)) -> ModeInfo {
        let row = (b.mi_row as i32 + pos.0) as usize;
        let col = (b.mi_col as i32 + pos.1) as usize;
        *self.mi(self.st.mi_index(row, col))
    }

    fn clamp_mv_ref(&self, b: &Blk, mv: &mut Mv) {
        mv.col = i32::from(mv.col).clamp(
            b.mb_to_left_edge - MV_BORDER,
            b.mb_to_right_edge + MV_BORDER,
        ) as i16;
        mv.row = i32::from(mv.row).clamp(
            b.mb_to_top_edge - MV_BORDER,
            b.mb_to_bottom_edge + MV_BORDER,
        ) as i16;
    }

    fn mode_context(&self, b: &Blk) -> usize {
        let mut counter = 0;
        for i in 0..2 {
            let pos = self.mv_ref_pos(b, i);
            if self.is_inside(b, pos) {
                counter += MODE_2_COUNTER[usize::from(self.candidate(b, pos).mode)];
            }
        }
        COUNTER_TO_CONTEXT[counter as usize] as usize
    }

    fn mv_ref_pos(&self, b: &Blk, i: usize) -> (i32, i32) {
        let p = MV_REF_BLOCKS[b.bsize][i];
        (p[0], p[1])
    }

    /// The motion vector candidates for `mode` against `ref_frame`; returns how many it settled on.
    fn find_mv_refs(
        &self,
        b: &Blk,
        mode: u8,
        ref_frame: i8,
        list: &mut [Mv; 2],
        block: i32,
    ) -> usize {
        let sign_bias = self.st.hdr.sign_bias;
        let mut count = 0usize;
        let mut different_ref_found = false;
        let prev = if self.st.use_prev_frame_mvs {
            self.st
                .prev_mvs
                .as_ref()
                .map(|f| f.mvs[b.mi_row * self.st.mi_cols + b.mi_col])
        } else {
            None
        };
        let early_break = mode != NEARMV;
        *list = [Mv::default(); 2];

        // Returns true once the list is settled.
        let add = |mv: Mv, list: &mut [Mv; 2], count: &mut usize| -> bool {
            if *count != 0 {
                if mv != list[0] {
                    list[*count] = mv;
                    *count += 1;
                    return true;
                }
            } else {
                list[*count] = mv;
                *count += 1;
                if early_break {
                    return true;
                }
            }
            false
        };

        'search: {
            let mut i = 0;
            if block >= 0 {
                while i < 2 {
                    let pos = self.mv_ref_pos(b, i);
                    if self.is_inside(b, pos) {
                        let cand = self.candidate(b, pos);
                        different_ref_found = true;
                        let sub = |which: usize| -> Mv {
                            if usize::from(cand.sb_type) < BLOCK_8X8 {
                                cand.bmi_mv[IDX_N_COLUMN_TO_SUBBLOCK[block as usize]
                                    [usize::from(pos.1 == 0)]][which]
                            } else {
                                cand.mv[which]
                            }
                        };
                        if cand.ref_frame[0] == ref_frame {
                            if add(sub(0), list, &mut count) {
                                break 'search;
                            }
                        } else if cand.ref_frame[1] == ref_frame && add(sub(1), list, &mut count) {
                            break 'search;
                        }
                    }
                    i += 1;
                }
            }
            while i < 8 {
                let pos = self.mv_ref_pos(b, i);
                if self.is_inside(b, pos) {
                    let cand = self.candidate(b, pos);
                    different_ref_found = true;
                    if cand.ref_frame[0] == ref_frame {
                        if add(cand.mv[0], list, &mut count) {
                            break 'search;
                        }
                    } else if cand.ref_frame[1] == ref_frame && add(cand.mv[1], list, &mut count) {
                        break 'search;
                    }
                }
                i += 1;
            }
            if let Some(p) = prev {
                if p.ref_frame[0] == ref_frame {
                    if add(p.mv[0], list, &mut count) {
                        break 'search;
                    }
                } else if p.ref_frame[1] == ref_frame && add(p.mv[1], list, &mut count) {
                    break 'search;
                }
            }
            if different_ref_found {
                for i in 0..8 {
                    let pos = self.mv_ref_pos(b, i);
                    if !self.is_inside(b, pos) {
                        continue;
                    }
                    let cand = self.candidate(b, pos);
                    if cand.is_inter() {
                        let scale = |which: usize| -> Mv {
                            let mut mv = cand.mv[which];
                            if sign_bias[cand.ref_frame[which] as usize]
                                != sign_bias[ref_frame as usize]
                            {
                                mv.row = mv.row.wrapping_neg();
                                mv.col = mv.col.wrapping_neg();
                            }
                            mv
                        };
                        if cand.ref_frame[0] != ref_frame && add(scale(0), list, &mut count) {
                            break 'search;
                        }
                        if cand.has_second_ref()
                            && cand.ref_frame[1] != ref_frame
                            && cand.mv[1] != cand.mv[0]
                            && add(scale(1), list, &mut count)
                        {
                            break 'search;
                        }
                    }
                }
            }
            if let Some(p) = prev {
                if p.ref_frame[0] != ref_frame && p.ref_frame[0] > INTRA_FRAME {
                    let mut mv = p.mv[0];
                    if sign_bias[p.ref_frame[0] as usize] != sign_bias[ref_frame as usize] {
                        mv.row = mv.row.wrapping_neg();
                        mv.col = mv.col.wrapping_neg();
                    }
                    if add(mv, list, &mut count) {
                        break 'search;
                    }
                }
                if p.ref_frame[1] > INTRA_FRAME && p.ref_frame[1] != ref_frame && p.mv[1] != p.mv[0]
                {
                    let mut mv = p.mv[1];
                    if sign_bias[p.ref_frame[1] as usize] != sign_bias[ref_frame as usize] {
                        mv.row = mv.row.wrapping_neg();
                        mv.col = mv.col.wrapping_neg();
                    }
                    if add(mv, list, &mut count) {
                        break 'search;
                    }
                }
            }
            count = if mode == NEARMV { 2 } else { 1 };
        }
        for mv in list.iter_mut().take(count) {
            self.clamp_mv_ref(b, mv);
        }
        count
    }

    fn append_sub8x8_mvs(&self, b: &Blk, b_mode: u8, block: usize, which: usize) -> Mv {
        let mi = *self.mi(b.mi);
        let bmi = |i: usize| mi.bmi_mv[i][which];
        let mut list = [Mv::default(); 2];
        match block {
            0 => {
                let count =
                    self.find_mv_refs(b, b_mode, mi.ref_frame[which], &mut list, block as i32);
                list[count - 1]
            }
            1 | 2 => {
                if b_mode == NEARESTMV {
                    bmi(0)
                } else {
                    self.find_mv_refs(b, b_mode, mi.ref_frame[which], &mut list, block as i32);
                    list.iter()
                        .copied()
                        .find(|&mv| bmi(0) != mv)
                        .unwrap_or_default()
                }
            }
            _ => {
                if b_mode == NEARESTMV {
                    bmi(2)
                } else {
                    if bmi(2) != bmi(1) {
                        return bmi(1);
                    }
                    if bmi(2) != bmi(0) {
                        return bmi(0);
                    }
                    self.find_mv_refs(b, b_mode, mi.ref_frame[which], &mut list, block as i32);
                    list.iter()
                        .copied()
                        .find(|&mv| bmi(2) != mv)
                        .unwrap_or_default()
                }
            }
        }
    }

    /// Reads or derives the vectors of `mode`; false when a new vector falls out of range.
    fn assign_mv(
        &mut self,
        mode: u8,
        ref_mv: [Mv; 2],
        near: [Mv; 2],
        compound: bool,
        allow_hp: bool,
    ) -> Option<[Mv; 2]> {
        let mut mv = [Mv::default(); 2];
        match mode {
            NEWMV => {
                for i in 0..1 + usize::from(compound) {
                    mv[i] = self.read_mv(ref_mv[i], allow_hp);
                    let (row, col) = (i32::from(mv[i].row), i32::from(mv[i].col));
                    if !(row > MV_LOW && row < MV_UPP && col > MV_LOW && col < MV_UPP) {
                        return None;
                    }
                }
            }
            NEARMV | NEARESTMV => mv = near,
            ZEROMV => {}
            _ => return None,
        }
        Some(mv)
    }

    fn read_inter_block_mode_info(&mut self, b: &Blk) -> bool {
        let allow_hp = self.st.hdr.allow_hp;
        let segment_id = self.mi(b.mi).segment_id;
        let ref_frame = self.read_ref_frames(b, segment_id);
        self.st.mi_pool[b.mi].ref_frame = ref_frame;
        let compound = ref_frame[1] > INTRA_FRAME;
        let inter_mode_ctx = self.mode_context(b);
        let mut ok = true;

        if self.st.seg.active(usize::from(segment_id), SEG_LVL_SKIP) {
            self.st.mi_pool[b.mi].mode = ZEROMV;
            if b.bsize < BLOCK_8X8 {
                return false;
            }
        } else if b.bsize >= BLOCK_8X8 {
            let mode = self.read_inter_mode(inter_mode_ctx);
            self.st.mi_pool[b.mi].mode = mode;
        }

        let interp = if self.st.hdr.interp_filter == SWITCHABLE {
            let ctx = self.switchable_interp_context(b);
            let probs = self.st.fc.switchable_interp[ctx];
            let t = self.r.read_tree(&SWITCHABLE_INTERP_TREE, &probs);
            self.count(|c| c.switchable_interp[ctx][usize::from(t)] += 1);
            t
        } else {
            self.st.hdr.interp_filter
        };
        self.st.mi_pool[b.mi].interp_filter = interp;

        let mut best_ref_mvs = [Mv::default(); 2];
        if b.bsize < BLOCK_8X8 {
            let num_w = 1usize << b.bmode_blocks_wl;
            let num_h = 1usize << b.bmode_blocks_hl;
            let mut got_new = false;
            let mut best_sub8x8 = [Mv::default(); 2];
            let mut b_mode = 0;
            let mut idy = 0;
            while idy < 2 {
                let mut idx = 0;
                while idx < 2 {
                    let j = idy * 2 + idx;
                    b_mode = self.read_inter_mode(inter_mode_ctx);
                    if b_mode == NEARESTMV || b_mode == NEARMV {
                        for r in 0..1 + usize::from(compound) {
                            best_sub8x8[r] = self.append_sub8x8_mvs(b, b_mode, j, r);
                        }
                    } else if b_mode == NEWMV && !got_new {
                        for r in 0..1 + usize::from(compound) {
                            let mut tmp = [Mv::default(); 2];
                            self.find_mv_refs(b, NEWMV, ref_frame[r], &mut tmp, -1);
                            lower_mv_precision(&mut tmp[0], allow_hp);
                            best_ref_mvs[r] = tmp[0];
                            got_new = true;
                        }
                    }
                    match self.assign_mv(b_mode, best_ref_mvs, best_sub8x8, compound, allow_hp) {
                        Some(mv) => {
                            let mi = &mut self.st.mi_pool[b.mi];
                            mi.bmi_mv[j] = mv;
                            if num_h == 2 {
                                mi.bmi_mv[j + 2] = mv;
                            }
                            if num_w == 2 {
                                mi.bmi_mv[j + 1] = mv;
                            }
                        }
                        None => return false,
                    }
                    idx += num_w;
                }
                idy += num_h;
            }
            let mi = &mut self.st.mi_pool[b.mi];
            mi.mode = b_mode;
            mi.mv = mi.bmi_mv[3];
        } else {
            let mode = self.mi(b.mi).mode;
            if mode != ZEROMV {
                for r in 0..1 + usize::from(compound) {
                    let mut tmp = [Mv::default(); 2];
                    let count = self.find_mv_refs(b, mode, ref_frame[r], &mut tmp, -1);
                    lower_mv_precision(&mut tmp[count - 1], allow_hp);
                    best_ref_mvs[r] = tmp[count - 1];
                }
            }
            match self.assign_mv(mode, best_ref_mvs, best_ref_mvs, compound, allow_hp) {
                Some(mv) => self.st.mi_pool[b.mi].mv = mv,
                None => ok = false,
            }
        }
        ok
    }

    fn read_inter_frame_mode_info(&mut self, b: &Blk) {
        let segment_id = self.read_inter_segment_id(b);
        self.st.mi_pool[b.mi].segment_id = segment_id;
        let skip = self.read_skip(b, segment_id);
        self.st.mi_pool[b.mi].skip = skip;
        let inter_block = if self
            .st
            .seg
            .active(usize::from(segment_id), SEG_LVL_REF_FRAME)
        {
            self.st.seg.data(usize::from(segment_id), SEG_LVL_REF_FRAME) != i32::from(INTRA_FRAME)
        } else {
            let above = b.above.map(|i| *self.mi(i));
            let left = b.left.map(|i| *self.mi(i));
            let ctx = match (above, left) {
                (Some(a), Some(l)) => {
                    let (al, ll) = (!a.is_inter(), !l.is_inter());
                    if al && ll {
                        3
                    } else {
                        usize::from(al || ll)
                    }
                }
                (Some(e), None) | (None, Some(e)) => 2 * usize::from(!e.is_inter()),
                (None, None) => 0,
            };
            let is_inter = self.r.read(self.st.fc.intra_inter[ctx]);
            self.count(|c| c.intra_inter[ctx][usize::from(is_inter)] += 1);
            is_inter
        };
        let tx_size = self.read_tx_size(b, !skip || !inter_block);
        self.st.mi_pool[b.mi].tx_size = tx_size;
        if inter_block {
            self.read_inter_block_mode_info(b);
        } else {
            self.read_intra_block_mode_info(b);
        }
    }
}
