//! The in-loop deblocking filter, built from per-superblock edge masks as libvpx's 4:2:0 path does.

use super::frame::*;
use super::header::LoopFilterParams;
use super::types::*;

const LEFT_64X64_TXFORM_MASK: [u64; 4] = [!0, !0, 0x5555_5555_5555_5555, 0x1111_1111_1111_1111];
const ABOVE_64X64_TXFORM_MASK: [u64; 4] = [!0, !0, 0x00ff_00ff_00ff_00ff, 0x0000_00ff_0000_00ff];
const LEFT_PREDICTION_MASK: [u64; 13] = [
    1,
    1,
    1,
    1,
    0x101,
    1,
    0x101,
    0x0101_0101,
    0x101,
    0x0101_0101,
    0x0101_0101_0101_0101,
    0x0101_0101,
    0x0101_0101_0101_0101,
];
const ABOVE_PREDICTION_MASK: [u64; 13] = [1, 1, 1, 1, 1, 3, 3, 3, 0xf, 0xf, 0xf, 0xff, 0xff];
const SIZE_MASK: [u64; 13] = [
    1,
    1,
    1,
    1,
    0x101,
    3,
    0x303,
    0x0303_0303,
    0x0f0f,
    0x0f0f_0f0f,
    0x0f0f_0f0f_0f0f_0f0f,
    0xffff_ffff,
    !0,
];
const LEFT_BORDER: u64 = 0x1111_1111_1111_1111;
const ABOVE_BORDER: u64 = 0x0000_00ff_0000_00ff;
const LEFT_64X64_TXFORM_MASK_UV: [u16; 4] = [0xffff, 0xffff, 0x5555, 0x1111];
const ABOVE_64X64_TXFORM_MASK_UV: [u16; 4] = [0xffff, 0xffff, 0x0f0f, 0x000f];
const LEFT_PREDICTION_MASK_UV: [u16; 13] =
    [1, 1, 1, 1, 1, 1, 1, 0x11, 1, 0x11, 0x1111, 0x11, 0x1111];
const ABOVE_PREDICTION_MASK_UV: [u16; 13] = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0xf, 0xf];
const SIZE_MASK_UV: [u16; 13] = [1, 1, 1, 1, 1, 1, 1, 0x11, 3, 0x33, 0x3333, 0xff, 0xffff];
const LEFT_BORDER_UV: u16 = 0x1111;
const ABOVE_BORDER_UV: u16 = 0x000f;
const MODE_LF_LUT: [usize; 14] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1];

/// Fills the filter levels of every segment, reference and mode.
pub fn frame_init(lf: &LoopFilterParams, seg: &super::header::Segmentation) -> [[[u8; 2]; 4]; 8] {
    let level = i32::from(lf.level);
    let scale = 1 << (level >> 5);
    let mut out = [[[0u8; 2]; 4]; 8];
    for (seg_id, row) in out.iter_mut().enumerate() {
        let mut lvl_seg = level;
        if seg.active(seg_id, SEG_LVL_ALT_LF) {
            let data = seg.data(seg_id, SEG_LVL_ALT_LF);
            lvl_seg = (if seg.abs_delta { data } else { level + data }).clamp(0, 63);
        }
        if !lf.delta_enabled {
            *row = [[lvl_seg as u8; 2]; 4];
        } else {
            let intra = lvl_seg + i32::from(lf.ref_deltas[0]) * scale;
            row[0][0] = intra.clamp(0, 63) as u8;
            for r in 1..4 {
                for mode in 0..2 {
                    let inter = lvl_seg
                        + i32::from(lf.ref_deltas[r]) * scale
                        + i32::from(lf.mode_deltas[mode]) * scale;
                    row[r][mode] = inter.clamp(0, 63) as u8;
                }
            }
        }
    }
    out
}

fn filter_level(st: &FrameState, mi: &ModeInfo) -> u8 {
    st.lf_levels[usize::from(mi.segment_id)][mi.ref_frame[0] as usize]
        [MODE_LF_LUT[usize::from(mi.mode)]]
}

/// Adds the edges of the block at `mi` to its superblock's masks.
pub fn build_mask(
    st: &mut FrameState,
    mi_idx: usize,
    mi_row: usize,
    mi_col: usize,
    bw: usize,
    bh: usize,
) {
    let mi = st.mi_pool[mi_idx];
    let block_size = usize::from(mi.sb_type);
    let tx_size_y = usize::from(mi.tx_size);
    let level = filter_level(st, &mi);
    let tx_size_uv = usize::from(uv_tx_size(block_size, mi.tx_size));
    let sb_stride = st.mi_cols.div_ceil(8);
    let lfm = &mut st.lfm[(mi_row >> 3) * sb_stride + (mi_col >> 3)];
    let row_in_sb = mi_row & 7;
    let col_in_sb = mi_col & 7;
    let shift_y = col_in_sb + (row_in_sb << 3);
    let shift_uv = (col_in_sb >> 1) + ((row_in_sb >> 1) << 2);
    let build_uv = row_in_sb.is_multiple_of(2) && col_in_sb.is_multiple_of(2);

    if level == 0 {
        return;
    }
    let mut index = shift_y;
    for _ in 0..bh {
        lfm.lfl_y[index..index + bw].fill(level);
        index += 8;
    }

    lfm.above_y[tx_size_y] |= ABOVE_PREDICTION_MASK[block_size] << shift_y;
    lfm.left_y[tx_size_y] |= LEFT_PREDICTION_MASK[block_size] << shift_y;
    if build_uv {
        lfm.above_uv[tx_size_uv] |= ABOVE_PREDICTION_MASK_UV[block_size] << shift_uv;
        lfm.left_uv[tx_size_uv] |= LEFT_PREDICTION_MASK_UV[block_size] << shift_uv;
    }
    if mi.skip && mi.is_inter() {
        return;
    }
    lfm.above_y[tx_size_y] |=
        (SIZE_MASK[block_size] & ABOVE_64X64_TXFORM_MASK[tx_size_y]) << shift_y;
    lfm.left_y[tx_size_y] |= (SIZE_MASK[block_size] & LEFT_64X64_TXFORM_MASK[tx_size_y]) << shift_y;
    if build_uv {
        lfm.above_uv[tx_size_uv] |=
            (SIZE_MASK_UV[block_size] & ABOVE_64X64_TXFORM_MASK_UV[tx_size_uv]) << shift_uv;
        lfm.left_uv[tx_size_uv] |=
            (SIZE_MASK_UV[block_size] & LEFT_64X64_TXFORM_MASK_UV[tx_size_uv]) << shift_uv;
    }
    if tx_size_y == 0 {
        lfm.int_4x4_y |= SIZE_MASK[block_size] << shift_y;
    }
    if build_uv && tx_size_uv == 0 {
        lfm.int_4x4_uv |= SIZE_MASK_UV[block_size] << shift_uv;
    }
}

fn adjust_mask(
    lfm: &mut LoopFilterMask,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    mi_cols: usize,
) {
    lfm.left_y[2] |= lfm.left_y[3];
    lfm.above_y[2] |= lfm.above_y[3];
    lfm.left_uv[2] |= lfm.left_uv[3];
    lfm.above_uv[2] |= lfm.above_uv[3];

    lfm.left_y[1] |= lfm.left_y[0] & LEFT_BORDER;
    lfm.left_y[0] &= !LEFT_BORDER;
    lfm.above_y[1] |= lfm.above_y[0] & ABOVE_BORDER;
    lfm.above_y[0] &= !ABOVE_BORDER;
    lfm.left_uv[1] |= lfm.left_uv[0] & LEFT_BORDER_UV;
    lfm.left_uv[0] &= !LEFT_BORDER_UV;
    lfm.above_uv[1] |= lfm.above_uv[0] & ABOVE_BORDER_UV;
    lfm.above_uv[0] &= !ABOVE_BORDER_UV;

    if mi_row + 8 > mi_rows {
        let rows = (mi_rows - mi_row) as u64;
        let mask_y = (1u64 << (rows << 3)) - 1;
        let mask_uv = ((1u32 << (((rows + 1) >> 1) << 2)) - 1) as u16;
        for i in 0..3 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv;
        if rows == 1 {
            lfm.above_uv[1] |= lfm.above_uv[2];
            lfm.above_uv[2] = 0;
        }
        if rows == 5 {
            lfm.above_uv[1] |= lfm.above_uv[2] & 0xff00;
            lfm.above_uv[2] &= !(lfm.above_uv[2] & 0xff00);
        }
    }

    if mi_col + 8 > mi_cols {
        let columns = (mi_cols - mi_col) as u64;
        let mask_y = ((1u64 << columns) - 1) * 0x0101_0101_0101_0101;
        let mask_uv = (((1u32 << ((columns + 1) >> 1)) - 1) * 0x1111) as u16;
        let mask_uv_int = (((1u32 << (columns >> 1)) - 1) * 0x1111) as u16;
        for i in 0..3 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv_int;
        if columns == 1 {
            lfm.left_uv[1] |= lfm.left_uv[2];
            lfm.left_uv[2] = 0;
        }
        if columns == 5 {
            lfm.left_uv[1] |= lfm.left_uv[2] & 0xcccc;
            lfm.left_uv[2] &= !(lfm.left_uv[2] & 0xcccc);
        }
    }

    if mi_col == 0 {
        for i in 0..3 {
            lfm.left_y[i] &= 0xfefe_fefe_fefe_fefe;
            lfm.left_uv[i] &= 0xeeee;
        }
    }
}

#[derive(Clone, Copy)]
struct Thresholds {
    mblim: i32,
    lim: i32,
    hev: i32,
}

fn thresholds(sharpness: u8) -> [Thresholds; 64] {
    let sharp = i32::from(sharpness);
    std::array::from_fn(|lvl| {
        let lvl = lvl as i32;
        let mut limit = lvl >> (i32::from(sharp > 0) + i32::from(sharp > 4));
        if sharp > 0 && limit > 9 - sharp {
            limit = 9 - sharp;
        }
        limit = limit.max(1);
        Thresholds {
            mblim: 2 * (lvl + 2) + limit,
            lim: limit,
            hev: lvl >> 4,
        }
    })
}

fn clamp8(t: i32) -> i32 {
    t.clamp(-128, 127)
}

struct Pixels<'a> {
    data: &'a mut [u8],
    /// Index of the first pixel past the edge, the edge lying between `-across` and `0`.
    at: isize,
    across: isize,
}

impl Pixels<'_> {
    fn get(&self, k: isize) -> i32 {
        i32::from(self.data[(self.at + k * self.across) as usize])
    }

    fn set(&mut self, k: isize, v: i32) {
        self.data[(self.at + k * self.across) as usize] = v as u8;
    }
}

fn filter_mask(limit: i32, blimit: i32, p: &[i32; 4], q: &[i32; 4]) -> bool {
    (p[3] - p[2]).abs() <= limit
        && (p[2] - p[1]).abs() <= limit
        && (p[1] - p[0]).abs() <= limit
        && (q[1] - q[0]).abs() <= limit
        && (q[2] - q[1]).abs() <= limit
        && (q[3] - q[2]).abs() <= limit
        && (p[0] - q[0]).abs() * 2 + (p[1] - q[1]).abs() / 2 <= blimit
}

fn flat_mask4(thresh: i32, p: &[i32; 4], q: &[i32; 4]) -> bool {
    (1..4).all(|i| (p[i] - p[0]).abs() <= thresh && (q[i] - q[0]).abs() <= thresh)
}

fn filter4(px: &mut Pixels, mask: bool, thresh: i32) {
    let (p1, p0, q0, q1) = (px.get(-2), px.get(-1), px.get(0), px.get(1));
    let (ps1, ps0, qs0, qs1) = (p1 - 128, p0 - 128, q0 - 128, q1 - 128);
    let hev = (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh;
    let mut filter = if hev { clamp8(ps1 - qs1) } else { 0 };
    filter = if mask {
        clamp8(filter + 3 * (qs0 - ps0))
    } else {
        0
    };
    let filter1 = clamp8(filter + 4) >> 3;
    let filter2 = clamp8(filter + 3) >> 3;
    px.set(0, clamp8(qs0 - filter1) + 128);
    px.set(-1, clamp8(ps0 + filter2) + 128);
    let outer = if hev { 0 } else { (filter1 + 1) >> 1 };
    px.set(1, clamp8(qs1 - outer) + 128);
    px.set(-2, clamp8(ps1 + outer) + 128);
}

fn filter8(px: &mut Pixels, mask: bool, flat: bool, thresh: i32) {
    if flat && mask {
        let (p3, p2, p1, p0) = (px.get(-4), px.get(-3), px.get(-2), px.get(-1));
        let (q0, q1, q2, q3) = (px.get(0), px.get(1), px.get(2), px.get(3));
        px.set(-3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
        px.set(-2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
        px.set(-1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
        px.set(0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
        px.set(1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
        px.set(2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
    } else {
        filter4(px, mask, thresh);
    }
}

fn filter16(px: &mut Pixels, mask: bool, flat: bool, flat2: bool, thresh: i32) {
    if flat2 && flat && mask {
        let p: Vec<i32> = (1..=8).map(|k| px.get(-k)).collect();
        let q: Vec<i32> = (0..8).map(|k| px.get(k)).collect();
        // p[i] is p_i counting away from the edge, q[i] likewise.
        let (p0, p1, p2, p3, p4, p5, p6, p7) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        let (q0, q1, q2, q3, q4, q5, q6, q7) = (q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
        let r = |sum: i32| (sum + 8) >> 4;
        px.set(-7, r(p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0));
        px.set(
            -6,
            r(p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1),
        );
        px.set(
            -5,
            r(p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2),
        );
        px.set(
            -4,
            r(p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3),
        );
        px.set(
            -3,
            r(p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4),
        );
        px.set(
            -2,
            r(p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5),
        );
        px.set(
            -1,
            r(p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6),
        );
        px.set(
            0,
            r(p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7),
        );
        px.set(
            1,
            r(p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2),
        );
        px.set(
            2,
            r(p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3),
        );
        px.set(
            3,
            r(p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4),
        );
        px.set(
            4,
            r(p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5),
        );
        px.set(
            5,
            r(p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6),
        );
        px.set(6, r(p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7));
    } else {
        filter8(px, mask, flat, thresh);
    }
}

/// Filters `count` pixels along an edge: kind 4, 8 or 16.
fn lpf(
    data: &mut [u8],
    at: usize,
    across: isize,
    along: isize,
    count: usize,
    kind: u32,
    t: Thresholds,
) {
    for i in 0..count {
        let mut px = Pixels {
            data: &mut data[..],
            at: at as isize + i as isize * along,
            across,
        };
        let p = [px.get(-1), px.get(-2), px.get(-3), px.get(-4)];
        let q = [px.get(0), px.get(1), px.get(2), px.get(3)];
        let mask = filter_mask(t.lim, t.mblim, &p, &q);
        match kind {
            4 => filter4(&mut px, mask, t.hev),
            8 => {
                let flat = flat_mask4(1, &p, &q);
                filter8(&mut px, mask, flat, t.hev);
            }
            _ => {
                let flat = flat_mask4(1, &p, &q);
                let (p0, q0) = (p[0], q[0]);
                let flat2 = (4..8).all(|k| (px.get(-(k as isize) - 1) - p0).abs() <= 1)
                    && (4..8).all(|k| (px.get(k as isize) - q0).abs() <= 1);
                filter16(&mut px, mask, flat, flat2, t.hev);
            }
        }
    }
}

/// `filter_selectively_vert_row2`: two rows of 8x8 blocks, `lfl_forward` entries apart in `lfl`.
fn filter_vert_row2(
    subsampled: bool,
    plane: &mut Plane,
    origin: usize,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    thr: &[Thresholds; 64],
    lfl: &[u8],
) {
    let pitch = plane.stride;
    let dual_mask_cutoff: u32 = if subsampled { 0xff } else { 0xffff };
    let lfl_forward = if subsampled { 4 } else { 8 };
    let dual_one: u32 = 1 | (1 << lfl_forward);
    let mut s = origin;
    let mut lfl_at = 0usize;
    let mut mask = (mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int) & dual_mask_cutoff;
    while mask != 0 {
        if mask & dual_one != 0 {
            let lfis = [
                thr[lfl[lfl_at] as usize],
                thr[lfl[lfl_at + lfl_forward] as usize],
            ];
            let ss = [s, s + 8 * pitch];
            let data = &mut plane.data[..];
            let (m16, m8, m4, m4i) = (
                mask_16x16 & dual_one,
                mask_8x8 & dual_one,
                mask_4x4 & dual_one,
                mask_4x4_int & dual_one,
            );
            if m16 != 0 {
                if m16 == dual_one {
                    lpf(data, ss[0], 1, pitch as isize, 16, 16, lfis[0]);
                } else {
                    let which = usize::from(m16 & 1 == 0);
                    lpf(data, ss[which], 1, pitch as isize, 8, 16, lfis[which]);
                }
            }
            if m8 != 0 {
                if m8 == dual_one {
                    lpf(data, ss[0], 1, pitch as isize, 8, 8, lfis[0]);
                    lpf(data, ss[1], 1, pitch as isize, 8, 8, lfis[1]);
                } else {
                    let which = usize::from(m8 & 1 == 0);
                    lpf(data, ss[which], 1, pitch as isize, 8, 8, lfis[which]);
                }
            }
            if m4 != 0 {
                if m4 == dual_one {
                    lpf(data, ss[0], 1, pitch as isize, 8, 4, lfis[0]);
                    lpf(data, ss[1], 1, pitch as isize, 8, 4, lfis[1]);
                } else {
                    let which = usize::from(m4 & 1 == 0);
                    lpf(data, ss[which], 1, pitch as isize, 8, 4, lfis[which]);
                }
            }
            if m4i != 0 {
                if m4i == dual_one {
                    lpf(data, ss[0] + 4, 1, pitch as isize, 8, 4, lfis[0]);
                    lpf(data, ss[1] + 4, 1, pitch as isize, 8, 4, lfis[1]);
                } else {
                    let which = usize::from(m4i & 1 == 0);
                    lpf(data, ss[which] + 4, 1, pitch as isize, 8, 4, lfis[which]);
                }
            }
        }
        s += 8;
        lfl_at += 1;
        mask_16x16 >>= 1;
        mask_8x8 >>= 1;
        mask_4x4 >>= 1;
        mask_4x4_int >>= 1;
        mask = (mask & !dual_one) >> 1;
    }
}

/// `filter_selectively_horiz` for one row of 8x8 blocks.
fn filter_horiz(
    plane: &mut Plane,
    origin: usize,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    thr: &[Thresholds; 64],
    lfl: &[u8],
) {
    let pitch = plane.stride;
    let across = pitch as isize;
    let data = &mut plane.data[..];
    let mut s = origin;
    let mut lfl_at = 0usize;
    let mut mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
    while mask != 0 {
        let mut count = 1;
        if mask & 1 != 0 {
            let lfi = thr[lfl[lfl_at] as usize];
            if mask_16x16 & 1 != 0 {
                if mask_16x16 & 3 == 3 {
                    lpf(data, s, across, 1, 16, 16, lfi);
                    count = 2;
                } else {
                    lpf(data, s, across, 1, 8, 16, lfi);
                }
            } else if mask_8x8 & 1 != 0 {
                if mask_8x8 & 3 == 3 {
                    let lfin = thr[lfl[lfl_at + 1] as usize];
                    lpf(data, s, across, 1, 8, 8, lfi);
                    lpf(data, s + 8, across, 1, 8, 8, lfin);
                    if mask_4x4_int & 3 == 3 {
                        lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
                        lpf(data, s + 8 + 4 * pitch, across, 1, 8, 4, lfin);
                    } else if mask_4x4_int & 1 != 0 {
                        lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
                    } else if mask_4x4_int & 2 != 0 {
                        lpf(data, s + 8 + 4 * pitch, across, 1, 8, 4, lfin);
                    }
                    count = 2;
                } else {
                    lpf(data, s, across, 1, 8, 8, lfi);
                    if mask_4x4_int & 1 != 0 {
                        lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
                    }
                }
            } else if mask_4x4 & 1 != 0 {
                if mask_4x4 & 3 == 3 {
                    let lfin = thr[lfl[lfl_at + 1] as usize];
                    lpf(data, s, across, 1, 8, 4, lfi);
                    lpf(data, s + 8, across, 1, 8, 4, lfin);
                    if mask_4x4_int & 3 == 3 {
                        lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
                        lpf(data, s + 8 + 4 * pitch, across, 1, 8, 4, lfin);
                    } else if mask_4x4_int & 1 != 0 {
                        lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
                    } else if mask_4x4_int & 2 != 0 {
                        lpf(data, s + 8 + 4 * pitch, across, 1, 8, 4, lfin);
                    }
                    count = 2;
                } else {
                    lpf(data, s, across, 1, 8, 4, lfi);
                    if mask_4x4_int & 1 != 0 {
                        lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
                    }
                }
            } else {
                lpf(data, s + 4 * pitch, across, 1, 8, 4, lfi);
            }
        }
        s += 8 * count;
        lfl_at += count;
        mask_16x16 >>= count;
        mask_8x8 >>= count;
        mask_4x4 >>= count;
        mask_4x4_int >>= count;
        mask >>= count;
    }
}

fn filter_block_plane_ss00(
    plane: &mut Plane,
    origin: usize,
    mi_row: usize,
    mi_rows: usize,
    lfm: &LoopFilterMask,
    thr: &[Thresholds; 64],
) {
    let mut m16 = lfm.left_y[2];
    let mut m8 = lfm.left_y[1];
    let mut m4 = lfm.left_y[0];
    let mut m4i = lfm.int_4x4_y;
    let pitch = plane.stride;
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        filter_vert_row2(
            false,
            plane,
            origin + r * 8 * pitch,
            m16 as u32,
            m8 as u32,
            m4 as u32,
            m4i as u32,
            thr,
            &lfm.lfl_y[r << 3..],
        );
        m16 >>= 16;
        m8 >>= 16;
        m4 >>= 16;
        m4i >>= 16;
        r += 2;
    }
    let mut m16 = lfm.above_y[2];
    let mut m8 = lfm.above_y[1];
    let mut m4 = lfm.above_y[0];
    let mut m4i = lfm.int_4x4_y;
    for r in 0..8 {
        if mi_row + r >= mi_rows {
            break;
        }
        let (m16r, m8r, m4r) = if mi_row + r == 0 {
            (0, 0, 0)
        } else {
            ((m16 & 0xff) as u32, (m8 & 0xff) as u32, (m4 & 0xff) as u32)
        };
        filter_horiz(
            plane,
            origin + r * 8 * pitch,
            m16r,
            m8r,
            m4r,
            (m4i & 0xff) as u32,
            thr,
            &lfm.lfl_y[r << 3..],
        );
        m16 >>= 8;
        m8 >>= 8;
        m4 >>= 8;
        m4i >>= 8;
    }
}

fn filter_block_plane_ss11(
    plane: &mut Plane,
    origin: usize,
    mi_row: usize,
    mi_rows: usize,
    lfm: &LoopFilterMask,
    thr: &[Thresholds; 64],
) {
    let pitch = plane.stride;
    let mut lfl_uv = [0u8; 16];
    let mut m16 = u32::from(lfm.left_uv[2]);
    let mut m8 = u32::from(lfm.left_uv[1]);
    let mut m4 = u32::from(lfm.left_uv[0]);
    let mut m4i = u32::from(lfm.int_4x4_uv);
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        for c in 0..4 {
            lfl_uv[(r << 1) + c] = lfm.lfl_y[(r << 3) + (c << 1)];
            lfl_uv[((r + 2) << 1) + c] = lfm.lfl_y[((r + 2) << 3) + (c << 1)];
        }
        filter_vert_row2(
            true,
            plane,
            origin + (r >> 2) * 16 * pitch,
            m16,
            m8,
            m4,
            m4i,
            thr,
            &lfl_uv[r << 1..],
        );
        m16 >>= 8;
        m8 >>= 8;
        m4 >>= 8;
        m4i >>= 8;
        r += 4;
    }
    let mut m16 = u32::from(lfm.above_uv[2]);
    let mut m8 = u32::from(lfm.above_uv[1]);
    let mut m4 = u32::from(lfm.above_uv[0]);
    let m4i_all = u32::from(lfm.int_4x4_uv);
    let mut m4i = m4i_all;
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        let skip_border_4x4_r = mi_row + r == mi_rows - 1;
        let m4i_r = if skip_border_4x4_r { 0 } else { m4i & 0xf };
        let (m16r, m8r, m4r) = if mi_row + r == 0 {
            (0, 0, 0)
        } else {
            (m16 & 0xf, m8 & 0xf, m4 & 0xf)
        };
        filter_horiz(
            plane,
            origin + (r >> 1) * 8 * pitch,
            m16r,
            m8r,
            m4r,
            m4i_r,
            thr,
            &lfl_uv[r << 1..],
        );
        m16 >>= 4;
        m8 >>= 4;
        m4 >>= 4;
        m4i >>= 4;
        r += 2;
    }
}

/// Filters the whole frame in superblock raster order.
pub fn filter_frame(st: &mut FrameState, lf: &LoopFilterParams) {
    let thr = thresholds(lf.sharpness);
    let sb_stride = st.mi_cols.div_ceil(8);
    let (mi_rows, mi_cols) = (st.mi_rows, st.mi_cols);
    for mi_row in (0..mi_rows).step_by(8) {
        for mi_col in (0..mi_cols).step_by(8) {
            let mut lfm = st.lfm[(mi_row >> 3) * sb_stride + (mi_col >> 3)];
            adjust_mask(&mut lfm, mi_row, mi_col, mi_rows, mi_cols);
            let y_origin = {
                let p = &st.cur.planes[0];
                mi_row * 8 * p.stride + mi_col * 8
            };
            filter_block_plane_ss00(&mut st.cur.planes[0], y_origin, mi_row, mi_rows, &lfm, &thr);
            for plane in 1..3 {
                let origin = {
                    let p = &st.cur.planes[plane];
                    (mi_row * 4) * p.stride + mi_col * 4
                };
                filter_block_plane_ss11(
                    &mut st.cur.planes[plane],
                    origin,
                    mi_row,
                    mi_rows,
                    &lfm,
                    &thr,
                );
            }
        }
    }
}
