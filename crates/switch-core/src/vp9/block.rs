//! Partition, block decode and reconstruction.

use super::idct::*;
use super::inter;
use super::intra;
use super::loopfilter;
use super::modeinfo::{Blk, TileDecoder};
use super::tables::KF_PARTITION_PROBS;
use super::tokens::scan_order;
use super::types::*;

const INTRA_MODE_TO_TX_TYPE: [u8; 10] = [
    DCT_DCT, ADST_DCT, DCT_ADST, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_ADST, ADST_DCT,
    ADST_ADST,
];

fn round_mv_q4(v: i32) -> i32 {
    (if v < 0 { v - 2 } else { v + 2 }) / 4
}

impl TileDecoder<'_> {
    fn read_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        has_rows: bool,
        has_cols: bool,
        bsl: usize,
    ) -> u8 {
        let above = usize::from((self.st.above_seg_ctx[mi_col] >> bsl) & 1);
        let left = usize::from((self.left_seg_ctx[mi_row & 7] >> bsl) & 1);
        let ctx = left * 2 + above + bsl * 4;
        let probs = if self.st.hdr.is_intra() {
            KF_PARTITION_PROBS[ctx]
        } else {
            self.st.fc.partition[ctx]
        };
        let partition = if has_rows && has_cols {
            self.r.read_tree(&PARTITION_TREE, &probs)
        } else if !has_rows && has_cols {
            if self.r.read(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows && !has_cols {
            if self.r.read(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if let Some(counts) = self.st.counts.as_mut() {
            counts.partition[ctx][usize::from(partition)] += 1;
        }
        partition
    }

    pub fn decode_partition(&mut self, mi_row: usize, mi_col: usize, bsize: usize, n4x4_l2: usize) {
        let n8x8_l2 = n4x4_l2 - 1;
        let num_8x8 = 1usize << n8x8_l2;
        let hbs = num_8x8 >> 1;
        let has_rows = mi_row + hbs < self.st.mi_rows;
        let has_cols = mi_col + hbs < self.st.mi_cols;
        if mi_row >= self.st.mi_rows || mi_col >= self.st.mi_cols {
            return;
        }
        let partition = self.read_partition(mi_row, mi_col, has_rows, has_cols, n8x8_l2);
        let subsize = usize::from(SUBSIZE[usize::from(partition)][bsize]);
        if hbs == 0 {
            let wl = 1 >> usize::from(partition & PARTITION_VERT != 0);
            let hl = 1 >> usize::from(partition & PARTITION_HORZ != 0);
            self.decode_block(mi_row, mi_col, subsize, 1, 1, (wl, hl));
        } else {
            match partition {
                PARTITION_NONE => {
                    self.decode_block(mi_row, mi_col, subsize, n4x4_l2, n4x4_l2, (0, 0))
                }
                PARTITION_HORZ => {
                    self.decode_block(mi_row, mi_col, subsize, n4x4_l2, n8x8_l2, (0, 0));
                    if has_rows {
                        self.decode_block(mi_row + hbs, mi_col, subsize, n4x4_l2, n8x8_l2, (0, 0));
                    }
                }
                PARTITION_VERT => {
                    self.decode_block(mi_row, mi_col, subsize, n8x8_l2, n4x4_l2, (0, 0));
                    if has_cols {
                        self.decode_block(mi_row, mi_col + hbs, subsize, n8x8_l2, n4x4_l2, (0, 0));
                    }
                }
                _ => {
                    self.decode_partition(mi_row, mi_col, subsize, n8x8_l2);
                    self.decode_partition(mi_row, mi_col + hbs, subsize, n8x8_l2);
                    self.decode_partition(mi_row + hbs, mi_col, subsize, n8x8_l2);
                    self.decode_partition(mi_row + hbs, mi_col + hbs, subsize, n8x8_l2);
                }
            }
        }
        if bsize >= BLOCK_8X8 && (bsize == BLOCK_8X8 || partition != PARTITION_SPLIT) {
            let (above, left) = PARTITION_CONTEXT[subsize];
            self.st.above_seg_ctx[mi_col..mi_col + num_8x8].fill(above);
            for i in 0..num_8x8 {
                self.left_seg_ctx[(mi_row + i) & 7] = left;
            }
        }
    }

    fn decode_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize: usize,
        bwl: usize,
        bhl: usize,
        bmode: (usize, usize),
    ) {
        let bw = 1usize << (bwl - 1);
        let bh = 1usize << (bhl - 1);
        let x_mis = bw.min(self.st.mi_cols - mi_col);
        let y_mis = bh.min(self.st.mi_rows - mi_row);

        let mi = self.st.mi_pool.len();
        self.st.mi_pool.push(super::frame::ModeInfo {
            sb_type: bsize as u8,
            ..Default::default()
        });
        for y in 0..y_mis {
            for x in 0..x_mis {
                self.st.mi_grid[(mi_row + y) * self.st.mi_cols + mi_col + x] = mi as u32;
            }
        }
        let above = (mi_row != 0).then(|| self.st.mi_index(mi_row - 1, mi_col));
        let left = (mi_col > self.mi_col_start).then(|| self.st.mi_index(mi_row, mi_col - 1));
        let b = Blk {
            mi_row,
            mi_col,
            bsize,
            bw,
            bh,
            x_mis,
            y_mis,
            mb_to_left_edge: -((mi_col * 8) as i32 * 8),
            mb_to_right_edge: (self.st.mi_cols as i32 - bw as i32 - mi_col as i32) * 8 * 8,
            mb_to_top_edge: -((mi_row * 8) as i32 * 8),
            mb_to_bottom_edge: (self.st.mi_rows as i32 - bh as i32 - mi_row as i32) * 8 * 8,
            mi,
            above,
            left,
            bmode_blocks_wl: bmode.0,
            bmode_blocks_hl: bmode.1,
            bwl,
            bhl,
        };

        self.read_mode_info(&b);
        let info = self.st.mi_pool[mi];
        if info.skip {
            self.reset_skip_context(&b);
        }

        if !info.is_inter() {
            for plane in 0..3 {
                let tx_size = if plane > 0 {
                    uv_tx_size(bsize, info.tx_size)
                } else {
                    info.tx_size
                };
                let (max_w, max_h) = self.max_blocks(&b, plane);
                let step = 1usize << tx_size;
                for row in (0..max_h).step_by(step) {
                    for col in (0..max_w).step_by(step) {
                        self.predict_and_reconstruct_intra(&b, plane, row, col, tx_size);
                    }
                }
            }
        } else {
            self.build_inter_predictors(&b);
            if !info.skip {
                let mut eob_total = 0;
                for plane in 0..3 {
                    let tx_size = if plane > 0 {
                        uv_tx_size(bsize, info.tx_size)
                    } else {
                        info.tx_size
                    };
                    let (max_w, max_h) = self.max_blocks(&b, plane);
                    let step = 1usize << tx_size;
                    for row in (0..max_h).step_by(step) {
                        for col in (0..max_w).step_by(step) {
                            eob_total += self.reconstruct_inter(&b, plane, row, col, tx_size);
                        }
                    }
                }
                if bsize >= BLOCK_8X8 && eob_total == 0 {
                    self.st.mi_pool[mi].skip = true;
                }
            }
        }

        if self.st.lf.level != 0 {
            loopfilter::build_mask(self.st, mi, mi_row, mi_col, bw, bh);
        }
    }

    /// The transform blocks of `plane` that start inside the frame, in 4x4 units.
    fn max_blocks(&self, b: &Blk, plane: usize) -> (usize, usize) {
        let ss = usize::from(plane > 0);
        let n4_w = (1usize << b.bwl) >> ss;
        let n4_h = (1usize << b.bhl) >> ss;
        let wide = if b.mb_to_right_edge >= 0 {
            n4_w
        } else {
            (n4_w as i32 + (b.mb_to_right_edge >> (5 + ss))) as usize
        };
        let high = if b.mb_to_bottom_edge >= 0 {
            n4_h
        } else {
            (n4_h as i32 + (b.mb_to_bottom_edge >> (5 + ss))) as usize
        };
        (wide, high)
    }

    fn reset_skip_context(&mut self, b: &Blk) {
        for plane in 0..3 {
            let ss = usize::from(plane > 0);
            let n4_w = (1usize << b.bwl) >> ss;
            let n4_h = (1usize << b.bhl) >> ss;
            let a = (b.mi_col * 2) >> ss;
            let l = ((b.mi_row & 7) * 2) >> ss;
            self.st.above_ctx[plane][a..a + n4_w].fill(0);
            self.left_ctx[plane][l..l + n4_h].fill(0);
        }
    }

    /// Reads the tokens of one transform block; returns the end of block position.
    fn decode_block_tokens(
        &mut self,
        b: &Blk,
        plane: usize,
        row: usize,
        col: usize,
        tx_size: u8,
        tx_type: u8,
    ) -> usize {
        let ss = usize::from(plane > 0);
        let info = self.st.mi_pool[b.mi];
        let (scan, nb) = scan_order(tx_size, tx_type);
        let n = 1usize << tx_size;
        let a_off = ((b.mi_col * 2) >> ss) + col;
        let l_off = (((b.mi_row & 7) * 2) >> ss) + row;
        let a_nz = self.st.above_ctx[plane][a_off..a_off + n]
            .iter()
            .any(|&v| v != 0);
        let l_nz = self.left_ctx[plane][l_off..l_off + n]
            .iter()
            .any(|&v| v != 0);
        let ctx = usize::from(a_nz) + usize::from(l_nz);
        let dq = self.st.dequant[usize::from(info.segment_id)][ss];
        let eob = self.decode_coefs(ss, tx_size, info.is_inter(), dq, ctx, scan, nb);

        let (max_w, max_h) = self.max_blocks(b, plane);
        let clip_w = if b.mb_to_right_edge >= 0 {
            usize::MAX
        } else {
            max_w
        };
        let clip_h = if b.mb_to_bottom_edge >= 0 {
            usize::MAX
        } else {
            max_h
        };
        for i in 0..n {
            self.st.above_ctx[plane][a_off + i] = u8::from(eob > 0 && col + i < clip_w);
            self.left_ctx[plane][l_off + i] = u8::from(eob > 0 && row + i < clip_h);
        }
        eob
    }

    fn add_residual(&mut self, plane: usize, pos: usize, tx_size: u8, tx_type: u8, lossless: bool) {
        let n = 4usize << tx_size;
        let stride = self.st.cur.planes[plane].stride;
        let dest = &mut self.st.cur.planes[plane].data[pos..];
        if lossless {
            inverse_wht_add(&self.dqcoeff[..16], dest, stride);
        } else {
            inverse_transform_add(tx_size, tx_type, &self.dqcoeff[..n * n], dest, stride);
        }
        self.dqcoeff[..n * n].fill(0);
    }

    fn predict_and_reconstruct_intra(
        &mut self,
        b: &Blk,
        plane: usize,
        row: usize,
        col: usize,
        tx_size: u8,
    ) {
        let info = self.st.mi_pool[b.mi];
        let ss = usize::from(plane > 0);
        let mut mode = if plane == 0 { info.mode } else { info.uv_mode };
        if usize::from(info.sb_type) < BLOCK_8X8 && plane == 0 {
            mode = info.bmi_mode[(row << 1) + col];
        }
        let stride = self.st.cur.planes[plane].stride;
        let x = ((b.mi_col * 8) >> ss) + col * 4;
        let y = ((b.mi_row * 8) >> ss) + row * 4;
        let pos = y * stride + x;
        let txw = 1usize << tx_size;
        let plane_ref = &self.st.cur.planes[plane];
        let neighbours = intra::Neighbours {
            have_top: row > 0 || b.above.is_some(),
            have_left: col > 0 || b.left.is_some(),
            have_right: col + txw < (1usize << (b.bwl - ss)),
            x0: x as i32,
            y0: y as i32,
            frame_width: plane_ref.aligned_width as i32,
            frame_height: plane_ref.aligned_height as i32,
            past_right: b.mb_to_right_edge < 0,
            past_bottom: b.mb_to_bottom_edge < 0,
        };
        intra::predict(
            &mut self.st.cur.planes[plane].data,
            pos,
            stride,
            mode,
            tx_size,
            &neighbours,
        );

        if !info.skip {
            let lossless = self.st.hdr.lossless;
            let tx_type = if plane > 0 || lossless {
                DCT_DCT
            } else {
                INTRA_MODE_TO_TX_TYPE[usize::from(mode)]
            };
            let eob = self.decode_block_tokens(b, plane, row, col, tx_size, tx_type);
            if eob > 0 {
                self.add_residual(plane, pos, tx_size, tx_type, lossless);
            }
        }
    }

    fn reconstruct_inter(
        &mut self,
        b: &Blk,
        plane: usize,
        row: usize,
        col: usize,
        tx_size: u8,
    ) -> usize {
        let ss = usize::from(plane > 0);
        let stride = self.st.cur.planes[plane].stride;
        let x = ((b.mi_col * 8) >> ss) + col * 4;
        let y = ((b.mi_row * 8) >> ss) + row * 4;
        let eob = self.decode_block_tokens(b, plane, row, col, tx_size, DCT_DCT);
        if eob > 0 {
            let lossless = self.st.hdr.lossless;
            self.add_residual(plane, y * stride + x, tx_size, DCT_DCT, lossless);
        }
        eob
    }

    fn build_inter_predictors(&mut self, b: &Blk) {
        let info = self.st.mi_pool[b.mi];
        let compound = info.has_second_ref();
        for which in 0..1 + usize::from(compound) {
            let frame = (info.ref_frame[which] - LAST_FRAME) as usize;
            let reference = self.st.refs[frame]
                .clone()
                .expect("inter block without a reference frame");
            for plane in 0..3 {
                let ss = usize::from(plane > 0);
                let n4_w = (1usize << b.bwl) >> ss;
                let n4_h = (1usize << b.bhl) >> ss;
                let stride = self.st.cur.planes[plane].stride;
                let base_x = ((b.mi_col * 8) >> ss) as i32;
                let base_y = ((b.mi_row * 8) >> ss) as i32;
                if usize::from(info.sb_type) < BLOCK_8X8 {
                    let mut i = 0;
                    for y in 0..n4_h {
                        for x in 0..n4_w {
                            let mv = self.average_split_mv(&info, which, i, ss);
                            i += 1;
                            let (px, py) = (base_x + 4 * x as i32, base_y + 4 * y as i32);
                            inter::predict(
                                &mut self.st.cur.planes[plane].data,
                                py as usize * stride + px as usize,
                                stride,
                                &reference.planes[plane],
                                (px, py),
                                (mv.1 * (1 << (1 - ss)), mv.0 * (1 << (1 - ss))),
                                (4, 4),
                                info.interp_filter,
                                which > 0,
                            );
                        }
                    }
                } else {
                    let mv = info.mv[which];
                    inter::predict(
                        &mut self.st.cur.planes[plane].data,
                        base_y as usize * stride + base_x as usize,
                        stride,
                        &reference.planes[plane],
                        (base_x, base_y),
                        (
                            i32::from(mv.col) * (1 << (1 - ss)),
                            i32::from(mv.row) * (1 << (1 - ss)),
                        ),
                        (n4_w * 4, n4_h * 4),
                        info.interp_filter,
                        which > 0,
                    );
                }
            }
        }
    }

    /// The motion vector (row, col) of 4x4 block `block` in a chroma plane: sub-blocks averaged.
    fn average_split_mv(
        &self,
        info: &super::frame::ModeInfo,
        which: usize,
        block: usize,
        ss: usize,
    ) -> (i32, i32) {
        let mv = |i: usize| info.bmi_mv[i][which];
        let (rows, cols): (i32, i32);
        if ss == 0 {
            let m = mv(block);
            return (i32::from(m.row), i32::from(m.col));
        }
        let all = (0..4).map(mv);
        rows = round_mv_q4(all.clone().map(|m| i32::from(m.row)).sum());
        cols = round_mv_q4(all.map(|m| i32::from(m.col)).sum());
        (rows, cols)
    }
}
