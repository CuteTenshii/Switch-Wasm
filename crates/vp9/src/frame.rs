//! Frames, per-block mode info, and the state one frame's decode shares across its tiles.

use super::header::{FrameHeader, LoopFilterParams, Segmentation};
use super::probs::{Counts, FrameContext};
use super::types::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

/// What the next frame's motion vector prediction reads of this one.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvRef {
    pub ref_frame: [i8; 2],
    pub mv: [Mv; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ModeInfo {
    pub sb_type: u8,
    pub mode: u8,
    pub uv_mode: u8,
    pub ref_frame: [i8; 2],
    pub mv: [Mv; 2],
    pub bmi_mode: [u8; 4],
    pub bmi_mv: [[Mv; 2]; 4],
    pub interp_filter: u8,
    pub tx_size: u8,
    pub skip: bool,
    pub segment_id: u8,
    pub seg_id_predicted: bool,
}

impl ModeInfo {
    pub fn is_inter(&self) -> bool {
        self.ref_frame[0] > INTRA_FRAME
    }

    pub fn has_second_ref(&self) -> bool {
        self.ref_frame[1] > INTRA_FRAME
    }

    /// `get_y_mode`: the mode of 4x4 `block`, or the block's own mode when it is not split.
    pub fn y_mode(&self, block: usize) -> u8 {
        if usize::from(self.sb_type) < BLOCK_8X8 {
            self.bmi_mode[block]
        } else {
            self.mode
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Plane {
    pub data: Vec<u8>,
    pub stride: usize,
    /// The visible size.
    pub width: usize,
    pub height: usize,
    /// The size rounded up to a multiple of 8, where decoding is defined.
    pub aligned_width: usize,
    pub aligned_height: usize,
}

impl Plane {
    fn new(width: usize, height: usize, aligned_width: usize, aligned_height: usize) -> Plane {
        let stride = (aligned_width + 63) & !63;
        let rows = (aligned_height + 63) & !63;
        Plane {
            data: vec![0; stride * rows],
            stride,
            width,
            height,
            aligned_width,
            aligned_height,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub planes: [Plane; 3],
    pub mi_rows: usize,
    pub mi_cols: usize,
    /// Motion vectors of every 8x8, for the next frame's prediction.
    pub mvs: Vec<MvRef>,
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Frame {
        let (w, h) = (width as usize, height as usize);
        let (aw, ah) = ((w + 7) & !7, (h + 7) & !7);
        let mi_cols = w.div_ceil(8);
        let mi_rows = h.div_ceil(8);
        Frame {
            width,
            height,
            planes: [
                Plane::new(w, h, aw, ah),
                Plane::new(w.div_ceil(2), h.div_ceil(2), aw / 2, ah / 2),
                Plane::new(w.div_ceil(2), h.div_ceil(2), aw / 2, ah / 2),
            ],
            mi_rows,
            mi_cols,
            mvs: vec![MvRef::default(); mi_rows * mi_cols],
        }
    }
}

/// The loop filter's per-superblock edge masks.
#[derive(Clone, Copy)]
pub struct LoopFilterMask {
    pub left_y: [u64; 4],
    pub above_y: [u64; 4],
    pub int_4x4_y: u64,
    pub left_uv: [u16; 4],
    pub above_uv: [u16; 4],
    pub int_4x4_uv: u16,
    pub lfl_y: [u8; 64],
}

impl Default for LoopFilterMask {
    fn default() -> LoopFilterMask {
        LoopFilterMask {
            left_y: [0; 4],
            above_y: [0; 4],
            int_4x4_y: 0,
            left_uv: [0; 4],
            above_uv: [0; 4],
            int_4x4_uv: 0,
            lfl_y: [0; 64],
        }
    }
}

/// Everything one frame's tiles read and write.
pub struct FrameState {
    pub hdr: FrameHeader,
    pub fc: FrameContext,
    pub counts: Option<Counts>,
    pub seg: Segmentation,
    pub lf: LoopFilterParams,
    pub cur: Frame,
    pub refs: [Option<Arc<Frame>>; 3],
    pub prev_mvs: Option<Arc<Frame>>,
    pub mi_cols: usize,
    pub mi_rows: usize,
    /// Mode info of every block, index 0 being an all-zero block for what is not decoded.
    pub mi_pool: Vec<ModeInfo>,
    pub mi_grid: Vec<u32>,
    pub above_ctx: [Vec<u8>; 3],
    pub above_seg_ctx: Vec<u8>,
    pub last_seg_map: Vec<u8>,
    pub cur_seg_map: Vec<u8>,
    /// `[segment][plane > 0][dc, ac]`.
    pub dequant: [[[i32; 2]; 2]; 8],
    pub lfm: Vec<LoopFilterMask>,
    pub lf_levels: [[[u8; 2]; 4]; 8],
    pub use_prev_frame_mvs: bool,
}

impl FrameState {
    pub fn mi_index(&self, row: usize, col: usize) -> usize {
        self.mi_grid[row * self.mi_cols + col] as usize
    }
}
