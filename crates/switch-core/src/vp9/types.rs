//! Block, transform and mode enumerations with the small lookup tables the decoder indexes by them.

pub const BLOCK_4X4: usize = 0;
pub const BLOCK_4X8: usize = 1;
pub const BLOCK_8X4: usize = 2;
pub const BLOCK_8X8: usize = 3;
pub const BLOCK_8X16: usize = 4;
pub const BLOCK_16X8: usize = 5;
pub const BLOCK_16X16: usize = 6;
pub const BLOCK_16X32: usize = 7;
pub const BLOCK_32X16: usize = 8;
pub const BLOCK_32X32: usize = 9;
pub const BLOCK_32X64: usize = 10;
pub const BLOCK_64X32: usize = 11;
pub const BLOCK_64X64: usize = 12;
pub const BLOCK_INVALID: u8 = 13;

pub const PARTITION_NONE: u8 = 0;
pub const PARTITION_HORZ: u8 = 1;
pub const PARTITION_VERT: u8 = 2;
pub const PARTITION_SPLIT: u8 = 3;

pub const TX_4X4: u8 = 0;
pub const TX_8X8: u8 = 1;
pub const TX_16X16: u8 = 2;
pub const TX_32X32: u8 = 3;

pub const ONLY_4X4: u8 = 0;
pub const ALLOW_32X32: u8 = 3;
pub const TX_MODE_SELECT: u8 = 4;

pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const D45_PRED: u8 = 3;
pub const D135_PRED: u8 = 4;
pub const D117_PRED: u8 = 5;
pub const D153_PRED: u8 = 6;
pub const D207_PRED: u8 = 7;
pub const D63_PRED: u8 = 8;
pub const TM_PRED: u8 = 9;
pub const NEARESTMV: u8 = 10;
pub const NEARMV: u8 = 11;
pub const ZEROMV: u8 = 12;
pub const NEWMV: u8 = 13;

pub const INTRA_FRAME: i8 = 0;
pub const LAST_FRAME: i8 = 1;
pub const GOLDEN_FRAME: i8 = 2;
pub const ALTREF_FRAME: i8 = 3;
pub const NO_REF: i8 = -1;

pub const EIGHTTAP: u8 = 0;
pub const EIGHTTAP_SMOOTH: u8 = 1;
pub const EIGHTTAP_SHARP: u8 = 2;
pub const BILINEAR: u8 = 3;
pub const SWITCHABLE: u8 = 4;

pub const SINGLE_REFERENCE: u8 = 0;
pub const COMPOUND_REFERENCE: u8 = 1;
pub const REFERENCE_MODE_SELECT: u8 = 2;

pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_ALT_LF: usize = 1;
pub const SEG_LVL_REF_FRAME: usize = 2;
pub const SEG_LVL_SKIP: usize = 3;

pub const B_WIDTH_LOG2: [u8; 13] = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4];
pub const B_HEIGHT_LOG2: [u8; 13] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
pub const NUM_4X4_WIDE: [u8; 13] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16];
pub const NUM_4X4_HIGH: [u8; 13] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16];
pub const NUM_8X8_WIDE: [u8; 13] = [1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8];
pub const NUM_8X8_HIGH: [u8; 13] = [1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8];
pub const SIZE_GROUP: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
pub const MAX_TX_SIZE: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
pub const TX_MODE_TO_BIGGEST_TX_SIZE: [u8; 5] = [0, 1, 2, 3, 3];

/// `above` and `left` partition context written by a block of each size.
pub const PARTITION_CONTEXT: [(u8, u8); 13] = [
    (15, 15),
    (15, 14),
    (14, 15),
    (14, 14),
    (14, 12),
    (12, 14),
    (12, 12),
    (12, 8),
    (8, 12),
    (8, 8),
    (8, 0),
    (0, 8),
    (0, 0),
];

const INV: u8 = BLOCK_INVALID;

pub const SUBSIZE: [[u8; 13]; 4] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [INV, INV, INV, 2, INV, INV, 5, INV, INV, 8, INV, INV, 11],
    [INV, INV, INV, 1, INV, INV, 4, INV, INV, 7, INV, INV, 10],
    [INV, INV, INV, 0, INV, INV, 3, INV, INV, 6, INV, INV, 9],
];

/// The 4:2:0 chroma block of each luma block size.
pub const UV_BLOCK_420: [u8; 13] = [INV, INV, INV, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// The chroma transform size of a luma block of `bsize` with transform `tx` in 4:2:0.
pub fn uv_tx_size(bsize: usize, tx: u8) -> u8 {
    if bsize < BLOCK_8X8 {
        return TX_4X4;
    }
    tx.min(MAX_TX_SIZE[UV_BLOCK_420[bsize] as usize])
}

pub const INTRA_MODE_TREE: [i8; 18] = [
    -(DC_PRED as i8),
    2,
    -(TM_PRED as i8),
    4,
    -(V_PRED as i8),
    6,
    8,
    12,
    -(H_PRED as i8),
    10,
    -(D135_PRED as i8),
    -(D117_PRED as i8),
    -(D45_PRED as i8),
    14,
    -(D63_PRED as i8),
    16,
    -(D153_PRED as i8),
    -(D207_PRED as i8),
];

/// Leaves are `mode - NEARESTMV`.
pub const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];

pub const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const SWITCHABLE_INTERP_TREE: [i8; 4] = [0, 2, -1, -2];
pub const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
pub const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
pub const MV_CLASS0_TREE: [i8; 2] = [0, -1];
pub const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
