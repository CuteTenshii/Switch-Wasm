//! The decoder: frame headers, tiles, loop filter, adaptation and reference management.

use super::bool_decoder::BoolDecoder;
use super::frame::*;
use super::header::*;
use super::loopfilter;
use super::modeinfo::TileDecoder;
use super::probs::{Counts, FrameContext};
use super::tables::{AC_QLOOKUP, DC_QLOOKUP};
use super::Error;
use std::sync::Arc;

pub struct Decoder {
    pub(super) ref_frame_map: [Option<Arc<Frame>>; 8],
    pub(super) frame_contexts: [FrameContext; 4],
    pub(super) fc: FrameContext,
    pub(super) seg: Segmentation,
    pub(super) lf: LoopFilterParams,
    pub(super) last_seg_map: Vec<u8>,
    pub(super) current_seg_map: Vec<u8>,
    pub(super) frame_key: bool,
    pub(super) last_frame_key: bool,
    pub(super) intra_only: bool,
    pub(super) last_intra_only: bool,
    last_show_frame: bool,
    last_width: u32,
    last_height: u32,
    prev_frame: Option<Arc<Frame>>,
}

impl Default for Decoder {
    fn default() -> Decoder {
        Decoder::new()
    }
}

fn dc_quant(qindex: i32, delta: i32) -> i32 {
    i32::from(DC_QLOOKUP[(qindex + delta).clamp(0, 255) as usize])
}

fn ac_quant(qindex: i32, delta: i32) -> i32 {
    i32::from(AC_QLOOKUP[(qindex + delta).clamp(0, 255) as usize])
}

/// The size of each frame in a packet, from its superframe index when it has one.
pub fn split_superframe(data: &[u8]) -> Vec<&[u8]> {
    if let Some(&marker) = data.last() {
        if marker & 0xe0 == 0xc0 {
            let frames = usize::from(marker & 7) + 1;
            let mag = usize::from((marker >> 3) & 3) + 1;
            let index_size = 2 + mag * frames;
            if data.len() >= index_size && data[data.len() - index_size] == marker {
                let mut out = Vec::new();
                let mut at = 0;
                let index = &data[data.len() - index_size + 1..];
                for i in 0..frames {
                    let size = (0..mag).fold(0usize, |v, j| {
                        v | usize::from(index[i * mag + j]) << (8 * j)
                    });
                    if at + size > data.len() - index_size {
                        break;
                    }
                    out.push(&data[at..at + size]);
                    at += size;
                }
                return out;
            }
        }
    }
    vec![data]
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder {
            ref_frame_map: Default::default(),
            frame_contexts: [FrameContext::defaults(); 4],
            fc: FrameContext::defaults(),
            seg: Segmentation::default(),
            lf: LoopFilterParams::default(),
            last_seg_map: Vec::new(),
            current_seg_map: Vec::new(),
            frame_key: false,
            last_frame_key: false,
            intra_only: false,
            last_intra_only: false,
            last_show_frame: false,
            last_width: 0,
            last_height: 0,
            prev_frame: None,
        }
    }

    /// Decodes one packet, which may hold several frames, and returns the frames to show.
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<Arc<Frame>>, Error> {
        let mut shown = Vec::new();
        for frame in split_superframe(data) {
            if let Some(frame) = self.decode_frame(frame)? {
                shown.push(frame);
            }
        }
        Ok(shown)
    }

    pub fn decode_frame(&mut self, data: &[u8]) -> Result<Option<Arc<Frame>>, Error> {
        let decoded = self.decode_frame_inspecting(data, &mut |_, _| {})?;
        Ok(decoded.shown.then_some(decoded.frame))
    }

    /// Decodes one frame, shown or not, handing `inspect` what its tiles are decoded with and
    /// the tile data first. A frame that shows an existing one is not inspected.
    pub fn decode_frame_inspecting(
        &mut self,
        data: &[u8],
        inspect: &mut dyn FnMut(&FrameInput, &[u8]),
    ) -> Result<Decoded, Error> {
        let mut hdr = self.read_uncompressed_header(data)?;
        if hdr.show_existing_frame {
            return self.ref_frame_map[hdr.frame_to_show]
                .clone()
                .map(|frame| Decoded { frame, shown: true })
                .ok_or(Error::Corrupt("showing an empty frame slot"));
        }
        let compressed_start = hdr.uncompressed_header_size;
        let compressed_end = compressed_start + hdr.compressed_header_size;
        if compressed_end > data.len() {
            return Err(Error::Truncated);
        }

        let (mi_cols, mi_rows) = (hdr.mi_cols(), hdr.mi_rows());
        if self.last_seg_map.len() != mi_cols * mi_rows
            || self.last_width != hdr.width
            || self.last_height != hdr.height
        {
            self.last_seg_map = vec![0; mi_cols * mi_rows];
            self.current_seg_map = vec![0; mi_cols * mi_rows];
        }

        let mut refs: [Option<Arc<Frame>>; 3] = Default::default();
        if !hdr.is_intra() {
            for i in 0..3 {
                let frame = self.ref_frame_map[hdr.ref_frame_idx[i]]
                    .clone()
                    .ok_or(Error::Corrupt("missing reference"))?;
                if frame.width != hdr.width || frame.height != hdr.height {
                    return Err(Error::Unsupported("scaled reference frames"));
                }
                refs[i] = Some(frame);
            }
        }

        let use_prev_frame_mvs = !hdr.error_resilient
            && hdr.width == self.last_width
            && hdr.height == self.last_height
            && !self.last_intra_only
            && self.last_show_frame
            && !self.last_frame_key;

        self.fc = self.frame_contexts[hdr.frame_context_idx];
        self.read_compressed_header(&mut hdr, &data[compressed_start..compressed_end])?;

        let input = FrameInput {
            hdr: hdr.clone(),
            fc: self.fc,
            seg: self.seg.clone(),
            lf: self.lf.clone(),
            refs,
            prev_mvs: self.prev_frame.clone(),
            use_prev_frame_mvs,
            last_seg_map: std::mem::take(&mut self.last_seg_map),
            cur_seg_map: std::mem::take(&mut self.current_seg_map),
            count: !hdr.error_resilient && !hdr.frame_parallel,
        };
        inspect(&input, &data[compressed_end..]);
        let mut output = decode_pixels(input, &data[compressed_end..]);
        self.last_seg_map = std::mem::take(&mut output.last_seg_map);
        self.current_seg_map = std::mem::take(&mut output.cur_seg_map);
        output.result?;
        let (cur, counts) = (output.frame, output.counts);

        if !hdr.error_resilient && !hdr.frame_parallel {
            let counts = counts.unwrap();
            let pre = self.frame_contexts[hdr.frame_context_idx];
            self.fc
                .adapt_coef(&pre, &counts, hdr.is_intra(), self.last_frame_key);
            if !hdr.is_intra() {
                self.fc.adapt_modes(
                    &pre,
                    &counts,
                    hdr.interp_filter == super::types::SWITCHABLE,
                    hdr.tx_mode == super::types::TX_MODE_SELECT,
                );
                self.fc.adapt_mv(&pre, &counts, hdr.allow_hp);
            }
        }
        if hdr.refresh_frame_context {
            self.frame_contexts[hdr.frame_context_idx] = self.fc;
        }

        let frame = Arc::new(cur);
        for i in 0..8 {
            if hdr.refresh_frame_flags >> i & 1 != 0 {
                self.ref_frame_map[i] = Some(frame.clone());
            }
        }
        self.last_show_frame = hdr.show_frame;
        self.prev_frame = Some(frame.clone());
        if self.seg.enabled {
            std::mem::swap(&mut self.last_seg_map, &mut self.current_seg_map);
        }
        self.last_width = hdr.width;
        self.last_height = hdr.height;
        Ok(Decoded {
            frame,
            shown: hdr.show_frame,
        })
    }
}

/// A decoded frame and whether the stream shows it.
pub struct Decoded {
    pub frame: Arc<Frame>,
    pub shown: bool,
}

/// What decoding a frame's tiles needs, apart from the tile data.
pub struct FrameInput {
    pub hdr: FrameHeader,
    pub fc: FrameContext,
    pub seg: Segmentation,
    pub lf: LoopFilterParams,
    pub refs: [Option<Arc<Frame>>; 3],
    pub prev_mvs: Option<Arc<Frame>>,
    pub use_prev_frame_mvs: bool,
    pub last_seg_map: Vec<u8>,
    pub cur_seg_map: Vec<u8>,
    /// Gather the symbol counts backward adaptation needs.
    pub count: bool,
}

pub struct FrameOutput {
    pub frame: Frame,
    pub counts: Option<Counts>,
    pub last_seg_map: Vec<u8>,
    pub cur_seg_map: Vec<u8>,
    pub result: Result<(), Error>,
}

/// Decodes the tiles of one frame into pixels and applies the loop filter.
pub fn decode_pixels(input: FrameInput, tiles: &[u8]) -> FrameOutput {
    let hdr = input.hdr;
    let (mi_cols, mi_rows) = (hdr.mi_cols(), hdr.mi_rows());
    let mut dequant = [[[0i32; 2]; 2]; 8];
    for (i, entry) in dequant.iter_mut().enumerate() {
        let q = if input.seg.enabled {
            input.seg.qindex(i, hdr.base_qindex)
        } else {
            hdr.base_qindex
        };
        *entry = [
            [dc_quant(q, hdr.y_dc_delta_q), ac_quant(q, 0)],
            [
                dc_quant(q, hdr.uv_dc_delta_q),
                ac_quant(q, hdr.uv_ac_delta_q),
            ],
        ];
    }
    let sb_cols = mi_cols.div_ceil(8);
    let aligned_cols = sb_cols * 8;
    let lf_levels = loopfilter::frame_init(&input.lf, &input.seg);
    let mut st = FrameState {
        fc: input.fc,
        counts: input.count.then(Counts::default),
        seg: input.seg,
        lf: input.lf,
        cur: Frame::new(hdr.width, hdr.height),
        refs: input.refs,
        prev_mvs: input.prev_mvs,
        mi_cols,
        mi_rows,
        mi_pool: vec![ModeInfo::default()],
        mi_grid: vec![0; mi_cols * mi_rows],
        above_ctx: [
            vec![0; aligned_cols * 2 + 16],
            vec![0; aligned_cols * 2 + 16],
            vec![0; aligned_cols * 2 + 16],
        ],
        above_seg_ctx: vec![0; aligned_cols + 8],
        last_seg_map: input.last_seg_map,
        cur_seg_map: input.cur_seg_map,
        dequant,
        lfm: vec![LoopFilterMask::default(); sb_cols * mi_rows.div_ceil(8)],
        lf_levels,
        use_prev_frame_mvs: input.use_prev_frame_mvs,
        hdr,
    };
    let result = decode_tiles(&mut st, tiles);
    if result.is_ok() && st.lf.level != 0 {
        let lf = st.lf.clone();
        loopfilter::filter_frame(&mut st, &lf);
    }
    FrameOutput {
        frame: st.cur,
        counts: st.counts,
        last_seg_map: st.last_seg_map,
        cur_seg_map: st.cur_seg_map,
        result,
    }
}

fn tile_offset(index: usize, mis: usize, log2: u32) -> usize {
    let sbs = mis.div_ceil(8);
    (((index * sbs) >> log2) << 3).min(mis)
}

fn decode_tiles(st: &mut FrameState, data: &[u8]) -> Result<(), Error> {
    let tile_rows = 1usize << st.hdr.log2_tile_rows;
    let tile_cols = 1usize << st.hdr.log2_tile_cols;
    let mut at = 0usize;
    for tile_row in 0..tile_rows {
        for tile_col in 0..tile_cols {
            let last = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
            let size = if last {
                data.len().checked_sub(at).ok_or(Error::Truncated)?
            } else {
                let bytes = data.get(at..at + 4).ok_or(Error::Truncated)?;
                at += 4;
                u32::from_be_bytes(bytes.try_into().unwrap()) as usize
            };
            let tile = data.get(at..at + size).ok_or(Error::Truncated)?;
            at += size;

            let mi_row_start = tile_offset(tile_row, st.mi_rows, st.hdr.log2_tile_rows);
            let mi_row_end = tile_offset(tile_row + 1, st.mi_rows, st.hdr.log2_tile_rows);
            let mi_col_start = tile_offset(tile_col, st.mi_cols, st.hdr.log2_tile_cols);
            let mi_col_end = tile_offset(tile_col + 1, st.mi_cols, st.hdr.log2_tile_cols);
            let mut decoder = TileDecoder {
                r: BoolDecoder::new(tile)?,
                st,
                mi_col_start,
                mi_col_end,
                left_ctx: [[0; 16]; 3],
                left_seg_ctx: [0; 8],
                dqcoeff: Box::new([0; 1024]),
            };
            for mi_row in (mi_row_start..mi_row_end).step_by(8) {
                decoder.left_ctx = [[0; 16]; 3];
                decoder.left_seg_ctx = [0; 8];
                for mi_col in (mi_col_start..mi_col_end).step_by(8) {
                    decoder.decode_partition(mi_row, mi_col, super::types::BLOCK_64X64, 4);
                }
            }
        }
    }
    Ok(())
}
