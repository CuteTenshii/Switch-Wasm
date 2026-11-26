//! Whole VP9 frames rebuilt from what nvdec is given, for a host decoder that reads its own
//! headers. nvdec gets the tile data alone: the driver has read both headers and applied their
//! probability updates, so each frame's headers are written anew from the picture setup.
//!
//! The rebuilt stream never saves a probability context, so the host decoder's saved contexts
//! stay at their defaults and every frame codes its probabilities as updates from them.

use super::bool_encoder::{BitWriter, BoolEncoder};
use super::header::{
    inv_remap_prob, setup_compound_reference_mode, FrameHeader, LoopFilterParams, Segmentation,
};
use super::nvdec::PicSetup;
use super::probs::FrameContext;
use super::tables::INV_MAP_TABLE;
use super::types::*;

/// Why a frame cannot be rebuilt.
pub type Refusal = &'static str;

/// A rebuilt frame: one chunk for the host decoder, which shows exactly one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub data: Vec<u8>,
    pub key: bool,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    surface: u64,
    used: u64,
}

/// What the host decoder keeps of the last frame it decoded, which decides whether it predicts
/// from that frame's motion vectors.
#[derive(Debug, Clone, Copy)]
struct Last {
    width: u32,
    height: u32,
    shown: bool,
    intra_only: bool,
}

/// One stream as the host decoder sees it: the surface in each of its eight reference slots and
/// the frame it decoded last.
#[derive(Debug, Default)]
pub struct Rebuilder {
    slots: [Option<Slot>; 8],
    clock: u64,
    last: Option<Last>,
}

impl Rebuilder {
    /// The surface in each of the host decoder's reference slots.
    pub fn slots(&self) -> [Option<u64>; 8] {
        self.slots.map(|slot| slot.map(|s| s.surface))
    }

    /// Rebuilds the frame an nvdec `EXECUTE` describes. `setup.seg` must hold the probability
    /// table's segmentation probabilities; `surfaces` are the luma surfaces of the last, golden
    /// and altref references, then of the frame written.
    pub fn frame(
        &mut self,
        setup: &PicSetup,
        fc: &FrameContext,
        tiles: &[u8],
        surfaces: [u64; 4],
    ) -> Result<Chunk, Refusal> {
        let key = setup.is_key_frame();
        let intra_only = !key && setup.header.intra_only;
        if !key && self.last.is_none() {
            return Err("the stream has not reached a key frame");
        }
        let clock = self.clock + 1;
        let mut slots = self.slots;
        let mut hdr = setup.header.clone();
        hdr.key_frame = key;
        hdr.intra_only = intra_only;
        // Only a hidden frame may be intra only; one is shown from its slot afterwards.
        hdr.show_frame = !intra_only;
        hdr.reset_frame_context = 0;
        hdr.refresh_frame_context = false;
        hdr.frame_parallel = true;
        hdr.frame_context_idx = 0;
        hdr.ref_frame_idx = [0; 3];
        if key {
            hdr.refresh_frame_flags = 0xff;
        } else {
            if !intra_only {
                for (i, &surface) in surfaces[..3].iter().enumerate() {
                    let slot = slots
                        .iter()
                        .position(|s| s.is_some_and(|s| s.surface == surface))
                        .ok_or("a reference frame is not in the host decoder")?;
                    hdr.ref_frame_idx[i] = slot;
                    if let Some(s) = &mut slots[slot] {
                        s.used = clock;
                    }
                }
            }
            // The surface's old picture is gone wherever a slot still holds it.
            let stale = (0..8).fold(0u8, |mask, i| {
                mask | u8::from(slots[i].is_some_and(|s| s.surface == surfaces[3])) << i
            });
            hdr.refresh_frame_flags = if stale != 0 {
                stale
            } else {
                let free = (0..8)
                    .filter(|i| intra_only || !hdr.ref_frame_idx.contains(i))
                    .min_by_key(|&i| slots[i].map_or(0, |s| s.used))
                    .expect("three references leave five slots");
                1 << free
            };
        }

        // The host predicts from the frame it decoded last when that frame was shown, which every
        // rebuilt frame but an intra-only one is; a hidden filler between stops it.
        let filler = !hdr.is_intra()
            && self.last.is_some_and(|last| {
                let reused = !hdr.error_resilient
                    && last.shown
                    && !last.intra_only
                    && (last.width, last.height) == (hdr.width, hdr.height);
                let wanted = !hdr.error_resilient
                    && setup.last_show_frame()
                    && setup.last_size == setup.cur_size;
                reused && !wanted
            });

        let frame = write_frame(&hdr, &setup.lf, &setup.seg, fc, tiles)?;
        let data = if intra_only {
            let slot = hdr.refresh_frame_flags.trailing_zeros() as u8;
            superframe(&[&frame, &[show_existing(slot)]])
        } else if filler {
            superframe(&[&filler_frame(&hdr), &frame])
        } else {
            frame
        };

        for (i, slot) in slots.iter_mut().enumerate() {
            if hdr.refresh_frame_flags >> i & 1 != 0 {
                *slot = Some(Slot {
                    surface: surfaces[3],
                    used: clock,
                });
            }
        }
        self.slots = slots;
        self.clock = clock;
        self.last = Some(Last {
            width: hdr.width,
            height: hdr.height,
            shown: hdr.show_frame,
            intra_only,
        });
        Ok(Chunk { data, key })
    }
}

/// The uncompressed header, the compressed header, then the tiles.
fn write_frame(
    hdr: &FrameHeader,
    lf: &LoopFilterParams,
    seg: &Segmentation,
    fc: &FrameContext,
    tiles: &[u8],
) -> Result<Vec<u8>, Refusal> {
    let compressed = compressed_header(hdr, fc)?;
    let mut w = BitWriter::default();
    uncompressed_header(&mut w, hdr, lf, seg)?;
    let size = u16::try_from(compressed.len()).map_err(|_| "the compressed header is too long")?;
    w.write(u32::from(size), 16);
    let mut out = w.into_bytes();
    out.extend_from_slice(&compressed);
    out.extend_from_slice(tiles);
    Ok(out)
}

fn write_sync_code(w: &mut BitWriter) {
    w.write(0x49_83_42, 24);
}

fn uncompressed_header(
    w: &mut BitWriter,
    hdr: &FrameHeader,
    lf: &LoopFilterParams,
    seg: &Segmentation,
) -> Result<(), Refusal> {
    w.write(2, 2);
    // Profile 0, and not a frame that shows an existing one.
    w.write(0, 3);
    w.write_bit(!hdr.key_frame);
    w.write_bit(hdr.show_frame);
    w.write_bit(hdr.error_resilient);
    if hdr.key_frame {
        write_sync_code(w);
        // Colour space unknown, studio range: neither changes the decoded samples.
        w.write(0, 3);
        w.write_bit(false);
        frame_size(w, hdr)?;
    } else {
        if !hdr.show_frame {
            w.write_bit(hdr.intra_only);
        }
        if !hdr.error_resilient {
            w.write(u32::from(hdr.reset_frame_context), 2);
        }
        if hdr.intra_only {
            write_sync_code(w);
            w.write(u32::from(hdr.refresh_frame_flags), 8);
            frame_size(w, hdr)?;
        } else {
            w.write(u32::from(hdr.refresh_frame_flags), 8);
            for i in 0..3 {
                w.write(hdr.ref_frame_idx[i] as u32, 3);
                w.write_bit(hdr.sign_bias[1 + i]);
            }
            // The size follows in full rather than from a reference.
            w.write(0, 3);
            frame_size(w, hdr)?;
            w.write_bit(hdr.allow_hp);
            if hdr.interp_filter == SWITCHABLE {
                w.write_bit(true);
            } else {
                let literal = [EIGHTTAP_SMOOTH, EIGHTTAP, EIGHTTAP_SHARP, BILINEAR]
                    .iter()
                    .position(|&f| f == hdr.interp_filter)
                    .ok_or("an unknown interpolation filter")?;
                w.write_bit(false);
                w.write(literal as u32, 2);
            }
        }
    }
    if !hdr.error_resilient {
        w.write_bit(hdr.refresh_frame_context);
        w.write_bit(hdr.frame_parallel);
    }
    w.write(hdr.frame_context_idx as u32, 2);
    loop_filter(w, lf)?;
    quantizer(w, hdr)?;
    segmentation(w, seg)?;
    tile_info(w, hdr)
}

/// The frame size, and a render size equal to it.
fn frame_size(w: &mut BitWriter, hdr: &FrameHeader) -> Result<(), Refusal> {
    let range = 1..=1 << 16;
    if !range.contains(&hdr.width) || !range.contains(&hdr.height) {
        return Err("a frame size VP9 cannot code");
    }
    w.write(hdr.width - 1, 16);
    w.write(hdr.height - 1, 16);
    w.write_bit(false);
    Ok(())
}

fn loop_filter(w: &mut BitWriter, lf: &LoopFilterParams) -> Result<(), Refusal> {
    if lf.level > 63 || lf.sharpness > 7 {
        return Err("a loop filter level or sharpness out of range");
    }
    w.write(u32::from(lf.level), 6);
    w.write(u32::from(lf.sharpness), 3);
    w.write_bit(lf.delta_enabled);
    if lf.delta_enabled {
        // Every delta is sent, so none depends on what an earlier frame left behind.
        w.write_bit(true);
        for &delta in lf.ref_deltas.iter().chain(&lf.mode_deltas) {
            if !(-63..=63).contains(&delta) {
                return Err("a loop filter delta out of range");
            }
            w.write_bit(true);
            w.write_signed(i32::from(delta), 6);
        }
    }
    Ok(())
}

fn quantizer(w: &mut BitWriter, hdr: &FrameHeader) -> Result<(), Refusal> {
    let q = u8::try_from(hdr.base_qindex).map_err(|_| "a quantizer index out of range")?;
    w.write(u32::from(q), 8);
    let deltas = [hdr.y_dc_delta_q, hdr.uv_dc_delta_q, hdr.uv_ac_delta_q];
    for delta in deltas {
        if !(-15..=15).contains(&delta) {
            return Err("a quantizer delta out of range");
        }
        w.write_bit(delta != 0);
        if delta != 0 {
            w.write_signed(delta, 4);
        }
    }
    // The host derives lossless coding from the quantizer, nvdec reads a flag.
    if hdr.lossless != (q == 0 && deltas == [0; 3]) {
        return Err("the setup's lossless flag disagrees with its quantizer");
    }
    Ok(())
}

fn segmentation(w: &mut BitWriter, seg: &Segmentation) -> Result<(), Refusal> {
    const BITS: [u32; 4] = [8, 6, 2, 0];
    const MAX: [u32; 4] = [255, 63, 3, 0];
    const SIGNED: [bool; 4] = [true, true, false, false];
    let optional_prob = |w: &mut BitWriter, prob: u8| {
        w.write_bit(prob != 255);
        if prob != 255 {
            w.write(u32::from(prob), 8);
        }
    };
    w.write_bit(seg.enabled);
    if !seg.enabled {
        return Ok(());
    }
    w.write_bit(seg.update_map);
    if seg.update_map {
        for &prob in &seg.tree_probs {
            optional_prob(w, prob);
        }
        w.write_bit(seg.temporal_update);
        if seg.temporal_update {
            for &prob in &seg.pred_probs {
                optional_prob(w, prob);
            }
        }
    }
    // The features are always sent, for the same reason as the loop filter deltas.
    w.write_bit(true);
    w.write_bit(seg.abs_delta);
    for segment in &seg.features {
        for (f, feature) in segment.iter().enumerate() {
            w.write_bit(feature.enabled);
            if !feature.enabled || BITS[f] == 0 {
                continue;
            }
            let data = i32::from(feature.data);
            if data.unsigned_abs() > MAX[f] || (!SIGNED[f] && data < 0) {
                return Err("a segmentation feature value out of range");
            }
            w.write(data.unsigned_abs(), BITS[f]);
            if SIGNED[f] {
                w.write_bit(data < 0);
            }
        }
    }
    Ok(())
}

/// The fewest and most tile columns, as log2, a frame of `sb_cols` superblocks may have.
fn tile_cols_log2_range(sb_cols: usize) -> (u32, u32) {
    let mut min = 0;
    while (64usize << min) < sb_cols {
        min += 1;
    }
    let mut max = 1;
    while (sb_cols >> max) >= 4 {
        max += 1;
    }
    (min, max - 1)
}

fn tile_info(w: &mut BitWriter, hdr: &FrameHeader) -> Result<(), Refusal> {
    let (min, max) = tile_cols_log2_range(hdr.mi_cols().div_ceil(8));
    if hdr.log2_tile_cols < min || hdr.log2_tile_cols > max.max(min) {
        return Err("a tile column count the frame width does not allow");
    }
    for _ in min..hdr.log2_tile_cols {
        w.write_bit(true);
    }
    if hdr.log2_tile_cols < max {
        w.write_bit(false);
    }
    match hdr.log2_tile_rows {
        0 => w.write_bit(false),
        1 | 2 => {
            w.write_bit(true);
            w.write_bit(hdr.log2_tile_rows == 2);
        }
        _ => return Err("more than four tile rows"),
    }
    Ok(())
}

fn compressed_header(hdr: &FrameHeader, fc: &FrameContext) -> Result<Vec<u8>, Refusal> {
    let base = FrameContext::defaults();
    let mut e = BoolEncoder::new();
    if hdr.tx_mode > TX_MODE_SELECT {
        return Err("an unknown transform mode");
    }
    if hdr.lossless {
        if hdr.tx_mode != ONLY_4X4 {
            return Err("a lossless frame with transforms larger than 4x4");
        }
    } else {
        e.write_literal(u32::from(hdr.tx_mode.min(ALLOW_32X32)), 2);
        if hdr.tx_mode >= ALLOW_32X32 {
            e.write_bit(hdr.tx_mode == TX_MODE_SELECT);
        }
    }
    if hdr.tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            update(&mut e, base.tx8[i][0], fc.tx8[i][0])?;
        }
        for i in 0..2 {
            for j in 0..2 {
                update(&mut e, base.tx16[i][j], fc.tx16[i][j])?;
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                update(&mut e, base.tx32[i][j], fc.tx32[i][j])?;
            }
        }
    }

    for tx in 0..=usize::from(TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(hdr.tx_mode)]) {
        let (from, to) = (&base.coef[tx], &fc.coef[tx]);
        let coded = || {
            (0..2).flat_map(move |i| {
                (0..2).flat_map(move |j| {
                    (0..6).flat_map(move |k| {
                        (0..if k == 0 { 3 } else { 6 })
                            .flat_map(move |l| (0..3).map(move |m| (i, j, k, l, m)))
                    })
                })
            })
        };
        let changed = coded().any(|(i, j, k, l, m)| from[i][j][k][l][m] != to[i][j][k][l][m]);
        e.write_bit(changed);
        if changed {
            for (i, j, k, l, m) in coded() {
                update(&mut e, from[i][j][k][l][m], to[i][j][k][l][m])?;
            }
        }
    }

    for i in 0..3 {
        update(&mut e, base.skip[i], fc.skip[i])?;
    }
    if hdr.is_intra() {
        return Ok(e.finish());
    }

    if fc.uv_mode != base.uv_mode {
        return Err(
            "chroma mode probabilities off their defaults, which only backward adaptation moves",
        );
    }
    for i in 0..7 {
        for j in 0..3 {
            update(&mut e, base.inter_mode[i][j], fc.inter_mode[i][j])?;
        }
    }
    if hdr.interp_filter == SWITCHABLE {
        for i in 0..4 {
            for j in 0..2 {
                update(
                    &mut e,
                    base.switchable_interp[i][j],
                    fc.switchable_interp[i][j],
                )?;
            }
        }
    }
    for i in 0..4 {
        update(&mut e, base.intra_inter[i], fc.intra_inter[i])?;
    }

    if hdr.reference_mode > REFERENCE_MODE_SELECT {
        return Err("an unknown reference mode");
    }
    let compound_allowed = (1..3).any(|i| hdr.sign_bias[i + 1] != hdr.sign_bias[1]);
    if compound_allowed {
        e.write_bit(hdr.reference_mode != SINGLE_REFERENCE);
        if hdr.reference_mode != SINGLE_REFERENCE {
            e.write_bit(hdr.reference_mode == REFERENCE_MODE_SELECT);
        }
    } else if hdr.reference_mode != SINGLE_REFERENCE {
        return Err("compound prediction between references of one sign bias");
    }
    if hdr.reference_mode != SINGLE_REFERENCE {
        let mut derived = hdr.clone();
        setup_compound_reference_mode(&mut derived);
        if (derived.comp_fixed_ref, derived.comp_var_ref) != (hdr.comp_fixed_ref, hdr.comp_var_ref)
        {
            return Err("compound references other than the sign biases imply");
        }
    }
    if hdr.reference_mode == REFERENCE_MODE_SELECT {
        for i in 0..5 {
            update(&mut e, base.comp_inter[i], fc.comp_inter[i])?;
        }
    }
    if hdr.reference_mode != COMPOUND_REFERENCE {
        for i in 0..5 {
            for j in 0..2 {
                update(&mut e, base.single_ref[i][j], fc.single_ref[i][j])?;
            }
        }
    }
    if hdr.reference_mode != SINGLE_REFERENCE {
        for i in 0..5 {
            update(&mut e, base.comp_ref[i], fc.comp_ref[i])?;
        }
    }
    for i in 0..4 {
        for j in 0..9 {
            update(&mut e, base.y_mode[i][j], fc.y_mode[i][j])?;
        }
    }
    for i in 0..16 {
        for j in 0..3 {
            update(&mut e, base.partition[i][j], fc.partition[i][j])?;
        }
    }

    let (from, to) = (&base.mv, &fc.mv);
    mv_updates(&mut e, &from.joints, &to.joints)?;
    for c in 0..2 {
        let (from, to) = (&from.comps[c], &to.comps[c]);
        mv_updates(&mut e, &[from.sign], &[to.sign])?;
        mv_updates(&mut e, &from.classes, &to.classes)?;
        mv_updates(&mut e, &from.class0, &to.class0)?;
        mv_updates(&mut e, &from.bits, &to.bits)?;
    }
    for c in 0..2 {
        let (from, to) = (&from.comps[c], &to.comps[c]);
        for j in 0..2 {
            mv_updates(&mut e, &from.class0_fp[j], &to.class0_fp[j])?;
        }
        mv_updates(&mut e, &from.fp, &to.fp)?;
    }
    if hdr.allow_hp {
        for c in 0..2 {
            let (from, to) = (&from.comps[c], &to.comps[c]);
            mv_updates(&mut e, &[from.class0_hp], &[to.class0_hp])?;
            mv_updates(&mut e, &[from.hp], &[to.hp])?;
        }
    }
    Ok(e.finish())
}

/// The index into `INV_MAP_TABLE` of each value, the first where a value repeats.
const MAP_TABLE: [u8; 256] = {
    let mut map = [u8::MAX; 256];
    let mut delta = INV_MAP_TABLE.len();
    while delta > 0 {
        delta -= 1;
        map[INV_MAP_TABLE[delta] as usize] = delta as u8;
    }
    map
};

fn recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > m << 1 {
        v
    } else if v >= m {
        (v - m) << 1
    } else {
        ((m - v) << 1) - 1
    }
}

/// `remap_prob`: the delta that turns `from` into `to`, the inverse of [`inv_remap_prob`].
fn remap_prob(to: u8, from: u8) -> Option<usize> {
    if to == 0 || from == 0 || to == from {
        return None;
    }
    let (v, m) = (i32::from(to) - 1, i32::from(from) - 1);
    let index = if m << 1 <= 255 {
        recenter_nonneg(v, m)
    } else {
        recenter_nonneg(254 - v, 254 - m)
    };
    let delta = usize::from(*MAP_TABLE.get(usize::try_from(index).ok()?)?);
    (inv_remap_prob(delta, from) == to).then_some(delta)
}

/// `vp9_cond_prob_diff_update`, coding the change from `from` to `to`.
fn update(e: &mut BoolEncoder, from: u8, to: u8) -> Result<(), Refusal> {
    if from == to {
        e.write(false, 252);
        return Ok(());
    }
    let delta = remap_prob(to, from).ok_or("a probability no update reaches")? as u32;
    e.write(true, 252);
    if delta < 16 {
        e.write_bit(false);
        e.write_literal(delta, 4);
    } else if delta < 32 {
        e.write_bit(true);
        e.write_bit(false);
        e.write_literal(delta - 16, 4);
    } else if delta < 64 {
        e.write_literal(0b110, 3);
        e.write_literal(delta - 32, 5);
    } else {
        e.write_literal(0b111, 3);
        let uniform = delta - 64;
        if uniform < 65 {
            e.write_literal(uniform, 7);
        } else {
            e.write_literal(65 + ((uniform - 65) >> 1), 7);
            e.write_bit((uniform - 65) & 1 != 0);
        }
    }
    Ok(())
}

/// Motion vector probabilities are sent as seven bits, so only odd values are reachable.
fn mv_updates(e: &mut BoolEncoder, from: &[u8], to: &[u8]) -> Result<(), Refusal> {
    for (&from, &to) in from.iter().zip(to) {
        if from == to {
            e.write(false, 252);
        } else if to & 1 == 0 {
            return Err("an even motion vector probability, which only backward adaptation makes");
        } else {
            e.write(true, 252);
            e.write_literal(u32::from(to >> 1), 7);
        }
    }
    Ok(())
}

/// Tile data of all-zero bools for a frame: every superblock one intra block with DC
/// prediction and no coefficients, which any decoder accepts.
fn zero_tiles(hdr: &FrameHeader) -> Vec<u8> {
    /// Far more than such a superblock reads at 32x32 transforms.
    const BYTES_PER_SUPERBLOCK: usize = 16;
    let (cols, rows) = (1usize << hdr.log2_tile_cols, 1usize << hdr.log2_tile_rows);
    let sb_cols = hdr.mi_cols().div_ceil(8);
    let sb_rows = hdr.mi_rows().div_ceil(8);
    let tile_size = BYTES_PER_SUPERBLOCK * sb_cols.div_ceil(cols) * sb_rows.div_ceil(rows) + 16;
    let mut tiles = Vec::with_capacity((tile_size + 4) * cols * rows);
    for i in 0..cols * rows {
        if i + 1 < cols * rows {
            tiles.extend_from_slice(&(tile_size as u32).to_be_bytes());
        }
        tiles.resize(tiles.len() + tile_size, 0);
    }
    tiles
}

/// A hidden inter frame that refreshes no slot, decoded only so the next frame does not predict
/// from the motion vectors of the frame before it.
fn filler_frame(next: &FrameHeader) -> Vec<u8> {
    let mut hdr = FrameHeader {
        width: next.width,
        height: next.height,
        ref_frame_idx: next.ref_frame_idx,
        frame_parallel: true,
        base_qindex: 1,
        tx_mode: ALLOW_32X32,
        interp_filter: EIGHTTAP,
        reference_mode: SINGLE_REFERENCE,
        ..Default::default()
    };
    hdr.log2_tile_cols = tile_cols_log2_range(hdr.mi_cols().div_ceil(8)).0;
    let lf = LoopFilterParams {
        delta_enabled: false,
        ..Default::default()
    };
    write_frame(
        &hdr,
        &lf,
        &Segmentation::default(),
        &FrameContext::defaults(),
        &zero_tiles(&hdr),
    )
    .expect("a filler frame is always expressible")
}

/// The one-byte header of a frame that shows `slot` again.
fn show_existing(slot: u8) -> u8 {
    0b1000_1000 | slot
}

/// Frames packed into one chunk behind a superframe index of four-byte sizes.
fn superframe(frames: &[&[u8]]) -> Vec<u8> {
    let marker = 0xc0 | 3 << 3 | (frames.len() as u8 - 1);
    let mut out = frames.concat();
    out.push(marker);
    for frame in frames {
        out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    }
    out.push(marker);
    out
}

#[cfg(test)]
mod tests {
    use super::super::bool_decoder::BoolDecoder;
    use super::super::decoder::{Decoder, FrameInput};
    use super::super::header::{diff_update, SegFeature};
    use super::super::nvdec::Dimensions;
    use super::*;

    const WIDTH: u32 = 136;
    const HEIGHT: u32 = 72;

    fn dimensions(width: u32, height: u32) -> Dimensions {
        Dimensions {
            width: width as u16,
            height: height as u16,
            luma_pitch: width as u16,
            chroma_pitch: width as u16 / 2,
        }
    }

    /// A setup as the driver fills it; `flags` as in the setup's word at `+0x68`.
    fn setup(flags: u32) -> PicSetup {
        let size = dimensions(WIDTH, HEIGHT);
        PicSetup {
            bitstream_size: 0,
            last_size: size,
            golden_size: size,
            alt_size: size,
            cur_size: size,
            flags,
            header: FrameHeader {
                key_frame: flags & 1 != 0,
                error_resilient: flags >> 3 & 1 != 0,
                frame_parallel: flags >> 2 & 1 != 0,
                intra_only: flags >> 5 & 1 != 0,
                width: WIDTH,
                height: HEIGHT,
                base_qindex: 60,
                tx_mode: ALLOW_32X32,
                interp_filter: EIGHTTAP,
                ..Default::default()
            },
            seg: Segmentation::default(),
            lf: LoopFilterParams::default(),
        }
    }

    const KEY: u32 = 1;
    const LAST_SHOWN: u32 = 1 << 4;
    const INTRA_ONLY: u32 = 1 << 5;

    fn tiles(setup: &PicSetup) -> Vec<u8> {
        zero_tiles(&setup.header)
    }

    fn rebuild(
        rebuilder: &mut Rebuilder,
        setup: &PicSetup,
        fc: &FrameContext,
        surfaces: [u64; 4],
    ) -> Result<Chunk, Refusal> {
        rebuilder.frame(setup, fc, &tiles(setup), surfaces)
    }

    /// What a decoder made of each frame in `chunk`, and the frames it showed.
    fn decode(decoder: &mut Decoder, chunk: &Chunk) -> (Vec<FrameInput>, usize) {
        let mut inputs = Vec::new();
        let mut shown = 0;
        for frame in super::super::decoder::split_superframe(&chunk.data) {
            let decoded = decoder
                .decode_frame_inspecting(frame, &mut |input, _| {
                    inputs.push(FrameInput {
                        hdr: input.hdr.clone(),
                        fc: input.fc,
                        seg: input.seg.clone(),
                        lf: input.lf.clone(),
                        refs: Default::default(),
                        prev_mvs: None,
                        use_prev_frame_mvs: input.use_prev_frame_mvs,
                        last_seg_map: Vec::new(),
                        cur_seg_map: Vec::new(),
                        count: input.count,
                    })
                })
                .expect("the rebuilt frame decodes");
            shown += usize::from(decoded.shown);
        }
        (inputs, shown)
    }

    #[test]
    fn every_probability_change_reads_back() {
        let mut e = BoolEncoder::new();
        for from in 1..=255u8 {
            for to in 1..=255u8 {
                update(&mut e, from, to).unwrap();
            }
        }
        let bytes = e.finish();
        let mut r = BoolDecoder::new(&bytes).unwrap();
        for from in 1..=255u8 {
            for to in 1..=255u8 {
                let mut prob = from;
                diff_update(&mut r, &mut prob);
                assert_eq!(prob, to, "from {from}");
            }
        }
        assert!(!r.overran());
    }

    #[test]
    fn a_key_frame_reads_back_with_its_setup_and_probabilities() {
        let mut key = setup(KEY);
        key.header.tx_mode = TX_MODE_SELECT;
        key.header.y_dc_delta_q = -3;
        key.header.uv_ac_delta_q = 15;
        key.header.log2_tile_rows = 1;
        key.lf = LoopFilterParams {
            level: 33,
            sharpness: 5,
            delta_enabled: true,
            ref_deltas: [2, 0, -63, 9],
            mode_deltas: [-1, 4],
        };
        key.seg = Segmentation {
            enabled: true,
            update_map: true,
            temporal_update: false,
            tree_probs: [12, 255, 200, 1, 255, 77, 254],
            ..Default::default()
        };
        key.seg.features[3][0] = SegFeature {
            enabled: true,
            data: -40,
        };
        key.seg.features[7][2] = SegFeature {
            enabled: true,
            data: 3,
        };
        key.seg.features[1][3] = SegFeature {
            enabled: true,
            data: 0,
        };
        let mut fc = FrameContext::defaults();
        fc.tx16[1][0] = 3;
        fc.skip[2] = 250;
        fc.coef[2][1][0][3][4][1] = 1;
        fc.coef[0][0][1][0][2][0] = 255;

        let chunk = rebuild(&mut Rebuilder::default(), &key, &fc, [0, 0, 0, 0xa0]).unwrap();
        assert!(chunk.key);
        let (inputs, shown) = decode(&mut Decoder::new(), &chunk);
        assert_eq!((inputs.len(), shown), (1, 1));
        let input = &inputs[0];
        assert!(input.hdr.key_frame && !input.hdr.refresh_frame_context);
        assert_eq!((input.hdr.width, input.hdr.height), (WIDTH, HEIGHT));
        assert_eq!(
            (
                input.hdr.base_qindex,
                input.hdr.y_dc_delta_q,
                input.hdr.uv_dc_delta_q,
                input.hdr.uv_ac_delta_q
            ),
            (60, -3, 0, 15)
        );
        assert_eq!(
            (input.hdr.tx_mode, input.hdr.log2_tile_rows),
            (TX_MODE_SELECT, 1)
        );
        assert_eq!(input.lf, key.lf);
        assert_eq!(input.seg.features, key.seg.features);
        assert_eq!(input.seg.tree_probs, key.seg.tree_probs);
        assert_eq!(input.fc, fc);
    }

    #[test]
    fn an_inter_frame_names_its_references_by_slot_and_codes_its_probabilities() {
        let mut rebuilder = Rebuilder::default();
        let mut decoder = Decoder::new();
        let fc = FrameContext::defaults();
        let key = rebuild(&mut rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap();
        decode(&mut decoder, &key);
        let golden = rebuild(
            &mut rebuilder,
            &setup(LAST_SHOWN),
            &fc,
            [0xa0, 0xa0, 0xa0, 0xb0],
        );
        let golden = golden.unwrap();
        decode(&mut decoder, &golden);

        let mut inter = setup(LAST_SHOWN);
        inter.header.sign_bias = [false, false, false, true];
        inter.header.reference_mode = REFERENCE_MODE_SELECT;
        inter.header.comp_fixed_ref = ALTREF_FRAME;
        inter.header.comp_var_ref = [LAST_FRAME, GOLDEN_FRAME];
        inter.header.interp_filter = SWITCHABLE;
        inter.header.allow_hp = true;
        let mut fc = FrameContext::defaults();
        fc.comp_inter[4] = 99;
        fc.single_ref[2][1] = 1;
        fc.comp_ref[0] = 200;
        fc.switchable_interp[3][1] = 7;
        fc.partition[15][2] = 44;
        fc.y_mode[2][8] = 250;
        fc.mv.joints[1] = 65;
        fc.mv.comps[1].hp = 255;
        fc.mv.comps[0].class0_fp[1][0] = 3;

        let chunk = rebuild(&mut rebuilder, &inter, &fc, [0xb0, 0xa0, 0xa0, 0xc0]).unwrap();
        assert!(!chunk.key);
        let (inputs, shown) = decode(&mut decoder, &chunk);
        assert_eq!(
            (inputs.len(), shown),
            (1, 1),
            "no filler: the last frame was shown"
        );
        let hdr = &inputs[0].hdr;
        assert_eq!(hdr.sign_bias, [false, false, false, true]);
        assert_eq!(hdr.reference_mode, REFERENCE_MODE_SELECT);
        assert_eq!((hdr.interp_filter, hdr.allow_hp), (SWITCHABLE, true));
        assert_eq!(inputs[0].fc, fc);
        assert!(inputs[0].use_prev_frame_mvs);

        let slot_of = |surface| {
            rebuilder
                .slots
                .iter()
                .position(|s| s.is_some_and(|s| s.surface == surface))
        };
        assert_eq!(hdr.ref_frame_idx[0], slot_of(0xb0).unwrap());
        assert_eq!(hdr.ref_frame_idx[1], slot_of(0xa0).unwrap());
        assert_eq!(hdr.ref_frame_idx[2], slot_of(0xa0).unwrap());
        assert!(slot_of(0xc0).is_some());
    }

    #[test]
    fn a_frame_after_a_hidden_one_is_kept_from_its_motion_vectors_by_a_filler() {
        let mut rebuilder = Rebuilder::default();
        let mut decoder = Decoder::new();
        let fc = FrameContext::defaults();
        decode(
            &mut decoder,
            &rebuild(&mut rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap(),
        );
        // The driver decodes this one hidden; the rebuilt stream shows it.
        let hidden = rebuild(
            &mut rebuilder,
            &setup(LAST_SHOWN),
            &fc,
            [0xa0, 0xa0, 0xa0, 0xb0],
        )
        .unwrap();
        decode(&mut decoder, &hidden);

        let chunk = rebuild(&mut rebuilder, &setup(0), &fc, [0xa0, 0xa0, 0xb0, 0xc0]).unwrap();
        let (inputs, shown) = decode(&mut decoder, &chunk);
        assert_eq!((inputs.len(), shown), (2, 1));
        assert!(!inputs[0].hdr.show_frame && inputs[0].hdr.refresh_frame_flags == 0);
        assert!(!inputs[1].use_prev_frame_mvs);

        let next = rebuild(
            &mut rebuilder,
            &setup(LAST_SHOWN),
            &fc,
            [0xc0, 0xa0, 0xb0, 0xd0],
        )
        .unwrap();
        let (inputs, _) = decode(&mut decoder, &next);
        assert_eq!(inputs.len(), 1);
        assert!(inputs[0].use_prev_frame_mvs);
    }

    #[test]
    fn an_intra_only_frame_is_decoded_hidden_and_shown_from_its_slot() {
        let mut rebuilder = Rebuilder::default();
        let mut decoder = Decoder::new();
        let fc = FrameContext::defaults();
        decode(
            &mut decoder,
            &rebuild(&mut rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap(),
        );
        let chunk = rebuild(&mut rebuilder, &setup(INTRA_ONLY), &fc, [0, 0, 0, 0xb0]).unwrap();
        let frames = super::super::decoder::split_superframe(&chunk.data);
        assert_eq!(frames.len(), 2);
        let slot = rebuilder
            .slots
            .iter()
            .position(|s| s.is_some_and(|s| s.surface == 0xb0))
            .unwrap();
        assert_eq!(frames[1], [0x88 | slot as u8]);
        let (inputs, shown) = decode(&mut decoder, &chunk);
        assert_eq!((inputs.len(), shown), (1, 1));
        assert!(inputs[0].hdr.intra_only);
    }

    #[test]
    fn a_stream_starts_at_a_key_frame_and_needs_its_references_there() {
        let mut rebuilder = Rebuilder::default();
        let fc = FrameContext::defaults();
        assert!(rebuild(&mut rebuilder, &setup(LAST_SHOWN), &fc, [1, 1, 1, 2]).is_err());
        rebuild(&mut rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap();
        let unknown = [0xa0, 0xa0, 0xf0, 0xb0];
        assert!(rebuild(&mut rebuilder, &setup(LAST_SHOWN), &fc, unknown).is_err());
    }

    #[test]
    fn surfaces_in_use_keep_their_slots_as_others_rotate_through() {
        let mut rebuilder = Rebuilder::default();
        let mut decoder = Decoder::new();
        let fc = FrameContext::defaults();
        let key = rebuild(&mut rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap();
        decode(&mut decoder, &key);
        // Twelve surfaces in turn, the key frame staying golden throughout.
        let mut last = 0xa0;
        for i in 0..40u32 {
            let surface = 0x100 + u64::from(i % 12) * 0x10;
            let chunk = rebuild(
                &mut rebuilder,
                &setup(LAST_SHOWN),
                &fc,
                [last, 0xa0, last, surface],
            )
            .unwrap_or_else(|why| panic!("frame {i}: {why}"));
            decode(&mut decoder, &chunk);
            last = surface;
        }
    }

    #[test]
    fn probabilities_only_backward_adaptation_makes_are_refused() {
        let fc = FrameContext::defaults();
        let start = |rebuilder: &mut Rebuilder| {
            rebuild(rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap();
        };
        let mut uv = FrameContext::defaults();
        uv.uv_mode[4][2] += 1;
        let mut rebuilder = Rebuilder::default();
        start(&mut rebuilder);
        assert!(rebuild(
            &mut rebuilder,
            &setup(LAST_SHOWN),
            &uv,
            [0xa0, 0xa0, 0xa0, 0xb0]
        )
        .is_err());

        let mut even = FrameContext::defaults();
        even.mv.comps[0].bits[3] = 100;
        let mut rebuilder = Rebuilder::default();
        start(&mut rebuilder);
        assert!(rebuild(
            &mut rebuilder,
            &setup(LAST_SHOWN),
            &even,
            [0xa0, 0xa0, 0xa0, 0xb0]
        )
        .is_err());
        // A refused frame leaves the stream as it was.
        assert!(rebuild(
            &mut rebuilder,
            &setup(LAST_SHOWN),
            &fc,
            [0xa0, 0xa0, 0xa0, 0xb0]
        )
        .is_ok());
    }

    #[test]
    fn a_filler_frame_decodes_to_nothing_the_stream_keeps() {
        let mut decoder = Decoder::new();
        let fc = FrameContext::defaults();
        let mut rebuilder = Rebuilder::default();
        decode(
            &mut decoder,
            &rebuild(&mut rebuilder, &setup(KEY), &fc, [0, 0, 0, 0xa0]).unwrap(),
        );
        let filler = filler_frame(&setup(0).header);
        let decoded = decoder
            .decode_frame_inspecting(&filler, &mut |_, _| {})
            .unwrap();
        assert!(!decoded.shown);
    }
}
