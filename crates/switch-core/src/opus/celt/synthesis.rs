//! Postfilter, de-emphasis and synthesis.

use crate::opus::mdct::Mdct;
use crate::opus::tables_celt::*;
use super::bands::denormalise_bands;
use super::{COMBFILTER_MINPERIOD, MAX_LM, NB_EBANDS, OVERLAP, PREEMPH, SHORT_MDCT_SIZE, SIG_SCALE};

/// Pitch postfilter, cross-fading from `t0`/`g0` to `t1`/`g1` over the overlap.
#[allow(clippy::too_many_arguments)]
pub(super) fn comb_filter(
    buf: &mut [f32],
    at: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: Option<&[f32]>,
    overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        return;
    }
    // A zero gain leaves the period unset; zero would read before the buffer.
    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);
    let g = [
        [
            g0 * COMB_GAINS[tapset0][0],
            g0 * COMB_GAINS[tapset0][1],
            g0 * COMB_GAINS[tapset0][2],
        ],
        [
            g1 * COMB_GAINS[tapset1][0],
            g1 * COMB_GAINS[tapset1][1],
            g1 * COMB_GAINS[tapset1][2],
        ],
    ];
    let overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap
    };

    let mut i = 0usize;
    if let Some(window) = window {
        while i < overlap {
            let f = window[i] * window[i];
            let old = g[0][0] * buf[at + i - t0]
                + g[0][1] * (buf[at + i - t0 + 1] + buf[at + i - t0 - 1])
                + g[0][2] * (buf[at + i - t0 + 2] + buf[at + i - t0 - 2]);
            let new = g[1][0] * buf[at + i - t1]
                + g[1][1] * (buf[at + i - t1 + 1] + buf[at + i - t1 - 1])
                + g[1][2] * (buf[at + i - t1 + 2] + buf[at + i - t1 - 2]);
            buf[at + i] += (1.0 - f) * old + f * new;
            i += 1;
        }
    }
    if g1 == 0.0 {
        return;
    }
    while i < n {
        buf[at + i] += g[1][0] * buf[at + i - t1]
            + g[1][1] * (buf[at + i - t1 + 1] + buf[at + i - t1 - 1])
            + g[1][2] * (buf[at + i - t1 + 2] + buf[at + i - t1 - 2]);
        i += 1;
    }
}

/// The filter without cross-fade, buffer to buffer, for concealment.
pub(super) fn comb_filter_const(
    out: &mut [f32],
    src: &[f32],
    at: usize,
    t: usize,
    n: usize,
    g: f32,
    tapset: usize,
) {
    if g == 0.0 {
        out[..n].copy_from_slice(&src[at..at + n]);
        return;
    }
    let t = t.max(COMBFILTER_MINPERIOD);
    let (g0, g1, g2) = (
        g * COMB_GAINS[tapset][0],
        g * COMB_GAINS[tapset][1],
        g * COMB_GAINS[tapset][2],
    );
    for i in 0..n {
        out[i] = src[at + i]
            + g0 * src[at + i - t]
            + g1 * (src[at + i - t + 1] + src[at + i - t - 1])
            + g2 * (src[at + i - t + 2] + src[at + i - t - 2]);
    }
}

/// Undo pre-emphasis and output PCM; stateful across frames.
pub(super) fn deemphasis(
    channels: &[Vec<f32>],
    at: usize,
    pcm: &mut [f32],
    n: usize,
    cc: usize,
    downsample: usize,
    mem: &mut [f32; 2],
) {
    let nd = n / downsample;
    let mut scratch = vec![0.0f32; n];
    for c in 0..cc {
        let mut m = mem[c];
        let x = &channels[c][at..at + n];
        if downsample > 1 {
            for j in 0..n {
                let tmp = x[j] + m;
                m = PREEMPH * tmp;
                scratch[j] = tmp;
            }
            for j in 0..nd {
                pcm[j * cc + c] = scratch[j * downsample] * (1.0 / SIG_SCALE);
            }
        } else {
            for j in 0..n {
                let tmp = x[j] + m;
                m = PREEMPH * tmp;
                pcm[j * cc + c] = tmp * (1.0 / SIG_SCALE);
            }
        }
        mem[c] = m;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn celt_synthesis(
    mdct: &mut Mdct,
    x: &[f32],
    decode_mem: &mut [Vec<f32>],
    at: usize,
    old_bande: &[f32],
    start: usize,
    eff_end: usize,
    c: usize,
    cc: usize,
    is_transient: bool,
    lm: usize,
    downsample: usize,
    silence: bool,
) {
    let n = SHORT_MDCT_SIZE << lm;
    let m = 1usize << lm;
    let (blocks, nb, shift) = if is_transient {
        (m, SHORT_MDCT_SIZE, MAX_LM)
    } else {
        (1, SHORT_MDCT_SIZE << lm, MAX_LM - lm)
    };
    let mut freq = vec![0.0f32; n];

    if cc == 2 && c == 1 {
        denormalise_bands(
            x, &mut freq, old_bande, start, eff_end, m, downsample, silence,
        );
        for out in 0..2 {
            for b in 0..blocks {
                mdct.backward(
                    &freq[b..],
                    &mut decode_mem[out][at + nb * b..],
                    &WINDOW120,
                    OVERLAP,
                    shift,
                    blocks,
                );
            }
        }
    } else if cc == 1 && c == 2 {
        let mut freq2 = vec![0.0f32; n];
        denormalise_bands(
            x, &mut freq, old_bande, start, eff_end, m, downsample, silence,
        );
        denormalise_bands(
            &x[n..],
            &mut freq2,
            &old_bande[NB_EBANDS..],
            start,
            eff_end,
            m,
            downsample,
            silence,
        );
        for i in 0..n {
            freq[i] = 0.5 * freq[i] + 0.5 * freq2[i];
        }
        for b in 0..blocks {
            mdct.backward(
                &freq[b..],
                &mut decode_mem[0][at + nb * b..],
                &WINDOW120,
                OVERLAP,
                shift,
                blocks,
            );
        }
    } else {
        for ch in 0..cc {
            denormalise_bands(
                &x[ch * n..],
                &mut freq,
                &old_bande[ch * NB_EBANDS..],
                start,
                eff_end,
                m,
                downsample,
                silence,
            );
            for b in 0..blocks {
                mdct.backward(
                    &freq[b..],
                    &mut decode_mem[ch][at + nb * b..],
                    &WINDOW120,
                    OVERLAP,
                    shift,
                    blocks,
                );
            }
        }
    }
}
