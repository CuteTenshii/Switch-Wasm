//! The CELT layer: the MDCT half of Opus, and the whole of it above 8 kHz.
//!
//! CELT codes a frame as *shape* and *energy*, separately. Energy is one
//! value per band, differentially coded across time and frequency; shape is a
//! unit-norm vector per band, coded as a point on a hypersphere by the
//! algebraic PVQ. Nothing in the shape carries level, so a band always comes
//! back with exactly the energy that was signalled for it, which is why the
//! codec degrades by going grainy rather than by dropping bands.
//!
//! The bit allocation is the part that makes it work and the part that makes
//! it unforgiving. Encoder and decoder both compute, from the frame size, the
//! coded bandwidth and the bits consumed so far, exactly how many bits each
//! band gets, no side information. Everything downstream of a disagreement
//! is noise, so every count here is integer arithmetic that has to match the
//! encoder's bit for bit, right down to the direction each division rounds.
//!
//! Structure of one frame (RFC 6716 §4.3): silence flag, postfilter
//! parameters, transient flag, coarse energy, time-frequency resolution,
//! spread, per-band dynamic allocation boosts, allocation trim, the bands
//! themselves, then the fine energy that uses up whatever is left.

use super::mdct::Mdct;
use super::range::{ilog, RangeDecoder, BITRES};
use super::tables_celt::*;

pub(super) const NB_EBANDS: usize = 21;
const EFF_EBANDS: usize = 21;
pub(super) const OVERLAP: usize = 120;
const MAX_LM: usize = 3;
const SHORT_MDCT_SIZE: usize = 120;
const NB_ALLOC_VECTORS: usize = 11;
const MDCT_SIZE: usize = 1920;

/// The decoder's history, long enough for the pitch predictor's longest lag.
const DECODE_BUFFER_SIZE: usize = 2048;
const LPC_ORDER: usize = 24;
const MAX_PERIOD: usize = 1024;
const COMBFILTER_MINPERIOD: usize = 15;
const PLC_PITCH_LAG_MAX: usize = 720;
const PLC_PITCH_LAG_MIN: usize = 100;

const LOG_MAX_PSEUDO: i32 = 6;
const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const ALLOC_STEPS: i32 = 6;

const SPREAD_NONE: usize = 0;
const SPREAD_NORMAL: usize = 2;
const SPREAD_AGGRESSIVE: usize = 3;

/// The pre-emphasis the encoder applied, which synthesis has to undo.
const PREEMPH: f32 = 0.85000610;

/// The internal signal scale: one unit of `celt_sig` is one 16-bit sample
/// step, so a full-scale signal runs to ±32768.
pub(super) const SIG_SCALE: f32 = 32768.0;

/// Everything about the 48 kHz / 960 mode that is computed rather than
/// tabulated. One per decoder; it is small, and sharing it would buy nothing.
pub(super) struct Mode {
    mdct: Mdct,
}

impl Mode {
    pub(super) fn new() -> Self {
        Mode {
            mdct: Mdct::new(MDCT_SIZE, MAX_LM),
        }
    }
}

/// One tap of the MDCT window, which the mode-transition cross-fades borrow
/// so their two halves sum the same way an overlap-add would.
pub(super) fn window_at(i: usize) -> f32 {
    WINDOW120[i]
}

/// `U(n, k)`: how many PVQ codewords place `k` pulses in `n` dimensions with
/// the first dimension's pulse positive. The table is symmetric, so it is
/// stored as ragged rows indexed by the smaller of the two.
fn pvq_u(n: usize, k: usize) -> u32 {
    let (lo, hi) = if n < k { (n, k) } else { (k, n) };
    PVQ_U_DATA[PVQ_U_ROW[lo] + hi]
}

/// `V(n, k)`: the full codebook size, both signs of the leading pulse.
fn pvq_v(n: usize, k: usize) -> u32 {
    pvq_u(n, k).wrapping_add(pvq_u(n, k + 1))
}

/// Turn a codeword index back into its pulse vector, returning the vector's
/// squared norm. This walks the combinatorial ranking one dimension at a
/// time, subtracting the count of codewords that start with fewer pulses
/// until the index falls inside the current dimension's block.
fn cwrsi(mut n: usize, mut k: usize, mut i: u32, y: &mut [i32]) -> f32 {
    let mut yy = 0.0f32;
    let mut at = 0usize;
    while n > 2 {
        let (p, s, k0);
        if k >= n {
            // More pulses than dimensions: the leading dimension almost
            // certainly holds some, so search down from `k`.
            let row = PVQ_U_ROW[n];
            let pv = PVQ_U_DATA[row + k + 1];
            s = if i >= pv { -1i32 } else { 0 };
            i -= pv & (s as u32);
            k0 = k;
            let q = PVQ_U_DATA[row + n];
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    if PVQ_U_DATA[PVQ_U_ROW[k] + n] <= i {
                        break;
                    }
                }
                p = PVQ_U_DATA[PVQ_U_ROW[k] + n];
            } else {
                let mut pp = PVQ_U_DATA[row + k];
                while pp > i {
                    k -= 1;
                    pp = PVQ_U_DATA[row + k];
                }
                p = pp;
            }
            i -= p;
        } else {
            // More dimensions than pulses: this one is most likely empty.
            let pv = pvq_u(k, n);
            let q = pvq_u(k + 1, n);
            if pv <= i && i < q {
                i -= pv;
                y[at] = 0;
                at += 1;
                n -= 1;
                continue;
            }
            s = if i >= q { -1i32 } else { 0 };
            i -= q & (s as u32);
            k0 = k;
            loop {
                k -= 1;
                if pvq_u(k, n) <= i {
                    break;
                }
            }
            p = pvq_u(k, n);
            i -= p;
        }
        let val = ((k0 - k) as i32 + s) ^ s;
        y[at] = val;
        at += 1;
        yy += (val * val) as f32;
        n -= 1;
    }
    // n == 2: the ranking is linear from here.
    let p = 2 * k as u32 + 1;
    let s = if i >= p { -1i32 } else { 0 };
    i -= p & (s as u32);
    let k0 = k;
    k = ((i + 1) >> 1) as usize;
    if k != 0 {
        i -= 2 * k as u32 - 1;
    }
    let val = ((k0 - k) as i32 + s) ^ s;
    y[at] = val;
    at += 1;
    yy += (val * val) as f32;
    // n == 1: whatever is left, and its sign.
    let s = -(i as i32);
    let val = (k as i32 + s) ^ s;
    y[at] = val;
    yy += (val * val) as f32;
    yy
}

fn decode_pulses(y: &mut [i32], n: usize, k: usize, dec: &mut RangeDecoder) -> f32 {
    let index = dec.decode_uint(pvq_v(n, k));
    cwrsi(n, k, index, y)
}

/// How many pulses a pseudo-pulse count stands for. Above 8 the count is
/// coded logarithmically, because the codebook grows faster than the ear
/// cares.
fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// The cache row for one band at one block size: `bits[0]` is the number of
/// entries, and `bits[q]` is one less than the bits `q` pseudo-pulses cost.
fn cache_row(band: usize, lm: i32) -> &'static [u8] {
    let index = CACHE_INDEX50[((lm + 1) as usize) * NB_EBANDS + band];
    debug_assert!(
        index >= 0,
        "pulse cache row for a band with no coefficients"
    );
    &CACHE_BITS50[index.max(0) as usize..]
}

/// The largest pseudo-pulse count whose codebook fits in `bits`.
fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let cache = cache_row(band, lm);
    let mut lo = 0i32;
    let mut hi = i32::from(cache[0]);
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if i32::from(cache[mid as usize]) >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let below = if lo == 0 {
        -1
    } else {
        i32::from(cache[lo as usize])
    };
    if bits - below <= i32::from(cache[hi as usize]) - bits {
        lo
    } else {
        hi
    }
}

fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        i32::from(cache_row(band, lm)[pulses as usize]) + 1
    }
}

/// A cosine accurate to the last bit on every platform. The bit allocation
/// depends on it, so an implementation that merely rounded differently would
/// hand the bands different budgets than the encoder used.
fn bitexact_cos(x: i16) -> i32 {
    let tmp = (4096 + i32::from(x) * i32::from(x)) >> 13;
    let x2 = (32767 - tmp) + frac_mul16(tmp, -7651 + frac_mul16(tmp, 8277 + frac_mul16(-626, tmp)));
    1 + x2
}

fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ilog(icos as u32) as i32;
    let ls = ilog(isin as u32) as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// Multiply two Q15 values, rounding, exactly as the reference does, both
/// operands are truncated to 16 bits first, and the allocation depends on it.
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// The linear congruential generator CELT fills empty bands with. Its exact
/// sequence is part of the format: the encoder assumed these samples when it
/// decided the band needed no bits.
fn lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// A Laplace-distributed value, used for every coarse energy delta.
fn laplace_decode(dec: &mut RangeDecoder, fs0: u32, decay: i32) -> i32 {
    /// The floor probability of any delta, out of 32768.
    const MINP: u32 = 1;
    /// How many deltas either side are guaranteed representable.
    const NMIN: u32 = 16;

    let fm = dec.decode_bin(15);
    let mut fl = 0u32;
    let mut fs = fs0;
    let mut val = 0i32;
    if fm >= fs {
        val += 1;
        fl = fs;
        let ft = 32768 - MINP * (2 * NMIN) - fs0;
        fs = ((ft * (16384 - decay) as u32) >> 15) + MINP;
        while fs > MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (((fs - 2 * MINP) * decay as u32) >> 15) + MINP;
            val += 1;
        }
        if fs <= MINP {
            let di = (fm - fl) >> 1;
            val += di as i32;
            fl += 2 * di * MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    dec.update(fl, (fl + fs).min(32768), 32768);
    val
}

/// Coarse energy: one value per band per channel, predicted from the band
/// below and from the same band in the previous frame.
///
/// The inter-frame predictor is why a lost frame is audible beyond itself,
/// and why `intra` exists, to break the chain at a cost in bits.
fn unquant_coarse_energy(
    start: usize,
    end: usize,
    old_bande: &mut [f32],
    intra: bool,
    dec: &mut RangeDecoder,
    channels: usize,
    lm: usize,
    total_bytes: usize,
) {
    const PRED_COEF: [f32; 4] = [29440.0 / 32768.0, 26112.0 / 32768.0, 21248.0 / 32768.0, 0.5];
    const BETA_COEF: [f32; 4] = [
        30147.0 / 32768.0,
        22282.0 / 32768.0,
        12124.0 / 32768.0,
        6554.0 / 32768.0,
    ];
    const BETA_INTRA: f32 = 4915.0 / 32768.0;

    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let budget = (total_bytes * 8) as i32;
    let model = &E_PROB_MODEL[(lm * 2 + usize::from(intra)) * 42..][..42];
    let mut prev = [0.0f32; 2];

    for i in start..end {
        for c in 0..channels {
            let tell = dec.tell();
            let qi = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                laplace_decode(
                    dec,
                    u32::from(model[pi]) << 7,
                    i32::from(model[pi + 1]) << 6,
                )
            } else if budget - tell >= 2 {
                let qi = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (qi >> 1) ^ -(qi & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.decode_bit_logp(1))
            } else {
                -1
            };
            let q = qi as f32;
            let slot = &mut old_bande[c * NB_EBANDS + i];
            *slot = slot.max(-9.0);
            *slot = coef * *slot + prev[c] + q;
            prev[c] = prev[c] + q - beta * q;
        }
    }
}

/// Fine energy: the bits the allocator set aside to refine each coarse value,
/// read as a plain uniform fraction of the coarse step.
fn unquant_fine_energy(
    start: usize,
    end: usize,
    old_bande: &mut [f32],
    fine_quant: &[i32],
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        for c in 0..channels {
            let q2 = dec.decode_bits(fine_quant[i] as u32);
            let offset = (q2 as f32 + 0.5) * ((1 << (14 - fine_quant[i])) as f32) / 16384.0 - 0.5;
            old_bande[c * NB_EBANDS + i] += offset;
        }
    }
}

/// Whatever bits are left after everything else, spent one at a time on the
/// bands the allocator marked as having been rounded down.
fn unquant_energy_finalise(
    start: usize,
    end: usize,
    old_bande: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for c in 0..channels {
                let q2 = dec.decode_bits(1);
                let offset = (q2 as f32 - 0.5) * ((1 << (14 - fine_quant[i] - 1)) as f32) / 16384.0;
                old_bande[c * NB_EBANDS + i] += offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// Rotate a band's samples so that a small number of pulses spreads across it
/// rather than sitting as isolated spikes. The encoder rotated the other way
/// before searching, so this is exactly invertible.
fn exp_rotation(x: &mut [f32], len: usize, stride: usize, k: i32, spread: usize) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if 2 * k >= len as i32 || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[spread - 1];
    let gain = len as f32 / (len as i32 + factor * k) as f32;
    let theta = 0.5 * gain * gain;
    let c = (0.5 * core::f32::consts::PI * theta).cos();
    let s = (0.5 * core::f32::consts::PI * (1.0 - theta)).cos();

    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        // sqrt(len/stride), rounded: grow while (stride2+0.5)^2 fits.
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let block = len / stride;
    for i in 0..stride {
        let part = &mut x[i * block..(i + 1) * block];
        if stride2 != 0 {
            exp_rotation1(part, block, stride2, s, c);
        }
        exp_rotation1(part, block, 1, c, s);
    }
}

fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let ms = -s;
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
    for i in (0..len.saturating_sub(2 * stride)).rev() {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
}

/// Which of a band's `blocks` sub-blocks got at least one pulse. A block with
/// none has collapsed, and [`anti_collapse`] will refill it with noise rather
/// than leave a hole that pumps.
fn extract_collapse_mask(y: &[i32], n: usize, blocks: usize) -> u32 {
    if blocks <= 1 {
        return 1;
    }
    let size = n / blocks;
    let mut mask = 0u32;
    for i in 0..blocks {
        let any = y[i * size..(i + 1) * size].iter().any(|&v| v != 0);
        mask |= u32::from(any) << i;
    }
    mask
}

fn renormalise_vector(x: &mut [f32], gain: f32) {
    let energy: f32 = 1e-15 + x.iter().map(|&v| v * v).sum::<f32>();
    let g = gain / energy.sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// Decode one band's shape: a PVQ codeword, scaled to the gain the caller
/// worked out from the energy split.
fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: usize,
    blocks: usize,
    dec: &mut RangeDecoder,
    gain: f32,
) -> u32 {
    let mut iy = vec![0i32; n];
    let ryy = decode_pulses(&mut iy, n, k as usize, dec);
    let g = gain / ryy.sqrt();
    for i in 0..n {
        x[i] = g * iy[i] as f32;
    }
    exp_rotation(x, n, blocks, k, spread);
    extract_collapse_mask(&iy, n, blocks)
}

/// One step of a Haar transform, which is how CELT trades frequency
/// resolution for time resolution inside a band.
fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    const SQRT_HALF: f32 = 0.70710678;
    let half = n0 >> 1;
    for i in 0..stride {
        for j in 0..half {
            let a = SQRT_HALF * x[stride * 2 * j + i];
            let b = SQRT_HALF * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = a + b;
            x[stride * (2 * j + 1) + i] = a - b;
        }
    }
}

fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    for i in 0..stride {
        let dest = if hadamard {
            ORDERY_TABLE[stride - 2 + i]
        } else {
            i
        };
        for j in 0..n0 {
            tmp[dest * n0 + j] = x[j * stride + i];
        }
    }
    x[..n].copy_from_slice(&tmp);
}

fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    for i in 0..stride {
        let src = if hadamard {
            ORDERY_TABLE[stride - 2 + i]
        } else {
            i
        };
        for j in 0..n0 {
            tmp[j * stride + i] = x[src * n0 + j];
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// How finely the mid/side angle may be coded, given the bits available.
fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n as i32 - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    // The cap keeps a stereo split with the angle hard over from leaving the
    // side with no bits at all: a side band with no pulses is not folded, so
    // it would collapse to silence.
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES) >> 1 {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        (qn + 1) >> 1 << 1
    }
}

/// What a stereo (or time) split decided, and what it cost.
struct SplitCtx {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

/// Everything a band decode carries between its recursive halves.
struct BandCtx<'a, 'p> {
    dec: &'a mut RangeDecoder<'p>,
    band: usize,
    intensity: usize,
    spread: usize,
    tf_change: i32,
    remaining_bits: i32,
    seed: u32,
    disable_inv: bool,
}

/// Decode the angle between the two halves of a split, and work out how the
/// bits divide between them.
fn compute_theta(
    ctx: &mut BandCtx,
    n: usize,
    b: &mut i32,
    blocks: usize,
    b0: usize,
    lm: i32,
    stereo: bool,
    fill: &mut i32,
) -> SplitCtx {
    let pulse_cap = i32::from(LOG_N400[ctx.band]) + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && ctx.band >= ctx.intensity {
        qn = 1;
    }
    let tell = ctx.dec.tell_frac() as i32;
    let mut itheta = 0i32;
    let mut inv = false;

    if qn != 1 {
        // A uniform pdf for a time split, a step for stereo, a triangular one
        // for the rest: the shapes the angle actually takes.
        if stereo && n > 2 {
            let p0 = 3u32;
            let x0 = (qn / 2) as u32;
            let ft = p0 * (x0 + 1) + x0;
            let fs = ctx.dec.decode(ft);
            let value = if fs < (x0 + 1) * p0 {
                fs / p0
            } else {
                x0 + 1 + (fs - (x0 + 1) * p0)
            };
            let (fl, fh) = if value <= x0 {
                (p0 * value, p0 * (value + 1))
            } else {
                (
                    (value - 1 - x0) + (x0 + 1) * p0,
                    (value - x0) + (x0 + 1) * p0,
                )
            };
            ctx.dec.update(fl, fh, ft);
            itheta = value as i32;
        } else if b0 > 1 || stereo {
            itheta = ctx.dec.decode_uint(qn as u32 + 1) as i32;
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fm = ctx.dec.decode(ft as u32) as i32;
            let (fl, fs);
            if fm < ((qn >> 1) * ((qn >> 1) + 1)) >> 1 {
                itheta = (isqrt32(8 * fm as u32 + 1) as i32 - 1) >> 1;
                fs = itheta + 1;
                fl = (itheta * (itheta + 1)) >> 1;
            } else {
                itheta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) as u32 + 1) as i32) >> 1;
                fs = qn + 1 - itheta;
                fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
            }
            ctx.dec.update(fl as u32, (fl + fs) as u32, ft as u32);
        }
        debug_assert!(itheta >= 0);
        itheta = (itheta * 16384) / qn;
    } else if stereo {
        // With no angle to code, the side may still have been inverted.
        inv = if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            ctx.dec.decode_bit_logp(2)
        } else {
            false
        };
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }
    let qalloc = ctx.dec.tell_frac() as i32 - tell;
    *b -= qalloc;

    let (imid, iside, delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1 << blocks) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << blocks) - 1) << blocks;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16);
        iside = bitexact_cos((16384 - itheta) as i16);
        // The mid/side bit split that minimises squared error in this band.
        delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }
    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

/// Integer square root, matching the reference's exactly, the triangular
/// angle pdf inverts through it, so a value one off decodes a different
/// angle.
fn isqrt32(mut val: u32) -> u32 {
    let mut g = 0u32;
    let mut bshift = (ilog(val) as i32 - 1) >> 1;
    let mut b = 1u32 << bshift;
    loop {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// A band of one coefficient: nothing to shape, just a sign.
fn quant_band_n1(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut channels: [&mut [f32]; 2];
    let count;
    match y {
        Some(y) => {
            channels = [x, y];
            count = 2;
        }
        None => {
            channels = [x, &mut []];
            count = 1;
        }
    }
    for ch in channels.iter_mut().take(count) {
        let mut sign = 0u32;
        if ctx.remaining_bits >= 1 << BITRES {
            sign = ctx.dec.decode_bits(1);
            ctx.remaining_bits -= 1 << BITRES;
        }
        ch[0] = if sign != 0 { -1.0 } else { 1.0 };
    }
    if let Some(out) = lowband_out {
        out[0] = channels[0][0];
    }
    1
}

/// Decode one partition, splitting it in two and coding the energy angle
/// between the halves whenever a single PVQ codeword would need more bits
/// than the band was given.
fn quant_partition(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: i32,
) -> u32 {
    let b0 = blocks;
    let mut fill = fill;

    let splittable = if lm != -1 && n > 2 {
        let cache = cache_row(ctx.band, lm);
        b > i32::from(cache[cache[0] as usize]) + 12
    } else {
        false
    };

    if splittable {
        let half = n >> 1;
        let lm = lm - 1;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        let blocks = (blocks + 1) >> 1;

        let mut b = b;
        let sctx = compute_theta(ctx, half, &mut b, blocks, b0, lm, false, &mut fill);
        let mid = sctx.imid as f32 * (1.0 / 32768.0);
        let side = sctx.iside as f32 * (1.0 / 32768.0);
        let mut delta = sctx.delta;

        // Short blocks that carry little energy still need enough bits not to
        // pre-echo, so bias the split towards the quieter half.
        if b0 > 1 && (sctx.itheta & 0x3fff) != 0 {
            if sctx.itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = 0.min(delta + ((half as i32) << BITRES >> (5 - lm)));
            }
        }
        let mut mbits = 0.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sctx.qalloc;

        let (xl, xr) = x.split_at_mut(half);
        let (mut lb_lo, mut lb_hi) = match lowband {
            Some(lb) => {
                let (a, c) = lb.split_at_mut(half);
                (Some(a), Some(c))
            }
            None => (None, None),
        };

        let rebalance = ctx.remaining_bits;
        let cm;
        if mbits >= sbits {
            cm = quant_partition(
                ctx,
                xl,
                half,
                mbits,
                blocks,
                lb_lo.take(),
                lm,
                gain * mid,
                fill,
            );
            let spent = mbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 0 {
                sbits += spent - (3 << BITRES);
            }
            cm | quant_partition(
                ctx,
                xr,
                half,
                sbits,
                blocks,
                lb_hi.take(),
                lm,
                gain * side,
                fill >> blocks,
            ) << (b0 >> 1)
        } else {
            let cm2 = quant_partition(
                ctx,
                xr,
                half,
                sbits,
                blocks,
                lb_hi.take(),
                lm,
                gain * side,
                fill >> blocks,
            ) << (b0 >> 1);
            let spent = sbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 16384 {
                mbits += spent - (3 << BITRES);
            }
            cm2 | quant_partition(
                ctx,
                xl,
                half,
                mbits,
                blocks,
                lb_lo.take(),
                lm,
                gain * mid,
                fill,
            )
        }
    } else {
        let mut q = bits2pulses(ctx.band, lm, b);
        let mut curr_bits = pulses2bits(ctx.band, lm, q);
        ctx.remaining_bits -= curr_bits;
        // Never bust the budget: drop a pulse at a time until it fits.
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(ctx.band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            alg_unquant(x, n, get_pulses(q), ctx.spread, blocks, ctx.dec, gain)
        } else {
            // No pulses: fill the band from somewhere rather than leave it
            // empty, because silence in one band of a loud frame is audible.
            let mask = ((1u32 << blocks) - 1) as i32;
            fill &= mask;
            if fill == 0 {
                x[..n].fill(0.0);
                0
            } else {
                match lowband {
                    None => {
                        for v in x.iter_mut().take(n) {
                            ctx.seed = lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f32;
                        }
                        renormalise_vector(&mut x[..n], gain);
                        mask as u32
                    }
                    Some(lb) => {
                        // Folded spectrum: a copy of a lower band, dithered
                        // about 48 dB below the normal folding level.
                        for j in 0..n {
                            ctx.seed = lcg_rand(ctx.seed);
                            let tmp = if ctx.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            x[j] = lb[j] + tmp;
                        }
                        renormalise_vector(&mut x[..n], gain);
                        fill as u32
                    }
                }
            }
        }
    }
}

/// Decode one band of one channel, applying whatever time-frequency
/// resolution change the frame asked for around the partition decode.
#[allow(clippy::too_many_arguments)]
fn quant_band(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: i32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let mut blocks = blocks;
    let b0 = blocks;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;
    let mut fill = fill;
    let mut tf_change = ctx.tf_change;

    n_b /= blocks;

    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }

    if tf_change > 0 {
        recombine = tf_change;
    }

    // `lowband` is the caller's own copy of the folding source, not the
    // `norm` history itself, because the transforms below rewrite it in
    // place and a later band still has to fold from the original.
    let mut lowband = lowband;

    for k in 0..recombine {
        if let Some(lb) = lowband.as_deref_mut() {
            haar1(lb, n >> k, 1 << k);
        }
        fill = i32::from(BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize])
            | i32::from(BIT_INTERLEAVE_TABLE[(fill >> 4) as usize]) << 2;
    }
    blocks >>= recombine;
    n_b <<= recombine;

    while (n_b & 1) == 0 && tf_change < 0 {
        if let Some(lb) = lowband.as_deref_mut() {
            haar1(lb, n_b, blocks);
        }
        fill |= fill << blocks;
        blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    let b0 = blocks;
    let n_b0 = n_b;

    if b0 > 1 {
        if let Some(lb) = lowband.as_deref_mut() {
            deinterleave_hadamard(lb, n_b >> recombine, b0 << recombine, long_blocks);
        }
    }

    let mut cm = quant_partition(ctx, x, n, b, blocks, lowband, lm, gain, fill);

    if b0 > 1 {
        interleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
    }

    let mut n_b = n_b0;
    let mut blocks = b0;
    for _ in 0..time_divide {
        blocks >>= 1;
        n_b <<= 1;
        cm |= cm >> blocks;
        haar1(x, n_b, blocks);
    }
    for k in 0..recombine {
        cm = u32::from(BIT_DEINTERLEAVE_TABLE[(cm & 0xF) as usize]);
        haar1(x, n0 >> k, 1 << k);
    }
    blocks <<= recombine;

    // Scale for whoever folds from this band later: folding wants the band at
    // the amplitude it would have if it were a full-length spectrum.
    if let Some(out) = lowband_out {
        let scale = (n0 as f32).sqrt();
        for j in 0..n0 {
            out[j] = scale * x[j];
        }
    }
    cm & ((1 << blocks) - 1)
}

/// Decode one band of both channels together, coding the angle between them
/// rather than each channel's energy separately.
#[allow(clippy::too_many_arguments)]
fn quant_band_stereo(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    fill: i32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }
    let orig_fill = fill;
    let mut fill = fill;
    let mut b = b;

    let sctx = compute_theta(ctx, n, &mut b, blocks, blocks, lm, true, &mut fill);
    let mid = sctx.imid as f32 * (1.0 / 32768.0);
    let side = sctx.iside as f32 * (1.0 / 32768.0);

    let cm;
    if n == 2 {
        // Mid and side are orthogonal here, so the side needs only a sign.
        let mut mbits = b;
        let mut sbits = 0;
        if sctx.itheta != 0 && sctx.itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let swapped = sctx.itheta > 8192;
        ctx.remaining_bits -= sctx.qalloc + sbits;

        let sign = if sbits != 0 {
            ctx.dec.decode_bits(1)
        } else {
            0
        };
        let sign = 1.0 - 2.0 * sign as f32;

        let (x2, y2) = if swapped {
            (&mut *y, &mut *x)
        } else {
            (&mut *x, &mut *y)
        };
        // `orig_fill`, not `fill`: the side is still folded even when the
        // angle cleared the low bits.
        cm = quant_band(
            ctx,
            x2,
            n,
            mbits,
            blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            orig_fill,
        );
        y2[0] = -sign * x2[1];
        y2[1] = sign * x2[0];

        x[0] *= mid;
        x[1] *= mid;
        y[0] *= side;
        y[1] *= side;
        let tmp = x[0];
        x[0] = tmp - y[0];
        y[0] = tmp + y[0];
        let tmp = x[1];
        x[1] = tmp - y[1];
        y[1] = tmp + y[1];
    } else {
        let mut mbits = 0.max(b.min((b - sctx.delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sctx.qalloc;
        let rebalance = ctx.remaining_bits;

        if mbits >= sbits {
            // The mid keeps unit gain because later bands fold from it.
            let cm0 = quant_band(
                ctx,
                x,
                n,
                mbits,
                blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            let spent = mbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 0 {
                sbits += spent - (3 << BITRES);
            }
            cm = cm0
                | quant_band(
                    ctx,
                    y,
                    n,
                    sbits,
                    blocks,
                    None,
                    lm,
                    None,
                    side,
                    fill >> blocks,
                );
        } else {
            let cm0 = quant_band(
                ctx,
                y,
                n,
                sbits,
                blocks,
                None,
                lm,
                None,
                side,
                fill >> blocks,
            );
            let spent = sbits - (rebalance - ctx.remaining_bits);
            if spent > 3 << BITRES && sctx.itheta != 16384 {
                mbits += spent - (3 << BITRES);
            }
            cm = cm0
                | quant_band(
                    ctx,
                    x,
                    n,
                    mbits,
                    blocks,
                    lowband,
                    lm,
                    lowband_out,
                    1.0,
                    fill,
                );
        }
    }

    if n != 2 {
        stereo_merge(x, y, mid, n);
    }
    if sctx.inv {
        for v in y.iter_mut().take(n) {
            *v = -*v;
        }
    }
    cm
}

/// Turn a decoded mid/side pair back into left and right, normalising each to
/// the energy the angle implies.
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    let mut xp = 0.0f32;
    let mut side = 0.0f32;
    for j in 0..n {
        xp += x[j] * y[j];
        side += y[j] * y[j];
    }
    xp *= mid;
    let mid2 = mid;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

/// Copy enough of the first coded band's folding data forward that the second
/// band has something to fold from. Only hybrid frames, which start above
/// band 0, need it.
fn special_hybrid_folding(norm: &mut [f32], norm2: Option<&mut [f32]>, start: usize, m: usize) {
    let n1 = m * (EBAND_5MS[start + 1] - EBAND_5MS[start]) as usize;
    let n2 = m * (EBAND_5MS[start + 2] - EBAND_5MS[start + 1]) as usize;
    if n2 <= n1 {
        return;
    }
    norm.copy_within(2 * n1 - n2..n1, n1);
    if let Some(norm2) = norm2 {
        norm2.copy_within(2 * n1 - n2..n1, n1);
    }
}

/// Decode every band of the frame, in order, tracking the running bit balance
/// the encoder used to decide what each band could afford.
#[allow(clippy::too_many_arguments)]
fn quant_all_bands(
    start: usize,
    end: usize,
    x: &mut [f32],
    channels: usize,
    collapse_masks: &mut [u8],
    pulses: &[i32],
    short_blocks: bool,
    spread: usize,
    mut dual_stereo: bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    dec: &mut RangeDecoder,
    lm: usize,
    coded_bands: usize,
    seed: &mut u32,
    disable_inv: bool,
) {
    let m = 1usize << lm;
    let blocks = if short_blocks { m } else { 1 };
    let norm_offset = m * EBAND_5MS[start] as usize;
    let norm_len = m * EBAND_5MS[NB_EBANDS - 1] as usize - norm_offset;
    let n_total = m * SHORT_MDCT_SIZE;

    let mut norm = vec![0.0f32; norm_len];
    let mut norm2 = vec![0.0f32; if channels == 2 { norm_len } else { 0 }];

    let (xch, ych) = x.split_at_mut(n_total);
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

    let mut ctx = BandCtx {
        dec,
        band: start,
        intensity,
        spread,
        tf_change: 0,
        remaining_bits: 0,
        seed: *seed,
        disable_inv,
    };

    for i in start..end {
        ctx.band = i;
        let last = i == end - 1;
        let lo = m * EBAND_5MS[i] as usize;
        let hi = m * EBAND_5MS[i + 1] as usize;
        let n = hi - lo;
        let tell = ctx.dec.tell_frac() as i32;

        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i <= coded_bands - 1 {
            let curr_balance = balance / 3.min(coded_bands - i) as i32;
            0.max(16383.min((remaining_bits + 1).min(pulses[i] + curr_balance)))
        } else {
            0
        };

        if (lo >= norm_offset + n || i == start + 1) && (update_lowband || lowband_offset == 0) {
            lowband_offset = i;
        }
        if i == start + 1 {
            let norm2_ref = if channels == 2 {
                Some(norm2.as_mut_slice())
            } else {
                None
            };
            special_hybrid_folding(&mut norm, norm2_ref, start, m);
        }

        ctx.tf_change = tf_res[i];

        // A conservative estimate of which sub-blocks of the folding source
        // carry energy: where it collapsed, folding from it would leave this
        // band silent too, and the encoder assumed the same.
        let mut effective_lowband: Option<usize> = None;
        let mut x_cm;
        let mut y_cm;
        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || blocks > 1 || ctx.tf_change < 0) {
            // Never repeat spectral content within one band.
            let eff = (m * EBAND_5MS[lowband_offset] as usize).saturating_sub(norm_offset + n);
            effective_lowband = Some(eff);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m * EBAND_5MS[fold_start] as usize <= eff + norm_offset {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if fold_end >= i || m * EBAND_5MS[fold_end] as usize >= eff + norm_offset + n {
                    break;
                }
            }
            x_cm = 0u32;
            y_cm = 0u32;
            for fold_i in fold_start..fold_end {
                x_cm |= u32::from(collapse_masks[fold_i * channels]);
                y_cm |= u32::from(collapse_masks[fold_i * channels + channels - 1]);
            }
        } else {
            // Nothing to fold from, so the LCG fills the band and every block
            // is (almost always) non-zero.
            x_cm = (1u32 << blocks) - 1;
            y_cm = x_cm;
        }

        if dual_stereo && i == intensity {
            // Intensity coding takes over here, so the two folding histories
            // become one.
            dual_stereo = false;
            for j in 0..lo - norm_offset {
                norm[j] = 0.5 * (norm[j] + norm2[j]);
            }
        }

        let split = lo - norm_offset;
        // A private copy of the folding source: the transforms inside
        // `quant_band` rewrite it, and in a hybrid frame the region it comes
        // from overlaps the one this band is about to write back.
        let mut lowband = effective_lowband.map(|o| norm[o..o + n].to_vec());
        let mut lowband2 = if dual_stereo {
            effective_lowband.map(|o| norm2[o..o + n].to_vec())
        } else {
            None
        };

        if dual_stereo {
            x_cm = quant_band(
                &mut ctx,
                &mut xch[lo..hi],
                n,
                b / 2,
                blocks,
                lowband.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[split..split + n])
                },
                1.0,
                x_cm as i32,
            );
            y_cm = quant_band(
                &mut ctx,
                &mut ych[lo..hi],
                n,
                b / 2,
                blocks,
                lowband2.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm2[split..split + n])
                },
                1.0,
                y_cm as i32,
            );
        } else if channels == 2 {
            x_cm = quant_band_stereo(
                &mut ctx,
                &mut xch[lo..hi],
                &mut ych[lo..hi],
                n,
                b,
                blocks,
                lowband.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[split..split + n])
                },
                (x_cm | y_cm) as i32,
            );
            y_cm = x_cm;
        } else {
            x_cm = quant_band(
                &mut ctx,
                &mut xch[lo..hi],
                n,
                b,
                blocks,
                lowband.as_deref_mut(),
                lm as i32,
                if last {
                    None
                } else {
                    Some(&mut norm[split..split + n])
                },
                1.0,
                (x_cm | y_cm) as i32,
            );
            y_cm = x_cm;
        }
        collapse_masks[i * channels] = x_cm as u8;
        collapse_masks[i * channels + channels - 1] = y_cm as u8;
        balance += pulses[i] + tell;

        // Keep moving the folding source forward only while the band has at
        // least one bit per sample to be folded from.
        update_lowband = b > (n as i32) << BITRES;
    }
    *seed = ctx.seed;
}

/// The most bits a band can use before more would be wasted: past this the
/// PVQ codebook is finer than the band's own energy resolution.
fn init_caps(cap: &mut [i32], lm: usize, channels: usize) {
    for i in 0..NB_EBANDS {
        let n = ((EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32) << lm;
        let row = NB_EBANDS * (2 * lm + channels - 1);
        cap[i] = ((i32::from(CACHE_CAPS50[row + i]) + 64) * channels as i32 * n) >> 2;
    }
}

/// Interpolate between the two nearest rows of the allocation table, then
/// split each band's share into fine-energy bits and PVQ bits.
///
/// The bisection is the whole trick: both ends run it identically, so the
/// only thing on the wire is which bands were skipped.
#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    balance_out: &mut i32,
    skip_rsv: i32,
    intensity: &mut usize,
    mut intensity_rsv: i32,
    dual_stereo: &mut bool,
    mut dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    channels: usize,
    lm: usize,
    dec: &mut RangeDecoder,
) -> usize {
    let alloc_floor = (channels as i32) << BITRES;
    let stereo = channels > 1;
    let log_m = (lm as i32) << BITRES;

    let mut lo = 0i32;
    let mut hi = 1i32 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Decide which bands to skip, working back from the top. Never skip the
    // first band or one dynalloc boosted: either would spend a bit saying the
    // bits just requested should be thrown away.
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        }
        let mut left = total - psum;
        let percoeff = left / (EBAND_5MS[coded_bands] - EBAND_5MS[start]) as i32;
        left -= (EBAND_5MS[coded_bands] - EBAND_5MS[start]) as i32 * percoeff;
        let rem = 0.max(left - (EBAND_5MS[j] - EBAND_5MS[start]) as i32);
        let band_width = (EBAND_5MS[coded_bands] - EBAND_5MS[j]) as i32;
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        // Only code a skip decision when the band could afford the flag;
        // below that it is force-skipped and nothing is transmitted.
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if dec.decode_bit_logp(1) {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = i32::from(LOG2_FRAC_TABLE[j - start]);
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }

    if intensity_rsv > 0 {
        *intensity = start + dec.decode_uint((coded_bands + 1 - start) as u32) as usize;
    } else {
        *intensity = 0;
    }
    if *intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    *dual_stereo = dual_stereo_rsv > 0 && dec.decode_bit_logp(1);

    // Hand out what is left, a whole coefficient at a time.
    let mut left = total - psum;
    let percoeff = left / (EBAND_5MS[coded_bands] - EBAND_5MS[start]) as i32;
    left -= (EBAND_5MS[coded_bands] - EBAND_5MS[start]) as i32 * percoeff;
    for j in start..coded_bands {
        bits[j] += percoeff * (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
    }
    for j in start..coded_bands {
        let tmp = left.min((EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32);
        bits[j] += tmp;
        left -= tmp;
    }

    let mut balance = 0i32;
    for j in start..coded_bands {
        let n0 = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let n = n0 << lm;
        let bit = bits[j] + balance;
        let mut excess;

        if n > 1 {
            excess = 0.max(bit - cap[j]);
            bits[j] = bit - excess;

            // Stereo has an extra degree of freedom when the two channels are
            // coded jointly, and it costs bits like any other.
            let den = channels as i32 * n
                + i32::from(channels == 2 && n > 2 && !*dual_stereo && j < *intensity);
            let nclogn = den * (i32::from(LOG_N400[j]) + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            // The second and third fine bits are worth more than the curve
            // says, so bring them forward.
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclogn >> 3;
            }
            ebits[j] = 0.max(bits[j] + offset + (den << (BITRES - 1)));
            ebits[j] = (ebits[j] / den) >> BITRES;
            if channels as i32 * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> u32::from(stereo) >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            // A band rounded down here is a candidate for the final pass that
            // spends whatever is left over.
            fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= (channels as i32 * ebits[j]) << BITRES;
        } else {
            // One coefficient: everything but the sign goes to fine energy.
            excess = 0.max(bit - ((channels as i32) << BITRES));
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }

        // Fine energy cannot use the rebalancing that happens while the bands
        // are decoded, so rebalance it here instead.
        if excess > 0 {
            let extra_fine = (excess >> (u32::from(stereo) + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * channels as i32) << BITRES;
            fine_priority[j] = i32::from(extra_bits >= excess - balance);
            excess -= extra_bits;
        }
        balance = excess;
    }
    *balance_out = balance;

    // A skipped band spends all it has on fine energy.
    for j in coded_bands..end {
        ebits[j] = bits[j] >> u32::from(stereo) >> BITRES;
        bits[j] = 0;
        fine_priority[j] = i32::from(ebits[j] < 1);
    }
    coded_bands
}

/// Work out how many bits each band gets, from the frame size, the coded
/// bandwidth, the dynamic-allocation boosts and the trim.
#[allow(clippy::too_many_arguments)]
fn compute_allocation(
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut usize,
    dual_stereo: &mut bool,
    mut total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    channels: usize,
    lm: usize,
    dec: &mut RangeDecoder,
) -> usize {
    total = total.max(0);
    let mut skip_start = start;
    // One bit says where manual skipping stops.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;

    let mut intensity_rsv = 0i32;
    let mut dual_stereo_rsv = 0i32;
    if channels == 2 {
        intensity_rsv = i32::from(LOG2_FRAC_TABLE[end - start]);
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];

    for j in start..end {
        let width = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        // Below this a band gets no PVQ bits at all.
        thresh[j] = ((channels as i32) << BITRES).max(((3 * width) << lm << BITRES) >> 4);
        trim_offset[j] = (channels as i32
            * width
            * (alloc_trim - 5 - lm as i32)
            * (end - j - 1) as i32
            * (1i32 << (lm + BITRES as usize)))
            >> 6;
        // A band of one coefficient gains more from a coarse value per
        // coefficient than from resolution, so give it less.
        if width << lm == 1 {
            trim_offset[j] -= (channels as i32) << BITRES;
        }
    }

    let mut lo = 1i32;
    let mut hi = NB_ALLOC_VECTORS as i32 - 1;
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
            let mut bitsj =
                (channels as i32 * n * i32::from(BAND_ALLOCATION[mid as usize * NB_EBANDS + j]))
                    << lm
                    >> 2;
            if bitsj > 0 {
                bitsj = 0.max(bitsj + trim_offset[j]);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= (channels as i32) << BITRES {
                psum += (channels as i32) << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    hi = lo;
    lo -= 1;

    for j in start..end {
        let n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let mut bits1j =
            (channels as i32 * n * i32::from(BAND_ALLOCATION[lo as usize * NB_EBANDS + j])) << lm
                >> 2;
        let mut bits2j = if hi >= NB_ALLOC_VECTORS as i32 {
            cap[j]
        } else {
            (channels as i32 * n * i32::from(BAND_ALLOCATION[hi as usize * NB_EBANDS + j])) << lm
                >> 2
        };
        if bits1j > 0 {
            bits1j = 0.max(bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = 0.max(bits2j + trim_offset[j]);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2[j] = 0.max(bits2j - bits1j);
        bits1[j] = bits1j;
    }

    interp_bits2pulses(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        channels,
        lm,
        dec,
    )
}

/// Read the per-band time-frequency resolution changes, then the one bit that
/// selects which of two interpretations of them the frame meant.
fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: usize,
    dec: &mut RangeDecoder,
    total_bytes: usize,
) {
    let mut budget = (total_bytes * 8) as i32;
    let mut tell = dec.tell();
    let mut logp = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    budget -= i32::from(tf_select_rsv);
    let mut curr = 0i32;
    let mut tf_changed = 0i32;
    for i in start..end {
        if tell + logp <= budget {
            curr ^= i32::from(dec.decode_bit_logp(logp as u32));
            tell = dec.tell();
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let row = &TF_SELECT_TABLE[lm * 8..][..8];
    let base = 4 * usize::from(is_transient);
    let mut tf_select = 0usize;
    if tf_select_rsv && row[base + tf_changed as usize] != row[base + 2 + tf_changed as usize] {
        tf_select = usize::from(dec.decode_bit_logp(1));
    }
    for i in start..end {
        tf_res[i] = i32::from(row[base + 2 * tf_select + tf_res[i] as usize]);
    }
}

/// Scale each band's unit-norm shape back up to the energy that was coded for
/// it. This is where the two halves of the codec come back together.
fn denormalise_bands(
    x: &[f32],
    freq: &mut [f32],
    band_loge: &[f32],
    start: usize,
    end: usize,
    m: usize,
    downsample: usize,
    silence: bool,
) {
    let n = m * SHORT_MDCT_SIZE;
    let mut bound = m * EBAND_5MS[end] as usize;
    if downsample != 1 {
        bound = bound.min(n / downsample);
    }
    let (start, end) = if silence {
        bound = 0;
        (0, 0)
    } else {
        (start, end)
    };
    for f in freq.iter_mut().take(m * EBAND_5MS[start] as usize) {
        *f = 0.0;
    }
    for i in start..end {
        let g = exp2_approx((band_loge[i] + E_MEANS[i]).min(32.0));
        for j in m * EBAND_5MS[i] as usize..m * EBAND_5MS[i + 1] as usize {
            freq[j] = x[j] * g;
        }
    }
    for f in freq[bound..n].iter_mut() {
        *f = 0.0;
    }
}

/// `2^x`. The energy is coded in base-2 log units, so this is the only place
/// the decoder needs an exponential, and it needs one that cannot overflow
/// on a corrupt band.
fn exp2_approx(x: f32) -> f32 {
    if x <= -128.0 {
        0.0
    } else {
        (x * core::f32::consts::LN_2).exp()
    }
}

/// Refill blocks that a transient left with no pulses at all.
///
/// Without this, a short block that got nothing decodes to silence between
/// two loud ones, which is heard as a rattle rather than as quiet.
#[allow(clippy::too_many_arguments)]
fn anti_collapse(
    x: &mut [f32],
    collapse_masks: &[u8],
    lm: usize,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize;
        // Depth in eighths of a bit.
        let depth = ((1 + pulses[i]) / (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32) >> lm;
        let thresh = 0.5 * exp2_approx(-0.125 * depth as f32);
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();

        for c in 0..channels {
            let mut prev1 = prev1_log_e[c * NB_EBANDS + i];
            let mut prev2 = prev2_log_e[c * NB_EBANDS + i];
            if channels == 1 {
                prev1 = prev1.max(prev1_log_e[NB_EBANDS + i]);
                prev2 = prev2.max(prev2_log_e[NB_EBANDS + i]);
            }
            let ediff = (log_e[c * NB_EBANDS + i] - prev1.min(prev2)).max(0.0);
            // Short blocks carry less energy than long ones, so the noise
            // that replaces a collapsed one has to be scaled up to match.
            let mut r = 2.0 * exp2_approx(-ediff);
            if lm == 3 {
                r *= 1.41421356;
            }
            r = r.min(thresh) * sqrt_1;

            let base = c * size + ((EBAND_5MS[i] as usize) << lm);
            let mut renormalize = false;
            for k in 0..1usize << lm {
                if collapse_masks[i * channels + c] & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = lcg_rand(seed);
                        x[base + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x[base..base + (n0 << lm)], 1.0);
            }
        }
    }
}

/// The pitch postfilter, run over the synthesis to put back the harmonic
/// structure the MDCT smeared. `t0`/`g0` are the previous frame's period and
/// gain, cross-faded into `t1`/`g1` over the overlap.
#[allow(clippy::too_many_arguments)]
fn comb_filter(
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
    // A zero gain leaves its period unset, and a period of zero would read
    // whatever is in front of the buffer.
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
    // No change means no cross-fade to do.
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

/// The same filter with no cross-fade, reading one buffer and writing
/// another. Only the concealment path needs this shape.
fn comb_filter_const(
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

/// Undo the encoder's pre-emphasis and hand the result out as PCM.
///
/// The filter has state across frames, so this is also what makes a decoder
/// that skipped a frame sound different from one that did not.
fn deemphasis(
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

/// Levinson-Durbin: turn an autocorrelation into the LPC filter that whitens
/// it. Concealment works in the excitation domain, and this is the filter
/// that gets it there and back.
fn celt_lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    lpc[..p].fill(0.0);
    if ac[0] <= 1e-10 {
        return;
    }
    let mut error = ac[0];
    for i in 0..p {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }
        error -= r * r * error;
        // Thirty dB of prediction gain is as much as this is worth.
        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

/// Autocorrelation over `x`, tapered at both ends by `window` so the estimate
/// is not dominated by the discontinuity at the edges.
fn celt_autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
) {
    let windowed;
    let src: &[f32] = match window {
        None => &x[..n],
        Some(w) => {
            let mut tmp = x[..n].to_vec();
            for i in 0..overlap {
                tmp[i] = x[i] * w[i];
                tmp[n - i - 1] = x[n - i - 1] * w[i];
            }
            windowed = tmp;
            &windowed
        }
    };
    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in k..n {
            d += src[i] * src[i - k];
        }
        ac[k] = d;
    }
}

/// FIR: run the LPC analysis filter to get the excitation.
fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = x[ord + i];
        for j in 0..ord {
            sum += num[j] * x[ord + i - j - 1];
        }
        y[i] = sum;
    }
}

/// IIR: run the LPC synthesis filter to turn an excitation back into signal.
fn celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        y[i] = sum;
    }
}

/// Halve the sample rate and whiten, which is what the pitch search actually
/// runs on: at 24 kHz the correlation peak is just as sharp and costs a
/// quarter as much to find.
fn pitch_downsample(channels: &[Vec<f32>], x_lp: &mut [f32], len: usize, cc: usize) {
    for i in 1..len >> 1 {
        x_lp[i] = 0.25 * channels[0][2 * i - 1]
            + 0.25 * channels[0][2 * i + 1]
            + 0.5 * channels[0][2 * i];
    }
    x_lp[0] = 0.25 * channels[0][1] + 0.5 * channels[0][0];
    if cc == 2 {
        for i in 1..len >> 1 {
            x_lp[i] += 0.25 * channels[1][2 * i - 1]
                + 0.25 * channels[1][2 * i + 1]
                + 0.5 * channels[1][2 * i];
        }
        x_lp[0] += 0.25 * channels[1][1] + 0.5 * channels[1][0];
    }

    let mut ac = [0.0f32; 5];
    celt_autocorr(x_lp, &mut ac, None, 0, 4, len >> 1);
    // A noise floor 40 dB down, and lag windowing, so the recursion below
    // cannot produce a filter that rings.
    ac[0] *= 1.0001;
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut lpc = [0.0f32; 4];
    celt_lpc(&mut lpc, &ac, 4);
    let mut tmp = 1.0f32;
    for coef in lpc.iter_mut() {
        tmp *= 0.9;
        *coef *= tmp;
    }
    // Add a zero at 0.8, which flattens the spectrum further.
    let c1 = 0.8f32;
    let lpc2 = [
        lpc[0] + 0.8,
        lpc[1] + c1 * lpc[0],
        lpc[2] + c1 * lpc[1],
        lpc[3] + c1 * lpc[2],
        c1 * lpc[3],
    ];
    let mut mem = [0.0f32; 5];
    for i in 0..len >> 1 {
        let mut sum = x_lp[i];
        for j in 0..5 {
            sum += lpc2[j] * mem[j];
        }
        for j in (1..5).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = x_lp[i];
        x_lp[i] = sum;
    }
}

/// The two best normalised correlation peaks, and where they are.
fn find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [usize; 2],
) {
    let mut syy = 1.0f32;
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    best_pitch[0] = 0;
    best_pitch[1] = 1;
    for j in 0..len {
        syy += y[j] * y[j];
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            // Scaled down before squaring so the product cannot overflow.
            let x16 = xcorr[i] * 1e-12;
            let num = x16 * x16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
}

/// Find the pitch period, coarsely at a quarter rate and then refined.
fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    let x_lp4: Vec<f32> = (0..len >> 2).map(|j| x_lp[2 * j]).collect();
    let y_lp4: Vec<f32> = (0..lag >> 2).map(|j| y[2 * j]).collect();

    let mut xcorr = vec![0.0f32; max_pitch >> 1];
    for i in 0..max_pitch >> 2 {
        xcorr[i] = (0..len >> 2).map(|j| x_lp4[j] * y_lp4[i + j]).sum();
    }
    let mut best_pitch = [0usize; 2];
    find_best_pitch(
        &xcorr[..max_pitch >> 2],
        &y_lp4,
        len >> 2,
        max_pitch >> 2,
        &mut best_pitch,
    );

    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0.0;
        let d0 = (i as i32 - 2 * best_pitch[0] as i32).abs();
        let d1 = (i as i32 - 2 * best_pitch[1] as i32).abs();
        if d0 > 2 && d1 > 2 {
            continue;
        }
        let sum: f32 = (0..len >> 1).map(|j| x_lp[j] * y[i + j]).sum();
        xcorr[i] = sum.max(-1.0);
    }
    find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1, &mut best_pitch);

    // Pseudo-interpolation between the peak and its neighbours.
    let offset = if best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if c - a > 0.7 * (b - a) {
            1i32
        } else if a - c > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best_pitch[0] as i32 - offset) as usize
}

/// Turn the decoded, denormalised spectrum back into time.
#[allow(clippy::too_many_arguments)]
fn celt_synthesis(
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
        // One coded channel played out of two.
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
        // Two coded channels played out of one.
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

/// One CELT decoder: the mode's tables, and everything that has to survive
/// from one frame to the next.
pub(super) struct CeltDecoder {
    mode: Mode,
    /// How many channels are played out.
    channels: usize,
    /// How many channels this frame actually codes, which may be fewer.
    pub(super) stream_channels: usize,
    /// 48000 / output rate: CELT always decodes at 48 kHz and decimates.
    pub(super) downsample: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    disable_inv: bool,
    pub(super) rng: u32,
    last_pitch_index: usize,
    loss_duration: i32,
    skip_plc: bool,
    postfilter_period: usize,
    postfilter_period_old: usize,
    postfilter_gain: f32,
    postfilter_gain_old: f32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    preemph_mem: [f32; 2],
    decode_mem: Vec<Vec<f32>>,
    lpc: Vec<[f32; LPC_ORDER]>,
    old_bande: Vec<f32>,
    old_loge: Vec<f32>,
    old_loge2: Vec<f32>,
    background_loge: Vec<f32>,
}

impl CeltDecoder {
    pub(super) fn new(channels: usize) -> Self {
        let mut dec = CeltDecoder {
            mode: Mode::new(),
            channels,
            stream_channels: channels,
            downsample: 1,
            start: 0,
            end: EFF_EBANDS,
            // Mono never inverts the side, because there is no side.
            disable_inv: channels == 1,
            rng: 0,
            last_pitch_index: 0,
            loss_duration: 0,
            skip_plc: true,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            preemph_mem: [0.0; 2],
            decode_mem: vec![vec![0.0; DECODE_BUFFER_SIZE + OVERLAP]; channels],
            lpc: vec![[0.0; LPC_ORDER]; channels],
            old_bande: vec![0.0; 2 * NB_EBANDS],
            old_loge: vec![-28.0; 2 * NB_EBANDS],
            old_loge2: vec![-28.0; 2 * NB_EBANDS],
            background_loge: vec![0.0; 2 * NB_EBANDS],
        };
        dec.reset();
        dec
    }

    pub(super) fn reset(&mut self) {
        self.rng = 0;
        self.last_pitch_index = 0;
        self.loss_duration = 0;
        self.skip_plc = true;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0.0;
        self.postfilter_gain_old = 0.0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.preemph_mem = [0.0; 2];
        for mem in self.decode_mem.iter_mut() {
            mem.fill(0.0);
        }
        for lpc in self.lpc.iter_mut() {
            lpc.fill(0.0);
        }
        self.old_bande.fill(0.0);
        self.old_loge.fill(-28.0);
        self.old_loge2.fill(-28.0);
        self.background_loge.fill(0.0);
    }

    pub(super) fn set_channels(&mut self, stream_channels: usize) {
        self.stream_channels = stream_channels;
    }

    /// Conceal one lost frame.
    ///
    /// Two strategies: repeat the last pitch period through the LPC synthesis
    /// filter, which holds a voiced sound together, or fill the bands with
    /// noise at the energy the signal was decaying towards. The second is
    /// what a long loss ends in either way, a pitch repeated for half a
    /// second is a tone, not concealment.
    fn decode_lost(&mut self, n: usize, lm: usize) {
        let cc = self.channels;
        let at = DECODE_BUFFER_SIZE - n;
        let loss_duration = self.loss_duration;
        let start = self.start;
        let noise_based = loss_duration >= 40 || start != 0 || self.skip_plc;

        if noise_based {
            let end = self.end;
            let eff_end = start.max(end.min(EFF_EBANDS));
            let mut x = vec![0.0f32; cc * n];
            for mem in self.decode_mem.iter_mut() {
                mem.copy_within(n..DECODE_BUFFER_SIZE + (OVERLAP >> 1), 0);
            }
            // Decay towards the background estimate rather than to silence.
            let decay = if loss_duration == 0 { 1.5 } else { 0.5 };
            for c in 0..cc {
                for i in start..end {
                    let idx = c * NB_EBANDS + i;
                    self.old_bande[idx] =
                        self.background_loge[idx].max(self.old_bande[idx] - decay);
                }
            }
            let mut seed = self.rng;
            for c in 0..cc {
                for i in start..eff_end {
                    let boffs = n * c + ((EBAND_5MS[i] as usize) << lm);
                    let blen = ((EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize) << lm;
                    for j in 0..blen {
                        seed = lcg_rand(seed);
                        x[boffs + j] = ((seed as i32) >> 20) as f32;
                    }
                    renormalise_vector(&mut x[boffs..boffs + blen], 1.0);
                }
            }
            self.rng = seed;
            celt_synthesis(
                &mut self.mode.mdct,
                &x,
                &mut self.decode_mem,
                at,
                &self.old_bande,
                start,
                eff_end,
                cc,
                cc,
                false,
                lm,
                self.downsample,
                false,
            );
        } else {
            let pitch_index = if loss_duration == 0 {
                let mut lp = vec![0.0f32; DECODE_BUFFER_SIZE >> 1];
                pitch_downsample(&self.decode_mem, &mut lp, DECODE_BUFFER_SIZE, cc);
                let found = pitch_search(
                    &lp[PLC_PITCH_LAG_MAX >> 1..],
                    &lp,
                    DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
                    PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
                );
                self.last_pitch_index = PLC_PITCH_LAG_MAX - found;
                self.last_pitch_index
            } else {
                self.last_pitch_index
            };
            let fade = if loss_duration == 0 { 1.0f32 } else { 0.8 };
            // Two pitch periods, so a decaying signal can be recognised as
            // one and not have energy added back.
            let exc_length = (2 * pitch_index).min(MAX_PERIOD);

            for c in 0..cc {
                let mut exc = vec![0.0f32; MAX_PERIOD + LPC_ORDER];
                for i in 0..MAX_PERIOD + LPC_ORDER {
                    exc[i] = self.decode_mem[c][DECODE_BUFFER_SIZE - MAX_PERIOD - LPC_ORDER + i];
                }
                if loss_duration == 0 {
                    let mut ac = [0.0f32; LPC_ORDER + 1];
                    celt_autocorr(
                        &exc[LPC_ORDER..],
                        &mut ac,
                        Some(&WINDOW120),
                        OVERLAP,
                        LPC_ORDER,
                        MAX_PERIOD,
                    );
                    ac[0] *= 1.0001;
                    for i in 1..=LPC_ORDER {
                        ac[i] -= ac[i] * (0.008 * 0.008) * (i * i) as f32;
                    }
                    celt_lpc(&mut self.lpc[c], &ac, LPC_ORDER);
                }
                let mut fir_tmp = vec![0.0f32; exc_length];
                celt_fir(
                    &exc[LPC_ORDER + MAX_PERIOD - exc_length - LPC_ORDER..],
                    &self.lpc[c],
                    &mut fir_tmp,
                    exc_length,
                    LPC_ORDER,
                );
                exc[LPC_ORDER + MAX_PERIOD - exc_length..LPC_ORDER + MAX_PERIOD]
                    .copy_from_slice(&fir_tmp);

                // Is the waveform decaying, and how fast?
                let decay_length = exc_length >> 1;
                let mut e1 = 1.0f32;
                let mut e2 = 1.0f32;
                for i in 0..decay_length {
                    let a = exc[LPC_ORDER + MAX_PERIOD - decay_length + i];
                    let b = exc[LPC_ORDER + MAX_PERIOD - 2 * decay_length + i];
                    e1 += a * a;
                    e2 += b * b;
                }
                let decay = (e1.min(e2) / e2).sqrt();

                self.decode_mem[c].copy_within(n..DECODE_BUFFER_SIZE, 0);

                // Repeat the last period, decaying a little more each time.
                let extrapolation_offset = MAX_PERIOD - pitch_index;
                let extrapolation_len = n + OVERLAP;
                let mut attenuation = fade * decay;
                let mut s1 = 0.0f32;
                let mut j = 0usize;
                for i in 0..extrapolation_len {
                    if j >= pitch_index {
                        j -= pitch_index;
                        attenuation *= decay;
                    }
                    self.decode_mem[c][at + i] =
                        attenuation * exc[LPC_ORDER + extrapolation_offset + j];
                    let tmp = self.decode_mem[c]
                        [DECODE_BUFFER_SIZE - MAX_PERIOD - n + extrapolation_offset + j];
                    s1 += tmp * tmp;
                    j += 1;
                }
                {
                    let mut lpc_mem = [0.0f32; LPC_ORDER];
                    for i in 0..LPC_ORDER {
                        lpc_mem[i] = self.decode_mem[c][DECODE_BUFFER_SIZE - n - 1 - i];
                    }
                    let input: Vec<f32> = self.decode_mem[c][at..at + extrapolation_len].to_vec();
                    let mut out = vec![0.0f32; extrapolation_len];
                    celt_iir(
                        &input,
                        &self.lpc[c],
                        &mut out,
                        extrapolation_len,
                        LPC_ORDER,
                        &mut lpc_mem,
                    );
                    self.decode_mem[c][at..at + extrapolation_len].copy_from_slice(&out);
                }

                // The synthesis filter can ring. Written as a negated
                // greater-than rather than a less-or-equal so that a NaN out
                // of the filter fails the test and is silenced too.
                let mut s2 = 0.0f32;
                for i in 0..extrapolation_len {
                    let tmp = self.decode_mem[c][at + i];
                    s2 += tmp * tmp;
                }
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(s1 > 0.2 * s2) {
                    for i in 0..extrapolation_len {
                        self.decode_mem[c][at + i] = 0.0;
                    }
                } else if s1 < s2 {
                    let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                    for i in 0..OVERLAP {
                        let g = 1.0 - WINDOW120[i] * (1.0 - ratio);
                        self.decode_mem[c][at + i] *= g;
                    }
                    for i in OVERLAP..extrapolation_len {
                        self.decode_mem[c][at + i] *= ratio;
                    }
                }

                // Pre-filter the overlap the next frame will post-filter, so
                // the two blend the way an uninterrupted stream would.
                let mut etmp = vec![0.0f32; OVERLAP];
                comb_filter_const(
                    &mut etmp,
                    &self.decode_mem[c],
                    DECODE_BUFFER_SIZE,
                    self.postfilter_period,
                    OVERLAP,
                    -self.postfilter_gain,
                    self.postfilter_tapset,
                );
                for i in 0..OVERLAP / 2 {
                    self.decode_mem[c][DECODE_BUFFER_SIZE + i] =
                        WINDOW120[i] * etmp[OVERLAP - 1 - i] + WINDOW120[OVERLAP - i - 1] * etmp[i];
                }
            }
        }
        self.loss_duration = 10000.min(loss_duration + (1 << lm));
    }

    /// Decode one CELT frame, or conceal one if `dec` is `None`.
    pub(super) fn decode(
        &mut self,
        dec: Option<&mut RangeDecoder>,
        len: usize,
        pcm: &mut [f32],
        frame_size: usize,
    ) -> usize {
        let cc = self.channels;
        let frame_size = frame_size * self.downsample;
        let mut lm = 0usize;
        while lm <= MAX_LM {
            if SHORT_MDCT_SIZE << lm == frame_size {
                break;
            }
            lm += 1;
        }
        let n = SHORT_MDCT_SIZE << lm;
        let at = DECODE_BUFFER_SIZE - n;
        let start = self.start;
        let end = self.end;
        let eff_end = end.min(EFF_EBANDS);

        let dec = match dec {
            Some(dec) if len > 1 => dec,
            _ => {
                self.decode_lost(n, lm);
                deemphasis(
                    &self.decode_mem,
                    at,
                    pcm,
                    n,
                    cc,
                    self.downsample,
                    &mut self.preemph_mem,
                );
                return frame_size / self.downsample;
            }
        };

        // Only turn the pitch-based concealment on once two packets have
        // arrived in a row; one packet after a loss has no history to work
        // from that the loss did not invent.
        self.skip_plc = self.loss_duration != 0;

        let c = self.stream_channels;
        if c == 1 {
            for i in 0..NB_EBANDS {
                self.old_bande[i] = self.old_bande[i].max(self.old_bande[NB_EBANDS + i]);
            }
        }

        let total_bits = (len * 8) as i32;
        let mut tell = dec.tell();
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.decode_bit_logp(15)
        } else {
            false
        };
        if silence {
            // Pretend the rest of the frame was read.
            dec.skip_to_end(len);
            tell = total_bits;
        }

        let mut postfilter_gain = 0.0f32;
        let mut postfilter_pitch = 0usize;
        let mut postfilter_tapset = 0usize;
        if start == 0 && tell + 16 <= total_bits {
            if dec.decode_bit_logp(1) {
                let octave = dec.decode_uint(6);
                postfilter_pitch = ((16 << octave) + dec.decode_bits(4 + octave) - 1) as usize;
                let qg = dec.decode_bits(3);
                if dec.tell() + 2 <= total_bits {
                    postfilter_tapset = dec.decode_icdf(&TAPSET_ICDF, 2);
                }
                postfilter_gain = 0.09375 * (qg + 1) as f32;
            }
            tell = dec.tell();
        }

        let is_transient = if lm > 0 && tell + 3 <= total_bits {
            let t = dec.decode_bit_logp(3);
            tell = dec.tell();
            t
        } else {
            false
        };

        let intra_ener = tell + 3 <= total_bits && dec.decode_bit_logp(3);
        unquant_coarse_energy(start, end, &mut self.old_bande, intra_ener, dec, c, lm, len);

        let mut tf_res = [0i32; NB_EBANDS];
        tf_decode(start, end, is_transient, &mut tf_res, lm, dec, len);

        let mut spread = SPREAD_NORMAL;
        if dec.tell() + 4 <= total_bits {
            spread = dec.decode_icdf(&SPREAD_ICDF, 5);
        }

        let mut cap = [0i32; NB_EBANDS];
        init_caps(&mut cap, lm, c);

        // Dynamic allocation: extra bits for bands the encoder found needed
        // them, coded as a run of increasingly cheap flags.
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_frac = total_bits << BITRES;
        let mut tell_frac = dec.tell_frac() as i32;
        for i in start..end {
            let width = (c * (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize) << lm;
            // Six bits at a time, but never more than one bit per sample nor
            // less than an eighth.
            let quanta = ((width << BITRES) as i32).min((6 << BITRES).max(width as i32));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                let flag = dec.decode_bit_logp(dynalloc_loop_logp as u32);
                tell_frac = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_frac -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
        }

        let alloc_trim = if tell_frac + (6 << BITRES) <= total_bits_frac {
            dec.decode_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        let mut bits = ((len as i32 * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;

        let mut pulses = [0i32; NB_EBANDS];
        let mut fine_quant = [0i32; NB_EBANDS];
        let mut fine_priority = [0i32; NB_EBANDS];
        let mut intensity = 0usize;
        let mut dual_stereo = false;
        let mut balance = 0i32;
        let coded_bands = compute_allocation(
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            c,
            lm,
            dec,
        );

        unquant_fine_energy(start, end, &mut self.old_bande, &fine_quant, dec, c);

        for mem in self.decode_mem.iter_mut() {
            mem.copy_within(n..DECODE_BUFFER_SIZE + OVERLAP / 2, 0);
        }

        let mut collapse_masks = vec![0u8; c * NB_EBANDS];
        let mut x = vec![0.0f32; c * n];
        let mut seed = self.rng;
        quant_all_bands(
            start,
            end,
            &mut x,
            c,
            &mut collapse_masks,
            &pulses,
            is_transient,
            spread,
            dual_stereo,
            intensity,
            &tf_res,
            (len as i32 * (8 << BITRES)) - anti_collapse_rsv,
            balance,
            dec,
            lm,
            coded_bands,
            &mut seed,
            self.disable_inv,
        );
        self.rng = seed;

        let anti_collapse_on = anti_collapse_rsv > 0 && dec.decode_bits(1) != 0;

        unquant_energy_finalise(
            start,
            end,
            &mut self.old_bande,
            &fine_quant,
            &fine_priority,
            len as i32 * 8 - dec.tell(),
            dec,
            c,
        );

        if anti_collapse_on {
            anti_collapse(
                &mut x,
                &collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &self.old_bande,
                &self.old_loge,
                &self.old_loge2,
                &pulses,
                self.rng,
            );
        }
        if silence {
            for v in self.old_bande.iter_mut() {
                *v = -28.0;
            }
        }

        celt_synthesis(
            &mut self.mode.mdct,
            &x,
            &mut self.decode_mem,
            at,
            &self.old_bande,
            start,
            eff_end,
            c,
            cc,
            is_transient,
            lm,
            self.downsample,
            silence,
        );

        for ch in 0..cc {
            self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
            self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
            comb_filter(
                &mut self.decode_mem[ch],
                at,
                self.postfilter_period_old,
                self.postfilter_period,
                SHORT_MDCT_SIZE,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                Some(&WINDOW120),
                OVERLAP,
            );
            if lm != 0 {
                comb_filter(
                    &mut self.decode_mem[ch],
                    at + SHORT_MDCT_SIZE,
                    self.postfilter_period,
                    postfilter_pitch,
                    n - SHORT_MDCT_SIZE,
                    self.postfilter_gain,
                    postfilter_gain,
                    self.postfilter_tapset,
                    postfilter_tapset,
                    Some(&WINDOW120),
                    OVERLAP,
                );
            }
        }
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        if c == 1 {
            let (lo, hi) = self.old_bande.split_at_mut(NB_EBANDS);
            hi[..NB_EBANDS].copy_from_slice(lo);
        }
        if !is_transient {
            self.old_loge2.copy_from_slice(&self.old_loge);
            self.old_loge.copy_from_slice(&self.old_bande);
        } else {
            for i in 0..2 * NB_EBANDS {
                self.old_loge[i] = self.old_loge[i].min(self.old_bande[i]);
            }
        }
        // The noise floor may rise by 2.4 dB a second normally; after a run
        // of lost packets it may make up all of what it missed at once.
        let max_background_increase = 160.min(self.loss_duration + (1 << lm)) as f32 * 0.001;
        for i in 0..2 * NB_EBANDS {
            self.background_loge[i] =
                (self.background_loge[i] + max_background_increase).min(self.old_bande[i]);
        }
        for ch in 0..2 {
            for i in 0..start {
                self.old_bande[ch * NB_EBANDS + i] = 0.0;
                self.old_loge[ch * NB_EBANDS + i] = -28.0;
                self.old_loge2[ch * NB_EBANDS + i] = -28.0;
            }
            for i in end..NB_EBANDS {
                self.old_bande[ch * NB_EBANDS + i] = 0.0;
                self.old_loge[ch * NB_EBANDS + i] = -28.0;
                self.old_loge2[ch * NB_EBANDS + i] = -28.0;
            }
        }
        self.rng = dec.rng();

        deemphasis(
            &self.decode_mem,
            at,
            pcm,
            n,
            cc,
            self.downsample,
            &mut self.preemph_mem,
        );
        self.loss_duration = 0;
        frame_size / self.downsample
    }
}
