//! The implicit bit allocation.

use crate::opus::range::{RangeDecoder, BITRES};
use crate::opus::tables_celt::*;
use super::{ALLOC_STEPS, FINE_OFFSET, LOG_MAX_PSEUDO, MAX_FINE_BITS, NB_ALLOC_VECTORS, NB_EBANDS};

/// Pulses a pseudo-pulse count stands for; above 8 it is logarithmic.
pub(super) fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Cache row for a band and block size: `bits[0]` is the entry count,
/// `bits[q]` one less than the cost of `q` pseudo-pulses.
pub(super) fn cache_row(band: usize, lm: i32) -> &'static [u8] {
    let index = CACHE_INDEX50[((lm + 1) as usize) * NB_EBANDS + band];
    debug_assert!(
        index >= 0,
        "pulse cache row for a band with no coefficients"
    );
    &CACHE_BITS50[index.max(0) as usize..]
}

/// The largest pseudo-pulse count whose codebook fits in `bits`.
pub(super) fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
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

pub(super) fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        i32::from(cache_row(band, lm)[pulses as usize]) + 1
    }
}

/// The most bits a band can use before PVQ outresolves its energy.
pub(super) fn init_caps(cap: &mut [i32], lm: usize, channels: usize) {
    for i in 0..NB_EBANDS {
        let n = ((EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32) << lm;
        let row = NB_EBANDS * (2 * lm + channels - 1);
        cap[i] = ((i32::from(CACHE_CAPS50[row + i]) + 64) * channels as i32 * n) >> 2;
    }
}

/// Interpolate the allocation table rows, then split into fine-energy and PVQ
/// bits. Both ends bisect identically; only skips are transmitted.
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

    // Skip bands from the top, never the first or a dynalloc-boosted one.
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
        // Below the flag's cost a band is force-skipped silently.
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

            // Joint stereo costs bits for its extra degree of freedom.
            let den = channels as i32 * n
                + i32::from(channels == 2 && n > 2 && !*dual_stereo && j < *intensity);
            let nclogn = den * (i32::from(LOG_N400[j]) + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            // Bring the second and third fine bits forward.
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
            // Rounded-down bands are candidates for the final pass.
            fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= (channels as i32 * ebits[j]) << BITRES;
        } else {
            // One coefficient: all but the sign goes to fine energy.
            excess = 0.max(bit - ((channels as i32) << BITRES));
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }

        // Rebalance fine energy here, since band-decode rebalancing can't reach it.
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

    for j in coded_bands..end {
        ebits[j] = bits[j] >> u32::from(stereo) >> BITRES;
        bits[j] = 0;
        fine_priority[j] = i32::from(ebits[j] < 1);
    }
    coded_bands
}

/// Per-band bits from frame size, bandwidth, dynalloc boosts and trim.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_allocation(
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
        thresh[j] = ((channels as i32) << BITRES).max(((3 * width) << lm << BITRES) >> 4);
        trim_offset[j] = (channels as i32
            * width
            * (alloc_trim - 5 - lm as i32)
            * (end - j - 1) as i32
            * (1i32 << (lm + BITRES as usize)))
            >> 6;
        // Single-coefficient bands get less.
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
