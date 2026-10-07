//! LPC and pitch analysis for concealment.

/// Levinson-Durbin: autocorrelation to whitening LPC filter, for concealment.
pub(super) fn celt_lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
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
        // Stop at 30 dB of prediction gain.
        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

/// Windowed autocorrelation.
pub(super) fn celt_autocorr(
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

/// LPC analysis filter.
pub(super) fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = x[ord + i];
        for j in 0..ord {
            sum += num[j] * x[ord + i - j - 1];
        }
        y[i] = sum;
    }
}

/// LPC synthesis filter.
pub(super) fn celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
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

/// Halve the rate and whiten for the pitch search.
pub(super) fn pitch_downsample(channels: &[Vec<f32>], x_lp: &mut [f32], len: usize, cc: usize) {
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
    // 40 dB noise floor and lag windowing so the filter can't ring.
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
            // Scaled before squaring to avoid overflow.
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

/// Pitch period, coarse at quarter rate then refined.
pub(super) fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
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
