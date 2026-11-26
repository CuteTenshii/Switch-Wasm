//! CELT's inverse MDCT, computed through an `N/4`-point forward complex FFT
//! with pre- and post-rotation.

use core::f32::consts::PI;

type Cpx = (f32, f32);

pub(super) struct Fft {
    n: usize,
    /// `exp(-i·2πk/n)` for `k` in `0..n`.
    twiddles: Vec<Cpx>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let twiddles = (0..n)
            .map(|k| {
                let phase = -2.0 * PI * k as f32 / n as f32;
                (phase.cos(), phase.sin())
            })
            .collect();
        Fft { n, twiddles }
    }

    fn forward(&self, input: &[Cpx], out: &mut [Cpx]) {
        self.recurse(input, 0, 1, out, self.n, 1);
    }

    /// One decimation-in-time level; `fstride` is the twiddle step for this level.
    fn recurse(
        &self,
        input: &[Cpx],
        offset: usize,
        stride: usize,
        out: &mut [Cpx],
        n: usize,
        fstride: usize,
    ) {
        if n == 1 {
            out[0] = input[offset];
            return;
        }
        let p = radix(n);
        let m = n / p;
        if m == 1 {
            for q in 0..p {
                out[q] = input[offset + q * stride];
            }
        } else {
            for q in 0..p {
                self.recurse(
                    input,
                    offset + q * stride,
                    stride * p,
                    &mut out[q * m..(q + 1) * m],
                    m,
                    fstride * p,
                );
            }
        }
        // `W_n^(q·j)` gathers the sub-transforms; `W_p^(q·t)` is `W_n` with step `n/p`.
        match p {
            2 => self.butterfly2(out, m, fstride),
            4 => self.butterfly4(out, m, fstride),
            _ => self.butterfly_generic(out, m, p, fstride),
        }
    }

    fn butterfly2(&self, out: &mut [Cpx], m: usize, fstride: usize) {
        for j in 0..m {
            let a = out[j];
            let b = cmul(out[m + j], self.twiddles[j * fstride]);
            out[j] = cadd(a, b);
            out[m + j] = csub(a, b);
        }
    }

    fn butterfly4(&self, out: &mut [Cpx], m: usize, fstride: usize) {
        for j in 0..m {
            let s0 = out[j];
            let s1 = cmul(out[m + j], self.twiddles[j * fstride]);
            let s2 = cmul(out[2 * m + j], self.twiddles[2 * j * fstride]);
            let s3 = cmul(out[3 * m + j], self.twiddles[3 * j * fstride]);
            let t0 = cadd(s0, s2);
            let t1 = csub(s0, s2);
            let t2 = cadd(s1, s3);
            let t3 = csub(s1, s3);
            out[j] = cadd(t0, t2);
            out[2 * m + j] = csub(t0, t2);
            // Odd outputs differ by a multiply by -i and +i.
            out[m + j] = (t1.0 + t3.1, t1.1 - t3.0);
            out[3 * m + j] = (t1.0 - t3.1, t1.1 + t3.0);
        }
    }

    fn butterfly_generic(&self, out: &mut [Cpx], m: usize, p: usize, fstride: usize) {
        let step_p = m * fstride;
        let mut scratch = [(0.0f32, 0.0f32); MAX_RADIX];
        for j in 0..m {
            for q in 0..p {
                scratch[q] = cmul(out[q * m + j], self.twiddles[q * j * fstride]);
            }
            for t in 0..p {
                let mut acc = scratch[0];
                for q in 1..p {
                    acc = cadd(acc, cmul(scratch[q], self.twiddles[(q * t) % p * step_p]));
                }
                out[j + t * m] = acc;
            }
        }
    }
}

const MAX_RADIX: usize = 5;

/// Radix-4 is preferred over two radix-2 levels.
fn radix(n: usize) -> usize {
    for p in [4, 2, 3, 5] {
        if n % p == 0 {
            return p;
        }
    }
    n
}

fn cmul(a: Cpx, b: Cpx) -> Cpx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cadd(a: Cpx, b: Cpx) -> Cpx {
    (a.0 + b.0, a.1 + b.1)
}

fn csub(a: Cpx, b: Cpx) -> Cpx {
    (a.0 - b.0, a.1 - b.1)
}

/// Inverse MDCT for every block size of one mode, selected by `shift`.
pub(super) struct Mdct {
    n: usize,
    spectrum: Vec<Cpx>,
    transformed: Vec<Cpx>,
    /// Per shift, `cos(2π(i+1/8)/N)` for `i` in `0..N/2`.
    trig: Vec<Vec<f32>>,
    ffts: Vec<Fft>,
}

impl Mdct {
    pub(super) fn new(n: usize, max_shift: usize) -> Self {
        let mut trig = Vec::with_capacity(max_shift + 1);
        let mut ffts = Vec::with_capacity(max_shift + 1);
        for shift in 0..=max_shift {
            let size = n >> shift;
            let half = size >> 1;
            trig.push(
                (0..half)
                    .map(|i| (2.0 * PI * (i as f32 + 0.125) / size as f32).cos())
                    .collect(),
            );
            ffts.push(Fft::new(size >> 2));
        }
        Mdct {
            n,
            spectrum: vec![(0.0, 0.0); n >> 2],
            transformed: vec![(0.0, 0.0); n >> 2],
            trig,
            ffts,
        }
    }

    /// `out[..overlap/2]` must hold the tail of the previous block; the final
    /// mirroring step is the overlap-add.
    pub(super) fn backward(
        &mut self,
        input: &[f32],
        out: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let size = self.n >> shift;
        let half = size >> 1;
        let quarter = size >> 2;
        let trig = &self.trig[shift];
        let base = overlap >> 1;

        let spectrum = &mut self.spectrum[..quarter];
        for i in 0..quarter {
            let x1 = input[2 * i * stride];
            let x2 = input[(half - 1 - 2 * i) * stride];
            let yr = x2 * trig[i] + x1 * trig[quarter + i];
            let yi = x1 * trig[i] - x2 * trig[quarter + i];
            // Swapped because a forward FFT stands in for an inverse one.
            spectrum[i] = (yi, yr);
        }
        let transformed = &mut self.transformed[..quarter];
        self.ffts[shift].forward(spectrum, transformed);

        for i in 0..(quarter + 1) >> 1 {
            let (im0, re0) = transformed[i];
            let (t0, t1) = (trig[i], trig[quarter + i]);
            let yr = re0 * t0 + im0 * t1;
            let yi = re0 * t1 - im0 * t0;

            let (im1, re1) = transformed[quarter - 1 - i];
            let (t2, t3) = (trig[quarter - i - 1], trig[half - i - 1]);
            let yr2 = re1 * t2 + im1 * t3;
            let yi2 = re1 * t3 - im1 * t2;

            out[base + 2 * i] = yr;
            out[base + half - 2 - 2 * i + 1] = yi;
            out[base + half - 2 - 2 * i] = yr2;
            out[base + 2 * i + 1] = yi2;
        }

        for i in 0..overlap / 2 {
            let x1 = out[overlap - 1 - i];
            let x2 = out[i];
            out[i] = window[overlap - 1 - i] * x2 - window[i] * x1;
            out[overlap - 1 - i] = window[i] * x2 + window[overlap - 1 - i] * x1;
        }
    }
}
