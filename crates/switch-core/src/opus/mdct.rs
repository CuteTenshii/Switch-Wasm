//! The inverse MDCT the CELT layer synthesises through, and the complex FFT
//! underneath it.
//!
//! CELT does not run an IMDCT directly. It pre-rotates the spectrum by
//! `exp(-i·2π(k+1/8)/N)`, runs a *forward* complex FFT of `N/4` points over
//! the result, post-rotates, and then mirrors the two ends into each other —
//! which is where the time-domain alias cancellation that makes overlapping
//! blocks reconstruct exactly comes from. Doing it that way costs one
//! quarter-length complex transform instead of a real transform of length
//! `N`.
//!
//! The FFT here is an ordinary mixed-radix Cooley-Tukey, not the
//! bit-reversal-in-the-caller arrangement the reference uses. The reference
//! folds the permutation into the pre-rotation to keep the transform
//! in-place; writing the pre-rotation in natural order and letting the
//! recursion do its own reordering computes the same spectrum, and is a great
//! deal easier to be sure of.

use core::f32::consts::PI;

/// A complex value, as `(re, im)`.
type Cpx = (f32, f32);

/// A forward complex FFT of one fixed size, with its twiddles.
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

    /// `out[k] = sum_j input[j] · exp(-i·2πjk/n)`, unscaled.
    fn forward(&self, input: &[Cpx], out: &mut [Cpx]) {
        self.recurse(input, 0, 1, out, self.n, 1);
    }

    /// One decimation-in-time level: split `n` into `p` interleaved
    /// sub-transforms of `m = n/p` points, then combine them with a `p`-point
    /// DFT. `stride` walks the input, `fstride` is `self.n / n` — how far one
    /// step of this level's twiddle moves in the full table.
    fn recurse(&self, input: &[Cpx], offset: usize, stride: usize, out: &mut [Cpx], n: usize, fstride: usize) {
        if n == 1 {
            out[0] = input[offset];
            return;
        }
        let p = radix(n);
        let m = n / p;
        if m == 1 {
            // The leaves are single points, so gathering them here saves a
            // call per point — which at these sizes is most of the calls.
            for q in 0..p {
                out[q] = input[offset + q * stride];
            }
        } else {
            for q in 0..p {
                self.recurse(input, offset + q * stride, stride * p, &mut out[q * m..(q + 1) * m], m, fstride * p);
            }
        }
        // `W_n^(q·j)` gathers the sub-transforms; `W_p^(q·t)` is the p-point
        // DFT across them. Both come out of the same table: the second is the
        // first with a step of `n/p`. The gather index is always inside the
        // table — `q·j·fstride < p·m·fstride = self.n` — so it needs no wrap.
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
            // The odd outputs differ by a quarter turn, which for a forward
            // transform is a multiply by -i and by +i.
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

/// The largest radix [`radix`] will pick, which bounds the combine scratch.
const MAX_RADIX: usize = 5;

/// The factor to split `n` by. Four before two, because one radix-4 level
/// costs less than two radix-2 ones.
fn radix(n: usize) -> usize {
    for p in [4, 2, 3, 5] {
        if n % p == 0 {
            return p;
        }
    }
    // Every size CELT asks for factors into 2, 3 and 5; a prime that does not
    // is still handled correctly, just as one slow level.
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

/// The inverse MDCT for every block size one mode uses. `shift` selects the
/// size: `n >> shift` points in, half that many out.
pub(super) struct Mdct {
    n: usize,
    /// Scratch for one transform, kept so a frame of eight short blocks does
    /// not allocate twice per block.
    spectrum: Vec<Cpx>,
    transformed: Vec<Cpx>,
    /// Per shift, `cos(2π(i+1/8)/N)` for `i` in `0..N/2`. The eighth-sample
    /// offset is what makes the transform its own inverse across the overlap.
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
        Mdct { n, spectrum: vec![(0.0, 0.0); n >> 2], transformed: vec![(0.0, 0.0); n >> 2], trig, ffts }
    }

    /// Transform `input` — `n>>shift` halved, taken every `stride` values —
    /// into `out`, windowing the first `overlap` samples against what is
    /// already there.
    ///
    /// `out[..overlap/2]` must hold the tail of the previous block: the
    /// mirroring step at the end is the overlap-add, not a separate pass.
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
            // Real and imaginary are swapped because this is a forward FFT
            // standing in for an inverse one.
            spectrum[i] = (yi, yr);
        }
        let transformed = &mut self.transformed[..quarter];
        self.ffts[shift].forward(spectrum, transformed);

        // Post-rotate, walking in from both ends so the two halves land
        // de-shuffled without a second buffer.
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
