//! Motion compensated prediction from an unscaled reference frame.

use super::frame::Plane;
use super::tables::*;

const KERNELS: [&[[i16; 8]; 16]; 4] = [
    &FILTER_REGULAR,
    &FILTER_SMOOTH,
    &FILTER_SHARP,
    &FILTER_BILINEAR,
];

fn clip(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

/// The `w`x`h` prediction at plane position `(x0, y0)` displaced by `(mv_col, mv_row)` sixteenths of
/// a pixel, written to `dst` (or averaged into it for a second reference of a compound block).
pub fn predict(
    dst: &mut [u8],
    dst_pos: usize,
    dst_stride: usize,
    reference: &Plane,
    (x0, y0): (i32, i32),
    (mv_col, mv_row): (i32, i32),
    (w, h): (usize, usize),
    interp_filter: u8,
    average: bool,
) {
    let kernel = KERNELS[usize::from(interp_filter)];
    let (subpel_x, subpel_y) = ((mv_col & 15) as usize, (mv_row & 15) as usize);
    let (px, py) = (x0 + (mv_col >> 4), y0 + (mv_row >> 4));
    let (max_x, max_y) = (reference.width as i32 - 1, reference.height as i32 - 1);

    let patch;
    let inside = px >= 3 && py >= 3 && px + w as i32 + 3 <= max_x && py + h as i32 + 3 <= max_y;
    let (src, base, stride) = if inside {
        let base = (py - 3) as usize * reference.stride + (px - 3) as usize;
        (&reference.data[..], base, reference.stride)
    } else {
        // The filter's three pixels before and four after, edge pixels repeated outside the frame.
        let patch_w = w + 7;
        let mut p = vec![0u8; patch_w * (h + 7)];
        for y in 0..h + 7 {
            let ry = (py - 3 + y as i32).clamp(0, max_y) as usize;
            let row = &reference.data[ry * reference.stride..];
            for x in 0..patch_w {
                p[y * patch_w + x] = row[(px - 3 + x as i32).clamp(0, max_x) as usize];
            }
        }
        patch = p;
        (&patch[..], 0, patch_w)
    };

    let mut emit = |y: usize, fill: &mut dyn FnMut(&mut [u8])| {
        let d = &mut dst[dst_pos + y * dst_stride..][..w];
        if average {
            let mut row = [0u8; 64];
            fill(&mut row[..w]);
            for (d, &p) in d.iter_mut().zip(&row) {
                *d = ((u16::from(*d) + u16::from(p) + 1) >> 1) as u8;
            }
        } else {
            fill(d);
        }
    };
    let row_at = |y: usize, x: usize| &src[base + y * stride + x..][..w];

    if subpel_x == 0 && subpel_y == 0 {
        for y in 0..h {
            emit(y, &mut |o| o.copy_from_slice(row_at(y + 3, 3)));
        }
    } else if subpel_y == 0 {
        let f = &kernel[subpel_x];
        for y in 0..h {
            emit(y, &mut |o| {
                convolve(std::array::from_fn(|k| row_at(y + 3, k)), f, o)
            });
        }
    } else if subpel_x == 0 {
        let f = &kernel[subpel_y];
        for y in 0..h {
            emit(y, &mut |o| {
                convolve(std::array::from_fn(|k| row_at(y + k, 3)), f, o)
            });
        }
    } else {
        let fx = &kernel[subpel_x];
        let fy = &kernel[subpel_y];
        let mut temp = [0u8; 64 * 71];
        for y in 0..h + 7 {
            convolve(
                std::array::from_fn(|k| row_at(y, k)),
                fx,
                &mut temp[y * 64..][..w],
            );
        }
        for y in 0..h {
            emit(y, &mut |o| {
                convolve(std::array::from_fn(|k| &temp[(y + k) * 64..][..w]), fy, o)
            });
        }
    }
}

/// One output row of an 8-tap filter: tap `k` of every pixel comes from `taps[k]`.
fn convolve(taps: [&[u8]; 8], f: &[i16; 8], out: &mut [u8]) {
    match out.len() {
        4 => convolve_fixed::<4>(taps, f, out),
        8 => convolve_fixed::<8>(taps, f, out),
        16 => convolve_fixed::<16>(taps, f, out),
        32 => convolve_fixed::<32>(taps, f, out),
        64 => convolve_fixed::<64>(taps, f, out),
        _ => {
            for (x, o) in out.iter_mut().enumerate() {
                let sum: i32 = (0..8)
                    .map(|k| i32::from(taps[k][x]) * i32::from(f[k]))
                    .sum();
                *o = clip((sum + 64) >> 7);
            }
        }
    }
}

fn convolve_fixed<const W: usize>(taps: [&[u8]; 8], f: &[i16; 8], out: &mut [u8]) {
    let mut acc = [64i32; W];
    for (tap, &c) in taps.iter().zip(f) {
        let tap: &[u8; W] = tap[..W].try_into().unwrap();
        let c = i32::from(c);
        for x in 0..W {
            acc[x] += c * i32::from(tap[x]);
        }
    }
    let out: &mut [u8; W] = (&mut out[..W]).try_into().unwrap();
    for x in 0..W {
        out[x] = clip(acc[x] >> 7);
    }
}
