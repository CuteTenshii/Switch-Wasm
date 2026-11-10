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

    // The reference block with the filter's three pixels before and four after, edge pixels
    // repeated outside the frame.
    let patch_w = w + 7;
    let patch_h = h + 7;
    let mut patch = [0u8; 71 * 71];
    for y in 0..patch_h {
        let ry = (py - 3 + y as i32).clamp(0, max_y) as usize;
        let row = &reference.data[ry * reference.stride..];
        for x in 0..patch_w {
            let rx = (px - 3 + x as i32).clamp(0, max_x) as usize;
            patch[y * patch_w + x] = row[rx];
        }
    }

    let mut out = [0u8; 64 * 64];
    let at = |x: usize, y: usize| i32::from(patch[y * patch_w + x]);
    if subpel_x == 0 && subpel_y == 0 {
        for y in 0..h {
            for x in 0..w {
                out[y * 64 + x] = at(x + 3, y + 3) as u8;
            }
        }
    } else if subpel_y == 0 {
        let f = &kernel[subpel_x];
        for y in 0..h {
            for x in 0..w {
                let sum: i32 = (0..8).map(|k| at(x + k, y + 3) * i32::from(f[k])).sum();
                out[y * 64 + x] = clip((sum + 64) >> 7);
            }
        }
    } else if subpel_x == 0 {
        let f = &kernel[subpel_y];
        for y in 0..h {
            for x in 0..w {
                let sum: i32 = (0..8).map(|k| at(x + 3, y + k) * i32::from(f[k])).sum();
                out[y * 64 + x] = clip((sum + 64) >> 7);
            }
        }
    } else {
        let fx = &kernel[subpel_x];
        let fy = &kernel[subpel_y];
        let mut temp = [0u8; 64 * 71];
        for y in 0..h + 7 {
            for x in 0..w {
                let sum: i32 = (0..8).map(|k| at(x + k, y) * i32::from(fx[k])).sum();
                temp[y * 64 + x] = clip((sum + 64) >> 7);
            }
        }
        for y in 0..h {
            for x in 0..w {
                let sum: i32 = (0..8)
                    .map(|k| i32::from(temp[(y + k) * 64 + x]) * i32::from(fy[k]))
                    .sum();
                out[y * 64 + x] = clip((sum + 64) >> 7);
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            let d = &mut dst[dst_pos + y * dst_stride + x];
            let p = out[y * 64 + x];
            *d = if average {
                ((u16::from(*d) + u16::from(p) + 1) >> 1) as u8
            } else {
                p
            };
        }
    }
}
