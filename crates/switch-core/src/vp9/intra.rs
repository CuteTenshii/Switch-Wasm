//! Intra prediction of one transform block from its reconstructed neighbours.

use super::types::*;

const NEED_LEFT: u8 = 1 << 1;
const NEED_ABOVE: u8 = 1 << 2;
const NEED_ABOVE_RIGHT: u8 = 1 << 3;

const EXTEND_MODES: [u8; 10] = [
    NEED_ABOVE | NEED_LEFT,
    NEED_ABOVE,
    NEED_LEFT,
    NEED_ABOVE_RIGHT,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT,
    NEED_ABOVE_RIGHT,
    NEED_LEFT | NEED_ABOVE,
];

/// The row above (with the corner one before it) and the column to the left of a block.
pub struct Edges {
    above: [u8; ABOVE_BASE + 64],
    left: [u8; 32],
}

const ABOVE_BASE: usize = 16;

impl Edges {
    fn above(&self, i: isize) -> i32 {
        i32::from(self.above[(ABOVE_BASE as isize + i) as usize])
    }

    fn left(&self, i: usize) -> i32 {
        i32::from(self.left[i])
    }
}

/// What `predict` needs to know about the block's surroundings.
pub struct Neighbours {
    pub have_top: bool,
    pub have_left: bool,
    pub have_right: bool,
    /// Position of the block in its plane, in pixels.
    pub x0: i32,
    pub y0: i32,
    /// The plane's size rounded up to a multiple of 8.
    pub frame_width: i32,
    pub frame_height: i32,
    /// The block's macroblock reaches past the right or bottom of the frame.
    pub past_right: bool,
    pub past_bottom: bool,
}

fn avg2(a: i32, b: i32) -> u8 {
    ((a + b + 1) >> 1) as u8
}

fn avg3(a: i32, b: i32, c: i32) -> u8 {
    ((a + 2 * b + c + 2) >> 2) as u8
}

/// Gathers the edges of the block at `pos` in `plane` as `build_intra_predictors` does.
fn edges(plane: &[u8], pos: usize, stride: usize, mode: u8, tx_size: u8, n: &Neighbours) -> Edges {
    let bs = 4usize << tx_size;
    let mut e = Edges {
        above: [0; ABOVE_BASE + 64],
        left: [0; 32],
    };
    let need = EXTEND_MODES[usize::from(mode)];
    let pix = |offset: isize| plane[(pos as isize + offset) as usize];
    let at = ABOVE_BASE;
    let (x0, y0) = (n.x0, n.y0);
    let bs_i = bs as i32;

    if need & NEED_LEFT != 0 {
        if n.have_left {
            if n.past_bottom && y0 + bs_i > n.frame_height {
                let extend = (n.frame_height - y0) as usize;
                for i in 0..bs {
                    let row = i.min(extend - 1);
                    e.left[i] = pix(row as isize * stride as isize - 1);
                }
            } else {
                for i in 0..bs {
                    e.left[i] = pix(i as isize * stride as isize - 1);
                }
            }
        } else {
            e.left[..bs].fill(129);
        }
    }

    let above_ref = -(stride as isize);
    let copy = |e: &mut Edges, count: usize| {
        for i in 0..count {
            e.above[at + i] = pix(above_ref + i as isize);
        }
    };
    let fill = |e: &mut Edges, from: usize, count: usize| {
        let v = e.above[at + from - 1];
        e.above[at + from..at + from + count].fill(v);
    };

    if need & NEED_ABOVE != 0 {
        if n.have_top {
            if n.past_right {
                if x0 + bs_i <= n.frame_width {
                    copy(&mut e, bs);
                } else if x0 <= n.frame_width {
                    let r = (n.frame_width - x0) as usize;
                    copy(&mut e, r);
                    fill(&mut e, r, bs - r);
                }
            } else {
                copy(&mut e, bs);
            }
            e.above[at - 1] = if n.have_left { pix(above_ref - 1) } else { 129 };
        } else {
            e.above[at..at + bs].fill(127);
            e.above[at - 1] = 127;
        }
    }

    if need & NEED_ABOVE_RIGHT != 0 {
        if n.have_top {
            if n.past_right {
                if x0 + 2 * bs_i <= n.frame_width {
                    if n.have_right && bs == 4 {
                        copy(&mut e, 2 * bs);
                    } else {
                        copy(&mut e, bs);
                        fill(&mut e, bs, bs);
                    }
                } else if x0 + bs_i <= n.frame_width {
                    let r = (n.frame_width - x0) as usize;
                    if n.have_right && bs == 4 {
                        copy(&mut e, r);
                        fill(&mut e, r, 2 * bs - r);
                    } else {
                        copy(&mut e, bs);
                        fill(&mut e, bs, bs);
                    }
                } else if x0 <= n.frame_width {
                    let r = (n.frame_width - x0) as usize;
                    copy(&mut e, r);
                    fill(&mut e, r, 2 * bs - r);
                }
            } else {
                copy(&mut e, bs);
                if bs == 4 && n.have_right {
                    for i in 0..bs {
                        e.above[at + bs + i] = pix(above_ref + (bs + i) as isize);
                    }
                } else {
                    fill(&mut e, bs, bs);
                }
            }
            e.above[at - 1] = if n.have_left { pix(above_ref - 1) } else { 129 };
        } else {
            e.above[at..at + 2 * bs].fill(127);
            e.above[at - 1] = 127;
        }
    }
    e
}

/// Writes the `mode` prediction of the `4 << tx_size` square block at `pos`.
pub fn predict(plane: &mut [u8], pos: usize, stride: usize, mode: u8, tx_size: u8, n: &Neighbours) {
    let e = edges(plane, pos, stride, mode, tx_size, n);
    predict_from(
        plane,
        pos,
        stride,
        mode,
        tx_size,
        &e,
        n.have_left,
        n.have_top,
    );
}

pub(super) fn predict_from(
    plane: &mut [u8],
    pos: usize,
    stride: usize,
    mode: u8,
    tx_size: u8,
    e: &Edges,
    have_left: bool,
    have_top: bool,
) {
    let bs = 4usize << tx_size;
    let mut put = |x: usize, y: usize, v: u8| plane[pos + y * stride + x] = v;
    match mode {
        DC_PRED => {
            let above_sum: i32 = (0..bs).map(|i| e.above(i as isize)).sum();
            let left_sum: i32 = (0..bs).map(|i| e.left(i)).sum();
            let bsi = bs as i32;
            let dc = match (have_left, have_top) {
                (true, true) => (above_sum + left_sum + bsi) / (2 * bsi),
                (true, false) => (left_sum + (bsi >> 1)) / bsi,
                (false, true) => (above_sum + (bsi >> 1)) / bsi,
                (false, false) => 128,
            } as u8;
            for y in 0..bs {
                for x in 0..bs {
                    put(x, y, dc);
                }
            }
        }
        V_PRED => {
            for y in 0..bs {
                for x in 0..bs {
                    put(x, y, e.above(x as isize) as u8);
                }
            }
        }
        H_PRED => {
            for y in 0..bs {
                for x in 0..bs {
                    put(x, y, e.left(y) as u8);
                }
            }
        }
        TM_PRED => {
            let corner = e.above(-1);
            for y in 0..bs {
                for x in 0..bs {
                    put(
                        x,
                        y,
                        (e.left(y) + e.above(x as isize) - corner).clamp(0, 255) as u8,
                    );
                }
            }
        }
        D45_PRED if bs == 4 => {
            let a = |i| e.above(i);
            let v = [
                avg3(a(0), a(1), a(2)),
                avg3(a(1), a(2), a(3)),
                avg3(a(2), a(3), a(4)),
                avg3(a(3), a(4), a(5)),
                avg3(a(4), a(5), a(6)),
                avg3(a(5), a(6), a(7)),
            ];
            for y in 0..4 {
                for x in 0..4 {
                    let d = x + y;
                    put(x, y, if d < 6 { v[d] } else { a(7) as u8 });
                }
            }
        }
        D45_PRED => {
            let above_right = e.above(bs as isize - 1) as u8;
            let row0: Vec<u8> = (0..bs)
                .map(|x| {
                    if x < bs - 1 {
                        avg3(
                            e.above(x as isize),
                            e.above(x as isize + 1),
                            e.above(x as isize + 2),
                        )
                    } else {
                        above_right
                    }
                })
                .collect();
            for (x, &v) in row0.iter().enumerate() {
                put(x, 0, v);
            }
            for r in 1..bs {
                let size = bs - 1 - r;
                for c in 0..bs {
                    put(c, r, if c < size { row0[c + r] } else { above_right });
                }
            }
        }
        D63_PRED if bs == 4 => {
            let a = |i| e.above(i);
            let mut put4 = |x: usize, y: usize, v: u8| put(x, y, v);
            put4(0, 0, avg2(a(0), a(1)));
            let v = avg2(a(1), a(2));
            put4(1, 0, v);
            put4(0, 2, v);
            let v = avg2(a(2), a(3));
            put4(2, 0, v);
            put4(1, 2, v);
            let v = avg2(a(3), a(4));
            put4(3, 0, v);
            put4(2, 2, v);
            put4(3, 2, avg2(a(4), a(5)));
            put4(0, 1, avg3(a(0), a(1), a(2)));
            let v = avg3(a(1), a(2), a(3));
            put4(1, 1, v);
            put4(0, 3, v);
            let v = avg3(a(2), a(3), a(4));
            put4(2, 1, v);
            put4(1, 3, v);
            let v = avg3(a(3), a(4), a(5));
            put4(3, 1, v);
            put4(2, 3, v);
            put4(3, 3, avg3(a(4), a(5), a(6)));
        }
        D63_PRED => {
            let tail = e.above(bs as isize - 1) as u8;
            let mut rows = vec![vec![0u8; bs]; bs];
            for c in 0..bs {
                let c = c as isize;
                rows[0][c as usize] = avg2(e.above(c), e.above(c + 1));
                rows[1][c as usize] = avg3(e.above(c), e.above(c + 1), e.above(c + 2));
            }
            let (mut r, mut size) = (2, bs - 2);
            while r < bs {
                for k in 0..bs {
                    rows[r][k] = if k < size {
                        rows[0][(r >> 1) + k]
                    } else {
                        tail
                    };
                    rows[r + 1][k] = if k < size {
                        rows[1][(r >> 1) + k]
                    } else {
                        tail
                    };
                }
                r += 2;
                size -= 1;
            }
            for (y, row) in rows.iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    put(x, y, v);
                }
            }
        }
        D117_PRED if bs == 4 => {
            let (i, j, k) = (e.left(0), e.left(1), e.left(2));
            let (x_, a, b, c, d) = (e.above(-1), e.above(0), e.above(1), e.above(2), e.above(3));
            let v = avg2(x_, a);
            put(0, 0, v);
            put(1, 2, v);
            let v = avg2(a, b);
            put(1, 0, v);
            put(2, 2, v);
            let v = avg2(b, c);
            put(2, 0, v);
            put(3, 2, v);
            put(3, 0, avg2(c, d));
            put(0, 3, avg3(k, j, i));
            put(0, 2, avg3(j, i, x_));
            let v = avg3(i, x_, a);
            put(0, 1, v);
            put(1, 3, v);
            let v = avg3(x_, a, b);
            put(1, 1, v);
            put(2, 3, v);
            let v = avg3(a, b, c);
            put(2, 1, v);
            put(3, 3, v);
            put(3, 1, avg3(b, c, d));
        }
        D117_PRED => {
            let mut rows = vec![vec![0u8; bs]; bs];
            for c in 0..bs {
                rows[0][c] = avg2(e.above(c as isize - 1), e.above(c as isize));
            }
            rows[1][0] = avg3(e.left(0), e.above(-1), e.above(0));
            for c in 1..bs {
                rows[1][c] = avg3(
                    e.above(c as isize - 2),
                    e.above(c as isize - 1),
                    e.above(c as isize),
                );
            }
            rows[2][0] = avg3(e.above(-1), e.left(0), e.left(1));
            for r in 3..bs {
                rows[r][0] = avg3(e.left(r - 3), e.left(r - 2), e.left(r - 1));
            }
            for r in 2..bs {
                for c in 1..bs {
                    rows[r][c] = rows[r - 2][c - 1];
                }
            }
            for (y, row) in rows.iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    put(x, y, v);
                }
            }
        }
        D135_PRED if bs == 4 => {
            let (i, j, k, l) = (e.left(0), e.left(1), e.left(2), e.left(3));
            let (x_, a, b, c, d) = (e.above(-1), e.above(0), e.above(1), e.above(2), e.above(3));
            put(0, 3, avg3(j, k, l));
            let v = avg3(i, j, k);
            put(1, 3, v);
            put(0, 2, v);
            let v = avg3(x_, i, j);
            put(2, 3, v);
            put(1, 2, v);
            put(0, 1, v);
            let v = avg3(a, x_, i);
            for p in 0..4 {
                put(p, p, v);
            }
            let v = avg3(b, a, x_);
            put(3, 2, v);
            put(2, 1, v);
            put(1, 0, v);
            let v = avg3(c, b, a);
            put(3, 1, v);
            put(2, 0, v);
            put(3, 0, avg3(d, c, b));
        }
        D135_PRED => {
            let mut border = vec![0u8; 2 * bs - 1];
            for i in 0..bs - 2 {
                border[i] = avg3(e.left(bs - 3 - i), e.left(bs - 2 - i), e.left(bs - 1 - i));
            }
            border[bs - 2] = avg3(e.above(-1), e.left(0), e.left(1));
            border[bs - 1] = avg3(e.left(0), e.above(-1), e.above(0));
            border[bs] = avg3(e.above(-1), e.above(0), e.above(1));
            for i in 0..bs - 2 {
                border[bs + 1 + i] = avg3(
                    e.above(i as isize),
                    e.above(i as isize + 1),
                    e.above(i as isize + 2),
                );
            }
            for y in 0..bs {
                for x in 0..bs {
                    put(x, y, border[bs - 1 - y + x]);
                }
            }
        }
        D153_PRED if bs == 4 => {
            let (i, j, k, l) = (e.left(0), e.left(1), e.left(2), e.left(3));
            let (x_, a, b, c) = (e.above(-1), e.above(0), e.above(1), e.above(2));
            let v = avg2(i, x_);
            put(0, 0, v);
            put(2, 1, v);
            let v = avg2(j, i);
            put(0, 1, v);
            put(2, 2, v);
            let v = avg2(k, j);
            put(0, 2, v);
            put(2, 3, v);
            put(0, 3, avg2(l, k));
            put(3, 0, avg3(a, b, c));
            put(2, 0, avg3(x_, a, b));
            let v = avg3(i, x_, a);
            put(1, 0, v);
            put(3, 1, v);
            let v = avg3(j, i, x_);
            put(1, 1, v);
            put(3, 2, v);
            let v = avg3(k, j, i);
            put(1, 2, v);
            put(3, 3, v);
            put(1, 3, avg3(l, k, j));
        }
        D153_PRED => {
            let mut rows = vec![vec![0u8; bs]; bs];
            rows[0][0] = avg2(e.above(-1), e.left(0));
            for r in 1..bs {
                rows[r][0] = avg2(e.left(r - 1), e.left(r));
            }
            rows[0][1] = avg3(e.left(0), e.above(-1), e.above(0));
            rows[1][1] = avg3(e.above(-1), e.left(0), e.left(1));
            for r in 2..bs {
                rows[r][1] = avg3(e.left(r - 2), e.left(r - 1), e.left(r));
            }
            for c in 0..bs - 2 {
                rows[0][2 + c] = avg3(
                    e.above(c as isize - 1),
                    e.above(c as isize),
                    e.above(c as isize + 1),
                );
            }
            for r in 1..bs {
                for c in 0..bs - 2 {
                    rows[r][2 + c] = rows[r - 1][c];
                }
            }
            for (y, row) in rows.iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    put(x, y, v);
                }
            }
        }
        D207_PRED if bs == 4 => {
            let (i, j, k, l) = (e.left(0), e.left(1), e.left(2), e.left(3));
            put(0, 0, avg2(i, j));
            let v = avg2(j, k);
            put(2, 0, v);
            put(0, 1, v);
            let v = avg2(k, l);
            put(2, 1, v);
            put(0, 2, v);
            put(1, 0, avg3(i, j, k));
            let v = avg3(j, k, l);
            put(3, 0, v);
            put(1, 1, v);
            let v = avg3(k, l, l);
            put(3, 1, v);
            put(1, 2, v);
            for (x, y) in [(3, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)] {
                put(x, y, l as u8);
            }
        }
        D207_PRED => {
            let mut rows = vec![vec![0u8; bs]; bs];
            for r in 0..bs - 1 {
                rows[r][0] = avg2(e.left(r), e.left(r + 1));
            }
            rows[bs - 1][0] = e.left(bs - 1) as u8;
            for r in 0..bs - 2 {
                rows[r][1] = avg3(e.left(r), e.left(r + 1), e.left(r + 2));
            }
            rows[bs - 2][1] = avg3(e.left(bs - 2), e.left(bs - 1), e.left(bs - 1));
            rows[bs - 1][1] = e.left(bs - 1) as u8;
            for c in 0..bs - 2 {
                rows[bs - 1][2 + c] = e.left(bs - 1) as u8;
            }
            for r in (0..=bs - 2).rev() {
                for c in 0..bs - 2 {
                    rows[r][2 + c] = rows[r + 1][c];
                }
            }
            for (y, row) in rows.iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    put(x, y, v);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
pub(super) fn edges_from(above_with_corner: &[u8], left: &[u8]) -> Edges {
    let mut e = Edges {
        above: [0; ABOVE_BASE + 64],
        left: [0; 32],
    };
    e.above[ABOVE_BASE - 1..ABOVE_BASE - 1 + above_with_corner.len()]
        .copy_from_slice(above_with_corner);
    e.left[..left.len()].copy_from_slice(left);
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks every predictor against libvpx's C output: the file named by `VP9_INTRA_VECTORS`.
    #[test]
    fn predictors_match_libvpx_vectors() {
        let Some(path) = std::env::var_os("VP9_INTRA_VECTORS") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let mut at = 0;
        let mut cases = 0;
        while at < data.len() {
            let (mode, size, have_left, have_top) =
                (data[at], data[at + 1], data[at + 2] != 0, data[at + 3] != 0);
            at += 4;
            let bs = 4usize << size;
            let above = &data[at..at + 1 + 2 * bs];
            at += 1 + 2 * bs;
            let left = &data[at..at + bs];
            at += bs;
            let expect = &data[at..at + bs * bs];
            at += bs * bs;
            let mut plane = vec![0u8; bs * bs];
            predict_from(
                &mut plane,
                0,
                bs,
                mode,
                size,
                &edges_from(above, left),
                have_left,
                have_top,
            );
            let first = plane.iter().zip(expect).position(|(a, b)| a != b);
            assert!(first.is_none(), "case {cases}: mode {mode}, size {size}, left {have_left}, top {have_top}, first mismatch at {first:?}");
            cases += 1;
        }
        assert!(cases > 0);
    }
}
