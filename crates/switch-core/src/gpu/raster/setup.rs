//! Triangle setup, edge functions and coverage.

use super::{Bounds, ScreenVertex};

/// Rasterize one triangle to the pixels it covers, under the top-left fill
/// rule. Coverage only.
pub fn rasterize_triangle(
    v0: ScreenVertex,
    v1: ScreenVertex,
    v2: ScreenVertex,
    bounds: Bounds,
) -> Vec<(u32, u32)> {
    rasterize_triangle_weighted(v0, v1, v2, bounds)
        .into_iter()
        .map(|(x, y, ..)| (x, y))
        .collect()
}

/// Like [`rasterize_triangle`], but also returns each covered pixel's
/// screen-space barycentric weights; the shader does perspective correction.
pub fn rasterize_triangle_weighted(
    v0: ScreenVertex,
    v1: ScreenVertex,
    v2: ScreenVertex,
    bounds: Bounds,
) -> Vec<(u32, u32, f32, f32, f32)> {
    let Some(tri) = TriangleSetup::new(v0, v1, v2) else {
        return Vec::new();
    };
    let (min_x, max_x, min_y, max_y) = tri.bbox(bounds);
    let mut out = Vec::new();
    for y in min_y..max_y {
        for x in min_x..max_x {
            if let Some([w0, w1, w2]) = tri.coverage(x as f32 + 0.5, y as f32 + 0.5) {
                out.push((x, y, w0, w1, w2));
            }
        }
    }
    out
}

/// The samples alpha-to-coverage leaves for a fragment of this alpha: a fixed
/// prefix rather than hardware's dither, which averages the same.
pub(super) fn alpha_coverage(alpha: f32, count: u32) -> u32 {
    let kept = (alpha.clamp(0.0, 1.0) * count as f32).round() as u32;
    if kept >= count {
        u32::MAX
    } else {
        (1u32 << kept) - 1
    }
}

fn edge(a: ScreenVertex, b: ScreenVertex, px: f32, py: f32) -> f32 {
    (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
}

fn is_top_left(a: ScreenVertex, b: ScreenVertex) -> bool {
    (a.y == b.y && a.x > b.x) || (a.y > b.y)
}

/// A triangle prepared for coverage queries: edge functions, top-left
/// tie-breaks and winding fix-up, resolved once for all samples.
#[derive(Debug, Clone, Copy)]
pub struct TriangleSetup {
    v0: ScreenVertex,
    v1: ScreenVertex,
    v2: ScreenVertex,
    /// The caller's winding was clockwise, so `w1` and `w2` swap back.
    clockwise: bool,
    area: f32,
    /// Whether each of the edges `v0-v1`, `v1-v2`, `v2-v0` is a top or left
    /// one, in that order.
    top_left: [bool; 3],
}

impl TriangleSetup {
    /// `None` for a degenerate triangle: zero area, nothing covered.
    pub fn new(v0: ScreenVertex, v1: ScreenVertex, v2: ScreenVertex) -> Option<TriangleSetup> {
        let signed_area = edge(v0, v1, v2.x, v2.y);
        if signed_area == 0.0 {
            return None;
        }
        // Wind counter-clockwise before applying the fill rule, so two triangles
        // sharing an edge always walk it in opposite directions (SDL's quads mix
        // windings).
        let clockwise = signed_area < 0.0;
        let (v1, v2) = if clockwise { (v2, v1) } else { (v1, v2) };
        Some(TriangleSetup {
            v0,
            v1,
            v2,
            clockwise,
            area: signed_area.abs(),
            top_left: [
                is_top_left(v0, v1),
                is_top_left(v1, v2),
                is_top_left(v2, v0),
            ],
        })
    }

    /// The half-open pixel range the triangle can reach, clipped to `bounds`.
    pub fn bbox(&self, bounds: Bounds) -> (u32, u32, u32, u32) {
        let (v0, v1, v2) = (self.v0, self.v1, self.v2);
        (
            v0.x.min(v1.x).min(v2.x).floor().max(bounds.x0 as f32) as u32,
            (v0.x.max(v1.x).max(v2.x).ceil() as u32).min(bounds.x1),
            v0.y.min(v1.y).min(v2.y).floor().max(bounds.y0 as f32) as u32,
            (v0.y.max(v1.y).max(v2.y).ceil() as u32).min(bounds.y1),
        )
    }

    /// The three edge functions at `(px, py)`, in `top_left` order.
    fn edges(&self, px: f32, py: f32) -> [f32; 3] {
        [
            edge(self.v0, self.v1, px, py),
            edge(self.v1, self.v2, px, py),
            edge(self.v2, self.v0, px, py),
        ]
    }

    /// `w0` is opposite `v0` (edge `v1-v2`), and so on.
    fn weights_from(&self, e: [f32; 3]) -> [f32; 3] {
        let (w0, w1, w2) = (e[1] / self.area, e[2] / self.area, e[0] / self.area);
        if self.clockwise {
            [w0, w2, w1]
        } else {
            [w0, w1, w2]
        }
    }

    /// Barycentric weights at `(px, py)` whether or not it is covered: pixel
    /// centre evaluation extrapolates on partially covered pixels.
    pub fn weights(&self, px: f32, py: f32) -> [f32; 3] {
        self.weights_from(self.edges(px, py))
    }

    /// Weights at `(px, py)`, or `None` where the fill rule leaves it uncovered.
    pub fn coverage(&self, px: f32, py: f32) -> Option<[f32; 3]> {
        let e = self.edges(px, py);
        let inside = |value: f32, top_left: bool| value > 0.0 || (top_left && value == 0.0);
        (0..3)
            .all(|i| inside(e[i], self.top_left[i]))
            .then(|| self.weights_from(e))
    }
}
