use super::*;
use crate::gpu::engine::threed::DrawCall;
use crate::gpu::syncpt::Host1x;
use crate::gpu::testing::{
    derivative_fragment_shader, solid_fragment_shader, write_vertex, Harness,
};
use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
use crate::mem::Memory;

mod attrib;
mod draw;
mod shading;

#[test]
fn triangles_assemble_into_disjoint_triples() {
    assert_eq!(
        assemble(Primitive::Triangles, 6),
        vec![[0, 1, 2], [3, 4, 5]]
    );
}

#[test]
fn triangle_strip_alternates_winding() {
    assert_eq!(
        assemble(Primitive::TriangleStrip, 5),
        vec![[0, 1, 2], [2, 1, 3], [2, 3, 4]]
    );
}

#[test]
fn a_right_triangle_covers_exactly_its_staircase_of_pixels() {
    // (0,0)-(4,0)-(0,4): the classic staircase.
    let covered = rasterize_triangle(
        ScreenVertex { x: 0.0, y: 0.0 },
        ScreenVertex { x: 4.0, y: 0.0 },
        ScreenVertex { x: 0.0, y: 4.0 },
        Bounds {
            x0: 0,
            y0: 0,
            x1: 8,
            y1: 8,
        },
    );
    let mut covered = covered;
    covered.sort();
    assert_eq!(
        covered,
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
    );
}

#[test]
fn a_quad_split_into_two_oppositely_wound_triangles_is_watertight() {
    // SDL's 8x8 quad: mixed windings, a 45-degree shared diagonal with
    // pixel centres on it.
    let (a, b) = (
        ScreenVertex { x: 0.0, y: 0.0 },
        ScreenVertex { x: 8.0, y: 0.0 },
    );
    let (c, d) = (
        ScreenVertex { x: 0.0, y: 8.0 },
        ScreenVertex { x: 8.0, y: 8.0 },
    );
    let bounds = Bounds {
        x0: 0,
        y0: 0,
        x1: 8,
        y1: 8,
    };

    let mut covered = rasterize_triangle(a, b, c, bounds);
    covered.extend(rasterize_triangle(b, c, d, bounds));
    covered.sort();

    let expected: Vec<(u32, u32)> = (0..8).flat_map(|x| (0..8).map(move |y| (x, y))).collect();
    let mut sorted = expected.clone();
    sorted.sort();
    assert_eq!(covered, sorted, "every pixel of the quad exactly once");
}

#[test]
fn rasterization_is_clipped_to_bounds() {
    let covered = rasterize_triangle(
        ScreenVertex { x: 0.0, y: 0.0 },
        ScreenVertex { x: 10.0, y: 0.0 },
        ScreenVertex { x: 0.0, y: 10.0 },
        Bounds {
            x0: 2,
            y0: 2,
            x1: 4,
            y1: 4,
        },
    );
    let mut covered = covered;
    covered.sort();
    assert_eq!(covered, vec![(2, 2), (2, 3), (3, 2), (3, 3)]);
}

#[test]
fn a_degenerate_triangle_covers_nothing() {
    let covered = rasterize_triangle(
        ScreenVertex { x: 1.0, y: 1.0 },
        ScreenVertex { x: 2.0, y: 2.0 },
        ScreenVertex { x: 3.0, y: 3.0 },
        Bounds {
            x0: 0,
            y0: 0,
            x1: 8,
            y1: 8,
        },
    );
    assert!(covered.is_empty());
}
