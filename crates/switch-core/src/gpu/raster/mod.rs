//! The software rasterizer: vertex fetch, primitive assembly, rasterization
//! and fragment shading. Coverage is per sample and shading per pixel (see
//! [`crate::gpu::surface::SampleGrid`]). `MultisampleCoverageToColor` and
//! `SetMultisampleRasterEnable` are not implemented.

use crate::{Error, Result};

mod attrib;
mod draw;
mod fragment;
mod setup;
#[cfg(test)]
mod tests;
mod vertex;

pub use attrib::fetch_attribute;
pub use draw::draw;
pub use fragment::QUAD;
pub use setup::{rasterize_triangle, rasterize_triangle_weighted, TriangleSetup};

/// The `DkPrimitive` topologies (deko3d.h). Points and lines are recognised
/// but not rasterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
    QuadStrip,
    Polygon,
}

impl Primitive {
    pub fn from_raw(raw: u32) -> Result<Primitive> {
        match raw {
            0 => Ok(Primitive::Points),
            1 => Ok(Primitive::Lines),
            2 => Ok(Primitive::LineLoop),
            3 => Ok(Primitive::LineStrip),
            4 => Ok(Primitive::Triangles),
            5 => Ok(Primitive::TriangleStrip),
            6 => Ok(Primitive::TriangleFan),
            7 => Ok(Primitive::Quads),
            8 => Ok(Primitive::QuadStrip),
            9 => Ok(Primitive::Polygon),
            other => Err(Error::Gpu(format!("raster: unknown DkPrimitive {other}"))),
        }
    }
}

/// Break a `count`-vertex draw into triangles, as vertex-ordinal triples.
/// Strips keep a consistent winding; point and line topologies produce nothing.
pub fn assemble(primitive: Primitive, count: u32) -> Vec<[u32; 3]> {
    match primitive {
        Primitive::Points | Primitive::Lines | Primitive::LineLoop | Primitive::LineStrip => {
            Vec::new()
        }
        Primitive::Triangles => (0..count / 3)
            .map(|t| [t * 3, t * 3 + 1, t * 3 + 2])
            .collect(),
        Primitive::TriangleStrip => {
            if count < 3 {
                return Vec::new();
            }
            (0..count - 2)
                .map(|i| {
                    if i % 2 == 0 {
                        [i, i + 1, i + 2]
                    } else {
                        [i + 1, i, i + 2]
                    }
                })
                .collect()
        }
        Primitive::TriangleFan | Primitive::Polygon => {
            if count < 3 {
                return Vec::new();
            }
            (0..count - 2).map(|i| [0, i + 1, i + 2]).collect()
        }
        Primitive::Quads => (0..count / 4)
            .flat_map(|q| {
                let b = q * 4;
                [[b, b + 1, b + 2], [b, b + 2, b + 3]]
            })
            .collect(),
        Primitive::QuadStrip => {
            if count < 4 {
                return Vec::new();
            }
            (0..(count - 2) / 2)
                .flat_map(|q| {
                    let b = q * 2;
                    [[b, b + 1, b + 2], [b + 2, b + 1, b + 3]]
                })
                .collect()
        }
    }
}

/// A vertex position in screen space (pixels, `y` growing downward).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenVertex {
    pub x: f32,
    pub y: f32,
}

/// Inclusive-exclusive pixel bounds `[x0, x1) x [y0, y1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}
