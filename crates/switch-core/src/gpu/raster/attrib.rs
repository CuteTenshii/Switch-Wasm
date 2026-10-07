//! Vertex attribute fetch and format decoding.

use crate::gpu::engine::threed::{VertexArray, VertexAttrib};
use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::f16_to_f32;
use crate::{Error, Result};

/// `DkVtxAttribSize`'s component count and per-component bit width
/// (deko3d.h), for the shapes this fetcher decodes.
fn attrib_shape(size: u32) -> Option<(u32, u32)> {
    match size {
        0x01 => Some((4, 32)), // 4x32
        0x02 => Some((3, 32)), // 3x32
        0x04 => Some((2, 32)), // 2x32
        0x12 => Some((1, 32)), // 1x32
        0x03 => Some((4, 16)), // 4x16
        0x05 => Some((3, 16)), // 3x16
        0x0f => Some((2, 16)), // 2x16
        0x1b => Some((1, 16)), // 1x16
        0x0a => Some((4, 8)),  // 4x8
        0x13 => Some((3, 8)),  // 3x8
        0x18 => Some((2, 8)),  // 2x8
        0x1d => Some((1, 8)),  // 1x8
        _ => None,
    }
}

/// Size `0x30`: four fields packed 10-10-10-2 into one word, red lowest.
pub(super) const ATTRIB_SIZE_10_10_10_2: u32 = 0x30;

pub(super) const ATTRIB_TYPE_SNORM: u32 = 1;
pub(super) const ATTRIB_TYPE_UNORM: u32 = 2;
pub(super) const ATTRIB_TYPE_SINT: u32 = 3;
pub(super) const ATTRIB_TYPE_UINT: u32 = 4;
/// An integer converted to the float of its value, unsigned and signed.
pub(super) const ATTRIB_TYPE_USCALED: u32 = 5;
pub(super) const ATTRIB_TYPE_SSCALED: u32 = 6;
pub(super) const ATTRIB_TYPE_FLOAT: u32 = 7;

/// What a "fixed" attribute (no data supplied) reads as.
const ATTRIB_DEFAULT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// A 10-10-10-2 attribute: red in bits 0-9, green 10-19, blue 20-29, alpha
/// the top two. `wgsl::unpack_1010102` must match.
fn fetch_1010102(
    attrib: VertexAttrib,
    array: VertexArray,
    vertex_index: u32,
    ctx: &ExecCtx,
) -> Result<[f32; 4]> {
    let addr = array.start + vertex_index as u64 * array.stride as u64 + attrib.offset as u64;
    // Past the array's limit a fetch reads zeros, as for every format.
    let word = if array.limit != 0 && addr + 4 > array.limit + 1 {
        0
    } else {
        ctx.read_u32(addr)?
    };
    let fields = [(0u32, 10u32), (10, 10), (20, 10), (30, 2)];
    let mut out = [0.0f32; 4];
    for (slot, (offset, bits)) in out.iter_mut().zip(fields) {
        let raw = (word >> offset) & ((1 << bits) - 1);
        let signed = sext_u32(raw, bits);
        *slot = match attrib.ty {
            ATTRIB_TYPE_SNORM => {
                // The most negative value and the one above it both mean -1.
                (signed as f32 / ((1 << (bits - 1)) - 1) as f32).max(-1.0)
            }
            ATTRIB_TYPE_UNORM => raw as f32 / ((1u32 << bits) - 1) as f32,
            ATTRIB_TYPE_SINT => f32::from_bits(signed as u32),
            ATTRIB_TYPE_UINT => f32::from_bits(raw),
            ty => {
                return Err(Error::Gpu(format!(
                    "raster: unsupported 10-10-10-2 vertex attribute type {ty}"
                )))
            }
        };
    }
    Ok(out)
}

/// `value`'s low `bits` bits, sign extended.
fn sext_u32(value: u32, bits: u32) -> i32 {
    ((value << (32 - bits)) as i32) >> (32 - bits)
}

/// Fetch one vertex's attribute from GPU memory, padded to 4 components with
/// [`ATTRIB_DEFAULT`]. `is_bgra` swaps the first and third components.
pub fn fetch_attribute(
    attrib: VertexAttrib,
    array: VertexArray,
    vertex_index: u32,
    ctx: &ExecCtx,
) -> Result<[f32; 4]> {
    // A "fixed" attribute reads the default; the draw binds nothing to it.
    if attrib.is_fixed {
        return Ok(ATTRIB_DEFAULT);
    }
    // A disabled buffer means state was misread; refuse rather than invent.
    if !array.enabled {
        return Err(Error::Gpu(format!(
            "raster: attribute reads from disabled vertex buffer {}",
            attrib.buffer_id
        )));
    }
    if attrib.size == ATTRIB_SIZE_10_10_10_2 {
        return fetch_1010102(attrib, array, vertex_index, ctx);
    }
    let (components, bits) = attrib_shape(attrib.size).ok_or_else(|| {
        Error::Gpu(format!(
            "raster: unsupported vertex attribute size {:#x}",
            attrib.size
        ))
    })?;

    let addr = array.start + vertex_index as u64 * array.stride as u64 + attrib.offset as u64;

    let mut out = ATTRIB_DEFAULT;
    // An integer attribute's missing `w` is integer 1, as WebGPU fills it.
    if matches!(attrib.ty, ATTRIB_TYPE_SINT | ATTRIB_TYPE_UINT) && components < 4 {
        out[3] = f32::from_bits(1);
    }
    // Past the array's limit a fetch reads zeros, as on hardware.
    let bytes = u64::from(components * bits / 8);
    if array.limit != 0 && addr + bytes > array.limit + 1 {
        for value in out.iter_mut().take(components as usize) {
            *value = 0.0;
        }
        return Ok(out);
    }
    match (attrib.ty, bits) {
        // An integer's bits are carried as they are, like a float's.
        (ATTRIB_TYPE_FLOAT | ATTRIB_TYPE_SINT | ATTRIB_TYPE_UINT, 32) => {
            for c in 0..components {
                let bits = ctx.read_u32(addr + c as u64 * 4)?;
                out[c as usize] = f32::from_bits(bits);
            }
        }
        (ATTRIB_TYPE_USCALED | ATTRIB_TYPE_SSCALED, bits) => {
            let packed = ctx.read_pixel(addr, components * bits / 8)?;
            for c in 0..components {
                let raw = (packed >> (c * bits)) as u32 & (u32::MAX >> (32 - bits));
                out[c as usize] = if attrib.ty == ATTRIB_TYPE_SSCALED {
                    sext_u32(raw, bits) as f32
                } else {
                    raw as f32
                };
            }
        }
        // 16-bit shapes are read as one packed value, translating the address once.
        (ATTRIB_TYPE_FLOAT, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                out[c as usize] = f16_to_f32((packed >> (c * 16)) as u16);
            }
        }
        (ATTRIB_TYPE_UNORM, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                out[c as usize] = f32::from((packed >> (c * 16)) as u16) / 65535.0;
            }
        }
        (ATTRIB_TYPE_SNORM, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                let value = (packed >> (c * 16)) as u16 as i16;
                // -32768 and -32767 both mean -1, as at eight bits.
                out[c as usize] = (f32::from(value) / 32767.0).max(-1.0);
            }
        }
        (ATTRIB_TYPE_SINT, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                let value = (packed >> (c * 16)) as u16 as i16;
                out[c as usize] = f32::from_bits(i32::from(value) as u32);
            }
        }
        (ATTRIB_TYPE_UINT, 16) => {
            let packed = ctx.read_pixel(addr, components * 2)?;
            for c in 0..components {
                out[c as usize] = f32::from_bits(u32::from((packed >> (c * 16)) as u16));
            }
        }
        // 8-bit shapes read only their own bytes; nothing may follow a mapping's end.
        (ATTRIB_TYPE_UNORM, 8) => {
            let packed = ctx.read_pixel(addr, components)? as u32;
            for c in 0..components {
                let byte = (packed >> (c * 8)) & 0xff;
                out[c as usize] = byte as f32 / 255.0;
            }
        }
        (ATTRIB_TYPE_SNORM, 8) => {
            let packed = ctx.read_pixel(addr, components)? as u32;
            for c in 0..components {
                let byte = ((packed >> (c * 8)) & 0xff) as u8 as i8;
                // -128 and -127 both mean -1.
                out[c as usize] = (byte as f32 / 127.0).max(-1.0);
            }
        }
        // Integer attributes carry their bits, not a converted float.
        (ATTRIB_TYPE_SINT, 8) => {
            let packed = ctx.read_pixel(addr, components)? as u32;
            for c in 0..components {
                let byte = ((packed >> (c * 8)) & 0xff) as u8 as i8;
                out[c as usize] = f32::from_bits(byte as i32 as u32);
            }
        }
        (ATTRIB_TYPE_UINT, 8) => {
            let packed = ctx.read_pixel(addr, components)? as u32;
            for c in 0..components {
                out[c as usize] = f32::from_bits((packed >> (c * 8)) & 0xff);
            }
        }
        (ty, bits) => {
            return Err(Error::Gpu(format!(
                "raster: unsupported vertex attribute type {} at {} bits",
                ty, bits
            )));
        }
    }
    if attrib.is_bgra {
        out.swap(0, 2);
    }
    Ok(out)
}
