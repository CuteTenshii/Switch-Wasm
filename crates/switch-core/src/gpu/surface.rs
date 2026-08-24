//! Surface layout and pixel formats.
//!
//! Maxwell renders into *block-linear* surfaces: memory is grouped into 512
//! byte GOBs ("group of bytes", 64 bytes wide by 8 rows), GOBs are stacked
//! into blocks that are `2^block_height_log2` GOBs tall, and blocks are laid
//! out left-to-right then top-to-bottom. Reading a pixel therefore means
//! swizzling its (x, y) into that order — the layout is what makes a naive
//! memory dump of a Switch framebuffer look shredded.
//!
//! A surface can also be *pitch* (plain linear rows), which the display path
//! and the 2D engine both use.

use crate::gpu::exec::ExecCtx;
use crate::{Error, Result};

/// GOB dimensions on Fermi and later.
pub const GOB_WIDTH: u32 = 64;
pub const GOB_HEIGHT: u32 = 8;
pub const GOB_SIZE: u32 = GOB_WIDTH * GOB_HEIGHT;

/// Byte offset of `(x, y)` inside a single GOB, where `x` is a byte offset in
/// the row and both are already reduced modulo the GOB size.
#[inline]
pub fn gob_offset(x: u32, y: u32) -> u32 {
    let x = x % GOB_WIDTH;
    let y = y % GOB_HEIGHT;
    (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + (x % 16)
}

/// Byte offset of `(x_bytes, y)` in a block-linear 2D surface.
///
/// `width_bytes` is the surface's row length in bytes (it is rounded up to a
/// whole number of GOBs, as the hardware does) and `block_height_gobs` is
/// `2^height` from the surface's tile mode.
pub fn block_linear_offset(
    x_bytes: u32,
    y: u32,
    width_bytes: u32,
    block_height_gobs: u32,
) -> u32 {
    let block_height_gobs = block_height_gobs.max(1);
    let width_gobs = width_bytes.div_ceil(GOB_WIDTH).max(1);
    let block_bytes = GOB_SIZE * block_height_gobs;
    let block_row_bytes = width_gobs * block_bytes;
    let rows_per_block = GOB_HEIGHT * block_height_gobs;

    let block_y = y / rows_per_block;
    let gob_y = (y % rows_per_block) / GOB_HEIGHT;
    let gob_x = x_bytes / GOB_WIDTH;

    block_y * block_row_bytes + gob_x * block_bytes + gob_y * GOB_SIZE + gob_offset(x_bytes, y)
}

/// How a surface's rows are arranged in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Plain rows of `pitch` bytes.
    Pitch { pitch: u32 },
    /// Block-linear with `2^n` GOBs per block vertically.
    BlockLinear { block_height_gobs: u32 },
}

impl Layout {
    /// Byte offset of the pixel at `(x, y)` for a surface `width_bytes` wide.
    pub fn offset(&self, x_bytes: u32, y: u32, width_bytes: u32) -> u32 {
        match *self {
            Layout::Pitch { pitch } => y * pitch + x_bytes,
            Layout::BlockLinear { block_height_gobs } => {
                block_linear_offset(x_bytes, y, width_bytes, block_height_gobs)
            }
        }
    }
}

/// The most samples per pixel any Maxwell `MsaaMode` names (`4x4`).
pub const MAX_SAMPLES: usize = 16;

/// How a multisampled surface lays its samples out.
///
/// Maxwell stores more than one sample per pixel by expanding the surface
/// *spatially*: a pixel owns a `samples_x` by `samples_y` tile of texels, so a
/// 4x-multisampled 1280x720 target is a 2560x1440 surface in memory. The
/// render- and depth-target registers describe that expanded surface, while
/// the scissor, the viewport and the clear rectangle stay in pixels — which is
/// why every write into a multisampled target goes through here to turn a
/// pixel and a sample number into a texel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleGrid {
    pub samples_x: u32,
    pub samples_y: u32,
    /// Where each sample sits inside its pixel, on `[0, 1)` per axis.
    positions: [[f32; 2]; MAX_SAMPLES],
    /// Which texel of the pixel's tile stores each sample.
    slots: [(u32, u32); MAX_SAMPLES],
}

impl Default for SampleGrid {
    fn default() -> SampleGrid {
        SampleGrid::single()
    }
}

impl SampleGrid {
    /// One sample per pixel, at the pixel centre: a surface whose texels and
    /// pixels are the same thing.
    pub fn single() -> SampleGrid {
        SampleGrid {
            samples_x: 1,
            samples_y: 1,
            positions: [[0.5, 0.5]; MAX_SAMPLES],
            slots: [(0, 0); MAX_SAMPLES],
        }
    }

    /// Build the grid a `MultisampleMode` and a `MultisampleSampleLocations`
    /// table describe. `locations` holds one packed byte per sample, as the
    /// four location registers store them.
    pub fn new(mode: u32, locations: &[u8; MAX_SAMPLES]) -> Result<SampleGrid> {
        let (samples_x, samples_y) = msaa_mode_grid(mode)?;
        let count = (samples_x * samples_y) as usize;
        // A location table nothing has written would put every sample at the
        // same spot. Fall back to the centre of each sample's own texel, which
        // is exactly what `sample_slots` then maps back to raster order.
        let programmed = locations[..count].iter().any(|&b| b != 0);
        let mut positions = [[0.5f32; 2]; MAX_SAMPLES];
        for (i, position) in positions.iter_mut().enumerate().take(count) {
            *position = if programmed {
                // `x | (y << 4)`, each in sixteenths of a pixel — deko3d's
                // `encodeSampleLocation`.
                [
                    (locations[i] & 0xF) as f32 / 16.0,
                    (locations[i] >> 4) as f32 / 16.0,
                ]
            } else {
                [
                    ((i as u32 % samples_x) as f32 + 0.5) / samples_x as f32,
                    ((i as u32 / samples_x) as f32 + 0.5) / samples_y as f32,
                ]
            };
        }
        let slots = sample_slots(&positions, count, samples_x, samples_y);
        Ok(SampleGrid { samples_x, samples_y, positions, slots })
    }

    pub fn count(&self) -> u32 {
        self.samples_x * self.samples_y
    }

    /// Whether each pixel is a single texel, so a caller can keep its
    /// one-sample fast path instead of walking a grid of one.
    pub fn is_single(&self) -> bool {
        self.samples_x == 1 && self.samples_y == 1
    }

    /// Where `sample` sits inside its pixel — the point the coverage test and
    /// the depth interpolation use.
    pub fn position(&self, sample: u32) -> [f32; 2] {
        self.positions[sample as usize]
    }

    /// Texel coordinates of `sample` of the pixel at `(x, y)`.
    pub fn texel(&self, x: u32, y: u32, sample: u32) -> (u32, u32) {
        let (offset_x, offset_y) = self.slots[sample as usize];
        (x * self.samples_x + offset_x, y * self.samples_y + offset_y)
    }

    /// The pixel extent of a surface `width` by `height` *texels* — what the
    /// target registers hold, converted to what the scissor talks about.
    pub fn pixels(&self, width: u32, height: u32) -> (u32, u32) {
        (width / self.samples_x, height / self.samples_y)
    }
}

/// The sample tile a `MsaaMode` describes, as `(x, y)`.
///
/// Values and their spellings come from deko3d's `MsaaMode` enum
/// (`texture_image_control_block.h`) paired with the `m_samplesX`/`m_samplesY`
/// it derives from each (`dk_image.cpp`). The virtual-coverage modes store
/// their colour samples on the same grid as the plain mode they extend; only
/// the coverage bits they add on top differ, and nothing here consumes those.
fn msaa_mode_grid(mode: u32) -> Result<(u32, u32)> {
    Ok(match mode {
        0 => (1, 1),               // 1x1
        1 | 5 => (2, 1),           // 2x1, 2x1_D3D
        2 | 8 | 9 => (2, 2),       // 2x2, 2x2_VC4, 2x2_VC12
        3 | 4 | 10 | 11 => (4, 2), // 4x2, 4x2_D3D, 4x2_VC8, 4x2_VC24
        6 => (4, 4),               // 4x4
        other => return Err(Error::Gpu(format!("surface: unknown MsaaMode {:#x}", other))),
    })
}

/// Which texel of a pixel's tile holds each sample.
///
/// Hardware fixes this mapping per mode and constrains a programmable sample
/// location to stay inside its own texel — so the texel a sample's location
/// falls in *is* its slot. Deriving it that way reproduces the tables deko3d
/// ships for 4x and 8x (`locationsMS4`/`locationsMS8`) without hard-coding one
/// per mode, and it keeps a guest's custom locations stored where a resolve
/// that box-filters the tile expects to find them.
///
/// Two samples landing in one texel is not a table hardware accepts. If one
/// turns up anyway, fall back to raster order rather than aliasing two samples
/// onto the same storage and silently losing one.
fn sample_slots(
    positions: &[[f32; 2]; MAX_SAMPLES],
    count: usize,
    samples_x: u32,
    samples_y: u32,
) -> [(u32, u32); MAX_SAMPLES] {
    let mut slots = [(0u32, 0u32); MAX_SAMPLES];
    let mut taken = [false; MAX_SAMPLES];
    let mut distinct = true;
    for (i, slot) in slots.iter_mut().enumerate().take(count) {
        let x = ((positions[i][0] * samples_x as f32) as u32).min(samples_x - 1);
        let y = ((positions[i][1] * samples_y as f32) as u32).min(samples_y - 1);
        *slot = (x, y);
        let flat = (y * samples_x + x) as usize;
        distinct &= !taken[flat];
        taken[flat] = true;
    }
    if !distinct {
        for (i, slot) in slots.iter_mut().enumerate().take(count) {
            *slot = (i as u32 % samples_x, i as u32 / samples_x);
        }
    }
    slots
}

/// A Maxwell colour render-target format (`ColorSurfaceFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorFormat {
    pub raw: u32,
    pub bytes_per_pixel: u32,
}

/// Component order of an 8-bit-per-channel format, as stored little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order8 {
    Rgba,
    Bgra,
}

impl ColorFormat {
    pub fn from_raw(raw: u32) -> Result<ColorFormat> {
        let bytes_per_pixel = match raw {
            0xC0..=0xC5 => 16,                     // RGBA32 / RGBX32
            0xC6..=0xCE => 8,                      // RGBA16 / RG32 / RGBX16
            0xCF..=0xE7 | 0xF9 | 0xFA | 0xFD | 0xFE | 0xFF => 4, // 32-bit formats
            0xE8 | 0xE9 | 0xEA..=0xEF | 0xF0..=0xF2 | 0xF8 | 0xFB | 0xFC => 2,
            0xF3..=0xF7 => 1,
            0x00 => 0, // disabled render target
            other => {
                return Err(Error::Gpu(format!(
                    "surface: unknown colour format {:#x}",
                    other
                )))
            }
        };
        Ok(ColorFormat { raw, bytes_per_pixel })
    }

    fn order8(&self) -> Option<Order8> {
        match self.raw {
            0xD5..=0xD9 | 0xF9 | 0xFA => Some(Order8::Rgba),
            0xCF | 0xD0 | 0xE6 | 0xE7 | 0xFD | 0xFE => Some(Order8::Bgra),
            _ => None,
        }
    }

    /// Whether the format's colour channels are sRGB-encoded.
    pub fn is_srgb(&self) -> bool {
        matches!(self.raw, 0xD0 | 0xD6 | 0xE7 | 0xFA)
    }

    /// Whether the alpha channel exists (an "X" format ignores it).
    pub fn has_alpha(&self) -> bool {
        !matches!(self.raw, 0xE6 | 0xE7 | 0xF9 | 0xFA | 0xFD | 0xFE | 0xF8)
    }

    /// Pack a normalized RGBA colour into this format's raw pixel bytes.
    pub fn encode(&self, rgba: [f32; 4]) -> Result<u128> {
        let unorm8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        if let Some(order) = self.order8() {
            let (r, g, b, a) = (
                unorm8(rgba[0]),
                unorm8(rgba[1]),
                unorm8(rgba[2]),
                if self.has_alpha() { unorm8(rgba[3]) } else { 0xFF },
            );
            return Ok(match order {
                Order8::Rgba => (r | (g << 8) | (b << 16) | (a << 24)) as u128,
                Order8::Bgra => (b | (g << 8) | (r << 16) | (a << 24)) as u128,
            });
        }
        match self.raw {
            // B5G6R5
            0xE8 => {
                let r = (rgba[0].clamp(0.0, 1.0) * 31.0 + 0.5) as u32;
                let g = (rgba[1].clamp(0.0, 1.0) * 63.0 + 0.5) as u32;
                let b = (rgba[2].clamp(0.0, 1.0) * 31.0 + 0.5) as u32;
                Ok((b | (g << 5) | (r << 11)) as u128)
            }
            // BGR5A1 / BGR5X1
            0xE9 | 0xF8 => {
                let r = (rgba[0].clamp(0.0, 1.0) * 31.0 + 0.5) as u32;
                let g = (rgba[1].clamp(0.0, 1.0) * 31.0 + 0.5) as u32;
                let b = (rgba[2].clamp(0.0, 1.0) * 31.0 + 0.5) as u32;
                let a = (rgba[3] >= 0.5) as u32;
                Ok((b | (g << 5) | (r << 10) | (a << 15)) as u128)
            }
            // RGB10A2 / BGR10A2
            0xD1 | 0xDF => {
                let scale = |v: f32| (v.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32;
                let (c0, c2) = if self.raw == 0xD1 {
                    (scale(rgba[0]), scale(rgba[2]))
                } else {
                    (scale(rgba[2]), scale(rgba[0]))
                };
                let a = (rgba[3].clamp(0.0, 1.0) * 3.0 + 0.5) as u32;
                Ok((c0 | (scale(rgba[1]) << 10) | (c2 << 20) | (a << 30)) as u128)
            }
            // R32Float
            0xE5 => Ok(rgba[0].to_bits() as u128),
            // RGBA32Float
            0xC0 | 0xC3 => {
                let mut v = 0u128;
                for (i, c) in rgba.iter().enumerate() {
                    v |= (c.to_bits() as u128) << (32 * i);
                }
                Ok(v)
            }
            // RGBA16Float
            0xCA | 0xCE => {
                let mut v = 0u128;
                for (i, c) in rgba.iter().enumerate() {
                    v |= (f32_to_f16(*c) as u128) << (16 * i);
                }
                Ok(v)
            }
            // RG8Unorm
            0xEA => Ok((unorm8(rgba[0]) | (unorm8(rgba[1]) << 8)) as u128),
            // R8Unorm / A8Unorm
            0xF3 => Ok(unorm8(rgba[0]) as u128),
            0xF7 => Ok(unorm8(rgba[3]) as u128),
            other => Err(Error::Gpu(format!(
                "surface: encoding colour format {:#x} is not implemented",
                other
            ))),
        }
    }

    /// Unpack raw pixel bytes into a normalized RGBA colour.
    pub fn decode(&self, raw: u128) -> Result<[f32; 4]> {
        let unorm8 = |v: u32| (v & 0xFF) as f32 / 255.0;
        if let Some(order) = self.order8() {
            let v = raw as u32;
            let (c0, c1, c2, c3) = (v, v >> 8, v >> 16, v >> 24);
            let a = if self.has_alpha() { unorm8(c3) } else { 1.0 };
            return Ok(match order {
                Order8::Rgba => [unorm8(c0), unorm8(c1), unorm8(c2), a],
                Order8::Bgra => [unorm8(c2), unorm8(c1), unorm8(c0), a],
            });
        }
        match self.raw {
            0xE8 => {
                let v = raw as u32;
                Ok([
                    ((v >> 11) & 0x1F) as f32 / 31.0,
                    ((v >> 5) & 0x3F) as f32 / 63.0,
                    (v & 0x1F) as f32 / 31.0,
                    1.0,
                ])
            }
            0xE9 | 0xF8 => {
                let v = raw as u32;
                Ok([
                    ((v >> 10) & 0x1F) as f32 / 31.0,
                    ((v >> 5) & 0x1F) as f32 / 31.0,
                    (v & 0x1F) as f32 / 31.0,
                    if self.raw == 0xE9 && (v >> 15) & 1 == 0 { 0.0 } else { 1.0 },
                ])
            }
            0xD1 | 0xDF => {
                let v = raw as u32;
                let c0 = (v & 0x3FF) as f32 / 1023.0;
                let c1 = ((v >> 10) & 0x3FF) as f32 / 1023.0;
                let c2 = ((v >> 20) & 0x3FF) as f32 / 1023.0;
                let a = ((v >> 30) & 3) as f32 / 3.0;
                Ok(if self.raw == 0xD1 { [c0, c1, c2, a] } else { [c2, c1, c0, a] })
            }
            0xE5 => Ok([f32::from_bits(raw as u32), 0.0, 0.0, 1.0]),
            0xC0 | 0xC3 => {
                let mut out = [0.0f32; 4];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = f32::from_bits((raw >> (32 * i)) as u32);
                }
                Ok(out)
            }
            0xCA | 0xCE => {
                let mut out = [0.0f32; 4];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = f16_to_f32((raw >> (16 * i)) as u16);
                }
                Ok(out)
            }
            0xEA => Ok([unorm8(raw as u32), unorm8((raw >> 8) as u32), 0.0, 1.0]),
            0xF3 => Ok([unorm8(raw as u32), 0.0, 0.0, 1.0]),
            0xF7 => Ok([0.0, 0.0, 0.0, unorm8(raw as u32)]),
            other => Err(Error::Gpu(format!(
                "surface: decoding colour format {:#x} is not implemented",
                other
            ))),
        }
    }
}

/// Convert a linear colour channel to 8-bit sRGB, for presenting an sRGB
/// render target on a canvas that expects sRGB bytes.
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// The inverse of [`linear_to_srgb`] — what a sampler applies on the way *out*
/// of an sRGB-encoded texture, so that shading happens in linear light.
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x7F_FFFF;
    if exp <= 0 {
        sign
    } else if exp >= 0x1F {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | ((mantissa >> 13) as u16)
    }
}

pub(crate) fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v as u32) & 0x8000) << 16;
    let exp = ((v as u32) >> 10) & 0x1F;
    let mantissa = (v as u32) & 0x3FF;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: normalise it. The mantissa occupies the low ten bits
        // of a u32, so it has at least 22 leading zeros, and the shift that
        // brings its top set bit to bit 10 is that count less 22. Taking 21
        // here put every subnormal at half its value.
        let shift = mantissa.leading_zeros() - 22;
        let exp = 127 - 15 - shift;
        let mantissa = (mantissa << (shift + 1)) & 0x3FF;
        f32::from_bits(sign | (exp << 23) | (mantissa << 13))
    } else if exp == 0x1F {
        f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mantissa << 13))
    }
}

/// A described image in GPU memory: enough to compute where `(x, y)` lives
/// and how to decode it. Shared by the 2D engine's blits and the 3D engine's
/// texture sampling — both are "read a described surface", nothing more.
#[derive(Debug, Clone, Copy)]
pub struct Surface {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub layout: Layout,
}

impl Surface {
    pub fn offset(&self, x: u32, y: u32) -> u32 {
        let bpp = self.format.bytes_per_pixel;
        let width_bytes = match self.layout {
            Layout::Pitch { pitch } => pitch,
            Layout::BlockLinear { .. } => self.width * bpp,
        };
        self.layout.offset(x * bpp, y, width_bytes)
    }

    pub fn texel(&self, x: u32, y: u32, ctx: &ExecCtx) -> Result<[f32; 4]> {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let va = self.addr + self.offset(x, y) as u64;
        self.format.decode(ctx.read_pixel(va, self.format.bytes_per_pixel)?)
    }

    pub fn sample_point(&self, u: f64, v: f64, ctx: &ExecCtx) -> Result<[f32; 4]> {
        self.texel(u.max(0.0) as u32, v.max(0.0) as u32, ctx)
    }

    pub fn sample_bilinear(&self, u: f64, v: f64, ctx: &ExecCtx) -> Result<[f32; 4]> {
        bilinear(u, v, |x, y| self.texel(x, y, ctx))
    }
}

/// Bilinear filtering over whatever `texel` fetches.
///
/// Taking the fetch as a callback is what lets a block-compressed texture,
/// whose texels come out of a decoded block rather than straight from memory,
/// filter identically to a plain one instead of growing its own copy of this.
pub fn bilinear(
    u: f64,
    v: f64,
    mut texel: impl FnMut(u32, u32) -> Result<[f32; 4]>,
) -> Result<[f32; 4]> {
    let u = (u - 0.5).max(0.0);
    let v = (v - 0.5).max(0.0);
    let x0 = u as u32;
    let y0 = v as u32;
    let fx = (u - x0 as f64) as f32;
    let fy = (v - y0 as f64) as f32;
    let c00 = texel(x0, y0)?;
    let c10 = texel(x0 + 1, y0)?;
    let c01 = texel(x0, y0 + 1)?;
    let c11 = texel(x0 + 1, y0 + 1)?;
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let top = c00[i] + (c10[i] - c00[i]) * fx;
        let bottom = c01[i] + (c11[i] - c01[i]) * fx;
        out[i] = top + (bottom - top) * fy;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a `MultisampleSampleLocations` register table the way the four
    /// registers hold it: one byte per sample, low byte first.
    fn locations(words: [u32; 4]) -> [u8; MAX_SAMPLES] {
        let mut out = [0u8; MAX_SAMPLES];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = (words[i / 4] >> (8 * (i % 4))) as u8;
        }
        out
    }

    #[test]
    fn a_4x_grid_matches_deko3ds_sample_table() {
        // deko3d's `locationsMS4`, which is what Just Dance 2019 programs.
        let grid = SampleGrid::new(2, &locations([0xEAA2_6E26; 4])).unwrap();
        assert_eq!((grid.samples_x, grid.samples_y), (2, 2));
        assert_eq!(grid.count(), 4);
        // Each byte is `x | (y << 4)`, in sixteenths of a pixel.
        assert_eq!(grid.position(0), [6.0 / 16.0, 2.0 / 16.0]);
        assert_eq!(grid.position(3), [10.0 / 16.0, 14.0 / 16.0]);
        // Every sample stores in the texel its own position falls in.
        let slots: Vec<(u32, u32)> = (0..4).map(|s| grid.texel(0, 0, s)).collect();
        assert_eq!(slots, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn an_8x_grid_gives_every_sample_its_own_texel() {
        // deko3d's `locationsMS8`. Its samples are not in raster order, which
        // is the case a hard-coded index-to-texel table would get wrong.
        let table = locations([0x359D_B759, 0x1FFB_71D3, 0x359D_B759, 0x1FFB_71D3]);
        let grid = SampleGrid::new(4, &table).unwrap(); // 4x2_D3D
        assert_eq!((grid.samples_x, grid.samples_y), (4, 2));
        assert_eq!(grid.texel(0, 0, 0), (2, 0));
        let mut slots: Vec<(u32, u32)> = (0..grid.count()).map(|s| grid.texel(0, 0, s)).collect();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), 8, "two samples share a texel");
    }

    #[test]
    fn an_unwritten_location_table_falls_back_to_raster_order() {
        let grid = SampleGrid::new(2, &[0u8; MAX_SAMPLES]).unwrap();
        assert_eq!(grid.position(0), [0.25, 0.25]);
        assert_eq!(grid.position(3), [0.75, 0.75]);
        let slots: Vec<(u32, u32)> = (0..4).map(|s| grid.texel(0, 0, s)).collect();
        assert_eq!(slots, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn a_multisampled_surface_holds_more_texels_than_pixels() {
        let grid = SampleGrid::new(2, &locations([0xEAA2_6E26; 4])).unwrap();
        // Just Dance 2019's target: 2560x1440 texels is 1280x720 pixels.
        assert_eq!(grid.pixels(2560, 1440), (1280, 720));
        assert_eq!(grid.texel(1279, 719, 3), (2559, 1439));
    }

    #[test]
    fn a_single_sample_grid_leaves_coordinates_alone() {
        let grid = SampleGrid::single();
        assert!(grid.is_single());
        assert_eq!(grid.count(), 1);
        assert_eq!(grid.position(0), [0.5, 0.5]);
        assert_eq!(grid.pixels(1280, 720), (1280, 720));
        assert_eq!(grid.texel(7, 9, 0), (7, 9));
    }

    #[test]
    fn an_unknown_msaa_mode_is_reported() {
        assert!(SampleGrid::new(7, &[0u8; MAX_SAMPLES]).is_err());
    }

    #[test]
    fn gob_offsets_cover_the_gob_exactly_once() {
        let mut seen = vec![false; GOB_SIZE as usize];
        for y in 0..GOB_HEIGHT {
            for x in 0..GOB_WIDTH {
                let off = gob_offset(x, y) as usize;
                assert!(!seen[off], "offset {} produced twice", off);
                seen[off] = true;
            }
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn gob_offset_matches_known_values() {
        // Hand-checked against the Tegra GOB swizzle.
        assert_eq!(gob_offset(0, 0), 0);
        assert_eq!(gob_offset(15, 0), 15);
        assert_eq!(gob_offset(16, 0), 32);
        assert_eq!(gob_offset(32, 0), 256);
        assert_eq!(gob_offset(0, 1), 16);
        assert_eq!(gob_offset(0, 2), 64);
    }

    #[test]
    fn block_linear_covers_a_whole_surface_exactly_once() {
        let width_bytes = 128; // two GOBs wide
        let height = 16; // two GOBs tall
        let block_height = 2;
        let mut seen = vec![false; (width_bytes * height) as usize];
        for y in 0..height {
            for x in 0..width_bytes {
                let off = block_linear_offset(x, y, width_bytes, block_height) as usize;
                assert!(off < seen.len(), "offset {} out of surface", off);
                assert!(!seen[off], "offset {} produced twice", off);
                seen[off] = true;
            }
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn pitch_layout_is_row_major() {
        let layout = Layout::Pitch { pitch: 256 };
        assert_eq!(layout.offset(8, 3, 256), 3 * 256 + 8);
    }

    #[test]
    fn rgba8_roundtrip() {
        let fmt = ColorFormat::from_raw(0xD5).unwrap();
        assert_eq!(fmt.bytes_per_pixel, 4);
        let raw = fmt.encode([1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(raw as u32, 0xFF00_00FF);
        let back = fmt.decode(raw).unwrap();
        assert_eq!(back, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn bgra8_swaps_red_and_blue() {
        let fmt = ColorFormat::from_raw(0xCF).unwrap();
        let raw = fmt.encode([1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(raw as u32, 0xFFFF_0000);
    }

    #[test]
    fn half_float_roundtrip() {
        for v in [0.0f32, 1.0, 0.5, -2.5, 65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{}", v);
        }
    }

    /// Subnormal halves are their own branch, and one nothing reached until
    /// BC6H started producing them: every value below 2^-14 arrives there.
    #[test]
    fn subnormal_halves_decode_to_their_true_value() {
        // A subnormal's value is its mantissa times 2^-24, exactly.
        for mantissa in [1u16, 2, 3, 0x155, 0x200, 0x3FF] {
            let expected = mantissa as f32 * 2.0f32.powi(-24);
            assert_eq!(f16_to_f32(mantissa), expected, "half {mantissa:#06x}");
            assert_eq!(f16_to_f32(mantissa | 0x8000), -expected, "negative {mantissa:#06x}");
        }
        // The largest subnormal and the smallest normal are adjacent.
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        assert!(f16_to_f32(0x03FF) < f16_to_f32(0x0400));
    }

    #[test]
    fn unknown_format_is_reported() {
        assert!(ColorFormat::from_raw(0x77).is_err());
    }
}
