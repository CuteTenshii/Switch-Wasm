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

fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v as u32) & 0x8000) << 16;
    let exp = ((v as u32) >> 10) & 0x1F;
    let mantissa = (v as u32) & 0x3FF;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: normalize it.
        let shift = mantissa.leading_zeros() - 21;
        let exp = 127 - 15 - shift;
        let mantissa = (mantissa << (shift + 1)) & 0x3FF;
        f32::from_bits(sign | (exp << 23) | (mantissa << 13))
    } else if exp == 0x1F {
        f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mantissa << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn unknown_format_is_reported() {
        assert!(ColorFormat::from_raw(0x77).is_err());
    }
}
