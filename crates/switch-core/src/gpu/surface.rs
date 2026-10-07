//! Surface layout and pixel formats.
//!
//! Block-linear surfaces group memory into 512-byte GOBs (64 bytes by 8 rows), stacked
//! `2^block_height_log2` GOBs tall per block; pitch surfaces are plain linear rows.

use crate::gpu::exec::ExecCtx;
use crate::{Error, Result};

/// Every `v / 255.0` an 8-bit channel can produce, indexed by the byte.
const UNORM8: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = i as f32 / 255.0;
        i += 1;
    }
    table
};

/// GOB dimensions on Fermi and later.
pub const GOB_WIDTH: u32 = 64;
pub const GOB_HEIGHT: u32 = 8;
pub const GOB_SIZE: u32 = GOB_WIDTH * GOB_HEIGHT;

/// Byte offset of `(x, y)` inside a single GOB; `x` is in bytes and both are already reduced.
#[inline]
pub fn gob_offset(x: u32, y: u32) -> u32 {
    let x = x % GOB_WIDTH;
    let y = y % GOB_HEIGHT;
    (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + (x % 16)
}

/// The row-dependent half of a block-linear address, hoistable out of an inner loop.
pub fn block_linear_row(y: u32, width_bytes: u32, block_height_gobs: u32) -> u32 {
    let block_height_gobs = block_height_gobs.max(1);
    let width_gobs = width_bytes.div_ceil(GOB_WIDTH).max(1);
    let block_row_bytes = width_gobs * GOB_SIZE * block_height_gobs;
    let rows_per_block = GOB_HEIGHT * block_height_gobs;

    let block_y = y / rows_per_block;
    let gob_y = (y % rows_per_block) / GOB_HEIGHT;
    // The `y` half of `gob_offset`.
    let in_gob = ((y % GOB_HEIGHT) / 2) * 64 + (y % 2) * 16;

    block_y * block_row_bytes + gob_y * GOB_SIZE + in_gob
}

/// The column-dependent half of a block-linear address.
pub fn block_linear_column(x_bytes: u32, block_height_gobs: u32) -> u32 {
    let block_bytes = GOB_SIZE * block_height_gobs.max(1);
    let gob_x = x_bytes / GOB_WIDTH;
    // The `x` half of `gob_offset`.
    let x = x_bytes % GOB_WIDTH;
    gob_x * block_bytes + (x / 32) * 256 + ((x % 32) / 16) * 32 + (x % 16)
}

/// `block_height_gobs` is `2^height` from the tile mode.
pub fn block_linear_offset(x_bytes: u32, y: u32, width_bytes: u32, block_height_gobs: u32) -> u32 {
    block_linear_row(y, width_bytes, block_height_gobs)
        + block_linear_column(x_bytes, block_height_gobs)
}

/// Block-linear volume offset; slices interleave inside a block (Eden's `SwizzleImpl`).
pub fn block_linear_volume_offset(
    x_bytes: u32,
    y: u32,
    z: u32,
    width_bytes: u32,
    height: u32,
    block_height_gobs: u32,
    block_depth_gobs: u32,
) -> u32 {
    let bh = block_height_gobs.max(1);
    let bd = block_depth_gobs.max(1);
    let width_gobs = width_bytes.div_ceil(GOB_WIDTH).max(1);
    let block_bytes = width_gobs * GOB_SIZE * bh * bd;
    let rows_per_block = GOB_HEIGHT * bh;
    let blocks_down = height.div_ceil(rows_per_block).max(1);
    let slice_bytes = blocks_down * block_bytes;

    let offset_z = (z / bd) * slice_bytes + (z % bd) * GOB_SIZE * bh;
    let block_y = y / GOB_HEIGHT;
    let offset_y = (block_y / bh) * block_bytes + (block_y % bh) * GOB_SIZE;
    let offset_x = (x_bytes / GOB_WIDTH) * block_bytes / width_gobs;
    offset_z + offset_y + offset_x + gob_offset(x_bytes, y)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Plain rows of `pitch` bytes.
    Pitch { pitch: u32 },
    /// `2^n` GOBs per block vertically.
    BlockLinear { block_height_gobs: u32 },
}

impl Layout {
    pub fn offset(&self, x_bytes: u32, y: u32, width_bytes: u32) -> u32 {
        self.row_offset(y, width_bytes) + self.column_offset(x_bytes)
    }

    /// The row-dependent part of [`Layout::offset`].
    pub fn row_offset(&self, y: u32, width_bytes: u32) -> u32 {
        match *self {
            Layout::Pitch { pitch } => y * pitch,
            Layout::BlockLinear { block_height_gobs } => {
                block_linear_row(y, width_bytes, block_height_gobs)
            }
        }
    }

    /// The column-dependent part of [`Layout::offset`].
    pub fn column_offset(&self, x_bytes: u32) -> u32 {
        match *self {
            Layout::Pitch { .. } => x_bytes,
            Layout::BlockLinear { block_height_gobs } => {
                block_linear_column(x_bytes, block_height_gobs)
            }
        }
    }

    /// The offset of `(x_bytes, y)` and how many bytes from there are contiguous.
    #[inline]
    pub fn run_at(&self, x_bytes: u32, y: u32, width_bytes: u32) -> (u32, u32) {
        /// The linear stretch inside a GOB: the low four bits of `x`.
        const RUN: u32 = 16;
        match *self {
            Layout::Pitch { pitch } => (y * pitch + x_bytes, width_bytes.saturating_sub(x_bytes)),
            Layout::BlockLinear { block_height_gobs } => (
                block_linear_offset(x_bytes, y, width_bytes, block_height_gobs),
                RUN - (x_bytes % RUN),
            ),
        }
    }

    /// Bytes from one array layer to the next, rounded up to whole blocks.
    pub fn layer_stride(&self, width_bytes: u32, height: u32) -> u32 {
        match *self {
            Layout::Pitch { pitch } => pitch * height,
            Layout::BlockLinear { block_height_gobs } => {
                let block_height_gobs = block_height_gobs.max(1);
                let width_gobs = width_bytes.div_ceil(GOB_WIDTH).max(1);
                let block_row_bytes = width_gobs * GOB_SIZE * block_height_gobs;
                let rows_per_block = GOB_HEIGHT * block_height_gobs;
                height.div_ceil(rows_per_block) * block_row_bytes
            }
        }
    }
}

/// The most samples per pixel any `MsaaMode` names (`4x4`).
pub const MAX_SAMPLES: usize = 16;

/// Multisampled surfaces expand spatially: each pixel owns a `samples_x` by `samples_y` texel tile.
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
    /// One sample per pixel, at the pixel centre.
    pub fn single() -> SampleGrid {
        SampleGrid {
            samples_x: 1,
            samples_y: 1,
            positions: [[0.5, 0.5]; MAX_SAMPLES],
            slots: [(0, 0); MAX_SAMPLES],
        }
    }

    /// `locations` holds one packed byte per sample, as the location registers store them.
    pub fn new(mode: u32, locations: &[u8; MAX_SAMPLES]) -> Result<SampleGrid> {
        let (samples_x, samples_y) = msaa_mode_grid(mode)?;
        let count = (samples_x * samples_y) as usize;
        // An unwritten table means each sample at its own texel's centre.
        let programmed = locations[..count].iter().any(|&b| b != 0);
        let mut positions = [[0.5f32; 2]; MAX_SAMPLES];
        for (i, position) in positions.iter_mut().enumerate().take(count) {
            *position = if programmed {
                // `x | (y << 4)` in sixteenths of a pixel (deko3d `encodeSampleLocation`).
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
        Ok(SampleGrid {
            samples_x,
            samples_y,
            positions,
            slots,
        })
    }

    /// Coverage at the pixel centre for every sample (`AntiAliasEnable = 0`); slots are unchanged.
    pub fn per_pixel_coverage(mut self) -> SampleGrid {
        self.positions = [[0.5, 0.5]; MAX_SAMPLES];
        self
    }

    pub fn count(&self) -> u32 {
        self.samples_x * self.samples_y
    }

    pub fn is_single(&self) -> bool {
        self.samples_x == 1 && self.samples_y == 1
    }

    /// Where `sample` sits inside its pixel.
    pub fn position(&self, sample: u32) -> [f32; 2] {
        self.positions[sample as usize]
    }

    /// Texel coordinates of `sample` of the pixel at `(x, y)`.
    pub fn texel(&self, x: u32, y: u32, sample: u32) -> (u32, u32) {
        let (offset_x, offset_y) = self.slot(sample);
        (x * self.samples_x + offset_x, y * self.samples_y + offset_y)
    }

    /// Which texel of a pixel's tile holds `sample`.
    pub fn slot(&self, sample: u32) -> (u32, u32) {
        self.slots[sample as usize]
    }

    /// Whether every sample sits at the centre of its texel, the only arrangement a backend
    /// can reproduce by rendering the expanded surface per texel.
    pub fn samples_at_texel_centres(&self) -> bool {
        (0..self.count()).all(|sample| {
            let (dx, dy) = self.slot(sample);
            let centre = [
                (dx as f32 + 0.5) / self.samples_x as f32,
                (dy as f32 + 0.5) / self.samples_y as f32,
            ];
            self.position(sample) == centre
        })
    }

    /// Inverse of [`SampleGrid::slot`], indexed by `dy * samples_x + dx`; only `count()` entries are valid.
    pub fn sample_of_slot(&self) -> [u32; MAX_SAMPLES] {
        let mut out = [0u32; MAX_SAMPLES];
        for sample in 0..self.count() {
            let (dx, dy) = self.slot(sample);
            out[(dy * self.samples_x + dx) as usize] = sample;
        }
        out
    }

    /// Converts a texel extent to pixels.
    pub fn pixels(&self, width: u32, height: u32) -> (u32, u32) {
        (width / self.samples_x, height / self.samples_y)
    }
}

/// The sample tile a `MsaaMode` describes, from deko3d's `MsaaMode` and `dk_image.cpp`.
fn msaa_mode_grid(mode: u32) -> Result<(u32, u32)> {
    Ok(match mode {
        0 => (1, 1),               // 1x1
        1 | 5 => (2, 1),           // 2x1, 2x1_D3D
        2 | 8 | 9 => (2, 2),       // 2x2, 2x2_VC4, 2x2_VC12
        3 | 4 | 10 | 11 => (4, 2), // 4x2, 4x2_D3D, 4x2_VC8, 4x2_VC24
        6 => (4, 4),               // 4x4
        other => {
            return Err(Error::Gpu(format!(
                "surface: unknown MsaaMode {:#x}",
                other
            )))
        }
    })
}

/// The texel a sample's location falls in is its slot; falls back to raster order if two collide.
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

/// How a stored pixel becomes the host's `0xAABBGGRR` word by a byte shuffle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostShuffle {
    swap_red_blue: bool,
    keep_alpha: u32,
    force_alpha: u32,
}

impl HostShuffle {
    #[inline(always)]
    pub fn apply(self, raw: u32) -> u32 {
        let rgb = if self.swap_red_blue {
            ((raw >> 16) & 0xFF) | (raw & 0x0000_FF00) | ((raw & 0xFF) << 16)
        } else {
            raw & 0x00FF_FFFF
        };
        rgb | (raw & self.keep_alpha) | self.force_alpha
    }
}

impl ColorFormat {
    pub fn from_raw(raw: u32) -> Result<ColorFormat> {
        let bytes_per_pixel = match raw {
            0xC0..=0xC5 => 16,                                   // RGBA32 / RGBX32
            0xC6..=0xCE => 8,                                    // RGBA16 / RG32 / RGBX16
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
        Ok(ColorFormat {
            raw,
            bytes_per_pixel,
        })
    }

    /// Channel positions and encoding; codes are Maxwell's `RenderTargetFormat` (Eden `gpu.h`).
    fn packing(&self) -> Option<Packing> {
        use Numeric::{Float, Sint, Snorm, Uint, Unorm};
        Some(match self.raw {
            0xC0 | 0xC3 => Packing::chain(4, 32, Float),
            0xC1 | 0xC4 => Packing::chain(4, 32, Sint),
            0xC2 | 0xC5 => Packing::chain(4, 32, Uint),
            0xC6 => Packing::chain(4, 16, Unorm),
            0xC7 => Packing::chain(4, 16, Snorm),
            0xC8 => Packing::chain(4, 16, Sint),
            0xC9 => Packing::chain(4, 16, Uint),
            0xCA | 0xCE => Packing::chain(4, 16, Float),
            0xCB => Packing::chain(2, 32, Float),
            0xCC => Packing::chain(2, 32, Sint),
            0xCD => Packing::chain(2, 32, Uint),
            // A8R8G8B8 and its X, Z and O variants: blue lowest.
            0xCF | 0xD0 | 0xE6 | 0xE7 | 0xFD | 0xFE => {
                Packing::packed([ch(16, 8), ch(8, 8), ch(0, 8), ch(24, 8)], Unorm)
            }
            0xD1 => Packing::packed([ch(0, 10), ch(10, 10), ch(20, 10), ch(30, 2)], Unorm),
            0xD2 => Packing::packed([ch(0, 10), ch(10, 10), ch(20, 10), ch(30, 2)], Uint),
            // A2R10G10B10: the same word with red and blue exchanged.
            0xDF => Packing::packed([ch(20, 10), ch(10, 10), ch(0, 10), ch(30, 2)], Unorm),
            0xD5 | 0xD6 | 0xF9 | 0xFA => Packing::chain(4, 8, Unorm),
            0xD7 => Packing::chain(4, 8, Snorm),
            0xD8 => Packing::chain(4, 8, Sint),
            0xD9 => Packing::chain(4, 8, Uint),
            0xDA => Packing::chain(2, 16, Unorm),
            0xDB => Packing::chain(2, 16, Snorm),
            0xDC => Packing::chain(2, 16, Sint),
            0xDD => Packing::chain(2, 16, Uint),
            0xDE => Packing::chain(2, 16, Float),
            0xE0 => Packing::packed([ch(0, 11), ch(11, 11), ch(22, 10), NO_CHANNEL], Float),
            0xE3 => Packing::chain(1, 32, Sint),
            0xE4 => Packing::chain(1, 32, Uint),
            0xE5 => Packing::chain(1, 32, Float),
            0xE8 => Packing::packed([ch(11, 5), ch(5, 6), ch(0, 5), NO_CHANNEL], Unorm),
            0xE9 | 0xF8 => Packing::packed([ch(10, 5), ch(5, 5), ch(0, 5), ch(15, 1)], Unorm),
            0xEA => Packing::chain(2, 8, Unorm),
            0xEB => Packing::chain(2, 8, Snorm),
            0xEC => Packing::chain(2, 8, Sint),
            0xED => Packing::chain(2, 8, Uint),
            0xEE => Packing::chain(1, 16, Unorm),
            0xEF => Packing::chain(1, 16, Snorm),
            0xF0 => Packing::chain(1, 16, Sint),
            0xF1 => Packing::chain(1, 16, Uint),
            0xF2 => Packing::chain(1, 16, Float),
            0xF3 => Packing::chain(1, 8, Unorm),
            0xF4 => Packing::chain(1, 8, Snorm),
            0xF5 => Packing::chain(1, 8, Sint),
            0xF6 => Packing::chain(1, 8, Uint),
            0xF7 => Packing::packed([NO_CHANNEL, NO_CHANNEL, NO_CHANNEL, ch(0, 8)], Unorm),
            _ => return None,
        })
    }

    /// The byte permutation of an 8-bit UNORM format, if it is one.
    fn order8(&self) -> Option<Order8> {
        let packing = self.packing()?;
        if packing.numeric != Numeric::Unorm {
            return None;
        }
        match packing.channels.map(|c| (c.shift, c.bits)) {
            [(0, 8), (8, 8), (16, 8), (24, 8)] => Some(Order8::Rgba),
            [(16, 8), (8, 8), (0, 8), (24, 8)] => Some(Order8::Bgra),
            _ => None,
        }
    }

    pub fn is_srgb(&self) -> bool {
        matches!(self.raw, 0xD0 | 0xD6 | 0xE7 | 0xFA)
    }

    /// The range the blend unit clamps into; `None` for float and integer targets.
    pub fn source_clamp(&self) -> Option<(f32, f32)> {
        match self.packing() {
            Some(packing) => match packing.numeric {
                Numeric::Unorm => Some((0.0, 1.0)),
                Numeric::Snorm => Some((-1.0, 1.0)),
                Numeric::Uint | Numeric::Sint | Numeric::Float => None,
            },
            None => Some((0.0, 1.0)),
        }
    }

    /// Whether alpha exists as a channel; "X", "Z" and "O" formats have the bits only.
    pub fn has_alpha(&self) -> bool {
        if matches!(
            self.raw,
            0xC3 | 0xC4 | 0xC5 | 0xCE | 0xE6 | 0xE7 | 0xF8 | 0xF9 | 0xFA | 0xFD | 0xFE
        ) {
            return false;
        }
        match self.packing() {
            Some(packing) => packing.channels[3].bits > 0,
            None => true,
        }
    }

    /// Encode a linear colour into the stored representation, applying sRGB where needed.
    pub fn encode(&self, rgba: [f32; 4]) -> Result<u128> {
        if self.is_srgb() {
            let mut encoded = rgba;
            // Alpha is linear even in an sRGB format.
            for channel in encoded.iter_mut().take(3) {
                *channel = linear_to_srgb(*channel);
            }
            return self.encode_stored(encoded);
        }
        self.encode_stored(rgba)
    }

    /// Decode one stored value into linear light.
    pub fn decode(&self, raw: u128) -> Result<[f32; 4]> {
        let mut rgba = self.decode_stored(raw)?;
        if self.is_srgb() {
            for channel in rgba.iter_mut().take(3) {
                *channel = srgb_to_linear(*channel);
            }
        }
        Ok(rgba)
    }

    fn encode_stored(&self, rgba: [f32; 4]) -> Result<u128> {
        let packing = self.packing().ok_or_else(|| {
            Error::Gpu(format!(
                "surface: encoding colour format {:#x} is not implemented",
                self.raw
            ))
        })?;
        // Fast path for the common 8-bit formats.
        if let Some(order) = self.order8() {
            return Ok(encode_order8(order, self.has_alpha(), rgba));
        }
        let mut stored = 0u128;
        for (i, channel) in packing.channels.iter().enumerate() {
            // An unused alpha slot stores one, so readers see it as opaque.
            let value = if i == 3 && !self.has_alpha() {
                1.0
            } else {
                rgba[i]
            };
            stored |= encode_channel(*channel, packing.numeric, value);
        }
        Ok(stored)
    }

    /// The host word for a stored pixel when the format is a byte shuffle of it; `None` otherwise.
    pub fn host_shuffle(&self) -> Option<HostShuffle> {
        // sRGB is a curve, not a permutation.
        if self.is_srgb() {
            return None;
        }
        // An "X" format has no alpha to read, and the host word is opaque.
        let (keep_alpha, force_alpha) = if self.has_alpha() {
            (0xFF00_0000, 0)
        } else {
            (0, 0xFF00_0000)
        };
        Some(HostShuffle {
            swap_red_blue: self.order8()? == Order8::Bgra,
            keep_alpha,
            force_alpha,
        })
    }

    /// Whether decode then encode returns the stored bytes unchanged, so copies can move bytes.
    #[inline]
    pub fn is_byte_exact(&self) -> bool {
        !self.is_srgb() && self.has_alpha() && self.order8().is_some()
    }

    /// Packs `sust.p` registers; integer channels keep their low bits instead of going through `f32`.
    pub fn encode_registers(&self, regs: [u32; 4]) -> Result<u128> {
        let Some(packing) = self.packing().filter(|p| p.numeric.is_integer()) else {
            return self.encode(regs.map(f32::from_bits));
        };
        Ok(packing
            .channels
            .iter()
            .zip(regs)
            .filter(|(channel, _)| channel.bits != 0)
            .fold(0u128, |stored, (channel, reg)| {
                stored | (u128::from(reg) & ((1u128 << channel.bits) - 1)) << channel.shift
            }))
    }

    /// Unpacks for `suld.p`; signed integers are sign-extended and missing integer alpha reads as 1.
    pub fn decode_registers(&self, raw: u128) -> Result<[u32; 4]> {
        let Some(packing) = self.packing().filter(|p| p.numeric.is_integer()) else {
            return Ok(self.decode(raw)?.map(f32::to_bits));
        };
        let mut regs = packing.channels.map(|channel| {
            if channel.bits == 0 {
                return 0;
            }
            let bits = (raw >> channel.shift) & ((1u128 << channel.bits) - 1);
            match packing.numeric {
                Numeric::Sint => {
                    (((bits << (128 - channel.bits)) as i128) >> (128 - channel.bits)) as u32
                }
                _ => bits as u32,
            }
        });
        if !self.has_alpha() {
            regs[3] = 1;
        }
        Ok(regs)
    }

    /// Unpack raw pixel bytes into a normalized RGBA colour.
    fn decode_stored(&self, raw: u128) -> Result<[f32; 4]> {
        let packing = self.packing().ok_or_else(|| {
            Error::Gpu(format!(
                "surface: decoding colour format {:#x} is not implemented",
                self.raw
            ))
        })?;
        // The mirror of `encode_stored`'s shuffle.
        if let Some(order) = self.order8() {
            return Ok(decode_order8(order, self.has_alpha(), raw));
        }
        let mut rgba = [0.0f32; 4];
        for (i, out) in rgba.iter_mut().enumerate() {
            *out = decode_channel(packing.channels[i], packing.numeric, raw);
        }
        if !self.has_alpha() {
            rgba[3] = 1.0;
        }
        Ok(rgba)
    }
}

impl ColorFormat {
    /// This format's conversions with its layout looked up once.
    pub fn codec(&self) -> Codec {
        let plain8 = match self.is_srgb() {
            true => None,
            false => self.order8().map(|order| (order, self.has_alpha())),
        };
        Codec {
            format: *self,
            plain8,
        }
    }
}

/// [`ColorFormat::decode`] and [`ColorFormat::encode`] for one format.
#[derive(Debug, Clone, Copy)]
pub struct Codec {
    format: ColorFormat,
    /// Channel order and whether alpha is stored, for a linear 8-bit UNORM format.
    plain8: Option<(Order8, bool)>,
}

impl Codec {
    pub fn decode(&self, raw: u128) -> Result<[f32; 4]> {
        match self.plain8 {
            Some((order, alpha)) => Ok(decode_order8(order, alpha, raw)),
            None => self.format.decode(raw),
        }
    }

    pub fn encode(&self, rgba: [f32; 4]) -> Result<u128> {
        match self.plain8 {
            Some((order, alpha)) => Ok(encode_order8(order, alpha, rgba)),
            None => self.format.encode(rgba),
        }
    }
}

fn decode_order8(order: Order8, alpha: bool, raw: u128) -> [f32; 4] {
    let unorm8 = |v: u32| UNORM8[(v & 0xFF) as usize];
    let v = raw as u32;
    let (c0, c1, c2, c3) = (v, v >> 8, v >> 16, v >> 24);
    let a = if alpha { unorm8(c3) } else { 1.0 };
    match order {
        Order8::Rgba => [unorm8(c0), unorm8(c1), unorm8(c2), a],
        Order8::Bgra => [unorm8(c2), unorm8(c1), unorm8(c0), a],
    }
}

fn encode_order8(order: Order8, alpha: bool, rgba: [f32; 4]) -> u128 {
    let unorm8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let (r, g, b) = (unorm8(rgba[0]), unorm8(rgba[1]), unorm8(rgba[2]));
    let a = if alpha { unorm8(rgba[3]) } else { 0xFF };
    match order {
        Order8::Rgba => (r | (g << 8) | (b << 16) | (a << 24)) as u128,
        Order8::Bgra => (b | (g << 8) | (r << 16) | (a << 24)) as u128,
    }
}

/// One channel's place in a stored pixel; width zero means absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Channel {
    shift: u32,
    bits: u32,
}

const NO_CHANNEL: Channel = Channel { shift: 0, bits: 0 };

const fn ch(shift: u32, bits: u32) -> Channel {
    Channel { shift, bits }
}

/// How a channel's bits are read as a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uint,
    Sint,
    /// 32 bits is `f32`, 16 is `f16`, 11 and 10 are unsigned halves with a 5-bit exponent.
    Float,
}

impl Numeric {
    fn is_integer(self) -> bool {
        matches!(self, Numeric::Uint | Numeric::Sint)
    }
}

/// R, G, B and A in order, and how to read each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Packing {
    channels: [Channel; 4],
    numeric: Numeric,
}

impl Packing {
    const fn packed(channels: [Channel; 4], numeric: Numeric) -> Packing {
        Packing { channels, numeric }
    }

    /// `count` channels of `bits` each, red lowest.
    const fn chain(count: u32, bits: u32, numeric: Numeric) -> Packing {
        let mut channels = [NO_CHANNEL; 4];
        let mut i = 0;
        while i < count as usize {
            channels[i] = ch(i as u32 * bits, bits);
            i += 1;
        }
        Packing { channels, numeric }
    }
}

fn encode_channel(channel: Channel, numeric: Numeric, value: f32) -> u128 {
    if channel.bits == 0 {
        return 0;
    }
    let mask = (1u128 << channel.bits) - 1;
    // A NaN casts to zero.
    let stored = match numeric {
        Numeric::Unorm => (value.clamp(0.0, 1.0) * mask as f32 + 0.5) as u128,
        Numeric::Snorm => {
            let scaled = (value.clamp(-1.0, 1.0) * (mask >> 1) as f32).round() as i128;
            scaled as u128 & mask
        }
        Numeric::Uint => (f64::from(value)).clamp(0.0, mask as f64) as u128,
        Numeric::Sint => {
            let max = (mask >> 1) as f64;
            (f64::from(value)).clamp(-max - 1.0, max) as i128 as u128 & mask
        }
        Numeric::Float => match channel.bits {
            32 => u128::from(value.to_bits()),
            16 => u128::from(f32_to_f16(value)),
            bits => u128::from(pack_small_float(value, bits - 5)),
        },
    };
    (stored & mask) << channel.shift
}

fn decode_channel(channel: Channel, numeric: Numeric, raw: u128) -> f32 {
    if channel.bits == 0 {
        return 0.0;
    }
    let mask = (1u128 << channel.bits) - 1;
    let bits = (raw >> channel.shift) & mask;
    // Two's complement in `channel.bits`, widened to the whole of an i128.
    let signed = || ((bits << (128 - channel.bits)) as i128) >> (128 - channel.bits);
    match numeric {
        Numeric::Unorm => bits as f32 / mask as f32,
        // The most negative value clamps to -1.0.
        Numeric::Snorm => (signed() as f32 / (mask >> 1) as f32).max(-1.0),
        Numeric::Uint => bits as f32,
        Numeric::Sint => signed() as f32,
        Numeric::Float => match channel.bits {
            32 => f32::from_bits(bits as u32),
            16 => f16_to_f32(bits as u16),
            b => unpack_small_float(bits as u32, b - 5),
        },
    }
}

/// Convert a linear colour channel to sRGB.
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// The inverse of [`linear_to_srgb`].
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// One unsigned `B10G11R11_FLOAT` channel: a half without its sign, mantissa narrowed; negatives become zero.
fn pack_small_float(v: f32, mantissa_bits: u32) -> u32 {
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN belongs on this side
    if !(v > 0.0) {
        return 0;
    }
    let shift = 10 - mantissa_bits;
    let half = u32::from(f32_to_f16(v)) & 0x7FFF;
    let max = (1 << (5 + mantissa_bits)) - 1;
    ((half + (1 << (shift - 1))) >> shift).min(max)
}

/// The inverse: widen the mantissa back out to a half and decode that.
fn unpack_small_float(bits: u32, mantissa_bits: u32) -> f32 {
    f16_to_f32((bits << (10 - mantissa_bits)) as u16)
}

/// Convert to a half, round to nearest even (as Maxwell's fp16 ALU and WGSL do).
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    // A NaN keeps a set mantissa so it stays a NaN.
    if exp == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let exp = exp - 127 + 15;
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        // Subnormal result; rounding may carry into the smallest normal.
        let shift = (1 - exp) as u32 + 13;
        if shift > 25 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        return sign | (round_to_nearest_even(significand, shift) as u16);
    }
    // Rounded as one number so a mantissa carry steps the exponent, up to infinity.
    sign | (round_to_nearest_even(((exp as u32) << 23) | mantissa, 13) as u16)
}

/// `value >> shift`, rounded to nearest with ties to even.
fn round_to_nearest_even(value: u32, shift: u32) -> u32 {
    let kept = value >> shift;
    let dropped = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if dropped > halfway || (dropped == halfway && kept & 1 != 0) {
        kept + 1
    } else {
        kept
    }
}

pub fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v as u32) & 0x8000) << 16;
    let exp = ((v as u32) >> 10) & 0x1F;
    let mantissa = (v as u32) & 0x3FF;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: shift the top set bit to bit 10.
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

/// A described image in GPU memory, shared by 2D blits and texture sampling.
#[derive(Debug, Clone, Copy)]
pub struct Surface {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub layout: Layout,
}

impl Surface {
    /// The pitch when there is one, the packed width otherwise.
    pub fn width_bytes(&self) -> u32 {
        match self.layout {
            Layout::Pitch { pitch } => pitch,
            Layout::BlockLinear { .. } => self.width * self.format.bytes_per_pixel,
        }
    }

    pub fn size(&self) -> u32 {
        self.layout.layer_stride(self.width_bytes(), self.height)
    }

    pub fn offset(&self, x: u32, y: u32) -> u32 {
        self.layout
            .offset(x * self.format.bytes_per_pixel, y, self.width_bytes())
    }

    pub fn texel(&self, x: u32, y: u32, ctx: &ExecCtx) -> Result<[f32; 4]> {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        let va = self.addr + self.offset(x, y) as u64;
        self.format
            .decode(ctx.read_pixel(va, self.format.bytes_per_pixel)?)
    }

    /// The stored bytes of a texel, undecoded.
    pub fn texel_raw(&self, x: u32, y: u32, ctx: &ExecCtx) -> Result<u128> {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        ctx.read_pixel(
            self.addr + self.offset(x, y) as u64,
            self.format.bytes_per_pixel,
        )
    }

    pub fn sample_point(&self, u: f64, v: f64, ctx: &ExecCtx) -> Result<[f32; 4]> {
        self.texel(u.max(0.0) as u32, v.max(0.0) as u32, ctx)
    }

    pub fn sample_bilinear(&self, u: f64, v: f64, ctx: &ExecCtx) -> Result<[f32; 4]> {
        bilinear(u, v, |x, y| self.texel(x, y, ctx))
    }
}

/// Bilinear filtering over whatever `texel` fetches.
pub fn bilinear(
    u: f64,
    v: f64,
    mut texel: impl FnMut(u32, u32) -> Result<[f32; 4]>,
) -> Result<[f32; 4]> {
    let (x0, fx) = taps(u);
    let (y0, fy) = taps(v);
    let c00 = texel(x0, y0)?;
    let c10 = texel(x0 + 1, y0)?;
    let c01 = texel(x0, y0 + 1)?;
    let c11 = texel(x0 + 1, y0 + 1)?;
    Ok(blend(c00, c10, c01, c11, fx, fy))
}

/// The first texel along one axis at coordinate `c`, and the second's weight.
#[inline]
pub fn taps(c: f64) -> (u32, f32) {
    let c = (c - 0.5).max(0.0);
    let first = c as u32;
    (first, (c - first as f64) as f32)
}

/// Four bilinear taps blended by the weights [`taps`] gives.
#[inline]
pub fn blend(
    c00: [f32; 4],
    c10: [f32; 4],
    c01: [f32; 4],
    c11: [f32; 4],
    fx: f32,
    fy: f32,
) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let top = c00[i] + (c10[i] - c00[i]) * fx;
        let bottom = c01[i] + (c11[i] - c01[i]) * fx;
        out[i] = top + (bottom - top) * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_codec_converts_as_its_format_does() {
        // RGBA8, BGRA8, sRGB RGBA8, BGRX8 and B5G6R5.
        for raw in [0xD5, 0xCF, 0xD6, 0xE6, 0xE8] {
            let format = ColorFormat::from_raw(raw).unwrap();
            let codec = format.codec();
            for word in [0u128, 0x8040_20FF, 0xFFFF_FFFF, 0x1234_5678] {
                assert_eq!(
                    codec.decode(word).unwrap(),
                    format.decode(word).unwrap(),
                    "{raw:#x}"
                );
            }
            for rgba in [[0.0, 0.5, 1.0, 0.25], [0.2, 0.7, -1.0, 2.0]] {
                assert_eq!(
                    codec.encode(rgba).unwrap(),
                    format.encode(rgba).unwrap(),
                    "{raw:#x}"
                );
            }
        }
    }

    /// One byte per sample, low byte first.
    fn locations(words: [u32; 4]) -> [u8; MAX_SAMPLES] {
        let mut out = [0u8; MAX_SAMPLES];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = (words[i / 4] >> (8 * (i % 4))) as u8;
        }
        out
    }

    /// Integers survive a formatted store and load exactly; signed ones are sign-extended.
    #[test]
    fn formatted_registers_keep_integers_exact() {
        let r32_uint = ColorFormat::from_raw(0xE4).unwrap();
        let raw = r32_uint.encode_registers([0xFFFF_FFFF, 7, 7, 7]).unwrap();
        assert_eq!(raw, 0xFFFF_FFFF);
        assert_eq!(
            r32_uint.decode_registers(raw).unwrap(),
            [0xFFFF_FFFF, 0, 0, 1]
        );

        let rg8_uint = ColorFormat::from_raw(0xED).unwrap();
        // Only the low bits of each channel are kept.
        let raw = rg8_uint.encode_registers([0x1_23, 0x45, 0, 0]).unwrap();
        assert_eq!(raw, 0x45_23);

        let r32_sint = ColorFormat::from_raw(0xE3).unwrap();
        let minus_two = (-2i32) as u32;
        let raw = r32_sint.encode_registers([minus_two, 0, 0, 0]).unwrap();
        assert_eq!(r32_sint.decode_registers(raw).unwrap()[0], minus_two);

        let rgba32_float = ColorFormat::from_raw(0xC0).unwrap();
        let regs = [1.5f32, -2.0, 0.25, 1.0].map(f32::to_bits);
        let raw = rgba32_float.encode_registers(regs).unwrap();
        assert_eq!(rgba32_float.decode_registers(raw).unwrap(), regs);
    }

    /// One GOB of depth at slice zero matches the 2D addressing.
    #[test]
    fn a_volume_one_gob_deep_addresses_like_a_surface() {
        for &bh in &[1u32, 2, 4, 8, 16] {
            for y in 0..40u32 {
                for x in (0..320u32).step_by(4) {
                    assert_eq!(
                        block_linear_volume_offset(x, y, 0, 256, 64, bh, 1),
                        block_linear_offset(x, y, 256, bh),
                        "bh {bh} at ({x}, {y})"
                    );
                }
            }
        }
    }

    /// Slices of a deep block sit one GOB apart; the next block is a whole slice along.
    #[test]
    fn a_deep_block_interleaves_its_slices() {
        // One GOB wide, one GOB tall per block, four deep.
        let at = |z| block_linear_volume_offset(0, 0, z, 64, 8, 1, 4);
        assert_eq!(at(0), 0);
        assert_eq!(at(1), 512);
        assert_eq!(at(3), 3 * 512);
        // The fifth slice starts the next block of depth.
        assert_eq!(at(4), 4 * 512);
        // Two GOBs tall: the second slice is past both of the first's.
        let tall = |z| block_linear_volume_offset(0, 0, z, 64, 16, 2, 2);
        assert_eq!(tall(0), 0);
        assert_eq!(tall(1), 2 * 512);
    }

    #[test]
    fn a_4x_grid_matches_deko3ds_sample_table() {
        // deko3d's `locationsMS4`.
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
        // deko3d's `locationsMS8`, whose samples are not in raster order.
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
        // 2560x1440 texels is 1280x720 pixels.
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
    fn an_unprogrammed_grid_puts_every_sample_at_its_texel_centre() {
        for mode in [0, 1, 2, 3, 6] {
            let grid = SampleGrid::new(mode, &[0; MAX_SAMPLES]).unwrap();
            assert!(grid.samples_at_texel_centres(), "mode {mode}");
        }
    }

    #[test]
    fn a_programmed_table_is_told_apart_from_the_centres_it_may_still_name() {
        // The centres of a 2x2 grid are 4/16 and 12/16, the same as the unprogrammed grid.
        let centres = [
            0x4 | (0x4 << 4),
            0xC | (0x4 << 4),
            0x4 | (0xC << 4),
            0xC | (0xC << 4),
        ];
        let mut locations = [0u8; MAX_SAMPLES];
        locations[..4].copy_from_slice(&centres);
        assert!(SampleGrid::new(2, &locations)
            .unwrap()
            .samples_at_texel_centres());

        // Moved a sixteenth of a pixel off centre.
        let mut moved = locations;
        moved[0] = 0x5 | (0x4 << 4);
        assert!(!SampleGrid::new(2, &moved)
            .unwrap()
            .samples_at_texel_centres());
    }

    #[test]
    fn coverage_per_pixel_is_not_the_texel_centres() {
        // Every sample at the pixel centre is a different arrangement.
        let grid = SampleGrid::new(2, &[0; MAX_SAMPLES]).unwrap();
        assert!(!grid.per_pixel_coverage().samples_at_texel_centres());
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

    /// `row_offset + column_offset` must equal `offset` for every layout.
    #[test]
    fn a_row_and_a_column_sum_to_the_offset() {
        let layouts = [
            Layout::Pitch { pitch: 320 },
            Layout::BlockLinear {
                block_height_gobs: 1,
            },
            Layout::BlockLinear {
                block_height_gobs: 2,
            },
            Layout::BlockLinear {
                block_height_gobs: 16,
            },
        ];
        for layout in layouts {
            for width_bytes in [64u32, 128, 320] {
                for y in 0..40u32 {
                    for x in 0..width_bytes {
                        assert_eq!(
                            layout.offset(x, y, width_bytes),
                            layout.row_offset(y, width_bytes) + layout.column_offset(x),
                            "{layout:?} width {width_bytes} at ({x},{y})"
                        );
                    }
                }
            }
        }
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

    /// sRGB decode followed by scan-out's encode passes bytes through unchanged.
    #[test]
    fn an_srgb_surface_survives_the_trip_to_scan_out_unchanged() {
        let srgb = ColorFormat::from_raw(0xD6).unwrap(); // RGBA8Unorm_sRGB
        assert!(srgb.is_srgb());
        for byte in 0..=255u32 {
            let stored = byte | (byte << 8) | (byte << 16) | (0xFF << 24);
            let linear = srgb.decode(stored as u128).unwrap();
            // What `Gpu::present` does with the decoded value.
            let out = (linear_to_srgb(linear[0]).clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
            assert_eq!(out, byte, "stored byte {byte}");
            assert_eq!(linear[3], 1.0, "alpha is not sRGB-encoded");
        }
    }

    #[test]
    fn an_srgb_format_round_trips_a_linear_colour() {
        let srgb = ColorFormat::from_raw(0xD6).unwrap();
        for step in 0..=32u32 {
            let value = step as f32 / 32.0;
            let raw = srgb.encode([value, value, value, 1.0]).unwrap();
            let back = srgb.decode(raw).unwrap();
            // A quarter of a step is the worst rounding error.
            assert!(
                (back[0] - value).abs() < 0.01,
                "{value} came back {}",
                back[0]
            );
        }
    }

    #[test]
    fn the_same_bytes_read_darker_through_an_srgb_format() {
        let linear = ColorFormat::from_raw(0xD5).unwrap(); // RGBA8Unorm
        let srgb = ColorFormat::from_raw(0xD6).unwrap();
        assert!(!linear.is_srgb() && srgb.is_srgb());
        let stored = 0xFF80_8080u32 as u128;
        assert!(srgb.decode(stored).unwrap()[0] < linear.decode(stored).unwrap()[0]);
        // Both ends of the range are fixed points of the transfer function.
        assert_eq!(srgb.decode(0xFF00_0000u32 as u128).unwrap()[0], 0.0);
        assert_eq!(srgb.decode(0xFFFF_FFFFu32 as u128).unwrap()[0], 1.0);
    }

    #[test]
    fn half_float_roundtrip() {
        for v in [0.0f32, 1.0, 0.5, -2.5, 65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{}", v);
        }
    }

    /// Encoding rounds to nearest even and produces subnormals.
    #[test]
    fn halves_round_to_nearest_even() {
        // 1 + 2^-11 is halfway between 0x3C00 and 0x3C01: ties to even gives 0x3C00.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        // Halfway between 0x3C01 and 0x3C02 goes to 0x3C02.
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        // Just over halfway rounds up whatever the parity.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) * 1.01), 0x3C01);
        // Subnormals are m * 2^-24; the largest is adjacent to the smallest normal.
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(-2f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_f16(0x3FF as f32 * 2f32.powi(-24)), 0x03FF);
        assert_eq!(f32_to_f16(2f32.powi(-14)), 0x0400);
        // Half the smallest subnormal ties to zero.
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-30)), 0x0000);
        // Out of range in both directions.
        assert_eq!(f32_to_f16(70000.0), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    /// Subnormal halves decode exactly.
    #[test]
    fn subnormal_halves_decode_to_their_true_value() {
        // A subnormal's value is its mantissa times 2^-24, exactly.
        for mantissa in [1u16, 2, 3, 0x155, 0x200, 0x3FF] {
            let expected = mantissa as f32 * 2.0f32.powi(-24);
            assert_eq!(f16_to_f32(mantissa), expected, "half {mantissa:#06x}");
            assert_eq!(
                f16_to_f32(mantissa | 0x8000),
                -expected,
                "negative {mantissa:#06x}"
            );
        }
        // The largest subnormal and the smallest normal are adjacent.
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        assert!(f16_to_f32(0x03FF) < f16_to_f32(0x0400));
    }

    /// `B10G11R11_FLOAT`: halves without the sign bit and some mantissa.
    #[test]
    fn b10g11r11_round_trips_what_its_mantissa_can_hold() {
        let format = ColorFormat::from_raw(0xE0).unwrap();
        assert_eq!(format.bytes_per_pixel, 4);
        // Values exactly representable in six and five mantissa bits.
        let colour = [1.0, 0.5, 0.25, 1.0];
        let stored = format.encode(colour).unwrap();
        assert_eq!(format.decode_stored(stored).unwrap(), colour);
        // Channels at bits 0, 11 and 22, red lowest; alpha is not stored.
        assert_eq!(stored & 0x7FF, 0x3C0); // 1.0: exponent 15, mantissa 0
        assert_eq!(format.decode_stored(0).unwrap(), [0.0, 0.0, 0.0, 1.0]);
        // The format is unsigned: a negative has no encoding.
        assert_eq!(format.encode([-1.0, 0.0, 0.0, 1.0]).unwrap(), 0);
        // Rounding carries into the exponent: the largest 11-bit value is finite, the next is infinity.
        let big = format.encode([65024.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(big & 0x7FF, 0x7BF);
        assert!(format.decode_stored(big).unwrap()[0].is_finite());
    }

    #[test]
    fn a_described_format_is_as_wide_as_from_raw_says() {
        for raw in 0u32..=0xFF {
            let Ok(format) = ColorFormat::from_raw(raw) else {
                continue;
            };
            let Some(packing) = format.packing() else {
                continue;
            };
            let bits = packing
                .channels
                .iter()
                .map(|c| c.shift + c.bits)
                .max()
                .unwrap();
            assert_eq!(bits.div_ceil(8), format.bytes_per_pixel, "{raw:#x}");
        }
    }

    /// The shuffle fast paths must agree with the channel table.
    #[test]
    fn the_byte_shuffle_and_the_channel_table_are_the_same_answer() {
        for raw in 0u32..=0xFF {
            let Ok(format) = ColorFormat::from_raw(raw) else {
                continue;
            };
            if format.order8().is_none() {
                continue;
            }
            let packing = format.packing().unwrap();
            for word in [0u32, 0xFFFF_FFFF, 0x1234_5678, 0x80FF_007F] {
                let stored = u128::from(word);
                let mut walked = [0.0f32; 4];
                for (i, out) in walked.iter_mut().enumerate() {
                    *out = decode_channel(packing.channels[i], packing.numeric, stored);
                }
                if !format.has_alpha() {
                    walked[3] = 1.0;
                }
                let shuffled = format.decode_stored(stored).unwrap();
                assert_eq!(shuffled, walked, "{raw:#x} decoding {word:#010x}");

                let mut packed = 0u128;
                for (i, channel) in packing.channels.iter().enumerate() {
                    let value = if i == 3 && !format.has_alpha() {
                        1.0
                    } else {
                        shuffled[i]
                    };
                    packed |= encode_channel(*channel, packing.numeric, value);
                }
                assert_eq!(
                    format.encode_stored(shuffled).unwrap(),
                    packed,
                    "{raw:#x} encoding {word:#010x}"
                );
            }
        }
    }

    /// `A8B8G8R8_SNORM` shares bytes with the UNORM code but is signed.
    #[test]
    fn a_snorm_target_stores_the_signed_range() {
        let format = ColorFormat::from_raw(0xD7).unwrap();
        assert_eq!(format.bytes_per_pixel, 4);
        let stored = format.encode([1.0, -1.0, 0.0, 0.5]).unwrap();
        assert_eq!(stored & 0xFF, 127);
        assert_eq!((stored >> 8) & 0xFF, 0x81); // -127
        assert_eq!((stored >> 16) & 0xFF, 0);
        assert_eq!((stored >> 24) & 0xFF, 64);
        let back = format.decode(stored).unwrap();
        assert_eq!(&back[..3], &[1.0, -1.0, 0.0]);
        assert_eq!(back[3], 64.0 / 127.0);
        // The one value past -1.0 clamps to it.
        assert_eq!(format.decode(0x80).unwrap()[0], -1.0);
        // Out of range in, saturated out.
        assert_eq!(format.encode([-2.0, 0.0, 0.0, 0.0]).unwrap() & 0xFF, 0x81);
    }

    /// An integer target holds the value, not a fraction of its range.
    #[test]
    fn an_integer_target_stores_the_value_it_is_given() {
        let format = ColorFormat::from_raw(0xD9).unwrap(); // A8B8G8R8_UINT
        let stored = format.encode([255.0, 300.0, -5.0, 1.0]).unwrap();
        assert_eq!(stored, 255 | (255 << 8) | (1 << 24));
        assert_eq!(format.decode(stored).unwrap(), [255.0, 255.0, 0.0, 1.0]);

        let signed = ColorFormat::from_raw(0xD8).unwrap(); // A8B8G8R8_SINT
        let stored = signed.encode([-128.0, 127.0, -200.0, 0.0]).unwrap();
        assert_eq!(stored & 0xFF, 0x80);
        assert_eq!((stored >> 16) & 0xFF, 0x80); // clamped, not wrapped
        assert_eq!(signed.decode(stored).unwrap(), [-128.0, 127.0, -128.0, 0.0]);
    }

    /// `B10G11R11_FLOAT` is not clamped to 1.0.
    #[test]
    fn the_blend_source_clamp_is_the_targets_own_range() {
        let clamp = |raw| ColorFormat::from_raw(raw).unwrap().source_clamp();
        assert_eq!(clamp(0xD5), Some((0.0, 1.0))); // A8B8G8R8_UNORM
        assert_eq!(clamp(0xD7), Some((-1.0, 1.0))); // A8B8G8R8_SNORM
        assert_eq!(clamp(0xE0), None); // B10G11R11_FLOAT
        assert_eq!(clamp(0xF2), None); // R16_FLOAT
        assert_eq!(clamp(0xCA), None); // R16G16B16A16_FLOAT
        assert_eq!(clamp(0xD9), None); // A8B8G8R8_UINT
    }

    /// Every format Eden's `RenderTargetFormat` names.
    #[test]
    fn every_named_render_target_format_can_be_written_and_read() {
        const NAMED: [u32; 54] = [
            0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD,
            0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD,
            0xDE, 0xDF, 0xE0, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
            0xEE, 0xEF, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF8, 0xF9, 0xFA,
        ];
        for raw in NAMED {
            let format = ColorFormat::from_raw(raw).unwrap();
            let packing = format.packing().unwrap();
            let colour = if packing.numeric.is_integer() {
                [3.0, 1.0, 2.0, 1.0]
            } else {
                [0.25, 0.5, 0.75, 1.0]
            };
            let stored = format
                .encode(colour)
                .unwrap_or_else(|e| panic!("{raw:#x}: {e:?}"));
            let width = format.bytes_per_pixel * 8;
            assert!(
                width == 128 || stored >> width == 0,
                "{raw:#x} stored outside its {} bytes",
                format.bytes_per_pixel
            );
            let decoded = format
                .decode(stored)
                .unwrap_or_else(|e| panic!("{raw:#x}: {e:?}"));
            for (i, channel) in packing.channels.iter().enumerate() {
                let steps = ((1u64 << channel.bits) - 1) as f32;
                // Two quantisation steps, enough for an sRGB curve.
                let (want, tolerance) = match (channel.bits, packing.numeric) {
                    _ if i == 3 && !format.has_alpha() => (1.0, 0.0),
                    (0, _) => (0.0, 0.0),
                    (_, Numeric::Unorm) => (colour[i], 2.0 / steps),
                    (_, Numeric::Snorm) => (colour[i], 4.0 / steps),
                    _ => (colour[i], 0.0),
                };
                assert!(
                    (decoded[i] - want).abs() <= tolerance,
                    "{raw:#x} channel {i}: {} for {want}",
                    decoded[i]
                );
            }
        }
    }

    #[test]
    fn unknown_format_is_reported() {
        assert!(ColorFormat::from_raw(0x77).is_err());
    }
}
