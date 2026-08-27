//! Texture Image Control (TIC) / Texture Sampler Control (TSC) descriptors
//! and sampling.
//!
//! Layouts are ported from envytools' `gm200_texture.xml` (Maxwell TIC) and
//! `g80_texture.xml` (TSC, unchanged since Tesla) — real hardware register
//! documentation (envytools, github.com/envytools/envytools, MIT-style
//! license), not guesses. The bindless-texture-handle convention (`imageId
//! | samplerId << 20`, each indexing 32-byte entries in their own pool)
//! matches devkitPro/deko3d's public `dkMakeTextureHandle` exactly.
//!
//! Which constant bank a `texs` immediate indexes is not a constant at all:
//! it is `TexCbIndex`, a register the driver programs
//! ([`Engine3D::tex_cb_index`](crate::gpu::engine::threed::Engine3D::tex_cb_index)).
//! nouveau reserves bank 15 for its driver constants and writes 15 there;
//! deko3d writes 0. The immediate indexes that bank in **dwords**, not
//! bytes; [`handle_offset`] carries the story of why that took a second
//! look.

use crate::gpu::exec::ExecCtx;
use crate::gpu::bcn::{self, Codec};
use crate::gpu::surface::{self, bilinear, ColorFormat, Layout};
use crate::{Error, Result};
use std::cell::RefCell;

/// What nouveau programs `TexCbIndex` to: bank 15, the buffer it reserves
/// for driver constants on every shader stage. Only a default for test
/// fixtures captured from a Mesa run — a real draw reads the register, since
/// deko3d answers 0.
pub const NOUVEAU_TEX_CB_INDEX: u8 = 15;

/// Where a `texs`'s 13-bit immediate reads its handle in
/// the bank `TexCbIndex` names, as a byte offset.
///
/// The immediate is a **dword index**, not a byte offset: nouveau's lowering
/// pass emits `tex.r = texBindBase / 4 + unit`, so the handle for texture
/// unit *n* sits at `(texBindBase / 4 + n) * 4`. Reading the immediate as a
/// byte offset lands a quarter of the way into the buffer, in the fixed
/// header nouveau keeps ahead of the handle table — which on GM107 begins
/// `0, 1, 2, 3, 4, 5, 6, 7`. That looked exactly like a handle table of
/// sequential `imageId`s with `samplerId == 0`, which is why the byte
/// reading survived: every draw resolved to a plausible handle, and every
/// draw resolved to the *same* one, so a page of text drew one glyph over
/// and over.
pub fn handle_offset(immediate: u16) -> u16 {
    immediate.wrapping_mul(4)
}

pub fn image_id(handle: u32) -> u32 {
    handle & 0xF_FFFF
}

pub fn sampler_id(handle: u32) -> u32 {
    handle >> 20
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrap {
    Repeat,
    Mirror,
    ClampToEdge,
    ClampToBorder,
}

fn decode_wrap(bits: u32) -> Wrap {
    match bits & 0x7 {
        0 => Wrap::Repeat,
        1 => Wrap::Mirror,
        2 => Wrap::ClampToEdge,
        _ => Wrap::ClampToBorder,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sampler {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub mag_linear: bool,
    pub min_linear: bool,
}

/// Parse one 32-byte TSC entry (`g80_texture.xml`'s `TSC` domain).
pub fn read_sampler(ctx: &ExecCtx, addr: u64) -> Result<Sampler> {
    let w0 = ctx.read_u32(addr)?;
    let w1 = ctx.read_u32(addr + 4)?;
    Ok(Sampler {
        wrap_u: decode_wrap(w0),
        wrap_v: decode_wrap(w0 >> 3),
        mag_linear: (w1 & 0x3) == 2,
        min_linear: ((w1 >> 4) & 0x3) == 2,
    })
}

/// Where one component of a sampled texel comes from — `TIC2`'s
/// `X_SOURCE`..`W_SOURCE`. A texture's channels are not handed to the shader
/// in memory order: the driver picks, per component, one of the stored
/// channels or a constant, which is how one `R8` image serves GL's `RED`
/// (`r,0,0,1`), `ALPHA` (`0,0,0,r`) and `LUMINANCE` (`r,r,r,1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleSource {
    Zero,
    R,
    G,
    B,
    A,
    One,
}

fn decode_swizzle_source(bits: u32) -> Result<SwizzleSource> {
    match bits & 0x7 {
        0 => Ok(SwizzleSource::Zero),
        2 => Ok(SwizzleSource::R),
        3 => Ok(SwizzleSource::G),
        4 => Ok(SwizzleSource::B),
        5 => Ok(SwizzleSource::A),
        // ONE_INT and ONE_FLOAT differ only for an integer texture, which
        // this sampler does not produce; 1 is not a documented value.
        6 | 7 => Ok(SwizzleSource::One),
        other => Err(Error::Gpu(format!("texture: unknown TIC swizzle source {other}"))),
    }
}

/// How a texture's texels are stored.
///
/// The distinction is not cosmetic: a plain texel can be read on its own,
/// while a compressed one only exists as part of a block that has to be
/// decoded whole. That changes the addressing as well as the decode — a
/// compressed surface is swizzled in units of blocks, not texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexelKind {
    Plain(ColorFormat),
    Block(Codec),
}

/// A parsed TIC: where the texels are and how to read them, plus the
/// per-component swizzle to apply once one is decoded.
#[derive(Debug, Clone, Copy)]
pub struct Texture {
    pub addr: u64,
    /// Extent in texels, which for a compressed texture is not its extent in
    /// blocks — the last block of a row or column may be partly outside it.
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    pub kind: TexelKind,
    /// The stored channels are sRGB-encoded, so sampling converts them back to
    /// linear light before the shader sees them.
    pub srgb: bool,
    pub swizzle: [SwizzleSource; 4],
    /// Bytes from one array layer to the next. Zero for a plain 2D image,
    /// where there is only ever layer 0.
    pub layer_stride: u32,
    /// How many array layers the image has, and 1 for a plain 2D one.
    ///
    /// Sampling never needed this — the shader says which layer it wants —
    /// but copying the image out does, since nothing else says where it
    /// ends.
    pub layers: u32,
}

impl Texture {
    /// Fetch and decode one texel, clamped to the texture's extent.
    pub fn texel(
        &self,
        x: u32,
        y: u32,
        layer: u32,
        ctx: &ExecCtx,
        blocks: &RefCell<BlockCache>,
    ) -> Result<[f32; 4]> {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        // The layer is not part of the swizzle: an array's slices sit back to
        // back, each one a whole surface.
        let layer_base = self.addr + u64::from(layer) * u64::from(self.layer_stride);
        let mut texel = match self.kind {
            TexelKind::Plain(format) => {
                let bpp = format.bytes_per_pixel;
                let width_bytes = match self.layout {
                    Layout::Pitch { pitch } => pitch,
                    Layout::BlockLinear { .. } => self.width * bpp,
                };
                let va = layer_base + self.layout.offset(x * bpp, y, width_bytes) as u64;
                format.decode(ctx.read_pixel(va, bpp)?)?
            }
            TexelKind::Block(codec) => {
                // The swizzle addresses a compressed surface in blocks: one
                // "pixel" of it is a whole block, and a row is as many bytes
                // as the row has blocks. Reading it in texels instead is the
                // mistake that shreds a compressed image into diagonal
                // ribbons, because the stride comes out a whole block too big.
                let bytes = codec.bytes_per_block();
                let (block_w, block_h) = codec.block_size();
                let blocks_wide = self.width.div_ceil(block_w);
                let width_bytes = match self.layout {
                    Layout::Pitch { pitch } => pitch,
                    Layout::BlockLinear { .. } => blocks_wide * bytes,
                };
                let va = layer_base
                    + self.layout.offset((x / block_w) * bytes, y / block_h, width_bytes) as u64;
                let index = ((y % block_h) * block_w + (x % block_w)) as usize;
                // Decoding a block yields every texel in it, and the next
                // fetch almost always wants one of them: bilinear asks for
                // four texels that are usually two or three of the same block,
                // and the pixel to the right asks for that block again.
                // Bound to a local first: the `Ref` a `borrow()` in a match
                // scrutinee produces lives until the end of the whole match,
                // which would still be held when the miss arm borrows mutably.
                let cached = blocks.borrow().get(va).map(|block| block[index]);
                match cached {
                    Some(texel) => texel,
                    None => {
                        let raw = ctx.read_pixel(va, bytes)?.to_le_bytes();
                        let mut cache = blocks.borrow_mut();
                        // Decoded straight into the way it will live in: an
                        // ASTC block is 2.3 KiB of texels, and building one on
                        // the stack to copy it in cost that twice per miss.
                        let way = cache.claim();
                        bcn::decode_into(codec, &raw[..bytes as usize], &mut cache.texels[way])?;
                        cache.va[way] = Some(va);
                        cache.texels[way][index]
                    }
                }
            }
        };
        if self.srgb {
            // Alpha is never sRGB-encoded, whatever the colour channels are.
            for channel in texel.iter_mut().take(3) {
                *channel = surface::srgb_to_linear(*channel);
            }
        }
        Ok(texel)
    }

    /// [`Texture::texel`] with a cache of its own, for tests that fetch a
    /// handful of texels and do not care about reuse between them.
    #[cfg(test)]
    pub fn texel_cached(&self, x: u32, y: u32, layer: u32, ctx: &ExecCtx) -> Result<[f32; 4]> {
        self.texel(x, y, layer, ctx, &RefCell::new(BlockCache::default()))
    }

    pub fn sample_point(
        &self,
        u: f64,
        v: f64,
        layer: u32,
        ctx: &ExecCtx,
        blocks: &RefCell<BlockCache>,
    ) -> Result<[f32; 4]> {
        self.texel(u.max(0.0) as u32, v.max(0.0) as u32, layer, ctx, blocks)
    }

    pub fn sample_bilinear(
        &self,
        u: f64,
        v: f64,
        layer: u32,
        ctx: &ExecCtx,
        blocks: &RefCell<BlockCache>,
    ) -> Result<[f32; 4]> {
        bilinear(u, v, |x, y| self.texel(x, y, layer, ctx, blocks))
    }
}

/// How many decoded blocks [`BlockCache`] keeps.
const BLOCK_CACHE_WAYS: usize = 4;

/// The most recently decoded compressed blocks, keyed by the address each came
/// from.
///
/// A block-compressed texel is not stored on its own: fetching one decodes the
/// whole 4x4 block — up to 12x12 for ASTC — that it sits in, and throws the
/// other 15 (or 143) away. Doing that per fetch was 7% of the Home Menu's
/// frame in ASTC decoding alone, on top of what it cost inside `texel`.
///
/// Four ways, because bilinear filtering straddling a block corner touches
/// four of them at once. Replacement is round-robin: a texture is walked in
/// scanline order, so the oldest entry is reliably the one furthest from where
/// sampling is now.
pub struct BlockCache {
    va: [Option<u64>; BLOCK_CACHE_WAYS],
    texels: Box<[[[f32; 4]; bcn::MAX_TEXELS]; BLOCK_CACHE_WAYS]>,
    next: usize,
}

impl Default for BlockCache {
    fn default() -> BlockCache {
        BlockCache {
            va: [None; BLOCK_CACHE_WAYS],
            texels: Box::new([[[0.0; 4]; bcn::MAX_TEXELS]; BLOCK_CACHE_WAYS]),
            next: 0,
        }
    }
}

impl BlockCache {
    fn get(&self, va: u64) -> Option<&[[f32; 4]; bcn::MAX_TEXELS]> {
        let way = self.va.iter().position(|held| *held == Some(va))?;
        Some(&self.texels[way])
    }

    /// Take the next way to decode into, leaving it invalid until the caller
    /// sets its address — so a decode that fails does not leave the way
    /// claiming to hold texels it never wrote.
    fn claim(&mut self) -> usize {
        let way = self.next;
        self.va[way] = None;
        self.next = (self.next + 1) % BLOCK_CACHE_WAYS;
        way
    }
}



/// How a TIC's `COMPONENTS_SIZES` and `R_DATA_TYPE` pair describes its texels.
///
/// The uncompressed sizes and [`ColorFormat`] name channels the same way —
/// most significant first — so `A8B8G8R8` and `RGBA8Unorm` are the same bytes,
/// and so are `G8R8`/`RG8Unorm` and `R8`/`R8Unorm`. Only the sizes whose
/// channel order is unambiguous under that reading are listed.
///
/// The compressed sizes are the `ImageFormat` values deko3d writes
/// (`image_formats.h`); their data type distinguishes the signed and unsigned
/// readings of the same block layout. Anything else is a clear, honest error
/// rather than a guess at where its channels sit.
fn texel_kind_for(components_sizes: u32, data_type: u32) -> Result<TexelKind> {
    fn astc(width: u8, height: u8) -> Codec {
        Codec::Astc { width, height }
    }
    const SNORM: u32 = 1;
    const UNORM: u32 = 2;
    const FLOAT: u32 = 7;
    let codec = match (components_sizes, data_type) {
        (0x24, UNORM) => Some(Codec::Bc1),      // DXT1
        (0x25, UNORM) => Some(Codec::Bc2),      // DXT23
        (0x26, UNORM) => Some(Codec::Bc3),      // DXT45
        (0x27, UNORM) => Some(Codec::Bc4Unorm), // DXN1
        (0x27, SNORM) => Some(Codec::Bc4Snorm),
        (0x28, UNORM) => Some(Codec::Bc5Unorm), // DXN2
        (0x28, SNORM) => Some(Codec::Bc5Snorm),
        (0x17, UNORM) => Some(Codec::Bc7),
        (0x10, FLOAT) => Some(Codec::Bc6hSf16), // signed half
        (0x11, FLOAT) => Some(Codec::Bc6hUf16), // unsigned half
        // ASTC's footprint is part of the format number, not a separate field.
        // The values are deko3d's `ImageFormat_ASTC_2D_*`; 0x43 is not one.
        (0x40, UNORM) => Some(astc(4, 4)),
        (0x41, UNORM) => Some(astc(5, 5)),
        (0x42, UNORM) => Some(astc(6, 6)),
        (0x44, UNORM) => Some(astc(8, 8)),
        (0x45, UNORM) => Some(astc(10, 10)),
        (0x46, UNORM) => Some(astc(12, 12)),
        (0x50, UNORM) => Some(astc(5, 4)),
        (0x51, UNORM) => Some(astc(6, 5)),
        (0x52, UNORM) => Some(astc(8, 6)),
        (0x53, UNORM) => Some(astc(10, 8)),
        (0x54, UNORM) => Some(astc(12, 10)),
        (0x55, UNORM) => Some(astc(8, 5)),
        (0x56, UNORM) => Some(astc(10, 5)),
        (0x57, UNORM) => Some(astc(10, 6)),
        _ => None,
    };
    if let Some(codec) = codec {
        return Ok(TexelKind::Block(codec));
    }
    // An HDR title renders into a float surface and samples it back to
    // tonemap: "A Short Hike" composites its frame out of an
    // R16_G16_B16_A16 FLOAT target, and refusing that one sample left the
    // whole frame transparent. The float sizes here are the ones
    // [`ColorFormat`] already decodes; the rest stay an honest error.
    let raw = match (components_sizes, data_type) {
        (0x08, UNORM) => 0xD5, // A8B8G8R8          -> RGBA8Unorm
        (0x18, UNORM) => 0xEA, // G8R8              -> RG8Unorm
        (0x1D, UNORM) => 0xF3, // R8                -> R8Unorm
        (0x01, FLOAT) => 0xC0, // R32_G32_B32_A32   -> RGBA32Float
        (0x03, FLOAT) => 0xCA, // R16_G16_B16_A16   -> RGBA16Float
        (0x0F, FLOAT) => 0xE5, // R32               -> R32Float
        (other, UNORM) => {
            return Err(Error::Gpu(format!(
                "texture: unsupported TIC COMPONENTS_SIZES {other:#x}"
            )))
        }
        _ => {
            return Err(Error::Gpu(format!(
                "texture: unsupported TIC R_DATA_TYPE {data_type} for COMPONENTS_SIZES \
                 {components_sizes:#x}"
            )))
        }
    };
    Ok(TexelKind::Plain(ColorFormat::from_raw(raw)?))
}

/// Parse one 32-byte TIC entry (`gm200_texture.xml`'s `TIC2` domain) into a
/// [`Texture`] ready for `Surface::sample_point`/`sample_bilinear`. Only 2D,
/// pitch or block-linear is supported, in the texel kinds [`texel_kind_for`]
/// lists; anything else is a clear, honest error rather than a guess.
pub fn read_image(ctx: &ExecCtx, addr: u64) -> Result<Texture> {
    let dw = |i: u64| -> Result<u32> { ctx.read_u32(addr + i * 4) };
    let dw0 = dw(0)?;
    let dw1 = dw(1)?;
    let dw2 = dw(2)?;
    let dw3 = dw(3)?;
    let dw4 = dw(4)?;
    let dw5 = dw(5)?;

    let kind = texel_kind_for(dw0 & 0x7f, (dw0 >> 7) & 0x7)?;
    let swizzle = [
        decode_swizzle_source(dw0 >> 19)?,
        decode_swizzle_source(dw0 >> 22)?,
        decode_swizzle_source(dw0 >> 25)?,
        decode_swizzle_source(dw0 >> 28)?,
    ];

    let header_version = (dw2 >> 21) & 0x7;
    let (addr_low, layout) = match header_version {
        3 | 4 => {
            // BLOCKLINEAR[_COLORKEY]: 27 address MSBs, 512B-aligned.
            let block_height_gobs = 1u32 << ((dw3 >> 3) & 0x7);
            ((dw1 >> 9) << 9, Layout::BlockLinear { block_height_gobs })
        }
        1 | 2 => {
            // PITCH[_COLORKEY]: 27 address MSBs, 32B-aligned; pitch is a
            // separate 16-bit field, also in 32B units.
            let pitch = (dw3 & 0xffff) << 5;
            ((dw1 >> 5) << 5, Layout::Pitch { pitch })
        }
        other => {
            return Err(Error::Gpu(format!(
                "texture: unsupported TIC HEADER_VERSION {} (only PITCH/BLOCKLINEAR)",
                other
            )))
        }
    };
    let addr_hi = (dw2 & 0xffff) as u64;
    let tex_addr = (addr_hi << 32) | addr_low as u64;

    // `TextureType_2D` and `_2DNoMipmap` differ only in whether the image has
    // levels below the one sampled here; anything else has an extent this
    // decoder would silently misread as a 2D image's.
    // 1 and 7 are `TextureType_2D` and `_2DNoMipmap`, 3 is `_2DArray`. The
    // array's slices are laid out back to back and differ from a plain 2D
    // image only by the stride between them, so the same decode serves both.
    let texture_type = (dw4 >> 23) & 0xF;
    if !matches!(texture_type, 1 | 3 | 7) {
        return Err(Error::Gpu(format!(
            "texture: TIC TextureType {texture_type} is not a 2D image or array"
        )));
    }
    let srgb = (dw4 >> 22) & 1 != 0;
    let width = (dw4 & 0xffff) + 1;
    let height = (dw5 & 0xffff) + 1;
    // `DEPTH_MINUS_ONE`, which only an array has: a plain 2D image leaves
    // whatever is in the field, and reading it would give it layers it has
    // no memory for.
    let layers = if texture_type == 3 { ((dw5 >> 16) & 0x3fff) + 1 } else { 1 };
    // The TIC carries no layer stride: it is the size of one swizzled slice,
    // worked out from the extent and the layout the same way the offset of a
    // texel inside one is.
    let width_bytes = match kind {
        TexelKind::Plain(format) => width * format.bytes_per_pixel,
        TexelKind::Block(codec) => {
            let (block_w, _) = codec.block_size();
            width.div_ceil(block_w) * codec.bytes_per_block()
        }
    };
    let layer_height = match kind {
        TexelKind::Plain(_) => height,
        TexelKind::Block(codec) => {
            let (_, block_h) = codec.block_size();
            height.div_ceil(block_h)
        }
    };
    let layer_stride = layout.layer_stride(width_bytes, layer_height);

    let texture = Texture {
        addr: tex_addr,
        width,
        height,
        layout,
        kind,
        srgb,
        swizzle,
        layer_stride,
        layers,
    };
    if let Some(dir) = dump_textures() {
        dump_texture(&texture, ctx, dir)?;
    }
    if trace_textures() {
        eprintln!(
            "[tex] {addr:#x} dw={dw0:#010x},{dw1:#010x},{dw2:#010x},{dw3:#010x},{dw4:#010x},\
             {dw5:#010x} sizes={:#04x} type={:#x} -> {texture:x?}",
            dw0 & 0x7f,
            (dw0 >> 7) & 0x7,
        );
    }
    Ok(texture)
}

/// Where to write every texture as a PPM (`DUMP_TEX=<dir>`), if anywhere.
fn dump_textures() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("DUMP_TEX").ok()).as_deref()
}

/// Decode a whole texture through the same path a sample takes and write it
/// out, so that "the image is the right shape and the wrong colour" can be
/// pinned on the decoder or ruled out in one look.
fn dump_texture(texture: &Texture, ctx: &ExecCtx, dir: &str) -> Result<()> {
    let (w, h) = (texture.width.min(4096), texture.height.min(4096));
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    let blocks = RefCell::new(BlockCache::default());
    for y in 0..h {
        for x in 0..w {
            let texel = texture.texel(x, y, 0, ctx, &blocks)?;
            for channel in &texel[..3] {
                ppm.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    let path = format!("{dir}/tex_{:x}_{w}x{h}.ppm", texture.addr);
    std::fs::write(&path, ppm).map_err(|e| Error::Io(format!("{path}: {e}")))?;
    eprintln!("[tex] wrote {path}");
    Ok(())
}

/// Whether to print every TIC as it is parsed (`TRACE_TEX=1`).
///
/// Its own switch rather than `TRACE_GPU`'s: a frame is a million method
/// traces and a few dozen textures, and the descriptor is what says why a
/// correctly-shaped image came out the wrong colour.
fn trace_textures() -> bool {
    crate::env_flag!("TRACE_TEX")
}

/// Rearrange a decoded texel into what the shader reads.
fn apply_swizzle(swizzle: [SwizzleSource; 4], texel: [f32; 4]) -> [f32; 4] {
    swizzle.map(|source| match source {
        SwizzleSource::Zero => 0.0,
        SwizzleSource::R => texel[0],
        SwizzleSource::G => texel[1],
        SwizzleSource::B => texel[2],
        SwizzleSource::A => texel[3],
        SwizzleSource::One => 1.0,
    })
}

/// What one bindless handle resolves to: its TIC and its TSC, both parsed.
///
/// Kept as a pair because they are looked up together and, for a given
/// handle, decode to the same thing for every pixel of a draw.
#[derive(Debug, Clone, Copy)]
pub struct Descriptors {
    pub texture: Texture,
    pub sampler: Sampler,
}

/// Resolve a bindless `handle` (as a `texs` instruction's constant-buffer
/// read produces it) against the bound TIC/TSC pools.
pub fn read_descriptors(
    ctx: &ExecCtx,
    tex_header_pool: u64,
    tex_sampler_pool: u64,
    handle: u32,
) -> Result<Descriptors> {
    Ok(Descriptors {
        texture: read_image(ctx, tex_header_pool + image_id(handle) as u64 * 32)?,
        sampler: read_sampler(ctx, tex_sampler_pool + sampler_id(handle) as u64 * 32)?,
    })
}

/// Resolve a bindless `handle` against the bound TIC/TSC pools and sample at
/// normalized texture coordinates `(u, v)`.
pub fn sample(
    ctx: &ExecCtx,
    tex_header_pool: u64,
    tex_sampler_pool: u64,
    handle: u32,
    u: f64,
    v: f64,
    layer: u32,
) -> Result<[f32; 4]> {
    let descriptors = read_descriptors(ctx, tex_header_pool, tex_sampler_pool, handle)?;
    sample_with(ctx, &descriptors, u, v, layer, &RefCell::new(BlockCache::default()))
}

/// Sample already-resolved descriptors at normalized coordinates `(u, v)` of
/// array layer `layer`, which is 0 for everything that is not an array.
pub fn sample_with(
    ctx: &ExecCtx,
    d: &Descriptors,
    u: f64,
    v: f64,
    layer: u32,
    blocks: &RefCell<BlockCache>,
) -> Result<[f32; 4]> {
    let (texture, sampler) = (&d.texture, d.sampler);
    let image = texture;

    let wrap = |mode: Wrap, t: f64, size: u32| -> f64 {
        let t = match mode {
            Wrap::Repeat => t - t.floor(),
            // Mirrored repeat folds every other period back on itself, which
            // is the whole difference from plain repeat: treating it as
            // repeat puts a seam where the reflection should be.
            Wrap::Mirror => {
                let period = t.rem_euclid(2.0);
                if period > 1.0 {
                    2.0 - period
                } else {
                    period
                }
            }
            Wrap::ClampToEdge | Wrap::ClampToBorder => t.clamp(0.0, 1.0),
        };
        t * size as f64
    };
    let px = wrap(sampler.wrap_u, u, image.width);
    let py = wrap(sampler.wrap_v, v, image.height);

    // Filtering happens before the swizzle, which is free to do: selecting a
    // component commutes with interpolating each one.
    let texel = if sampler.mag_linear {
        image.sample_bilinear(px, py, layer, ctx, blocks)?
    } else {
        image.sample_point(px, py, layer, ctx, blocks)?
    };
    Ok(apply_swizzle(texture.swizzle, texel))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

    /// `X_SOURCE=R, Y_SOURCE=G, Z_SOURCE=B, W_SOURCE=A` in TIC dword 0 — the
    /// identity swizzle, which a plain RGBA texture carries.
    const IDENTITY_SWIZZLE: u32 = (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
    /// `TextureType_2D` in dword4, which every real TIC carries.
    const TYPE_2D: u32 = 1 << 23;

    fn harness() -> (Memory, AddressSpace, u64) {
        let mut mem = Memory::new();
        mem.map_zero(0x8000_0000, 0x2000).unwrap();
        let mut vmm = AddressSpace::new();
        let base = vmm.map(0x8000_0000, 0x2000, 1, 0, SMALL_PAGE_SIZE, 0, 0).unwrap();
        (mem, vmm, base)
    }

    #[test]
    fn image_id_and_sampler_id_match_dkmaketexturehandle() {
        // dkMakeTextureHandle(imageId, samplerId) = imageId | (samplerId << 20).
        let handle = 7u32 | (3u32 << 20);
        assert_eq!(image_id(handle), 7);
        assert_eq!(sampler_id(handle), 3);
    }

    #[test]
    fn reads_a_pitch_a8b8g8r8_unorm_tic() {
        let (mut mem, vmm, base) = harness();
        let tex_addr = base + 0x400;
        // dword0: COMPONENTS_SIZES=A8B8G8R8(0x08), R_DATA_TYPE=UNORM(2)@bit7.
        vmm.write_u32(&mut mem, base, 0x08 | (2 << 7) | IDENTITY_SWIZZLE).unwrap();
        // dword1: pitch-aligned low address bits (32B units).
        vmm.write_u32(&mut mem, base + 4, ((tex_addr as u32) >> 5) << 5).unwrap();
        // dword2: HEADER_VERSION=PITCH(2)@bits21-23, plus address hi16.
        vmm.write_u32(&mut mem, base + 8, ((tex_addr >> 32) as u32) | (2 << 21)).unwrap();
        // dword3: pitch = 64 bytes, in 32B units -> 2.
        vmm.write_u32(&mut mem, base + 12, 2).unwrap();
        // dword4: width - 1 = 15 (width 16).
        vmm.write_u32(&mut mem, base + 16, 15 | TYPE_2D).unwrap();
        // dword5: height - 1 = 7 (height 8).
        vmm.write_u32(&mut mem, base + 20, 7).unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let image = read_image(&ctx, base).unwrap();
        assert_eq!(image.addr, tex_addr);
        assert_eq!(image.width, 16);
        assert_eq!(image.height, 8);
        assert_eq!(image.layout, Layout::Pitch { pitch: 64 });
    }

    /// A BC1 block whose endpoints are equal decodes to one flat colour, which
    /// makes a block's identity readable from any texel inside it.
    fn flat_bc1_block(rgb565: u16) -> u64 {
        // Both endpoints the same colour and every index zero, as the eight
        // little-endian bytes of one block.
        rgb565 as u64 | ((rgb565 as u64) << 16)
    }

    const RED: u16 = 0xF800;
    const GREEN: u16 = 0x07E0;
    const BLUE: u16 = 0x001F;
    const WHITE: u16 = 0xFFFF;

    /// Write a BC1 TIC for a `width` x `height` texel image at `tex_addr`.
    fn write_bc1_tic(
        mem: &mut Memory,
        vmm: &AddressSpace,
        tic_addr: u64,
        tex_addr: u64,
        width: u32,
        height: u32,
        pitch_bytes: Option<u32>,
    ) {
        // COMPONENTS_SIZES = DXT1 (0x24), R_DATA_TYPE = UNORM.
        vmm.write_u32(mem, tic_addr, 0x24 | (2 << 7) | IDENTITY_SWIZZLE).unwrap();
        match pitch_bytes {
            Some(pitch) => {
                vmm.write_u32(mem, tic_addr + 4, ((tex_addr as u32) >> 5) << 5).unwrap();
                vmm.write_u32(mem, tic_addr + 8, ((tex_addr >> 32) as u32) | (2 << 21)).unwrap();
                vmm.write_u32(mem, tic_addr + 12, pitch / 32).unwrap();
            }
            None => {
                vmm.write_u32(mem, tic_addr + 4, ((tex_addr as u32) >> 9) << 9).unwrap();
                vmm.write_u32(mem, tic_addr + 8, ((tex_addr >> 32) as u32) | (3 << 21)).unwrap();
                vmm.write_u32(mem, tic_addr + 12, 0).unwrap(); // block_height_gobs = 1
            }
        }
        vmm.write_u32(mem, tic_addr + 16, (width - 1) | TYPE_2D).unwrap();
        vmm.write_u32(mem, tic_addr + 20, height - 1).unwrap();
    }

    /// A compressed surface is addressed in blocks. Four 4x4 blocks laid out
    /// across one 16-texel row must be found at 8-byte steps, not at the
    /// 8-bytes-per-*texel* steps a decoder that forgot the distinction would
    /// use — which is the difference between an image and diagonal ribbons.
    #[test]
    fn a_pitch_bc1_texture_is_addressed_in_blocks() {
        let (mut mem, vmm, base) = harness();
        let tic_addr = base;
        let tex_addr = base + 0x400;
        write_bc1_tic(&mut mem, &vmm, tic_addr, tex_addr, 16, 8, Some(32));

        // Row 0: red, green, blue, white. Row 1 starts one 32-byte pitch on.
        for (i, colour) in [RED, GREEN, BLUE, WHITE].into_iter().enumerate() {
            vmm.write_u64(&mut mem, tex_addr + i as u64 * 8, flat_bc1_block(colour)).unwrap();
        }
        vmm.write_u64(&mut mem, tex_addr + 32, flat_bc1_block(GREEN)).unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let texture = read_image(&ctx, tic_addr).unwrap();
        assert_eq!(texture.kind, TexelKind::Block(Codec::Bc1));
        assert_eq!(texture.width, 16);

        // Every texel of a block reads as that block's colour.
        assert_eq!(texture.texel_cached(0, 0, 0, &ctx).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(texture.texel_cached(3, 3, 0, &ctx).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(texture.texel_cached(4, 0, 0, &ctx).unwrap(), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(texture.texel_cached(8, 2, 0, &ctx).unwrap(), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(texture.texel_cached(15, 3, 0, &ctx).unwrap(), [1.0, 1.0, 1.0, 1.0]);
        // The next block *row* is one pitch on, not one block on.
        assert_eq!(texture.texel_cached(0, 4, 0, &ctx).unwrap(), [0.0, 1.0, 0.0, 1.0]);
    }

    /// The same, swizzled: the block-linear stride of a compressed surface is
    /// its row length in *blocks*, so a whole GOB holds eight block rows
    /// rather than eight texel rows.
    #[test]
    fn a_block_linear_bc1_texture_swizzles_in_blocks() {
        use crate::gpu::surface::block_linear_offset;
        let (mut mem, vmm, base) = harness();
        let tic_addr = base;
        let tex_addr = base + 0x600; // 512-byte aligned, as BLOCKLINEAR requires
        write_bc1_tic(&mut mem, &vmm, tic_addr, tex_addr, 16, 8, None);

        let width_bytes = 4 * 8; // four blocks per row, eight bytes each
        let place = |mem: &mut Memory, bx: u32, by: u32, colour: u16| {
            let at = block_linear_offset(bx * 8, by, width_bytes, 1);
            vmm.write_u64(mem, tex_addr + at as u64, flat_bc1_block(colour)).unwrap();
        };
        place(&mut mem, 0, 0, RED);
        place(&mut mem, 3, 0, WHITE);
        place(&mut mem, 0, 1, BLUE);

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let texture = read_image(&ctx, tic_addr).unwrap();
        assert_eq!(texture.layout, Layout::BlockLinear { block_height_gobs: 1 });
        assert_eq!(texture.texel_cached(1, 1, 0, &ctx).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(texture.texel_cached(13, 2, 0, &ctx).unwrap(), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(texture.texel_cached(2, 5, 0, &ctx).unwrap(), [0.0, 0.0, 1.0, 1.0]);
    }

    /// An sRGB texture stores encoded values and hands the shader linear ones.
    /// An ASTC texture is addressed in blocks like any other compressed one,
    /// but its footprint is neither square nor four: 8x5 here, so a decoder
    /// that transposed the two would put row 5 in the wrong block.
    #[test]
    fn an_astc_texture_is_addressed_by_its_own_footprint() {
        let (mut mem, vmm, base) = harness();
        let tic_addr = base;
        let tex_addr = base + 0x400;
        // COMPONENTS_SIZES = ASTC_2D_8X5 (0x55), UNORM, pitch layout.
        vmm.write_u32(&mut mem, tic_addr, 0x55 | (2 << 7) | IDENTITY_SWIZZLE).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 4, ((tex_addr as u32) >> 5) << 5).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 8, ((tex_addr >> 32) as u32) | (2 << 21)).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 12, 1).unwrap(); // pitch 32 bytes = two blocks
        vmm.write_u32(&mut mem, tic_addr + 16, 15 | TYPE_2D).unwrap(); // width 16
        vmm.write_u32(&mut mem, tic_addr + 20, 9).unwrap(); // height 10

        // A void-extent block is the simplest valid ASTC block: one flat
        // colour, with the "extends nowhere" encoding in its extent fields.
        let place = |mem: &mut Memory, at: u64, r: u16, g: u16, b: u16| {
            let low: u64 =
                0x1FC | (0x1FFF << 12) | (0x1FFF << 25) | (0x1FFF << 38) | (0x1FFF << 51);
            let high: u64 =
                r as u64 | ((g as u64) << 16) | ((b as u64) << 32) | ((0xFFFFu64) << 48);
            vmm.write_u64(mem, tex_addr + at, low).unwrap();
            vmm.write_u64(mem, tex_addr + at + 8, high).unwrap();
        };
        place(&mut mem, 0, 65535, 0, 0); // block (0, 0) red
        place(&mut mem, 16, 0, 65535, 0); // block (1, 0) green
        place(&mut mem, 32, 0, 0, 65535); // block (0, 1) blue
        place(&mut mem, 48, 65535, 65535, 65535); // block (1, 1) white

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let texture = read_image(&ctx, tic_addr).unwrap();
        assert_eq!(texture.kind, TexelKind::Block(Codec::Astc { width: 8, height: 5 }));

        assert_eq!(texture.texel_cached(0, 0, 0, &ctx).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(texture.texel_cached(7, 4, 0, &ctx).unwrap(), [1.0, 0.0, 0.0, 1.0], "last texel of it");
        assert_eq!(texture.texel_cached(8, 0, 0, &ctx).unwrap(), [0.0, 1.0, 0.0, 1.0], "next block across");
        // Row 5 is the second block row, which it would not be for a 5-tall
        // footprint read as 8 tall.
        assert_eq!(texture.texel_cached(0, 5, 0, &ctx).unwrap(), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(texture.texel_cached(15, 9, 0, &ctx).unwrap(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn an_srgb_texture_samples_in_linear_light() {
        let (mut mem, vmm, base) = harness();
        let tic_addr = base;
        let tex_addr = base + 0x400;
        write_bc1_tic(&mut mem, &vmm, tic_addr, tex_addr, 16, 8, Some(32));
        // Mid grey: 0x8410 is RGB565's closest to 50% in every channel.
        vmm.write_u64(&mut mem, tex_addr, flat_bc1_block(0x8410)).unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let linear = read_image(&ctx, tic_addr).unwrap();
        assert!(!linear.srgb);
        let plain = linear.texel_cached(0, 0, 0, &ctx).unwrap()[0];

        // Set the sRGB bit and read the same bytes again.
        vmm.write_u32(&mut mem, tic_addr + 16, 15 | TYPE_2D | (1 << 22)).unwrap();
        let ctx =
            ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let encoded = read_image(&ctx, tic_addr).unwrap();
        assert!(encoded.srgb);
        let converted = encoded.texel_cached(0, 0, 0, &ctx).unwrap();
        assert!(converted[0] < plain, "sRGB decoding darkens a mid grey");
        // 565's mid grey expands to 132/255 = 0.5176, whose linear value is
        // ((0.5176 + 0.055) / 1.055) ^ 2.4.
        assert!((converted[0] - 0.2307).abs() < 0.001, "got {}", converted[0]);
        assert_eq!(converted[3], 1.0, "alpha is never sRGB-encoded");
    }

    /// An HDR pass samples its own float render target back. `0x03`/FLOAT is
    /// the one "A Short Hike" composites its frame through.
    #[test]
    fn the_float_tic_formats_map_to_their_surface_format() {
        let plain = |sizes, ty| match texel_kind_for(sizes, ty).unwrap() {
            TexelKind::Plain(format) => format,
            other => panic!("{sizes:#x}/{ty} is {other:?}, not a plain format"),
        };
        assert_eq!(plain(0x03, 7).raw, 0xCA); // R16_G16_B16_A16 -> RGBA16Float
        assert_eq!(plain(0x03, 7).bytes_per_pixel, 8);
        assert_eq!(plain(0x01, 7).raw, 0xC0); // R32_G32_B32_A32 -> RGBA32Float
        assert_eq!(plain(0x0F, 7).raw, 0xE5); // R32             -> R32Float
        // The UNORM readings of those sizes are a different format, and none
        // of them is one this decodes.
        assert!(texel_kind_for(0x03, 2).is_err());
        // A size that is not a format at all still reports as one.
        assert!(texel_kind_for(0x09, 7).is_err());
    }

    #[test]
    fn every_compressed_tic_format_maps_to_its_codec() {
        // BC6H's two data types are the signed and unsigned half readings of
        // the same block layout, and deko3d numbers SF16 below UF16.
        assert_eq!(texel_kind_for(0x10, 7).unwrap(), TexelKind::Block(Codec::Bc6hSf16));
        assert_eq!(texel_kind_for(0x11, 7).unwrap(), TexelKind::Block(Codec::Bc6hUf16));
        assert_eq!(texel_kind_for(0x24, 2).unwrap(), TexelKind::Block(Codec::Bc1));
        assert_eq!(texel_kind_for(0x26, 2).unwrap(), TexelKind::Block(Codec::Bc3));
        assert_eq!(texel_kind_for(0x27, 1).unwrap(), TexelKind::Block(Codec::Bc4Snorm));
        assert_eq!(texel_kind_for(0x28, 2).unwrap(), TexelKind::Block(Codec::Bc5Unorm));
        assert_eq!(texel_kind_for(0x17, 2).unwrap(), TexelKind::Block(Codec::Bc7));

        // Every ASTC footprint Maxwell can name, and the fourteen of them are
        // not contiguous: 0x43 is not a format.
        let footprints = [
            (0x40, 4, 4), (0x41, 5, 5), (0x42, 6, 6), (0x44, 8, 8),
            (0x45, 10, 10), (0x46, 12, 12), (0x50, 5, 4), (0x51, 6, 5),
            (0x52, 8, 6), (0x53, 10, 8), (0x54, 12, 10), (0x55, 8, 5),
            (0x56, 10, 5), (0x57, 10, 6),
        ];
        for (raw, width, height) in footprints {
            assert_eq!(
                texel_kind_for(raw, 2).unwrap(),
                TexelKind::Block(Codec::Astc { width, height }),
                "COMPONENTS_SIZES {raw:#x}"
            );
        }
        assert!(texel_kind_for(0x43, 2).is_err(), "0x43 is not an ASTC footprint");
        // A block never carries more texels than the largest footprint.
        for (raw, width, height) in footprints {
            let TexelKind::Block(codec) = texel_kind_for(raw, 2).unwrap() else { panic!() };
            assert_eq!(codec.block_size(), (width as u32, height as u32));
            assert_eq!(codec.bytes_per_block(), 16);
            assert!((width as usize) * (height as usize) <= crate::gpu::bcn::MAX_TEXELS);
        }
    }

    #[test]
    fn mirrored_repeat_folds_alternate_periods_back() {
        // 1.25 is a quarter into the second period, which mirrors to 0.75;
        // plain repeat would answer 0.25 and put a seam at every integer.
        let (mut mem, vmm, base) = harness();
        let tic_addr = base;
        let tsc_addr = base + 0x100;
        let tex_addr = base + 0x400;

        vmm.write_u32(&mut mem, tic_addr, 0x08 | (2 << 7) | IDENTITY_SWIZZLE).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 4, ((tex_addr as u32) >> 5) << 5).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 8, ((tex_addr >> 32) as u32) | (2 << 21)).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 12, 1).unwrap(); // pitch 32
        vmm.write_u32(&mut mem, tic_addr + 16, 3 | TYPE_2D).unwrap(); // width 4
        vmm.write_u32(&mut mem, tic_addr + 20, 0).unwrap(); // height 1
        // MIRROR on u, clamp on v; nearest on both.
        vmm.write_u32(&mut mem, tsc_addr, 1 | (2 << 3)).unwrap();
        vmm.write_u32(&mut mem, tsc_addr + 4, 1 | (1 << 4)).unwrap();
        for (i, colour) in [0xFF0000FFu32, 0xFF00FF00, 0xFFFF0000, 0xFFFFFFFF].iter().enumerate() {
            vmm.write_u32(&mut mem, tex_addr + 4 * i as u64, *colour).unwrap();
        }

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };

        // u = 1.9 mirrors to 0.1 -> texel 0 (red); plain repeat would give
        // 0.9 -> texel 3 (white).
        assert_eq!(sample(&ctx, tic_addr, tsc_addr, 0, 1.9, 0.5, 0).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        // u = -0.1 mirrors to 0.1 as well.
        assert_eq!(sample(&ctx, tic_addr, tsc_addr, 0, -0.1, 0.5, 0).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        // Inside the first period nothing changes.
        assert_eq!(sample(&ctx, tic_addr, tsc_addr, 0, 0.9, 0.5, 0).unwrap(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn reads_a_sampler_and_maps_repeat_and_linear() {
        let (mut mem, vmm, base) = harness();
        // ADDRESS_U=REPEAT(0), ADDRESS_V=CLAMP_TO_EDGE(2)@bits3-5.
        vmm.write_u32(&mut mem, base, 0 | (2 << 3)).unwrap();
        // MAG_FILTER=LINEAR(2), MIN_FILTER=LINEAR(2)@bits4-5.
        vmm.write_u32(&mut mem, base + 4, 2 | (2 << 4)).unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let sampler = read_sampler(&ctx, base).unwrap();
        assert_eq!(sampler.wrap_u, Wrap::Repeat);
        assert_eq!(sampler.wrap_v, Wrap::ClampToEdge);
        assert!(sampler.mag_linear);
        assert!(sampler.min_linear);
    }

    #[test]
    fn samples_a_known_texture_at_known_normalized_uv() {
        let (mut mem, vmm, base) = harness();
        let tic_addr = base;
        let tsc_addr = base + 0x100;
        let tex_addr = base + 0x400;

        vmm.write_u32(&mut mem, tic_addr, 0x08 | (2 << 7) | IDENTITY_SWIZZLE).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 4, ((tex_addr as u32) >> 5) << 5).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 8, ((tex_addr >> 32) as u32) | (2 << 21)).unwrap();
        vmm.write_u32(&mut mem, tic_addr + 12, 1).unwrap(); // PITCH_BITS_20_TO_5 = 1 -> pitch = 32
        vmm.write_u32(&mut mem, tic_addr + 16, 1 | TYPE_2D).unwrap(); // width - 1 = 1 (width 2)
        vmm.write_u32(&mut mem, tic_addr + 20, 1).unwrap(); // height - 1 = 1 (height 2)

        // NEAREST filtering, both axes clamp-to-edge.
        vmm.write_u32(&mut mem, tsc_addr, 2 | (2 << 3)).unwrap();
        vmm.write_u32(&mut mem, tsc_addr + 4, 1 | (1 << 4)).unwrap();

        // A 2x2 RGBA8 texture: (0,0)=red, (1,0)=green, (0,1)=blue, (1,1)=white.
        vmm.write_u32(&mut mem, tex_addr, 0xFF0000FF).unwrap(); // r=1,g=0,b=0,a=1 (R in low byte)
        vmm.write_u32(&mut mem, tex_addr + 4, 0xFF00FF00).unwrap(); // g=1
        vmm.write_u32(&mut mem, tex_addr + 32, 0xFFFF0000).unwrap(); // b=1
        vmm.write_u32(&mut mem, tex_addr + 36, 0xFFFFFFFF).unwrap(); // white

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };

        let handle = 0u32; // imageId=0, samplerId=0; both pools point straight at our entries.
        let red = sample(&ctx, tic_addr, tsc_addr, handle, 0.25, 0.25, 0).unwrap();
        assert_eq!(red, [1.0, 0.0, 0.0, 1.0]);
        let green = sample(&ctx, tic_addr, tsc_addr, handle, 0.75, 0.25, 0).unwrap();
        assert_eq!(green, [0.0, 1.0, 0.0, 1.0]);
        let blue = sample(&ctx, tic_addr, tsc_addr, handle, 0.25, 0.75, 0).unwrap();
        assert_eq!(blue, [0.0, 0.0, 1.0, 1.0]);
    }
}
