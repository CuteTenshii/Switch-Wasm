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
use crate::gpu::surface::{ColorFormat, Layout, Surface};
use crate::{Error, Result};

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

#[derive(Debug, Clone, Copy)]
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

/// A parsed TIC: where the texels are and how to read them, plus the
/// per-component swizzle to apply once one is decoded.
#[derive(Debug, Clone, Copy)]
pub struct Texture {
    pub surface: Surface,
    pub swizzle: [SwizzleSource; 4],
}

/// The [`ColorFormat`] that stores a TIC `COMPONENTS_SIZES` layout.
///
/// The two enumerations name channels the same way — most significant
/// first — so `A8B8G8R8` and `RGBA8Unorm` are the same bytes, and so are
/// `G8R8`/`RG8Unorm` and `R8`/`R8Unorm`. Only the sizes whose channel order
/// is unambiguous under that reading are listed; anything else is a clear,
/// honest error rather than a guess at where its channels sit.
fn color_format_for_components(components_sizes: u32) -> Result<ColorFormat> {
    let raw = match components_sizes {
        0x08 => 0xD5, // A8B8G8R8 -> RGBA8Unorm
        0x18 => 0xEA, // G8R8     -> RG8Unorm
        0x1D => 0xF3, // R8       -> R8Unorm
        other => {
            return Err(Error::Gpu(format!(
                "texture: unsupported TIC COMPONENTS_SIZES {other:#x}"
            )))
        }
    };
    ColorFormat::from_raw(raw)
}

/// Parse one 32-byte TIC entry (`gm200_texture.xml`'s `TIC2` domain) into a
/// [`Texture`] ready for `Surface::sample_point`/`sample_bilinear`. Only
/// `UNORM`, 2D, pitch or block-linear is supported, in the component sizes
/// [`color_format_for_components`] lists — the common cases for a real UI
/// texture; anything else is a clear, honest error rather than a guess.
pub fn read_image(ctx: &ExecCtx, addr: u64) -> Result<Texture> {
    let dw = |i: u64| -> Result<u32> { ctx.read_u32(addr + i * 4) };
    let dw0 = dw(0)?;
    let dw1 = dw(1)?;
    let dw2 = dw(2)?;
    let dw3 = dw(3)?;
    let dw4 = dw(4)?;
    let dw5 = dw(5)?;

    let format = color_format_for_components(dw0 & 0x7f)?;
    let r_data_type = (dw0 >> 7) & 0x7;
    if r_data_type != 2 {
        return Err(Error::Gpu(format!(
            "texture: unsupported TIC R_DATA_TYPE {} (only UNORM)",
            r_data_type
        )));
    }
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

    let width = (dw4 & 0xffff) + 1;
    let height = (dw5 & 0xffff) + 1;

    Ok(Texture {
        surface: Surface { addr: tex_addr, width, height, format, layout },
        swizzle,
    })
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
) -> Result<[f32; 4]> {
    let descriptors = read_descriptors(ctx, tex_header_pool, tex_sampler_pool, handle)?;
    sample_with(ctx, &descriptors, u, v)
}

/// Sample already-resolved descriptors at normalized coordinates `(u, v)`.
pub fn sample_with(ctx: &ExecCtx, d: &Descriptors, u: f64, v: f64) -> Result<[f32; 4]> {
    let (texture, sampler) = (&d.texture, d.sampler);
    let image = &texture.surface;

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
        image.sample_bilinear(px, py, ctx)?
    } else {
        image.sample_point(px, py, ctx)?
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
        vmm.write_u32(&mut mem, base + 16, 15).unwrap();
        // dword5: height - 1 = 7 (height 8).
        vmm.write_u32(&mut mem, base + 20, 7).unwrap();

        let mut host1x = Host1x::new();
        let mut stats = Default::default();
        let ctx = ExecCtx { mem: &mut mem, vmm: &vmm, host1x: &mut host1x, stats: &mut stats, trace: false };
        let image = read_image(&ctx, base).unwrap().surface;
        assert_eq!(image.addr, tex_addr);
        assert_eq!(image.width, 16);
        assert_eq!(image.height, 8);
        assert_eq!(image.layout, Layout::Pitch { pitch: 64 });
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
        vmm.write_u32(&mut mem, tic_addr + 16, 3).unwrap(); // width 4
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
        assert_eq!(sample(&ctx, tic_addr, tsc_addr, 0, 1.9, 0.5).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        // u = -0.1 mirrors to 0.1 as well.
        assert_eq!(sample(&ctx, tic_addr, tsc_addr, 0, -0.1, 0.5).unwrap(), [1.0, 0.0, 0.0, 1.0]);
        // Inside the first period nothing changes.
        assert_eq!(sample(&ctx, tic_addr, tsc_addr, 0, 0.9, 0.5).unwrap(), [1.0, 1.0, 1.0, 1.0]);
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
        vmm.write_u32(&mut mem, tic_addr + 16, 1).unwrap(); // width - 1 = 1 (width 2)
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
        let red = sample(&ctx, tic_addr, tsc_addr, handle, 0.25, 0.25).unwrap();
        assert_eq!(red, [1.0, 0.0, 0.0, 1.0]);
        let green = sample(&ctx, tic_addr, tsc_addr, handle, 0.75, 0.25).unwrap();
        assert_eq!(green, [0.0, 1.0, 0.0, 1.0]);
        let blue = sample(&ctx, tic_addr, tsc_addr, handle, 0.25, 0.75).unwrap();
        assert_eq!(blue, [0.0, 0.0, 1.0, 1.0]);
    }
}
