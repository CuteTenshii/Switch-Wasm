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
//! Which constant bank a `texs` immediate offsets into was an open question
//! even after that — resolved empirically against a live JKSV (real Mesa/
//! nouveau driver, not deko3d) capture: bank 15 is nouveau's reserved
//! "driver constants" buffer on every stage, and reading it at the shader's
//! own immediate offset yields exactly the expected pattern (sequential
//! small `imageId` values, `samplerId` always 0 in that capture).

use crate::gpu::exec::ExecCtx;
use crate::gpu::surface::{ColorFormat, Layout, Surface};
use crate::{Error, Result};

/// The constant-buffer bank nouveau's driver reserves for bindless texture
/// handles, on every shader stage — see this module's doc comment.
pub const DRIVER_CONSTBUF_BANK: u8 = 15;

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

/// Parse one 32-byte TIC entry (`gm200_texture.xml`'s `TIC2` domain) into a
/// [`Surface`] ready for `Surface::sample_point`/`sample_bilinear`. Only
/// `A8B8G8R8`/`UNORM`, 2D, pitch or block-linear is supported — the common
/// case for a real UI texture; anything else is a clear, honest error
/// rather than a guess.
pub fn read_image(ctx: &ExecCtx, addr: u64) -> Result<Surface> {
    let dw = |i: u64| -> Result<u32> { ctx.read_u32(addr + i * 4) };
    let dw0 = dw(0)?;
    let dw1 = dw(1)?;
    let dw2 = dw(2)?;
    let dw3 = dw(3)?;
    let dw4 = dw(4)?;
    let dw5 = dw(5)?;

    let components_sizes = dw0 & 0x7f;
    if components_sizes != 0x08 {
        return Err(Error::Gpu(format!(
            "texture: unsupported TIC COMPONENTS_SIZES {:#x} (only A8B8G8R8)",
            components_sizes
        )));
    }
    let r_data_type = (dw0 >> 7) & 0x7;
    if r_data_type != 2 {
        return Err(Error::Gpu(format!(
            "texture: unsupported TIC R_DATA_TYPE {} (only UNORM)",
            r_data_type
        )));
    }
    // A8B8G8R8 + UNORM is byte-for-byte the same layout as the display/
    // render-target path's RGBA8Unorm (`gpu::display_color_format`'s A8B8G8R8
    // row): both name channels MSB-first with red in the lowest byte.
    let format = ColorFormat::from_raw(0xD5)?;

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

    Ok(Surface { addr: tex_addr, width, height, format, layout })
}

/// Resolve a bindless `handle` (as a `texs` instruction's constant-buffer
/// read produces it) against the bound TIC/TSC pools and sample at
/// normalized texture coordinates `(u, v)`.
pub fn sample(
    ctx: &ExecCtx,
    tex_header_pool: u64,
    tex_sampler_pool: u64,
    handle: u32,
    u: f64,
    v: f64,
) -> Result<[f32; 4]> {
    let image = read_image(ctx, tex_header_pool + image_id(handle) as u64 * 32)?;
    let sampler = read_sampler(ctx, tex_sampler_pool + sampler_id(handle) as u64 * 32)?;

    let wrap = |mode: Wrap, t: f64, size: u32| -> f64 {
        let t = match mode {
            Wrap::Repeat | Wrap::Mirror => t - t.floor(),
            Wrap::ClampToEdge | Wrap::ClampToBorder => t.clamp(0.0, 1.0),
        };
        t * size as f64
    };
    let px = wrap(sampler.wrap_u, u, image.width);
    let py = wrap(sampler.wrap_v, v, image.height);

    if sampler.mag_linear {
        image.sample_bilinear(px, py, ctx)
    } else {
        image.sample_point(px, py, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::syncpt::Host1x;
    use crate::gpu::vmm::{AddressSpace, SMALL_PAGE_SIZE};
    use crate::mem::Memory;

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
        vmm.write_u32(&mut mem, base, 0x08 | (2 << 7)).unwrap();
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
        let image = read_image(&ctx, base).unwrap();
        assert_eq!(image.addr, tex_addr);
        assert_eq!(image.width, 16);
        assert_eq!(image.height, 8);
        assert_eq!(image.layout, Layout::Pitch { pitch: 64 });
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

        vmm.write_u32(&mut mem, tic_addr, 0x08 | (2 << 7)).unwrap();
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
