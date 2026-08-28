//! GM20B (Tegra X1 Maxwell) GPU model.
//!
//! The pieces mirror the hardware: [`nvmap`] is the memory-object table,
//! [`vmm`] is the graphics MMU, [`syncpt`] is host1x, [`channel`] is the
//! command processor with the [`engine`] classes behind it, and [`nvdrv`] is
//! the driver the guest actually talks to over IPC.
//!
//! Everything the GPU touches lives in the same [`Memory`] the ARM core runs
//! from, because on Tegra it genuinely does: the guest allocates a buffer, and
//! nvmap plus the GMMU just make that buffer visible at a GPU address.

pub mod bcn;
pub mod channel;
pub mod compute;
pub mod engine;
pub mod exec;
pub mod macro_engine;
pub mod nvdrv;
pub mod nvmap;
pub mod pipeline;
pub mod qmd;
pub mod raster;
pub mod renderer;
pub mod shader;
pub mod surface;
pub mod syncpt;
pub mod testing;
pub mod texture;
pub mod upload;
pub mod vmm;

use crate::mem::Memory;
use crate::{Error, Result};
use channel::Channel;
use exec::{ExecCtx, GpuStats};
use nvmap::NvMap;
use surface::{ColorFormat, Layout};
use syncpt::{Host1x, NvFence};
use vmm::AddressSpace;
use std::collections::HashMap;

/// An image ready for display, as 32-bit `0xAABBGGRR` pixels (what a canvas
/// `ImageData` wants).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl Framebuffer {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// A surface handed to the display, as described by an `NvGraphicBuffer`
/// plane in the binder parcel the compositor receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayBuffer {
    /// nvmap object id (not a handle — the parcel crosses processes).
    pub nvmap_id: u32,
    /// Byte offset of the plane inside the nvmap object.
    pub offset: u32,
    pub width: u32,
    pub height: u32,
    /// Row stride in bytes; only meaningful for a pitch-linear buffer.
    pub pitch: u32,
    /// `NvLayout`: 1 = pitch, 3 = block-linear.
    pub layout: u32,
    pub block_height_log2: u32,
    /// Low byte of `NvColorFormat` is bits-per-pixel; the whole value selects
    /// the channel order.
    pub color_format: u64,
}

/// `NvLayout_Pitch`.
pub const NV_LAYOUT_PITCH: u32 = 1;
/// `NvLayout_BlockLinear`.
pub const NV_LAYOUT_BLOCK_LINEAR: u32 = 3;

#[derive(Debug)]
pub struct Gpu {
    pub nvmap: NvMap,
    pub host1x: Host1x,
    pub address_spaces: HashMap<u32, AddressSpace>,
    pub channels: HashMap<u32, Channel>,
    pub stats: GpuStats,
    /// The most recently presented frame, ready for the host to display.
    pub framebuffer: Framebuffer,
    /// Frames presented since boot.
    pub frames: u64,
    /// Emit a per-method trace to stderr.
    pub trace: bool,
    next_as_id: u32,
    next_channel_id: u32,
}

impl Default for Gpu {
    fn default() -> Self {
        Gpu::new()
    }
}

impl Gpu {
    pub fn new() -> Gpu {
        Gpu {
            nvmap: NvMap::new(),
            host1x: Host1x::new(),
            address_spaces: HashMap::new(),
            channels: HashMap::new(),
            stats: GpuStats::default(),
            framebuffer: Framebuffer::default(),
            frames: 0,
            trace: false,
            next_as_id: 1,
            next_channel_id: 1,
        }
    }

    pub fn create_address_space(&mut self) -> u32 {
        let id = self.next_as_id;
        self.next_as_id += 1;
        self.address_spaces.insert(id, AddressSpace::new());
        id
    }

    pub fn address_space_mut(&mut self, id: u32) -> Result<&mut AddressSpace> {
        self.address_spaces
            .get_mut(&id)
            .ok_or_else(|| Error::Gpu(format!("gpu: no address space {}", id)))
    }

    /// Create a channel with its own host1x syncpoint.
    pub fn create_channel(&mut self) -> Result<u32> {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        let syncpt = self.host1x.allocate()?;
        self.channels.insert(id, Channel::new(id, syncpt));
        Ok(id)
    }

    pub fn channel_mut(&mut self, id: u32) -> Result<&mut Channel> {
        self.channels
            .get_mut(&id)
            .ok_or_else(|| Error::Gpu(format!("gpu: no channel {}", id)))
    }

    /// Execute a GPFIFO submission on `channel_id` and return the fence the
    /// guest should wait on. The work runs to completion here, so the fence is
    /// already expired by the time the caller sees it.
    pub fn submit(
        &mut self,
        channel_id: u32,
        mem: &mut Memory,
        entries: &[u64],
        increments: u32,
    ) -> Result<NvFence> {
        let syncpt = {
            let chan = self
                .channels
                .get(&channel_id)
                .ok_or_else(|| Error::Gpu(format!("gpu: submit on unknown channel {}", channel_id)))?;
            chan.syncpt
        };
        let value = self.host1x.incr_max(syncpt, increments.max(1))?;
        let fence = NvFence { id: syncpt, value };

        let chan = self.channels.get_mut(&channel_id).expect("channel checked above");
        let as_id = chan
            .as_id
            .ok_or_else(|| Error::Gpu(format!("gpu: channel {} has no address space", channel_id)))?;
        let vmm = self
            .address_spaces
            .get(&as_id)
            .ok_or_else(|| Error::Gpu(format!("gpu: channel {} bound to missing address space {}", channel_id, as_id)))?;
        let mut ctx = ExecCtx {
            mem,
            vmm,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: self.trace,
        };
        chan.submit(entries, fence, &mut ctx)?;
        Ok(fence)
    }

    /// Hand back anything a GPU backend is holding, so that whatever reads a
    /// render target next reads what was drawn into it.
    ///
    /// Called before [`Gpu::present`], which is the reader that always
    /// matters. A backend with nothing to hand back — the software
    /// rasterizer, which writes guest memory as it goes — does nothing here.
    pub fn flush_renderers(&mut self, mem: &mut Memory) -> Result<renderer::Flush> {
        let mut state = renderer::Flush::Done;
        for channel in self.channels.values_mut() {
            let Some(as_id) = channel.as_id else { continue };
            let Some(vmm) = self.address_spaces.get(&as_id) else { continue };
            let mut ctx = ExecCtx {
                mem,
                vmm,
                host1x: &mut self.host1x,
                stats: &mut self.stats,
                trace: self.trace,
            };
            // Every channel is asked, so every readback is started, and one
            // that is not ready does not stop the others from making progress.
            if channel.three_d.flush_renderer(&mut ctx)? == renderer::Flush::Pending {
                state = renderer::Flush::Pending;
            }
        }
        Ok(state)
    }

    /// Read a surface the display was handed and convert it to the RGBA8888
    /// [`Framebuffer`] the host presents. This is the scan-out step: the guest
    /// renders into a block-linear image, the compositor is given its nvmap
    /// id, and the display controller de-swizzles it on the way to the panel.
    pub fn present(&mut self, mem: &Memory, buffer: &DisplayBuffer) -> Result<()> {
        let handle = self
            .nvmap
            .by_id(buffer.nvmap_id)
            .ok_or_else(|| Error::Gpu(format!("present: no nvmap object with id {}", buffer.nvmap_id)))?;
        if !handle.allocated {
            return Err(Error::Gpu(format!(
                "present: nvmap object {} has no memory yet",
                buffer.nvmap_id
            )));
        }
        let base = handle.cpu_addr.wrapping_add(buffer.offset);
        if self.trace {
            eprintln!(
                "[gpu] present nvmap={} offset={:#x} -> cpu {:#x} {}x{}",
                buffer.nvmap_id, buffer.offset, base, buffer.width, buffer.height
            );
        }
        let format = display_color_format(buffer.color_format)?;
        let layout = match buffer.layout {
            NV_LAYOUT_PITCH => Layout::Pitch { pitch: buffer.pitch },
            NV_LAYOUT_BLOCK_LINEAR => Layout::BlockLinear {
                block_height_gobs: 1 << buffer.block_height_log2,
            },
            other => {
                return Err(Error::Gpu(format!("present: unsupported NvLayout {}", other)))
            }
        };
        let bpp = format.bytes_per_pixel;
        let width_bytes = match layout {
            Layout::Pitch { pitch } => pitch,
            Layout::BlockLinear { .. } => buffer.width * bpp,
        };

        // Asked once, not once per pixel: the answer is the same for every
        // one of the 921,600 in a 720p frame.
        let srgb = format.is_srgb();
        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;

        let mut pixels = Vec::with_capacity((buffer.width * buffer.height) as usize);
        for y in 0..buffer.height {
            // Swizzled once per contiguous run rather than once per pixel:
            // `Layout::run_at` says how far the addresses stay linear, which
            // at 32 bits a pixel is four of them.
            let mut x = 0;
            while x < buffer.width {
                let (offset, run) = layout.run_at(x * bpp, y, width_bytes);
                let addr = base.wrapping_add(offset);
                let count = (run / bpp).clamp(1, buffer.width - x);
                for i in 0..count {
                    let raw = mem.read_le(addr.wrapping_add(i * bpp), bpp)?;
                    // The common surface is already the word the canvas
                    // wants; only a format whose decode is real work goes
                    // through linear light and straight back again.
                    if let Some(word) = format.host_word(raw as u32) {
                        pixels.push(word);
                        continue;
                    }
                    let mut rgba = format.decode(raw)?;
                    if srgb {
                        for c in rgba.iter_mut().take(3) {
                            *c = surface::linear_to_srgb(*c);
                        }
                    }
                    pixels.push(
                        to8(rgba[0])
                            | (to8(rgba[1]) << 8)
                            | (to8(rgba[2]) << 16)
                            | (to8(rgba[3]) << 24),
                    );
                }
                x += count;
            }
        }
        self.framebuffer = Framebuffer { width: buffer.width, height: buffer.height, pixels };
        self.frames += 1;
        Ok(())
    }
}

/// Map an `NvColorFormat` onto the equivalent Maxwell colour surface format,
/// so the same decoder serves both the engines and scan-out.
///
/// `NvColorFormat` names its channels most-significant first and ends with the
/// bits-per-pixel byte, so `A8B8G8R8` stores red in the lowest byte — the same
/// order Maxwell calls `RGBA8`.
fn display_color_format(nv_format: u64) -> Result<ColorFormat> {
    let raw = match nv_format {
        0x01_0053_2120 => 0xD5, // A8B8G8R8    -> RGBA8Unorm
        0x02_0053_2120 => 0xD6, // A8B8G8R8_sRGB
        0x01_0A53_2120 => 0xF9, // X8B8G8R8    -> RGBX8Unorm
        0x02_0A53_2120 => 0xFA, // X8B8G8R8_sRGB
        0x01_060A_2120 => 0xCF, // B8G8R8A8    -> BGRA8Unorm
        0x01_00D1_2120 => 0xCF, // A8R8G8B8
        0x01_0A0A_2120 => 0xE6, // B8G8R8X8    -> BGRX8Unorm
        0x01_0688_2120 => 0xD5, // R8G8B8A8
        0x01_0053_2020 => 0xD1, // A2B10G10R10 -> RGB10A2Unorm
        0x01_060A_2320 => 0xDF, // B10G10R10A2 -> BGR10A2Unorm
        0x01_0A88_1210 => 0xE8, // R5G6B5      -> 16-bit 565, red in the high bits
        0x01_0053_1410 => 0xE9, // A1B5G5R5    -> BGR5A1Unorm
        0x01_0A88_1810 => 0xF8, // R5G5B5X1    -> BGR5X1Unorm
        other => {
            return Err(Error::Gpu(format!(
                "present: unsupported NvColorFormat {:#x}",
                other
            )))
        }
    };
    ColorFormat::from_raw(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::vmm::SMALL_PAGE_SIZE;

    #[test]
    fn channels_get_distinct_syncpoints() {
        let mut gpu = Gpu::new();
        let a = gpu.create_channel().unwrap();
        let b = gpu.create_channel().unwrap();
        let syncpt_a = gpu.channel_mut(a).unwrap().syncpt;
        let syncpt_b = gpu.channel_mut(b).unwrap().syncpt;
        assert_ne!(syncpt_a, syncpt_b);
    }

    #[test]
    fn submit_needs_an_address_space() {
        let mut gpu = Gpu::new();
        let mut mem = Memory::new();
        let id = gpu.create_channel().unwrap();
        assert!(gpu.submit(id, &mut mem, &[], 1).is_err());
    }

    #[test]
    fn submit_advances_the_channel_fence() {
        let mut gpu = Gpu::new();
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();
        let as_id = gpu.create_address_space();
        gpu.address_space_mut(as_id)
            .unwrap()
            .map(0x3000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();
        let id = gpu.create_channel().unwrap();
        gpu.channel_mut(id).unwrap().as_id = Some(as_id);

        let fence = gpu.submit(id, &mut mem, &[], 1).unwrap();
        assert_eq!(fence.value, 1);
        assert!(gpu.host1x.is_expired(fence.id, fence.value).unwrap());
        let second = gpu.submit(id, &mut mem, &[], 1).unwrap();
        assert_eq!(second.value, 2);
    }

    #[test]
    fn present_deswizzles_a_block_linear_buffer() {
        let mut gpu = Gpu::new();
        let mut mem = Memory::new();
        mem.map_zero(0x4000_0000, 0x1000).unwrap();
        let handle = gpu.nvmap.create(0x1000);
        gpu.nvmap.alloc(handle, 0, 1, 0x1000, 0, 0x4000_0000).unwrap();
        let id = gpu.nvmap.get(handle).unwrap().id;

        // A 16x8 RGBA8 image is exactly one GOB; write a distinct value at the
        // swizzled position of pixel (1, 0) and (0, 1).
        let at = |x: u32, y: u32| 0x4000_0000 + surface::gob_offset(x * 4, y);
        mem.write_u32(at(1, 0), 0xFF00_0000).unwrap();
        mem.write_u32(at(0, 1), 0x0000_00FF).unwrap();

        gpu.present(
            &mem,
            &DisplayBuffer {
                nvmap_id: id,
                offset: 0,
                width: 16,
                height: 8,
                pitch: 64,
                layout: NV_LAYOUT_BLOCK_LINEAR,
                block_height_log2: 0,
                color_format: 0x0100_5321_20,
            },
        )
        .unwrap();

        assert_eq!(gpu.framebuffer.width, 16);
        assert_eq!(gpu.framebuffer.height, 8);
        assert_eq!(gpu.framebuffer.pixels[1], 0xFF00_0000);
        assert_eq!(gpu.framebuffer.pixels[16], 0x0000_00FF);
        assert_eq!(gpu.frames, 1);
    }

    #[test]
    fn present_of_an_unknown_buffer_is_reported() {
        let mut gpu = Gpu::new();
        let mem = Memory::new();
        let err = gpu.present(
            &mem,
            &DisplayBuffer {
                nvmap_id: 99,
                offset: 0,
                width: 4,
                height: 4,
                pitch: 16,
                layout: NV_LAYOUT_PITCH,
                block_height_log2: 0,
                color_format: 0x0100_5321_20,
            },
        );
        assert!(err.is_err());
    }
}
