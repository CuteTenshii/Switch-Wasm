//! GM20B (Tegra X1 Maxwell) GPU model.
//!
//! [`nvmap`] is the memory-object table, [`vmm`] the GMMU, [`syncpt`] host1x, [`channel`] the
//! command processor with the [`engine`] classes behind it, and [`nvdrv`] the driver the guest
//! talks to. Everything the GPU touches lives in the same [`Memory`] the CPU runs from.

pub mod activity;
pub mod bcn;
pub mod channel;
pub mod compute;
pub mod engine;
pub mod exec;
pub mod macro_engine;
pub mod multimedia;
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
use std::collections::HashMap;
use surface::{ColorFormat, Layout};
use syncpt::{Host1x, NvFence};
use vmm::AddressSpace;

/// An image ready for display, as `0xAABBGGRR` pixels.
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

/// A surface handed to the display, from an `NvGraphicBuffer` plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayBuffer {
    /// nvmap object id, not a handle.
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
    /// Low byte of `NvColorFormat` is bits-per-pixel; the whole value selects the channel order.
    pub color_format: u64,
    /// `NATIVE_WINDOW_TRANSFORM_*` bits: how the image is stored versus shown.
    pub transform: u32,
    /// Which part of the surface is the image.
    pub crop: Crop,
}

/// The `Rect` a producer queues: the window of the surface that is the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crop {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl Crop {
    pub const ALL: Crop = Crop {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };

    /// Android calls a rectangle with no area empty.
    pub fn is_empty(&self) -> bool {
        self.right <= self.left || self.bottom <= self.top
    }

    /// `(x, y, width, height)` against the surface, clamped since the rectangle is guest data.
    pub fn window(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        if self.is_empty() {
            return (0, 0, width, height);
        }
        let x = (self.left.max(0) as u32).min(width);
        let y = (self.top.max(0) as u32).min(height);
        let right = (self.right.max(0) as u32).min(width);
        let bottom = (self.bottom.max(0) as u32).min(height);
        // Off the surface entirely: show the whole thing.
        if right <= x || bottom <= y {
            return (0, 0, width, height);
        }
        (x, y, right - x, bottom - y)
    }
}

pub const TRANSFORM_FLIP_H: u32 = 0x01;
pub const TRANSFORM_FLIP_V: u32 = 0x02;
/// Not applied: it would change the frame's shape.
pub const TRANSFORM_ROT_90: u32 = 0x04;

pub const NV_LAYOUT_PITCH: u32 = 1;
pub const NV_LAYOUT_BLOCK_LINEAR: u32 = 3;

#[derive(Debug)]
pub struct Gpu {
    activity: activity::GpuActivity,
    pub nvmap: NvMap,
    pub host1x: Host1x,
    pub address_spaces: HashMap<u32, AddressSpace>,
    pub channels: HashMap<u32, Channel>,
    /// The session's one backend, lent to whichever channel is executing.
    renderer: Box<dyn renderer::Renderer>,
    /// The channel the backend was last lent to, whose address space a flush uses.
    last_channel: Option<u32>,
    pub stats: GpuStats,
    pub framebuffer: Framebuffer,
    /// Reused between frames to avoid a per-frame allocation.
    scan_out: Vec<u8>,
    pub frames: u64,
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
            activity: activity::GpuActivity::default(),
            nvmap: NvMap::new(),
            host1x: Host1x::new(),
            address_spaces: HashMap::new(),
            channels: HashMap::new(),
            renderer: Box::new(renderer::Software),
            last_channel: None,
            stats: GpuStats::default(),
            framebuffer: Framebuffer::default(),
            scan_out: Vec::new(),
            frames: 0,
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

    /// Runs the GPFIFO submission to completion, so the returned fence is already expired.
    pub fn submit(
        &mut self,
        channel_id: u32,
        mem: &mut Memory,
        entries: &[u64],
        increments: u32,
    ) -> Result<NvFence> {
        let syncpt = {
            let chan = self.channels.get(&channel_id).ok_or_else(|| {
                Error::Gpu(format!("gpu: submit on unknown channel {}", channel_id))
            })?;
            chan.syncpt
        };
        let value = self.host1x.incr_max(syncpt, increments.max(1))?;
        let fence = NvFence { id: syncpt, value };

        let chan = self
            .channels
            .get_mut(&channel_id)
            .expect("channel checked above");
        let as_id = chan.as_id.ok_or_else(|| {
            Error::Gpu(format!("gpu: channel {} has no address space", channel_id))
        })?;
        let vmm = self.address_spaces.get(&as_id).ok_or_else(|| {
            Error::Gpu(format!(
                "gpu: channel {} bound to missing address space {}",
                channel_id, as_id
            ))
        })?;
        let mut ctx = ExecCtx {
            mem,
            vmm,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: crate::trace::enabled(crate::trace::Trace::Gpu),
        };
        // Lent for the submission and taken back even on a fault.
        chan.three_d.swap_renderer(&mut self.renderer);
        let submitted = chan.submit(entries, fence, &mut ctx);
        chan.three_d.swap_renderer(&mut self.renderer);
        self.last_channel = Some(channel_id);
        submitted?;
        Ok(fence)
    }

    /// Everything drawn, cleared, copied, presented and refused since the last call.
    pub fn take_activity(&mut self) -> activity::GpuActivity {
        let mut all = std::mem::take(&mut self.activity);
        for channel in self.channels.values_mut() {
            all.absorb(&mut channel.three_d.activity);
            all.absorb(&mut channel.three_d.inline.activity);
            all.absorb(&mut channel.two_d.activity);
            all.absorb(&mut channel.copy.activity);
            all.absorb(&mut channel.compute.activity);
            all.absorb(&mut channel.compute.inline.activity);
        }
        all
    }

    /// Replaces the default [`renderer::Software`].
    pub fn set_renderer(&mut self, renderer: Box<dyn renderer::Renderer>) {
        self.renderer = renderer;
    }

    pub fn renderer_report(&self) -> String {
        self.renderer.report_json()
    }

    pub fn renderer_lost(&self) -> bool {
        self.renderer.lost()
    }

    /// The channel that last submitted, or any channel still holding an address space.
    fn flush_channel(&self) -> Option<u32> {
        let usable = |id: u32| {
            self.channels.get(&id).is_some_and(|channel| {
                channel
                    .as_id
                    .is_some_and(|as_id| self.address_spaces.contains_key(&as_id))
            })
        };
        if self.last_channel.is_some_and(usable) {
            return self.last_channel;
        }
        self.channels.keys().copied().find(|&id| usable(id))
    }

    /// Writes back anything the backend holds, so the next reader of a render target sees it.
    pub fn flush_renderers(&mut self, mem: &mut Memory) -> Result<renderer::Flush> {
        let Some(channel_id) = self.flush_channel() else {
            return Ok(renderer::Flush::Done);
        };
        let channel = self
            .channels
            .get_mut(&channel_id)
            .expect("flush_channel returns a channel that exists");
        let as_id = channel
            .as_id
            .expect("flush_channel checks the address space");
        let vmm = self
            .address_spaces
            .get(&as_id)
            .expect("flush_channel checks the address space");
        channel.three_d.swap_renderer(&mut self.renderer);
        let mut ctx = ExecCtx {
            mem,
            vmm,
            host1x: &mut self.host1x,
            stats: &mut self.stats,
            trace: crate::trace::enabled(crate::trace::Trace::Gpu),
        };
        let flushed = channel.three_d.flush_renderer(&mut ctx);
        channel.three_d.swap_renderer(&mut self.renderer);
        flushed
    }

    /// Scan-out: convert a display surface to the RGBA8888 [`Framebuffer`].
    pub fn present(&mut self, mem: &Memory, buffer: &DisplayBuffer) -> Result<()> {
        let handle = self.nvmap.by_id(buffer.nvmap_id).ok_or_else(|| {
            Error::Gpu(format!(
                "present: no nvmap object with id {}",
                buffer.nvmap_id
            ))
        })?;
        if !handle.allocated {
            return Err(Error::Gpu(format!(
                "present: nvmap object {} has no memory yet",
                buffer.nvmap_id
            )));
        }
        let base = handle.cpu_addr.wrapping_add(buffer.offset);
        self.activity.note(
            activity::Kind::Present,
            u64::from(base),
            u64::from(buffer.nvmap_id),
            1,
            false,
            || {
                format!(
                    "nvmap {} at cpu {base:#x} {}x{} fmt {:#x} layout {} pitch {}",
                    buffer.nvmap_id,
                    buffer.width,
                    buffer.height,
                    buffer.color_format,
                    buffer.layout,
                    buffer.pitch
                )
            },
        );
        if crate::trace::enabled(crate::trace::Trace::Gpu)
            || crate::trace::enabled(crate::trace::Trace::Present)
        {
            crate::traceln!(
                "[gpu] present frame={} nvmap={} offset={:#x} -> cpu {:#x} {}x{} fmt={:#x} \
                 layout={} pitch={} crop={:?} transform={:#x}",
                self.frames,
                buffer.nvmap_id,
                buffer.offset,
                base,
                buffer.width,
                buffer.height,
                buffer.color_format,
                buffer.layout,
                buffer.pitch,
                buffer.crop,
                buffer.transform
            );
        }
        let format = display_color_format(buffer.color_format)?;
        let layout = match buffer.layout {
            NV_LAYOUT_PITCH => Layout::Pitch {
                pitch: buffer.pitch,
            },
            NV_LAYOUT_BLOCK_LINEAR => Layout::BlockLinear {
                block_height_gobs: 1 << buffer.block_height_log2,
            },
            other => {
                return Err(Error::Gpu(format!(
                    "present: unsupported NvLayout {}",
                    other
                )))
            }
        };
        let bpp = format.bytes_per_pixel;
        let width_bytes = match layout {
            Layout::Pitch { pitch } => pitch,
            Layout::BlockLinear { .. } => buffer.width * bpp,
        };

        let srgb = format.is_srgb();
        let shuffle = format.host_shuffle();
        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;

        // The whole surface in one page-table walk, into a buffer reused between frames.
        let swizzled = layout.layer_stride(width_bytes, buffer.height) as usize;
        let mut raw_bytes = std::mem::take(&mut self.scan_out);
        raw_bytes.resize(swizzled, 0);
        let held = mem.read_into(base, &mut raw_bytes[..swizzled]).is_ok();

        let flip_v = buffer.transform & TRANSFORM_FLIP_V != 0;
        let flip_h = buffer.transform & TRANSFORM_FLIP_H != 0;
        if buffer.transform & TRANSFORM_ROT_90 != 0 {
            return Err(Error::Gpu(format!(
                "present: no rotation for queue transform {:#x}",
                buffer.transform
            )));
        }

        // Only the window the producer queued.
        let (crop_x, crop_y, out_width, out_height) =
            buffer.crop.window(buffer.width, buffer.height);

        let mut pixels = Vec::with_capacity((out_width * out_height) as usize);
        // The surface row each image row comes from.
        let surface_row = |row: u32| crop_y + if flip_v { out_height - 1 - row } else { row };

        // Fast path for 32-bit shuffle formats: row and column swizzle halves are computed once.
        let mut scanned = false;
        if let Some(shuffle) = shuffle.filter(|_| held && bpp == 4) {
            let columns: Vec<u32> = (0..out_width)
                .map(|x| layout.column_offset((crop_x + x) * bpp))
                .collect();
            let rows: Vec<u32> = (0..out_height)
                .map(|row| layout.row_offset(surface_row(row), width_bytes))
                .collect();
            let furthest = u64::from(rows.iter().copied().max().unwrap_or(0))
                + u64::from(columns.iter().copied().max().unwrap_or(0))
                + u64::from(bpp);
            if furthest <= swizzled as u64 && swizzled >= 4 {
                // `furthest` bounds every offset; the slice lets the compiler drop per-read checks.
                let scan = &raw_bytes[..swizzled];
                let last = scan.len() - 4;
                for &row_offset in &rows {
                    let row_start = pixels.len();
                    pixels.extend(columns.iter().map(|&column| {
                        let at = ((row_offset + column) as usize).min(last);
                        shuffle.apply(u32::from_le_bytes(
                            scan[at..at + 4].try_into().expect("four bytes"),
                        ))
                    }));
                    if flip_h {
                        pixels[row_start..].reverse();
                    }
                }
                scanned = true;
            }
        }

        let rows_left = if scanned { 0 } else { out_height };
        for row in 0..rows_left {
            let y = surface_row(row);
            let row_start = pixels.len();
            // Swizzled once per contiguous run rather than per pixel.
            let mut x = 0;
            while x < out_width {
                let (offset, run) = layout.run_at((crop_x + x) * bpp, y, width_bytes);
                let addr = base.wrapping_add(offset);
                let count = (run / bpp).clamp(1, out_width - x);
                let run_bytes = offset as usize..(offset + count * bpp) as usize;
                if let Some(shuffle) = shuffle.filter(|_| held && run_bytes.end <= swizzled) {
                    // A shuffle is always 8 bits per channel.
                    pixels.extend(
                        raw_bytes[run_bytes]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| shuffle.apply(u32::from_le_bytes(*b))),
                    );
                    x += count;
                    continue;
                }
                for i in 0..count {
                    let at = (offset + i * bpp) as usize;
                    let raw = match (held, at + bpp as usize <= swizzled) {
                        (true, true) if bpp == 4 => u128::from(u32::from_le_bytes(
                            raw_bytes[at..at + 4].try_into().expect("four bytes"),
                        )),
                        (true, true) if bpp == 2 => u128::from(u16::from_le_bytes(
                            raw_bytes[at..at + 2].try_into().expect("two bytes"),
                        )),
                        _ => mem.read_le(addr.wrapping_add(i * bpp), bpp)?,
                    };
                    // Shuffle formats are already the canvas word; others decode through linear light.
                    if let Some(shuffle) = shuffle {
                        pixels.push(shuffle.apply(raw as u32));
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
            if flip_h {
                pixels[row_start..].reverse();
            }
        }
        self.scan_out = raw_bytes;
        self.framebuffer = Framebuffer {
            width: out_width,
            height: out_height,
            pixels,
        };
        self.frames += 1;
        Ok(())
    }
}

/// `NvColorFormat` to the equivalent Maxwell colour surface format.
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

    /// A named backend that counts its flushes.
    #[derive(Debug)]
    struct Stub {
        flushes: std::rc::Rc<std::cell::Cell<u32>>,
    }

    impl renderer::Renderer for Stub {
        fn draw(
            &mut self,
            _engine: &crate::gpu::engine::threed::Engine3D,
            _ctx: &mut ExecCtx,
        ) -> Result<()> {
            Ok(())
        }

        fn clear_color(
            &mut self,
            _engine: &crate::gpu::engine::threed::Engine3D,
            _ctx: &mut ExecCtx,
            _target: u32,
            _layer: u32,
            _channels: [bool; 4],
        ) -> Result<()> {
            Ok(())
        }

        fn clear_depth_stencil(
            &mut self,
            _engine: &crate::gpu::engine::threed::Engine3D,
            _ctx: &mut ExecCtx,
            _depth: bool,
            _stencil: bool,
        ) -> Result<()> {
            Ok(())
        }

        fn flush(&mut self, _ctx: &mut ExecCtx) -> Result<renderer::Flush> {
            self.flushes.set(self.flushes.get() + 1);
            Ok(renderer::Flush::Done)
        }

        fn report_json(&self) -> String {
            "{\"backend\":\"stub\"}".to_string()
        }
    }

    /// The backend belongs to the `Gpu`, not a channel: titles open several and close them.
    #[test]
    fn the_backend_outlives_more_channels_and_a_close() {
        let mut gpu = Gpu::new();
        let first = gpu.create_channel().unwrap();
        gpu.set_renderer(Box::new(Stub {
            flushes: std::rc::Rc::new(std::cell::Cell::new(0)),
        }));
        for _ in 0..3 {
            gpu.create_channel().unwrap();
        }
        assert_eq!(
            gpu.renderer_report(),
            "{\"backend\":\"stub\"}",
            "three more channels do not displace the backend"
        );
        gpu.channels.remove(&first);
        assert_eq!(
            gpu.renderer_report(),
            "{\"backend\":\"stub\"}",
            "closing the channel it was installed on does not take it away"
        );
    }

    /// One pushbuffer command word, the encoding [`channel::Command`] decodes.
    fn header(mode: u32, arg: u32, subchannel: u32, method: u32) -> u32 {
        (method & 0x1FFF) | ((subchannel & 7) << 13) | ((arg & 0x1FFF) << 16) | (mode << 29)
    }

    /// A title drawing through its second channel still reaches the backend,
    /// and the backend is back on the `Gpu` afterwards.
    #[test]
    fn a_later_channel_reaches_the_backend() {
        let mut gpu = Gpu::new();
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();
        let as_id = gpu.create_address_space();
        let gpu_va = gpu
            .address_space_mut(as_id)
            .unwrap()
            .map(0x3000_0000, 0x1000, 1, 0, SMALL_PAGE_SIZE, 0, 0)
            .unwrap();

        // A copy flushes the 3D engine's surfaces first, so this submission must reach the backend.
        let words = [
            header(3, 1, 0, 0),
            crate::gpu::engine::CLASS_COPY,
            header(3, 1, 0, crate::gpu::engine::copy::LAUNCH_DMA),
            0,
        ];
        for (index, &word) in words.iter().enumerate() {
            mem.write_u32(0x3000_0000 + index as u32 * 4, word).unwrap();
        }

        let flushes = std::rc::Rc::new(std::cell::Cell::new(0));
        gpu.set_renderer(Box::new(Stub {
            flushes: flushes.clone(),
        }));
        let _first = gpu.create_channel().unwrap();
        let second = gpu.create_channel().unwrap();
        gpu.channel_mut(second).unwrap().as_id = Some(as_id);

        let entry = gpu_va | ((words.len() as u64) << 42);
        gpu.submit(second, &mut mem, &[entry], 1).unwrap();
        assert_eq!(
            flushes.get(),
            1,
            "the submission on the second channel reached the backend"
        );
        assert_eq!(
            gpu.renderer_report(),
            "{\"backend\":\"stub\"}",
            "the backend came back off the channel it was lent to"
        );
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
        gpu.nvmap
            .alloc(handle, 0, 1, 0x1000, 0, 0x4000_0000)
            .unwrap();
        let id = gpu.nvmap.get(handle).unwrap().id;

        // A 16x8 RGBA8 image is exactly one GOB.
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
                transform: 0,
                crop: Crop::ALL,
            },
        )
        .unwrap();

        assert_eq!(gpu.framebuffer.width, 16);
        assert_eq!(gpu.framebuffer.height, 8);
        assert_eq!(gpu.framebuffer.pixels[1], 0xFF00_0000);
        assert_eq!(gpu.framebuffer.pixels[16], 0x0000_00FF);
        assert_eq!(gpu.frames, 1);
    }

    /// `FLIP_V` is applied at scan-out.
    #[test]
    fn a_flipped_queue_transform_turns_the_frame_over() {
        let mut gpu = Gpu::new();
        let mut mem = Memory::new();
        mem.map_zero(0x4000_0000, 0x1000).unwrap();
        let handle = gpu.nvmap.create(0x1000);
        gpu.nvmap
            .alloc(handle, 0, 0, 0x1000, 0, 0x4000_0000)
            .unwrap();
        let id = gpu.nvmap.get(handle).unwrap().id;
        let at = |x: u32, y: u32| 0x4000_0000 + surface::gob_offset(x * 4, y);
        // A pixel in the top row and one in the bottom row of an 8-row image.
        mem.write_u32(at(0, 0), 0xFF00_0000).unwrap();
        mem.write_u32(at(0, 7), 0x0000_00FF).unwrap();
        let buffer = |transform| DisplayBuffer {
            nvmap_id: id,
            offset: 0,
            width: 16,
            height: 8,
            pitch: 64,
            layout: NV_LAYOUT_BLOCK_LINEAR,
            block_height_log2: 0,
            color_format: 0x0100_5321_20,
            transform,
            crop: Crop::ALL,
        };

        gpu.present(&mem, &buffer(0)).unwrap();
        assert_eq!(gpu.framebuffer.pixels[0], 0xFF00_0000);
        assert_eq!(gpu.framebuffer.pixels[7 * 16], 0x0000_00FF);

        gpu.present(&mem, &buffer(TRANSFORM_FLIP_V)).unwrap();
        assert_eq!(
            gpu.framebuffer.pixels[0], 0x0000_00FF,
            "the last row is shown first"
        );
        assert_eq!(gpu.framebuffer.pixels[7 * 16], 0xFF00_0000);

        // Left to right, about the same image.
        mem.write_u32(at(15, 0), 0x00FF_0000).unwrap();
        gpu.present(&mem, &buffer(TRANSFORM_FLIP_H)).unwrap();
        assert_eq!(gpu.framebuffer.pixels[15], 0xFF00_0000);
        assert_eq!(gpu.framebuffer.pixels[0], 0x00FF_0000);

        // A rotation is refused rather than shown the wrong shape.
        assert!(gpu.present(&mem, &buffer(TRANSFORM_ROT_90)).is_err());
    }

    /// Only the queued crop window is scanned out.
    #[test]
    fn a_queued_crop_is_the_frame() {
        let mut gpu = Gpu::new();
        let mut mem = Memory::new();
        mem.map_zero(0x4000_0000, 0x1000).unwrap();
        let handle = gpu.nvmap.create(0x1000);
        gpu.nvmap
            .alloc(handle, 0, 0, 0x1000, 0, 0x4000_0000)
            .unwrap();
        let id = gpu.nvmap.get(handle).unwrap().id;
        let at = |x: u32, y: u32| 0x4000_0000 + surface::gob_offset(x * 4, y);
        // The corners of the window (4, 2)..(12, 6), and one pixel outside it.
        mem.write_u32(at(4, 2), 0xFF00_0000).unwrap();
        mem.write_u32(at(11, 5), 0x0000_00FF).unwrap();
        mem.write_u32(at(0, 0), 0x00FF_0000).unwrap();
        let buffer = |crop, transform| DisplayBuffer {
            nvmap_id: id,
            offset: 0,
            width: 16,
            height: 8,
            pitch: 64,
            layout: NV_LAYOUT_BLOCK_LINEAR,
            block_height_log2: 0,
            color_format: 0x0100_5321_20,
            transform,
            crop,
        };
        let window = Crop {
            left: 4,
            top: 2,
            right: 12,
            bottom: 6,
        };

        gpu.present(&mem, &buffer(window, 0)).unwrap();
        assert_eq!((gpu.framebuffer.width, gpu.framebuffer.height), (8, 4));
        assert_eq!(gpu.framebuffer.pixels[0], 0xFF00_0000);
        assert_eq!(gpu.framebuffer.pixels[3 * 8 + 7], 0x0000_00FF);
        assert!(
            !gpu.framebuffer.pixels.contains(&0x00FF_0000),
            "a pixel outside the crop is not in the frame"
        );

        // The flip is about the window, not the surface.
        gpu.present(&mem, &buffer(window, TRANSFORM_FLIP_V))
            .unwrap();
        assert_eq!(gpu.framebuffer.pixels[3 * 8], 0xFF00_0000);
        assert_eq!(gpu.framebuffer.pixels[7], 0x0000_00FF);

        // Empty and off-surface rectangles both mean the whole surface.
        for crop in [
            Crop::ALL,
            Crop {
                left: 9,
                top: 0,
                right: 4,
                bottom: 8,
            },
            Crop {
                left: 40,
                top: 0,
                right: 60,
                bottom: 8,
            },
        ] {
            gpu.present(&mem, &buffer(crop, 0)).unwrap();
            assert_eq!((gpu.framebuffer.width, gpu.framebuffer.height), (16, 8));
        }
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
                transform: 0,
                crop: Crop::ALL,
            },
        );
        assert!(err.is_err());
    }
}
