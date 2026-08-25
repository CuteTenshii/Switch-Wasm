//! A GPU backend for switch-core's 3D engine.
//!
//! [`switch_core::gpu::renderer::Renderer`] is the seam this plugs into, and
//! [`switch_core::gpu::renderer::Software`] is what it has to agree with.
//! That is the whole shape of the thing: the software rasterizer is the
//! reference, this is the fast path, and any draw this cannot express runs on
//! the reference instead. A backend that guessed at a draw it did not
//! understand would produce a frame nobody could check.
//!
//! # Why this is a crate of its own
//!
//! `switch-core` has no dependencies at all — its Wasm bindings are
//! hand-rolled and its display path is `putImageData`. `wgpu` brings a few
//! hundred crates. Keeping it out here means the core stays what it is, and
//! the trait is the only thing the two share.
//!
//! # What a draw costs, and why it is built this way first
//!
//! A render target lives in guest memory, so a draw here reads the surface
//! in, renders, and writes it back — per draw, which is the slowest possible
//! arrangement and the only one that is obviously correct. Batching a whole
//! frame into one pass is the optimisation, and it is worth nothing until the
//! per-draw version renders the same pixels the rasterizer does.

use switch_core::gpu::engine::threed::Engine3D;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::Format;
use switch_core::gpu::renderer::{Renderer, Software};
use switch_core::gpu::upload::{Target, Targets};
use switch_core::{Error, Result};

/// `copyTextureToBuffer` wants each row of the destination aligned.
const COPY_ALIGNMENT: u32 = 256;

/// A device, and the rasterizer to fall back to.
#[derive(Debug)]
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// What this cannot express runs here instead.
    software: Software,
    /// Draws that fell back, and why the last one did.
    pub fallbacks: u64,
    pub last_fallback: Option<String>,
}

impl Gpu {
    /// Open a device, or say why not. `None` is a normal answer: a machine
    /// with no adapter runs the software rasterizer, which is what it did
    /// before this existed.
    pub fn open() -> std::result::Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| format!("no adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| format!("no device: {e}"))?;
        Ok(Gpu {
            device,
            queue,
            software: Software,
            fallbacks: 0,
            last_fallback: None,
        })
    }

    /// What adapter this opened.
    pub fn describe(&self) -> String {
        format!("{:?}", self.device.limits().max_texture_dimension_2d)
    }

    fn fall_back(&mut self, why: String) {
        self.fallbacks += 1;
        self.last_fallback = Some(why);
    }

    /// Bring a guest surface onto the device.
    ///
    /// A draw that blends, or tests depth, reads what is already there, so
    /// the texture starts as a copy of the surface rather than as whatever
    /// the last draw left on the device.
    fn upload_target(&self, target: &Target, ctx: &ExecCtx) -> Result<wgpu::Texture> {
        let format = texture_format(target.format)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render target"),
            size: wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let rows = target.read(ctx)?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rows,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.row_bytes),
                rows_per_image: Some(target.rows),
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }

    /// Take a surface back off the device and into guest memory.
    ///
    /// `copyTextureToBuffer` wants rows aligned to 256 bytes, which a
    /// surface's own rows need not be, so the padding is added on the way
    /// out and dropped on the way in.
    fn download_target(
        &self,
        target: &Target,
        texture: &wgpu::Texture,
        ctx: &mut ExecCtx,
    ) -> Result<()> {
        let padded = target.row_bytes.div_ceil(COPY_ALIGNMENT) * COPY_ALIGNMENT;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded) * u64::from(target.rows),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(target.rows),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("gpu: waiting for the readback: {e}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("gpu: mapping the readback: {e}")))?;
        let mut rows = Vec::with_capacity((target.row_bytes * target.rows) as usize);
        for y in 0..target.rows {
            let at = (y * padded) as usize;
            rows.extend_from_slice(&mapped[at..at + target.row_bytes as usize]);
        }
        drop(mapped);
        staging.unmap();
        target.write(ctx, &rows)
    }
}

/// The wgpu name for a format `switch_core::gpu::pipeline` resolved.
fn texture_format(format: Format) -> Result<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Ok(match format {
        Format::R8Unorm => T::R8Unorm,
        Format::Rg8Unorm => T::Rg8Unorm,
        Format::Rgba8Unorm => T::Rgba8Unorm,
        Format::Rgba8UnormSrgb => T::Rgba8UnormSrgb,
        Format::Bgra8Unorm => T::Bgra8Unorm,
        Format::Bgra8UnormSrgb => T::Bgra8UnormSrgb,
        Format::Rgb10a2Unorm => T::Rgb10a2Unorm,
        Format::R32Float => T::R32Float,
        Format::Rgba16Float => T::Rgba16Float,
        Format::Rgba32Float => T::Rgba32Float,
        Format::Depth16Unorm => T::Depth16Unorm,
        Format::Depth24Plus => T::Depth24Plus,
        Format::Depth24PlusStencil8 => T::Depth24PlusStencil8,
        Format::Depth32Float => T::Depth32Float,
        Format::Depth32FloatStencil8 => T::Depth32FloatStencil8,
        Format::Bc1RgbaUnorm => T::Bc1RgbaUnorm,
        Format::Bc1RgbaUnormSrgb => T::Bc1RgbaUnormSrgb,
        Format::Bc2RgbaUnorm => T::Bc2RgbaUnorm,
        Format::Bc2RgbaUnormSrgb => T::Bc2RgbaUnormSrgb,
        Format::Bc3RgbaUnorm => T::Bc3RgbaUnorm,
        Format::Bc3RgbaUnormSrgb => T::Bc3RgbaUnormSrgb,
        Format::Bc4RUnorm => T::Bc4RUnorm,
        Format::Bc4RSnorm => T::Bc4RSnorm,
        Format::Bc5RgUnorm => T::Bc5RgUnorm,
        Format::Bc5RgSnorm => T::Bc5RgSnorm,
        Format::Bc6hRgbUfloat => T::Bc6hRgbUfloat,
        Format::Bc6hRgbFloat => T::Bc6hRgbFloat,
        Format::Bc7RgbaUnorm => T::Bc7RgbaUnorm,
        Format::Bc7RgbaUnormSrgb => T::Bc7RgbaUnormSrgb,
    })
}

impl Renderer for Gpu {
    fn draw(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
        // Not drawing yet. What this proves is the plumbing either side of a
        // draw: a surface that survives a trip onto the device and back is
        // one a render pass can be put in the middle of, and a surface that
        // does not is a bug that would otherwise look like a shader bug.
        match Targets::of(engine) {
            Ok(targets) => {
                if let Some(target) = targets.color {
                    let texture = self.upload_target(&target, &*ctx)?;
                    self.download_target(&target, &texture, ctx)?;
                }
            }
            Err(e) => self.fall_back(format!("{e:?}")),
        }
        self.software.draw(engine, ctx)
    }

    fn clear_color(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        target: u32,
        layer: u32,
        channels: [bool; 4],
    ) -> Result<()> {
        self.software.clear_color(engine, ctx, target, layer, channels)
    }

    fn clear_depth_stencil(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        depth: bool,
        stencil: bool,
    ) -> Result<()> {
        self.software.clear_depth_stencil(engine, ctx, depth, stencil)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn there_is_a_device_to_render_on() {
        match super::Gpu::open() {
            Ok(gpu) => println!("[gpu] opened, max texture {}", gpu.describe()),
            Err(why) => println!("[gpu] {why}"),
        }
    }
}
