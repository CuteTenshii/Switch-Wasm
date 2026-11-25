//! The page's canvas as the device's screen: the surface being scanned out is drawn onto it
//! straight from the device, and a frame composited on the CPU is uploaded to it.

use crate::builtin::SCAN_OUT_WGSL;
use crate::Gpu;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::Format;
use switch_core::gpu::renderer::ScanOut;
use switch_core::gpu::Framebuffer;

#[derive(Debug)]
pub(crate) struct Screen {
    surface: wgpu::Surface<'static>,
    /// The canvas's format; its sRGB twin is a view format, for sources that are sRGB.
    format: wgpu::TextureFormat,
    alpha: wgpu::CompositeAlphaMode,
    /// What the canvas is configured at.
    size: (u32, u32),
    /// A CPU frame's texels, kept while its size holds.
    pixels: Option<(wgpu::Texture, (u32, u32))>,
}

impl Gpu {
    /// Draw onto `surface`, a canvas, instead of leaving frames to the page. A canvas whose
    /// formats are not the plain 8-bit ones is left alone.
    pub fn set_screen(&mut self, surface: wgpu::Surface<'static>) {
        let caps = surface.get_capabilities(&self.adapter);
        let Some(format) = caps.formats.iter().copied().find(|f| {
            matches!(
                f,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            )
        }) else {
            return;
        };
        let alpha = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|a| *a == wgpu::CompositeAlphaMode::Opaque)
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);
        self.screen = Some(Screen {
            surface,
            format,
            alpha,
            size: (0, 0),
            pixels: None,
        });
    }

    /// The held surface the display reads, drawn onto the screen. `false`: the device does not
    /// hold it as the display reads it.
    pub(super) fn scan_out_held(&mut self, ctx: &ExecCtx, shot: &ScanOut) -> bool {
        // The display shows the bytes as they are, whether or not the surface calls them sRGB.
        let order = |format: Format| match format {
            Format::Rgba8Unorm | Format::Rgba8UnormSrgb => Some(false),
            Format::Bgra8Unorm | Format::Bgra8UnormSrgb => Some(true),
            _ => None,
        };
        let Some(bgra) = order(shot.format) else {
            return false;
        };
        let Some(held) = self.space.held.values().find(|held| {
            order(held.target.format) == Some(bgra)
                && held.companion.is_none()
                && held.other.is_none()
                && held.target.layout == shot.layout
                && held.target.width == shot.width
                && held.target.height == shot.height
                && ctx.vmm.translate(held.target.addr).map(|(at, _)| at) == Some(shot.addr)
        }) else {
            return false;
        };
        let srgb = matches!(
            held.target.format,
            Format::Rgba8UnormSrgb | Format::Bgra8UnormSrgb
        );
        // A guest write since the last draw makes the device's copy the stale one.
        if self.space.aliased.contains(&held.target.addr) {
            return false;
        }
        let texture = held.texture.clone();
        self.draw_screen(&texture, srgb, shot.crop, (shot.flip_h, shot.flip_v))
    }

    /// A CPU frame, uploaded and drawn onto the screen.
    pub(super) fn show_frame(&mut self, frame: &Framebuffer) -> bool {
        let (width, height) = (frame.width, frame.height);
        if self.screen.is_none()
            || self.give_up()
            || width == 0
            || height == 0
            || frame.pixels.len() < (width * height) as usize
        {
            return false;
        }
        let kept = self
            .screen
            .as_ref()
            .and_then(|s| s.pixels.as_ref())
            .filter(|(_, size)| *size == (width, height))
            .map(|(texture, _)| texture.clone());
        let texture = kept.unwrap_or_else(|| {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            if let Some(screen) = self.screen.as_mut() {
                if let Some((old, _)) = screen.pixels.replace((texture.clone(), (width, height))) {
                    old.destroy();
                }
            }
            texture
        });
        let bytes: Vec<u8> = frame.pixels[..(width * height) as usize]
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.draw_screen(&texture, false, (0, 0, width, height), (false, false))
    }

    /// Draw the window `crop` of `source` onto the canvas, one texel to a pixel, flipped as asked.
    fn draw_screen(
        &mut self,
        source: &wgpu::Texture,
        srgb: bool,
        crop: (u32, u32, u32, u32),
        flip: (bool, bool),
    ) -> bool {
        let (x, y, width, height) = crop;
        if width == 0 || height == 0 {
            return false;
        }
        let Some(mut screen) = self.screen.take() else {
            return false;
        };
        let drawn = self.draw_on(&mut screen, source, srgb, (x, y, width, height), flip);
        self.screen = Some(screen);
        drawn
    }

    fn draw_on(
        &mut self,
        screen: &mut Screen,
        source: &wgpu::Texture,
        srgb: bool,
        (x, y, width, height): (u32, u32, u32, u32),
        (flip_h, flip_v): (bool, bool),
    ) -> bool {
        if screen.size != (width, height) {
            screen.surface.configure(
                &self.device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: screen.format,
                    width,
                    height,
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: screen.alpha,
                    view_formats: vec![screen.format.add_srgb_suffix()],
                    color_space: wgpu::SurfaceColorSpace::Auto,
                },
            );
            screen.size = (width, height);
        }
        let frame = match screen.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            _ => {
                // The next draw configures again.
                screen.size = (0, 0);
                return false;
            }
        };
        let target = if srgb {
            screen.format.add_srgb_suffix()
        } else {
            screen.format
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(target),
            ..Default::default()
        });
        self.scan_out_pass(
            source,
            (x, y, width, height),
            (flip_h, flip_v),
            &view,
            target,
        );
        self.queue.present(frame);
        true
    }

    /// The window `crop` of `source`, flipped as asked, drawn one texel to a pixel onto `target`.
    pub(super) fn scan_out_pass(
        &mut self,
        source: &wgpu::Texture,
        (x, y, width, height): (u32, u32, u32, u32),
        (flip_h, flip_v): (bool, bool),
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        let pipeline = match self.scan_pipelines.get(&format) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let (_, module) = self.module("scan-out", SCAN_OUT_WGSL);
                let pipeline =
                    self.device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some("scan-out"),
                            layout: None,
                            vertex: wgpu::VertexState {
                                module: &module,
                                entry_point: Some("vs_main"),
                                compilation_options: Default::default(),
                                buffers: &[],
                            },
                            primitive: wgpu::PrimitiveState::default(),
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState::default(),
                            fragment: Some(wgpu::FragmentState {
                                module: &module,
                                entry_point: Some("fs_main"),
                                compilation_options: Default::default(),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format,
                                    blend: None,
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                            }),
                            multiview_mask: None,
                            cache: None,
                        });
                self.scan_pipelines.insert(format, pipeline.clone());
                pipeline
            }
        };
        let mut parameters = Vec::with_capacity(24);
        for value in [x, y, width, height, u32::from(flip_h), u32::from(flip_v)] {
            parameters.extend_from_slice(&value.to_le_bytes());
        }
        let parameters = self.buffer("scan-out", &parameters, wgpu::BufferUsages::UNIFORM);
        let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-out"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scan-out"),
            });
        {
            let timed = self.time_pass(|| "scan-out".into());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scan-out"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self.render_timestamps(timed),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.submit_with_batch(encoder);
    }
}
