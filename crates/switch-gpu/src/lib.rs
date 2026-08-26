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
//! # A draw never blocks
//!
//! A render target lives in guest memory, so the obvious arrangement is to
//! read the surface in, render, and write it back per draw. That is what this
//! did first, and it was byte-identical to the rasterizer, and it cannot work
//! in a browser: reading a texture back means waiting on a promise, and a
//! blocking wait there is not slow but deadlocked, because the event loop
//! that would resolve it cannot run.
//!
//! So a surface stays on the device once it is there, across every draw that
//! targets it, and goes back to guest memory only at
//! [`Renderer::flush`] — which the engine calls before `present`, the one
//! reader that always matters. Draws encode and return. The waiting happens
//! at the frame boundary, which in a browser is exactly where a worker is
//! free to await.
//!
//! That is not only a portability change. It is the same change that turns
//! eighty-eight round trips a frame into one.

/// What to ask a device for: the compressed texture families this adapter
/// actually has, and nothing else.
///
/// A Switch title's textures are block-compressed, and WebGPU hands those out
/// only on request — `DeviceDescriptor::default()` asks for none of them, so
/// the first BC1 texture threw. Masking against the adapter keeps the request
/// itself from failing on hardware that lacks a family, and
/// [`device_texture_format`] turns whatever is still missing into a fallback
/// rather than a crash.
pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let wanted = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2;
    wgpu::DeviceDescriptor {
        required_features: adapter.features() & wanted,
        ..Default::default()
    }
}

/// The `wgpu` this was built against, so a caller that has to name a device
/// type does not have to guess at a matching version.
pub use wgpu;

use switch_core::gpu::engine::threed::{Engine3D, ShaderStage};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::pipeline::{self as state, Format, Pipeline};
use switch_core::gpu::renderer::{Renderer, Software};
use switch_core::gpu::shader::compiled::Compiled;
use switch_core::gpu::shader::wgsl::{self, Layout, Stage, Translation};
use switch_core::gpu::upload::{Banks, Target, Targets, Uploads};
use switch_core::{Error, Result};

/// `copyTextureToBuffer` wants each row of the destination aligned.
const COPY_ALIGNMENT: u32 = 256;

/// Where a module's textures start binding; see
/// `switch_core::gpu::shader::wgsl`.
const TEXTURE_BINDING: u32 = 32;

/// A binding and what fills it, held until the bind group is built.
enum Resource {
    Buffer(u32, wgpu::Buffer),
    Texture(u32, wgpu::Texture, wgpu::TextureViewDimension),
    Sampler(u32, wgpu::Sampler),
}

fn topology(topology: state::Topology) -> wgpu::PrimitiveTopology {
    match topology {
        state::Topology::PointList => wgpu::PrimitiveTopology::PointList,
        state::Topology::LineList => wgpu::PrimitiveTopology::LineList,
        state::Topology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        state::Topology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        state::Topology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

/// The guest's per-channel colour write enables, as WebGPU spells them.
fn write_mask(mask: [bool; 4]) -> wgpu::ColorWrites {
    let channels = [
        wgpu::ColorWrites::RED,
        wgpu::ColorWrites::GREEN,
        wgpu::ColorWrites::BLUE,
        wgpu::ColorWrites::ALPHA,
    ];
    let mut writes = wgpu::ColorWrites::empty();
    for (enabled, channel) in mask.into_iter().zip(channels) {
        if enabled {
            writes |= channel;
        }
    }
    writes
}

fn vertex_format(format: state::VertexFormat) -> wgpu::VertexFormat {
    match format {
        state::VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        state::VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        state::VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        state::VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        state::VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
    }
}

fn blend_factor(factor: state::BlendFactor) -> wgpu::BlendFactor {
    match factor {
        state::BlendFactor::Zero => wgpu::BlendFactor::Zero,
        state::BlendFactor::One => wgpu::BlendFactor::One,
        state::BlendFactor::Src => wgpu::BlendFactor::Src,
        state::BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
        state::BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        state::BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        state::BlendFactor::Dst => wgpu::BlendFactor::Dst,
        state::BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
        state::BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        state::BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        state::BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        state::BlendFactor::Constant => wgpu::BlendFactor::Constant,
        state::BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
    }
}

fn blend_operation(operation: state::BlendOperation) -> wgpu::BlendOperation {
    match operation {
        state::BlendOperation::Add => wgpu::BlendOperation::Add,
        state::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        state::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        state::BlendOperation::Min => wgpu::BlendOperation::Min,
        state::BlendOperation::Max => wgpu::BlendOperation::Max,
    }
}

fn blend(blend: state::Blend) -> wgpu::BlendState {
    let component = |c: state::BlendComponent| wgpu::BlendComponent {
        src_factor: blend_factor(c.src_factor),
        dst_factor: blend_factor(c.dst_factor),
        operation: blend_operation(c.operation),
    };
    wgpu::BlendState { color: component(blend.color), alpha: component(blend.alpha) }
}

/// A readback that has been asked for and not yet copied out.
///
/// Kept as a type because asking and collecting are the two halves a browser
/// has to put an `await` between — see [`Gpu::write_back`], which today does
/// both with a wait in the middle.
#[derive(Debug)]
struct Pending {
    staging: wgpu::Buffer,
    target: Target,
    /// The row stride the copy used, which is the surface's rounded up to
    /// 256 bytes.
    padded: u32,
}

/// A render target held on the device, and where in guest memory it came
/// from.
#[derive(Debug)]
struct Held {
    texture: wgpu::Texture,
    target: Target,
    /// Whether anything has been drawn into it since it was uploaded. A
    /// surface nothing touched need not be written back.
    dirty: bool,
}

/// A device, and the rasterizer to fall back to.
#[derive(Debug)]
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Surfaces this is holding, by the guest address they came from.
    ///
    /// A frame's draws all target the same surface, so this is what makes a
    /// frame one upload and one readback instead of eighty-eight of each.
    held: std::collections::HashMap<u64, Held>,
    /// Surfaces the guest rebound out from under, still owing a write-back.
    /// Kept rather than written back where it happened, so that no draw ever
    /// waits on a device.
    evicted: Vec<Held>,
    /// Compiled shader modules, by a hash of the WGSL that produced them.
    ///
    /// Compiling a module is the whole cost of a draw: WGSL through naga to
    /// SPIR-V is about 59 ms, against 2 ms to translate the shader and half
    /// a millisecond to read every buffer the draw touches. The Home Menu
    /// has five distinct shader pairs and drew eighty-eight times a frame,
    /// so this is not an optimisation so much as not doing the same work
    /// eighty-eight times.
    ///
    /// Keyed by the source rather than by the shader's address, because the
    /// source is what the module *is*: two draws whose shaders live at the
    /// same address but were assembled with a different texture swizzle are
    /// two different modules, and a guest is free to overwrite a shader in
    /// place.
    modules: std::collections::HashMap<u64, wgpu::ShaderModule>,
    /// Set by the device when it rejects something. Read on the next draw,
    /// because asking sooner means waiting.
    failed: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// What this cannot express runs here instead.
    software: Software,
    /// Draws this rendered, and draws that fell back with why the last one
    /// did — the two numbers that say how much of a frame is really running
    /// here.
    pub drawn: u64,
    pub fallbacks: u64,
    pub last_fallback: Option<String>,
    /// Every distinct reason a draw fell back, in the order first seen.
    pub reasons: Vec<String>,
    /// Which draw of the current frame this is, counting from the clear that
    /// starts it. Only `GPU_ONLY` reads it.
    in_frame: u32,
    /// `GPU_TIMES=1` accumulates where a draw's time goes, and prints it
    /// when this is dropped. Native only: it reads a clock, and a browser's
    /// answer to that is another question entirely.
    times: Option<Times>,
    /// `GPU_ONLY=<i>` renders only the i-th draw of each frame here and
    /// leaves the rest to the rasterizer.
    ///
    /// Which is how you find the draw that renders differently. The
    /// difference between a frame and the reference is then exactly one
    /// draw's, and bisecting over the range costs a handful of runs rather
    /// than reading a shader.
    only: Option<u32>,
}

/// Where a draw's time goes, in microseconds, over a whole run.
#[derive(Debug, Default, Clone, Copy)]
struct Times {
    /// Decoding both shaders out of guest memory and translating them.
    translate: u128,
    /// Reading the vertices, indices, constants and textures.
    upload: u128,
    /// Generating the WGSL and handing it to the device.
    modules: u128,
    /// Building the pipeline and its bind groups.
    pipeline: u128,
    /// Encoding and submitting the pass.
    encode: u128,
    /// Handing surfaces back to guest memory.
    flush: u128,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        eprintln!("[gpu] {} draws rendered, {} fell back", self.drawn, self.fallbacks);
        if let Some(t) = self.times {
            let ms = |v: u128| v as f64 / 1000.0;
            eprintln!(
                "[gpu] translate {:.0}ms  upload {:.0}ms  modules {:.0}ms  \
                 pipeline {:.0}ms  encode {:.0}ms  flush {:.0}ms",
                ms(t.translate),
                ms(t.upload),
                ms(t.modules),
                ms(t.pipeline),
                ms(t.encode),
                ms(t.flush),
            );
        }
    }
}

/// Add `at.elapsed()` to `slot`, if timing is on.
macro_rules! timed {
    ($self:ident, $field:ident, $body:expr) => {{
        if $self.times.is_none() {
            $body
        } else {
            let at = std::time::Instant::now();
            let out = $body;
            if let Some(t) = $self.times.as_mut() {
                t.$field += at.elapsed().as_micros();
            }
            out
        }
    }};
}

impl Gpu {
    /// Take a device somebody else opened.
    ///
    /// The browser's entry point. Opening a device there is asynchronous —
    /// `requestAdapter` and `requestDevice` are promises — and nothing in
    /// this crate may wait on a promise, so the waiting happens outside and
    /// the result is handed in. [`Gpu::open`] is the native convenience that
    /// does it by blocking, which is fine on a thread that owns itself.
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue) -> Gpu {
        // Where a rejection lands, since nothing in a draw ever stops to ask.
        let failed: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let sink = failed.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = sink.lock() {
                slot.get_or_insert_with(|| e.to_string());
            }
        }));
        Gpu {
            device,
            queue,
            held: std::collections::HashMap::new(),
            evicted: Vec::new(),
            modules: std::collections::HashMap::new(),
            failed,
            software: Software,
            drawn: 0,
            fallbacks: 0,
            last_fallback: None,
            reasons: Vec::new(),
            in_frame: 0,
            times: switch_core::env_flag!("GPU_TIMES").then(Times::default),
            only: std::env::var("GPU_ONLY").ok().and_then(|v| v.parse().ok()),
        }
    }

    /// Open a device by blocking on it, which only a native thread may do.
    pub fn open() -> std::result::Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| format!("no adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&device_descriptor(&adapter)))
                .map_err(|e| format!("no device: {e}"))?;
        Ok(Gpu::with_device(device, queue))
    }

    /// What adapter this opened.
    pub fn describe(&self) -> String {
        format!("{:?}", self.device.limits().max_texture_dimension_2d)
    }

    fn fall_back(&mut self, why: String) {
        self.fallbacks += 1;
        // Each distinct reason once: a draw that falls back does it every
        // frame, and the interesting thing is the list rather than the count.
        if !self.reasons.contains(&why) {
            eprintln!("[gpu] falling back: {why}");
            self.reasons.push(why.clone());
        }
        self.last_fallback = Some(why);
    }

    /// The device texture for a surface, uploading it if this is the first
    /// draw to reach it.
    ///
    /// A draw that blends reads what is already there. The first time that
    /// is guest memory; afterwards it is whatever the previous draw left on
    /// the device, which is the same thing and already in the right place.
    fn hold(&mut self, target: &Target, ctx: &ExecCtx) -> Result<()> {
        match self.held.get(&target.addr) {
            // The same surface as last time. Anything about it having
            // changed — a different format or extent at the same address —
            // means the guest rebound it, and the old contents are not this
            // one's.
            Some(held) if held.target == *target => return Ok(()),
            // A different surface at the same address: the guest rebound
            // it. The old one still has to go back, but not here — a draw
            // that read a texture back is a draw that blocks.
            Some(_) => {
                if let Some(held) = self.held.remove(&target.addr) {
                    self.evicted.push(held);
                }
            }
            None => {}
        }
        let texture = self.upload_target(target, ctx)?;
        self.held.insert(target.addr, Held { texture, target: *target, dirty: false });
        Ok(())
    }

    /// Write one held surface back into guest memory and stop holding it.
    fn flush_one(&mut self, addr: u64, ctx: &mut ExecCtx) -> Result<()> {
        let Some(held) = self.held.remove(&addr) else { return Ok(()) };
        self.write_back(&held, ctx)
    }

    /// Ask for a surface back, wait for it, and put it in guest memory.
    ///
    /// The wait is why this cannot run in a browser yet, and the two halves
    /// are kept apart — [`Gpu::start_read_back`] and [`Gpu::land`] — because
    /// that is where the seam goes when it can. What does *not* work is
    /// simply deferring the landing to the next flush: the Home Menu
    /// double-buffers, so the surface `present` reads is always the one whose
    /// readback was just asked for and never one that has arrived. Tried, and
    /// it presented a black frame every time.
    fn write_back(&mut self, held: &Held, ctx: &mut ExecCtx) -> Result<()> {
        // A surface nothing drew into is already what guest memory says.
        if !held.dirty {
            return Ok(());
        }
        let pending = self.start_read_back(&held.target, &held.texture);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("waiting for the readback: {e}")))?;
        self.land(&pending, ctx)
    }

    /// Bring a guest surface onto the device.
    fn upload_target(&self, target: &Target, ctx: &ExecCtx) -> Result<wgpu::Texture> {
        let format = device_texture_format(&self.device, target.format)?;
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

    /// Take a surface off the device.
    ///
    /// `copyTextureToBuffer` wants rows aligned to 256 bytes, which a
    /// surface's own rows need not be, so the padding is added on the way
    /// out and dropped on the way in.
    ///
    /// This is the one place that waits, and the reason [`Renderer::flush`]
    /// exists to be the only caller: on a browser the wait is a deadlock
    /// anywhere the event loop is not free to run.
    fn start_read_back(&self, target: &Target, texture: &wgpu::Texture) -> Pending {
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

        // Asked for, not waited on. The map completes when the device is next
        // polled — which on a browser is when the event loop next runs, and
        // there is no way to make that happen from inside a blocking call.
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        Pending { staging, target: *target, padded }
    }

    /// Copy a finished readback into guest memory, dropping the row padding
    /// `copyTextureToBuffer` insisted on.
    fn land(&self, pending: &Pending, ctx: &mut ExecCtx) -> Result<()> {
        let slice = pending.staging.slice(..);
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapping the readback: {e}")))?;
        let target = &pending.target;
        let mut rows = Vec::with_capacity((target.row_bytes * target.rows) as usize);
        for y in 0..target.rows {
            let at = (y * pending.padded) as usize;
            rows.extend_from_slice(&mapped[at..at + target.row_bytes as usize]);
        }
        drop(mapped);
        pending.staging.unmap();
        target.write(ctx, &rows)
    }
}

impl Gpu {
    /// Build the pipeline and run the pass.
    ///
    /// Everything is created per draw: the modules, the pipeline, the
    /// buffers, the bind groups. That is the slow arrangement and the one
    /// worth having first, because a cache is only ever as right as the
    /// thing it caches.
    fn render(&mut self, p: &Prepared, ctx: &mut ExecCtx) -> std::result::Result<(), String> {
        let target_format =
            device_texture_format(&self.device, p.color.format).map_err(|e| format!("{e:?}"))?;
        if let Some(e) = self.device_error() {
            return Err(format!("the device rejected an earlier draw: {e}"));
        }
        let (vs_module, fs_module) = timed!(self, modules, {
            let vs_source = wgsl::module(&p.vs, Stage::Vertex, &p.vs_layout);
            let fs_source = wgsl::module(&p.fs, Stage::Fragment, &p.fs_layout);
            match (vs_source, fs_source) {
                (Ok(vs), Ok(fs)) => Ok((self.module("vertex", &vs), self.module("fragment", &fs))),
                (Err(e), _) | (_, Err(e)) => Err(format!("module: {e}")),
            }
        })?;

        // Vertex buffers, and the attributes the vertex shader actually
        // reads: a draw binds sixteen slots and a shader reads one.
        let mut bound: Vec<(wgpu::Buffer, Vec<wgpu::VertexAttribute>)> = Vec::new();
        for buffer in &p.state.vertex_buffers {
            let attributes: Vec<wgpu::VertexAttribute> = buffer
                .attributes
                .iter()
                .filter(|a| p.vs_layout.attributes.contains(&(a.location as usize)))
                .map(|a| wgpu::VertexAttribute {
                    format: vertex_format(a.format),
                    offset: u64::from(a.offset),
                    shader_location: a.location,
                })
                .collect();
            if attributes.is_empty() {
                continue;
            }
            if buffer.step == state::StepMode::Instance {
                // An instanced array is fetched at the absolute instance
                // index, and only this instance's element was uploaded.
                return Err("an instanced vertex array".into());
            }
            let upload = p
                .uploads
                .vertex
                .iter()
                .find(|v| v.array == buffer.index)
                .ok_or("a bound vertex array with no bytes")?;
            bound.push((
                self.buffer("vertex", &upload.bytes, wgpu::BufferUsages::VERTEX),
                attributes,
            ));
        }
        // Every location the shader declares has to be fed, or the pipeline
        // will not build.
        let fed: Vec<usize> = bound
            .iter()
            .flat_map(|(_, a)| a.iter().map(|a| a.shader_location as usize))
            .collect();
        if let Some(missing) = p.vs_layout.attributes.iter().find(|l| !fed.contains(l)) {
            return Err(format!("attribute {missing} is bound to nothing"));
        }
        let layouts: Vec<Option<wgpu::VertexBufferLayout>> = bound
            .iter()
            .zip(&p.state.vertex_buffers)
            .map(|((_, attributes), buffer)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: u64::from(buffer.stride),
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes,
                })
            })
            .collect();

        let (vs_group_layout, vs_group) =
            timed!(self, pipeline, self.bind_group(p, ShaderStage::VertexB, 0))?;
        let (fs_group_layout, fs_group) =
            timed!(self, pipeline, self.bind_group(p, ShaderStage::Fragment, 1))?;
        let pipeline_layout =
            self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("draw"),
                bind_group_layouts: &[Some(&vs_group_layout), Some(&fs_group_layout)],
                immediate_size: 0,
            });
        let descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: topology(p.state.topology),
                strip_index_format: None,
                front_face: match p.state.front_face {
                    state::FrontFace::Ccw => wgpu::FrontFace::Ccw,
                    state::FrontFace::Cw => wgpu::FrontFace::Cw,
                },
                cull_mode: match p.state.cull {
                    state::Cull::None => None,
                    state::Cull::Front => Some(wgpu::Face::Front),
                    state::Cull::Back => Some(wgpu::Face::Back),
                },
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: p.state.target.and_then(|t| t.blend).map(blend),
                    write_mask: p.state.target.map_or(wgpu::ColorWrites::ALL, |t| {
                        write_mask(t.write_mask)
                    }),
                })],
            }),
            multiview_mask: None,
            cache: None,
        };
        let pipeline = timed!(self, pipeline, self.device.create_render_pipeline(&descriptor));

        // Held across the frame: the first draw brings the surface onto the
        // device and every later one finds it already there.
        self.hold(&p.color, ctx).map_err(|e| format!("{e:?}"))?;
        let held = self.held.get_mut(&p.color.addr).ok_or("the surface was not held")?;
        held.dirty = true;
        let view = held.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let index = p.uploads.index.as_ref().map(|index| {
            (
                self.buffer("index", &index.bytes, wgpu::BufferUsages::INDEX),
                match index.format {
                    switch_core::gpu::upload::IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                    switch_core::gpu::upload::IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                },
                index.lowest,
            )
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("draw") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Loaded, never cleared: a clear is its own method,
                        // and this pass is one draw in the middle of a frame.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &vs_group, &[]);
            pass.set_bind_group(1, &fs_group, &[]);
            for (slot, (buffer, _)) in bound.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buffer.slice(..));
            }
            let viewport = &p.state.viewport;
            pass.set_viewport(
                viewport.x,
                viewport.y,
                viewport.width,
                viewport.height,
                viewport.min_depth.clamp(0.0, 1.0),
                viewport.max_depth.clamp(0.0, 1.0),
            );
            let scissor = p.state.scissor;
            pass.set_scissor_rect(
                scissor.x0,
                scissor.y0,
                scissor.x1.saturating_sub(scissor.x0),
                scissor.y1.saturating_sub(scissor.y0),
            );
            if let Some(constant) = p.state.target.and_then(|t| t.blend) {
                let _ = constant;
                let [r, g, b, a] = p.state.blend_constant;
                pass.set_blend_constant(wgpu::Color {
                    r: f64::from(r),
                    g: f64::from(g),
                    b: f64::from(b),
                    a: f64::from(a),
                });
            }
            // One instance, numbered so that `@builtin(instance_index)` is
            // the `gl_InstanceID` the rasterizer would have used.
            let instances = p.instance..p.instance + 1;
            match &index {
                Some((buffer, format, lowest)) => {
                    pass.set_index_buffer(buffer.slice(..), *format);
                    // The vertex buffer starts at the draw's lowest index, so
                    // every index in it is that much too high.
                    pass.draw_indexed(0..p.count, -(*lowest as i32), instances);
                }
                None => pass.draw(0..p.count, instances),
            }
        }
        timed!(self, encode, self.queue.submit([encoder.finish()]));
        Ok(())
    }

    /// One stage's bindings: its constant banks, and its textures with
    /// their samplers.
    fn bind_group(
        &self,
        p: &Prepared,
        stage: ShaderStage,
        group: u32,
    ) -> std::result::Result<(wgpu::BindGroupLayout, wgpu::BindGroup), String> {
        // The dimensionality is the layout's, because it is what the module
        // declared the binding as.
        let declared = if stage == ShaderStage::VertexB { &p.vs_layout } else { &p.fs_layout };
        let mut entries = Vec::new();
        let mut resources: Vec<Resource> = Vec::new();

        for upload in p.uploads.constants.iter().filter(|c| c.stage == stage) {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: upload.bank,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            resources.push(Resource::Buffer(
                upload.bank,
                self.buffer("constants", &upload.bytes, wgpu::BufferUsages::STORAGE),
            ));
        }

        for (index, upload) in p.uploads.textures.iter().filter(|t| t.stage == stage).enumerate() {
            let dim = declared
                .textures
                .iter()
                .find(|b| b.immediate == upload.immediate)
                .map(|b| b.dim)
                .ok_or("a texture the module never declared")?;
            let view_dimension = match dim {
                switch_core::gpu::shader::isa::TexDim::T2dArray => {
                    wgpu::TextureViewDimension::D2Array
                }
                _ => wgpu::TextureViewDimension::D2,
            };
            let binding = TEXTURE_BINDING + 2 * index as u32;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding + 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            resources.push(Resource::Texture(binding, self.texture(upload)?, view_dimension));
            resources.push(Resource::Sampler(binding + 1, self.sampler(upload)));
        }

        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stage"),
            entries: &entries,
        });
        // The views have to outlive the descriptor that borrows them.
        let views: Vec<Option<wgpu::TextureView>> = resources
            .iter()
            .map(|r| match r {
                Resource::Texture(_, texture, dimension) => {
                    Some(texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(*dimension),
                        ..Default::default()
                    }))
                }
                _ => None,
            })
            .collect();
        let bindings: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .zip(&views)
            .map(|(resource, view)| match resource {
                Resource::Buffer(binding, buffer) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                },
                Resource::Texture(binding, _, _) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(view.as_ref().expect("a view")),
                },
                Resource::Sampler(binding, sampler) => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            })
            .collect();
        let group_name = format!("group {group}");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&group_name),
            layout: &layout,
            entries: &bindings,
        });
        Ok((layout, bind_group))
    }

    fn texture(
        &self,
        upload: &switch_core::gpu::upload::TextureUpload,
    ) -> std::result::Result<wgpu::Texture, String> {
        let format =
            device_texture_format(&self.device, upload.format).map_err(|e| format!("{e:?}"))?;
        let size = wgpu::Extent3d {
            width: upload.width.max(1),
            height: upload.height.max(1),
            depth_or_array_layers: upload.layers.max(1),
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload.row_bytes),
                rows_per_image: Some(upload.rows),
            },
            size,
        );
        Ok(texture)
    }

    fn sampler(&self, upload: &switch_core::gpu::upload::TextureUpload) -> wgpu::Sampler {
        use switch_core::gpu::texture::Wrap;
        let wrap = |w: Wrap| match w {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
            Wrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            // WebGPU has no border mode at all -- `clamp-to-edge`, `repeat`
            // and `mirror-repeat` are the whole list -- and asking wgpu's web
            // backend for one is not an error it returns but a panic it
            // raises, which on wasm stops the core.
            //
            // Edge is also what this has to agree with: the rasterizer takes
            // `ClampToBorder` down the same arm as `ClampToEdge`
            // (`texture::Wrap`), so neither renderer samples a border colour
            // and the two match. When one of them learns to, they both must.
            Wrap::ClampToBorder => wgpu::AddressMode::ClampToEdge,
        };
        let filter = |linear: bool| {
            if linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            }
        };
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler"),
            address_mode_u: wrap(upload.sampler.wrap_u),
            address_mode_v: wrap(upload.sampler.wrap_v),
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter(upload.sampler.mag_linear),
            min_filter: filter(upload.sampler.min_linear),
            ..Default::default()
        })
    }

    /// A shader module, without asking the device whether it liked it.
    ///
    /// Asking means an error scope, and popping one means waiting. What the
    /// device rejects arrives through its uncaptured-error handler instead
    /// and is read at the start of the next draw — a frame late, which is
    /// what "do not block" costs and is cheap: the WGSL has already been
    /// through `naga` before it gets here.
    fn module(&mut self, what: &str, source: &str) -> wgpu::ShaderModule {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some(module) = self.modules.get(&key) {
            return module.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(what),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.modules.insert(key, module.clone());
        module
    }

    /// Whatever the device rejected since this was last asked.
    fn device_error(&self) -> Option<String> {
        self.failed.lock().ok().and_then(|mut e| e.take())
    }

    fn buffer(&self, what: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        // Padded to four bytes, which every buffer binding wants and a
        // twelve-byte index buffer is not.
        let mut padded = bytes.to_vec();
        while !padded.len().is_multiple_of(4) || padded.is_empty() {
            padded.push(0);
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(what),
            size: padded.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &padded);
        buffer
    }
}

/// The wgpu name for a format `switch_core::gpu::pipeline` resolved, checked
/// against what this device was actually given.
///
/// WebGPU gates the compressed families behind optional features, and creating
/// a texture in one the device does not have is not an error you can catch --
/// `createTexture` throws, and wgpu's web backend unwraps it. That is a panic
/// in the middle of a draw, which on wasm is a bare `unreachable` that takes
/// the whole core down: Just Dance 2019 reached its first BC1 texture and the
/// run loop stopped.
///
/// Refusing it here instead makes it what every other thing this backend
/// cannot express already is -- one draw on the software rasterizer.
fn device_texture_format(
    device: &wgpu::Device,
    format: Format,
) -> Result<wgpu::TextureFormat> {
    let wanted = texture_format(format)?;
    let needs = wanted.required_features();
    if !device.features().contains(needs) {
        return Err(Error::Gpu(format!(
            "the device was not given {needs:?}, which {wanted:?} needs"
        )));
    }
    Ok(wanted)
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

/// One draw, resolved into everything a device needs.
struct Prepared {
    state: Pipeline,
    color: Target,
    vs: Translation,
    fs: Translation,
    vs_layout: Layout,
    fs_layout: Layout,
    uploads: Uploads,
    /// Vertices for a sequential draw, indices for an indexed one.
    count: u32,
    /// `gl_InstanceID`, which WebGPU reproduces as the first instance of a
    /// one-instance draw.
    instance: u32,
}

impl Gpu {
    /// Read a draw out of the engine, or say what stops it running here.
    ///
    /// Everything this reaches for is `switch-core`'s: the translation, the
    /// pipeline state and the uploads all already exist and are already
    /// tested against the rasterizer. What is left is arranging them.
    fn prepare(
        &mut self,
        engine: &Engine3D,
        ctx: &ExecCtx,
    ) -> std::result::Result<Prepared, String> {
        let state = Pipeline::of(engine).map_err(|e| e.to_string())?;
        let color = Targets::of(engine)
            .map_err(|e| format!("{e:?}"))?
            .color
            .ok_or("a depth-only pass")?;

        // Unfolded, so a module depends only on the shader binary and not on
        // what happened to be in a constant buffer when it was translated.
        let (vs, fs) = timed!(self, translate, {
            (
                self.translate(engine, ctx, ShaderStage::VertexB),
                self.translate(engine, ctx, ShaderStage::Fragment),
            )
        });
        let (vs, fs) = (vs?, fs?);

        let mut vs_layout = Layout::of(&vs, Stage::Vertex);
        let mut fs_layout = Layout::of(&fs, Stage::Fragment);
        // The two stages have to name the same varyings: WebGPU will not
        // link a fragment input nothing produces. The union is what agrees
        // with the rasterizer, where a varying the vertex shader never wrote
        // reads as zero rather than as absent.
        let mut varyings = vs_layout.varyings.clone();
        varyings.extend(fs_layout.varyings.iter().copied());
        varyings.sort_unstable();
        varyings.dedup();
        vs_layout.varyings = varyings.clone();
        fs_layout.varyings = varyings;
        // Neither correction is anything the program says. Negated, because
        // WebGPU mirrors y on its own: the two agree exactly when the
        // guest's transform mirrors too, and the shader has to do it when
        // the guest's does not. See `Layout::flip_y`.
        vs_layout.flip_y = !state.viewport.flip_y;
        vs_layout.depth_minus_one_to_one = state.viewport.depth_minus_one_to_one();
        // One bind group per stage; see `Layout::group`.
        vs_layout.group = 0;
        fs_layout.group = 1;

        let mut immediates: Vec<(ShaderStage, u16)> = Vec::new();
        immediates.extend(vs.textures.iter().map(|&(imm, _)| (ShaderStage::VertexB, imm)));
        immediates.extend(fs.textures.iter().map(|&(imm, _)| (ShaderStage::Fragment, imm)));
        let mut banks: Vec<(ShaderStage, u32)> = Vec::new();
        banks.extend(vs.const_banks.iter().map(|&b| (ShaderStage::VertexB, u32::from(b))));
        banks.extend(fs.const_banks.iter().map(|&b| (ShaderStage::Fragment, u32::from(b))));
        let uploads = timed!(self, upload, {
            Uploads::of(engine, &state, ctx, Banks::Read(&banks), &immediates)
        })
        .map_err(|e| format!("{e:?}"))?;

        // The swizzle is in the descriptor, which is guest memory the draw
        // points at, so the translation cannot know it and the layout has to
        // be told. WebGPU has no per-texture component swizzle.
        for (layout, stage) in
            [(&mut vs_layout, ShaderStage::VertexB), (&mut fs_layout, ShaderStage::Fragment)]
        {
            for binding in &mut layout.textures {
                if let Some(upload) = uploads
                    .textures
                    .iter()
                    .find(|t| t.stage == stage && t.immediate == binding.immediate)
                {
                    binding.swizzle = upload.swizzle;
                }
            }
        }

        if switch_core::env_flag!("TRACE_GPU_TEX") {
            for t in &uploads.textures {
                eprintln!(
                    "[gpu-tex] imm={} {:?} {}x{} swizzle={:?} sampler={:?}",
                    t.immediate, t.format, t.width, t.height, t.swizzle, t.sampler
                );
            }
        }
        Ok(Prepared {
            state,
            color,
            vs,
            fs,
            vs_layout,
            fs_layout,
            uploads,
            count: engine.last_draw.count,
            instance: engine.instance_id(),
        })
    }

    fn translate(
        &self,
        engine: &Engine3D,
        ctx: &ExecCtx,
        stage: ShaderStage,
    ) -> std::result::Result<Translation, String> {
        let binding = engine.program(stage).ok_or_else(|| format!("no {stage:?} program"))?;
        let program = switch_core::gpu::shader::decode_program_from_memory(
            ctx,
            binding.addr,
            &|bank: u8| engine.bound_constbuf(stage, u32::from(bank)),
        )
        .map_err(|e| format!("{e:?}"))?;
        wgsl::translate(&Compiled::new(&program)).map_err(|e| e.to_string())
    }
}

impl Renderer for Gpu {
    fn draw(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) -> Result<()> {
        // Anything at all going wrong here runs the draw on the rasterizer
        // instead. That is not timidity: the rasterizer is the reference, so
        // a frame is always either right or a frame the reference produced,
        // and `cmp` against it measures how much of the work this actually
        // did rather than how much it got away with.
        let index = self.in_frame;
        self.in_frame += 1;
        if self.only.is_some_and(|only| only != index) {
            self.flush(ctx)?;
            return self.software.draw(engine, ctx);
        }
        match self.prepare(engine, &*ctx) {
            Ok(prepared) => match self.render(&prepared, ctx) {
                Ok(()) => {
                    self.drawn += 1;
                    return Ok(());
                }
                Err(why) => self.fall_back(why),
            },
            Err(why) => self.fall_back(why),
        }
        // The rasterizer reads and writes guest memory, so it has to be the
        // truth before a draw runs there.
        self.flush(ctx)?;
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
        // A clear goes through the rasterizer, which writes guest memory —
        // so anything held has to go back first, or the clear would be
        // overwritten by a surface handed back after it.
        self.flush(ctx)?;
        // A frame starts at its clear.
        self.in_frame = 0;
        self.software.clear_color(engine, ctx, target, layer, channels)
    }

    fn clear_depth_stencil(
        &mut self,
        engine: &Engine3D,
        ctx: &mut ExecCtx,
        depth: bool,
        stencil: bool,
    ) -> Result<()> {
        // A clear goes through the rasterizer, so guest memory has to be the
        // truth again before it does.
        self.flush(ctx)?;
        self.software.clear_depth_stencil(engine, ctx, depth, stencil)
    }

    fn flush(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        let at = self.times.map(|_| std::time::Instant::now());
        let result = self.flush_inner(ctx);
        if let (Some(at), Some(t)) = (at, self.times.as_mut()) {
            t.flush += at.elapsed().as_micros();
        }
        result
    }
}

impl Gpu {
    fn flush_inner(&mut self, ctx: &mut ExecCtx) -> Result<()> {
        for held in std::mem::take(&mut self.evicted) {
            self.write_back(&held, ctx)?;
        }
        let addresses: Vec<u64> = self.held.keys().copied().collect();
        for addr in addresses {
            self.flush_one(addr, ctx)?;
        }
        Ok(())
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
