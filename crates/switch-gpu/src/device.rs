//! Opening the device, falling back, and backend settings.

use crate::stats::{DeviceErrors, Times, UploadBytes};
use crate::{device_descriptor, Gpu, Space};
use switch_core::gpu::engine::threed::Engine3D;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::renderer::Software;

impl Gpu {
    /// Take a device opened elsewhere (asynchronously, in a browser).
    /// The instance and adapter must be kept: see [`Gpu::_instance`].
    pub fn with_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Gpu {
        let failed: std::sync::Arc<std::sync::Mutex<DeviceErrors>> =
            std::sync::Arc::new(std::sync::Mutex::new(DeviceErrors::default()));
        let sink = failed.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = sink.lock() {
                slot.record(e.to_string());
            }
        }));
        let lost: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let sink = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut slot) = sink.lock() {
                slot.get_or_insert(format!("{reason:?}: {message}"));
            }
        });
        let vertex_buffer_limit = device.limits().max_vertex_buffers;
        Gpu {
            _instance: instance,
            adapter,
            device,
            queue,
            vertex_buffer_limit,
            space_id: 0,
            space: Space::default(),
            parked: std::collections::HashMap::new(),
            scratch: Vec::new(),
            modules: std::collections::HashMap::new(),
            pipelines: std::collections::HashMap::new(),
            samplers: std::collections::HashMap::new(),
            group_layouts: std::collections::HashMap::new(),
            depth_loaders: std::collections::HashMap::new(),
            packed_depth: None,
            clear_pipelines: std::collections::HashMap::new(),
            resample_pipelines: std::collections::HashMap::new(),
            failed,
            lost,
            deferred_readbacks: false,
            defer_readbacks: false,
            interleave: false,
            software_frame: false,
            fell_back_this_frame: false,
            checked_this_frame: false,
            would_fall_back_this_frame: false,
            clean_frames: 0,
            clean_frames_needed: 1,
            unlatched: 0,
            gave_up: false,
            web_limits: false,
            device_msaa: false,
            report: None,
            software: Software,
            drawn: 0,
            fallbacks: 0,
            last_fallback: None,
            reasons: Vec::new(),
            direct: 0,
            expanded: 0,
            multisampled: 0,
            per_pixel: 0,
            in_frame: 0,
            times: cfg!(target_arch = "wasm32").then(Times::default),
            uploaded: UploadBytes::default(),
            shader_to_watch: Vec::new(),
            cached_bytes: 0,
            texture_hits: 0,
            shader_hits: 0,
            shader_misses: 0,
            texture_misses: 0,
            gpu_texture_bytes: 0,
            to_remember: Vec::new(),
            only: None,
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
        Ok(Gpu::with_device(instance, adapter, device, queue))
    }

    pub fn describe(&self) -> String {
        format!("{:?}", self.device.limits().max_texture_dimension_2d)
    }

    pub(super) fn fall_back(&mut self, why: String) {
        self.fallbacks += 1;
        self.fell_back_this_frame = true;
        if !self.reasons.contains(&why) {
            eprintln!("[gpu] falling back: {why}");
            self.reasons.push(why.clone());
        }
        self.last_fallback = Some(why);
    }

    /// Hand the frame to the rasterizer for good once the device is lost; answers whether it has been.
    pub(super) fn give_up(&mut self) -> bool {
        if self.gave_up {
            return true;
        }
        let Some(why) = self.lost.lock().ok().and_then(|slot| slot.clone()) else {
            return false;
        };
        self.gave_up = true;
        let said = format!("the device was lost ({why}); the rasterizer has the frame from here");
        eprintln!("[gpu] {said}");
        self.report = Some(said);
        self.space = Space::default();
        self.parked.clear();
        self.scratch.clear();
        true
    }

    /// Check a draw of the rasterizer's frame to see whether the device could have drawn it.
    pub(super) fn check_for_release(&mut self, engine: &Engine3D, ctx: &mut ExecCtx) {
        self.evict_written(ctx);
        let verdict = self.check(engine, &*ctx);
        self.remember_textures(ctx);
        self.checked_this_frame = true;
        if verdict.is_err() {
            self.would_fall_back_this_frame = true;
        }
    }

    /// At the clear that ends a rasterizer's frame, count it towards releasing the latch.
    pub(super) fn release_if_clean(&mut self) {
        let (checked, would_fall_back) = (self.checked_this_frame, self.would_fall_back_this_frame);
        self.checked_this_frame = false;
        self.would_fall_back_this_frame = false;
        if !self.software_frame || !checked {
            return;
        }
        if would_fall_back {
            self.clean_frames = 0;
            return;
        }
        self.clean_frames += 1;
        if self.clean_frames < self.clean_frames_needed {
            return;
        }
        self.software_frame = false;
        self.clean_frames = 0;
        self.unlatched += 1;
        eprintln!(
            "[gpu] every draw of the last {} frame(s) could have run on the device; \
             it has the frames again",
            self.clean_frames_needed
        );
    }

    /// The `GPU_INTERLEAVE` flag. See [`Gpu::interleave`].
    pub fn set_interleave(&mut self, interleave: bool) {
        self.interleave = interleave;
    }

    /// The `GPU_WEB_LIMITS` flag. See [`Gpu::web_limits`].
    pub fn set_web_limits(&mut self, web_limits: bool) {
        self.web_limits = web_limits;
    }

    /// What this device may be asked for; see [`Gpu::web_limits`].
    pub(super) fn features(&self) -> wgpu::Features {
        let features = self.device.features();
        if self.web_limits {
            features - wgpu::Features::SUBGROUP - wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        } else {
            features
        }
    }

    /// The `GPU_DEVICE_MSAA` flag. See [`Gpu::route`].
    pub fn set_device_msaa(&mut self, device_msaa: bool) {
        self.device_msaa = device_msaa;
    }

    /// The `GPU_DEFER_READBACKS` flag. See [`Gpu::defer_readbacks`].
    pub fn set_defer_readbacks(&mut self, defer_readbacks: bool) {
        self.defer_readbacks = defer_readbacks;
    }

    /// The `GPU_TIMES` flag: record phase timings, which a browser always does.
    pub fn set_times(&mut self, on: bool) {
        if on != self.times.is_some() {
            self.times = on.then(Times::default);
        }
    }

    /// The `GPU_ONLY` range. See [`Gpu::only`].
    pub fn set_only(&mut self, only: Option<std::ops::Range<u32>>) {
        self.only = only;
    }
}
