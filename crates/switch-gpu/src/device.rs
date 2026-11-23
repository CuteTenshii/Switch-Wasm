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
        let (device_features, device_limits) = (device.features(), device.limits());
        let vertex_buffer_limit = device_limits.max_vertex_buffers;
        let info = adapter.get_info();
        let warp_subgroups = device_features.contains(wgpu::Features::SUBGROUP)
            && (info.subgroup_min_size, info.subgroup_max_size) == (32, 32);
        Gpu {
            warp_subgroups,
            _instance: instance,
            adapter,
            device,
            device_features,
            device_limits,
            queue,
            vertex_buffer_limit,
            space_id: 0,
            space: Space::default(),
            parked: std::collections::HashMap::new(),
            scratch: Vec::new(),
            bind_groups: std::collections::HashMap::new(),
            modules: std::collections::HashMap::new(),
            built_modules: std::collections::HashMap::new(),
            translations: 0,
            no_fragment: None,
            constant_arena: None,
            sequential_indices: std::collections::HashMap::new(),
            views: std::collections::HashMap::new(),
            pass_calls_skipped: 0,
            later_instances: 0,
            instances_merged: 0,
            constant_staging: Vec::new(),
            constants_flushed: 0,
            last_constants: [None; 2],
            constants_unrecorded: false,
            shared_buffers: std::collections::HashMap::new(),
            shared_bytes: 0,
            passes_joined: 0,
            buffers_uploaded: 0,
            buffers_reused: 0,
            pipelines: std::collections::HashMap::new(),
            samplers: std::collections::HashMap::new(),
            group_layouts: std::collections::HashMap::new(),
            depth_loaders: std::collections::HashMap::new(),
            depth_readers: std::collections::HashMap::new(),
            packed_depth: std::collections::HashMap::new(),
            clear_pipelines: std::collections::HashMap::new(),
            blit_pipelines: std::collections::HashMap::new(),
            screen: None,
            scan_pipelines: std::collections::HashMap::new(),
            sync_every_present: false,
            resample_pipelines: std::collections::HashMap::new(),
            failed,
            lost,
            deferred_readbacks: false,
            crop_switches: 0,
            pass: None,
            batch: None,
            batched_draws: 0,
            device_is_shared: false,
            defer_readbacks: false,
            interleave: false,
            software_frame: false,
            forced_software: false,
            syncs: [0; 6],
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
            bind_hits: 0,
            bind_misses: 0,
            bind_uncached: 0,
            bind_uncached_why: [0; 4],
            held_copies_made: 0,
            held_copies_reused: 0,
            held_copies_updated: 0,
            cropped_binds: 0,
            compute_dispatched: 0,
            inputs_uploaded: 0,
            inputs_reused: 0,
            software_compute: false,
            compute_refusals: std::collections::BTreeMap::new(),
            compute_pipelines: std::collections::HashMap::new(),
            texture_misses: 0,
            gpu_texture_bytes: 0,
            texture_pages_to_watch: Vec::new(),
            only: None,
        }
    }

    /// Open a device by blocking on it, which only a native thread may do.
    pub fn open() -> std::result::Result<Gpu, String> {
        let (instance, adapter, device, queue) = Gpu::open_parts()?;
        Ok(Gpu::with_device(instance, adapter, device, queue))
    }

    pub(crate) fn open_parts(
    ) -> std::result::Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), String>
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| format!("no adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&device_descriptor(&adapter)))
                .map_err(|e| format!("no device: {e}"))?;
        Ok((instance, adapter, device, queue))
    }

    pub fn describe(&self) -> String {
        format!("{:?}", self.device_limits.max_texture_dimension_2d)
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
        self.bind_groups.clear();
        self.shared_buffers.clear();
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
        if !self.software_frame || !checked || self.forced_software {
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

    /// The `--software-compute` flag: leave every compute launch to the software dispatch.
    pub fn set_software_compute(&mut self, software: bool) {
        self.software_compute = software;
    }

    /// The `--interleave` flag. See [`Gpu::interleave`].
    pub fn set_interleave(&mut self, interleave: bool) {
        self.interleave = interleave;
    }

    /// The `--web-limits` flag. See [`Gpu::web_limits`].
    pub fn set_web_limits(&mut self, web_limits: bool) {
        self.web_limits = web_limits;
    }

    /// What this device may be asked for; see [`Gpu::web_limits`].
    pub(super) fn features(&self) -> wgpu::Features {
        let features = self.device_features;
        if self.web_limits {
            features - wgpu::Features::SUBGROUP - wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        } else {
            features
        }
    }

    /// The `--device-msaa` flag. See [`Gpu::route`].
    pub fn set_device_msaa(&mut self, device_msaa: bool) {
        self.device_msaa = device_msaa;
    }

    /// The `--defer-readbacks` flag. See [`Gpu::defer_readbacks`].
    pub fn set_defer_readbacks(&mut self, defer_readbacks: bool) {
        self.defer_readbacks = defer_readbacks;
    }

    /// The `--sync-every-present` flag. See [`Gpu::sync_every_present`].
    pub fn set_sync_every_present(&mut self, sync: bool) {
        self.sync_every_present = sync;
    }

    /// The `--times` flag: record phase timings, which a browser always does.
    pub fn set_times(&mut self, on: bool) {
        if on != self.times.is_some() {
            self.times = on.then(Times::default);
        }
    }

    /// The `--only` range. See [`Gpu::only`].
    pub fn set_only(&mut self, only: Option<std::ops::Range<u32>>) {
        self.only = only;
    }
}
