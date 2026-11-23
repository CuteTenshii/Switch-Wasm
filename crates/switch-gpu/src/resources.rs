//! Shader modules, scratch buffers, device errors, and cache eviction.

use crate::bind::{bytes_fingerprint, same_bytes};
use crate::readback::Scratch;
use crate::{
    Gpu, Input, CONSTANT_ARENA_BYTES, INPUT_CACHE_BYTES, PAGE_BITS, SHARED_BUFFER_BYTES,
    TEXTURE_CACHE_BYTES,
};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::renderer::Software;
use switch_core::gpu::shader::wgsl::{self, Layout, Stage, Translation};
use switch_core::gpu::upload::{TextureKey, TextureUpload};

/// Bounds [`Gpu::views`]; a title's draws bind far fewer textures.
const MAX_VIEWS: usize = 4096;

/// Bounds [`Gpu::built_modules`]; a title draws with far fewer program and layout pairs.
const BUILT_MODULE_ENTRIES: usize = 4096;

/// The CPU pages behind `len` bytes at `addr`, as far as the mapping goes.
pub(super) fn guest_pages(ctx: &ExecCtx, addr: u64, len: u64) -> Vec<u32> {
    let end = addr.saturating_add(len);
    let mut pages = Vec::new();
    let mut at = addr;
    while at < end {
        let Some((cpu, run)) = ctx.vmm.translate(at) else {
            break;
        };
        let take = run.min(end - at);
        if take == 0 {
            break;
        }
        let first = cpu >> PAGE_BITS;
        let last = (cpu + take - 1) >> PAGE_BITS;
        pages.extend((first..=last).map(|p| p as u32));
        at += take;
    }
    pages
}

impl Gpu {
    /// The module for this WGSL and its cache key; rejections arrive via the error handler.
    pub(super) fn module(&mut self, what: &str, source: &str) -> (u64, wgpu::ShaderModule) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some(module) = self.modules.get(&key) {
            return (key, module.clone());
        }
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(what),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.modules.insert(key, module.clone());
        (key, module)
    }

    /// The module of `translation`, numbered `serial`, as `stage` with `layout`; written and
    /// hashed only the first time.
    pub(super) fn built_module(
        &mut self,
        what: &str,
        (translation, serial): (&Translation, u64),
        stage: Stage,
        layout: &Layout,
    ) -> std::result::Result<(u64, wgpu::ShaderModule), String> {
        let key = (
            serial,
            match stage {
                Stage::Vertex => 0,
                Stage::Fragment => 1,
                Stage::Compute => 2,
            },
        );
        if let Some((_, hash, module)) = self
            .built_modules
            .get(&key)
            .and_then(|built| built.iter().find(|(l, _, _)| l == layout))
        {
            return Ok((*hash, module.clone()));
        }
        let source =
            wgsl::module(translation, stage, layout).map_err(|e| format!("module: {e}"))?;
        let (hash, module) = self.module(what, &source);
        // Translations replaced since leave entries no draw will ask for.
        if self.built_modules.len() >= BUILT_MODULE_ENTRIES {
            self.built_modules.clear();
        }
        self.built_modules
            .entry(key)
            .or_default()
            .push((layout.clone(), hash, module.clone()));
        Ok((hash, module))
    }

    /// Whatever the device rejected since this was last asked.
    pub(super) fn device_error(&self) -> Option<String> {
        self.failed.lock().ok().and_then(|mut e| e.fresh.take())
    }

    /// Every distinct rejection and the total count, without draining.
    pub(super) fn device_errors(&self) -> (u64, Vec<String>) {
        match self.failed.lock() {
            Ok(e) => (e.count, e.distinct.clone()),
            Err(_) => (0, Vec::new()),
        }
    }

    /// A buffer for the draw in progress; see [`Gpu::scratch`].
    pub(super) fn buffer(
        &mut self,
        what: &str,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buffer = self.kept_buffer(what, bytes, usage);
        self.scratch.push(Scratch::Buffer(buffer.clone()));
        buffer
    }

    /// A buffer of `bytes` the device may already hold from an earlier draw. Nothing may write
    /// to it, since later draws share it.
    pub(super) fn shared_buffer(
        &mut self,
        what: &str,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let key = (usage.bits(), bytes_fingerprint(bytes));
        if let Some((_, buffer)) = self
            .shared_buffers
            .get(&key)
            .and_then(|kept| kept.iter().find(|(held, _)| same_bytes(held, bytes)))
        {
            self.buffers_reused += 1;
            return buffer.clone();
        }
        self.buffers_uploaded += 1;
        let buffer = self.kept_buffer(what, bytes, usage);
        if self.shared_bytes + bytes.len() as u64 > SHARED_BUFFER_BYTES {
            // Recorded draws may still bind them, so they go when the scratch does.
            for (_, buffer) in self.shared_buffers.drain().flat_map(|(_, kept)| kept) {
                self.scratch.push(Scratch::Buffer(buffer));
            }
            self.shared_bytes = 0;
        }
        self.shared_bytes += bytes.len() as u64;
        self.shared_buffers
            .entry(key)
            .or_default()
            .push((bytes.to_vec(), buffer.clone()));
        buffer
    }

    /// A buffer of `bytes` that its owner destroys.
    pub(super) fn kept_buffer(
        &mut self,
        what: &str,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Queue writes take whole words, so only a ragged or empty upload is copied to pad it.
        let padded;
        let bytes = if bytes.len().is_multiple_of(4) && !bytes.is_empty() {
            bytes
        } else {
            let mut words = bytes.to_vec();
            words.resize(bytes.len().next_multiple_of(4).max(4), 0);
            padded = words;
            &padded[..]
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(what),
            size: bytes.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// Drop every cached texture the guest has written over.
    pub(super) fn evict_written(&mut self, ctx: &mut ExecCtx) {
        if !ctx.mem.has_dirty_gpu() {
            self.watch_held(ctx, &[]);
            return;
        }
        let dirty = ctx.mem.dirty_gpu_pages();
        self.watch_held(ctx, &dirty);
        self.evict_pages(dirty);
    }

    /// Stencil the device does not hold goes straight to guest memory, on a held surface's
    /// pages: what was just written there is ours, and no reason to sync. Run `write`
    /// between what the guest wrote before it and what it writes after.
    pub(super) fn writing_guest_memory<T>(
        &mut self,
        ctx: &mut ExecCtx,
        write: impl FnOnce(&mut Software, &mut ExecCtx) -> T,
    ) -> T {
        self.evict_written(ctx);
        let out = write(&mut self.software, ctx);
        let dirty = ctx.mem.dirty_gpu_pages();
        for held in self.space.held.values() {
            for &page in held.watched.iter().filter(|page| dirty.contains(page)) {
                ctx.mem.mark_gpu_page(u64::from(page) << PAGE_BITS);
            }
        }
        self.evict_pages(dirty);
        out
    }

    fn evict_pages(&mut self, dirty: Vec<u32>) {
        self.evict_shaders(&dirty);
        for page in &dirty {
            for key in self.space.input_pages.remove(page).unwrap_or_default() {
                self.forget_input(key);
            }
        }
        for page in dirty {
            let Some(keys) = self.space.page_owners.remove(&page) else {
                continue;
            };
            for key in keys {
                if let Some(bytes) = self.space.texture_cache.remove(&key) {
                    self.cached_bytes -= bytes.len() as u64;
                    self.drop_gpu_texture(&key, bytes.len() as u64);
                }
            }
        }
    }

    /// The device copy of `len` bytes at `addr` that a compute launch only reads, uploaded
    /// on first use and kept until the guest writes them.
    pub(super) fn compute_input(
        &mut self,
        addr: u64,
        len: usize,
        ctx: &ExecCtx,
    ) -> std::result::Result<wgpu::Buffer, String> {
        let key = (addr, len as u64);
        let pages = guest_pages(ctx, addr, len as u64);
        if let Some(kept) = self.space.inputs.get(&key) {
            if kept.pages == pages {
                self.inputs_reused += 1;
                return Ok(kept.buffer.clone());
            }
        }
        self.forget_input(key);
        // Watched before the read, so a store racing it is reported.
        for &page in &pages {
            ctx.mem.mark_gpu_page(u64::from(page) << PAGE_BITS);
        }
        let mut bytes = vec![0u8; len];
        ctx.vmm
            .read_into(ctx.mem, addr, &mut bytes)
            .map_err(|e| format!("{e:?}"))?;
        let buffer = self.kept_buffer("compute input", &bytes, wgpu::BufferUsages::STORAGE);
        self.inputs_uploaded += 1;
        if pages.is_empty() {
            self.scratch.push(Scratch::Buffer(buffer.clone()));
            return Ok(buffer);
        }
        if self.space.input_bytes + buffer.size() > INPUT_CACHE_BYTES {
            let keys: Vec<_> = self.space.inputs.keys().copied().collect();
            for key in keys {
                self.forget_input(key);
            }
        }
        for &page in &pages {
            self.space.input_pages.entry(page).or_default().push(key);
        }
        self.space.input_bytes += buffer.size();
        self.space.inputs.insert(
            key,
            Input {
                buffer: buffer.clone(),
                pages,
            },
        );
        Ok(buffer)
    }

    /// Drop a kept compute input; launches recorded but not yet submitted may still bind it.
    fn forget_input(&mut self, key: (u64, u64)) {
        let Some(input) = self.space.inputs.remove(&key) else {
            return;
        };
        for page in &input.pages {
            if let Some(keys) = self.space.input_pages.get_mut(page) {
                keys.retain(|k| *k != key);
                if keys.is_empty() {
                    self.space.input_pages.remove(page);
                }
            }
        }
        self.space.input_bytes -= input.buffer.size();
        self.scratch.push(Scratch::Buffer(input.buffer));
    }

    /// Destroy a cached texture's device copy.
    fn drop_gpu_texture(&mut self, key: &TextureKey, bytes: u64) {
        if let Some(made) = self.space.gpu_textures.remove(key) {
            self.submit_batch();
            for (_, texture) in made {
                texture.destroy();
                self.gpu_texture_bytes = self.gpu_texture_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Keep the textures a draw read before it binds them, so a new texture's device copy is
    /// kept from its first draw; textures with no watchable pages are not kept.
    pub(super) fn keep_textures(&mut self, ctx: &ExecCtx, uploads: &[TextureUpload]) {
        for upload in uploads {
            let key = upload.key;
            if self.space.texture_cache.contains_key(&key) {
                continue;
            }
            self.uploaded.add_texture(upload.bytes.len());
            self.texture_misses += 1;
            // `source_len` is an upper bound that can exceed the mapping, so stop where the mapping does.
            let pages = guest_pages(ctx, key.addr, upload.source_len);
            // A null image's stand-in has no pages, and nothing can write it.
            if pages.is_empty() && key.addr != 0 {
                continue;
            }
            // Whole-cache eviction, every address space's: the limit is shared.
            let len = upload.bytes.len() as u64;
            if self.cached_bytes + len > TEXTURE_CACHE_BYTES {
                self.submit_batch();
                for space in std::iter::once(&mut self.space).chain(self.parked.values_mut()) {
                    for (_, made) in space.gpu_textures.drain() {
                        for (_, texture) in made {
                            texture.destroy();
                        }
                    }
                    space.texture_cache.clear();
                    space.page_owners.clear();
                }
                self.gpu_texture_bytes = 0;
                self.cached_bytes = 0;
            }
            for &page in &pages {
                self.texture_pages_to_watch.push(page);
                self.space.page_owners.entry(page).or_default().push(key);
            }
            self.space.texture_cache.insert(key, upload.bytes.clone());
            self.cached_bytes += len;
        }
    }

    /// Watch the pages of what the last draw or launch kept, now that `ctx` is mutable.
    pub(super) fn remember_textures(&mut self, ctx: &mut ExecCtx) {
        for (page, key) in std::mem::take(&mut self.shader_to_watch) {
            ctx.mem.mark_gpu_page(u64::from(page) << PAGE_BITS);
            self.space.shader_pages.entry(page).or_default().push(key);
        }
        for page in std::mem::take(&mut self.texture_pages_to_watch) {
            ctx.mem.mark_gpu_page(u64::from(page) << PAGE_BITS);
        }
    }

    /// A view of the whole of `texture`, of `dimension` or the texture's own; kept, since a
    /// browser makes each view a JavaScript object and draws keep binding the same ones.
    pub(super) fn view(
        &mut self,
        texture: &wgpu::Texture,
        dimension: Option<wgpu::TextureViewDimension>,
    ) -> wgpu::TextureView {
        let key = (texture.clone(), dimension);
        if let Some(view) = self.views.get(&key) {
            return view.clone();
        }
        // Views of destroyed textures are never asked for again; this bounds them.
        if self.views.len() >= MAX_VIEWS {
            self.views.clear();
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension,
            ..Default::default()
        });
        self.views.insert(key, view.clone());
        view
    }

    /// The buffer [`Gpu::stage_constants`] places constants in.
    pub(super) fn constant_arena(&mut self) -> wgpu::Buffer {
        self.constant_arena
            .get_or_insert_with(|| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("constants"),
                    size: CONSTANT_ARENA_BYTES,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .clone()
    }

    /// Place a stage's packed constants in the arena, answering the dynamic offset and the size
    /// to bind; `None` when they could never fit. The size is rounded up to a power of two so
    /// that bind groups repeat: `cbRead` bounds each bank by the table, never by the binding.
    pub(super) fn stage_constants(&mut self, bytes: &[u8]) -> Option<(u32, u64)> {
        let size = (bytes.len() as u64).next_power_of_two().max(256);
        if size > CONSTANT_ARENA_BYTES {
            return None;
        }
        // Whether this draw placed constants already, which the arena must keep until it is
        // recorded and submitted.
        let placed_already = self.constants_unrecorded;
        self.constants_unrecorded = true;
        let align = u64::from(self.device_limits.min_storage_buffer_offset_alignment);
        let mut offset = (self.constant_staging.len() as u64).next_multiple_of(align);
        if offset + size > CONSTANT_ARENA_BYTES {
            // Starting over would let later draws overwrite what this one placed first.
            if placed_already {
                return None;
            }
            // Every draw placed so far goes with this submit, so the arena can start over.
            self.constants_unrecorded = false;
            self.submit_batch();
            self.constants_unrecorded = true;
            offset = 0;
        }
        self.constant_staging.resize(offset as usize, 0);
        self.constant_staging.extend_from_slice(bytes);
        // Writes to the queue take whole words.
        let words = self.constant_staging.len().next_multiple_of(4);
        self.constant_staging.resize(words, 0);
        Some((offset as u32, size))
    }

    /// Start the arena over once a submit has taken every draw that reads it.
    fn restart_constants(&mut self) {
        if !self.constants_unrecorded {
            self.constant_staging.clear();
            self.constants_flushed = 0;
        }
    }

    /// Write what was placed in the arena since the last submit, ahead of the next one. Only
    /// new bytes are written: draws recorded but not yet submitted still read the older ones.
    fn flush_constants(&mut self) {
        let from = self.constants_flushed;
        if self.constant_staging.len() > from {
            if let Some(arena) = &self.constant_arena {
                self.queue
                    .write_buffer(arena, from as u64, &self.constant_staging[from..]);
            }
            self.constants_flushed = self.constant_staging.len();
        }
    }

    /// Submit the draws recorded so far. A queue submit costs over a millisecond in a browser,
    /// so draws share one; anything else that submits, or destroys what they use, goes first.
    pub(super) fn submit_batch(&mut self) {
        self.end_pass();
        self.flush_constants();
        if let Some(batch) = self.batch.take() {
            self.batched_draws = 0;
            timed!(self, encode, self.queue.submit([batch.finish()]));
            self.restart_constants();
        }
    }

    /// Submit an encoder after the draws recorded before it.
    pub(super) fn submit_with_batch(&mut self, encoder: wgpu::CommandEncoder) {
        self.end_pass();
        self.flush_constants();
        let batch = self.batch.take().map(|batch| batch.finish());
        self.batched_draws = 0;
        timed!(
            self,
            encode,
            self.queue
                .submit(batch.into_iter().chain([encoder.finish()]))
        );
        self.restart_constants();
    }

    pub(super) fn release_scratch(&mut self) {
        if self.batch.is_some() {
            // Recorded draws still use it, unless they are submitted first: a pass of hundreds of
            // draws can each leave a large stale copy behind.
            let pending: u64 = self.scratch.iter().map(Scratch::bytes).sum();
            if pending < crate::MAX_SCRATCH_BYTES {
                return;
            }
            self.submit_batch();
        }
        for made in self.scratch.drain(..) {
            match made {
                Scratch::Buffer(buffer) => buffer.destroy(),
                Scratch::Texture(texture) => texture.destroy(),
            }
        }
        if self.bind_groups.len() >= crate::MAX_BIND_GROUPS {
            for binding in self.bind_groups.drain().map(|(_, binding)| binding) {
                for buffer in binding.buffers {
                    buffer.destroy();
                }
            }
        }
    }
}
