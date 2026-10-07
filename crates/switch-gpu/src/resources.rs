//! Shader modules, scratch buffers, device errors, and cache eviction.

use crate::readback::Scratch;
use crate::{Gpu, PAGE_BITS, TEXTURE_CACHE_BYTES};
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::upload::TextureKey;

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
        // Padded to four bytes.
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
        self.scratch.push(Scratch::Buffer(buffer.clone()));
        buffer
    }

    /// Drop every cached texture the guest has written over.
    pub(super) fn evict_written(&mut self, ctx: &mut ExecCtx) {
        if !ctx.mem.has_dirty_gpu() {
            return;
        }
        let dirty = ctx.mem.dirty_gpu_pages();
        self.evict_shaders(&dirty);
        for page in dirty {
            let Some(keys) = self.page_owners.remove(&page) else {
                continue;
            };
            for key in keys {
                if let Some(bytes) = self.texture_cache.remove(&key) {
                    self.cached_bytes -= bytes.len() as u64;
                    self.drop_gpu_texture(&key, bytes.len() as u64);
                }
            }
        }
    }

    /// Destroy a cached texture's device copy.
    fn drop_gpu_texture(&mut self, key: &TextureKey, bytes: u64) {
        if let Some(made) = self.gpu_textures.remove(key) {
            for (_, texture) in made {
                texture.destroy();
                self.gpu_texture_bytes = self.gpu_texture_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Keep what this draw read and watch its pages; textures with no watchable pages are not kept.
    pub(super) fn remember_textures(&mut self, ctx: &mut ExecCtx) {
        for (page, key) in std::mem::take(&mut self.shader_to_watch) {
            ctx.mem.mark_gpu_page(page << PAGE_BITS);
            self.shader_pages.entry(page).or_default().push(key);
        }
        for (key, bytes, source_len) in std::mem::take(&mut self.to_remember) {
            if self.texture_cache.contains_key(&key) {
                continue;
            }
            self.texture_misses += 1;
            // `source_len` is an upper bound that can exceed the mapping, so stop where the mapping does.
            let end = key.addr.saturating_add(source_len);
            let mut pages: Vec<u32> = Vec::new();
            let mut at = key.addr;
            while at < end {
                let Some((cpu, run)) = ctx.vmm.translate(at) else {
                    break;
                };
                let take = run.min(end - at);
                if take == 0 {
                    break;
                }
                let first = u64::from(cpu) >> PAGE_BITS;
                let last = (u64::from(cpu) + take - 1) >> PAGE_BITS;
                pages.extend((first..=last).map(|p| p as u32));
                at += take;
            }
            if pages.is_empty() {
                continue;
            }
            // Whole-cache eviction: reaching the limit means the textures changed wholesale.
            let len = bytes.len() as u64;
            if self.cached_bytes + len > TEXTURE_CACHE_BYTES {
                for (_, made) in self.gpu_textures.drain() {
                    for (_, texture) in made {
                        texture.destroy();
                    }
                }
                self.gpu_texture_bytes = 0;
                self.texture_cache.clear();
                self.page_owners.clear();
                self.cached_bytes = 0;
            }
            for &page in &pages {
                ctx.mem.mark_gpu_page(page << PAGE_BITS);
                self.page_owners.entry(page).or_default().push(key);
            }
            self.texture_cache.insert(key, bytes);
            self.cached_bytes += len;
        }
    }

    pub(super) fn release_scratch(&mut self) {
        for made in self.scratch.drain(..) {
            match made {
                Scratch::Buffer(buffer) => buffer.destroy(),
                Scratch::Texture(texture) => texture.destroy(),
            }
        }
    }
}
