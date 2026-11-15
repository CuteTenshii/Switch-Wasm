//! Compute launches on the device. What a launch writes stays in held buffers until the next
//! flush, and draws that read it as vertices copy it on the device; anything else that reads
//! it flushes first. A launch this cannot run goes to the software dispatch.

use crate::bind::pack_banks;
use crate::readback::Readback;
use crate::{Feed, GlobalUpload, Gpu, Patch};
use switch_core::gpu::compute::{self as software, Launch};
use switch_core::gpu::engine::compute::EngineCompute;
use switch_core::gpu::engine::threed::ShaderStage;
use switch_core::gpu::exec::ExecCtx;
use switch_core::gpu::shader::wgsl::{self, ComputeShape, Layout, Stage};
use switch_core::gpu::upload::ConstantUpload;
use switch_core::Result;

/// A compute pipeline, by its module and its bind group layout.
pub(crate) type ComputeKey = (u64, Vec<wgpu::BindGroupLayoutEntry>);

impl Gpu {
    /// See [`switch_core::gpu::renderer::Renderer::dispatch`].
    pub(super) fn dispatch_compute(
        &mut self,
        engine: &EngineCompute,
        ctx: &mut ExecCtx,
    ) -> Result<bool> {
        self.enter_space(ctx.as_id);
        // A frame on the rasterizer reads guest memory, which a held buffer would leave stale.
        if self.give_up() || self.software_frame || self.only.is_some() || self.software_compute {
            return Ok(false);
        }
        self.evict_written(ctx);
        let Ok(launch) = software::read_launch(engine, ctx) else {
            return Ok(false);
        };
        if !launch.qmd.is_empty() {
            if let Err(why) = self.encode_dispatch(&launch, ctx) {
                self.release_scratch();
                *self.compute_refusals.entry(why).or_default() += 1;
                return Ok(false);
            }
            self.release_scratch();
            self.compute_dispatched += 1;
        }
        // Commands are queued in order, so whatever waits on the release comes after the work.
        software::release(&launch.qmd, ctx)?;
        Ok(true)
    }

    fn encode_dispatch(
        &mut self,
        launch: &Launch,
        ctx: &ExecCtx,
    ) -> std::result::Result<(), String> {
        let qmd = &launch.qmd;
        let limits = self.device_limits.clone();
        let [x, y, z] = qmd.cta_threads;
        if x > limits.max_compute_workgroup_size_x
            || y > limits.max_compute_workgroup_size_y
            || z > limits.max_compute_workgroup_size_z
            || qmd.threads_per_cta() > limits.max_compute_invocations_per_workgroup
        {
            return Err(format!(
                "a CTA of {:?} threads, past the device's workgroup",
                qmd.cta_threads
            ));
        }
        if qmd
            .cta_raster
            .iter()
            .any(|&n| n > limits.max_compute_workgroups_per_dimension)
        {
            return Err(format!(
                "a grid of {:?} CTAs, past the device's",
                qmd.cta_raster
            ));
        }

        let bank_of = |bank: u8| {
            qmd.constant_buffers
                .get(usize::from(bank))
                .copied()
                .flatten()
                .map(|cb| (cb.addr, cb.size))
        };
        let (translation, serial) = timed!(
            self,
            translate,
            self.translate_program((launch.program_addr, None), &bank_of, ctx)
        )?;
        let mut layout = Layout::of(&translation, Stage::Compute);
        layout.compute = Some(ComputeShape {
            block: qmd.cta_threads,
            shared_size: qmd.shared_memory_size,
            local_size: qmd.local_memory_size,
        });
        if !layout.textures.is_empty() {
            return Err("samples a texture".into());
        }
        // A browser allows as few as eight; past the limit the layout is rejected.
        let buffers = usize::from(!layout.const_banks.is_empty()) + layout.globals.len();
        let storage_limit = limits.max_storage_buffers_per_shader_stage as usize;
        if buffers > storage_limit {
            return Err(format!(
                "{buffers} storage buffers, past the device's {storage_limit}"
            ));
        }

        // The banks it reads, as far as it reads them.
        let mut banks = Vec::new();
        for (&bank, &extent) in layout.const_banks.iter().zip(&layout.const_extents) {
            let Some((addr, size)) = bank_of(bank) else {
                continue;
            };
            let size = size.min(extent);
            if size == 0 {
                continue;
            }
            let mut bytes = vec![0u8; size as usize];
            ctx.vmm
                .read_into(ctx.mem, addr, &mut bytes)
                .map_err(|e| format!("constant bank {bank}: {e:?}"))?;
            banks.push(ConstantUpload {
                // A compute program has no draw stage; the packing ignores it.
                stage: ShaderStage::VertexB,
                bank: u32::from(bank),
                bytes,
            });
        }
        let bank_bytes = |bank: u8| {
            banks
                .iter()
                .find(|c| c.bank == u32::from(bank))
                .map(|c| c.bytes.as_slice())
        };
        let globals = self.globals_through(&layout, None, &bank_bytes, ctx)?;
        // What it reads from guest memory must be current there: not a surface drawn on the
        // device, and for a buffer it only reads, nothing a launch wrote that has not landed.
        let span = |global: &GlobalUpload| global.addr..global.addr + global.len;
        let uploaded: Vec<_> = globals.iter().filter(|g| g.addr != 0).map(span).collect();
        let only_read: Vec<_> = globals
            .iter()
            .filter(|g| g.addr != 0 && !g.writes)
            .map(span)
            .collect();
        if !self.touches_no_surface(&uploaded) || !self.no_global_in(&only_read) {
            return Err("reads memory held on the device".into());
        }

        let (module_key, module) = timed!(
            self,
            modules,
            self.built_module("compute", (&translation, serial), Stage::Compute, &layout)
        )?;

        let entries = compute_entries(&layout, &globals);
        let group_layout = match self.group_layouts.get(&entries) {
            Some(layout) => layout.clone(),
            None => {
                let created =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("compute"),
                            entries: &entries,
                        });
                self.group_layouts.insert(entries.clone(), created.clone());
                created
            }
        };
        let key: ComputeKey = (module_key, entries);
        let pipeline = match self.compute_pipelines.get(&key) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let pipeline_layout =
                    self.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("compute"),
                            bind_group_layouts: &[Some(&group_layout)],
                            immediate_size: 0,
                        });
                let pipeline = timed!(
                    self,
                    pipeline,
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("compute"),
                            layout: Some(&pipeline_layout),
                            module: &module,
                            entry_point: Some("cs_main"),
                            compilation_options: Default::default(),
                            cache: None,
                        })
                );
                self.compute_pipelines.insert(key, pipeline.clone());
                pipeline
            }
        };

        let mut buffers: Vec<(u32, wgpu::Buffer)> = Vec::new();
        if !layout.const_banks.is_empty() {
            let packed = pack_banks(banks.iter());
            buffers.push((
                wgsl::CONSTANTS_BINDING,
                self.buffer("constants", &packed, wgpu::BufferUsages::STORAGE),
            ));
        }
        for global in &globals {
            let buffer = match &global.held {
                Some(held) => held.clone(),
                None => self.buffer("global", &global.bytes, wgpu::BufferUsages::STORAGE),
            };
            buffers.push((wgsl::GLOBAL_BINDING + global.slot, buffer));
        }
        let bindings: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute"),
            layout: &group_layout,
            entries: &bindings,
        });

        self.end_pass();
        let mut encoder = self.batch.take().unwrap_or_else(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("draws"),
                })
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            let [gx, gy, gz] = qmd.cta_raster;
            pass.dispatch_workgroups(gx, gy, gz);
        }
        self.batch = Some(encoder);
        self.batched_draws += 1;
        if self.batched_draws >= crate::MAX_BATCHED_DRAWS {
            self.submit_batch();
        }
        Ok(())
    }
}

impl Gpu {
    /// Buffers shaders wrote on the device that guest memory does not have yet, oldest first:
    /// those on their way back, then those still held.
    fn written_on_device(&self) -> Vec<(std::ops::Range<u64>, &wgpu::Buffer)> {
        let landing = self
            .space
            .pending
            .iter()
            .filter_map(|pending| match &pending.what {
                Readback::Global {
                    addr,
                    original,
                    source,
                } => Some((*addr..*addr + original.len() as u64, source)),
                Readback::Surface { .. } => None,
            });
        let held = self
            .space
            .globals
            .iter()
            .map(|(&addr, held)| (addr..addr + held.original.len() as u64, &held.buffer));
        landing.chain(held).collect()
    }

    /// The device copies that put what compute launches wrote over a vertex feed's bytes.
    pub(super) fn vertex_patches(&self, feed: &Feed) -> std::result::Result<Vec<Patch>, String> {
        if self.space.globals.is_empty() && self.space.pending.is_empty() {
            return Ok(Vec::new());
        }
        let written = self.written_on_device();
        let overlaps =
            |a: &std::ops::Range<u64>, b: &std::ops::Range<u64>| a.start < b.end && b.start < a.end;
        if !written.iter().any(|(span, _)| overlaps(span, &feed.read)) {
            return Ok(Vec::new());
        }
        let Some((start, valid)) = feed.origin else {
            return Err("rearranges a vertex array a compute launch wrote on the device".into());
        };
        let wanted = start..start + valid;
        // Two held buffers over the same bytes leave no telling which wrote them last.
        let held = written.len() - self.space.globals.len();
        let current: Vec<_> = written[held..]
            .iter()
            .filter(|(span, _)| overlaps(span, &wanted))
            .collect();
        if current
            .iter()
            .enumerate()
            .any(|(i, (a, _))| current[i + 1..].iter().any(|(b, _)| overlaps(a, b)))
        {
            return Err("reads vertices two held buffers both cover".into());
        }
        let mut patches = Vec::new();
        for (span, source) in &written {
            let (from, to) = (span.start.max(wanted.start), span.end.min(wanted.end));
            if from >= to {
                continue;
            }
            let patch = Patch {
                source: (*source).clone(),
                from: from - span.start,
                to: from - start,
                len: to - from,
            };
            if !(patch.from | patch.to | patch.len).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                return Err("reads vertices a compute launch wrote off a word boundary".into());
            }
            patches.push(patch);
        }
        Ok(patches)
    }
}

/// The bind group layout a compute module declares: its banks, then each global.
fn compute_entries(layout: &Layout, globals: &[GlobalUpload]) -> Vec<wgpu::BindGroupLayoutEntry> {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = Vec::new();
    if !layout.const_banks.is_empty() {
        entries.push(storage(wgsl::CONSTANTS_BINDING, true));
    }
    for global in globals {
        entries.push(storage(wgsl::GLOBAL_BINDING + global.slot, !global.writes));
    }
    entries
}
