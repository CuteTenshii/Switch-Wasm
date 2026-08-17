//! nvdrv: the Horizon service in front of the GPU.
//!
//! The guest opens device nodes (`/dev/nvmap`, `/dev/nvhost-as-gpu`,
//! `/dev/nvhost-gpu`, …) and drives them with Linux-style ioctls whose numbers
//! encode direction, type, command and argument size. The argument is a single
//! in/out struct, so each handler here decodes the struct the guest sent,
//! performs the operation on the [`Gpu`], and writes the results back into the
//! same buffer.
//!
//! Struct layouts and ioctl numbers match libnx's `nvidia/ioctl` sources,
//! which is what real homebrew is compiled against.

use crate::gpu::syncpt::NvFence;
use crate::gpu::vmm::{BIG_REGION_END, SMALL_PAGE_SIZE, SMALL_REGION_BASE, SMALL_REGION_END};
use crate::gpu::Gpu;
use crate::mem::Memory;
use crate::{Error, Result};
use std::collections::HashMap;

/// `NvError` values the driver returns in the ioctl reply.
pub const NV_OK: u32 = 0;
pub const NV_NOT_IMPLEMENTED: u32 = 1;
pub const NV_NOT_SUPPORTED: u32 = 2;
pub const NV_BAD_PARAMETER: u32 = 4;
pub const NV_INVALID_STATE: u32 = 8;

/// ioctl type ("magic") bytes.
const TYPE_NVHOST: u32 = 0x00;
const TYPE_NVMAP: u32 = 0x01;
const TYPE_AS_GPU: u32 = 0x41;
const TYPE_CTRL_GPU: u32 = 0x47;
const TYPE_CHANNEL: u32 = 0x48;

/// What an open file descriptor refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvFile {
    NvMap,
    NvHostCtrl,
    NvHostCtrlGpu,
    /// `/dev/nvhost-as-gpu`, owning one GPU address space.
    AddressSpace { as_id: u32 },
    /// `/dev/nvhost-gpu`, owning one channel.
    Channel { channel_id: u32 },
    /// A node we recognise but do not model (nvdec, vic, …).
    Unsupported { path: String },
}

#[derive(Debug)]
pub struct NvDrv {
    pub gpu: Gpu,
    files: HashMap<u32, NvFile>,
    next_fd: u32,
    /// Transfer-memory size the guest handed us in `Initialize`.
    pub transfer_mem_size: u32,
    pub initialized: bool,
}

impl Default for NvDrv {
    fn default() -> Self {
        NvDrv::new()
    }
}

impl NvDrv {
    pub fn new() -> NvDrv {
        NvDrv {
            gpu: Gpu::new(),
            files: HashMap::new(),
            next_fd: 1,
            transfer_mem_size: 0,
            initialized: false,
        }
    }

    pub fn file(&self, fd: u32) -> Option<&NvFile> {
        self.files.get(&fd)
    }

    /// `nvOpen`. Returns `(fd, error)`.
    pub fn open(&mut self, path: &str) -> Result<(u32, u32)> {
        let file = match path {
            "/dev/nvmap" => NvFile::NvMap,
            "/dev/nvhost-ctrl" => NvFile::NvHostCtrl,
            "/dev/nvhost-ctrl-gpu" => NvFile::NvHostCtrlGpu,
            "/dev/nvhost-as-gpu" => NvFile::AddressSpace { as_id: self.gpu.create_address_space() },
            "/dev/nvhost-gpu" => NvFile::Channel { channel_id: self.gpu.create_channel()? },
            other => NvFile::Unsupported { path: other.to_owned() },
        };
        let unsupported = matches!(file, NvFile::Unsupported { .. });
        let fd = self.next_fd;
        self.next_fd += 1;
        self.files.insert(fd, file);
        if self.gpu.trace {
            eprintln!("[nv] open {} -> fd {}{}", path, fd, if unsupported { " (unsupported)" } else { "" });
        }
        Ok((fd, if unsupported { NV_NOT_SUPPORTED } else { NV_OK }))
    }

    /// `nvClose`.
    pub fn close(&mut self, fd: u32) -> u32 {
        match self.files.remove(&fd) {
            Some(NvFile::AddressSpace { as_id }) => {
                self.gpu.address_spaces.remove(&as_id);
                NV_OK
            }
            Some(NvFile::Channel { channel_id }) => {
                if let Some(chan) = self.gpu.channels.remove(&channel_id) {
                    self.gpu.host1x.release(chan.syncpt);
                }
                NV_OK
            }
            Some(_) => NV_OK,
            None => NV_BAD_PARAMETER,
        }
    }

    /// `nvIoctl` / `nvIoctl2` / `nvIoctl3`. `data` is the in/out argument
    /// struct, resized by the caller to the ioctl's declared size; `inline_in`
    /// carries `nvIoctl2`'s extra input buffer. Returns the `NvError` the
    /// guest sees, or a hard [`Error`] when the GPU model itself faults.
    pub fn ioctl(
        &mut self,
        mem: &mut Memory,
        fd: u32,
        request: u32,
        data: &mut [u8],
        inline_in: &[u8],
    ) -> Result<u32> {
        let ioc_type = (request >> 8) & 0xFF;
        let nr = request & 0xFF;
        let file = match self.files.get(&fd) {
            Some(file) => file.clone(),
            None => return Ok(NV_BAD_PARAMETER),
        };
        if self.gpu.trace {
            eprintln!(
                "[nv] ioctl fd={} {:?} type={:#04x} nr={:#04x} size={} ({} bytes in)",
                fd,
                file,
                ioc_type,
                nr,
                (request >> 16) & 0x3FFF,
                data.len()
            );
        }
        match (&file, ioc_type) {
            (NvFile::NvMap, TYPE_NVMAP) => self.nvmap_ioctl(nr, data),
            (NvFile::NvHostCtrl, TYPE_NVHOST) => self.nvhost_ctrl_ioctl(nr, data),
            (NvFile::NvHostCtrlGpu, TYPE_CTRL_GPU) => self.ctrl_gpu_ioctl(nr, data),
            (NvFile::AddressSpace { as_id }, TYPE_AS_GPU) => self.as_gpu_ioctl(*as_id, nr, data),
            (NvFile::Channel { channel_id }, _) => {
                self.channel_ioctl(mem, *channel_id, ioc_type, nr, data, inline_in)
            }
            (NvFile::Unsupported { .. }, _) => Ok(NV_NOT_SUPPORTED),
            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }

    /// `nvQueryEvent`: the guest wants a kernel event it can wait on. Work is
    /// retired synchronously here, so the event is always already signalled.
    pub fn query_event(&mut self, fd: u32, event_id: u32) -> u32 {
        if self.files.contains_key(&fd) {
            let _ = self.gpu.host1x.register_event(event_id.min(63));
            NV_OK
        } else {
            NV_BAD_PARAMETER
        }
    }

    // -- /dev/nvmap ------------------------------------------------------

    fn nvmap_ioctl(&mut self, nr: u32, data: &mut [u8]) -> Result<u32> {
        match nr {
            // Create { in u32 size; out u32 handle; }
            0x01 => {
                let size = read_u32(data, 0);
                let handle = self.gpu.nvmap.create(size);
                write_u32(data, 4, handle);
                Ok(NV_OK)
            }
            // FromId { in u32 id; out u32 handle; }
            0x03 => match self.gpu.nvmap.from_id(read_u32(data, 0)) {
                Some(handle) => {
                    write_u32(data, 4, handle);
                    Ok(NV_OK)
                }
                None => Ok(NV_BAD_PARAMETER),
            },
            // Alloc { handle, heapmask, flags, align, u8 kind, pad[7], u64 addr }
            0x04 => {
                let handle = read_u32(data, 0);
                let heap_mask = read_u32(data, 4);
                let flags = read_u32(data, 8);
                let align = read_u32(data, 0x0C);
                let kind = data.get(0x10).copied().unwrap_or(0);
                let addr = read_u64(data, 0x18);
                if addr > u32::MAX as u64 {
                    return Err(Error::Gpu(format!(
                        "nvmap: buffer address {:#x} is outside the emulated 32-bit space",
                        addr
                    )));
                }
                self.gpu.nvmap.alloc(handle, heap_mask, flags, align, kind, addr as u32)?;
                Ok(NV_OK)
            }
            // Free { in handle; pad; out u64 refcount; out u32 size; out u32 flags }
            0x05 => {
                let handle = read_u32(data, 0);
                match self.gpu.nvmap.free(handle) {
                    Some(h) => {
                        write_u64(data, 8, h.refcount as u64);
                        write_u32(data, 0x10, h.size);
                        write_u32(data, 0x14, if h.refcount > 0 { 1 } else { 0 });
                        Ok(NV_OK)
                    }
                    None => Ok(NV_BAD_PARAMETER),
                }
            }
            // Param { in handle; in param; out result }
            0x09 => {
                let handle = read_u32(data, 0);
                let param = read_u32(data, 4);
                match self.gpu.nvmap.param(handle, param) {
                    Ok(value) => {
                        write_u32(data, 8, value);
                        Ok(NV_OK)
                    }
                    Err(_) => Ok(NV_BAD_PARAMETER),
                }
            }
            // GetId { out id; in handle }
            0x0E => {
                let handle = read_u32(data, 4);
                match self.gpu.nvmap.get(handle) {
                    Some(h) => {
                        write_u32(data, 0, h.id);
                        Ok(NV_OK)
                    }
                    None => Ok(NV_BAD_PARAMETER),
                }
            }
            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }

    // -- /dev/nvhost-ctrl ------------------------------------------------

    fn nvhost_ctrl_ioctl(&mut self, nr: u32, data: &mut [u8]) -> Result<u32> {
        match nr {
            // SyncptRead { in id; out value }
            0x14 => {
                let value = self.gpu.host1x.read(read_u32(data, 0))?;
                write_u32(data, 4, value);
                Ok(NV_OK)
            }
            // SyncptIncr { in id }
            0x15 => {
                self.gpu.host1x.increment(read_u32(data, 0))?;
                Ok(NV_OK)
            }
            // SyncptWait / SyncptWaitEx: submissions retire inside their
            // ioctl, so a wait is only ever asked about work already done.
            0x16 | 0x19 => {
                let id = read_u32(data, 0);
                let threshold = read_u32(data, 4);
                if self.gpu.host1x.is_expired(id, threshold)? {
                    Ok(NV_OK)
                } else {
                    // Nothing else can advance it, so report the timeout the
                    // guest would eventually see rather than hanging.
                    Ok(NV_INVALID_STATE)
                }
            }
            // SyncptReadMax { in id; out value }
            0x1A => {
                let value = self.gpu.host1x.read_max(read_u32(data, 0))?;
                write_u32(data, 4, value);
                Ok(NV_OK)
            }
            // EventSignal / EventRegister / EventUnregister { in event_id }
            0x1C => Ok(NV_OK),
            0x1F => {
                self.gpu.host1x.register_event(read_u32(data, 0).min(63))?;
                Ok(NV_OK)
            }
            0x20 => {
                self.gpu.host1x.unregister_event(read_u32(data, 0).min(63))?;
                Ok(NV_OK)
            }
            // EventWait { in syncpt_id, threshold, timeout; inout value }
            0x1D => {
                let id = read_u32(data, 0);
                let threshold = read_u32(data, 4);
                let slot = read_u32(data, 0x0C);
                if let Some(event) = self.gpu.host1x.events.get_mut(slot.min(63) as usize) {
                    event.fence = NvFence { id, value: threshold };
                    event.signalled = true;
                }
                write_u32(data, 0x0C, slot);
                Ok(NV_OK)
            }
            // EventWaitAsync { in syncpt_id, threshold, timeout, event_id }
            0x1E => Ok(NV_OK),
            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }

    // -- /dev/nvhost-ctrl-gpu --------------------------------------------

    fn ctrl_gpu_ioctl(&mut self, nr: u32, data: &mut [u8]) -> Result<u32> {
        match nr {
            // ZCullGetCtxSize { out u32 }
            0x01 => {
                write_u32(data, 0, 0x1000);
                Ok(NV_OK)
            }
            // ZCullGetInfo { out nvioctl_zcull_info }
            0x02 => {
                for (i, value) in [
                    0x20u32, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10,
                ]
                .iter()
                .enumerate()
                {
                    write_u32(data, i * 4, *value);
                }
                Ok(NV_OK)
            }
            // ZbcSetTable / ZbcQueryTable: the zero-bandwidth-clear cache is
            // a pure optimisation; accepting the table is enough.
            0x03 | 0x04 => Ok(NV_OK),
            // GetCharacteristics { in u64 buf_size, buf_addr; out gc }
            0x05 => {
                write_u64(data, 0, GPU_CHARACTERISTICS.len() as u64);
                for (i, byte) in GPU_CHARACTERISTICS.iter().enumerate() {
                    if let Some(slot) = data.get_mut(0x10 + i) {
                        *slot = *byte;
                    }
                }
                Ok(NV_OK)
            }
            // GetTpcMasks { in bufsize, pad, bufaddr; out u8[8] }
            0x06 => {
                write_u32(data, 0x10, 0x3); // two TPCs in GPC 0
                write_u32(data, 0x14, 0);
                Ok(NV_OK)
            }
            // ZbcGetActiveSlotMask { out slot, mask }
            0x14 => {
                write_u32(data, 0, 0x07);
                write_u32(data, 4, 0x01);
                Ok(NV_OK)
            }
            // GetGpuTime { out u64 timestamp, u64 reserved }
            0x1C => {
                write_u64(data, 0, self.gpu.stats.submissions * 1_000_000);
                write_u64(data, 8, 0);
                Ok(NV_OK)
            }
            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }

    // -- /dev/nvhost-as-gpu ----------------------------------------------

    fn as_gpu_ioctl(&mut self, as_id: u32, nr: u32, data: &mut [u8]) -> Result<u32> {
        match nr {
            // BindChannel { in u32 fd }
            0x01 => {
                let channel_fd = read_u32(data, 0);
                match self.files.get(&channel_fd) {
                    Some(NvFile::Channel { channel_id }) => {
                        let channel_id = *channel_id;
                        self.gpu.channel_mut(channel_id)?.as_id = Some(as_id);
                        Ok(NV_OK)
                    }
                    _ => Ok(NV_BAD_PARAMETER),
                }
            }
            // AllocSpace { in pages, page_size, flags; pad; inout u64 offset }
            0x02 => {
                let pages = read_u32(data, 0);
                let page_size = read_u32(data, 4);
                let flags = read_u32(data, 8);
                let requested = read_u64(data, 0x10);
                let offset =
                    self.gpu.address_space_mut(as_id)?.alloc_space(pages, page_size, flags, requested)?;
                write_u64(data, 0x10, offset);
                Ok(NV_OK)
            }
            // FreeSpace { in u64 offset; in pages, page_size }
            0x03 => {
                let offset = read_u64(data, 0);
                let pages = read_u32(data, 8);
                let page_size = read_u32(data, 0x0C);
                self.gpu.address_space_mut(as_id)?.free_space(offset, pages, page_size)?;
                Ok(NV_OK)
            }
            // UnmapBuffer { in u64 offset }
            0x05 => {
                let offset = read_u64(data, 0);
                self.gpu.address_space_mut(as_id)?.unmap(offset)?;
                Ok(NV_OK)
            }
            // MapBufferEx { flags, kind, nvmap_handle, page_size,
            //               u64 buffer_offset, mapping_size; inout u64 offset }
            0x06 => {
                let flags = read_u32(data, 0);
                let kind = read_u32(data, 4);
                let nvmap_handle = read_u32(data, 8);
                let page_size = read_u32(data, 0x0C);
                let buffer_offset = read_u64(data, 0x10);
                let mapping_size = read_u64(data, 0x18);
                let requested = read_u64(data, 0x20);

                let handle = match self.gpu.nvmap.get(nvmap_handle) {
                    Some(h) if h.allocated => *h,
                    Some(_) => return Ok(NV_INVALID_STATE),
                    None => return Ok(NV_BAD_PARAMETER),
                };
                let size = if mapping_size != 0 { mapping_size } else { handle.size as u64 };
                let kind = if kind == u32::MAX { handle.kind } else { kind as u8 };
                let page_size = if page_size == 0 { SMALL_PAGE_SIZE } else { page_size as u64 };
                let cpu_addr = handle.cpu_addr.wrapping_add(buffer_offset as u32);
                let offset = self.gpu.address_space_mut(as_id)?.map(
                    cpu_addr,
                    size,
                    nvmap_handle,
                    kind,
                    page_size,
                    flags,
                    requested,
                )?;
                write_u32(data, 0x0C, page_size as u32);
                write_u64(data, 0x20, offset);
                Ok(NV_OK)
            }
            // GetVARegions { u64 not_used; inout bufsize; pad; out regions[2] }
            0x08 => {
                write_u32(data, 8, 2 * 24);
                let regions = [
                    (SMALL_REGION_BASE, SMALL_PAGE_SIZE, (SMALL_REGION_END - SMALL_REGION_BASE) / SMALL_PAGE_SIZE),
                    (
                        SMALL_REGION_END,
                        self.gpu.address_space_mut(as_id)?.big_page_size,
                        (BIG_REGION_END - SMALL_REGION_END)
                            / self.gpu.address_spaces[&as_id].big_page_size.max(1),
                    ),
                ];
                for (i, (offset, page_size, pages)) in regions.iter().enumerate() {
                    let at = 0x10 + i * 24;
                    write_u64(data, at, *offset);
                    write_u32(data, at + 8, *page_size as u32);
                    write_u32(data, at + 12, 0);
                    write_u64(data, at + 16, *pages);
                }
                Ok(NV_OK)
            }
            // InitializeEx { flags, as_fd, big_page_size, reserved, unk0..2 }
            0x09 => {
                let big_page_size = read_u32(data, 8);
                if big_page_size != 0 {
                    self.gpu.address_space_mut(as_id)?.big_page_size = big_page_size as u64;
                }
                Ok(NV_OK)
            }
            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }

    // -- /dev/nvhost-gpu -------------------------------------------------

    fn channel_ioctl(
        &mut self,
        mem: &mut Memory,
        channel_id: u32,
        ioc_type: u32,
        nr: u32,
        data: &mut [u8],
        inline_in: &[u8],
    ) -> Result<u32> {
        match (ioc_type, nr) {
            // SetNvmapFd / SetTimeout / ZCullBind / SetErrorNotifier /
            // SetPriority / SetUserData: bookkeeping with no visible effect on
            // the model.
            (TYPE_CHANNEL, 0x01) | (TYPE_CHANNEL, 0x03) | (TYPE_CHANNEL, 0x0B)
            | (TYPE_CHANNEL, 0x0C) | (TYPE_CHANNEL, 0x0D) | (TYPE_CTRL_GPU, 0x14)
            | (TYPE_NVHOST, 0x07) | (TYPE_NVHOST, 0x08) => Ok(NV_OK),

            // SubmitGpfifo { u64 gpfifo; num_entries; flags; fence; entries[] }
            (TYPE_CHANNEL, 0x08) => {
                let num_entries = read_u32(data, 8);
                let mut entries = Vec::with_capacity(num_entries as usize);
                for i in 0..num_entries as usize {
                    entries.push(read_u64(data, 0x18 + i * 8));
                }
                let fence = self.gpu.submit(channel_id, mem, &entries, 1)?;
                write_u32(data, 0x10, fence.id);
                write_u32(data, 0x14, fence.value);
                Ok(NV_OK)
            }

            // KickoffPb: same as SubmitGpfifo, entries in the inline buffer.
            (TYPE_CHANNEL, 0x1B) => {
                let num_entries = read_u32(data, 8) as usize;
                let mut entries = Vec::with_capacity(num_entries);
                for i in 0..num_entries {
                    entries.push(read_u64(inline_in, i * 8));
                }
                let fence = self.gpu.submit(channel_id, mem, &entries, 1)?;
                write_u32(data, 0x10, fence.id);
                write_u32(data, 0x14, fence.value);
                Ok(NV_OK)
            }

            // AllocObjCtx { class_num; flags; out u64 obj_id }
            (TYPE_CHANNEL, 0x09) => {
                let class_num = read_u32(data, 0);
                write_u64(data, 8, class_num as u64);
                Ok(NV_OK)
            }

            // AllocGpfifoEx2 { num_entries; flags; unk0; out fence; unk1..3 }
            (TYPE_CHANNEL, 0x1A) => {
                let num_entries = read_u32(data, 0);
                let chan = self.gpu.channel_mut(channel_id)?;
                chan.gpfifo_entries = num_entries;
                let syncpt = chan.syncpt;
                let value = self.gpu.host1x.read(syncpt)?;
                write_u32(data, 0x0C, syncpt);
                write_u32(data, 0x10, value);
                Ok(NV_OK)
            }

            // GetErrorInfo / GetErrorNotification: no errors to report.
            (TYPE_CHANNEL, 0x16) | (TYPE_CHANNEL, 0x17) => {
                for byte in data.iter_mut() {
                    *byte = 0;
                }
                Ok(NV_OK)
            }

            // GetSyncpt { in module id; out syncpt }
            (TYPE_NVHOST, 0x02) => {
                let syncpt = self.gpu.channel_mut(channel_id)?.syncpt;
                write_u32(data, 4, syncpt);
                Ok(NV_OK)
            }

            // GetModuleClockRate: report the GM20B's boost clock in kHz.
            (TYPE_NVHOST, 0x14) | (TYPE_NVHOST, 0x23) => {
                write_u32(data, 0, 768_000);
                Ok(NV_OK)
            }

            // MapCommandBuffer / UnmapCommandBuffer: host1x-class submission,
            // which the GPU channel path does not use.
            (TYPE_NVHOST, 0x09) | (TYPE_NVHOST, 0x0A) => Ok(NV_OK),

            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }
}

/// `nvioctl_gpu_characteristics` for the GM20B, as the Switch reports it.
static GPU_CHARACTERISTICS: [u8; 0xA0] = {
    let mut out = [0u8; 0xA0];
    // Written as a const fn would be, but `const` loops over a table are
    // clearer here: each entry is (byte offset, value, width in bytes).
    macro_rules! put {
        ($out:ident, $off:expr, $value:expr, 4) => {{
            let v: u32 = $value;
            $out[$off] = v as u8;
            $out[$off + 1] = (v >> 8) as u8;
            $out[$off + 2] = (v >> 16) as u8;
            $out[$off + 3] = (v >> 24) as u8;
        }};
        ($out:ident, $off:expr, $value:expr, 8) => {{
            let v: u64 = $value;
            let mut i = 0;
            while i < 8 {
                $out[$off + i] = (v >> (8 * i)) as u8;
                i += 1;
            }
        }};
    }
    put!(out, 0x00, 0x120, 4); // arch: NVGPU_GPU_ARCH_GM200
    put!(out, 0x04, 0x0B, 4); // impl: GM20B
    put!(out, 0x08, 0xA1, 4); // rev A1
    put!(out, 0x0C, 1, 4); // num_gpc
    put!(out, 0x10, 0x40000, 8); // L2 cache size
    put!(out, 0x18, 0, 8); // on-board video memory (none: it is system RAM)
    put!(out, 0x20, 2, 4); // num_tpc_per_gpc
    put!(out, 0x24, 0x20, 4); // bus type: AXI
    put!(out, 0x28, 0x20000, 4); // big_page_size
    put!(out, 0x2C, 0x20000, 4); // compression_page_size
    put!(out, 0x30, 0x1B, 4); // pde_coverage_bit_count
    put!(out, 0x34, 0x30000, 4); // available_big_page_sizes
    put!(out, 0x38, 1, 4); // gpc_mask
    put!(out, 0x3C, 0x503, 4); // sm_arch_sm_version
    put!(out, 0x40, 0x503, 4); // sm_arch_spa_version
    put!(out, 0x44, 0x80, 4); // sm_arch_warp_count
    put!(out, 0x48, 0x28, 4); // gpu_va_bit_count (40)
    put!(out, 0x4C, 0, 4); // reserved
    put!(out, 0x50, 0x55, 8); // flags
    put!(out, 0x58, 0x902D, 4); // twod_class
    put!(out, 0x5C, 0xB197, 4); // threed_class
    put!(out, 0x60, 0xB1C0, 4); // compute_class
    put!(out, 0x64, 0xB06F, 4); // gpfifo_class
    put!(out, 0x68, 0xA140, 4); // inline_to_memory_class
    put!(out, 0x6C, 0xB0B5, 4); // dma_copy_class
    put!(out, 0x70, 1, 4); // max_fbps_count
    put!(out, 0x74, 0, 4); // fbp_en_mask
    put!(out, 0x78, 2, 4); // max_ltc_per_fbp
    put!(out, 0x7C, 1, 4); // max_lts_per_ltc
    put!(out, 0x80, 0, 4); // max_tex_per_tpc
    put!(out, 0x84, 1, 4); // max_gpc_count
    put!(out, 0x88, 0x21D70, 4); // rop_l2_en_mask_0
    put!(out, 0x8C, 0, 4); // rop_l2_en_mask_1
    put!(out, 0x90, 0x62_3032_6D67, 8); // chipname: "gm20b"
    put!(out, 0x98, 0, 8); // gr_compbit_store_base_hw
    out
};

fn read_u32(data: &[u8], at: usize) -> u32 {
    let mut v = 0u32;
    for i in 0..4 {
        v |= (data.get(at + i).copied().unwrap_or(0) as u32) << (8 * i);
    }
    v
}

fn read_u64(data: &[u8], at: usize) -> u64 {
    (read_u32(data, at) as u64) | ((read_u32(data, at + 4) as u64) << 32)
}

fn write_u32(data: &mut [u8], at: usize, value: u32) {
    for i in 0..4 {
        if let Some(slot) = data.get_mut(at + i) {
            *slot = (value >> (8 * i)) as u8;
        }
    }
}

fn write_u64(data: &mut [u8], at: usize, value: u64) {
    write_u32(data, at, value as u32);
    write_u32(data, at + 4, (value >> 32) as u32);
}

/// Build the ioctl number the way libnx's `_NV_IOC` macros do, for tests and
/// for callers that need to synthesize a request.
pub fn make_ioctl(dir: u32, ioc_type: u32, nr: u32, size: u32) -> u32 {
    (dir << 30) | ((size & 0x3FFF) << 16) | ((ioc_type & 0xFF) << 8) | (nr & 0xFF)
}

/// Size in bytes of an ioctl's argument struct.
pub fn ioctl_size(request: u32) -> u32 {
    (request >> 16) & 0x3FFF
}

/// Direction bits: 1 = write (guest → driver), 2 = read (driver → guest).
pub fn ioctl_direction(request: u32) -> u32 {
    (request >> 30) & 3
}

#[cfg(test)]
mod tests {
    use super::*;

    const IOWR: u32 = 3;

    fn ioctl(drv: &mut NvDrv, mem: &mut Memory, fd: u32, ty: u32, nr: u32, data: &mut [u8]) -> u32 {
        let request = make_ioctl(IOWR, ty, nr, data.len() as u32);
        drv.ioctl(mem, fd, request, data, &[]).unwrap()
    }

    #[test]
    fn open_maps_known_device_nodes() {
        let mut drv = NvDrv::new();
        let (map_fd, err) = drv.open("/dev/nvmap").unwrap();
        assert_eq!(err, NV_OK);
        assert_eq!(drv.file(map_fd), Some(&NvFile::NvMap));
        let (as_fd, err) = drv.open("/dev/nvhost-as-gpu").unwrap();
        assert_eq!(err, NV_OK);
        assert!(matches!(drv.file(as_fd), Some(NvFile::AddressSpace { .. })));
        let (chan_fd, err) = drv.open("/dev/nvhost-gpu").unwrap();
        assert_eq!(err, NV_OK);
        assert!(matches!(drv.file(chan_fd), Some(NvFile::Channel { .. })));
        let (_, err) = drv.open("/dev/nvhost-nvdec").unwrap();
        assert_eq!(err, NV_NOT_SUPPORTED);
    }

    #[test]
    fn nvmap_create_alloc_and_param() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvmap").unwrap();

        let mut create = [0u8; 8];
        write_u32(&mut create, 0, 0x2000);
        assert_eq!(ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x01, &mut create), NV_OK);
        let handle = read_u32(&create, 4);
        assert_ne!(handle, 0);

        let mut alloc = [0u8; 0x20];
        write_u32(&mut alloc, 0, handle);
        write_u32(&mut alloc, 0x0C, 0x1000);
        alloc[0x10] = 0xFE;
        write_u64(&mut alloc, 0x18, 0x3000_0000);
        assert_eq!(ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x04, &mut alloc), NV_OK);

        let mut param = [0u8; 0x0C];
        write_u32(&mut param, 0, handle);
        write_u32(&mut param, 4, 3); // Base
        assert_eq!(ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x09, &mut param), NV_OK);
        assert_eq!(read_u32(&param, 8), 0x3000_0000);
    }

    #[test]
    fn get_characteristics_reports_the_gm20b() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl-gpu").unwrap();
        let mut data = [0u8; 0x10 + 0xA0];
        assert_eq!(ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x05, &mut data), NV_OK);
        assert_eq!(read_u32(&data, 0x10), 0x120); // arch
        assert_eq!(read_u32(&data, 0x10 + 0x5C), 0xB197); // threed_class
        assert_eq!(read_u64(&data, 0x10 + 0x90), 0x62_3032_6D67); // "gm20b"
    }

    #[test]
    fn map_buffer_ex_places_a_buffer_in_the_address_space() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();
        let (map_fd, _) = drv.open("/dev/nvmap").unwrap();
        let (as_fd, _) = drv.open("/dev/nvhost-as-gpu").unwrap();

        let mut create = [0u8; 8];
        write_u32(&mut create, 0, 0x1000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x01, &mut create);
        let handle = read_u32(&create, 4);
        let mut alloc = [0u8; 0x20];
        write_u32(&mut alloc, 0, handle);
        write_u64(&mut alloc, 0x18, 0x3000_0000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x04, &mut alloc);

        let mut map = [0u8; 0x28];
        write_u32(&mut map, 8, handle);
        write_u64(&mut map, 0x18, 0x1000); // mapping size
        assert_eq!(ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x06, &mut map), NV_OK);
        let gpu_va = read_u64(&map, 0x20);
        assert_ne!(gpu_va, 0);

        let NvFile::AddressSpace { as_id } = *drv.file(as_fd).unwrap() else {
            panic!("expected an address-space fd");
        };
        assert_eq!(
            drv.gpu.address_spaces[&as_id].translate(gpu_va),
            Some((0x3000_0000, 0x1000))
        );
    }

    #[test]
    fn bind_channel_links_the_address_space() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (as_fd, _) = drv.open("/dev/nvhost-as-gpu").unwrap();
        let (chan_fd, _) = drv.open("/dev/nvhost-gpu").unwrap();

        let mut bind = [0u8; 4];
        write_u32(&mut bind, 0, chan_fd);
        assert_eq!(ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x01, &mut bind), NV_OK);

        let NvFile::Channel { channel_id } = *drv.file(chan_fd).unwrap() else {
            panic!("expected a channel fd");
        };
        assert!(drv.gpu.channel_mut(channel_id).unwrap().as_id.is_some());
    }

    #[test]
    fn submit_gpfifo_runs_the_pushbuffer_and_returns_a_fence() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x1000).unwrap();

        let (map_fd, _) = drv.open("/dev/nvmap").unwrap();
        let (as_fd, _) = drv.open("/dev/nvhost-as-gpu").unwrap();
        let (chan_fd, _) = drv.open("/dev/nvhost-gpu").unwrap();

        let mut create = [0u8; 8];
        write_u32(&mut create, 0, 0x1000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x01, &mut create);
        let handle = read_u32(&create, 4);
        let mut alloc = [0u8; 0x20];
        write_u32(&mut alloc, 0, handle);
        write_u64(&mut alloc, 0x18, 0x3000_0000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x04, &mut alloc);
        let mut map = [0u8; 0x28];
        write_u32(&mut map, 8, handle);
        write_u64(&mut map, 0x18, 0x1000);
        ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x06, &mut map);
        let gpu_va = read_u64(&map, 0x20);

        let mut bind = [0u8; 4];
        write_u32(&mut bind, 0, chan_fd);
        ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x01, &mut bind);

        // A pushbuffer that binds the 3D class and sets a clear-colour word.
        let words = [
            (0x0000) | (0 << 13) | (1 << 16) | (3 << 29), // NonIncreasing SetObject
            0xB197u32,
            (0x360) | (0 << 13) | (0x55 << 16) | (4 << 29), // Inline write
        ];
        for (i, w) in words.iter().enumerate() {
            mem.write_u32(0x3000_0000 + (i * 4) as u32, *w).unwrap();
        }
        let entry = (gpu_va & 0xFF_FFFF_FFFC) | ((words.len() as u64) << 42);

        let mut submit = vec![0u8; 0x18 + 8];
        write_u32(&mut submit, 8, 1); // num_entries
        write_u64(&mut submit, 0x18, entry);
        assert_eq!(ioctl(&mut drv, &mut mem, chan_fd, TYPE_CHANNEL, 0x08, &mut submit), NV_OK);

        let fence_id = read_u32(&submit, 0x10);
        let fence_value = read_u32(&submit, 0x14);
        assert!(drv.gpu.host1x.is_expired(fence_id, fence_value).unwrap());

        let NvFile::Channel { channel_id } = *drv.file(chan_fd).unwrap() else {
            panic!("expected a channel fd");
        };
        assert_eq!(drv.gpu.channel_mut(channel_id).unwrap().three_d.regs.get(0x360), 0x55);
    }

    #[test]
    fn syncpt_read_reflects_submissions() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (ctrl_fd, _) = drv.open("/dev/nvhost-ctrl").unwrap();
        drv.gpu.host1x.set(9, 3).unwrap();
        let mut data = [0u8; 8];
        write_u32(&mut data, 0, 9);
        assert_eq!(ioctl(&mut drv, &mut mem, ctrl_fd, TYPE_NVHOST, 0x14, &mut data), NV_OK);
        assert_eq!(read_u32(&data, 4), 3);
    }

    #[test]
    fn unknown_ioctl_is_reported_not_implemented() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvmap").unwrap();
        let mut data = [0u8; 8];
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x7F, &mut data),
            NV_NOT_IMPLEMENTED
        );
    }

    #[test]
    fn ioctl_number_roundtrip() {
        let request = make_ioctl(3, 0x01, 0x04, 0x20);
        assert_eq!(ioctl_size(request), 0x20);
        assert_eq!(ioctl_direction(request), 3);
        assert_eq!((request >> 8) & 0xFF, 0x01);
        assert_eq!(request & 0xFF, 0x04);
    }
}
