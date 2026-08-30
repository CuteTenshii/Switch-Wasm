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
use crate::gpu::vmm::{
    BIG_REGION_END, FLAG_FIXED_OFFSET, FLAG_REMAP_SUB_RANGE, SMALL_PAGE_SIZE, SMALL_REGION_BASE,
    SMALL_REGION_END,
};
use crate::gpu::Gpu;
use crate::mem::Memory;
use crate::{Error, Result};
use std::collections::HashMap;

/// `NvError` values the driver returns in the ioctl reply.
pub const NV_OK: u32 = 0;
pub const NV_NOT_IMPLEMENTED: u32 = 1;
pub const NV_NOT_SUPPORTED: u32 = 2;
pub const NV_BAD_PARAMETER: u32 = 4;
pub const NV_INSUFFICIENT_MEMORY: u32 = 6;
pub const NV_INVALID_STATE: u32 = 8;

/// `NVGPU_ZBC_TYPE_*`: which of the two zero-bandwidth-clear tables an entry
/// belongs to. `INVALID` is not an error — a query passes it to ask for the
/// table size and nothing else.
const ZBC_TYPE_INVALID: u32 = 0;
const ZBC_TYPE_COLOR: u32 = 1;
const ZBC_TYPE_DEPTH: u32 = 2;

/// The GM20B's shader units, as `GetCharacteristics` reports them. The TPC
/// mask and the virtual-SM map are derived from these rather than written out
/// again: a driver told two different chips reconciles them by indexing one
/// with the other's count.
const GPU_NUM_GPC: u32 = 1;
const GPU_TPC_PER_GPC: u32 = 2;

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
    AddressSpace {
        as_id: u32,
    },
    /// `/dev/nvhost-gpu`, owning one channel.
    Channel {
        channel_id: u32,
    },
    /// A node we recognise but do not model (nvdec, vic, …).
    Unsupported {
        path: String,
    },
}

/// Slots in each of the driver's two zero-bandwidth-clear tables
/// (`GK20A_ZBC_TABLE_SIZE`).
pub const ZBC_TABLE_SIZE: usize = 16;

/// One zero-bandwidth-clear table entry: a colour (in both the depth-stencil
/// and the L2 encoding) or a depth value that the hardware can encode into a
/// surface's compression bits instead of writing out pixels.
///
/// Nothing here clears that way — the rasterizer writes the pixels — so the
/// table changes no rendering. It is kept because it is *readable*:
/// `ZbcQueryTable` hands back what `ZbcSetTable` put in, and a driver that
/// asks which clear values it already registered and is told "none, ever"
/// registers them again until the table it cannot see fills up.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZbcEntry {
    pub color_ds: [u32; 4],
    pub color_l2: [u32; 4],
    pub depth: u32,
    /// How many times this value has been registered. `ZbcQueryTable` reports
    /// it, which is how a driver tells a slot it owns from one it shares.
    pub ref_cnt: u32,
    pub format: u32,
}

/// One of the two tables, filled from the bottom. An entry is never dropped:
/// the driver interface has no way to remove one, only to add a value it may
/// already hold.
#[derive(Debug, Default)]
pub struct ZbcTable {
    entries: [ZbcEntry; ZBC_TABLE_SIZE],
    used: usize,
}

impl ZbcTable {
    /// Register `entry`, or take another reference to the slot that already
    /// holds this value. False when the table is full.
    fn add(&mut self, entry: ZbcEntry) -> bool {
        let same = |a: &ZbcEntry| {
            a.color_ds == entry.color_ds
                && a.color_l2 == entry.color_l2
                && a.depth == entry.depth
                && a.format == entry.format
        };
        if let Some(index) = self.entries[..self.used].iter().position(same) {
            self.entries[index].ref_cnt += 1;
            return true;
        }
        let Some(slot) = self.entries.get_mut(self.used) else {
            return false;
        };
        *slot = ZbcEntry {
            ref_cnt: 1,
            ..entry
        };
        self.used += 1;
        true
    }

    pub fn get(&self, index: usize) -> Option<&ZbcEntry> {
        self.entries.get(index)
    }

    pub fn used(&self) -> usize {
        self.used
    }
}

#[derive(Debug)]
pub struct NvDrv {
    pub gpu: Gpu,
    files: HashMap<u32, NvFile>,
    next_fd: u32,
    /// Transfer-memory size the guest handed us in `Initialize`.
    pub transfer_mem_size: u32,
    /// The applet the session belongs to, from `SetAruid`. There is one
    /// applet here, so it is recorded and never consulted.
    pub applet_resource_user_id: u64,
    pub initialized: bool,
    /// The two zero-bandwidth-clear tables `/dev/nvhost-ctrl-gpu` keeps,
    /// addressed by `NVGPU_ZBC_TYPE_COLOR` and `..._DEPTH`.
    pub zbc_color: ZbcTable,
    pub zbc_depth: ZbcTable,
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
            applet_resource_user_id: 0,
            initialized: false,
            zbc_color: ZbcTable::default(),
            zbc_depth: ZbcTable::default(),
        }
    }

    pub fn file(&self, fd: u32) -> Option<&NvFile> {
        self.files.get(&fd)
    }

    /// The device node `fd` was opened on, for diagnostics. An ioctl number
    /// means nothing on its own — the same number is a different command on
    /// every node — so anything reporting one has to say which node it was.
    pub fn device_name(&self, fd: u32) -> &str {
        match self.files.get(&fd) {
            Some(NvFile::NvMap) => "/dev/nvmap",
            Some(NvFile::NvHostCtrl) => "/dev/nvhost-ctrl",
            Some(NvFile::NvHostCtrlGpu) => "/dev/nvhost-ctrl-gpu",
            Some(NvFile::AddressSpace { .. }) => "/dev/nvhost-as-gpu",
            Some(NvFile::Channel { .. }) => "/dev/nvhost-gpu",
            Some(NvFile::Unsupported { path }) => path,
            None => "(closed)",
        }
    }

    /// `nvOpen`. Returns `(fd, error)`.
    pub fn open(&mut self, path: &str) -> Result<(u32, u32)> {
        let file = match path {
            "/dev/nvmap" => NvFile::NvMap,
            "/dev/nvhost-ctrl" => NvFile::NvHostCtrl,
            "/dev/nvhost-ctrl-gpu" => NvFile::NvHostCtrlGpu,
            "/dev/nvhost-as-gpu" => NvFile::AddressSpace {
                as_id: self.gpu.create_address_space(),
            },
            "/dev/nvhost-gpu" => NvFile::Channel {
                channel_id: self.gpu.create_channel()?,
            },
            other => NvFile::Unsupported {
                path: other.to_owned(),
            },
        };
        let unsupported = matches!(file, NvFile::Unsupported { .. });
        let fd = self.next_fd;
        self.next_fd += 1;
        self.files.insert(fd, file);
        if self.gpu.trace {
            eprintln!(
                "[nv] open {} -> fd {}{}",
                path,
                fd,
                if unsupported { " (unsupported)" } else { "" }
            );
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
    /// carries `nvIoctl2`'s extra input buffer and `inline_out` receives
    /// `nvIoctl3`'s extra *output* buffer. Returns the `NvError` the guest
    /// sees, or a hard [`Error`] when the GPU model itself faults.
    ///
    /// The inline output is not decoration. An ioctl whose argument struct
    /// carries a `{ buf_size, buf_addr }` pair returns its payload *through*
    /// that pair, and `nvIoctl3` is how a caller asks for it to come back in a
    /// second buffer rather than inline. `libnx` uses `nvIoctl` and reads the
    /// payload from `data`; `nnSdk` uses `nvIoctl3` and reads it from here, so
    /// leaving this empty handed a retail title a **zeroed** GPU
    /// characteristics struct — it closed `/dev/nvhost-ctrl-gpu` and returned
    /// a null device, which its caller then dereferenced.
    pub fn ioctl(
        &mut self,
        mem: &mut Memory,
        fd: u32,
        request: u32,
        data: &mut [u8],
        inline_in: &[u8],
        inline_out: &mut Vec<u8>,
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
        let outcome = match (&file, ioc_type) {
            (NvFile::NvMap, TYPE_NVMAP) => self.nvmap_ioctl(nr, data),
            (NvFile::NvHostCtrl, TYPE_NVHOST) => self.nvhost_ctrl_ioctl(nr, data),
            (NvFile::NvHostCtrlGpu, TYPE_CTRL_GPU) => self.ctrl_gpu_ioctl(nr, data, inline_out),
            (NvFile::AddressSpace { as_id }, TYPE_AS_GPU) => self.as_gpu_ioctl(*as_id, nr, data),
            (NvFile::Channel { channel_id }, _) => {
                self.channel_ioctl(mem, *channel_id, ioc_type, nr, data, inline_in)
            }
            (NvFile::Unsupported { .. }, _) => Ok(NV_NOT_SUPPORTED),
            _ => Ok(NV_NOT_IMPLEMENTED),
        };
        // An ioctl that fails is traced whether or not tracing is on. The
        // successes are noise and the failures are not: every driver call here
        // is one the guest believes cannot fail, and the ones that do are
        // invisible otherwise -- they leave no line at all, because the traces
        // that would carry them run *after* the work they describe.
        //
        // "No handler for this one" is left out: `Cpu::nvdrv_request` reports
        // that once per command through the diagnostic channel the browser
        // drains, where this `eprintln!` goes nowhere at all.
        match &outcome {
            Ok(code) if !matches!(*code, NV_OK | NV_NOT_IMPLEMENTED | NV_NOT_SUPPORTED) => {
                eprintln!("[nv] FAILED {file:?} type={ioc_type:#04x} nr={nr:#04x} -> {code:#x}")
            }
            Err(error) => {
                eprintln!("[nv] ERROR {file:?} type={ioc_type:#04x} nr={nr:#04x}: {error}")
            }
            _ => {}
        }
        outcome
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
                if self.gpu.trace {
                    // With whatever the handle was bound to before. A second
                    // alloc on a handle that is already mapped would leave the
                    // GPU address space pointing at the old memory, which is
                    // the kind of thing that shows up as a buffer full of
                    // zeroes and nothing else to explain it.
                    let was = self
                        .gpu
                        .nvmap
                        .get(handle)
                        .map(|h| (h.cpu_addr, h.allocated));
                    eprintln!(
                        "[nv] nvmap alloc handle={handle} addr={:#x} (was {was:x?})",
                        addr as u32
                    );
                }
                self.gpu
                    .nvmap
                    .alloc(handle, heap_mask, flags, align, kind, addr as u32)?;
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
            // EventSignal { in u32 event_id }: force a slot signalled without
            // a fence reaching its threshold, which is how a driver releases
            // a thread parked on an event it is about to tear down. Answering
            // a bare success left that thread waiting on a slot nothing would
            // ever set.
            0x1C => {
                let slot = read_u32(data, 0) as usize;
                match self.gpu.host1x.events.get_mut(slot) {
                    Some(event) => {
                        event.signalled = true;
                        Ok(NV_OK)
                    }
                    None => Ok(NV_BAD_PARAMETER),
                }
            }
            // EventRegister / EventUnregister { in event_id }
            0x1F => {
                self.gpu.host1x.register_event(read_u32(data, 0).min(63))?;
                Ok(NV_OK)
            }
            0x20 => {
                self.gpu
                    .host1x
                    .unregister_event(read_u32(data, 0).min(63))?;
                Ok(NV_OK)
            }
            // EventWait { in syncpt_id, threshold, timeout; inout value }
            0x1D => {
                let id = read_u32(data, 0);
                let threshold = read_u32(data, 4);
                let slot = read_u32(data, 0x0C);
                if let Some(event) = self.gpu.host1x.events.get_mut(slot.min(63) as usize) {
                    event.fence = NvFence {
                        id,
                        value: threshold,
                    };
                    event.signalled = true;
                }
                write_u32(data, 0x0C, slot);
                Ok(NV_OK)
            }
            // GetConfig { in char domain[0x41], key[0x41]; out char value[0x101] }
            //
            // The driver's settings lookup, and in practice its debug
            // overrides: `nv!NVRM_GPU_NVGPU_NO_SYNCPOINTS`,
            // `nv!NVRM_GPU_PREVENT_USE`, `nv!NVN_THROUGH_OPENGL` and some
            // ninety more, which a retail title asks for one at a time while
            // it starts. Switchbrew records the ioctl as "not available in
            // production mode", and a console that boots normally is in
            // production mode: there is no setting to find, whatever is
            // asked for.
            //
            // Refused rather than answered empty, because the guest *can*
            // tell those apart: `NvOsGetConfigString` maps a successful
            // ioctl to "this key is set" without ever reading the value it
            // got back. An empty success therefore enables every override
            // the driver has, `NVWSI_FILL` included -- which makes the WSI
            // layer fill each dequeued buffer a pixel at a time, and was 45%
            // of a Just Dance 2017 frame.
            0x1B => {
                if self.gpu.trace {
                    eprintln!(
                        "[nv] GetConfig {}!{} -> refused (production mode)",
                        ascii_field(data, 0, 0x41),
                        ascii_field(data, 0x41, 0x41)
                    );
                }
                Ok(NV_NOT_IMPLEMENTED)
            }
            // EventWaitAsync { in syncpt_id, threshold, timeout, event_id }:
            // the same wait, arming a slot instead of blocking. Submissions
            // retire inside their own ioctl, so by the time anyone asks the
            // fence has already passed and the slot is signalled on arrival —
            // which is the one thing the bare success it used to answer did
            // not do.
            0x1E => {
                let id = read_u32(data, 0);
                let threshold = read_u32(data, 4);
                let slot = read_u32(data, 0x0C) as usize;
                match self.gpu.host1x.events.get_mut(slot) {
                    Some(event) => {
                        event.fence = NvFence {
                            id,
                            value: threshold,
                        };
                        event.signalled = true;
                        Ok(NV_OK)
                    }
                    None => Ok(NV_BAD_PARAMETER),
                }
            }
            _ => Ok(NV_NOT_IMPLEMENTED),
        }
    }

    // -- /dev/nvhost-ctrl-gpu --------------------------------------------

    fn ctrl_gpu_ioctl(
        &mut self,
        nr: u32,
        data: &mut [u8],
        inline_out: &mut Vec<u8>,
    ) -> Result<u32> {
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
            // ZbcSetTable { in u32 color_ds[4], color_l2[4], depth, format,
            //               type }. Registering the same value twice takes a
            // second reference to the slot that already holds it rather than
            // spending another, which is what makes a table this small last.
            0x03 => {
                let entry = ZbcEntry {
                    color_ds: [0, 1, 2, 3].map(|i| read_u32(data, i * 4)),
                    color_l2: [0, 1, 2, 3].map(|i| read_u32(data, 0x10 + i * 4)),
                    depth: read_u32(data, 0x20),
                    ref_cnt: 0,
                    format: read_u32(data, 0x24),
                };
                let table = match read_u32(data, 0x28) {
                    ZBC_TYPE_COLOR => &mut self.zbc_color,
                    ZBC_TYPE_DEPTH => &mut self.zbc_depth,
                    _ => return Ok(NV_BAD_PARAMETER),
                };
                // A full table is what a driver finds out about here; there
                // is no eviction, on hardware either.
                if table.add(entry) {
                    Ok(NV_OK)
                } else {
                    Ok(NV_INSUFFICIENT_MEMORY)
                }
            }
            // ZbcQueryTable { inout nvioctl_zbc_entry }, whose `type` selects
            // a table and whose trailing `index_size` field carries the index
            // in and the table size back out. Type 0 (`INVALID`) asks for
            // nothing but that size, which is what `libnx`'s wrapper does.
            0x04 => {
                let index = read_u32(data, 0x30) as usize;
                let table = match read_u32(data, 0x2C) {
                    ZBC_TYPE_INVALID => {
                        write_u32(data, 0x30, ZBC_TABLE_SIZE as u32);
                        return Ok(NV_OK);
                    }
                    ZBC_TYPE_COLOR => &self.zbc_color,
                    ZBC_TYPE_DEPTH => &self.zbc_depth,
                    _ => return Ok(NV_BAD_PARAMETER),
                };
                let Some(entry) = table.get(index) else {
                    return Ok(NV_BAD_PARAMETER);
                };
                for (i, value) in entry.color_ds.iter().enumerate() {
                    write_u32(data, i * 4, *value);
                }
                for (i, value) in entry.color_l2.iter().enumerate() {
                    write_u32(data, 0x10 + i * 4, *value);
                }
                write_u32(data, 0x20, entry.depth);
                write_u32(data, 0x24, entry.ref_cnt);
                write_u32(data, 0x28, entry.format);
                write_u32(data, 0x30, ZBC_TABLE_SIZE as u32);
                Ok(NV_OK)
            }
            // GetCharacteristics { in u64 buf_size, buf_addr; out gc }.
            // The payload goes both inline (where `nvIoctl` callers read it)
            // and into the inline-output buffer (where `nvIoctl3` callers do).
            0x05 => {
                write_u64(data, 0, GPU_CHARACTERISTICS.len() as u64);
                for (i, byte) in GPU_CHARACTERISTICS.iter().enumerate() {
                    if let Some(slot) = data.get_mut(0x10 + i) {
                        *slot = *byte;
                    }
                }
                inline_out.clear();
                inline_out.extend_from_slice(&GPU_CHARACTERISTICS);
                Ok(NV_OK)
            }
            // GetTpcMasks { in bufsize, pad, bufaddr; out u8[8] }, the same
            // shape and so the same two destinations.
            0x06 => {
                let mask = (1u32 << GPU_TPC_PER_GPC) - 1; // the TPCs present in GPC 0
                write_u32(data, 0x10, mask);
                write_u32(data, 0x14, 0);
                inline_out.clear();
                inline_out.extend_from_slice(&mask.to_le_bytes());
                inline_out.extend_from_slice(&0u32.to_le_bytes());
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
            // VsmsMapping { in u64 vsms_map_buf_addr }: which (GPC, TPC) each
            // of the chip's "virtual SMs" sits on, one
            // `{ u8 gpc_index, u8 tpc_index }` entry per TPC.
            //
            // This is in no libnx header and no other emulator implements it,
            // so it was identified from its caller rather than from a table.
            // `nnSdk`'s bundled `nvrm_gpu` builds the request inline (`movz
            // w23, #0x4713` / `movk w23, #0xc008`, at `sdk!0xd740950` in
            // Tomodachi Life) and hands the driver two buffer descriptors:
            // the 8-byte argument, and an array of `num_tpc_per_gpc` **u16**
            // entries. That is upstream's `nvgpu_gpu_vsms_mapping_args`
            // exactly — the argument is a bare buffer address, zero here
            // because the Switch passes the buffer out-of-line rather than as
            // a pointer, and the entries are `nvgpu_gpu_vsms_mapping_entry`.
            // GM20B's one GPC and two TPCs are the four bytes the guest
            // actually offers.
            0x13 => {
                inline_out.clear();
                for gpc in 0..GPU_NUM_GPC {
                    for tpc in 0..GPU_TPC_PER_GPC {
                        inline_out.push(gpc as u8);
                        inline_out.push(tpc as u8);
                    }
                }
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
                let offset = self
                    .gpu
                    .address_space_mut(as_id)?
                    .alloc_space(pages, page_size, flags, requested)?;
                write_u64(data, 0x10, offset);
                Ok(NV_OK)
            }
            // FreeSpace { in u64 offset; in pages, page_size }
            0x03 => {
                let offset = read_u64(data, 0);
                let pages = read_u32(data, 8);
                let page_size = read_u32(data, 0x0C);
                self.gpu
                    .address_space_mut(as_id)?
                    .free_space(offset, pages, page_size)?;
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

                // The remap form maps nothing new: `offset` names a mapping
                // that already exists and `nvmap_handle` is unused (0 here),
                // so it has to be split off before the handle lookup below —
                // which is what rejected it with `BadParameter`, and what
                // aborted `deko3d`'s image setup before it drew a frame.
                if flags & FLAG_REMAP_SUB_RANGE != 0 {
                    let gpu_va = requested.wrapping_add(buffer_offset);
                    let kind = (kind != u32::MAX).then_some(kind as u8);
                    let space = self.gpu.address_space_mut(as_id)?;
                    return if space.remap(gpu_va, mapping_size, kind) {
                        write_u64(data, 0x20, gpu_va);
                        Ok(NV_OK)
                    } else {
                        Ok(NV_BAD_PARAMETER)
                    };
                }
                let handle = match self.gpu.nvmap.get(nvmap_handle) {
                    Some(h) if h.allocated => *h,
                    Some(_) => return Ok(NV_INVALID_STATE),
                    None => return Ok(NV_BAD_PARAMETER),
                };
                let size = if mapping_size != 0 {
                    mapping_size
                } else {
                    handle.size as u64
                };
                let kind = if kind == u32::MAX {
                    handle.kind
                } else {
                    kind as u8
                };
                let page_size = if page_size == 0 {
                    SMALL_PAGE_SIZE
                } else {
                    page_size as u64
                };
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
                if self.gpu.trace {
                    eprintln!(
                        "[nv] map handle={nvmap_handle} cpu={:#x}+{buffer_offset:#x} size={size:#x} -> gpu_va={offset:#x} (obj cpu={:#x} size={:#x})",
                        cpu_addr, handle.cpu_addr, handle.size
                    );
                }
                write_u32(data, 0x0C, page_size as u32);
                write_u64(data, 0x20, offset);
                Ok(NV_OK)
            }
            // GetVARegions { u64 not_used; inout bufsize; pad; out regions[2] }
            0x08 => {
                write_u32(data, 8, 2 * 24);
                let regions = [
                    (
                        SMALL_REGION_BASE,
                        SMALL_PAGE_SIZE,
                        (SMALL_REGION_END - SMALL_REGION_BASE) / SMALL_PAGE_SIZE,
                    ),
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
            // Remap { u16 flags, kind; u32 nvmap_handle, map_offset,
            //         gpu_offset, pages }[], every offset and length in big
            // pages. Unlike `MapBufferEx` this is a batch, and it names the
            // GPU VA outright rather than asking for one: it is how a driver
            // fills in a range it reserved as sparse, and how a title gives
            // one buffer several block-linear kinds by mapping it repeatedly.
            // A zero handle means "leave this range unmapped".
            0x14 => {
                const OP_SIZE: usize = 0x14;
                if data.len() < OP_SIZE || !data.len().is_multiple_of(OP_SIZE) {
                    return Ok(NV_BAD_PARAMETER);
                }
                let trace = self.gpu.trace;
                for at in (0..data.len()).step_by(OP_SIZE) {
                    let kind = read_u16(data, at + 2);
                    let nvmap_handle = read_u32(data, at + 4);
                    let map_offset = u64::from(read_u32(data, at + 8)) << 16;
                    let gpu_va = u64::from(read_u32(data, at + 0x0C)) << 16;
                    let size = u64::from(read_u32(data, at + 0x10)) << 16;
                    if size == 0 {
                        return Ok(NV_BAD_PARAMETER);
                    }
                    if nvmap_handle == 0 {
                        self.gpu.address_space_mut(as_id)?.unmap_range(gpu_va, size);
                        continue;
                    }
                    let handle = match self.gpu.nvmap.get(nvmap_handle) {
                        Some(h) if h.allocated => *h,
                        Some(_) => return Ok(NV_INVALID_STATE),
                        None => return Ok(NV_BAD_PARAMETER),
                    };
                    let cpu_addr = handle.cpu_addr.wrapping_add(map_offset as u32);
                    let space = self.gpu.address_space_mut(as_id)?;
                    let page_size = space.big_page_size;
                    space.unmap_range(gpu_va, size);
                    space.map(
                        cpu_addr,
                        size,
                        nvmap_handle,
                        kind as u8,
                        page_size,
                        FLAG_FIXED_OFFSET,
                        gpu_va,
                    )?;
                    if trace {
                        eprintln!(
                            "[nv] remap handle={nvmap_handle} cpu={cpu_addr:#x} \
                             size={size:#x} kind={kind:#x} -> gpu_va={gpu_va:#x}"
                        );
                    }
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
            // SetTimeslice (0x1D) joins them: how long the channel holds the
            // GPU before the host1x scheduler moves on is a real knob on
            // hardware and nothing at all on a command processor that runs
            // each submission to completion inside its own ioctl. Refusing it
            // is the one answer that is definitely wrong -- nnSdk's nvn driver
            // checks, and a channel it failed to configure is one it has no
            // reason to trust.
            (TYPE_CHANNEL, 0x01)
            | (TYPE_CHANNEL, 0x03)
            | (TYPE_CHANNEL, 0x0B)
            | (TYPE_CHANNEL, 0x0C)
            | (TYPE_CHANNEL, 0x0D)
            | (TYPE_CHANNEL, 0x1D)
            | (TYPE_CTRL_GPU, 0x14)
            | (TYPE_NVHOST, 0x07)
            | (TYPE_NVHOST, 0x08) => Ok(NV_OK),

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
                if self.gpu.trace {
                    eprintln!("[nv] channel {nr:#04x} in={:02x?}", data);
                }
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
    put!(out, 0x0C, GPU_NUM_GPC, 4); // num_gpc
    put!(out, 0x10, 0x40000, 8); // L2 cache size
    put!(out, 0x18, 0, 8); // on-board video memory (none: it is system RAM)
    put!(out, 0x20, GPU_TPC_PER_GPC, 4); // num_tpc_per_gpc
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

/// A fixed-width NUL-padded ASCII field, the way the nvhost config ioctls
/// carry their strings.
fn ascii_field(data: &[u8], at: usize, len: usize) -> String {
    let bytes = data.get(at..at + len).unwrap_or(&[]);
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn read_u16(data: &[u8], at: usize) -> u16 {
    let mut v = 0u16;
    for i in 0..2 {
        v |= u16::from(data.get(at + i).copied().unwrap_or(0)) << (8 * i);
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
        let mut inline_out = Vec::new();
        let request = make_ioctl(IOWR, ty, nr, data.len() as u32);
        drv.ioctl(mem, fd, request, data, &[], &mut inline_out)
            .unwrap()
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
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x01, &mut create),
            NV_OK
        );
        let handle = read_u32(&create, 4);
        assert_ne!(handle, 0);

        let mut alloc = [0u8; 0x20];
        write_u32(&mut alloc, 0, handle);
        write_u32(&mut alloc, 0x0C, 0x1000);
        alloc[0x10] = 0xFE;
        write_u64(&mut alloc, 0x18, 0x3000_0000);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x04, &mut alloc),
            NV_OK
        );

        let mut param = [0u8; 0x0C];
        write_u32(&mut param, 0, handle);
        write_u32(&mut param, 4, 3); // Base
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVMAP, 0x09, &mut param),
            NV_OK
        );
        assert_eq!(read_u32(&param, 8), 0x3000_0000);
    }

    /// The same, keeping the out-of-line reply an `nvIoctl3` caller reads its
    /// payload from.
    fn ioctl_out(
        drv: &mut NvDrv,
        mem: &mut Memory,
        fd: u32,
        ty: u32,
        nr: u32,
        data: &mut [u8],
    ) -> (u32, Vec<u8>) {
        let mut inline_out = Vec::new();
        let request = make_ioctl(IOWR, ty, nr, data.len() as u32);
        let code = drv
            .ioctl(mem, fd, request, data, &[], &mut inline_out)
            .unwrap();
        (code, inline_out)
    }

    #[test]
    fn the_virtual_sm_map_names_a_gpc_and_a_tpc_for_every_shader_unit() {
        // `VsmsMapping` is asked for once during `nvrm_gpu`'s device probe,
        // and the buffer the guest offers is sized from the TPC count it was
        // given moments earlier by `GetCharacteristics` — so the two have to
        // agree or the driver indexes one chip with the other's count.
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl-gpu").unwrap();
        let mut arg = [0u8; 8];
        let (code, map) = ioctl_out(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x13, &mut arg);
        assert_eq!(code, NV_OK);
        assert_eq!(map, vec![0, 0, 0, 1], "a (gpc, tpc) pair per TPC of GPC 0");
        assert_eq!(
            map.len() as u32,
            2 * GPU_NUM_GPC * GPU_TPC_PER_GPC,
            "the map is sized by the counts GetCharacteristics reports"
        );

        // And those counts are the ones the characteristics and the TPC mask
        // are built from, so nothing here can drift apart.
        let mut chars = [0u8; 0xB0];
        let (code, gc) = ioctl_out(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x05, &mut chars);
        assert_eq!(code, NV_OK);
        assert_eq!(read_u32(&gc, 0x0C), GPU_NUM_GPC, "num_gpc");
        assert_eq!(read_u32(&gc, 0x20), GPU_TPC_PER_GPC, "num_tpc_per_gpc");
        let mut masks = [0u8; 0x18];
        let (code, tpc) = ioctl_out(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x06, &mut masks);
        assert_eq!(code, NV_OK);
        assert_eq!(
            read_u32(&tpc, 0).count_ones(),
            GPU_TPC_PER_GPC,
            "a bit per TPC"
        );
    }

    #[test]
    fn the_zbc_table_hands_back_the_clear_values_it_was_given() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl-gpu").unwrap();

        // ZbcSetTable { color_ds[4], color_l2[4], depth, format, type }.
        let mut set = [0u8; 0x2C];
        for i in 0..4 {
            write_u32(&mut set, i * 4, 0x1111_1111 * (i as u32 + 1));
            write_u32(&mut set, 0x10 + i * 4, 0x2222_2222 * (i as u32 + 1));
        }
        write_u32(&mut set, 0x20, 0x3F80_0000); // depth 1.0
        write_u32(&mut set, 0x24, 0x0A); // format
        write_u32(&mut set, 0x28, ZBC_TYPE_COLOR);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x03, &mut set),
            NV_OK
        );
        // The same value again takes a second reference rather than a second
        // slot: a table of sixteen does not survive a driver that re-registers
        // its clear colour every frame.
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x03, &mut set),
            NV_OK
        );
        assert_eq!(drv.zbc_color.used(), 1);
        assert_eq!(
            drv.zbc_depth.used(),
            0,
            "a colour entry landed in the depth table"
        );

        // ZbcQueryTable { inout nvioctl_zbc_entry }: type selects the table,
        // the trailing field carries the index in and the table size out.
        let mut query = [0u8; 0x34];
        write_u32(&mut query, 0x2C, ZBC_TYPE_COLOR);
        write_u32(&mut query, 0x30, 0);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x04, &mut query),
            NV_OK
        );
        assert_eq!(read_u32(&query, 0x00), 0x1111_1111);
        assert_eq!(read_u32(&query, 0x0C), 0x4444_4444);
        assert_eq!(read_u32(&query, 0x10), 0x2222_2222);
        assert_eq!(read_u32(&query, 0x20), 0x3F80_0000, "depth");
        assert_eq!(read_u32(&query, 0x24), 2, "ref count");
        assert_eq!(read_u32(&query, 0x28), 0x0A, "format");
        assert_eq!(read_u32(&query, 0x30), ZBC_TABLE_SIZE as u32, "table size");

        // Type 0 asks for the size and nothing else, which is the only form
        // libnx's own wrapper sends.
        let mut size_only = [0u8; 0x34];
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x04, &mut size_only),
            NV_OK
        );
        assert_eq!(read_u32(&size_only, 0x30), ZBC_TABLE_SIZE as u32);

        // Past the end of the table is a refusal, not a zeroed entry.
        write_u32(&mut query, 0x2C, ZBC_TYPE_COLOR);
        write_u32(&mut query, 0x30, ZBC_TABLE_SIZE as u32);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x04, &mut query),
            NV_BAD_PARAMETER
        );
    }

    #[test]
    fn a_full_zbc_table_is_refused_rather_than_silently_dropping_entries() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl-gpu").unwrap();
        let mut set = [0u8; 0x2C];
        write_u32(&mut set, 0x28, ZBC_TYPE_DEPTH);
        for i in 0..ZBC_TABLE_SIZE as u32 {
            write_u32(&mut set, 0x20, i + 1);
            assert_eq!(
                ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x03, &mut set),
                NV_OK
            );
        }
        write_u32(&mut set, 0x20, 0xFFFF);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x03, &mut set),
            NV_INSUFFICIENT_MEMORY
        );
        assert_eq!(drv.zbc_depth.used(), ZBC_TABLE_SIZE);
    }

    /// A success here is read as "this key is set" no matter what value came
    /// back with it, so an unset key has to be refused rather than answered
    /// empty. Answering `NV_OK` enabled `NVWSI_FILL`, and the WSI layer then
    /// filled every dequeued buffer a pixel at a time.
    #[test]
    fn get_config_refuses_every_key() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl").unwrap();

        // domain[0x41], key[0x41], value[0x101].
        let mut arg = [0u8; 0x183];
        arg[..2].copy_from_slice(b"nv");
        arg[0x41..0x41 + 24].copy_from_slice(b"NVRM_GPU_NVGPU_NO_ZCULL\0");

        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVHOST, 0x1B, &mut arg),
            NV_NOT_IMPLEMENTED
        );
        // The keys the caller asked about are left where they were.
        assert_eq!(&arg[..2], b"nv");
        assert_eq!(&arg[0x41..0x41 + 23], b"NVRM_GPU_NVGPU_NO_ZCULL");
    }

    #[test]
    fn an_event_is_signalled_by_the_ioctls_that_say_so() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl").unwrap();

        let mut arg = [0u8; 0x10];
        write_u32(&mut arg, 0, 3); // EventRegister slot 3
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVHOST, 0x1F, &mut arg),
            NV_OK
        );
        assert!(!drv.gpu.host1x.events[3].signalled);

        // EventWaitAsync { syncpt_id, threshold, timeout, event_id }: the work
        // it names retired inside its own submission, so the slot comes back
        // already signalled instead of waiting for something that will not
        // happen again.
        write_u32(&mut arg, 0, 9); // syncpt
        write_u32(&mut arg, 4, 5); // threshold
        write_u32(&mut arg, 0x0C, 3); // slot
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVHOST, 0x1E, &mut arg),
            NV_OK
        );
        assert!(drv.gpu.host1x.events[3].signalled);
        assert_eq!(drv.gpu.host1x.events[3].fence, NvFence { id: 9, value: 5 });

        // EventSignal on a slot that does not exist is refused rather than
        // reported as a success nothing acted on.
        let mut signal = [0u8; 4];
        write_u32(&mut signal, 0, 64);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVHOST, 0x1C, &mut signal),
            NV_BAD_PARAMETER
        );
        write_u32(&mut signal, 0, 7);
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_NVHOST, 0x1C, &mut signal),
            NV_OK
        );
        assert!(drv.gpu.host1x.events[7].signalled);
    }

    #[test]
    fn remap_fills_a_reserved_range_and_a_zero_handle_clears_it() {
        // `REMAP` is how a title backs address space it reserved as sparse,
        // and how it gives one buffer several block-linear kinds. It names
        // the GPU VA outright and counts everything in big pages, so the
        // request carries neither a page size nor a "fixed offset" flag.
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x2_0000).unwrap();
        let (map_fd, _) = drv.open("/dev/nvmap").unwrap();
        let (as_fd, _) = drv.open("/dev/nvhost-as-gpu").unwrap();

        let mut create = [0u8; 8];
        write_u32(&mut create, 0, 0x2_0000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x01, &mut create);
        let handle = read_u32(&create, 4);
        let mut alloc = [0u8; 0x20];
        write_u32(&mut alloc, 0, handle);
        write_u64(&mut alloc, 0x18, 0x3000_0000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x04, &mut alloc);

        // One op: the second big page of the handle, at GPU VA 0x8_0000,
        // with kind 0xFE.
        let mut op = [0u8; 0x14];
        write_u32(&mut op, 0, 0x00FE_0000); // flags 0, kind 0xFE
        write_u32(&mut op, 4, handle);
        write_u32(&mut op, 8, 1); // map offset: one big page in
        write_u32(&mut op, 0x0C, 8); // gpu offset
        write_u32(&mut op, 0x10, 1); // one big page
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x14, &mut op),
            NV_OK
        );

        let as_id = match drv.file(as_fd) {
            Some(NvFile::AddressSpace { as_id }) => *as_id,
            other => panic!("not an address space: {other:?}"),
        };
        let space = drv.gpu.address_space_mut(as_id).unwrap();
        assert_eq!(space.translate(0x8_0000), Some((0x3001_0000, 0x1_0000)));
        assert_eq!(space.mapping_at(0x8_0000).map(|m| m.kind), Some(0xFE));

        write_u32(&mut op, 4, 0); // no handle: unmap the range again
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x14, &mut op),
            NV_OK
        );
        assert_eq!(
            drv.gpu
                .address_space_mut(as_id)
                .unwrap()
                .translate(0x8_0000),
            None
        );
    }

    #[test]
    fn map_buffer_ex_remaps_a_sub_range_without_an_nvmap_handle() {
        // `deko3d` maps a memory block once and then re-maps the ranges
        // holding block-linear images over the top with the kind that
        // describes their swizzle. That second call sets
        // `NVGPU_AS_MAP_BUFFER_FLAGS_MODIFY`, names the existing mapping in
        // `offset`, and leaves the nvmap handle **0** — which the ordinary
        // map path rejected as `BadParameter`, aborting before the first frame.
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        mem.map_zero(0x3000_0000, 0x4000).unwrap();
        let (map_fd, _) = drv.open("/dev/nvmap").unwrap();
        let (as_fd, _) = drv.open("/dev/nvhost-as-gpu").unwrap();

        let mut create = [0u8; 8];
        write_u32(&mut create, 0, 0x4000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x01, &mut create);
        let handle = read_u32(&create, 4);
        let mut alloc = [0u8; 0x20];
        write_u32(&mut alloc, 0, handle);
        write_u64(&mut alloc, 0x18, 0x3000_0000);
        ioctl(&mut drv, &mut mem, map_fd, TYPE_NVMAP, 0x04, &mut alloc);

        let mut map = [0u8; 0x28];
        write_u32(&mut map, 8, handle);
        write_u64(&mut map, 0x18, 0x4000);
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x06, &mut map),
            NV_OK
        );
        let gpu_va = read_u64(&map, 0x20);

        // Re-map the middle 0x1000 bytes with kind 0xdb (a block-linear
        // kind), the way the driver is asked to.
        let mut remap = [0u8; 0x28];
        write_u32(&mut remap, 0, 1 << 8); // MODIFY
        write_u32(&mut remap, 4, 0xdb);
        write_u64(&mut remap, 0x10, 0x1000); // buffer_offset
        write_u64(&mut remap, 0x18, 0x1000); // mapping_size
        write_u64(&mut remap, 0x20, gpu_va);
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x06, &mut remap),
            NV_OK
        );
        assert_eq!(read_u64(&remap, 0x20), gpu_va + 0x1000);

        let NvFile::AddressSpace { as_id } = *drv.file(as_fd).unwrap() else {
            panic!("expected an address-space fd");
        };
        let space = &drv.gpu.address_spaces[&as_id];
        // The backing memory does not move: every byte of the original
        // mapping still resolves to the CPU address it did before, across
        // both of the boundaries the split introduced.
        for offset in [0u64, 0xFFF, 0x1000, 0x1FFF, 0x2000, 0x3FFF] {
            assert_eq!(
                space.translate(gpu_va + offset).map(|(cpu, _)| cpu),
                Some(0x3000_0000 + offset as u32),
                "{offset:#x}"
            );
        }
        // Only the re-mapped range carries the new kind.
        assert_eq!(space.mapping_at(gpu_va).map(|m| m.kind), Some(0));
        assert_eq!(
            space.mapping_at(gpu_va + 0x1000).map(|m| m.kind),
            Some(0xdb)
        );
        assert_eq!(space.mapping_at(gpu_va + 0x2000).map(|m| m.kind), Some(0));

        // A range no mapping covers is the one thing this really cannot do.
        let mut orphan = [0u8; 0x28];
        write_u32(&mut orphan, 0, 1 << 8);
        write_u64(&mut orphan, 0x18, 0x1000);
        write_u64(&mut orphan, 0x20, gpu_va + 0x10_0000);
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x06, &mut orphan),
            NV_BAD_PARAMETER
        );
    }

    #[test]
    fn get_characteristics_reports_the_gm20b() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (fd, _) = drv.open("/dev/nvhost-ctrl-gpu").unwrap();
        let mut data = [0u8; 0x10 + 0xA0];
        assert_eq!(
            ioctl(&mut drv, &mut mem, fd, TYPE_CTRL_GPU, 0x05, &mut data),
            NV_OK
        );
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
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x06, &mut map),
            NV_OK
        );
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
        assert_eq!(
            ioctl(&mut drv, &mut mem, as_fd, TYPE_AS_GPU, 0x01, &mut bind),
            NV_OK
        );

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
        assert_eq!(
            ioctl(&mut drv, &mut mem, chan_fd, TYPE_CHANNEL, 0x08, &mut submit),
            NV_OK
        );

        let fence_id = read_u32(&submit, 0x10);
        let fence_value = read_u32(&submit, 0x14);
        assert!(drv.gpu.host1x.is_expired(fence_id, fence_value).unwrap());

        let NvFile::Channel { channel_id } = *drv.file(chan_fd).unwrap() else {
            panic!("expected a channel fd");
        };
        assert_eq!(
            drv.gpu
                .channel_mut(channel_id)
                .unwrap()
                .three_d
                .regs
                .get(0x360),
            0x55
        );
    }

    #[test]
    fn syncpt_read_reflects_submissions() {
        let mut drv = NvDrv::new();
        let mut mem = Memory::new();
        let (ctrl_fd, _) = drv.open("/dev/nvhost-ctrl").unwrap();
        drv.gpu.host1x.set(9, 3).unwrap();
        let mut data = [0u8; 8];
        write_u32(&mut data, 0, 9);
        assert_eq!(
            ioctl(&mut drv, &mut mem, ctrl_fd, TYPE_NVHOST, 0x14, &mut data),
            NV_OK
        );
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
