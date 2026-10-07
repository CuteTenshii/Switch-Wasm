//! `vi`: the display service, its layers, and the `IHOSBinderDriver` parcels
//! that carry Android's buffer queue to [`crate::display`].

use crate::cpu::Cpu;
use crate::Result;

/// The refresh rate `ListDisplayModes` reports, in Hz.
const DISPLAY_REFRESH_HZ: f32 = 60.0;

/// The one display's name, as `OpenDisplay` and `ListDisplays` use it.
const DISPLAY_NAME: &str = "Default";

const DISPLAY_ID: u64 = 1;

/// The one layer, shared by `OpenLayer`, `CreateStrayLayer` and `CreateManagedLayer`.
const LAYER_ID: u64 = 1;

impl Cpu {
    pub(crate) fn vi_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        // Control requests. Use `ipc_is_control_request`, not `type == 5`:
        // `nnSdk` sends the with-context encoding (type 7).
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "vi:root");
                    let raw = obj.to_le_bytes();
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = if self.ipc_is_domain_request(tls) {
            self.ipc_domain_object_id(tls)
        } else {
            0xFFFFFFFF
        };
        let is_domain = object_id != 0xFFFFFFFF;
        // Display commands are identical on every sub-interface and dialect.
        if let Some(done) = self.vi_common_command(tls, cmd_id) {
            return done;
        }
        if !is_domain {
            // Non-domain sessions marshal output objects as move handles;
            // dispatch on the per-handle sub-interface, defaulting to the vi root.
            let iface = self
                .vi_ifaces
                .get(&handle)
                .cloned()
                .unwrap_or_else(|| "vi:root".to_owned());
            match iface.as_str() {
                // IHOSBinderDriverRelay: TransactParcel (0), AdjustRefcount (1),
                // GetNativeHandle (2), TransactParcelAuto (3, 3.0.0+). 0 and 3
                // differ only in parcel marshalling.
                "vi:ihosbd" => match cmd_id {
                    Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    // GetNativeHandle: the buffer queue's event, as a copy handle.
                    Some(2) => {
                        let h = self.vi_binder_event();
                        self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                    }
                    Some(0) | Some(3) => self.vi_transact_parcel(tls),
                    _ => self.vi_unhandled(tls, &iface, cmd_id),
                },
                // vi root: cmd 2 hands out the IApplicationDisplayService.
                "vi:root" => match cmd_id {
                    Some(2) => self.vi_out_session(tls, "vi:iads"),
                    _ => self.vi_unhandled(tls, &iface, cmd_id),
                },
                // IApplicationDisplayService and the other display services.
                _ => match cmd_id {
                    Some(100) => self.vi_out_session(tls, "vi:ihosbd"),
                    Some(101) => self.vi_out_session(tls, "vi:isds"),
                    Some(102) => self.vi_out_session(tls, "vi:imds"),
                    Some(103) => self.vi_out_session(tls, "vi:ihosbdind"),
                    // GetDisplayVsyncEvent: a copy handle, signalled per presented frame.
                    Some(5202) => {
                        let h = self.alloc_event("vi:vsync", true);
                        self.vsync_event = Some(h);
                        self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                    }
                    _ => self.vi_unhandled(tls, &iface, cmd_id),
                },
            }
        } else {
            match self.domain_interface(handle, object_id) {
                Some("vi:root") => match cmd_id {
                    Some(2) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:iads");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    _ => self.vi_unhandled(tls, "vi:root", cmd_id),
                },
                Some("vi:iads") => match cmd_id {
                    Some(100) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:ihosbd");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    Some(101) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:isds");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    Some(102) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:imds");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    Some(103) => {
                        let obj = self.alloc_domain_object();
                        self.record_domain_object(handle, obj, "vi:ihosbdind");
                        self.write_ipc_response(tls, 0, &[], &[], &[obj])
                    }
                    // GetDisplayVsyncEvent: a copy handle, signalled per presented frame.
                    Some(5202) => {
                        let h = self.alloc_event("vi:vsync", true);
                        self.vsync_event = Some(h);
                        self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                    }
                    _ => self.vi_unhandled(tls, "vi:iads", cmd_id),
                },
                // The binder relay works the same on a domain session.
                Some("vi:ihosbd") => match cmd_id {
                    Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    Some(2) => {
                        let h = self.vi_binder_event();
                        self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                    }
                    Some(0) | Some(3) => self.vi_transact_parcel(tls),
                    _ => self.vi_unhandled(tls, "vi:ihosbd", cmd_id),
                },
                Some("vi:isds") => self.vi_unhandled(tls, "vi:isds", cmd_id),
                Some("vi:imds") => self.vi_unhandled(tls, "vi:imds", cmd_id),
                _ => self.vi_unhandled(tls, "vi:m", cmd_id),
            }
        }
    }

    /// The `vi` commands that answer with data, identical across the display
    /// services and dialects. Returns `None` for anything not answered here.
    fn vi_common_command(&mut self, tls: u32, cmd_id: Option<u32>) -> Option<Result<()>> {
        let (width, height) = self.operation_mode().display_size();
        let raw: Vec<u8> = match cmd_id? {
            // ListDisplays: one display.
            1000 => {
                let mut info = [0u8; 0x60];
                let name = DISPLAY_NAME.as_bytes();
                info[..name.len()].copy_from_slice(name);
                info[0x40] = 1; // layer_limit_enabled
                info[0x48..0x50].copy_from_slice(&1u64.to_le_bytes()); // layer_limit_max
                info[0x50..0x58].copy_from_slice(&u64::from(width).to_le_bytes());
                info[0x58..0x60].copy_from_slice(&u64::from(height).to_le_bytes());
                self.vi_fill_out_buffer(tls, &info);
                1u64.to_le_bytes().to_vec()
            }
            // OpenDisplay / OpenDefaultDisplay.
            1010 | 1011 => DISPLAY_ID.to_le_bytes().to_vec(),
            // GetDisplayResolution.
            1102 => {
                let mut raw = Vec::with_capacity(0x10);
                raw.extend_from_slice(&u64::from(width).to_le_bytes());
                raw.extend_from_slice(&u64::from(height).to_le_bytes());
                raw
            }
            // GetZOrderCountMin / GetZOrderCountMax: one layer stack.
            1200 | 1202 => 0i64.to_le_bytes().to_vec(),
            // GetDisplayLogicalResolution, as two s32.
            1203 => {
                let mut raw = Vec::with_capacity(8);
                raw.extend_from_slice(&(width as i32).to_le_bytes());
                raw.extend_from_slice(&(height as i32).to_le_bytes());
                raw
            }
            // CreateManagedLayer.
            2010 => LAYER_ID.to_le_bytes().to_vec(),
            // _viOpenLayer (2020) / _viCreateStrayLayer (2030 / 2012 / 2312):
            // a native-window parcel, returning its size.
            2020 => return Some(self.vi_native_window(tls, 8)),
            2030 | 2012 | 2312 => return Some(self.vi_native_window(tls, 16)),
            // SetLayerScalingMode(u32 mode, u64 layer_id): only `ScaleToWindow` (2)
            // and `PreserveAspectRatio` (4) are supported.
            2101 => {
                const VI_OPERATION_FAILED: u32 = 114 | (1 << 9);
                const VI_NOT_SUPPORTED: u32 = 114 | (6 << 9);
                let data = self.ipc_request_data(tls);
                let result = match self.mem.read_u32(data).unwrap_or(0) {
                    2 | 4 => 0,
                    0..=4 => VI_NOT_SUPPORTED,
                    _ => VI_OPERATION_FAILED,
                };
                return Some(self.write_ipc_response(tls, result, &[], &[], &[]));
            }
            // ConvertScalingMode: always ScalingMode_PreserveAspectRatio.
            2102 => 2u64.to_le_bytes().to_vec(),
            // GetLayerZ.
            2204 => 0u64.to_le_bytes().to_vec(),
            // ListDisplayModes: one mode.
            3000 => {
                let mut mode = [0u8; 0x10];
                mode[0..4].copy_from_slice(&width.to_le_bytes());
                mode[4..8].copy_from_slice(&height.to_le_bytes());
                mode[8..12].copy_from_slice(&DISPLAY_REFRESH_HZ.to_le_bytes());
                self.vi_fill_out_buffer(tls, &mode);
                1u64.to_le_bytes().to_vec()
            }
            // ListDisplayRgbRanges / ListDisplayContentTypes: one automatic (0) entry each.
            3001 | 3002 => {
                self.vi_fill_out_buffer(tls, &0u32.to_le_bytes());
                1u64.to_le_bytes().to_vec()
            }
            // GetDisplayMode.
            3200 => {
                let mut raw = Vec::with_capacity(0x10);
                raw.extend_from_slice(&width.to_le_bytes());
                raw.extend_from_slice(&height.to_le_bytes());
                raw.extend_from_slice(&DISPLAY_REFRESH_HZ.to_le_bytes());
                raw.extend_from_slice(&0u32.to_le_bytes());
                raw
            }
            // GetDisplayUnderscan.
            3202 => 0i64.to_le_bytes().to_vec(),
            // GetDisplayContentType / GetDisplayRgbRange / GetDisplayCmuMode: automatic (0).
            3204 | 3206 | 3208 => 0u32.to_le_bytes().to_vec(),
            // GetDisplayContrastRatio.
            3210 => 1.0f32.to_le_bytes().to_vec(),
            // The system shared buffer the Home Menu and system applets draw through.
            8225 | 8250 | 8251 | 8252 | 8253 | 8254 | 8255 | 8256 | 8258 => {
                return Some(self.vi_shared_buffer(tls, cmd_id?));
            }
            _ => return None,
        };
        Some(self.write_ipc_response(tls, 0, &[], &raw, &[]))
    }

    /// `ISystemDisplayService`'s shared-buffer commands: applets acquire a slot
    /// of one shared buffer, render into it and present it.
    fn vi_shared_buffer(&mut self, tls: u32, cmd_id: u32) -> Result<()> {
        use crate::cpu::{SHARED_BUFFER_ADDR, SHARED_BUFFER_SLOTS, SHARED_BUFFER_USABLE_SLOTS};
        // See [`crate::cpu::SHARED_BUFFER_GEOMETRY`]; does not follow the dock.
        let mode = crate::cpu::SHARED_BUFFER_GEOMETRY;
        let (shared_width, shared_height) = mode.display_size();
        let slot_size = mode.shared_buffer_slot_size();
        /// `NvMultiFence`: a count and four `{ id, value }` pairs.
        const FENCE_SIZE: usize = 4 + 4 * 8;
        match cmd_id {
            // GetSharedBufferMemoryHandleId(u64 buffer_id, aruid) -> s32
            // nvmap_handle, u64 size, and the pool layout in the out buffer.
            8225 => {
                let (handle, _) = self.shared_buffer_object();
                let mut layout = [0u8; 0x188];
                layout[..4].copy_from_slice(&(SHARED_BUFFER_SLOTS as i32).to_le_bytes());
                for slot in 0..SHARED_BUFFER_SLOTS as usize {
                    let at = 8 + slot * 0x18;
                    let offset = u64::from(slot_size) * slot as u64;
                    layout[at..at + 8].copy_from_slice(&offset.to_le_bytes());
                    layout[at + 8..at + 16].copy_from_slice(&u64::from(slot_size).to_le_bytes());
                    layout[at + 16..at + 20].copy_from_slice(&(shared_width as i32).to_le_bytes());
                    layout[at + 20..at + 24].copy_from_slice(&(shared_height as i32).to_le_bytes());
                }
                if crate::trace::enabled(crate::trace::Trace::Nv) {
                    crate::traceln!(
                        "[vi] shared pool layout -> recv buffer {:x?}, static {:x?}",
                        self.ipc_recv_buffer(tls, 0),
                        self.ipc_recv_static_buffers(tls)
                    );
                }
                self.vi_fill_out_buffer(tls, &layout);
                let mut raw = Vec::with_capacity(0x10);
                raw.extend_from_slice(&handle.to_le_bytes());
                raw.extend_from_slice(&[0; 4]);
                raw.extend_from_slice(&u64::from(mode.shared_buffer_size()).to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // AcquireSharedFrameBuffer(u64 layer_id) -> fence (empty), s32 slots[4],
            // s64 target slot.
            8254 => {
                let slot = self.shared_buffer_slot;
                self.shared_buffer_slot = (slot + 1) % SHARED_BUFFER_USABLE_SLOTS;
                let mut raw = vec![0u8; FENCE_SIZE];
                for i in 0..4i32 {
                    let index = if (i as u32) < SHARED_BUFFER_USABLE_SLOTS {
                        i
                    } else {
                        -1
                    };
                    raw.extend_from_slice(&index.to_le_bytes());
                }
                raw.resize(raw.len().next_multiple_of(8), 0);
                raw.extend_from_slice(&i64::from(slot).to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // PresentSharedFrameBuffer(fence, Rect crop, u32 transform,
            // s32 swap interval, u64 layer_id, s64 slot). The fence is 36 bytes,
            // so the crop is at 0x24 and the transform at 0x34.
            8255 => {
                let data = self.ipc_request_data(tls);
                let word = |at: u32| self.mem.read_u32(data.wrapping_add(at)).unwrap_or(0);
                let slot = self.mem.read_u64(data.wrapping_add(0x48)).unwrap_or(0) as u32;
                let crop = crate::gpu::Crop {
                    left: word(0x24) as i32,
                    top: word(0x28) as i32,
                    right: word(0x2C) as i32,
                    bottom: word(0x30) as i32,
                };
                let transform = word(0x34);
                if crate::trace::enabled(crate::trace::Trace::Nv) {
                    crate::traceln!(
                        "[vi] present shared slot={slot} crop={crop:?} transform={transform:#x}"
                    );
                }
                let (_, id) = self.shared_buffer_object();
                let buffer = crate::gpu::DisplayBuffer {
                    nvmap_id: id,
                    offset: slot_size.wrapping_mul(slot),
                    width: shared_width,
                    height: shared_height,
                    pitch: mode.shared_buffer_stride(),
                    layout: crate::gpu::NV_LAYOUT_BLOCK_LINEAR,
                    block_height_log2: 4,
                    color_format: 0x01_0053_2120, // A8B8G8R8
                    transform,
                    crop,
                };
                // GPU backends keep render targets on the device; the display reads guest memory.
                if self.nv.gpu.flush_renderers(&mut self.mem)? == crate::gpu::renderer::Flush::Done
                {
                    self.nv.gpu.present(&self.mem, &buffer)?;
                    if crate::trace::enabled(crate::trace::Trace::Nv) {
                        crate::traceln!(
                            "[vi] presented shared frame {} from slot {slot}",
                            self.nv.gpu.frames
                        );
                    }
                } else {
                    self.pending_present = Some(buffer);
                }
                let _ = SHARED_BUFFER_ADDR;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetSharedFrameBufferAcquirableEvent: a slot is always free, so it stays signalled.
            8256 => {
                let h = self.alloc_event("vi:shared-buffer", false);
                self.signal_event(h);
                self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
            }
            // Open/Close/Connect/DisconnectSharedLayer, CancelSharedFrameBuffer.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// The system shared buffer's nvmap `(handle, id)`, created on first use.
    /// System-owned, so it is registered here rather than via `NVMAP_IOC_CREATE`.
    fn shared_buffer_object(&mut self) -> (u32, u32) {
        if let Some(pair) = self.shared_buffer {
            return pair;
        }
        // Sized for the docked geometry, since docking can happen later.
        let size = crate::cpu::SHARED_BUFFER_RESERVED_SIZE;
        let addr = crate::cpu::SHARED_BUFFER_ADDR;
        let handle = self.nv.gpu.nvmap.create(size);
        let _ = self.nv.gpu.nvmap.alloc(handle, 0, 0, 0x1000, 0, addr);
        let id = self.nv.gpu.nvmap.get(handle).map(|h| h.id).unwrap_or(0);
        self.shared_buffer = Some((handle, id));
        (handle, id)
    }

    /// Write one element into the out buffer if it fits.
    fn vi_fill_out_buffer(&mut self, tls: u32, entry: &[u8]) {
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return;
        };
        for (i, &b) in entry.iter().take(size as usize).enumerate() {
            let _ = self.mem.write_u8(addr.wrapping_add(i as u32), b);
        }
    }

    /// An empty success, traced under `TRACE_IPC` in case the command had an out parameter.
    fn vi_unhandled(&mut self, tls: u32, iface: &str, cmd_id: Option<u32>) -> Result<()> {
        if crate::trace::enabled(crate::trace::Trace::Ipc) {
            crate::traceln!("[ipc] no implementation: {iface} cmd={cmd_id:?}");
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// Hand out a non-domain vi session as a move handle.
    pub(crate) fn vi_out_session(&mut self, tls: u32, iface: &str) -> Result<()> {
        let h = self.alloc_handle();
        self.record_handle(h, "vi:m");
        self.vi_ifaces.insert(h, iface.to_owned());
        self.write_ipc_response(tls, 0, &[h], &[], &[])
    }

    /// IHOSBinderDriver `TransactParcel`: one `IGraphicBufferProducer`
    /// transaction. Request data is `{ s32 session_id, u32 code, u32 flags }`
    /// with the parcel in a map-alias send buffer; the reply goes to the
    /// receive buffer.
    pub(crate) fn vi_transact_parcel(&mut self, tls: u32) -> Result<()> {
        let data = self.ipc_request_data(tls);
        let code = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
        let (send, recv) = self.ipc_buffers(tls);
        let request = match send.first() {
            Some(&(addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };

        let (reply, action) = self.display.transact(code, &request);
        if crate::trace::enabled(crate::trace::Trace::Nv) {
            crate::traceln!(
                "[vi] transact code={code} in={} out={} bytes",
                request.len(),
                reply.len()
            );
        }
        if let crate::display::Action::Present(buffer) = action {
            // A device-backed surface may not be ready yet;
            // `Cpu::complete_pending_present` presents it later.
            if self.nv.gpu.flush_renderers(&mut self.mem)? == crate::gpu::renderer::Flush::Done {
                self.nv.gpu.present(&self.mem, &buffer)?;
                if crate::trace::enabled(crate::trace::Trace::Nv) {
                    crate::traceln!(
                        "[vi] presented frame {} ({}x{})",
                        self.nv.gpu.frames,
                        buffer.width,
                        buffer.height
                    );
                }
            } else {
                self.pending_present = Some(buffer);
            }
            // Paced to the refresh rate either way.
            self.pace_present();
        }

        if let Some(&(addr, size)) = recv.first() {
            for (i, &byte) in reply.iter().take(size as usize).enumerate() {
                self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// The native-window parcel `OpenLayer` (2020) and `CreateStrayLayer` (2030)
    /// return: a full 0x28-byte `flat_binder_object` naming the layer's
    /// `IGraphicBufferProducer` (`nnSdk` checks the interface name), then the
    /// object offset table. `out_size` is the reply data word count.
    pub(crate) fn vi_native_window(&mut self, tls: u32, out_size: usize) -> Result<()> {
        /// The one `IGraphicBufferProducer` every layer shares.
        const BINDER_ID: u64 = 1;

        let mut payload = Vec::with_capacity(0x28);
        payload.extend_from_slice(&2u32.to_le_bytes()); // type: a binder handle
        payload.extend_from_slice(&0u32.to_le_bytes()); // flags
        payload.extend_from_slice(&BINDER_ID.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // cookie
        payload.extend_from_slice(b"dispdrv\0"); // the interface's name
        payload.extend_from_slice(&0u64.to_le_bytes()); // trailing pad
        let objects = 0u32.to_le_bytes(); // one object, at payload offset 0

        let payload_off = 16u32;
        let objects_off = payload_off + payload.len() as u32;
        let parcel_size = objects_off + objects.len() as u32;
        let mut parcel = Vec::with_capacity(parcel_size as usize);
        parcel.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        parcel.extend_from_slice(&payload_off.to_le_bytes());
        parcel.extend_from_slice(&(objects.len() as u32).to_le_bytes());
        parcel.extend_from_slice(&objects_off.to_le_bytes());
        parcel.extend_from_slice(&payload);
        parcel.extend_from_slice(&objects);

        let mut raw = Vec::with_capacity(out_size);
        if out_size >= 16 {
            // 2030: { layer_id, native_window_size }
            raw.extend_from_slice(&LAYER_ID.to_le_bytes());
            raw.extend_from_slice(&(parcel_size as u64).to_le_bytes());
        } else {
            // 2020: native_window_size
            raw.extend_from_slice(&(parcel_size as u64).to_le_bytes());
        }

        if let Some(buf) = self.ipc_output_buffer_addr(tls, 0) {
            for (i, &b) in parcel.iter().enumerate() {
                let _ = self.mem.write_u8(buf.wrapping_add(i as u32), b);
            }
        }
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    /// The buffer queue's event: one per process, created signalled and
    /// manual-reset because a buffer is always free.
    fn vi_binder_event(&mut self) -> u64 {
        match self.binder_event {
            Some(h) => h,
            None => {
                let h = self.alloc_event("vi:binder", false);
                self.signal_event(h);
                self.binder_event = Some(h);
                h
            }
        }
    }
}
