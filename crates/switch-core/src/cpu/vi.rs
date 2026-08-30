//! `vi`: the display service, the layers on it, and the `IHOSBinderDriver`
//! parcels that carry Android's buffer queue underneath.
//!
//! A frame reaches the screen through here — `vi:m`/`vi:u` open a display, a
//! layer on it produces a binder, and the parcel transactions on that binder
//! are what queue and dequeue the buffers [`crate::display`] presents.

use super::Cpu;
use crate::Result;

/// The refresh rate `ListDisplayModes` reports, in Hz.
const DISPLAY_REFRESH_HZ: f32 = 60.0;

/// That display's name. `OpenDisplay` takes it as a 0x40-byte string and
/// `ListDisplays` hands it back; there is no second display to name.
const DISPLAY_NAME: &str = "Default";

/// The id `OpenDisplay` returns and every later display command carries back.
const DISPLAY_ID: u64 = 1;

/// The id of the one layer, shared by `OpenLayer`, `CreateStrayLayer` and
/// `CreateManagedLayer` — they all end up on the single buffer queue.
const LAYER_ID: u64 = 1;

impl Cpu {
    pub(super) fn vi_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        // Control requests: cmd 0 = ConvertToDomain, cmd 3 =
        // QueryPointerBufferSize. Older libnx always converts the session to a
        // domain before dispatching; hbmenu's libnx (NX_SERVICE_ASSUME_NON_DOMAIN)
        // instead sends cmd 3 and then uses raw non-domain requests.
        //
        // Control-ness is `ipc_is_control_request`, never `type == 5`: a
        // control message has a with-context encoding too (type 7), and that
        // is the one `nnSdk` sends. Testing for 5 alone read the Home Menu's
        // QueryPointerBufferSize as command **3 on the binder relay** and ran
        // a parcel transaction for it, which answered a size query with a
        // failed binder reply.
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
        // The display commands answer the same way on either dialect and on
        // whichever sub-interface they arrive at, so they are dispatched ahead
        // of the object getters below rather than duplicated into four arms.
        // None of their command ids collide with a getter's.
        if let Some(done) = self.vi_common_command(tls, cmd_id) {
            return done;
        }
        if !is_domain {
            // Non-domain (NX_SERVICE_ASSUME_NON_DOMAIN) sessions marshal output
            // objects as move handles. Dispatch on the sub-interface (tracked per
            // handle); unknown handles default to the vi root.
            // Owned rather than borrowed: the arms below take `&mut self`.
            let iface = self
                .vi_ifaces
                .get(&handle)
                .cloned()
                .unwrap_or_else(|| "vi:root".to_owned());
            match iface.as_str() {
                // IHOSBinderDriverRelay: the binder protocol — TransactParcel
                // (0), AdjustRefcount (1), GetNativeHandle (2),
                // TransactParcelAuto (3).
                //
                // 0 and 3 are the same transaction; they differ only in how
                // the parcel is marshalled, and `ipc_buffers` reads either
                // form. 3 arrived in 3.0.0, so a caller built against an SDK
                // older than that sends 0 and only 0 — Just Dance 2017 does,
                // and answering it with an empty success queued every frame it
                // rendered into nothing.
                "vi:ihosbd" => match cmd_id {
                    Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    // GetNativeHandle: the buffer queue's own event, and a
                    // **copy** handle like every other event a service hands
                    // out. Sent in the move slot it arrives as 0.
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
                    // GetDisplayVsyncEvent, on a session that never became a
                    // domain. It used to hand back a bare handle in the *move*
                    // slot and register nothing: a copy handle read from the
                    // move slot is 0, and `signal_vsync` had no event to fire
                    // even if the caller had got one. So a render loop paced
                    // by vsync was waiting on handle 0 forever — and only kept
                    // running at all because a wait on an unknown handle is
                    // answered as satisfied.
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
                    // GetDisplayVsyncEvent: a real copy handle, signalled
                    // once per presented frame by `Cpu::signal_vsync`.
                    Some(5202) => {
                        let h = self.alloc_event("vi:vsync", true);
                        self.vsync_event = Some(h);
                        self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                    }
                    _ => self.vi_unhandled(tls, "vi:iads", cmd_id),
                },
                // The binder relay does the same work on a domain session as
                // on a plain one. Answering `TransactParcel` with an empty
                // success instead meant a caller that converted its vi session
                // to a domain — which is what libnx does by default — queued
                // every frame into nothing and presented none of them.
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

    /// The `vi` commands that answer with data rather than with an object.
    ///
    /// They are the same on `IApplicationDisplayService`,
    /// `ISystemDisplayService` and `IManagerDisplayService` — the command ids
    /// do not overlap — and identical on a domain session and a plain one, so
    /// dispatching them once ahead of the sub-interface match beats writing
    /// each of them four times.
    ///
    /// The shape of the bug this closes is worth stating: a command with an
    /// `out` parameter answered by an *empty success* is worse than a refusal.
    /// `ListDisplayModes` did exactly that, so the Home Menu read its mode
    /// count out of whatever the previous reply had left in the TLS data area
    /// and then walked a buffer nothing had written — a billion instructions
    /// of spinning without a single syscall, and no way to tell from the
    /// outside that a display query was where it went wrong.
    ///
    /// Returns `None` for anything not answered here, so the caller falls
    /// through to its own dispatch.
    fn vi_common_command(&mut self, tls: u32, cmd_id: Option<u32>) -> Option<Result<()>> {
        let (width, height) = self.operation_mode().display_size();
        let raw: Vec<u8> = match cmd_id? {
            // ListDisplays: the one display, and a count of one.
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
            // GetDisplayResolution, on the application and manager interfaces
            // alike.
            1102 => {
                let mut raw = Vec::with_capacity(0x10);
                raw.extend_from_slice(&u64::from(width).to_le_bytes());
                raw.extend_from_slice(&u64::from(height).to_le_bytes());
                raw
            }
            // GetZOrderCountMin / GetZOrderCountMax: one layer stack, so 0 is
            // both the lowest and the highest z a layer can take.
            1200 | 1202 => 0i64.to_le_bytes().to_vec(),
            // GetDisplayLogicalResolution: the same size, as two s32.
            1203 => {
                let mut raw = Vec::with_capacity(8);
                raw.extend_from_slice(&(width as i32).to_le_bytes());
                raw.extend_from_slice(&(height as i32).to_le_bytes());
                raw
            }
            // CreateManagedLayer: the managed form of the same single layer.
            2010 => LAYER_ID.to_le_bytes().to_vec(),
            // _viOpenLayer (2020) / _viCreateStrayLayer (2030 / 2012 / 2312):
            // fill the native-window receive buffer with a Binder parcel whose
            // payload[2] is the IGraphicBufferProducer binder id, and return
            // the parcel size. viCreateLayer parses exactly that.
            2020 => return Some(self.vi_native_window(tls, 8)),
            2030 | 2012 | 2312 => return Some(self.vi_native_window(tls, 16)),
            // ConvertScalingMode: every nn mode this composes ends up as
            // ScalingMode_PreserveAspectRatio.
            2102 => 2u64.to_le_bytes().to_vec(),
            // GetLayerZ.
            2204 => 0u64.to_le_bytes().to_vec(),
            // ListDisplayModes: one mode, the display's own.
            3000 => {
                let mut mode = [0u8; 0x10];
                mode[0..4].copy_from_slice(&width.to_le_bytes());
                mode[4..8].copy_from_slice(&height.to_le_bytes());
                mode[8..12].copy_from_slice(&DISPLAY_REFRESH_HZ.to_le_bytes());
                self.vi_fill_out_buffer(tls, &mode);
                1u64.to_le_bytes().to_vec()
            }
            // ListDisplayRgbRanges / ListDisplayContentTypes: one entry each,
            // and 0 is the automatic setting in both enums.
            3001 | 3002 => {
                self.vi_fill_out_buffer(tls, &0u32.to_le_bytes());
                1u64.to_le_bytes().to_vec()
            }
            // GetDisplayMode: that one mode, by value.
            3200 => {
                let mut raw = Vec::with_capacity(0x10);
                raw.extend_from_slice(&width.to_le_bytes());
                raw.extend_from_slice(&height.to_le_bytes());
                raw.extend_from_slice(&DISPLAY_REFRESH_HZ.to_le_bytes());
                raw.extend_from_slice(&0u32.to_le_bytes());
                raw
            }
            // GetDisplayUnderscan: none, on a panel that is not a television.
            3202 => 0i64.to_le_bytes().to_vec(),
            // GetDisplayContentType / GetDisplayRgbRange / GetDisplayCmuMode:
            // all automatic, which is 0 in each of the three enums.
            3204 | 3206 | 3208 => 0u32.to_le_bytes().to_vec(),
            // GetDisplayContrastRatio: unadjusted.
            3210 => 1.0f32.to_le_bytes().to_vec(),
            // The system shared buffer, which is how the Home Menu and every
            // system applet actually draw. See [`Cpu::vi_shared_buffer`].
            8225 | 8250 | 8251 | 8252 | 8253 | 8254 | 8255 | 8256 | 8258 => {
                return Some(self.vi_shared_buffer(tls, cmd_id?));
            }
            _ => return None,
        };
        Some(self.write_ipc_response(tls, 0, &[], &raw, &[]))
    }

    /// `ISystemDisplayService`'s shared-buffer commands.
    ///
    /// AM hands the system's applets one buffer between them rather than a
    /// layer each: an applet asks for a slot, renders into it and presents the
    /// slot back. This is the path the Home Menu takes, and it takes it the
    /// moment `IsSystemBufferSharingEnabled` succeeds — refuse that and it
    /// falls back to building a swapchain of its own, which it then never
    /// draws a single triangle into.
    fn vi_shared_buffer(&mut self, tls: u32, cmd_id: u32) -> Result<()> {
        use super::{SHARED_BUFFER_ADDR, SHARED_BUFFER_SLOTS, SHARED_BUFFER_USABLE_SLOTS};
        // The shared layer's geometry, which is not the display's and does
        // not follow the dock — see [`super::SHARED_BUFFER_GEOMETRY`].
        let mode = super::SHARED_BUFFER_GEOMETRY;
        let (shared_width, shared_height) = mode.display_size();
        let slot_size = mode.shared_buffer_slot_size();
        /// `NvMultiFence`: a count and four `{ id, value }` pairs.
        const FENCE_SIZE: usize = 4 + 4 * 8;
        match cmd_id {
            // GetSharedBufferMemoryHandleId(u64 buffer_id, aruid) ->
            // s32 nvmap_handle, u64 size, and the pool layout in the out
            // buffer. The buffer is ours, not the guest's, so this is where it
            // comes into being and gets an nvmap handle to be mapped by.
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
                if self.trace_nv {
                    eprintln!(
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
            // AcquireSharedFrameBuffer(u64 layer_id) -> fence, s32 slots[4],
            // s64 target slot. The fence is empty: whatever was drawn into the
            // slot last time has already been scanned out.
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
            // s32 swap interval, u64 layer_id, s64 slot). The slot is the last
            // field, and it is the frame.
            //
            // `android::Fence` is a count and four `{ id, value }` pairs — 36
            // bytes, not 40 — so the crop starts at 0x24 and the transform is
            // at 0x34. Reading the transform one field along lands on the swap
            // interval, which the Home Menu queues as 1 and which decodes as
            // `FLIP_H` on a frame that is plainly not mirrored.
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
                if self.trace_nv {
                    eprintln!(
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
                // A GPU backend holds its render targets on the device; the
                // display reads them out of guest memory.
                if self.nv.gpu.flush_renderers(&mut self.mem)? == crate::gpu::renderer::Flush::Done
                {
                    self.nv.gpu.present(&self.mem, &buffer)?;
                    if self.trace_nv {
                        eprintln!(
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
            // GetSharedFrameBufferAcquirableEvent: a slot is always free, so
            // the event is signalled and stays that way. An applet waits on it
            // before every acquire.
            8256 => {
                let h = self.alloc_event("vi:shared-buffer", false);
                self.signal_event(h);
                self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
            }
            // Open/Close/Connect/DisconnectSharedLayer, CancelSharedFrameBuffer:
            // there is one shared layer and it is always connected.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// The system shared buffer's nvmap `(handle, id)`, creating it on first
    /// use. Unlike every other nvmap object this one is not the guest's — the
    /// system owns it and the applet is only lent slots in it — so it is
    /// registered here rather than through `NVMAP_IOC_CREATE`.
    fn shared_buffer_object(&mut self) -> (u32, u32) {
        if let Some(pair) = self.shared_buffer {
            return pair;
        }
        // Reserved for the docked geometry whatever mode this is: the buffer
        // is created once and the console can be docked afterwards.
        let size = super::SHARED_BUFFER_RESERVED_SIZE;
        let addr = super::SHARED_BUFFER_ADDR;
        let handle = self.nv.gpu.nvmap.create(size);
        let _ = self.nv.gpu.nvmap.alloc(handle, 0, 0, 0x1000, 0, addr);
        let id = self.nv.gpu.nvmap.get(handle).map(|h| h.id).unwrap_or(0);
        self.shared_buffer = Some((handle, id));
        (handle, id)
    }

    /// Write one element into the request's out buffer, if the caller left
    /// room for it. Nothing `vi` lists here has a second entry, so one element
    /// is the whole list.
    fn vi_fill_out_buffer(&mut self, tls: u32, entry: &[u8]) {
        let Some((addr, size)) = self.ipc_recv_buffer(tls, 0) else {
            return;
        };
        for (i, &b) in entry.iter().take(size as usize).enumerate() {
            let _ = self.mem.write_u8(addr.wrapping_add(i as u32), b);
        }
    }

    /// Answer a `vi` command nothing implements.
    ///
    /// It cannot refuse: most of what lands here is a void setter, and an
    /// empty success is the right answer for those. But it says so under
    /// `TRACE_IPC`, because when the command did have an out parameter this
    /// line is the only place the silence becomes visible.
    fn vi_unhandled(&mut self, tls: u32, iface: &str, cmd_id: Option<u32>) -> Result<()> {
        if crate::env_flag!("TRACE_IPC") {
            eprintln!("[ipc] no implementation: {iface} cmd={cmd_id:?}");
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// Non-domain vi session hand-out: allocate a fresh handle, record it as a
    /// vi session so later SendSyncRequests route back here, and return it as a
    /// move handle (how NX_SERVICE_ASSUME_NON_DOMAIN marshals output objects).
    pub(super) fn vi_out_session(&mut self, tls: u32, iface: &str) -> Result<()> {
        let h = self.alloc_handle();
        self.record_handle(h, "vi:m");
        self.vi_ifaces.insert(h, iface.to_owned());
        self.write_ipc_response(tls, 0, &[h], &[], &[])
    }

    /// IHOSBinderDriver `TransactParcel`: run one `IGraphicBufferProducer`
    /// transaction against the app's buffer queue.
    ///
    /// The request data is `{ s32 session_id, u32 code, u32 flags }` followed
    /// by the incoming parcel in a map-alias send buffer; the reply parcel
    /// goes into the receive buffer. When the app queues a finished frame, the
    /// GPU scans that buffer out — this is where a rendered frame becomes
    /// something the host can display.
    pub(super) fn vi_transact_parcel(&mut self, tls: u32) -> Result<()> {
        let data = self.ipc_request_data(tls);
        let code = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
        let (send, recv) = self.ipc_buffers(tls);
        let request = match send.first() {
            Some(&(addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };

        let (reply, action) = self.display.transact(code, &request);
        if self.trace_nv {
            // The binder transaction code, not the IPC command: this is the
            // level a stuck buffer-queue loop shows up at.
            eprintln!(
                "[vi] transact code={code} in={} out={} bytes",
                request.len(),
                reply.len()
            );
        }
        if let crate::display::Action::Present(buffer) = action {
            // A GPU backend holds its render targets on the device; the
            // display reads them out of guest memory.
            // A backend holding this surface on a device may not be able to
            // hand it back yet; `Cpu::complete_pending_present` puts the frame
            // up when it can. The software rasterizer is always `Done`.
            if self.nv.gpu.flush_renderers(&mut self.mem)? == crate::gpu::renderer::Flush::Done {
                self.nv.gpu.present(&self.mem, &buffer)?;
                if self.trace_nv {
                    eprintln!(
                        "[vi] presented frame {} ({}x{})",
                        self.nv.gpu.frames, buffer.width, buffer.height
                    );
                }
            } else {
                self.pending_present = Some(buffer);
            }
            // Paced either way: what the guest is being held to is the
            // refresh rate, not how fast a readback happens to land.
            self.pace_present();
        }

        if let Some(&(addr, size)) = recv.first() {
            for (i, &byte) in reply.iter().take(size as usize).enumerate() {
                self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// viOpenLayer / viCreateStrayLayer reply: fill the request's native-window
    /// receive buffer with a Binder parcel (ParcelHeader + payload whose third
    /// word is the IGraphicBufferProducer binder object id), then return the
    /// parcel size. `out_size` is the number of reply data words (8 for 2020's
    /// single u64, 16 for 2030's layer_id+size pair).
    /// The native-window parcel `OpenLayer` (2020) and `CreateStrayLayer`
    /// (2030) hand back: an Android `Parcel` holding one flattened binder
    /// object that names the layer's `IGraphicBufferProducer`.
    ///
    /// libnx only reads the binder id out of it, but `nnSdk` also checks the
    /// interface name, so the object is written in full: the 0x28-byte
    /// `flat_binder_object` real `vi` sends, followed by the four-byte object
    /// offset table the parcel header points at.
    pub(super) fn vi_native_window(&mut self, tls: u32, out_size: usize) -> Result<()> {
        /// The binder handle every layer here shares — `vi_transact_parcel`
        /// serves the one `IGraphicBufferProducer` this emulator has.
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

        if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
            for (i, &b) in parcel.iter().enumerate() {
                let _ = self.mem.write_u8(buf.wrapping_add(i as u32), b);
            }
        }
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    /// The buffer queue's own event, which a producer waits on before it
    /// dequeues. It is created **signalled** and manual-reset: the queue here
    /// always has a free buffer — `dequeueBuffer` never refuses — so the state
    /// it reports is "a buffer is available", permanently and truthfully.
    ///
    /// One object per process rather than one per `GetNativeHandle`, because a
    /// caller that asks twice has to be given the event it is already waiting
    /// on.
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
