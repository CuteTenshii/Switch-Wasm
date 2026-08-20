//! Horizon IPC: parsing CMIF/HIPC requests out of the TLS message buffer,
//! synthesizing replies, and the service implementations behind the
//! session handles homebrew opens (`sm:`, `fsp-srv`, `vi:m`, `nvdrv`).

use super::{AudrenParams, Cpu};
use crate::Result;

impl Cpu {
    /// Offset of a CMIF request's `SFCI` header inside the TLS message buffer.
    ///
    /// Where it lands depends on how many descriptors the request carries: the
    /// data area follows the message header, the optional pid, and the static
    /// and buffer descriptors. A `KICKOFF_PB` with its gpfifo-entry buffers puts
    /// it at 0x40 — a fixed scan of the first 0x40 bytes missed it entirely, so
    /// the submit was dispatched as "unknown command", answered with a generic
    /// success, and the GPU never saw the frame.
    pub(super) fn ipc_cmif_header_offset(&self, tls: u32) -> Option<u32> {
        const SFCI: u32 = 0x4943_4653;
        // The computed data area, then the same 0x10 further in for a domain
        // request (which puts a `CmifDomainInHeader` first).
        let start = self.ipc_reply_start(tls);
        for candidate in [start, start.wrapping_add(0x10)] {
            if self.mem.read_u32(tls.wrapping_add(candidate)).unwrap_or(0) == SFCI {
                return Some(candidate);
            }
        }
        // Otherwise search the whole 0x100-byte message buffer, for layouts the
        // descriptor walk above doesn't model exactly.
        (0..0x100u32)
            .step_by(4)
            .find(|&i| self.mem.read_u32(tls.wrapping_add(i)).unwrap_or(0) == SFCI)
    }

    /// The command id a CMIF request carries (`CmifInHeader::command_id`), or
    /// `None` when the buffer doesn't look like a CMIF request.
    pub(super) fn ipc_command_id(&self, tls: u32) -> Option<u32> {
        if let Some(offset) = self.ipc_cmif_header_offset(tls) {
            return self.mem.read_u32(tls.wrapping_add(offset + 8)).ok();
        }
        // Older libnx (pre-CMIF) sessions — e.g. the FsDir object NX-Shell's
        // `fsDirRead` uses — marshal requests as {type=2, object_id, cmd_id,
        // ...} with no SFCI magic. Fall back to reading the command there.
        let start = self.ipc_reply_start(tls);
        if self.mem.read_u32(tls.wrapping_add(start)).unwrap_or(0) == 2 {
            return self.mem.read_u32(tls.wrapping_add(start + 8)).ok();
        }
        None
    }

    /// Compute where the reply starts in the TLS IPC buffer, mirroring libnx's
    /// `cmifGetAlignedDataStart`: walk the request's hipc header (16-byte
    /// message header, optional special header + pid, then buffer descriptors)
    /// to the data area, and round up to 16 bytes.
    pub(super) fn ipc_reply_start(&self, tls: u32) -> u32 {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        let num_recv_buffers = (hdr1 >> 24) & 0xf;
        let num_exch_buffers = (hdr1 >> 28) & 0xf;
        let has_special = (hdr2 >> 31) & 1;
        let mut data_off = 8u32;
        if has_special != 0 {
            data_off += 4;
            let sphdr = self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0);
            if sphdr & 1 != 0 {
                data_off += 8; // pid
            }
        }
        data_off += 8 * num_send_statics;
        data_off += 12 * (num_send_buffers + num_recv_buffers + num_exch_buffers);
        (data_off + 15) & !15
    }

    /// Address of a CMIF request's raw payload — the bytes after the 16-byte
    /// `CmifInHeader`.
    ///
    /// Its distance from the data area is not fixed: a domain request carries a
    /// `CmifDomainInHeader` in front of the `CmifInHeader`, so the payload sits
    /// 0x20 rather than 0x10 bytes in. Locating the "SFCI" magic covers both.
    /// (Assuming 0x10 made `fsFileRead` on the domain session libnx uses for
    /// `fsp-srv` read its offset and size out of the header, so every read
    /// asked for 0 bytes at offset 0 and `romfsMountSelf` failed.)
    pub(super) fn ipc_request_data(&self, tls: u32) -> u32 {
        match self.ipc_cmif_header_offset(tls) {
            Some(offset) => tls.wrapping_add(offset + 0x10),
            None => tls.wrapping_add(self.ipc_reply_start(tls) + 0x10),
        }
    }

    pub(super) fn alloc_handle(&mut self) -> u64 {
        let h = self.next_handle as u64;
        self.next_handle = self.next_handle.wrapping_add(1);
        h
    }

    /// Address of the `index`-th receive buffer in a hipc request, from its
    /// buffer descriptors (walk the same layout as [`Cpu::ipc_reply_start`]).
    /// Returns `None` if the request has no such buffer.
    pub(super) fn ipc_recv_buffer_addr(&self, tls: u32, index: u32) -> Option<u32> {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        let num_recv_buffers = (hdr1 >> 24) & 0xf;
        let _num_exch_buffers = (hdr1 >> 28) & 0xf;
        if index >= num_recv_buffers {
            return None;
        }
        let has_special = (hdr2 >> 31) & 1;
        let mut off = 8u32;
        if has_special != 0 {
            off += 4;
            let sphdr = self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0);
            if sphdr & 1 != 0 {
                off += 8; // pid
            }
        }
        off += 8 * num_send_statics;
        off += 12 * (num_send_buffers + index);
        let address_low = self.mem.read_u32(tls.wrapping_add(off + 4)).unwrap_or(0);
        let desc = self.mem.read_u32(tls.wrapping_add(off + 8)).unwrap_or(0);
        let addr_mid = (desc >> 4) & 0xf;
        let addr_high = (desc >> 6) & 0x3F_FFFF;
        let addr = (address_low as u64) | ((addr_mid as u64) << 32) | ((addr_high as u64) << 36);
        Some(addr as u32)
    }

    /// The send-static ("pointer") buffers of a hipc request, as
    /// `(address, size)`.
    ///
    /// Each descriptor is two words: `{ index:6, address_high:6,
    /// address_mid:4, size:16 }` then the low 32 bits of the address. Services
    /// that take a path — all of `fsp-srv`'s — send it this way rather than as
    /// a map-alias buffer.
    pub(super) fn ipc_static_buffers(&self, tls: u32) -> Vec<(u32, u32)> {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let count = (hdr1 >> 16) & 0xf;
        let mut off = 8u32;
        if (hdr2 >> 31) & 1 != 0 {
            off += 4;
            if self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0) & 1 != 0 {
                off += 8; // pid
            }
        }
        (0..count)
            .map(|index| {
                let at = off + 8 * index;
                let word = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
                let address = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
                (address, word >> 16)
            })
            .collect()
    }

    /// The NUL-terminated path a filesystem request sent in its first
    /// static buffer, normalized to a leading-slash form with no trailing
    /// slash (`sdmc:/switch/` becomes `/switch`).
    pub(super) fn ipc_request_path(&self, tls: u32) -> String {
        let raw = match self.ipc_static_buffers(tls).first() {
            Some(&(addr, size)) => self.read_string(addr, size.min(0x301)),
            None => return String::new(),
        };
        let without_device = match raw.split_once(":/") {
            Some((_, rest)) => rest,
            None => raw.trim_start_matches('/'),
        };
        let trimmed = without_device.trim_matches('/');
        if trimmed.is_empty() { "/".to_owned() } else { format!("/{}", trimmed) }
    }

    /// Hand a sub-interface back to the caller the way its session expects.
    ///
    /// A domain session (libnx converts `fsp-srv` to one) takes an out-object
    /// id in the response's domain header; a plain session — libtransistor
    /// never converts, so sdl-hello's `fsp-srv` is one — takes a real session
    /// handle as a move handle, and validates the count, so answering with a
    /// domain object made `fsp_srv_open_sd_card_filesystem` fail. Returns the
    /// key the new object's state is filed under.
    pub(super) fn reply_with_interface(
        &mut self,
        tls: u32,
        handle: u64,
        name: &str,
    ) -> Result<u64> {
        if self.ipc_is_domain_request(tls) {
            let obj = self.alloc_domain_object();
            self.record_domain_object(handle, obj, name);
            self.write_ipc_response(tls, 0, &[], &[], &[obj])?;
            Ok(Self::object_key(handle, obj))
        } else {
            let sub = self.alloc_handle();
            self.record_handle(sub, name);
            self.write_ipc_response(tls, 0, &[sub], &[], &[])?;
            Ok(Self::object_key(sub, 0))
        }
    }

    /// Key for the per-object state maps (`fs_files`, `fs_dirs`). A domain
    /// object id is only unique within its session, and a plain sub-session is
    /// identified by its own handle, so both go in as `handle:object_id`.
    pub(super) fn object_key(handle: u64, object_id: u32) -> u64 {
        (handle << 32) | u64::from(object_id)
    }

    pub(super) fn record_handle(&mut self, handle: u64, name: &str) {
        self.service_handles.insert(handle, name.to_owned());
    }

    /// Drop everything recorded for a session the guest has closed. Handles are
    /// never reused, so this only keeps the tables from growing.
    pub(super) fn forget_handle(&mut self, handle: u64) {
        self.service_handles.remove(&handle);
        self.domain_objects.retain(|&(owner, _), _| owner != handle);
        let session = handle << 32;
        self.fs_files.retain(|&key, _| key & !0xFFFF_FFFF != session);
        self.fs_dirs.retain(|&key, _| key & !0xFFFF_FFFF != session);
    }

    pub(super) fn service_name(&self, handle: u64) -> Option<&str> {
        self.service_handles.get(&handle).map(|s| s.as_str())
    }

    pub(super) fn read_port_name(&self, ptr: u32) -> String {
        let mut name = Vec::new();
        for i in 0..16u32 {
            match self.mem.read_u8(ptr.wrapping_add(i)) {
                Ok(0) | Err(_) => break,
                Ok(b) => name.push(b),
            }
        }
        String::from_utf8_lossy(&name).into_owned()
    }

    pub(super) fn u64_to_service_name(&self, value: u64) -> String {
        let bytes = value.to_le_bytes();
        let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
        String::from_utf8_lossy(&bytes[..len]).into_owned()
    }

    pub(super) fn ipc_message_type(&self, tls: u32) -> u32 {
        self.mem.read_u32(tls).unwrap_or(0) & 0xFFFF
    }

    /// Whether the request is a *control* message — the session-management
    /// commands (ConvertToDomain, Clone, QueryPointerBufferSize) rather than a
    /// command on the interface behind the session.
    ///
    /// There are two encodings of every message kind: the plain one
    /// (`Request` = 4, `Control` = 5) and the "with context" one
    /// (`RequestWithContext` = 6, `ControlWithContext` = 7), which prefixes the
    /// raw data with a 16-byte tracing context. `libnx` sends the plain form;
    /// **`nnSdk` sends the context form for everything**, so testing `== 5`
    /// classified every retail control command as an ordinary command on the
    /// interface. `appletOE`'s very first message is
    /// `QueryPointerBufferSize`, which arrives as type 7 and was being answered
    /// as though it were IApplicationProxyService command 3 — a command that
    /// does not exist — which killed the applet chain before it opened.
    pub(super) fn ipc_is_control_request(&self, tls: u32) -> bool {
        matches!(self.ipc_message_type(tls), 5 | 7)
    }

    /// Whether the request is a domain message. Domain-ness is NOT encoded in
    /// the hipc type field: a domain request still carries type 4
    /// (`CmifCommandType_Request`), and the domain header (`CmifDomainInHeader`)
    /// lives at the start of the data area with its `type` byte set to
    /// `CmifDomainRequestType_SendMessage` (1). Reading it from where
    /// [`Cpu::ipc_reply_start`] puts the data area avoids misfiring on plain
    /// requests that happen to have a nonzero word in the same spot.
    pub(super) fn ipc_is_domain_request(&self, tls: u32) -> bool {
        let start = self.ipc_reply_start(tls);
        self.mem.read_u8(tls.wrapping_add(start)).unwrap_or(0) == 1
    }

    pub(super) fn ipc_domain_object_id(&self, tls: u32) -> u32 {
        let start = self.ipc_reply_start(tls);
        self.mem.read_u32(tls.wrapping_add(start + 4)).unwrap_or(0xFFFFFFFF)
    }

    pub(super) fn alloc_domain_object(&mut self) -> u32 {
        let id = self.next_domain_object_id;
        self.next_domain_object_id = id.wrapping_add(1);
        if id == 0 { 1 } else { id }
    }

    pub(super) fn record_domain_object(&mut self, handle: u64, object_id: u32, name: &str) {
        self.domain_objects.insert((handle, object_id), name.to_owned());
    }

    pub(super) fn domain_interface(&self, handle: u64, object_id: u32) -> Option<&str> {
        self.domain_objects.get(&(handle, object_id)).map(|s| s.as_str())
    }

    /// Write a complete HIPC response into the TLS IPC buffer. `move_handles`
    /// are emitted in the handle descriptor, `raw_data` lands after the
    /// SFCO/result header, and `domain_objects` produces a domain response
    /// (message type 4) with the given out-object ids.
    pub(super) fn write_ipc_response(
        &mut self,
        tls: u32,
        result: u32,
        move_handles: &[u64],
        raw_data: &[u8],
        domain_objects: &[u32],
    ) -> Result<()> {
        let is_domain = self.ipc_is_domain_request(tls);
        // A reply's `type` field (bits[15:0] of word 0) is 0: the counts in the
        // rest of the word are what matter. libnx ignores the field entirely,
        // but libtransistor validates it (`type != 0 && type != 4` → its error
        // 0x7E0DD), which is what made sdl-hello's "Failed to open connection
        // to fsp-srv" — a 0x40 here fails that check on every single reply.
        self.mem.write_u32(tls, 0)?;
        let has_handles = !move_handles.is_empty();
        let handle_desc = (move_handles.len() as u32) << 5;
        let raw_data_words = ((raw_data.len() as u32) + 3) / 4;
        let object_words = ((domain_objects.len() as u32) * 4 + 3) / 4;
        // SFCO header (4 words) + raw data + domain header/objects when needed,
        // padded so pre+post = 4 words.
        let mut raw_section_words = 4 + raw_data_words + 4;
        if is_domain {
            raw_section_words += 4 + object_words;
        }
        let mut header1 = raw_section_words;
        if has_handles {
            header1 |= 1 << 31;
        }
        self.mem.write_u32(tls.wrapping_add(4), header1)?;
        let mut off = 8u32;
        if has_handles {
            self.mem.write_u32(tls.wrapping_add(off), handle_desc)?;
            off += 4;
            for &h in move_handles {
                self.mem.write_u32(tls.wrapping_add(off), h as u32)?;
                off += 4;
            }
        }
        // Align to 16 bytes.
        let pre = (16 - (off % 16)) % 16;
        off += pre;
        if is_domain {
            self.mem.write_u32(tls.wrapping_add(off), domain_objects.len() as u32)?;
            self.mem.write_u32(tls.wrapping_add(off + 4), 0)?;
            self.mem.write_u32(tls.wrapping_add(off + 8), 0)?;
            self.mem.write_u32(tls.wrapping_add(off + 12), 0)?;
            off += 16;
        }
        // SFCO header.
        self.mem.write_u32(tls.wrapping_add(off), 0x4F43_4653)?;
        self.mem.write_u32(tls.wrapping_add(off + 4), 0)?;
        self.mem.write_u32(tls.wrapping_add(off + 8), result)?;
        self.mem.write_u32(tls.wrapping_add(off + 12), 0)?;
        off += 16;
        // Raw data.
        for (i, &b) in raw_data.iter().enumerate() {
            self.mem.write_u8(tls.wrapping_add(off + i as u32), b)?;
        }
        off += raw_data_words * 4;
        // Domain object ids.
        for (i, &obj) in domain_objects.iter().enumerate() {
            self.mem.write_u32(tls.wrapping_add(off + (i as u32) * 4), obj)?;
        }
        Ok(())
    }

    pub(super) fn sm_request(&mut self, tls: u32, cmd_id: Option<u32>, _handle: u64) -> Result<()> {
        match cmd_id {
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(1) => {
                // GetService: raw data is the 8-byte service name.
                let name_raw = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0);
                let name = self.u64_to_service_name(name_raw);
                let handle = self.alloc_handle();
                self.record_handle(handle, &name);
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            Some(2) => {
                let handle = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    pub(super) fn fsp_srv_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        match cmd_id {
            // 0 = ConvertToDomain: hand back a fresh domain object id so the
            // session becomes a domain (libnx's serviceConvertToDomain reads it
            // from the out data). All later fsp-srv requests then carry the
            // object id in the CmifDomainInHeader.
            Some(0) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "fsp-srv");
                self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
            }
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // 18 = fsOpenSdCardFileSystem, 11 = fsOpenBisFileSystem: both hand
            // out an FsFileSystem session as a domain out-object.
            Some(18) | Some(11) => {
                self.reply_with_interface(tls, handle, "fsp-srv-fs")?;
                Ok(())
            }
            // 200 = OpenDataStorageByCurrentProcess: hands back the calling
            // title's own RomFS as a raw `IStorage` (offset/size reads only —
            // no paths). libnx's `romfsMount`/`nn::fs::MountRom` parse the
            // RomFS header and directory/file tables entirely in guest code
            // against this; the emulator only has to serve byte ranges.
            Some(200) => {
                if self.romfs.is_none() {
                    // No NCA was decrypted this session (homebrew, or a
                    // title with no RomFS section) — report "not found"
                    // rather than handing out a storage backed by nothing.
                    const PATH_NOT_FOUND: u32 = 2 | (1 << 9);
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                self.reply_with_interface(tls, handle, "fsp-srv-storage")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IStorage`, backed by the current process's decrypted RomFS
    /// ([`Cpu::set_romfs`]). Cmd 0 = Read(u64 offset, u64 size), cmd 4 =
    /// GetSize — the same shape as `IFile`, but offset-addressed rather than
    /// path-addressed since there's exactly one of these per process.
    pub(super) fn fs_storage_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        let romfs = self.romfs.as_deref().unwrap_or(&[]);
        match cmd_id {
            // Read(u64 offset, u64 size) -> bytes into the recv buffer.
            Some(0) => {
                let data = self.ipc_request_data(tls);
                let offset = self.mem.read_u64(data.wrapping_add(8))? as usize;
                let requested = self.mem.read_u64(data.wrapping_add(0x10))? as usize;
                let start = offset.min(romfs.len());
                let end = start.saturating_add(requested).min(romfs.len());
                let chunk = &romfs[start..end];
                if let Some(addr) = self.ipc_recv_buffer_addr(tls, 0) {
                    for (i, &byte) in chunk.iter().enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetSize -> u64
            Some(4) => {
                self.write_ipc_response(tls, 0, &[], &(romfs.len() as u64).to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IFileSystem`, backed by the emulated SD card in [`crate::vfs`].
    ///
    /// Paths arrive in the request's first static buffer, so every command
    /// resolves against the real tree: a missing path reports
    /// `FsError_PathNotFound` rather than pretending to succeed, which is what
    /// stops a menu from recursing forever into directories that do not exist.
    pub(super) fn fs_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        /// Horizon `fs` results: module 2, descriptions 1 (path not found) and
        /// 2 (path already exists).
        const PATH_NOT_FOUND: u32 = 2 | (1 << 9);
        let path = self.ipc_request_path(tls);
        if std::env::var("TRACE_IPC").is_ok() {
            eprintln!("[fs] pc={:#x} cmd={:?} path={:?}", self.pc, cmd_id, path);
        }
        match cmd_id {
            // CreateFile(u64 size, u32 flags) / CreateDirectory
            Some(0) => {
                self.fs.write_file(&path, Vec::new());
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(2) => {
                self.fs.create_dir(&path);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // DeleteFile / DeleteDirectory / DeleteDirectoryRecursively
            Some(1) | Some(3) | Some(4) => {
                if self.fs.remove(&path) {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                } else {
                    self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[])
                }
            }
            // GetEntryType -> FsDirEntryType
            Some(7) => match self.fs.entry_type(&path) {
                Some(kind) => {
                    self.write_ipc_response(tls, 0, &[], &(kind as u32).to_le_bytes(), &[])
                }
                None => self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]),
            },
            // OpenFile(u32 mode) -> IFile
            Some(8) => {
                if self.fs.entry_type(&path) != Some(crate::vfs::ENTRY_TYPE_FILE) {
                    return self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]);
                }
                let key = self.reply_with_interface(tls, handle, "fsp-srv-fs-file")?;
                self.fs_files.insert(key, path);
                Ok(())
            }
            // OpenDirectory(u32 mode) -> IDirectory
            Some(9) => match self.fs.read_dir(&path) {
                Some(entries) => {
                    let key = self.reply_with_interface(tls, handle, "fsp-srv-fs-dir")?;
                    self.fs_dirs.insert(key, entries);
                    Ok(())
                }
                None => self.write_ipc_response(tls, PATH_NOT_FOUND, &[], &[], &[]),
            },
            // GetFreeSpaceSize / GetTotalSpaceSize: report a 32 GiB card.
            Some(11) | Some(12) => {
                let bytes = 32u64 << 30;
                self.write_ipc_response(tls, 0, &[], &bytes.to_le_bytes(), &[])
            }
            // Commit and the remaining bookkeeping commands.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IDirectory`: cmd 0 = `fsDirRead` (fill the out buffer with
    /// `FsDirectoryEntry` structs), cmd 1 = `fsDirGetEntryCount`.
    pub(super) fn fs_dir_request(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
        /// `sizeof(FsDirectoryEntry)`: a 0x301-byte name, padding, the entry
        /// type, more padding, then the 8-aligned size.
        const ENTRY_SIZE: u32 = 0x310;
        match cmd_id {
            Some(0) => {
                let entries = self.fs_dirs.remove(&key).unwrap_or_default();
                if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
                    for (i, entry) in entries.iter().enumerate() {
                        let base = buf.wrapping_add(i as u32 * ENTRY_SIZE);
                        let name = entry.name.as_bytes();
                        for j in 0..0x301u32 {
                            let byte = name.get(j as usize).copied().unwrap_or(0);
                            self.mem.write_u8(base.wrapping_add(j), byte)?;
                        }
                        self.mem.write_u8(base.wrapping_add(0x304), entry.kind)?;
                        self.mem.write_u64(base.wrapping_add(0x308), entry.size)?;
                    }
                }
                let count = entries.len() as u64;
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            Some(1) => {
                let count =
                    self.fs_dirs.get(&key).map(|v| v.len() as u64).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IFile`: cmd 0 = Read, cmd 4 = GetSize.
    pub(super) fn fs_file_request(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
        let path = self.fs_files.get(&key).cloned().unwrap_or_default();
        match cmd_id {
            // Read(u32 option, u64 offset, u64 size) -> u64 bytes_read
            Some(0) => {
                let data = self.ipc_request_data(tls);
                let offset = self.mem.read_u64(data.wrapping_add(8))?;
                let requested = self.mem.read_u64(data.wrapping_add(0x10))? as usize;
                let mut buf = vec![0u8; requested.min(1 << 24)];
                let read = self.fs.read(&path, offset, &mut buf).unwrap_or(0);
                if std::env::var("TRACE_IPC").is_ok() {
                    eprintln!(
                        "[fs-file] read path={:?} offset={:#x} size={:#x} -> {:#x} buf={:?}",
                        path,
                        offset,
                        requested,
                        read,
                        self.ipc_recv_buffer_addr(tls, 0)
                    );
                }
                if let Some(addr) = self.ipc_recv_buffer_addr(tls, 0) {
                    for (i, &byte) in buf[..read].iter().enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &(read as u64).to_le_bytes(), &[])
            }
            // GetSize -> u64
            Some(4) => {
                let size = self.fs.size(&path).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// The `set` service: system language settings.
    ///
    /// `SetLanguage` is an index into this list, and `setMakeLanguage` maps a
    /// language code back to it by searching the array
    /// `GetAvailableLanguageCodes` returns — so the order matters and both
    /// commands have to agree.
    /// `pl:u` (`IPlatformServiceManager`): the shared system fonts.
    ///
    /// A guest asks for the fonts by type, gets back an offset and a size, and
    /// reads the font data straight out of pl's shared memory — hbmenu hands
    /// that pointer to `FT_New_Memory_Face`. There are no Nintendo fonts here,
    /// so every type resolves to the host-supplied font
    /// ([`Cpu::set_shared_font`]), which the shared memory is filled with when
    /// the guest maps it. Without a font the set is reported as loaded but
    /// empty, which is a well-formed "no font data" answer rather than a guest
    /// spinning on `GetLoadState`.
    pub(super) fn pl_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        let font_size = self.shared_font.len() as u32;
        match cmd_id {
            // RequestLoad(u32 SharedFontType)
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetLoadState(u32) -> u32 (1 = Loaded)
            Some(1) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetSize(u32) -> u32
            Some(2) => self.write_ipc_response(tls, 0, &[], &font_size.to_le_bytes(), &[]),
            // GetSharedMemoryAddressOffset(u32) -> u32: the font sits at the
            // start of the shared memory.
            Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetSharedMemoryNativeHandle -> a shared memory handle;
            // `svcMapSharedMemory` fills the region with the font.
            Some(4) => {
                let handle = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            // GetSharedFontInOrderOfPriority(u64 LanguageCode) ->
            // { u8 Loaded, u8 pad[3], s32 total_fonts }, with the types, the
            // offsets and the sizes of the fonts in three output buffers.
            Some(5) => {
                let (_, recv) = self.ipc_map_buffers(tls);
                let count = if font_size == 0 { 0u32 } else { 1 };
                if count == 1 {
                    // PlSharedFontType_Standard, at offset 0.
                    for (buffer, value) in recv.iter().zip([0u32, 0, font_size]) {
                        let (addr, size) = *buffer;
                        if size >= 4 {
                            self.mem.write_u32(addr, value)?;
                        }
                    }
                }
                let mut raw = [0u8; 8];
                raw[0] = 1; // Loaded
                raw[4..].copy_from_slice(&count.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    pub(super) fn set_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        // Language codes in `SetLanguage` order, as NUL-padded ASCII in a u64.
        const LANGUAGE_CODES: [&str; 18] = [
            "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW",
            "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR",
        ];
        // The language the emulated console is set to (`SetLanguage_ENUS`).
        const SYSTEM_LANGUAGE: usize = 1;

        let code = |index: usize| -> u64 {
            let mut packed = [0u8; 8];
            let name = LANGUAGE_CODES[index].as_bytes();
            packed[..name.len()].copy_from_slice(name);
            u64::from_le_bytes(packed)
        };

        match cmd_id {
            // GetRegionCode -> SetRegion (SetRegion_USA), paired with
            // SYSTEM_LANGUAGE (en-US) above rather than a separate constant.
            Some(4) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetLanguageCode
            Some(0) => {
                let raw = code(SYSTEM_LANGUAGE).to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // MakeLanguageCode(SetLanguage) -> u64 code
            Some(2) => {
                let language = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                let index = (language as usize).min(LANGUAGE_CODES.len() - 1);
                self.write_ipc_response(tls, 0, &[], &code(index).to_le_bytes(), &[])
            }
            // GetAvailableLanguageCodes: fill the out buffer with the codes and
            // return how many were written.
            Some(5) => {
                let (_, recv) = self.ipc_map_buffers(tls);
                let mut written = 0usize;
                if let Some(&(addr, size)) = recv.first() {
                    written = (size as usize / 8).min(LANGUAGE_CODES.len());
                    for index in 0..written {
                        self.mem.write_u64(addr.wrapping_add((index * 8) as u32), code(index))?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &(written as u32).to_le_bytes(), &[])
            }
            // GetAvailableLanguageCodeCount (3 = pre-4.0.0, 6 = current).
            Some(3) | Some(6) => {
                let total = LANGUAGE_CODES.len() as u32;
                self.write_ipc_response(tls, 0, &[], &total.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `set:sys` — system settings not covered by the plain `set` service
    /// above (language codes). Only `GetSerialNumber` is implemented; every
    /// other command falls through to a generic empty-success reply, same as
    /// `set_request`'s default arm.
    pub(super) fn set_sys_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // GetSerialNumber -> SetSysSerialNumber { char number[0x18] }.
            // Real hardware's is burned in at manufacturing and unique per
            // console; this is a fixed placeholder, not a real serial.
            Some(68) => {
                const SERIAL: &[u8] = b"XAW00000000000";
                let mut number = [0u8; 0x18];
                number[..SERIAL.len()].copy_from_slice(SERIAL);
                self.write_ipc_response(tls, 0, &[], &number, &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    pub(super) fn vi_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        let req_type = self.ipc_message_type(tls);
        // Control (type 5) requests: cmd 0 = ConvertToDomain, cmd 3 =
        // QueryPointerBufferSize. Older libnx always converts the session to a
        // domain before dispatching; hbmenu's libnx (NX_SERVICE_ASSUME_NON_DOMAIN)
        // instead sends cmd 3 and then uses raw non-domain requests.
        if req_type == 5 {
            return match cmd_id {
                Some(0) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "vi:root");
                    let raw = obj.to_le_bytes();
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // QueryPointerBufferSize: report 0 so libnx marshals every
                // `SfBufferAttr_HipcAutoSelect` buffer as a map-alias range —
                // the only buffer form this IPC layer implements.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let object_id = if self.ipc_is_domain_request(tls) {
            self.ipc_domain_object_id(tls)
        } else {
            0xFFFFFFFF
        };
        let is_domain = object_id != 0xFFFFFFFF;
        if !is_domain {
            // Non-domain (NX_SERVICE_ASSUME_NON_DOMAIN) sessions marshal output
            // objects as move handles. Dispatch on the sub-interface (tracked per
            // handle); unknown handles default to the vi root.
            let iface = self.vi_ifaces.get(&handle).map(|s| s.as_str()).unwrap_or("vi:root");
            match iface {
                // IHOSBinderDriverRelay: libnx binder protocol — AdjustRefcount
                // (1), GetNativeHandle (2), TransactParcel (3).
                "vi:ihosbd" => match cmd_id {
                    Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                    Some(2) => {
                        let h = self.alloc_handle();
                        self.write_ipc_response(tls, 0, &[h], &[], &[])
                    }
                    Some(3) => self.vi_transact_parcel(tls),
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                // vi root: cmd 2 hands out the IApplicationDisplayService.
                "vi:root" => match cmd_id {
                    Some(2) => self.vi_out_session(tls, "vi:iads"),
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                // IApplicationDisplayService and the other display services.
                _ => match cmd_id {
                    Some(100) => self.vi_out_session(tls, "vi:ihosbd"),
                    Some(101) => self.vi_out_session(tls, "vi:isds"),
                    Some(102) => self.vi_out_session(tls, "vi:imds"),
                    Some(103) => self.vi_out_session(tls, "vi:ihosbdind"),
                    Some(1010) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                    Some(5202) => {
                        let h = self.alloc_handle();
                        self.write_ipc_response(tls, 0, &[h], &[], &[])
                    }
                    // _viOpenLayer (2020) / _viCreateStrayLayer (2030 / 2012 / 2312):
                    // fill the native-window receive buffer with a Binder parcel whose
                    // payload[2] is the IGraphicBufferProducer binder id, and return the
                    // parcel size. viCreateLayer parses exactly that.
                    Some(2020) => self.vi_native_window(tls, 8),
                    Some(2030) | Some(2012) | Some(2312) => self.vi_native_window(tls, 16),
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
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
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
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
                    // OpenDisplay: return a display id of 1.
                    Some(1010) => {
                        let raw = 1u64.to_le_bytes();
                        self.write_ipc_response(tls, 0, &[], &raw, &[])
                    }
                    // GetDisplayVsyncEvent: return a copy handle.
                    Some(5202) => {
                        let h = self.alloc_handle();
                        self.write_ipc_response(tls, 0, &[h], &[], &[])
                    }
                    _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
                },
                Some("vi:ihosbd") | Some("vi:isds") | Some("vi:imds") => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            }
        }
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
        let (send, recv) = self.ipc_map_buffers(tls);
        let request = match send.first() {
            Some(&(addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };

        let (reply, action) = self.display.transact(code, &request);
        if let crate::display::Action::Present(buffer) = action {
            self.nv.gpu.present(&self.mem, &buffer)?;
            if self.trace_nv {
                eprintln!(
                    "[vi] presented frame {} ({}x{})",
                    self.nv.gpu.frames, buffer.width, buffer.height
                );
            }
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
    pub(super) fn vi_native_window(&mut self, tls: u32, out_size: usize) -> Result<()> {
        let payload: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]; // payload[2] = binder id 1
        let parcel_size = 16 + payload.len() as u32;
        let mut parcel = Vec::with_capacity(parcel_size as usize);
        parcel.extend_from_slice(&payload.len().to_le_bytes()); // payload_size
        parcel.extend_from_slice(&16u32.to_le_bytes()); // payload_off
        parcel.extend_from_slice(&0u32.to_le_bytes()); // objects_size
        parcel.extend_from_slice(&parcel_size.to_le_bytes()); // objects_off
        parcel.extend_from_slice(&payload);

        let mut raw = Vec::with_capacity(out_size);
        if out_size >= 16 {
            // 2030: { layer_id, native_window_size }
            raw.extend_from_slice(&1u64.to_le_bytes());
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

    /// The `INvDrvServices` interface: the guest's door to the GPU.
    ///
    /// Command ids follow libnx's `services/nv.c`: 0 Open, 1 Ioctl, 2 Close,
    /// 3 Initialize, 4 QueryEvent, 8 SetClientPID, 11 Ioctl2, 12 Ioctl3.
    /// Every one of them answers with a `u32` NvError (Open also returns the
    /// fd), and the ioctl argument struct travels as a map-alias buffer in
    /// each direction.
    pub(super) fn nvdrv_request(&mut self, tls: u32, cmd_id: Option<u32>, _handle: u64) -> Result<()> {
        // Control requests (message type 5) are session management, not the
        // nv interface. QueryPointerBufferSize must report 0 so libnx's
        // `SfBufferAttr_HipcAutoSelect` buffers are marshalled as map-alias
        // ranges rather than through a server pointer buffer we do not have.
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                // CloneCurrentObject(Ex): libnx clones the nvdrv session and
                // sends SubmitGpfifo/KickoffPb down the clone, so the new
                // handle has to route back to the same driver.
                Some(2) | Some(4) => {
                    let clone = self.alloc_handle();
                    self.record_handle(clone, "nvdrv");
                    self.write_ipc_response(tls, 0, &[clone], &[], &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let data = self.ipc_request_data(tls);
        let (send, recv) = self.ipc_map_buffers(tls);
        if self.trace_nv {
            eprintln!(
                "[nv] cmd={:?} send={:x?} recv={:x?} from {:#x?}",
                cmd_id,
                send,
                recv,
                self.backtrace(6)
            );
        }
        match cmd_id {
            // Open(path buffer) -> { u32 fd, u32 error }
            Some(0) => {
                let path = match send.first() {
                    Some(&(addr, size)) => self.read_string(addr, size),
                    None => String::new(),
                };
                let (fd, error) = self.nv.open(&path)?;
                let mut raw = [0u8; 8];
                raw[..4].copy_from_slice(&fd.to_le_bytes());
                raw[4..].copy_from_slice(&error.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // Ioctl / Ioctl2 / Ioctl3 { u32 fd, u32 request } -> u32 error.
            // Ioctl2 adds an inline input buffer between the argument buffers;
            // Ioctl3 adds an extra output buffer after them.
            Some(1) | Some(11) | Some(12) => {
                let fd = self.mem.read_u32(data)?;
                let request = self.mem.read_u32(data.wrapping_add(4))?;
                let inline_in: Vec<u8> = match cmd_id {
                    Some(11) => match send.get(1) {
                        Some(&(addr, size)) => self.read_bytes(addr, size),
                        None => Vec::new(),
                    },
                    _ => Vec::new(),
                };
                let size = crate::gpu::nvdrv::ioctl_size(request) as usize;
                let mut argp = match send.first() {
                    Some(&(addr, len)) if len > 0 => self.read_bytes(addr, len),
                    _ => Vec::new(),
                };
                argp.resize(size, 0);
                let error = self.nv.ioctl(&mut self.mem, fd, request, &mut argp, &inline_in)?;
                if error != 0 && std::env::var("TRACE_NV").is_ok() {
                    eprintln!("[nv] ioctl fd={fd} request={request:#x} -> error {error}");
                }
                if let Some(&(addr, len)) = recv.first() {
                    for (i, &byte) in argp.iter().take(len as usize).enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &error.to_le_bytes(), &[])
            }
            // Close(u32 fd) -> u32 error
            Some(2) => {
                let fd = self.mem.read_u32(data)?;
                let error = self.nv.close(fd);
                self.write_ipc_response(tls, 0, &[], &error.to_le_bytes(), &[])
            }
            // Initialize(u32 transfer_mem_size, handles) -> u32 error. libnx
            // ignores the out word, but libtransistor checks the reply's raw
            // size, so omitting it failed sdl-hello's nv init.
            Some(3) => {
                self.nv.transfer_mem_size = self.mem.read_u32(data).unwrap_or(0);
                self.nv.initialized = true;
                self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
            }
            // QueryEvent(u32 fd, u32 event_id) -> u32 error + a copy handle
            Some(4) => {
                let fd = self.mem.read_u32(data)?;
                let event_id = self.mem.read_u32(data.wrapping_add(4))?;
                let error = self.nv.query_event(fd, event_id);
                let handle = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[handle], &error.to_le_bytes(), &[])
            }
            // SetClientPID / everything else: acknowledge with no out data.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `ITimeServiceManager` (`time:s`/`time:u`/`time:a`/`time:r`): hands out
    /// the system/steady clocks and the timezone service.
    ///
    /// Its own commands (`GetStandardUserSystemClock` and friends) share
    /// command ids with `ConvertToDomain`/`QueryPointerBufferSize`, which
    /// arrive as a Control request (message type 5) rather than a normal
    /// one — the same distinction `vi_request` makes for `vi:m` — so the control
    /// path has to be checked first or a domain conversion would be read as
    /// `GetStandardUserSystemClock`.
    pub(super) fn time_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "time");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        const GET_STANDARD_USER_SYSTEM_CLOCK: u32 = 0;
        const GET_STANDARD_NETWORK_SYSTEM_CLOCK: u32 = 1;
        const GET_STANDARD_STEADY_CLOCK: u32 = 2;
        const GET_TIME_ZONE_SERVICE: u32 = 3;
        const GET_STANDARD_LOCAL_SYSTEM_CLOCK: u32 = 4;
        const GET_STANDARD_STEADY_CLOCK_RTC_VALUE: u32 = 51;
        const IS_STANDARD_USER_SYSTEM_CLOCK_AUTOMATIC_CORRECTION_ENABLED: u32 = 100;
        match cmd_id {
            // GetStandardUserSystemClock / GetStandardNetworkSystemClock /
            // GetStandardLocalSystemClock: there is no network time sync or
            // per-region offset here, so all three hand out the same clock.
            Some(GET_STANDARD_USER_SYSTEM_CLOCK)
            | Some(GET_STANDARD_NETWORK_SYSTEM_CLOCK)
            | Some(GET_STANDARD_LOCAL_SYSTEM_CLOCK) => {
                self.reply_with_interface(tls, handle, "time:system-clock")?;
                Ok(())
            }
            Some(GET_STANDARD_STEADY_CLOCK) => {
                self.reply_with_interface(tls, handle, "time:steady-clock")?;
                Ok(())
            }
            Some(GET_TIME_ZONE_SERVICE) => {
                self.reply_with_interface(tls, handle, "time:timezone")?;
                Ok(())
            }
            // -> u64, the RTC reading the steady clock is seeded from.
            Some(GET_STANDARD_STEADY_CLOCK_RTC_VALUE) => self.write_ipc_response(
                tls,
                0,
                &[],
                &(self.steady_clock_seconds() as u64).to_le_bytes(),
                &[],
            ),
            // -> bool. The host pushes wall-clock time directly
            // (`Cpu::set_unix_time`), so it is always "corrected".
            Some(IS_STANDARD_USER_SYSTEM_CLOCK_AUTOMATIC_CORRECTION_ENABLED) => {
                self.write_ipc_response(tls, 0, &[], &[1u8], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `ISystemClock`: wall-clock time, as POSIX seconds. The value comes
    /// straight from [`Cpu::set_unix_time`] — there is no persisted offset or
    /// network sync here, so `SetCurrentTime`/`SetSystemClockContext` are
    /// accepted but don't change what a later read sees.
    pub(super) fn time_system_clock_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        const GET_CURRENT_TIME: u32 = 0;
        const SET_CURRENT_TIME: u32 = 1;
        const GET_SYSTEM_CLOCK_CONTEXT: u32 = 2;
        const SET_SYSTEM_CLOCK_CONTEXT: u32 = 3;
        match cmd_id {
            // -> s64 PosixTime
            Some(GET_CURRENT_TIME) => {
                let posix = self.unix_time();
                self.write_ipc_response(tls, 0, &[], &posix.to_le_bytes(), &[])
            }
            Some(SET_CURRENT_TIME) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // -> SystemClockContext { s64 offset; SteadyClockTimePoint
            // timestamp }. The offset is left at 0 (the steady clock's own
            // value already reads as seconds-since-boot) and the timestamp
            // mirrors GetCurrentTimePoint.
            Some(GET_SYSTEM_CLOCK_CONTEXT) => {
                let mut raw = [0u8; 0x20];
                raw[0x08..0x10].copy_from_slice(&self.steady_clock_seconds().to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            Some(SET_SYSTEM_CLOCK_CONTEXT) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `ISteadyClock`: a monotonic clock unrelated to wall time.
    pub(super) fn time_steady_clock_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        const GET_CURRENT_TIME_POINT: u32 = 0;
        const GET_RTC_VALUE: u32 = 100;
        const IS_RTC_RESET_DETECTED: u32 = 101;
        const GET_SETUP_RESULT_VALUE: u32 = 102;
        match cmd_id {
            // -> SteadyClockTimePoint { s64 value; u8 source_id[0x10] }. The
            // source id is left zeroed: nothing here ever compares two time
            // points' ids, only their values.
            Some(GET_CURRENT_TIME_POINT) => {
                let mut raw = [0u8; 0x18];
                raw[..8].copy_from_slice(&self.steady_clock_seconds().to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // -> u64, the same RTC reading GetCurrentTimePoint's value is
            // seeded from.
            Some(GET_RTC_VALUE) => self.write_ipc_response(
                tls,
                0,
                &[],
                &(self.steady_clock_seconds() as u64).to_le_bytes(),
                &[],
            ),
            // -> bool. There is no real RTC to lose power and reset here.
            Some(IS_RTC_RESET_DETECTED) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // -> Result, as a raw u32. The RTC "setup" at boot always
            // succeeds.
            Some(GET_SETUP_RESULT_VALUE) => {
                self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// Seconds since this `Cpu` started, for the steady clock. Instructions
    /// retired stands in for elapsed wall time — the same arbitrary scale
    /// `svcGetSystemTick`'s `cycles * 1000` already uses — since only
    /// monotonicity matters here, not the rate.
    fn steady_clock_seconds(&self) -> i64 {
        (self.cycles / 1_000_000) as i64
    }

    /// `ITimeZoneService`: there is no bundled TZif database, so the device's
    /// timezone is fixed at UTC — the same "one hard-coded answer, no locale
    /// data" shortcut `set_request` takes for the system language.
    pub(super) fn time_timezone_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        const LOCATION_NAME: &[u8] = b"UTC";
        const GET_DEVICE_LOCATION_NAME: u32 = 0;
        const GET_TOTAL_LOCATION_NAME_COUNT: u32 = 2;
        const LOAD_LOCATION_NAME_LIST: u32 = 3;
        const LOAD_TIME_ZONE_RULE: u32 = 4;
        const TO_CALENDAR_TIME: u32 = 100;
        const TO_CALENDAR_TIME_WITH_MY_RULE: u32 = 101;
        const TO_POSIX_TIME: u32 = 201;
        const TO_POSIX_TIME_WITH_MY_RULE: u32 = 202;
        match cmd_id {
            // -> LocationName (0x24 bytes, NUL-padded).
            Some(GET_DEVICE_LOCATION_NAME) => {
                let mut raw = [0u8; 0x24];
                raw[..LOCATION_NAME.len()].copy_from_slice(LOCATION_NAME);
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            Some(GET_TOTAL_LOCATION_NAME_COUNT) => {
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            // LoadLocationNameList(u32 index) -> (u32 count, buffer<LocationName[]>)
            Some(LOAD_LOCATION_NAME_LIST) => {
                if let Some(&(addr, size)) = self.ipc_map_buffers(tls).1.first() {
                    if size >= LOCATION_NAME.len() as u32 {
                        for (i, &b) in LOCATION_NAME.iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(i as u32), b)?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            // LoadTimeZoneRule(LocationName) -> TimeZoneRule. The rule blob's
            // contents are never read back: ToCalendarTime(WithMyRule) below
            // always resolves against UTC regardless of which rule a caller
            // loaded, so there's nothing to fill the receive buffer with.
            Some(LOAD_TIME_ZONE_RULE) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // ToCalendarTime(s64, TimeZoneRule buffer) /
            // ToCalendarTimeWithMyRule(s64): both resolve against UTC; the
            // incoming rule buffer (TO_CALENDAR_TIME only) is ignored.
            Some(TO_CALENDAR_TIME) | Some(TO_CALENDAR_TIME_WITH_MY_RULE) => {
                let posix = self.mem.read_u64(self.ipc_request_data(tls)).unwrap_or(0) as i64;
                self.write_ipc_response(tls, 0, &[], &Self::to_calendar_time(posix), &[])
            }
            // ToPosixTime(CalendarTime, rule buffer) /
            // ToPosixTimeWithMyRule(CalendarTime): both resolve against UTC
            // and, since there's no DST to make a wall-clock time ambiguous,
            // always report exactly one match.
            Some(TO_POSIX_TIME) | Some(TO_POSIX_TIME_WITH_MY_RULE) => {
                let data = self.ipc_request_data(tls);
                let posix = Self::from_calendar_time(
                    self.mem.read_u16(data).unwrap_or(1970),
                    self.mem.read_u8(data.wrapping_add(2)).unwrap_or(1),
                    self.mem.read_u8(data.wrapping_add(3)).unwrap_or(1),
                    self.mem.read_u8(data.wrapping_add(4)).unwrap_or(0),
                    self.mem.read_u8(data.wrapping_add(5)).unwrap_or(0),
                    self.mem.read_u8(data.wrapping_add(6)).unwrap_or(0),
                );
                if let Some(&(addr, size)) = self.ipc_map_buffers(tls).1.first() {
                    if size >= 8 {
                        self.mem.write_u64(addr, posix as u64)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }


    /// `{ CalendarTime, CalendarAdditionalInfo }` for a POSIX time, assuming
    /// UTC: `CalendarTime { u16 year; u8 month, day, hour, minute, second,
    /// pad; }` (8 bytes) followed by `CalendarAdditionalInfo { u32
    /// day_of_week, day_of_year; u8 name[8]; u32 utc_offset_seconds; u8 dst,
    /// pad[3]; }` (0x18 bytes) — 0x20 bytes total.
    fn to_calendar_time(posix: i64) -> [u8; 0x20] {
        let days = posix.div_euclid(86400);
        let secs_of_day = posix.rem_euclid(86400);
        let (year, month, day) = civil_from_days(days);
        let day_of_week = (days + 4).rem_euclid(7); // 1970-01-01 was a Thursday
        let day_of_year = days - days_from_civil(year, 1, 1);

        let mut raw = [0u8; 0x20];
        raw[0..2].copy_from_slice(&(year.clamp(0, u16::MAX as i64) as u16).to_le_bytes());
        raw[2] = month as u8;
        raw[3] = day as u8;
        raw[4] = (secs_of_day / 3600) as u8;
        raw[5] = ((secs_of_day / 60) % 60) as u8;
        raw[6] = (secs_of_day % 60) as u8;
        raw[8..12].copy_from_slice(&(day_of_week as u32).to_le_bytes());
        raw[12..16].copy_from_slice(&(day_of_year as u32).to_le_bytes());
        raw[16..19].copy_from_slice(b"UTC");
        raw
    }

    /// Inverse of [`Cpu::to_calendar_time`], assuming UTC.
    fn from_calendar_time(year: u16, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> i64 {
        let days = days_from_civil(year as i64, month.max(1) as u32, day.max(1) as u32);
        days * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64
    }

    /// `psm` (Power State Management): the battery. Its own commands share
    /// command ids with `ConvertToDomain`/`QueryPointerBufferSize` the same
    /// way `time_request` does, so the control path is checked first.
    pub(super) fn psm_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "psm");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        const GET_BATTERY_CHARGE_PERCENTAGE: u32 = 0;
        const GET_CHARGER_TYPE: u32 = 1;
        const ENABLE_BATTERY_CHARGING: u32 = 2;
        const DISABLE_BATTERY_CHARGING: u32 = 3;
        const IS_BATTERY_CHARGING_ENABLED: u32 = 4;
        const OPEN_SESSION: u32 = 7;
        // ChargerType: 0 Unconnected, 1 EnoughPower, 2 LowPower, 3 NotSupported.
        // The Battery Status API (where the host exposes one) only reports a
        // charging bool, not the power level a real charger negotiates, so
        // charging maps to EnoughPower and not charging to Unconnected.
        const CHARGER_UNCONNECTED: u32 = 0;
        const CHARGER_ENOUGH_POWER: u32 = 1;
        match cmd_id {
            Some(GET_BATTERY_CHARGE_PERCENTAGE) => {
                let (percent, _) = self.battery();
                self.write_ipc_response(tls, 0, &[], &(percent as u32).to_le_bytes(), &[])
            }
            Some(GET_CHARGER_TYPE) => {
                let (_, charging) = self.battery();
                let charger = if charging { CHARGER_ENOUGH_POWER } else { CHARGER_UNCONNECTED };
                self.write_ipc_response(tls, 0, &[], &charger.to_le_bytes(), &[])
            }
            // EnableBatteryCharging/DisableBatteryCharging: accepted, but
            // charging state here mirrors the host battery, not a guest
            // setting — there is nothing to actually stop charging.
            Some(ENABLE_BATTERY_CHARGING) | Some(DISABLE_BATTERY_CHARGING) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(IS_BATTERY_CHARGING_ENABLED) => {
                self.write_ipc_response(tls, 0, &[], &[1u8], &[])
            }
            Some(OPEN_SESSION) => {
                self.reply_with_interface(tls, handle, "psm-session")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IPsmSession`: the live charger/battery-state-change notifications a
    /// caller can subscribe to. There is no push channel from the host
    /// battery here — [`Cpu::set_battery`] is polled, the way
    /// `GetBatteryChargePercentage` already is — so the bound event is
    /// handed out but never signalled; a caller has to keep polling rather
    /// than wait on it.
    pub(super) fn psm_session_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        const BIND_STATE_CHANGE_EVENT: u32 = 0;
        const UNBIND_STATE_CHANGE_EVENT: u32 = 1;
        const SET_CHARGER_TYPE_CHANGE_EVENT_ENABLED: u32 = 2;
        const SET_POWER_SUPPLY_CHANGE_EVENT_ENABLED: u32 = 3;
        const SET_BATTERY_VOLTAGE_STATE_CHANGE_EVENT_ENABLED: u32 = 4;
        match cmd_id {
            Some(BIND_STATE_CHANGE_EVENT) => {
                let event = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[event], &[], &[])
            }
            Some(UNBIND_STATE_CHANGE_EVENT)
            | Some(SET_CHARGER_TYPE_CHANGE_EVENT_ENABLED)
            | Some(SET_POWER_SUPPLY_CHANGE_EVENT_ENABLED)
            | Some(SET_BATTERY_VOLTAGE_STATE_CHANGE_EVENT_ENABLED) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// Answers an `am` command the applet stub does not actually implement.
    ///
    /// Everything `am` hands back is a live kernel object or a piece of applet
    /// state the caller then acts on, so a blanket "success, no data" reply is
    /// not a neutral placeholder — it is a wrong answer the guest believes.
    /// That is exactly how `nn::oe::SetupGpuErrorHandler` ended up waiting on
    /// handle **0**: the old catch-all answered
    /// `GetGpuErrorDetectedSystemEvent` with success and no copy handle at
    /// all, and the SDK's system worker took the missing handle at face value.
    /// Reporting `cmif`'s "unknown command id" instead makes the guest fail at
    /// the command that is genuinely missing, and the warning names the one to
    /// implement next.
    pub(super) fn am_unimplemented(
        &mut self,
        tls: u32,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `cmif` (module 10) description 221: what a real `sf` server answers
        /// when a session has no handler for the requested command id.
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        if self.unimplemented_ipc.insert((iface.to_string(), cmd_id)) {
            let pc = self.pc;
            self.diagnostic(&format!("[am] unimplemented: {iface} cmd={cmd_id:?} (pc={pc:#x})"));
        }
        self.write_ipc_response(tls, UNKNOWN_COMMAND_ID, &[], &[], &[])
    }

    /// Note that a service reached over IPC has no implementation behind it at
    /// all, and is about to be answered with a fabricated object id.
    ///
    /// Unlike [`Cpu::am_unimplemented`] this does not change the reply — the
    /// generic success is load-bearing for homebrew that only checks the
    /// Result — it just stops the gap being invisible. Whatever this prints is
    /// the list of services a guest is asking for and not getting.
    pub(super) fn warn_no_implementation(&mut self, service: &str, cmd_id: Option<u32>) {
        if self.unimplemented_ipc.insert((service.to_string(), cmd_id)) {
            self.diagnostic(&format!("[ipc] no implementation: {service} cmd={cmd_id:?}"));
        }
    }

    /// `IApplicationProxyService`/`IApplicationProxy`: the applet-lifecycle
    /// chain homebrew opens as `appletOE` (or `appletAE`, for a non-application
    /// applet). `appletMainLoop` polls `ICommonStateGetter` every frame — the
    /// event handle, then `ReceiveMessage`/`GetOperationMode`/
    /// `GetCurrentFocusState` — to decide whether to keep running; an earlier
    /// generic stub answered every one of those the same way regardless of
    /// which sub-interface actually made the call (and re-sent the initial
    /// "focus changed" message on every single poll), which made at least one
    /// real homebrew (JKSV) treat every frame as a fresh focus transition and
    /// give up after a handful of them.
    ///
    /// Only the commands listed below are implemented. Everything else goes to
    /// [`Cpu::am_unimplemented`] rather than a fabricated success — see there
    /// for why.
    pub(super) fn applet_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "am:proxy-service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.am_unimplemented(tls, "am:control", cmd_id),
            };
        }
        // Which `am` sub-interface this request is actually for. A caller that
        // converted the session to a domain (`libnx`) addresses each one by
        // object id on the one `appletOE` handle; a caller that did not
        // (`nnSdk`) got a separate session handle per interface out of
        // [`Cpu::reply_with_interface`], and the name is recorded against the
        // handle instead. Resolving only the domain case left every `nnSdk`
        // request answered as `am:unknown`.
        let object_id = self.ipc_domain_object_id(tls);
        // A domain *close* request (`CmifDomainRequestType_Close`, the domain
        // header's type byte set to 2) drops one object out of the session and
        // carries no `CmifInHeader` at all. [`Cpu::ipc_command_id`] falls back
        // to scanning the whole message buffer for an `SFCI` magic, so on a
        // close it finds the *previous* request's header still sitting there
        // and reports that command id — which is why `appletExit`'s teardown
        // used to look like a flurry of command 0s.
        if self.mem.read_u8(tls.wrapping_add(self.ipc_reply_start(tls))).unwrap_or(0) == 2 {
            self.domain_objects.remove(&(handle, object_id));
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("am:unknown").to_string()
        } else {
            match self.service_name(handle) {
                // The root session before any ConvertToDomain *is*
                // IApplicationProxyService.
                Some("appletOE") | Some("appletAE") | None => "am:proxy-service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            // IApplicationProxyService::OpenApplicationProxy.
            "am:proxy-service" => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "am:application-proxy")?;
                    Ok(())
                }
                _ => self.am_unimplemented(tls, &iface, cmd_id),
            },
            // IApplicationProxy's Get* accessors, each handing back one of the
            // sub-interfaces below.
            "am:application-proxy" => {
                let sub = match cmd_id {
                    Some(0) => Some("am:common-state-getter"),
                    Some(1) => Some("am:self-controller"),
                    Some(2) => Some("am:window-controller"),
                    Some(3) => Some("am:audio-controller"),
                    Some(4) => Some("am:display-controller"),
                    Some(11) => Some("am:library-applet-creator"),
                    Some(20) => Some("am:application-functions"),
                    Some(1000) => Some("am:debug-functions"),
                    _ => None,
                };
                match sub {
                    Some(name) => {
                        self.reply_with_interface(tls, handle, name)?;
                        Ok(())
                    }
                    None => self.am_unimplemented(tls, &iface, cmd_id),
                }
            }
            // ICommonStateGetter: the state `appletMainLoop` polls every frame.
            "am:common-state-getter" => match cmd_id {
                // GetEventHandle: a copy handle the guest waits on before
                // polling ReceiveMessage. WaitSynchronization treats any
                // handle outside the thread/mutex tables as already
                // signaled — the same trick `vi:m`'s GetDisplayVsyncEvent
                // uses — so the wait never blocks.
                Some(0) => {
                    let h = self.alloc_handle();
                    self.write_ipc_response(tls, 0, &[h], &[], &[])
                }
                // ReceiveMessage: real AM enqueues one FocusStateChanged at
                // startup and then reports "no message" until the state
                // actually changes; answering every poll with a fresh message
                // is what made JKSV think focus kept changing.
                Some(1) => {
                    const NO_MESSAGES: u32 = 128 | (3 << 9); // am, "no message"
                    const FOCUS_STATE_CHANGED: u32 = 15;
                    if std::mem::replace(&mut self.applet_focus_sent, true) {
                        self.write_ipc_response(tls, NO_MESSAGES, &[], &[], &[])
                    } else {
                        self.write_ipc_response(tls, 0, &[], &FOCUS_STATE_CHANGED.to_le_bytes(), &[])
                    }
                }
                Some(5) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]), // GetOperationMode: Handheld
                Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]), // GetPerformanceMode: Normal
                Some(9) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]), // GetCurrentFocusState: InFocus
                // GetBootMode: Normal.
                Some(8) => self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[]),
                // GetAcquiredSleepLockEvent / GetDefaultDisplayResolutionChangeEvent:
                // handles the caller waits on. Nothing here ever sleeps or
                // changes resolution, so they are handed out and never
                // signalled — see the note on GetEventHandle above for why a
                // wait on them still returns.
                Some(13) | Some(61) => {
                    let h = self.alloc_handle();
                    self.write_ipc_response(tls, 0, &[h], &[], &[])
                }
                // GetDefaultDisplayResolution: the same 1280x720 the display
                // stub hands out as its native window.
                Some(60) => {
                    let mut raw = Vec::with_capacity(8);
                    raw.extend_from_slice(&1280u32.to_le_bytes());
                    raw.extend_from_slice(&720u32.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                // RequestToAcquireSleepLock / ReleaseSleepLock /
                // ReleaseSleepLockTransiently / SetCpuBoostMode: there is no
                // sleep state or clock governor to move, so accepting the
                // request is the whole implementation.
                Some(10) | Some(11) | Some(12) | Some(66) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.am_unimplemented(tls, &iface, cmd_id),
            },
            // IApplicationFunctions: PopLaunchParameter fails when hbmenu
            // launched the app without forwarding arguments, same as on real
            // hardware — an earlier stub's success-with-an-unrelated-object-id
            // left callers treating that id as a launch-parameter storage
            // object that was never actually registered as one.
            "am:application-functions" => match cmd_id {
                Some(1) => {
                    const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9); // am
                    self.write_ipc_response(tls, LAUNCH_PARAMETER_NOT_FOUND, &[], &[], &[])
                }
                // EnsureSaveData -> the save data size it ensured.
                Some(20) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // GetDesiredLanguage -> an `nn::settings::LanguageCode`, which
                // is the null-padded BCP-47 tag as eight raw bytes.
                Some(21) => {
                    let mut code = [0u8; 8];
                    code[..5].copy_from_slice(b"en-US");
                    self.write_ipc_response(tls, 0, &[], &code, &[])
                }
                // GetDisplayVersion -> a 16-byte version string.
                Some(23) => {
                    let mut version = [0u8; 16];
                    version[..5].copy_from_slice(b"1.0.0");
                    self.write_ipc_response(tls, 0, &[], &version, &[])
                }
                // NotifyRunning -> whether the notification was the first one.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[]),
                // GetPseudoDeviceId -> a 16-byte per-console, per-title id.
                // Zero is a legitimate value and nothing here derives anything
                // from it, but it must be the right *size* — a caller copies
                // 16 bytes out of the reply either way.
                Some(50) => self.write_ipc_response(tls, 0, &[], &[0u8; 16], &[]),
                // GetGpuErrorDetectedSystemEvent: the event `nn::oe::
                // SetupGpuErrorHandler` registers with the SDK's system
                // worker, so that a GPU fault wakes a handler instead of
                // hanging the title. It is the first thing a retail `nnSdk`
                // asks `am` for that it cannot start without — answering it
                // with anything but a copy handle aborts `nn::oe::Initialize`.
                // Nothing here ever faults the GPU, so the event is handed out
                // and never signalled.
                Some(130) => {
                    let h = self.alloc_handle();
                    self.write_ipc_response(tls, 0, &[h], &[], &[])
                }
                // SetTerminateResult / InitializeGamePlayRecording /
                // SetGamePlayRecordingState / SetDelayTimeToAbortOnGpuError:
                // nothing to record, nothing to fault, nothing to report back.
                Some(22) | Some(66) | Some(67) | Some(131) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.am_unimplemented(tls, &iface, cmd_id),
            },
            // ISelfController: the applet's own lifecycle knobs.
            "am:self-controller" => match cmd_id {
                // Exit / LockExit / UnlockExit / EnterFatalSection /
                // LeaveFatalSection / SetScreenShotPermission /
                // Set{Operation,Performance}ModeChangedNotification /
                // SetFocusHandlingMode / SetRestartMessageEnabled /
                // SetScreenShotAppletIdentityInfo /
                // SetOutOfFocusSuspendingEnabled /
                // SetScreenShotImageOrientation / SetHandlesRequestToDisplay /
                // SetIdleTimeDetectionExtension / SetAutoSleepDisabled /
                // SetAlbumImageTakenNotificationEnabled /
                // SetApplicationAlbumUserData / SetRecordVolumeMuted.
                //
                // Every one of these is a setter or a notifier whose whole
                // reply is a Result. There is no suspend, screenshot, album or
                // exit-lock behaviour behind them to change, so accepting the
                // setting really is the complete implementation — unlike the
                // commands below it, a bare success here is the truth.
                Some(0..=4) | Some(10..=16) | Some(19) | Some(50) | Some(62) | Some(68)
                | Some(100) | Some(110) | Some(130) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetLibraryAppletLaunchableEvent /
                // GetAccumulatedSuspendedTickChangedEvent: copy handles the
                // caller stores and later waits on. `libnx`'s `appletInitialize`
                // asks for the second one on 6.0.0+ and keeps whatever handle
                // came back, so answering with success and *no* handle left it
                // holding 0 — the same shape of bug that had `nnSdk`'s system
                // worker waiting on handle 0.
                Some(9) | Some(91) => {
                    let h = self.alloc_handle();
                    self.write_ipc_response(tls, 0, &[h], &[], &[])
                }
                // GetAccumulatedSuspendedTickValue: nothing has ever been
                // suspended.
                Some(90) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // CreateManagedDisplayLayer -> the layer id the caller then
                // passes to `vi`'s OpenLayer. The display stub only models one
                // layer and calls it 1 (see [`Cpu::vi_native_window`]), so this
                // has to agree with it.
                Some(40) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // CreateManagedDisplaySeparableLayer -> the same layer plus a
                // recording layer, which nothing here records from.
                Some(44) => {
                    let mut raw = Vec::with_capacity(16);
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    raw.extend_from_slice(&1u64.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &raw, &[])
                }
                _ => self.am_unimplemented(tls, &iface, cmd_id),
            },
            // IWindowController: foreground rights and the applet resource id
            // every other service tags this process's requests with.
            "am:window-controller" => match cmd_id {
                // GetAppletResourceUserId / GetAppletResourceUserIdOfCallerApplet.
                // There is one process here, so it gets one id; the `vi` and
                // `hid` stubs ignore which id a request carries.
                Some(1) | Some(2) => {
                    self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[])
                }
                // AcquireForegroundRights / ReleaseForegroundRights /
                // RejectToChangeIntoBackground: nothing else is competing for
                // the foreground.
                Some(10) | Some(11) | Some(12) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.am_unimplemented(tls, &iface, cmd_id),
            },
            // IAudioController: the applet's volume relative to the system's.
            "am:audio-controller" => match cmd_id {
                // SetExpectedMasterVolume / ChangeMainAppletMasterVolume /
                // SetTransparentVolumeRate.
                Some(0) | Some(3) | Some(4) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // Get{Main,Library}AppletExpectedMasterVolume -> an f32.
                Some(1) | Some(2) => {
                    self.write_ipc_response(tls, 0, &[], &1.0f32.to_le_bytes(), &[])
                }
                _ => self.am_unimplemented(tls, &iface, cmd_id),
            },
            // IDisplayController (capture buffers), ILibraryAppletCreator
            // (launching another applet), IDebugFunctions, and any session that
            // never named itself. Nothing here can answer those honestly: a
            // capture buffer has no contents, and a library applet has nowhere
            // to run.
            _ => self.am_unimplemented(tls, &iface, cmd_id),
        }
    }

    /// `nifm:u`'s root session: session control plus
    /// `CreateGeneralServiceOld`/`CreateGeneralService`, which hand back the
    /// `IGeneralService` homebrew actually queries connectivity through.
    pub(super) fn nifm_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "nifm:u");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        const CREATE_GENERAL_SERVICE_OLD: u32 = 4;
        const CREATE_GENERAL_SERVICE: u32 = 5;
        match cmd_id {
            Some(CREATE_GENERAL_SERVICE_OLD) | Some(CREATE_GENERAL_SERVICE) => {
                self.reply_with_interface(tls, handle, "nifm:general-service")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IGeneralService`: reports a wired connection that is up and has
    /// internet access, and hands out `IRequest` objects that immediately
    /// look accepted — there is no real network stack behind this, so every
    /// homebrew that only checks "is there a connection" sees a permanent
    /// wired one instead of the emulator looking offline.
    pub(super) fn nifm_general_service_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        const GET_CLIENT_ID: u32 = 1;
        const CREATE_REQUEST: u32 = 4;
        const GET_CURRENT_IP_ADDRESS: u32 = 15;
        const GET_INTERNET_CONNECTION_STATUS: u32 = 12;
        match cmd_id {
            Some(GET_CLIENT_ID) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            Some(CREATE_REQUEST) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, "nifm:request");
                self.write_ipc_response(tls, 0, &[], &[], &[obj])
            }
            // NifmInternetConnectionType_Ethernet, no Wi-Fi strength to
            // report, NifmInternetConnectionStatus_Connected.
            Some(GET_INTERNET_CONNECTION_STATUS) => {
                self.write_ipc_response(tls, 0, &[], &[2u8, 0u8, 2u8], &[])
            }
            Some(GET_CURRENT_IP_ADDRESS) => {
                self.write_ipc_response(tls, 0, &[], &[192, 168, 1, 100], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `audren:u` (`IAudioRendererManager`): the factory for `IAudioRenderer`.
    /// Never converted to a domain (libnx builds it with
    /// `NX_SERVICE_ASSUME_NON_DOMAIN`), so `OpenAudioRenderer` hands its
    /// session out as a move handle, the same as `vi:m`/`nvdrv`.
    ///
    /// Answering `GetWorkBufferSize` with an empty reply (the old generic
    /// stub) left `workBufSize` as whatever garbage was already in that
    /// stack slot; `tmemCreate`ing a transfer memory block of that size
    /// reliably failed, and `audrenInitialize` — and so `SDL_OpenAudioDevice`,
    /// and so `JKSV::initialize_sdl`, and so `JKSV::JKSV()` itself — gave up
    /// before a single frame ever rendered.
    pub(super) fn audren_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        let data = self.ipc_request_data(tls);
        // `AudioRendererParameter`: sample_rate, sample_count, mix_buffer_count,
        // submix_count, voice_count, sink_count, effect_count, unk1, unk2+pad,
        // splitter_count, unk3, unk4, revision.
        let voice_count = self.mem.read_u32(data.wrapping_add(16)).unwrap_or(0);
        let sink_count = self.mem.read_u32(data.wrapping_add(20)).unwrap_or(0);
        let effect_count = self.mem.read_u32(data.wrapping_add(24)).unwrap_or(0);
        let revision = self.mem.read_u32(data.wrapping_add(48)).unwrap_or(0);
        match cmd_id {
            // GetWorkBufferSize: any page-sized answer works — nothing here
            // actually allocates real renderer memory out of it.
            Some(1) => self.write_ipc_response(tls, 0, &[], &0x10_0000u64.to_le_bytes(), &[]),
            // OpenAudioRenderer.
            Some(0) => {
                let renderer = self.alloc_handle();
                self.record_handle(renderer, "audren:iaudiorenderer");
                self.audren_renderers.insert(
                    renderer,
                    AudrenParams { revision, voice_count, sink_count, effect_count },
                );
                self.write_ipc_response(tls, 0, &[renderer], &[], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IAudioRenderer`. `RequestUpdateAudioRenderer` fills in the exact
    /// shape `audrvUpdate` expects (`AudioRendererUpdateDataHeader` + one
    /// `AudioRendererMemPoolInfoOut` per mempool + one
    /// `AudioRendererVoiceInfoOut` per voice + one `AudioRendererSinkInfoOut`
    /// per sink + the performance/behavior tails), all zeroed — no mempool
    /// ever needs attaching, no voice ever played anything, so the all-zero
    /// answer is a truthful "did nothing" rather than a guess. Getting the
    /// `_sz` fields right matters: `audrvUpdate` rejects a reply whose sizes
    /// do not match what it computed from the same voice/sink/effect counts,
    /// and it runs every frame the app is alive, not just at startup.
    pub(super) fn audren_renderer_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        match cmd_id {
            // GetState: 0 is a live sample-generation counter to a real
            // console; any stable value here is a legal "nothing changed".
            Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // RequestUpdateAudioRenderer, cmd 4 pre-3.0.0 / cmd 10 since.
            Some(4) | Some(10) => self.audren_write_update_reply(tls, handle),
            Some(5) | Some(6) | Some(8) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // QuerySystemEvent: a copy handle WaitSynchronization treats as
            // already signaled, the same trick `vi:m`'s vsync event uses —
            // the audio thread's frame wait never blocks.
            Some(7) => {
                let h = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[h], &[], &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    fn audren_write_update_reply(&mut self, tls: u32, handle: u64) -> Result<()> {
        let params = self.audren_renderers.get(&handle).copied().unwrap_or(AudrenParams {
            revision: 0,
            voice_count: 0,
            sink_count: 0,
            effect_count: 0,
        });
        let mempool_count = params.effect_count + 4 * params.voice_count;
        const HEADER_SZ: u32 = 64;
        const MEMPOOL_OUT_SZ: u32 = 16;
        const VOICE_OUT_SZ: u32 = 16;
        const SINK_OUT_SZ: u32 = 32;
        const PERFMGR_OUT_SZ: u32 = 16;
        const BEHAVIOR_OUT_SZ: u32 = 176;
        let mempools_sz = mempool_count * MEMPOOL_OUT_SZ;
        let voices_sz = params.voice_count * VOICE_OUT_SZ;
        let sinks_sz = params.sink_count * SINK_OUT_SZ;
        let total_sz = HEADER_SZ + mempools_sz + voices_sz + sinks_sz + PERFMGR_OUT_SZ + BEHAVIOR_OUT_SZ;

        let mut reply = vec![0u8; total_sz as usize];
        reply[0..4].copy_from_slice(&params.revision.to_le_bytes());
        reply[4..8].copy_from_slice(&BEHAVIOR_OUT_SZ.to_le_bytes());
        reply[8..12].copy_from_slice(&mempools_sz.to_le_bytes());
        reply[12..16].copy_from_slice(&voices_sz.to_le_bytes());
        // channels_sz, effects_sz, mixes_sz stay 0: not part of this revision's
        // output layout (libnx's own size helpers don't count them either).
        reply[28..32].copy_from_slice(&sinks_sz.to_le_bytes());
        reply[32..36].copy_from_slice(&PERFMGR_OUT_SZ.to_le_bytes());
        reply[60..64].copy_from_slice(&total_sz.to_le_bytes());
        // Every MemPoolInfoOut/VoiceInfoOut/SinkInfoOut/PerformanceBufferInfoOut
        // entry after the header is left zeroed: `AudioRendererMemPoolState_Invalid`
        // (0) tells the caller to leave that mempool's state alone, and zeroed
        // voice/sink/performance counters are a truthful "nothing happened".

        let (_, recv) = self.ipc_map_buffers(tls);
        if let Some(&(addr, size)) = recv.first() {
            let n = (size as usize).min(reply.len());
            for (i, &byte) in reply[..n].iter().enumerate() {
                self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// Read `len` bytes of guest memory, stopping at the first fault.
    pub(super) fn read_bytes(&self, addr: u32, len: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(len as usize);
        for i in 0..len {
            match self.mem.read_u8(addr.wrapping_add(i)) {
                Ok(byte) => out.push(byte),
                Err(_) => break,
            }
        }
        out
    }

    pub(super) fn read_string(&self, addr: u32, len: u32) -> String {
        let bytes = self.read_bytes(addr, len);
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }

    /// Every map-alias buffer descriptor in a hipc request, as
    /// `(send, receive)` lists of `(address, size)`.
    ///
    /// The descriptors sit after the message header (and the optional special
    /// header plus pid) and the send statics; each is three words:
    /// `{size_low, address_low, packed}` where the packed word holds the mode
    /// in bits 0..1, address bits 36..57 in bits 2..23, size bits 32..35 in
    /// bits 24..27 and address bits 32..35 in bits 28..31.
    pub(super) fn ipc_map_buffers(&self, tls: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        let num_recv_buffers = (hdr1 >> 24) & 0xf;
        let mut off = 8u32;
        if (hdr2 >> 31) & 1 != 0 {
            off += 4;
            if self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0) & 1 != 0 {
                off += 8; // pid
            }
        }
        off += 8 * num_send_statics;

        let mut read_descriptor = |index: u32| -> (u32, u32) {
            let at = off + 12 * index;
            let size = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
            let address = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
            (address, size)
        };
        let send = (0..num_send_buffers).map(&mut read_descriptor).collect();
        let recv = (num_send_buffers..num_send_buffers + num_recv_buffers)
            .map(&mut read_descriptor)
            .collect();
        (send, recv)
    }
}

/// Proleptic-Gregorian day count (days since 1970-01-01) to (year, month,
/// day). Howard Hinnant's `civil_from_days`
/// (<http://howardhinnant.github.io/date_algorithms.html>), the standard
/// integer algorithm for this — no `chrono` dependency needed for the one
/// calendar conversion `ITimeZoneService` requires.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

/// Inverse of [`civil_from_days`].
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let (m, d) = (m as i64, d as i64);
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400); // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::Cpu;

    const TLS: u32 = 0x2000;
    const SFCI: u32 = 0x4943_4653;

    /// A CMIF request in the TLS buffer with no buffer descriptors:
    /// `CmifDomainInHeader` (when `domain`) then `CmifInHeader` then payload.
    fn request(domain: bool, command_id: u32, payload: &[u8]) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.write_u32(TLS, 4).unwrap(); // CmifCommandType_Request
        cpu.mem.write_u32(TLS + 4, 8).unwrap(); // num_data_words
        let mut at = TLS + 0x10; // the aligned data area
        if domain {
            cpu.mem.write_u8(at, 1).unwrap(); // CmifDomainRequestType_SendMessage
            cpu.mem.write_u32(at + 4, 7).unwrap(); // object id
            at += 0x10;
        }
        cpu.mem.write_u32(at, SFCI).unwrap();
        cpu.mem.write_u32(at + 8, command_id).unwrap();
        at += 0x10;
        for (i, &byte) in payload.iter().enumerate() {
            cpu.mem.write_u8(at + i as u32, byte).unwrap();
        }
        cpu
    }

    #[test]
    fn request_payload_skips_the_domain_header() {
        // fsFileRead's payload is { u32 option, u32 pad, s64 offset, u64 size }.
        let mut payload = [0u8; 0x18];
        payload[8..16].copy_from_slice(&0x10u64.to_le_bytes());
        payload[16..24].copy_from_slice(&0x70u64.to_le_bytes());

        let plain = request(false, 0, &payload);
        assert_eq!(plain.ipc_request_data(TLS), TLS + 0x20);
        assert_eq!(plain.ipc_command_id(TLS), Some(0));
        assert!(!plain.ipc_is_domain_request(TLS));

        // libnx converts the fsp-srv session to a domain, which pushes the
        // payload another 16 bytes in. Assuming a fixed 0x10 read the offset and
        // size out of the CmifInHeader, so every read asked for 0 bytes at
        // offset 0 and `romfsMountSelf` failed with an I/O error.
        let domain = request(true, 0, &payload);
        assert_eq!(domain.ipc_request_data(TLS), TLS + 0x30);
        assert_eq!(domain.ipc_command_id(TLS), Some(0));
        assert!(domain.ipc_is_domain_request(TLS));
        assert_eq!(domain.ipc_domain_object_id(TLS), 7);

        let data = domain.ipc_request_data(TLS);
        assert_eq!(domain.mem.read_u64(data + 8).unwrap(), 0x10);
        assert_eq!(domain.mem.read_u64(data + 0x10).unwrap(), 0x70);
    }

    #[test]
    fn the_cmif_header_is_found_past_the_buffer_descriptors() {
        // A request with buffer descriptors pushes its CMIF header further into
        // the message buffer: nvdrv's KICKOFF_PB lands at 0x40. Scanning only
        // the first 0x40 bytes reported "no command id", so the GPU submit was
        // answered as an unknown command with a generic success and hbmenu's
        // frame fence never signalled.
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        // type 4, 0 statics, 2 send buffers, 1 recv buffer → the data area is
        // 8 + 3*12 = 44 bytes in, rounded up to 0x30.
        cpu.mem.write_u32(TLS, 4 | (2 << 20) | (1 << 24)).unwrap();
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        let data_area = cpu.ipc_reply_start(TLS);
        assert_eq!(data_area, 0x30);
        cpu.mem.write_u32(TLS + data_area, SFCI).unwrap();
        cpu.mem.write_u32(TLS + data_area + 8, 0x1b).unwrap();
        assert_eq!(cpu.ipc_command_id(TLS), Some(0x1b));
        assert_eq!(cpu.ipc_request_data(TLS), TLS + data_area + 0x10);

        // And when the descriptor walk doesn't land exactly on it (0x40 here),
        // the scan of the message buffer still finds it.
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.write_u32(TLS, 4).unwrap();
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        cpu.mem.write_u32(TLS + 0x40, SFCI).unwrap();
        cpu.mem.write_u32(TLS + 0x48, 0x1b).unwrap();
        assert_eq!(cpu.ipc_command_id(TLS), Some(0x1b));
        assert_eq!(cpu.ipc_request_data(TLS), TLS + 0x50);
    }

    #[test]
    fn reply_header_type_is_zero_and_carries_the_move_handle() {
        let mut cpu = request(false, 1, &[]);
        cpu.write_ipc_response(TLS, 0, &[0x1234], &7u32.to_le_bytes(), &[]).unwrap();

        // A reply's type field is 0. libnx ignores it, but libtransistor
        // rejects anything other than 0 or 4 with its error 0x7E0DD, which is
        // what made sdl-hello fail to open fsp-srv.
        assert_eq!(cpu.mem.read_u32(TLS).unwrap() & 0xFFFF, 0);
        // Word 1: the raw-data word count with bit 31 set for the handle
        // descriptor that follows.
        let header1 = cpu.mem.read_u32(TLS + 4).unwrap();
        assert_eq!(header1 >> 31, 1);
        assert_eq!(header1 & 0x1FF, 4 + 1 + 4); // SFCO + one word + padding
        // Handle descriptor: one move handle, no pid, no copy handles.
        assert_eq!(cpu.mem.read_u32(TLS + 8).unwrap(), 1 << 5);
        assert_eq!(cpu.mem.read_u32(TLS + 12).unwrap(), 0x1234);
        // The data section is 16-byte aligned: SFCO, version, result, token,
        // then the payload.
        assert_eq!(cpu.mem.read_u32(TLS + 0x10).unwrap(), 0x4F43_4653);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 7);
    }

    #[test]
    fn pl_serves_the_host_font_from_its_shared_memory() {
        // `plGetSharedFont` asks for the sizes and offsets of the shared fonts
        // and then reads the font data straight out of pl's shared memory, which
        // is where the host font lands when the guest maps it. The three output
        // buffers take the type, the offset and the size of each font.
        const BUFFERS: u32 = 0x3000;
        let font = b"not really a font, but bytes are bytes".to_vec();

        let mut cpu = request(false, 2, &[]);
        cpu.set_shared_font(font.clone());
        cpu.pl_request(TLS, Some(2)).unwrap();
        let size = cpu.mem.read_u32(TLS + 0x20).unwrap();
        assert_eq!(size as usize, font.len(), "GetSize reports the font's length");

        cpu.pl_request(TLS, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "font is at offset 0");

        // GetSharedFontInOrderOfPriority: one request, three out map-aliases.
        let mut cpu = Cpu::new();
        cpu.set_shared_font(font.clone());
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(BUFFERS, 0x100).unwrap();
        cpu.mem.write_u32(TLS, 4 | (3 << 24)).unwrap(); // 3 recv buffers
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        for i in 0..3u32 {
            let at = TLS + 8 + 12 * i;
            cpu.mem.write_u32(at, 4 * 6).unwrap(); // size: PlSharedFontType_Total
            cpu.mem.write_u32(at + 4, BUFFERS + 0x40 * i).unwrap();
        }
        let data_area = cpu.ipc_reply_start(TLS);
        cpu.mem.write_u32(TLS + data_area, SFCI).unwrap();
        cpu.mem.write_u32(TLS + data_area + 8, 5).unwrap();
        cpu.pl_request(TLS, Some(5)).unwrap();

        assert_eq!(cpu.mem.read_u32(BUFFERS).unwrap(), 0); // Standard
        assert_eq!(cpu.mem.read_u32(BUFFERS + 0x40).unwrap(), 0); // offset
        assert_eq!(cpu.mem.read_u32(BUFFERS + 0x80).unwrap() as usize, font.len());
        // { u8 fonts_loaded, u8 pad[3], s32 total_fonts }
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1);
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 1);
    }

    #[test]
    fn without_a_font_pl_reports_an_empty_set() {
        // A guest must get a well-formed "no fonts" answer rather than spin in
        // `_plRequestLoadWait` or read a font that isn't there.
        let mut cpu = request(false, 5, &[]);
        cpu.pl_request(TLS, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "reported as loaded");
        cpu.pl_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 0, "no fonts");
    }

    #[test]
    fn a_closed_session_forgets_its_recorded_state() {
        let mut cpu = Cpu::new();
        cpu.record_handle(9, "fsp-srv");
        cpu.record_domain_object(9, 1, "fsp-srv-fs");
        cpu.fs_files.insert(Cpu::object_key(9, 1), "/a.txt".to_owned());
        cpu.record_handle(10, "vi:m");
        cpu.forget_handle(9);
        assert!(cpu.service_name(9).is_none());
        assert!(cpu.domain_interface(9, 1).is_none());
        assert!(cpu.fs_files.is_empty());
        assert_eq!(cpu.service_name(10), Some("vi:m"));
    }

    #[test]
    fn civil_days_round_trip_the_epoch_and_a_leap_day() {
        use super::{civil_from_days, days_from_civil};
        // 1970-01-01 is day 0 by definition, and was a Thursday.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        for &(y, m, d) in &[
            (1969, 12, 31), // just before the epoch
            (2024, 2, 29),  // a leap day
            (2001, 9, 9),   // the "1 billion seconds" date
            (1900, 1, 1),   // not a leap year despite ending in 00
            (2000, 2, 29),  // is a leap year (divisible by 400)
        ] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m, d), "{y}-{m}-{d} (days={days})");
        }
    }

    #[test]
    fn to_calendar_time_matches_the_epoch_and_a_known_date() {
        let epoch = Cpu::to_calendar_time(0);
        assert_eq!(&epoch[0..2], &1970u16.to_le_bytes()[..]);
        assert_eq!(epoch[2], 1); // month
        assert_eq!(epoch[3], 1); // day
        assert_eq!(epoch[4], 0); // hour
        assert_eq!(epoch[5], 0); // minute
        assert_eq!(epoch[6], 0); // second
        assert_eq!(u32::from_le_bytes(epoch[8..12].try_into().unwrap()), 4); // Thursday
        assert_eq!(u32::from_le_bytes(epoch[12..16].try_into().unwrap()), 0);

        // The well-known "1 billion seconds" moment: 2001-09-09 01:46:40 UTC.
        let billion = Cpu::to_calendar_time(1_000_000_000);
        assert_eq!(&billion[0..2], &2001u16.to_le_bytes()[..]);
        assert_eq!(billion[2], 9);
        assert_eq!(billion[3], 9);
        assert_eq!(billion[4], 1);
        assert_eq!(billion[5], 46);
        assert_eq!(billion[6], 40);

        assert_eq!(Cpu::from_calendar_time(2001, 9, 9, 1, 46, 40), 1_000_000_000);
    }

    #[test]
    fn system_clock_get_current_time_reports_the_host_supplied_value() {
        let mut cpu = request(false, 0, &[]);
        cpu.set_unix_time(1_700_000_000);
        cpu.time_system_clock_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, 1_700_000_000);
    }

    #[test]
    fn timezone_service_converts_posix_time_to_calendar_time_over_ipc() {
        let mut cpu = request(false, 101, &1_000_000_000i64.to_le_bytes());
        cpu.time_timezone_request(TLS, Some(101)).unwrap();
        assert_eq!(cpu.mem.read_u16(TLS + 0x20).unwrap(), 2001);
        assert_eq!(cpu.mem.read_u8(TLS + 0x22).unwrap(), 9); // month
        assert_eq!(cpu.mem.read_u8(TLS + 0x23).unwrap(), 9); // day
    }

    #[test]
    fn psm_reports_the_host_supplied_battery_level() {
        let mut cpu = request(false, 0, &[]);
        cpu.set_battery(42, false);
        cpu.psm_request(TLS, Some(0), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 42);

        let mut cpu = request(false, 1, &[]);
        cpu.set_battery(42, false);
        cpu.psm_request(TLS, Some(1), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "not charging -> Unconnected");

        cpu.set_battery(100, true);
        cpu.psm_request(TLS, Some(1), 9).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "charging -> EnoughPower");
    }

    #[test]
    fn set_sys_get_serial_number_returns_a_nul_padded_placeholder() {
        let mut cpu = request(false, 68, &[]);
        cpu.set_sys_request(TLS, Some(68)).unwrap();
        let mut got = [0u8; 0x18];
        for (i, byte) in got.iter_mut().enumerate() {
            *byte = cpu.mem.read_u8(TLS + 0x20 + i as u32).unwrap();
        }
        assert!(got.starts_with(b"XAW00000000000"));
        assert_eq!(got[b"XAW00000000000".len()], 0, "NUL-padded, not garbage");
    }

    #[test]
    fn set_get_region_code_reports_usa() {
        let mut cpu = request(false, 4, &[]);
        cpu.set_request(TLS, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "SetRegion_USA");
    }
}
