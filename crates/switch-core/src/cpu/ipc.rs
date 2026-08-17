//! Horizon IPC: parsing CMIF/HIPC requests out of the TLS message buffer,
//! synthesizing replies, and the service implementations behind the
//! session handles homebrew opens (`sm:`, `fsp-srv`, `vi:m`, `nvdrv`).

use super::Cpu;
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

    pub(super) fn stub_sm(&mut self, tls: u32, cmd_id: Option<u32>, _handle: u64) -> Result<()> {
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

    pub(super) fn stub_fsp_srv(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
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
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IFileSystem`, backed by the emulated SD card in [`crate::vfs`].
    ///
    /// Paths arrive in the request's first static buffer, so every command
    /// resolves against the real tree: a missing path reports
    /// `FsError_PathNotFound` rather than pretending to succeed, which is what
    /// stops a menu from recursing forever into directories that do not exist.
    pub(super) fn stub_fs(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
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
    pub(super) fn stub_fs_dir(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
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
    pub(super) fn stub_fs_file(&mut self, tls: u32, cmd_id: Option<u32>, key: u64) -> Result<()> {
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
    /// There are no Nintendo fonts here, so the set is reported as loaded but
    /// empty — a guest gets a well-formed "no font data" answer instead of
    /// spinning on `GetLoadState`. Font-rendering homebrew will have to fall
    /// back to whatever it ships itself.
    pub(super) fn stub_pl(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // RequestLoad(u32 SharedFontType)
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetLoadState(u32) -> u32 (1 = Loaded)
            Some(1) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // GetSize(u32) -> u32, GetSharedMemoryAddressOffset(u32) -> u32
            Some(2) | Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetSharedMemoryNativeHandle -> a shared memory handle, which
            // `svcMapSharedMemory` backs with zeroed pages.
            Some(4) => {
                let handle = self.alloc_handle();
                self.write_ipc_response(tls, 0, &[handle], &[], &[])
            }
            // GetSharedFontInOrderOfPriority(u64 LanguageCode) ->
            // { u8 Loaded, u8 pad[3], u32 total_fonts } with three output
            // buffers; reporting zero fonts is consistent with the empty set.
            Some(5) => {
                let mut raw = [0u8; 8];
                raw[0] = 1; // Loaded
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    pub(super) fn stub_set(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
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

    pub(super) fn stub_vi(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
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
        if self.ipc_message_type(tls) == 5 {
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
}
