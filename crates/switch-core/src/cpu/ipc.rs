//! Horizon IPC: parsing CMIF/HIPC/TIPC requests from the TLS message buffer
//! and writing replies, plus `sm:` and a few trivial services.

use super::{Cpu, GapKind, ServiceGap};
use crate::trace::Level;
use crate::Result;

/// Maximum distinct gaps [`Cpu::count_gap`] tracks between readings.
const GAP_CAP: usize = 64;

/// The process id `svcGetProcessId` and `pm` report.
const PROCESS_ID: u64 = 1;
/// Program id used until a loader sets one (the Album applet's).
pub(super) const DEFAULT_PROGRAM_ID: u64 = 0x0100_0000_0000_1000;

/// A request's buffers, as `(address, length)` pairs in descriptor order.
pub(super) type Buffers = Vec<(u32, u32)>;
/// The `DeviceId` `spl:` reports.
const SPL_DEVICE_ID: u64 = 0x0000_5357_4153_4D00;

/// What every session answers `QueryPointerBufferSize` with.
pub const POINTER_BUFFER_SIZE: u16 = 0x8000;

/// Pick between the static and map-alias descriptors of an AutoSelect buffer.
fn ipc_pick_buffer(map: Option<(u32, u32)>, pointer: Option<(u32, u32)>) -> Option<(u32, u32)> {
    match (map, pointer) {
        (Some((_, 0)), Some(pointer)) => Some(pointer),
        (map, pointer) => map.or(pointer),
    }
}

/// The counts packed into a hipc message header.
#[derive(Debug, Clone, Copy)]
pub(super) struct HipcHeader {
    /// Send-static ("pointer") descriptors.
    pub send_statics: u32,
    /// Map-alias descriptors, after the statics.
    pub send_buffers: u32,
    pub recv_buffers: u32,
    pub exch_buffers: u32,
    /// Words of raw data.
    pub data_words: u32,
    /// Receive-static descriptor count, decoded from the 0/2/2+n field.
    pub recv_statics: u32,
}

impl Cpu {
    pub(super) fn ipc_message_type(&self, tls: u32) -> u32 {
        self.mem.read_u32(tls).unwrap_or(0) & 0xFFFF
    }

    /// Whether the request is TIPC (command id in the type field as `16 + cmd`).
    pub(super) fn ipc_is_tipc_request(&self, tls: u32) -> bool {
        self.ipc_message_type(tls) >= 16
    }

    pub(super) fn ipc_header(&self, tls: u32) -> HipcHeader {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        HipcHeader {
            send_statics: (hdr1 >> 16) & 0xf,
            send_buffers: (hdr1 >> 20) & 0xf,
            recv_buffers: (hdr1 >> 24) & 0xf,
            exch_buffers: (hdr1 >> 28) & 0xf,
            data_words: hdr2 & 0x3ff,
            recv_statics: match (hdr2 >> 10) & 0xf {
                0 | 1 => 0,
                2 => 1,
                mode => mode - 2,
            },
        }
    }

    /// Offset of the descriptor area, past the header and optional special header.
    pub(super) fn ipc_descriptor_start(&self, tls: u32) -> u32 {
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let mut off = 8u32;
        if (hdr2 >> 31) & 1 != 0 {
            let special = self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0);
            off += 4;
            if special & 1 != 0 {
                off += 8; // pid
            }
            // `HipcSpecialHeader { send_pid:1, num_copy_handles:4, num_move_handles:4 }`
            off += 4 * (((special >> 1) & 0xf) + ((special >> 5) & 0xf));
        }
        off
    }

    /// The data area past the descriptors, where TIPC arguments begin.
    pub(super) fn ipc_data_area(&self, tls: u32) -> u32 {
        let header = self.ipc_header(tls);
        self.ipc_descriptor_start(tls)
            + 8 * header.send_statics
            + 12 * (header.send_buffers + header.recv_buffers + header.exch_buffers)
    }

    /// Start of a CMIF reply: the data area aligned to 16 bytes.
    pub(super) fn ipc_reply_start(&self, tls: u32) -> u32 {
        (self.ipc_data_area(tls) + 15) & !15
    }

    /// Offset of a CMIF request's `SFCI` header in the TLS buffer.
    pub(super) fn ipc_cmif_header_offset(&self, tls: u32) -> Option<u32> {
        const SFCI: u32 = 0x4943_4653;
        // The data area, or 0x10 further in for a domain request.
        let start = self.ipc_reply_start(tls);
        for candidate in [start, start.wrapping_add(0x10)] {
            if self.mem.read_u32(tls.wrapping_add(candidate)).unwrap_or(0) == SFCI {
                return Some(candidate);
            }
        }
        // Otherwise scan the whole message buffer.
        (0..0x100u32)
            .step_by(4)
            .find(|&i| self.mem.read_u32(tls.wrapping_add(i)).unwrap_or(0) == SFCI)
    }

    /// The request's command id, or `None` if it isn't a recognized request.
    pub(super) fn ipc_command_id(&self, tls: u32) -> Option<u32> {
        if self.ipc_is_tipc_request(tls) {
            return Some(self.ipc_message_type(tls) - 16);
        }
        if let Some(offset) = self.ipc_cmif_header_offset(tls) {
            return self.mem.read_u32(tls.wrapping_add(offset + 8)).ok();
        }
        // Pre-CMIF sessions: {type=2, object_id, cmd_id, ...} with no SFCI magic.
        let start = self.ipc_reply_start(tls);
        if self.mem.read_u32(tls.wrapping_add(start)).unwrap_or(0) == 2 {
            return self.mem.read_u32(tls.wrapping_add(start + 8)).ok();
        }
        None
    }

    /// Address of a request's payload, past the `CmifInHeader` and any domain header.
    pub(super) fn ipc_request_data(&self, tls: u32) -> u32 {
        if self.ipc_is_tipc_request(tls) {
            return tls.wrapping_add(self.ipc_data_area(tls));
        }
        match self.ipc_cmif_header_offset(tls) {
            Some(offset) => tls.wrapping_add(offset + 0x10),
            None => tls.wrapping_add(self.ipc_reply_start(tls) + 0x10),
        }
    }

    /// The `u8` argument at `offset` bytes into a request's payload.
    pub(super) fn ipc_arg_u8(&self, tls: u32, offset: u32) -> u8 {
        let data = self.ipc_request_data(tls);
        self.mem.read_u8(data.wrapping_add(offset)).unwrap_or(0)
    }

    /// The `u32` argument at `offset` bytes into a request's payload.
    pub(super) fn ipc_arg_u32(&self, tls: u32, offset: u32) -> u32 {
        let data = self.ipc_request_data(tls);
        self.mem.read_u32(data.wrapping_add(offset)).unwrap_or(0)
    }

    /// The `u64` argument at `offset` bytes into a request's payload.
    pub(super) fn ipc_arg_u64(&self, tls: u32, offset: u32) -> u64 {
        let data = self.ipc_request_data(tls);
        self.mem.read_u64(data.wrapping_add(offset)).unwrap_or(0)
    }

    /// The `f32` argument at `offset` bytes into a request's payload.
    pub(super) fn ipc_arg_f32(&self, tls: u32, offset: u32) -> f32 {
        let data = self.ipc_request_data(tls);
        f32::from_bits(self.mem.read_u32(data.wrapping_add(offset)).unwrap_or(0))
    }

    /// The `slot`-th map-alias descriptor (send, recv, then exchange), low 32 bits only.
    fn ipc_map_descriptor(&self, tls: u32, slot: u32) -> (u32, u32) {
        let at = self.ipc_descriptor_start(tls) + 8 * self.ipc_header(tls).send_statics + 12 * slot;
        let size = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
        let address = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
        (address, size)
    }

    /// The `index`-th map-alias send buffer.
    pub(super) fn ipc_send_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        (index < self.ipc_header(tls).send_buffers).then(|| self.ipc_map_descriptor(tls, index))
    }

    /// The `index`-th map-alias receive buffer.
    pub(super) fn ipc_recv_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        let header = self.ipc_header(tls);
        (index < header.recv_buffers)
            .then(|| self.ipc_map_descriptor(tls, header.send_buffers + index))
    }

    /// The send-static ("pointer") buffers, as `(address, size)`.
    pub(super) fn ipc_static_buffers(&self, tls: u32) -> Vec<(u32, u32)> {
        let start = self.ipc_descriptor_start(tls);
        (0..self.ipc_header(tls).send_statics)
            .map(|index| {
                let at = start + 8 * index;
                let packed = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
                let address = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
                (address, packed >> 16)
            })
            .collect()
    }

    /// The receive-static output buffers, which sit after the raw data.
    pub(super) fn ipc_recv_static_buffers(&self, tls: u32) -> Vec<(u32, u32)> {
        let header = self.ipc_header(tls);
        let start = self.ipc_data_area(tls) + 4 * header.data_words;
        (0..header.recv_statics)
            .map(|index| {
                let at = start + 8 * index;
                let address = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
                let packed = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
                (address, packed >> 16)
            })
            .collect()
    }

    /// The `index`-th input buffer, sent as either a pointer or a map-alias buffer.
    pub(super) fn ipc_input_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        ipc_pick_buffer(
            self.ipc_send_buffer(tls, index),
            self.ipc_static_buffers(tls).get(index as usize).copied(),
        )
    }

    /// The `index`-th output buffer, either a map-alias or a receive-static buffer.
    pub(super) fn ipc_output_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        ipc_pick_buffer(
            self.ipc_recv_buffer(tls, index),
            self.ipc_recv_static_buffers(tls)
                .get(index as usize)
                .copied(),
        )
    }

    /// Address of the `index`-th output buffer.
    pub(super) fn ipc_output_buffer_addr(&self, tls: u32, index: u32) -> Option<u32> {
        self.ipc_output_buffer(tls, index)
            .map(|(address, _)| address)
    }

    /// Every input and output buffer a request carries.
    pub(super) fn ipc_buffers(&self, tls: u32) -> (Buffers, Buffers) {
        let header = self.ipc_header(tls);
        let send = (0..header.send_buffers.max(header.send_statics))
            .filter_map(|index| self.ipc_input_buffer(tls, index))
            .collect();
        let recv = (0..header.recv_buffers.max(header.recv_statics))
            .filter_map(|index| self.ipc_output_buffer(tls, index))
            .collect();
        (send, recv)
    }

    /// The path in the first static buffer, normalized (`sdmc:/switch/` becomes `/switch`).
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
        if trimmed.is_empty() {
            "/".to_owned()
        } else {
            format!("/{}", trimmed)
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

    /// Fill the `index`-th output buffer with zeros.
    pub(super) fn zero_output_buffer(&mut self, tls: u32, index: u32) {
        let Some((addr, size)) = self.ipc_output_buffer(tls, index) else {
            return;
        };
        for offset in 0..size {
            let _ = self.mem.write_u8(addr.wrapping_add(offset), 0);
        }
    }

    /// Write `bytes` into the `index`-th output buffer, truncated; returns bytes written.
    pub(super) fn write_output_buffer(&mut self, tls: u32, index: u32, bytes: &[u8]) -> u32 {
        let Some((addr, size)) = self.ipc_output_buffer(tls, index) else {
            return 0;
        };
        let written = bytes.len().min(size as usize);
        for (offset, &byte) in bytes.iter().take(written).enumerate() {
            let _ = self.mem.write_u8(addr.wrapping_add(offset as u32), byte);
        }
        written as u32
    }

    /// Write a HIPC response; `domain_objects` makes it a domain reply.
    pub(super) fn write_ipc_response(
        &mut self,
        tls: u32,
        result: u32,
        move_handles: &[u64],
        raw_data: &[u8],
        domain_objects: &[u32],
    ) -> Result<()> {
        self.write_ipc_reply(tls, result, &[], move_handles, raw_data, domain_objects)
    }

    /// [`Cpu::write_ipc_response`] with copy handles as well as move handles.
    pub(super) fn write_ipc_reply(
        &mut self,
        tls: u32,
        result: u32,
        copy_handles: &[u64],
        move_handles: &[u64],
        raw_data: &[u8],
        domain_objects: &[u32],
    ) -> Result<()> {
        self.last_ipc_result = Some(result);
        if result != 0 && crate::trace::enabled(crate::trace::Trace::Ipc) {
            let module = result & 0x1FF;
            let description = (result >> 9) & 0x1FFF;
            crate::traceln!(
                "[ipc] error {result:#x} (module {module}, description {description}) from {:?} cmd={:?}",
                self.service_name(self.read_zr(0)),
                self.ipc_command_id(tls)
            );
        }
        if self.ipc_is_tipc_request(tls) {
            return self.write_tipc_reply(tls, result, copy_handles, move_handles, raw_data);
        }
        let is_domain = self.ipc_is_domain_request(tls);
        // Reply type is 0; libtransistor rejects anything but 0 or 4.
        self.mem.write_u32(tls, 0)?;
        let has_handles = !copy_handles.is_empty() || !move_handles.is_empty();
        // { send_pid:1, num_copy:4, num_move:4 }
        let handle_desc = ((copy_handles.len() as u32) << 1) | ((move_handles.len() as u32) << 5);
        let raw_data_words = (raw_data.len() as u32).div_ceil(4);
        let object_words = ((domain_objects.len() as u32) * 4).div_ceil(4);
        // SFCO header + raw data + padding, plus the domain header and objects.
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
            // Copy handles come first, then move handles.
            for &h in copy_handles.iter().chain(move_handles) {
                self.mem.write_u32(tls.wrapping_add(off), h as u32)?;
                off += 4;
            }
        }
        // Align to 16 bytes.
        let pre = (16 - (off % 16)) % 16;
        off += pre;
        // Zero the declared section so unwritten out parameters don't leak request bytes.
        for i in 0..raw_section_words * 4 {
            self.mem.write_u8(tls.wrapping_add(off + i), 0)?;
        }
        if is_domain {
            self.mem
                .write_u32(tls.wrapping_add(off), domain_objects.len() as u32)?;
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
        for (i, &b) in raw_data.iter().enumerate() {
            self.mem.write_u8(tls.wrapping_add(off + i as u32), b)?;
        }
        off += raw_data_words * 4;
        // Domain object ids.
        for (i, &obj) in domain_objects.iter().enumerate() {
            self.mem
                .write_u32(tls.wrapping_add(off + (i as u32) * 4), obj)?;
        }
        Ok(())
    }

    /// A TIPC reply: the data words start with the `Result`, no SFCO header or alignment.
    fn write_tipc_reply(
        &mut self,
        tls: u32,
        result: u32,
        copy_handles: &[u64],
        move_handles: &[u64],
        raw_data: &[u8],
    ) -> Result<()> {
        let raw_words = 1 + raw_data.len().div_ceil(4) as u32;
        let has_handles = !copy_handles.is_empty() || !move_handles.is_empty();
        self.mem.write_u32(tls, 0)?;
        let mut header1 = raw_words;
        if has_handles {
            header1 |= 1 << 31;
        }
        self.mem.write_u32(tls.wrapping_add(4), header1)?;
        let mut off = 8u32;
        if has_handles {
            let desc = ((copy_handles.len() as u32) << 1) | ((move_handles.len() as u32) << 5);
            self.mem.write_u32(tls.wrapping_add(off), desc)?;
            off += 4;
            for &h in copy_handles.iter().chain(move_handles) {
                self.mem.write_u32(tls.wrapping_add(off), h as u32)?;
                off += 4;
            }
        }
        // Clear before filling, as in the CMIF path.
        for i in 0..raw_words * 4 {
            self.mem.write_u8(tls.wrapping_add(off + i), 0)?;
        }
        self.mem.write_u32(tls.wrapping_add(off), result)?;
        off += 4;
        for (i, &b) in raw_data.iter().enumerate() {
            self.mem.write_u8(tls.wrapping_add(off + i as u32), b)?;
        }
        Ok(())
    }

    /// Hand back a sub-interface as a domain object or a moved session handle.
    /// Returns the key its state is filed under.
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

    pub(super) fn alloc_handle(&mut self) -> u64 {
        let h = self.next_handle as u64;
        self.next_handle = self.next_handle.wrapping_add(1);
        h
    }

    pub(super) fn record_handle(&mut self, handle: u64, name: &str) {
        self.service_handles.insert(handle, name.to_owned());
    }

    /// Drop everything recorded for a closed session.
    pub(super) fn forget_handle(&mut self, handle: u64) {
        self.service_handles.remove(&handle);
        self.domain_objects.retain(|&(owner, _), _| owner != handle);
        let session = handle << 32;
        self.fs_files
            .retain(|&key, _| key & !0xFFFF_FFFF != session);
        self.fs_dirs.retain(|&key, _| key & !0xFFFF_FFFF != session);
        self.acc_profiles
            .retain(|&key, _| key & !0xFFFF_FFFF != session);
        self.erpt_readers
            .retain(|&key, _| key & !0xFFFF_FFFF != session);
        self.opus_decoders
            .retain(|&key, _| key & !0xFFFF_FFFF != session);
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

    /// The interface a request is addressed to, by domain object or session handle,
    /// or `root` if nothing named it.
    pub(super) fn ipc_interface(&self, tls: u32, handle: u64, root: &'static str) -> String {
        if self.ipc_is_domain_request(tls) {
            let object_id = self.ipc_domain_object_id(tls);
            self.domain_interface(handle, object_id)
                .unwrap_or(root)
                .to_owned()
        } else {
            self.service_name(handle).unwrap_or(root).to_owned()
        }
    }

    /// Key for the per-object state maps: `handle:object_id`.
    pub(super) fn object_key(handle: u64, object_id: u32) -> u64 {
        (handle << 32) | u64::from(object_id)
    }

    /// The state key of this request's object, as [`Cpu::reply_with_interface`] returned it.
    pub(super) fn ipc_object_key(&self, tls: u32, handle: u64) -> u64 {
        if self.ipc_is_domain_request(tls) {
            Self::object_key(handle, self.ipc_domain_object_id(tls))
        } else {
            Self::object_key(handle, 0)
        }
    }

    /// The key of the `index`-th object a request sends: a domain in-object or a moved session.
    pub(super) fn ipc_input_object_key(&self, tls: u32, handle: u64, index: u32) -> Option<u64> {
        if self.ipc_is_domain_request(tls) {
            let start = self.ipc_reply_start(tls);
            let count = self.mem.read_u8(tls.wrapping_add(start + 1)).ok()?;
            if index >= u32::from(count) {
                return None;
            }
            let data_size = u32::from(self.mem.read_u16(tls.wrapping_add(start + 2)).ok()?);
            let at = start + 0x10 + data_size + 4 * index;
            let object_id = self.mem.read_u32(tls.wrapping_add(at)).ok()?;
            Some(Self::object_key(handle, object_id))
        } else {
            if self.mem.read_u32(tls.wrapping_add(4)).ok()? >> 31 == 0 {
                return None;
            }
            let special = self.mem.read_u32(tls.wrapping_add(8)).ok()?;
            let copies = (special >> 1) & 0xf;
            let moves = (special >> 5) & 0xf;
            if index >= moves {
                return None;
            }
            let mut at = 12;
            if special & 1 != 0 {
                at += 8; // pid
            }
            let session = self
                .mem
                .read_u32(tls.wrapping_add(at + 4 * (copies + index)))
                .ok()?;
            Some(Self::object_key(u64::from(session), 0))
        }
    }

    /// Whether the request is a control message (type 5, or 7 with context).
    pub(super) fn ipc_is_control_request(&self, tls: u32) -> bool {
        matches!(self.ipc_message_type(tls), 5 | 7)
    }

    /// Whether the request is a domain message (domain header type byte 1).
    pub(super) fn ipc_is_domain_request(&self, tls: u32) -> bool {
        if self.ipc_is_tipc_request(tls) {
            return false;
        }
        let start = self.ipc_reply_start(tls);
        self.mem.read_u8(tls.wrapping_add(start)).unwrap_or(0) == 1
    }

    /// Whether the request is a domain close (type byte 2), which has no `CmifInHeader`.
    pub(super) fn ipc_is_domain_close(&self, tls: u32) -> bool {
        if self.ipc_is_tipc_request(tls) {
            return false;
        }
        self.mem
            .read_u8(tls.wrapping_add(self.ipc_reply_start(tls)))
            .unwrap_or(0)
            == 2
    }

    /// Forget one object and acknowledge the close.
    pub(super) fn close_domain_object(
        &mut self,
        tls: u32,
        handle: u64,
        object_id: u32,
    ) -> Result<()> {
        // Closes never reach `ssl_request`, so its context count drops here.
        if self.domain_interface(handle, object_id) == Some("ssl:context") {
            self.ssl_contexts = self.ssl_contexts.saturating_sub(1);
        }
        self.opus_decoders
            .remove(&Self::object_key(handle, object_id));
        self.domain_objects.remove(&(handle, object_id));
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    pub(super) fn ipc_domain_object_id(&self, tls: u32) -> u32 {
        let start = self.ipc_reply_start(tls);
        self.mem
            .read_u32(tls.wrapping_add(start + 4))
            .unwrap_or(0xFFFFFFFF)
    }

    pub(super) fn alloc_domain_object(&mut self) -> u32 {
        let id = self.next_domain_object_id;
        self.next_domain_object_id = id.wrapping_add(1);
        if id == 0 {
            1
        } else {
            id
        }
    }

    pub(super) fn record_domain_object(&mut self, handle: u64, object_id: u32, name: &str) {
        self.domain_objects
            .insert((handle, object_id), name.to_owned());
    }

    pub(super) fn domain_interface(&self, handle: u64, object_id: u32) -> Option<&str> {
        self.domain_objects
            .get(&(handle, object_id))
            .map(|s| s.as_str())
    }

    /// Answer a control message (`ConvertToDomain` and the rest); returns whether it was one.
    pub(super) fn ipc_answer_control(
        &mut self,
        tls: u32,
        handle: u64,
        name: &str,
        cmd_id: Option<u32>,
    ) -> Result<bool> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if !self.ipc_is_control_request(tls) {
            return Ok(false);
        }
        match cmd_id {
            Some(CONVERT_TO_DOMAIN) => {
                let obj = self.alloc_domain_object();
                self.record_domain_object(handle, obj, name);
                self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])?;
            }
            _ => self.write_ipc_response(tls, 0, &[], &POINTER_BUFFER_SIZE.to_le_bytes(), &[])?,
        }
        Ok(true)
    }

    /// An event a service object hands out, allocated on first request and reused.
    pub(super) fn kept_event(&mut self, purpose: &'static str, object: u64) -> u64 {
        if let Some(&event) = self.service_events.get(&(purpose, object)) {
            return event;
        }
        let event = self.alloc_event(purpose, false);
        self.service_events.insert((purpose, object), event);
        event
    }

    /// Refuse a command with `cmif`'s unknown-command-id result, warning once.
    pub(super) fn unimplemented_command(
        &mut self,
        tls: u32,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `cmif` module 10, description 221.
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        self.count_gap(GapKind::Refused, iface, cmd_id);
        if self.unimplemented_ipc.insert((iface.to_string(), cmd_id)) {
            let pc = self.pc;
            // Log the request's shape: data words and buffers.
            let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
            let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
            let statics = (hdr1 >> 16) & 0xf;
            let send = (hdr1 >> 20) & 0xf;
            let recv = (hdr1 >> 24) & 0xf;
            let recv_static = matches!((hdr2 >> 10) & 0xf, 2..) as u32;
            let words = hdr2 & 0x3ff;
            self.diagnostic(
                Level::Warn,
                &format!(
                    "[ipc] unimplemented: {iface} cmd={cmd_id:?} (pc={pc:#x}, {words} data words, \
                 buffers: {statics} static/{send} send/{recv} recv/{recv_static} recv-static)"
                ),
            );
        }
        self.write_ipc_response(tls, UNKNOWN_COMMAND_ID, &[], &[], &[])
    }

    /// Answer an unimplemented command with success, a reused sub-session or domain
    /// object, and an event nothing signals.
    pub(super) fn reply_with_fabricated_object(
        &mut self,
        tls: u32,
        handle: u64,
        name: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        self.warn_stub(
            name,
            cmd_id,
            "a fabricated out-object and an event nothing signals",
        );
        let key = (handle, cmd_id.unwrap_or(u32::MAX));
        let (object_id, sub, event) = match self.fabricated_objects.get(&key) {
            Some(&triple) => triple,
            None => {
                let object_id = self.next_object_id;
                self.next_object_id = object_id.wrapping_add(1);
                let sub = self.alloc_handle();
                let event = self.alloc_event("ipc:fabricated", true);
                self.fabricated_objects.insert(key, (object_id, sub, event));
                (object_id, sub, event)
            }
        };
        if self.ipc_is_domain_request(tls) {
            self.record_domain_object(handle, object_id, name);
            self.write_ipc_reply(
                tls,
                0,
                &[event],
                &[],
                &object_id.to_le_bytes(),
                &[object_id],
            )
        } else {
            self.record_handle(sub, name);
            self.write_ipc_reply(tls, 0, &[event], &[sub], &object_id.to_le_bytes(), &[])
        }
    }

    /// Warn once that a service has no implementation at all.
    pub(super) fn warn_no_implementation(&mut self, service: &str, cmd_id: Option<u32>) {
        self.count_gap(GapKind::Missing, service, cmd_id);
        if self.unimplemented_ipc.insert((service.to_string(), cmd_id)) {
            self.diagnostic(
                Level::Warn,
                &format!("[ipc] no implementation: {service} cmd={cmd_id:?}"),
            );
        }
    }

    /// Warn once that a command was answered with an invented value or an unsignalled event.
    pub(super) fn warn_stub(&mut self, iface: &str, cmd_id: Option<u32>, what: &str) {
        self.count_gap(GapKind::Stub, iface, cmd_id);
        if self.stubbed_ipc.insert((iface.to_string(), cmd_id)) {
            self.diagnostic(
                Level::Warn,
                &format!("[ipc] stub: {iface} cmd={cmd_id:?} ({what})"),
            );
        }
    }

    /// Count one call towards the next [`Cpu::take_service_gaps`].
    pub(super) fn count_gap(&mut self, kind: GapKind, name: &str, command: Option<u32>) {
        // Look up first so an already-counted pair doesn't allocate.
        if let Some(calls) = self
            .gap_calls
            .iter_mut()
            .find(|((k, n, c), _)| *k == kind && n == name && *c == command)
            .map(|(_, calls)| calls)
        {
            *calls += 1;
            return;
        }
        if self.gap_calls.len() < GAP_CAP {
            self.gap_calls.insert((kind, name.to_owned(), command), 1);
        }
    }

    /// Every gap counted since the last call, in kind and name order.
    pub fn take_service_gaps(&mut self) -> Vec<ServiceGap> {
        std::mem::take(&mut self.gap_calls)
            .into_iter()
            .map(|((kind, name, command), calls)| ServiceGap {
                kind,
                name,
                command,
                calls,
            })
            .collect()
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

    /// `csrng`: pseudo-random bytes (splitmix64 seeded from the emulated clock).
    pub(super) fn csrng_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // GenerateRandomBytes -> the bytes, in an output buffer.
            Some(0) => {
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        let mut offset = 0u32;
                        while offset < size {
                            let word = self.next_random_u64().to_le_bytes();
                            for &byte in word.iter().take((size - offset).min(8) as usize) {
                                self.mem.write_u8(addr.wrapping_add(offset), byte)?;
                                offset += 1;
                            }
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, "csrng", cmd_id),
        }
    }

    /// `spl:`: only `GetConfig`, answered as an Icosa retail unit.
    pub(super) fn spl_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // GetConfig(u32 ConfigItem) -> u64.
            Some(0) => {
                let item = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                let value: u64 = match item {
                    // DisableProgramVerification: verification is on.
                    0 => 0,
                    // DramId.
                    1 => 0,
                    // HardwareType: Icosa, the original console.
                    4 => 0,
                    // HardwareState: Production, not a development unit.
                    5 => 1,
                    // IsRecoveryBoot: this booted normally.
                    6 => 0,
                    // DeviceId: a fixed placeholder.
                    7 => SPL_DEVICE_ID,
                    // MemoryArrange: the standard 4 GiB layout.
                    9 => 0,
                    // IsDebugMode: no.
                    10 => 0,
                    // Everything else, including Atmosphère's 65000+ extensions, reads as zero.
                    _ => 0,
                };
                self.write_ipc_response(tls, 0, &[], &value.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "spl:", cmd_id),
        }
    }

    /// `pm:*`: the process manager, answering for the single application process.
    pub(super) fn pm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        let iface = self.service_name(handle).unwrap_or("pm:shell").to_string();
        match iface.as_str() {
            // IDebugMonitorInterface.
            "pm:dmnt" => match cmd_id {
                // GetJitDebugProcessIdList -> 0.
                Some(0) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                // GetProcessId / GetApplicationProcessId -> this process.
                Some(2) | Some(4) => {
                    self.write_ipc_response(tls, 0, &[], &PROCESS_ID.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IInformationInterface::GetProgramId(u64 pid) -> u64 program_id.
            "pm:info" => match cmd_id {
                Some(0) => {
                    let program_id = self.program_id;
                    self.write_ipc_response(tls, 0, &[], &program_id.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IBootModeInterface::GetBootMode -> Normal.
            "pm:bm" => match cmd_id {
                Some(0) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IShellInterface: NotifyBootFinished, GetApplicationProcessIdForShell.
            _ => match cmd_id {
                Some(7) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                Some(8) => self.write_ipc_response(tls, 0, &[], &PROCESS_ID.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `btm:sys` and its `IBtmSystemCore`: no radio, and nothing ever pairs.
    pub(super) fn btm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "btm:sys", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "btm:sys");
        match iface.as_str() {
            "btm:core" => match cmd_id {
                // StartGamepadPairing / CancelGamepadPairing.
                Some(0) | Some(1) => {
                    self.bt_gamepad_pairing = cmd_id == Some(0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ClearGamepadPairingDatabase: already empty.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetPairedGamepadCount -> u8.
                Some(3) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // EnableRadio / DisableRadio / IsRadioEnabled, backed by the system setting.
                Some(4) | Some(5) => {
                    let on = cmd_id == Some(4);
                    self.store_system_settings(|settings| settings.bluetooth_enable = on);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(6) => {
                    let enabled = u8::from(self.system_settings().bluetooth_enable);
                    self.write_ipc_response(tls, 0, &[], &[enabled], &[])
                }
                // AcquireRadioEvent / AcquireGamepadPairingEvent -> (acquired, event).
                Some(7) | Some(8) => {
                    let purpose = if cmd_id == Some(7) {
                        "btm:radio"
                    } else {
                        "btm:gamepad-pairing"
                    };
                    let event = self.kept_event(purpose, handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[1u8], &[])
                }
                // IsGamepadPairingStarted -> bool.
                Some(9) => {
                    let started = u8::from(self.bt_gamepad_pairing);
                    self.write_ipc_response(tls, 0, &[], &[started], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IBtmSystem: GetCore.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "btm:core")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `nfc:sys` and its `ISystem`: no reader attached; the enabled flag is a setting.
    pub(super) fn nfc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        /// `nn::nfc::State`.
        const STATE_NON_INITIALIZED: u32 = 0;
        const STATE_INITIALIZED: u32 = 1;
        if self.ipc_answer_control(tls, handle, "nfc:sys", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "nfc:sys");
        match iface.as_str() {
            "nfc:system" => match cmd_id {
                // Initialize / Finalize, and the 4.0.0+ System variants.
                Some(0) | Some(400) => {
                    self.nfc_initialized = true;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(1) | Some(401) => {
                    self.nfc_initialized = false;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetStateOld / GetState -> nn::nfc::State.
                Some(2) | Some(402) => {
                    let state = if self.nfc_initialized {
                        STATE_INITIALIZED
                    } else {
                        STATE_NON_INITIALIZED
                    };
                    self.write_ipc_response(tls, 0, &[], &state.to_le_bytes(), &[])
                }
                // IsNfcEnabledOld / IsNfcEnabled -> bool.
                Some(3) | Some(403) => {
                    let enabled = u8::from(self.system_settings().nfc_enable);
                    self.write_ipc_response(tls, 0, &[], &[enabled], &[])
                }
                // SetNfcEnabledOld / SetNfcEnabled(bool).
                Some(100) | Some(500) => {
                    let on = self.ipc_arg_u8(tls, 0) != 0;
                    self.store_system_settings(|settings| settings.nfc_enable = on);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ListDevices: none.
                Some(404) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                // AttachAvailabilityChangeEvent: never signalled.
                Some(407) => {
                    let event = self.kept_event("nfc:availability", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Everything else names a device, and there are none.
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemManager: CreateSystemInterface.
            _ => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "nfc:system")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `ngc:u` and `ngct:u`, the profanity filter: nothing is profane.
    pub(super) fn ngc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        /// Nonzero, though there is no word list.
        const CONTENT_VERSION: u32 = 1;
        const NOTHING_PROFANE: u32 = 0;

        let root = match self.service_name(handle) {
            Some("ngct:u") | Some("ngct:s") => "ngct:u",
            _ => "ngc:u",
        };
        if self.ipc_answer_control(tls, handle, root, cmd_id)? {
            return Ok(());
        }
        if root == "ngct:u" {
            return match cmd_id {
                // Match(text) -> false.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // Filter(text) -> the text unchanged.
                Some(1) => {
                    let text = self.input_text(tls);
                    self.write_output_buffer(tls, 0, &text);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, root, cmd_id),
            };
        }
        match cmd_id {
            // GetContentVersion -> u32.
            Some(0) => self.write_ipc_response(tls, 0, &[], &CONTENT_VERSION.to_le_bytes(), &[]),
            // Check / Check2 -> no flags.
            Some(1) | Some(4) => {
                self.write_ipc_response(tls, 0, &[], &NOTHING_PROFANE.to_le_bytes(), &[])
            }
            // Mask / Mask2 -> no flags, and the text written back unchanged.
            Some(2) | Some(5) => {
                let text = self.input_text(tls);
                self.write_output_buffer(tls, 0, &text);
                self.write_ipc_response(tls, 0, &[], &NOTHING_PROFANE.to_le_bytes(), &[])
            }
            // Reload: re-read the word list from system data. There is none.
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, root, cmd_id),
        }
    }

    /// The text a request carries in its first input buffer, as bytes.
    fn input_text(&self, tls: u32) -> Vec<u8> {
        match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) if addr != 0 => self.read_bytes(addr, size),
            _ => Vec::new(),
        }
    }

    /// `npns:s` / `npns:u`, the push-notification client: never connected.
    pub(super) fn npns_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        /// `nn::npns::State`.
        const STATE_NOT_CONNECTED: u32 = 0;

        let root = match self.service_name(handle) {
            Some("npns:u") => "npns:u",
            _ => "npns:s",
        };
        if self.ipc_answer_control(tls, handle, root, cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // ListenAll / ListenTo / ListenToByName / ListenToMyApplicationId.
            Some(1) | Some(2) | Some(8) | Some(26) => {
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetReceiveEvent / GetStateChangeEvent: never signalled.
            Some(5) | Some(7) => {
                let purpose = if cmd_id == Some(5) {
                    "npns:receive"
                } else {
                    "npns:state-change"
                };
                let event = self.kept_event(purpose, handle);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // GetState -> u32.
            Some(103) => {
                self.write_ipc_response(tls, 0, &[], &STATE_NOT_CONNECTED.to_le_bytes(), &[])
            }
            // GetLastNotifiedTime -> 0.
            Some(106) => self.write_ipc_response(tls, 0, &[], &0i64.to_le_bytes(), &[]),
            // Suspend / Resume: there is no session to suspend.
            Some(101) | Some(102) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Receive / ReceiveRaw are refused: the empty-queue error is undocumented.
            _ => self.unimplemented_command(tls, root, cmd_id),
        }
    }
}

/// Request builders shared by service tests.
#[cfg(test)]
pub(super) mod testing {
    use super::Cpu;

    pub(crate) const TLS: u32 = 0x2000;
    pub(crate) const SFCI: u32 = 0x4943_4653;

    /// A CMIF request with no buffer descriptors, optionally on a domain.
    pub(crate) fn request(domain: bool, command_id: u32, payload: &[u8]) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        marshal(&mut cpu, domain, command_id, payload);
        cpu
    }

    /// Marshal a request into the TLS buffer.
    pub(crate) fn marshal(cpu: &mut Cpu, domain: bool, command_id: u32, payload: &[u8]) {
        for i in (0..0x200u32).step_by(4) {
            cpu.mem.write_u32(TLS + i, 0).unwrap();
        }
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

        // A domain header pushes the payload 16 bytes further in.
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
        // Buffer descriptors push the CMIF header further in.
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
        cpu.write_ipc_response(TLS, 0, &[0x1234], &7u32.to_le_bytes(), &[])
            .unwrap();

        // Reply type is 0 (libtransistor rejects anything but 0 or 4).
        assert_eq!(cpu.mem.read_u32(TLS).unwrap() & 0xFFFF, 0);
        // Word 1: the raw-data word count with bit 31 set for the handle
        // descriptor that follows.
        let header1 = cpu.mem.read_u32(TLS + 4).unwrap();
        assert_eq!(header1 >> 31, 1);
        assert_eq!(header1 & 0x1FF, 4 + 1 + 4); // SFCO + one word + padding
        assert_eq!(cpu.mem.read_u32(TLS + 8).unwrap(), 1 << 5);
        assert_eq!(cpu.mem.read_u32(TLS + 12).unwrap(), 0x1234);
        // The data section is 16-byte aligned: SFCO, version, result, token,
        // then the payload.
        assert_eq!(cpu.mem.read_u32(TLS + 0x10).unwrap(), 0x4F43_4653);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 7);
    }

    #[test]
    fn an_auto_select_buffer_is_found_in_whichever_form_it_arrived() {
        // AutoSelect fills both descriptor forms and nulls the unused one.
        const BUFFER: u32 = 0x3000;
        const SIZE: u32 = 0x40;
        for through_pointer in [true, false] {
            let cpu = auto_select_request(1, &[], BUFFER, SIZE, through_pointer);
            assert_eq!(
                cpu.ipc_input_buffer(TLS, 0),
                Some((BUFFER, SIZE)),
                "input, through_pointer={through_pointer}"
            );
            assert_eq!(
                cpu.ipc_output_buffer(TLS, 0),
                Some((BUFFER, SIZE)),
                "output, through_pointer={through_pointer}"
            );
            assert_eq!(
                cpu.ipc_buffers(TLS),
                (vec![(BUFFER, SIZE)], vec![(BUFFER, SIZE)]),
                "both, through_pointer={through_pointer}"
            );
        }
    }

    #[test]
    fn a_static_buffer_is_found_past_the_handles_a_special_header_carries() {
        // The static descriptor sits past the special header's copy and move handles.
        const PATH: u32 = 0x3000;

        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(PATH, 0x100).unwrap();
        for (i, &byte) in b"/save/data.bin\0".iter().enumerate() {
            cpu.mem.write_u8(PATH + i as u32, byte).unwrap();
        }

        // A Request with one send-static, and a special header carrying two
        // copy handles and one move handle ahead of the descriptors.
        cpu.mem.write_u32(TLS, 4 | (1 << 16)).unwrap();
        cpu.mem.write_u32(TLS + 4, 8 | (1 << 31)).unwrap();
        cpu.mem.write_u32(TLS + 8, (2 << 1) | (1 << 5)).unwrap();
        for slot in 0..3 {
            cpu.mem
                .write_u32(TLS + 12 + slot * 4, 0xDEAD_0000 + slot)
                .unwrap();
        }
        // `{ index:6, address_high:6, address_mid:4, size:16 }`, then the
        // low word of the address.
        let descriptor = TLS + 12 + 3 * 4;
        cpu.mem.write_u32(descriptor, 0x10 << 16).unwrap();
        cpu.mem.write_u32(descriptor + 4, PATH).unwrap();

        assert_eq!(cpu.ipc_static_buffers(TLS), vec![(PATH, 0x10)]);
        assert_eq!(cpu.ipc_request_path(TLS), "/save/data.bin");
    }

    /// A CMIF request whose first send-static buffer carries a path.
    pub(crate) fn request_with_path(command_id: u32, path: &str, payload: &[u8]) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(PATH_AT, 0x400).unwrap();
        write_path_request(&mut cpu, command_id, path, payload);
        cpu
    }

    /// Where [`write_path_request`] parks the path it hands the service.
    const PATH_AT: u32 = 0x3800;

    /// [`request_with_path`] into an existing `Cpu`.
    pub(crate) fn write_path_request(cpu: &mut Cpu, command_id: u32, path: &str, payload: &[u8]) {
        for offset in (0..0x100u32).step_by(4) {
            cpu.mem.write_u32(TLS + offset, 0).unwrap();
        }
        for (i, &byte) in path.as_bytes().iter().enumerate() {
            cpu.mem.write_u8(PATH_AT + i as u32, byte).unwrap();
        }
        cpu.mem.write_u8(PATH_AT + path.len() as u32, 0).unwrap();
        cpu.mem.write_u32(TLS, 4 | (1 << 16)).unwrap(); // one send-static
        cpu.mem.write_u32(TLS + 4, 12).unwrap();
        cpu.mem
            .write_u32(TLS + 8, (path.len() as u32 + 1) << 16)
            .unwrap();
        cpu.mem.write_u32(TLS + 12, PATH_AT).unwrap();
        let at = TLS + 0x20;
        cpu.mem.write_u32(at, SFCI).unwrap();
        cpu.mem.write_u32(at + 8, command_id).unwrap();
        for (i, &byte) in payload.iter().enumerate() {
            cpu.mem.write_u8(at + 0x10 + i as u32, byte).unwrap();
        }
    }

    #[test]
    fn a_closed_session_forgets_its_recorded_state() {
        let mut cpu = Cpu::new();
        cpu.record_handle(9, "fsp-srv");
        cpu.record_domain_object(9, 1, "fsp-srv-fs");
        cpu.fs_files
            .insert(Cpu::object_key(9, 1), "/a.txt".to_owned());
        cpu.record_handle(10, "vi:m");
        cpu.forget_handle(9);
        assert!(cpu.service_name(9).is_none());
        assert!(cpu.domain_interface(9, 1).is_none());
        assert!(cpu.fs_files.is_empty());
        assert_eq!(cpu.service_name(10), Some("vi:m"));
    }

    /// Overwrite the TLS buffer with a fresh request, clearing it first.
    pub(crate) fn write_request(cpu: &mut Cpu, command_id: u32, payload: &[u8]) {
        for offset in (0..0x100u32).step_by(4) {
            cpu.mem.write_u32(TLS + offset, 0).unwrap();
        }
        cpu.mem.write_u32(TLS, 4).unwrap();
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        let at = TLS + 0x10;
        cpu.mem.write_u32(at, SFCI).unwrap();
        cpu.mem.write_u32(at + 8, command_id).unwrap();
        for (index, &byte) in payload.iter().enumerate() {
            cpu.mem.write_u8(at + 0x10 + index as u32, byte).unwrap();
        }
    }

    /// A CMIF request carrying one map-alias receive buffer.
    pub(crate) fn request_with_recv_buffer(
        command_id: u32,
        payload: &[u8],
        buffer: u32,
        size: u32,
    ) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        write_map_buffer_request(&mut cpu, command_id, payload, buffer, size, false);
        cpu
    }

    /// Write a request carrying one map-alias send or receive buffer.
    pub(crate) fn write_map_buffer_request(
        cpu: &mut Cpu,
        command_id: u32,
        payload: &[u8],
        buffer: u32,
        size: u32,
        send: bool,
    ) {
        let one = [(buffer, size)];
        let (send, recv) = if send {
            (&one[..], &[][..])
        } else {
            (&[][..], &one[..])
        };
        write_buffer_request(cpu, command_id, payload, send, recv);
    }

    /// Marshal a request carrying map-alias send and receive buffers.
    pub(crate) fn write_buffer_request(
        cpu: &mut Cpu,
        command_id: u32,
        payload: &[u8],
        send: &[(u32, u32)],
        recv: &[(u32, u32)],
    ) {
        for offset in (0..0x200u32).step_by(4) {
            cpu.mem.write_u32(TLS + offset, 0).unwrap();
        }
        let counts = ((send.len() as u32) << 20) | ((recv.len() as u32) << 24);
        cpu.mem.write_u32(TLS, 4 | counts).unwrap();
        cpu.mem.write_u32(TLS + 4, 16).unwrap();
        // Each descriptor: size, low address word, packed high bits.
        for (index, &(address, size)) in send.iter().chain(recv).enumerate() {
            let at = TLS + 8 + 12 * index as u32;
            cpu.mem.write_u32(at, size).unwrap();
            cpu.mem.write_u32(at + 4, address).unwrap();
            cpu.mem.write_u32(at + 8, 0).unwrap();
        }
        let descriptors = 8 + 12 * (send.len() + recv.len()) as u32;
        let at = TLS + descriptors.div_ceil(16) * 16;
        cpu.mem.write_u32(at, SFCI).unwrap();
        cpu.mem.write_u32(at + 8, command_id).unwrap();
        for (index, &byte) in payload.iter().enumerate() {
            cpu.mem.write_u8(at + 0x10 + index as u32, byte).unwrap();
        }
    }

    /// A CMIF request offering one receive-static output buffer, after the data words.
    pub(crate) fn request_with_recv_static(
        command_id: u32,
        payload: &[u8],
        buffer: u32,
        size: u32,
    ) -> Cpu {
        let mut cpu = request(false, command_id, payload);
        // Padding, the CmifInHeader, then the payload.
        let data_words = 2 + 4 + payload.len().div_ceil(4) as u32;
        // recv_static_mode = 2 + one buffer.
        cpu.mem.write_u32(TLS + 4, data_words | (3 << 10)).unwrap();
        let at = TLS + 8 + 4 * data_words;
        cpu.mem.write_u32(at, buffer).unwrap();
        cpu.mem.write_u32(at + 4, size << 16).unwrap();
        cpu
    }

    /// A CMIF request with one AutoSelect buffer; `through_pointer` picks which form is real.
    pub(crate) fn auto_select_request(
        command_id: u32,
        payload: &[u8],
        buffer: u32,
        size: u32,
        through_pointer: bool,
    ) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        for offset in (0..0x200u32).step_by(4) {
            cpu.mem.write_u32(TLS + offset, 0).unwrap();
        }
        let (pointer, alias) = if through_pointer {
            ((buffer, size), (0, 0))
        } else {
            ((0, 0), (buffer, size))
        };
        // One send-static, one send and one receive buffer.
        cpu.mem
            .write_u32(TLS, 4 | (1 << 16) | (1 << 20) | (1 << 24))
            .unwrap();
        cpu.mem.write_u32(TLS + 8, pointer.1 << 16).unwrap();
        cpu.mem.write_u32(TLS + 12, pointer.0).unwrap();
        for (index, &(address, len)) in [alias, alias].iter().enumerate() {
            let at = TLS + 16 + 12 * index as u32;
            cpu.mem.write_u32(at, len).unwrap();
            cpu.mem.write_u32(at + 4, address).unwrap();
            cpu.mem.write_u32(at + 8, 0).unwrap();
        }
        // Padding, the CmifInHeader, then the payload.
        let data_words = 2 + 4 + payload.len().div_ceil(4) as u32;
        cpu.mem.write_u32(TLS + 4, data_words | (3 << 10)).unwrap();
        let data_area = TLS + 40;
        cpu.mem.write_u32(data_area + 8, SFCI).unwrap();
        cpu.mem.write_u32(data_area + 16, command_id).unwrap();
        for (index, &byte) in payload.iter().enumerate() {
            cpu.mem
                .write_u8(data_area + 24 + index as u32, byte)
                .unwrap();
        }
        let recv_list = data_area + 4 * data_words;
        cpu.mem.write_u32(recv_list, pointer.0).unwrap();
        cpu.mem.write_u32(recv_list + 4, pointer.1 << 16).unwrap();
        cpu
    }

    /// Marshal a request carrying map-alias send buffers.
    pub(crate) fn write_send_buffer_request(
        cpu: &mut Cpu,
        command_id: u32,
        payload: &[u8],
        buffers: &[(u32, u32)],
    ) {
        write_buffer_request(cpu, command_id, payload, buffers, &[]);
    }

    #[test]
    fn csrng_fills_the_buffer_with_bytes_that_differ() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(0, &[], BUFFER, 0x20);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.set_unix_time(1_700_000_000);
        cpu.csrng_request(TLS, Some(0)).unwrap();
        let first = cpu.read_bytes(BUFFER, 0x20);
        assert_ne!(first, vec![0u8; 0x20], "the buffer was written");
        assert!(
            first.windows(8).any(|w| w != &first[..8]),
            "not one value repeated"
        );

        write_map_buffer_request(&mut cpu, 0, &[], BUFFER, 0x20, false);
        cpu.csrng_request(TLS, Some(0)).unwrap();
        assert_ne!(cpu.read_bytes(BUFFER, 0x20), first, "a second call differs");
    }

    #[test]
    fn spl_reports_a_retail_console() {
        // 4: HardwareType (Icosa), 5: HardwareState (Production), 10: IsDebugMode.
        for (item, expected) in [(4u32, 0u64), (5, 1), (10, 0)] {
            let mut cpu = request(false, 0, &item.to_le_bytes());
            cpu.spl_request(TLS, Some(0)).unwrap();
            assert_eq!(
                cpu.mem.read_u64(TLS + 0x20).unwrap(),
                expected,
                "config item {item}"
            );
        }
        // The device id is fixed and nonzero.
        let mut cpu = request(false, 0, &7u32.to_le_bytes());
        cpu.spl_request(TLS, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), super::SPL_DEVICE_ID);
    }

    /// A CMIF control request (message type 5).
    pub(crate) fn control_request(command_id: u32) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.write_u32(TLS, 5).unwrap(); // CmifCommandType_Control
        cpu.mem.write_u32(TLS + 4, 8).unwrap();
        cpu.mem.write_u32(TLS + 0x10, SFCI).unwrap();
        cpu.mem.write_u32(TLS + 0x18, command_id).unwrap();
        cpu
    }

    #[test]
    fn pm_agrees_with_the_kernel_about_which_process_this_is() {
        let mut cpu = request(false, 4, &[]);
        cpu.register_service_handle(9, "pm:dmnt");
        cpu.pm_request(TLS, 9, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), super::PROCESS_ID);

        // pm:info maps it to the program id.
        let mut cpu = request(false, 0, &super::PROCESS_ID.to_le_bytes());
        cpu.register_service_handle(9, "pm:info");
        cpu.set_program_id(0x0100_4890_117B_2000);
        cpu.pm_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0x0100_4890_117B_2000);
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;

    #[test]
    fn ngc_answers_a_version_rather_than_a_fabricated_object_id() {
        let mut cpu = request(false, 0, &[]);
        cpu.register_service_handle(9, "ngc:u");
        cpu.ngc_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "content version");

        // Check -> the flags saying what was found. Nothing was.
        write_request(&mut cpu, 1, &[]);
        cpu.ngc_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "nothing profane");
    }

    #[test]
    fn ngc_writes_the_masked_text_back_rather_than_leaving_the_buffer() {
        const IN: u32 = 0x4000;
        const OUT: u32 = 0x5000;
        const TEXT: &[u8] = b"a perfectly ordinary console name";

        let mut cpu = request(false, 2, &[]);
        cpu.mem.map_zero(IN, 0x200).unwrap();
        cpu.mem.map_zero(OUT, 0x200).unwrap();
        for (index, &byte) in TEXT.iter().enumerate() {
            cpu.mem.write_u8(IN + index as u32, byte).unwrap();
        }
        cpu.register_service_handle(9, "ngc:u");
        write_buffer_request(&mut cpu, 2, &[], &[(IN, TEXT.len() as u32)], &[(OUT, 0x80)]);
        cpu.ngc_request(TLS, 9, Some(2)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "nothing masked");
        assert_eq!(cpu.read_bytes(OUT, TEXT.len() as u32), TEXT);
    }

    #[test]
    fn ngct_matches_nothing_and_filters_nothing() {
        // Match is a bool, and Filter writes its buffer like Mask.
        const IN: u32 = 0x4000;
        const OUT: u32 = 0x5000;
        const TEXT: &[u8] = b"still ordinary";

        let mut cpu = request(false, 0, &[]);
        cpu.mem.map_zero(IN, 0x200).unwrap();
        cpu.mem.map_zero(OUT, 0x200).unwrap();
        for (index, &byte) in TEXT.iter().enumerate() {
            cpu.mem.write_u8(IN + index as u32, byte).unwrap();
        }
        cpu.register_service_handle(9, "ngct:u");
        cpu.ngc_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "nothing matched");

        write_buffer_request(&mut cpu, 1, &[], &[(IN, TEXT.len() as u32)], &[(OUT, 0x80)]);
        cpu.ngc_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.read_bytes(OUT, TEXT.len() as u32), TEXT);
    }

    #[test]
    fn npns_hands_back_the_same_receive_event_every_time() {
        let mut cpu = request(false, 5, &[]);
        cpu.register_service_handle(9, "npns:s");
        cpu.npns_request(TLS, 9, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        let first = cpu.mem.read_u32(TLS + 0x0c).unwrap();
        assert_ne!(first, 0, "no event came back");

        write_request(&mut cpu, 5, &[]);
        cpu.npns_request(TLS, 9, Some(5)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x0c).unwrap(), first, "same event");

        // And the state beside it: not connected, never notified.
        write_request(&mut cpu, 103, &[]);
        cpu.npns_request(TLS, 9, Some(103)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "GetState result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "not connected");
        write_request(&mut cpu, 106, &[]);
        cpu.npns_request(TLS, 9, Some(106)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "GetLastNotifiedTime result"
        );
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0, "never notified");
    }

    #[test]
    fn npns_refuses_to_invent_a_notification() {
        let mut cpu = request(false, 3, &[]);
        cpu.register_service_handle(9, "npns:s");
        const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
        cpu.npns_request(TLS, 9, Some(3)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), UNKNOWN_COMMAND_ID);
    }

    #[test]
    fn every_call_to_a_gap_is_counted_and_taken_once() {
        use crate::cpu::{Cpu, GapKind, ServiceGap};
        let mut cpu = Cpu::new();
        for _ in 0..3 {
            cpu.count_gap(GapKind::Stub, "am:IApplicationFunctions", Some(40));
        }
        cpu.count_gap(GapKind::Ioctl, "/dev/nvhost-gpu", Some(0x1b));
        assert_eq!(
            cpu.take_service_gaps(),
            vec![
                ServiceGap {
                    kind: GapKind::Stub,
                    name: "am:IApplicationFunctions".to_owned(),
                    command: Some(40),
                    calls: 3,
                },
                ServiceGap {
                    kind: GapKind::Ioctl,
                    name: "/dev/nvhost-gpu".to_owned(),
                    command: Some(0x1b),
                    calls: 1,
                },
            ]
        );
        assert!(cpu.take_service_gaps().is_empty(), "taken, not read");
    }

    #[test]
    fn failed_ioctls_are_counted_by_node_request_and_error() {
        let mut cpu = crate::cpu::Cpu::new();
        for _ in 0..3 {
            cpu.count_nv_error("/dev/nvhost-as-gpu", 0xC038_4106, 4);
        }
        cpu.count_nv_error("/dev/nvmap", 0xC008_0103, 4);
        assert_eq!(
            cpu.take_nv_errors(),
            vec![
                ("/dev/nvhost-as-gpu".to_owned(), 0xC038_4106, 4, 3),
                ("/dev/nvmap".to_owned(), 0xC008_0103, 4, 1),
            ]
        );
        assert!(cpu.take_nv_errors().is_empty(), "taken, not read");
    }

    #[test]
    fn gaps_past_the_cap_go_uncounted() {
        use crate::cpu::{Cpu, GapKind};
        let mut cpu = Cpu::new();
        for command in 0..super::GAP_CAP as u32 + 5 {
            cpu.count_gap(GapKind::Refused, "svc", Some(command));
        }
        assert_eq!(cpu.take_service_gaps().len(), super::GAP_CAP);
    }
}
