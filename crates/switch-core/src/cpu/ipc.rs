//! Horizon IPC: parsing CMIF/HIPC requests out of the TLS message buffer and
//! synthesizing the replies.
//!
//! This is the marshalling layer every service is built on — the descriptor
//! walks, the domain and control-message handling, the handle bookkeeping and
//! [`Cpu::write_ipc_reply`]. The services themselves live one module per
//! domain beside this one (`am`, `fs`, `vi`, `hid`, `net`, `audout`,
//! `audren`, …); `svc.rs` dispatches to them by session name.
//!
//! What is still *here* is `sm:`, which hands out every other session, and the
//! handful of services whose whole implementation is an answer or two —
//! `csrng`, `spl`, `pm`, `btm`, `nfc`. Alongside them are the fallbacks every
//! service falls back *to*: [`Cpu::unimplemented_command`] and
//! [`Cpu::reply_with_fabricated_object`].

use super::Cpu;
use crate::Result;

/// The process id `svcGetProcessId` reports, and so the one `pm` has to
/// report for the application: there is one process here, and two answers to
/// "which process is this" would be one too many.
const PROCESS_ID: u64 = 1;
/// The program id a guest runs under until a loader sets one. This is the
/// Album applet's, which is what hbmenu-launched homebrew runs as on real
/// hardware — not an invention, and not a title id belonging to somebody.
pub(super) const DEFAULT_PROGRAM_ID: u64 = 0x0100_0000_0000_1000;
/// The `DeviceId` `spl:` reports. A real console's is fused in at
/// manufacturing and unique; nothing here derives anything from it.
const SPL_DEVICE_ID: u64 = 0x0000_5357_4153_4D00;

/// The counts packed into a hipc message header, decoded once.
///
/// Seven separate walks over the TLS buffer need some subset of these, and
/// each used to re-derive them with its own shifts. That is how
/// [`Cpu::ipc_static_buffers`] came to skip a special header's pid but not the
/// copy and move handles behind it — four bytes short per handle, against a
/// [`Cpu::ipc_descriptor_start`] that had been fixed to skip both.
#[derive(Debug, Clone, Copy)]
pub(super) struct HipcHeader {
    /// Send-static ("pointer") descriptors, which sit first.
    pub send_statics: u32,
    /// Map-alias descriptors, in the order they appear after the statics.
    pub send_buffers: u32,
    pub recv_buffers: u32,
    pub exch_buffers: u32,
    /// Words of raw data, which the receive-static descriptors sit past.
    pub data_words: u32,
    /// How many receive-static descriptors the request offers for output. The
    /// field encodes this rather than counting it: 0 for none, 2 for a single
    /// buffer the server sizes, and `2 + count` otherwise.
    pub recv_statics: u32,
}

impl Cpu {
    // Reading a request: the headers, and where each part of it starts.
    pub(super) fn ipc_message_type(&self, tls: u32) -> u32 {
        self.mem.read_u32(tls).unwrap_or(0) & 0xFFFF
    }

    /// Whether the request is a **TIPC** message rather than a CMIF one.
    ///
    /// TIPC is the lighter serialization Nintendo moved `sm:` to in 12.0.0:
    /// the same hipc header, but the command id is carried *in the type
    /// field* as `16 + command`, and the data area holds the arguments
    /// directly — no `SFCI` header, no 16-byte alignment, and no domains. A
    /// type below 16 is one of the eight hipc command types and therefore
    /// CMIF; anything at or above it is TIPC.
    ///
    /// Every system module built against a new enough SDK talks to `sm:` this
    /// way, and this emulator only understood CMIF: `cabinet`'s very first
    /// request came back with no command id at all, so `sm:` answered
    /// whatever the fallthrough answered and the applet aborted before its
    /// second syscall.
    pub(super) fn ipc_is_tipc_request(&self, tls: u32) -> bool {
        self.ipc_message_type(tls) >= 16
    }

    /// Decode the two header words at the top of a hipc message.
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

    /// Offset of a hipc request's descriptor area — its first send-static
    /// descriptor — walking the header the way libnx's `hipcParseRequest`
    /// does: the 8-byte message header, then the optional special header with
    /// whatever it declares.
    ///
    /// The special header is not just a pid flag. It also carries copy and
    /// move handle counts, and those handles sit between it and the
    /// descriptors, so skipping only the pid leaves every offset derived from
    /// here four bytes short per handle. Every request in this emulator's path
    /// until now either had no special header or carried nothing but a pid in
    /// it, which is why that went unnoticed.
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

    /// The request's data area: past the header, the descriptors and the
    /// buffers, with no further alignment. This is where a **TIPC** message's
    /// arguments begin.
    pub(super) fn ipc_data_area(&self, tls: u32) -> u32 {
        let header = self.ipc_header(tls);
        self.ipc_descriptor_start(tls)
            + 8 * header.send_statics
            + 12 * (header.send_buffers + header.recv_buffers + header.exch_buffers)
    }

    /// Compute where a **CMIF** reply starts in the TLS IPC buffer, mirroring
    /// libnx's `cmifGetAlignedDataStart`: the data area, rounded up to 16
    /// bytes. TIPC does no such rounding — see [`Cpu::ipc_is_tipc_request`].
    pub(super) fn ipc_reply_start(&self, tls: u32) -> u32 {
        (self.ipc_data_area(tls) + 15) & !15
    }

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
        if self.ipc_is_tipc_request(tls) {
            return Some(self.ipc_message_type(tls) - 16);
        }
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
        if self.ipc_is_tipc_request(tls) {
            return tls.wrapping_add(self.ipc_data_area(tls));
        }
        match self.ipc_cmif_header_offset(tls) {
            Some(offset) => tls.wrapping_add(offset + 0x10),
            None => tls.wrapping_add(self.ipc_reply_start(tls) + 0x10),
        }
    }

    /// The `u8` argument at `offset` bytes into a request's payload — a bool,
    /// or one of the small enums these services take.
    pub(super) fn ipc_arg_u8(&self, tls: u32, offset: u32) -> u8 {
        let data = self.ipc_request_data(tls);
        self.mem.read_u8(data.wrapping_add(offset)).unwrap_or(0)
    }

    /// The `u32` argument at `offset` bytes into a request's payload.
    pub(super) fn ipc_arg_u32(&self, tls: u32, offset: u32) -> u32 {
        let data = self.ipc_request_data(tls);
        self.mem.read_u32(data.wrapping_add(offset)).unwrap_or(0)
    }

    /// The float argument at `offset` bytes into a request's payload. Every
    /// `lbl` setter takes one, and a float read as an integer is a brightness
    /// of 1065353216.
    pub(super) fn ipc_arg_f32(&self, tls: u32, offset: u32) -> f32 {
        let data = self.ipc_request_data(tls);
        f32::from_bits(self.mem.read_u32(data.wrapping_add(offset)).unwrap_or(0))
    }

    // Its buffers. A caller marshals one of these four ways and a service
    // that reads only the form it expects reads nothing at all, so
    // `ipc_input_buffer`/`ipc_output_buffer` are what a service should reach
    // for unless it knows which form it is being sent.
    /// The `slot`-th map-alias descriptor, as `(address, size)`. Slots are
    /// numbered across the send, receive and exchange descriptors in that
    /// order, which is the order they sit in.
    ///
    /// Each is three words — `{size_low, address_low, packed}` — where the
    /// packed word holds the mode in bits 0..1, address bits 36..57 in bits
    /// 2..23, size bits 32..35 in bits 24..27 and address bits 32..35 in bits
    /// 28..31. **Only the low 32 bits are read.** Guest memory here is
    /// `u32`-indexed, so everything the packed word contributes lands above
    /// bit 32; one of the three walks this replaces reconstructed the full
    /// 57-bit address and then returned it `as u32`, which is the same answer
    /// by a longer route.
    fn ipc_map_descriptor(&self, tls: u32, slot: u32) -> (u32, u32) {
        let at = self.ipc_descriptor_start(tls) + 8 * self.ipc_header(tls).send_statics + 12 * slot;
        let size = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
        let address = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
        (address, size)
    }

    /// The `index`-th map-alias **send** buffer, as `(address, size)`. These
    /// sit before the receive buffers.
    pub(super) fn ipc_send_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        (index < self.ipc_header(tls).send_buffers).then(|| self.ipc_map_descriptor(tls, index))
    }

    /// The `index`-th map-alias **receive** buffer as `(address, size)`. The
    /// size matters when the reply's length is whatever fits —
    /// `GetReleasedAudioOutBuffer` hands back as many tags as the guest left
    /// room for.
    pub(super) fn ipc_recv_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        let header = self.ipc_header(tls);
        (index < header.recv_buffers)
            .then(|| self.ipc_map_descriptor(tls, header.send_buffers + index))
    }

    /// Address of the `index`-th map-alias receive buffer, for the callers
    /// that write into one and never ask how big it is.
    pub(super) fn ipc_recv_buffer_addr(&self, tls: u32, index: u32) -> Option<u32> {
        self.ipc_recv_buffer(tls, index).map(|(address, _)| address)
    }

    /// Every map-alias buffer descriptor in a hipc request, as
    /// `(send, receive)` lists of `(address, size)`.
    pub(super) fn ipc_map_buffers(&self, tls: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
        let header = self.ipc_header(tls);
        let send = (0..header.send_buffers)
            .map(|slot| self.ipc_map_descriptor(tls, slot))
            .collect();
        let recv = (0..header.recv_buffers)
            .map(|slot| self.ipc_map_descriptor(tls, header.send_buffers + slot))
            .collect();
        (send, recv)
    }

    /// The send-static ("pointer") buffers of a hipc request, as
    /// `(address, size)`.
    ///
    /// Each descriptor is two words: `{ index:6, address_high:6,
    /// address_mid:4, size:16 }` then the low 32 bits of the address. Services
    /// that take a path — all of `fsp-srv`'s — send it this way rather than as
    /// a map-alias buffer.
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

    /// The receive-static ("pointer") buffers a request offers for **output**,
    /// as `(address, size)`.
    ///
    /// These are the only descriptors that sit *after* the raw data rather
    /// than before it, at the data area plus the words of payload — which
    /// counts the padding that aligns the CMIF header, so the walk lands past
    /// it either way.
    ///
    /// `IProfile::Get` is why this exists — its `AccountUserData` comes back
    /// through a fixed-size pointer buffer rather than a map-alias one.
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

    /// The `index`-th **input** buffer, whichever form the caller marshalled
    /// it in: `libnx`'s AutoSelect picks a send-static ("pointer") buffer when
    /// the server advertises room for one and a map-alias send buffer
    /// otherwise, so a service that answers `QueryPointerBufferSize` with a
    /// real size has to read both.
    pub(super) fn ipc_input_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        self.ipc_send_buffer(tls, index)
            .or_else(|| self.ipc_static_buffers(tls).get(index as usize).copied())
    }

    /// The `index`-th **output** buffer, whichever form the caller marshalled
    /// it in — the mirror of [`Cpu::ipc_input_buffer`]. A map-alias receive
    /// buffer if the request carries one, else a receive-static pointer
    /// buffer.
    pub(super) fn ipc_output_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        self.ipc_recv_buffer(tls, index)
            .or_else(|| self.ipc_recv_static_buffers(tls).get(index as usize).copied())
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

    /// Fill a request's `index`-th output buffer with zeros.
    ///
    /// A command whose whole answer is a struct in a buffer has to *write*
    /// that struct: the buffer is the caller's own memory, and a reply that
    /// leaves it untouched hands back whatever the caller had there before.
    /// Every struct the services below would fill describes something this
    /// console does not have — a local network, a peer group, a save
    /// transfer — and an all-zero one is what "none of that" looks like in
    /// each of them.
    pub(super) fn zero_output_buffer(&mut self, tls: u32, index: u32) {
        let Some((addr, size)) = self.ipc_output_buffer(tls, index) else {
            return;
        };
        for offset in 0..size {
            let _ = self.mem.write_u8(addr.wrapping_add(offset), 0);
        }
    }

    /// Write `bytes` into the request's `index`-th output buffer, truncated to
    /// what the caller left room for, and report how many bytes were written.
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

    // Writing the reply. `write_ipc_response` is what a service calls; the
    // two below it are the CMIF and TIPC forms it picks between, and getting
    // that choice wrong is invisible until the caller validates the header.
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
        self.write_ipc_reply(tls, result, &[], move_handles, raw_data, domain_objects)
    }

    /// The full form: a reply may carry **copy** handles as well as move ones,
    /// and the difference is not cosmetic. A move handle transfers ownership
    /// (a sub-session from `reply_with_interface`); a copy handle duplicates
    /// one the server keeps (every event a service hands out). They live in
    /// different fields of the handle descriptor and in that order in the
    /// reply, so a copy handle sent in the move slot is read back as **0** —
    /// which is exactly why `nnSdk` spent the whole boot waiting on handle 0
    /// after asking for `GetGpuErrorDetectedSystemEvent`.
    pub(super) fn write_ipc_reply(
        &mut self,
        tls: u32,
        result: u32,
        copy_handles: &[u64],
        move_handles: &[u64],
        raw_data: &[u8],
        domain_objects: &[u32],
    ) -> Result<()> {
        // Every reply that carries an error, named. A refused command prints
        // itself, but a command that is *answered* with a failure does not,
        // and that is the shape an initialisation step that quietly gives up
        // takes: the caller reads the Result, stops, and asks for nothing more.
        if result != 0 && std::env::var("TRACE_IPC").is_ok() {
            let module = result & 0x1FF;
            let description = (result >> 9) & 0x1FFF;
            eprintln!(
                "[ipc] error {result:#x} (module {module}, description {description}) from {:?} cmd={:?}",
                self.service_name(self.read_zr(0)),
                self.ipc_command_id(tls)
            );
        }
        if self.ipc_is_tipc_request(tls) {
            return self.write_tipc_reply(tls, result, copy_handles, move_handles, raw_data);
        }
        let is_domain = self.ipc_is_domain_request(tls);
        // A reply's `type` field (bits[15:0] of word 0) is 0: the counts in the
        // rest of the word are what matter. libnx ignores the field entirely,
        // but libtransistor validates it (`type != 0 && type != 4` → its error
        // 0x7E0DD), which is what made sdl-hello's "Failed to open connection
        // to fsp-srv" — a 0x40 here fails that check on every single reply.
        self.mem.write_u32(tls, 0)?;
        let has_handles = !copy_handles.is_empty() || !move_handles.is_empty();
        // { send_pid:1, num_copy:4, num_move:4 }
        let handle_desc = ((copy_handles.len() as u32) << 1) | ((move_handles.len() as u32) << 5);
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
            // Copy handles come first, then move handles.
            for &h in copy_handles.iter().chain(move_handles) {
                self.mem.write_u32(tls.wrapping_add(off), h as u32)?;
                off += 4;
            }
        }
        // Align to 16 bytes.
        let pre = (16 - (off % 16)) % 16;
        off += pre;
        // Clear the section this reply is about to declare, before filling it.
        //
        // A reply is written *over* the request, in the same TLS buffer, and
        // whatever it does not write stays as the request's bytes. The padding
        // the header counts is four words wide, which is room for a small out
        // parameter — so a command answered with an empty success never handed
        // the caller nothing. It handed the caller stale TLS, in a reply whose
        // declared size passed every length check `nnSdk` and libnx make.
        //
        // That is how `ListDisplayModes` cost the Home Menu a billion
        // instructions: it read its mode count out of the previous reply's
        // leftovers and walked a buffer nothing had written. Zeroed, an
        // unimplemented command's out parameters read as 0 — still wrong, but
        // the same wrong every time and survivable, which is the difference
        // between a bug that can be found and one that cannot.
        for i in 0..raw_section_words * 4 {
            self.mem.write_u8(tls.wrapping_add(off + i), 0)?;
        }
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

    /// A **TIPC** reply: the hipc header, the handles, and then the data
    /// words — which start with the `Result` itself rather than with an SFCO
    /// header, and are not aligned to 16 bytes the way a CMIF reply's are.
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
        // Same reason as the CMIF path: clear before filling, so the tail of
        // a partly-filled last word is a zero rather than a request byte.
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

    // Handles, sessions and domains: who this request is for, and what the
    // object it names was last said to be.
    pub(super) fn alloc_handle(&mut self) -> u64 {
        let h = self.next_handle as u64;
        self.next_handle = self.next_handle.wrapping_add(1);
        h
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
        self.erpt_readers.retain(|&key, _| key & !0xFFFF_FFFF != session);
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

    /// Which interface a request is addressed to: the sub-interface its
    /// domain object was filed under, or the name its session handle was
    /// recorded with.
    ///
    /// A service that hands out sub-interfaces is reached by two routes — an
    /// object id inside a domain, or a session handle of its own — and both
    /// arrive at the same handler, so both have to resolve to the same name.
    /// `root` is the answer for a session nothing has named, which is the
    /// service itself.
    pub(super) fn ipc_interface(&self, tls: u32, handle: u64, root: &'static str) -> String {
        if self.ipc_is_domain_request(tls) {
            let object_id = self.ipc_domain_object_id(tls);
            self.domain_interface(handle, object_id).unwrap_or(root).to_owned()
        } else {
            self.service_name(handle).unwrap_or(root).to_owned()
        }
    }

    /// Key for the per-object state maps (`fs_files`, `fs_dirs`). A domain
    /// object id is only unique within its session, and a plain sub-session is
    /// identified by its own handle, so both go in as `handle:object_id`.
    pub(super) fn object_key(handle: u64, object_id: u32) -> u64 {
        (handle << 32) | u64::from(object_id)
    }

    /// The key *this* request's object files its state under — the same one
    /// [`Cpu::reply_with_interface`] returned when it handed the object out.
    /// A domain object is identified by its id within the session, a plain
    /// sub-session by its own handle.
    pub(super) fn ipc_object_key(&self, tls: u32, handle: u64) -> u64 {
        if self.ipc_is_domain_request(tls) {
            Self::object_key(handle, self.ipc_domain_object_id(tls))
        } else {
            Self::object_key(handle, 0)
        }
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
        if self.ipc_is_tipc_request(tls) {
            return false;
        }
        let start = self.ipc_reply_start(tls);
        self.mem.read_u8(tls.wrapping_add(start)).unwrap_or(0) == 1
    }

    /// Whether the request is a domain *close* (`CmifDomainRequestType_Close`,
    /// the domain header's type byte set to 2): it drops one object out of the
    /// session and carries no `CmifInHeader` at all. [`Cpu::ipc_command_id`]
    /// falls back to scanning the whole message buffer for an `SFCI` magic, so
    /// on a close it finds the *previous* request's header still sitting there
    /// and reports that command id — which is why `appletExit`'s teardown used
    /// to look like a flurry of command 0s.
    pub(super) fn ipc_is_domain_close(&self, tls: u32) -> bool {
        if self.ipc_is_tipc_request(tls) {
            return false;
        }
        self.mem.read_u8(tls.wrapping_add(self.ipc_reply_start(tls))).unwrap_or(0) == 2
    }

    /// Forget one object and acknowledge the close.
    pub(super) fn close_domain_object(&mut self, tls: u32, handle: u64, object_id: u32) -> Result<()> {
        // `ssl` counts its live contexts, and this is where one stops being
        // live. The count lives here rather than in `ssl_request` because a
        // close never reaches a service handler any more — see the dispatch in
        // `horizon_syscall`.
        if self.domain_interface(handle, object_id) == Some("ssl:context") {
            self.ssl_contexts = self.ssl_contexts.saturating_sub(1);
        }
        self.domain_objects.remove(&(handle, object_id));
        self.write_ipc_response(tls, 0, &[], &[], &[])
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

    /// Answer the session-management commands every service session has, if
    /// this request is one of them, and report whether it was.
    ///
    /// `ConvertToDomain` files the session's own interface under a fresh
    /// object id — the name given here is what later requests on that object
    /// dispatch on — and `QueryPointerBufferSize` answers 0, which is what
    /// keeps a caller off the pointer-buffer path this IPC layer does not
    /// read. Every service below opens with this, because a control message
    /// is not a command on the interface at all and answering it as one is
    /// how `appletOE`'s first message was once read as command 3.
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
            _ => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])?,
        }
        Ok(true)
    }

    /// An event a service object hands out, allocated on first ask and kept.
    ///
    /// A caller that asks for the same event twice has to be given the *same*
    /// object: handed a second copy, it waits on a handle the service would
    /// not signal even if it signalled the first. `am`'s applet events have
    /// their own table for exactly this reason
    /// ([`Cpu::library_applet_event`]); everything else shares this one,
    /// keyed by what the event is for and which object handed it out.
    ///
    /// Almost nothing here ever signals one. Each describes something that
    /// does not happen on this console — a Bluetooth radio turning on, a save
    /// transfer finishing, a news article arriving — so a caller waiting on it
    /// is waiting for something that genuinely never comes, which is the
    /// truthful state rather than the silent one. The exception is `erpt`'s
    /// report-created event, which fires because a report really is filed.
    pub(super) fn kept_event(&mut self, purpose: &'static str, object: u64) -> u64 {
        if let Some(&event) = self.service_events.get(&(purpose, object)) {
            return event;
        }
        let event = self.alloc_event(purpose, false);
        self.service_events.insert((purpose, object), event);
        event
    }

    // What a service answers when there is nothing behind it.
    /// Answers a command a service does not actually implement.
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
    pub(super) fn unimplemented_command(
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
            // The request's shape, which is most of its signature: how many
            // argument words it carries, and whether it left a buffer for the
            // reply to fill. Answering a command that wants an out-object or
            // an out-buffer with a bare success is worse than refusing it —
            // the caller reads a zero and fails somewhere else entirely — so
            // this is what says which kind it is.
            let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
            let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
            let statics = (hdr1 >> 16) & 0xf;
            let send = (hdr1 >> 20) & 0xf;
            let recv = (hdr1 >> 24) & 0xf;
            let recv_static = matches!((hdr2 >> 10) & 0xf, 2..) as u32;
            let words = hdr2 & 0x3ff;
            self.diagnostic(&format!(
                "[ipc] unimplemented: {iface} cmd={cmd_id:?} (pc={pc:#x}, {words} data words, \
                 buffers: {statics} static/{send} send/{recv} recv/{recv_static} recv-static)"
            ));
        }
        self.write_ipc_response(tls, UNKNOWN_COMMAND_ID, &[], &[], &[])
    }

    /// Answer a command nothing implements with a fabricated success, in a
    /// shape whose out-object the caller can actually use.
    ///
    /// This reply used to carry the fabricated object id in the raw data and
    /// nothing else. On a plain session that is not where an out-object
    /// lives: `nnSdk` reads one as a **move handle**, and a reply carrying no
    /// handle is not an error to it — the handle parses as 0, the client
    /// quietly skips constructing the proxy, and the command still returns
    /// **success**. The caller then makes its first virtual call through a
    /// null `SharedPointer`. That is how boot2 reached `pc=0` one instruction
    /// after `gpio`'s `OpenSession2` was answered "successfully", with
    /// nothing in between to say which command had lied.
    ///
    /// So the reply now carries a real sub-session — or a real domain
    /// out-object, when the session is a domain — *as well as* the raw object
    /// id, which is what a caller reading a plain out value has always read
    /// here, `ConvertToDomain` most of all.
    ///
    /// The reply also carries an **event**, in the copy-handle slot, for the
    /// same reason it carries a sub-session in the move slot: an out-object
    /// and an out-event are the two things a command can hand back that a
    /// caller cannot invent for itself, and nothing here knows which of them
    /// an unimplemented command was supposed to return. Filling both costs one
    /// handle and removes the case where a caller waits forever on handle 0 —
    /// which is not a hypothetical: it is where the Home Menu's message thread
    /// stopped, three created-but-never-started threads behind it, and there
    /// was nothing in any trace to say which command had failed to hand it an
    /// event. The event is never signalled, so a caller that waits on it is
    /// waiting for something that genuinely never happens, rather than acting
    /// on one that never will.
    ///
    /// All three are allocated once per `(session, command)` and reused: a
    /// guest polling a command nothing implements would otherwise be handed a
    /// fresh handle on every single call.
    pub(super) fn reply_with_fabricated_object(
        &mut self,
        tls: u32,
        handle: u64,
        name: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
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
            self.write_ipc_reply(tls, 0, &[event], &[], &object_id.to_le_bytes(), &[object_id])
        } else {
            self.record_handle(sub, name);
            self.write_ipc_reply(tls, 0, &[event], &[sub], &object_id.to_le_bytes(), &[])
        }
    }

    /// Note that a service reached over IPC has no implementation behind it at
    /// all, and is about to be answered with a fabricated object id.
    ///
    /// Unlike [`Cpu::unimplemented_command`] this does not change the reply — the
    /// generic success is load-bearing for homebrew that only checks the
    /// Result — it just stops the gap being invisible. Whatever this prints is
    /// the list of services a guest is asking for and not getting.
    pub(super) fn warn_no_implementation(&mut self, service: &str, cmd_id: Option<u32>) {
        if self.unimplemented_ipc.insert((service.to_string(), cmd_id)) {
            self.diagnostic(&format!("[ipc] no implementation: {service} cmd={cmd_id:?}"));
        }
    }

    // The services that live here: `sm:`, which hands out every other
    // session, and the ones whose whole implementation is an answer or two.
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

    /// `csrng` (`IRandomInterface`): the console's random number generator.
    ///
    /// Real hardware answers this out of the security processor's hardware
    /// RNG. There is none here, and `wasm32-unknown-unknown` has no OS entropy
    /// to borrow either, so what a caller gets is **pseudo**-random: splitmix64
    /// over a state seeded from the emulated clock. That distinction is real —
    /// nothing that comes out of here should be used as a key — but it is a
    /// far better answer than the generic fallback's, which left the caller's
    /// buffer untouched: a "random" number that is whatever was on the stack
    /// is both non-random *and* undetectably so.
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

    /// `spl:` (`IGeneralInterface`): the liaison to the security processor.
    ///
    /// Everything it exists for — key derivation, AES with device-unique keys,
    /// unwrapping title keys in TrustZone — is out of reach here, and the one
    /// command a guest actually asks this emulator for is `GetConfig`, which
    /// reports what kind of console it is running on. That much this can
    /// answer truthfully: an original (Icosa) retail unit, not in debug mode.
    /// The device id is a fixed placeholder rather than a real fused id.
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
                    // DramId: the 4 GiB Samsung part an original unit shipped
                    // with. `MAX_MAPPED_BYTES` is the real memory limit here;
                    // this only names the part.
                    1 => 0,
                    // HardwareType: Icosa, the original console.
                    4 => 0,
                    // HardwareState: Production, not a development unit.
                    5 => 1,
                    // IsRecoveryBoot: this booted normally.
                    6 => 0,
                    // DeviceId: a real console's is fused in and unique. This
                    // one is fixed, and nothing derives a key from it.
                    7 => SPL_DEVICE_ID,
                    // MemoryArrange: the standard 4 GiB layout.
                    9 => 0,
                    // IsDebugMode: no.
                    10 => 0,
                    // Everything else — Version, BootReason, kernel
                    // configuration, quest state, regulator and key
                    // generation — reads as zero, which is the "nothing
                    // unusual" answer for each of them.
                    //
                    // That deliberately includes Atmosphère's own extensions
                    // at 65000 and up, which are what a real guest asks this
                    // service for first: NX-Fetch wants the CFW's API version
                    // (65000) and emummc type (65007). Zero there reads as "no
                    // custom firmware, booted from internal storage", and this
                    // emulator is indeed not Atmosphère — answering with a
                    // version would be claiming a CFW whose behaviour nothing
                    // here implements.
                    _ => 0,
                };
                self.write_ipc_response(tls, 0, &[], &value.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "spl:", cmd_id),
        }
    }

    /// `pm:*`: the process manager. `pm:shell` starts and stops processes,
    /// `pm:dmnt` finds them, `pm:info` maps one to its program, and `pm:bm`
    /// reports how the console booted.
    ///
    /// There is exactly one process here and nothing can create another —
    /// `LaunchProgram` has nothing to launch and no second address space to
    /// launch it into — so what these can answer honestly is *identity*: which
    /// process is the application (this one), and which program it is running.
    /// The process id agrees with `svcGetProcessId`'s, which is the same
    /// question asked through the kernel instead.
    pub(super) fn pm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]);
        }
        let iface = self.service_name(handle).unwrap_or("pm:shell").to_string();
        match iface.as_str() {
            // IDebugMonitorInterface.
            "pm:dmnt" => match cmd_id {
                // GetJitDebugProcessIdList -> s32 count: nothing is being
                // JIT-debugged.
                Some(0) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                // GetProcessId(u64 program_id) / GetApplicationProcessId ->
                // u64 pid. Either way it is this process.
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
            // IBootModeInterface::GetBootMode -> u32: Normal. The maintenance
            // mode this could otherwise report is a state the console is put
            // in deliberately, and nothing here does.
            "pm:bm" => match cmd_id {
                Some(0) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IShellInterface. NotifyBootFinished is an announcement, and
            // GetApplicationProcessIdForShell asks the same identity question
            // `pm:dmnt` does.
            _ => match cmd_id {
                Some(7) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                Some(8) => self.write_ipc_response(tls, 0, &[], &PROCESS_ID.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
        }
    }

    /// `btm:sys` — "nn::btm::IBtmSystem", and the `IBtmSystemCore` it hands
    /// out: the Bluetooth radio and the controller-pairing flow the Home
    /// Menu's "Change Grip/Order" screen drives.
    ///
    /// There is no Bluetooth radio here and no controller to pair over it —
    /// input arrives through `hid`'s shared memory from the host's Gamepad
    /// API, which is not a pairing at all. So the radio can be turned on and
    /// off (it is a setting, and the menu reads it back), nothing is ever
    /// paired, and the two events that would report a change are handed out
    /// and never signalled.
    ///
    /// `GetCore` is the reason this needs an implementation rather than the
    /// fallback: it is the *first* command, every other one goes through the
    /// object it returns, and the generic reply's fabricated object id is not
    /// an `IBtmSystemCore` a caller can call.
    pub(super) fn btm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "btm:sys", cmd_id)? {
            return Ok(());
        }
        let iface = self.ipc_interface(tls, handle, "btm:sys");
        match iface.as_str() {
            "btm:core" => match cmd_id {
                // StartGamepadPairing / CancelGamepadPairing. Pairing runs
                // until something pairs or the caller stops it; nothing will.
                Some(0) | Some(1) => {
                    self.bt_gamepad_pairing = cmd_id == Some(0);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ClearGamepadPairingDatabase: already empty.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetPairedGamepadCount -> u8.
                Some(3) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
                // EnableRadio / DisableRadio / IsRadioEnabled.
                Some(4) | Some(5) => {
                    self.bt_radio_enabled = cmd_id == Some(4);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(6) => {
                    let enabled = u8::from(self.bt_radio_enabled);
                    self.write_ipc_response(tls, 0, &[], &[enabled], &[])
                }
                // AcquireRadioEvent / AcquireGamepadPairingEvent -> a bool
                // saying whether the event was there to take, and the event
                // itself. The bool is not the radio's state: it is whether
                // this caller got the one event the service has.
                Some(7) | Some(8) => {
                    let purpose =
                        if cmd_id == Some(7) { "btm:radio" } else { "btm:gamepad-pairing" };
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

    /// `nfc:sys` — "nn::nfc::detail::ISystemManager", and the `ISystem` it
    /// hands out.
    ///
    /// The NFC reader lives in the right Joy-Con, and nothing here emulates
    /// one, so the device list is empty and every command that names a device
    /// has no device to name. That is a state a real console reaches too,
    /// with the controller detached — it is not a broken console, it is one
    /// with nothing to scan.
    ///
    /// Whether NFC is *enabled* is a different question from whether a reader
    /// is attached, and it is a setting rather than a fact: the system
    /// settings applet writes it and reads it straight back, so `SetNfcEnabled`
    /// stores what it was told and `IsNfcEnabled` answers with that.
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
                // Initialize / Finalize, and the 4.0.0+ InitializeSystem /
                // FinalizeSystem that replaced them. Both pairs drive the same
                // state, which is the one GetState reports.
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
                    let enabled = u8::from(self.nfc_enabled);
                    self.write_ipc_response(tls, 0, &[], &[enabled], &[])
                }
                // SetNfcEnabledOld / SetNfcEnabled(bool).
                Some(100) | Some(500) => {
                    self.nfc_enabled = self.ipc_arg_u8(tls, 0) != 0;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // ListDevices: the device handles go into an output buffer,
                // and the reply says how many were written. None were.
                Some(404) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
                // AttachAvailabilityChangeEvent -> the event that fires when a
                // reader is attached or detached. None ever is.
                Some(407) => {
                    let event = self.kept_event("nfc:availability", handle);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // Everything past here — GetDeviceState, StartDetection,
                // GetTagInfo, the Mifare pass-through — names a device out of
                // the list ListDevices reports as empty, so a caller can only
                // reach it with a handle this service never handed out.
                // Refusing says so; answering would invent a reader.
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
}

/// Request builders shared by every service module's tests, and the
/// marshalling tests that exercise them directly.
///
/// A service test is mostly "marshal this command into the TLS buffer, run it,
/// read the reply back", and the marshalling is the part none of them should
/// be spelling out for themselves.
#[cfg(test)]
pub(super) mod testing {
    use super::Cpu;

    pub(crate) const TLS: u32 = 0x2000;
    pub(crate) const SFCI: u32 = 0x4943_4653;

    /// A CMIF request in the TLS buffer with no buffer descriptors:
    /// `CmifDomainInHeader` (when `domain`) then `CmifInHeader` then payload.
    pub(crate) fn request(domain: bool, command_id: u32, payload: &[u8]) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        marshal(&mut cpu, domain, command_id, payload);
        cpu
    }

    /// Marshal a request into a session's TLS buffer. A reply is written over
    /// the request it answered, so a second command on the same session has to
    /// be marshalled again rather than patched.
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
    fn a_static_buffer_is_found_past_the_handles_a_special_header_carries() {
        // `ipc_descriptor_start` skips a special header's copy and move
        // handles; `ipc_static_buffers` kept an older copy of that walk which
        // skipped only the pid, so a request carrying both a handle and a path
        // read the path out of the handle words. Nothing in the emulator's
        // path had sent that combination — which is why it went unnoticed, not
        // why it was safe.
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
            cpu.mem.write_u32(TLS + 12 + slot * 4, 0xDEAD_0000 + slot).unwrap();
        }
        // `{ index:6, address_high:6, address_mid:4, size:16 }`, then the
        // low word of the address.
        let descriptor = TLS + 12 + 3 * 4;
        cpu.mem.write_u32(descriptor, 0x10 << 16).unwrap();
        cpu.mem.write_u32(descriptor + 4, PATH).unwrap();

        assert_eq!(cpu.ipc_static_buffers(TLS), vec![(PATH, 0x10)]);
        assert_eq!(cpu.ipc_request_path(TLS), "/save/data.bin");
    }


    /// A CMIF request whose first send-static ("pointer") buffer carries a
    /// path, the way every `IFileSystem` command names the file it acts on.
    pub(crate) fn request_with_path(command_id: u32, path: &str, payload: &[u8]) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(PATH_AT, 0x400).unwrap();
        write_path_request(&mut cpu, command_id, path, payload);
        cpu
    }

    /// Where [`write_path_request`] parks the path it hands the service.
    const PATH_AT: u32 = 0x3800;

    /// The same, into an existing `Cpu`, so a test can drive a second command
    /// against the tree the first one left behind.
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
        cpu.mem.write_u32(TLS + 8, (path.len() as u32 + 1) << 16).unwrap();
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
        cpu.fs_files.insert(Cpu::object_key(9, 1), "/a.txt".to_owned());
        cpu.record_handle(10, "vi:m");
        cpu.forget_handle(9);
        assert!(cpu.service_name(9).is_none());
        assert!(cpu.domain_interface(9, 1).is_none());
        assert!(cpu.fs_files.is_empty());
        assert_eq!(cpu.service_name(10), Some("vi:m"));
    }

    /// Overwrite the TLS buffer with a fresh request, so a test can drive a
    /// second command against the state the first one left behind. The buffer
    /// is cleared first: a reply leaves an `SFCO` header in it, and the
    /// command-id scan looks for a magic.
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

    /// A CMIF request carrying one map-alias **receive** buffer, the way
    /// `ListAllUsers` marshals the array the server fills.
    pub(crate) fn request_with_recv_buffer(command_id: u32, payload: &[u8], buffer: u32, size: u32) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        write_map_buffer_request(&mut cpu, command_id, payload, buffer, size, false);
        cpu
    }

    /// Write a request carrying one map-alias buffer, on the send side or the
    /// receive side, into an existing `Cpu`'s TLS.
    pub(crate) fn write_map_buffer_request(
        cpu: &mut Cpu,
        command_id: u32,
        payload: &[u8],
        buffer: u32,
        size: u32,
        send: bool,
    ) {
        for offset in (0..0x100u32).step_by(4) {
            cpu.mem.write_u32(TLS + offset, 0).unwrap();
        }
        let counts = if send { 1 << 20 } else { 1 << 24 };
        cpu.mem.write_u32(TLS, 4 | counts).unwrap();
        cpu.mem.write_u32(TLS + 4, 12).unwrap();
        // The descriptor: size, the low half of the address, then the packed
        // word holding the rest of it.
        cpu.mem.write_u32(TLS + 8, size).unwrap();
        cpu.mem.write_u32(TLS + 12, buffer).unwrap();
        cpu.mem.write_u32(TLS + 16, 0).unwrap();
        // The data area, aligned up from the 20 bytes of header + descriptor.
        let at = TLS + 0x20;
        cpu.mem.write_u32(at, SFCI).unwrap();
        cpu.mem.write_u32(at + 8, command_id).unwrap();
        for (index, &byte) in payload.iter().enumerate() {
            cpu.mem.write_u8(at + 0x10 + index as u32, byte).unwrap();
        }
    }

    /// A CMIF request offering one receive-static ("pointer") output buffer,
    /// the way `IProfile::Get` marshals its `AccountUserData`. Unlike every
    /// other descriptor, this one sits *after* the data words.
    pub(crate) fn request_with_recv_static(command_id: u32, payload: &[u8], buffer: u32, size: u32) -> Cpu {
        let mut cpu = request(false, command_id, payload);
        // Two words of padding aligning the CmifInHeader, the header, then the
        // payload — what the walk has to skip to reach the receive list.
        let data_words = 2 + 4 + payload.len().div_ceil(4) as u32;
        // recv_static_mode = 2 + one buffer.
        cpu.mem.write_u32(TLS + 4, data_words | (3 << 10)).unwrap();
        let at = TLS + 8 + 4 * data_words;
        cpu.mem.write_u32(at, buffer).unwrap();
        cpu.mem.write_u32(at + 4, size << 16).unwrap();
        cpu
    }


    /// Marshal a request carrying `buffers` map-alias **send** buffers into an
    /// existing session's TLS — the shape `erpt`'s context commands arrive in,
    /// which carry two and three of them.
    pub(crate) fn write_send_buffer_request(
        cpu: &mut Cpu,
        command_id: u32,
        payload: &[u8],
        buffers: &[(u32, u32)],
    ) {
        for offset in (0..0x200u32).step_by(4) {
            cpu.mem.write_u32(TLS + offset, 0).unwrap();
        }
        cpu.mem.write_u32(TLS, 4 | ((buffers.len() as u32) << 20)).unwrap();
        cpu.mem.write_u32(TLS + 4, 16).unwrap();
        for (index, &(address, size)) in buffers.iter().enumerate() {
            let at = TLS + 8 + 12 * index as u32;
            cpu.mem.write_u32(at, size).unwrap();
            cpu.mem.write_u32(at + 4, address).unwrap();
            cpu.mem.write_u32(at + 8, 0).unwrap();
        }
        let at = TLS + (8 + 12 * buffers.len() as u32).div_ceil(16) * 16;
        cpu.mem.write_u32(at, SFCI).unwrap();
        cpu.mem.write_u32(at + 8, command_id).unwrap();
        for (index, &byte) in payload.iter().enumerate() {
            cpu.mem.write_u8(at + 0x10 + index as u32, byte).unwrap();
        }
    }

    #[test]
    fn csrng_fills_the_buffer_with_bytes_that_differ() {
        // Not a CSPRNG — see `Cpu::next_random_u64` — but a caller asking for
        // random bytes has to get bytes, and different ones each call. The
        // generic reply left the buffer untouched, so a "random" value was
        // whatever the caller's stack already held.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(0, &[], BUFFER, 0x20);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.set_unix_time(1_700_000_000);
        cpu.csrng_request(TLS, Some(0)).unwrap();
        let first = cpu.read_bytes(BUFFER, 0x20);
        assert_ne!(first, vec![0u8; 0x20], "the buffer was written");
        assert!(first.windows(8).any(|w| w != &first[..8]), "not one value repeated");

        write_map_buffer_request(&mut cpu, 0, &[], BUFFER, 0x20, false);
        cpu.csrng_request(TLS, Some(0)).unwrap();
        assert_ne!(cpu.read_bytes(BUFFER, 0x20), first, "a second call differs");
    }

    #[test]
    fn spl_reports_a_retail_console() {
        // ConfigItem 4 is HardwareType (0 = Icosa, the original console) and 5
        // is HardwareState (1 = Production). Reporting a development unit
        // would send a guest down paths this emulator does not implement.
        for (item, expected) in [(4u32, 0u64), (5, 1), (10, 0)] {
            let mut cpu = request(false, 0, &item.to_le_bytes());
            cpu.spl_request(TLS, Some(0)).unwrap();
            assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), expected, "config item {item}");
        }
        // The device id is fixed, but it must not read as "no device".
        let mut cpu = request(false, 0, &7u32.to_le_bytes());
        cpu.spl_request(TLS, Some(0)).unwrap();
        assert_ne!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0);
    }

    /// A CMIF **control** request (message type 5) — the session-management
    /// commands `libnx` sends on a handle the moment `sm` hands it over,
    /// before any command of the service's own.
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
        // pm:dmnt's GetApplicationProcessId and svcGetProcessId answer the
        // same question through different doors; two answers would be one too
        // many.
        let mut cpu = request(false, 4, &[]);
        cpu.register_service_handle(9, "pm:dmnt");
        cpu.pm_request(TLS, 9, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), super::PROCESS_ID);

        // pm:info maps it to the program it is running: the Album applet's id
        // for homebrew, or whatever a loader set.
        let mut cpu = request(false, 0, &super::PROCESS_ID.to_le_bytes());
        cpu.register_service_handle(9, "pm:info");
        cpu.set_program_id(0x0100_4890_117B_2000);
        cpu.pm_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0x0100_4890_117B_2000);
    }

}

