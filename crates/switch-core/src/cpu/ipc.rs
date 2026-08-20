//! Horizon IPC: parsing CMIF/HIPC requests out of the TLS message buffer,
//! synthesizing replies, and the service implementations behind the
//! session handles homebrew opens (`sm:`, `fsp-srv`, `vi:m`, `nvdrv`).

use super::{AudioOut, AudrenParams, BsdSocket, Cpu};
use crate::Result;
use std::collections::VecDeque;

/// The uid of the console's one user account.
///
/// Any 128-bit value does as long as it is **not zero**: zero is what
/// `AccountUid` means by "no user", and a title handed it back from
/// `GetLastOpenedUser` concludes nobody is signed in. Spelling it in ASCII
/// makes it recognisable in a trace, and it is exactly the 16 bytes a uid is.
const ACCOUNT_UID: [u8; 16] = *b"switch-wasm user";
/// `nn::account::ProfileBase`: uid, last-edit timestamp, then the nickname.
const PROFILE_BASE_LEN: usize = 0x38;
/// `nn::account::UserData`, the block `IProfile::Get` fills in beside the base
/// (icon id, background colour, mii id).
const ACCOUNT_USER_DATA_LEN: usize = 0x80;
/// acc's "that user does not exist" (module 124, description 100).
///
/// Only a caller that invented a uid can reach this: the only uid this service
/// ever hands out is [`ACCOUNT_UID`], so anything else was not obtained from
/// here.
const ACCOUNT_USER_NOT_EXIST: u32 = 124 | (100 << 9);
/// The `NetworkServiceAccountId` `IManagerForApplication::GetAccountId`
/// reports. Nonzero, since zero is that field's "no account" sentinel; the
/// value itself is arbitrary and nothing derives anything from it.
const NETWORK_SERVICE_ACCOUNT_ID: u64 = 0x0000_0001_0000_0001;
/// `nn::account::Nickname`: a fixed NUL-terminated field inside `ProfileBase`.
pub(super) const NICKNAME_LEN: usize = 0x20;
/// The nickname the console's user has until a host or the guest changes it.
pub(super) const DEFAULT_NICKNAME: &str = "Player";

/// The system version `set:sys` reports, as major/minor/micro.
///
/// libnx seeds `hosversionGet` from this and branches on it everywhere, so the
/// number is load-bearing rather than decorative: it is picked to sit past the
/// gates the services here implement (6.0.0 for `acc`'s qualified-user list)
/// and before the ones they do not (17.0.0, where `ts` moves its measurement
/// onto a different interface).
const FIRMWARE_VERSION: (u8, u8, u8) = (12, 1, 0);

/// The temperatures `ts` reports, in degrees Celsius: the SoC
/// (`TsLocation_Internal`) first, the PCB (`TsLocation_External`) second.
///
/// Fixed, and deliberately an idle console's: nothing this emulator runs makes
/// silicon warm, so an idle reading is the true state rather than a
/// placeholder for one that could not be taken.
const TS_TEMPERATURE_C: [i32; 2] = [40, 35];
/// The range `ts` says its sensors report over. Both readings above sit inside
/// it — a caller that scales a gauge by this range would otherwise draw the
/// needle off the end.
const TS_TEMPERATURE_RANGE_C: (i32, i32) = (0, 100);

/// The address `nifm` reports for the console's wired link, and the one
/// `bsd` reports for a socket that was never bound.
const NIFM_LOCAL_IP: [u8; 4] = [192, 168, 1, 100];

/// `bsd` errnos, in **FreeBSD's** numbering — which is what the real service
/// returns, and so what guest code is written against (`EAGAIN` is 35 here,
/// not the 11 a Linux-hosted build would use).
const BSD_EBADF: i32 = 9;
const BSD_EINVAL: i32 = 22;
const BSD_EAGAIN: i32 = 35;
const BSD_ENETUNREACH: i32 = 51;
const BSD_ENOTCONN: i32 = 57;
const BSD_ECONNREFUSED: i32 = 61;
/// `SOCK_DGRAM`. The only place a socket's type changes the answer is the data
/// path, where a datagram socket has nowhere to send *to* (`ENETUNREACH`) and
/// a stream socket has no connection to send *on* (`ENOTCONN`).
const BSD_SOCK_DGRAM: u32 = 2;
/// `FIONBIO`, `F_GETFL`/`F_SETFL`, and FreeBSD's `O_NONBLOCK` — the last only
/// so that the `ioctl` route sets the same bit the `fcntl` route reads back.
const BSD_FIONBIO: u32 = 0x8004_667E;
const BSD_F_GETFL: u32 = 3;
const BSD_F_SETFL: u32 = 4;
const BSD_O_NONBLOCK: u32 = 0x0004;

/// `ApmPerformanceMode_Normal`: the handheld clock profile, and the mode
/// `am`'s `ICommonStateGetter::GetPerformanceMode` already reports. Boost (1)
/// is the docked one.
const APM_PERFORMANCE_MODE_NORMAL: u32 = 0;
/// The `ApmPerformanceConfiguration` each performance mode starts at, indexed
/// by mode.
///
/// Nothing here derives a clock from these — no CPU, GPU or memory frequency
/// in this emulator is settable — but they cannot be zero, which is
/// `ApmPerformanceConfiguration_Invalid`, and whatever a title sets has to
/// read back unchanged.
pub(super) const APM_DEFAULT_CONFIGURATION: [u32; 2] = [0x0001_0000, 0x0002_0000];

/// Real profile icons are 256x256.
const PROFILE_IMAGE_SIZE: u16 = 256;
/// The icon's colour, a neutral slate. Nothing derives anything from it.
const PROFILE_IMAGE_COLOR: (u8, u8, u8) = (0x4B, 0x50, 0x5A);

/// JPEG markers, for the profile icon [`solid_jpeg`] encodes.
const JPEG_SOI: u8 = 0xD8;
const JPEG_APP0: u8 = 0xE0;
const JPEG_DQT: u8 = 0xDB;
const JPEG_SOF0: u8 = 0xC0;
const JPEG_DHT: u8 = 0xC4;
const JPEG_SOS: u8 = 0xDA;
const JPEG_EOI: u8 = 0xD9;
/// Every entry of the quantization table. 8 is what makes a constant block's
/// DC coefficient (`8x`) quantize to exactly `x`.
const JPEG_QUANT: u8 = 8;
/// The AC symbol for "end of block": the rest of this block is zeros.
const JPEG_EOB: u8 = 0x00;
/// The DC Huffman table: the twelve magnitude categories, four coded in three
/// bits and eight in four, which is a complete code.
const JPEG_DC_BITS: [u8; 16] = [0, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const JPEG_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// The AC Huffman table: end-of-block and run-of-sixteen-zeros, one bit each.
/// A constant image only ever emits the first, but a two-symbol table is a
/// complete code where a one-symbol table would not be.
const JPEG_AC_BITS: [u8; 16] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const JPEG_AC_VALUES: [u8; 2] = [JPEG_EOB, 0xF0];

/// `nn::audio::PcmFormat`: 16-bit signed samples, the only format `audout`
/// takes here and the one every caller asks for.
const PCM_FORMAT_INT16: u32 = 2;
/// `nn::audio::AudioOutState`, as `IAudioOut` reports it.
const AUDIO_OUT_STARTED: u32 = 0;
const AUDIO_OUT_STOPPED: u32 = 1;

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

    /// The `index`-th map-alias **receive** buffer as `(address, size)`. Same
    /// walk as [`Cpu::ipc_recv_buffer_addr`], keeping the size a caller needs
    /// when the reply's length is whatever fits (`GetReleasedAudioOutBuffer`
    /// hands back as many tags as the guest left room for).
    pub(super) fn ipc_recv_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        let num_recv_buffers = (hdr1 >> 24) & 0xf;
        if index >= num_recv_buffers {
            return None;
        }
        let mut off = 8u32;
        if (hdr2 >> 31) & 1 != 0 {
            off += 4;
            if self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0) & 1 != 0 {
                off += 8; // pid
            }
        }
        off += 8 * num_send_statics;
        off += 12 * (num_send_buffers + index);
        let size = self.mem.read_u32(tls.wrapping_add(off)).ok()?;
        let addr = self.ipc_recv_buffer_addr(tls, index)?;
        Some((addr, size))
    }

    /// The `index`-th map-alias **send** buffer of a hipc request, as
    /// `(address, size)`. These sit before the receive buffers in the
    /// descriptor area, so the walk is the same as
    /// [`Cpu::ipc_recv_buffer_addr`]'s with a different offset.
    ///
    /// Only the low 32 bits of each field are read: the guest address space is
    /// 32-bit here, so a descriptor's `address_high`/`size_high` bits are
    /// always zero.
    pub(super) fn ipc_send_buffer(&self, tls: u32, index: u32) -> Option<(u32, u32)> {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_send_buffers = (hdr1 >> 20) & 0xf;
        if index >= num_send_buffers {
            return None;
        }
        let mut off = 8u32;
        if (hdr2 >> 31) & 1 != 0 {
            off += 4;
            if self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0) & 1 != 0 {
                off += 8; // pid
            }
        }
        off += 8 * num_send_statics;
        off += 12 * index;
        let size = self.mem.read_u32(tls.wrapping_add(off)).ok()?;
        let addr = self.mem.read_u32(tls.wrapping_add(off + 4)).ok()?;
        Some((addr, size))
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

    /// The receive-static ("pointer") buffers a request offers for **output**,
    /// as `(address, size)`.
    ///
    /// These are the only descriptors that sit *after* the raw data rather
    /// than before it, at the unaligned data offset plus `num_data_words` —
    /// which counts the padding that aligns the CMIF header, so the walk lands
    /// past the payload either way. How many there are is encoded, not
    /// counted: `recv_static_mode` (hdr2 bits 10..14) is 0 for none, 2 for a
    /// single buffer the server sizes, and `2 + count` otherwise.
    ///
    /// `IProfile::Get` is why this exists — its `AccountUserData` comes back
    /// through a fixed-size pointer buffer rather than a map-alias one.
    pub(super) fn ipc_recv_static_buffers(&self, tls: u32) -> Vec<(u32, u32)> {
        let hdr1 = self.mem.read_u32(tls).unwrap_or(0);
        let hdr2 = self.mem.read_u32(tls.wrapping_add(4)).unwrap_or(0);
        let num_send_statics = (hdr1 >> 16) & 0xf;
        let num_buffers = ((hdr1 >> 20) & 0xf) + ((hdr1 >> 24) & 0xf) + ((hdr1 >> 28) & 0xf);
        let num_data_words = hdr2 & 0x3ff;
        let count = match (hdr2 >> 10) & 0xf {
            0 | 1 => 0,
            2 => 1,
            mode => mode - 2,
        };
        let mut off = 8u32;
        if (hdr2 >> 31) & 1 != 0 {
            off += 4;
            if self.mem.read_u32(tls.wrapping_add(8)).unwrap_or(0) & 1 != 0 {
                off += 8; // pid
            }
        }
        off += 8 * num_send_statics + 12 * num_buffers + 4 * num_data_words;
        (0..count)
            .map(|index| {
                let at = off + 8 * index;
                let address = self.mem.read_u32(tls.wrapping_add(at)).unwrap_or(0);
                let packed = self.mem.read_u32(tls.wrapping_add(at + 4)).unwrap_or(0);
                (address, packed >> 16)
            })
            .collect()
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

    /// Whether the request is a domain *close* (`CmifDomainRequestType_Close`,
    /// the domain header's type byte set to 2): it drops one object out of the
    /// session and carries no `CmifInHeader` at all. [`Cpu::ipc_command_id`]
    /// falls back to scanning the whole message buffer for an `SFCI` magic, so
    /// on a close it finds the *previous* request's header still sitting there
    /// and reports that command id — which is why `appletExit`'s teardown used
    /// to look like a flurry of command 0s.
    pub(super) fn ipc_is_domain_close(&self, tls: u32) -> bool {
        self.mem.read_u8(tls.wrapping_add(self.ipc_reply_start(tls))).unwrap_or(0) == 2
    }

    /// Forget one object and acknowledge the close.
    pub(super) fn close_domain_object(&mut self, tls: u32, handle: u64, object_id: u32) -> Result<()> {
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
                // `IStorage::Read(s64 offset, u64 size)`. Note the layout is
                // **not** `IFile::Read`'s: a file read leads with a `u32
                // option` and pads to 8, putting its offset at +8 and its size
                // at +0x10. This used to read those two fields, so every
                // storage read came back as "0 bytes at offset 0x50" — the
                // guest mounted its RomFS, parsed an empty header, and
                // `nn::fs::OpenDirectory("rom:/Data")` found nothing.
                let offset = self.mem.read_u64(data)? as usize;
                let requested = self.mem.read_u64(data.wrapping_add(8))? as usize;
                if std::env::var("TRACE_IPC").is_ok() {
                    eprintln!(
                        "[storage] read offset={offset:#x} size={requested:#x} of {:#x}",
                        romfs.len()
                    );
                }
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
        if self.ipc_is_control_request(tls) {
            // `GetFirmwareVersion` returns its struct through a receive-static
            // ("pointer") buffer, so this has to claim room for one.
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        match cmd_id {
            // GetFirmwareVersion / GetFirmwareVersion2 -> a
            // `SetSysFirmwareVersion` in an output buffer.
            //
            // This is not cosmetic. libnx's `__appInit` seeds `hosversionGet`
            // from it, and everything version-gated downstream branches on
            // that: which `acc` commands exist, which `ts` interface carries
            // the temperature, which audio-renderer revision is negotiated.
            // The generic empty-success answer left the caller reading its own
            // uninitialized buffer as the version — NX-Fetch reported "Horizon
            // OS 115.119.105", which is the ASCII of `switch-wasm user`, the
            // uid this emulator had left in that same buffer earlier.
            Some(3) | Some(4) => {
                let version = Self::firmware_version();
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for (index, &byte) in
                            version.iter().take(size as usize).enumerate()
                        {
                            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
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

    /// `SetSysFirmwareVersion`, the 0x100-byte block `set:sys` reports the
    /// system version in: the numeric version, then the platform, the build
    /// hash, and the two display strings the settings applet shows.
    fn firmware_version() -> [u8; 0x100] {
        let mut version = [0u8; 0x100];
        version[0] = FIRMWARE_VERSION.0;
        version[1] = FIRMWARE_VERSION.1;
        version[2] = FIRMWARE_VERSION.2;
        version[4] = 1; // revision_major
        let mut write = |offset: usize, text: &str, room: usize| {
            let bytes = text.as_bytes();
            let len = bytes.len().min(room - 1);
            version[offset..offset + len].copy_from_slice(&bytes[..len]);
        };
        write(0x08, "NX", 0x20);
        write(0x28, "switch-wasm", 0x40);
        let display = format!(
            "{}.{}.{}",
            FIRMWARE_VERSION.0, FIRMWARE_VERSION.1, FIRMWARE_VERSION.2
        );
        write(0x68, &display, 0x18);
        write(0x80, &format!("NintendoSDK Firmware for NX {display}-1.0"), 0x80);
        version
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
                    // GetDisplayVsyncEvent: a real copy handle, signalled
                    // once per presented frame by `Cpu::signal_vsync`.
                    Some(5202) => {
                        let h = self.alloc_event("vi:vsync", true);
                        self.vsync_event = Some(h);
                        self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
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
                let mut inline_out = Vec::new();
                let error =
                    self.nv.ioctl(&mut self.mem, fd, request, &mut argp, &inline_in, &mut inline_out)?;
                if error != 0 && std::env::var("TRACE_NV").is_ok() {
                    eprintln!("[nv] ioctl fd={fd} request={request:#x} -> error {error}");
                }
                if let Some(&(addr, len)) = recv.first() {
                    for (i, &byte) in argp.iter().take(len as usize).enumerate() {
                        self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                    }
                }
                // `nvIoctl3`'s second receive buffer: where a caller that
                // asked for its payload out-of-line reads it from. Leaving it
                // untouched is how a retail title ended up with a zeroed GPU
                // characteristics struct.
                if let Some(&(addr, len)) = recv.get(1) {
                    for (i, &byte) in inline_out.iter().take(len as usize).enumerate() {
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
                let handle = self.alloc_event("nvdrv:query-event", true);
                self.write_ipc_reply(tls, 0, &[handle], &[], &error.to_le_bytes(), &[])
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
                let event = self.alloc_event("psm:state-change", true);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
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
            self.diagnostic(&format!("[ipc] unimplemented: {iface} cmd={cmd_id:?} (pc={pc:#x})"));
        }
        self.write_ipc_response(tls, UNKNOWN_COMMAND_ID, &[], &[], &[])
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
    /// [`Cpu::unimplemented_command`] rather than a fabricated success — see there
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
                _ => self.unimplemented_command(tls, "am:control", cmd_id),
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
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
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
                _ => self.unimplemented_command(tls, &iface, cmd_id),
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
                    None => self.unimplemented_command(tls, &iface, cmd_id),
                }
            }
            // ICommonStateGetter: the state `appletMainLoop` polls every frame.
            "am:common-state-getter" => match cmd_id {
                // GetEventHandle: the copy handle the guest waits on before
                // polling ReceiveMessage.
                //
                // It stays **unsignalled**. Firing it looks right — AM really
                // does have one message queued at startup, which
                // ReceiveMessage below hands out — but nothing here enqueues
                // messages asynchronously, and `nnSdk`'s system worker waits
                // on this event holding no callback for it: reporting it
                // signalled made the worker dispatch a handler that does not
                // exist, and jump to 0. A waiter times out and polls
                // ReceiveMessage, which is where the message actually is.
                Some(0) => {
                    let h = self.alloc_event("am:applet-message", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
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
                    let h = self.alloc_event("am:common-state-getter", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
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
                _ => self.unimplemented_command(tls, &iface, cmd_id),
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
                    let h = self.alloc_event("am:self-controller", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
                }
                // SetTerminateResult / InitializeGamePlayRecording /
                // SetGamePlayRecordingState / SetDelayTimeToAbortOnGpuError:
                // nothing to record, nothing to fault, nothing to report back.
                Some(22) | Some(66) | Some(67) | Some(131) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
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
                    let h = self.alloc_event("am:gpu-error", true);
                    self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
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
                _ => self.unimplemented_command(tls, &iface, cmd_id),
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
                _ => self.unimplemented_command(tls, &iface, cmd_id),
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
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IDisplayController (capture buffers), ILibraryAppletCreator
            // (launching another applet), IDebugFunctions, and any session that
            // never named itself. Nothing here can answer those honestly: a
            // capture buffer has no contents, and a library applet has nowhere
            // to run.
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `lm`: the log manager, which is where a title's own diagnostic output
    /// goes. `nnSdk`'s `NN_LOG` and everything built on it ends up here rather
    /// than at `svcOutputDebugString`, so without this a retail title's
    /// logging is simply thrown away — which is exactly the information that
    /// makes the next failure legible.
    ///
    /// `ILogService::OpenLogger` hands back an `ILogger`, whose `Log` command
    /// carries one **LogPacket** in a send buffer: a 0x18-byte header
    /// (`pid`, `thread id`, `flags`, `severity`, `verbosity`, `payload_size`)
    /// followed by TLV chunks keyed by field. The text of the message is key
    /// 2; the rest are context the guest may or may not attach. A long message
    /// is split across packets, with `flags` bit 0 marking the first and bit 1
    /// the last, so the text is accumulated and only terminated with a newline
    /// on the tail packet.
    pub(super) fn lm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "lm:service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "lm:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("lm:service").to_string()
        } else {
            match self.service_name(handle) {
                Some("lm") | None => "lm:service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            // ILogService::OpenLogger(pid) -> ILogger.
            "lm:service" => match cmd_id {
                Some(0) => {
                    self.reply_with_interface(tls, handle, "lm:logger")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "lm:logger" => match cmd_id {
                Some(0) => {
                    // Log(buffer). `logSend` marks the buffer AutoSelect, and
                    // this service answers QueryPointerBufferSize with 0, so it
                    // always arrives as a map-alias send buffer rather than a
                    // send-static.
                    if let Some((addr, size)) = self.ipc_send_buffer(tls, 0) {
                        self.absorb_log_packet(addr, size);
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetDestination(u32): which of the console's log sinks to use.
                // There is one sink here and the guest is already using it.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// Parse one `lm` LogPacket and append its text to the guest's console
    /// output, so a title's own logging lands in the same place its
    /// `svcOutputDebugString` writes do.
    fn absorb_log_packet(&mut self, addr: u32, size: u32) {
        const HEADER_LEN: u32 = 0x18;
        const FLAG_HEAD: u8 = 1 << 0;
        const FLAG_TAIL: u8 = 1 << 1;
        // TLV keys. Only the ones worth putting in front of a human are read;
        // the rest (line number, file, function, drop count, timestamps) are
        // skipped by length like any other chunk.
        const KEY_TEXT: u8 = 2;
        const KEY_MODULE: u8 = 6;
        if size < HEADER_LEN {
            return;
        }
        let flags = self.mem.read_u8(addr.wrapping_add(0x10)).unwrap_or(0);
        let severity = self.mem.read_u8(addr.wrapping_add(0x12)).unwrap_or(0);
        let payload_size = self.mem.read_u32(addr.wrapping_add(0x14)).unwrap_or(0);
        // Trust the smaller of the declared payload and the buffer: a caller
        // that got either wrong should not walk off the end of the mapping.
        let end = payload_size.min(size - HEADER_LEN);

        let mut module = String::new();
        let mut text = String::new();
        let mut off = 0u32;
        while off + 2 <= end {
            let key = self.mem.read_u8(addr.wrapping_add(HEADER_LEN + off)).unwrap_or(0);
            let len = u32::from(
                self.mem.read_u8(addr.wrapping_add(HEADER_LEN + off + 1)).unwrap_or(0),
            );
            off += 2;
            if off + len > end {
                break;
            }
            if key == KEY_TEXT || key == KEY_MODULE {
                let mut chunk = String::with_capacity(len as usize);
                for i in 0..len {
                    match self.mem.read_u8(addr.wrapping_add(HEADER_LEN + off + i)) {
                        Ok(0) => break,
                        Ok(b) => chunk.push(b as char),
                        Err(_) => break,
                    }
                }
                if key == KEY_TEXT {
                    text.push_str(&chunk);
                } else {
                    module = chunk;
                }
            }
            off += len;
        }
        if text.is_empty() {
            return;
        }
        // A message longer than one packet is split, head to tail; only the
        // first carries the prefix and only the last ends the line.
        if flags & FLAG_HEAD != 0 {
            let level = match severity {
                0 => "TRACE",
                1 => "INFO",
                2 => "WARN",
                3 => "ERROR",
                _ => "FATAL",
            };
            let prefix = if module.is_empty() {
                format!("[lm/{level}] ")
            } else {
                format!("[lm/{level}/{module}] ")
            };
            self.out.extend_from_slice(prefix.as_bytes());
        }
        self.out.extend_from_slice(text.as_bytes());
        if flags & FLAG_TAIL != 0 && !self.out.ends_with(b"\n") {
            self.out.push(b'\n');
        }
    }

    /// `ssl`: the system TLS stack.
    ///
    /// Switch does not let a title bring its own TLS — the OS owns the
    /// implementation and the certificate store, and a title asks it to build
    /// connections: `ISslService::CreateContext` gives an `ISslContext`, whose
    /// `CreateConnection` gives an `ISslConnection` wrapping a `bsd:u` socket.
    ///
    /// The local half of that is real here: contexts and their options are
    /// ordinary objects that exist whether or not anything can be reached. The
    /// connection half is not, and is left to report itself rather than hand
    /// back a connection that can never connect — there is no socket layer
    /// under it. "A Short Hike" is offline and only calls
    /// `SetInterfaceVersion`, which `nnSdk` issues at startup because `ssl` is
    /// in the title's NPDM service list.
    pub(super) fn ssl_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ssl:service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ssl:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            if self.domain_interface(handle, object_id) == Some("ssl:context") {
                self.ssl_contexts = self.ssl_contexts.saturating_sub(1);
            }
            return self.close_domain_object(tls, handle, object_id);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("ssl:service").to_string()
        } else {
            match self.service_name(handle) {
                Some("ssl") | None => "ssl:service".to_string(),
                Some(name) => name.to_string(),
            }
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            "ssl:service" => match cmd_id {
                // CreateContext(SslVersion, pid placeholder) -> ISslContext.
                Some(0) => {
                    self.ssl_contexts += 1;
                    self.reply_with_interface(tls, handle, "ssl:context")?;
                    Ok(())
                }
                // GetContextCount.
                Some(1) => {
                    let count = self.ssl_contexts;
                    self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
                }
                // SetInterfaceVersion(u32): which revision of the interface the
                // caller speaks. Recording it is the whole implementation, and
                // it is the only `ssl` command a retail title issues at all
                // unless it goes online.
                Some(5) => {
                    self.ssl_interface_version = self.mem.read_u32(data)?;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // FlushSessionCache: nothing has been cached to flush.
                Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "ssl:context" => match cmd_id {
                // Set/GetOption(SslContextOption, s32). Options are per-context
                // state a caller reads back, so they are stored rather than
                // acknowledged and forgotten.
                Some(0) => {
                    let option = self.mem.read_u32(data)?;
                    let value = self.mem.read_u32(data.wrapping_add(4))?;
                    let key = Self::object_key(handle, object_id);
                    self.ssl_options.insert((key, option), value);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(1) => {
                    let option = self.mem.read_u32(data)?;
                    let key = Self::object_key(handle, object_id);
                    let value = self.ssl_options.get(&(key, option)).copied().unwrap_or(0);
                    self.write_ipc_response(tls, 0, &[], &value.to_le_bytes(), &[])
                }
                // GetConnectionCount: none, and none can be made — see below.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `hid`: the input service.
    ///
    /// Input arrives on Switch in two halves, and only one of them is IPC. The
    /// **data** — buttons, sticks, touch points — lives in a 256 KiB shared
    /// memory region the `hid` sysmodule writes continuously and the
    /// application reads directly, with no IPC per frame; this emulator
    /// already fills it from [`Cpu::set_gamepad_state`]. What `IHidServer`
    /// does is the **negotiation** around it: which controller styles and
    /// player slots the app supports, turning the npads and touchscreen on,
    /// and handing over the shared memory in the first place:
    ///
    /// ```text
    /// IHidServer::CreateAppletResource(aruid) -> IAppletResource
    /// IAppletResource::GetSharedMemoryHandle() -> a copy handle
    /// svcMapSharedMemory(handle, addr, 0x40000)
    /// ```
    ///
    /// None of that existed. `libnx` survived it because it maps the region by
    /// size and this emulator recognises it that way, so homebrew got working
    /// input out of a fabricated reply — but `nnSdk` calls a method on the
    /// `IAppletResource` it was handed, and a fabricated object id is not one.
    pub(super) fn hid_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "hid:server");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                // QueryPointerBufferSize: how much the server will accept in
                // a send-static ("pointer") buffer. This has to be non-zero
                // here: `nn::hid::SetSupportedNpadIdType` marshals its npad id
                // array as a pointer buffer, and `nnSdk`'s client checks the
                // negotiated size before it sends, failing outright when the
                // server claims it cannot take any.
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "hid:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("hid:server").to_string()
        } else {
            match self.service_name(handle) {
                Some("hid") | Some("hid:dbg") | None => "hid:server".to_string(),
                Some(name) => name.to_string(),
            }
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            "hid:server" => match cmd_id {
                // CreateAppletResource(aruid) -> IAppletResource.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "hid:applet-resource")?;
                    Ok(())
                }
                // Activate{DebugPad,TouchScreen,Mouse,Keyboard,Npad},
                // ActivateNpadWithRevision, DeactivateNpad, DisconnectNpad,
                // Start/StopSixAxisSensor, the joy-assignment modes,
                // SetNpadHandheldActivationMode, and the Set* half of the
                // style/id negotiation below.
                //
                // Every one of these is a setter: the shared memory this
                // emulator publishes always carries one connected handheld
                // pad, whatever the caller asks to activate, so accepting the
                // request is the whole implementation.
                Some(1) | Some(11) | Some(21) | Some(31) | Some(66) | Some(67) | Some(103)
                | Some(104) | Some(107) | Some(109) | Some(122..=125) | Some(128) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // SetSupportedNpadStyleSet(u32 style_set, aruid) and its
                // readback. A caller that sets a style set and reads back
                // something else decides the pad it wants does not exist —
                // which is what the generic reply's incrementing object id
                // looked like.
                Some(100) => {
                    self.npad_style_set = self.mem.read_u32(data)?;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(101) => {
                    let styles = self.npad_style_set;
                    self.write_ipc_response(tls, 0, &[], &styles.to_le_bytes(), &[])
                }
                // SetSupportedNpadIdType: the id list arrives in a buffer, and
                // there is one pad here regardless of which slots are asked
                // for.
                Some(102) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // AcquireNpadStyleSetUpdateEventHandle(npad_id, aruid, u64):
                // fires when a controller is connected or its style changes.
                // Nothing here hot-plugs, so it is handed out and never
                // signalled.
                Some(106) => {
                    let event = self.alloc_event("hid:npad-style-update", true);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                // GetPlayerLedPattern(npad_id) -> the four player LEDs. One
                // pad, so player 1: the first LED.
                Some(108) => self.write_ipc_response(tls, 0, &[], &1u64.to_le_bytes(), &[]),
                // Set/GetNpadJoyHoldType(aruid, u64).
                Some(120) => {
                    self.npad_joy_hold_type = self.mem.read_u64(data.wrapping_add(8))?;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(121) => {
                    let hold = self.npad_joy_hold_type;
                    self.write_ipc_response(tls, 0, &[], &hold.to_le_bytes(), &[])
                }
                // GetNpadHandheldActivationMode.
                Some(129) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
                // ---- vibration ----
                //
                // A `HidVibrationValue` is four floats: amplitude and
                // frequency for a low band and a high band. Switch rumble is
                // two linear resonant actuators driven independently, which is
                // also what the browser's Gamepad API exposes as
                // `dual-rumble`'s strong and weak magnitudes — so the two
                // amplitudes are kept and [`Cpu::vibration`] hands them to the
                // page.
                //
                // GetVibrationDeviceInfo -> { device_type, position }: a
                // linear resonant actuator (1) on the left (0).
                Some(200) => {
                    let mut info = Vec::with_capacity(8);
                    info.extend_from_slice(&1u32.to_le_bytes());
                    info.extend_from_slice(&0u32.to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &info, &[])
                }
                // SendVibrationValue(handle, HidVibrationValue, aruid): the
                // value follows the u32 handle, so the amplitudes are at +4
                // and +0xc.
                Some(201) => {
                    let low = f32::from_bits(self.mem.read_u32(data.wrapping_add(4))?);
                    let high = f32::from_bits(self.mem.read_u32(data.wrapping_add(0xc))?);
                    self.set_vibration(low, high);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetActualVibrationValue -> what is actually playing.
                Some(202) => {
                    let (low, high) = self.vibration();
                    let mut value = Vec::with_capacity(16);
                    value.extend_from_slice(&low.to_bits().to_le_bytes());
                    value.extend_from_slice(&160.0f32.to_bits().to_le_bytes());
                    value.extend_from_slice(&high.to_bits().to_le_bytes());
                    value.extend_from_slice(&320.0f32.to_bits().to_le_bytes());
                    self.write_ipc_response(tls, 0, &[], &value, &[])
                }
                // CreateActiveVibrationDeviceList -> IActiveVibrationDeviceList.
                Some(203) => {
                    self.reply_with_interface(tls, handle, "hid:vibration-devices")?;
                    Ok(())
                }
                // PermitVibration / Begin/EndPermitVibrationSession.
                Some(204) | Some(209) | Some(210) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // IsVibrationPermitted / IsVibrationDeviceMounted: there is a
                // pad and the page decides whether it can actually rumble.
                Some(205) | Some(211) => {
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                // SendVibrationValues(handles[], values[]): the arrays arrive
                // as buffers. Only the first value is kept — this emulator
                // drives one actuator pair, not one per device.
                Some(206) => {
                    if let Some((addr, size)) = self.ipc_input_buffer(tls, 1) {
                        if size >= 16 {
                            let low = f32::from_bits(self.mem.read_u32(addr)?);
                            let high = f32::from_bits(self.mem.read_u32(addr.wrapping_add(8))?);
                            self.set_vibration(low, high);
                        }
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IActiveVibrationDeviceList::InitializeVibrationDevice.
            "hid:vibration-devices" => match cmd_id {
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // IAppletResource: the handover of the shared memory the input
            // data actually lives in.
            "hid:applet-resource" => match cmd_id {
                Some(0) => {
                    let shmem = self.alloc_handle();
                    self.hid_shmem_handle = Some(shmem);
                    self.write_ipc_reply(tls, 0, &[shmem], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `pctl` and its aliases (`pctl:s`, `pctl:a`, `pctl:r`): parental
    /// controls, reported as **switched off**.
    ///
    /// There is nobody to restrict here — no accounts, no PIN, no play timer,
    /// no linked guardian — so "off" is not a placeholder, it is the true
    /// state of this console. That makes every answer determinate: a
    /// permission check succeeds (a real denial is an error `Result`, not a
    /// `false`), an "is this restricted" query is `false`, and an "is this
    /// still allowed" query is `true`. Note which way round those go — the two
    /// families read in opposite directions, and a blanket `false` would have
    /// reported free communication as *unavailable*.
    ///
    /// A retail title asks for this early: "A Short Hike" opens all four
    /// aliases before it touches the filesystem, and `nnSdk` will not start an
    /// application it believes is restricted.
    pub(super) fn pctl_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "pctl:factory");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "pctl:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("pctl:factory").to_string()
        } else {
            match self.service_name(handle) {
                // The root session is IParentalControlServiceFactory itself.
                Some("pctl") | Some("pctl:s") | Some("pctl:a") | Some("pctl:r") | None => {
                    "pctl:factory".to_string()
                }
                Some(name) => name.to_string(),
            }
        };
        match iface.as_str() {
            // IParentalControlServiceFactory::CreateService /
            // CreateServiceWithoutInitialize. The difference is whether the
            // returned interface arrives already initialized; with no settings
            // to load, both hand back the same thing.
            "pctl:factory" => match cmd_id {
                Some(0) | Some(1) => {
                    self.reply_with_interface(tls, handle, "pctl:service")?;
                    Ok(())
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "pctl:service" => match cmd_id {
                // Initialize.
                Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // The permission checks: CheckFreeCommunicationPermission,
                // ConfirmLaunchApplicationPermission,
                // ConfirmResumeApplicationPermission,
                // ConfirmSnsPostPermission,
                // ConfirmSystemSettingsPermission,
                // ConfirmStereoVisionPermission, ConfirmShowNewsPermission,
                // EndFreeCommunication,
                // ResetConfirmedStereoVisionPermission.
                //
                // These answer with a bare `Result`: success *is* "permitted",
                // and a restriction shows up as an error the caller checks for
                // by value. Nothing is restricted, so they all succeed.
                Some(1001..=1005) | Some(1013) | Some(1016) | Some(1017) | Some(1064) => {
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // IsRestrictionTemporaryUnlocked /
                // IsRestrictedSystemSettingsEntered / IsRestrictionEnabled /
                // IsPlayTimerEnabled / IsRestrictedByPlayTimer: "is something
                // restricting you" — all false.
                Some(1006) | Some(1010) | Some(1031) | Some(1453) | Some(1455) => {
                    self.write_ipc_response(tls, 0, &[], &0u8.to_le_bytes(), &[])
                }
                // IsFreeCommunicationAvailable / IsStereoVisionPermitted: "is
                // something still allowed" — the opposite sense, so both true.
                Some(1018) | Some(1065) => {
                    self.write_ipc_response(tls, 0, &[], &1u8.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// `acc:u0` (`IAccountServiceForApplication`), `acc:u1`
    /// (`IAccountServiceForSystemService`) and `acc:su`
    /// (`IAccountServiceForAdministrator`): the console's user accounts.
    ///
    /// There is **one** user here and it is always signed in. That is not a
    /// placeholder for a user database — it is what this console is: no
    /// account applet to register a second user with, no profile UI, and
    /// nowhere to persist one to. So every "who is the current user" question
    /// has a determinate answer ([`ACCOUNT_UID`]), and every list is one entry
    /// long.
    ///
    /// A title asks early and does not proceed without an answer:
    /// `nn::account::Initialize` runs before save data is mounted, and
    /// `GetLastOpenedUser`/`TrySelectUserWithoutInteraction` are how it picks
    /// whose save to open. A zero uid is the "nobody is signed in" sentinel,
    /// which is what the generic fabricated-object-id fallback was effectively
    /// answering with before this existed.
    ///
    /// The three services share commands 0..=51 and diverge from 100 up, where
    /// the *same* command id means different things (100 is
    /// `InitializeApplicationInfo` on `acc:u0` but `GetUserRegistrationNotifier`
    /// on `acc:u1`), so those arms dispatch on which service the session was
    /// opened under rather than on the command alone.
    pub(super) fn acc_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    // Which of the three services this session is decides what
                    // its 100+ commands mean, so the domain object inherits the
                    // name rather than being recorded as a generic "acc".
                    let name = self.service_name(handle).unwrap_or("acc:u0").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                // `IProfile::Get` returns its `AccountUserData` through a
                // receive-static ("pointer") buffer, and a client told the
                // server has no room for one sends no descriptor at all — then
                // reads the icon id and background colour back out of its own
                // uninitialized stack. Same reasoning as `hid`'s.
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "acc:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("acc:u0").to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "acc:u0".to_string(),
            }
        };
        match iface.as_str() {
            "acc:u0" | "acc:u1" | "acc:su" => {
                self.acc_user_service_request(tls, handle, &iface, cmd_id)
            }
            "acc:profile" | "acc:profile-editor" => self.acc_profile_request(tls, &iface, cmd_id),
            "acc:manager" => self.acc_manager_request(tls, handle, cmd_id),
            "acc:async-context" => self.acc_async_context_request(tls, cmd_id),
            // `INotifier::GetSystemEvent`, for the several notifiers `acc:u1`
            // hands out. The event is real and stays **unsignalled**: nothing
            // here ever registers a user, changes one's state, or syncs a
            // profile, so a notifier that never fires is the truthful model of
            // this console rather than a gap. (An event reported signalled
            // sends `nnSdk`'s system worker looking for a callback that was
            // never registered — see `am:applet-message`.)
            "acc:notifier" => match cmd_id {
                Some(0) => {
                    let event = self.alloc_event("acc:notifier", false);
                    self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The commands on the account service itself, as opposed to the objects
    /// it hands out. `iface` is the service the session was opened under.
    fn acc_user_service_request(
        &mut self,
        tls: u32,
        handle: u64,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        // `acc:u0` is the application-facing service, `acc:u1`/`acc:su` the
        // system-facing ones.
        let application = iface == "acc:u0";
        match cmd_id {
            // GetUserCount -> s32.
            Some(0) => self.write_ipc_response(tls, 0, &[], &1i32.to_le_bytes(), &[]),
            // GetUserExistence(AccountUid) -> bool.
            Some(1) => {
                let exists = self.acc_requested_uid(tls) == ACCOUNT_UID;
                self.write_ipc_response(tls, 0, &[], &[u8::from(exists)], &[])
            }
            // ListAllUsers / ListOpenUsers / ListOpenContextStoredUsers /
            // ListQualifiedUsers: the same one-entry list each time. The user
            // exists, is signed in, has an open context, and qualifies for
            // whatever the title is about to do — there is no sign-out, and no
            // second account, to make those four lists differ.
            //
            // 140 is the one a real title was seen asking for
            // (`[ipc] unimplemented: acc:u0 cmd=Some(140)`); libnx marshals it
            // exactly like `ListAllUsers`, an output array plus an `s32`
            // count.
            Some(2) | Some(3) | Some(60) | Some(140) => self.acc_write_user_list(tls),
            // GetLastOpenedUser -> AccountUid.
            Some(4) => self.write_ipc_response(tls, 0, &[], &ACCOUNT_UID, &[]),
            // GetProfile(AccountUid) -> IProfile.
            Some(5) => {
                if self.acc_requested_uid(tls) != ACCOUNT_UID {
                    return self.write_ipc_response(tls, ACCOUNT_USER_NOT_EXIST, &[], &[], &[]);
                }
                self.reply_with_interface(tls, handle, "acc:profile")?;
                Ok(())
            }
            // IsUserRegistrationRequestPermitted(u64) -> bool. Registering a
            // user means running the account applet, which does not exist
            // here — the one permission query on this console that is honestly
            // "no".
            Some(50) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // TrySelectUserWithoutInteraction(bool network_account_required)
            // -> AccountUid. This is how a title gets a user without putting
            // up the selector applet, and with one account it is also the
            // right answer: there is nothing to choose between.
            Some(51) => self.write_ipc_response(tls, 0, &[], &ACCOUNT_UID, &[]),
            // DebugActivateOpenContextRetention: retention is unconditional
            // here, since the one user's context is never dropped.
            Some(99) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // InitializeApplicationInfo(u64, pid): the title naming itself to
            // acc. Nothing here varies by application.
            Some(100) if application => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetBaasAccountManagerForApplication(AccountUid) ->
            // IManagerForApplication.
            Some(101) if application => {
                self.reply_with_interface(tls, handle, "acc:manager")?;
                Ok(())
            }
            // AuthenticateApplicationAsync / CheckNetworkServiceAvailabilityAsync
            // -> IAsyncContext.
            Some(102) | Some(103) if application => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // From here down the session is `acc:u1`/`acc:su`, since every
            // application arm above is guarded and matches first.
            //
            // GetUserRegistrationNotifier / GetUserStateChangeNotifier /
            // GetBaasUserAvailabilityChangeNotifier / GetProfileUpdateNotifier
            // / GetProfileSyncNotifier -> INotifier.
            Some(100) | Some(101) | Some(103) | Some(104) | Some(106) => {
                self.reply_with_interface(tls, handle, "acc:notifier")?;
                Ok(())
            }
            // GetBaasAccountManagerForSystemService(AccountUid) ->
            // IManagerForSystemService — the same interface `acc:u0`'s command
            // 101 hands an application.
            Some(102) => {
                self.reply_with_interface(tls, handle, "acc:manager")?;
                Ok(())
            }
            // CheckNetworkServiceAvailabilityAsync -> IAsyncContext.
            Some(105) => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // StoreSaveDataThumbnail(AccountUid, buffer) /
            // ClearSaveDataThumbnail(AccountUid): the picture the home menu
            // shows beside a save. There is no home menu and no thumbnail
            // store, so the thumbnail is accepted and dropped — failing a call
            // a title makes on every save would be the larger lie.
            Some(110) | Some(111) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // IsUserAccountSwitchLocked -> bool. Locked: with one account
            // there is nothing to switch to, so a title that offers the
            // switch would be offering a dead end.
            Some(150) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // IAccountServiceForAdministrator::GetProfileEditor(AccountUid) ->
            // IProfileEditor, the only route by which the nickname can be
            // changed from inside the guest.
            Some(205) if iface == "acc:su" => {
                if self.acc_requested_uid(tls) != ACCOUNT_UID {
                    return self.write_ipc_response(tls, ACCOUNT_USER_NOT_EXIST, &[], &[], &[]);
                }
                self.reply_with_interface(tls, handle, "acc:profile-editor")?;
                Ok(())
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `IProfile`, and `IProfileEditor` — the same interface plus the two
    /// store commands, which is why they share an arm.
    fn acc_profile_request(&mut self, tls: u32, iface: &str, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // Get -> ProfileBase, with the AccountUserData in an output
            // buffer. The userdata is written even though every field of it is
            // zero here: the buffer belongs to the caller, and left untouched
            // it reads back as whatever was on that stack — an icon id and a
            // background colour chosen out of garbage.
            Some(0) => {
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for i in 0..(size as usize).min(ACCOUNT_USER_DATA_LEN) as u32 {
                            self.mem.write_u8(addr.wrapping_add(i), 0)?;
                        }
                    }
                }
                let base = self.acc_profile_base();
                self.write_ipc_response(tls, 0, &[], &base, &[])
            }
            // GetBase -> ProfileBase.
            Some(1) => {
                let base = self.acc_profile_base();
                self.write_ipc_response(tls, 0, &[], &base, &[])
            }
            // GetImageSize -> u32, which has to be the exact length command 11
            // then writes: a caller sizes its buffer from this.
            Some(10) => {
                let size = profile_image().len() as u32;
                self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
            }
            // LoadImage(out buffer) -> u32 bytes written.
            Some(11) => {
                let image = profile_image();
                let mut written = 0u32;
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        let len = image.len().min(size as usize);
                        for (i, &byte) in image[..len].iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
                        }
                        written = len as u32;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &written.to_le_bytes(), &[])
            }
            // IProfileEditor::Store(ProfileBase, userdata) / StoreWithImage:
            // the nickname is the one part of this profile that is real state,
            // so a store writes it back and a later GetBase reads out what was
            // stored. Accepting an edit and then reporting the old name is the
            // failure mode a `Set`/`Get` pair always has.
            Some(100) | Some(101) if iface == "acc:profile-editor" => {
                let at = self.ipc_request_data(tls);
                let nickname = self.read_string(at.wrapping_add(0x18), NICKNAME_LEN as u32);
                self.set_user_nickname(&nickname);
                self.account_edited_at = self.unix_time;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `IManagerForApplication`/`IManagerForSystemService`: the Nintendo
    /// Account linked to the user, as far as a title can see it.
    fn acc_manager_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // CheckAvailability -> Result, where success means "this user has
            // a network service account".
            //
            // There is no Nintendo Account behind this user and no network
            // stack to authenticate one against, so this is the same trade
            // `nifm`'s permanently-connected ethernet link makes: reporting
            // the account unavailable sends a title down its offline path
            // (or into an error dialog) rather than letting it start. What it
            // still cannot get is a *token* — command 3 hands back an empty
            // one — so anything that genuinely authenticates fails there,
            // where the missing piece actually is.
            Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetAccountId -> u64 NetworkServiceAccountId. Nonzero, since zero
            // is this field's "no account" sentinel.
            Some(1) => {
                let id = NETWORK_SERVICE_ACCOUNT_ID.to_le_bytes();
                self.write_ipc_response(tls, 0, &[], &id, &[])
            }
            // EnsureIdTokenCacheAsync -> IAsyncContext.
            Some(2) => {
                self.reply_with_interface(tls, handle, "acc:async-context")?;
                Ok(())
            }
            // LoadIdTokenCache(out buffer) -> u32 size. There is no token to
            // cache, and an empty one is what an unlinked account has.
            Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "acc:manager", cmd_id),
        }
    }

    /// `IAsyncContext`: the object an `*Async` command hands back so the
    /// caller can wait for the work.
    ///
    /// Every one of those commands here answered from state that was already
    /// in hand, so the context it returns is one that has already finished:
    /// its event is signalled the moment the guest asks for it, `HasDone` is
    /// true, and the result is success. A context that never completes hangs
    /// whatever is waiting on it.
    fn acc_async_context_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        match cmd_id {
            // GetSystemEvent.
            Some(0) => {
                let event = self.alloc_event("acc:async-context", false);
                self.signal_event(event);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // Cancel: nothing is running to cancel.
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // HasDone -> bool.
            Some(2) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // GetResult -> Result.
            Some(3) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "acc:async-context", cmd_id),
        }
    }

    /// The `AccountUid` an acc command carries as the first 16 bytes of its
    /// payload.
    fn acc_requested_uid(&self, tls: u32) -> [u8; 16] {
        let at = self.ipc_request_data(tls);
        let mut uid = [0u8; 16];
        for (index, byte) in uid.iter_mut().enumerate() {
            *byte = self.mem.read_u8(at.wrapping_add(index as u32)).unwrap_or(0);
        }
        uid
    }

    /// `nn::account::ProfileBase`: the uid, when the profile was last edited,
    /// and the nickname as a NUL-padded 0x20-byte field.
    fn acc_profile_base(&self) -> [u8; PROFILE_BASE_LEN] {
        let mut base = [0u8; PROFILE_BASE_LEN];
        base[..0x10].copy_from_slice(&ACCOUNT_UID);
        base[0x10..0x18].copy_from_slice(&self.account_edited_at.to_le_bytes());
        let nickname = self.account_nickname.as_bytes();
        let len = nickname.len().min(NICKNAME_LEN - 1);
        base[0x18..0x18 + len].copy_from_slice(&nickname[..len]);
        base
    }

    /// Write the console's one uid into a list command's output buffer, and
    /// answer with how many were written.
    ///
    /// The count goes in the reply whether or not the buffer had room for the
    /// uid: the client reads a fixed-size `s32` out of the raw data, and a
    /// reply too short for it fails in its CMIF parse rather than in the
    /// command that was actually asked.
    fn acc_write_user_list(&mut self, tls: u32) -> Result<()> {
        let mut written = 0i32;
        if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
            if addr != 0 && size as usize >= ACCOUNT_UID.len() {
                for (index, &byte) in ACCOUNT_UID.iter().enumerate() {
                    self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                }
                written = 1;
            }
        }
        self.write_ipc_response(tls, 0, &[], &written.to_le_bytes(), &[])
    }

    /// `ts` (`IMeasurementServer`): the console's thermometers.
    ///
    /// Real hardware has two — one on the SoC die (`TsLocation_Internal`) and
    /// one on the PCB beside it (`TsLocation_External`) — and system-info
    /// homebrew puts their readings on screen. There is no silicon here to be
    /// warm, so both report a **fixed idle temperature**, which is the true
    /// state of a console that is not dissipating anything rather than a
    /// number standing in for one that could not be read.
    ///
    /// The two commands that report the same measurement in different units
    /// have to agree — `GetTemperatureMilliC` is `GetTemperature` times a
    /// thousand — and the reading has to sit inside the range
    /// `GetTemperatureRange` reports, or a caller that scales a gauge by that
    /// range draws it off the end.
    pub(super) fn ts_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ts");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ts:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        // A session reached over either route — its own handle, or an object
        // id on a domain — is a different interface from the server.
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("ts").to_string()
        } else {
            self.service_name(handle).unwrap_or("ts").to_string()
        };
        if iface.starts_with("ts:session") {
            return self.ts_session_request(tls, &iface, cmd_id);
        }
        // The location is a single byte of the payload: 0 = Internal (the
        // SoC), 1 = External (the PCB). Anything else reads as Internal.
        let location = self.mem.read_u8(self.ipc_request_data(tls)).unwrap_or(0);
        let celsius = TS_TEMPERATURE_C[usize::from(location).min(TS_TEMPERATURE_C.len() - 1)];
        match cmd_id {
            // GetTemperatureRange(TsLocation) -> (s32 min, s32 max): the
            // range the sensor can report over, not today's weather.
            Some(0) => {
                let mut range = [0u8; 8];
                range[..4].copy_from_slice(&TS_TEMPERATURE_RANGE_C.0.to_le_bytes());
                range[4..].copy_from_slice(&TS_TEMPERATURE_RANGE_C.1.to_le_bytes());
                self.write_ipc_response(tls, 0, &[], &range, &[])
            }
            // GetTemperature(TsLocation) -> s32 degrees Celsius.
            Some(1) => self.write_ipc_response(tls, 0, &[], &celsius.to_le_bytes(), &[]),
            // SetMeasurementMode(TsLocation, TsMeasurementMode): how often the
            // sensor is sampled. Nothing here samples anything, and the
            // reading does not vary with the mode.
            Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // GetTemperatureMilliC(TsLocation) -> s32 millidegrees.
            Some(3) => {
                let milli = celsius * 1000;
                self.write_ipc_response(tls, 0, &[], &milli.to_le_bytes(), &[])
            }
            // OpenSession(u32 device_code) -> ISession, the per-device
            // interface later firmware moved the measurement onto.
            //
            // Which sensor the session is for rides in the interface name
            // rather than in a side table, and the two names route straight
            // back here.
            //
            // The device code's **high byte** is what separates them —
            // `0x41……` is the SoC and `0x43……` the PCB — not its low byte,
            // which varies between the codes a guest may use for the same
            // sensor: NX-Fetch asks for `0x41000002` and labels what comes
            // back "CPU", so reading the low byte made it print the PCB's
            // temperature under the SoC's name.
            Some(4) => {
                let device_code = self.mem.read_u32(self.ipc_request_data(tls)).unwrap_or(0);
                let name = match device_code >> 24 {
                    0x43 => "ts:session-external",
                    _ => "ts:session-internal",
                };
                self.reply_with_interface(tls, handle, name)?;
                Ok(())
            }
            _ => self.unimplemented_command(tls, "ts", cmd_id),
        }
    }

    /// `ISession`, the per-sensor interface `ts::OpenSession` hands out.
    ///
    /// Its `GetTemperature` is **command 4 and reports a `float`**, where the
    /// server's own command 4 is `OpenSession` and its temperature commands
    /// report integers. Sharing one dispatch between the two therefore
    /// answered a session's temperature request with another session object,
    /// and NX-Fetch drew whatever the first word of that reply happened to be
    /// as the console's temperature.
    pub(super) fn ts_session_request(
        &mut self,
        tls: u32,
        iface: &str,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        let celsius = match iface {
            "ts:session-external" => TS_TEMPERATURE_C[1],
            _ => TS_TEMPERATURE_C[0],
        };
        match cmd_id {
            // GetTemperature -> f32 degrees Celsius.
            Some(4) => {
                let reading = celsius as f32;
                self.write_ipc_response(tls, 0, &[], &reading.to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, iface, cmd_id),
        }
    }

    /// `bsd:u`/`bsd:s`, the socket service — `nn::socket` and libnx's
    /// `socketInitialize` sit on top of it.
    ///
    /// **There are no peers.** A browser tab cannot open a TCP socket, and
    /// nothing in this emulator proxies one, so what is modelled is a console
    /// whose link is up (which is what `nifm` reports) and on which nothing
    /// ever answers: sockets can be created, bound, listened on, configured
    /// and closed — all of which are genuinely local operations that really do
    /// succeed — while every operation that needs someone at the other end
    /// fails, immediately and with a definite errno. `connect` is
    /// `ECONNREFUSED` rather than a timeout precisely because a title that
    /// checks for an update should find out now rather than block a frame loop
    /// that has no other thread to run.
    ///
    /// The errnos are **FreeBSD's**, not Linux's or newlib's (`EAGAIN` is 35,
    /// not 11), because that is what the real service returns and guest code
    /// is written against the real service.
    ///
    /// Both save managers here reach it the same way: `RegisterClient`,
    /// `StartMonitoring`, then a socket that gets an option set, is bound, and
    /// is closed again.
    pub(super) fn bsd_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "bsd:u");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                // Every buffer `bsd` takes is marshalled as a map-alias range
                // (libnx's AutoSelect falls back to one when the server claims
                // no pointer-buffer room), which is what `ipc_input_buffer` and
                // `ipc_output_buffer` then find.
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "bsd:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        let data = self.ipc_request_data(tls);
        let word = |cpu: &Cpu, index: u32| cpu.mem.read_u32(data.wrapping_add(index * 4)).unwrap_or(0);
        match cmd_id {
            // RegisterClient(BsdInitConfig, pid, tmem_size, tmem) -> u64. The
            // transfer memory is the buffer pool a real bsd server allocates
            // out of; nothing here needs it.
            Some(0) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // StartMonitoring(pid).
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Socket(domain, type, protocol) / SocketExempt.
            //
            // The family is not validated. `AF_INET6` is a different number in
            // FreeBSD, in newlib and in Linux, so a guest built against any of
            // them would be rejected for the wrong reason — and nothing here
            // behaves differently per family anyway, since no socket of any
            // family can reach anything.
            Some(2) | Some(3) => {
                let socket = BsdSocket {
                    domain: word(self, 0),
                    kind: word(self, 1),
                    bound: Vec::new(),
                    flags: 0,
                    listening: false,
                };
                let fd = self.next_bsd_fd;
                self.next_bsd_fd = self.next_bsd_fd.wrapping_add(1);
                self.bsd_sockets.insert(fd, socket);
                self.bsd_reply(tls, fd, 0)
            }
            // Select(nfds, timeout, read/write/except sets): nothing is ever
            // ready, so this is the timeout case — 0 descriptors, and the
            // out-sets cleared so a caller that reads them anyway sees the
            // empty set rather than its own input.
            Some(5) => {
                for index in 0..3 {
                    if let Some((addr, size)) = self.ipc_output_buffer(tls, index) {
                        for offset in 0..size {
                            self.mem.write_u8(addr.wrapping_add(offset), 0)?;
                        }
                    }
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Poll(nfds, timeout): the fds come in and go back out, with each
            // `revents` cleared — no event ever fires. Copying the array
            // through matters: the caller reads its `revents` out of the
            // *output* buffer, which is a different range from the input one.
            Some(6) => {
                if let (Some((src, src_size)), Some((dst, dst_size))) =
                    (self.ipc_input_buffer(tls, 0), self.ipc_output_buffer(tls, 0))
                {
                    // struct pollfd { s32 fd; s16 events; s16 revents; }
                    for offset in (0..src_size.min(dst_size)).step_by(8) {
                        let fd = self.mem.read_u32(src.wrapping_add(offset)).unwrap_or(0);
                        let events = self.mem.read_u16(src.wrapping_add(offset + 4)).unwrap_or(0);
                        self.mem.write_u32(dst.wrapping_add(offset), fd)?;
                        self.mem.write_u16(dst.wrapping_add(offset + 4), events)?;
                        self.mem.write_u16(dst.wrapping_add(offset + 6), 0)?;
                    }
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Recv / RecvFrom / Send / SendTo / Write / Read: the data path.
            // Nothing is connected and there is nowhere to send to, so a
            // stream socket reports ENOTCONN and a datagram one ENETUNREACH —
            // the two honest answers for "this went nowhere".
            Some(8) | Some(9) | Some(10) | Some(11) | Some(24) | Some(25) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.get(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if socket.kind == BSD_SOCK_DGRAM => {
                        self.bsd_reply(tls, -1, BSD_ENETUNREACH)
                    }
                    Some(_) => self.bsd_reply(tls, -1, BSD_ENOTCONN),
                }
            }
            // Accept(fd): a listening socket nobody will ever connect to.
            // EAGAIN says "not right now" rather than failing the listener
            // outright — which is exactly what a server socket on an idle
            // network reports, and unlike blocking forever it leaves the
            // guest's own loop able to run.
            Some(12) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.get(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if !socket.listening => self.bsd_reply(tls, -1, BSD_EINVAL),
                    Some(_) => self.bsd_reply(tls, -1, BSD_EAGAIN),
                }
            }
            // Bind(fd, sockaddr): genuinely local, and genuinely succeeds. The
            // address is kept because `GetSockName` has to report it back.
            Some(13) => {
                let address = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(0x80)),
                    None => Vec::new(),
                };
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.get_mut(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) => {
                        socket.bound = address;
                        self.bsd_reply(tls, 0, 0)
                    }
                }
            }
            // Connect(fd, sockaddr).
            Some(14) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.contains_key(&fd) {
                    false => self.bsd_reply(tls, -1, BSD_EBADF),
                    true => self.bsd_reply(tls, -1, BSD_ECONNREFUSED),
                }
            }
            // GetPeerName(fd): there is no peer.
            Some(15) => self.bsd_reply(tls, -1, BSD_ENOTCONN),
            // GetSockName(fd) -> the bound address, or the console's own
            // address (the one `nifm` reports) when nothing was bound.
            Some(16) => {
                let fd = word(self, 0) as i32;
                let address = match self.bsd_sockets.get(&fd) {
                    None => return self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if !socket.bound.is_empty() => socket.bound.clone(),
                    Some(_) => Self::bsd_local_address().to_vec(),
                };
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    for (index, &byte) in address.iter().take(size as usize).enumerate() {
                        self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                    }
                }
                self.bsd_reply(tls, 0, 0)
            }
            // GetSockOpt(fd, level, option) -> the option's value in the
            // output buffer. Options are read back, so they are stored rather
            // than acknowledged and forgotten — the same reason `ssl`'s are.
            Some(17) => {
                let (fd, level, option) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                if !self.bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let value = self.bsd_socket_options.get(&(fd, level, option)).copied().unwrap_or(0);
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if size >= 4 {
                        self.mem.write_u32(addr, value)?;
                    }
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Listen(fd, backlog).
            Some(18) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.get_mut(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) => {
                        socket.listening = true;
                        self.bsd_reply(tls, 0, 0)
                    }
                }
            }
            // Ioctl(fd, request, ...): only FIONBIO, the other way to set
            // non-blocking mode. It folds into the same flags word `fcntl`
            // reads back, so the two routes cannot disagree.
            Some(19) => {
                let (fd, request) = (word(self, 0) as i32, word(self, 1));
                let nonblocking = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) if size >= 4 => self.mem.read_u32(addr).unwrap_or(0) != 0,
                    _ => false,
                };
                match self.bsd_sockets.get_mut(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if request == BSD_FIONBIO => {
                        if nonblocking {
                            socket.flags |= BSD_O_NONBLOCK;
                        } else {
                            socket.flags &= !BSD_O_NONBLOCK;
                        }
                        self.bsd_reply(tls, 0, 0)
                    }
                    Some(_) => self.bsd_reply(tls, -1, BSD_EINVAL),
                }
            }
            // Fcntl(fd, cmd, arg): F_GETFL and F_SETFL, which between them are
            // how a guest sets and reads back O_NONBLOCK.
            //
            // F_SETFL stores the flags word **verbatim** and F_GETFL hands
            // that same word back, rather than decoding it: `O_NONBLOCK` is a
            // different bit in FreeBSD, in newlib and in Linux, and the one
            // thing that has to hold is that a guest reads back the flags it
            // set, whichever of those it was built against.
            Some(20) => {
                let (fd, command, arg) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                match self.bsd_sockets.get_mut(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) => match command {
                        BSD_F_GETFL => {
                            let flags = socket.flags as i32;
                            self.bsd_reply(tls, flags, 0)
                        }
                        BSD_F_SETFL => {
                            socket.flags = arg;
                            self.bsd_reply(tls, 0, 0)
                        }
                        _ => self.bsd_reply(tls, -1, BSD_EINVAL),
                    },
                }
            }
            // SetSockOpt(fd, level, option, value).
            Some(21) => {
                let (fd, level, option) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                if !self.bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let value = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) if size >= 4 => self.mem.read_u32(addr).unwrap_or(0),
                    _ => 0,
                };
                self.bsd_socket_options.insert((fd, level, option), value);
                self.bsd_reply(tls, 0, 0)
            }
            // Shutdown(fd, how) / ShutdownAllSockets(how): there is no
            // connection to tear down, and saying so would be a failure the
            // caller has no way to act on.
            Some(22) | Some(23) => self.bsd_reply(tls, 0, 0),
            // Close(fd).
            Some(26) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.remove(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(_) => {
                        self.bsd_socket_options.retain(|&(owner, _, _), _| owner != fd);
                        self.bsd_reply(tls, 0, 0)
                    }
                }
            }
            // DuplicateSocket(fd): a second descriptor for the same socket.
            // Nothing here shares state between the two beyond what a fresh
            // one has, since neither can carry data.
            Some(27) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.get(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) => {
                        let copy = BsdSocket {
                            domain: socket.domain,
                            kind: socket.kind,
                            bound: socket.bound.clone(),
                            flags: socket.flags,
                            listening: socket.listening,
                        };
                        let duplicate = self.next_bsd_fd;
                        self.next_bsd_fd = self.next_bsd_fd.wrapping_add(1);
                        self.bsd_sockets.insert(duplicate, copy);
                        self.bsd_reply(tls, duplicate, 0)
                    }
                }
            }
            _ => self.unimplemented_command(tls, "bsd:u", cmd_id),
        }
    }

    /// Answer a `bsd` command: every one of them replies with `{ s32 ret, s32
    /// errno }`, where `ret` is -1 on failure and `errno` is 0 on success.
    /// Reporting a failure with a zero errno is the one combination a caller
    /// cannot make sense of.
    fn bsd_reply(&mut self, tls: u32, ret: i32, errno: i32) -> Result<()> {
        let mut raw = [0u8; 8];
        raw[..4].copy_from_slice(&ret.to_le_bytes());
        raw[4..].copy_from_slice(&errno.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    /// The `sockaddr_in` `GetSockName` reports for a socket that was never
    /// bound: the address `nifm` says this console has, on port 0.
    ///
    /// Horizon's `sockaddr` is FreeBSD's — a length byte and a family byte
    /// where Linux has a 16-bit family — and the address is in network order.
    fn bsd_local_address() -> [u8; 16] {
        let mut address = [0u8; 16];
        address[0] = 16; // sin_len
        address[1] = 2; // AF_INET
        address[4..8].copy_from_slice(&NIFM_LOCAL_IP);
        address
    }

    /// `apm` (`IManager`) and `apm:sys` (`ISystemManager`): performance
    /// management — which clock profile the console runs at.
    ///
    /// There is nothing to clock here. The CPU is an interpreter, the GPU is a
    /// software rasterizer, and neither runs faster because a title asked for
    /// the docked profile. What `apm` still has to do is be *consistent*: it
    /// reports the same performance mode `am`'s `ICommonStateGetter` does
    /// (Normal — this console is handheld), and a configuration it was told to
    /// set is the configuration it reports back. A title that sets a profile
    /// and reads back a different one concludes the request failed.
    ///
    /// `apm` is opened by more or less everything: `libnx`'s `apmInitialize`
    /// runs from `__appInit`, so JKSV asks for it before it draws anything.
    pub(super) fn apm_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("apm").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                Some(QUERY_POINTER_BUFFER_SIZE) => {
                    self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "apm:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        if self.ipc_is_domain_close(tls) {
            return self.close_domain_object(tls, handle, object_id);
        }
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id).unwrap_or("apm").to_string()
        } else {
            match self.service_name(handle) {
                Some(name) => name.to_string(),
                None => "apm".to_string(),
            }
        };
        let data = self.ipc_request_data(tls);
        match iface.as_str() {
            // IManager. `apm:p` and `apm:am` are the same interface at higher
            // privilege; nothing here distinguishes them.
            "apm" | "apm:p" | "apm:am" => match cmd_id {
                // OpenSession -> ISession.
                Some(0) => {
                    self.reply_with_interface(tls, handle, "apm:session")?;
                    Ok(())
                }
                // GetPerformanceMode -> ApmPerformanceMode. Normal, which is
                // the mode `am`'s GetPerformanceMode reports and the one that
                // goes with a handheld operation mode.
                Some(1) => self.write_ipc_response(tls, 0, &[], &APM_PERFORMANCE_MODE_NORMAL.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISession.
            "apm:session" => match cmd_id {
                // SetPerformanceConfiguration(ApmPerformanceMode,
                // ApmPerformanceConfiguration): remembered per mode, because
                // command 1 has to give it back.
                Some(0) => {
                    let mode = self.mem.read_u32(data).unwrap_or(0);
                    let configuration = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
                    if let Some(slot) = self.apm_configuration.get_mut(mode as usize) {
                        *slot = configuration;
                    }
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetPerformanceConfiguration(ApmPerformanceMode) ->
                // ApmPerformanceConfiguration.
                Some(1) => {
                    let mode = self.mem.read_u32(data).unwrap_or(0) as usize;
                    let configuration = self.apm_configuration(mode);
                    self.write_ipc_response(tls, 0, &[], &configuration.to_le_bytes(), &[])
                }
                // SetCpuOverclockEnabled(bool): there is no clock to raise.
                Some(2) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            // ISystemManager, the privileged side: the system, not a title,
            // decides the mode on real hardware.
            "apm:sys" => match cmd_id {
                // RequestPerformanceMode(ApmPerformanceMode): accepted, and
                // changes nothing — the same answer as a console that is
                // already in the mode it was asked for.
                Some(0) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // ClearLastThrottlingState / LoadAndApplySettings /
                // SetCpuBoostMode(u32): nothing throttles and nothing boosts.
                Some(4) | Some(5) | Some(6) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                // GetCurrentPerformanceConfiguration ->
                // ApmPerformanceConfiguration, for the mode the console is
                // actually in.
                Some(7) => {
                    let configuration = self.apm_configuration(APM_PERFORMANCE_MODE_NORMAL as usize);
                    self.write_ipc_response(tls, 0, &[], &configuration.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The `ApmPerformanceConfiguration` in force for a performance mode: what
    /// was last set for it, or the console's default.
    fn apm_configuration(&self, mode: usize) -> u32 {
        *self
            .apm_configuration
            .get(mode)
            .unwrap_or(&APM_DEFAULT_CONFIGURATION[APM_PERFORMANCE_MODE_NORMAL as usize])
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
                self.write_ipc_response(tls, 0, &[], &NIFM_LOCAL_IP, &[])
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
    /// `IAudioOutManager` (`audout:u`): the plain PCM-out device, which is
    /// what `nn::audio::OpenDefaultAudioOut` and libnx's `audoutInitialize`
    /// open. The renderer (`audren`) is a separate, much larger interface.
    ///
    /// Only one device exists here, `DeviceOut`, at whatever rate and channel
    /// count the guest asks for. Real `audout` resamples everything to 48 kHz
    /// stereo; the samples are handed to the host verbatim instead, with the
    /// format alongside them, so nothing is resampled twice.
    pub(super) fn audout_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        // Both clients that reach this — `nnSdk` and libnx's `audoutInitialize`
        // — keep `audout` as a plain session and take the `IAudioOut` back as a
        // move handle. A domain request would need the reply to carry an object
        // id instead, so say so rather than hand back a handle it cannot use.
        if self.ipc_is_domain_request(tls) {
            return self.unimplemented_command(tls, "audout:u (domain)", cmd_id);
        }
        /// `AudioOutName`: a fixed 0x20-byte NUL-padded device name.
        const NAME_LEN: u32 = 0x20;
        /// The name real `audout` reports for the console's only output.
        const DEVICE: &[u8] = b"DeviceOut\0";
        match cmd_id {
            // ListAudioOuts / ListAudioOutsAuto: one device.
            Some(0) | Some(2) => {
                if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
                    for i in 0..NAME_LEN {
                        let b = DEVICE.get(i as usize).copied().unwrap_or(0);
                        let _ = self.mem.write_u8(buf.wrapping_add(i), b);
                    }
                }
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            // OpenAudioOut / OpenAudioOutAuto: in { u32 sample_rate, u32
            // channel_count, u64 aruid }, out { u32 sample_rate, u32
            // channel_count, u32 pcm_format, u32 state } and the IAudioOut.
            Some(1) | Some(3) => {
                let data = self.ipc_request_data(tls);
                let asked_rate = self.mem.read_u32(data).unwrap_or(0);
                // The channel count is 16 bits wide on the wire and the two
                // bytes above it are padding the caller does not initialise.
                // Reading the whole word and echoing it back is how `nnSdk`
                // came to believe the device had 0xcafe0002 channels --
                // negative, so its "channelCount > 0" check failed, so Unity
                // tore audio down and re-opened, and the second open aborted
                // the process with `audio` result 2153-0009.
                let asked_channels = self.mem.read_u16(data.wrapping_add(4)).unwrap_or(0);
                // A guest that asks for 0 means "whatever the device is".
                let sample_rate = if asked_rate == 0 { 48_000 } else { asked_rate };
                let channel_count = u32::from(if asked_channels == 0 { 2 } else { asked_channels });

                if let Some(buf) = self.ipc_recv_buffer_addr(tls, 0) {
                    for i in 0..NAME_LEN {
                        let b = DEVICE.get(i as usize).copied().unwrap_or(0);
                        let _ = self.mem.write_u8(buf.wrapping_add(i), b);
                    }
                }

                let handle = self.alloc_handle();
                self.record_handle(handle, "audout:iaudioout");
                let event = self.alloc_event("audout:buffer", true);
                self.audio_outs.insert(
                    handle,
                    AudioOut {
                        sample_rate,
                        channel_count,
                        started: false,
                        volume: 1.0,
                        event,
                        released: VecDeque::new(),
                        played_frames: 0,
                    },
                );
                self.audio_format = (sample_rate, channel_count);

                let mut raw = Vec::with_capacity(16);
                raw.extend_from_slice(&sample_rate.to_le_bytes());
                raw.extend_from_slice(&channel_count.to_le_bytes());
                raw.extend_from_slice(&PCM_FORMAT_INT16.to_le_bytes());
                raw.extend_from_slice(&AUDIO_OUT_STOPPED.to_le_bytes());
                self.write_ipc_response(tls, 0, &[handle], &raw, &[])
            }
            _ => self.unimplemented_command(tls, "audout:u", cmd_id),
        }
    }

    /// `IAudioOut`: one open output device.
    ///
    /// The buffer protocol is the whole interface. The guest appends a buffer,
    /// waits on the event from `RegisterBufferEvent`, then collects the tags of
    /// the buffers the device has finished with. Here a buffer is finished the
    /// moment its samples have been copied out for the host, so every append
    /// releases immediately — a device that never falls behind.
    pub(super) fn audio_out_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u16.to_le_bytes(), &[]),
                _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
            };
        }
        match cmd_id {
            // GetAudioOutState.
            Some(0) => {
                let started =
                    self.audio_outs.get(&handle).map(|d| d.started).unwrap_or(false);
                let state = if started { AUDIO_OUT_STARTED } else { AUDIO_OUT_STOPPED };
                self.write_ipc_response(tls, 0, &[], &state.to_le_bytes(), &[])
            }
            // StartAudioOut / StopAudioOut.
            Some(1) | Some(2) => {
                let started = cmd_id == Some(1);
                if let Some(device) = self.audio_outs.get_mut(&handle) {
                    device.started = started;
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // AppendAudioOutBuffer / AppendAudioOutBufferAuto.
            Some(3) | Some(7) => self.audio_out_append(tls, handle),
            // RegisterBufferEvent: the event a released buffer signals. Events
            // are copy handles.
            Some(4) => {
                let Some(event) = self.audio_outs.get(&handle).map(|d| d.event) else {
                    return self.unimplemented_command(tls, "audout:iaudioout", cmd_id);
                };
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // GetReleasedAudioOutBuffer / ...Auto: as many tags as fit.
            Some(5) | Some(8) => self.audio_out_release(tls, handle),
            // ContainsAudioOutBuffer.
            Some(6) => {
                let data = self.ipc_request_data(tls);
                let tag = self.mem.read_u64(data).unwrap_or(0);
                let held = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.released.contains(&tag))
                    .unwrap_or(false);
                self.write_ipc_response(tls, 0, &[], &[u8::from(held)], &[])
            }
            // GetAudioOutBufferCount: buffers appended and not yet collected.
            Some(9) => {
                let count = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.released.len() as u32)
                    .unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // GetAudioOutPlayedSampleCount.
            Some(10) => {
                let frames =
                    self.audio_outs.get(&handle).map(|d| d.played_frames).unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &frames.to_le_bytes(), &[])
            }
            // FlushAudioOutBuffers: nothing is ever in flight, so nothing is
            // ever flushed — the bool says so.
            Some(11) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // SetAudioOutVolume / GetAudioOutVolume.
            Some(12) => {
                let data = self.ipc_request_data(tls);
                let volume = f32::from_bits(self.mem.read_u32(data).unwrap_or(0));
                if let Some(device) = self.audio_outs.get_mut(&handle) {
                    device.volume = if volume.is_finite() { volume.clamp(0.0, 1.0) } else { 1.0 };
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(13) => {
                let volume =
                    self.audio_outs.get(&handle).map(|d| d.volume).unwrap_or(1.0);
                self.write_ipc_response(tls, 0, &[], &volume.to_bits().to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "audout:iaudioout", cmd_id),
        }
    }

    /// `AppendAudioOutBuffer`: copy the guest's samples out for the host and
    /// release the buffer's tag.
    fn audio_out_append(&mut self, tls: u32, handle: u64) -> Result<()> {
        let data = self.ipc_request_data(tls);
        let tag = self.mem.read_u64(data).unwrap_or(0);
        // `AudioOutBuffer`: { next, buffer, buffer_size, data_size,
        // data_offset }, all 8 bytes, travelling as an input buffer.
        let mut samples = Vec::new();
        if let Some((desc, _)) = self.ipc_input_buffer(tls, 0) {
            let buffer = self.mem.read_u64(desc.wrapping_add(8)).unwrap_or(0) as u32;
            let data_size = self.mem.read_u64(desc.wrapping_add(24)).unwrap_or(0) as u32;
            let data_offset = self.mem.read_u64(desc.wrapping_add(32)).unwrap_or(0) as u32;
            let start = buffer.wrapping_add(data_offset);
            for i in 0..data_size / 2 {
                let sample = self.mem.read_u16(start.wrapping_add(i * 2)).unwrap_or(0);
                samples.push(sample as i16);
            }
        }
        let Some(device) = self.audio_outs.get_mut(&handle) else {
            return self.unimplemented_command(tls, "audout:iaudioout", Some(3));
        };
        let channels = device.channel_count.max(1) as usize;
        let format = (device.sample_rate, device.channel_count);
        device.played_frames += (samples.len() / channels) as u64;
        device.released.push_back(tag);
        let volume = device.volume;
        let event = device.event;
        // A stopped device is not playing: its buffers still come back (the
        // guest is entitled to its memory) but the samples are not queued.
        let playing = device.started;
        if playing {
            // Whichever device is actually producing samples defines the
            // format the host plays them in.
            self.audio_format = format;
            let scaled = samples
                .into_iter()
                .map(move |s| ((s as f32) * volume).round().clamp(-32768.0, 32767.0) as i16);
            self.queue_audio(scaled);
        }
        self.signal_event(event);
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// `GetReleasedAudioOutBuffer`: hand back the tags of finished buffers,
    /// as many as the guest's out buffer has room for.
    fn audio_out_release(&mut self, tls: u32, handle: u64) -> Result<()> {
        let room = self
            .ipc_recv_buffer(tls, 0)
            .map(|(_, size)| size / 8)
            .unwrap_or(0);
        let addr = self.ipc_recv_buffer_addr(tls, 0);
        let mut tags = Vec::new();
        if let Some(device) = self.audio_outs.get_mut(&handle) {
            while (tags.len() as u32) < room {
                match device.released.pop_front() {
                    Some(tag) => tags.push(tag),
                    None => break,
                }
            }
        }
        if let Some(addr) = addr {
            for (i, &tag) in tags.iter().enumerate() {
                let _ = self.mem.write_u64(addr.wrapping_add(i as u32 * 8), tag);
            }
        }
        self.write_ipc_response(tls, 0, &[], &(tags.len() as u32).to_le_bytes(), &[])
    }

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

/// The user's profile picture, as the JPEG `IProfile::LoadImage` is defined to
/// return.
///
/// `acc` hands the icon out as an encoded JPEG and callers feed what they get
/// straight to a decoder, so answering with zero bytes leaves anything that
/// draws the user's picture with nothing to decode. There is no icon on this
/// console to hand over, so one is made: a plain field of colour, which is
/// what an account with no picture set should look like.
fn profile_image() -> Vec<u8> {
    solid_jpeg(PROFILE_IMAGE_SIZE, PROFILE_IMAGE_COLOR)
}

/// A baseline JPEG of a single solid colour, `size` x `size` pixels.
///
/// Encoding a constant image needs no DCT and no zig-zag: the transform of a
/// block of constant level-shifted value `x` is one DC coefficient of `8x`
/// with every AC coefficient zero. So each block is a Huffman-coded DC
/// *difference* — nonzero only in the first block of each component, since the
/// predictor is the previous block's DC and every block is the same — followed
/// by end-of-block. With a quantization table of 8 throughout, `8x`
/// quantizes to exactly `x` and dequantizes back to `8x`, so the colour
/// survives the round trip unchanged.
///
/// The Huffman tables are minimal rather than Annex K's: an encoder that emits
/// only DC categories and EOB needs no other symbols, and the tables travel in
/// the file anyway. Both are complete codes (their Kraft sums are 1), which is
/// what a decoder building a derived table expects.
fn solid_jpeg(size: u16, rgb: (u8, u8, u8)) -> Vec<u8> {
    let (red, green, blue) = (f32::from(rgb.0), f32::from(rgb.1), f32::from(rgb.2));
    let round = |value: f32| value.round().clamp(0.0, 255.0) as i32;
    // JFIF's RGB -> YCbCr (BT.601), the colour space a baseline JPEG's three
    // components are in.
    let components = [
        round(0.299 * red + 0.587 * green + 0.114 * blue),
        round(-0.168_736 * red - 0.331_264 * green + 0.5 * blue + 128.0),
        round(0.5 * red - 0.418_688 * green - 0.081_312 * blue + 128.0),
    ];

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&[0xFF, JPEG_SOI]);
    // APP0/JFIF: version 1.1, no density units, no thumbnail.
    segment(&mut out, JPEG_APP0, b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00");
    // One quantization table (id 0), 8-bit precision, used by all three
    // components.
    let mut quant = vec![0u8];
    quant.extend_from_slice(&[JPEG_QUANT; 64]);
    segment(&mut out, JPEG_DQT, &quant);
    // SOF0: 8-bit samples, `size` square, three components each sampled 1x1
    // (no chroma subsampling, so one block per component per MCU).
    let mut frame = vec![8];
    frame.extend_from_slice(&size.to_be_bytes());
    frame.extend_from_slice(&size.to_be_bytes());
    frame.push(3);
    for id in 1..=3u8 {
        frame.extend_from_slice(&[id, 0x11, 0]);
    }
    segment(&mut out, JPEG_SOF0, &frame);
    // The two Huffman tables: class 0 (DC) id 0, then class 1 (AC) id 0.
    for (class, bits, values) in [
        (0x00u8, &JPEG_DC_BITS, &JPEG_DC_VALUES[..]),
        (0x10u8, &JPEG_AC_BITS, &JPEG_AC_VALUES[..]),
    ] {
        let mut table = vec![class];
        table.extend_from_slice(bits);
        table.extend_from_slice(values);
        segment(&mut out, JPEG_DHT, &table);
    }
    // SOS: all three components, each using table pair 0, full spectral
    // selection (a baseline sequential scan).
    let mut scan = vec![3];
    for id in 1..=3u8 {
        scan.extend_from_slice(&[id, 0x00]);
    }
    scan.extend_from_slice(&[0, 63, 0]);
    segment(&mut out, JPEG_SOS, &scan);

    let dc_codes = huffman_codes(&JPEG_DC_BITS, &JPEG_DC_VALUES);
    let ac_codes = huffman_codes(&JPEG_AC_BITS, &JPEG_AC_VALUES);
    let code_for = |codes: &[(u8, u16, u8)], symbol: u8| -> (u32, u32) {
        let (_, code, length) = codes
            .iter()
            .find(|&&(candidate, _, _)| candidate == symbol)
            .expect("the tables above cover every symbol this emits");
        (u32::from(*code), u32::from(*length))
    };

    let mcus = u32::from(size).div_ceil(8) * u32::from(size).div_ceil(8);
    let mut bits = JpegBits::default();
    for mcu in 0..mcus {
        for &component in &components {
            // The level shift, and the DC predictor: the first block of each
            // component carries the whole value, every later one differs from
            // its predecessor by nothing.
            let diff = if mcu == 0 { component - 128 } else { 0 };
            let category = if diff == 0 { 0 } else { 32 - diff.unsigned_abs().leading_zeros() };
            let (code, length) = code_for(&dc_codes, category as u8);
            bits.push(code, length);
            if category > 0 {
                // A negative difference is sent as its one's complement in
                // `category` bits, which is what makes the leading bit the
                // sign.
                let value = if diff > 0 { diff } else { diff + (1 << category) - 1 };
                bits.push(value as u32, category);
            }
            // Every AC coefficient of a constant block is zero.
            let (code, length) = code_for(&ac_codes, JPEG_EOB);
            bits.push(code, length);
        }
    }
    out.extend_from_slice(&bits.finish());
    out.extend_from_slice(&[0xFF, JPEG_EOI]);
    out
}

/// A marker segment: `FF <marker>`, the payload length including its own two
/// bytes, then the payload.
fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// Canonical Huffman codes from a JPEG `BITS`/`HUFFVAL` pair, as
/// `(symbol, code, length)` — the generation procedure from the spec's Annex
/// C, and the same walk a decoder makes to rebuild them from the DHT segment.
fn huffman_codes(bits: &[u8; 16], values: &[u8]) -> Vec<(u8, u16, u8)> {
    let mut codes = Vec::with_capacity(values.len());
    let mut code = 0u16;
    let mut next = 0usize;
    for (index, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            codes.push((values[next], code, index as u8 + 1));
            code += 1;
            next += 1;
        }
        code <<= 1;
    }
    codes
}

/// The entropy-coded segment's bit stream, most significant bit first.
#[derive(Default)]
struct JpegBits {
    out: Vec<u8>,
    accumulator: u32,
    filled: u32,
}

impl JpegBits {
    fn push(&mut self, code: u32, length: u32) {
        for shift in (0..length).rev() {
            self.accumulator = (self.accumulator << 1) | ((code >> shift) & 1);
            self.filled += 1;
            if self.filled == 8 {
                let byte = self.accumulator as u8;
                self.out.push(byte);
                // Byte stuffing: an 0xFF inside the entropy stream is followed
                // by a 0x00 so a decoder cannot mistake it for a marker.
                if byte == 0xFF {
                    self.out.push(0x00);
                }
                self.accumulator = 0;
                self.filled = 0;
            }
        }
    }

    /// Pad the final partial byte with 1 bits, which is what the spec calls
    /// for — a 1-filled tail cannot be confused with the start of a marker.
    fn finish(mut self) -> Vec<u8> {
        while self.filled != 0 {
            self.push(1, 1);
        }
        self.out
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

    /// Overwrite the TLS buffer with a fresh request, so a test can drive a
    /// second command against the state the first one left behind. The buffer
    /// is cleared first: a reply leaves an `SFCO` header in it, and the
    /// command-id scan looks for a magic.
    fn write_request(cpu: &mut Cpu, command_id: u32, payload: &[u8]) {
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
    fn request_with_recv_buffer(command_id: u32, payload: &[u8], buffer: u32, size: u32) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        write_map_buffer_request(&mut cpu, command_id, payload, buffer, size, false);
        cpu
    }

    /// Write a request carrying one map-alias buffer, on the send side or the
    /// receive side, into an existing `Cpu`'s TLS.
    fn write_map_buffer_request(
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
    fn request_with_recv_static(command_id: u32, payload: &[u8], buffer: u32, size: u32) -> Cpu {
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

    /// Drive one acc command on a session opened under `service`.
    fn acc(cpu: &mut Cpu, service: &str, command_id: u32) {
        cpu.register_service_handle(9, service);
        cpu.acc_request(TLS, 9, Some(command_id)).unwrap();
    }

    #[test]
    fn acc_reports_one_user_who_is_signed_in() {
        // GetUserCount.
        let mut cpu = request(false, 0, &[]);
        acc(&mut cpu, "acc:u0", 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1);

        // GetLastOpenedUser: the uid, and not the zero that means "nobody is
        // signed in".
        let mut cpu = request(false, 4, &[]);
        acc(&mut cpu, "acc:u0", 4);
        let uid = cpu.read_bytes(TLS + 0x20, 16);
        assert_eq!(uid, super::ACCOUNT_UID.to_vec());
        assert_ne!(uid, vec![0u8; 16]);

        // TrySelectUserWithoutInteraction hands back the same one, since there
        // is nothing to choose between.
        let mut cpu = request(false, 51, &[0, 0, 0, 0]);
        acc(&mut cpu, "acc:u0", 51);
        assert_eq!(cpu.read_bytes(TLS + 0x20, 16), super::ACCOUNT_UID.to_vec());
    }

    #[test]
    fn acc_list_all_users_fills_the_output_buffer_and_counts_what_it_wrote() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(2, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        acc(&mut cpu, "acc:u0", 2);

        assert_eq!(cpu.read_bytes(BUFFER, 16), super::ACCOUNT_UID.to_vec());
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "one uid written");
    }

    #[test]
    fn acc_knows_only_its_own_uid() {
        // GetUserExistence for the one user, then for a uid nothing handed out.
        let mut cpu = request(false, 1, &super::ACCOUNT_UID);
        acc(&mut cpu, "acc:u0", 1);
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1);

        let mut cpu = request(false, 1, &[0xAB; 16]);
        acc(&mut cpu, "acc:u0", 1);
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0);

        // GetProfile for that same invented uid fails rather than handing back
        // the one user's profile under someone else's id.
        let mut cpu = request(false, 5, &[0xAB; 16]);
        acc(&mut cpu, "acc:u0", 5);
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), super::ACCOUNT_USER_NOT_EXIST);
    }

    #[test]
    fn acc_profile_get_writes_the_userdata_into_its_pointer_buffer() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_static(0, &[], BUFFER, 0x80);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        // Whatever was on the caller's stack. Left alone, this is what it
        // would read back as its icon id and background colour.
        for offset in 0..0x80 {
            cpu.mem.write_u8(BUFFER + offset, 0xAA).unwrap();
        }
        assert_eq!(cpu.ipc_recv_static_buffers(TLS), vec![(BUFFER, 0x80)]);

        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(0)).unwrap();

        assert_eq!(cpu.read_bytes(BUFFER, 0x80), vec![0u8; 0x80], "userdata zeroed, not left as stack garbage");
        // ProfileBase: the uid, then the never-edited timestamp, then the
        // nickname.
        assert_eq!(cpu.read_bytes(TLS + 0x20, 16), super::ACCOUNT_UID.to_vec());
        assert_eq!(cpu.mem.read_u64(TLS + 0x30).unwrap(), 0);
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Player");
    }

    #[test]
    fn acc_profile_editor_stores_a_nickname_that_reads_back() {
        let mut store = [0u8; super::PROFILE_BASE_LEN];
        store[..16].copy_from_slice(&super::ACCOUNT_UID);
        store[0x18..0x18 + 5].copy_from_slice(b"Yuuto");
        let mut cpu = request(false, 100, &store);
        cpu.set_unix_time(1_700_000_000);
        cpu.register_service_handle(9, "acc:profile-editor");
        cpu.acc_request(TLS, 9, Some(100)).unwrap();
        assert_eq!(cpu.user_nickname(), "Yuuto");

        // GetBase reports what was stored, timestamp included — a store the
        // service accepts and then forgets is the failure mode every
        // Set/Get pair has.
        write_request(&mut cpu, 1, &[]);
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(1)).unwrap();
        assert_eq!(cpu.read_string(TLS + 0x38, 0x20), "Yuuto");
        assert_eq!(cpu.mem.read_u64(TLS + 0x30).unwrap(), 1_700_000_000);
    }

    #[test]
    fn acc_the_same_command_id_means_different_things_on_u0_and_u1() {
        // 101 is GetBaasAccountManagerForApplication on acc:u0 and
        // GetUserStateChangeNotifier on acc:u1. Both hand back a session, so
        // the only way to tell them apart is what that session then answers:
        // the notifier has a GetSystemEvent, the manager has a GetAccountId.
        for (service, iface) in [("acc:u0", "acc:manager"), ("acc:u1", "acc:notifier")] {
            let mut cpu = request(false, 101, &super::ACCOUNT_UID);
            acc(&mut cpu, service, 101);
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_eq!(cpu.service_name(session), Some(iface), "{service} cmd 101");
        }
    }

    #[test]
    fn acc_async_contexts_report_work_that_is_already_finished() {
        // CheckNetworkServiceAvailabilityAsync, then HasDone on what it
        // returned. A context that never completes hangs its waiter.
        let mut cpu = request(false, 103, &[]);
        acc(&mut cpu, "acc:u0", 103);
        let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(session), Some("acc:async-context"));

        let mut cpu = request(false, 2, &[]);
        cpu.register_service_handle(session, "acc:async-context");
        cpu.acc_request(TLS, session, Some(2)).unwrap();
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1);
    }

    #[test]
    fn acc_load_image_writes_exactly_the_size_it_advertised() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request(false, 10, &[]);
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(10)).unwrap();
        let advertised = cpu.mem.read_u32(TLS + 0x20).unwrap();
        assert!(advertised > 0, "an icon of no bytes is nothing to decode");

        let mut cpu = request_with_recv_buffer(11, &[], BUFFER, advertised);
        cpu.mem.map_zero(BUFFER, advertised as usize + 0x100).unwrap();
        cpu.register_service_handle(9, "acc:profile");
        cpu.acc_request(TLS, 9, Some(11)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), advertised);
        assert_eq!(cpu.read_bytes(BUFFER, advertised), super::profile_image());
    }

    /// Decode the profile icon: walk its markers, rebuild the Huffman tables
    /// out of the DHT segments the file itself carries, and run the whole
    /// entropy-coded scan.
    ///
    /// A constant image is the strongest thing to assert against — every one
    /// of the 3072 blocks has to decode to the same colour, and the bit stream
    /// has to run out exactly at the EOI marker. That covers the tables, the
    /// canonical code generation, the DC prediction and the byte stuffing,
    /// none of which can be checked by eye.
    #[test]
    fn the_profile_icon_is_a_jpeg_that_decodes_to_one_colour() {
        let jpeg = super::profile_image();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI");

        // Walk the marker segments, keeping what the scan needs.
        let mut quant = [0u8; 64];
        let mut tables: Vec<(u8, Vec<(u8, u16, u8)>)> = Vec::new();
        let (mut width, mut height) = (0u32, 0u32);
        let mut components = 0usize;
        let mut scan_start = 0usize;
        let mut at = 2usize;
        while at + 4 <= jpeg.len() {
            assert_eq!(jpeg[at], 0xFF, "a segment starts with a marker");
            let marker = jpeg[at + 1];
            let length = u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]) as usize;
            let payload = &jpeg[at + 4..at + 2 + length];
            match marker {
                super::JPEG_DQT => {
                    assert_eq!(payload[0], 0, "8-bit precision, table 0");
                    quant.copy_from_slice(&payload[1..65]);
                }
                super::JPEG_SOF0 => {
                    assert_eq!(payload[0], 8, "8-bit samples");
                    height = u32::from(u16::from_be_bytes([payload[1], payload[2]]));
                    width = u32::from(u16::from_be_bytes([payload[3], payload[4]]));
                    components = payload[5] as usize;
                    for index in 0..components {
                        // 1x1 sampling: one block per component per MCU, so no
                        // subsampling to undo below.
                        assert_eq!(payload[7 + index * 3], 0x11);
                    }
                }
                super::JPEG_DHT => {
                    let bits: [u8; 16] = payload[1..17].try_into().unwrap();
                    let count: usize = bits.iter().map(|&b| b as usize).sum();
                    let mut codes = Vec::new();
                    let (mut code, mut next) = (0u16, 0usize);
                    for (index, &in_this_length) in bits.iter().enumerate() {
                        for _ in 0..in_this_length {
                            codes.push((payload[17 + next], code, index as u8 + 1));
                            code += 1;
                            next += 1;
                        }
                        code <<= 1;
                    }
                    assert_eq!(next, count);
                    tables.push((payload[0], codes));
                }
                super::JPEG_SOS => {
                    for index in 0..components {
                        assert_eq!(payload[2 + index * 2], 0x00, "both tables are id 0");
                    }
                    scan_start = at + 2 + length;
                    break;
                }
                _ => {}
            }
            at += 2 + length;
        }
        assert_eq!((width, height), (256, 256));
        assert_eq!(components, 3);

        // The entropy-coded segment, up to the EOI. 0xFF00 is a stuffed 0xFF.
        let mut scan = Vec::new();
        let mut at = scan_start;
        while at < jpeg.len() {
            if jpeg[at] == 0xFF {
                match jpeg[at + 1] {
                    0x00 => {
                        scan.push(0xFFu8);
                        at += 2;
                        continue;
                    }
                    super::JPEG_EOI => break,
                    other => panic!("unexpected marker {other:#x} inside the scan"),
                }
            }
            scan.push(jpeg[at]);
            at += 1;
        }
        assert_eq!(at, jpeg.len() - 2, "the scan runs right up to the EOI");

        /// A cursor over the scan's bits, most significant bit of each byte
        /// first, which is the order an entropy-coded segment is packed in.
        struct Reader<'a> {
            data: &'a [u8],
            bit: usize,
        }
        impl Reader<'_> {
            fn bit(&mut self) -> u32 {
                let value = u32::from(self.data[self.bit / 8] >> (7 - self.bit % 8)) & 1;
                self.bit += 1;
                value
            }

            /// Read one Huffman-coded symbol: extend the code a bit at a time
            /// until it matches one the table defines, which is unambiguous
            /// because no code is a prefix of another.
            fn symbol(&mut self, codes: &[(u8, u16, u8)]) -> u8 {
                let (mut code, mut length) = (0u16, 0u8);
                for _ in 0..16 {
                    code = (code << 1) | self.bit() as u16;
                    length += 1;
                    let found = codes.iter().find(|&&(_, candidate, candidate_length)| {
                        candidate == code && candidate_length == length
                    });
                    if let Some(&(symbol, _, _)) = found {
                        return symbol;
                    }
                }
                panic!("no Huffman code matched");
            }
        }
        let dc_table = &tables.iter().find(|(id, _)| *id == 0x00).unwrap().1;
        let ac_table = &tables.iter().find(|(id, _)| *id == 0x10).unwrap().1;
        let mut reader = Reader { data: &scan, bit: 0 };

        // Every block, in MCU order: a DC difference then an immediate
        // end-of-block, with the DC predictor carried per component.
        let blocks = width.div_ceil(8) * height.div_ceil(8);
        let mut predictor = [0i32; 3];
        for mcu in 0..blocks {
            for component in 0..3usize {
                let category = reader.symbol(dc_table);
                let mut diff = 0i32;
                if category > 0 {
                    let mut value = 0i32;
                    for _ in 0..category {
                        value = (value << 1) | reader.bit() as i32;
                    }
                    // The sign convention: a leading zero bit means the value
                    // is negative and stored as its one's complement.
                    diff = if value >= 1 << (category - 1) {
                        value
                    } else {
                        value - (1 << category) + 1
                    };
                }
                predictor[component] += diff;
                assert_eq!(reader.symbol(ac_table), super::JPEG_EOB, "AC of a flat block");

                // Dequantize and undo the level shift: the inverse DCT of a
                // lone DC coefficient is that coefficient over 8, everywhere.
                let value = predictor[component] * i32::from(quant[0]) / 8 + 128;
                let (red, green, blue) = super::PROFILE_IMAGE_COLOR;
                let (red, green, blue) = (f32::from(red), f32::from(green), f32::from(blue));
                let expected = match component {
                    0 => 0.299 * red + 0.587 * green + 0.114 * blue,
                    1 => -0.168_736 * red - 0.331_264 * green + 0.5 * blue + 128.0,
                    _ => 0.5 * red - 0.418_688 * green - 0.081_312 * blue + 128.0,
                };
                assert_eq!(value, expected.round() as i32, "mcu {mcu} component {component}");
            }
        }
        // Only the 1-padding of the last byte may be left over.
        assert!(
            scan.len() * 8 - reader.bit < 8,
            "the scan decodes to exactly the blocks the frame declares"
        );
    }

    /// Read a `bsd` command's `{ s32 ret, s32 errno }` reply.
    fn bsd_result(cpu: &Cpu) -> (i32, i32) {
        (
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            cpu.mem.read_u32(TLS + 0x24).unwrap() as i32,
        )
    }

    /// Open a socket of `kind` on a fresh `bsd:u` session, returning the cpu
    /// and the descriptor.
    fn bsd_socket(kind: u32) -> (Cpu, i32) {
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&2u32.to_le_bytes()); // AF_INET
        payload[4..8].copy_from_slice(&kind.to_le_bytes());
        let mut cpu = request(false, 2, &payload);
        cpu.register_service_handle(9, "bsd:u");
        cpu.bsd_request(TLS, 9, Some(2)).unwrap();
        let (fd, errno) = bsd_result(&cpu);
        assert_eq!(errno, 0);
        (cpu, fd)
    }

    #[test]
    fn ts_open_session_picks_the_sensor_by_the_device_code() {
        // The high byte separates them: 0x41…… is the SoC and 0x43…… the PCB.
        // NX-Fetch asks for 0x41000002 and labels what comes back "CPU", so
        // reading the *low* byte handed it the PCB's temperature under the
        // SoC's name.
        for (device_code, expected) in
            [(0x4100_0002u32, "ts:session-internal"), (0x4300_0001, "ts:session-external")]
        {
            let mut cpu = request(false, 4, &device_code.to_le_bytes());
            cpu.register_service_handle(9, "ts");
            cpu.ts_request(TLS, 9, Some(4)).unwrap();
            let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
            assert_eq!(cpu.service_name(session), Some(expected), "{device_code:#x}");
        }
    }

    #[test]
    fn ts_sessions_report_their_own_sensor_as_a_float() {
        // `ISession::GetTemperature` is command 4 — the same id the *server*
        // uses for OpenSession — and reports a float. Sharing one dispatch
        // between them answered a temperature request with a session object,
        // which is what NX-Fetch drew as "8 C".
        for (iface, expected) in [
            ("ts:session-internal", super::TS_TEMPERATURE_C[0]),
            ("ts:session-external", super::TS_TEMPERATURE_C[1]),
        ] {
            let mut cpu = request(false, 4, &[]);
            cpu.register_service_handle(9, iface);
            cpu.ts_request(TLS, 9, Some(4)).unwrap();
            let reading = f32::from_le_bytes(
                cpu.read_bytes(TLS + 0x20, 4).try_into().unwrap(),
            );
            assert_eq!(reading, expected as f32, "{iface}");
        }
    }

    #[test]
    fn set_sys_reports_a_real_firmware_version_into_its_pointer_buffer() {
        // libnx seeds `hosversionGet` from this, and everything version-gated
        // branches on it. Answering with an empty success left the caller
        // reading its own stale buffer: NX-Fetch showed "Horizon OS
        // 115.119.105", the ASCII of the uid this emulator had left there.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_static(3, &[], BUFFER, 0x100);
        cpu.mem.map_zero(BUFFER, 0x200).unwrap();
        for offset in 0..0x100 {
            cpu.mem.write_u8(BUFFER + offset, b'x').unwrap();
        }
        cpu.set_sys_request(TLS, Some(3)).unwrap();

        let (major, minor, micro) = super::FIRMWARE_VERSION;
        assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), major);
        assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), minor);
        assert_eq!(cpu.mem.read_u8(BUFFER + 2).unwrap(), micro);
        assert_eq!(cpu.read_string(BUFFER + 0x08, 0x20), "NX");
        // The display strings agree with the numbers above them.
        let display = format!("{major}.{minor}.{micro}");
        assert_eq!(cpu.read_string(BUFFER + 0x68, 0x18), display);
        assert!(cpu.read_string(BUFFER + 0x80, 0x80).ends_with(&format!("{display}-1.0")));
    }

    #[test]
    fn ts_reports_the_same_temperature_in_both_units_and_inside_its_range() {
        // Internal (the SoC), then External (the PCB): two sensors, two
        // readings, and the pair of commands that report each one in degrees
        // and in millidegrees have to agree.
        for location in [0u8, 1] {
            let mut cpu = request(false, 1, &[location]);
            cpu.register_service_handle(9, "ts");
            cpu.ts_request(TLS, 9, Some(1)).unwrap();
            let celsius = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;

            write_request(&mut cpu, 3, &[location]);
            cpu.ts_request(TLS, 9, Some(3)).unwrap();
            let milli = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
            assert_eq!(milli, celsius * 1000, "location {location}");

            // And the reading has to sit inside the range the same service
            // reports, or a caller scaling a gauge by it draws off the end.
            write_request(&mut cpu, 0, &[location]);
            cpu.ts_request(TLS, 9, Some(0)).unwrap();
            let low = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
            let high = cpu.mem.read_u32(TLS + 0x24).unwrap() as i32;
            assert!(low <= celsius && celsius <= high, "{celsius} outside {low}..={high}");
        }
    }

    #[test]
    fn bsd_hands_out_descriptors_and_takes_them_back() {
        let (mut cpu, fd) = bsd_socket(1);
        assert!(fd >= 3, "past the standard streams a C library already holds");

        write_request(&mut cpu, 26, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "close");

        // Closing it twice is a bad descriptor, not a second success — a
        // socket table that never forgets anything would report the latter.
        write_request(&mut cpu, 26, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_EBADF));
    }

    #[test]
    fn bsd_fails_where_there_is_no_peer_rather_than_pretending() {
        // Connect: refused, at once. A title that checks for an update has to
        // find out now — there is no other thread here to run while it blocks.
        let (mut cpu, fd) = bsd_socket(1);
        write_request(&mut cpu, 14, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ECONNREFUSED));

        // A stream socket has no connection to send on...
        write_request(&mut cpu, 10, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ENOTCONN));

        // ...and a datagram socket has nowhere to send to.
        let (mut cpu, fd) = bsd_socket(super::BSD_SOCK_DGRAM);
        write_request(&mut cpu, 11, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(11)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ENETUNREACH));

        // Accept on a socket that never listened is the caller's mistake;
        // after listen it is an idle network, which is EAGAIN.
        let (mut cpu, fd) = bsd_socket(1);
        write_request(&mut cpu, 12, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(12)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_EINVAL));

        write_request(&mut cpu, 18, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(18)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "listen");
        write_request(&mut cpu, 12, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(12)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_EAGAIN));
    }

    #[test]
    fn bsd_socket_options_and_flags_read_back() {
        const BUFFER: u32 = 0x4000;
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();

        // SetSockOpt(fd, level, option) with the value in a send buffer.
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&fd.to_le_bytes());
        payload[4..8].copy_from_slice(&0xFFFFu32.to_le_bytes()); // SOL_SOCKET
        payload[8..].copy_from_slice(&0x0004u32.to_le_bytes()); // SO_REUSEADDR
        write_map_buffer_request(&mut cpu, 21, &payload, BUFFER, 4, true);
        cpu.mem.write_u32(BUFFER, 1).unwrap();
        cpu.bsd_request(TLS, 9, Some(21)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));

        // GetSockOpt hands it back rather than a zero it never stored.
        cpu.mem.write_u32(BUFFER, 0).unwrap();
        write_map_buffer_request(&mut cpu, 17, &payload, BUFFER, 4, false);
        cpu.bsd_request(TLS, 9, Some(17)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        assert_eq!(cpu.mem.read_u32(BUFFER).unwrap(), 1);

        // fcntl's flags word survives verbatim, whichever libc's O_NONBLOCK
        // the guest was built against.
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&fd.to_le_bytes());
        payload[4..8].copy_from_slice(&4u32.to_le_bytes()); // F_SETFL
        payload[8..].copy_from_slice(&0x0800u32.to_le_bytes());
        write_request(&mut cpu, 20, &payload);
        cpu.bsd_request(TLS, 9, Some(20)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));

        payload[4..8].copy_from_slice(&3u32.to_le_bytes()); // F_GETFL
        write_request(&mut cpu, 20, &payload);
        cpu.bsd_request(TLS, 9, Some(20)).unwrap();
        assert_eq!(bsd_result(&cpu), (0x0800, 0));
    }

    #[test]
    fn bsd_get_sock_name_reports_the_address_nifm_does() {
        const BUFFER: u32 = 0x4000;
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        write_map_buffer_request(&mut cpu, 16, &fd.to_le_bytes(), BUFFER, 0x10, false);
        cpu.bsd_request(TLS, 9, Some(16)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        // FreeBSD's sockaddr_in: a length byte and a family byte, then the
        // port and the address in network order.
        assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), 16);
        assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), 2, "AF_INET");
        assert_eq!(cpu.read_bytes(BUFFER + 4, 4), super::NIFM_LOCAL_IP.to_vec());
    }

    #[test]
    fn apm_agrees_with_am_about_the_performance_mode() {
        // `IManager::GetPerformanceMode` and `ICommonStateGetter::
        // GetPerformanceMode` are two routes to the same fact, and a title
        // that gets two answers concludes the mode changed underneath it.
        let mut cpu = request(false, 1, &[]);
        cpu.register_service_handle(9, "apm");
        cpu.apm_request(TLS, 9, Some(1)).unwrap();
        let from_apm = cpu.mem.read_u32(TLS + 0x20).unwrap();

        let mut cpu = request(false, 6, &[]);
        cpu.register_service_handle(9, "am:common-state-getter");
        cpu.applet_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(from_apm, cpu.mem.read_u32(TLS + 0x20).unwrap());
    }

    #[test]
    fn apm_gives_back_the_performance_configuration_it_was_given() {
        // OpenSession, then set a configuration for Boost and read it back.
        let mut cpu = request(false, 0, &[]);
        cpu.register_service_handle(9, "apm");
        cpu.apm_request(TLS, 9, Some(0)).unwrap();
        let session = cpu.mem.read_u32(TLS + 0x0C).unwrap() as u64;
        assert_eq!(cpu.service_name(session), Some("apm:session"));

        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&1u32.to_le_bytes()); // Boost
        payload[4..].copy_from_slice(&0x0002_0003u32.to_le_bytes());
        write_request(&mut cpu, 0, &payload);
        cpu.apm_request(TLS, session, Some(0)).unwrap();

        write_request(&mut cpu, 1, &1u32.to_le_bytes());
        cpu.apm_request(TLS, session, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0x0002_0003);

        // Normal keeps its own, un-set, configuration: the two modes are
        // separate settings.
        write_request(&mut cpu, 1, &0u32.to_le_bytes());
        cpu.apm_request(TLS, session, Some(1)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), super::APM_DEFAULT_CONFIGURATION[0]);
        assert_ne!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "0 is Invalid");
    }

    #[test]
    fn set_get_region_code_reports_usa() {
        let mut cpu = request(false, 4, &[]);
        cpu.set_request(TLS, Some(4)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "SetRegion_USA");
    }
}
