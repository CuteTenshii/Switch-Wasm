//! The networking services: an empty network. Nothing off the console answers,
//! but `bsd` loopback between sockets of this process works.

use std::collections::VecDeque;

use super::netbridge::{NetCommand, NetEvent, NetFailure, StreamId};
use crate::cpu::Cpu;
use crate::Result;

#[derive(Debug)]
pub(crate) struct SslCertificate {
    /// `CaCertificateId`; -1 is `All`.
    id: i32,
    /// `TrustedCertStatus`, passed through as stored.
    status: u32,
    der: Vec<u8>,
}

/// The firmware system data archive holding the certificate store, and the file in it.
const CERT_STORE_DATA_ID: u64 = 0x0100_0000_0000_0800;
const CERT_STORE_PATH: &str = "/ssl_TrustedCerts.bdf";

/// `CaCertificateId_All`.
const CERT_ID_ALL: i32 = -1;

/// `BuiltInCertificateInfo`, one per certificate plus a terminator, ahead of the DER bytes.
const CERT_INFO_SIZE: u32 = 0x18;

/// Parse `ssl_TrustedCerts.bdf`: an `sslT` header with a count, 0x10-byte entries,
/// then DER bodies. Offsets are from the end of the header.
fn parse_cert_store(data: &[u8]) -> Vec<SslCertificate> {
    const MAGIC: u32 = u32::from_le_bytes(*b"sslT");
    const HEADER_SIZE: usize = 8;
    const ENTRY_SIZE: usize = 0x10;
    let read_u32 = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
    };
    if read_u32(0) != Some(MAGIC) {
        return Vec::new();
    }
    let Some(count) = read_u32(4) else {
        return Vec::new();
    };
    let mut certs = Vec::new();
    for index in 0..count as usize {
        let at = HEADER_SIZE + index * ENTRY_SIZE;
        let (Some(id), Some(status), Some(size), Some(offset)) = (
            read_u32(at),
            read_u32(at + 4),
            read_u32(at + 8),
            read_u32(at + 12),
        ) else {
            break;
        };
        let from = HEADER_SIZE + offset as usize;
        let Some(der) = data.get(from..from.saturating_add(size as usize)) else {
            continue;
        };
        certs.push(SslCertificate {
            id: id as i32,
            status,
            der: der.to_vec(),
        });
    }
    certs
}

/// One open `bsd:u` socket. It can only connect to sockets of this process.
#[derive(Debug, Clone)]
pub(crate) struct BsdSocket {
    /// The family is kept for `DuplicateSocket`; the type picks the data path errno.
    pub domain: u32,
    pub kind: u32,
    /// Normalized by [`Cpu::bsd_normalize_bind`]; empty until `bind`.
    pub bound: Vec<u8>,
    /// Stored verbatim so `F_GETFL` returns what the guest wrote.
    pub flags: u32,
    pub listening: bool,
    /// Connections made to this listener and not yet accepted.
    pub incoming: VecDeque<i32>,
    pub peer: Option<i32>,
    pub rx: VecDeque<u8>,
    /// The peer closed or shut down writing: reads return end-of-file, unlike `peer: None`.
    pub peer_closed: bool,
    /// The connection the host carries, for a socket connected outside this console.
    pub stream: Option<Stream>,
}

/// A connection the host carries: where it stands and the peer's `sockaddr`.
#[derive(Debug, Clone)]
pub(crate) struct Stream {
    id: StreamId,
    state: StreamState,
    address: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Connecting,
    Open,
    /// The errno `SO_ERROR` reports once.
    Failed(i32),
}

impl BsdSocket {
    fn new(domain: u32, kind: u32) -> BsdSocket {
        BsdSocket {
            domain,
            kind,
            bound: Vec::new(),
            flags: 0,
            listening: false,
            incoming: VecDeque::new(),
            peer: None,
            rx: VecDeque::new(),
            peer_closed: false,
            stream: None,
        }
    }

    /// Readable: has bytes, a pending connection, or a closed peer (EOF).
    fn readable(&self) -> bool {
        !self.rx.is_empty()
            || !self.incoming.is_empty()
            || self.peer_closed
            || self
                .stream_state()
                .is_some_and(|s| matches!(s, StreamState::Failed(_)))
    }

    fn stream_state(&self) -> Option<StreamState> {
        self.stream.as_ref().map(|stream| stream.state)
    }

    /// Writable: a live connection or any datagram socket; nothing here has a send buffer.
    fn writable(&self) -> bool {
        self.kind == BSD_SOCK_DGRAM
            || (self.peer.is_some() && !self.peer_closed)
            || matches!(
                self.stream_state(),
                Some(StreamState::Open | StreamState::Failed(_))
            )
    }
}

/// The console's address, as reported by `nifm` and for unbound sockets.
const NIFM_LOCAL_IP: [u8; 4] = [192, 168, 1, 100];

/// `sfdnsres` failures: the definitive ones, not try-again, in FreeBSD's positive numbering.
const SFDNSRES_EAI_NONAME: i32 = 8;

const SFDNSRES_HOST_NOT_FOUND: i32 = 1;

/// `bsd` errnos in FreeBSD's numbering (`EAGAIN` is 35).
const BSD_EBADF: i32 = 9;

const BSD_EINVAL: i32 = 22;

const BSD_EAGAIN: i32 = 35;

const BSD_EPIPE: i32 = 32;

const BSD_ENETUNREACH: i32 = 51;

const BSD_ETIMEDOUT: i32 = 60;

const BSD_EHOSTUNREACH: i32 = 65;

const BSD_EINPROGRESS: i32 = 36;

const BSD_EALREADY: i32 = 37;

const BSD_EISCONN: i32 = 56;

const BSD_ENOTCONN: i32 = 57;

const BSD_ECONNREFUSED: i32 = 61;

/// `SOCK_DGRAM`.
const BSD_SOCK_DGRAM: u32 = 2;

/// `AF_INET`, in the `sin_family` byte of Horizon's `sockaddr`.
const BSD_AF_INET: u8 = 2;

/// `sizeof(sockaddr_in)`, also its `sin_len`.
const BSD_SOCKADDR_IN_LEN: usize = 16;

/// `EAI_SERVICE`.
const SFDNSRES_EAI_SERVICE: i32 = 9;

/// `AI_NUMERICHOST`, `SOCK_STREAM`, `IPPROTO_TCP` and `IPPROTO_UDP` in FreeBSD's numbering.
const AI_NUMERICHOST: u32 = 0x4;
const SOCK_STREAM: u32 = 1;
const IPPROTO_TCP: u32 = 6;
const IPPROTO_UDP: u32 = 17;

/// First word of every serialized `addrinfo`.
const ADDRINFO_MAGIC: u32 = 0xBEEF_CAFE;

/// The fields of a serialized `addrinfo` that `getaddrinfo`'s hints carry.
struct AddrInfoHints {
    flags: u32,
    family: u32,
    socktype: u32,
}

impl AddrInfoHints {
    /// All zero, which asks for anything, when there are no hints or they are malformed.
    fn parse(raw: &[u8]) -> AddrInfoHints {
        let word = |index: usize| {
            raw.get(index * 4..index * 4 + 4)
                .map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        };
        if word(0) != ADDRINFO_MAGIC {
            return AddrInfoHints {
                flags: 0,
                family: 0,
                socktype: 0,
            };
        }
        AddrInfoHints {
            flags: word(1),
            family: word(2),
            socktype: word(3),
        }
    }
}

/// A numeric port, or one of the few service names a title passes; `None` for any other.
fn service_port(service: &str) -> Option<u16> {
    match service {
        "" => Some(0),
        "http" => Some(80),
        "https" => Some(443),
        _ => service.parse().ok(),
    }
}

/// A DNS name the host could resolve: dot-separated labels of letters, digits, `-` and `_`.
fn is_hostname(name: &str) -> bool {
    let name = name.strip_suffix('.').unwrap_or(name);
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(|label| {
            (1..=63).contains(&label.len())
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        })
}

/// Append one `addrinfo` entry to the `sfdnsres` wire format: big-endian words, then a
/// `sockaddr_in` of family (16 bits), port, address and 8 zero bytes, then an empty canonical name.
fn serialize_addrinfo(out: &mut Vec<u8>, socktype: u32, protocol: u32, ip: [u8; 4], port: u16) {
    for word in [
        ADDRINFO_MAGIC,
        0,
        u32::from(BSD_AF_INET),
        socktype,
        protocol,
        BSD_SOCKADDR_IN_LEN as u32,
    ] {
        out.extend_from_slice(&word.to_be_bytes());
    }
    out.extend_from_slice(&u16::from(BSD_AF_INET).to_be_bytes());
    out.extend_from_slice(&port.to_be_bytes());
    out.extend_from_slice(&ip);
    out.extend_from_slice(&[0; 8]);
    out.push(0);
}

/// Start of the ephemeral port range for `bind` to port 0.
pub(crate) const BSD_FIRST_EPHEMERAL_PORT: u16 = 49152;

const BSD_LOOPBACK_IP: [u8; 4] = [127, 0, 0, 1];

const BSD_ANY_IP: [u8; 4] = [0, 0, 0, 0];

/// `FD_SETSIZE`: the 128-byte bitmap the caller marshals.
const BSD_MAX_SELECT_FDS: u32 = 1024;

/// Longest an empty `select` or `poll` parks, so asio's loopback wakeups still land.
const BSD_WAIT_SLICE_NS: i64 = 1_000_000;

/// The `(address, port)` of an `AF_INET` `sockaddr_in`, or `None`.
fn sockaddr_in(raw: &[u8]) -> Option<([u8; 4], u16)> {
    if raw.len() < 8 || raw[1] != BSD_AF_INET {
        return None;
    }
    let port = u16::from_be_bytes([raw[2], raw[3]]);
    Some(([raw[4], raw[5], raw[6], raw[7]], port))
}

/// A FreeBSD `sockaddr_in`: length byte, family byte, then port and address in network order.
fn sockaddr_in_bytes(ip: [u8; 4], port: u16) -> Vec<u8> {
    let mut raw = vec![0u8; BSD_SOCKADDR_IN_LEN];
    raw[0] = BSD_SOCKADDR_IN_LEN as u8;
    raw[1] = BSD_AF_INET;
    raw[2..4].copy_from_slice(&port.to_be_bytes());
    raw[4..8].copy_from_slice(&ip);
    raw
}

/// Whether an address names this console.
fn is_local_ip(ip: [u8; 4]) -> bool {
    ip == BSD_LOOPBACK_IP || ip == BSD_ANY_IP || ip == NIFM_LOCAL_IP
}

/// `FIONBIO`, `F_GETFL`/`F_SETFL`, and FreeBSD's `O_NONBLOCK`.
const BSD_FIONBIO: u32 = 0x8004_667E;

const BSD_F_GETFL: u32 = 3;

const BSD_F_SETFL: u32 = 4;

const BSD_O_NONBLOCK: u32 = 0x0004;

/// `SOL_SOCKET` and `SO_ERROR`.
const BSD_SOL_SOCKET: u32 = 0xFFFF;

const BSD_SO_ERROR: u32 = 0x1007;

impl Cpu {
    /// `ssl`: contexts and options are real objects; connections never connect.
    pub(crate) fn ssl_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "ssl:service");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "ssl:control", cmd_id),
            };
        }
        let object_id = self.ipc_domain_object_id(tls);
        let iface = if self.ipc_is_domain_request(tls) {
            self.domain_interface(handle, object_id)
                .unwrap_or("ssl:service")
                .to_string()
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
                    self.process_mut().ssl_contexts += 1;
                    self.reply_with_interface(tls, handle, "ssl:context")?;
                    Ok(())
                }
                // GetContextCount.
                Some(1) => {
                    let count = self.process().ssl_contexts;
                    self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
                }
                // SetInterfaceVersion(u32).
                Some(5) => {
                    self.process_mut().ssl_interface_version = self.mem.read_u32(data)?;
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                // GetCertificateBufSize(ids in a buffer) -> u32 size.
                Some(3) => {
                    let ids = self.ssl_requested_ids(tls);
                    let (size, _) = self.ssl_certificate_extent(&ids);
                    self.write_ipc_response(tls, 0, &[], &size.to_le_bytes(), &[])
                }
                // GetCertificates(ids) -> u32 count, with records, a terminator and the DER
                // bodies in an out buffer.
                Some(2) => {
                    let ids = self.ssl_requested_ids(tls);
                    let count = self.ssl_write_certificates(tls, &ids)?;
                    self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
                }
                // FlushSessionCache.
                Some(6) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            "ssl:context" => match cmd_id {
                // Set/GetOption(SslContextOption, s32), stored per context.
                Some(0) => {
                    let option = self.mem.read_u32(data)?;
                    let value = self.mem.read_u32(data.wrapping_add(4))?;
                    let key = Self::object_key(handle, object_id);
                    self.process_mut().ssl_options.insert((key, option), value);
                    self.write_ipc_response(tls, 0, &[], &[], &[])
                }
                Some(1) => {
                    let option = self.mem.read_u32(data)?;
                    let key = Self::object_key(handle, object_id);
                    let value = self
                        .process()
                        .ssl_options
                        .get(&(key, option))
                        .copied()
                        .unwrap_or(0);
                    self.write_ipc_response(tls, 0, &[], &value.to_le_bytes(), &[])
                }
                // GetConnectionCount: none.
                Some(3) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
                // ImportServerPki / ImportClientPki(format, certificates) -> u64 id. Accepted, not kept.
                Some(4) | Some(5) => {
                    let id = self.ssl_next_pki_id;
                    self.ssl_next_pki_id = id.wrapping_add(1);
                    self.write_ipc_response(tls, 0, &[], &id.to_le_bytes(), &[])
                }
                // RemoveServerPki / RemoveClientPki(id).
                Some(6) | Some(7) => self.write_ipc_response(tls, 0, &[], &[], &[]),
                _ => self.unimplemented_command(tls, &iface, cmd_id),
            },
            _ => self.unimplemented_command(tls, &iface, cmd_id),
        }
    }

    /// The `CaCertificateId`s a request named. A single `All`, or no buffer, means every root.
    fn ssl_requested_ids(&mut self, tls: u32) -> Vec<i32> {
        let Some((addr, len)) = self.ipc_input_buffer(tls, 0) else {
            return vec![CERT_ID_ALL];
        };
        let mut ids = Vec::new();
        for index in 0..len / 4 {
            match self.mem.read_u32(addr.wrapping_add(index * 4)) {
                Ok(id) => ids.push(id as i32),
                Err(_) => break,
            }
        }
        if ids.is_empty() {
            ids.push(CERT_ID_ALL);
        }
        ids
    }

    fn ssl_certificate_store(&mut self) -> &[SslCertificate] {
        if self.ssl_certificates.is_none() {
            let mut certs = Vec::new();
            if let Some(src) = self.data_archives.get(&CERT_STORE_DATA_ID) {
                let mut image = vec![0u8; src.len() as usize];
                if src.read_at(0, &mut image).is_ok() {
                    if let Some(file) = crate::romfs::RomFs::parse(&image)
                        .ok()
                        .and_then(|romfs| romfs.read_path(CERT_STORE_PATH))
                    {
                        certs = parse_cert_store(file);
                    }
                }
            }
            if certs.is_empty() {
                self.log(
                    crate::trace::Level::Warn,
                    "[ssl] no certificate store: a browser aborts without one; register a \
                     firmware directory",
                );
            }
            self.ssl_certificates = Some(certs);
        }
        self.ssl_certificates.as_deref().unwrap_or(&[])
    }

    fn ssl_wanted(ids: &[i32], cert: &SslCertificate) -> bool {
        ids == [CERT_ID_ALL] || ids.contains(&cert.id)
    }

    /// The size `GetCertificates` needs for `ids`, and how many it writes. DER bodies are
    /// padded to 4 bytes; the terminator is always counted.
    fn ssl_certificate_extent(&mut self, ids: &[i32]) -> (u32, u32) {
        let mut size = CERT_INFO_SIZE;
        let mut count = 0u32;
        let store = self.ssl_certificate_store();
        for cert in store {
            if !Self::ssl_wanted(ids, cert) {
                continue;
            }
            size += CERT_INFO_SIZE + (cert.der.len() as u32).next_multiple_of(4);
            count += 1;
        }
        (size, count)
    }

    /// Write the records and DER bodies, returning the count. A buffer too small for
    /// [`Cpu::ssl_certificate_extent`] is left alone.
    fn ssl_write_certificates(&mut self, tls: u32, ids: &[i32]) -> Result<u32> {
        let (size, count) = self.ssl_certificate_extent(ids);
        let Some((addr, len)) = self.ipc_output_buffer(tls, 0) else {
            return Ok(0);
        };
        if len < size {
            return Ok(0);
        }
        let mut der_at = (count + 1) * CERT_INFO_SIZE;
        let mut info_at = 0u32;
        let store = std::mem::take(&mut self.ssl_certificates).unwrap_or_default();
        for cert in &store {
            if !Self::ssl_wanted(ids, cert) {
                continue;
            }
            let mut info = [0u8; CERT_INFO_SIZE as usize];
            info[0..4].copy_from_slice(&cert.id.to_le_bytes());
            info[4..8].copy_from_slice(&cert.status.to_le_bytes());
            info[8..16].copy_from_slice(&(cert.der.len() as u64).to_le_bytes());
            info[16..24].copy_from_slice(&u64::from(der_at).to_le_bytes());
            for (index, &byte) in info.iter().enumerate() {
                self.mem
                    .write_u8(addr.wrapping_add(info_at + index as u32), byte)?;
            }
            for (index, &byte) in cert.der.iter().enumerate() {
                self.mem
                    .write_u8(addr.wrapping_add(der_at + index as u32), byte)?;
            }
            info_at += CERT_INFO_SIZE;
            der_at += (cert.der.len() as u32).next_multiple_of(4);
        }
        self.ssl_certificates = Some(store);
        // The terminator: `CaCertificateId_All` with an empty body.
        let mut end = [0u8; CERT_INFO_SIZE as usize];
        end[0..4].copy_from_slice(&CERT_ID_ALL.to_le_bytes());
        for (index, &byte) in end.iter().enumerate() {
            self.mem
                .write_u8(addr.wrapping_add(info_at + index as u32), byte)?;
        }
        Ok(count)
    }

    /// `sfdnsres` (`IResolver`): nothing resolves. Every lookup fails definitively,
    /// `EAI_NONAME` or `HOST_NOT_FOUND`, including numeric addresses.
    pub(crate) fn sfdnsres_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // GetHostByNameRequest / GetHostByAddrRequest and WithOptions forms (`h_errno`).
            Some(2) | Some(3) | Some(10) | Some(11) => {
                self.sfdnsres_failure(tls, SFDNSRES_HOST_NOT_FOUND)
            }
            // GetAddrInfoRequest and its WithOptions form, once the host carries traffic.
            Some(6) | Some(12) if self.net.enabled => self.sfdnsres_addrinfo(tls),
            // GetAddrInfoRequest / GetNameInfoRequest and WithOptions forms (`gai` error).
            Some(6) | Some(7) | Some(12) | Some(13) => {
                self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME)
            }
            // GetHostStringErrorRequest / GetGaiStringErrorRequest.
            Some(4) | Some(5) => {
                let message: &[u8] = b"Name or service not known\0";
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if addr != 0 {
                        for (index, &byte) in message.iter().take(size as usize).enumerate() {
                            self.mem.write_u8(addr.wrapping_add(index as u32), byte)?;
                        }
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // RequestCancelHandleRequest -> u32 token.
            Some(8) => {
                let handle = self.next_object_id;
                self.next_object_id = handle.wrapping_add(1);
                self.write_ipc_response(tls, 0, &[], &handle.to_le_bytes(), &[])
            }
            // CancelRequest, and the resolver options.
            Some(9) | Some(14) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(15) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "sfdnsres", cmd_id),
        }
    }

    /// `nsd:a`/`nsd:u`: name resolution for the production environment, `lp1`.
    pub(crate) fn nsd_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const FQDN_SIZE: u32 = 0x100;
        if self.ipc_answer_control(tls, handle, "nsd", cmd_id)? {
            return Ok(());
        }
        match cmd_id {
            // Resolve / ResolveEx(buffer FQDN in, buffer FQDN out); Ex adds an inner result.
            Some(20) | Some(21) => {
                let fqdn = self
                    .ipc_input_buffer(tls, 0)
                    .map(|(addr, size)| self.read_string(addr, size.min(FQDN_SIZE)))
                    .unwrap_or_default();
                let mut resolved = fqdn.replace('%', "lp1").into_bytes();
                resolved.resize(FQDN_SIZE as usize, 0);
                self.write_output_buffer(tls, 0, &resolved);
                let data: &[u8] = if cmd_id == Some(21) { &[0; 4] } else { &[] };
                self.write_ipc_response(tls, 0, &[], data, &[])
            }
            // SetChangeEnvironmentIdentifierDisabled: no environment here changes.
            Some(15) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            _ => self.unimplemented_command(tls, "nsd", cmd_id),
        }
    }

    /// `GetAddrInfoRequest` with a network: a literal address answers as itself, a name as
    /// its synthetic address, which `bsd` maps back to the name when the guest connects.
    fn sfdnsres_addrinfo(&mut self, tls: u32) -> Result<()> {
        let read = |cpu: &Cpu, index: u32| {
            cpu.ipc_input_buffer(tls, index)
                .map(|(addr, size)| cpu.read_bytes(addr, size))
                .unwrap_or_default()
        };
        let text = |bytes: Vec<u8>| {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            String::from_utf8_lossy(&bytes[..end]).into_owned()
        };
        let host = text(read(self, 0));
        let service = text(read(self, 1));
        let hints = AddrInfoHints::parse(&read(self, 2));

        let Some(port) = service_port(&service) else {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_SERVICE);
        };
        if hints.family != 0 && hints.family != u32::from(BSD_AF_INET) {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME);
        }
        let literal = host
            .parse::<std::net::Ipv4Addr>()
            .ok()
            .map(|ip| ip.octets());
        let address = match literal {
            Some(address) => Some(address),
            None if hints.flags & AI_NUMERICHOST != 0 || !is_hostname(&host) => None,
            None => self.net.synthesize(&host),
        };
        let Some(address) = address else {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME);
        };

        let mut serialized = Vec::new();
        for (socktype, protocol) in [(SOCK_STREAM, IPPROTO_TCP), (BSD_SOCK_DGRAM, IPPROTO_UDP)] {
            if hints.socktype == 0 || hints.socktype == socktype {
                serialize_addrinfo(&mut serialized, socktype, protocol, address, port);
            }
        }
        serialized.extend_from_slice(&0u32.to_be_bytes());
        let fits = self
            .ipc_output_buffer(tls, 0)
            .is_some_and(|(_, size)| serialized.len() <= size as usize);
        if !fits {
            return self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME);
        }
        self.write_output_buffer(tls, 0, &serialized);
        let mut results = [0u8; 12];
        results[8..].copy_from_slice(&(serialized.len() as u32).to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }

    /// A failed lookup. `SfdnsresRequestResults` is { return value, errno, bytes
    /// written }; the error goes in the first word, errno stays 0.
    fn sfdnsres_failure(&mut self, tls: u32, error: i32) -> Result<()> {
        let mut results = [0u8; 12];
        results[..4].copy_from_slice(&error.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }

    /// `bsd:u`/`bsd:s`, the socket service. The only peer is this console: loopback
    /// connections work, everything else is refused immediately. Errnos are FreeBSD's.
    pub(crate) fn bsd_request(&mut self, tls: u32, handle: u64, cmd_id: Option<u32>) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, "bsd:u");
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
                }
                _ => self.unimplemented_command(tls, "bsd:control", cmd_id),
            };
        }
        let data = self.ipc_request_data(tls);
        let word =
            |cpu: &Cpu, index: u32| cpu.mem.read_u32(data.wrapping_add(index * 4)).unwrap_or(0);
        match cmd_id {
            // RegisterClient(BsdInitConfig, pid, tmem_size, tmem) -> u64.
            Some(0) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // StartMonitoring(pid).
            Some(1) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // Socket(domain, type, protocol) / SocketExempt. The family is not validated.
            Some(2) | Some(3) => {
                let socket = BsdSocket::new(word(self, 0), word(self, 1));
                let fd = self.alloc_bsd_fd();
                self.process_mut().bsd_sockets.insert(fd, socket);
                self.bsd_reply(tls, fd, 0)
            }
            // Select(nfds, timeval timeout, bool null_timeout), sets in buffers 0, 1 and 2.
            // The out-sets are always written.
            Some(5) => {
                let nfds = word(self, 0).min(BSD_MAX_SELECT_FDS);
                let seconds = self.mem.read_u64(data.wrapping_add(8)).unwrap_or(0);
                let micros = self.mem.read_u64(data.wrapping_add(16)).unwrap_or(0);
                let timeout = if self.mem.read_u8(data.wrapping_add(24)).unwrap_or(0) != 0 {
                    -1
                } else {
                    seconds
                        .saturating_mul(1_000_000_000)
                        .saturating_add(micros.saturating_mul(1_000))
                        .min(i64::MAX as u64) as i64
                };
                let mut ready = 0;
                for index in 0..3 {
                    ready += self.bsd_select_set(tls, index, nfds)?;
                }
                if ready == 0 {
                    self.bsd_park(timeout);
                }
                self.bsd_reply(tls, ready, 0)
            }
            // Poll(nfds, timeout): copies the fds to the output buffer with fresh `revents`.
            // A wait that finds nothing parks, since threads only switch at blocking syscalls.
            Some(6) => {
                let timeout = word(self, 1) as i32;
                let mut ready = 0;
                if let (Some((src, src_size)), Some((dst, dst_size))) = (
                    self.ipc_input_buffer(tls, 0),
                    self.ipc_output_buffer(tls, 0),
                ) {
                    // struct pollfd { s32 fd; s16 events; s16 revents; }
                    for offset in (0..src_size.min(dst_size)).step_by(8) {
                        let fd = self.mem.read_u32(src.wrapping_add(offset)).unwrap_or(0);
                        let events = self.mem.read_u16(src.wrapping_add(offset + 4)).unwrap_or(0);
                        let revents = self.bsd_poll_revents(fd as i32, events);
                        ready += u32::from(revents != 0);
                        self.mem.write_u32(dst.wrapping_add(offset), fd)?;
                        self.mem.write_u16(dst.wrapping_add(offset + 4), events)?;
                        self.mem.write_u16(dst.wrapping_add(offset + 6), revents)?;
                    }
                }
                if ready == 0 {
                    self.bsd_park(i64::from(timeout).saturating_mul(1_000_000));
                }
                self.bsd_reply(tls, ready as i32, 0)
            }
            // Recv(fd, flags) / Read(fd).
            Some(8) | Some(25) => {
                let fd = word(self, 0) as i32;
                self.bsd_receive(tls, fd, None)
            }
            // RecvFrom(fd, flags), with the sender's address in the second buffer.
            Some(9) => {
                let fd = word(self, 0) as i32;
                self.bsd_receive(tls, fd, Some(1))
            }
            // Send(fd, flags) / SendTo(fd, flags, sockaddr) / Write(fd). See [`Cpu::bsd_send`].
            Some(10) | Some(24) => {
                let fd = word(self, 0) as i32;
                self.bsd_send(tls, fd, None)
            }
            Some(11) => {
                let fd = word(self, 0) as i32;
                self.bsd_send(tls, fd, Some(1))
            }
            // Accept(fd): head of the listener's queue, address in the output buffer and its
            // length in the third reply word. EAGAIN when empty.
            Some(12) => {
                let fd = word(self, 0) as i32;
                let accepted = match self.process_mut().bsd_sockets.get_mut(&fd) {
                    None => return self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if !socket.listening => {
                        return self.bsd_reply(tls, -1, BSD_EINVAL)
                    }
                    Some(socket) => socket.incoming.pop_front(),
                };
                let Some(accepted) = accepted else {
                    return self.bsd_reply(tls, -1, BSD_EAGAIN);
                };
                let address = self.bsd_peer_address(accepted);
                let written = self.bsd_write_address(tls, 0, &address)?;
                self.bsd_reply_len(tls, accepted, 0, written)
            }
            // Bind(fd, sockaddr): kept, normalized, for `GetSockName`.
            Some(13) => {
                let address = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(0x80)),
                    None => Vec::new(),
                };
                let fd = word(self, 0) as i32;
                if !self.process().bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let address = self.bsd_normalize_bind(address);
                if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
                    socket.bound = address;
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Connect(fd, sockaddr): to a listener of this process, or refused.
            Some(14) => {
                let address = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(0x80)),
                    None => Vec::new(),
                };
                let fd = word(self, 0) as i32;
                self.bsd_connect(tls, fd, &address)
            }
            // GetPeerName(fd).
            Some(15) => {
                let fd = word(self, 0) as i32;
                match self.process().bsd_sockets.get(&fd) {
                    None => return self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket)
                        if socket.peer.is_none()
                            && socket.stream_state() != Some(StreamState::Open) =>
                    {
                        return self.bsd_reply(tls, -1, BSD_ENOTCONN)
                    }
                    Some(_) => {}
                }
                let address = self.bsd_peer_address(fd);
                let written = self.bsd_write_address(tls, 0, &address)?;
                self.bsd_reply_len(tls, 0, 0, written)
            }
            // GetSockName(fd): the bound address, or the console's own. The third reply word
            // (length) is required; nnSdk passes it on to the next call.
            Some(16) => {
                let fd = word(self, 0) as i32;
                let address = match self.process().bsd_sockets.get(&fd) {
                    None => return self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if !socket.bound.is_empty() => socket.bound.clone(),
                    Some(_) => Self::bsd_local_address(),
                };
                let written = self.bsd_write_address(tls, 0, &address)?;
                self.bsd_reply_len(tls, 0, 0, written)
            }
            // GetSockOpt(fd, level, option): stored value in the output buffer, length in the third word.
            Some(17) => {
                let (fd, level, option) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                if !self.process().bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let value = if (level, option) == (BSD_SOL_SOCKET, BSD_SO_ERROR) {
                    self.bsd_take_stream_error(fd)
                } else {
                    self.process()
                        .bsd_socket_options
                        .get(&(fd, level, option))
                        .copied()
                        .unwrap_or(0)
                };
                let mut written = 0;
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if size >= 4 {
                        self.mem.write_u32(addr, value)?;
                        written = 4;
                    }
                }
                self.bsd_reply_len(tls, 0, 0, written)
            }
            // Listen(fd, backlog).
            Some(18) => {
                let fd = word(self, 0) as i32;
                match self.process_mut().bsd_sockets.get_mut(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) => {
                        socket.listening = true;
                        self.bsd_reply(tls, 0, 0)
                    }
                }
            }
            // Ioctl(fd, request, ...): only FIONBIO, folded into the `fcntl` flags word.
            Some(19) => {
                let (fd, request) = (word(self, 0) as i32, word(self, 1));
                let nonblocking = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) if size >= 4 => self.mem.read_u32(addr).unwrap_or(0) != 0,
                    _ => false,
                };
                match self.process_mut().bsd_sockets.get_mut(&fd) {
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
            // Fcntl(fd, cmd, arg): F_GETFL / F_SETFL, flags stored verbatim since `O_NONBLOCK`
            // differs between FreeBSD, newlib and Linux.
            Some(20) => {
                let (fd, command, arg) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                match self.process_mut().bsd_sockets.get_mut(&fd) {
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
                if !self.process().bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let value = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) if size >= 4 => self.mem.read_u32(addr).unwrap_or(0),
                    _ => 0,
                };
                self.process_mut()
                    .bsd_socket_options
                    .insert((fd, level, option), value);
                self.bsd_reply(tls, 0, 0)
            }
            // Shutdown(fd, how): `SHUT_WR` and `SHUT_RDWR` end the peer's reading.
            Some(22) => {
                const SHUT_RD: u32 = 0;
                let (fd, how) = (word(self, 0) as i32, word(self, 1));
                if !self.process().bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                if how != SHUT_RD {
                    self.bsd_orphan_peer(fd);
                }
                self.bsd_reply(tls, 0, 0)
            }
            // ShutdownAllSockets(how).
            Some(23) => {
                let fds: Vec<i32> = self.bsd_descriptors();
                for fd in fds {
                    self.bsd_orphan_peer(fd);
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Close(fd). The peer then reads end-of-file.
            Some(26) => {
                if !self
                    .process()
                    .bsd_sockets
                    .contains_key(&(word(self, 0) as i32))
                {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let fd = word(self, 0) as i32;
                self.bsd_close(fd);
                self.bsd_reply(tls, 0, 0)
            }
            // DuplicateSocket(fd): copies local state (family, address, flags), not the connection.
            Some(27) => {
                let fd = word(self, 0) as i32;
                let Some(socket) = self.process().bsd_sockets.get(&fd) else {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                };
                let mut copy = BsdSocket::new(socket.domain, socket.kind);
                copy.bound = socket.bound.clone();
                copy.flags = socket.flags;
                copy.listening = socket.listening;
                let duplicate = self.alloc_bsd_fd();
                self.process_mut().bsd_sockets.insert(duplicate, copy);
                self.bsd_reply(tls, duplicate, 0)
            }
            _ => self.unimplemented_command(tls, "bsd:u", cmd_id),
        }
    }

    /// Reply `{ s32 ret, s32 errno }`: `ret` is -1 on failure, `errno` 0 on success.
    fn bsd_reply(&mut self, tls: u32, ret: i32, errno: i32) -> Result<()> {
        let mut raw = [0u8; 8];
        raw[..4].copy_from_slice(&ret.to_le_bytes());
        raw[4..].copy_from_slice(&errno.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    /// Park a wait that found nothing; `timeout` in ns, negative for none, zero for a probe.
    fn bsd_park(&mut self, timeout: i64) {
        let wait = if timeout < 0 {
            BSD_WAIT_SLICE_NS
        } else {
            timeout.min(BSD_WAIT_SLICE_NS)
        };
        self.pending_sleep = self.wait_deadline(wait);
    }

    /// Reply with a third word: bytes written into the caller's output buffer.
    fn bsd_reply_len(&mut self, tls: u32, ret: i32, errno: i32, len: u32) -> Result<()> {
        let mut raw = [0u8; 12];
        raw[..4].copy_from_slice(&ret.to_le_bytes());
        raw[4..8].copy_from_slice(&errno.to_le_bytes());
        raw[8..].copy_from_slice(&len.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    /// The next descriptor; monotonic.
    fn alloc_bsd_fd(&mut self) -> i32 {
        let fd = self.process().next_bsd_fd;
        self.process_mut().next_bsd_fd = self.process().next_bsd_fd.wrapping_add(1);
        fd
    }

    /// Every open descriptor, sorted for deterministic runs.
    fn bsd_descriptors(&self) -> Vec<i32> {
        let mut fds: Vec<i32> = self.process().bsd_sockets.keys().copied().collect();
        fds.sort_unstable();
        fds
    }

    /// The `sockaddr_in` reported for an unbound socket: the `nifm` address, port 0.
    fn bsd_local_address() -> Vec<u8> {
        sockaddr_in_bytes(NIFM_LOCAL_IP, 0)
    }

    /// The peer's address, for `GetPeerName`, `Accept` and `RecvFrom`.
    fn bsd_peer_address(&self, fd: i32) -> Vec<u8> {
        if let Some(stream) = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.stream.as_ref())
        {
            return stream.address.clone();
        }
        let peer = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.peer);
        match peer.and_then(|peer| self.process().bsd_sockets.get(&peer)) {
            Some(peer) if !peer.bound.is_empty() => peer.bound.clone(),
            _ => Self::bsd_local_address(),
        }
    }

    /// Put an address in the caller's `index`-th output buffer, returning how much fitted.
    /// No buffer is a length of zero.
    fn bsd_write_address(&mut self, tls: u32, index: u32, address: &[u8]) -> Result<u32> {
        let Some((addr, size)) = self.ipc_output_buffer(tls, index) else {
            return Ok(0);
        };
        if addr == 0 {
            return Ok(0);
        }
        let mut written = 0;
        for (offset, &byte) in address.iter().take(size as usize).enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
            written = offset as u32 + 1;
        }
        Ok(written)
    }

    /// Normalize a `bind` address so `GetSockName` reports something connectable: fix
    /// `sin_len` and assign a port for port 0. Other families pass through.
    fn bsd_normalize_bind(&mut self, address: Vec<u8>) -> Vec<u8> {
        let Some((ip, port)) = sockaddr_in(&address) else {
            return address;
        };
        let port = if port == 0 {
            self.bsd_assign_port()
        } else {
            port
        };
        sockaddr_in_bytes(ip, port)
    }

    /// A port no open socket is bound to, wrapping the ephemeral range once.
    fn bsd_assign_port(&mut self) -> u16 {
        let range = u16::MAX - BSD_FIRST_EPHEMERAL_PORT + 1;
        let mut port = self.next_bsd_port;
        for _ in 0..range {
            let candidate = port;
            port = if candidate == u16::MAX {
                BSD_FIRST_EPHEMERAL_PORT
            } else {
                candidate + 1
            };
            let taken = self
                .process()
                .bsd_sockets
                .values()
                .any(|socket| sockaddr_in(&socket.bound).is_some_and(|(_, p)| p == candidate));
            if !taken {
                self.next_bsd_port = port;
                return candidate;
            }
        }
        self.next_bsd_port = port;
        port
    }

    fn bsd_listener_on(&self, port: u16) -> Option<i32> {
        self.bsd_descriptors().into_iter().find(|fd| {
            self.process().bsd_sockets.get(fd).is_some_and(|socket| {
                socket.listening && sockaddr_in(&socket.bound).is_some_and(|(_, p)| p == port)
            })
        })
    }

    /// `Connect(fd, sockaddr)`. Completes immediately to a listener of this process;
    /// anything else is `ECONNREFUSED`.
    fn bsd_connect(&mut self, tls: u32, fd: i32, address: &[u8]) -> Result<()> {
        let (domain, kind) = match self.process().bsd_sockets.get(&fd) {
            None => return self.bsd_reply(tls, -1, BSD_EBADF),
            Some(socket) if socket.peer.is_some() => return self.bsd_reply(tls, -1, BSD_EISCONN),
            Some(socket) if socket.kind == BSD_SOCK_DGRAM => {
                return self.bsd_reply(tls, -1, BSD_ECONNREFUSED)
            }
            Some(socket) => (socket.domain, socket.kind),
        };
        let target = sockaddr_in(address).filter(|&(ip, port)| is_local_ip(ip) && port != 0);
        let listener = target.and_then(|(_, port)| self.bsd_listener_on(port));
        let Some(listener) = listener else {
            return match sockaddr_in(address) {
                Some((ip, port)) if self.net.enabled && port != 0 => {
                    self.bsd_connect_outside(tls, fd, ip, port, address)
                }
                _ => self.bsd_reply(tls, -1, BSD_ECONNREFUSED),
            };
        };

        // The accepted end answers on the listener's address.
        let mut accepted = BsdSocket::new(domain, kind);
        accepted.bound = self
            .process()
            .bsd_sockets
            .get(&listener)
            .map(|l| l.bound.clone())
            .unwrap_or_default();
        accepted.peer = Some(fd);
        let accepted_fd = self.alloc_bsd_fd();
        self.process_mut().bsd_sockets.insert(accepted_fd, accepted);

        // An unbound client gets an address now, for `GetPeerName` on the accepted end.
        let unbound = self
            .process()
            .bsd_sockets
            .get(&fd)
            .is_some_and(|socket| socket.bound.is_empty());
        let client_address =
            unbound.then(|| sockaddr_in_bytes(BSD_LOOPBACK_IP, self.bsd_assign_port()));
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            if let Some(address) = client_address {
                socket.bound = address;
            }
            socket.peer = Some(accepted_fd);
        }
        if let Some(listener) = self.process_mut().bsd_sockets.get_mut(&listener) {
            listener.incoming.push_back(accepted_fd);
        }
        self.bsd_reply(tls, 0, 0)
    }

    /// `connect` to a peer outside this console: the host does the connecting, so the call
    /// is always in progress. The socket turns writable once it opens, or readable and writable
    /// with the error `SO_ERROR` reports if it fails.
    fn bsd_connect_outside(
        &mut self,
        tls: u32,
        fd: i32,
        ip: [u8; 4],
        port: u16,
        address: &[u8],
    ) -> Result<()> {
        let state = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.stream_state());
        match state {
            Some(StreamState::Connecting) => return self.bsd_reply(tls, -1, BSD_EALREADY),
            Some(StreamState::Open) => return self.bsd_reply(tls, -1, BSD_EISCONN),
            Some(StreamState::Failed(errno)) => {
                self.bsd_clear_stream(fd);
                return self.bsd_reply(tls, -1, errno);
            }
            None => {}
        }
        let id = self.net.open_stream();
        let peer = self.net.peer_for(ip);
        self.net.outbox.push(NetCommand::Connect {
            stream: id,
            udp: false,
            peer,
            port,
        });
        let unbound = self
            .process()
            .bsd_sockets
            .get(&fd)
            .is_some_and(|socket| socket.bound.is_empty());
        let local = unbound.then(|| sockaddr_in_bytes(NIFM_LOCAL_IP, self.bsd_assign_port()));
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            if let Some(local) = local {
                socket.bound = local;
            }
            socket.stream = Some(Stream {
                id,
                state: StreamState::Connecting,
                address: address.to_vec(),
            });
        }
        self.bsd_reply(tls, -1, BSD_EINPROGRESS)
    }

    /// `SO_ERROR`: the failure of a connect, reported once; zero otherwise.
    fn bsd_take_stream_error(&mut self, fd: i32) -> u32 {
        let state = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.stream_state());
        let Some(StreamState::Failed(errno)) = state else {
            return 0;
        };
        self.bsd_clear_stream(fd);
        errno as u32
    }

    /// Forget a stream that failed once its error has been reported.
    fn bsd_clear_stream(&mut self, fd: i32) {
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            socket.stream = None;
        }
    }

    /// `Send`/`SendTo`/`Write`: everything goes into the peer's queue at once.
    fn bsd_send(&mut self, tls: u32, fd: i32, destination: Option<u32>) -> Result<()> {
        let peer = match self.process().bsd_sockets.get(&fd) {
            None => return self.bsd_reply(tls, -1, BSD_EBADF),
            Some(socket) if socket.peer_closed => return self.bsd_reply(tls, -1, BSD_EPIPE),
            Some(socket) if socket.stream.is_some() => return self.bsd_send_outside(tls, fd),
            Some(socket) => match socket.peer {
                Some(peer) => peer,
                None => return self.bsd_send_datagram(tls, fd, destination),
            },
        };
        let bytes = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let sent = bytes.len() as i32;
        if let Some(peer) = self.process_mut().bsd_sockets.get_mut(&peer) {
            peer.rx.extend(bytes);
        }
        self.bsd_reply(tls, sent, 0)
    }

    /// `Send` on a host-carried stream: queued for the host once it is open.
    fn bsd_send_outside(&mut self, tls: u32, fd: i32) -> Result<()> {
        let Some(socket) = self.process().bsd_sockets.get(&fd) else {
            return self.bsd_reply(tls, -1, BSD_EBADF);
        };
        let (Some(StreamState::Open), Some(stream)) = (socket.stream_state(), &socket.stream)
        else {
            return self.bsd_reply(tls, -1, BSD_ENOTCONN);
        };
        let stream = stream.id;
        let data = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let sent = data.len() as i32;
        self.net.outbox.push(NetCommand::Send { stream, data });
        self.bsd_reply(tls, sent, 0)
    }

    /// `SendTo` from a datagram socket with no peer: the link takes it and drops it.
    fn bsd_send_datagram(&mut self, tls: u32, fd: i32, destination: Option<u32>) -> Result<()> {
        let datagram = self
            .process()
            .bsd_sockets
            .get(&fd)
            .is_some_and(|socket| socket.kind == BSD_SOCK_DGRAM);
        let addressed = destination
            .and_then(|index| self.ipc_input_buffer(tls, index))
            .map(|(addr, size)| self.read_bytes(addr, size.min(0x80)))
            .is_some_and(|address| sockaddr_in(&address).is_some());
        if !datagram || !addressed {
            return self.bsd_unconnected(tls, fd);
        }
        let sent = self
            .ipc_input_buffer(tls, 0)
            .map_or(0, |(_, size)| size as i32);
        self.bsd_reply(tls, sent, 0)
    }

    /// `Recv`/`RecvFrom`/`Read`, with the sender's address when `address_buffer` names one.
    fn bsd_receive(&mut self, tls: u32, fd: i32, address_buffer: Option<u32>) -> Result<()> {
        let (ret, errno) = self.bsd_receive_bytes(tls, fd)?;
        let Some(index) = address_buffer else {
            return self.bsd_reply(tls, ret, errno);
        };
        let address = if ret >= 0 {
            self.bsd_peer_address(fd)
        } else {
            Vec::new()
        };
        let written = self.bsd_write_address(tls, index, &address)?;
        self.bsd_reply_len(tls, ret, errno, written)
    }

    /// Drain the peer's bytes into the caller's buffer. An empty live queue is
    /// `EAGAIN` plus a yield.
    fn bsd_receive_bytes(&mut self, tls: u32, fd: i32) -> Result<(i32, i32)> {
        match self.process().bsd_sockets.get(&fd) {
            None => return Ok((-1, BSD_EBADF)),
            // A stream socket needs a connection; a datagram socket just has an empty queue.
            Some(socket)
                if socket.kind != BSD_SOCK_DGRAM
                    && socket.peer.is_none()
                    && !socket.peer_closed
                    && socket.stream_state() != Some(StreamState::Open) =>
            {
                return Ok((-1, BSD_ENOTCONN));
            }
            Some(_) => {}
        }
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return Ok((-1, BSD_EINVAL));
        };
        let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) else {
            return Ok((-1, BSD_EBADF));
        };
        let take = size.min(socket.rx.len() as u32) as usize;
        if take == 0 {
            // End-of-file: a read of zero bytes.
            if socket.peer_closed {
                return Ok((0, 0));
            }
            self.pending_yield = true;
            return Ok((-1, BSD_EAGAIN));
        }
        let bytes: Vec<u8> = socket.rx.drain(..take).collect();
        for (offset, &byte) in bytes.iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
        }
        Ok((take as i32, 0))
    }

    /// The error for a send with no destination.
    fn bsd_unconnected(&mut self, tls: u32, fd: i32) -> Result<()> {
        match self.process().bsd_sockets.get(&fd) {
            None => self.bsd_reply(tls, -1, BSD_EBADF),
            Some(socket) if socket.kind == BSD_SOCK_DGRAM => {
                self.bsd_reply(tls, -1, BSD_ENETUNREACH)
            }
            Some(_) => self.bsd_reply(tls, -1, BSD_ENOTCONN),
        }
    }

    /// One of `select`'s descriptor sets: read it, write back the ready ones, count them.
    /// Descriptor n is bit n of the byte array for both 32- and 64-bit `fd_mask`.
    fn bsd_select_set(&mut self, tls: u32, index: u32, nfds: u32) -> Result<i32> {
        let Some((dst, dst_size)) = self.ipc_output_buffer(tls, index) else {
            return Ok(0);
        };
        let wanted = self
            .ipc_input_buffer(tls, index)
            .map(|(addr, size)| self.read_bytes(addr, size))
            .unwrap_or_default();
        for offset in 0..dst_size {
            self.mem.write_u8(dst.wrapping_add(offset), 0)?;
        }
        let mut ready = 0;
        for fd in 0..nfds {
            let (byte, bit) = ((fd / 8) as usize, 1u8 << (fd % 8));
            if wanted.get(byte).copied().unwrap_or(0) & bit == 0 || byte as u32 >= dst_size {
                continue;
            }
            // Unknown descriptors are reported not ready.
            let is_ready = match self.process().bsd_sockets.get(&(fd as i32)) {
                Some(socket) if index == 0 => socket.readable(),
                Some(socket) if index == 1 => socket.writable(),
                _ => false,
            };
            if !is_ready {
                continue;
            }
            let at = dst.wrapping_add(byte as u32);
            let already = self.mem.read_u8(at)?;
            self.mem.write_u8(at, already | bit)?;
            ready += 1;
        }
        Ok(ready)
    }

    /// The events a `poll` asked about that have happened on `fd`. Unknown descriptors
    /// get no events rather than `POLLNVAL`.
    fn bsd_poll_revents(&self, fd: i32, events: u16) -> u16 {
        const POLLIN: u16 = 0x0001;
        const POLLOUT: u16 = 0x0004;
        const POLLERR: u16 = 0x0008;
        const POLLHUP: u16 = 0x0010;
        let Some(socket) = self.process().bsd_sockets.get(&fd) else {
            return 0;
        };
        let mut revents = 0;
        if events & POLLIN != 0 && socket.readable() {
            revents |= POLLIN;
        }
        if events & POLLOUT != 0 && socket.writable() {
            revents |= POLLOUT;
        }
        if socket.peer_closed && socket.rx.is_empty() {
            revents |= POLLHUP;
        }
        if matches!(socket.stream_state(), Some(StreamState::Failed(_))) {
            revents |= POLLERR;
        }
        revents
    }

    /// Tell `fd`'s peer nothing more is coming. The link is kept for `GetPeerName`.
    fn bsd_orphan_peer(&mut self, fd: i32) {
        if let Some(stream) = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.stream.as_ref())
            .filter(|stream| stream.state == StreamState::Open)
        {
            let stream = stream.id;
            self.net.outbox.push(NetCommand::ShutdownWrite { stream });
        }
        let Some(peer) = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.peer)
        else {
            return;
        };
        if let Some(peer) = self.process_mut().bsd_sockets.get_mut(&peer) {
            peer.peer_closed = true;
        }
    }

    /// Drop a descriptor; a listener takes its unaccepted connections with it.
    fn bsd_close(&mut self, fd: i32) {
        let Some(socket) = self.process_mut().bsd_sockets.remove(&fd) else {
            return;
        };
        self.process_mut()
            .bsd_socket_options
            .retain(|&(owner, _, _), _| owner != fd);
        if let Some(stream) = socket
            .stream
            .as_ref()
            .filter(|s| !matches!(s.state, StreamState::Failed(_)))
        {
            self.net
                .outbox
                .push(NetCommand::Close { stream: stream.id });
        }
        if let Some(peer) = socket
            .peer
            .and_then(|peer| self.process_mut().bsd_sockets.get_mut(&peer))
        {
            peer.peer_closed = true;
        }
        for pending in socket.incoming {
            self.bsd_close(pending);
        }
    }

    /// Let guests reach the outside through the host, or back to an empty network.
    pub fn set_network(&mut self, enabled: bool) {
        self.net.enabled = enabled;
    }

    /// What the guest asked the host to do since the last call.
    pub fn take_net_commands(&mut self) -> Vec<NetCommand> {
        std::mem::take(&mut self.net.outbox)
    }

    /// Apply what the host reports about its streams; one for a socket since closed is dropped.
    pub fn deliver_net_events(&mut self, events: impl IntoIterator<Item = NetEvent>) {
        for event in events {
            let id = match &event {
                NetEvent::Connected { stream }
                | NetEvent::Failed { stream, .. }
                | NetEvent::Data { stream, .. }
                | NetEvent::Closed { stream } => *stream,
            };
            let socket = self
                .machine
                .processes
                .values_mut()
                .flat_map(|process| process.bsd_sockets.values_mut())
                .find(|socket| socket.stream.as_ref().is_some_and(|s| s.id == id));
            let Some(socket) = socket else {
                continue;
            };
            let Some(stream) = socket.stream.as_mut() else {
                continue;
            };
            match event {
                NetEvent::Connected { .. } => stream.state = StreamState::Open,
                NetEvent::Failed { reason, .. } => {
                    stream.state = StreamState::Failed(match reason {
                        NetFailure::Refused => BSD_ECONNREFUSED,
                        NetFailure::Unreachable => BSD_ENETUNREACH,
                        NetFailure::TimedOut => BSD_ETIMEDOUT,
                        NetFailure::NotFound => BSD_EHOSTUNREACH,
                    });
                }
                NetEvent::Data { data, .. } => socket.rx.extend(data),
                NetEvent::Closed { .. } => socket.peer_closed = true,
            }
        }
    }

    /// `nifm`'s root session (`nifm:u`, `nifm:s`, `nifm:a`), handing out `IGeneralService`.
    pub(crate) fn nifm_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    let name = self.service_name(handle).unwrap_or("nifm:u").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
                    self.write_ipc_response(tls, 0, &[], &obj.to_le_bytes(), &[])
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

    /// `IGeneralService`: a Wi-Fi link that is up with internet access.
    pub(crate) fn nifm_general_service_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // GetClientId.
            Some(1) => self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[]),
            // CreateScanRequest / CreateRequest / CreateTemporaryNetworkProfile.
            Some(2) | Some(4) | Some(14) => {
                self.reply_with_interface(tls, handle, "nifm:request")?;
                Ok(())
            }
            // EnumerateNetworkInterfaces / EnumerateNetworkProfiles: zero.
            Some(6) | Some(7) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetCurrentNetworkProfile / GetCurrentAccessPoint (wired, so no access point): the
            // buffer must still be written.
            Some(5) | Some(36) => {
                if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                    for i in 0..len {
                        self.mem.write_u8(addr.wrapping_add(i), 0)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetCurrentIpAddress.
            Some(12) => self.write_ipc_response(tls, 0, &[], &NIFM_LOCAL_IP, &[]),
            // GetCurrentIpConfigInfo -> IpAddressSetting { bool is_automatic; address; subnet;
            // gateway } then a DnsSetting.
            Some(15) => {
                let mut raw = Vec::with_capacity(0x18);
                raw.push(1); // is_automatic
                raw.extend_from_slice(&NIFM_LOCAL_IP);
                raw.extend_from_slice(&[255, 255, 255, 0]);
                raw.extend_from_slice(&[NIFM_LOCAL_IP[0], NIFM_LOCAL_IP[1], NIFM_LOCAL_IP[2], 1]);
                raw.resize(0x18, 0); // the DnsSetting, which resolves nothing
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // SetWirelessCommunicationEnabled / IsWirelessCommunicationEnabled: the setting, whose
            // radios off together are Airplane Mode in the home menu.
            Some(16) => {
                let on = self.ipc_arg_u8(tls, 0) != 0;
                self.store_system_settings(|settings| settings.wireless_lan_enable = on);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(17) => {
                let enabled = u8::from(self.system_settings().wireless_lan_enable);
                self.write_ipc_response(tls, 0, &[], &[enabled], &[])
            }
            // GetInternetConnectionStatus -> { type, wifi strength, status }: Ieee80211, 3, connected.
            Some(18) => self.write_ipc_response(tls, 0, &[], &[1u8, 3u8, 2u8], &[]),
            // IsEthernetCommunicationEnabled.
            Some(20) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // IsAnyInternetRequestAccepted / IsAnyForegroundRequestAccepted.
            Some(21) | Some(22) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            _ => {
                self.warn_no_implementation("nifm:general-service", cmd_id);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// `IRequest`: accepted from the moment it exists; its events start signalled.
    pub(crate) fn nifm_request_object_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // GetRequestState -> NifmRequestState_Accepted.
            Some(0) => self.write_ipc_response(tls, 0, &[], &3u32.to_le_bytes(), &[]),
            // GetSystemEventReadableHandles -> two copy handles: state change and completion.
            Some(2) => {
                let state = self.alloc_event("nifm:request-state", true);
                let done = self.alloc_event("nifm:request-done", true);
                self.signal_event(state);
                self.signal_event(done);
                self.write_ipc_reply(tls, 0, &[state, done], &[], &[], &[])
            }
            // GetRevision.
            Some(20) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetResult, Cancel, Submit, SubmitAndWait and the requirement setters.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::Cpu;
    use crate::kernel::ipc::testing::*;
    use crate::services::netbridge::{NetCommand, NetEvent, NetFailure, Peer};

    #[test]
    fn nsd_resolves_the_environment_placeholder() {
        const IN: u32 = 0x1000;
        const OUT: u32 = 0x1200;
        let mut cpu = Cpu::new();
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(IN, 0x400).unwrap();
        cpu.mem.write_bytes(IN, b"%.nintendo.net\0").unwrap();
        write_buffer_request(&mut cpu, 21, &[], &[(IN, 0x100)], &[(OUT, 0x100)]);
        cpu.register_service_handle(9, "nsd:a");
        cpu.nsd_request(TLS, 9, Some(21)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "inner result");
        assert_eq!(cpu.read_string(OUT, 0x100), "lp1.nintendo.net");
    }

    /// Read a `bsd` command's `{ s32 ret, s32 errno }` reply.
    fn bsd_result(cpu: &Cpu) -> (i32, i32) {
        (
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            cpu.mem.read_u32(TLS + 0x24).unwrap() as i32,
        )
    }

    /// The third reply word: bytes of the output buffer filled.
    fn bsd_result_len(cpu: &Cpu) -> u32 {
        cpu.mem.read_u32(TLS + 0x28).unwrap()
    }

    /// Open a socket of `kind` on a fresh `bsd:u` session.
    fn bsd_socket(kind: u32) -> (Cpu, i32) {
        let mut cpu = request(false, 2, &[]);
        cpu.register_service_handle(9, "bsd:u");
        let fd = open_socket(&mut cpu, kind);
        (cpu, fd)
    }

    fn open_socket(cpu: &mut Cpu, kind: u32) -> i32 {
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&2u32.to_le_bytes()); // AF_INET
        payload[4..8].copy_from_slice(&kind.to_le_bytes());
        write_request(cpu, 2, &payload);
        cpu.bsd_request(TLS, 9, Some(2)).unwrap();
        let (fd, errno) = bsd_result(cpu);
        assert_eq!(errno, 0, "socket");
        fd
    }

    const SCRATCH: u32 = 0x4000;

    /// `127.0.0.1:port` with `sin_len` left at zero, as a memsetting guest writes it.
    fn loopback_sockaddr(port: u16) -> [u8; 16] {
        let mut raw = [0u8; 16];
        raw[1] = 2; // AF_INET
        raw[2..4].copy_from_slice(&port.to_be_bytes());
        raw[4..8].copy_from_slice(&[127, 0, 0, 1]);
        raw
    }

    fn place(cpu: &mut Cpu, at: u32, bytes: &[u8]) {
        for (offset, &byte) in bytes.iter().enumerate() {
            cpu.mem.write_u8(at + offset as u32, byte).unwrap();
        }
    }

    /// asio's `socket_select_interrupter` setup: bind to an ephemeral loopback port,
    /// connect to the port `getsockname` reports, accept.
    fn connected_pair(cpu: &mut Cpu, listener: i32) -> (i32, i32) {
        place(cpu, SCRATCH, &loopback_sockaddr(0));
        write_map_buffer_request(cpu, 13, &listener.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(13)).unwrap();
        assert_eq!(bsd_result(cpu), (0, 0), "bind");

        place(cpu, SCRATCH, &[0u8; 16]);
        write_map_buffer_request(cpu, 16, &listener.to_le_bytes(), SCRATCH, 16, false);
        cpu.bsd_request(TLS, 9, Some(16)).unwrap();
        assert_eq!(bsd_result(cpu), (0, 0), "getsockname");

        write_request(cpu, 18, &listener.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(18)).unwrap();
        assert_eq!(bsd_result(cpu), (0, 0), "listen");

        let client = open_socket(cpu, 1);
        write_map_buffer_request(cpu, 14, &client.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(cpu), (0, 0), "connect");

        write_request(cpu, 12, &listener.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(12)).unwrap();
        let (server, errno) = bsd_result(cpu);
        assert_eq!(errno, 0, "accept");
        (client, server)
    }

    fn send_on(cpu: &mut Cpu, fd: i32, bytes: &[u8]) -> (i32, i32) {
        const AT: u32 = SCRATCH + 0x40;
        place(cpu, AT, bytes);
        write_map_buffer_request(cpu, 10, &fd.to_le_bytes(), AT, bytes.len() as u32, true);
        cpu.bsd_request(TLS, 9, Some(10)).unwrap();
        bsd_result(cpu)
    }

    fn recv_on(cpu: &mut Cpu, fd: i32, len: u32) -> ((i32, i32), Vec<u8>) {
        const AT: u32 = SCRATCH + 0x80;
        place(cpu, AT, &vec![0u8; len as usize]);
        write_map_buffer_request(cpu, 8, &fd.to_le_bytes(), AT, len, false);
        cpu.bsd_request(TLS, 9, Some(8)).unwrap();
        let result = bsd_result(cpu);
        let read = if result.0 > 0 {
            cpu.read_bytes(AT, result.0 as u32)
        } else {
            Vec::new()
        };
        (result, read)
    }

    #[test]
    fn sfdnsres_fails_every_lookup_definitively() {
        // getaddrinfo: EAI_NONAME, not EAI_AGAIN.
        let mut cpu = request(false, 6, &[]);
        cpu.sfdnsres_request(TLS, Some(6)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            super::SFDNSRES_EAI_NONAME
        );
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 0, "errno");
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "serialized size");

        // gethostbyname reports through h_errno, which has its own numbering.
        let mut cpu = request(false, 2, &[]);
        cpu.sfdnsres_request(TLS, Some(2)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            super::SFDNSRES_HOST_NOT_FOUND
        );
    }

    #[test]
    fn sfdnsres_explains_the_failure_it_reports() {
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(5, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.sfdnsres_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.read_string(BUFFER, 0x40), "Name or service not known");
    }

    const LOOKUP_HOST: u32 = 0x5000;
    const LOOKUP_PORT: u32 = 0x5100;
    const LOOKUP_HINTS: u32 = 0x5200;
    const LOOKUP_OUT: u32 = 0x5800;

    /// `getaddrinfo(host, port, hints)` with the network on; the reply words and the output.
    fn lookup(host: &str, port: &str, socktype: Option<u32>) -> ((i32, u32), Vec<u8>) {
        let mut cpu = Cpu::new();
        cpu.net.enabled = true;
        cpu.mem.map_zero(TLS, 0x200).unwrap();
        cpu.mem.map_zero(LOOKUP_HOST, 0x2000).unwrap();
        place(&mut cpu, LOOKUP_HOST, format!("{host}\0").as_bytes());
        place(&mut cpu, LOOKUP_PORT, format!("{port}\0").as_bytes());
        let mut hints = Vec::new();
        if let Some(socktype) = socktype {
            for word in [super::ADDRINFO_MAGIC, 0, 0, socktype, 0, 0] {
                hints.extend_from_slice(&word.to_be_bytes());
            }
        }
        place(&mut cpu, LOOKUP_HINTS, &hints);
        let send = [
            (LOOKUP_HOST, 0x100),
            (LOOKUP_PORT, 0x100),
            (LOOKUP_HINTS, 0x400),
        ];
        write_buffer_request(&mut cpu, 6, &[0; 16], &send, &[(LOOKUP_OUT, 0x1000)]);
        cpu.sfdnsres_request(TLS, Some(6)).unwrap();
        let ret = cpu.mem.read_u32(TLS + 0x20).unwrap() as i32;
        let len = cpu.mem.read_u32(TLS + 0x28).unwrap();
        ((ret, len), cpu.read_bytes(LOOKUP_OUT, len))
    }

    /// The `(socktype, protocol, address, port)` of each entry in a serialized chain.
    fn entries(chain: &[u8]) -> Vec<(u32, u32, [u8; 4], u16)> {
        let word = |at: usize| u32::from_be_bytes(chain[at..at + 4].try_into().unwrap());
        let mut at = 0;
        let mut found = Vec::new();
        while word(at) != 0 {
            assert_eq!(word(at), super::ADDRINFO_MAGIC);
            assert_eq!(word(at + 8), 2, "AF_INET");
            assert_eq!(word(at + 20), 16, "ai_addrlen");
            let addr = at + 24;
            assert_eq!(u16::from_be_bytes([chain[addr], chain[addr + 1]]), 2);
            let port = u16::from_be_bytes([chain[addr + 2], chain[addr + 3]]);
            let ip = chain[addr + 4..addr + 8].try_into().unwrap();
            found.push((word(at + 12), word(at + 16), ip, port));
            assert_eq!(chain[addr + 16], 0, "empty canonical name");
            at = addr + 17;
        }
        assert_eq!(at + 4, chain.len(), "the list ends with its sentinel");
        found
    }

    #[test]
    fn sfdnsres_answers_a_name_with_the_same_synthetic_address_each_time() {
        let ((ret, _), chain) = lookup("Example.com", "443", None);
        assert_eq!(ret, 0);
        let found = entries(&chain);
        assert_eq!(found.len(), 2, "stream and datagram");
        assert_eq!((found[0].0, found[0].1, found[0].3), (1, 6, 443));
        assert_eq!((found[1].0, found[1].1, found[1].3), (2, 17, 443));
        assert_eq!(found[0].2[..2], [198, 18]);
        assert_eq!(found[0].2, found[1].2);
        let (_, again) = lookup("example.com", "443", None);
        assert_eq!(entries(&again)[0].2, found[0].2);
    }

    #[test]
    fn sfdnsres_honours_the_hinted_socket_type_and_literal_addresses() {
        let (_, chain) = lookup("1.2.3.4", "http", Some(1));
        assert_eq!(entries(&chain), vec![(1, 6, [1, 2, 3, 4], 80)]);
        let (_, chain) = lookup("1.2.3.4", "", Some(2));
        assert_eq!(entries(&chain), vec![(2, 17, [1, 2, 3, 4], 0)]);
    }

    #[test]
    fn sfdnsres_refuses_what_is_not_a_name_or_a_service() {
        assert_eq!(lookup("", "80", None).0 .0, super::SFDNSRES_EAI_NONAME);
        assert_eq!(
            lookup("bad name", "80", None).0 .0,
            super::SFDNSRES_EAI_NONAME
        );
        assert_eq!(lookup("a..b", "80", None).0 .0, super::SFDNSRES_EAI_NONAME);
        assert_eq!(
            lookup("example.com", "gopher", None).0 .0,
            super::SFDNSRES_EAI_SERVICE
        );
    }

    /// `connect(fd, ip:port)`.
    fn connect_to(cpu: &mut Cpu, fd: i32, ip: [u8; 4], port: u16) -> (i32, i32) {
        let mut address = loopback_sockaddr(port);
        address[4..8].copy_from_slice(&ip);
        place(cpu, SCRATCH, &address);
        write_map_buffer_request(cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        bsd_result(cpu)
    }

    /// `getsockopt(fd, SOL_SOCKET, SO_ERROR)`.
    fn so_error(cpu: &mut Cpu, fd: i32) -> u32 {
        const AT: u32 = SCRATCH + 0x100;
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&fd.to_le_bytes());
        payload[4..8].copy_from_slice(&super::BSD_SOL_SOCKET.to_le_bytes());
        payload[8..].copy_from_slice(&super::BSD_SO_ERROR.to_le_bytes());
        write_map_buffer_request(cpu, 17, &payload, AT, 4, false);
        cpu.bsd_request(TLS, 9, Some(17)).unwrap();
        cpu.mem.read_u32(AT).unwrap()
    }

    fn bsd_command(cpu: &mut Cpu, command: u32, words: &[u32]) -> (i32, i32) {
        let payload: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        write_request(cpu, command, &payload);
        cpu.bsd_request(TLS, 9, Some(command)).unwrap();
        bsd_result(cpu)
    }

    /// A socket on a console whose guest may reach the outside, and the synthetic address
    /// `sfdnsres` would have given `name`.
    fn networked_socket(name: &str) -> (Cpu, i32, [u8; 4]) {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x400).unwrap();
        cpu.set_network(true);
        let address = cpu.net.synthesize(name).unwrap();
        (cpu, fd, address)
    }

    const POLLIN: u16 = 1;
    const POLLOUT: u16 = 4;

    #[test]
    fn bsd_connect_to_a_name_is_in_progress_until_the_host_opens_it() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EINPROGRESS)
        );
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Connect {
                stream: 1,
                udp: false,
                peer: Peer::Name("example.com".into()),
                port: 443,
            }]
        );
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EALREADY)
        );
        assert_eq!(cpu.bsd_poll_revents(fd, POLLOUT), 0, "not writable yet");
        assert_eq!(send_on(&mut cpu, fd, b"early"), (-1, super::BSD_ENOTCONN));

        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        assert_eq!(cpu.bsd_poll_revents(fd, POLLOUT), POLLOUT);
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EISCONN)
        );
        assert_eq!(so_error(&mut cpu, fd), 0);
    }

    #[test]
    fn bsd_carries_bytes_both_ways_and_ends_when_the_host_does() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        connect_to(&mut cpu, fd, address, 80);
        cpu.take_net_commands();
        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);

        assert_eq!(send_on(&mut cpu, fd, b"GET /"), (5, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Send {
                stream: 1,
                data: b"GET /".to_vec()
            }]
        );

        assert_eq!(recv_on(&mut cpu, fd, 16).0, (-1, super::BSD_EAGAIN));
        assert_eq!(cpu.bsd_poll_revents(fd, POLLIN), 0);
        cpu.deliver_net_events([NetEvent::Data {
            stream: 1,
            data: b"200 OK".to_vec(),
        }]);
        assert_eq!(cpu.bsd_poll_revents(fd, POLLIN), POLLIN);
        assert_eq!(recv_on(&mut cpu, fd, 16), ((6, 0), b"200 OK".to_vec()));

        cpu.deliver_net_events([NetEvent::Closed { stream: 1 }]);
        assert_eq!(recv_on(&mut cpu, fd, 16).0, (0, 0), "end of file");
        assert_eq!(send_on(&mut cpu, fd, b"more"), (-1, super::BSD_EPIPE));
    }

    #[test]
    fn bsd_reports_a_failed_connect_once_through_so_error() {
        let (mut cpu, fd, address) = networked_socket("nowhere.example");
        connect_to(&mut cpu, fd, address, 443);
        cpu.deliver_net_events([NetEvent::Failed {
            stream: 1,
            reason: NetFailure::Refused,
        }]);
        assert_eq!(
            cpu.bsd_poll_revents(fd, POLLIN | POLLOUT),
            POLLIN | POLLOUT | 0x0008
        );
        assert_eq!(so_error(&mut cpu, fd), super::BSD_ECONNREFUSED as u32);
        assert_eq!(so_error(&mut cpu, fd), 0, "reported once");

        // A second connect starts a new stream.
        cpu.take_net_commands();
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EINPROGRESS)
        );
        assert_eq!(cpu.take_net_commands().len(), 1);
    }

    #[test]
    fn bsd_connect_after_a_failure_returns_the_error_itself() {
        let (mut cpu, fd, address) = networked_socket("nowhere.example");
        connect_to(&mut cpu, fd, address, 443);
        cpu.deliver_net_events([NetEvent::Failed {
            stream: 1,
            reason: NetFailure::TimedOut,
        }]);
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_ETIMEDOUT)
        );
    }

    #[test]
    fn bsd_tells_the_host_when_the_guest_stops_sending_or_closes() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        connect_to(&mut cpu, fd, address, 80);
        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        cpu.take_net_commands();

        assert_eq!(bsd_command(&mut cpu, 22, &[fd as u32, 1]), (0, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::ShutdownWrite { stream: 1 }]
        );
        assert_eq!(bsd_command(&mut cpu, 26, &[fd as u32]), (0, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Close { stream: 1 }]
        );

        // Events for the closed socket go nowhere.
        cpu.deliver_net_events([NetEvent::Data {
            stream: 1,
            data: vec![1],
        }]);
    }

    #[test]
    fn bsd_peer_name_of_a_stream_is_what_the_guest_connected_to() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        connect_to(&mut cpu, fd, address, 443);
        assert_eq!(
            bsd_command(&mut cpu, 15, &[fd as u32]),
            (-1, super::BSD_ENOTCONN)
        );
        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        write_map_buffer_request(&mut cpu, 15, &fd.to_le_bytes(), SCRATCH + 0x200, 16, false);
        cpu.bsd_request(TLS, 9, Some(15)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        let raw = cpu.read_bytes(SCRATCH + 0x200, 16);
        assert_eq!(super::sockaddr_in(&raw), Some((address, 443)));
    }

    #[test]
    fn bsd_stays_an_empty_network_until_enabled() {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        let address = cpu.net.synthesize("example.com").unwrap();
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_ECONNREFUSED)
        );
        assert!(cpu.take_net_commands().is_empty());
    }

    #[test]
    fn bsd_hands_out_descriptors_and_takes_them_back() {
        let (mut cpu, fd) = bsd_socket(1);
        assert!(
            fd >= 3,
            "past the standard streams a C library already holds"
        );

        write_request(&mut cpu, 26, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "close");

        write_request(&mut cpu, 26, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_EBADF));
    }

    #[test]
    fn bsd_fails_where_there_is_no_peer_rather_than_pretending() {
        let (mut cpu, fd) = bsd_socket(1);
        write_request(&mut cpu, 14, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ECONNREFUSED));

        // A stream socket has no connection to send on...
        write_request(&mut cpu, 10, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(10)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ENOTCONN));

        // ...and a datagram socket that named no destination has nowhere to send to.
        let (mut cpu, fd) = bsd_socket(super::BSD_SOCK_DGRAM);
        write_request(&mut cpu, 11, &fd.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(11)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ENETUNREACH));

        // Accept before listen is an error; after listen it is EAGAIN.
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
    fn bsd_sends_an_addressed_datagram_the_link_would_carry() {
        // RakNet's `BindShared` sends a test datagram to its own address and fails startup
        // if it does not go out.
        let (mut cpu, fd) = bsd_socket(super::BSD_SOCK_DGRAM);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        const PAYLOAD: u32 = SCRATCH + 0x40;
        const PORT: u16 = 19132;

        place(&mut cpu, SCRATCH, &loopback_sockaddr(PORT));
        write_map_buffer_request(&mut cpu, 13, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(13)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "bind");

        let send_to = |cpu: &mut Cpu, fd: i32, address: &[u8; 16]| {
            place(cpu, PAYLOAD, &[1u8, 2, 3, 4]);
            place(cpu, SCRATCH, address);
            write_buffer_request(
                cpu,
                11,
                &fd.to_le_bytes(),
                &[(PAYLOAD, 4), (SCRATCH, 16)],
                &[],
            );
            cpu.bsd_request(TLS, 9, Some(11)).unwrap();
            bsd_result(cpu)
        };

        assert_eq!(
            send_to(&mut cpu, fd, &loopback_sockaddr(PORT)),
            (4, 0),
            "the address it just bound"
        );

        let mut broadcast = loopback_sockaddr(PORT);
        broadcast[4..8].copy_from_slice(&[255, 255, 255, 255]);
        assert_eq!(send_to(&mut cpu, fd, &broadcast), (4, 0), "broadcast");

        // Nothing arrives: EAGAIN and a yield, not `ENETUNREACH`.
        cpu.pending_yield = false;
        let (result, bytes) = recv_on(&mut cpu, fd, 4);
        assert_eq!(result, (-1, super::BSD_EAGAIN), "nothing yet");
        assert!(bytes.is_empty());
        assert!(cpu.pending_yield, "a read that waits has to reschedule");

        // Datagram sockets poll writable.
        const POLL_IN: u32 = SCRATCH + 0xc0;
        const POLL_OUT: u32 = SCRATCH + 0xd0;
        const POLLOUT: u16 = 0x0004;
        let mut pollfd = [0u8; 8];
        pollfd[..4].copy_from_slice(&fd.to_le_bytes());
        pollfd[4..6].copy_from_slice(&POLLOUT.to_le_bytes());
        place(&mut cpu, POLL_IN, &pollfd);
        place(&mut cpu, POLL_OUT, &[0u8; 8]);
        let mut poll = [0u8; 8];
        poll[..4].copy_from_slice(&1u32.to_le_bytes());
        write_buffer_request(&mut cpu, 6, &poll, &[(POLL_IN, 8)], &[(POLL_OUT, 8)]);
        cpu.bsd_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(bsd_result(&cpu), (1, 0), "ready to send");
        assert_eq!(cpu.mem.read_u16(POLL_OUT + 6).unwrap(), POLLOUT);

        // A destination is not a connection for a stream socket.
        let (mut cpu, stream) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        assert_eq!(
            send_to(&mut cpu, stream, &loopback_sockaddr(PORT)),
            (-1, super::BSD_ENOTCONN)
        );
    }

    #[test]
    fn bsd_socket_options_and_flags_read_back() {
        const BUFFER: u32 = 0x4000;
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();

        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&fd.to_le_bytes());
        payload[4..8].copy_from_slice(&0xFFFFu32.to_le_bytes()); // SOL_SOCKET
        payload[8..].copy_from_slice(&0x0004u32.to_le_bytes()); // SO_REUSEADDR
        write_map_buffer_request(&mut cpu, 21, &payload, BUFFER, 4, true);
        cpu.mem.write_u32(BUFFER, 1).unwrap();
        cpu.bsd_request(TLS, 9, Some(21)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));

        cpu.mem.write_u32(BUFFER, 0).unwrap();
        write_map_buffer_request(&mut cpu, 17, &payload, BUFFER, 4, false);
        cpu.bsd_request(TLS, 9, Some(17)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        assert_eq!(cpu.mem.read_u32(BUFFER).unwrap(), 1);

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
    fn a_poll_with_a_timeout_gives_up_the_cpu() {
        // A poll that waits must yield, or a polling loop starves every other thread.
        let (mut cpu, _fd) = bsd_socket(1);
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&1u32.to_le_bytes()); // nfds
        payload[4..].copy_from_slice(&200i32.to_le_bytes()); // timeout, ms
        write_request(&mut cpu, 6, &payload);
        cpu.bsd_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "no descriptor is ever ready");
        assert_eq!(
            cpu.pending_sleep.take(),
            cpu.wait_deadline(super::BSD_WAIT_SLICE_NS),
            "a poll that waits parks, but only for a slice"
        );

        payload[4..].copy_from_slice(&(-1i32).to_le_bytes());
        write_request(&mut cpu, 6, &payload);
        cpu.bsd_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(
            cpu.pending_sleep.take(),
            cpu.wait_deadline(super::BSD_WAIT_SLICE_NS)
        );

        // A zero timeout is a non-blocking probe.
        payload[4..].copy_from_slice(&0i32.to_le_bytes());
        write_request(&mut cpu, 6, &payload);
        cpu.bsd_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        assert_eq!(cpu.pending_sleep, None);
    }

    #[test]
    fn bsd_bind_assigns_a_port_and_get_sock_name_reports_a_usable_address() {
        // asio binds to port 0 and connects to what `getsockname` reports, so the reply
        // must carry a real port and `sin_len`.
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        place(&mut cpu, SCRATCH, &loopback_sockaddr(0));
        write_map_buffer_request(&mut cpu, 13, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(13)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "bind");

        place(&mut cpu, SCRATCH, &[0u8; 16]);
        write_map_buffer_request(&mut cpu, 16, &fd.to_le_bytes(), SCRATCH, 16, false);
        cpu.bsd_request(TLS, 9, Some(16)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        assert_eq!(
            bsd_result_len(&cpu),
            16,
            "the length of the reported address"
        );

        let reported = cpu.read_bytes(SCRATCH, 16);
        assert_eq!(reported[0], 16, "sin_len");
        assert_eq!(reported[1], 2, "AF_INET");
        assert_eq!(
            &reported[4..8],
            &[127, 0, 0, 1],
            "the address stays the one bound"
        );
        let port = u16::from_be_bytes([reported[2], reported[3]]);
        assert!(
            port >= super::BSD_FIRST_EPHEMERAL_PORT,
            "an ephemeral port, not 0: {port}"
        );

        let other = open_socket(&mut cpu, 1);
        place(&mut cpu, SCRATCH, &loopback_sockaddr(8080));
        write_map_buffer_request(&mut cpu, 13, &other.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(13)).unwrap();
        place(&mut cpu, SCRATCH, &[0u8; 16]);
        write_map_buffer_request(&mut cpu, 16, &other.to_le_bytes(), SCRATCH, 16, false);
        cpu.bsd_request(TLS, 9, Some(16)).unwrap();
        assert_eq!(cpu.read_bytes(SCRATCH, 4)[2..], 8080u16.to_be_bytes());
    }

    #[test]
    fn bsd_builds_the_socket_pair_asio_wakes_its_own_select_with() {
        // asio's `socket_select_interrupter::open_descriptors`, then an interrupt.
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);
        assert_ne!(client, server);

        // `interrupt()` writes one byte...
        assert_eq!(send_on(&mut cpu, client, &[0x7f]), (1, 0), "send");
        // ...and `reset()` drains it at the other end.
        assert_eq!(
            recv_on(&mut cpu, server, 0x20),
            ((1, 0), vec![0x7f]),
            "recv"
        );
        assert_eq!(recv_on(&mut cpu, server, 0x20).0, (-1, super::BSD_EAGAIN));

        assert_eq!(recv_on(&mut cpu, client, 0x20).0, (-1, super::BSD_EAGAIN));

        place(&mut cpu, SCRATCH, &[0u8; 16]);
        write_map_buffer_request(&mut cpu, 15, &server.to_le_bytes(), SCRATCH, 16, false);
        cpu.bsd_request(TLS, 9, Some(15)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "getpeername");
        assert_eq!(bsd_result_len(&cpu), 16);
        assert_eq!(cpu.read_bytes(SCRATCH, 2), vec![16, 2]);
    }

    #[test]
    fn bsd_select_names_the_descriptor_that_has_a_byte() {
        const SET: u32 = 0x80;
        const SETS: u32 = 0x6000;
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        cpu.mem.map_zero(SETS, 6 * SET as usize).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);

        let sets: Vec<(u32, u32)> = (0..6).map(|index| (SETS + index * SET, SET)).collect();
        let select = |cpu: &mut Cpu, watch: i32, micros: u64| {
            for offset in 0..6 * SET {
                cpu.mem.write_u8(SETS + offset, 0).unwrap();
            }
            let bit = 1u8 << (watch % 8);
            cpu.mem.write_u8(SETS + (watch as u32 / 8), bit).unwrap();
            let mut payload = [0u8; 32];
            payload[..4].copy_from_slice(&((watch + 1) as u32).to_le_bytes());
            payload[16..24].copy_from_slice(&micros.to_le_bytes());
            write_buffer_request(cpu, 5, &payload, &sets[..3], &sets[3..]);
            cpu.pending_sleep = None;
            cpu.bsd_request(TLS, 9, Some(5)).unwrap();
            let ready = bsd_result(cpu);
            let out = cpu
                .mem
                .read_u8(SETS + 3 * SET + (watch as u32 / 8))
                .unwrap();
            (ready, out & bit != 0)
        };

        // Nothing sent, so nothing ready, and the wait parks for its timeout.
        assert_eq!(select(&mut cpu, server, 300), ((0, 0), false));
        assert_eq!(cpu.pending_sleep, cpu.wait_deadline(300_000));
        // A long wait parks only for a slice.
        assert_eq!(select(&mut cpu, server, 1_000_000), ((0, 0), false));
        assert_eq!(
            cpu.pending_sleep,
            cpu.wait_deadline(super::BSD_WAIT_SLICE_NS)
        );

        assert_eq!(send_on(&mut cpu, client, &[0x7f]), (1, 0));
        // Readiness is reported in the output set.
        assert_eq!(select(&mut cpu, server, 1_000_000), ((1, 0), true));
        assert_eq!(cpu.pending_sleep, None, "nothing to wait for");
    }

    #[test]
    fn a_peer_that_closed_is_end_of_file_and_not_a_connection() {
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);

        assert_eq!(send_on(&mut cpu, client, &[1, 2, 3]), (3, 0));
        write_request(&mut cpu, 26, &client.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "close");
        assert_eq!(recv_on(&mut cpu, server, 0x20), ((3, 0), vec![1, 2, 3]));

        // Then end-of-file.
        assert_eq!(recv_on(&mut cpu, server, 0x20).0, (0, 0));
        assert_eq!(send_on(&mut cpu, server, &[4]), (-1, super::BSD_EPIPE));
    }

    #[test]
    fn bsd_connects_to_this_console_and_refuses_everywhere_else() {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();

        // A loopback port nothing listens on is refused.
        place(&mut cpu, SCRATCH, &loopback_sockaddr(9999));
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(
            bsd_result(&cpu),
            (-1, super::BSD_ECONNREFUSED),
            "nothing is listening"
        );

        // And an address off this console has no route at all.
        let mut remote = loopback_sockaddr(53);
        remote[4..8].copy_from_slice(&[8, 8, 8, 8]);
        place(&mut cpu, SCRATCH, &remote);
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(
            bsd_result(&cpu),
            (-1, super::BSD_ECONNREFUSED),
            "off the console"
        );

        let listener = open_socket(&mut cpu, 1);
        let (client, _server) = connected_pair(&mut cpu, listener);
        write_map_buffer_request(&mut cpu, 14, &client.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_EISCONN));
    }

    #[test]
    fn bsd_get_sock_name_reports_the_address_nifm_does() {
        const BUFFER: u32 = 0x4000;
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        write_map_buffer_request(&mut cpu, 16, &fd.to_le_bytes(), BUFFER, 0x10, false);
        cpu.bsd_request(TLS, 9, Some(16)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        // FreeBSD's sockaddr_in: length, family, then port and address in network order.
        assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), 16);
        assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), 2, "AF_INET");
        assert_eq!(cpu.read_bytes(BUFFER + 4, 4), super::NIFM_LOCAL_IP.to_vec());
    }
}
