//! The networking services, and the answer they all give.
//!
//! There is no network behind any of this. `sfdnsres` resolves nothing; `nifm`
//! reports a console that is connected to a LAN with no route off it; `ssl`
//! builds contexts that never handshake; a `bsd` socket aimed at any address
//! but this console's own is refused.
//!
//! That is deliberate: an *empty* network is a state a real console reaches
//! and every caller has a path for. A failure is the path built for hardware
//! that broke.
//!
//! **Loopback is not the network.** A console with its cable out still talks
//! to itself, and enough middleware assumes it that "no network" and "no
//! loopback" are not the same offer: asio builds one socket pair per
//! `io_context` — bind a listener to `127.0.0.1:0`, connect to the port
//! `getsockname` reports, accept the other end — purely so it can wake its own
//! `select`. Nothing crosses a wire. Asphalt 9 asserts three times over that
//! pair before it draws anything, so this service connects, accepts and
//! carries bytes between two sockets of *this* process, and refuses everything
//! aimed anywhere else.

use std::collections::VecDeque;

use super::Cpu;
use crate::Result;

/// One open `bsd:u` socket.
///
/// A socket here can be created, configured, bound and listened on, and can
/// be connected to another socket of this same process — see
/// [`Cpu::bsd_request`]. It can never reach anything off the console.
#[derive(Debug, Clone)]
pub(crate) struct BsdSocket {
    /// The address family and socket type it was created with. The family is
    /// carried for `DuplicateSocket`; the type decides which "went nowhere"
    /// errno the data path reports.
    pub domain: u32,
    pub kind: u32,
    /// The `sockaddr` this socket answers to, normalized — see
    /// [`Cpu::bsd_normalize_bind`]. Empty until `bind`, and reported by
    /// `GetSockName`.
    pub bound: Vec<u8>,
    /// The flags word `fcntl(F_SETFL)` set, stored verbatim so `F_GETFL` hands
    /// back exactly what the guest wrote.
    pub flags: u32,
    /// Whether `listen` was called — an `accept` on a socket that never
    /// listened is a different error from one nobody has connected to.
    pub listening: bool,
    /// Connections that have been made to this listener and not yet accepted.
    /// A `connect` completes the moment it is issued, so the queue is what
    /// `accept` drains rather than a backlog anything waits in.
    pub incoming: VecDeque<i32>,
    /// The descriptor at the other end of this connection.
    pub peer: Option<i32>,
    /// Bytes the peer has sent and this socket has not read yet.
    pub rx: VecDeque<u8>,
    /// Whether the peer has closed or shut down its writing half. Distinct
    /// from `peer: None`: a socket that was never connected reports
    /// `ENOTCONN`, while one whose peer left reads end-of-file.
    pub peer_closed: bool,
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
        }
    }

    /// Whether a `select` or `poll` would call this descriptor readable: it
    /// has bytes, it has a connection waiting to be accepted, or its peer has
    /// gone and the read that reports end-of-file will not block.
    fn readable(&self) -> bool {
        !self.rx.is_empty() || !self.incoming.is_empty() || self.peer_closed
    }

    /// Whether a `select` or `poll` would call it writable. Nothing here has a
    /// send buffer that can fill, so a live connection always is.
    fn writable(&self) -> bool {
        self.peer.is_some() && !self.peer_closed
    }
}

/// The address `nifm` reports for the console's wired link, and the one
/// `bsd` reports for a socket that was never bound.
const NIFM_LOCAL_IP: [u8; 4] = [192, 168, 1, 100];

/// `sfdnsres` failures. `EAI_NONAME` ("name or service not known") is the
/// `getaddrinfo` family's, `HOST_NOT_FOUND` the `gethostbyname` family's, and
/// both are the **definitive** failure rather than the try-again one: a caller
/// told to retry retries, and there is no other thread here to run while it
/// does. These are FreeBSD's positive `EAI_*` values, matching the errnos
/// below rather than glibc's negative ones.
const SFDNSRES_EAI_NONAME: i32 = 8;

const SFDNSRES_HOST_NOT_FOUND: i32 = 1;

/// `bsd` errnos, in **FreeBSD's** numbering — which is what the real service
/// returns, and so what guest code is written against (`EAGAIN` is 35 here,
/// not the 11 a Linux-hosted build would use).
const BSD_EBADF: i32 = 9;

const BSD_EINVAL: i32 = 22;

const BSD_EAGAIN: i32 = 35;

const BSD_EPIPE: i32 = 32;

const BSD_ENETUNREACH: i32 = 51;

const BSD_EISCONN: i32 = 56;

const BSD_ENOTCONN: i32 = 57;

const BSD_ECONNREFUSED: i32 = 61;

/// `SOCK_DGRAM`. A datagram socket has nowhere to send *to* (`ENETUNREACH`)
/// where a stream socket has no connection to send *on* (`ENOTCONN`), and only
/// a stream socket can be one end of the loopback pair below.
const BSD_SOCK_DGRAM: u32 = 2;

/// `AF_INET`, in the `sin_family` byte of Horizon's `sockaddr`.
const BSD_AF_INET: u8 = 2;

/// `sizeof(sockaddr_in)`, which is also the `sin_len` every well-formed one
/// carries.
const BSD_SOCKADDR_IN_LEN: usize = 16;

/// Where [`Cpu::bsd_assign_port`] starts handing out ports for a `bind` to
/// port 0: the bottom of IANA's ephemeral range, which is where FreeBSD's own
/// allocator starts.
pub(super) const BSD_FIRST_EPHEMERAL_PORT: u16 = 49152;

/// The loopback address. A listener bound to it, to `INADDR_ANY` or to the
/// address `nifm` reports is reachable from this process; nothing else is.
const BSD_LOOPBACK_IP: [u8; 4] = [127, 0, 0, 1];

const BSD_ANY_IP: [u8; 4] = [0, 0, 0, 0];

/// How many descriptors a `select` set can name: `FD_SETSIZE`, which is what
/// the 128-byte bitmap the caller marshals holds.
const BSD_MAX_SELECT_FDS: u32 = 1024;

/// The `(address, port)` an `AF_INET` `sockaddr_in` names, or `None` for any
/// other family — nothing else can be an endpoint of this process.
fn sockaddr_in(raw: &[u8]) -> Option<([u8; 4], u16)> {
    if raw.len() < 8 || raw[1] != BSD_AF_INET {
        return None;
    }
    let port = u16::from_be_bytes([raw[2], raw[3]]);
    Some(([raw[4], raw[5], raw[6], raw[7]], port))
}

/// A well-formed `sockaddr_in`. Horizon's is FreeBSD's — a length byte and a
/// family byte where Linux has a 16-bit family — and both the port and the
/// address are in network order.
fn sockaddr_in_bytes(ip: [u8; 4], port: u16) -> Vec<u8> {
    let mut raw = vec![0u8; BSD_SOCKADDR_IN_LEN];
    raw[0] = BSD_SOCKADDR_IN_LEN as u8;
    raw[1] = BSD_AF_INET;
    raw[2..4].copy_from_slice(&port.to_be_bytes());
    raw[4..8].copy_from_slice(&ip);
    raw
}

/// Whether an address names this console. `connect` reaches a listener only
/// through one of these; everything else is off the console and refused.
fn is_local_ip(ip: [u8; 4]) -> bool {
    ip == BSD_LOOPBACK_IP || ip == BSD_ANY_IP || ip == NIFM_LOCAL_IP
}

/// `FIONBIO`, `F_GETFL`/`F_SETFL`, and FreeBSD's `O_NONBLOCK` — the last only
/// so that the `ioctl` route sets the same bit the `fcntl` route reads back.
const BSD_FIONBIO: u32 = 0x8004_667E;

const BSD_F_GETFL: u32 = 3;

const BSD_F_SETFL: u32 = 4;

const BSD_O_NONBLOCK: u32 = 0x0004;

impl Cpu {
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

    /// `sfdnsres` (`IResolver`): the DNS resolver, and the other half of the
    /// socket stack `bsd` is the transport half of. `getaddrinfo`,
    /// `gethostbyname` and `getnameinfo` are all IPC calls into this, and
    /// libnx's `socketInitialize` opens it alongside `bsd:u`.
    ///
    /// **Nothing resolves.** There is no resolver here and no network to reach
    /// a name server on, so every lookup fails the way a name that does not
    /// exist fails: `EAI_NONAME` for the `getaddrinfo` family, `HOST_NOT_FOUND`
    /// for the `gethostbyname` one. That is deliberately the *definitive*
    /// failure rather than `EAI_AGAIN`, which invites a caller to retry
    /// forever — the same reasoning as `bsd`'s `ECONNREFUSED`, and for the same
    /// reason: there is no other thread here to run while a guest retries.
    ///
    /// A numeric address string would resolve on real hardware without any DNS
    /// at all, and this fails that too. Serializing an `addrinfo` into the
    /// packed form Horizon returns is guesswork this cannot verify against a
    /// real console, and the connect that would follow is refused by `bsd`
    /// anyway — so the lookup fails where the guest can act on it, rather than
    /// succeeding into a reply whose layout might be wrong.
    ///
    /// The error *strings* are worth answering properly: a guest that prints
    /// why a lookup failed gets a sentence, not an empty line.
    pub(super) fn sfdnsres_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &0x1000u16.to_le_bytes(), &[]);
        }
        match cmd_id {
            // GetHostByNameRequest / GetHostByAddrRequest, and their
            // WithOptions forms: the `gethostbyname` family, which reports
            // through `h_errno`.
            Some(2) | Some(3) | Some(10) | Some(11) => {
                self.sfdnsres_failure(tls, SFDNSRES_HOST_NOT_FOUND)
            }
            // GetAddrInfoRequest / GetNameInfoRequest and their WithOptions
            // forms: the `getaddrinfo` family, which reports a `gai` error.
            Some(6) | Some(7) | Some(12) | Some(13) => {
                self.sfdnsres_failure(tls, SFDNSRES_EAI_NONAME)
            }
            // GetHostStringErrorRequest / GetGaiStringErrorRequest: the text
            // for an error code, into an output buffer.
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
            // RequestCancelHandleRequest -> u32: the token a caller passes to
            // CancelRequest to abandon a lookup in flight. Every lookup here
            // finishes before it returns, so the token is only ever handed
            // back and cancelled.
            Some(8) => {
                let handle = self.next_object_id;
                self.next_object_id = handle.wrapping_add(1);
                self.write_ipc_response(tls, 0, &[], &handle.to_le_bytes(), &[])
            }
            // CancelRequest, and the resolver options: there is nothing in
            // flight to cancel, and no resolver whose behaviour an option
            // could change.
            Some(9) | Some(14) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(15) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "sfdnsres", cmd_id),
        }
    }

    /// A failed lookup: the error in the first word, no `errno` behind it, and
    /// nothing serialized into the output buffer.
    ///
    /// The three words are `SfdnsresRequestResults` — return value, `errno`,
    /// and how many bytes were written to the caller's buffer. Putting the
    /// failure in the *first* word is what makes this robust to the exact
    /// field order: a caller checking the return value sees the error, and one
    /// that reads the serialized size sees zero either way. `errno` stays 0
    /// because these errors are not `EAI_SYSTEM` — there is no underlying
    /// system call that failed.
    fn sfdnsres_failure(&mut self, tls: u32, error: i32) -> Result<()> {
        let mut results = [0u8; 12];
        results[..4].copy_from_slice(&error.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &results, &[])
    }

    /// `bsd:u`/`bsd:s`, the socket service — `nn::socket` and libnx's
    /// `socketInitialize` sit on top of it.
    ///
    /// **The only peer is this console.** A browser tab cannot open a TCP
    /// socket and nothing here proxies one, so what is modelled is a console
    /// whose link is up (which is what `nifm` reports) and on which nothing
    /// off the box ever answers. Everything aimed at this console itself is
    /// real: two sockets of this process connect, accept and carry bytes
    /// between them, `select` and `poll` report which of them are ready, and a
    /// `close` at one end is end-of-file at the other. Everything aimed
    /// anywhere else is `ECONNREFUSED` — at once rather than as a timeout,
    /// precisely because a title checking for an update should find out now
    /// rather than block a frame loop that has no other thread to run.
    ///
    /// The errnos are **FreeBSD's**, not Linux's or newlib's (`EAGAIN` is 35,
    /// not 11), because that is what the real service returns and guest code
    /// is written against the real service. A title whose own `strerror` table
    /// is Linux's will print the wrong sentence for the right number, on
    /// hardware as much as here.
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
                let socket = BsdSocket::new(word(self, 0), word(self, 1));
                let fd = self.alloc_bsd_fd();
                self.bsd_sockets.insert(fd, socket);
                self.bsd_reply(tls, fd, 0)
            }
            // Select(nfds, timeval timeout) with the read, write and except
            // sets in buffers 0, 1 and 2 on each side.
            //
            // The out-sets are always written, never left as the caller's own
            // input: a caller handed a success reads readiness out of the
            // *output* buffer, and one that finds its own request there sees
            // every descriptor it asked about as ready.
            //
            // The timeout is a `timeval` at the second word — asio asks for
            // 300 seconds when it has no timer pending, which is its own cap
            // and not a number this can honour. A wait that finds nothing
            // gives up the CPU for the reason [`Cpu::bsd_request`]'s `Poll`
            // arm gives; a zero timeout is an explicit probe and does not.
            Some(5) => {
                let nfds = word(self, 0).min(BSD_MAX_SELECT_FDS);
                let timeout = self.mem.read_u64(data.wrapping_add(8)).unwrap_or(0)
                    | self.mem.read_u64(data.wrapping_add(16)).unwrap_or(0);
                let mut ready = 0;
                for index in 0..3 {
                    ready += self.bsd_select_set(tls, index, nfds)?;
                }
                self.pending_yield = ready == 0 && timeout != 0;
                self.bsd_reply(tls, ready, 0)
            }
            // Poll(nfds, timeout): the fds come in and go back out, with each
            // `revents` cleared — no event ever fires. Copying the array
            // through matters: the caller reads its `revents` out of the
            // *output* buffer, which is a different range from the input one.
            //
            // A poll with a timeout is a *wait*, and answering it instantly is
            // what breaks a guest, not the empty answer. NXpotify's Zeroconf
            // listener runs `if (poll(&pfd, 1, 200) <= 0) continue;`, which on
            // hardware sleeps a fifth of a second per turn; returning zero
            // immediately turned it into a loop that never makes a blocking
            // syscall, and threads here only switch at those — so it starved
            // every other thread, main included, and no frame was ever drawn.
            // Reschedule instead, once the reply is written.
            Some(6) => {
                let timeout = word(self, 1) as i32;
                let mut ready = 0;
                if let (Some((src, src_size)), Some((dst, dst_size))) =
                    (self.ipc_input_buffer(tls, 0), self.ipc_output_buffer(tls, 0))
                {
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
                // `timeout == 0` is an explicit non-blocking probe, and comes
                // back at once on hardware too.
                self.pending_yield = ready == 0 && timeout != 0;
                self.bsd_reply(tls, ready as i32, 0)
            }
            // Recv(fd, flags) / Read(fd): bytes the peer sent, out of this
            // socket's queue and into the caller's buffer.
            Some(8) | Some(25) => {
                let fd = word(self, 0) as i32;
                self.bsd_receive(tls, fd, None)
            }
            // RecvFrom(fd, flags): the same, and the sender's address — which
            // for a connected socket is the peer's — in the second buffer.
            Some(9) => {
                let fd = word(self, 0) as i32;
                self.bsd_receive(tls, fd, Some(1))
            }
            // Send(fd, flags) / SendTo(fd, flags, sockaddr) / Write(fd): into
            // the peer's queue.
            //
            // `SendTo`'s destination is ignored on a connected socket, which
            // is what the connected end of a loopback pair is; on an
            // unconnected one there is nowhere to send to and it fails like
            // the rest.
            Some(10) | Some(11) | Some(24) => {
                let fd = word(self, 0) as i32;
                self.bsd_send(tls, fd)
            }
            // Accept(fd) -> the connection at the head of the listener's
            // queue, its address in the output buffer and the length of that
            // address in the third reply word.
            //
            // EAGAIN when the queue is empty says "not right now" rather than
            // failing the listener outright — which is what a server socket on
            // an idle network reports, and unlike blocking forever it leaves
            // the guest's own loop able to run.
            Some(12) => {
                let fd = word(self, 0) as i32;
                let accepted = match self.bsd_sockets.get_mut(&fd) {
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
            // Bind(fd, sockaddr): genuinely local, and genuinely succeeds. The
            // address is kept because `GetSockName` has to report it back, and
            // normalized on the way in because what it reports is what a
            // caller then connects to.
            Some(13) => {
                let address = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(0x80)),
                    None => Vec::new(),
                };
                let fd = word(self, 0) as i32;
                if !self.bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let address = self.bsd_normalize_bind(address);
                if let Some(socket) = self.bsd_sockets.get_mut(&fd) {
                    socket.bound = address;
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Connect(fd, sockaddr): to a listener of this process, or
            // nowhere.
            Some(14) => {
                let address = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) => self.read_bytes(addr, size.min(0x80)),
                    None => Vec::new(),
                };
                let fd = word(self, 0) as i32;
                self.bsd_connect(tls, fd, &address)
            }
            // GetPeerName(fd) -> the address of the socket at the other end.
            Some(15) => {
                let fd = word(self, 0) as i32;
                match self.bsd_sockets.get(&fd) {
                    None => return self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if socket.peer.is_none() => {
                        return self.bsd_reply(tls, -1, BSD_ENOTCONN)
                    }
                    Some(_) => {}
                }
                let address = self.bsd_peer_address(fd);
                let written = self.bsd_write_address(tls, 0, &address)?;
                self.bsd_reply_len(tls, 0, 0, written)
            }
            // GetSockName(fd) -> the bound address, or the console's own
            // address (the one `nifm` reports) when nothing was bound.
            //
            // The **third** reply word is the length of what was written, and
            // it is not optional: nnSdk hands that length to whatever the
            // caller does next, so a `getsockname` that reports zero turns the
            // `connect` after it into `EINVAL` inside the SDK, which never
            // reaches this service at all. That is what Asphalt 9's
            // `socket_select_interrupter: Invalid argument` was.
            Some(16) => {
                let fd = word(self, 0) as i32;
                let address = match self.bsd_sockets.get(&fd) {
                    None => return self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if !socket.bound.is_empty() => socket.bound.clone(),
                    Some(_) => Self::bsd_local_address(),
                };
                let written = self.bsd_write_address(tls, 0, &address)?;
                self.bsd_reply_len(tls, 0, 0, written)
            }
            // GetSockOpt(fd, level, option) -> the option's value in the
            // output buffer, and its length in the third reply word. Options
            // are read back, so they are stored rather than acknowledged and
            // forgotten — the same reason `ssl`'s are.
            Some(17) => {
                let (fd, level, option) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                if !self.bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let value = self.bsd_socket_options.get(&(fd, level, option)).copied().unwrap_or(0);
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
            // Shutdown(fd, how): `SHUT_WR` and `SHUT_RDWR` end the peer's
            // reading, which is how a guest signals end-of-file without giving
            // up the descriptor. `SHUT_RD` only stops this end reading, and
            // nothing here reads on the caller's behalf.
            Some(22) => {
                const SHUT_RD: u32 = 0;
                let (fd, how) = (word(self, 0) as i32, word(self, 1));
                if !self.bsd_sockets.contains_key(&fd) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                if how != SHUT_RD {
                    self.bsd_orphan_peer(fd);
                }
                self.bsd_reply(tls, 0, 0)
            }
            // ShutdownAllSockets(how): every connection at once, which is what
            // a process tearing its socket layer down issues.
            Some(23) => {
                let fds: Vec<i32> = self.bsd_descriptors();
                for fd in fds {
                    self.bsd_orphan_peer(fd);
                }
                self.bsd_reply(tls, 0, 0)
            }
            // Close(fd). The peer is left readable rather than merely
            // disconnected: a read on it now reports end-of-file, which is
            // what the other end of a closed connection does.
            Some(26) => {
                if !self.bsd_sockets.contains_key(&(word(self, 0) as i32)) {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                }
                let fd = word(self, 0) as i32;
                self.bsd_close(fd);
                self.bsd_reply(tls, 0, 0)
            }
            // DuplicateSocket(fd): a second descriptor for the same socket.
            //
            // The copy carries this socket's *local* state — its family, its
            // address, its flags — and not its connection: two descriptors
            // sharing one byte queue would need an indirection this table does
            // not have, and a copy that claimed the connection would swallow
            // the bytes the original is owed. Nothing in this emulator's path
            // duplicates a connected socket.
            Some(27) => {
                let fd = word(self, 0) as i32;
                let Some(socket) = self.bsd_sockets.get(&fd) else {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                };
                let mut copy = BsdSocket::new(socket.domain, socket.kind);
                copy.bound = socket.bound.clone();
                copy.flags = socket.flags;
                copy.listening = socket.listening;
                let duplicate = self.alloc_bsd_fd();
                self.bsd_sockets.insert(duplicate, copy);
                self.bsd_reply(tls, duplicate, 0)
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

    /// Answer a command whose reply carries a **third** word: how many bytes
    /// were written into the caller's output buffer.
    ///
    /// `GetSockName`, `GetPeerName`, `Accept`, `RecvFrom` and `GetSockOpt` all
    /// have it, and nnSdk passes that length on to whatever the caller does
    /// next rather than assuming the buffer was filled. Answering those five
    /// with the two-word reply left the length reading zero, which is a
    /// `sockaddr` of no bytes — see the `GetSockName` arm above.
    fn bsd_reply_len(&mut self, tls: u32, ret: i32, errno: i32, len: u32) -> Result<()> {
        let mut raw = [0u8; 12];
        raw[..4].copy_from_slice(&ret.to_le_bytes());
        raw[4..8].copy_from_slice(&errno.to_le_bytes());
        raw[8..].copy_from_slice(&len.to_le_bytes());
        self.write_ipc_response(tls, 0, &[], &raw, &[])
    }

    /// The next descriptor. Monotonic, so a closed one is never handed out
    /// again while anything still holds it.
    fn alloc_bsd_fd(&mut self) -> i32 {
        let fd = self.next_bsd_fd;
        self.next_bsd_fd = self.next_bsd_fd.wrapping_add(1);
        fd
    }

    /// Every open descriptor, in order. Sorted rather than in the table's own
    /// order because a hash map's is not stable between runs, and two runs of
    /// the same title have to make the same calls.
    fn bsd_descriptors(&self) -> Vec<i32> {
        let mut fds: Vec<i32> = self.bsd_sockets.keys().copied().collect();
        fds.sort_unstable();
        fds
    }

    /// The `sockaddr_in` `GetSockName` reports for a socket that was never
    /// bound: the address `nifm` says this console has, on port 0.
    fn bsd_local_address() -> Vec<u8> {
        sockaddr_in_bytes(NIFM_LOCAL_IP, 0)
    }

    /// Where a connected socket's peer can be reached, which is what
    /// `GetPeerName` reports and what `Accept` and `RecvFrom` write out.
    fn bsd_peer_address(&self, fd: i32) -> Vec<u8> {
        let peer = self.bsd_sockets.get(&fd).and_then(|socket| socket.peer);
        match peer.and_then(|peer| self.bsd_sockets.get(&peer)) {
            Some(peer) if !peer.bound.is_empty() => peer.bound.clone(),
            _ => Self::bsd_local_address(),
        }
    }

    /// Put an address in the caller's `index`-th output buffer, and say how
    /// much of it fitted.
    ///
    /// A caller may offer no buffer at all — asio's `accept` passes a null
    /// one, because it does not care who connected — and that is a length of
    /// zero rather than a failure.
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

    /// Normalize what `bind` was handed, so that what `GetSockName` reports
    /// back is an address something can actually connect to.
    ///
    /// Two things are wrong with echoing the caller's own bytes. A guest that
    /// memsets a `sockaddr_in` and fills in only the family leaves `sin_len`
    /// at zero, which is not a `sockaddr` any nnSdk call will accept; and a
    /// bind to port 0 is a request for *a* port, not a socket that answers on
    /// port 0. An address of any other family is passed through untouched —
    /// this service does not know what it means, and reporting back exactly
    /// what it was given is the one answer that cannot be wrong.
    fn bsd_normalize_bind(&mut self, address: Vec<u8>) -> Vec<u8> {
        let Some((ip, port)) = sockaddr_in(&address) else {
            return address;
        };
        let port = if port == 0 { self.bsd_assign_port() } else { port };
        sockaddr_in_bytes(ip, port)
    }

    /// A port no open socket is bound to. Wraps around the ephemeral range
    /// rather than growing without bound, and gives up after one lap — at
    /// which point every port really is taken and reusing one is the least
    /// wrong answer left.
    fn bsd_assign_port(&mut self) -> u16 {
        let range = u16::MAX - BSD_FIRST_EPHEMERAL_PORT + 1;
        let mut port = self.next_bsd_port;
        for _ in 0..range {
            let candidate = port;
            port = if candidate == u16::MAX { BSD_FIRST_EPHEMERAL_PORT } else { candidate + 1 };
            let taken = self
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

    /// The listening socket that answers on `port`, lowest descriptor first.
    fn bsd_listener_on(&self, port: u16) -> Option<i32> {
        self.bsd_descriptors().into_iter().find(|fd| {
            self.bsd_sockets.get(fd).is_some_and(|socket| {
                socket.listening && sockaddr_in(&socket.bound).is_some_and(|(_, p)| p == port)
            })
        })
    }

    /// `Connect(fd, sockaddr)`: to a listener of this process, or nowhere.
    ///
    /// The connection completes here rather than being queued for the
    /// listener to finish, because there is no listener *thread* to finish it
    /// — both ends are this process, and a connect that returned "in progress"
    /// would be waiting on the guest to run code it only runs after the
    /// connect returns.
    ///
    /// Everything that is not a listener of this process is `ECONNREFUSED`,
    /// at once, for the reason the module doc gives: a title that checks for
    /// an update has to find out now, and there is no other thread here to run
    /// while it blocks.
    fn bsd_connect(&mut self, tls: u32, fd: i32, address: &[u8]) -> Result<()> {
        let (domain, kind) = match self.bsd_sockets.get(&fd) {
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
            return self.bsd_reply(tls, -1, BSD_ECONNREFUSED);
        };

        // The accepted end answers on the listener's own address, which is
        // what `GetSockName` on it has to report.
        let mut accepted = BsdSocket::new(domain, kind);
        accepted.bound =
            self.bsd_sockets.get(&listener).map(|l| l.bound.clone()).unwrap_or_default();
        accepted.peer = Some(fd);
        let accepted_fd = self.alloc_bsd_fd();
        self.bsd_sockets.insert(accepted_fd, accepted);

        // A client that never bound gets an address now, so that
        // `GetPeerName` on the accepted end names something.
        let unbound =
            self.bsd_sockets.get(&fd).is_some_and(|socket| socket.bound.is_empty());
        let client_address =
            unbound.then(|| sockaddr_in_bytes(BSD_LOOPBACK_IP, self.bsd_assign_port()));
        if let Some(socket) = self.bsd_sockets.get_mut(&fd) {
            if let Some(address) = client_address {
                socket.bound = address;
            }
            socket.peer = Some(accepted_fd);
        }
        if let Some(listener) = self.bsd_sockets.get_mut(&listener) {
            listener.incoming.push_back(accepted_fd);
        }
        self.bsd_reply(tls, 0, 0)
    }

    /// `Send`/`SendTo`/`Write`: into the peer's queue, all of it, at once.
    /// Nothing here has a send buffer that can fill, so a short write is not a
    /// state this can reach.
    fn bsd_send(&mut self, tls: u32, fd: i32) -> Result<()> {
        let peer = match self.bsd_sockets.get(&fd) {
            None => return self.bsd_reply(tls, -1, BSD_EBADF),
            // The peer is gone; on hardware this raises `SIGPIPE` as well, and
            // a guest that blocked it reads the errno instead.
            Some(socket) if socket.peer_closed => return self.bsd_reply(tls, -1, BSD_EPIPE),
            Some(socket) => match socket.peer {
                Some(peer) => peer,
                None => return self.bsd_unconnected(tls, fd),
            },
        };
        let bytes = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let sent = bytes.len() as i32;
        if let Some(peer) = self.bsd_sockets.get_mut(&peer) {
            peer.rx.extend(bytes);
        }
        self.bsd_reply(tls, sent, 0)
    }

    /// `Recv`/`RecvFrom`/`Read`, and — when `address_buffer` names one — the
    /// sender's address alongside the bytes.
    fn bsd_receive(&mut self, tls: u32, fd: i32, address_buffer: Option<u32>) -> Result<()> {
        let (ret, errno) = self.bsd_receive_bytes(tls, fd)?;
        let Some(index) = address_buffer else {
            return self.bsd_reply(tls, ret, errno);
        };
        let address = if ret >= 0 { self.bsd_peer_address(fd) } else { Vec::new() };
        let written = self.bsd_write_address(tls, index, &address)?;
        self.bsd_reply_len(tls, ret, errno, written)
    }

    /// Drain what the peer sent into the caller's buffer, and say how the read
    /// went.
    ///
    /// An empty queue on a live connection is `EAGAIN` **and** a reschedule,
    /// not a block: a blocking read would have to be resumed from inside the
    /// syscall that made it, and the guest re-checks its own predicate in a
    /// loop anyway. Giving up the CPU is what lets the thread that will send
    /// the bytes run — a read that spins here would starve it, which is the
    /// same trap [`Cpu::bsd_request`]'s `Poll` arm describes.
    fn bsd_receive_bytes(&mut self, tls: u32, fd: i32) -> Result<(i32, i32)> {
        match self.bsd_sockets.get(&fd) {
            None => return Ok((-1, BSD_EBADF)),
            Some(socket) if socket.peer.is_none() && !socket.peer_closed => {
                let unconnected =
                    if socket.kind == BSD_SOCK_DGRAM { BSD_ENETUNREACH } else { BSD_ENOTCONN };
                return Ok((-1, unconnected));
            }
            Some(_) => {}
        }
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return Ok((-1, BSD_EINVAL));
        };
        let Some(socket) = self.bsd_sockets.get_mut(&fd) else {
            return Ok((-1, BSD_EBADF));
        };
        let take = size.min(socket.rx.len() as u32) as usize;
        if take == 0 {
            // A peer that has gone is end-of-file, which is a read of zero
            // bytes and not an error.
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

    /// The answer for an operation that needs the other end of a connection
    /// this socket does not have: a datagram socket has nowhere to send *to*,
    /// a stream socket has no connection to send *on*.
    fn bsd_unconnected(&mut self, tls: u32, fd: i32) -> Result<()> {
        match self.bsd_sockets.get(&fd) {
            None => self.bsd_reply(tls, -1, BSD_EBADF),
            Some(socket) if socket.kind == BSD_SOCK_DGRAM => {
                self.bsd_reply(tls, -1, BSD_ENETUNREACH)
            }
            Some(_) => self.bsd_reply(tls, -1, BSD_ENOTCONN),
        }
    }

    /// One of `select`'s three descriptor sets: read the caller's, write back
    /// which of those descriptors are ready, and count them.
    ///
    /// A set is a bitmap indexed by descriptor. FreeBSD's `fd_mask` is 64 bits
    /// wide and Linux's is 32, and on a little-endian machine both put
    /// descriptor *n* in bit *n* of the byte array either way — so this walks
    /// bytes and does not have to know which.
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
            // A descriptor this service never handed out is nothing it can
            // report on. Saying "not ready" leaves the caller waiting, which
            // is what it was already doing before any of this was modelled.
            let is_ready = match self.bsd_sockets.get(&(fd as i32)) {
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

    /// Which of the events a `poll` asked about have happened on `fd`.
    ///
    /// A descriptor this service does not know is answered with no events
    /// rather than `POLLNVAL`: a guest polls its own pipes and standard
    /// streams alongside its sockets, and none of those are this service's to
    /// call invalid.
    fn bsd_poll_revents(&self, fd: i32, events: u16) -> u16 {
        const POLLIN: u16 = 0x0001;
        const POLLOUT: u16 = 0x0004;
        const POLLHUP: u16 = 0x0010;
        let Some(socket) = self.bsd_sockets.get(&fd) else {
            return 0;
        };
        let mut revents = 0;
        if events & POLLIN != 0 && socket.readable() {
            revents |= POLLIN;
        }
        if events & POLLOUT != 0 && socket.writable() {
            revents |= POLLOUT;
        }
        // Reported whether or not it was asked for, the way `poll` does.
        if socket.peer_closed && socket.rx.is_empty() {
            revents |= POLLHUP;
        }
        revents
    }

    /// Tell `fd`'s peer that nothing more is coming: its reads report
    /// end-of-file, and it is no longer writable.
    ///
    /// The link itself is left in place so `GetPeerName` still names who was
    /// there — a peer that is gone is not a connection that never existed.
    fn bsd_orphan_peer(&mut self, fd: i32) {
        let Some(peer) = self.bsd_sockets.get(&fd).and_then(|socket| socket.peer) else {
            return;
        };
        if let Some(peer) = self.bsd_sockets.get_mut(&peer) {
            peer.peer_closed = true;
        }
    }

    /// Drop a descriptor, and everything that was only reachable through it: a
    /// listener takes its unaccepted connections with it, exactly as closing
    /// one on hardware does.
    fn bsd_close(&mut self, fd: i32) {
        let Some(socket) = self.bsd_sockets.remove(&fd) else {
            return;
        };
        self.bsd_socket_options.retain(|&(owner, _, _), _| owner != fd);
        if let Some(peer) = socket.peer.and_then(|peer| self.bsd_sockets.get_mut(&peer)) {
            peer.peer_closed = true;
        }
        for pending in socket.incoming {
            self.bsd_close(pending);
        }
    }

    /// `nifm`'s root session (`nifm:u`, `nifm:s`, `nifm:a`): session control
    /// plus `CreateGeneralServiceOld`/`CreateGeneralService`, which hand back
    /// the `IGeneralService` connectivity is actually queried through.
    ///
    /// The three names are the same interface at three privilege levels, and
    /// only `nifm:u` used to be routed here — so a system title, which opens
    /// `nifm:s`, had every one of its network calls answered by the generic
    /// fallback instead.
    pub(super) fn nifm_request(&mut self, tls: u32, cmd_id: Option<u32>, handle: u64) -> Result<()> {
        const CONVERT_TO_DOMAIN: u32 = 0;
        const QUERY_POINTER_BUFFER_SIZE: u32 = 3;
        if self.ipc_is_control_request(tls) {
            return match cmd_id {
                Some(CONVERT_TO_DOMAIN) => {
                    // Under the session's own name, so the three aliases stay
                    // distinguishable in a trace.
                    let name = self.service_name(handle).unwrap_or("nifm:u").to_string();
                    let obj = self.alloc_domain_object();
                    self.record_domain_object(handle, obj, &name);
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
    /// caller that only checks "is there a connection" sees a permanent wired
    /// one instead of the emulator looking offline.
    ///
    /// The command ids used to be crossed: 12 answered with the connection
    /// status triple and 15 with the IP address, when 12 *is*
    /// `GetCurrentIpAddress`, 15 is `GetCurrentIpConfigInfo` and 18 is
    /// `GetInternetConnectionStatus`. So a caller asking for the console's
    /// address got `{2, 0, 2}` for one, and the one query that matters — is
    /// there internet — fell through to a bare success.
    pub(super) fn nifm_general_service_request(
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
            // EnumerateNetworkInterfaces / EnumerateNetworkProfiles: the list
            // goes in a buffer nothing fills, and the count that comes back
            // with it is what a caller iterates on. Zero of them.
            Some(6) | Some(7) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetCurrentNetworkProfile: the whole answer is an
            // `SfNetworkProfileData` written into the caller's buffer, with
            // nothing in the raw reply. The link reported here is wired, and a
            // wired link genuinely has no wireless profile — but the buffer
            // still has to be *written*, because a caller handed a success and
            // an untouched buffer reads its profile off its own stack.
            Some(5) => {
                if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                    for i in 0..len {
                        self.mem.write_u8(addr.wrapping_add(i), 0)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetCurrentIpAddress.
            Some(12) => self.write_ipc_response(tls, 0, &[], &NIFM_LOCAL_IP, &[]),
            // GetCurrentIpConfigInfo -> IpAddressSetting { bool is_automatic;
            // address; subnet; gateway } then a DnsSetting. Automatic, with
            // the address `bsd` also reports, a /24 behind it and the router
            // at .1 — the three have to agree or a caller computing its own
            // broadcast address gets one off this subnet.
            Some(15) => {
                let mut raw = Vec::with_capacity(0x18);
                raw.push(1); // is_automatic
                raw.extend_from_slice(&NIFM_LOCAL_IP);
                raw.extend_from_slice(&[255, 255, 255, 0]);
                raw.extend_from_slice(&[NIFM_LOCAL_IP[0], NIFM_LOCAL_IP[1], NIFM_LOCAL_IP[2], 1]);
                raw.resize(0x18, 0); // the DnsSetting, which resolves nothing
                self.write_ipc_response(tls, 0, &[], &raw, &[])
            }
            // IsWirelessCommunicationEnabled: the link this reports is wired,
            // so the radio is off — and saying otherwise invites a caller to
            // scan for access points that do not exist.
            Some(17) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // GetInternetConnectionStatus -> { NifmInternetConnectionType,
            // wifi strength, status }: Ethernet, no strength to report,
            // connected.
            Some(18) => self.write_ipc_response(tls, 0, &[], &[2u8, 0u8, 2u8], &[]),
            // IsEthernetCommunicationEnabled: that is the link.
            Some(20) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            // IsAnyInternetRequestAccepted / IsAnyForegroundRequestAccepted:
            // a request made here is accepted the moment it is made, so both.
            Some(21) | Some(22) => self.write_ipc_response(tls, 0, &[], &[1u8], &[]),
            _ => {
                self.warn_no_implementation("nifm:general-service", cmd_id);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
        }
    }

    /// `IRequest`: one application's claim on the network.
    ///
    /// The link is up and nothing else is competing for it, so a request is
    /// **Accepted** from the moment it exists and its result is success.
    /// The two events it hands out start signalled for the same reason: a
    /// caller waits on them for the state to settle, and it already has.
    /// Answering those two with nothing — which is what a bare success does —
    /// left a caller holding handle 0 for the one and a session for the other.
    pub(super) fn nifm_request_object_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        match cmd_id {
            // GetRequestState -> NifmRequestState_Accepted.
            Some(0) => self.write_ipc_response(tls, 0, &[], &3u32.to_le_bytes(), &[]),
            // GetSystemEventReadableHandles -> **two** copy handles: the state
            // change and the request's completion.
            Some(2) => {
                let state = self.alloc_event("nifm:request-state", true);
                let done = self.alloc_event("nifm:request-done", true);
                self.signal_event(state);
                self.signal_event(done);
                self.write_ipc_reply(tls, 0, &[state, done], &[], &[], &[])
            }
            // GetRevision.
            Some(20) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
            // GetResult, Cancel, Submit, SubmitAndWait and the whole family of
            // requirement setters: a bare Result, and nothing here to set.
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;
    use crate::cpu::Cpu;

    /// Read a `bsd` command's `{ s32 ret, s32 errno }` reply.
    fn bsd_result(cpu: &Cpu) -> (i32, i32) {
        (
            cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
            cpu.mem.read_u32(TLS + 0x24).unwrap() as i32,
        )
    }

    /// The third word of the replies that carry one: how many bytes of the
    /// caller's output buffer were filled.
    fn bsd_result_len(cpu: &Cpu) -> u32 {
        cpu.mem.read_u32(TLS + 0x28).unwrap()
    }

    /// Open a socket of `kind` on a fresh `bsd:u` session, returning the cpu
    /// and the descriptor.
    fn bsd_socket(kind: u32) -> (Cpu, i32) {
        let mut cpu = request(false, 2, &[]);
        cpu.register_service_handle(9, "bsd:u");
        let fd = open_socket(&mut cpu, kind);
        (cpu, fd)
    }

    /// A second (and third) socket on a session that already has one.
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

    /// Where the tests below park a `sockaddr` and a byte or two of payload.
    const SCRATCH: u32 = 0x4000;

    /// `127.0.0.1:port`, as a caller writes it before a `bind` or a `connect`
    /// — `sin_len` left at zero, because a guest that memsets the struct and
    /// fills in only the family is exactly the case this has to survive.
    fn loopback_sockaddr(port: u16) -> [u8; 16] {
        let mut raw = [0u8; 16];
        raw[1] = 2; // AF_INET
        raw[2..4].copy_from_slice(&port.to_be_bytes());
        raw[4..8].copy_from_slice(&[127, 0, 0, 1]);
        raw
    }

    /// Put an address where a request's buffer descriptor will point at it.
    fn place(cpu: &mut Cpu, at: u32, bytes: &[u8]) {
        for (offset, &byte) in bytes.iter().enumerate() {
            cpu.mem.write_u8(at + offset as u32, byte).unwrap();
        }
    }

    /// Run asio's `socket_select_interrupter` dance and hand back the two
    /// descriptors it ends up holding: bind a listener to an ephemeral port on
    /// the loopback address, connect to the port `getsockname` reports, and
    /// accept the other end.
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

        // Straight back out of the buffer `getsockname` filled, which is what
        // asio connects to.
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

    /// Send `bytes` on `fd`, and report what the service said.
    fn send_on(cpu: &mut Cpu, fd: i32, bytes: &[u8]) -> (i32, i32) {
        const AT: u32 = SCRATCH + 0x40;
        place(cpu, AT, bytes);
        write_map_buffer_request(cpu, 10, &fd.to_le_bytes(), AT, bytes.len() as u32, true);
        cpu.bsd_request(TLS, 9, Some(10)).unwrap();
        bsd_result(cpu)
    }

    /// Read up to `len` bytes off `fd`, and report what the service said and
    /// what landed in the buffer.
    fn recv_on(cpu: &mut Cpu, fd: i32, len: u32) -> ((i32, i32), Vec<u8>) {
        const AT: u32 = SCRATCH + 0x80;
        place(cpu, AT, &vec![0u8; len as usize]);
        write_map_buffer_request(cpu, 8, &fd.to_le_bytes(), AT, len, false);
        cpu.bsd_request(TLS, 9, Some(8)).unwrap();
        let result = bsd_result(cpu);
        let read = if result.0 > 0 { cpu.read_bytes(AT, result.0 as u32) } else { Vec::new() };
        (result, read)
    }

    #[test]
    fn sfdnsres_fails_every_lookup_definitively() {
        // getaddrinfo: EAI_NONAME, not EAI_AGAIN. A caller told to try again
        // tries again, and there is no other thread here to run while it does.
        let mut cpu = request(false, 6, &[]);
        cpu.sfdnsres_request(TLS, Some(6)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap() as i32, super::SFDNSRES_EAI_NONAME);
        // Nothing was serialized into the caller's buffer, and no errno is
        // claimed behind the failure.
        assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), 0, "errno");
        assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "serialized size");

        // gethostbyname reports through h_errno, which has its own numbering.
        let mut cpu = request(false, 2, &[]);
        cpu.sfdnsres_request(TLS, Some(2)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap() as i32, super::SFDNSRES_HOST_NOT_FOUND);
    }

    #[test]
    fn sfdnsres_explains_the_failure_it_reports() {
        // A guest that prints why a lookup failed should get a sentence, not
        // an empty line.
        const BUFFER: u32 = 0x4000;
        let mut cpu = request_with_recv_buffer(5, &[], BUFFER, 0x40);
        cpu.mem.map_zero(BUFFER, 0x100).unwrap();
        cpu.sfdnsres_request(TLS, Some(5)).unwrap();
        assert_eq!(cpu.read_string(BUFFER, 0x40), "Name or service not known");
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
    fn a_poll_with_a_timeout_gives_up_the_cpu() {
        // NXpotify's Zeroconf listener runs `if (poll(&pfd, 1, 200) <= 0)
        // continue;`, which on hardware sleeps a fifth of a second per turn.
        // Answering "nothing ready" instantly turned that into a loop with no
        // blocking syscall in it, and threads here only switch at those — so
        // it starved every other thread, main included, and the app never drew
        // a frame. The empty answer is right; returning it without yielding is
        // not.
        let (mut cpu, _fd) = bsd_socket(1);
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&1u32.to_le_bytes()); // nfds
        payload[4..].copy_from_slice(&200i32.to_le_bytes()); // timeout, ms
        write_request(&mut cpu, 6, &payload);
        cpu.bsd_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "no descriptor is ever ready");
        assert!(cpu.pending_yield, "a poll that waits has to reschedule");

        // A zero timeout is an explicit non-blocking probe: hardware answers
        // that one immediately too, so there is nothing to give up.
        payload[4..].copy_from_slice(&0i32.to_le_bytes());
        write_request(&mut cpu, 6, &payload);
        cpu.bsd_request(TLS, 9, Some(6)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        assert!(!cpu.pending_yield);
    }

    #[test]
    fn bsd_bind_assigns_a_port_and_get_sock_name_reports_a_usable_address() {
        // Asphalt 9's `socket_select_interrupter: Invalid argument`. asio
        // binds to port 0, asks `getsockname` where that landed, and connects
        // there — so echoing the caller's own bytes back hands it
        // `127.0.0.1:0` with `sin_len` still zero, and nnSdk rejects the
        // connect without ever issuing it.
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
        // The third word is the one that was missing, and the one nnSdk hands
        // to whatever the caller does next.
        assert_eq!(bsd_result_len(&cpu), 16, "the length of the reported address");

        let reported = cpu.read_bytes(SCRATCH, 16);
        assert_eq!(reported[0], 16, "sin_len");
        assert_eq!(reported[1], 2, "AF_INET");
        assert_eq!(&reported[4..8], &[127, 0, 0, 1], "the address stays the one bound");
        let port = u16::from_be_bytes([reported[2], reported[3]]);
        assert!(port >= super::BSD_FIRST_EPHEMERAL_PORT, "an ephemeral port, not 0: {port}");

        // A port the caller *did* ask for is its own, not one reassigned.
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
        // The whole of `socket_select_interrupter::open_descriptors`, and then
        // the interrupt it exists to deliver. Every asio `io_context` builds
        // one of these before it runs anything, so a console that cannot make
        // the pair cannot run the middleware.
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);
        assert_ne!(client, server);

        // `interrupt()` writes one byte...
        assert_eq!(send_on(&mut cpu, client, &[0x7f]), (1, 0), "send");
        // ...and `reset()` drains it at the other end.
        assert_eq!(recv_on(&mut cpu, server, 0x20), ((1, 0), vec![0x7f]), "recv");
        // Drained, not merely peeked at: the second read finds nothing.
        assert_eq!(recv_on(&mut cpu, server, 0x20).0, (-1, super::BSD_EAGAIN));

        // The bytes go one way only. The client's own queue is still empty.
        assert_eq!(recv_on(&mut cpu, client, 0x20).0, (-1, super::BSD_EAGAIN));

        // GetPeerName names the other end rather than reporting no peer.
        place(&mut cpu, SCRATCH, &[0u8; 16]);
        write_map_buffer_request(&mut cpu, 15, &server.to_le_bytes(), SCRATCH, 16, false);
        cpu.bsd_request(TLS, 9, Some(15)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "getpeername");
        assert_eq!(bsd_result_len(&cpu), 16);
        assert_eq!(cpu.read_bytes(SCRATCH, 2), vec![16, 2]);
    }

    #[test]
    fn bsd_select_names_the_descriptor_that_has_a_byte() {
        /// One `fd_set`: a bitmap with room for `FD_SETSIZE` descriptors.
        const SET: u32 = 0x80;
        const SETS: u32 = 0x6000;
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        cpu.mem.map_zero(SETS, 6 * SET as usize).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);

        let sets: Vec<(u32, u32)> =
            (0..6).map(|index| (SETS + index * SET, SET)).collect();
        let select = |cpu: &mut Cpu, watch: i32, seconds: u64| {
            for offset in 0..6 * SET {
                cpu.mem.write_u8(SETS + offset, 0).unwrap();
            }
            let bit = 1u8 << (watch % 8);
            cpu.mem.write_u8(SETS + (watch as u32 / 8), bit).unwrap();
            let mut payload = [0u8; 24];
            payload[..4].copy_from_slice(&((watch + 1) as u32).to_le_bytes());
            payload[8..16].copy_from_slice(&seconds.to_le_bytes());
            write_buffer_request(cpu, 5, &payload, &sets[..3], &sets[3..]);
            cpu.pending_yield = false;
            cpu.bsd_request(TLS, 9, Some(5)).unwrap();
            let ready = bsd_result(cpu);
            let out = cpu.mem.read_u8(SETS + 3 * SET + (watch as u32 / 8)).unwrap();
            (ready, out & bit != 0)
        };

        // Nothing has been sent, so nothing is ready — and a wait that finds
        // nothing gives up the CPU rather than spinning the guest's loop.
        assert_eq!(select(&mut cpu, server, 1), ((0, 0), false));
        assert!(cpu.pending_yield, "a select that waits has to reschedule");

        assert_eq!(send_on(&mut cpu, client, &[0x7f]), (1, 0));
        // Now it is, and the *output* set is what says so. A caller handed a
        // success reads readiness out of that buffer, never out of its own
        // request — the two are different ranges.
        assert_eq!(select(&mut cpu, server, 1), ((1, 0), true));
        assert!(!cpu.pending_yield, "nothing to wait for");
    }

    #[test]
    fn a_peer_that_closed_is_end_of_file_and_not_a_connection() {
        let (mut cpu, listener) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        let (client, server) = connected_pair(&mut cpu, listener);

        // Bytes already sent survive the sender: they are the receiver's now.
        assert_eq!(send_on(&mut cpu, client, &[1, 2, 3]), (3, 0));
        write_request(&mut cpu, 26, &client.to_le_bytes());
        cpu.bsd_request(TLS, 9, Some(26)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "close");
        assert_eq!(recv_on(&mut cpu, server, 0x20), ((3, 0), vec![1, 2, 3]));

        // Then end-of-file, which is a read of zero bytes and not an error —
        // a caller told EAGAIN forever waits forever.
        assert_eq!(recv_on(&mut cpu, server, 0x20).0, (0, 0));
        // And writing into it is a broken pipe rather than a silent success.
        assert_eq!(send_on(&mut cpu, server, &[4]), (-1, super::BSD_EPIPE));
    }

    #[test]
    fn bsd_connects_to_this_console_and_refuses_everywhere_else() {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();

        // A loopback port nothing is listening on is refused, exactly as a
        // remote address is: the address being local is not the same as
        // something being there.
        place(&mut cpu, SCRATCH, &loopback_sockaddr(9999));
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ECONNREFUSED), "nothing is listening");

        // And an address off this console has no route at all.
        let mut remote = loopback_sockaddr(53);
        remote[4..8].copy_from_slice(&[8, 8, 8, 8]);
        place(&mut cpu, SCRATCH, &remote);
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, super::BSD_ECONNREFUSED), "off the console");

        // A second connect on a socket that already has a peer is the
        // caller's mistake, not another connection.
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
        // FreeBSD's sockaddr_in: a length byte and a family byte, then the
        // port and the address in network order.
        assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), 16);
        assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), 2, "AF_INET");
        assert_eq!(cpu.read_bytes(BUFFER + 4, 4), super::NIFM_LOCAL_IP.to_vec());
    }
}
