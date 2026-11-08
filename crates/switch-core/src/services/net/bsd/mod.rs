//! `bsd:u`/`bsd:s`, the socket service. Loopback between sockets of this process works;
//! with the network on, stream connections outside it go to the host through [`NetBridge`].

use std::collections::VecDeque;

use self::datagram::{Datagram, MAX_DATAGRAM};
use self::stream::{Stream, StreamState, SEND_WINDOW};
use super::nifm::NIFM_LOCAL_IP;
use crate::cpu::Cpu;
use crate::services::netbridge::{NetCommand, NetEvent, NetFailure, Source, StreamId};
use crate::trace::Level;
use crate::Result;

/// One open `bsd:u` socket. It can only connect to sockets of this process.
mod datagram;
mod mmsg;
mod stream;
#[cfg(test)]
pub(in crate::services::net) mod tests;

/// What a receive comes to: the result and errno, the bytes, where a datagram came from, and
/// whether it was cut to fit.
pub(super) struct Receipt {
    pub ret: i32,
    pub errno: i32,
    pub data: Vec<u8>,
    pub source: Option<Vec<u8>>,
    pub truncated: bool,
}

impl Receipt {
    fn failed(errno: i32) -> Receipt {
        Receipt {
            ret: -1,
            errno,
            data: Vec::new(),
            source: None,
            truncated: false,
        }
    }

    pub(super) fn read(data: Vec<u8>, source: Option<Vec<u8>>, truncated: bool) -> Receipt {
        Receipt {
            ret: data.len() as i32,
            errno: 0,
            data,
            source,
            truncated,
        }
    }
}

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
    /// What a datagram socket exchanges with the host.
    pub datagram: Datagram,
    /// For each thread blocked in a call on this socket, when the call gives up, in guest
    /// cycles; `u64::MAX` for never. Set by the call's first run and cleared by its reply.
    pub waits: Vec<(usize, u64)>,
    /// `shutdown(SHUT_RD)`: what is queued is gone and reads are end-of-file.
    pub shut_read: bool,
    /// `shutdown(SHUT_WR)`: sends are `EPIPE`.
    pub shut_write: bool,
    /// The host listener taking connections for this one, with the network on.
    pub host_listener: Option<StreamId>,
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
            datagram: Datagram::default(),
            waits: Vec::new(),
            shut_read: false,
            shut_write: false,
            host_listener: None,
        }
    }

    /// The host stream this socket's connection runs on, and whether it has opened.
    pub(crate) fn host_stream(&self) -> Option<(StreamId, bool)> {
        self.stream
            .as_ref()
            .map(|stream| (stream.id, stream.state == StreamState::Open))
    }

    /// Readable: has bytes, a pending connection, a closed peer or a shut reading side (EOF),
    /// or an error to report.
    fn readable(&self) -> bool {
        !self.rx.is_empty()
            || !self.datagram.received.is_empty()
            || !self.incoming.is_empty()
            || self.peer_closed
            || self.shut_read
            || self
                .stream_state()
                .is_some_and(|s| matches!(s, StreamState::Failed(_) | StreamState::Reset { .. }))
    }

    fn stream_state(&self) -> Option<StreamState> {
        self.stream.as_ref().map(|stream| stream.state)
    }

    /// Writable: any datagram socket, a live loopback connection, a host connection with room
    /// in its send window, or one with an error a send would report.
    fn writable(&self) -> bool {
        self.kind == BSD_SOCK_DGRAM
            || (self.peer.is_some() && !self.peer_closed)
            || self
                .stream
                .as_ref()
                .is_some_and(|stream| match stream.state {
                    StreamState::Open => stream.in_flight < SEND_WINDOW,
                    StreamState::Failed(_) | StreamState::Reset { .. } => true,
                    StreamState::Connecting => false,
                })
    }
}

/// `bsd` errnos in Linux's numbering (`EAGAIN` is 11): libnx converts them to newlib's with a table
/// indexed that way ("Nintendo actually used the Linux errno definitions for their FreeBSD build").
const BSD_ENOENT: i32 = 2;

const BSD_EBADF: i32 = 9;

const BSD_EINVAL: i32 = 22;

const BSD_EMSGSIZE: i32 = 90;

const BSD_EADDRINUSE: i32 = 98;

const BSD_ECONNRESET: i32 = 104;

const BSD_EAGAIN: i32 = 11;

const BSD_EPIPE: i32 = 32;

const BSD_ENETUNREACH: i32 = 101;

const BSD_ETIMEDOUT: i32 = 110;

const BSD_EHOSTUNREACH: i32 = 113;

const BSD_EINPROGRESS: i32 = 115;

const BSD_EALREADY: i32 = 114;

const BSD_EISCONN: i32 = 106;

const BSD_ENOTCONN: i32 = 107;

const BSD_ECONNREFUSED: i32 = 111;

/// `SOCK_DGRAM`.
pub(super) const BSD_SOCK_DGRAM: u32 = 2;

/// `AF_INET`, in the `sin_family` byte of Horizon's `sockaddr`.
pub(super) const BSD_AF_INET: u8 = 2;

/// `sizeof(sockaddr_in)`, also its `sin_len`.
pub(super) const BSD_SOCKADDR_IN_LEN: usize = 16;

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

/// Whether a datagram to `ip` has somewhere to go outside the console: not multicast (224/4),
/// broadcast or the reserved class E, which no host carries and a guest probing for mDNS or
/// SSDP would otherwise cost a stream each time.
fn is_carried(ip: [u8; 4]) -> bool {
    !is_local_ip(ip) && ip[0] < 224
}

/// `FIONBIO` as FreeBSD and Linux number it, `F_GETFL`/`F_SETFL`, and `O_NONBLOCK`, which
/// is Linux's: a guest's `fcntl` stored 0x800.
const BSD_FIONBIO: u32 = 0x8004_667E;

const BSD_FIONBIO_LINUX: u32 = 0x5421;

const BSD_F_GETFL: u32 = 3;

const BSD_F_SETFL: u32 = 4;

const BSD_O_NONBLOCK: u32 = 0x0800;

/// `SOL_SOCKET`, `SO_RCVTIMEO` and `SO_ERROR`, FreeBSD's.
const BSD_SOL_SOCKET: u32 = 0xFFFF;

const BSD_SO_REUSEADDR: u32 = 0x0004;

const BSD_SO_REUSEPORT: u32 = 0x0200;

const BSD_SO_SNDTIMEO: u32 = 0x1005;

const BSD_SO_RCVTIMEO: u32 = 0x1006;

const BSD_SO_ERROR: u32 = 0x1007;

/// `recv` flags, FreeBSD's: libnx passes the guest's through unchanged.
const BSD_MSG_OOB: u32 = 0x01;

const BSD_MSG_PEEK: u32 = 0x02;

const BSD_MSG_WAITALL: u32 = 0x40;

const BSD_MSG_DONTWAIT: u32 = 0x80;

/// `FIONREAD`, `_IOR('f', 127, int)`: bytes waiting to be read.
const BSD_FIONREAD: u32 = 0x4004_667F;

/// Most bytes of an option kept: a 64-bit `timeval`.
const BSD_MAX_OPTION: u32 = 16;

impl Cpu {
    /// `bsd:u`/`bsd:s`, the socket service. The only peer is this console: loopback
    /// connections work, everything else is refused immediately. Errnos are Linux's.
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
            // RegisterClient(BsdInitConfig, pid, tmem_size, tmem) -> u64, and
            // RegisterClientShared, the same without the transfer memory.
            Some(0) | Some(33) => self.write_ipc_response(tls, 0, &[], &0u64.to_le_bytes(), &[]),
            // Open(flags, path): only `/dev/bpf` on a console, and there is no device here.
            Some(4) => self.bsd_reply(tls, -1, BSD_ENOENT),
            // Sysctl(name, new, old) -> old length: no MIB answers.
            Some(7) => self.bsd_reply_len(tls, -1, BSD_ENOENT, 0),
            // RecvMMsg(fd, vlen, flags, padding, timespec) and SendMMsg(fd, vlen, flags).
            Some(29) => {
                let (fd, count, flags) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                self.bsd_receive_messages(tls, fd, count, flags)
            }
            Some(30) => {
                let (fd, count, flags) = (word(self, 0) as i32, word(self, 1), word(self, 2));
                self.bsd_send_messages(tls, fd, count, flags)
            }
            // Cmd36(pid, version, pid_reserved) -> u32: a stub that returns 0 on the Switch.
            Some(36) => self.write_ipc_response(tls, 0, &[], &0u32.to_le_bytes(), &[]),
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
            Some(8) => {
                let (fd, flags) = (word(self, 0) as i32, word(self, 1));
                self.bsd_receive(tls, fd, flags, None)
            }
            Some(25) => {
                let fd = word(self, 0) as i32;
                self.bsd_receive(tls, fd, 0, None)
            }
            // RecvFrom(fd, flags), with the sender's address in the second buffer.
            Some(9) => {
                let (fd, flags) = (word(self, 0) as i32, word(self, 1));
                self.bsd_receive(tls, fd, flags, Some(1))
            }
            // Send(fd, flags) / SendTo(fd, flags, sockaddr) / Write(fd). See [`Cpu::bsd_send`].
            Some(10) => {
                let (fd, flags) = (word(self, 0) as i32, word(self, 1));
                self.bsd_send(tls, fd, flags, None)
            }
            Some(24) => {
                let fd = word(self, 0) as i32;
                self.bsd_send(tls, fd, 0, None)
            }
            Some(11) => {
                let (fd, flags) = (word(self, 0) as i32, word(self, 1));
                self.bsd_send(tls, fd, flags, Some(1))
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
                    if self.bsd_waits(fd, None, false) {
                        return Ok(());
                    }
                    return self.bsd_reply(tls, -1, BSD_EAGAIN);
                };
                self.bsd_end_wait(fd);
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
                if self.bsd_port_taken(fd, &address) {
                    return self.bsd_reply(tls, -1, BSD_EADDRINUSE);
                }
                let address = self.bsd_normalize_bind(address);
                let outside = sockaddr_in(&address).is_some_and(|(ip, _)| ip != BSD_LOOPBACK_IP);
                if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
                    socket.bound = address;
                }
                // A datagram socket that may hear from outside gets its host socket now, so it
                // hears before it has sent anything.
                if outside {
                    self.bsd_udp_host(fd);
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
                            && socket.datagram.default_peer.is_none()
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
                    self.bsd_take_stream_error(fd).to_le_bytes().to_vec()
                } else {
                    self.process()
                        .bsd_socket_options
                        .get(&(fd, level, option))
                        .cloned()
                        .unwrap_or_else(|| vec![0; 4])
                };
                let mut written = 0;
                if let Some((addr, size)) = self.ipc_output_buffer(tls, 0) {
                    if size as usize >= value.len() {
                        for (offset, &byte) in value.iter().enumerate() {
                            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
                        }
                        written = value.len() as u32;
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
                        self.bsd_listen_outside(fd);
                        self.bsd_reply(tls, 0, 0)
                    }
                }
            }
            // Ioctl(fd, request, bufcount): FIONBIO, folded into the `fcntl` flags word, and
            // FIONREAD, into the first output buffer.
            Some(19) if word(self, 1) == BSD_FIONREAD => {
                let fd = word(self, 0) as i32;
                let Some(socket) = self.process().bsd_sockets.get(&fd) else {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                };
                let queued = socket.rx.len() + socket.datagram.queued_bytes();
                let queued = u32::try_from(queued).unwrap_or(u32::MAX);
                match self.ipc_output_buffer(tls, 0) {
                    Some((addr, size)) if size >= 4 => {
                        self.mem.write_u32(addr, queued)?;
                        self.bsd_reply(tls, 0, 0)
                    }
                    _ => self.bsd_reply(tls, -1, BSD_EINVAL),
                }
            }
            Some(19) => {
                let (fd, request) = (word(self, 0) as i32, word(self, 1));
                let nonblocking = match self.ipc_input_buffer(tls, 0) {
                    Some((addr, size)) if size >= 4 => self.mem.read_u32(addr).unwrap_or(0) != 0,
                    _ => false,
                };
                match self.process_mut().bsd_sockets.get_mut(&fd) {
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                    Some(socket) if request == BSD_FIONBIO || request == BSD_FIONBIO_LINUX => {
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
                    Some((addr, size)) if size >= 4 => {
                        self.read_bytes(addr, size.min(BSD_MAX_OPTION))
                    }
                    _ => vec![0; 4],
                };
                self.process_mut()
                    .bsd_socket_options
                    .insert((fd, level, option), value);
                self.bsd_reply(tls, 0, 0)
            }
            // Shutdown(fd, how): `SHUT_WR` and `SHUT_RDWR` end the peer's reading.
            Some(22) => {
                const SHUT_RD: u32 = 0;
                const SHUT_WR: u32 = 1;
                const SHUT_RDWR: u32 = 2;
                let (fd, how) = (word(self, 0) as i32, word(self, 1));
                let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) else {
                    return self.bsd_reply(tls, -1, BSD_EBADF);
                };
                if how > SHUT_RDWR {
                    return self.bsd_reply(tls, -1, BSD_EINVAL);
                }
                if how != SHUT_WR {
                    socket.shut_read = true;
                    socket.rx.clear();
                    socket.datagram.received.clear();
                }
                if how != SHUT_RD {
                    socket.shut_write = true;
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
                match self.bsd_duplicate(fd) {
                    Some(duplicate) => self.bsd_reply(tls, duplicate, 0),
                    None => self.bsd_reply(tls, -1, BSD_EBADF),
                }
            }
            _ => self.unimplemented_command(tls, "bsd:u", cmd_id),
        }
    }

    /// A new descriptor with `fd`'s family, address and flags, and none of its connection.
    pub(crate) fn bsd_duplicate(&mut self, fd: i32) -> Option<i32> {
        let socket = self.process().bsd_sockets.get(&fd)?;
        let mut copy = BsdSocket::new(socket.domain, socket.kind);
        copy.bound = socket.bound.clone();
        copy.flags = socket.flags;
        copy.listening = socket.listening;
        let duplicate = self.alloc_bsd_fd();
        self.process_mut().bsd_sockets.insert(duplicate, copy);
        Some(duplicate)
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
        if let Some(address) = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.datagram.default_address())
        {
            return address;
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
                return self.bsd_connect_datagram(tls, fd, address)
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

    /// `Send`/`SendTo`/`Write`, from the request's buffers. See [`Cpu::bsd_send_bytes`].
    fn bsd_send(&mut self, tls: u32, fd: i32, flags: u32, destination: Option<u32>) -> Result<()> {
        let data = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let address = destination
            .and_then(|index| self.ipc_input_buffer(tls, index))
            .map(|(addr, size)| self.read_bytes(addr, size.min(0x80)));
        let Some((ret, errno)) = self.bsd_send_bytes(fd, data, address.as_deref(), flags) else {
            return Ok(());
        };
        self.bsd_end_wait(fd);
        self.bsd_reply(tls, ret, errno)
    }

    /// Send `data` on `fd`, to `address` when one is given. `(ret, errno)`, or `None` while a
    /// blocking send waits for room on a host connection (the call writes no reply).
    pub(super) fn bsd_send_bytes(
        &mut self,
        fd: i32,
        data: Vec<u8>,
        address: Option<&[u8]>,
        flags: u32,
    ) -> Option<(i32, i32)> {
        let peer = match self.process().bsd_sockets.get(&fd) {
            None => return Some((-1, BSD_EBADF)),
            Some(socket) if socket.shut_write => return Some((-1, BSD_EPIPE)),
            Some(socket) if socket.stream.is_some() => {
                return self.bsd_send_outside(fd, data, flags);
            }
            Some(socket) if socket.peer_closed && socket.kind != BSD_SOCK_DGRAM => {
                return Some((-1, BSD_EPIPE));
            }
            Some(socket) => match socket.peer {
                Some(peer) => peer,
                None => return Some(self.bsd_send_datagram(fd, data, address)),
            },
        };
        let sent = data.len() as i32;
        if let Some(peer) = self.process_mut().bsd_sockets.get_mut(&peer) {
            if !peer.shut_read {
                peer.rx.extend(data);
            }
        }
        Some((sent, 0))
    }

    /// A send from a datagram socket with no loopback peer: to this console it is delivered
    /// here, outside it goes to the host, and anywhere else (multicast, broadcast) the link
    /// takes it and drops it.
    fn bsd_send_datagram(&mut self, fd: i32, data: Vec<u8>, address: Option<&[u8]>) -> (i32, i32) {
        let Some(socket) = self.process().bsd_sockets.get(&fd) else {
            return (-1, BSD_EBADF);
        };
        let default_peer = socket.datagram.default_peer;
        if socket.kind != BSD_SOCK_DGRAM {
            return self.bsd_unconnected(fd);
        }
        let Some((ip, port)) = address.and_then(sockaddr_in).or(default_peer) else {
            return self.bsd_unconnected(fd);
        };
        if data.len() > MAX_DATAGRAM {
            return (-1, BSD_EMSGSIZE);
        }
        if is_local_ip(ip) {
            return self.bsd_send_local_datagram(fd, data, ip, port);
        }
        if is_carried(ip) && port != 0 {
            return self.bsd_send_outside_datagram(fd, data, ip, port);
        }
        (data.len() as i32, 0)
    }

    /// `Recv`/`RecvFrom`/`Read`, with the sender's address when `address_buffer` names one.
    fn bsd_receive(
        &mut self,
        tls: u32,
        fd: i32,
        flags: u32,
        address_buffer: Option<u32>,
    ) -> Result<()> {
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return self.bsd_reply(tls, -1, BSD_EINVAL);
        };
        let Some(receipt) = self.bsd_receive_bytes(fd, size as usize, flags) else {
            return Ok(());
        };
        self.bsd_end_wait(fd);
        for (offset, &byte) in receipt.data.iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
        }
        let Some(index) = address_buffer else {
            return self.bsd_reply(tls, receipt.ret, receipt.errno);
        };
        let address = match receipt.source {
            Some(source) => source,
            None if receipt.ret >= 0 => self.bsd_peer_address(fd),
            None => Vec::new(),
        };
        let written = self.bsd_write_address(tls, index, &address)?;
        self.bsd_reply_len(tls, receipt.ret, receipt.errno, written)
    }

    /// Take up to `room` of the peer's bytes, or one datagram with its source. An empty live
    /// queue waits on a blocking socket (`None`: the call writes no reply) and is `EAGAIN` plus
    /// a yield on a non-blocking one, under `MSG_DONTWAIT`, or once the wait has timed out.
    /// `MSG_PEEK` leaves what it read queued; `MSG_WAITALL` on a stream socket waits for the
    /// whole of `room`, and returns what there is when the peer closes or the wait times out.
    /// A reset connection reports `ECONNRESET` once, then reads end-of-file.
    pub(super) fn bsd_receive_bytes(
        &mut self,
        fd: i32,
        room: usize,
        flags: u32,
    ) -> Option<Receipt> {
        // There is never urgent data to read.
        if flags & BSD_MSG_OOB != 0 {
            return Some(Receipt::failed(BSD_EINVAL));
        }
        if let Some(errno) = self.bsd_take_reset(fd) {
            return Some(Receipt::failed(errno));
        }
        if self
            .process()
            .bsd_sockets
            .get(&fd)
            .is_some_and(|socket| socket.shut_read)
        {
            return Some(Receipt::read(Vec::new(), None, false));
        }
        let peek = flags & BSD_MSG_PEEK != 0;
        if let Some(received) = self.bsd_receive_datagram(fd, room, peek) {
            return Some(received);
        }
        let socket = match self.process().bsd_sockets.get(&fd) {
            None => return Some(Receipt::failed(BSD_EBADF)),
            // A stream socket needs a connection; a datagram socket just has an empty queue.
            Some(socket)
                if socket.kind != BSD_SOCK_DGRAM
                    && socket.peer.is_none()
                    && !socket.peer_closed
                    && socket.stream_state() != Some(StreamState::Open) =>
            {
                return Some(Receipt::failed(BSD_ENOTCONN));
            }
            Some(socket) => socket,
        };
        let (queued, closed) = (socket.rx.len(), socket.peer_closed);
        let wait_all = flags & BSD_MSG_WAITALL != 0 && !peek && socket.kind != BSD_SOCK_DGRAM;
        let short = queued == 0 || (wait_all && queued < room);
        if short && !closed {
            let timeout = self.bsd_option_timeout(fd, BSD_SO_RCVTIMEO);
            if self.bsd_waits(fd, timeout, flags & BSD_MSG_DONTWAIT != 0) {
                return None;
            }
            if queued == 0 {
                self.pending_yield = true;
                return Some(Receipt::failed(BSD_EAGAIN));
            }
        }
        // With nothing queued, the peer has closed: end-of-file is a read of zero bytes.
        let take = room.min(queued);
        let socket = self.process_mut().bsd_sockets.get_mut(&fd)?;
        let data: Vec<u8> = if peek {
            socket.rx.iter().take(take).copied().collect()
        } else {
            socket.rx.drain(..take).collect()
        };
        let stream = socket.stream.as_ref().map(|stream| stream.id);
        if let Some(stream) = stream.filter(|_| !peek && take > 0) {
            let bytes = take as u32;
            self.net.outbox.push(NetCommand::Consumed { stream, bytes });
        }
        Some(Receipt::read(data, None, false))
    }

    /// Whether a call that found nothing on `fd` waits for it. A blocking socket does: the
    /// call writes no reply and is run again, until there is something or `timeout_ns` of guest
    /// time has passed (never, for `None`). A non-blocking one, a call told not to wait, or a
    /// wait that has timed out does not, and the call answers with what it has. Each thread
    /// blocked on the socket has a deadline of its own.
    fn bsd_waits(&mut self, fd: i32, timeout_ns: Option<i64>, dont_wait: bool) -> bool {
        if dont_wait {
            return false;
        }
        let fresh = timeout_ns
            .and_then(|ns| self.wait_deadline(ns))
            .unwrap_or(u64::MAX);
        let (now, thread) = (self.cycles, self.current_thread);
        let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) else {
            return false;
        };
        if socket.flags & BSD_O_NONBLOCK != 0 {
            return false;
        }
        let until = match socket.waits.iter().find(|(waiting, _)| *waiting == thread) {
            Some(&(_, until)) => until,
            None => {
                socket.waits.push((thread, fresh));
                fresh
            }
        };
        if now >= until {
            socket.waits.retain(|(waiting, _)| *waiting != thread);
            return false;
        }
        self.retry_after(BSD_WAIT_SLICE_NS);
        true
    }

    /// The blocking call this thread made on `fd` has its answer.
    fn bsd_end_wait(&mut self, fd: i32) {
        let thread = self.current_thread;
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            socket.waits.retain(|(waiting, _)| *waiting != thread);
        }
    }

    /// `SO_RCVTIMEO` in nanoseconds; see [`Cpu::bsd_option_timeout`].
    #[cfg(test)]
    fn bsd_receive_timeout(&self, fd: i32) -> Option<i64> {
        self.bsd_option_timeout(fd, BSD_SO_RCVTIMEO)
    }

    /// A `timeval` option (`SO_RCVTIMEO`, `SO_SNDTIMEO`) in nanoseconds, `None` when unset or
    /// zero. The `timeval` is 16 bytes for a 64-bit guest and for a 32-bit FreeBSD one, whose
    /// `tv_usec` is a 32-bit `long` padded to 8; reading `tv_usec` as 32 bits covers both. An
    /// 8-byte one is two 32-bit words.
    pub(super) fn bsd_option_timeout(&self, fd: i32, option: u32) -> Option<i64> {
        let raw = self
            .process()
            .bsd_socket_options
            .get(&(fd, BSD_SOL_SOCKET, option))?;
        let word = |at: usize| {
            raw.get(at..at + 4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        };
        let (seconds, micros) = if raw.len() >= 16 {
            let seconds = i64::from_le_bytes(raw[..8].try_into().unwrap());
            (seconds, i64::from(word(8)?))
        } else {
            (i64::from(word(0)?), i64::from(word(4)?))
        };
        let ns = seconds
            .saturating_mul(1_000_000_000)
            .saturating_add(micros.saturating_mul(1_000));
        (ns > 0).then_some(ns)
    }

    /// Whether binding `fd` to `address` would take a port another socket of its kind holds on
    /// an overlapping address. Two sockets that both set `SO_REUSEADDR`, or both
    /// `SO_REUSEPORT`, may share one.
    fn bsd_port_taken(&self, fd: i32, address: &[u8]) -> bool {
        let Some((ip, port)) = sockaddr_in(address).filter(|&(_, port)| port != 0) else {
            return false;
        };
        let Some(kind) = self
            .process()
            .bsd_sockets
            .get(&fd)
            .map(|socket| socket.kind)
        else {
            return false;
        };
        let options = &self.process().bsd_socket_options;
        let set = |fd: i32, option: u32| {
            options
                .get(&(fd, BSD_SOL_SOCKET, option))
                .is_some_and(|value| value.iter().any(|&byte| byte != 0))
        };
        let shares = |other: i32| {
            [BSD_SO_REUSEADDR, BSD_SO_REUSEPORT]
                .iter()
                .any(|&option| set(fd, option) && set(other, option))
        };
        self.process().bsd_sockets.iter().any(|(&other, socket)| {
            other != fd
                && socket.kind == kind
                && sockaddr_in(&socket.bound).is_some_and(|(held, held_port)| {
                    held_port == port && (held == ip || held == BSD_ANY_IP || ip == BSD_ANY_IP)
                })
                && !shares(other)
        })
    }

    /// With the network on, a listener bound to an address outside loopback also listens on the
    /// host, which hands it connections when the relay takes inbound traffic.
    fn bsd_listen_outside(&mut self, fd: i32) {
        let Some(socket) = self.process().bsd_sockets.get(&fd) else {
            return;
        };
        let port = sockaddr_in(&socket.bound)
            .filter(|&(ip, port)| ip != BSD_LOOPBACK_IP && port != 0)
            .map(|(_, port)| port);
        let (Some(port), true, None, false) = (
            port,
            self.net.enabled,
            socket.host_listener,
            socket.kind == BSD_SOCK_DGRAM,
        ) else {
            return;
        };
        let id = self.net.open_stream();
        self.log(
            Level::Info,
            &format!("[net] listen fd {fd} on port {port} as stream {id}"),
        );
        self.net
            .outbox
            .push(NetCommand::Listen { stream: id, port });
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            socket.host_listener = Some(id);
        }
    }

    /// The error for a send with no destination.
    fn bsd_unconnected(&self, fd: i32) -> (i32, i32) {
        match self.process().bsd_sockets.get(&fd) {
            None => (-1, BSD_EBADF),
            Some(socket) if socket.kind == BSD_SOCK_DGRAM => (-1, BSD_ENETUNREACH),
            Some(_) => (-1, BSD_ENOTCONN),
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
        // A host connection whose peer only finished sending can still be written to.
        let hung_up = match socket.stream_state() {
            Some(StreamState::Reset { .. }) => true,
            Some(_) => socket.peer_closed && socket.shut_write,
            None => socket.peer_closed,
        };
        if hung_up && socket.rx.is_empty() {
            revents |= POLLHUP;
        }
        if matches!(
            socket.stream_state(),
            Some(StreamState::Failed(_) | StreamState::Reset { reported: false })
        ) {
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
            // Still connecting is fine: the relay acts on it once the connection opens.
            .filter(|stream| matches!(stream.state, StreamState::Open | StreamState::Connecting))
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
        for stream in socket.datagram.host.into_iter().chain(socket.host_listener) {
            self.net.outbox.push(NetCommand::Close { stream });
        }
        if let Some(stream) = socket
            .stream
            .as_ref()
            .filter(|s| !matches!(s.state, StreamState::Failed(_)))
        {
            self.log(
                Level::Info,
                &format!(
                    "[net] close fd {fd} stream {} while {:?}",
                    stream.id, stream.state
                ),
            );
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
}
