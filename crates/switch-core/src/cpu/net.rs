//! The networking services, and the answer they all give.
//!
//! There is no network behind any of this. `bsd` sockets can be created,
//! configured, bound and listened on — every local operation a real one
//! supports — and can never carry a byte; `sfdnsres` resolves nothing;
//! `nifm` reports a console that is connected to a LAN with no route off it;
//! `ssl` builds contexts that never handshake.
//!
//! That is deliberate: an *empty* network is a state a real console reaches
//! and every caller has a path for. A failure is the path built for hardware
//! that broke.

use super::Cpu;
use crate::Result;

/// One open `bsd:u` socket.
///
/// A socket here can be created, configured, bound and listened on — every
/// local operation a real one supports — and can never carry a byte, because
/// there is no network behind this service. See [`Cpu::bsd_request`].
#[derive(Debug, Clone)]
pub(crate) struct BsdSocket {
    /// The address family and socket type it was created with. The family is
    /// carried for `DuplicateSocket`; the type decides which "went nowhere"
    /// errno the data path reports.
    pub domain: u32,
    pub kind: u32,
    /// The raw `sockaddr` bytes `bind` was given, which `GetSockName` reports
    /// back.
    pub bound: Vec<u8>,
    /// The flags word `fcntl(F_SETFL)` set, stored verbatim so `F_GETFL` hands
    /// back exactly what the guest wrote.
    pub flags: u32,
    /// Whether `listen` was called — an `accept` on a socket that never
    /// listened is a different error from one nobody has connected to.
    pub listening: bool,
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
                // `timeout == 0` is an explicit non-blocking probe, and comes
                // back at once on hardware too.
                self.pending_yield = timeout != 0;
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
