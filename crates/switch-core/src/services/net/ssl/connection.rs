//! `ISslConnection`: TLS over a socket the host carries. The host runs the protocol (see
//! [`NetCommand::StartTls`]); this object keeps the connection's settings and where its handshake
//! stands, and moves plaintext between the guest and the socket's queues.

use std::sync::Arc;

use super::*;
use crate::services::net::BsdSocket;
use crate::services::netbridge::{NetCommand, NetEvent, StreamId, TlsFailure};
use crate::trace::Level;

/// A `Result` of module 123, the SSL service's.
pub(super) const fn result(description: u32) -> u32 {
    123 | (description << 9)
}

pub(super) const RESULT_INTERNAL_LOGIC: u32 = result(13);
const RESULT_INVALID_SOCKET: u32 = result(116);
const RESULT_WOULD_BLOCK: u32 = result(204);
const RESULT_IO_TIMEOUT: u32 = result(205);
const RESULT_NO_CONNECTION: u32 = result(208);
const RESULT_CONNECTION_RESET: u32 = result(209);
const RESULT_SOCKET_SHUTDOWN: u32 = result(211);
const RESULT_INVALID_CERT_DOMAIN: u32 = result(301);
const RESULT_EXPIRED_CERT: u32 = result(303);
const RESULT_BAD_CERT: u32 = result(307);
const RESULT_UNKNOWN_CA: u32 = result(308);
const RESULT_ALERT_HANDSHAKE_FAILURE: u32 = result(1507);
const RESULT_ALERT_PROTOCOL_VERSION: u32 = result(1520);

/// `VerifyOption` bits.
const VERIFY_HOST_NAME: u32 = 1 << 1;

/// `IoMode`: only non-blocking returns at once.
const IO_NONBLOCKING: u32 = 2;

/// `OptionType`s a connection keeps.
const OPTION_DO_NOT_CLOSE_SOCKET: usize = 0;
const OPTION_GET_SERVER_CERT_CHAIN: usize = 1;
const OPTION_ENABLE_ALPN: usize = 3;
const OPTIONS: usize = 4;

/// `AlpnProtoState`.
const ALPN_NO_SUPPORT: u32 = 0;
const ALPN_NEGOTIATED: u32 = 1;

/// How long a blocking call waits for the host: the real service's five minutes.
const BLOCKING_WAIT_NS: i64 = 300_000_000_000;

/// How long a call that cannot finish sleeps before it is run again.
const RETRY_SLICE_NS: i64 = 1_000_000;

/// A certificate chain's header: "CertChMN", then the count, then 8-byte entries from +0x10.
const CERT_CHAIN_MAGIC: u64 = 0x4E4D_6843_7472_6543;

/// `CipherInfo`: a 0x40-byte cipher name and an 8-byte protocol version.
const CIPHER_INFO_SIZE: u32 = 0x48;

/// `TrustedCertStatus::EnabledTrusted`.
const CERT_ENABLED_TRUSTED: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Handshake {
    Idle,
    /// Asked of the host, not answered yet.
    Pending,
    Done,
    Failed(u32),
}

#[derive(Debug, Clone)]
pub(crate) struct SslConnection {
    /// The socket the guest handed over; its connection is the one TLS runs on.
    fd: Option<i32>,
    host: String,
    verify: u32,
    io_mode: u32,
    session_cache: u32,
    renegotiation: u32,
    io_timeout_ms: u32,
    options: [bool; OPTIONS],
    alpn: Vec<Vec<u8>>,
    handshake: Handshake,
    stream: Option<StreamId>,
    /// When a call that waits gives up, set by its first run.
    deadline: Option<u64>,
    /// The result `GetVerifyCertError` reports, once.
    cert_error: u32,
    chain: Vec<Vec<u8>>,
    protocol: String,
    cipher: String,
    negotiated: Option<Vec<u8>>,
}

impl Default for SslConnection {
    /// What a connection is when created: verify the chain and the name, block.
    fn default() -> SslConnection {
        SslConnection {
            fd: None,
            host: String::new(),
            verify: 0x3,
            io_mode: 1,
            session_cache: 0,
            renegotiation: 0,
            io_timeout_ms: 0,
            options: [false; OPTIONS],
            alpn: Vec::new(),
            handshake: Handshake::Idle,
            stream: None,
            deadline: None,
            cert_error: 0,
            chain: Vec::new(),
            protocol: String::new(),
            cipher: String::new(),
            negotiated: None,
        }
    }
}

impl SslConnection {
    /// The server's certificates as `GetServerCertChain` lays them out, or just the leaf.
    fn certificates(&self) -> (Vec<u8>, u32) {
        if self.options[OPTION_GET_SERVER_CERT_CHAIN] {
            let count = self.chain.len() as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&CERT_CHAIN_MAGIC.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            let mut offset = 0x10 + 8 * self.chain.len() as u32;
            for der in &self.chain {
                out.extend_from_slice(&(der.len() as u32).to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                offset += der.len() as u32;
            }
            for der in &self.chain {
                out.extend_from_slice(der);
            }
            (out, count)
        } else {
            match self.chain.first() {
                Some(leaf) => (leaf.clone(), 1),
                None => (Vec::new(), 0),
            }
        }
    }
}

impl Cpu {
    pub(super) fn ssl_connection_request(
        &mut self,
        tls: u32,
        key: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if !self.process().ssl_connections.contains_key(&key) {
            return self.unimplemented_command(tls, "ssl:connection", cmd_id);
        }
        let data = self.ipc_request_data(tls);
        let word =
            |cpu: &Cpu, index: u32| cpu.mem.read_u32(data.wrapping_add(index * 4)).unwrap_or(0);
        let reply = |cpu: &mut Cpu, bytes: &[u8]| cpu.write_ipc_response(tls, 0, &[], bytes, &[]);
        let fail = |cpu: &mut Cpu, code: u32| cpu.write_ipc_response(tls, code, &[], &[], &[]);
        let connection = |cpu: &Cpu| cpu.process().ssl_connections[&key].clone();
        let has_socket = connection(self).fd.is_some();
        match cmd_id {
            // SetSocketDescriptor(s32 fd) -> s32 duplicate.
            Some(0) => {
                let fd = word(self, 0) as i32;
                if has_socket {
                    return fail(self, RESULT_INTERNAL_LOGIC);
                }
                let Some(duplicate) = self.bsd_duplicate(fd) else {
                    return fail(self, RESULT_INVALID_SOCKET);
                };
                self.connection_mut(key).fd = Some(fd);
                reply(self, &duplicate.to_le_bytes())
            }
            // SetHostName(buffer): `nsd` swaps in the environment for `%`.
            Some(1) => {
                let host = match self.ipc_input_buffer(tls, 0) {
                    Some((_, size)) if size > 0xFF => return fail(self, RESULT_INTERNAL_LOGIC),
                    Some((addr, size)) => self.read_string(addr, size).replace('%', "lp1"),
                    None => String::new(),
                };
                self.connection_mut(key).host = host;
                reply(self, &[])
            }
            // SetVerifyOption(u32).
            Some(2) => {
                self.connection_mut(key).verify = word(self, 0);
                reply(self, &[])
            }
            // SetIoMode(u32), SetSessionCacheMode, SetRenegotiationMode: kept once a socket is set.
            Some(3 | 17 | 20) if !has_socket => fail(self, RESULT_INTERNAL_LOGIC),
            Some(3) => {
                self.connection_mut(key).io_mode = word(self, 0);
                reply(self, &[])
            }
            Some(17) => {
                self.connection_mut(key).session_cache = word(self, 0);
                reply(self, &[])
            }
            Some(20) => {
                self.connection_mut(key).renegotiation = word(self, 0);
                reply(self, &[])
            }
            // GetSocketDescriptor, GetSessionCacheMode, GetRenegotiationMode need a socket too.
            Some(4 | 18 | 21) if !has_socket => fail(self, RESULT_INTERNAL_LOGIC),
            Some(4) => {
                let fd = connection(self).fd.unwrap_or(-1);
                reply(self, &fd.to_le_bytes())
            }
            Some(18) => reply(self, &connection(self).session_cache.to_le_bytes()),
            Some(21) => reply(self, &connection(self).renegotiation.to_le_bytes()),
            // GetHostName(out buffer) -> u32 length.
            Some(5) => {
                let host = connection(self).host;
                let mut text = host.clone().into_bytes();
                text.push(0);
                self.write_output_buffer(tls, 0, &text);
                reply(self, &(host.len() as u32).to_le_bytes())
            }
            Some(6) => reply(self, &connection(self).verify.to_le_bytes()),
            Some(7) => reply(self, &connection(self).io_mode.to_le_bytes()),
            // DoHandshake, DoHandshakeGetServerCert.
            Some(8) => self.ssl_handshake(tls, key, false),
            Some(9) => self.ssl_handshake(tls, key, true),
            // Read, Peek.
            Some(10) => self.ssl_read(tls, key, false),
            Some(13) => self.ssl_read(tls, key, true),
            Some(11) => self.ssl_write(tls, key),
            // Pending -> s32.
            Some(12) => {
                let pending = self.ssl_socket(key).map_or(0, |socket| socket.rx.len());
                reply(self, &(pending as i32).to_le_bytes())
            }
            Some(14) => self.ssl_poll(tls, key, word(self, 0), word(self, 1)),
            // GetVerifyCertError: what the handshake's verification said, once.
            Some(15) => {
                let error = std::mem::take(&mut self.connection_mut(key).cert_error);
                fail(self, error)
            }
            // GetNeededServerCertBufferSize -> u32.
            Some(16) => {
                let (certificates, _) = connection(self).certificates();
                reply(self, &(certificates.len() as u32).to_le_bytes())
            }
            // FlushSessionCache.
            Some(19) => reply(self, &[]),
            // SetOption(u8 on, OptionType), GetOption(OptionType) -> u8.
            Some(22) => {
                let on = self.mem.read_u8(data).unwrap_or(0) != 0;
                let option = word(self, 1) as usize;
                if option >= OPTIONS || (option == OPTION_DO_NOT_CLOSE_SOCKET && has_socket) {
                    return fail(self, RESULT_INTERNAL_LOGIC);
                }
                self.connection_mut(key).options[option] = on;
                reply(self, &[])
            }
            Some(23) => {
                let option = word(self, 0) as usize;
                let on = connection(self)
                    .options
                    .get(option)
                    .copied()
                    .unwrap_or(false);
                reply(self, &[u8::from(on)])
            }
            // GetVerifyCertErrors(out buffer) -> (u32, u32): none, and the two must agree.
            Some(24) => {
                self.zero_output_buffer(tls, 0);
                reply(self, &[0; 8])
            }
            // GetCipherInfo(u32 = 1, out buffer of CipherInfo).
            Some(25) => {
                let sized = self
                    .ipc_output_buffer(tls, 0)
                    .is_some_and(|(_, size)| size == CIPHER_INFO_SIZE);
                if word(self, 0) != 1 || !sized || !has_socket {
                    return fail(self, RESULT_INTERNAL_LOGIC);
                }
                let connection = connection(self);
                let mut info = vec![0u8; CIPHER_INFO_SIZE as usize];
                let cipher = connection.cipher.as_bytes();
                info[..cipher.len().min(0x3F)].copy_from_slice(&cipher[..cipher.len().min(0x3F)]);
                let protocol = connection.protocol.as_bytes();
                info[0x40..0x40 + protocol.len().min(7)]
                    .copy_from_slice(&protocol[..protocol.len().min(7)]);
                self.write_output_buffer(tls, 0, &info);
                reply(self, &[])
            }
            // SetNextAlpnProto(buffer of { u8 length, bytes }).
            Some(26) => {
                let Some((addr, size)) = self.ipc_input_buffer(tls, 0).filter(|&(_, s)| s >= 2)
                else {
                    return fail(self, RESULT_INTERNAL_LOGIC);
                };
                let raw = self.read_bytes(addr, size);
                let mut protocols = Vec::new();
                let mut at = 0;
                while at < raw.len() {
                    let end = at + 1 + usize::from(raw[at]);
                    let Some(name) = raw.get(at + 1..end) else {
                        return fail(self, RESULT_INTERNAL_LOGIC);
                    };
                    protocols.push(name.to_vec());
                    at = end;
                }
                self.connection_mut(key).alpn = protocols;
                reply(self, &[])
            }
            // GetNextAlpnProto(out buffer) -> (AlpnProtoState, u32 length).
            Some(27) => {
                let connection = connection(self);
                let (state, name) = match &connection.negotiated {
                    Some(name) if connection.options[OPTION_ENABLE_ALPN] => {
                        (ALPN_NEGOTIATED, name.clone())
                    }
                    _ => (ALPN_NO_SUPPORT, Vec::new()),
                };
                self.zero_output_buffer(tls, 0);
                self.write_output_buffer(tls, 0, &name);
                let mut out = state.to_le_bytes().to_vec();
                out.extend_from_slice(&(name.len() as u32).to_le_bytes());
                reply(self, &out)
            }
            // SetPrivateOption: accepted.
            Some(30) => reply(self, &[]),
            // SetIoTimeout(u32 ms), GetIoTimeout.
            Some(34) => {
                self.connection_mut(key).io_timeout_ms = word(self, 0);
                reply(self, &[])
            }
            Some(35) => reply(self, &connection(self).io_timeout_ms.to_le_bytes()),
            _ => self.unimplemented_command(tls, "ssl:connection", cmd_id),
        }
    }

    fn connection_mut(&mut self, key: u64) -> &mut SslConnection {
        self.process_mut()
            .ssl_connections
            .get_mut(&key)
            .expect("the caller checked the connection exists")
    }

    /// The socket a connection runs over.
    fn ssl_socket(&self, key: u64) -> Option<&BsdSocket> {
        let fd = self.process().ssl_connections.get(&key)?.fd?;
        self.process().bsd_sockets.get(&fd)
    }

    fn ssl_reply(&mut self, tls: u32, key: u64, result: u32, data: &[u8]) -> Result<()> {
        self.connection_mut(key).deadline = None;
        self.write_ipc_response(tls, result, &[], data, &[])
    }

    /// A call that cannot finish yet. A non-blocking one says so; a blocking one writes no
    /// reply and is run again until it can, or until `timeout_ns` of guest time has passed.
    fn ssl_wait(&mut self, tls: u32, key: u64, timeout_ns: i64, may_block: bool) -> Result<()> {
        if !may_block {
            return self.ssl_reply(tls, key, RESULT_WOULD_BLOCK, &[]);
        }
        let now = self.cycles;
        let fresh = self.wait_deadline(timeout_ns.max(1)).unwrap_or(now + 1);
        let deadline = *self.connection_mut(key).deadline.get_or_insert(fresh);
        if now >= deadline {
            return self.ssl_reply(tls, key, RESULT_IO_TIMEOUT, &[]);
        }
        self.retry_after(RETRY_SLICE_NS);
        Ok(())
    }

    /// How a call that is not told its own timeout waits: not at all when non-blocking.
    fn ssl_io_wait(&mut self, tls: u32, key: u64) -> Result<()> {
        let connection = &self.process().ssl_connections[&key];
        let blocking = connection.io_mode != IO_NONBLOCKING;
        let timeout = match connection.io_timeout_ms {
            0 => BLOCKING_WAIT_NS,
            ms => i64::from(ms) * 1_000_000,
        };
        self.ssl_wait(tls, key, timeout, blocking)
    }

    /// `DoHandshake` and `DoHandshakeGetServerCert`: asked of the host once, then answered when it has.
    fn ssl_handshake(&mut self, tls: u32, key: u64, with_certificates: bool) -> Result<()> {
        let connection = self.process().ssl_connections[&key].clone();
        let Some(fd) = connection.fd else {
            return self.ssl_reply(tls, key, RESULT_INTERNAL_LOGIC, &[]);
        };
        match connection.handshake {
            Handshake::Idle => {
                if connection.verify & VERIFY_HOST_NAME != 0 && connection.host.is_empty() {
                    return self.ssl_reply(tls, key, RESULT_INTERNAL_LOGIC, &[]);
                }
                let stream = self
                    .process()
                    .bsd_sockets
                    .get(&fd)
                    .and_then(BsdSocket::host_stream);
                let Some((stream, true)) = stream else {
                    return self.ssl_reply(tls, key, RESULT_NO_CONNECTION, &[]);
                };
                let roots = self.ssl_roots_to_send();
                self.log(
                    Level::Info,
                    &format!("[net] tls on stream {stream} for {:?}", connection.host),
                );
                self.net.outbox.push(NetCommand::StartTls {
                    stream,
                    server_name: connection.host.clone(),
                    verify: connection.verify,
                    alpn: if connection.options[OPTION_ENABLE_ALPN] {
                        connection.alpn.clone()
                    } else {
                        Vec::new()
                    },
                    roots,
                });
                let connection = self.connection_mut(key);
                connection.handshake = Handshake::Pending;
                connection.stream = Some(stream);
                self.ssl_io_wait(tls, key)
            }
            Handshake::Pending => self.ssl_io_wait(tls, key),
            Handshake::Failed(code) => self.ssl_reply(tls, key, code, &[]),
            Handshake::Done if with_certificates => {
                let (certificates, count) = connection.certificates();
                let (certificates, count) = if connection.verify & 1 != 0 {
                    (certificates, count)
                } else {
                    (Vec::new(), 0)
                };
                let size = self.write_output_buffer(tls, 0, &certificates);
                let mut out = size.to_le_bytes().to_vec();
                out.extend_from_slice(&count.to_le_bytes());
                self.ssl_reply(tls, key, 0, &out)
            }
            Handshake::Done => self.ssl_reply(tls, key, 0, &[]),
        }
    }

    /// The trusted roots, the first time the host is asked to run TLS.
    fn ssl_roots_to_send(&mut self) -> Option<Arc<Vec<Vec<u8>>>> {
        if self.net.roots_sent {
            return None;
        }
        self.net.roots_sent = true;
        let roots = self
            .ssl_certificate_store()
            .iter()
            .filter(|cert| cert.status == CERT_ENABLED_TRUSTED)
            .map(|cert| cert.der.clone())
            .collect();
        Some(Arc::new(roots))
    }

    /// `Read` and `Peek`: what the host has decrypted, or end of file, or a wait.
    fn ssl_read(&mut self, tls: u32, key: u64, peek: bool) -> Result<()> {
        let connection = self.process().ssl_connections[&key].clone();
        let (Some(fd), Some(stream)) = (connection.fd, connection.stream) else {
            return self.ssl_reply(tls, key, RESULT_NO_CONNECTION, &[]);
        };
        if connection.handshake != Handshake::Done {
            return self.ssl_reply(tls, key, RESULT_NO_CONNECTION, &[]);
        }
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return self.ssl_reply(tls, key, RESULT_INTERNAL_LOGIC, &[]);
        };
        let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) else {
            return self.ssl_reply(tls, key, RESULT_INVALID_SOCKET, &[]);
        };
        let take = (size as usize).min(socket.rx.len());
        if take == 0 {
            if socket.peer_closed {
                return self.ssl_reply(tls, key, 0, &0u32.to_le_bytes());
            }
            return self.ssl_io_wait(tls, key);
        }
        let bytes: Vec<u8> = if peek {
            socket.rx.iter().take(take).copied().collect()
        } else {
            socket.rx.drain(..take).collect()
        };
        for (offset, &byte) in bytes.iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
        }
        if !peek {
            self.net.outbox.push(NetCommand::Consumed {
                stream,
                bytes: take as u32,
            });
        }
        self.ssl_reply(tls, key, 0, &(take as u32).to_le_bytes())
    }

    /// `Write`: handed to the host as plaintext, which it encrypts.
    fn ssl_write(&mut self, tls: u32, key: u64) -> Result<()> {
        let connection = self.process().ssl_connections[&key].clone();
        let (Some(_), Some(stream)) = (connection.fd, connection.stream) else {
            return self.ssl_reply(tls, key, RESULT_NO_CONNECTION, &[]);
        };
        if let Handshake::Failed(code) = connection.handshake {
            return self.ssl_reply(tls, key, code, &[]);
        }
        if connection.handshake != Handshake::Done {
            return self.ssl_reply(tls, key, RESULT_NO_CONNECTION, &[]);
        }
        if self
            .ssl_socket(key)
            .is_some_and(|socket| socket.peer_closed)
        {
            return self.ssl_reply(tls, key, RESULT_SOCKET_SHUTDOWN, &[]);
        }
        let data = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let sent = data.len() as u32;
        self.net.outbox.push(NetCommand::Send { stream, data });
        self.ssl_reply(tls, key, 0, &sent.to_le_bytes())
    }

    /// `Poll(PollEvent, timeout ms)`: which of readable and writable hold, waiting up to the
    /// timeout for one.
    fn ssl_poll(&mut self, tls: u32, key: u64, events: u32, timeout_ms: u32) -> Result<()> {
        const READ: u32 = 1;
        const WRITE: u32 = 2;
        let done = self.process().ssl_connections[&key].handshake == Handshake::Done;
        let readable = self
            .ssl_socket(key)
            .is_some_and(|socket| !socket.rx.is_empty() || socket.peer_closed);
        let mut ready = 0;
        if events & READ != 0 && readable {
            ready |= READ;
        }
        if events & WRITE != 0 && done {
            ready |= WRITE;
        }
        if ready != 0 {
            return self.ssl_reply(tls, key, 0, &ready.to_le_bytes());
        }
        if timeout_ms == 0 {
            return self.ssl_reply(tls, key, RESULT_IO_TIMEOUT, &[]);
        }
        self.ssl_wait(tls, key, i64::from(timeout_ms) * 1_000_000, true)
    }

    /// What the host says about a TLS handshake it ran. `false` for any other event.
    pub(in crate::services::net) fn ssl_event(&mut self, id: StreamId, event: &NetEvent) -> bool {
        if !matches!(
            event,
            NetEvent::TlsEstablished { .. } | NetEvent::TlsFailed { .. }
        ) {
            return false;
        }
        let Some(connection) = self
            .machine
            .processes
            .values_mut()
            .flat_map(|process| process.ssl_connections.values_mut())
            .find(|connection| connection.stream == Some(id))
        else {
            return true;
        };
        match event {
            NetEvent::TlsEstablished {
                protocol,
                cipher,
                alpn,
                chain,
                ..
            } => {
                connection.handshake = Handshake::Done;
                connection.protocol = protocol.clone();
                connection.cipher = cipher.clone();
                connection.negotiated = alpn.clone();
                connection.chain = chain.clone();
            }
            NetEvent::TlsFailed { reason, .. } => {
                let code = match reason {
                    TlsFailure::Reset => RESULT_CONNECTION_RESET,
                    TlsFailure::UnknownCa => RESULT_UNKNOWN_CA,
                    TlsFailure::Expired => RESULT_EXPIRED_CERT,
                    TlsFailure::NotValidForName => RESULT_INVALID_CERT_DOMAIN,
                    TlsFailure::BadCertificate => RESULT_BAD_CERT,
                    TlsFailure::ProtocolVersion => RESULT_ALERT_PROTOCOL_VERSION,
                    TlsFailure::Handshake => RESULT_ALERT_HANDSHAKE_FAILURE,
                };
                connection.handshake = Handshake::Failed(code);
                if matches!(
                    reason,
                    TlsFailure::UnknownCa
                        | TlsFailure::Expired
                        | TlsFailure::NotValidForName
                        | TlsFailure::BadCertificate
                ) {
                    connection.cert_error = code;
                }
            }
            _ => {}
        }
        true
    }

    /// A session closed: its connections go, and the ones that ran TLS send a close_notify.
    pub(crate) fn ssl_forget_session(&mut self, session: u64) {
        let streams: Vec<StreamId> = self
            .process()
            .ssl_connections
            .iter()
            .filter(|&(&key, connection)| {
                key & !0xFFFF_FFFF == session << 32 && connection.handshake == Handshake::Done
            })
            .filter_map(|(_, connection)| connection.stream)
            .collect();
        self.process_mut()
            .ssl_connections
            .retain(|&key, _| key & !0xFFFF_FFFF != session << 32);
        for stream in streams {
            self.net.outbox.push(NetCommand::TlsClose { stream });
        }
    }
}

#[cfg(test)]
mod tests;
