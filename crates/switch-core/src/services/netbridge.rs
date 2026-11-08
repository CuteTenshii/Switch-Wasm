//! The hand-off between the guest's sockets and the host that carries them off the console.
//! The core never touches a network: it queues [`NetCommand`]s for the host and applies the
//! [`NetEvent`]s the host answers with, between slices. Disabled, no command is ever queued
//! and `bsd` keeps its empty network.

use std::collections::HashMap;
use std::sync::Arc;

pub type StreamId = u32;

/// Where a stream goes: a name the guest resolved, or a literal address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Peer {
    Name(String),
    Address([u8; 4]),
}

/// Where something came from: an address as the host saw it, or the name the guest sent to,
/// for a reply from a host it reached by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    V4([u8; 4]),
    V6([u8; 16]),
    Name(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetCommand {
    /// A TCP connection.
    Connect {
        stream: StreamId,
        peer: Peer,
        port: u16,
    },
    /// A UDP socket on the host for one guest socket; `port` is the guest's, which the host keeps
    /// when it can.
    OpenUdp {
        stream: StreamId,
        port: u16,
    },
    /// One datagram from a UDP socket.
    SendTo {
        stream: StreamId,
        peer: Peer,
        port: u16,
        data: Vec<u8>,
    },
    /// Accept TCP connections on the host's `port`, as the guest's listener on it.
    Listen {
        stream: StreamId,
        port: u16,
    },
    Send {
        stream: StreamId,
        data: Vec<u8>,
    },
    /// The guest is done sending; the peer may still answer.
    ShutdownWrite {
        stream: StreamId,
    },
    Close {
        stream: StreamId,
    },
    /// The guest read this many bytes the host delivered, so the host may send that many more.
    Consumed {
        stream: StreamId,
        bytes: u32,
    },
    /// Run TLS over this stream from now on: what is sent afterwards is plaintext for the host
    /// to encrypt, and what arrives is plaintext it decrypted. `verify` is the guest's
    /// `VerifyOption` bitmask, `roots` the DER certificates to trust (sent with the first
    /// request only, the host keeps them) and `alpn` the protocols to offer.
    StartTls {
        stream: StreamId,
        server_name: String,
        verify: u32,
        alpn: Vec<Vec<u8>>,
        roots: Option<Arc<Vec<Vec<u8>>>>,
    },
    /// Send a TLS close_notify.
    TlsClose {
        stream: StreamId,
    },
}

/// Why a handshake failed, as far as the guest tells apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsFailure {
    /// The connection went away.
    Reset,
    UnknownCa,
    Expired,
    NotValidForName,
    BadCertificate,
    ProtocolVersion,
    Handshake,
}

/// Why a connection, socket or listener did not open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetFailure {
    Refused,
    Unreachable,
    TimedOut,
    NotFound,
    /// Something else holds the port.
    InUse,
    /// The host does not let guests have it.
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetEvent {
    Connected {
        stream: StreamId,
    },
    Failed {
        stream: StreamId,
        reason: NetFailure,
    },
    Data {
        stream: StreamId,
        data: Vec<u8>,
    },
    /// The peer finished sending; the guest may still send.
    Closed {
        stream: StreamId,
    },
    /// The connection was reset or broke: nothing more either way.
    Reset {
        stream: StreamId,
    },
    /// The host has delivered this many more of the bytes the guest sent.
    Sent {
        stream: StreamId,
        bytes: u32,
    },
    /// A datagram a UDP socket received.
    Datagram {
        stream: StreamId,
        source: Source,
        port: u16,
        data: Vec<u8>,
    },
    /// A connection to a listener, carried from now on as `stream`.
    Incoming {
        listener: StreamId,
        stream: StreamId,
        source: Source,
        port: u16,
    },
    /// The handshake finished: what was negotiated, and the server's certificates, DER, leaf first.
    TlsEstablished {
        stream: StreamId,
        protocol: String,
        cipher: String,
        alpn: Option<Vec<u8>>,
        chain: Vec<Vec<u8>>,
    },
    TlsFailed {
        stream: StreamId,
        reason: TlsFailure,
    },
}

/// First synthetic address, in the range RFC 2544 reserves for benchmarking: nothing real
/// lives there, so a guest cannot mistake one for a host it could reach by number.
const SYNTHETIC_BASE: [u8; 2] = [198, 18];

/// Most names the table holds: the rest of the /15.
const SYNTHETIC_CAPACITY: usize = 1 << 17;

#[derive(Debug, Default)]
pub(crate) struct NetBridge {
    pub enabled: bool,
    names: Vec<String>,
    by_name: HashMap<String, usize>,
    next_stream: StreamId,
    pub outbox: Vec<NetCommand>,
    /// The host has the trusted roots already.
    pub roots_sent: bool,
}

impl NetBridge {
    /// A stable address for `name`, the same one every time. `None` once the table is full.
    pub fn synthesize(&mut self, name: &str) -> Option<[u8; 4]> {
        let name = name.to_ascii_lowercase();
        let index = match self.by_name.get(&name) {
            Some(&index) => index,
            None if self.names.len() >= SYNTHETIC_CAPACITY => return None,
            None => {
                let index = self.names.len();
                self.by_name.insert(name.clone(), index);
                self.names.push(name);
                index
            }
        };
        let [a, b] = SYNTHETIC_BASE;
        Some([a, b | (index >> 16) as u8, (index >> 8) as u8, index as u8])
    }

    /// The name a synthetic address stands for.
    pub fn name_of(&self, address: [u8; 4]) -> Option<&str> {
        let [a, b] = SYNTHETIC_BASE;
        if address[0] != a || address[1] & !1 != b {
            return None;
        }
        let index = (usize::from(address[1] & 1) << 16)
            | (usize::from(address[2]) << 8)
            | usize::from(address[3]);
        self.names.get(index).map(String::as_str)
    }

    /// Where a connect to `address` should go: the name behind a synthetic address, else the
    /// address itself.
    pub fn peer_for(&self, address: [u8; 4]) -> Peer {
        self.name_of(address)
            .map_or(Peer::Address(address), |name| Peer::Name(name.to_owned()))
    }

    /// The IPv4 address the guest sees for `source`: itself; for a name, the synthetic address
    /// the guest sent to; for an IPv6 address, the synthetic address of its text, which the
    /// host resolves back to it. `None` once the table is full.
    pub fn address_for(&mut self, source: &Source) -> Option<[u8; 4]> {
        match source {
            Source::V4(address) => Some(*address),
            Source::V6(address) => self.synthesize(&std::net::Ipv6Addr::from(*address).to_string()),
            Source::Name(name) => self.synthesize(name),
        }
    }

    pub fn open_stream(&mut self) -> StreamId {
        self.next_stream = self.next_stream.wrapping_add(1);
        self.next_stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_keeps_its_address_and_the_address_names_it_back() {
        let mut bridge = NetBridge::default();
        let first = bridge.synthesize("Example.COM").unwrap();
        let second = bridge.synthesize("example.org").unwrap();
        assert_ne!(first, second);
        assert_eq!(bridge.synthesize("example.com"), Some(first));
        assert_eq!(bridge.name_of(first), Some("example.com"));
        assert_eq!(bridge.name_of(second), Some("example.org"));
        assert_eq!(first[..2], [198, 18]);
    }

    #[test]
    fn only_synthetic_addresses_name_a_host() {
        let mut bridge = NetBridge::default();
        let known = bridge.synthesize("example.com").unwrap();
        assert_eq!(bridge.name_of([198, 18, 200, 200]), None);
        assert_eq!(bridge.name_of([10, 0, 0, 1]), None);
        assert_eq!(bridge.peer_for(known), Peer::Name("example.com".into()));
        assert_eq!(bridge.peer_for([1, 1, 1, 1]), Peer::Address([1, 1, 1, 1]));
    }

    #[test]
    fn the_table_spills_into_the_second_octet() {
        let mut bridge = NetBridge::default();
        for index in 0..300 {
            bridge.synthesize(&format!("host{index}.test")).unwrap();
        }
        let address = bridge.synthesize("host299.test").unwrap();
        assert_eq!(address, [198, 18, 1, 43]);
        assert_eq!(bridge.name_of(address), Some("host299.test"));
    }

    #[test]
    fn an_ipv6_source_stands_as_a_synthetic_address_that_names_it() {
        let mut bridge = NetBridge::default();
        let mut v6 = [0u8; 16];
        v6[..4].copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8]);
        v6[15] = 1;
        let address = bridge.address_for(&Source::V6(v6)).unwrap();
        assert_eq!(address[..2], [198, 18]);
        assert_eq!(bridge.peer_for(address), Peer::Name("2001:db8::1".into()));
        assert_eq!(bridge.address_for(&Source::V6(v6)), Some(address), "stable");
        assert_eq!(
            bridge.address_for(&Source::V4([1, 2, 3, 4])),
            Some([1, 2, 3, 4])
        );
        let named = bridge.synthesize("dns.example").unwrap();
        assert_eq!(
            bridge.address_for(&Source::Name("dns.example".into())),
            Some(named),
            "a reply from a named host comes from where the guest sent"
        );
    }

    #[test]
    fn streams_are_numbered_from_one() {
        let mut bridge = NetBridge::default();
        assert_eq!(bridge.open_stream(), 1);
        assert_eq!(bridge.open_stream(), 2);
    }
}
