//! The hand-off between the guest's sockets and the host that carries them off the console.
//! The core never touches a network: it queues [`NetCommand`]s for the host and applies the
//! [`NetEvent`]s the host answers with, between slices. Disabled, no command is ever queued
//! and `bsd` keeps its empty network.

use std::collections::HashMap;

pub type StreamId = u32;

/// Where a stream goes: a name the guest resolved, or a literal address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Peer {
    Name(String),
    Address([u8; 4]),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetCommand {
    Connect {
        stream: StreamId,
        udp: bool,
        peer: Peer,
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
}

/// Why a connection did not open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetFailure {
    Refused,
    Unreachable,
    TimedOut,
    NotFound,
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
    Closed {
        stream: StreamId,
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
    fn streams_are_numbered_from_one() {
        let mut bridge = NetBridge::default();
        assert_eq!(bridge.open_stream(), 1);
        assert_eq!(bridge.open_stream(), 2);
    }
}
