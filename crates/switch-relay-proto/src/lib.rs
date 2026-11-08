//! The frames the page and the relay exchange over one WebSocket; `docs/relay.md` is the
//! spec. A frame is `u8 op | u32 stream | payload`, little-endian, one per binary message.

pub const VERSION: u8 = 3;

/// The oldest version a page still understands.
pub const OLDEST_VERSION: u8 = 3;

/// The first version whose relay sends `ALIVE`.
pub const ALIVE_SINCE: u8 = 2;

/// Most bytes in one `DATA` frame, either way.
pub const MAX_DATA: usize = 16 * 1024;

/// Most bytes in one datagram: the largest UDP payload IPv4 can carry.
pub const MAX_DATAGRAM: usize = 65_507;

/// Most bytes in one frame: a full datagram, with its header and the longest peer.
pub const MAX_FRAME: usize = MAX_DATAGRAM + 0x200;

/// Streams the relay opens (connections to a listener) are numbered from here; the page numbers
/// its own below it.
pub const RELAY_STREAMS: StreamId = 0x8000_0000;

pub type StreamId = u32;

/// Where something is: an address of either family, or a name for the relay to resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Peer {
    Address([u8; 4]),
    Name(String),
    Address6([u8; 16]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    Refused,
    Unreachable,
    TimedOut,
    NotFound,
    Denied,
    /// A `LISTEN` or `UDP_OPEN` for a port something else holds.
    InUse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientFrame {
    /// A TCP connection.
    Connect {
        stream: StreamId,
        port: u16,
        window: u32,
        peer: Peer,
    },
    Data {
        stream: StreamId,
        data: Vec<u8>,
    },
    ShutdownWrite {
        stream: StreamId,
    },
    Close {
        stream: StreamId,
    },
    Credit {
        stream: StreamId,
        bytes: u32,
    },
    /// A UDP socket, on `port` when it is not 0 and the relay takes inbound traffic.
    UdpOpen {
        stream: StreamId,
        port: u16,
        window: u32,
    },
    /// One datagram from a UDP socket to `peer`.
    SendTo {
        stream: StreamId,
        port: u16,
        peer: Peer,
        data: Vec<u8>,
    },
    /// Accept TCP connections on `port`; each gets `window`.
    Listen {
        stream: StreamId,
        port: u16,
        window: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerFrame {
    Hello {
        version: u8,
        max_streams: u16,
    },
    Connected {
        stream: StreamId,
    },
    Failed {
        stream: StreamId,
        reason: Failure,
    },
    Data {
        stream: StreamId,
        data: Vec<u8>,
    },
    /// The peer finished sending; the guest may still send.
    Closed {
        stream: StreamId,
    },
    /// The relay is still there; sent on a timer, whatever else is going on.
    Alive,
    /// The connection was reset or broke: nothing more either way.
    Reset {
        stream: StreamId,
    },
    /// The relay wrote this many more of the stream's bytes to the peer.
    Sent {
        stream: StreamId,
        bytes: u32,
    },
    /// One datagram a UDP socket received, and where from.
    ReceivedFrom {
        stream: StreamId,
        port: u16,
        peer: Peer,
        data: Vec<u8>,
    },
    /// A connection to a listener, as the new `stream`.
    Incoming {
        stream: StreamId,
        listener: StreamId,
        port: u16,
        peer: Peer,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    UnknownOp(u8),
    BadPeer(u8),
    BadName,
    BadFailure(u8),
    TrailingBytes,
}

const CONNECT: u8 = 0x01;
const CLIENT_DATA: u8 = 0x02;
const SHUTDOWN_WRITE: u8 = 0x03;
const CLOSE: u8 = 0x04;
const CREDIT: u8 = 0x05;
const UDP_OPEN: u8 = 0x06;
const SEND_TO: u8 = 0x07;
const LISTEN: u8 = 0x08;

const HELLO: u8 = 0x80;
const CONNECTED: u8 = 0x81;
const FAILED: u8 = 0x82;
const SERVER_DATA: u8 = 0x83;
const CLOSED: u8 = 0x84;
const ALIVE: u8 = 0x85;
const RESET: u8 = 0x86;
const SENT: u8 = 0x87;
const RECEIVED_FROM: u8 = 0x88;
const INCOMING: u8 = 0x89;

const PEER_ADDRESS: u8 = 0;
const PEER_NAME: u8 = 1;
const PEER_ADDRESS6: u8 = 2;

/// A cursor over a frame's payload.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], DecodeError> {
        if self.0.len() < len {
            return Err(DecodeError::Truncated);
        }
        let (head, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn peer(&mut self) -> Result<Peer, DecodeError> {
        Ok(match self.u8()? {
            PEER_ADDRESS => Peer::Address(self.take(4)?.try_into().unwrap()),
            PEER_ADDRESS6 => Peer::Address6(self.take(16)?.try_into().unwrap()),
            PEER_NAME => {
                let len = usize::from(self.u8()?);
                let name =
                    std::str::from_utf8(self.take(len)?).map_err(|_| DecodeError::BadName)?;
                Peer::Name(name.to_owned())
            }
            other => return Err(DecodeError::BadPeer(other)),
        })
    }

    /// Everything left, which is the frame's last field.
    fn rest(self) -> Vec<u8> {
        self.0.to_vec()
    }

    fn finish(self) -> Result<(), DecodeError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(DecodeError::TrailingBytes)
        }
    }
}

fn header(out: &mut Vec<u8>, op: u8, stream: StreamId) {
    out.push(op);
    out.extend_from_slice(&stream.to_le_bytes());
}

fn put_peer(out: &mut Vec<u8>, peer: &Peer) {
    match peer {
        Peer::Address(ip) => {
            out.push(PEER_ADDRESS);
            out.extend_from_slice(ip);
        }
        Peer::Address6(ip) => {
            out.push(PEER_ADDRESS6);
            out.extend_from_slice(ip);
        }
        Peer::Name(name) => {
            // A name over 255 bytes cannot be a hostname; the relay refuses the cut one.
            let name = &name.as_bytes()[..name.len().min(255)];
            out.push(PEER_NAME);
            out.push(name.len() as u8);
            out.extend_from_slice(name);
        }
    }
}

/// The op and stream id, and a reader over the rest.
fn split(raw: &[u8]) -> Result<(u8, StreamId, Reader<'_>), DecodeError> {
    let mut reader = Reader(raw);
    let op = reader.u8()?;
    let stream = reader.u32()?;
    Ok((op, stream, reader))
}

impl Failure {
    fn code(self) -> u8 {
        match self {
            Failure::Refused => 0,
            Failure::Unreachable => 1,
            Failure::TimedOut => 2,
            Failure::NotFound => 3,
            Failure::Denied => 4,
            Failure::InUse => 5,
        }
    }

    fn from_code(code: u8) -> Result<Failure, DecodeError> {
        Ok(match code {
            0 => Failure::Refused,
            1 => Failure::Unreachable,
            2 => Failure::TimedOut,
            3 => Failure::NotFound,
            4 => Failure::Denied,
            5 => Failure::InUse,
            _ => return Err(DecodeError::BadFailure(code)),
        })
    }
}

impl ClientFrame {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            ClientFrame::Connect {
                stream,
                port,
                window,
                peer,
            } => {
                header(&mut out, CONNECT, *stream);
                out.extend_from_slice(&port.to_le_bytes());
                out.extend_from_slice(&window.to_le_bytes());
                put_peer(&mut out, peer);
            }
            ClientFrame::Data { stream, data } => {
                header(&mut out, CLIENT_DATA, *stream);
                out.extend_from_slice(data);
            }
            ClientFrame::ShutdownWrite { stream } => header(&mut out, SHUTDOWN_WRITE, *stream),
            ClientFrame::Close { stream } => header(&mut out, CLOSE, *stream),
            ClientFrame::Credit { stream, bytes } => {
                header(&mut out, CREDIT, *stream);
                out.extend_from_slice(&bytes.to_le_bytes());
            }
            ClientFrame::UdpOpen {
                stream,
                port,
                window,
            } => {
                header(&mut out, UDP_OPEN, *stream);
                out.extend_from_slice(&port.to_le_bytes());
                out.extend_from_slice(&window.to_le_bytes());
            }
            ClientFrame::SendTo {
                stream,
                port,
                peer,
                data,
            } => {
                header(&mut out, SEND_TO, *stream);
                out.extend_from_slice(&port.to_le_bytes());
                put_peer(&mut out, peer);
                out.extend_from_slice(data);
            }
            ClientFrame::Listen {
                stream,
                port,
                window,
            } => {
                header(&mut out, LISTEN, *stream);
                out.extend_from_slice(&port.to_le_bytes());
                out.extend_from_slice(&window.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(raw: &[u8]) -> Result<ClientFrame, DecodeError> {
        let (op, stream, mut rest) = split(raw)?;
        let frame = match op {
            CONNECT => {
                let port = rest.u16()?;
                let window = rest.u32()?;
                let peer = rest.peer()?;
                rest.finish()?;
                ClientFrame::Connect {
                    stream,
                    port,
                    window,
                    peer,
                }
            }
            CLIENT_DATA => ClientFrame::Data {
                stream,
                data: rest.rest(),
            },
            SHUTDOWN_WRITE => {
                rest.finish()?;
                ClientFrame::ShutdownWrite { stream }
            }
            CLOSE => {
                rest.finish()?;
                ClientFrame::Close { stream }
            }
            CREDIT => {
                let bytes = rest.u32()?;
                rest.finish()?;
                ClientFrame::Credit { stream, bytes }
            }
            UDP_OPEN | LISTEN => {
                let port = rest.u16()?;
                let window = rest.u32()?;
                rest.finish()?;
                if op == UDP_OPEN {
                    ClientFrame::UdpOpen {
                        stream,
                        port,
                        window,
                    }
                } else {
                    ClientFrame::Listen {
                        stream,
                        port,
                        window,
                    }
                }
            }
            SEND_TO => {
                let port = rest.u16()?;
                let peer = rest.peer()?;
                ClientFrame::SendTo {
                    stream,
                    port,
                    peer,
                    data: rest.rest(),
                }
            }
            other => return Err(DecodeError::UnknownOp(other)),
        };
        Ok(frame)
    }
}

impl ServerFrame {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            ServerFrame::Hello {
                version,
                max_streams,
            } => {
                header(&mut out, HELLO, 0);
                out.push(*version);
                out.extend_from_slice(&max_streams.to_le_bytes());
            }
            ServerFrame::Connected { stream } => header(&mut out, CONNECTED, *stream),
            ServerFrame::Failed { stream, reason } => {
                header(&mut out, FAILED, *stream);
                out.push(reason.code());
            }
            ServerFrame::Data { stream, data } => {
                header(&mut out, SERVER_DATA, *stream);
                out.extend_from_slice(data);
            }
            ServerFrame::Closed { stream } => header(&mut out, CLOSED, *stream),
            ServerFrame::Alive => header(&mut out, ALIVE, 0),
            ServerFrame::Reset { stream } => header(&mut out, RESET, *stream),
            ServerFrame::Sent { stream, bytes } => {
                header(&mut out, SENT, *stream);
                out.extend_from_slice(&bytes.to_le_bytes());
            }
            ServerFrame::ReceivedFrom {
                stream,
                port,
                peer,
                data,
            } => {
                header(&mut out, RECEIVED_FROM, *stream);
                out.extend_from_slice(&port.to_le_bytes());
                put_peer(&mut out, peer);
                out.extend_from_slice(data);
            }
            ServerFrame::Incoming {
                stream,
                listener,
                port,
                peer,
            } => {
                header(&mut out, INCOMING, *stream);
                out.extend_from_slice(&listener.to_le_bytes());
                out.extend_from_slice(&port.to_le_bytes());
                put_peer(&mut out, peer);
            }
        }
        out
    }

    pub fn decode(raw: &[u8]) -> Result<ServerFrame, DecodeError> {
        let (op, stream, mut rest) = split(raw)?;
        let frame = match op {
            HELLO => {
                let version = rest.u8()?;
                let max_streams = rest.u16()?;
                rest.finish()?;
                ServerFrame::Hello {
                    version,
                    max_streams,
                }
            }
            CONNECTED => {
                rest.finish()?;
                ServerFrame::Connected { stream }
            }
            FAILED => {
                let reason = Failure::from_code(rest.u8()?)?;
                rest.finish()?;
                ServerFrame::Failed { stream, reason }
            }
            SERVER_DATA => ServerFrame::Data {
                stream,
                data: rest.rest(),
            },
            CLOSED => {
                rest.finish()?;
                ServerFrame::Closed { stream }
            }
            ALIVE => {
                rest.finish()?;
                ServerFrame::Alive
            }
            RESET => {
                rest.finish()?;
                ServerFrame::Reset { stream }
            }
            SENT => {
                let bytes = rest.u32()?;
                rest.finish()?;
                ServerFrame::Sent { stream, bytes }
            }
            RECEIVED_FROM => {
                let port = rest.u16()?;
                let peer = rest.peer()?;
                ServerFrame::ReceivedFrom {
                    stream,
                    port,
                    peer,
                    data: rest.rest(),
                }
            }
            INCOMING => {
                let listener = rest.u32()?;
                let port = rest.u16()?;
                let peer = rest.peer()?;
                rest.finish()?;
                ServerFrame::Incoming {
                    stream,
                    listener,
                    port,
                    peer,
                }
            }
            other => return Err(DecodeError::UnknownOp(other)),
        };
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: &[u8] = b"GET / HTTP/1.1\r\nHost: google.com\r\nConnection: close\r\n\r\n";

    #[test]
    fn the_documented_connect_is_these_bytes() {
        let frame = ClientFrame::Connect {
            stream: 1,
            port: 80,
            window: 65536,
            peer: Peer::Name("google.com".into()),
        };
        let mut expected = vec![0x01, 0x01, 0, 0, 0, 0x50, 0x00, 0, 0, 1, 0, 0x01, 0x0a];
        expected.extend_from_slice(b"google.com");
        assert_eq!(frame.encode(), expected);
        assert_eq!(ClientFrame::decode(&expected), Ok(frame));
    }

    #[test]
    fn the_documented_data_is_the_op_the_stream_and_the_request() {
        let frame = ClientFrame::Data {
            stream: 1,
            data: REQUEST.to_vec(),
        };
        let bytes = frame.encode();
        assert_eq!(bytes.len(), 60);
        assert_eq!(&bytes[..5], [0x02, 1, 0, 0, 0]);
        assert_eq!(&bytes[5..], REQUEST);
    }

    #[test]
    fn the_documented_datagram_frames_are_these_bytes() {
        let send = ClientFrame::SendTo {
            stream: 2,
            port: 53,
            peer: Peer::Address([1, 1, 1, 1]),
            data: b"q".to_vec(),
        };
        assert_eq!(
            send.encode(),
            [0x07, 2, 0, 0, 0, 53, 0, 0, 1, 1, 1, 1, b'q']
        );
        let mut v6 = [0u8; 16];
        v6[15] = 1;
        let received = ServerFrame::ReceivedFrom {
            stream: 2,
            port: 53,
            peer: Peer::Address6(v6),
            data: b"a".to_vec(),
        };
        let mut expected = vec![0x88, 2, 0, 0, 0, 53, 0, 2];
        expected.extend_from_slice(&v6);
        expected.push(b'a');
        assert_eq!(received.encode(), expected);
    }

    #[test]
    fn the_documented_server_frames_are_these_bytes() {
        let hello = ServerFrame::Hello {
            version: VERSION,
            max_streams: 64,
        };
        assert_eq!(hello.encode(), [0x80, 0, 0, 0, 0, 3, 0x40, 0]);
        assert_eq!(
            ServerFrame::Connected { stream: 1 }.encode(),
            [0x81, 1, 0, 0, 0]
        );
        assert_eq!(
            ServerFrame::Closed { stream: 1 }.encode(),
            [0x84, 1, 0, 0, 0]
        );
        assert_eq!(ServerFrame::Alive.encode(), [0x85, 0, 0, 0, 0]);
        assert_eq!(
            ServerFrame::Reset { stream: 1 }.encode(),
            [0x86, 1, 0, 0, 0]
        );
        assert_eq!(
            ServerFrame::Sent {
                stream: 1,
                bytes: 0x0102
            }
            .encode(),
            [0x87, 1, 0, 0, 0, 2, 1, 0, 0]
        );
        assert_eq!(
            ServerFrame::Incoming {
                stream: RELAY_STREAMS,
                listener: 3,
                port: 0x1f90,
                peer: Peer::Address([8, 8, 4, 4]),
            }
            .encode(),
            [0x89, 0, 0, 0, 0x80, 3, 0, 0, 0, 0x90, 0x1f, 0, 8, 8, 4, 4]
        );
    }

    #[test]
    fn every_client_frame_survives_a_round_trip() {
        let frames = [
            ClientFrame::Connect {
                stream: 7,
                port: 53,
                window: 0,
                peer: Peer::Address([1, 1, 1, 1]),
            },
            ClientFrame::Connect {
                stream: u32::MAX,
                port: 443,
                window: u32::MAX,
                peer: Peer::Name("a.b".into()),
            },
            ClientFrame::Connect {
                stream: 8,
                port: 443,
                window: 1,
                peer: Peer::Address6([0xfe; 16]),
            },
            ClientFrame::Data {
                stream: 3,
                data: Vec::new(),
            },
            ClientFrame::Data {
                stream: 3,
                data: vec![0, 255, 7],
            },
            ClientFrame::ShutdownWrite { stream: 9 },
            ClientFrame::Close { stream: 9 },
            ClientFrame::Credit {
                stream: 2,
                bytes: 4096,
            },
            ClientFrame::UdpOpen {
                stream: 4,
                port: 19132,
                window: 1 << 16,
            },
            ClientFrame::SendTo {
                stream: 4,
                port: 3478,
                peer: Peer::Name("stun.example".into()),
                data: vec![1, 2],
            },
            ClientFrame::SendTo {
                stream: 4,
                port: 3478,
                peer: Peer::Address([9, 9, 9, 9]),
                data: Vec::new(),
            },
            ClientFrame::Listen {
                stream: 5,
                port: 8080,
                window: 1024,
            },
        ];
        for frame in frames {
            assert_eq!(ClientFrame::decode(&frame.encode()).as_ref(), Ok(&frame));
        }
    }

    #[test]
    fn every_server_frame_survives_a_round_trip() {
        let frames = [
            ServerFrame::Hello {
                version: 1,
                max_streams: u16::MAX,
            },
            ServerFrame::Connected { stream: 5 },
            ServerFrame::Failed {
                stream: 5,
                reason: Failure::Refused,
            },
            ServerFrame::Failed {
                stream: 5,
                reason: Failure::InUse,
            },
            ServerFrame::Data {
                stream: 5,
                data: vec![1, 2, 3],
            },
            ServerFrame::Closed { stream: 5 },
            ServerFrame::Alive,
            ServerFrame::Reset { stream: 5 },
            ServerFrame::Sent {
                stream: 5,
                bytes: u32::MAX,
            },
            ServerFrame::ReceivedFrom {
                stream: 5,
                port: 1,
                peer: Peer::Address([1, 2, 3, 4]),
                data: vec![9; 3],
            },
            ServerFrame::ReceivedFrom {
                stream: 5,
                port: 1,
                peer: Peer::Address6([7; 16]),
                data: Vec::new(),
            },
            ServerFrame::Incoming {
                stream: RELAY_STREAMS + 1,
                listener: 5,
                port: 50000,
                peer: Peer::Address6([1; 16]),
            },
        ];
        for frame in frames {
            assert_eq!(ServerFrame::decode(&frame.encode()).as_ref(), Ok(&frame));
        }
    }

    #[test]
    fn a_name_too_long_for_its_length_byte_is_cut_not_corrupted() {
        let frame = ClientFrame::Connect {
            stream: 1,
            port: 1,
            window: 0,
            peer: Peer::Name("x".repeat(300)),
        };
        let Ok(ClientFrame::Connect {
            peer: Peer::Name(name),
            ..
        }) = ClientFrame::decode(&frame.encode())
        else {
            panic!("a cut name still decodes");
        };
        assert_eq!(name.len(), 255);
    }

    #[test]
    fn malformed_frames_are_refused_with_the_reason() {
        assert_eq!(ClientFrame::decode(&[]), Err(DecodeError::Truncated));
        assert_eq!(ClientFrame::decode(&[1, 0, 0]), Err(DecodeError::Truncated));
        assert_eq!(
            ClientFrame::decode(&[0x7f, 0, 0, 0, 0]),
            Err(DecodeError::UnknownOp(0x7f))
        );
        // A server op is not a client op, and the other way round.
        assert_eq!(
            ClientFrame::decode(&[0x81, 0, 0, 0, 0]),
            Err(DecodeError::UnknownOp(0x81))
        );
        assert_eq!(
            ServerFrame::decode(&[0x01, 0, 0, 0, 0]),
            Err(DecodeError::UnknownOp(0x01))
        );
        assert_eq!(
            ClientFrame::decode(&[0x04, 1, 0, 0, 0, 9]),
            Err(DecodeError::TrailingBytes)
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]),
            Err(DecodeError::BadPeer(9))
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, b'a']),
            Err(DecodeError::Truncated)
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0xff]),
            Err(DecodeError::BadName)
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2]),
            Err(DecodeError::Truncated),
            "an IPv6 address is 16 bytes"
        );
        assert_eq!(
            ServerFrame::decode(&[0x82, 1, 0, 0, 0, 9]),
            Err(DecodeError::BadFailure(9))
        );
    }
}
