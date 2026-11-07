//! The frames the page and the relay exchange over one WebSocket; `docs/relay.md` is the
//! spec. A frame is `u8 op | u32 stream | payload`, little-endian, one per binary message.

pub const VERSION: u8 = 1;

/// Most bytes a relay puts in one `DATA` frame.
pub const MAX_DATA: usize = 16 * 1024;

pub type StreamId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proto {
    Tcp,
    Udp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Peer {
    Address([u8; 4]),
    Name(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    Refused,
    Unreachable,
    TimedOut,
    NotFound,
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientFrame {
    Connect {
        stream: StreamId,
        proto: Proto,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerFrame {
    Hello { version: u8, max_streams: u16 },
    Connected { stream: StreamId },
    Failed { stream: StreamId, reason: Failure },
    Data { stream: StreamId, data: Vec<u8> },
    Closed { stream: StreamId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    UnknownOp(u8),
    BadProto(u8),
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

const HELLO: u8 = 0x80;
const CONNECTED: u8 = 0x81;
const FAILED: u8 = 0x82;
const SERVER_DATA: u8 = 0x83;
const CLOSED: u8 = 0x84;

const PEER_ADDRESS: u8 = 0;
const PEER_NAME: u8 = 1;

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
        }
    }

    fn from_code(code: u8) -> Result<Failure, DecodeError> {
        Ok(match code {
            0 => Failure::Refused,
            1 => Failure::Unreachable,
            2 => Failure::TimedOut,
            3 => Failure::NotFound,
            4 => Failure::Denied,
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
                proto,
                port,
                window,
                peer,
            } => {
                header(&mut out, CONNECT, *stream);
                out.push(match proto {
                    Proto::Tcp => 0,
                    Proto::Udp => 1,
                });
                out.extend_from_slice(&port.to_le_bytes());
                out.extend_from_slice(&window.to_le_bytes());
                match peer {
                    Peer::Address(ip) => {
                        out.push(PEER_ADDRESS);
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
        }
        out
    }

    pub fn decode(raw: &[u8]) -> Result<ClientFrame, DecodeError> {
        let (op, stream, mut rest) = split(raw)?;
        let frame = match op {
            CONNECT => {
                let proto = match rest.u8()? {
                    0 => Proto::Tcp,
                    1 => Proto::Udp,
                    other => return Err(DecodeError::BadProto(other)),
                };
                let port = rest.u16()?;
                let window = rest.u32()?;
                let peer = match rest.u8()? {
                    PEER_ADDRESS => Peer::Address(rest.take(4)?.try_into().unwrap()),
                    PEER_NAME => {
                        let len = usize::from(rest.u8()?);
                        let name = std::str::from_utf8(rest.take(len)?)
                            .map_err(|_| DecodeError::BadName)?;
                        Peer::Name(name.to_owned())
                    }
                    other => return Err(DecodeError::BadPeer(other)),
                };
                rest.finish()?;
                ClientFrame::Connect {
                    stream,
                    proto,
                    port,
                    window,
                    peer,
                }
            }
            CLIENT_DATA => ClientFrame::Data {
                stream,
                data: rest.0.to_vec(),
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
                data: rest.0.to_vec(),
            },
            CLOSED => {
                rest.finish()?;
                ServerFrame::Closed { stream }
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
            proto: Proto::Tcp,
            port: 80,
            window: 65536,
            peer: Peer::Name("google.com".into()),
        };
        let mut expected = vec![
            0x01, 0x01, 0, 0, 0, 0x00, 0x50, 0x00, 0, 0, 1, 0, 0x01, 0x0a,
        ];
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
    fn the_documented_server_frames_are_these_bytes() {
        let hello = ServerFrame::Hello {
            version: VERSION,
            max_streams: 64,
        };
        assert_eq!(hello.encode(), [0x80, 0, 0, 0, 0, 1, 0x40, 0]);
        assert_eq!(
            ServerFrame::Connected { stream: 1 }.encode(),
            [0x81, 1, 0, 0, 0]
        );
        assert_eq!(
            ServerFrame::Closed { stream: 1 }.encode(),
            [0x84, 1, 0, 0, 0]
        );
    }

    #[test]
    fn every_client_frame_survives_a_round_trip() {
        let frames = [
            ClientFrame::Connect {
                stream: 7,
                proto: Proto::Udp,
                port: 53,
                window: 0,
                peer: Peer::Address([1, 1, 1, 1]),
            },
            ClientFrame::Connect {
                stream: u32::MAX,
                proto: Proto::Tcp,
                port: 443,
                window: u32::MAX,
                peer: Peer::Name("a.b".into()),
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
                reason: Failure::Denied,
            },
            ServerFrame::Data {
                stream: 5,
                data: vec![1, 2, 3],
            },
            ServerFrame::Closed { stream: 5 },
        ];
        for frame in frames {
            assert_eq!(ServerFrame::decode(&frame.encode()).as_ref(), Ok(&frame));
        }
    }

    #[test]
    fn a_name_too_long_for_its_length_byte_is_cut_not_corrupted() {
        let frame = ClientFrame::Connect {
            stream: 1,
            proto: Proto::Tcp,
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
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
            Err(DecodeError::BadProto(2))
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]),
            Err(DecodeError::BadPeer(9))
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, b'a']),
            Err(DecodeError::Truncated)
        );
        assert_eq!(
            ClientFrame::decode(&[0x01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0xff]),
            Err(DecodeError::BadName)
        );
        assert_eq!(
            ServerFrame::decode(&[0x82, 1, 0, 0, 0, 9]),
            Err(DecodeError::BadFailure(9))
        );
    }
}
