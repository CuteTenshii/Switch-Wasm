//! The frames the page and the relay exchange over one WebSocket; `docs/relay.md` is the
//! spec. A frame is `u8 op | u32 stream | payload`, little-endian, one per binary message.

pub const VERSION: u8 = 4;

/// The oldest version a page still understands.
pub const OLDEST_VERSION: u8 = 3;

/// The first version whose relay sends `ALIVE`.
pub const ALIVE_SINCE: u8 = 2;

/// The first version whose relay takes `GET_SETTINGS` and `SET_SETTINGS`; an older one closes
/// the link on them.
pub const SETTINGS_SINCE: u8 = 4;

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

/// The settings a page asks the relay to have, all of them at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsChange {
    pub port: u16,
    pub allow_private: bool,
    pub allow_inbound: bool,
    /// The origins the settings trust.
    pub origins: Vec<String>,
}

/// The relay's settings as it reports them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// The port the relay listens on now.
    pub listening: u16,
    /// The port it listens on from its next start.
    pub port: u16,
    pub allow_private: bool,
    pub allow_inbound: bool,
    /// Whether changes are saved, or last only until the relay exits.
    pub saved: bool,
    /// A flag sets the port for this run.
    pub port_locked: bool,
    pub private_locked: bool,
    pub inbound_locked: bool,
    /// The origins the settings trust, which a page may change.
    pub origins: Vec<String>,
    /// The origins trusted whatever the settings say.
    pub fixed_origins: Vec<String>,
}

/// Why the relay did not make a settings change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsRefusal {
    /// The user said no, or there was no way to ask.
    Denied,
    /// Another change waits for its answer.
    Busy,
    /// A flag sets this for the run.
    Locked,
    /// A port of 0, too many origins, or something that is not an origin.
    Invalid,
    /// The settings file could not be written; nothing changed.
    Unsaved,
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
    GetSettings,
    SetSettings(SettingsChange),
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
    /// The settings, asked for or just changed.
    Settings(Settings),
    /// A settings change waits for an answer on the relay's computer.
    Asking,
    SettingsRefused(SettingsRefusal),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    UnknownOp(u8),
    BadPeer(u8),
    BadName,
    BadFailure(u8),
    BadRefusal(u8),
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
const GET_SETTINGS: u8 = 0x09;
const SET_SETTINGS: u8 = 0x0a;

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
const SETTINGS: u8 = 0x8a;
const ASKING: u8 = 0x8b;
const SETTINGS_REFUSED: u8 = 0x8c;

const ALLOW_PRIVATE: u8 = 1 << 0;
const ALLOW_INBOUND: u8 = 1 << 1;
const SAVED: u8 = 1 << 2;
const PORT_LOCKED: u8 = 1 << 3;
const PRIVATE_LOCKED: u8 = 1 << 4;
const INBOUND_LOCKED: u8 = 1 << 5;

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

    /// A UTF-8 string after its `u8` length.
    fn text(&mut self) -> Result<String, DecodeError> {
        let len = usize::from(self.u8()?);
        let text = std::str::from_utf8(self.take(len)?).map_err(|_| DecodeError::BadName)?;
        Ok(text.to_owned())
    }

    /// A `u8` count and that many strings.
    fn texts(&mut self) -> Result<Vec<String>, DecodeError> {
        let count = self.u8()?;
        (0..count).map(|_| self.text()).collect()
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

/// A string cut to the 255 bytes its length byte can say, at a character boundary.
fn put_text(out: &mut Vec<u8>, text: &str) {
    let mut end = text.len().min(255);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    out.push(end as u8);
    out.extend_from_slice(&text.as_bytes()[..end]);
}

/// A count and the strings; past 255 the rest are left out.
fn put_texts(out: &mut Vec<u8>, texts: &[String]) {
    let texts = &texts[..texts.len().min(255)];
    out.push(texts.len() as u8);
    for text in texts {
        put_text(out, text);
    }
}

fn flags(bits: &[(bool, u8)]) -> u8 {
    bits.iter()
        .filter(|(set, _)| *set)
        .fold(0, |flags, (_, bit)| flags | bit)
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

impl SettingsRefusal {
    fn code(self) -> u8 {
        match self {
            SettingsRefusal::Denied => 0,
            SettingsRefusal::Busy => 1,
            SettingsRefusal::Locked => 2,
            SettingsRefusal::Invalid => 3,
            SettingsRefusal::Unsaved => 4,
        }
    }

    fn from_code(code: u8) -> Result<SettingsRefusal, DecodeError> {
        Ok(match code {
            0 => SettingsRefusal::Denied,
            1 => SettingsRefusal::Busy,
            2 => SettingsRefusal::Locked,
            3 => SettingsRefusal::Invalid,
            4 => SettingsRefusal::Unsaved,
            _ => return Err(DecodeError::BadRefusal(code)),
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
            ClientFrame::GetSettings => header(&mut out, GET_SETTINGS, 0),
            ClientFrame::SetSettings(change) => {
                header(&mut out, SET_SETTINGS, 0);
                out.extend_from_slice(&change.port.to_le_bytes());
                out.push(flags(&[
                    (change.allow_private, ALLOW_PRIVATE),
                    (change.allow_inbound, ALLOW_INBOUND),
                ]));
                put_texts(&mut out, &change.origins);
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
            GET_SETTINGS => {
                rest.finish()?;
                ClientFrame::GetSettings
            }
            SET_SETTINGS => {
                let port = rest.u16()?;
                let flags = rest.u8()?;
                let origins = rest.texts()?;
                rest.finish()?;
                ClientFrame::SetSettings(SettingsChange {
                    port,
                    allow_private: flags & ALLOW_PRIVATE != 0,
                    allow_inbound: flags & ALLOW_INBOUND != 0,
                    origins,
                })
            }
            other => return Err(DecodeError::UnknownOp(other)),
        };
        Ok(frame)
    }
}

impl ServerFrame {
    /// Whether `raw` is `HELLO` or a settings frame, which belong to no stream, by its op
    /// alone.
    pub fn is_link_frame(raw: &[u8]) -> bool {
        matches!(
            raw.first(),
            Some(&(HELLO | SETTINGS | ASKING | SETTINGS_REFUSED))
        )
    }

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
            ServerFrame::Settings(settings) => {
                header(&mut out, SETTINGS, 0);
                out.extend_from_slice(&settings.listening.to_le_bytes());
                out.extend_from_slice(&settings.port.to_le_bytes());
                out.push(flags(&[
                    (settings.allow_private, ALLOW_PRIVATE),
                    (settings.allow_inbound, ALLOW_INBOUND),
                    (settings.saved, SAVED),
                    (settings.port_locked, PORT_LOCKED),
                    (settings.private_locked, PRIVATE_LOCKED),
                    (settings.inbound_locked, INBOUND_LOCKED),
                ]));
                put_texts(&mut out, &settings.origins);
                put_texts(&mut out, &settings.fixed_origins);
            }
            ServerFrame::Asking => header(&mut out, ASKING, 0),
            ServerFrame::SettingsRefused(reason) => {
                header(&mut out, SETTINGS_REFUSED, 0);
                out.push(reason.code());
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
            SETTINGS => {
                let listening = rest.u16()?;
                let port = rest.u16()?;
                let flags = rest.u8()?;
                let origins = rest.texts()?;
                let fixed_origins = rest.texts()?;
                rest.finish()?;
                ServerFrame::Settings(Settings {
                    listening,
                    port,
                    allow_private: flags & ALLOW_PRIVATE != 0,
                    allow_inbound: flags & ALLOW_INBOUND != 0,
                    saved: flags & SAVED != 0,
                    port_locked: flags & PORT_LOCKED != 0,
                    private_locked: flags & PRIVATE_LOCKED != 0,
                    inbound_locked: flags & INBOUND_LOCKED != 0,
                    origins,
                    fixed_origins,
                })
            }
            ASKING => {
                rest.finish()?;
                ServerFrame::Asking
            }
            SETTINGS_REFUSED => {
                let reason = SettingsRefusal::from_code(rest.u8()?)?;
                rest.finish()?;
                ServerFrame::SettingsRefused(reason)
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
        assert_eq!(hello.encode(), [0x80, 0, 0, 0, 0, 4, 0x40, 0]);
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
                peer: Peer::Address([1, 2, 3, 4]),
            }
            .encode(),
            [0x89, 0, 0, 0, 0x80, 3, 0, 0, 0, 0x90, 0x1f, 0, 1, 2, 3, 4]
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
            ClientFrame::GetSettings,
            ClientFrame::SetSettings(SettingsChange {
                port: 18787,
                allow_private: true,
                allow_inbound: false,
                origins: vec!["https://a.example".into(), "http://b:8000".into()],
            }),
            ClientFrame::SetSettings(SettingsChange {
                port: 1,
                allow_private: false,
                allow_inbound: true,
                origins: Vec::new(),
            }),
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
            ServerFrame::Settings(Settings {
                listening: 18787,
                port: 9000,
                allow_private: false,
                allow_inbound: true,
                saved: true,
                port_locked: false,
                private_locked: true,
                inbound_locked: false,
                origins: vec!["https://a.example".into()],
                fixed_origins: vec!["https://tenshii.moe".into(), "http://localhost:8000".into()],
            }),
            ServerFrame::Asking,
            ServerFrame::SettingsRefused(SettingsRefusal::Denied),
            ServerFrame::SettingsRefused(SettingsRefusal::Unsaved),
        ];
        for frame in frames {
            assert_eq!(ServerFrame::decode(&frame.encode()).as_ref(), Ok(&frame));
        }
    }

    #[test]
    fn the_documented_settings_frames_are_these_bytes() {
        assert_eq!(ClientFrame::GetSettings.encode(), [0x09, 0, 0, 0, 0]);
        let change = ClientFrame::SetSettings(SettingsChange {
            port: 18787,
            allow_private: true,
            allow_inbound: true,
            origins: vec!["https://a.b".into()],
        });
        let mut expected = vec![0x0a, 0, 0, 0, 0, 0x63, 0x49, 0b11, 1, 11];
        expected.extend_from_slice(b"https://a.b");
        assert_eq!(change.encode(), expected);
        let settings = ServerFrame::Settings(Settings {
            listening: 18787,
            port: 18787,
            allow_private: false,
            allow_inbound: false,
            saved: true,
            port_locked: true,
            private_locked: false,
            inbound_locked: true,
            origins: Vec::new(),
            fixed_origins: vec!["http://x".into()],
        });
        let mut expected = vec![0x8a, 0, 0, 0, 0, 0x63, 0x49, 0x63, 0x49, 0b10_1100, 0, 1, 8];
        expected.extend_from_slice(b"http://x");
        assert_eq!(settings.encode(), expected);
        assert_eq!(ServerFrame::Asking.encode(), [0x8b, 0, 0, 0, 0]);
        assert_eq!(
            ServerFrame::SettingsRefused(SettingsRefusal::Locked).encode(),
            [0x8c, 0, 0, 0, 0, 2]
        );
    }

    #[test]
    fn link_frames_are_told_apart_by_their_op() {
        for frame in [
            ServerFrame::Hello {
                version: VERSION,
                max_streams: 1,
            },
            ServerFrame::Asking,
            ServerFrame::SettingsRefused(SettingsRefusal::Busy),
        ] {
            assert!(ServerFrame::is_link_frame(&frame.encode()), "{frame:?}");
        }
        for frame in [ServerFrame::Alive, ServerFrame::Connected { stream: 1 }] {
            assert!(!ServerFrame::is_link_frame(&frame.encode()), "{frame:?}");
        }
        assert!(!ServerFrame::is_link_frame(&[]));
    }

    #[test]
    fn a_text_too_long_for_its_length_byte_is_cut_at_a_character() {
        let change = ClientFrame::SetSettings(SettingsChange {
            port: 1,
            allow_private: false,
            allow_inbound: false,
            origins: vec!["é".repeat(200)],
        });
        let Ok(ClientFrame::SetSettings(decoded)) = ClientFrame::decode(&change.encode()) else {
            panic!("a cut text still decodes");
        };
        assert_eq!(decoded.origins[0], "é".repeat(127));
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
        assert_eq!(
            ServerFrame::decode(&[0x8c, 0, 0, 0, 0, 9]),
            Err(DecodeError::BadRefusal(9))
        );
        assert_eq!(
            ClientFrame::decode(&[0x0a, 0, 0, 0, 0, 1, 0, 0, 2, 1, b'a']),
            Err(DecodeError::Truncated),
            "two origins announced, one sent"
        );
        assert_eq!(
            ClientFrame::decode(&[0x09, 0, 0, 0, 0, 1]),
            Err(DecodeError::TrailingBytes)
        );
    }
}
