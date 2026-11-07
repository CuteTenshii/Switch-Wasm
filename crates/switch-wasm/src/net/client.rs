//! The page's end of the relay protocol, apart from the socket: what the guest asked for
//! becomes frames, and what the relay sent becomes events for the core.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::tls::{self, TlsStream};
use rustls::RootCertStore;
use switch_core::json;
use switch_core::services::netbridge::{NetCommand, NetEvent, NetFailure, Peer, StreamId};
use switch_relay_proto::{
    ClientFrame, DecodeError, Failure, Peer as WirePeer, Proto, ServerFrame, VERSION,
};

/// Bytes the relay may send on a stream before the guest has read any.
const WINDOW: u32 = 256 * 1024;

/// Connections kept for the page to list, live ones and the most recent finished ones.
const HISTORY: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Asked for, but the frame is not on the socket yet.
    Waiting,
    Connecting,
    Open,
    /// The relay said the peer is done; the guest still has to close its end.
    Ended,
    /// The guest closed it, or the link went.
    Closed,
    Failed(NetFailure),
}

impl State {
    fn live(self) -> bool {
        matches!(
            self,
            State::Waiting | State::Connecting | State::Open | State::Ended
        )
    }

    fn label(self) -> &'static str {
        match self {
            State::Waiting => "waiting",
            State::Connecting => "connecting",
            State::Open => "open",
            State::Ended => "ended",
            State::Closed => "closed",
            State::Failed(NetFailure::Refused) => "refused",
            State::Failed(NetFailure::Unreachable) => "unreachable",
            State::Failed(NetFailure::TimedOut) => "timed out",
            State::Failed(NetFailure::NotFound) => "not found",
        }
    }
}

struct Connection {
    host: String,
    port: u16,
    udp: bool,
    state: State,
    sent: u64,
    received: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    Undecodable(DecodeError),
    Version(u8),
}

/// What one step of the client makes: frames for the relay and events for the core.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Pump {
    pub frames: Vec<Vec<u8>>,
    pub events: Vec<NetEvent>,
}

impl Pump {
    fn data(&mut self, stream: StreamId, bytes: Vec<u8>) {
        if !bytes.is_empty() {
            self.frames.push(
                ClientFrame::Data {
                    stream,
                    data: bytes,
                }
                .encode(),
            );
        }
    }

    fn credit(&mut self, stream: StreamId, bytes: u32) {
        if bytes > 0 {
            self.frames
                .push(ClientFrame::Credit { stream, bytes }.encode());
        }
    }
}

#[derive(Default)]
pub(crate) struct Client {
    connections: BTreeMap<StreamId, Connection>,
    /// The streams the guest runs TLS on, which this side encrypts and decrypts.
    tls: HashMap<StreamId, TlsStream>,
    /// The roots the core sent with its first TLS request.
    roots: Option<Arc<RootCertStore>>,
}

impl Client {
    /// What the guest asked since the last call. What it read is handed back as one `CREDIT`
    /// per stream; on a TLS stream the credit is for the ciphertext that plaintext came from.
    pub fn outgoing(&mut self, commands: Vec<NetCommand>) -> Pump {
        let mut pump = Pump::default();
        let mut credit: Vec<(StreamId, u32)> = Vec::new();
        for command in commands {
            match command {
                NetCommand::Connect {
                    stream,
                    udp,
                    peer,
                    port,
                } => {
                    self.connections.insert(
                        stream,
                        Connection {
                            host: match &peer {
                                Peer::Address([a, b, c, d]) => format!("{a}.{b}.{c}.{d}"),
                                Peer::Name(name) => name.clone(),
                            },
                            port,
                            udp,
                            state: State::Waiting,
                            sent: 0,
                            received: 0,
                        },
                    );
                    self.forget_old();
                    pump.frames.push(
                        ClientFrame::Connect {
                            stream,
                            proto: if udp { Proto::Udp } else { Proto::Tcp },
                            port,
                            window: WINDOW,
                            peer: match peer {
                                Peer::Address(ip) => WirePeer::Address(ip),
                                Peer::Name(name) => WirePeer::Name(name),
                            },
                        }
                        .encode(),
                    );
                }
                NetCommand::Send { stream, data } => {
                    if let Some(connection) = self.connections.get_mut(&stream) {
                        connection.sent += data.len() as u64;
                    }
                    match self.tls.get_mut(&stream) {
                        Some(tls) => {
                            let encrypted = tls.write(&data);
                            pump.data(stream, encrypted);
                        }
                        None => pump.data(stream, data),
                    }
                }
                NetCommand::ShutdownWrite { stream } => {
                    if let Some(tls) = self.tls.get_mut(&stream) {
                        let alert = tls.close_notify();
                        pump.data(stream, alert);
                    }
                    pump.frames
                        .push(ClientFrame::ShutdownWrite { stream }.encode());
                }
                NetCommand::Close { stream } => {
                    self.set_state(stream, State::Closed);
                    self.tls.remove(&stream);
                    credit.retain(|&(held, _)| held != stream);
                    pump.frames.push(ClientFrame::Close { stream }.encode());
                }
                NetCommand::Consumed { stream, bytes } => {
                    let bytes = match self.tls.get_mut(&stream) {
                        Some(tls) => tls.consumed(bytes),
                        None => bytes,
                    };
                    match credit.iter_mut().find(|(held, _)| *held == stream) {
                        Some((_, total)) => *total = total.saturating_add(bytes),
                        None => credit.push((stream, bytes)),
                    }
                }
                NetCommand::StartTls {
                    stream,
                    server_name,
                    verify,
                    alpn,
                    roots,
                } => {
                    if let Some(der) = roots {
                        self.roots = Some(tls::roots(&der));
                    }
                    let roots = self.roots.clone().unwrap_or_else(|| tls::roots(&[]));
                    match TlsStream::new(&server_name, verify, alpn, &roots) {
                        Ok((tls, hello)) => {
                            self.tls.insert(stream, tls);
                            pump.data(stream, hello);
                        }
                        Err(reason) => pump.events.push(NetEvent::TlsFailed { stream, reason }),
                    }
                }
                NetCommand::TlsClose { stream } => {
                    if let Some(tls) = self.tls.get_mut(&stream) {
                        let alert = tls.close_notify();
                        pump.data(stream, alert);
                    }
                }
            }
        }
        for (stream, bytes) in credit {
            if self
                .connections
                .get(&stream)
                .is_some_and(|c| c.state.live())
            {
                pump.credit(stream, bytes);
            }
        }
        pump
    }

    /// What a frame the relay sent comes to. A frame that cannot be understood ends the link.
    pub fn incoming(&mut self, raw: &[u8]) -> Result<Pump, Refused> {
        let frame = ServerFrame::decode(raw).map_err(Refused::Undecodable)?;
        let mut pump = Pump::default();
        match frame {
            ServerFrame::Hello { version, .. } if version != VERSION => {
                return Err(Refused::Version(version))
            }
            ServerFrame::Hello { .. } => {}
            ServerFrame::Connected { stream } => {
                self.set_state(stream, State::Open);
                pump.events.push(NetEvent::Connected { stream });
            }
            ServerFrame::Failed { stream, reason } => {
                let reason = match reason {
                    Failure::Refused => NetFailure::Refused,
                    Failure::TimedOut => NetFailure::TimedOut,
                    Failure::NotFound => NetFailure::NotFound,
                    // A destination the policy keeps out of reach has no route, from here.
                    Failure::Unreachable | Failure::Denied => NetFailure::Unreachable,
                };
                self.set_state(stream, State::Failed(reason));
                self.tls.remove(&stream);
                pump.events.push(NetEvent::Failed { stream, reason });
            }
            ServerFrame::Data { stream, data } => match self.tls.get_mut(&stream) {
                Some(tls) => {
                    let fed = tls.feed(&data);
                    let established = tls.is_established();
                    pump.data(stream, fed.outgoing);
                    pump.credit(stream, fed.credit);
                    if let Some(found) = fed.established {
                        pump.events.push(NetEvent::TlsEstablished {
                            stream,
                            protocol: found.protocol,
                            cipher: found.cipher,
                            alpn: found.alpn,
                            chain: found.chain,
                        });
                    }
                    if let Some(reason) = fed.failure {
                        // Once the guest has its connection a break is the end of it.
                        pump.events.push(if established {
                            NetEvent::Closed { stream }
                        } else {
                            NetEvent::TlsFailed { stream, reason }
                        });
                    }
                    if !fed.plaintext.is_empty() {
                        if let Some(connection) = self.connections.get_mut(&stream) {
                            connection.received += fed.plaintext.len() as u64;
                        }
                        pump.events.push(NetEvent::Data {
                            stream,
                            data: fed.plaintext,
                        });
                    }
                    if fed.closed {
                        pump.events.push(NetEvent::Closed { stream });
                    }
                }
                None => {
                    if let Some(connection) = self.connections.get_mut(&stream) {
                        connection.received += data.len() as u64;
                    }
                    pump.events.push(NetEvent::Data { stream, data });
                }
            },
            ServerFrame::Closed { stream } => {
                self.set_state(stream, State::Ended);
                pump.events.push(NetEvent::Closed { stream });
            }
        }
        Ok(pump)
    }

    /// The link is gone: every stream the guest still has ends.
    pub fn lost(&mut self) -> Vec<NetEvent> {
        self.tls.clear();
        let mut events = Vec::new();
        for (&stream, connection) in &mut self.connections {
            match connection.state {
                State::Waiting | State::Connecting => {
                    connection.state = State::Failed(NetFailure::Unreachable);
                    events.push(NetEvent::Failed {
                        stream,
                        reason: NetFailure::Unreachable,
                    });
                }
                State::Open => {
                    connection.state = State::Closed;
                    events.push(NetEvent::Closed { stream });
                }
                State::Ended => connection.state = State::Closed,
                State::Closed | State::Failed(_) => {}
            }
        }
        events
    }

    /// The connections, newest first, as `[{id, host, port, proto, state, sent, received}]`.
    pub fn report(&self) -> String {
        json::array(|a| {
            for (&id, c) in self.connections.iter().rev() {
                a.object(|o| {
                    o.field("id", id)
                        .field("host", c.host.as_str())
                        .field("port", u32::from(c.port))
                        .field("proto", if c.udp { "udp" } else { "tcp" })
                        .field("state", c.state.label())
                        .field("sent", c.sent)
                        .field("received", c.received);
                });
            }
        })
    }

    /// The frames given out by `outgoing` are on the socket.
    pub fn sent(&mut self) {
        for connection in self.connections.values_mut() {
            if connection.state == State::Waiting {
                connection.state = State::Connecting;
            }
        }
    }

    /// Move a live connection on. A finished one keeps how it ended, except that a guest which
    /// gave up before the answer still gets to learn why it failed.
    fn set_state(&mut self, stream: StreamId, state: State) {
        if let Some(connection) = self.connections.get_mut(&stream) {
            let late_failure =
                connection.state == State::Closed && matches!(state, State::Failed(_));
            if connection.state.live() || late_failure {
                connection.state = state;
            }
        }
    }

    /// Drop the oldest finished connections past the history.
    fn forget_old(&mut self) {
        while self.connections.len() > HISTORY {
            let Some(oldest) = self
                .connections
                .iter()
                .find(|(_, connection)| !connection.state.live())
                .map(|(&id, _)| id)
            else {
                return;
            };
            self.connections.remove(&oldest);
        }
    }

    /// The answers for commands given while there is no link: connects fail, the rest
    /// are for streams that never existed.
    pub fn refuse(commands: Vec<NetCommand>) -> Vec<NetEvent> {
        commands
            .into_iter()
            .filter_map(|command| match command {
                NetCommand::Connect { stream, .. } => Some(NetEvent::Failed {
                    stream,
                    reason: NetFailure::Unreachable,
                }),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connect(stream: StreamId) -> NetCommand {
        NetCommand::Connect {
            stream,
            udp: false,
            peer: Peer::Name("example.com".into()),
            port: 443,
        }
    }

    fn decoded(pump: Pump) -> Vec<ClientFrame> {
        pump.frames
            .iter()
            .map(|raw| ClientFrame::decode(raw).unwrap())
            .collect()
    }

    #[test]
    fn a_connect_asks_for_a_window_and_names_the_peer() {
        let mut client = Client::default();
        let frames = decoded(client.outgoing(vec![connect(1)]));
        assert_eq!(
            frames,
            vec![ClientFrame::Connect {
                stream: 1,
                proto: Proto::Tcp,
                port: 443,
                window: WINDOW,
                peer: WirePeer::Name("example.com".into()),
            }]
        );
    }

    #[test]
    fn bytes_the_guest_read_come_back_as_one_credit_per_stream() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1), connect(2)]);
        let frames = decoded(client.outgoing(vec![
            NetCommand::Consumed {
                stream: 1,
                bytes: 100,
            },
            NetCommand::Consumed {
                stream: 2,
                bytes: 7,
            },
            NetCommand::Consumed {
                stream: 1,
                bytes: 50,
            },
        ]));
        assert_eq!(
            frames,
            vec![
                ClientFrame::Credit {
                    stream: 1,
                    bytes: 150
                },
                ClientFrame::Credit {
                    stream: 2,
                    bytes: 7
                },
            ]
        );
    }

    #[test]
    fn credit_for_a_stream_that_was_closed_is_not_sent() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1)]);
        let frames = decoded(client.outgoing(vec![
            NetCommand::Consumed {
                stream: 1,
                bytes: 10,
            },
            NetCommand::Close { stream: 1 },
            NetCommand::Consumed {
                stream: 9,
                bytes: 10,
            },
        ]));
        assert_eq!(frames, vec![ClientFrame::Close { stream: 1 }]);
    }

    #[test]
    fn data_and_half_close_pass_straight_through() {
        let mut client = Client::default();
        let frames = decoded(client.outgoing(vec![
            NetCommand::Send {
                stream: 3,
                data: b"hi".to_vec(),
            },
            NetCommand::ShutdownWrite { stream: 3 },
        ]));
        assert_eq!(
            frames,
            vec![
                ClientFrame::Data {
                    stream: 3,
                    data: b"hi".to_vec()
                },
                ClientFrame::ShutdownWrite { stream: 3 },
            ]
        );
    }

    #[test]
    fn relay_frames_become_core_events() {
        let mut client = Client::default();
        let event = |frame: ServerFrame| client_event(&mut Client::default(), frame);
        assert_eq!(
            event(ServerFrame::Data {
                stream: 1,
                data: vec![9]
            }),
            Some(NetEvent::Data {
                stream: 1,
                data: vec![9]
            })
        );
        assert_eq!(
            client_event(&mut client, ServerFrame::Connected { stream: 1 }),
            Some(NetEvent::Connected { stream: 1 })
        );
        assert_eq!(
            event(ServerFrame::Failed {
                stream: 2,
                reason: Failure::Denied
            }),
            Some(NetEvent::Failed {
                stream: 2,
                reason: NetFailure::Unreachable
            })
        );
        assert_eq!(
            event(ServerFrame::Failed {
                stream: 2,
                reason: Failure::NotFound
            }),
            Some(NetEvent::Failed {
                stream: 2,
                reason: NetFailure::NotFound
            })
        );
    }

    fn client_event(client: &mut Client, frame: ServerFrame) -> Option<NetEvent> {
        client
            .incoming(&frame.encode())
            .unwrap()
            .events
            .into_iter()
            .next()
    }

    #[test]
    fn hello_carries_no_event_and_a_foreign_version_ends_the_link() {
        let mut client = Client::default();
        let hello = ServerFrame::Hello {
            version: VERSION,
            max_streams: 8,
        };
        assert_eq!(client.incoming(&hello.encode()), Ok(Pump::default()));
        let foreign = ServerFrame::Hello {
            version: VERSION + 1,
            max_streams: 8,
        };
        assert_eq!(
            client.incoming(&foreign.encode()),
            Err(Refused::Version(VERSION + 1))
        );
        assert_eq!(
            client.incoming(&[0x7f, 0, 0, 0, 0]),
            Err(Refused::Undecodable(DecodeError::UnknownOp(0x7f)))
        );
    }

    #[test]
    fn losing_the_link_fails_what_was_connecting_and_closes_what_was_open() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1), connect(2), connect(3)]);
        client_event(&mut client, ServerFrame::Connected { stream: 2 });
        client_event(&mut client, ServerFrame::Connected { stream: 3 });
        client_event(&mut client, ServerFrame::Closed { stream: 3 });
        assert_eq!(
            client.lost(),
            vec![
                NetEvent::Failed {
                    stream: 1,
                    reason: NetFailure::Unreachable
                },
                NetEvent::Closed { stream: 2 },
            ]
        );
        assert!(client.lost().is_empty(), "each stream ends once");
    }

    #[test]
    fn with_no_link_a_connect_fails_and_nothing_else_answers() {
        let events = Client::refuse(vec![
            connect(1),
            NetCommand::Send {
                stream: 1,
                data: vec![1],
            },
            NetCommand::Close { stream: 1 },
        ]);
        assert_eq!(
            events,
            vec![NetEvent::Failed {
                stream: 1,
                reason: NetFailure::Unreachable
            }]
        );
    }

    #[test]
    fn the_report_lists_connections_newest_first_with_what_moved() {
        let mut client = Client::default();
        client.outgoing(vec![
            connect(1),
            NetCommand::Connect {
                stream: 2,
                udp: true,
                peer: Peer::Address([1, 1, 1, 1]),
                port: 53,
            },
        ]);
        client_event(&mut client, ServerFrame::Connected { stream: 1 });
        client.outgoing(vec![NetCommand::Send {
            stream: 1,
            data: vec![0; 5],
        }]);
        client_event(
            &mut client,
            ServerFrame::Data {
                stream: 1,
                data: vec![0; 7],
            },
        );
        client_event(
            &mut client,
            ServerFrame::Failed {
                stream: 2,
                reason: Failure::Refused,
            },
        );
        assert_eq!(
            client.report(),
            concat!(
                r#"[{"id":2,"host":"1.1.1.1","port":53,"proto":"udp","state":"refused","sent":0,"received":0},"#,
                r#"{"id":1,"host":"example.com","port":443,"proto":"tcp","state":"open","sent":5,"received":7}]"#
            )
        );
        client.outgoing(vec![NetCommand::Close { stream: 1 }]);
        assert!(client.report().contains(r#""state":"closed""#));
        assert_eq!(Client::default().report(), "[]");
    }

    #[test]
    fn old_finished_connections_are_dropped_but_live_ones_stay() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1)]);
        for stream in 2..=(HISTORY as u32 + 20) {
            client.outgoing(vec![connect(stream), NetCommand::Close { stream }]);
        }
        let report = client.report();
        assert_eq!(report.matches(r#""id":"#).count(), HISTORY);
        assert!(report.contains(r#""id":1,"#), "the open one is not history");
        assert!(
            !report.contains(r#""id":2,"#),
            "the oldest finished one is gone"
        );
    }

    #[test]
    fn a_connection_keeps_how_it_ended() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1), connect(2)]);
        client_event(
            &mut client,
            ServerFrame::Failed {
                stream: 1,
                reason: Failure::Refused,
            },
        );
        client.outgoing(vec![
            NetCommand::Close { stream: 1 },
            NetCommand::Close { stream: 2 },
        ]);
        // A late success for a stream the guest already closed changes nothing.
        client_event(&mut client, ServerFrame::Connected { stream: 2 });
        let report = client.report();
        assert!(report
            .contains(r#""id":1,"host":"example.com","port":443,"proto":"tcp","state":"refused""#));
        assert!(report
            .contains(r#""id":2,"host":"example.com","port":443,"proto":"tcp","state":"closed""#));
    }

    #[test]
    fn a_connection_waits_until_its_frame_is_on_the_socket() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1)]);
        assert!(client.report().contains(r#""state":"waiting""#));
        client.sent();
        assert!(client.report().contains(r#""state":"connecting""#));
        // Losing the link before then fails it all the same.
        client.outgoing(vec![connect(2)]);
        assert_eq!(client.lost().len(), 2);
    }

    #[test]
    fn a_guest_that_gave_up_still_learns_why_it_failed() {
        let mut client = Client::default();
        client.outgoing(vec![connect(1)]);
        client.outgoing(vec![NetCommand::Close { stream: 1 }]);
        client_event(
            &mut client,
            ServerFrame::Failed {
                stream: 1,
                reason: Failure::Denied,
            },
        );
        assert!(client.report().contains(r#""state":"unreachable""#));
    }

    /// The ciphertext the client put in `Data` frames for `stream`, and the credit it granted.
    fn wire(pump: &Pump, stream: StreamId) -> (Vec<u8>, u32) {
        let (mut bytes, mut credit) = (Vec::new(), 0);
        for raw in &pump.frames {
            match ClientFrame::decode(raw).unwrap() {
                ClientFrame::Data { stream: s, data } if s == stream => bytes.extend(data),
                ClientFrame::Credit { stream: s, bytes } if s == stream => credit += bytes,
                _ => {}
            }
        }
        (bytes, credit)
    }

    fn start_tls(name: &str, verify: u32, roots: Vec<Vec<u8>>) -> NetCommand {
        NetCommand::StartTls {
            stream: 1,
            server_name: name.into(),
            verify,
            alpn: Vec::new(),
            roots: Some(Arc::new(roots)),
        }
    }

    /// Run a handshake between the client and a server through relay frames; every event.
    fn open_tls(
        client: &mut Client,
        server: &mut rustls::ServerConnection,
        start: NetCommand,
    ) -> Vec<NetEvent> {
        let mut pump = client.outgoing(vec![start]);
        let mut events = std::mem::take(&mut pump.events);
        for _ in 0..8 {
            let reply = super::super::tls::harness::serve(server, &wire(&pump, 1).0);
            if reply.is_empty() {
                break;
            }
            pump = client
                .incoming(
                    &ServerFrame::Data {
                        stream: 1,
                        data: reply,
                    }
                    .encode(),
                )
                .unwrap();
            events.append(&mut pump.events);
        }
        events
    }

    #[test]
    fn a_stream_that_starts_tls_is_encrypted_on_the_way_to_the_relay() {
        use super::super::tls::harness;
        let mut client = Client::default();
        client.outgoing(vec![connect(1)]);
        client_event(&mut client, ServerFrame::Connected { stream: 1 });
        let mut server = harness::server(harness::SERVER, &[]);
        let events = open_tls(
            &mut client,
            &mut server,
            start_tls("localhost", 7, vec![harness::ca_der()]),
        );
        let [NetEvent::TlsEstablished {
            protocol, chain, ..
        }] = &events[..]
        else {
            panic!("one TlsEstablished, got {events:?}");
        };
        assert_eq!((protocol.as_str(), chain.len()), ("TLSv1.3", 1));

        // What the guest sends is encrypted for the relay; the server reads it.
        let sent = client.outgoing(vec![NetCommand::Send {
            stream: 1,
            data: b"GET /".to_vec(),
        }]);
        let (ciphertext, _) = wire(&sent, 1);
        assert!(!ciphertext.windows(5).any(|w| w == b"GET /"));
        harness::serve(&mut server, &ciphertext);

        // What comes back is decrypted for the core, and reading it credits the ciphertext.
        std::io::Write::write_all(&mut server.writer(), b"200 OK").unwrap();
        let mut reply = Vec::new();
        while server.wants_write() {
            server.write_tls(&mut reply).unwrap();
        }
        let got = client
            .incoming(
                &ServerFrame::Data {
                    stream: 1,
                    data: reply.clone(),
                }
                .encode(),
            )
            .unwrap();
        assert_eq!(
            got.events,
            vec![NetEvent::Data {
                stream: 1,
                data: b"200 OK".to_vec()
            }]
        );
        assert!(client.report().contains(r#""sent":5,"received":6"#));
        let read = client.outgoing(vec![NetCommand::Consumed {
            stream: 1,
            bytes: 6,
        }]);
        assert_eq!(wire(&read, 1).1, reply.len() as u32);
    }

    #[test]
    fn a_handshake_that_fails_tells_the_core_why_and_a_bad_name_fails_at_once() {
        use super::super::tls::harness;
        let mut client = Client::default();
        let mut server = harness::server(harness::SERVER, &[]);
        let events = open_tls(
            &mut client,
            &mut server,
            start_tls("localhost", 3, Vec::new()),
        );
        assert_eq!(
            events,
            vec![NetEvent::TlsFailed {
                stream: 1,
                reason: switch_core::services::netbridge::TlsFailure::UnknownCa
            }]
        );

        let pump = client.outgoing(vec![start_tls("not a host name", 0, Vec::new())]);
        assert_eq!(
            pump.events,
            vec![NetEvent::TlsFailed {
                stream: 1,
                reason: switch_core::services::netbridge::TlsFailure::Handshake
            }]
        );
        assert!(pump.frames.is_empty());
    }

    #[test]
    fn the_relay_closing_the_connection_still_ends_a_tls_stream() {
        use super::super::tls::harness;
        let mut client = Client::default();
        client.outgoing(vec![connect(1)]);
        client_event(&mut client, ServerFrame::Connected { stream: 1 });
        let mut server = harness::server(harness::SERVER, &[]);
        open_tls(
            &mut client,
            &mut server,
            start_tls("localhost", 7, vec![harness::ca_der()]),
        );
        assert_eq!(
            client_event(&mut client, ServerFrame::Closed { stream: 1 }),
            Some(NetEvent::Closed { stream: 1 })
        );
        assert!(client.report().contains(r#""state":"ended""#));
        // And what the guest closes goes: the TLS state with it, a plain Close to the relay.
        let pump = client.outgoing(vec![NetCommand::Close { stream: 1 }]);
        assert_eq!(decoded(pump), vec![ClientFrame::Close { stream: 1 }]);
    }
}
