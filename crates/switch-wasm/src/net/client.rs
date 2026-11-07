//! The page's end of the relay protocol, apart from the socket: what the guest asked for
//! becomes frames, and what the relay sent becomes events for the core.

use std::collections::BTreeMap;

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
        matches!(self, State::Connecting | State::Open | State::Ended)
    }

    fn label(self) -> &'static str {
        match self {
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

#[derive(Default)]
pub(crate) struct Client {
    connections: BTreeMap<StreamId, Connection>,
}

impl Client {
    /// The frames for what the guest asked since the last call. What it read is handed
    /// back as one `CREDIT` per stream.
    pub fn outgoing(&mut self, commands: Vec<NetCommand>) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
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
                            state: State::Connecting,
                            sent: 0,
                            received: 0,
                        },
                    );
                    self.forget_old();
                    frames.push(
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
                    frames.push(ClientFrame::Data { stream, data }.encode());
                }
                NetCommand::ShutdownWrite { stream } => {
                    frames.push(ClientFrame::ShutdownWrite { stream }.encode());
                }
                NetCommand::Close { stream } => {
                    if let Some(connection) = self.connections.get_mut(&stream) {
                        connection.state = State::Closed;
                    }
                    credit.retain(|&(held, _)| held != stream);
                    frames.push(ClientFrame::Close { stream }.encode());
                }
                NetCommand::Consumed { stream, bytes } => {
                    match credit.iter_mut().find(|(held, _)| *held == stream) {
                        Some((_, total)) => *total = total.saturating_add(bytes),
                        None => credit.push((stream, bytes)),
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
                frames.push(ClientFrame::Credit { stream, bytes }.encode());
            }
        }
        frames
    }

    /// The event for a frame the relay sent, if it carries one. A frame that cannot be
    /// understood ends the link.
    pub fn incoming(&mut self, raw: &[u8]) -> Result<Option<NetEvent>, Refused> {
        let frame = ServerFrame::decode(raw).map_err(Refused::Undecodable)?;
        Ok(match frame {
            ServerFrame::Hello { version, .. } if version != VERSION => {
                return Err(Refused::Version(version))
            }
            ServerFrame::Hello { .. } => None,
            ServerFrame::Connected { stream } => {
                self.set_state(stream, State::Open);
                Some(NetEvent::Connected { stream })
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
                Some(NetEvent::Failed { stream, reason })
            }
            ServerFrame::Data { stream, data } => {
                if let Some(connection) = self.connections.get_mut(&stream) {
                    connection.received += data.len() as u64;
                }
                Some(NetEvent::Data { stream, data })
            }
            ServerFrame::Closed { stream } => {
                self.set_state(stream, State::Ended);
                Some(NetEvent::Closed { stream })
            }
        })
    }

    /// The link is gone: every stream the guest still has ends.
    pub fn lost(&mut self) -> Vec<NetEvent> {
        let mut events = Vec::new();
        for (&stream, connection) in &mut self.connections {
            match connection.state {
                State::Connecting => {
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

    fn set_state(&mut self, stream: StreamId, state: State) {
        if let Some(connection) = self.connections.get_mut(&stream) {
            connection.state = state;
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

    fn decoded(frames: Vec<Vec<u8>>) -> Vec<ClientFrame> {
        frames
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
        client.incoming(&frame.encode()).unwrap()
    }

    #[test]
    fn hello_carries_no_event_and_a_foreign_version_ends_the_link() {
        let mut client = Client::default();
        let hello = ServerFrame::Hello {
            version: VERSION,
            max_streams: 8,
        };
        assert_eq!(client.incoming(&hello.encode()), Ok(None));
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
}
