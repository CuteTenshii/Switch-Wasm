//! A relay on a loopback port, driven over a real WebSocket against local servers.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use switch_relay::config::{Approval, Config, Overrides, Settings};
use switch_relay::policy::Unknown;
use switch_relay::Timeouts;
use switch_relay_proto::{
    ClientFrame, Failure, Peer, ServerFrame, Settings as RelaySettings, SettingsChange,
    SettingsRefusal, VERSION,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

const ORIGIN: &str = "https://tenshii.moe";

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// A relay trusting the emulator's origin, and the address to reach it at.
async fn relay(allow_private: bool) -> String {
    relay_with(allow_private, Timeouts::default()).await
}

async fn relay_with(allow_private: bool, timeouts: Timeouts) -> String {
    relay_full(allow_private, false, timeouts).await
}

/// A relay that also takes inbound traffic for its guests.
async fn inbound_relay() -> String {
    relay_full(true, true, Timeouts::default()).await
}

async fn relay_full(allow_private: bool, allow_inbound: bool, timeouts: Timeouts) -> String {
    let overrides = Overrides {
        allow_private,
        allow_inbound,
        ..Overrides::default()
    };
    relay_config(overrides, Approval::Deny, timeouts).await
}

/// A relay with nothing saved, whose settings a page may change; `approval` answers for the
/// user.
async fn relay_config(overrides: Overrides, approval: Approval, timeouts: Timeouts) -> String {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("ws://{address}");
    let settings = Settings::default();
    let config = Config::new(
        settings,
        overrides,
        Unknown::Deny,
        approval,
        None,
        address.port(),
    );
    tokio::spawn(switch_relay::serve(listener, Arc::new(config), timeouts));
    url
}

/// Pings fast enough that a test sees several and gives up on silence well within a second.
const BRISK: Timeouts = Timeouts {
    handshake: Duration::from_millis(100),
    ping_every: Duration::from_millis(50),
    silence: Duration::from_millis(200),
};

async fn connect(url: &str, origin: Option<&str>) -> Socket {
    let mut request = url.into_client_request().unwrap();
    if let Some(origin) = origin {
        request
            .headers_mut()
            .insert("origin", origin.parse().unwrap());
    }
    connect_async(request).await.unwrap().0
}

async fn page(url: &str) -> Socket {
    let mut socket = connect(url, Some(ORIGIN)).await;
    assert!(matches!(
        next(&mut socket).await,
        ServerFrame::Hello {
            version: VERSION,
            ..
        }
    ));
    socket
}

async fn send(socket: &mut Socket, frame: ClientFrame) {
    socket
        .send(Message::Binary(frame.encode().into()))
        .await
        .unwrap();
}

/// The next relay frame. Pings, which the socket answers as it reads, `ALIVE` and `SENT` are
/// skipped; [`next_raw`] keeps `SENT`.
async fn next(socket: &mut Socket) -> ServerFrame {
    loop {
        match next_raw(socket).await {
            ServerFrame::Sent { .. } => continue,
            frame => return frame,
        }
    }
}

async fn next_raw(socket: &mut Socket) -> ServerFrame {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
            .await
            .expect("the relay answers")
            .expect("the socket stays open")
            .unwrap();
        match message {
            Message::Binary(raw) => match ServerFrame::decode(&raw).unwrap() {
                ServerFrame::Alive => continue,
                frame => return frame,
            },
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("a binary frame, got {other:?}"),
        }
    }
}

/// Nothing arrives for a short while.
async fn quiet(socket: &mut Socket) {
    let waited = tokio::time::timeout(Duration::from_millis(300), socket.next()).await;
    assert!(waited.is_err(), "unexpected {waited:?}");
}

fn tcp(stream: u32, port: u16, window: u32) -> ClientFrame {
    ClientFrame::Connect {
        stream,
        port,
        window,
        peer: Peer::Address([127, 0, 0, 1]),
    }
}

/// A TCP server that echoes until the client stops sending, then closes.
async fn echo() -> u16 {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buffer = [0u8; 1024];
                while let Ok(read) = socket.read(&mut buffer).await {
                    if read == 0 || socket.write_all(&buffer[..read]).await.is_err() {
                        break;
                    }
                }
            });
        }
    });
    port
}

#[tokio::test]
async fn tcp_bytes_go_out_and_come_back_until_the_guest_stops_sending() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(&mut socket, tcp(1, port, 4096)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );

    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: b"ping".to_vec(),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Data {
            stream: 1,
            data: b"ping".to_vec()
        }
    );

    send(&mut socket, ClientFrame::ShutdownWrite { stream: 1 }).await;
    assert_eq!(next(&mut socket).await, ServerFrame::Closed { stream: 1 });
}

#[tokio::test]
async fn the_relay_sends_no_more_than_the_guest_has_credited() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = socket.write_all(&vec![7u8; 40_000]).await;
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    let mut socket = page(&relay(true).await).await;
    send(&mut socket, tcp(1, port, 1000)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );

    let mut received = 0;
    while received < 1000 {
        let ServerFrame::Data { data, .. } = next(&mut socket).await else {
            panic!("data");
        };
        received += data.len();
    }
    assert_eq!(received, 1000, "stopped at the window");
    quiet(&mut socket).await;

    send(
        &mut socket,
        ClientFrame::Credit {
            stream: 1,
            bytes: 2000,
        },
    )
    .await;
    let mut more = 0;
    while more < 2000 {
        let ServerFrame::Data { data, .. } = next(&mut socket).await else {
            panic!("data");
        };
        more += data.len();
    }
    assert_eq!(more, 2000);
    quiet(&mut socket).await;
}

#[tokio::test]
async fn a_closed_port_is_refused() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let mut socket = page(&relay(true).await).await;
    send(&mut socket, tcp(1, port, 0)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 1,
            reason: Failure::Refused
        }
    );
}

#[tokio::test]
async fn loopback_is_denied_unless_allowed() {
    let port = echo().await;
    let mut socket = page(&relay(false).await).await;
    send(&mut socket, tcp(1, port, 0)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 1,
            reason: Failure::Denied
        }
    );

    // The same address reached by name is judged by what it resolves to.
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 2,
            port,
            window: 0,
            peer: Peer::Name("localhost".into()),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 2,
            reason: Failure::Denied
        }
    );
}

#[tokio::test]
async fn a_name_that_does_not_resolve_is_not_found() {
    let mut socket = page(&relay(false).await).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            port: 80,
            window: 0,
            peer: Peer::Name("nothing.invalid".into()),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 1,
            reason: Failure::NotFound
        }
    );
}

/// A UDP server on loopback that answers each datagram from a second socket on another port as
/// well as its own, and the port of the first.
async fn udp_echo() -> (u16, u16) {
    let server = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let other = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let (port, other_port) = (
        server.local_addr().unwrap().port(),
        other.local_addr().unwrap().port(),
    );
    tokio::spawn(async move {
        let mut buffer = [0u8; 64];
        while let Ok((read, from)) = server.recv_from(&mut buffer).await {
            let _ = server.send_to(&buffer[..read], from).await;
            let _ = other.send_to(&buffer[..read], from).await;
        }
    });
    (port, other_port)
}

async fn udp_open(socket: &mut Socket, stream: u32) {
    send(
        socket,
        ClientFrame::UdpOpen {
            stream,
            port: 0,
            window: 1 << 16,
        },
    )
    .await;
    assert_eq!(next(socket).await, ServerFrame::Connected { stream });
}

fn send_to(stream: u32, port: u16, data: &[u8]) -> ClientFrame {
    ClientFrame::SendTo {
        stream,
        port,
        peer: Peer::Address([127, 0, 0, 1]),
        data: data.to_vec(),
    }
}

#[tokio::test]
async fn udp_datagrams_keep_their_boundaries_and_replies_come_from_any_port() {
    let (port, other_port) = udp_echo().await;
    let mut socket = page(&relay(true).await).await;
    udp_open(&mut socket, 1).await;
    for datagram in [&b"one"[..], b"second", b"3"] {
        send(&mut socket, send_to(1, port, datagram)).await;
        let mut ports = Vec::new();
        for _ in 0..2 {
            let ServerFrame::ReceivedFrom {
                stream: 1,
                port,
                peer: Peer::Address([127, 0, 0, 1]),
                data,
            } = next(&mut socket).await
            else {
                panic!("a datagram");
            };
            assert_eq!(data, datagram);
            ports.push(port);
        }
        ports.sort_unstable();
        let mut expected = vec![port, other_port];
        expected.sort_unstable();
        assert_eq!(ports, expected);
    }
}

#[tokio::test]
async fn one_udp_socket_sends_from_one_port_to_every_destination() {
    let first = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let second = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let mut socket = page(&relay(true).await).await;
    udp_open(&mut socket, 1).await;
    send(
        &mut socket,
        send_to(1, first.local_addr().unwrap().port(), b"a"),
    )
    .await;
    send(
        &mut socket,
        send_to(1, second.local_addr().unwrap().port(), b"b"),
    )
    .await;
    let mut buffer = [0u8; 8];
    let (_, from_first) = first.recv_from(&mut buffer).await.unwrap();
    let (_, from_second) = second.recv_from(&mut buffer).await.unwrap();
    assert_eq!(from_first.port(), from_second.port());
}

#[tokio::test]
async fn a_udp_socket_hears_only_hosts_it_sent_to_unless_inbound_is_allowed() {
    for (url, heard) in [(relay(true).await, false), (inbound_relay().await, true)] {
        let mut socket = page(&url).await;
        udp_open(&mut socket, 1).await;
        // Learn the relay's port from a datagram to a host that then says nothing back.
        let silent = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        send(
            &mut socket,
            send_to(1, silent.local_addr().unwrap().port(), b"x"),
        )
        .await;
        let mut buffer = [0u8; 8];
        let (_, relay_side) = silent.recv_from(&mut buffer).await.unwrap();

        // A stranger: on IPv6 loopback, which this socket never sent to.
        let stranger = UdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).await.unwrap();
        let to = std::net::SocketAddr::from((Ipv6Addr::LOCALHOST, relay_side.port()));
        stranger.send_to(b"hello", to).await.unwrap();
        if heard {
            let ServerFrame::ReceivedFrom { peer, data, .. } = next(&mut socket).await else {
                panic!("the stranger's datagram");
            };
            assert_eq!(peer, Peer::Address6(Ipv6Addr::LOCALHOST.octets()));
            assert_eq!(data, b"hello");
        } else {
            quiet(&mut socket).await;
        }
    }
}

#[tokio::test]
async fn a_closed_stream_can_be_reused_by_a_new_id_and_the_old_one_is_ignored() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(&mut socket, tcp(1, port, 100)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
    send(&mut socket, ClientFrame::Close { stream: 1 }).await;
    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: vec![1],
        },
    )
    .await;
    send(
        &mut socket,
        ClientFrame::Credit {
            stream: 1,
            bytes: 1,
        },
    )
    .await;
    send(&mut socket, tcp(2, port, 100)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 2 }
    );
    send(
        &mut socket,
        ClientFrame::Data {
            stream: 2,
            data: b"x".to_vec(),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Data {
            stream: 2,
            data: b"x".to_vec()
        }
    );
}

async fn closed_with(socket: &mut Socket) -> CloseCode {
    loop {
        match tokio::time::timeout(Duration::from_secs(5), socket.next())
            .await
            .expect("the relay closes")
        {
            Some(Ok(Message::Close(Some(frame)))) => return frame.code,
            Some(Ok(_)) => continue,
            other => panic!("closed without a code: {other:?}"),
        }
    }
}

#[tokio::test]
async fn an_untrusted_or_missing_origin_is_closed_before_hello() {
    let url = relay(true).await;
    for origin in [Some("https://evil.example"), None] {
        let mut socket = connect(&url, origin).await;
        assert_eq!(
            closed_with(&mut socket).await,
            CloseCode::Policy,
            "{origin:?}"
        );
    }
}

#[tokio::test]
async fn a_malformed_frame_or_a_reused_stream_id_closes_the_socket() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    socket
        .send(Message::Binary(vec![0x7f, 0, 0, 0, 0].into()))
        .await
        .unwrap();
    assert_eq!(closed_with(&mut socket).await, CloseCode::Protocol);

    let mut socket = page(&url).await;
    send(&mut socket, tcp(1, port, 0)).await;
    send(&mut socket, tcp(1, port, 0)).await;
    assert_eq!(closed_with(&mut socket).await, CloseCode::Protocol);

    let mut socket = page(&url).await;
    socket.send(Message::text("hello")).await.unwrap();
    assert_eq!(closed_with(&mut socket).await, CloseCode::Unsupported);
}

fn is_alive(message: &Message) -> bool {
    matches!(message, Message::Binary(raw) if ServerFrame::decode(raw) == Ok(ServerFrame::Alive))
}

/// Keep reading for `span`, answering pings: how many pings and `ALIVE` frames came. Nothing
/// else may arrive.
async fn keepalives_during(socket: &mut Socket, span: Duration) -> (usize, usize) {
    let until = tokio::time::Instant::now() + span;
    let (mut pings, mut alive) = (0, 0);
    while let Ok(message) = tokio::time::timeout_at(until, socket.next()).await {
        match message {
            Some(Ok(Message::Ping(_))) => pings += 1,
            Some(Ok(message)) if is_alive(&message) => alive += 1,
            other => panic!("only pings and ALIVE, got {other:?}"),
        }
    }
    (pings, alive)
}

#[tokio::test]
async fn a_page_that_answers_pings_stays_connected_however_quiet_it_is() {
    let (url, port) = (relay_with(true, BRISK).await, echo().await);
    let mut socket = page(&url).await;
    send(&mut socket, tcp(1, port, 100)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );

    let (pings, alive) = keepalives_during(&mut socket, BRISK.silence * 3).await;
    assert!(pings >= 5, "pinged {pings} times");
    // One of the pair can land just past the end of the wait.
    assert!(
        alive.abs_diff(pings) <= 1,
        "{alive} ALIVE for {pings} pings"
    );

    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: b"still here".to_vec(),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Data {
            stream: 1,
            data: b"still here".to_vec()
        }
    );
}

#[tokio::test]
async fn a_page_that_stops_answering_is_dropped_with_its_streams() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let peer_closed = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0u8; 64];
        while socket.read(&mut buffer).await.is_ok_and(|read| read > 0) {}
    });
    let mut socket = page(&relay_with(true, BRISK).await).await;
    send(&mut socket, tcp(1, port, 100)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );

    // Not polling the socket is what leaves the relay's pings unanswered.
    tokio::time::timeout(Duration::from_secs(5), peer_closed)
        .await
        .expect("the relay closes the guest's connection")
        .unwrap();
    loop {
        match tokio::time::timeout(Duration::from_secs(5), socket.next())
            .await
            .expect("the relay hangs up")
        {
            Some(Ok(Message::Ping(_))) => continue,
            Some(Ok(message)) if is_alive(&message) => continue,
            None | Some(Err(_)) => break,
            other => panic!("dropped without a close frame, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_connection_that_never_finishes_the_upgrade_is_closed() {
    let url = relay_with(true, BRISK).await;
    let address = url.trim_start_matches("ws://");
    for sent in [&b""[..], b"GET / HTTP/1.1\r\nHost: relay\r\n"] {
        let mut socket = TcpStream::connect(address).await.unwrap();
        socket.write_all(sent).await.unwrap();
        let mut buffer = [0u8; 64];
        let read = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buffer))
            .await
            .expect("the relay hangs up");
        assert!(
            read.is_err() || read.is_ok_and(|read| read == 0),
            "{sent:?}"
        );
    }
}

/// An echo server like [`echo`], listening only on `[::1]`.
async fn echo_v6() -> u16 {
    let listener = TcpListener::bind((Ipv6Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buffer = [0u8; 1024];
                while let Ok(read) = socket.read(&mut buffer).await {
                    if read == 0 || socket.write_all(&buffer[..read]).await.is_err() {
                        break;
                    }
                }
            });
        }
    });
    port
}

#[tokio::test]
async fn a_name_reaches_its_ipv6_address_when_the_ipv4_one_refuses() {
    let (url, port) = (relay(true).await, echo_v6().await);
    let mut socket = page(&url).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            port,
            window: 100,
            peer: Peer::Name("localhost".into()),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: b"v6".to_vec(),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Data {
            stream: 1,
            data: b"v6".to_vec()
        }
    );
}

#[tokio::test]
async fn ipv6_loopback_is_denied_unless_allowed() {
    let port = echo_v6().await;
    let mut socket = page(&relay(false).await).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            port,
            window: 0,
            peer: Peer::Name("::1".into()),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 1,
            reason: Failure::Denied
        }
    );
}

#[tokio::test]
async fn data_larger_than_a_frame_may_carry_closes_the_socket() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(&mut socket, tcp(1, port, 0)).await;
    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: vec![0; switch_relay_proto::MAX_DATA + 1],
        },
    )
    .await;
    assert_eq!(closed_with(&mut socket).await, CloseCode::Protocol);
}

#[tokio::test]
async fn a_name_falls_back_to_ipv4_when_ipv6_refuses() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            port,
            window: 100,
            peer: Peer::Name("localhost".into()),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
}

/// A server that takes one connection and hands it to the test.
async fn one_connection() -> (u16, tokio::sync::oneshot::Receiver<TcpStream>) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (give, take) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let _ = give.send(socket);
    });
    (port, take)
}

#[tokio::test]
async fn written_bytes_are_acknowledged() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(&mut socket, tcp(1, port, 0)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: vec![7; 300],
        },
    )
    .await;
    assert_eq!(
        next_raw(&mut socket).await,
        ServerFrame::Sent {
            stream: 1,
            bytes: 300
        }
    );
}

#[tokio::test]
async fn the_guest_may_send_after_the_peer_finishes() {
    let (port, accepted) = one_connection().await;
    let mut socket = page(&relay(true).await).await;
    send(&mut socket, tcp(1, port, 100)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
    let mut peer = accepted.await.unwrap();
    peer.shutdown().await.unwrap();
    assert_eq!(next(&mut socket).await, ServerFrame::Closed { stream: 1 });

    send(
        &mut socket,
        ClientFrame::Data {
            stream: 1,
            data: b"still".to_vec(),
        },
    )
    .await;
    let mut buffer = [0u8; 8];
    let read = peer.read(&mut buffer).await.unwrap();
    assert_eq!(&buffer[..read], b"still");
}

#[tokio::test]
async fn a_peer_that_resets_is_a_reset_not_an_end() {
    let (port, accepted) = one_connection().await;
    let mut socket = page(&relay(true).await).await;
    send(&mut socket, tcp(1, port, 100)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
    let peer = accepted.await.unwrap();
    // Closing with a zero linger sends RST instead of FIN.
    peer.set_zero_linger().unwrap();
    drop(peer);
    assert_eq!(next(&mut socket).await, ServerFrame::Reset { stream: 1 });
    quiet(&mut socket).await;
}

#[tokio::test]
async fn listening_needs_inbound_and_then_announces_each_connection() {
    let mut socket = page(&relay(true).await).await;
    let listen = ClientFrame::Listen {
        stream: 1,
        port: 0,
        window: 100,
    };
    send(&mut socket, listen.clone()).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 1,
            reason: Failure::Denied
        }
    );

    // Find a free port, then listen on it through the relay.
    let free = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0)).await.unwrap();
    let port = free.local_addr().unwrap().port();
    drop(free);
    let mut socket = page(&inbound_relay().await).await;
    send(
        &mut socket,
        ClientFrame::Listen {
            stream: 1,
            port,
            window: 100,
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );

    let mut client = TcpStream::connect((Ipv4Addr::LOCALHOST, port))
        .await
        .unwrap();
    let ServerFrame::Incoming {
        stream,
        listener: 1,
        peer: Peer::Address([127, 0, 0, 1]),
        ..
    } = next(&mut socket).await
    else {
        panic!("an incoming connection");
    };
    assert!(stream >= switch_relay_proto::RELAY_STREAMS);
    client.write_all(b"hi").await.unwrap();
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Data {
            stream,
            data: b"hi".to_vec()
        }
    );
    send(
        &mut socket,
        ClientFrame::Data {
            stream,
            data: b"back".to_vec(),
        },
    )
    .await;
    let mut buffer = [0u8; 8];
    let read = client.read(&mut buffer).await.unwrap();
    assert_eq!(&buffer[..read], b"back");

    // A second listener on the same port is told the port is taken.
    send(
        &mut socket,
        ClientFrame::Listen {
            stream: 2,
            port,
            window: 100,
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 2,
            reason: Failure::InUse
        }
    );
}

#[tokio::test]
async fn a_page_may_not_use_the_relay_s_stream_numbers() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(&mut socket, tcp(switch_relay_proto::RELAY_STREAMS, port, 0)).await;
    assert_eq!(closed_with(&mut socket).await, CloseCode::Protocol);
}

#[tokio::test]
async fn a_reply_from_a_host_reached_by_name_comes_from_that_name() {
    let (port, _) = udp_echo().await;
    let mut socket = page(&relay(true).await).await;
    udp_open(&mut socket, 1).await;
    send(
        &mut socket,
        ClientFrame::SendTo {
            stream: 1,
            port,
            peer: Peer::Name("localhost".into()),
            data: b"q".to_vec(),
        },
    )
    .await;
    let ServerFrame::ReceivedFrom { peer, data, .. } = next(&mut socket).await else {
        panic!("the reply");
    };
    assert_eq!(peer, Peer::Name("localhost".into()));
    assert_eq!(data, b"q");
}

fn opening_up() -> SettingsChange {
    SettingsChange {
        port: 18787,
        allow_private: true,
        allow_inbound: false,
        origins: vec!["https://added.example".into()],
    }
}

async fn settings(socket: &mut Socket) -> RelaySettings {
    send(socket, ClientFrame::GetSettings).await;
    match next(socket).await {
        ServerFrame::Settings(settings) => settings,
        other => panic!("settings, got {other:?}"),
    }
}

#[tokio::test]
async fn a_page_reads_the_settings_over_its_link() {
    let overrides = Overrides {
        allow_inbound: true,
        origins: vec!["https://flag.example".into()],
        ..Overrides::default()
    };
    let url = relay_config(overrides, Approval::Deny, Timeouts::default()).await;
    let mut socket = page(&url).await;
    let port: u16 = url.rsplit(':').next().unwrap().parse().unwrap();
    assert_eq!(
        settings(&mut socket).await,
        RelaySettings {
            listening: port,
            port: 18787,
            allow_private: false,
            allow_inbound: true,
            saved: false,
            port_locked: false,
            private_locked: false,
            inbound_locked: true,
            origins: Vec::new(),
            fixed_origins: vec![
                "http://localhost:8000".into(),
                "https://tenshii.moe".into(),
                "https://flag.example".into(),
            ],
        }
    );
}

#[tokio::test]
async fn opening_the_relay_up_waits_for_a_yes() {
    let url = relay_config(Overrides::default(), Approval::Deny, Timeouts::default()).await;
    let mut socket = page(&url).await;
    send(&mut socket, ClientFrame::SetSettings(opening_up())).await;
    assert_eq!(next(&mut socket).await, ServerFrame::Asking);
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::SettingsRefused(SettingsRefusal::Denied)
    );

    let port = echo().await;
    send(&mut socket, tcp(1, port, 1024)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 1,
            reason: Failure::Denied
        }
    );
    let mut added = connect(&url, Some("https://added.example")).await;
    assert_eq!(closed_with(&mut added).await, CloseCode::Policy);
}

#[tokio::test]
async fn a_yes_changes_the_running_relay() {
    let url = relay_config(Overrides::default(), Approval::Allow, Timeouts::default()).await;
    let mut socket = page(&url).await;
    send(&mut socket, ClientFrame::SetSettings(opening_up())).await;
    assert_eq!(next(&mut socket).await, ServerFrame::Asking);
    let ServerFrame::Settings(changed) = next(&mut socket).await else {
        panic!("the new settings");
    };
    assert!(changed.allow_private);
    assert_eq!(changed.origins, ["https://added.example"]);

    let port = echo().await;
    send(&mut socket, tcp(1, port, 1024)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );

    let mut added = connect(&url, Some("https://added.example")).await;
    assert!(matches!(next(&mut added).await, ServerFrame::Hello { .. }));

    // Closing back down needs no answer, so there is no `ASKING` before it.
    let closing = SettingsChange {
        allow_private: false,
        ..opening_up()
    };
    send(&mut socket, ClientFrame::SetSettings(closing)).await;
    let ServerFrame::Settings(closed) = next(&mut socket).await else {
        panic!("the new settings");
    };
    assert!(!closed.allow_private);
    send(&mut socket, tcp(2, port, 1024)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Failed {
            stream: 2,
            reason: Failure::Denied
        }
    );
}

#[tokio::test]
async fn a_change_that_is_not_one_is_refused() {
    let url = relay_config(Overrides::default(), Approval::Allow, Timeouts::default()).await;
    let mut socket = page(&url).await;
    let bad = SettingsChange {
        origins: vec!["https://added.example/path".into()],
        ..opening_up()
    };
    send(&mut socket, ClientFrame::SetSettings(bad)).await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::SettingsRefused(SettingsRefusal::Invalid)
    );
}
