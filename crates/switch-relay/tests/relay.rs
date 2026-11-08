//! A relay on a loopback port, driven over a real WebSocket against local servers.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use switch_relay::policy::{Policy, Unknown};
use switch_relay::Timeouts;
use switch_relay_proto::{ClientFrame, Failure, Peer, Proto, ServerFrame, VERSION};
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
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let policy = Policy::new(Vec::new(), Unknown::Deny, allow_private);
    tokio::spawn(switch_relay::serve(listener, Arc::new(policy), timeouts));
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

/// The next relay frame. Pings, which the socket answers as it reads, and `ALIVE` are skipped.
async fn next(socket: &mut Socket) -> ServerFrame {
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
        proto: Proto::Tcp,
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
            proto: Proto::Tcp,
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
            proto: Proto::Tcp,
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

#[tokio::test]
async fn udp_datagrams_keep_their_boundaries() {
    let server = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = server.local_addr().unwrap().port();
    tokio::spawn(async move {
        let mut buffer = [0u8; 64];
        while let Ok((read, from)) = server.recv_from(&mut buffer).await {
            let _ = server.send_to(&buffer[..read], from).await;
        }
    });
    let mut socket = page(&relay(true).await).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            proto: Proto::Udp,
            port,
            window: 1000,
            peer: Peer::Address([127, 0, 0, 1]),
        },
    )
    .await;
    assert_eq!(
        next(&mut socket).await,
        ServerFrame::Connected { stream: 1 }
    );
    for datagram in [&b"one"[..], b"second", b"3"] {
        send(
            &mut socket,
            ClientFrame::Data {
                stream: 1,
                data: datagram.to_vec(),
            },
        )
        .await;
        assert_eq!(
            next(&mut socket).await,
            ServerFrame::Data {
                stream: 1,
                data: datagram.to_vec()
            }
        );
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
            proto: Proto::Tcp,
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
async fn a_name_falls_back_to_ipv4_when_ipv6_refuses() {
    let (url, port) = (relay(true).await, echo().await);
    let mut socket = page(&url).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            proto: Proto::Tcp,
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

#[tokio::test]
async fn ipv6_loopback_is_denied_unless_allowed() {
    let port = echo_v6().await;
    let mut socket = page(&relay(false).await).await;
    send(
        &mut socket,
        ClientFrame::Connect {
            stream: 1,
            proto: Proto::Tcp,
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
