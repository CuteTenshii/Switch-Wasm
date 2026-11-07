//! One guest connection: resolve, connect, then move bytes both ways under the guest's credit.

use std::net::{Ipv4Addr, SocketAddrV4};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Duration;

use switch_relay_proto::{Failure, Peer, Proto, ServerFrame, StreamId, MAX_DATA};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{lookup_host, TcpStream, UdpSocket};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Notify;

use crate::policy::Policy;

/// Most window the relay honours for one stream.
pub const MAX_WINDOW: u32 = 1 << 20;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Bytes the relay may still send on a stream before the guest reads more.
pub struct Credit {
    bytes: AtomicU32,
    notify: Notify,
}

impl Credit {
    pub fn new(window: u32) -> Credit {
        Credit {
            bytes: AtomicU32::new(window.min(MAX_WINDOW)),
            notify: Notify::new(),
        }
    }

    pub fn add(&self, bytes: u32) {
        let _ = self.bytes.fetch_update(Relaxed, Relaxed, |held| {
            Some(held.saturating_add(bytes).min(MAX_WINDOW))
        });
        self.notify.notify_one();
    }

    /// Wait until there is credit, then say how much.
    async fn available(&self) -> usize {
        loop {
            let held = self.bytes.load(Relaxed);
            if held > 0 {
                return held as usize;
            }
            self.notify.notified().await;
        }
    }

    fn spend(&self, bytes: usize) {
        self.bytes.fetch_sub(bytes as u32, Relaxed);
    }

    /// Spend `bytes` only if all of them are covered.
    fn try_spend(&self, bytes: usize) -> bool {
        self.bytes
            .fetch_update(Relaxed, Relaxed, |held| held.checked_sub(bytes as u32))
            .is_ok()
    }
}

/// A task that ends with its owner, including when the owner is aborted.
struct Reader(tokio::task::JoinHandle<()>);

impl Drop for Reader {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// What the session tells a stream to do.
pub enum Command {
    Data(Vec<u8>),
    ShutdownWrite,
}

pub struct Open {
    pub id: StreamId,
    pub proto: Proto,
    pub port: u16,
    pub peer: Peer,
}

/// Run a stream to its end. `done` hears the id when nothing more can happen on it.
pub async fn run(
    open: Open,
    policy: Arc<Policy>,
    credit: Arc<Credit>,
    out: UnboundedSender<ServerFrame>,
    commands: UnboundedReceiver<Command>,
    done: UnboundedSender<StreamId>,
) {
    let id = open.id;
    let target = match &open.peer {
        Peer::Address([a, b, c, d]) => format!("{a}.{b}.{c}.{d}:{}", open.port),
        Peer::Name(name) => format!("{name}:{}", open.port),
    };
    let proto = if open.proto == Proto::Udp {
        "udp"
    } else {
        "tcp"
    };
    eprintln!("#{id} {proto} {target}");
    match connect(&open, &policy).await {
        Err(reason) => {
            eprintln!("#{id} failed: {reason:?}");
            let _ = out.send(ServerFrame::Failed { stream: id, reason });
        }
        Ok(Connection::Tcp(socket)) => {
            eprintln!("#{id} open");
            let _ = out.send(ServerFrame::Connected { stream: id });
            pump_tcp(id, socket, credit, out, commands).await;
        }
        Ok(Connection::Udp(socket)) => {
            eprintln!("#{id} open");
            let _ = out.send(ServerFrame::Connected { stream: id });
            pump_udp(id, socket, credit, out, commands).await;
        }
    }
    let _ = done.send(id);
}

enum Connection {
    Tcp(TcpStream),
    Udp(UdpSocket),
}

async fn connect(open: &Open, policy: &Policy) -> Result<Connection, Failure> {
    let address = resolve(&open.peer, open.port, policy).await?;
    match open.proto {
        Proto::Tcp => {
            let socket = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(address))
                .await
                .map_err(|_| Failure::TimedOut)?
                .map_err(|error| match error.kind() {
                    std::io::ErrorKind::ConnectionRefused => Failure::Refused,
                    std::io::ErrorKind::TimedOut => Failure::TimedOut,
                    _ => Failure::Unreachable,
                })?;
            let _ = socket.set_nodelay(true);
            Ok(Connection::Tcp(socket))
        }
        Proto::Udp => {
            let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
                .await
                .map_err(|_| Failure::Unreachable)?;
            socket
                .connect(address)
                .await
                .map_err(|_| Failure::Unreachable)?;
            Ok(Connection::Udp(socket))
        }
    }
}

/// The first allowed IPv4 address of the peer. The policy judges what a name resolved to,
/// not the name.
async fn resolve(peer: &Peer, port: u16, policy: &Policy) -> Result<SocketAddrV4, Failure> {
    let candidates: Vec<SocketAddrV4> = match peer {
        Peer::Address(ip) => vec![SocketAddrV4::new(Ipv4Addr::from(*ip), port)],
        Peer::Name(name) => lookup_host((name.as_str(), port))
            .await
            .map_err(|_| Failure::NotFound)?
            .filter_map(|address| match address {
                std::net::SocketAddr::V4(address) => Some(address),
                std::net::SocketAddr::V6(_) => None,
            })
            .collect(),
    };
    if candidates.is_empty() {
        return Err(Failure::NotFound);
    }
    candidates
        .into_iter()
        .find(|address| policy.address_allowed(*address.ip()))
        .ok_or(Failure::Denied)
}

async fn pump_tcp(
    id: StreamId,
    socket: TcpStream,
    credit: Arc<Credit>,
    out: UnboundedSender<ServerFrame>,
    mut commands: UnboundedReceiver<Command>,
) {
    let (mut reader, mut writer) = socket.into_split();
    let _reading = Reader(tokio::spawn({
        let out = out.clone();
        async move {
            let mut buffer = vec![0u8; MAX_DATA];
            loop {
                let room = credit.available().await.min(MAX_DATA);
                match reader.read(&mut buffer[..room]).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        credit.spend(read);
                        let data = buffer[..read].to_vec();
                        if out.send(ServerFrame::Data { stream: id, data }).is_err() {
                            return;
                        }
                    }
                }
            }
            let _ = out.send(ServerFrame::Closed { stream: id });
        }
    }));
    while let Some(command) = commands.recv().await {
        let result = match command {
            Command::Data(data) => writer.write_all(&data).await,
            Command::ShutdownWrite => writer.shutdown().await,
        };
        if result.is_err() {
            let _ = out.send(ServerFrame::Closed { stream: id });
            break;
        }
    }
}

async fn pump_udp(
    id: StreamId,
    socket: UdpSocket,
    credit: Arc<Credit>,
    out: UnboundedSender<ServerFrame>,
    mut commands: UnboundedReceiver<Command>,
) {
    let socket = Arc::new(socket);
    let _reading = Reader(tokio::spawn({
        let (socket, out) = (socket.clone(), out.clone());
        async move {
            let mut buffer = vec![0u8; 65536];
            loop {
                let Ok(read) = socket.recv(&mut buffer).await else {
                    break;
                };
                // A datagram that does not fit a frame or the guest's credit is lost, as on a link.
                if read <= MAX_DATA && credit.try_spend(read) {
                    let data = buffer[..read].to_vec();
                    if out.send(ServerFrame::Data { stream: id, data }).is_err() {
                        return;
                    }
                }
            }
            let _ = out.send(ServerFrame::Closed { stream: id });
        }
    }));
    while let Some(command) = commands.recv().await {
        if let Command::Data(data) = command {
            let _ = socket.send(&data).await;
        }
    }
}
