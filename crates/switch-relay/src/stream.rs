//! One stream: a guest's TCP connection, UDP socket or listener, and what it does on the host.

use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use socket2::{Domain, Protocol, Socket, Type};
use switch_relay_proto::{Failure, Peer, ServerFrame, StreamId, MAX_DATA, MAX_DATAGRAM};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{lookup_host, TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Notify;
use tokio::task::JoinSet;

use crate::policy::Policy;

/// Most window the relay honours for one stream.
pub const MAX_WINDOW: u32 = 1 << 20;

/// Longest a connect may take, the name lookup included.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// How long one address gets before the next is tried alongside it.
const ATTEMPT_DELAY: Duration = Duration::from_millis(250);

/// How long a UDP socket keeps what a name resolved to.
const NAME_TTL: Duration = Duration::from_secs(60);

/// Longest a UDP socket waits for a name to resolve before dropping the datagram.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Pause after a socket call that failed, which is usually the process out of descriptors:
/// retrying at once would spin until one is freed.
const RETRY_PAUSE: Duration = Duration::from_millis(250);

/// Connections a listener's socket queues before the relay accepts them.
const BACKLOG: i32 = 128;

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
        let _ = self.bytes.try_update(Relaxed, Relaxed, |held| {
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
            .try_update(Relaxed, Relaxed, |held| held.checked_sub(bytes as u32))
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

/// Most bytes of `DATA` and `SEND_TO` a stream may have waiting for it. The page keeps far
/// less unacknowledged; only a page that ignores `SENT`, or sends to a stream that takes
/// nothing, such as a listener, gets here.
pub const MAX_BACKLOG: usize = 4 << 20;

/// What the session tells a stream to do.
pub enum Command {
    Data(Vec<u8>),
    ShutdownWrite,
    SendTo {
        port: u16,
        peer: Peer,
        data: Vec<u8>,
    },
}

impl Command {
    fn len(&self) -> usize {
        match self {
            Command::Data(data) | Command::SendTo { data, .. } => data.len(),
            Command::ShutdownWrite => 0,
        }
    }
}

/// A stream's command queue, from the session's end.
pub struct Commands {
    sender: UnboundedSender<Command>,
    backlog: Arc<AtomicUsize>,
}

impl Commands {
    /// Queue `command`; `false`, with nothing queued, when it would take the stream's backlog
    /// past [`MAX_BACKLOG`].
    pub fn send(&self, command: Command) -> bool {
        let len = command.len();
        let fits = self
            .backlog
            .try_update(Relaxed, Relaxed, |held| {
                held.checked_add(len).filter(|&total| total <= MAX_BACKLOG)
            })
            .is_ok();
        if fits {
            let _ = self.sender.send(command);
        }
        fits
    }
}

/// A stream's command queue, from the stream's end.
pub struct CommandQueue {
    receiver: UnboundedReceiver<Command>,
    backlog: Arc<AtomicUsize>,
}

impl CommandQueue {
    async fn recv(&mut self) -> Option<Command> {
        let command = self.receiver.recv().await?;
        self.backlog.fetch_sub(command.len(), Relaxed);
        Some(command)
    }
}

/// A new stream's command queue.
pub fn commands() -> (Commands, CommandQueue) {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let backlog = Arc::new(AtomicUsize::new(0));
    (
        Commands {
            sender,
            backlog: backlog.clone(),
        },
        CommandQueue { receiver, backlog },
    )
}

pub enum Kind {
    Tcp {
        port: u16,
        peer: Peer,
    },
    Udp {
        port: u16,
    },
    /// Each connection accepted gets `window`.
    Listen {
        port: u16,
        window: u32,
    },
}

pub struct Open {
    pub id: StreamId,
    pub kind: Kind,
}

/// A connection a listener accepted, for the session to make a stream of.
pub struct Accepted {
    pub listener: StreamId,
    pub socket: TcpStream,
    pub from: SocketAddr,
    pub window: u32,
}

/// Where a stream's frames go, and who hears when it ends.
pub struct Links {
    pub out: UnboundedSender<ServerFrame>,
    pub commands: CommandQueue,
    pub done: UnboundedSender<StreamId>,
}

/// Run a stream to its end. `done` hears the id when nothing more can happen on it.
pub async fn run(
    open: Open,
    policy: Arc<Policy>,
    credit: Arc<Credit>,
    links: Links,
    accepted: UnboundedSender<Accepted>,
) {
    let Links {
        out,
        commands,
        done,
    } = links;
    let id = open.id;
    match open.kind {
        Kind::Tcp { port, peer } => {
            eprintln!("#{id} tcp {}", describe(&peer, port));
            let connected = tokio::time::timeout(CONNECT_TIMEOUT, connect(&peer, port, &policy))
                .await
                .unwrap_or(Err(Failure::TimedOut));
            match connected {
                Err(reason) => fail(id, reason, &out),
                Ok(socket) => {
                    eprintln!("#{id} open");
                    let _ = out.send(ServerFrame::Connected { stream: id });
                    pump_tcp(id, socket, credit, out, commands).await;
                }
            }
        }
        Kind::Udp { port } => match open_udp(port, &policy) {
            Err(reason) => fail(id, reason, &out),
            Ok(sockets) => {
                eprintln!("#{id} udp on port {}", sockets.port());
                let _ = out.send(ServerFrame::Connected { stream: id });
                pump_udp(id, sockets, policy, credit, out, commands).await;
            }
        },
        Kind::Listen { port, window } => match open_listener(port, &policy) {
            Err(reason) => fail(id, reason, &out),
            Ok(listeners) => {
                eprintln!("#{id} listening on port {}", listeners.port());
                let _ = out.send(ServerFrame::Connected { stream: id });
                listen(id, listeners, window, accepted).await;
            }
        },
    }
    let _ = done.send(id);
}

/// Run a connection a listener accepted, which the session has already announced.
pub async fn run_accepted(id: StreamId, socket: TcpStream, credit: Arc<Credit>, links: Links) {
    let _ = socket.set_nodelay(true);
    pump_tcp(id, socket, credit, links.out, links.commands).await;
    let _ = links.done.send(id);
}

fn fail(id: StreamId, reason: Failure, out: &UnboundedSender<ServerFrame>) {
    eprintln!("#{id} failed: {reason:?}");
    let _ = out.send(ServerFrame::Failed { stream: id, reason });
}

fn describe(peer: &Peer, port: u16) -> String {
    match peer {
        Peer::Address(ip) => SocketAddr::from((*ip, port)).to_string(),
        Peer::Address6(ip) => SocketAddr::from((*ip, port)).to_string(),
        Peer::Name(name) => format!("{name}:{port}"),
    }
}

/// The address an IPv4-mapped IPv6 address stands for; any other unchanged.
fn unmapped(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        IpAddr::V4(_) => ip,
    }
}

/// How the page is told about an address.
pub fn peer_of(ip: IpAddr) -> Peer {
    match unmapped(ip) {
        IpAddr::V4(v4) => Peer::Address(v4.octets()),
        IpAddr::V6(v6) => Peer::Address6(v6.octets()),
    }
}

async fn connect(peer: &Peer, port: u16, policy: &Policy) -> Result<TcpStream, Failure> {
    let addresses = resolve(peer, port, policy).await?;
    let socket = connect_tcp(addresses).await?;
    let _ = socket.set_nodelay(true);
    Ok(socket)
}

/// Every address of the peer the policy allows, the two families taking turns from whichever
/// the resolver put first, as happy eyeballs (RFC 8305) orders them. The policy judges what a
/// name resolved to, not the name.
async fn resolve(peer: &Peer, port: u16, policy: &Policy) -> Result<Vec<SocketAddr>, Failure> {
    let found: Vec<SocketAddr> = match peer {
        Peer::Address(ip) => vec![SocketAddr::from((*ip, port))],
        Peer::Address6(ip) => vec![SocketAddr::from((*ip, port))],
        Peer::Name(name) => lookup_host((name.as_str(), port))
            .await
            .map_err(|_| Failure::NotFound)?
            .collect(),
    };
    if found.is_empty() {
        return Err(Failure::NotFound);
    }
    let allowed: Vec<SocketAddr> = found
        .into_iter()
        .filter(|address| policy.address_allowed(address.ip()))
        .collect();
    let first_v6 = allowed.first().ok_or(Failure::Denied)?.is_ipv6();
    let (mut same, mut other): (VecDeque<_>, VecDeque<_>) = allowed
        .into_iter()
        .partition(|address| address.is_ipv6() == first_v6);
    let mut ordered = Vec::new();
    while let Some(address) = same.pop_front() {
        ordered.push(address);
        ordered.extend(other.pop_front());
    }
    ordered.extend(other);
    Ok(ordered)
}

/// Connect to the first of `addresses` that answers. Each attempt gets a head start of
/// [`ATTEMPT_DELAY`] before the next begins, and a refusal starts the next at once.
async fn connect_tcp(addresses: Vec<SocketAddr>) -> Result<TcpStream, Failure> {
    let mut waiting = addresses.into_iter();
    let mut attempts = JoinSet::new();
    let mut last = Failure::Unreachable;
    loop {
        if let Some(address) = waiting.next() {
            attempts.spawn(TcpStream::connect(address));
        }
        let more = waiting.len() > 0;
        tokio::select! {
            joined = attempts.join_next() => match joined {
                Some(Ok(Ok(socket))) => return Ok(socket),
                Some(Ok(Err(error))) => {
                    last = match error.kind() {
                        std::io::ErrorKind::ConnectionRefused => Failure::Refused,
                        std::io::ErrorKind::TimedOut => Failure::TimedOut,
                        _ => Failure::Unreachable,
                    };
                    if attempts.is_empty() && !more {
                        return Err(last);
                    }
                }
                Some(Err(_)) => {}
                None if !more => return Err(last),
                None => {}
            },
            () = tokio::time::sleep(ATTEMPT_DELAY), if more => {}
        }
    }
}

/// Bytes both ways under the guest's credit. The peer finishing is `CLOSED`, and the guest may
/// still send; a reset or a failed write is `RESET`, after which nothing more is said.
async fn pump_tcp(
    id: StreamId,
    socket: TcpStream,
    credit: Arc<Credit>,
    out: UnboundedSender<ServerFrame>,
    mut commands: CommandQueue,
) {
    let (mut reader, mut writer) = socket.into_split();
    let broke = Arc::new(Notify::new());
    let _reading = Reader(tokio::spawn({
        let (out, broke) = (out.clone(), broke.clone());
        async move {
            let mut buffer = vec![0u8; MAX_DATA];
            loop {
                let room = credit.available().await.min(MAX_DATA);
                match reader.read(&mut buffer[..room]).await {
                    Ok(0) => {
                        let _ = out.send(ServerFrame::Closed { stream: id });
                        return;
                    }
                    Err(_) => {
                        broke.notify_one();
                        return;
                    }
                    Ok(read) => {
                        credit.spend(read);
                        let data = buffer[..read].to_vec();
                        if out.send(ServerFrame::Data { stream: id, data }).is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }));
    loop {
        tokio::select! {
            command = commands.recv() => {
                let written = match command {
                    None => return,
                    Some(Command::Data(data)) => writer.write_all(&data).await.map(|()| data.len()),
                    Some(Command::ShutdownWrite) => writer.shutdown().await.map(|()| 0),
                    Some(Command::SendTo { .. }) => Ok(0),
                };
                match written {
                    Err(_) => break,
                    Ok(0) => {}
                    Ok(bytes) => {
                        let _ = out.send(ServerFrame::Sent { stream: id, bytes: bytes as u32 });
                    }
                }
            }
            () = broke.notified() => break,
        }
    }
    eprintln!("#{id} reset");
    let _ = out.send(ServerFrame::Reset { stream: id });
}

/// A UDP socket of each family the host has, on the same port where it can be.
struct UdpSockets {
    v4: Option<UdpSocket>,
    v6: Option<UdpSocket>,
}

impl UdpSockets {
    fn port(&self) -> u16 {
        self.v4
            .iter()
            .chain(&self.v6)
            .find_map(|socket| socket.local_addr().ok())
            .map_or(0, |address| address.port())
    }

    fn for_address(&self, address: SocketAddr) -> Option<&UdpSocket> {
        if address.is_ipv4() {
            self.v4.as_ref()
        } else {
            self.v6.as_ref()
        }
    }
}

/// A socket bound to `address`; an IPv6 one leaves IPv4 to its own socket.
fn raw_socket(address: SocketAddr, kind: Type) -> std::io::Result<Socket> {
    let protocol = if kind == Type::DGRAM {
        Protocol::UDP
    } else {
        Protocol::TCP
    };
    let socket = Socket::new(Domain::for_address(address), kind, Some(protocol))?;
    if address.is_ipv6() {
        socket.set_only_v6(true)?;
    }
    if kind == Type::STREAM {
        socket.set_reuse_address(true)?;
    }
    socket.set_nonblocking(true)?;
    socket.bind(&address.into())?;
    Ok(socket)
}

fn bind_udp(address: SocketAddr) -> Option<UdpSocket> {
    let socket = raw_socket(address, Type::DGRAM).ok()?;
    UdpSocket::from_std(socket.into()).ok()
}

/// The guest's port when the relay takes inbound traffic and it is free, any port otherwise:
/// a socket that cannot be reached from outside has no use for a particular one.
fn open_udp(port: u16, policy: &Policy) -> Result<UdpSockets, Failure> {
    let wanted = if policy.inbound() { port } else { 0 };
    let v4 = bind_udp((Ipv4Addr::UNSPECIFIED, wanted).into())
        .or_else(|| bind_udp((Ipv4Addr::UNSPECIFIED, 0).into()));
    let port = v4
        .as_ref()
        .and_then(|socket| socket.local_addr().ok())
        .map_or(wanted, |address| address.port());
    let v6 = bind_udp((Ipv6Addr::UNSPECIFIED, port).into())
        .or_else(|| bind_udp((Ipv6Addr::UNSPECIFIED, 0).into()));
    if v4.is_none() && v6.is_none() {
        return Err(Failure::Unreachable);
    }
    Ok(UdpSockets { v4, v6 })
}

/// Datagrams both ways. Without inbound traffic allowed, only hosts this socket has sent to are
/// heard from, as a NAT that filters by address does; any port of theirs will do. A host the
/// guest named is reported by that name, so its reply comes from where the guest sent.
async fn pump_udp(
    id: StreamId,
    sockets: UdpSockets,
    policy: Arc<Policy>,
    credit: Arc<Credit>,
    out: UnboundedSender<ServerFrame>,
    mut commands: CommandQueue,
) {
    let sockets = Arc::new(sockets);
    // Each host sent to, and the name the guest gave it, if it gave one.
    let contacted: Arc<Mutex<HashMap<IpAddr, Option<String>>>> = Arc::default();
    let mut receiving = Vec::new();
    for family in [sockets.v4.is_some(), sockets.v6.is_some()]
        .into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index))
    {
        let (sockets, contacted, credit, out) = (
            sockets.clone(),
            contacted.clone(),
            credit.clone(),
            out.clone(),
        );
        let open = policy.inbound();
        receiving.push(Reader(tokio::spawn(async move {
            let socket = if family == 0 {
                sockets.v4.as_ref()
            } else {
                sockets.v6.as_ref()
            };
            let Some(socket) = socket else {
                return;
            };
            let mut buffer = vec![0u8; MAX_DATAGRAM + 1];
            loop {
                let Ok((read, from)) = socket.recv_from(&mut buffer).await else {
                    tokio::time::sleep(RETRY_PAUSE).await;
                    continue;
                };
                let ip = unmapped(from.ip());
                let known = contacted.lock().unwrap().get(&ip).cloned();
                // Too long for a frame, or past the guest's credit: lost, as on a link.
                if !(open || known.is_some()) || read > MAX_DATAGRAM || !credit.try_spend(read) {
                    continue;
                }
                let peer = known.flatten().map_or_else(|| peer_of(ip), Peer::Name);
                let frame = ServerFrame::ReceivedFrom {
                    stream: id,
                    port: from.port(),
                    peer,
                    data: buffer[..read].to_vec(),
                };
                if out.send(frame).is_err() {
                    return;
                }
            }
        })));
    }
    let mut names: HashMap<String, (IpAddr, Instant)> = HashMap::new();
    while let Some(command) = commands.recv().await {
        let Command::SendTo { port, peer, data } = command else {
            continue;
        };
        let Some(ip) = udp_destination(&peer, &sockets, &policy, &mut names).await else {
            continue;
        };
        let address = SocketAddr::new(ip, port);
        if let Some(socket) = sockets.for_address(address) {
            let name = match peer {
                Peer::Name(name) => Some(name),
                Peer::Address(_) | Peer::Address6(_) => None,
            };
            contacted.lock().unwrap().insert(ip, name);
            let _ = socket.send_to(&data, address).await;
        }
    }
}

/// Where a datagram to `peer` goes: an address the policy allows and this socket has a family
/// for, IPv4 first, since nothing tells a datagram that the IPv6 path is broken the way a refused
/// connect does. A name keeps what it resolved to for [`NAME_TTL`]. `None` drops the datagram.
async fn udp_destination(
    peer: &Peer,
    sockets: &UdpSockets,
    policy: &Policy,
    names: &mut HashMap<String, (IpAddr, Instant)>,
) -> Option<IpAddr> {
    let usable = |ip: IpAddr| {
        policy.address_allowed(ip) && sockets.for_address(SocketAddr::new(ip, 0)).is_some()
    };
    let name = match peer {
        Peer::Address(ip) => return Some(IpAddr::from(*ip)).filter(|&ip| usable(ip)),
        Peer::Address6(ip) => return Some(unmapped(IpAddr::from(*ip))).filter(|&ip| usable(ip)),
        Peer::Name(name) => name,
    };
    if let Some(&(ip, at)) = names.get(name) {
        if at.elapsed() < NAME_TTL {
            return Some(ip);
        }
    }
    let found = tokio::time::timeout(LOOKUP_TIMEOUT, lookup_host((name.as_str(), 0)))
        .await
        .ok()?
        .ok()?;
    let usable: Vec<IpAddr> = found
        .map(|address| unmapped(address.ip()))
        .filter(|&ip| usable(ip))
        .collect();
    let ip = *usable
        .iter()
        .find(|ip| ip.is_ipv4())
        .or_else(|| usable.first())?;
    names.insert(name.clone(), (ip, Instant::now()));
    Some(ip)
}

/// Listening sockets of each family the host has, on the same port.
struct Listeners {
    v4: Option<TcpListener>,
    v6: Option<TcpListener>,
}

impl Listeners {
    fn port(&self) -> u16 {
        self.v4
            .iter()
            .chain(&self.v6)
            .find_map(|listener| listener.local_addr().ok())
            .map_or(0, |address| address.port())
    }
}

fn bind_listener(address: SocketAddr) -> std::io::Result<TcpListener> {
    let socket = raw_socket(address, Type::STREAM)?;
    socket.listen(BACKLOG)?;
    TcpListener::from_std(socket.into())
}

/// Only when the relay takes inbound traffic. Unlike a UDP socket, a listener on another port
/// than the guest asked for is no use to anyone who was told the port, so a taken one fails.
fn open_listener(port: u16, policy: &Policy) -> Result<Listeners, Failure> {
    if !policy.inbound() {
        return Err(Failure::Denied);
    }
    let in_use = |error: std::io::Error| match error.kind() {
        std::io::ErrorKind::AddrInUse => Failure::InUse,
        _ => Failure::Unreachable,
    };
    let v4 = bind_listener((Ipv4Addr::UNSPECIFIED, port).into()).map_err(in_use)?;
    let port = v4.local_addr().map_or(port, |address| address.port());
    let v6 = bind_listener((Ipv6Addr::UNSPECIFIED, port).into()).ok();
    Ok(Listeners { v4: Some(v4), v6 })
}

async fn listen(
    id: StreamId,
    listeners: Listeners,
    window: u32,
    accepted: UnboundedSender<Accepted>,
) {
    let mut accepting = JoinSet::new();
    for listener in [listeners.v4, listeners.v6].into_iter().flatten() {
        let accepted = accepted.clone();
        accepting.spawn(async move {
            loop {
                let Ok((socket, from)) = listener.accept().await else {
                    tokio::time::sleep(RETRY_PAUSE).await;
                    continue;
                };
                eprintln!("#{id} accepted {from}");
                let connection = Accepted {
                    listener: id,
                    socket,
                    from,
                    window,
                };
                if accepted.send(connection).is_err() {
                    return;
                }
            }
        });
    }
    while accepting.join_next().await.is_some() {}
}
