//! Datagram sockets. One the host carries has a single host UDP socket, opened by its `bind` or
//! its first `sendto` outside the console, which sends to every address from one port and hears
//! from whoever the host lets through; what arrives is queued with its source and its
//! boundaries. A datagram to this console goes straight to the socket bound to its port.

use super::*;
use crate::trace::Level;

/// Most datagrams held for a guest that is not reading. Past it they are dropped, as a full
/// socket buffer does, and their credit goes back to the host.
const MAX_QUEUED: usize = 256;

/// The largest UDP payload IPv4 carries; a longer `sendto` is `EMSGSIZE`.
pub(super) const MAX_DATAGRAM: usize = 65_507;

/// A datagram received: the host stream it came on (none from this console), where it came
/// from, and its bytes.
#[derive(Debug, Clone)]
pub(super) struct Received {
    stream: Option<StreamId>,
    source: Vec<u8>,
    payload: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Datagram {
    /// The host UDP socket this one sends and receives through.
    pub(super) host: Option<StreamId>,
    pub(super) received: VecDeque<Received>,
    /// Where `send` goes after a `connect`, and the only source heard from then.
    pub(super) default_peer: Option<([u8; 4], u16)>,
}

impl Datagram {
    /// Every byte of every datagram waiting, which is what FreeBSD's `FIONREAD` counts.
    pub(super) fn queued_bytes(&self) -> usize {
        self.received.iter().map(|r| r.payload.len()).sum()
    }

    /// The address a connected socket reports as its peer's.
    pub(super) fn default_address(&self) -> Option<Vec<u8>> {
        self.default_peer
            .map(|(ip, port)| sockaddr_in_bytes(ip, port))
    }

    /// Whether a datagram from `ip:port` reaches a socket connected to its peer.
    fn admits(&self, ip: [u8; 4], port: u16) -> bool {
        self.default_peer.is_none_or(|peer| peer == (ip, port))
    }
}

impl Cpu {
    /// `connect` on a datagram socket: it only names where `send` goes and whom to hear. This
    /// console, or with the network on any address the host carries; anything else is refused
    /// as it always was.
    pub(super) fn bsd_connect_datagram(&mut self, tls: u64, fd: i32, address: &[u8]) -> Result<()> {
        let enabled = self.net.enabled;
        let Some((ip, port)) = sockaddr_in(address)
            .filter(|&(ip, port)| port != 0 && (is_local_ip(ip) || (enabled && is_carried(ip))))
        else {
            return self.bsd_reply(tls, -1, BSD_ECONNREFUSED);
        };
        if is_carried(ip) {
            self.bsd_udp_host(fd);
        }
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            socket.datagram.default_peer = Some((ip, port));
        }
        self.bsd_reply(tls, 0, 0)
    }

    /// The socket's bound port, binding it to the console's address and a free port first if it
    /// has none.
    fn bsd_ensure_bound(&mut self, fd: i32) -> Option<u16> {
        let bound = self.process().bsd_sockets.get(&fd)?.bound.clone();
        if let Some((_, port)) = sockaddr_in(&bound) {
            return Some(port);
        }
        let port = self.bsd_assign_port();
        self.process_mut().bsd_sockets.get_mut(&fd)?.bound = sockaddr_in_bytes(NIFM_LOCAL_IP, port);
        Some(port)
    }

    /// The host UDP socket of datagram socket `fd`, opened if it has none, on the guest's port
    /// for the host to keep when it can. `None` with the network off.
    pub(super) fn bsd_udp_host(&mut self, fd: i32) -> Option<StreamId> {
        let socket = self.process().bsd_sockets.get(&fd)?;
        if !self.net.enabled || socket.kind != BSD_SOCK_DGRAM {
            return None;
        }
        if let Some(id) = socket.datagram.host {
            return Some(id);
        }
        let port = self.bsd_ensure_bound(fd)?;
        let id = self.net.open_stream();
        self.log(
            Level::Info,
            &format!("[net] udp fd {fd} on port {port} as stream {id}"),
        );
        self.net
            .outbox
            .push(NetCommand::OpenUdp { stream: id, port });
        self.process_mut().bsd_sockets.get_mut(&fd)?.datagram.host = Some(id);
        Some(id)
    }

    /// `sendto` (or `send` after a `connect`) to an address outside the console. With the
    /// network off the link takes it and drops it.
    pub(super) fn bsd_send_outside_datagram(
        &mut self,
        fd: i32,
        data: Vec<u8>,
        ip: [u8; 4],
        port: u16,
    ) -> (i32, i32) {
        let sent = data.len() as i32;
        if let Some(stream) = self.bsd_udp_host(fd) {
            let peer = self.net.peer_for(ip);
            self.net.outbox.push(NetCommand::SendTo {
                stream,
                peer,
                port,
                data,
            });
        }
        (sent, 0)
    }

    /// A datagram to this console: queued on the datagram socket bound to `port`, from the
    /// sender's port at the address it was sent to (loopback stays loopback). With no socket
    /// there, or one connected elsewhere, it is lost, as it would be.
    pub(super) fn bsd_send_local_datagram(
        &mut self,
        fd: i32,
        data: Vec<u8>,
        ip: [u8; 4],
        port: u16,
    ) -> (i32, i32) {
        let sent = data.len() as i32;
        let Some(from_port) = self.bsd_ensure_bound(fd) else {
            return (-1, BSD_EBADF);
        };
        let from_ip = if ip == BSD_LOOPBACK_IP {
            BSD_LOOPBACK_IP
        } else {
            NIFM_LOCAL_IP
        };
        let receiver = self.bsd_descriptors().into_iter().find(|candidate| {
            self.process()
                .bsd_sockets
                .get(candidate)
                .is_some_and(|socket| {
                    socket.kind == BSD_SOCK_DGRAM
                        && sockaddr_in(&socket.bound).is_some_and(|(bound, p)| {
                            p == port && (bound == ip || bound == BSD_ANY_IP)
                        })
                })
        });
        let Some(socket) = receiver.and_then(|fd| self.process_mut().bsd_sockets.get_mut(&fd))
        else {
            return (sent, 0);
        };
        let datagram = &mut socket.datagram;
        if datagram.admits(from_ip, from_port) && datagram.received.len() < MAX_QUEUED {
            datagram.received.push_back(Received {
                stream: None,
                source: sockaddr_in_bytes(from_ip, from_port),
                payload: data,
            });
        }
        (sent, 0)
    }

    /// `recv`/`recvfrom` on a datagram socket with one queued: one datagram, cut to `room`, and
    /// where it came from; `peek` leaves it queued. `None` when there is nothing to hand over.
    pub(super) fn bsd_receive_datagram(
        &mut self,
        fd: i32,
        room: usize,
        peek: bool,
    ) -> Option<Receipt> {
        let socket = self.process_mut().bsd_sockets.get_mut(&fd)?;
        let queue = &mut socket.datagram.received;
        let received = if peek {
            queue.front().cloned()
        } else {
            queue.pop_front()
        }?;
        // The whole datagram was read as far as the host's window goes, however much fitted.
        if let Some(stream) = received.stream.filter(|_| !peek) {
            self.net.outbox.push(NetCommand::Consumed {
                stream,
                bytes: received.payload.len() as u32,
            });
        }
        let truncated = received.payload.len() > room;
        let mut data = received.payload;
        data.truncate(room);
        Some(Receipt::read(data, Some(received.source), truncated))
    }

    /// What the host says about a datagram socket's stream `id`. `true` when the stream is one
    /// of its own.
    pub(super) fn bsd_datagram_event(&mut self, id: StreamId, event: &NetEvent) -> bool {
        // Before the socket is borrowed: an IPv6 source takes a synthetic address here.
        let address = match event {
            NetEvent::Datagram { source, .. } => self.net.address_for(source),
            _ => None,
        };
        let Some(socket) = self
            .machine
            .processes
            .values_mut()
            .flat_map(|process| process.bsd_sockets.values_mut())
            .find(|socket| socket.datagram.host == Some(id))
        else {
            return false;
        };
        let datagram = &mut socket.datagram;
        match event {
            NetEvent::Datagram { port, data, .. } => {
                let kept = address.filter(|&ip| {
                    datagram.admits(ip, *port)
                        && datagram.received.len() < MAX_QUEUED
                        && !socket.shut_read
                });
                match kept {
                    Some(ip) => datagram.received.push_back(Received {
                        stream: Some(id),
                        source: sockaddr_in_bytes(ip, *port),
                        payload: data.clone(),
                    }),
                    None => self.net.outbox.push(NetCommand::Consumed {
                        stream: id,
                        bytes: data.len() as u32,
                    }),
                }
            }
            // The host socket is gone; the next send opens another.
            NetEvent::Failed { .. } | NetEvent::Closed { .. } | NetEvent::Reset { .. } => {
                datagram.host = None;
            }
            NetEvent::Connected { .. }
            | NetEvent::Data { .. }
            | NetEvent::Sent { .. }
            | NetEvent::Incoming { .. }
            | NetEvent::TlsEstablished { .. }
            | NetEvent::TlsFailed { .. } => {}
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;
    use crate::kernel::ipc::testing::*;
    use crate::services::netbridge::Peer;

    const DATA: u64 = SCRATCH + 0x40;
    const ADDRESS: u64 = SCRATCH + 0x100;
    const INTO: u64 = SCRATCH + 0x180;
    const SOURCE: u64 = SCRATCH + 0x300;

    /// A datagram socket on a console that reaches the outside, and the address that stands for
    /// `name` there.
    fn udp_socket(name: &str) -> (Cpu, i32, [u8; 4]) {
        let (mut cpu, fd) = bsd_socket(2);
        cpu.mem.map_zero(SCRATCH, 0x800).unwrap();
        cpu.set_network(true);
        let address = cpu.net.synthesize(name).unwrap();
        (cpu, fd, address)
    }

    fn destination(ip: [u8; 4], port: u16) -> [u8; 16] {
        let mut raw = loopback_sockaddr(port);
        raw[4..8].copy_from_slice(&ip);
        raw
    }

    /// `sendto(fd, data, ip:port)`.
    fn sendto(cpu: &mut Cpu, fd: i32, data: &[u8], ip: [u8; 4], port: u16) -> (i32, i32) {
        place(cpu, DATA, data);
        place(cpu, ADDRESS, &destination(ip, port));
        let payload = [fd.to_le_bytes(), 0u32.to_le_bytes()].concat();
        let send = [(DATA, data.len() as u32), (ADDRESS, 16)];
        write_buffer_request(cpu, 11, &payload, &send, &[]);
        cpu.bsd_request(TLS, 9, Some(11)).unwrap();
        bsd_result(cpu)
    }

    /// `bind(fd, ip:port)`.
    fn bind(cpu: &mut Cpu, fd: i32, ip: [u8; 4], port: u16) -> (i32, i32) {
        place(cpu, ADDRESS, &destination(ip, port));
        write_map_buffer_request(cpu, 13, &fd.to_le_bytes(), ADDRESS, 16, true);
        cpu.bsd_request(TLS, 9, Some(13)).unwrap();
        bsd_result(cpu)
    }

    /// `recvfrom(fd, buffer of room bytes)`: the result, the bytes, and the source address.
    fn recvfrom(cpu: &mut Cpu, fd: i32, room: u32) -> ((i32, i32), Vec<u8>, Vec<u8>) {
        recvfrom_with(cpu, fd, room, 0)
    }

    fn recvfrom_with(
        cpu: &mut Cpu,
        fd: i32,
        room: u32,
        flags: u32,
    ) -> ((i32, i32), Vec<u8>, Vec<u8>) {
        place(cpu, INTO, &vec![0u8; room as usize]);
        place(cpu, SOURCE, &[0u8; 16]);
        let payload = [fd.to_le_bytes(), flags.to_le_bytes()].concat();
        write_buffer_request(cpu, 9, &payload, &[], &[(INTO, room), (SOURCE, 16)]);
        cpu.bsd_request(TLS, 9, Some(9)).unwrap();
        let result = bsd_result(cpu);
        let read = result.0.max(0) as u32;
        (
            result,
            cpu.read_bytes(INTO, read),
            cpu.read_bytes(SOURCE, 16),
        )
    }

    /// A datagram the host delivers on stream 1, from the host the guest named `name`.
    fn from_name(name: &str, port: u16, data: &[u8]) -> NetEvent {
        NetEvent::Datagram {
            stream: 1,
            source: Source::Name(name.into()),
            port,
            data: data.to_vec(),
        }
    }

    /// The port `fd` is bound to.
    fn bound_port(cpu: &Cpu, fd: i32) -> u16 {
        sockaddr_in(&cpu.process().bsd_sockets[&fd].bound)
            .unwrap()
            .1
    }

    #[test]
    fn one_host_socket_sends_to_every_address() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        let b = cpu.net.synthesize("b.example").unwrap();
        assert_eq!(sendto(&mut cpu, fd, b"one", a, 53), (3, 0));
        let port = bound_port(&cpu, fd);
        assert_eq!(
            cpu.take_net_commands(),
            vec![
                NetCommand::OpenUdp { stream: 1, port },
                NetCommand::SendTo {
                    stream: 1,
                    peer: Peer::Name("a.example".into()),
                    port: 53,
                    data: b"one".to_vec()
                },
            ]
        );
        assert_eq!(sendto(&mut cpu, fd, b"x", b, 123), (1, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::SendTo {
                stream: 1,
                peer: Peer::Name("b.example".into()),
                port: 123,
                data: b"x".to_vec()
            }]
        );
    }

    #[test]
    fn replies_come_back_one_datagram_at_a_time_from_where_the_guest_sent() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events([
            from_name("a.example", 53, b"first"),
            from_name("a.example", 53, b"second"),
        ]);

        let (result, bytes, source) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((5, 0), &b"first"[..]));
        assert_eq!(sockaddr_in(&source), Some((a, 53)));
        let (result, bytes, _) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((6, 0), &b"second"[..]));
        assert_eq!(
            cpu.take_net_commands(),
            vec![
                NetCommand::Consumed {
                    stream: 1,
                    bytes: 5
                },
                NetCommand::Consumed {
                    stream: 1,
                    bytes: 6
                },
            ],
            "the host is told what was read"
        );
        set_nonblocking(&mut cpu, fd);
        assert_eq!(
            recvfrom(&mut cpu, fd, 64).0,
            (-1, BSD_EAGAIN),
            "nothing queued"
        );
    }

    #[test]
    fn a_datagram_from_an_ipv6_host_comes_from_an_address_that_reaches_it_again() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        let mut v6 = [0u8; 16];
        v6[..4].copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8]);
        v6[15] = 7;
        cpu.deliver_net_events([NetEvent::Datagram {
            stream: 1,
            source: Source::V6(v6),
            port: 3478,
            data: b"hi".to_vec(),
        }]);
        let (_, _, source) = recvfrom(&mut cpu, fd, 64);
        let (ip, port) = sockaddr_in(&source).unwrap();
        assert_eq!((ip[..2].to_vec(), port), (vec![198, 18], 3478));
        cpu.take_net_commands();
        sendto(&mut cpu, fd, b"back", ip, port);
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::SendTo {
                stream: 1,
                peer: Peer::Name("2001:db8::7".into()),
                port: 3478,
                data: b"back".to_vec()
            }]
        );
    }

    #[test]
    fn a_datagram_longer_than_the_buffer_is_cut_and_the_rest_is_gone() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events([
            from_name("a.example", 53, b"abcdefgh"),
            from_name("a.example", 53, b"next"),
        ]);
        let (result, bytes, _) = recvfrom(&mut cpu, fd, 3);
        assert_eq!((result, bytes.as_slice()), ((3, 0), &b"abc"[..]));
        let (_, bytes, _) = recvfrom(&mut cpu, fd, 64);
        assert_eq!(bytes, b"next", "the next read starts at the next datagram");
        assert_eq!(
            cpu.take_net_commands()[0],
            NetCommand::Consumed {
                stream: 1,
                bytes: 8
            },
            "the whole datagram is credited"
        );
    }

    #[test]
    fn a_queued_datagram_makes_the_socket_readable() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        assert_eq!(cpu.bsd_poll_revents(fd, 1), 0);
        cpu.deliver_net_events([from_name("a.example", 53, &[1])]);
        assert_eq!(cpu.bsd_poll_revents(fd, 1), 1);
    }

    #[test]
    fn a_connected_socket_sends_to_its_peer_and_hears_only_from_it() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        place(&mut cpu, SCRATCH, &destination(a, 123));
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));

        assert_eq!(send_on(&mut cpu, fd, b"ping"), (4, 0));
        let commands = cpu.take_net_commands();
        assert!(matches!(commands[0], NetCommand::OpenUdp { stream: 1, .. }));
        assert_eq!(
            commands[1],
            NetCommand::SendTo {
                stream: 1,
                peer: Peer::Name("a.example".into()),
                port: 123,
                data: b"ping".to_vec()
            }
        );

        cpu.deliver_net_events([
            from_name("a.example", 124, b"other port"),
            from_name("b.example", 123, b"other host"),
            from_name("a.example", 123, b"pong"),
        ]);
        let (result, bytes, _) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((4, 0), &b"pong"[..]));
        let credited: u32 = cpu
            .take_net_commands()
            .iter()
            .map(|c| match c {
                NetCommand::Consumed { bytes, .. } => *bytes,
                _ => 0,
            })
            .sum();
        assert_eq!(credited, 10 + 10 + 4, "what was dropped is credited too");

        write_map_buffer_request(&mut cpu, 15, &fd.to_le_bytes(), SCRATCH + 0x200, 16, false);
        cpu.bsd_request(TLS, 9, Some(15)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "a connected socket has a peer");
        let raw = cpu.read_bytes(SCRATCH + 0x200, 16);
        assert_eq!(sockaddr_in(&raw), Some((a, 123)));
    }

    #[test]
    fn a_host_socket_that_went_away_is_reopened_by_the_next_send() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events([NetEvent::Failed {
            stream: 1,
            reason: NetFailure::Unreachable,
        }]);
        sendto(&mut cpu, fd, b"again", a, 53);
        assert!(matches!(
            cpu.take_net_commands()[0],
            NetCommand::OpenUdp { stream: 2, .. }
        ));
    }

    #[test]
    fn closing_the_socket_closes_its_host_socket() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        assert_eq!(bsd_command(&mut cpu, 26, &[fd as u32]), (0, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Close { stream: 1 }]
        );
    }

    #[test]
    fn datagrams_the_guest_does_not_read_are_dropped_past_a_limit_and_credited() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events((0..MAX_QUEUED + 3).map(|_| from_name("a.example", 53, &[0; 10])));
        let credited: u32 = cpu
            .take_net_commands()
            .iter()
            .map(|c| match c {
                NetCommand::Consumed { bytes, .. } => *bytes,
                _ => 0,
            })
            .sum();
        assert_eq!(credited, 30, "what was dropped went back to the host");
    }

    #[test]
    fn a_datagram_to_this_console_reaches_the_socket_bound_to_its_port() {
        let (mut cpu, receiver) = bsd_socket(2);
        cpu.mem.map_zero(SCRATCH, 0x800).unwrap();
        cpu.set_network(true);
        assert_eq!(bind(&mut cpu, receiver, [127, 0, 0, 1], 9000), (0, 0));
        let sender = open_socket(&mut cpu, 2);
        assert_eq!(
            sendto(&mut cpu, sender, b"hi", [127, 0, 0, 1], 9000),
            (2, 0)
        );
        assert!(
            cpu.take_net_commands().is_empty(),
            "nothing leaves the console"
        );
        let (result, bytes, source) = recvfrom(&mut cpu, receiver, 64);
        assert_eq!((result, bytes.as_slice()), ((2, 0), &b"hi"[..]));
        assert_eq!(
            sockaddr_in(&source),
            Some(([127, 0, 0, 1], bound_port(&cpu, sender)))
        );
        assert!(cpu.take_net_commands().is_empty(), "and nothing to credit");

        // To a port nobody holds, it is lost.
        assert_eq!(sendto(&mut cpu, sender, b"x", [127, 0, 0, 1], 9001), (1, 0));
        assert!(cpu.process().bsd_sockets[&receiver]
            .datagram
            .received
            .is_empty());
    }

    #[test]
    fn a_socket_hears_what_it_sends_to_its_own_address() {
        // RakNet's `BindShared` sends a test datagram to its own address.
        let (mut cpu, fd) = bsd_socket(2);
        cpu.mem.map_zero(SCRATCH, 0x800).unwrap();
        bind(&mut cpu, fd, [0, 0, 0, 0], 19132);
        assert_eq!(sendto(&mut cpu, fd, b"self", NIFM_LOCAL_IP, 19132), (4, 0));
        let (result, bytes, source) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((4, 0), &b"self"[..]));
        assert_eq!(sockaddr_in(&source), Some((NIFM_LOCAL_IP, 19132)));
    }

    #[test]
    fn binding_outside_loopback_opens_the_host_socket_at_once() {
        let (mut cpu, fd) = bsd_socket(2);
        cpu.mem.map_zero(SCRATCH, 0x800).unwrap();
        cpu.set_network(true);
        bind(&mut cpu, fd, [0, 0, 0, 0], 19132);
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::OpenUdp {
                stream: 1,
                port: 19132
            }]
        );
        let other = open_socket(&mut cpu, 2);
        bind(&mut cpu, other, [127, 0, 0, 1], 19133);
        assert!(
            cpu.take_net_commands().is_empty(),
            "loopback has nothing to hear from outside"
        );
    }

    #[test]
    fn multicast_and_broadcast_are_taken_and_dropped_without_a_host_socket() {
        let (mut cpu, fd, _) = udp_socket("a.example");
        for ip in [
            [224, 0, 0, 251],
            [239, 255, 255, 250],
            [255, 255, 255, 255],
            [240, 0, 0, 1],
        ] {
            assert_eq!(sendto(&mut cpu, fd, b"hi", ip, 5353), (2, 0), "{ip:?}");
        }
        assert!(cpu.take_net_commands().is_empty());
        // The last unicast address below the multicast range is still carried.
        assert_eq!(
            sendto(&mut cpu, fd, b"hi", [223, 255, 255, 255], 53),
            (2, 0)
        );
        assert_eq!(
            cpu.take_net_commands().len(),
            2,
            "a host socket and its datagram"
        );
    }

    #[test]
    fn a_datagram_larger_than_udp_carries_is_emsgsize() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        cpu.mem.map_zero(0x10_0000, 0x2_0000).unwrap();
        place(&mut cpu, ADDRESS, &destination(a, 53));
        let payload = [fd.to_le_bytes(), 0u32.to_le_bytes()].concat();
        let size = MAX_DATAGRAM as u32 + 1;
        write_buffer_request(
            &mut cpu,
            11,
            &payload,
            &[(0x10_0000, size), (ADDRESS, 16)],
            &[],
        );
        cpu.bsd_request(TLS, 9, Some(11)).unwrap();
        assert_eq!(bsd_result(&cpu), (-1, BSD_EMSGSIZE));
    }

    #[test]
    fn a_peek_leaves_the_datagram_and_its_credit_and_fionread_counts_them_all() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events([
            from_name("a.example", 53, b"first"),
            from_name("a.example", 53, b"second"),
        ]);
        assert_eq!(cpu.process().bsd_sockets[&fd].datagram.queued_bytes(), 11);

        let (result, bytes, source) = recvfrom_with(&mut cpu, fd, 64, BSD_MSG_PEEK);
        assert_eq!((result, bytes.as_slice()), ((5, 0), &b"first"[..]));
        assert_eq!(sockaddr_in(&source), Some((a, 53)));
        assert!(cpu.take_net_commands().is_empty(), "nothing was taken");

        let (result, bytes, _) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((5, 0), &b"first"[..]));
        assert_eq!(cpu.take_net_commands().len(), 1, "now it was");
    }

    #[test]
    fn a_blocking_recvfrom_waits_for_the_reply() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        recvfrom(&mut cpu, fd, 64);
        assert!(cpu.pending_retry.take().is_some(), "nothing yet");

        cpu.deliver_net_events([from_name("a.example", 53, b"answer")]);
        let (result, bytes, source) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((6, 0), &b"answer"[..]));
        assert_eq!(sockaddr_in(&source), Some((a, 53)));
        assert!(cpu.pending_retry.is_none());
    }

    #[test]
    fn with_the_network_off_a_datagram_is_still_taken_and_dropped() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        cpu.set_network(false);
        assert_eq!(sendto(&mut cpu, fd, b"hi", a, 53), (2, 0));
        assert!(cpu.take_net_commands().is_empty());
    }
}
