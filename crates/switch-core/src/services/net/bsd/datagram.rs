//! Datagram sockets the host carries: one relay stream per remote address, opened by the
//! first `sendto` to it, and what comes back queued with its source and its boundaries.

use super::*;
use crate::trace::Level;

/// Most remote addresses one socket may talk to; a guest scanning ports stops here.
const MAX_PEERINGS: usize = 32;

/// Most datagrams held for a guest that is not reading. Past it they are dropped, as a full
/// socket buffer does, and their credit goes back to the host.
const MAX_QUEUED: usize = 256;

/// One remote address a socket has sent to, and the host stream that carries it.
#[derive(Debug, Clone)]
pub(crate) struct Peering {
    pub(super) id: StreamId,
    ip: [u8; 4],
    port: u16,
}

impl Peering {
    fn address(&self) -> Vec<u8> {
        sockaddr_in_bytes(self.ip, self.port)
    }
}

/// A datagram received, with the stream it came on and where it came from.
#[derive(Debug, Clone)]
pub(super) struct Received {
    stream: StreamId,
    source: Vec<u8>,
    payload: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Datagram {
    pub(super) peerings: Vec<Peering>,
    pub(super) received: VecDeque<Received>,
    /// Where `send` goes after a `connect`.
    pub(super) default_peer: Option<([u8; 4], u16)>,
}

impl Datagram {
    /// The address a connected socket reports as its peer's.
    pub(super) fn default_address(&self) -> Option<Vec<u8>> {
        self.default_peer
            .map(|(ip, port)| sockaddr_in_bytes(ip, port))
    }
}

impl Cpu {
    /// `connect` on a datagram socket: it only names where `send` goes. Outside the console
    /// that is the host's to carry; anything else is refused as it always was.
    pub(super) fn bsd_connect_datagram(&mut self, tls: u32, fd: i32, address: &[u8]) -> Result<()> {
        let Some((ip, port)) = sockaddr_in(address)
            .filter(|&(ip, port)| self.net.enabled && is_carried(ip) && port != 0)
        else {
            return self.bsd_reply(tls, -1, BSD_ECONNREFUSED);
        };
        if self.bsd_peering(fd, ip, port).is_none() {
            return self.bsd_reply(tls, -1, BSD_ENOBUFS);
        }
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            socket.datagram.default_peer = Some((ip, port));
        }
        self.bsd_reply(tls, 0, 0)
    }

    /// `sendto` (or `send` after a `connect`) to an address outside the console.
    pub(super) fn bsd_send_outside_datagram(
        &mut self,
        tls: u32,
        fd: i32,
        ip: [u8; 4],
        port: u16,
    ) -> Result<()> {
        let data = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let Some(stream) = self.bsd_peering(fd, ip, port) else {
            return self.bsd_reply(tls, -1, BSD_ENOBUFS);
        };
        let sent = data.len() as i32;
        self.net.outbox.push(NetCommand::Send { stream, data });
        self.bsd_reply(tls, sent, 0)
    }

    /// The host stream for `ip:port`, opened if this socket has not sent there; `None` when the
    /// socket already talks to as many addresses as it may.
    fn bsd_peering(&mut self, fd: i32, ip: [u8; 4], port: u16) -> Option<StreamId> {
        let socket = self.process().bsd_sockets.get(&fd)?;
        if let Some(peering) = socket
            .datagram
            .peerings
            .iter()
            .find(|peering| (peering.ip, peering.port) == (ip, port))
        {
            return Some(peering.id);
        }
        if socket.datagram.peerings.len() >= MAX_PEERINGS {
            return None;
        }
        let unbound = socket.bound.is_empty();
        let id = self.net.open_stream();
        let peer = self.net.peer_for(ip);
        self.log(
            Level::Info,
            &format!("[net] udp fd {fd} to {peer:?}:{port} as stream {id}"),
        );
        self.net.outbox.push(NetCommand::Connect {
            stream: id,
            udp: true,
            peer,
            port,
        });
        let local = unbound.then(|| sockaddr_in_bytes(NIFM_LOCAL_IP, self.bsd_assign_port()));
        let socket = self.process_mut().bsd_sockets.get_mut(&fd)?;
        if let Some(local) = local {
            socket.bound = local;
        }
        socket.datagram.peerings.push(Peering { id, ip, port });
        Some(id)
    }

    /// `recv`/`recvfrom` on a datagram socket with one queued: one datagram, cut to the
    /// caller's buffer, and where it came from. `None` when there is nothing to hand over.
    pub(super) fn bsd_receive_datagram(&mut self, tls: u32, fd: i32) -> Result<Option<Recv>> {
        let has = self
            .process()
            .bsd_sockets
            .get(&fd)
            .is_some_and(|socket| !socket.datagram.received.is_empty());
        if !has {
            return Ok(None);
        }
        let Some((addr, size)) = self.ipc_output_buffer(tls, 0) else {
            return Ok(Some((-1, BSD_EINVAL, None)));
        };
        let Some(received) = self
            .process_mut()
            .bsd_sockets
            .get_mut(&fd)
            .and_then(|socket| socket.datagram.received.pop_front())
        else {
            return Ok(None);
        };
        let take = (size as usize).min(received.payload.len());
        for (offset, &byte) in received.payload[..take].iter().enumerate() {
            self.mem.write_u8(addr.wrapping_add(offset as u32), byte)?;
        }
        // The whole datagram was read as far as the host's window goes, however much fitted.
        self.net.outbox.push(NetCommand::Consumed {
            stream: received.stream,
            bytes: received.payload.len() as u32,
        });
        Ok(Some((take as i32, 0, Some(received.source))))
    }

    /// What the host says about a datagram socket's stream `id`. `true` when the stream is one
    /// of its own.
    pub(super) fn bsd_datagram_event(&mut self, id: StreamId, event: &NetEvent) -> bool {
        let Some(socket) = self
            .machine
            .processes
            .values_mut()
            .flat_map(|process| process.bsd_sockets.values_mut())
            .find(|socket| socket.datagram.peerings.iter().any(|p| p.id == id))
        else {
            return false;
        };
        let Some(index) = socket.datagram.peerings.iter().position(|p| p.id == id) else {
            return false;
        };
        match event {
            NetEvent::Data { data, .. } if socket.datagram.received.len() < MAX_QUEUED => {
                let source = socket.datagram.peerings[index].address();
                socket.datagram.received.push_back(Received {
                    stream: id,
                    source,
                    payload: data.clone(),
                });
            }
            NetEvent::Data { data, .. } => {
                self.net.outbox.push(NetCommand::Consumed {
                    stream: id,
                    bytes: data.len() as u32,
                });
            }
            // Nothing says a datagram was lost, so the stream is simply forgotten and the next
            // send to that address opens another.
            NetEvent::Failed { .. } | NetEvent::Closed { .. } => {
                socket.datagram.peerings.remove(index);
            }
            NetEvent::Connected { .. }
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

    const DATA: u32 = SCRATCH + 0x40;
    const ADDRESS: u32 = SCRATCH + 0x100;
    const INTO: u32 = SCRATCH + 0x180;
    const SOURCE: u32 = SCRATCH + 0x300;

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

    /// `recvfrom(fd, buffer of room bytes)`: the result, the bytes, and the source address.
    fn recvfrom(cpu: &mut Cpu, fd: i32, room: u32) -> ((i32, i32), Vec<u8>, Vec<u8>) {
        place(cpu, INTO, &vec![0u8; room as usize]);
        place(cpu, SOURCE, &[0u8; 16]);
        let payload = [fd.to_le_bytes(), 0u32.to_le_bytes()].concat();
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

    fn connect_commands(cpu: &mut Cpu) -> Vec<NetCommand> {
        cpu.take_net_commands()
    }

    #[test]
    fn sendto_goes_to_the_host_over_one_udp_stream_per_address() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        let b = cpu.net.synthesize("b.example").unwrap();
        assert_eq!(sendto(&mut cpu, fd, b"one", a, 53), (3, 0));
        assert_eq!(
            connect_commands(&mut cpu),
            vec![
                NetCommand::Connect {
                    stream: 1,
                    udp: true,
                    peer: Peer::Name("a.example".into()),
                    port: 53,
                },
                NetCommand::Send {
                    stream: 1,
                    data: b"one".to_vec()
                },
            ]
        );

        // The same address again reuses the stream; another opens its own.
        assert_eq!(sendto(&mut cpu, fd, b"two", a, 53), (3, 0));
        assert_eq!(sendto(&mut cpu, fd, b"x", b, 123), (1, 0));
        let commands = connect_commands(&mut cpu);
        assert_eq!(commands.len(), 3);
        assert!(matches!(commands[0], NetCommand::Send { stream: 1, .. }));
        assert!(matches!(
            commands[1],
            NetCommand::Connect {
                stream: 2,
                udp: true,
                port: 123,
                ..
            }
        ));
        assert!(matches!(commands[2], NetCommand::Send { stream: 2, .. }));
    }

    #[test]
    fn replies_come_back_one_datagram_at_a_time_with_their_source() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events([
            NetEvent::Data {
                stream: 1,
                data: b"first".to_vec(),
            },
            NetEvent::Data {
                stream: 1,
                data: b"second".to_vec(),
            },
        ]);

        let (result, bytes, source) = recvfrom(&mut cpu, fd, 64);
        assert_eq!((result, bytes.as_slice()), ((5, 0), &b"first"[..]));
        assert_eq!(super::super::sockaddr_in(&source), Some((a, 53)));
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
        assert_eq!(
            recvfrom(&mut cpu, fd, 64).0,
            (-1, BSD_EAGAIN),
            "nothing queued"
        );
    }

    #[test]
    fn a_datagram_longer_than_the_buffer_is_cut_and_the_rest_is_gone() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events([
            NetEvent::Data {
                stream: 1,
                data: b"abcdefgh".to_vec(),
            },
            NetEvent::Data {
                stream: 1,
                data: b"next".to_vec(),
            },
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
        cpu.deliver_net_events([NetEvent::Data {
            stream: 1,
            data: vec![1],
        }]);
        assert_eq!(cpu.bsd_poll_revents(fd, 1), 1);
    }

    #[test]
    fn a_connected_datagram_socket_sends_to_the_address_it_named() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        place(&mut cpu, SCRATCH, &destination(a, 123));
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));

        assert_eq!(send_on(&mut cpu, fd, b"ping"), (4, 0));
        let commands = cpu.take_net_commands();
        assert!(matches!(
            commands[0],
            NetCommand::Connect {
                stream: 1,
                udp: true,
                port: 123,
                ..
            }
        ));
        assert_eq!(
            commands[1],
            NetCommand::Send {
                stream: 1,
                data: b"ping".to_vec()
            }
        );

        write_map_buffer_request(&mut cpu, 15, &fd.to_le_bytes(), SCRATCH + 0x200, 16, false);
        cpu.bsd_request(TLS, 9, Some(15)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0), "a connected socket has a peer");
        let raw = cpu.read_bytes(SCRATCH + 0x200, 16);
        assert_eq!(super::super::sockaddr_in(&raw), Some((a, 123)));
    }

    #[test]
    fn a_failed_stream_is_forgotten_and_the_next_send_opens_another() {
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
            NetCommand::Connect {
                stream: 2,
                udp: true,
                ..
            }
        ));
    }

    #[test]
    fn closing_the_socket_closes_each_stream_it_opened() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        let b = cpu.net.synthesize("b.example").unwrap();
        sendto(&mut cpu, fd, b"q", a, 53);
        sendto(&mut cpu, fd, b"q", b, 53);
        cpu.take_net_commands();
        assert_eq!(bsd_command(&mut cpu, 26, &[fd as u32]), (0, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![
                NetCommand::Close { stream: 1 },
                NetCommand::Close { stream: 2 }
            ]
        );
    }

    #[test]
    fn datagrams_the_guest_does_not_read_are_dropped_past_a_limit_and_credited() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        sendto(&mut cpu, fd, b"q", a, 53);
        cpu.take_net_commands();
        cpu.deliver_net_events((0..MAX_QUEUED + 3).map(|_| NetEvent::Data {
            stream: 1,
            data: vec![0; 10],
        }));
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
    fn an_address_on_this_console_is_still_taken_and_dropped_by_the_link() {
        let (mut cpu, fd, _) = udp_socket("a.example");
        assert_eq!(sendto(&mut cpu, fd, b"hi", [127, 0, 0, 1], 9), (2, 0));
        assert!(cpu.take_net_commands().is_empty());
    }

    #[test]
    fn multicast_and_broadcast_are_taken_and_dropped_without_a_host_stream() {
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
            "a stream and its datagram"
        );
    }

    #[test]
    fn with_the_network_off_a_datagram_is_still_taken_and_dropped() {
        let (mut cpu, fd, a) = udp_socket("a.example");
        cpu.set_network(false);
        assert_eq!(sendto(&mut cpu, fd, b"hi", a, 53), (2, 0));
        assert!(cpu.take_net_commands().is_empty());
    }
}
