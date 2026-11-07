//! Stream sockets the host carries, and the events it answers with.

use super::*;
use crate::trace::Level;

/// A connection the host carries: where it stands and the peer's `sockaddr`.
#[derive(Debug, Clone)]
pub(crate) struct Stream {
    pub(super) id: StreamId,
    pub(super) state: StreamState,
    pub(super) address: Vec<u8>,
    /// A blocking `connect` is waiting on this stream: its next run reports how it ended.
    pub(super) deferred: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamState {
    Connecting,
    Open,
    /// The errno `SO_ERROR` reports once.
    Failed(i32),
}

impl Cpu {
    /// `connect` to a peer outside this console: the host does the connecting. A non-blocking
    /// socket gets `EINPROGRESS`, turns writable once it opens, or readable and writable with
    /// the error `SO_ERROR` reports if it fails. A blocking one waits: the call is run again
    /// until the host has answered, and that run reports the outcome.
    pub(super) fn bsd_connect_outside(
        &mut self,
        tls: u32,
        fd: i32,
        ip: [u8; 4],
        port: u16,
        address: &[u8],
    ) -> Result<()> {
        let (state, waiting, blocking) = match self.process().bsd_sockets.get(&fd) {
            Some(socket) => (
                socket.stream_state(),
                socket.stream.as_ref().is_some_and(|stream| stream.deferred),
                socket.flags & BSD_O_NONBLOCK == 0,
            ),
            None => return self.bsd_reply(tls, -1, BSD_EBADF),
        };
        match state {
            Some(StreamState::Connecting) if waiting => {
                self.retry_after(BSD_WAIT_SLICE_NS);
                return Ok(());
            }
            Some(StreamState::Connecting) => return self.bsd_reply(tls, -1, BSD_EALREADY),
            Some(StreamState::Open) if waiting => {
                self.bsd_settle_connect(fd);
                return self.bsd_reply(tls, 0, 0);
            }
            Some(StreamState::Open) => return self.bsd_reply(tls, -1, BSD_EISCONN),
            Some(StreamState::Failed(errno)) => {
                self.bsd_clear_stream(fd);
                return self.bsd_reply(tls, -1, errno);
            }
            None => {}
        }
        let id = self.net.open_stream();
        let peer = self.net.peer_for(ip);
        let flags = self.process().bsd_sockets.get(&fd).map_or(0, |s| s.flags);
        self.log(
            Level::Info,
            &format!("[net] connect fd {fd} to {peer:?}:{port} as stream {id} (flags {flags:#x})"),
        );
        self.net.outbox.push(NetCommand::Connect {
            stream: id,
            udp: false,
            peer,
            port,
        });
        let unbound = self
            .process()
            .bsd_sockets
            .get(&fd)
            .is_some_and(|socket| socket.bound.is_empty());
        let local = unbound.then(|| sockaddr_in_bytes(NIFM_LOCAL_IP, self.bsd_assign_port()));
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            if let Some(local) = local {
                socket.bound = local;
            }
            socket.stream = Some(Stream {
                id,
                state: StreamState::Connecting,
                address: address.to_vec(),
                deferred: blocking,
            });
        }
        if blocking {
            self.retry_after(BSD_WAIT_SLICE_NS);
            return Ok(());
        }
        self.bsd_reply(tls, -1, BSD_EINPROGRESS)
    }

    /// The blocking `connect` that was waiting on `fd` has its answer.
    fn bsd_settle_connect(&mut self, fd: i32) {
        if let Some(stream) = self
            .process_mut()
            .bsd_sockets
            .get_mut(&fd)
            .and_then(|socket| socket.stream.as_mut())
        {
            stream.deferred = false;
        }
    }

    /// `SO_ERROR`: the failure of a connect, reported once; zero otherwise.
    pub(super) fn bsd_take_stream_error(&mut self, fd: i32) -> u32 {
        let state = self
            .process()
            .bsd_sockets
            .get(&fd)
            .and_then(|socket| socket.stream_state());
        let Some(StreamState::Failed(errno)) = state else {
            return 0;
        };
        self.bsd_clear_stream(fd);
        errno as u32
    }

    /// Forget a stream that failed once its error has been reported.
    fn bsd_clear_stream(&mut self, fd: i32) {
        if let Some(socket) = self.process_mut().bsd_sockets.get_mut(&fd) {
            socket.stream = None;
        }
    }

    /// `Send` on a host-carried stream: queued for the host once it is open.
    pub(super) fn bsd_send_outside(&mut self, tls: u32, fd: i32) -> Result<()> {
        let Some(socket) = self.process().bsd_sockets.get(&fd) else {
            return self.bsd_reply(tls, -1, BSD_EBADF);
        };
        let (Some(StreamState::Open), Some(stream)) = (socket.stream_state(), &socket.stream)
        else {
            return self.bsd_reply(tls, -1, BSD_ENOTCONN);
        };
        let stream = stream.id;
        let data = match self.ipc_input_buffer(tls, 0) {
            Some((addr, size)) => self.read_bytes(addr, size),
            None => Vec::new(),
        };
        let sent = data.len() as i32;
        self.net.outbox.push(NetCommand::Send { stream, data });
        self.bsd_reply(tls, sent, 0)
    }

    /// Let guests reach the outside through the host, or back to an empty network.
    pub fn set_network(&mut self, enabled: bool) {
        self.net.enabled = enabled;
    }

    /// What the guest asked the host to do since the last call.
    pub fn take_net_commands(&mut self) -> Vec<NetCommand> {
        std::mem::take(&mut self.net.outbox)
    }

    /// Apply what the host reports about its streams; one for a socket since closed is dropped.
    pub fn deliver_net_events(&mut self, events: impl IntoIterator<Item = NetEvent>) {
        for event in events {
            let id = match &event {
                NetEvent::Connected { stream }
                | NetEvent::Failed { stream, .. }
                | NetEvent::Data { stream, .. }
                | NetEvent::Closed { stream }
                | NetEvent::TlsEstablished { stream, .. }
                | NetEvent::TlsFailed { stream, .. } => *stream,
            };
            if self.ssl_event(id, &event) || self.bsd_datagram_event(id, &event) {
                continue;
            }
            let socket = self
                .machine
                .processes
                .values_mut()
                .flat_map(|process| process.bsd_sockets.values_mut())
                .find(|socket| socket.stream.as_ref().is_some_and(|s| s.id == id));
            let Some(socket) = socket else {
                continue;
            };
            let Some(stream) = socket.stream.as_mut() else {
                continue;
            };
            match event {
                NetEvent::Connected { .. } => stream.state = StreamState::Open,
                NetEvent::Failed { reason, .. } => {
                    stream.state = StreamState::Failed(match reason {
                        NetFailure::Refused => BSD_ECONNREFUSED,
                        NetFailure::Unreachable => BSD_ENETUNREACH,
                        NetFailure::TimedOut => BSD_ETIMEDOUT,
                        NetFailure::NotFound => BSD_EHOSTUNREACH,
                    });
                }
                NetEvent::Data { data, .. } => socket.rx.extend(data),
                NetEvent::Closed { .. } => socket.peer_closed = true,
                NetEvent::TlsEstablished { .. } | NetEvent::TlsFailed { .. } => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;
    use crate::kernel::ipc::testing::*;
    use crate::services::netbridge::Peer;

    /// `connect(fd, ip:port)`.
    fn connect_to(cpu: &mut Cpu, fd: i32, ip: [u8; 4], port: u16) -> (i32, i32) {
        let mut address = loopback_sockaddr(port);
        address[4..8].copy_from_slice(&ip);
        place(cpu, SCRATCH, &address);
        write_map_buffer_request(cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
        bsd_result(cpu)
    }

    /// `getsockopt(fd, SOL_SOCKET, SO_ERROR)`.
    fn so_error(cpu: &mut Cpu, fd: i32) -> u32 {
        const AT: u32 = SCRATCH + 0x100;
        let mut payload = [0u8; 12];
        payload[..4].copy_from_slice(&fd.to_le_bytes());
        payload[4..8].copy_from_slice(&super::BSD_SOL_SOCKET.to_le_bytes());
        payload[8..].copy_from_slice(&super::BSD_SO_ERROR.to_le_bytes());
        write_map_buffer_request(cpu, 17, &payload, AT, 4, false);
        cpu.bsd_request(TLS, 9, Some(17)).unwrap();
        cpu.mem.read_u32(AT).unwrap()
    }

    /// A socket on a console whose guest may reach the outside, and the synthetic address
    /// `sfdnsres` would have given `name`.
    fn networked_socket(name: &str) -> (Cpu, i32, [u8; 4]) {
        let (mut cpu, fd, address) = blocking_socket(name);
        cpu.process_mut().bsd_sockets.get_mut(&fd).unwrap().flags |= BSD_O_NONBLOCK;
        (cpu, fd, address)
    }

    /// The same, left blocking, as a socket is until the guest says otherwise.
    fn blocking_socket(name: &str) -> (Cpu, i32, [u8; 4]) {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x400).unwrap();
        cpu.set_network(true);
        let address = cpu.net.synthesize(name).unwrap();
        (cpu, fd, address)
    }

    /// Run the `connect` already in the buffer again, as the kernel does after a wait.
    fn rerun_connect(cpu: &mut Cpu) {
        cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    }

    const POLLIN: u16 = 1;

    const POLLOUT: u16 = 4;

    #[test]
    fn bsd_connect_to_a_name_is_in_progress_until_the_host_opens_it() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EINPROGRESS)
        );
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Connect {
                stream: 1,
                udp: false,
                peer: Peer::Name("example.com".into()),
                port: 443,
            }]
        );
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EALREADY)
        );
        assert_eq!(cpu.bsd_poll_revents(fd, POLLOUT), 0, "not writable yet");
        assert_eq!(send_on(&mut cpu, fd, b"early"), (-1, super::BSD_ENOTCONN));

        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        assert_eq!(cpu.bsd_poll_revents(fd, POLLOUT), POLLOUT);
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EISCONN)
        );
        assert_eq!(so_error(&mut cpu, fd), 0);
    }

    #[test]
    fn bsd_carries_bytes_both_ways_and_ends_when_the_host_does() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        connect_to(&mut cpu, fd, address, 80);
        cpu.take_net_commands();
        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);

        assert_eq!(send_on(&mut cpu, fd, b"GET /"), (5, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Send {
                stream: 1,
                data: b"GET /".to_vec()
            }]
        );

        assert_eq!(recv_on(&mut cpu, fd, 16).0, (-1, super::BSD_EAGAIN));
        assert_eq!(cpu.bsd_poll_revents(fd, POLLIN), 0);
        cpu.deliver_net_events([NetEvent::Data {
            stream: 1,
            data: b"200 OK".to_vec(),
        }]);
        assert_eq!(cpu.bsd_poll_revents(fd, POLLIN), POLLIN);
        assert_eq!(recv_on(&mut cpu, fd, 16), ((6, 0), b"200 OK".to_vec()));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Consumed {
                stream: 1,
                bytes: 6
            }],
            "what the guest read is handed back to the host"
        );

        cpu.deliver_net_events([NetEvent::Closed { stream: 1 }]);
        assert_eq!(recv_on(&mut cpu, fd, 16).0, (0, 0), "end of file");
        assert_eq!(send_on(&mut cpu, fd, b"more"), (-1, super::BSD_EPIPE));
    }

    #[test]
    fn bsd_reports_a_failed_connect_once_through_so_error() {
        let (mut cpu, fd, address) = networked_socket("nowhere.example");
        connect_to(&mut cpu, fd, address, 443);
        cpu.deliver_net_events([NetEvent::Failed {
            stream: 1,
            reason: NetFailure::Refused,
        }]);
        assert_eq!(
            cpu.bsd_poll_revents(fd, POLLIN | POLLOUT),
            POLLIN | POLLOUT | 0x0008
        );
        assert_eq!(so_error(&mut cpu, fd), super::BSD_ECONNREFUSED as u32);
        assert_eq!(so_error(&mut cpu, fd), 0, "reported once");

        // A second connect starts a new stream.
        cpu.take_net_commands();
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_EINPROGRESS)
        );
        assert_eq!(cpu.take_net_commands().len(), 1);
    }

    #[test]
    fn bsd_connect_after_a_failure_returns_the_error_itself() {
        let (mut cpu, fd, address) = networked_socket("nowhere.example");
        connect_to(&mut cpu, fd, address, 443);
        cpu.deliver_net_events([NetEvent::Failed {
            stream: 1,
            reason: NetFailure::TimedOut,
        }]);
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_ETIMEDOUT)
        );
    }

    #[test]
    fn bsd_tells_the_host_when_the_guest_stops_sending_or_closes() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        connect_to(&mut cpu, fd, address, 80);
        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        cpu.take_net_commands();

        assert_eq!(bsd_command(&mut cpu, 22, &[fd as u32, 1]), (0, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::ShutdownWrite { stream: 1 }]
        );
        assert_eq!(bsd_command(&mut cpu, 26, &[fd as u32]), (0, 0));
        assert_eq!(
            cpu.take_net_commands(),
            vec![NetCommand::Close { stream: 1 }]
        );

        // Events for the closed socket go nowhere.
        cpu.deliver_net_events([NetEvent::Data {
            stream: 1,
            data: vec![1],
        }]);
    }

    #[test]
    fn bsd_peer_name_of_a_stream_is_what_the_guest_connected_to() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        connect_to(&mut cpu, fd, address, 443);
        assert_eq!(
            bsd_command(&mut cpu, 15, &[fd as u32]),
            (-1, super::BSD_ENOTCONN)
        );
        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        write_map_buffer_request(&mut cpu, 15, &fd.to_le_bytes(), SCRATCH + 0x200, 16, false);
        cpu.bsd_request(TLS, 9, Some(15)).unwrap();
        assert_eq!(bsd_result(&cpu), (0, 0));
        let raw = cpu.read_bytes(SCRATCH + 0x200, 16);
        assert_eq!(super::sockaddr_in(&raw), Some((address, 443)));
    }

    #[test]
    fn bsd_stays_an_empty_network_until_enabled() {
        let (mut cpu, fd) = bsd_socket(1);
        cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
        let address = cpu.net.synthesize("example.com").unwrap();
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, super::BSD_ECONNREFUSED)
        );
        assert!(cpu.take_net_commands().is_empty());
    }

    #[test]
    fn a_blocking_connect_gives_no_reply_until_the_host_has_answered() {
        let (mut cpu, fd, address) = blocking_socket("example.com");
        connect_to(&mut cpu, fd, address, 443);
        assert!(cpu.pending_retry.take().is_some(), "the call waits");
        assert_eq!(cpu.take_net_commands().len(), 1, "one connect asked for");

        // Still waiting: the request was left alone, so running it again is the same call.
        rerun_connect(&mut cpu);
        assert!(cpu.pending_retry.take().is_some());
        assert!(
            cpu.take_net_commands().is_empty(),
            "and it does not ask again"
        );

        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        rerun_connect(&mut cpu);
        assert!(cpu.pending_retry.is_none());
        assert_eq!(bsd_result(&cpu), (0, 0), "the connect succeeded");
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, BSD_EISCONN),
            "a second connect is a second connect"
        );
    }

    #[test]
    fn a_blocking_connect_that_fails_returns_the_error() {
        let (mut cpu, fd, address) = blocking_socket("example.com");
        connect_to(&mut cpu, fd, address, 443);
        cpu.pending_retry = None;
        cpu.deliver_net_events([NetEvent::Failed {
            stream: 1,
            reason: NetFailure::Refused,
        }]);
        rerun_connect(&mut cpu);
        assert!(cpu.pending_retry.is_none());
        assert_eq!(bsd_result(&cpu), (-1, BSD_ECONNREFUSED));
        assert_eq!(
            so_error(&mut cpu, fd),
            0,
            "the error was reported by the connect"
        );
    }

    #[test]
    fn a_non_blocking_connect_never_waits() {
        let (mut cpu, fd, address) = networked_socket("example.com");
        assert_eq!(
            connect_to(&mut cpu, fd, address, 443),
            (-1, BSD_EINPROGRESS)
        );
        assert!(cpu.pending_retry.is_none());
    }

    /// Through the kernel: an unanswered call rewinds onto its `svc` and X0 is left alone, so
    /// the guest never sees a result until the call has one.
    #[test]
    fn the_kernel_runs_a_waiting_connect_again_and_answers_it_once() {
        const AFTER_SVC: u32 = 0x0800_0004;
        let (mut cpu, fd, address) = blocking_socket("example.com");
        cpu.tpidr = u64::from(TLS);
        let mut remote = loopback_sockaddr(443);
        remote[4..8].copy_from_slice(&address);
        place(&mut cpu, SCRATCH, &remote);
        write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
        let issue = |cpu: &mut Cpu| {
            cpu.pc = AFTER_SVC;
            cpu.write_zr(0, 9);
            cpu.horizon_syscall(0x21).unwrap();
        };

        issue(&mut cpu);
        assert_eq!(cpu.pc, AFTER_SVC - 4, "back on the svc");
        assert_eq!(cpu.read_zr(0), 9, "no result yet");

        cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
        issue(&mut cpu);
        assert_eq!(cpu.pc, AFTER_SVC, "the svc completed");
        assert_eq!(cpu.read_zr(0), 0, "RESULT_OK");
        assert_eq!(bsd_result(&cpu), (0, 0));
    }
}
