use crate::cpu::Cpu;
use crate::services::netbridge::{NetCommand, NetEvent, NetFailure, Source};

use crate::kernel::ipc::testing::*;

/// Read a `bsd` command's `{ s32 ret, s32 errno }` reply.
pub(in crate::services::net) fn bsd_result(cpu: &Cpu) -> (i32, i32) {
    (
        cpu.mem.read_u32(TLS + 0x20).unwrap() as i32,
        cpu.mem.read_u32(TLS + 0x24).unwrap() as i32,
    )
}

/// The third reply word: bytes of the output buffer filled.
pub(in crate::services::net) fn bsd_result_len(cpu: &Cpu) -> u32 {
    cpu.mem.read_u32(TLS + 0x28).unwrap()
}

/// Open a socket of `kind` on a fresh `bsd:u` session.
pub(in crate::services::net) fn bsd_socket(kind: u32) -> (Cpu, i32) {
    let mut cpu = request(false, 2, &[]);
    cpu.register_service_handle(9, "bsd:u");
    let fd = open_socket(&mut cpu, kind);
    (cpu, fd)
}

pub(in crate::services::net) fn open_socket(cpu: &mut Cpu, kind: u32) -> i32 {
    let mut payload = [0u8; 12];
    payload[..4].copy_from_slice(&2u32.to_le_bytes()); // AF_INET
    payload[4..8].copy_from_slice(&kind.to_le_bytes());
    write_request(cpu, 2, &payload);
    cpu.bsd_request(TLS, 9, Some(2)).unwrap();
    let (fd, errno) = bsd_result(cpu);
    assert_eq!(errno, 0, "socket");
    fd
}

pub(in crate::services::net) const SCRATCH: u32 = 0x4000;

/// What `fcntl(F_SETFL, O_NONBLOCK)` leaves behind.
pub(in crate::services::net) fn set_nonblocking(cpu: &mut Cpu, fd: i32) {
    cpu.process_mut().bsd_sockets.get_mut(&fd).unwrap().flags |= super::BSD_O_NONBLOCK;
}

/// `127.0.0.1:port` with `sin_len` left at zero, as a memsetting guest writes it.
pub(in crate::services::net) fn loopback_sockaddr(port: u16) -> [u8; 16] {
    let mut raw = [0u8; 16];
    raw[1] = 2; // AF_INET
    raw[2..4].copy_from_slice(&port.to_be_bytes());
    raw[4..8].copy_from_slice(&[127, 0, 0, 1]);
    raw
}

pub(in crate::services::net) fn place(cpu: &mut Cpu, at: u32, bytes: &[u8]) {
    for (offset, &byte) in bytes.iter().enumerate() {
        cpu.mem.write_u8(at + offset as u32, byte).unwrap();
    }
}

/// asio's `socket_select_interrupter` setup: bind to an ephemeral loopback port,
/// connect to the port `getsockname` reports, accept.
pub(in crate::services::net) fn connected_pair(cpu: &mut Cpu, listener: i32) -> (i32, i32) {
    place(cpu, SCRATCH, &loopback_sockaddr(0));
    write_map_buffer_request(cpu, 13, &listener.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(13)).unwrap();
    assert_eq!(bsd_result(cpu), (0, 0), "bind");

    place(cpu, SCRATCH, &[0u8; 16]);
    write_map_buffer_request(cpu, 16, &listener.to_le_bytes(), SCRATCH, 16, false);
    cpu.bsd_request(TLS, 9, Some(16)).unwrap();
    assert_eq!(bsd_result(cpu), (0, 0), "getsockname");

    write_request(cpu, 18, &listener.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(18)).unwrap();
    assert_eq!(bsd_result(cpu), (0, 0), "listen");

    let client = open_socket(cpu, 1);
    write_map_buffer_request(cpu, 14, &client.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(bsd_result(cpu), (0, 0), "connect");

    write_request(cpu, 12, &listener.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(12)).unwrap();
    let (server, errno) = bsd_result(cpu);
    assert_eq!(errno, 0, "accept");
    (client, server)
}

pub(in crate::services::net) fn send_on(cpu: &mut Cpu, fd: i32, bytes: &[u8]) -> (i32, i32) {
    const AT: u32 = SCRATCH + 0x40;
    place(cpu, AT, bytes);
    write_map_buffer_request(cpu, 10, &fd.to_le_bytes(), AT, bytes.len() as u32, true);
    cpu.bsd_request(TLS, 9, Some(10)).unwrap();
    bsd_result(cpu)
}

pub(in crate::services::net) fn recv_on(cpu: &mut Cpu, fd: i32, len: u32) -> ((i32, i32), Vec<u8>) {
    recv_with(cpu, fd, len, 0)
}

/// `recv(fd, buffer of len bytes, flags)`.
pub(in crate::services::net) fn recv_with(
    cpu: &mut Cpu,
    fd: i32,
    len: u32,
    flags: u32,
) -> ((i32, i32), Vec<u8>) {
    const AT: u32 = SCRATCH + 0x80;
    place(cpu, AT, &vec![0u8; len as usize]);
    let payload = [fd.to_le_bytes(), flags.to_le_bytes()].concat();
    write_map_buffer_request(cpu, 8, &payload, AT, len, false);
    cpu.bsd_request(TLS, 9, Some(8)).unwrap();
    let result = bsd_result(cpu);
    let read = if result.0 > 0 {
        cpu.read_bytes(AT, result.0 as u32)
    } else {
        Vec::new()
    };
    (result, read)
}

#[test]
pub(in crate::services::net) fn bsd_hands_out_descriptors_and_takes_them_back() {
    let (mut cpu, fd) = bsd_socket(1);
    assert!(
        fd >= 3,
        "past the standard streams a C library already holds"
    );

    write_request(&mut cpu, 26, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(26)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "close");

    write_request(&mut cpu, 26, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(26)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_EBADF));
}

#[test]
pub(in crate::services::net) fn bsd_fails_where_there_is_no_peer_rather_than_pretending() {
    let (mut cpu, fd) = bsd_socket(1);
    write_request(&mut cpu, 14, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_ECONNREFUSED));

    // A stream socket has no connection to send on...
    write_request(&mut cpu, 10, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(10)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_ENOTCONN));

    // ...and a datagram socket that named no destination has nowhere to send to.
    let (mut cpu, fd) = bsd_socket(super::BSD_SOCK_DGRAM);
    write_request(&mut cpu, 11, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(11)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_ENETUNREACH));

    // Accept before listen is an error; after listen, with nothing to accept, it is EAGAIN.
    let (mut cpu, fd) = bsd_socket(1);
    set_nonblocking(&mut cpu, fd);
    write_request(&mut cpu, 12, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(12)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_EINVAL));

    write_request(&mut cpu, 18, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(18)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "listen");
    write_request(&mut cpu, 12, &fd.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(12)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_EAGAIN));
}

#[test]
pub(in crate::services::net) fn bsd_sends_an_addressed_datagram_the_link_would_carry() {
    // RakNet's `BindShared` sends a test datagram to its own address and fails startup
    // if it does not go out.
    let (mut cpu, fd) = bsd_socket(super::BSD_SOCK_DGRAM);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    const PAYLOAD: u32 = SCRATCH + 0x40;
    const PORT: u16 = 19132;

    place(&mut cpu, SCRATCH, &loopback_sockaddr(PORT));
    write_map_buffer_request(&mut cpu, 13, &fd.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(13)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "bind");

    let send_to = |cpu: &mut Cpu, fd: i32, address: &[u8; 16]| {
        place(cpu, PAYLOAD, &[1u8, 2, 3, 4]);
        place(cpu, SCRATCH, address);
        write_buffer_request(
            cpu,
            11,
            &fd.to_le_bytes(),
            &[(PAYLOAD, 4), (SCRATCH, 16)],
            &[],
        );
        cpu.bsd_request(TLS, 9, Some(11)).unwrap();
        bsd_result(cpu)
    };

    assert_eq!(
        send_to(&mut cpu, fd, &loopback_sockaddr(PORT)),
        (4, 0),
        "the address it just bound"
    );

    let mut broadcast = loopback_sockaddr(PORT);
    broadcast[4..8].copy_from_slice(&[255, 255, 255, 255]);
    assert_eq!(send_to(&mut cpu, fd, &broadcast), (4, 0), "broadcast");

    // The datagram to itself arrives, as on a console; then nothing does: EAGAIN and a yield,
    // not `ENETUNREACH`.
    assert_eq!(recv_on(&mut cpu, fd, 4), ((4, 0), vec![1, 2, 3, 4]));
    set_nonblocking(&mut cpu, fd);
    cpu.pending_yield = false;
    let (result, bytes) = recv_on(&mut cpu, fd, 4);
    assert_eq!(result, (-1, super::BSD_EAGAIN), "nothing yet");
    assert!(bytes.is_empty());
    assert!(cpu.pending_yield, "a read that waits has to reschedule");

    // Datagram sockets poll writable.
    const POLL_IN: u32 = SCRATCH + 0xc0;
    const POLL_OUT: u32 = SCRATCH + 0xd0;
    const POLLOUT: u16 = 0x0004;
    let mut pollfd = [0u8; 8];
    pollfd[..4].copy_from_slice(&fd.to_le_bytes());
    pollfd[4..6].copy_from_slice(&POLLOUT.to_le_bytes());
    place(&mut cpu, POLL_IN, &pollfd);
    place(&mut cpu, POLL_OUT, &[0u8; 8]);
    let mut poll = [0u8; 8];
    poll[..4].copy_from_slice(&1u32.to_le_bytes());
    write_buffer_request(&mut cpu, 6, &poll, &[(POLL_IN, 8)], &[(POLL_OUT, 8)]);
    cpu.bsd_request(TLS, 9, Some(6)).unwrap();
    assert_eq!(bsd_result(&cpu), (1, 0), "ready to send");
    assert_eq!(cpu.mem.read_u16(POLL_OUT + 6).unwrap(), POLLOUT);

    // A destination is not a connection for a stream socket.
    let (mut cpu, stream) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    assert_eq!(
        send_to(&mut cpu, stream, &loopback_sockaddr(PORT)),
        (-1, super::BSD_ENOTCONN)
    );
}

#[test]
pub(in crate::services::net) fn bsd_socket_options_and_flags_read_back() {
    const BUFFER: u32 = 0x4000;
    let (mut cpu, fd) = bsd_socket(1);
    cpu.mem.map_zero(BUFFER, 0x100).unwrap();

    let mut payload = [0u8; 12];
    payload[..4].copy_from_slice(&fd.to_le_bytes());
    payload[4..8].copy_from_slice(&0xFFFFu32.to_le_bytes()); // SOL_SOCKET
    payload[8..].copy_from_slice(&0x0004u32.to_le_bytes()); // SO_REUSEADDR
    write_map_buffer_request(&mut cpu, 21, &payload, BUFFER, 4, true);
    cpu.mem.write_u32(BUFFER, 1).unwrap();
    cpu.bsd_request(TLS, 9, Some(21)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));

    cpu.mem.write_u32(BUFFER, 0).unwrap();
    write_map_buffer_request(&mut cpu, 17, &payload, BUFFER, 4, false);
    cpu.bsd_request(TLS, 9, Some(17)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));
    assert_eq!(cpu.mem.read_u32(BUFFER).unwrap(), 1);

    let mut payload = [0u8; 12];
    payload[..4].copy_from_slice(&fd.to_le_bytes());
    payload[4..8].copy_from_slice(&4u32.to_le_bytes()); // F_SETFL
    payload[8..].copy_from_slice(&0x0800u32.to_le_bytes());
    write_request(&mut cpu, 20, &payload);
    cpu.bsd_request(TLS, 9, Some(20)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));

    payload[4..8].copy_from_slice(&3u32.to_le_bytes()); // F_GETFL
    write_request(&mut cpu, 20, &payload);
    cpu.bsd_request(TLS, 9, Some(20)).unwrap();
    assert_eq!(bsd_result(&cpu), (0x0800, 0));
}

#[test]
pub(in crate::services::net) fn a_poll_with_a_timeout_gives_up_the_cpu() {
    // A poll that waits must yield, or a polling loop starves every other thread.
    let (mut cpu, _fd) = bsd_socket(1);
    let mut payload = [0u8; 8];
    payload[..4].copy_from_slice(&1u32.to_le_bytes()); // nfds
    payload[4..].copy_from_slice(&200i32.to_le_bytes()); // timeout, ms
    write_request(&mut cpu, 6, &payload);
    cpu.bsd_request(TLS, 9, Some(6)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "no descriptor is ever ready");
    assert_eq!(
        cpu.pending_sleep.take(),
        cpu.wait_deadline(super::BSD_WAIT_SLICE_NS),
        "a poll that waits parks, but only for a slice"
    );

    payload[4..].copy_from_slice(&(-1i32).to_le_bytes());
    write_request(&mut cpu, 6, &payload);
    cpu.bsd_request(TLS, 9, Some(6)).unwrap();
    assert_eq!(
        cpu.pending_sleep.take(),
        cpu.wait_deadline(super::BSD_WAIT_SLICE_NS)
    );

    // A zero timeout is a non-blocking probe.
    payload[4..].copy_from_slice(&0i32.to_le_bytes());
    write_request(&mut cpu, 6, &payload);
    cpu.bsd_request(TLS, 9, Some(6)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));
    assert_eq!(cpu.pending_sleep, None);
}

#[test]
pub(in crate::services::net) fn bsd_bind_assigns_a_port_and_get_sock_name_reports_a_usable_address()
{
    // asio binds to port 0 and connects to what `getsockname` reports, so the reply
    // must carry a real port and `sin_len`.
    let (mut cpu, fd) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    place(&mut cpu, SCRATCH, &loopback_sockaddr(0));
    write_map_buffer_request(&mut cpu, 13, &fd.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(13)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "bind");

    place(&mut cpu, SCRATCH, &[0u8; 16]);
    write_map_buffer_request(&mut cpu, 16, &fd.to_le_bytes(), SCRATCH, 16, false);
    cpu.bsd_request(TLS, 9, Some(16)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));
    assert_eq!(
        bsd_result_len(&cpu),
        16,
        "the length of the reported address"
    );

    let reported = cpu.read_bytes(SCRATCH, 16);
    assert_eq!(reported[0], 16, "sin_len");
    assert_eq!(reported[1], 2, "AF_INET");
    assert_eq!(
        &reported[4..8],
        &[127, 0, 0, 1],
        "the address stays the one bound"
    );
    let port = u16::from_be_bytes([reported[2], reported[3]]);
    assert!(
        port >= super::BSD_FIRST_EPHEMERAL_PORT,
        "an ephemeral port, not 0: {port}"
    );

    let other = open_socket(&mut cpu, 1);
    place(&mut cpu, SCRATCH, &loopback_sockaddr(8080));
    write_map_buffer_request(&mut cpu, 13, &other.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(13)).unwrap();
    place(&mut cpu, SCRATCH, &[0u8; 16]);
    write_map_buffer_request(&mut cpu, 16, &other.to_le_bytes(), SCRATCH, 16, false);
    cpu.bsd_request(TLS, 9, Some(16)).unwrap();
    assert_eq!(cpu.read_bytes(SCRATCH, 4)[2..], 8080u16.to_be_bytes());
}

#[test]
pub(in crate::services::net) fn bsd_builds_the_socket_pair_asio_wakes_its_own_select_with() {
    // asio's `socket_select_interrupter::open_descriptors`, then an interrupt.
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    let (client, server) = connected_pair(&mut cpu, listener);
    assert_ne!(client, server);
    // `open_descriptors` makes both ends non-blocking.
    set_nonblocking(&mut cpu, client);
    set_nonblocking(&mut cpu, server);

    // `interrupt()` writes one byte...
    assert_eq!(send_on(&mut cpu, client, &[0x7f]), (1, 0), "send");
    // ...and `reset()` drains it at the other end.
    assert_eq!(
        recv_on(&mut cpu, server, 0x20),
        ((1, 0), vec![0x7f]),
        "recv"
    );
    assert_eq!(recv_on(&mut cpu, server, 0x20).0, (-1, super::BSD_EAGAIN));

    assert_eq!(recv_on(&mut cpu, client, 0x20).0, (-1, super::BSD_EAGAIN));

    place(&mut cpu, SCRATCH, &[0u8; 16]);
    write_map_buffer_request(&mut cpu, 15, &server.to_le_bytes(), SCRATCH, 16, false);
    cpu.bsd_request(TLS, 9, Some(15)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "getpeername");
    assert_eq!(bsd_result_len(&cpu), 16);
    assert_eq!(cpu.read_bytes(SCRATCH, 2), vec![16, 2]);
}

#[test]
pub(in crate::services::net) fn bsd_select_names_the_descriptor_that_has_a_byte() {
    const SET: u32 = 0x80;
    const SETS: u32 = 0x6000;
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    cpu.mem.map_zero(SETS, 6 * SET as usize).unwrap();
    let (client, server) = connected_pair(&mut cpu, listener);

    let sets: Vec<(u32, u32)> = (0..6).map(|index| (SETS + index * SET, SET)).collect();
    let select = |cpu: &mut Cpu, watch: i32, micros: u64| {
        for offset in 0..6 * SET {
            cpu.mem.write_u8(SETS + offset, 0).unwrap();
        }
        let bit = 1u8 << (watch % 8);
        cpu.mem.write_u8(SETS + (watch as u32 / 8), bit).unwrap();
        let mut payload = [0u8; 32];
        payload[..4].copy_from_slice(&((watch + 1) as u32).to_le_bytes());
        payload[16..24].copy_from_slice(&micros.to_le_bytes());
        write_buffer_request(cpu, 5, &payload, &sets[..3], &sets[3..]);
        cpu.pending_sleep = None;
        cpu.bsd_request(TLS, 9, Some(5)).unwrap();
        let ready = bsd_result(cpu);
        let out = cpu
            .mem
            .read_u8(SETS + 3 * SET + (watch as u32 / 8))
            .unwrap();
        (ready, out & bit != 0)
    };

    // Nothing sent, so nothing ready, and the wait parks for its timeout.
    assert_eq!(select(&mut cpu, server, 300), ((0, 0), false));
    assert_eq!(cpu.pending_sleep, cpu.wait_deadline(300_000));
    // A long wait parks only for a slice.
    assert_eq!(select(&mut cpu, server, 1_000_000), ((0, 0), false));
    assert_eq!(
        cpu.pending_sleep,
        cpu.wait_deadline(super::BSD_WAIT_SLICE_NS)
    );

    assert_eq!(send_on(&mut cpu, client, &[0x7f]), (1, 0));
    // Readiness is reported in the output set.
    assert_eq!(select(&mut cpu, server, 1_000_000), ((1, 0), true));
    assert_eq!(cpu.pending_sleep, None, "nothing to wait for");
}

#[test]
pub(in crate::services::net) fn a_peer_that_closed_is_end_of_file_and_not_a_connection() {
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    let (client, server) = connected_pair(&mut cpu, listener);

    assert_eq!(send_on(&mut cpu, client, &[1, 2, 3]), (3, 0));
    write_request(&mut cpu, 26, &client.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(26)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "close");
    assert_eq!(recv_on(&mut cpu, server, 0x20), ((3, 0), vec![1, 2, 3]));

    // Then end-of-file.
    assert_eq!(recv_on(&mut cpu, server, 0x20).0, (0, 0));
    assert_eq!(send_on(&mut cpu, server, &[4]), (-1, super::BSD_EPIPE));
}

#[test]
pub(in crate::services::net) fn bsd_connects_to_this_console_and_refuses_everywhere_else() {
    let (mut cpu, fd) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();

    // A loopback port nothing listens on is refused.
    place(&mut cpu, SCRATCH, &loopback_sockaddr(9999));
    write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(
        bsd_result(&cpu),
        (-1, super::BSD_ECONNREFUSED),
        "nothing is listening"
    );

    // And an address off this console has no route at all.
    let mut remote = loopback_sockaddr(53);
    remote[4..8].copy_from_slice(&[8, 8, 8, 8]);
    place(&mut cpu, SCRATCH, &remote);
    write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(
        bsd_result(&cpu),
        (-1, super::BSD_ECONNREFUSED),
        "off the console"
    );

    let listener = open_socket(&mut cpu, 1);
    let (client, _server) = connected_pair(&mut cpu, listener);
    write_map_buffer_request(&mut cpu, 14, &client.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(bsd_result(&cpu), (-1, super::BSD_EISCONN));
}

#[test]
pub(in crate::services::net) fn bsd_get_sock_name_reports_the_address_nifm_does() {
    const BUFFER: u32 = 0x4000;
    let (mut cpu, fd) = bsd_socket(1);
    cpu.mem.map_zero(BUFFER, 0x100).unwrap();
    write_map_buffer_request(&mut cpu, 16, &fd.to_le_bytes(), BUFFER, 0x10, false);
    cpu.bsd_request(TLS, 9, Some(16)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));
    // FreeBSD's sockaddr_in: length, family, then port and address in network order.
    assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), 16);
    assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), 2, "AF_INET");
    assert_eq!(cpu.read_bytes(BUFFER + 4, 4), super::NIFM_LOCAL_IP.to_vec());
}

pub(in crate::services::net) fn bsd_command(
    cpu: &mut Cpu,
    command: u32,
    words: &[u32],
) -> (i32, i32) {
    let payload: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_request(cpu, command, &payload);
    cpu.bsd_request(TLS, 9, Some(command)).unwrap();
    bsd_result(cpu)
}

/// `setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, timeval)`, the `timeval` as the guest laid it out.
fn set_receive_timeout(cpu: &mut Cpu, fd: i32, timeval: &[u8]) {
    const AT: u32 = SCRATCH + 0x180;
    place(cpu, AT, timeval);
    let mut payload = [0u8; 12];
    payload[..4].copy_from_slice(&fd.to_le_bytes());
    payload[4..8].copy_from_slice(&super::BSD_SOL_SOCKET.to_le_bytes());
    payload[8..].copy_from_slice(&super::BSD_SO_RCVTIMEO.to_le_bytes());
    write_map_buffer_request(cpu, 21, &payload, AT, timeval.len() as u32, true);
    cpu.bsd_request(TLS, 9, Some(21)).unwrap();
    assert_eq!(bsd_result(cpu), (0, 0), "setsockopt");
}

/// A 64-bit guest's `timeval`.
fn timeval(seconds: i64, micros: i64) -> Vec<u8> {
    [seconds.to_le_bytes(), micros.to_le_bytes()].concat()
}

/// A connected pair of loopback stream sockets, both left blocking.
fn blocking_pair() -> (Cpu, i32, i32) {
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    let (client, server) = connected_pair(&mut cpu, listener);
    (cpu, client, server)
}

#[test]
pub(in crate::services::net) fn a_blocking_read_waits_for_bytes_and_then_answers() {
    let (mut cpu, client, server) = blocking_pair();
    recv_on(&mut cpu, server, 0x20);
    assert!(cpu.pending_retry.take().is_some(), "the call waits");
    recv_on(&mut cpu, server, 0x20);
    assert!(cpu.pending_retry.take().is_some(), "and keeps waiting");

    send_on(&mut cpu, client, b"hi");
    assert_eq!(recv_on(&mut cpu, server, 0x20), ((2, 0), b"hi".to_vec()));
    assert!(cpu.pending_retry.is_none());
}

#[test]
pub(in crate::services::net) fn a_blocking_read_ends_when_the_peer_closes() {
    let (mut cpu, client, server) = blocking_pair();
    recv_on(&mut cpu, server, 0x20);
    assert!(cpu.pending_retry.take().is_some());
    assert_eq!(bsd_command(&mut cpu, 26, &[client as u32]), (0, 0), "close");
    assert_eq!(recv_on(&mut cpu, server, 0x20).0, (0, 0), "end of file");
    assert!(cpu.pending_retry.is_none());
}

#[test]
pub(in crate::services::net) fn a_blocking_read_gives_up_after_so_rcvtimeo() {
    let (mut cpu, _client, server) = blocking_pair();
    set_receive_timeout(&mut cpu, server, &timeval(0, 5_000));
    let deadline = cpu.wait_deadline(5_000_000).unwrap();

    recv_on(&mut cpu, server, 0x20);
    assert!(
        cpu.pending_retry.take().is_some(),
        "waits within the timeout"
    );
    cpu.cycles = deadline - 1;
    recv_on(&mut cpu, server, 0x20);
    assert!(
        cpu.pending_retry.take().is_some(),
        "the deadline is the first run's"
    );

    cpu.cycles = deadline;
    assert_eq!(recv_on(&mut cpu, server, 0x20).0, (-1, super::BSD_EAGAIN));
    assert!(cpu.pending_retry.is_none());

    // The next read is a new call with a timeout of its own.
    recv_on(&mut cpu, server, 0x20);
    assert!(cpu.pending_retry.take().is_some());
}

#[test]
pub(in crate::services::net) fn so_rcvtimeo_reads_back_whole_and_parses_either_layout() {
    let (mut cpu, fd) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    let mut value = timeval(2, 500_000);
    set_receive_timeout(&mut cpu, fd, &value);
    assert_eq!(cpu.bsd_receive_timeout(fd), Some(2_500_000_000));

    const OUT: u32 = SCRATCH + 0x100;
    let mut payload = [0u8; 12];
    payload[..4].copy_from_slice(&fd.to_le_bytes());
    payload[4..8].copy_from_slice(&super::BSD_SOL_SOCKET.to_le_bytes());
    payload[8..].copy_from_slice(&super::BSD_SO_RCVTIMEO.to_le_bytes());
    write_map_buffer_request(&mut cpu, 17, &payload, OUT, 16, false);
    cpu.bsd_request(TLS, 9, Some(17)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0));
    assert_eq!(bsd_result_len(&cpu), 16);
    assert_eq!(cpu.read_bytes(OUT, 16), value);

    // A 32-bit FreeBSD guest's `tv_usec` is a 32-bit long; what pads it is not part of it.
    value[12..].copy_from_slice(&[0xAA; 4]);
    set_receive_timeout(&mut cpu, fd, &value);
    assert_eq!(cpu.bsd_receive_timeout(fd), Some(2_500_000_000));

    let short = [3i32.to_le_bytes(), 0i32.to_le_bytes()].concat();
    set_receive_timeout(&mut cpu, fd, &short);
    assert_eq!(cpu.bsd_receive_timeout(fd), Some(3_000_000_000));

    set_receive_timeout(&mut cpu, fd, &timeval(0, 0));
    assert_eq!(cpu.bsd_receive_timeout(fd), None, "zero waits forever");
}

#[test]
pub(in crate::services::net) fn a_blocking_accept_waits_for_a_connection() {
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    place(&mut cpu, SCRATCH, &loopback_sockaddr(0));
    write_map_buffer_request(&mut cpu, 13, &listener.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(13)).unwrap();
    assert_eq!(
        bsd_command(&mut cpu, 18, &[listener as u32]),
        (0, 0),
        "listen"
    );

    write_request(&mut cpu, 12, &listener.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(12)).unwrap();
    assert!(cpu.pending_retry.take().is_some(), "nothing to accept yet");

    let port = super::sockaddr_in(&cpu.process().bsd_sockets[&listener].bound)
        .unwrap()
        .1;
    let client = open_socket(&mut cpu, 1);
    place(&mut cpu, SCRATCH, &loopback_sockaddr(port));
    write_map_buffer_request(&mut cpu, 14, &client.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "connect");

    write_request(&mut cpu, 12, &listener.to_le_bytes());
    cpu.bsd_request(TLS, 9, Some(12)).unwrap();
    assert!(cpu.pending_retry.is_none());
    let (accepted, errno) = bsd_result(&cpu);
    assert_eq!(errno, 0);
    assert!(accepted > client);
}

/// `ioctl(fd, FIONREAD)`: the result and the count.
fn bytes_waiting(cpu: &mut Cpu, fd: i32) -> ((i32, i32), u32) {
    const OUT: u32 = SCRATCH + 0x1c0;
    let payload: Vec<u8> = [fd as u32, super::BSD_FIONREAD, 1]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    place(cpu, OUT, &[0xFF; 4]);
    write_map_buffer_request(cpu, 19, &payload, OUT, 4, false);
    cpu.bsd_request(TLS, 9, Some(19)).unwrap();
    (bsd_result(cpu), cpu.mem.read_u32(OUT).unwrap())
}

#[test]
pub(in crate::services::net) fn msg_peek_reads_without_taking() {
    let (mut cpu, client, server) = blocking_pair();
    send_on(&mut cpu, client, b"abc");
    assert_eq!(
        recv_with(&mut cpu, server, 2, super::BSD_MSG_PEEK),
        ((2, 0), b"ab".to_vec())
    );
    assert_eq!(recv_on(&mut cpu, server, 0x20), ((3, 0), b"abc".to_vec()));
}

#[test]
pub(in crate::services::net) fn msg_dontwait_makes_one_read_non_blocking() {
    let (mut cpu, _client, server) = blocking_pair();
    cpu.pending_yield = false;
    let (result, _) = recv_with(&mut cpu, server, 0x20, super::BSD_MSG_DONTWAIT);
    assert_eq!(result, (-1, super::BSD_EAGAIN));
    assert!(cpu.pending_retry.is_none(), "no wait");
    assert!(cpu.pending_yield);

    recv_on(&mut cpu, server, 0x20);
    assert!(
        cpu.pending_retry.is_some(),
        "the socket itself still blocks"
    );
}

#[test]
pub(in crate::services::net) fn msg_waitall_waits_for_the_whole_buffer_or_the_end() {
    let (mut cpu, client, server) = blocking_pair();
    let wait_all = super::BSD_MSG_WAITALL;
    send_on(&mut cpu, client, b"ab");
    recv_with(&mut cpu, server, 4, wait_all);
    assert!(cpu.pending_retry.take().is_some(), "two of four bytes");
    send_on(&mut cpu, client, b"cd");
    assert_eq!(
        recv_with(&mut cpu, server, 4, wait_all),
        ((4, 0), b"abcd".to_vec())
    );
    assert!(cpu.pending_retry.is_none());

    send_on(&mut cpu, client, b"x");
    recv_with(&mut cpu, server, 4, wait_all);
    assert!(cpu.pending_retry.take().is_some());
    bsd_command(&mut cpu, 26, &[client as u32]);
    assert_eq!(
        recv_with(&mut cpu, server, 4, wait_all),
        ((1, 0), b"x".to_vec()),
        "what there was when the peer closed"
    );
}

#[test]
pub(in crate::services::net) fn msg_waitall_gives_what_there_is_when_the_wait_times_out() {
    let (mut cpu, client, server) = blocking_pair();
    set_receive_timeout(&mut cpu, server, &timeval(0, 1_000));
    let deadline = cpu.wait_deadline(1_000_000).unwrap();
    send_on(&mut cpu, client, b"ab");
    recv_with(&mut cpu, server, 4, super::BSD_MSG_WAITALL);
    assert!(cpu.pending_retry.take().is_some());
    cpu.cycles = deadline;
    assert_eq!(
        recv_with(&mut cpu, server, 4, super::BSD_MSG_WAITALL),
        ((2, 0), b"ab".to_vec())
    );
}

#[test]
pub(in crate::services::net) fn msg_oob_finds_no_urgent_data() {
    let (mut cpu, client, server) = blocking_pair();
    send_on(&mut cpu, client, b"a");
    assert_eq!(
        recv_with(&mut cpu, server, 4, super::BSD_MSG_OOB).0,
        (-1, super::BSD_EINVAL)
    );
}

#[test]
pub(in crate::services::net) fn fionread_counts_what_is_queued() {
    let (mut cpu, client, server) = blocking_pair();
    assert_eq!(bytes_waiting(&mut cpu, server), ((0, 0), 0));
    send_on(&mut cpu, client, b"hello");
    assert_eq!(bytes_waiting(&mut cpu, server), ((0, 0), 5));
    recv_on(&mut cpu, server, 2);
    assert_eq!(bytes_waiting(&mut cpu, server), ((0, 0), 3));
    assert_eq!(bytes_waiting(&mut cpu, 999).0, (-1, super::BSD_EBADF));
}

#[test]
pub(in crate::services::net) fn open_and_sysctl_find_nothing_and_the_registration_variants_succeed()
{
    let (mut cpu, _) = bsd_socket(1);
    assert_eq!(
        bsd_command(&mut cpu, 4, &[0]),
        (-1, super::BSD_ENOENT),
        "open"
    );
    assert_eq!(
        bsd_command(&mut cpu, 7, &[0]),
        (-1, super::BSD_ENOENT),
        "sysctl"
    );
    assert_eq!(
        bsd_command(&mut cpu, 33, &[0; 8]),
        (0, 0),
        "RegisterClientShared"
    );
    assert_eq!(bsd_command(&mut cpu, 36, &[0; 4]).0, 0, "Cmd36");
    assert!(
        cpu.unimplemented_ipc.is_empty(),
        "{:?}",
        cpu.unimplemented_ipc
    );
}

/// `setsockopt(fd, SOL_SOCKET, option, value)`.
fn set_option(cpu: &mut Cpu, fd: i32, option: u32, value: &[u8]) {
    const AT: u32 = SCRATCH + 0x1e0;
    place(cpu, AT, value);
    let payload: Vec<u8> = [fd as u32, super::BSD_SOL_SOCKET, option]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    write_map_buffer_request(cpu, 21, &payload, AT, value.len() as u32, true);
    cpu.bsd_request(TLS, 9, Some(21)).unwrap();
    assert_eq!(bsd_result(cpu), (0, 0), "setsockopt");
}

/// `bind(fd, ip:port)`.
fn bind_to(cpu: &mut Cpu, fd: i32, ip: [u8; 4], port: u16) -> (i32, i32) {
    let mut address = loopback_sockaddr(port);
    address[4..8].copy_from_slice(&ip);
    place(cpu, SCRATCH, &address);
    write_map_buffer_request(cpu, 13, &fd.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(13)).unwrap();
    bsd_result(cpu)
}

#[test]
pub(in crate::services::net) fn a_port_held_by_a_socket_of_the_same_kind_is_in_use() {
    let (mut cpu, first) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    assert_eq!(bind_to(&mut cpu, first, [0, 0, 0, 0], 8080), (0, 0));
    let second = open_socket(&mut cpu, 1);
    assert_eq!(
        bind_to(&mut cpu, second, [127, 0, 0, 1], 8080),
        (-1, super::BSD_EADDRINUSE),
        "the wildcard covers loopback"
    );
    assert_eq!(bind_to(&mut cpu, second, [0, 0, 0, 0], 8081), (0, 0));

    let datagram = open_socket(&mut cpu, 2);
    assert_eq!(
        bind_to(&mut cpu, datagram, [0, 0, 0, 0], 8080),
        (0, 0),
        "UDP and TCP ports are apart"
    );

    let third = open_socket(&mut cpu, 1);
    let fourth = open_socket(&mut cpu, 1);
    set_option(
        &mut cpu,
        third,
        super::BSD_SO_REUSEADDR,
        &1u32.to_le_bytes(),
    );
    set_option(
        &mut cpu,
        fourth,
        super::BSD_SO_REUSEADDR,
        &1u32.to_le_bytes(),
    );
    assert_eq!(bind_to(&mut cpu, third, [0, 0, 0, 0], 9000), (0, 0));
    assert_eq!(
        bind_to(&mut cpu, fourth, [0, 0, 0, 0], 9000),
        (0, 0),
        "both asked to share"
    );
}

#[test]
pub(in crate::services::net) fn shut_rd_drops_what_is_queued_and_reads_end_of_file() {
    let (mut cpu, client, server) = blocking_pair();
    send_on(&mut cpu, client, b"before");
    assert_eq!(bsd_command(&mut cpu, 22, &[server as u32, 0]), (0, 0));
    assert_eq!(recv_on(&mut cpu, server, 16).0, (0, 0));
    send_on(&mut cpu, client, b"after");
    assert_eq!(
        recv_on(&mut cpu, server, 16).0,
        (0, 0),
        "and what comes later"
    );
    assert_eq!(send_on(&mut cpu, server, b"x"), (1, 0), "it may still send");
    assert_eq!(bsd_command(&mut cpu, 22, &[server as u32, 1]), (0, 0));
    assert_eq!(send_on(&mut cpu, server, b"x"), (-1, super::BSD_EPIPE));
    assert_eq!(
        bsd_command(&mut cpu, 22, &[server as u32, 3]),
        (-1, super::BSD_EINVAL)
    );
}

#[test]
pub(in crate::services::net) fn a_listener_outside_loopback_takes_connections_the_host_accepts() {
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    cpu.set_network(true);
    bind_to(&mut cpu, listener, [0, 0, 0, 0], 8080);
    assert_eq!(bsd_command(&mut cpu, 18, &[listener as u32, 8]), (0, 0));
    assert_eq!(
        cpu.take_net_commands(),
        vec![NetCommand::Listen {
            stream: 1,
            port: 8080
        }]
    );

    let incoming = 0x8000_0000;
    cpu.deliver_net_events([
        NetEvent::Connected { stream: 1 },
        NetEvent::Incoming {
            listener: 1,
            stream: incoming,
            source: Source::V4([203, 0, 113, 9]),
            port: 50000,
        },
        NetEvent::Data {
            stream: incoming,
            data: b"hello".to_vec(),
        },
    ]);
    place(&mut cpu, SCRATCH, &[0u8; 16]);
    write_map_buffer_request(&mut cpu, 12, &listener.to_le_bytes(), SCRATCH, 16, false);
    cpu.bsd_request(TLS, 9, Some(12)).unwrap();
    let (accepted, errno) = bsd_result(&cpu);
    assert_eq!(errno, 0);
    assert_eq!(
        super::sockaddr_in(&cpu.read_bytes(SCRATCH, 16)),
        Some(([203, 0, 113, 9], 50000))
    );
    assert_eq!(recv_on(&mut cpu, accepted, 16), ((5, 0), b"hello".to_vec()));
    assert_eq!(send_on(&mut cpu, accepted, b"hi"), (2, 0));
    assert!(cpu.take_net_commands().contains(&NetCommand::Send {
        stream: incoming,
        data: b"hi".to_vec()
    }));

    // Closing the listener closes its host listener.
    assert_eq!(bsd_command(&mut cpu, 26, &[listener as u32]), (0, 0));
    assert_eq!(
        cpu.take_net_commands(),
        vec![NetCommand::Close { stream: 1 }]
    );
    // A connection for a listener that is gone is closed.
    cpu.deliver_net_events([NetEvent::Incoming {
        listener: 1,
        stream: incoming + 1,
        source: Source::V4([203, 0, 113, 9]),
        port: 50001,
    }]);
    assert_eq!(
        cpu.take_net_commands(),
        vec![NetCommand::Close {
            stream: incoming + 1
        }]
    );
}

#[test]
pub(in crate::services::net) fn a_listener_the_host_refuses_still_listens_on_this_console() {
    let (mut cpu, listener) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    cpu.set_network(true);
    bind_to(&mut cpu, listener, [0, 0, 0, 0], 8080);
    bsd_command(&mut cpu, 18, &[listener as u32, 8]);
    cpu.deliver_net_events([NetEvent::Failed {
        stream: 1,
        reason: NetFailure::Denied,
    }]);
    assert_eq!(cpu.process().bsd_sockets[&listener].host_listener, None);
    let client = open_socket(&mut cpu, 1);
    place(&mut cpu, SCRATCH, &loopback_sockaddr(8080));
    write_map_buffer_request(&mut cpu, 14, &client.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    assert_eq!(bsd_result(&cpu), (0, 0), "loopback connects still work");
}

#[test]
pub(in crate::services::net) fn each_blocked_thread_keeps_its_own_deadline() {
    let (mut cpu, _client, server) = blocking_pair();
    set_receive_timeout(&mut cpu, server, &timeval(0, 5_000));
    let first_deadline = cpu.wait_deadline(5_000_000).unwrap();
    recv_on(&mut cpu, server, 8);
    assert!(cpu.pending_retry.take().is_some(), "thread 0 waits");

    cpu.current_thread = 1;
    cpu.cycles = first_deadline - 10;
    recv_on(&mut cpu, server, 8);
    assert!(
        cpu.pending_retry.take().is_some(),
        "thread 1 waits from now"
    );

    cpu.cycles = first_deadline;
    recv_on(&mut cpu, server, 8);
    assert!(
        cpu.pending_retry.take().is_some(),
        "thread 1's deadline is later"
    );
    cpu.current_thread = 0;
    assert_eq!(
        recv_on(&mut cpu, server, 8).0,
        (-1, super::BSD_EAGAIN),
        "thread 0's has passed"
    );
}

/// `EventFd(initval, flags)` on a fresh `bsd:u` session, with `SCRATCH` mapped.
fn eventfd(initval: u64, flags: u32) -> (Cpu, i32) {
    let mut cpu = request(false, 31, &[]);
    cpu.register_service_handle(9, "bsd:u");
    cpu.mem.map_zero(SCRATCH, 0x200).unwrap();
    let mut payload = [0u8; 12];
    payload[..8].copy_from_slice(&initval.to_le_bytes());
    payload[8..].copy_from_slice(&flags.to_le_bytes());
    write_request(&mut cpu, 31, &payload);
    cpu.bsd_request(TLS, 9, Some(31)).unwrap();
    let (fd, errno) = bsd_result(&cpu);
    assert_eq!(errno, 0, "eventfd");
    (cpu, fd)
}

/// `Read(fd)` into `SCRATCH`: the result and the 8 bytes read.
fn read_counter(cpu: &mut Cpu, fd: i32) -> ((i32, i32), u64) {
    write_map_buffer_request(cpu, 25, &fd.to_le_bytes(), SCRATCH, 8, false);
    cpu.bsd_request(TLS, 9, Some(25)).unwrap();
    (bsd_result(cpu), cpu.mem.read_u64(SCRATCH).unwrap())
}

#[test]
fn an_eventfd_reads_back_what_was_written_and_is_readable_only_until_then() {
    const POLLIN: u16 = 1;
    const POLLOUT: u16 = 4;
    let (mut cpu, fd) = eventfd(2, 0);
    assert_eq!(cpu.bsd_poll_revents(fd, POLLIN | POLLOUT), POLLIN | POLLOUT);

    place(&mut cpu, SCRATCH, &5u64.to_le_bytes());
    write_map_buffer_request(&mut cpu, 24, &fd.to_le_bytes(), SCRATCH, 8, true);
    cpu.bsd_request(TLS, 9, Some(24)).unwrap();
    assert_eq!(bsd_result(&cpu), (8, 0), "write");

    assert_eq!(read_counter(&mut cpu, fd), ((8, 0), 7), "the whole count");
    assert_eq!(cpu.bsd_poll_revents(fd, POLLIN | POLLOUT), POLLOUT);
    set_nonblocking(&mut cpu, fd);
    assert_eq!(read_counter(&mut cpu, fd).0, (-1, super::BSD_EAGAIN));
}

#[test]
fn a_semaphore_eventfd_reads_one_at_a_time() {
    let (mut cpu, fd) = eventfd(2, super::BSD_EFD_SEMAPHORE);
    set_nonblocking(&mut cpu, fd);
    assert_eq!(read_counter(&mut cpu, fd), ((8, 0), 1));
    assert_eq!(read_counter(&mut cpu, fd), ((8, 0), 1));
    assert_eq!(read_counter(&mut cpu, fd).0, (-1, super::BSD_EAGAIN));
}
