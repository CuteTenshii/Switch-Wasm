use crate::cpu::Cpu;

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
    const AT: u32 = SCRATCH + 0x80;
    place(cpu, AT, &vec![0u8; len as usize]);
    write_map_buffer_request(cpu, 8, &fd.to_le_bytes(), AT, len, false);
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

    // Accept before listen is an error; after listen it is EAGAIN.
    let (mut cpu, fd) = bsd_socket(1);
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

    // Nothing arrives: EAGAIN and a yield, not `ENETUNREACH`.
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
