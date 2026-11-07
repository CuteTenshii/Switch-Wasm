use super::*;
use crate::kernel::ipc::testing::*;
use crate::services::net::bsd::tests::{bsd_socket, loopback_sockaddr, place, SCRATCH};

const KEY: u64 = 7 << 32;
const BUFFER: u32 = SCRATCH + 0x400;
const NONBLOCKING: u32 = 2;

/// A console with the network on, a socket connected to `example.com` and open on the host,
/// and a connection that holds it.
fn connected() -> (Cpu, i32) {
    let (mut cpu, fd) = bsd_socket(1);
    cpu.mem.map_zero(SCRATCH, 0x1000).unwrap();
    cpu.set_network(true);
    cpu.process_mut().bsd_sockets.get_mut(&fd).unwrap().flags |= 0x800;
    let mut remote = loopback_sockaddr(443);
    remote[4..8].copy_from_slice(&cpu.net.synthesize("example.com").unwrap());
    place(&mut cpu, SCRATCH, &remote);
    write_map_buffer_request(&mut cpu, 14, &fd.to_le_bytes(), SCRATCH, 16, true);
    cpu.bsd_request(TLS, 9, Some(14)).unwrap();
    cpu.take_net_commands();
    cpu.deliver_net_events([NetEvent::Connected { stream: 1 }]);
    cpu.process_mut()
        .ssl_connections
        .insert(KEY, SslConnection::default());
    (cpu, fd)
}

/// Issue a command and read the result and the first reply words.
fn call(
    cpu: &mut Cpu,
    cmd: u32,
    payload: &[u8],
    send: &[(u32, u32)],
    recv: &[(u32, u32)],
) -> (u32, [u32; 2]) {
    write_buffer_request(cpu, cmd, payload, send, recv);
    cpu.ssl_connection_request(TLS, KEY, Some(cmd)).unwrap();
    let word = |at: u32| cpu.mem.read_u32(TLS + at).unwrap();
    (word(0x18), [word(0x20), word(0x24)])
}

fn simple(cpu: &mut Cpu, cmd: u32, payload: &[u8]) -> (u32, [u32; 2]) {
    call(cpu, cmd, payload, &[], &[])
}

fn set_socket(cpu: &mut Cpu, fd: i32) {
    assert_eq!(simple(cpu, 0, &fd.to_le_bytes()).0, 0);
}

fn established(chain: Vec<Vec<u8>>) -> NetEvent {
    NetEvent::TlsEstablished {
        stream: 1,
        protocol: "TLSv1.3".into(),
        cipher: "TLS13_AES_128_GCM_SHA256".into(),
        alpn: Some(b"h2".to_vec()),
        chain,
    }
}

fn host_name(cpu: &mut Cpu, name: &str) {
    place(cpu, BUFFER, name.as_bytes());
    let (result, _) = call(cpu, 1, &[], &[(BUFFER, name.len() as u32)], &[]);
    assert_eq!(result, 0);
}

/// A connection that has completed its handshake, set to the given I/O mode.
fn handshaken(io_mode: u32) -> (Cpu, i32) {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 3, &io_mode.to_le_bytes());
    simple(&mut cpu, 8, &[]);
    cpu.pending_retry = None;
    cpu.take_net_commands();
    cpu.deliver_net_events([established(vec![vec![1, 2, 3], vec![4, 5]])]);
    (cpu, fd)
}

#[test]
fn a_new_connection_verifies_and_blocks_until_told_otherwise() {
    let (mut cpu, fd) = connected();
    assert_eq!(simple(&mut cpu, 6, &[]).1[0], 0x3);
    assert_eq!(simple(&mut cpu, 7, &[]).1[0], 1);
    assert_eq!(simple(&mut cpu, 2, &0x1u32.to_le_bytes()).0, 0);
    assert_eq!(simple(&mut cpu, 6, &[]).1[0], 0x1);
    // The modes that hang off a socket need one first.
    assert_eq!(
        simple(&mut cpu, 3, &2u32.to_le_bytes()).0,
        RESULT_INTERNAL_LOGIC
    );
    assert_eq!(simple(&mut cpu, 4, &[]).0, RESULT_INTERNAL_LOGIC);
    set_socket(&mut cpu, fd);
    assert_eq!(simple(&mut cpu, 3, &2u32.to_le_bytes()).0, 0);
    assert_eq!(simple(&mut cpu, 7, &[]).1[0], 2);
    assert_eq!(simple(&mut cpu, 4, &[]).1[0] as i32, fd);
}

#[test]
fn the_socket_comes_back_as_a_duplicate_and_can_only_be_given_once() {
    let (mut cpu, fd) = connected();
    let (result, [duplicate, _]) = simple(&mut cpu, 0, &fd.to_le_bytes());
    assert_eq!(result, 0);
    assert_ne!(duplicate as i32, fd);
    assert!(cpu.process().bsd_sockets.contains_key(&(duplicate as i32)));
    assert_eq!(
        simple(&mut cpu, 0, &fd.to_le_bytes()).0,
        RESULT_INTERNAL_LOGIC
    );

    cpu.process_mut()
        .ssl_connections
        .insert(KEY, SslConnection::default());
    assert_eq!(
        simple(&mut cpu, 0, &99i32.to_le_bytes()).0,
        RESULT_INVALID_SOCKET
    );
}

#[test]
fn the_host_name_is_kept_with_the_environment_filled_in() {
    let (mut cpu, _) = connected();
    host_name(&mut cpu, "api.%.example");
    place(&mut cpu, BUFFER, &[0; 64]);
    let (result, [length, _]) = call(&mut cpu, 5, &[], &[], &[(BUFFER, 64)]);
    assert_eq!(result, 0);
    assert_eq!(length, "api.lp1.example".len() as u32);
    assert_eq!(cpu.read_string(BUFFER, 64), "api.lp1.example");
    place(&mut cpu, BUFFER, &[b'a'; 0x100]);
    let long = call(&mut cpu, 1, &[], &[(BUFFER, 0x100)], &[]).0;
    assert_eq!(long, RESULT_INTERNAL_LOGIC, "a name is at most 0xFF bytes");
}

#[test]
fn a_non_blocking_handshake_says_wait_until_the_host_has_finished() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 3, &NONBLOCKING.to_le_bytes());

    assert_eq!(simple(&mut cpu, 8, &[]).0, RESULT_WOULD_BLOCK);
    let commands = cpu.take_net_commands();
    let [NetCommand::StartTls {
        stream,
        server_name,
        verify,
        roots,
        ..
    }] = &commands[..]
    else {
        panic!("one StartTls, got {commands:?}");
    };
    assert_eq!(
        (*stream, server_name.as_str(), *verify),
        (1, "example.com", 3)
    );
    assert!(roots.is_some(), "the first request brings the roots");

    assert_eq!(simple(&mut cpu, 8, &[]).0, RESULT_WOULD_BLOCK);
    assert!(cpu.take_net_commands().is_empty(), "asked once");

    cpu.deliver_net_events([established(vec![vec![1]])]);
    assert_eq!(simple(&mut cpu, 8, &[]).0, 0);
    assert!(cpu.pending_retry.is_none());
}

#[test]
fn a_blocking_handshake_waits_in_the_kernel_and_then_succeeds() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 8, &[]);
    assert!(cpu.pending_retry.take().is_some(), "no reply, run again");
    simple(&mut cpu, 8, &[]);
    assert!(cpu.pending_retry.take().is_some());
    assert_eq!(
        cpu.take_net_commands().len(),
        1,
        "the host is asked once, however often the call runs"
    );
    cpu.deliver_net_events([established(vec![])]);
    write_buffer_request(&mut cpu, 8, &[], &[], &[]);
    cpu.ssl_connection_request(TLS, KEY, Some(8)).unwrap();
    assert!(cpu.pending_retry.is_none());
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);
}

#[test]
fn a_blocking_wait_gives_up_after_five_minutes_of_guest_time() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 8, &[]);
    assert!(cpu.pending_retry.take().is_some());
    cpu.cycles += 301 * 1_020_000_000;
    assert_eq!(simple(&mut cpu, 8, &[]).0, RESULT_IO_TIMEOUT);
}

#[test]
fn a_handshake_needs_a_name_when_the_name_is_checked_and_a_connection_to_run_on() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    assert_eq!(
        simple(&mut cpu, 8, &[]).0,
        RESULT_INTERNAL_LOGIC,
        "no host name"
    );
    simple(&mut cpu, 2, &0x1u32.to_le_bytes());
    simple(&mut cpu, 3, &NONBLOCKING.to_le_bytes());
    assert_eq!(
        simple(&mut cpu, 8, &[]).0,
        RESULT_WOULD_BLOCK,
        "no name check, no name needed"
    );

    // A socket that never connected has no stream.
    let (mut cpu, _) = connected();
    let unconnected = cpu.bsd_duplicate(3).unwrap();
    set_socket(&mut cpu, unconnected);
    host_name(&mut cpu, "example.com");
    assert_eq!(simple(&mut cpu, 8, &[]).0, RESULT_NO_CONNECTION);
}

#[test]
fn a_failed_handshake_returns_the_service_result_and_the_certificate_error_once() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 3, &NONBLOCKING.to_le_bytes());
    simple(&mut cpu, 8, &[]);
    cpu.deliver_net_events([NetEvent::TlsFailed {
        stream: 1,
        reason: TlsFailure::UnknownCa,
    }]);
    assert_eq!(simple(&mut cpu, 8, &[]).0, RESULT_UNKNOWN_CA);
    assert_eq!(simple(&mut cpu, 15, &[]).0, RESULT_UNKNOWN_CA);
    assert_eq!(simple(&mut cpu, 15, &[]).0, 0, "reported once");

    for (reason, code) in [
        (TlsFailure::Expired, RESULT_EXPIRED_CERT),
        (TlsFailure::NotValidForName, RESULT_INVALID_CERT_DOMAIN),
        (TlsFailure::Handshake, RESULT_ALERT_HANDSHAKE_FAILURE),
        (TlsFailure::Reset, RESULT_CONNECTION_RESET),
    ] {
        let (mut cpu, fd) = connected();
        set_socket(&mut cpu, fd);
        host_name(&mut cpu, "example.com");
        simple(&mut cpu, 3, &NONBLOCKING.to_le_bytes());
        simple(&mut cpu, 8, &[]);
        cpu.deliver_net_events([NetEvent::TlsFailed { stream: 1, reason }]);
        assert_eq!(simple(&mut cpu, 8, &[]).0, code, "{reason:?}");
    }
}

#[test]
fn plaintext_written_goes_to_the_host_and_what_it_decrypted_is_read_back() {
    let (mut cpu, _) = handshaken(1);
    place(&mut cpu, BUFFER, b"GET /");
    let (result, [sent, _]) = call(&mut cpu, 11, &[], &[(BUFFER, 5)], &[]);
    assert_eq!((result, sent), (0, 5));
    assert_eq!(
        cpu.take_net_commands(),
        vec![NetCommand::Send {
            stream: 1,
            data: b"GET /".to_vec()
        }]
    );

    cpu.deliver_net_events([NetEvent::Data {
        stream: 1,
        data: b"200 OK".to_vec(),
    }]);
    assert_eq!(simple(&mut cpu, 12, &[]).1[0], 6, "Pending");
    place(&mut cpu, BUFFER, &[0; 16]);
    let (_, [peeked, _]) = call(&mut cpu, 13, &[], &[], &[(BUFFER, 3)]);
    assert_eq!((peeked, cpu.read_bytes(BUFFER, 3)), (3, b"200".to_vec()));
    assert!(cpu.take_net_commands().is_empty(), "a peek reads nothing");
    let (result, [read, _]) = call(&mut cpu, 10, &[], &[], &[(BUFFER, 16)]);
    assert_eq!((result, read), (0, 6));
    assert_eq!(cpu.read_bytes(BUFFER, 6), b"200 OK");
    assert_eq!(
        cpu.take_net_commands(),
        vec![NetCommand::Consumed {
            stream: 1,
            bytes: 6
        }]
    );
}

#[test]
fn reading_with_nothing_there_blocks_or_says_wait_and_end_of_file_is_zero_bytes() {
    let (mut cpu, _) = handshaken(1);
    call(&mut cpu, 10, &[], &[], &[(BUFFER, 16)]);
    assert!(cpu.pending_retry.take().is_some(), "a blocking read waits");

    cpu.deliver_net_events([NetEvent::Closed { stream: 1 }]);
    let (result, [read, _]) = call(&mut cpu, 10, &[], &[], &[(BUFFER, 16)]);
    assert_eq!((result, read), (0, 0), "the server closed");

    let (mut cpu, _) = handshaken(NONBLOCKING);
    let (result, _) = call(&mut cpu, 10, &[], &[], &[(BUFFER, 16)]);
    assert_eq!(result, RESULT_WOULD_BLOCK);
    assert!(cpu.pending_retry.is_none());
}

#[test]
fn io_before_the_handshake_has_no_connection_and_writing_after_close_is_refused() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    place(&mut cpu, BUFFER, b"x");
    assert_eq!(
        call(&mut cpu, 11, &[], &[(BUFFER, 1)], &[]).0,
        RESULT_NO_CONNECTION
    );
    assert_eq!(
        call(&mut cpu, 10, &[], &[], &[(BUFFER, 1)]).0,
        RESULT_NO_CONNECTION
    );

    let (mut cpu, _) = handshaken(1);
    cpu.deliver_net_events([NetEvent::Closed { stream: 1 }]);
    place(&mut cpu, BUFFER, b"x");
    assert_eq!(
        call(&mut cpu, 11, &[], &[(BUFFER, 1)], &[]).0,
        RESULT_SOCKET_SHUTDOWN
    );
}

#[test]
fn poll_reports_what_is_ready_and_waits_up_to_its_timeout() {
    let (mut cpu, _) = handshaken(1);
    let poll = |cpu: &mut Cpu, events: u32, timeout: u32| {
        let payload = [events.to_le_bytes(), timeout.to_le_bytes()].concat();
        simple(cpu, 14, &payload)
    };
    let (result, [events, _]) = poll(&mut cpu, 0b11, 0);
    assert_eq!((result, events), (0, 0b10), "writable at once");
    assert_eq!(
        poll(&mut cpu, 0b01, 0).0,
        RESULT_IO_TIMEOUT,
        "nothing to read, no time"
    );
    poll(&mut cpu, 0b01, 50);
    assert!(cpu.pending_retry.take().is_some(), "it waits");
    cpu.deliver_net_events([NetEvent::Data {
        stream: 1,
        data: vec![1],
    }]);
    assert_eq!(poll(&mut cpu, 0b11, 0).1[0], 0b11);
}

#[test]
fn the_server_certificate_comes_back_as_the_leaf_or_as_a_chain() {
    let (mut cpu, _) = handshaken(1);
    place(&mut cpu, BUFFER, &[0; 0x100]);
    let (result, [size, count]) = call(&mut cpu, 9, &[], &[], &[(BUFFER, 0x100)]);
    assert_eq!((result, size, count), (0, 3, 1));
    assert_eq!(cpu.read_bytes(BUFFER, 3), [1, 2, 3]);
    assert_eq!(simple(&mut cpu, 16, &[]).1[0], 3, "the size to allocate");

    // With the chain option the buffer is a header, an entry per certificate, then the bytes.
    assert_eq!(simple(&mut cpu, 22, &[1, 0, 0, 0, 1, 0, 0, 0]).0, 0);
    let (_, [size, count]) = call(&mut cpu, 9, &[], &[], &[(BUFFER, 0x100)]);
    assert_eq!((size, count), (0x10 + 16 + 5, 2));
    assert_eq!(cpu.mem.read_u64(BUFFER).unwrap(), CERT_CHAIN_MAGIC);
    assert_eq!(cpu.mem.read_u32(BUFFER + 8).unwrap(), 2);
    let entry = |at: u32| {
        (
            cpu.mem.read_u32(BUFFER + at).unwrap(),
            cpu.mem.read_u32(BUFFER + at + 4).unwrap(),
        )
    };
    assert_eq!(entry(0x10), (3, 0x20));
    assert_eq!(entry(0x18), (2, 0x23));
    assert_eq!(cpu.read_bytes(BUFFER + 0x20, 5), [1, 2, 3, 4, 5]);

    // No chain check, no certificates.
    simple(&mut cpu, 2, &0u32.to_le_bytes());
    let (_, [size, count]) = call(&mut cpu, 9, &[], &[], &[(BUFFER, 0x100)]);
    assert_eq!((size, count), (0, 0));
}

#[test]
fn the_negotiated_cipher_and_protocol_are_reported_in_a_cipher_info() {
    let (mut cpu, _) = handshaken(1);
    place(&mut cpu, BUFFER, &[0; 0x48]);
    let (result, _) = call(&mut cpu, 25, &1u32.to_le_bytes(), &[], &[(BUFFER, 0x48)]);
    assert_eq!(result, 0);
    assert_eq!(cpu.read_string(BUFFER, 0x40), "TLS13_AES_128_GCM_SHA256");
    assert_eq!(cpu.read_string(BUFFER + 0x40, 8), "TLSv1.3");
    assert_ne!(
        call(&mut cpu, 25, &1u32.to_le_bytes(), &[], &[(BUFFER, 0x40)]).0,
        0,
        "the buffer is the struct's size"
    );
}

#[test]
fn alpn_protocols_are_offered_when_enabled_and_the_chosen_one_is_reported() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 3, &NONBLOCKING.to_le_bytes());
    place(&mut cpu, BUFFER, b"\x02h2\x08http/1.1");
    assert_eq!(call(&mut cpu, 26, &[], &[(BUFFER, 12)], &[]).0, 0);
    assert_eq!(
        simple(&mut cpu, 22, &[1, 0, 0, 0, 3, 0, 0, 0]).0,
        0,
        "EnableAlpn"
    );
    simple(&mut cpu, 8, &[]);
    let commands = cpu.take_net_commands();
    let [NetCommand::StartTls { alpn, .. }] = &commands[..] else {
        panic!("StartTls");
    };
    assert_eq!(alpn, &vec![b"h2".to_vec(), b"http/1.1".to_vec()]);

    cpu.deliver_net_events([established(vec![])]);
    place(&mut cpu, BUFFER, &[0; 16]);
    let (_, [state, length]) = call(&mut cpu, 27, &[], &[], &[(BUFFER, 16)]);
    assert_eq!((state, length), (ALPN_NEGOTIATED, 2));
    assert_eq!(cpu.read_bytes(BUFFER, 2), b"h2");
}

#[test]
fn options_and_modes_read_back() {
    let (mut cpu, fd) = connected();
    assert_eq!(simple(&mut cpu, 23, &1u32.to_le_bytes()).1[0] & 0xFF, 0);
    simple(&mut cpu, 22, &[1, 0, 0, 0, 1, 0, 0, 0]);
    assert_eq!(simple(&mut cpu, 23, &1u32.to_le_bytes()).1[0] & 0xFF, 1);
    // DoNotCloseSocket can only be set before a socket.
    assert_eq!(simple(&mut cpu, 22, &[1, 0, 0, 0, 0, 0, 0, 0]).0, 0);
    set_socket(&mut cpu, fd);
    assert_eq!(
        simple(&mut cpu, 22, &[1, 0, 0, 0, 0, 0, 0, 0]).0,
        RESULT_INTERNAL_LOGIC
    );
    assert_eq!(
        simple(&mut cpu, 22, &[1, 0, 0, 0, 9, 0, 0, 0]).0,
        RESULT_INTERNAL_LOGIC
    );

    simple(&mut cpu, 17, &2u32.to_le_bytes());
    simple(&mut cpu, 20, &1u32.to_le_bytes());
    simple(&mut cpu, 34, &1500u32.to_le_bytes());
    assert_eq!(simple(&mut cpu, 18, &[]).1[0], 2);
    assert_eq!(simple(&mut cpu, 21, &[]).1[0], 1);
    assert_eq!(simple(&mut cpu, 35, &[]).1[0], 1500);
}

#[test]
fn the_roots_go_to_the_host_once() {
    let (mut cpu, fd) = connected();
    set_socket(&mut cpu, fd);
    host_name(&mut cpu, "example.com");
    simple(&mut cpu, 3, &NONBLOCKING.to_le_bytes());
    simple(&mut cpu, 8, &[]);
    assert!(matches!(
        &cpu.take_net_commands()[..],
        [NetCommand::StartTls { roots: Some(_), .. }]
    ));
    assert!(cpu.ssl_roots_to_send().is_none(), "the host keeps them");
}

#[test]
fn closing_the_session_closes_tls_on_the_connections_that_ran_it() {
    let (mut cpu, _) = handshaken(1);
    cpu.ssl_forget_session(7);
    assert_eq!(
        cpu.take_net_commands(),
        vec![NetCommand::TlsClose { stream: 1 }]
    );
    assert!(cpu.process().ssl_connections.is_empty());

    // A connection that never finished a handshake has nothing to close.
    let (mut cpu, _) = connected();
    cpu.ssl_forget_session(7);
    assert!(cpu.take_net_commands().is_empty());
}

#[test]
fn create_connection_makes_an_object_the_context_counts() {
    let mut cpu = request(false, 2, &[]);
    cpu.register_service_handle(9, "ssl:context");
    cpu.ssl_request(TLS, 9, Some(2)).unwrap();
    assert_eq!(cpu.process().ssl_connections.len(), 1);
    write_request(&mut cpu, 3, &[]);
    cpu.ssl_request(TLS, 9, Some(3)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1);
}

#[test]
fn the_system_service_answers_as_ssl_does_and_adds_its_own_creators() {
    let mut cpu = request(false, 5, &[0; 4]);
    cpu.register_service_handle(9, "ssl:s");
    // SetInterfaceVersion, the command that went unanswered.
    cpu.ssl_request(TLS, 9, Some(5)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0);

    // CreateContextForSystem opens a context, and CreateConnectionEx on one a connection.
    write_request(&mut cpu, 100, &[0; 16]);
    cpu.ssl_request(TLS, 9, Some(100)).unwrap();
    assert_eq!(cpu.process().ssl_contexts, 1);
    cpu.register_service_handle(10, "ssl:context");
    write_request(&mut cpu, 100, &[]);
    cpu.ssl_request(TLS, 10, Some(100)).unwrap();
    assert_eq!(cpu.process().ssl_connections.len(), 1);
}

/// A command to `ssl:s` (session 9) with the given buffers; the result and the reply words.
fn service(
    cmd: u32,
    payload: &[u8],
    send: &[(u32, u32)],
    recv: &[(u32, u32)],
    cpu: &mut Cpu,
) -> (u32, u64) {
    write_buffer_request(cpu, cmd, payload, send, recv);
    cpu.ssl_request(TLS, 9, Some(cmd)).unwrap();
    (
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        cpu.mem.read_u64(TLS + 0x20).unwrap(),
    )
}

fn system_service() -> Cpu {
    let mut cpu = request(false, 5, &[]);
    cpu.mem.map_zero(BUFFER, 0x100).unwrap();
    cpu.register_service_handle(9, "ssl:s");
    cpu
}

#[test]
fn the_debug_option_byte_is_kept_and_read_back() {
    let mut cpu = system_service();
    cpu.mem.write_u8(BUFFER, 0x5A).unwrap();
    assert_eq!(
        service(7, &0u32.to_le_bytes(), &[(BUFFER, 1)], &[], &mut cpu).0,
        0
    );
    cpu.mem.write_u8(BUFFER + 0x10, 0).unwrap();
    assert_eq!(
        service(8, &0u32.to_le_bytes(), &[], &[(BUFFER + 0x10, 1)], &mut cpu).0,
        0
    );
    assert_eq!(cpu.mem.read_u8(BUFFER + 0x10).unwrap(), 0x5A);

    // Only option 0 exists, and the buffer must be there.
    assert_eq!(
        service(7, &1u32.to_le_bytes(), &[(BUFFER, 1)], &[], &mut cpu).0,
        RESULT_INTERNAL_LOGIC
    );
    assert_eq!(
        service(7, &0u32.to_le_bytes(), &[(0, 0)], &[], &mut cpu).0,
        RESULT_INTERNAL_LOGIC
    );
    assert_eq!(
        service(8, &1u32.to_le_bytes(), &[], &[(BUFFER + 0x10, 1)], &mut cpu).0,
        RESULT_INTERNAL_LOGIC
    );
    assert_eq!(
        cpu.process().ssl_debug_option,
        0x5A,
        "a refused call changes nothing"
    );
}

#[test]
fn clearing_the_tls_12_fallback_flag_succeeds() {
    let mut cpu = system_service();
    assert_eq!(service(9, &[], &[], &[], &mut cpu).0, 0);
}

#[test]
fn the_thread_core_mask_starts_as_the_process_cores_and_takes_only_those() {
    let mut cpu = system_service();
    cpu.set_process_core_mask(0b0111);
    assert_eq!(service(102, &[], &[], &[], &mut cpu), (0, 0b0111));

    assert_eq!(
        service(101, &0b0100u64.to_le_bytes(), &[], &[], &mut cpu).0,
        0
    );
    assert_eq!(service(102, &[], &[], &[], &mut cpu), (0, 0b0100));

    for refused in [0u64, 0b1000, 0b1100] {
        let (result, _) = service(101, &refused.to_le_bytes(), &[], &[], &mut cpu);
        assert_eq!(result, RESULT_INTERNAL_LOGIC, "{refused:#b}");
    }
    assert_eq!(
        service(102, &[], &[], &[], &mut cpu).1,
        0b0100,
        "a refused mask changes nothing"
    );
}
