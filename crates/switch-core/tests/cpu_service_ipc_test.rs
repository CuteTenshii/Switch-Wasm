//! IPC mechanics shared by every service: handles, domains, events and replies.

mod cpu;

use cpu::*;
use switch_core::cpu::POINTER_BUFFER_SIZE;

#[test]
fn events_are_copy_handles_and_start_unsignalled() {
    // Events are copy handles; one sent in the move slot reads back as 0.
    const APPLET: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy_service), 0);
    let proxy = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 20); // IApplicationFunctions
    let functions = cpu.mem.read_u32(tls + 0x30).unwrap();

    // GetGpuErrorDetectedSystemEvent.
    ipc_request(&mut cpu, APPLET, 4, Some(functions), 130);
    // { send_pid:1, num_copy:4, num_move:4 }: one copy handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0, "the guest must receive a real handle");

    // Unfired, so a poll times out.
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    let (result, _) = wait_sync(&mut cpu, &[event], 0);
    assert_eq!(result, RESULT_TIMED_OUT);

    // A second, unsignalled event so the index below is a real position.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0); // ICommonStateGetter
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(state_getter), 13);
    let quiet = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(quiet, event);

    // Auto-clear: once signalled it reports its index, then times out again.
    cpu.signal_event(u64::from(event));
    let (result, index) = wait_sync(&mut cpu, &[quiet, event], 0);
    assert_eq!(result, 0);
    assert_eq!(index, 1, "the index of the handle that fired, not a count");
    let (result, _) = wait_sync(&mut cpu, &[event], 0);
    assert_eq!(result, RESULT_TIMED_OUT);

    // Handles not modelled as events are treated as ready.
    let (result, index) = wait_sync(&mut cpu, &[0x1234], 0);
    assert_eq!(result, 0);
    assert_eq!(index, 0);
}

#[test]
fn control_clone_hands_back_a_working_session() {
    // CloneCurrentObject (control 2) must return a new session as a move handle;
    // nnSdk clones fsp-srv before mounting. The handle is clear of `alloc_handle`'s range.
    const FS: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(FS, "fsp-srv");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, FS, 5, None, 0);
    let object = cpu.mem.read_u32(tls + 0x20).unwrap();

    ipc_request(&mut cpu, FS, 5, None, 2); // CloneCurrentObject
    assert_eq!(cpu.read_x(0), 0);
    // Move handles follow the 8-byte hipc header and a descriptor word.
    let clone = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(clone, 0, "clone must hand back a real handle, not 0");
    assert_ne!(clone, FS, "the clone is a separate session");

    let handles = cpu.service_handles_snapshot();
    assert!(handles
        .iter()
        .any(|(h, name)| *h == clone && name == "fsp-srv"));
    ipc_request(&mut cpu, clone, 4, Some(object), 1); // SetCurrentProcess
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
}

#[test]
fn a_fabricated_reply_fills_both_handle_slots() {
    // An unimplemented command replies with both a move-handle object and a
    // copy-handle event, since the intended out type is unknown and a missing
    // handle parses as 0.
    const NCM: u64 = 0x9200;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(NCM, "ncm");
    let tls = cpu.tls_base();

    // IContentManager::OpenContentStorage(StorageId) -> IContentStorage.
    ipc_request_plain(&mut cpu, NCM, 4, &[1, 0, 0, 0]);
    // { send_pid:1, num_copy:4, num_move:4 }: one of each. Copy handles come
    // first (event +0x0c, object +0x10), then the raw section at the next 16 bytes.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), (1 << 1) | (1 << 5));
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "OpenContentStorage failed"
    );
    let event = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    let storage = u64::from(cpu.mem.read_u32(tls + 0x10).unwrap());
    assert_ne!(
        storage, 0,
        "a success with no object is worse than a failure"
    );
    assert_ne!(
        event, 0,
        "a success with no event is the same bug in the other slot"
    );
    assert_ne!(event, storage);

    // The sub-session dispatches to the same service.
    let handles = cpu.service_handles_snapshot();
    assert!(handles
        .iter()
        .any(|(h, name)| *h == storage && name == "ncm"));

    // The event exists and never fires.
    assert_eq!(wait_sync(&mut cpu, &[event as u32], 0).0, 0xEA01);

    // Asked again, the same pair comes back.
    ipc_request_plain(&mut cpu, NCM, 4, &[1, 0, 0, 0]);
    assert_eq!(u64::from(cpu.mem.read_u32(tls + 0x10).unwrap()), storage);
    assert_eq!(u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap()), event);
}

#[test]
fn a_service_with_no_stub_still_answers_its_control_commands() {
    // A service without its own stub must still answer control commands; a
    // fabricated pointer buffer size makes callers use pointer buffers.
    const NIFM: u64 = 0xD000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(NIFM, "btm:sys");
    let tls = cpu.tls_base();

    for msg_type in [5u32, 7] {
        build_ipc_request(&mut cpu, msg_type, None, 3); // QueryPointerBufferSize
        run_ipc_request(&mut cpu, NIFM);
        assert_eq!(
            cpu.mem.read_u32(tls + 0x18).unwrap(),
            0,
            "type {msg_type} refused"
        );
        assert_eq!(
            cpu.mem.read_u16(tls + 0x20).unwrap(),
            POINTER_BUFFER_SIZE,
            "type {msg_type}: not a size"
        );
        // No handle comes with a size.
        assert_eq!(
            cpu.mem.read_u32(tls + 0x04).unwrap() >> 31,
            0,
            "type {msg_type}: handles"
        );
    }

    // ConvertToDomain does answer with an object id.
    build_ipc_request(&mut cpu, 5, None, 0);
    run_ipc_request(&mut cpu, NIFM);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    let object = cpu.mem.read_u32(tls + 0x20).unwrap();
    assert_ne!(object, 0, "no domain object came back");
    assert_eq!(
        cpu.domain_interface_name(NIFM, object).as_deref(),
        Some("btm:sys")
    );
}

#[test]
fn closing_a_domain_object_is_not_command_zero() {
    // `CmifDomainRequestType_Close` has no command id; it must not dispatch as command 0.
    const FS: u64 = 0xC000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(FS, "fsp-srv");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, FS, 5, None, 0);
    let root = cpu.mem.read_u32(tls + 0x20).unwrap();
    cpu.set_romfs(vec![0xAB; 0x400]);
    ipc_request(&mut cpu, FS, 4, Some(root), 200);
    let storage = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(storage, 0, "no IStorage came back");
    assert_eq!(
        cpu.domain_interface_name(FS, storage),
        Some("fsp-srv-storage".to_owned())
    );

    // A close: domain header type byte 2, no CmifInHeader.
    for i in (0..0x100u32).step_by(4) {
        cpu.mem.write_u32(tls + i, 0).unwrap();
    }
    cpu.mem.write_u32(tls, 4).unwrap();
    cpu.mem.write_u32(tls + 4, 8).unwrap();
    cpu.mem.write_u32(tls + 0x10, 2).unwrap(); // CmifDomainRequestType_Close
    cpu.mem.write_u32(tls + 0x14, storage).unwrap();
    run_ipc_request(&mut cpu, FS);

    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "the close was refused"
    );
    assert_eq!(
        cpu.domain_interface_name(FS, storage),
        None,
        "the object is still open after being closed"
    );
}

#[test]
fn reset_signal_reports_whether_the_event_had_fired() {
    // `svcResetSignal` clears a signalled event and fails if there was nothing to clear.
    const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
    let (mut cpu, applet, _proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();

    // The applet message event starts signalled with the startup focus change.
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 0); // GetEventHandle
    assert_eq!(
        cpu.mem.read_u32(tls + 0x08).unwrap(),
        1 << 1,
        "events are copy handles"
    );
    let message = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_eq!(
        reset_signal(&mut cpu, message),
        0,
        "the queued message did not announce itself"
    );
    assert_eq!(
        reset_signal(&mut cpu, message),
        RESULT_INVALID_STATE,
        "it announced itself twice"
    );
}

#[test]
fn an_unfilled_out_parameter_reads_as_zero_not_as_the_request() {
    // A reply overwrites the request in TLS with four words of padding, so a
    // bare success must not leak stale request bytes.
    const VI: u64 = 0xB200;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    // CloseDisplay is unimplemented; poison the out-parameter bytes first.
    build_ipc_request(&mut cpu, 4, None, 1020);
    for i in 0..4u32 {
        cpu.mem.write_u32(tls + 0x20 + i * 4, 0xDEAD_BEEF).unwrap();
    }
    run_ipc_request(&mut cpu, display);

    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "CloseDisplay refused"
    );
    for i in 0..4u32 {
        assert_eq!(
            cpu.mem.read_u32(tls + 0x20 + i * 4).unwrap(),
            0,
            "word {i} of the reply is a leftover of the request"
        );
    }
}
