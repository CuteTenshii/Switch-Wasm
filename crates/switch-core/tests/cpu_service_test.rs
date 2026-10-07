//! The Horizon services, reached over IPC: `hid`, `am`, `vi`, the audio pair,
//! `ldr:ro`, `hwopus` and the rest.

mod cpu;

use cpu::*;
use switch_core::cpu::POINTER_BUFFER_SIZE;

#[test]
fn ssl_keeps_context_state_and_refuses_connections() {
    // ssl contexts and their options are real; connections are not (no sockets).
    const SSL: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(SSL, "ssl");
    let tls = cpu.tls_base();
    ipc_request(&mut cpu, SSL, 5, None, 0); // ConvertToDomain
    let service = cpu.mem.read_u32(tls + 0x20).unwrap();

    // SetInterfaceVersion: nnSdk initialises ssl at startup.
    ipc_request_with_payload(&mut cpu, SSL, service, 5, &4u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // CreateContext -> ISslContext, and the count follows it.
    ipc_request(&mut cpu, SSL, 4, Some(service), 1);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0);
    ipc_request(&mut cpu, SSL, 4, Some(service), 0);
    let context = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(context, service);
    ipc_request(&mut cpu, SSL, 4, Some(service), 1);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);

    let mut args = Vec::new();
    args.extend_from_slice(&2u32.to_le_bytes()); // option
    args.extend_from_slice(&1u32.to_le_bytes()); // value
    ipc_request_with_payload(&mut cpu, SSL, context, 0, &args);
    ipc_request_with_payload(&mut cpu, SSL, context, 1, &2u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);
    // An option never set reads as 0 rather than as another option's value.
    ipc_request_with_payload(&mut cpu, SSL, context, 1, &7u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0);

    // CreateConnection is refused rather than returning a connection that cannot connect.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, SSL, 4, Some(context), 2);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn hid_hands_over_the_input_shared_memory() {
    // nnSdk calls methods on the IAppletResource, so it must be a real object.
    let (mut cpu, hid, server) = hid_server();
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, hid, 4, Some(server), 0); // CreateAppletResource
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    let resource = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(resource, server);

    // GetSharedMemoryHandle -> a copy handle, not a move handle.
    ipc_request(&mut cpu, hid, 4, Some(resource), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    assert_ne!(cpu.mem.read_u32(tls + 0x0c).unwrap(), 0);

    // nn::hid::SetSupportedNpadIdType needs a non-zero pointer buffer size.
    ipc_request(&mut cpu, hid, 5, None, 3);
    assert_eq!(cpu.mem.read_u16(tls + 0x20).unwrap(), POINTER_BUFFER_SIZE);
}

#[test]
fn hid_reads_back_what_the_guest_configured() {
    // A style set must read back as it was set.
    let (mut cpu, hid, server) = hid_server();
    let tls = cpu.tls_base();
    const STYLE_SET: u32 = 0b1101;

    ipc_request_with_payload(&mut cpu, hid, server, 100, &STYLE_SET.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    ipc_request(&mut cpu, hid, 4, Some(server), 101);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), STYLE_SET);

    // Set/GetNpadJoyHoldType: the hold type follows the aruid.
    let mut args = [0u8; 16];
    args[8..].copy_from_slice(&1u64.to_le_bytes());
    ipc_request_with_payload(&mut cpu, hid, server, 120, &args);
    ipc_request(&mut cpu, hid, 4, Some(server), 121);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), 1);
}

#[test]
fn hid_vibration_reaches_the_host() {
    // SendVibrationValue(handle, HidVibrationValue, aruid): band amplitudes at +4 and +0xc.
    let (mut cpu, hid, server) = hid_server();
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&0u32.to_le_bytes()); // device handle
    args.extend_from_slice(&0.75f32.to_bits().to_le_bytes()); // amp_low
    args.extend_from_slice(&160.0f32.to_bits().to_le_bytes()); // freq_low
    args.extend_from_slice(&0.25f32.to_bits().to_le_bytes()); // amp_high
    args.extend_from_slice(&320.0f32.to_bits().to_le_bytes()); // freq_high
    ipc_request_with_payload(&mut cpu, hid, server, 201, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.vibration(), (0.75, 0.25));

    // GetActualVibrationValue reports what is playing.
    ipc_request(&mut cpu, hid, 4, Some(server), 202);
    assert_eq!(f32::from_bits(cpu.mem.read_u32(tls + 0x30).unwrap()), 0.75);
    assert_eq!(f32::from_bits(cpu.mem.read_u32(tls + 0x38).unwrap()), 0.25);

    // Out-of-range or non-finite values are clamped.
    let mut args = vec![0u8; 4];
    args.extend_from_slice(&5.0f32.to_bits().to_le_bytes());
    args.extend_from_slice(&0u32.to_le_bytes());
    args.extend_from_slice(&f32::NAN.to_bits().to_le_bytes());
    args.extend_from_slice(&0u32.to_le_bytes());
    ipc_request_with_payload(&mut cpu, hid, server, 201, &args);
    assert_eq!(cpu.vibration(), (1.0, 0.0));
}

#[test]
fn hid_sys_is_its_own_interface_and_answers_before_any_command() {
    // libnx's hidsysInitialize queries the pointer buffer size on open, which
    // must not be answered with a fabricated object id.
    const HIDSYS: u64 = 0x9100;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HIDSYS, "hid:sys");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, HIDSYS, 5, None, 3); // QueryPointerBufferSize
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(cpu.mem.read_u16(tls + 0x20).unwrap(), POINTER_BUFFER_SIZE);

    ipc_request(&mut cpu, HIDSYS, 5, None, 0); // ConvertToDomain
    let server = cpu.mem.read_u32(tls + 0x20).unwrap();

    // EnableAppletToGetInput.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 503);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // GetMaskedSupportedNpadStyleSet(u64 aruid) -> NpadStyleSet: what the system
    // permits, including handheld, regardless of SetSupportedNpadStyleSet.
    ipc_request_with_payload(&mut cpu, HIDSYS, server, 310, &[0u8; 8]);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    let styles = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(styles & (1 << 1), 0, "handheld is not supported");
    assert_ne!(styles & (1 << 0), 0, "a full-key pad is not supported");

    // SetNpadSystemExtStateEnabled(bool, u64 aruid).
    let mut args = [0u8; 0x10];
    args[0] = 1;
    ipc_request_with_payload(&mut cpu, HIDSYS, server, 322, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // IsJoyConRailEnabled / IsJoyConAttachedOnAllRail: true in handheld mode.
    for cmd in [523u32, 525] {
        ipc_request(&mut cpu, HIDSYS, 4, Some(server), cmd);
        assert_eq!(
            cpu.mem.read_u32(tls + 0x28).unwrap(),
            0,
            "cmd {cmd} refused"
        );
        assert_eq!(cpu.mem.read_u8(tls + 0x30).unwrap(), 1, "cmd {cmd}");
    }

    // SetFirmwareHotfixUpdateSkipEnabled(bool): a refusal is fatal.
    ipc_request_with_payload(&mut cpu, HIDSYS, server, 1120, &[1u8, 0, 0, 0]);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // GetUniquePadIds -> 0: the built-in handheld pad is not detachable.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 703);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), 0);

    // AcquireHomeButtonEventHandle -> a copy handle, never signalled.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 101);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    assert_ne!(cpu.mem.read_u32(tls + 0x0c).unwrap(), 0);

    // A domain conversion must not turn hid:sys into IHidServer.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

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
fn storage_read_uses_the_istorage_field_layout() {
    // IStorage::Read is (s64 offset, u64 size), unlike IFile::Read.
    const FS: u64 = 0x1000;
    const OUT: u32 = 0x6000;
    let romfs: Vec<u8> = (0..64u8).collect();
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.set_romfs(romfs.clone());
    cpu.register_service_handle(FS, "fsp-srv-storage");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&4u64.to_le_bytes()); // offset
    args.extend_from_slice(&8u64.to_le_bytes()); // size
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 16, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    for i in 0..8u32 {
        assert_eq!(
            cpu.mem.read_u8(OUT + i).unwrap(),
            romfs[4 + i as usize],
            "byte {i}"
        );
    }
    assert_eq!(cpu.mem.read_u8(OUT + 8).unwrap(), 0);

    // A read past the end is refused with 2002-3005, not clamped.
    const OUT_OF_RANGE: u32 = 2 | (3005 << 9);
    cpu.mem.write_u8(OUT, 0xAA).unwrap();
    let mut args = Vec::new();
    args.extend_from_slice(&(romfs.len() as u64 - 2).to_le_bytes());
    args.extend_from_slice(&64u64.to_le_bytes());
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 64, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), OUT_OF_RANGE);
    assert_eq!(cpu.mem.read_u8(OUT).unwrap(), 0xAA, "buffer left alone");

    let mut args = Vec::new();
    args.extend_from_slice(&(romfs.len() as u64 - 2).to_le_bytes());
    args.extend_from_slice(&2u64.to_le_bytes());
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 64, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(OUT).unwrap(), romfs[romfs.len() - 2]);

    ipc_request(&mut cpu, FS, 4, Some(1), 4);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), romfs.len() as u64);
}

#[test]
fn lm_writes_the_guests_own_log_to_the_console() {
    const LM: u64 = 0x1000;
    const PACKET: u32 = 0x5000;
    const KEY_TEXT: u8 = 2;
    const KEY_MODULE: u8 = 6;
    const HEAD: u8 = 1;
    const TAIL: u8 = 2;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(LM, "lm");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, LM, 5, None, 0); // Control::ConvertToDomain
    let service = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, LM, 4, Some(service), 0); // OpenLogger
    let logger = cpu.mem.read_u32(tls + 0x30).unwrap();

    // One packet: severity 3 is Error, module from key 6, text from key 2.
    let len = write_log_packet(
        &mut cpu,
        PACKET,
        HEAD | TAIL,
        3,
        &[(KEY_MODULE, b"Game"), (KEY_TEXT, b"hello world")],
    );
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(
        String::from_utf8_lossy(&cpu.out),
        "[lm/ERROR/Game] hello world\n"
    );

    // A message split across packets joins into one line.
    cpu.out.clear();
    let len = write_log_packet(&mut cpu, PACKET, HEAD, 1, &[(KEY_TEXT, b"split ")]);
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    let len = write_log_packet(&mut cpu, PACKET, TAIL, 1, &[(KEY_TEXT, b"message")]);
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(
        String::from_utf8_lossy(&cpu.out),
        "[lm/INFO] split message\n"
    );

    // A packet's claimed length is trusted only up to the buffer size.
    cpu.out.clear();
    let len = write_log_packet(
        &mut cpu,
        PACKET,
        HEAD | TAIL,
        0,
        &[(KEY_TEXT, b"truncated")],
    );
    cpu.mem.write_u32(PACKET + 0x14, 0xFFFF).unwrap();
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(String::from_utf8_lossy(&cpu.out), "[lm/TRACE] truncated\n");
}

#[test]
fn fatal_service_keeps_the_result_after_its_trace_is_drained() {
    const FATAL: u64 = 0x1000;
    const RESULT: u32 = 0x0000_7201;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(FATAL, "fatal:u");

    ipc_request_with_payload(&mut cpu, FATAL, 0, 2, &RESULT.to_le_bytes());
    cpu.trace.clear();

    let fatal = cpu.guest_fatal().expect("fatal result was not retained");
    assert!(fatal.contains("0x00007201 = 1-0057 (cmd Some(2))"));
}

#[test]
fn pctl_reports_parental_controls_off() {
    const PCTL: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(PCTL, "pctl");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, PCTL, 5, None, 0);
    let factory = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, PCTL, 4, Some(factory), 1);
    let service = cpu.mem.read_u32(tls + 0x30).unwrap();

    // Permission checks: success means permitted.
    for cmd in [1001u32, 1004, 1013, 1017] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
    }

    // The two query families read in opposite senses.
    for cmd in [1031u32, 1010, 1453, 1455] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
        assert_eq!(
            cpu.mem.read_u8(tls + 0x30).unwrap(),
            0,
            "cmd {cmd} restricted"
        );
    }
    for cmd in [1018u32, 1065] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
        assert_eq!(cpu.mem.read_u8(tls + 0x30).unwrap(), 1, "cmd {cmd} allowed");
    }

    // GenerateInquiryCode -> ten digits NUL-padded to 0x20.
    ipc_request(&mut cpu, PCTL, 4, Some(service), 1204);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd 1204 refused");
    let code: Vec<u8> = (0..0x20)
        .map(|i| cpu.mem.read_u8(tls + 0x30 + i).unwrap())
        .collect();
    assert!(
        code[..10].iter().all(u8::is_ascii_digit) && code[10..].iter().all(|&b| b == 0),
        "inquiry code is not ten digits in a 0x20 block: {code:?}"
    );

    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, PCTL, 4, Some(service), 1203); // SetPinCode
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn applet_common_state_getter_reports_focus_once() {
    // ReceiveMessage (cmd 1) hands out the startup FocusStateChanged (15) once,
    // then "no message", not AM_BUSY or a repeated focus change.
    let (mut cpu, handle, _proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();

    // The message event fires for one poll, then clears.
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 0); // GetEventHandle
    let message = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_eq!(wait_sync(&mut cpu, &[message], 0).0, 0, "never announced");
    assert_eq!(
        wait_sync(&mut cpu, &[message], 0).0,
        RESULT_TIMED_OUT,
        "it announced itself twice"
    );

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    assert_eq!(cpu.read_x(0), 0); // svc result
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0); // Result: success
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 15); // FocusStateChanged

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    const NO_MESSAGES: u32 = 128 | (3 << 9);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), NO_MESSAGES);

    // GetCurrentFocusState (cmd 9) reports InFocus.
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 9);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);
}

#[test]
fn applet_unimplemented_command_is_an_error_not_a_fake_success() {
    // Unimplemented `am` commands report "unknown command id", not success.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    let (mut cpu, handle, proxy, _state_getter) = applet_chain();
    let tls = cpu.tls_base();

    // GetDisplayController, then 10 (AcquireLastApplicationCaptureBuffer), unimplemented.
    ipc_request(&mut cpu, handle, 4, Some(proxy), 4);
    let display_controller = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, handle, 4, Some(display_controller), 10);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn gamepad_input_writes_input_reg_and_hid_shmem() {
    // MapSharedMemory (svc 0x13) of a hid-sized region; set_gamepad_state
    // mirrors the pad into INPUT_ADDR and the npad LIFOs (npad at 0x9A00,
    // 0x5000 per controller, `full_key_lifo` at +0x28, `handheld_lifo` at +0x378).
    const SHMEM: u32 = 0x3000_0000;
    const NPAD: u32 = SHMEM + 0x9A00;
    const HANDHELD: u32 = NPAD + 8 * 0x5000;
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, SHMEM as u64);
    cpu.set_reg(2, 0x40000);
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);

    cpu.set_gamepad_state(0x3, -1000, 30000, 0, 0);

    // StickLUp (1 << 17) is added; the small horizontal deflection is below threshold.
    let expected_buttons = 0x3 | (1 << 17);
    assert_eq!(
        cpu.mem.read_u64(switch_core::INPUT_ADDR).unwrap(),
        expected_buttons
    );

    for (base, lifo_off, style, device) in [
        (NPAD, 0x28, 1 << 0, 1 << 0), // player 1, Pro Controller
        (HANDHELD, 0x378, 1 << 1, (1 << 2) | (1 << 3)), // handheld
    ] {
        assert_eq!(cpu.mem.read_u32(base).unwrap(), style, "style_set");
        assert_eq!(
            cpu.mem.read_u32(base + 0x4188).unwrap(),
            device,
            "device_type"
        );
        let lifo = base + lifo_off;
        assert_eq!(cpu.mem.read_u64(lifo + 0x08).unwrap(), 17, "buffer_count");
        assert_eq!(cpu.mem.read_u64(lifo + 0x10).unwrap(), 0, "tail");
        assert_eq!(cpu.mem.read_u64(lifo + 0x18).unwrap(), 1, "count");
        let entry = lifo + 0x20;
        let sample = cpu.mem.read_u64(entry).unwrap();
        assert!(sample > 0, "sampling number must advance");
        // Bit 0 of the storage number is the seqlock flag, so it is doubled.
        assert_eq!(cpu.mem.read_u64(entry + 0x08).unwrap() * 2, sample);
        assert_eq!(cpu.mem.read_u64(entry + 0x10).unwrap(), expected_buttons);
        assert_eq!(
            cpu.mem.read_u32(entry + 0x18).unwrap(),
            1000u32.wrapping_neg()
        );
        assert_eq!(cpu.mem.read_u32(entry + 0x1C).unwrap(), 30000);
        assert_eq!(cpu.mem.read_u32(entry + 0x28).unwrap() & 1, 1);

        // Power info after `system_button_properties`: full batteries.
        for info in 0..3u32 {
            let level = cpu.mem.read_u32(base + 0x4198 + info * 4).unwrap();
            assert_eq!(level, 4, "battery_level[{info}]");
        }
        // PowerConnected bits set, Charging bits clear.
        let properties = cpu.mem.read_u32(base + 0x4190).unwrap();
        assert_eq!(properties & 0x38, 0x38, "PowerConnected");
        assert_eq!(properties & 0x7, 0, "Charging");
    }
}

#[test]
fn touch_input_writes_the_hid_touchscreen_lifo() {
    // `HidSharedMemory.touch_screen` at 0x400: a LIFO of `{u64 sampling_number,
    // HidTouchScreenState}`; each 0x28-byte `HidTouchState` has finger_id at
    // +0x0C, x at +0x10 and y at +0x14.
    use switch_core::cpu::TouchPoint;
    const SHMEM: u32 = 0x3000_0000;
    const LIFO: u32 = SHMEM + 0x400;
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, SHMEM as u64);
    cpu.set_reg(2, 0x40000);
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);

    cpu.set_touch_state(&[
        TouchPoint {
            finger_id: 0,
            x: 640,
            y: 360,
        },
        TouchPoint {
            finger_id: 3,
            x: 100,
            y: 700,
        },
    ]);

    assert_eq!(cpu.mem.read_u64(LIFO + 0x08).unwrap(), 17, "buffer_count");
    assert_eq!(cpu.mem.read_u64(LIFO + 0x10).unwrap(), 0, "tail");
    assert_eq!(cpu.mem.read_u64(LIFO + 0x18).unwrap(), 1, "count");

    let storage = LIFO + 0x20;
    let sample = cpu.mem.read_u64(storage).unwrap();
    assert!(sample > 0, "sampling number must advance");
    let state = storage + 8;
    // The storage number is the state's doubled.
    assert_eq!(
        cpu.mem.read_u64(state).unwrap() * 2,
        sample,
        "state sampling number"
    );
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 2, "contact count");

    let touch = |i: u32| state + 0x10 + i * 0x28;
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x0C).unwrap(), 0, "finger_id");
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x10).unwrap(), 640, "x");
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x14).unwrap(), 360, "y");
    assert!(cpu.mem.read_u32(touch(0) + 0x18).unwrap() > 0, "diameter_x");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x0C).unwrap(), 3, "finger_id");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x10).unwrap(), 100, "x");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x14).unwrap(), 700, "y");

    // New contacts carry `start_touch`; UIs tap on that transition.
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x08).unwrap(), 1, "start 0");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x08).unwrap(), 1, "start 3");

    // A lifted finger is published once more with `end_touch`; the held one's
    // attributes return to zero.
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 5,
        y: 6,
    }]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 2, "contact count");
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x08).unwrap(), 0, "held");
    assert_eq!(
        cpu.mem.read_u32(touch(1) + 0x0C).unwrap(),
        3,
        "the lifted id"
    );
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x08).unwrap(), 2, "end");
    assert!(
        cpu.mem.read_u64(storage).unwrap() > sample,
        "sample must advance"
    );

    // Then it is gone and its slot cleared.
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 5,
        y: 6,
    }]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 1, "contact count");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x10).unwrap(), 0, "vacated x");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x14).unwrap(), 0, "vacated y");

    // A full lift is published, not silent.
    cpu.set_touch_state(&[]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 1, "the end sample");
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x08).unwrap(), 2, "end");
    cpu.set_touch_state(&[]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 0, "contact count");

    // Coordinates clamp to the digitizer; slots to sixteen.
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 99_999,
        y: 99_999,
    }]);
    assert_eq!(
        cpu.mem.read_u32(touch(0) + 0x10).unwrap(),
        1279,
        "clamped x"
    );
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x14).unwrap(), 719, "clamped y");
}

#[test]
fn mapping_pl_shared_memory_delivers_the_shared_font() {
    use switch_core::cpu::PL_SHMEM_SIZE;
    // pl's fonts must be in shared memory when the mapping returns, each behind
    // an eight-byte header.
    const ADDR: u32 = 0x2000_0000;
    const HEADER: u32 = 8;
    let font: Vec<u8> = (0..=255u8).cycle().take(0x2000).collect();
    let mut cpu = cpu_at(0x1000);
    cpu.set_shared_font(font.clone());
    cpu.set_reg(1, ADDR as u64);
    cpu.set_reg(2, u64::from(PL_SHMEM_SIZE));
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);
    assert_eq!(cpu.mem.dump(ADDR + HEADER, font.len()).unwrap(), font);

    // A font set after mapping still reaches the guest.
    let replacement: Vec<u8> = vec![0xAB; 0x1000];
    cpu.set_shared_font(replacement.clone());
    assert_eq!(
        cpu.mem.dump(ADDR + HEADER, replacement.len()).unwrap(),
        replacement
    );
}

#[test]
fn caps_a_reports_a_mounted_empty_album() {
    // The Album applet's startup queries: an empty, mounted album with no SD auto-save.
    const CAPS: u64 = 0xCA00;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(CAPS, "caps:a");
    let tls = cpu.tls_base();

    // Unknown18 -> bytes written into the caller's buffer.
    ipc_request_plain(&mut cpu, CAPS, 18, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "Unknown18 failed");
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "claimed to have written bytes"
    );

    // IsAlbumMounted(AlbumStorage::Nand) -> bool.
    ipc_request_plain(&mut cpu, CAPS, 5, &0u8.to_le_bytes());
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "IsAlbumMounted failed"
    );
    assert_eq!(
        cpu.mem.read_u8(tls + 0x20).unwrap(),
        1,
        "the album is not mounted"
    );

    // GetAutoSavingStorage -> bool.
    ipc_request_plain(&mut cpu, CAPS, 401, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "GetAutoSavingStorage failed"
    );
    assert_eq!(
        cpu.mem.read_u8(tls + 0x20).unwrap(),
        0,
        "captures are being auto-saved"
    );

    // The album is empty by both the count and the list.
    for cmd in [0u32, 1, 100, 101] {
        ipc_request_plain(&mut cpu, CAPS, cmd, &0u8.to_le_bytes());
        assert_eq!(
            cpu.mem.read_u32(tls + 0x18).unwrap(),
            0,
            "caps:a {cmd} failed"
        );
        assert_eq!(
            cpu.mem.read_u64(tls + 0x20).unwrap(),
            0,
            "caps:a {cmd} found a file"
        );
    }
}

#[test]
fn the_applet_capture_buffer_names_a_slot_nothing_renders_into() {
    // AcquireCallerAppletCaptureSharedBuffer with no caller: a black slot, not
    // slot -1, which `nnSdk` retries forever.
    const APPLET: u64 = 0xA1000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "am:display-controller");
    let tls = cpu.tls_base();

    for cmd in [22u32, 24, 26] {
        ipc_request_plain(&mut cpu, APPLET, cmd, &[]);
        assert_eq!(
            cpu.mem.read_u32(tls + 0x18).unwrap(),
            0,
            "acquire {cmd} failed"
        );
        assert_eq!(
            cpu.mem.read_u32(tls + 0x20).unwrap(),
            1,
            "acquire {cmd} wrote nothing"
        );
        // A slot past those `AcquireSharedFrameBuffer` hands out stays black.
        let slot = cpu.mem.read_u32(tls + 0x24).unwrap();
        assert!(
            (switch_core::cpu::SHARED_BUFFER_USABLE_SLOTS..switch_core::cpu::SHARED_BUFFER_SLOTS)
                .contains(&slot),
            "slot {slot} is not a spare one"
        );
    }
}

#[test]
fn the_capture_image_getters_clear_the_buffer_they_fill() {
    // The same black screen as pixels: a 1280x720 RGBA8888 image in a map-alias buffer.
    const APPLET: u64 = 0xA1000;
    const REGION: u32 = 0x20_0000;
    const ROOM: u32 = 0x4000;
    const START: u32 = REGION + 0x40; // unaligned, and spanning three pages
    const SIZE: u32 = 0x2800;
    const PATTERN: u32 = 0xDEAD_BEEF;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "am:display-controller");
    cpu.mem.map_zero(REGION, ROOM as usize).unwrap();
    let tls = cpu.tls_base();

    for cmd in [5u32, 6, 7] {
        for at in (REGION..REGION + ROOM).step_by(4) {
            cpu.mem.write_u32(at, PATTERN).unwrap();
        }
        ipc_request_plain_with_buffer(&mut cpu, APPLET, cmd, START, SIZE, true, &[]);
        assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "capture {cmd}");
        assert_eq!(
            cpu.mem.read_u8(tls + 0x20).unwrap(),
            1,
            "capture {cmd} wrote nothing"
        );
        assert_eq!(
            cpu.mem.dump(START, SIZE as usize).unwrap(),
            vec![0u8; SIZE as usize],
            "capture {cmd} left the buffer as it was"
        );
        // Nothing past the buffer is cleared.
        assert_eq!(
            cpu.mem.read_u32(START - 4).unwrap(),
            PATTERN,
            "capture {cmd} wrote before the buffer"
        );
        assert_eq!(
            cpu.mem.read_u32(START + SIZE).unwrap(),
            PATTERN,
            "capture {cmd} wrote past the buffer"
        );
    }
}

#[test]
fn the_caller_applet_stack_is_the_one_applet_above_this_one() {
    // GetCallerAppletIdentityInfoStack: just the menu, clamped to the buffer.
    const APPLET: u64 = 0xA3000;
    const STACK: u32 = 0x30_0000;
    const ENTRY: u32 = 0x10;
    const QLAUNCH_TITLE_ID: u64 = 0x0100_0000_0000_1000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "am:library-applet-self-accessor");
    cpu.mem.map_zero(STACK, 0x1000).unwrap();
    let tls = cpu.tls_base();

    ipc_request_plain_with_buffer(&mut cpu, APPLET, 17, STACK, 4 * ENTRY, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "refused");
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "entries written");
    assert_eq!(cpu.mem.read_u32(STACK).unwrap(), 3, "SystemAppletMenu");
    assert_eq!(cpu.mem.read_u64(STACK + 8).unwrap(), QLAUNCH_TITLE_ID);

    // No room for an entry gives a count of zero.
    ipc_request_plain_with_buffer(&mut cpu, APPLET, 17, STACK, ENTRY - 1, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "refused");
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0, "entries written");
}

#[test]
fn a_library_applet_is_told_which_keyboard_layout_to_open_with() {
    // GetDesirableKeyboardLayout with no caller: the layout for `set`'s en-US.
    const APPLET: u64 = 0xA2000;
    const ENGLISH_US: u32 = 1;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "am:library-applet-self-accessor");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, APPLET, 19, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "refused");
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), ENGLISH_US);
}

#[test]
fn audout_plays_the_buffers_the_guest_hands_it() {
    // `audout`'s buffer protocol: append, wait on the event, collect released tags.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000; // the AudioOutBuffer struct
    const PCM: u32 = 0x8100; // its samples
    const TAGS: u32 = 0x8200; // where released tags come back
    const TAG: u64 = 0xFEED_0001;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    // OpenAudioOut(48 kHz, stereo) -> { rate, channels, format, state } and an
    // IAudioOut move handle.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "OpenAudioOut failed"
    );
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 48_000);
    assert_eq!(cpu.mem.read_u32(tls + 0x24).unwrap(), 2);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 2, "PcmFormat::Int16");
    assert_eq!(
        cpu.mem.read_u32(tls + 0x2c).unwrap(),
        1,
        "a device opens stopped"
    );
    // { send_pid:1, num_copy:4, num_move:4 }: one move handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(device, 0, "no IAudioOut came back");

    // RegisterBufferEvent: a copy handle.
    ipc_request_plain(&mut cpu, device, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0);
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "event fired early"
    );

    // StartAudioOut, then hand over one buffer of four stereo frames.
    ipc_request_plain(&mut cpu, device, 1, &[]);
    ipc_request_plain(&mut cpu, device, 0, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0, "started");

    let samples: [i16; 8] = [1, -1, 2, -2, 3, -3, 4, -4];
    for (i, &s) in samples.iter().enumerate() {
        cpu.mem.write_u16(PCM + i as u32 * 2, s as u16).unwrap();
    }
    // AudioOutBuffer { next, buffer, buffer_size, data_size, data_offset }.
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "AppendAudioOutBuffer failed"
    );

    // The samples reach the host on arrival; only the tag waits for the device.
    let mut played = [0i16; 8];
    assert_eq!(cpu.take_audio(&mut played), 8);
    assert_eq!(played, samples);

    // Not released until played: four frames at 48 kHz is 85,000 cycles at 1.02 GHz.
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "released before it could play"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "a tag came back early"
    );

    // Spend the cycles with a branch-to-self; the clock is the instruction count.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0,
        "the played buffer did not fire"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);

    // GetAudioOutPlayedSampleCount counts frames, not samples.
    ipc_request_plain(&mut cpu, device, 10, &[]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 4);

    assert_eq!(cpu.audio_format(), (48_000, 2));

    // One device, started, one four-frame buffer in and out, eight samples.
    let activity = cpu.audio_activity();
    assert_eq!(
        (
            activity.sample_rate,
            activity.channels,
            activity.produced,
            activity.taken
        ),
        (48_000, 2, 8, 8)
    );
    assert_eq!((activity.dropped, activity.backlog), (0, 0));
    let [output] = activity.outputs.as_slice() else {
        panic!("{} devices reported", activity.outputs.len());
    };
    assert!(output.started);
    assert_eq!(
        (
            output.appended_buffers,
            output.appended_frames,
            output.released_buffers,
            output.pending_buffers,
            output.discarded_frames
        ),
        (1, 4, 1, 0, 0)
    );
}

#[test]
fn audout_release_zeroes_the_entry_after_the_last_tag() {
    // `nn::audio` reads the released tag from an uninitialised stack slot without
    // checking the count, so an empty release must write a zero terminator.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;
    const TAG: u64 = 0xFEED_0003;
    const GARBAGE: u64 = 0x0868_BBF8;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, device, 1, &[]); // StartAudioOut

    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());

    // Nothing has played, so the release is empty and terminated.
    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    cpu.mem.write_u64(TAGS + 8, GARBAGE).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "a tag came back early"
    );
    assert_eq!(
        cpu.mem.read_u64(TAGS).unwrap(),
        0,
        "the guest kept reading its own stack"
    );

    // Once played, the tag lands in the first slot and the zero after it.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    cpu.mem.write_u64(TAGS + 8, GARBAGE).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);
    assert_eq!(
        cpu.mem.read_u64(TAGS + 8).unwrap(),
        0,
        "no terminator after the last tag"
    );
}

#[test]
fn audout_release_answers_the_auto_commands_pointer_buffer() {
    // `GetReleasedAudioOutBufferAuto` offers a receive-static buffer and a null
    // map-alias descriptor; the reply must go to the former.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;
    const TAG: u64 = 0xFEED_0004;
    const GARBAGE: u64 = 0x0AA2_8F50;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, device, 1, &[]); // StartAudioOut

    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());

    // The terminator reaches the guest's slot, not address 0.
    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    ipc_request_auto_recv(&mut cpu, device, 8, TAGS, 16, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "a tag came back early"
    );
    assert_eq!(
        cpu.mem.read_u64(TAGS).unwrap(),
        0,
        "the pointer buffer was never written"
    );

    // And once played, so does the tag.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    cpu.mem.write_u64(TAGS + 8, GARBAGE).unwrap();
    ipc_request_auto_recv(&mut cpu, device, 8, TAGS, 16, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);
    assert_eq!(
        cpu.mem.read_u64(TAGS + 8).unwrap(),
        0,
        "no terminator after the last tag"
    );
}

#[test]
fn audren_update_reply_has_a_section_for_every_count_the_renderer_was_opened_with() {
    // `RequestUpdateAudioRenderer`'s reply is walked section by section against
    // caller-computed sizes, so every section must be present.
    const AUDREN: u64 = 0xB000;
    const OUT: u32 = 0x9000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDREN, "audren:u");
    let tls = cpu.tls_base();

    // `AudioRendererParameter`: voices +16, sinks +20, effects +24, revision +48.
    let renderer_with = |cpu: &mut Cpu, revision: &[u8; 4]| -> u64 {
        let mut params = vec![0u8; 52];
        params[16..20].copy_from_slice(&2u32.to_le_bytes());
        params[20..24].copy_from_slice(&1u32.to_le_bytes());
        params[24..28].copy_from_slice(&3u32.to_le_bytes());
        params[48..52].copy_from_slice(revision);
        ipc_request_plain(cpu, AUDREN, 0, &params);
        u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap())
    };
    let section = |cpu: &Cpu, at: u32| cpu.mem.read_u32(OUT + at).unwrap();

    let renderer = renderer_with(&mut cpu, b"REV9");
    assert_ne!(renderer, 0, "no IAudioRenderer came back");
    ipc_request_plain_with_buffer(&mut cpu, renderer, 4, OUT, 0x1000, true, &[]);

    // MemPoolInfoOut per mempool (effects + four per voice), VoiceInfoOut per
    // voice, revision-9 EffectOutStatus per effect, SinkInfoOut per sink, then
    // the performance, behaviour and renderer-info tails.
    assert_eq!(section(&cpu, 0x08), (3 + 4 * 2) * 16, "mempools");
    assert_eq!(section(&cpu, 0x0c), 2 * 16, "voices");
    assert_eq!(section(&cpu, 0x14), 3 * 0x90, "effects");
    assert_eq!(section(&cpu, 0x1c), 32, "sinks");
    assert_eq!(section(&cpu, 0x20), 16, "performance");
    assert_eq!(section(&cpu, 0x04), 176, "behaviour");
    assert_eq!(section(&cpu, 0x28), 16, "renderer info");
    let total = 64 + 176 + 32 + 3 * 0x90 + 32 + 16 + 176 + 16;
    assert_eq!(section(&cpu, 0x3c), total, "total size");

    // Before revision 5: no renderer info, and the narrow effect status.
    let renderer = renderer_with(&mut cpu, b"REV4");
    ipc_request_plain_with_buffer(&mut cpu, renderer, 4, OUT, 0x1000, true, &[]);
    assert_eq!(section(&cpu, 0x14), 3 * 16, "revision-4 effects");
    assert_eq!(section(&cpu, 0x28), 0, "revision-4 renderer info");
    assert_eq!(
        section(&cpu, 0x3c),
        64 + 176 + 32 + 3 * 16 + 32 + 16 + 176,
        "revision-4 total"
    );
}

#[test]
fn audren_mixes_a_voice_through_to_the_host() {
    // The renderer mixes wave buffers into PCM.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const FRAMES: u32 = 240;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    // A ramp, so a resampler off-by-one shows as a shift.
    let samples: Vec<i16> = (0..FRAMES).map(|i| (i as i16 - 120) * 100).collect();
    for (i, &s) in samples.iter().enumerate() {
        cpu.mem.write_u16(PCM + i as u32 * 2, s as u16).unwrap();
    }

    let mut update = AudrenUpdate::new(1, 1, 1);
    update.voice(0, PCM_INT16, 1, PCM, FRAMES * 2, FRAMES);
    update.route(0, 0, 1.0);
    update.route(0, 1, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    // One frame of emulated time renders exactly one frame.
    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; FRAMES as usize * 2];
    assert_eq!(
        cpu.take_audio(&mut played),
        played.len(),
        "the mix never reached the host"
    );
    assert_eq!(cpu.audio_format(), (48_000, 2));
    // Mono into both mix buffers and outputs: the source doubled, bit-exact.
    for (i, &s) in samples.iter().enumerate() {
        assert_eq!(played[i * 2], s, "left channel at sample {i}");
        assert_eq!(played[i * 2 + 1], s, "right channel at sample {i}");
    }

    // No time elapsed renders nothing further.
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);
    let mut again = [0i16; 2];
    assert_eq!(
        cpu.take_audio(&mut again),
        0,
        "a frame was rendered that no time had come due for"
    );
}

#[test]
fn audren_reports_the_wave_buffers_it_finished_with() {
    // `num_wavebufs_consumed` drives the guest's refills.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const FRAMES: u32 = 240;
    /// The reply's voice section: past the header and four mempools per voice.
    const VOICE_OUT: u32 = 64 + 4 * 16;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    for i in 0..FRAMES {
        cpu.mem.write_u16(PCM + i * 2, 0x1234).unwrap();
    }
    let mut update = AudrenUpdate::new(1, 1, 1);
    update.voice(0, PCM_INT16, 1, PCM, FRAMES * 2, FRAMES);
    update.route(0, 0, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    assert_eq!(
        cpu.mem.read_u64(OUT + VOICE_OUT).unwrap(),
        u64::from(FRAMES),
        "played sample count"
    );
    assert_eq!(
        cpu.mem.read_u32(OUT + VOICE_OUT + 8).unwrap(),
        1,
        "the wave buffer never came back"
    );
}

#[test]
fn audren_decodes_the_adpcm_a_retail_voice_is_encoded_in() {
    // Nintendo 4-bit ADPCM: 14 samples per 8 bytes, a header byte (shift and
    // predictor pair) then seven bytes of nibbles.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const DATA: u32 = 0x5_0000;
    const COEFS: u32 = 0x5_1000;
    const SAMPLES: u32 = 28;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    // Pair 0 predicts nothing; pair 1 is 1.0 in Q11, adding the previous sample.
    let coefficients: [i16; 16] = [0, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, &c) in coefficients.iter().enumerate() {
        cpu.mem.write_u16(COEFS + i as u32 * 2, c as u16).unwrap();
    }

    // Frame 0: pair 0, shift 0, nibbles 1..7 then -8..-2.
    // Frame 1: pair 1, shift 0, every nibble 1, a running +1 from -2.
    let data: [u8; 16] = [
        0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11,
    ];
    for (i, &b) in data.iter().enumerate() {
        cpu.mem.write_u8(DATA + i as u32, b).unwrap();
    }
    let expected: [i16; 28] = [
        1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12,
    ];

    let mut update = AudrenUpdate::new(1, 1, 1);
    update.voice(0, PCM_ADPCM, 1, DATA, data.len() as u32, SAMPLES);
    update.extra_params(0, COEFS, 32);
    update.route(0, 0, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; 240 * 2];
    assert_eq!(cpu.take_audio(&mut played), played.len());
    for (i, &want) in expected.iter().enumerate() {
        assert_eq!(played[i * 2], want, "ADPCM sample {i}");
    }
    // Past the end the voice interpolates to silence rather than holding.
    assert_eq!(
        played[expected.len() * 2],
        0,
        "the voice kept playing past its data"
    );
}

#[test]
fn audren_frame_event_fires_on_the_clock() {
    // The renderer event paces `audrenWaitFrame` and must be a real event.
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);
    let tls = cpu.tls_base();

    // QuerySystemEvent -> a copy handle.
    ipc_request_plain(&mut cpu, renderer, 7, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x08).unwrap(),
        1 << 1,
        "not a copy handle"
    );
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0, "no frame event came back");

    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "the frame event fired early"
    );

    // Five milliseconds later, it is.
    cpu.cycles += AUDREN_FRAME_CYCLES;
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0,
        "the frame event never fired"
    );
}

#[test]
fn audren_refuses_a_wave_buffer_that_is_outside_its_allocation() {
    // Where `end_sample_offset` exceeds the allocation, the allocation wins;
    // the buffer is still consumed.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const VOICE_OUT: u32 = 64 + 4 * 16;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    for i in 0..240 {
        cpu.mem.write_u16(PCM + i * 2, 0x7FFF).unwrap();
    }
    let mut update = AudrenUpdate::new(1, 1, 1);
    // 240 samples claimed from a buffer with room for none.
    update.voice(0, PCM_INT16, 1, PCM, 0, 240);
    update.route(0, 0, 1.0);
    update.route(0, 1, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; 240 * 2];
    assert_eq!(
        cpu.take_audio(&mut played),
        played.len(),
        "the sink stopped producing frames"
    );
    assert!(
        played.iter().all(|&s| s == 0),
        "unplayable samples reached the host"
    );
    assert_eq!(
        cpu.mem.read_u32(OUT + VOICE_OUT + 8).unwrap(),
        1,
        "the buffer never came back"
    );
}

#[test]
fn audout_refuses_a_buffer_whose_samples_are_outside_it() {
    // `data_offset + data_size` must fit inside `buffer_size` (the Mii editor's
    // does not). The buffer still comes back; only its samples are dropped.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;
    const TAG: u64 = 0xFEED_0002;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, device, 1, &[]); // StartAudioOut

    // Playable data where the bad arithmetic would land, so a missing check is audible.
    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    // buffer_size is 8 bytes; data_offset alone is past it.
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 8).unwrap();
    cpu.mem.write_u64(DESC + 24, 8).unwrap();
    cpu.mem.write_u64(DESC + 32, 16).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 7, DESC, 40, false, &TAG.to_le_bytes());
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "the append is still accepted"
    );

    let mut played = [0i16; 8];
    assert_eq!(
        cpu.take_audio(&mut played),
        0,
        "unplayable samples reached the host"
    );

    // The guest still gets its buffer back.
    ipc_request_plain_with_buffer(&mut cpu, device, 8, TAGS, 16, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        1,
        "the buffer was never released"
    );
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);
}

#[test]
fn audout_reads_the_channel_count_as_sixteen_bits() {
    // `OpenAudioOut`'s channel count is 16 bits; the upper two bytes are padding.
    const AUDOUT: u64 = 0xA000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&0u32.to_le_bytes()); // sample rate: device default
    args.extend_from_slice(&0xcafe_0002u32.to_le_bytes()); // stereo, plus junk
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        48_000,
        "device default rate"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x24).unwrap(),
        2,
        "the padding leaked through"
    );
}

#[test]
fn audout_does_not_play_a_stopped_device() {
    // An unstarted device returns buffers but queues nothing for the host.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    cpu.mem.write_u16(PCM, 0x1234).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 2).unwrap();
    cpu.mem.write_u64(DESC + 24, 2).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &7u64.to_le_bytes());

    let mut played = [0i16; 4];
    assert_eq!(
        cpu.take_audio(&mut played),
        0,
        "a stopped device played something"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "released count");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), 7, "the tag");
}

#[test]
fn the_binder_transacts_on_the_command_a_pre_3_0_0_sdk_sends() {
    // `TransactParcel` (0, map-alias) and `TransactParcelAuto` (3, auto-select,
    // 3.0.0+) must answer identically; older SDKs send only 0.
    const VI: u64 = 0xB800;
    const PARCEL: u32 = 0x9000;
    const REPLY: u32 = 0x9400;
    /// `NATIVE_WINDOW_WIDTH`.
    const QUERY_WIDTH: u32 = 0;
    const QUERY: u32 = 9;

    for cmd in [0u32, 3] {
        let mut cpu = cpu_at(0x1000);
        cpu.bootstrap();
        cpu.set_pc(0x1000);
        cpu.register_service_handle(VI, "vi:m");
        let tls = cpu.tls_base();

        // vi root -> IApplicationDisplayService (2) -> IHOSBinderDriver (100).
        ipc_request_plain(&mut cpu, VI, 2, &[]);
        let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
        assert_ne!(display, 0, "no IApplicationDisplayService");
        ipc_request_plain(&mut cpu, display, 100, &[]);
        let relay = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
        assert_ne!(relay, 0, "no IHOSBinderDriver");

        let parcel = binder_parcel(&QUERY_WIDTH.to_le_bytes());
        for (i, &b) in parcel.iter().enumerate() {
            cpu.mem.write_u8(PARCEL + i as u32, b).unwrap();
        }
        for i in (0..0x100u32).step_by(4) {
            cpu.mem.write_u32(REPLY + i, 0).unwrap();
        }

        // `{ s32 binder_id, u32 code, u32 flags }`, parcel in the send buffer.
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&QUERY.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        ipc_request_plain_with_both_buffers(
            &mut cpu,
            relay,
            cmd,
            (PARCEL, parcel.len() as u32),
            (REPLY, 0x100),
            &data,
        );

        assert_eq!(
            cpu.mem.read_u32(tls + 0x18).unwrap(),
            0,
            "cmd {cmd} was refused"
        );
        // The reply parcel `{ i32 value, i32 status }` behind the usual header.
        let payload_size = cpu.mem.read_u32(REPLY).unwrap();
        let payload_off = cpu.mem.read_u32(REPLY + 4).unwrap();
        assert_eq!(payload_off, 16, "cmd {cmd}: no reply parcel came back");
        assert_eq!(
            payload_size, 8,
            "cmd {cmd}: the reply is a value and a status"
        );
        assert_eq!(
            cpu.mem.read_u32(REPLY + payload_off).unwrap(),
            1280,
            "cmd {cmd}: the queue answered a width query with something else"
        );
        assert_eq!(
            cpu.mem.read_u32(REPLY + payload_off + 4).unwrap(),
            0,
            "cmd {cmd} failed"
        );
    }
}

#[test]
fn vi_native_window_names_the_binder_interface() {
    // `OpenLayer`'s parcel holds a full flattened binder; nnSdk checks the
    // interface name (vi 114-1 otherwise).
    const VI: u64 = 0xB000;
    const WINDOW: u32 = 0x8000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // OpenLayer lives on IApplicationDisplayService.
    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(display, 0, "no IApplicationDisplayService");

    // OpenLayer with the 0x100-byte native-window receive buffer.
    ipc_request_plain_with_buffer(&mut cpu, display, 2020, WINDOW, 0x100, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "OpenLayer failed");
    let size = cpu.mem.read_u64(tls + 0x20).unwrap() as u32;

    // Parcel header: { payload_size, payload_off, objects_size, objects_off }.
    let payload_size = cpu.mem.read_u32(WINDOW).unwrap();
    let payload_off = cpu.mem.read_u32(WINDOW + 4).unwrap();
    let objects_size = cpu.mem.read_u32(WINDOW + 8).unwrap();
    let objects_off = cpu.mem.read_u32(WINDOW + 12).unwrap();
    assert_eq!(payload_size, 0x28, "a flat_binder_object is 0x28 bytes");
    assert_eq!(payload_off, 0x10);
    assert_eq!(objects_size, 4, "one object in the offset table");
    assert_eq!(objects_off, payload_off + payload_size);
    assert_eq!(
        size,
        objects_off + objects_size,
        "the reported size must cover it all"
    );

    let payload = WINDOW + payload_off;
    assert_eq!(
        cpu.mem.read_u32(payload).unwrap(),
        2,
        "flat_binder_object type"
    );
    let binder = cpu.mem.read_u64(payload + 8).unwrap();
    assert_ne!(binder, 0, "no IGraphicBufferProducer id");
    let mut name = [0u8; 8];
    for (i, slot) in name.iter_mut().enumerate() {
        *slot = cpu.mem.read_u8(payload + 0x18 + i as u32).unwrap();
    }
    assert_eq!(&name, b"dispdrv\0", "the interface has to name itself");
}

#[test]
fn an_undriven_gpio_pad_reads_high() {
    // Undriven GPIO pads read High; Low on both volume pads means maintenance mode.
    const GPIO: u64 = 0x9100;
    const VOLUME_UP: u32 = 0x3500_0003;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(GPIO, "gpio");
    let tls = cpu.tls_base();

    // IManager::OpenSession2(DeviceCode, AccessMode) -> IPadSession.
    let mut args = Vec::new();
    args.extend_from_slice(&VOLUME_UP.to_le_bytes());
    args.extend_from_slice(&1u32.to_le_bytes());
    ipc_request_plain(&mut cpu, GPIO, 7, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "OpenSession2 failed"
    );
    // { send_pid:1, num_copy:4, num_move:4 }: a move handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let pad = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(pad, 0, "no IPadSession came back");

    // IPadSession::GetValue -> GpioValue::High.
    ipc_request_plain(&mut cpu, pad, 9, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "GetValue failed");
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        1,
        "an undriven pad is High"
    );

    // GetInterruptStatus: nothing pending.
    ipc_request_plain(&mut cpu, pad, 6, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0);
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
fn the_home_menu_opens_a_system_applet_proxy() {
    // qlaunch opens IAllSystemAppletProxiesService command 100 and aborts on error.
    const APPLET: u64 = 0x9300;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletAE");
    let tls = cpu.tls_base();

    // OpenSystemAppletProxy(u64 reserved, pid, process handle) -> ISystemAppletProxy.
    ipc_request_plain(&mut cpu, APPLET, 100, &0u64.to_le_bytes());
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "the Home Menu was refused"
    );
    let proxy = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(proxy, 0, "no ISystemAppletProxy came back");

    // GetHomeMenuFunctions, only on this proxy.
    ipc_request_plain(&mut cpu, proxy, 20, &[]);
    let home = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(home, 0);
    // IsSleepEnabled -> bool.
    ipc_request_plain(&mut cpu, home, 40, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 1);
    // GetHomeButtonWriterLockAccessor -> a real ILockAccessor session.
    ipc_request_plain(&mut cpu, home, 30, &[]);
    let lock = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(lock, 0, "no ILockAccessor came back");
    ipc_request_plain(&mut cpu, lock, 4, &[]); // IsLocked
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 0);

    // GetGlobalStateController: ShouldSleepOnBoot is false.
    ipc_request_plain(&mut cpu, proxy, 21, &[]);
    let global = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(global, 0);
    ipc_request_plain(&mut cpu, global, 14, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 0);

    // Sleep and shutdown sequences stay refused.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request_plain(&mut cpu, global, 3, &[]); // StartShutdownSequence
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn the_display_answers_what_it_is() {
    // `ListDisplays` and `ListDisplayModes` must write their out data; an empty
    // success leaves the caller reading stale padding as a count.
    const VI: u64 = 0xB100;
    const BUF: u32 = 0x8000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, VI, 2, &[]); // GetDisplayService
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(display, 0, "no IApplicationDisplayService");

    // ListDisplays -> one DisplayInfo { char name[0x40]; bool limited; pad[7];
    // u64 layer_limit; u64 width; u64 height }, and a count of one.
    ipc_request_plain_with_buffer(&mut cpu, display, 1000, BUF, 0xc0, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "ListDisplays failed"
    );
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 1, "no displays");
    let name: Vec<u8> = (0..7).map(|i| cpu.mem.read_u8(BUF + i).unwrap()).collect();
    assert_eq!(&name, b"Default");
    assert_eq!(
        cpu.mem.read_u8(BUF + 0x40).unwrap(),
        1,
        "layer limit not enabled"
    );
    assert_eq!(cpu.mem.read_u64(BUF + 0x50).unwrap(), 1280);
    assert_eq!(cpu.mem.read_u64(BUF + 0x58).unwrap(), 720);

    // OpenDisplay takes that name and returns the display id.
    let mut open = [0u8; 0x40];
    open[..7].copy_from_slice(b"Default");
    ipc_request_plain(&mut cpu, display, 1010, &open);
    let display_id = cpu.mem.read_u64(tls + 0x20).unwrap();
    assert_ne!(
        display_id, 0,
        "a display id of 0 is the no-display sentinel"
    );

    // SetLayerScalingMode(mode, layer): only the two supported modes succeed.
    let scaling = |mode: u32| {
        let mut args = [0u8; 16];
        args[..4].copy_from_slice(&mode.to_le_bytes());
        args[8..].copy_from_slice(&1u64.to_le_bytes());
        args
    };
    for (mode, result) in [(2, 0), (4, 0), (1, 114 | (6 << 9)), (5, 114 | (1 << 9))] {
        ipc_request_plain(&mut cpu, display, 2101, &scaling(mode));
        assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), result, "mode {mode}");
    }

    // GetDisplayResolution must agree.
    ipc_request_plain(&mut cpu, display, 1102, &display_id.to_le_bytes());
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 1280);
    assert_eq!(cpu.mem.read_u64(tls + 0x28).unwrap(), 720);

    // ListDisplayModes (ISystemDisplayService): one
    // DisplayModeInfo { u32 width; u32 height; f32 refresh; u32 }.
    ipc_request_plain(&mut cpu, display, 101, &[]);
    let system = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(system, 0, "no ISystemDisplayService");

    for i in 0..0x40u32 {
        cpu.mem.write_u32(BUF + i * 4, 0xDEAD_BEEF).unwrap();
    }
    ipc_request_plain_with_buffer(
        &mut cpu,
        system,
        3000,
        BUF,
        0x100,
        true,
        &display_id.to_le_bytes(),
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "ListDisplayModes failed"
    );
    assert_eq!(
        cpu.mem.read_u64(tls + 0x20).unwrap(),
        1,
        "no modes to pick from"
    );
    assert_eq!(cpu.mem.read_u32(BUF).unwrap(), 1280);
    assert_eq!(cpu.mem.read_u32(BUF + 4).unwrap(), 720);
    assert_eq!(f32::from_bits(cpu.mem.read_u32(BUF + 8).unwrap()), 60.0);
}

#[test]
fn nifm_answers_a_system_title_the_same_as_an_application() {
    // `nifm:u`, `nifm:s` and `nifm:a` share one interface; 12 is
    // GetCurrentIpAddress and 18 GetInternetConnectionStatus.
    const NIFM: u64 = 0xD100;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(NIFM, "nifm:s");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, NIFM, 5, &[]); // CreateGeneralService
    let general = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(general, 0, "no IGeneralService for nifm:s");

    ipc_request_plain(&mut cpu, general, 12, &[]); // GetCurrentIpAddress
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap().to_le_bytes(),
        [192, 168, 1, 100],
        "GetCurrentIpAddress did not answer with an address"
    );
    ipc_request_plain(&mut cpu, general, 18, &[]); // GetInternetConnectionStatus
    assert_eq!(
        cpu.mem.read_u8(tls + 0x20).unwrap(),
        2,
        "not an ethernet link"
    );
    assert_eq!(cpu.mem.read_u8(tls + 0x22).unwrap(), 2, "not connected");

    // A request on an up link is accepted immediately, its events already fired.
    ipc_request_plain(&mut cpu, general, 4, &[]); // CreateRequest
    let request = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(request, 0, "no IRequest");
    ipc_request_plain(&mut cpu, request, 0, &[]); // GetRequestState
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        3,
        "the request was not accepted"
    );

    ipc_request_plain(&mut cpu, request, 2, &[]); // GetSystemEventReadableHandles
                                                  // { send_pid:1, num_copy:4, num_move:4 }: two copy handles.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 2 << 1);
    let state = cpu.mem.read_u32(tls + 0x0c).unwrap();
    let done = cpu.mem.read_u32(tls + 0x10).unwrap();
    assert_ne!(state, 0);
    assert_ne!(done, 0);
    assert_ne!(state, done);
    assert_eq!(
        wait_sync(&mut cpu, &[state], 0).0,
        0,
        "the state never settled"
    );
    assert_eq!(
        wait_sync(&mut cpu, &[done], 0).0,
        0,
        "the request never finished"
    );
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
fn the_display_refreshes_without_being_drawn_to() {
    // Vsync fires on a period as well as on present, since titles wait for it
    // before rendering.
    const VI: u64 = 0xB500;
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, display, 5202, &[]);
    // { send_pid:1, num_copy:4, num_move:4 }: one copy handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let vsync = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(vsync, 0, "the guest must receive a real handle");

    assert_eq!(
        wait_sync(&mut cpu, &[vsync], 0).0,
        RESULT_TIMED_OUT,
        "vsync fired early"
    );

    // Run out the refresh period on nops; the auto-clear event fires once per period.
    cpu.mem.map_zero(0x2000, 0x100).unwrap();
    cpu.mem.map(0x2000, &nop().to_le_bytes()).unwrap();
    for _ in 0..switch_core::cpu::VSYNC_PERIOD_CYCLES {
        cpu.set_pc(0x2000);
        cpu.step().unwrap();
    }
    assert_eq!(
        wait_sync(&mut cpu, &[vsync], 0).0,
        0,
        "the display never refreshed"
    );
    assert_eq!(
        wait_sync(&mut cpu, &[vsync], 0).0,
        RESULT_TIMED_OUT,
        "it refreshed twice"
    );

    // A present signals it too.
    cpu.signal_event(u64::from(vsync));
    assert_eq!(
        wait_sync(&mut cpu, &[vsync], 0).0,
        0,
        "a signalled vsync did not fire"
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
fn vi_reads_a_control_request_in_either_encoding() {
    // With-context control (type 7) ConvertToDomain returns an object id, not
    // a binder AdjustRefcount.
    const VI: u64 = 0xB400;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    build_ipc_request(&mut cpu, 7, None, 0);
    run_ipc_request(&mut cpu, VI);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_ne!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "no domain object came back"
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
fn the_system_shared_buffer_hands_out_slots_an_applet_can_present() {
    // System applets draw through AM's shared buffer once
    // `IsSystemBufferSharingEnabled` succeeds.
    const VI: u64 = 0xB500;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetSharedBufferMemoryHandleId -> the buffer's nvmap handle and size.
    ipc_request(&mut cpu, VI, 4, None, 8225);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_ne!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "no nvmap handle came back"
    );
    assert_eq!(
        cpu.mem.read_u64(tls + 0x28).unwrap(),
        u64::from(switch_core::cpu::SHARED_BUFFER_GEOMETRY.shared_buffer_size())
    );

    // AcquireSharedFrameBuffer -> an empty fence, the slots, and the slot to draw.
    // The two slots alternate.
    let mut acquired = Vec::new();
    for _ in 0..4 {
        ipc_request(&mut cpu, VI, 4, None, 8254);
        assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
        assert_eq!(
            cpu.mem.read_u32(tls + 0x20).unwrap(),
            0,
            "the fence should be empty"
        );
        assert_eq!(cpu.mem.read_u32(tls + 0x44).unwrap(), 0);
        assert_eq!(cpu.mem.read_u32(tls + 0x48).unwrap(), 1);
        assert_eq!(cpu.mem.read_u32(tls + 0x4c).unwrap() as i32, -1);
        acquired.push(cpu.mem.read_u64(tls + 0x58).unwrap());
    }
    assert_eq!(
        acquired,
        vec![0, 1, 0, 1],
        "the two slots did not alternate"
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

#[test]
fn an_applet_is_told_it_came_into_the_foreground_not_that_focus_changed() {
    // Applets get `ChangeIntoForeground`, not the application's `FocusStateChanged`.
    const CHANGE_INTO_FOREGROUND: u32 = 1;
    const APPLET: u64 = 0x9700;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletAE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    // OpenSystemAppletProxy, as qlaunch opens it.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy_service), 100);
    let proxy = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0); // ICommonStateGetter
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();

    ipc_request(&mut cpu, APPLET, 4, Some(state_getter), 1); // ReceiveMessage
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "no message was waiting"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        CHANGE_INTO_FOREGROUND
    );
}

#[test]
fn an_applet_that_handles_its_own_display_is_asked_to_display() {
    // `SetHandlesRequestToDisplay(true)` makes AM queue `RequestToDisplay`;
    // the applet draws nothing until it reads it.
    const REQUEST_TO_DISPLAY: u32 = 41;
    const FOCUS_STATE_CHANGED: u32 = 15;
    const NO_MESSAGES: u32 = 128 | (3 << 9);
    let (mut cpu, applet, proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();
    ipc_request(&mut cpu, applet, 4, Some(proxy), 1); // GetSelfController
    let self_controller = cpu.mem.read_u32(tls + 0x30).unwrap();

    // Drain the startup message.
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 0); // GetEventHandle
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 1); // ReceiveMessage
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), FOCUS_STATE_CHANGED);
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 1);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        NO_MESSAGES,
        "the queue should be empty"
    );

    build_ipc_request(&mut cpu, 4, Some(self_controller), 50);
    cpu.mem.write_u8(tls + 0x30, 1).unwrap();
    run_ipc_request(&mut cpu, applet);

    ipc_request(&mut cpu, applet, 4, Some(state_getter), 1);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "nothing was queued to display"
    );
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), REQUEST_TO_DISPLAY);

    // Once, not on every poll.
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 1);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        NO_MESSAGES,
        "it was queued twice"
    );

    // The approval that follows is accepted.
    ipc_request(&mut cpu, applet, 4, Some(self_controller), 51);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "ApproveToDisplay was refused"
    );
}

#[test]
fn parental_control_hands_out_the_events_it_is_asked_for() {
    // The Home Menu aborts unless pctl's synchronisation event is handed out;
    // it never fires.
    const PCTL: u64 = 0x9500;
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(PCTL, "pctl");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, PCTL, 0, &[]); // CreateService
    let service = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(service, 0, "no IParentalControlService");

    for cmd in [1207u32, 1432, 1457, 1473] {
        ipc_request_plain(&mut cpu, service, cmd, &[]);
        assert_eq!(
            cpu.mem.read_u32(tls + 0x18).unwrap(),
            0,
            "pctl {cmd} refused"
        );
        assert_eq!(
            cpu.mem.read_u32(tls + 0x08).unwrap(),
            1 << 1,
            "pctl {cmd} is not a copy handle"
        );
        let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
        assert_ne!(event, 0, "pctl {cmd} handed back no event");
        assert_eq!(
            wait_sync(&mut cpu, &[event], 0).0,
            RESULT_TIMED_OUT,
            "pctl {cmd} fired"
        );
    }

    // "Is restricted" is false; "is allowed" is true.
    ipc_request_plain(&mut cpu, service, 1031, &[]); // IsRestrictionEnabled
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 0);
    ipc_request_plain(&mut cpu, service, 1458, &[]); // IsPlayTimerAlarmDisabled
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 1);
}

#[test]
fn the_vibration_device_list_is_a_hid_session_not_a_fabricated_object() {
    // `CreateActiveVibrationDeviceList`'s sub-session must be routed:
    // `nn::hid::InitializeVibrationDevice` calls its command 0 per motor.
    const HID: u64 = 0x1000;
    const SFCO: u32 = 0x4F43_4653;
    const HAS_HANDLE_DESCRIPTOR: u32 = 1 << 31;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HID, "hid");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, HID, 4, None, 203);
    let list = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(
        list, 0,
        "CreateActiveVibrationDeviceList moved no session back"
    );

    ipc_request(&mut cpu, list, 4, None, 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x10).unwrap(), SFCO);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(
        cpu.mem.read_u32(tls).unwrap() & HAS_HANDLE_DESCRIPTOR,
        0,
        "InitializeVibrationDevice was answered with handles"
    );
}

#[test]
fn ldr_ro_initialize_is_not_a_fabricated_object() {
    // RegisterProcessHandle (cmd 4), `nn::ro::Initialize`'s first call: a bare Result.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x10).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    // Bit 31 of the second header word marks handles; there are none.
    assert_eq!(cpu.mem.read_u32(tls + 4).unwrap() >> 31, 0);
}

#[test]
fn ldr_ro_maps_a_module_where_nothing_else_lives() {
    // LoadModule must actually map the NRO at the returned address.
    use switch_core::cpu::{RO_MODULE_REGION_ADDR, RO_MODULE_REGION_SIZE};
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    let base = cpu.mem.read_u64(tls + 0x20).unwrap();
    assert!(
        (u64::from(RO_MODULE_REGION_ADDR)
            ..u64::from(RO_MODULE_REGION_ADDR) + u64::from(RO_MODULE_REGION_SIZE))
            .contains(&base),
        "a module must land in the region set aside for one, not at {base:#x}"
    );
    let base = base as u32;

    // The three segments in file order, then a zero-filled BSS.
    assert_eq!(cpu.mem.read_u32(base).unwrap(), 0x1400_0010);
    assert_eq!(cpu.mem.read_u8(base + 0x1000).unwrap(), 0xAA);
    assert_eq!(cpu.mem.read_u8(base + 0x2000).unwrap(), 0xBB);
    assert_eq!(cpu.mem.read_u8(base + 0x3000).unwrap(), 0);

    // `.text` is read-only; `.data` is writable for relocations.
    assert!(cpu.mem.write_u32(base, 0).is_err());
    assert!(cpu.mem.write_u32(base + 0x2000, 0).is_ok());
}

#[test]
fn ldr_ro_unload_frees_the_address_space_and_the_protection() {
    // Unloading removes the pages and the `.text` read-only marking.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let base = cpu.mem.read_u64(tls + 0x20).unwrap();
    ldr_ro_request(&mut cpu, handle, 1, &[base]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert!(cpu.mem.write_u32(base as u32, 0).is_ok());

    // The address space is reused by the next load.
    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), base);

    // Unloading something never loaded is NotLoaded.
    const NOT_LOADED: u32 = 22 | (1028 << 9);
    ldr_ro_request(&mut cpu, handle, 1, &[0x2800_0000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), NOT_LOADED);
}

#[test]
fn ldr_ro_two_modules_do_not_overlap() {
    // The second module goes behind the first, BSS included.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let first = cpu.mem.read_u64(tls + 0x20).unwrap();
    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let second = cpu.mem.read_u64(tls + 0x20).unwrap();
    assert_eq!(
        second,
        first + 0x4000,
        "image plus BSS, and no gap to waste"
    );
}

#[test]
fn ldr_ro_refuses_what_is_not_a_module() {
    // A bad NRO or an undersized BSS is refused.
    const INVALID_NRO: u32 = 22 | (4 << 9);
    const INVALID_ADDRESS: u32 = 22 | (1025 << 9);
    const INVALID_SIZE: u32 = 22 | (1026 << 9);
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[0x1100_0000, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_NRO);

    ldr_ro_request(
        &mut cpu,
        handle,
        0,
        &[NRO_SOURCE + 8, 0x3000, NRO_BSS, 0x1000],
    );
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_ADDRESS);

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_SIZE);
}

#[test]
fn ldr_ro_module_info_is_registered_before_it_is_unregistered() {
    // NRRs are tracked, so unregistering an unknown one is an error.
    const INVALID_NRR: u32 = 22 | (6 << 9);
    const NOT_REGISTERED: u32 = 22 | (1029 << 9);
    const NRR: u64 = 0x1020_0000;
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 3, &[NRR]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), NOT_REGISTERED);

    // Nothing is at that address yet.
    ldr_ro_request(&mut cpu, handle, 2, &[NRR, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), INVALID_NRR);

    cpu.mem.map(NRR as u32, b"NRR0").unwrap();
    ldr_ro_request(&mut cpu, handle, 2, &[NRR, 0x1000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    ldr_ro_request(&mut cpu, handle, 3, &[NRR]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
}

#[test]
fn docking_moves_every_answer_that_depends_on_it() {
    // The operation mode drives `am`, `apm`, `vi`, `clkrst` and touch together.
    use switch_core::cpu::OperationMode;
    let (mut cpu, handle, _proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();
    assert_eq!(
        cpu.operation_mode(),
        OperationMode::Handheld,
        "a console starts undocked"
    );

    // Handheld: mode 0, performance Normal (0).
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 5);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0, "GetOperationMode");
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 6);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        0,
        "GetPerformanceMode"
    );

    // Clear the startup focus message.
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);

    cpu.set_operation_mode(OperationMode::Docked);
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 5);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        1,
        "docked is Console (1)"
    );
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 6);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        1,
        "docked is Boost (1)"
    );

    // The title is notified of the change.
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "a message is waiting"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        30,
        "OperationModeChanged"
    );
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        31,
        "PerformanceModeChanged"
    );

    // GetDefaultDisplayResolution (60) must match the mode.
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 60);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x30).unwrap(),
        1920,
        "docked default width"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x34).unwrap(),
        1080,
        "docked default height"
    );

    // Docking a docked console announces nothing.
    const NO_MESSAGES: u32 = 128 | (3 << 9);
    cpu.set_operation_mode(OperationMode::Docked);
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), NO_MESSAGES);
}

#[test]
fn the_dock_resizes_the_display_and_takes_the_touchscreen_away() {
    use switch_core::cpu::{OperationMode, TouchPoint};
    const SHMEM: u32 = 0x3000_0000;
    const LIFO: u32 = SHMEM + 0x400;
    const VI: u64 = 0x2000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetDisplayMode (3200) reports width, height and refresh by value.
    ipc_request(&mut cpu, VI, 6, None, 3200);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        1280,
        "handheld width"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x24).unwrap(),
        720,
        "handheld height"
    );

    cpu.set_operation_mode(OperationMode::Docked);
    ipc_request(&mut cpu, VI, 6, None, 3200);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1920, "docked width");
    assert_eq!(cpu.mem.read_u32(tls + 0x24).unwrap(), 1080, "docked height");

    // Docked, touch samples are still published but carry no contacts.
    cpu.set_reg(1, SHMEM as u64);
    cpu.set_reg(2, 0x40000);
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    cpu.set_pc(0x1000);
    let state = LIFO + 0x20 + 8;
    let before = cpu.mem.read_u64(LIFO + 0x20).unwrap();
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 640,
        y: 360,
    }]);
    assert_eq!(
        cpu.mem.read_u32(state + 0x08).unwrap(),
        0,
        "docked reports no contacts"
    );
    assert!(
        cpu.mem.read_u64(LIFO + 0x20).unwrap() > before,
        "the sample still advances"
    );

    // Undocked again, the same contact lands.
    cpu.set_operation_mode(OperationMode::Handheld);
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 640,
        y: 360,
    }]);
    assert_eq!(
        cpu.mem.read_u32(state + 0x08).unwrap(),
        1,
        "handheld reports the contact"
    );
    assert_eq!(cpu.mem.read_u32(state + 0x10 + 0x10).unwrap(), 640, "x");
}

#[test]
fn the_shared_buffer_does_not_move_when_the_console_is_docked() {
    // The shared buffer pool layout is fixed at GetSharedBufferMemoryHandleId
    // and must not change on dock; qlaunch lays out at 1280x720 regardless.
    use switch_core::cpu::{OperationMode, SHARED_BUFFER_GEOMETRY};
    const VI: u64 = 0x2000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetSharedBufferMemoryHandleId: total size, then per-slot offset, size, width and height.
    let pool = |cpu: &mut Cpu| {
        ipc_request(cpu, VI, 4, None, 8225);
        assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
        cpu.mem.read_u64(tls + 0x28).unwrap()
    };

    let handheld = pool(&mut cpu);
    assert_eq!(
        handheld,
        u64::from(SHARED_BUFFER_GEOMETRY.shared_buffer_size())
    );

    cpu.set_operation_mode(OperationMode::Docked);
    assert_eq!(
        pool(&mut cpu),
        handheld,
        "docking moved the pool the applet had mapped"
    );

    // The display size does change.
    assert_eq!(cpu.operation_mode().display_size(), (1920, 1080));
    assert_eq!(SHARED_BUFFER_GEOMETRY.display_size(), (1280, 720));

    // Rows round up to a 128-row block: 720 -> 768, 1080 -> 1152.
    assert_eq!(OperationMode::Handheld.shared_buffer_rows(), 768);
    assert_eq!(OperationMode::Docked.shared_buffer_rows(), 1152);
}

#[test]
fn the_resolution_change_event_fires_on_the_dock() {
    // `GetDefaultDisplayResolutionChangeEvent` is one shared event, signalled on change.
    use switch_core::cpu::OperationMode;
    let (mut cpu, handle, _proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 61);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap() as u64;
    assert_ne!(
        event, 0,
        "an event has to come back in the copy-handle slot"
    );
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 61);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x0c).unwrap() as u64,
        event,
        "asking twice has to give back the object already being waited on"
    );
    assert_eq!(
        cpu.event_signaled(event),
        Some(false),
        "dark until something changes"
    );

    cpu.set_operation_mode(OperationMode::Docked);
    assert_eq!(
        cpu.event_signaled(event),
        Some(true),
        "the dock is what changes it"
    );
}

#[test]
fn hwopus_reports_a_work_buffer_size_before_it_opens_anything() {
    // `nn::codec` allocates the reported work buffer size before opening a decoder.
    const HWOPUS: u64 = 0xC000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HWOPUS, "hwopus");
    let tls = cpu.tls_base();

    // GetWorkBufferSizeEx { sample_rate, channel_count, use_large_frame_size }.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 5, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "GetWorkBufferSizeEx failed"
    );
    let stereo = cpu.mem.read_u32(tls + 0x20).unwrap();
    assert!(
        stereo > 0x1000,
        "a work buffer of {stereo:#x} bytes is not one"
    );

    // The large-frame form fits a 120 ms packet.
    args[8] = 1;
    ipc_request_plain(&mut cpu, HWOPUS, 5, &args);
    let large = cpu.mem.read_u32(tls + 0x20).unwrap();
    assert!(
        large > stereo,
        "the large-frame size {large:#x} is not above {stereo:#x}"
    );

    // An unsupported rate is refused.
    let mut bad = Vec::new();
    bad.extend_from_slice(&44_100u32.to_le_bytes());
    bad.extend_from_slice(&2u32.to_le_bytes());
    bad.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 5, &bad);
    let result = cpu.mem.read_u32(tls + 0x18).unwrap();
    assert_eq!(result & 0x1FF, 111, "not an hwopus error: {result:#x}");
    assert_eq!(
        result >> 9,
        1001,
        "not the invalid-sample-rate error: {result:#x}"
    );
}

#[test]
fn hwopus_decodes_a_packet_into_the_buffer_the_caller_offered() {
    // Packets carry an eight-byte big-endian { size, final_range } header,
    // counted in "bytes consumed".
    const HWOPUS: u64 = 0xC000;
    const INPUT: u32 = 0x9000;
    const OUTPUT: u32 = 0x9400;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map_zero(INPUT, 0x400).unwrap();
    cpu.mem.map_zero(OUTPUT, 0x1000).unwrap();
    cpu.register_service_handle(HWOPUS, "hwopus");
    let tls = cpu.tls_base();

    // OpenHardwareOpusDecoderEx { rate, channels, large_frame } + work size.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&1u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    args.extend_from_slice(&0x8000u32.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 4, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "OpenHardwareOpusDecoderEx failed"
    );
    // { send_pid:1, num_copy:4, num_move:4 }: a move handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let decoder = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(decoder, 0, "no IHardwareOpusDecoder came back");

    // One 20 ms CELT-only packet, 48 kHz mono, from the reference encoder.
    let len = OPUS_PACKET.len() as u32;
    for (i, &byte) in (len).to_be_bytes().iter().enumerate() {
        cpu.mem.write_u8(INPUT + i as u32, byte).unwrap();
    }
    for (i, &byte) in OPUS_PACKET.iter().enumerate() {
        cpu.mem.write_u8(INPUT + 8 + i as u32, byte).unwrap();
    }

    // DecodeInterleaved: reset flag in, { bytes read, samples } out.
    ipc_request_plain_with_both_buffers(
        &mut cpu,
        decoder,
        8,
        (INPUT, 8 + len),
        (OUTPUT, 0x1000),
        &[0u8, 0, 0, 0],
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "DecodeInterleaved failed"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        8 + len,
        "wrong byte count"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x24).unwrap(),
        960,
        "a 20 ms frame is 960 samples"
    );

    // The samples are not all zero.
    let loudest = (0..960)
        .map(|i| i32::from(cpu.mem.read_u16(OUTPUT + i * 2).unwrap() as i16).abs())
        .max()
        .unwrap();
    assert!(
        loudest > 1000,
        "the decode is silent (loudest sample {loudest})"
    );
}

#[test]
fn hwopus_refuses_a_packet_shorter_than_its_own_header() {
    // A header size beyond the buffer, or no room for a header, is an error.
    const HWOPUS: u64 = 0xC000;
    const INPUT: u32 = 0x9000;
    const OUTPUT: u32 = 0x9400;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map_zero(INPUT, 0x400).unwrap();
    cpu.mem.map_zero(OUTPUT, 0x1000).unwrap();
    cpu.register_service_handle(HWOPUS, "hwopus");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&1u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    args.extend_from_slice(&0x8000u32.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 4, &args);
    let decoder = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    // A header claiming more payload than the buffer holds.
    for (i, &byte) in 0x1000u32.to_be_bytes().iter().enumerate() {
        cpu.mem.write_u8(INPUT + i as u32, byte).unwrap();
    }
    ipc_request_plain_with_both_buffers(
        &mut cpu,
        decoder,
        8,
        (INPUT, 64),
        (OUTPUT, 0x1000),
        &[0u8; 4],
    );
    let result = cpu.mem.read_u32(tls + 0x18).unwrap();
    assert_eq!(result & 0x1FF, 111, "not an hwopus error: {result:#x}");
    assert_eq!(
        result >> 9,
        3,
        "not the buffer-too-small error: {result:#x}"
    );

    // A buffer with nothing but the header in it.
    ipc_request_plain_with_both_buffers(
        &mut cpu,
        decoder,
        8,
        (INPUT, 8),
        (OUTPUT, 0x1000),
        &[0u8; 4],
    );
    let result = cpu.mem.read_u32(tls + 0x18).unwrap();
    assert_eq!(result >> 9, 8, "not the input-too-small error: {result:#x}");
}
