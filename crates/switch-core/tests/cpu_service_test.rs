//! The Horizon services, reached over IPC: `hid`, `am`, `vi`, the audio pair,
//! `ldr:ro`, `hwopus` and the rest.

mod cpu;

use cpu::*;
use switch_core::cpu::POINTER_BUFFER_SIZE;

#[test]
fn ssl_keeps_context_state_and_refuses_connections() {
    // ssl is the system TLS stack: a title asks the OS to build connections
    // rather than bringing its own implementation. The local half -- contexts
    // and their options -- is real here; the connection half is not, because
    // there is no socket layer under it.
    const SSL: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(SSL, "ssl");
    let tls = cpu.tls_base();
    ipc_request(&mut cpu, SSL, 5, None, 0); // ConvertToDomain
    let service = cpu.mem.read_u32(tls + 0x20).unwrap();

    // SetInterfaceVersion is the only ssl command an offline retail title
    // issues, because ssl is in its NPDM service list and nnSdk initialises it
    // at startup regardless.
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

    // Options are per-context state a caller reads back.
    let mut args = Vec::new();
    args.extend_from_slice(&2u32.to_le_bytes()); // option
    args.extend_from_slice(&1u32.to_le_bytes()); // value
    ipc_request_with_payload(&mut cpu, SSL, context, 0, &args);
    ipc_request_with_payload(&mut cpu, SSL, context, 1, &2u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);
    // An option never set reads as 0 rather than as another option's value.
    ipc_request_with_payload(&mut cpu, SSL, context, 1, &7u32.to_le_bytes());
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 0);

    // CreateConnection reports itself rather than handing back a connection
    // that can never connect.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, SSL, 4, Some(context), 2);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn hid_hands_over_the_input_shared_memory() {
    // The input *data* lives in a shared memory region the guest reads
    // directly; hid's IPC is the negotiation that hands it over. libnx got
    // working input out of the old fabricated reply only because it maps that
    // region by size and this emulator recognises it that way -- nnSdk calls a
    // method on the IAppletResource it is given, and an object id is not one.
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

    // QueryPointerBufferSize has to be non-zero: nn::hid::SetSupportedNpadIdType
    // marshals its id array as a pointer buffer, and nnSdk checks the
    // negotiated size before it sends.
    ipc_request(&mut cpu, hid, 5, None, 3);
    assert_ne!(cpu.mem.read_u16(tls + 0x20).unwrap(), 0);
}

#[test]
fn hid_reads_back_what_the_guest_configured() {
    // A caller that sets a controller style set and reads back something else
    // decides the pad it wanted is not there -- which is what the generic
    // reply's incrementing object id looked like.
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
    // SendVibrationValue(handle, HidVibrationValue, aruid): the value is four
    // floats, so the two band amplitudes sit at +4 and +0xc after the u32
    // handle. The frontend maps them onto dual-rumble's magnitudes.
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

    // Out of range or not finite is clamped rather than handed to the browser.
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
    // `libnx` opens hid:sys in hidsysInitialize and records the session's
    // pointer buffer size on it before sending anything, so for a title that
    // never calls a hid:sys command -- Checkpoint is one -- opening the
    // service *is* the only traffic there ever is. With hid:sys unrouted that
    // control request fell through to the generic reply and was answered with
    // a fabricated object id, exactly the way ns:am2 was.
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

    // EnableAppletToGetInput: a setter over state this emulator does not have.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 503);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // GetMaskedSupportedNpadStyleSet(u64 aruid) -> NpadStyleSet. This is what
    // the system permits, not what the caller asked for -- so it has to name
    // controllers even though nothing has called SetSupportedNpadStyleSet,
    // and handheld above all, since that is the mode this console reports.
    ipc_request_with_payload(&mut cpu, HIDSYS, server, 310, &[0u8; 8]);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    let styles = cpu.mem.read_u32(tls + 0x30).unwrap();
    assert_ne!(styles & (1 << 1), 0, "handheld is not supported");
    assert_ne!(styles & (1 << 0), 0, "a full-key pad is not supported");

    // SetNpadSystemExtStateEnabled(bool, u64 aruid), the same: this console
    // publishes the SystemExt style already, so there is no permission left
    // for it to grant.
    let mut args = [0u8; 0x10];
    args[0] = 1;
    ipc_request_with_payload(&mut cpu, HIDSYS, server, 322, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // IsJoyConRailEnabled / IsJoyConAttachedOnAllRail -> bool. Handheld play
    // is the state where both pads are seated on the rails, and this console
    // reports handheld mode and publishes a handheld npad, so both are true.
    for cmd in [523u32, 525] {
        ipc_request(&mut cpu, HIDSYS, 4, Some(server), cmd);
        assert_eq!(
            cpu.mem.read_u32(tls + 0x28).unwrap(),
            0,
            "cmd {cmd} refused"
        );
        assert_eq!(cpu.mem.read_u8(tls + 0x30).unwrap(), 1, "cmd {cmd}");
    }

    // SetFirmwareHotfixUpdateSkipEnabled(bool): whether to skip the hotfix a
    // controller firmware update would apply. The pad here has no firmware to
    // flash, so there is nothing to skip -- but a refusal is a fatal.
    ipc_request_with_payload(&mut cpu, HIDSYS, server, 1120, &[1u8, 0, 0, 0]);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);

    // GetUniquePadIds -> an s64 count. A unique pad is a *detachable*
    // controller and the one here is the built-in handheld pad, so there are
    // none and the pointer buffer is left alone.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 703);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(tls + 0x30).unwrap(), 0);

    // AcquireHomeButtonEventHandle -> a copy handle. There is no Home button,
    // so it is handed out and never signalled.
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 101);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    assert_ne!(cpu.mem.read_u32(tls + 0x0c).unwrap(), 0);

    // Converting the session to a domain must not quietly turn it into
    // IHidServer: command 0 there is CreateAppletResource, and hid:sys has no
    // command 0 at all.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, HIDSYS, 4, Some(server), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn events_are_copy_handles_and_start_unsignalled() {
    // Every event a service hands out is a **copy** handle: a move handle
    // transfers ownership and lives in a different field of the handle
    // descriptor, so an event sent in the move slot is read back as 0. That is
    // why nnSdk spent whole boots waiting on handle 0 after asking for
    // GetGpuErrorDetectedSystemEvent.
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
    // { send_pid:1, num_copy:4, num_move:4 } -- one copy handle, no move ones.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0, "the guest must receive a real handle");

    // Nothing has fired it, so a poll times out. Reporting the wait satisfied
    // is what told nn::oe::GpuErrorHandler that the GPU had faulted.
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    let (result, _) = wait_sync(&mut cpu, &[event], 0);
    assert_eq!(result, RESULT_TIMED_OUT);

    // A second event, left unsignalled, so the index below is a real position
    // rather than "the first handle". GetAcquiredSleepLockEvent, not
    // GetEventHandle -- nothing here ever sleeps, while the applet-message
    // event *starts* signalled because AM really does have one message queued
    // at startup.
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0); // ICommonStateGetter
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(state_getter), 13);
    let quiet = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(quiet, event);

    // Once signalled it reports the index that fired, and consumes it: these
    // are auto-clear events, so a second poll times out again.
    cpu.signal_event(u64::from(event));
    let (result, index) = wait_sync(&mut cpu, &[quiet, event], 0);
    assert_eq!(result, 0);
    assert_eq!(index, 1, "the index of the handle that fired, not a count");
    let (result, _) = wait_sync(&mut cpu, &[event], 0);
    assert_eq!(result, RESULT_TIMED_OUT);

    // A handle this emulator does not model as an event is still treated as
    // ready, which is what keeps thread handles and unmodelled service handles
    // behaving as they always have.
    let (result, index) = wait_sync(&mut cpu, &[0x1234], 0);
    assert_eq!(result, 0);
    assert_eq!(index, 0);
}

#[test]
fn control_clone_hands_back_a_working_session() {
    // CloneCurrentObject (control command 2) duplicates a session, and the
    // reply has to carry a **new session handle as a move handle**. Answering
    // it with a bare success and no handle left nnSdk -- which clones fsp-srv
    // before mounting anything -- talking to handle 0, so nn::fs::MountRom
    // failed without ever issuing a filesystem command.
    // Clear of `alloc_handle`'s own range, which starts at 0x1000 -- a real
    // session handle always comes from there, but this one is hand-registered.
    const FS: u64 = 0x9000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(FS, "fsp-srv");
    let tls = cpu.tls_base();

    // Convert to a domain first, so the clone has objects to inherit.
    ipc_request(&mut cpu, FS, 5, None, 0);
    let object = cpu.mem.read_u32(tls + 0x20).unwrap();

    ipc_request(&mut cpu, FS, 5, None, 2); // CloneCurrentObject
    assert_eq!(cpu.read_x(0), 0);
    // Move handles land right after the 8-byte hipc header: a descriptor word
    // then the handles themselves.
    let clone = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(clone, 0, "clone must hand back a real handle, not 0");
    assert_ne!(clone, FS, "the clone is a separate session");

    // The clone reaches the same service, holding the same domain objects.
    let handles = cpu.service_handles_snapshot();
    assert!(handles
        .iter()
        .any(|(h, name)| *h == clone && name == "fsp-srv"));
    ipc_request(&mut cpu, clone, 4, Some(object), 1); // SetCurrentProcess
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
}

#[test]
fn storage_read_uses_the_istorage_field_layout() {
    // IStorage::Read is (s64 offset, u64 size) -- *not* IFile::Read, which
    // leads with a u32 option and pads to 8, putting its offset at +8 and its
    // size at +0x10. Reading those two fields here meant every RomFS read came
    // back as "0 bytes at offset 0x50": the guest mounted its RomFS, parsed an
    // empty header, and found none of its own files.
    const FS: u64 = 0x1000;
    const OUT: u32 = 0x6000;
    let romfs: Vec<u8> = (0..64u8).collect();
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.set_romfs(romfs.clone());
    cpu.register_service_handle(FS, "fsp-srv-storage");
    let tls = cpu.tls_base();

    // Read(offset = 4, size = 8).
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
    // Nothing past the requested size is touched.
    assert_eq!(cpu.mem.read_u8(OUT + 8).unwrap(), 0);

    // A read that runs off the end is refused, not clamped: `fs` checks the
    // range against the storage and reports 2002-3005 rather than filling
    // what exists. It used to be clamped, which reports success over a buffer
    // the caller's own bytes are still in, and a caller that trusts the
    // Result reads those as data.
    const OUT_OF_RANGE: u32 = 2 | (3005 << 9);
    cpu.mem.write_u8(OUT, 0xAA).unwrap();
    let mut args = Vec::new();
    args.extend_from_slice(&(romfs.len() as u64 - 2).to_le_bytes());
    args.extend_from_slice(&64u64.to_le_bytes());
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 64, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), OUT_OF_RANGE);
    assert_eq!(cpu.mem.read_u8(OUT).unwrap(), 0xAA, "buffer left alone");

    // The same read, sized to what is actually there, succeeds.
    let mut args = Vec::new();
    args.extend_from_slice(&(romfs.len() as u64 - 2).to_le_bytes());
    args.extend_from_slice(&2u64.to_le_bytes());
    ipc_request_with_buffer(&mut cpu, FS, 1, 0, OUT, 64, true, &args);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(OUT).unwrap(), romfs[romfs.len() - 2]);

    // GetSize reports the whole RomFS.
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

    // One whole message in a single packet: severity 3 is Error, and the
    // module name comes from key 6, the text from key 2.
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

    // A message split across packets: only the head carries the prefix and
    // only the tail ends the line, so the two halves join into one message.
    cpu.out.clear();
    let len = write_log_packet(&mut cpu, PACKET, HEAD, 1, &[(KEY_TEXT, b"split ")]);
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    let len = write_log_packet(&mut cpu, PACKET, TAIL, 1, &[(KEY_TEXT, b"message")]);
    ipc_request_with_buffer(&mut cpu, LM, logger, 0, PACKET, len, false, &[]);
    assert_eq!(
        String::from_utf8_lossy(&cpu.out),
        "[lm/INFO] split message\n"
    );

    // A packet claiming more payload than the buffer holds is trusted only as
    // far as the buffer goes, rather than walking off the end of the mapping.
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
fn pctl_reports_parental_controls_off() {
    const PCTL: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(PCTL, "pctl");
    let tls = cpu.tls_base();

    // Control::ConvertToDomain -> IParentalControlServiceFactory, then
    // CreateServiceWithoutInitialize -> IParentalControlService.
    ipc_request(&mut cpu, PCTL, 5, None, 0);
    let factory = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, PCTL, 4, Some(factory), 1);
    let service = cpu.mem.read_u32(tls + 0x30).unwrap();

    // A permission check answers with a bare Result: success *is* "permitted",
    // and a restriction is an error the caller checks for by value.
    for cmd in [1001u32, 1004, 1013, 1017] {
        ipc_request(&mut cpu, PCTL, 4, Some(service), cmd);
        assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd {cmd}");
    }

    // The two query families read in opposite directions, and answering both
    // the same way would report free communication as unavailable.
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

    // GenerateInquiryCode answers with the ten digits a guardian would read
    // out, NUL-padded to 0x20. Refusing it is what a caller turns into a
    // fatal 2010-0221 rather than carrying on without a code.
    ipc_request(&mut cpu, PCTL, 4, Some(service), 1204);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0, "cmd 1204 refused");
    let code: Vec<u8> = (0..0x20)
        .map(|i| cpu.mem.read_u8(tls + 0x30 + i).unwrap())
        .collect();
    assert!(
        code[..10].iter().all(u8::is_ascii_digit) && code[10..].iter().all(|&b| b == 0),
        "inquiry code is not ten digits in a 0x20 block: {code:?}"
    );

    // Anything else still reports honestly rather than fabricating a success.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request(&mut cpu, PCTL, 4, Some(service), 1203); // SetPinCode
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn applet_common_state_getter_reports_focus_once() {
    // ICommonStateGetter::ReceiveMessage (cmd 1) must hand out the startup
    // FocusStateChanged (15) exactly once and then report "no message", NOT
    // the AM_BUSY error (0x19280) that wedges hbmenu in its "wait for applet"
    // sleep loop, and not a fresh focus change on every poll, which made JKSV
    // treat every frame as a new focus transition.
    let (mut cpu, handle, _proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    assert_eq!(cpu.read_x(0), 0); // svc result
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0); // Result: success
    assert_ne!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0x19280);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 15); // FocusStateChanged

    ipc_request(&mut cpu, handle, 4, Some(state_getter), 1);
    const NO_MESSAGES: u32 = 128 | (3 << 9);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), NO_MESSAGES);

    // GetCurrentFocusState (cmd 9) reports InFocus so libnx's applet-mainloop
    // wait loop terminates.
    ipc_request(&mut cpu, handle, 4, Some(state_getter), 9);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 0);
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), 1);
}

#[test]
fn applet_unimplemented_command_is_an_error_not_a_fake_success() {
    // An `am` command with no implementation behind it must report cmif's
    // "unknown command id" rather than a bare success. Everything `am` returns
    // is a live handle or a piece of applet state the caller then acts on, so
    // a fabricated success is a wrong answer the guest believes: answering
    // IApplicationFunctions::GetGpuErrorDetectedSystemEvent that way left
    // nnSdk's system worker waiting on handle 0.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    let (mut cpu, handle, proxy, _state_getter) = applet_chain();
    let tls = cpu.tls_base();

    // IApplicationProxy::GetDisplayController, then a command it does not
    // have. 10 is AcquireLastApplicationCaptureBuffer, which hands back a
    // transfer memory handle -- exactly the shape a fabricated success gets
    // wrong, and one of the capture commands still not implemented.
    ipc_request(&mut cpu, handle, 4, Some(proxy), 4);
    let display_controller = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, handle, 4, Some(display_controller), 10);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn applet_control_command_with_context_is_not_a_normal_command() {
    // nnSdk sends every message in the "with context" encoding,
    // ControlWithContext (7) rather than Control (5). Reading only type 5 as a
    // control message turned `appletOE`'s opening QueryPointerBufferSize into
    // IApplicationProxyService command 3, which does not exist, and the applet
    // chain died before it ever opened.
    const APPLET: u64 = 0x1000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 7, None, 3); // QueryPointerBufferSize
    assert_eq!(cpu.mem.read_u32(tls + 0x10).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0); // success, not an error
}

#[test]
fn gamepad_input_writes_input_reg_and_hid_shmem() {
    // MapSharedMemory (svc 0x13) with x1=addr, x2=size must back the region
    // with real memory and, for a region hid's size, record it; set_gamepad_state
    // then mirrors the pad into INPUT_ADDR and into the two npad slots libnx
    // reads. The offsets are `HidSharedMemory`'s: npad at 0x9A00, 0x5000 per
    // controller, `full_key_lifo` at +0x28 and `handheld_lifo` at +0x378 within
    // `HidNpadInternalState`, each LIFO holding a 0x20-byte header then storage
    // entries of {sampling_number, HidNpadCommonState}.
    const SHMEM: u32 = 0x3000_0000;
    const NPAD: u32 = SHMEM + 0x9A00;
    const HANDHELD: u32 = NPAD + 8 * 0x5000;
    let mut cpu = cpu_at(0x1000);
    cpu.set_reg(1, SHMEM as u64);
    cpu.set_reg(2, 0x40000);
    cpu.mem.map(0x1000, &svc(0x13).to_le_bytes()).unwrap();
    cpu.run(1).unwrap();
    assert_eq!(cpu.read_x(0), 0);

    // A|B held, left stick pushed fully up and slightly left.
    cpu.set_gamepad_state(0x3, -1000, 30000, 0, 0);

    // The mask handed to the guest gains HidNpadButton_StickLUp (1 << 17); the
    // small horizontal deflection stays below the pseudo-button threshold.
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
        // Bit 0 of the storage's number is the seqlock's "being written" flag,
        // so it holds the state's own number doubled.
        assert_eq!(cpu.mem.read_u64(entry + 0x08).unwrap() * 2, sample);
        assert_eq!(cpu.mem.read_u64(entry + 0x10).unwrap(), expected_buttons);
        assert_eq!(
            cpu.mem.read_u32(entry + 0x18).unwrap(),
            1000u32.wrapping_neg()
        );
        assert_eq!(cpu.mem.read_u32(entry + 0x1C).unwrap(), 30000);
        // IsConnected, whatever else the controller reports about its halves.
        assert_eq!(cpu.mem.read_u32(entry + 0x28).unwrap() & 1, 1);

        // Power info, straight after `system_button_properties`: a full
        // battery for the pad and for each of its two halves. An unwritten
        // `battery_level` reads back as 0, which is `HidPowerInfo`'s flat
        // step rather than a missing reading, so a controller UI drew every
        // pad here as empty.
        for info in 0..3u32 {
            let level = cpu.mem.read_u32(base + 0x4198 + info * 4).unwrap();
            assert_eq!(level, 4, "battery_level[{info}]");
        }
        // PowerInfo{0,1,2}PowerConnected set, their Charging counterparts in
        // bits 0-2 clear: attached to the console, and already full.
        let properties = cpu.mem.read_u32(base + 0x4190).unwrap();
        assert_eq!(properties & 0x38, 0x38, "PowerConnected");
        assert_eq!(properties & 0x7, 0, "Charging");
    }
}

#[test]
fn touch_input_writes_the_hid_touchscreen_lifo() {
    // `HidSharedMemory.touch_screen` sits at 0x400, straight after the debug
    // pad's 0x400, and holds a `HidTouchScreenLifo`: the same 0x20-byte header
    // the npad LIFOs use, then storage entries of `{u64 sampling_number,
    // HidTouchScreenState}`. That state is `{u64 sampling_number, s32 count,
    // u32 reserved, HidTouchState touches[16]}`, and a `HidTouchState` is 0x28
    // bytes with finger_id at +0x0C, x at +0x10 and y at +0x14.
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
    // The storage's number is the state's doubled, so bit 0 stays clear for a
    // reader that treats it as the seqlock's "being written" flag.
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

    // Both contacts are new, so both carry `start_touch`. A UI taps on that
    // transition rather than on a finger being in the list, which is why
    // publishing zero here left the Home Menu registering every tap and acting
    // on none of them.
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x08).unwrap(), 1, "start 0");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x08).unwrap(), 1, "start 3");

    // Lifting one of the two publishes it **once more**, still counted, with
    // `end_touch`: the finger has to be seen going up, not merely stop being
    // there. The one still down is held, so its attributes go back to zero.
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

    // And only then is it gone, with the slot it vacated cleared so a reader
    // that scans the array rather than trusting the count finds no ghost.
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 5,
        y: 6,
    }]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 1, "contact count");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x10).unwrap(), 0, "vacated x");
    assert_eq!(cpu.mem.read_u32(touch(1) + 0x14).unwrap(), 0, "vacated y");

    // A full lift is a published state carrying the finger's end, not silence:
    // a title polling the LIFO has to see it go up.
    cpu.set_touch_state(&[]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 1, "the end sample");
    assert_eq!(cpu.mem.read_u32(touch(0) + 0x08).unwrap(), 2, "end");
    cpu.set_touch_state(&[]);
    assert_eq!(cpu.mem.read_u32(state + 0x08).unwrap(), 0, "contact count");

    // Coordinates are clamped to the digitizer, and the slot count to sixteen.
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
fn touch_input_before_hid_shared_memory_is_mapped_is_dropped() {
    // Nothing is buffered: with no mapping there is nowhere to put a contact,
    // and the host keeps sending while the finger is down anyway.
    use switch_core::cpu::TouchPoint;
    let mut cpu = cpu_at(0x1000);
    cpu.set_touch_state(&[TouchPoint {
        finger_id: 0,
        x: 1,
        y: 2,
    }]);
    assert_eq!(cpu.hid_shmem_addr(), 0);
}

#[test]
fn mapping_pl_shared_memory_delivers_the_shared_font() {
    use switch_core::cpu::PL_SHMEM_SIZE;
    // `plInitialize` maps pl's shared memory and homebrew then reads the font
    // out of it at the offset pl reported, so the bytes have to be there by
    // the time the mapping syscall returns. Each font sits behind the
    // eight-byte header a console stores it behind, which is why the offset
    // `GetSharedMemoryAddressOffset` reports is not zero.
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

    // A font handed over after the guest mapped the region still reaches it:
    // the guest is holding a pointer into memory it already mapped.
    let replacement: Vec<u8> = vec![0xAB; 0x1000];
    cpu.set_shared_font(replacement.clone());
    assert_eq!(
        cpu.mem.dump(ADDR + HEADER, replacement.len()).unwrap(),
        replacement
    );
}

#[test]
fn caps_a_reports_a_mounted_empty_album() {
    // The Album applet asks three things before it will draw anything: an
    // unnamed command 18, whether the album is mounted, and whether captures
    // are being auto-saved to the SD card. Nothing implemented `caps:a` at
    // all, so each came back as a fabricated object id, as a *bool*, a large
    // number read one byte at a time.
    //
    // There is no NAND album and no SD card here, so what these describe is a
    // freshly initialised console: mounted, and empty. Reporting it unmounted
    // is the card-removed error, which is a screen of its own rather than a
    // gallery.
    const CAPS: u64 = 0xCA00;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(CAPS, "caps:a");
    let tls = cpu.tls_base();

    // Unknown18 -> the number of bytes written into the caller's buffer.
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

    // GetAutoSavingStorage -> bool. There is no SD card to save to.
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

    // And the album it says is mounted is empty, by both the count and the
    // list: a caller told "mounted" asks these next, and a count it cannot
    // trust is worse than no service at all.
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
    // `AcquireCallerAppletCaptureSharedBuffer` hands back the screen of
    // whatever was on display before this applet. Booted alone, there is no
    // such screen, but "nothing written, slot -1" is not how to say so:
    // `nnSdk` reads it as not-ready-yet and asks again, and the Album applet
    // spent every frame of a 300M-instruction run in that retry loop without
    // ever reaching a draw.
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
        // The slot named is past the ones `AcquireSharedFrameBuffer` hands
        // out, so the black it claims to be stays black.
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
    // The same black screen the three `Acquire`s hand out as a slot, asked
    // for as pixels instead: a 1280x720 RGBA8888 image in a map-alias out
    // buffer. Nothing was captured, and leaving the buffer alone while
    // reporting one was written hands the applet whatever it had in there to
    // draw. Refusing the command instead aborted `nnSdk` outright.
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
        // And only the buffer: the clear walks a page at a time, so an
        // overrun would land on the page after the one it ends in.
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
    // GetCallerAppletIdentityInfoStack walks up the chain of applets that
    // launched this one. Nothing here launched it, so the chain above it is
    // the menu and nothing else -- the same identity 12 and 14 answer with.
    // The count has to fit the buffer the caller sized: one that overruns it
    // is worse than a short one.
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

    // A buffer with no room for an entry gets a count of zero, not one that
    // names an entry the caller has nowhere to read.
    ipc_request_plain_with_buffer(&mut cpu, APPLET, 17, STACK, ENTRY - 1, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0, "refused");
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0, "entries written");
}

#[test]
fn a_library_applet_is_told_which_keyboard_layout_to_open_with() {
    // GetDesirableKeyboardLayout is the layout the applet's caller asked it
    // to open with, and hardware errors when no caller set one. There is no
    // caller here, so it answers with the layout that goes with the language
    // `set` reports -- en-US, so EnglishUs.
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
    // `audout` is the plain PCM-out device, and the whole interface is the
    // buffer protocol: append a buffer, wait on the event, collect the tags of
    // the buffers the device is done with. A device that accepts buffers and
    // never releases them hangs the guest's audio thread forever.
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
    // IAudioOut as a *move* handle.
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
    // { send_pid:1, num_copy:4, num_move:4 }: one move handle, no copy ones.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(device, 0, "no IAudioOut came back");

    // RegisterBufferEvent: an event, and events are *copy* handles.
    ipc_request_plain(&mut cpu, device, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0);
    // Nothing has been played, so nothing has been released.
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

    // The samples reached the host, unchanged, at full volume. That happens on
    // arrival: it is the *tag* that waits for the device, not the audio.
    let mut played = [0i16; 8];
    assert_eq!(cpu.take_audio(&mut played), 8);
    assert_eq!(played, samples);

    // The buffer is not back yet, because the device has not finished playing
    // it. Four stereo frames at 48 kHz take 4/48000 of a second, which is
    // 85,000 of the 1.02 GHz cycles one emulated instruction stands for.
    // Releasing on arrival is what let Just Dance 2019 run its audio clock at
    // 205x real time and drop every frame of its boot video.
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

    // Give the device that long. A branch-to-self is the cheapest way to spend
    // the cycles, and spending them is the point: the clock is the instruction
    // count.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    // Now it comes back: the event fires and the tag is collectable.
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0,
        "the played buffer did not fire"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);

    // GetAudioOutPlayedSampleCount counts frames, not samples: four stereo
    // frames, not eight.
    ipc_request_plain(&mut cpu, device, 10, &[]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 4);

    // And the host is told what to play it at.
    assert_eq!(cpu.audio_format(), (48_000, 2));
}

#[test]
fn audout_release_zeroes_the_entry_after_the_last_tag() {
    // `nn::audio`'s wrapper around `GetReleasedAudioOutBuffer` points the
    // receive buffer at an uninitialised stack slot and returns that slot
    // without ever reading the count. So a release that hands back nothing has
    // to leave a zero there, or the caller takes whatever the last call left
    // on the stack for an `AudioOutBuffer`. The Album applet's audio thread
    // took a `bl`'s return address, and de-interleaved its samples over the
    // `.text` the address pointed into.
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

    // Nothing has played yet, so the release is empty, and the slot the guest
    // reads regardless has to say so.
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

    // Once the device is done with it the tag lands in the first slot, and the
    // zero moves along to the one after it.
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
    // `nnSdk` reaches this through `GetReleasedAudioOutBufferAuto`, which
    // offers the out buffer as a receive-static and leaves the map-alias
    // descriptor null. Reading only the map-alias one wrote the reply to
    // address 0, so the caller kept the uninitialised stack slot it points at:
    // "A Short Hike"'s mixer took a `bl`'s return address for an
    // `AudioOutBuffer` and stored its samples over its own `.text`.
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

    // Nothing has played yet, so the terminator has to reach the guest's own
    // slot rather than address 0.
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

    // And once the device is done with it, so does the tag.
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
    // `RequestUpdateAudioRenderer` runs every frame, and both `audrvUpdate`
    // and `nnSdk` walk its reply section by section against sizes they
    // computed themselves: a section left out is not ignored, it desynchronises
    // the walk and aborts. Tomodachi Life opens a revision-15 renderer with 17
    // effects; the reply had no effects section and no renderer info, and its
    // audio setup ended the boot on an `nn::audio` result.
    const AUDREN: u64 = 0xB000;
    const OUT: u32 = 0x9000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDREN, "audren:u");
    let tls = cpu.tls_base();

    // `AudioRendererParameter`: voices at +16, sinks at +20, effects at +24
    // and the revision magic at +48.
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

    // One `MemPoolInfoOut` per mempool (effects + four per voice), one
    // `VoiceInfoOut` per voice, one revision-9 `EffectOutStatus` per effect,
    // one `SinkInfoOut` per sink, then the performance, behaviour and
    // renderer-info tails.
    assert_eq!(section(&cpu, 0x08), (3 + 4 * 2) * 16, "mempools");
    assert_eq!(section(&cpu, 0x0c), 2 * 16, "voices");
    assert_eq!(section(&cpu, 0x14), 3 * 0x90, "effects");
    assert_eq!(section(&cpu, 0x1c), 32, "sinks");
    assert_eq!(section(&cpu, 0x20), 16, "performance");
    assert_eq!(section(&cpu, 0x04), 176, "behaviour");
    assert_eq!(section(&cpu, 0x28), 16, "renderer info");
    let total = 64 + 176 + 32 + 3 * 0x90 + 32 + 16 + 176 + 16;
    assert_eq!(section(&cpu, 0x3c), total, "total size");

    // Before revision 5 there is no renderer info at all, and an effect's
    // status is the narrow form.
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
    // The renderer is where retail audio actually lives: `nn::audio` and
    // libnx's `audrv` hand it wave buffers, a pitch and a routing matrix, and
    // expect mixed PCM out the far end. It used to answer every update with a
    // correctly shaped, entirely zeroed reply, the right size for the caller
    // to accept and no sound whatsoever.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const FRAMES: u32 = 240;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    // A ramp: every sample differs from its neighbours, so an off-by-one in
    // the resampler shows up as a shift rather than as plausible noise.
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

    // One frame of emulated time, and one frame is what comes out: the
    // renderer produces what the clock says has come due and not a sample
    // more, which is the same rule `AudioOut::free_at` follows and the reason
    // a title's audio clock runs at 1x rather than at whatever multiple of
    // real time the emulator manages.
    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; FRAMES as usize * 2];
    assert_eq!(
        cpu.take_audio(&mut played),
        played.len(),
        "the mix never reached the host"
    );
    assert_eq!(cpu.audio_format(), (48_000, 2));
    // The voice is mono into both mix buffers and the sink reads one into each
    // output channel, so the frame is the source doubled up, and it is
    // bit-exact, because a 16-bit source at unity gain has no arithmetic done
    // to it that it should not survive.
    for (i, &s) in samples.iter().enumerate() {
        assert_eq!(played[i * 2], s, "left channel at sample {i}");
        assert_eq!(played[i * 2 + 1], s, "right channel at sample {i}");
    }

    // And nothing is queued twice: a second update with no time elapsed
    // renders no further frames.
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
    // `num_wavebufs_consumed` is the load-bearing number in the reply: the
    // guest advances its own ring head by the delta and refills only the
    // buffers this has accounted for. A renderer that reports zero is one
    // whose title queues four buffers, waits for one back, and stops.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const FRAMES: u32 = 240;
    /// The reply's voice section: past the header and one `MemPoolInfoOut`
    /// per mempool, of which there are four per voice.
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

    // Exactly one frame of samples, so the buffer is played out exactly.
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
    // Nintendo's 4-bit ADPCM is what retail voices are stored as, 14 samples
    // in every 8 bytes, one header byte naming a shift and one of eight
    // predictor pairs, then seven bytes of nibbles. A renderer that decodes
    // only PCM is silent on almost everything that ships.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const DATA: u32 = 0x5_0000;
    const COEFS: u32 = 0x5_1000;
    const SAMPLES: u32 = 28;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    // Two coefficient pairs, chosen so the arithmetic is checkable by hand:
    // pair 0 predicts nothing, so a sample is its own nibble; pair 1 is 1.0 in
    // the predictor's Q11, so a sample is its nibble plus the one before it.
    let coefficients: [i16; 16] = [0, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, &c) in coefficients.iter().enumerate() {
        cpu.mem.write_u16(COEFS + i as u32 * 2, c as u16).unwrap();
    }

    // Frame 0: pair 0, shift 0, nibbles 1..7 then -8..-2.
    // Frame 1: pair 1, shift 0, every nibble 1, a running +1 from the -2 the
    // first frame ended on.
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
    // Past the end of the wave buffer the voice interpolates down to silence
    // rather than holding its last sample, which would leave a DC step behind.
    assert_eq!(
        played[expected.len() * 2],
        0,
        "the voice kept playing past its data"
    );
}

#[test]
fn audren_frame_event_fires_on_the_clock() {
    // `audrenWaitFrame` blocks on this event, and every mixer built on it
    // paces itself by how often it comes back. The handle used to be a bare
    // one, not modelled as an event at all, so `WaitSynchronization` treated
    // it as permanently ready and the wait returned instantly. That is a
    // renderer with no clock, which is how a title ends up running its audio
    // at whatever multiple of real time the emulator manages.
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);
    let tls = cpu.tls_base();

    // QuerySystemEvent. Events are *copy* handles: sent as a move handle it
    // reads back as 0 on the other side.
    ipc_request_plain(&mut cpu, renderer, 7, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x08).unwrap(),
        1 << 1,
        "not a copy handle"
    );
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0, "no frame event came back");

    // No time has passed, so the frame is not due.
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "the frame event fired early"
    );

    // Five milliseconds of it, and it is.
    cpu.cycles += AUDREN_FRAME_CYCLES;
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0,
        "the frame event never fired"
    );
}

#[test]
fn audren_refuses_a_wave_buffer_that_is_outside_its_allocation() {
    // `end_sample_offset` is the guest's claim about its own buffer and `size`
    // is what it allocated. Where they disagree the allocation wins, because
    // the samples past it are somebody else's memory read as PCM, which is
    // exactly the buzzing `audout` produced from the Mii editor's descriptor
    // until it started checking.
    //
    // The buffer is still consumed: the guest is entitled to it back however
    // unplayable it was, and a buffer that never comes back stalls the voice
    // that queued it.
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
    // 240 samples claimed out of a buffer with room for none of them.
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
    // `data_offset + data_size` has to fit inside `buffer_size`. The Mii
    // editor submits one where it does not: `buffer` is 6 and `data_offset`
    // is a pointer, and `buffer + data_offset` then lands inside the
    // `AudioOutBuffer` struct itself, so what reached the speakers was that
    // struct's own pointers read as PCM, thousands of times a second. It
    // buzzed.
    //
    // The buffer still comes back to the guest; only its samples are dropped.
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

    // Something the device could play, sitting where the bad descriptor's
    // arithmetic would land, so a failure to check would be audible rather
    // than silent.
    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    // buffer_size says 8 bytes; data_offset alone is already past it.
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

    // And the guest gets its buffer back, so its audio thread does not stall
    // waiting for one it will never see again.
    let cycles = 200_000;
    for _ in 0..cycles {
        let _ = cpu.step();
    }
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
    // `OpenAudioOut` takes the channel count as a 16-bit field, and the two
    // bytes above it are padding the caller never writes. Reading the whole
    // word and echoing it back handed `nnSdk` a channel count of 0xcafe0002 --
    // negative, so its own "channelCount > 0" check failed and the title tore
    // its audio down and re-opened, which aborts.
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
    // A device that has not been started is not playing. Its buffers still
    // come back -- the memory is the guest's -- but nothing is queued for the
    // host, because nothing was heard.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;

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
    // The tag still comes back.
    ipc_request_plain(&mut cpu, device, 9, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1);
}

#[test]
fn the_binder_transacts_on_the_command_a_pre_3_0_0_sdk_sends() {
    // `IHOSBinderDriver` has two transactions that do the same work and differ
    // only in how the parcel is marshalled: `TransactParcel` (0), which takes
    // map-alias buffers, and `TransactParcelAuto` (3), which takes auto-select
    // ones and arrived in 3.0.0. An SDK older than that sends 0 and only 0.
    //
    // Only 3 was implemented, so 0 fell to the answer `vi_unhandled` gives a
    // void setter: an empty success, with nothing written into the reply
    // buffer. Just Dance 2017 -- built against a 2016 SDK -- drove its whole
    // buffer queue through it, 402 transactions in three billion instructions,
    // and presented **no frame at all**: every `QUEUE_BUFFER` was accepted and
    // discarded, so `Action::Present` never reached the GPU.
    const VI: u64 = 0xB800;
    const PARCEL: u32 = 0x9000;
    const REPLY: u32 = 0x9400;
    /// `NATIVE_WINDOW_WIDTH`, the field `QUERY` is being asked for here.
    const QUERY_WIDTH: u32 = 0;
    const QUERY: u32 = 9;

    // Both commands have to answer identically -- that is the whole claim.
    for cmd in [0u32, 3] {
        let mut cpu = cpu_at(0x1000);
        cpu.bootstrap();
        cpu.set_pc(0x1000);
        cpu.register_service_handle(VI, "vi:m");
        let tls = cpu.tls_base();

        // The relay lives two objects down: vi root -> IApplicationDisplayService
        // (2) -> IHOSBinderDriver (100).
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

        // `{ s32 binder_id, u32 code, u32 flags }`, the parcel in the send
        // buffer and the reply into the receive buffer.
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
        // The reply parcel: `{ i32 value, i32 status }` behind the same
        // four-word header the request carries.
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
    // `OpenLayer` answers with an Android parcel holding one flattened binder
    // object. libnx only reads the binder id out of it; nnSdk also checks the
    // interface name, and rejected the whole layer -- vi result 114-1, an
    // abort inside nn::vi::CreateLayer -- while the parcel carried a bare id
    // and nothing else.
    const VI: u64 = 0xB000;
    const WINDOW: u32 = 0x8000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetDisplayService first: OpenLayer lives on the
    // IApplicationDisplayService, not on the vi root.
    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(display, 0, "no IApplicationDisplayService");

    // OpenLayer, with the 0x100-byte native-window receive buffer the caller
    // always provides.
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
    // A GPIO pad is one wire into the SoC, and nothing is wired to this
    // console. The level an undriven pad reads is not cosmetic: the buttons
    // are active-low, and boot2 reads the two volume pads and enters
    // maintenance mode when *both* read Low. Answering 0 here boots the
    // console into maintenance mode on every single launch.
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
    // { send_pid:1, num_copy:4, num_move:4 }: the session is a move handle.
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

    // GetInterruptStatus: nothing drives the pad, so nothing is pending.
    ipc_request_plain(&mut cpu, pad, 6, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0);
}

#[test]
fn a_fabricated_reply_fills_both_handle_slots() {
    // The reply for a command nothing implements used to carry only a raw
    // object id in its data. On a plain session that is not where an
    // out-object lives -- nnSdk reads one as a move handle, and a reply
    // carrying none is not an error to it: the handle parses as 0, the client
    // skips constructing the proxy, and the command still returns *success*.
    // The caller then makes its first virtual call through a null pointer,
    // which is how boot2 reached pc=0 one instruction after `gpio`'s
    // OpenSession2 was answered "successfully".
    //
    // An out-*event* is the same trap in the other handle slot, and nothing
    // here knows which of the two an unimplemented command was meant to
    // return -- so the reply carries one of each. That is what the Home Menu's
    // message thread was missing when it settled into waiting on handle 0,
    // three created-but-never-started threads behind it.
    const NCM: u64 = 0x9200;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(NCM, "ncm");
    let tls = cpu.tls_base();

    // IContentManager::OpenContentStorage(StorageId) -> IContentStorage.
    ipc_request_plain(&mut cpu, NCM, 4, &[1, 0, 0, 0]);
    // { send_pid:1, num_copy:4, num_move:4 }: one of each. Copy handles come
    // first in the reply, so the event is at +0x0c and the object at +0x10,
    // and the raw section starts at the next 16-byte boundary after them.
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

    // The sub-session reaches the same service, so a command on it is
    // dispatched rather than falling through as an untracked handle.
    let handles = cpu.service_handles_snapshot();
    assert!(handles
        .iter()
        .any(|(h, name)| *h == storage && name == "ncm"));

    // The event is real and quiet: a caller that waits on it is waiting for
    // something that never happens, which is the truth, rather than acting on
    // something that never will.
    assert_eq!(wait_sync(&mut cpu, &[event as u32], 0).0, 0xEA01);

    // Asked again, the same pair comes back: a guest polling a command nothing
    // implements must not be handed fresh handles every call.
    ipc_request_plain(&mut cpu, NCM, 4, &[1, 0, 0, 0]);
    assert_eq!(u64::from(cpu.mem.read_u32(tls + 0x10).unwrap()), storage);
    assert_eq!(u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap()), event);
}

#[test]
fn the_home_menu_opens_a_system_applet_proxy() {
    // qlaunch is neither an application nor a library applet. It is the one
    // process that outlives every title, and it declares that by opening
    // IAllSystemAppletProxiesService command 100 -- then R_ABORT_UNLESSes on
    // the spot if the answer is an error. 2010-0221, `cmif`'s "unknown command
    // id", is exactly what this stub used to reply, so the Home Menu died on
    // its first applet call with an svcBreak and nothing else to go on.
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

    // GetHomeMenuFunctions, which only this proxy exposes -- a library applet
    // reaches the same interface at 22, and an application not at all.
    ipc_request_plain(&mut cpu, proxy, 20, &[]);
    let home = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(home, 0);
    // IsSleepEnabled -> bool: what an unrestricted retail console reports.
    ipc_request_plain(&mut cpu, home, 40, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 1);
    // GetHomeButtonWriterLockAccessor -> ILockAccessor, and it must be a real
    // session for the same reason every other out-object must be.
    ipc_request_plain(&mut cpu, home, 30, &[]);
    let lock = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(lock, 0, "no ILockAccessor came back");
    ipc_request_plain(&mut cpu, lock, 4, &[]); // IsLocked
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 0);

    // GetGlobalStateController: ShouldSleepOnBoot is false, because a console
    // that was slept rather than shut down would resume straight back to
    // sleep, and this one always boots awake.
    ipc_request_plain(&mut cpu, proxy, 21, &[]);
    let global = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(global, 0);
    ipc_request_plain(&mut cpu, global, 14, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 0);

    // The sleep and shutdown sequences stay refused: a console that answers
    // "done" to a shutdown it did not perform is worse than one that refuses.
    const UNKNOWN_COMMAND_ID: u32 = 10 | (221 << 9);
    ipc_request_plain(&mut cpu, global, 3, &[]); // StartShutdownSequence
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), UNKNOWN_COMMAND_ID);
}

#[test]
fn the_display_answers_what_it_is() {
    // `ListDisplays` and `ListDisplayModes` used to fall through to `vi`'s
    // catch-all, which answers with an empty success. That is not "no
    // displays" -- a reply's declared raw section is four words of padding
    // wide, so the caller read its count out of whatever the *request* had
    // left in those bytes and then walked an out-buffer nothing had written.
    // The Home Menu spent a billion instructions in that walk without making
    // a single syscall, and there was nothing in any trace to say where it had
    // gone.
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

    // OpenDisplay takes that name and hands back the id every later display
    // command carries.
    let mut open = [0u8; 0x40];
    open[..7].copy_from_slice(b"Default");
    ipc_request_plain(&mut cpu, display, 1010, &open);
    let display_id = cpu.mem.read_u64(tls + 0x20).unwrap();
    assert_ne!(
        display_id, 0,
        "a display id of 0 is the no-display sentinel"
    );

    // GetDisplayResolution, on the same interface, has to agree with it.
    ipc_request_plain(&mut cpu, display, 1102, &display_id.to_le_bytes());
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 1280);
    assert_eq!(cpu.mem.read_u64(tls + 0x28).unwrap(), 720);

    // ListDisplayModes lives on ISystemDisplayService: one
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
    // `nifm:u`, `nifm:s` and `nifm:a` are one interface at three privilege
    // levels, and only the first was routed to the implementation -- so a
    // system title, which opens `nifm:s`, had every network call answered by
    // the generic fallback instead.
    //
    // The command ids were crossed underneath that: 12 answered with the
    // connection-status triple and 15 with the IP address, when 12 *is*
    // GetCurrentIpAddress and 18 is GetInternetConnectionStatus. A caller
    // asking this console for its own address got `{2, 0, 2}` back.
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

    // A request on a link that is up is accepted the moment it is made, and
    // the two events a caller waits on for that have already happened.
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
                                                  // { send_pid:1, num_copy:4, num_move:4 }: **two** copy handles. One left
                                                  // the caller holding a session for the second, and a bare success left it
                                                  // holding 0 for both.
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
    // The control commands belong to the session, not to whatever is behind
    // it, so a service with no dedicated stub still has to answer them itself.
    // The generic fallback used to hand them the same fabricated object id it
    // hands every other command -- and as a *pointer buffer size* that is a
    // large number, which is how a caller decides to marshal its buffers as
    // pointer buffers, the one form this IPC layer does not read. `friend`,
    // `olsc`, `prepo` and `btm` were all being told to send their data
    // somewhere nothing looks.
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
        // And no handle came with it: a size is not an object.
        assert_eq!(
            cpu.mem.read_u32(tls + 0x04).unwrap() >> 31,
            0,
            "type {msg_type}: handles"
        );
    }

    // ConvertToDomain is the other control command, and it *does* answer with
    // an object id -- the one the session's later requests carry.
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
    // The vsync event used to be fired by one thing only: the guest's own
    // present. That is a circle a title never gets into, because it waits for
    // vsync *before* it renders the frame that would have fired it. A real
    // panel refreshes whether or not anything drew, so the event fires on a
    // period as well -- and a present still fires it, so a guest that draws
    // faster than the panel is not held to it.
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
    let vsync = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(vsync, 0);

    // The period has not passed yet, and nothing has been presented.
    assert_eq!(
        wait_sync(&mut cpu, &[vsync], 0).0,
        RESULT_TIMED_OUT,
        "vsync fired early"
    );

    // Run out the refresh period on nops. The event fires on its own, with no
    // frame behind it, and being auto-clearing it fires once per period.
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
}

#[test]
fn closing_a_domain_object_is_not_command_zero() {
    // `CmifDomainRequestType_Close` sits where SendMessage's type byte would,
    // and carries no command id at all -- so a close dispatched to a service
    // is read as command 0, which on most interfaces is a real operation. The
    // Home Menu's `IStorage` close ran as a **Read**, with the reply's own
    // "SFCO" magic for an offset, and left the object open behind it.
    const FS: u64 = 0xC000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(FS, "fsp-srv");
    let tls = cpu.tls_base();

    // Convert to a domain, then open the process's own RomFS as an IStorage.
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

    // A close, marshalled the way a caller marshals one: the domain header's
    // type byte is 2 and there is no CmifInHeader behind it.
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
    // Control-ness is `ipc_is_control_request`, never `type == 5`: a control
    // message has a with-context encoding too (type 7), and that is the one
    // nnSdk sends. Testing for 5 alone read the Home Menu's
    // QueryPointerBufferSize as command **3 on the binder relay** and ran a
    // parcel transaction for it, answering a size query with a failed binder
    // reply.
    //
    // The size is the same either way: it is the session's, not the
    // interface's, and it is answered before the request reaches `vi` at all.
    const VI: u64 = 0xB400;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    for msg_type in [5u32, 7] {
        build_ipc_request(&mut cpu, msg_type, None, 3);
        run_ipc_request(&mut cpu, VI);
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
    }

    // And ConvertToDomain, the other control command, still hands back an
    // object id rather than being read as a binder AdjustRefcount.
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
    // `svcResetSignal` is `nn::os::TryWaitSystemEvent`: it clears a signalled
    // event and fails if there was nothing to clear. Succeeding
    // unconditionally told every guest that every event it ever polled had
    // fired, so a loop that drains a queue while its event keeps signalling
    // had no reason to stop.
    const APPLET: u64 = 0x9800;
    const RESULT_INVALID_STATE: u64 = 1 | (125 << 9);
    let (mut cpu, applet, _proxy, state_getter) = applet_chain();
    let _ = APPLET;
    let tls = cpu.tls_base();

    // The applet message event starts signalled: AM has the startup focus
    // transition waiting.
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 0); // GetEventHandle
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
    // The Home Menu and the system's own applets do not render into a layer of
    // their own. AM shares one buffer between them: the applet asks for a slot
    // in it, draws there, and presents the slot back. The whole path turns on
    // `IsSystemBufferSharingEnabled` succeeding -- refuse that and qlaunch
    // builds a swapchain instead and never draws one triangle into it.
    const VI: u64 = 0xB500;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetSharedBufferMemoryHandleId -> the nvmap handle the applet maps the
    // buffer by, and how big it is. The buffer is the system's, so this is
    // where it comes into being; nothing in the guest ever created it.
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

    // AcquireSharedFrameBuffer -> an empty fence, the slots that exist, and
    // the one to draw into. Two slots exist and they alternate; handing out
    // the same one twice would have the applet overwrite the frame the display
    // is still scanning.
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
    // A reply is written *over* the request, in the same TLS buffer, and the
    // padding its header declares is four words wide -- room for a small out
    // parameter. So a command answered with a bare success never handed the
    // caller nothing: it handed the caller stale request bytes, in a reply
    // whose declared size passes every length check nnSdk and libnx make.
    const VI: u64 = 0xB200;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    // CloseDisplay: void, and nothing implements it. Poison the bytes an out
    // parameter would be read from before sending.
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
fn the_vsync_event_is_a_copy_handle_on_a_plain_session() {
    // GetDisplayVsyncEvent on a session that never became a domain used to
    // hand back a bare handle in the *move* slot and register nothing. A copy
    // handle read out of the move slot is 0, and there was no event behind it
    // for a present to fire -- so a render loop paced by vsync waited on
    // handle 0 forever, and only kept running because a wait on an unknown
    // handle is answered as satisfied.
    const VI: u64 = 0xB300;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    ipc_request_plain(&mut cpu, VI, 2, &[]);
    let display = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    ipc_request_plain(&mut cpu, display, 5202, &[]);
    // { send_pid:1, num_copy:4, num_move:4 }: one copy handle, no move ones.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let vsync = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(vsync, 0, "the guest must receive a real handle");

    // It is a real event, and quiet until the display advances.
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    assert_eq!(wait_sync(&mut cpu, &[vsync], 0).0, RESULT_TIMED_OUT);
    cpu.signal_event(u64::from(vsync));
    assert_eq!(
        wait_sync(&mut cpu, &[vsync], 0).0,
        0,
        "a signalled vsync did not fire"
    );
}

#[test]
fn the_applet_message_event_starts_signalled_and_clears() {
    // An applet does not draw until it has been told it is in focus, and it
    // asks by polling this event with a zero timeout rather than by calling
    // ReceiveMessage. Left dark, the one message AM queues at startup sat
    // there with nothing ever coming to collect it: the Mii editor idled in
    // `appletMainLoop` with a dequeued buffer in hand and not one draw behind
    // it. The event is auto-clearing, so the first poll takes the message and
    // every later one times out.
    const APPLET: u64 = 0x9400;
    const RESULT_TIMED_OUT: u64 = 0xEA01;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletOE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy_service), 0);
    let proxy = cpu.mem.read_u32(tls + 0x30).unwrap();
    ipc_request(&mut cpu, APPLET, 4, Some(proxy), 0); // ICommonStateGetter
    let state_getter = cpu.mem.read_u32(tls + 0x30).unwrap();

    ipc_request(&mut cpu, APPLET, 4, Some(state_getter), 0); // GetEventHandle
    assert_eq!(
        cpu.mem.read_u32(tls + 0x08).unwrap(),
        1 << 1,
        "events are copy handles"
    );
    let message = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_eq!(
        wait_sync(&mut cpu, &[message], 0).0,
        0,
        "the queued message never announced itself"
    );
    assert_eq!(
        wait_sync(&mut cpu, &[message], 0).0,
        RESULT_TIMED_OUT,
        "it announced itself twice"
    );

    // And the message behind it is the focus change.
    const FOCUS_STATE_CHANGED: u32 = 15;
    // A domain reply with no handles carries its CmifDomainOutHeader first,
    // so the result and the data sit 0x10 further in than on a plain session.
    ipc_request(&mut cpu, APPLET, 4, Some(state_getter), 1); // ReceiveMessage
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "no message was queued"
    );
    assert_eq!(cpu.mem.read_u32(tls + 0x30).unwrap(), FOCUS_STATE_CHANGED);
}

#[test]
fn an_applet_is_told_it_came_into_the_foreground_not_that_focus_changed() {
    // `FocusStateChanged` is the *application's* message. An applet -- every
    // one of the system's own, the Home Menu included -- is told
    // `ChangeIntoForeground` instead, and which one it gets is decided by
    // which proxy it opened. Sending an applet the application's message is
    // sending it one its own framework does not act on.
    const CHANGE_INTO_FOREGROUND: u32 = 1;
    const APPLET: u64 = 0x9700;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(APPLET, "appletAE");
    let tls = cpu.tls_base();

    ipc_request(&mut cpu, APPLET, 5, None, 0);
    let proxy_service = cpu.mem.read_u32(tls + 0x20).unwrap();
    // IAllSystemAppletProxiesService::OpenSystemAppletProxy -- what qlaunch
    // opens, and what says it is not an application.
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
    // `SetHandlesRequestToDisplay(true)` is an applet saying it will decide
    // when it appears. AM answers by queueing `RequestToDisplay`, and the
    // applet draws nothing until it has read that message and approved
    // itself. With only the startup focus change to hand out, the Home Menu
    // finished its layer, preallocated both swapchain buffers and then ran its
    // frame loop for thirty seconds of console time without ever dequeuing
    // one.
    const REQUEST_TO_DISPLAY: u32 = 41;
    const FOCUS_STATE_CHANGED: u32 = 15;
    const NO_MESSAGES: u32 = 128 | (3 << 9);
    let (mut cpu, applet, proxy, state_getter) = applet_chain();
    let tls = cpu.tls_base();
    ipc_request(&mut cpu, applet, 4, Some(proxy), 1); // GetSelfController
    let self_controller = cpu.mem.read_u32(tls + 0x30).unwrap();

    // Drain the one message AM has waiting before the applet's first poll.
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

    // Once, not on every poll -- the mistake that made JKSV re-process a focus
    // change every frame.
    ipc_request(&mut cpu, applet, 4, Some(state_getter), 1);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        NO_MESSAGES,
        "it was queued twice"
    );

    // And the approval that follows is accepted. It used to reach
    // `unimplemented_command`, which `nnSdk` answers with an svcBreak.
    ipc_request(&mut cpu, applet, 4, Some(self_controller), 51);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x28).unwrap(),
        0,
        "ApproveToDisplay was refused"
    );
}

#[test]
fn parental_control_hands_out_the_events_it_is_asked_for() {
    // The Home Menu opens `pctl`, asks IParentalControlService for its
    // synchronisation event, and R_ABORT_UNLESSes on the answer. Refusing the
    // command -- 2010-0221, `cmif`'s "unknown command id" -- killed it there
    // with an svcBreak, one applet call into its own boot. There is no
    // guardian account to synchronise with, so the event is real and never
    // fires, which is the true state rather than a placeholder for one.
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

    // Nothing restricts anything here, and the two families of query read in
    // opposite directions: "is something restricting you" is false, "is
    // something still allowed" is true.
    ipc_request_plain(&mut cpu, service, 1031, &[]); // IsRestrictionEnabled
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 0);
    ipc_request_plain(&mut cpu, service, 1458, &[]); // IsPlayTimerAlarmDisabled
    assert_eq!(cpu.mem.read_u8(tls + 0x20).unwrap(), 1);
}

#[test]
fn the_vibration_device_list_is_a_hid_session_not_a_fabricated_object() {
    // `IHidServer::CreateActiveVibrationDeviceList` hands back a sub-session,
    // and `nn::hid::InitializeVibrationDevice` calls command 0 on it once per
    // motor. That name was missing from the session router, so every one of
    // those calls fell through to the fabricated-object fallback -- which
    // answers a command whose whole reply is a Result with an object id and
    // two handles nobody asked for, and which said so as
    // "[ipc] no implementation: hid:vibration-devices" on a real title's boot.
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
    // RegisterProcessHandle (cmd 4) is the first call `nn::ro::Initialize`
    // makes, and the one that reported `ldr:ro` as having no implementation at
    // all. The generic fallback answers it with an object id, a sub-session
    // and an event: for a command that returns nothing but a Result.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x10).unwrap(), 0x4F43_4653); // "SFCO"
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    // No handle descriptor: bit 31 of the second header word is what says a
    // reply carries handles, and this reply has none to carry.
    assert_eq!(cpu.mem.read_u32(tls + 4).unwrap() >> 31, 0);
}

#[test]
fn ldr_ro_maps_a_module_where_nothing_else_lives() {
    // LoadModule has to actually map the NRO: the caller relocates against the
    // address returned here and then jumps into it, so an address with no
    // image behind it is a branch into whatever the region happened to hold.
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

    // The three segments, in the order the file has them, and the BSS behind
    // them: zero-filled, whatever the caller's own buffer held.
    assert_eq!(cpu.mem.read_u32(base).unwrap(), 0x1400_0010);
    assert_eq!(cpu.mem.read_u8(base + 0x1000).unwrap(), 0xAA);
    assert_eq!(cpu.mem.read_u8(base + 0x2000).unwrap(), 0xBB);
    assert_eq!(cpu.mem.read_u8(base + 0x3000).unwrap(), 0);

    // `.text` is read-execute, the way a real kernel maps it. Its `.data` is
    // not: that is where the relocations the caller is about to apply land.
    assert!(cpu.mem.write_u32(base, 0).is_err());
    assert!(cpu.mem.write_u32(base + 0x2000, 0).is_ok());
}

#[test]
fn ldr_ro_unload_frees_the_address_space_and_the_protection() {
    // A module that has been unloaded has to leave nothing behind: not the
    // pages, and not the read-only marking on its `.text`, which would
    // outlive the mapping and fault whatever is loaded over it next.
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    let base = cpu.mem.read_u64(tls + 0x20).unwrap();
    ldr_ro_request(&mut cpu, handle, 1, &[base]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), 0);
    assert!(cpu.mem.write_u32(base as u32, 0).is_ok());

    // And the address space is free again, so the next load reuses it rather
    // than walking up the region.
    ldr_ro_request(&mut cpu, handle, 0, &[NRO_SOURCE, 0x3000, NRO_BSS, 0x1000]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), base);

    // Unloading something that was never loaded is `ro`'s NotLoaded, not a
    // success the caller then treats as a freed module.
    const NOT_LOADED: u32 = 22 | (1028 << 9);
    ldr_ro_request(&mut cpu, handle, 1, &[0x2800_0000]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), NOT_LOADED);
}

#[test]
fn ldr_ro_two_modules_do_not_overlap() {
    // The second module has to go behind the first, BSS included: `ro` hands
    // out one address space between every module a title loads.
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
    // A bad NRO is refused rather than mapped: the caller jumps into what this
    // command returns, so "success" over an image with no segment table is a
    // branch into nothing. Same for a BSS the caller sized too small, the
    // module's zero-initialized data would land past the mapping, on whatever
    // is loaded next.
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
    // An NRR cannot be verified here: there is no key to check its signature
    // chain against, but it can be *tracked*, so unregistering one that was
    // never registered is an error rather than a success the caller believes.
    const INVALID_NRR: u32 = 22 | (6 << 9);
    const NOT_REGISTERED: u32 = 22 | (1029 << 9);
    const NRR: u64 = 0x1020_0000;
    let (mut cpu, handle) = ldr_ro_session();
    let tls = cpu.tls_base();

    ldr_ro_request(&mut cpu, handle, 3, &[NRR]);
    assert_eq!(cpu.mem.read_u32(tls + 0x18).unwrap(), NOT_REGISTERED);

    // Nothing is at that address yet, so there is no NRR to register.
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
    // The operation mode is not one service's opinion: `am` reports it,
    // `am` and `apm` both derive the performance mode from it, `vi` sizes the
    // display by it, `clkrst` clocks the GPU by it, and the touchscreen only
    // exists on one side of it. A title that picks its render target from one
    // of those and scans out through another draws at the wrong scale, so what
    // is pinned here is that they move together.
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

    // The startup focus message, out of the way, so what is left in the queue
    // below is the dock's doing and nothing else.
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

    // And the title is *told*, which is the half that makes it act: it read
    // the mode once at startup and laid out for that answer. Without these it
    // never goes back to ask, and the new number is one nobody reads.
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

    // GetDefaultDisplayResolution (60) is the answer that sits *beside* the
    // mode on a title's own screen, so the two coming from different places
    // is how NX-Fetch came to print "1280x720 @ 60Hz [Docked]".
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

    // Docking a docked console is not a transition. AM does not announce one
    // that did not happen, and a title told to re-lay-out does the work
    // whether or not anything changed.
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

    // Touch is handheld-only: the screen is in the dock, so nothing can be on
    // it. The sample is still published: a LIFO that stops advancing is not
    // "no touches" to a reader waiting for the next one, but it carries none.
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
    // The pool layout is a promise: it goes out once, at
    // GetSharedBufferMemoryHandleId, and the applet maps it and renders into
    // it for as long as it holds it. Sizing it by the display broke that
    // promise on the dock, every slot in the pool moved while qlaunch was
    // still drawing into the old ones, and the present that followed read
    // from the wrong offset at the wrong pitch. Thirteen frames after a dock
    // the Home Menu was 0 of 2073600 pixels non-black, with the guest drawing
    // perfectly well.
    //
    // Nor did the larger pool buy anything: qlaunch lays its UI out at
    // 1280x720 whatever `vi` reports, so the docked frame was the undocked
    // frame at the origin, to the pixel, and black across the rest.
    use switch_core::cpu::{OperationMode, SHARED_BUFFER_GEOMETRY};
    const VI: u64 = 0x2000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(VI, "vi:m");
    let tls = cpu.tls_base();

    // GetSharedBufferMemoryHandleId reports the pool's total size, and per
    // slot an offset, a size and the slot's width and height.
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

    // The display did move, though: the pool is the shared layer's geometry
    // and the display is the panel's, and the two are no longer the same
    // number.
    assert_eq!(cpu.operation_mode().display_size(), (1920, 1080));
    assert_eq!(SHARED_BUFFER_GEOMETRY.display_size(), (1280, 720));

    // Rows round up to a 128-row block-linear block: 720 -> 768, 1080 -> 1152.
    assert_eq!(OperationMode::Handheld.shared_buffer_rows(), 768);
    assert_eq!(OperationMode::Docked.shared_buffer_rows(), 1152);
}

#[test]
fn the_resolution_change_event_fires_on_the_dock() {
    // `GetDefaultDisplayResolutionChangeEvent` is how a title that is not
    // polling AM's message queue finds out to go and re-read the resolution.
    // It used to be a fresh event per caller, handed out dark, on the grounds
    // that the resolution never changed -- so even once one did, nothing could
    // have signalled the object anybody was actually waiting on.
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

/// One 20 ms CELT-only Opus packet, 48 kHz mono, from the reference
/// encoder. Its decode is 960 samples per channel.
#[test]
fn hwopus_reports_a_work_buffer_size_before_it_opens_anything() {
    // `nn::codec` asks for the work buffer size, allocates that much as
    // transfer memory, and only then opens a decoder. A size of zero, which
    // is what the generic fallback answered: is an allocation that fails, so
    // nothing ever gets as far as decoding.
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

    // The large-frame form asks for room for a 120 ms packet, so it is bigger.
    args[8] = 1;
    ipc_request_plain(&mut cpu, HWOPUS, 5, &args);
    let large = cpu.mem.read_u32(tls + 0x20).unwrap();
    assert!(
        large > stereo,
        "the large-frame size {large:#x} is not above {stereo:#x}"
    );

    // A rate Opus does not have is refused rather than sized.
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
    // The packet does not arrive bare: `nn::codec` puts an eight-byte
    // big-endian { size, final_range } header in front of it, and the reply's
    // "bytes consumed" counts that header. A decoder that read the header as
    // little-endian, or reported only the payload, would leave the caller
    // walking its own buffer wrong and desynchronising after one packet.
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
    // { send_pid:1, num_copy:4, num_move:4 }: the decoder is a move handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let decoder = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(decoder, 0, "no IHardwareOpusDecoder came back");

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

    // The samples are 16-bit and are not all zero: a decoder that answered
    // success and wrote nothing would pass every check above.
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
    // The header says how long the packet is. A size longer than the buffer,
    // or a buffer with no room for the header at all, is a caller that has
    // lost its place in the stream; decoding whatever follows would turn that
    // into noise rather than an error it can act on.
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
