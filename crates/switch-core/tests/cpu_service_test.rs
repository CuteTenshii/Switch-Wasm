//! The Horizon services, reached over IPC: `hid`, `am`, `vi`, the audio pair,
//! `ldr:ro`, `hwopus` and the rest.

mod cpu;

use cpu::*;

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
