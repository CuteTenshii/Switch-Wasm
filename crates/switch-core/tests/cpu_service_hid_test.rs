//! `hid`: input shared memory, gamepad, touch and vibration.

mod cpu;

use cpu::*;
use switch_core::cpu::POINTER_BUFFER_SIZE;

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
