//! `vi` and the binder: the display, its shared buffer and docking.

mod cpu;

use cpu::*;

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
