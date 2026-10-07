//! `am`: applet state, capture buffers and the applet stack.

mod cpu;

use cpu::*;

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
