use crate::cpu::Cpu;
use crate::kernel::ipc::testing::*;

/// `AppletId_LibraryAppletWeb`.
const APPLET_WEB: u32 = 0x13;

#[test]
fn the_system_process_common_functions_chain_hands_back_real_sessions() {
    let mut cpu = request(false, 450, &[]);
    cpu.register_service_handle(9, "appletAE");
    cpu.set_applet_is_application(false);
    cpu.applet_request(TLS, 9, Some(450)).unwrap();
    let functions = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(
        functions, 0,
        "GetSystemProcessCommonFunctions moved no session back"
    );
    assert_eq!(
        cpu.service_name(functions),
        Some("am:system-process-common-functions")
    );

    marshal(&mut cpu, false, 1, &[]);
    cpu.applet_request(TLS, functions, Some(1)).unwrap();
    let observer = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(observer, 0, "cmd 1 moved no observer back");
    assert_eq!(cpu.service_name(observer), Some("am:application-observer"));

    marshal(&mut cpu, false, 460, &[]);
    cpu.applet_request(TLS, 9, Some(460)).unwrap();
    let alternative = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(
        alternative, 0,
        "GetAppletAlternativeFunctions moved no session back"
    );
    assert_eq!(
        cpu.service_name(alternative),
        Some("am:applet-alternative-functions")
    );

    // Neither opens a proxy, so the applet-kind flag is unchanged.
    assert!(!cpu.applet_is_application);
}

/// Marshal a `PushOutData`-shaped request carrying one object.
fn push_storage(cpu: &mut Cpu, command_id: u32, storage: u64) {
    for i in (0..0x200u32).step_by(4) {
        cpu.mem.write_u32(TLS + i, 0).unwrap();
    }
    cpu.mem.write_u32(TLS, 4).unwrap();
    cpu.mem.write_u32(TLS + 4, 8 | (1 << 31)).unwrap();
    cpu.mem.write_u32(TLS + 8, 1 << 5).unwrap();
    cpu.mem.write_u32(TLS + 12, storage as u32).unwrap();
    cpu.mem.write_u32(TLS + 0x10, SFCI).unwrap();
    cpu.mem.write_u32(TLS + 0x18, command_id).unwrap();
}

#[test]
fn the_applet_result_is_kept_rather_than_dropped() {
    const CONTROLLER: u64 = 0x0100_0000_0000_1003;
    const PUSH_OUT_DATA: u32 = 1;
    // { s8 player_count = 1, pad[3], u32 selected_id = 0, u32 result = 0 }.
    let mut result = vec![0u8; 0xC];
    result[0] = 1;

    const STORAGE: u64 = 0x21;
    let mut cpu = Cpu::new();
    cpu.mem.map_zero(TLS, 0x200).unwrap();
    cpu.set_program_id(CONTROLLER);
    cpu.register_service_handle(9, "am:library-applet-self-accessor");
    cpu.register_service_handle(STORAGE, "am:storage");
    cpu.am_storages
        .insert(Cpu::object_key(STORAGE, 0), result.clone());
    push_storage(&mut cpu, PUSH_OUT_DATA, STORAGE);
    cpu.applet_request(TLS, 9, Some(PUSH_OUT_DATA)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "push refused");
    assert_eq!(cpu.library_applet_results(), [result.clone()]);

    const STORAGE_OBJECT: u32 = 5;
    let mut cpu = request(true, PUSH_OUT_DATA, &[]);
    cpu.set_program_id(CONTROLLER);
    cpu.record_domain_object(9, 7, "am:library-applet-self-accessor");
    cpu.record_domain_object(9, STORAGE_OBJECT, "am:storage");
    cpu.am_storages
        .insert(Cpu::object_key(9, STORAGE_OBJECT), result.clone());
    // num_in_objects, then a `data_size` of just the `CmifInHeader`.
    cpu.mem.write_u8(TLS + 0x11, 1).unwrap();
    cpu.mem.write_u16(TLS + 0x12, 0x10).unwrap();
    cpu.mem.write_u32(TLS + 0x30, STORAGE_OBJECT).unwrap();
    cpu.applet_request(TLS, 9, Some(PUSH_OUT_DATA)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), 0, "push refused");
    assert_eq!(cpu.library_applet_results(), [result.clone()]);

    assert_eq!(
        super::applet_result_summary(CONTROLLER, &result),
        "confirmed, 1 player(s), npad 0"
    );
    result[8] = 2;
    assert_eq!(
        super::applet_result_summary(CONTROLLER, &result),
        "cancelled, 1 player(s), npad 0"
    );
}

#[test]
fn the_applet_can_be_answered_by_the_host_that_started_it() {
    const SWKBD: u64 = 0x0100_0000_0000_1008;
    const PUSH_INTERACTIVE_OUT_DATA: u32 = 3;
    const POP_INTERACTIVE_IN_DATA: u32 = 2;
    const GET_POP_INTERACTIVE_IN_DATA_EVENT: u32 = 6;
    const NO_DATA: u32 = 128 | (3 << 9);
    const STORAGE: u64 = 0x21;

    let mut cpu = Cpu::new();
    cpu.mem.map_zero(TLS, 0x200).unwrap();
    cpu.set_program_id(SWKBD);
    cpu.register_service_handle(9, "am:library-applet-self-accessor");
    cpu.register_service_handle(STORAGE, "am:storage");
    // `u64 size` then the text.
    let mut message = 4u64.to_le_bytes().to_vec();
    message.extend_from_slice(&[0x68, 0, 0x69, 0]);
    cpu.am_storages
        .insert(Cpu::object_key(STORAGE, 0), message.clone());
    push_storage(&mut cpu, PUSH_INTERACTIVE_OUT_DATA, STORAGE);
    cpu.applet_request(TLS, 9, Some(PUSH_INTERACTIVE_OUT_DATA))
        .unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "push refused");
    assert_eq!(cpu.library_applet_interactive_messages(), [message]);

    marshal(&mut cpu, false, GET_POP_INTERACTIVE_IN_DATA_EVENT, &[]);
    cpu.applet_request(TLS, 9, Some(GET_POP_INTERACTIVE_IN_DATA_EVENT))
        .unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(event, 0, "no event handed back");
    assert_eq!(cpu.event_name(event), Some("am:applet-interactive-in-data"));
    assert_eq!(cpu.event_signaled(event), Some(false), "answered already");

    marshal(&mut cpu, false, POP_INTERACTIVE_IN_DATA, &[]);
    cpu.applet_request(TLS, 9, Some(POP_INTERACTIVE_IN_DATA))
        .unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), NO_DATA, "pop");

    // `SwkbdTextCheckResult` Success and an empty message.
    let answer = vec![0u8; 0x8];
    cpu.push_applet_interactive_in_data(answer.clone());
    assert_eq!(cpu.event_signaled(event), Some(true), "answer unannounced");

    marshal(&mut cpu, false, POP_INTERACTIVE_IN_DATA, &[]);
    cpu.applet_request(TLS, 9, Some(POP_INTERACTIVE_IN_DATA))
        .unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "pop refused");
    let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_eq!(cpu.service_name(storage), Some("am:storage"));
    assert_eq!(cpu.am_storages[&Cpu::object_key(storage, 0)], answer);

    assert_eq!(cpu.event_signaled(event), Some(false), "still signalled");
}

#[test]
fn the_in_data_event_is_signalled_while_there_is_something_to_pop() {
    const SWKBD: u64 = 0x0100_0000_0000_1008;
    const GET_POP_IN_DATA_EVENT: u32 = 5;
    const POP_IN_DATA: u32 = 0;

    let mut cpu = request(false, GET_POP_IN_DATA_EVENT, &[]);
    cpu.set_program_id(SWKBD);
    cpu.seed_applet_launch_arguments();
    cpu.register_service_handle(9, "am:library-applet-self-accessor");
    cpu.applet_request(TLS, 9, Some(GET_POP_IN_DATA_EVENT))
        .unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_eq!(cpu.event_name(event), Some("am:applet-in-data"));
    assert_eq!(cpu.event_signaled(event), Some(true), "storages waiting");

    for _ in 0..3 {
        marshal(&mut cpu, false, POP_IN_DATA, &[]);
        cpu.applet_request(TLS, 9, Some(POP_IN_DATA)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "pop refused");
    }
    assert_eq!(cpu.event_signaled(event), Some(false), "queue not empty");
}

#[test]
fn the_idle_detection_setters_are_accepted_rather_than_refused() {
    for (cmd, payload) in [
        (60u32, &[0u8; 0x10][..]),
        (64, &[0u8; 4][..]),
        (65, &[][..]),
        (72, &[0u8; 4][..]),
    ] {
        let mut cpu = request(false, cmd, payload);
        cpu.register_service_handle(9, "am:self-controller");
        cpu.applet_request(TLS, 9, Some(cmd)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "cmd {cmd} refused"
        );
    }
}

#[test]
fn the_auto_sleep_settings_read_back_what_was_set() {
    const EXTENSION: u32 = 3;
    let mut cpu = request(false, 62, &EXTENSION.to_le_bytes());
    cpu.register_service_handle(9, "am:self-controller");
    cpu.applet_request(TLS, 9, Some(62)).unwrap();

    marshal(&mut cpu, false, 63, &[]);
    cpu.applet_request(TLS, 9, Some(63)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "refused");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), EXTENSION);

    marshal(&mut cpu, false, 68, &[1u8]);
    cpu.applet_request(TLS, 9, Some(68)).unwrap();
    marshal(&mut cpu, false, 69, &[]);
    cpu.applet_request(TLS, 9, Some(69)).unwrap();
    assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "auto sleep off");

    marshal(&mut cpu, false, 68, &[0u8]);
    cpu.applet_request(TLS, 9, Some(68)).unwrap();
    marshal(&mut cpu, false, 69, &[]);
    cpu.applet_request(TLS, 9, Some(69)).unwrap();
    assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "auto sleep on");
}

#[test]
fn the_home_button_double_click_setting_reads_back_what_was_set() {
    let mut cpu = request(false, 50, &[1u8]);
    cpu.register_service_handle(9, "am:applet-common-functions");
    cpu.applet_request(TLS, 9, Some(50)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "set refused");

    marshal(&mut cpu, false, 51, &[]);
    cpu.applet_request(TLS, 9, Some(51)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "get refused");
    assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "double click on");

    marshal(&mut cpu, false, 50, &[0u8]);
    cpu.applet_request(TLS, 9, Some(50)).unwrap();
    marshal(&mut cpu, false, 51, &[]);
    cpu.applet_request(TLS, 9, Some(51)).unwrap();
    assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "double click off");
}

#[test]
fn every_button_lock_accessor_hands_back_a_lock() {
    const HOME_BUTTON: u32 = 0;
    for cmd in [30u32, 31, 32] {
        let mut cpu = request(false, cmd, &HOME_BUTTON.to_le_bytes());
        cpu.register_service_handle(9, "am:common-state-getter");
        cpu.applet_request(TLS, 9, Some(cmd)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "cmd {cmd} refused"
        );
        let lock = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(lock, 0, "cmd {cmd} moved no accessor back");
        assert_eq!(cpu.service_name(lock), Some("am:lock-accessor"));
    }
}

#[test]
fn am_reports_the_handheld_operation_mode_it_always_claimed_to() {
    let mut cpu = request(false, 5, &[]);
    cpu.register_service_handle(9, "am:common-state-getter");
    cpu.applet_request(TLS, 9, Some(5)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "Handheld");
}

#[test]
fn the_copyright_notice_for_captures_is_accepted() {
    for cmd in [100, 101, 102] {
        let mut cpu = request(false, cmd, &[]);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(cmd)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "command {cmd}");
    }
}

#[test]
fn each_applet_event_is_named_after_the_interface_that_hands_it_out() {
    let mut cpu = request(false, 130, &[]);
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(130)).unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(
        event, 0,
        "GetGpuErrorDetectedSystemEvent handed back no handle"
    );
    assert_eq!(cpu.event_name(event), Some("am:gpu-error"));

    let mut cpu = request(false, 91, &[]);
    cpu.register_service_handle(9, "am:self-controller");
    cpu.applet_request(TLS, 9, Some(91)).unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(event, 0);
    assert_eq!(
        cpu.event_name(event),
        Some("am:accumulated-suspended-tick-changed")
    );
    assert_eq!(cpu.event_signaled(event), Some(false));

    let mut cpu = request(false, 9, &[]);
    cpu.register_service_handle(9, "am:self-controller");
    cpu.applet_request(TLS, 9, Some(9)).unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(event, 0);
    assert_eq!(cpu.event_name(event), Some("am:library-applet-launchable"));
    assert_eq!(cpu.event_signaled(event), Some(true));

    cpu.applet_request(TLS, 9, Some(9)).unwrap();
    assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
}

#[test]
fn am_gives_back_the_terminate_result_the_title_set() {
    const TERMINATE_RESULT: u32 = 202 | (30 << 9);
    let mut cpu = request(false, 22, &TERMINATE_RESULT.to_le_bytes());
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(22)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.am_terminate_result, TERMINATE_RESULT);

    marshal(&mut cpu, false, 200, &[]);
    cpu.applet_request(TLS, 9, Some(200)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), TERMINATE_RESULT);
}

#[test]
fn the_preselected_user_is_handed_over_once_and_then_it_is_gone() {
    const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9);
    const SFCO: u32 = 0x4F43_4653;

    let kind = super::LAUNCH_PARAMETER_PRESELECTED_USER.to_le_bytes();
    let mut cpu = request(false, 1, &kind);
    cpu.seed_launch_parameters();
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(1)).unwrap();

    let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(storage, 0, "PopLaunchParameter moved no storage back");
    assert_eq!(cpu.service_name(storage), Some("am:storage"));

    // Magic, version, and a non-zero uid at offset 8.
    let data = cpu.am_storages[&Cpu::object_key(storage, 0)].clone();
    assert_eq!(data.len(), 0x88);
    assert_eq!(
        u32::from_le_bytes(data[..4].try_into().unwrap()),
        0xC794_97CA
    );
    assert_eq!(data[4], 1, "layout version");
    assert_eq!(&data[8..0x18], &crate::services::acc::DEFAULT_USER_UID[..]);

    marshal(&mut cpu, false, 1, &kind);
    cpu.applet_request(TLS, 9, Some(1)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x10).unwrap(), SFCO);
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        LAUNCH_PARAMETER_NOT_FOUND
    );
}

#[test]
fn a_launch_parameter_nobody_left_is_still_refused() {
    const USER_CHANNEL: u32 = 1;
    const LAUNCH_PARAMETER_NOT_FOUND: u32 = 128 | (2 << 9);

    let mut cpu = request(false, 1, &USER_CHANNEL.to_le_bytes());
    cpu.seed_launch_parameters();
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(1)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        LAUNCH_PARAMETER_NOT_FOUND
    );
}

#[test]
fn application_functions_210_hands_out_one_event_and_keeps_handing_out_that_one() {
    let mut cpu = request(false, 210, &[]);
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(210)).unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(event, 0, "command 210 handed back no event handle");
    assert_eq!(cpu.event_name(event), Some("am:application-functions-210"));

    assert_eq!(cpu.event_signaled(event), Some(false));

    marshal(&mut cpu, false, 210, &[]);
    cpu.applet_request(TLS, 9, Some(210)).unwrap();
    assert_eq!(u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap()), event);
}

#[test]
fn a_stubbed_answer_names_itself_once_and_still_succeeds() {
    let mut cpu = request(false, 66, &[]);
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(66)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
    let trace = String::from_utf8_lossy(&cpu.trace).into_owned();
    assert!(
        trace.contains("[ipc] stub: am:application-functions cmd=Some(66)"),
        "the stub went unreported: {trace:?}"
    );

    // Once per (interface, command).
    marshal(&mut cpu, false, 66, &[]);
    cpu.applet_request(TLS, 9, Some(66)).unwrap();
    let repeated = String::from_utf8_lossy(&cpu.trace)
        .matches("[ipc] stub:")
        .count();
    assert_eq!(repeated, 1, "the stub was reported on every call");
}

#[test]
fn get_save_data_size_reports_the_quota_the_title_was_actually_allotted() {
    let mut payload = [0u8; 0x18];
    payload[0] = 1; // SaveDataType::Account
    payload[8..].copy_from_slice(&crate::services::acc::DEFAULT_USER_UID);

    // Tomodachi Life's NACP figures.
    const SAVE: i64 = 56_623_104;
    const JOURNAL: i64 = 10_485_760;

    let mut cpu = request(false, 26, &payload);
    cpu.set_save_data_quota(crate::services::fs::SaveDataQuota {
        size: SAVE,
        journal_size: JOURNAL,
        ..Default::default()
    });
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(26)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");
    assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, SAVE);
    assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap() as i64, JOURNAL);
}

#[test]
fn extending_a_save_grants_it_and_the_size_read_back_is_the_extended_one() {
    const SIZE: i64 = 0x1200_0000;
    const JOURNAL: i64 = 0x0100_0000;
    let mut payload = [0u8; 0x28];
    payload[0] = 1; // SaveDataType::Account
    payload[8..0x18].copy_from_slice(&crate::services::acc::DEFAULT_USER_UID);
    payload[0x18..0x20].copy_from_slice(&SIZE.to_le_bytes());
    payload[0x20..].copy_from_slice(&JOURNAL.to_le_bytes());

    let mut cpu = request(false, 25, &payload);
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(25)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "Result");

    marshal(&mut cpu, false, 26, &[0u8; 0x18]);
    cpu.applet_request(TLS, 9, Some(26)).unwrap();
    assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap() as i64, SIZE);
    assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap() as i64, JOURNAL);
}

#[test]
fn the_save_data_ceilings_are_reported_apart_from_the_sizes() {
    let quota = crate::services::fs::SaveDataQuota {
        size: 1,
        journal_size: 2,
        size_max: 3,
        journal_size_max: 4,
        device_size_max: 5,
        device_journal_size_max: 6,
        ..Default::default()
    };
    for (command, expected) in [(26, (1i64, 2i64)), (28, (3, 4)), (35, (5, 6))] {
        let mut cpu = request(false, command, &[0u8; 0x18]);
        cpu.set_save_data_quota(quota);
        cpu.register_service_handle(9, "am:application-functions");
        cpu.applet_request(TLS, 9, Some(command)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            0,
            "Result of {command}"
        );
        let got = (
            cpu.mem.read_u64(TLS + 0x20).unwrap() as i64,
            cpu.mem.read_u64(TLS + 0x28).unwrap() as i64,
        );
        assert_eq!(got, expected, "command {command}");
    }
}

#[test]
fn a_declared_ceiling_of_zero_is_reported_as_zero() {
    let mut cpu = request(false, 28, &[]);
    cpu.set_save_data_quota(crate::services::fs::SaveDataQuota {
        size: 56_623_104,
        journal_size: 10_485_760,
        size_max: 0,
        journal_size_max: 0,
        ..Default::default()
    });
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(28)).unwrap();
    assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 0);
    assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap(), 0);
}

#[test]
fn get_cache_storage_max_aligns_its_size_after_its_count() {
    // s32 then s64 at +8; +4 is padding.
    let mut cpu = request(false, 29, &[]);
    cpu.set_save_data_quota(crate::services::fs::SaveDataQuota {
        cache_storage_index_max: 3,
        cache_storage_size_max: 0x40_0000,
        ..Default::default()
    });
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(29)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 3, "index max");
    assert_eq!(cpu.mem.read_u64(TLS + 0x28).unwrap(), 0x40_0000, "size max");
}

#[test]
fn a_title_whose_nacp_nobody_read_still_gets_room_to_save() {
    let mut payload = [0u8; 0x18];
    payload[0] = 1;
    let mut cpu = request(false, 26, &payload);
    cpu.register_service_handle(9, "am:application-functions");
    cpu.applet_request(TLS, 9, Some(26)).unwrap();
    let size = cpu.mem.read_u64(TLS + 0x20).unwrap() as i64;
    let journal = cpu.mem.read_u64(TLS + 0x28).unwrap() as i64;
    assert_eq!(size, crate::services::fs::DEFAULT_SAVE_DATA_SIZE);
    assert_eq!(journal, crate::services::fs::DEFAULT_SAVE_DATA_JOURNAL_SIZE);
    assert!(
        size >= 56_623_104,
        "default quota is smaller than a real title's save"
    );
    assert!(
        journal >= 10_485_760,
        "default journal is smaller than a real title's"
    );
}

fn library_applet(applet_id: u32) -> (Cpu, u64) {
    let mut payload = [0u8; 8];
    payload[..4].copy_from_slice(&applet_id.to_le_bytes());
    let mut cpu = request(false, 0, &payload);
    cpu.register_service_handle(9, "am:library-applet-creator");
    cpu.applet_request(TLS, 9, Some(0)).unwrap();
    let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(accessor, 0, "CreateLibraryApplet moved no object back");
    assert_eq!(
        cpu.service_name(accessor),
        Some("am:library-applet-accessor")
    );
    (cpu, accessor)
}

#[test]
fn the_keyboard_and_the_controller_applet_pop_three_storages() {
    const SWKBD: u64 = 0x0100_0000_0000_1008;
    const CONTROLLER: u64 = 0x0100_0000_0000_1003;
    const NO_DATA: u32 = 128 | (3 << 9);

    for program_id in [SWKBD, CONTROLLER] {
        let mut cpu = request(false, 0, &[]);
        cpu.set_program_id(program_id);
        cpu.seed_applet_launch_arguments();
        cpu.register_service_handle(9, "am:library-applet-self-accessor");

        let mut sizes = Vec::new();
        for pop in 0..3 {
            write_request(&mut cpu, 0, &[]);
            cpu.applet_request(TLS, 9, Some(0)).unwrap();
            assert_eq!(
                cpu.mem.read_u32(TLS + 0x18).unwrap(),
                0,
                "{program_id:#x} pop {pop} was refused"
            );
            let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
            assert_eq!(cpu.service_name(storage), Some("am:storage"));
            write_request(&mut cpu, 0, &[]);
            cpu.applet_request(TLS, storage, Some(0)).unwrap();
            let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
            write_request(&mut cpu, 0, &[]);
            cpu.applet_request(TLS, accessor, Some(0)).unwrap();
            sizes.push(cpu.mem.read_u64(TLS + 0x20).unwrap());
        }

        let expected: [u64; 3] = match program_id {
            SWKBD => [0x20, 0x4C8, 0x1000],
            _ => [0x20, 0x14, 0x430],
        };
        assert_eq!(sizes, expected, "{program_id:#x} storage sizes");

        write_request(&mut cpu, 0, &[]);
        cpu.applet_request(TLS, 9, Some(0)).unwrap();
        assert_eq!(
            cpu.mem.read_u32(TLS + 0x18).unwrap(),
            NO_DATA,
            "{program_id:#x} was handed a fourth storage"
        );
    }
}

#[test]
fn every_applet_title_id_maps_to_the_id_switchbrew_gives_it() {
    for (program_id, applet_id) in [
        (0x0100_0000_0000_1000u64, 0x03u32), // qlaunch
        (0x0100_0000_0000_1003, 0x0C),       // controller
        (0x0100_0000_0000_1008, 0x11),       // swkbd
        (0x0100_0000_0000_100C, 0x02),       // overlayDisp
        (0x0100_0000_0000_100D, 0x15),       // photoViewer
        (0x0100_0000_0000_1011, 0x19),       // wifiWebAuth
        (0x0100_0000_0000_1012, 0x04),       // starter, a SystemApplication
        (0x0100_0000_0000_1013, 0x1A),       // myPage
    ] {
        assert_eq!(
            super::applet_id_for(program_id),
            applet_id,
            "{program_id:#x}"
        );
    }
    assert!(super::is_library_applet(0x0100_0000_0000_1013));
    assert!(!super::is_library_applet(0x0100_0000_0000_1012));

    assert_eq!(
        super::applet_interface_version(0x0100_0000_0000_1013),
        0x1_0000
    );
    let arg = &super::applet_launch_storages(
        0x0100_0000_0000_1013,
        crate::services::acc::DEFAULT_USER_UID,
    )[0];
    assert_eq!(arg.len(), 0x10A8);
    assert_eq!(arg[8..24], crate::services::acc::DEFAULT_USER_UID);
}

#[test]
fn an_applet_that_pushes_its_result_and_exits_stops_the_process() {
    let mut cpu = request(false, 1, &[]);
    cpu.register_service_handle(9, "am:library-applet-self-accessor");

    cpu.applet_request(TLS, 9, Some(1)).unwrap(); // PushOutData
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "PushOutData");
    assert!(!cpu.halted, "the applet has not asked to exit yet");

    write_request(&mut cpu, 10, &[]);
    cpu.applet_request(TLS, 9, Some(10)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        0,
        "ExitProcessAndReturn"
    );
    assert!(cpu.halted, "the applet asked to exit and kept running");
}

#[test]
fn the_controller_applet_is_told_what_it_may_offer() {
    const CONTROLLER: u64 = 0x0100_0000_0000_1003;
    let storages =
        super::applet_launch_storages(CONTROLLER, crate::services::acc::DEFAULT_USER_UID);
    let private = &storages[0];
    assert_eq!(
        u32::from_le_bytes(private[..4].try_into().unwrap()),
        private.len() as u32
    );
    assert_eq!(
        u32::from_le_bytes(private[4..8].try_into().unwrap()) as usize,
        storages[1].len()
    );
    assert_eq!(super::applet_interface_version(CONTROLLER), 8);

    let styles = u32::from_le_bytes(private[0x0C..0x10].try_into().unwrap());
    assert_ne!(
        styles & crate::cpu::hid_shmem::STYLE_HANDHELD,
        0,
        "handheld is not on offer"
    );
    for pad in crate::cpu::NPAD_PRESENTATIONS {
        assert_ne!(
            styles & pad.style,
            0,
            "style {:#x} is not on offer",
            pad.style
        );
    }
}

#[test]
fn a_library_applet_ends_the_moment_it_is_started() {
    let (mut cpu, accessor) = library_applet(APPLET_WEB);

    write_request(&mut cpu, 0, &[]);
    cpu.applet_request(TLS, accessor, Some(0)).unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_eq!(cpu.event_name(event), Some("am:library-applet-state"));
    assert_eq!(
        cpu.event_signaled(event),
        Some(false),
        "nothing has started it yet"
    );

    write_request(&mut cpu, 10, &[]); // Start
    cpu.applet_request(TLS, accessor, Some(10)).unwrap();
    assert_eq!(cpu.event_signaled(event), Some(true));

    write_request(&mut cpu, 1, &[]); // IsCompleted
    cpu.applet_request(TLS, accessor, Some(1)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap() & 0xff, 1);

    write_request(&mut cpu, 30, &[]);
    cpu.applet_request(TLS, accessor, Some(30)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        128 | (22 << 9),
        "cancelled"
    );
}

#[test]
fn the_applet_state_event_is_signalled_when_it_is_asked_for_after_the_start() {
    let (mut cpu, accessor) = library_applet(APPLET_WEB);
    write_request(&mut cpu, 10, &[]);
    cpu.applet_request(TLS, accessor, Some(10)).unwrap();

    write_request(&mut cpu, 0, &[]);
    cpu.applet_request(TLS, accessor, Some(0)).unwrap();
    let event = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_eq!(cpu.event_signaled(event), Some(true));
}

#[test]
fn an_applet_that_never_ran_has_no_output_to_pop() {
    let (mut cpu, accessor) = library_applet(APPLET_WEB);
    write_request(&mut cpu, 10, &[]);
    cpu.applet_request(TLS, accessor, Some(10)).unwrap();

    write_request(&mut cpu, 101, &[]); // PopOutData
    cpu.applet_request(TLS, accessor, Some(101)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        128 | (3 << 9),
        "no data"
    );
    assert_eq!(cpu.mem.read_u32(TLS + 0x0c).unwrap(), 0, "and no storage");
}

#[test]
fn created_storage_is_as_long_as_the_caller_asked_for() {
    let mut cpu = request(false, 10, &0x1000u64.to_le_bytes());
    cpu.register_service_handle(9, "am:library-applet-creator");
    cpu.applet_request(TLS, 9, Some(10)).unwrap();
    let storage = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_eq!(cpu.service_name(storage), Some("am:storage"));
    assert_eq!(cpu.am_storages[&Cpu::object_key(storage, 0)].len(), 0x1000);

    write_request(&mut cpu, 10, &u64::MAX.to_le_bytes());
    cpu.applet_request(TLS, 9, Some(10)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        1 | (104 << 9),
        "out of memory"
    );
}

#[test]
fn the_ex_form_of_a_creation_is_the_same_creation() {
    const CALLER_THREAD: u64 = 0x2a;
    const FOREGROUND: u32 = 1;

    let mut payload = [0u8; 16];
    payload[..4].copy_from_slice(&APPLET_WEB.to_le_bytes());
    payload[4..8].copy_from_slice(&FOREGROUND.to_le_bytes());
    payload[8..].copy_from_slice(&CALLER_THREAD.to_le_bytes());
    let mut cpu = request(false, 3, &payload);
    cpu.register_service_handle(9, "am:library-applet-creator");
    cpu.applet_request(TLS, 9, Some(3)).unwrap();

    let accessor = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
    assert_ne!(accessor, 0, "CreateLibraryAppletEx moved no object back");
    assert_eq!(
        cpu.service_name(accessor),
        Some("am:library-applet-accessor")
    );
    let applet = &cpu.am_applets[&Cpu::object_key(accessor, 0)];
    assert_eq!(applet.id, APPLET_WEB);
    assert_eq!(applet.mode, FOREGROUND);
}
