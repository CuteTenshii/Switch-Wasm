use super::Cpu;
use crate::cpu::ipc::testing::*;

#[test]
fn set_sys_reports_a_platform_region_that_is_in_the_enum() {
    // `PlatformRegion` is Global (1) or Terra (2) and has no zero.
    let mut cpu = request(false, 183, &[]);
    cpu.set_sys_request(TLS, Some(183)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    let region = cpu.mem.read_u32(TLS + 0x20).unwrap();
    assert_eq!(region, 1, "this console is the Global one");
}

#[test]
fn set_sys_reports_an_accepted_eula() {
    const BUFFER: u32 = 0x4000;
    const ENTRY: u32 = 0x30;

    let mut cpu = request_with_recv_buffer(21, &[], BUFFER, 4 * ENTRY);
    cpu.mem.map_zero(BUFFER, 0x200).unwrap();
    cpu.set_sys_request(TLS, Some(21)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "one agreement");
    assert_ne!(cpu.mem.read_u32(BUFFER).unwrap(), 0, "a version was set");
    assert_eq!(cpu.mem.read_u32(BUFFER + 4).unwrap(), 1, "SetRegion_USA");

    // No room for an entry gives a count of zero.
    write_map_buffer_request(&mut cpu, 21, &[], BUFFER, ENTRY - 1, false);
    cpu.set_sys_request(TLS, Some(21)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "nothing fits");
}

#[test]
fn set_sys_get_serial_number_returns_a_nul_padded_placeholder() {
    let mut cpu = request(false, 68, &[]);
    cpu.set_sys_request(TLS, Some(68)).unwrap();
    let mut got = [0u8; 0x18];
    for (i, byte) in got.iter_mut().enumerate() {
        *byte = cpu.mem.read_u8(TLS + 0x20 + i as u32).unwrap();
    }
    assert!(got.starts_with(b"XAW00000000000"));
    assert_eq!(got[b"XAW00000000000".len()], 0, "NUL-padded, not garbage");
}

#[test]
fn set_sys_fills_the_settings_blocks_that_outrun_the_reply_padding() {
    // Replies zero only 16 bytes of padding; the scribble catches wider blocks left stale.
    const STALE: u8 = 0xa5;

    let mut cpu = request(false, 39, &[]);
    for offset in 0x30..0x40 {
        cpu.mem.write_u8(TLS + offset, STALE).unwrap();
    }
    cpu.set_sys_request(TLS, Some(39)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0xc, "TvFlag");
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x24).unwrap(),
        0,
        "TvResolution_Auto"
    );
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x28).unwrap(),
        4,
        "HdmiContentType_Game"
    );
    assert_eq!(
        f32::from_bits(cpu.mem.read_u32(TLS + 0x38).unwrap()),
        1.0,
        "tv_gama, past the padding"
    );
    assert_eq!(
        f32::from_bits(cpu.mem.read_u32(TLS + 0x3c).unwrap()),
        0.5,
        "contrast_ratio, past the padding"
    );

    let mut cpu = request(false, 29, &[]);
    for offset in 0x30..0x40 {
        cpu.mem.write_u8(TLS + offset, STALE).unwrap();
    }
    cpu.set_sys_request(TLS, Some(29)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        0x300,
        "NotificationFlag"
    );
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x24).unwrap(),
        2,
        "NotificationVolume_High"
    );
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x28).unwrap(),
        9,
        "quiet hours start"
    );
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x30).unwrap(),
        21,
        "and they stop, past the padding"
    );
    assert_eq!(cpu.mem.read_u32(TLS + 0x34).unwrap(), 0, "on the hour");
}

#[test]
fn set_sys_answers_the_enums_whose_zero_is_wrong() {
    // `ProductModel` starts at 1; `KeyboardLayout`'s zero is `Japanese`.
    let mut cpu = request(false, 79, &[]);
    cpu.set_sys_request(TLS, Some(79)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "ProductModel_Nx");

    let mut cpu = request(false, 136, &[]);
    cpu.set_sys_request(TLS, Some(136)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        1,
        "KeyboardLayout_EnglishUs"
    );
}

#[test]
fn set_sys_sleeps_never_rather_than_at_an_arbitrary_hour() {
    // The sleep plans are indices, so zero would mean "sleep after one minute".
    const NEVER: u32 = 5;
    let mut cpu = request(false, 71, &[]);
    cpu.set_sys_request(TLS, Some(71)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 0, "flags");
    assert_eq!(cpu.mem.read_u32(TLS + 0x24).unwrap(), NEVER, "handheld");
    assert_eq!(cpu.mem.read_u32(TLS + 0x28).unwrap(), NEVER, "console");
}

#[test]
fn set_sys_names_the_settings_it_used_to_leave_to_the_reply_padding() {
    // The padding reads 0 too, so only the stub list tells these apart.
    for (cmd, want) in [
        (17u32, 0u32), // GetAccountSettings
        (23, 0),       // GetColorSetId, BasicWhite
        (31, 0),       // GetAccountNotificationSettings, no overrides
        (63, 0),       // GetPrimaryAlbumStorage, Nand
        (124, 0),      // GetErrorReportSharePermission, NotConfirmed
        (126, 0),      // GetAppletLaunchFlags
        (170, 0),      // GetChineseTraditionalInputMethod
    ] {
        let mut cpu = request(false, cmd, &[]);
        cpu.set_sys_request(TLS, Some(cmd)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "cmd {cmd} result");
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), want, "cmd {cmd}");
        assert!(cpu.stubbed_ipc().is_empty(), "cmd {cmd} fell to the stub");
    }
    // QuestFlag is a `u8`: Retail, not Kiosk.
    for cmd in [7u32, 47, 95, 99, 201] {
        let mut cpu = request(false, cmd, &[]);
        cpu.set_sys_request(TLS, Some(cmd)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "cmd {cmd} result");
        assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "cmd {cmd}");
        assert!(cpu.stubbed_ipc().is_empty(), "cmd {cmd} fell to the stub");
    }
}

#[test]
fn set_sys_reports_a_real_firmware_version_into_its_pointer_buffer() {
    const BUFFER: u32 = 0x4000;
    let mut cpu = request_with_recv_static(3, &[], BUFFER, 0x100);
    cpu.mem.map_zero(BUFFER, 0x200).unwrap();
    for offset in 0..0x100 {
        cpu.mem.write_u8(BUFFER + offset, b'x').unwrap();
    }
    cpu.set_sys_request(TLS, Some(3)).unwrap();

    let (major, minor, micro) = super::FIRMWARE_VERSION;
    assert_eq!(cpu.mem.read_u8(BUFFER).unwrap(), major);
    assert_eq!(cpu.mem.read_u8(BUFFER + 1).unwrap(), minor);
    assert_eq!(cpu.mem.read_u8(BUFFER + 2).unwrap(), micro);
    assert_eq!(cpu.read_string(BUFFER + 0x08, 0x20), "NX");
    // The display strings agree with the numbers above them.
    let display = format!("{major}.{minor}.{micro}");
    assert_eq!(cpu.read_string(BUFFER + 0x68, 0x18), display);
    assert!(cpu
        .read_string(BUFFER + 0x80, 0x80)
        .ends_with(&format!("{display}-1.0")));
}

#[test]
fn set_sys_reads_back_what_its_setters_were_given() {
    const COLOR_SET: u32 = 24;
    const KEYBOARD_LAYOUT: u32 = 137;
    const LOCK_SCREEN: u32 = 8;
    const VIBRATION_VOLUME: u32 = 36;

    let mut cpu = request(false, COLOR_SET, &1u32.to_le_bytes());
    cpu.set_sys_request(TLS, Some(COLOR_SET)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    write_request(&mut cpu, 23, &[]);
    cpu.set_sys_request(TLS, Some(23)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        1,
        "ColorSet_BasicBlack"
    );

    write_request(&mut cpu, KEYBOARD_LAYOUT, &4u32.to_le_bytes());
    cpu.set_sys_request(TLS, Some(KEYBOARD_LAYOUT)).unwrap();
    write_request(&mut cpu, 136, &[]);
    cpu.set_sys_request(TLS, Some(136)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        4,
        "KeyboardLayout_French"
    );

    write_request(&mut cpu, LOCK_SCREEN, &[1]);
    cpu.set_sys_request(TLS, Some(LOCK_SCREEN)).unwrap();
    write_request(&mut cpu, 7, &[]);
    cpu.set_sys_request(TLS, Some(7)).unwrap();
    assert_eq!(
        cpu.mem.read_u8(TLS + 0x20).unwrap(),
        1,
        "the lock screen is on"
    );

    write_request(&mut cpu, VIBRATION_VOLUME, &0.25f32.to_le_bytes());
    cpu.set_sys_request(TLS, Some(VIBRATION_VOLUME)).unwrap();
    write_request(&mut cpu, 35, &[]);
    cpu.set_sys_request(TLS, Some(35)).unwrap();
    assert_eq!(
        f32::from_bits(cpu.mem.read_u32(TLS + 0x20).unwrap()),
        0.25,
        "a float setting, not its bit pattern read as an integer"
    );
}

#[test]
fn set_sys_keeps_a_settings_block_whole_past_the_reply_padding() {
    const SET_TV_SETTINGS: u32 = 40;
    let mut block = [0u8; 0x20];
    block[0x00..0x04].copy_from_slice(&1u32.to_le_bytes()); // Allows4k
    block[0x04..0x08].copy_from_slice(&2u32.to_le_bytes()); // 720p
    block[0x18..0x1c].copy_from_slice(&2.2f32.to_le_bytes()); // tv_gama
    block[0x1c..0x20].copy_from_slice(&0.75f32.to_le_bytes()); // contrast

    let mut cpu = request(false, SET_TV_SETTINGS, &block);
    cpu.set_sys_request(TLS, Some(SET_TV_SETTINGS)).unwrap();
    write_request(&mut cpu, 39, &[]);
    cpu.set_sys_request(TLS, Some(39)).unwrap();
    for (index, &byte) in block.iter().enumerate() {
        assert_eq!(
            cpu.mem.read_u8(TLS + 0x20 + index as u32).unwrap(),
            byte,
            "byte {index} of TvSettings"
        );
    }
}

#[test]
fn set_sys_keeps_one_audio_mode_per_output() {
    const SET: u32 = 44;
    const GET: u32 = 43;
    const HEADPHONE: u32 = 3;
    const HDMI: u32 = 1;
    const CH_5_1: u32 = 2;

    let mut payload = Vec::new();
    payload.extend_from_slice(&HEADPHONE.to_le_bytes());
    payload.extend_from_slice(&CH_5_1.to_le_bytes());
    let mut cpu = request(false, SET, &payload);
    cpu.set_sys_request(TLS, Some(SET)).unwrap();

    write_request(&mut cpu, GET, &HEADPHONE.to_le_bytes());
    cpu.set_sys_request(TLS, Some(GET)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        CH_5_1,
        "the headphones"
    );

    write_request(&mut cpu, GET, &HDMI.to_le_bytes());
    cpu.set_sys_request(TLS, Some(GET)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        super::AUDIO_OUTPUT_STEREO,
        "and the dock, which nothing changed"
    );
}

#[test]
fn set_sys_replaces_the_eula_list_from_the_buffer_it_was_handed() {
    // SetEulaVersions replaces the list.
    const SET: u32 = 22;
    const GET: u32 = 21;
    const ENTRY: u32 = 0x30;
    const IN: u32 = 0x4000;
    const OUT: u32 = 0x5000;

    let mut cpu = request(false, SET, &[]);
    cpu.mem.map_zero(IN, 0x200).unwrap();
    cpu.mem.map_zero(OUT, 0x200).unwrap();
    cpu.mem.write_u32(IN, 0x2_0000).unwrap();
    cpu.mem.write_u32(IN + ENTRY, 0x3_0000).unwrap();
    write_map_buffer_request(&mut cpu, SET, &[], IN, 2 * ENTRY, true);
    cpu.set_sys_request(TLS, Some(SET)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");

    write_map_buffer_request(&mut cpu, GET, &[], OUT, 4 * ENTRY, false);
    cpu.set_sys_request(TLS, Some(GET)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 2, "two agreements");
    assert_eq!(cpu.mem.read_u32(OUT).unwrap(), 0x2_0000);
    assert_eq!(cpu.mem.read_u32(OUT + ENTRY).unwrap(), 0x3_0000);
}

#[test]
fn the_device_nick_name_set_sys_was_given_reads_back_through_both_services() {
    // Through a buffer, not the raw data, which would store the descriptor words.
    const SET: u32 = 78;
    const SYS_GET: u32 = 77;
    const SET_GET: u32 = 11;
    const IN: u32 = 0x4000;
    const OUT: u32 = 0x5000;
    const NAME: &[u8] = b"the console in the tab";

    let mut cpu = request(false, SET, &[]);
    cpu.mem.map_zero(IN, 0x200).unwrap();
    cpu.mem.map_zero(OUT, 0x200).unwrap();
    for (index, &byte) in NAME.iter().enumerate() {
        cpu.mem.write_u8(IN + index as u32, byte).unwrap();
    }
    write_map_buffer_request(&mut cpu, SET, &[], IN, 0x80, true);
    cpu.set_sys_request(TLS, Some(SET)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "set result");

    cpu.register_service_handle(9, "set");
    for (service, get) in [("set:sys", SYS_GET), ("set", SET_GET)] {
        // An untouched buffer must not pass as the name.
        for offset in 0..0x80 {
            cpu.mem.write_u8(OUT + offset, 0xa5).unwrap();
        }
        write_map_buffer_request(&mut cpu, get, &[], OUT, 0x80, false);
        if service == "set" {
            cpu.set_request(TLS, 9, Some(get)).unwrap();
        } else {
            cpu.set_sys_request(TLS, Some(get)).unwrap();
        }
        assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "{service} result");
        assert_eq!(
            cpu.read_string(OUT, 0x80),
            "the console in the tab",
            "{service}"
        );
    }
}

#[test]
fn a_setting_survives_the_session_that_wrote_it() {
    // The settings persist through system save data `8000000000000050`.
    const SET_COLOR_SET: u32 = 24;
    const GET_COLOR_SET: u32 = 23;

    let mut cpu = request(false, SET_COLOR_SET, &1u32.to_le_bytes());
    cpu.set_sys_request(TLS, Some(SET_COLOR_SET)).unwrap();

    let save = cpu
        .save_data(super::SYSTEM_SETTINGS_SAVE)
        .expect("the setter files the settings in their save");
    assert!(save.pending_changes() > 0, "the host is told to persist it");
    let stored = save
        .file(super::SYSTEM_SETTINGS_FILE)
        .expect("and it is a file in that save")
        .to_vec();

    // A fresh session, handed the save back the way `saveRestore` does.
    let mut next = request(false, GET_COLOR_SET, &[]);
    next.save_data_mut(super::SYSTEM_SETTINGS_SAVE)
        .write_file(super::SYSTEM_SETTINGS_FILE, stored);
    assert_eq!(
        next.save_data(super::SYSTEM_SETTINGS_SAVE)
            .unwrap()
            .pending_changes(),
        0,
        "restoring a save is not a change to write straight back"
    );
    next.set_sys_request(TLS, Some(GET_COLOR_SET)).unwrap();
    assert_eq!(
        next.mem.read_u32(TLS + 0x20).unwrap(),
        1,
        "the colour set the session before it chose"
    );
}

#[test]
fn settings_that_were_never_stored_keep_their_defaults() {
    // A block from an older build restores the missing settings as defaults.
    let settings = super::SystemSettings {
        color_set: 1,
        ..Default::default()
    };
    let mut stored = settings.serialize();
    // Drop the last record, as an older build would.
    stored.truncate(stored.len() - 8);
    let read = super::SystemSettings::parse(&stored).expect("a block this build wrote");
    assert_eq!(read.color_set, 1, "what was stored");
    assert_eq!(
        read.panel_crc_mode,
        super::SystemSettings::default().panel_crc_mode,
        "and the default for what was not"
    );

    // Bytes this build did not write are not a settings block at all.
    assert!(super::SystemSettings::parse(b"not a settings block").is_none());
    assert!(super::SystemSettings::parse(&[]).is_none());
}

#[test]
fn the_language_set_sys_is_given_is_the_one_set_reports() {
    const SET_LANGUAGE_CODE: u32 = 0;
    const SET_REGION_CODE: u32 = 57;
    const REGION_EUROPE: u32 = 2;
    let french = super::language_code(2);

    let mut cpu = request(false, SET_LANGUAGE_CODE, &french.to_le_bytes());
    cpu.register_service_handle(9, "set");
    cpu.set_sys_request(TLS, Some(SET_LANGUAGE_CODE)).unwrap();
    write_request(&mut cpu, 0, &[]);
    cpu.set_request(TLS, 9, Some(0)).unwrap();
    assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), french, "fr");

    write_request(&mut cpu, SET_REGION_CODE, &REGION_EUROPE.to_le_bytes());
    cpu.set_sys_request(TLS, Some(SET_REGION_CODE)).unwrap();
    write_request(&mut cpu, 4, &[]);
    cpu.set_request(TLS, 9, Some(4)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), REGION_EUROPE);

    write_request(&mut cpu, 2, &0u32.to_le_bytes());
    cpu.set_request(TLS, 9, Some(2)).unwrap();
    assert_eq!(
        cpu.mem.read_u64(TLS + 0x20).unwrap(),
        super::language_code(0)
    );
}

#[test]
fn the_nfc_switch_is_one_switch() {
    const SET_NFC_ENABLE_FLAG: u32 = 70;
    const IS_NFC_ENABLED: u32 = 403;

    let mut cpu = request(false, SET_NFC_ENABLE_FLAG, &[1]);
    cpu.set_sys_request(TLS, Some(SET_NFC_ENABLE_FLAG)).unwrap();
    cpu.register_service_handle(9, "nfc:system");
    write_request(&mut cpu, IS_NFC_ENABLED, &[]);
    cpu.nfc_request(TLS, 9, Some(IS_NFC_ENABLED)).unwrap();
    assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 1, "nfc:sys sees it");

    // And `nfc:sys`'s setter writes the setting the settings applet reads.
    write_request(&mut cpu, 500, &[0]);
    cpu.nfc_request(TLS, 9, Some(500)).unwrap();
    write_request(&mut cpu, 69, &[]);
    cpu.set_sys_request(TLS, Some(69)).unwrap();
    assert_eq!(cpu.mem.read_u8(TLS + 0x20).unwrap(), 0, "set:sys sees that");
}

#[test]
fn set_sys_serves_a_settings_item_and_refuses_one_it_has_not_got() {
    const SIZE_OF: u32 = 37;
    const VALUE_OF: u32 = 38;
    const CATEGORY: u32 = 0x4000;
    const NAME: u32 = 0x4100;
    const OUT: u32 = 0x5000;

    let mut cpu = request(false, SIZE_OF, &[]);
    cpu.mem.map_zero(CATEGORY, 0x200).unwrap();
    cpu.mem.map_zero(OUT, 0x200).unwrap();
    let write_name = |cpu: &mut Cpu, at: u32, text: &str| {
        for (index, &byte) in text.as_bytes().iter().enumerate() {
            cpu.mem.write_u8(at + index as u32, byte).unwrap();
        }
        cpu.mem.write_u8(at + text.len() as u32, 0).unwrap();
    };
    write_name(&mut cpu, CATEGORY, "hbloader");
    write_name(&mut cpu, NAME, "applet_heap_reservation_size");

    let names = [(CATEGORY, 0x48), (NAME, 0x48)];
    write_buffer_request(&mut cpu, SIZE_OF, &[], &names, &[]);
    cpu.set_sys_request(TLS, Some(SIZE_OF)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 8, "a u64 item");

    write_buffer_request(&mut cpu, VALUE_OF, &[], &names, &[(OUT, 8)]);
    cpu.set_sys_request(TLS, Some(VALUE_OF)).unwrap();
    assert_eq!(cpu.mem.read_u64(TLS + 0x20).unwrap(), 8, "and its size");
    assert_eq!(
        cpu.mem.read_u64(OUT).unwrap(),
        0x860_0000,
        "the reservation hbloader reads"
    );

    write_name(&mut cpu, NAME, "applet_heap_size_in_bananas");
    write_buffer_request(&mut cpu, VALUE_OF, &[], &names, &[(OUT, 8)]);
    cpu.set_sys_request(TLS, Some(VALUE_OF)).unwrap();
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x18).unwrap(),
        105 | (11 << 9),
        "ResultSettingsItemNotFound, not a fabricated zero"
    );
}

#[test]
fn set_get_region_code_reports_usa() {
    let mut cpu = request(false, 4, &[]);
    cpu.register_service_handle(9, "set");
    cpu.set_request(TLS, 9, Some(4)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 1, "SetRegion_USA");
}

#[test]
fn set_writes_the_language_codes_into_a_pointer_buffer() {
    // The pre-4.0.0 form fills a receive-static buffer.
    const BUFFER: u32 = 0x4000;
    let mut cpu = request_with_recv_static(1, &[], BUFFER, 0x80);
    cpu.mem.map_zero(BUFFER, 0x100).unwrap();
    cpu.register_service_handle(9, "set");
    cpu.set_request(TLS, 9, Some(1)).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(
        cpu.mem.read_u32(TLS + 0x20).unwrap(),
        15,
        "the pre-4.0.0 count"
    );
    assert_eq!(&cpu.read_bytes(BUFFER, 2), b"ja");
    assert_eq!(&cpu.read_bytes(BUFFER + 8, 5), b"en-US");
    assert_eq!(&cpu.read_bytes(BUFFER + 13 * 8, 5), b"fr-CA");
    // Nothing past the fifteen it reported.
    assert_eq!(cpu.read_bytes(BUFFER + 15 * 8, 8), vec![0u8; 8]);
}

#[test]
fn set_answers_the_pointer_buffer_size_and_not_its_language_count() {
    // Control command 3 is `QueryPointerBufferSize`, not `set`'s own command 3.
    let mut cpu = control_request(3);
    cpu.tpidr = u64::from(TLS);
    cpu.register_service_handle(9, "set");
    cpu.write_zr(0, 9);
    cpu.horizon_syscall(0x20).unwrap();
    assert_eq!(cpu.mem.read_u32(TLS + 0x18).unwrap(), 0, "result");
    assert_eq!(
        cpu.mem.read_u16(TLS + 0x20).unwrap(),
        super::super::ipc::POINTER_BUFFER_SIZE
    );
}
