use super::*;
use switch_core::cpu::SaveKey;

use crate::activity::switch_log_activity;
use crate::boot::switch_load_nca_from_nsp;
use crate::container::{switch_open_nsp, switch_read_file};
use crate::debug::switch_set_trace_channels;
use crate::input::switch_set_input;
use crate::log::{switch_log_line, switch_log_next};
use crate::storage::{switch_fs_create_dir, switch_fs_write};
use crate::users::{switch_user_stage, switch_users_commit};
use switch_core::vfs::{Change, ENTRY_TYPE_DIR, ENTRY_TYPE_FILE};

/// Serializes tests: the session table is not thread-safe under `cargo test`.
static HOST: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A session and the lock. Restores the default panic hook so failures are readable.
fn new_session() -> std::sync::MutexGuard<'static, ()> {
    let guard = HOST.lock().unwrap_or_else(|e| e.into_inner());
    switch_init();
    let _ = std::panic::take_hook();
    guard
}

#[test]
fn a_panic_message_is_cut_between_characters_not_through_one() {
    // Truncation must not split a character.
    let msg = "PANIC: ⚠⚠⚠⚠";
    for limit in 0..msg.len() {
        let n = floor_char_boundary(msg, limit);
        assert!(n <= limit);
        assert!(
            std::str::from_utf8(&msg.as_bytes()[..n]).is_ok(),
            "cut at {limit} split a character"
        );
    }
    assert_eq!(floor_char_boundary(msg, msg.len() + 10), msg.len());
}

#[test]
fn the_page_turns_every_trace_channel_on_and_off() {
    let _host = HOST.lock().unwrap_or_else(|e| e.into_inner());
    switch_set_trace_channels(1);
    let all_on = switch_core::trace::ALL
        .iter()
        .all(|&channel| switch_core::trace::enabled(channel));
    switch_set_trace_channels(0);
    assert!(all_on);
    assert_eq!(switch_core::trace::mask(), 0);
}

/// Every queued log line, collecting from the session first, as `(level, text)`.
fn take_log() -> Vec<(u32, String)> {
    let mut lines = Vec::new();
    loop {
        let level = switch_log_next();
        if level == 0 {
            return lines;
        }
        lines.push((level, switch_log_line()));
    }
}

#[test]
fn the_activity_report_names_each_file_and_hands_it_over_once() {
    let _host = new_session();
    take_log();
    switch_log_activity(0.0);
    let first = take_log();
    assert!(
        first
            .iter()
            .any(|(_, line)| line.starts_with("[thread] 0* (handle ")),
        "{first:?}"
    );
    assert!(
        !first.iter().any(|(_, line)| line.starts_with("[fs]")),
        "{first:?}"
    );

    let activity = &mut session().cpu.fs_activity;
    activity.read("romfs", 0x100);
    activity.read("romfs", 0x20);
    activity.wrote("sdmc:/cfg.json", 7);
    activity.record("fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)".to_owned());
    switch_log_activity(1.0);
    let fs: Vec<(u32, String)> = take_log()
        .into_iter()
        .filter(|(_, line)| line.starts_with("[fs]"))
        .collect();
    let debug = log::PageLevel::Debug as u32;
    assert_eq!(
        fs,
        [
            (debug, "[fs] romfs: 2 reads (288 B)".to_owned()),
            (debug, "[fs] sdmc:/cfg.json: 1 write (7 B)".to_owned()),
            (
                debug,
                "[fs] fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)".to_owned()
            ),
        ]
    );
    switch_log_activity(1.0);
    let again = take_log();
    assert!(
        !again.iter().any(|(_, line)| line.starts_with("[fs]")),
        "taken, not read: {again:?}"
    );
}

#[test]
fn each_line_carries_its_level_so_the_page_reads_none_of_its_text() {
    let _host = new_session();
    take_log();
    let cpu = &mut session().cpu;
    cpu.out
        .extend_from_slice(b"[lm/WARN/nn::fs] low on space\n[lm/TRACE] tick\nplain stdout\n");
    cpu.trace.push(switch_core::trace::Level::Error.marker());
    cpu.trace.extend_from_slice(b"fault at 0x80\n  x0=0x1\n");
    use log::PageLevel::*;
    assert_eq!(
        take_log(),
        [
            (Warn as u32, "[lm/WARN/nn::fs] low on space".to_owned()),
            (Debug as u32, "[lm/TRACE] tick".to_owned()),
            (Log as u32, "plain stdout".to_owned()),
            (Error as u32, "fault at 0x80".to_owned()),
            (Error as u32, "  x0=0x1".to_owned()),
        ]
    );
}

/// A new session starts without the old one's panic, but keeps every line it logged.
#[test]
fn a_new_session_inherits_the_log_but_not_the_panic_of_the_one_before_it() {
    let _host = new_session();
    take_log();

    // SAFETY: single-threaded under the `HOST` lock.
    let planted = b"PANIC: the old session died here";
    let guard = unsafe { &mut *PANIC_MSG.get() };
    guard[..planted.len()].copy_from_slice(planted);
    guard[planted.len()] = 0;
    switch_core::trace::emit("[test] the old session traced this");
    session()
        .cpu
        .out
        .extend_from_slice(b"the old session printed this\n");

    switch_init();

    let lines: Vec<String> = take_log().into_iter().map(|(_, line)| line).collect();
    for kept in [
        "the old session printed this",
        "[test] the old session traced this",
    ] {
        assert!(
            lines.iter().any(|line| line == kept),
            "{kept:?} was dropped: {lines:?}"
        );
    }

    assert_eq!(
        switch_last_error(),
        "",
        "the dead session's panic is not this session's last error"
    );
}

fn change(path: &str, kind: Option<u8>, size: u64) -> Change {
    Change {
        path: path.to_owned(),
        kind,
        size,
    }
}

fn put(path: &str, data: &[u8]) {
    switch_fs_write("sd", path, data).unwrap();
}

#[test]
fn save_data_round_trips_and_stays_out_of_the_sd_card() {
    const SAVE: u64 = 0x0100_0000_0000_1000;
    const USER_LO: u64 = 0x0706_0504_0302_0100;
    const USER_HI: u64 = 0x0f0e_0d0c_0b0a_0908;
    let key = SaveKey::from_halves(SAVE, USER_LO, USER_HI);
    let _host = new_session();

    // Restores are not reported as changes.
    let path = "/settings.dat";
    let body = b"saved";
    switch_fs_write(
        "0100000000001000@000102030405060708090a0b0c0d0e0f",
        path,
        body,
    )
    .unwrap();
    assert_eq!(session().cpu.save_data_mut(key).pending_changes(), 0);

    // A save with only a directory is listed; the shared save is a separate one.
    switch_fs_create_dir("0100000000001000", "/").unwrap();
    let mut ids: Vec<String> = session()
        .cpu
        .save_keys()
        .iter()
        .map(ToString::to_string)
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        [
            "0100000000001000",
            "0100000000001000@000102030405060708090a0b0c0d0e0f"
        ]
    );

    // A guest write is a change, in the save rather than on the card.
    session()
        .cpu
        .save_data_mut(key)
        .write("/settings.dat", 0, b"12345")
        .unwrap();
    assert_eq!(session().cpu.save_data_mut(key).pending_changes(), 1);
    assert_eq!(
        session()
            .cpu
            .save_data_mut(SaveKey::shared(SAVE))
            .pending_changes(),
        0
    );
    assert_eq!(
        session().cpu.save_data_mut(key).take_changes(),
        [change("/settings.dat", Some(ENTRY_TYPE_FILE), 5)]
    );
    assert_eq!(session().cpu.save_data_mut(key).pending_changes(), 0);
    assert_eq!(session().cpu.fs.entry_type("/settings.dat"), None);

    assert_eq!(session().cpu.save_data_mut(key).size(path), Some(5));
    assert_eq!(
        session()
            .cpu
            .save_data_mut(SaveKey::from_halves(SAVE, 0, 0))
            .size(path),
        None,
        "another user's save, or the shared one, does not have it"
    );
    let mut out = [0u8; 16];
    assert_eq!(
        session().cpu.save_data_mut(key).read(path, 0, &mut out),
        Some(5)
    );
    assert_eq!(&out[..5], b"12345");
}

#[test]
fn users_are_staged_committed_and_read_back() {
    let _host = new_session();
    let ann = "000102030405060708090a0b0c0d0e0f";
    let ben = "11110000000000002222000000000000";
    let picture = [0xFFu8, 0xD8, 0xFF, 0xD9];
    let stage = |uid: &str, name: &str, picture: Option<Vec<u8>>| {
        switch_user_stage(uid, name, 1_700_000_000.0, picture)
    };
    stage(ann, "Ann \"A\"", Some(picture.to_vec()));
    stage(ben, "Ben", None);
    assert_eq!(switch_users_commit(ben), 0);
    assert_eq!(session().cpu.user_nickname(), "Ben");

    let users = session().cpu.users();
    assert_eq!(users.len(), 2);
    assert_eq!(
        users[0].uid,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
    assert_eq!(users[0].nickname, "Ann \"A\"");
    assert_eq!(users[0].edited_at, 1_700_000_000);
    assert_eq!(users[0].picture.as_deref(), Some(&picture[..]));
    assert_eq!(users[1].nickname, "Ben");
    assert_eq!(users[1].picture, None);

    // A list naming a player not in it is refused and changes nothing.
    stage(ann, "Ann", None);
    assert_eq!(switch_users_commit("09000000000000000900000000000000"), 4);
    assert_eq!(session().cpu.users().len(), 2);
    assert!(!session().cpu.take_profile_edits());
}

#[test]
fn the_sd_card_round_trips_through_the_host_entry_points() {
    let _host = new_session();

    // Restores are not reported as changes.
    put("sdmc:/switch/restored.txt", b"hello");
    assert_eq!(session().cpu.fs.pending_changes(), 0);
    assert!(session().cpu.fs.take_changes().is_empty());

    // A guest write is: `IFile::Write` at offset 0 of a new file.
    {
        let cpu = &mut session().cpu;
        assert!(cpu.fs.create_file("/switch/cfg.json", 0));
        cpu.fs.write("/switch/cfg.json", 0, br#"{"v":5}"#).unwrap();
        cpu.fs.guest_create_dir("/switch/saves");
        cpu.fs.remove("/switch/restored.txt");
    }
    assert_eq!(session().cpu.fs.pending_changes(), 3);
    assert_eq!(
        session().cpu.fs.take_changes(),
        [
            change("/switch/cfg.json", Some(ENTRY_TYPE_FILE), 7),
            change("/switch/restored.txt", None, 0),
            change("/switch/saves", Some(ENTRY_TYPE_DIR), 0),
        ]
    );
    // Draining clears them.
    assert_eq!(session().cpu.fs.pending_changes(), 0);
    assert!(session().cpu.fs.take_changes().is_empty());

    let fs = &session().cpu.fs;
    let path = "/switch/cfg.json";
    assert_eq!(fs.size(path), Some(7));
    let mut out = [0u8; 16];
    assert_eq!(fs.read(path, 0, &mut out), Some(7));
    assert_eq!(&out[..7], br#"{"v":5}"#);
    assert_eq!(fs.read(path, 4, &mut out), Some(3));
    assert_eq!(&out[..3], b":5}");
    assert_eq!(fs.size("/switch"), None);
    assert_eq!(fs.read("/switch/nope", 0, &mut out), None);
}

/// Build a PFS0: header, entry table, string table, payloads.
fn build_nsp(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut names = Vec::new();
    let mut name_offsets = Vec::new();
    for (name, _) in files {
        name_offsets.push(names.len() as u32);
        names.extend_from_slice(name.as_bytes());
        names.push(0);
    }
    let entries_end = 0x10 + files.len() * 24;
    let payload_base = entries_end + names.len();

    let mut image = Vec::new();
    image.extend_from_slice(&0x3053_4650u32.to_le_bytes()); // "PFS0"
    image.extend_from_slice(&(files.len() as u32).to_le_bytes());
    image.extend_from_slice(&(names.len() as u32).to_le_bytes());
    image.extend_from_slice(&0u32.to_le_bytes());
    let mut at = payload_base as u64;
    for (i, (_, payload)) in files.iter().enumerate() {
        image.extend_from_slice(&at.to_le_bytes());
        image.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        image.extend_from_slice(&name_offsets[i].to_le_bytes());
        image.extend_from_slice(&0u32.to_le_bytes());
        at += payload.len() as u64;
    }
    image.extend_from_slice(&names);
    for (_, payload) in files {
        image.extend_from_slice(payload);
    }
    image
}

/// The container is read through `host_read` without being staged.
#[test]
fn a_container_is_read_through_the_host_without_being_staged() {
    let _host = new_session();
    let payload: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let image = build_nsp(&[("main.nca", &payload), ("notes.txt", b"hello")]);
    let size = image.len() as u64;
    set_host_container(image);

    let json = switch_open_nsp(size);
    assert!(json.contains(r#"{"name":"main.nca","offset":"#), "{json}");
    assert!(json.contains(r#""size":4096}"#), "{json}");
    assert!(json.contains(r#"{"name":"notes.txt""#), "{json}");

    // Reads are relative to the inner file and stop at its end.
    assert_eq!(
        switch_read_file(0, 0x1000 - 8, 32).unwrap(),
        payload[0x1000 - 8..]
    );
    assert_eq!(switch_read_file(1, 0, 32).unwrap(), b"hello");

    // Past a file's end is empty; past the table is an error.
    assert_eq!(switch_read_file(1, 5, 32).unwrap(), b"");
    assert!(switch_read_file(7, 0, 32).is_err());

    // A non-NCA payload fails with a readable error.
    assert_eq!(switch_load_nca_from_nsp(0), -1);
    let text = switch_last_error();
    assert!(text.contains("bad magic"), "{text}");
}

/// A cartridge image opens through the same entry point as an `.nsp`.
#[test]
fn a_cartridge_image_opens_as_a_container() {
    use switch_core::nsp::testing::partition_fs;
    use switch_core::nsp::PartitionKind;

    let _host = new_session();
    let payload: Vec<u8> = (0..=255u8).cycle().take(2048).collect();
    let secure = partition_fs(
        PartitionKind::Hfs0,
        &[("program.nca", &payload), ("meta.cnmt.nca", b"cnmt")],
    );
    // The firmware partition's NCAs are not listed as the game's.
    let update = partition_fs(PartitionKind::Hfs0, &[("system.nca", b"firmware")]);
    let image = switch_core::xci::testing::cartridge(&[("update", &update), ("secure", &secure)]);
    let size = image.len() as u64;
    set_host_container(image);

    let json = switch_open_nsp(size);
    assert!(json.contains(r#"{"name":"program.nca""#), "{json}");
    assert!(json.contains(r#"{"name":"meta.cnmt.nca""#), "{json}");
    assert!(!json.contains("system.nca"), "{json}");

    // Offsets are the image's own.
    assert_eq!(
        switch_read_file(0, 0x700, 16).unwrap(),
        payload[0x700..0x710]
    );
}

#[test]
fn an_error_message_comes_back_whole() {
    let _host = new_session();
    const MSG: &str = "no container is open";
    session().last_error = MSG.to_string();
    assert_eq!(switch_last_error(), MSG);
}

/// The translator's and the GPU's reports, in the bytes the page parses.
#[test]
fn the_translator_and_gpu_reports_are_the_json_the_page_reads() {
    use crate::stats::{switch_gpu_report_json, switch_jit_stats_json};
    let _guard = new_session();
    assert_eq!(
        switch_jit_stats_json(),
        concat!(
            r#"{"enabled":true,"blocks":0,"translated":0,"executed":0,"linked":0,"#,
            r#""invalidated":0,"interpreted":0,"interpretedGroups":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"#,
            r#""emitted":0,"enteredEmitted":0,"chained":0}"#
        )
    );
    let memory = crate::stats::switch_memory_json();
    assert!(
        memory.starts_with(r#"{"backed":"#)
            && memory.contains(r#"{"key":"code","start":134217728,"end":402653184,"bytes":"#),
        "{memory}"
    );
    assert!(
        memory.contains(r#""elsewhere":"#) && memory.ends_with('}'),
        "{memory}"
    );
    // The software rasterizer reports nothing of its own.
    assert_eq!(switch_gpu_report_json(), r#"{"frames":0}"#);
}

/// A title's control data and an unreadable NCA, in the bytes the page parses.
#[test]
fn control_data_and_nca_summaries_are_the_json_the_page_reads() {
    use crate::container::{switch_control_json, switch_parse_nca};
    use switch_core::control::{CaptureSupport, Control, Nacp, Rating, StartupUserAccount, Title};
    let _guard = new_session();
    assert_eq!(switch_control_json(), "{}", "nothing cached");

    let mut nacp = Nacp::empty();
    nacp.titles = vec![
        Title {
            language: "AmericanEnglish",
            name: "JUST DANCE® 2017".into(),
            publisher: "Ubisoft".into(),
        },
        Title {
            language: "Japanese",
            name: "日本語".into(),
            publisher: "\"Q\"".into(),
        },
    ];
    nacp.display_version = "1.0.2".into();
    nacp.isbn = String::new();
    nacp.is_demo = true;
    nacp.startup_user_account = StartupUserAccount::Required;
    nacp.screenshot = CaptureSupport::Screenshot(1);
    nacp.video_capture = CaptureSupport::Video(2);
    nacp.ratings = vec![Rating {
        organisation: "PEGI",
        age: 12,
    }];
    nacp.add_on_content_base_id = 0x0100_0000_0000_2000;
    nacp.save_data_owner_id = 0;
    nacp.user_account_save_data_size = 0x40_0000;
    nacp.user_account_save_data_journal_size = 0x10_0000;
    nacp.device_save_data_size = -1;
    nacp.device_save_data_journal_size = 2;
    nacp.bcat_delivery_cache_storage_size = 3;
    nacp.application_error_code_category = "2181\n".into();
    session().control = Some(Control {
        title_id: 0x0100_0000_0000_1000,
        language: "AmericanEnglish",
        name: "JUST DANCE® 2017".into(),
        publisher: "Ubi \\ soft".into(),
        icon: vec![0xFF, 0xD8, 0xFF, 0xE0],
        nacp,
        nacp_raw: Vec::new(),
    });
    assert_eq!(
        switch_control_json(),
        concat!(
            r#"{"title_id":"0100000000001000","name":"JUST DANCE® 2017","publisher":"Ubi \\ soft","#,
            r#""language":"AmericanEnglish","version":"1.0.2","isbn":"","error_code_category":"2181\n","#,
            r#""startup_user_account":"required","screenshot":"blocked","video_capture":"enabled","#,
            r#""demo":true,"languages":["AmericanEnglish","Japanese"],"#,
            r#""ratings":[{"organisation":"PEGI","age":12}],"#,
            r#""add_on_content_base_id":"0100000000002000","save_data_owner_id":"0000000000000000","#,
            r#""user_save_size":4194304,"user_save_journal_size":1048576,"device_save_size":-1,"#,
            r#""device_save_journal_size":2,"bcat_storage_size":3,"icon_mime":"image/jpeg","icon_size":4}"#
        )
    );

    assert_eq!(
        switch_parse_nca(&[0u8; 16]),
        r#"{"error":"NCA header: expected at least 1024 bytes, got 16"}"#
    );
}

fn published_buttons() -> u32 {
    session()
        .cpu
        .mem
        .read_u32(switch_core::INPUT_ADDR)
        .expect("the input register is mapped")
}

fn advance_frames(n: u64) {
    session().cpu.nv.gpu.frames += n;
    crate::input::release_if_seen(&mut session().cpu);
}

#[test]
fn a_tap_between_slices_stays_down_until_the_guest_has_had_a_frame() {
    let _host = new_session();
    switch_set_input(0b1, 0, 0, 0, 0);
    switch_set_input(0, 0, 0, 0, 0);
    assert_eq!(published_buttons(), 0b1, "released before any frame ran");

    advance_frames(1);
    assert_eq!(published_buttons(), 0b1, "one frame may not have been seen");

    advance_frames(1);
    assert_eq!(published_buttons(), 0, "two frames have passed");
}

#[test]
fn a_held_button_stays_down_after_its_latch_lets_go() {
    let _host = new_session();
    switch_set_input(0b10, 0, 0, 0, 0);
    advance_frames(2);
    assert_eq!(published_buttons(), 0b10);
    switch_set_input(0, 0, 0, 0, 0);
    assert_eq!(published_buttons(), 0);
}

#[test]
fn a_press_is_let_go_after_enough_slices_even_if_nothing_is_presented() {
    let _host = new_session();
    switch_set_input(0b100, 0, 0, 0, 0);
    switch_set_input(0, 0, 0, 0, 0);
    for _ in 0..63 {
        advance_frames(0);
    }
    assert_eq!(published_buttons(), 0b100);
    advance_frames(0);
    assert_eq!(published_buttons(), 0);
}

#[test]
fn a_stick_flick_latches_like_a_press_and_a_reset_forgets_it() {
    let _host = new_session();
    let stick = |cpu: &switch_core::cpu::Cpu| {
        cpu.mem
            .read_u32(switch_core::INPUT_ADDR + 8)
            .expect("the input register is mapped") as i32
    };
    switch_set_input(0, 30000, 0, 0, 0);
    switch_set_input(0, 0, 0, 0, 0);
    assert_eq!(stick(&session().cpu), 30000);
    advance_frames(2);
    assert_eq!(stick(&session().cpu), 0);

    switch_set_input(0b1, 0, 0, 0, 0);
    crate::input::reset();
    switch_set_input(0b1, 0, 0, 0, 0);
    switch_set_input(0, 0, 0, 0, 0);
    advance_frames(0);
    assert_eq!(
        published_buttons(),
        0b1,
        "a press after a reset latches afresh"
    );
}
