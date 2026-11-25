use super::*;
use switch_core::cpu::SaveKey;

use crate::activity::switch_log_activity;
use crate::boot::switch_launch;
use crate::container::{open_file, peek, switch_peek_icon};
use crate::debug::switch_set_trace_channels;
use crate::log::switch_log_drain;
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
    switch_set_trace_channels(true);
    let all_on = switch_core::trace::ALL
        .iter()
        .all(|&channel| switch_core::trace::enabled(channel));
    switch_set_trace_channels(false);
    assert!(all_on);
    assert_eq!(switch_core::trace::mask(), 0);
}

/// Every queued log line, collecting from the session first, as `(level, text)`.
fn take_log() -> Vec<(u32, String)> {
    switch_log_drain()
        .into_iter()
        .map(|line| (line[..1].parse().unwrap(), line[1..].to_owned()))
        .collect()
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

#[test]
fn docking_tells_the_page_the_resolution_titles_are_now_offered() {
    let _host = new_session();
    take_log();
    crate::system::switch_set_operation_mode(true);
    crate::system::switch_set_operation_mode(false);
    let log = log::PageLevel::Log as u32;
    assert_eq!(
        take_log(),
        [
            (
                log,
                "Docked - 1080p, boost clocks, no touchscreen.".to_owned()
            ),
            (
                log,
                "Handheld - 720p, normal clocks, touchscreen.".to_owned()
            ),
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
        switch_panic_message(),
        "",
        "the dead session's panic is not this session's"
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

    // A page edit is a change, so the next flush stores it.
    let path = "/settings.dat";
    let body = b"saved";
    switch_fs_write(
        "0100000000001000@000102030405060708090a0b0c0d0e0f",
        path,
        body,
    )
    .unwrap();
    assert_eq!(session().cpu.save_data_mut(key).pending_changes(), 1);
    session().cpu.save_data_mut(key).take_changes();

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
fn a_cache_storage_is_its_own_volume_by_its_stored_name() {
    let _host = new_session();
    switch_fs_write("0100000000001000.cache1", "/blob", b"cached").unwrap();
    let key = SaveKey::cache_storage(0x0100_0000_0000_1000, 1);
    assert_eq!(key.to_string(), "0100000000001000.cache1");
    assert_eq!(session().cpu.save_data_mut(key).size("/blob"), Some(6));
    assert_eq!(
        session()
            .cpu
            .save_data_mut(SaveKey::shared(0x0100_0000_0000_1000))
            .size("/blob"),
        None
    );
    assert!(switch_fs_write("0100000000001000.cachex", "/blob", b"").is_err());
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
    // The page shows why from the log rather than the code.
    let log = switch_log_drain();
    assert!(
        log.iter()
            .any(|line| line.contains("refused the list (the playing user is not in it)")),
        "{log:?}"
    );
}

#[test]
fn the_sd_card_round_trips_through_the_host_entry_points() {
    let _host = new_session();

    // A page edit is a change, so the next flush stores it.
    put("sdmc:/switch/restored.txt", b"hello");
    assert_eq!(
        session().cpu.fs.take_changes(),
        [change("/switch/restored.txt", Some(ENTRY_TYPE_FILE), 5)]
    );

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

/// Open the host container as a dropped file named `name` would be.
fn open_host_container(image: Vec<u8>, name: &str) -> Result<String, String> {
    let src = HostSource {
        file: 0,
        len: image.len() as u64,
    };
    set_host_container(image);
    open_file(session(), src, name, || src)
}

/// `len` bytes of the open container's file `index` from `offset`.
fn read_nsp_file(index: u32, offset: u64, len: u64) -> Vec<u8> {
    nsp_file_source(session(), index)
        .unwrap()
        .read_vec(offset, len)
        .unwrap()
}

/// The container is read through `host_read` without being staged.
#[test]
fn a_container_is_read_through_the_host_without_being_staged() {
    let _host = new_session();
    let payload: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let image = build_nsp(&[("main.nca", &payload), ("notes.txt", b"hello")]);

    let json = open_host_container(image, "game.nsp").unwrap();
    assert!(
        json.starts_with(r#"{"kind":"container","files":["#),
        "{json}"
    );
    assert!(json.contains(r#"{"name":"main.nca","offset":"#), "{json}");
    assert!(json.contains(r#""size":4096}"#), "{json}");
    assert!(json.contains(r#"{"name":"notes.txt""#), "{json}");

    // Reads are relative to the inner file.
    assert_eq!(read_nsp_file(0, 0x1000 - 8, 8), payload[0x1000 - 8..]);
    assert_eq!(read_nsp_file(1, 0, 5), b"hello");
    assert!(nsp_file_source(session(), 7).is_err());

    // A non-NCA payload fails with a readable error.
    let text = switch_launch(Some(0)).unwrap_err();
    assert!(text.contains("bad magic"), "{text}");
}

/// A cleartext Program NCA header, too short to hold an ExeFS.
fn program_nca_header() -> Vec<u8> {
    let mut header = vec![0u8; 0xC00];
    header[0x200..0x204].copy_from_slice(&switch_core::nca::NCA_MAGIC.to_le_bytes());
    header[0x210..0x218].copy_from_slice(&0x0100_0000_0000_1000u64.to_le_bytes());
    header
}

/// A launch without an index boots the Program NCA found when the container opened.
#[test]
fn a_launch_finds_the_program_the_open_scanned() {
    let _host = new_session();
    let image = build_nsp(&[("notes.txt", b"hello"), ("main.nca", &program_nca_header())]);
    assert!(open_host_container(image, "game.nsp")
        .unwrap()
        .contains("main.nca"));
    let text = switch_launch(None).unwrap_err();
    assert!(text.contains("ExeFS"), "{text}");

    let image = build_nsp(&[("notes.txt", b"hello")]);
    open_host_container(image, "empty.nsp").unwrap();
    let text = switch_launch(None).unwrap_err();
    assert!(text.contains("no Program NCA"), "{text}");
}

/// A peek says what a file holds, by its control data, and leaves the open container be.
#[test]
fn a_peek_says_what_a_file_holds_and_leaves_the_open_container_alone() {
    use switch_core::control::testing::nacp;
    use switch_core::nca::testing::romfs_nca;
    use switch_core::nca::{ContentType, ENCRYPTION_AES_CTR_EX, ENCRYPTION_NONE};
    use switch_core::source::SliceSource;

    const BASE: u64 = 0x0100_BCE0_0059_8000;
    const ICON: &[u8] = b"\xFF\xD8\xFF\xE0jpeg";
    let _host = new_session();
    let key = [0x42u8; 32];
    session().keys.header_key = Some(key);
    let romfs = |files: &[(&str, &[u8])]| switch_core::romfs::build(files).unwrap();
    let control = |version: &str| {
        let nacp = nacp("Just Dance 2017", "Ubisoft", version);
        let image = romfs(&[("control.nacp", &nacp), ("icon_AmericanEnglish.dat", ICON)]);
        romfs_nca(&key, ContentType::Control, BASE, &image, ENCRYPTION_NONE)
    };
    let program = |encryption: u8| {
        romfs_nca(
            &key,
            ContentType::Program,
            BASE,
            &romfs(&[("data", b"x")]),
            encryption,
        )
    };
    open_host_container(build_nsp(&[("notes.txt", b"hello")]), "open.nsp").unwrap();

    let game = build_nsp(&[
        ("program.nca", &program(ENCRYPTION_NONE)),
        ("control.nca", &control("1.0.0")),
    ]);
    assert_eq!(
        peek(session(), &SliceSource(&game)).unwrap(),
        r#"{"kind":"game","title_id":"0100bce000598000","name":"Just Dance 2017","publisher":"Ubisoft","version":"1.0.0","icon_mime":"image/jpeg","icon_size":8}"#
    );
    assert_eq!(switch_peek_icon(), ICON);
    assert!(switch_peek_icon().is_empty());

    let update = build_nsp(&[
        ("program.nca", &program(ENCRYPTION_AES_CTR_EX)),
        ("control.nca", &control("1.0.1")),
    ]);
    assert_eq!(
        peek(session(), &SliceSource(&update)).unwrap(),
        r#"{"kind":"update","title_id":"0100bce000598000","version":"1.0.1"}"#
    );

    let pack = romfs_nca(
        &key,
        ContentType::Data,
        BASE + 0x1001,
        &romfs(&[("song", b"x")]),
        ENCRYPTION_NONE,
    );
    let dlc = build_nsp(&[("pack.nca", &pack)]);
    assert_eq!(
        peek(session(), &SliceSource(&dlc)).unwrap(),
        r#"{"kind":"dlc","title_id":"0100bce000598000","packs":[{"id":"0100bce000599001","title_id":"0100bce000598000","index":1}]}"#
    );

    let nro = switch_core::nro::testing::with_assets(
        ICON,
        &nacp("NX-Shell", "DefenderOfHyrule", "4.0.2"),
    );
    assert_eq!(
        peek(session(), &SliceSource(&nro)).unwrap(),
        r#"{"kind":"homebrew","title_id":"","name":"NX-Shell","publisher":"DefenderOfHyrule","version":"4.0.2","icon_mime":"image/jpeg","icon_size":8}"#
    );

    let text = peek(session(), &SliceSource(&[b'x'; 0x400])).unwrap_err();
    assert!(text.contains("magic"), "{text}");
    session().keys.header_key = None;
    let text = peek(session(), &SliceSource(&game)).unwrap_err();
    assert!(text.contains("load prod.keys"), "{text}");

    // The container opened before is still the one open, with nothing held for it.
    assert_eq!(read_nsp_file(0, 0, 5), b"hello");
    assert!(session().update.is_none() && session().dlc.is_empty());
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

    let json = open_host_container(image, "game.xci").unwrap();
    assert!(json.contains(r#"{"name":"program.nca""#), "{json}");
    assert!(json.contains(r#"{"name":"meta.cnmt.nca""#), "{json}");
    assert!(!json.contains("system.nca"), "{json}");

    // Offsets are the image's own.
    assert_eq!(read_nsp_file(0, 0x700, 16), payload[0x700..0x710]);
}

#[test]
fn an_error_message_comes_back_whole() {
    let _host = new_session();
    const MSG: &str = "no container is open";
    session().last_error = MSG.to_string();
    assert_eq!(switch_last_error(), MSG);
}

/// The thread table reads the activity report's last reading, which reading it does not drain.
#[test]
fn the_thread_table_shows_the_last_reading_of_the_activity_report() {
    use crate::stats::switch_threads_json;
    let _guard = new_session();
    assert_eq!(switch_threads_json(), r#"{"seconds":null,"threads":[]}"#);

    session().cpu.steps += 5000;
    switch_log_activity(0.0);
    let first = switch_threads_json();
    assert!(
        first.starts_with(r#"{"seconds":null,"threads":[{"index":0,"#)
            && first.contains(r#""running":true,"ran":5000,"#),
        "the first reading covers no known span: {first}"
    );

    session().cpu.steps += 2000;
    switch_log_activity(0.5);
    let second = switch_threads_json();
    assert!(
        second.starts_with(r#"{"seconds":0.5,"#) && second.contains(r#""ran":2000,"#),
        "{second}"
    );
    assert_eq!(switch_threads_json(), second, "read, not taken");

    crate::boot::switch_end_title();
    assert_eq!(switch_threads_json(), r#"{"seconds":null,"threads":[]}"#);
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
            r#""emitted":0,"enteredEmitted":0,"chained":0,"retired":0,"retiredEmitted":0,"#,
            r#""dropped":0}"#
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
    use crate::container::{switch_control_json, switch_inspect_nca};
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

    set_host_container(vec![0u8; 16]);
    session().container = Some(HostSource { file: 0, len: 16 });
    assert_eq!(
        switch_inspect_nca(None),
        r#"{"error":"NCA header: expected at least 1024 bytes, got 16"}"#
    );
}

/// The page writes its state and the next slice takes it.
fn set_input(buttons: u32, slx: i32, sly: i32, srx: i32, sry: i32) {
    crate::input::page_write(buttons, [slx, sly, srx, sry]);
    crate::input::exchange(&mut session().cpu);
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
    set_input(0b1, 0, 0, 0, 0);
    set_input(0, 0, 0, 0, 0);
    assert_eq!(published_buttons(), 0b1, "released before any frame ran");

    advance_frames(1);
    assert_eq!(published_buttons(), 0b1, "one frame may not have been seen");

    advance_frames(1);
    assert_eq!(published_buttons(), 0, "two frames have passed");
}

#[test]
fn a_tap_the_page_wrote_and_lifted_between_slices_is_still_seen() {
    let _host = new_session();
    crate::input::page_write(0b1, [0; 4]);
    crate::input::page_write(0, [0; 4]);
    crate::input::exchange(&mut session().cpu);
    assert_eq!(published_buttons(), 0b1);

    advance_frames(2);
    assert_eq!(published_buttons(), 0);
}

#[test]
fn a_held_button_stays_down_after_its_latch_lets_go() {
    let _host = new_session();
    set_input(0b10, 0, 0, 0, 0);
    advance_frames(2);
    assert_eq!(published_buttons(), 0b10);
    set_input(0, 0, 0, 0, 0);
    assert_eq!(published_buttons(), 0);
}

#[test]
fn a_press_is_let_go_after_enough_slices_even_if_nothing_is_presented() {
    let _host = new_session();
    set_input(0b100, 0, 0, 0, 0);
    set_input(0, 0, 0, 0, 0);
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
    set_input(0, 30000, 0, 0, 0);
    set_input(0, 0, 0, 0, 0);
    assert_eq!(stick(&session().cpu), 30000);
    advance_frames(2);
    assert_eq!(stick(&session().cpu), 0);

    set_input(0b1, 0, 0, 0, 0);
    crate::input::reset();
    set_input(0b1, 0, 0, 0, 0);
    set_input(0, 0, 0, 0, 0);
    advance_frames(0);
    assert_eq!(
        published_buttons(),
        0b1,
        "a press after a reset latches afresh"
    );
}
