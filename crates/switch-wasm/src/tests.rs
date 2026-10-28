use super::*;
use switch_core::cpu::SaveKey;

use crate::activity::switch_log_activity;
use crate::boot::switch_load_nca_from_nsp;
use crate::container::{switch_nsp_files_json, switch_open_nsp, switch_read_file};
use crate::debug::switch_set_trace_channels;
use crate::log::{switch_log_line, switch_log_next};
use crate::storage::{
    switch_save_create_dir, switch_save_file_size, switch_save_ids_json,
    switch_save_pending_changes, switch_save_read_file, switch_save_take_changes_json,
    switch_save_write_file, switch_sd_file_size, switch_sd_pending_changes, switch_sd_read_file,
    switch_sd_take_changes_json, switch_sd_write_file,
};
use crate::users::{
    switch_take_profile_edits, switch_user_picture, switch_user_stage, switch_users_commit,
    switch_users_json,
};

/// Serializes tests: the session table is not thread-safe under `cargo test`.
static HOST: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A session and the lock. Restores the default panic hook so failures are readable.
fn new_session() -> (std::sync::MutexGuard<'static, ()>, u32) {
    let guard = HOST.lock().unwrap_or_else(|e| e.into_inner());
    let handle = switch_new();
    let _ = std::panic::take_hook();
    (guard, handle)
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

/// Every queued log line, collecting from `handle` first, as `(level, text)`.
fn take_log(handle: i32) -> Vec<(u32, String)> {
    let mut lines = Vec::new();
    loop {
        let level = switch_log_next(handle);
        if level == 0 {
            return lines;
        }
        let mut buf = vec![0u8; switch_log_line(std::ptr::null_mut(), 0) as usize];
        let n = switch_log_line(buf.as_mut_ptr(), buf.len() as u32);
        buf.truncate(n as usize);
        lines.push((level, String::from_utf8(buf).unwrap()));
    }
}

#[test]
fn the_activity_report_names_each_file_and_hands_it_over_once() {
    let (_host, handle) = new_session();
    take_log(handle as i32);
    switch_log_activity(handle, 0.0);
    let first = take_log(handle as i32);
    assert!(
        first
            .iter()
            .any(|(_, line)| line.starts_with("[thread] 0* (the process entry, ")),
        "{first:?}"
    );
    assert!(
        !first.iter().any(|(_, line)| line.starts_with("[fs]")),
        "{first:?}"
    );

    let activity = &mut session(handle).cpu.fs_activity;
    activity.read("romfs", 0x100);
    activity.read("romfs", 0x20);
    activity.wrote("sdmc:/cfg.json", 7);
    activity.record("fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)".to_owned());
    switch_log_activity(handle, 1.0);
    let fs: Vec<(u32, String)> = take_log(handle as i32)
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
    switch_log_activity(handle, 1.0);
    let again = take_log(handle as i32);
    assert!(
        !again.iter().any(|(_, line)| line.starts_with("[fs]")),
        "taken, not read: {again:?}"
    );
}

#[test]
fn each_line_carries_its_level_so_the_page_reads_none_of_its_text() {
    let (_host, handle) = new_session();
    take_log(handle as i32);
    let cpu = &mut session(handle).cpu;
    cpu.out
        .extend_from_slice(b"[lm/WARN/nn::fs] low on space\n[lm/TRACE] tick\nplain stdout\n");
    cpu.trace.push(switch_core::trace::Level::Error.marker());
    cpu.trace.extend_from_slice(b"fault at 0x80\n  x0=0x1\n");
    use log::PageLevel::*;
    assert_eq!(
        take_log(handle as i32),
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
    let (_host, first) = new_session();
    take_log(first as i32);

    // SAFETY: single-threaded under the `HOST` lock.
    let planted = b"PANIC: the old session died here";
    let guard = unsafe { &mut *PANIC_MSG.get() };
    guard[..planted.len()].copy_from_slice(planted);
    guard[planted.len()] = 0;
    switch_core::trace::emit("[test] the old session traced this");
    session(first)
        .cpu
        .out
        .extend_from_slice(b"the old session printed this\n");

    switch_free_session(first);
    let second = switch_new();

    let lines: Vec<String> = take_log(second as i32)
        .into_iter()
        .map(|(_, line)| line)
        .collect();
    for kept in [
        "the old session printed this",
        "[test] the old session traced this",
    ] {
        assert!(
            lines.iter().any(|line| line == kept),
            "{kept:?} was dropped: {lines:?}"
        );
    }

    let mut buf = [0u8; 256];
    let n = switch_last_error(second, buf.as_mut_ptr(), buf.len() as u32);
    assert_eq!(
        &buf[..n as usize],
        b"",
        "the dead session's panic is not this session's last error"
    );
}

fn take_changes(handle: u32) -> String {
    let cap = 64 * 1024;
    let mut buf = vec![0u8; cap];
    let n = switch_sd_take_changes_json(handle, buf.as_mut_ptr(), cap as u32);
    String::from_utf8(buf[..n as usize].to_vec()).unwrap()
}

fn put(handle: u32, path: &str, data: &[u8]) {
    switch_sd_write_file(
        handle,
        path.as_ptr(),
        path.len() as u32,
        data.as_ptr(),
        data.len() as u32,
    );
}

#[test]
fn save_data_round_trips_and_stays_out_of_the_sd_card() {
    const SAVE: u64 = 0x0100_0000_0000_1000;
    const USER_LO: u64 = 0x0706_0504_0302_0100;
    const USER_HI: u64 = 0x0f0e_0d0c_0b0a_0908;
    let key = SaveKey::from_halves(SAVE, USER_LO, USER_HI);
    let (_host, handle) = new_session();

    // Restores are not reported as changes.
    let path = "/settings.dat";
    let body = b"saved";
    assert_eq!(
        switch_save_write_file(
            handle,
            SAVE,
            USER_LO,
            USER_HI,
            path.as_ptr(),
            path.len() as u32,
            body.as_ptr(),
            body.len() as u32,
        ),
        0
    );
    assert_eq!(
        switch_save_pending_changes(handle, SAVE, USER_LO, USER_HI),
        0
    );

    // A save with only a directory is listed; the shared save is a separate one.
    switch_save_create_dir(handle, SAVE, 0, 0, b"/".as_ptr(), 1);
    let mut ids = [0u8; 128];
    let n = switch_save_ids_json(handle, ids.as_mut_ptr(), ids.len() as u32) as usize;
    assert_eq!(
        std::str::from_utf8(&ids[..n]).unwrap(),
        r#"["0100000000001000","0100000000001000@000102030405060708090a0b0c0d0e0f"]"#
    );

    // A guest write is a change, in the save rather than on the card.
    session(handle)
        .cpu
        .save_data_mut(key)
        .write("/settings.dat", 0, b"12345")
        .unwrap();
    assert_eq!(
        switch_save_pending_changes(handle, SAVE, USER_LO, USER_HI),
        1
    );
    assert_eq!(switch_save_pending_changes(handle, SAVE, 0, 0), 0);
    let mut buf = [0u8; 256];
    let n = switch_save_take_changes_json(
        handle,
        SAVE,
        USER_LO,
        USER_HI,
        buf.as_mut_ptr(),
        buf.len() as u32,
    );
    assert_eq!(
        std::str::from_utf8(&buf[..n as usize]).unwrap(),
        r#"[{"path":"/settings.dat","kind":"file","size":5}]"#
    );
    assert_eq!(
        switch_save_pending_changes(handle, SAVE, USER_LO, USER_HI),
        0
    );
    assert_eq!(session(handle).cpu.fs.entry_type("/settings.dat"), None);

    assert_eq!(
        switch_save_file_size(
            handle,
            SAVE,
            USER_LO,
            USER_HI,
            path.as_ptr(),
            path.len() as u32
        ),
        5
    );
    assert_eq!(
        switch_save_file_size(handle, SAVE, 0, 0, path.as_ptr(), path.len() as u32),
        -1,
        "another user's save, or the shared one, does not have it"
    );
    let mut out = [0u8; 16];
    let read = switch_save_read_file(
        handle,
        SAVE,
        USER_LO,
        USER_HI,
        path.as_ptr(),
        path.len() as u32,
        0,
        out.as_mut_ptr(),
        out.len() as u32,
    );
    assert_eq!(read, 5);
    assert_eq!(&out[..5], b"12345");
}

#[test]
fn users_are_staged_committed_and_read_back() {
    let (_host, handle) = new_session();
    let (ann_lo, ann_hi) = (0x0706_0504_0302_0100u64, 0x0f0e_0d0c_0b0a_0908u64);
    let (ben_lo, ben_hi) = (0x1111u64, 0x2222u64);
    let picture = [0xFFu8, 0xD8, 0xFF, 0xD9];
    let stage = |lo, hi, name: &str, picture: &[u8]| {
        switch_user_stage(
            handle,
            lo,
            hi,
            name.as_ptr(),
            name.len() as u32,
            1_700_000_000,
            picture.as_ptr(),
            picture.len() as u32,
        )
    };
    stage(ann_lo, ann_hi, "Ann \"A\"", &picture);
    stage(ben_lo, ben_hi, "Ben", &[]);
    assert_eq!(switch_users_commit(handle, ben_lo, ben_hi), 0);
    assert_eq!(session(handle).cpu.user_nickname(), "Ben");

    let mut buf = [0u8; 512];
    let n = switch_users_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
    assert_eq!(
        std::str::from_utf8(&buf[..n]).unwrap(),
        concat!(
            r#"[{"uid":"000102030405060708090a0b0c0d0e0f","nickname":"Ann \"A\"","#,
            r#""editedAt":1700000000,"pictureLen":4},"#,
            r#"{"uid":"11110000000000002222000000000000","nickname":"Ben","#,
            r#""editedAt":1700000000,"pictureLen":0}]"#
        )
    );
    let mut out = [0u8; 8];
    assert_eq!(
        switch_user_picture(handle, ann_lo, ann_hi, out.as_mut_ptr(), out.len() as u32),
        4
    );
    assert_eq!(&out[..4], &picture);
    assert_eq!(
        switch_user_picture(handle, ben_lo, ben_hi, out.as_mut_ptr(), out.len() as u32),
        0
    );

    // A list naming a player not in it is refused and changes nothing.
    stage(ann_lo, ann_hi, "Ann", &[]);
    assert_eq!(switch_users_commit(handle, 9, 9), 4);
    assert_eq!(session(handle).cpu.users().len(), 2);
    assert_eq!(switch_take_profile_edits(handle), 0);
}

#[test]
fn the_sd_card_round_trips_through_the_host_entry_points() {
    let (_host, handle) = new_session();

    // Restores are not reported as changes.
    put(handle, "sdmc:/switch/restored.txt", b"hello");
    assert_eq!(switch_sd_pending_changes(handle), 0);
    assert_eq!(take_changes(handle), "[]");

    // A guest write is: `IFile::Write` at offset 0 of a new file.
    {
        let cpu = &mut session(handle).cpu;
        assert!(cpu.fs.create_file("/switch/cfg.json", 0));
        cpu.fs.write("/switch/cfg.json", 0, br#"{"v":5}"#).unwrap();
        cpu.fs.guest_create_dir("/switch/saves");
        cpu.fs.remove("/switch/restored.txt");
    }
    assert_eq!(switch_sd_pending_changes(handle), 3);
    assert_eq!(
        take_changes(handle),
        r#"[{"path":"/switch/cfg.json","kind":"file","size":7},"#.to_owned()
            + r#"{"path":"/switch/restored.txt","kind":"deleted","size":0},"#
            + r#"{"path":"/switch/saves","kind":"dir","size":0}]"#
    );
    // Draining clears them.
    assert_eq!(switch_sd_pending_changes(handle), 0);
    assert_eq!(take_changes(handle), "[]");

    let path = "/switch/cfg.json";
    assert_eq!(
        switch_sd_file_size(handle, path.as_ptr(), path.len() as u32),
        7
    );
    let mut out = [0u8; 16];
    let n = switch_sd_read_file(
        handle,
        path.as_ptr(),
        path.len() as u32,
        0,
        out.as_mut_ptr(),
        out.len() as u32,
    );
    assert_eq!(n, 7);
    assert_eq!(&out[..7], br#"{"v":5}"#);

    let n = switch_sd_read_file(
        handle,
        path.as_ptr(),
        path.len() as u32,
        4,
        out.as_mut_ptr(),
        out.len() as u32,
    );
    assert_eq!(n, 3);
    assert_eq!(&out[..3], b":5}");

    let dir = "/switch";
    assert_eq!(
        switch_sd_file_size(handle, dir.as_ptr(), dir.len() as u32),
        -1
    );
    let missing = "/switch/nope";
    assert_eq!(
        switch_sd_read_file(
            handle,
            missing.as_ptr(),
            missing.len() as u32,
            0,
            out.as_mut_ptr(),
            out.len() as u32
        ),
        -1
    );
    switch_free_session(handle);
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
    let (_host, handle) = new_session();
    let payload: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let image = build_nsp(&[("main.nca", &payload), ("notes.txt", b"hello")]);
    let size = image.len() as u64;
    set_host_container(image);

    assert_eq!(switch_open_nsp(handle, size), 0);

    let mut buf = vec![0u8; 4096];
    let n = switch_nsp_files_json(handle, buf.as_mut_ptr(), buf.len() as u32);
    let json = String::from_utf8(buf[..n as usize].to_vec()).unwrap();
    assert!(json.contains(r#"{"name":"main.nca","offset":"#), "{json}");
    assert!(json.contains(r#""size":4096}"#), "{json}");
    assert!(json.contains(r#"{"name":"notes.txt""#), "{json}");

    // Reads are relative to the inner file and stop at its end.
    let mut out = vec![0u8; 32];
    let got = switch_read_file(handle, 0, 0x1000 - 8, out.as_mut_ptr(), out.len() as u32);
    assert_eq!(got, 8);
    assert_eq!(&out[..8], &payload[0x1000 - 8..]);

    let got = switch_read_file(handle, 1, 0, out.as_mut_ptr(), out.len() as u32);
    assert_eq!(got, 5);
    assert_eq!(&out[..5], b"hello");

    // Past a file's end is empty; past the table is an error.
    assert_eq!(switch_read_file(handle, 1, 5, out.as_mut_ptr(), 32), 0);
    assert_eq!(switch_read_file(handle, 7, 0, out.as_mut_ptr(), 32), -1);

    // A non-NCA payload fails with a readable error.
    assert_eq!(switch_load_nca_from_nsp(handle, 0), -1);
    let mut err = vec![0u8; 512];
    let n = switch_last_error(handle, err.as_mut_ptr(), err.len() as u32);
    let text = String::from_utf8(err[..n as usize].to_vec()).unwrap();
    assert!(text.contains("bad magic"), "{text}");

    switch_free_session(handle);
}

/// A cartridge image opens through the same entry point as an `.nsp`.
#[test]
fn a_cartridge_image_opens_as_a_container() {
    use switch_core::nsp::testing::partition_fs;
    use switch_core::nsp::PartitionKind;

    let (_host, handle) = new_session();
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

    assert_eq!(switch_open_nsp(handle, size), 0);

    let mut buf = vec![0u8; 4096];
    let n = switch_nsp_files_json(handle, buf.as_mut_ptr(), buf.len() as u32);
    let json = String::from_utf8(buf[..n as usize].to_vec()).unwrap();
    assert!(json.contains(r#"{"name":"program.nca""#), "{json}");
    assert!(json.contains(r#"{"name":"meta.cnmt.nca""#), "{json}");
    assert!(!json.contains("system.nca"), "{json}");

    // Offsets are the image's own.
    let mut out = vec![0u8; 16];
    let got = switch_read_file(handle, 0, 0x700, out.as_mut_ptr(), out.len() as u32);
    assert_eq!(got, 16);
    assert_eq!(out[..], payload[0x700..0x710]);

    switch_free_session(handle);
}

#[test]
fn a_path_json_cannot_carry_raw_is_escaped() {
    let (_host, handle) = new_session();
    session(handle).cpu.fs.create_file(r#"/switch/a"b\c"#, 0);
    let json = take_changes(handle);
    assert_eq!(
        json,
        r#"[{"path":"/switch/a\"b\\c","kind":"file","size":0}]"#
    );
    switch_free_session(handle);
}

#[test]
fn an_error_message_keeps_its_last_character() {
    // The NUL comes out of the buffer, not the message.
    let (_host, handle) = new_session();
    const MSG: &str = "no container is open";
    session(handle).last_error = MSG.to_string();

    let mut buf = [0xAAu8; 64];
    let n = switch_last_error(handle, buf.as_mut_ptr(), buf.len() as u32);
    assert_eq!(n as usize, MSG.len());
    assert_eq!(&buf[..n as usize], MSG.as_bytes());
    assert_eq!(buf[n as usize], 0, "the copy has to stay NUL-terminated");

    // A message that does not fit loses exactly what the buffer cannot hold.
    session(handle).last_error = MSG.to_string();
    let mut small = [0xAAu8; 8];
    let n = switch_last_error(handle, small.as_mut_ptr(), small.len() as u32);
    assert_eq!(n as usize, small.len() - 1);
    assert_eq!(&small[..n as usize], &MSG.as_bytes()[..small.len() - 1]);
    assert_eq!(small[small.len() - 1], 0);

    switch_free_session(handle);
}

/// The translator's and the GPU's reports, in the bytes the page parses.
#[test]
fn the_translator_and_gpu_reports_are_the_json_the_page_reads() {
    use crate::stats::{switch_gpu_report_json, switch_jit_stats_json};
    let (_guard, handle) = new_session();
    let mut buf = vec![0u8; 4096];
    let n = switch_jit_stats_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
    assert_eq!(
        std::str::from_utf8(&buf[..n]).unwrap(),
        concat!(
            r#"{"enabled":true,"blocks":0,"translated":0,"executed":0,"linked":0,"#,
            r#""invalidated":0,"interpreted":0,"interpretedGroups":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"#,
            r#""emitted":0,"enteredEmitted":0,"chained":0}"#
        )
    );
    let n = crate::stats::switch_memory_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
    let memory = std::str::from_utf8(&buf[..n]).unwrap();
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
    let n = switch_gpu_report_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
    assert_eq!(&buf[..n], br#"{"frames":0}"#);
    switch_free_session(handle);
}

/// A title's control data and an unreadable NCA, in the bytes the page parses.
#[test]
fn control_data_and_nca_summaries_are_the_json_the_page_reads() {
    use crate::container::{switch_control_json, switch_parse_nca};
    use switch_core::control::{CaptureSupport, Control, Nacp, Rating, StartupUserAccount, Title};
    let (_guard, handle) = new_session();
    let mut buf = vec![0u8; 4096];
    let n = switch_control_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
    assert_eq!(&buf[..n], b"{}", "nothing cached");

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
    session(handle).control = Some(Control {
        title_id: 0x0100_0000_0000_1000,
        language: "AmericanEnglish",
        name: "JUST DANCE® 2017".into(),
        publisher: "Ubi \\ soft".into(),
        icon: vec![0xFF, 0xD8, 0xFF, 0xE0],
        nacp,
        nacp_raw: Vec::new(),
    });
    let n = switch_control_json(handle, buf.as_mut_ptr(), buf.len() as u32) as usize;
    assert_eq!(
        std::str::from_utf8(&buf[..n]).unwrap(),
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

    let garbage = [0u8; 16];
    let n = switch_parse_nca(
        handle,
        garbage.as_ptr(),
        garbage.len() as u32,
        buf.as_mut_ptr(),
        buf.len() as u32,
    ) as usize;
    assert_eq!(
        std::str::from_utf8(&buf[..n]).unwrap(),
        r#"{"error":"NCA header: expected at least 1024 bytes, got 16"}"#
    );
    switch_free_session(handle);
}
