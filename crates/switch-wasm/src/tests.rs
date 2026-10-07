use super::*;
use switch_core::cpu::SaveKey;
use switch_core::trace::Level;

use crate::boot::switch_load_nca_from_nsp;
use crate::container::{switch_nsp_files_json, switch_open_nsp, switch_read_file};
use crate::debug::{
    switch_crash_report_json, switch_set_trace_channels, switch_start_created_threads,
    switch_thread_dump, switch_unimplemented_json, switch_wake_blocked,
};
use crate::stats::switch_activity_json;
use crate::storage::{
    switch_save_create, switch_save_file_size, switch_save_ids_json, switch_save_pending_changes,
    switch_save_read_file, switch_save_take_changes_json, switch_save_write_file,
    switch_sd_file_size, switch_sd_pending_changes, switch_sd_read_file,
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

fn json_from(fill: impl Fn(*mut u8, u32) -> u32) -> String {
    let cap = 1024 * 1024;
    let mut buf = vec![0u8; cap];
    let n = fill(buf.as_mut_ptr(), cap as u32);
    String::from_utf8(buf[..n as usize].to_vec()).unwrap()
}

/// The raw JSON text of a `"key":` field.
fn field<'a>(json: &'a str, key: &str) -> &'a str {
    let at = json
        .find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("no {key} in {json:.400}"));
    let rest = &json[at + key.len() + 3..];
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in rest.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
                if depth == 0 {
                    return &rest[..i + 1];
                }
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    return &rest[..i + 1];
                }
            }
            ',' | '}' | ']' if depth == 0 => return &rest[..i],
            _ => {}
        }
    }
    rest
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

#[test]
fn the_activity_report_names_each_file_and_hands_it_over_once() {
    let (_host, handle) = new_session();
    let json = json_from(|buf, cap| switch_activity_json(handle, buf, cap));
    assert_eq!(field(&json, "draws"), "0");
    assert_eq!(field(&json, "files"), "[]");
    assert_eq!(field(&json, "gpu"), "[]");
    assert_eq!(field(&json, "threadLog"), "[]");
    assert!(field(&json, "threads").contains("\"index\":0"), "{json}");
    assert!(field(&json, "threads").contains("\"idleMs\":"), "{json}");
    assert_eq!(field(&json, "journal"), "[]");
    for list in ["refusals", "gaps", "nvErrors"] {
        assert_eq!(field(&json, list), "[]", "{list}");
    }
    assert_eq!(field(&json, "input"), r#"{"supported":0,"presented":1}"#);
    assert_eq!(field(&json, "problemsDropped"), "0");

    let activity = &mut session(handle).cpu.fs_activity;
    activity.read("romfs", 0x100);
    activity.read("romfs", 0x20);
    activity.wrote("sdmc:/cfg.json", 7);
    activity.record("fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)".to_owned());
    let json = json_from(|buf, cap| switch_activity_json(handle, buf, cap));
    assert_eq!(
        field(&json, "files"),
        r#"[{"name":"romfs","reads":2,"readBytes":288,"writes":0,"writeBytes":0},{"name":"sdmc:/cfg.json","reads":0,"readBytes":0,"writes":1,"writeBytes":7}]"#
    );
    assert_eq!(
        field(&json, "journal"),
        r#"["fs OpenFile \"/b\" on sdmc -> 2002-0001 (0x202)"]"#
    );
    let again = json_from(|buf, cap| switch_activity_json(handle, buf, cap));
    assert_eq!(field(&again, "files"), "[]", "taken, not read");
    assert_eq!(field(&again, "journal"), "[]", "taken, not read");

    // An answer too small stays parseable and counts what it dropped.
    let activity = &mut session(handle).cpu.fs_activity;
    activity.read(&"x".repeat(1000), 1);
    activity.record("y".repeat(1000));
    let mut buf = vec![0u8; 700];
    let n = switch_activity_json(handle, buf.as_mut_ptr(), buf.len() as u32);
    let small = String::from_utf8(buf[..n as usize].to_vec()).unwrap();
    assert!(small.ends_with('}'), "{small}");
    assert_eq!(field(&small, "files"), "[]");
    assert_eq!(field(&small, "filesDropped"), "1");
    assert_eq!(field(&small, "dropped"), "1");
}

#[test]
fn a_crash_report_names_the_build_even_with_no_session_behind_it() {
    // A report needs no live session.
    let _host = HOST.lock().unwrap_or_else(|e| e.into_inner());
    let json = json_from(|buf, cap| switch_crash_report_json(u32::MAX, buf, cap));
    assert_eq!(field(&json, "session"), "null");
    assert!(field(&json, "version").len() > 2, "{json}");
    assert!(json.starts_with('{') && json.ends_with('}'), "{json}");
}

#[test]
fn a_crash_report_carries_the_run_it_is_about() {
    let (_host, handle) = new_session();
    let cpu = &mut session(handle).cpu;
    cpu.diagnostic(Level::Error, "[test] the thing that went wrong");
    session(handle).last_error = "a fault worth reporting".to_string();

    let json = json_from(|buf, cap| switch_crash_report_json(handle, buf, cap));
    assert_eq!(field(&json, "lastError"), "\"a fault worth reporting\"");
    for key in [
        "version",
        "panicked",
        "title",
        "cpu",
        "jit",
        "gpu",
        "backtrace",
        "registers",
        "threads",
        "unimplemented",
        "stubbed",
        "trace",
    ] {
        assert!(
            !field(&json, key).is_empty(),
            "{key} is empty in {json:.400}"
        );
    }
    assert!(
        field(&json, "trace").contains("the thing that went wrong"),
        "the trace has to carry what was said: {json:.400}"
    );
    assert!(field(&json, "backtrace").starts_with('['));
    assert!(field(&json, "registers").contains("pc="));
}

#[test]
fn the_threads_a_guest_parked_can_be_released_from_the_browser() {
    let (_host, handle) = new_session();
    assert!(!json_from(|buf, cap| { switch_thread_dump(handle, buf, cap) }).is_empty());
    assert_eq!(switch_wake_blocked(handle), 0);
    assert_eq!(switch_start_created_threads(handle), 0);

    // Two threads, one started; main then waits on a zero word, yielding to it.
    let cpu = &mut session(handle).cpu;
    let mut create = || {
        guest_svc(cpu, 0x08, &[0, 0x0800_1000, 0, 0x2880_0000, 44, 0]);
        assert_eq!(cpu.reg(0), 0, "CreateThread failed");
        cpu.reg(1)
    };
    let started = create();
    create();
    guest_svc(cpu, 0x09, &[started]);
    guest_svc(cpu, 0x34, &[0x0800_2000, 2, 0, u64::MAX]);
    assert_eq!(switch_wake_blocked(handle), 1);
    assert_eq!(switch_start_created_threads(handle), 1);
    assert_eq!(switch_start_created_threads(handle), 0);
}

/// Run one `svc #imm` with `args` in X0 upward.
fn guest_svc(cpu: &mut Cpu, imm: u32, args: &[u64]) {
    let pc = 0x0800_0000 + imm * 4;
    cpu.mem.write_u32(pc, 0xD400_0001 | imm << 5).unwrap();
    for (i, &arg) in args.iter().enumerate() {
        cpu.set_reg(i as u8, arg);
    }
    cpu.set_pc(pc);
    cpu.step().unwrap();
}

#[test]
fn what_a_title_asked_for_and_did_not_get_is_a_list_not_a_scrollback() {
    let (_host, handle) = new_session();
    let json = json_from(|buf, cap| switch_unimplemented_json(handle, buf, cap));
    assert_eq!(field(&json, "unimplemented"), "[]");
    assert_eq!(field(&json, "stubbed"), "[]");

    // A request (type 4) for command 5 on an unopened handle.
    let cpu = &mut session(handle).cpu;
    let tls = cpu.tls_base();
    for (i, word) in [4, 8, 0, 0, 0x4943_4653, 0, 5, 0].into_iter().enumerate() {
        cpu.mem.write_u32(tls + i as u32 * 4, word).unwrap();
    }
    guest_svc(cpu, 0x21, &[0x1234]);
    let json = json_from(|buf, cap| switch_unimplemented_json(handle, buf, cap));
    assert_eq!(
        field(&json, "unimplemented"),
        r#"[{"iface":"<untracked session>","cmd":5}]"#
    );
    assert_eq!(field(&json, "stubbed"), "[]");
}

/// Reset clears module-level panic and trace state.
#[test]
fn a_new_session_inherits_nothing_from_the_one_before_it() {
    let (_host, first) = new_session();

    PANICKED.store(true, Ordering::Relaxed);
    // SAFETY: single-threaded under the `HOST` lock.
    let planted = b"PANIC: the old session died here";
    let guard = unsafe { &mut *PANIC_MSG.get() };
    guard[..planted.len()].copy_from_slice(planted);
    guard[planted.len()] = 0;
    switch_core::trace::emit("[test] the old session traced this");

    switch_free_session(first);
    let second = switch_new();

    let json = json_from(|buf, cap| switch_crash_report_json(second, buf, cap));
    assert_eq!(
        field(&json, "panicked"),
        "false",
        "a reset console has not panicked"
    );
    assert!(
        !field(&json, "trace").contains("the old session traced this"),
        "the new session opened with the old one's trace: {json:.400}"
    );

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

    // Opening a save is enough to list it; the shared save is a separate one.
    switch_save_create(handle, SAVE, 0, 0);
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
fn a_non_ascii_title_name_survives_the_json() {
    // A `\uXXXX` escape names a code point, so multi-byte characters go out raw.
    let mut out = Vec::new();
    json_escape("JUST DANCE® 2017 - 日本語", &mut out);
    assert_eq!(String::from_utf8(out).unwrap(), "JUST DANCE® 2017 - 日本語");

    let mut out = Vec::new();
    json_escape("a\"b\\c\nd\u{7}e", &mut out);
    assert_eq!(String::from_utf8(out).unwrap(), r#"a\"b\\c\nd\u0007e"#);
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
