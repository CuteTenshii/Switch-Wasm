//! The SD card and save data.

use switch_core::cpu::SaveKey;

use crate::session;

// The emulated SD card. Paths are guest paths; `sdmc:` and extra slashes are
// normalized away.

/// Read a UTF-8 path out of guest-supplied wasm memory.
pub(crate) fn sd_path(ptr: *const u8, len: u32) -> String {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Put a file on the SD card. Not reported as a change: this is the host's restore path.
#[no_mangle]
pub extern "C" fn switch_sd_write_file(
    handle: u32,
    path_ptr: *const u8,
    path_len: u32,
    data_ptr: *const u8,
    data_len: u32,
) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
    s.cpu
        .fs
        .write_file(&sd_path(path_ptr, path_len), data.to_vec());
    0
}

/// Create a directory and any missing parents. Not reported as a change.
#[no_mangle]
pub extern "C" fn switch_sd_create_dir(handle: u32, path_ptr: *const u8, path_len: u32) -> i32 {
    session(handle)
        .cpu
        .fs
        .create_dir(&sd_path(path_ptr, path_len));
    0
}

/// Delete a path from the SD card. Returns 1 if something was there, 0 if not.
#[no_mangle]
pub extern "C" fn switch_sd_remove(handle: u32, path_ptr: *const u8, path_len: u32) -> i32 {
    i32::from(session(handle).cpu.fs.remove(&sd_path(path_ptr, path_len)))
}

/// Size of a file on the SD card, or -1 when the path is not a file.
#[no_mangle]
pub extern "C" fn switch_sd_file_size(handle: u32, path_ptr: *const u8, path_len: u32) -> i64 {
    match session(handle).cpu.fs.size(&sd_path(path_ptr, path_len)) {
        Some(size) => size as i64,
        None => -1,
    }
}

/// Copy a file off the SD card into `buf` from `offset`. Returns bytes copied, or -1.
#[no_mangle]
pub extern "C" fn switch_sd_read_file(
    handle: u32,
    path_ptr: *const u8,
    path_len: u32,
    offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, maxlen as usize) };
    match s.cpu.fs.read(&sd_path(path_ptr, path_len), offset, out) {
        Some(n) => n as i64,
        None => -1,
    }
}

/// How many paths the guest has changed and not yet drained.
#[no_mangle]
pub extern "C" fn switch_sd_pending_changes(handle: u32) -> u32 {
    session(handle).cpu.fs.pending_changes() as u32
}

/// Drain the guest's SD card changes as JSON, e.g.
/// `[{"path":"/switch/a.json","kind":"file","size":12}]`. Drains even if the
/// result does not fit; size `buf` from `switch_sd_pending_changes`.
#[no_mangle]
pub extern "C" fn switch_sd_take_changes_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let changes = session(handle).cpu.fs.take_changes();
    write_changes_json(&changes, buf, maxlen)
}

/// Serialize drained [`Change`](switch_core::vfs::Change)s into `buf`.
fn write_changes_json(changes: &[switch_core::vfs::Change], buf: *mut u8, maxlen: u32) -> u32 {
    let mut out = Vec::from("[");
    for (i, change) in changes.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        let kind = match change.kind {
            Some(switch_core::vfs::ENTRY_TYPE_DIR) => "dir",
            Some(_) => "file",
            None => "deleted",
        };
        out.extend_from_slice(b"{\"path\":\"");
        for &byte in change.path.as_bytes() {
            match byte {
                b'"' | b'\\' => {
                    out.push(b'\\');
                    out.push(byte);
                }
                0x00..=0x1F => out.extend_from_slice(format!("\\u{:04x}", byte).as_bytes()),
                _ => out.push(byte),
            }
        }
        out.extend_from_slice(b"\",\"kind\":\"");
        out.extend_from_slice(kind.as_bytes());
        out.extend_from_slice(b"\",\"size\":");
        out.extend_from_slice(change.size.to_string().as_bytes());
        out.push(b'}');
    }
    out.push(b']');
    let n = out.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&out[..n]);
    n as u32
}

// save data
//
// A save is `save_id` plus the uid as `user_lo` (first eight bytes) and `user_hi`
// (last eight), little-endian, both zero for a save no user owns. See `SaveKey`.

/// Every save the session has opened, as JSON:
/// `["8000000000000050","0100000000001000@<32 hex digits of uid>"]`.
#[no_mangle]
pub extern "C" fn switch_save_ids_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let mut keys = session(handle).cpu.save_keys();
    keys.sort_unstable();
    let mut out = Vec::from("[");
    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(format!("\"{key}\"").as_bytes());
    }
    out.push(b']');
    let n = out.len().min(maxlen as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(buf, n) };
    dst.copy_from_slice(&out[..n]);
    n as u32
}

/// How many paths the guest has changed in this save and not yet had drained.
#[no_mangle]
pub extern "C" fn switch_save_pending_changes(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
) -> u32 {
    let key = SaveKey::from_halves(save_id, user_lo, user_hi);
    session(handle).cpu.save_data_mut(key).pending_changes() as u32
}

/// Drain a save's changes, as `switch_sd_take_changes_json` does. Drains even if the
/// result does not fit.
#[no_mangle]
pub extern "C" fn switch_save_take_changes_json(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    buf: *mut u8,
    maxlen: u32,
) -> u32 {
    let key = SaveKey::from_halves(save_id, user_lo, user_hi);
    let changes = session(handle).cpu.save_data_mut(key).take_changes();
    write_changes_json(&changes, buf, maxlen)
}

/// Put a file into a save, creating the save. Not reported as a change.
#[no_mangle]
pub extern "C" fn switch_save_write_file(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
    data_ptr: *const u8,
    data_len: u32,
) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
    let path = sd_path(path_ptr, path_len);
    s.cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .write_file(&path, data.to_vec());
    0
}

/// Create a directory in a save and any missing parents. Not reported as a change.
#[no_mangle]
pub extern "C" fn switch_save_create_dir(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i32 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    s.cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .create_dir(&path);
    0
}

/// Remove one entry from a save; a directory's contents are removed separately.
#[no_mangle]
pub extern "C" fn switch_save_remove(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i32 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    i32::from(
        s.cpu
            .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
            .remove(&path),
    )
}

/// Size of a file in a save, or -1 when the path is not one.
#[no_mangle]
pub extern "C" fn switch_save_file_size(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
) -> i64 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    match s
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .size(&path)
    {
        Some(size) => size as i64,
        None => -1,
    }
}

/// Copy a file out of a save into `buf` from `offset`. Returns bytes copied, or -1.
#[no_mangle]
pub extern "C" fn switch_save_read_file(
    handle: u32,
    save_id: u64,
    user_lo: u64,
    user_hi: u64,
    path_ptr: *const u8,
    path_len: u32,
    offset: u64,
    buf: *mut u8,
    maxlen: u32,
) -> i64 {
    let s = session(handle);
    let path = sd_path(path_ptr, path_len);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, maxlen as usize) };
    match s
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .read(&path, offset, out)
    {
        Some(n) => n as i64,
        None => -1,
    }
}

/// Create a save in a fresh session so the host can restore it. Returns 0.
#[no_mangle]
pub extern "C" fn switch_save_create(handle: u32, save_id: u64, user_lo: u64, user_hi: u64) -> i32 {
    let key = SaveKey::from_halves(save_id, user_lo, user_hi);
    session(handle).cpu.save_data_mut(key);
    0
}
