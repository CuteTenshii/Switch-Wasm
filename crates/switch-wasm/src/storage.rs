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

// save data
//
// A save is `save_id` plus the uid as `user_lo` (first eight bytes) and `user_hi`
// (last eight), little-endian, both zero for a save no user owns. See `SaveKey`.

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
