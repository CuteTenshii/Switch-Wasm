//! The SD card and save data, as the page changes them while a session runs.

use switch_core::cpu::SaveKey;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

// The emulated SD card. Paths are guest paths; `sdmc:` and extra slashes are
// normalized away. None of these are reported as changes: this is the host's restore path.

/// Put a file on the SD card.
#[wasm_bindgen]
pub fn switch_sd_write_file(path: &str, data: &[u8]) {
    session().cpu.fs.write_file(path, data.to_vec());
}

/// Create a directory and any missing parents.
#[wasm_bindgen]
pub fn switch_sd_create_dir(path: &str) {
    session().cpu.fs.create_dir(path);
}

/// Delete a path from the SD card.
#[wasm_bindgen]
pub fn switch_sd_remove(path: &str) {
    session().cpu.fs.remove(path);
}

// A save is `save_id` plus the uid as `user_lo` (first eight bytes) and `user_hi`
// (last eight), little-endian, both zero for a save no user owns. See `SaveKey`.

/// Put a file into a save, creating the save.
#[wasm_bindgen]
pub fn switch_save_write_file(save_id: u64, user_lo: u64, user_hi: u64, path: &str, data: &[u8]) {
    session()
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .write_file(path, data.to_vec());
}

/// Create a directory in a save and any missing parents.
#[wasm_bindgen]
pub fn switch_save_create_dir(save_id: u64, user_lo: u64, user_hi: u64, path: &str) {
    session()
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .create_dir(path);
}

/// Remove one entry from a save; a directory's contents are removed separately.
#[wasm_bindgen]
pub fn switch_save_remove(save_id: u64, user_lo: u64, user_hi: u64, path: &str) {
    session()
        .cpu
        .save_data_mut(SaveKey::from_halves(save_id, user_lo, user_hi))
        .remove(path);
}
