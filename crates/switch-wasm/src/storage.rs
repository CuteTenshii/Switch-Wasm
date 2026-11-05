//! The SD card and save data, as the page changes them while a session runs.

use switch_core::cpu::SaveKey;
use switch_core::vfs::Vfs;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{session, Session};

/// A save's [`Display`](std::fmt::Display) form: `<16 hex digits>` or `<id>@<32 hex digits of uid>`.
pub(crate) fn parse_save_key(text: &str) -> Option<SaveKey> {
    let (id, user) = text.split_once('@').unwrap_or((text, ""));
    let mut key = SaveKey::shared(u64::from_str_radix(id, 16).ok()?);
    if !user.is_empty() {
        if user.len() != 32 {
            return None;
        }
        for (byte, pair) in key.user.iter_mut().zip(user.as_bytes().chunks(2)) {
            *byte = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
        }
    }
    Some(key)
}

/// The files of a volume: the SD card (`sd`), or a save by its id.
fn volume<'a>(s: &'a mut Session, volume: &str) -> Result<&'a mut Vfs, String> {
    if volume == "sd" {
        return Ok(&mut s.cpu.fs);
    }
    match parse_save_key(volume) {
        Some(key) => Ok(s.cpu.save_data_mut(key)),
        None => Err(format!("\"{volume}\" names no storage")),
    }
}

// Paths are guest paths; `sdmc:` and extra slashes are normalized away. None of these are
// reported as changes: this is the host's restore path.

/// Put a file on a volume, creating its directories and, for a save, the save.
#[wasm_bindgen]
pub fn switch_fs_write(storage: &str, path: &str, data: &[u8]) -> Result<(), String> {
    volume(session(), storage)?.write_file(path, data.to_vec());
    Ok(())
}

/// Create a directory and any missing parents.
#[wasm_bindgen]
pub fn switch_fs_create_dir(storage: &str, path: &str) -> Result<(), String> {
    volume(session(), storage)?.create_dir(path);
    Ok(())
}

/// Remove one entry; a directory's contents are removed separately.
#[wasm_bindgen]
pub fn switch_fs_remove(storage: &str, path: &str) -> Result<(), String> {
    volume(session(), storage)?.remove(path);
    Ok(())
}
