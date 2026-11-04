//! The SD card and saves in IndexedDB: restored into the session when it starts, and
//! written as the guest changes them, without a trip through the page.

use std::cell::RefCell;
use std::collections::BTreeMap;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use switch_core::cpu::SaveKey;
use switch_core::trace::Level;
use switch_core::vfs::{Change, Vfs, ENTRY_TYPE_DIR};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use super::{
    self as idb, NAND_DB, NAND_SAVES, NAND_STORES, NAND_VERSION, SD_DB, SD_STORE, SD_VERSION,
};
use crate::session_if_live;

/// Stored entries by key; `None` deletes the key.
type Backlog = BTreeMap<String, Option<JsValue>>;

#[derive(Default)]
struct Flush {
    /// Taken from the core but not yet stored; IndexedDB refusals wait for the next flush.
    sd: Backlog,
    saves: Backlog,
    running: bool,
    /// Flushes that arrived while one ran, resolved when it has stored what they took.
    waiters: Vec<Function>,
}

thread_local! {
    static FLUSH: RefCell<Flush> = RefCell::new(Flush::default());
}

/// Paths shown in a store line before the rest is summarized.
const STORED_NAMED_MAX: usize = 8;

/// Read at a time when copying a file out of the core.
const CHUNK: usize = 1 << 20;

fn log(handle: u32, level: Level, line: &str) {
    if let Some(s) = session_if_live(handle) {
        s.cpu.log(level, line);
    }
}

/// Load the stored card and saves through the host's restore path, which is not a guest change.
#[wasm_bindgen]
pub async fn switch_restore_storage(handle: u32) -> u32 {
    match restore_sd(handle).await {
        Ok(0) => {}
        Ok(n) => log(
            handle,
            Level::Debug,
            &format!("SD card: restored {n} entries"),
        ),
        Err(err) => log(
            handle,
            Level::Error,
            &format!("SD card: could not be read ({err:?})"),
        ),
    }
    match restore_saves(handle).await {
        Ok(0) => {}
        Ok(n) => log(
            handle,
            Level::Debug,
            &format!("Saves: restored {n} entries"),
        ),
        Err(err) => log(
            handle,
            Level::Error,
            &format!("Saves: could not be read ({err:?})"),
        ),
    }
    super::controls::restore(handle).await;
    0
}

async fn stored(
    db: &str,
    version: u32,
    stores: &[&str],
    store: &str,
) -> Result<Vec<Stored>, JsValue> {
    let db = idb::open(db, version, stores).await?;
    let entries = idb::get_all(&db, store).await;
    db.close();
    let mut entries: Vec<Stored> = entries?
        .into_iter()
        .map(|(key, value)| Stored::new(key, &value))
        .collect();
    // Directories first, so empty ones survive.
    entries.sort_by_key(|entry| entry.data.is_some());
    Ok(entries)
}

/// One stored `{ kind, data? }`; `data` is `None` for a directory.
struct Stored {
    key: String,
    data: Option<Vec<u8>>,
}

impl Stored {
    fn new(key: String, value: &JsValue) -> Stored {
        let is_dir = Reflect::get(value, &"kind".into())
            .ok()
            .and_then(|k| k.as_string())
            == Some("dir".into());
        let data = (!is_dir).then(|| {
            Reflect::get(value, &"data".into())
                .ok()
                .and_then(|data| data.dyn_into::<Uint8Array>().ok())
                .map_or_else(Vec::new, |data| data.to_vec())
        });
        Stored { key, data }
    }
}

async fn restore_sd(handle: u32) -> Result<usize, JsValue> {
    let entries = stored(SD_DB, SD_VERSION, &[SD_STORE], SD_STORE).await?;
    let Some(s) = session_if_live(handle) else {
        return Ok(0);
    };
    for Stored { key, data } in &entries {
        match data {
            None => s.cpu.fs.create_dir(key),
            Some(data) => s.cpu.fs.write_file(key, data.clone()),
        }
    }
    Ok(entries.len())
}

async fn restore_saves(handle: u32) -> Result<usize, JsValue> {
    let entries = stored(NAND_DB, NAND_VERSION, &NAND_STORES, NAND_SAVES).await?;
    let Some(s) = session_if_live(handle) else {
        return Ok(0);
    };
    let mut restored = 0;
    for Stored { key, data } in entries {
        let Some((save, path)) = key.find('/').map(|cut| key.split_at(cut)) else {
            continue;
        };
        let Some(save) = parse_save_key(save) else {
            continue;
        };
        let vfs = s.cpu.save_data_mut(save);
        match data {
            None => vfs.create_dir(path),
            Some(data) => vfs.write_file(path, data),
        }
        restored += 1;
    }
    Ok(restored)
}

/// A save's [`Display`](std::fmt::Display) form: `<16 hex digits>` or `<id>@<32 hex digits of uid>`.
fn parse_save_key(text: &str) -> Option<SaveKey> {
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

/// Take what the guest changed and store it. Resolves once everything taken is stored, or
/// has failed to be and is kept for the next flush.
#[wasm_bindgen]
pub async fn switch_flush_storage(handle: u32) -> u32 {
    take_changes(handle);
    if FLUSH.with_borrow(|f| f.running) {
        let waiting = Promise::new(&mut |resolve: Function, _| {
            FLUSH.with_borrow_mut(|f| f.waiters.push(resolve));
        });
        JsFuture::from(waiting).await.ok();
        return 0;
    }
    FLUSH.with_borrow_mut(|f| f.running = true);
    while store(handle).await {}
    FLUSH.with_borrow_mut(|f| {
        f.running = false;
        for waiter in f.waiters.drain(..) {
            waiter.call0(&JsValue::NULL).ok();
        }
    });
    0
}

fn take_changes(handle: u32) {
    let Some(s) = session_if_live(handle) else {
        return;
    };
    let sd: Vec<_> = s
        .cpu
        .fs
        .take_changes()
        .into_iter()
        .map(|change| (change.path.clone(), entry(&s.cpu.fs, &change)))
        .collect();
    let mut saves = Vec::new();
    for key in s.cpu.save_keys() {
        let vfs = s.cpu.save_data_mut(key);
        for change in vfs.take_changes() {
            saves.push((format!("{key}{}", change.path), entry(vfs, &change)));
        }
    }
    FLUSH.with_borrow_mut(|f| {
        f.sd.extend(sd);
        f.saves.extend(saves);
    });
}

/// The stored form of a changed path: `{ kind: 'dir' }`, `{ kind: 'file', data }`, or
/// `None` once deleted.
fn entry(vfs: &Vfs, change: &Change) -> Option<JsValue> {
    let kind = change.kind?;
    let object = Object::new();
    let set = |name: &str, value: &JsValue| Reflect::set(&object, &name.into(), value).ok();
    if kind == ENTRY_TYPE_DIR {
        set("kind", &"dir".into());
        return Some(object.into());
    }
    let data = Uint8Array::new_with_length(change.size as u32);
    let mut chunk = vec![0u8; CHUNK.min(change.size as usize)];
    let mut at = 0u64;
    while at < change.size {
        let Some(n) = vfs.read(&change.path, at, &mut chunk).filter(|&n| n > 0) else {
            break;
        };
        data.subarray(at as u32, at as u32 + n as u32)
            .copy_from(&chunk[..n]);
        at += n as u64;
    }
    set("kind", &"file".into());
    set("data", &data);
    Some(object.into())
}

/// Store the backlogs. Returns whether anything was written, so the caller can go again for
/// what was taken meanwhile.
async fn store(handle: u32) -> bool {
    let mut wrote = false;
    for (what, db, version, stores, store, is_sd) in [
        (
            "SD card",
            SD_DB,
            SD_VERSION,
            &[SD_STORE][..],
            SD_STORE,
            true,
        ),
        (
            "saves",
            NAND_DB,
            NAND_VERSION,
            &NAND_STORES[..],
            NAND_SAVES,
            false,
        ),
    ] {
        let changes: Vec<_> = FLUSH.with_borrow_mut(|f| {
            std::mem::take(if is_sd { &mut f.sd } else { &mut f.saves })
                .into_iter()
                .collect()
        });
        if changes.is_empty() {
            continue;
        }
        match write(db, version, stores, store, &changes).await {
            Ok(()) => {
                wrote = true;
                log_stored(handle, what, &changes);
            }
            Err(err) => {
                // Put them back under anything newer that arrived meanwhile.
                FLUSH.with_borrow_mut(|f| {
                    let backlog = if is_sd { &mut f.sd } else { &mut f.saves };
                    for (key, value) in changes {
                        backlog.entry(key).or_insert(value);
                    }
                });
                log(
                    handle,
                    Level::Error,
                    &format!(
                        "{what}: could not be written ({err:?}) - retrying on the next flush."
                    ),
                );
            }
        }
    }
    wrote
}

async fn write(
    db: &str,
    version: u32,
    stores: &[&str],
    store: &str,
    changes: &[(String, Option<JsValue>)],
) -> Result<(), JsValue> {
    let db = idb::open(db, version, stores).await?;
    let outcome = idb::apply(&db, store, changes).await;
    db.close();
    outcome
}

fn log_stored(handle: u32, what: &str, changes: &[(String, Option<JsValue>)]) {
    let mut bytes = 0.0;
    let mut named = Vec::new();
    for (path, entry) in changes {
        if let Some(data) = entry
            .as_ref()
            .and_then(|e| Reflect::get(e, &"data".into()).ok())
            .and_then(|data| data.dyn_into::<Uint8Array>().ok())
        {
            bytes += f64::from(data.byte_length());
        }
        if named.len() < STORED_NAMED_MAX {
            named.push(if entry.is_some() {
                path.clone()
            } else {
                format!("{path} (deleted)")
            });
        }
    }
    let more = match changes.len().saturating_sub(named.len()) {
        0 => String::new(),
        n => format!(", and {n} more"),
    };
    log(
        handle,
        Level::Debug,
        &format!(
            "[io] {what}: stored {} changes ({:.1} KiB) in IndexedDB: {}{more}",
            changes.len(),
            bytes / 1024.0,
            named.join(", "),
        ),
    );
}
