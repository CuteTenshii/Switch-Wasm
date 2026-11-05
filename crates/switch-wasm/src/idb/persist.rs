//! The SD card and saves in IndexedDB: restored into the session when it starts, and
//! written as the guest changes them, without a trip through the page.

use std::cell::RefCell;
use std::collections::BTreeMap;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use switch_core::trace::Level;
use switch_core::vfs::{Change, Vfs, ENTRY_TYPE_DIR};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use super::{
    self as idb, NAND_DB, NAND_SAVES, NAND_STORES, NAND_VERSION, SD_DB, SD_STORE, SD_VERSION,
};
use crate::post::post;
use crate::session_if_live;
use crate::storage::parse_save_key;

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

/// Least time between two housekeeping rounds.
const TICK_INTERVAL_MS: f64 = 100.0;

static TICK_AT: crate::SyncCell<f64> = crate::SyncCell::new(f64::NEG_INFINITY);
static TICK_RUNNING: crate::SyncCell<bool> = crate::SyncCell::new(false);

/// Store what the guest changed since the last round, and tell the page when it edited a
/// profile. Called after every slice; acts at most every [`TICK_INTERVAL_MS`].
pub(crate) fn tick() {
    let now = js_sys::Date::now();
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
    let (at, running) = unsafe { (&mut *TICK_AT.get(), &mut *TICK_RUNNING.get()) };
    if *running || now - *at < TICK_INTERVAL_MS {
        return;
    }
    *at = now;
    *running = true;
    wasm_bindgen_futures::spawn_local(async {
        switch_flush_storage().await;
        if super::users::switch_store_profile_edits().await {
            post("profiles_edited", &[]);
        }
        unsafe { *TICK_RUNNING.get() = false };
    });
}

/// Paths shown in a store line before the rest is summarized.
const STORED_NAMED_MAX: usize = 8;

/// Read at a time when copying a file out of the core.
const CHUNK: usize = 1 << 20;

fn log(level: Level, line: &str) {
    if let Some(s) = session_if_live() {
        s.cpu.log(level, line);
    }
}

/// Load the stored card and saves through the host's restore path, which is not a guest change.
#[wasm_bindgen]
pub async fn switch_restore_storage() -> u32 {
    match restore_sd().await {
        Ok(0) => {}
        Ok(n) => log(Level::Debug, &format!("SD card: restored {n} entries")),
        Err(err) => log(
            Level::Error,
            &format!("SD card: could not be read ({err:?})"),
        ),
    }
    match restore_saves().await {
        Ok(0) => {}
        Ok(n) => log(Level::Debug, &format!("Saves: restored {n} entries")),
        Err(err) => log(Level::Error, &format!("Saves: could not be read ({err:?})")),
    }
    super::controls::restore().await;
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

async fn restore_sd() -> Result<usize, JsValue> {
    let entries = stored(SD_DB, SD_VERSION, &[SD_STORE], SD_STORE).await?;
    let Some(s) = session_if_live() else {
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

async fn restore_saves() -> Result<usize, JsValue> {
    let entries = stored(NAND_DB, NAND_VERSION, &NAND_STORES, NAND_SAVES).await?;
    let Some(s) = session_if_live() else {
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

/// Take what the guest changed and store it. Resolves once everything taken is stored, or
/// has failed to be and is kept for the next flush.
#[wasm_bindgen]
pub async fn switch_flush_storage() -> u32 {
    take_changes();
    if FLUSH.with_borrow(|f| f.running) {
        let waiting = Promise::new(&mut |resolve: Function, _| {
            FLUSH.with_borrow_mut(|f| f.waiters.push(resolve));
        });
        JsFuture::from(waiting).await.ok();
        return 0;
    }
    FLUSH.with_borrow_mut(|f| f.running = true);
    while store().await {}
    FLUSH.with_borrow_mut(|f| {
        f.running = false;
        for waiter in f.waiters.drain(..) {
            waiter.call0(&JsValue::NULL).ok();
        }
    });
    0
}

fn take_changes() {
    let Some(s) = session_if_live() else {
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
async fn store() -> bool {
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
                log_stored(what, &changes);
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

fn log_stored(what: &str, changes: &[(String, Option<JsValue>)]) {
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
        Level::Debug,
        &format!(
            "[io] {what}: stored {} changes ({:.1} KiB) in IndexedDB: {}{more}",
            changes.len(),
            bytes / 1024.0,
            named.join(", "),
        ),
    );
}
