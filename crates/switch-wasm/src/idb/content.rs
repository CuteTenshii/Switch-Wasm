//! NAND content: NCA files in OPFS, or in IndexedDB's `content` store where OPFS is missing.
//! The `titles` index stays in IndexedDB and is written after the content, so it never names a
//! file that is not whole. Content older builds kept in IndexedDB is moved over in the background.

use std::cell::RefCell;
use std::collections::HashSet;
use std::future::Future;

use js_sys::{Array, ArrayBuffer, Promise, Reflect};
use switch_core::trace::Level;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, spawn_local, JsFuture};
use web_sys::{Blob, FileSystemDirectoryHandle, IdbDatabase};

use super::{self as idb, NAND_CONTENT, NAND_DB, NAND_STORES, NAND_TITLES, NAND_VERSION};
use crate::opfs::{self, PART};
use crate::session_if_live;

/// The OPFS directory holding NAND content, a file per NCA.
pub(crate) const NAND_OPFS_DIR: &str = "nand";

thread_local! {
    /// Settles with the content directory, or null, once the startup sweep is done.
    static READY: RefCell<Option<Promise>> = const { RefCell::new(None) };
    /// The last queued write or removal, settled either way.
    static TAIL: RefCell<Option<Promise>> = const { RefCell::new(None) };
    /// Names handed out as Files this worker's life. A File stops reading once its OPFS file is
    /// removed or rewritten, and the session keeps its host files, so these stay until the next start.
    static LENT: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

fn log(level: Level, line: &str) {
    if let Some(s) = session_if_live() {
        s.cpu.log(level, line);
    }
}

async fn open() -> Result<IdbDatabase, JsValue> {
    idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await
}

fn is_lent(name: &str) -> bool {
    LENT.with(|lent| lent.borrow().contains(name))
}

/// The content directory once leftovers are cleared, or `None` when content stays in IndexedDB.
async fn ready() -> Option<FileSystemDirectoryHandle> {
    let promise = READY.with(|ready| {
        ready
            .borrow_mut()
            .get_or_insert_with(|| {
                future_to_promise(async { Ok(start().await.map_or(JsValue::NULL, JsValue::from)) })
            })
            .clone()
    });
    JsFuture::from(promise).await.ok()?.dyn_into().ok()
}

async fn start() -> Option<FileSystemDirectoryHandle> {
    let Some(dir) = opfs::dir(NAND_OPFS_DIR, true).await else {
        log(
            Level::Debug,
            "[nand] content is kept in IndexedDB: there is no OPFS to write to",
        );
        return None;
    };
    if let Err(err) = sweep(&dir).await {
        log(
            Level::Warn,
            &format!("[nand] leftover OPFS files were not cleared: {err:?}"),
        );
    }
    let moving = dir.clone();
    spawn_local(async move { migrate(&moving).await });
    Some(dir)
}

/// Run `job` after every earlier write or removal has settled, so a `.part` file only ever
/// belongs to the write in progress.
async fn serial<F, Fut>(job: F) -> Result<JsValue, JsValue>
where
    F: FnOnce(Option<FileSystemDirectoryHandle>) -> Fut + 'static,
    Fut: Future<Output = Result<JsValue, JsValue>> + 'static,
{
    let previous = TAIL.with(|tail| tail.borrow().clone());
    let run = future_to_promise(async move {
        if let Some(previous) = previous {
            JsFuture::from(previous).await.ok();
        }
        job(ready().await).await
    });
    let waited = run.clone();
    let settled = future_to_promise(async move {
        JsFuture::from(waited).await.ok();
        Ok(JsValue::UNDEFINED)
    });
    TAIL.with(|tail| *tail.borrow_mut() = Some(settled));
    JsFuture::from(run).await
}

async fn indexed_names(db: &IdbDatabase) -> Result<HashSet<String>, JsValue> {
    Ok(idb::get_all(db, NAND_TITLES)
        .await?
        .into_iter()
        .filter_map(|(_, entry)| Reflect::get(&entry, &"name".into()).ok()?.as_string())
        .collect())
}

/// Interrupted writes, and content removed while a session was reading it.
async fn sweep(dir: &FileSystemDirectoryHandle) -> Result<(), JsValue> {
    let db = open().await?;
    let named = indexed_names(&db).await;
    db.close();
    let named = named?;
    for (name, _) in opfs::files(dir).await? {
        if name.ends_with(PART) || !named.contains(&name) {
            opfs::remove(dir, &name).await?;
        }
    }
    Ok(())
}

/// Remove the OPFS files no index entry names, except the ones a session may still read.
async fn release(
    dir: &FileSystemDirectoryHandle,
    db: &IdbDatabase,
    names: &[String],
) -> Result<(), JsValue> {
    let named = indexed_names(db).await?;
    let mut kept = 0;
    for name in names.iter().filter(|name| !named.contains(*name)) {
        if is_lent(name) {
            kept += 1;
        } else {
            opfs::remove(dir, name).await?;
        }
    }
    if kept > 0 {
        log(
            Level::Debug,
            &format!("[nand] {kept} removed file(s) stay on disk until the next start: this session reads them"),
        );
    }
    Ok(())
}

/// Content older builds kept in IndexedDB, as a `Blob`; an `ArrayBuffer` is rewritten as one
/// when `rewrite`, as it would otherwise be copied again every session.
async fn legacy(db: &IdbDatabase, name: &str, rewrite: bool) -> Result<Option<Blob>, JsValue> {
    let stored = idb::get(db, NAND_CONTENT, name).await?;
    if let Some(blob) = stored.dyn_ref::<Blob>() {
        return Ok(Some(blob.clone()));
    }
    if !stored.is_instance_of::<ArrayBuffer>() {
        return Ok(None);
    }
    let blob = Blob::new_with_buffer_source_sequence(&Array::of1(&stored))?;
    if rewrite {
        // It is readable either way; a failure only costs the next session the same copy.
        idb::write(
            db,
            &[idb::Write::Put(NAND_CONTENT, name, blob.clone().into())],
        )
        .await
        .ok();
    }
    Ok(Some(blob))
}

/// Move what older builds kept in IndexedDB, one file per queued job so installs are not held up.
async fn migrate(dir: &FileSystemDirectoryHandle) {
    let names = match open().await {
        Ok(db) => {
            let names = idb::keys(&db, NAND_CONTENT).await;
            db.close();
            names
        }
        Err(err) => Err(err),
    };
    let names = match names {
        Ok(names) if names.is_empty() => return,
        Ok(names) => names,
        Err(err) => {
            log(
                Level::Warn,
                &format!(
                    "[nand] IndexedDB content could not be listed for moving to OPFS: {err:?}"
                ),
            );
            return;
        }
    };
    log(
        Level::Debug,
        &format!(
            "[nand] moving {} file(s) from IndexedDB to OPFS",
            names.len()
        ),
    );
    for (moved, name) in names.iter().enumerate() {
        let dir = dir.clone();
        let key = name.clone();
        let step = serial(move |_| async move {
            let db = open().await?;
            let outcome = migrate_one(&dir, &db, &key).await;
            db.close();
            outcome.map(|()| JsValue::UNDEFINED)
        })
        .await;
        if let Err(err) = step {
            log(
                Level::Warn,
                &format!(
                    "[nand] moving {name} to OPFS stopped after {moved} of {}: {err:?}; the rest stays in IndexedDB and is moved at the next start",
                    names.len()
                ),
            );
            return;
        }
    }
    log(
        Level::Debug,
        &format!(
            "[nand] moved {} file(s) from IndexedDB to OPFS",
            names.len()
        ),
    );
}

async fn migrate_one(
    dir: &FileSystemDirectoryHandle,
    db: &IdbDatabase,
    name: &str,
) -> Result<(), JsValue> {
    if let Some(content) = legacy(db, name, false).await? {
        let existing = opfs::file(dir, name).await?;
        // A copy of the same size is one an interrupted move finished.
        if existing.map(|f| f.size()) != Some(content.size()) {
            opfs::write(dir, name, &content).await?;
        }
    }
    idb::write(db, &[idb::Write::Delete(NAND_CONTENT, name)]).await
}

async fn find(name: &str) -> Result<Option<Blob>, JsValue> {
    let dir = ready().await;
    if let Some(dir) = &dir {
        if let Some(file) = opfs::file(dir, name).await? {
            return Ok(Some(file.into()));
        }
    }
    let db = open().await?;
    let found = legacy(&db, name, dir.is_none()).await;
    db.close();
    found
}

/// The stored content named `name`, kept on disk while this worker lives since it is read from.
pub(crate) async fn lend(name: &str) -> Result<Option<Blob>, JsValue> {
    let found = find(name).await?;
    if found.is_some() {
        LENT.with(|lent| lent.borrow_mut().insert(name.to_owned()));
    }
    Ok(found)
}

/// The stored content named `name`, to read from; `undefined` if there is none.
#[wasm_bindgen]
pub async fn switch_nand_content(name: String) -> Result<JsValue, JsValue> {
    Ok(lend(&name).await?.map_or(JsValue::UNDEFINED, JsValue::from))
}

/// The size of the stored content named `name`; `undefined` if there is none.
#[wasm_bindgen]
pub async fn switch_nand_content_size(name: String) -> Result<JsValue, JsValue> {
    Ok(find(&name)
        .await?
        .map_or(JsValue::UNDEFINED, |blob| JsValue::from_f64(blob.size())))
}

/// Store `content` as `name` and index it as title `id` of `kind`.
#[wasm_bindgen]
pub async fn switch_nand_install(
    content: Blob,
    name: String,
    id: String,
    kind: f64,
) -> Result<(), JsValue> {
    serial(move |dir| async move {
        let db = open().await?;
        let outcome = install(dir.as_ref(), &db, &content, &name, &id, kind).await;
        db.close();
        outcome.map(|()| JsValue::UNDEFINED)
    })
    .await
    .map(drop)
}

async fn install(
    dir: Option<&FileSystemDirectoryHandle>,
    db: &IdbDatabase,
    content: &Blob,
    name: &str,
    id: &str,
    kind: f64,
) -> Result<(), JsValue> {
    let entry = js_sys::Object::new();
    Reflect::set(&entry, &"name".into(), &name.into())?;
    Reflect::set(&entry, &"kind".into(), &kind.into())?;
    let Some(dir) = dir else {
        return idb::write(
            db,
            &[
                idb::Write::Put(NAND_CONTENT, name, content.into()),
                idb::Write::Put(NAND_TITLES, id, entry.into()),
            ],
        )
        .await;
    };
    let previous = Reflect::get(&idb::get(db, NAND_TITLES, id).await?, &"name".into())
        .ok()
        .and_then(|name| name.as_string());
    let existing = opfs::file(dir, name).await?;
    // A lent file must not change under the session; NCA names are content hashes, so equal size is equal content.
    if !(is_lent(name) && existing.map(|f| f.size()) == Some(content.size())) {
        opfs::write(dir, name, content).await?;
    }
    // An older IndexedDB copy would otherwise be moved over this one.
    idb::write(
        db,
        &[
            idb::Write::Put(NAND_TITLES, id, entry.into()),
            idb::Write::Delete(NAND_CONTENT, name),
        ],
    )
    .await?;
    match previous {
        Some(previous) if previous != name => release(dir, db, &[previous]).await,
        _ => Ok(()),
    }
}

/// Remove title `id` and its content.
#[wasm_bindgen]
pub async fn switch_nand_remove(id: String) -> Result<(), JsValue> {
    serial(move |dir| async move {
        let db = open().await?;
        let outcome = remove(dir.as_ref(), &db, &id).await;
        db.close();
        outcome.map(|()| JsValue::UNDEFINED)
    })
    .await
    .map(drop)
}

async fn remove(
    dir: Option<&FileSystemDirectoryHandle>,
    db: &IdbDatabase,
    id: &str,
) -> Result<(), JsValue> {
    let entry = idb::get(db, NAND_TITLES, id).await?;
    let Some(name) = Reflect::get(&entry, &"name".into())
        .ok()
        .and_then(|n| n.as_string())
    else {
        return Ok(());
    };
    idb::write(
        db,
        &[
            idb::Write::Delete(NAND_TITLES, id),
            idb::Write::Delete(NAND_CONTENT, &name),
        ],
    )
    .await?;
    match dir {
        Some(dir) => release(dir, db, &[name]).await,
        None => Ok(()),
    }
}

/// Remove every title and all content.
#[wasm_bindgen]
pub async fn switch_nand_erase() -> Result<(), JsValue> {
    serial(|dir| async move {
        let db = open().await?;
        let mut outcome = idb::write(
            &db,
            &[
                idb::Write::Clear(NAND_CONTENT),
                idb::Write::Clear(NAND_TITLES),
            ],
        )
        .await;
        if let (Ok(()), Some(dir)) = (&outcome, &dir) {
            outcome = match opfs::files(dir).await {
                Ok(files) => {
                    let names: Vec<String> = files.into_iter().map(|(name, _)| name).collect();
                    release(dir, &db, &names).await
                }
                Err(err) => Err(err),
            };
        }
        db.close();
        outcome.map(|()| JsValue::UNDEFINED)
    })
    .await
    .map(drop)
}
