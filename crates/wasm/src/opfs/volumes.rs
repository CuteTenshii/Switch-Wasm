//! The SD card and saves as OPFS files, `sd/<path>` and `saves/<save key>/<path>`, read and
//! written in place. The session's `Vfs` holds the tree; a file the guest creates lives in
//! memory until the next round writes it here and hands the session the file instead.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::Promise;
use switch_core::cpu::SaveKey;
use switch_core::trace::Level;
use switch_core::vfs::{Vfs, ENTRY_TYPE_DIR, ENTRY_TYPE_FILE};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};
use web_sys::{Blob, DomException, File, FileSystemDirectoryHandle, FileSystemFileHandle};

use super::file::OpfsFile;
use super::{child, entries, escape, remove_tree, unescape, wait, Entry};
use crate::idb::persist::{stored, Stored};
use crate::idb::{
    self, NAND_DB, NAND_SAVES, NAND_STORES, NAND_VERSION, SD_DB, SD_STORE, SD_VERSION,
};
use crate::storage::{claimed, parse_save_key, writable, READ_ONLY};
use crate::{session_if_live, Session};

const SD_DIR: &str = "sd";
const SAVES_DIR: &str = "saves";
/// Uploads are written here and moved into place, so no partial file ever shows on a volume.
const STAGING_DIR: &str = "staging";
/// Paths named in a store line before the rest is counted.
const STORED_NAMED_MAX: usize = 8;

struct Roots {
    sd: FileSystemDirectoryHandle,
    /// `None` only in a read-only tab that found no saves stored.
    saves: Option<FileSystemDirectoryHandle>,
    staging: Option<FileSystemDirectoryHandle>,
}

/// A volume's directories and files by guest path, each file with its handle and snapshot.
type Tree = Vec<(String, Option<(FileSystemFileHandle, File)>)>;

thread_local! {
    /// Set by a restore that found OPFS; the SD card and saves are in IndexedDB otherwise.
    static ROOTS: RefCell<Option<Rc<Roots>>> = const { RefCell::new(None) };
    /// The last flush, settled either way.
    static TAIL: RefCell<Option<Promise>> = const { RefCell::new(None) };
    /// Paths whose change could not be stored, tried again by the next flush.
    static BACKLOG: RefCell<Vec<(Volume, String)>> = const { RefCell::new(Vec::new()) };
}

fn roots() -> Option<Rc<Roots>> {
    ROOTS.with_borrow(Clone::clone)
}

/// Whether the SD card and saves live in OPFS this session.
pub(crate) fn active() -> bool {
    roots().is_some()
}

fn log(level: Level, line: &str) {
    if let Some(s) = session_if_live() {
        s.cpu.log(level, line);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Volume {
    Sd,
    Save(SaveKey),
}

impl Volume {
    fn vfs(self, s: &mut Session) -> &mut Vfs {
        match self {
            Volume::Sd => &mut s.cpu.fs,
            Volume::Save(key) => s.cpu.save_data_mut(key),
        }
    }

    fn label(self) -> String {
        match self {
            Volume::Sd => "SD card".to_owned(),
            Volume::Save(key) => format!("save {key}"),
        }
    }

    async fn root(
        self,
        roots: &Roots,
        create: bool,
    ) -> Result<Option<FileSystemDirectoryHandle>, JsValue> {
        match self {
            Volume::Sd => Ok(Some(roots.sd.clone())),
            Volume::Save(key) => match &roots.saves {
                Some(saves) => child(saves, &escape(&key.to_string()), create).await,
                None => Ok(None),
            },
        }
    }
}

/// `/a/b/c` as its directories and its escaped name; `None` for the root.
fn split(path: &str) -> Option<(Vec<String>, String)> {
    let mut parts: Vec<String> = Vfs::normalize(path)
        .split('/')
        .filter(|part| !part.is_empty())
        .map(escape)
        .collect();
    let name = parts.pop()?;
    Some((parts, name))
}

/// The directory `dirs` under `root`; `None` if a part is missing and not to be made.
async fn resolve(
    root: FileSystemDirectoryHandle,
    dirs: &[String],
    create: bool,
) -> Result<Option<FileSystemDirectoryHandle>, JsValue> {
    let mut dir = root;
    for part in dirs {
        match child(&dir, part, create).await? {
            Some(next) => dir = next,
            None => return Ok(None),
        }
    }
    Ok(Some(dir))
}

fn has_name(err: &JsValue, name: &str) -> bool {
    err.dyn_ref::<DomException>()
        .is_some_and(|e| e.name() == name)
}

/// Every directory and file under `dir`.
async fn walk(dir: FileSystemDirectoryHandle) -> Result<Tree, JsValue> {
    let mut out = Vec::new();
    let mut stack = vec![(String::new(), dir)];
    while let Some((prefix, dir)) = stack.pop() {
        for (name, entry) in entries(&dir).await? {
            let path = format!("{prefix}/{}", unescape(&name));
            match entry {
                Entry::Dir(sub) => {
                    out.push((path.clone(), None));
                    stack.push((path, sub));
                }
                Entry::File(handle) => {
                    let snapshot: File = wait(handle.get_file()).await?.unchecked_into();
                    out.push((path, Some((handle, snapshot))));
                }
            }
        }
    }
    // Directories before the files in them, so empty ones are kept and parents are made once.
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Load the volumes from OPFS, moving what older builds kept in IndexedDB first. `None` when
/// there is no OPFS to use, so the caller restores from IndexedDB.
pub(crate) async fn restore() -> Option<Result<(usize, usize), JsValue>> {
    let writable = claimed().await;
    let sd = super::dir(SD_DIR, writable).await?;
    Some(restore_from(sd, writable).await)
}

async fn restore_from(
    sd: FileSystemDirectoryHandle,
    writable: bool,
) -> Result<(usize, usize), JsValue> {
    let saves = super::dir(SAVES_DIR, writable).await;
    let staging = if writable {
        let staging = super::dir(STAGING_DIR, true).await;
        if let Some(staging) = &staging {
            for (name, _) in entries(staging).await? {
                remove_tree(staging, &name).await?;
            }
        }
        staging
    } else {
        None
    };
    let roots = Rc::new(Roots { sd, saves, staging });
    if writable {
        migrate(&roots).await?;
    }
    let card = walk(roots.sd.clone()).await?;
    let mut saves = Vec::new();
    if let Some(root) = &roots.saves {
        for (name, entry) in entries(root).await? {
            let (Entry::Dir(dir), Some(key)) = (entry, parse_save_key(&unescape(&name))) else {
                continue;
            };
            saves.push((key, walk(dir).await?));
        }
    }
    ROOTS.with_borrow_mut(|slot| *slot = Some(roots));
    let Some(s) = session_if_live() else {
        return Ok((0, 0));
    };
    let card_count = card.len();
    mount(&mut s.cpu.fs, card);
    let mut save_count = 0;
    for (key, tree) in saves {
        save_count += tree.len();
        mount(s.cpu.save_data_mut(key), tree);
    }
    Ok((card_count, save_count))
}

fn mount(vfs: &mut Vfs, tree: Tree) {
    for (path, file) in tree {
        match file {
            None => vfs.create_dir(&path),
            Some((handle, snapshot)) => {
                vfs.attach_host(&path, OpfsFile::closed(handle, snapshot.into()))
            }
        }
    }
}

/// Copy the card and saves older builds kept in IndexedDB, then empty their stores. A move cut
/// short leaves IndexedDB as it was, and the next start copies everything again over it.
async fn migrate(roots: &Roots) -> Result<(), JsValue> {
    let card = stored(SD_DB, SD_VERSION, &[SD_STORE], SD_STORE).await?;
    let saves = stored(NAND_DB, NAND_VERSION, &NAND_STORES, NAND_SAVES).await?;
    if card.is_empty() && saves.is_empty() {
        return Ok(());
    }
    log(
        Level::Info,
        &format!(
            "Storage: moving {} SD card and {} save entries from IndexedDB to OPFS",
            card.len(),
            saves.len()
        ),
    );
    for Stored { key, data } in &card {
        store_entry(roots.sd.clone(), key, data.as_deref()).await?;
    }
    for Stored { key, data } in &saves {
        let Some((save, path)) = key.find('/').map(|cut| key.split_at(cut)) else {
            continue;
        };
        if parse_save_key(save).is_none() {
            continue;
        }
        let Some(saves) = &roots.saves else {
            continue;
        };
        let Some(root) = child(saves, &escape(save), true).await? else {
            continue;
        };
        store_entry(root, path, data.as_deref()).await?;
    }
    let db = idb::open(SD_DB, SD_VERSION, &[SD_STORE]).await?;
    let cleared = idb::write(&db, &[idb::Write::Clear(SD_STORE)]).await;
    db.close();
    cleared?;
    let db = idb::open(NAND_DB, NAND_VERSION, &NAND_STORES).await?;
    let cleared = idb::write(&db, &[idb::Write::Clear(NAND_SAVES)]).await;
    db.close();
    cleared?;
    log(
        Level::Info,
        "Storage: the SD card and saves are in OPFS now",
    );
    Ok(())
}

/// A directory (`data` is `None`) or file at `path` under `root`, replacing what is there.
async fn store_entry(
    root: FileSystemDirectoryHandle,
    path: &str,
    data: Option<&[u8]>,
) -> Result<(), JsValue> {
    let Some((dirs, name)) = split(path) else {
        return Ok(());
    };
    let dir = resolve(root, &dirs, true).await?.expect("made");
    let Some(data) = data else {
        child(&dir, &name, true).await?;
        return Ok(());
    };
    let handle = create_replacing(&dir, &name).await?;
    let access = super::open(&handle).await?;
    let written = access
        .truncate_with_f64(0.0)
        .and_then(|()| super::write_at(&access, 0, data))
        .and_then(|()| access.flush());
    access.close();
    written
}

/// The file `name` in `dir`, made if missing, replacing a directory of that name.
async fn create_replacing(
    dir: &FileSystemDirectoryHandle,
    name: &str,
) -> Result<FileSystemFileHandle, JsValue> {
    match super::create(dir, name).await {
        Err(err) if has_name(&err, "TypeMismatchError") => {
            remove_tree(dir, name).await?;
            super::create(dir, name).await
        }
        other => other,
    }
}

/// Store what changed since the last flush. Flushes run one after another, so one that
/// resolves has stored everything taken before it began.
pub(crate) async fn flush() {
    let previous = TAIL.with_borrow(Clone::clone);
    let run = future_to_promise(async move {
        if let Some(previous) = previous {
            JsFuture::from(previous).await.ok();
        }
        flush_once().await;
        Ok(JsValue::UNDEFINED)
    });
    TAIL.with_borrow_mut(|tail| *tail = Some(run.clone()));
    JsFuture::from(run).await.ok();
}

fn take_changes() -> Vec<(Volume, String)> {
    let Some(s) = session_if_live() else {
        return Vec::new();
    };
    let mut out: Vec<(Volume, String)> = s
        .cpu
        .fs
        .take_changes()
        .into_iter()
        .map(|change| (Volume::Sd, change.path))
        .collect();
    for key in s.cpu.save_keys() {
        for change in s.cpu.save_data_mut(key).take_changes() {
            out.push((Volume::Save(key), change.path));
        }
    }
    out
}

async fn flush_once() {
    let mut changes = BACKLOG.with_borrow_mut(std::mem::take);
    for change in take_changes() {
        if !changes.contains(&change) {
            changes.push(change);
        }
    }
    // A read-only tab keeps the guest's changes in its session and stores none of them.
    let Some(roots) = roots().filter(|_| writable()) else {
        return;
    };
    let mut stored: Vec<(String, Vec<String>)> = Vec::new();
    for (volume, path) in changes {
        match store(&roots, volume, &path).await {
            Ok(gone) => {
                let label = volume.label();
                let named = if gone {
                    format!("{path} (deleted)")
                } else {
                    path
                };
                match stored.iter_mut().find(|(what, _)| *what == label) {
                    Some((_, paths)) => paths.push(named),
                    None => stored.push((label, vec![named])),
                }
            }
            Err(err) => {
                log(
                    Level::Error,
                    &format!(
                        "{}: {path} could not be stored ({err:?}) - retrying on the next flush.",
                        volume.label()
                    ),
                );
                BACKLOG.with_borrow_mut(|backlog| backlog.push((volume, path)));
            }
        }
    }
    for (what, paths) in stored {
        let named = paths
            .iter()
            .take(STORED_NAMED_MAX)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = match paths.len().saturating_sub(STORED_NAMED_MAX) {
            0 => String::new(),
            n => format!(", and {n} more"),
        };
        log(
            Level::Debug,
            &format!(
                "[io] {what}: stored {} changes in OPFS: {named}{more}",
                paths.len()
            ),
        );
    }
    for file in OpfsFile::all() {
        if let Err(err) = OpfsFile::settle(&file).await.and_then(|()| file.flush()) {
            log(Level::Error, &format!("[io] a file could not be written to OPFS ({err:?}) - retrying on the next flush."));
        }
    }
}

/// Bring `path` in OPFS in line with the session: gone, a directory, or a file the session
/// holds in memory, which is written and handed back as an OPFS file. Returns whether it is gone.
async fn store(roots: &Roots, volume: Volume, path: &str) -> Result<bool, JsValue> {
    let Some((dirs, name)) = split(path) else {
        return Ok(false);
    };
    let (kind, in_memory) = {
        let Some(s) = session_if_live() else {
            return Ok(false);
        };
        let vfs = volume.vfs(s);
        (vfs.entry_type(path), vfs.file(path).is_some())
    };
    let Some(root) = volume.root(roots, kind.is_some()).await? else {
        return Ok(kind.is_none());
    };
    match kind {
        None => {
            if let Some(dir) = resolve(root, &dirs, false).await? {
                remove_tree(&dir, &name).await?;
            }
            Ok(true)
        }
        Some(ENTRY_TYPE_DIR) => {
            let dir = resolve(root, &dirs, true).await?.expect("made");
            match child(&dir, &name, true).await {
                Err(err) if has_name(&err, "TypeMismatchError") => {
                    remove_tree(&dir, &name).await?;
                    child(&dir, &name, true).await.map(|_| false)
                }
                other => other.map(|_| false),
            }
        }
        Some(ENTRY_TYPE_FILE) if in_memory => {
            let dir = resolve(root, &dirs, true).await?.expect("made");
            let handle = create_replacing(&dir, &name).await?;
            let access = super::open(&handle).await?;
            // From here to the end nothing yields, so the guest cannot change the file under it.
            let Some(s) = session_if_live() else {
                access.close();
                return Ok(false);
            };
            let vfs = volume.vfs(s);
            let Some(data) = vfs.file(path) else {
                // It changed since; that change is in the next flush.
                access.close();
                return Ok(false);
            };
            let written = access
                .truncate_with_f64(0.0)
                .and_then(|()| super::write_at(&access, 0, data))
                .and_then(|()| access.flush());
            if let Err(err) = written {
                access.close();
                return Err(err);
            }
            vfs.attach_host(path, OpfsFile::open(handle, access));
            Ok(false)
        }
        // An OPFS file already, written through its handle.
        Some(_) => Ok(false),
    }
}

/// Store `file` at `path` on a volume (`sd` or a save id) and put it in the session. In OPFS
/// it is copied in by range and never held in memory; elsewhere it is read and stored as a
/// change.
#[wasm_bindgen]
pub async fn switch_fs_upload(storage: String, path: String, file: Blob) -> Result<(), JsValue> {
    let volume = match storage.as_str() {
        "sd" => Volume::Sd,
        id => Volume::Save(parse_save_key(id).ok_or_else(|| format!("\"{id}\" names no storage"))?),
    };
    if !writable() {
        return Err(READ_ONLY.into());
    }
    let roots = roots();
    let staging = roots.as_ref().and_then(|roots| roots.staging.clone());
    let (Some(roots), Some(staging)) = (roots, staging) else {
        let data = crate::blob::read_vec(&file).await?;
        let s = session_if_live().ok_or("there is no session")?;
        volume.vfs(s).guest_write_file(&path, data);
        return Ok(());
    };
    let (dirs, name) = split(&path).ok_or("the root is no file")?;
    let root = volume.root(&roots, true).await?.expect("made");
    let dir = resolve(root, &dirs, true).await?.expect("made");
    // The session's copy holds the file open, and an open file cannot be replaced.
    if let Some(s) = session_if_live() {
        volume.vfs(s).write_file(&path, Vec::new());
    }
    let stored = super::write_via(&staging, &dir, &name, &file).await;
    let handle = match stored {
        Ok(handle) => handle,
        Err(err) => {
            // Put back what OPFS still has, so the session shows what is stored.
            if let Ok(Some(old)) = super::file(&dir, &name).await {
                if let (Some(s), Ok(handle)) =
                    (session_if_live(), wait(dir.get_file_handle(&name)).await)
                {
                    volume
                        .vfs(s)
                        .attach_host(&path, OpfsFile::closed(handle.unchecked_into(), old.into()));
                }
            }
            return Err(err);
        }
    };
    let snapshot: File = wait(handle.get_file()).await?.unchecked_into();
    let s = session_if_live().ok_or("there is no session")?;
    volume
        .vfs(s)
        .attach_host(&path, OpfsFile::closed(handle, snapshot.into()));
    Ok(())
}
