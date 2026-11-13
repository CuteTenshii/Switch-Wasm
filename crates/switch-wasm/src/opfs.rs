//! The origin private file system from the worker, as futures. Every entry point is `None` or
//! an error where the engine has no OPFS or refuses it, so callers can keep to IndexedDB.

use js_sys::{AsyncIterator, Function, Promise, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Blob, DomException, File, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemReadWriteOptions,
    FileSystemSyncAccessHandle, WorkerGlobalScope,
};

/// Suffix of a file being written; it is moved over the real name once whole.
pub(crate) const PART: &str = ".part";

/// Read from the source at a time while copying it in.
const COPY_CHUNK: f64 = (8 << 20) as f64;

async fn wait(promise: Promise) -> Result<JsValue, JsValue> {
    JsFuture::from(promise).await
}

pub(crate) fn is_not_found(err: &JsValue) -> bool {
    err.dyn_ref::<DomException>()
        .is_some_and(|e| e.name() == "NotFoundError")
}

/// Whether files can be written here: `createSyncAccessHandle` is what writes them.
fn writable() -> bool {
    let global = js_sys::global();
    let Ok(class) = Reflect::get(&global, &"FileSystemFileHandle".into()) else {
        return false;
    };
    if class.is_undefined() {
        return false;
    }
    Reflect::get(&class, &"prototype".into())
        .and_then(|proto| Reflect::has(&proto, &"createSyncAccessHandle".into()))
        .unwrap_or(false)
}

/// The directory `name` under the origin's root, or `None` without OPFS or, unless `create`,
/// when it does not exist.
pub(crate) async fn dir(name: &str, create: bool) -> Option<FileSystemDirectoryHandle> {
    if !writable() {
        return None;
    }
    let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
    let storage = scope.navigator().storage();
    // Through `Reflect`, since `getDirectory` throws outright in some engines rather than rejecting.
    let get_directory = Reflect::get(&storage, &"getDirectory".into())
        .ok()?
        .dyn_into::<Function>()
        .ok()?;
    let root: FileSystemDirectoryHandle =
        wait(get_directory.call0(&storage).ok()?.unchecked_into())
            .await
            .ok()?
            .unchecked_into();
    let options = FileSystemGetDirectoryOptions::new();
    options.set_create(create);
    wait(root.get_directory_handle_with_options(name, &options))
        .await
        .ok()
        .map(JsCast::unchecked_into)
}

/// The file `name` in `dir` as it is now, or `None` if there is none.
pub(crate) async fn file(
    dir: &FileSystemDirectoryHandle,
    name: &str,
) -> Result<Option<File>, JsValue> {
    let handle = match wait(dir.get_file_handle(name)).await {
        Ok(handle) => handle.unchecked_into::<FileSystemFileHandle>(),
        Err(err) if is_not_found(&err) => return Ok(None),
        Err(err) => return Err(err),
    };
    Ok(Some(wait(handle.get_file()).await?.unchecked_into()))
}

/// Remove `name` from `dir`; a missing file is no error.
pub(crate) async fn remove(dir: &FileSystemDirectoryHandle, name: &str) -> Result<(), JsValue> {
    match wait(dir.remove_entry(name)).await {
        Err(err) if !is_not_found(&err) => Err(err),
        _ => Ok(()),
    }
}

/// The files in `dir`, with each one's size.
pub(crate) async fn files(dir: &FileSystemDirectoryHandle) -> Result<Vec<(String, u64)>, JsValue> {
    let entries: AsyncIterator = dir.values();
    let mut out = Vec::new();
    loop {
        let step = wait(entries.next()?).await?;
        if Reflect::get(&step, &"done".into())?.is_truthy() {
            return Ok(out);
        }
        let Ok(handle) = Reflect::get(&step, &"value".into())?.dyn_into::<FileSystemFileHandle>()
        else {
            continue;
        };
        // A file removed since the listing began is no longer there to report.
        if let Ok(file) = wait(handle.get_file()).await {
            let file: File = file.unchecked_into();
            out.push((file.name(), file.size() as u64));
        }
    }
}

/// Closes the handle however the copy ends, so the file is not left locked.
struct Access(FileSystemSyncAccessHandle);

impl Drop for Access {
    fn drop(&mut self) {
        self.0.close();
    }
}

async fn copy_into(handle: &FileSystemFileHandle, content: &Blob) -> Result<(), JsValue> {
    let access = Access(
        wait(handle.create_sync_access_handle())
            .await?
            .unchecked_into(),
    );
    access.0.truncate_with_f64(0.0)?;
    let size = content.size();
    let options = FileSystemReadWriteOptions::new();
    let mut at = 0.0;
    while at < size {
        let end = (at + COPY_CHUNK).min(size);
        let bytes =
            Uint8Array::new(&wait(content.slice_with_f64_and_f64(at, end)?.array_buffer()).await?);
        options.set_at(at);
        let wrote = access
            .0
            .write_with_buffer_source_and_options(&bytes, &options)?;
        if wrote != end - at {
            return Err(format!("OPFS took {wrote} of {} bytes at {at}", end - at).into());
        }
        at = end;
    }
    access.0.flush()
}

async fn create(
    dir: &FileSystemDirectoryHandle,
    name: &str,
) -> Result<FileSystemFileHandle, JsValue> {
    let options = FileSystemGetFileOptions::new();
    options.set_create(true);
    Ok(wait(dir.get_file_handle_with_options(name, &options))
        .await?
        .unchecked_into())
}

async fn rename(handle: &FileSystemFileHandle, name: &str) -> Result<(), JsValue> {
    let mv: Function = Reflect::get(handle, &"move".into())?.unchecked_into();
    wait(mv.call1(handle, &name.into())?.unchecked_into())
        .await
        .map(drop)
}

/// Store `content` as `name` in `dir`. It is written beside the old file and moved over it, so
/// `name` is never a partial file; engines without `move` write it in place.
pub(crate) async fn write(
    dir: &FileSystemDirectoryHandle,
    name: &str,
    content: &Blob,
) -> Result<(), JsValue> {
    let part_name = format!("{name}{PART}");
    let part = create(dir, &part_name).await?;
    if let Err(err) = copy_into(&part, content).await {
        remove(dir, &part_name).await.ok();
        return Err(err);
    }
    if !Reflect::has(&part, &"move".into())? {
        remove(dir, &part_name).await?;
        return copy_into(&create(dir, name).await?, content).await;
    }
    if rename(&part, name).await.is_ok() {
        return Ok(());
    }
    // Some engines refuse to move onto an existing file.
    remove(dir, name).await?;
    rename(&part, name).await
}
