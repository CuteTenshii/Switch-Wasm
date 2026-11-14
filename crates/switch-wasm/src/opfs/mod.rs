//! The origin private file system from the worker, as futures. Every entry point is `None` or
//! an error where the engine has no OPFS or refuses it, so callers can keep to IndexedDB.

pub(crate) mod file;
pub(crate) mod volumes;

use js_sys::{AsyncIterator, Function, Promise, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Blob, DomException, File, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemReadWriteOptions,
    FileSystemRemoveOptions, FileSystemSyncAccessHandle, WorkerGlobalScope,
};

/// Suffix of a file being written; it is moved over the real name once whole.
pub(crate) const PART: &str = ".part";

/// Read from the source at a time while copying it in.
const COPY_CHUNK: f64 = (8 << 20) as f64;

pub(crate) async fn wait(promise: Promise) -> Result<JsValue, JsValue> {
    JsFuture::from(promise).await
}

pub(crate) fn is_not_found(err: &JsValue) -> bool {
    err.dyn_ref::<DomException>()
        .is_some_and(|e| e.name() == "NotFoundError")
}

/// Whether the engine can write OPFS files: `createSyncAccessHandle` is what writes them.
fn available() -> bool {
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
    if !available() {
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

/// The directory `name` in `dir`, made if `create`; `None` if it does not exist and is not made.
pub(crate) async fn child(
    dir: &FileSystemDirectoryHandle,
    name: &str,
    create: bool,
) -> Result<Option<FileSystemDirectoryHandle>, JsValue> {
    let options = FileSystemGetDirectoryOptions::new();
    options.set_create(create);
    match wait(dir.get_directory_handle_with_options(name, &options)).await {
        Ok(handle) => Ok(Some(handle.unchecked_into())),
        Err(err) if is_not_found(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Remove `name` from `dir` and everything under it; a missing entry is no error.
pub(crate) async fn remove_tree(
    dir: &FileSystemDirectoryHandle,
    name: &str,
) -> Result<(), JsValue> {
    let options = FileSystemRemoveOptions::new();
    options.set_recursive(true);
    match wait(dir.remove_entry_with_options(name, &options)).await {
        Err(err) if !is_not_found(&err) => Err(err),
        _ => Ok(()),
    }
}

/// One entry of a directory.
pub(crate) enum Entry {
    Dir(FileSystemDirectoryHandle),
    File(FileSystemFileHandle),
}

/// The entries of `dir` by name.
pub(crate) async fn entries(
    dir: &FileSystemDirectoryHandle,
) -> Result<Vec<(String, Entry)>, JsValue> {
    let listing: AsyncIterator = dir.entries();
    let mut out = Vec::new();
    loop {
        let step = wait(listing.next()?).await?;
        if Reflect::get(&step, &"done".into())?.is_truthy() {
            return Ok(out);
        }
        let pair: js_sys::Array = Reflect::get(&step, &"value".into())?.unchecked_into();
        let Some(name) = pair.get(0).as_string() else {
            continue;
        };
        let handle = pair.get(1);
        if let Some(file) = handle.dyn_ref::<FileSystemFileHandle>() {
            out.push((name, Entry::File(file.clone())));
        } else if let Ok(dir) = handle.dyn_into::<FileSystemDirectoryHandle>() {
            out.push((name, Entry::Dir(dir)));
        }
    }
}

/// A guest file name as an OPFS one: OPFS refuses a backslash, so it and the escape `%` are encoded.
pub(crate) fn escape(name: &str) -> String {
    name.replace('%', "%25").replace('\\', "%5C")
}

pub(crate) fn unescape(name: &str) -> String {
    name.replace("%5C", "\\").replace("%25", "%")
}

/// Open `handle` for synchronous reads and writes; fails while anything else has it open.
pub(crate) async fn open(
    handle: &FileSystemFileHandle,
) -> Result<FileSystemSyncAccessHandle, JsValue> {
    Ok(wait(handle.create_sync_access_handle())
        .await?
        .unchecked_into())
}

/// Write all of `data` at `at`. The handle reads straight from wasm memory where the engine
/// takes a view of shared memory, and through a copy where it does not.
pub(crate) fn write_at(
    access: &FileSystemSyncAccessHandle,
    at: u64,
    data: &[u8],
) -> Result<(), JsValue> {
    let options = FileSystemReadWriteOptions::new();
    options.set_at(at as f64);
    // SAFETY: the view is used before anything can allocate and grow memory.
    let view = unsafe { Uint8Array::view(data) };
    let wrote = match access.write_with_buffer_source_and_options(&view, &options) {
        Ok(n) => n,
        Err(_) => access.write_with_buffer_source_and_options(&Uint8Array::from(data), &options)?,
    };
    if wrote as usize != data.len() {
        return Err(format!("OPFS took {wrote} of {} bytes at {at}", data.len()).into());
    }
    Ok(())
}

/// Read into `out` from `at`; how many bytes there were.
pub(crate) fn read_at(
    access: &FileSystemSyncAccessHandle,
    at: u64,
    out: &mut [u8],
) -> Result<usize, JsValue> {
    let options = FileSystemReadWriteOptions::new();
    options.set_at(at as f64);
    // SAFETY: as in `write_at`.
    let view = unsafe { Uint8Array::view_mut_raw(out.as_mut_ptr(), out.len()) };
    if let Ok(n) = access.read_with_buffer_source_and_options(&view, &options) {
        return Ok(n as usize);
    }
    let copy = Uint8Array::new_with_length(out.len() as u32);
    let n = access.read_with_buffer_source_and_options(&copy, &options)? as usize;
    copy.subarray(0, n as u32).copy_to(&mut out[..n]);
    Ok(n)
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

pub(crate) async fn create(
    dir: &FileSystemDirectoryHandle,
    name: &str,
) -> Result<FileSystemFileHandle, JsValue> {
    let options = FileSystemGetFileOptions::new();
    options.set_create(true);
    Ok(wait(dir.get_file_handle_with_options(name, &options))
        .await?
        .unchecked_into())
}

async fn rename(
    handle: &FileSystemFileHandle,
    to: &FileSystemDirectoryHandle,
    name: &str,
) -> Result<(), JsValue> {
    let mv: Function = Reflect::get(handle, &"move".into())?.unchecked_into();
    wait(mv.call2(handle, to, &name.into())?.unchecked_into())
        .await
        .map(drop)
}

/// Store `content` as `name` in `dir`. It is written beside the old file and moved over it, so
/// `name` is never a partial file; engines without `move` write it in place.
pub(crate) async fn write(
    dir: &FileSystemDirectoryHandle,
    name: &str,
    content: &Blob,
) -> Result<FileSystemFileHandle, JsValue> {
    write_via(dir, dir, name, content).await
}

/// [`write`], with the partial file kept in `staging` rather than beside `name`, for a
/// directory whose every name means something.
pub(crate) async fn write_via(
    staging: &FileSystemDirectoryHandle,
    dir: &FileSystemDirectoryHandle,
    name: &str,
    content: &Blob,
) -> Result<FileSystemFileHandle, JsValue> {
    let part_name = format!("{name}{PART}");
    let part = create(staging, &part_name).await?;
    if let Err(err) = copy_into(&part, content).await {
        remove(staging, &part_name).await.ok();
        return Err(err);
    }
    if !Reflect::has(&part, &"move".into())? {
        remove(staging, &part_name).await?;
        let file = create(dir, name).await?;
        copy_into(&file, content).await?;
        return Ok(file);
    }
    if rename(&part, dir, name).await.is_err() {
        // Some engines refuse to move onto an existing file.
        remove(dir, name).await?;
        rename(&part, dir, name).await?;
    }
    // The moved handle names the file at its new place.
    Ok(part)
}
