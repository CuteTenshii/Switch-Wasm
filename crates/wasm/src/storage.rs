//! The SD card and save data as the page sees and changes them, and whether this tab may
//! store anything at all.

use switch_core::cpu::SaveKey;
use switch_core::vfs::{Vfs, ENTRY_TYPE_FILE};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{session, Session, SyncCell};

/// Set by [`claimed`]; false until the claim settles, so nothing is stored before it does.
static WRITABLE: SyncCell<bool> = SyncCell::new(false);

/// Whether this tab stores the SD card, saves and NAND, as the settled claim decided. A second
/// tab reads them but keeps its changes in memory, since the first holds the files open.
pub(crate) fn writable() -> bool {
    // Host builds have one session and nothing to share storage with.
    if cfg!(not(target_arch = "wasm32")) {
        return true;
    }
    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
    unsafe { *WRITABLE.get() }
}

pub(crate) const READ_ONLY: &str =
    "This tab can't change anything stored: the emulator is open in another tab.";

fn refuse_read_only() -> Result<(), String> {
    if writable() {
        Ok(())
    } else {
        Err(READ_ONLY.to_owned())
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use lock::{claimed, start_claim};

/// One tab at a time stores: the first takes a Web Lock for its worker's life.
#[cfg(target_arch = "wasm32")]
mod lock {
    use std::cell::RefCell;

    use js_sys::{Function, Object, Promise, Reflect};
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::prelude::wasm_bindgen;
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::{future_to_promise, JsFuture};
    use web_sys::AbortController;

    use super::WRITABLE;
    use crate::post::post;

    const LOCK_NAME: &str = "switch-wasm-storage";
    /// A reload's worker waits this long for the old one's lock before taking itself for a
    /// second tab.
    const CLAIM_WAIT_MS: f64 = 2000.0;

    thread_local! {
        static CLAIM: RefCell<Option<Promise>> = const { RefCell::new(None) };
    }

    /// Ask for the lock now, so the answer is ready by the time storage is restored.
    pub(crate) fn start_claim() {
        CLAIM.with_borrow_mut(|slot| {
            slot.get_or_insert_with(|| {
                future_to_promise(async {
                    let writable = claim().await;
                    // SAFETY: only the system worker calls exports; see the `SESSION` comment.
                    unsafe { *WRITABLE.get() = writable };
                    if !writable {
                        post("storage_read_only", &[]);
                    }
                    Ok(JsValue::from_bool(writable))
                })
            });
        });
    }

    /// Whether this tab stores anything, once the claim has settled.
    pub(crate) async fn claimed() -> bool {
        start_claim();
        let pending = CLAIM.with_borrow(|slot| slot.clone().expect("just started"));
        JsFuture::from(pending)
            .await
            .is_ok_and(|writable| writable.is_truthy())
    }

    /// Whether this tab stores anything; the page asks before changing profiles.
    #[wasm_bindgen]
    pub async fn switch_storage_writable() -> bool {
        claimed().await
    }

    /// Take the lock, or report that another tab has it. Without Web Locks there is no telling
    /// tabs apart, so every tab stores, as before.
    async fn claim() -> bool {
        let global = js_sys::global();
        let Some(locks) = Reflect::get(&global, &"navigator".into())
            .and_then(|navigator| Reflect::get(&navigator, &"locks".into()))
            .ok()
            .filter(|locks| !locks.is_undefined())
        else {
            return true;
        };
        let Ok(request) =
            Reflect::get(&locks, &"request".into()).and_then(JsCast::dyn_into::<Function>)
        else {
            return true;
        };
        let Ok(abort) = AbortController::new() else {
            return true;
        };
        let options = Object::new();
        Reflect::set(&options, &"signal".into(), &abort.signal()).ok();
        let granted = Promise::new(&mut |resolve: Function, _reject: Function| {
            let on_grant = resolve.clone();
            // The callback's promise never settles, so the lock is held until the worker ends.
            let hold = Closure::once_into_js(move |_lock: JsValue| -> JsValue {
                on_grant.call1(&JsValue::NULL, &JsValue::TRUE).ok();
                Promise::new(&mut |_, _| {}).into()
            });
            let refused: Function = Closure::once_into_js(move |_err: JsValue| {
                resolve.call1(&JsValue::NULL, &JsValue::FALSE).ok();
            })
            .unchecked_into();
            match request.call3(&locks, &LOCK_NAME.into(), &options, &hold) {
                Ok(pending) => {
                    let pending: Promise = pending.unchecked_into();
                    let catch = Reflect::get(&pending, &"catch".into())
                        .and_then(JsCast::dyn_into::<Function>);
                    if let Ok(catch) = catch {
                        catch.call1(&pending, &refused).ok();
                    }
                }
                Err(err) => {
                    refused.call1(&JsValue::NULL, &err).ok();
                }
            }
        });
        let give_up = Closure::once_into_js(move || abort.abort());
        if let Ok(set_timeout) =
            Reflect::get(&global, &"setTimeout".into()).and_then(JsCast::dyn_into::<Function>)
        {
            set_timeout
                .call2(&global, &give_up, &CLAIM_WAIT_MS.into())
                .ok();
        }
        JsFuture::from(granted)
            .await
            .is_ok_and(|granted| granted.is_truthy())
    }
}

/// A save's [`Display`](std::fmt::Display) form: `<16 hex digits>`, `<id>@<32 hex digits of
/// uid>` or `<id>.cache<index>`.
pub(crate) fn parse_save_key(text: &str) -> Option<SaveKey> {
    if let Some((id, index)) = text.split_once(".cache") {
        return Some(SaveKey::cache_storage(
            u64::from_str_radix(id, 16).ok()?,
            index.parse().ok()?,
        ));
    }
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
pub(crate) fn volume<'a>(s: &'a mut Session, volume: &str) -> Result<&'a mut Vfs, String> {
    if volume == "sd" {
        return Ok(&mut s.cpu.fs);
    }
    match parse_save_key(volume) {
        Some(key) => Ok(s.cpu.save_data_mut(key)),
        None => Err(format!("\"{volume}\" names no storage")),
    }
}

// Paths are guest paths; `sdmc:` and extra slashes are normalized away. Changes are recorded
// like the guest's, so the next flush stores them.

/// Put a file on a volume, creating its directories and, for a save, the save.
#[wasm_bindgen]
pub fn switch_fs_write(storage: &str, path: &str, data: &[u8]) -> Result<(), String> {
    refuse_read_only()?;
    volume(session(), storage)?.guest_write_file(path, data.to_vec());
    Ok(())
}

/// Create a directory and any missing parents.
#[wasm_bindgen]
pub fn switch_fs_create_dir(storage: &str, path: &str) -> Result<(), String> {
    refuse_read_only()?;
    volume(session(), storage)?.guest_create_dir(path);
    Ok(())
}

/// Every path on a volume as `[path, kind, size]`, `kind` 0 for a directory and 1 for a file.
#[wasm_bindgen]
pub fn switch_fs_entries(storage: &str) -> Result<js_sys::Array, String> {
    let out = js_sys::Array::new();
    for (path, kind, size) in volume(session(), storage)?.entries() {
        out.push(&js_sys::Array::of3(
            &path.into(),
            &f64::from(kind).into(),
            &(size as f64).into(),
        ));
    }
    Ok(out)
}

/// A file's bytes, or `undefined` for a directory or a missing path.
#[wasm_bindgen]
pub fn switch_fs_read(storage: &str, path: &str) -> Result<Option<Vec<u8>>, String> {
    let vfs = volume(session(), storage)?;
    if vfs.entry_type(path) != Some(ENTRY_TYPE_FILE) {
        return Ok(None);
    }
    let mut data = vec![0; vfs.size(path).unwrap_or(0) as usize];
    let read = vfs.read(path, 0, &mut data).unwrap_or(0);
    if read != data.len() {
        return Err(format!("{path}: read {read} of {} bytes", data.len()));
    }
    Ok(Some(data))
}

/// The saves holding anything, in their [`Display`](std::fmt::Display) form.
#[wasm_bindgen]
pub fn switch_saves() -> Vec<String> {
    let s = session();
    let mut ids: Vec<String> = s
        .cpu
        .save_keys()
        .into_iter()
        .filter(|key| {
            s.cpu
                .save_data(*key)
                .is_some_and(|vfs| !vfs.entries().is_empty())
        })
        .map(|key| key.to_string())
        .collect();
    ids.sort();
    ids
}

/// Remove one entry; a directory's contents are removed separately.
#[wasm_bindgen]
pub fn switch_fs_remove(storage: &str, path: &str) -> Result<(), String> {
    refuse_read_only()?;
    volume(session(), storage)?.remove(path);
    Ok(())
}
