//! What IndexedDB holds and the origin has left, reported to the core for the
//! filesystem space queries (which cannot wait on a promise themselves).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use js_sys::{Array, ArrayBuffer, Function, Promise, Reflect, Uint8Array};
use switch_core::cpu::StorageUsage;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    Blob, IdbCursorWithValue, IdbDatabase, IdbFactory, StorageEstimate, WorkerGlobalScope,
};

use super::{drive, NAND_CONTENT, NAND_DB, NAND_SAVES, SD_DB, SD_STORE};
use crate::session_if_live;

/// Least time between two scans.
const REFRESH_INTERVAL_MS: f64 = 2000.0;

static IN_FLIGHT: crate::SyncCell<bool> = crate::SyncCell::new(false);
static LAST_START: crate::SyncCell<f64> = crate::SyncCell::new(f64::NEG_INFINITY);

/// Scan in the background when the guest has asked for space since the last scan, or never has.
pub(crate) fn refresh_if_due(wanted: bool) {
    let now = js_sys::Date::now();
    // SAFETY: only the system worker calls exports; see the `SESSIONS` comment.
    let (in_flight, last) = unsafe { (&mut *IN_FLIGHT.get(), &mut *LAST_START.get()) };
    let never = *last == f64::NEG_INFINITY;
    if *in_flight || !(never || wanted && now - *last >= REFRESH_INTERVAL_MS) {
        return;
    }
    *in_flight = true;
    *last = now;
    spawn_local(async move {
        let usage = measure().await;
        unsafe { *IN_FLIGHT.get() = false };
        let Some(s) = session_if_live() else {
            return;
        };
        match usage {
            Ok(usage) => s.cpu.set_storage_usage(usage),
            Err(err) => s.cpu.log(
                switch_core::trace::Level::Warn,
                &format!("[fs] could not measure storage: {err:?}"),
            ),
        }
    });
}

async fn measure() -> Result<StorageUsage, JsValue> {
    let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
    let estimate: StorageEstimate = JsFuture::from(scope.navigator().storage().estimate()?)
        .await?
        .unchecked_into();
    let quota = estimate.get_quota().unwrap_or(0.0);
    let used = estimate.get_usage().unwrap_or(0.0);
    let factory = scope.indexed_db()?.ok_or("no IndexedDB")?;
    let present = databases(&factory).await?;
    let mut usage = StorageUsage {
        free: (quota - used).max(0.0) as u64,
        ..StorageUsage::default()
    };
    if present.iter().any(|name| name == SD_DB) {
        usage.sd = store_bytes(&factory, SD_DB, SD_STORE, entry_bytes).await?;
    }
    if present.iter().any(|name| name == NAND_DB) {
        usage.system = store_bytes(&factory, NAND_DB, NAND_CONTENT, content_bytes).await?;
        usage.user = store_bytes(&factory, NAND_DB, NAND_SAVES, entry_bytes).await?;
    }
    Ok(usage)
}

/// Names of the databases that exist, so a missing one is not created by being opened.
async fn databases(factory: &IdbFactory) -> Result<Vec<String>, JsValue> {
    let list = Reflect::get(factory, &"databases".into())?
        .dyn_into::<Function>()?
        .call0(factory)?;
    let list: Array = JsFuture::from(list.unchecked_into::<Promise>())
        .await?
        .unchecked_into();
    Ok(list
        .iter()
        .filter_map(|info| Reflect::get(&info, &"name".into()).ok()?.as_string())
        .collect())
}

/// Total of `size` over every value of `store`, read one at a time.
async fn store_bytes(
    factory: &IdbFactory,
    db: &str,
    store: &str,
    size: fn(&JsValue) -> u64,
) -> Result<u64, JsValue> {
    let opened = Rc::new(RefCell::new(JsValue::UNDEFINED));
    let request = factory.open(db)?;
    let slot = opened.clone();
    drive(&request, move |result| {
        *slot.borrow_mut() = result;
        true
    })
    .await?;
    let db: IdbDatabase = opened.borrow().clone().unchecked_into();
    let total = Rc::new(Cell::new(0u64));
    let outcome = if db.object_store_names().contains(store) {
        let request = db
            .transaction_with_str(store)?
            .object_store(store)?
            .open_cursor()?;
        let total = total.clone();
        drive(&request, move |result| {
            match result.dyn_into::<IdbCursorWithValue>() {
                Ok(cursor) => {
                    total.set(total.get() + size(&cursor.value().unwrap_or(JsValue::UNDEFINED)));
                    cursor.continue_().is_err()
                }
                Err(_) => true,
            }
        })
        .await
    } else {
        Ok(())
    };
    db.close();
    outcome.map(|()| total.get())
}

/// An SD card or save entry: `{ kind, data? }`.
fn entry_bytes(entry: &JsValue) -> u64 {
    Reflect::get(entry, &"data".into())
        .ok()
        .and_then(|data| data.dyn_into::<Uint8Array>().ok())
        .map_or(0, |data| u64::from(data.byte_length()))
}

/// A NAND content file, stored as a `Blob` (an `ArrayBuffer` in older stores).
fn content_bytes(content: &JsValue) -> u64 {
    if let Some(blob) = content.dyn_ref::<Blob>() {
        blob.size() as u64
    } else if let Some(buffer) = content.dyn_ref::<ArrayBuffer>() {
        u64::from(buffer.byte_length())
    } else {
        0
    }
}
