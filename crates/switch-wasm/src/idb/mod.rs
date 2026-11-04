//! IndexedDB from the worker, as futures.

pub(crate) mod persist;
pub(crate) mod usage;
pub(crate) mod users;

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, Function, Promise};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Event, IdbDatabase, IdbRequest, IdbTransaction, IdbTransactionMode, WorkerGlobalScope,
};

pub(crate) const SD_DB: &str = "switch-wasm-sd";
pub(crate) const SD_STORE: &str = "entries";
pub(crate) const NAND_DB: &str = "switch-wasm-nand";
pub(crate) const NAND_CONTENT: &str = "content";
pub(crate) const NAND_SAVES: &str = "saves";
pub(crate) const NAND_USERS: &str = "users";

/// The page creates and upgrades these databases too (`web/shared/db.ts`); keep both in step.
pub(crate) const SD_VERSION: u32 = 1;
pub(crate) const NAND_VERSION: u32 = 4;
pub(crate) const NAND_STORES: [&str; 5] =
    [NAND_CONTENT, "titles", NAND_SAVES, NAND_USERS, "controls"];

/// Open `name` at `version`, creating any of `stores` it lacks.
pub(crate) async fn open(
    name: &str,
    version: u32,
    stores: &[&str],
) -> Result<IdbDatabase, JsValue> {
    let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
    let factory = scope.indexed_db()?.ok_or("no IndexedDB")?;
    let request = factory.open_with_u32(name, version)?;
    let stores: Vec<String> = stores.iter().map(|s| (*s).to_owned()).collect();
    let source = request.clone();
    let upgrade = Closure::<dyn FnMut(Event)>::new(move |_| {
        if let Ok(db) = source.result() {
            let db: IdbDatabase = db.unchecked_into();
            for store in &stores {
                if !db.object_store_names().contains(store) {
                    db.create_object_store(store).ok();
                }
            }
        }
    });
    request.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));
    let opened = Rc::new(RefCell::new(JsValue::UNDEFINED));
    let slot = opened.clone();
    let outcome = drive(&request, move |result| {
        *slot.borrow_mut() = result;
        true
    })
    .await;
    request.set_onupgradeneeded(None);
    outcome?;
    let db = opened.borrow().clone();
    Ok(db.unchecked_into())
}

/// Every key and value of `store`, read in one transaction so they stay in step.
pub(crate) async fn get_all(
    db: &IdbDatabase,
    store: &str,
) -> Result<Vec<(String, JsValue)>, JsValue> {
    let tx = db.transaction_with_str(store)?;
    let store = tx.object_store(store)?;
    let keys = store.get_all_keys()?;
    let values = store.get_all()?;
    finish(&tx).await?;
    let keys: Array = keys.result()?.unchecked_into();
    let values: Array = values.result()?.unchecked_into();
    Ok(keys
        .iter()
        .zip(values.iter())
        .filter_map(|(key, value)| Some((key.as_string()?, value)))
        .collect())
}

/// Store the values and delete the keys whose value is `None`, in one transaction.
pub(crate) async fn apply(
    db: &IdbDatabase,
    store: &str,
    entries: &[(String, Option<JsValue>)],
) -> Result<(), JsValue> {
    let tx = db.transaction_with_str_and_mode(store, IdbTransactionMode::Readwrite)?;
    let store = tx.object_store(store)?;
    for (key, value) in entries {
        let key = JsValue::from_str(key);
        match value {
            Some(value) => store.put_with_key(value, &key)?,
            None => store.delete(&key)?,
        };
    }
    finish(&tx).await
}

/// Await a transaction's commit.
async fn finish(tx: &IdbTransaction) -> Result<(), JsValue> {
    let mut handlers = None;
    let done = Promise::new(&mut |resolve: Function, reject: Function| {
        let on_complete = Closure::<dyn FnMut(Event)>::new(move |_| {
            resolve.call0(&JsValue::NULL).ok();
        });
        let source = tx.clone();
        let on_error = Closure::<dyn FnMut(Event)>::new(move |_| {
            let error = source
                .error()
                .map_or_else(|| "IndexedDB transaction aborted".into(), JsValue::from);
            reject.call1(&JsValue::NULL, &error).ok();
        });
        tx.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
        tx.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        tx.set_onabort(Some(on_error.as_ref().unchecked_ref()));
        handlers = Some((on_complete, on_error));
    });
    let result = JsFuture::from(done).await.map(|_| ());
    tx.set_oncomplete(None);
    tx.set_onerror(None);
    tx.set_onabort(None);
    result
}

/// Await `request` until `step` (called with each result) returns true.
pub(crate) async fn drive(
    request: &IdbRequest,
    step: impl FnMut(JsValue) -> bool + 'static,
) -> Result<(), JsValue> {
    let mut handlers = None;
    let mut step = Some(step);
    let done = Promise::new(&mut |resolve: Function, reject: Function| {
        let mut step = step.take().expect("the executor runs once");
        let source = request.clone();
        let on_success = Closure::<dyn FnMut(Event)>::new(move |_| {
            let finished = match source.result() {
                Ok(result) => step(result),
                Err(_) => true,
            };
            if finished {
                resolve.call0(&JsValue::NULL).ok();
            }
        });
        let source = request.clone();
        let on_error = Closure::<dyn FnMut(Event)>::new(move |_| {
            let error = source
                .error()
                .ok()
                .flatten()
                .map_or(JsValue::NULL, JsValue::from);
            reject.call1(&JsValue::NULL, &error).ok();
        });
        request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        request.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        handlers = Some((on_success, on_error));
    });
    let result = JsFuture::from(done).await.map(|_| ());
    request.set_onsuccess(None);
    request.set_onerror(None);
    result
}
