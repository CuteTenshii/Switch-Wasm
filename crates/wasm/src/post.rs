//! Messages from the worker to the page, posted without a trip through JavaScript.

use js_sys::{Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::DedicatedWorkerGlobalScope;

/// Post `{ type, ...fields }` to the page.
pub(crate) fn post(kind: &str, fields: &[(&str, JsValue)]) {
    let message = Object::new();
    Reflect::set(&message, &"type".into(), &kind.into()).ok();
    for (key, value) in fields {
        Reflect::set(&message, &(*key).into(), value).ok();
    }
    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    scope.post_message(&message).ok();
}
