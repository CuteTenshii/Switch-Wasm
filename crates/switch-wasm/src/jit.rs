//! Compiles the core's emitted wasm blocks into this module's function table,
//! so the core can enter them with a plain `call_indirect`.

use js_sys::{Function, Object, Reflect, Uint8Array, WebAssembly};
use std::cell::RefCell;
use switch_core::cpu::{set_jit_host, tail_call_probe, Entry, JitHost};
use wasm_bindgen::{JsCast, JsValue};

thread_local! {
    /// Freed table slots awaiting reuse.
    static FREE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

/// Lets the core emit blocks.
pub fn attach() {
    set_jit_host(JitHost {
        install,
        release,
        tail_calls: tail_calls(),
    });
}

fn tail_calls() -> bool {
    let probe = tail_call_probe();
    let bytes = Uint8Array::new_with_length(probe.len() as u32);
    bytes.copy_from(&probe);
    WebAssembly::validate(bytes.as_ref()).unwrap_or(false)
}

fn table() -> Option<WebAssembly::Table> {
    wasm_bindgen::function_table().dyn_into().ok()
}

/// Compiles `code` into a table slot, answering its index or zero on failure.
fn install(code: &[u8]) -> Entry {
    compile(code).unwrap_or(0)
}

fn compile(code: &[u8]) -> Option<Entry> {
    let bytes = Uint8Array::new_with_length(code.len() as u32);
    bytes.copy_from(code);
    let module = WebAssembly::Module::new(bytes.as_ref()).ok()?;

    // This module's own linear memory, and its function table for blocks that jump to others.
    let env = Object::new();
    Reflect::set(&env, &JsValue::from_str("m"), &wasm_bindgen::memory()).ok()?;
    Reflect::set(
        &env,
        &JsValue::from_str("t"),
        &wasm_bindgen::function_table(),
    )
    .ok()?;
    let imports = Object::new();
    Reflect::set(&imports, &JsValue::from_str("e"), &env).ok()?;

    let instance = WebAssembly::Instance::new(&module, &imports).ok()?;
    let run = Reflect::get(&instance.exports(), &JsValue::from_str("run")).ok()?;
    let run: Function = run.dyn_into().ok()?;

    let table = table()?;
    let index = match FREE.with(|free| free.borrow_mut().pop()) {
        Some(entry) => entry as u32,
        // `grow` answers the old length, which is the new slot's index.
        None => table.grow(1).ok()?,
    };
    // Slot zero is the null function pointer and means "not installed".
    if index == 0 {
        return None;
    }
    table.set(index, &run).ok()?;
    Some(index as Entry)
}

/// Takes a slot back for reuse.
fn release(entry: Entry) {
    FREE.with(|free| free.borrow_mut().push(entry));
}
