//! Making the core's emitted wasm callable.
//!
//! `switch-core` writes a translated block out as a wasm module but has no way
//! to compile one: it has no dependencies, so it cannot reach
//! `WebAssembly.Module`. It asks for the two operations it cannot do through
//! [`switch_core::cpu::JitHost`], and this is where they are done.
//!
//! # The table is the whole point
//!
//! An emitted block goes into the **function table of this very module**, and
//! the core calls it by index. On `wasm32` a Rust function pointer is exactly
//! such an index, so entering an emitted block is one `call_indirect`.
//!
//! The obvious shape, an import the core calls through, would put a JS frame
//! between the interpreter and every block it enters, and a retail frame
//! enters a block every six guest instructions. The boundary is crossed once
//! per block, when it is installed, and never while it runs.
//!
//! `wasm_bindgen::function_table` and `wasm_bindgen::memory` are that table
//! and the memory the core's guest state lives in; an emitted module imports
//! the memory as `e`.`m`, which is what lets a block reach a guest register
//! with a plain `i64.load` instead of a call.
//!
//! # Where this runs
//!
//! `WebAssembly.Module` here is the synchronous constructor, because the core
//! asks for a block from inside a run slice and cannot wait for a promise. On
//! a window that would refuse anything over 4 KiB; the emulator runs in a
//! worker, where it is allowed at any size, and an emitted block is a few
//! hundred bytes in any case.

use js_sys::{Function, Object, Reflect, Uint8Array, WebAssembly};
use std::cell::RefCell;
use switch_core::cpu::{set_jit_host, Entry, JitHost};
use wasm_bindgen::{JsCast, JsValue};

thread_local! {
    /// Table slots whose blocks have been dropped, waiting to be used again.
    ///
    /// Without this the table would grow by a slot per block the guest ever
    /// translated. The cache drops blocks constantly — every store that lands
    /// on translated code, and every rotation — so on a retail title that is
    /// six figures of compiled code nothing can reach, held for the life of
    /// the page.
    ///
    /// A freed slot keeps pointing at its old function until something takes
    /// it, because the table has no way to write an empty one: the entry type
    /// is a function, and there is no null to put there. So the instance
    /// behind a freed slot outlives its block and is collected when the slot
    /// is reused, which bounds what is held by the most blocks ever installed
    /// at once rather than by how many there have been.
    static FREE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

/// Let the core emit blocks, if it has not been told this already.
///
/// Called before a session runs rather than when the module loads, so that a
/// build which never creates one never touches `WebAssembly` at all.
pub fn attach() {
    set_jit_host(JitHost { install, release });
}

/// The function table this module makes its indirect calls through, which is
/// what an installed block's index is an index into.
fn table() -> Option<WebAssembly::Table> {
    wasm_bindgen::function_table().dyn_into().ok()
}

/// Compile `code` and answer the table slot its `run` went to, or zero.
///
/// The slot number is what the core calls an [`Entry`]: a function pointer on
/// `wasm32` is an index into this table, so the core can call the slot
/// directly and the number needs no translating on the way.
///
/// Every failure answers zero, and the core takes that as "this block is
/// interpreted from now on". There is nothing else to do with one: a module
/// that would not compile now will not compile later, and running the block on
/// the interpreter is what the emulator did before any of this existed.
fn install(code: &[u8]) -> Entry {
    compile(code).unwrap_or(0)
}

fn compile(code: &[u8]) -> Option<Entry> {
    let bytes = Uint8Array::new_with_length(code.len() as u32);
    bytes.copy_from(code);
    let module = WebAssembly::Module::new(bytes.as_ref()).ok()?;

    // The one import: this module's own linear memory, under the name the
    // emitter writes. Guest state is in it already, so there is nothing to
    // copy in or out.
    let env = Object::new();
    Reflect::set(&env, &JsValue::from_str("m"), &wasm_bindgen::memory()).ok()?;
    let imports = Object::new();
    Reflect::set(&imports, &JsValue::from_str("e"), &env).ok()?;

    let instance = WebAssembly::Instance::new(&module, &imports).ok()?;
    let run = Reflect::get(&instance.exports(), &JsValue::from_str("run")).ok()?;
    let run: Function = run.dyn_into().ok()?;

    let table = table()?;
    let index = match FREE.with(|free| free.borrow_mut().pop()) {
        Some(entry) => entry as u32,
        // `grow` answers the length the table had, which is the index of the
        // slot it just added.
        None => table.grow(1).ok()?,
    };
    // Zero is how the core says "not installed", and it is also the slot the
    // linker keeps empty so that calling a null function pointer traps. It
    // cannot be handed out, but nothing here would notice if it were, so this
    // is where that is made sure of.
    if index == 0 {
        return None;
    }
    table.set(index, &run).ok()?;
    Some(index as Entry)
}

/// Take a slot back, because the block that held it is gone.
fn release(entry: Entry) {
    FREE.with(|free| free.borrow_mut().push(entry));
}
