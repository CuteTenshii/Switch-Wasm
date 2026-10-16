//! Guest cores on the core workers: the session lends them threads.

use crate::session;
use std::sync::Arc;
use switch_core::cpu::CoreSlot;

/// Another core for the session; its slot goes to [`switch_serve_core`] on a core worker.
#[no_mangle]
pub extern "C" fn switch_add_core(handle: u32) -> u32 {
    Arc::into_raw(session(handle).cpu.add_core()) as u32
}

/// Run the core `slot` names on this worker until its session stops it. Blocks.
#[no_mangle]
pub extern "C" fn switch_serve_core(slot: u32) {
    crate::install_panic_hook();
    // SAFETY: `slot` came from `switch_add_core`, which gave up one count for this call.
    let slot = unsafe { Arc::from_raw(slot as *const CoreSlot) };
    switch_core::cpu::serve(slot);
}
