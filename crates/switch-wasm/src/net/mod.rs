//! The guest's network, carried by a relay over a WebSocket; see `docs/relay.md`.

// Only the browser build links it; host builds compile it for the tests.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod client;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod settings;
#[cfg(target_arch = "wasm32")]
mod socket;
mod tls;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod watchdog;

#[cfg(target_arch = "wasm32")]
pub(crate) use socket::exchange;

/// Without a browser there is no link.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn exchange(_cpu: &mut switch_core::cpu::Cpu) {}
