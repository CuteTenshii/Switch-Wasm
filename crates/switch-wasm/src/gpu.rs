//! Opening a GPU backend from the browser.
//!
//! The device cannot be opened from Rust the way it is natively.
//! `requestAdapter` and `requestDevice` are promises, and nothing in the
//! emulator may wait on one: the event loop that would resolve it is the same
//! one blocked by waiting. So this is `async`, driven by the browser through
//! `wasm-bindgen-futures`, and hands the finished device to
//! [`switch_gpu::Gpu::with_device`] — which exists for exactly this.
//!
//! Reaching WebGPU at all means `wasm-bindgen`, because wgpu's web backend is
//! the WebGPU JS API called through generated glue. That is the whole cost of
//! the `gpu` feature, and why it is a feature.

use wasm_bindgen::prelude::wasm_bindgen;

/// Open a device and install the backend on session `handle`'s 3D channel.
///
/// Answers a message rather than a bool: a machine without WebGPU is a normal
/// thing, and the answer to it is the software rasterizer, which is what ran
/// before this existed.
#[wasm_bindgen]
pub async fn switch_gpu_open(handle: u32) -> String {
    let instance = switch_gpu::wgpu::Instance::new(switch_gpu::wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match instance.request_adapter(&switch_gpu::wgpu::RequestAdapterOptions::default()).await {
        Ok(adapter) => adapter,
        Err(e) => return format!("no adapter: {e}"),
    };
    // Not `DeviceDescriptor::default()`: that asks for no optional features,
    // and WebGPU keeps the compressed texture families behind them. A title's
    // textures are block-compressed, so the first one threw inside
    // `createTexture` and wgpu unwrapped it — a panic mid-draw, which on wasm
    // is a bare `unreachable` that stops the core. See
    // `switch_gpu::device_descriptor`.
    let (device, queue) =
        match adapter.request_device(&switch_gpu::device_descriptor(&adapter)).await {
            Ok(pair) => pair,
            Err(e) => return format!("no device: {e}"),
        };
    let name = adapter.get_info().name;
    match crate::install_gpu(handle, switch_gpu::Gpu::with_device(device, queue)) {
        Ok(()) => format!("rendering on {name}"),
        Err(why) => why,
    }
}
