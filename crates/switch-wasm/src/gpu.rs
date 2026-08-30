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

/// The prefix the worker matches on, and the whole message on its own when the
/// browser declines to name the adapter -- see `name` in [`switch_gpu_open`].
const RENDERING_ON: &str = "rendering on";

/// Open a device and install the backend on session `handle`'s 3D channel.
///
/// Answers a message rather than a bool: a machine without WebGPU is a normal
/// thing, and the answer to it is the software rasterizer, which is what ran
/// before this existed.
///
/// `device_msaa` is the browser's spelling of the backend's
/// `GPU_DEVICE_MSAA`, which a wasm build has no environment to read: it lets
/// the device do the multisampling where WebGPU offers the sample count,
/// which is four and only four. That shades once per pixel instead of once
/// per sample, and anti-aliases every edge differently from the rasterizer —
/// see `switch_gpu::Gpu::route` for the trade.
/// `interleave` is the browser's spelling of `GPU_INTERLEAVE`: keep handing
/// single fallback draws to the rasterizer inside a device frame, rather than
/// giving the frame after one to the rasterizer whole. A browser's readback
/// lands after the call that asked for it, which is what makes the difference
/// — see `switch_gpu::Gpu::interleave` for the measured trade.
#[wasm_bindgen]
pub async fn switch_gpu_open(handle: u32, device_msaa: bool, interleave: bool) -> String {
    // Before anything is opened, not after. `requestDevice` builds a device in
    // the GPU process whether or not there is a channel to install it on, and
    // one built too early used to be dropped — which on wgpu's web backend
    // frees nothing. See [`crate::gpu_channel_open`].
    if !crate::gpu_channel_open(handle) {
        return crate::NO_CHANNEL_YET.to_string();
    }
    let instance = switch_gpu::wgpu::Instance::new(
        switch_gpu::wgpu::InstanceDescriptor::new_without_display_handle(),
    );
    let adapter = match instance
        .request_adapter(&switch_gpu::wgpu::RequestAdapterOptions::default())
        .await
    {
        Ok(adapter) => adapter,
        Err(e) => return format!("no adapter: {e}"),
    };
    // Not `DeviceDescriptor::default()`: that asks for no optional features,
    // and WebGPU keeps the compressed texture families behind them. A title's
    // textures are block-compressed, so the first one threw inside
    // `createTexture` and wgpu unwrapped it — a panic mid-draw, which on wasm
    // is a bare `unreachable` that stops the core. See
    // `switch_gpu::device_descriptor`.
    let (device, queue) = match adapter
        .request_device(&switch_gpu::device_descriptor(&adapter))
        .await
    {
        Ok(pair) => pair,
        Err(e) => return format!("no device: {e}"),
    };
    // wgpu takes this from `GPUAdapterInfo.description`, which Chrome leaves
    // empty on macOS and Firefox leaves empty always. The worker names those.
    let name = adapter.get_info().name;
    // The instance and the adapter are handed over rather than dropped here:
    // see `switch_gpu::Gpu::_instance` for what a browser does to a device
    // whose instance has no external reference left.
    let mut gpu = switch_gpu::Gpu::with_device(instance, adapter, device, queue);
    gpu.set_device_msaa(device_msaa);
    gpu.set_interleave(interleave);
    crate::install_gpu(handle, gpu);
    if name.is_empty() {
        RENDERING_ON.to_string()
    } else {
        format!("{RENDERING_ON} {name}")
    }
}
