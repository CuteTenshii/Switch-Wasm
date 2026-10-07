//! Opening a GPU backend from the browser. Device requests are promises, so this
//! is `async`, driven by `wasm-bindgen-futures`, and hands the device to
//! [`switch_gpu::Gpu::with_device`].

use wasm_bindgen::prelude::wasm_bindgen;

/// The prefix the worker matches on; the whole message when the adapter is unnamed.
const RENDERING_ON: &str = "rendering on";

/// Open a device and install the backend on session `handle`'s 3D channel,
/// returning a message. `device_msaa` and `interleave` are the browser's
/// `GPU_DEVICE_MSAA` and `GPU_INTERLEAVE`.
#[wasm_bindgen]
pub async fn switch_gpu_open(handle: u32, device_msaa: bool, interleave: bool) -> String {
    // Check the channel before opening: a dropped device frees nothing on the web backend.
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
    // Request the optional features that expose compressed texture formats.
    let (device, queue) = match adapter
        .request_device(&switch_gpu::device_descriptor(&adapter))
        .await
    {
        Ok(pair) => pair,
        Err(e) => return format!("no device: {e}"),
    };
    // Empty on some browsers; the worker names those.
    let name = adapter.get_info().name;
    // Hand over the instance and adapter; see `switch_gpu::Gpu::_instance`.
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
