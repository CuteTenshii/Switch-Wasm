//! Opening a GPU backend from the browser. Device requests are promises, so this
//! is `async`, driven by `wasm-bindgen-futures`, and hands the device to
//! [`switch_gpu::Gpu::with_device`].

use std::cell::Cell;

use wasm_bindgen::prelude::wasm_bindgen;

thread_local! {
    static OPENING: Cell<bool> = const { Cell::new(false) };
}

/// Whether a device is being opened, which may yet take the canvas.
#[cfg(target_arch = "wasm32")]
pub(crate) fn opening() -> bool {
    OPENING.get()
}

/// Marks a device as being opened until dropped.
struct Opening;

impl Opening {
    fn begin() -> Self {
        OPENING.set(true);
        Opening
    }
}

impl Drop for Opening {
    fn drop(&mut self) {
        OPENING.set(false);
    }
}

/// What the worker matches on.
const INSTALLED: &str = "installed";

/// Open a device and install the backend on the session's `Gpu`,
/// returning a message. `device_msaa` and `interleave` are the browser's
/// `--device-msaa` and `--interleave`.
#[wasm_bindgen]
pub async fn switch_gpu_open(device_msaa: bool, interleave: bool) -> String {
    let _opening = Opening::begin();
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
    // After the device: a canvas that takes the device's context can no longer be painted from the CPU.
    #[cfg(target_arch = "wasm32")]
    let screen = crate::screen::canvas_for_device().and_then(|canvas| {
        instance
            .create_surface(switch_gpu::wgpu::SurfaceTarget::OffscreenCanvas(canvas))
            .map_err(|e| {
                switch_core::trace::emit_at(
                    switch_core::trace::Level::Info,
                    &format!("[gpu] the canvas stays with the CPU: {e}"),
                )
            })
            .ok()
    });
    // Hand over the instance and adapter; see `switch_gpu::Gpu::_instance`.
    let mut gpu = switch_gpu::Gpu::with_device(instance, adapter, device, queue);
    #[cfg(target_arch = "wasm32")]
    if let Some(surface) = screen {
        gpu.set_screen(surface);
    }
    gpu.set_device_msaa(device_msaa);
    gpu.set_interleave(interleave);
    crate::install_gpu(gpu);
    INSTALLED.to_string()
}
