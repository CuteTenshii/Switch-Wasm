//! The page's canvas, handed to the worker: each new frame is painted here as the run
//! loop sees it, with no trip through the page. With a device that can draw on the canvas,
//! frames go there from the device; otherwise they are painted from the CPU's framebuffer.

use std::cell::RefCell;

use js_sys::Uint8ClampedArray;
use wasm_bindgen::prelude::*;
use web_sys::{ImageData, OffscreenCanvas, OffscreenCanvasRenderingContext2d};

use crate::session;

/// A canvas takes one kind of context for good, so this is decided at the first frame.
enum Mode {
    /// No frame has been presented since the canvas arrived, or the device is still opening.
    Undecided,
    /// The device owns the canvas; see [`switch_core::gpu::renderer::Renderer::has_screen`].
    Device,
    Cpu {
        context: OffscreenCanvasRenderingContext2d,
        /// Reused between frames: the opaque pixels, and the page-side copy `ImageData` wraps.
        rgba: Vec<u8>,
        image: Option<(Uint8ClampedArray, ImageData)>,
    },
}

struct Screen {
    canvas: OffscreenCanvas,
    mode: Mode,
    /// The frame painted last, so an unchanged one is not painted again.
    painted: u64,
}

thread_local! {
    // The canvas outlives sessions: the page hands it over once.
    static SCREEN: RefCell<Option<Screen>> = const { RefCell::new(None) };
}

/// The canvas, for the device to draw on, unless the CPU already paints it.
pub(crate) fn canvas_for_device() -> Option<OffscreenCanvas> {
    SCREEN.with_borrow(|screen| {
        screen
            .as_ref()
            .filter(|screen| !matches!(screen.mode, Mode::Cpu { .. }))
            .map(|screen| screen.canvas.clone())
    })
}

/// Paint frames on `canvas` from now on.
#[wasm_bindgen]
pub fn switch_attach_screen(canvas: OffscreenCanvas) -> Result<(), JsValue> {
    SCREEN.with_borrow_mut(|screen| {
        *screen = Some(Screen {
            canvas,
            mode: Mode::Undecided,
            painted: 0,
        })
    });
    Ok(())
}

fn context_2d(canvas: &OffscreenCanvas) -> Result<OffscreenCanvasRenderingContext2d, JsValue> {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"alpha".into(), &false.into())?;
    Ok(canvas
        .get_context_with_context_options("2d", &options)?
        .ok_or("this browser has no 2d context for an OffscreenCanvas")?
        .dyn_into::<OffscreenCanvasRenderingContext2d>()?)
}

/// Paint the session's newest frame if it is one the screen has not shown. Returns
/// whether it painted.
#[wasm_bindgen]
pub fn switch_present() -> Result<bool, JsValue> {
    let gpu = &mut session().cpu.nv.gpu;
    SCREEN.with_borrow_mut(|screen| {
        let Some(screen) = screen else {
            return Ok(false);
        };
        if gpu.frames == 0 || gpu.frames == screen.painted {
            return Ok(false);
        }
        if matches!(screen.mode, Mode::Undecided) {
            if crate::gpu::opening() {
                return Ok(false);
            }
            screen.mode = if gpu.device_has_screen() {
                Mode::Device
            } else {
                Mode::Cpu {
                    context: context_2d(&screen.canvas)?,
                    rgba: Vec::new(),
                    image: None,
                }
            };
        }
        match &mut screen.mode {
            Mode::Undecided => Ok(false),
            Mode::Device => {
                // A frame the device presented itself has no pixels on the CPU.
                if !gpu.framebuffer.pixels.is_empty() && !gpu.show_on_device() {
                    return Ok(false);
                }
                screen.painted = gpu.frames;
                Ok(true)
            }
            Mode::Cpu {
                context,
                rgba,
                image,
            } => {
                let fb = &gpu.framebuffer;
                if screen.canvas.width() != fb.width || screen.canvas.height() != fb.height {
                    screen.canvas.set_width(fb.width);
                    screen.canvas.set_height(fb.height);
                    *image = None;
                }
                // Scan-out ignores alpha.
                rgba.clear();
                rgba.extend(
                    fb.pixels
                        .iter()
                        .flat_map(|p| (p | 0xFF00_0000).to_le_bytes()),
                );
                let (pixels, image) = match image {
                    Some(image) => image,
                    None => {
                        let pixels = Uint8ClampedArray::new_with_length(rgba.len() as u32);
                        let data = ImageData::new_with_js_u8_clamped_array_and_sh(
                            &pixels, fb.width, fb.height,
                        )?;
                        image.insert((pixels, data))
                    }
                };
                pixels.copy_from(rgba);
                context.put_image_data(image, 0.0, 0.0)?;
                screen.painted = gpu.frames;
                Ok(true)
            }
        }
    })
}

/// Blank the screen, as for a session that has not presented yet.
#[wasm_bindgen]
pub fn switch_clear_screen() {
    SCREEN.with_borrow_mut(|screen| {
        let Some(screen) = screen else {
            return;
        };
        match &screen.mode {
            Mode::Undecided => return,
            Mode::Device => {
                session().cpu.nv.gpu.blank_device_screen();
            }
            Mode::Cpu { context, .. } => {
                let (width, height) = (screen.canvas.width(), screen.canvas.height());
                context.clear_rect(0.0, 0.0, f64::from(width), f64::from(height));
            }
        }
        screen.painted = 0;
    })
}
