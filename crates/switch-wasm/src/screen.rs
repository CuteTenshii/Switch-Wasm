//! The page's canvas, handed to the worker: each new frame is painted here as the run
//! loop sees it, with no trip through the page.

use std::cell::RefCell;

use js_sys::Uint8ClampedArray;
use wasm_bindgen::prelude::*;
use web_sys::{ImageData, OffscreenCanvas, OffscreenCanvasRenderingContext2d};

use crate::session;

struct Screen {
    canvas: OffscreenCanvas,
    context: OffscreenCanvasRenderingContext2d,
    /// The frame painted last, so an unchanged one is not painted again.
    painted: u64,
    /// Reused between frames: the opaque pixels, and the page-side copy `ImageData` wraps.
    rgba: Vec<u8>,
    image: Option<(Uint8ClampedArray, ImageData)>,
}

thread_local! {
    // The canvas outlives sessions: the page hands it over once.
    static SCREEN: RefCell<Option<Screen>> = const { RefCell::new(None) };
}

/// Paint frames on `canvas` from now on.
#[wasm_bindgen]
pub fn switch_attach_screen(canvas: OffscreenCanvas) -> Result<(), JsValue> {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"alpha".into(), &false.into())?;
    let context = canvas
        .get_context_with_context_options("2d", &options)?
        .ok_or("this browser has no 2d context for an OffscreenCanvas")?
        .dyn_into::<OffscreenCanvasRenderingContext2d>()?;
    SCREEN.with_borrow_mut(|screen| {
        *screen = Some(Screen {
            canvas,
            context,
            painted: 0,
            rgba: Vec::new(),
            image: None,
        })
    });
    Ok(())
}

/// Paint the session's newest frame if it is one the screen has not shown. Returns
/// whether it painted.
#[wasm_bindgen]
pub fn switch_present(handle: u32) -> Result<bool, JsValue> {
    let gpu = &session(handle).cpu.nv.gpu;
    SCREEN.with_borrow_mut(|screen| {
        let Some(screen) = screen else {
            return Ok(false);
        };
        if gpu.frames == 0 || gpu.frames == screen.painted {
            return Ok(false);
        }
        let fb = &gpu.framebuffer;
        if screen.canvas.width() != fb.width || screen.canvas.height() != fb.height {
            screen.canvas.set_width(fb.width);
            screen.canvas.set_height(fb.height);
            screen.image = None;
        }
        // Scan-out ignores alpha.
        screen.rgba.clear();
        screen.rgba.extend(
            fb.pixels
                .iter()
                .flat_map(|p| (p | 0xFF00_0000).to_le_bytes()),
        );
        let (pixels, image) = match &screen.image {
            Some(image) => image,
            None => {
                let pixels = Uint8ClampedArray::new_with_length(screen.rgba.len() as u32);
                let image =
                    ImageData::new_with_js_u8_clamped_array_and_sh(&pixels, fb.width, fb.height)?;
                screen.image.insert((pixels, image))
            }
        };
        pixels.copy_from(&screen.rgba);
        screen.context.put_image_data(image, 0.0, 0.0)?;
        screen.painted = gpu.frames;
        Ok(true)
    })
}

/// Blank the screen, as for a session that has not presented yet.
#[wasm_bindgen]
pub fn switch_clear_screen() {
    SCREEN.with_borrow_mut(|screen| {
        if let Some(screen) = screen {
            let (width, height) = (screen.canvas.width(), screen.canvas.height());
            screen
                .context
                .clear_rect(0.0, 0.0, f64::from(width), f64::from(height));
            screen.painted = 0;
        }
    })
}
