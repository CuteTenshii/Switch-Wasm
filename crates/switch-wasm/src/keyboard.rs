//! The library applets the page shows: the software keyboard, which the page
//! answers, and the error applet, which it only displays.

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use crate::session;
use crate::storage::sd_path;

/// Tell the page about the error applets the guest raised and the keyboard it waits on,
/// each once. Errors are taken from the queue; the keyboard stays until answered.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub(crate) fn announce(cpu: &mut switch_core::cpu::Cpu) {
    use crate::post::post;
    use wasm_bindgen::JsValue;
    while let Some(dialog) = cpu.take_error_dialog() {
        post(
            "error_dialog",
            &[
                (
                    "code",
                    dialog
                        .code
                        .as_deref()
                        .map_or(JsValue::NULL, JsValue::from_str),
                ),
                ("message", JsValue::from_str(&dialog.message)),
            ],
        );
    }
    let waiting = cpu.keyboard_request();
    if let (Some(request), false) = (waiting, KEYBOARD_POSTED.swap(waiting.is_some(), Relaxed)) {
        post(
            "keyboard",
            &[
                ("header", JsValue::from_str(&request.header)),
                ("sub", JsValue::from_str(&request.sub)),
                ("guide", JsValue::from_str(&request.guide)),
                ("ok", JsValue::from_str(&request.ok)),
                ("maxLength", request.max_length.into()),
                ("minLength", request.min_length.into()),
                ("password", request.password.into()),
            ],
        );
    }
}

/// Whether the page has been told about the keyboard that is waiting now.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
static KEYBOARD_POSTED: AtomicBool = AtomicBool::new(false);

/// Answer the waiting keyboard with UTF-8 text, or cancel it when `submitted` is 0.
/// Returns 1 if a keyboard was waiting.
#[no_mangle]
pub extern "C" fn switch_keyboard_answer(
    text_ptr: *const u8,
    text_len: u32,
    submitted: u32,
) -> u32 {
    let text = (submitted != 0).then(|| sd_path(text_ptr, text_len));
    u32::from(session().cpu.answer_keyboard(text.as_deref()))
}
