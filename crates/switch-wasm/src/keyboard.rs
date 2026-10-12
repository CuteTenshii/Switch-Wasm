//! The library applets the page shows: the software keyboard, which the page
//! answers, and the error applet, which it only displays.

use crate::storage::sd_path;
use crate::{json_escape, session, write_into};

/// The next error applet as JSON, `{"code":"2002-1002","message":""}` with `code` null when
/// there is none, taken from the queue. Writes nothing when none is waiting. A message is at
/// most 2 KiB, so 16 KiB holds any escaping.
#[no_mangle]
pub extern "C" fn switch_error_dialog_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let Some(dialog) = session(handle).cpu.take_error_dialog() else {
        return 0;
    };
    let mut out = b"{\"code\":".to_vec();
    match &dialog.code {
        Some(code) => {
            out.push(b'"');
            json_escape(code, &mut out);
            out.push(b'"');
        }
        None => out.extend_from_slice(b"null"),
    }
    out.extend_from_slice(b",\"message\":\"");
    json_escape(&dialog.message, &mut out);
    out.extend_from_slice(b"\"}");
    write_into(buf, maxlen, &out)
}

/// The waiting keyboard as JSON: `{"header":"","sub":"","guide":"","ok":"","maxLength":500,
/// "minLength":0,"password":false}`. Writes nothing when no keyboard is waiting.
#[no_mangle]
pub extern "C" fn switch_keyboard_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let Some(request) = session(handle).cpu.keyboard_request() else {
        return 0;
    };
    let mut out = Vec::new();
    for (index, (name, text)) in [
        ("header", &request.header),
        ("sub", &request.sub),
        ("guide", &request.guide),
        ("ok", &request.ok),
    ]
    .into_iter()
    .enumerate()
    {
        out.extend_from_slice(if index == 0 { b"{\"" } else { b",\"" });
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b"\":\"");
        json_escape(text, &mut out);
        out.push(b'"');
    }
    out.extend_from_slice(
        format!(
            ",\"maxLength\":{},\"minLength\":{},\"password\":{}}}",
            request.max_length, request.min_length, request.password
        )
        .as_bytes(),
    );
    write_into(buf, maxlen, &out)
}

/// Answer the waiting keyboard with UTF-8 text, or cancel it when `submitted` is 0.
/// Returns 1 if a keyboard was waiting.
#[no_mangle]
pub extern "C" fn switch_keyboard_answer(
    handle: u32,
    text_ptr: *const u8,
    text_len: u32,
    submitted: u32,
) -> u32 {
    let text = (submitted != 0).then(|| sd_path(text_ptr, text_len));
    u32::from(session(handle).cpu.answer_keyboard(text.as_deref()))
}
