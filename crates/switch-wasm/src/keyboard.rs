//! The library applets the page shows: the software keyboard, which the page
//! answers, and the error applet, which it only displays.

use crate::storage::sd_path;
use crate::{session, write_into};
use switch_core::cpu::{ErrorDialog, KeyboardRequest};
use switch_core::json;

/// The next error applet as JSON, `{"code":"2002-1002","message":""}` with `code` null when
/// there is none, taken from the queue. Writes nothing when none is waiting. A message is at
/// most 2 KiB, so 16 KiB holds any escaping.
#[no_mangle]
pub extern "C" fn switch_error_dialog_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let Some(dialog) = session(handle).cpu.take_error_dialog() else {
        return 0;
    };
    write_into(buf, maxlen, error_dialog_json(&dialog).as_bytes())
}

fn error_dialog_json(dialog: &ErrorDialog) -> String {
    json::object(|o| {
        o.field("code", dialog.code.as_deref())
            .field("message", &dialog.message);
    })
}

/// The waiting keyboard as JSON: `{"header":"","sub":"","guide":"","ok":"","maxLength":500,
/// "minLength":0,"password":false}`. Writes nothing when no keyboard is waiting.
#[no_mangle]
pub extern "C" fn switch_keyboard_json(handle: u32, buf: *mut u8, maxlen: u32) -> u32 {
    let Some(request) = session(handle).cpu.keyboard_request() else {
        return 0;
    };
    write_into(buf, maxlen, keyboard_json(request).as_bytes())
}

fn keyboard_json(request: &KeyboardRequest) -> String {
    json::object(|o| {
        o.field("header", &request.header)
            .field("sub", &request.sub)
            .field("guide", &request.guide)
            .field("ok", &request.ok)
            .field("maxLength", request.max_length)
            .field("minLength", request.min_length)
            .field("password", request.password);
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dialog_is_the_json_the_page_reads() {
        let dialog = ErrorDialog {
            code: Some("2002-1002".into()),
            message: "Line one\n\"quoted\"".into(),
        };
        assert_eq!(
            error_dialog_json(&dialog),
            r#"{"code":"2002-1002","message":"Line one\n\"quoted\""}"#
        );
        let uncoded = ErrorDialog {
            code: None,
            message: String::new(),
        };
        assert_eq!(error_dialog_json(&uncoded), r#"{"code":null,"message":""}"#);
    }

    #[test]
    fn a_keyboard_is_the_json_the_page_reads() {
        let request = KeyboardRequest {
            header: "Name".into(),
            sub: "Your \"nickname\"".into(),
            guide: String::new(),
            ok: "OK".into(),
            max_length: 10,
            min_length: 1,
            password: true,
        };
        assert_eq!(
            keyboard_json(&request),
            concat!(
                r#"{"header":"Name","sub":"Your \"nickname\"","guide":"","ok":"OK","#,
                r#""maxLength":10,"minLength":1,"password":true}"#
            )
        );
    }
}
