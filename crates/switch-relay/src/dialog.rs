//! An Allow / Deny question in a window on the relay's desktop, which no page can answer.

/// The window's title.
const TITLE: &str = "switch-relay";

/// Show `text` and wait for the answer: `None` when no window could be shown at all. Blocks.
pub fn ask(text: &str) -> Option<bool> {
    platform::ask(text)
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use std::io::ErrorKind;
    use std::process::Command;

    use super::TITLE;

    pub fn ask(text: &str) -> Option<bool> {
        let desktop = ["DISPLAY", "WAYLAND_DISPLAY"]
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()));
        if !desktop {
            return None;
        }
        let zenity = Command::new("zenity")
            .args([
                "--question",
                "--no-markup",
                "--default-cancel",
                "--icon=dialog-warning",
            ])
            .args([
                "--title",
                TITLE,
                "--ok-label",
                "Allow",
                "--cancel-label",
                "Deny",
            ])
            .args(["--text", text])
            .status();
        let status = match zenity {
            Err(error) if error.kind() == ErrorKind::NotFound => Command::new("kdialog")
                .args([
                    "--title",
                    TITLE,
                    "--yes-label",
                    "Allow",
                    "--no-label",
                    "Deny",
                ])
                .args(["--warningyesno", text])
                .status(),
            other => other,
        };
        // Both say yes with 0. Any other exit, a closed window or a tool that failed, is no.
        status.ok().map(|status| status.success())
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::process::Command;

    use super::TITLE;

    /// The text goes in as an argument, so nothing in it is read as AppleScript.
    const SCRIPT: [&str; 3] = [
        "on run argv",
        "display dialog (item 1 of argv) with title (item 2 of argv) buttons {\"Deny\", \"Allow\"} default button \"Deny\" cancel button \"Deny\" with icon caution giving up after 120",
        "end run",
    ];

    pub fn ask(text: &str) -> Option<bool> {
        let mut command = Command::new("osascript");
        for line in SCRIPT {
            command.args(["-e", line]);
        }
        let output = command.args([text, TITLE]).output().ok()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Some(stdout.contains("button returned:Allow"));
        }
        // -128 is the cancel button; anything else means no dialog was shown.
        String::from_utf8_lossy(&output.stderr)
            .contains("(-128)")
            .then_some(false)
    }
}

#[cfg(windows)]
mod platform {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDNO, IDYES, MB_DEFBUTTON2, MB_ICONWARNING, MB_SETFOREGROUND, MB_TOPMOST,
        MB_YESNO,
    };

    use super::TITLE;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    pub fn ask(text: &str) -> Option<bool> {
        let (text, title) = (wide(text), wide(TITLE));
        let style = MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_TOPMOST | MB_SETFOREGROUND;
        // SAFETY: both strings are NUL-terminated and outlive the call; no owner window.
        let answer =
            unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), style) };
        match answer {
            IDYES => Some(true),
            IDNO => Some(false),
            _ => None,
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    pub fn ask(_text: &str) -> Option<bool> {
        None
    }
}
