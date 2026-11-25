//! An Allow / Deny question in a window on the relay's desktop, which no page can answer.

use std::time::Duration;

/// The window's title.
const TITLE: &str = "switch-relay";

/// What came of a question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shown {
    Answered(bool),
    /// The window was up for the whole timeout and is gone again.
    TimedOut,
}

/// Show `text` for up to `timeout` and wait for the answer: `None` when no window could be
/// shown at all. Blocks.
pub fn ask(text: &str, timeout: Duration) -> Option<Shown> {
    platform::ask(text, timeout)
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use std::io::ErrorKind;
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};

    use super::{Shown, TITLE};

    /// How often a waiting question checks whether its window was answered.
    const POLL: Duration = Duration::from_millis(100);

    pub fn ask(text: &str, timeout: Duration) -> Option<Shown> {
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
            .spawn();
        let child = match zenity {
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
                .spawn(),
            other => other,
        };
        wait(child.ok()?, timeout)
    }

    /// Both tools say yes with 0. Any other exit, a closed window or a tool that failed, is no.
    fn wait(mut child: Child, timeout: Duration) -> Option<Shown> {
        let deadline = Instant::now() + timeout;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Some(Shown::Answered(status.success())),
                Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL),
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Some(Shown::TimedOut);
                }
                Err(_) => return Some(Shown::Answered(false)),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn run(program: &str, args: &[&str], timeout: Duration) -> Option<Shown> {
            wait(Command::new(program).args(args).spawn().unwrap(), timeout)
        }

        #[test]
        fn the_exit_status_is_the_answer() {
            let long = Duration::from_secs(5);
            assert_eq!(run("true", &[], long), Some(Shown::Answered(true)));
            assert_eq!(run("false", &[], long), Some(Shown::Answered(false)));
        }

        #[test]
        fn a_window_left_up_is_closed_at_the_timeout() {
            let started = Instant::now();
            let shown = run("sleep", &["30"], Duration::from_millis(300));
            assert_eq!(shown, Some(Shown::TimedOut));
            assert!(started.elapsed() < Duration::from_secs(5));
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::process::Command;
    use std::time::Duration;

    use super::{Shown, TITLE};

    /// The text goes in as an argument, so nothing in it is read as AppleScript.
    const SCRIPT: [&str; 3] = [
        "on run argv",
        "display dialog (item 1 of argv) with title (item 2 of argv) buttons {\"Deny\", \"Allow\"} default button \"Deny\" cancel button \"Deny\" with icon caution giving up after (item 3 of argv as integer)",
        "end run",
    ];

    pub fn ask(text: &str, timeout: Duration) -> Option<Shown> {
        let mut command = Command::new("osascript");
        for line in SCRIPT {
            command.args(["-e", line]);
        }
        let seconds = timeout.as_secs().max(1).to_string();
        let output = command.args([text, TITLE, &seconds]).output().ok()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("gave up:true") {
                return Some(Shown::TimedOut);
            }
            return Some(Shown::Answered(stdout.contains("button returned:Allow")));
        }
        // -128 is the cancel button; anything else means no dialog was shown.
        String::from_utf8_lossy(&output.stderr)
            .contains("(-128)")
            .then_some(Shown::Answered(false))
    }
}

#[cfg(windows)]
mod platform {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::sync::Arc;
    use std::time::Duration;

    use windows_sys::core::BOOL;
    use windows_sys::Win32::Foundation::{HWND, LPARAM};
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumThreadWindows, MessageBoxW, PostMessageW, IDNO, IDYES, MB_DEFBUTTON2, MB_ICONWARNING,
        MB_SETFOREGROUND, MB_TOPMOST, MB_YESNO, WM_COMMAND,
    };

    use super::{Shown, TITLE};

    /// How often a box that outlived its timeout is told again to close, in case the first
    /// came before its window existed.
    const RETRY: Duration = Duration::from_millis(100);

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    /// Click Deny on a window of the box's thread. A Yes/No box ignores `WM_CLOSE`.
    unsafe extern "system" fn deny(window: HWND, _: LPARAM) -> BOOL {
        // SAFETY: posting needs no access to the window's memory; a stale handle just fails.
        unsafe { PostMessageW(window, WM_COMMAND, IDNO as usize, 0) };
        1
    }

    pub fn ask(text: &str, timeout: Duration) -> Option<Shown> {
        let (text, title) = (wide(text), wide(TITLE));
        let style = MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_TOPMOST | MB_SETFOREGROUND;
        // SAFETY: no arguments, no preconditions.
        let thread = unsafe { GetCurrentThreadId() };
        let timed_out = Arc::new(AtomicBool::new(false));
        let (answered, waiting) = channel::<()>();
        let closer = std::thread::spawn({
            let timed_out = timed_out.clone();
            move || {
                if waiting.recv_timeout(timeout) != Err(RecvTimeoutError::Timeout) {
                    return;
                }
                timed_out.store(true, Relaxed);
                while waiting.recv_timeout(RETRY) == Err(RecvTimeoutError::Timeout) {
                    // SAFETY: `deny` matches WNDENUMPROC and keeps nothing past the call.
                    unsafe { EnumThreadWindows(thread, Some(deny), 0) };
                }
            }
        });
        // SAFETY: both strings are NUL-terminated and outlive the call; no owner window.
        let answer =
            unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), style) };
        drop(answered);
        let _ = closer.join();
        match answer {
            IDNO if timed_out.load(Relaxed) => Some(Shown::TimedOut),
            IDYES => Some(Shown::Answered(true)),
            IDNO => Some(Shown::Answered(false)),
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use std::time::Instant;

        use super::*;

        #[test]
        fn a_box_left_up_is_denied_at_the_timeout() {
            let started = Instant::now();
            let shown = ask("switch-relay test", Duration::from_millis(500));
            assert_eq!(shown, Some(Shown::TimedOut));
            assert!(started.elapsed() < Duration::from_secs(10));
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    use std::time::Duration;

    use super::Shown;

    pub fn ask(_text: &str, _timeout: Duration) -> Option<Shown> {
        None
    }
}
