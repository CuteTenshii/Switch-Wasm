//! The page's log: finished lines with their levels, handed over one at a time so the
//! page shows them without reading anything out of their text.

use std::collections::VecDeque;

use switch_core::cpu::Cpu;
use switch_core::trace::Level;

use crate::{session_if_live, write_into, SyncCell};

/// A line's level as the page numbers it; 0 from [`switch_log_next`] means no line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum PageLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    /// Guest output with no level of its own.
    Log = 4,
    Debug = 5,
}

impl From<Level> for PageLevel {
    fn from(level: Level) -> Self {
        match level {
            Level::Error => PageLevel::Error,
            Level::Warn => PageLevel::Warn,
            Level::Info => PageLevel::Info,
            Level::Debug => PageLevel::Debug,
        }
    }
}

struct Queue {
    lines: VecDeque<(PageLevel, String)>,
    /// The line [`switch_log_next`] took last, for [`switch_log_line`].
    current: String,
    /// The level unmarked trace lines continue, kept across collections.
    trace_level: Level,
}

/// Global rather than per session, so lines outlive the session that wrote them.
static QUEUE: SyncCell<Queue> = SyncCell::new(Queue {
    lines: VecDeque::new(),
    current: String::new(),
    trace_level: Level::Debug,
});

fn queue() -> &'static mut Queue {
    // SAFETY: only the system worker calls exports; see the `SESSIONS` comment.
    unsafe { &mut *QUEUE.get() }
}

/// Queue `text` at `level`, one entry per line.
pub(crate) fn push(level: PageLevel, text: &str) {
    let text = text.strip_suffix('\n').unwrap_or(text);
    for line in text.split('\n') {
        queue().lines.push_back((level, line.to_owned()));
    }
}

/// Queue trace text, whose lines open with a [`Level::marker`] byte or continue the last.
fn push_marked(bytes: &[u8]) {
    let bytes = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    for line in bytes.split(|&b| b == b'\n') {
        let marked = [Level::Error, Level::Warn, Level::Info, Level::Debug]
            .into_iter()
            .find(|level| line.first() == Some(&level.marker()));
        let line = match marked {
            Some(level) => {
                queue().trace_level = level;
                &line[1..]
            }
            None => line,
        };
        let level = queue().trace_level.into();
        queue()
            .lines
            .push_back((level, String::from_utf8_lossy(line).into_owned()));
    }
}

/// The level `lm` writes into a guest line's `[lm/LEVEL]` tag; untagged output has none.
fn guest_level(line: &str) -> PageLevel {
    let Some(tag) = line.strip_prefix("[lm/") else {
        return PageLevel::Log;
    };
    let level = tag.split(['/', ']']).next().unwrap_or("");
    match level {
        "FATAL" | "ERROR" => PageLevel::Error,
        "WARN" => PageLevel::Warn,
        "INFO" => PageLevel::Info,
        "TRACE" => PageLevel::Debug,
        _ => PageLevel::Log,
    }
}

/// Move the session's guest output and diagnostics into the queue, in that order.
pub(crate) fn collect(cpu: &mut Cpu) {
    if !cpu.out.is_empty() {
        let out = std::mem::take(&mut cpu.out);
        let text = String::from_utf8_lossy(&out);
        let text = text.strip_suffix('\n').unwrap_or(&text);
        for line in text.split('\n') {
            queue()
                .lines
                .push_back((guest_level(line), line.to_owned()));
        }
    }
    cpu.absorb_traces();
    if !cpu.trace.is_empty() {
        push_marked(&std::mem::take(&mut cpu.trace));
    }
}

/// Take the next line, collecting from session `handle` first when it is live (the
/// diagnostics of code with no session are collected either way). Returns its level,
/// or 0 when there is none; [`switch_log_line`] has its text.
#[no_mangle]
pub extern "C" fn switch_log_next(handle: i32) -> u32 {
    match u32::try_from(handle).ok().and_then(session_if_live) {
        Some(s) => collect(&mut s.cpu),
        None => {
            let pending = switch_core::trace::take_pending();
            if !pending.is_empty() {
                push_marked(&pending);
            }
        }
    }
    let q = queue();
    match q.lines.pop_front() {
        Some((level, line)) => {
            q.current = line;
            level as u32
        }
        None => {
            q.current.clear();
            0
        }
    }
}

/// Copy the line [`switch_log_next`] took into `buf`. Returns its whole length in
/// bytes, which can exceed `maxlen`, so the page can size the buffer from a first call.
#[no_mangle]
pub extern "C" fn switch_log_line(buf: *mut u8, maxlen: u32) -> u32 {
    let line = &queue().current;
    write_into(buf, maxlen, line.as_bytes());
    line.len() as u32
}
