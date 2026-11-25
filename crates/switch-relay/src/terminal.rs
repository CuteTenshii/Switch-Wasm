//! Questions on the relay's terminal.

use std::io::BufRead;
use std::sync::OnceLock;
use std::time::Duration;

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tokio::sync::Mutex;

/// Lines typed on the terminal, from one reader for the whole run: a reader per question would
/// outlive a question that timed out and take the next one's answer.
fn lines() -> &'static Mutex<UnboundedReceiver<String>> {
    static LINES: OnceLock<Mutex<UnboundedReceiver<String>>> = OnceLock::new();
    LINES.get_or_init(|| {
        let (sender, receiver) = unbounded_channel();
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { return };
                if sender.send(line).is_err() {
                    return;
                }
            }
        });
        Mutex::new(receiver)
    })
}

/// Ask `question` with `[y/N]` and wait up to `timeout` for a line: `Some(true)` for yes,
/// `Some(false)` for anything else or a terminal that closed, `None` when nobody answered.
/// Holding the lines makes questions take turns, so none reads another's answer.
pub async fn ask(question: &str, timeout: Duration) -> Option<bool> {
    let mut lines = lines().lock().await;
    // Lines typed while nothing was asked answer nothing.
    while lines.try_recv().is_ok() {}
    eprint!("{question} [y/N] ");
    match tokio::time::timeout(timeout, lines.recv()).await {
        Ok(Some(line)) => {
            let line = line.trim();
            Some(line.eq_ignore_ascii_case("y") || line.eq_ignore_ascii_case("yes"))
        }
        Ok(None) => Some(false),
        Err(_) => {
            eprintln!("(no answer, taken as no)");
            None
        }
    }
}
