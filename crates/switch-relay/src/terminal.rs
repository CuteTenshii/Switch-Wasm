//! Questions on the relay's terminal.

use std::io::BufRead;

use tokio::sync::Mutex;

/// One question at a time: two would read each other's answers.
static ASKING: Mutex<()> = Mutex::const_new(());

/// Ask `question` with `[y/N]` and wait for a line; anything but yes is no.
pub async fn ask(question: &str) -> bool {
    let _turn = ASKING.lock().await;
    eprint!("{question} [y/N] ");
    let answer = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).ok()?;
        Some(line)
    })
    .await
    .ok()
    .flatten();
    answer.is_some_and(|line| matches!(line.trim(), "y" | "Y" | "yes"))
}
