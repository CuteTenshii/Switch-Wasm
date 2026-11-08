use std::io::IsTerminal;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::process::ExitCode;
use std::sync::Arc;

use switch_relay::policy::{Policy, Unknown, TRUSTED_ORIGINS};
use switch_relay::Timeouts;
use tokio::net::TcpListener;

const DEFAULT_PORT: u16 = 18787;

const USAGE: &str = "switch-relay: carries the emulator's TCP and UDP traffic over a WebSocket.

Run it with no arguments. The emulator finds it at ws://127.0.0.1:18787.

  --port <n>             listen on another port
  --allow-origin <url>   trust another page without asking (repeatable)
  --allow-private        let guests reach loopback and local networks
  --allow-inbound        let guests listen for connections and hear from hosts they never
                         sent to, on this machine's ports
";

#[tokio::main]
async fn main() -> ExitCode {
    let mut port = DEFAULT_PORT;
    let mut origins = Vec::new();
    let mut allow_private = false;
    let mut allow_inbound = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => match args.next().and_then(|value| value.parse().ok()) {
                Some(value) => port = value,
                None => return usage("--port needs a number"),
            },
            "--allow-origin" => match args.next() {
                Some(value) => origins.push(value),
                None => return usage("--allow-origin needs an origin"),
            },
            "--allow-private" => allow_private = true,
            "--allow-inbound" => allow_inbound = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => return usage(&format!("unknown argument {other}")),
        }
    }
    let listener = match TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("cannot listen on 127.0.0.1:{port}: {error}");
            return ExitCode::FAILURE;
        }
    };
    // Where `localhost` means IPv6 first; a host without IPv6 does without it.
    let listener_v6 = match TcpListener::bind((Ipv6Addr::LOCALHOST, port)).await {
        Ok(listener) => Some(listener),
        Err(error) => {
            eprintln!("not listening on [::1]:{port}: {error}");
            None
        }
    };
    let unknown = if std::io::stdin().is_terminal() {
        Unknown::Ask
    } else {
        Unknown::Deny
    };
    eprintln!("switch-relay listening on ws://127.0.0.1:{port}");
    eprintln!("trusted: {}", TRUSTED_ORIGINS.join(", "));
    let policy = Arc::new(Policy::new(origins, unknown, allow_private, allow_inbound));
    if let Some(listener) = listener_v6 {
        tokio::spawn(switch_relay::serve(
            listener,
            policy.clone(),
            Timeouts::default(),
        ));
    }
    switch_relay::serve(listener, policy, Timeouts::default()).await;
    ExitCode::SUCCESS
}

fn usage(problem: &str) -> ExitCode {
    eprintln!("{problem}\n\n{USAGE}");
    ExitCode::FAILURE
}
