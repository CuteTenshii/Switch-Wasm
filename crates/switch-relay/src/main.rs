use std::io::IsTerminal;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::process::ExitCode;
use std::sync::Arc;

use switch_relay::config::{self, Approval, Config, Overrides};
use switch_relay::policy::{Unknown, TRUSTED_ORIGINS};
use switch_relay::Timeouts;
use tokio::net::TcpListener;

const USAGE: &str = "switch-relay: carries the emulator's TCP and UDP traffic over a WebSocket.

Run it with no arguments. The emulator finds it at ws://127.0.0.1:18787.

Its settings are in switch-relay.toml, in this user's config directory, and the emulator's
Network tab can change them. A flag sets its setting for this run, whatever the file says.

  --port <n>             listen on another port
  --allow-origin <url>   trust another page without asking (repeatable)
  --allow-private        let guests reach loopback and local networks
  --allow-inbound        let guests listen for connections and hear from hosts they never
                         sent to, on this machine's ports
";

#[tokio::main]
async fn main() -> ExitCode {
    let mut overrides = Overrides::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => match args.next().and_then(|value| value.parse().ok()) {
                Some(0) | None => return usage("--port needs a number from 1 to 65535"),
                Some(value) => overrides.port = Some(value),
            },
            "--allow-origin" => match args.next().as_deref().map(config::origin) {
                Some(Some(origin)) => overrides.origins.push(origin),
                Some(None) => {
                    return usage("--allow-origin needs an origin, like https://example.org")
                }
                None => return usage("--allow-origin needs an origin"),
            },
            "--allow-private" => overrides.allow_private = true,
            "--allow-inbound" => overrides.allow_inbound = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => return usage(&format!("unknown argument {other}")),
        }
    }
    let path = config::default_path();
    let settings = match path.as_deref().map(config::load).transpose() {
        Ok(settings) => settings.unwrap_or_default(),
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let port = overrides.port.unwrap_or(settings.port);
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
    let terminal = std::io::stdin().is_terminal();
    let unknown = if terminal {
        Unknown::Ask
    } else {
        Unknown::Deny
    };
    eprintln!("switch-relay listening on ws://127.0.0.1:{port}");
    eprintln!("trusted: {}", TRUSTED_ORIGINS.join(", "));
    match &path {
        Some(path) => eprintln!("settings: {}", path.display()),
        None => eprintln!("no config directory: settings changes last until the relay exits"),
    }
    let approval = Approval::Ask { terminal };
    let config = Config::new(settings, overrides, unknown, approval, path, port);
    let config = Arc::new(config);
    if let Some(listener) = listener_v6 {
        tokio::spawn(switch_relay::serve(
            listener,
            config.clone(),
            Timeouts::default(),
        ));
    }
    switch_relay::serve(listener, config, Timeouts::default()).await;
    ExitCode::SUCCESS
}

fn usage(problem: &str) -> ExitCode {
    eprintln!("{problem}\n\n{USAGE}");
    ExitCode::FAILURE
}
