//! The relay server: one WebSocket per page, many TCP and UDP streams over it.
//! `docs/relay.md` is the protocol.

pub mod policy;
mod session;
mod stream;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_hdr_async_with_config;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

use futures_util::SinkExt;
use policy::Policy;

/// How long the relay waits on a page before it gives up on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// How long a connection may take to finish the WebSocket upgrade.
    pub handshake: Duration,
    /// How often the relay pings the page.
    pub ping_every: Duration,
    /// How long the page may send nothing, not even a pong, before the relay drops it.
    pub silence: Duration,
}

impl Default for Timeouts {
    fn default() -> Timeouts {
        Timeouts {
            handshake: Duration::from_secs(10),
            ping_every: Duration::from_secs(15),
            silence: Duration::from_secs(45),
        }
    }
}

/// Pause after a failed accept, which is usually the process out of descriptors: retrying at
/// once would spin until one is freed.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(250);

/// Accept pages for as long as the relay runs.
pub async fn serve(listener: TcpListener, policy: Arc<Policy>, timeouts: Timeouts) {
    loop {
        match listener.accept().await {
            Ok((socket, _)) => {
                tokio::spawn(handle(socket, policy.clone(), timeouts));
            }
            Err(error) => {
                eprintln!("cannot accept a page: {error}");
                tokio::time::sleep(ACCEPT_BACKOFF).await;
            }
        }
    }
}

/// The largest message a page may send: a full datagram and its header.
fn websocket_config() -> WebSocketConfig {
    let largest = switch_relay_proto::MAX_FRAME;
    WebSocketConfig::default()
        .max_message_size(Some(largest))
        .max_frame_size(Some(largest))
}

/// Note the page's `Origin` and accept the upgrade; whether it may stay is decided after.
#[allow(clippy::result_large_err)]
fn remember_origin(
    seen: &Mutex<Option<String>>,
    request: &Request,
    response: Response,
) -> Result<Response, ErrorResponse> {
    let header = request
        .headers()
        .get("origin")
        .and_then(|value| value.to_str().ok());
    *seen.lock().unwrap() = header.map(str::to_owned);
    Ok(response)
}

#[allow(clippy::result_large_err)]
async fn handle(socket: TcpStream, policy: Arc<Policy>, timeouts: Timeouts) {
    let origin = Arc::new(Mutex::new(None::<String>));
    let seen = origin.clone();
    let upgrade = accept_hdr_async_with_config(
        socket,
        move |request: &Request, response| remember_origin(&seen, request, response),
        Some(websocket_config()),
    );
    let Ok(Ok(mut socket)) = tokio::time::timeout(timeouts.handshake, upgrade).await else {
        return;
    };
    let origin = origin.lock().unwrap().take();
    let from = origin.as_deref().unwrap_or("no origin");
    if !policy.origin_allowed(origin.as_deref()).await {
        eprintln!("refused {from}");
        let refusal = Message::Close(Some(CloseFrame {
            code: CloseCode::Policy,
            reason: "".into(),
        }));
        let _ = socket.send(refusal).await;
        return;
    }
    eprintln!("page {from} connected");
    match session::run(socket, policy, timeouts).await {
        session::End::Left => eprintln!("page {from} left"),
        session::End::Silent => eprintln!("page {from} stopped answering, dropped it"),
    }
}
