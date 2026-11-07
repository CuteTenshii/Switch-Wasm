//! The relay server: one WebSocket per page, many TCP and UDP streams over it.
//! `docs/relay.md` is the protocol.

pub mod policy;
mod session;
mod stream;

use std::sync::{Arc, Mutex};

use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;

use futures_util::SinkExt;
use policy::Policy;

/// Accept pages until the listener fails.
pub async fn serve(listener: TcpListener, policy: Arc<Policy>) {
    while let Ok((socket, _)) = listener.accept().await {
        tokio::spawn(handle(socket, policy.clone()));
    }
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
async fn handle(socket: TcpStream, policy: Arc<Policy>) {
    let origin = Arc::new(Mutex::new(None::<String>));
    let seen = origin.clone();
    let accepted = accept_hdr_async(socket, move |request: &Request, response| {
        remember_origin(&seen, request, response)
    })
    .await;
    let Ok(mut socket) = accepted else {
        return;
    };
    let origin = origin.lock().unwrap().take();
    if !policy.origin_allowed(origin.as_deref()).await {
        let refusal = Message::Close(Some(CloseFrame {
            code: CloseCode::Policy,
            reason: "".into(),
        }));
        let _ = socket.send(refusal).await;
        return;
    }
    session::run(socket, policy).await;
}
