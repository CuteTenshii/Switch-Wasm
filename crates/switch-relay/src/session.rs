//! One WebSocket: its streams, and the frames between them and the page.

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use switch_relay_proto::{ClientFrame, ServerFrame, StreamId, VERSION};
use tokio::net::TcpStream;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use crate::policy::Policy;
use crate::stream::{self, Command, Credit, Open};

/// Most streams one WebSocket may have open.
pub const MAX_STREAMS: u16 = 256;

struct Handle {
    commands: UnboundedSender<Command>,
    credit: Arc<Credit>,
    task: JoinHandle<()>,
}

fn binary(frame: &ServerFrame) -> Message {
    Message::Binary(frame.encode().into())
}

fn close(code: CloseCode) -> Message {
    Message::Close(Some(CloseFrame {
        code,
        reason: "".into(),
    }))
}

pub async fn run(socket: WebSocketStream<TcpStream>, policy: Arc<Policy>) {
    let (mut sink, mut source) = socket.split();
    let (out, mut outgoing) = unbounded_channel::<Message>();
    let writer = tokio::spawn(async move {
        while let Some(message) = outgoing.recv().await {
            let last = matches!(message, Message::Close(_));
            if sink.send(message).await.is_err() || last {
                break;
            }
        }
    });
    let (frames, mut frames_rx) = unbounded_channel::<ServerFrame>();
    let forward = tokio::spawn({
        let out = out.clone();
        async move {
            while let Some(frame) = frames_rx.recv().await {
                if out.send(binary(&frame)).is_err() {
                    break;
                }
            }
        }
    });
    let _ = frames.send(ServerFrame::Hello {
        version: VERSION,
        max_streams: MAX_STREAMS,
    });

    let (done, mut finished) = unbounded_channel::<StreamId>();
    let mut streams: HashMap<StreamId, Handle> = HashMap::new();
    let mut ending = CloseCode::Normal;
    loop {
        tokio::select! {
            message = source.next() => {
                let frame = match message {
                    None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                    Some(Ok(Message::Binary(raw))) => ClientFrame::decode(&raw),
                    Some(Ok(Message::Text(_))) => {
                        ending = CloseCode::Unsupported;
                        break;
                    }
                    Some(Ok(_)) => continue,
                };
                let Ok(frame) = frame else {
                    ending = CloseCode::Protocol;
                    break;
                };
                match frame {
                    ClientFrame::Connect { stream, proto, port, window, peer } => {
                        if streams.contains_key(&stream) {
                            ending = CloseCode::Protocol;
                            break;
                        }
                        if streams.len() >= usize::from(MAX_STREAMS) {
                            let reason = switch_relay_proto::Failure::Denied;
                            let _ = frames.send(ServerFrame::Failed { stream, reason });
                            continue;
                        }
                        let (commands, commands_rx) = unbounded_channel();
                        let credit = Arc::new(Credit::new(window));
                        let task = tokio::spawn(stream::run(
                            Open { id: stream, proto, port, peer },
                            policy.clone(),
                            credit.clone(),
                            frames.clone(),
                            commands_rx,
                            done.clone(),
                        ));
                        streams.insert(stream, Handle { commands, credit, task });
                    }
                    ClientFrame::Data { stream, data } => {
                        if let Some(handle) = streams.get(&stream) {
                            let _ = handle.commands.send(Command::Data(data));
                        }
                    }
                    ClientFrame::ShutdownWrite { stream } => {
                        if let Some(handle) = streams.get(&stream) {
                            let _ = handle.commands.send(Command::ShutdownWrite);
                        }
                    }
                    ClientFrame::Credit { stream, bytes } => {
                        if let Some(handle) = streams.get(&stream) {
                            handle.credit.add(bytes);
                        }
                    }
                    ClientFrame::Close { stream } => {
                        if let Some(handle) = streams.remove(&stream) {
                            handle.task.abort();
                        }
                    }
                }
            }
            Some(id) = finished.recv() => {
                streams.remove(&id);
            }
        }
    }
    for handle in streams.values() {
        handle.task.abort();
    }
    if ending != CloseCode::Normal {
        let _ = out.send(close(ending));
    }
    drop(out);
    forward.abort();
    let _ = writer.await;
}
