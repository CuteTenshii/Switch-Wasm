//! The WebSocket to the relay. Its callbacks only queue; `exchange` moves the queue
//! into the core once per slice.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use js_sys::{ArrayBuffer, Uint8Array};
use switch_core::cpu::Cpu;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{BinaryType, MessageEvent, WebSocket};

use super::client::Client;
use crate::post::post;

/// What the callbacks leave for `exchange`.
#[derive(Default)]
struct Inbox {
    frames: VecDeque<Vec<u8>>,
    ended: bool,
}

struct Link {
    socket: WebSocket,
    inbox: Rc<RefCell<Inbox>>,
    /// Frames for a socket that has not opened yet.
    pending: Vec<Vec<u8>>,
    /// Kept alive for as long as the socket can call them.
    _callbacks: [Closure<dyn FnMut(JsValue)>; 3],
}

#[derive(Default)]
struct State {
    client: Client,
    /// The page asked for a network: set from `switch_net_connect` until `switch_net_disconnect`.
    enabled: bool,
    link: Option<Link>,
    /// Frames written to and read from the socket since the page asked for a network.
    frames_out: u32,
    frames_in: u32,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Open a link to the relay at `url`, replacing any other. The guest sees a network from the
/// next slice; until the socket opens its connects wait.
#[wasm_bindgen]
pub fn switch_net_connect(url: &str) -> Result<(), JsValue> {
    let socket = WebSocket::new(url)?;
    socket.set_binary_type(BinaryType::Arraybuffer);
    let inbox = Rc::new(RefCell::new(Inbox::default()));

    let on_message = {
        let inbox = inbox.clone();
        Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
            let Ok(event) = event.dyn_into::<MessageEvent>() else {
                return;
            };
            if let Ok(buffer) = event.data().dyn_into::<ArrayBuffer>() {
                inbox
                    .borrow_mut()
                    .frames
                    .push_back(Uint8Array::new(&buffer).to_vec());
            }
        })
    };
    let on_open = Closure::<dyn FnMut(JsValue)>::new(|_: JsValue| {
        post("net_state", &[("state", "open".into())]);
    });
    // A failed connect closes too, so this is the one place a lost link is noticed.
    let on_close = {
        let inbox = inbox.clone();
        Closure::<dyn FnMut(JsValue)>::new(move |_: JsValue| {
            inbox.borrow_mut().ended = true;
            post("net_state", &[("state", "unreachable".into())]);
        })
    };
    let callback = |closure: &Closure<dyn FnMut(JsValue)>| -> js_sys::Function {
        closure.as_ref().unchecked_ref::<js_sys::Function>().clone()
    };
    socket.set_onmessage(Some(&callback(&on_message)));
    socket.set_onopen(Some(&callback(&on_open)));
    socket.set_onclose(Some(&callback(&on_close)));

    post("net_state", &[("state", "connecting".into())]);
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(old) = state.link.take() {
            close(&old.socket);
        }
        state.enabled = true;
        state.frames_out = 0;
        state.frames_in = 0;
        state.link = Some(Link {
            socket,
            inbox,
            pending: Vec::new(),
            _callbacks: [on_message, on_open, on_close],
        });
    });
    Ok(())
}

/// The socket's state and how many frames have crossed it, as
/// `{"socket":"none|connecting|open|closing|closed","out":n,"in":n}`.
#[wasm_bindgen]
pub fn switch_net_status_json() -> String {
    STATE.with(|state| {
        let state = state.borrow();
        let socket = match state.link.as_ref().map(|link| link.socket.ready_state()) {
            None => "none",
            Some(WebSocket::CONNECTING) => "connecting",
            Some(WebSocket::OPEN) => "open",
            Some(WebSocket::CLOSING) => "closing",
            Some(_) => "closed",
        };
        switch_core::json::object(|o| {
            o.field("socket", socket)
                .field("out", state.frames_out)
                .field("in", state.frames_in);
        })
    })
}

/// The guest's connections as JSON, newest first; see `Client::report`.
#[wasm_bindgen]
pub fn switch_net_connections_json() -> String {
    STATE.with(|state| state.borrow().client.report())
}

/// Close the link and take the network away from the guest.
#[wasm_bindgen]
pub fn switch_net_disconnect() {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(link) = state.link.take() {
            close(&link.socket);
        }
        state.enabled = false;
        let lost = state.client.lost();
        if let Some(session) = crate::session_if_live() {
            session.cpu.deliver_net_events(lost);
            session.cpu.set_network(false);
        }
    });
    post("net_state", &[("state", "off".into())]);
}

/// A view of this module's memory is over a `SharedArrayBuffer`, which `WebSocket.send` refuses,
/// so the frame goes in a plain array first.
fn send(socket: &WebSocket, frame: &[u8]) -> bool {
    let array = Uint8Array::new_with_length(frame.len() as u32);
    array.copy_from(frame);
    socket.send_with_array_buffer_view(&array).is_ok()
}

fn close(socket: &WebSocket) {
    socket.set_onmessage(None);
    socket.set_onopen(None);
    socket.set_onclose(None);
    let _ = socket.close();
}

/// Once per slice: what the relay sent goes to the guest's sockets, and what the guest
/// asked goes to the relay.
pub(crate) fn exchange(cpu: &mut Cpu) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if !state.enabled {
            return;
        }
        cpu.set_network(true);
        let State {
            client,
            link,
            frames_out,
            frames_in,
            ..
        } = &mut *state;

        let mut events = Vec::new();
        // What answering the relay needs sent back to it, such as a handshake's next message.
        let mut replies = Vec::new();
        let mut gone = false;
        let mut broken = false;
        if let Some(link) = link.as_mut() {
            let frames: Vec<_> = link.inbox.borrow_mut().frames.drain(..).collect();
            *frames_in += frames.len() as u32;
            for raw in frames {
                match client.incoming(&raw) {
                    Ok(pump) => {
                        events.extend(pump.events);
                        replies.extend(pump.frames);
                    }
                    Err(_) => {
                        broken = true;
                        break;
                    }
                }
            }
            gone = broken || link.inbox.borrow().ended;
        }

        let commands = cpu.take_net_commands();
        match link.as_mut() {
            Some(link) if !gone => {
                let pump = client.outgoing(commands);
                events.extend(pump.events);
                let frames: Vec<_> = replies.into_iter().chain(pump.frames).collect();
                if link.socket.ready_state() == WebSocket::OPEN {
                    for frame in link.pending.drain(..).chain(frames) {
                        if send(&link.socket, &frame) {
                            *frames_out += 1;
                        }
                    }
                    client.sent();
                } else {
                    link.pending.extend(frames);
                }
            }
            _ => events.extend(Client::refuse(commands)),
        }

        if gone {
            if let Some(link) = link.take() {
                close(&link.socket);
            }
            events.extend(client.lost());
            // A closed socket has already said so; one that spoke nonsense has not.
            if broken {
                post("net_state", &[("state", "unreachable".into())]);
            }
        }
        cpu.deliver_net_events(events);
    });
}
