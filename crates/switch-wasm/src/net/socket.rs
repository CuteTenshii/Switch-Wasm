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
        state.link = Some(Link {
            socket,
            inbox,
            pending: Vec::new(),
            _callbacks: [on_message, on_open, on_close],
        });
    });
    Ok(())
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
        let State { client, link, .. } = &mut *state;

        let mut events = Vec::new();
        let mut gone = false;
        let mut broken = false;
        if let Some(link) = link.as_mut() {
            let frames: Vec<_> = link.inbox.borrow_mut().frames.drain(..).collect();
            for raw in frames {
                match client.incoming(&raw) {
                    Ok(event) => events.extend(event),
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
                let frames = client.outgoing(commands);
                if link.socket.ready_state() == WebSocket::OPEN {
                    for frame in link.pending.drain(..).chain(frames) {
                        let _ = link.socket.send_with_u8_array(&frame);
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
