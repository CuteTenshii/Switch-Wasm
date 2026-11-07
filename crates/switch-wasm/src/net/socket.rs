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
    /// The page has been told the socket opened.
    announced: bool,
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
    let ended = |inbox: &Rc<RefCell<Inbox>>| {
        let inbox = inbox.clone();
        Closure::<dyn FnMut(JsValue)>::new(move |_: JsValue| inbox.borrow_mut().ended = true)
    };
    let (on_error, on_close) = (ended(&inbox), ended(&inbox));
    socket.set_onmessage(Some(
        on_message.as_ref().unchecked_ref::<js_sys::Function>(),
    ));
    socket.set_onerror(Some(on_error.as_ref().unchecked_ref::<js_sys::Function>()));
    socket.set_onclose(Some(on_close.as_ref().unchecked_ref::<js_sys::Function>()));

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
            announced: false,
            _callbacks: [on_message, on_error, on_close],
        });
    });
    Ok(())
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
    socket.set_onerror(None);
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
        if let Some(link) = link.as_mut() {
            let frames: Vec<_> = link.inbox.borrow_mut().frames.drain(..).collect();
            for raw in frames {
                match client.incoming(&raw) {
                    Ok(event) => events.extend(event),
                    Err(_) => {
                        gone = true;
                        break;
                    }
                }
            }
            gone |= link.inbox.borrow().ended;
        }

        let commands = cpu.take_net_commands();
        match link.as_mut() {
            Some(link) if !gone => {
                let frames = client.outgoing(commands);
                if link.socket.ready_state() == WebSocket::OPEN {
                    if !std::mem::replace(&mut link.announced, true) {
                        post("net_state", &[("state", "open".into())]);
                    }
                    for frame in link.pending.drain(..).chain(frames) {
                        let _ = link.socket.send_with_u8_array(&frame);
                    }
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
            post("net_state", &[("state", "unreachable".into())]);
        }
        cpu.deliver_net_events(events);
    });
}
