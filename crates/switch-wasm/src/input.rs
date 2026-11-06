//! Controller, touch and rumble.

use std::cell::RefCell;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI32, AtomicU32};

use switch_core::cpu::{Cpu, TouchPoint, TOUCH_MAX};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

/// Controller state and rumble in linear memory, which the page reads and writes over the
/// shared buffer, so no message carries them. The page writes `buttons`, `pressed` and
/// `sticks`; the core writes `rumble`.
#[repr(C)]
struct Mailbox {
    /// A `HidNpadButton` bitfield of what is down.
    buttons: AtomicU32,
    /// Buttons that went down since the last slice, so a tap shorter than one is kept.
    pressed: AtomicU32,
    /// The analog sticks, -32768..32767, positive right and up.
    sticks: [AtomicI32; 4],
    /// The last rumble request, packed as `(weak << 16) | strong`, each 0..=1000.
    rumble: AtomicU32,
}

static MAILBOX: Mailbox = Mailbox {
    buttons: AtomicU32::new(0),
    pressed: AtomicU32::new(0),
    sticks: [const { AtomicI32::new(0) }; 4],
    rumble: AtomicU32::new(0),
};

/// Hand the page the mailbox over the shared memory.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub(crate) fn announce_mailbox() {
    use js_sys::WebAssembly;
    use wasm_bindgen::JsCast;
    let memory: WebAssembly::Memory = wasm_bindgen::memory().unchecked_into();
    crate::post::post(
        "input_mailbox",
        &[
            ("buffer", memory.buffer()),
            ("at", (&MAILBOX as *const Mailbox as u32).into()),
        ],
    );
}

// A press is held until the guest's frame counter has advanced twice, so taps between run
// slices are not lost. Keys still down are published live.

/// Matches `HID_STICK_THRESHOLD` in `services/input.rs`; a flick latches like a press.
const STICK_THRESHOLD: u32 = 0x4000;
/// Frames the guest gets to see a latched input, and the slices after which a program that
/// never presents lets it go anyway.
const LATCH_FRAMES: u64 = 2;
const MAX_LATCH_SLICES: u32 = 64;

type Sticks = [i32; 4];

#[derive(Default)]
struct Latch {
    /// What the host says is physically down right now.
    held: u64,
    /// Pressed, but not yet guaranteed seen by the guest.
    latched: u64,
    /// The newest analog values.
    sticks: Sticks,
    /// A deflection held like a press.
    latched_sticks: Option<Sticks>,
    /// The newest host contacts.
    touches: Vec<TouchPoint>,
    /// A tap held until a frame passes.
    latched_touches: Option<Vec<TouchPoint>>,
    /// Finger ids down at the last sample.
    touch_ids: Vec<u32>,
    /// Contacts the guest was last told about.
    published_touches: usize,
    /// The frame the latch was armed at.
    armed_at: u64,
    slices: u32,
}

thread_local! {
    static LATCH: RefCell<Latch> = RefCell::new(Latch::default());
}

fn deflected(sticks: &Sticks) -> bool {
    sticks.iter().any(|v| v.unsigned_abs() > STICK_THRESHOLD)
}

impl Latch {
    fn has_latched(&self) -> bool {
        self.latched != 0 || self.latched_sticks.is_some() || self.latched_touches.is_some()
    }

    /// Start or restart the wait; restarting extends it for earlier presses too.
    fn arm(&mut self, cpu: &Cpu) {
        self.armed_at = cpu.nv.gpu.frames;
        self.slices = 0;
    }

    fn publish(&mut self, cpu: &mut Cpu) {
        // Live deflection wins; a latched flick stands in once the stick recentres.
        let sticks = match self.latched_sticks {
            Some(latched) if !deflected(&self.sticks) => latched,
            _ => self.sticks,
        };
        cpu.set_gamepad_state(
            self.held | self.latched,
            sticks[0],
            sticks[1],
            sticks[2],
            sticks[3],
        );
        // Live contacts win; a latched tap stands in once the finger is up.
        let source = if self.touches.is_empty() {
            self.latched_touches.as_deref().unwrap_or(&[])
        } else {
            &self.touches
        };
        let count = source.len().min(TOUCH_MAX);
        if count == 0 && self.published_touches == 0 {
            return;
        }
        cpu.set_touch_state(&source[..count]);
        self.published_touches = count;
    }

    fn set_gamepad(&mut self, cpu: &mut Cpu, buttons: u64, tapped: u64, sticks: Sticks) {
        // Edges, not level.
        let pressed = (buttons & !self.held) | tapped;
        self.held = buttons;
        self.sticks = sticks;
        let flicked = deflected(&sticks);
        if flicked {
            self.latched_sticks = Some(sticks);
        }
        if pressed != 0 || flicked {
            self.latched |= pressed;
            self.arm(cpu);
        }
        self.publish(cpu);
    }

    fn set_touch(&mut self, cpu: &mut Cpu, touches: Vec<TouchPoint>) {
        let ids: Vec<u32> = touches.iter().map(|t| t.finger_id).collect();
        // A new finger id arms the latch.
        let fresh = ids.iter().any(|id| !self.touch_ids.contains(id));
        if fresh {
            self.latched_touches = Some(touches.clone());
            self.arm(cpu);
        }
        self.touches = touches;
        self.touch_ids = ids;
        self.publish(cpu);
    }

    /// Called once per run slice; drops the latch after a full visible frame.
    fn release_if_seen(&mut self, cpu: &mut Cpu) {
        if !self.has_latched() {
            return;
        }
        self.slices += 1;
        let seen = cpu.nv.gpu.frames.wrapping_sub(self.armed_at) >= LATCH_FRAMES;
        if !seen && self.slices < MAX_LATCH_SLICES {
            return;
        }
        self.latched = 0;
        self.latched_sticks = None;
        self.latched_touches = None;
        self.slices = 0;
        self.publish(cpu);
    }
}

/// Let go of everything held and latched, for a title that has ended.
pub(crate) fn reset() {
    LATCH.with_borrow_mut(|latch| *latch = Latch::default());
}

/// What the page does on each of its ticks: write the state, noting new presses.
#[cfg(test)]
pub(crate) fn page_write(buttons: u32, sticks: [i32; 4]) {
    let newly = buttons & !MAILBOX.buttons.swap(buttons, Relaxed);
    MAILBOX.pressed.fetch_or(newly, Relaxed);
    for (slot, value) in MAILBOX.sticks.iter().zip(sticks) {
        slot.store(value, Relaxed);
    }
}

/// Take what the page wrote since the last slice, and tell it the rumble the guest asked for.
pub(crate) fn exchange(cpu: &mut Cpu) {
    let buttons = u64::from(MAILBOX.buttons.load(Relaxed));
    let tapped = u64::from(MAILBOX.pressed.swap(0, Relaxed));
    let sticks = MAILBOX.sticks.each_ref().map(|v| v.load(Relaxed));
    LATCH.with_borrow_mut(|latch| {
        if tapped != 0 || buttons != latch.held || sticks != latch.sticks {
            latch.set_gamepad(cpu, buttons, tapped, sticks);
        }
    });
    let (low, high) = cpu.vibration();
    let scale = |v: f32| (v * 1000.0).round().clamp(0.0, 1000.0) as u32;
    MAILBOX
        .rumble
        .store((scale(high) << 16) | scale(low), Relaxed);
}

/// Drop latched input the guest has had a frame to see.
pub(crate) fn release_if_seen(cpu: &mut Cpu) {
    LATCH.with_borrow_mut(|latch| latch.release_if_seen(cpu));
}

/// Feed touch contacts: `(finger_id, x, y)` triples in 1280x720 digitizer space. No triples is
/// a lift. A new finger is kept until the guest has had a frame to see it.
#[wasm_bindgen]
pub fn switch_set_touch(points: &[u32]) {
    let touches = points
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[finger_id, x, y]| TouchPoint { finger_id, x, y })
        .collect();
    let cpu = &mut session().cpu;
    LATCH.with_borrow_mut(|latch| latch.set_touch(cpu, touches));
}
