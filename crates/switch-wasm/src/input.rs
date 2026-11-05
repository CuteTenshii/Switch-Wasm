//! Controller, touch and rumble.

use std::cell::RefCell;

use switch_core::cpu::{Cpu, TouchPoint, TOUCH_MAX};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::session;

/// The last rumble request, packed as `(weak << 16) | strong`, each 0..=1000.
#[no_mangle]
pub extern "C" fn switch_vibration() -> u32 {
    let s = session();
    let (low, high) = s.cpu.vibration();
    let scale = |v: f32| (v * 1000.0).round().clamp(0.0, 1000.0) as u32;
    (scale(high) << 16) | scale(low)
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

    fn set_gamepad(&mut self, cpu: &mut Cpu, buttons: u64, sticks: Sticks) {
        // Edges, not level.
        let pressed = buttons & !self.held;
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

/// Drop latched input the guest has had a frame to see.
pub(crate) fn release_if_seen(cpu: &mut Cpu) {
    LATCH.with_borrow_mut(|latch| latch.release_if_seen(cpu));
}

/// Feed gamepad state. `buttons` is a `HidNpadButton` bitfield (A=1<<0, B=1<<1,
/// X=1<<2, Y=1<<3, StickL=1<<4, StickR=1<<5, L=1<<6, R=1<<7, ZL=1<<8, ZR=1<<9,
/// Plus=1<<10, Minus=1<<11, DpadLeft=1<<12, DpadUp=1<<13, DpadRight=1<<14,
/// DpadDown=1<<15); sticks are -32768..32767, positive right and up. A press is kept
/// until the guest has had a frame to see it.
#[no_mangle]
pub extern "C" fn switch_set_input(
    buttons: u64,
    stick_lx: i32,
    stick_ly: i32,
    stick_rx: i32,
    stick_ry: i32,
) {
    let cpu = &mut session().cpu;
    LATCH.with_borrow_mut(|latch| {
        latch.set_gamepad(cpu, buttons, [stick_lx, stick_ly, stick_rx, stick_ry]);
    });
}

/// Feed touch contacts: `(finger_id, x, y)` triples in 1280x720 digitizer space. No triples is
/// a lift. A new finger is kept until the guest has had a frame to see it.
#[wasm_bindgen]
pub fn switch_set_touch(points: &[u32]) {
    let touches = points
        .chunks_exact(3)
        .map(|point| TouchPoint {
            finger_id: point[0],
            x: point[1],
            y: point[2],
        })
        .collect();
    let cpu = &mut session().cpu;
    LATCH.with_borrow_mut(|latch| latch.set_touch(cpu, touches));
}
