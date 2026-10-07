//! Controller, touch and rumble.

use switch_core::cpu::TouchPoint;

use crate::session;

/// The last rumble request, packed as `(weak << 16) | strong`, each 0..=1000.
#[no_mangle]
pub extern "C" fn switch_vibration(handle: u32) -> u32 {
    let s = session(handle);
    let (low, high) = s.cpu.vibration();
    let scale = |v: f32| (v * 1000.0).round().clamp(0.0, 1000.0) as u32;
    (scale(high) << 16) | scale(low)
}

/// Write `len` bytes from `ptr` into emulated memory at `addr` (used for the
/// memory-mapped input register and similar). Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn switch_write_mem(handle: u32, addr: u32, ptr: *const u8, len: u32) -> i32 {
    let s = session(handle);
    let data = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    match s.cpu.mem.map(addr, data) {
        Ok(()) => 0,
        Err(e) => {
            s.last_error = e.to_string();
            -1
        }
    }
}

/// Feed gamepad state. `buttons` is a `HidNpadButton` bitfield (A=1<<0, B=1<<1,
/// X=1<<2, Y=1<<3, StickL=1<<4, StickR=1<<5, L=1<<6, R=1<<7, ZL=1<<8, ZR=1<<9,
/// Plus=1<<10, Minus=1<<11, DpadLeft=1<<12, DpadUp=1<<13, DpadRight=1<<14,
/// DpadDown=1<<15); sticks are -32768..32767, positive right and up.
#[no_mangle]
pub extern "C" fn switch_set_input(
    handle: u32,
    buttons: u64,
    stick_lx: i32,
    stick_ly: i32,
    stick_rx: i32,
    stick_ry: i32,
) {
    session(handle)
        .cpu
        .set_gamepad_state(buttons, stick_lx, stick_ly, stick_rx, stick_ry);
}

/// Feed touch contacts: `count` packed `u32` triples (`finger_id`, `x`, `y`) in
/// 1280x720 digitizer space, truncated to 16. `count` 0 is a lift.
#[no_mangle]
pub extern "C" fn switch_set_touch(handle: u32, ptr: *const u32, count: u32) {
    let n = (count as usize).min(switch_core::cpu::TOUCH_MAX);
    let mut points = [TouchPoint::default(); switch_core::cpu::TOUCH_MAX];
    if n > 0 && !ptr.is_null() {
        let raw = unsafe { std::slice::from_raw_parts(ptr, n * 3) };
        for (i, point) in points[..n].iter_mut().enumerate() {
            point.finger_id = raw[i * 3];
            point.x = raw[i * 3 + 1];
            point.y = raw[i * 3 + 2];
        }
    }
    session(handle).cpu.set_touch_state(&points[..n]);
}
