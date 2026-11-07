//! Host input published to hid shared memory.

use crate::cpu::*;

/// One style the pad can present in hid's shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NpadPresentation {
    /// `HidNpadStyleTag` bit.
    pub(crate) style: u32,
    /// `HidDeviceTypeBits`.
    pub(crate) device_type: u32,
    /// The per-style LIFO.
    pub(crate) lifo: u32,
    /// `HidNpadAttribute`.
    pub(crate) attributes: u32,
    /// `HidNpadJoyAssignmentMode`.
    pub(crate) joy_assignment: u32,
}

/// Offsets into libnx's `HidSharedMemory` (`switch/services/hid.h`).
pub(crate) mod hid_shmem {
    /// `offsetof(HidSharedMemory, npad)`.
    pub const NPAD: u32 = 0x9A00;
    /// `sizeof(HidNpadSharedMemoryEntry)`; `internal_state` sits at its start.
    pub const ENTRY_SIZE: u32 = 0x5000;
    /// Slot `HidNpadIdType_Handheld` reads (players 1-8 are slots 0-7).
    pub const HANDHELD_SLOT: u32 = 8;

    pub const STYLE_SET: u32 = 0x00;
    pub const JOY_ASSIGNMENT_MODE: u32 = 0x04;
    pub const FULL_KEY_LIFO: u32 = 0x28;
    pub const HANDHELD_LIFO: u32 = 0x378;
    /// The remaining per-style LIFOs at a 0x350 stride: Joy-Con pair, left, right, system.
    pub const JOY_DUAL_LIFO: u32 = 0x6C8;
    pub const JOY_LEFT_LIFO: u32 = 0xA18;
    pub const JOY_RIGHT_LIFO: u32 = 0xD68;
    pub const SYSTEM_EXT_LIFO: u32 = 0x1408;
    pub const DEVICE_TYPE: u32 = 0x4188;
    /// `HidNpadSystemProperties`, then the three `HidPowerInfo` battery levels.
    pub const SYSTEM_PROPERTIES: u32 = 0x4190;
    pub const BATTERY_LEVEL: u32 = 0x4198;
    /// Power infos per npad: the whole pad, then its left and right halves.
    pub const POWER_INFO_COUNT: u32 = 3;

    /// `HidNpadCommonLifo`: a 0x20-byte header (unused/buffer_count/tail/count) and 17 entries.
    pub const LIFO_BUFFER_COUNT: u32 = 0x08;
    pub const LIFO_TAIL: u32 = 0x10;
    pub const LIFO_COUNT: u32 = 0x18;
    pub const LIFO_STORAGE: u32 = 0x20;
    pub const LIFO_CAPACITY: u64 = 17;

    /// `HidNpadCommonStateAtomicStorage`: sampling number (doubled; bit 0 is the seqlock flag),
    /// then the `HidNpadCommonState`.
    pub const STORAGE_SAMPLING_NUMBER: u32 = 0x00;
    pub const STATE_SAMPLING_NUMBER: u32 = 0x08;
    pub const STATE_BUTTONS: u32 = 0x10;
    pub const STATE_STICK_L: u32 = 0x18;
    pub const STATE_STICK_R: u32 = 0x20;
    pub const STATE_ATTRIBUTES: u32 = 0x28;

    /// `HidNpadStyleTag` bits.
    pub const STYLE_FULL_KEY: u32 = 1 << 0;
    pub const STYLE_HANDHELD: u32 = 1 << 1;
    pub const STYLE_JOY_DUAL: u32 = 1 << 2;
    pub const STYLE_JOY_LEFT: u32 = 1 << 3;
    pub const STYLE_JOY_RIGHT: u32 = 1 << 4;
    pub const STYLE_SYSTEM_EXT: u32 = 1 << 29;

    pub const DEVICE_FULL_KEY: u32 = 1 << 0;
    pub const DEVICE_HANDHELD: u32 = (1 << 2) | (1 << 3); // HandheldLeft|Right
    pub const DEVICE_JOY_LEFT: u32 = 1 << 4;
    pub const DEVICE_JOY_RIGHT: u32 = 1 << 5;

    /// `HidNpadJoyAssignmentMode`.
    pub const JOY_ASSIGNMENT_DUAL: u32 = 0;
    pub const JOY_ASSIGNMENT_SINGLE: u32 = 1;

    /// `HidPowerInfo::battery_level`, 0 to 4.
    pub const BATTERY_FULL: u32 = 4;

    /// `PowerInfo{0,1,2}PowerConnected` bits of `system_properties`; `Charging` stays clear.
    pub const SYSTEM_PROP_POWER_CONNECTED: u32 = (1 << 3) | (1 << 4) | (1 << 5);
    /// Button capabilities: ABXY, plus/minus, d-pad.
    pub const SYSTEM_PROP_FULL_BUTTONS: u32 = (1 << 11) | (1 << 13) | (1 << 14) | (1 << 15);

    pub const ATTR_CONNECTED: u32 = 1 << 0;
    pub const ATTR_WIRED: u32 = 1 << 1;
    pub const ATTR_LEFT_CONNECTED: u32 = 1 << 2;
    pub const ATTR_LEFT_WIRED: u32 = 1 << 3;
    pub const ATTR_RIGHT_CONNECTED: u32 = 1 << 4;
    pub const ATTR_RIGHT_WIRED: u32 = 1 << 5;

    /// `offsetof(HidSharedMemory, npad_condition)`. `nn::hid::GetNpadJoyHoldType` reads it
    /// directly and aborts (`2202-0710`) unless `is_valid` is set.
    pub const NPAD_CONDITION: u32 = 0x3E200;
    /// Its four words: reserved, initialized flag, hold type, valid flag.
    pub const NPAD_CONDITION_INITIALIZED: u32 = 0x04;
    pub const NPAD_CONDITION_HOLD_TYPE: u32 = 0x08;
    pub const NPAD_CONDITION_VALID: u32 = 0x0C;

    /// `offsetof(HidSharedMemory, touch_screen)`; its LIFO uses the npad LIFO header.
    pub const TOUCH_SCREEN: u32 = 0x400;
    /// The state begins one `u64` into each storage entry.
    pub const TOUCH_STATE: u32 = 0x08;

    /// `HidTouchScreenState` fields: sampling number, live count, then the slots.
    pub const TOUCH_SAMPLING_NUMBER: u32 = 0x00;
    pub const TOUCH_COUNT: u32 = 0x08;
    pub const TOUCH_TOUCHES: u32 = 0x10;

    /// `sizeof(HidTouchState)` and the fields of one.
    pub const TOUCH_SIZE: u32 = 0x28;
    pub const TOUCH_DELTA_TIME: u32 = 0x00;
    pub const TOUCH_ATTRIBUTES: u32 = 0x08;
    /// `nn::hid::TouchAttribute` start and end bits.
    pub const TOUCH_ATTR_START: u32 = 1 << 0;
    pub const TOUCH_ATTR_END: u32 = 1 << 1;
    pub const TOUCH_FINGER_ID: u32 = 0x0C;
    pub const TOUCH_X: u32 = 0x10;
    pub const TOUCH_Y: u32 = 0x14;
    pub const TOUCH_DIAMETER_X: u32 = 0x18;
    pub const TOUCH_DIAMETER_Y: u32 = 0x1C;
    pub const TOUCH_ROTATION_ANGLE: u32 = 0x20;

    /// The six `HidNpadSixAxisSensorLifo`s follow `SYSTEM_EXT_LIFO`: full key, handheld, joy
    /// dual left and right, joy left, joy right.
    pub const SIX_AXIS_FULL_KEY_LIFO: u32 = 0x1758;
    pub const SIX_AXIS_LIFO_SIZE: u32 = 0x708;
    pub const SIX_AXIS_HANDHELD: u32 = 1;
    pub const SIX_AXIS_JOY_DUAL_LEFT: u32 = 2;
    pub const SIX_AXIS_JOY_DUAL_RIGHT: u32 = 3;
    pub const SIX_AXIS_JOY_LEFT: u32 = 4;
    pub const SIX_AXIS_JOY_RIGHT: u32 = 5;

    /// `HidSixAxisSensorState` fields; the storage entry holds a sampling number, then it.
    pub const SIX_AXIS_STATE: u32 = 0x08;
    pub const SIX_AXIS_DELTA_TIME: u32 = 0x00;
    pub const SIX_AXIS_SAMPLING_NUMBER: u32 = 0x08;
    /// Eighteen `f32`s: acceleration, angular velocity, rotation, then the 3x3 orientation.
    pub const SIX_AXIS_FLOATS: u32 = 0x10;
    pub const SIX_AXIS_ATTRIBUTES: u32 = 0x58;
    pub const SIX_AXIS_ATTR_CONNECTED: u32 = 1 << 0;
}

/// The host's motion sensor: accelerometer in G, gyroscope in revolutions per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Motion {
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
    pub connected: bool,
    /// Gyro integrated since the sensor connected, in revolutions.
    rotation: [f32; 3],
    /// The sensor's axes as unit rows, in the frame it connected in.
    orientation: [[f32; 3]; 3],
    sample: u64,
    /// A connected state has been published, so a disconnect needs one more.
    published: bool,
}

const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl Default for Motion {
    fn default() -> Self {
        Self {
            accel: [0.0; 3],
            gyro: [0.0; 3],
            connected: false,
            rotation: [0.0; 3],
            orientation: IDENTITY,
            sample: 0,
            published: false,
        }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalized(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len == 0.0 {
        return v;
    }
    v.map(|c| c / len)
}

impl Motion {
    /// Turn the axes by the gyro over `seconds`, keeping them orthonormal.
    fn integrate(&mut self, seconds: f32) {
        if !self.connected {
            *self = Self {
                published: self.published,
                ..Self::default()
            };
            return;
        }
        for (rot, rate) in self.rotation.iter_mut().zip(self.gyro) {
            *rot += rate * seconds;
        }
        let [x, y, z] = self.orientation;
        let turn = std::f32::consts::TAU * seconds;
        let spin = [0, 1, 2]
            .map(|i| turn * (self.gyro[0] * x[i] + self.gyro[1] * y[i] + self.gyro[2] * z[i]));
        let step = |e: [f32; 3]| {
            let c = cross(spin, e);
            [e[0] + c[0], e[1] + c[1], e[2] + c[2]]
        };
        let x = normalized(step(x));
        let z = normalized(cross(x, step(y)));
        self.orientation = [x, cross(z, x), z];
    }
}

/// Styles the pad can be published as, best first. A style the title does not
/// support is no pad at all to `nn::hid`. `SystemExt` is published alongside, not here.
pub(crate) const NPAD_PRESENTATIONS: [NpadPresentation; 4] = [
    NpadPresentation {
        style: hid_shmem::STYLE_FULL_KEY,
        device_type: hid_shmem::DEVICE_FULL_KEY,
        lifo: hid_shmem::FULL_KEY_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED | hid_shmem::ATTR_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_DUAL,
    },
    NpadPresentation {
        style: hid_shmem::STYLE_JOY_DUAL,
        device_type: hid_shmem::DEVICE_JOY_LEFT | hid_shmem::DEVICE_JOY_RIGHT,
        lifo: hid_shmem::JOY_DUAL_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED
            | hid_shmem::ATTR_WIRED
            | hid_shmem::ATTR_LEFT_CONNECTED
            | hid_shmem::ATTR_LEFT_WIRED
            | hid_shmem::ATTR_RIGHT_CONNECTED
            | hid_shmem::ATTR_RIGHT_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_DUAL,
    },
    NpadPresentation {
        style: hid_shmem::STYLE_JOY_LEFT,
        device_type: hid_shmem::DEVICE_JOY_LEFT,
        lifo: hid_shmem::JOY_LEFT_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED
            | hid_shmem::ATTR_WIRED
            | hid_shmem::ATTR_LEFT_CONNECTED
            | hid_shmem::ATTR_LEFT_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_SINGLE,
    },
    NpadPresentation {
        style: hid_shmem::STYLE_JOY_RIGHT,
        device_type: hid_shmem::DEVICE_JOY_RIGHT,
        lifo: hid_shmem::JOY_RIGHT_LIFO,
        attributes: hid_shmem::ATTR_CONNECTED
            | hid_shmem::ATTR_WIRED
            | hid_shmem::ATTR_RIGHT_CONNECTED
            | hid_shmem::ATTR_RIGHT_WIRED,
        joy_assignment: hid_shmem::JOY_ASSIGNMENT_SINGLE,
    },
];

/// The handheld pad: its own npad id, not one of player 1's styles.
const NPAD_HANDHELD: NpadPresentation = NpadPresentation {
    style: hid_shmem::STYLE_HANDHELD,
    device_type: hid_shmem::DEVICE_HANDHELD,
    lifo: hid_shmem::HANDHELD_LIFO,
    attributes: hid_shmem::ATTR_CONNECTED
        | hid_shmem::ATTR_LEFT_CONNECTED
        | hid_shmem::ATTR_LEFT_WIRED
        | hid_shmem::ATTR_RIGHT_CONNECTED
        | hid_shmem::ATTR_RIGHT_WIRED,
    joy_assignment: hid_shmem::JOY_ASSIGNMENT_DUAL,
};

/// `HidNpadIdType` values of the two slots a pad can occupy.
const NPAD_ID_PLAYER_1: u32 = 0;
pub(crate) const NPAD_ID_HANDHELD: u32 = 0x20;

/// Every style the pad can be published in, as reported to the controller applet and `hid:sys`.
pub(crate) fn supported_npad_style_set() -> u32 {
    NPAD_PRESENTATIONS.iter().fold(
        NPAD_HANDHELD.style | hid_shmem::STYLE_SYSTEM_EXT,
        |set, pad| set | pad.style,
    )
}

/// Player 1's presentation for the title's style set; Pro Controller when none match.
pub(crate) fn npad_presentation_for(style_set: u32) -> NpadPresentation {
    NPAD_PRESENTATIONS
        .into_iter()
        .find(|presentation| style_set & presentation.style != 0)
        .unwrap_or(NPAD_PRESENTATIONS[0])
}

/// Deflection past which the stick pseudo-buttons are reported.
const HID_STICK_THRESHOLD: i32 = 0x4000;

/// The touchscreen digitizer resolution; touches use this space whatever the guest renders at.
pub const TOUCH_SCREEN_WIDTH: u32 = 1280;
pub const TOUCH_SCREEN_HEIGHT: u32 = 720;

pub const TOUCH_MAX: usize = 16;

/// The time between six-axis samples, in nanoseconds.
const SIX_AXIS_DELTA_NS: u64 = 5_000_000;

/// Contact size reported for every touch; only checked to be non-zero.
const TOUCH_DIAMETER: u32 = 10;

/// One finger in digitizer coordinates; `finger_id` is stable while it stays down.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TouchPoint {
    pub finger_id: u32,
    pub x: u32,
    pub y: u32,
}

impl Cpu {
    /// Publish host gamepad state to [`crate::INPUT_ADDR`] and, once mapped, hid shared memory.
    /// `buttons` is a `HidNpadButton` mask; sticks are -32768..32767 with up positive.
    /// Stick pseudo-buttons are derived here.
    pub fn set_gamepad_state(
        &mut self,
        buttons: u64,
        stick_lx: i32,
        stick_ly: i32,
        stick_rx: i32,
        stick_ry: i32,
    ) {
        self.last_gamepad = (buttons, stick_lx, stick_ly, stick_rx, stick_ry);
        let buttons = buttons | Self::stick_pseudo_buttons(stick_lx, stick_ly, stick_rx, stick_ry);

        // Host-to-guest register: a u64 mask, then two analog sticks.
        let _ = self.mem.write_u64(crate::INPUT_ADDR, buttons);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 8, stick_lx as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 12, stick_ly as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 16, stick_rx as u32);
        let _ = self.mem.write_u32(crate::INPUT_ADDR + 20, stick_ry as u32);

        if !self.hid_mapped {
            return;
        }
        self.write_hid_gamepad_state(buttons, stick_lx, stick_ly, stick_rx, stick_ry);
    }

    /// Take the host's motion sensor reading; it is published on the next `hid` sample.
    pub fn set_motion_state(&mut self, accel: [f32; 3], gyro: [f32; 3], connected: bool) {
        self.motion.accel = accel;
        self.motion.gyro = gyro;
        self.motion.connected = connected;
    }

    /// Publish the sensor into the six-axis LIFOs of the style the title can use.
    fn write_six_axis(&mut self) {
        use hid_shmem as h;
        if !self.hid_mapped || !(self.motion.connected || self.motion.published) {
            return;
        }
        let period = HID_SAMPLE_PERIOD_CYCLES as f32 / 1_020_000_000.0;
        self.motion.integrate(period);
        self.motion.published = self.motion.connected;
        self.motion.sample = self.motion.sample.wrapping_add(1);
        let (slot, presentation) = self.active_npad();
        let lifos: &[u32] = match presentation.style {
            _ if slot == h::HANDHELD_SLOT => &[h::SIX_AXIS_HANDHELD],
            h::STYLE_FULL_KEY => &[0],
            h::STYLE_JOY_DUAL => &[h::SIX_AXIS_JOY_DUAL_LEFT, h::SIX_AXIS_JOY_DUAL_RIGHT],
            h::STYLE_JOY_LEFT => &[h::SIX_AXIS_JOY_LEFT],
            _ => &[h::SIX_AXIS_JOY_RIGHT],
        };
        let base = h::NPAD + slot * h::ENTRY_SIZE + h::SIX_AXIS_FULL_KEY_LIFO;
        for index in lifos {
            self.write_six_axis_lifo(base + index * h::SIX_AXIS_LIFO_SIZE);
        }
    }

    fn write_six_axis_lifo(&mut self, lifo: u32) {
        use hid_shmem as h;
        let m = self.motion;
        let _ = self
            .hid_shmem
            .write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.hid_shmem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.hid_shmem.write_u64(lifo + h::LIFO_COUNT, 1);
        let entry = lifo + h::LIFO_STORAGE;
        let _ = self
            .hid_shmem
            .write_u64(entry + h::STORAGE_SAMPLING_NUMBER, m.sample << 1);
        let state = entry + h::SIX_AXIS_STATE;
        let _ = self
            .hid_shmem
            .write_u64(state + h::SIX_AXIS_DELTA_TIME, SIX_AXIS_DELTA_NS);
        let _ = self
            .hid_shmem
            .write_u64(state + h::SIX_AXIS_SAMPLING_NUMBER, m.sample);
        let floats = m
            .accel
            .iter()
            .chain(&m.gyro)
            .chain(&m.rotation)
            .chain(m.orientation.iter().flatten());
        for (i, value) in floats.enumerate() {
            let at = state + h::SIX_AXIS_FLOATS + 4 * i as u32;
            let _ = self.hid_shmem.write_u32(at, value.to_bits());
        }
        let attributes = if m.connected {
            h::SIX_AXIS_ATTR_CONNECTED
        } else {
            0
        };
        let _ = self
            .hid_shmem
            .write_u32(state + h::SIX_AXIS_ATTRIBUTES, attributes);
    }

    /// Publish a fresh `hid` sample when the 200 Hz timer comes round.
    pub(crate) fn hid_tick(&mut self) {
        if !self.hid_mapped
            || self.cycles.wrapping_sub(self.last_hid_cycles) < HID_SAMPLE_PERIOD_CYCLES
        {
            return;
        }
        self.last_hid_cycles = self.cycles;
        let (buttons, lx, ly, rx, ry) = self.last_gamepad;
        self.set_gamepad_state(buttons, lx, ly, rx, ry);
        let touches = std::mem::take(&mut self.last_touches);
        self.set_touch_state(&touches);
        self.write_six_axis();
    }

    /// `HidNpadButton_StickL*`/`StickR*` bits derived from stick deflection.
    fn stick_pseudo_buttons(lx: i32, ly: i32, rx: i32, ry: i32) -> u64 {
        let mut mask = 0u64;
        for (i, (x, y)) in [(lx, ly), (rx, ry)].iter().enumerate() {
            let base = 16 + 4 * i as u64; // StickLLeft, then StickRLeft
            if *x < -HID_STICK_THRESHOLD {
                mask |= 1 << base;
            }
            if *y > HID_STICK_THRESHOLD {
                mask |= 1 << (base + 1);
            }
            if *x > HID_STICK_THRESHOLD {
                mask |= 1 << (base + 2);
            }
            if *y < -HID_STICK_THRESHOLD {
                mask |= 1 << (base + 3);
            }
        }
        mask
    }

    /// Publish the pad into `HidSharedMemory` in the one slot and style the title
    /// can use (see [`Cpu::active_npad`]); the other slot reads as nothing connected.
    fn write_hid_gamepad_state(&mut self, buttons: u64, lx: i32, ly: i32, rx: i32, ry: i32) {
        use hid_shmem as h;
        self.sample_counter = self.sample_counter.wrapping_add(1);
        let sample = self.sample_counter;
        // Published on mapping, before the guest reads a pad.
        self.write_npad_condition();
        let (slot, presentation) = self.active_npad();
        self.write_npad_slot(slot, presentation, sample, buttons, (lx, ly, rx, ry));
        let idle = if slot == h::HANDHELD_SLOT {
            0
        } else {
            h::HANDHELD_SLOT
        };
        self.write_npad_disconnected(idle, sample);
    }

    /// The one pad there is: on the rails when undocked, else player 1, as far as the title
    /// accepts either.
    pub(crate) fn active_npad(&self) -> (u32, NpadPresentation) {
        let styles = self.npad_style_set;
        let id_accepted = |id| self.npad_id_types.is_empty() || self.npad_id_types.contains(&id);
        let handheld =
            (styles == 0 || styles & NPAD_HANDHELD.style != 0) && id_accepted(NPAD_ID_HANDHELD);
        let player_1 = id_accepted(NPAD_ID_PLAYER_1)
            && (styles == 0 || NPAD_PRESENTATIONS.iter().any(|pad| styles & pad.style != 0));
        if handheld && (self.operation_mode == OperationMode::Handheld || !player_1) {
            (hid_shmem::HANDHELD_SLOT, NPAD_HANDHELD)
        } else {
            (0, npad_presentation_for(styles))
        }
    }

    /// `nn::hid::system::AppletFooterUiType` of the pad behind an npad id; 0 is no pad.
    pub(crate) fn npad_footer_type(&self, id: u32) -> u8 {
        use hid_shmem as h;
        const HANDHELD_JOY_CON_PAIR: u8 = 4;
        const JOY_DUAL: u8 = 5;
        const JOY_LEFT_HORIZONTAL: u8 = 8;
        const JOY_RIGHT_HORIZONTAL: u8 = 10;
        const PRO_CONTROLLER: u8 = 12;
        let (slot, presentation) = self.active_npad();
        let active_id = if slot == h::HANDHELD_SLOT {
            NPAD_ID_HANDHELD
        } else {
            NPAD_ID_PLAYER_1
        };
        if id != active_id {
            return 0;
        }
        match presentation.style {
            h::STYLE_FULL_KEY => PRO_CONTROLLER,
            h::STYLE_HANDHELD => HANDHELD_JOY_CON_PAIR,
            h::STYLE_JOY_DUAL => JOY_DUAL,
            h::STYLE_JOY_LEFT => JOY_LEFT_HORIZONTAL,
            _ => JOY_RIGHT_HORIZONTAL,
        }
    }

    fn write_npad_disconnected(&mut self, slot: u32, sample: u64) {
        use hid_shmem as h;
        let base = h::NPAD.wrapping_add(slot.wrapping_mul(h::ENTRY_SIZE));
        for field in [
            h::STYLE_SET,
            h::JOY_ASSIGNMENT_MODE,
            h::DEVICE_TYPE,
            h::SYSTEM_PROPERTIES,
        ] {
            let _ = self.hid_shmem.write_u32(base + field, 0);
        }
        for lifo in [
            h::FULL_KEY_LIFO,
            h::HANDHELD_LIFO,
            h::JOY_DUAL_LIFO,
            h::JOY_LEFT_LIFO,
            h::JOY_RIGHT_LIFO,
            h::SYSTEM_EXT_LIFO,
        ] {
            self.write_npad_lifo(base.wrapping_add(lifo), sample, 0, (0, 0, 0, 0), 0);
        }
    }

    /// Publish `nn::hid::NpadCondition`, including the stored joy-con hold type.
    pub(crate) fn write_npad_condition(&mut self) {
        use hid_shmem as h;
        if !self.hid_mapped {
            return;
        }
        let at = h::NPAD_CONDITION;
        let _ = self
            .hid_shmem
            .write_u32(at + h::NPAD_CONDITION_INITIALIZED, 1);
        let _ = self.hid_shmem.write_u32(
            at + h::NPAD_CONDITION_HOLD_TYPE,
            self.npad_joy_hold_type as u32,
        );
        let _ = self.hid_shmem.write_u32(at + h::NPAD_CONDITION_VALID, 1);
    }

    fn write_npad_slot(
        &mut self,
        slot: u32,
        presentation: NpadPresentation,
        sample: u64,
        buttons: u64,
        sticks: (i32, i32, i32, i32),
    ) {
        use hid_shmem as h;
        let base = h::NPAD.wrapping_add(slot.wrapping_mul(h::ENTRY_SIZE));
        let _ = self
            .hid_shmem
            .write_u32(base + h::STYLE_SET, presentation.style);
        let _ = self
            .hid_shmem
            .write_u32(base + h::JOY_ASSIGNMENT_MODE, presentation.joy_assignment);
        let _ = self
            .hid_shmem
            .write_u32(base + h::DEVICE_TYPE, presentation.device_type);

        // Report external power and a full battery for the pad and each half; zero reads as flat.
        let _ = self.hid_shmem.write_u32(
            base + h::SYSTEM_PROPERTIES,
            h::SYSTEM_PROP_POWER_CONNECTED | h::SYSTEM_PROP_FULL_BUTTONS,
        );
        for info in 0..h::POWER_INFO_COUNT {
            let _ = self
                .hid_shmem
                .write_u32(base + h::BATTERY_LEVEL + info * 4, h::BATTERY_FULL);
        }

        self.write_npad_lifo(
            base.wrapping_add(presentation.lifo),
            sample,
            buttons,
            sticks,
            presentation.attributes,
        );
        // SystemExt: a second copy every pad carries; the Home Menu reads only this LIFO.
        self.write_npad_lifo(
            base.wrapping_add(h::SYSTEM_EXT_LIFO),
            sample,
            buttons,
            sticks,
            presentation.attributes,
        );
    }

    /// Publish one state into a `HidNpadCommonLifo`. The sampling number is doubled
    /// because bit 0 is the seqlock's "being written" flag.
    fn write_npad_lifo(
        &mut self,
        lifo: u32,
        sample: u64,
        buttons: u64,
        sticks: (i32, i32, i32, i32),
        attributes: u32,
    ) {
        use hid_shmem as h;
        let (lx, ly, rx, ry) = sticks;
        let _ = self
            .hid_shmem
            .write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.hid_shmem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.hid_shmem.write_u64(lifo + h::LIFO_COUNT, 1);

        let entry = lifo.wrapping_add(h::LIFO_STORAGE);
        let _ = self
            .hid_shmem
            .write_u64(entry + h::STORAGE_SAMPLING_NUMBER, sample << 1);
        let _ = self
            .hid_shmem
            .write_u64(entry + h::STATE_SAMPLING_NUMBER, sample);
        let _ = self.hid_shmem.write_u64(entry + h::STATE_BUTTONS, buttons);
        let _ = self
            .hid_shmem
            .write_u32(entry + h::STATE_STICK_L, lx as u32);
        let _ = self
            .hid_shmem
            .write_u32(entry + h::STATE_STICK_L + 4, ly as u32);
        let _ = self
            .hid_shmem
            .write_u32(entry + h::STATE_STICK_R, rx as u32);
        let _ = self
            .hid_shmem
            .write_u32(entry + h::STATE_STICK_R + 4, ry as u32);
        let _ = self
            .hid_shmem
            .write_u32(entry + h::STATE_ATTRIBUTES, attributes);
    }

    /// Publish touchscreen contacts for `hidGetTouchScreenStates`. New ids get
    /// `start_touch`; lifted ids are published once more with `end_touch`. Docked, no
    /// contacts are reported, but the sample still advances.
    pub fn set_touch_state(&mut self, touches: &[TouchPoint]) {
        self.last_touches = touches.to_vec();
        if !self.hid_mapped {
            return;
        }
        use hid_shmem as h;
        self.touch_sample_counter = self.touch_sample_counter.wrapping_add(1);
        let sample = self.touch_sample_counter;

        let lifo = h::TOUCH_SCREEN;
        let _ = self
            .hid_shmem
            .write_u64(lifo + h::LIFO_BUFFER_COUNT, h::LIFO_CAPACITY);
        let _ = self.hid_shmem.write_u64(lifo + h::LIFO_TAIL, 0);
        let _ = self.hid_shmem.write_u64(lifo + h::LIFO_COUNT, 1);

        let storage = lifo.wrapping_add(h::LIFO_STORAGE);
        let _ = self
            .hid_shmem
            .write_u64(storage + h::STORAGE_SAMPLING_NUMBER, sample << 1);
        let state = storage.wrapping_add(h::TOUCH_STATE);
        let _ = self
            .hid_shmem
            .write_u64(state + h::TOUCH_SAMPLING_NUMBER, sample);

        // Docked: every contact is gone.
        let down: &[TouchPoint] = match self.operation_mode {
            OperationMode::Handheld => touches,
            OperationMode::Docked => &[],
        };
        // Contacts down now, then those lifted since the last sample.
        let mut published: Vec<(TouchPoint, u32)> = Vec::with_capacity(TOUCH_MAX);
        for touch in down.iter().take(TOUCH_MAX) {
            let held = self
                .touch_down
                .iter()
                .any(|prev| prev.finger_id == touch.finger_id);
            let attributes = if held { 0 } else { h::TOUCH_ATTR_START };
            published.push((*touch, attributes));
        }
        for prev in &self.touch_down {
            if published.len() == TOUCH_MAX {
                break;
            }
            if !down.iter().any(|touch| touch.finger_id == prev.finger_id) {
                published.push((*prev, h::TOUCH_ATTR_END));
            }
        }
        self.touch_down = down.iter().take(TOUCH_MAX).copied().collect();

        let count = published.len();
        let _ = self
            .hid_shmem
            .write_u32(state + h::TOUCH_COUNT, count as u32);
        let slot = |i: usize| state + h::TOUCH_TOUCHES + i as u32 * h::TOUCH_SIZE;
        for (i, (touch, attributes)) in published.iter().enumerate() {
            let e = slot(i);
            // delta_time is not measured.
            let _ = self.hid_shmem.write_u64(e + h::TOUCH_DELTA_TIME, 0);
            let _ = self
                .hid_shmem
                .write_u32(e + h::TOUCH_ATTRIBUTES, *attributes);
            let _ = self
                .hid_shmem
                .write_u32(e + h::TOUCH_FINGER_ID, touch.finger_id);
            let _ = self
                .hid_shmem
                .write_u32(e + h::TOUCH_X, touch.x.min(TOUCH_SCREEN_WIDTH - 1));
            let _ = self
                .hid_shmem
                .write_u32(e + h::TOUCH_Y, touch.y.min(TOUCH_SCREEN_HEIGHT - 1));
            let _ = self
                .hid_shmem
                .write_u32(e + h::TOUCH_DIAMETER_X, TOUCH_DIAMETER);
            let _ = self
                .hid_shmem
                .write_u32(e + h::TOUCH_DIAMETER_Y, TOUCH_DIAMETER);
            let _ = self.hid_shmem.write_u32(e + h::TOUCH_ROTATION_ANGLE, 0);
        }
        // Clear slots a previous larger sample left.
        for i in count..self.touch_published {
            let e = slot(i);
            for off in (0..h::TOUCH_SIZE).step_by(4) {
                let _ = self.hid_shmem.write_u32(e + off, 0);
            }
        }
        self.touch_published = count;
    }

    /// Rumble `(low, high)` amplitudes in 0.0..=1.0.
    pub fn vibration(&self) -> (f32, f32) {
        self.vibration
    }

    pub(crate) fn set_vibration(&mut self, low: f32, high: f32) {
        let clamp = |v: f32| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        self.vibration = (clamp(low), clamp(high));
    }

    pub fn hid_shmem_addr(&self) -> u32 {
        self.process().hid_shmem_addr
    }
}
