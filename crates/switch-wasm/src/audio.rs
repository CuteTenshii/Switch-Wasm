//! Audio output.

use std::sync::atomic::AtomicI16;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use crate::session;

/// The PCM format the guest produces, packed as `(channels << 24) | sample_rate`.
/// Zero until the guest opens an audio device.
fn format(handle: u32) -> u32 {
    let (rate, channels) = session(handle).cpu.audio_format();
    if rate == 0 {
        return 0;
    }
    (channels << 24) | (rate & 0x00ff_ffff)
}

/// Samples the ring holds, a power of two: about 0.68 s of 48 kHz stereo.
const RING_SAMPLES: usize = 1 << 16;

/// Where audio goes: a ring in linear memory that the page's audio worklet reads over the
/// shared buffer, so no message carries samples. The worklet advances `read`.
#[repr(C)]
struct Ring {
    written: AtomicU32,
    read: AtomicU32,
    /// `(channels << 24) | rate` of what is in the ring; 0 before a device opens.
    format: AtomicU32,
    samples: [AtomicI16; RING_SAMPLES],
}

static RING: Ring = Ring {
    written: AtomicU32::new(0),
    read: AtomicU32::new(0),
    format: AtomicU32::new(0),
    samples: [const { AtomicI16::new(0) }; RING_SAMPLES],
};

/// Hand the page the ring over the shared memory, for its audio worklet to read.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub(crate) fn announce_ring() {
    use js_sys::{Object, Reflect, WebAssembly};
    use wasm_bindgen::JsCast;
    let memory: WebAssembly::Memory = wasm_bindgen::memory().unchecked_into();
    let message = Object::new();
    for (key, value) in [
        ("type", "audio_ring".into()),
        ("buffer", memory.buffer()),
        ("at", (&RING as *const Ring as u32).into()),
        ("samples", (RING_SAMPLES as u32).into()),
    ] {
        Reflect::set(&message, &key.into(), &value).ok();
    }
    post(&message);
}

/// Tell the page the format changed, so it opens the output on the first sound.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
fn announce_format(format: u32) {
    use js_sys::{Object, Reflect};
    let message = Object::new();
    Reflect::set(&message, &"type".into(), &"audio_format".into()).ok();
    Reflect::set(&message, &"format".into(), &format.into()).ok();
    post(&message);
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
fn post(message: &js_sys::Object) {
    use wasm_bindgen::JsCast;
    let scope: web_sys::DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    scope.post_message(message).ok();
}

/// Drop what is queued in the ring, for a title that has ended.
pub(crate) fn drop_queued() {
    RING.read.store(RING.written.load(Acquire), Release);
}

/// Move what the guest queued into the ring, as far as it has room.
pub(crate) fn feed(handle: u32) {
    let s = session(handle);
    let format = format(handle);
    #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
    if RING.format.swap(format, Relaxed) != format {
        announce_format(format);
    }
    #[cfg(not(all(feature = "gpu", target_arch = "wasm32")))]
    RING.format.store(format, Relaxed);
    let written = RING.written.load(Relaxed);
    let free = RING_SAMPLES - written.wrapping_sub(RING.read.load(Acquire)) as usize;
    const CHUNK: usize = 4096;
    let mut chunk = [0i16; CHUNK];
    let mut moved = 0;
    while moved < free {
        let n = s.cpu.take_audio(&mut chunk[..(free - moved).min(CHUNK)]);
        if n == 0 {
            break;
        }
        for (i, &sample) in chunk[..n].iter().enumerate() {
            let at = written.wrapping_add((moved + i) as u32) as usize & (RING_SAMPLES - 1);
            RING.samples[at].store(sample, Relaxed);
        }
        moved += n;
    }
    RING.written
        .store(written.wrapping_add(moved as u32), Release);
}
