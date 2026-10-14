//! Audio output.

use std::sync::atomic::AtomicI16;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use crate::session;

/// The PCM format the guest produces, packed as `(channels << 24) | sample_rate`.
/// Zero until the guest opens an audio device.
#[no_mangle]
pub extern "C" fn switch_audio_format(handle: u32) -> u32 {
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

/// The ring's address and capacity in samples, for the worklet.
#[no_mangle]
pub extern "C" fn switch_audio_ring() -> u32 {
    &RING as *const Ring as u32
}

#[no_mangle]
pub extern "C" fn switch_audio_ring_samples() -> u32 {
    RING_SAMPLES as u32
}

/// Move what the guest queued into the ring, as far as it has room.
#[no_mangle]
pub extern "C" fn switch_audio_feed(handle: u32) {
    let s = session(handle);
    RING.format.store(switch_audio_format(handle), Relaxed);
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
