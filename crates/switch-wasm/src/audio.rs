//! Audio output.

use crate::session;

/// The PCM format `switch_audio_pull` returns, packed as `(channels << 24) | sample_rate`.
/// Zero until the guest opens an audio device.
#[no_mangle]
pub extern "C" fn switch_audio_format(handle: u32) -> u32 {
    let (rate, channels) = session(handle).cpu.audio_format();
    if rate == 0 {
        return 0;
    }
    (channels << 24) | (rate & 0x00ff_ffff)
}

/// Move up to `max_samples` interleaved 16-bit samples into `buf`, returning the count.
#[no_mangle]
pub extern "C" fn switch_audio_pull(handle: u32, buf: *mut u8, max_samples: u32) -> u32 {
    let s = session(handle);
    let mut samples = vec![0i16; max_samples as usize];
    let n = s.cpu.take_audio(&mut samples);
    let out = unsafe { std::slice::from_raw_parts_mut(buf, n * 2) };
    for (chunk, sample) in out.as_chunks_mut::<2>().0.iter_mut().zip(samples.iter()) {
        *chunk = sample.to_le_bytes();
    }
    n as u32
}
