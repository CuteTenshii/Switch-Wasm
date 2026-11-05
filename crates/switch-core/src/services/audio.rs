//! Audio mixing and host-facing audio activity.

use crate::cpu::*;
use std::collections::VecDeque;

/// The rate of [`Cpu::take_audio`]'s samples.
pub(crate) const OUTPUT_RATE: u32 = 48_000;
/// Interleaved channels of [`Cpu::take_audio`]'s samples.
const OUTPUT_CHANNELS: usize = 2;

/// Who produced a stream of samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AudioSource {
    Out(u64),
    Renderer(u64),
}

/// One source's samples, already stereo at [`OUTPUT_RATE`], waiting to be mixed.
#[derive(Debug, Default)]
pub(crate) struct AudioStream {
    pcm: VecDeque<i16>,
    /// The previous input frame and the next output's position past it, in input frames.
    last: Option<[f32; 2]>,
    position: f64,
}

impl AudioStream {
    fn push(&mut self, rate: u32, channels: usize, samples: impl Iterator<Item = i16>) {
        let samples: Vec<i16> = samples.collect();
        let frames = samples.chunks_exact(channels).map(downmix);
        if rate == OUTPUT_RATE {
            self.pcm.extend(frames.flatten());
            return;
        }
        let step = f64::from(rate) / f64::from(OUTPUT_RATE);
        for next in frames {
            let next = next.map(f32::from);
            let Some(last) = self.last.replace(next) else {
                continue;
            };
            while self.position < 1.0 {
                let t = self.position as f32;
                self.pcm.extend(std::array::from_fn::<_, 2, _>(|c| {
                    (last[c] + (next[c] - last[c]) * t).round() as i16
                }));
                self.position += step;
            }
            self.position -= 1.0;
        }
    }
}

/// One frame of `channels` samples as stereo; 5.1 folds the centre and rears into both sides.
fn downmix(frame: &[i16]) -> [i16; 2] {
    let at = |c: usize| i32::from(frame[c]);
    let clamp = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    match frame.len() {
        1 => [frame[0]; 2],
        0..=5 => [frame[0], frame[1]],
        _ => {
            let side = |front: usize, rear: usize| {
                clamp(
                    at(front)
                        + ((at(2) + at(rear)) as f32 * std::f32::consts::FRAC_1_SQRT_2) as i32,
                )
            };
            [side(0, 4), side(1, 5)]
        }
    }
}

/// What [`Cpu::audio_activity`] reports; sample counts are interleaved samples since session start.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioActivity {
    /// 0 before anything has played.
    pub sample_rate: u32,
    pub channels: u32,
    pub produced: u64,
    pub taken: u64,
    pub dropped: u64,
    pub backlog: u64,
    pub outputs: Vec<AudioOutActivity>,
    pub renderers: Vec<AudioRendererActivity>,
}

/// One open `audout` device; counts run from when it was opened.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOutActivity {
    pub handle: u64,
    pub sample_rate: u32,
    pub channels: u32,
    pub started: bool,
    pub volume: f32,
    pub appended_buffers: u64,
    pub appended_frames: u64,
    pub released_buffers: u64,
    /// Appended and not yet handed back.
    pub pending_buffers: u64,
    /// Frames appended while stopped, which never play.
    pub discarded_frames: u64,
    /// Buffers whose descriptor pointed outside itself; see `audio_out_append`.
    pub unplayable_buffers: u64,
}

/// One open audio renderer; counts run from when it was opened.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioRendererActivity {
    pub handle: u64,
    pub sample_rate: u32,
    pub started: bool,
    pub updates: u64,
    pub rendered_frames: u64,
    pub voices: u32,
    pub voices_playing: u32,
    /// 0 when no playable sink is configured.
    pub sink_channels: u32,
}

impl Cpu {
    /// `(sample_rate, channels)` of [`Cpu::take_audio`]'s samples; `(0, 0)` before anything plays.
    pub fn audio_format(&self) -> (u32, u32) {
        self.audio_format
    }

    /// Mix up to `out.len()` queued interleaved samples of every stream into `out`; returns the
    /// count, which is the longest any stream had.
    pub fn take_audio(&mut self, out: &mut [i16]) -> usize {
        let mut mixed = vec![0i32; out.len()];
        let mut n = 0;
        for stream in self.audio_streams.values_mut() {
            let taken = out.len().min(stream.pcm.len());
            for (sum, sample) in mixed.iter_mut().zip(stream.pcm.drain(..taken)) {
                *sum += i32::from(sample);
            }
            n = n.max(taken);
        }
        self.audio_streams
            .retain(|_, stream| !stream.pcm.is_empty());
        for (slot, sum) in out.iter_mut().zip(mixed).take(n) {
            *slot = sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        self.audio_taken += n as u64;
        n
    }

    /// Every open audio device and renderer, and the samples queued for the host.
    pub fn audio_activity(&self) -> AudioActivity {
        let mut outputs: Vec<AudioOutActivity> = self
            .process()
            .audio_outs
            .iter()
            .map(|(&handle, device)| AudioOutActivity {
                handle,
                sample_rate: device.sample_rate,
                channels: device.channel_count,
                started: device.started,
                volume: device.volume,
                appended_buffers: device.appended_buffers,
                appended_frames: device.appended_frames,
                released_buffers: device.released_buffers,
                pending_buffers: device.queued.len() as u64,
                discarded_frames: device.discarded_frames,
                unplayable_buffers: device.unplayable_buffers,
            })
            .collect();
        outputs.sort_by_key(|device| device.handle);
        let mut renderers: Vec<AudioRendererActivity> = self
            .process()
            .audren_renderers
            .iter()
            .map(|(&handle, renderer)| AudioRendererActivity {
                handle,
                sample_rate: renderer.sample_rate,
                started: renderer.started,
                updates: renderer.updates,
                rendered_frames: renderer.elapsed_frames,
                voices: renderer.voices.len() as u32,
                voices_playing: renderer
                    .voices
                    .iter()
                    .filter(|voice| voice.in_use && voice.playing && voice.remaining > 0)
                    .count() as u32,
                sink_channels: renderer.sink.as_ref().map_or(0, |sink| sink.channels),
            })
            .collect();
        renderers.sort_by_key(|renderer| renderer.handle);
        AudioActivity {
            sample_rate: self.audio_format.0,
            channels: self.audio_format.1,
            produced: self.audio_produced,
            taken: self.audio_taken,
            dropped: self.audio_dropped,
            backlog: self
                .audio_streams
                .values()
                .map(|s| s.pcm.len() as u64)
                .sum(),
            outputs,
            renderers,
        }
    }

    /// Queue a source's interleaved PCM to be mixed, dropping its oldest past [`Cpu::AUDIO_QUEUE_LIMIT`].
    pub(crate) fn queue_audio(
        &mut self,
        source: AudioSource,
        rate: u32,
        channels: u32,
        samples: impl Iterator<Item = i16>,
    ) {
        self.audio_format = (OUTPUT_RATE, OUTPUT_CHANNELS as u32);
        let stream = self.audio_streams.entry(source).or_default();
        let before = stream.pcm.len();
        stream.push(rate.max(1), channels.max(1) as usize, samples);
        let pushed = stream.pcm.len() - before;
        let over = stream.pcm.len().saturating_sub(Self::AUDIO_QUEUE_LIMIT);
        stream.pcm.drain(..over);
        self.audio_produced += pushed as u64;
        self.audio_dropped += over as u64;
    }

    /// About a second of 48 kHz stereo, per stream.
    pub(crate) const AUDIO_QUEUE_LIMIT: usize = OUTPUT_RATE as usize * OUTPUT_CHANNELS;
}
