//! Audio output queue and host-facing audio activity.

use super::*;

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
    /// `(sample_rate, channels)` of [`Cpu::take_audio`]'s samples; `(0, 0)` before a device opens.
    pub fn audio_format(&self) -> (u32, u32) {
        self.audio_format
    }

    /// Move up to `out.len()` queued interleaved samples into `out`; returns the count.
    pub fn take_audio(&mut self, out: &mut [i16]) -> usize {
        let n = out.len().min(self.audio_pcm.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.audio_pcm.pop_front().unwrap_or(0);
        }
        self.audio_taken += n as u64;
        n
    }

    /// Every open audio device and renderer, and the samples queued for the host.
    pub fn audio_activity(&self) -> AudioActivity {
        let mut outputs: Vec<AudioOutActivity> = self
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
            backlog: self.audio_pcm.len() as u64,
            outputs,
            renderers,
        }
    }

    /// Queue interleaved PCM, dropping the oldest past [`Cpu::AUDIO_QUEUE_LIMIT`].
    pub(crate) fn queue_audio(&mut self, samples: impl Iterator<Item = i16>) {
        let before = self.audio_pcm.len();
        self.audio_pcm.extend(samples);
        self.audio_produced += (self.audio_pcm.len() - before) as u64;
        let over = self.audio_pcm.len().saturating_sub(Self::AUDIO_QUEUE_LIMIT);
        self.audio_pcm.drain(..over);
        self.audio_dropped += over as u64;
    }

    /// About a second of 48 kHz stereo.
    pub(crate) const AUDIO_QUEUE_LIMIT: usize = 48_000 * 2;
}
