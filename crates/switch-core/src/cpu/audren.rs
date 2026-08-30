//! `audren`, the audio renderer: the path from a title's voices to the host's
//! speakers.
//!
//! `audout` (in `ipc.rs`) is a *device* — the guest hands it finished PCM and
//! it plays it. The renderer is a *mixer*, and what the guest hands it is
//! sources: wave buffers of PCM or ADPCM, a pitch, a volume and a routing
//! matrix, re-sent in full once every 5 ms. Nearly every retail title reaches
//! audio this way, through `nn::audio` or libnx's `audrv`; `audout` is mostly
//! the homebrew path.
//!
//! One `RequestUpdateAudioRenderer` carries the whole renderer state — every
//! mempool, channel resource, voice, mix and sink — as one flat buffer whose
//! header declares the size of each section. The reply is the mirror of it,
//! and the caller walks that reply section by section against sizes it
//! computed itself, so both halves have to agree exactly. See
//! [`Cpu::audren_write_update_reply`].
//!
//! The renderer runs on a clock, for the same reason [`crate::cpu::Cpu::audio_tick`]
//! makes `audout` release buffers on one: a title schedules against how fast
//! its audio drains, and a mixer that renders as fast as the guest can ask
//! runs its clock at whatever multiple of real time the emulator happens to
//! manage.

use super::power::CLOCK_RATES_HZ;
use super::Cpu;
use crate::mem::Memory;
use crate::Result;

/// The renderer produces one frame every 5 ms — `AUDREN_TIMER_FREQ_HZ`, 200 Hz
/// — at whatever rate it was opened with. 240 samples at 48 kHz, 160 at 32.
const FRAMES_PER_SECOND: u64 = 200;

/// A renderer frame in the emulated cycles that are this machine's only clock,
/// counted the same way [`Cpu::audio_play_cycles`] counts a device's buffer.
const FRAME_CYCLES: u64 = CLOCK_RATES_HZ[0] as u64 / FRAMES_PER_SECOND;

/// How far behind the renderer is allowed to fall before the backlog is
/// dropped rather than paid off. A guest that spent half a second loading owes
/// no half-second of audio: what it did not ask to render was never heard, and
/// rendering it now only puts every later frame that much further behind.
const MAX_CATCHUP_FRAMES: u64 = 8;

/// `AudioRendererChannelInfoIn::mix` is 24 factors long — one per mix buffer
/// its destination mix can have.
const MAX_MIX_BUFFERS: usize = 24;

/// `AudioRendererVoiceInfoIn::channel_ids`.
const MAX_VOICE_CHANNELS: usize = 6;

/// The wave-buffer ring is four deep and the guest's head index wraps in it.
const WAVE_BUFFERS: usize = 4;

/// `AudioRendererDeviceSinkInfoIn::inputs`.
const MAX_SINK_INPUTS: usize = 6;

/// Section strides in the update *input*, from libnx's `audren.h`. The reply's
/// own sizes live in [`Cpu::audren_write_update_reply`] — an input entry and
/// an output entry for the same object are different sizes.
const HEADER_SZ: u32 = 0x40;
const MEMPOOL_IN_SZ: u32 = 0x20;
const CHANNEL_IN_SZ: u32 = 0x70;
const VOICE_IN_SZ: u32 = 0x170;
const WAVE_BUF_IN_SZ: u32 = 0x38;

/// `PcmFormat`, from libnx's `audio.h`.
const PCM_INT8: u8 = 1;
const PCM_INT16: u8 = 2;
const PCM_INT24: u8 = 3;
const PCM_INT32: u8 = 4;
const PCM_FLOAT: u8 = 5;
const PCM_ADPCM: u8 = 6;

/// `AudioRendererVoicePlayState`.
const PLAY_STATE_STARTED: u8 = 0;

/// `AudioRendererMemPoolState`. A pool the guest asked to attach comes back
/// `Attached`, one it asked to detach comes back `Detached`, and anything else
/// comes back `Invalid` — which means "unchanged", not "broken".
const MEMPOOL_REQUEST_DETACH: u32 = 2;
const MEMPOOL_DETACHED: u32 = 3;
const MEMPOOL_REQUEST_ATTACH: u32 = 4;
const MEMPOOL_ATTACHED: u32 = 5;

/// `AudioRendererSinkType_Device`: the sink that plays through the console's
/// output, and so through the host's.
const SINK_TYPE_DEVICE: u8 = 1;

/// `AudioRendererSinkType_CircularBuffer`: a sink that writes the mix into a
/// ring in guest memory instead of playing it.
const SINK_TYPE_CIRCULAR: u8 = 2;

/// A mix id meaning "not routed anywhere" — `AUDREN_UNUSED_MIX_ID`.
const UNUSED_MIX_ID: u32 = 0x7FFF_FFFF;

/// Nintendo's 4-bit ADPCM packs 14 samples into every 8 bytes: one header byte
/// carrying the scale and coefficient index, then seven bytes of nibbles.
const ADPCM_SAMPLES_PER_FRAME: u32 = 14;
const ADPCM_BYTES_PER_FRAME: u32 = 8;

/// The fixed-point shift the ADPCM predictor's coefficients are in.
const ADPCM_COEF_SHIFT: i64 = 11;

/// One `IAudioRenderer` session: the counts `OpenAudioRenderer` fixed for its
/// lifetime, and everything the guest has told it about since.
///
/// The counts are not bookkeeping — they are what every later
/// `RequestUpdateAudioRenderer` reply has to be sized against, because
/// `audrvUpdate` and `nnSdk` both compute the same sizes from the same numbers
/// and reject a reply that disagrees.
#[derive(Debug, Clone)]
pub(crate) struct AudioRenderer {
    /// The `REVn` magic the renderer was opened with, echoed into every reply,
    /// and the parsed revision number that decides which sections the reply
    /// carries.
    pub revision_magic: u32,
    pub revision: u32,
    /// The rate the mix is produced at, and how many frames one 5 ms render
    /// frame is: 48 kHz/240, or 32 kHz/160.
    pub sample_rate: u32,
    pub sample_count: u32,
    /// How many mix buffers exist across every mix object.
    pub mix_buffer_count: u32,
    pub voice_count: u32,
    pub sink_count: u32,
    pub effect_count: u32,
    pub submix_count: u32,
    /// Whether `StopAudioRenderer` has been called.
    ///
    /// Open renderers start *started*. `StartAudioRenderer` exists, but libnx
    /// never calls it — `audrenInitialize` opens the renderer, queries the
    /// frame event and returns — so a renderer that only produced sound once
    /// started would be silent for every libnx title.
    pub started: bool,
    /// The event `QuerySystemEvent` handed out, fired once per rendered frame.
    /// This is what `audrenWaitFrame` blocks on, and firing it on the clock is
    /// what paces a title's mixer to real time.
    pub frame_event: Option<u64>,
    /// The cycle the next frame is due to be signalled at.
    pub next_frame_at: u64,
    /// The cycle the mix has been rendered up to. What separates it from
    /// `next_frame_at` is that the event fires from a wait and the samples are
    /// produced from an update, and a guest need not interleave the two.
    pub rendered_through: u64,
    /// Frames rendered since the renderer was opened, which is what the
    /// `RendererInfoOut` tail reports.
    pub elapsed_frames: u64,
    /// Per-voice playback state, indexed by voice id. Rebuilt from the update
    /// every frame except for the parts only the renderer knows — where in its
    /// wave buffer each voice has got to.
    pub voices: Vec<Voice>,
    /// The state each mempool should be reported as this update, or 0 for
    /// "unchanged". Guest memory is the renderer's memory here, so attaching a
    /// pool is bookkeeping the guest can see and nothing more.
    pub mempool_states: Vec<u32>,
    /// Per-channel-resource mix factors: `mix[i]` is the gain from this
    /// channel into buffer `i` of the destination mix.
    pub channels: Vec<ChannelResource>,
    /// The mix objects, in the order the update sent them.
    pub mixes: Vec<Mix>,
    /// The sink that plays: its output channel count and which mix buffer
    /// feeds each of those channels.
    pub sink: Option<DeviceSink>,
    /// Whether a sink type this renderer cannot play has already been reported.
    pub warned_unplayable_sink: bool,
}

/// A voice's routing: one gain per mix buffer of the mix it plays into.
#[derive(Debug, Clone)]
pub(crate) struct ChannelResource {
    pub is_used: bool,
    pub mix: [f32; MAX_MIX_BUFFERS],
}

impl Default for ChannelResource {
    fn default() -> Self {
        ChannelResource {
            is_used: false,
            mix: [0.0; MAX_MIX_BUFFERS],
        }
    }
}

/// One mix object: a group of mix buffers, and where they go when they are
/// done. The final mix (id 0) goes to the sink; a submix goes to another mix
/// through its own matrix.
#[derive(Debug, Clone, Default)]
pub(crate) struct Mix {
    pub is_used: bool,
    pub mix_id: u32,
    pub volume: f32,
    pub buffer_count: u32,
    /// Where this mix's buffers start in the renderer's flat buffer array.
    /// Assigned in mix-id order, which is how the buffers were handed out:
    /// the final mix takes the first `buffer_count` of them and each submix
    /// the next range.
    pub buffer_offset: u32,
    pub dest_mix_id: u32,
    /// `mix[src][dest]`, the gain from this mix's buffer `src` into buffer
    /// `dest` of `dest_mix_id`.
    pub matrix: Vec<f32>,
}

/// The device sink, as the renderer plays it: how many output channels, and
/// which mix buffer each of them reads.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeviceSink {
    pub channels: u32,
    pub inputs: [u8; MAX_SINK_INPUTS],
}

/// One wave buffer as the guest queued it. Offsets are in samples per channel,
/// not bytes: what a sample *is* depends on the voice's format.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WaveBuf {
    pub address: u32,
    pub size: u32,
    pub start: u32,
    pub end: u32,
    pub looping: bool,
    pub end_of_stream: bool,
    /// Where the ADPCM decoder's history for this buffer is, so a loop can
    /// restart from the state the encoder left rather than from silence.
    pub context: u32,
}

/// The running state of Nintendo's ADPCM predictor: the two previous output
/// samples, and which sample index they are the history *for*.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AdpcmState {
    pub history0: i16,
    pub history1: i16,
    /// The next sample index this history is valid for, or `None` when the
    /// decoder has to re-seed. ADPCM is differential, so a sample can only be
    /// decoded from the one before it.
    pub next: Option<u32>,
}

/// One biquad section's stored coefficients, in the Q14 form
/// `audrvVoiceSetBiquadFilter` writes them in.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Biquad {
    pub enabled: bool,
    pub numerator: [i16; 3],
    pub denominator: [i16; 2],
}

/// A biquad's per-channel delay line, in the transposed direct form II the
/// renderer's filters are defined in.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BiquadState {
    pub s0: f32,
    pub s1: f32,
}

/// One voice: a source of samples, and where it has got to in them.
///
/// Everything down to `wavebufs` is replaced wholesale by every update — the
/// guest re-sends its whole voice array every frame. Everything below it is
/// the renderer's own, and survives, because "where in this wave buffer am I"
/// is the one thing the guest cannot tell it.
#[derive(Debug, Clone)]
pub(crate) struct Voice {
    pub in_use: bool,
    pub playing: bool,
    pub format: u8,
    pub sample_rate: u32,
    pub channel_count: u32,
    pub pitch: f32,
    pub volume: f32,
    pub dest_mix_id: u32,
    pub channel_ids: [u32; MAX_VOICE_CHANNELS],
    pub wavebufs: [WaveBuf; WAVE_BUFFERS],
    pub biquads: [Biquad; 2],
    /// The ADPCM coefficient table, from the voice's extra parameters.
    pub adpcm_coefficients: [i16; 16],
    /// Which slot of the ring is playing, and how many buffers from it are
    /// still valid. Both are re-seeded from the update's `wavebuf_head` and
    /// `wavebuf_count`, which the guest advances by the consumed count the
    /// last reply reported — so the two stay in step without either side
    /// telling the other where it is.
    pub slot: usize,
    pub remaining: u32,
    /// Samples consumed out of the wave buffer in `slot`.
    pub offset: u32,
    /// The resampler's position between `prev` and `cur`, in samples.
    pub frac: f32,
    pub prev: [f32; MAX_VOICE_CHANNELS],
    pub cur: [f32; MAX_VOICE_CHANNELS],
    /// Whether `prev`/`cur` hold real samples yet.
    pub primed: bool,
    pub adpcm: AdpcmState,
    pub biquad_state: [[BiquadState; 2]; MAX_VOICE_CHANNELS],
    /// Cumulative, and reported back every update: the guest reads
    /// `num_wavebufs_consumed` to know which of its buffers it may refill, and
    /// a voice that never reports one consumed is a voice whose title runs out
    /// of buffers and stops.
    pub played_samples: u64,
    pub wavebufs_consumed: u32,
    pub drops: u32,
}

impl Default for Voice {
    fn default() -> Self {
        Voice {
            in_use: false,
            playing: false,
            format: 0,
            sample_rate: 0,
            channel_count: 0,
            pitch: 1.0,
            volume: 1.0,
            dest_mix_id: UNUSED_MIX_ID,
            channel_ids: [0; MAX_VOICE_CHANNELS],
            wavebufs: [WaveBuf::default(); WAVE_BUFFERS],
            biquads: [Biquad::default(); 2],
            adpcm_coefficients: [0; 16],
            slot: 0,
            remaining: 0,
            offset: 0,
            frac: 0.0,
            prev: [0.0; MAX_VOICE_CHANNELS],
            cur: [0.0; MAX_VOICE_CHANNELS],
            primed: false,
            adpcm: AdpcmState::default(),
            biquad_state: [[BiquadState::default(); 2]; MAX_VOICE_CHANNELS],
            played_samples: 0,
            wavebufs_consumed: 0,
            drops: 0,
        }
    }
}

impl Voice {
    /// Forget where this voice was. `is_new` means the guest has just built
    /// the voice on a slot whatever was there before had finished with, so
    /// none of the old position, history or filter state describes it.
    fn restart(&mut self) {
        self.slot = 0;
        self.remaining = 0;
        self.offset = 0;
        self.frac = 0.0;
        self.prev = [0.0; MAX_VOICE_CHANNELS];
        self.cur = [0.0; MAX_VOICE_CHANNELS];
        self.primed = false;
        self.adpcm = AdpcmState::default();
        self.biquad_state = [[BiquadState::default(); 2]; MAX_VOICE_CHANNELS];
        self.played_samples = 0;
        self.wavebufs_consumed = 0;
        self.drops = 0;
    }
}

/// The version number in an `AudioRendererParameter`'s revision magic —
/// `REV1`, `REV2`, … — or 0 for anything that is not one. The count runs past
/// nine into the next ASCII characters (`REV:` is 10), which is why this
/// subtracts rather than parsing a digit.
///
/// The number decides the reply's shape: revision 5 added the renderer-info
/// tail and revision 9 widened an effect's status.
pub(crate) fn audren_revision(magic: u32) -> u32 {
    let [r, e, v, version] = magic.to_le_bytes();
    if [r, e, v] == *b"REV" {
        u32::from(version.wrapping_sub(b'0'))
    } else {
        0
    }
}

/// Sign-extend the low four bits of an ADPCM nibble.
fn adpcm_nibble(raw: u8) -> i64 {
    let value = i64::from(raw & 0xF);
    if value >= 8 {
        value - 16
    } else {
        value
    }
}

impl AudioRenderer {
    /// The mix buffer a `(channel resource, destination buffer)` pair names,
    /// as an index into the renderer's flat buffer array.
    fn buffer_index(&self, mix_id: u32, dest: usize) -> Option<usize> {
        let mix = self.mixes.iter().find(|m| m.mix_id == mix_id)?;
        if dest >= mix.buffer_count as usize {
            return None;
        }
        Some(mix.buffer_offset as usize + dest)
    }
}

impl Cpu {
    /// `IAudioRendererManager` (`audren:u`): opens renderers, and hands out the
    /// device interface that says which output they play through.
    pub(super) fn audren_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        let data = self.ipc_request_data(tls);
        // `AudioRendererParameter`: sample_rate, sample_count, mix_buffer_count,
        // submix_count, voice_count, sink_count, effect_count, unk1, unk2+pad,
        // splitter_count, unk3, unk4, revision.
        let sample_rate = self.mem.read_u32(data).unwrap_or(0);
        let sample_count = self.mem.read_u32(data.wrapping_add(4)).unwrap_or(0);
        let mix_buffer_count = self.mem.read_u32(data.wrapping_add(8)).unwrap_or(0);
        let submix_count = self.mem.read_u32(data.wrapping_add(12)).unwrap_or(0);
        let voice_count = self.mem.read_u32(data.wrapping_add(16)).unwrap_or(0);
        let sink_count = self.mem.read_u32(data.wrapping_add(20)).unwrap_or(0);
        let effect_count = self.mem.read_u32(data.wrapping_add(24)).unwrap_or(0);
        let revision_magic = self.mem.read_u32(data.wrapping_add(48)).unwrap_or(0);
        match cmd_id {
            // GetWorkBufferSize: any page-sized answer works — nothing here
            // actually allocates real renderer memory out of it.
            Some(1) => self.write_ipc_response(tls, 0, &[], &0x10_0000u64.to_le_bytes(), &[]),
            // OpenAudioRenderer.
            Some(0) => {
                let renderer = self.alloc_handle();
                self.record_handle(renderer, "audren:iaudiorenderer");
                let now = self.cycles;
                // A rate of 0 is a caller that did not fill the field in;
                // 48 kHz is what every renderer that names a rate asks for.
                let sample_rate = if sample_rate == 0 {
                    48_000
                } else {
                    sample_rate
                };
                let sample_count = if sample_count == 0 {
                    (sample_rate / FRAMES_PER_SECOND as u32).max(1)
                } else {
                    sample_count
                };
                self.audren_renderers.insert(
                    renderer,
                    AudioRenderer {
                        revision_magic,
                        revision: audren_revision(revision_magic),
                        sample_rate,
                        sample_count,
                        mix_buffer_count,
                        voice_count,
                        sink_count,
                        effect_count,
                        submix_count,
                        started: true,
                        frame_event: None,
                        next_frame_at: now.wrapping_add(FRAME_CYCLES),
                        rendered_through: now,
                        elapsed_frames: 0,
                        voices: vec![Voice::default(); voice_count as usize],
                        mempool_states: Vec::new(),
                        channels: Vec::new(),
                        mixes: Vec::new(),
                        sink: None,
                        warned_unplayable_sink: false,
                    },
                );
                self.write_ipc_response(tls, 0, &[renderer], &[], &[])
            }
            // GetAudioDeviceService / GetAudioDeviceServiceWithRevisionInfo
            // -> IAudioDevice. This used to fall into the catch-all below and
            // answer *success with no object at all*: the caller stored a null
            // where its device belonged, closed the session it had just been
            // given, and jumped through the null vtable several thousand
            // instructions later, with nothing left to say where it came from.
            Some(2) | Some(4) => {
                self.reply_with_interface(tls, handle, "audren:iaudiodevice")?;
                Ok(())
            }
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// `IAudioDevice`: which output the renderer is playing through, and how
    /// loud. There is one device here — the host's — and nothing routes
    /// between outputs, so it answers as a console docked to a TV.
    pub(super) fn audio_device_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        _handle: u64,
    ) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        /// `AudioDeviceName` is a fixed 0x100-byte NUL-padded string.
        const NAME_LEN: usize = 0x100;
        const ACTIVE_DEVICE: &[u8] = b"AudioTvOutput";
        match cmd_id {
            // ListAudioDeviceName (its Auto form, and the output-only list
            // that replaced it) -> the names, one 0x100-byte slot each, plus
            // how many were written. Every device here is an output, so the
            // three answers are the same three names.
            Some(0) | Some(6) | Some(14) => {
                let names: [&[u8]; 3] = [
                    ACTIVE_DEVICE,
                    b"AudioStereoJackOutput",
                    b"AudioBuiltInSpeakerOutput",
                ];
                let mut written = 0u32;
                if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                    for (i, name) in names.iter().enumerate() {
                        let at = (i * NAME_LEN) as u32;
                        if at + NAME_LEN as u32 > len {
                            break;
                        }
                        for j in 0..NAME_LEN as u32 {
                            let byte = name.get(j as usize).copied().unwrap_or(0);
                            self.mem.write_u8(addr.wrapping_add(at + j), byte)?;
                        }
                        written += 1;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &written.to_le_bytes(), &[])
            }
            // SetAudioDeviceOutputVolume(f32, name in a buffer). The host
            // owns the volume that is actually played, but the setting is
            // still the caller's to read back — see `AudioControl`.
            Some(1) | Some(7) => {
                let volume = f32::from_bits(self.mem.read_u32(self.ipc_request_data(tls))?);
                self.audio_control.set_device_volume(volume);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetAudioDeviceOutputVolume -> f32: whatever was last set, full
            // scale until something sets otherwise.
            Some(2) | Some(8) => {
                let volume = self.audio_control.device_volume();
                self.write_ipc_response(tls, 0, &[], &volume.to_le_bytes(), &[])
            }
            // GetActiveAudioDeviceName / ...Auto / GetActiveAudioOutputDeviceName.
            Some(3) | Some(10) | Some(13) => {
                if let Some((addr, len)) = self.ipc_output_buffer(tls, 0) {
                    for j in 0..NAME_LEN.min(len as usize) as u32 {
                        let byte = ACTIVE_DEVICE.get(j as usize).copied().unwrap_or(0);
                        self.mem.write_u8(addr.wrapping_add(j), byte)?;
                    }
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // QueryAudioDeviceSystemEvent / ...InputEvent / ...OutputEvent:
            // copy handles signalled when the audio output changes. Nothing
            // here ever changes it, so they are handed out and never fire.
            Some(4) | Some(11) | Some(12) => {
                let h = self.alloc_event("audren:device", true);
                self.write_ipc_reply(tls, 0, &[h], &[], &[], &[])
            }
            // GetActiveChannelCount: stereo.
            Some(5) => self.write_ipc_response(tls, 0, &[], &2u32.to_le_bytes(), &[]),
            _ => self.unimplemented_command(tls, "audren:iaudiodevice", cmd_id),
        }
    }

    /// `IAudioRenderer`.
    pub(super) fn audren_renderer_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        match cmd_id {
            // GetSampleRate / GetSampleCount / GetMixBufferCount: what the
            // renderer was opened with.
            Some(0) | Some(1) | Some(2) => {
                let renderer = self.audren_renderers.get(&handle);
                let value = match (cmd_id, renderer) {
                    (Some(0), Some(r)) => r.sample_rate,
                    (Some(1), Some(r)) => r.sample_count,
                    (Some(2), Some(r)) => r.mix_buffer_count,
                    _ => 0,
                };
                self.write_ipc_response(tls, 0, &[], &value.to_le_bytes(), &[])
            }
            // GetState: 0 while the renderer is running, 1 once it is stopped.
            Some(3) => {
                let started = self
                    .audren_renderers
                    .get(&handle)
                    .map(|r| r.started)
                    .unwrap_or(false);
                let state = u32::from(!started);
                self.write_ipc_response(tls, 0, &[], &state.to_le_bytes(), &[])
            }
            // RequestUpdateAudioRenderer, cmd 4 pre-3.0.0 / cmd 10 since.
            Some(4) | Some(10) => {
                self.audren_update(tls, handle)?;
                // The update is a round trip into the audio process, and the
                // caller is descheduled for its duration. Yielding here is
                // what keeps a mixer that never blocks from owning the CPU —
                // the same reason `AppendAudioOutBuffer` does it.
                self.pending_yield = true;
                Ok(())
            }
            // StartAudioRenderer / StopAudioRenderer.
            Some(5) | Some(6) => {
                let started = cmd_id == Some(5);
                let now = self.cycles;
                if let Some(renderer) = self.audren_renderers.get_mut(&handle) {
                    renderer.started = started;
                    // A renderer coming back from stopped starts its clock
                    // from the present. The silence while it was stopped is
                    // not a backlog it owes.
                    renderer.rendered_through = now;
                    renderer.next_frame_at = now.wrapping_add(FRAME_CYCLES);
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // QuerySystemEvent: the frame event, fired once every 5 ms of
            // emulated time by `Cpu::audren_tick`. It is a **copy** handle and
            // a real event: handing back a bare handle instead made every
            // `audrenWaitFrame` return at once, which is a renderer with no
            // clock at all.
            Some(7) => {
                let event = match self
                    .audren_renderers
                    .get(&handle)
                    .and_then(|r| r.frame_event)
                {
                    Some(event) => event,
                    None => {
                        let event = self.alloc_event("audren:frame", true);
                        if let Some(renderer) = self.audren_renderers.get_mut(&handle) {
                            renderer.frame_event = Some(event);
                        }
                        event
                    }
                };
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // SetAudioRendererRenderingTimeLimit / Get...: the share of a
            // frame the renderer may spend. Nothing here is scheduled against
            // it.
            Some(8) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            Some(9) => self.write_ipc_response(tls, 0, &[], &100u32.to_le_bytes(), &[]),
            _ => self.write_ipc_response(tls, 0, &[], &[], &[]),
        }
    }

    /// Fire the frame event of every renderer whose 5 ms period has come, and
    /// report the earliest cycle at which one of `handles` will fire.
    ///
    /// This is the renderer's half of [`Cpu::audio_tick`], and exists for the
    /// same reason: nothing runs in the background here, so a periodic tick
    /// has to be noticed by somebody, and the guest asking to wait is the
    /// moment that matters. The deadline is what makes the wait safe to
    /// honour — the waiter can be parked knowing exactly when it will wake.
    pub(super) fn audren_tick(&mut self, handles: &[u64]) -> Option<u64> {
        let now = self.cycles;
        let mut fire = Vec::new();
        let mut next = None;
        for renderer in self.audren_renderers.values_mut() {
            let Some(event) = renderer.frame_event else {
                continue;
            };
            if !renderer.started {
                continue;
            }
            if renderer.next_frame_at <= now {
                // From now, not from when it was due: a renderer that went
                // unwaited-on for a while has no backlog of frames to fire.
                renderer.next_frame_at = now.wrapping_add(FRAME_CYCLES);
                fire.push(event);
            } else if handles.contains(&event) {
                let due = renderer.next_frame_at;
                next = Some(next.map_or(due, |soonest: u64| soonest.min(due)));
            }
        }
        for event in fire {
            self.signal_event(event);
        }
        next
    }

    /// `RequestUpdateAudioRenderer`: take the whole renderer state the guest
    /// sent, render whatever frames have come due since the last update, and
    /// report back what the voices did.
    fn audren_update(&mut self, tls: u32, handle: u64) -> Result<()> {
        if let Some((addr, size)) = self.ipc_input_buffer(tls, 0) {
            self.audren_parse_update(handle, addr, size);
        }
        self.audren_render(handle);
        self.audren_write_update_reply(tls, handle)
    }

    /// Read one update's input buffer into the renderer's state.
    ///
    /// The sections are walked using the sizes the *guest* declared in its own
    /// header rather than sizes computed here. That is what makes one parser
    /// work across revisions: an effect or a mix entry grew between them, and
    /// a walk that assumed a stride would land mid-struct on the next section
    /// and read a voice out of a mix.
    fn audren_parse_update(&mut self, handle: u64, addr: u32, size: u32) {
        let Some(mut renderer) = self.audren_renderers.remove(&handle) else {
            return;
        };
        let read_u32 = |cpu: &Cpu, at: u32| cpu.mem.read_u32(at).unwrap_or(0);

        if size < HEADER_SZ {
            self.audren_renderers.insert(handle, renderer);
            return;
        }
        let behavior_sz = read_u32(self, addr.wrapping_add(0x04));
        let mempools_sz = read_u32(self, addr.wrapping_add(0x08));
        let voices_sz = read_u32(self, addr.wrapping_add(0x0c));
        let channels_sz = read_u32(self, addr.wrapping_add(0x10));
        let effects_sz = read_u32(self, addr.wrapping_add(0x14));
        let mixes_sz = read_u32(self, addr.wrapping_add(0x18));
        let sinks_sz = read_u32(self, addr.wrapping_add(0x1c));
        // The word libnx leaves zero and calls padding is the splitter
        // section: a renderer opened with splitters puts one here, and
        // stepping over it is the only reason to read it.
        let splitters_sz = read_u32(self, addr.wrapping_add(0x24));

        let mut at = addr.wrapping_add(HEADER_SZ).wrapping_add(behavior_sz);
        self.audren_parse_mempools(&mut renderer, at, mempools_sz);
        at = at.wrapping_add(mempools_sz);
        self.audren_parse_channels(&mut renderer, at, channels_sz);
        at = at.wrapping_add(channels_sz);
        self.audren_parse_voices(&mut renderer, at, voices_sz);
        at = at
            .wrapping_add(voices_sz)
            .wrapping_add(effects_sz)
            .wrapping_add(splitters_sz);
        self.audren_parse_mixes(&mut renderer, at, mixes_sz);
        at = at.wrapping_add(mixes_sz);
        self.audren_parse_sinks(&mut renderer, at, sinks_sz);

        self.audren_renderers.insert(handle, renderer);
    }

    /// `AudioRendererMemPoolInfoIn[]`. Guest memory *is* the renderer's memory
    /// here, so attaching a pool changes nothing about what can be read — but
    /// the acknowledgement is not optional: a pool the guest asked to attach
    /// and never saw attached is one it will keep asking about.
    fn audren_parse_mempools(&mut self, renderer: &mut AudioRenderer, addr: u32, size: u32) {
        let count = (size / MEMPOOL_IN_SZ) as usize;
        renderer.mempool_states = vec![0; count];
        for i in 0..count {
            let at = addr.wrapping_add(i as u32 * MEMPOOL_IN_SZ);
            let state = self.mem.read_u32(at.wrapping_add(0x10)).unwrap_or(0);
            renderer.mempool_states[i] = match state {
                MEMPOOL_REQUEST_ATTACH => MEMPOOL_ATTACHED,
                MEMPOOL_REQUEST_DETACH => MEMPOOL_DETACHED,
                // Anything else is left alone, which `Invalid` is the word
                // for: the guest reads it as "no transition happened".
                _ => 0,
            };
        }
    }

    /// `AudioRendererChannelInfoIn[]`: one voice channel's gain into each
    /// buffer of the mix it plays into.
    fn audren_parse_channels(&mut self, renderer: &mut AudioRenderer, addr: u32, size: u32) {
        let count = (size / CHANNEL_IN_SZ) as usize;
        renderer.channels.resize(count, ChannelResource::default());
        for i in 0..count {
            let at = addr.wrapping_add(i as u32 * CHANNEL_IN_SZ);
            let channel = &mut renderer.channels[i];
            for slot in 0..MAX_MIX_BUFFERS {
                let raw = self
                    .mem
                    .read_u32(at.wrapping_add(4 + slot as u32 * 4))
                    .unwrap_or(0);
                let gain = f32::from_bits(raw);
                channel.mix[slot] = if gain.is_finite() { gain } else { 0.0 };
            }
            channel.is_used = self.mem.read_u8(at.wrapping_add(0x64)).unwrap_or(0) != 0;
        }
    }

    /// `AudioRendererVoiceInfoIn[]`.
    fn audren_parse_voices(&mut self, renderer: &mut AudioRenderer, addr: u32, size: u32) {
        let count = (size / VOICE_IN_SZ) as usize;
        if renderer.voices.len() < count {
            renderer.voices.resize(count, Voice::default());
        }
        for i in 0..count {
            let at = addr.wrapping_add(i as u32 * VOICE_IN_SZ);
            // The voice's own id is what indexes the renderer's state, not its
            // position in the array — nothing promises the two agree.
            let id = self.mem.read_u32(at).unwrap_or(0) as usize;
            let id = if id < renderer.voices.len() { id } else { i };
            let is_new = self.mem.read_u8(at.wrapping_add(0x08)).unwrap_or(0) != 0;
            let in_use = self.mem.read_u8(at.wrapping_add(0x09)).unwrap_or(0) != 0;
            let play_state = self.mem.read_u8(at.wrapping_add(0x0a)).unwrap_or(0);
            let format = self.mem.read_u8(at.wrapping_add(0x0b)).unwrap_or(0);
            let sample_rate = self.mem.read_u32(at.wrapping_add(0x0c)).unwrap_or(0);
            let channel_count = self.mem.read_u32(at.wrapping_add(0x18)).unwrap_or(0);
            let pitch = f32::from_bits(self.mem.read_u32(at.wrapping_add(0x1c)).unwrap_or(0));
            let volume = f32::from_bits(self.mem.read_u32(at.wrapping_add(0x20)).unwrap_or(0));
            let wavebuf_count = self.mem.read_u32(at.wrapping_add(0x3c)).unwrap_or(0);
            let wavebuf_head = self.mem.read_u16(at.wrapping_add(0x40)).unwrap_or(0);
            let extra_params = self.mem.read_u32(at.wrapping_add(0x48)).unwrap_or(0);
            let dest_mix_id = self.mem.read_u32(at.wrapping_add(0x58)).unwrap_or(0);

            let voice = &mut renderer.voices[id];
            if is_new {
                voice.restart();
            }
            voice.in_use = in_use;
            voice.playing = play_state == PLAY_STATE_STARTED;
            voice.format = format;
            voice.sample_rate = sample_rate;
            voice.channel_count = channel_count.min(MAX_VOICE_CHANNELS as u32);
            voice.pitch = if pitch.is_finite() && pitch > 0.0 {
                pitch
            } else {
                1.0
            };
            voice.volume = if volume.is_finite() { volume } else { 0.0 };
            voice.dest_mix_id = dest_mix_id;

            for filter in 0..2 {
                let base = at.wrapping_add(0x24 + filter as u32 * 0x0c);
                let biquad = &mut voice.biquads[filter];
                biquad.enabled = self.mem.read_u8(base).unwrap_or(0) != 0;
                for n in 0..3 {
                    let raw = self
                        .mem
                        .read_u16(base.wrapping_add(2 + n as u32 * 2))
                        .unwrap_or(0);
                    biquad.numerator[n] = raw as i16;
                }
                for n in 0..2 {
                    let raw = self
                        .mem
                        .read_u16(base.wrapping_add(8 + n as u32 * 2))
                        .unwrap_or(0);
                    biquad.denominator[n] = raw as i16;
                }
            }

            for slot in 0..WAVE_BUFFERS {
                let base = at.wrapping_add(0x60 + slot as u32 * WAVE_BUF_IN_SZ);
                let start = self.mem.read_u32(base.wrapping_add(0x10)).unwrap_or(0);
                let end = self.mem.read_u32(base.wrapping_add(0x14)).unwrap_or(0);
                voice.wavebufs[slot] = WaveBuf {
                    address: self.mem.read_u32(base).unwrap_or(0),
                    size: self.mem.read_u32(base.wrapping_add(0x08)).unwrap_or(0),
                    start,
                    end,
                    looping: self.mem.read_u8(base.wrapping_add(0x18)).unwrap_or(0) != 0,
                    end_of_stream: self.mem.read_u8(base.wrapping_add(0x19)).unwrap_or(0) != 0,
                    context: self.mem.read_u32(base.wrapping_add(0x20)).unwrap_or(0),
                };
            }

            for channel in 0..MAX_VOICE_CHANNELS {
                let raw = self
                    .mem
                    .read_u32(at.wrapping_add(0x140 + channel as u32 * 4));
                voice.channel_ids[channel] = raw.unwrap_or(0);
            }

            // The ADPCM coefficient table travels as the voice's extra
            // parameters — sixteen s16, the eight predictor pairs a frame
            // header selects between.
            if format == PCM_ADPCM && extra_params != 0 {
                for n in 0..16 {
                    let raw = self.mem.read_u16(extra_params.wrapping_add(n as u32 * 2));
                    voice.adpcm_coefficients[n] = raw.unwrap_or(0) as i16;
                }
            }

            // Where the guest's ring head is now, which it advanced by exactly
            // the consumed count the last reply reported. Re-seeding from it
            // every update is what keeps the two sides in step without either
            // sending the other a position.
            voice.slot = usize::from(wavebuf_head) % WAVE_BUFFERS;
            voice.remaining = wavebuf_count.min(WAVE_BUFFERS as u32);
        }
    }

    /// `AudioRendererMixInfoIn[]`. Mix buffers are handed out in mix-id order,
    /// the final mix first, so an offset is the sum of the buffer counts
    /// before it.
    fn audren_parse_mixes(&mut self, renderer: &mut AudioRenderer, addr: u32, size: u32) {
        let count = (renderer.submix_count + 1) as usize;
        let stride = if count > 0 { size / count as u32 } else { 0 };
        renderer.mixes.clear();
        if stride == 0 {
            // A renderer whose update carried no mixes still has a final mix
            // to play through — it is the destination every voice names.
            renderer.mixes.push(Mix {
                is_used: true,
                mix_id: 0,
                volume: 1.0,
                buffer_count: renderer.mix_buffer_count.min(MAX_MIX_BUFFERS as u32),
                buffer_offset: 0,
                dest_mix_id: UNUSED_MIX_ID,
                matrix: Vec::new(),
            });
            return;
        }
        for i in 0..count {
            let at = addr.wrapping_add(i as u32 * stride);
            let volume = f32::from_bits(self.mem.read_u32(at).unwrap_or(0));
            let buffer_count = self.mem.read_u32(at.wrapping_add(0x08)).unwrap_or(0);
            let is_used = self.mem.read_u8(at.wrapping_add(0x0c)).unwrap_or(0) != 0;
            let mix_id = self.mem.read_u32(at.wrapping_add(0x10)).unwrap_or(0);
            let dest_mix_id = self.mem.read_u32(at.wrapping_add(0x924)).unwrap_or(0);
            let buffer_count = buffer_count.min(MAX_MIX_BUFFERS as u32);
            let mut matrix = Vec::new();
            // A submix's matrix is only ever read for what it sends onward, so
            // it is only worth reading for a mix that sends somewhere.
            if dest_mix_id != UNUSED_MIX_ID && buffer_count > 0 {
                matrix = vec![0.0; MAX_MIX_BUFFERS * MAX_MIX_BUFFERS];
                for src in 0..buffer_count as usize {
                    for dest in 0..MAX_MIX_BUFFERS {
                        let offset = 0x24 + ((src * MAX_MIX_BUFFERS + dest) * 4) as u32;
                        let raw = self.mem.read_u32(at.wrapping_add(offset)).unwrap_or(0);
                        let gain = f32::from_bits(raw);
                        matrix[src * MAX_MIX_BUFFERS + dest] =
                            if gain.is_finite() { gain } else { 0.0 };
                    }
                }
            }
            renderer.mixes.push(Mix {
                is_used,
                mix_id,
                volume: if volume.is_finite() { volume } else { 1.0 },
                buffer_count,
                buffer_offset: 0,
                dest_mix_id,
                matrix,
            });
        }
        // Assign the buffer ranges in mix-id order: the final mix took the
        // first of them and each submix the next range, which is the order
        // `audrvMixAdd` hands them out in.
        renderer.mixes.sort_by_key(|mix| mix.mix_id);
        let mut offset = 0;
        for mix in renderer.mixes.iter_mut() {
            mix.buffer_offset = offset;
            offset += mix.buffer_count;
        }
    }

    /// `AudioRendererSinkInfoIn[]`: what the finished mix plays through.
    fn audren_parse_sinks(&mut self, renderer: &mut AudioRenderer, addr: u32, size: u32) {
        let count = renderer.sink_count as usize;
        let stride = if count > 0 { size / count as u32 } else { 0 };
        renderer.sink = None;
        if stride == 0 {
            return;
        }
        for i in 0..count {
            let at = addr.wrapping_add(i as u32 * stride);
            let kind = self.mem.read_u8(at).unwrap_or(0);
            let is_used = self.mem.read_u8(at.wrapping_add(0x01)).unwrap_or(0) != 0;
            if !is_used {
                continue;
            }
            if kind != SINK_TYPE_DEVICE {
                if kind == SINK_TYPE_CIRCULAR && !renderer.warned_unplayable_sink {
                    renderer.warned_unplayable_sink = true;
                    crate::traceln!(
                        "[audio] audren: a circular-buffer sink is not written; \
                         only the device sink reaches the host"
                    );
                }
                continue;
            }
            // The union starts past the type, the node id and three reserved
            // words; a device sink's name fills the 0x100 bytes after that.
            let sink = at.wrapping_add(0x20);
            let channels = self.mem.read_u32(sink.wrapping_add(0x100)).unwrap_or(0);
            let mut inputs = [0u8; MAX_SINK_INPUTS];
            for (channel, slot) in inputs.iter_mut().enumerate() {
                *slot = self
                    .mem
                    .read_u8(sink.wrapping_add(0x104 + channel as u32))
                    .unwrap_or(0);
            }
            renderer.sink = Some(DeviceSink {
                channels: channels.clamp(1, MAX_SINK_INPUTS as u32),
                inputs,
            });
            break;
        }
    }

    /// Render every 5 ms frame that has come due and queue the result for the
    /// host.
    ///
    /// The frames are counted off `cycles`, the same clock `audout` releases
    /// its buffers on, so the mix drains at the rate a real renderer's would
    /// however fast or slow the emulator is running.
    fn audren_render(&mut self, handle: u64) {
        let now = self.cycles;
        let Some(mut renderer) = self.audren_renderers.remove(&handle) else {
            return;
        };
        if !renderer.started {
            renderer.rendered_through = now;
            self.audren_renderers.insert(handle, renderer);
            return;
        }
        let elapsed = now.saturating_sub(renderer.rendered_through);
        let due = (elapsed / FRAME_CYCLES).min(MAX_CATCHUP_FRAMES);
        renderer.rendered_through = now;
        if due == 0 {
            self.audren_renderers.insert(handle, renderer);
            return;
        }

        let channels = renderer.sink.as_ref().map(|s| s.channels).unwrap_or(2);
        let frames = renderer.sample_count as usize;
        let mut pcm: Vec<i16> = Vec::with_capacity(due as usize * frames * channels as usize);
        for _ in 0..due {
            render_frame(&self.mem, &mut renderer, &mut pcm);
            renderer.elapsed_frames = renderer.elapsed_frames.wrapping_add(1);
        }

        if crate::trace::enabled(crate::trace::Trace::Audio) {
            let playing = renderer
                .voices
                .iter()
                .filter(|voice| voice.in_use && voice.playing && voice.remaining > 0)
                .count();
            crate::traceln!(
                "[audio] audren frames={due} voices={playing}/{} rate={} channels={channels}",
                renderer.voices.len(),
                renderer.sample_rate
            );
        }
        let format = (renderer.sample_rate, channels);
        self.audren_renderers.insert(handle, renderer);
        if !pcm.is_empty() {
            // Whichever device is actually producing samples defines the
            // format the host plays them in, the same rule `audout` follows.
            self.audio_format = format;
            self.queue_audio(pcm.into_iter());
        }
    }

    /// `RequestUpdateAudioRenderer`'s reply: an `AudioRendererUpdateDataHeader`
    /// and then, in this order, one status per mempool, per voice, per effect
    /// and per sink, the performance and behaviour tails, and the renderer
    /// info.
    ///
    /// Getting the `_sz` fields right matters: both `audrvUpdate` and `nnSdk`
    /// walk the reply section by section against the sizes they computed from
    /// the voice/sink/effect counts the renderer was opened with, and abort on
    /// the first that disagrees — every frame the title is alive, not just at
    /// startup.
    pub(super) fn audren_write_update_reply(&mut self, tls: u32, handle: u64) -> Result<()> {
        let Some(renderer) = self.audren_renderers.get(&handle) else {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        };
        let voice_count = renderer.voice_count;
        let sink_count = renderer.sink_count;
        let effect_count = renderer.effect_count;
        let revision_magic = renderer.revision_magic;
        let revision = renderer.revision;
        let mempool_count = effect_count + 4 * voice_count;

        const HEADER_OUT_SZ: u32 = 64;
        const MEMPOOL_OUT_SZ: u32 = 16;
        const VOICE_OUT_SZ: u32 = 16;
        const SINK_OUT_SZ: u32 = 32;
        const PERFMGR_OUT_SZ: u32 = 16;
        const BEHAVIOR_OUT_SZ: u32 = 176;
        /// `RendererInfoOut`: an elapsed-frame counter and its reserved half,
        /// the last section of the reply. Revision 5 added it.
        const RENDER_INFO_OUT_SZ: u32 = 16;
        /// `EffectOutStatus`, and the wider revision-9 form that carries an
        /// aux-buffer/limiter report alongside the state byte.
        const EFFECT_OUT_SZ: u32 = 16;
        const EFFECT_OUT_V2_SZ: u32 = 0x90;

        let mempools_sz = mempool_count * MEMPOOL_OUT_SZ;
        let voices_sz = voice_count * VOICE_OUT_SZ;
        let effects_sz = effect_count
            * if revision >= 9 {
                EFFECT_OUT_V2_SZ
            } else {
                EFFECT_OUT_SZ
            };
        let sinks_sz = sink_count * SINK_OUT_SZ;
        let render_info_sz = if revision >= 5 { RENDER_INFO_OUT_SZ } else { 0 };
        let total_sz = HEADER_OUT_SZ
            + mempools_sz
            + voices_sz
            + effects_sz
            + sinks_sz
            + PERFMGR_OUT_SZ
            + BEHAVIOR_OUT_SZ
            + render_info_sz;

        let mut reply = vec![0u8; total_sz as usize];
        reply[0..4].copy_from_slice(&revision_magic.to_le_bytes());
        reply[4..8].copy_from_slice(&BEHAVIOR_OUT_SZ.to_le_bytes());
        reply[8..12].copy_from_slice(&mempools_sz.to_le_bytes());
        reply[12..16].copy_from_slice(&voices_sz.to_le_bytes());
        reply[20..24].copy_from_slice(&effects_sz.to_le_bytes());
        // channels_sz and mixes_sz stay 0: the renderer reports nothing back
        // for either, and neither is a section of the reply.
        reply[28..32].copy_from_slice(&sinks_sz.to_le_bytes());
        reply[32..36].copy_from_slice(&PERFMGR_OUT_SZ.to_le_bytes());
        reply[40..44].copy_from_slice(&render_info_sz.to_le_bytes());
        reply[60..64].copy_from_slice(&total_sz.to_le_bytes());

        // `MemPoolInfoOut`: the transition each pool made, or 0 for none.
        let mut at = HEADER_OUT_SZ as usize;
        for i in 0..mempool_count as usize {
            let state = renderer.mempool_states.get(i).copied().unwrap_or(0);
            reply[at..at + 4].copy_from_slice(&state.to_le_bytes());
            at += MEMPOOL_OUT_SZ as usize;
        }

        // `VoiceInfoOut`: how far each voice has got. `num_wavebufs_consumed`
        // is the load-bearing one — the guest advances its own ring head by
        // the delta and only refills a buffer this has accounted for, so a
        // renderer that reports zero is one whose title runs out of buffers
        // and stops.
        for i in 0..voice_count as usize {
            let (played, consumed, drops) = match renderer.voices.get(i) {
                Some(voice) => (voice.played_samples, voice.wavebufs_consumed, voice.drops),
                None => (0, 0, 0),
            };
            reply[at..at + 8].copy_from_slice(&played.to_le_bytes());
            reply[at + 8..at + 12].copy_from_slice(&consumed.to_le_bytes());
            reply[at + 12..at + 16].copy_from_slice(&drops.to_le_bytes());
            at += VOICE_OUT_SZ as usize;
        }

        // Effects and sinks report nothing: no effect is processed here, and
        // `last_written_offset` belongs to the circular-buffer sink, which is
        // not written. Zero is the truthful answer for both.
        at += effects_sz as usize;
        at += sinks_sz as usize;
        at += PERFMGR_OUT_SZ as usize;
        at += BEHAVIOR_OUT_SZ as usize;

        // `RendererInfoOut`: frames rendered since the renderer was opened.
        if render_info_sz != 0 && at + 8 <= reply.len() {
            reply[at..at + 8].copy_from_slice(&renderer.elapsed_frames.to_le_bytes());
        }

        let (_, recv) = self.ipc_buffers(tls);
        if let Some(&(addr, size)) = recv.first() {
            let n = (size as usize).min(reply.len());
            for (i, &byte) in reply[..n].iter().enumerate() {
                self.mem.write_u8(addr.wrapping_add(i as u32), byte)?;
            }
        }
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }
}

/// Render one 5 ms frame: every playing voice into its destination mix, every
/// submix onward into its own destination, then the final mix out through the
/// sink as interleaved 16-bit PCM.
fn render_frame(mem: &Memory, renderer: &mut AudioRenderer, out: &mut Vec<i16>) {
    let frames = renderer.sample_count as usize;
    let buffer_count = renderer
        .mixes
        .iter()
        .map(|mix| mix.buffer_offset + mix.buffer_count)
        .max()
        .unwrap_or(renderer.mix_buffer_count)
        .max(1)
        .min(MAX_MIX_BUFFERS as u32) as usize;
    let mut buffers = vec![vec![0f32; frames]; buffer_count];

    for index in 0..renderer.voices.len() {
        render_voice(mem, renderer, index, &mut buffers);
    }

    // Submixes feed their destination before the final mix is read. Highest
    // mix id first, because a submix is always created after the mix it sends
    // to — so descending id is the order that never reads a buffer another
    // mix has yet to write.
    let mut order: Vec<usize> = (0..renderer.mixes.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(renderer.mixes[i].mix_id));
    for i in order {
        let mix = renderer.mixes[i].clone();
        if !mix.is_used || mix.dest_mix_id == UNUSED_MIX_ID || mix.matrix.is_empty() {
            continue;
        }
        let Some(dest) = renderer
            .mixes
            .iter()
            .find(|m| m.mix_id == mix.dest_mix_id)
            .cloned()
        else {
            continue;
        };
        for src in 0..mix.buffer_count as usize {
            let from = mix.buffer_offset as usize + src;
            if from >= buffers.len() {
                continue;
            }
            for to_index in 0..dest.buffer_count as usize {
                let gain = mix.matrix[src * MAX_MIX_BUFFERS + to_index] * mix.volume;
                if gain == 0.0 {
                    continue;
                }
                let to = dest.buffer_offset as usize + to_index;
                if to >= buffers.len() || to == from {
                    continue;
                }
                #[allow(clippy::needless_range_loop)] // two rows, one slice
                for sample in 0..frames {
                    buffers[to][sample] += buffers[from][sample] * gain;
                }
            }
        }
    }

    let final_mix = renderer.mixes.iter().find(|mix| mix.mix_id == 0).cloned();
    let final_volume = final_mix.as_ref().map(|mix| mix.volume).unwrap_or(1.0);
    let sink = renderer.sink.clone().unwrap_or(DeviceSink {
        channels: 2,
        inputs: [0, 1, 2, 3, 4, 5],
    });
    let base = final_mix.map(|mix| mix.buffer_offset as usize).unwrap_or(0);
    for sample in 0..frames {
        for channel in 0..sink.channels as usize {
            let index = base + usize::from(sink.inputs[channel]);
            let value = buffers.get(index).map(|buf| buf[sample]).unwrap_or(0.0) * final_volume;
            // 32768, not 32767: the decoders divide by 32768, so scaling back
            // by the same figure makes a 16-bit source that passes through at
            // unity gain come out bit-exact rather than a count light.
            out.push((value * 32768.0).clamp(-32768.0, 32767.0) as i16);
        }
    }
}

/// Mix one voice into the buffers of the mix it plays into: decode its source
/// samples, resample them to the renderer's rate, filter them, and add them at
/// the gain its channel resources name.
fn render_voice(
    mem: &Memory,
    renderer: &mut AudioRenderer,
    index: usize,
    buffers: &mut [Vec<f32>],
) {
    let frames = renderer.sample_count as usize;
    let rate = renderer.sample_rate.max(1);
    let Some(voice) = renderer.voices.get(index) else {
        return;
    };
    if !voice.in_use || !voice.playing || voice.channel_count == 0 || voice.remaining == 0 {
        return;
    }
    let dest_mix_id = voice.dest_mix_id;
    if dest_mix_id == UNUSED_MIX_ID {
        return;
    }
    let channel_count = voice.channel_count as usize;
    // Where each of this voice's channels goes, resolved once: a gain per mix
    // buffer, already flattened to the buffer indices they land in.
    let mut routing: Vec<Vec<(usize, f32)>> = Vec::with_capacity(channel_count);
    for channel in 0..channel_count {
        let id = voice.channel_ids[channel] as usize;
        let mut gains = Vec::new();
        if let Some(resource) = renderer.channels.get(id) {
            for dest in 0..MAX_MIX_BUFFERS {
                let gain = resource.mix[dest];
                if gain == 0.0 {
                    continue;
                }
                if let Some(buffer) = renderer.buffer_index(dest_mix_id, dest) {
                    if buffer < buffers.len() {
                        gains.push((buffer, gain));
                    }
                }
            }
        }
        routing.push(gains);
    }
    if routing.iter().all(|gains| gains.is_empty()) {
        // Nothing this voice produces is routed anywhere, so decoding it would
        // be work with no destination — but it still has to *advance*, or the
        // guest never gets its wave buffers back.
        advance_silently(renderer, index);
        return;
    }

    let voice = &mut renderer.voices[index];
    let volume = voice.volume;
    let step = (voice.sample_rate as f32 / rate as f32) * voice.pitch;
    // A source rate of zero, or a pitch that resamples to a standstill, would
    // read one sample forever. Nothing plays at that rate.
    if !(step.is_finite() && step > 0.0) {
        return;
    }

    let mut mixed = vec![0f32; frames * channel_count];
    for frame in 0..frames {
        if !voice.primed {
            // Two samples, because interpolating needs the one on either side
            // of the position. The first output frame sits exactly on the
            // first sample, so nothing is skipped by reading ahead.
            if !pull_source(mem, voice) {
                break;
            }
            voice.prev = voice.cur;
            if !pull_source(mem, voice) {
                voice.cur = [0.0; MAX_VOICE_CHANNELS];
            }
            voice.primed = true;
            voice.frac = 0.0;
        }
        for channel in 0..channel_count {
            let a = voice.prev[channel];
            let b = voice.cur[channel];
            mixed[frame * channel_count + channel] = a + (b - a) * voice.frac;
        }
        voice.frac += step;
        while voice.frac >= 1.0 {
            voice.frac -= 1.0;
            voice.prev = voice.cur;
            if !pull_source(mem, voice) {
                // The voice ran dry. Interpolating toward zero rather than
                // holding the last sample is what keeps the end of a stream
                // from leaving a DC step behind, and `primed` stays set so the
                // sample already in `prev` is still played — clearing it here
                // dropped the last sample of every wave buffer.
                voice.cur = [0.0; MAX_VOICE_CHANNELS];
            }
        }
    }

    for channel in 0..channel_count {
        for filter in 0..2 {
            let biquad = voice.biquads[filter];
            if !biquad.enabled {
                continue;
            }
            let mut state = voice.biquad_state[channel][filter];
            for frame in 0..frames {
                let sample = &mut mixed[frame * channel_count + channel];
                *sample = apply_biquad(&biquad, &mut state, *sample);
            }
            voice.biquad_state[channel][filter] = state;
        }
    }

    for channel in 0..channel_count {
        for &(buffer, gain) in &routing[channel] {
            let scale = gain * volume;
            for frame in 0..frames {
                buffers[buffer][frame] += mixed[frame * channel_count + channel] * scale;
            }
        }
    }
}

/// Advance a voice through the samples this frame would have consumed without
/// mixing them anywhere.
///
/// A voice routed nowhere is still playing, and the guest is still waiting for
/// the wave buffers back. Skipping the decode entirely would leave it holding
/// them forever.
fn advance_silently(renderer: &mut AudioRenderer, index: usize) {
    let frames = renderer.sample_count as u64;
    let rate = renderer.sample_rate.max(1);
    let voice = &mut renderer.voices[index];
    let step = (voice.sample_rate as f32 / rate as f32) * voice.pitch;
    if !(step.is_finite() && step > 0.0) {
        return;
    }
    let mut remaining = (frames as f32 * step) as u64;
    while remaining > 0 {
        if !skip_source(voice) {
            break;
        }
        remaining -= 1;
    }
}

/// Step a voice one source sample forward without decoding it, moving to the
/// next wave buffer — or looping — exactly as [`pull_source`] would.
fn skip_source(voice: &mut Voice) -> bool {
    loop {
        if voice.remaining == 0 {
            return false;
        }
        let wavebuf = voice.wavebufs[voice.slot];
        let total = playable_samples(voice, wavebuf);
        if voice.offset < total {
            voice.offset += 1;
            voice.played_samples = voice.played_samples.wrapping_add(1);
            return true;
        }
        if wavebuf.looping && total > 0 {
            voice.offset = 0;
            voice.adpcm.next = None;
            continue;
        }
        finish_wavebuf(voice, wavebuf);
    }
}

/// How many samples of a wave buffer a voice can actually play.
///
/// `end_sample_offset` is the guest's claim about its own buffer and `size` is
/// the buffer it allocated; where they disagree, the allocation wins. `audout`
/// learned this the expensive way — the Mii editor submits a descriptor whose
/// offsets land outside the buffer entirely, and what reached the speakers was
/// the struct's own pointers read as PCM.
fn playable_samples(voice: &Voice, wavebuf: WaveBuf) -> u32 {
    if wavebuf.address == 0 {
        return 0;
    }
    let fits = match voice.format {
        PCM_ADPCM => (wavebuf.size / ADPCM_BYTES_PER_FRAME) * ADPCM_SAMPLES_PER_FRAME,
        format => {
            let width = match format {
                PCM_INT8 => 1,
                PCM_INT16 => 2,
                PCM_INT24 => 3,
                PCM_INT32 | PCM_FLOAT => 4,
                _ => return 0,
            };
            let frame = voice.channel_count.max(1) * width;
            wavebuf.size / frame
        }
    };
    wavebuf.end.min(fits).saturating_sub(wavebuf.start)
}

/// Retire the wave buffer a voice has just played out and move to the next.
fn finish_wavebuf(voice: &mut Voice, wavebuf: WaveBuf) {
    voice.slot = (voice.slot + 1) % WAVE_BUFFERS;
    voice.remaining = voice.remaining.saturating_sub(1);
    voice.wavebufs_consumed = voice.wavebufs_consumed.wrapping_add(1);
    voice.offset = 0;
    // The next buffer's ADPCM history is its own, seeded from its context.
    voice.adpcm.next = None;
    if wavebuf.end_of_stream {
        voice.remaining = 0;
    }
}

/// Decode the next source sample of every channel of a voice into `cur`,
/// advancing through the wave-buffer ring as buffers run out.
///
/// Returns false once the voice has nothing left to play, which is a voice
/// whose guest has not queued a buffer in time — silence, not an error.
fn pull_source(mem: &Memory, voice: &mut Voice) -> bool {
    loop {
        if voice.remaining == 0 {
            return false;
        }
        let wavebuf = voice.wavebufs[voice.slot];
        let total = playable_samples(voice, wavebuf);
        if voice.offset >= total {
            if wavebuf.looping && total > 0 {
                voice.offset = 0;
                // A loop restarts the decoder from the state the encoder left
                // at the loop point, not from silence.
                seed_adpcm(mem, voice, wavebuf);
                continue;
            }
            finish_wavebuf(voice, wavebuf);
            continue;
        }
        let index = wavebuf.start.wrapping_add(voice.offset);
        let channels = voice.channel_count as usize;
        voice.cur = [0.0; MAX_VOICE_CHANNELS];
        if voice.format == PCM_ADPCM {
            // ADPCM is a mono codec here: a multi-channel source is encoded as
            // one voice per channel, which is the only arrangement
            // `audrvVoiceInit` and every title that uses it produce.
            if voice.adpcm.next != Some(index) {
                seed_adpcm(mem, voice, wavebuf);
                decode_adpcm_from_frame_start(mem, voice, wavebuf, index);
            }
            let sample = decode_adpcm(mem, voice, wavebuf, index);
            for channel in 0..channels {
                voice.cur[channel] = sample;
            }
        } else {
            for channel in 0..channels {
                voice.cur[channel] = decode_pcm(mem, voice, wavebuf, index, channel);
            }
        }
        voice.offset += 1;
        voice.played_samples = voice.played_samples.wrapping_add(1);
        return true;
    }
}

/// Load the ADPCM predictor's history from the wave buffer's context, or clear
/// it if the buffer carries none.
fn seed_adpcm(mem: &Memory, voice: &mut Voice, wavebuf: WaveBuf) {
    if wavebuf.context != 0 {
        // `AudioRendererAdpcmContext`: a frame index, then the two history
        // samples the decoder should resume with.
        voice.adpcm.history0 = mem.read_u16(wavebuf.context.wrapping_add(2)).unwrap_or(0) as i16;
        voice.adpcm.history1 = mem.read_u16(wavebuf.context.wrapping_add(4)).unwrap_or(0) as i16;
    } else {
        voice.adpcm.history0 = 0;
        voice.adpcm.history1 = 0;
    }
    voice.adpcm.next = None;
}

/// Decode from the start of the 14-sample frame `index` falls in, up to it.
///
/// ADPCM samples are differences from the two before them, so one cannot be
/// decoded on its own. A frame boundary is the coarsest point the decoder can
/// resume at, and the predictor converges within a frame, so this is also what
/// hardware does when a voice seeks.
fn decode_adpcm_from_frame_start(mem: &Memory, voice: &mut Voice, wavebuf: WaveBuf, index: u32) {
    let frame_start = (index / ADPCM_SAMPLES_PER_FRAME) * ADPCM_SAMPLES_PER_FRAME;
    voice.adpcm.next = Some(frame_start);
    for at in frame_start..index {
        decode_adpcm(mem, voice, wavebuf, at);
    }
}

/// Decode one 4-bit ADPCM sample, advancing the predictor.
fn decode_adpcm(mem: &Memory, voice: &mut Voice, wavebuf: WaveBuf, index: u32) -> f32 {
    let frame = index / ADPCM_SAMPLES_PER_FRAME;
    let within = index % ADPCM_SAMPLES_PER_FRAME;
    let frame_at = wavebuf.address.wrapping_add(frame * ADPCM_BYTES_PER_FRAME);
    let header = mem.read_u8(frame_at).unwrap_or(0);
    let scale = i64::from(header & 0xF);
    let coefficient = usize::from(header >> 4) & 0xF;
    let coef0 = i64::from(voice.adpcm_coefficients[coefficient * 2]);
    let coef1 = i64::from(voice.adpcm_coefficients[coefficient * 2 + 1]);

    let byte = mem
        .read_u8(frame_at.wrapping_add(1 + within / 2))
        .unwrap_or(0);
    let raw = if within.is_multiple_of(2) {
        byte >> 4
    } else {
        byte
    };
    let nibble = adpcm_nibble(raw);

    let history0 = i64::from(voice.adpcm.history0);
    let history1 = i64::from(voice.adpcm.history1);
    let prediction = coef0 * history0 + coef1 * history1;
    // The rounding constant is half of the coefficients' fixed-point unit, so
    // the shift rounds to nearest rather than toward negative infinity.
    let rounding = 1 << (ADPCM_COEF_SHIFT - 1);
    let value = ((nibble << scale) << ADPCM_COEF_SHIFT) + prediction + rounding;
    let sample = (value >> ADPCM_COEF_SHIFT).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;

    voice.adpcm.history1 = voice.adpcm.history0;
    voice.adpcm.history0 = sample;
    voice.adpcm.next = Some(index + 1);
    f32::from(sample) / 32768.0
}

/// Read one interleaved PCM sample of a voice's channel, as a float in
/// -1.0..=1.0.
fn decode_pcm(mem: &Memory, voice: &Voice, wavebuf: WaveBuf, index: u32, channel: usize) -> f32 {
    let channels = voice.channel_count.max(1);
    let slot = index.wrapping_mul(channels).wrapping_add(channel as u32);
    match voice.format {
        PCM_INT8 => {
            let at = wavebuf.address.wrapping_add(slot);
            f32::from(mem.read_u8(at).unwrap_or(0) as i8) / 128.0
        }
        PCM_INT16 => {
            let at = wavebuf.address.wrapping_add(slot.wrapping_mul(2));
            f32::from(mem.read_u16(at).unwrap_or(0) as i16) / 32768.0
        }
        PCM_INT24 => {
            let at = wavebuf.address.wrapping_add(slot.wrapping_mul(3));
            let low = u32::from(mem.read_u8(at).unwrap_or(0));
            let mid = u32::from(mem.read_u8(at.wrapping_add(1)).unwrap_or(0));
            let high = u32::from(mem.read_u8(at.wrapping_add(2)).unwrap_or(0));
            // Sign-extend from 24 bits by landing the value in the top three
            // bytes of an i32 and shifting back down.
            let raw = ((low << 8) | (mid << 16) | (high << 24)) as i32;
            (raw >> 8) as f32 / 8_388_608.0
        }
        PCM_INT32 => {
            let at = wavebuf.address.wrapping_add(slot.wrapping_mul(4));
            mem.read_u32(at).unwrap_or(0) as i32 as f32 / 2_147_483_648.0
        }
        PCM_FLOAT => {
            let at = wavebuf.address.wrapping_add(slot.wrapping_mul(4));
            let value = f32::from_bits(mem.read_u32(at).unwrap_or(0));
            if value.is_finite() {
                value
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Run one biquad section over a sample.
///
/// The coefficients arrive in Q14 — `audrvVoiceSetBiquadFilter` scales by
/// 16384 — and the form is the transposed direct II the renderer's filters are
/// defined in, where the denominators are stored already negated. The output
/// is clamped because the guest chooses the coefficients and nothing stops it
/// choosing an unstable set.
fn apply_biquad(biquad: &Biquad, state: &mut BiquadState, input: f32) -> f32 {
    const Q14: f32 = 1.0 / 16384.0;
    let b0 = f32::from(biquad.numerator[0]) * Q14;
    let b1 = f32::from(biquad.numerator[1]) * Q14;
    let b2 = f32::from(biquad.numerator[2]) * Q14;
    let a1 = f32::from(biquad.denominator[0]) * Q14;
    let a2 = f32::from(biquad.denominator[1]) * Q14;
    let output = input * b0 + state.s0;
    state.s0 = input * b1 + output * a1 + state.s1;
    state.s1 = input * b2 + output * a2;
    if !state.s0.is_finite() || !state.s1.is_finite() {
        state.s0 = 0.0;
        state.s1 = 0.0;
    }
    output.clamp(-1.0, 1.0)
}
