//! `audout`, the plain PCM-out device, and `audctl`, the system-wide audio
//! settings beside it.
//!
//! This is the device path: the guest hands over finished PCM and it plays.
//! The renderer — what nearly every retail title actually uses — is
//! [`super::audren`].
//!
//! **The device plays in time.** A buffer is released once the CPU has run for
//! as long as its samples take at the device's rate, queued behind whatever is
//! still playing. Releasing on arrival hands the guest an infinitely fast
//! sound card, and a title's audio clock is what its video is scheduled
//! against — one title pushed 205x real time and its video player dropped
//! every frame of the boot video.

use super::power::CLOCK_RATES_HZ;
use super::Cpu;
use crate::Result;
use std::collections::VecDeque;

/// One open `IAudioOut` session: what it was opened with, and the buffer
/// bookkeeping its client polls.
///
/// A real device releases a buffer once its samples have been clocked out to
/// the DAC. There is no DAC here — the samples are copied into
/// [`Cpu::audio_pcm`] for the host to play — but *when* a buffer comes back is
/// the whole of the guest's audio clock, so the device keeps a clock of its
/// own: a buffer is released once the emulated CPU has run for as long as its
/// samples take to play.
///
/// Releasing on arrival instead, which this used to do, hands the guest a
/// device infinitely faster than the panel beside it. Just Dance 2019 fed
/// 19,693,344 samples per second of emulated time through a 48 kHz stereo
/// device — **205× real time** — and its video player, which schedules frames
/// against the audio clock, concluded every frame of the boot video was too
/// late to show and dropped all of them. The title presented a white clear
/// sixty times a second and never issued a single draw.
#[derive(Debug, Clone)]
pub(crate) struct AudioOut {
    /// Sample rate and channel count the device was opened with.
    pub sample_rate: u32,
    pub channel_count: u32,
    /// Whether `StartAudioOut` has been called and `StopAudioOut` has not.
    pub started: bool,
    /// The volume the guest set, 0.0..=1.0. Applied when samples are taken.
    pub volume: f32,
    /// Signalled every time a buffer is released — what
    /// `audoutWaitPlayFinish` blocks on.
    pub event: u64,
    /// Buffers the guest has appended and not yet collected, each with the
    /// cycle count at which the device will have finished playing it.
    /// `GetReleasedAudioOutBuffer` hands back the ones whose time has come.
    pub queued: VecDeque<(u64, u64)>,
    /// The cycle the device finishes everything queued so far — where the next
    /// buffer starts playing. A device that has fallen silent starts again
    /// from the present rather than from whenever it last stopped, so a gap in
    /// the guest's submissions is a gap in the audio, not a debt the device
    /// has to work off.
    pub free_at: u64,
    /// Frames handed over since the device was opened, which is what
    /// `GetAudioOutPlayedSampleCount` reports.
    pub played_frames: u64,
}

/// `nn::audio::PcmFormat`: 16-bit signed samples, the only format `audout`
/// takes here and the one every caller asks for.
const PCM_FORMAT_INT16: u32 = 2;

/// `nn::audio::AudioOutState`, as `IAudioOut` reports it.
const AUDIO_OUT_STARTED: u32 = 0;

const AUDIO_OUT_STOPPED: u32 = 1;

/// How many `nn::settings::system::AudioOutputModeTarget` values there are —
/// None, Hdmi, Speaker, Headphone, Type3, Type4. Every `audctl` setting is
/// per-target, and a target outside this range is one no console has.
pub(super) const AUDIO_TARGETS: usize = 6;

/// `nn::settings::system::AudioOutputModeTarget::Speaker`. This console has
/// no HDMI and no headphone jack, so the speaker is both the default target
/// and the active one.
pub(super) const AUDIO_TARGET_SPEAKER: u32 = 2;

/// `nn::settings::system::AudioOutputMode::ch_2` — stereo, which is what
/// `audout` opens and therefore the only layout this console can be in.
pub(super) const AUDIO_OUTPUT_MODE_STEREO: u32 = 1;

/// The volume scale `audctl` reports. Both ends are fixed in the firmware
/// rather than derived from anything, and a caller draws its slider from
/// them.
pub(super) const AUDIO_VOLUME_MAX: i32 = 15;

/// `audctl`'s state: the system-wide audio settings, which are settings
/// rather than facts about the hardware and so are stored and handed back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AudioControl {
    /// Volume and mute per output target.
    volume: [i32; AUDIO_TARGETS],
    muted: [bool; AUDIO_TARGETS],
    /// The channel layout each target is configured for.
    output_mode: [u32; AUDIO_TARGETS],
    /// Which target the user chose, and the master volume applied on top of
    /// the per-target one.
    default_target: u32,
    master_volume: f32,
    /// `ForceMutePolicy::Disable` — whether the speaker is cut when
    /// headphones are unplugged. There is no jack to unplug.
    force_mute_policy: u32,
    /// `HeadphoneOutputLevelMode::Normal`, and whether the speaker mutes
    /// itself when something else is playing.
    headphone_output_level_mode: u32,
    speaker_auto_mute: bool,
    /// `IAudioDevice`'s own output volume, which is a different setting from
    /// `master_volume` above and belongs to whoever last set it. Kept because
    /// a caller reads it straight back: the web applet sets a volume, gets the
    /// answer to `GetAudioDeviceOutputVolumeAuto`, and aborts if it is not the
    /// one it just asked for.
    device_volume: f32,
}

impl Default for AudioControl {
    fn default() -> AudioControl {
        AudioControl {
            volume: [AUDIO_VOLUME_MAX; AUDIO_TARGETS],
            muted: [false; AUDIO_TARGETS],
            output_mode: [AUDIO_OUTPUT_MODE_STEREO; AUDIO_TARGETS],
            default_target: AUDIO_TARGET_SPEAKER,
            master_volume: 1.0,
            force_mute_policy: 0,
            headphone_output_level_mode: 0,
            speaker_auto_mute: false,
            device_volume: 1.0,
        }
    }
}

impl AudioControl {
    /// `IAudioDevice`'s output volume, as last set.
    pub(super) fn device_volume(&self) -> f32 {
        self.device_volume
    }

    /// Record what a caller set it to. A volume that is not a real number is
    /// dropped rather than stored: it would come back out of the getter and
    /// fail the same comparison a wrong value would.
    pub(super) fn set_device_volume(&mut self, volume: f32) {
        if volume.is_finite() {
            self.device_volume = volume;
        }
    }
}

impl Cpu {
    /// `audren:u` (`IAudioRendererManager`): the factory for `IAudioRenderer`.
    /// Never converted to a domain (libnx builds it with
    /// `NX_SERVICE_ASSUME_NON_DOMAIN`), so `OpenAudioRenderer` hands its
    /// session out as a move handle, the same as `vi:m`/`nvdrv`.
    ///
    /// Answering `GetWorkBufferSize` with an empty reply (the old generic
    /// stub) left `workBufSize` as whatever garbage was already in that
    /// stack slot; `tmemCreate`ing a transfer memory block of that size
    /// reliably failed, and `audrenInitialize` — and so `SDL_OpenAudioDevice`,
    /// and so `JKSV::initialize_sdl`, and so `JKSV::JKSV()` itself — gave up
    /// before a single frame ever rendered.
    /// `IAudioOutManager` (`audout:u`): the plain PCM-out device, which is
    /// what `nn::audio::OpenDefaultAudioOut` and libnx's `audoutInitialize`
    /// open. The renderer (`audren`) is a separate, much larger interface.
    ///
    /// Only one device exists here, `DeviceOut`, at whatever rate and channel
    /// count the guest asks for. Real `audout` resamples everything to 48 kHz
    /// stereo; the samples are handed to the host verbatim instead, with the
    /// format alongside them, so nothing is resampled twice.
    pub(super) fn audout_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        // Both clients that reach this — `nnSdk` and libnx's `audoutInitialize`
        // — keep `audout` as a plain session and take the `IAudioOut` back as a
        // move handle. A domain request would need the reply to carry an object
        // id instead, so say so rather than hand back a handle it cannot use.
        if self.ipc_is_domain_request(tls) {
            return self.unimplemented_command(tls, "audout:u (domain)", cmd_id);
        }
        /// `AudioOutName`: a fixed 0x20-byte NUL-padded device name.
        const NAME_LEN: u32 = 0x20;
        /// The name real `audout` reports for the console's only output.
        const DEVICE: &[u8] = b"DeviceOut\0";
        match cmd_id {
            // ListAudioOuts / ListAudioOutsAuto: one device.
            Some(0) | Some(2) => {
                if let Some(buf) = self.ipc_output_buffer_addr(tls, 0) {
                    for i in 0..NAME_LEN {
                        let b = DEVICE.get(i as usize).copied().unwrap_or(0);
                        let _ = self.mem.write_u8(buf.wrapping_add(i), b);
                    }
                }
                self.write_ipc_response(tls, 0, &[], &1u32.to_le_bytes(), &[])
            }
            // OpenAudioOut / OpenAudioOutAuto: in { u32 sample_rate, u32
            // channel_count, u64 aruid }, out { u32 sample_rate, u32
            // channel_count, u32 pcm_format, u32 state } and the IAudioOut.
            Some(1) | Some(3) => {
                let data = self.ipc_request_data(tls);
                let asked_rate = self.mem.read_u32(data).unwrap_or(0);
                // The channel count is 16 bits wide on the wire and the two
                // bytes above it are padding the caller does not initialise.
                // Reading the whole word and echoing it back is how `nnSdk`
                // came to believe the device had 0xcafe0002 channels --
                // negative, so its "channelCount > 0" check failed, so Unity
                // tore audio down and re-opened, and the second open aborted
                // the process with `audio` result 2153-0009.
                let asked_channels = self.mem.read_u16(data.wrapping_add(4)).unwrap_or(0);
                // A guest that asks for 0 means "whatever the device is".
                let sample_rate = if asked_rate == 0 { 48_000 } else { asked_rate };
                let channel_count = u32::from(if asked_channels == 0 {
                    2
                } else {
                    asked_channels
                });

                if let Some(buf) = self.ipc_output_buffer_addr(tls, 0) {
                    for i in 0..NAME_LEN {
                        let b = DEVICE.get(i as usize).copied().unwrap_or(0);
                        let _ = self.mem.write_u8(buf.wrapping_add(i), b);
                    }
                }

                let handle = self.alloc_handle();
                self.record_handle(handle, "audout:iaudioout");
                let event = self.alloc_event("audout:buffer", true);
                self.audio_outs.insert(
                    handle,
                    AudioOut {
                        sample_rate,
                        channel_count,
                        started: false,
                        volume: 1.0,
                        event,
                        queued: VecDeque::new(),
                        free_at: 0,
                        played_frames: 0,
                    },
                );
                self.audio_format = (sample_rate, channel_count);

                let mut raw = Vec::with_capacity(16);
                raw.extend_from_slice(&sample_rate.to_le_bytes());
                raw.extend_from_slice(&channel_count.to_le_bytes());
                raw.extend_from_slice(&PCM_FORMAT_INT16.to_le_bytes());
                raw.extend_from_slice(&AUDIO_OUT_STOPPED.to_le_bytes());
                self.write_ipc_response(tls, 0, &[handle], &raw, &[])
            }
            _ => self.unimplemented_command(tls, "audout:u", cmd_id),
        }
    }

    /// How long `frames` samples take to play at `sample_rate`, in the
    /// emulated CPU cycles that are this machine's only clock. One instruction
    /// stands for one cycle of [`CLOCK_RATES_HZ`]'s first entry, the same rate
    /// the display tick and the thread deadlines are counted in.
    fn audio_play_cycles(frames: u64, sample_rate: u32) -> u64 {
        frames.saturating_mul(u64::from(CLOCK_RATES_HZ[0])) / u64::from(sample_rate.max(1))
    }

    /// `IAudioOut`: one open output device.
    ///
    /// The buffer protocol is the whole interface. The guest appends a buffer,
    /// waits on the event from `RegisterBufferEvent`, then collects the tags of
    /// the buffers the device has finished with. Here a buffer is finished the
    /// moment its samples have been copied out for the host, so every append
    /// releases immediately — a device that never falls behind.
    pub(super) fn audio_out_request(
        &mut self,
        tls: u32,
        cmd_id: Option<u32>,
        handle: u64,
    ) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        match cmd_id {
            // GetAudioOutState.
            Some(0) => {
                let started = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.started)
                    .unwrap_or(false);
                let state = if started {
                    AUDIO_OUT_STARTED
                } else {
                    AUDIO_OUT_STOPPED
                };
                self.write_ipc_response(tls, 0, &[], &state.to_le_bytes(), &[])
            }
            // StartAudioOut / StopAudioOut.
            Some(1) | Some(2) => {
                let started = cmd_id == Some(1);
                if let Some(device) = self.audio_outs.get_mut(&handle) {
                    device.started = started;
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // AppendAudioOutBuffer / AppendAudioOutBufferAuto.
            Some(3) | Some(7) => self.audio_out_append(tls, handle),
            // RegisterBufferEvent: the event a released buffer signals. Events
            // are copy handles.
            Some(4) => {
                let Some(event) = self.audio_outs.get(&handle).map(|d| d.event) else {
                    return self.unimplemented_command(tls, "audout:iaudioout", cmd_id);
                };
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // GetReleasedAudioOutBuffer / ...Auto: as many tags as fit.
            Some(5) | Some(8) => self.audio_out_release(tls, handle),
            // ContainsAudioOutBuffer.
            Some(6) => {
                let data = self.ipc_request_data(tls);
                let tag = self.mem.read_u64(data).unwrap_or(0);
                let held = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.queued.iter().any(|&(queued, _)| queued == tag))
                    .unwrap_or(false);
                self.write_ipc_response(tls, 0, &[], &[u8::from(held)], &[])
            }
            // GetAudioOutBufferCount: buffers appended and not yet collected.
            Some(9) => {
                let count = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.queued.len() as u32)
                    .unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &count.to_le_bytes(), &[])
            }
            // GetAudioOutPlayedSampleCount.
            Some(10) => {
                let frames = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.played_frames)
                    .unwrap_or(0);
                self.write_ipc_response(tls, 0, &[], &frames.to_le_bytes(), &[])
            }
            // FlushAudioOutBuffers: nothing is ever in flight, so nothing is
            // ever flushed — the bool says so.
            Some(11) => self.write_ipc_response(tls, 0, &[], &[0u8], &[]),
            // SetAudioOutVolume / GetAudioOutVolume.
            Some(12) => {
                let data = self.ipc_request_data(tls);
                let volume = f32::from_bits(self.mem.read_u32(data).unwrap_or(0));
                if let Some(device) = self.audio_outs.get_mut(&handle) {
                    device.volume = if volume.is_finite() {
                        volume.clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                }
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(13) => {
                let volume = self
                    .audio_outs
                    .get(&handle)
                    .map(|d| d.volume)
                    .unwrap_or(1.0);
                self.write_ipc_response(tls, 0, &[], &volume.to_bits().to_le_bytes(), &[])
            }
            _ => self.unimplemented_command(tls, "audout:iaudioout", cmd_id),
        }
    }

    /// Fire the buffer event of every device that has just finished playing
    /// something, and report the earliest cycle at which one of `handles` will
    /// have a buffer to hand back.
    ///
    /// This is the audio counterpart of the display tick in `svcWaitSynchron-
    /// ization`: nothing here runs in the background, so "the device finished a
    /// buffer" has to be noticed by somebody, and the guest asking to wait is
    /// the moment that matters. The deadline it returns is what makes that wait
    /// safe to honour — a waiter can be put to sleep knowing exactly when the
    /// device will wake it, which is the one property `Cpu::events` otherwise
    /// cannot offer.
    pub(super) fn audio_tick(&mut self, handles: &[u64]) -> Option<u64> {
        let now = self.cycles;
        let mut fire = Vec::new();
        let mut next = None;
        for device in self.audio_outs.values() {
            let Some(&(_, done_at)) = device.queued.front() else {
                continue;
            };
            if done_at <= now {
                fire.push(device.event);
            } else if handles.contains(&device.event) {
                next = Some(next.map_or(done_at, |soonest: u64| soonest.min(done_at)));
            }
        }
        for event in fire {
            self.signal_event(event);
        }
        // The renderer runs on a clock of its own — a frame every 5 ms rather
        // than a buffer every however long the guest made it — but a wait does
        // not care which of the two will wake it, only which comes first.
        let frame = self.audren_tick(handles);
        match (next, frame) {
            (Some(buffer), Some(frame)) => Some(buffer.min(frame)),
            (next, frame) => next.or(frame),
        }
    }

    /// `AppendAudioOutBuffer`: copy the guest's samples out for the host and
    /// queue the buffer for playback.
    fn audio_out_append(&mut self, tls: u32, handle: u64) -> Result<()> {
        let now = self.cycles;
        let data = self.ipc_request_data(tls);
        let tag = self.mem.read_u64(data).unwrap_or(0);
        // `AudioOutBuffer`: { next, buffer, buffer_size, data_size,
        // data_offset }, all 8 bytes, travelling as an input buffer.
        let mut samples = Vec::new();
        if let Some((desc, _)) = self.ipc_input_buffer(tls, 0) {
            // `AudioOutBuffer`: { next, buffer, buffer_size, data_size,
            // data_offset }, all 8 bytes, travelling as an input buffer.
            let buffer = self.mem.read_u64(desc.wrapping_add(8)).unwrap_or(0) as u32;
            let buffer_size = self.mem.read_u64(desc.wrapping_add(16)).unwrap_or(0) as u32;
            let data_size = self.mem.read_u64(desc.wrapping_add(24)).unwrap_or(0) as u32;
            let data_offset = self.mem.read_u64(desc.wrapping_add(32)).unwrap_or(0) as u32;
            if crate::trace::enabled(crate::trace::Trace::Audio) {
                crate::traceln!(
                    "[audio] append buffer={buffer:#x} cap={buffer_size:#x} \
                     size={data_size:#x} offset={data_offset:#x}"
                );
            }
            // `data_offset + data_size` has to fit inside `buffer_size`: that
            // is what the field means, and a device cannot play samples the
            // guest did not say were there.
            //
            // Checking it is not defensive tidiness. The Mii editor submits a
            // descriptor whose `buffer` is 6 and whose `data_offset` is a
            // pointer, and `buffer + data_offset` then lands *inside the
            // AudioOutBuffer struct itself* — so what reached the speakers was
            // the bytes of that struct's own pointers, read as PCM, several
            // thousand times a second. That is the buzzing. Hardware's audio
            // DSP has no way to produce sound from such a descriptor either.
            //
            // The buffer is still queued below, because the guest is entitled
            // to get it back however unplayable it was; only its samples are
            // dropped.
            let playable = buffer != 0
                && u64::from(data_offset) + u64::from(data_size) <= u64::from(buffer_size);
            if playable {
                let start = buffer.wrapping_add(data_offset);
                for i in 0..data_size / 2 {
                    let sample = self.mem.read_u16(start.wrapping_add(i * 2)).unwrap_or(0);
                    samples.push(sample as i16);
                }
            } else if self
                .unimplemented_ipc
                .insert(("audout:unplayable".to_string(), None))
            {
                crate::traceln!(
                    "[audio] refusing an unplayable buffer: {data_offset:#x}+{data_size:#x} \
                     is outside a {buffer_size:#x}-byte buffer at {buffer:#x}"
                );
            }
        }
        let Some(device) = self.audio_outs.get_mut(&handle) else {
            return self.unimplemented_command(tls, "audout:iaudioout", Some(3));
        };
        let channels = device.channel_count.max(1) as usize;
        let format = (device.sample_rate, device.channel_count);
        let frames = (samples.len() / channels) as u64;
        device.played_frames += frames;
        // Where this buffer plays: after whatever is still queued, or from now
        // if the device has caught up. `free_at` is what makes the guest's
        // audio clock advance at the same rate as its own.
        let starts_at = device.free_at.max(now);
        let rate = device.sample_rate;
        device.free_at = starts_at.wrapping_add(Self::audio_play_cycles(frames, rate));
        device.queued.push_back((tag, device.free_at));
        let volume = device.volume;
        // A stopped device is not playing: its buffers still come back (the
        // guest is entitled to its memory) but the samples are not queued.
        let playing = device.started;
        if playing {
            // Whichever device is actually producing samples defines the
            // format the host plays them in.
            self.audio_format = format;
            let scaled = samples
                .into_iter()
                .map(move |s| ((s as f32) * volume).round().clamp(-32768.0, 32767.0) as i16);
            self.queue_audio(scaled);
        }
        // Nothing is signalled here: the buffer event fires when a buffer
        // *finishes*, not when one arrives. See [`Cpu::audio_tick`].
        // Give up the CPU here. On hardware this call is a round trip into the
        // audio process and the caller is descheduled for its duration, and it
        // matters for scheduling too: the emulator only switches threads at a
        // blocking syscall, so a mixer that never blocks never yields. It
        // owned the CPU outright -- "A Short Hike"'s main thread was left
        // `Runnable` and unscheduled for a billion instructions while FMOD
        // converted float samples to 16-bit forever.
        self.pending_yield = true;
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// `GetReleasedAudioOutBuffer`: hand back the tags of the buffers the
    /// device has finished playing, as many as the guest's out buffer has room
    /// for. A buffer whose samples are still playing is not one of them.
    ///
    /// The entry after the last tag is zeroed, because `nn::audio`'s wrapper
    /// around this command **returns the first entry without looking at the
    /// count** — it never initialises the stack slot it points the receive
    /// buffer at, so an empty release leaves the caller reading whatever the
    /// previous call left on the stack. The Album applet's audio thread did
    /// exactly that: it took a `bl`'s return address for an `AudioOutBuffer`
    /// and wrote its de-interleaved samples over its own `.text`.
    fn audio_out_release(&mut self, tls: u32, handle: u64) -> Result<()> {
        let now = self.cycles;
        let out = self.ipc_output_buffer(tls, 0);
        let room = out.map(|(_, size)| size / 8).unwrap_or(0);
        let addr = out.map(|(address, _)| address);
        let mut tags = Vec::new();
        if let Some(device) = self.audio_outs.get_mut(&handle) {
            while (tags.len() as u32) < room {
                match device.queued.front() {
                    Some(&(tag, done_at)) if done_at <= now => {
                        device.queued.pop_front();
                        tags.push(tag);
                    }
                    _ => break,
                }
            }
        }
        if crate::trace::enabled(crate::trace::Trace::Audio) {
            crate::traceln!("[audio] release room={room} addr={addr:x?} tags={tags:#x?}");
        }
        if let Some(addr) = addr {
            for (i, &tag) in tags.iter().enumerate() {
                let _ = self.mem.write_u64(addr.wrapping_add(i as u32 * 8), tag);
            }
            if (tags.len() as u32) < room {
                let _ = self
                    .mem
                    .write_u64(addr.wrapping_add(tags.len() as u32 * 8), 0);
            }
        }
        self.write_ipc_response(tls, 0, &[], &(tags.len() as u32).to_le_bytes(), &[])
    }

    /// `audctl` — "nn::audioctrl::detail::IAudioController", the system-wide
    /// audio settings behind the volume buttons and the sound page of system
    /// settings.
    ///
    /// Everything here is a setting rather than a property of the hardware,
    /// which is why it is stored: the volume a caller sets is the volume the
    /// next caller reads, and a console that forgets between the two has a
    /// slider that snaps back. The facts this console does contribute are
    /// that the only output target is the speaker and the only layout is
    /// stereo — `audout` opens a two-channel device, and "one console, one
    /// answer" means `audctl` cannot claim 5.1.
    pub(super) fn audctl_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "audctl", cmd_id)? {
            return Ok(());
        }
        // Every per-target command names its target as a `u32`; which
        // argument holds it differs by command, so each arm reads its own.
        let target = |value: u32| (value as usize).min(AUDIO_TARGETS - 1);
        match cmd_id {
            // GetTargetVolume(target) -> s32, SetTargetVolume(target, s32).
            Some(0) => {
                let volume = self.audio_control.volume[target(self.ipc_arg_u32(tls, 0))];
                self.write_ipc_response(tls, 0, &[], &volume.to_le_bytes(), &[])
            }
            Some(1) => {
                let index = target(self.ipc_arg_u32(tls, 0));
                let volume = self.ipc_arg_u32(tls, 4) as i32;
                self.audio_control.volume[index] = volume.clamp(0, AUDIO_VOLUME_MAX);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // GetTargetVolumeMin / GetTargetVolumeMax. Both are fixed in the
            // firmware rather than derived from the output device.
            Some(2) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            Some(3) => self.write_ipc_response(tls, 0, &[], &AUDIO_VOLUME_MAX.to_le_bytes(), &[]),
            // IsTargetMute(target) -> bool, and SetTargetMute, whose bool
            // comes *before* its target.
            Some(4) => {
                let muted = u8::from(self.audio_control.muted[target(self.ipc_arg_u32(tls, 0))]);
                self.write_ipc_response(tls, 0, &[], &[muted], &[])
            }
            Some(5) => {
                let muted = self.ipc_arg_u8(tls, 0) != 0;
                let index = target(self.ipc_arg_u32(tls, 4));
                self.audio_control.muted[index] = muted;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // IsTargetConnected(target) -> bool, removed in 18.0.0. The
            // speaker is the only target that is there.
            Some(6) => {
                let connected = u8::from(self.ipc_arg_u32(tls, 0) == AUDIO_TARGET_SPEAKER);
                self.write_ipc_response(tls, 0, &[], &[connected], &[])
            }
            // SetDefaultTarget(target, ...) / GetDefaultTarget.
            Some(7) => {
                self.audio_control.default_target = self.ipc_arg_u32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(8) => {
                let default_target = self.audio_control.default_target;
                self.write_ipc_response(tls, 0, &[], &default_target.to_le_bytes(), &[])
            }
            // GetAudioOutputMode(target) / GetOutputModeSetting(target), and
            // their setters. The two pairs address the same per-target
            // layout: 9/10 are what the system is using, 13/14 what the user
            // chose, and with no HDMI to negotiate with they cannot differ.
            Some(9) | Some(13) => {
                let mode = self.audio_control.output_mode[target(self.ipc_arg_u32(tls, 0))];
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            Some(10) | Some(14) => {
                let index = target(self.ipc_arg_u32(tls, 0));
                self.audio_control.output_mode[index] = self.ipc_arg_u32(tls, 4);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SetForceMutePolicy(u32) / GetForceMutePolicy: whether the
            // speaker is cut when headphones are unplugged. Both were removed
            // after 13.2.1.
            Some(11) => {
                self.audio_control.force_mute_policy = self.ipc_arg_u32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(12) => {
                let policy = self.audio_control.force_mute_policy;
                self.write_ipc_response(tls, 0, &[], &policy.to_le_bytes(), &[])
            }
            // SetOutputTarget / SetInputTargetForceEnabled: both void, and
            // both address hardware routing there is none of here.
            Some(15) | Some(16) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetHeadphoneOutputLevelMode(u32) / Get.
            Some(17) => {
                self.audio_control.headphone_output_level_mode = self.ipc_arg_u32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(18) => {
                let mode = self.audio_control.headphone_output_level_mode;
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            // NotifyHeadphoneVolumeWarningDisplayedEvent, and
            // UpdateHeadphoneSettings(bool) — `ns` passes parental control's
            // restriction flag through the second one. Neither has an answer
            // beyond its Result.
            Some(22) | Some(26) => self.write_ipc_response(tls, 0, &[], &[], &[]),
            // SetSystemOutputMasterVolume(float) / Get.
            Some(23) => {
                self.audio_control.master_volume = self.ipc_arg_f32(tls, 0).clamp(0.0, 1.0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(24) => {
                let volume = self.audio_control.master_volume;
                self.write_ipc_response(tls, 0, &[], &volume.to_bits().to_le_bytes(), &[])
            }
            // SetSpeakerAutoMuteEnabled(bool) / IsSpeakerAutoMuteEnabled.
            Some(30) => {
                self.audio_control.speaker_auto_mute = self.ipc_arg_u8(tls, 0) != 0;
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(31) => {
                let enabled = u8::from(self.audio_control.speaker_auto_mute);
                self.write_ipc_response(tls, 0, &[], &[enabled], &[])
            }
            // GetActiveOutputTarget -> where sound is coming out right now,
            // which is not the same question as GetDefaultTarget: plugging in
            // headphones changes this one and not that one. Nothing can be
            // plugged in here.
            Some(32) => {
                self.write_ipc_response(tls, 0, &[], &AUDIO_TARGET_SPEAKER.to_le_bytes(), &[])
            }
            // AcquireTargetNotification -> the event that fires when the
            // output target changes. It cannot change.
            Some(34) => {
                let event = self.kept_event("audctl:target", handle);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // 19.0.0+, and unnamed: switchbrew lists the id and nothing else.
            // Eden's `audio/audio_controller.cpp` reads it as handing back a
            // second reference to the same `IAudioController`, which is what
            // this does — every setting above lives on the console rather than
            // on the object, so the two sessions cannot drift apart.
            Some(5000) => {
                self.reply_with_interface(tls, handle, "audctl")?;
                Ok(())
            }
            _ => self.unimplemented_command(tls, "audctl", cmd_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::ipc::testing::*;

    #[test]
    fn audctl_5000_hands_back_a_second_session_onto_the_same_settings() {
        // The command returns an interface, and `nnSdk` reads one as a move
        // handle: answered with a bare success it would read handle 0, skip
        // constructing the proxy, and fault on its first call through it.
        let mut cpu = request(false, 5000, &[]);
        cpu.register_service_handle(9, "audctl");
        cpu.audctl_request(TLS, 9, Some(5000)).unwrap();
        let duplicate = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(duplicate, 0, "audctl 5000 moved no session back");
        assert_eq!(cpu.service_name(duplicate), Some("audctl"));

        // Both sessions address the console's settings rather than the
        // object's, so a volume set through one reads back through the other.
        marshal(&mut cpu, false, 1, &[]);
        let _ = cpu.mem.write_u32(TLS + 0x20, 0);
        let _ = cpu.mem.write_u32(TLS + 0x24, 7);
        cpu.audctl_request(TLS, 9, Some(1)).unwrap();

        marshal(&mut cpu, false, 0, &[]);
        let _ = cpu.mem.write_u32(TLS + 0x20, 0);
        cpu.audctl_request(TLS, duplicate, Some(0)).unwrap();
        assert_eq!(cpu.mem.read_u32(TLS + 0x20).unwrap(), 7);
    }
}
