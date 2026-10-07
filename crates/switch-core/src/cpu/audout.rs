//! `audout`, the plain PCM-out device, and `audctl`, the system-wide audio
//! settings. The renderer is [`super::audren`].

use super::power::CLOCK_RATES_HZ;
use super::Cpu;
use crate::Result;
use std::collections::VecDeque;

/// One open `IAudioOut` session. Buffers are released once the emulated CPU
/// has run for as long as their samples take to play.
#[derive(Debug, Clone)]
pub(crate) struct AudioOut {
    /// Sample rate and channel count the device was opened with.
    pub sample_rate: u32,
    pub channel_count: u32,
    pub started: bool,
    /// The volume the guest set, 0.0..=1.0. Applied when samples are taken.
    pub volume: f32,
    /// Signalled every time a buffer is released.
    pub event: u64,
    /// Appended, uncollected buffers with the cycle at which each finishes playing.
    pub queued: VecDeque<(u64, u64)>,
    /// The cycle the device finishes everything queued so far.
    pub free_at: u64,
    /// Reported by `GetAudioOutPlayedSampleCount`.
    pub played_frames: u64,
    /// Activity report counters since the device was opened.
    pub appended_buffers: u64,
    pub appended_frames: u64,
    pub released_buffers: u64,
    pub discarded_frames: u64,
    pub unplayable_buffers: u64,
}

/// `nn::audio::PcmFormat`: 16-bit signed samples.
const PCM_FORMAT_INT16: u32 = 2;

/// `nn::audio::AudioOutState`, as `IAudioOut` reports it.
const AUDIO_OUT_STARTED: u32 = 0;

const AUDIO_OUT_STOPPED: u32 = 1;

/// `AudioOutputModeTarget` count: None, Hdmi, Speaker, Headphone, Type3, Type4.
pub(super) const AUDIO_TARGETS: usize = 6;

/// `AudioOutputModeTarget::Speaker`, the only target this console has.
pub(super) const AUDIO_TARGET_SPEAKER: u32 = 2;

/// `AudioOutputMode::ch_2`, stereo.
pub(super) const AUDIO_OUTPUT_MODE_STEREO: u32 = 1;

pub(super) const AUDIO_VOLUME_MAX: i32 = 15;

/// `audctl`'s stored system-wide audio settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AudioControl {
    volume: [i32; AUDIO_TARGETS],
    muted: [bool; AUDIO_TARGETS],
    output_mode: [u32; AUDIO_TARGETS],
    default_target: u32,
    master_volume: f32,
    /// `ForceMutePolicy::Disable`.
    force_mute_policy: u32,
    /// `HeadphoneOutputLevelMode::Normal`.
    headphone_output_level_mode: u32,
    speaker_auto_mute: bool,
    /// `IAudioDevice`'s output volume, distinct from `master_volume`.
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
    pub(super) fn device_volume(&self) -> f32 {
        self.device_volume
    }

    /// Non-finite volumes are dropped.
    pub(super) fn set_device_volume(&mut self, volume: f32) {
        if volume.is_finite() {
            self.device_volume = volume;
        }
    }
}

impl Cpu {
    /// `IAudioOutManager` (`audout:u`). One device, `DeviceOut`; samples go to the
    /// host at the guest's rate and channel count, unresampled.
    pub(super) fn audout_request(&mut self, tls: u32, cmd_id: Option<u32>) -> Result<()> {
        if self.ipc_is_control_request(tls) {
            return self.write_ipc_response(tls, 0, &[], &[], &[]);
        }
        // Clients keep `audout` as a plain session; a domain request is unsupported.
        if self.ipc_is_domain_request(tls) {
            return self.unimplemented_command(tls, "audout:u (domain)", cmd_id);
        }
        /// `AudioOutName`: a fixed 0x20-byte NUL-padded device name.
        const NAME_LEN: u32 = 0x20;
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
            // OpenAudioOut / OpenAudioOutAuto.
            Some(1) | Some(3) => {
                let data = self.ipc_request_data(tls);
                let asked_rate = self.mem.read_u32(data).unwrap_or(0);
                // The channel count is 16 bits; the upper two bytes are uninitialised padding.
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
                        appended_buffers: 0,
                        appended_frames: 0,
                        released_buffers: 0,
                        discarded_frames: 0,
                        unplayable_buffers: 0,
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

    /// How long `frames` samples take to play at `sample_rate`, in emulated cycles.
    fn audio_play_cycles(frames: u64, sample_rate: u32) -> u64 {
        frames.saturating_mul(u64::from(CLOCK_RATES_HZ[0])) / u64::from(sample_rate.max(1))
    }

    /// `IAudioOut`: one open output device.
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
            // RegisterBufferEvent.
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
            // FlushAudioOutBuffers: nothing is ever in flight.
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

    /// Fire the buffer event of every device that has finished a buffer, and
    /// return the earliest cycle at which one of `handles` will have one to hand back.
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
        let frame = self.audren_tick(handles);
        match (next, frame) {
            (Some(buffer), Some(frame)) => Some(buffer.min(frame)),
            (next, frame) => next.or(frame),
        }
    }

    fn audio_out_append(&mut self, tls: u32, handle: u64) -> Result<()> {
        let now = self.cycles;
        let data = self.ipc_request_data(tls);
        let tag = self.mem.read_u64(data).unwrap_or(0);
        let mut samples = Vec::new();
        let mut unplayable = false;
        if let Some((desc, _)) = self.ipc_input_buffer(tls, 0) {
            // `AudioOutBuffer`: { next, buffer, buffer_size, data_size, data_offset }, all u64.
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
            // Out-of-bounds data is dropped, but the buffer is still queued and returned.
            let playable = buffer != 0
                && u64::from(data_offset) + u64::from(data_size) <= u64::from(buffer_size);
            unplayable = !playable;
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
        device.appended_buffers += 1;
        device.appended_frames += frames;
        device.unplayable_buffers += u64::from(unplayable);
        if !device.started {
            device.discarded_frames += frames;
        }
        let starts_at = device.free_at.max(now);
        let rate = device.sample_rate;
        device.free_at = starts_at.wrapping_add(Self::audio_play_cycles(frames, rate));
        device.queued.push_back((tag, device.free_at));
        let volume = device.volume;
        // A stopped device still returns its buffers but does not queue samples.
        let playing = device.started;
        if playing {
            self.audio_format = format;
            let scaled = samples
                .into_iter()
                .map(move |s| ((s as f32) * volume).round().clamp(-32768.0, 32767.0) as i16);
            self.queue_audio(scaled);
        }
        // The buffer event fires on finish (see [`Cpu::audio_tick`]). Yield so the
        // caller is descheduled as it would be on hardware.
        self.pending_yield = true;
        self.write_ipc_response(tls, 0, &[], &[], &[])
    }

    /// `GetReleasedAudioOutBuffer`: hand back the tags of finished buffers. The entry
    /// after the last tag is zeroed because `nn::audio` reads it without checking the count.
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
            device.released_buffers += tags.len() as u64;
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

    /// `audctl` (`nn::audioctrl::detail::IAudioController`): system-wide audio settings.
    pub(super) fn audctl_request(
        &mut self,
        tls: u32,
        handle: u64,
        cmd_id: Option<u32>,
    ) -> Result<()> {
        if self.ipc_answer_control(tls, handle, "audctl", cmd_id)? {
            return Ok(());
        }
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
            // GetTargetVolumeMin / GetTargetVolumeMax.
            Some(2) => self.write_ipc_response(tls, 0, &[], &0i32.to_le_bytes(), &[]),
            Some(3) => self.write_ipc_response(tls, 0, &[], &AUDIO_VOLUME_MAX.to_le_bytes(), &[]),
            // IsTargetMute(target) -> bool, and SetTargetMute(bool, target).
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
            // IsTargetConnected(target) -> bool. Only the speaker is connected.
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
            // GetAudioOutputMode / GetOutputModeSetting and their setters, sharing one layout.
            Some(9) | Some(13) => {
                let mode = self.audio_control.output_mode[target(self.ipc_arg_u32(tls, 0))];
                self.write_ipc_response(tls, 0, &[], &mode.to_le_bytes(), &[])
            }
            Some(10) | Some(14) => {
                let index = target(self.ipc_arg_u32(tls, 0));
                self.audio_control.output_mode[index] = self.ipc_arg_u32(tls, 4);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            // SetForceMutePolicy(u32) / GetForceMutePolicy.
            Some(11) => {
                self.audio_control.force_mute_policy = self.ipc_arg_u32(tls, 0);
                self.write_ipc_response(tls, 0, &[], &[], &[])
            }
            Some(12) => {
                let policy = self.audio_control.force_mute_policy;
                self.write_ipc_response(tls, 0, &[], &policy.to_le_bytes(), &[])
            }
            // SetOutputTarget / SetInputTargetForceEnabled.
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
            // NotifyHeadphoneVolumeWarningDisplayedEvent / UpdateHeadphoneSettings(bool).
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
            // GetActiveOutputTarget.
            Some(32) => {
                self.write_ipc_response(tls, 0, &[], &AUDIO_TARGET_SPEAKER.to_le_bytes(), &[])
            }
            // AcquireTargetNotification: the target cannot change.
            Some(34) => {
                let event = self.kept_event("audctl:target", handle);
                self.write_ipc_reply(tls, 0, &[event], &[], &[], &[])
            }
            // 19.0.0+, unnamed: hands back another reference to the same `IAudioController`.
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
        // Answered without a handle, `nnSdk` would read handle 0 and fault.
        let mut cpu = request(false, 5000, &[]);
        cpu.register_service_handle(9, "audctl");
        cpu.audctl_request(TLS, 9, Some(5000)).unwrap();
        let duplicate = u64::from(cpu.mem.read_u32(TLS + 0x0c).unwrap());
        assert_ne!(duplicate, 0, "audctl 5000 moved no session back");
        assert_eq!(cpu.service_name(duplicate), Some("audctl"));

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
