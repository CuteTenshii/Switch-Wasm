//! `audout`, `audren` and `hwopus`.

mod cpu;

use cpu::*;

#[test]
fn audout_plays_the_buffers_the_guest_hands_it() {
    // `audout`'s buffer protocol: append, wait on the event, collect released tags.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000; // the AudioOutBuffer struct
    const PCM: u32 = 0x8100; // its samples
    const TAGS: u32 = 0x8200; // where released tags come back
    const TAG: u64 = 0xFEED_0001;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    // OpenAudioOut(48 kHz, stereo) -> { rate, channels, format, state } and an
    // IAudioOut move handle.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "OpenAudioOut failed"
    );
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 48_000);
    assert_eq!(cpu.mem.read_u32(tls + 0x24).unwrap(), 2);
    assert_eq!(cpu.mem.read_u32(tls + 0x28).unwrap(), 2, "PcmFormat::Int16");
    assert_eq!(
        cpu.mem.read_u32(tls + 0x2c).unwrap(),
        1,
        "a device opens stopped"
    );
    // { send_pid:1, num_copy:4, num_move:4 }: one move handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(device, 0, "no IAudioOut came back");

    // RegisterBufferEvent: a copy handle.
    ipc_request_plain(&mut cpu, device, 4, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 1);
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0);
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "event fired early"
    );

    // StartAudioOut, then hand over one buffer of four stereo frames.
    ipc_request_plain(&mut cpu, device, 1, &[]);
    ipc_request_plain(&mut cpu, device, 0, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 0, "started");

    let samples: [i16; 8] = [1, -1, 2, -2, 3, -3, 4, -4];
    for (i, &s) in samples.iter().enumerate() {
        cpu.mem.write_u16(PCM + i as u32 * 2, s as u16).unwrap();
    }
    // AudioOutBuffer { next, buffer, buffer_size, data_size, data_offset }.
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "AppendAudioOutBuffer failed"
    );

    // The samples reach the host on arrival; only the tag waits for the device.
    let mut played = [0i16; 8];
    assert_eq!(cpu.take_audio(&mut played), 8);
    assert_eq!(played, samples);

    // Not released until played: four frames at 48 kHz is 85,000 cycles at 1.02 GHz.
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "released before it could play"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "a tag came back early"
    );

    // Spend the cycles with a branch-to-self; the clock is the instruction count.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0,
        "the played buffer did not fire"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);

    // GetAudioOutPlayedSampleCount counts frames, not samples.
    ipc_request_plain(&mut cpu, device, 10, &[]);
    assert_eq!(cpu.mem.read_u64(tls + 0x20).unwrap(), 4);

    assert_eq!(cpu.audio_format(), (48_000, 2));

    // One device, started, one four-frame buffer in and out, eight samples.
    let activity = cpu.audio_activity();
    assert_eq!(
        (
            activity.sample_rate,
            activity.channels,
            activity.produced,
            activity.taken
        ),
        (48_000, 2, 8, 8)
    );
    assert_eq!((activity.dropped, activity.backlog), (0, 0));
    let [output] = activity.outputs.as_slice() else {
        panic!("{} devices reported", activity.outputs.len());
    };
    assert!(output.started);
    assert_eq!(
        (
            output.appended_buffers,
            output.appended_frames,
            output.released_buffers,
            output.pending_buffers,
            output.discarded_frames
        ),
        (1, 4, 1, 0, 0)
    );
}

#[test]
fn audout_release_zeroes_the_entry_after_the_last_tag() {
    // `nn::audio` reads the released tag from an uninitialised stack slot without
    // checking the count, so an empty release must write a zero terminator.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;
    const TAG: u64 = 0xFEED_0003;
    const GARBAGE: u64 = 0x0868_BBF8;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, device, 1, &[]); // StartAudioOut

    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());

    // Nothing has played, so the release is empty and terminated.
    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    cpu.mem.write_u64(TAGS + 8, GARBAGE).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "a tag came back early"
    );
    assert_eq!(
        cpu.mem.read_u64(TAGS).unwrap(),
        0,
        "the guest kept reading its own stack"
    );

    // Once played, the tag lands in the first slot and the zero after it.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    cpu.mem.write_u64(TAGS + 8, GARBAGE).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);
    assert_eq!(
        cpu.mem.read_u64(TAGS + 8).unwrap(),
        0,
        "no terminator after the last tag"
    );
}

#[test]
fn audout_release_answers_the_auto_commands_pointer_buffer() {
    // `GetReleasedAudioOutBufferAuto` offers a receive-static buffer and a null
    // map-alias descriptor; the reply must go to the former.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;
    const TAG: u64 = 0xFEED_0004;
    const GARBAGE: u64 = 0x0AA2_8F50;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, device, 1, &[]); // StartAudioOut

    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 16).unwrap();
    cpu.mem.write_u64(DESC + 24, 16).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &TAG.to_le_bytes());

    // The terminator reaches the guest's slot, not address 0.
    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    ipc_request_auto_recv(&mut cpu, device, 8, TAGS, 16, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        0,
        "a tag came back early"
    );
    assert_eq!(
        cpu.mem.read_u64(TAGS).unwrap(),
        0,
        "the pointer buffer was never written"
    );

    // And once played, so does the tag.
    const SPIN: u32 = 0x9000;
    cpu.mem.map(SPIN, &0x1400_0000u32.to_le_bytes()).unwrap(); // b .
    cpu.set_pc(SPIN);
    cpu.run(90_000).unwrap();
    cpu.set_pc(0x1000);

    cpu.mem.write_u64(TAGS, GARBAGE).unwrap();
    cpu.mem.write_u64(TAGS + 8, GARBAGE).unwrap();
    ipc_request_auto_recv(&mut cpu, device, 8, TAGS, 16, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "no tag released");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);
    assert_eq!(
        cpu.mem.read_u64(TAGS + 8).unwrap(),
        0,
        "no terminator after the last tag"
    );
}

#[test]
fn audren_update_reply_has_a_section_for_every_count_the_renderer_was_opened_with() {
    // `RequestUpdateAudioRenderer`'s reply is walked section by section against
    // caller-computed sizes, so every section must be present.
    const AUDREN: u64 = 0xB000;
    const OUT: u32 = 0x9000;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDREN, "audren:u");
    let tls = cpu.tls_base();

    // `AudioRendererParameter`: voices +16, sinks +20, effects +24, revision +48.
    let renderer_with = |cpu: &mut Cpu, revision: &[u8; 4]| -> u64 {
        let mut params = vec![0u8; 52];
        params[16..20].copy_from_slice(&2u32.to_le_bytes());
        params[20..24].copy_from_slice(&1u32.to_le_bytes());
        params[24..28].copy_from_slice(&3u32.to_le_bytes());
        params[48..52].copy_from_slice(revision);
        ipc_request_plain(cpu, AUDREN, 0, &params);
        u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap())
    };
    let section = |cpu: &Cpu, at: u32| cpu.mem.read_u32(OUT + at).unwrap();

    let renderer = renderer_with(&mut cpu, b"REV9");
    assert_ne!(renderer, 0, "no IAudioRenderer came back");
    ipc_request_plain_with_buffer(&mut cpu, renderer, 4, OUT, 0x1000, true, &[]);

    // MemPoolInfoOut per mempool (effects + four per voice), VoiceInfoOut per
    // voice, revision-9 EffectOutStatus per effect, SinkInfoOut per sink, then
    // the performance, behaviour and renderer-info tails.
    assert_eq!(section(&cpu, 0x08), (3 + 4 * 2) * 16, "mempools");
    assert_eq!(section(&cpu, 0x0c), 2 * 16, "voices");
    assert_eq!(section(&cpu, 0x14), 3 * 0x90, "effects");
    assert_eq!(section(&cpu, 0x1c), 32, "sinks");
    assert_eq!(section(&cpu, 0x20), 16, "performance");
    assert_eq!(section(&cpu, 0x04), 176, "behaviour");
    assert_eq!(section(&cpu, 0x28), 16, "renderer info");
    let total = 64 + 176 + 32 + 3 * 0x90 + 32 + 16 + 176 + 16;
    assert_eq!(section(&cpu, 0x3c), total, "total size");

    // Before revision 5: no renderer info, and the narrow effect status.
    let renderer = renderer_with(&mut cpu, b"REV4");
    ipc_request_plain_with_buffer(&mut cpu, renderer, 4, OUT, 0x1000, true, &[]);
    assert_eq!(section(&cpu, 0x14), 3 * 16, "revision-4 effects");
    assert_eq!(section(&cpu, 0x28), 0, "revision-4 renderer info");
    assert_eq!(
        section(&cpu, 0x3c),
        64 + 176 + 32 + 3 * 16 + 32 + 16 + 176,
        "revision-4 total"
    );
}

#[test]
fn audren_mixes_a_voice_through_to_the_host() {
    // The renderer mixes wave buffers into PCM.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const FRAMES: u32 = 240;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    // A ramp, so a resampler off-by-one shows as a shift.
    let samples: Vec<i16> = (0..FRAMES).map(|i| (i as i16 - 120) * 100).collect();
    for (i, &s) in samples.iter().enumerate() {
        cpu.mem.write_u16(PCM + i as u32 * 2, s as u16).unwrap();
    }

    let mut update = AudrenUpdate::new(1, 1, 1);
    update.voice(0, PCM_INT16, 1, PCM, FRAMES * 2, FRAMES);
    update.route(0, 0, 1.0);
    update.route(0, 1, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    // One frame of emulated time renders exactly one frame.
    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; FRAMES as usize * 2];
    assert_eq!(
        cpu.take_audio(&mut played),
        played.len(),
        "the mix never reached the host"
    );
    assert_eq!(cpu.audio_format(), (48_000, 2));
    // Mono into both mix buffers and outputs: the source doubled, bit-exact.
    for (i, &s) in samples.iter().enumerate() {
        assert_eq!(played[i * 2], s, "left channel at sample {i}");
        assert_eq!(played[i * 2 + 1], s, "right channel at sample {i}");
    }

    // No time elapsed renders nothing further.
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);
    let mut again = [0i16; 2];
    assert_eq!(
        cpu.take_audio(&mut again),
        0,
        "a frame was rendered that no time had come due for"
    );
}

#[test]
fn audren_reports_the_wave_buffers_it_finished_with() {
    // `num_wavebufs_consumed` drives the guest's refills.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const FRAMES: u32 = 240;
    /// The reply's voice section: past the header and four mempools per voice.
    const VOICE_OUT: u32 = 64 + 4 * 16;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    for i in 0..FRAMES {
        cpu.mem.write_u16(PCM + i * 2, 0x1234).unwrap();
    }
    let mut update = AudrenUpdate::new(1, 1, 1);
    update.voice(0, PCM_INT16, 1, PCM, FRAMES * 2, FRAMES);
    update.route(0, 0, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    assert_eq!(
        cpu.mem.read_u64(OUT + VOICE_OUT).unwrap(),
        u64::from(FRAMES),
        "played sample count"
    );
    assert_eq!(
        cpu.mem.read_u32(OUT + VOICE_OUT + 8).unwrap(),
        1,
        "the wave buffer never came back"
    );
}

#[test]
fn audren_decodes_the_adpcm_a_retail_voice_is_encoded_in() {
    // Nintendo 4-bit ADPCM: 14 samples per 8 bytes, a header byte (shift and
    // predictor pair) then seven bytes of nibbles.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const DATA: u32 = 0x5_0000;
    const COEFS: u32 = 0x5_1000;
    const SAMPLES: u32 = 28;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    // Pair 0 predicts nothing; pair 1 is 1.0 in Q11, adding the previous sample.
    let coefficients: [i16; 16] = [0, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, &c) in coefficients.iter().enumerate() {
        cpu.mem.write_u16(COEFS + i as u32 * 2, c as u16).unwrap();
    }

    // Frame 0: pair 0, shift 0, nibbles 1..7 then -8..-2.
    // Frame 1: pair 1, shift 0, every nibble 1, a running +1 from -2.
    let data: [u8; 16] = [
        0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11,
    ];
    for (i, &b) in data.iter().enumerate() {
        cpu.mem.write_u8(DATA + i as u32, b).unwrap();
    }
    let expected: [i16; 28] = [
        1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12,
    ];

    let mut update = AudrenUpdate::new(1, 1, 1);
    update.voice(0, PCM_ADPCM, 1, DATA, data.len() as u32, SAMPLES);
    update.extra_params(0, COEFS, 32);
    update.route(0, 0, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; 240 * 2];
    assert_eq!(cpu.take_audio(&mut played), played.len());
    for (i, &want) in expected.iter().enumerate() {
        assert_eq!(played[i * 2], want, "ADPCM sample {i}");
    }
    // Past the end the voice interpolates to silence rather than holding.
    assert_eq!(
        played[expected.len() * 2],
        0,
        "the voice kept playing past its data"
    );
}

#[test]
fn audren_frame_event_fires_on_the_clock() {
    // The renderer event paces `audrenWaitFrame` and must be a real event.
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);
    let tls = cpu.tls_base();

    // QuerySystemEvent -> a copy handle.
    ipc_request_plain(&mut cpu, renderer, 7, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x08).unwrap(),
        1 << 1,
        "not a copy handle"
    );
    let event = cpu.mem.read_u32(tls + 0x0c).unwrap();
    assert_ne!(event, 0, "no frame event came back");

    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0xEA01,
        "the frame event fired early"
    );

    // Five milliseconds later, it is.
    cpu.cycles += AUDREN_FRAME_CYCLES;
    assert_eq!(
        wait_sync(&mut cpu, &[event], 0).0,
        0,
        "the frame event never fired"
    );
}

#[test]
fn audren_refuses_a_wave_buffer_that_is_outside_its_allocation() {
    // Where `end_sample_offset` exceeds the allocation, the allocation wins;
    // the buffer is still consumed.
    const IN: u32 = 0x3_0000;
    const OUT: u32 = 0x4_0000;
    const PCM: u32 = 0x5_0000;
    const VOICE_OUT: u32 = 64 + 4 * 16;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    let renderer = audren_stereo(&mut cpu);

    for i in 0..240 {
        cpu.mem.write_u16(PCM + i * 2, 0x7FFF).unwrap();
    }
    let mut update = AudrenUpdate::new(1, 1, 1);
    // 240 samples claimed from a buffer with room for none.
    update.voice(0, PCM_INT16, 1, PCM, 0, 240);
    update.route(0, 0, 1.0);
    update.route(0, 1, 1.0);
    update.mix(2);
    update.sink(&[0, 1]);

    cpu.cycles += AUDREN_FRAME_CYCLES;
    update.send(&mut cpu, renderer, IN, OUT, 0x2000);

    let mut played = vec![0i16; 240 * 2];
    assert_eq!(
        cpu.take_audio(&mut played),
        played.len(),
        "the sink stopped producing frames"
    );
    assert!(
        played.iter().all(|&s| s == 0),
        "unplayable samples reached the host"
    );
    assert_eq!(
        cpu.mem.read_u32(OUT + VOICE_OUT + 8).unwrap(),
        1,
        "the buffer never came back"
    );
}

#[test]
fn audout_refuses_a_buffer_whose_samples_are_outside_it() {
    // `data_offset + data_size` must fit inside `buffer_size` (the Mii editor's
    // does not). The buffer still comes back; only its samples are dropped.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;
    const TAG: u64 = 0xFEED_0002;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    ipc_request_plain(&mut cpu, device, 1, &[]); // StartAudioOut

    // Playable data where the bad arithmetic would land, so a missing check is audible.
    for i in 0..8u32 {
        cpu.mem.write_u16(PCM + i * 2, 0x4000).unwrap();
    }
    // buffer_size is 8 bytes; data_offset alone is past it.
    cpu.mem.write_u64(DESC, 0).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 8).unwrap();
    cpu.mem.write_u64(DESC + 24, 8).unwrap();
    cpu.mem.write_u64(DESC + 32, 16).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 7, DESC, 40, false, &TAG.to_le_bytes());
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "the append is still accepted"
    );

    let mut played = [0i16; 8];
    assert_eq!(
        cpu.take_audio(&mut played),
        0,
        "unplayable samples reached the host"
    );

    // The guest still gets its buffer back.
    ipc_request_plain_with_buffer(&mut cpu, device, 8, TAGS, 16, true, &[]);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        1,
        "the buffer was never released"
    );
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), TAG);
}

#[test]
fn audout_reads_the_channel_count_as_sixteen_bits() {
    // `OpenAudioOut`'s channel count is 16 bits; the upper two bytes are padding.
    const AUDOUT: u64 = 0xA000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&0u32.to_le_bytes()); // sample rate: device default
    args.extend_from_slice(&0xcafe_0002u32.to_le_bytes()); // stereo, plus junk
    args.extend_from_slice(&0u64.to_le_bytes()); // aruid
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        48_000,
        "device default rate"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x24).unwrap(),
        2,
        "the padding leaked through"
    );
}

#[test]
fn audout_does_not_play_a_stopped_device() {
    // An unstarted device returns buffers but queues nothing for the host.
    const AUDOUT: u64 = 0xA000;
    const DESC: u32 = 0x8000;
    const PCM: u32 = 0x8100;
    const TAGS: u32 = 0x8200;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(AUDOUT, "audout:u");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, AUDOUT, 1, &args);
    let device = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    cpu.mem.write_u16(PCM, 0x1234).unwrap();
    cpu.mem.write_u64(DESC + 8, u64::from(PCM)).unwrap();
    cpu.mem.write_u64(DESC + 16, 2).unwrap();
    cpu.mem.write_u64(DESC + 24, 2).unwrap();
    cpu.mem.write_u64(DESC + 32, 0).unwrap();
    ipc_request_plain_with_buffer(&mut cpu, device, 3, DESC, 40, false, &7u64.to_le_bytes());

    let mut played = [0i16; 4];
    assert_eq!(
        cpu.take_audio(&mut played),
        0,
        "a stopped device played something"
    );
    ipc_request_plain_with_buffer(&mut cpu, device, 5, TAGS, 16, true, &[]);
    assert_eq!(cpu.mem.read_u32(tls + 0x20).unwrap(), 1, "released count");
    assert_eq!(cpu.mem.read_u64(TAGS).unwrap(), 7, "the tag");
}

#[test]
fn hwopus_reports_a_work_buffer_size_before_it_opens_anything() {
    // `nn::codec` allocates the reported work buffer size before opening a decoder.
    const HWOPUS: u64 = 0xC000;
    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.register_service_handle(HWOPUS, "hwopus");
    let tls = cpu.tls_base();

    // GetWorkBufferSizeEx { sample_rate, channel_count, use_large_frame_size }.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&2u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 5, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "GetWorkBufferSizeEx failed"
    );
    let stereo = cpu.mem.read_u32(tls + 0x20).unwrap();
    assert!(
        stereo > 0x1000,
        "a work buffer of {stereo:#x} bytes is not one"
    );

    // The large-frame form fits a 120 ms packet.
    args[8] = 1;
    ipc_request_plain(&mut cpu, HWOPUS, 5, &args);
    let large = cpu.mem.read_u32(tls + 0x20).unwrap();
    assert!(
        large > stereo,
        "the large-frame size {large:#x} is not above {stereo:#x}"
    );

    // An unsupported rate is refused.
    let mut bad = Vec::new();
    bad.extend_from_slice(&44_100u32.to_le_bytes());
    bad.extend_from_slice(&2u32.to_le_bytes());
    bad.extend_from_slice(&0u64.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 5, &bad);
    let result = cpu.mem.read_u32(tls + 0x18).unwrap();
    assert_eq!(result & 0x1FF, 111, "not an hwopus error: {result:#x}");
    assert_eq!(
        result >> 9,
        1001,
        "not the invalid-sample-rate error: {result:#x}"
    );
}

#[test]
fn hwopus_decodes_a_packet_into_the_buffer_the_caller_offered() {
    // Packets carry an eight-byte big-endian { size, final_range } header,
    // counted in "bytes consumed".
    const HWOPUS: u64 = 0xC000;
    const INPUT: u32 = 0x9000;
    const OUTPUT: u32 = 0x9400;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map_zero(INPUT, 0x400).unwrap();
    cpu.mem.map_zero(OUTPUT, 0x1000).unwrap();
    cpu.register_service_handle(HWOPUS, "hwopus");
    let tls = cpu.tls_base();

    // OpenHardwareOpusDecoderEx { rate, channels, large_frame } + work size.
    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&1u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    args.extend_from_slice(&0x8000u32.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 4, &args);
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "OpenHardwareOpusDecoderEx failed"
    );
    // { send_pid:1, num_copy:4, num_move:4 }: a move handle.
    assert_eq!(cpu.mem.read_u32(tls + 0x08).unwrap(), 1 << 5);
    let decoder = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());
    assert_ne!(decoder, 0, "no IHardwareOpusDecoder came back");

    // One 20 ms CELT-only packet, 48 kHz mono, from the reference encoder.
    let len = OPUS_PACKET.len() as u32;
    for (i, &byte) in (len).to_be_bytes().iter().enumerate() {
        cpu.mem.write_u8(INPUT + i as u32, byte).unwrap();
    }
    for (i, &byte) in OPUS_PACKET.iter().enumerate() {
        cpu.mem.write_u8(INPUT + 8 + i as u32, byte).unwrap();
    }

    // DecodeInterleaved: reset flag in, { bytes read, samples } out.
    ipc_request_plain_with_both_buffers(
        &mut cpu,
        decoder,
        8,
        (INPUT, 8 + len),
        (OUTPUT, 0x1000),
        &[0u8, 0, 0, 0],
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x18).unwrap(),
        0,
        "DecodeInterleaved failed"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x20).unwrap(),
        8 + len,
        "wrong byte count"
    );
    assert_eq!(
        cpu.mem.read_u32(tls + 0x24).unwrap(),
        960,
        "a 20 ms frame is 960 samples"
    );

    // The samples are not all zero.
    let loudest = (0..960)
        .map(|i| i32::from(cpu.mem.read_u16(OUTPUT + i * 2).unwrap() as i16).abs())
        .max()
        .unwrap();
    assert!(
        loudest > 1000,
        "the decode is silent (loudest sample {loudest})"
    );
}

#[test]
fn hwopus_refuses_a_packet_shorter_than_its_own_header() {
    // A header size beyond the buffer, or no room for a header, is an error.
    const HWOPUS: u64 = 0xC000;
    const INPUT: u32 = 0x9000;
    const OUTPUT: u32 = 0x9400;

    let mut cpu = cpu_at(0x1000);
    cpu.bootstrap();
    cpu.set_pc(0x1000);
    cpu.mem.map_zero(INPUT, 0x400).unwrap();
    cpu.mem.map_zero(OUTPUT, 0x1000).unwrap();
    cpu.register_service_handle(HWOPUS, "hwopus");
    let tls = cpu.tls_base();

    let mut args = Vec::new();
    args.extend_from_slice(&48_000u32.to_le_bytes());
    args.extend_from_slice(&1u32.to_le_bytes());
    args.extend_from_slice(&0u64.to_le_bytes());
    args.extend_from_slice(&0x8000u32.to_le_bytes());
    ipc_request_plain(&mut cpu, HWOPUS, 4, &args);
    let decoder = u64::from(cpu.mem.read_u32(tls + 0x0c).unwrap());

    // A header claiming more payload than the buffer holds.
    for (i, &byte) in 0x1000u32.to_be_bytes().iter().enumerate() {
        cpu.mem.write_u8(INPUT + i as u32, byte).unwrap();
    }
    ipc_request_plain_with_both_buffers(
        &mut cpu,
        decoder,
        8,
        (INPUT, 64),
        (OUTPUT, 0x1000),
        &[0u8; 4],
    );
    let result = cpu.mem.read_u32(tls + 0x18).unwrap();
    assert_eq!(result & 0x1FF, 111, "not an hwopus error: {result:#x}");
    assert_eq!(
        result >> 9,
        3,
        "not the buffer-too-small error: {result:#x}"
    );

    // A buffer with nothing but the header in it.
    ipc_request_plain_with_both_buffers(
        &mut cpu,
        decoder,
        8,
        (INPUT, 8),
        (OUTPUT, 0x1000),
        &[0u8; 4],
    );
    let result = cpu.mem.read_u32(tls + 0x18).unwrap();
    assert_eq!(result >> 9, 8, "not the input-too-small error: {result:#x}");
}
