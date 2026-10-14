# Audio

`services/audout.rs`, `services/audren.rs`, `services/hwopus.rs` and `src/opus/`.

`audout` is a *device*: the guest hands it finished PCM. `audren` is a *mixer*: the guest hands it sources and gets mixed PCM back. Homebrew mostly takes the first, retail the second. Both end in `Cpu::queue_audio`, and whichever produced samples last sets `Cpu::audio_format`.

**Both play in time, on the `cycles` clock the display and thread deadlines use.** Releasing a buffer on arrival hands the guest an infinitely fast sound card, and a title schedules its video against its audio clock: releasing on arrival ran audio at 205x real time in Just Dance 2019 and its video player dropped every frame.

## `crates/switch-core/src/services/audout.rs`

- A buffer is released once the CPU has run for as long as its samples take at the device's rate, queued behind whatever is still playing (`free_at`). Samples copy to the host on arrival; it is the *tag* that waits, and the buffer event fires on the clock. A silent device restarts from now, so a submission gap is a gap, not a debt.
- **Do not answer the buffer-event wait with a bare success**: `nn::audio`'s mixer takes the event as proof a buffer is waiting and reads its queue head unchecked.
- `audio_tick` is the audio counterpart of the display tick in `svcWaitSynchronization`: it returns a deadline so a waiter can sleep knowing when the device will wake it, and folds in the renderer's frame clock (`audren_tick`).
- `AppendAudioOutBuffer` validates `data_offset + data_size <= buffer_size`: the Mii editor submits descriptors whose data points inside the `AudioOutBuffer` struct itself (buzzing). The buffer is still queued and returned.
- `AppendAudioOutBuffer` sets `pending_yield`: the emulator only switches threads at blocking syscalls, so a mixer that never blocks starves other threads (A Short Hike / FMOD).
- `GetReleasedAudioOutBuffer` zeroes the entry after the last tag because `nn::audio` returns the first entry without checking the count (the Album applet wrote samples over its `.text`).
- `OpenAudioOut`'s channel count is u16 on the wire; the upper bytes are uninitialised padding (reading u32 gave 0xcafe0002 channels and Unity aborted with 2153-0009).
- `audctl`: settings are stored per target; the only target is the speaker and the only layout stereo. Command 5000 (19.0.0+) returns another reference to the same controller; `nnSdk` reads the reply as a move handle.

## `crates/switch-core/src/services/audren.rs`

- The guest re-sends the whole renderer state (mempools, channels, voices, mixes, sinks) every update as one flat buffer whose header declares each section's size. `audren_parse_update` walks it **by those declared sizes, not by strides computed here**: that is what makes one parser serve every `REVn` (entries grew between revisions). Layout is libnx's `audren.h`.
- The reply mirrors it; `audrvUpdate` and `nnSdk` check every section size against sizes computed from the counts the renderer was opened with, and abort on mismatch, every frame. Revision 5 added `RendererInfoOut`; revision 9 widened `EffectOutStatus` to 0x90.
- Signal path: wave buffers, decode (PCM8/16/24/32/float and Nintendo 4-bit ADPCM, which is what retail voices are), linear resample by `rate × pitch`, per-voice biquads, per-channel gains into the destination mix, submixes highest mix id first, the sink's channel map, interleaved i16.
- **A frame every 5 ms of `cycles`** (`FRAME_CYCLES`), never one per update, so a title's mixer is paced to real time. Backlog beyond 8 frames is dropped; a renderer restarted after stop or left unwaited starts from now. `audren_tick` returns the next deadline so waits can park safely.
- `QuerySystemEvent` must hand back a **real event as a copy handle**: a bare handle is "not an event", which reads as always ready, so `audrenWaitFrame` returns instantly and the renderer has no clock.
- **A renderer opens started**; libnx never calls `StartAudioRenderer`.
- `RequestUpdateAudioRenderer` sets `pending_yield` like `AppendAudioOutBuffer` so a non-blocking mixer cannot own the CPU.
- `GetAudioDeviceService` must return an object; a bare success left callers with a null vtable crash much later.
- `IAudioDevice` volume is stored and read back exactly: the web applet aborts if the volume it reads differs from what it set.
- **`num_wavebufs_consumed` is load-bearing**: the guest advances its ring head by the delta, so a reply of zero is a title that queues four buffers and stops. Unrouted voices still advance.
- Voice state is re-sent whole every update; only playback position (slot, offset, resampler, ADPCM history) and filter state survive, and `is_new` clears them. Re-seeding the slot from `wavebuf_head` keeps the guest's ring and ours in step.
- Mix buffers are assigned in mix-id order (final mix first), matching `audrvMixAdd`. Submixes render highest id first since a submix is created after the mix it sends to.
- **`end_sample_offset` is a claim; `size` is the allocation, and it wins** (`playable_samples`, the same lesson as the audout Mii editor bug). The buffer is still consumed: one that never comes back stalls its voice.
- At end of stream, interpolate toward zero (avoids a DC step) and keep `primed` so the last sample still plays.
- ADPCM is decoded from the start of the 14-sample frame when seeking; loops restart from the context's history. Output is scaled by 32768 so unity 16-bit passes bit-exact.
- Not modelled: effects (parsed for sizing), splitters (stepped over), the circular-buffer sink (reported once, never written). Each is a truthful zero.

## `crates/switch-core/src/services/hwopus.rs`

- The one service whose implementation is a codec: there is nothing to answer *as*, the caller wants audio back.
- **The packets are not bare Opus**: each carries an eight-byte `{ size, final_range }` header, **big-endian**, and the reply's bytes-consumed counts it.

## `crates/switch-core/src/opus/mod.rs`

- The ported modules (celt, mdct, silk, tables) transcribe RFC 6716's decoder: tables are normative and integer rounding must match the encoder. Several clippy suggestions are actively wrong there (shortening a table constant or using `FRAC_1_SQRT_2` desynchronises the range decoder), so those lints are allowed only for the ported modules.
- **SILK is integer arithmetic and has to be**: the filter a frame ends with is what the next predicts from, so floats drift. CELT's is float and matches the reference bit for bit on every unconcealed frame.
- Mode switches are faded (codec overlap windows do not line up, a hard cut clicks) and both decoders are kept warm. CELT is decoded even when its audio is unused because the final range depends on it; CELT state is reset after a mode change; leaving hybrid lets the MDCT fade the CELT half out.
- Loss concealment: whole 2.5 ms steps, only codec frame lengths (2.5/5/10/20 ms), SILK no shorter than 10 ms, keeping the last packet's channel count and internal rate. A one-byte payload is a silent-frame marker and is concealed. Packet parameters are committed only after a successful parse.
- Tests compare each packet's final range (proves the same symbols were read) and sample RMS against libopus; `survives_arbitrary_input` requires no panic on random, corrupted or truncated input since guests control the data. `opus_testvectors` (see [repro.md](repro.md)) requires the range coder's state to match on *every* packet; samples only have to pass `opus_compare`.

## `crates/switch-core/src/opus/range.rs`

- Symbols are read forwards and raw bits backwards from the frame end, sharing one bit counter; `tell_frac` must match the encoder to 1/8 bit because CELT allocates bands from it.
- Reads past the end return zero by spec (RFC 6716 4.1), not leniency.

## `crates/switch-core/src/opus/mdct.rs`

- The IMDCT follows CELT: pre-rotate by `exp(-i·2π(k+1/8)/N)`, forward complex FFT of `N/4` points, post-rotate, then mirror the ends (which is the overlap-add / TDAC).
- The FFT is a plain recursive mixed-radix Cooley-Tukey in natural order, not the reference's bit-reversal folded into the pre-rotation; it computes the same spectrum and is easier to verify. Radix 4 is tried before 2; every CELT size factors into 2, 3 and 5, and other primes fall back to one slow DFT level.

## `crates/switch-core/src/opus/silk/`

- In hybrid mode CELT carries everything above 8 kHz and is summed with SILK's output.
- Every resampler path is padded to the same total delay (`RESAMPLER_DELAY_MATRIX_DEC`, in `opus/tables_silk.rs`) so a bandwidth switch does not shift the signal in time.
- Concealment pitch gain is clamped to a range that neither dies immediately nor rings forever; comfort noise tracks falls faster than rises because too-loud noise is more audible than too-quiet.

## `web/main/audio.ts`

- Each pump schedules everything queued since the last as one buffer, butted against the previous one. The emulator rarely runs retail titles in real time, so underruns are normal: the cursor restarts slightly ahead of `currentTime` instead of stretching audio.
