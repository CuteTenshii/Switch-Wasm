# Audio

The rules AGENTS.md summarises, in full: `cpu/audout.rs`, `cpu/audren.rs`,
`cpu/hwopus.rs` and `src/opus/`.

`audout` is a *device*: the guest hands it finished PCM; `audren` is a *mixer*:
the guest hands it sources and gets mixed PCM back. Homebrew mostly takes the
first, retail the second. Both end in `Cpu::queue_audio`, and whichever produced
samples last sets `Cpu::audio_format`.

**Both play in time, on the `cycles` clock the display and thread deadlines
use.** Releasing a buffer on arrival hands the guest an infinitely fast sound
card, and a title's audio clock is what its video is scheduled against.

- **`audout`** releases a buffer once the CPU has run for as long as its samples
  take at the device's rate, queued behind whatever is still playing; samples
  copy to the host on arrival, it is the *tag* that waits, and the buffer event
  fires on the clock (`Cpu::audio_tick`). **Do not answer that wait with a bare
  success**: `nn::audio`'s mixer takes the event as proof a buffer is waiting
  and reads its queue head unchecked.
- **`audren`** takes the whole renderer state as one flat buffer whose header
  declares each section's size, and `Cpu::audren_parse_update` walks it by
  **those declared sizes, not by strides computed here**: that is what makes one
  parser serve REV1 through REV15. Layout is libnx's `audren.h`. Signal path:
  wave buffers → decode (PCM8/16/24/32/float and Nintendo 4-bit ADPCM, which is
  what retail voices are) → linear resample by `rate × pitch` → per-voice biquads
  → per-channel gains into the destination mix → submixes, highest mix id first →
  the sink's channel map → interleaved i16.
- **A frame every 5 ms, counted off `cycles`** (`FRAME_CYCLES`), never one per
  update. `QuerySystemEvent` must hand back a **real event as a copy handle**: a
  bare handle is "not an event", which reads as always ready, so `audrenWaitFrame`
  returns instantly and the renderer has no clock.
- **`num_wavebufs_consumed` is load-bearing**: the guest advances its ring head
  by the delta, so a reply of zero is a title that queues four buffers and stops.
- **A renderer opens *started***; libnx never calls `StartAudioRenderer`.
- **Voice state is re-sent whole every update**: only position, ADPCM history and
  filter state survive, and `is_new` clears those.
- **`end_sample_offset` is a claim; `size` is the allocation, and it wins.** The
  buffer is still consumed: one that never comes back stalls its voice.
- Not modelled: effects (parsed for sizing), splitters (stepped over), the
  circular-buffer sink (reported once, never written). Each is a truthful zero.

**`hwopus`** (`cpu/hwopus.rs`, `src/opus/`) is the one service whose
implementation is a codec: there is nothing to answer *as*, the caller wants
audio back. **The packets are not bare Opus**: each carries an eight-byte
`{ size, final_range }` header, **big-endian**, and the reply's bytes-consumed
counts it. **SILK is integer arithmetic and has to be**: the filter a frame ends
with is what the next predicts from, so floats drift; CELT's is float and matches
the reference bit for bit on every unconcealed frame. `--example
opus_testvectors` runs the RFC 8251 vectors and requires the range coder's state
to match on *every* packet; samples only have to pass `opus_compare`.

## Notes by file

### `web/main/audio.ts`

- Each pump schedules everything queued since the last as one buffer, butted against the previous one. The emulator rarely runs retail titles in real time, so underruns are normal: the cursor restarts slightly ahead of `currentTime` instead of stretching audio.

### `crates/switch-core/src/opus/mdct.rs`

- The IMDCT follows CELT: pre-rotate by `exp(-i·2π(k+1/8)/N)`, forward complex FFT of `N/4` points, post-rotate, then mirror the ends (which is the overlap-add / TDAC).
- The FFT is a plain recursive mixed-radix Cooley-Tukey in natural order, not the reference's bit-reversal folded into the pre-rotation; it computes the same spectrum and is easier to verify. Radix 4 is tried before 2; every CELT size factors into 2, 3, 5, other primes fall back to one slow DFT level.

### `crates/switch-core/src/opus/mod.rs`

- The ported modules (celt, mdct, silk, tables) transcribe RFC 6716's decoder: tables are normative and integer rounding must match the encoder. Several clippy suggestions are actively wrong there (shortening a table constant or using `FRAC_1_SQRT_2` desynchronises the range decoder), so those lints are allowed only for the ported modules.
- Mode switches are faded (codec overlap windows do not line up, a hard cut clicks) and both decoders are kept warm. CELT is decoded even when its audio is unused because the final range depends on it; CELT state is reset after a mode change; leaving hybrid lets the MDCT fade the CELT half out.
- Loss concealment: whole 2.5 ms steps, only codec frame lengths (2.5/5/10/20 ms), SILK no shorter than 10 ms, keeping the last packet's channel count and internal rate. A one-byte payload is a silent-frame marker and is concealed. Packet parameters are committed only after a successful parse.
- Tests compare each packet's final range (proves the same symbols were read) and sample RMS against libopus; the fuzz test requires no panic on random/corrupted/truncated input since guests control the data.

### `crates/switch-core/src/opus/tables_silk.rs`

- Every resampler path is padded to the same total delay so a bandwidth switch does not shift the signal in time.

### `crates/switch-core/src/services/audout.rs`

- The device plays in time: a buffer is released once the emulated CPU has run as long as its samples take at the device rate, queued behind what is still playing (`free_at`). Releasing on arrival ran audio at 205x real time in Just Dance 2019 and its video player dropped every frame. A silent device restarts from now, so a submission gap is a gap, not a debt.
- `audio_tick` is the audio counterpart of the display tick in `svcWaitSynchronization`: it returns a deadline so a waiter can sleep knowing when the device will wake it; it also folds in the renderer's 5 ms frame clock.
- `AppendAudioOutBuffer` validates `data_offset + data_size <= buffer_size`: the Mii editor submits descriptors whose data points inside the `AudioOutBuffer` struct itself (buzzing). The buffer is still queued and returned.
- `AppendAudioOutBuffer` sets `pending_yield`: the emulator only switches threads at blocking syscalls, so a mixer that never blocks starves other threads (A Short Hike / FMOD).
- `GetReleasedAudioOutBuffer` zeroes the entry after the last tag because `nn::audio` returns the first entry without checking the count (Album applet wrote samples over its `.text`).
- `OpenAudioOut` channel count is u16 on the wire; the upper bytes are uninitialised padding (reading u32 gave 0xcafe0002 channels and Unity aborted with 2153-0009).
- `IAudioDevice` volume is stored and read back exactly: the web applet aborts if `GetAudioDeviceOutputVolumeAuto` differs from what it set.
- `audctl`: settings are stored per target; the only target is the speaker and the only layout stereo. Command 5000 (19.0.0+) returns another reference to the same controller; `nnSdk` reads the reply as a move handle.

### `crates/switch-core/src/services/audren.rs`

- The renderer is a mixer: the guest re-sends the whole state (mempools, channels, voices, mixes, sinks) every update as one flat buffer whose header declares each section's size. Sections are walked by the guest's declared sizes so one parser works across revisions (entries grew between revisions).
- The reply mirrors it; `audrvUpdate` and `nnSdk` check every section size against sizes computed from the counts the renderer was opened with, and abort on mismatch, every frame. Revision 5 added `RendererInfoOut`; revision 9 widened `EffectOutStatus` to 0x90.
- The renderer runs on the emulated clock (one frame per 5 ms of cycles) so a title's mixer is paced to real time. Backlog beyond 8 frames is dropped; a renderer restarted after stop or left unwaited starts from now. `audren_tick` returns the next deadline so waits can park safely. `QuerySystemEvent` must return a real copy-handle event, otherwise `audrenWaitFrame` never blocks.
- Renderers open started: libnx never calls `StartAudioRenderer`.
- `RequestUpdateAudioRenderer` sets `pending_yield` like `AppendAudioOutBuffer` so a non-blocking mixer cannot own the CPU.
- `GetAudioDeviceService` must return an object; a bare success left callers with a null vtable crash much later.
- Voice playback position (slot, offset, resampler, ADPCM history) persists across updates; the guest advances its ring head by the consumed count each reply reports, so re-seeding slot from `wavebuf_head` keeps both in step. A voice that never reports consumed buffers starves its title. Unrouted voices still advance.
- Mix buffers are assigned in mix-id order (final mix first), matching `audrvMixAdd`. Submixes render highest id first since a submix is created after the mix it sends to.
- `playable_samples` clamps `end_sample_offset` to the allocated size (same lesson as the audout Mii editor bug).
- At end of stream, interpolate toward zero (avoids a DC step) and keep `primed` so the last sample still plays.
- ADPCM is decoded from the start of the 14-sample frame when seeking; loops restart from the context's history. Output scaled by 32768 so unity 16-bit passes bit-exact.

### `crates/switch-core/src/opus/range.rs`

- Symbols are read forwards and raw bits backwards from the frame end, sharing one bit counter; `tell_frac` must match the encoder to 1/8 bit because CELT allocates bands from it.
- Reads past the end return zero by spec (RFC 6716 4.1), not leniency.

### `crates/switch-core/src/opus/silk.rs`

- In hybrid mode CELT carries everything above 8 kHz and is summed with SILK's output.
- Concealment pitch gain is clamped to a range that neither dies immediately nor rings forever; comfort noise tracks falls faster than rises because too-loud noise is more audible than too-quiet.
