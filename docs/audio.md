# Audio

The rules AGENTS.md summarises, in full — `cpu/audout.rs`, `cpu/audren.rs`,
`cpu/hwopus.rs` and `src/opus/`.

`audout` is a *device* — the guest hands it finished PCM; `audren` is a *mixer* —
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
  success** — `nn::audio`'s mixer takes the event as proof a buffer is waiting
  and reads its queue head unchecked.
- **`audren`** takes the whole renderer state as one flat buffer whose header
  declares each section's size, and `Cpu::audren_parse_update` walks it by
  **those declared sizes, not by strides computed here** — that is what makes one
  parser serve REV1 through REV15. Layout is libnx's `audren.h`. Signal path:
  wave buffers → decode (PCM8/16/24/32/float and Nintendo 4-bit ADPCM, which is
  what retail voices are) → linear resample by `rate × pitch` → per-voice biquads
  → per-channel gains into the destination mix → submixes, highest mix id first →
  the sink's channel map → interleaved i16.
- **A frame every 5 ms, counted off `cycles`** (`FRAME_CYCLES`), never one per
  update. `QuerySystemEvent` must hand back a **real event as a copy handle**: a
  bare handle is "not an event", which reads as always ready, so `audrenWaitFrame`
  returns instantly and the renderer has no clock.
- **`num_wavebufs_consumed` is load-bearing** — the guest advances its ring head
  by the delta, so a reply of zero is a title that queues four buffers and stops.
- **A renderer opens *started***; libnx never calls `StartAudioRenderer`.
- **Voice state is re-sent whole every update**: only position, ADPCM history and
  filter state survive, and `is_new` clears those.
- **`end_sample_offset` is a claim; `size` is the allocation, and it wins.** The
  buffer is still consumed — one that never comes back stalls its voice.
- Not modelled: effects (parsed for sizing), splitters (stepped over), the
  circular-buffer sink (reported once, never written). Each is a truthful zero.

**`hwopus`** (`cpu/hwopus.rs`, `src/opus/`) is the one service whose
implementation is a codec — there is nothing to answer *as*, the caller wants
audio back. **The packets are not bare Opus**: each carries an eight-byte
`{ size, final_range }` header, **big-endian**, and the reply's bytes-consumed
counts it. **SILK is integer arithmetic and has to be** — the filter a frame ends
with is what the next predicts from, so floats drift; CELT's is float and matches
the reference bit for bit on every unconcealed frame. `--example
opus_testvectors` runs the RFC 8251 vectors and requires the range coder's state
to match on *every* packet; samples only have to pass `opus_compare`.

