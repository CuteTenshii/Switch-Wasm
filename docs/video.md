# Video decoding

Titles decode video on the console's nvdec engine and hand the frames to VIC. `gpu/multimedia.rs` models both. VP9 (codec 9) goes to the browser's decoder as frames rebuilt from what nvdec is given (`vp9/rebuild.rs`), with `vp9/` as the software decoder where the browser has none or gives up; H.264 (codec 3) goes to the browser's decoder (`gpu/h264.rs`). Both browser decoders are `WebDecoder` in `switch-wasm/src/video.rs`. Other codecs, and H.264 in a native build, only retire their syncpoints.

## The decoder

`vp9/` is bit-exact with libvpx v1.17.0 and checked against it, not against ffmpeg alone:

- `tools/extract_vp9_tables.py <libvpx checkout> crates/switch-core/src/vp9/tables.rs` writes the constant tables from the C initializers; `tools/extract_vp9_transforms.py` translates the 1-D inverse transforms. Both are pure Python; neither is needed to build.
- `cargo run --release -p switch-core --example vp9_check -- <dir>` decodes every `x.ivf` beside an `x.yuv` that ffmpeg wrote (`ffmpeg -i x.ivf -f rawvideo -pix_fmt yuv420p x.yuv`) and compares every shown frame. The clips covered tiles, hidden frames in superframes, segmentation, error-resilient and frame-parallel streams, odd sizes, lossless and compound prediction.
- `vp9_vectors idct|intra <file>` checks the transforms and the intra predictors against the output of a small C harness linked to `libvpx.a` (random blocks through `vp9_iht*_add_c`, `vpx_idct32x32_1024_add_c`, `vpx_iwht4x4_16_add_c` and `vpx_*_predictor_*_c`).
- `vp9_capture <dir> <out.ppm>` decodes the first frame of a captured nvdec stream (`c10_001_{setup,bits,probs}.bin`) to a PPM.
- `cargo run --release -p switch-core --example vp9_bench -- <clip.ivf>` times the decoder natively.
- Unsupported: scaled reference frames (an inter frame whose references differ from it in size; a key or intra-only frame may change size), profiles 1 to 3.

## Device addresses

- Engine methods hold a device address shifted right by 8, which `MAP_BUFFER` hands out 32 bits wide. A buffer below 4 GiB is named by its own address; one above (a 64-bit title's heap and alias regions are at 64 GiB and up) gets room in `DEVICE_WINDOW`, a 64-bit process's unused low heap region, and `DeviceSpace` turns method addresses back into CPU addresses. Relocations patch device addresses too. `UNMAP_BUFFER` frees the room.
- Decoded surfaces are keyed by the CPU address of their luma plane, not the method word naming it. Just Dance 2022 unmaps its buffers after each submit and maps them again for the next, so the room the window hands out is reused: its nvdec output is `0x302900` on every frame, a different buffer each time, and the key frame a later frame references is named by another word.

## nvdec

An `EXECUTE` (method 0xC0) with codec 9 in method 0x80 decodes one frame. The driver supplies:

- The picture setup (`SET_DRV_PIC_SETUP_OFFSET`, 0x101). Offsets: `+0x30` tile data size, `+0x48/0x50/0x58/0x60` last, golden, altref and current frame size (`u16` width, height, luma pitch, chroma pitch), `+0x68` flags, `+0x6c` reference sign biases, `+0x70..0x7f` loop filter level and sharpness, quantizer, tx mode, interpolation filter, reference mode, compound references, tile log2s, `+0x80` segmentation, `+0xe4` loop filter deltas. Flags: bit 0 key frame, bit 1 last frame was a key frame, bit 2 frame-parallel, bit 3 error-resilient, bit 4 last frame was shown, bit 5 intra-only.
- The tile data (`SET_IN_BUF_BASE_OFFSET`, 0x102): the tiles only, each but the last behind a big-endian size. The compressed header is not there; the driver applies its probability updates itself.
- The probability table (`SET_PROB_TAB_BUF_OFFSET`, 0x170), already holding the frame's final probabilities. Offsets are in `vp9/nvdec.rs`: the segmentation tree and prediction probabilities are ten bytes at +903 (a stream without segmentation leaves them zero), the intra mode tables keep their ninth probability apart, three-probability rows are padded to four bytes, coefficients are `[tx][plane][ref][band][6 contexts][4]` from `+0x5a0`.
- Luma surface addresses in slots 0 to 2 (last, golden, altref) and 3 (the frame written), at `SET_PICTURE_LUMA_OFFSET + i` (0x10c).

The decoded frame is kept by its slot 3 luma plane in `Video::surfaces`; nothing is written to the surface itself.

## VP9 on the browser's decoder

nvdec gets the tiles alone, so `vp9::rebuild::Rebuilder` writes each frame's uncompressed and compressed headers back from the picture setup and the probability table, and the browser decodes the result (`vp09.00.61.08`, no hardware preference: libvpx in Chrome where there is no hardware decoder). A channel picks the browser at its first key frame if `VideoDecoder.isConfigSupported` said yes at startup, and the software decoder otherwise.

- The rebuilt stream never saves a probability context (`refresh_frame_context` 0), so the browser's saved contexts stay at their defaults and each compressed header codes the table as updates from them. No update reaches the chroma mode probabilities, which only backward adaptation moves, or an even motion vector probability (updates are seven bits, `v << 1 | 1`). A stream that adapts its probabilities makes both, and such a frame is refused.
- Loop filter deltas and segmentation features are sent in full every frame, so no frame depends on what an earlier rebuilt one left behind.
- The browser has eight reference slots where nvdec names surfaces. The rebuilder keeps the surface in each slot; a frame refreshes the slots still holding the surface it overwrites, or else the least recently used slot none of its references is in.
- Every frame is shown, hidden ones included, so its picture comes back for its surface. That changes one thing in the decoder's state: it predicts from the previous frame's motion vectors only when that frame was shown. When the setup says the previous frame was hidden and the browser would predict from it, a filler goes first in a superframe: a hidden inter frame whose all-zero tile data decodes to DC blocks with no coefficients, refreshing no slot.
- An intra-only frame can only be hidden. It goes hidden, with a `show_existing_frame` of its slot behind it in the same superframe.
- A frame the rebuilder refuses, or a picture the browser hands back as `None`, ends the stream's time in the browser. Frames are dropped until the next key frame, which the software decoder takes after `Channel::reset`, and keeps from then on.
- `cargo run --release -p switch-core --example vp9_rebuild_check -- <dir> [--out <dir>]` hands every frame of each `x.ivf` to the rebuilder as nvdec would get it and compares each rebuilt picture, hidden ones included, with the original's. `--out` writes `x.rebuilt.ivf` and the `x.rebuilt.yuv` it must decode to; `vpxdec --i420` and `node tools/vp9_webcodecs_check.mjs <dir>` (Chromium's WebCodecs) decode them to the same bytes. The clips, from vpxenc, covered altref frames, tiles, cyclic-refresh and variance segmentation, compound prediction, error-resilient and lossless streams and odd sizes. One without frame-parallel mode was refused at its first frame whose chroma mode probabilities had adapted.
- Not checked: intra-only frames against libvpx (vpxenc writes none; the unit tests decode them with `vp9/`), and a hardware VP9 decoder (Playwright's Chromium decodes in software).

## H.264

The driver keeps a stream's parameter sets and gives nvdec only slice data: Annex B slices with their start codes, then two end-of-stream units, `stream_len` covering both, and the picture setup (`nvdec_h264_pic_s` in NVIDIA's `nvdec_drv.h`, whose offsets match Mario Kart 8 Deluxe's setups). The decoded picture's surface is picture index `CurrPicIdx`'s luma address (method `0x10c + i`).

- `h264::parameter_sets` writes an SPS and PPS back from the setup: High profile at level 5.1, the PPS numbered as the first slice asks, scaling matrices in the PPS when they are not flat. What the setup does not carry is set to what any stream allows: 16 reference frames, no cropping. Picture order count type 1 keeps its cycle in the SPS alone and is refused.
- The SPS's VUI asks for no reordering (`max_num_reorder_frames` 0). Chrome's hardware decoder then hands every picture back as soon as it is decoded, in decode order and with the same bytes; nvdec writes each picture to its surface at once, so the order does not matter. Without it the decoder held up to four pictures for display order.
- The browser's decoder (`WebDecoder`, WebCodecs on the system worker, which runs every SVC) is asynchronous, so a picture lands between slices. A VIC submit whose slot 0 surface is still decoding is asked again (`Submitted::Waiting`, the ioctl retried 1 ms later), before it reserves any syncpoint, and composes without it after 1000 asks.
- `~/Perso/nvdec-h264` (outside the repository) is a homebrew that drives nvdec and VIC as a title's driver does over an x264 clip with CABAC, 8x8 transforms and B-frames, and prints a hash of each composed frame's planes. In Chrome all 24 match ffmpeg's decode.
- Chrome's software decoder (ffmpeg, where there is no hardware H.264) takes no reordering literally and drops pictures that arrive out of display order: half of the test clip's.

## VIC

An `EXECUTE` composes slot 0 into the output surface. The configuration struct (`SET_CONFIG_STRUCT_OFFSET`, 0x1c2) holds the output surface at `+0x20`: pixel format in bits 0 to 6 (0x44 is NV12), block kind in bits 11 to 14, then width and height minus one in 14-bit fields. VIC finds the frame decoded into the slot's luma address and writes it, scaled bilinearly when the sizes differ, as pitch-linear NV12 to the output luma and chroma offsets (0x1c8, 0x1c9), pitches rounded up to 64.

## Not done

- Backward adaptation: the game's library did not adapt from the counter buffer for the stream tested (a pattern written into it changed nothing), so the counters are never written. A stream that adapts needs them, in a layout not known yet.
- The decoded surfaces are never written to guest memory, so a title that reads nvdec's output without VIC sees nothing.
- VIC pitch alignment (64) is unconfirmed for widths that are not a multiple of 64.
