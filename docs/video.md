# Video decoding

Titles decode video on the console's nvdec engine and hand the frames to VIC. `gpu/multimedia.rs` models both; `vp9/` is the VP9 decoder behind nvdec. Only VP9 (nvdec codec 9, profile 0, 8-bit 4:2:0) is decoded; other codecs still only retire their syncpoints.

## The decoder

`vp9/` is bit-exact with libvpx v1.17.0 and is checked against it, not against ffmpeg alone:

- `tools/extract_vp9_tables.py <libvpx checkout> crates/switch-core/src/vp9/tables.rs` writes the constant tables from the C initializers; `tools/extract_vp9_transforms.py` translates the 1-D inverse transforms. Both are pure Python; neither is needed to build.
- Tests that need external files run only when their variable is set, and pass otherwise:
  - `VP9_CLIPS=<dir>`: every `x.ivf` beside an `x.yuv` that ffmpeg wrote (`ffmpeg -i x.ivf -f rawvideo -pix_fmt yuv420p x.yuv`). The clips covered tiles, hidden frames in superframes, segmentation, error-resilient and frame-parallel streams, odd sizes, lossless and compound prediction.
  - `VP9_IDCT_VECTORS` and `VP9_INTRA_VECTORS`: output of a small C harness linked to `libvpx.a` (random blocks through `vp9_iht*_add_c`, `vpx_idct32x32_1024_add_c`, `vpx_iwht4x4_16_add_c` and `vpx_*_predictor_*_c`).
- `cargo run --release -p switch-core --example vp9_bench -- <clip.ivf>` times the decoder natively.
- Unsupported: scaled reference frames, profiles 1 to 3, a frame whose size changes mid-stream.

## nvdec

An `EXECUTE` (method 0xC0) with codec 9 in method 0x80 decodes one frame. The driver supplies:

- The picture setup (`SET_DRV_PIC_SETUP_OFFSET`, 0x101). Offsets: `+0x30` tile data size, `+0x48/0x50/0x58/0x60` last, golden, altref and current frame size (`u16` width, height, luma pitch, chroma pitch), `+0x68` flags, `+0x6c` reference sign biases, `+0x70..0x7f` loop filter level and sharpness, quantizer, tx mode, interpolation filter, reference mode, compound references, tile log2s, `+0x80` segmentation, `+0xe4` loop filter deltas. Flags: bit 0 key frame, bit 1 last frame was a key frame, bit 4 last frame was shown.
- The tile data (`SET_IN_BUF_BASE_OFFSET`, 0x102): the tiles only, each but the last behind a big-endian size. The compressed header is not there; the driver applies its probability updates itself.
- The probability table (`SET_PROB_TAB_BUF_OFFSET`, 0x170), already holding the frame's final probabilities. Offsets are in `vp9/nvdec.rs`: the segmentation tree and prediction probabilities are ten bytes at +903 (a stream without segmentation leaves them zero), the intra mode tables keep their ninth probability apart, three-probability rows are padded to four bytes, coefficients are `[tx][plane][ref][band][6 contexts][4]` from `+0x5a0`.
- Luma surface addresses in slots 0 to 2 (last, golden, altref) and 3 (the frame written), at `SET_PICTURE_LUMA_OFFSET + i` (0x10c).

The decoded frame is kept by its slot 3 luma address in `Video::surfaces`; nothing is written to the surface itself.

## VIC

An `EXECUTE` composes slot 0 into the output surface. The configuration struct (`SET_CONFIG_STRUCT_OFFSET`, 0x1c2) holds the output surface at `+0x20`: pixel format in bits 0 to 6 (0x44 is NV12), block kind in bits 11 to 14, then width and height minus one in 14 bit fields. VIC finds the frame decoded into the slot's luma address and writes it, scaled bilinearly when the sizes differ, as pitch-linear NV12 to the output luma and chroma offsets (0x1c8, 0x1c9), pitches rounded up to 64.

## Not done

- Backward adaptation: the game's library did not adapt from the counter buffer for the stream tested (a pattern written into it changed nothing), so the counters are never written. A stream that adapts needs them, in a layout not known yet.
- The decoded surfaces are never written to guest memory, so a title that reads nvdec's output without VIC sees nothing.
- VIC pitch alignment (64) is unconfirmed for widths that are not a multiple of 64.
