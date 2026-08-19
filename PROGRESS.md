# switch-wasm — boot status

Goal: get the bundled demo and real homebrew (the Homebrew Menu NRO,
`hbmenu.nro`) to actually **run** on the interpreter, and put what it renders
on the canvas.

## Current state: hbmenu's menu renders and takes input

hbmenu now draws its **whole UI** — title and version, the clock, the entry's
name/author/version, the footer's paths and button hints — and **responds to a
controller**: pressing + exits the menu, which is how the input path was
verified end to end.

| Symptom | Root cause | Fix |
|---|---|---|
| No text anywhere, only the bitmap logo | `pl:u` reported the shared font set as loaded but **empty**, and homebrew has no font of its own — it feeds pl's shared memory straight to FreeType | `Cpu::set_shared_font` + a real `GetSharedFontInOrderOfPriority`; the frontend fetches `web/assets/font.ttf` (built by `tools/make_font.py`) |
| Controller did nothing | The `HidSharedMemory` writer used invented offsets (`npad` at 0x3D7C0, lifo at +0x20) and only filled one slot; the frontend also used the old `KEY_*` bit order | Offsets taken from libnx's `hid.h` (npad 0x9A00, 0x5000 stride, full_key_lifo +0x28, handheld_lifo +0x378), both player 1 and handheld published, Horizon's button order, stick pseudo-buttons derived |
| Faults with a clobbered link register once input woke a parked thread | `virtmemFindStack` found no room in the reported stack region and returned NULL, and the no-op `svcMapMemory` let **every thread's stack mirror land at address 0** — two threads shared one stack | A stack region clear of the main stack, and `svcMapMemory`/`svcUnmapMemory` that really map, copy and free |

### Frame rate: ~0.7 fps, and where it goes

hbmenu is correct but slow. The measurements (`examples/hotspots.rs`,
`tools/wasm_bench.mjs`):

| | |
|---|---|
| One steady hbmenu frame | **~30M emulated instructions** |
| …of which hbmenu's own software gradient fill | 72% (~20M, about 20 instructions per pixel over 1280x720) |
| …FreeType text rasterisation | ~10% |
| …the emulator's GPU/display work | ~10% |
| Interpreter throughput, host | 28M instructions/s (was 18M) |
| Interpreter throughput, wasm in V8 | 21M instructions/s (was 17M) |
| Resulting frame rate in the browser | **0.72 fps** (was 0.57) |

The 26% that came out of it: routing by the A64 top-level encoding group before
running a group's decoder (worth 55% natively, nothing in wasm), inlining
`Memory`'s in-page fast paths while pushing the straddling and unmapped cases out
of line (worth ~15% in wasm), one translation per pixel instead of per byte in the
GPU's pixel accessors, and `Cpu::run` no longer paying a call per instruction.

The floor is now ~9ns per instruction natively and ~20ns in wasm, nearly all of it
fetch plus dispatch, so **no further reordering will make this smooth**: a 30M
instruction frame is a second of wall clock in the browser. The next real step is a
decoded-block cache — decode each basic block once into a compact form and execute
from that, which is where an interpreter of this shape normally finds its 2-4x.
Beyond that only generating code (a wasm JIT) reaches real time, and hbmenu itself
would still be asking for 30M instructions a frame.

### Known interpreter bug: hinted TrueType collapses horizontally

With a font that carries hinting programs (`fpgm`/`prep`/`cvt`), glyphs render
with correct heights and advances but each bitmap is 1-3 px wide, as if the
outline's x coordinates collapsed — untouched points look like they never get
interpolated. The same subset with `--no-hinting` renders perfectly, so this is
the emulator mis-executing something the TrueType bytecode interpreter relies
on, not a font problem.

Reproduce with two subsets of the same font (one hinted, one not) through
`cargo run -p switch-core --example screenshot -- hbmenu.nro out.ppm 1 <font>`.
The shipped font has no hinting, so this is invisible in normal use — but it is
a real correctness gap and hinting is also 8x slower to emulate.

## How hbmenu's rendering got here

`hbmenu.nro` boots, initialises the display and **presents 1280x720 frames that
reach the HTML canvas**. The whole path is real: nvdrv ioctls → nvmap → the
graphics MMU → the display buffer queue → block-linear de-swizzle → RGBA8888 →
`putImageData`.

`hbmenu.nro` **draws its menu**: the theme's background and swoosh, the entry
tile, the icon and every string all appear, composited through the real path —
hbmenu draws with the CPU into a linear buffer, deko3d's recorded command list
blits it into the tiled swapchain image with the copy engine, and the binder
presents it. The icon's JPEG decode is **pixel-exact** against a reference
decode (mean error 0, max 0 over the 256x256 image).

Two bugs stood between "presents blank frames" and this:

| Symptom | Root cause | Fix |
|---|---|---|
| `GpuStats { submissions: 0 }` — no pushbuffer ever ran, so `dkFenceWait` spun forever waiting for syncpoint 8 to reach 1 | nvdrv's `KICKOFF_PB` request carries buffer descriptors, so its CMIF header sits at 0x40; the header scan stopped at 0x40 and reported no command id, and the submit was answered as an unknown command with a generic success | `ipc_cmif_header_offset` walks the descriptors (and scans the whole message buffer as a fallback) |
| `pfifo: method 0xb on subchannel 6 before any class was bound` | deko3d writes gpfifo methods (syncpoint increment, cache flush) to subchannel 6 without a `SetObject`, because nvhost binds the channel's own class at creation | Pre-bind `SUBCHANNEL_GPFIFO` to `CLASS_GPFIFO` |

**hbmenu never needed the shader core**: its command list is one
`dkCmdBufCopyBufferToImage` plus a fence signal, and its assets are raw RGBA
bitmaps.

Getting there needed guest threads (cooperative, with real mutex/condvar
handoff), the `blr x30` fix, correct TRN/ZIP/UZP semantics, the AdvSIMD
by-element and three-different multiply groups, the scalar shift/misc/FP
forms, and the `NextLoadPath` environment entry `launchInit()` requires.
`tools/difftest.py` now checks the SIMD decode against qemu-aarch64 directly.

## GPU: a GM20B model, not a stub

`crates/switch-core/src/gpu` implements the Tegra X1 GPU the way the CPU
implements ARM64 — see AGENTS.md for the module map. Register numbers are
taken from deko3d's generated Maxwell class headers and the driver ABI from
libnx's `nvidia/ioctl`, so real command streams decode as-is.

Working: the nvdrv device/ioctl layer, nvmap, the GMMU, host1x syncpoints,
channels and the GPFIFO/pushbuffer command processor, the MME macro engine,
`ClearBuffers`, report semaphores, the copy engine (including block-linear
conversion and component remap), the 2D blitter, inline-to-memory uploads,
and scan-out.

Not implemented: the shader core. Draw calls are decoded and recorded but not
rasterized, and compute dispatches record their QMD without running warps.
Homebrew that renders through deko3d's 3D pipeline (or the EGL/GLAD stack, as
NX-Shell does) will not produce an image until that exists.

## CPU bugs found by running real code

| Symptom | Root cause | Fix |
|---|---|---|
| `aligned_alloc` returned NULL for every allocation, so deko3d/libnx graphics init failed | Register 31 in the **shifted-register** ADD/SUB form is XZR, not SP. `neg x1, x0` is `sub x1, xzr, x0`, and reading SP produced a garbage rounded size. | `add_sub` takes the form explicitly: SP for immediate/extended, XZR for shifted. |
| A vectorised table-fill loop never terminated | SIMD&FP load/store mode `0b00` only handled the unscaled STUR/LDUR form; bits[11:10] also select post-index and pre-index, whose base write-back was missing. | Decode the index field and write the base back. |
| `unimplemented instruction 0x6f3f077b` | The AdvSIMD shift-by-immediate group (SSHR/USHR/SSRA/SRSHR/SRI/SHL/SLI/SSHLL) was absent. | Implemented; the narrowing forms were already handled separately. |
| `unimplemented SIMD three-same op=0b10011` | MUL/MLA/MLS, SMAX/SMIN, SABD/SABA and CMGT/CMHI were missing from the three-same group. | Added, with a destination-reading variant for the accumulating forms. |
| `unimplemented system instruction 0xd50b7e28` | Only `DC ZVA` was handled; libnx flushes the data cache around every GPU buffer. | The remaining `DC`/`IC` maintenance ops retire as no-ops — memory here is always coherent. |

## CPU/IPC bugs found by getting past PhysicsFS

| Symptom | Root cause | Fix |
|---|---|---|
| `PHYSFS_init() failed: no error` | `ld1 {v1.16b, v2.16b}, [x2], #32` never wrote its base back (writeback was keyed off `Rm != 31`, but the immediate post-index form *is* `Rm == 31`), so newlib's `strrchr` returned a pointer 32 bytes below `argv[0]`; physfs then asked `malloc` for a negative length, and `PHYSFS_deinit` cleared the error code on the way out | Rewrote both AdvSIMD structure load/store groups against the ARM pseudocode: bit 23 selects writeback, `Rm` chooses immediate vs register increment, the interleaved LD2/LD3/LD4 forms work, `LD1R` replicates, and the single-lane index is decoded from `Q:S:size` |
| `assetsInit() failed: 2345-0010` (`romfsMountSelf` → `LibnxError_IoError`) | `fsFileRead`'s payload was read at `data_area + 0x10`, but libnx converts `fsp-srv` to a domain, which puts a `CmifDomainInHeader` first — so every read asked for 0 bytes at offset 0 | `Cpu::ipc_request_data` finds the payload after the "SFCI" header wherever it is |
| Same error, next layer: `PHYSFS_mount("romfs:/assets.zip")` reported "not found" | BSL/BIT/BIF took their mask from the wrong register, so newlib's vectorised `strchr` missed the `:` in the path; `FindDevice` then fell back to the default device and looked the file up on the SD card | Mask is Vd for BSL, Vm for BIT/BIF |
| NX-Shell: `blr` to 0, then `unimplemented instruction` on `fmov s0, s15`, then `fmadd d0, d31, d26, d0` | The scalar-FP 1-source and 3-source groups were unreachable: bits[15:10] were matched as a unit although the opcode's low bit lands in bit 15, and the 3-source test sat inside a branch that had already required a different top byte | Both groups decoded properly, FMOV as a bit-exact copy, FMADD/FMSUB/FNMADD/FNMSUB fused (`mul_add`) |
| NX-Shell: a cache-flush loop running 16x too long | `CTR_EL0` read as 0, so `4 << DminLine` gave a 4-byte stride | Report the Cortex-A57 `0x8444C004` |
| NX-Shell: `unimplemented SIMD three-same op=0b11011` (`scvtf v28.4s, v31.4s`), then `fdiv v28.4s, …` | The two-register misc and FP three-same groups both fell into the integer three-same decode | Implemented both: REV/CLS/CLZ/CNT/NOT/RBIT/ABS/NEG, the compares against zero, XTN/SQXTN/UQXTN/SQXTUN/SHLL, SADDLP/UADALP, FCVTL/FCVTN, the FRINTx and FCVTxS/U families, SCVTF/UCVTF, FABS/FNEG/FSQRT, and FADD/FSUB/FMUL/FDIV/FMLA/FMLS/FMAX/FMIN(NM)/FABD/FCMxx/FACGx/FADDP/FMAXP/FMINP |

## NX-Shell: no CPU faults left

NX-Shell now runs from boot to a **clean exit** (`ExitProcess`, ~16.3M steps).
Each fault along the way was an emulator bug, mostly whole decode groups that
were unreachable because a guard tested a field that included a fixed bit:

| Symptom | Root cause | Fix |
|---|---|---|
| `ldr s28, [x7]` with `x7 == -1` | `ucvtf d0, x1` decoded as FCVTMU (the int↔float class's `rmode`/`opcode` were read as one 6-bit field including the fixed bit21) and wrote **x0**, clobbering a live pointer | Decode `rmode` (bits[20:19]) and `opcode` (bits[18:16]) separately; every rounding mode now maps to the right instruction |
| `unimplemented SIMD three-same op=0b1000` | `ext v31.16b, …, #8` — the vector-extract group was missing, and the permute guard ignored bit29 so it executed EXT as UZP1 | EXT implemented; permute requires bit29 == 0 |
| `unimplemented instruction 0x7f6007fe` | `ushr d30, d31, #32` — only the *vector* shift-by-immediate encodings were decoded | The scalar forms (bit28 set) share the helper with one 64-bit lane |
| `unimplemented instruction 0x1e3ecffe` | `fcsel`/`fccmp` were guarded on bit21 being **clear**; they have it set, so both were dead code — and that branch was intercepting the FP↔fixed-point conversions | Conditionals moved to bit21 == 1 keyed on bits[11:10]; the fixed-point conversions implemented |
| `unimplemented instruction 0x7e21d9ad` | `ucvtf s13, s13` — the *scalar* two-register-misc group | Shares the vector implementation with a one-lane count |
| 190k identical `pl:u` calls | `GetLoadState` fell through to the generic reply, which answers command 1 with the applet's `ReceiveMessage` value (15), so the shared-font poll never saw "loaded" | The applet guesses only apply to applet services; `pl:u` has its own stub |
| `unimplemented instruction 0x0` at `pc=0` | Not a crash: libnx's `__nx_exit` branches to the loader return address, which `envSetup` takes from `__libnx_init`'s third argument. The constructor pass zeroes the registers, so it was 0 | Pass the exit trampoline in x2 when resuming the crt0 |

What stops it now is content, not code: it wants the **shared system fonts**
from `pl:u`, and there are none here, so it takes its own exit path. Its
renderer needs the shader core regardless.

## sdl-hello: libtransistor's stricter IPC

`sdl-hello.nro` uses libtransistor, which validates replies libnx ignores. Three
fixes took it from "Failed to open connection to fsp-srv: 7e0dd" to **SDL
initialised, window created, window surface obtained**:

| Symptom | Root cause | Fix |
|---|---|---|
| `Failed to open connection to fsp-srv: 7e0dd` | Replies carried `type = 0x40`; libtransistor requires 0 or 4 | Replies write `type = 0`, which is what a real server sends |
| `Failed to mount sdcard on fsp-srv: 7ecdd` (move-handle count mismatch) | `OpenSdCardFileSystem` answered with a domain out-object, but libtransistor never converts to a domain and expects a session handle | `Cpu::reply_with_interface` answers with an out-object for a domain request and a move handle otherwise |
| `SDL init failed:` right after `nvdrv` Initialize | `Initialize` returned no raw data; it really returns a `u32 error`, and libtransistor checks the reply's size | Return the error word |

It still presents no frame: its `vi`/binder requests come in libtransistor's
own (non-CMIF) marshalling, which the command-id scan misreads (four requests
decode as command `0x10100000`), and the GPU path needs the shader core anyway.

## Integer-decode bugs behind the JPEG corruption

hbmenu's icon decoded with grey luma and magenta chroma. The DC coefficients came
out `+1023` off: libjpeg-turbo's `HUFF_EXTEND` sign-extends the DC difference
branchlessly with `(x - (1 << (s-1))) >> 31`, and our 32-bit arithmetic shift was
masking the operand to 32 bits and then shifting it as a *positive* `i64`, so the
sign-extension mask came out 0. `tools/difftest.py --scalar` (new) then compared
71 integer instructions against qemu-aarch64 and found six more:

| Instruction | Was | Should be |
|---|---|---|
| `asr w, w, w` | shifted the masked value as positive | sign-extend from bit 31 |
| `asr w0, w0, #31`, `sbfx`, `sxth` | left the sign in bits 63:32 | a W write zeroes the top half |
| `extr` | took Rm as the high half | Rn is the high half of `Rn:Rm` |
| `adcs` / `ngc` | bit29 read as "subtract" | bit30 = subtract, bit29 = set flags |
| `sdiv w` | divided unsigned | sign-extend from the operand width |
| `smaddl` / `umaddl` | multiplied the full 64-bit registers | the low 32 bits, sign/zero-extended |
| `cls` | counted the sign bit too | count the bits *after* the sign |

## Services

- **nvdrv** is the real `INvDrvServices` interface (Open/Ioctl/Ioctl2/Ioctl3/
  Close/Initialize/QueryEvent). `QueryPointerBufferSize` reports 0 so libnx
  marshals every auto-select buffer as a map-alias range, and
  `CloneCurrentObject` is honoured because libnx sends `SubmitGpfifo` down the
  cloned session.
- **vi** hands out an `IHOSBinderDriver` whose `TransactParcel` runs a real
  `IGraphicBufferProducer`: `SET_PREALLOCATED_BUFFER` decodes the
  `NvGraphicBuffer`, `DEQUEUE_BUFFER` hands back a slot, and `QUEUE_BUFFER`
  scans the image out.
- **fsp-srv** is backed by `vfs::Vfs`, a path-addressed SD card. Paths arrive
  in HIPC static buffers, so a missing path reports `FsError_PathNotFound`
  instead of pretending to succeed — which is what stopped hbmenu recursing
  forever through directories that did not exist.
- **set** reports a real language-code table so `setMakeLanguage` resolves
  `en-US`; hbmenu's `textInit()` depends on it.

## Repro / verification

- Host: `cargo run -p switch-core --release --example screenshot -- \
  test-nros/hbmenu.nro out.ppm 3` writes the third presented frame.
- `cargo run -p switch-core --release --example trace -- <nro>` profiles the
  hottest PCs, or breaks on given PCs and dumps registers.
- `TRACE_NV=1` traces nvdrv IPC (with guest backtraces), `TRACE_GPU=1` traces
  device opens, ioctls and engine methods, `TRACE_IPC=1` traces all services.
- Browser: `make serve`, load `hbmenu.nro` with the "Horizon (stubbed)" ABI.
- Regression suite: `make test` — 199 tests.

## Next

1. **hbmenu's entry label** is drawn as a blank white box (the tile's text area),
   so its FreeType text path is worth a look next.
2. **Shader core**: a Maxwell SASS interpreter plus a software rasterizer, so
   `VertexBegin`/`DrawArrays` produce pixels. This is what deko3d-rendering and
   EGL/OpenGL homebrew need.
3. **NX-Shell** now dies on that `x7 == -1` pointer; it renders through
   EGL/GLAD, so it needs (2) regardless.

## Historical notes

## Input / controller support

Gamepad and keyboard input is wired end-to-end:

- `web/main.js` polls the Browser Gamepad API (plus a keyboard fallback) every
  16 ms and calls a new `switch_set_input(handle, buttons, lx, ly, rx, ry)`
  export. `buttons` is the `HidNpadButton` bitfield; sticks are -32768..32767.
- The core mirrors the state into two places: the memory-mapped input register
  (`INPUT_ADDR`, a simple host→guest poll), and the libnx `HidSharedMemory`
  player-1 layout so real homebrew using `padInitialize`/`padUpdate` sees it.
- `svcMapSharedMemory` (hid's shared memory) now actually backs the region
  with real zeroed memory and records where it is, so `padUpdate` reads live
  state instead of faulting.

## NSP / NCA frontend fixes

- **PFS0 offset rebasing**: some repacked NSPs (e.g. ROMSLAB) store PFS0 file
  offsets relative to the end of the string table rather than the file start,
  so extraction returned the wrong bytes ("bad magic"). The parser now
  detects an entry pointing inside the header and rebases by the payload
  start.
- **CDN NCA headers are encrypted**: `Nca::parse` fails with "bad magic"
  because the NCA3 magic at 0x200 is scrambled until the header is decrypted
  with the title key. The frontend now says "NCA header is encrypted (CDN) —
  needs the title key from the .tik" instead of a bare "bad magic".
- **Wasm memory leaks**: staging buffers for NSP/NRO loads and NCA extraction
  were never freed, so repeated loads accumulated wasm memory until large
  allocations overflowed it (the `toWasm` RangeError). They are now freed
  after use, and `toWasm` re-fetches the linear-memory buffer (which is
  detached on growth).

## NCA body decryption and launching — verified against a real title

Previously an NCA could only be *inspected* (header metadata) — there was no
path from "this is a Program NCA" to "boot its executable". That gap is
closed and **proven against a real commercial title** ("A Short Hike"),
decrypted end-to-end with its own SHA-256 hash verified against Nintendo's own
stored master hash — not a synthetic test. Given `prod.keys` (and either a
`title.keys` entry or, more commonly, the ticket a scene NSP release bundles
right next to the content), the emulator decrypts a Program NCA's ExeFS
section, extracts `main` and boots it, via a "Launch" button on Program NCAs
in the NSP inspector — for an NCA embedded in an NSP
(`switch_load_nca_from_nsp`) or a standalone `.nca` dropped directly
(`switch_load_nca`) — both funnel into `Cpu::boot_program`. A CLI equivalent
(`cargo run -p switch-core --example boot_nsp -- <nsp> <prod.keys>
[title.keys]`) exists for debugging without a browser.

The pieces:

- **`crypto.rs`**: AES-128-CTR (cross-checked against `openssl enc
  -aes-128-ctr`) and SHA-256 (cross-checked against `hashlib`), alongside the
  existing AES-ECB/XTS used for headers.
- **`keys.rs`**: `key_area_key_<application|ocean|system>_XX` and
  `titlekek_XX` parsed directly from `prod.keys` (by generation suffix), like
  `header_key` — stored as dumped, not derived, since deriving them needs
  Nintendo's secret seed constants this project doesn't embed.
- **`ticket.rs`**: parses an ES ticket (`.tik`) and decrypts its title key
  (Common crypto only — Personalized needs a console's ETicket RSA key,
  out of scope). `find_and_decrypt_title_key` locates `<rights_id-hex>.tik`
  among an NSP's files, so a title-key-crypto title works out of the box
  without a separate personal `title.keys` dump, the same way real scene
  releases are set up to be opened.
- **`nca.rs`**: decrypts the 4 per-section FS headers, unlocks the AES-CTR
  section key (ticket/`title.keys` title key for rights-id titles, key-area
  slot 2 otherwise), decrypts a section body, and verifies it against the FS
  header's own master hash before trusting it — AES-CTR with the wrong key
  still "decrypts" into plausible-looking garbage, so this hash check is the
  only way to know the keys were actually right (and is how the bugs below
  were actually found and fixed, by treating a real file as an oracle rather
  than trusting the public spec by itself).
- **`lz4.rs`** / **`nso.rs`**: raw-block LZ4 decompression and the NSO0
  loader (three segments + BSS, like NRO, but per-segment compressed and
  with no external relocator — the linked crt0 relocates itself).
- **`Cpu::boot_program`**: places the NSO and jumps to its entry — no
  homebrew ABI env block to set up for a retail program.

Bugs a real file caught that no synthetic test could (each confirmed via the
master-hash oracle, which flips from "mismatch" to "match" the instant the
fix lands):

| Symptom | Root cause | Fix |
|---|---|---|
| A real Program NCA's section 0 decoded at a multi-terabyte offset, 1 byte long | The section table entry is `u32 start_offset; u32 end_offset` in 0x200-byte media units, not a `u64 offset; u64 size` byte pair | `SectionHeader` parsing now reads the u32 pair and scales by 0x200 |
| AES-CTR decrypted to garbage (hash mismatch) even with the *correct* title key | The AES-CTR counter's low 8 bytes reset to 0 at each section's start; Nintendo's own `nca_calculate_section_ctr` runs the counter across the section's *absolute* position in the file instead | `FsHeader::initial_counter` takes the section's `media_offset` and seeds the counter with `media_offset >> 4` |
| Resolved title key decrypted to garbage even with the right ticket bytes | The ticket's `common_key_id` needs the same "stored value is one more than the real generation, except 0" adjustment the NCA header's key-area generation already gets — `common_key_id` 0x0b needs `titlekek_0a`, not `titlekek_0b` | `Ticket::master_key_revision` applies `saturating_sub(1)` |
| Inspector showed "Crypto: cleartext" and "File size: 0 B" for an obviously-encrypted, 304 MiB real NCA | `crypto_type` (0x21C) is 0 for title-key-crypto titles by design (it only describes key-*area* crypto) — not a bug, just an incomplete check; separately, `file_size` was read from 0x340, which isn't where content size actually lives | `is_encrypted()` now also checks `has_rights_id()`; `file_size`/`program_id` moved to their real offsets (0x208, 0x210 — found by searching the decrypted header for the byte pattern of the file's known real size) |

**Real entry point found and fixed**: the ExeFS extracts a completely real,
sensible layout (`main`, `main.npdm`, `rtld`, `sdk`, `subsdk0`), and `main`
now actually executes. The first fault was at the very first instruction —
confirmed (via the raw LZ4 literal bytes, not just decompression output) to
be genuine file content: `.text` starts with `00 00 00 00 / 00 00 00 08 /
"MOD0"`, a `ModulePtr`/MOD0 header (data), not a branch instruction the way
devkitA64 homebrew's crt0 starts. Disassembling forward with the project's
own disassembler (`switch_core::disasm`) found where real code actually
begins: everything up to `.text`+0x30 decodes as garbage (it's the
`ModulePtr` + `MOD0` header + alignment padding), and at exactly 0x30 a
textbook crt0 prologue appears (`sub sp, sp, #0x90` / `stp x29, x30, [...]`)
followed by the same constructor-array-iteration pattern (`blr` in a loop)
homebrew's own crt0 runs. `nso::load_nso` now reports `entry` as `.text`+
[`NSO_ENTRY_OFFSET`] (0x30) instead of `.text`+0 — real SDK-linked NSOs, not
just this title, since the ModulePtr/MOD0/padding layout is a fixed part of
the format, not something that varies per build.

Booting "A Short Hike"'s `main` with this fix now runs 61 real instructions
(up from 0) before faulting on the next missing piece — a genuinely new,
separate wall (not a decoding bug in the entry sequence, which behaves
exactly like a normal function call/return so far). Beyond that: a much
larger service surface (`acc`, `ns`, `aoc`, `bcat`, …) than any homebrew has
ever needed, and no shader core — expect real titles to run only until the
next missing piece, the same "boot as far as it goes" pattern as everything
else in this file.

## RomFS mounting — verified against a real title

`OpenDataStorageByCurrentProcess` (fsp-srv cmd 200) hands back an `IStorage`
backed by `Cpu`'s decrypted RomFS bytes (`Cpu::set_romfs`, `fs_storage_request`
in `ipc.rs`) — cmd 0 = `Read(offset, size)`, cmd 4 = `GetSize`. This is
deliberately thin: libnx's `romfsMount`/`nn::fs::MountRom` parse the RomFS
header and directory/file tables entirely in guest code against raw byte
reads, so the host only ever needs to serve byte ranges, never parse the
filesystem itself. `Nca::romfs_section_index`/`decrypt_romfs_section` find
and decrypt the section, then slice out the actual RomFS body (see below —
it isn't at byte 0), sanity-checked against RomFS's own `header_size` field
(always 0x50).

**Fixing this needed a real reference implementation, not just the public
wiki write-up.** The AES-CTR key and counter construction were both already
byte-for-byte correct (independently confirmed: building `hactool` from
source — `/Users/thelio/Perso/hactool` — and pointing it at this exact NCA
with these exact keys reproduces the identical `Section CTR` and decrypted
title key this project computes). The actual bug was architectural: an IVFC
(`HierarchicalIntegrity`) section is a multi-level hash tree — byte 0 of the
decrypted section is Level 0's (coarsest) hash table, not RomFS's own header.
The real RomFS data lives at the *last* level's `logical_offset`
(`FsHeader::romfs_data_offset`, parsed from the FS header's `ivfc_hdr_t`,
always `level_headers[5]` — hactool's own `nca.c` reads a fixed index 5
regardless of the header's `num_levels` field, which reads `7` on this real
file despite the level array only holding 6 entries; deriving the index from
it instead of hardcoding 5 reads 24 bytes into the trailing padding and
silently returns 0). Once addressed, "A Short Hike"'s ~276 MiB RomFS section
decrypts cleanly on every run — same title key, same `decrypt_section` code
path as the already-hash-verified ExeFS, just sliced at the right offset.

Along the way, `hactool` also confirmed two things about the ticket/title-key
work from the previous session were exactly right: the decrypted title key
(`95c1b034b8151c9d058126216efde161`) and the `common_key_id` → `titlekek`
generation adjustment (hactool independently reports "Master Key Revision:
0xA" for `common_key_id` `0x0b`, matching `Ticket::master_key_revision`'s
`saturating_sub(1)`).

**Byte-for-byte correction to the FS header field names**: cross-checking
against `nca_fs_header_t` also caught that this project's `fs_type`/`hash_type`
field names were swapped relative to hactool's actual struct — byte 2 is
`partition_type` (0=RomFs, 1=Pfs0), byte 3 is `fs_type` (2=Pfs0, 3=RomFs) and
doubles as what this project called `hash_type` (there's no separate hash-type
byte). The byte *positions* being read were already right, so this was a
naming fix, not a behavior change — `exefs_section_index`/`romfs_section_index`
now use the real names.

Loading a title with no RomFS section (Meta/Control-only) or one whose RomFS
decrypt fails still boots (`main` still runs, or tries to) — this is treated
as optional and non-fatal, same as a missing Horizon service. Full multi-level
IVFC hash verification (walking all 6 levels, the way `decrypt_pfs0_section`
verifies `HierarchicalSha256`) isn't implemented — only the `header_size`
sanity check — so a corruption subtler than "wrong key" wouldn't be caught.


## Multi-module loading (`rtld`+`main`+`subsdk`+`sdk`) — real entry-point bug found and fixed

A retail title's ExeFS is multiple NSO modules sharing one address space
(`rtld`, `main`, `subsdk0..9`, `sdk`), not just `main` — `main`'s own
GOT/PLT-style indirect calls are unrelocated until `rtld` (Nintendo's own
runtime linker, loaded and run *first*) processes them.
`Cpu::boot_retail_program` now loads every present module sequentially at
page-aligned addresses (`collect_modules` in `switch-wasm`, in Nintendo's
required order) and jumps to `rtld`'s entry instead of `main`'s.

**Bug found by tracing "A Short Hike"'s real `rtld` for 25 million
instructions**: it ran cleanly for a long time, then hit
`unimplemented instruction 0x00000000` deep inside its own module — at an
address that, moments earlier, held a legitimate `stp`/`add`/`cmp`/`b.hi`
zero-fill loop. The loop was overwriting its own code. Root cause:
[`NSO_ENTRY_OFFSET`] (`.text`+0x30) is only correct for modules that actually
have a `ModulePtr`/`MOD0` header at `.text`+0 (`main`/`subsdk*`/`sdk` do —
confirmed earlier by disassembling a real module's `.text`+0, which is inert
data up to +0x30 where a textbook crt0 begins). **`rtld` has no such
header** — its `.text`+0 is real code: a `b` that skips an inline
PC-relative literal used by a `bl`+literal idiom (`bl #8; .word disp;
ldr wN,[x30]; sbfx; add xN,xN,x30`) to compute its own load address, since
`rtld` must establish where it was loaded before it can do anything else,
including finding its own `MOD0`. Jumping straight past that bootstrap (as
`.text`+0x30 does) left `x0` — meant to hold `rtld`'s own base address — at
`0`, so a later `bss_end - base` computation produced a ~4 GB byte count fed
to that zero-fill loop, which walked forward through nearly the entire
32-bit address space until it reached back around into its own `.text` and
clobbered the very instructions running it. `nso::entry_offset` now checks
for the `ModulePtr`+`MOD0` signature (`reserved==0` at `.text`+0, magic
`"MOD0"` at the offset it points to) instead of assuming it's always there;
absent it, entry is `.text`+0.

With that fixed, `rtld`'s real bootstrap runs correctly (byte count for the
zero-fill is now a sane `0xe8`, not billions) — and it gets *much* further
before its next fault, at instruction 9727 instead of 25 million:
`br x16` to a null pointer, from a lazy-PLT-style resolver trampoline
(`mov x16, x0` / spill args / call resolver / reload args / `br x16`). The
program's own console output names exactly what it was trying to resolve:
`[rtld] Unresolved symbol: '_ZN2nn4init5StartEmmPFvvES2_'`
(`nn::init::Start`, normally exported by the `sdk` module).

**This turned out not to need a loader-config handoff at all — `rtld`
doesn't wait to be told what modules exist, it finds them itself.**
Disassembling the whole 6790-byte module (small enough to read start to
finish) turned up its actual discovery algorithm, right down to the exact
filter it applies: a loop that calls `svcQueryMemory` (the only `svc #0x6`
in the module) across the address space and, for each region, checks
`type == 3` (`CodeStatic`) *and* `perm == 5` (`R-X`, built as
`movz w24,#0x4f4d; movk w24,#0x3044` — literally the `"MOD0"` magic,
compared against `[region_base + mod0_offset]`) before treating it as a
module and processing its dynamic/export table. Two things broke this in
this project's own `svcQueryMemory` stub (`svc.rs`, `0x06`): it reported a
blanket RWX (`0b111`) permission on every mapped page instead of real R-X on
`.text`, so no region ever matched `perm == 5` and every other module was
invisible to the scan; and `Memory`'s read-only tracking
(`Memory::mark_readonly`, backing `.text` write-protection) held only a
*single* `(start, end)` range, silently unprotecting every earlier module's
`.text` each time a later one loaded — fine for a lone homebrew NRO, wrong
for four back-to-back NSOs. Fixed by turning `Memory`'s read-only tracking
into a list of ranges (one push per loaded module, `Memory::is_readonly`
checks all of them) and having `svcQueryMemory` consult it to report R-X
specifically on `.text` pages, RWX elsewhere — reusing the exact ranges
already tracked for write-protection rather than adding a parallel table.

Both fixes together took "A Short Hike"'s real ExeFS from faulting inside
`rtld` at instruction 51 (before any of this session's work) to running
**116 million instructions** — through `rtld`'s relocation of itself and the
other three modules, `main`'s init, `subsdk0`, and into `sdk`'s own code —
before hitting a deliberate `svcBreak` (an SDK-side abort call, not a CPU
crash) inside `sdk`. That's expected: this project has no Horizon service
support for retail games yet, so a real title is expected to run until the
first missing service or explicit abort rather than reach a menu — but it's
the first time a real retail title's actual multi-module boot sequence has
run this deep.

### Chasing the `svcBreak` itself — real Horizon `InfoType_RandomEntropy` gap found, root cause of *this* abort still open

Traced the exact `svcBreak(reason=Panic, arg=&1u32, size=4)` call site by
walking the frame-pointer chain (`Cpu::backtrace`) instead of raw
instruction history — much cheaper than reading through the resolver's own
symbol-hash loops, which dominate any short instruction window around a
call this deep in `sdk`. Two things came out of that:

- `svcGetInfo` type `11` (`RandomEntropy` — four kernel-supplied random
  words, real hardware's seed for stack canaries/ASLR cookies) wasn't
  implemented at all (fell into the stub's `_ => 0` default). Confirmed via
  the same instrumented run that `sdk` init does query it (`infoSubValue`
  `2` and `3`) shortly before the abort. Added it (`svc.rs`, `0x29`
  handler): a SplitMix64 mix keyed by the subvalue, so it's non-zero and
  varies per word — there's no real entropy source to draw on, and it only
  needs to look "not obviously broken" to whatever init-time check reads it.
- That fix changed nothing about *this* abort — same instruction count
  (116828674) to the same `svcBreak`, byte for byte. Disassembling the
  final ~40 instructions before it (`0xcf11ea0`..`0xcf12090`) explains why:
  that whole span is `sdk`'s raw per-syscall wrapper table (`svc #N; ret`,
  one thunk per number), and the specific sequence executed right before
  the break (`QueryMemory`, `GetThreadPriority`, `GetThreadId`, then
  `Break`) is `nn::diag`'s abort path gathering thread/memory context to
  attach to the crash report, not a condition being checked — by the time
  any of that runs, the decision to abort has already been made further
  back, in `sdk` code with no visible symbols and no distinguishing
  landmark (no Result-code comparison, no recognizable constant) in reach
  of frame-pointer backtracing alone.

Finding *which* precondition actually fails would need either real nnSdk
symbols to match this disassembly against, or a much deeper manual
reverse-engineering pass than the instruction-history/backtrace tools used
so far — a different kind of effort than the loader/`svc`-table bugs above,
each of which had a concrete, verifiable signature (a wrong offset, a wrong
permission code) to chase. The `RandomEntropy` fix stays in regardless: it's
a real gap (every retail title's `sdk` init reads it) independent of whether
it explains this specific abort.

`svcBreak` (`svc.rs`, `0x26`) used to just log a bare `"[svcBreak]"` marker
and halt, throwing away exactly the info that identified the cause the last
few times it fired (a resolver's own `reason`/`arg`/`size` triple, the value
`arg` points at, the call stack). It now decodes `reason` against libnx's
`BreakReason` enum, dereferences `arg` when `size` is a plain integer width,
and appends a `Cpu::backtrace` frame-pointer walk — all into `self.out`, so
it shows up in the console unconditionally instead of needing trace mode on.

### Chasing the abort with Binary Ninja — one real bug found and fixed, one still open

Picked this back up with an actual decompiler (Binary Ninja, driven both via
its MCP server and, for headless scripting where the MCP surface had no
"create function" primitive, its own bundled Python API) instead of hand-
reading disassembly. Dumped `sdk`'s loaded memory image, wrapped it in a
synthetic ELF at its real runtime base so the tool could auto-detect
architecture and load address, and worked backward from `Cpu::backtrace`'s
frame-pointer walk at the halt point — each frame a function to decompile,
each indirect call resolved by reading its GOT/vtable slot's *live* value out
of the emulator (Binary Ninja's own view is a static snapshot; only the
running `Cpu`'s memory reflects relocations and runtime writes).

That walk reached all the way back to `nn::init::Start` itself — recognizable
by its exact signature (two integers, two function-pointer calls), the same
symbol `rtld` failed to resolve at the very start of this investigation,
closing the loop. Its own internal setup calls into a chain of GOT-indirected
helpers ending in a "does the current thread's context match what's
expected" check (`(*arg1 & 0xbfffffff) == *(x8 + 0x1b0)`, `x8` read via
`TPIDRRO_EL0`) that reads both sides as `0` and treats that as a match,
walking into an unconditional abort helper.

**Real bug #1, found along the way and fixed**: `TPIDR_EL0` and `TPIDRRO_EL0`
were aliased to the same backing field (`cpu/mod.rs`/`system.rs`). Real
AArch64 makes these two separate registers — `TPIDR_EL0` freely
read/write by EL0, `TPIDRRO_EL0` fixed once by the kernel and read-only
thereafter. `nnSdk`'s own init legitimately writes `TPIDR_EL0` for its own
per-thread bookkeeping; with the two aliased, that write silently corrupted
what should have been the *stable* kernel-provided TLS pointer everything
else (IPC dispatch, the thread-context check above) reads through
`TPIDRRO_EL0`. Split into `tpidr` (kernel-fixed, unchanged semantics — still
backs IPC's TLS-relative buffer lookup and thread creation) and a new
`tpidr_rw` (the guest-writable one). Verified with a live memory probe:
before the fix, `TPIDRRO_EL0` read back `0` after `nnSdk`'s write (an
address-space-near-zero pointer registration lands at literal absolute
`0x1f8` instead of the intended `0x1fe001f8`); after, it stays at its
bootstrap value (`0x1fe00000`) for the entire run, and the per-thread struct
pointer registration lands at the correct address. Zero regressions
(199/199 lib tests).

**Real bug #2, still open**: that fix changed nothing about the actual
abort — same instruction count, same backtrace, same values, confirmed by
re-running the identical probe before and after. The struct pointer itself
(`x8`) resolves correctly either way (self-consistent write-then-read
regardless of which base address it's relative to); the specific field at
`x8 + 0x1b0` is simply never written by anything, for the entire 117M-
instruction run (confirmed with a full-run memory watch). Finding what's
*supposed* to write it means reverse-engineering another offset in
Nintendo's undocumented private per-thread structure — a real target, but
without a spec to check against like the TPIDR/TPIDRRO split had, so it's
open-ended guessing rather than a provable fix.

## deko3d / nv (unresolved)

The nv GPU path is still stubbed at the service boundary:

- `stub_nvdrv` (cpu.rs): cmd 0 → fd=1, cmd 1/2 → 0, cmd 3 → empty success
  reply. hbmenu sends only cmd 3 (Initialize) before its device init stalls.
- deko3d's `nvLibInit` (dk_device.cpp) does `nvInitialize → nvGpuInit (nvOpen
  cmd 0 for `/dev/nvhost-ctrl-gpu` etc.) → nvFenceInit → nvMapInit`; the
  device paths (`/dev/nvhost-gpu`, `/dev/nvmap`, `/dev/nvhost-as-gpu`,
  `/dev/nvhost-ctrl`) are known but the code never reaches `nvOpen`.
- To render: stub the nvdrv Open/Ioctl chain (GetCharacteristics, channel
  setup, SUBMIT_GPFIFO), capture deko3d's command buffer, and software-render
  it (clear color → solid fills → textured quads) into the swapchain
  framebuffer, then push that to the canvas.


## NCA header decryption with prod.keys / title.keys

CDN NCA headers are AES-128-XTS encrypted, so the NCA3 magic at 0x200 is
invisible until decrypted. The tool now decrypts and inspects them when the
user supplies key files:

- **`crypto.rs`**: hand-rolled AES-128 (FIPS-197 S-box, key expansion) plus
  AES-128-XTS (OpenSSL/`cryptography`-verified tweak advance), used to decrypt
  the 0x400-byte NCA header as two 0x200-byte XTS sectors with the global
  `header_key`.
- **`keys.rs`**: parses `prod.keys` / `title.keys` (`name = hex` lines),
  derives the header key from the sources when `header_key` isn't provided
  (hactool `pki.c` master-key chain), and indexes title keys by rights id.
- **`Nca::parse_with_keys`**: tries the cleartext header, then decrypts with
  the keyset; the frontend loads both key files and re-parses.
- Frontend UI: "Load prod.keys" / "Load title.keys" buttons in the NSP card;
  the status line reflects what's loaded.

Also fixed along the way:
- **NSP ownership**: `switch_load_nsp` now takes ownership of the staged
  buffer instead of copying, halving wasm memory for multi-GB NSPs (a 954 MiB
  NSP previously pushed the browser past its ~2 GB linear-memory ceiling,
  causing traps).
- **Panic resilience**: the wasm session table no longer uses `std::sync::Mutex`
  (the single-threaded wasm backend asserts on any reentrant `lock()`, and a
  panic while held leaves it locked forever under `panic=abort`); it uses a
  `SyncCell` and a panic hook that surfaces Rust panics to the frontend.
- **`switch_read_file`**: reads just a slice of an NSP file, so inspecting an
  NCA header no longer allocates the whole (hundreds-of-MB) payload.

## sdl-hello.nro — now boots and exits cleanly

`sdl-hello.nro` (vgmoose's SDL hello-world, SDL2 on deko3d) now **boots all the
way through libnx + SDL init and exits cleanly** via `ExitProcess` (svc 0x07,
exit code -14 at the deko3d/nv GPU layer). Console output ends with
`set up stdout.` — the point where SDL_Init reaches the deko3d video backend,
which we don't emulate. This is the first real NRO to fully run its init
sequence without faulting.

Bugs found while getting there (each cross-checked against QEMU / qemu-aarch64):

| Symptom | Root cause | Fix |
|---|---|---|
| `panic: attempt to shift left with overflow` in `sext_u64` | `sext_u64(v, 64)` did `1u64 << 64` | `bits >= 64` returns `v` (64-bit sign-extend is identity) |
| `CPU: unallocated logical immediate` on `mov w20, #0x80808080` (0x3201c3f4) | `decode_bit_mask` rejected valid masks: extra `imms & !levels` check; QEMU masks `imms` down to the element size | Rewrote to match QEMU `logic_imm_decode_wmask` |
| `ldrsw x8,[x9,x8,lsl#2]` loaded 0, branched into the jump table | register-offset shift used the byte count (4) instead of `log2(size)` (2) | `offset_from_reg` shifts by `sz` |
| `memset` ran forever (x3 ran past the buffer) | `mrs x5, dczid_el0` returned 0; musl/newlib strides the DC-ZVA loop by `4 << BS` | return BS=4 (64-byte block, Cortex-A57) |
| `CPU: unimplemented system instruction` on `dc zva, x3` | DC ZVA wasn't implemented | zero the 64-byte block at Xt |
| `mov v0.d[1], x9` (INS) unimplemented | copy-group guard rejected `imm5 >= 16` (bit20 is part of imm5, not a guard bit) and q=0 forms | guard is `bits[29:21]==001110000` (QEMU pattern); added INS |
| `ldr d0,[x9,#0xd0]`, `stp d8,d9`, `str d0,[x8,x10]` unimplemented | SIMD&FP immediate/register-offset size mapping was wrong: B/H/S/D are size 00/01/10/11 (the code had S=00, D=01) | corrected size map + added the register-offset FP form and S/D/Q store/load pairs |
| `SIMD MOVI cmode=0b1110` rejected on `movi d8, #0` (0x2f00e408) | imm8 is NOT contiguous: `abcdefgh` = bits 18:16 ++ 9:5 (cmode at 15:12); also only cmode 0000 was implemented | full `AdvSIMDExpandImm` (`simd_imm_const`), QEMU-verified values |
| `setup_fs` returned `-97` (`sqfs_pread` failed) right after boot | `CSEL` family was applying `csinv`/`csinc`/`csneg` invert/increment to the *selected* value instead of the *else* value | compute the else value with invert/inc and then select; `csel` unchanged |

The frontend has a "Bundled app" selector (demo / hbmenu / NX-Shell /
sdl-hello) and `prod.keys`/`title.keys` persist in `localStorage`.

## Frontend: web worker refactor

`web/worker.js` hosts the wasm module (`WebAssembly.instantiateStreaming` of
`assets/switch_wasm.wasm`) and `web/main.js` talks to it via promise-based RPC
over `postMessage`. The step budget is unlimited for normal runs
(`Number.MAX_SAFE_INTEGER`) and 5000 in trace mode, so a well-behaved NRO can
run to exit/teardown instead of being cut off mid-init.

## Next

1. Find why deko3d's slot-table base (`0x08254080`) is never populated during
   hbmenu's `graphicsInit`, so the cleanup path stops aborting with `svcBreak`
   0x1159 — and see what `graphicsInit` returns before that (the abort path is
   a failure handler, so the real error is earlier).
2. After that, complete `stub_nvdrv` so the deko3d device init fires `nvOpen`
   (cmd 0) and the ioctls succeed, letting hbmenu reach its render loop.
3. Capture deko3d's per-frame command buffer (SUBMIT_GPFIFO) and implement a
   software GM20B renderer (clear color → solid fills → textured quads),
   pushing the swapchain framebuffer to the canvas.
