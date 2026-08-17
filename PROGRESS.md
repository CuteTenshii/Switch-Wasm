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
