# switch-wasm — boot status

Goal: get real homebrew and real retail titles to **run** on the interpreter,
and put what they render on the canvas.

## Current state

- **Homebrew (NRO)**: `hbmenu.nro` boots, renders its full UI, and responds to
  a controller. `sdl-hello.nro` and `NX-Shell.nro` boot all the way through
  libnx/SDL init to a clean exit — both stop only where they need the GPU
  shader core, which doesn't exist yet.
- **GPU**: the 2D/copy-engine path is real (`crates/switch-core/src/gpu`) —
  enough for hbmenu, which never needs a shader. The 3D shader core (Maxwell
  SASS + rasterizer) is not implemented; see [GPU](#gpu-gm20b-model).
- **Retail NCA/NSP**: a real, encrypted commercial title ("A Short Hike") can
  be decrypted, its RomFS mounted, and its full multi-module boot sequence
  (`rtld` → `main` → `subsdk*` → `sdk`) run for 117M+ instructions through
  real `nnSdk` init — `nn::init::Start` → `nn::oe::Initialize` →
  `nn::oe::InitializeApplet` — and start the SDK's system worker thread. All
  of `nn::oe::Initialize` now completes; at 117.8M instructions the title is
  in its own `main`, and stops setting up its heap in
  `nn::mem::StandardAllocator::Initialize`. See
  [Retail NCA/NSP loading](#retail-ncansp-loading).

See [Next](#next) for the live open threads.

## Homebrew (NRO) boot & rendering

### hbmenu — full render + input

hbmenu draws its **whole UI** — title and version, the clock, the entry's
name/author/version, the footer's paths and button hints — and **responds to
a controller**: pressing + exits the menu, which is how the input path was
verified end to end.

| Symptom | Root cause | Fix |
|---|---|---|
| No text anywhere, only the bitmap logo | `pl:u` reported the shared font set as loaded but **empty**, and homebrew has no font of its own — it feeds pl's shared memory straight to FreeType | `Cpu::set_shared_font` + a real `GetSharedFontInOrderOfPriority`; the frontend fetches `web/assets/font.ttf` (built by `tools/make_font.py`) |
| Controller did nothing | The `HidSharedMemory` writer used invented offsets (`npad` at 0x3D7C0, lifo at +0x20) and only filled one slot; the frontend also used the old `KEY_*` bit order | Offsets taken from libnx's `hid.h` (npad 0x9A00, 0x5000 stride, full_key_lifo +0x28, handheld_lifo +0x378), both player 1 and handheld published, Horizon's button order, stick pseudo-buttons derived |
| Faults with a clobbered link register once input woke a parked thread | `virtmemFindStack` found no room in the reported stack region and returned NULL, and the no-op `svcMapMemory` let **every thread's stack mirror land at address 0** — two threads shared one stack | A stack region clear of the main stack, and `svcMapMemory`/`svcUnmapMemory` that really map, copy and free |

The rendering path is real end to end: nvdrv ioctls → nvmap → the graphics
MMU → the display buffer queue → block-linear de-swizzle → RGBA8888 →
`putImageData`. hbmenu draws with the CPU into a linear buffer, deko3d's
recorded command list blits it into the tiled swapchain image with the copy
engine, and the binder presents it. The icon's JPEG decode is **pixel-exact**
against a reference decode (mean error 0, max 0 over the 256x256 image).
hbmenu never needed the shader core: its command list is one
`dkCmdBufCopyBufferToImage` plus a fence signal, and its assets are raw RGBA
bitmaps.

Getting there needed guest threads (cooperative, with real mutex/condvar
handoff), the `blr x30` fix, correct TRN/ZIP/UZP semantics, the AdvSIMD
by-element and three-different multiply groups, the scalar shift/misc/FP
forms, and the `NextLoadPath` environment entry `launchInit()` requires.
`tools/difftest.py` now checks the SIMD decode against qemu-aarch64 directly.

**CPU/IPC bugs found along the way:**

| Symptom | Root cause | Fix |
|---|---|---|
| `aligned_alloc` returned NULL for every allocation, so deko3d/libnx graphics init failed | Register 31 in the **shifted-register** ADD/SUB form is XZR, not SP. `neg x1, x0` is `sub x1, xzr, x0`, and reading SP produced a garbage rounded size. | `add_sub` takes the form explicitly: SP for immediate/extended, XZR for shifted. |
| A vectorised table-fill loop never terminated | SIMD&FP load/store mode `0b00` only handled the unscaled STUR/LDUR form; bits[11:10] also select post-index and pre-index, whose base write-back was missing. | Decode the index field and write the base back. |
| `unimplemented instruction 0x6f3f077b` | The AdvSIMD shift-by-immediate group (SSHR/USHR/SSRA/SRSHR/SRI/SHL/SLI/SSHLL) was absent. | Implemented; the narrowing forms were already handled separately. |
| `unimplemented SIMD three-same op=0b10011` | MUL/MLA/MLS, SMAX/SMIN, SABD/SABA and CMGT/CMHI were missing from the three-same group. | Added, with a destination-reading variant for the accumulating forms. |
| `unimplemented system instruction 0xd50b7e28` | Only `DC ZVA` was handled; libnx flushes the data cache around every GPU buffer. | The remaining `DC`/`IC` maintenance ops retire as no-ops — memory here is always coherent. |
| `PHYSFS_init() failed: no error` | `ld1 {v1.16b, v2.16b}, [x2], #32` never wrote its base back (writeback was keyed off `Rm != 31`, but the immediate post-index form *is* `Rm == 31`), so newlib's `strrchr` returned a pointer 32 bytes below `argv[0]`; physfs then asked `malloc` for a negative length, and `PHYSFS_deinit` cleared the error code on the way out | Rewrote both AdvSIMD structure load/store groups against the ARM pseudocode: bit 23 selects writeback, `Rm` chooses immediate vs register increment, the interleaved LD2/LD3/LD4 forms work, `LD1R` replicates, and the single-lane index is decoded from `Q:S:size` |
| `assetsInit() failed: 2345-0010` (`romfsMountSelf` → `LibnxError_IoError`) | `fsFileRead`'s payload was read at `data_area + 0x10`, but libnx converts `fsp-srv` to a domain, which puts a `CmifDomainInHeader` first — so every read asked for 0 bytes at offset 0 | `Cpu::ipc_request_data` finds the payload after the "SFCI" header wherever it is |
| Same error, next layer: `PHYSFS_mount("romfs:/assets.zip")` reported "not found" | BSL/BIT/BIF took their mask from the wrong register, so newlib's vectorised `strchr` missed the `:` in the path; `FindDevice` then fell back to the default device and looked the file up on the SD card | Mask is Vd for BSL, Vm for BIT/BIF |

**Integer-decode bugs behind a JPEG corruption**: hbmenu's icon decoded with
grey luma and magenta chroma. The DC coefficients came out `+1023` off:
libjpeg-turbo's `HUFF_EXTEND` sign-extends the DC difference branchlessly with
`(x - (1 << (s-1))) >> 31`, and our 32-bit arithmetic shift was masking the
operand to 32 bits and then shifting it as a *positive* `i64`, so the
sign-extension mask came out 0. `tools/difftest.py --scalar` then compared 71
integer instructions against qemu-aarch64 and found six more:

| Instruction | Was | Should be |
|---|---|---|
| `asr w, w, w` | shifted the masked value as positive | sign-extend from bit 31 |
| `asr w0, w0, #31`, `sbfx`, `sxth` | left the sign in bits 63:32 | a W write zeroes the top half |
| `extr` | took Rm as the high half | Rn is the high half of `Rn:Rm` |
| `adcs` / `ngc` | bit29 read as "subtract" | bit30 = subtract, bit29 = set flags |
| `sdiv w` | divided unsigned | sign-extend from the operand width |
| `smaddl` / `umaddl` | multiplied the full 64-bit registers | the low 32 bits, sign/zero-extended |
| `cls` | counted the sign bit too | count the bits *after* the sign |

**Frame rate: ~0.7 fps, and where it goes.** Measurements
(`examples/hotspots.rs`, `tools/wasm_bench.mjs`): one steady hbmenu frame is
**~30M emulated instructions**, of which its own software gradient fill is
72% (~20 instructions/pixel over 1280x720), FreeType text rasterisation
~10%, and the GPU/display path ~10%. Interpreter throughput is 28M
instructions/s natively and 21M/s in wasm-in-V8. The floor is now ~9ns/insn
natively and ~20ns in wasm, almost all fetch+dispatch, so a 30M-instruction
frame is genuinely a second of wall clock — no further reordering fixes this.
The next real step for speed is a decoded-block cache (decode each basic
block once, execute from that); beyond that only generating code (a wasm
JIT) reaches real time.

**Known interpreter bug, still open**: with a font that carries hinting
programs (`fpgm`/`prep`/`cvt`), glyphs render with correct heights and
advances but each bitmap is 1-3px wide, as if the outline's x coordinates
collapsed — untouched points look like they never get interpolated. The same
subset with `--no-hinting` renders perfectly, so this is the emulator
mis-executing something the TrueType bytecode interpreter relies on, not a
font problem. Reproduce with two subsets of the same font through
`cargo run -p switch-core --example screenshot -- hbmenu.nro out.ppm 1 <font>`.
Invisible in normal use (the shipped font has no hinting) but a real
correctness gap, and hinting is also 8x slower to emulate.

### NX-Shell — no CPU faults left

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
| `unimplemented SIMD three-same op=0b11011` (`scvtf v28.4s, v31.4s`), then `fdiv v28.4s, …` | The two-register misc and FP three-same groups both fell into the integer three-same decode | Implemented both: REV/CLS/CLZ/CNT/NOT/RBIT/ABS/NEG, compares against zero, XTN/SQXTN/UQXTN/SQXTUN/SHLL, SADDLP/UADALP, FCVTL/FCVTN, FRINTx/FCVTxS/U, SCVTF/UCVTF, FABS/FNEG/FSQRT, FADD/FSUB/FMUL/FDIV/FMLA/FMLS/FMAX/FMIN(NM)/FABD/FCMxx/FACGx/FADDP/FMAXP/FMINP |
| A cache-flush loop running 16x too long | `CTR_EL0` read as 0, so `4 << DminLine` gave a 4-byte stride | Report the Cortex-A57 `0x8444C004` |
| `blr` to 0, then `unimplemented instruction` on `fmov s0, s15`, then `fmadd d0, d31, d26, d0` | The scalar-FP 1-source and 3-source groups were unreachable: bits[15:10] were matched as a unit although the opcode's low bit lands in bit 15, and the 3-source test sat inside a branch that had already required a different top byte | Both groups decoded properly, FMOV as a bit-exact copy, FMADD/FMSUB/FNMADD/FNMSUB fused (`mul_add`) |

What stops it now is content, not code: it wants the **shared system fonts**
from `pl:u`, and there are none here, so it takes its own exit path. Its
renderer needs the shader core regardless.

### sdl-hello — boots and exits cleanly

`sdl-hello.nro` (vgmoose's SDL hello-world, SDL2 on deko3d) uses
libtransistor, which validates replies libnx ignores, and **now boots all
the way through libnx + SDL init and exits cleanly** via `ExitProcess` (svc
0x07, exit code -14 at the deko3d/nv GPU layer). Console output ends with
`set up stdout.` — the point where `SDL_Init` reaches the deko3d video
backend, which isn't emulated.

| Symptom | Root cause | Fix |
|---|---|---|
| `Failed to open connection to fsp-srv: 7e0dd` | Replies carried `type = 0x40`; libtransistor requires 0 or 4 | Replies write `type = 0`, which is what a real server sends |
| `Failed to mount sdcard on fsp-srv: 7ecdd` (move-handle count mismatch) | `OpenSdCardFileSystem` answered with a domain out-object, but libtransistor never converts to a domain and expects a session handle | `Cpu::reply_with_interface` answers with an out-object for a domain request and a move handle otherwise |
| `SDL init failed:` right after `nvdrv` Initialize | `Initialize` returned no raw data; it really returns a `u32 error`, and libtransistor checks the reply's size | Return the error word |
| `panic: attempt to shift left with overflow` in `sext_u64` | `sext_u64(v, 64)` did `1u64 << 64` | `bits >= 64` returns `v` (64-bit sign-extend is identity) |
| `CPU: unallocated logical immediate` on `mov w20, #0x80808080` (0x3201c3f4) | `decode_bit_mask` rejected valid masks: extra `imms & !levels` check; QEMU masks `imms` down to the element size | Rewrote to match QEMU `logic_imm_decode_wmask` |
| `ldrsw x8,[x9,x8,lsl#2]` loaded 0, branched into the jump table | register-offset shift used the byte count (4) instead of `log2(size)` (2) | `offset_from_reg` shifts by `sz` |
| `memset` ran forever (x3 ran past the buffer) | `mrs x5, dczid_el0` returned 0; musl/newlib strides the DC-ZVA loop by `4 << BS` | return BS=4 (64-byte block, Cortex-A57) |
| `CPU: unimplemented system instruction` on `dc zva, x3` | DC ZVA wasn't implemented | zero the 64-byte block at Xt |
| `mov v0.d[1], x9` (INS) unimplemented | copy-group guard rejected `imm5 >= 16` (bit20 is part of imm5, not a guard bit) and q=0 forms | guard is `bits[29:21]==001110000` (QEMU pattern); added INS |
| `ldr d0,[x9,#0xd0]`, `stp d8,d9`, `str d0,[x8,x10]` unimplemented | SIMD&FP immediate/register-offset size mapping was wrong: B/H/S/D are size 00/01/10/11 (the code had S=00, D=01) | corrected size map + added the register-offset FP form and S/D/Q store/load pairs |
| `SIMD MOVI cmode=0b1110` rejected on `movi d8, #0` (0x2f00e408) | imm8 is NOT contiguous: `abcdefgh` = bits 18:16 ++ 9:5 (cmode at 15:12); also only cmode 0000 was implemented | full `AdvSIMDExpandImm` (`simd_imm_const`), QEMU-verified values |
| `setup_fs` returned `-97` (`sqfs_pread` failed) right after boot | `CSEL` family was applying `csinv`/`csinc`/`csneg` invert/increment to the *selected* value instead of the *else* value | compute the else value with invert/inc and then select; `csel` unchanged |

It still presents no frame: its `vi`/binder requests come in libtransistor's
own (non-CMIF) marshalling, which the command-id scan misreads, and the GPU
path needs the shader core anyway.

The frontend has a "Bundled app" selector (demo / hbmenu / NX-Shell /
sdl-hello) and `prod.keys`/`title.keys` persist in `localStorage`.

## GPU (GM20B model)

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

### deko3d / nv (unresolved)

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

## Services (IPC)

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
  `en-US`; hbmenu's `textInit()` depends on it. **`set:sys` does not exist**
  (only `set`'s language-code commands are implemented) — no region code,
  serial number, or firmware version handling.
- Storage size queries (`IFileSystem` cmd 11/12, `GetFreeSpaceSize`/
  `GetTotalSpaceSize`) are a single hardcoded `32 GiB` for both free and
  total — no NAND (BIS) vs SD distinction.
- **psm** / **time** follow the same pattern as the rest: a domain-conversion
  check first, then named commands matched by id, replies written with
  `write_ipc_response`. Use this pattern for any new service.

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

## Retail NCA/NSP loading

### NCA header decryption with prod.keys / title.keys

CDN NCA headers are AES-128-XTS encrypted, so the NCA3 magic at 0x200 is
invisible until decrypted. Given key files, the tool decrypts and inspects
them:

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

Also fixed along the way: NSP loads now take ownership of the staged buffer
instead of copying (halved wasm memory for multi-GB NSPs); the wasm session
table no longer uses `std::sync::Mutex` (the single-threaded wasm backend
asserts on any reentrant `lock()`, and a panic while held leaves it locked
forever under `panic=abort`) — it uses a `SyncCell` and a panic hook that
surfaces Rust panics to the frontend; `switch_read_file` reads just a slice
of an NSP file, so inspecting an NCA header no longer allocates the whole
payload.

### NCA body decryption and launching — verified against a real title

Booting from "this is a Program NCA" to "boot its executable", **proven
against a real commercial title** ("A Short Hike"), decrypted end-to-end with
its own SHA-256 hash verified against Nintendo's own stored master hash —
not a synthetic test. Given `prod.keys` (and either a `title.keys` entry or
the ticket a scene NSP release bundles right next to the content), the
emulator decrypts a Program NCA's ExeFS section, extracts `main` and boots
it, via a "Launch" button on Program NCAs in the NSP inspector — for an NCA
embedded in an NSP (`switch_load_nca_from_nsp`) or a standalone `.nca`
(`switch_load_nca`). A CLI equivalent
(`cargo run -p switch-core --example boot_nsp -- <nsp> <prod.keys>
[title.keys] [max_steps]`) exists for debugging without a browser.

The pieces:

- **`crypto.rs`**: AES-128-CTR (cross-checked against `openssl enc
  -aes-128-ctr`) and SHA-256 (cross-checked against `hashlib`), alongside the
  existing AES-ECB/XTS used for headers.
- **`keys.rs`**: `key_area_key_<application|ocean|system>_XX` and
  `titlekek_XX` parsed directly from `prod.keys` (by generation suffix), like
  `header_key` — stored as dumped, not derived, since deriving them needs
  Nintendo's secret seed constants this project doesn't embed.
- **`ticket.rs`**: parses an ES ticket (`.tik`) and decrypts its title key
  (Common crypto only — Personalized needs a console's ETicket RSA key, out
  of scope). `find_and_decrypt_title_key` locates `<rights_id-hex>.tik`
  among an NSP's files, so a title-key-crypto title works out of the box.
- **`nca.rs`**: decrypts the 4 per-section FS headers, unlocks the AES-CTR
  section key (ticket/`title.keys` title key for rights-id titles, key-area
  slot 2 otherwise), decrypts a section body, and verifies it against the FS
  header's own master hash before trusting it — AES-CTR with the wrong key
  still "decrypts" into plausible-looking garbage, so this hash check is the
  only way to know the keys were actually right (and is how the bugs below
  were found and fixed, by treating a real file as an oracle rather than
  trusting the public spec by itself).
- **`lz4.rs`** / **`nso.rs`**: raw-block LZ4 decompression and the NSO0
  loader (three segments + BSS, like NRO, but per-segment compressed and
  with no external relocator — the linked crt0 relocates itself).

Bugs a real file caught that no synthetic test could (each confirmed via the
master-hash oracle, which flips from "mismatch" to "match" the instant the
fix lands):

| Symptom | Root cause | Fix |
|---|---|---|
| A real Program NCA's section 0 decoded at a multi-terabyte offset, 1 byte long | The section table entry is `u32 start_offset; u32 end_offset` in 0x200-byte media units, not a `u64 offset; u64 size` byte pair | `SectionHeader` parsing now reads the u32 pair and scales by 0x200 |
| AES-CTR decrypted to garbage (hash mismatch) even with the *correct* title key | The AES-CTR counter's low 8 bytes reset to 0 at each section's start; Nintendo's own `nca_calculate_section_ctr` runs the counter across the section's *absolute* position in the file instead | `FsHeader::initial_counter` takes the section's `media_offset` and seeds the counter with `media_offset >> 4` |
| Resolved title key decrypted to garbage even with the right ticket bytes | The ticket's `common_key_id` needs the same "stored value is one more than the real generation, except 0" adjustment the NCA header's key-area generation already gets — `common_key_id` 0x0b needs `titlekek_0a`, not `titlekek_0b` | `Ticket::master_key_revision` applies `saturating_sub(1)` |
| Inspector showed "Crypto: cleartext" and "File size: 0 B" for an obviously-encrypted, 304 MiB real NCA | `crypto_type` (0x21C) is 0 for title-key-crypto titles by design (it only describes key-*area* crypto) — not a bug, just an incomplete check; separately, `file_size` was read from 0x340, which isn't where content size actually lives | `is_encrypted()` now also checks `has_rights_id()`; `file_size`/`program_id` moved to their real offsets (0x208, 0x210) |

### RomFS mounting — verified against a real title

`OpenDataStorageByCurrentProcess` (fsp-srv cmd 200) hands back an `IStorage`
backed by `Cpu`'s decrypted RomFS bytes (`Cpu::set_romfs`, `fs_storage_request`
in `ipc.rs`) — cmd 0 = `Read(offset, size)`, cmd 4 = `GetSize`. This is
deliberately thin: libnx's `romfsMount`/`nn::fs::MountRom` parse the RomFS
header and directory/file tables entirely in guest code against raw byte
reads, so the host only ever needs to serve byte ranges, never parse the
filesystem itself. `Nca::romfs_section_index`/`decrypt_romfs_section` find
and decrypt the section, then slice out the actual RomFS body (it isn't at
byte 0 — see below), sanity-checked against RomFS's own `header_size` field
(always 0x50).

**Fixing this needed a real reference implementation, not just the public
wiki write-up.** The AES-CTR key and counter construction were both already
byte-for-byte correct (independently confirmed: building `hactool` from
source and pointing it at this exact NCA with these exact keys reproduces
the identical `Section CTR` and decrypted title key this project computes).
The actual bug was architectural: an IVFC (`HierarchicalIntegrity`) section
is a multi-level hash tree — byte 0 of the decrypted section is Level 0's
(coarsest) hash table, not RomFS's own header. The real RomFS data lives at
the *last* level's `logical_offset` (`FsHeader::romfs_data_offset`, parsed
from the FS header's `ivfc_hdr_t`, always `level_headers[5]` — hactool's own
`nca.c` reads a fixed index 5 regardless of the header's `num_levels` field,
which reads `7` on this real file despite the level array only holding 6
entries; deriving the index from it instead of hardcoding 5 reads 24 bytes
into the trailing padding and silently returns 0). Once addressed, "A Short
Hike"'s ~276 MiB RomFS section decrypts cleanly on every run — same title
key, same `decrypt_section` code path as the already-hash-verified ExeFS,
just sliced at the right offset.

Along the way, `hactool` also confirmed the ticket/title-key work was exactly
right: the decrypted title key (`95c1b034b8151c9d058126216efde161`) and the
`common_key_id` → `titlekek` generation adjustment (hactool independently
reports "Master Key Revision: 0xA" for `common_key_id` `0x0b`, matching
`Ticket::master_key_revision`'s `saturating_sub(1)`).

Cross-checking against `nca_fs_header_t` also caught that this project's
`fs_type`/`hash_type` field names were swapped relative to hactool's actual
struct — byte 2 is `partition_type` (0=RomFs, 1=Pfs0), byte 3 is `fs_type`
(2=Pfs0, 3=RomFs) and doubles as what this project called `hash_type`
(there's no separate hash-type byte). The byte *positions* being read were
already right, so this was a naming fix, not a behavior change.

Loading a title with no RomFS section (Meta/Control-only) or one whose RomFS
decrypt fails still boots — treated as optional and non-fatal, same as a
missing Horizon service. Full multi-level IVFC hash verification (walking
all 6 levels, the way ExeFS's `HierarchicalSha256` is verified) isn't
implemented — only the `header_size` sanity check — so a corruption subtler
than "wrong key" wouldn't be caught.

### Multi-module loading (`rtld`+`main`+`subsdk`+`sdk`)

A retail title's ExeFS is multiple NSO modules sharing one address space
(`rtld`, `main`, `subsdk0..9`, `sdk`), not just `main` — `main`'s own
GOT/PLT-style indirect calls are unrelocated until `rtld` (Nintendo's own
runtime linker, loaded and run *first*) processes them.
`Cpu::boot_retail_program` loads every present module sequentially at
page-aligned addresses (`collect_modules` in `switch-wasm`, in Nintendo's
required order) and jumps to `rtld`'s entry instead of `main`'s.

**Bug found by tracing "A Short Hike"'s real `rtld` for 25 million
instructions**: it ran cleanly for a long time, then hit
`unimplemented instruction 0x00000000` deep inside its own module — at an
address that, moments earlier, held a legitimate `stp`/`add`/`cmp`/`b.hi`
zero-fill loop. The loop was overwriting its own code. Root cause:
[`NSO_ENTRY_OFFSET`] (`.text`+0x30) is only correct for modules that actually
have a `ModulePtr`/`MOD0` header at `.text`+0 (`main`/`subsdk*`/`sdk` do —
confirmed by disassembling a real module's `.text`+0, which is inert data up
to +0x30 where a textbook crt0 begins). **`rtld` has no such header** — its
`.text`+0 is real code: a `b` that skips an inline PC-relative literal used
by a `bl`+literal idiom (`bl #8; .word disp; ldr wN,[x30]; sbfx; add
xN,xN,x30`) to compute its own load address, since `rtld` must establish
where it was loaded before it can do anything else, including finding its
own `MOD0`. Jumping straight past that bootstrap (as `.text`+0x30 does) left
`x0` — meant to hold `rtld`'s own base address — at `0`, so a later
`bss_end - base` computation produced a ~4 GB byte count fed to that
zero-fill loop, which walked forward through nearly the entire 32-bit
address space until it reached back around into its own `.text` and
clobbered the very instructions running it. `nso::entry_offset` now checks
for the `ModulePtr`+`MOD0` signature (`reserved==0` at `.text`+0, magic
`"MOD0"` at the offset it points to) instead of assuming it's always there;
absent it, entry is `.text`+0.

With that fixed, `rtld`'s real bootstrap runs correctly and gets *much*
further before its next fault, at instruction 9727 instead of 25 million:
`br x16` to a null pointer, from a lazy-PLT-style resolver trampoline. The
program's own console output named exactly what it was trying to resolve:
`[rtld] Unresolved symbol: '_ZN2nn4init5StartEmmPFvvES2_'` (`nn::init::Start`,
normally exported by the `sdk` module).

**This turned out not to need a loader-config handoff at all — `rtld`
doesn't wait to be told what modules exist, it finds them itself.**
Disassembling the whole 6790-byte module turned up its actual discovery
algorithm: a loop that calls `svcQueryMemory` across the address space and,
for each region, checks `type == 3` (`CodeStatic`) *and* `perm == 5` (`R-X`,
built as `movz w24,#0x4f4d; movk w24,#0x3044` — literally the `"MOD0"` magic)
before treating it as a module and processing its dynamic/export table. Two
things broke this in this project's own `svcQueryMemory` stub: it reported a
blanket RWX permission on every mapped page instead of real R-X on `.text`,
so no region ever matched `perm == 5`; and `Memory`'s read-only tracking held
only a *single* `(start, end)` range, silently unprotecting every earlier
module's `.text` each time a later one loaded — fine for a lone homebrew
NRO, wrong for four back-to-back NSOs. Fixed by turning `Memory`'s read-only
tracking into a list of ranges and having `svcQueryMemory` consult it to
report R-X specifically on `.text` pages, RWX elsewhere.

Both fixes together took "A Short Hike"'s real ExeFS from faulting inside
`rtld` at instruction 51 to running **116 million instructions** — through
`rtld`'s relocation of itself and the other three modules, `main`'s init,
`subsdk0`, and into `sdk`'s own code — before hitting a deliberate
`svcBreak` (an SDK-side abort call, not a CPU crash) inside `sdk`. That's
expected: this project has no Horizon service support for retail games yet,
so a real title is expected to run until the first missing service or
explicit abort rather than reach a menu.

### Chasing the `sdk` abort — two real bugs found and fixed, one still open

**`svcGetInfo` type `11` (`RandomEntropy`)** — four kernel-supplied random
words, real hardware's seed for stack canaries/ASLR cookies — wasn't
implemented at all (fell into the stub's `_ => 0` default), confirmed via an
instrumented run that `sdk` init does query it shortly before the abort.
Added (`svc.rs`, `0x29` handler): a SplitMix64 mix keyed by the subvalue —
there's no real entropy source to draw on, and it only needs to look "not
obviously broken" to whatever init-time check reads it. This alone didn't
change the abort (same instruction count, same `svcBreak`), but it's a real
gap independent of this specific abort.

**`svcGetThreadId`/`svcGetProcessId`** had the wrong calling convention: the
real signature is `Result GetXId(u64* out_id, Handle handle)` — `x0` should
be `RESULT_OK` (0) and the id goes in `x1`. The stubs instead put a raw `1`
in `x0` itself, which every caller reads as a *failure* Result code, with
`x1` left stale. Found by decompiling `sdk` with Binary Ninja (driven via its
MCP server, and its own bundled Python API for cases the MCP surface had no
"create function" primitive for) and tracing `Cpu::backtrace`'s frame chain
back from the `svcBreak` site. Fixing it changed the abort's exact
instruction count and the diagnostic value it carries, confirming real
effect — but a *different* abort now fires slightly later, inside
`nn::init::Start` itself (recognizable by its exact signature: two integers,
two function-pointer calls — the same symbol `rtld` failed to resolve at the
very start of this investigation, closing the loop).

Its own internal setup calls into a chain of GOT-indirected helpers ending
in a lock-word check (`(*arg1 & 0xbfffffff) == *(x8 + 0x1b0)`, `x8` read via
`TPIDRRO_EL0`) that reads both sides as `0` and treats that as a match,
walking into an unconditional abort helper. Chasing `x8`'s origin surfaced a
second real bug:

**`TPIDR_EL0` and `TPIDRRO_EL0` were aliased to the same backing field**
(`cpu/mod.rs`/`system.rs`). Real AArch64 makes these two separate registers
— `TPIDR_EL0` freely read/write by EL0, `TPIDRRO_EL0` fixed once by the
kernel and read-only thereafter. `nnSdk`'s own init legitimately writes
`TPIDR_EL0` for its own per-thread bookkeeping; with the two aliased, that
write silently corrupted what should have been the *stable* kernel-provided
TLS pointer everything else (IPC dispatch, the lock-word check above) reads
through `TPIDRRO_EL0`. Split into `tpidr` (kernel-fixed, unchanged semantics)
and a new `tpidr_rw` (the guest-writable one). Verified with a live memory
probe: before the fix, `TPIDRRO_EL0` read back `0` after `nnSdk`'s write (a
per-thread struct pointer registration landed at literal absolute `0x1f8`
instead of the intended `0x1fe001f8`); after, it stays at its bootstrap
value for the entire run, and the registration lands at the correct address.
Zero regressions.

**Resolved**: that `TPIDRRO_EL0` fix changed nothing about the abort on its
own, and the guess that `+0x1b0` was `crit` (a lock word) was wrong. Naming
the backtrace settled it. `sdk`'s NSO carries a full `DT_HASH` dynamic symbol
table — 36,622 symbols — so every address in a real run can be resolved
exactly; `examples/dump_exefs.rs` decrypts the ExeFS, lays the modules out at
the same addresses `boot_retail_program` uses and writes a flat image plus a
sorted `symbols.txt`, and `examples/disasm_flat.rs` disassembles either at its
real load address. The original backtrace reads, innermost last:

```
nn::diag::detail::Abort(nn::Result const*)          <- svcBreak
nn::diag::detail::VAbortImpl(...)
nn::diag::detail::AbortImpl(...)
nn::diag::detail::OnAssertionFailure(...)
nn::os::SdkMutexType::Lock()                        <- the assertion
nn::oe::Initialize()
nninitInitializeSdkModule()
nn::init::Start(...)
```

and the predicate `SdkMutexType::Lock` asserts on is
`nn::os::detail::InternalCriticalSectionImplByHorizon::IsLockedByCurrentThread`,
eleven instructions long:

```
mrs x8, TPIDRRO_EL0
ldr x8, [x8, #0x1f8]      ; ThreadType* out of TLS
ldr w9, [x0]              ; the mutex's lock word
ldr w8, [x8, #0x1b0]      ; the current thread's *handle*
and w9, w9, #0xbfffffff   ; drop the has-waiters bit
cmp w9, w8
csinc w0, wzr, wzr, ne
```

So `+0x1b0` is the thread handle, not `crit`, and the abort was a *recursive
lock* assertion: `Lock()` on a mutex the caller already holds. Both sides read
`0` — an unlocked mutex against a thread handle of zero — so an untouched
mutex looked self-owned.

Three real bugs came out of it, each one moving the boot further:

- **The main thread handle was never delivered.** Horizon's process entry ABI
  puts the launch argument in X0 and the **main thread's handle in X1**;
  `rtld`'s first two instructions are literally `cmp x0, #0` / `mov w19, w1`.
  `boot_retail_program` zeroed all 31 registers and jumped, so `nnSdk` filed a
  handle of 0 into the main `ThreadType` and every `SdkMutex` matched it.
  Seeding X1 with `MAIN_THREAD_HANDLE` (the same value `boot_homebrew` already
  advertises through the homebrew ABI env block) fixed it.
- **`svcGetInfo` CoreMask (0) and PriorityMask (1) fell into the `_ => 0`
  default.** The next abort was `nn::oe::Initialize` →
  `nn::oe::InitializeApplet` → `nn::oe::SetupGpuErrorHandler` →
  `nn::os::RegisterSystemWorkerHandler`, whose inlined highest-set-bit scan
  over `nn::os::GetThreadAvailableCoreMask()` asserts on an empty mask. The
  right values are in the title's own `main.npdm`: its `ThreadInfo` kernel
  capability says `min_core=0 max_core=2` (mask `0b111`) and priorities
  28..=59, which is what every retail application gets.
- **`svcWaitSynchronization` answered X1 = 1 unconditionally.** X1 is the
  *index* of the handle that signaled, not a count, so 1 is out of range for
  the single-handle waits `nn::os::detail::MultiWaitImpl::WaitAny` issues. The
  SDK's system worker took the returned index into its `MultiWaitHolderType`
  list, read one past the end, and `blr`'d the null handler pointer at
  `holder+0x38` — PC 0. Answering 0 is both in range and what "every object is
  pretended signaled" actually implies. hbmenu, sysinfo and NX-Fetch render
  byte-identical frames in the same instruction counts either way.

### The applet stub stopped guessing

The three fixes above got the title to 117.59M instructions and an abort in
`nn::oe::GpuErrorHandler`. Chasing it turned out to be a stub-design problem
rather than a missing-kernel problem: **the `am` stub answered every command it
did not implement with a bare success**, so a caller could not tell an
implemented command from an unimplemented one. `nn::oe::SetupGpuErrorHandler`
asked for `IApplicationFunctions::GetGpuErrorDetectedSystemEvent` (command
130), got "success" and *no copy handle*, and filed handle **0** as the
GPU-error event.

`Cpu::am_unimplemented` now reports `cmif`'s `UnknownCommandId` (`0x1ba0a`) and
prints `[am] unimplemented: <interface> cmd=<n>` once per pair. Two more real
bugs fell out of turning that on:

- **`nnSdk` sends every message in the "with context" encoding.**
  `RequestWithContext` is type 6 and `ControlWithContext` is type 7, where
  `libnx` sends 4 and 5. Every stub tested `ipc_message_type(tls) == 5` for
  control-ness, so `appletOE`'s *first* message from a retail title —
  `QueryPointerBufferSize`, type 7 — was dispatched as IApplicationProxyService
  command 3, which does not exist. `Cpu::ipc_is_control_request` accepts both.
- **The `am` sub-interfaces were unreachable over their own session handles.**
  `libnx` converts `appletOE` to a domain; `nnSdk` never converts, so each
  sub-interface comes back as a separate session handle recorded as `am:…`.
  Those names were not in `svc.rs`'s dispatch, so every retail `am` request
  landed in the generic object-id reply instead of `applet_request`.

With `GetGpuErrorDetectedSystemEvent` answering with a real copy handle the
title reaches **117.82M** instructions — past `nn::oe::Initialize` entirely —
and stops in its own `main`:

```
main!nninitStartup+0x68
sdk!nn::mem::StandardAllocator::Initialize(void*, unsigned long, bool)+0xd0
sdk!nn::diag::detail::OnAssertionFailure(...)
```

**Now open**: the title's own heap setup. `nn::mem::StandardAllocator::
Initialize` asserts on the block it was handed, which points at `svcSetHeapSize`
/ the heap region `svcGetInfo` reports rather than at anything applet-related.

Two other things stay open behind this one. The kernel still has no waitable
object model — `svcWaitSynchronization` reports every handle instantly
signaled, so an event handed out by `am` is "already signaled" the moment a
caller polls it; that needs a per-handle signaled flag, `svcCreateEvent`,
`svcSignalEvent`/`svcClearEvent`/`svcResetSignal`, and a wait that actually
blocks and reschedules. And the rest of the applet surface the title asks for
is enumerated in its `main.npdm` service-access-control list: `appletOE`,
`fsp-srv`, `hid`, `nvdrv`, `vi:u`, `pl:u`, `set`, `time:u`, `audout:u`,
`audren:u`, `nifm:u`, `ssl`, `acc:u0`, `csrng`, and others.

Homebrew reports two honest gaps of its own, both harmless — the caller checks
the Result and carries on, and every frame is byte-identical to before:
`IApplicationFunctions` command 30 (JKSV) and command 60 (nxdumptool).

## Frontend

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
- **Web worker architecture**: `web/worker.js` hosts the wasm module
  (`WebAssembly.instantiateStreaming`) and `web/main.js` talks to it via
  promise-based RPC over `postMessage`. The step budget is unlimited for
  normal runs and 5000 in trace mode, so a well-behaved NRO can run to
  exit/teardown instead of being cut off mid-init.

## Repro / verification

- Host: `cargo run -p switch-core --release --example screenshot -- \
  test-nros/hbmenu.nro out.ppm 3` writes the third presented frame.
- `cargo run -p switch-core --release --example trace -- <nro>` profiles the
  hottest PCs, or breaks on given PCs and dumps registers.
- `cargo run -p switch-core --release --example boot_nsp -- <nsp> <prod.keys>
  [title.keys] [max_steps]` boots a real NSP/NCA from the CLI — the
  equivalent of the browser's "Launch" button, for debugging without one.
- `cargo run -p switch-core --release --example dump_exefs -- <nsp>
  <prod.keys> <title.keys> <out_dir>` decrypts the Program ExeFS and writes
  every module as a flat image at the address `boot_retail_program` loads it
  at, the raw `main.npdm`/NSO files, and a sorted `symbols.txt` of all 36k+
  `DT_HASH` dynamic symbols. **This is what makes a retail backtrace
  readable** — `0x0ce6c0c8` on its own says nothing;
  `sdk!nn::diag::detail::Abort+0x18` says everything.
- `cargo run -p switch-core --release --example disasm_flat -- <module.bin>
  <base> <addr> [count]` disassembles one of those images at its real load
  address.
- `cargo run -p switch-core --release --example retail_trace -- <nsp>
  <prod.keys> <title.keys> [tail]` boots the same way but keeps a ring buffer
  of the last N instructions and dumps it on halt or fault. `RING_FROM=<pc>`
  starts recording at a pc, `RING_MIN=<addr>` drops everything below an
  address — set it past `rtld` (`0x08004000`), whose lazy-binding resolver
  runs hundreds of steps per call and would otherwise fill the whole ring.
- `TRACE_NV=1` traces nvdrv IPC (with guest backtraces), `TRACE_GPU=1` traces
  device opens, ioctls and engine methods, `TRACE_IPC=1` traces all services.
- Browser: `make serve`, load `hbmenu.nro` with the "Horizon (stubbed)" ABI.
- Regression suite: `make test`.

## Next

Two live, independent threads — pick either:

1. **`nn::diag` internals**: resolve `sub_d339760`'s real behavior (see
   above) to find what should populate the `+0x1b0` lock-word comparison, or
   accept it's speculative without Nintendo symbols and move on.
2. **Shader core**: a Maxwell SASS interpreter plus a software rasterizer,
   so `VertexBegin`/`DrawArrays` produce pixels. Needed by NX-Shell and
   `sdl-hello` (both otherwise complete) and any deko3d/EGL-rendering
   homebrew — and by `deko3d`/`nv` service work (above), which currently has
   nothing to render even once stubbed.

Also still open, lower priority: hbmenu's entry label renders as a blank box
(its FreeType text path is worth a look), and `set:sys`/NAND-vs-SD storage
size reporting don't exist (noted under Services).
