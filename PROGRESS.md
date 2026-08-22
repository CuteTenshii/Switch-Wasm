# switch-wasm — boot status

Goal: get real homebrew and real retail titles to **run** on the interpreter,
and put what they render on the canvas.

## Current state

Everything below was re-measured against the tree, not carried forward; the
[fact check](#fact-check) at the end lists what was verified and what turned
out to be stale.

| Title | Result | Frame |
|---|---|---|
| `hbmenu.nro` | full UI, responds to a controller | yes |
| `sysinfo.nro` | renders | yes |
| `NX-Fetch.nro` | renders | yes |
| `JKSV.nro` | full UI: text, icons and save tiles ([below](#jksv-drew-one-glyph-for-a-whole-page-of-text)) | yes |
| `nxdumptool.nro` | renders | yes |
| `NX-Shell.nro` | clean exit at 15.7M steps, no font (below) | no |
| `Checkpoint.nro` | 6.8M steps, no exit | no |
| "A Short Hike" (NSP) | runs 1.5B steps with no fault or abort | not yet |

- **GPU**: the nvdrv/nvmap/GMMU/channel/copy-engine path is real
  (`crates/switch-core/src/gpu`), and so is the 3D shader core: a Maxwell
  SASS interpreter feeding a software rasterizer. See
  [GPU](#gpu-gm20b-model).
- **Retail NCA/NSP**: a real, encrypted commercial title can be decrypted, its
  RomFS mounted, and its multi-module boot sequence (`rtld` → `main` →
  `subsdk*` → `sdk`) run through real `nnSdk` init. `nn::init::Start` →
  `nn::oe::Initialize` → `nn::oe::InitializeApplet` all complete, the heap is
  set up, the title mounts and reads its own RomFS, gets real kernel events and
  real input, brings its whole graphics stack up, creates its display layer,
  opens its audio device and starts feeding it — and then runs on into its own
  loop instead of aborting. **Every service it asks for now has a real
  implementation behind it** — a full boot logs no `no implementation` and no
  `unimplemented` lines. See [Retail NCA/NSP
  loading](#retail-ncansp-loading).

See [Next](#next) for the live open threads.

## Homebrew (NRO) boot & rendering

### hbmenu — full render + input

hbmenu draws its **whole UI** — title and version, the clock, the entry's
name/author/version, the footer's paths and button hints — and **responds to a
controller**: pressing + exits the menu, which is how the input path was
verified end to end.

- **No text anywhere, only the bitmap logo**
  *Root cause:* `pl:u` reported the shared font set as loaded but **empty**,
  and homebrew has no font of its own — it feeds pl's shared memory straight to
  FreeType
  *Fix:* `Cpu::set_shared_font` + a real `GetSharedFontInOrderOfPriority`; the
  frontend fetches `web/assets/font.ttf` (built by `tools/make_font.py`)

- **Controller did nothing**
  *Root cause:* The `HidSharedMemory` writer used invented offsets (`npad` at
  0x3D7C0, lifo at +0x20) and only filled one slot; the frontend also used the
  old `KEY_*` bit order
  *Fix:* Offsets taken from libnx's `hid.h` (npad 0x9A00, 0x5000 stride,
  full_key_lifo +0x28, handheld_lifo +0x378), both player 1 and handheld
  published, Horizon's button order, stick pseudo-buttons derived

- **Faults with a clobbered link register once input woke a parked thread**
  *Root cause:* `virtmemFindStack` found no room in the reported stack region
  and returned NULL, and the no-op `svcMapMemory` let **every thread's stack
  mirror land at address 0** — two threads shared one stack
  *Fix:* A stack region clear of the main stack, and
  `svcMapMemory`/`svcUnmapMemory` that really map, copy and free

The rendering path is real end to end: nvdrv ioctls → nvmap → the graphics MMU
→ the display buffer queue → block-linear de-swizzle → RGBA8888 →
`putImageData`. hbmenu draws with the CPU into a linear buffer, deko3d's
recorded command list blits it into the tiled swapchain image with the copy
engine, and the binder presents it. The icon's JPEG decode is **pixel-exact**
against a reference decode (mean error 0, max 0 over the 256x256 image). hbmenu
never needed the shader core: its command list is one
`dkCmdBufCopyBufferToImage` plus a fence signal, and its assets are raw RGBA
bitmaps.

Getting there needed guest threads (cooperative, with real mutex/condvar
handoff), the `blr x30` fix, correct TRN/ZIP/UZP semantics, the AdvSIMD
by-element and three-different multiply groups, the scalar shift/misc/FP forms,
and the `NextLoadPath` environment entry `launchInit()` requires.
`tools/difftest.py` now checks the SIMD decode against qemu-aarch64 directly.

**CPU/IPC bugs found along the way:**

- **`aligned_alloc` returned NULL for every allocation, so deko3d/libnx
  graphics init failed**
  *Root cause:* Register 31 in the **shifted-register** ADD/SUB form is XZR,
  not SP. `neg x1, x0` is `sub x1, xzr, x0`, and reading SP produced a garbage
  rounded size.
  *Fix:* `add_sub` takes the form explicitly: SP for immediate/extended, XZR
  for shifted.

- **A vectorised table-fill loop never terminated**
  *Root cause:* SIMD&FP load/store mode `0b00` only handled the unscaled
  STUR/LDUR form; bits[11:10] also select post-index and pre-index, whose base
  write-back was missing.
  *Fix:* Decode the index field and write the base back.

- **`unimplemented instruction 0x6f3f077b`**
  *Root cause:* The AdvSIMD shift-by-immediate group
  (SSHR/USHR/SSRA/SRSHR/SRI/SHL/SLI/SSHLL) was absent.
  *Fix:* Implemented; the narrowing forms were already handled separately.

- **`unimplemented SIMD three-same op=0b10011`**
  *Root cause:* MUL/MLA/MLS, SMAX/SMIN, SABD/SABA and CMGT/CMHI were missing
  from the three-same group.
  *Fix:* Added, with a destination-reading variant for the accumulating forms.

- **`unimplemented system instruction 0xd50b7e28`**
  *Root cause:* Only `DC ZVA` was handled; libnx flushes the data cache around
  every GPU buffer.
  *Fix:* The remaining `DC`/`IC` maintenance ops retire as no-ops — memory here
  is always coherent.

- **`PHYSFS_init() failed: no error`**
  *Root cause:* `ld1 {v1.16b, v2.16b}, [x2], #32` never wrote its base back
  (writeback was keyed off `Rm != 31`, but the immediate post-index form *is*
  `Rm == 31`), so newlib's `strrchr` returned a pointer 32 bytes below
  `argv[0]`; physfs then asked `malloc` for a negative length, and
  `PHYSFS_deinit` cleared the error code on the way out
  *Fix:* Rewrote both AdvSIMD structure load/store groups against the ARM
  pseudocode: bit 23 selects writeback, `Rm` chooses immediate vs register
  increment, the interleaved LD2/LD3/LD4 forms work, `LD1R` replicates, and the
  single-lane index is decoded from `Q:S:size`

- **`assetsInit() failed: 2345-0010` (`romfsMountSelf` →
  `LibnxError_IoError`)**
  *Root cause:* `fsFileRead`'s payload was read at `data_area + 0x10`, but
  libnx converts `fsp-srv` to a domain, which puts a `CmifDomainInHeader` first
  — so every read asked for 0 bytes at offset 0
  *Fix:* `Cpu::ipc_request_data` finds the payload after the "SFCI" header
  wherever it is

- **Same error, next layer: `PHYSFS_mount("romfs:/assets.zip")` reported "not
  found"**
  *Root cause:* BSL/BIT/BIF took their mask from the wrong register, so
  newlib's vectorised `strchr` missed the `:` in the path; `FindDevice` then
  fell back to the default device and looked the file up on the SD card
  *Fix:* Mask is Vd for BSL, Vm for BIT/BIF

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
**~30M emulated instructions**, of which its own software gradient fill is 72%
(~20 instructions/pixel over 1280x720), FreeType text rasterisation ~10%, and
the GPU/display path ~10%. Interpreter throughput is 28M instructions/s
natively and 21M/s in wasm-in-V8. The floor is now ~9ns/insn natively and ~20ns
in wasm, almost all fetch+dispatch, so a 30M-instruction frame is genuinely a
second of wall clock — no further reordering fixes this. The next real step for
speed is a decoded-block cache (decode each basic block once, execute from
that); beyond that only generating code (a wasm JIT) reaches real time.

**Known interpreter bug, still open**: with a font that carries hinting
programs (`fpgm`/`prep`/`cvt`), glyphs render with correct heights and advances
but each bitmap is 1-3px wide, as if the outline's x coordinates collapsed —
untouched points look like they never get interpolated. The same subset with
`--no-hinting` renders perfectly, so this is the emulator mis-executing
something the TrueType bytecode interpreter relies on, not a font problem.
Reproduce with two subsets of the same font through `cargo run -p switch-core
--example screenshot -- hbmenu.nro out.ppm 1 <font>`. Invisible in normal use
(the shipped font has no hinting) but a real correctness gap, and hinting is
also 8x slower to emulate.

### NX-Shell — no CPU faults left

NX-Shell runs from boot to a **clean exit** (`ExitProcess`, measured at
15,692,155 steps) **when no shared font is supplied**. Both the frontend and
the `screenshot` example *do* supply one (`web/assets/font.ttf`), and on that
path it does not exit: it runs past a 200M-step budget and presents no frame.
Whatever it does with the font it never finishes, and that has not been
diagnosed. Each fault along the way was an emulator bug, mostly whole decode
groups that were unreachable because a guard tested a field that included a
fixed bit:

- **`ldr s28, [x7]` with `x7 == -1`**
  *Root cause:* `ucvtf d0, x1` decoded as FCVTMU (the int↔float class's
  `rmode`/`opcode` were read as one 6-bit field including the fixed bit21) and
  wrote **x0**, clobbering a live pointer
  *Fix:* Decode `rmode` (bits[20:19]) and `opcode` (bits[18:16]) separately;
  every rounding mode now maps to the right instruction

- **`unimplemented SIMD three-same op=0b1000`**
  *Root cause:* `ext v31.16b, …, #8` — the vector-extract group was missing,
  and the permute guard ignored bit29 so it executed EXT as UZP1
  *Fix:* EXT implemented; permute requires bit29 == 0

- **`unimplemented instruction 0x7f6007fe`**
  *Root cause:* `ushr d30, d31, #32` — only the *vector* shift-by-immediate
  encodings were decoded
  *Fix:* The scalar forms (bit28 set) share the helper with one 64-bit lane

- **`unimplemented instruction 0x1e3ecffe`**
  *Root cause:* `fcsel`/`fccmp` were guarded on bit21 being **clear**; they
  have it set, so both were dead code — and that branch was intercepting the
  FP↔fixed-point conversions
  *Fix:* Conditionals moved to bit21 == 1 keyed on bits[11:10]; the fixed-point
  conversions implemented

- **`unimplemented instruction 0x7e21d9ad`**
  *Root cause:* `ucvtf s13, s13` — the *scalar* two-register-misc group
  *Fix:* Shares the vector implementation with a one-lane count

- **190k identical `pl:u` calls**
  *Root cause:* `GetLoadState` fell through to the generic reply, which answers
  command 1 with the applet's `ReceiveMessage` value (15), so the shared-font
  poll never saw "loaded"
  *Fix:* The applet guesses only apply to applet services; `pl:u` has its own
  stub

- **`unimplemented instruction 0x0` at `pc=0`**
  *Root cause:* Not a crash: libnx's `__nx_exit` branches to the loader return
  address, which `envSetup` takes from `__libnx_init`'s third argument. The
  constructor pass zeroes the registers, so it was 0
  *Fix:* Pass the exit trampoline in x2 when resuming the crt0

- **`unimplemented SIMD three-same op=0b11011` (`scvtf v28.4s, v31.4s`), then
  `fdiv v28.4s, …`**
  *Root cause:* The two-register misc and FP three-same groups both fell into
  the integer three-same decode
  *Fix:* Implemented both: REV/CLS/CLZ/CNT/NOT/RBIT/ABS/NEG, compares against
  zero, XTN/SQXTN/UQXTN/SQXTUN/SHLL, SADDLP/UADALP, FCVTL/FCVTN,
  FRINTx/FCVTxS/U, SCVTF/UCVTF, FABS/FNEG/FSQRT,
  FADD/FSUB/FMUL/FDIV/FMLA/FMLS/FMAX/FMIN(NM)/FABD/FCMxx/FACGx/FADDP/FMAXP/FMINP

- **A cache-flush loop running 16x too long**
  *Root cause:* `CTR_EL0` read as 0, so `4 << DminLine` gave a 4-byte stride
  *Fix:* Report the Cortex-A57 `0x8444C004`

- **`blr` to 0, then `unimplemented instruction` on `fmov s0, s15`, then `fmadd
  d0, d31, d26, d0`**
  *Root cause:* The scalar-FP 1-source and 3-source groups were unreachable:
  bits[15:10] were matched as a unit although the opcode's low bit lands in bit
  15, and the 3-source test sat inside a branch that had already required a
  different top byte
  *Fix:* Both groups decoded properly, FMOV as a bit-exact copy,
  FMADD/FMSUB/FNMADD/FNMSUB fused (`mul_add`)

The clean exit is it giving up for want of the **shared system fonts** from
`pl:u` — that is the no-font path above, not a success. With a font it gets
further and then stalls, un-diagnosed.

### sdl-hello — historical, no longer in the tree

**`sdl-hello.nro` is not in `web/assets/`**, so none of this is currently
reproducible; it is kept because the root causes below are real emulator bugs
that were found through it and are still fixed. Note that `web/index.html`
still offers `sdl-hello` in its bundled-app selector, which therefore 404s —
see the [fact check](#fact-check).

`sdl-hello.nro` (vgmoose's SDL hello-world, SDL2 on deko3d) uses libtransistor,
which validates replies libnx ignores, and **now boots all the way through
libnx + SDL init and exits cleanly** via `ExitProcess` (svc 0x07, exit code -14
at the deko3d/nv GPU layer). Console output ends with `set up stdout.` — the
point where `SDL_Init` reaches the deko3d video backend, which isn't emulated.

- **`Failed to open connection to fsp-srv: 7e0dd`**
  *Root cause:* Replies carried `type = 0x40`; libtransistor requires 0 or 4
  *Fix:* Replies write `type = 0`, which is what a real server sends

- **`Failed to mount sdcard on fsp-srv: 7ecdd` (move-handle count mismatch)**
  *Root cause:* `OpenSdCardFileSystem` answered with a domain out-object, but
  libtransistor never converts to a domain and expects a session handle
  *Fix:* `Cpu::reply_with_interface` answers with an out-object for a domain
  request and a move handle otherwise

- **`SDL init failed:` right after `nvdrv` Initialize**
  *Root cause:* `Initialize` returned no raw data; it really returns a `u32
  error`, and libtransistor checks the reply's size
  *Fix:* Return the error word

- **`panic: attempt to shift left with overflow` in `sext_u64`**
  *Root cause:* `sext_u64(v, 64)` did `1u64 << 64`
  *Fix:* `bits >= 64` returns `v` (64-bit sign-extend is identity)

- **`CPU: unallocated logical immediate` on `mov w20, #0x80808080`
  (0x3201c3f4)**
  *Root cause:* `decode_bit_mask` rejected valid masks: extra `imms & !levels`
  check; QEMU masks `imms` down to the element size
  *Fix:* Rewrote to match QEMU `logic_imm_decode_wmask`

- **`ldrsw x8,[x9,x8,lsl#2]` loaded 0, branched into the jump table**
  *Root cause:* register-offset shift used the byte count (4) instead of
  `log2(size)` (2)
  *Fix:* `offset_from_reg` shifts by `sz`

- **`memset` ran forever (x3 ran past the buffer)**
  *Root cause:* `mrs x5, dczid_el0` returned 0; musl/newlib strides the DC-ZVA
  loop by `4 << BS`
  *Fix:* return BS=4 (64-byte block, Cortex-A57)

- **`CPU: unimplemented system instruction` on `dc zva, x3`**
  *Root cause:* DC ZVA wasn't implemented
  *Fix:* zero the 64-byte block at Xt

- **`mov v0.d[1], x9` (INS) unimplemented**
  *Root cause:* copy-group guard rejected `imm5 >= 16` (bit20 is part of imm5,
  not a guard bit) and q=0 forms
  *Fix:* guard is `bits[29:21]==001110000` (QEMU pattern); added INS

- **`ldr d0,[x9,#0xd0]`, `stp d8,d9`, `str d0,[x8,x10]` unimplemented**
  *Root cause:* SIMD&FP immediate/register-offset size mapping was wrong:
  B/H/S/D are size 00/01/10/11 (the code had S=00, D=01)
  *Fix:* corrected size map + added the register-offset FP form and S/D/Q
  store/load pairs

- **`SIMD MOVI cmode=0b1110` rejected on `movi d8, #0` (0x2f00e408)**
  *Root cause:* imm8 is NOT contiguous: `abcdefgh` = bits 18:16 ++ 9:5 (cmode
  at 15:12); also only cmode 0000 was implemented
  *Fix:* full `AdvSIMDExpandImm` (`simd_imm_const`), QEMU-verified values

- **`setup_fs` returned `-97` (`sqfs_pread` failed) right after boot**
  *Root cause:* `CSEL` family was applying `csinv`/`csinc`/`csneg`
  invert/increment to the *selected* value instead of the *else* value
  *Fix:* compute the else value with invert/inc and then select; `csel`
  unchanged

It still presents no frame: its `vi`/binder requests come in libtransistor's
own (non-CMIF) marshalling, which the command-id scan misreads.

## GPU (GM20B model)

`crates/switch-core/src/gpu` implements the Tegra X1 GPU the way the CPU
implements ARM64 — see AGENTS.md for the module map. Register numbers are taken
from deko3d's generated Maxwell class headers and the driver ABI from libnx's
`nvidia/ioctl`, so real command streams decode as-is.

Working: the nvdrv device/ioctl layer, nvmap, the GMMU, host1x syncpoints,
channels and the GPFIFO/pushbuffer command processor, the MME macro engine,
`ClearBuffers`, report semaphores, the copy engine (including block-linear
conversion and component remap), the 2D blitter, inline-to-memory uploads,
scan-out, and the 3D shader core — see [the shader core](#the-shader-core).

Not implemented: compute. A dispatch records its QMD without running warps.

### nv device init

This section used to say the nv path was "still stubbed at the service
boundary", that a `stub_nvdrv` in `cpu.rs` answered four commands, and that
"the code never reaches `nvOpen`". **None of that is true any more** — there is
no `stub_nvdrv` symbol in the tree at all, and a retail boot opens four device
nodes and issues hundreds of ioctls. It is left recorded here because it was
wrong for long enough to mislead.

What a retail title's device init actually does, traced with `TRACE_GPU=1`:

```
open /dev/nvhost-ctrl-gpu      GetCharacteristics, GetTpcMasks, ZCull{CtxSize,Info}
open /dev/nvhost-as-gpu        address space: bind, alloc space, map buffers
open /dev/nvmap                create / alloc / get-id for each buffer
open /dev/nvhost-gpu           a channel, then GPFIFO setup
```

One transport bug stopped all of it. An ioctl whose argument struct carries a
`{ buf_size, buf_addr }` pair returns its payload *through* that pair, and
`nvIoctl3` is how a caller asks for it in a second buffer rather than inline.
The IPC layer wrote back only the **first** receive buffer, so `libnx` — which
uses `nvIoctl` and reads the payload inline — worked, while `nnSdk`, which uses
`nvIoctl3`, read a **zeroed** 160-byte GPU characteristics struct out of a
buffer nothing had written. A GPU with no architecture, no implementation and
no class numbers is one a driver rejects: it closed the device and returned a
null handle, which `NvRmGpuDeviceGetInfo` then dereferenced.

`NvDrv::ioctl` now takes an `inline_out` buffer, `ctrl_gpu_ioctl` fills it for
the two ioctls with that shape, and the IPC layer writes it into the second
receive buffer.

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
  `en-US`; hbmenu's `textInit()` depends on it. **`set:sys` does exist**
  (`set_sys_request`, dispatched in `svc.rs`) — an earlier version of this file
  claimed it did not.
- **lm** (the log manager), **pctl** (parental controls, reported off), **hid**
  (input negotiation and rumble), **ssl** (the system TLS stack), **acc** (one
  user account, always signed in), **apm** (clock profiles), **bsd** (sockets
  that exist but reach nothing), **ts** (the temperature sensors), **csrng**,
  **spl:**, **pdm:qry**, **pm:\***, **pcv**/**clkrst** and **sfdnsres** are
  implemented; see their sections below.
- **A service with no dedicated handler still answers**, with a fabricated
  object id — that is load-bearing for homebrew which only checks the Result —
  but it now records `[ipc] no implementation: <service> cmd=<n>` once per
  pair, and an `am`-style unimplemented command reports `cmif`'s
  `UnknownCommandId`. Running a guest and reading those lines *is* the
  inventory of what it wants and is not getting.
- Storage size queries (`IFileSystem` cmd 11/12, `GetFreeSpaceSize`/
  `GetTotalSpaceSize`) are a single hardcoded `32 GiB` for both free and total
  — no NAND (BIS) vs SD distinction.
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
- `svcMapSharedMemory` (hid's shared memory) now actually backs the region with
  real zeroed memory and records where it is, so `padUpdate` reads live state
  instead of faulting.

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
surfaces Rust panics to the frontend; `switch_read_file` reads just a slice of
an NSP file, so inspecting an NCA header no longer allocates the whole payload.

### NCA body decryption and launching — verified against a real title

Booting from "this is a Program NCA" to "boot its executable", **proven against
a real commercial title** ("A Short Hike"), decrypted end-to-end with its own
SHA-256 hash verified against Nintendo's own stored master hash — not a
synthetic test. Given `prod.keys` (and either a `title.keys` entry or the
ticket a scene NSP release bundles right next to the content), the emulator
decrypts a Program NCA's ExeFS section, extracts `main` and boots it, via a
"Launch" button on Program NCAs in the NSP inspector — for an NCA embedded in
an NSP (`switch_load_nca_from_nsp`) or a standalone `.nca` (`switch_load_nca`).
A CLI equivalent (`cargo run -p switch-core --example boot_nsp -- <nsp>
<prod.keys> [title.keys] [max_steps]`) exists for debugging without a browser.

The pieces:

- **`crypto.rs`**: AES-128-CTR (cross-checked against `openssl enc
  -aes-128-ctr`) and SHA-256 (cross-checked against `hashlib`), alongside the
  existing AES-ECB/XTS used for headers.
- **`keys.rs`**: `key_area_key_<application|ocean|system>_XX` and `titlekek_XX`
  parsed directly from `prod.keys` (by generation suffix), like `header_key` —
  stored as dumped, not derived, since deriving them needs Nintendo's secret
  seed constants this project doesn't embed.
- **`ticket.rs`**: parses an ES ticket (`.tik`) and decrypts its title key
  (Common crypto only — Personalized needs a console's ETicket RSA key, out of
  scope). `find_and_decrypt_title_key` locates `<rights_id-hex>.tik` among an
  NSP's files, so a title-key-crypto title works out of the box.
- **`nca.rs`**: decrypts the 4 per-section FS headers, unlocks the AES-CTR
  section key (ticket/`title.keys` title key for rights-id titles, key-area
  slot 2 otherwise), decrypts a section body, and verifies it against the FS
  header's own master hash before trusting it — AES-CTR with the wrong key
  still "decrypts" into plausible-looking garbage, so this hash check is the
  only way to know the keys were actually right (and is how the bugs below were
  found and fixed, by treating a real file as an oracle rather than trusting
  the public spec by itself).
- **`lz4.rs`** / **`nso.rs`**: raw-block LZ4 decompression and the NSO0 loader
  (three segments + BSS, like NRO, but per-segment compressed and with no
  external relocator — the linked crt0 relocates itself).

Bugs a real file caught that no synthetic test could (each confirmed via the
master-hash oracle, which flips from "mismatch" to "match" the instant the fix
lands):

- **A real Program NCA's section 0 decoded at a multi-terabyte offset, 1 byte
  long**
  *Root cause:* The section table entry is `u32 start_offset; u32 end_offset`
  in 0x200-byte media units, not a `u64 offset; u64 size` byte pair
  *Fix:* `SectionHeader` parsing now reads the u32 pair and scales by 0x200

- **AES-CTR decrypted to garbage (hash mismatch) even with the *correct* title
  key**
  *Root cause:* The AES-CTR counter's low 8 bytes reset to 0 at each section's
  start; Nintendo's own `nca_calculate_section_ctr` runs the counter across the
  section's *absolute* position in the file instead
  *Fix:* `FsHeader::initial_counter` takes the section's `media_offset` and
  seeds the counter with `media_offset >> 4`

- **Resolved title key decrypted to garbage even with the right ticket bytes**
  *Root cause:* The ticket's `common_key_id` needs the same "stored value is
  one more than the real generation, except 0" adjustment the NCA header's
  key-area generation already gets — `common_key_id` 0x0b needs `titlekek_0a`,
  not `titlekek_0b`
  *Fix:* `Ticket::master_key_revision` applies `saturating_sub(1)`

- **Inspector showed "Crypto: cleartext" and "File size: 0 B" for an
  obviously-encrypted, 304 MiB real NCA**
  *Root cause:* `crypto_type` (0x21C) is 0 for title-key-crypto titles by
  design (it only describes key-*area* crypto) — not a bug, just an incomplete
  check; separately, `file_size` was read from 0x340, which isn't where content
  size actually lives
  *Fix:* `is_encrypted()` now also checks `has_rights_id()`;
  `file_size`/`program_id` moved to their real offsets (0x208, 0x210)

### RomFS mounting — verified against a real title

`OpenDataStorageByCurrentProcess` (fsp-srv cmd 200) hands back an `IStorage`
backed by `Cpu`'s decrypted RomFS bytes (`Cpu::set_romfs`, `fs_storage_request`
in `ipc.rs`) — cmd 0 = `Read(offset, size)`, cmd 4 = `GetSize`. This is
deliberately thin: libnx's `romfsMount`/`nn::fs::MountRom` parse the RomFS
header and directory/file tables entirely in guest code against raw byte reads,
so the host only ever needs to serve byte ranges, never parse the filesystem
itself. `Nca::romfs_section_index`/`decrypt_romfs_section` find and decrypt the
section, then slice out the actual RomFS body (it isn't at byte 0 — see below),
sanity-checked against RomFS's own `header_size` field (always 0x50).

**Fixing this needed a real reference implementation, not just the public wiki
write-up.** The AES-CTR key and counter construction were both already
byte-for-byte correct (independently confirmed: building `hactool` from source
and pointing it at this exact NCA with these exact keys reproduces the
identical `Section CTR` and decrypted title key this project computes). The
actual bug was architectural: an IVFC (`HierarchicalIntegrity`) section is a
multi-level hash tree — byte 0 of the decrypted section is Level 0's (coarsest)
hash table, not RomFS's own header. The real RomFS data lives at the *last*
level's `logical_offset` (`FsHeader::romfs_data_offset`, parsed from the FS
header's `ivfc_hdr_t`, always `level_headers[5]` — hactool's own `nca.c` reads
a fixed index 5 regardless of the header's `num_levels` field, which reads `7`
on this real file despite the level array only holding 6 entries; deriving the
index from it instead of hardcoding 5 reads 24 bytes into the trailing padding
and silently returns 0). Once addressed, "A Short Hike"'s ~276 MiB RomFS
section decrypts cleanly on every run — same title key, same `decrypt_section`
code path as the already-hash-verified ExeFS, just sliced at the right offset.

Along the way, `hactool` also confirmed the ticket/title-key work was exactly
right: the decrypted title key (`95c1b034b8151c9d058126216efde161`) and the
`common_key_id` → `titlekek` generation adjustment (hactool independently
reports "Master Key Revision: 0xA" for `common_key_id` `0x0b`, matching
`Ticket::master_key_revision`'s `saturating_sub(1)`).

Cross-checking against `nca_fs_header_t` also caught that this project's
`fs_type`/`hash_type` field names were swapped relative to hactool's actual
struct — byte 2 is `partition_type` (0=RomFs, 1=Pfs0), byte 3 is `fs_type`
(2=Pfs0, 3=RomFs) and doubles as what this project called `hash_type` (there's
no separate hash-type byte). The byte *positions* being read were already
right, so this was a naming fix, not a behavior change.

Loading a title with no RomFS section (Meta/Control-only) or one whose RomFS
decrypt fails still boots — treated as optional and non-fatal, same as a
missing Horizon service. Full multi-level IVFC hash verification (walking all 6
levels, the way ExeFS's `HierarchicalSha256` is verified) isn't implemented —
only the `header_size` sanity check — so a corruption subtler than "wrong key"
wouldn't be caught.

### Multi-module loading (`rtld`+`main`+`subsdk`+`sdk`)

A retail title's ExeFS is multiple NSO modules sharing one address space
(`rtld`, `main`, `subsdk0..9`, `sdk`), not just `main` — `main`'s own
GOT/PLT-style indirect calls are unrelocated until `rtld` (Nintendo's own
runtime linker, loaded and run *first*) processes them.
`Cpu::boot_retail_program` loads every present module sequentially at
page-aligned addresses (`collect_modules` in `switch-wasm`, in Nintendo's
required order) and jumps to `rtld`'s entry instead of `main`'s.

**Bug found by tracing "A Short Hike"'s real `rtld` for 25 million
instructions**: it ran cleanly for a long time, then hit `unimplemented
instruction 0x00000000` deep inside its own module — at an address that,
moments earlier, held a legitimate `stp`/`add`/`cmp`/`b.hi` zero-fill loop. The
loop was overwriting its own code. Root cause: [`NSO_ENTRY_OFFSET`]
(`.text`+0x30) is only correct for modules that actually have a
`ModulePtr`/`MOD0` header at `.text`+0 (`main`/`subsdk*`/`sdk` do — confirmed
by disassembling a real module's `.text`+0, which is inert data up to +0x30
where a textbook crt0 begins). **`rtld` has no such header** — its `.text`+0 is
real code: a `b` that skips an inline PC-relative literal used by a
`bl`+literal idiom (`bl #8; .word disp; ldr wN,[x30]; sbfx; add xN,xN,x30`) to
compute its own load address, since `rtld` must establish where it was loaded
before it can do anything else, including finding its own `MOD0`. Jumping
straight past that bootstrap (as `.text`+0x30 does) left `x0` — meant to hold
`rtld`'s own base address — at `0`, so a later `bss_end - base` computation
produced a ~4 GB byte count fed to that zero-fill loop, which walked forward
through nearly the entire 32-bit address space until it reached back around
into its own `.text` and clobbered the very instructions running it.
`nso::entry_offset` now checks for the `ModulePtr`+`MOD0` signature
(`reserved==0` at `.text`+0, magic `"MOD0"` at the offset it points to) instead
of assuming it's always there; absent it, entry is `.text`+0.

With that fixed, `rtld`'s real bootstrap runs correctly and gets *much* further
before its next fault, at instruction 9727 instead of 25 million: `br x16` to a
null pointer, from a lazy-PLT-style resolver trampoline. The program's own
console output named exactly what it was trying to resolve: `[rtld] Unresolved
symbol: '_ZN2nn4init5StartEmmPFvvES2_'` (`nn::init::Start`, normally exported
by the `sdk` module).

**This turned out not to need a loader-config handoff at all — `rtld` doesn't
wait to be told what modules exist, it finds them itself.** Disassembling the
whole 6790-byte module turned up its actual discovery algorithm: a loop that
calls `svcQueryMemory` across the address space and, for each region, checks
`type == 3` (`CodeStatic`) *and* `perm == 5` (`R-X`, built as `movz
w24,#0x4f4d; movk w24,#0x3044` — literally the `"MOD0"` magic) before treating
it as a module and processing its dynamic/export table. Two things broke this
in this project's own `svcQueryMemory` stub: it reported a blanket RWX
permission on every mapped page instead of real R-X on `.text`, so no region
ever matched `perm == 5`; and `Memory`'s read-only tracking held only a
*single* `(start, end)` range, silently unprotecting every earlier module's
`.text` each time a later one loaded — fine for a lone homebrew NRO, wrong for
four back-to-back NSOs. Fixed by turning `Memory`'s read-only tracking into a
list of ranges and having `svcQueryMemory` consult it to report R-X
specifically on `.text` pages, RWX elsewhere.

Both fixes together took "A Short Hike"'s real ExeFS from faulting inside
`rtld` at instruction 51 to running **116 million instructions** — through
`rtld`'s relocation of itself and the other three modules, `main`'s init,
`subsdk0`, and into `sdk`'s own code — before hitting a deliberate `svcBreak`
(an SDK-side abort call, not a CPU crash) inside `sdk`. That's expected: this
project has no Horizon service support for retail games yet, so a real title is
expected to run until the first missing service or explicit abort rather than
reach a menu.

### Chasing the `sdk` abort — two real bugs found and fixed, one still open

**`svcGetInfo` type `11` (`RandomEntropy`)** — four kernel-supplied random
words, real hardware's seed for stack canaries/ASLR cookies — wasn't
implemented at all (fell into the stub's `_ => 0` default), confirmed via an
instrumented run that `sdk` init does query it shortly before the abort. Added
(`svc.rs`, `0x29` handler): a SplitMix64 mix keyed by the subvalue — there's no
real entropy source to draw on, and it only needs to look "not obviously
broken" to whatever init-time check reads it. This alone didn't change the
abort (same instruction count, same `svcBreak`), but it's a real gap
independent of this specific abort.

**`svcGetThreadId`/`svcGetProcessId`** had the wrong calling convention: the
real signature is `Result GetXId(u64* out_id, Handle handle)` — `x0` should be
`RESULT_OK` (0) and the id goes in `x1`. The stubs instead put a raw `1` in
`x0` itself, which every caller reads as a *failure* Result code, with `x1`
left stale. Found by decompiling `sdk` with Binary Ninja (driven via its MCP
server, and its own bundled Python API for cases the MCP surface had no "create
function" primitive for) and tracing `Cpu::backtrace`'s frame chain back from
the `svcBreak` site. Fixing it changed the abort's exact instruction count and
the diagnostic value it carries, confirming real effect — but a *different*
abort now fires slightly later, inside `nn::init::Start` itself (recognizable
by its exact signature: two integers, two function-pointer calls — the same
symbol `rtld` failed to resolve at the very start of this investigation,
closing the loop).

Its own internal setup calls into a chain of GOT-indirected helpers ending in a
lock-word check (`(*arg1 & 0xbfffffff) == *(x8 + 0x1b0)`, `x8` read via
`TPIDRRO_EL0`) that reads both sides as `0` and treats that as a match, walking
into an unconditional abort helper. Chasing `x8`'s origin surfaced a second
real bug:

**`TPIDR_EL0` and `TPIDRRO_EL0` were aliased to the same backing field**
(`cpu/mod.rs`/`system.rs`). Real AArch64 makes these two separate registers —
`TPIDR_EL0` freely read/write by EL0, `TPIDRRO_EL0` fixed once by the kernel
and read-only thereafter. `nnSdk`'s own init legitimately writes `TPIDR_EL0`
for its own per-thread bookkeeping; with the two aliased, that write silently
corrupted what should have been the *stable* kernel-provided TLS pointer
everything else (IPC dispatch, the lock-word check above) reads through
`TPIDRRO_EL0`. Split into `tpidr` (kernel-fixed, unchanged semantics) and a new
`tpidr_rw` (the guest-writable one). Verified with a live memory probe: before
the fix, `TPIDRRO_EL0` read back `0` after `nnSdk`'s write (a per-thread struct
pointer registration landed at literal absolute `0x1f8` instead of the intended
`0x1fe001f8`); after, it stays at its bootstrap value for the entire run, and
the registration lands at the correct address. Zero regressions.

**Resolved**: that `TPIDRRO_EL0` fix changed nothing about the abort on its
own, and the guess that `+0x1b0` was `crit` (a lock word) was wrong. Naming the
backtrace settled it. `sdk`'s NSO carries a full `DT_HASH` dynamic symbol table
— 36,622 symbols — so every address in a real run can be resolved exactly;
`examples/dump_exefs.rs` decrypts the ExeFS, lays the modules out at the same
addresses `boot_retail_program` uses and writes a flat image plus a sorted
`symbols.txt`, and `examples/disasm_flat.rs` disassembles either at its real
load address. The original backtrace reads, innermost last:

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
`0` — an unlocked mutex against a thread handle of zero — so an untouched mutex
looked self-owned.

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

`Cpu::unimplemented_command` now reports `cmif`'s `UnknownCommandId`
(`0x1ba0a`) and prints `[ipc] unimplemented: <interface> cmd=<n>` once per
pair. Two more real bugs fell out of turning that on:

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

### `nn::init` and the application heap

`nn::mem::StandardAllocator::Initialize(this, address, size, cache)` asserts on
three things, and the retail boot hit all of them in turn. Disassembling it is
what made that legible: it aborts if the allocator is already initialized, if
the page-aligned span is empty, or if it is under 16 KiB (`ubfx x8, x2,
#14, #50` — the span in 16 KiB units — then `cbz`).

- **`svcGetInfo` InfoType 21/22 (Total/UsedNonSystemMemorySize) fell into the
  `_ => 0` default.** `nn::init`'s startup sizes the application heap as their
  difference and hands it straight to `StandardAllocator::Initialize`, so the
  span was 0. They now report the same figures as 6/7 with the system resource
  taken out.
- **A retail title never calls `svcSetHeapSize`.** Built for the 39-bit address
  space, it picks an address out of its alias region and calls
  `svcMapPhysicalMemory` (0x2c) instead — which was unimplemented. The pages
  are left to the soft map (`bootstrap` soft-maps 0..0x8000_0000, so unwritten
  pages read as zeros and allocate on first write); 0x2d hands them back.
- **The reported region bases were not representable.** `svcGetInfo` answered
  with Horizon's real alias base, 0x10_0000_0000, and guest memory is indexed
  with a `u32` — so `nnSdk` asked to map physical memory at an address that
  truncates to 0. Alias and heap now live at `GUEST_ALIAS_REGION_ADDR`
  (0x4000_0000) and `GUEST_HEAP_REGION_ADDR` (0x3000_0000).
- **InfoType 16 (SystemResourceSizeTotal) has to be 0**, and it is the one
  value here deliberately *not* taken from the title's own NPDM, which asks for
  16 MiB. Reporting 16 MiB makes
  `nn::os::detail::VammManager::InitializeIfEnabled` switch the whole heap onto
  a virtual-address-memory manager that reserves address space out of the alias
  region and backs it page by page. That needs kernel machinery this emulator
  does not have, so `nn::os::AllocateAddressRegion` returned os result 3-12 and
  the allocator aborted anyway. Reporting 0 states what is actually true — no
  memory is reserved for the kernel here — and puts `nnSdk` on its plain heap
  path.

With the heap up, the title runs **127.9M** instructions and stops in its own
asset loading:

```
main!...+0x93348
sdk!nn::fs::OpenDirectory(nn::fs::DirectoryHandle*, char const*, int)+0x218
  -> fs result 2-3005
```

`pctl` is implemented (parental controls, reported off) — a retail title opens
all four aliases before it touches the filesystem, and `nnSdk` will not start
an application it believes is restricted. `lm` is implemented too, so a title's
own `NN_LOG` output now reaches the console instead of being discarded; "A
Short Hike" opens a logger but writes nothing before it aborts, which is normal
for a retail build with its logging compiled out. Between them, **every service
that title reaches now has a real implementation behind it.**

### The filesystem

`nn::fs::OpenDirectory("rom:/Data")` failed with fs result 2-3005, which looked
like a missing mount-name model. It was not: `nn::fs`'s `MountTable` is
client-side inside `sdk`, so a mount name never reaches the emulator at all.
Two bugs underneath it were stopping the mount from happening.

- **`CloneCurrentObject` returned no session handle.** Control command 2 (and 4
  for the Ex form) duplicates a session and must reply with a **new session
  handle as a move handle**. Every service's control path answered it with a
  bare success and nothing else. `nnSdk` clones `fsp-srv` before it mounts
  anything, so `nn::fs::MountRom("rom", …)` failed while talking to handle 0 —
  without ever issuing a single filesystem command, which is why the whole
  `fsp-srv` session showed only three control requests and no `fs` traffic at
  all. It is answered centrally in `svc.rs` now, since it is session management
  and identical for every service.
- **`IStorage::Read` used `IFile::Read`'s field layout.** A file read leads
  with a `u32 option` and pads to 8, so its offset is at +8 and its size at
  +0x10; `IStorage::Read(s64 offset, u64 size)` has neither. Every RomFS read
  came back as "0 bytes at offset 0x50", so the guest mounted its RomFS, parsed
  an empty header, and `HierarchicalRomFileTable` found none of its files.

With both fixed the title mounts `rom:`, reads its RomFS through
`OpenDataStorageByCurrentProcess`, and runs **355.8M** instructions — 2.8x
further than before.

### Kernel events

The title's system worker was being told the GPU had faulted. Two things were
wrong, and only one of them was the wait.

- **Every event was handed out as a *move* handle.** A move handle transfers
  ownership (a sub-session from `reply_with_interface`); a copy handle
  duplicates one the server keeps, which is what every event is. They live in
  different fields of the handle descriptor, so an event in the move slot reads
  back as **0** — `TRACE_WAIT` showed the whole boot waiting on handle 0.
  `Cpu::write_ipc_reply` now emits both lists, copy handles first.
- **Nothing tracked whether an event had fired.** `Cpu::alloc_event` names one
  and records it; `svcWaitSynchronization` answers a **poll** (timeout 0, which
  is what `nn::os::TryWaitSystemEvent` issues) with Timeout when nothing has
  fired, instead of reporting the wait satisfied. That is what stopped
  `nn::oe::GpuErrorHandler` aborting. The display's vsync event is signalled
  from the guest's own presented frames, the only periodic tick here.

A blocking wait with nothing signalled still reports the first handle ready,
and that is deliberate rather than unfinished: `nn::os::detail::MultiWaitImpl::
WaitAny` answers a timeout by returning a **null holder** which
`nn::os::RegisterSystemWorkerHandler` then calls without checking, so telling
that thread the truth jumps to 0 — measured, at 117.5M instructions. Blocking
it for real is worse: nothing here fires those events, so the last runnable
thread has nowhere to go. Getting this honest needs events that actually fire,
not a different answer in the wait.

The title now runs **361.2M** instructions.

### ssl

The system TLS stack: Switch owns the implementation and the certificate store,
and a title asks it to build connections rather than bringing its own. Contexts
and their options are implemented — they are ordinary local objects — while
`CreateConnection` reports itself, because there is no socket layer beneath it
and a connection that can never connect is the fabricated-success problem
again. "A Short Hike" is offline and calls exactly one `ssl` command,
`SetInterfaceVersion`, which `nnSdk` issues at startup because `ssl` is in the
title's NPDM service list.

**Every service the retail title asks for now has a real implementation behind
it** — a full boot logs no `no implementation` and no `unimplemented` lines at
all. With the nv transport fixed on top of that, it reaches **362.5M**
instructions and stops in `nn::vi::CreateLayer` with `vi` result 114-1.

### hid, and rumble

Input arrives in two halves and only one of them is IPC. The **data** lives in
a 256 KiB shared memory region the guest reads directly, which
`Cpu::set_gamepad_state` already filled; `IHidServer` is the **negotiation**
around it, and none of that existed. `libnx` survived on a fabricated reply
because it maps the region by size and this emulator recognises it that way —
`nnSdk` called a method on the null `IAppletResource` it was handed.

`CreateAppletResource` → `IAppletResource::GetSharedMemoryHandle` now hands
over a real copy handle, which `svcMapSharedMemory` prefers over the size
guess. Two things had to be right beyond the obvious:

- **`QueryPointerBufferSize` had to stop being 0.**
  `nn::hid::SetSupportedNpadIdType` marshals its id array as a send-static
  buffer and `nnSdk` checks the negotiated size before sending, so a server
  claiming no room fails the call outright — `hid` result 11-141, aborting
  inside `SetSupportedNpadIdType` itself.
- **The `Set*`/`Get*` pairs have to agree**, because they are read back.

The title runs **362.2M** instructions with this, through input init and into
the GPU driver — where `sdk!NvRmGpuDeviceGetInfo+0x10` called a null pointer,
which is [the nv section](#nv-device-init)'s subject.

Rumble came along with it, and works in the browser: `SendVibrationValue`
carries amplitude/frequency for a low and a high band, the two amplitudes go to
`Cpu::vibration`, and the page maps them onto the Gamepad API's `dual-rumble`
`strongMagnitude`/`weakMagnitude` — Switch rumble is two independently driven
linear resonant actuators, the same shape the browser exposes. Only
Chromium-family browsers implement `vibrationActuator`, so it is best-effort
and silent elsewhere.

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

### The layer, and the binder object in its parcel

`vi`'s `OpenLayer` answers with an Android `Parcel` that describes the layer's
`IGraphicBufferProducer`. What this sent was a twelve-byte payload with the
binder id at offset 8 and nothing else. libnx reads the id out of it and
ignores the rest, so every homebrew here was happy; `nnSdk` also looks at
*what the object is*, found no interface it recognised, and aborted the
process from inside `nn::vi::CreateLayer` with `vi` result 114-1.

The parcel now carries the whole `flat_binder_object` real `vi` sends — type,
flags, the binder id, a cookie, and the interface name `"dispdrv"` — followed
by the four-byte object offset table its header had always pointed at. The id
stays at offset 8, so libnx reads exactly what it read before: hbmenu,
sysinfo, NX-Fetch, JKSV and nxdumptool all render byte-identical frames.

That took the title to **383.1M** instructions, through Unity's RomFS asset
loading and into `nn::audio::OpenDefaultAudioOut`.

### audio

`audout:u` is the plain PCM output device — the one
`nn::audio::OpenDefaultAudioOut` and libnx's `audoutInitialize` open, distinct
from the renderer (`audren`). It was reaching the generic "no implementation"
fallback, which answers with a fabricated object id; `OpenAudioOut` needs an
`IAudioOut` back as a move handle, so the title got a null interface and
branched to pc=0.

`IAudioOutManager` now lists the console's one device and opens it, and
`IAudioOut` implements the buffer protocol that is the whole interface: append
a buffer, wait on the event from `RegisterBufferEvent`, collect the tags of the
buffers the device has finished with. A real device releases a buffer once its
samples have been clocked out; there is no DAC here, so a buffer is released as
soon as its samples have been copied into a queue for the host — a device that
never falls behind, rather than one that claims to have played what it dropped.
A device that has not been started still gets its buffers back, because the
memory is the guest's, but queues nothing. The queue is capped at about a
second and drops from the front, so a paused tab cannot grow it without bound.

The host end is `switch_audio_format` and `switch_audio_pull`, and the page
schedules each pull as one `AudioBuffer` butted against the end of the last.
The emulator does not run a retail title in real time, so underruns are the
normal case; the cursor restarts slightly ahead of `currentTime` rather than
stretching anything to cover the gap.

Then one field width cost the whole title:

- **`OpenAudioOut`'s channel count is 16 bits on the wire**, and the two bytes
  above it are padding the caller never initialises. Reading the whole word and
  echoing it back in the reply told `nnSdk` the device had 0xcafe0002 channels.
  `nn::audio::GetAudioOutChannelCount` hands that straight to its caller, and
  0xcafe0002 is negative, so Unity's `channelCount > 0` check failed. Unity
  treated audio init as failed and tore it down — without calling
  `CloseAudioOut`, because as far as it knew there was nothing to close — then
  retried. The retry hit `nnSdk`'s own registry, which still held the device as
  open, and got `audio` result 2153-0009, which Unity aborts on.

Reading the field at its real width, the title opens its audio device, queues
four 4 KiB buffers, starts it, and goes on to `vi`'s buffer queue and a long
run of `nvdrv` ioctls.

Finding that needed two additions to `retail_trace`: `MARK` prints a line every
time one of a list of pcs runs, so a whole API can be watched being called in
order without recording the steps in between, and `MARK_DUMP` dumps memory at
each mark — here the reply struct `nn::audio::OpenAudioOut` was about to copy
into the caller's `AudioOut`. Unity's own binary has no symbols, so its call
sites were resolved by walking `main`'s `.rela.plt` and matching GOT addresses
back to imported names.

### acc, apm, bsd, and ts

`acc` models a console with **one user account**, always signed in, uid
`ACCOUNT_UID` (`"switch-wasm user"` — sixteen bytes, and nonzero, which is the
part that matters: zero is `AccountUid`'s "nobody is signed in" sentinel).
That is not a stub standing in for a user database. It is what this console is
— no account applet to register a second user with, no profile UI, nowhere to
persist one — so every "who is the current user" question has one determinate
answer and every list is one entry long.

`acc:u0` is the application-facing service and `acc:u1`/`acc:su` the
system-facing ones. They share commands 0..=51 (`GetUserCount`,
`GetUserExistence`, the three user lists, `GetLastOpenedUser`, `GetProfile`,
`TrySelectUserWithoutInteraction`) and **diverge from 100 up, where the same
command id means different things**: 100 is `InitializeApplicationInfo` on
`acc:u0` and `GetUserRegistrationNotifier` on `acc:u1`, 101 is
`GetBaasAccountManagerForApplication` against `GetUserStateChangeNotifier`.
Those arms dispatch on the service the session was opened under, which is what
the test `acc_the_same_command_id_means_different_things_on_u0_and_u1` pins.

Three details were load-bearing:

- **`IProfile::Get` returns its `AccountUserData` through a receive-static
  ("pointer") buffer** — the one descriptor kind that sits *after* the raw
  data, at the unaligned data offset plus `num_data_words` (which counts the
  padding that aligns the CMIF header). Nothing in `ipc.rs` parsed those, so
  `Cpu::ipc_recv_static_buffers` is new, along with `Cpu::ipc_output_buffer`,
  the mirror of `ipc_input_buffer`. `acc` therefore answers
  `QueryPointerBufferSize` with a real size, for the same reason `hid` does: a
  client told the server has no room sends no descriptor at all, and then reads
  the icon id and background colour back out of its own stack.
- **`LoadImage` has to return a real JPEG.** A caller feeds what it gets
  straight to a decoder, so zero bytes is nothing to decode. `solid_jpeg`
  encodes one: a constant image needs no DCT, since the transform of a flat
  block is a single DC coefficient with every AC term zero, so each block is
  one Huffman-coded DC difference (nonzero only in the first block of each
  component) followed by end-of-block, with minimal Huffman tables rather than
  Annex K's. The test decodes the icon back — markers, tables rebuilt out of
  the file's own DHT segments, all 3072 blocks — and checks every one comes
  back as the same colour with the bit stream ending exactly at the EOI.
- **The nickname is real state.** `IProfileEditor::Store` writes it into
  `Cpu::account_nickname` (the host can seed it with `set_user_nickname`) and
  `GetBase` reads it out again, timestamp included. A `Set`/`Get` pair that
  disagrees is the failure mode this whole file keeps rediscovering.

The one place `acc` claims more than it has is `IManagerForApplication`:
`CheckAvailability` succeeds and `GetAccountId` reports a nonzero network
service account, the same trade `nifm`'s permanently-connected ethernet link
makes. It still cannot produce a token — `LoadIdTokenCache` returns zero bytes
— so anything that genuinely authenticates fails there, where the missing piece
actually is.

`apm` is the clock profiles, and there is nothing here to clock: the CPU is an
interpreter and the GPU a software rasterizer. What it must do is agree with
what is already reported elsewhere — `GetPerformanceMode` answers the same
Normal that `am`'s `ICommonStateGetter::GetPerformanceMode` does, which the
test asserts by asking both — and give back per mode whatever
`SetPerformanceConfiguration` was last handed. libnx's `apmInitialize` runs
from `__appInit`, so JKSV opens it before it draws anything, and it was
reaching the fabricated-object-id fallback.

`bsd` is the socket service, and there is **no network behind it**: a browser
tab cannot open a TCP socket and nothing here proxies one. What it models is a
console whose link is up (which is what `nifm` already reports) and on which
nothing ever answers. Sockets can be created, bound, listened on, configured
and closed — those are local operations that genuinely succeed — and everything
that needs a peer fails at once with a definite errno: `connect` is
`ECONNREFUSED`, the data path is `ENOTCONN` on a stream socket and
`ENETUNREACH` on a datagram one, `accept` is `EAGAIN`, `select`/`poll` report
nothing ready. Failing immediately is the point: there is no other thread to
run while a guest waits on a socket, so a timeout would stall the frame loop
that a save manager's update check sits in.

Two details are worth keeping: the errnos are **FreeBSD's**, not Linux's or
newlib's (`EAGAIN` is 35, not 11), because that is what the real service
returns and what guest code is written against; and `fcntl`'s flags word is
stored and returned **verbatim** rather than decoded, since `O_NONBLOCK` is a
different bit in each of those three C libraries and the only thing that has to
hold is that a guest reads back what it set. **`sfdnsres`**, the resolver, is the other half of that stack — `getaddrinfo`,
`gethostbyname` and `getnameinfo` are all IPC calls into it, and
`socketInitialize` opens it alongside `bsd:u`. Nothing resolves, and it says so
definitively: `EAI_NONAME` for the `getaddrinfo` family and `HOST_NOT_FOUND`
for the `gethostbyname` one, rather than the try-again errors that invite a
caller to spin. A numeric address string fails as well, which real hardware
resolves without touching DNS — serializing an `addrinfo` into Horizon's packed
form is guesswork nothing here can check against a real console, and `bsd`
refuses the connect that would follow, so the lookup fails at the point the
guest can act on it. The failure goes in the first word of the three-word
result, which is what makes it robust to the exact field order. The error
strings *are* answered, so a guest that prints why a lookup failed gets a
sentence instead of an empty line.

`ts` is the last of the four: the two thermometers real hardware carries, on
the SoC and on the PCB, both reporting a fixed idle temperature. There is no
silicon here to heat, so idle is the honest reading; what the implementation
has to get right is that the same measurement in degrees and in millidegrees
agrees, and that both sit inside the range the service itself reports. Two
traps, both caught by looking at what NX-Fetch actually drew: `ISession`'s
`GetTemperature` is **command 4, returning a `float`** — the same command id
the server uses for `OpenSession` — so one shared dispatch handed a session's
temperature request another session object, which NX-Fetch printed as "8 C";
and the sensor is selected by the device code's **high byte** (`0x41……` SoC,
`0x43……` PCB), not its low byte, which had the PCB's reading appearing under
the "CPU" label.

### The system version was being read out of stale memory

`set:sys`'s `GetFirmwareVersion`/`GetFirmwareVersion2` had no implementation,
so they took the generic empty-success path and never wrote the
`SetSysFirmwareVersion` the caller was waiting for. The caller then read its
own uninitialized buffer as the system version, and NX-Fetch displayed
**"Horizon OS 115.119.105"** — 115, 119, 105 is the ASCII of `swi`, the start
of `switch-wasm user`, the `acc` uid this emulator had left in that buffer on
an earlier call. A stale-buffer read is exactly the failure the new
pointer-buffer write in `IProfile::Get` was added to avoid, in a service that
predates it.

That number is load-bearing rather than decorative: libnx's `__appInit` seeds
`hosversionGet()` from it and every version gate downstream reads it, so a
garbage version means a guest picking service commands and interface revisions
at random — Checkpoint's `acc` call was `ListQualifiedUsers`, which libnx only
issues on 6.0.0 and later, decided by a version it had read out of stale stack.
`set:sys` now reports **12.1.0**, chosen to sit past the gates the services
here implement and before the ones they do not (17.0.0 moves `ts`'s
measurement). NX-Fetch reads it back correctly, and prints `Player@` beside it
— the `acc` nickname, on screen, through the whole profile path.

Checkpoint is the guest that exercises `acc` here: it opens `acc:u0` and calls
`ListQualifiedUsers` (command 140), which now answers with the console's uid
and a count of 1 instead of a fabricated object id. `bsd` is the other service it wanted:
`RegisterClient`, `StartMonitoring`, then a socket that gets an option set, is
bound, and is closed again — all of which now answer as a real socket layer
would. It still aborts — `svcBreak` Panic with result `0x367` from `0x818bb8c`
— exactly as it did before, and the last IPC before the abort is a long run of
`nvdrv` ioctls, so whatever stops Checkpoint is on the GPU side, not here.

### The rest of what homebrew opens: csrng, spl, pdm, pm, pcv/clkrst

Five more services, each reaching the fabricated-object-id fallback, and each
with a determinate answer once you ask what this console actually is.

**csrng** is the random number generator. Real hardware answers out of the
security processor's hardware RNG; there is none here, and
`wasm32-unknown-unknown` has no OS entropy to borrow, so `Cpu::next_random_u64`
is splitmix64 seeded from the emulated clock. That is **not** a CSPRNG and
nothing out of it should be used as a key — but the fallback left the caller's
buffer untouched, so a "random" value was whatever the stack already held:
non-random, and undetectably so.

**spl:** is the liaison to TrustZone. Everything it exists for is out of reach,
and `GetConfig` — the one command a guest asks for — is answerable: an original
(Icosa) retail unit, not in debug mode, with a fixed device id. Worth knowing
what a real guest asks first: NX-Fetch wants **Atmosphère's extension items at
65000 and up** (CFW API version, emummc type), and zero there reads as "no
custom firmware, booted from internal storage", which is exactly right.

**pdm:qry** is the play-history database, and nothing has ever been played
here: no `pdm:ntfy` records launches, nothing survives a page reload. So every
query answers empty — no events, an empty range, zeroed statistics — which is
the state of a factory-fresh console rather than a placeholder.

**pm:\*** is the process manager, four interfaces on four service names. There
is one process and nothing can create another, so what they answer is identity:
which process is the application, and which program it runs. The process id is
`svcGetProcessId`'s, because those two are the same question through different
doors. The program id defaults to the Album applet's — what hbmenu-launched
homebrew runs as on real hardware — and `Cpu::set_program_id` is there for a
loader that decrypted an NCA and knows the real title id.

**pcv**/**clkrst** are the same clock manager either side of 8.0.0. Their
numbering differs by an offset, not a rename: a `clkrst` device code is
`0x40000000 + module + 1` over the `PcvModule` value `pcv` takes directly.
NX-Fetch asks for `0x40000001`, `0x40000002` and `0x40000039` and labels the
answers CPU, GPU and Memory, so those are CpuBus, GPU and EMC — and reading the
code's low bits as the module puts the GPU's rate under the CPU's name.

**`am` command 30** — the one JKSV logged as an honest gap — is
`BeginBlockingHomeButtonShortAndLongPressed`, which a title asks for before
doing something it must not be interrupted in the middle of (JKSV blocks the
home button while writing a save). There is no home button here and no home
menu to return to, so the request is granted because it is already true.

### One console, one answer

These services describe the same machine from different angles, and a guest
reads several of them. `am`'s `GetOperationMode` answered **1 (Console)** under
a comment that said Handheld — `AppletOperationMode_Handheld` is 0 — so
NX-Fetch printed "Docked" beside a 720p handheld framebuffer, and a title that
picks its resolution by operation mode was being told to render at 1080p. With
that corrected, the operation mode, `apm`'s Normal performance mode and
`clkrst`'s handheld rates all describe one console.

NX-Fetch is the proof: it now reads `Player@` (the `acc` nickname), `Horizon OS
12.1.0`, `1280x720 @ 60Hz [Handheld]`, `CPU ... @ 1020 MHz [40.0C]`, `GPU ... @
384 MHz` and `Memory ... @ 1600 MHz` — every one of those a different service
answering for the same console. Its "Hardware: Unknown" line is the one that
stays blank: no probe identified which command feeds it, and inventing an
answer for a command that cannot be named is what the fabricating fallback did.

### Suspending threads, and reading their registers

Past audio the title stopped twice more, each time on an unimplemented
syscall rather than an abort, and both belong to the same caller: IL2CPP's
garbage collector, which suspends every thread and then scans the roots
living in their registers.

- **`svcSetThreadActivity`** takes a thread out of the scheduler's rotation or
  puts it back. Suspension is tracked apart from `ThreadState` on purpose: it
  does not replace what the thread was doing, so a paused thread blocked on a
  mutex is still blocked on it when it resumes. Horizon refuses to suspend the
  calling thread (`Busy`) and reports a thread already in the requested state
  (`InvalidState`) rather than treating the call as a no-op, and so does this.
- **`svcGetThreadContext3`** fills the 0x320-byte `ThreadContext`: x0..x28, fp,
  lr, sp, pc, pstate, the vector registers, fpcr/fpsr and the thread pointer.
  The register file has to be the real one, so the running thread's comes from
  the live registers and a switched-out thread's from what was saved when it
  last gave up the CPU. No FPCR/FPSR is modelled; both report their reset
  value.

With those two, "A Short Hike" runs **1.5 billion** instructions — the step
budget, not a fault — with no abort and no unimplemented syscall. It still
presents no frame, and the reason is not the GPU: over that whole run it
issues no draw call at all (see [the shader core](#the-shader-core)).

### The shader core

Both halves of this existed before and both were narrow enough that only the
fixtures they were built against could get through them. Every opcode value,
mask, operand position and modifier sub-table below is transcribed from
envytools' `gm107.c`, plus NVIDIA's own `cl9097.h` for the 3D class
registers.

**The decoder** (`gpu/shader/isa.rs`) covered ten opcodes and rejected any
instruction carrying a guard predicate — which is most of them, in anything
with control flow. It now decodes the predicate rather than giving up on it,
along with source negate/absolute and result saturate, and about forty
opcodes: the float ALU (`fadd`/`fmul`/`ffma`/`fmnmx`/`fset`/`fsetp`/`mufu`),
the integer ALU (`iadd`/`iadd3`/`imnmx`/`iscadd`/`isetp`/`iset`/`icmp`/
`imul`/`xmad`), the bitwise and shift group (`lop`/`lop3`/`shl`/`shr`/`shf`/
`bfe`/`popc`/`flo`), conversions (`i2f`/`f2i`/`f2f`/`i2i`), moves
(`mov`/`mov32i`/`s2r`/`sel`/`psetp`), memory (`ld`/`st`/`ldc`/`ldg`/`stg`/
`ldl`/`stl`), the interpolator (`ipa`), `texs`, and control flow
(`bra`/`ssy`/`sync`/`pbk`/`brk`/`pcnt`/`cont`/`exit`/`kil`).

Three decode bugs worth recording, all caught by tests written from the
tables rather than from a capture:

- Opcode groups are matched on the top 16 bits, but the low three of those
  are modifier bits — the group mask is `0xfff8`. Matching all sixteen made
  `ld` (`0xefd9`) miss its own arm.
- `nop` (`0x50b0`) and `f2i` share a low byte, and `ffma` (`0x49a0`) and
  `sel` (`0x4ca0`) share another. The shared-form dispatch now gates on the
  *high* byte first.
- A shader's instructions are not contiguous. Maxwell packs 32-byte blocks
  of one scheduling control word plus three instructions, so a program's
  byte offsets run 0x08, 0x10, 0x18, **0x28**, 0x30, 0x38 — the decoder
  walks slots rather than striding.

**The interpreter** (`gpu/shader/interp.rs`) is a scalar machine: a program
counter, seven predicate registers, an untyped 32-bit register file, and the
reconvergence stack that `ssy`/`sync`, `pbk`/`brk` and `pcnt`/`cont` push and
pop. Constants are read as `u32` and reinterpreted by the consuming
instruction, because that is what the hardware does — the previous `f32`
constant source could not serve an `ldc` feeding integer arithmetic. `kil`
discards the fragment, so the depth store had to move after shading rather
than happening alongside the test.

**The rasterizer** (`gpu/raster.rs`) gained indexed draws (u8/u16/u32
indices, with a vertex cache keyed on index — an indexed mesh reuses vertices
heavily and re-shading each reference is the most expensive thing the loop
can do), all ten primitive topologies, near-plane clipping, face culling, and
scissor rectangles.

One bug here was caught only by the screenshot regression, and is the reason
that check exists: `texs` resolves its texture handle through a constant
bank, and the bank/offset convention was guessed rather than looked up.
JKSV's frame differed in 42059 pixels — a 256x269 region turned flat grey —
and `TRACE_GPU` showed eighteen "read from unbound constant bank 2" errors
where the baseline had two. The handle comes from the driver constant bank
(15) at the offset the instruction names; with that fixed the frame was
byte-identical again and the error counts matched exactly. (The *bank* was
right and the *scaling* was not — the immediate is a dword index, which
[JKSV's text](#jksv-drew-one-glyph-for-a-whole-page-of-text) is what caught.)

A second, subtler one: a *fixed* vertex attribute — bit 6 of
`VertexAttribState[i]`, meaning "this input has no vertex buffer behind it" —
was an error, which dropped the entire draw and every attribute that *was*
bound with it. It reads the `vec4` default (`0,0,0,1`) instead. Two JKSV
draws were being thrown away over an attribute their shader never reads: a
full-screen background quad and the fade-in overlay on top of it. With them
restored JKSV's first frame is nearly black and brightens over the next
fifteen — which is the fade the app actually asked for, and which the dropped
draws had been hiding. The same frame also loses two divider rules that JKSV
itself paints over in that frame; they are its own overdraw, not a
regression. Reading from a *disabled* buffer is still an error, because that
one means some register was read wrong and there is no correct value to
invent.

hbmenu, sysinfo, NX-Fetch and nxdumptool still render byte-identical frames.

**This does not make "A Short Hike" render.** The title's whole run submits
one pushbuffer and issues no draw at all: `GpuStats { submissions: 1,
methods: 3536, clears: 0, draws: 0, copies: 0, macros: 35, inert_methods:
554 }`. Whatever it is waiting for is upstream of the GPU.

### NXpotify: a missing shuffle, then a thread that never let go

Two separate faults, in a row, in an app that reaches further into the system
than anything else in `web/assets` — it links Mesa's nouveau driver, SDL2,
FreeType, curl and mbedTLS, so it exercises the GPU, the socket stack and the
scheduler at once.

**`0x4e1c03bf` was `tbl v31.16b, {v29.16b}, v28.16b`**, in a NEON SHA-512
round. The decoder had no table lookup at all, and the reason it was missing
is worth keeping: TBL/TBX share bits[29:21] with the copy group
(DUP/INS/UMOV/SMOV) and were being swallowed by its guard. Every copy
encoding sets bit10; table lookup has bit15 == 0 and bits[11:10] == 00. The
table is `len+1` consecutive registers, wrapping past v31, and an index past
the end reads zero for TBL but leaves the destination byte untouched for TBX
— which is the whole point of TBX, since it lets a second lookup fill in what
the first missed.

**Then it hung, and the hang was ours, not the app's.** The symptom was a
process pinned at ~119 MiB that looked like a leak and looked like a tight
loop. Neither: the memory is Mesa's up-front pool (one 67 MiB allocation; 16
nvmap allocations in the whole run), and `Cpu::thread_dump` — added for this —
showed the real shape at a glance. Seven threads, all Runnable, and only one
of them ever running:

```
  [0]  handle=0x1    state=Runnable pc=0x8683028
  ...
  [6]* handle=0x102d state=Runnable pc=0x8683b14
```

Thread 6 is NXpotify's Zeroconf HTTP listener, whose loop is
`if (poll(&pfd, 1, 200) <= 0) continue;`. On hardware that sleeps a fifth of
a second per turn. `bsd`'s `Poll` correctly reported nothing ready — and
returned instantly, which turned the loop into one with no blocking syscall
in it. Threads here only hand over at those, so it starved every other
thread, main included, and no frame was ever presented.

The fix is that an empty answer and an *instant* answer are different things:
a `Poll` carrying a non-zero timeout sets `Cpu::pending_yield`, and
`svcSendSyncRequest` reschedules once the reply and X0 are written — it has
to be after, because switching threads swaps the register file, so a
`write_zr(0, …)` past the yield would land on whichever thread runs next. A
zero timeout is an explicit non-blocking probe and still returns at once.

With both fixed NXpotify presents frames (`draws: 50` by frame 10). What
shows is the settings gear in the corner and nothing else — its text is drawn
through SDL2_ttf into GL textures and those draws produce nothing visible, a
GPU-side gap that is unrelated to either bug here. hbmenu (918713 non-black
pixels) and JKSV (21 draws, step 83964291) are unchanged.

That text gap was the `texs` destination and handle pair below
([JKSV](#jksv-drew-one-glyph-for-a-whole-page-of-text)), which is the same
SDL2-through-Mesa path. **Not re-measured against NXpotify** — the `.nro` is
not in this tree — so what it renders now is untested rather than fixed.

Two things this cost time on, worth avoiding next time: `[ipc]` goes to
stdout and `[nv]`/`[vi]` to stderr, so a merged `2>&1 | tail` interleaves them
by *buffer flush*, not by time — a six-call sequence read as an endless loop
three separate times. Count with `grep -c` before believing a tail.

### NXpotify was slow because a texture result rescanned the program

The lag was not the CPU interpreter: one steady-state frame is 481 147 guest
instructions, which the interpreter gets through in about 40 ms. It was the
GPU. A frame is five draws, and those five draws shade **956 000 fragments** —
essentially one full-screen pass — at 2.6 s a frame. Short-circuiting
`shade_fragment` proved it: 2.61 s/frame became 0.32 s.

Counting what a fragment actually does gave the shape: 12 interpreted SASS
instructions, one constant read and one texture sample each, at 2.7 µs a
fragment. Twelve instructions cannot cost 2.7 µs, so the cost was not in the
instructions.

It was in `texs`. A texture result arrives late on hardware, so the
interpreter parks it and lands it just before the first later instruction that
reads the register. Finding that instruction meant scanning forward through
the program, and the scan called `reads()`/`writes()` per instruction — each
of which **builds a `Vec<u8>`**. Four channels, twelve instructions, two
vectors each: about a hundred heap allocations for every covered pixel. But
where a result lands is a property of the *decoded program*, not of the
invocation. It is now worked out once, in a `OnceCell` on `Program` (lazy
rather than filled at construction, so a `Program` built any other way — the
test helpers do — cannot carry a stale table).

Two smaller ones alongside it:

- `Invocation::attr_in`/`attr_out` were `HashMap<u16, f32>`. A fragment writes
  17 of them and the map allocates on its first insert, so that was a hash per
  component plus an allocation per pixel. `a[]` is a ten-bit byte address —
  the whole space is 256 words — so it is a flat array now, with a 256-bit
  written-mask beside it. The mask is not bookkeeping for its own sake: a
  vertex shader that never writes `clip.w` has to get the default 1.0 rather
  than 0.0, so "never written" has to stay distinguishable from "wrote zero".
  It also makes clearing a 32-byte wipe instead of a 1 KiB one.
- The rasterizer built a fresh `Invocation` per covered pixel — a 1 KiB
  register-file wipe and two map allocations each. One per draw now, reset per
  pixel.

**2.61 s/frame → 0.67 s/frame, and every frame is byte-identical** (nxpotify,
hbmenu, JKSV, NX-Fetch). What is left splits as ~0.19 s of rasterization,
depth and blend, ~0.11 s of texture sampling — `sample` still re-reads and
re-parses the 32-byte TIC and TSC out of GPU memory for every sampled pixel,
which is the obvious next one — and the rest in the interpreter itself.

Reusing the `pending` vector across fragments was also tried and measured at
no effect, so it was dropped rather than kept on the theory that it should
have helped.

### JKSV drew one glyph for a whole page of text

JKSV came up as a nearly empty grey screen with two rules across it and a
single letter K at the top left, and it took four separate bugs to get from
there to its real UI. Each one masked the next, which is why the symptom
never looked like four things.

**A `texs` has two destination registers, not one run of four.** The `n`th
enabled channel lands in `dst + n` for the first two and `dst2 + (n - 2)` for
the rest. The interpreter wrote all four consecutively from `dst`. That is
*invisible* whenever `dst2 == dst + 2`, which is what the `tex.frag` fixture
the decoder was first checked against does — so the run-of-four reading
survived every test. JKSV's glyph shader has `dst = $r4, dst2 = $r2`:
channels 2 and 3 landed on `$r6`/`$r7`, and `$r6` was holding the `1/w` that
every later `ipa` multiplies by. Every glyph fragment came out a constant
`(0, 1, 1, 0)` — alpha zero, so the text was there and completely invisible.

The same instruction's channel *mask* has the same shape of problem: the
three-bit selector indexes a different row depending on how many destination
registers the instruction has (`rgb`/`rga`/`rba`/`gba`/`rgba` with two,
`r`/`g`/`b`/`a`/`rg`/`ra`/`ga`/`ba` with one). Only the two-destination row
was decoded.

**The handle immediate is a dword index into the driver constant bank, not a
byte offset.** nouveau's lowering pass emits `tex.r = texBindBase / 4 +
unit`, and the bank's handle table starts 0x20 bytes in. Reading the
immediate as bytes lands a quarter of the way along — in the fixed header
ahead of the table, which begins `0, 1, 2, 3, 4, 5, 6, 7`. That looks
*exactly* like a handle table of sequential `imageId`s with `samplerId` 0,
which is why it was believed: every draw resolved to a plausible handle, and
every draw resolved to the same one. A page of text sampled a single 18x25
glyph texture over and over — hence the K, in every position, at every size.

**Whether the viewport flips y is a register, not a constant.** GL's window
origin is bottom-left and a render target's row 0 is at the top, so Mesa
hands the default framebuffer a negative `VIEWPORT_TRANSFORM.SCALE_Y` and a
user FBO a positive one. JKSV's own capture shows both: `-360` for the
1280x720 window, `+128` for the 256x256 target it renders a save tile into.
`to_screen` hard-coded the flip, so every offscreen target came out upside
down — a tile's label read `ǝɔıʌǝᗡ`, and the tiles stacked in reverse order.
The transform is read from the registers now, on all three axes (`z` was
already `ndc * 0.5 + 0.5`, which is what the registers hold).

**`SetDstWidth` counts elements, not bytes.** A block-linear surface's row
length in *bytes* is what decides how many GOBs a row spans, so a 256-pixel
RGBA row is 1024 bytes wide. Taking the register as bytes made the row four
GOBs wide instead of sixteen and shredded the image into strips — which is
why JKSV's Settings and Extras icons arrived as enormous curved fragments.
Every copy with the remap *off* is unaffected, because an element is one
byte there and the two readings coincide; that is how deko3d drives it, so
hbmenu never saw it.

One artifact was left after all four: a one-pixel dark diagonal through every
save tile. SDL emits a quad as one counter-clockwise and one clockwise
triangle, so both walk the shared diagonal in the *same* direction, agree on
`is_top_left`, and — when the answer is `false` — neither claims the pixels
exactly on it. Consistently-wound geometry never hits this because the two
triangles walk the edge in opposite directions. The tiles are 128x128, so
their diagonal is exactly 45 degrees and pixel centres land right on it. The
rasterizer winds a triangle counter-clockwise before applying the rule now,
swapping the barycentrics back on the way out.

hbmenu, NX-Fetch, lennytube and Checkpoint all render byte-identical frames
across the five fixes.

## Frontend

- **PFS0 offset rebasing**: some repacked NSPs (e.g. ROMSLAB) store PFS0 file
  offsets relative to the end of the string table rather than the file start,
  so extraction returned the wrong bytes ("bad magic"). The parser now detects
  an entry pointing inside the header and rebases by the payload start.
- **CDN NCA headers are encrypted**: `Nca::parse` fails with "bad magic"
  because the NCA3 magic at 0x200 is scrambled until the header is decrypted
  with the title key. The frontend now says "NCA header is encrypted (CDN) —
  needs the title key from the .tik" instead of a bare "bad magic".
- **Wasm memory leaks**: staging buffers for NSP/NRO loads and NCA extraction
  were never freed, so repeated loads accumulated wasm memory until large
  allocations overflowed it (the `toWasm` RangeError). They are now freed after
  use, and `toWasm` re-fetches the linear-memory buffer (which is detached on
  growth).
- **Web worker architecture**: `web/worker.js` hosts the wasm module
  (`WebAssembly.instantiateStreaming`) and `web/main.js` talks to it via
  promise-based RPC over `postMessage`. The step budget is unlimited for normal
  runs and 5000 in trace mode, so a well-behaved NRO can run to exit/teardown
  instead of being cut off mid-init.

- **The bundled-app boot menu is gone.** The overlay offered a "Boot" button
  over a list of NROs fetched from `assets/`, which are not shipped in
  production, so it 404'd for everyone but a local checkout — and the list was
  stale even there: it named `sdl-hello.nro`, which is not in the tree, while
  the NROs that are were never offered. Opening a file and dropping one on the
  stage both still work, and the local `.nro` files stay for the screenshot
  regression tooling.
- **Audio reaches the speakers.** Each run slice pulls whatever PCM `audout`
  has queued and schedules it as one `AudioBuffer` butted against the end of
  the last, at the rate and channel count the guest opened its device with.
- **There is no syscall mode any more.** `SyscallMode` had two variants and
  nothing chose between them: everything that boots a program set `Horizon`,
  and `None` was only the default the bare-CPU tests never changed, where it
  meant "`svc #0` halts". Horizon numbers its syscalls from 1, so 0 is free and
  is now permanently the host halt trap. `switch_set_syscall_mode`, the worker
  command behind it, and `applySyscallMode`'s four call sites are gone.

## Repro / verification

- Host: `cargo run -p switch-core --release --example screenshot -- \
  web/assets/hbmenu.nro out.ppm 3` writes the third presented frame. (There is
  no `test-nros/` directory; the NROs live in `web/assets/`.) The example also
  feeds in `web/assets/font.ttf` as the shared system font, which changes what
  some homebrew does — see NX-Shell above.
- `cargo run -p switch-core --release --example trace -- <nro>` profiles the
  hottest PCs, or breaks on given PCs and dumps registers.
- `cargo run -p switch-core --release --example boot_nsp -- <nsp> <prod.keys>
  [title.keys] [max_steps]` boots a real NSP/NCA from the CLI — the equivalent
  of the browser's "Launch" button, for debugging without one.
- `cargo run -p switch-core --release --example dump_exefs -- <nsp> <prod.keys>
  <title.keys> <out_dir>` decrypts the Program ExeFS and writes every module as
  a flat image at the address `boot_retail_program` loads it at, the raw
  `main.npdm`/NSO files, and a sorted `symbols.txt` of all 36k+ `DT_HASH`
  dynamic symbols. **This is what makes a retail backtrace readable** —
  `0x0ce6c0c8` on its own says nothing; `sdk!nn::diag::detail::Abort+0x18` says
  everything.
- `cargo run -p switch-core --release --example disasm_flat -- <module.bin>
  <base> <addr> [count]` disassembles one of those images at its real load
  address.
- `cargo run -p switch-core --release --example retail_trace -- <nsp>
  <prod.keys> <title.keys> [tail]` boots the same way but keeps a ring buffer
  of the last N instructions and dumps it on halt or fault. `RING_FROM=<pc>`
  starts recording at a pc, `RING_MIN=<addr>` drops everything below an address
  — set it past `rtld` (`0x08004000`), whose lazy-binding resolver runs
  hundreds of steps per call and would otherwise fill the whole ring.
- `cargo run -p switch-core --release --example boot_nx -- <nro>` boots an NRO
  with no font and no framebuffer capture — the shortest path to "does it halt
  cleanly".
- Tracing, all environment-gated and all **host-only**: `TRACE_IPC=1` (every
  service request), `TRACE_SVC=1` (every syscall except the three hot ones,
  plus each `svcGetInfo` answer), `TRACE_WAIT=1` (events as they are handed out
  and every `svcWaitSynchronization`), `TRACE_NV=1` (nvdrv IPC with guest
  backtraces), `TRACE_GPU=1` (device opens, ioctls, engine methods).
  `wasm32-unknown-unknown` has no WASI, so `std::env::var` always fails there
  and none of these can be turned on in the browser; diagnostics that must
  reach a browser user go through `Cpu::diagnostic` instead, which also writes
  the trace buffer the page drains.
- Browser: `make serve`, then drop in an `.nro` or an `.nsp`.
- Regression suite: `make test`. **Two tests fail and have failed throughout**:
  `bootstrap_provides_stack_and_low_memory` and `tpidr_el0_roundtrip` (the
  latter is stale — it writes `TPIDR_EL0` while IPC reads `TPIDRRO_EL0`, which
  was a deliberate split). 231 unit tests and 109 integration tests pass.

## Next

The old item 1 here — "resolve `sub_d339760` to find what should populate the
`+0x1b0` lock-word comparison" — is **resolved**: `+0x1b0` in
`nn::os::ThreadType` is the thread handle, and the process entry ABI delivers
it in X1. See [the applet section](#the-applet-stub-stopped-guessing). So is
the one after it, `nn::vi::CreateLayer` — see [the layer
section](#the-layer-and-the-binder-object-in-its-parcel).

The old item 1, "a Maxwell SASS interpreter plus a software rasterizer", is
also **resolved** — see [the shader core](#the-shader-core). It did not make
the retail title render, for the reason recorded there.

1. **Find out why "A Short Hike" issues no draws.** It has its layer, its
   buffer queue and a stream of `nvdrv` ioctls, and over 1.5 billion
   instructions it submits one pushbuffer carrying 3536 methods and zero
   draw calls. Whatever it is waiting on sits above the GPU: the next thing
   to do is find which thread is spinning at `pc=0xa70b7ec` and on what.
2. **The rest of the thread syscalls.** The title's own scheduler and IL2CPP's
   garbage collector reach for them one at a time as it runs;
   `SetThreadActivity` and `GetThreadContext3` are done, and whatever it asks
   for next will show up as an `unimplemented Horizon syscall` fault.
3. **Homebrew service gaps.** Run a title and read the `[ipc] no
   implementation` lines. Currently: `hid`, `audout`, `acc`, `apm`, `bsd`,
   `ts`, `csrng`, `spl:`, `pdm:qry`, `pm:*`, `pcv`/`clkrst` and `sfdnsres` are
   done, but `usb:hs`, `ncm` and `ns:am2` are not. Save data behind
   `acc` is also still missing: `fs`'s save-data mounts are not implemented, so
   the account exists but has nothing filed under it. What a homebrew run still
   logs is a service opened under an **empty name** — the guest really does ask
   `sm` for one (Checkpoint does; the name word it sends is zero), and `sm`
   hands out a working handle instead of failing the way real `sm` would.

Lower priority: hbmenu's entry label renders as a blank box (its FreeType text
path is worth a look); NAND-vs-SD storage sizes are one hardcoded 32 GiB for
both free and total; `NX-Shell` with a font never finishes; `Checkpoint` never
presents a frame.

## Fact check

This file was re-checked against the tree. What was **wrong** and is now
corrected above:

- "The nv GPU path is still stubbed at the service boundary", `stub_nvdrv` in
  `cpu.rs`, "the code never reaches `nvOpen`" — all false. No such symbol
  exists; a retail boot opens four device nodes and issues hundreds of ioctls.
- "**`set:sys` does not exist**" — false, it is dispatched in `svc.rs`.
- "`nn::diag` internals / `sub_d339760` / the `+0x1b0` lock word" listed as an
  open thread — resolved long ago.
- "NX-Shell runs from boot to a clean exit" stated unconditionally — true only
  with no shared font; with one it never finishes.
- "`sdl-hello.nro` ... boots and exits cleanly" — the file is not in the tree,
  so the claim is not reproducible. `web/index.html` still lists `sdl-hello` in
  its bundled-app selector, and that option **404s**; `NX-Shell`, `sysinfo`,
  `JKSV`, `NX-Fetch`, `nxdumptool`, `Checkpoint` are all present and none are
  offered.
- Repro used `test-nros/hbmenu.nro`; there is no such directory.
- Repro said to load hbmenu "with the 'Horizon (stubbed)' ABI"; that string
  does not appear in the frontend.
- The trace list omitted `TRACE_SVC` and `TRACE_WAIT`, and did not say that
  none of them work in the browser.
- `make test` was listed without noting that two tests have been failing
  throughout.

Verified still true: the GPU module inventory, `nvdrv`'s
`QueryPointerBufferSize` reporting 0, the single hardcoded 32 GiB storage size,
and the root-cause tables for hbmenu, NX-Shell and sdl-hello (those are
historical records of real fixes and were not re-run).
