# switch-wasm — boot status

Goal: get the bundled demo and real homebrew (the Homebrew Menu NRO,
`hbmenu.nro`) to actually **run** on the integer-only Phase 1 interpreter.

## Demo

The demo runs identically in Chromium and Firefox 153 (verified headless via
Playwright): it halts at `pc=0x08000024` after 78,161 steps with the gradient
painted and `00000000000068ac` printed.

## hbmenu.nro

`hbmenu.nro` (nx-hbmenu v3.6.1) boots through crt0 → libnx heap/TLS init →
service IPC handshakes (`apm`, `appletOE`, `hid`, `time:u`, `fsp-srv`, `vi:m`,
`set:sys`, `set`, `nvdrv`) → deko3d's `graphicsInit` (`dkDeviceCreate` →
`dkMemBlockCreate` → `dkSwapchainCreate` → `dkQueueCreate`). It connects to
`nvdrv` and sends only cmd 3 (Initialize); deko3d's device init never reaches
`nvOpen` (cmd 0), so no GPU device/command stream is set up.

### What was wrong and what was fixed

| Finding | Fix |
|---|---|
| **`appletInitialize` busy loop**: `_appletGetSessionProxy` returned `0x19280` (`AM_BUSY_ERROR`), and libnx loops `svcSleepThread(100ms)` → retry while the result matches that — an infinite "wait for applet" spin. | SendSyncRequest now synthesizes a proper CMIF reply with Result `0` (success), plus a fresh non-zero domain object id for the out subservice. |
| **Applet focus-state loop**: after the busy loop, libnx waits for `AppletMessage_FocusStateChanged` before leaving `appletInitialize`; the reply payload was left as a random value, so `appletMainLoop` spun `eventWait` → `ReceiveMessage` → retry. | Reply synthesis parses the request's command id (`SFCI` header) and returns plausible applet data: `ReceiveMessage`→15 (FocusStateChanged), `GetCurrentFocusState`→1 (InFocus), `GetOperationMode`→1, `GetPerformanceMode`→0. |
| The old reply was written into two overlapping "candidate slots" that clobbered each other (`[tls+0x30]` got both the magic and the payload). | Rewrote the reply writer to compute the reply start from the request's hipc header (special header + pid + buffer descriptors) and write one coherent reply. |
| `UMSUBL`/`UMADDL`/`SMULH`/`UMULH` (multiply-long) were unimplemented — hbmenu's number formatting uses them (`0xcccd` divide-by-10 magic). | Added the multiply-long group (signed/unsigned multiply-add/sub and high-product forms). |
| hbmenu's SIMD `strchr` (`'='` search in the menu UI) used ~20 NEON instructions beyond the DUP/MOV/MOVI subset. | Added three-same integer ops (ADD/SUB/CMEQ/CMTST/CMGE/CMHS, halving/saturating add-sub, shift-by-register, ADDP/SMAXP/UMAXP), the bitwise logicals (AND/BIC/ORR/ORN/EOR/BSL/BIT/BIF), the ZIP/UZP/TRN permutes, CMEQ-#0, LD/ST single- and multiple-structure, and fixed DUP element-size decoding (`8 << ctz(imm5)`). |
| Scalar FP (FMOV, FADD, FCVTZS, …) was entirely unimplemented — hbmenu's UI sets up float geometry. | Added a scalar FP subset (FMOV, FADD/FSUB/FMUL/FDIV/FNMUL/FMAX/FMIN/FMAXNM/FMINNM, FABS/FNEG/FSQRT/FRINTx/FCVT, FMADD/FMSUB/FNMADD/FNMSUB, FCMP/FCCMP/FCSEL, SCVTF/UCVTF/FCVTZS/ZU/FCVTN*) using Rust's IEEE f32/f64, plus S/D/Q SIMD&FP loads/stores. |
| Scalar SIMD compare-to-zero (`cmge v1.2s, v2.2s, #0` in NX-Shell's font code) hit "unimplemented". | Added `CMGE/CMGT/CMLE/CMLT <Dd>,<Dn>,#0` (bits[28:25]=0b1111, op at 15:10). |
| hbmenu/NX-Shell's IPC requests wrote CMIF headers over the heap's first chunk, corrupting the malloc free-list (`TRACE_HEAP` showed bogus `0xc00000004`-style list pointers and malloc spins). | Root cause: `tpidr` (TLS base) `0x2000_0000` collided with the heap base; app IPC staging writes landed on the heap. Moved TLS to `0x0FF0_0000`. |
| Static constructors never ran (NX-Shell's `std::string` statics / hbmenu's constructors), so e.g. NX-Shell's cwd/device (`sdmc:`) globals were never set. | `boot_homebrew` in the CPU runs crt0 through the `bl main`, sets ThreadVars at TLS+0x1E0 (magic `!TV$`, zeroed `_REENT` at `0x1FF10000`), runs `DT_INIT_ARRAY` (parsed from MOD0/dynamic in `nro.rs`), then resumes at the `bl main`. |
| `svcGetInfo` used the wrong InfoType numbering, so libnx's `g_AslrRegion` was `{0, 39}` and deko3d's address-space reservation returned NULL (→ NULL slot base at `0x08254080` → svcBreak 0x1159). | Corrected to the hbmenu libnx numbering (2/3 Alias, 4/5 Heap, 6/7 Total/Used, 12/13 Aslr, 14/15 Stack). |
| `svcQueryMemory` reported the whole low 2 GiB as one RWX region, so the virtmem allocator saw every candidate as "mapped" and failed. | QueryMemory now reports the contiguous run of pages in the same state (allocated = RWX, untouched soft pages = unmapped). |
| SIMD&FP LDR/STR immediate loads with `imm12 >= 0x800` were misdecoded as register-offset (bit 21 is the top bit of imm12, not a register flag), e.g. `ldr b29, [x0, #0xc80]` used a garbage Rm and a constructor read `x14+x18`. | Register-offset form is `mode 0b00` + bit 21 set; the immediate (unsigned) form is `mode 0b01` with imm12 at bits[21:10]. |
| `fcvtzs w24, d1` (0x1e780038) hit "unimplemented" in sdl-hello's stdout setup. The FCVTZS/FCVTZU handler matched the wrong opc (`0b011000` instead of `0b101000`/`0b111000` for S/D) and its `use_double` was `ftype==0b01 && sf==1` (broke 32-bit-dest double-source conversions). | Match `0b101000|0b111000` (signed) and `0b101001|0b111001` (unsigned); `use_double = ftype == 0b01`. `fcvtzs x8, d1` = 0x9e780028. |

### Current blockers

- **hbmenu renders nothing (framebuffer alloc fails)**: hbmenu now runs its
  full init + `graphicsInit` + main loop (the deko3d multi-wait returns thanks
  to `svcWaitSynchronization` reporting X1=1). But the linear framebuffer
  (`dkMemBlockCreate` for the 1280x720x4 workmem) **returns NULL** — the
  deko3d memblock allocator's dlmalloc gets a bad size arg (`x20` =
  `0x8254998`, a global, instead of the block size) and fails. So hbmenu has
  no CPU-side framebuffer to fill; the menu never visibly renders.
- **NX-Shell uses the devkitPro EGL/OpenGL stack, not deko3d directly**
  (links `-lglad -lEGL -lglapi -ldrm_nouveau`). It runs its full init (14
  constructors), but `Config::Load`'s `GetDirList` fails (fsdev resolves an
  empty path), main returns, and the deko3d teardown faults on a NULL device
  callback — the binary's deko3d region is never entered, so its device list
  is an uninitialized global.

Rendering a frame would require emulating the nv GPU services (nvmap/nvhost)
plus a software GM20B command-stream renderer, or stubbing deko3d itself;
each unblocked step typically exposes the next.

## Repro / verification

- Host: `cargo run -p switch-core --example boot_hbmenu -- /path/to/hbmenu.nro`.
- Browser: `make serve`, then load `hbmenu.nro` (copy into `web/assets/`)
  with the "Horizon (stubbed)" ABI selected.
- Regression suite: `make test` — 65 tests, including coverage for the
  applet reply synthesis, the new three-same/logical/permute SIMD ops, the
  scalar FP subset, the multiply-long forms, the hid shared-memory input
  path, and PFS0 offset rebasing.

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
