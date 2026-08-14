# switch-wasm — boot status

Goal: get the bundled demo and real homebrew (the Homebrew Menu NRO,
`hbmenu.nro`) to actually **run** on the integer-only Phase 1 interpreter.

## Demo

The demo runs identically in Chromium and Firefox 153 (verified headless via
Playwright): it halts at `pc=0x08000024` after 78,161 steps with the gradient
painted and `00000000000068ac` printed.

## hbmenu.nro

`hbmenu.nro` (nx-hbmenu v3.6.1) now boots **~15,900 instructions** through
crt0 → libnx heap init → TLS setup → service IPC handshakes → the complete
applet initialization → hbmenu's own menu code (a NEON `strchr`, FP-based UI
geometry) before stopping at an `svcBreak` inside deko3d's GPU device
creation, which a software interpreter cannot emulate.

### What was wrong and what was fixed

| Finding | Fix |
|---|---|
| **`appletInitialize` busy loop**: `_appletGetSessionProxy` returned `0x19280` (`AM_BUSY_ERROR`), and libnx loops `svcSleepThread(100ms)` → retry while the result matches that — an infinite "wait for applet" spin. | SendSyncRequest now synthesizes a proper CMIF reply with Result `0` (success), plus a fresh non-zero domain object id for the out subservice. |
| **Applet focus-state loop**: after the busy loop, libnx waits for `AppletMessage_FocusStateChanged` before leaving `appletInitialize`; the reply payload was left as a random value, so `appletMainLoop` spun `eventWait` → `ReceiveMessage` → retry. | Reply synthesis parses the request's command id (`SFCI` header) and returns plausible applet data: `ReceiveMessage`→15 (FocusStateChanged), `GetCurrentFocusState`→1 (InFocus), `GetOperationMode`→1, `GetPerformanceMode`→0. |
| The old reply was written into two overlapping "candidate slots" that clobbered each other (`[tls+0x30]` got both the magic and the payload). | Rewrote the reply writer to compute the reply start from the request's hipc header (special header + pid + buffer descriptors) and write one coherent reply. |
| `UMSUBL`/`UMADDL`/`SMULH`/`UMULH` (multiply-long) were unimplemented — hbmenu's number formatting uses them (`0xcccd` divide-by-10 magic). | Added the multiply-long group (signed/unsigned multiply-add/sub and high-product forms). |
| hbmenu's SIMD `strchr` (`'='` search in the menu UI) used ~20 NEON instructions beyond the DUP/MOV/MOVI subset. | Added three-same integer ops (ADD/SUB/CMEQ/CMTST/CMGE/CMHS, halving/saturating add-sub, shift-by-register, ADDP/SMAXP/UMAXP), the bitwise logicals (AND/BIC/ORR/ORN/EOR/BSL/BIT/BIF), the ZIP/UZP/TRN permutes, CMEQ-#0, LD/ST single- and multiple-structure, and fixed DUP element-size decoding (`8 << ctz(imm5)`). |
| Scalar FP (FMOV, FADD, FCVTZS, …) was entirely unimplemented — hbmenu's UI sets up float geometry. | Added a scalar FP subset (FMOV, FADD/FSUB/FMUL/FDIV/FNMUL/FMAX/FMIN/FMAXNM/FMINNM, FABS/FNEG/FSQRT/FRINTx/FCVT, FMADD/FMSUB/FNMADD/FNMSUB, FCMP/FCCMP/FCSEL, SCVTF/UCVTF/FCVTZS/ZU/FCVTN*) using Rust's IEEE f32/f64, plus S/D/Q SIMD&FP loads/stores. |

### Current blocker

The next stop is inside deko3d's `dkDeviceCreate` (via hbmenu's
`graphicsInit`): the GPU driver queries a device-state flag that a stubbed IPC
layer never sets, so hbmenu takes its fatal `svcBreak` error path. Rendering
the menu would require emulating the nv GPU services (or stubbing deko3d
itself), which is beyond a quick fix; each unblocked step typically exposes
the next.

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

Stubbing the nv/deko3d GPU path so the menu renders in the canvas is still
blocked: hbmenu now runs past the earlier deko3d `svcBreak` and instead hangs
in a libnx-init `memcpy` with a corrupt ~4 GiB count. The size field is read
from a data-segment pointer table (`0x822f018`) that points into a localized
string table, producing garbage. The pointer values look correctly relocated,
so this is a subtle libnx-init data/segment-layout issue rather than a simple
decode bug — the next step is investigating the NRO segment layout vs the
RELR relocation addresses (they assume the linker's layout, which must match
the loader's).


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

The frontend now has a "Bundled app" selector (demo / hbmenu / sdl-hello) and
`prod.keys`/`title.keys` persist in `localStorage`.

Next: deko3d device create needs the nv GPU services (nvmap/nvhost) stubbed, or
a software-renderer path, to actually draw the hello-world frame.
