# switch-wasm — boot status

Goal: get real homebrew and real retail titles to **run**, and put what they
render on the canvas. This is the log of what broke and what it taught.
AGENTS.md is the standing state.

## Where it stands

`.nro`/`.nsp` files are gitignored, so only `test-nros/` re-runs from a clean
checkout. "measured" = run against this tree; the rest is carried forward from
whenever it was last recorded.

| Title | Result | Frame | |
|---|---|---|---|
| `hbmenu.nro` | full UI, responds to a controller | yes | measured |
| `sdl-hello.nro` | exits cleanly at 11.1M steps, 2061 non-black pixels | yes | measured |
| `NX-Shell.nro` | halts at 433,783 steps, no output — **regressed**, see below | no | measured |
| Home Menu (`qlaunch`) | draws; 88 draws a frame on the WebGPU backend | yes | AGENTS.md |
| `sysinfo` / `NX-Fetch` / `nxdumptool` | render | yes | carried |
| `JKSV.nro` | full UI: text, icons, save tiles | yes | carried |
| `Checkpoint.nro` | layout and chrome; text was solid blocks for want of `SHFL`, unchecked since | yes | carried |
| "A Short Hike" (NSP) | composites a full 1280x720 frame; steady state is 2 draws a frame and never a scene | partial | measured |
| "Tomodachi Life" (NSP) | 1.2B steps, no fault, no abort, **no draw calls** | no | carried |

A retail title decrypts, mounts its RomFS, runs `rtld` → `main` → `subsdk*` →
`sdk` through real `nnSdk` init, gets its heap, events and input, brings up its
graphics stack, opens its audio device, and runs on into its own loop. **Every
service it asks for has a real implementation** — a full boot logs no `no
implementation` and no `unimplemented` lines.

`make test`: **783 tests, all passing.**

## Homebrew

**hbmenu** draws its whole UI and responds to a controller (+ exits, which is
how input was verified end to end). Needed guest threads with real mutex/condvar
handoff, the `blr x30` fix, and a shared font — `pl:u` reported the font set
loaded but empty, and homebrew has no font of its own. Its icon's JPEG decode
is pixel-exact against a reference. It never touches the shader core: its
command list is one `dkCmdBufCopyBufferToImage` plus a fence.

**NX-Shell** halts at 433,783 steps with no console output and no frame, with
or without a font. This file used to record a clean `ExitProcess` at 15,692,155
steps *with* output — 36x further. **That is an unchased regression**, and the
`.nro` is in `test-nros/`, which makes it the cheapest thing here to bisect.

**sdl-hello** (libtransistor, which validates replies libnx ignores) boots
through libnx + SDL init and exits cleanly, presenting a partial frame. Its
`vi`/binder requests use libtransistor's own non-CMIF marshalling, which the
command-id scan only partly reads.

**About thirty decode bugs came out of these three.** Nearly all were one
shape: *a guard that tested a field including a fixed bit, so a whole encoding
group was dead code.* The rest were sign-extension widths. Two lessons, and
they are the only part worth carrying:

- **Prove a new guard reaches a real encoding** before believing it.
- **Reach for `tools/difftest.py` before hand-deriving an expected value.** It
  diffs the decode against real ARM under `qemu-aarch64`; it found six integer
  bugs in one pass after a JPEG came out grey and magenta.

## Block translation (`cpu/jit.rs`)

First visit to an address translates forward into ops — operands extracted,
immediates decoded — until something moves the PC; that block is cached and
every later visit runs it with no decoding. It removes **decode, not dispatch**,
and generates no code, so anything untranslated falls back to the interpreter
and the two are the same computation (`jit_bench` diffs the two machines and
they agree).

Worth ~1.9-2.3x. Take the *ratio*, from a min-of-N — the absolute throughput is
the machine's, not the tree's. hbmenu enters each of its 10,645 blocks 535
times, which is why it pays; a run too short to re-enter its blocks shows 1.09x.

Staleness is the memory's job, not the translator's: `Memory` keeps a dirty bit
per page and the JIT drops blocks from dirtied pages. A block never spans a
page, so invalidation is exact.

A real wasm JIT is blocked by the **memory model, not the browser**: a generated
module can only address its own linear memory, and guest memory is a page table.
Flattening it behind a base-plus-bounds check comes first.

## GPU

The nvdrv/nvmap/GMMU/channel/copy-engine path is real, as is the 3D shader
core — a Maxwell SASS interpreter feeding a software rasterizer, with compute
dispatches on the same interpreter one thread at a time. `switch-gpu` is a
separate `wgpu`/WebGPU backend that translates SASS to WGSL; the rasterizer is
the reference it must agree with, and anything it cannot express falls back.

**One transport bug stopped all of device init.** An ioctl whose argument
carries a `{ buf_size, buf_addr }` pair returns its payload *through* that pair,
and the IPC layer wrote back only the first receive buffer. `libnx` reads the
payload inline and worked; `nnSdk` uses `nvIoctl3` and read a **zeroed** GPU
characteristics struct. A GPU with no architecture is one a driver rejects — it
closed the device and returned null, which `NvRmGpuDeviceGetInfo` dereferenced.

**What is left unimplemented, counted rather than guessed.** Tracing every
ioctl five guests issue and grouping by `(type, nr)` is a much shorter list than
the ioctl tables suggest: every command `libnx` sends is implemented, and so is
every command the two retail titles send. `ZbcSetTable`/`ZbcQueryTable` had to
hold what they are given rather than answering a bare success — a driver told
"nothing is registered, ever" re-registers forever. Same for `EventSignal`/
`EventWaitAsync`: a driver parking on a slot nothing would set.

**`VsmsMapping` is the cautionary tale.** `/dev/nvhost-ctrl-gpu` `0x13` is in
no libnx header and no other emulator implements it; it was identified from its
caller's disassembly rather than a table. It was also the only line a whole
retail run logged, **which made it look causal, and it is not** — answering it
with any scalar, refusing it, and implementing it properly all give
byte-identical GPU statistics and the same spinning pc. *The emulator only logs
the gaps it knows it has; a title that never issues a draw logs nothing at all.
Silence is not evidence.*

**Four `texs` bugs, and the reason the screenshot regression exists.** Each
masked the next, so the symptom never looked like four things:

- A `texs` has **two destination registers, not one run of four** — invisible
  whenever `dst2 == dst + 2`, which is exactly what the fixture it was first
  checked against did. JKSV's glyph shader clobbered the `1/w` every later
  `ipa` multiplies by, so every glyph came out alpha-zero: text present and
  completely invisible.
- **The handle immediate is a dword index into the driver constant bank, not a
  byte offset.** Reading it as bytes landed in the fixed header ahead of the
  table, which begins `0, 1, 2, 3…` — that looks *exactly* like a plausible
  handle table, so every draw resolved to a plausible handle, and every draw
  resolved to the same one. A page of text sampled one glyph over and over.
- **`TexCbIndex` is a register, not a constant** — Mesa writes 15, deko3d
  writes 0. Hard-coding 15 sent every deko3d texture fetch at a bank deko3d
  never binds.
- **Whether the viewport flips y is a register, not a constant.** Hard-coding
  the flip turned every offscreen target upside down.

Also: a *fixed* vertex attribute (no vertex buffer behind it) was an error,
which dropped the whole draw rather than reading the `vec4` default — two JKSV
draws thrown away over an attribute their shader never reads. And `SetDstWidth`
counts **elements, not bytes**, which shredded block-linear images into strips.
Reading from a *disabled* buffer is still an error, because that means a
register was read wrong and there is no correct value to invent.

**A Unity title is written in `half`, and none of it decoded.** "A Short Hike"
renders its scene into an RGBA16Float target and composites the frame out of it
with two full-screen quads. Both quads were dropped — one on sampling a float
texture, which `texel_kind_for` accepted in UNORM only, and one on the fp16
ALU — along with 145 other draws. So nothing ever wrote the swapchain buffer
and it presented the zeros it was allocated with: not a black frame, an
**empty** one, which on a canvas is transparent rather than dark.

The whole half-precision group (`hadd2`/`hmul2`/`hfma2`/`hset2`/`hsetp2`,
opcodes and field positions from Eden's `maxwell.inc`) is decoded now. Two
things about it are worth carrying:

- **110 of the 145 were `hadd2.f32`** — swizzles and merge all `F32`, which is
  a plain float add issued on the half unit. Most of what a `half` shader costs
  you is not half arithmetic.
- **`f32_to_f16` had to stop truncating.** Rounding towards zero cost at most
  an ulp of a render target; as the rounding step of every fp16 instruction it
  biases a whole shader. It rounds to nearest, ties to even, and reaches the
  subnormals — the mode WGSL's `pack2x16float` uses, so both backends agree.

With those closed every one of the 295 draws rasterizes and the frame is no
longer empty. 293 of them still cover no pixels — 161 triangles culled and 60
degenerate with all three vertices on one screen point — and **that is not a
bug**: they are one-time Unity shader warm-up, which renders every variant
once with degenerate geometry on purpose. They happen at frame 1 and never
again.

**The steady state is two draws a frame, forever.** Frames 30 and 300 are
byte-identical to frame 1 (519,438 lit pixels), and each costs three clears, a
blit, two full-screen quads and 1.8M instructions. So the title is not still
loading — it has settled. `TRACE_IPC` says it asks the system for nothing but
its two nvdrv ioctls and two `vi` parcels per frame: no filesystem, no `am`,
nothing pending. Its own code is running — 47% of steady-state instructions
are the Boehm GC's mark loop, which is IL2CPP working normally — and simply
produces no renderers.

That is the shape to remember: **Unity culls on the CPU**, so a title whose
scene or camera is wrong emits *no draw calls at all* rather than draws that
cover nothing. Zero scene draws is a symptom of the title's own state, not of
the rasterizer, and chasing it through the GPU is chasing the wrong end. The
vertex buffer those warm-up draws read is written by the title's own `memcpy`
from a source that is already zero — real, and a red herring for the same
reason.

**`AntiAliasEnable` does not size a surface; `MsaaMode` does.** What the frame
*did* say was that its content sat in exactly the left 640x720 of a 1280x720
image — half the width, and the same quarter-of-the-frame shape Just Dance
2019 had before the sample grid existed at all. The title binds a 2560x720
target and sets `MsaaMode` 5 (`2x1_D3D`) the four times it binds it: that is
1280x720 *pixels* at two samples each. `sample_grid` answered "one sample per
pixel" whenever the enable bit was clear, so the surface read as 2560 pixels
wide, its 1280-pixel clear rect covered the left half, and the title's own
2:1 resolve blit shrank that to a quarter.

The gate was a guess — that a guest might leave a stale mode behind — and it
cannot work, because the surface registers count texels either way. Eden sizes
a render target from `anti_alias_samples_mode` alone and never reads
`anti_alias_enable` at all. What the bit really means is GL's `GL_MULTISAMPLE`:
coverage is evaluated once at the pixel centre instead of per sample, which is
`SampleGrid::per_pixel_coverage` and moves no texels. The frame goes from
519,438 lit pixels to **921,600** — exactly 1280x720, the whole resolved
image — with hbmenu and the Home Menu byte-identical.

**Then run it in the mode the frontend actually uses.** Every example booted
handheld, because that is `OperationMode::default()` and none of them set
anything else — while the browser's dock toggle is usually on. A title told
720p whose swapchain is 1080p composites its frame into a corner of that
buffer, and the corner reads exactly like a rendering bug. It is not one:
`DOCKED=1` — in `Title::boot`, so every retail and applet example has it —
fills all **2,073,600** pixels of a 1080p frame, byte-identical to what the
browser was showing. Two frames that disagree are worth checking for a
configuration difference before they are worth debugging.

**The colour write mask was unimplemented**, which the same trace turned up:
`SetCtWrite` (0x680, one register per target, a nibble per channel) plus
`ColorMaskCommon` (0x3E4). "A Short Hike" writes it 420 times a frame and
turns **alpha off for 99 of its draws** and every channel off for one. Writing
all four regardless overwrites exactly what a title meant to keep — and alpha
is what a frame's opacity is read out of, which is the same failure the fp16
gap produced by a different route. An unwritten mask has to read as *all*
channels: zero is the register file's initial value and would blank every
guest that leaves it alone.

**`SHFL` is decoded now, and it took a quad to run.** A scalar interpreter
cannot answer it — the value belongs to another invocation — so the four pixels
of a 2x2 quad run in lock-step: each lane runs to its next shuffle, the warp
exchanges, and all of them go on. That is the barrier machinery a compute
dispatch already had (`Halt::Shuffle` beside `Halt::Barrier`), which is why the
same resolver serves a fragment quad and a kernel's warp of 32. Three things
came out of it:

- **A shuffle is only half of a derivative.** `SHFL.BFLY` fetches the
  neighbour's value and `FSWZADD` subtracts it in whichever direction the
  lane's own position in the quad calls for — two lanes add and two subtract,
  one instruction, no branch. Decoding the shuffle alone would leave every one
  of those shaders failing on the instruction *after* the one just
  implemented.
- **Both of its operands are a register or an immediate, and the immediate
  sits in a different field from the register** (lane index: five bits at 20
  or a register at 20; clamp/segment: thirteen bits at 34 or a register at
  39), with a flag apiece. Reading the wrong field gives a *plausible* lane
  number rather than an error — the same shape as the `texs` handle bug.
- **Helper lanes are the point, not an artefact.** A quad shades the pixels
  its triangle misses and throws the colour away, because they are what the
  covered lanes difference against. That is real work no other draw does, so
  the quad walk is gated on the program containing a `SHFL` or `FSWZADD`;
  everything else keeps the pixel-at-a-time loop. The Home Menu's frame is
  byte-identical across the change.

Checkpoint's `.nro` is not in the tree, so the text it draws with these has
not been looked at since.

**Where a frame's time goes** (ablation on 30 JKSV frames): fragment shader
interpretation 36%, rasterize/depth/blend/pixel I/O 30%, ARM interpreter 25%,
texture sampling 9%.

**A fragment shader runs once per covered pixel, and a full-screen pass is
921,600 of them.** NXpotify was 2.6 s/frame because a texture result rescanned
the program to find where to land — and the scan built a `Vec` per instruction,
about a hundred heap allocations per pixel. Where a result lands is a property
of the *decoded program*, not the invocation. That plus two allocations
(`HashMap` attributes → flat array, one `Invocation` per draw instead of per
pixel) took it to 0.67 s/frame, every frame byte-identical. Reusing the
`pending` vector was measured at no effect and dropped rather than kept on the
theory that it should have helped.

## Performance

**Two things nobody was looking at cost more than everything they were.** A
`perf` profile of a Home Menu boot came back **37% `getenv`** and **18.7%
SipHash** — together more than the shader interpreter, the rasterizer and the
ARM interpreter combined. Neither is emulation:

- `std::env::var("TRACE_...")` is a linear scan of the environment, and forty
  of those sat in per-syscall, per-IPC and per-draw paths. `env_flag!` reads
  each switch once into a `OnceLock`. Nothing can change the environment under
  a running emulator.
- `HashMap`'s default hasher is built to survive keys an attacker chose. Every
  key here is a handle or a guest address this emulator minted itself, and
  `horizon_syscall` looks several up per syscall. `IdMap` is the same map with
  a multiply for a hash.

Together **73.6 s -> 27.7 s** on the same boot, min-of-3, every frame
byte-identical. The lesson is the measurement, not the fixes: nothing in the
tree *looked* slow, and a profile found half the run in code that does no
work.

One trap worth keeping. The obvious integer hash is one multiply, and it moves
entropy **upward** only — while `HashMap` picks its bucket with the **low**
bits. Straight fxhash put all 4096 page-aligned addresses in one bucket. The
fix is a shift and an xor in `finish`, and the test asks for the birthday
limit on dense, page-aligned and sector-aligned keys rather than for
perfection.

## Retail NCA/NSP

**Decryption is verified against a real commercial title**, not a synthetic
test: a Program NCA's ExeFS decrypts and its SHA-256 matches Nintendo's own
stored master hash. That hash check is the whole method — AES-CTR with the
wrong key still "decrypts" into plausible-looking garbage, so **the file is the
oracle and the spec is not**. Every bug below flipped it from mismatch to match:

- The section table is `u32 start; u32 end` in 0x200-byte media units, not a
  `u64 offset; u64 size` byte pair.
- The AES-CTR counter runs across the section's **absolute** position in the
  file, not from 0 at each section.
- A ticket's `common_key_id` needs the same "stored value is one more than the
  real generation" adjustment the key-area generation gets.
- An IVFC section's byte 0 is the **hash table**, not RomFS's header — the real
  data is at the last level's `logical_offset`. hactool reads a fixed index 5
  regardless of the `num_levels` field, which reads 7 on a real file whose level
  array holds 6.

**`rtld` has no MOD0 header, and that cost 25 million instructions to find.**
`NSO_ENTRY_OFFSET` (`.text`+0x30) is only right for modules that have one.
`rtld`'s `.text`+0 is real code — it must establish its own load address before
it can find anything, including its own `MOD0`. Jumping past that bootstrap
left its base at 0, so a `bss_end - base` computation produced a ~4 GB
zero-fill that walked the address space until it overwrote the very loop
running it. The symptom was `unimplemented instruction 0x00000000` at an
address that had held valid code moments earlier.

**`rtld` finds its own modules** — it does not wait to be told. It calls
`svcQueryMemory` across the address space looking for `type == CodeStatic` and
`perm == R-X`. Two things broke that: a blanket RWX permission on every mapped
page, and read-only tracking that held a *single* range, so each new module
silently unprotected the previous one's `.text`.

**The `sdk` abort was a recursive-lock assertion on an untouched mutex.**
`SdkMutexType::Lock` compares the mutex's lock word against the current thread's
handle at `ThreadType+0x1b0`; both read 0, so an unlocked mutex looked
self-owned. Three real bugs fell out of naming the backtrace:

- **The main thread handle was never delivered.** Horizon's process entry ABI
  puts it in **X1** — `rtld`'s first two instructions are literally `cmp x0, #0`
  / `mov w19, w1` — and the loader zeroed all 31 registers.
- **`svcGetInfo` CoreMask/PriorityMask fell into a `_ => 0` default**, so an
  inlined highest-set-bit scan over an empty core mask asserted. The right
  values are in the title's own `main.npdm`.
- **`svcWaitSynchronization` answered X1 = 1** — X1 is the *index* of the
  handle that signaled, not a count, so 1 is out of range for a single-handle
  wait. The SDK read one past the end of its holder list and `blr`'d a null.

*Getting the backtrace is what made all three findable.* `dump_exefs` lays the
modules out at their real load addresses and writes a sorted `symbols.txt` from
`sdk`'s 36,622 `DT_HASH` symbols; `0x0ce6c0c8` says nothing,
`sdk!nn::diag::detail::Abort+0x18` says everything.

**A stub that answers every unknown command with success is worse than one that
fails.** The `am` stub did, so `SetupGpuErrorHandler` asked for an event, got
"success" and no handle, and filed handle **0**. Reporting `UnknownCommandId`
and logging once per pair immediately surfaced two more: `nnSdk` sends every
message in the "with context" encoding (types 6/7 where libnx sends 4/5), and
its `am` sub-interfaces arrive as separate session handles rather than a domain.

**The same bug class, over and over: success with an unfilled out parameter.**
`GetFirmwareVersion` never wrote its struct, so NX-Fetch read its own
uninitialized buffer and displayed "Horizon OS 115.119.105" — the ASCII of
`swi`, from `switch-wasm user`, left there by an earlier `acc` call. That
number is load-bearing: libnx seeds `hosversionGet()` from it and every version
gate downstream branches on it. Likewise `CloneCurrentObject` returned no
session handle, so `nnSdk` mounted its RomFS while talking to handle 0; and
`IStorage::Read` used `IFile::Read`'s field layout, so every RomFS read came
back as "0 bytes at offset 0x50".

**Events must be copy handles, not move handles.** A move handle transfers
ownership; an event is one the server keeps. In the wrong slot it reads back as
0, and the whole boot waited on handle 0.

**`nn::mem::StandardAllocator::Initialize` asserts on an empty span**, and
`svcGetInfo` InfoType 21/22 fell into the `_ => 0` default, so the heap was
sized 0. Also: a retail title never calls `svcSetHeapSize` — built for a 39-bit
address space, it picks an address and calls `svcMapPhysicalMemory`.

**One field width cost a whole title.** `OpenAudioOut`'s channel count is **16
bits on the wire**, and the two bytes above it are padding the caller never
initialises. Echoing the whole word back told `nnSdk` the device had 0xcafe0002
channels — negative, so Unity's `channelCount > 0` check failed, it tore audio
down *without* calling `CloseAudioOut`, and the retry hit `nnSdk`'s registry,
which still held the device open, and aborted.

**Tomodachi Life started a thread on a null 800M instructions in**, reported as
a CPU fault at an address that is not code. Its thread constructor stores an
allocation **without checking it**; its allocator had run dry. The real cause
was the address-space split: `svcGetInfo` reported 1.5 GiB total, the title
asked for exactly that, and spent it on pools it sizes *from that same figure*.
The other 1.5 GiB was an alias region this title never touches. `nnSdk` picks
its heap route at init from the NPDM's system resource size, so each layout now
spends the space on the region its own titles grow into.

**An update NSP holds no game.** Its Program NCA carries a complete ExeFS —
patched modules, not a delta — and a RomFS section encrypted `AesCtrEx`: the
BKTR form, holding only the ranges the update changed plus the two tables that
index them against the base title's. Booting one alone loads and runs and then
aborts, 114 M instructions in, after `OpenDataStorageByCurrentProcess` answers
`0x202` three times — which is what Just Dance 2017's `svcBreak` was, not a CPU
or a GPU bug. `bktr.rs` composes the pair, streaming both containers. Two
things the format does not say out loud:

- The subsection counter replaces the section counter's **generation** word,
  not its secure value. The wrong way round is quiet: the tables are written
  under the section's own counter and still decrypt, so everything validates
  and every byte of *data* is noise.
- An update's Program NCA carries the **base** title id — the `...800` update
  id is only on the container's Meta NCA. So pairing is by program id, and
  what identifies a container as an update at all is that its RomFS is a patch.

**DLC is not an update, and is much less than one.** A DLC container has no
Program NCA and no patch: it is one Data NCA with an ordinary RomFS, whose
title id is the base title's plus an index — `0100bee017fc1001` is Just Dance
2023's add-on content #1 — and a title mounts it by that id through
`OpenDataStorageByDataId`, the same path a system data archive takes. So the
reading half was already built. What was missing was `aoc:u` saying the content
exists: it answered `CountAddOnContent` with 0 by design, and a title never
asks for content the list does not have. It now reports what the host
registered, and `ListAddOnContent` writes the real indices into the caller's
buffer — both halves out of one registration, since an index listed but not
mountable is worse than one never listed. The base id comes from the NACP when
it declares one and is derived (base program id, low 13 bits masked, plus
0x1000) when it does not, which is what Eden's `IAddOnContentManager` does.

**A `Poll` that returns instantly is not the same as one that reports nothing
ready.** NXpotify's Zeroconf listener is `if (poll(&pfd, 1, 200) <= 0)
continue;`. On hardware that sleeps 200 ms; here it turned into a loop with no
blocking syscall, and threads only hand over at those — so it starved every
other thread and no frame was ever presented. A non-zero timeout now yields.

**A step counter that keeps climbing is not a title that keeps running.** Just
Dance 2023 reached seven billion steps with no frame. Profiling it at four
budgets showed the main thread retiring a *constant* 760M instructions at 1B,
1.5B, 3B and 6B — it had stopped at 765M and everything after was two threads
inside `svcWaitSynchronization`, one on `am:gpu-error` and one on `vi:vsync`,
each rewinding onto the `svc` and yielding on every scheduler slice. 70% of the
machine, going nowhere, and the *Steps* readout counting all of it. Three
things came out of it:

- **A blocking wait parks now.** Only `signal_event` can change the answer, so
  re-asking between signals learns nothing. `ThreadState::WaitEvent` holds the
  thread with its PC on the `svc` until a signal or the display tick. The Home
  Menu reaches its tenth frame in **39.2M steps instead of 170.6M**, the frame
  byte-identical; Just Dance's main thread went from 25% of the machine to 82%.
  The catch found on the way: waking on every `signal_event` call rather than
  on an event's *transition* put the spin straight back, because `audio_tick`
  re-signals a device whose buffer has come due on every wait in the process.
- **The top 16 MiB of the stack region was not the guest's to have.**
  `svcGetInfo` 14/15 advertised 0x1800_0000 + 128 MiB, and the return
  trampolines (0x1F00_0000) and every thread's TLS block (0x1FE0_0000 up) sat
  inside it. `nn::os::CreateThread` picks an address in there itself: the title
  was already mapping stacks at 0x1fdc8000, one page short of the main thread's
  TLS, and a stack that landed on it would overwrite the thread pointer every
  `SdkMutex` reads. The trampolines and TLS moved to 0x2000_0000, above the
  region, so the advertised range is free and none of the 128 MiB is lost.

  **Shrinking the region instead is the wrong fix, and it looked like the right
  one.** Ending it at the trampoline removes the same overlap and costs 16 MiB;
  Just Dance then got *past* an abort in `nn::os::detail::ThreadManager::
  CreateAliasStackUnsafe` it had always taken, which read as a cure. It was
  luck: the region size is the modulus nnSdk's random placement uses, so a
  different size is a different address sequence. With the region back at 128
  MiB the title maps the same 38 stacks at the same 38 addresses as before and
  aborts on the 39th, exactly as it did — and "A Short Hike", which the 16 MiB
  cost had started aborting in the same function, runs to budget again.

  So **`CreateAliasStackUnsafe` is still an open bug**, and the overlap was
  never what caused it. `svcMapMemory` here refuses nothing, so those 38 maps
  all succeeded and the 39th never reached a syscall: whatever asserts is
  inside `nn::os::detail::AslrSpaceAllocator`'s own bookkeeping, which is built
  from `svcGetInfo` 12/13 and has the heap and alias regions outside the ASLR
  region it is told about.
- **One undecoded opcode cost every draw.** `VOTE.VTG` (0x50e0/0xfff8) sits two
  instructions before `exit` in the loading screen's vertex shader, and a
  refused instruction fails the whole draw — all 52 of them, every frame the
  clear colour. It writes neither register nor predicate: Eden stubs it too
  (`translate/impl/vote.cpp`). The draws are no longer refused; the frame is
  still the clear colour, which is a separate question.

Booted **without its update** the same title deadlocks for real at 765M steps —
main polling a semaphore on a 1 ms timed wait, two workers parked on one
condvar, the 16-thread job pool idle — and no IPC of any kind follows. Patched
with 1.0.1 it presents 76 frames and dies elsewhere. The base container's
deadlock is not chased; the update is the configuration that runs.

## Services

Every service the retail title asks for is implemented; `usb:hs` and `ncm` are
the two still absent. The design rule throughout is **answer as the console you
actually are**: one user account (uid nonzero, because zero means "nobody is
signed in"), an idle temperature, a link that is up with nothing behind it,
empty play history, a factory-fresh console — not a stub, and not a failure,
because a failure puts callers on the path built for hardware that broke.

Two things that follow from it:

- **The angles have to agree.** `am`'s operation mode, `apm`'s performance mode
  and `clkrst`'s rates all describe one console; `GetOperationMode` answered
  Console under a comment saying Handheld, so NX-Fetch printed "Docked" beside
  a 720p handheld framebuffer.
- **`Set`/`Get` pairs are read back**, and a pair that disagrees is the failure
  mode this file keeps rediscovering.

Where a stub would have to invent something unverifiable, it fails instead:
`ssl`'s `CreateConnection` (no socket layer beneath it), `sfdnsres` (definite
`EAI_NONAME`, not a try-again that invites a spin), `acc`'s `LoadIdTokenCache`
(zero bytes, so authentication fails where the missing piece actually is).

**`hwopus` is the exception to all of that**: there is nothing to answer *as*,
because the caller wants audio back. So `src/opus/` is a full Opus decoder —
range coder, CELT, SILK, hybrid, concealment and multi-stream, no
dependencies — and `cpu/hwopus.rs` is the thin service in front of it. The
work buffer the caller allocates as transfer memory is sized and never read;
the decode happens on this side. Just Dance 2023 was the title that reported
`hwopus` missing.

Conformance is checked rather than assumed. `--example opus_testvectors` runs
the RFC 8251 vectors and requires the range coder's final state to match on
every packet of all twelve, decoded both to stereo and to mono; the samples
score 96-100% on `opus_compare`, which is what libopus's own float build
scores against the same references. Throughput is ~350x real time.

## Frontend

- **PFS0 offsets are counted from the end of the string table.** Detecting that
  by looking for an entry pointing inside the header only works when some file
  sits at offset 0; a repack that pads to an alignment boundary has no such
  entry. The base is chosen from the extents instead.
- **An update is paired, not opened, and so is DLC.** Dropping either on the
  container panel registers it against the open title — either order, since
  neither file is read until Launch — and the launch boots the update's
  modules over the base game's RomFS and mounts the add-on content beside it. A `File` reference dies on reload, so the page remembers which update
  a title was last launched with and asks for that file again rather than
  quietly running the base version. Each host file gets its own chunk cache:
  a patched read crosses between the two containers constantly, and one
  shared LRU has them evicting each other at every crossing.
- **Wasm buffers detach on growth**, so a cached view goes stale; staging
  buffers also have to be freed or repeated loads overflow linear memory.
- The frontend is TypeScript on Vite (`web/main/`, `web/worker/`), and
  `make wasm` always builds `--features gpu`, so the worker imports
  wasm-bindgen glue rather than instantiating a bare module.

## Repro

The `.nro`/`.nsp` files are gitignored; `test-nros/` holds the local homebrew.

- `--example screenshot <nro> out.ppm 3` — writes the third presented frame,
  feeding in `web/font.ttf` as the shared font unless told otherwise.
- `--example boot_nx <nro>` — shortest path to "does it halt cleanly".
- `--example boot_nsp <nsp> <prod.keys> [title.keys] [steps]` — the browser's
  Launch button, without a browser. `SHOT=<f.ppm>` writes a frame; prefer it
  over reading `frames presented: 0` off a budget too short to reach one.
  `UPDATE=<update.nsp>` runs the title patched and `DLC=<a.nsp>,<b.nsp>` mounts
  its add-on content, which is the page's pairing of the containers with no
  page in the way.
- `--example dump_exefs …` — flat module images at their real load addresses
  plus a sorted `symbols.txt`. **This is what makes a retail backtrace
  readable.** `--example disasm_flat` disassembles them there.
- `--example retail_trace …` — a ring buffer of the last N instructions, dumped
  on halt or fault. `RING_MIN` past `rtld` (`0x08004000`), whose lazy-binding
  resolver would otherwise fill the whole ring. `MARK`/`MARK_DUMP` watch an API
  being called in order without recording the steps between.
- `--example jit_bench <nro>` — both engines, with every state difference.
  `SWITCH_NO_JIT=1` disables translation for host tools.
- **Checking the WebGPU backend**: run `screenshot_nca` and `switch-gpu`'s
  `screenshot_gpu` over the same frame and `cmp` the PPMs. `GPU_ONLY=<i>` puts
  only the i-th draw on the device, so a difference is exactly one draw's.
- Tracing is environment-gated and **host-only** (`TRACE_IPC`, `TRACE_SVC`,
  `TRACE_WAIT`, `TRACE_NV`, `TRACE_GPU`, and a dozen more): wasm has no WASI,
  so `std::env::var` always fails there. Browser diagnostics go through
  `Cpu::diagnostic`.
- `--example opus_testvectors <dir>` — the Opus decoder against the RFC 8251
  vectors (`opus_testvectors-rfc8251.tar.gz` from opus-codec.org). It fails on
  the first packet whose range coder state disagrees with the encoder's, and
  writes `<name>.rs.dec` beside each vector for `opus_compare` to score.
  `--example opus_difftest <dir>` does the same against a reference decode you
  generate yourself, which is how the output rates below 48 kHz and the
  multi-stream layouts get covered.
- Browser: `make wasm` once, then `bun run dev`. Tests: `make test`.

## Next

1. **Why does a retail title issue no draws?** Two titles, same shape: layer,
   buffer queue, a stream of `nvdrv` ioctls, and zero draw calls over billions
   of instructions. Whatever they wait on is above the GPU — find which thread
   is spinning (`pc=0xa70b7ec` for A Short Hike, `0xd4c36f0` for Tomodachi Life)
   and on what. The Home Menu *does* draw, so this is these titles' own gap,
   not the retail path's.
2. **NX-Shell regressed** to 433,783 steps with no output. Cheapest bisect
   here — the `.nro` is in the tree.
3. **Check Checkpoint's text.** `SHFL`/`FSWZADD` and the quad that runs them
   are implemented and tested, but the title itself has never been run against
   them — its `.nro` is not in the tree.
4. **`usb:hs` and `ncm`**, the last two services. Homebrew also still opens a
   service under an **empty name** (Checkpoint does), and `sm` hands out a
   working handle instead of failing the way real `sm` would.
5. **Known interpreter bug, open**: with a font carrying hinting programs
   (`fpgm`/`prep`/`cvt`), glyphs get correct heights and advances but each
   bitmap is 1-3px wide, as if untouched points never get interpolated. The
   same subset with `--no-hinting` renders perfectly. Invisible in normal use —
   the shipped font has no hinting — but a real correctness gap.

Lower priority: hbmenu's entry label renders as a blank box; NAND-vs-SD storage
is one hardcoded 32 GiB for both free and total; Checkpoint never presents a
frame.
