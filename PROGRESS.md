# switch-wasm: boot status

Goal: get real homebrew and real retail titles to **run**, and put what they
render on the canvas. This is the log of what broke and what it taught; AGENTS.md
is the standing state and carries every rule that came out of it.

## Where it stands

`.nro`/`.nsp` files are gitignored, so only `test-nros/` re-runs from a clean
checkout. "measured" = run against this tree; the rest is carried forward.

| Title | Result | Frame | |
|---|---|---|---|
| `hbmenu.nro` | full UI, responds to a controller | yes | measured |
| `sdl-hello.nro` | exits cleanly at 11.1M steps, 2061 non-black pixels | yes | measured |
| `NX-Shell.nro` | halts at 433,783 steps, no output - **regressed** | no | measured |
| Home Menu (`qlaunch`) | draws; 88 draws a frame on the WebGPU backend | yes | carried |
| `sysinfo` / `NX-Fetch` / `nxdumptool` | render | yes | carried |
| `JKSV.nro` | full UI: text, icons, save tiles | yes | carried |
| `Checkpoint.nro` | layout and chrome; text unchecked since `SHFL` landed | yes | carried |
| "A Short Hike" (NSP) | composites 1280x720; steady state 2 draws, never a scene | partial | measured |
| "Minecraft" (NSP) | the world, on the device: 110 draws a frame, no fallbacks | yes | measured |
| "Tomodachi Life" (NSP) | its loading screen, every pixel, at 3.98B steps | yes | measured |
| "Echoes of Wisdom" (NSP) | the prologue scene at frame 150, both renderers within 3/255 | partial | measured |
| "Mario Kart 8 Deluxe" 1.0 (NSP, A32) | boots to 4.4B steps, then a virtual call through a corrupt vtable | no | measured |

A retail title decrypts, mounts its RomFS, runs `rtld` → `main` → `subsdk*` →
`sdk` through real `nnSdk` init, gets its heap, events and input, brings up its
graphics stack, opens its audio device, and runs on into its own loop. **Every
service it asks for has a real implementation**: a full boot logs no `no
implementation` and no `unimplemented` lines. `make test`: **1,124 tests passing**.

## Method, which is the part that generalises

- **Silence is not evidence.** `/dev/nvhost-ctrl-gpu` `0x13` (`VsmsMapping`) was
  the only line a whole retail run logged, which made it look causal. Answering
  it with any scalar, refusing it, and implementing it properly all give
  byte-identical GPU statistics and the same spinning pc. The emulator only logs
  the gaps it knows it has.
- **A step counter that keeps climbing is not a title that keeps running.** Just
  Dance 2023 reached seven billion steps with the main thread retiring a
  *constant* 760M at every budget: everything after was two threads re-asking
  `svcWaitSynchronization` on each slice. Profiling at four budgets is what
  showed it; the fix (parking waits) took the Home Menu's tenth frame from 170.6M
  steps to 39.2M, byte-identical.
- **A frozen title is often a configuration difference, not a bug.** Host
  examples boot handheld and the browser is usually docked, so a title told 720p
  with a 1080p swapchain composites into a corner that reads exactly like a
  rendering fault. `DOCKED=1` in `Title::boot` settles it.
- **Prove a new decode guard reaches a real encoding**, and reach for
  `tools/difftest.py` before hand-deriving an expected value. About thirty decode
  bugs came out of the three homebrew titles and nearly all were one shape: a
  guard that tested a field including a fixed bit, so a whole encoding group was
  dead code. The rest were sign-extension widths.
- **The file is the oracle, the spec is not.** AES-CTR with the wrong key still
  "decrypts" into plausible garbage, so decryption is verified by matching a
  Program NCA's ExeFS against Nintendo's own stored SHA-256.
- **Ask where a draw's fragments died before blaming the pipeline.** `TRACE_DRAW`
  tallies culled/degenerate/uncovered/killed/written per draw, and `culled=2` of
  `tris=2` on exactly the blits is what found the winding bug below. The `[gpu]
  draw` line names the render target's cpu address beside the cull state, which
  is what tells a title compositing offscreen from a title whose composite was
  dropped: both are a black frame otherwise.
- **Getting the backtrace is what makes a fault findable.** `dump_exefs` lays the
  modules out at their real load addresses and writes a sorted `symbols.txt` from
  `sdk`'s 36,622 `DT_HASH` symbols; `0x0ce6c0c8` says nothing,
  `sdk!nn::diag::detail::Abort+0x18` says everything.
- **A title that quits by itself is a bug until proven otherwise, and one the
  translator alone exposes may still not be the translator's.** Just Dance 2019
  halted at step 1,097,299,725 with the JIT on and never with it off. The trail
  showed a freshly created HTTP thread returning through a zeroed return
  address, and the `ret`-to-0 special case turning that into `ExitProcess`.
  Polling the slot every 16 instructions caught the writer: a *different*
  thread, joining the old thread whose `ThreadType` the new one reused.
  `svcWaitSynchronization` treated a thread handle as always ready, so the join
  returned while the thread ran. Whether the joiner got the CPU in that window
  depended only on where the scheduler switched, and the translator switches at
  block boundaries rather than at every instruction. A thread handle is now
  signalled when its thread exits. The lockstep tool that found the divergence
  point (`jit_bisect`) cannot follow a multithreaded title past its first
  thread switch, because the two engines legitimately interleave differently;
  the slot poll is what reached the bug.
- **Nothing in the tree looked slow.** A `perf` profile of a Home Menu boot came
  back 37% `getenv` and 18.7% SipHash, together more than the shader
  interpreter, the rasterizer and the ARM interpreter combined. 73.6 s → 27.7 s,
  every frame byte-identical.

## Retail boot: what each fault was

- **`rtld` has no MOD0 header.** `NSO_ENTRY_OFFSET` (`.text`+0x30) is only right
  for modules that have one; `rtld`'s `.text`+0 is real code that establishes its
  own load address. Jumping past it left the base at 0 and a `bss_end - base`
  zero-fill overwrote the loop running it. Cost 25M instructions to find.
- **`rtld` finds its own modules** with `svcQueryMemory`, looking for
  `CodeStatic` + `R-X`. A blanket RWX permission and single-range read-only
  tracking both broke that.
- **The `sdk` abort was a recursive-lock assertion on an untouched mutex**, and
  three real bugs fell out of naming the backtrace: the main thread handle was
  never delivered (it is X1), `svcGetInfo` CoreMask/PriorityMask fell into a
  `_ => 0` default, and `svcWaitSynchronization` answered X1 = 1 where X1 is the
  signalling handle's *index*.
- **Success with an unfilled out parameter, repeatedly.** `GetFirmwareVersion`
  never wrote its struct, so NX-Fetch displayed "Horizon OS 115.119.105", the
  ASCII of `swi`, left in the buffer by an earlier `acc` call, and load-bearing
  because libnx seeds `hosversionGet()` from it. `CloneCurrentObject` returned no
  session handle. `IStorage::Read` used `IFile::Read`'s field layout.
- **A stub that answers every unknown command with success is worse than one that
  fails.** `SetupGpuErrorHandler` asked for an event, got "success" and no
  handle, and filed handle 0.
- **One field width cost a whole title.** `OpenAudioOut`'s channel count is 16
  bits on the wire and the two bytes above are uninitialised padding; echoing the
  whole word back reported 0xcafe0002 channels, so Unity tore audio down without
  `CloseAudioOut` and the retry hit a registry that still held the device open.
- **The address-space split, not the allocator.** Tomodachi Life started a thread
  on a null 800M instructions in because `svcGetInfo` reported 1.5 GiB total and
  the title spent it on pools it sizes *from that same figure*, while the other
  1.5 GiB was an alias region it never touches. Each layout now spends the space
  on the region its own titles grow into.
- **A store-release run as a store-exclusive.** ARMv8 added `LDA`/`STL{,B,H}`
  to A32 beside `LDREX`/`STREX`, differing only in bits 9:8, which the decoder
  ignored. With no monitor open, Mario Kart 8 Deluxe's `STLH` dropped its store
  (and wrote its status into `Rd`, which it fills with 1111: `pc`), so the SDK's
  cached pointer-buffer size read back as 0 and the first pointer buffer it sent
  was refused: the same 11-141 as below, from an entirely different cause. The
  A32 media group is now complete and checked against `qemu-arm`
  (`tools/a32_media_reference.py`).
- **A pointer buffer a caller is told it cannot use.** Every session answered
  `QueryPointerBufferSize` with 0, and `nnSdk` measures an explicit
  `SfBufferAttr_HipcPointer` argument against it before sending:
  `PointerBufferTooSmall`, nothing sent. That was Tomodachi Life's `sf` 11-141
  abort at 2,524,316,651 steps. Answering 0x8000 then exposed the second half:
  `cmifRequestInAutoBuffer` fills in both descriptor forms and nulls the one it
  did not choose, so nvdrv's AutoSelect ioctl argument came through as a null
  map-alias buffer and `nvn` gave up on graphics at 33M steps. `ipc_pick_buffer`
  is the rule; `ipc_map_buffers` is gone.
- **An `nn::hid` state is read under a seqlock and bit 0 is the lock.** A storage
  entry's sampling number is the state's own **doubled**. Publishing it undoubled
  made the first sample odd and Tomodachi Life's main thread sat in that retry
  loop for 20 billion instructions with every counter byte-identical. `libnx`
  never looks at the bit, so no homebrew could have shown it.
- **`hid` samples on a clock, not on input.** Publishing only on host input froze
  the sampling number, and a title waiting for a newer sample waited forever.
  That was also why the CLI could not reproduce what the browser saw: a hand on
  the keyboard kept the LIFO moving.
- **Every thread said it was on core 0.** Tomodachi Life creates its threads on
  core 0 and moves each with `SetThreadCoreMask(-1, one-core mask)`, which was a
  stub, and `GetCurrentProcessorNumber` answered 0 regardless. Two render
  workers meant for cores 1 and 2 then ran the same job into one command list,
  and at 15.57G steps, just after the first save, a record half-written by each
  called a vtable slot that is null in a pool's base class: pc=0. What found it,
  in order: the object's vtable (`DUMP=*x25`), that the vptr was written once at
  boot and never again (`TRAP_WRITE` + `TRAP_LAST`), then that two threads wrote
  the record with identical arguments. The exclusive monitor and the loader were
  both suspects and both innocent.
- **Every thread had id 1.** `svcGetThreadId` answered 1 whatever it was
  asked, so to a title comparing ids every thread was the same thread. The
  Legend of Zelda: Echoes of Wisdom's actor factory checks whether it is on one
  of the threads it keeps a list of, and took the main thread for another: it
  initialised Link inside a creation that holds a global `std::mutex`, Link's
  initialisation created his sword through the same path, and the SDK aborted
  (2162-0001) on the second lock at 1.37G steps. The trail, for the next one:
  the abort was `pthread_mutex_lock` refusing a self-deadlock; `TRAP_WRITE`
  showed the lock taken once and never released; `WATCH_DUMP` read the two
  registry names (`PrologueLink`, `PlayerItemSword`); and resolving the
  factory's PLT imports found `nn::os::GetThreadId` beside the branch.
- **A `Poll` that returns instantly is not one that reports nothing ready.**
  NXpotify's Zeroconf listener is `if (poll(&pfd, 1, 200) <= 0) continue;`, with
  no blocking syscall it starved every other thread.

## Containers

- **A cartridge image is the same container one layer down.** An XCI's root is an
  HFS0 whose entries are the cartridge's partitions, each an HFS0 holding NCAs;
  HFS0 is PFS0 with a 0x40-byte entry. So the reader is `Pfs0`'s given an offset
  and an entry stride, and `Xci::content` flattens the partitions into the one
  file table every reader above already takes. Two rules the flattening keeps:
  `update` is left out (it is a firmware bundle with Program content among it,
  and every search that follows is "the first/last NCA of type X"), and `secure`
  goes last, because the Program scan keeps the last match. Verified by repacking
  "A Short Hike" into an XCI and booting it to the same frame 3.
- **Four decryption bugs, each flipping the hash from mismatch to match**: the
  section table is `u32 start; u32 end` in 0x200-byte media units, not a `u64`
  pair; the AES-CTR counter runs across the section's **absolute** position; a
  ticket's `common_key_id` needs the same +1 generation adjustment the key area
  gets; an IVFC section's byte 0 is the hash table, and the real data is at the
  last level's `logical_offset` (hactool reads a fixed index 5 regardless of
  `num_levels`, which reads 7 on a real file whose level array holds 6).
- **An update NSP holds no game.** Its Program NCA carries a complete ExeFS:
  patched modules, not a delta, and a RomFS section encrypted `AesCtrEx`: the
  BKTR form, holding only the changed ranges plus the two tables indexing them
  against the base. `bktr.rs` composes the pair, streaming both containers. The
  subsection counter replaces the section counter's **generation** word, not its
  secure value: the wrong way round is quiet, because the tables still decrypt
  and every byte of *data* is noise. An update's Program NCA carries the **base**
  title id, so pairing is by program id and what identifies a container as an
  update is that its RomFS is a patch.
- **DLC is not an update, and is much less than one**: one Data NCA with an
  ordinary RomFS whose title id is the base's plus an index, mounted through
  `OpenDataStorageByDataId`, the same path a system data archive takes, so the
  reading half was already built. What was missing was `aoc:u` saying the content
  exists, since a title never asks for content the list does not have.

## GPU

The nvdrv/nvmap/GMMU/channel/copy-engine path is real, as is the 3D shader core:
a Maxwell SASS interpreter feeding a software rasterizer, with compute
dispatches on the same interpreter one thread at a time. `switch-gpu` is a
separate `wgpu` backend translating SASS to WGSL; the rasterizer is the reference
it must agree with.

**One transport bug stopped all of device init.** An ioctl whose argument carries
a `{ buf_size, buf_addr }` pair returns its payload *through* that pair, and the
IPC layer wrote back only the first receive buffer. `libnx` reads the payload
inline and worked; `nnSdk` uses `nvIoctl3` and read a zeroed GPU characteristics
struct, so the driver closed the device and returned null.

**An ioctl that holds nothing is worse than one that fails.**
`ZbcSetTable`/`ZbcQueryTable` had to keep what they are given rather than answer
a bare success: a driver told "nothing is registered, ever" re-registers
forever. Same for `EventSignal`/`EventWaitAsync`, where the driver parks on a
slot nothing would set.

**Four `texs` bugs, each masking the next**, which is why the symptom never
looked like four things:

- A `texs` has **two destination registers, not one run of four**, invisible
  whenever `dst2 == dst + 2`, exactly what the first fixture did. JKSV's glyph
  shader clobbered the `1/w` every later `ipa` multiplies by, so every glyph came
  out alpha-zero: text present and completely invisible.
- **The handle immediate is a dword index into the driver constant bank, not a
  byte offset.** Reading it as bytes landed in the header ahead of the table,
  which begins `0, 1, 2, 3…`, a plausible handle table, so every draw resolved
  to a plausible handle, and every draw resolved to the same one.
- **`TexCbIndex` is a register, not a constant**: Mesa writes 15, deko3d 0.
- **Whether the viewport flips y is a register, not a constant.**

**A Unity title is written in `half`, and none of it decoded.** "A Short Hike"
renders into an RGBA16Float target and composites with two full-screen quads;
both were dropped, one on sampling a float texture and one on the fp16 ALU, so
nothing wrote the swapchain and it presented the zeros it was allocated with:
transparent, not black. Two things worth carrying: **110 of the 145 dropped draws
were `hadd2.f32`**, a plain float add issued on the half unit, so most of what a
`half` shader costs is not half arithmetic; and **`f32_to_f16` had to stop
truncating**, as the rounding step of every fp16 instruction, round-to-zero
biases a whole shader. It now rounds to nearest, ties to even, reaching the
subnormals, which is the mode WGSL's `pack2x16float` uses.

**Unity culls on the CPU**, so a title whose scene or camera is wrong emits *no
draw calls at all* rather than draws that cover nothing. A Short Hike's steady
state is two draws a frame forever, frames 30 and 300 byte-identical to frame 1,
47% of instructions in the Boehm GC's mark loop: IL2CPP working normally, simply
producing no renderers. Zero scene draws is the title's own state; chasing it
through the GPU is chasing the wrong end.

**Minecraft: three gaps, each hiding the next.** Every draw read a `4x16`
half-float attribute neither backend could fetch (110 draws, 110 skipped, 0
pixels lit, and in a browser the first fallback latches `software_frame`, which
is where the frame time went). Then every fragment shader opened with
`ipa.centroid`, and `Op::Unimplemented` is fatal to the interpreter as well as
untranslatable. Then the frame came out upside down: not the viewport, but
`QUEUE_BUFFER` throwing away a `QueueBufferInput` that said `FLIP_V`.

**Minecraft's fourth gap was one refused datagram**, and it fired 29.6 billion
instructions in, long after the title is drawing. The run jumps to address 0,
in the browser and on the CLI alike, at exactly step 29,593,165,824. The vtable
the faulting method belongs to names the class (`19RakNetServerLocator`, out of
its typeinfo): it is building the LAN discovery ping list, and calls
`GetNumberOfAddresses` through a `RakPeerInterface*` that is null. It is null
because the setup function *nulls it deliberately*: `if (Startup(...) != 0) {
Shutdown(); destroy(peer); m_peer = nullptr; }`, and the caller dereferences it
without checking. Startup failed because RakNet's `BindShared` sends a test
datagram to the address it just bound and reads a failed send as
`BR_FAILED_SEND_TEST`. The last seven service calls before the fault are the
whole story: Socket, SetSockOpt x4, Bind, GetSockName, SendTo (4-byte payload,
16-byte `sockaddr_in`), Close.

**A link that is up and a datagram that cannot leave are not the same console**,
and `bsd` was claiming both: `nifm` reports 192.168.1.100 while `sendto`
answered `ENETUNREACH`, which is what an interface that is *down* reports. An
addressed `SendTo` now reports the byte count and drops the bytes; unaddressed
datagrams and unconnected stream sockets fail as they did.

**The crash was the smaller half of it.** The title was not merely dying at
29.6B steps, it was *stuck* long before: 6 frames and 12 draws, unchanged from
4 billion steps to 29 billion. With the send answered, the same 40-billion-step
run presents **504 frames and 9,720 draws** across 1,515 submissions, and gets
as far as `nsd:u` cmd 21 and raising its own error applet, which is what a
console with no route off the LAN should make it do. So the retry around the
failed `Startup` was eating the run, and "Minecraft is CPU-bound: 21.9 billion
instructions buys 20 frames" below was measured through it.

**The same contradiction had two more halves**, neither of which any title had
reached yet, both now closed. A `recvfrom` on an unconnected datagram socket
answered `ENETUNREACH`, a broken link, where a bound socket on an idle one
reports that nothing has arrived *yet*: `EAGAIN` and a reschedule, the answer a
live connection with an empty queue already gave. And `select`/`poll` never
called a datagram socket writable, because writability was "has a peer", so a
caller that waits for the socket to be ready before sending would never send at
all. A datagram socket needs no peer to send; it is always writable. Only a
stream socket with no connection still answers `ENOTCONN` to a read, which is
what hardware does.

**The fault the emulator reported was not the fault the guest took.** A call
through a null vtable read `[0]`, `[0x1f8]` and then branched to 0, and every
one of those succeeded: `Cpu::bootstrap` soft-maps the whole space, so unwritten
pages read as zeros, the instruction fetch at 0 included. What is reported is
`unimplemented instruction 0x00000000 at pc=0x0`, and only the last-64
instruction trail says it was a null dispatch. A fetch from a page nothing has
ever written is always a wild jump; a fault at the branch would name it.

**`SHFL` took a quad to run**, and three things came out of it: a shuffle is only
half of a derivative (`FSWZADD` subtracts in whichever direction the lane's
position calls for, so decoding the shuffle alone leaves every such shader
failing on the *next* instruction); both operands are a register or an immediate
and the immediate sits in a *different field* from the register, so reading the
wrong field gives a plausible lane number rather than an error; and helper lanes
are the point, not an artefact, which is why the quad walk is gated on the
program containing a `shfl` or `fswzadd`.

**Which winding is front is decided in window space, after the viewport.**
Facing was first read off screen-space area as if y pointed up, which inverted
culling for every title whose driver flips y, every one built against nnSdk:
Tomodachi Life's full-screen composite was thrown away and the frame was black.
The fix read it off the NDC winding instead, and that was a second misreading
that happened to agree: Eden decides facing in framebuffer space after the
viewport, reversed by `SetWindowOrigin`'s FlipY and by nothing else, and the NDC
rule only matches that through a viewport that mirrors y. Echoes of Wisdom's
offscreen post-processing runs through one that does not, with front=CCW, FlipY
and back-face culling, so every full-screen quad of it was culled and its frame
stayed black on both renderers. **When a rule is confirmed by one title, check
it against the case that title does not exercise.** Both renderers now carry the
guest's front face over unchanged and judge the winding the target holds.

Smaller ones: **`AntiAliasEnable` does not size a surface; `MsaaMode` does** (a
2560x720 `2x1_D3D` target read as 2560 pixels wide, and the title's own resolve
shrank the frame to a quarter). **The colour write mask was unimplemented**:
A Short Hike writes `SetCtWrite` (0x680, a nibble per
channel) and `ColorMaskCommon` (0x3E4) 420 times a frame and turns alpha off for
99 draws, and an unwritten mask must read as *all* channels, since zero is the
register file's initial value. **`VOTE.VTG` writes neither register nor
predicate**, but a refused instruction fails the whole draw, all 52 of them. A
*fixed* vertex attribute with no buffer behind it must read the `vec4` default
rather than dropping the draw; a *disabled* one is still an error, because that
means a register was read wrong. **`SetDstWidth` counts elements, not bytes.**

**Where a frame's time goes** (ablation on 30 JKSV frames): fragment shader
interpretation 36%, rasterize/depth/blend/pixel I/O 30%, ARM interpreter 25%,
texture sampling 9%. A full-screen pass is 921,600 fragment invocations, so
NXpotify's 2.6 s/frame was a texture result rescanning the decoded program and
building a `Vec` per instruction, about a hundred heap allocations per pixel.
Where a result lands is a property of the *program*, not the invocation; that
plus two allocation fixes took it to 0.67 s/frame, byte-identical.

## JIT: emitted wasm, and what it took to reach it

The translator writes a hot block out as wasm and the browser runs it. A block
that survives 512 entries (`HOT`) is emitted, compiled through
`WebAssembly.Module`, and its `run` export is put in **this module's own
function table**; the core then calls the slot. A function pointer on wasm32
*is* a table index, so entering a compiled block is one `call_indirect`. The
shape that suggests itself instead (an import the core calls emitted code
through) would put a JS frame on a boundary crossed every six guest
instructions, which costs more than the dispatch the emitter exists to remove.

**`--growable-table` is load-bearing and silent.** LLD emits the function table
with its maximum equal to its minimum, so `table.grow` fails, every install
answers "could not", and the translator interprets everything it translated
while looking, from the outside, exactly like a build with no emitter at all.
Nothing failed, nothing warned; the counters said 1,001 block entries and 0
compiled. It is in `.cargo/config.toml` now, and `tools/jit_wasm_check.mjs`
asserts that blocks are not only compiled but **entered**, because the register
comparison alone passes happily on a build that compiled nothing.

The wiring is checked from both ends, because neither end can check the other:

- `tests/jit_emitted_test.rs` drives the state machine on the host with a Rust
  function standing in for the compiler: an entry point is a code address in
  whatever sense the target has one, so that is the real interface and not a
  mock of it. It covers when a block is emitted, the clock and the step budget
  across a block that stops early, and that a dropped or invalidated block
  gives its slot back.
- `tools/jit_wasm_check.mjs` runs the artefact `make wasm` produces under V8.
  That is the only place the target-specific parts exist: the field offsets an
  emitted block reaches guest state by are wasm32's, and a page-table entry is
  four bytes there against eight on the host. Perturbing one offset by eight
  makes it name the exact registers that diverged.

**Guest state is reached by `offset_of!`, never by a written-down number.**
Neither `Cpu` nor `Memory` is `repr(C)`, so the only offsets that are right are
the ones the build chose, and the emitter runs in that build. A test walks them
from the address of a `Cpu` the way emitted code does: register file, NZCV,
both watchpoints, the write-protected envelope, the watched-page bitmap, and
the page-table walk down to the byte.

What the emitter cannot yet write is control flow, so only a block with no
conditional branch in it is emitted at all. Everything else is unchanged and
still interpreted. The infrastructure is what was missing; writing branches now
pays immediately, and batching several blocks into one module is the next
compile-time win (7,062 blocks compile in 1.8 ms when batched).

**A module per block has two costs that grow with the number of modules**, and
both made the emitter a net loss on Just Dance 2019 until they were found. At
`HOT = 16` a run emitted 7,990 blocks:

- **Every `memory.grow` walks every instance.** Each emitted module imports the
  core's memory, and V8 re-points each one's cached size on a grow
  (`SetInstanceMemory`). std's allocator grows 64 KiB at a time and guest RAM is
  backed a 4 KiB page at a time, so an asset-loading burst was ~500 grows, and
  half the main thread for 0.4 s. `switch-wasm` now allocates with `dlmalloc` at a
  4 MiB granularity (`src/heap.rs`): host cycles over a 40-frame run -41%, host
  instructions -25%, frame mean -7%, guest work identical.
- **Scattered code stalls.** With growth fixed, the emitter was still 4% slower
  per steady frame than interpreting everything, with host *instructions* level
  and *cycles* up: thousands of modules, each in a code region of its own,
  entered through one indirect call. Emitted code itself is fast; a
  microbenchmark with one or two blocks shows it at parity or better. At
  `HOT = 512` the run emits 1,185 blocks, which take 97% of the entries the 7,990
  did, and the emitter beats the interpreter (-0.8% instructions on every pair,
  -1.6% cycles). 4096 (461 blocks) measured the same as 512.

**Per-entry overhead dwarfs what emitting saves.** A four-instruction guest loop
costs 20-35 ns per block entry under V8 whether or not its body is emitted, so on
JD's 2M entries a frame the block-to-block machinery in `run_jit`, not the body,
is where an emitter has to win: blocks that chain to each other inside emitted
code, rather than returning to the interpreter between every one.

## Next

1. **The rasterizer renders Minecraft black, and it is the reference.** The
   device draws the title correctly, but `screenshot_nsp` over the same frame
   reports 110 draws, `draws_skipped: 0` and 0 of 921,600 pixels lit, at frame 3
   and frame 20 alike. Every other path is checked by agreeing with the
   rasterizer, so this is a hole in the check itself. Bisect with `GPU_ONLY`. The
   winding fix did not clear it, though frame 3 is 6 draws and not the 110-draw
   one, so frame 20 is the measurement that would settle it.
2. **A retail title that draws but never a scene.** Tomodachi Life presents at
   3.98B steps with 7 draws; A Short Hike no longer reaches a frame at all, at
   HEAD it spins after one submission and 3,536 methods, and past the
   pointer-buffer fix it takes a `write to read-only address 0x0aa28f50` at
   `pc=0xa70b814`, step 404,553,728. Whatever they wait on is above the GPU.
3. **`CreateAliasStackUnsafe` is an open abort.** Just Dance 2023 maps 38 thread
   stacks and aborts on the 39th, which never reaches a syscall, so it is inside
   `nn::os::detail::AslrSpaceAllocator`'s own bookkeeping, built from `svcGetInfo`
   12/13 with the heap and alias regions outside the ASLR region it is told
   about. Shrinking the stack region *looks* like a cure and is luck: the region
   size is the modulus nnSdk's random placement uses, so a different size is a
   different address sequence. Booted without its update the title deadlocks for
   real at 765M steps; patched with 1.0.1 it presents 76 frames and dies
   elsewhere.
4. **NX-Shell regressed** to 433,783 steps with no output, from a recorded clean
   `ExitProcess` at 15,692,155 steps *with* output. The cheapest bisect here:
   the `.nro` is in `test-nros/`.
5. **Check Checkpoint's text.** `SHFL`/`FSWZADD` and the quad are implemented and
   tested but the title has never been run against them; its `.nro` is not in the
   tree.
6. **`usb:hs` and `ncm`**, the last two services. Homebrew also still opens a
   service under an **empty name** (Checkpoint does), and `sm` hands out a working
   handle instead of failing the way real `sm` would.
7. **Known interpreter bug**: with a font carrying hinting programs
   (`fpgm`/`prep`/`cvt`), glyphs get correct heights and advances but each bitmap
   is 1-3px wide, as if untouched points never get interpolated. The same subset
   with `--no-hinting` renders perfectly. Invisible in normal use (the shipped
   font has no hinting) but a real correctness gap.
8. **Minecraft is CPU-bound**, not GPU-bound: 21.9 billion instructions buys 20
   frames, and a steady frame is 173-220 ms with every draw on the device.

Lower priority: hbmenu's entry label renders as a blank box; NAND-vs-SD storage
is one hardcoded 32 GiB for both free and total; Checkpoint never presents a
frame; the applet path's queue transform offset is unverified.


## 2026-09-09: JIT validation and WASM performance baseline

The current working tree's expanded two-source JIT decode included reserved
opcodes 0x0c–0x0f, which reached an `unreachable!()` in its helper, and accepted
CRC operand widths the interpreter rejects. Restrict those forms to the
interpreter's error path. A differential regression covers both widths and
compares errors and guest state. All switch-core tests and core clippy pass.

With the existing in-progress JIT optimizations, the shipped GPU-feature WASM
build running `node tools/wasm_bench.mjs web/assets/hbmenu.nro --frames=8`
under Node v26.8.1 measured 242.5 M instructions/s, 111.4 ms/frame (8.98 fps),
27,000,000 instructions/frame, and zero interpreted fallbacks. This is one
baseline run, not an isolated speedup measurement, and is below the 1B/s goal.
Next measure hot WASM functions: extending opcode coverage cannot improve this
particular workload's zero-fallback execution.

The installed wasm-bindgen CLI was 0.2.128 while Cargo.lock requires 0.2.127.
The matching CLI is installed at
`/private/tmp/switch-wasm-bindgen-0.2.127/bin/wasm-bindgen`; use that directory
at the front of PATH for `make wasm` without changing the project's lockfile.


### WASM block-execution profile and inlining

A 40-frame Node V8 CPU profile of the browser's WASM artifact put 3,478 ms
in `exec_block`, versus 329 ms in `switch_run` and 250 ms in `Gpu::present`.
Force-inlining `exec_block` into the run loop gave 253.7 and 247.0 M/s in
40-frame runs; alternating the saved previous WASM artifact gave 238.9 and
231.3 M/s. These are noisy local measurements suggesting a modest gain, not
proof of browser end-to-end performance or the 1B/s target. Both versions
retired identical work (26,902,439 instructions/frame), with identical JIT
counts. All 15 differential JIT tests pass; the site builds and typechecks.

`wasm_bench.mjs --shot=<file.ppm>` now captures the final framebuffer outside
the timed window using the browser worker's snapshot export. Frame 43 from
the two artifacts was byte-identical (`cmp`). The shipped site in `dist/`
has been rebuilt with the inlining change. The goal remains a browser Switch
emulator; Node measurements guide optimization but do not substitute for
final in-browser verification.


### Rejected bookkeeping and arithmetic micro-optimizations

Combining an ordinary branch's retirement with its block body preserved
frame 43 byte-for-byte and passed all 15 JIT tests, but measured 261.4 M/s
versus 263.4 M/s for the saved baseline. Removed the experiment.

Explicitly skipping carry/overflow work in non-flag-setting ADD/SUB also
preserved the frame and passed JIT tests/core clippy. Three alternating
40-frame WASM benchmark pairs measured baseline 266.7/267.7/257.6 M/s and
candidate 268.4/261.6/263.3 M/s. Their means differ by under 0.2%; removed
this experiment too. These results also show why separate-session headline
numbers are insufficient evidence of an improvement.

The guest instruction census (`hotspots`, used for counts only) puts 26.04%
in loads/stores, 23.35% in register ALU, 22.95% in immediate ALU, 18.34% in
branches/system, and 9.33% in SIMD/FP. Guest pages 0x08005000 and 0x08004000
account for 71.65% of hbmenu's frame. Next investigate reducing dispatches
for common instruction sequences or generating code, instead of more
bookkeeping micro-optimizations. The 1B/s browser goal is still unverified.


### Register-copy specialization and instruction-level census

`hotspots --instructions` now lists individual PCs, counts, and disassembly.
The top hbmenu loop includes adjacent MOV x8,x25 / MOV x6,x24, each 360,192
times per frame. The JIT now uses Mov32/Mov64 for ORR-with-zero copy aliases,
removing the second register read and general logical-op selection. The
interpreter uses the same copy helper. Width, register overlap, and zero-register
cases are covered by a new differential test with direct copy-value checks.

Three alternating 40-frame V8 pairs measured baseline 268.9/268.9/269.7 M/s
and specialized copies 270.6/269.2/272.3 M/s (means 269.2 vs 270.7 M/s).
This is a small ~0.6% local improvement, not evidence of a broad browser
speedup. Final frame 43 is byte-identical. Core clippy, differential tests,
site build and TypeScript checks pass. The remaining gap calls for reducing
per-instruction dispatch, rather than expecting operand specializations alone
to provide the nearly fourfold improvement still needed for 1B/s.


### Fused update/compare/branch

Extended compare fusion to recognize an immediate register update followed by
CMP/CMN and B.cond. The translated exit covers three guest instructions in one
execution dispatch. Partial budgets (after each prefix), taken/not-taken paths,
and resumed runs are covered by differential tests; a decoder unit test confirms
one three-instruction exit. The current WASM run measured 273.2 M instructions/s
(98.5 ms/frame) versus the saved pre-fusion artifact at 263.3 M/s (102.2
ms/frame), a 3.8% improvement in this paired run. Frame 43 was byte-identical.
The measurement is promising but one pair is not a stable browser benchmark; the
1B/s objective remains open.


### Three-instruction loop fusion

The translator now fuses an immediate register update, CMP/CMN, and B.cond
into one `UpdateCmpImm` exit. This targets the dominant hbmenu loop shape and
removes two intermediate dispatches while retaining exact partial-budget behavior.
Differential coverage exercises add/sub updates, both branch outcomes, budgets
ending after each instruction, and resumed execution. Full switch-core tests
(770 library tests plus examples), core Clippy, and the site build/typecheck pass.

A 40-frame comparison was 273.2 M/s versus 263.3 M/s for the saved artifact,
and frame 43 was byte-identical. Three 32-frame alternating pairs were noisy:
one old run hit 84.4 M/s under host contention, with other old runs 214.9/219.4
and new runs 200.8/195.2/245.2 M/s. Therefore no stable speedup claim is made
from this sample, though the dispatch reduction is structurally aligned with the
hot instruction census. The browser artifact remains far below the 1B/s goal.


### Scalar load/store specialization (rejected)

Tried separate JIT variants for common 32/64-bit immediate and register loads
and stores. The larger `Op` match outweighed the removed `Acc` match: three
alternating 32-frame WASM runs measured old 271.6/271.0/271.9 M/s versus new
255.8/269.4/267.6 M/s. Removed the variants. Full core tests, Clippy, WASM
release build, and Vite site build remain green.


### Rejected FP PC-write removal

Considered omitting the `self.pc = pc` write around JIT FP/SIMD execution.
SIMD handlers construct errors using `self.pc`, so removing it would make fault
diagnostics report the previous instruction. The optimization was rejected to
preserve architectural diagnostics. The final browser artifact was rebuilt after
removing this experiment; tests and Clippy remain green.


### Unaligned memory read fast path (rejected)

Tried replacing same-page read arrays with `read_unaligned` for 32/64-bit
loads. Three alternating 32-frame WASM pairs were effectively neutral
(275.1–275.4 M/s candidate versus 275.3–275.5 M/s baseline), with one slight
regression. Removed the unsafe code. Final site rebuilt; JIT tests pass.


### Register accessor specialization

The padded 64-slot register file now uses debug-asserted unchecked indexing for
its internal `reg_at`, `set_reg_at`, `read_x`, `write_x`, and zero-register write
accessors. This removes the per-access mask and bounds-check scaffolding while
keeping invalid slot detection in debug builds. The slot constructors still map
all architectural meanings of register 31 to valid slots. Full switch-core and
switch-wasm tests, Clippy, and TypeScript checks pass.

Three alternating 32-frame WASM pairs measured old 271.0/269.9/264.9 M/s and
new 265.9/272.2/274.3 M/s. The means suggest a small ~0.8% gain but overlap
with noise. A 24-frame run measured 266.3 versus 261.0 M/s and framebuffer
output was byte-identical. The final Vite site was rebuilt. This is a measured
browser-artifact improvement candidate, not completion of the 1B/s target.


### SIMD register-file access

Wrapped the 32-entry Q-register array in an inline unchecked indexer. Decoder
register fields are five bits, and the public accessors preserve that invariant;
debug builds still assert before the unchecked access. The wrapper retains copy,
thread-context save/restore, and iterator behavior, so it does not change the
guest-visible register layout. A three-run 24-frame comparison against the
pre-wrapper WASM artifact was effectively neutral (new 267.6 M/s, old 267.3
M/s), while a paired screenshot was byte-identical. Core tests and the release
WASM build pass; this is retained as a low-risk bounds-check removal, with the
browser still well below 1B instructions/s.


### Zero-register read accessor (rejected)

Tried unchecked indexing after masking the five-bit XZR operand. Three
alternating WASM pairs were neutral to slightly slower (275.3–276.1 versus
275.1–278.6 M instructions/s), so the unsafe accessor was removed.


### Chained-block link access (rejected)

Tried replacing the single-threaded JIT block's `RefCell` successor link with
an `UnsafeCell` to remove its dynamic borrow. Frame output stayed identical,
but measurements varied from a small gain to a 2–3% slowdown (the latest
264.7 M/s versus 271.2 M/s over 16 frames), and the unsafe aliasing provided no
reliable benefit. Restored the `RefCell` implementation.


### Conditional branch condition lookup

Replaced repeated NZCV bit extraction and a 16-way condition-code match with a
compact sixteen-entry mask table indexed by the guest's NZCV nibble. This is
used by both interpreter conditionals and JIT exits and preserves AL/NV and all
signed/unsigned condition relationships. The complete core suite passes. WASM
measurements remain noisy (roughly 252–274 M/s across alternating 32/128-frame
runs), so no stable percentage is claimed; the change is retained pending
longer uncontended browser measurements, and the 1B/s goal remains open.


### Scalar FP register access inlining (rejected)

Tried forcing the four tiny scalar FP register getters and setters to
`#[inline(always)]`. Although one noisy benchmark pair looked faster, the
before/after WASM artifacts were byte-for-byte identical, proving LLVM had
already inlined them. Removed the no-op annotations.


### Browser WASM SIMD

Enabled `+simd128` for the `wasm32-unknown-unknown` target in
`.cargo/config.toml`. This lets LLVM use WebAssembly SIMD for the emulator's
128-bit vector state and bulk operations while leaving native builds and the
Rust examples unchanged. A paired 24-frame run on the same Node WASM harness
measured 275.5 versus 262.4 M instructions/s with byte-identical screenshots;
additional runs were host-contended, so the precise gain is not claimed. The
normal `make assets` path now ships this SIMD artifact.


### Self-loop execution (rejected)

Tried executing an unconditional self-loop repeatedly inside `exec_block` to
avoid chain transitions. Focused JIT tests passed, but the real WASM hbmenu run
stopped presenting frames and advanced indefinitely, so the experiment was
reverted without benchmarking or retaining any of its changes.


### Local JIT statistics batching

Moved translated-block and linked-block counters from per-transition writes
through `Cpu::jit` into local `run_jit` counters, committing them once when the
slice completes. The exceptional path commits before returning a translated
fault, preserving crash diagnostics and `jit_stats`. Three alternating 32-frame
comparisons on the SIMD WASM artifact measured 277.1/274.0/269.5 M/s with
batching versus 268.7/265.0/254.1 M/s before it; a 24-frame screenshot was
byte-identical. Focused JIT tests and Clippy pass. The result is a measured
browser-path improvement, while throughput remains below 1B/s.


### Direct mapped-page read (rejected)

Tried matching the page table directly in `read_bytes_in_page` instead of
converting `page_ref`'s `Result` to an `Option`. Output remained byte-identical,
but the paired 24-frame run measured 266.0 versus 274.4 M instructions/s, and
three 32-frame pairs were inconsistent. Restored the existing helper path.


### Multi-instruction run coalescing (rejected)

Tried extending the fault-trace ring's adjacent-run coalescing to JIT blocks
that retire more than one instruction. The extra arithmetic and branch cost
made the WASM result neutral to slightly slower across three 32-frame pairs,
so the original one-instruction fast path was restored.


### Local slice accounting (rejected)

Tried keeping `slice_used` in a local JIT variable to reduce per-block field
writes. Review found that `yield_thread` and SVC-driven context switches can
reset the field while a block is running; the local copy would miss that reset.
WASM measurements were mixed, so the experiment was reverted and architectural
preemption accounting remains on the CPU field.


### Fused-exit placeholder skipping (rejected)

Tried to shorten the segment before fused exits by subtracting the exit span.
The IR stores a fused exit's index at the first instruction it covers, so that
formula skipped a real prefix operation; the differential JIT tests caught
register divergence immediately. Restored the original segment boundary, which
already excludes the fused placeholders safely.


### Width-specialized MADD (rejected)

Tried separate 32-bit and 64-bit MADD/MSUB IR variants so W-register products
could use native `u32` arithmetic. Differential tests and screenshots matched,
but three WASM pairs showed no reliable gain and the paired frame was slower;
the generic compact `Madd` variant was restored.


### Watchpoint active flags (rejected)

Tried adding boolean fast-path flags so the normal write/read paths could skip
their disabled watchpoint range comparisons. Focused tests passed, but browser
measurements were noisy and below the retained artifact baseline, so the
existing predictable range checks remain.


### Scan-out format classification

`Gpu::present` used to call `ColorFormat::host_word` for every pixel. That
repeated the sRGB and channel-order classification across a full framebuffer.
It now computes a small host-word mode once per frame and keeps the exact
decoder for formats that need it. The alpha-channel property is also cached
outside the pixel loop. The browser artifact still produces the same screenshot
and measured 16-frame runs reached 273--289M instructions/s on the current
host; the change is retained as a browser-side scan-out optimization.


### Scan-out pixel writes

The scan-out buffer has an exact crop size, but the hot loop used `Vec::push`
for every output pixel. It now reserves that exact size once and writes through
an initialized `MaybeUninit`-backed allocation, then restores the typed `Vec`
after all pixels are filled. Row reversal and the format paths are unchanged.
The generated browser artifact produced a byte-identical screenshot and three
16-frame runs measured 282.6--293.3M instructions/s; the change is retained.


### Reused scan-out pixel allocation (rejected)

Tried recycling the previous framebuffer's pixel allocation through the
`MaybeUninit` direct-write path. Screenshots remained identical, but three
browser runs measured 284--293M instructions/s, below the retained path, so
per-frame allocation was restored.


### Larger translated blocks (rejected)

Raised `MAX_BLOCK_OPS` from 64 to 96 to amortize block-entry bookkeeping.
The workload's conditional exits already capped most blocks; repeated browser
runs averaged below the retained baseline, so the 64-op limit was restored.


### JIT dispatcher heuristic (rejected)

Changed the large translated `exec_op` dispatcher from forced inlining to
LLVM's normal `#[inline]` heuristic to reduce code size. The browser artifact
dropped to 201.7M instructions/s, so `#[inline(always)]` was restored.


### Native 32-bit carry arithmetic (rejected)

Replaced the shared helper's 32-bit `u64` sum and carry extraction with two
native `u32::overflowing_add` operations. Arithmetic tests passed, but browser
runs measured 264--287M instructions/s, so the original mixed-width helper was
restored.


### Unaligned scanout word loads (rejected)

Replaced the checked four-byte slice conversion in the buffered scanout path
with an unsafe unaligned pointer read. The output test passed, but the browser
benchmark fell to 286.0M instructions/s over 16 frames. The safe slice-based
read is restored.


### Inline UBFIZ shortcut (rejected)

Added a direct mask-and-shift path for the UBFIZ alias inside the existing
bitfield helper. The hot loop contains this instruction, but the extra runtime
branch reduced the browser benchmark to 284.2M instructions/s over 16 frames.
The generic bitfield implementation is restored.


### Larger direct block lookup table (rejected)

Expanded the direct-mapped JIT lookup from 4,096 to 16,384 slots to reduce
collisions on the measured title. JIT tests passed, but the browser benchmark
fell to 292.2M instructions/s over 16 frames, so the 4,096-slot table is
restored.


### Exposing JIT link hits in the browser (diagnostics)

Added the translated block link-hit counter to both the worker's JIT stats and
crash-report JSON, with the TypeScript protocol and no-session fallback updated
to match. This is diagnostic only: it does not alter execution or the shipped
WASM hot path. The browser benchmark still reports about 294M instructions/s
over 16 frames and now shows 32.6M linked block entries out of 44.2M, confirming
that the successor cache is active for the measured title.


### Dedicated scalar FP compare op (rejected)

Specialized the common scalar compare forms so translation stored their vector
registers and zero-compare bit, bypassing the scalar FP form dispatcher. JIT
differential tests passed, but the added Op arm enlarged the inlined executor
enough to reduce the browser benchmark from about 294M to 194M instructions/s.
The specialization was removed and the generic Fp operation is restored.


### Scalar FP fast wrapper (rejected)

Split the existing scalar FP data-proc handler into a tiny compare fast path
and an inlined generic helper, without changing the JIT IR. Differential tests
passed, but the browser benchmark fell to 277.4M instructions/s over 16 frames.
The wrapper was removed; the original single handler is restored.


### Register-offset width shortcut in the existing arm (rejected)

Added Load32/Store32 branches inside the existing register-offset load/store
operation, avoiding the generic access helper without adding IR variants. JIT
tests passed, but paired browser runs remained around 294.1M instructions/s,
so the extra branch was removed and the generic access path is restored.


### Width-specific scalar FP compare flags (rejected)

Added a direct f32 comparison path to avoid widening single-precision operands
to f64 before setting NZCV. Differential tests passed, but repeated browser
runs were 290.2--295.9M instructions/s over 16 frames and 288.6M over 32
frames, providing no reliable gain over the existing path. The generic f64
based helper is restored.


### Scan-out host-word mode

Folded alpha presence into the cached scan-out mode so the common RGBA8 path
writes the stored word directly, with no per-pixel alpha branch or mask. The
full decoder remains selected for sRGB and packed formats. The browser
screenshot stayed byte-identical and the rebuilt artifact measured 295.5M
instructions/s over 16 frames.


### Unchecked page lookup (rejected)

Tried replacing `Memory::page_ref`'s slice lookup with `get_unchecked`, since
guest addresses already produce a bounded page index. JIT tests passed, but the
browser measurement was neutral (295.0M versus 295.5M instructions/s), so the
checked helper was restored.


### Deferred JIT retirement (rejected)

Tried batching `cycles` and `steps` writes across translated blocks, flushing
before timing-visible scheduler, syscall, fault, and run boundaries. The
semantic tests passed, but three browser runs fell to 274--284M instructions/s
from the retained roughly 292--295M range. Immediate block retirement was
restored.


### Width-specialized JIT arithmetic (rejected)

Added const-width ADD/SUB helpers for translated operations whose `sf` bit is
known at translation time. Differential tests passed, but three browser runs
measured only 269--273M instructions/s versus the retained 292--295M range;
the shared helper was restored.


### Larger JIT lookup table (rejected)

Increased the direct-mapped JIT lookup from 4,096 to 16,384 slots because the
workload translates over 7,000 blocks. Repeated browser runs measured only
258--269M instructions/s, so the original 4,096-slot table was restored.


### Strong JIT successor links (rejected)

Tried replacing per-transition `Weak::upgrade` with strong `Rc` successor links,
clearing all links during invalidation and cache eviction. Tests passed, but
browser throughput collapsed to 128--130M instructions/s, so weak links were
restored.


### UBFIZ specialization (rejected)

Added a dedicated translated path for the common `UBFIZ` bitfield alias. The
browser artifact remained slower at 269--279M instructions/s across three
runs, so the generic bitfield helper was restored.


### Raw JIT successor pointers (rejected)

Tested non-owning `NonNull<Block>` links with explicit cache-wide clearing
before invalidation and eviction. Focused tests passed, but browser runs fell
to 281--285M instructions/s, so the existing weak-link implementation remains.


### NZCV sign-bit table (rejected)

Tried replacing the runtime sign-bit shift in `set_nzcv_from_alu` with a
two-entry constant lookup. Arithmetic tests passed, but three browser runs
were neutral to slower at 283--296M instructions/s, so the original helper
was restored.


### Direct JIT retirement accounting (retained)

JIT blocks now account for their retired run directly in the clock, step
counter, and recent-instruction ring. This avoids the interpreter-only
single-instruction coalescing branch while preserving the same fault trail and
cycle semantics. Focused JIT tests, formatting, clippy, and browser typechecks
pass. Paired 16-frame browser runs measured 291.6 and 293.9M instructions/s;
the result is effectively neutral, but the helper is smaller on the hot path
and frame output is deterministic across repeated runs.


### Predecoded scalar FP conversions (rejected)

Moved integer-to-float and float-to-integer conversion operands into dedicated
JIT IR variants and reused shared conversion helpers for the interpreter. JIT
tests passed, but paired browser runs fell to 283.8--284.4M instructions/s
from the retained roughly 292--298M range. The raw `Op::Fp` path is restored.


### Cached guest read page (rejected)

Added a last-page pointer cache to `Memory::page_ref` for sequential guest
loads. Memory tests passed, but paired browser runs fell to 285.5--285.7M
instructions/s. The additional cache branch and invalidation work cost more
than the page-table lookup, so the checked page-table path remains.


### Run-local successor cache (rejected)

Tested a bounded two-entry cache of recently followed JIT blocks to avoid
`RefCell` borrows and `Weak::upgrade` calls without creating ownership cycles.
Focused tests passed, but paired browser runs measured 283.5--283.8M
instructions/s, so the existing link path is faster.


### Binaryen post-link optimization (rejected for speed)

Applied Binaryen `wasm-opt -O4` to the generated browser module. It reduced the
WASM file size, but paired 16-frame browser runs remained 282.8--283.5M
instructions/s, with no throughput gain over the compiler-produced module.
The normal release artifact is restored.


### Focused 32-bit immediate load/store ops (retained)

Added dedicated JIT operations for the common unsigned-offset `LDR W` and
`STR W` forms without writeback. They bypass the runtime `Acc` direction and
width match while leaving all other addressing modes on the shared access
helper. JIT tests, formatting, Clippy, and TypeScript checks pass; paired
16-frame browser runs measured 290.8 and 292.3M instructions/s, and repeated
frame captures are byte-identical.


### Focused 32-bit register-offset load/store ops (rejected)

Extended the same specialization to register-offset `LDR W`/`STR W`. JIT and
Clippy checks passed, but paired browser runs dropped to 279.8--283.0M
instructions/s. The extra variants increased dispatch and code footprint more
than they reduced the shared access work, so register-offset operations remain
generic.


### Deferred sequential PC updates (rejected)

Tried deriving the PC from block indices and calculating it only for raw
operations and faults. Focused JIT tests passed, but paired browser runs were
neutral to slightly slower at 290.6--292.1M instructions/s. The original
per-instruction PC increment remains faster and simpler.


### Browser CPU profile after focused JIT work

Node's `--cpu-prof` over a timed 16-frame hbmenu run confirms that the shipped
hot path is overwhelmingly `Cpu::run_jit`; the next sampled functions are
`run_fp`, `Gpu::present`, and SIMD helpers. This rules out frontend or
post-link work as the current 1B-instruction bottleneck and points the next
large change at the JIT's block execution model itself.


### Single-exit executor fast path (rejected)

Special-cased blocks with one conditional exit to avoid the general optional
lookup. JIT tests passed, but paired browser runs fell to 261.1--264.1M
instructions/s. The uniform exit loop remains faster.


### Per-block dirty-write classification (rejected)

Added a conservative `may_write_memory` flag to translated blocks and skipped
dirty-code checks for pure register blocks. Correctness tests passed, but the
larger block layout and extra branch regressed paired browser runs to
249.6--257.3M instructions/s. The unconditional cheap dirty-queue check is
restored.


### Out-of-line FP dispatcher (rejected)

Moved the scalar FP/SIMD fallback branch out of the inlined JIT operation
dispatcher to reduce hot function size. Focused tests passed, but paired
browser runs fell to 266.3--272.2M instructions/s; the inlined dispatcher is
faster under V8 and is restored.


### Smaller 32-op translation blocks (rejected)

Reduced `MAX_BLOCK_OPS` from 64 to 32 to test whether smaller translated
traces would tier faster in V8. Paired browser runs measured 285.3--286.3M
instructions/s and translated 7,317 blocks instead of 7,122, so the original
64-op cap is restored.


### Unchecked translated-op pointer walk (rejected)

Replaced the translated-op slice iterator with an unchecked raw pointer loop to
test for residual WASM bounds checks. Two identical browser runs measured
285.1M instructions/s, below the retained path; the safe iterator is restored.


### Direct checked-page 32-bit load (rejected)

Added a backed-page, in-page `read_u32` fast path for the specialized JIT load,
falling back for watchpoints, faults, and page boundaries. Paired browser runs
measured 289.5--290.6M instructions/s, so the existing checked accessor is
restored.


### Load-to-UBFIZ superinstruction (rejected)

Fused the common 32-bit load followed immediately by UBFIZ into one translated
operation while retaining a filler slot for block indexing. Differential tests
passed, but the larger operation arm reduced the browser benchmark to 282.4M
instructions/s over 16 frames. The original separate operations are restored.


### Direct RGBA8 scanout branch (rejected)

Added a fast branch for the common host-word RGBA8 mode to bypass the
per-pixel four-way format match. The browser benchmark fell to 288.4M
instructions/s over 16 frames, so the original invariant match is restored.


### Scalar FP compare form (rejected)

Added a dedicated FpForm classification for FCMP/FCMPE so those instructions
could bypass the generic scalar data-proc decoder without adding an Op variant.
Differential tests passed, but browser throughput fell to 287.7M
instructions/s over 16 frames. The generic FP form and handler are restored.


### Relaxed SIMD target feature (rejected)

Enabled relaxed SIMD in the browser WASM build alongside simd128. The build
completed, but the V8 benchmark fell to 234.7M instructions/s over 16 frames.
The target feature is reverted to standard simd128 for compatibility and speed.


### Explicit one-byte JIT Op tag (rejected)

Marked the translated operation enum with repr(u8) to narrow its discriminant
while keeping the 16-byte payload layout. JIT tests passed, but browser
throughput fell to 284.6M instructions/s over 16 frames. The compiler-selected
enum representation is restored.


### Two-entry successor cache (rejected)

Expanded each block's weak successor cache from one entry to two to cover
alternating branch targets. JIT tests passed, but the browser benchmark fell
to 286.8M instructions/s over 16 frames. The one-entry cache is restored.


### Merged 32-bit and 64-bit move variants (rejected)

Merged Mov32 and Mov64 into one operation carrying an sf flag to remove a
dispatcher arm. JIT tests passed, but browser throughput fell to 281.8M
instructions/s over 16 frames. The separate variants are restored.


### Split successor PC and weak pointer (rejected)

Stored the cached successor PC separately from its weak block pointer so a
miss could avoid borrowing the pointer cell. JIT tests passed, but browser
throughput fell to 290.0M instructions/s over 16 frames. The original tuple
cache is restored.


### Arena-indexed successor links (rejected)

Replaced weak-pointer upgrades with stable arena indices and validity flags to
remove per-link reference-count work. JIT tests passed, but browser throughput
was 293.1M instructions/s over 16 frames and invalidated blocks were retained
until a wholesale cache clear. The original weak links are restored.


### Register-offset 32-bit access specialization (rejected)

Added dedicated JIT IR variants for common `LDR/STR W` register-offset forms,
then reverted them after JIT tests passed but browser throughput dropped from
about 300M to 127.0M instructions/s. The larger enum dispatch outweighed the
removed generic access path.


### Inline register-offset access fast path (rejected)

Added a conditional fast path inside the existing register-offset arm for
32-bit loads and stores. Correctness remained green, but browser throughput
fell to 239.7M instructions/s, so the generic `Cpu::access` path is restored.


### 128-operation JIT blocks (rejected)

Raised `MAX_BLOCK_OPS` from 64 to 128. The browser benchmark fell to 243.9M
instructions/s over 16 frames; 64 operations is restored.


### 48-operation JIT blocks (rejected)

Reduced `MAX_BLOCK_OPS` from 64 to 48. The browser benchmark measured 293.8M
instructions/s over 8 frames, below the restored 294.4M 64-op result, so 64 is
retained.


### Repository optimization issue inventory

Checked the Forgejo issue and pull-request APIs for `Tenshii/Switch-Wasm`.
There are no open issues or pull requests, and the only historical item is a
closed prose cleanup. The remaining performance work is therefore in the JIT
and benchmark evidence documented here rather than an untracked issue.


### Outlined access dispatcher (rejected)

Changed the inlined `Cpu::access` width and direction switch to
`inline(never)` to reduce the JIT dispatcher footprint. The browser benchmark
dropped to 270.3M instructions/s over 16 frames, so forced inlining is restored.


### Direct 32-bit page accesses (rejected)

Replaced the generic byte-array conversion in `Memory::read_u32` and
`write_u32` with manual in-page byte assembly. The browser benchmark dropped to
250.3M instructions/s over 16 frames; the generic page helpers are restored.


### Simplified JIT housekeeping subtraction (rejected)

Replaced per-block `saturating_sub` deadline accounting with an explicit
comparison and subtraction. The browser benchmark measured 294.3M
instructions/s over 16 frames, effectively neutral to the restored path, so the
original operation remains.


### Outlined block executor (rejected)

Changed `exec_block` from forced inlining to an out-of-line call. The browser
benchmark dropped to 272.6M instructions/s over 16 frames, so inlining remains.


### Unsafe exit lookup (rejected)

Replaced the conditional-exit slice lookup with an unchecked indexed read. The
browser benchmark measured 294.5M instructions/s over 16 frames, neutral to the
checked lookup, so the safe form remains.
