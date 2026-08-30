# switch-wasm — boot status

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
| `NX-Shell.nro` | halts at 433,783 steps, no output — **regressed** | no | measured |
| Home Menu (`qlaunch`) | draws; 88 draws a frame on the WebGPU backend | yes | carried |
| `sysinfo` / `NX-Fetch` / `nxdumptool` | render | yes | carried |
| `JKSV.nro` | full UI: text, icons, save tiles | yes | carried |
| `Checkpoint.nro` | layout and chrome; text unchecked since `SHFL` landed | yes | carried |
| "A Short Hike" (NSP) | composites 1280x720; steady state 2 draws, never a scene | partial | measured |
| "Minecraft" (NSP) | the world, on the device: 110 draws a frame, no fallbacks | yes | measured |
| "Tomodachi Life" (NSP) | its loading screen, every pixel, at 3.98B steps | yes | measured |

A retail title decrypts, mounts its RomFS, runs `rtld` → `main` → `subsdk*` →
`sdk` through real `nnSdk` init, gets its heap, events and input, brings up its
graphics stack, opens its audio device, and runs on into its own loop. **Every
service it asks for has a real implementation** — a full boot logs no `no
implementation` and no `unimplemented` lines. `make test`: **908 tests passing**.

## Method, which is the part that generalises

- **Silence is not evidence.** `/dev/nvhost-ctrl-gpu` `0x13` (`VsmsMapping`) was
  the only line a whole retail run logged, which made it look causal. Answering
  it with any scalar, refusing it, and implementing it properly all give
  byte-identical GPU statistics and the same spinning pc. The emulator only logs
  the gaps it knows it has.
- **A step counter that keeps climbing is not a title that keeps running.** Just
  Dance 2023 reached seven billion steps with the main thread retiring a
  *constant* 760M at every budget — everything after was two threads re-asking
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
  dropped — both are a black frame otherwise.
- **Getting the backtrace is what makes a fault findable.** `dump_exefs` lays the
  modules out at their real load addresses and writes a sorted `symbols.txt` from
  `sdk`'s 36,622 `DT_HASH` symbols; `0x0ce6c0c8` says nothing,
  `sdk!nn::diag::detail::Abort+0x18` says everything.
- **Nothing in the tree looked slow.** A `perf` profile of a Home Menu boot came
  back 37% `getenv` and 18.7% SipHash — together more than the shader
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
  never wrote its struct, so NX-Fetch displayed "Horizon OS 115.119.105" — the
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
- **A pointer buffer a caller is told it cannot use.** Every session answered
  `QueryPointerBufferSize` with 0, and `nnSdk` measures an explicit
  `SfBufferAttr_HipcPointer` argument against it before sending —
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
- **A `Poll` that returns instantly is not one that reports nothing ready.**
  NXpotify's Zeroconf listener is `if (poll(&pfd, 1, 200) <= 0) continue;` — with
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
- **An update NSP holds no game.** Its Program NCA carries a complete ExeFS —
  patched modules, not a delta — and a RomFS section encrypted `AesCtrEx`: the
  BKTR form, holding only the changed ranges plus the two tables indexing them
  against the base. `bktr.rs` composes the pair, streaming both containers. The
  subsection counter replaces the section counter's **generation** word, not its
  secure value — the wrong way round is quiet, because the tables still decrypt
  and every byte of *data* is noise. An update's Program NCA carries the **base**
  title id, so pairing is by program id and what identifies a container as an
  update is that its RomFS is a patch.
- **DLC is not an update, and is much less than one**: one Data NCA with an
  ordinary RomFS whose title id is the base's plus an index, mounted through
  `OpenDataStorageByDataId` — the same path a system data archive takes, so the
  reading half was already built. What was missing was `aoc:u` saying the content
  exists, since a title never asks for content the list does not have.

## GPU

The nvdrv/nvmap/GMMU/channel/copy-engine path is real, as is the 3D shader core
— a Maxwell SASS interpreter feeding a software rasterizer, with compute
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
a bare success — a driver told "nothing is registered, ever" re-registers
forever. Same for `EventSignal`/`EventWaitAsync`, where the driver parks on a
slot nothing would set.

**Four `texs` bugs, each masking the next**, which is why the symptom never
looked like four things:

- A `texs` has **two destination registers, not one run of four** — invisible
  whenever `dst2 == dst + 2`, exactly what the first fixture did. JKSV's glyph
  shader clobbered the `1/w` every later `ipa` multiplies by, so every glyph came
  out alpha-zero: text present and completely invisible.
- **The handle immediate is a dword index into the driver constant bank, not a
  byte offset.** Reading it as bytes landed in the header ahead of the table,
  which begins `0, 1, 2, 3…` — a plausible handle table, so every draw resolved
  to a plausible handle, and every draw resolved to the same one.
- **`TexCbIndex` is a register, not a constant** — Mesa writes 15, deko3d 0.
- **Whether the viewport flips y is a register, not a constant.**

**A Unity title is written in `half`, and none of it decoded.** "A Short Hike"
renders into an RGBA16Float target and composites with two full-screen quads;
both were dropped, one on sampling a float texture and one on the fp16 ALU, so
nothing wrote the swapchain and it presented the zeros it was allocated with —
transparent, not black. Two things worth carrying: **110 of the 145 dropped draws
were `hadd2.f32`**, a plain float add issued on the half unit, so most of what a
`half` shader costs is not half arithmetic; and **`f32_to_f16` had to stop
truncating** — as the rounding step of every fp16 instruction, round-to-zero
biases a whole shader. It now rounds to nearest, ties to even, reaching the
subnormals, which is the mode WGSL's `pack2x16float` uses.

**Unity culls on the CPU**, so a title whose scene or camera is wrong emits *no
draw calls at all* rather than draws that cover nothing. A Short Hike's steady
state is two draws a frame forever, frames 30 and 300 byte-identical to frame 1,
47% of instructions in the Boehm GC's mark loop — IL2CPP working normally, simply
producing no renderers. Zero scene draws is the title's own state; chasing it
through the GPU is chasing the wrong end.

**Minecraft: three gaps, each hiding the next.** Every draw read a `4x16`
half-float attribute neither backend could fetch (110 draws, 110 skipped, 0
pixels lit, and in a browser the first fallback latches `software_frame`, which
is where the frame time went). Then every fragment shader opened with
`ipa.centroid`, and `Op::Unimplemented` is fatal to the interpreter as well as
untranslatable. Then the frame came out upside down — not the viewport, but
`QUEUE_BUFFER` throwing away a `QueueBufferInput` that said `FLIP_V`.

**`SHFL` took a quad to run**, and three things came out of it: a shuffle is only
half of a derivative (`FSWZADD` subtracts in whichever direction the lane's
position calls for, so decoding the shuffle alone leaves every such shader
failing on the *next* instruction); both operands are a register or an immediate
and the immediate sits in a *different field* from the register, so reading the
wrong field gives a plausible lane number rather than an error; and helper lanes
are the point, not an artefact, which is why the quad walk is gated on the
program containing a `shfl` or `fswzadd`.

**Which winding is front is decided in NDC.** Facing was read off screen-space
signed area, on the reasoning that the viewport's y scale decides winding "as it
does on hardware". It does not — `SetWindowOrigin` bit 4 is the only thing that
reverses it, and deko3d drives that and `viewportFlipY()` as two separate flags.
So culling was inverted for every title whose driver flips y, which is every
title built against nnSdk: Tomodachi Life's single full-screen composite quad was
thrown away and the frame was black. The rasterizer's own culling fixture had
encoded the inversion too — its "clockwise in NDC" triangle has a shoelace of
**+4**. hbmenu, JKSV, sysinfo, NX-Fetch and the Home Menu are byte-identical
either way: content that does not cull, or does not flip y, never reached it.

Smaller ones: **`AntiAliasEnable` does not size a surface; `MsaaMode` does** (a
2560x720 `2x1_D3D` target read as 2560 pixels wide, and the title's own resolve
shrank the frame to a quarter). **The colour write mask was unimplemented** —
A Short Hike writes `SetCtWrite` (0x680, a nibble per
channel) and `ColorMaskCommon` (0x3E4) 420 times a frame and turns alpha off for
99 draws, and an unwritten mask must read as *all* channels, since zero is the
register file's initial value. **`VOTE.VTG` writes neither register nor
predicate**, but a refused instruction fails the whole draw — all 52 of them. A
*fixed* vertex attribute with no buffer behind it must read the `vec4` default
rather than dropping the draw; a *disabled* one is still an error, because that
means a register was read wrong. **`SetDstWidth` counts elements, not bytes.**

**Where a frame's time goes** (ablation on 30 JKSV frames): fragment shader
interpretation 36%, rasterize/depth/blend/pixel I/O 30%, ARM interpreter 25%,
texture sampling 9%. A full-screen pass is 921,600 fragment invocations, so
NXpotify's 2.6 s/frame was a texture result rescanning the decoded program and
building a `Vec` per instruction — about a hundred heap allocations per pixel.
Where a result lands is a property of the *program*, not the invocation; that
plus two allocation fixes took it to 0.67 s/frame, byte-identical.

## Next

1. **The rasterizer renders Minecraft black, and it is the reference.** The
   device draws the title correctly, but `screenshot_nsp` over the same frame
   reports 110 draws, `draws_skipped: 0` and 0 of 921,600 pixels lit, at frame 3
   and frame 20 alike. Every other path is checked by agreeing with the
   rasterizer, so this is a hole in the check itself. Bisect with `GPU_ONLY`. The
   winding fix did not clear it, though frame 3 is 6 draws and not the 110-draw
   one, so frame 20 is the measurement that would settle it.
2. **A retail title that draws but never a scene.** Tomodachi Life presents at
   3.98B steps with 7 draws; A Short Hike no longer reaches a frame at all — at
   HEAD it spins after one submission and 3,536 methods, and past the
   pointer-buffer fix it takes a `write to read-only address 0x0aa28f50` at
   `pc=0xa70b814`, step 404,553,728. Whatever they wait on is above the GPU.
3. **`CreateAliasStackUnsafe` is an open abort.** Just Dance 2023 maps 38 thread
   stacks and aborts on the 39th, which never reaches a syscall — so it is inside
   `nn::os::detail::AslrSpaceAllocator`'s own bookkeeping, built from `svcGetInfo`
   12/13 with the heap and alias regions outside the ASLR region it is told
   about. Shrinking the stack region *looks* like a cure and is luck: the region
   size is the modulus nnSdk's random placement uses, so a different size is a
   different address sequence. Booted without its update the title deadlocks for
   real at 765M steps; patched with 1.0.1 it presents 76 frames and dies
   elsewhere.
4. **NX-Shell regressed** to 433,783 steps with no output, from a recorded clean
   `ExitProcess` at 15,692,155 steps *with* output. The cheapest bisect here —
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
   with `--no-hinting` renders perfectly. Invisible in normal use — the shipped
   font has no hinting — but a real correctness gap.
8. **Minecraft is CPU-bound**, not GPU-bound: 21.9 billion instructions buys 20
   frames, and a steady frame is 173-220 ms with every draw on the device.

Lower priority: hbmenu's entry label renders as a blank box; NAND-vs-SD storage
is one hardcoded 32 GiB for both free and total; Checkpoint never presents a
frame; the applet path's queue transform offset is unverified.
