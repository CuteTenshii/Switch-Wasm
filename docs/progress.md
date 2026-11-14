# switch-wasm: boot status

Goal: get real homebrew and real retail titles to **run**, and put what they render on the canvas. This is the log of what broke and what it taught; AGENTS.md is the standing state and carries every rule that came out of it.

## Where it stands

Per-title status lives in [compat.md](compat.md).

A retail title decrypts, mounts its RomFS, runs `rtld` → `main` → `subsdk*` → `sdk` through real `nnSdk` init, gets its heap, events and input, brings up its graphics stack, opens its audio device, and runs on into its own loop. Almost every service command it asks for has a real implementation; `acc:manager` 150, which Asphalt 9 calls, is one that does not.

## Method, which is the part that generalises

- **Silence is not evidence.** `/dev/nvhost-ctrl-gpu` `0x13` (`VsmsMapping`) was the only line a whole retail run logged, so it looked causal. Answering it with any scalar, refusing it, and implementing it properly all gave byte-identical GPU statistics and the same spinning pc. The emulator only logs the gaps it knows it has.
- **A step counter that keeps climbing is not a title that keeps running.** Just Dance 2023 reached seven billion steps with the main thread retiring a *constant* 760M at every budget; the rest was two threads re-asking `svcWaitSynchronization` each slice. Profiling at four budgets showed it. Parking waits took the Home Menu's tenth frame from 170.6M steps to 39.2M, byte-identical.
- **A frozen title is often a configuration difference, not a bug.** Host examples boot handheld and the browser is usually docked, so a title told 720p with a 1080p swapchain composites into a corner that looks exactly like a rendering fault. `--docked` settles it.
- **Prove a new decode guard reaches a real encoding**, and reach for `tools/difftest.py` before hand-deriving an expected value. About thirty decode bugs came out of the three homebrew titles, nearly all one shape: a guard testing a field that includes a fixed bit, so a whole encoding group was dead code ([a64-decode.md](a64-decode.md)). The rest were sign-extension widths.
- **The file is the oracle, the spec is not.** AES-CTR with the wrong key still "decrypts" into plausible garbage, so decryption is verified by matching a Program NCA's ExeFS against Nintendo's own stored SHA-256.
- **Ask where a draw's fragments died before blaming the pipeline.** `TRACE_DRAW` tallies culled/degenerate/uncovered/killed/written per draw, and `culled=2` of `tris=2` on exactly the blits found the winding bug below. The `[gpu] draw` line names the render target's cpu address beside the cull state, which separates a title compositing offscreen from one whose composite was dropped: otherwise both are a black frame.
- **Getting the backtrace is what makes a fault findable.** `dump_exefs` lays the modules out at their real load addresses and writes a sorted `symbols.txt` from `sdk`'s 36,622 `DT_HASH` symbols; `0x0ce6c0c8` says nothing, `sdk!nn::diag::detail::Abort+0x18` says everything.
- **A title that quits by itself is a bug until proven otherwise, and one only the translator exposes may still not be the translator's.** Just Dance 2019 halted at step 1,097,299,725 with the JIT on and never with it off. A freshly created HTTP thread returned through a zeroed return address, and the `ret`-to-0 special case turned that into `ExitProcess`. Polling the slot every 16 instructions caught the writer: a *different* thread, joining the old thread whose `ThreadType` the new one reused. `svcWaitSynchronization` treated a thread handle as always ready, so the join returned while the thread ran; whether the joiner got the CPU in that window depended only on where the scheduler switched, and the translator switches at block boundaries, not at every instruction. A thread handle is now signalled when its thread exits. `jit_bisect` cannot follow a multithreaded title past its first thread switch, because the two engines legitimately interleave differently; the slot poll is what reached the bug.
- **Nothing in the tree looked slow.** A `perf` profile of a Home Menu boot came back 37% `getenv` and 18.7% SipHash, together more than the shader interpreter, the rasterizer and the ARM interpreter combined. Removing both ([formats.md](formats.md)) took it from 73.6 s to 27.7 s, every frame byte-identical.

## Retail boot: what each fault was

- **`rtld` has no MOD0 header**, so the `.text`+0x30 entry skipped the code that establishes its load address; the base stayed 0 and a `bss_end - base` zero-fill overwrote the loop running it. Cost 25M instructions to find ([formats.md](formats.md)).
- **`rtld` finds its own modules** with `svcQueryMemory`, looking for `CodeStatic` + `R-X`. A blanket RWX permission and single-range read-only tracking both broke that ([kernel.md](kernel.md)).
- **The `sdk` abort was a recursive-lock assertion on an untouched mutex**, and naming the backtrace turned up three real bugs: the main thread handle was never delivered (it is X1), `svcGetInfo` CoreMask/PriorityMask fell into a `_ => 0` default, and `svcWaitSynchronization` answered X1 = 1 where X1 is the signalling handle's *index*.
- **Success with an unfilled out parameter, repeatedly.** `GetFirmwareVersion` never wrote its struct, so NX-Fetch showed "Horizon OS 115.119.105": the ASCII of `swi`, left in the buffer by an earlier `acc` call, and load-bearing because libnx seeds `hosversionGet()` from it. `CloneCurrentObject` returned no session handle. `IStorage::Read` used `IFile::Read`'s field layout.
- **One field width cost a whole title.** Reading `OpenAudioOut`'s 16-bit channel count as 32 bits reported 0xcafe0002 channels ([audio.md](audio.md)), so Unity tore audio down without `CloseAudioOut` and the retry hit a registry that still held the device open.
- **The address-space split, not the allocator.** Tomodachi Life started a thread on a null 800M instructions in: `svcGetInfo` reported 1.5 GiB total, the title sized its pools *from that figure*, and the other 1.5 GiB was an alias region it never touches. Each layout now spends the space on the region its own titles grow into ([kernel.md](kernel.md)).
- **A store-release run as a store-exclusive.** The A32 decoder ignored bits 9:8, which separate ARMv8's `LDA`/`STL{,B,H}` from `LDREX`/`STREX` ([cpu.md](cpu.md)). Mario Kart 8 Deluxe's `STLH` dropped its store, so the SDK's cached pointer-buffer size read back as 0 and its first pointer buffer was refused: the same 11-141 as below, from an entirely different cause. The A32 media group is now complete and checked against `qemu-arm` (`tools/a32_media_reference.py`).
- **A pointer buffer a caller is told it cannot use.** Answering `QueryPointerBufferSize` with 0 was Tomodachi Life's `sf` 11-141 abort at 2,524,316,651 steps ([kernel.md](kernel.md)). Answering 0x8000 exposed the second half: `cmifRequestInAutoBuffer` fills in both descriptor forms and nulls the one it did not choose, so nvdrv's AutoSelect ioctl argument arrived as a null map-alias buffer and `nvn` gave up on graphics at 33M steps ([kernel.md](kernel.md#kernelipcrs)). `ipc_pick_buffer` is the rule.
- **An `nn::hid` state is read under a seqlock and bit 0 is the lock**, so a storage entry's sampling number is the state's own doubled ([services.md](services.md#hidrs)). Undoubled, the first sample was odd and Tomodachi Life's main thread sat in that retry loop for 20 billion instructions with every counter byte-identical. `libnx` never looks at the bit, so no homebrew could have shown it.
- **`hid` samples on a clock, not on input.** Publishing only on host input froze the sampling number, and a title waiting for a newer sample waited forever. That was also why the CLI could not reproduce what the browser saw: a hand on the keyboard kept the LIFO moving.
- **Every thread said it was on core 0.** Tomodachi Life creates its threads on core 0 and moves each with `SetThreadCoreMask(-1, one-core mask)`, which was a stub, and `GetCurrentProcessorNumber` answered 0 regardless. Two render workers meant for cores 1 and 2 ran the same job into one command list, and at 15.57G steps, just after the first save, a record half-written by each called a vtable slot that is null in a pool's base class: pc=0. What found it, in order: the object's vtable (`--dump '*x25'`), that the vptr was written once at boot and never again (`--trap-write` + `--trap-last`), then that two threads wrote the record with identical arguments. The exclusive monitor and the loader were both suspects and both innocent.
- **Every thread had id 1.** `svcGetThreadId` answered 1 whatever it was asked, so to a title comparing ids every thread was the same thread. The Legend of Zelda: Echoes of Wisdom's actor factory checks whether it is on one of the threads it keeps a list of, and took the main thread for another: it initialised Link inside a creation that holds a global `std::mutex`, Link's initialisation created his sword through the same path, and the SDK aborted (2162-0001) on the second lock at 1.37G steps. The trail: the abort was `pthread_mutex_lock` refusing a self-deadlock; `--trap-write` showed the lock taken once and never released; `--watch-dump` read the two registry names (`PrologueLink`, `PlayerItemSword`); and resolving the factory's PLT imports found `nn::os::GetThreadId` beside the branch.

## Containers

- **A cartridge image is the same container one layer down.** An XCI's root is an HFS0 whose entries are the cartridge's partitions, each an HFS0 holding NCAs, and HFS0 is PFS0 with a 0x40-byte entry, so one reader serves both ([formats.md](formats.md)). `Xci::content` flattens the partitions into one file table under two rules: `update` is left out (a firmware bundle with Program content among it, and every search that follows is "the first/last NCA of type X"), and `secure` goes last, because the Program scan keeps the last match. Verified by repacking "A Short Hike" into an XCI and booting it to the same frame 3.
- **Four decryption bugs, each flipping the hash from mismatch to match**: the section table's units, the CTR counter's absolute position, the title key's generation (Asphalt 9's ticket says 0, its content uses 7), and where an IVFC section's data starts. Each rule is in [formats.md](formats.md).
- **An update NSP holds no game.** Its Program NCA carries patched modules, not a delta, and a BKTR (`AesCtrEx`) RomFS holding only the changed ranges plus the tables indexing them against the base; `bktr.rs` composes the pair, streaming both containers. Putting the subsection counter in the wrong word is quiet: the tables still decrypt and every byte of *data* is noise. Pairing is by program id ([formats.md](formats.md), [web.md](web.md)).
- **DLC needed only `aoc:u`.** Its Data NCA mounts through `OpenDataStorageByDataId`, the path system data already took; what was missing was `aoc:u` saying the content exists, since a title never asks for content the list does not have ([services.md](services.md)).

## GPU

The shader core is a Maxwell SASS interpreter feeding a software rasterizer; `switch-gpu` translates SASS to WGSL for `wgpu`, with the rasterizer as the reference it must agree with ([gpu.md](gpu.md), [gpu-backend.md](gpu-backend.md)).

**One transport bug stopped all of device init.** `nnSdk` reads an ioctl's payload back through its `{ buf_size, buf_addr }` pair, the IPC layer wrote back only the first receive buffer, and the driver read zeroed GPU characteristics, closed the device and returned null. `libnx` reads the payload inline and worked ([gpu.md](gpu.md)).

**Four `texs` bugs, each masking the next**, which is why the symptom never looked like four things:

- A `texs` has **two destination registers, not one run of four**, invisible whenever `dst2 == dst + 2`, exactly what the first fixture did. JKSV's glyph shader clobbered the `1/w` every later `ipa` multiplies by, so every glyph came out alpha-zero: text present and completely invisible.
- **The handle immediate is a dword index, and `TexCbIndex` is a register, not a constant.** Read as bytes it landed on a header that looks like a handle table, so every draw resolved to the same plausible handle ([gpu.md](gpu.md)).
- **Whether the viewport flips y is a register, not a constant.**

**A Unity title is written in `half`, and none of it decoded.** "A Short Hike" renders into an RGBA16Float target and composites with two full-screen quads; one was dropped on sampling a float texture and one on the fp16 ALU, so nothing wrote the swapchain and it presented the zeros it was allocated with: transparent, not black. **110 of the 145 dropped draws were `hadd2.f32`**, a plain float add issued on the half unit, so most of what a `half` shader costs is not half arithmetic. `f32_to_f16` also had to stop truncating ([gpu.md](gpu.md)).

**Unity culls on the CPU**, so a title whose scene or camera is wrong emits *no draw calls at all* rather than draws that cover nothing. A Short Hike's steady state was two draws a frame forever, frames 30 and 300 byte-identical to frame 1, 47% of instructions in the Boehm GC's mark loop: IL2CPP working normally and producing no renderers. Zero scene draws is the title's own state; chasing it through the GPU is chasing the wrong end.

**Minecraft: three gaps, each hiding the next.** Every draw read a `4x16` half-float attribute neither backend could fetch (110 draws, 110 skipped, 0 pixels lit, and in a browser a fallback latches `software_frame`, which is where the frame time went). Then every fragment shader opened with `ipa.centroid`, and `Op::Unimplemented` is fatal to the interpreter as well as untranslatable. Then the frame came out upside down: not the viewport, but `QUEUE_BUFFER` throwing away a `QueueBufferInput` that said `FLIP_V`.

**Minecraft's fourth gap was one refused datagram**, 29.6 billion instructions in, long after the title is drawing. The run jumped to address 0, in the browser and on the CLI alike, at exactly step 29,593,165,824. The faulting method's vtable names the class (`19RakNetServerLocator`, from its typeinfo): it builds the LAN discovery ping list through a `RakPeerInterface*` that its setup *nulls deliberately* after a failed `Startup`, and the caller does not check. `Startup` failed because `bsd` answered RakNet's test datagram with `ENETUNREACH` while `nifm` reported the link up ([services.md](services.md)). The last seven service calls before the fault are the whole story: Socket, SetSockOpt x4, Bind, GetSockName, SendTo (4-byte payload, 16-byte `sockaddr_in`), Close.

**The crash was the smaller half of it.** The title was *stuck* long before: 6 frames and 12 draws, unchanged from 4 billion steps to 29 billion. With the send answered, the same 40-billion-step run presents **504 frames and 9,720 draws** across 1,515 submissions and gets as far as `nsd:u` cmd 21 and its own error applet, which is what a console with no route off the LAN should do. The retry around the failed `Startup` was eating the run, and an earlier "21.9 billion instructions buys 20 frames" was measured through it.

**The fault the emulator reported was not the fault the guest took.** A call through a null vtable read `[0]`, `[0x1f8]` and then branched to 0, and every one of those succeeded: `Cpu::bootstrap` soft-maps the whole space, so unwritten pages read as zeros, the instruction fetch at 0 included. What is reported is `unimplemented instruction 0x00000000 at pc=0x0`, and only the last-64 instruction trail says it was a null dispatch. A fetch from a page nothing has ever written is always a wild jump; a fault at the branch would name it.

**`SHFL` took a quad to run**, and three things came out of it: a shuffle is only half of a derivative (`FSWZADD` subtracts in whichever direction the lane's position calls for, so decoding the shuffle alone leaves every such shader failing on the *next* instruction); the immediate operand sits in a *different field* from the register, so reading the wrong field gives a plausible lane number rather than an error; and helper lanes are the point, not an artefact, which is why the quad walk is gated on the program containing a `shfl` or `fswzadd`.

**Which winding is front is decided in window space, after the viewport.** Facing was first read off screen-space area as if y pointed up, which inverted culling for every title built against nnSdk and threw away Tomodachi Life's full-screen composite. The fix read it off the NDC winding instead, a second misreading that agrees only through a viewport that mirrors y, and Echoes of Wisdom's offscreen post-processing runs through one that does not: every full-screen quad was culled and its frame stayed black on both renderers ([gpu.md](gpu.md)). **When a rule is confirmed by one title, check it against the case that title does not exercise.**

Smaller ones: **`AntiAliasEnable` does not size a surface; `MsaaMode` does** (a 2560x720 `2x1_D3D` target read as 2560 pixels wide, and the title's own resolve shrank the frame to a quarter). **The colour write mask was unimplemented**: A Short Hike writes `SetCtWrite` (0x680, a nibble per channel) and `ColorMaskCommon` (0x3E4) 420 times a frame and turns alpha off for 99 draws, and an unwritten mask must read as *all* channels, since zero is the register file's initial value. **`VOTE.VTG` writes neither register nor predicate**, but a refused instruction fails the whole draw, all 52 of them. A *disabled* vertex attribute is still an error, unlike a fixed one, because it means a register was read wrong.

**The rasterizer's edge functions were f32, and Minecraft's clouds are clipped to `w=1e-6`.** Near-plane clipping puts vertices 4e8 pixels out, where an f32 edge product is off by tens of pixels: misplaced blocks and black triangles across the right half of frame 20, 133,573 pixels past 3/255 of the device frame. Edges and area are f64 now, and the two frames agree within 3/255.

**Where a frame's time goes** (ablation on 30 JKSV frames): fragment shader interpretation 36%, rasterize/depth/blend/pixel I/O 30%, ARM interpreter 25%, texture sampling 9%. A full-screen pass is 921,600 fragment invocations, so NXpotify's 2.6 s/frame was a texture result rescanning the decoded program and building a `Vec` per instruction, about a hundred heap allocations per pixel. Where a result lands is a property of the *program*, not the invocation; that plus two allocation fixes took it to 0.67 s/frame, byte-identical.

## JIT: emitted wasm, and what it took to reach it

How emitted regions are compiled, installed and entered is in [jit.md](jit.md).

**`--growable-table` is load-bearing and silent.** LLD emits the function table with its maximum equal to its minimum, so `table.grow` fails, every install answers "could not", and the translator interprets everything while looking exactly like a build with no emitter. Nothing failed or warned; the counters said 1,001 block entries and 0 compiled. It is in `.cargo/config.toml` now, and `tools/jit_wasm_check.mjs` asserts that blocks are not only compiled but **entered**, because the register comparison alone passes on a build that compiled nothing.

The wiring is checked from both ends, because neither end can check the other: `jit_emitted_test.rs` drives the state machine on the host, and `jit_wasm_check.mjs` runs the `make wasm` artefact under V8, the only place the wasm32-specific offsets exist. Perturbing one offset by eight makes it name the exact registers that diverged. Those offsets come from `offset_of!`, never a written-down number, since neither `Cpu` nor `Memory` is `repr(C)`.

**A module per block had two costs that grow with the number of modules** (a module now holds a region), and both made the emitter a net loss on Just Dance 2019 until they were found ([jit.md](jit.md)). At `HOT = 16` a run emitted 7,990 blocks:

- **Every `memory.grow` walks every instance.** About 500 grows in an asset-loading burst took half the main thread for 0.4 s. `dlmalloc` at a 4 MiB granularity (`src/heap.rs`) cut host cycles over a 40-frame run by 41%, host instructions by 25% and the frame mean by 7%, guest work identical.
- **Scattered code stalls.** With growth fixed, the emitter was still 4% slower per steady frame than interpreting everything, host *instructions* level and *cycles* up. Emitted code itself is fast: a microbenchmark with one or two blocks shows it at parity or better. `HOT = 512` emits 1,185 blocks taking 97% of the entries, and beats the interpreter (-0.8% instructions on every pair, -1.6% cycles); 4096 (461 blocks) measured the same as 512.

**Per-entry overhead dwarfs what emitting saves.** A four-instruction guest loop costs 20-35 ns per block entry under V8 whether or not its body is emitted, so on JD's 2M entries a frame the block-to-block machinery in `run_jit`, not the body, is where an emitter had to win: blocks that chain to each other inside emitted code rather than returning to the interpreter between every one.

### Compiled blocks jump into each other

A compiled block that leaves at a taken branch or a written terminator tail-calls the target's compiled block through a chain table ([jit.md](jit.md)). Clock, steps and registers match the interpreter (`jit_wasm_check`, and a 6-frame Minecraft capture is byte-identical).

The chain table size decided the result. With 4,096 slots for about 10K compiled Minecraft blocks, only 32% of compiled entries came through a jump and the steady frame did not improve; with 65,536 it was 90%. Two alternating pairs of Minecraft runs (`wasm_bench`, profiled, 6 frames, machine under outside load) put CPU emulation at 44.0 and 43.8 ms a frame against 70.2 and 60.5 unchained (17.0M instructions a frame), with `run_jit` itself down from about 40 to 14 ms. The boot phase gained about 10%. A later 262,144-slot table cut jumps lost to collisions from 46M to 4.6M over a run and measured no faster, but was kept.

### Regions, register locals, and SIMD128

Compiling hot blocks as regions (registers and NZCV in locals, branches between blocks kept inside, PLT-folded calls written out) moved Just Dance 2017's menus (no-draw build, frames 700 to 1100) only from about 217 to 234M instructions/s. A tight emitted loop already ran at 1.4G/s and a call/return loop at 1.16G/s, so neither the per-instruction cost nor the jumps were what held real code back.

The interpreter was: about 8% of instructions sat in blocks the emitter refused, nearly all Advanced SIMD (libvpx's VP8 decoder) and scalar float, and at interpreter speed that 8% was over half the time. Emitting those as SIMD128 and wasm floats took the same window from 142 to 313M instructions/s in an alternated pair; with the machine quiet it now runs at 600M/s, a CPU-only mean of 15 ms a frame. Work per frame was unchanged throughout (6,830,424 instructions).

Declaring every register local up front ran about a third slower than allocating them on first use: Liftoff zero-initializes every declared local on every entry.

## Emitted code size and region entry (October 2026)

- **Minecraft's steady frames are its two "Streaming Pool" threads**: 97% of instructions, reading assets one byte at a time through a virtual `read(buf, 1)`, a trampoline and a PLT `memcpy`. The vsync is 17M cycles and every runnable thread advances the clock, so a frame costs 17M instructions whatever the main thread needs.
- **Regions are large**: median 29 KB of wasm, about 84 bytes per guest op (stores 32%, loads 28%, terminators 12%, add/sub 11%, conditional exits 10%). By 4B instructions of Minecraft V8 had compiled 2,100 regions in Liftoff, lazily on first call on the emulator thread (about 10% of steady time, charged to `enter_emitted`), and tiered 1,409 up for 67 s of background TurboFan CPU. A smaller tiering budget made it slower.
- The hottest region's TurboFan code is 30,000 instructions; its prologue loads about 30 guest registers and spills many straight to the stack.
- Rejected on a guest microbenchmark: registers written through to the `Cpu` and loaded per block (tight loops twice as slow) or on edges between blocks (tight loops 40% slower); one combined deopt test per access with the bitmap always allocated (6% fewer bytes, slower loops). Region caps of 16 and 64 blocks were never compared on a title: see the next point.
- **A comparison harness must load each build from its own directory.** `wasm_bench.mjs` (since deleted) always loaded `target/.../release`, so every two-build title comparison in this round ran one build against itself, and its title verdicts (Minecraft "10 to 20% faster", A Short Hike and Just Dance "unchanged") are void. The microbenchmark loaded each build from its own directory and stands.
- Screenshots of one build differed between runs (hbmenu's clock, all of a Minecraft loading frame), so byte-identical comparison needs a deterministic setup. Under shared load wall-clock A/B runs swung ±20%; the emulator thread's `process.threadCpuUsage` swings less.

## Frames were spent outside the CPU (October 2026)

The emulator thread was saturated while retiring tens of millions of instructions a second, and the device timers covered about a tenth of a run. `perf` on the renderer, with V8's symbol map applied by hand since `perf` will not use it for `[anon:v8]` memory, put CPU emulation at 11% of that thread in A Short Hike. The rest was pixel work on rendered surfaces.

- **2D blits went through guest memory, and every present read back and destroyed every held surface.** Both now stay on the device ([gpu-backend.md](gpu-backend.md)); frames at Short Hike's frame 120, Minecraft's 25 and Just Dance 2019's 60 are byte-identical to a full sync on the native device backend.
- **The "Pipelines" timer included `bind_group`**, which uploads textures: 20 s against 14 ms of real pipeline creation. `createRenderPipeline` costs 0.02 ms on the calling thread; the GPU process does the compiling. Bind groups have their own row now, and are one-time loading: 7.4 s at 60 s and 7.5 s at 200 s.
- **One unemitted instruction refused 48% of hbmenu's frame**: scalar `FCVTZU Sd, Sn` in its pixel loop. Emitting it took `wasm_bench` from 174.9 to 100.5 ms a frame (one run on a busy machine). A Short Hike has no such gap (7,454 of 17M sampled instructions are in refused blocks), so check a title's refusals before assuming either way.

### The clock follows real time in a browser

It counted instructions and jumped over idle time, so a busy title's vsync cost 17M core-0 instructions whatever it needed, and an idle one ran ahead (appstore produced 152 s of audio in about 60 s and the page dropped the rest). Pacing to real time ([multicore.md](multicore.md)) keeps audio level (1.0 s produced per 1.0 s taken), and A Short Hike drew about 690 frames in 140 s against 325. Waiting out an idle jump in full hung A Short Hike at 349M instructions: its threads hold timeouts of over a thousand hours, and nothing serves the other cores during the wait.

## Tuning the IR executor

Emitted regions sit on top of the IR executor (`exec_block`/`exec_op`), which still runs every cold or refused block. About seventy experiments on it (September 2026, `wasm_bench` under V8, hbmenu around 290M instructions/s) came down to this.

Kept, and still in the code:

- `exec_block` and `exec_op` forced inline; out of line, each dropped throughput by 7 to 30%.
- ORR-with-zero copies as `Mov32`/`Mov64`, and an immediate update, compare and branch fused into one `Exit::UpdateCmpImm`.
- The condition-code mask table, unchecked `VRegs` indexing, block-entry counts batched in `run_jit` locals, and translated runs pushed to the trail without the interpreter's single-step merge.
- `+simd128` for the core itself.
- Two-source and CRC forms the interpreter refuses are refused by the translator too: the reserved opcodes 0x0c to 0x0f reached an `unreachable!()`.

Rejected, against structures that are still live:

- **A bigger `Op` enum or a wider arm loses under V8**, whatever it removes. Register-offset `LDR`/`STR W` variants took throughput from about 300M to 127M/s; FP compare and conversion ops, UBFIZ shortcuts, a load-plus-UBFIZ superinstruction, width-split `MADD`/`ADD`/`SUB` and `repr(u8)` on `Op` all lost.
- Block cap: 32, 48, 96 and 128 operations all lost to 64. Lookup table: 16,384 slots lost to 4,096.
- Successor links: strong `Rc`, raw `NonNull`, arena indices, a split PC and pointer, and a run-local cache all lost to `Weak` links (now four per block).
- Unchecked rewrites (exit lookup, op walk, page lookup, XZR read, unaligned reads) were neutral; fast-path flags and caches (watchpoint flags, a last-page cache, a per-block dirty-write flag, a single-exit path) lost.
- Keeping `slice_used` in a local is wrong, not just slow: `yield_thread` and SVC context switches reset it mid-block.
- `wasm-opt -O4` and relaxed SIMD: no gain, and relaxed SIMD lost 20%.

What carries over:

- Under shared load, trust only best-of-three alternated runs of the same artifact; one binary measured anywhere from 122 to 295M/s.
- Check a claimed inlining by diffing the artifacts: forcing four FP accessors inline produced a byte-identical module.
- Byte-identical frames and the differential JIT tests gated every change.
- Since hot code is emitted, executor tuning mostly moves cold and refused blocks. What moved real titles was emitting what had been refused (Regions, above).

## Next

- **A retail title that draws but never a scene.** A Short Hike never reaches a scene ([compat.md](compat.md)); Unity culls on the CPU, so whatever it waits on is above the GPU. Tomodachi Life still shows only its loading screen at frame 300 (8.5B steps).
- **Minecraft regressed.** It now opens the error applet (code 0x208) and faults on a null read at 19.5B steps, 7 frames in; on 2026-10-13 it reached frame 20 ([compat.md](compat.md)).
- **`ncm`**, the last service. Homebrew also still opens a service under an **empty name** (Checkpoint does), and `sm` hands out a working handle instead of failing the way real `sm` would.
- **Hinted fonts render 1-3px wide.** The guest's own FreeType, running on the emulated CPU, gives glyphs from a font with hinting programs (`fpgm`/`prep`/`cvt`) correct heights and advances but collapsed bitmaps, as if untouched points never get interpolated; the same subset built with `tools/make_font.py --no-hinting` renders perfectly, so something the emulator runs under the hinting interpreter is wrong. The stand-in `web/font.ttf` has no hinting; whether the firmware fonts `pl:u` serves do is unchecked, and the bug has not been retested since many decode fixes landed.
