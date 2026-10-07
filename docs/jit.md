# The block translator and emitter

How AArch64 blocks are translated, run, and written out as wasm.

## `crates/switch-core/src/cpu/jit/ir.rs`

- Size budgets (asserted in tests): `Op` is 16 bytes (one 64-bit payload plus tag), which is why `Op::MovK` stores shift+halfword instead of mask+value and `SysReg::Fixed` is a `u32`; `Exit` is 16 bytes with its own tag byte (a niche-hidden tag cost several instructions per taken exit), which is why `Exit::CmpImm` keeps the encoded immediate instead of the inverted `u64`; `Branch` is 24 bytes, its precomputed span fitting in existing padding.
- Specialized variants (`Load64` etc., `Pair` for X registers) exist because the general form dispatched twice (op then access), and each indirect jump mispredicted separately under V8.
- Block links (`LINKS` successors, most recent first) are an inline cache. On a Just Dance 2019 frame: 1 slot linked 73.3% of entries, 2 slots 79.8%, 4 slots 82.5%, 8 slots 82.6%; the remainder is `RET` from functions with many callers.
- Links are `Weak` so a block invalidated by a guest store to its page is unreachable through stale links with no separate invalidation pass.
- Blocks start cold and are counted before emission: most translated blocks run once or twice, so emitting them costs more than interpreting them. A block that keeps handing back at its first instruction is marked never-emit.
- `Block::new` trims `ops`/`words`, which the translator sizes for the longest block a page allows; untrimmed, a full cache was about 80 MiB of mostly empty heap.
- Dropping a block frees its function-table slot; otherwise the table grows by one slot per block ever translated (six figures on retail titles).

## `crates/switch-wasm/src/heap.rs`

- dlmalloc grows linear memory in 64 KiB steps, and guest RAM is backed per 4 KiB page on first touch, so asset loading caused hundreds of `memory.grow`s (about 500 for 32 MiB in Just Dance 2019). Each grow costs V8 time proportional to the number of instances importing the memory (each JIT block is an instance; about 8000 on a retail title): 0.4 s of `SetInstanceMemory` over 60 frames. A 4 MiB granularity cuts that to about eight grows; unused grown memory is not backed until written.

## `tools/jit_wasm_check.mjs`

- Covers what host tests cannot: emitted blocks use wasm32 field offsets, 4-byte page-table entries (8 on the host), and function-table slots as entry points. Each program runs with the translator on and off and must agree on every register and the clock; the `emitted`/`enteredEmitted` stats guard against a build that silently interprets everything, and `expect` guards against a run that faulted immediately.
- Programs cover: plain arithmetic, a store/load through the emitted page-table walk, a fused `cmp`+`b.ne` back edge (the block reports its branch target), and a `tbz` branching somewhere other than the block start (so only the emitted target can put control there).

## `crates/switch-core/src/cpu/jit/exec.rs`

- `HOT` threshold: each emitted block is its own wasm module with its own V8 code region, so entering thousands of scattered modules via one indirect call stalls more than their bodies save. Just Dance 2019: threshold 16 emitted 7,990 blocks and ran 4% slower than pure interpretation; 512 emits 1,185 blocks covering 97% of the same entries and beats the interpreter.
- `MAX_MISSES`: a hand-back at instruction zero retires nothing. One or two are normal (first touch of a soft-mapped page), but a block whose first instruction always needs the slow path (a store to a watched page in a loop) is dropped back to the interpreter. Any progress resets the count.
- The step budget is exact: blocks are a cache, not a unit of execution. Emitted code stops where it decides, so a block that does not fit the remaining budget is interpreted that visit. A budget that splits a fused exit runs the fitting part so progress is never zero (otherwise `run_jit` spins).
- `exec_block` is inlined into `run_jit`: as a call each entry paid a 384-byte V8 frame; inlining took a Just Dance 2019 wasm frame from 240 to 229 ms with no change in host cycle count (host and wasm timings are not related by a constant).
- `Here` (stretch start) is carried instead of a per-op address because the per-op version cost a reload, add and spill under V8 on every op.
- The last-target cache matters: a block boundary falls every ~6.1 instructions on a retail frame and nearly all lead where they led before.
- Clock, step counter, trail, and `pc` are settled at block end or fault, not per instruction. Clock is retired after the terminator, as in `step_inner`: retiring early gave every SVC a tick the interpreter had not spent (sdl-hello diverged by one cycle via a sleep deadline).
- `take_exit` uses one match for all branch kinds; checking "followed?" first cost 6% of a Just Dance 2019 wasm frame. `exec_op`, `take_exit`, `apply_compare`, `exec_term` take references: by value, the compiler loads the 16-byte `Op` and hoists every field's extraction above the jump table (9 loads + 6 shifts per dispatch); by reference hbmenu retired 7.3% fewer host instructions.
- `load_store_fast` makes no calls: V8 spills values live across a call at their definition whether or not the call happens (a load paid six stores). The slow path restarts the instruction from scratch, which works because the fast path commits nothing until it finishes. The PLT-stub fold follows the same rule.
- `BIC`/`ORN`/`EON` invert the shifted operand, not the register (as dynarmic and the ARM ARM do).

## `crates/switch-core/examples/jit_coverage.rs`

- Untranslated coverage is answered statically by `switch_core::cpu::translates` weighted by a real frame's instruction mix, not by host timing (which the old `examples/bench.rs` did). The mix is engine-independent; `jit_difftest` checks that.

## `crates/switch-core/examples/emit_selftest.rs`

- Complements `emit_difftest` (which only covers what a real title runs) with adversarial operands: 0 and -1 divisors, `INT_MIN` dividends, shift amounts at 32/64, all 16 NZCV settings per condition. Needs no title or keys, so it runs in CI.
- Encodings are swept from `cpu::emits` over all 2048 bits-31:21 prefixes rather than hand-written, since a wrong hand encoding silently tests a different instruction. Disassembly is used only for grouping and naming; the interpreter is the reference.
- Buckets are keyed by the instruction's disassembly with registers/immediates replaced; mnemonic plus bit 31 conflated `ldr w`/`ldr x` and left 64-bit and sign-extending loads untested.
- Guest memory: a 17-page window at address 0 (so small seeds are addresses), pages scattered in module memory (prime count) so cross-page accesses must take the boundary path. Fill pattern differs at every load width and sign. Read and write watchpoints, a cached page and two write-protected ranges with a writable gap are armed so every hand-back reason is reachable; the gap is the only place the store envelope test is observable.
- Faulting cases are kept: the expected result is "block retired nothing, state unchanged".
- Register seed sets pair `i64::MIN` with -1 (SDIV overflow, which traps in wasm) and ramp shift amounts around 32 and 64 (A64 takes them modulo operand width, wasm modulo its own). One set targets the last bytes of pages so accesses straddle into a mapped page.

## `crates/switch-core/src/cpu/jit/mod.rs`

- Translation removes decode, not dispatch: a block is a run of pre-decoded `ir::Op`s cached by entry address.
- Blocks continue through conditional branches (`B.cond`, `CBZ/CBNZ`, `TBZ/TBNZ`) as `ir::Exit`s; only an always-leaving instruction ends a block. Measured on hbmenu: average block length 7 -> 13 instructions.
- A block is emitted to wasm after 16 entries (emitting costs more than a few interpreted runs; most blocks run once). Modules are batched: 7,062 blocks compiled in 1.8 ms.
- Emitted modules import the core's linear memory, so guest registers/NZCV are plain loads at fixed `Cpu` offsets. Memory accesses the page table cannot answer alone are handed back to the host, mirroring the interpreter's split. Exclusives (`LDXR`/`STXR`) are not emitted.
- Only the browser build compiles emitted code; host builds translate and interpret. `emit_difftest` and `emit_selftest` cover emitted code under V8.
- `run_jit` keeps step budgets exact by entering a block only if it fits in the remaining budget. A block that hands back at its first instruction counts a miss and is eventually not entered, to avoid looping at the same pc.
- Fidelity: ops call the same helpers as the interpreter; unknown forms become `Op::Interpret`. Mid-block `Interpret` is only allowed for instructions that cannot move the PC except to the next one; the branch/exception/system group is a terminator except `D503xxxx` hints/barriers and `MRS`/`MSR`/cache maintenance.
- Staleness: blocks never span a page; `Memory` reports stored-to translated pages and `jit_block_at` drops their blocks before each lookup.

## `crates/switch-core/src/cpu/jit/cache.rs`

- Two-generation cache: when `blocks` reaches MAX_BLOCKS, it becomes `older` and the previous `older` is dropped. `Jit::get` moves (not copies) blocks found in `older` back into `blocks`, so live code survives any number of rotations and the generations stay disjoint (ceiling 2 * MAX_BLOCKS). Replaced a single-generation flush: Just Dance 2019 hit the cap during boot and retranslated everything (105,340 translated for 39,804 held), losing all links.
- `by_page` lists a block under every page it read. Dropping via one page leaves stale entries under the others: these can only cause spurious retranslation, never a stale block. It is rebuilt from the surviving generation on rotation.
- Invalidation must check both generations, since `get` still reaches `older`. Every drop clears the lookup hints so no hint outlives its block.
- Diagnostics: `linked / executed` is the link-chaining hit rate; `interpreted` vs the run's step count is the share not translated; `entered_emitted / executed` is the emitter's useful coverage.

## `crates/switch-core/tests/jit_test.rs`

- The JIT tests are differential (regs, flags, vectors, pc, retired count, written memory) against the interpreter, on an `llvm-mc` + `ld.lld` corpus that also samples ops the translator deliberately hands back (system registers, scalar FP). A separate test checks the corpus actually executes, since two runs that both fault immediately would also match.
- A parking syscall rewinds pc onto its own `svc`, so an `svc` becomes a block entry only then; `Term::Svc` retires the `svc` before dispatch so a thread switch resumes the outgoing thread after it.
- A 32-slot register file with a pinned-zero slot 31 was measured: no gain native or wasm, so the file stays 31 slots. The XZR test exists to catch anyone retrying it.
- PLT stubs are folded into the calling `BL`; the stub code is assumed fixed but the GOT slot is re-read, and an unmapped slot falls back to running the stub so the fault lands on its `ldr`.

## `tools/emit_difftest.mjs`

- Split from the Rust half because `switch-core` has no dependencies and the host has no wasm engine; the interpreter reference lives in Rust, so the comparison spans two programs. Cases are `run(state) -> i32` modules over a bare `WebAssembly.Memory` at manifest-named offsets, so the tool needs no knowledge of `Cpu`'s layout.
- Guest memory uses the `crate::mem::Memory` shape: a page table of 4-byte entries per 4 KiB page, zero when unmapped (then every access hands back and `run` reports progress). Mapped pages are deliberately placed out of guest order so an access running off one page cannot find its guest neighbour behind it.
- `NO_PC` poisons the pc before each case so a block that should write it and does not is caught. `XZR`'s slot is skipped because a `CMP` fused into its branch is emitted without the discarded register write.
- A `none` mode case means the interpreter's watchpoint saw the access; an emitted block that went ahead anyway would blind the watchpoint. Memory is compared after every non-`exact` case, so a store with right register effects but wrong bytes or address is caught.

## `crates/switch-core/src/cpu/jit/decode.rs`

- `MAX_BLOCK_OPS` is not the binding limit: raising it to 160 moved hbmenu's block entries by 0.2%, since indirect branches and returns end blocks.
- `translates()` gives an exact, target-independent answer; timing two engines on a desktop is not evidence about the browser.
- Following direct `B`s took Just Dance 2019 block entries from 2.13M to 2.00M and 1% off the frame (wasm build). Following `BL` copies the callee into every call site and cost 4% more than it saved on the same frame.
- Running through conditional branches is why blocks exceed basic-block length; `b.cond` alone is 12% of hbmenu's frame. It also put CMP+B.cond in one block so they can fuse. A jump on the last slot is not followed (the block would only move the PC).
- PLT stub folding: cross-module calls otherwise cost two block transitions and four dispatches; on Just Dance 2019 that was 500K of 1.78M block entries (`nn::` lives in the sdk module). The stub's four words must be on one page, the one the block registers. The GOT slot is loaded on every call so rebinding imports still works.
- The hint/barrier group used to shortcut to `Op::Nop`, which made `CLREX` a hint while the interpreter cleared the monitor.

## `crates/switch-core/src/cpu/jit/emit.rs`

- The `Cpu` address is a parameter, not baked in: blocks belong to guest addresses and guest threads share one `Cpu`, but tests build many, and a baked pointer would address a freed one. The module imports host memory, so nothing is copied in or out.
- Everything the translator resolved (width, shift type/distance, extension, condition) is a constant in emitted code; a condition becomes its `CONDITION_MASKS` row shifted by NZCV.
- Coverage is a performance question: a refused block runs on the interpreter, so ops are added one at a time behind `emit_difftest`. `Refused::Op` carries the instruction word that `examples/emit_difftest.rs` ranks; `examples/emit_selftest.rs` sweeps the encoding space with `emits()`.
- Guest access is a four-instruction page-table walk. Soft regions, protection, watchpoints and cache reports are not on this path because `Memory::peek`/`poke` already split off the page-table-answerable part; emitted code makes their checks in their order. The cached-pages bitmap had to change from a `Vec` (unknown pointer/length layout) to a fixed-size bitmap behind one pointer so emitted stores can test it. Exclusives are not emitted (no reservation model).
- An access declines before writing any register, so the interpreter resumes the stopped instruction whole, and resumes by translating at that address rather than re-entering part-way. `defers()` is derived from the emitter (emit under two counts, compare bodies) so it cannot drift from the arms; difftests use it to know which `run` answer is right.
- `LEFT` is needed rather than inferred from a short count because a branch on the last body instruction retires everything yet must not fall into the terminator. A taken branch counts as retired so a block always makes progress. Terminators are not emitted.
- wasm vs A64: 32-bit variable shifts must mask the distance to 32 themselves; div by zero and `i64::MIN / -1` trap in wasm and are guarded with real branches (not `select`, which evaluates both arms); a 32-bit signed division cannot overflow since operands are sign-extended. No 128-bit integers, so `SMULH`/`UMULH` stay interpreted. In `add_with_carry`, a 32-bit op's carry is bit 32 of the untruncated sum, so only 64-bit ops need the explicit carry chain.
- Page-table entries are `Option<Box<[u8; PAGE_SIZE]>>`, guaranteed a bare pointer with null for `None`, which is how emitted code tests unmapped pages; 4 bytes on wasm32 (asserted at compile time).
- The write-protect envelope test is enough inline because every protected range is a module's `.text`; stores inside it hand back. The watched-page bitmap is unallocated until something watches a page (never, without JIT or GPU backend).
- Past bug: a 32-bit `CCMN` read Rm unnarrowed, carrying its top half into the bit-32 carry; `CCMP` hid it because inverting masks again.
- Fusing: a fused compare retires two instructions; misreporting would make the interpreter re-run the compare with stale operands. An exit the walk never reaches would make the guarded instructions run unconditionally, so such blocks are refused.

## `crates/switch-core/src/cpu/jit/wasm.rs`

- Hand-written because `switch-core` has no dependencies and the encoder must compile to wasm and stay small (it runs on every first block visit). No floats, SIMD, globals or multi-memory: unneeded ops would be untestable dead code.
- Signed vs unsigned LEB128 agree only up to 63; a constant 64 written unsigned decodes as -64 and still validates, hence separate names and a test.
- Load/store alignment is a hint the engine may rely on, so it must not exceed real alignment; guest addresses promise none, so callers pass it explicitly.
- The module imports the emulator's memory (minimum zero pages, since the host has sized it) so blocks reach guest state with static-offset loads.

## `crates/switch-wasm/src/jit.rs`

- Emitted blocks go into this module's own function table and are entered with `call_indirect` (a Rust fn pointer on wasm32 is a table index), avoiding a JS frame per block entry (a retail frame enters a block every ~6 guest instructions).
- Synchronous `WebAssembly.Module` is required (blocks are requested inside a run slice); windows refuse >4 KiB synchronously but workers allow any size, and blocks are a few hundred bytes.
- Freed slots are recycled: the cache drops blocks constantly, which would otherwise hold six figures of unreachable compiled code. A freed slot still points at its old function (no null function entry), so memory is bounded by the peak number installed. Slot 0 is the linker's null and means "not installed"; it must never be handed out. Failure to compile returns 0 and the block stays interpreted forever.
- `enable` is called before a session runs, so builds that never create one never touch `WebAssembly`.

## `crates/switch-core/src/cpu/jit/host.rs`

- Emitted blocks are called through the module's function table (an `Entry` is a table index on wasm32, a function address on host), not through a JS import: a retail frame enters a block every ~6.1 instructions, so a host call per entry would cost more than the dispatch the JIT removes. The JS boundary is crossed only at install time.
- `switch-core` cannot reach `WebAssembly.Module` (zero deps), so the embedder supplies `install`/`release`; `switch-wasm` uses `wasm_bindgen::memory` and `wasm_bindgen::function_table`.
- `install` failure is never an error: the block stays on the interpreter. The first `set_jit_host` wins because entries from one host mean nothing to another.
