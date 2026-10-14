# The block translator and emitter

How AArch64 blocks are translated, run, and written out as wasm.

## `crates/switch-core/src/cpu/jit/mod.rs`

- Translation removes decode, not dispatch: a block is a run of pre-decoded `ir::Op`s cached by entry address.
- Blocks continue through conditional branches (`B.cond`, `CBZ/CBNZ`, `TBZ/TBNZ`) as `ir::Exit`s; only an always-leaving instruction ends a block. On hbmenu this took the average block from 7 to 13 instructions.
- A block is compiled to wasm after `HOT` entries, together with the warm blocks it branches to (its region). Each region is one module, compiled and installed on its own by `exec.rs`'s `install`.
- Emitted modules import the core's linear memory, so guest registers and NZCV are plain loads at fixed `Cpu` offsets. Memory accesses the page table cannot answer alone are handed back to the host, mirroring the interpreter's split.
- Only the browser build compiles emitted code; host builds translate and interpret. `emit_difftest` and `emit_selftest` cover emitted code under V8.
- Fidelity: ops call the same helpers as the interpreter; unknown forms become `Op::Interpret`. Mid-block `Interpret` is only allowed for instructions that cannot move the PC except to the next one: the branch and exception group is a terminator, the `0xD5` system group (hints, barriers, `MRS`/`MSR`, cache maintenance) is not.
- Staleness: a block lists every page it read (it can span pages through followed `B`s and page boundaries); `Memory` reports stores to translated pages and `jit_block_at` drops their blocks before each lookup.
- Caches are per core and per process: `Cpu::jit` holds the running process's blocks, and `enter_process` swaps it with that process's entry in `parked_jits`. `forget_translations` drops an ended process's cache, `jit_flush` clears the parked ones too, and `run_jit` drops its held block when the pid changes, so one process's code is never linked to another's.
- The dirty-page list lives in the process's shared `Memory` and draining it empties it, so only one cache per process sees each store; two cores running one process would need their own reports.

## `crates/switch-core/src/cpu/jit/ir.rs`

- Size budgets (asserted in tests): `Op` is 16 bytes (one 64-bit payload plus tag), which is why `Op::MovK` stores shift+halfword instead of mask+value and `SysReg::Fixed` is a `u32`; `Exit` is 16 bytes with its own tag byte (a niche-hidden tag cost several instructions per taken exit), which is why `Exit::CmpImm` keeps the encoded immediate instead of the inverted `u64`; `Branch` is 24 bytes, its precomputed span fitting in existing padding.
- Specialized variants (`Load64`, `PairLoad64`, `PairStore64` etc.) exist because the general form dispatched twice (op then access), and each indirect jump mispredicted separately under V8.
- Block links (`LINKS` successors, most recent first) are an inline cache. On a Just Dance 2019 frame 1 slot linked 73% of entries and 4 slots 82.5%; 8 slots added nothing, the remainder being `RET` from functions with many callers.
- Links are `Weak` so a block invalidated by a guest store to its page is unreachable through stale links with no separate invalidation pass.
- Blocks start cold and are counted before emission: most translated blocks run once or twice, so emitting them costs more than interpreting them. A block that keeps handing back at its first instruction loses its compiled form for good.
- `Block::new` trims `ops`/`words`, which the translator sizes for the longest block; untrimmed, a full cache was about 80 MiB of mostly empty heap.
- Dropping a block frees its function-table slot and clears its region members; otherwise the table grows by one slot per block ever compiled.

## `crates/switch-core/src/cpu/jit/decode.rs`

- `MAX_BLOCK_OPS` is not the binding limit: raising it to 160 moved hbmenu's block entries by 0.2%, since indirect branches and returns end blocks.
- `translates()` gives an exact, target-independent answer; timing two engines on a desktop is not evidence about the browser.
- Direct `B`s are followed (cut entries and about 1% of a wasm frame). Following `BL` copies the callee into every call site and cost 4% more than it saved. A jump on the last slot is not followed (the block would only move the PC).
- Running through conditional branches also puts `CMP`+`B.cond` in one block so they can fuse.
- PLT stub folding: a cross-module call otherwise costs two block transitions and four dispatches (on Just Dance 2019, 500K of 1.78M block entries; `nn::` lives in the sdk module). The stub's four words must be on one page, which the block lists. The GOT slot is loaded on every call so rebinding imports still works.
- The `0xD5` group goes through `decode_system`, not a `Nop` shortcut: the shortcut made `CLREX` a hint while the interpreter cleared the monitor.

## `crates/switch-core/src/cpu/jit/exec.rs`

- `HOT` threshold: each region is its own wasm module with its own V8 code region, so entering thousands of scattered modules via one indirect call stalls more than their bodies save. Measured before regions, with one module per block, on Just Dance 2019 a threshold of 16 emitted 7,990 blocks and ran 4% slower than pure interpretation; 512 emitted 1,185 covering 97% of the same entries and beat the interpreter.
- `MAX_MISSES`: a hand-back at instruction zero retires nothing. One or two are normal (first touch of a soft-mapped page), but a block whose first instruction always needs the slow path (a store to a watched page in a loop) is dropped back to the interpreter. Any progress resets the count.
- The step budget is exact: blocks are a cache, not a unit of execution. A compiled block that does not fit the remaining budget is interpreted that visit. A budget that splits a fused exit runs the fitting part so progress is never zero (otherwise `run_jit` spins).
- `exec_block` is inlined into `run_jit`: as a call each entry paid a 384-byte V8 frame (about 5% of a wasm frame) with no change in host cycle count; host and wasm timings are not related by a constant.
- `Here` (stretch start) is carried instead of a per-op address because the per-op version cost a reload, add and spill under V8 on every op.
- The held block and its links matter: a block boundary falls every ~6 instructions on a retail frame and nearly all lead where they led before.
- Chain fuel is the point where `run_jit` would stop following blocks anyway (the budget less `MAX_WHOLE`, the end of the time slice, the next timed-wait sweep), so a chain retires exactly what entering each block from here would. Branches inside a region and jumps between regions both charge it. On return the left blocks' clock and steps (the fuel spent) are added and the block that answered is settled as if entered directly; one that stopped at its first instruction gets `pc` set to its start and is left for the next visit.
- Regions: when a block turns hot, the cached blocks its known branches reach (conditional targets, `B`, the fall-through of a cut block, a `BL`'s target and return point, a register branch's linked successors) that have run `REGION_WARM` times or are compiled already are emitted with it, breadth-first, up to `MAX_REGION_BLOCKS`. A member that refuses is dropped and the region re-emitted; one too long truncates it. Members are held by the head's `members`, which `run`'s answer indexes from 1.
- Only the block a region or chain stops in records trail runs; blocks left through a branch inside a region or a jump to another region do not.
- Folded PLT calls are written terminators: the stub's four instructions count in `whole`, so `MAX_WHOLE` reserves them, and `finish_emitted` records the stub as its own run.
- A JIT-against-interpreter lockstep cannot run far: the interpreter preempts at any instruction and the JIT only between blocks, so the first time-slice expiry inside a block already switches threads differently. Two JIT builds that chain differently diverge the same way. Just Dance 2017 stalls around frame 250 under some such schedules (a priority-43 thread parsing JSON while main waits on a lock) and not under others, so a stall that a one-instruction `TIME_SLICE` change makes disappear is a scheduling problem, not an emitter bug.
- Clock, step counter, trail and `pc` are settled at block end or fault, not per instruction. The clock is retired after the terminator, as in `step_inner`: retiring early gave every SVC a tick the interpreter had not spent (a homebrew title diverged by one cycle via a sleep deadline).
- `take_exit` uses one match for all branch kinds; checking "followed?" first cost 6% of a wasm frame. `exec_op`, `take_exit`, `apply_compare` and `exec_term` take references: by value, the compiler loads the 16-byte `Op` and hoists every field's extraction above the jump table (9 loads and 6 shifts per dispatch).
- `load_store_fast` makes no calls: V8 spills values live across a call at their definition whether or not the call happens (a load paid six stores). The slow path restarts the instruction from scratch, which works because the fast path commits nothing until it finishes. The PLT-stub fold follows the same rule.
- `BIC`/`ORN`/`EON` invert the shifted operand, not the register (as dynarmic and the ARM ARM do).

## `crates/switch-core/src/cpu/jit/cache.rs`

- Two-generation cache: when `blocks` reaches `MAX_BLOCKS`, it becomes `older` and the previous `older` is dropped. `Jit::get` moves (not copies) blocks found in `older` back into `blocks`, so live code survives any number of rotations and the generations stay disjoint (ceiling 2 * `MAX_BLOCKS`). A single-generation flush made Just Dance 2019 hit the cap during boot and retranslate everything, losing all links.
- `lookup` is a direct-mapped hint (`LOOKUP_SLOTS`) checked against the block's entry address.
- `by_page` lists a block under every page it read. Dropping via one page leaves stale entries under the others: these can only cause spurious retranslation, never a stale block. It is rebuilt from the surviving generation on rotation.
- Invalidation must check both generations, since `get` still reaches `older`. Every drop clears the lookup hints so no hint outlives its block.
- Diagnostics: `linked / executed` is the link hit rate; `interpreted` against the run's step count is the share not translated; `entered_emitted / executed` is the emitter's useful coverage; `chained` counts jumps between compiled regions.
- The chain table maps entry addresses to compiled entries for emitted code, direct-mapped, with `CHAIN_SLOTS` slots. Small tables lose most jumps to collisions: with about 10K compiled blocks on a Minecraft frame, 4,096 slots chained 32% of compiled entries. A vacant slot holds an address that hashes to another slot, so no lookup matches it.
- A region's head is listed under its members' pages too (`cover`, and the rotation rebuild through `region_pages`), so a store to any of them drops the region. Every block that leaves the cache drops its compiled form and its member list, since heads may hold each other and the `Rc` cycle would otherwise outlive the cache. A member that leaves the cache while its region lives keeps only its translation, which the region's answers still name.
- A block leaves the chain table when it leaves the cache (invalidation, rotation, clear) or loses its compiled form, which is before its function-table slot can be reused. A block compiled after it left the cache is never chained to.

## `crates/switch-core/src/cpu/jit/emit.rs`

- The `Cpu` address is a parameter, not baked in: blocks belong to guest addresses and guest threads share one `Cpu`, but tests build many, and a baked pointer would address a freed one. The module imports host memory, so nothing is copied in or out.
- Everything the translator resolved (width, shift type and distance, extension, condition) is a constant in emitted code; a condition becomes its `CONDITION_MASKS` row shifted by NZCV.
- Coverage is a performance question: a refused block runs on the interpreter, so ops are added one at a time behind `emit_difftest`. `Refused::Op` carries the instruction word that `examples/emit_difftest.rs` ranks; `examples/emit_selftest.rs` sweeps the encoding space with `emits()`.
- Guest access is a four-instruction page-table walk. Soft regions, protection, watchpoints and cache reports are not on this path because `Memory::peek`/`poke` already split off the page-table-answerable part; emitted code makes their checks in their order. The cached-pages bitmap is a fixed-size array behind one pointer (not a `Vec`, whose layout is unknown) so emitted stores can test it. Exclusives are not emitted (no reservation model).
- An access declines before writing any register, so the interpreter resumes the stopped instruction whole, and resumes by translating at that address rather than re-entering part-way. `defers()` is derived from the emitter (emit under two counts, compare bodies) so it cannot drift from the arms; difftests use it to know which `run` answer is right.
- `LEFT` is needed rather than inferred from a short count because a branch on the last body instruction retires everything yet must not fall into the terminator. A taken branch counts as retired so a block always makes progress. `B`, `BL`, `BR`, `BLR`, `RET` and PLT-folded calls are emitted and leave with `LEFT` (`writes_term`); `SVC` and the rest stay with `exec_term`.
- A region is one function: a `loop` whose head dispatches on a block index with `br_table`, each block falling into the next when that is where it goes. Forward branches inside it are plain `br`s to the target's label; only backward ones go through the `br_table`, an indirect jump shared by every edge.
- A register branch (`BR`, `BLR`, `RET`, a folded PLT call) compares its target with the region's blocks among its linked successors and, for a `RET`, with the return points of the region's calls, and branches inside on a match. Without this every call and return left the region: a guest loop doing one virtual call per byte ran at 0.48G instructions/s, and 1.85G with it. Every exit stores its answer and branches to one epilogue, so the stores are written once. `run`'s answer is the stopping block's retired count, its index above `MEMBER_SHIFT`, and `LEFT`; a lone block answers as it always did, which keeps the difftest harnesses unchanged.
- Guest registers and NZCV live in locals: the prologue loads every slot the region touches, the epilogue stores every slot it writes. Locals are allocated per slot on first use; declaring all 34 up front made every entry pay V8's zero-initialization and ran about a third slower. The zero register reads as a constant and the discard slot is never stored.
- `Layout` offsets for registers, NZCV, FPSR, `pc`, fuel and vector registers are `offset_of!` into `Cpu`. Guest memory state is not in `Cpu`: `Layout.memory` is where the core's `AddressSpace` keeps its `*const Memory` (`offset_of!(Cpu, mem) + AddressSpace::MEMORY_AT`), and the remaining fields are offsets into `Memory` (`mem::Offsets::OF_MEMORY`).
- The page-table pointer, watchpoints and watched-page bitmap pointer change only in host code, so the prologue loads the `Memory` address once and then each of these a region uses into an `L_ENV` local, rather than per access.
- Chained regions leave a taken exit or written terminator with `return_call_indirect` into the target's compiled region from the epilogue, when the chain table has it and the fuel covers what the stopping block retired. The jump stores the fuel, counts itself and records the target. A tail call keeps the native stack flat however long the chain, and costs little: what costs in real code is what stays interpreted.
- Advanced SIMD (`emit/simd.rs`) is written as SIMD128, decoded from the instruction word: three-same, three-different (long and wide), shift by immediate including the narrowing shifts, two-register misc (narrowing, ABS/NEG, CNT/NOT), permutes, EXT, DUP/INS/SMOV/UMOV, MOVI/MVNI/ORR/BIC immediate, and SIMD&FP loads and stores (LDR/STR of B to Q, S, D and Q pairs, LD1/ST1 of one lane, LD1R). Vector registers stay in memory. Rounding shifts are `(x >> s) + ((x >> (s - 1)) & 1)`, which never leaves the lane; wasm's narrows treat their input as signed, which is right for the signed-source forms and for an unsigned source already shifted right by at least one, and UQXTN clamps with `min_u` first. It more than doubled the instruction rate on Just Dance 2017's VP8 menus.
- Scalar floating point (`emit/fp.rs`) is wasm `f32`/`f64` where the interpreter's arm is IEEE arithmetic. Single-precision FMADD and its variants copy the interpreter's `f32::mul_add`, which is libm's `fmaf` on wasm32: the sum in double precision, nudged one unit when it lies exactly halfway between two singles but is inexact. Double FMADD stays interpreted. Float-to-integer converts and FDIV set FPSR's cumulative bits as the interpreter does, so `Layout` carries FPSR and both difftests seed and compare it.
- wasm vs A64: 32-bit variable shifts must mask the distance to 32 themselves; div by zero and `i64::MIN / -1` trap in wasm and are guarded with real branches (not `select`, which evaluates both arms); a 32-bit signed division cannot overflow since operands are sign-extended. No 128-bit integers, so `SMULH`/`UMULH` stay interpreted. In `add_carry`, a 32-bit op's carry is bit 32 of the untruncated sum, so only 64-bit ops need the explicit carry chain.
- Page-table entries are `AtomicPtr`s, null when unmapped, which is how emitted code tests unmapped pages; 4 bytes on wasm32 (asserted at compile time).
- Write-protected pages are set in the watched-page bitmap (`mark_readonly`, and again after a host write reports them), so a store tests one bit for protection and cached contents alike. The bitmap is unallocated until something watches or protects a page.
- Watchpoint tests are emitted only while one is armed: `run_jit` drops compiled code when that changes. Dropping them and the envelope test made a straight-line loop of loads and stores about 1.8 times faster.
- Past bug: a 32-bit `CCMN` read Rm unnarrowed, carrying its top half into the bit-32 carry; `CCMP` hid it because inverting masks again.
- Fusing: a fused compare retires two instructions; misreporting would make the interpreter re-run the compare with stale operands. An exit the walk never reaches would make the guarded instructions run unconditionally, so such blocks are refused (`Refused::ControlFlow`).

## `crates/switch-core/src/cpu/jit/wasm.rs`

- Hand-written because `switch-core` has no dependencies and the encoder must compile to wasm and stay small (it runs for every region compiled). No globals or multi-memory: unneeded ops would be untestable dead code. SIMD opcodes are passed through `simd()` with the numbers named where they are used.
- Signed vs unsigned LEB128 agree only up to 63; a constant 64 written unsigned decodes as -64 and still validates, hence separate names and a test.
- Load/store alignment is a hint the engine may rely on, so it must not exceed real alignment; guest addresses promise none, so callers pass it explicitly.
- The module imports the emulator's memory as `e`.`m` with a minimum of zero pages, since the host has sized it. When the core is built with atomics the import is declared shared with a 65,536-page maximum, because a shared memory only matches an import that says so.

## `crates/switch-core/src/cpu/jit/host.rs`

- Compiled regions are called through the embedder's function table (an `Entry` is a table index on wasm32, a function address on host), not through a JS import: a host call per entry would cost more than the dispatch the JIT removes. The JS boundary is crossed only at install time.
- `switch-core` cannot reach `WebAssembly.Module` (zero dependencies), so the embedder supplies `install`/`release` and whether the engine has tail calls and SIMD128.
- `install` failure is never an error: the block stays on the interpreter. The first `set_jit_host` wins because entries from one host mean nothing to another.

## `crates/switch-wasm/src/jit.rs`

- Regions go into this instance's own function table and are entered with `call_indirect` (a Rust fn pointer on wasm32 is a table index). Tables are per instance, so with several workers on one memory each has its own table and its own free list (`FREE` is thread-local).
- Synchronous `WebAssembly.Module` is required (regions are compiled inside a run slice); windows refuse large modules synchronously but workers allow any size.
- Freed slots are recycled: the cache drops blocks constantly. A freed slot still points at its old function (no null function entry), so memory is bounded by the peak number installed. Slot 0 is the linker's null and means "not installed"; it must never be handed out. Failure to compile returns 0 and the block stays interpreted.
- `attach` runs from `switch_init` right after instantiation. It probes tail calls (`tail_call_probe`) and SIMD128 (`simd_probe`) with `WebAssembly.validate`: without tail calls regions are emitted unchained, without SIMD128 Advanced SIMD stays interpreted. Every module is offered the function table as `e`.`t`; unchained ones do not import it.

## `crates/switch-wasm/src/heap.rs`

- The global allocator is dlmalloc with a 4 MiB growth granularity instead of 64 KiB. Guest RAM is backed per 4 KiB page on first touch, so asset loading caused hundreds of `memory.grow`s (about 500 for 32 MiB in Just Dance 2019), and each grow cost V8 time per instance importing the memory (every compiled block): 0.4 s of `SetInstanceMemory` over 60 frames, measured on the unshared build. Unused grown memory is not backed until written.
- The allocator has no lock. Its `SAFETY` comment still says the module is built without atomics, but the wasm build now has `+atomics` and `--shared-memory`, and core workers instantiate over the same memory, so it is only sound while one worker at a time allocates.

## `tools/jit_wasm_check.mjs`

- Covers what host tests cannot: emitted blocks use wasm32 field offsets, 4-byte page-table entries (8 on the host), and function-table slots as entry points. Each program runs with the translator on and off and must agree on every register and the clock; the `emitted`/`enteredEmitted` stats guard against a build that silently interprets everything, and `expect` guards against a run that faulted immediately.
- Programs cover plain arithmetic, a store/load through the emitted page-table walk, a fused `cmp`+`b.ne` back edge, a `tbz` branching somewhere other than the block start, and `bl`/`blr`/`ret` calls.
- It also fails unless some compiled block jumped into another, since a build without tail calls would pass every comparison unchained.

## `tools/emit_difftest.mjs`

- Split from the Rust half because `switch-core` has no dependencies and the host has no wasm engine; the interpreter reference lives in Rust, so the comparison spans two programs. Cases are `run(state) -> i32` modules over a bare `WebAssembly.Memory` at manifest-named offsets, so the tool needs no knowledge of `Cpu`'s layout.
- The manifest is split into lines from its bytes: a selftest manifest outgrew the longest string V8 can hold (512 MiB).
- `examples/emit_difftest.rs` takes `FROM_FRAME=<n>` to sample a title's later code. It maps no guest memory, so a folded PLT call (which reads its GOT slot) is not reported as covered.
- Guest memory uses the `crate::mem::Memory` shape: a page table of 4-byte entries per 4 KiB page, zero when unmapped (then every access hands back and `run` reports progress). Mapped pages are deliberately placed out of guest order so an access running off one page cannot find its guest neighbour behind it.
- `NO_PC` poisons the pc before each case so a block that should write it and does not is caught. The discard slot is skipped because a `CMP` fused into its branch is emitted without the discarded register write.
- A `none` mode case means the interpreter's watchpoint saw the access; an emitted block that went ahead anyway would blind the watchpoint. Memory is compared after every non-`exact` case, so a store with right register effects but wrong bytes or address is caught.

## `crates/switch-core/examples/emit_selftest.rs`

- Complements `emit_difftest` (which only covers what a real title runs) with adversarial operands: 0 and -1 divisors, `INT_MIN` dividends, shift amounts at 32/64, all 16 NZCV settings per condition. Needs no title or keys, so it can run anywhere; CI does not run it.
- Encodings are swept from `cpu::emits` over all 2048 bits-31:21 prefixes rather than hand-written, since a wrong hand encoding silently tests a different instruction. Disassembly is used only for grouping and naming; the interpreter is the reference. The disassembler names no Advanced SIMD form, so those are bucketed by their encoding with Rd, Rn and Rm masked.
- The sweep that added SIMD found two interpreter bugs, both confirmed on an arm64 host: ORR/BIC (vector, immediate) replaced the register instead of modifying it, and SIMD&FP register-offset loads read Rm 31 as SP instead of XZR.
- Buckets are keyed by the instruction's disassembly with registers and immediates replaced; mnemonic plus bit 31 conflated `ldr w`/`ldr x` and left 64-bit and sign-extending loads untested.
- Guest memory: a 17-page window at address 0 (so small seeds are addresses), pages scattered in module memory (prime count) so cross-page accesses must take the boundary path. The fill pattern differs at every load width and sign. Read and write watchpoints, a cached page and two write-protected pages with writable pages between are armed so every hand-back reason is reachable.
- Faulting cases are kept: the expected result is "block retired nothing, state unchanged".
- Register seed sets pair `i64::MIN` with -1 (SDIV overflow, which traps in wasm) and ramp shift amounts around 32 and 64 (A64 takes them modulo operand width, wasm modulo its own). One set targets the last bytes of pages so accesses straddle into a mapped page.

## `crates/switch-core/examples/jit_coverage.rs`

- Untranslated coverage is answered statically by `switch_core::cpu::translates` weighted by a real frame's instruction mix, not by host timing.

## `crates/switch-core/tests/jit_test.rs`

- The JIT tests are differential (registers, flags, vectors, pc, retired count, written memory) against the interpreter, on an `llvm-mc` + `ld.lld` corpus that also samples ops the translator deliberately hands back (system registers, scalar FP). A separate test checks the corpus actually executes, since two runs that both fault immediately would also match.
- A parking syscall rewinds pc onto its own `svc`, so an `svc` becomes a block entry only then; `Term::Svc` retires the `svc` before dispatch so a thread switch resumes the outgoing thread after it.
- PLT stubs are folded into the calling `BL`; the stub code is assumed fixed but the GOT slot is re-read, and an unreadable slot falls back to running the stub so the fault lands on its `ldr`.
- `jit_emitted_test.rs` runs the emitted path on the host with a fake `JitHost` whose entries are Rust functions doing what the block's first `retired` ops would.
