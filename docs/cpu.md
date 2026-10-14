# Instruction semantics

Interpreter details for A64, A32, SIMD and floating point that the code relies on.

## `crates/switch-core/src/cpu/bits.rs`

- `lane`/`set_lane` split the 128-bit register into 64-bit halves: a runtime-distance `u128` shift compiles to a `__lshrti3` call on wasm that V8 cannot inline (about 1% of a browser frame). Every A64 lane is aligned to its width, so no lane straddles the halves.
- Variable shifts (SSHL etc.) take the low 8 bits of the amount sign-extended, not masked to the element width; masking made negative (right) shifts impossible below 64 bits.
- Only scalar FP raises FPSR flags: IOC (converts, divides, square roots), DZC (divides) and IXC (float-to-integer converts). Other FPSR bits are guest-writable storage.

## `crates/switch-core/src/cpu/fp.rs`

- FRECPE/FRSQRTE must use ARM's 8-bit estimate tables, not exact division: titles that compare, hash or round the result branch differently otherwise (`1/1.5` is `0x3f2aaaab` exact vs `0x3f2a8000` on hardware). Negative FRSQRTE input yields the default NaN.
- `fp_form` separates classification from execution so the JIT settles the form once (`Op::Fp`) instead of walking eight guards per execution.
- `f64::powi` is a wasm libcall (`__powidf2`, about 1.4% of a translated frame via fcvtzs bounds); `pow2` builds powers of two from the exponent field instead.
- Past decode bugs: the 1-source opcode low bit is bit 15; FP<->int rmode/opcode must not include bit 21; FCSEL/FCCMP have bit 21 set; FCMP opcode2 is bits 4:0, not bits 9:8.

## `crates/switch-core/src/cpu/simd/`

- Decode ordering matters: AES/SHA before scalar DUP (they share bits 28:21); narrowing shifts before MOVI (`immh == 0` is MOVI); scalar DUP (element) before the vector copy group.
- The copy group (DUP/INS/UMOV/SMOV) is identified by bit 21 == 0, with bit 29 (`op`) free; matching on bit 29 excluded INS (element). libnx's `smEncodeName` builds service names with `ins` chains.
- Narrowing shift destination size comes from all of `immh`, not bit 22.
- FMLA by element is fused at source width (widening to f64 would round twice). FMAXNMV etc. reduce pairwise as (0 op 1) op (2 op 3). FCVTN rounds once via double.
- Scalar and vector FP use Rust IEEE arithmetic; FPCR's rounding mode is honoured only by FRINTX/FRINTI (scalar and vector). Half-precision by-element is out of scope.

## `crates/switch-core/src/cpu/system.rs`

- `CNTPCT_EL0` is the guest clock: `nn::os::GetSystemTick` is `mrs x0, cntpct_el0; ret`, never a syscall. Frequency is 19.2 MHz; one emulated instruction is one 1.02 GHz CPU cycle, so a tick is about 53 instructions.
- `SysOp::of` classification is done once at JIT translation (the hot `MRS TPIDRRO_EL0` sits late in the table). Constant registers fit in 32 bits (`SysReg::Fixed`) to keep `jit::ir::Op` at 16 bytes.
- DCZID_EL0 must report BS=4 (memset loops stride `4 << BS`; BS=0 runs away). CTR_EL0 must report real line sizes, or cache-flush loops walk 4 bytes at a time.
- FPCR/FPSR are op1=3 at EL0. CLREX must clear the exclusive monitor (it is not a no-op barrier).

## `crates/switch-core/src/cpu/a32/mod.rs`

- Horizon runs some retail titles in AArch32 (e.g. Mario Kart 8 Deluxe); bit 0 of the `main.npdm` flags byte at 0x0C (`Is64BitInstruction`) is clear for them. A 32-bit `rtld` starts with `b #+8`, which the A64 decoder reads as a valid `ANDS`.
- One `Cpu` and register file for both states; `ExecMode` is per thread and travels with the context. `r13`/`r14` replace A64's separate SP slot and X30. Q and GE live in `cpsr_q`/`cpsr_ge`.
- T32 is not implemented: MK8D's 4.8M instruction words contain one `BLX` immediate (a literal pool word). Interworking branches to Thumb are reported as errors rather than silently misexecuted.
- AArch32 syscalls: same numbers and low registers as A64, but 64-bit arguments are split across (not always adjacent) register pairs. They are accessors rather than a register shuffle because a blocking syscall is reissued and must re-read unmodified argument registers.
- `set_mode` re-assembles the `bootstrap` trampolines as A32 so `main` returning exits cleanly.

## `crates/switch-core/src/cpu/a32/neon.rs`

- Coverage was chosen from Mario Kart 8 Deluxe's NEON words (register fields masked): the head of a long tail is implemented and the rest is reported by name. The two-register miscellaneous group is complete and checked against `qemu-arm`. Nearly all element/structure loads/stores are `VLD1`/`VST1` of one or two `D` registers.
- Decode notes: in the same-length float group bit 21 selects subtract and bit 20 is `sz`, so the lane count cannot come from `size`. With bit 4 set, it is a modified immediate when bits 21:19 and bit 7 are clear, else a shift (a shift's `imm6` reaches every bits-21:19 pattern). `VEXT` and the two-register group both have 1011 in bits 23:20; bit 24 separates them.
- Shifts by register: a left shift by the lane width or more yields zero (plain) or saturates in the value's direction (saturating forms); a right shift past the lane leaves only the sign.

## `crates/switch-core/src/cpu/a32/loadstore.rs`

- LDMDA (decrement-after) ends at the base, so it starts one word above decrement-before. Getting this wrong broke Mario Kart 8 Deluxe.
- Bits 9:8 of the sync encodings: 11 exclusive, 10 exclusive with acquire/release, 00 plain acquire/release. The plain `LDA`/`STL` forms must not touch the exclusive monitor, and their `Rd` field is 1111 (`pc`) and must not be written. Treating `STLH` as `STREX` dropped stores and broke MK8D's pointer-buffer setup.
- A32 STREX uses the same monitor as the A64 exclusive pairs.

## `crates/switch-core/src/cpu/a32/vfp.rs`

- AArch32 D0..D31 alias A64 V0..V15: `D(2n)` is the low half of `Vn`, `D(2n+1)` the high half, so `Cpu::vregs` backs both states and context switches need no special casing.
- FPSCR rounding bits 23:22 match FPCR. `VCMP` sets FPSCR N/Z/C/V, not the condition flags; `VMRS APSR_nzcv` copies them.
- In the extension register load/store space only bits 24:21 = `0010` is the core-pair `VMOV`. Treating every P=0, W=0 encoding as one ran `vldmia sp, {d14-d15}` as a move into `lr` and `sp`, and Mario Kart 8 Deluxe returned through a zeroed stack.

## `crates/switch-core/tests/a32/mod.rs`

- A32 test encodings are assembled with `llvm-mc -triple=armv7-none-eabi`, never by hand, so tests cannot agree with a decoder mistake. `dead_code` is allowed because each test crate compiles the whole harness module.

## `crates/switch-core/tests/cpu/mod.rs`

- `RequestUpdateAudioRenderer` input strides come from libnx's `audren.h`; input and output entry sizes differ.
