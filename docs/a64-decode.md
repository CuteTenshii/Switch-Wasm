# A64 decode traps

Encodings that are easy to get wrong, and the guards that get them wrong.
Cross-check any new decode against `llvm-mc -triple=aarch64 -disassemble`
and `tools/difftest.py`.

- **Register 31 in ADD/SUB**: SP in the immediate and extended-register forms,
  XZR in the shifted-register form. `neg x1, x0` is `sub x1, xzr, x0`.
- **A write to a W register zeroes bits 63:32**, and a 32-bit operand must be
  sign-extended from *bit 31* before an arithmetic shift or signed divide —
  masking to 32 bits makes `asr w, w, w` and `sdiv w, w, w` unsigned.
- **BLR reads its target before linking.** `blr x30` is a legal
  return-and-relink; writing x30 first branches to itself+4.
- **A guard that includes a fixed bit kills the whole group.** The scalar-FP
  1-source group is `opcode(6) 10000` (bits[15:10] are `opcode<0>:10000`);
  FCSEL/FCCMP have bit21 *set*; the int↔float conversions read
  `rmode`:`opcode` as bits[21:16], which folds in fixed bit21 and made `ucvtf
  d0, x1` execute as FCVTMU. The 3-source group's top byte is `00011111`, so it
  must match before the `00011110` space. Prove a new guard reaches a real
  encoding.
- **SIMD&FP LDR/STR**: the register-offset form is `bits[25:24]=00` — do *not*
  detect it via bit 21, which is the top bit of `imm12` in the unsigned-offset
  form. Mode 0b00 is not only STUR/LDUR either: bits[11:10] select unscaled /
  post-index / pre-index.
- **AdvSIMD structure loads/stores**: writeback is **bit 23**, and `Rm == 31`
  means "increment by the transfer size" while any other `Rm` is a register
  increment. Single-lane forms spread the index across `Q:S:size`, and
  `scale == 0b11` is `LD1R`, not a doubleword lane insert.
- **The permute trio differ**: TRN interleaves the even (or odd) elements of
  *both* operands, ZIP interleaves one half of each, UZP packs every other
  element of Vn low and Vm's high.
- **BSL/BIT/BIF** differ only in the mask register: BSL selects with Vd, BIT
  and BIF with Vm.
- **EXT** shares bits[28:24] with the permute group, so permute must also
  require bit29 == 0.
- **TBL/TBX** share bits[29:21] with the copy group, so copy must let them past
  (every copy encoding sets bit10; table lookup has bit15 == 0 and
  bits[11:10] == 00). The table is `len+1` registers and **wraps past v31**; an
  out-of-range index reads zero for TBL and leaves the byte alone for TBX.
- **Vector FP** lives in two groups the integer three-same decode must not
  swallow: three-same opcodes from `0b11000` up (bits[23:22] are `a:sz`) and
  two-register misc, whose FP forms need `(U, size<1>, opcode)` together —
  opcode `11101` is SCVTF when `size<1> == 0` and FRECPE when it is 1.
- The AdvSIMD **scalar** forms are separate encodings: shift-by-immediate has
  bit28 set, two-register-misc is `01 U 11110 …`.
- **CTR_EL0** reports `0x8444C004`; cache-flush loops stride by `4 << DminLine`.

