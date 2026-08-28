#!/usr/bin/env python3
"""Differential-test the interpreter's SIMD/FP decode against real ARM semantics.

Assembles a list of instructions into a static AArch64 binary that dumps all 32
vector registers after each one, runs it under qemu-aarch64, runs the identical
instruction bytes through `cargo run --example difftest`, and reports the first
register whose value differs.

This is the tool that caught TRN1/TRN2 taking the wrong lanes (which stalled
hbmenu's NEON JPEG decoder) and the by-element multiplies' index decode.

Needs `clang` (with lld), `qemu-aarch64` and a Rust toolchain. Instructions are
listed in INSTRUCTIONS below; v0..v9 come pre-loaded with the mixed-sign inputs
in `INPUT_VECTORS`, x0 points at those inputs and x3 at 1 KiB of scratch.

    python3 tools/difftest.py            # run the whole list
    python3 tools/difftest.py --keep     # ... and leave the build in /tmp
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTRUCTIONS = [
    # permutes and extracts
    "trn1 v28.8h, v2.8h, v3.8h",
    "trn2 v16.8h, v2.8h, v3.8h",
    "zip1 v10.8h, v2.8h, v3.8h",
    "zip2 v11.8h, v2.8h, v3.8h",
    "uzp1 v12.8h, v2.8h, v3.8h",
    "uzp2 v13.8h, v2.8h, v3.8h",
    "zip1 v14.16b, v2.16b, v3.16b",
    "uzp1 v15.4s, v2.4s, v3.4s",
    "trn1 v2.4s, v28.4s, v29.4s",
    "trn2 v4.4s, v28.4s, v29.4s",
    "ext v9.16b, v2.16b, v3.16b, #5",
    "dup v7.4s, v2.s[1]",
    # integer arithmetic
    "add v18.8h, v4.8h, v8.8h",
    "sub v26.8h, v2.8h, v6.8h",
    "mul v26.8h, v2.8h, v3.8h",
    "sqdmulh v31.4s, v2.4s, v3.4s",
    "cls v25.4s, v2.4s",
    "rev32 v27.8h, v2.8h",
    "uaddlp v23.4s, v2.8h",
    # by-element multiplies
    "smull v18.4s, v18.4h, v0.h[2]",
    "smull2 v19.4s, v18.8h, v0.h[2]",
    "smlal v20.4s, v4.4h, v0.h[3]",
    "smlal2 v21.4s, v4.8h, v0.h[3]",
    "smlsl v22.4s, v4.4h, v0.h[3]",
    "umull v23.4s, v4.4h, v0.h[1]",
    "mul v30.8h, v2.8h, v0.h[4]",
    "mla v31.8h, v2.8h, v0.h[4]",
    "mls v1.8h, v2.8h, v0.h[4]",
    "smull v29.4s, v2.4h, v0.h[7]",
    "smlal2 v30.4s, v2.8h, v0.h[5]",
    # widening / narrowing (three different)
    "saddl v1.4s, v2.4h, v3.4h",
    "uaddl2 v3.4s, v4.8h, v5.8h",
    "ssubl v5.4s, v2.4h, v3.4h",
    "saddw v7.4s, v6.4s, v2.4h",
    "usubw2 v9.4s, v6.4s, v2.8h",
    "sabdl v11.4s, v2.4h, v3.4h",
    "sabal v13.4s, v2.4h, v3.4h",
    "smlsl v15.4s, v2.4h, v3.4h",
    "umull2 v17.4s, v2.8h, v3.8h",
    "addhn v19.4h, v2.4s, v3.4s",
    "subhn2 v21.8h, v2.4s, v3.4s",
    "raddhn v23.4h, v2.4s, v3.4s",
    # shifts
    "sshll v24.4s, v22.4h, #13",
    "sshll2 v25.4s, v22.8h, #13",
    "shrn v26.4h, v18.4s, #16",
    "shrn2 v20.8h, v18.4s, #3",
    "rshrn v21.4h, v18.4s, #3",
    "sqshrn v27.4h, v18.4s, #3",
    "sqrshrn v28.4h, v18.4s, #3",
    "sqrshrn2 v23.8h, v18.4s, #5",
    "sqxtn v29.4h, v18.4s",
    "xtn v29.4h, v2.4s",
    "shll v31.4s, v2.4h, #16",
    "ushr d11, d2, #7",
    "movi v7.4s, #7",
    "mvni v9.4s, #7",
    # floating point
    "fmul v13.4s, v2.4s, v0.s[1]",
    "scvtf v15.4s, v2.4s",
    "fcvtzs v17.4s, v15.4s",
    "ucvtf s19, s2",
    "fcsel s21, s2, s3, ne",
    "fcvtl v1.2d, v2.2s",
    "fcvtn v3.2s, v2.2d",
    # what libjpeg-turbo's NEON colour conversion uses
    "sqxtun v1.8b, v2.8h",
    "sqxtun2 v3.16b, v4.8h",
    "uaddw v5.8h, v6.8h, v2.8b",
    "uaddw2 v7.8h, v6.8h, v2.16b",
    "rshrn2 v9.8h, v18.4s, #11",
    "saddw2 v11.4s, v6.4s, v2.8h",
    # memory forms, last so their side effects do not disturb the others
    "st1 { v2.d }[1], [x3]",
    "ldr q11, [x3]",
    "st1 { v3.s }[2], [x3]",
    "ld1 { v13.8b, v14.8b, v15.8b, v16.8b }, [x0]",
    "ld1 { v17.16b, v18.16b }, [x0], #32",
    "ld1r { v19.4s }, [x3]",
    "ld1 { v21.h }[3], [x3]",
    # the interleaving stores/loads the pixel writer uses
    "st3 { v2.8b, v3.8b, v4.8b }, [x3]",
    "ldr q23, [x3]",
    "ldr q24, [x3, #16]",
    "st4 { v2.8b, v3.8b, v4.8b, v5.8b }, [x3]",
    "ldr q25, [x3]",
    "ldr q26, [x3, #16]",
    "ld3 { v27.8b, v28.8b, v29.8b }, [x3]",
    "ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0]",
    "st3 { v5.16b, v6.16b, v7.16b }, [x3]",
    "ldr q30, [x3, #32]",
]

# Scalar integer instructions. These run in a separate program that dumps
# x0..x25 after each one, because the vector harness needs its own pointers.
# 32-bit forms are included deliberately: every write to a W register zeroes
# bits 63:32, and getting that wrong is invisible until something uses the X
# form (it cost hbmenu's JPEG decode the sign of every DC difference).
SCALAR_INSTRUCTIONS = [
    # shifts by immediate (bitfield aliases) and by register
    "asr w0, w10, #31",
    "asr x1, x10, #63",
    "lsr w2, w10, #4",
    "lsl w3, w10, #4",
    "ror w4, w10, #7",
    "asr w5, w10, w11",
    "lsr w6, w10, w11",
    "lsl w7, w10, w11",
    "ror w8, w10, w11",
    "asr x9, x10, x11",
    # bitfield moves and extends
    "sbfx w0, w10, #4, #8",
    "ubfx w1, w10, #4, #8",
    "sbfiz w2, w10, #4, #8",
    "ubfiz w3, w10, #4, #8",
    "bfi w4, w10, #4, #8",
    "bfxil w5, w10, #4, #8",
    "sxtb w6, w10",
    "sxth w7, w10",
    "sxtw x8, w10",
    "uxtb w9, w10",
    "uxth w0, w10",
    "extr w1, w10, w11, #7",
    "extr x2, x10, x11, #33",
    # conditional select family (flags come from the previous compare)
    "cmp w10, w11",
    "csel w3, w12, w13, gt",
    "csinc w4, w12, w13, gt",
    "csinv w5, w12, w13, gt",
    "csneg w6, w12, w13, gt",
    "cset w7, lt",
    "csetm w8, lt",
    "cinc w9, w12, lt",
    "cneg w0, w12, lt",
    "ccmp w10, #3, #5, ne",
    "ccmn w10, #3, #5, ne",
    "ccmp x10, x11, #7, eq",
    # arithmetic, including the extending and flag-setting forms
    "adds w1, w10, w11",
    "subs w2, w10, w11",
    "adds x3, x10, x11",
    "sbcs w4, w10, w11",
    "adcs w5, w10, w11",
    "add w6, w10, w11, lsl #3",
    "sub w7, w10, w11, asr #2",
    "add x8, x10, w11, sxtw #2",
    "add x9, x10, w11, uxtw #1",
    "sub x0, x10, w11, sxth #3",
    "neg w1, w10",
    "ngc w2, w10",
    # multiply / divide
    "madd w3, w10, w11, w12",
    "msub w4, w10, w11, w12",
    "smaddl x5, w10, w11, x12",
    "umsubl x6, w10, w11, x12",
    "smulh x7, x10, x11",
    "umulh x8, x10, x11",
    "sdiv w9, w10, w11",
    "udiv w0, w10, w11",
    "sdiv x1, x10, x11",
    "udiv x2, x10, x11",
    # bit counting and reversal
    "rbit w3, w10",
    "rbit x4, x10",
    "rev w5, w10",
    "rev16 w6, w10",
    "rev32 x7, x10",
    "clz w8, w10",
    "cls w9, w10",
    "clz x0, x10",
    # MOVK, whose 32-bit form merges into a register it must also narrow.
    # x14 is all-ones, so a `movk w` that forgets to zero bits 63:32 leaves
    # them set and the X form of the result reads wrong.
    "mov x0, x14",
    "movk w0, #0x1234",
    "movk w0, #0x5678, lsl #16",
    "mov x1, x14",
    "movk x1, #0x9abc, lsl #32",
    # logicals with shifted operands
    "and w1, w10, w11, lsr #3",
    "orn w2, w10, w11, asr #4",
    "eor x3, x10, x11, ror #9",
    "bics w4, w10, w11, lsl #2",
    "tst w10, w11",
    "ands w5, w10, w11",
]

# The values loaded into x10..x25 before the scalar tests run.
SCALAR_INPUTS = [
    0xFFFF_FF00,
    0x0000_001F,
    0x8000_0000_0000_0001,
    0x0000_0000_7FFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0x0000_0000_0000_0003,
    0x1234_5678_9ABC_DEF0,
    0xFFFF_FFFF_0000_0000,
    0x0000_0000_0000_0000,
    0x7FFF_FFFF_FFFF_FFFF,
    0x0000_0000_8000_0000,
    0xFEDC_BA98_7654_3210,
    0x0000_0000_0000_00FF,
    0xFFFF_0000_FFFF_0000,
    0x0000_0000_0000_0007,
    0x5555_5555_AAAA_AAAA,
]

SCALAR_DUMP_REGS = 26
SCALAR_DUMP_BYTES = SCALAR_DUMP_REGS * 8

INPUT_VECTORS = [
    [0x0102, 0xFFFE, 0x7FFF, 0x8000, 0x0005, 0xFFF9, 0x1234, 0xABCD],
    [0x0011, 0x0022, 0x0033, 0x0044, 0x0055, 0x0066, 0x0077, 0x0088],
    [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000],
    [0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080],
    [0xFFFF, 0x0000, 0x7FFF, 0x8001, 0x00FF, 0xFF00, 0x0F0F, 0xF0F0],
    [0x3C00, 0x4000, 0xC000, 0x0100, 0x0200, 0x0300, 0x0400, 0x0500],
    [0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x000E, 0x000F, 0x0010],
    [0x8000, 0x8000, 0x7FFF, 0x7FFF, 0x0001, 0xFFFF, 0x0002, 0xFFFE],
    [0x0123, 0x4567, 0x89AB, 0xCDEF, 0xFEDC, 0xBA98, 0x7654, 0x3210],
    [0x0007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000],
]

DUMP_BYTES = 32 * 16


def build_asm(instructions):
    """A program that loads the inputs, then runs each instruction followed by a
    full vector-register dump."""
    body = [f"    ldr q{i}, [x0, #{i * 16}]" for i in range(10)]
    body += [
        "    cmp x2, x2",  # a known flag state, for fcsel
        "    adrp x3, scratch",
        "    add  x3, x3, :lo12:scratch",
    ]
    for insn in instructions:
        body.append(f"    {insn}")
        body += [f"    stp q{i}, q{i + 1}, [x1, #{i * 16}]" for i in range(0, 32, 2)]
        body.append(f"    add x1, x1, #{DUMP_BYTES}")
    total = len(instructions) * DUMP_BYTES
    data = "".join(
        "    .hword " + ", ".join(str(x) for x in v) + "\n" for v in INPUT_VECTORS
    )
    return f"""
    .text
    .global _start
_start:
    adrp x0, inputs
    add  x0, x0, :lo12:inputs
    adrp x1, outbuf
    add  x1, x1, :lo12:outbuf
    mov  x9, x1
{chr(10).join(body)}
    mov x0, #1
    mov x1, x9
    mov x2, #{total}
    mov x8, #64
    svc #0
    mov x0, #0
    mov x8, #93
    svc #0

    .data
    .balign 16
inputs:
{data}
    .balign 16
scratch:
    .space 1024

    .bss
    .balign 16
outbuf:
    .space {total + DUMP_BYTES}
"""


def build_scalar_asm(instructions):
    """A program that dumps x0..x25 after each scalar instruction. x26..x30 are
    reserved for the harness, so the tests only touch x0..x25."""
    body = [f"    ldr x{i}, [x27, #{(i - 10) * 8}]" for i in range(10, 26)]
    body.append("    cmp x10, x10")  # a known flag state to start from
    for insn in instructions:
        body.append(f"    {insn}")
        body += [
            f"    stp x{i}, x{i + 1}, [x28, #{i * 8}]"
            for i in range(0, SCALAR_DUMP_REGS, 2)
        ]
        body.append(f"    add x28, x28, #{SCALAR_DUMP_BYTES}")
    total = len(instructions) * SCALAR_DUMP_BYTES
    data = "".join(f"    .quad {v}\n" for v in SCALAR_INPUTS)
    return f"""
    .text
    .global _start
_start:
    adrp x27, inputs
    add  x27, x27, :lo12:inputs
    adrp x28, outbuf
    add  x28, x28, :lo12:outbuf
    mov  x26, x28
{chr(10).join(body)}
    mov x0, #1
    mov x1, x26
    mov x2, #{total}
    mov x8, #64
    svc #0
    mov x0, #0
    mov x8, #93
    svc #0

    .data
    .balign 16
inputs:
{data}
    .bss
    .balign 16
outbuf:
    .space {total + SCALAR_DUMP_BYTES}
"""


def sections(elf):
    """(address, offset, size) per section name."""
    data = open(elf, "rb").read()
    shoff, = struct.unpack_from("<Q", data, 0x28)
    entsize, count, strndx = struct.unpack_from("<HHH", data, 0x3A)
    names = struct.unpack_from("<Q", data, shoff + strndx * entsize + 0x18)[0]
    out = {}
    for i in range(count):
        base = shoff + i * entsize
        name_idx = struct.unpack_from("<I", data, base)[0]
        end = data.index(b"\0", names + name_idx)
        name = data[names + name_idx:end].decode()
        out[name] = struct.unpack_from("<QQQ", data, base + 0x10)
    return data, out


def run(work, asm_text, instructions, dump_bytes, inputs_bytes, prologue_insns):
    """Assemble, run under qemu, run the same bytes through the interpreter and
    report the first register that differs for each instruction."""
    asm = os.path.join(work, "test.s")
    elf = os.path.join(work, "test.elf")
    open(asm, "w").write(asm_text)
    subprocess.run(
        ["clang", "--target=aarch64-linux-gnu", "-nostdlib", "-static",
         "-fuse-ld=lld", asm, "-o", elf],
        check=True,
    )
    qemu = subprocess.run(["qemu-aarch64", elf], capture_output=True, check=True)

    data, sec = sections(elf)
    _, text_off, text_size = sec[".text"]
    code = data[text_off + prologue_insns * 4:text_off + text_size]
    open(os.path.join(work, "code.bin"), "wb").write(code)
    open(os.path.join(work, "inputs.bin"), "wb").write(inputs_bytes)
    subprocess.run(
        ["cargo", "run", "--quiet", "--release", "-p", "switch-core",
         "--example", "difftest", "--",
         os.path.join(work, "code.bin"), os.path.join(work, "inputs.bin"),
         os.path.join(work, "emu.bin"), hex(sec[".data"][0]),
         str(dump_bytes)],
        cwd=REPO, check=True,
    )

    expected = qemu.stdout
    actual = open(os.path.join(work, "emu.bin"), "rb").read()
    regs = dump_bytes // 16 if dump_bytes % 16 == 0 and dump_bytes >= 512 else dump_bytes // 8
    width = 16 if dump_bytes >= 512 else 8
    failures = 0
    for i, insn in enumerate(instructions):
        want = expected[i * dump_bytes:(i + 1) * dump_bytes]
        got = actual[i * dump_bytes:(i + 1) * dump_bytes]
        if len(got) < dump_bytes:
            print(f"stopped before {insn} (the emulator faulted or ran short)")
            failures += 1
            break
        previous = expected[(i - 1) * dump_bytes:i * dump_bytes] if i else bytes(dump_bytes)
        for reg in range(regs):
            lo, hi = reg * width, (reg + 1) * width
            if want[lo:hi] == previous[lo:hi]:
                continue  # this instruction didn't change it
            if want[lo:hi] != got[lo:hi]:
                kind = "v" if width == 16 else "x"
                print(f"MISMATCH {insn:<34} {kind}{reg}")
                print(f"    qemu {want[lo:hi][::-1].hex() if width == 8 else want[lo:hi].hex()}")
                print(f"    emu  {got[lo:hi][::-1].hex() if width == 8 else got[lo:hi].hex()}")
                failures += 1
    print(f"{len(instructions) - failures}/{len(instructions)} instructions match qemu")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the build directory")
    parser.add_argument("--scalar", action="store_true",
                        help="test the integer instructions instead of the SIMD ones")
    args = parser.parse_args()

    work = tempfile.mkdtemp(prefix="switch-difftest-")
    if args.scalar:
        failures = run(
            work,
            build_scalar_asm(SCALAR_INSTRUCTIONS),
            SCALAR_INSTRUCTIONS,
            SCALAR_DUMP_BYTES,
            b"".join(struct.pack("<Q", v) for v in SCALAR_INPUTS),
            prologue_insns=5,
        )
        if args.keep:
            print("build kept in", work)
        return 1 if failures else 0

    asm = os.path.join(work, "test.s")
    elf = os.path.join(work, "test.elf")
    open(asm, "w").write(build_asm(INSTRUCTIONS))
    subprocess.run(
        ["clang", "--target=aarch64-linux-gnu", "-nostdlib", "-static",
         "-fuse-ld=lld", asm, "-o", elf],
        check=True,
    )
    qemu = subprocess.run(["qemu-aarch64", elf], capture_output=True, check=True)
    open(os.path.join(work, "qemu.bin"), "wb").write(qemu.stdout)

    data, sec = sections(elf)
    text_addr, text_off, text_size = sec[".text"]
    # Skip the prologue that computes x0/x1/x9; the emulator sets them itself.
    code = data[text_off + 20:text_off + text_size]
    open(os.path.join(work, "code.bin"), "wb").write(code)
    inputs = b"".join(struct.pack("<8H", *v) for v in INPUT_VECTORS)
    open(os.path.join(work, "inputs.bin"), "wb").write(inputs)

    subprocess.run(
        ["cargo", "run", "--quiet", "--release", "-p", "switch-core",
         "--example", "difftest", "--",
         os.path.join(work, "code.bin"), os.path.join(work, "inputs.bin"),
         os.path.join(work, "emu.bin"), hex(sec[".data"][0])],
        cwd=REPO, check=True,
    )

    expected = open(os.path.join(work, "qemu.bin"), "rb").read()
    actual = open(os.path.join(work, "emu.bin"), "rb").read()
    failures = 0
    for i, insn in enumerate(INSTRUCTIONS):
        want = expected[i * DUMP_BYTES:(i + 1) * DUMP_BYTES]
        got = actual[i * DUMP_BYTES:(i + 1) * DUMP_BYTES]
        if len(got) < DUMP_BYTES:
            print(f"stopped before {insn} (the emulator faulted or ran short)")
            failures += 1
            break
        previous = expected[(i - 1) * DUMP_BYTES:i * DUMP_BYTES] if i else bytes(DUMP_BYTES)
        # Only the registers this instruction changed are interesting.
        for reg in range(32):
            lo, hi = reg * 16, (reg + 1) * 16
            if want[lo:hi] == previous[lo:hi]:
                continue
            if want[lo:hi] != got[lo:hi]:
                print(f"MISMATCH {insn:<34} v{reg}")
                print(f"    qemu {want[lo:hi].hex()}")
                print(f"    emu  {got[lo:hi].hex()}")
                failures += 1

    print(f"{len(INSTRUCTIONS) - failures}/{len(INSTRUCTIONS)} instructions match qemu")
    if args.keep:
        print("build kept in", work)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
