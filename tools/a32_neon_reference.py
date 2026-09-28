#!/usr/bin/env python3
"""Print the expected-value table of `crates/switch-core/tests/a32_neon_reference_test.rs`.

Each NEON case runs under `qemu-arm` on the same three vectors the test loads
into q8, q9 and q10, with r2 holding 0x9abcdef0, and the table records all
four afterwards: the destination, the source a permute rewrites, and the core
register a lane move reads or writes. Needs `llvm-mc`,
`clang` with `lld`, and `qemu-arm` (user-mode) on the PATH.

    python3 tools/a32_neon_reference.py > table.rs

To cover another instruction, add a line of assembly to OPS that reads q8
(or d16) and writes q10 (or d20), or a list of lines for a case that needs
more than one, such as a store read back.
"""

import re
import struct
import subprocess
import tempfile
from pathlib import Path

OPS = []
OPS += [f"vrev64.{s} q10, q8" for s in ("8", "16", "32")]
OPS += [f"vrev32.{s} q10, q8" for s in ("8", "16")]
OPS += ["vrev16.8 q10, q8"]
OPS += [f"vpaddl.{t}{s} q10, q8" for t in "su" for s in ("8", "16", "32")]
OPS += [f"vpadal.{t}{s} q10, q8" for t in "su" for s in ("8", "16", "32")]
OPS += [f"vcls.s{s} q10, q8" for s in ("8", "16", "32")]
OPS += [f"vclz.i{s} q10, q8" for s in ("8", "16", "32")]
OPS += ["vcnt.8 q10, q8", "vmvn q10, q8"]
OPS += [f"{op}.s{s} q10, q8" for op in ("vqabs", "vqneg") for s in ("8", "16", "32")]
OPS += [f"{op}.{t} q10, q8, #0"
        for op in ("vcgt", "vcge", "vceq", "vcle", "vclt")
        for t in ("s8", "s16", "s32", "f32")]
OPS += [f"{op}.{t} q10, q8" for op in ("vabs", "vneg") for t in ("s8", "s16", "s32", "f32")]
OPS += ["vswp q10, q8"]
OPS += [f"{op}.{s} q10, q8" for op in ("vtrn", "vuzp", "vzip") for s in ("8", "16", "32")]
OPS += [f"vmovn.i{s} d20, q8" for s in ("16", "32", "64")]
OPS += [f"vqmovun.s{s} d20, q8" for s in ("16", "32", "64")]
OPS += [f"vqmovn.{t}{s} d20, q8" for t in "su" for s in ("16", "32", "64")]
OPS += ["vshll.i8 q10, d16, #8", "vshll.i16 q10, d16, #16", "vshll.i32 q10, d16, #32"]
OPS += ["vcvt.f16.f32 d20, q8", "vcvt.f32.f16 q10, d16"]
OPS += ["vrecpe.f32 q10, q8", "vrsqrte.f32 q10, q8", "vrecpe.u32 q10, q8", "vrsqrte.u32 q10, q8"]
OPS += ["vcvt.f32.s32 q10, q8", "vcvt.f32.u32 q10, q8", "vcvt.s32.f32 q10, q8",
        "vcvt.u32.f32 q10, q8"]
# The integer three-register forms at every size, signed and unsigned.
for op in ("vhadd", "vqadd", "vrhadd", "vhsub", "vqsub", "vcgt", "vcge", "vshl", "vqshl",
           "vrshl", "vqrshl", "vmax", "vmin", "vabd", "vaba"):
    OPS += [f"{op}.{t}{s} q10, q8, q9" for t in "su" for s in ("8", "16", "32")]
OPS += [f"{op}.{t}64 q10, q8, q9" for op in ("vqadd", "vqsub", "vshl", "vqshl", "vrshl", "vqrshl")
        for t in "su"]
OPS += [f"{op}.{t}{s} d20, d16, d18" for op in ("vpmax", "vpmin") for t in "su"
        for s in ("8", "16", "32")]
OPS += [f"vpadd.i{s} d20, d16, d18" for s in ("8", "16", "32")]
OPS += [f"{op}.s{s} q10, q8, q9" for op in ("vqdmulh", "vqrdmulh") for s in ("16", "32")]
OPS += ["vmul.p8 q10, q8, q9"]
# The shifts by an immediate: right, rounding, accumulating and inserting,
# left and saturating left, narrowing, widening, and fixed point, at the
# smallest and largest amount of every size.
for t, s, amounts in (("8", 8, (1, 8)), ("16", 16, (1, 16)), ("32", 32, (1, 32)),
                      ("64", 64, (1, 64))):
    for n in amounts:
        OPS += [f"{op}.{u}{t} q10, q8, #{n}" for op in ("vshr", "vsra", "vrshr", "vrsra")
                for u in "su"]
        OPS += [f"vsri.{t} q10, q8, #{n}"]
    for n in (0, s - 1):
        OPS += [f"vshl.i{t} q10, q8, #{n}", f"vsli.{t} q10, q8, #{n}",
                f"vqshl.s{t} q10, q8, #{n}", f"vqshl.u{t} q10, q8, #{n}",
                f"vqshlu.s{t} q10, q8, #{n}"]
for t, s in (("16", 8), ("32", 16), ("64", 32)):
    for n in (1, s):
        OPS += [f"vshrn.i{t} d20, q8, #{n}", f"vrshrn.i{t} d20, q8, #{n}",
                f"vqshrun.s{t} d20, q8, #{n}", f"vqrshrun.s{t} d20, q8, #{n}",
                f"vqshrn.s{t} d20, q8, #{n}", f"vqshrn.u{t} d20, q8, #{n}",
                f"vqrshrn.s{t} d20, q8, #{n}", f"vqrshrn.u{t} d20, q8, #{n}"]
for t, s in (("8", 8), ("16", 16), ("32", 32)):
    OPS += [f"vmovl.s{t} q10, d16", f"vmovl.u{t} q10, d16", f"vshll.s{t} q10, d16, #1",
            f"vshll.u{t} q10, d16, #{s - 1}"]
OPS += [f"vcvt.{a}.{b} q10, q8, #{n}" for a, b in (("f32", "s32"), ("f32", "u32"),
        ("s32", "f32"), ("u32", "f32")) for n in (1, 16, 32)]
# Three registers of different lengths: long, wide and narrow.
for t in ("s8", "s16", "s32", "u8", "u16", "u32"):
    OPS += [f"{op}.{t} q10, d16, d18" for op in ("vaddl", "vsubl", "vabal", "vabdl", "vmlal",
                                                 "vmlsl", "vmull")]
    OPS += [f"{op}.{t} q10, q8, d18" for op in ("vaddw", "vsubw")]
OPS += [f"{op}.i{t} d20, q8, q9" for op in ("vaddhn", "vraddhn", "vsubhn", "vrsubhn")
        for t in ("16", "32", "64")]
OPS += [f"{op}.s{t} q10, d16, d18" for op in ("vqdmlal", "vqdmlsl", "vqdmull")
        for t in ("16", "32")]
OPS += ["vmull.p8 q10, d16, d18"]
# Moves between a core register and a lane, VDUP, and the interleaved
# structure loads and stores, each store read back into q10.
OPS += ["vmov.8 d20[5], r2", "vmov.16 d21[3], r2", "vmov.32 d20[1], r2",
        "vmov.s8 r2, d16[3]", "vmov.u8 r2, d17[7]", "vmov.s16 r2, d16[1]",
        "vmov.u16 r2, d17[2]", "vmov.32 r2, d17[1]",
        "vdup.8 q10, r2", "vdup.16 d20, r2", "vdup.32 q10, r2"]
OPS += [f"vld2.{s} {{d20, d21}}, [r0]" for s in ("8", "16", "32")]
OPS += [f"vld2.{s} {{d20, d22}}, [r0]" for s in ("8", "16", "32")]
OPS += ["vld2.16 {d18, d19, d20, d21}, [r0]", "vld3.8 {d18, d19, d20}, [r0]",
        "vld3.16 {d17, d19, d21}, [r0]", "vld4.32 {d18, d19, d20, d21}, [r0]",
        "vld4.8 {d16, d18, d20, d22}, [r0]"]
OPS += [[f"{st} {regs}, [r0]", "vld1.32 {d20, d21}, [r0]"]
        for st, regs in (("vst2.8", "{d16, d17}"), ("vst2.32", "{d16, d18}"),
                         ("vst3.16", "{d16, d17, d18}"), ("vst4.8", "{d16, d17, d18, d19}"))]
# The crypto extension on A32.
OPS += ["aese.8 q10, q8", "aesd.8 q10, q8", "aesmc.8 q10, q8", "aesimc.8 q10, q8",
        "sha1h.32 q10, q8", "sha1su1.32 q10, q8", "sha256su0.32 q10, q8",
        "sha1c.32 q10, q8, q9", "sha1p.32 q10, q8, q9", "sha1m.32 q10, q8, q9",
        "sha1su0.32 q10, q8, q9", "sha256h.32 q10, q8, q9", "sha256h2.32 q10, q8, q9",
        "sha256su1.32 q10, q8, q9"]
# The modified immediates, every cmode and op: moves, inversions, and the
# ORR and BIC forms that read the destination.
OPS += ["vmov.i32 q10, #0x5a", "vmov.i32 q10, #0x5a00", "vmov.i32 q10, #0x5a0000",
        "vmov.i32 q10, #0x5a000000", "vmov.i16 q10, #0x5a", "vmov.i16 q10, #0x5a00",
        "vmov.i32 q10, #0x5aff", "vmov.i32 q10, #0x5affff", "vmov.i8 q10, #0xa5",
        "vmov.i64 q10, #0xff00ff0000ffff00", "vmov.f32 q10, #1.0", "vmov.f32 q10, #-0.1875",
        "vmvn.i32 q10, #0x5a00", "vmvn.i16 q10, #0x5a", "vmvn.i32 q10, #0x5affff",
        "vorr.i32 q10, #0x5a00", "vorr.i16 q10, #0x5a00", "vbic.i32 q10, #0x5a000000",
        "vbic.i16 q10, #0x5a", "vmov.i8 d20, #0x3c"]

# q8, q9 and q10 as four words each: sign extremes and carries for the
# integer forms, and for the float forms a normal, a negative, the largest
# finite value, a denormal, a NaN, an infinity and a signed zero.
INPUTS = [
    ((0x80FF7F01, 0x00010203, 0xFFFF8000, 0x7FFFFFFF),
     (0x12345678, 0x9ABCDEF0, 0x0F0F0F0F, 0xF0F0F0F0),
     (0x11111111, 0x22222222, 0x33333333, 0x44444444)),
    ((0x3F800000, 0xC0200000, 0x7F7FFFFF, 0x000116C2),
     (0x7FC00001, 0xFF800000, 0x3E99999A, 0x004C4B40),
     (0x11111111, 0x22222222, 0x33333333, 0x44444444)),
    ((0x7FC00001, 0xFF800000, 0x80000000, 0x3F000000),
     (0xC1480000, 0x4B800001, 0x7F800000, 0x00800000),
     (0x00000000, 0x00000000, 0x00000000, 0x00000000)),
]


def encode(line):
    out = subprocess.run(
        ["llvm-mc", "-triple=armv8a-none-eabi", "-mattr=+neon,+fp16,+crypto", "-show-encoding"],
        input=line + "\n", capture_output=True, text=True, check=True,
    ).stdout
    octets = re.findall(r"0x([0-9a-f]{2})", out.split("encoding:")[1])
    return int("".join(reversed(octets)), 16)


def main():
    cases = [(op if isinstance(op, list) else [op], regs) for op in OPS for regs in INPUTS]
    asm = [".syntax unified", ".arm", ".fpu crypto-neon-fp-armv8", ".global _start", "_start:",
           "ldr r1, =out"]
    data = [".data", "in:"]
    for i, (ops, (q8, q9, q10)) in enumerate(cases):
        # Sixteen zero bytes after each case's vectors: a structure load or
        # store reaches up to 32 bytes from q10's copy, and the test's
        # scratch page is zero there too.
        data.append(".word " + ", ".join(f"{w:#x}" for w in q8 + q9 + q10 + (0, 0, 0, 0)))
        asm += [f"ldr r0, =in + {64 * i}", "vld1.32 {d16, d17, d18, d19}, [r0]!",
                "vld1.32 {d20, d21}, [r0]", "movw r2, #0xdef0", "movt r2, #0x9abc"]
        asm += ops
        asm += ["vst1.32 {d16, d17, d18, d19}, [r1]!", "vst1.32 {d20, d21}, [r1]!",
                "str r2, [r1], #16", "b 1f", ".ltorg", "1:"]
    size = 64 * len(cases)
    asm += ["mov r0, #1", "ldr r1, =out", f"ldr r2, ={size}", "mov r7, #4", "svc #0",
            "mov r0, #0", "mov r7, #1", "svc #0", ".ltorg"]
    asm += data + [".bss", f"out: .space {size}"]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "reference.s"
        binary = Path(tmp) / "reference"
        source.write_text("\n".join(asm) + "\n")
        subprocess.run(["clang", "--target=armv8a-linux-gnueabihf", "-mfpu=crypto-neon-fp-armv8",
                        "-nostdlib", "-static", "-fuse-ld=lld", "-o", str(binary),
                        str(source)], check=True)
        raw = subprocess.run(["qemu-arm", str(binary)], capture_output=True,
                             check=True).stdout

    print("const CASES: &[(&[u32], [u32; 12], [u32; 13])] = &[")
    for i, (ops, (q8, q9, q10)) in enumerate(cases):
        after = struct.unpack_from("<13I", raw, 64 * i)
        before = ", ".join(f"0x{w:08X}" for w in q8 + q9 + q10)
        got = ", ".join(f"0x{w:08X}" for w in after)
        words = ", ".join(f"0x{encode(op):08X}" for op in ops)
        print(f"    // {'; '.join(ops)}")
        print(f"    (&[{words}], [{before}], [{got}]),")
    print("];")


if __name__ == "__main__":
    main()
