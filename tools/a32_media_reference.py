#!/usr/bin/env python3
"""Print the expected-value table of `crates/switch-core/tests/a32_media_test.rs`.

Runs each case under `qemu-arm`; needs `llvm-mc`, `clang` with `lld`, and `qemu-arm`.

    python3 tools/a32_media_reference.py > table.rs

A case in OPS is assembly that reads r1..r4 and leaves its result in r0 (or r3/r4).
"""

import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

OPS = [[f"{prefix}{op} r0, r1, r2"]
       for prefix in ["s", "q", "sh", "u", "uq", "uh"]
       for op in ["add16", "asx", "sax", "sub16", "add8", "sub8"]]
OPS += [
    ["sadd8 r9, r1, r2", "sel r0, r1, r2"],
    ["usub16 r9, r1, r2", "sel r0, r1, r2"],
    ["pkhbt r0, r1, r2"],
    ["pkhbt r0, r1, r2, lsl #8"],
    ["pkhtb r0, r1, r2, asr #8"],
    ["pkhtb r0, r1, r2, asr #32"],
    ["ssat16 r0, #9, r1"],
    ["ssat16 r0, #16, r2"],
    ["usat16 r0, #7, r1"],
    ["usat16 r0, #0, r2"],
    ["usat16 r0, #15, r1"],
    ["smlald r3, r4, r1, r2"],
    ["smlaldx r3, r4, r1, r2"],
    ["smlsld r3, r4, r1, r2"],
    ["smlsldx r3, r4, r1, r2"],
    ["usad8 r0, r1, r2"],
    ["usada8 r0, r1, r2, r3"],
]

# r1, r2, r3, r4: signs, carries and saturation in every lane somewhere.
INPUTS = [
    (0x80FF7F01, 0x7F0180FF, 0x0001FFFF, 0xFFFFFFFF),
    (0x12345678, 0x9ABCDEF0, 0x80000000, 0x7FFFFFFF),
    (0xFFFF0000, 0x0001FFFF, 0x00000000, 0x00000000),
]


def encode(line):
    out = subprocess.run(
        ["llvm-mc", "-triple=armv8a-none-eabi", "-show-encoding"],
        input=line + "\n", capture_output=True, text=True, check=True,
    ).stdout
    octets = re.findall(r"0x([0-9a-f]{2})", out.split("encoding:")[1])
    return int("".join(reversed(octets)), 16)


def main():
    cases = [(ops, regs) for ops in OPS for regs in INPUTS]
    asm = [".syntax unified", ".arm", ".global _start", "_start:", "ldr r10, =buf"]
    for ops, (n, m, a, b) in cases:
        asm += [f"ldr r1, ={n:#x}", f"ldr r2, ={m:#x}", f"ldr r3, ={a:#x}",
                f"ldr r4, ={b:#x}", "mov r0, #0", "mvn r8, #0", "mov r7, #0",
                # Every GE flag set before the case runs, as in the test.
                "usub8 r9, r7, r7"]
        asm += ops
        asm += ["sel r6, r8, r7", "stmia r10!, {r0, r3, r4, r6}",
                "b 1f", ".ltorg", "1:"]
    size = 16 * len(cases)
    asm += ["mov r0, #1", "ldr r1, =buf", f"ldr r2, ={size}", "mov r7, #4",
            "svc #0", "mov r0, #0", "mov r7, #1", "svc #0", ".ltorg",
            ".bss", f"buf: .space {size}"]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "reference.s"
        binary = Path(tmp) / "reference"
        source.write_text("\n".join(asm) + "\n")
        subprocess.run(["clang", "--target=armv8a-linux-gnueabihf", "-nostdlib",
                        "-static", "-fuse-ld=lld", "-o", str(binary), str(source)],
                       check=True)
        raw = subprocess.run(["qemu-arm", str(binary)], capture_output=True,
                             check=True).stdout

    out = sys.stdout
    out.write("const CASES: &[(&[u32], [u32; 4], [u32; 4])] = &[\n")
    for i, (ops, (n, m, a, b)) in enumerate(cases):
        r0, r3, r4, ge = struct.unpack_from("<4I", raw, 16 * i)
        words = ", ".join(f"0x{encode(op):08X}" for op in ops)
        out.write(f"    // {'; '.join(ops)}\n")
        out.write(f"    (&[{words}], [0x{n:08X}, 0x{m:08X}, 0x{a:08X}, 0x{b:08X}], "
                  f"[0x{r0:08X}, 0x{r3:08X}, 0x{r4:08X}, 0x{ge:08X}]),\n")
    out.write("];\n")


if __name__ == "__main__":
    main()
