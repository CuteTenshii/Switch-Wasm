#!/usr/bin/env python3
"""Generate T32 snippets, run them under `qemu-system-arm`, and write the images and
results that `examples/thumb_difftest.rs` replays in the interpreter.

    python3 tools/thumb_difftest.py [--out DIR] [--cases N] [--seed S]
    cargo run --release -p switch-core --example thumb_difftest -- --dir DIR

Needs `llvm-mc`, `llvm-objcopy` and `qemu-system-arm`. Each snippet runs from a seeded
register file and flags, with a 256-byte buffer in memory, and leaves r0-r10, the flags,
SP and the buffer's checksums; both sides must agree on all of it.
"""

import os
import argparse
import random
import struct
import subprocess
import tempfile
from pathlib import Path

LLVM = "/opt/homebrew/opt/llvm/bin/"
BASE = 0x40200000
BUF = 0x41000000
RESULTS = 0x42000000
CASE = 64


def reg(rng, high=False):
    return f"r{rng.randrange(11 if high else 8)}"


def imm_t32(rng):
    b = rng.randrange(256)
    return rng.choice([b, b << 16 | b, b << 24 | b << 8, b * 0x01010101,
                       (0x80 | b & 0x7F) << rng.randrange(1, 24)])


SHIFTS = ["lsl", "lsr", "asr", "ror"]


def data_processing(rng):
    r = lambda: reg(rng, True)
    l = lambda: reg(rng)
    op = rng.choice(["add", "sub", "and", "eor", "orr", "bic", "adc", "sbc", "rsb", "orn", "mov", "mvn", "cmp", "cmn", "tst", "teq"])
    s = rng.choice(["", "s"]) if op not in ("cmp", "cmn", "tst", "teq") else ""
    kind = rng.randrange(5)
    d, n, m = r(), r(), r()
    dn, nn, mn = l(), l(), l()
    if op in ("cmp", "cmn", "tst", "teq"):
        regs = (n, ) 
    if kind == 0:
        return f"{op}{s}.w {d}, {n}, #{imm_t32(rng)}" if op not in ("mov", "mvn", "cmp", "cmn", "tst", "teq") else (f"{op}{s}.w {d}, #{imm_t32(rng)}" if op in ("mov", "mvn") else f"{op}.w {n}, #{imm_t32(rng)}")
    if kind == 1:
        sh = f", {rng.choice(SHIFTS)} #{rng.randrange(1, 32)}" if rng.randrange(4) else ", rrx"
        if op in ("mov", "mvn"):
            return f"{op}{s}.w {d}, {m}{sh}"
        if op in ("cmp", "cmn", "tst", "teq"):
            return f"{op}.w {n}, {m}{sh}"
        return f"{op}{s}.w {d}, {n}, {m}{sh}"
    if kind == 2 and op in ("add", "sub", "and", "eor", "orr", "bic", "adc", "sbc", "mov", "mvn", "cmp", "cmn", "tst", "rsb"):
        # Narrow forms; the assembler picks the encoding.
        if op == "rsb":
            return f"rsbs {dn}, {nn}, #0"
        if op == "mov":
            return f"movs {dn}, {mn}"
        if op == "mvn":
            return f"mvns {dn}, {mn}"
        if op in ("cmp", "cmn", "tst"):
            return f"{op} {dn}, {mn}"
        return f"{op}s {dn}, {dn}, {mn}"
    if kind == 3:
        return rng.choice([
            f"adds {dn}, {nn}, #{rng.randrange(8)}", f"subs {dn}, {nn}, #{rng.randrange(8)}",
            f"adds {dn}, #{rng.randrange(256)}", f"subs {dn}, #{rng.randrange(256)}",
            f"movs {dn}, #{rng.randrange(256)}", f"cmp {dn}, #{rng.randrange(256)}",
            f"adds {dn}, {nn}, {mn}", f"subs {dn}, {nn}, {mn}",
            f"{rng.choice(['lsls', 'lsrs', 'asrs'])} {dn}, {nn}, #{rng.randrange(1, 32)}",
            f"{rng.choice(['lsls', 'lsrs', 'asrs', 'rors'])} {dn}, {dn}, {mn}",
            f"muls {dn}, {nn}, {dn}", f"add {rng.choice(['r8', 'r9', 'r10'])}, {mn}",
            f"mov {rng.choice(['r8', 'r9', 'r10'])}, {mn}", f"cmp {rng.choice(['r8', 'r9'])}, {mn}",
            f"{rng.choice(['lsl', 'lsr', 'asr', 'ror'])}{s}.w {d}, {n}, {m}",
            f"{rng.choice(SHIFTS)}{s}.w {d}, {n}, #{rng.randrange(1, 32)}",
            f"movw {d}, #{rng.randrange(65536)}", f"movt {d}, #{rng.randrange(65536)}",
            f"addw {d}, {n}, #{rng.randrange(4096)}", f"subw {d}, {n}, #{rng.randrange(4096)}",
            f"mul {d}, {n}, {m}",
        ])
    return f"{op}{s}.w {d}, {n}, {m}" if op not in ("mov", "mvn", "cmp", "cmn", "tst", "teq") else (f"{op}{s}.w {d}, {m}" if op in ("mov", "mvn") else f"{op}.w {n}, {m}")


def media(rng):
    r = lambda: reg(rng, True)
    d, n, m, a = r(), r(), r(), r()
    lsb = rng.randrange(32)
    width = rng.randrange(1, 33 - lsb)
    return rng.choice([
        f"ubfx {d}, {n}, #{lsb}, #{width}", f"sbfx {d}, {n}, #{lsb}, #{width}",
        f"bfi {d}, {n}, #{lsb}, #{width}", f"bfc {d}, #{lsb}, #{width}",
        f"ssat {d}, #{rng.randrange(1, 33)}, {n}", f"usat {d}, #{rng.randrange(32)}, {n}",
        f"ssat {d}, #{rng.randrange(1, 33)}, {n}, asr #{rng.randrange(1, 32)}",
        f"usat {d}, #{rng.randrange(32)}, {n}, lsl #{rng.randrange(1, 32)}",
        f"ssat16 {d}, #{rng.randrange(1, 17)}, {n}", f"usat16 {d}, #{rng.randrange(16)}, {n}",
        f"{rng.choice(['sxtb', 'sxth', 'uxtb', 'uxth', 'sxtb16', 'uxtb16'])}{rng.choice(['', '.w'])} {d}, {m}{rng.choice(['', ', ror #8', ', ror #16', ', ror #24'])}",
        f"{rng.choice(['sxtab', 'sxtah', 'uxtab', 'uxtah', 'sxtab16', 'uxtab16'])} {d}, {n}, {m}{rng.choice(['', ', ror #8', ', ror #16'])}",
        f"{rng.choice(['rev', 'rev16', 'revsh', 'rbit', 'clz'])}.w {d}, {m}" if rng.randrange(2) else f"{rng.choice(['rev', 'rev16', 'revsh'])} {reg(rng)}, {reg(rng)}",
        f"{rng.choice(['qadd', 'qsub', 'qdadd', 'qdsub'])} {d}, {n}, {m}",
        f"{rng.choice(['s', 'q', 'sh', 'u', 'uq', 'uh'])}{rng.choice(['add16', 'asx', 'sax', 'sub16', 'add8', 'sub8'])} {d}, {n}, {m}",
        f"pkhbt {d}, {n}, {m}, lsl #{rng.randrange(32)}", f"pkhtb {d}, {n}, {m}, asr #{rng.randrange(1, 33)}",
        f"usad8 {d}, {n}, {m}", f"usada8 {d}, {n}, {m}, {a}",
        f"{rng.choice(['mla', 'mls'])} {d}, {n}, {m}, {a}",
        f"{rng.choice(['umull', 'smull', 'umlal', 'smlal', 'umaal'])} {d}, {a}, {n}, {m}" if d != a else f"umull {d}, r10, {n}, {m}",
        f"{rng.choice(['sdiv', 'udiv'])} {d}, {n}, {m}",
        f"{rng.choice(['smulbb', 'smulbt', 'smultb', 'smultt', 'smulwb', 'smulwt'])} {d}, {n}, {m}",
        f"{rng.choice(['smlabb', 'smlabt', 'smlatb', 'smlatt', 'smlawb', 'smlawt'])} {d}, {n}, {m}, {a}",
        f"{rng.choice(['smlad', 'smladx', 'smlsd', 'smlsdx', 'smmla', 'smmlar', 'smmls', 'smmlsr'])} {d}, {n}, {m}, {a}",
        f"{rng.choice(['smuad', 'smuadx', 'smusd', 'smusdx', 'smmul', 'smmulr'])} {d}, {n}, {m}",
        f"{rng.choice(['smlald', 'smlaldx', 'smlsld', 'smlsldx', 'smlalbb', 'smlaltt'])} {d}, {a}, {n}, {m}" if d != a else f"smlald {d}, r10, {n}, {m}",
    ]).replace("r10, r10", "r10, r9")


def memory(rng):
    """A load or store, with roles fixed so every address lands in the buffer:
    r8, r9, r4 or r1 the base, r6 a small index, and the rest the data."""
    n = rng.choice(["r1", "r4", "r8", "r9"])
    x = "r6"
    t, t2 = rng.sample([r for r in ["r0", "r2", "r3", "r5", "r7"] if r != n], 2)
    off = rng.choice([0, 4, 8, 0x10, 0x1c, rng.randrange(0, 0x30)])
    small = rng.randrange(1, 0x20)
    word = rng.randrange(0, 0x10) * 4
    sizes = ["ldr", "str", "ldrb", "strb", "ldrh", "strh", "ldrsb", "ldrsh"]
    lo = sorted({rng.choice(["r0", "r2", "r3", "r5", "r7"]) for _ in range(rng.randrange(1, 5))}, key=lambda r: int(r[1:]))
    hi = sorted({rng.choice(["r0", "r2", "r3", "r5", "r7", "r8", "r9"]) for _ in range(rng.randrange(1, 5))} - {n}, key=lambda r: int(r[1:])) or ["r0"]
    neg = rng.choice(["", "-"])
    shift = rng.choice(["", ", lsl #1", ", lsl #2", ", lsl #3"])
    kinds = [
        f"{rng.choice(sizes)} {t}, [{n}, #{off}]",
        f"{rng.choice(sizes)}.w {t}, [{n}, #{off}]",
        f"{rng.choice(sizes[:6])} {t}, [{n}, #-{small}]",
        f"{rng.choice(sizes)} {t}, [{n}, #{neg}{small}]!",
        f"{rng.choice(sizes)} {t}, [{n}], #{neg}{small}",
        f"{rng.choice(sizes)} {t}, [{n}, {x}{shift}]",
        f"{rng.choice(sizes)} {t}, [{n}, {x}]",
        f"{rng.choice(['ldrd', 'strd'])} {t}, {t2}, [{n}, #{neg}{word}]",
        f"{rng.choice(['ldrd', 'strd'])} {t}, {t2}, [{n}], #{neg}{word}",
        f"{rng.choice(['ldrd', 'strd'])} {t}, {t2}, [{n}, #{neg}{word}]!",
        f"{rng.choice(['ldm', 'stm', 'ldmia', 'stmia', 'ldmdb', 'stmdb'])}{rng.choice(['', '.w'])} {n}{rng.choice(['', '!'])}, {{{', '.join(hi)}}}",
        f"{rng.choice(['ldm', 'stm'])} {n}{rng.choice(['', '!'])}, {{{', '.join(r for r in lo if r != n) or 'r0'}}}",
        f"{rng.choice(['push', 'pop'])} {{{', '.join(lo)}}}",
        "push.w {r1, r8, r9}", "pop.w {r2, r8, r9}",
        f"{rng.choice(['ldr', 'str'])} {t}, [sp, #{rng.randrange(0, 0x10) * 4}]",
        f"add {t}, sp, #{rng.randrange(0, 0x10) * 4}", f"add sp, sp, #{rng.randrange(0, 0x10) * 4}",
        f"sub sp, sp, #{rng.randrange(0, 0x10) * 4}", "adr r0, 1f; nop; .balign 4; 1:",
        f"ldrex {t}, [{n}]", f"ldrex {t}, [{n}]; strex r10, {t}, [{n}]", f"strex {t2}, {t}, [{n}]",
        f"ldrexb {t}, [{n}]; strexb {t2}, {t}, [{n}]", f"ldrexh {t}, [{n}]; strexh {t2}, {t}, [{n}]",
        f"ldrexd r2, r3, [{n}]; strexd r5, r2, r3, [{n}]" if n not in ("r2", "r3", "r5") else "ldrexd r2, r3, [r8]; strexd r5, r2, r3, [r8]",
        "clrex; dmb ish; dsb ish; isb",
    ]
    return rng.choice(kinds), {int(n[1:]): BUF + 0x60, 6: rng.randrange(0, 0x10) * 4}


def branches(rng):
    if os.environ.get("BODY"):
        return os.environ["BODY"]
    cond = rng.choice(["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"])
    return rng.choice([
        f"cmp r0, r1; b{cond} 1f; adds r2, r2, #1; 1: adds r3, r3, #1",
        f"cmp r0, r1; b{cond}.w 1f; adds r2, r2, #1; nop; 1: adds r3, r3, #1",
        f"cbz r{rng.randrange(8)}, 1f; adds r2, r2, #1; 1: adds r3, r3, #1",
        f"cbnz r{rng.randrange(8)}, 1f; adds r2, r2, #1; 1: adds r3, r3, #1",
        "b 1f; adds r2, r2, #1; 1: adds r3, r3, #1",
        "b.w 1f; adds r2, r2, #1; 1: adds r3, r3, #1",
        "push {lr}; bl 1f; b 2f; 1: adds r0, r0, #5; bx lr; 2: pop {pc}",
        "push {lr}; adr r4, 1f; adds r4, #1; blx r4; b 2f; .balign 4; 1: adds r0, r0, #5; bx lr; 2: pop {pc}",
        "adr r4, 1f; adds r4, #1; bx r4; adds r2, r2, #1; .balign 4; 1: adds r3, r3, #1",
        "push {r4, lr}; adr r4, 1f; adds r4, #1; mov lr, r4; pop {r4, pc}; nop; .balign 4; 1: adds r3, r3, #1; bx lr",
        f"movs r4, #{rng.randrange(3)}; tbb [pc, r4]; 0: .byte (1f-0b)/2, (2f-0b)/2, (3f-0b)/2, 0; .balign 2; 1: adds r0, #1; b 4f; 2: adds r1, #2; b 4f; 3: adds r2, #3; 4:",
        f"movs r4, #{rng.randrange(3)}; tbh [pc, r4, lsl #1]; 0: .hword (1f-0b)/2, (2f-0b)/2, (3f-0b)/2; 1: adds r0, #1; b 4f; 2: adds r1, #2; b 4f; 3: adds r2, #3; 4:",
        "adr r4, 1f; mov pc, r4; adds r2, r2, #1; .balign 4; 1: adds r3, r3, #1",
        "adr r4, 1f; adds r4, #1; push {r4}; pop {pc}; adds r2, r2, #1; .balign 4; 1: adds r3, r3, #1",
        "adr r4, 1f; adds r4, #1; push {r4}; ldr pc, [sp], #4; adds r2, r2, #1; .balign 4; 1: adds r3, r3, #1",
        "adr r4, 1f; adds r4, #1; push {r4}; pop.w {pc}; adds r2, r2, #1; .balign 4; 1: adds r3, r3, #1",
        "adr r4, 1f; bx r4; .balign 4; .arm; 1: adds r0, r0, #1; adr r5, 2f; orr r5, r5, #1; bx r5; .thumb; .balign 2; 2:",
        "push {lr}; adr r4, 1f; blx r4; b 2f; .balign 4; .arm; 1: adds r0, r0, #1; bx lr; .thumb; 2: pop {pc}",
        "ldr r0, 1f; b 2f; .balign 4; 1: .word 0x12345678; 2:", "ldr.w r0, 1f; b 2f; .balign 4; 1: .word 0x87654321; 2:",
        "ldrb.w r0, 1f; b 2f; .balign 4; 1: .word 0x87654321; 2:", "ldrh.w r0, 1f; b 2f; .balign 4; 1: .word 0x87654321; 2:",
        "ldrsb.w r0, 1f; b 2f; .balign 4; 1: .word 0x87654381; 2:", "ldrsh.w r0, 1f; b 2f; .balign 4; 1: .word 0x87658321; 2:",
        "nop; nop.w; yield; sev; adr.w r5, 1f; adr r6, 1f; .balign 4; 1:",
        f"mov r4, pc; adds r4, #{rng.randrange(1, 8)}; mov r5, pc; mrs r6, apsr; and r6, r6, #0xf8000000; msr apsr_nzcvq, r0",
    ])


def simd(rng):
    a, b, d = rng.sample(range(8), 3)
    return rng.choice([
        f"vmov s{a}, r1; vmov s{b}, r2; {rng.choice(['vadd', 'vsub', 'vmul', 'vdiv'])}.f32 s{d}, s{a}, s{b}; vmov r0, s{d}",
        f"vmov d{a}, r1, r2; vmov d{b}, r3, r4; {rng.choice(['vadd', 'vsub', 'vmul'])}.f64 d{d}, d{a}, d{b}; vmov r0, r5, d{d}",
        f"vmov s{a}, r1; vcvt.s32.f32 s{d}, s{a}; vmov r0, s{d}", f"vmov s{a}, r1; vcvt.f32.s32 s{d}, s{a}; vmov r0, s{d}",
        f"vmov s{a}, r1; vsqrt.f32 s{d}, s{a}; vabs.f32 s{b}, s{d}; vneg.f32 s{a}, s{b}; vmov r0, s{a}; vmov r2, s{b}",
        f"vmov s{a}, r1; vmov s{b}, r2; vcmp.f32 s{a}, s{b}; vmrs APSR_nzcv, fpscr; mrs r0, apsr; and r0, r0, #0xf0000000",
        f"vdup.32 q{a}, r1; vdup.32 q{b}, r2; {rng.choice(['vadd.i32', 'vsub.i32', 'vmul.i32', 'vand', 'veor', 'vorr', 'vmax.s32', 'vmin.u32'])} q{d}, q{a}, q{b}; vmov r0, r3, d{2 * d}; vmov r5, r6, d{2 * d + 1}",
        f"vdup.8 d{a}, r1; vdup.16 d{b}, r2; vadd.i16 d{d}, d{a}, d{b}; vmov r0, r3, d{d}",
        f"vmov d{a}, r1, r2; vshl.i32 d{d}, d{a}, #{rng.randrange(32)}; vmov r0, r3, d{d}",
        f"vmov d{a}, r1, r2; vrev64.8 d{d}, d{a}; vmov r0, r3, d{d}",
        f"adr r4, 1f; vldr s{a}, [r4]; vmov r0, s{a}; b 2f; .balign 4; 1: .word 0x3fc00000; 2:",
        f"vmov s{a}, r1; vpush {{s{a}}}; pop {{r0}}; vpush {{s{a}}}; vpop {{s{b}}}; vmov r2, s{b}",
        f"vstmdb sp!, {{d{a}, d{b}}}; vldmia sp!, {{d{d}, d{a}}}; vmov r0, r3, d{d}",
    ])


def itblock(rng):
    cond = rng.choice(["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"])
    inv = {"eq": "ne", "ne": "eq", "cs": "cc", "cc": "cs", "mi": "pl", "pl": "mi", "vs": "vc", "vc": "vs", "hi": "ls", "ls": "hi", "ge": "lt", "lt": "ge", "gt": "le", "le": "gt"}
    pattern = rng.choice(["", "t", "e", "tt", "te", "et", "ee", "ttt", "tte", "tet", "ett", "eee"])
    insns = ["add{} r0, r0, r1", "mov{} r2, #7", "sub{} r3, r3, #1", "and{} r4, r4, r5", "lsl{} r5, r5, #2", "orr{} r6, r6, r7", "cmp{} r1, r2", "mul{} r7, r7, r6"]
    lines = [f"it{pattern} {cond}"]
    for i in range(len(pattern) + 1):
        c = cond if i == 0 or pattern[i - 1] == "t" else inv[cond]
        base = rng.choice(insns)
        text = base.format(c)
        # Flag-setting narrow forms inside a block do not set flags; wide ones need an S.
        lines.append(text)
    return "; ".join(lines)


def make_case(rng):
    pool = [data_processing] * 5 + [media] * 3 + [memory] * 4 + [branches] * 2 + [itblock] * 2 + [simd] * 2
    gen = rng.choice(pool)
    if os.environ.get('ONLY'):
        gen = globals()[os.environ['ONLY']]
    overrides = {}
    if gen is memory:
        body, overrides = gen(rng)
        init = [rng.getrandbits(32) for _ in range(11)]
    else:
        body = "; ".join(gen(rng) for _ in range(rng.randrange(1, 4))) if gen in (data_processing, media) else gen(rng)
        edge = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 31, 32, 33, 0xFFFF, 0x8000, 0x12345678]
        kind = rng.randrange(3)
        init = [rng.getrandbits(32) if kind == 0 else rng.choice(edge) if kind == 1 else rng.randrange(0, 8) for _ in range(11)]
    for reg_number, value in overrides.items():
        init[reg_number] = value
    flags = rng.getrandbits(5) << 27
    return body, init, flags & 0xF8000000


def mc(text, obj):
    subprocess.run([LLVM + "llvm-mc", "-triple=armv7-none-eabi", "-mcpu=cortex-a15", "-filetype=obj", "-o", str(obj)],
                   input=text.encode(), check=True)


def text_of(path, obj):
    subprocess.run([LLVM + "llvm-objcopy", "-O", "binary", "--only-section=.text", str(obj), str(path)], check=True)
    return Path(path).read_bytes()


def driver(count, qemu):
    qemu_tail = (
        "movw r2, #0\nmovt r2, #0x4300\nmovw r0, #0x743a\nmovt r0, #0x0074\nstr r0, [r2, #16]\n"
        "add r0, r2, #16\nstr r0, [r2]\nmov r0, #4\nstr r0, [r2, #4]\nmov r0, #3\nstr r0, [r2, #8]\n"
        "mov r1, r2\nmov r0, #1\nsvc #0x123456\nstr r0, [r2]\n"
        f"movw r0, #0\nmovt r0, #0x4200\nstr r0, [r2, #4]\nmovw r0, #{count * CASE & 0xFFFF}\nmovt r0, #{count * CASE >> 16}\nstr r0, [r2, #8]\n"
        "mov r1, r2\nmov r0, #5\nsvc #0x123456\n"
        "mov r0, #0x18\nmovw r1, #0x0026\nmovt r1, #2\nsvc #0x123456\nb .\n"
    )
    # The interpreter's image halts at once, and fills the rest with no-ops so the
    # snippets sit at the same addresses in both.
    tail = qemu_tail if qemu else "svc #0\n" + "nop\n" * (qemu_tail.count("\n") - 1)
    mmu_text = """
    movw r0, #0
    movt r0, #0x4010
    mov r1, #0
    movw r2, #0x0c0e
7:  orr r3, r2, r1, lsl #20
    str r3, [r0, r1, lsl #2]
    add r1, r1, #1
    cmp r1, #4096
    bne 7b
    mcr p15, 0, r0, c2, c0, 0
    mov r1, #0
    mcr p15, 0, r1, c2, c0, 2
    mvn r1, #0
    mcr p15, 0, r1, c3, c0, 0
    mrc p15, 0, r1, c1, c0, 0
    orr r1, r1, #1
    bic r1, r1, #2
    mcr p15, 0, r1, c1, c0, 0
    isb
    mrc p15, 0, r0, c1, c0, 2
    orr r0, r0, #0xf00000
    mcr p15, 0, r0, c1, c0, 2
    isb
    mov r0, #0x40000000
    vmsr fpexc, r0
"""
    # The interpreter's image has the same instructions at the same addresses, the
    # system register accesses being no-ops, so addresses in registers agree.
    mmu = mmu_text if qemu else "\n".join(
        "    nop" if line.split()[:1] in (["mcr"], ["mrc"], ["isb"], ["vmsr"]) else line for line in mmu_text.splitlines())
    return f"""
.arm
.text
{mmu}
    mov r11, #0
next_case:
    movw r0, #0
    movt r0, #{BUF >> 16:#x}
    movw r1, #0x7a35
    movt r1, #0x1234
    mov r2, #64
1:  str r1, [r0], #4
    movw r3, #0x79b1
    movt r3, #0x9e37
    mla r1, r1, r3, r3
    subs r2, r2, #1
    bne 1b
    movw r0, #0x80
    movt r0, #{BUF >> 16:#x}
    mov sp, r0
    adr r12, tables
    add r12, r12, r11, lsl #6
    ldr r10, [r12, #44]
    msr cpsr_f, r10
    ldm r12, {{r0-r10}}
    adr r12, tables
    movw lr, #{count * CASE & 0xFFFF}
    movt lr, #{count * CASE >> 16}
    add r12, r12, lr
    add r12, r12, r11, lsl #6
    orr r12, r12, #1
    blx r12
    movw r12, #0
    movt r12, #{RESULTS >> 16:#x}
    add r12, r12, r11, lsl #6
    stm r12, {{r0-r10}}
    mrs r0, cpsr
    str r0, [r12, #44]
    str sp, [r12, #48]
    movw r0, #0
    movt r0, #{BUF >> 16:#x}
    mov r1, #0
    mov r2, #0
    mov r3, #64
2:  ldr r4, [r0], #4
    add r1, r1, r4
    add r2, r4, r2, ror #5
    subs r3, r3, #1
    bne 2b
    str r1, [r12, #52]
    str r2, [r12, #56]
    add r11, r11, #1
    movw r0, #{count}
    cmp r11, r0
    bne next_case
{tail}
.balign 64
tables:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="target/thumb-difftest")
    ap.add_argument("--cases", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expected.bin").unlink(missing_ok=True)
    rng = random.Random(args.seed)
    cases, snippets = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        while len(cases) < args.cases:
            body, init, flags = make_case(rng)
            asm = f".syntax unified\n.thumb\n.thumb_func\n{body.replace('; ', chr(10))}\nbx lr\n"
            try:
                mc(asm, tmp / "s.o")
            except subprocess.CalledProcessError:
                continue
            code = text_of(tmp / "s.bin", tmp / "s.o")
            if len(code) > CASE - 2:
                continue
            # A snippet that returns by itself leaves the trailing `bx lr` unreached.
            cases.append((body, init, flags))
            snippets.append(code.ljust(CASE, b"\x00"))
        table = b"".join(struct.pack("<11I I", *init, flags).ljust(CASE, b"\x00") for _, init, flags in cases)
        images = {}
        for name, qemu in (("qemu", True), ("emu", False)):
            mc(driver(len(cases), qemu), tmp / "d.o")
            code = text_of(tmp / "d.bin", tmp / "d.o")
            code = code.ljust((len(code) + CASE - 1) // CASE * CASE, b"\x00")
            images[name] = code + table + b"".join(snippets)
        (out / "emu.bin").write_bytes(images["emu"])
        (out / "qemu.bin").write_bytes(images["qemu"])
    (out / "cases.txt").write_text("\n".join(f"{body}\t{' '.join(f'{v:x}' for v in init)} {flags:x}" for body, init, flags in cases) + "\n")
    result = subprocess.run(
        ["qemu-system-arm", "-M", "virt", "-cpu", "cortex-a15", "-m", "128M", "-nographic",
         "-semihosting-config", "enable=on,target=native", "-device",
         f"loader,file={out}/qemu.bin,addr={BASE:#x},cpu-num=0", "-serial", "none", "-monitor", "none"],
        capture_output=True, timeout=30)
    data = result.stdout
    if len(data) != len(cases) * CASE:
        raise SystemExit(f"qemu wrote {len(data)} bytes, expected {len(cases) * CASE}: {result.stderr.decode()[:500]}")
    (out / "expected.bin").write_bytes(data)
    print(f"{len(cases)} cases written to {out}/")


main()
