#!/usr/bin/env python3
"""Translate libvpx's 1-D inverse transforms from C to Rust.

usage: extract_vp9_transforms.py <libvpx checkout> <out.rs>

The transforms are long, regular and must match the reference bit for bit, so they are
translated mechanically instead of retyped. Only the C that vpx_dsp/inv_txfm.c uses in these
functions is understood; anything else stops the script.
"""
import re
import sys

FUNCTIONS = ["idct4_c", "idct8_c", "idct16_c", "idct32_c", "iadst4_c", "iadst8_c", "iadst16_c"]
TYPES = r"(?:int16_t|tran_high_t|tran_low_t|int)"


def function_body(text, name):
    start = re.search(r"\bvoid " + name + r"\(const tran_low_t \*input, tran_low_t \*output\) \{", text)
    if not start:
        sys.exit(f"{name}: not found")
    depth, at = 1, start.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[at], 0)
        at += 1
    return text[start.end() : at - 1]


def expression(rhs):
    rhs = re.sub(r"\(int16_t\)\s*input\[([^\]]+)\]", r"i64::from(input[\1] as i16)", rhs)
    rhs = re.sub(r"(?<!from\()(?<![A-Za-z_])input\[([^\]]+)\]", r"i64::from(input[\1])", rhs)
    for cast in ("(int16_t)", "(int)", "(tran_low_t)", "(tran_high_t)"):
        rhs = rhs.replace(cast, "")
    rhs = rhs.replace("WRAPLOW(", "wraplow(").replace("dct_const_round_shift(", "round_shift(")
    rhs = re.sub(r"\bcospi_(\d+)_64\b", r"COSPI_\1_64", rhs)
    rhs = re.sub(r"\bsinpi_(\d)_9\b", r"SINPI_\1_9", rhs)
    return rhs


def declaration(statement):
    match = re.fullmatch(TYPES + r"\s+(.*)", statement)
    if not match:
        return None
    out = []
    for part in re.split(r",\s*(?![^\[]*\])", match.group(1)):
        part = part.strip()
        array = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        init = re.fullmatch(r"(\w+)\s*=\s*(.+)", part)
        if array:
            out.append(f"let mut {array.group(1)} = [0i64; {array.group(2)}];")
        elif init:
            out.append(f"let mut {init.group(1)} = {expression(init.group(2))};")
        else:
            out.append(f"let mut {part} = 0i64;")
    return out


def translate(name, body):
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    body = " ".join(body.split())
    body = re.sub(
        r"if \(!\((.*?)\)\) \{ memset\(output, 0, (\d+) \* sizeof\(\*output\)\); return; \}",
        r"EARLY(\1, \2);",
        body,
    )
    lines = []
    for statement in (s.strip() for s in body.split(";")):
        if not statement:
            continue
        early = re.fullmatch(r"EARLY\((.*), (\d+)\)", statement)
        if early:
            lines.append(f"if ({early.group(1)}) == 0 {{")
            lines.append(f"    output[..{early.group(2)}].fill(0);")
            lines.append("    return;")
            lines.append("}")
            continue
        decl = declaration(statement)
        if decl is not None:
            lines.extend(decl)
            continue
        assign = re.fullmatch(r"([\w\[\]/ +*-]+?)\s*=\s*(.+)", statement)
        if not assign:
            sys.exit(f"{name}: cannot translate `{statement}`")
        lhs, rhs = assign.group(1), expression(assign.group(2))
        if lhs.startswith("output["):
            lines.append(f"{lhs} = ({rhs}) as i32;")
        else:
            lines.append(f"{lhs} = {rhs};")
    rust_name = name.removesuffix("_c")
    head = f"pub fn {rust_name}(input: &[i32], output: &mut [i32]) {{"
    return "\n".join([head] + ["    " + line for line in lines] + ["}", ""])


def constants(text):
    out = []
    for match in re.finditer(r"static const tran_coef_t (cospi_\d+_64|sinpi_\d_9) = (\d+);", text):
        out.append(f"pub const {match.group(1).upper()}: i64 = {match.group(2)};")
    return out


def main():
    root, out = sys.argv[1], sys.argv[2]
    source = open(f"{root}/vpx_dsp/inv_txfm.c").read()
    common = open(f"{root}/vpx_dsp/txfm_common.h").read()
    lines = [
        "//! The 1-D inverse transforms, translated from libvpx by tools/extract_vp9_transforms.py.",
        "//! Do not edit by hand.",
        "",
        "#![cfg_attr(rustfmt, rustfmt::skip)]",
        "#![allow(clippy::all, unused_assignments, unused_mut, unused_parens)]",
        "",
    ]
    lines += constants(common)
    lines += [
        "",
        "fn wraplow(x: i64) -> i64 {",
        "    x",
        "}",
        "",
        "fn round_shift(x: i64) -> i64 {",
        "    (x + (1 << 13)) >> 14",
        "}",
        "",
    ]
    for name in FUNCTIONS:
        lines.append(translate(name, function_body(source, name)))
    open(out, "w").write("\n".join(lines))


main()
