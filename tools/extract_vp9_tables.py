#!/usr/bin/env python3
"""Write the VP9 constant tables out of the libvpx sources as Rust.

usage: extract_vp9_tables.py <libvpx checkout> <out.rs>

The tables are the VP9 specification's normative data. They are read from the C
initializers by name, so none is retyped. Pure Python, any OS.
"""
import os
import re
import sys

# (C symbol, Rust name, element type, dimensions)
TABLES = [
    ("default_scan_4x4", "DEFAULT_SCAN_4X4", "i16", [16]),
    ("col_scan_4x4", "COL_SCAN_4X4", "i16", [16]),
    ("row_scan_4x4", "ROW_SCAN_4X4", "i16", [16]),
    ("default_scan_8x8", "DEFAULT_SCAN_8X8", "i16", [64]),
    ("col_scan_8x8", "COL_SCAN_8X8", "i16", [64]),
    ("row_scan_8x8", "ROW_SCAN_8X8", "i16", [64]),
    ("default_scan_16x16", "DEFAULT_SCAN_16X16", "i16", [256]),
    ("col_scan_16x16", "COL_SCAN_16X16", "i16", [256]),
    ("row_scan_16x16", "ROW_SCAN_16X16", "i16", [256]),
    ("default_scan_32x32", "DEFAULT_SCAN_32X32", "i16", [1024]),
    ("default_scan_4x4_neighbors", "DEFAULT_NEIGHBORS_4X4", "i16", [34]),
    ("col_scan_4x4_neighbors", "COL_NEIGHBORS_4X4", "i16", [34]),
    ("row_scan_4x4_neighbors", "ROW_NEIGHBORS_4X4", "i16", [34]),
    ("default_scan_8x8_neighbors", "DEFAULT_NEIGHBORS_8X8", "i16", [130]),
    ("col_scan_8x8_neighbors", "COL_NEIGHBORS_8X8", "i16", [130]),
    ("row_scan_8x8_neighbors", "ROW_NEIGHBORS_8X8", "i16", [130]),
    ("default_scan_16x16_neighbors", "DEFAULT_NEIGHBORS_16X16", "i16", [514]),
    ("col_scan_16x16_neighbors", "COL_NEIGHBORS_16X16", "i16", [514]),
    ("row_scan_16x16_neighbors", "ROW_NEIGHBORS_16X16", "i16", [514]),
    ("default_scan_32x32_neighbors", "DEFAULT_NEIGHBORS_32X32", "i16", [2050]),
    ("default_coef_probs_4x4", "DEFAULT_COEF_PROBS_4X4", "u8", [2, 2, 6, 6, 3]),
    ("default_coef_probs_8x8", "DEFAULT_COEF_PROBS_8X8", "u8", [2, 2, 6, 6, 3]),
    ("default_coef_probs_16x16", "DEFAULT_COEF_PROBS_16X16", "u8", [2, 2, 6, 6, 3]),
    ("default_coef_probs_32x32", "DEFAULT_COEF_PROBS_32X32", "u8", [2, 2, 6, 6, 3]),
    ("vp9_kf_y_mode_prob", "KF_Y_MODE_PROBS", "u8", [10, 10, 9]),
    ("vp9_kf_uv_mode_prob", "KF_UV_MODE_PROBS", "u8", [10, 9]),
    ("vp9_kf_partition_probs", "KF_PARTITION_PROBS", "u8", [16, 3]),
    ("default_partition_probs", "DEFAULT_PARTITION_PROBS", "u8", [16, 3]),
    ("default_if_y_probs", "DEFAULT_IF_Y_PROBS", "u8", [4, 9]),
    ("default_if_uv_probs", "DEFAULT_IF_UV_PROBS", "u8", [10, 9]),
    ("default_inter_mode_probs", "DEFAULT_INTER_MODE_PROBS", "u8", [7, 3]),
    ("default_skip_probs", "DEFAULT_SKIP_PROBS", "u8", [3]),
    ("default_comp_inter_p", "DEFAULT_COMP_INTER_PROBS", "u8", [5]),
    ("default_comp_ref_p", "DEFAULT_COMP_REF_PROBS", "u8", [5]),
    ("default_single_ref_p", "DEFAULT_SINGLE_REF_PROBS", "u8", [5, 2]),
    ("default_tx_probs", "DEFAULT_TX_PROBS_RAW", "u8", [12]),
    ("default_nmv_context", "DEFAULT_NMV_CONTEXT_RAW", "u8", [69]),
    ("dc_qlookup", "DC_QLOOKUP", "i16", [256]),
    ("ac_qlookup", "AC_QLOOKUP", "i16", [256]),
    ("sub_pel_filters_8", "FILTER_REGULAR", "i16", [16, 8]),
    ("sub_pel_filters_8lp", "FILTER_SMOOTH", "i16", [16, 8]),
    ("sub_pel_filters_8s", "FILTER_SHARP", "i16", [16, 8]),
    ("bilinear_filters", "FILTER_BILINEAR", "i16", [16, 8]),
    ("vp9_cat1_prob", "CAT1_PROB", "u8", [1]),
    ("vp9_cat2_prob", "CAT2_PROB", "u8", [2]),
    ("vp9_cat3_prob", "CAT3_PROB", "u8", [3]),
    ("vp9_cat4_prob", "CAT4_PROB", "u8", [4]),
    ("vp9_cat5_prob", "CAT5_PROB", "u8", [5]),
    ("vp9_cat6_prob", "CAT6_PROB", "u8", [14]),
    ("vp9_pareto8_full", "PARETO8_FULL", "u8", [255, 8]),
    ("vp9_coefband_trans_8x8plus", "COEFBAND_8X8PLUS", "u8", [1024]),
    ("vp9_coefband_trans_4x4", "COEFBAND_4X4", "u8", [16]),
    ("mode_2_counter", "MODE_2_COUNTER", "i32", [14]),
    ("counter_to_context", "COUNTER_TO_CONTEXT", "i32", [19]),
    ("mv_ref_blocks", "MV_REF_BLOCKS", "i32", [13, 8, 2]),
    ("inv_map_table", "INV_MAP_TABLE", "u8", [255]),
]

# vp9_mvref_common.h's motion_vector_context, which counter_to_context is written in.
ENUMS = {
    "BOTH_ZERO": 0,
    "ZERO_PLUS_PREDICTED": 1,
    "BOTH_PREDICTED": 2,
    "NEW_PLUS_NON_INTRA": 3,
    "BOTH_NEW": 4,
    "INTRA_PLUS_NON_INTRA": 5,
    "BOTH_INTRA": 6,
    "INVALID_CASE": 9,
}

ARCH_DIRS = ("x86", "arm", "mips", "ppc", "loongarch", "riscv", "test")


def sources(root):
    for sub in ("vp9", "vpx_dsp"):
        for base, dirs, files in os.walk(os.path.join(root, sub)):
            dirs[:] = [d for d in dirs if d not in ARCH_DIRS]
            for name in sorted(files):
                if name.endswith((".c", ".h")):
                    yield os.path.join(base, name)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def initializer(text, name):
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*(?:\[[^\]]*\]\s*)*\)?\s*=\s*\{")
    match = pattern.search(text)
    if not match:
        return None
    depth, at = 1, match.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[at], 0)
        at += 1
    return text[match.end() : at - 1]


def parse(body, name):
    """The initializer's braces as nested lists of integers."""
    body = re.sub(r"[A-Z_]+", lambda m: str(ENUMS.get(m.group(0), m.group(0))), body)
    if re.search(r"[A-Za-z_]", re.sub(r"0[xX][0-9a-fA-F]+", "", body)):
        sys.exit(f"{name}: the initializer is not plain numbers")
    stack = [[]]
    for token in re.findall(r"[{}]|-?0[xX][0-9a-fA-F]+|-?\d+", body):
        if token == "{":
            stack.append([])
        elif token == "}":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(int(token, 0))
    return stack[0]


def flatten(node):
    return [v for item in node for v in (flatten(item) if isinstance(item, list) else [item])]


def fill(node, shape):
    """The nested lists as `shape`, padding what C leaves out with zeros."""
    if len(shape) == 1:
        flat = flatten(node)
        return flat + [0] * (shape[0] - len(flat))
    rows = [fill(node[i], shape[1:]) if i < len(node) else fill([], shape[1:]) for i in range(shape[0])]
    return [v for row in rows for v in row]


def nest(values, shape):
    if len(shape) == 1:
        return "[" + ", ".join(str(v) for v in values) + "]"
    step = len(values) // shape[0]
    parts = (nest(values[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0]))
    return "[" + ", ".join(parts) + "]"


def main():
    root, out = sys.argv[1], sys.argv[2]
    texts = [(path, strip_comments(open(path, errors="replace").read())) for path in sources(root)]
    lines = [
        "//! VP9 constant tables, generated by tools/extract_vp9_tables.py from libvpx.",
        "//! Do not edit by hand.",
        "",
        "#![cfg_attr(rustfmt, rustfmt::skip)]",
        "",
    ]
    for cname, rname, ty, dims in TABLES:
        body = next((b for _, t in texts if (b := initializer(t, cname)) is not None), None)
        if body is None:
            sys.exit(f"{cname}: not found")
        tree = parse(body, cname)
        count = 1
        for d in dims:
            count *= d
        values = flatten(tree)
        if len(values) != count and len(dims) > 1:
            values = fill(tree, dims)
        if len(values) != count:
            sys.exit(f"{cname}: {len(values)} values, {count} expected")
        typ = ty
        for d in reversed(dims):
            typ = f"[{typ}; {d}]"
        lines.append(f"pub const {rname}: {typ} = {nest(values, dims)};")
        lines.append("")
    open(out, "w").write("\n".join(lines))


main()
