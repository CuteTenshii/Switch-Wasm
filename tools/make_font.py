#!/usr/bin/env python3
"""Build the shared font the emulator serves to guests.

Subsets a stock font, drops TrueType hinting, maps Nintendo's private-use
button glyphs to matching characters, and copies in the geometric shapes the
stock font lacks from a symbols font.

Usage: tools/make_font.py <input.ttf> --symbols <symbols.ttf> --output <output.ttf>
"""

import argparse
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

# Latin and its supplements, general punctuation, currency symbols and arrows.
UNICODES = "U+0000-024F,U+2000-206F,U+20A0-20BF,U+2190-21FF"

# Nintendo's private-use button glyphs (light 0xE0Ex and dark 0xE0Ax themes).
BUTTON_GLYPHS = {
    0xE0E0: "A", 0xE0E1: "B", 0xE0E2: "X", 0xE0E3: "Y",
    0xE0EF: "+", 0xE0F0: "-",
    0xE0A0: "A", 0xE0A1: "B", 0xE0A2: "X", 0xE0A3: "Y",
    0xE0B3: "+", 0xE0B4: "-",
}

# Geometric Shapes. Titles fall back to U+25A1 for a missing character and expect the
# system font to have it: Echoes of Wisdom reads through a null glyph without it.
SYMBOLS = range(0x25A0, 0x2600)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the stock font to subset")
    parser.add_argument(
        "--symbols", required=True, help="a TrueType font holding the geometric shapes"
    )
    parser.add_argument("--output", required=True, help="where the shared font is written")
    args = parser.parse_args()
    source, output = args.source, args.output

    subset.main([
        source,
        "--unicodes=" + UNICODES,
        "--no-hinting",
        "--drop-tables+=GSUB,GPOS,GDEF",
        "--output-file=" + output,
    ])

    font = TTFont(output)
    charmap = font.getBestCmap()
    for codepoint, char in BUTTON_GLYPHS.items():
        glyph = charmap.get(ord(char))
        if glyph is None:
            print(f"warning: {source} has no glyph for {char!r}", file=sys.stderr)
            continue
        for table in font["cmap"].tables:
            if table.isUnicode():
                table.cmap[codepoint] = glyph
    copy_symbols(font, TTFont(args.symbols))
    font.save(output)
    print(f"wrote {output}")


def copy_symbols(font, symbols):
    """Copy the outlines and advances of `SYMBOLS` the font lacks from `symbols`."""
    if symbols["head"].unitsPerEm != font["head"].unitsPerEm:
        sys.exit("error: the symbols font has a different unitsPerEm")
    if "glyf" not in symbols:
        sys.exit("error: the symbols font has no TrueType outlines")
    charmap = font.getBestCmap()
    source_map = symbols.getBestCmap()
    order = font.getGlyphOrder()
    added = 0
    for codepoint in SYMBOLS:
        if codepoint in charmap or codepoint not in source_map:
            continue
        glyph = symbols["glyf"][source_map[codepoint]]
        if glyph.isComposite():
            print(f"warning: U+{codepoint:04X} is a composite; skipped", file=sys.stderr)
            continue
        name = f"uni{codepoint:04X}"
        order.append(name)
        font["glyf"].glyphs[name] = glyph
        font["hmtx"].metrics[name] = symbols["hmtx"][source_map[codepoint]]
        for table in font["cmap"].tables:
            if table.isUnicode():
                table.cmap[codepoint] = name
        added += 1
    font.setGlyphOrder(order)
    font["glyf"].setGlyphOrder(order)
    print(f"copied {added} geometric shapes")


if __name__ == "__main__":
    main()
