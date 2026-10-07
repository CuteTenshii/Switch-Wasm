#!/usr/bin/env python3
"""Build the shared font the emulator serves to guests.

Subsets a stock font, drops TrueType hinting, and maps Nintendo's private-use
button glyphs to matching characters.

Usage: tools/make_font.py <input.ttf> <output.ttf>
"""

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


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    source, output = argv[1], argv[2]

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
    font.save(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main(sys.argv)
