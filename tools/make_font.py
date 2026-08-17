#!/usr/bin/env python3
"""Build the font the emulator serves to guests as the console's shared font.

The guest reads this file out of `pl:u`'s shared memory and renders it with its
own copy of FreeType, so it has to be a real TrueType/OpenType file. Two things
are done to a stock font to make it fit that role:

* The character set is cut down to what homebrew UIs actually draw, keeping the
  file small enough to ship and fetch (a full CJK font is tens of megabytes).
* The TrueType hinting programs are dropped. Hinted glyphs currently come out
  collapsed horizontally under the emulator's interpreter, and running the
  hinting bytecode costs about eight times more emulated instructions per
  frame — see PROGRESS.md.

Nintendo's UI fonts also carry controller-button glyphs in the private use
area, which homebrew uses for on-screen button hints ("Ⓐ Launch"). A stock font
has nothing there, so those codepoints are pointed at the matching letters and
signs; without this the hints render as arbitrary wrong glyphs.

Usage: tools/make_font.py <input.ttf> <output.ttf>
"""

import sys

from fontTools import subset
from fontTools.ttLib import TTFont

# Latin and its supplements, general punctuation, currency symbols and arrows.
UNICODES = "U+0000-024F,U+2000-206F,U+20A0-20BF,U+2190-21FF"

# Nintendo's private-use button glyphs, as used by nx-hbmenu's themes, mapped to
# the closest thing a text font has. Both the light theme's set (0xE0Ex) and the
# dark theme's (0xE0Ax) are covered.
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
