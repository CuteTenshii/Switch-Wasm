#!/usr/bin/env python3
"""Render the social preview card (web/public/assets/og.png) around `tools/screenshot.png`.

Usage: tools/make_og.py
"""

import base64
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT = os.path.join(HERE, "screenshot.png")
OUTPUT = os.path.join(HERE, os.pardir, "web", "public", "assets", "og.png")

WIDTH, HEIGHT = 1200, 630

TITLE = "switch-wasm"
HEADLINE = ["A Nintendo Switch", "emulator that runs", "in your browser"]
SUBTITLE = "ARM64 interpreter · GM20B GPU · WebAssembly"

# Above 1 scales the frame up and clips to the top-left corner.
SCREEN_ZOOM = 1.0
SCREEN_WIDTH = 512
SCREEN_HEIGHT = SCREEN_WIDTH * 9 // 16
# The window's title bar sits above the screen; the other sides are a thin bezel.
BEZEL_SIDE, BEZEL_TOP, BEZEL_BOTTOM = 16, 46, 23
WINDOW_WIDTH = SCREEN_WIDTH + 2 * BEZEL_SIDE
WINDOW_HEIGHT = SCREEN_HEIGHT + BEZEL_TOP + BEZEL_BOTTOM
WINDOW_X = WIDTH - 40 - WINDOW_WIDTH
WINDOW_Y = 315 - WINDOW_HEIGHT // 2
SCREEN_X = WINDOW_X + BEZEL_SIDE
SCREEN_Y = WINDOW_Y + BEZEL_TOP
CHIPS = ["no plugins", "runs locally", "open source"]

SANS = "Inter Display, Inter, Noto Sans, Liberation Sans, sans-serif"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chip(x: int, y: int, label: str) -> tuple[str, int]:
    """A pill with `label`, returning its markup and its advance width."""
    width = int(len(label) * 8.6) + 30
    markup = f"""
  <g>
    <rect x="{x}" y="{y}" width="{width}" height="30" rx="15"
          fill="#141a28" stroke="#242c40"/>
    <text x="{x + width / 2:.0f}" y="{y + 20}" font-family="{SANS}" font-size="14"
          fill="#8790a6" text-anchor="middle">{escape(label)}</text>
  </g>"""
    return markup, width + 10


def build_svg(screenshot_uri: str) -> str:
    chips_markup = ""
    cursor = 80
    for label in CHIPS:
        markup, advance = chip(cursor, 480, label)
        chips_markup += markup
        cursor += advance

    headline_markup = ""
    for index, line in enumerate(HEADLINE):
        colour = "#6ea8ff" if index == len(HEADLINE) - 1 else "#ffffff"
        headline_markup += (
            f'<text x="80" y="{212 + index * 62}" font-family="{SANS}" font-size="50"'
            f' font-weight="700" fill="{colour}">{escape(line)}</text>\n  '
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <defs>
    <radialGradient id="glow" cx="0.28" cy="0.0" r="0.9">
      <stop offset="0%" stop-color="#1b2740"/>
      <stop offset="55%" stop-color="#0d1119"/>
      <stop offset="100%" stop-color="#07080c"/>
    </radialGradient>
    <linearGradient id="mark" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#6ea8ff"/>
      <stop offset="100%" stop-color="#a06bff"/>
    </linearGradient>
    <linearGradient id="bezel" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1a2032"/>
      <stop offset="100%" stop-color="#0d1119"/>
    </linearGradient>
    <clipPath id="screenclip">
      <rect x="{SCREEN_X}" y="{SCREEN_Y}" width="{SCREEN_WIDTH}" height="{SCREEN_HEIGHT}" rx="4"/>
    </clipPath>
  </defs>

  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#glow)"/>

  <!-- brand -->
  <rect x="80" y="78" width="26" height="26" rx="7" fill="url(#mark)"/>
  <text x="120" y="99" font-family="{SANS}" font-size="24" font-weight="600"
        fill="#e6eaf5">{escape(TITLE)}</text>

  <!-- headline -->
  {headline_markup}

  <text x="80" y="416" font-family="{SANS}" font-size="24"
        fill="#8790a6">{escape(SUBTITLE)}</text>
  {chips_markup}

  <!-- device: the app window, with a real captured frame inside -->
  <g>
    <rect x="{WINDOW_X}" y="{WINDOW_Y}" width="{WINDOW_WIDTH}" height="{WINDOW_HEIGHT}" rx="14"
          fill="url(#bezel)" stroke="#252c3e"/>
    <circle cx="{WINDOW_X + 22}" cy="{WINDOW_Y + 21}" r="4.5" fill="#3a4358"/>
    <rect x="{SCREEN_X}" y="{SCREEN_Y}" width="{SCREEN_WIDTH}" height="{SCREEN_HEIGHT}" rx="4" fill="#000000"/>
    <image xlink:href="{screenshot_uri}" x="{SCREEN_X}" y="{SCREEN_Y}"
           width="{SCREEN_WIDTH * SCREEN_ZOOM:.0f}" height="{SCREEN_HEIGHT * SCREEN_ZOOM:.0f}"
           preserveAspectRatio="xMidYMid slice" clip-path="url(#screenclip)"/>
    <rect x="{SCREEN_X}" y="{SCREEN_Y}" width="{SCREEN_WIDTH}" height="{SCREEN_HEIGHT}" rx="4"
          fill="none" stroke="#1d2434"/>
    <text x="{WINDOW_X + 40}" y="{WINDOW_Y + 26}" font-family="{SANS}" font-size="12" fill="#5d6579">hbmenu.nro · 1280×720</text>
  </g>
</svg>
"""


def main() -> int:
    if not os.path.exists(SCREENSHOT):
        print(f"missing {SCREENSHOT}", file=sys.stderr)
        return 1
    with open(SCREENSHOT, "rb") as fh:
        uri = "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")

    svg = build_svg(uri)
    try:
        subprocess.run(
            ["rsvg-convert", "-w", str(WIDTH), "-h", str(HEIGHT), "-o", OUTPUT],
            input=svg.encode("utf-8"),
            check=True,
        )
    except FileNotFoundError:
        print("rsvg-convert not found (install librsvg)", file=sys.stderr)
        return 1
    print(f"wrote {os.path.realpath(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
