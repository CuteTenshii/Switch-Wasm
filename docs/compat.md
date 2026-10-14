# Compatibility

How far each title gets, judged from screenshots of what it shows, on the host. The Measured column dates each row:

- 2026-10-13: `screenshot` (homebrew) and `screenshot_title` (retail), firmware 21.1.0.
- 2026-10-07: retail with `screenshot_gpu` (the WebGPU backend, natively) at presented frames 300 and 1500, and `boot_nsp` up to 25B instructions (about 25 s of guest time), untouched and with <kbd>A</kbd> tapped every second; homebrew with `screenshot` at frame 120, with no input.

Playability needs a browser run; nothing below has been judged for it. On 2026-10-07 retail titles ran at about 4-5 fps in the browser.

Status, lowest to highest:

- **Fails**: no frame; it faults, exits or stalls first.
- **Boots**: presents frames but never reaches its menus.
- **Menus**: reaches its menus or main screen.
- **In-game**: reaches gameplay.
- **Playable**: in-game in the browser at a usable speed, with working input.

## Retail

| Title | Title ID | Version | Status | How far it gets | Measured |
|---|---|---|---|---|---|
| Minecraft | 0100D71004694000 | 1.0.0 | Menus | Frame 20 (16.4B steps, 110 draws) shows the world cleanly on the WebGPU device; the software rasterizer draws the right half wrong. On 2026-10-07, with <kbd>A</kbd>, it reached the Play screen (Worlds, Friends, Servers); creating a world needs menu navigation, so gameplay is untested. | 2026-10-13 |
| Mario Kart 8 Deluxe | 0100152000022000 | 1.0.0 | Menus | Its title screen, at frame 146 by 4.4B steps. | 2026-10-13 |
| Just Dance 2017 | 0100BCE000598000 | 1.0.0 | Menus | In the browser: dancer card creation and the song carousel, with its VP8 videos playing correctly. | 2026-10-07 |
| Asphalt 9: Legends | 01007B000C834000 | 1.3.1 | Boots | Title splash, then "Connection error: could not connect to the server to launch Asphalt 9 (error 2)". It needs Gameloft's servers. | 2026-10-07 |
| Just Dance 2019 | 010075600AE96000 | 284652.419607 | Boots | A white screen on both renderers through frame 6000 (about 100 s of guest time), with or without <kbd>A</kbd>. It keeps presenting frames, but is too slow in the browser to see further. | 2026-10-07 |
| A Short Hike | 01004890117B2000 | 1.0.0 | Boots | Presents 150 frames by 4.3B steps, but frame 150 is a white square on navy, with no scene. | 2026-10-13 |

## Homebrew

| Title | Status | How far it gets | Measured |
|---|---|---|---|
| `hbmenu.nro` | Menus | Its full menu, entry label included. | 2026-10-13 |
| `NX-Shell.nro` | Menus | Its file browser, presenting steadily without a fault. | 2026-10-13 |
| `JKSV.nro` | Menus | Its full UI. | 2026-10-13 |
| `Checkpoint.nro` | Menus | Its full UI, text included. | 2026-10-13 |
| `sysinfo.nro` | Menus | Its system information screen, rendered. | 2026-10-13 |
| `NX-Fetch.nro` | Menus | Its screen, rendered. | 2026-10-13 |
| `reset_parental_controls.nro` | Menus | Its menu. | 2026-10-07 |
| `SnakeNX.nro` | Menus | Title screen, "Press <kbd>A</kbd> to start"; gameplay untested. | 2026-10-07 |
| `DDLC-LOVE-Switch.nro` | Boots | LÖVE's own "Cannot load game at `sdmc:/switch/game`": the game data is not on the SD card. | 2026-10-07 |
| `dino.nro` | Boots | Its console output, then "webConfigShow failed with code 0x5d59": the web applet it runs in is not implemented. Needs a re-run now that NAND applets run as processes. | 2026-10-07 |
| `MilkOutside.nro` | Fails | No frame in 3B instructions; it waits after asking for the error applet. Needs a re-run now that NAND applets run as processes. | 2026-10-07 |
