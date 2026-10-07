# Compatibility

How far each title gets, judged from screenshots of what it shows. Captured on 2026-10-07 between `104b595` and `9575393`, on the host:

- Retail: `screenshot_gpu` (the WebGPU backend, natively) at presented frames 300 and 1500, and `boot_nsp` up to 25B instructions (about 25 s of guest time), untouched and with <kbd>A</kbd> tapped every second.
- Homebrew: `screenshot` at frame 120, with no input.

Playability needs a browser run; nothing below has been judged for it. Retail titles currently run at about 4-5 fps in the browser.

Status, lowest to highest:

- **Fails**: no frame; it faults, exits or stalls first.
- **Boots**: presents frames but never reaches its menus.
- **Menus**: reaches its menus or main screen.
- **In-game**: reaches gameplay.
- **Playable**: in-game in the browser at a usable speed, with working input.

## Retail

| Title | Title ID | Version | Status | How far it gets |
|---|---|---|---|---|
| Minecraft | 0100D71004694000 | 1.0.0 | Menus | Mojang logo, the Autosave notice over the title world, and with <kbd>A</kbd> the Play screen (Worlds, Friends, Servers). Creating a world needs menu navigation, so gameplay is untested. The software renderer draws blocky tile glitches on the right half of the title world. |
| Asphalt 9: Legends | 01007B000C834000 | 1.3.1 | Boots | Title splash, then "Connection error: could not connect to the server to launch Asphalt 9 (error 2)". It needs Gameloft's servers. |
| Just Dance 2017 | 0100BCE000598000 | 1.0.0 | Boots | The Joy-Con strap warning, then its background, where it stays; <kbd>A</kbd> changes nothing. Slow: 1143 frames in 25B instructions. |
| Just Dance 2019 | 010075600AE96000 | 284652.419607 | Boots | A white screen on both renderers through frame 6000 (about 100 s of guest time), with or without <kbd>A</kbd>. It keeps presenting frames, but is too slow in the browser to see further. |
| A Short Hike | 01004890117B2000 | 1.0.0 | Boots | A white square on a dark screen, then black through 7840 frames, with or without <kbd>A</kbd>. |

## Homebrew

| Title | Status | How far it gets |
|---|---|---|
| `hbmenu.nro` | Menus | Its full menu. |
| `JKSV.nro` | Menus | Its menu; the user tile is empty. |
| `Checkpoint.nro` | Menus | Its menu, showing "No saves". |
| `sysinfo.nro` | Menus | Its full system information screen. |
| `NX-Fetch.nro` | Menus | Its full screen. |
| `reset_parental_controls.nro` | Menus | Its menu. |
| `SnakeNX.nro` | Menus | Title screen, "Press <kbd>A</kbd> to start"; gameplay untested. |
| `DDLC-LOVE-Switch.nro` | Boots | LÖVE's own "Cannot load game at `sdmc:/switch/game`": the game data is not on the SD card. |
| `dino.nro` | Boots | Its console output, then "webConfigShow failed with code 0x5d59": the web applet it runs in is not implemented. |
| `NX-Shell.nro` | Fails | Faults at 224M instructions in Dear ImGui's font index, before a frame. |
| `MilkOutside.nro` | Fails | No frame in 3B instructions; it waits after asking for the error applet. |
