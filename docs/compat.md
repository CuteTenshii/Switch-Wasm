# Compatibility

Status, lowest to highest:

- **Fails**: no frame; it faults, exits or stalls first.
- **Boots**: presents frames but never reaches its menus.
- **Menus**: reaches its menus or main screen.
- **In-game**: reaches gameplay.
- **Playable**: in-game in the browser at a usable speed, with working input.

## Retail

| Title | Title ID | Version | Status | How far it gets | Measured |
|---|---|---|---|---|---|
| Minecraft | 0100D71004694000 | 1.0.0 | Playable | In-game at 15-20 fps. | 2026-11-14 |
| Just Dance 2017 | 0100BCE000598000 | 1.0.0 | In-game | The menus navigate and songs play, at 10-20 fps. | 2026-11-14 |
| Just Dance 2019 | 010075600AE96000 | 284652.419607 | In-game | The menus navigate and songs play, at 10-20 fps. | 2026-11-14 |
| Just Dance 2022 | 0100EA6014BB8000 | 326704.600931 | In-game | The menus navigate and songs play, at 10-20 fps. | 2026-11-14 |
| The Legend of Zelda: Echoes of Wisdom | 01008CF01BAAC000 | 1.0.0 | In-game | Falls back to the software rasterizer. The prologue's first scene by frame 125, but the right half of the screen is a 2x-zoomed copy of the left. | 2026-11-14 |
| Mario Kart 8 Deluxe | 0100152000022000 | 1.0.0 | Menus | Its menus render, at a low frame rate. Races untested. | 2026-11-14 |
| Home Menu (`qlaunch`) | 0100000000001000 | 21.2.0 | Menus | Navigable: settings can be browsed and changed, and the album and profile pages open as applets. | 2026-11-14 |
| A Short Hike | 01004890117B2000 | 1.0.0 | Menus | Its title screen at frame 1500 (11.9B steps), with the button glyph in "press … to start" missing. | 2026-11-14 |
| Tomodachi Life: Living the Dream | 010051F0207B2000 | 1.0.0 | Boots | Its loading screen through frame 1500 (19.3B steps). | 2026-11-14 |
| Persona 5 Royal | 01005CA01580E000 | 1.0.0 | Menus | Its main menu at frame 3810 (pressing A past the system-data screen and the title), all on the device but about 6 s a frame from the title on. | 2026-11-14 |
| Just Dance 2023 | 0100BEE017FC0000 | 1.0.0 | | Falls back to the software rasterizer; how far it gets is not recorded. | 2026-11-14 |

## Homebrew

| Title | Status | How far it gets | Measured |
|---|---|---|---|
| `hbmenu.nro` | Menus | Its full menu, entry label included. | 2026-11-14 |
| `NX-Shell.nro` | Menus | Its file browser, presenting steadily. It asks `sm` for command 65100 (an Atmosphère extension), which is unimplemented. | 2026-11-14 |
| `JKSV.nro` | Menus | Its full UI. | 2026-11-14 |
| `Checkpoint.nro` | Menus | Its full UI, text included. | 2026-11-14 |
| `sysinfo.nro` | Menus | Its system information screen. | 2026-11-14 |
| `NX-Fetch.nro` | Menus | Its screen. | 2026-11-14 |
| `reset_parental_controls.nro` | Menus | Its menu. | 2026-10-07, not on hand since |
| `SnakeNX.nro` | Menus | Title screen, "Press <kbd>A</kbd> to start"; gameplay untested. | 2026-10-07, not on hand since |
| `DDLC-LOVE-Switch.nro` | Boots | LÖVE's own "Cannot load game at `sdmc:/switch/game`": the game data is not on the SD card. | 2026-10-07, not on hand since |
| `dino.nro` | Boots | Its console output, then "webConfigShow failed with code 0x5d59": it ran before NAND applets ran as processes. | 2026-10-07, not on hand since |
| `MilkOutside.nro` | Fails | No frame in 3B instructions; it waited after asking for the error applet, before NAND applets ran as processes. | 2026-10-07, not on hand since |
