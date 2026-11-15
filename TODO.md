# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `cargo test --workspace` and clippy, keeps every title in `docs/compat.md` booting to the same frames, and is committed on its own.

1. nvmap handles are console-wide and name a CPU address in the allocating process; a handle another process maps or presents is read from that process's memory at the same address.

## Browser bindings

- A service worker that caches the page, the worker scripts and the `.wasm`, so the emulator starts offline and installs as an app with the existing `manifest.webmanifest`. Vite hashes asset names, so the build must hand it the list; it must sit under `base: './'`; and cached responses must keep the COOP/COEP headers, or the page loses cross-origin isolation and `SharedArrayBuffer`.
- Stretch audio when emulation runs below full speed, so it plays slowed down rather than in bursts with gaps.
- Motion controls from real controllers. The Gamepad API exposes no gyro or accelerometer and Web Bluetooth cannot reach classic HID, so read Joy-Con, DualShock 4 and DualSense reports through WebHID (Chromium desktop only, permission from a user gesture, main thread only) and forward the samples to the worker into the six-axis shared memory states. A Joy-Con needs subcommand `0x40` (enable IMU) and `0x03` with `0x30` (full reports, 3 samples per ~15 ms report); its calibration at SPI `0x6020` (factory) and `0x8026` (user), read with subcommand `0x10`, is what `hid:server` 87 (LoadSixAxisSensorCalibrationParameter, zeros today) should return.

## Mobile

- Performance on phones and tablets.

## Open gaps

- The host file table keeps every update and DLC added during a visit; it should drop a file the session no longer reads.
- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Switchbrew's NPNS page does not give the empty-queue result, so find it first.
- Once `Receive` is handled, raise the system's own alerts: low battery for the console and for each controller, controller connected and disconnected.
- Whether browsers let an `https://` page open `ws://127.0.0.1` (Chrome and Firefox are expected to; Safari is not known to) has not been tried on the deployed site.
- The relay socket, its frames and TLS run on the emulator's thread, once per slice: a handshake or a big download costs guest time, and every answer from the relay waits for the next slice. A network worker could own the WebSocket and `rustls` and share the command and event queue with `net::exchange` through a mutex.
- `ssl` and `ssl:s` refuse the commands switchbrew does not describe: `GetCertificateByIndex` (10), `GetTrustedCertificateCount` (11) and `VerifySignature` (103) are names only, and `DebugIoctl` (4) is an error on retail with no code given. A guest that sends one gets a refusal in the log.
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- Compute dispatches, and copy-engine transfers that are not one plain run or that overlap a held surface, still send every held surface back; a per-surface flush would keep the rest on the device.
- Minecraft's Terms and Conditions page panics in the `web` applet (`010000000000100a`): `Heap::allocate` fails for `0x2c000` bytes (2048 elements of `0x58`, built at `0x0818039c` in the 21.2.0 applet, count read from a config struct at `+0x9c`) from a heap object at `0x3004d000` in the applet's `SetHeapSize` region. Neither the applet heap size nor any `GetInfo` answer explains it; find where that heap's capacity is set.
- Tomodachi Life: Living the Dream stays on its loading screen through frame 1500 (19.3B steps; `docs/compat.md`).
- `ncm` has no implementation. Homebrew also still opens a service under an empty name (Checkpoint does), and `sm` hands out a working handle instead of failing as real `sm` would.
- Hinted fonts render 1-3px wide. The guest's own FreeType, running on the emulated CPU, gives glyphs from a font with hinting programs (`fpgm`/`prep`/`cvt`) correct heights and advances but collapsed bitmaps, as if untouched points never get interpolated; the same subset built with `tools/make_font.py --no-hinting` renders perfectly, so something the emulator runs under the hinting interpreter is wrong. The stand-in `web/font.ttf` has no hinting; whether the firmware fonts `pl:u` serves do is unchecked, and the bug has not been retested since many decode fixes landed.

## Video

- JD2022's intro video decodes and plays; see `docs/video.md` for what is still missing: counter output for streams that adapt probabilities, nvdec surfaces in guest memory, VIC pitch alignment, and VP9 decoder speed in the browser.
