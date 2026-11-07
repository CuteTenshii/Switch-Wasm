# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `make test` and clippy, keeps every title in `docs/progress.md` booting to the same frames, and is committed on its own.

1. nvmap handles are console-wide and name a CPU address in the allocating process; a handle another process maps or presents is read from that process's memory at the same address.

## Browser bindings

- A service worker that caches the page, the worker scripts and the `.wasm`, so the emulator starts offline and installs as an app with the existing `manifest.webmanifest`. Vite hashes asset names, so the build must hand it the list; it must sit under `base: './'`; and cached responses must keep the COOP/COEP headers, or the page loses cross-origin isolation and `SharedArrayBuffer`.
- Stretch audio when emulation runs below full speed, so it plays slowed down rather than in bursts with gaps.
- Motion controls from real controllers. The Gamepad API exposes no gyro or accelerometer and Web Bluetooth cannot reach classic HID, so read Joy-Con, DualShock 4 and DualSense reports through WebHID (Chromium desktop only, permission from a user gesture, main thread only) and forward the samples to the worker into the six-axis shared memory states. A Joy-Con needs subcommand `0x40` (enable IMU) and `0x03` with `0x30` (full reports, 3 samples per ~15 ms report); its calibration at SPI `0x6020` (factory) and `0x8026` (user), read with subcommand `0x10`, is what `hid:server` 87 (LoadSixAxisSensorCalibrationParameter, zeros today) should return.

## Mobile

- Performance on phones and tablets.

## Open gaps

- The worker's host file table keeps every update and DLC added during a visit, each with up to 16 MiB of chunk cache; it should drop a file the session no longer reads.
- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Switchbrew's NPNS page does not give the empty-queue result, so find it first.
- Once `Receive` is handled, raise the system's own alerts: low battery for the console and for each controller, controller connected and disconnected.
- Internet through the relay (`docs/relay.md`): `ssl:context` `CreateConnection` is unimplemented, so a guest that does HTTPS through the `ssl` service gets nothing; it needs a TLS stack in the emulator (`rustls` in wasm) over the guest's cert store. Homebrew that does its own TLS over `bsd` is not affected.
- `bsd` datagram sockets do not reach the relay: the relay speaks UDP, but the core still fakes `sendto` and has no receive path. A blocking `connect` outside the console returns `EINPROGRESS` rather than waiting, which needs a deferred IPC reply in the kernel.
- Whether browsers let an `https://` page open `ws://127.0.0.1` (Chrome and Firefox are expected to; Safari is not known to) has not been tried on the deployed site.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- Guest output since the last drain is lost when the run loop dies on a trap or panic, or the session is recycled mid-run (`web/main/runloop.ts` catch path, `recycleSession`).
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- Just Dance 2019 stops presenting on "Hold on... and warm up..." after one to two minutes and stays there (0 fps, over 100B instructions). The main thread only sleeps and polls, while two priority-29 threads run flat out on the other cores (a float-to-16-bit conversion loop, and `nn::hid::VibrationMixer`). Whether it waits on something the emulator never delivers is not known.
- Compute dispatches, and copy-engine transfers that are not one plain run or that overlap a held surface, still send every held surface back; a per-surface flush would keep the rest on the device.
- Minecraft's Terms and Conditions page panics in the `web` applet (`010000000000100a`): `Heap::allocate` fails for `0x2c000` bytes (2048 elements of `0x58`, built at `0x0818039c` in the 21.2.0 applet, count read from a config struct at `+0x9c`) from a heap object at `0x3004d000` in the applet's `SetHeapSize` region. Neither the applet heap size nor any `GetInfo` answer explains it; find where that heap's capacity is set.
