# TODO

## Multi-process kernel

Library applets must run as processes beside their caller, in one kernel with one set of console services.
Each stage builds, passes `make test` and clippy, keeps every title in `docs/progress.md` booting to the same frames, and is committed on its own.

1. nvmap handles are console-wide and name a CPU address in the allocating process; a handle another process maps or presents is read from that process's memory at the same address.

## Browser bindings

- A service worker that caches the page, the worker scripts and the `.wasm`, so the emulator starts offline and installs as an app with the existing `manifest.webmanifest`. Vite hashes asset names, so the build must hand it the list; it must sit under `base: './'`; and cached responses must keep the COOP/COEP headers, or the page loses cross-origin isolation and `SharedArrayBuffer`.
- Stretch audio when emulation runs below full speed, so it plays slowed down rather than in bursts with gaps.
- Motion controls from real controllers. The Gamepad API exposes no gyro or accelerometer and Web Bluetooth cannot reach classic HID, so read Joy-Con, DualShock 4 and DualSense reports through WebHID (Chromium desktop only, permission from a user gesture, main thread only) and forward the samples to the worker into the six-axis shared memory states. A Joy-Con needs subcommand `0x40` (enable IMU) and `0x03` with `0x30` (full reports, 3 samples per ~15 ms report); its calibration at SPI `0x6020` (factory) and `0x8026` (user), read with subcommand `0x10`, is what `hid:server` 87 (LoadSixAxisSensorCalibrationParameter, zeros today) should return.

## Performance

- Blocks with an instruction the emitter lacks stay in the op walk and break their region. `LDXP`/`STXP` are interpreted (Asphalt 9 runs them); the 64-bit pair store compares 16 bytes, which wasm has no atomic for. Double-precision FMADD and FRINTA/FRINTX/FRINTI are interpreted too.
- The device converts `S8D24` textures twice over: on the CPU at upload for an image in guest memory (`Copy::PackedDepth`, every texel through `DepthTexel::decode`), and on the GPU for one held on the device (`PACKED_DEPTH_WGSL`). Upload the raw texels as an integer texture and run the same pass for both, so the device has one conversion and the upload no CPU work; `the_packed_depth_pass_converts_depths_bit_for_bit` keeps it to the rasterizer's values.

## Mobile

- Performance on phones and tablets.

## Open gaps

- The worker's host file table keeps every update and DLC added during a visit, each with up to 16 MiB of chunk cache; it should drop a file the session no longer reads.
- `npns:s` cmd 3 (Receive) is refused; qlaunch polls it constantly. Needs the empty-queue result from switchbrew.
- `ssl:s` cmd 5 is stubbed and the `ssl:s` name is not routed like `ssl`.
- Guest output since the last drain is lost when the run loop dies on a trap or panic, or the session is recycled mid-run (`web/main/runloop.ts` catch path, `recycleSession`).
- The log file is hard to read: per-call `[ipc]`/`[wait]` trace lines, `[thread]` reports repeated every few seconds even when nothing changed, and periodic `[audio]`/`[io]`/`[fs]` stats.
- Just Dance 2019 stops presenting on "Hold on... and warm up..." after one to two minutes and stays there (0 fps, over 100B instructions). The main thread only sleeps and polls, while two priority-29 threads run flat out on the other cores (a float-to-16-bit conversion loop, and `nn::hid::VibrationMixer`). Whether it waits on something the emulator never delivers is not known.
- `emit_selftest` has 48 cases of three programs differing from the interpreter, on the sign of a zero or NaN in `FMAXNM` and `FNMUL`.
- Copy-engine and compute flushes still send every held surface back; a per-surface flush would keep the rest on the device.
- Minecraft frame 20 is clean on the WebGPU device but the software rasterizer differs from it, mostly in the right half.
