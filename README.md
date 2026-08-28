# Switch WASM

A Nintendo Switch emulator running directly in the browser, using WebAssembly and WebGPU.

## Build

```sh
bun install
make all
```

Requires `rustup target add wasm32-unknown-unknown`, and [bun](https://bun.com)
for the frontend.

## Serve

```sh
bun run dev       # Vite dev server, http://localhost:8000
bun run preview   # the built site from dist/
```

Both need the core built once (`make wasm`), since the frontend imports it.

## Acknowledgments

- [Eden Emulator](https://git.eden-emu.dev/eden-emu/eden)
- [libopus](https://opus-codec.org)
- [EnvyTools](https://github.com/envytools/envytools)
- [libnx](https://github.com/switchbrew/libnx)
- [SwitchBrew](https://switchbrew.org)
- [deko3d](https://github.com/devkitPro/deko3d)
- [Mesa / nouveau](https://gitlab.freedesktop.org/mesa/mesa)
- [hactool](https://github.com/SciresM/hactool)
- [libtransistor](https://github.com/reswitched/libtransistor)
