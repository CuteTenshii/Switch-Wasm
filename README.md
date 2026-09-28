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

`dev` needs the core built once with `make wasm`, since the frontend imports
it; `preview` serves what `make assets` built.

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
