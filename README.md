<p align="center"><img src="web/favicon.svg" alt="" width="96" height="96"></p>
<h1 align="center">Switch WASM</h1>
<p align="center">A Nintendo Switch emulator running directly in the browser, using WebAssembly and WebGPU.</p>
<p align="center"><a href="https://git.tenshii.moe/Tenshii/Switch-Wasm/actions?workflow=check.yml"><img src="https://git.tenshii.moe/Tenshii/Switch-Wasm/badges/workflows/check.yml/badge.svg?branch=master&label=ci" alt="CI status"></a>
<a href="https://discord.gg/uhy2P4vyhE"><img src="https://img.shields.io/badge/discord-join-5865F2?logo=discord&logoColor=white" alt="Join the Discord server"></a></p>

## Build

```sh
bun install
make
```

Requires [Bun](https://bun.com) for the frontend, and a nightly Rust toolchain, which the browser build needs to rebuild std with atomics:

```sh
rustup toolchain install nightly --profile minimal --component rust-src --target wasm32-unknown-unknown
```

It also needs a `wasm-bindgen` CLI matching the version in `Cargo.lock`.

## Serve

```sh
bun run dev       # Vite dev server, http://localhost:8000
bun run preview   # the built site from dist/
```

`dev` needs the core built once with `make wasm`, since the frontend imports it. `preview` serves what `make assets` built.

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
