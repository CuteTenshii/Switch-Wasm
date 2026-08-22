# Switch WASM

Run Switch games on your browser with WebAssembly

An ARM64 (A64) integer interpreter plus PFS0/NSP, NCA, NRO and ELF parsers, compiled to WASM. The frontend is TypeScript, bundled with Vite.

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

## Acknowlegments

- [Eden Emulator](https://git.eden-emu.dev/eden-emu/eden)
- [EnvyTools](https://github.com/envytools/envytools)