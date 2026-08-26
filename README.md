# Switch WASM

Run Switch games on your browser with WebAssembly

An ARM64 (A64) interpreter with a block-translating JIT, plus PFS0/NSP, NCA, NRO and ELF parsers, compiled to WASM. The frontend is TypeScript, bundled with Vite.

Guest code is translated into pre-decoded blocks the first time it runs, so
decoding is paid for once per basic block rather than once per instruction —
worth 1.9-2.1x on real homebrew. It removes decode, not dispatch: no code is
generated. Emitting wasm per block and compiling it at runtime is a real JIT
and is the next step for speed, but a generated module can only address its
own linear memory, and guest memory is a page table rather than a flat buffer
— see `cpu/jit.rs` for why. Anything the translator has no op for falls back
to the interpreter, and the two are the same computation:

```sh
cargo run --release -p switch-core --example jit_bench -- test-nros/hbmenu.nro
```

runs the same program both ways and reports the throughput of each alongside
every difference between the two machines. `SWITCH_NO_JIT=1` in the
environment turns translation off for the host tools; in the browser the debug
panel's *Translation* section does the same.

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
- [EnvyTools](https://github.com/envytools/envytools)
- [libnx](https://github.com/switchbrew/libnx)
- [SwitchBrew](https://switchbrew.org)
- [deko3d](https://github.com/devkitPro/deko3d)
- [Mesa / nouveau](https://gitlab.freedesktop.org/mesa/mesa)
- [hactool](https://github.com/SciresM/hactool)
- [libtransistor](https://github.com/reswitched/libtransistor)
