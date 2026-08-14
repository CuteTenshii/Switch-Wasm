# Switch WASM

Run Switch games on your browser with WebAssembly

An ARM64 (A64) integer interpreter plus PFS0/NSP, NCA, NRO and ELF parsers, compiled to WASM. The frontend is plain static JS with no bundler.

## Build

```sh
make all
```

Requires `rustup target add wasm32-unknown-unknown`.

## Serve

```sh
make serve
```

Then open http://localhost:8000
