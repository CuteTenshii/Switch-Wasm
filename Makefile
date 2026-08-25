.PHONY: all test wasm assets clean

TARGET := wasm32-unknown-unknown
WASM   := target/$(TARGET)/release/switch_wasm.wasm
DIST   := dist

all: test assets

# Host test suite: parsers, memory, CPU interpreter, loaders, demo boot, and
# the host-facing wasm entry points (which build for the host too, so the SD
# card's import/export API is covered without a browser).
test:
	cargo test -p switch-core
	cargo test -p switch-wasm
	cargo test -p switch-gpu

# Compile the wasm bindings crate, with the WebGPU backend.
#
# Always with it. `wgpu` reaches WebGPU through `wasm-bindgen`, so the
# artefact is a wasm-bindgen module with generated glue beside it rather than
# a bare one the worker hands to `WebAssembly.instantiateStreaming` — and
# carrying two shapes of core, two loaders and two answers to every question
# about the build costs more than the megabyte it would save. A machine
# without WebGPU still runs: the backend reports that it could not open a
# device and the software rasterizer takes the frame, which is what it did
# before any of this existed.
#
# `wasm-bindgen` is a build-time tool and has to match the crate version in
# Cargo.lock: `cargo install wasm-bindgen-cli --version <that>`.
wasm:
	cargo build --target $(TARGET) --release -p switch-wasm --features gpu
	wasm-bindgen --target web \
	  --out-dir target/$(TARGET)/release \
	  target/$(TARGET)/release/switch_wasm.wasm

# The whole site, from web/index.html down: Vite follows the page to the
# stylesheet, the worker, the font and the core, and emits every one of them
# into dist/assets under a content-hashed name.
#
# This target exists (where `bun run dev`, `preview` and `typecheck` do not)
# because the core is an *input* to the frontend build rather than something
# copied in after it, and only make knows how to build the core.
assets: wasm
	bun run build
	@ls -la $(DIST) $(DIST)/assets

clean:
	cargo clean
	rm -rf $(DIST)
