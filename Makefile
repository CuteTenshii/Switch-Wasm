.PHONY: all test wasm wasm-release assets clean

TARGET := wasm32-unknown-unknown
# Two builds of the module. `wasm` is cargo's dev profile, incremental and
# spread over every core, which is what `bun run dev` serves. `wasm-release`
# is the shipped one: fat LTO into a single codegen unit, which one core
# spends a minute on, and the only build performance is quoted from.
DEV     := target/$(TARGET)/debug
RELEASE := target/$(TARGET)/release
DIST    := dist

all: test assets

# Host test suite: parsers, memory, CPU interpreter, loaders, demo boot, and
# the host-facing wasm entry points (which build for the host too, so the SD
# card's import/export API is covered without a browser).
test:
	cargo test -p switch-core
	cargo test -p switch-wasm
	# wgpu device tests share the host adapter; parallel creation can crash the
	# software Vulkan driver used by headless CI runners.
	cargo test -p switch-gpu -- --test-threads=1

# Compile the wasm bindings crate, with the WebGPU backend.
#
# Always with it. `wgpu` reaches WebGPU through `wasm-bindgen`, so the
# artefact is a wasm-bindgen module with generated glue beside it rather than
# a bare one the worker hands to `WebAssembly.instantiateStreaming`, and
# carrying two shapes of core, two loaders and two answers to every question
# about the build costs more than the megabyte it would save. A machine
# without WebGPU still runs: the backend reports that it could not open a
# device and the software rasterizer takes the frame, which is what it did
# before any of this existed.
#
# `wasm-bindgen` is a build-time tool and has to match the crate version in
# Cargo.lock: `cargo install wasm-bindgen-cli --version <that>`.
wasm:
	cargo build --target $(TARGET) -p switch-wasm --features gpu
	wasm-bindgen --target web --out-dir $(DEV) $(DEV)/switch_wasm.wasm

wasm-release:
	cargo build --target $(TARGET) --release -p switch-wasm --features gpu
	wasm-bindgen --target web --out-dir $(RELEASE) $(RELEASE)/switch_wasm.wasm

# The whole site, from web/index.html down: Vite follows the page to the
# stylesheet, the worker, the font and the core, and emits every one of them
# into dist/assets under a content-hashed name.
#
# This target exists (where `bun run dev`, `preview` and `typecheck` do not)
# because the core is an *input* to the frontend build rather than something
# copied in after it, and only make knows how to build the core.
assets: wasm-release
	bun run build
	@ls -la $(DIST) $(DIST)/assets

clean:
	cargo clean
	rm -rf $(DIST)
