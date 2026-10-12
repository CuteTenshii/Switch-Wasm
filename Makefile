.PHONY: all test wasm wasm-release assets clean

TARGET := wasm32-unknown-unknown
DIST   := dist

# The browser build shares one memory between workers, which needs std rebuilt
# with atomics, and only nightly can. Host builds and tests stay on stable.
# Both binaries by path: a stable toolchain on PATH would otherwise supply rustc.
NIGHTLY       ?= nightly-2026-10-11
NIGHTLY_CARGO := $(shell rustup which --toolchain $(NIGHTLY) cargo 2>/dev/null)
NIGHTLY_RUSTC := $(shell rustup which --toolchain $(NIGHTLY) rustc 2>/dev/null)
NIGHTLY_HINT  := rustup toolchain install $(NIGHTLY) --profile minimal --component rust-src --target $(TARGET)

all: test assets

# Host test suite.
test:
	cargo test -p switch-core
	cargo test -p switch-wasm
	# wgpu device tests share the host adapter; parallel creation can crash software Vulkan.
	cargo test -p switch-gpu -- --test-threads=1

# The bindings crate with the WebGPU backend: `wasm` is the dev build `bun run dev`
# serves, `wasm-release` the shipped fat-LTO one. `wasm-bindgen` must match Cargo.lock.
wasm: PROFILE := debug
wasm-release: PROFILE := release
wasm-release: RELEASE_FLAG := --release
wasm wasm-release:
	$(if $(NIGHTLY_CARGO),,$(error $(NIGHTLY) is not installed; run: $(NIGHTLY_HINT)))
	RUSTC=$(NIGHTLY_RUSTC) $(NIGHTLY_CARGO) build -Z build-std=std,panic_abort \
		--target $(TARGET) $(RELEASE_FLAG) -p switch-wasm --features gpu
	wasm-bindgen --target web --out-dir target/$(TARGET)/$(PROFILE) \
		target/$(TARGET)/$(PROFILE)/switch_wasm.wasm

# The Vite site in dist, which takes the release core as an input.
assets: wasm-release
	bun run build
	@ls -la $(DIST) $(DIST)/assets

clean:
	cargo clean
	rm -rf $(DIST)
