.PHONY: all test wasm wasm-release assets clean

TARGET := wasm32-unknown-unknown
# `wasm` is the dev profile build `bun run dev` serves; `wasm-release` is the shipped, fat-LTO build.
DEV     := target/$(TARGET)/debug
RELEASE := target/$(TARGET)/release
DIST    := dist

all: test assets

# Host test suite.
test:
	cargo test -p switch-core
	cargo test -p switch-wasm
	# wgpu device tests share the host adapter; parallel creation can crash software Vulkan.
	cargo test -p switch-gpu -- --test-threads=1

# The wasm bindings crate with the WebGPU backend. `wasm-bindgen` must match the version in Cargo.lock.
wasm:
	cargo build --target $(TARGET) -p switch-wasm --features gpu
	wasm-bindgen --target web --out-dir $(DEV) $(DEV)/switch_wasm.wasm

wasm-release:
	cargo build --target $(TARGET) --release -p switch-wasm --features gpu
	wasm-bindgen --target web --out-dir $(RELEASE) $(RELEASE)/switch_wasm.wasm

# The Vite site in dist, which takes the release core as an input.
assets: wasm-release
	bun run build
	@ls -la $(DIST) $(DIST)/assets

clean:
	cargo clean
	rm -rf $(DIST)
