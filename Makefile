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

# Compile the wasm bindings crate.
wasm:
	cargo build --target $(TARGET) --release -p switch-wasm

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
