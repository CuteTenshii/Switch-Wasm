.PHONY: all test wasm demo assets serve clean

TARGET := wasm32-unknown-unknown
WASM   := target/$(TARGET)/release/switch_wasm.wasm
WEB    := web/assets

all: test wasm assets

# Host test suite: parsers, memory, CPU interpreter, loaders, demo boot, and
# the host-facing wasm entry points (which build for the host too, so the SD
# card's import/export API is covered without a browser).
test:
	cargo test -p switch-core
	cargo test -p switch-wasm

# Compile the wasm bindings crate.
wasm:
	cargo build --target $(TARGET) --release -p switch-wasm

# Copy build artifacts into the web/ tree.
assets: wasm
	cp $(WASM) $(WEB)/switch_wasm.wasm
	@ls -la $(WEB)

# Serve the frontend locally (no-cache headers so the browser never reuses a
# stale .wasm/.nro — python's http.server would otherwise let Firefox
# heuristically cache them).
serve: assets
	python3 tools/serve.py

clean:
	cargo clean
	rm -f $(WEB)/switch_wasm.wasm
