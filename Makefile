.PHONY: all test wasm demo assets serve clean

TARGET := wasm32-unknown-unknown
WASM   := target/$(TARGET)/release/switch_wasm.wasm
WEB    := web/assets

all: test wasm assets

# Host test suite: parsers, memory, CPU interpreter, loaders, demo boot.
test:
	cargo test -p switch-core

# Compile the wasm bindings crate.
wasm:
	cargo build --target $(TARGET) --release -p switch-wasm

# Copy build artifacts into the web/ tree and (re)generate the demo payload.
assets: wasm demo
	cp $(WASM) $(WEB)/switch_wasm.wasm
	@ls -la $(WEB)

# Regenerate the bundled demo NRO (hand-assembled homebrew).
demo:
	cargo run -p make-demo

# Serve the frontend locally.
serve: assets
	python3 -m http.server 8000 --directory web

clean:
	cargo clean
	rm -f $(WEB)/switch_wasm.wasm
