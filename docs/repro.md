# Reproducing and debugging

Every tool for getting a title to misbehave outside a browser, and what each
one is for. AGENTS.md's *Commands* covers the build; this covers the runs.

The `.nro`/`.nsp` files are gitignored.

- `--example screenshot <nro> out.ppm 3` — writes the third presented frame,
  feeding in `web/font.ttf` as the shared font unless told otherwise.
- `--example boot_nx <nro>` — shortest path to "does it halt cleanly".
- `--example boot_nsp <nsp> <prod.keys> [title.keys] [steps]` — the browser's
  Launch button, without a browser. `SHOT=<f.ppm>` writes a frame; prefer it
  over reading `frames presented: 0` off a budget too short to reach one.
  `UPDATE=<update.nsp>` runs the title patched and `DLC=<a.nsp>,<b.nsp>` mounts
  its add-on content, which is the page's pairing of the containers with no
  page in the way.
- **The shared debugging knobs**, on `boot_nsp` and `screenshot_title`
  alike: `TRAP_WRITE=<addr>:<hex size>` and `TRAP_READ=…` name
  the code that writes to or reads a region, `WATCH_PC=<addr>[,...]` the code
  that reaches one, and `DUMP=<base>[+<hex>][:<hex len>][,...]` hex-dumps guest
  memory wherever the run stopped — `<base>` being a register, `sp`, `pc` or an
  address, as in `DUMP=x23+0x1830:0x40`. Each used to live in one tool and be
  spelled differently in the next.
- Every tool that reads a container takes `<container> <prod.keys>
  [title.keys]`, in that order: the container may be an NSP, an XCI or a bare
  NCA (decided by its header, not its name), and `title.keys` may be left out
  when `prod.keys` already opens the title. A cartridge image is read through
  its `secure` partition — its `update` partition is a firmware bundle and is
  never the title.
- `--example dump_exefs …` — flat module images at their real load addresses
  plus a sorted `symbols.txt`. **This is what makes a retail backtrace
  readable.** `--example disasm_flat` disassembles them there.
- `--example retail_trace …` — a ring buffer of the last N instructions, dumped
  on halt or fault. `RING_MIN` past `rtld` (`0x08004000`), whose lazy-binding
  resolver would otherwise fill the whole ring. `MARK`/`MARK_DUMP` watch an API
  being called in order without recording the steps between.
- `--example jit_difftest <nro>` — both engines, with every state difference.
  `SWITCH_NO_JIT=1` disables translation for host tools.
- `--example screenshot_title <container> <prod.keys> [title.keys] out.ppm [n]`
  — the Nth *presented* frame of a retail title, either container kind. Stops
  at the frame rather than at a step budget, which `boot_nsp SHOT=` cannot: a
  title needs billions of steps to reach its first, and picking a budget that
  lands after it is guesswork. Was `screenshot_nsp` and `screenshot_nca`.
- **Checking the WebGPU backend**: run `screenshot_title` and `switch-gpu`'s
  `screenshot_gpu` over the same frame and `cmp` the PPMs. `GPU_ONLY=<i>` puts
  only the i-th draw on the device, so a difference is exactly one draw's.
- Tracing is one mask with nineteen channels (`TRACE_IPC`, `TRACE_SVC`,
  `TRACE_WAIT`, `TRACE_NV`, `TRACE_GPU`, ...). The environment seeds it, so a
  CLI run sets them exactly as it always did; in the browser the same nineteen
  are checkboxes under **Diagnostic channels** in the debug panel, because
  `std::env::var` always fails on wasm. `switch_core::trace::ALL` is the list
  both read from.
- `--example opus_testvectors <dir>` — the Opus decoder against the RFC 8251
  vectors (`opus_testvectors-rfc8251.tar.gz` from opus-codec.org). It fails on
  the first packet whose range coder state disagrees with the encoder's, and
  writes `<name>.rs.dec` beside each vector for `opus_compare` to score.
  `--example opus_difftest <dir>` does the same against a reference decode you
  generate yourself, which is how the output rates below 48 kHz and the
  multi-stream layouts get covered.
- Browser: `make wasm` once, then `bun run dev`. Tests: `make test`.

