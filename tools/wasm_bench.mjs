// Measure the build the browser actually runs:
//   node tools/wasm_bench.mjs <program.nro> [switch_wasm.wasm]
//
// Reports emulated instructions per second and the cost of one steady-state
// frame, which is the number that matters for the frontend's frame rate. The
// native `bench`/`hotspots` examples and this script disagree often enough to be
// worth checking both: V8 inlines differently than LLVM does for the host, so a
// change that helps one can do nothing for the other.
//
// With --cpu-prof node writes a .cpuprofile whose samples name the wasm
// functions, which is the only profiler available for the wasm build:
//   node --cpu-prof --cpu-prof-name=w.cpuprofile tools/wasm_bench.mjs prog.nro
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const nroPath = process.argv[2];
if (!nroPath) {
  console.error('usage: node tools/wasm_bench.mjs <program.nro> [switch_wasm.wasm]');
  process.exit(1);
}
const wasmPath = process.argv[3] || join(root, 'target/wasm32-unknown-unknown/release/switch_wasm.wasm');

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const api = instance.exports;

function toWasm(bytes) {
  const ptr = api.switch_alloc(bytes.length);
  new Uint8Array(api.memory.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

const handle = api.switch_new();
api.switch_set_syscall_mode(handle, 2); // Horizon
try {
  const font = readFileSync(join(root, 'web/assets/font.ttf'));
  api.switch_load_font(handle, toWasm(font), font.length);
} catch {
  console.log('no web/assets/font.ttf: the guest will render no text');
}
const nro = readFileSync(nroPath);
const entry = api.switch_load_nro(handle, toWasm(nro), nro.length);
if (entry < 0n) {
  console.error('load failed');
  process.exit(1);
}

// The same slice size the frontend runs.
const SLICE = 5_000_000n;
const FRAMES = 4;
const started = performance.now();
let steps = 0n;
let frames = 0;
const marks = [];
while (frames < FRAMES) {
  const ran = api.switch_run(handle, SLICE);
  if (ran < 0n) {
    console.error('fault after', steps, 'steps');
    break;
  }
  steps += ran;
  const now = api.switch_frame_count(handle);
  if (now !== frames) {
    frames = now;
    marks.push({ frame: now, steps: Number(steps), ms: performance.now() - started });
  }
  if (api.switch_halted(handle)) break;
}

const secs = (performance.now() - started) / 1000;
console.log(`${steps} instructions in ${secs.toFixed(2)}s = ${(Number(steps) / secs / 1e6).toFixed(1)} M/s`);
for (const m of marks) {
  console.log(`  frame ${m.frame}: ${m.steps} instructions, ${(m.ms / 1000).toFixed(2)}s`);
}
if (marks.length >= 2) {
  const prev = marks[marks.length - 2];
  const last = marks[marks.length - 1];
  const ms = last.ms - prev.ms;
  console.log(
    `  steady frame: ${last.steps - prev.steps} instructions, ` +
      `${(ms / 1000).toFixed(2)}s -> ${(1000 / ms).toFixed(2)} fps`,
  );
}
