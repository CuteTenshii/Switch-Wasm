// Measure the build the browser actually runs:
//   node tools/wasm_bench.mjs <program.nro> [switch_wasm.wasm] [--frames=N]
//
// Reports what one steady-state frame costs, which is the number the
// frontend's frame rate is made of. Every other measurement in this repo is
// taken from a host binary — rustc's x86-64 backend, 64-bit pointers, an
// unbounded address space — and none of that ships. What ships is wasm32,
// recompiled by the browser with its own register allocator and a bounds check
// on every guest load. The two are not related by a constant, so a host
// measurement is not a scaled version of this one: removing a libcall only
// wasm pays for was ~1.15x natively and ~1.44x in the browser.
//
// This is the only tool here whose milliseconds mean anything. The host
// examples count work instead — `frame_work` reports the instructions, block
// entries, methods and draws a frame asks for, and those are the same numbers
// under V8. Fix what the counts name, then confirm it here.
//
// Needs `make wasm` to have been run: this loads that artefact rather than
// building its own, because a second build would need its own copy of the
// feature flags and the wasm-bindgen step and would then be measuring a
// module the site does not ship.
//
// With --cpu-prof node writes a .cpuprofile whose samples name the wasm
// functions, which is the only profiler available for the wasm build:
//   node --cpu-prof --cpu-prof-name=w.cpuprofile tools/wasm_bench.mjs prog.nro
import { readFileSync } from 'node:fs';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const nroPath = positional[0];
if (!nroPath) {
  console.error('usage: node tools/wasm_bench.mjs <program.nro> [switch_wasm.wasm] [--frames=N]');
  process.exit(1);
}
const release = join(root, 'target/wasm32-unknown-unknown/release');
const wasmPath = positional[1] || join(release, 'switch_wasm_bg.wasm');

// Frames to present before the clock starts. They cover the program's loader
// and its first upload, and they also let V8 tier the hot code from Liftoff up
// to TurboFan — timing a baseline-compiled frame is timing the compiler.
const WARMUP_FRAMES = 2;
const FRAMES = Number(process.argv.find((a) => a.startsWith('--frames='))?.slice(9)) || 8;

// The core is a wasm-bindgen module — wgpu reaches WebGPU through its glue —
// so it cannot be instantiated by hand any more: its imports are
// wasm-bindgen's own, and doing it the old way died on
// `__wbindgen_placeholder__`. Load it the way the worker does, through the
// generated glue.
//
// The glue's one bare specifier is `@host/files`, which the frontend build
// aliases to `web/worker/hostfiles.ts` and node cannot resolve. It is pointed
// at a stub here: an NRO is handed over as a buffer, so `host_read` is never
// called. The rewritten glue is written beside the original so that its own
// relative paths still resolve.
const stub = 'data:text/javascript,export const hostRead = () => 0;';
const gluePath = join(release, 'switch_wasm.js');
const benchGlue = join(release, 'switch_wasm.bench.mjs');
writeFileSync(
  benchGlue,
  readFileSync(gluePath, 'utf8').replace("'@host/files'", JSON.stringify(stub)),
);
const init = (await import(pathToFileURL(benchGlue).href)).default;
const api = await init({ module_or_path: readFileSync(wasmPath) });

function toWasm(bytes) {
  const ptr = api.switch_alloc(bytes.length);
  new Uint8Array(api.memory.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

function text(call) {
  const cap = 4096;
  const ptr = api.switch_alloc(cap);
  const n = call(ptr, cap);
  const out = new TextDecoder().decode(new Uint8Array(api.memory.buffer, ptr, n));
  api.switch_free(ptr, cap);
  return out;
}

const handle = api.switch_new();
try {
  const font = readFileSync(join(root, 'web/font.ttf'));
  api.switch_load_font(handle, toWasm(font), font.length);
} catch {
  // Not cosmetic: a guest with no font renders no text, which is a quarter
  // less work per frame in a menu, and a run without one looks like a faster
  // emulator rather than a smaller frame.
  console.log('no web/font.ttf: the guest will render no text, and this frame is not that frame');
}
const nro = readFileSync(nroPath);
const entry = api.switch_load_nro(handle, toWasm(nro), nro.length);
if (entry < 0n) {
  console.error('load failed');
  process.exit(1);
}

// The same slice size the frontend runs.
const SLICE = 5_000_000n;

// Run until `want` frames have been presented, timing nothing.
function reach(want) {
  let steps = 0n;
  while (api.switch_frame_count(handle) < want && !api.switch_halted(handle)) {
    const ran = api.switch_run(handle, SLICE);
    if (ran <= 0n) break;
    steps += ran;
  }
  return steps;
}

const boot = reach(WARMUP_FRAMES);
if (api.switch_frame_count(handle) < WARMUP_FRAMES) {
  console.error(`never presented ${WARMUP_FRAMES} frames: stopped at ${api.switch_frame_count(handle)}`);
  process.exit(1);
}
console.log(`warmup: ${boot} instructions to frame ${WARMUP_FRAMES}, not timed`);

// The frame counter is sampled between slices of one run rather than around
// two whole runs. A program spends most of a run booting — a retail title
// presents its first frame around step 900,000,000 — and that boot swings by
// seconds between runs, which is larger than the frames being measured, so
// subtracting two runs measures the swing. Sampling inside one has no such
// term.
const deltas = [];
let seen = api.switch_frame_count(handle);
let last = performance.now();
let steps = 0n;
const started = performance.now();
// One more than asked for: the first frame after the window opens is charged
// the tail of whatever the warmup was in the middle of.
while (deltas.length <= FRAMES && !api.switch_halted(handle)) {
  const ran = api.switch_run(handle, SLICE);
  if (ran < 0n) {
    console.error(`fault after ${steps} steps`);
    break;
  }
  if (ran === 0n) break;
  steps += ran;
  const now = api.switch_frame_count(handle);
  if (now === seen) continue;
  // A slice can carry more than one present; share its time out evenly rather
  // than charge the whole slice to the last of them.
  const at = performance.now();
  const each = (at - last) / (now - seen);
  for (let i = 0; i < now - seen; i++) deltas.push(each);
  seen = now;
  last = at;
}

const steady = deltas.slice(1);
if (steady.length === 0) {
  console.error('no frame was presented in the window');
  process.exit(1);
}
const sorted = [...steady].sort((a, b) => a - b);
const mean = steady.reduce((a, b) => a + b, 0) / steady.length;
const secs = (performance.now() - started) / 1000;

console.log(`${steady.length} frames after the first, under V8 (${process.version})`);
console.log(
  `  frame: mean ${mean.toFixed(1)} ms  min ${sorted[0].toFixed(1)} ms` +
    `  median ${sorted[sorted.length >> 1].toFixed(1)} ms  -> ${(1000 / mean).toFixed(2)} fps`,
);
console.log(`  cpu:   ${(Number(steps) / secs / 1e6).toFixed(1)} M instructions/s over ${secs.toFixed(2)}s`);
// The count `examples/frame_work.rs` reports for the same program. It should
// match: the two builds run the same emulator over the same guest and differ
// only in what compiled them. Where it does not, one of the two is not running
// what you think it is.
console.log(
  `  work:  ${(Number(steps) / deltas.length).toFixed(0)} instructions/frame,` +
    ` ${(Number(api.switch_guest_ram(handle)) / (1024 * 1024)).toFixed(1)} MiB guest RAM`,
);
console.log(`  jit:   ${text((ptr, cap) => api.switch_jit_stats_json(handle, ptr, cap))}`);
