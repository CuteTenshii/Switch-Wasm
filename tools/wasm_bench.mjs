// Measure the build the browser actually runs:
//   node tools/wasm_bench.mjs <program.nro> [switch_wasm.wasm] [--no-jit]
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
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const nroPath = process.argv.slice(2).find((a) => !a.startsWith('--'));
if (!nroPath) {
  console.error('usage: node tools/wasm_bench.mjs <program.nro> [switch_wasm.wasm]');
  process.exit(1);
}
const release = join(root, 'target/wasm32-unknown-unknown/release');
const wasmPath =
  process.argv.slice(2).filter((a) => !a.startsWith('--'))[1] ||
  join(release, 'switch_wasm_bg.wasm');

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

const handle = api.switch_new();
// Block translation is on by default. `--no-jit` runs the plain interpreter,
// which is the comparison that says what translating is worth in V8 rather
// than under LLVM.
const noJit = process.argv.includes('--no-jit');
if (noJit) api.switch_set_jit(handle, 0);
try {
  const font = readFileSync(join(root, 'web/font.ttf'));
  api.switch_load_font(handle, toWasm(font), font.length);
} catch {
  console.log('no web/font.ttf: the guest will render no text');
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
console.log(
  `${steps} instructions in ${secs.toFixed(2)}s = ${(Number(steps) / secs / 1e6).toFixed(1)} M/s` +
    ` (${noJit ? 'interpreted' : 'translated'})`,
);
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
