// Measure the build the browser actually runs:
//   node tools/wasm_bench.mjs <container> [switch_wasm.wasm] [options]
//
//     --shot=<file.ppm>  save the final frame after timing for byte comparison
//     --frames=N          frames to time after the warmup (default 8)
//     --warmup=N          untimed frames to present first (default 2)
//     --keys=<file>       prod.keys; every encrypted container needs one
//     --title-keys=<file> title.keys, for content whose key is not bundled
//     --firmware=<dir>    register every .nca in it as a system data archive
//     --kind=<k>          override the header sniff: nsp, nca, nro or elf
//
// Reports steady-state frame time of the release wasm build (run
// `make wasm-release` first). Host timings do not transfer to wasm; use this
// for performance claims. For a wasm CPU profile:
//   node --cpu-prof --cpu-prof-name=w.cpuprofile tools/wasm_bench.mjs prog.nro
import { openSync, readSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const USAGE =
  'usage: node tools/wasm_bench.mjs <container> [switch_wasm.wasm]'
  + ' [--frames=N] [--warmup=N] [--shot=frame.ppm] [--keys=prod.keys] [--title-keys=title.keys] [--firmware=dir] [--kind=nsp|nca|nro|elf]';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const flag = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const containerPath = positional[0];
if (!containerPath) {
  console.error(USAGE);
  process.exit(1);
}
const release = join(root, 'target/wasm32-unknown-unknown/release');
const wasmPath = positional[1] || join(release, 'switch_wasm_bg.wasm');

// Untimed frames cover loading and let V8 tier up from Liftoff to TurboFan.
const WARMUP_FRAMES = Number(flag('warmup')) || 2;
const FRAMES = Number(flag('frames')) || 8;

// Load through the wasm-bindgen glue like the worker does, with `@host/files`
// rewritten to a shim over this file's reader.
const shim =
  'data:text/javascript,'
  + encodeURIComponent(
    'export const hostRead = (file, offset, ptr, len) =>'
    + ' globalThis.__benchHostRead(file, offset, ptr, len);',
  );
const gluePath = join(release, 'switch_wasm.js');
const benchGlue = join(release, 'switch_wasm.bench.mjs');
writeFileSync(
  benchGlue,
  readFileSync(gluePath, 'utf8').replace('\'@host/files\'', JSON.stringify(shim)),
);
const init = (await import(pathToFileURL(benchGlue).href)).default;
const api = await init({ module_or_path: readFileSync(wasmPath) });

// File 0 is the container; the rest are system data archives, as in the worker.
const hostFiles = [];

function addHostFile(path) {
  const fd = openSync(path, 'r');
  return hostFiles.push({ fd, size: statSync(path).size }) - 1;
}

// Mirrors `web/worker/hostfiles.ts`'s LRU so the wasm side sees the same access pattern.
const HOST_CHUNK = 1 << 20;
const HOST_CACHE_CHUNKS = 16;
const hostChunks = new Map();

function hostChunk(file, fileIndex, index) {
  let cache = hostChunks.get(fileIndex);
  if (!cache) hostChunks.set(fileIndex, (cache = new Map()));
  const hit = cache.get(index);
  if (hit) {
    cache.delete(index);
    cache.set(index, hit);
    return hit;
  }
  const start = index * HOST_CHUNK;
  const want = Math.min(HOST_CHUNK, file.size - start);
  const chunk = new Uint8Array(Math.max(want, 0));
  if (want > 0) readSync(file.fd, chunk, 0, want, start);
  cache.set(index, chunk);
  if (cache.size > HOST_CACHE_CHUNKS) {
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  return chunk;
}

// Fill `len` bytes at `ptr` from `offset` of host file `fileIndex`; return the count.
globalThis.__benchHostRead = (fileIndex, offset, ptr, len) => {
  ptr >>>= 0;
  len >>>= 0;
  fileIndex >>>= 0;
  const file = hostFiles[fileIndex];
  if (!file || !len) return 0;
  let at = Number(offset);
  const end = Math.min(at + len, file.size);
  if (at >= end) return 0;
  // Build the view here: heap growth detaches cached views.
  const out = new Uint8Array(api.memory.buffer, ptr, end - at);
  // Reads larger than a chunk bypass the cache.
  if (end - at > HOST_CHUNK) return readSync(file.fd, out, 0, end - at, at);
  let written = 0;
  while (at < end) {
    const index = Math.floor(at / HOST_CHUNK);
    const chunk = hostChunk(file, fileIndex, index);
    const from = at - index * HOST_CHUNK;
    const take = Math.min(chunk.length - from, end - at);
    if (take <= 0) break;
    out.set(chunk.subarray(from, from + take), written);
    written += take;
    at += take;
  }
  return written;
};

function toWasm(bytes) {
  const ptr = api.switch_alloc(bytes.length) >>> 0;
  new Uint8Array(api.memory.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

function text(call) {
  const cap = 4096;
  const ptr = api.switch_alloc(cap) >>> 0;
  const n = call(ptr, cap);
  const out = new TextDecoder().decode(new Uint8Array(api.memory.buffer, ptr, n));
  api.switch_free(ptr, cap);
  return out;
}

const handle = api.switch_new();
const lastError = () => text((ptr, cap) => api.switch_last_error(handle, ptr, cap));

function die(what) {
  const why = lastError();
  console.error(why ? `${what}: ${why}` : what);
  process.exit(1);
}

// Pick the loader as the core does: PFS0, then NRO (possibly behind a boot
// stub), then cartridge "HEAD" at 0x100; a bare NCA is the fallback.
function sniff(head) {
  const u32 = (at) => head.length >= at + 4 && head.readUInt32LE(at);
  if (u32(0) === 0x464c457f) return 'elf'; // "\x7fELF"
  if (u32(0) === 0x30534650) return 'nsp'; // "PFS0"
  for (let at = 0; at + 4 <= Math.min(head.length, 0x100); at += 4) {
    if (u32(at) === 0x304f524e) return 'nro'; // "NRO0"
  }
  if (u32(0x100) === 0x44414548) return 'nsp'; // "HEAD", a cartridge image
  return 'nca';
}

const headBytes = Buffer.alloc(0x200);
{
  const fd = openSync(containerPath, 'r');
  readSync(fd, headBytes, 0, headBytes.length, 0);
}
const kind = flag('kind') || sniff(headBytes);
if (!['nsp', 'nca', 'nro', 'elf'].includes(kind)) die(`unknown --kind=${kind}`);

const prod = flag('keys');
const titleKeys = flag('title-keys');
if (prod || titleKeys) {
  const p = prod ? readFileSync(prod) : null;
  const t = titleKeys ? readFileSync(titleKeys) : null;
  const ok = api.switch_load_keys(
    handle,
    p ? toWasm(p) : 0,
    p ? p.length : 0,
    t ? toWasm(t) : 0,
    t ? t.length : 0,
  );
  if (ok !== 0) die('could not parse the keys');
}

try {
  const font = readFileSync(join(root, 'web/font.ttf'));
  api.switch_load_font(handle, toWasm(font), font.length);
} catch {
  // Without a font a guest renders no text, which skews the frame cost.
  console.log('no web/font.ttf: the guest will render no text, and this frame is not that frame');
}

// Slot 0 is the container however it is loaded.
const containerIndex = addHostFile(containerPath);
const containerSize = hostFiles[containerIndex].size;

const firmware = flag('firmware');
if (firmware) {
  let added = 0;
  for (const name of readdirSync(firmware).sort()) {
    if (!name.toLowerCase().endsWith('.nca')) continue;
    const index = addHostFile(join(firmware, name));
    if (api.switch_add_archive(handle, index, BigInt(hostFiles[index].size)) === 0) added++;
  }
  console.log(`firmware: ${added} data archive(s) registered from ${firmware}`);
}

let entry;
if (kind === 'nro' || kind === 'elf') {
  const bytes = readFileSync(containerPath);
  entry =
    kind === 'nro'
      ? api.switch_load_nro(handle, toWasm(bytes), bytes.length)
      : api.switch_load_elf(handle, toWasm(bytes), bytes.length);
} else if (kind === 'nsp') {
  if (api.switch_open_nsp(handle, BigInt(containerSize)) !== 0) die('could not open the container');
  const index = api.switch_program_nca_index(handle);
  if (index < 0) die('nothing to boot in this container');
  entry = api.switch_load_nca_from_nsp(handle, index);
} else {
  if (api.switch_open_nca(handle, BigInt(containerSize)) !== 0) die('could not open the NCA');
  entry = api.switch_load_nca(handle);
}
// An entry of 0 is valid for some NSO layouts; only -1 is failure.
if (entry < 0n) die('load failed');
console.log(
  `${kind}: ${containerPath} (${(containerSize / (1024 * 1024)).toFixed(1)} MiB),`
  + ` entry ${'0x' + entry.toString(16)}`,
);

const SLICE = 1_000_000n;
// Progress output during long boots.
const REPORT_EVERY = 500_000_000n;

function reach(want) {
  let steps = 0n;
  let said = 0n;
  while (api.switch_frame_count(handle) < want && !api.switch_halted(handle)) {
    const ran = api.switch_run(handle, SLICE);
    if (ran <= 0n) break;
    steps += ran;
    if (steps - said >= REPORT_EVERY) {
      said = steps;
      process.stderr.write(
        `\r  booting: ${steps} instructions, ${api.switch_frame_count(handle)} frame(s)   `,
      );
    }
  }
  if (said > 0n) process.stderr.write('\n');
  return steps;
}

const boot = reach(WARMUP_FRAMES);
if (api.switch_frame_count(handle) < WARMUP_FRAMES) {
  const why = lastError();
  console.error(
    `never presented ${WARMUP_FRAMES} frames: stopped at ${api.switch_frame_count(handle)}`
    + (why ? ` (${why})` : ''),
  );
  process.exit(1);
}
console.log(`warmup: ${boot} instructions to frame ${WARMUP_FRAMES}, not timed`);

// Sample the frame counter between slices of one run; boot time varies too much
// to subtract two runs.
const deltas = [];
let seen = api.switch_frame_count(handle);
let last = performance.now();
let steps = 0n;
const started = performance.now();
// One extra frame: the first absorbs the tail of the warmup.
while (deltas.length <= FRAMES && !api.switch_halted(handle)) {
  const ran = api.switch_run(handle, SLICE);
  if (ran < 0n) {
    console.error(`fault after ${steps} steps: ${lastError()}`);
    break;
  }
  if (ran === 0n) break;
  steps += ran;
  const now = api.switch_frame_count(handle);
  if (now === seen) continue;
  // Split a multi-present slice's time evenly.
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
  `  frame: mean ${mean.toFixed(1)} ms  min ${sorted[0].toFixed(1)} ms`
  + `  median ${sorted[sorted.length >> 1].toFixed(1)} ms  -> ${(1000 / mean).toFixed(2)} fps`,
);
console.log(`  cpu:   ${(Number(steps) / secs / 1e6).toFixed(1)} M instructions/s over ${secs.toFixed(2)}s`);
// Should match `examples/frame_work.rs` for the same program.
console.log(
  `  work:  ${(Number(steps) / deltas.length).toFixed(0)} instructions/frame,`
  + ` ${(Number(api.switch_guest_ram(handle)) / (1024 * 1024)).toFixed(1)} MiB guest RAM`,
);
console.log(`  jit:   ${text((ptr, cap) => api.switch_jit_stats_json(handle, ptr, cap))}`);

// Capture outside the timed window.
const shot = flag('shot');
if (shot) {
  const width = api.switch_fb_width(handle);
  const height = api.switch_fb_height(handle);
  const len = width * height * 4;
  const ptr = api.switch_alloc(len) >>> 0;
  try {
    const copied = api.switch_fb_snapshot(handle, ptr, len);
    if (copied !== len) throw new Error(`incomplete framebuffer: ${copied}/${len} bytes`);
    const rgba = new Uint8Array(api.memory.buffer, ptr, len);
    const rgb = Buffer.alloc(width * height * 3);
    for (let i = 0, j = 0; i < len; i += 4, j += 3) {
      rgb[j] = rgba[i];
      rgb[j + 1] = rgba[i + 1];
      rgb[j + 2] = rgba[i + 2];
    }
    writeFileSync(shot, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), rgb]));
    console.log(`  shot:  frame ${api.switch_frame_count(handle)} -> ${shot}`);
  } finally {
    api.switch_free(ptr, len);
  }
}
