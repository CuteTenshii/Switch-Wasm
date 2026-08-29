// Measure the build the browser actually runs:
//   node tools/wasm_bench.mjs <container> [switch_wasm.wasm] [options]
//
//     --frames=N          frames to time after the warmup (default 8)
//     --keys=<file>       prod.keys; every encrypted container needs one
//     --title-keys=<file> title.keys, for content whose key is not bundled
//     --firmware=<dir>    register every .nca in it as a system data archive
//     --kind=<k>          override the header sniff: nsp, nca, nro or elf
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
// Any container the page accepts is accepted here, by the same calls in the
// same order: an `.nsp` or a cartridge image through `switch_open_nsp` (both,
// because their partitions flatten into one table), a bare Program `.nca`
// through `switch_open_nca`, and homebrew through `switch_load_nro` or
// `switch_load_elf`. A retail container is never staged in memory — it stays
// on disk and is read a range at a time through `host_read`, which is the one
// thing wasm32 leaves no choice about.
//
// Needs `make wasm` to have been run: this loads that artefact rather than
// building its own, because a second build would need its own copy of the
// feature flags and the wasm-bindgen step and would then be measuring a
// module the site does not ship.
//
// With --cpu-prof node writes a .cpuprofile whose samples name the wasm
// functions, which is the only profiler available for the wasm build:
//   node --cpu-prof --cpu-prof-name=w.cpuprofile tools/wasm_bench.mjs prog.nro
import { openSync, readSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const USAGE =
  'usage: node tools/wasm_bench.mjs <container> [switch_wasm.wasm]' +
  ' [--frames=N] [--keys=prod.keys] [--title-keys=title.keys] [--firmware=dir] [--kind=nsp|nca|nro|elf]';

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

// Frames to present before the clock starts. They cover the program's loader
// and its first upload, and they also let V8 tier the hot code from Liftoff up
// to TurboFan — timing a baseline-compiled frame is timing the compiler.
const WARMUP_FRAMES = 2;
const FRAMES = Number(flag('frames')) || 8;

// The core is a wasm-bindgen module — wgpu reaches WebGPU through its glue —
// so it cannot be instantiated by hand any more: its imports are
// wasm-bindgen's own, and doing it the old way died on
// `__wbindgen_placeholder__`. Load it the way the worker does, through the
// generated glue.
//
// The glue's one bare specifier is `@host/files`, which the frontend build
// aliases to `web/worker/hostfiles.ts` and node cannot resolve. It is pointed
// at a shim that forwards to this file's own reader, because a retail
// container is read through `host_read` rather than handed over. The rewritten
// glue is written beside the original so that its own relative paths still
// resolve.
const shim =
  'data:text/javascript,' +
  encodeURIComponent(
    'export const hostRead = (file, offset, ptr, len) =>' +
      ' globalThis.__benchHostRead(file, offset, ptr, len);',
  );
const gluePath = join(release, 'switch_wasm.js');
const benchGlue = join(release, 'switch_wasm.bench.mjs');
writeFileSync(
  benchGlue,
  readFileSync(gluePath, 'utf8').replace("'@host/files'", JSON.stringify(shim)),
);
const init = (await import(pathToFileURL(benchGlue).href)).default;
const api = await init({ module_or_path: readFileSync(wasmPath) });

// File 0 is the container being run; the rest are system data archives, which
// a title mounts by data id. Same shape as the worker's table, so the indices
// the wasm side is given mean the same thing on both.
const hostFiles = [];

function addHostFile(path) {
  const fd = openSync(path, 'r');
  return hostFiles.push({ fd, size: statSync(path).size }) - 1;
}

// The browser reads through a 1 MiB LRU because `FileReaderSync` is expensive
// per call and the guest asks for a few hundred bytes at a time as it walks
// its RomFS tables. `readSync` is cheaper than that, but the point of this
// tool is what the site serves, so the access pattern the wasm side sees is
// kept identical to `web/worker/hostfiles.ts` — same chunk size, same depth,
// same bypass for reads too large to cache — rather than made faster than
// anything that ships.
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

// Fill `len` bytes at `ptr` from `offset` of host file `fileIndex`, and return
// how many were filled. `offset` arrives as a BigInt (it is an i64) and
// `ptr`/`len` as signed i32s, hence the `>>> 0`.
globalThis.__benchHostRead = (fileIndex, offset, ptr, len) => {
  ptr >>>= 0;
  len >>>= 0;
  fileIndex >>>= 0;
  const file = hostFiles[fileIndex];
  if (!file || !len) return 0;
  let at = Number(offset);
  const end = Math.min(at + len, file.size);
  if (at >= end) return 0;
  // The view has to be built here, not cached: growing the heap detaches it.
  const out = new Uint8Array(api.memory.buffer, ptr, end - at);
  // A read bigger than a chunk is the ExeFS being pulled in one go. Serve it
  // straight from the file: it would evict the whole cache on its way through
  // and never be asked for again.
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
const lastError = () => text((ptr, cap) => api.switch_last_error(handle, ptr, cap));

function die(what) {
  const why = lastError();
  console.error(why ? `${what}: ${why}` : what);
  process.exit(1);
}

// Which loader a file wants, by the same evidence the core uses. A PFS0 magic
// at 0 settles it; a cartridge's "HEAD" sits 0x100 in, where an `.nsp`'s
// string table can spell anything a repacker put in a file name, so it is only
// consulted once PFS0 has been ruled out. An NRO's magic may sit behind a boot
// stub (hbmenu prepends one), so the first 0x100 bytes are scanned for it.
// A bare NCA cannot be sniffed at all — its header is encrypted — so it is
// what is left when nothing else matches.
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

// Keys before the container: opening one needs none, but finding the Program
// NCA inside it decrypts a header per file, and a data archive is parsed as it
// is registered.
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
  // Not cosmetic: a guest with no font renders no text, which is a quarter
  // less work per frame in a menu, and a run without one looks like a faster
  // emulator rather than a smaller frame.
  console.log('no web/font.ttf: the guest will render no text, and this frame is not that frame');
}

// Slot 0 is the container whichever way it is loaded, so that a homebrew run
// and a retail one number their archives the same.
const containerIndex = addHostFile(containerPath);
const containerSize = hostFiles[containerIndex].size;

// A system applet needs these far more than a game does — its fonts, icons and
// settings all live in firmware — and nothing is read until a title asks for
// one, so registering a directory costs a header parse per file.
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
// An entry of 0 is legitimate for some NSO layouts, so -1 is the only failure.
if (entry < 0n) die('load failed');
console.log(
  `${kind}: ${containerPath} (${(containerSize / (1024 * 1024)).toFixed(1)} MiB),` +
    ` entry ${'0x' + entry.toString(16)}`,
);

// The same slice size the frontend runs (`web/main/runloop.ts`).
const SLICE = 1_000_000n;
// A retail title spends billions of instructions before its first frame, and a
// silent tool is indistinguishable from a hung one for the minutes that takes.
const REPORT_EVERY = 500_000_000n;

// Run until `want` frames have been presented, timing nothing.
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
    `never presented ${WARMUP_FRAMES} frames: stopped at ${api.switch_frame_count(handle)}` +
      (why ? ` (${why})` : ''),
  );
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
    console.error(`fault after ${steps} steps: ${lastError()}`);
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
