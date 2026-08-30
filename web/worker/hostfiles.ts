/* The files the wasm side reads without ever holding them.

   A retail .nsp runs to several gigabytes: more than a wasm32 module can
   address, let alone allocate in one buffer, so it is never handed over as
   one. The File stays with the browser and the wasm side pulls ranges out of
   it through the `host_read` import below.

   That import has to answer *synchronously* - the emulator asks for RomFS
   ranges from inside `switch_run`, with nowhere to await a promise - and
   `FileReaderSync` exists only in a worker, which is the second reason the
   emulator lives in one.

   A `Blob` is as good as a `File` here, and that is what the NAND hands over:
   content the page stored in IndexedDB comes back as a handle the browser
   owns the bytes of, so a firmware dump is registered for the cost of its
   headers rather than pulled through the wasm heap. */

import { api } from './wasm';

// File 0 is the container being run; the rest are system data archives the
// page has added, which a title mounts by data id.
let hostFiles: (Blob | null)[] = [];
let hostReader: FileReaderSync | null = null;

// Reads land in bursts of a few hundred bytes as the guest walks its RomFS
// tables, so whole chunks are kept around; a Map iterates in insertion order,
// which makes it an LRU with no bookkeeping of its own.
//
// One cache per file rather than one shared between them. A title running
// with an update reads two containers at once - every relocation crossing
// swaps from one to the other - and a single LRU has them evicting each
// other's working set at each crossing. The budget is per file and half what
// the shared one was, so the pair costs what one file used to.
const HOST_CHUNK = 1 << 20;
const HOST_CACHE_CHUNKS = 16;
const hostChunks = new Map<number, Map<number, Uint8Array>>();

function reader(): FileReaderSync {
  if (!hostReader) hostReader = new FileReaderSync();
  return hostReader;
}

// Opening a container replaces slot 0 and leaves the rest alone: the wasm
// side holds sources that address archives by index, so the table can only
// ever grow. Only slot 0's cached chunks go with it.
export function openHostFile(file: Blob): bigint {
  reader();
  if (hostFiles.length === 0) hostFiles = [null];
  hostFiles[0] = file;
  hostChunks.delete(0);
  return BigInt(file.size);
}

// Add a file the wasm side can read, and return its index. Slot 0 stays
// reserved for the container even if nothing has been opened yet.
export function addHostFile(file: Blob): number {
  reader();
  if (hostFiles.length === 0) hostFiles = [null];
  return hostFiles.push(file) - 1;
}

// Every source addressing this table dies with the session that holds it, so
// the table only has to grow within one session. Clearing it when a session is
// freed is what keeps a NAND full of archives from costing another hundred
// handles - and another hundred chunk caches - on every reset.
export function resetHostFiles(): void {
  hostFiles = [];
  hostChunks.clear();
}

function readBlob(file: Blob, start: number, end: number): Uint8Array {
  return new Uint8Array(reader().readAsArrayBuffer(file.slice(start, end)));
}

function hostChunk(file: Blob, fileIndex: number, index: number): Uint8Array {
  let cache = hostChunks.get(fileIndex);
  if (!cache) hostChunks.set(fileIndex, (cache = new Map()));
  const hit = cache.get(index);
  if (hit) {
    cache.delete(index);
    cache.set(index, hit);
    return hit;
  }
  const start = index * HOST_CHUNK;
  const chunk = readBlob(file, start, Math.min(start + HOST_CHUNK, file.size));
  cache.set(index, chunk);
  if (cache.size > HOST_CACHE_CHUNKS) {
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  return chunk;
}

// The wasm import: fill `len` bytes at `ptr` from `offset` of the open file,
// and return how many were filled. `offset` arrives as a BigInt (it is an
// i64), and `ptr`/`len` as signed i32s, hence the `>>> 0`.
export function hostRead(
  fileIndex: number,
  offset: bigint,
  ptr: number,
  len: number,
): number {
  ptr >>>= 0;
  len >>>= 0;
  fileIndex >>>= 0;
  const file = hostFiles[fileIndex];
  if (!file || !len) return 0;
  let at = Number(offset);
  const end = Math.min(at + len, file.size);
  if (at >= end) return 0;
  // The view has to be built here, not cached: growing the heap detaches it.
  const out = new Uint8Array(api().memory.buffer, ptr, end - at);
  let written = 0;
  try {
    // A read bigger than a chunk is the ExeFS being pulled in one go. Serve
    // it straight from the file: it would evict the whole cache on its way
    // through and never be asked for again.
    if (end - at > HOST_CHUNK) {
      out.set(readBlob(file, at, end));
      return end - at;
    }
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
  } catch (e) {
    // The file was moved or replaced while it was open. Report the short
    // read; the wasm side turns that into an error with an offset on it.
    console.error('[switch-wasm] host read failed:', e);
  }
  return written;
}
