/* The files the wasm side reads without ever holding them.

   A retail .nsp runs to several gigabytes: more than a wasm32 module can
   address, let alone allocate in one buffer, so it is never handed over as
   one. The File stays with the browser and the wasm side pulls ranges out of
   it through the `host_read` import below.

   That import has to answer *synchronously* - the emulator asks for RomFS
   ranges from inside `switch_run`, with nowhere to await a promise - and
   `FileReaderSync` exists only in a worker, which is the second reason the
   emulator lives in one. */

import { api } from './wasm';

// File 0 is the container being run; the rest are system data archives the
// page has added, which a title mounts by data id.
let hostFiles: (File | null)[] = [];
let hostReader: FileReaderSync | null = null;

// Reads land in bursts of a few hundred bytes as the guest walks its RomFS
// tables, so whole chunks are kept around; a Map iterates in insertion order,
// which makes it an LRU with no bookkeeping of its own.
const HOST_CHUNK = 1 << 20;
const HOST_CACHE_CHUNKS = 32;
const hostChunks = new Map<string, Uint8Array>();

function reader(): FileReaderSync {
  if (!hostReader) hostReader = new FileReaderSync();
  return hostReader;
}

// Opening a container replaces slot 0 and leaves the rest alone: the wasm
// side holds sources that address archives by index, so the table can only
// ever grow. The chunk cache is keyed by index too, hence the flush.
export function openHostFile(file: File): bigint {
  reader();
  if (hostFiles.length === 0) hostFiles = [null];
  hostFiles[0] = file;
  hostChunks.clear();
  return BigInt(file.size);
}

// Add a file the wasm side can read, and return its index. Slot 0 stays
// reserved for the container even if nothing has been opened yet.
export function addHostFile(file: File): number {
  reader();
  if (hostFiles.length === 0) hostFiles = [null];
  return hostFiles.push(file) - 1;
}

function readBlob(file: File, start: number, end: number): Uint8Array {
  return new Uint8Array(reader().readAsArrayBuffer(file.slice(start, end)));
}

function hostChunk(file: File, index: number, key: string): Uint8Array {
  const hit = hostChunks.get(key);
  if (hit) {
    hostChunks.delete(key);
    hostChunks.set(key, hit);
    return hit;
  }
  const start = index * HOST_CHUNK;
  const chunk = readBlob(file, start, Math.min(start + HOST_CHUNK, file.size));
  hostChunks.set(key, chunk);
  if (hostChunks.size > HOST_CACHE_CHUNKS) {
    const oldest = hostChunks.keys().next();
    if (!oldest.done) hostChunks.delete(oldest.value);
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
  const file = hostFiles[fileIndex >>> 0];
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
      const chunk = hostChunk(file, index, `${fileIndex}:${index}`);
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
