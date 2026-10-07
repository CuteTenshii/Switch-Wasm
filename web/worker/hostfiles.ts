// Host files the wasm side reads by range through the synchronous `host_read` import.

import { workerLog } from './log';
import { api } from './wasm';

// File 0 is the container being run; the rest are system data archives.
let hostFiles: (Blob | null)[] = [];
let hostReader: FileReaderSync | null = null;

// Per-file chunk cache; Map insertion order makes it an LRU.
const HOST_CHUNK = 1 << 20;
const HOST_CACHE_CHUNKS = 16;
const hostChunks = new Map<number, Map<number, Uint8Array>>();

// Reads out of one host file since the last `takeHostIo`.
export interface HostIo {
  file: string;
  reads: number;
  bytes: number;
  chunkMisses: number;
  diskBytes: number;
  failures: number;
}

let io = new Map<number, HostIo>();

function ioOf(fileIndex: number): HostIo {
  let entry = io.get(fileIndex);
  if (!entry) {
    const file = hostFiles[fileIndex];
    const name = file instanceof File
      ? file.name
      : fileIndex === 0 ? 'the container' : `host file ${fileIndex}`;
    entry = { file: name, reads: 0, bytes: 0, chunkMisses: 0, diskBytes: 0, failures: 0 };
    io.set(fileIndex, entry);
  }
  return entry;
}

export function takeHostIo(): HostIo[] {
  const taken = [...io.values()];
  io = new Map();
  return taken;
}

function reader(): FileReaderSync {
  if (!hostReader) hostReader = new FileReaderSync();
  return hostReader;
}

// Replaces slot 0 only: the table can only grow within a session.
export function openHostFile(file: Blob): bigint {
  reader();
  if (hostFiles.length === 0) hostFiles = [null];
  hostFiles[0] = file;
  hostChunks.delete(0);
  return BigInt(file.size);
}

// Slot 0 stays reserved for the container.
export function addHostFile(file: Blob): number {
  reader();
  if (hostFiles.length === 0) hostFiles = [null];
  return hostFiles.push(file) - 1;
}

export function resetHostFiles(): void {
  hostFiles = [];
  hostChunks.clear();
}

function readBlob(stats: HostIo, file: Blob, start: number, end: number): Uint8Array {
  const bytes = new Uint8Array(reader().readAsArrayBuffer(file.slice(start, end)));
  stats.diskBytes += bytes.length;
  return bytes;
}

function hostChunk(stats: HostIo, file: Blob, fileIndex: number, index: number): Uint8Array {
  let cache = hostChunks.get(fileIndex);
  if (!cache) hostChunks.set(fileIndex, (cache = new Map<number, Uint8Array>()));
  const hit = cache.get(index);
  if (hit) {
    cache.delete(index);
    cache.set(index, hit);
    return hit;
  }
  stats.chunkMisses++;
  const start = index * HOST_CHUNK;
  const chunk = readBlob(stats, file, start, Math.min(start + HOST_CHUNK, file.size));
  cache.set(index, chunk);
  if (cache.size > HOST_CACHE_CHUNKS) {
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  return chunk;
}

// The wasm import: fill `len` bytes at `ptr` from `offset`, returning how many were filled.
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
  // Not cached: growing the heap detaches it.
  const out = new Uint8Array(api().memory.buffer, ptr, end - at);
  let written = 0;
  const stats = ioOf(fileIndex);
  stats.reads++;
  try {
    // Reads larger than a chunk bypass the cache.
    if (end - at > HOST_CHUNK) {
      out.set(readBlob(stats, file, at, end));
      stats.bytes += end - at;
      return end - at;
    }
    while (at < end) {
      const index = Math.floor(at / HOST_CHUNK);
      const chunk = hostChunk(stats, file, fileIndex, index);
      const from = at - index * HOST_CHUNK;
      const take = Math.min(chunk.length - from, end - at);
      if (take <= 0) break;
      out.set(chunk.subarray(from, from + take), written);
      written += take;
      at += take;
    }
  } catch (e) {
    stats.failures++;
    workerLog(`[io] ${stats.file}: read at ${at} failed: ${String(e)}`, 'err');
  }
  stats.bytes += written;
  return written;
}
