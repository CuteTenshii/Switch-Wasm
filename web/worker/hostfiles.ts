// Host files the wasm side reads by range through the synchronous `host_read` import.

import { workerLog } from './log';
import { core } from './wasm';

const HOST_CHUNK = 1 << 20;
const HOST_CACHE_CHUNKS = 32;
// Read-ahead on a sequential streak is 1 << run chunks.
const MAX_STREAK = 3;

// Reads out of one host file since the last `takeHostIo`.
export interface HostIo {
  file: string;
  reads: number;
  bytes: number;
  chunkMisses: number;
  diskBytes: number;
  failures: number;
}

let hostReader: FileReaderSync | null = null;

function reader(): FileReaderSync {
  return (hostReader ??= new FileReaderSync());
}

class HostFile {
  // Insertion order makes the chunk map an LRU.
  private readonly chunks = new Map<number, Uint8Array>();
  // The chunk a sequential reader misses next, and how many misses in a row were sequential.
  private nextMiss = -1;
  private streak = 0;
  io: HostIo | null = null;

  constructor(
    readonly blob: Blob,
    private readonly slot: number,
  ) {}

  private stats(): HostIo {
    return (this.io ??= {
      file: this.blob instanceof File
        ? this.blob.name
        : this.slot === 0 ? 'the container' : `host file ${this.slot}`,
      reads: 0,
      bytes: 0,
      chunkMisses: 0,
      diskBytes: 0,
      failures: 0,
    });
  }

  private readBlob(start: number, end: number): Uint8Array {
    const bytes = new Uint8Array(reader().readAsArrayBuffer(this.blob.slice(start, end)));
    this.stats().diskBytes += bytes.length;
    return bytes;
  }

  private chunk(index: number): Uint8Array {
    const hit = this.chunks.get(index);
    if (hit) {
      this.chunks.delete(index);
      this.chunks.set(index, hit);
      return hit;
    }
    this.stats().chunkMisses++;
    this.streak = index === this.nextMiss ? Math.min(this.streak + 1, MAX_STREAK) : 0;
    const count = Math.min(1 << this.streak, Math.ceil(this.blob.size / HOST_CHUNK) - index);
    this.nextMiss = index + count;
    const start = index * HOST_CHUNK;
    const span = this.readBlob(start, Math.min(start + count * HOST_CHUNK, this.blob.size));
    for (let i = 0; i < count; i++) {
      this.chunks.set(index + i, span.subarray(i * HOST_CHUNK, (i + 1) * HOST_CHUNK));
    }
    while (this.chunks.size > HOST_CACHE_CHUNKS) {
      this.chunks.delete(this.chunks.keys().next().value!);
    }
    return span.subarray(0, HOST_CHUNK);
  }

  // Fills `out` from `at`, returning how many bytes were filled.
  read(at: number, out: Uint8Array): number {
    const stats = this.stats();
    const end = at + out.length;
    stats.reads++;
    let written = 0;
    try {
      // Reads larger than a chunk bypass the cache.
      if (out.length > HOST_CHUNK) {
        out.set(this.readBlob(at, end));
        written = out.length;
      }
      while (written < out.length) {
        const index = Math.floor(at / HOST_CHUNK);
        const chunk = this.chunk(index);
        const from = at - index * HOST_CHUNK;
        const take = Math.min(chunk.length - from, end - at);
        if (take <= 0) break;
        out.set(chunk.subarray(from, from + take), written);
        written += take;
        at += take;
      }
    } catch (e) {
      stats.failures++;
      workerLog(`[io] ${stats.file}: read at ${at} failed: ${String(e)}`, 'error');
    }
    stats.bytes += written;
    return written;
  }
}

// File 0 is the container being run; the rest are system data archives.
const hostFiles: (HostFile | undefined)[] = [];

export function takeHostIo(): HostIo[] {
  const taken: HostIo[] = [];
  for (const file of hostFiles) {
    if (file?.io) taken.push(file.io);
    if (file) file.io = null;
  }
  return taken;
}

// Replaces slot 0 only: the wasm side addresses the other files by index.
export function openHostFile(file: Blob): bigint {
  reader();
  hostFiles[0] = new HostFile(file, 0);
  return BigInt(file.size);
}

// Slot 0 stays reserved for the container.
export function addHostFile(file: Blob): number {
  reader();
  const slot = Math.max(hostFiles.length, 1);
  hostFiles[slot] = new HostFile(file, slot);
  return slot;
}

// The wasm import: fill `len` bytes at `ptr` from `offset`, returning how many were filled.
export function hostRead(fileIndex: number, offset: bigint, ptr: number, len: number): number {
  const file = hostFiles[fileIndex >>> 0];
  if (!file || !len) return 0;
  const at = Number(offset);
  const size = Math.min(len >>> 0, file.blob.size - at);
  if (size <= 0) return 0;
  // Not cached: growing the heap detaches it.
  return file.read(at, new Uint8Array(core.memory.buffer, ptr >>> 0, size));
}
