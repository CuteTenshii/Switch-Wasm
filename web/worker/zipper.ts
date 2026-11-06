// Builds one zip off the emulator worker, whose slices a long checksum would hold up.

import { zip, type ZipEntry } from './zip';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

export type ZipReply = { blob: Blob } | { error: string };

ctx.onmessage = (e: MessageEvent<ZipEntry[]>) => {
  let reply: ZipReply;
  try {
    reply = { blob: zip(e.data) };
  } catch (err) {
    reply = { error: String(err) };
  }
  ctx.postMessage(reply);
};
