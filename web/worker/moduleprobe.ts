// Imports the module it is sent: the import's error names the module that failed and why, which a
// worker's own error event leaves out. Terminated once it replies, so a module that loads runs no further.

import type { ProbeReply } from '../shared/spawn';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = async (e: MessageEvent<string>) => {
  let reply: ProbeReply;
  try {
    await import(/* @vite-ignore */ e.data);
    reply = { probed: null };
  } catch (err) {
    reply = { probed: String(err) };
  }
  ctx.postMessage(reply);
};
