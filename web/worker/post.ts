// The emulator worker's side of the page channel.

import type { PageMessage, WorkerMessage } from '../shared/protocol';

// `self` is typed as the shared WorkerGlobalScope, which lacks postMessage.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

export function postToPage(message: WorkerMessage): void {
  ctx.postMessage(message);
}

export function onPageMessage(handler: (message: PageMessage) => void): void {
  ctx.onmessage = (e: MessageEvent<PageMessage>) => handler(e.data);
}
