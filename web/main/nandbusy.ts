// Whether a NAND operation is running. Nothing boots during one: a title started while the
// archives and applets are being registered finds some of them missing.

let running: string[] = [];
const listeners: (() => void)[] = [];

// Run `work` as the NAND operation `what` ("Registering system data archives"), so no title
// boots until it ends.
export async function nandOperation<T>(what: string, work: () => Promise<T>): Promise<T> {
  running.push(what);
  changed();
  try {
    return await work();
  } finally {
    const at = running.indexOf(what);
    running = running.filter((_, i) => i !== at);
    changed();
  }
}

// The NAND operation running, if any.
export function nandBusy(): string | null {
  return running[0] ?? null;
}

// Call `handler` whenever an operation starts or ends.
export function onNandBusy(handler: () => void): void {
  listeners.push(handler);
}

// Why `name` may not boot now, or null when it may.
export function bootRefusal(name: string): string | null {
  const busy = nandBusy();
  return busy
    ? name + ' was not started: ' + busy.toLowerCase() + ' is still running on the NAND.'
    + ' Start it again once that ends.'
    : null;
}

function changed(): void {
  for (const handler of listeners) handler();
}
