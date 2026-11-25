import probeUrl from '../worker/moduleprobe.ts?worker&url';

// The probe's answer: the error importing the module gave, or null when it loaded.
export type ProbeReply = { probed: string | null };

/**
 * Starts the module worker at `url` (a `?worker&url` import) and passes `onFail` the browser's own
 * error when it fails to load or throws. `type: 'module'` must match Vite's `worker.format: 'es'`.
 */
export function spawnWorker(url: string, name: string, onFail: (why: string) => void): Worker {
  const worker = new Worker(url, { type: 'module', name });
  worker.addEventListener('error', (e) => {
    if (e instanceof ErrorEvent && e.message) {
      onFail(`${name}: ${e.message}` + (e.filename ? ` at ${e.filename}:${e.lineno}:${e.colno}` : ''));
      return;
    }
    // A worker that fails to load gets an error without a reason; importing its module gives one.
    void importError(url).then((why) => onFail(why ?? `${name} failed to load, then loaded when retried`));
  });
  return worker;
}

function importError(url: string): Promise<string | null> {
  const probe = new Worker(probeUrl, { type: 'module', name: 'Module probe' });
  return new Promise((resolve) => {
    probe.onmessage = (e: MessageEvent<unknown>) => {
      // The module, once it loads, can post messages of its own first.
      if (typeof e.data !== 'object' || e.data === null || !('probed' in e.data)) return;
      probe.terminate();
      resolve((e.data as ProbeReply).probed);
    };
    probe.onerror = () => {
      probe.terminate();
      resolve(null);
    };
    probe.postMessage(new URL(url, location.href).href);
  });
}
