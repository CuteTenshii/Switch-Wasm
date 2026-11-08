// What a `Worker`'s error event says went wrong, phrased to follow the worker's name. A script
// that fails to load gets a plain `Event`: browsers withhold the reason, and only the console names the module.
export function workerErrorText(e: Event): string {
  if (e instanceof ErrorEvent && e.message) {
    return e.filename ? `threw ${e.message} at ${e.filename}:${e.lineno}:${e.colno}` : `threw ${e.message}`;
  }
  return 'could not load one of its modules (the browser console names which)';
}
