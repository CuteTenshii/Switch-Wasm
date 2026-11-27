// Screen wake lock while running. The browser drops it when the document is
// hidden, so a still-wanted lock is re-taken on visibility.

// Whether the run loop wants the screen awake, independent of holding a lock.
let wanted = false;
let held: WakeLockSentinel | null = null;
// A request in flight; a second one would take a lock nothing releases.
let acquiring = false;

async function acquire(): Promise<void> {
  if (!('wakeLock' in navigator) || held || acquiring || document.visibilityState !== 'visible') return;
  acquiring = true;
  try {
    const sentinel = await navigator.wakeLock.request('screen');
    // The run may have ended while the request was in flight.
    if (!wanted) {
      void sentinel.release().catch(() => {
        // Already released by the browser.
      });
      return;
    }
    held = sentinel;
    sentinel.addEventListener('release', () => {
      if (held === sentinel) held = null;
    });
  } catch {
    // Refused: the screen dims, nothing else changes.
  } finally {
    acquiring = false;
  }
}

// Keep the screen awake until `releaseWakeLock`. Idempotent.
export function holdWakeLock(): void {
  wanted = true;
  void acquire();
}

export function releaseWakeLock(): void {
  wanted = false;
  const sentinel = held;
  held = null;
  void sentinel?.release().catch(() => {
    // Already released by the browser.
  });
}

document.addEventListener('visibilitychange', () => {
  if (wanted) void acquire();
});
