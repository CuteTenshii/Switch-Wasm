// Screen wake lock while running. The browser drops it when the document is
// hidden, so a still-wanted lock is re-taken on visibility.

// Whether the run loop wants the screen awake, independent of holding a lock.
let wanted = false;
let held: WakeLockSentinel | null = null;

async function acquire(): Promise<void> {
  if (!('wakeLock' in navigator) || held || document.visibilityState !== 'visible') return;
  try {
    const sentinel = await navigator.wakeLock.request('screen');
    // The run may have ended while the request was in flight.
    if (!wanted) {
      await sentinel.release();
      return;
    }
    held = sentinel;
    sentinel.addEventListener('release', () => {
      if (held === sentinel) held = null;
    });
  } catch {
    // Refused: the screen dims, nothing else changes.
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
