/* screen wake lock

   A running emulator is a page the browser reads as idle: minutes go by with
   nothing typed into it, so the screen dims and the machine sleeps in the
   middle of whatever was on it. The Screen Wake Lock API is how a page says
   otherwise. It needs a secure context, and Firefox and Safari before 16.4
   never shipped it, so elsewhere the page behaves exactly as it did.

   The browser takes the lock back whenever the document stops being visible -
   a switched tab, a locked phone - and does not return it, so a lock that is
   still wanted has to be re-taken on the way back. */

// Whether the run loop currently wants the screen kept awake, which is not the
// same as holding a lock: a request in flight has neither, and a hidden
// document has the want without the lock.
let wanted = false;
let held: WakeLockSentinel | null = null;

async function acquire(): Promise<void> {
  if (!('wakeLock' in navigator) || held || document.visibilityState !== 'visible') return;
  try {
    const sentinel = await navigator.wakeLock.request('screen');
    // The run can end while the request is in flight, and a lock nobody wants
    // any more would otherwise be held until the next visibility change.
    if (!wanted) {
      await sentinel.release();
      return;
    }
    held = sentinel;
    sentinel.addEventListener('release', () => {
      if (held === sentinel) held = null;
    });
  } catch {
    // Refused - an unsupported policy, a battery saver, a document that went
    // hidden mid-request. The screen dims; nothing else about the run changes.
  }
}

/** Keep the screen awake until `releaseWakeLock`. Safe to call when a lock is
 *  already held. */
export function holdWakeLock(): void {
  wanted = true;
  void acquire();
}

export function releaseWakeLock(): void {
  wanted = false;
  const sentinel = held;
  held = null;
  void sentinel?.release().catch(() => {
    // Already gone - the browser releases on hide, and says so out of band.
  });
}

document.addEventListener('visibilitychange', () => {
  if (wanted) void acquire();
});
