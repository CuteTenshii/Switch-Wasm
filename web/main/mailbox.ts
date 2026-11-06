// Controller state and rumble, shared with the core's linear memory so no message carries them.

// Word offsets into the core's `Mailbox`.
const BUTTONS = 0;
const PRESSED = 1;
const STICKS = 2;
const RUMBLE = 6;
const WORDS = 7;

let words: Int32Array | null = null;
let down = 0;

export function setMailbox(buffer: SharedArrayBuffer, at: number): void {
  words = new Int32Array(buffer, at, WORDS);
  down = 0;
}

// A press is also noted on its own, so a tap shorter than a slice is still seen.
export function writeInput(buttons: number, sticks: readonly number[]): void {
  if (!words) return;
  const newly = buttons & ~down;
  down = buttons;
  if (newly) Atomics.or(words, PRESSED, newly);
  Atomics.store(words, BUTTONS, buttons);
  for (let i = 0; i < 4; i++) Atomics.store(words, STICKS + i, sticks[i]);
}

// The last rumble the guest asked for, packed as `(weak << 16) | strong`.
export function readRumble(): number {
  return words ? Atomics.load(words, RUMBLE) : 0;
}
