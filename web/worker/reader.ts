// Fills the host file cache for the emulator: the emulator names slots of shared buffers, and
// this worker reads each chunk into one and signals it, so a file read blocks only the emulator.

const CHUNK = 1 << 20;

interface Message {
  type: 'done' | 'file' | 'slot' | 'fetch';
  buffer?: SharedArrayBuffer;
  index?: number;
  blob?: Blob;
  slot?: number;
  file?: number;
  chunk?: number;
  size?: number;
  slots?: number[];
  generations?: number[];
}

const reader = new FileReaderSync();
const files: Blob[] = [];
const slots: Uint8Array[] = [];
let done: Int32Array;

self.onmessage = (e: MessageEvent<Message>) => {
  const m = e.data;
  if (m.type === 'done') done = new Int32Array(m.buffer!);
  else if (m.type === 'file') files[m.index!] = m.blob!;
  else if (m.type === 'slot') slots[m.slot!] = new Uint8Array(m.buffer!);
  else fetchChunks(m);
};

// Tells the system worker this one runs: its reads block the thread that starts it.
self.postMessage('running');

// Each slot is answered as soon as its chunk is in, with the generation it was asked for under.
function fetchChunks(m: Message): void {
  const blob = files[m.file!];
  m.slots!.forEach((slot, i) => {
    const start = (m.chunk! + i) * CHUNK;
    const generation = m.generations![i];
    let answer = -generation;
    try {
      const bytes = reader.readAsArrayBuffer(blob.slice(start, Math.min(start + CHUNK, m.size!)));
      slots[slot].set(new Uint8Array(bytes));
      answer = generation;
    } catch { /* the emulator is told through the sign */ }
    Atomics.store(done, slot, answer);
    Atomics.notify(done, slot);
  });
}
