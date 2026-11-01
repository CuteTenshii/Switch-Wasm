// Where a Chromium trace's threads spent their time, by event name:
//   node tools/trace_summary.mjs <trace.json> [--thread=regex]
// Durations are exclusive of events nested inside them, so a row is time in that event itself.
import { readFileSync } from 'node:fs';
import { commandLine, fail } from './common.mjs';

const ROWS = 30;
const { flag, positional } = commandLine();
if (!positional[0]) fail('usage: node tools/trace_summary.mjs <trace.json> [--thread=regex]');
const only = new RegExp(flag('thread') ?? 'CrGpuMain|DedicatedWorker');
const trace = JSON.parse(readFileSync(positional[0], 'utf8'));
const events = trace.traceEvents ?? trace;

const names = new Map();
for (const e of events) {
  if (e.ph === 'M' && e.name === 'thread_name') names.set(`${e.pid}:${e.tid}`, e.args.name);
}

const threads = new Map();
for (const e of events) {
  const key = `${e.pid}:${e.tid}`;
  if (!only.test(names.get(key) ?? '') || e.ph !== 'X' || !e.dur) continue;
  if (!threads.has(key)) threads.set(key, []);
  threads.get(key).push(e);
}

for (const [key, list] of threads) {
  list.sort((a, b) => a.ts - b.ts || b.dur - a.dur);
  const self = new Map();
  const counts = new Map();
  const stack = [];
  let first = Infinity;
  let last = 0;
  for (const e of list) {
    first = Math.min(first, e.ts);
    last = Math.max(last, e.ts + e.dur);
    while (stack.length && stack.at(-1).ts + stack.at(-1).dur <= e.ts) stack.pop();
    const parent = stack.at(-1);
    if (parent) self.set(parent.name, (self.get(parent.name) ?? 0) - e.dur);
    self.set(e.name, (self.get(e.name) ?? 0) + e.dur);
    counts.set(e.name, (counts.get(e.name) ?? 0) + 1);
    stack.push(e);
  }
  const span = last - first;
  const busy = [...self.values()].reduce((a, b) => a + b, 0);
  console.log(`\n${names.get(key)}: ${(busy / 1000).toFixed(0)} ms busy of ${(span / 1000).toFixed(0)} ms traced`);
  for (const [name, us] of [...self].sort((a, b) => b[1] - a[1]).slice(0, ROWS)) {
    console.log(`${(100 * us / span).toFixed(1).padStart(6)}%  ${(us / 1000).toFixed(0).padStart(8)} ms  ${String(counts.get(name)).padStart(7)}x  ${name}`);
  }
}
