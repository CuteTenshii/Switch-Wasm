// Where the JavaScript and WebAssembly of a Chromium trace spent their time, by function:
//   node tools/trace_cpu.mjs <trace.json> [--thread=regex] [--rows=N]
// The trace comes from `browser_boot.mjs --trace=... --trace-cpu`. A wasm function has a name only
// when the core was built with its name section (`make wasm-release WASM_BINDGEN_FLAGS=`).
import { readFileSync } from 'node:fs';
import { commandLine, fail } from './common.mjs';

const { flag, positional } = commandLine();
if (!positional[0]) fail('usage: node tools/trace_cpu.mjs <trace.json> [--thread=regex] [--rows=N]');
const only = new RegExp(flag('thread') ?? 'DedicatedWorker');
const rows = Number(flag('rows') ?? 40);
const trace = JSON.parse(readFileSync(positional[0], 'utf8'));
const events = trace.traceEvents ?? trace;

const threadNames = new Map();
for (const e of events) {
  if (e.ph === 'M' && e.name === 'thread_name') threadNames.set(`${e.pid}:${e.tid}`, e.args.name);
}

// One profile per sampled thread, named by the event that started it; its chunks come from a
// profiler thread, and carry the nodes and the sample at each time delta.
const owners = new Map();
for (const e of events) {
  if (e.name === 'Profile') owners.set(e.id, `${threadNames.get(`${e.pid}:${e.tid}`) ?? 'thread'} ${e.tid}`);
}
const profiles = new Map();
for (const e of events) {
  if (e.name !== 'ProfileChunk') continue;
  const profile = profiles.get(e.id) ?? { thread: owners.get(e.id) ?? e.id, nodes: new Map(), samples: [], deltas: [] };
  profiles.set(e.id, profile);
  const data = e.args.data.cpuProfile;
  for (const node of data.nodes ?? []) profile.nodes.set(node.id, node);
  profile.samples.push(...data.samples ?? []);
  profile.deltas.push(...e.args.data.timeDeltas ?? []);
}

for (const profile of profiles.values()) {
  if (!only.test(profile.thread)) continue;
  const self = new Map();
  let total = 0;
  profile.samples.forEach((id, i) => {
    const frame = profile.nodes.get(id)?.callFrame;
    const name = frame ? `${frame.functionName || '(anonymous)'} ${(frame.url ?? '').split('/').pop()}:${frame.lineNumber}` : '(unknown)';
    const dt = profile.deltas[i] ?? 0;
    self.set(name, (self.get(name) ?? 0) + dt);
    total += dt;
  });
  console.log(`\n${profile.thread}: ${(total / 1000).toFixed(0)} ms sampled`);
  for (const [name, us] of [...self].sort((a, b) => b[1] - a[1]).slice(0, rows)) {
    console.log(`${(100 * us / total).toFixed(1).padStart(6)}%  ${(us / 1000).toFixed(0).padStart(7)} ms  ${name}`);
  }
}
