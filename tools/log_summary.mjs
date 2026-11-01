// Boil a saved run log down to what is usually read out of it:
//   node tools/log_summary.mjs <log.txt>
// Takes `browser_boot.mjs --out` reports and the page's Save log files.
import { existsSync, readFileSync } from 'node:fs';
import { commandLine, fail } from './common.mjs';

const TIMELINE_ROWS = 12;
const LIST_ROWS = 15;
const THREAD_DUMPS = 30;

const [path] = commandLine().positional;
if (!path) fail('usage: node tools/log_summary.mjs <log.txt>');
if (!existsSync(path)) fail(`${path}: no such file`);
const lines = readFileSync(path, 'utf8').split('\n');
if (!lines.some(Boolean)) fail(`${path}: empty`);

const heading = (text) => console.log(`\n${text}`);
const unique = (rows) => [...new Set(rows)];
const count = (text) => {
  const match = /^([\d,.]+)([kMB]?)$/.exec(text);
  if (!match) return 0;
  const scale = { '': 1, 'k': 1e3, 'M': 1e6, 'B': 1e9 }[match[2]];
  return Number(match[1].replaceAll(',', '')) * scale;
};
const short = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n)}`);

const samples = lines.flatMap((line) => {
  const match = /^(\d+)s: (\w+), ([^,]*?)(?: \(([^)]*)\))?, ([\d.]+) fps/.exec(line);
  return match ? [{ at: Number(match[1]), state: match[2], total: match[3], rate: match[4] ?? '', fps: match[5] }] : [];
});
if (samples.length) {
  heading('Timeline');
  const step = Math.max(1, Math.ceil(samples.length / TIMELINE_ROWS));
  const shown = samples.filter((_, i) => i % step === 0 || i === samples.length - 1);
  for (const s of shown) {
    console.log(`${String(s.at).padStart(6)}s  ${s.state.padEnd(8)} ${s.fps.padStart(6)} fps  ${s.total.padStart(8)}  ${s.rate}`);
  }
}

heading('Faults');
const faults = unique(lines.filter((line) =>
  /svcBreak|Halted|did not start|device was lost|the page crashed|\[fatal\]|terminate result/.test(line))
  .map((line) => line.slice(0, 160)));
console.log(faults.length ? faults.slice(0, LIST_ROWS).join('\n') : 'none');

heading('Refused commands');
const refused = new Map();
for (const line of lines) {
  const match = /unanswered: (.+?) \(refused\)(?: x(\d+))?/.exec(line);
  if (match) refused.set(match[1], Math.max(refused.get(match[1]) ?? 0, Number(match[2] ?? 1)));
}
const refusedRows = [...refused].sort((a, b) => b[1] - a[1]);
console.log(refusedRows.length
  ? refusedRows.slice(0, LIST_ROWS).map(([name, times]) => `${String(times).padStart(6)}x  ${name}`).join('\n')
  : 'none');

const input = unique(lines.flatMap((line) => /\[input\] (.*)/.exec(line)?.[1] ?? []));
if (input.length) {
  heading('Input');
  console.log(input.join('\n'));
}

const start = lines.indexOf('=== rendering ===');
if (start >= 0) {
  heading('Rendering');
  const end = lines.indexOf('=== the page\'s console ===', start);
  const section = lines.slice(start + 1, end < 0 ? undefined : end);
  for (const label of ['Draws on the device', 'Fell back', 'Rejected by the device', 'Surfaces held', 'Read from guest memory']) {
    const at = section.indexOf(label);
    if (at >= 0) console.log(`${label}: ${section[at + 1]}`);
  }
  const why = section.indexOf('Fell back because');
  if (why >= 0) console.log(`Fell back because: ${section.slice(why + 1).filter(Boolean).slice(0, 3).join('; ')}`);
  const timers = section.indexOf('Stage\tPer frame\tTotal');
  if (timers >= 0) {
    for (const row of section.slice(timers + 1)) {
      const [stage, perFrame, total] = row.split('\t');
      if (total === undefined) break;
      console.log(`  ${stage.padEnd(16)} ${perFrame.padStart(8)} ms/frame ${total.padStart(11)} ms`);
    }
  }
}

const frames = lines.filter((line) => /\[gpu\] \d+ frames? \(/.test(line));
if (frames.length) {
  heading('Last frames');
  console.log(frames.slice(-3).map((line) => line.replace(/^\w> /, '')).join('\n'));
}

const THREAD_LINE = /\[thread\] (\d+)\*? (?:\("([^"]*)"|\(([^,)]*))[^:]*: ([\d,.]+[kMB]?) instructions/;
const dumps = lines.flatMap((line) => {
  const match = THREAD_LINE.exec(line);
  return match ? [match] : [];
});
if (dumps.length) {
  heading(`Busiest threads, last ${THREAD_DUMPS * 10} thread lines`);
  const totals = new Map();
  for (const [, id, quoted, plain, instructions] of dumps.slice(-THREAD_DUMPS * 10)) {
    // Unnamed threads would otherwise merge under a shared entry point.
    const name = `${id} ${quoted ?? plain}`;
    totals.set(name, (totals.get(name) ?? 0) + count(instructions));
  }
  const sum = [...totals.values()].reduce((a, b) => a + b, 0);
  for (const [name, n] of [...totals].sort((a, b) => b[1] - a[1]).slice(0, 8)) {
    console.log(`${String(Math.round(100 * n / sum)).padStart(5)}%  ${short(n).padStart(8)}  ${name}`);
  }
}
