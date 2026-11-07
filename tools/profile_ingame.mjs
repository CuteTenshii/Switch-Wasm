// Run a title through browser_boot.mjs, wait for a state worth measuring, then profile the page:
//   node tools/profile_ingame.mjs <container> | --nand=name [--when-fps=low-high] [--after=seconds]
//                                 [--settle=samples] [--record=seconds] [--dir=out] [browser_boot flags]
// Recording starts once `--settle` samples in a row, past `--after`, sit in the fps range. The report
// names the wasm functions of the busiest thread, and gives every thread's CPU in the page and the
// GPU process, which perf cannot attach to. Needs perf, and --seconds caps the wait.
import { spawn, spawnSync } from 'node:child_process';
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { commandLine, fail, repoRoot } from './common.mjs';

const OWN_FLAGS = ['when-fps', 'after', 'settle', 'record', 'dir'];
const SAMPLE_HZ = 500;
const TOP_INTERVAL_S = 5;
const FUNCTIONS_SHOWN = 45;

const { args, flag, positional } = commandLine();
if (!positional.length && !flag('nand')) {
  fail('usage: node tools/profile_ingame.mjs <container> | --nand=name [--when-fps=low-high]'
    + ' [--after=seconds] [--settle=samples] [--record=seconds] [--dir=out] [browser_boot flags]');
}
const [low, high] = (flag('when-fps') ?? '0.3-4').split('-').map(Number);
const after = Number(flag('after') ?? 0);
const settle = Number(flag('settle') ?? 6);
const record = Number(flag('record') ?? 40);
const dir = resolve(flag('dir') ?? 'profile');
mkdirSync(dir, { recursive: true });

const processes = () => spawnSync('ps', ['-eo', 'pid=,ppid=,pcpu=,args='], { encoding: 'utf8', maxBuffer: 1 << 26 })
  .stdout.split('\n').flatMap((line) => {
    const match = /^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$/.exec(line);
    return match ? [{ pid: Number(match[1]), ppid: Number(match[2]), cpu: Number(match[3]), args: match[4] }] : [];
  });

// The busiest process of a Chromium type under this run.
function browserProcess(rootPid, type) {
  const all = processes();
  const byPid = new Map(all.map((p) => [p.pid, p]));
  const under = (p) => {
    for (let at = p; at; at = byPid.get(at.ppid)) if (at.pid === rootPid) return true;
    return false;
  };
  return all.filter((p) => p.args.includes(`--type=${type}`) && under(p)).sort((a, b) => b.cpu - a.cpu)[0]?.pid;
}

function resolveSymbols(map) {
  if (!existsSync(map)) return [];
  return readFileSync(map, 'latin1').split('\n').flatMap((line) => {
    const match = /^([0-9a-f]+) ([0-9a-f]+) (.*)$/.exec(line);
    return match ? [{ start: BigInt(`0x${match[1]}`), size: BigInt(`0x${match[2]}`), name: match[3] }] : [];
  }).sort((a, b) => (a.start < b.start ? -1 : 1));
}

function symbolAt(symbols, ip) {
  let lo = 0;
  let hi = symbols.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (symbols[mid].start <= ip) lo = mid + 1;
    else hi = mid - 1;
  }
  const hit = symbols[hi];
  return hit && ip < hit.start + hit.size ? hit.name : null;
}

// perf does not apply a JIT map to anonymous V8 pages, so the samples are resolved here.
function reportPerf(data, map) {
  const script = spawnSync('perf', ['script', '-i', data, '-G', '-F', 'comm,tid,ip,dso'], { encoding: 'utf8', maxBuffer: 1 << 30 });
  const symbols = resolveSymbols(map);
  const threads = new Map();
  for (const line of script.stdout.split('\n')) {
    const match = /^\s*(.*?)\s+(\d+)\s+([0-9a-f]+)\s+\((.*)\)\s*$/.exec(line);
    if (!match) continue;
    const [, comm, tid, ip, dso] = match;
    const name = symbolAt(symbols, BigInt(`0x${ip}`)) ?? `[${dso.split('/').pop()}]`;
    const thread = threads.get(tid) ?? { comm, samples: 0, functions: new Map() };
    thread.samples += 1;
    thread.functions.set(name, (thread.functions.get(name) ?? 0) + 1);
    threads.set(tid, thread);
  }
  const ranked = [...threads].sort((a, b) => b[1].samples - a[1].samples);
  const out = ['Threads in the page, by samples:'];
  for (const [tid, thread] of ranked.slice(0, 8)) out.push(`${String(thread.samples).padStart(8)}  ${tid} ${thread.comm}`);
  for (const [tid, thread] of ranked.slice(0, 3)) {
    out.push('', `Functions on thread ${tid} (${thread.comm}), ${thread.samples} samples:`);
    const functions = [...thread.functions].sort((a, b) => b[1] - a[1]).slice(0, FUNCTIONS_SHOWN);
    for (const [name, samples] of functions) {
      out.push(`${(100 * samples / thread.samples).toFixed(2).padStart(7)}%  ${name.replace(/^(LazyCompile|Function|JS):/, '').slice(0, 100)}`);
    }
  }
  return out;
}

function reportTop(path, renderer, gpu) {
  const seen = new Map();
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const f = line.trim().split(/\s+/);
    if (f.length < 12 || !/^\d+$/.test(f[0])) continue;
    const key = f.slice(11).join(' ');
    seen.set(key, (seen.get(key) ?? 0) + Number(f[8]));
  }
  const rows = [...seen].sort((a, b) => b[1] - a[1]).slice(0, 15);
  const shots = Math.max(1, Math.round((record + TOP_INTERVAL_S) / TOP_INTERVAL_S));
  return ['', `CPU by thread name, page ${renderer} and GPU process ${gpu}:`,
    ...rows.map(([name, cpu]) => `${(cpu / shots).toFixed(1).padStart(7)}%  ${name}`)];
}

const forwarded = args.filter((a) => !OWN_FLAGS.some((name) => a.startsWith(`--${name}=`)));
if (!forwarded.some((a) => a.startsWith('--seconds='))) forwarded.push('--seconds=1500');
const boot = spawn(process.execPath, [join(repoRoot, 'tools/browser_boot.mjs'), ...forwarded,
  '--browser-arg=--no-sandbox',
  '--browser-arg=--js-flags=--perf-basic-prof --no-logfile-per-isolate --logfile=/dev/null',
], { stdio: ['ignore', 'pipe', 'inherit'] });
const bootLog = createWriteStream(join(dir, 'boot.txt'));
const ended = new Promise((done) => boot.on('close', done));

let inRange = 0;
let recording = false;
const lineReader = createInterface({ input: boot.stdout });
const triggered = new Promise((done) => {
  lineReader.on('line', (line) => {
    bootLog.write(`${line}\n`);
    const match = /^(\d+)s: running,.*?([\d.]+) fps/.exec(line);
    if (!match || recording) return;
    const at = Number(match[1]);
    const fps = Number(match[2]);
    inRange = at >= after && fps >= low && fps <= high ? inRange + 1 : 0;
    console.log(`${at}s: ${fps} fps, ${inRange}/${settle}`);
    if (inRange >= settle) {
      recording = true;
      done(true);
    }
  });
  ended.then(() => done(false));
});

if (!await triggered) {
  fail('the run ended before the fps range held; see boot.txt');
}
const renderer = browserProcess(boot.pid, 'renderer');
const gpu = browserProcess(boot.pid, 'gpu-process');
console.log(`recording ${record}s of page ${renderer}, GPU process ${gpu}`);
const data = join(dir, 'perf.data');
const topPath = join(dir, 'top.txt');
const topFile = createWriteStream(topPath);
const shots = String(Math.round(record / TOP_INTERVAL_S) + 1);
const top = spawn('top', ['-H', '-b', '-d', String(TOP_INTERVAL_S), '-n', shots, '-p', `${renderer},${gpu}`], { stdio: ['ignore', 'pipe', 'ignore'] });
top.stdout.pipe(topFile);
const perf = spawnSync('perf', ['record', '-q', '-e', 'cpu-clock:u', '-F', String(SAMPLE_HZ), '-p', String(renderer), '-o', data, '--', 'sleep', String(record)], { encoding: 'utf8' });
if (perf.status !== 0) console.error(`perf failed: ${perf.stderr.trim()}`);
await new Promise((done) => top.on('close', done));
boot.kill('SIGTERM');
await ended;
bootLog.end();

const report = [...(perf.status === 0 ? reportPerf(data, `/tmp/perf-${renderer}.map`) : []), ...reportTop(topPath, renderer, gpu)];
writeFileSync(join(dir, 'report.txt'), `${report.join('\n')}\n`);
console.log(report.join('\n'));
