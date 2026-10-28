// The debug panel's Memory group: the title's guest RAM over time and by region of its
// address space, and the WebAssembly heap everything lives in.

import type { GpuReport, Housekeeping, MemoryRegionKey, MemoryReport } from '../shared/protocol';
import { $, el } from './dom';
import { formatBytes } from './format';
import { call, hasSession } from './rpc';
import { titleBooted } from './session';
import { setNote } from './shell';

const HISTORY_MS = 5 * 60 * 1000;
const RATE_WINDOW_MS = 10 * 1000;
// Housekeeping arrives several times a second; a five-minute line needs one sample a second.
const SAMPLE_EVERY_MS = 1000;
const WASM_LIMIT = 4 * 1024 * 1024 * 1024;

const REGION_LABELS: Record<MemoryRegionKey, string> = {
  code: 'Program and its data',
  stacks: 'Thread stacks',
  runtime: 'Main stack and TLS',
  loaded: 'Modules loaded at run time',
  heap: 'Heap',
  alias: 'Alias region',
};

interface Sample {
  at: number;
  bytes: number;
}

let history: Sample[] = [];
let peak = 0;
// The sample the chart's readout is on, or -1 to follow the newest.
let picked = -1;

// Housekeeping's figure is the title's, so an applet on screen does not dip the line.
export function noteRam(ram: Housekeeping['ram']): void {
  const at = performance.now();
  peak = Math.max(peak, ram.guest);
  const last = history[history.length - 1];
  if (last && at - last.at < SAMPLE_EVERY_MS) return;
  history.push({ at, bytes: ram.guest });
  while (at - history[0].at > HISTORY_MS) {
    history.shift();
    if (picked > 0) picked -= 1;
  }
}

// A new title starts a new line.
export function resetMemoryHistory(): void {
  history = [];
  peak = 0;
  picked = -1;
}

const memorySection = $<HTMLDetailsElement>('memory-debug-section');
const heapSection = $<HTMLDetailsElement>('heap-debug-section');
const chart = $('memory-chart') as unknown as SVGSVGElement;
const SVG = 'http://www.w3.org/2000/svg';
const WIDTH = 300;
const HEIGHT = 72;

function hex(address: number): string {
  return '0x' + address.toString(16).padStart(8, '0');
}

function rate(): number | null {
  const last = history[history.length - 1];
  const first = history.find((s) => last && last.at - s.at <= RATE_WINDOW_MS);
  if (!last || !first || last.at === first.at) return null;
  return (last.bytes - first.bytes) / ((last.at - first.at) / 1000);
}

function formatRate(perSecond: number | null): string {
  if (perSecond === null) return '-';
  if (perSecond === 0) return 'steady';
  return (perSecond > 0 ? '+' : '-') + formatBytes(Math.abs(Math.round(perSecond))) + '/s';
}

function secondsAgo(sample: Sample): number {
  return Math.round((performance.now() - sample.at) / 1000);
}

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
}

function drawChart(): void {
  const readout = $('memory-readout');
  const scale = $('memory-scale');
  if (history.length < 2) {
    chart.replaceChildren();
    scale.textContent = '';
    readout.textContent = 'Collecting samples while the title runs.';
    return;
  }
  const now = performance.now();
  const top = Math.max(...history.map((s) => s.bytes)) * 1.1 || 1;
  const x = (s: Sample) => WIDTH * (1 - (now - s.at) / HISTORY_MS);
  const y = (bytes: number) => HEIGHT - 2 - (bytes / top) * (HEIGHT - 4);
  const points = history.map((s) => `${x(s).toFixed(1)},${y(s.bytes).toFixed(1)}`).join(' ');
  const shown = history[picked < 0 ? history.length - 1 : picked];
  const nodes: SVGElement[] = [
    svg('line', { class: 'memory-baseline', x1: 0, x2: WIDTH, y1: HEIGHT - 1, y2: HEIGHT - 1 }),
    svg('polyline', { class: 'memory-line', points }),
  ];
  if (picked >= 0) {
    nodes.push(svg('line', { class: 'memory-rule', x1: x(shown), x2: x(shown), y1: 0, y2: HEIGHT }));
  }
  chart.replaceChildren(...nodes);
  const marker = $('memory-marker');
  marker.hidden = picked < 0;
  marker.style.left = `${(x(shown) / WIDTH) * 100}%`;
  marker.style.top = `${(y(shown.bytes) / HEIGHT) * 100}%`;
  scale.textContent = `0 to ${formatBytes(top)}`;
  const ago = secondsAgo(shown);
  readout.textContent = `${formatBytes(shown.bytes)} backed, ${ago ? `${ago} s ago` : 'now'}`;
}

function pick(index: number): void {
  picked = Math.max(0, Math.min(history.length - 1, index));
  drawChart();
}

function release(): void {
  picked = -1;
  drawChart();
}

chart.addEventListener('pointermove', (e) => {
  if (history.length < 2) return;
  const box = chart.getBoundingClientRect();
  const at = performance.now() - HISTORY_MS * (1 - (e.clientX - box.left) / box.width);
  let nearest = 0;
  for (let i = 1; i < history.length; i++) {
    if (Math.abs(history[i].at - at) < Math.abs(history[nearest].at - at)) nearest = i;
  }
  pick(nearest);
});
chart.addEventListener('pointerleave', release);
chart.addEventListener('blur', release);
chart.addEventListener('keydown', (e) => {
  if (history.length < 2) return;
  const from = picked < 0 ? history.length - 1 : picked;
  const moves: Record<string, number> = {
    ArrowLeft: from - 1, ArrowRight: from + 1, Home: 0, End: history.length - 1,
  };
  if (e.key === 'Escape') {
    release();
  } else if (e.key in moves) {
    e.preventDefault();
    pick(moves[e.key]);
  }
});

function showRegions(report: MemoryReport): void {
  const share = (bytes: number) => report.backed
    ? `${((bytes * 100) / report.backed).toFixed(1)}%`
    : '-';
  const row = (label: string, range: string | null, bytes: number) => {
    const name = el('th', null, label);
    name.scope = 'row';
    if (range) name.append(el('span', 'memory-range', range));
    const tr = el('tr');
    tr.append(name, el('td', null, formatBytes(bytes)), el('td', null, share(bytes)));
    return tr;
  };
  $('memory-regions').replaceChildren(
    ...report.regions.map((r) => row(REGION_LABELS[r.key], `${hex(r.start)}–${hex(r.end)}`, r.bytes)),
    row('Elsewhere', null, report.elsewhere),
  );
}

async function showHeap(report: MemoryReport): Promise<void> {
  const gpu: GpuReport = hasSession() ? await call('gpu_report') : {};
  const textures = gpu.backend ? gpu.textureCacheBytes ?? 0 : 0;
  const share = (report.wasmHeap * 100) / WASM_LIMIT;
  setNote('heap-debug-badge', `${share.toFixed(0)}% of 4 GiB`, false);
  $('heap-size').textContent = `${formatBytes(report.wasmHeap)} of 4 GiB`;
  $('heap-guest').textContent = formatBytes(report.allProcesses);
  $('heap-textures').textContent = gpu.backend ? formatBytes(textures) : 'not in use';
  $('heap-rest').textContent = formatBytes(Math.max(0, report.wasmHeap - report.allProcesses - textures));
  $('heap-device-textures').textContent = gpu.backend
    ? `Textures on the device, outside this heap: ${formatBytes(gpu.deviceTextureBytes ?? 0)}.`
    : '';
}

// The heap exists before any title does, so only the guest RAM waits for one.
export async function updateMemoryDebug(): Promise<void> {
  let report: MemoryReport;
  try {
    report = await call('memory');
  } catch (err) {
    const error = $('memory-debug-error');
    error.hidden = false;
    error.textContent = 'The worker could not report memory: ' + String(err);
    return;
  }
  $('memory-debug-error').hidden = true;
  await showHeap(report);
  const running = titleBooted();
  $('memory-debug-empty').hidden = running;
  $('memory-debug-data').hidden = !running;
  if (!running) {
    setNote('memory-debug-badge', '-', false);
    return;
  }
  setNote('memory-debug-badge', formatRate(rate()), false);
  $('memory-backed').textContent = formatBytes(report.backed);
  $('memory-cap').textContent = formatBytes(report.cap);
  $('memory-rate').textContent = formatRate(rate());
  $('memory-peak').textContent = formatBytes(Math.max(peak, report.backed));
  showRegions(report);
  drawChart();
}

// Rendered, so its panel, tab, group and section are all open.
function onScreen(node: HTMLElement): boolean {
  return node.offsetParent !== null;
}

setInterval(() => {
  if (onScreen(memorySection) || onScreen(heapSection)) void updateMemoryDebug();
}, 1000);
