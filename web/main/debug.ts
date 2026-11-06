// The debug panel: instruction tracing, register dumps, and the translator and
// renderer readouts.

import type { GpuReport } from '../shared/protocol';
import {
  displayMetrics, resetDisplayMetrics, type DisplayMetrics, type DisplayTiming,
} from './display-metrics';
import { $, el } from './dom';
import './elements/info-tip';
import { fmtCount } from '../shared/format';
import { formatBytes } from './format';
import { log } from './log';
import { updateMemoryDebug } from './memory';
import { call, hasSession } from './rpc';
import { openPanel, setNote } from './shell';

const traceCb = $<HTMLInputElement>('trace-cb');

const debugGroups = document.querySelectorAll<HTMLButtonElement>('[data-debug-group]');
const debugSections = document.querySelectorAll<HTMLElement>('[data-debug-section]');

function selectDebugGroup(name: string): void {
  for (const button of debugGroups) {
    const selected = button.dataset.debugGroup === name;
    button.classList.toggle('is-active', selected);
    button.setAttribute('aria-pressed', String(selected));
  }
  for (const section of debugSections) {
    section.hidden = section.dataset.debugSection !== name;
  }
  if (name === 'graphics') {
    updateDisplayDebug();
    if (hasSession()) void updateRenderDebug();
  }
  if (name === 'memory') void updateMemoryDebug();
}

for (const button of debugGroups) {
  button.addEventListener('click', () => selectDebugGroup(button.dataset.debugGroup || 'cpu'));
}

// Tracing caps the run slice.
export function traceEnabled(): boolean {
  return traceCb.checked;
}

traceCb.addEventListener('change', () => {
  void call('setTrace', traceCb.checked ? 1 : 0);
  setNote('trace-badge', traceCb.checked ? 'on' : 'off', traceCb.checked);
  if (traceCb.checked) log('Tracing enabled - run slices are capped for readability. The trace shows at the Trace log level.');
});

const jitCb = $<HTMLInputElement>('jit-cb');

jitCb.addEventListener('change', () => {
  void call('setJit', jitCb.checked ? 1 : 0);
  setNote('jit-badge', jitCb.checked ? 'on' : 'off', jitCb.checked);
  if (!jitCb.checked) log('Block translation disabled - running the plain interpreter.');
});

const jitSection = $('jit-debug-section');

async function updateJitDebug(): Promise<void> {
  const s = await call('jitStats');
  const ofEntries = (part: number) =>
    fmtCount(part) + (s.executed ? ` (${((100 * part) / s.executed).toFixed(1)}%)` : '');
  $('jit-blocks').textContent = fmtCount(s.blocks);
  $('jit-translated').textContent = fmtCount(s.translated);
  $('jit-invalidated').textContent = fmtCount(s.invalidated);
  $('jit-emitted').textContent = fmtCount(s.emitted ?? 0);
  $('jit-executed').textContent = fmtCount(s.executed);
  $('jit-reuse').textContent = s.translated ? fmtCount(s.executed / s.translated) : '-';
  $('jit-entered-emitted').textContent = ofEntries(s.enteredEmitted ?? 0);
  $('jit-chained').textContent = ofEntries(s.chained ?? 0);
}

const renderSection = $<HTMLDetailsElement>('render-debug-section');

const RENDER_PHASES = ['translate', 'upload', 'modules', 'bind', 'pipeline', 'encode', 'flush'] as const;

// Who draws, in a word for the badge and a sentence for the section.
interface AdapterInfo {
  vendor: string;
  architecture: string;
  description: string;
  isFallbackAdapter: boolean;
}

// The page asks for the default adapter, as the worker does; wgpu reports only the description.
let adapterInfo: Promise<AdapterInfo | null> | undefined;
function queryAdapter(): Promise<AdapterInfo | null> {
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<{ info: AdapterInfo } | null> } }).gpu;
  adapterInfo ??= gpu
    ? gpu.requestAdapter().then((adapter) => adapter?.info ?? null, () => null)
    : Promise.resolve(null);
  return adapterInfo;
}

function adapterLabel(info: AdapterInfo | null): string {
  if (!info) return 'unknown';
  const name = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ');
  const fallback = info.isFallbackAdapter ? ' (CPU fallback)' : '';
  return (name || 'not named by the browser') + fallback;
}

interface RenderState {
  badge: string;
  status: string;
  on: boolean;
  detail: string;
}

function renderState(g: GpuReport, fallback: boolean): RenderState {
  const rasterizer = 'The software rasterizer draws every frame.';
  if (!g.backend) return { badge: 'software', status: 'No WebGPU device.', on: false, detail: rasterizer };
  if (g.gaveUp) {
    const why = g.lostBecause ? ` (${g.lostBecause})` : '';
    return { badge: 'lost', status: `The device was lost${why}.`, on: false, detail: rasterizer };
  }
  if (g.softwareFrame) {
    return {
      badge: 'software',
      status: 'The software rasterizer is drawing.',
      on: false,
      detail: 'A draw the device could not take fell back to the rasterizer, which draws every frame '
        + 'until the device can take whole frames again.',
    };
  }
  return {
    badge: 'WebGPU',
    status: fallback ? 'WebGPU draws, on a CPU fallback adapter.' : 'WebGPU draws.',
    on: true,
    detail: 'A draw the device cannot take runs on the software rasterizer instead, and once enough '
      + 'have, the rasterizer keeps every frame after, however well the device is working.',
  };
}

function showList(block: string, list: string, items: string[]): void {
  $(block).hidden = !items.length;
  $(list).replaceChildren(...items.map((item) => el('li', null, item)));
}

async function updateRenderDebug(): Promise<void> {
  const g = await call('gpuReport');
  const info = g.backend ? await queryAdapter() : null;
  const { badge, status, on, detail } = renderState(g, info?.isFallbackAdapter ?? false);
  setNote('render-debug-badge', badge, on);
  const unlatched = g.unlatched ? ` It has handed frames back to the device ${g.unlatched} time(s).` : '';
  $('render-status').textContent = status + unlatched;
  $('render-tip').setAttribute('text', detail);
  $('render-debug-data').hidden = !g.backend;
  showList('render-reasons-block', 'render-reasons', g.reasons ?? []);
  const errors = g.deviceErrors ?? [];
  const repeats = (g.deviceErrorCount ?? 0) - errors.length;
  showList('render-errors-block', 'render-errors',
    repeats > 0 ? [...errors, `and ${repeats} repeat(s)`] : errors);
  if (!g.backend) return;

  const drawn = g.drawn ?? 0;
  const fallbacks = g.fallbacks ?? 0;
  const share = drawn + fallbacks ? ((drawn * 100) / (drawn + fallbacks)).toFixed(1) : '0';
  $('render-adapter').textContent = adapterLabel(info);
  $('render-drawn').textContent = `${fmtCount(drawn)} (${share}%)`;
  $('render-fallbacks').textContent = fmtCount(fallbacks);
  $('render-rejected').textContent = fmtCount(g.deviceErrorCount ?? 0);
  $('render-pipelines').textContent = fmtCount(g.pipelines ?? 0);
  $('render-modules').textContent = fmtCount(g.modules ?? 0);
  $('render-held').textContent = `${g.held ?? 0} (${g.evicted ?? 0} evicted, ${g.pending ?? 0} pending)`;
  const hits = g.textureHits ?? 0;
  const lookups = hits + (g.textureMisses ?? 0);
  $('render-texture-hits').textContent = lookups ? `${((hits * 100) / lookups).toFixed(1)}%` : '-';
  const r = g.read;
  for (const kind of ['textures', 'vertex', 'constants', 'index'] as const) {
    $(`render-read-${kind}`).textContent = formatBytes(r?.[kind] ?? 0);
  }
  const frames = Math.max(g.frames ?? 0, 1);
  for (const phase of RENDER_PHASES) {
    const total = g.times?.[phase];
    $(`render-${phase}-frame`).textContent = total === undefined ? '-' : (total / frames).toFixed(1);
    $(`render-${phase}-total`).textContent = total === undefined ? '-' : String(total);
  }
}

// Rendered, so its panel, tab, group and section are all open.
function onScreen(node: HTMLElement): boolean {
  return node.offsetParent !== null;
}

renderSection.addEventListener('toggle', () => {
  if (renderSection.open && onScreen(renderSection) && hasSession()) void updateRenderDebug();
});
// Live sections ask the worker only while someone can see them.
setInterval(() => {
  if (!hasSession()) return;
  if (renderSection.open && onScreen(renderSection)) void updateRenderDebug();
  if (onScreen(jitSection)) void updateJitDebug();
}, 1000);

const displayDebugSection = $<HTMLDetailsElement>('display-debug-section');

const DISPLAY_TIMINGS: [keyof DisplayMetrics, string][] = [
  ['runSlice', 'slice'],
  ['canvasWrite', 'canvas'],
];

function showTiming(name: string, sample: DisplayTiming): void {
  const value = (n: number) => sample.count ? n.toFixed(2) : '-';
  $(`display-${name}-last`).textContent = value(sample.last);
  $(`display-${name}-mean`).textContent = value(sample.mean);
  $(`display-${name}-max`).textContent = value(sample.max);
}

function updateDisplayDebug(): void {
  const metrics = displayMetrics();
  const sampled = metrics.runSlice.count > 0;
  $('display-debug-data').hidden = !sampled;
  $<HTMLButtonElement>('btn-reset-display-debug').disabled = !sampled;
  if (!sampled) return;
  $('display-guest-frames').textContent = String(metrics.guestFrames);
  $('display-canvas-updates').textContent = String(metrics.canvasUpdates);
  $('display-skipped-frames').textContent = String(metrics.skippedFrames);
  for (const [key, name] of DISPLAY_TIMINGS) {
    showTiming(name, metrics[key] as DisplayTiming);
  }
}

displayDebugSection.addEventListener('toggle', () => {
  if (displayDebugSection.open) updateDisplayDebug();
});
$('btn-reset-display-debug').addEventListener('click', () => {
  resetDisplayMetrics();
  updateDisplayDebug();
});
setInterval(() => {
  if (displayDebugSection.open && !displayDebugSection.hidden) updateDisplayDebug();
}, 500);
updateDisplayDebug();

$('btn-dumptrace').addEventListener('click', async () => {
  openPanel('console');
  if (!await call('flushLog')) log('(no trace)');
});

$('btn-dumpregs').addEventListener('click', async () => {
  openPanel('console');
  await call('logRegs');
});

const regIdx = $<HTMLInputElement>('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  $('reg-val').textContent = await call('getReg', parseInt(regIdx.value, 10));
});
