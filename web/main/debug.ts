// The debug panel: instruction tracing, register dumps, and the translator and
// renderer readouts.

import type {
  DeviceTime, DeviceTimingMode, GpuReport, ThreadActivity, ThreadReading,
} from '../shared/protocol';
import {
  displayMetrics, resetDisplayMetrics, type DisplayMetrics, type DisplayTiming,
} from './display-metrics';
import { screenSize } from './display';
import { $, el } from './dom';
import './elements/info-tip';
import { fmtCount } from '../shared/format';
import { formatBytes } from './format';
import { log } from './log';
import { updateMemoryDebug } from './memory';
import { call, hasSession } from './rpc';
import { titleBooted } from './session';
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
  if (name === 'cpu') void updateThreadsDebug();
}

for (const button of debugGroups) {
  button.addEventListener('click', () => selectDebugGroup(button.dataset.debugGroup || 'cpu'));
}

// Tracing caps the run slice.
export function traceEnabled(): boolean {
  return traceCb.checked;
}

traceCb.addEventListener('change', () => {
  void call('setTrace', traceCb.checked);
  setNote('trace-badge', traceCb.checked ? 'on' : 'off', traceCb.checked);
  if (traceCb.checked) log('Tracing enabled - run slices are capped for readability. The trace shows at the Trace log level.');
});

const jitCb = $<HTMLInputElement>('jit-cb');

jitCb.addEventListener('change', () => {
  void call('setJit', jitCb.checked);
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
  const retired = s.retired ?? 0;
  const ran = s.retiredEmitted ?? 0;
  $('jit-retired-emitted').textContent =
    fmtCount(ran) + (retired ? ` (${((100 * ran) / retired).toFixed(1)}%)` : '');
  $('jit-dropped').textContent = fmtCount(s.dropped ?? 0);
}

const threadsSection = $<HTMLDetailsElement>('threads-debug-section');

function showThreadsStatus(text: string): void {
  const status = $('threads-status');
  status.textContent = text;
  status.hidden = false;
  $('threads-table').hidden = true;
  $('threads-idle').hidden = true;
  setNote('threads-badge', '');
}

function threadLabel(t: ThreadActivity): string {
  const who = t.name ?? t.entry ?? (t.index === 0 ? 'main' : null);
  return who ? `${t.index} ${who}` : String(t.index);
}

function threadRow(t: ThreadActivity, total: number, seconds: number | null): HTMLTableRowElement {
  const name = el('th', null, threadLabel(t));
  name.scope = 'row';
  const switches = `${fmtCount(t.switches)} ${t.switches === 1 ? 'switch' : 'switches'}`;
  name.append(
    el('span', 'thread-detail', `${t.state}, priority ${t.priority}, ${switches}`),
    el('span', 'thread-detail at', t.at),
  );
  const tr = el('tr');
  tr.append(
    name,
    el('td', null, total ? `${((100 * t.ran) / total).toFixed(1)}%` : '-'),
    el('td', null, seconds ? fmtCount(t.ran / seconds) : '-'),
  );
  return tr;
}

function showThreads(reading: ThreadReading): void {
  const byWork = [...reading.threads].sort((a, b) => b.ran - a.ran || a.index - b.index);
  const busy = byWork.filter((t) => t.ran > 0);
  // An exited thread that did nothing in the reading is history, not load.
  const idle = byWork.filter((t) => t.ran === 0 && t.state !== 'exited');
  if (!busy.length && !idle.length) {
    showThreadsStatus('Waiting for the first reading, a second after the title starts.');
    return;
  }
  const total = busy.reduce((sum, t) => sum + t.ran, 0);
  const status = $('threads-status');
  status.hidden = busy.length > 0;
  status.textContent = 'No thread ran in the last reading.';
  $('threads-table').hidden = !busy.length;
  $('threads-rows').replaceChildren(...busy.map((t) => threadRow(t, total, reading.seconds)));
  $('threads-idle').hidden = !idle.length;
  $('threads-idle-summary').textContent =
    `${idle.length} ${idle.length === 1 ? 'thread' : 'threads'} did not run`;
  $('threads-idle-list').replaceChildren(
    ...idle.map((t) => el('li', null, `${threadLabel(t)}: ${t.state}, priority ${t.priority}`)),
  );
  setNote('threads-badge', reading.seconds ? `${fmtCount(total / reading.seconds)} instructions/s` : '');
}

async function updateThreadsDebug(): Promise<void> {
  if (!titleBooted()) {
    showThreadsStatus('Start a title to see its threads.');
    return;
  }
  let reading: ThreadReading;
  try {
    reading = await call('threads');
  } catch (err) {
    showThreadsStatus(`Couldn't read the threads: ${String(err)}`);
    return;
  }
  // The title may have ended while the worker answered.
  if (titleBooted()) showThreads(reading);
  else showThreadsStatus('Start a title to see its threads.');
}

threadsSection.addEventListener('toggle', () => {
  if (threadsSection.open && onScreen(threadsSection)) void updateThreadsDebug();
});

const renderSection = $<HTMLDetailsElement>('render-debug-section');

$<HTMLInputElement>('render-force-software').addEventListener('change', (event) => {
  void call('forceSoftware', (event.target as HTMLInputElement).checked);
});

$<HTMLSelectElement>('render-device-time').addEventListener('change', (event) => {
  void call('setDeviceTiming', (event.target as HTMLSelectElement).value as DeviceTimingMode);
});

function showDeviceTime(time: DeviceTime | undefined): void {
  $('render-device-time-block').hidden = !time;
  if (!time) return;
  const rows = time.rows ?? [];
  $('render-device-time-table').hidden = !rows.length;
  let summary: string;
  if (time.mode === 'unsupported') {
    summary = 'This GPU has no timestamp queries, so its time cannot be measured.';
  } else if (!time.passes) {
    summary = 'No timed pass has come back from the GPU yet.';
  } else {
    const untimed = time.untimed ? `, ${fmtCount(time.untimed)} untimed` : '';
    summary = `${(time.ms ?? 0).toFixed(1)} ms on the GPU in ${fmtCount(time.passes)} passes `
      + `over ${time.seconds ?? 0} s${untimed}`;
  }
  $('render-device-time-summary').textContent = summary;
  $('render-device-time-rows').replaceChildren(...rows.map((row) => {
    const tr = el('tr');
    tr.append(
      el('td', null, row.ms.toFixed(1)),
      el('td', null, row.maxMs.toFixed(1)),
      el('td', null, fmtCount(row.passes)),
      el('td', null, row.what),
    );
    return tr;
  }));
}

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
    status: fallback ? 'WebGPU runs on a CPU fallback adapter.' : '',
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
  const line = (status + unlatched).trim();
  $('render-status').textContent = line;
  $('render-status').hidden = !line;
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
  $('render-held').textContent = `${g.held ?? 0} (${g.pending ?? 0} pending)`;
  $('render-syncs').textContent = `${fmtCount(g.syncAll ?? 0)} asked (${fmtCount(g.syncAliases ?? 0)} aliased, `
    + `${fmtCount(g.syncReads ?? 0)} read, ${fmtCount(g.syncWrites ?? 0)} guest-written), `
    + `${fmtCount(g.syncNoScanout ?? 0)} without a scan-out, ${fmtCount(g.syncOverBytes ?? 0)} over the cap`;
  const hits = g.textureHits ?? 0;
  const lookups = hits + (g.textureMisses ?? 0);
  $('render-texture-hits').textContent = lookups ? `${((hits * 100) / lookups).toFixed(1)}%` : '-';
  const binds = (g.bindHits ?? 0) + (g.bindMisses ?? 0) + (g.bindUncached ?? 0);
  $('render-bind-hits').textContent = binds
    ? `${(((g.bindHits ?? 0) * 100) / binds).toFixed(1)}% (${fmtCount(g.bindMisses ?? 0)} missed, ${fmtCount(g.bindUncached ?? 0)} uncached: `
    + `${fmtCount(g.bindUncachedHeld ?? 0)} held, ${fmtCount(g.bindUncachedShadow ?? 0)} shadow, `
    + `${fmtCount(g.bindUncachedTexture ?? 0)} texture, ${fmtCount(g.bindUncachedGlobal ?? 0)} global)`
    : '-';
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
  showDeviceTime(g.deviceTime);
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
  if (threadsSection.open && onScreen(threadsSection)) void updateThreadsDebug();
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
  const { width, height } = screenSize();
  $('display-frame-size').textContent = width ? `${width}×${height}` : '-';
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
  log(await call('regDump'));
});

const regIdx = $<HTMLInputElement>('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  $('reg-val').textContent = await call('getReg', Number.parseInt(regIdx.value, 10));
});
