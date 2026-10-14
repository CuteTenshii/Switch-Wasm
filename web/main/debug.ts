// The debug panel: instruction tracing, register dumps, and the trace buffer
// the emulator also uses for diagnostics.

import type { GpuReport } from '../shared/protocol';
import type { LogLevel } from './log';
import {
  displayMetrics, resetDisplayMetrics, type DisplayMetrics, type DisplayTiming,
} from './display-metrics';
import { $, el } from './dom';
import { fmtCount } from '../shared/format';
import { formatBytes } from './format';
import { log } from './log';
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
}

for (const button of debugGroups) {
  button.addEventListener('click', () => selectDebugGroup(button.dataset.debugGroup || 'cpu'));
}

// Tracing caps the run slice.
export function traceEnabled(): boolean {
  return traceCb.checked;
}

traceCb.addEventListener('change', () => {
  void call('set_trace', traceCb.checked ? 1 : 0);
  setNote('trace-badge', traceCb.checked ? 'on' : 'off', traceCb.checked);
  if (traceCb.checked) log('Tracing enabled - run slices are capped for readability. The trace shows at the Debug log level.');
});

// The trace buffer also carries diagnostics (unimplemented services, faults),
// so it is drained as the run goes.
export async function drainDiagnostics(): Promise<void> {
  logTrace(await drainTrace());
}

export async function drainTrace(): Promise<string> {
  const bytes = await call('drain_trace');
  if (bytes && bytes.length) return new TextDecoder().decode(bytes);
  return '';
}

// Trace line levels from `switch_core::trace::Level`; an unmarked line continues the previous one.
const MARKERS: Record<string, LogLevel> = {
  '\u0001': 'error',
  '\u0002': 'warn',
  '\u0003': 'info',
  '\u0004': 'debug',
};

export function logTrace(text: string): void {
  if (!text) return;
  let level: LogLevel = 'debug';
  for (const line of text.replace(/\n$/, '').split('\n')) {
    const marked = MARKERS[line[0]];
    if (marked) level = marked;
    log(marked ? line.slice(1) : line, level);
  }
}

const jitCb = $<HTMLInputElement>('jit-cb');

jitCb.addEventListener('change', () => {
  void call('set_jit', jitCb.checked ? 1 : 0);
  setNote('jit-badge', jitCb.checked ? 'on' : 'off', jitCb.checked);
  if (!jitCb.checked) log('Block translation disabled - running the plain interpreter.');
});

const jitSection = $('jit-debug-section');

async function updateJitDebug(): Promise<void> {
  const s = await call('jit_stats');
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

const RENDER_PHASES = ['translate', 'upload', 'modules', 'pipeline', 'encode', 'flush'] as const;

// Who draws, in a word for the badge and a sentence for the section.
function renderState(g: GpuReport): [string, string, boolean] {
  if (!g.backend) return ['software', 'No device: the software rasterizer draws every frame.', false];
  if (g.gaveUp) {
    const why = g.lostBecause ? ` (${g.lostBecause})` : '';
    return ['lost', `The device was lost${why}: the rasterizer draws every frame.`, false];
  }
  if (g.softwareFrame) {
    return ['software', 'A draw fell back, so the rasterizer draws until the device can take whole '
      + 'frames again.', false];
  }
  return ['device', 'The device draws.', true];
}

function showList(block: string, list: string, items: string[]): void {
  $(block).hidden = !items.length;
  $(list).replaceChildren(...items.map((item) => el('li', null, item)));
}

async function updateRenderDebug(): Promise<void> {
  const g = await call('gpu_report');
  const [badge, status, on] = renderState(g);
  setNote('render-debug-badge', badge, on);
  const unlatched = g.unlatched ? ` It has handed frames back to the device ${g.unlatched} time(s).` : '';
  $('render-status').textContent = status + unlatched;
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
  $('render-read').textContent = formatBytes(r ? r.textures + r.vertex + r.constants + r.index : 0);
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
  $('display-debug-empty').hidden = sampled;
  $('display-debug-data').hidden = !sampled;
  setNote(
    'display-debug-badge',
    sampled ? `${metrics.canvasUpdates}/${metrics.guestFrames} shown` : 'no samples',
    sampled,
  );
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
  const t = await drainTrace();
  openPanel('console');
  if (t) logTrace(t);
  else log('(no trace)');
});

$('btn-dumpregs').addEventListener('click', async () => {
  const s = await call('dump_regs');
  openPanel('console');
  if (s) log(s);
});

const regIdx = $<HTMLInputElement>('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  $('reg-val').textContent = await call('get_reg', parseInt(regIdx.value, 10));
});
