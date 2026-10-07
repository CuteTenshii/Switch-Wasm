// The debug panel: instruction tracing, register dumps, and the trace buffer
// the emulator also uses for diagnostics.

import type { LogClass } from './log';
import {
  displayMetrics, resetDisplayMetrics, type DisplayMetrics, type DisplayTiming,
} from './display-metrics';
import { $ } from './dom';
import { formatBytes } from './format';
import { consoleText, copyText, download, log, logBlock, stamp } from './log';
import { call } from './rpc';
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
  if (name === 'graphics') updateDisplayDebug();
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
const MARKERS: Record<string, LogClass> = {
  '\u0001': 'err',
  '\u0002': 'warn',
  '\u0003': 'ok',
  '\u0004': 'dim',
};

export function logTrace(text: string): void {
  if (!text) return;
  let cls: LogClass = 'dim';
  for (const line of text.replace(/\n$/, '').split('\n')) {
    const marked = MARKERS[line[0]];
    if (marked) cls = marked;
    log(marked ? line.slice(1) : line, cls);
  }
}

const jitCb = $<HTMLInputElement>('jit-cb');

jitCb.addEventListener('change', () => {
  void call('set_jit', jitCb.checked ? 1 : 0);
  setNote('jit-badge', jitCb.checked ? 'on' : 'off', jitCb.checked);
  if (!jitCb.checked) log('Block translation disabled - running the plain interpreter.');
});

$('btn-jitstats').addEventListener('click', async () => {
  const s = await call('jit_stats');
  openPanel('console');
  const reuse = s.translated ? (s.executed / s.translated).toFixed(1) : '0';
  log(
    `translation: ${s.enabled ? 'on' : 'off'}, ${s.blocks} blocks cached, `
    + `${s.translated} translated, ${s.executed} entered (${reuse}x each), `
    + `${s.invalidated} invalidated`,
  );
  const emitted = s.emitted ?? 0;
  const entered = s.enteredEmitted ?? 0;
  const share = s.executed ? ((100 * entered) / s.executed).toFixed(1) : '0';
  const chained = s.chained ?? 0;
  log(
    emitted
      ? `compiled: ${emitted} blocks, entered ${entered} times (${share}% of entries),`
      + ` ${chained} of them by a jump from another compiled block`
      : 'compiled: nothing - every block was interpreted',
  );
});

$('btn-gpustats').addEventListener('click', async () => {
  const g = await call('gpu_report');
  openPanel('console');
  if (!g.backend) {
    log('rendering: the software rasterizer has the frame - no device is installed.');
    return;
  }
  const drawn = g.drawn ?? 0;
  const fallbacks = g.fallbacks ?? 0;
  const errors = g.deviceErrorCount ?? 0;
  const share = drawn + fallbacks ? ((drawn * 100) / (drawn + fallbacks)).toFixed(1) : '0';
  log(
    `rendering: ${drawn} draws on the device, ${fallbacks} fell back (${share}% device), `
    + `${errors} rejected, `
    + `${g.pipelines ?? 0} pipelines, ${g.modules ?? 0} modules, `
    + `${g.held ?? 0} surfaces held (${g.evicted ?? 0} evicted, ${g.pending ?? 0} pending)`,
  );
  if (g.gaveUp) {
    const why = g.lostBecause ? `: ${g.lostBecause}` : '';
    log(`rendering: the device was lost - the rasterizer has every frame${why}.`, 'err');
  } else if (g.softwareFrame) {
    log('rendering: the software-frame latch has tripped - the rasterizer has the frames until the device could draw them all.', 'err');
  }
  if (g.unlatched) log(`rendering: the software-frame latch has let go ${g.unlatched} time(s).`);
  for (const why of g.reasons ?? []) log('  fell back: ' + why);
  // A rejected draw is still counted as drawn; this line contradicts a clean 100%.
  if (errors) {
    const distinct = g.deviceErrors ?? [];
    const rest = errors - distinct.length;
    log(
      `rendering: the device rejected ${errors} thing(s) - the draws above were counted anyway.`
      + (rest > 0 ? ` ${distinct.length} distinct, ${rest} repeat(s).` : ''),
      'err',
    );
    for (const e of distinct) log('  device rejected: ' + e, 'err');
  }
  const r = g.read;
  if (r) {
    const mib = (v: number) => (v / (1024 * 1024)).toFixed(1);
    log(
      `  read from guest memory: ${mib(r.textures)} MiB textures, ${mib(r.vertex)} MiB vertices, `
      + `${mib(r.constants)} MiB constants, ${mib(r.index)} MiB indices`,
    );
    const hits = g.textureHits ?? 0;
    const misses = g.textureMisses ?? 0;
    const rate = hits + misses ? ((hits * 100) / (hits + misses)).toFixed(1) : '0';
    log(`  texture cache: ${hits} hits, ${misses} misses (${rate}%)`);
  }
  const t = g.times;
  if (t) {
    log(
      `  device time (${g.frames ?? 0} frames): translate ${t.translate}ms, upload ${t.upload}ms, `
      + `modules ${t.modules}ms, pipeline ${t.pipeline}ms, encode ${t.encode}ms, `
      + `flush ${t.flush}ms`,
    );
    if (t.flushLand !== undefined) {
      const frames = Math.max(g.frames ?? 0, 1);
      const per = (v: number) => (v / frames).toFixed(1);
      log(
        `    flush: ask ${t.flushAsk}ms (${per(t.flushAsk ?? 0)}/frame), `
        + `wait ${t.flushWait}ms (${per(t.flushWait ?? 0)}/frame), `
        + `land ${t.flushLand}ms (${per(t.flushLand)}/frame)`,
      );
    }
  }
});

const displayDebugSection = $<HTMLDetailsElement>('display-debug-section');

const DISPLAY_TIMINGS: [keyof DisplayMetrics, string][] = [
  ['runSlice', 'slice'],
  ['paintWait', 'paint'],
  ['frameCounter', 'counter'],
  ['snapshot', 'snapshot'],
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
  const sampled = metrics.runSlice.count > 0 || metrics.frameCounter.count > 0;
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
  $('display-merged-requests').textContent = String(metrics.mergedRequests);
  $('display-snapshot-bytes').textContent = formatBytes(metrics.snapshotBytes);
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
  if (s) logBlock(s);
});

$('btn-threads').addEventListener('click', async () => {
  const dump = await call('thread_dump');
  openPanel('console');
  if (dump) logBlock(dump);
  else log('(no threads)');
  const frames = await call('backtrace', 16);
  if (frames.length) {
    log('  backtrace: ' + frames.map((pc) => '0x' + pc.toString(16)).join(' <- '));
  }
});

$('btn-wake').addEventListener('click', async () => {
  const woken = await call('wake_blocked');
  openPanel('console');
  log(
    woken
      ? `Woke ${woken} blocked thread(s). A guest re-checks its predicate, so a wake it did not `
      + 'need degrades to a spin rather than to a wrong answer.'
      : 'No thread was blocked - this process is idle for some other reason.',
  );
});

$('btn-start-threads').addEventListener('click', async () => {
  const started = await call('start_created_threads');
  openPanel('console');
  log(
    started
      ? `Started ${started} thread(s) the guest created and never ran.`
      : 'Every thread the guest created has been started.',
  );
});

$('btn-gaps').addEventListener('click', async () => {
  const gaps = await call('ipc_gaps');
  openPanel('console');
  const name = (g: { iface: string; cmd: number | null }) =>
    `${g.iface} cmd=${g.cmd === null ? '-' : g.cmd}`;
  if (!gaps.unimplemented.length && !gaps.stubbed.length) {
    log('Nothing this title has asked for has been refused or stubbed.');
    return;
  }
  if (gaps.unimplemented.length) {
    log(`Refused - no implementation behind them (${gaps.unimplemented.length}):`, 'warn');
    for (const g of gaps.unimplemented) log('  ' + name(g));
  }
  if (gaps.stubbed.length) {
    log(`Answered with nothing behind the answer (${gaps.stubbed.length}):`, 'warn');
    for (const g of gaps.stubbed) log('  ' + name(g));
  }
});

const regIdx = $<HTMLInputElement>('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  $('reg-val').textContent = await call('get_reg', parseInt(regIdx.value, 10));
});

// The crash report: everything an issue needs, in one file.
async function crashReport(): Promise<string> {
  const report = await call('crash_report');
  return JSON.stringify(
    {
      ...report,
      browser: {
        userAgent: navigator.userAgent,
        platform: (navigator as unknown as { platform?: string }).platform ?? '',
        hardwareConcurrency: navigator.hardwareConcurrency,
        webgpu: 'gpu' in navigator,
        deviceMemory: (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? null,
        display: displayMetrics(),
      },
      // The page's log holds worker errors, load failures and renderer notes.
      log: consoleText().split('\n'),
    },
    null,
    2,
  );
}

export async function saveCrashReport(): Promise<void> {
  const text = await crashReport();
  download(`switch-wasm-report-${stamp()}.json`, text, 'application/json');
  openPanel('console');
  log('Crash report saved. Attach it to the issue - it names the build, the title, '
    + 'the renderer, the registers and the run-up to the fault.', 'ok');
}

$('btn-crash-report').addEventListener('click', saveCrashReport);

$('btn-copy-report').addEventListener('click', async () => {
  const ok = await copyText(await crashReport());
  openPanel('console');
  log(ok ? 'Crash report copied to the clipboard.' : 'Could not copy the crash report.',
    ok ? 'ok' : 'err');
});
