/* The debug panel: instruction tracing, register dumps, and the trace buffer
   the emulator also uses for diagnostics. */

import { $ } from './dom';
import { log } from './log';
import { call } from './rpc';
import { openPanel, setNote } from './shell';

const traceCb = $<HTMLInputElement>('trace-cb');

/** Tracing caps the run slice, so the loop has to ask. */
export function traceEnabled(): boolean {
  return traceCb.checked;
}

traceCb.addEventListener('change', () => {
  call('set_trace', traceCb.checked ? 1 : 0);
  setNote('trace-badge', traceCb.checked ? 'on' : 'off', traceCb.checked);
  if (traceCb.checked) log('Tracing enabled - run slices are capped for readability.', 'dim');
});

// The trace buffer carries more than the per-instruction disassembly: the
// emulator records diagnostics there whether or not tracing is enabled -
// services and applet commands a guest asked for that have no implementation
// behind them. There is no stderr in the browser, so this is the only way they
// reach anyone. Drained as the run goes rather than only at the end.
export async function drainDiagnostics(): Promise<void> {
  const t = await drainTrace();
  if (t) log(t.replace(/\n$/, ''), 'dim');
}

export async function drainTrace(): Promise<string> {
  const bytes = await call('drain_trace');
  if (bytes && bytes.length) return new TextDecoder().decode(bytes);
  return '';
}

const jitCb = $<HTMLInputElement>('jit-cb');

jitCb.addEventListener('change', () => {
  call('set_jit', jitCb.checked ? 1 : 0);
  setNote('jit-badge', jitCb.checked ? 'on' : 'off', jitCb.checked);
  if (!jitCb.checked) log('Block translation disabled - running the plain interpreter.', 'dim');
});

$('btn-jitstats').addEventListener('click', async () => {
  const s = await call('jit_stats');
  openPanel('console');
  // Blocks entered per translation is the number that says whether
  // translating was worth it: one means every block was thrown away unused.
  const reuse = s.translated ? (s.executed / s.translated).toFixed(1) : '0';
  log(
    `translation: ${s.enabled ? 'on' : 'off'}, ${s.blocks} blocks cached, ` +
      `${s.translated} translated, ${s.executed} entered (${reuse}x each), ` +
      `${s.invalidated} invalidated`,
    'dim',
  );
});

$('btn-gpustats').addEventListener('click', async () => {
  const g = await call('gpu_report');
  openPanel('console');
  if (!g.backend) {
    log('rendering: the software rasterizer has the frame - no device is installed.', 'dim');
    return;
  }
  const drawn = g.drawn ?? 0;
  const fallbacks = g.fallbacks ?? 0;
  // Share of draws the device actually took. A frame can look fine and still
  // be almost entirely the rasterizer's.
  const share = drawn + fallbacks ? ((drawn * 100) / (drawn + fallbacks)).toFixed(1) : '0';
  log(
    `rendering: ${drawn} draws on the device, ${fallbacks} fell back (${share}% device), ` +
      `${g.pipelines ?? 0} pipelines, ${g.modules ?? 0} modules, ` +
      `${g.held ?? 0} surfaces held (${g.evicted ?? 0} evicted, ${g.pending ?? 0} pending)`,
    'dim',
  );
  if (g.gaveUp) log('rendering: the device was lost - the rasterizer has every frame.', 'err');
  else if (g.softwareFrame) {
    log('rendering: the software-frame latch has tripped - every frame from here is the rasterizer\'s.', 'err');
  }
  for (const why of g.reasons ?? []) log('  fell back: ' + why, 'dim');
  const t = g.times;
  if (t) {
    log(
      `  device time: translate ${t.translate}ms, upload ${t.upload}ms, ` +
        `modules ${t.modules}ms, pipeline ${t.pipeline}ms, encode ${t.encode}ms, ` +
        `flush ${t.flush}ms`,
      'dim',
    );
  }
});

$('btn-dumptrace').addEventListener('click', async () => {
  const t = await drainTrace();
  openPanel('console');
  log(t ? t.replace(/\n$/, '') : '(no trace)', 'dim');
});

$('btn-dumpregs').addEventListener('click', async () => {
  const s = await call('dump_regs');
  openPanel('console');
  if (s) log(s.replace(/\n$/, ''), 'dim');
});

const regIdx = $<HTMLInputElement>('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  $('reg-val').textContent = await call('get_reg', parseInt(regIdx.value, 10));
});
