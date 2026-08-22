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
