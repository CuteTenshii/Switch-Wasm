/* The debug panel: instruction tracing, register dumps, and the trace buffer
   the emulator also uses for diagnostics. */

import type { LogClass } from './log';
import { $, el } from './dom';
import { consoleText, copyText, download, log, logBlock, stamp } from './log';
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
// behind them, and the whole of a fault. There is no stderr in the browser, so
// this is the only way they reach anyone. Drained as the run goes rather than
// only at the end.
export async function drainDiagnostics(): Promise<void> {
  logTrace(await drainTrace());
}

export async function drainTrace(): Promise<string> {
  const bytes = await call('drain_trace');
  if (bytes && bytes.length) return new TextDecoder().decode(bytes);
  return '';
}

/* The level a line of trace carries, as `switch_core::trace::Level` writes it:
   a control byte at the head of the line. A line with no marker continues the
   one before it -- which is what keeps a fault's register dump and instruction
   trail with the fault instead of reverting to grey. */
const MARKERS: Record<string, LogClass> = {
  '\u0001': 'err',
  '\u0002': 'warn',
  '\u0003': 'ok',
  '\u0004': 'dim',
};

/** Put a drained trace into the console, each line at the level it carries. */
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
  const errors = g.deviceErrorCount ?? 0;
  // Share of draws the device actually took. A frame can look fine and still
  // be almost entirely the rasterizer's.
  const share = drawn + fallbacks ? ((drawn * 100) / (drawn + fallbacks)).toFixed(1) : '0';
  log(
    `rendering: ${drawn} draws on the device, ${fallbacks} fell back (${share}% device), ` +
      `${errors} rejected, ` +
      `${g.pipelines ?? 0} pipelines, ${g.modules ?? 0} modules, ` +
      `${g.held ?? 0} surfaces held (${g.evicted ?? 0} evicted, ${g.pending ?? 0} pending)`,
    'dim',
  );
  if (g.gaveUp) log('rendering: the device was lost - the rasterizer has every frame.', 'err');
  else if (g.softwareFrame) {
    log('rendering: the software-frame latch has tripped - every frame from here is the rasterizer\'s.', 'err');
  }
  for (const why of g.reasons ?? []) log('  fell back: ' + why, 'dim');
  // Loud, and above the counters: a rejected draw is still counted as drawn,
  // so this is the only line that contradicts a clean-looking 100% device.
  if (errors) {
    const distinct = g.deviceErrors ?? [];
    const rest = errors - distinct.length;
    log(
      `rendering: the device rejected ${errors} thing(s) - the draws above were counted anyway.` +
        (rest > 0 ? ` ${distinct.length} distinct, ${rest} repeat(s).` : ''),
      'err',
    );
    for (const e of distinct) log('  device rejected: ' + e, 'err');
  }
  const r = g.read;
  if (r) {
    const mib = (v: number) => (v / (1024 * 1024)).toFixed(1);
    log(
      `  read from guest memory: ${mib(r.textures)} MiB textures, ${mib(r.vertex)} MiB vertices, ` +
        `${mib(r.constants)} MiB constants, ${mib(r.index)} MiB indices`,
      'dim',
    );
    const hits = g.textureHits ?? 0;
    const misses = g.textureMisses ?? 0;
    const rate = hits + misses ? ((hits * 100) / (hits + misses)).toFixed(1) : '0';
    log(`  texture cache: ${hits} hits, ${misses} misses (${rate}%)`, 'dim');
  }
  const t = g.times;
  if (t) {
    log(
      `  device time (${g.frames ?? 0} frames): translate ${t.translate}ms, upload ${t.upload}ms, ` +
        `modules ${t.modules}ms, pipeline ${t.pipeline}ms, encode ${t.encode}ms, ` +
        `flush ${t.flush}ms`,
      'dim',
    );
    // Split out, because `flush` being most of the frame says nothing about
    // what to do next and these three each name a different fix. Per frame as
    // well as total: the totals grow with the run and only the per-frame cost
    // can be compared against the frame budget.
    if (t.flushLand !== undefined) {
      const frames = Math.max(g.frames ?? 0, 1);
      const per = (v: number) => (v / frames).toFixed(1);
      log(
        `    flush: ask ${t.flushAsk}ms (${per(t.flushAsk ?? 0)}/frame), ` +
          `wait ${t.flushWait}ms (${per(t.flushWait ?? 0)}/frame), ` +
          `land ${t.flushLand}ms (${per(t.flushLand)}/frame)`,
        'dim',
      );
    }
  }
});

$('btn-dumptrace').addEventListener('click', async () => {
  const t = await drainTrace();
  openPanel('console');
  if (t) logTrace(t);
  else log('(no trace)', 'dim');
});

$('btn-dumpregs').addEventListener('click', async () => {
  const s = await call('dump_regs');
  openPanel('console');
  if (s) logBlock(s, 'dim');
});

$('btn-threads').addEventListener('click', async () => {
  const dump = await call('thread_dump');
  openPanel('console');
  if (dump) logBlock(dump, 'dim');
  else log('(no threads)', 'dim');
  const frames = await call('backtrace', 16);
  if (frames.length) {
    log('  backtrace: ' + frames.map((pc) => '0x' + pc.toString(16)).join(' <- '), 'dim');
  }
});

// Both of these change what the guest is doing, so they say what they did:
// a lever that reports nothing is indistinguishable from one that found
// nothing to do.
$('btn-wake').addEventListener('click', async () => {
  const woken = await call('wake_blocked');
  openPanel('console');
  log(
    woken
      ? `Woke ${woken} blocked thread(s). A guest re-checks its predicate, so a wake it did not `
        + 'need degrades to a spin rather than to a wrong answer.'
      : 'No thread was blocked - this process is idle for some other reason.',
    'dim',
  );
});

$('btn-start-threads').addEventListener('click', async () => {
  const started = await call('start_created_threads');
  openPanel('console');
  log(
    started
      ? `Started ${started} thread(s) the guest created and never ran.`
      : 'Every thread the guest created has been started.',
    'dim',
  );
});

$('btn-gaps').addEventListener('click', async () => {
  const gaps = await call('ipc_gaps');
  openPanel('console');
  const name = (g: { iface: string; cmd: number | null }) =>
    `${g.iface} cmd=${g.cmd === null ? '-' : g.cmd}`;
  if (!gaps.unimplemented.length && !gaps.stubbed.length) {
    log('Nothing this title has asked for has been refused or stubbed.', 'dim');
    return;
  }
  if (gaps.unimplemented.length) {
    log(`Refused - no implementation behind them (${gaps.unimplemented.length}):`, 'warn');
    for (const g of gaps.unimplemented) log('  ' + name(g), 'dim');
  }
  if (gaps.stubbed.length) {
    log(`Answered with nothing behind the answer (${gaps.stubbed.length}):`, 'warn');
    for (const g of gaps.stubbed) log('  ' + name(g), 'dim');
  }
});

const regIdx = $<HTMLInputElement>('reg-idx');
$('btn-readreg').addEventListener('click', async () => {
  $('reg-val').textContent = await call('get_reg', parseInt(regIdx.value, 10));
});

/* The diagnostic channels.

   These are the emulator's `TRACE_*` switches. They were environment
   variables, which is a thing a browser does not have, so the most detailed
   account the emulator can give of itself was reachable only from the command
   line -- on a project whose target is the browser. The list is the core's
   own, fetched rather than duplicated here, so a channel added to the core
   appears here with nothing to change. */
const channelsEl = $('trace-channels');

/** Fill in the channel switches from the core's own list.
 *
 *  Called by the composition root once the worker is up, not at import time:
 *  the list comes from the module, and asking for it while the worker is
 *  still starting is a rejected promise and an empty panel. */
export async function initTraceChannels(): Promise<void> {
  const channels = await call('trace_channels');
  channelsEl.textContent = '';
  for (const channel of channels) {
    const label = el('label', 'check');
    const box = el('input');
    box.type = 'checkbox';
    box.checked = channel.on;
    box.dataset.bit = String(channel.bit);
    box.addEventListener('change', applyChannels);
    const text = el('span');
    // Without the `TRACE_` prefix on the page and with it in the title: the
    // prefix is the same on all nineteen and carries nothing, but it is also
    // the exact spelling someone needs to set the same channel from a shell.
    text.textContent = channel.name.replace(/^TRACE_/, '').toLowerCase();
    label.title = channel.name;
    label.append(box, text);
    channelsEl.appendChild(label);
  }
}

function applyChannels(): void {
  let mask = 0;
  let on = 0;
  for (const box of channelsEl.querySelectorAll<HTMLInputElement>('input[type=checkbox]')) {
    if (!box.checked) continue;
    mask |= Number(box.dataset.bit);
    on += 1;
  }
  call('set_trace_mask', mask);
  setNote('channels-badge', on ? `${on} on` : 'off', on > 0);
}


/* The crash report.

   Every field in it already existed and every one had to be asked for
   separately, through a different button, before the state that made it worth
   reading was gone. This is the bundling: one button, one file, everything an
   issue needs to be readable by somebody who was not there. */
async function crashReport(): Promise<string> {
  const report = await call('crash_report');
  // The browser is half of any report about a browser emulator, and only the
  // page can say what it is.
  return JSON.stringify(
    {
      ...report,
      browser: {
        userAgent: navigator.userAgent,
        platform: (navigator as unknown as { platform?: string }).platform ?? '',
        hardwareConcurrency: navigator.hardwareConcurrency,
        webgpu: 'gpu' in navigator,
        deviceMemory: (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? null,
      },
      // The page's own log too: it holds what the page said as well as what
      // the core did -- the worker errors, the load failures, the renderer
      // notes -- and none of that is in the core's trace.
      log: consoleText().split('\n'),
    },
    null,
    2,
  );
}

$('btn-crash-report').addEventListener('click', async () => {
  const text = await crashReport();
  download(`switch-wasm-report-${stamp()}.json`, text, 'application/json');
  openPanel('console');
  log('Crash report saved. Attach it to the issue - it names the build, the title, '
    + 'the renderer, the registers and the run-up to the fault.', 'ok');
});

$('btn-copy-report').addEventListener('click', async () => {
  const ok = await copyText(await crashReport());
  openPanel('console');
  log(ok ? 'Crash report copied to the clipboard.' : 'Could not copy the crash report.',
    ok ? 'ok' : 'err');
});
